"""Phase 3: table extraction with a real-table gate.

pdfplumber's `extract_tables()` is not used directly. `find_tables()` answers
"where are ruled cells?", which is not the same question as "is this a table?" --
a page frame crossed by any rule comes back as a grid. So every candidate goes
through `trimitdown_pdf.is_real_table` before anything is written.

Two corrections to the obvious implementation, both verified against the
library's own source:

1.  `Table.extract(**kwargs)` forwards its kwargs straight to
    `pdfplumber.utils.extract_text`, which reads only `x_tolerance`,
    `y_tolerance`, `x_tolerance_ratio`, `y_tolerance_ratio` and `layout`.
    Passing `strategy="text"` therefore does *nothing at all* -- it is silently
    discarded, and the call quietly reverts to the 3pt default, which re-glues
    words inside cells. The correct call is `extract(**TEXT_SETTINGS)`.

2.  pdfplumber reports one rectangle per cell, and those rectangles overlap
    wherever a ruling line stops short: a spanning cell keeps its full extent
    while the cells it covers are still reported separately. Reading each
    rectangle independently emits every char in the overlap once per rectangle,
    duplicating digits in the rendered row. `_extract_rows` re-extracts a parent
    cell from its own chars minus the chars its children already claim.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from collections import Counter
from typing import Any

import pdfplumber
from pdfplumber.utils import extract_text as _pdf_extract_text
from trimitdown_pdf import is_real_table

from ..common.config import Config
from ..common.jsonl import write_jsonl
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Nested-cell handling
# --------------------------------------------------------------------------- #
def _char_in(char: dict, bbox: tuple[float, float, float, float]) -> bool:
    """Half-open midpoint test matching pdfplumber's Table.extract exactly."""
    x0, top, x1, bottom = bbox
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2
    return x0 <= cx < x1 and top <= cy < bottom


def _encloses(outer: tuple, inner: tuple) -> bool:
    ox0, ot, ox1, ob = outer
    ix0, it, ix1, ib = inner
    if (ox1 - ox0) * (ob - ot) <= (ix1 - ix0) * (ib - it):
        return False
    return ox0 <= ix0 and ot <= it and ox1 >= ix1 and ob >= ib


def extract_rows(page: pdfplumber.page.Page, table: pdfplumber.table.Table, text_settings: dict) -> list[list[str]]:
    """Cell text for one grid, with nested cells subtracted from their parents."""
    rows: list[list[str | None]] = table.extract(**text_settings)
    # Table.rows rebuilds the row model on every access, so read it once.
    table_rows = table.rows
    cells = {c for row in table_rows for c in row.cells if c}
    nested = {c: [d for d in cells if d != c and _encloses(c, d)] for c in cells}
    nested = {c: inner for c, inner in nested.items() if inner}
    if not nested:
        return [[(c or "") for c in r] for r in rows]

    for ri, row in enumerate(table_rows):
        for ci, cell in enumerate(row.cells):
            inner = nested.get(cell) if cell else None
            if not inner:
                continue
            chars = [
                ch
                for ch in page.chars
                if _char_in(ch, cell) and not any(_char_in(ch, d) for d in inner)
            ]
            rows[ri][ci] = _pdf_extract_text(chars, **text_settings) if chars else ""
    return [[(c or "") for c in r] for r in rows]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _escape_pipe(value: str) -> str:
    return value.replace("|", "\\|")


def _clean_cell(value: str | None) -> str:
    return (value or "").replace("\n", " ").strip()


def rows_to_markdown(rows: list[list[str]], max_rows: int) -> tuple[str, bool]:
    """Render a GitHub-flavoured pipe table.

    Returns (markdown, truncated). A single-column grid would otherwise emit
    prose wrapped in pipes, so its rows are rendered as plain lines.
    """
    rows = [[_clean_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return "", False

    width = max(len(r) for r in rows)
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    rows = [r + [""] * (width - len(r)) for r in rows]

    if width == 1:
        return "\n".join(r[0] for r in rows), truncated

    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(_escape_pipe(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_escape_pipe(c) for c in r) + " |" for r in body]
    return "\n".join(lines), truncated


def find_caption_above(
    page: pdfplumber.page.Page, top: float, caption_patterns: list[re.Pattern[str]]
) -> str | None:
    """Caption text immediately above a grid.

    Codebooks put table captions above the grid; the nearest matching line is
    taken, and only within a short vertical distance so a caption on the previous
    table cannot be attached to this one.
    """
    best: tuple[float, str] | None = None
    for line in page.extract_text_lines(x_tolerance_ratio=0.12) or []:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        gap = top - line["bottom"]
        if not (-2 <= gap <= 60):
            continue
        if any(rx.match(text) for rx in caption_patterns):
            if best is None or gap < best[0]:
                best = (gap, text)
    return best[1] if best else None


def _safe_rows(page, grid, text_settings):
    """Rows for one candidate, or None if extraction blew up."""
    try:
        raw = extract_rows(page, grid, text_settings)
    except Exception:
        return None
    return [[_clean_cell(c) for c in r] for r in raw]


def _cell_text_agreement(variants: list[list[list[str]]]) -> float:
    """Fraction of cells where every strategy produced the same string.

    Strategies disagree legitimately (a text-aligned grid splits words that a
    lined grid keeps whole), so this is a similarity, not an equality. Cells are
    compared positionally after trimming; ragged variants are compared only over
    their overlapping shape.
    """
    if len(variants) < 2:
        return 1.0

    def score(offset: int) -> float:
        base = variants[0]
        total = matched = 0
        for ri, row in enumerate(base):
            for ci, cell in enumerate(row):
                if not (cell or "").strip():
                    continue  # empty cells carry no information
                total += 1
                ok = True
                for other in variants[1:]:
                    # Try a small row offset: strategies routinely disagree by a
                    # header row or a swallowed caption line, which shifts every
                    # cell by one and would otherwise score 0 on identical data.
                    value = None
                    for dr in (offset, 0, -offset):
                        r2 = ri + dr
                        if 0 <= r2 < len(other) and ci < len(other[r2]):
                            value = other[r2][ci]
                            break
                    if value is None or not _loose_equal((cell or "").strip(), (value or "").strip()):
                        ok = False
                        break
                if ok:
                    matched += 1
        return matched / total if total else 1.0

    return max(score(off) for off in (1, 2, 0))


def _loose_equal(a: str, b: str) -> bool:
    """Equal ignoring whitespace, which text strategies inject unpredictably."""
    return re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)


def _agreement(votes: dict[str, list[list[str]]]) -> float:
    return _cell_text_agreement(list(votes.values()))


def _render_region(pdf_path, page_no: int, bbox, dpi: int) -> str | None:
    """High-DPI crop of the table region, for human verification.

    Rendered next to the JSONL so a reviewer can diff the extraction against the
    source without reopening the PDF.
    """
    try:
        import pymupdf
    except ImportError:
        return None
    base = Path(_RENDER_DIR[0]) if _RENDER_DIR[0] else None
    if base is None:
        return None
    out_dir = base / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"table_p{page_no:04d}_{int(bbox[0])}_{int(bbox[1])}.png"
    try:
        doc = pymupdf.open(pdf_path)
        try:
            page = doc[page_no - 1]
            rect = pymupdf.Rect(*bbox)
            rect = pymupdf.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2) & page.rect
            page.get_pixmap(clip=rect, dpi=dpi).save(str(out_dir / name))
        finally:
            doc.close()
    except Exception:
        return None
    return f"tables/verification/{name}"


# Set by extract_tables so _render_region can write beside the JSONL.
_RENDER_DIR: list = [None]


def _page_table_captions(
    page: pdfplumber.page.Page, caption_patterns: list[re.Pattern[str]]
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Table caption lines on a page, with their rects."""
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    for line in page.extract_text_lines(x_tolerance_ratio=0.12) or []:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        if any(rx.match(text) for rx in caption_patterns):
            out.append((text, (line["x0"], line["top"], line["x1"], line["bottom"])))
    return out


def _find_borderedless(
    page: pdfplumber.page.Page,
    cap_rect: tuple[float, float, float, float],
    find_settings: dict,
    text_settings: dict,
    min_rows: int,
    min_cols: int,
    max_band: float,
):
    """Run the text strategy on the band a caption opens, yielding gated grids."""
    # A hair below the caption's reported rect: pdfplumber's line box can be
    # slightly shorter than the glyphs, and starting inside the caption folds
    # it in as a spurious first table row.
    band_top = cap_rect[3] + 1.5
    band_bottom = min(page.height, band_top + max_band)
    if band_bottom - band_top < 20:
        return
    try:
        cropped = page.crop((page.bbox[0], band_top, page.bbox[2], band_bottom))
    except Exception:
        return
    try:
        grids = cropped.find_tables(find_settings)
    except Exception:
        return

    for grid in grids:
        try:
            raw = [[_clean_cell(c) for c in r] for r in grid.extract(**text_settings)]
        except Exception:
            continue
        rows = [r for r in raw if any(r)]
        if not is_real_table(rows):
            continue
        if len(rows) < min_rows or max((len(r) for r in rows), default=0) < min_cols:
            continue
        # A pseudo-table that swallowed the page would fail this.
        h = (grid.bbox[3] - grid.bbox[1]) / max(1.0, page.height)
        if h > 0.6:
            continue
        yield grid, rows


def _iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _table_number(caption: str | None) -> str | None:
    if not caption:
        return None
    m = re.search(r"(?i)\btable\s*[-–—]?\s*([A-Z]?\d+(?:[.\-]\d+)*)", caption)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_tables(
    pdf_path: Path | str,
    config: Config,
    out_dir: Path | str,
    pages: tuple[int, int] | None = None,
) -> tuple[PhaseResult, list[dict[str, Any]], dict[int, list[tuple[float, float, float, float]]]]:
    """Extract tables.

    Returns the PhaseResult, the table records, and the kept-table bboxes per
    page (Phase 4 needs them so a figure is not cut out of a table).
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    v_strat = config.get("table_extraction.vertical_strategy", "lines")
    h_strat = config.get("table_extraction.horizontal_strategy", "lines")
    x_ratio = config.get("table_extraction.x_tolerance_ratio", 0.12)
    y_tol = config.get("table_extraction.y_tolerance", 3)
    min_rows = config.get("table_extraction.min_rows", 2)
    min_cols = config.get("table_extraction.min_cols", 2)
    max_rows = config.get("table_extraction.max_rows", 400)
    max_md_rows = config.get("table_extraction.max_markdown_rows", 60)

    standard = config.get("codebook.standard", "is456")
    patterns = config.clause_regexes(standard)
    table_caption_rx = [patterns["table_pattern"]] if "table_pattern" in patterns else []

    # Three candidate generators. See configs/default.yaml for why the obvious
    # find_tables/extract_table(strategy=...) triple is not used: `strategy=` is
    # silently dropped by pdfplumber, so all three would agree on everything.
    strategy_cfg = config.get("table_extraction.strategies", {}) or {}
    strategies: list[tuple[str, dict]] = []
    for name in ("lines", "text", "mixed"):
        s = strategy_cfg.get(name)
        if s:
            strategies.append((name, dict(s)))

    region_iou = config.get("table_extraction.region_iou", 0.55)
    min_votes = config.get("table_extraction.min_votes", 2)
    text_settings = {
        "x_tolerance_ratio": x_ratio,
        "y_tolerance": y_tol,
    }

    caption_band_fallback = config.get("table_extraction.caption_band_fallback", True)
    max_band_frac = config.get("table_extraction.caption_band_fraction", 0.45)
    render_png = config.get("table_extraction.render_verification_png", True)
    verify_dpi = config.get("table_extraction.verification_dpi", 300)

    _RENDER_DIR[0] = Path(out_dir) / "tables"
    result = PhaseResult(phase="phase3_tables", status=Status.SUCCESS)
    records: list[dict[str, Any]] = []
    kept_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    pages_with_candidates = 0
    pages_without_candidates = 0
    pages_all_rejected = 0

    with pdfplumber.open(pdf_path) as pdf:
        result.counts["pages"] = len(pdf.pages)
        target = None
        if pages:
            lo, hi = pages
            target = set(range(lo, hi + 1))
            result.counts["page_range"] = [lo, hi]

        n_candidates = 0
        n_regions = 0
        votes_hist: Counter[int] = Counter()

        for pno, page in enumerate(pdf.pages, start=1):
            if target is not None and pno not in target:
                continue

            # ---- collect candidates from every strategy ------------------- #
            # A region is a set of (strategy, grid) that agree geometrically.
            regions: list[dict[str, Any]] = []
            for strategy_name, find_settings in strategies:
                try:
                    grids = page.find_tables(find_settings)
                except Exception as exc:
                    result.warn(
                        f"page {pno}: find_tables({strategy_name}) raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                for grid in grids:
                    n_candidates += 1
                    gb = tuple(float(v) for v in grid.bbox)
                    rows = _safe_rows(page, grid, text_settings)
                    if rows is None:
                        continue
                    rows = [r for r in rows if any(r)]
                    if not is_real_table(rows):
                        continue  # the gate is the sole authority, per strategy
                    if len(rows) < min_rows or max((len(r) for r in rows), default=0) < min_cols:
                        continue
                    placed = False
                    for reg in regions:
                        if _iou(gb, reg["bbox"]) >= region_iou:
                            reg["votes"][strategy_name] = rows
                            placed = True
                            break
                    if not placed:
                        regions.append(
                            {"bbox": gb, "votes": {strategy_name: rows}, "strategy": strategy_name}
                        )

            # ---- borderless pass, cropped to each caption's band ----------- #
            # Only where a caption exists: run globally, the text strategy
            # invents tables out of aligned two-column prose, and Nurminen's
            # row-fill test cannot tell those apart from real tables.
            if caption_band_fallback:
                for cap_text, cap_rect in _page_table_captions(page, table_caption_rx):
                    band = _find_borderedless(
                        page, cap_rect, dict(strategy_cfg.get("text") or {}),
                        text_settings, min_rows, min_cols, max_band_frac * page.height,
                    )
                    for grid, rows in band:
                        gb = tuple(float(v) for v in grid.bbox)
                        for reg in regions:
                            if _iou(gb, reg["bbox"]) >= region_iou:
                                reg["votes"].setdefault("caption_band_text", rows)
                                break
                        else:
                            regions.append(
                                {"bbox": gb, "votes": {"caption_band_text": rows},
                                 "strategy": "caption_band_text"}
                            )

            kept = 0
            page_boxes: list[tuple[float, float, float, float]] = []
            for reg in regions:
                n_regions += 1
                # Attach the caption up front: it is the corroborating signal for
                # a borderless region and gates it below.
                reg["caption"] = find_caption_above(page, reg["bbox"][1], table_caption_rx)
                votes = reg["votes"]
                votes_hist[len(votes)] += 1
                # A borderless region is proposed only by the caption-band pass --
                # the ruled strategies find nothing there by definition -- so
                # requiring two of them would discard every borderless table.
                # Requiring a caption is the corroboration in that case, and the
                # absence of a second opinion is recorded rather than hidden.
                band_only = set(votes) == {"caption_band_text"}
                if band_only and not reg["caption"]:
                    continue  # uncorroborated: no ruling vote and no caption
                if not band_only and len(votes) < min_votes:
                    continue

                # Primary extraction: the strategy with the most cells, since a
                # table that more strategies agree on should also carry more of
                # its content.
                best_strategy = max(votes, key=lambda s: sum(len(r) for r in votes[s]))
                rows = votes[best_strategy]
                agreement = _agreement(votes)
                single_detector = band_only
                if single_detector:
                    # Never `unanimous`: nothing corroborated it.
                    confidence = "majority" if agreement >= 0.9 else "degraded"
                elif len(votes) >= len(strategies) and agreement == 1.0:
                    confidence = "unanimous"
                elif agreement >= 0.9:
                    confidence = "majority"
                else:
                    confidence = "degraded"

                caption = reg["caption"]
                number = _table_number(caption)
                markdown, md_truncated = rows_to_markdown(rows, max_md_rows)

                record = {
                    "table_id": "",
                    "table_number": number,
                    "clause_ref": None,  # filled by linker.py
                    "caption": caption,
                    "markdown": markdown,
                    "json_data": rows[:max_rows],
                    "page": pno,
                    "bbox": [round(float(v), 2) for v in reg["bbox"]],
                    "n_rows": len(rows),
                    "n_cols": max((len(r) for r in rows), default=0),
                    "markdown_truncated": md_truncated,
                    "rows_truncated": len(rows) > max_rows,
                    "confidence": confidence,
                    "single_detector": single_detector,
                    "vote_strategies": sorted(votes),
                    "n_votes": len(votes),
                    "cell_agreement": round(agreement, 3),
                    "strategy_variants": (
                        {s: votes[s][:max_rows] for s in votes} if agreement < 1.0 else None
                    ),
                    "verification_png": None,
                }
                if render_png:
                    png_name = _render_region(pdf_path, pno, reg["bbox"], verify_dpi)
                    record["verification_png"] = png_name
                records.append(record)
                page_boxes.append(reg["bbox"])
                kept += 1

            if kept:
                kept_boxes[pno] = page_boxes
            elif regions:
                pages_all_rejected += 1
            else:
                pages_without_candidates += 1

        # A caption band can re-propose a table the ruled strategies already
        # found, under the same caption. Prefer the corroborated extraction: drop
        # a single_detector record whose number is already claimed by a region
        # that multiple strategies agreed on.
        corroborated = {
            r["table_number"] for r in records
            if r.get("table_number") and not r.get("single_detector")
        }
        before = len(records)
        records = [
            r for r in records
            if not (
                r.get("single_detector")
                and r.get("table_number")
                and r["table_number"] in corroborated
            )
        ]
        shadowed = before - len(records)

        # Stable ids: prefer the caption's own number, else a per-page position.
        by_number_seen: dict[str, int] = {}
        per_page: dict[int, int] = {}
        for rec in records:
            number = rec["table_number"]
            if number:
                by_number_seen[number] = by_number_seen.get(number, 0) + 1
                dup = "" if by_number_seen[number] == 1 else f"_dup{by_number_seen[number]}"
                rec["table_id"] = f"T{number}{dup}"
            else:
                per_page[rec["page"]] = per_page.get(rec["page"], 0) + 1
                rec["table_id"] = f"p{rec['page']}_t{per_page[rec['page']]}"

        # A repeated table number means at least one occurrence is a footnote or
        # an in-text cross-reference rather than a real caption. Several
        # discriminators were tried on IS 456 and all failed: fill density
        # (0.67 vs 0.66), font size (both at body size), and caption shape. So
        # rather than guess which is genuine, the first occurrence in page order
        # is treated as primary and the rest are flagged for review.
        by_number: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            if rec["table_number"]:
                by_number.setdefault(rec["table_number"], []).append(rec)
        conflicts = 0
        for number, group in by_number.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda r: r["page"])
            for rank, rec in enumerate(group):
                rec["duplicate_table_number"] = rank > 0
                rec["duplicate_rank"] = rank
            conflicts += len(group) - 1

        result.counts["candidate_grids"] = n_candidates
        result.counts["candidate_regions"] = n_regions
        result.counts["tables_kept"] = len(records)
        result.counts["votes_histogram"] = {str(k): v for k, v in sorted(votes_hist.items())}
        result.counts["unanimous"] = sum(1 for r in records if r["confidence"] == "unanimous")
        result.counts["majority"] = sum(1 for r in records if r["confidence"] == "majority")
        result.counts["degraded"] = sum(1 for r in records if r["confidence"] == "degraded")
        result.counts["single_detector"] = sum(1 for r in records if r.get("single_detector"))
        result.counts["shadowed_by_better_region"] = shadowed
        result.counts["duplicate_table_numbers"] = conflicts
        result.counts["candidates_rejected"] = n_candidates - len(records)
        result.counts["pages_with_candidates"] = len(kept_boxes)
        result.counts["pages_no_candidates"] = pages_without_candidates
        result.counts["pages_all_rejected"] = pages_all_rejected
        if records:
            result.counts["with_caption"] = sum(1 for r in records if r["caption"])
            result.counts["with_number"] = sum(1 for r in records if r["table_number"])
            result.counts["markdown_truncated"] = sum(1 for r in records if r["markdown_truncated"])
        reject_pct = (100.0 * result.counts["candidates_rejected"] / n_candidates) if n_candidates else 0.0
        result.counts["rejected_pct"] = round(reject_pct, 1)

    if result.counts["candidate_grids"] == 0:
        result.degrade(
            "no ruled grids found on any page. trimitdown-pdf detects ruled "
            "tables only, so borderless tables are invisible to this phase."
        )
    if n_candidates:
        log.info(
            "table gate: %d candidate grid(s) -> %d kept (%.1f%% rejected)",
            n_candidates,
            len(records),
            reject_pct,
        )
    if records and result.counts.get("with_number", 0) < len(records):
        missing = len(records) - result.counts["with_number"]
        result.warn(
            f"{missing} table(s) have no 'Table N' caption within 60pt above the grid; "
            "their table_id is positional and clause linking will be weaker"
        )
    if result.counts.get("single_detector"):
        result.warn(
            f"{result.counts['single_detector']} table(s) were proposed by the "
            "caption-band pass alone, because the document draws them without "
            "ruling. They are gated by is_real_table and anchored to their caption, "
            "but no second extraction strategy could corroborate them"
        )
    if result.counts.get("degraded"):
        # A `degraded` label means the strategies saw genuinely different cell
        # contents in the same region. Keeping it as a normal table would hide
        # that, so the phase says so.
        bad = [r["table_id"] for r in records if r["confidence"] == "degraded"]
        result.degrade(
            f"{len(bad)} table(s) had disagreeing cell contents across extraction "
            f"strategies ({', '.join(bad[:10])}{'...' if len(bad) > 10 else ''}). Their "
            "markdown comes from the strategy with the most cells; the alternatives are "
            "kept in strategy_variants for inspection."
        )
    if result.counts.get("duplicate_table_numbers"):
        result.warn(
            f"{result.counts['duplicate_table_numbers']} table(s) reuse a table number "
            "already seen earlier in the document; these are footnotes or in-text "
            "cross-references, flagged duplicate_table_number=true. The first "
            "occurrence in page order is treated as the real table."
        )
    if result.counts.get("markdown_truncated", 0):
        result.warn(
            f"{result.counts['markdown_truncated']} table(s) exceeded "
            f"{max_md_rows} rows; markdown is truncated (json_data retains up to {max_rows} rows)"
        )

    write_jsonl(out_dir / "tables" / "tables.jsonl", records)
    result.counts["output"] = str(out_dir / "tables" / "tables.jsonl")
    return result, records, kept_boxes