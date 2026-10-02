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
    band_top = cap_rect[3]
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

    settings = {
        "vertical_strategy": v_strat,
        "horizontal_strategy": h_strat,
    }
    # pdfplumber ignores x_tolerance entirely once a ratio is present, so the
    # ratio is the only tolerance that reaches cell extraction.
    text_settings = {"x_tolerance_ratio": x_ratio, "y_tolerance": y_tol}

    # Second candidate generator. trimitdown-pdf's line strategy is blind to
    # borderless tables: on IS 456 it found 9 grids across 114 pages against 31
    # "Table N" captions in the text, i.e. ~15% recall. The `text` strategy finds
    # tables the line strategy scores zero candidates on.
    #
    # This does not bypass the gate -- it only produces *more candidates* for the
    # same `is_real_table` test to judge. The gate stays the sole authority on
    # what counts as a table; the strategy only decides what to offer it.
    text_fallback = config.get("table_extraction.text_strategy_fallback", True)
    text_tol = config.get("table_extraction.text_x_tolerance", 2)
    max_band_frac = config.get("table_extraction.caption_band_fraction", 0.45)
    text_settings_find = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "text_x_tolerance": text_tol,
    }

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
        n_from_text = 0
        for pno, page in enumerate(pdf.pages, start=1):
            if target is not None and pno not in target:
                continue

            def run(strategy_name: str, find_settings: dict) -> list[tuple[float, float, float, float]]:
                nonlocal n_candidates, n_from_text
                boxes: list[tuple[float, float, float, float]] = []
                try:
                    grids = page.find_tables(find_settings)
                except Exception as exc:
                    result.warn(
                        f"page {pno}: find_tables({strategy_name}) raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return boxes

                for grid in grids:
                    gb = tuple(float(v) for v in grid.bbox)
                    # The text strategy re-proposes grids the line strategy
                    # already accepted; offering a duplicate to the gate would
                    # emit the same table twice.
                    if any(_iou(gb, b) >= 0.6 for b in page_boxes):
                        continue
                    n_candidates += 1
                    if strategy_name == "text":
                        n_from_text += 1

                    try:
                        raw = extract_rows(page, grid, text_settings)
                    except Exception as exc:
                        result.warn(
                            f"page {pno} {strategy_name} grid at "
                            f"{[round(v) for v in grid.bbox]}: extraction raised "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                    rows = [[_clean_cell(c) for c in r] for r in raw]
                    rows = [r for r in rows if any(r)]

                    if not is_real_table(rows):
                        # Documented fallback: a rejected candidate produces no
                        # record. Nothing is forced through the gate.
                        continue
                    if len(rows) < min_rows or max((len(r) for r in rows), default=0) < min_cols:
                        continue

                    caption = find_caption_above(page, grid.bbox[1], table_caption_rx)
                    number = _table_number(caption)
                    markdown, md_truncated = rows_to_markdown(rows, max_md_rows)

                    records.append(
                        {
                            "table_id": "",
                            "table_number": number,
                            "clause_ref": None,  # filled by linker.py
                            "caption": caption,
                            "markdown": markdown,
                            "json_data": rows[:max_rows],
                            "page": pno,
                            "bbox": [round(float(v), 2) for v in grid.bbox],
                            "n_rows": len(rows),
                            "n_cols": max((len(r) for r in rows), default=0),
                            "markdown_truncated": md_truncated,
                            "rows_truncated": len(rows) > max_rows,
                            "confidence": "text_layer",
                            "detection_strategy": strategy_name,
                        }
                    )
                    boxes.append(gb)
                return boxes

            page_boxes: list[tuple[float, float, float, float]] = []

            # Pass 1: ruled grids.
            page_boxes.extend(run("lines", settings))
            n_ruled = len(page_boxes)

            # Pass 2: borderless tables.
            #
            # IS 456's tables carry no ruling at all (page 28: two horizontal
            # rules, both header/footer, and zero vertical rules), so the line
            # strategy correctly finds nothing and the text strategy, run on the
            # whole page, returns page-sized pseudo-tables that contain the
            # running head. Nurminen's row-fill test cannot separate a borderless
            # table from two-column prose -- both have rows with two filled
            # columns -- so the detector has to be told where to look.
            #
            # The document says where: a "Table N" caption marks the top. Each
            # caption opens a band, the page is cropped to it, and the text
            # strategy runs only inside. That is the whole reason this pass has
            # better precision than running it globally.
            if text_fallback and n_ruled == 0:
                captions = _page_table_captions(page, table_caption_rx)
                for cap_text, cap_rect in captions:
                    for grid, rows in _find_borderedless(
                        page, cap_rect, text_settings_find, text_settings,
                        min_rows, min_cols, max_band_frac * page.height,
                    ):
                        gb = tuple(float(v) for v in grid.bbox)
                        if any(_iou(gb, b) >= 0.6 for b in page_boxes):
                            continue
                        n_candidates += 1
                        n_from_text += 1
                        number = _table_number(cap_text)
                        markdown, md_truncated = rows_to_markdown(rows, max_md_rows)
                        records.append(
                            {
                                "table_id": "",
                                "table_number": number,
                                "clause_ref": None,
                                "caption": cap_text,
                                "markdown": markdown,
                                "json_data": rows[:max_rows],
                                "page": pno,
                                "bbox": [round(float(v), 2) for v in grid.bbox],
                                "n_rows": len(rows),
                                "n_cols": max((len(r) for r in rows), default=0),
                                "markdown_truncated": md_truncated,
                                "rows_truncated": len(rows) > max_rows,
                                "confidence": "text_layer",
                                "detection_strategy": "caption_band_text",
                            }
                        )
                        page_boxes.append(gb)
                        break  # one table per caption band

            if page_boxes:
                kept_boxes[pno] = page_boxes
            elif n_ruled:
                pages_all_rejected += 1
            else:
                pages_without_candidates += 1

        # Stable ids: prefer the caption's own number, else a per-page position.
        by_number_seen: dict[str, int] = {}
        per_page: dict[int, int] = {}
        for rec in records:
            number = rec["table_number"]
            if number:
                key = number
                by_number_seen[key] = by_number_seen.get(key, 0) + 1
                suffix = "" if by_number_seen[key] == 1 else f"_dup{by_number_seen[key]}"
                rec["table_id"] = f"T{number}{suffix}"
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
        result.counts["candidates_from_text_strategy"] = n_from_text
        result.counts["tables_kept"] = len(records)
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