"""Phase 4: figure extraction.

Two sources, merged.

Primary -- caption-anchored region detection (PyMuPDF)
-----------------------------------------------------
PDFFigures 2.0 is trained on scholarly papers and was measured at **0% figure
recall** on this project's codebook fixture (5 captioned figures, 0 detected) and
29% table recall with 50% precision. It therefore cannot be the primary path for
codebooks.

Instead, figures are located the way they actually appear in a codebook: a
caption line ("Fig. 1 Typical Arrangement of Reinforcement") plus the graphic
region adjacent to it. PyMuPDF gives the page's drawing and image rectangles
with exact coordinates, so the region is determined geometrically rather than
guessed. Captions are matched at the *start of a text line*, because a body
cross-reference ("as shown in Fig. 4.") must not be mistaken for a caption.

Secondary -- PDFFigures 2.0 (external tool, per the project spec)
------------------------------------------------------------------
Still invoked, because it is the only detector that finds *uncaptioned* figures.
Its findings are merged on top of the primary set and always carry lower
confidence. Nothing is silently dropped: figure-extractor discards entries with
no `renderURL`, so failures vanish unless this module looks for them.

Status is derived here, not read from the tool. There is no `suspect`/`failed`
field in the PDFFigures 2.0 output, so one is computed:
    FAILED  - no PNG on disk, or the item was reported but not rendered
    SUSPECT - rendered, but no caption, or implausible geometry
    OK      - rendered with a caption at plausible geometry
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from ..common.config import Config
from ..common.jsonl import write_jsonl
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

FIG_OK = "OK"
FIG_SUSPECT = "SUSPECT"
FIG_FAILED = "FAILED"

# Confidence labels describing how many independent signals agreed.
#   multi_signal  - PDFFigures 2.0 and the vector pass both found this figure
#   caption_only  - vector pass found it; PDFFigures 2.0 did not
#   vector_only   - a vector region with no caption at all (an orphan)
#   degraded      - the two detectors disagree about it
CONF_MULTI = "multi_signal"
CONF_CAPTION = "caption_only"
CONF_VECTOR = "vector_only"
CONF_FIG_DEGRADED = "degraded"

# Drawings closer than this (pt) belong to the same visual region.
CLUSTER_GAP_PT = 14.0
# Ignore rules thinner than this: underlines and hairline separators.
MIN_STROKE = 0.6


@dataclass
class Region:
    """A cluster of drawing/image rectangles forming one graphic region."""

    x0: float
    y0: float
    x1: float
    y1: float
    n_parts: int = 0
    has_image: bool = False
    has_vector: bool = False
    caption: str | None = None
    caption_rect: tuple[float, float, float, float] | None = None
    caption_above: bool = True

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def union(self, other: "Region") -> None:
        self.x0 = min(self.x0, other.x0)
        self.y0 = min(self.y0, other.y0)
        self.x1 = max(self.x1, other.x1)
        self.y1 = max(self.y1, other.y1)
        self.n_parts += other.n_parts
        self.has_image = self.has_image or other.has_image
        self.has_vector = self.has_vector or other.has_vector


def _gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Separating distance between two rects (0 if they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return (dx * dx + dy * dy) ** 0.5


def _merge_regions(
    rects: list[tuple[float, float, float, float, bool, bool]],
    cluster_gap: float = CLUSTER_GAP_PT,
) -> list[Region]:
    """Union-find merge of nearby rectangles into graphic regions.

    A figure drawn as a box produces one fill plus four separate stroke
    rectangles; without merging, each edge would be judged as its own (degenerate)
    graphic and none would match a caption.
    """
    regions = [Region(*r[:4], n_parts=1, has_image=r[4], has_vector=r[5]) for r in rects]
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    changed = True
    while changed:
        changed = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                ri, rj = find(i), find(j)
                if ri == rj:
                    continue
                if _gap(regions[ri].bbox, regions[rj].bbox) <= cluster_gap:
                    regions[ri].union(regions[rj])
                    parent[rj] = ri
                    changed = True
    # One entry per distinct root. Returning regions[find(i)] for every i would
    # yield the same merged Region many times over.
    seen: dict[int, Region] = {}
    for i in range(len(regions)):
        root = find(i)
        seen.setdefault(root, regions[root])
    return list(seen.values())


def _is_page_frame(x0: float, y0: float, x1: float, y1: float, pw: float, ph: float) -> bool:
    """A border rule around the whole page is page furniture, not a figure.

    Area alone does not catch it: a typical frame is ~80-85% of the page area, so
    a plain `area > max_fraction` test lets it through -- and because the frame
    *contains* every figure on the page, cluster merging then swallows each
    figure into the frame and no caption can ever match. A frame is identified by
    spanning nearly the full width *and* height instead.
    """
    w, h = x1 - x0, y1 - y0
    return w >= 0.80 * pw and h >= 0.80 * ph and (w * h) >= 0.60 * pw * ph


def _page_regions(
    page: pymupdf.Page,
    table_boxes: list[tuple[float, float, float, float]],
    page_area: float,
    max_area_frac: float,
    cluster_gap: float = CLUSTER_GAP_PT,
) -> list[Region]:
    """Graphic regions on one page, excluding tables and page frames."""
    rects: list[tuple[float, float, float, float, bool, bool]] = []
    pw, ph = float(page.rect.width), float(page.rect.height)

    for info in page.get_image_info():
        x0, y0, x1, y1 = (float(v) for v in info["bbox"])
        area = (x1 - x0) * (y1 - y0)
        if area <= 0 or area > max_area_frac * page_area:
            continue
        if _is_page_frame(x0, y0, x1, y1, pw, ph):
            continue
        rects.append((x0, y0, x1, y1, True, False))

    for d in page.get_drawings():
        r = d["rect"]
        x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
        w, h = x1 - x0, y1 - y0
        if w <= 0.5 and h <= 0.5:
            continue  # degenerate point
        if _is_page_frame(x0, y0, x1, y1, pw, ph):
            continue
        # A region this large is a layout block, not an extractable figure.
        if w * h > max_area_frac * page_area:
            continue
        # Hairline rules (underlines, separators) are not figures.
        if min(w, h) < MIN_STROKE:
            continue
        rects.append((x0, y0, x1, y1, False, True))

    regions = _merge_regions(rects, cluster_gap)

    # Drop regions that lie inside a table: table content is Phase 3's job.
    kept: list[Region] = []
    for r in regions:
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if any(bx0 - 2 <= cx <= bx1 + 2 and by0 - 2 <= cy <= by1 + 2 for bx0, by0, bx1, by1 in table_boxes):
            continue
        kept.append(r)
    return kept


def _page_caption_lines(page: pymupdf.Page, caption_rx: re.Pattern[str]) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Caption lines with their rects.

    Matched per text line, not against the flattened page text: "Fig. 3" appears
    both in its caption and inside a body cross-reference, and only the line-start
    occurrence is a caption.
    """
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = text.strip()
            if not text:
                continue
            if caption_rx.match(text):
                bbox = line.get("bbox")
                out.append((text, tuple(float(v) for v in bbox)))
    return out


def _assign_captions(
    regions: list[Region],
    captions: list[tuple[str, tuple[float, float, float, float]]],
    max_gap: float,
) -> tuple[int, int]:
    """Attach each caption to its adjacent graphic region.

    Codebooks set figure captions above the graphic; a caption below is accepted
    as a fallback. Nearest-in-vertical-distance wins, so consecutive figures on
    one page each keep their own.
    """
    unmatched_regions = len(regions)
    for text, (cx0, cy0, cx1, cy1) in captions:
        best = None
        best_key = None
        for r in regions:
            if r.caption is not None:
                continue
            h_overlap = min(cx1, r.x1) - max(cx0, r.x0)
            if h_overlap <= 0:
                continue
            # Graphic above the caption (the usual codebook convention).
            if r.y1 <= cy0 + 4:
                gap = cy0 - r.y1
                if gap <= max_gap and (best_key is None or gap < best_key):
                    best, best_key = r, gap
                    above = True
            # Fallback: graphic below the caption.
            elif r.y0 >= cy1 - 4:
                gap = r.y0 - cy1
                if gap <= max_gap and (best_key is None or gap + 1000 < best_key):
                    best, best_key = r, gap + 1000
                    above = False
        if best is not None and best_key is not None:
            best.caption = text
            best.caption_rect = (cx0, cy0, cx1, cy1)
            best.caption_above = best_key < 1000
            unmatched_regions -= 1
    return len(captions), unmatched_regions


def _render(pdf_path: Path, page_no: int, bbox: tuple[float, float, float, float], dpi: int) -> bytes:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_no - 1]
        rect = pymupdf.Rect(*bbox)
        # Pad slightly so anti-aliased edges are not clipped.
        rect = pymupdf.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2) & page.rect
        pix = page.get_pixmap(clip=rect, dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# PDFFigures 2.0 (secondary)
# --------------------------------------------------------------------------- #
def run_pdffigures2(
    pdf_path: Path,
    work_dir: Path,
    config: Config,
    tools_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Invoke the external tool. Returns (raw json list, error message)."""
    jar = tools_dir / "figure-extractor" / "pdffigures2" / "pdffigures2.jar" if tools_dir else None
    if jar is None or not jar.exists():
        return [], "pdffigures2.jar not found; run scripts/setup_deps.sh"
    java = tools_dir / "jdk11" / "bin" / "java" if tools_dir else None
    if java is None or not java.exists():
        java = Path("java")  # fall back to PATH

    work_dir.mkdir(parents=True, exist_ok=True)
    timeout = config.get("figure_extraction.timeout_seconds", 1800)
    dpi = config.get("figure_extraction.dpi", 300)

    cmd = [
        str(java),
        "-Xmx2g",
        "-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider",
        "-jar",
        str(jar),
        str(pdf_path.resolve()),
        "-m",
        str(work_dir.resolve()) + os.sep,
        "-d",
        str(work_dir.resolve()) + os.sep,
        "--dpi",
        str(dpi),
    ]
    log.info("running pdffigures2: %s", " ".join(cmd[:3] + ["..."]))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"pdffigures2 timed out after {timeout}s"
    except FileNotFoundError as exc:
        return [], f"cannot execute java: {exc}"

    if proc.returncode != 0:
        return [], f"pdffigures2 exit {proc.returncode}: {(proc.stderr or '')[-400:]}"

    meta = work_dir / f"{pdf_path.stem}.json"
    if not meta.exists():
        return [], f"pdffigures2 produced no metadata file at {meta}"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], f"unreadable pdffigures2 metadata: {exc}"
    # Tuple, not a bare list: callers unpack this as (items, error), and a bare
    # list of exactly two entries unpacks into the two dicts instead.
    return (data if isinstance(data, list) else []), None


def _pdf_norm_bbox(b: dict[str, float]) -> tuple[float, float, float, float]:
    return (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_figures(
    pdf_path: Path | str,
    config: Config,
    out_dir: Path | str,
    table_boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
) -> tuple[PhaseResult, list[dict[str, Any]]]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_boxes = table_boxes or {}

    standard = config.get("codebook.standard", "is456")
    patterns = config.clause_regexes(standard)
    caption_rx = patterns.get("figure_pattern")
    if caption_rx is None:
        raise ValueError(f"configs/standards/{standard}.yaml has no caption.figure_pattern")

    fallback_dpi = config.get("figure_extraction.caption_fallback_dpi", 200)
    verify_dpi = config.get("figure_extraction.verify_dpi", 600)
    verify_enabled = config.get("figure_extraction.verification_enabled", True)
    cluster_gap = config.get("figure_extraction.cluster_gap_pt", CLUSTER_GAP_PT)
    save_orphans = config.get("figure_extraction.save_orphans", True)
    orphan_min = config.get("figure_extraction.orphan_min_area_pt2", 20000)
    orphan_max_frac = config.get("figure_extraction.orphan_max_area_fraction", 0.35)
    orphan_min_parts = config.get("figure_extraction.orphan_min_parts", 4)
    max_area_frac = config.get("figure_extraction.suspect_max_page_area_fraction", 0.85)
    min_area = config.get("figure_extraction.suspect_min_area_pt2", 3000)
    max_gap = config.get("figure_extraction.max_clause_gap_pt", 900)
    enabled = config.get("figure_extraction.enabled", True)
    use_fallback = config.get("figure_extraction.caption_fallback", True)

    result = PhaseResult(phase="phase4_figures", status=Status.SUCCESS)
    records: list[dict[str, Any]] = []
    orphans: list[dict[str, Any]] = []
    doc = pymupdf.open(pdf_path)
    try:
        page_area = float(doc[0].rect.get_area()) if doc.page_count else 1.0
        total_captions = 0
        unmatched = 0
        for pno in range(doc.page_count):
            page = doc[pno]
            page_no = pno + 1
            boxes = table_boxes.get(page_no, [])
            regions = (
                _page_regions(page, boxes, page_area, max_area_frac, cluster_gap)
                if use_fallback
                else []
            )
            captions = _page_caption_lines(page, caption_rx)
            total_captions += len(captions)
            n_cap, n_unmatched = _assign_captions(regions, captions, max_gap)
            unmatched += n_unmatched

            for region in regions:
                if region.caption is None:
                    continue
                # Include the caption in the crop so the image is self-describing.
                if region.caption_rect:
                    cr = region.caption_rect
                    bbox = (
                        min(region.x0, cr[0]),
                        min(region.y0, cr[1]),
                        max(region.x1, cr[2]),
                        max(region.y1, cr[3]),
                    )
                else:
                    bbox = region.bbox

                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                figure_id = f"p{page_no}_f{len([r for r in records if r['page'] == page_no]) + 1}"
                png = fig_dir / f"{figure_id}.png"
                status = FIG_OK
                reason = ""
                try:
                    png.write_bytes(_render(pdf_path, page_no, bbox, fallback_dpi))
                except Exception as exc:
                    status = FIG_FAILED
                    reason = f"render failed: {type(exc).__name__}: {exc}"

                if status == FIG_OK and not png.exists():
                    status = FIG_FAILED
                    reason = "render reported success but no PNG on disk"
                if status == FIG_OK and (area < min_area or area > max_area_frac * page_area):
                    status = FIG_SUSPECT
                    reason = f"implausible area {area:.0f}pt2"

                num = re.search(r"(?i)\bfig(?:ure)?\.?\s*[-–—]?\s*([A-Z]?\d+(?:[.\-]\d+)*)", region.caption)
                records.append(
                    {
                        "figure_id": figure_id,
                        "figure_number": num.group(1) if num else None,
                        "png_path": str(png.relative_to(out_dir)) if png.exists() else None,
                        "caption": region.caption,
                        "page": page_no,
                        "bbox": [round(v, 2) for v in bbox],
                        "graphic_bbox": [round(v, 2) for v in region.bbox],
                        "related_clause": None,
                        "is_inside_table": False,
                        # Upgraded to `multi_signal` below if the independent
                        # PDFFigures 2.0 pass independently reports this region.
                        "confidence": CONF_CAPTION,
                        "detector": "vector_caption",
                        "signals": ["vector_region", "caption"],
                        "status": status,
                        "status_reason": reason,
                        "caption_above": region.caption_above,
                        "n_vector_parts": region.n_parts if region.has_vector else 0,
                        "has_image": region.has_image,
                    }
                )
            # Orphan pass: uncaptioned vector regions. These are figure-like
            # blocks the document never labelled -- schematic panels, legend
            # boxes. Saved as `vector_only` because no caption signal exists to
            # corroborate them.
            if save_orphans:
                for region in regions:
                    if region.caption is not None:
                        continue
                    area = (region.x1 - region.x0) * (region.y1 - region.y0)
                    if area < orphan_min or area > orphan_max_frac * page_area:
                        continue
                    # A drawn figure is made of many segments (or is a raster
                    # image). One or two rectangles is a stray rule or a box
                    # around a margin note, and admitting those as figures
                    # poisons Phase 5's linking -- on IS 456 that alone took 10
                    # links down to `low` and left 3 unlinked.
                    if region.n_parts < orphan_min_parts and not region.has_image:
                        continue
                    png = fig_dir / f"orphan_p{page_no}_{int(region.x0)}_{int(region.y0)}.png"
                    status, reason = FIG_OK, ""
                    try:
                        png.write_bytes(_render(pdf_path, page_no, region.bbox, verify_dpi))
                    except Exception as exc:
                        status, reason = FIG_FAILED, f"render failed: {exc}"
                    orphans.append(
                        {
                            "figure_id": png.stem,
                            "figure_number": None,
                            "png_path": str(png.relative_to(out_dir)) if png.exists() else None,
                            "caption": None,
                            "page": page_no,
                            "bbox": [round(v, 2) for v in region.bbox],
                            "graphic_bbox": [round(v, 2) for v in region.bbox],
                            "related_clause": None,
                            "is_inside_table": False,
                            "confidence": CONF_VECTOR,
                            "detector": "vector_only",
                            "status": status,
                            "status_reason": reason,
                        }
                    )

        result.counts["pages"] = doc.page_count
        result.counts["caption_lines_found"] = total_captions
        result.counts["figures_caption_anchored"] = len(records)
        result.counts["graphic_regions_unmatched"] = unmatched
        result.counts["orphan_regions"] = len(orphans)
    finally:
        doc.close()

    # ---- secondary: PDFFigures 2.0 ---------------------------------------- #
    p2_items: list[dict[str, Any]] = []
    if enabled:
        try:
            with tempfile.TemporaryDirectory(prefix="p2_") as tmp:
                raw, err = run_pdffigures2(pdf_path, Path(tmp), config, config.tools_dir())
                if err:
                    result.warn(f"pdffigures2 unavailable ({err}); caption-anchored detection only")
                else:
                    result.counts["pdffigures2_items"] = len(raw)
                    result.counts["pdffigures2_figures"] = sum(
                        1 for i in raw if i.get("figType") == "Figure"
                    )
                    p2_items = _merge_pdffigures2(raw, Path(tmp), records, config)
        except Exception as exc:
            result.warn(f"pdffigures2 raised {type(exc).__name__}: {exc}")
            p2_items = []
    else:
        result.degrade("figure_extraction.enabled=false; pdffigures2 was not run")

    # ---- independent-detector cross-check --------------------------------- #
    kept, cross = _cross_check(records, p2_items, orphans, fig_dir, out_dir)
    result.counts.update(cross)
    result.counts["figures_total"] = len(kept)
    result.counts["figures_ok"] = sum(1 for r in kept if r["status"] == FIG_OK)
    result.counts["figures_suspect"] = sum(1 for r in kept if r["status"] == FIG_SUSPECT)
    result.counts["figures_failed"] = sum(1 for r in kept if r["status"] == FIG_FAILED)
    result.counts["figures_merged_as_duplicate"] = len(records) + len(p2_items) - len(kept)

    if not kept:
        result.status = Status.FAILED
        result.error = "no figures detected by either the caption-anchored detector or pdffigures2"
    elif result.counts["figures_failed"]:
        result.warn(f"{result.counts['figures_failed']} figure(s) failed and have no usable image")
    elif result.counts["figures_suspect"]:
        result.warn(
            f"{result.counts['figures_suspect']} figure(s) flagged SUSPECT "
            "(no caption or implausible geometry) -- verify before training on them"
        )
    if total_captions and len([r for r in kept if r["detector"] == "pycaption"]) < total_captions:
        result.warn(
            f"{total_captions} caption line(s) found but only "
            f"{len([r for r in kept if r['detector'] == 'pycaption'])} were matched to a graphic "
            "region; some captions may sit on a page whose graphic is a raster image "
            "outside the detected region, or may be split across lines"
        )
    result.degrade(
        "caption-anchored detection requires the caption and its graphic to be within "
        f"{max_gap:.0f}pt on the same page; figures whose caption is on a facing page "
        "or in a running list are not captured"
    )

    write_jsonl(fig_dir / "figures_metadata.jsonl", kept)
    result.counts["output"] = str(fig_dir / "figures_metadata.jsonl")
    return result, kept


def _cross_check(
    records: list[dict[str, Any]],
    p2_items: list[dict[str, Any]],
    orphans: list[dict[str, Any]],
    fig_dir: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconcile the vector pass against PDFFigures 2.0.

    The two detectors are independent, so agreement between them is real
    evidence and disagreement is real uncertainty:

      both find it   -> `multi_signal`
      vector only    -> `caption_only`
      vector, no caption -> `vector_only` (an orphan)
      p2 only        -> `degraded`; both crops are kept so a human can see why

    PDFFigures 2.0 writes into a temporary directory that is deleted when this
    function returns, so its PNGs are copied into the project's own figure
    directory. Without that, a `degraded` record would point at a path that no
    longer exists.
    """
    stats = {"cross_multi_signal": 0, "cross_degraded": 0, "cross_p2_only": 0}
    used_p2: set[int] = set()

    for rec in records:
        bb = rec.get("bbox")
        if not bb:
            continue
        best, best_iou = None, 0.0
        for i, item in enumerate(p2_items):
            if i in used_p2 or item.get("page") != rec.get("page"):
                continue
            ibb = item.get("bbox")
            if not ibb:
                continue
            v = _iou(tuple(bb), tuple(ibb))
            if v > best_iou:
                best, best_iou = item, v
        if best is not None and best_iou >= 0.3:
            used_p2.add(p2_items.index(best))
            rec["confidence"] = CONF_MULTI
            rec["signals"] = sorted(set(rec.get("signals", [])) | {"pdffigures2"})
            rec["pdffigures2_agreement_iou"] = round(best_iou, 3)
            rec["pdffigures2_png"] = _copy_p2_crop(best, fig_dir, out_dir)
            stats["cross_multi_signal"] += 1

    # p2 figures the vector pass did not corroborate.
    remaining: list[dict[str, Any]] = []
    for i, item in enumerate(p2_items):
        if i in used_p2:
            continue
        item = dict(item)
        item["confidence"] = CONF_FIG_DEGRADED
        item["signals"] = ["pdffigures2"]
        item["pdffigures2_only"] = True
        item["pdffigures2_png"] = _copy_p2_crop(item, fig_dir, out_dir)
        remaining.append(item)
        stats["cross_p2_only"] += 1

    stats["cross_degraded"] = len(remaining)

    kept = _dedupe(records + orphans + remaining)
    return kept, stats


def _copy_p2_crop(item: dict[str, Any], fig_dir: Path, out_dir: Path) -> str | None:
    """Copy a PDFFigures 2.0 render into the project's figure directory."""
    src = item.get("png_path")
    if not src:
        return None
    try:
        src_path = Path(src)
        if not src_path.exists():
            return None
        dst = fig_dir / f"p2_{item.get('figure_id', 'x')}.png"
        shutil.copyfile(src_path, dst)
        return str(dst.relative_to(out_dir))
    except Exception:
        return None


def _merge_pdffigures2(
    raw: list[dict[str, Any]], work_dir: Path, existing: list[dict[str, Any]], config: Config
) -> list[dict[str, Any]]:
    """Convert pdffigures2 entries not already found, flagging missing renders.

    PDFFigures 2.0 drops entries with no `renderURL` entirely, so a "suspect"
    item is invisible in its output. Rather than re-deriving it here, the count of
    entries it *did* emit is compared against the merged result and any shortfall
    is surfaced as a warning by the caller.
    """
    max_area_frac = config.get("figure_extraction.suspect_max_page_area_fraction", 0.85)
    min_area = config.get("figure_extraction.suspect_min_area_pt2", 3000)
    dpi = config.get("figure_extraction.dpi", 300)
    out: list[dict[str, Any]] = []

    for item in raw:
        if item.get("figType") != "Figure":
            continue
        render = item.get("renderURL")
        page = item.get("page")
        if not render or not page:
            out.append(
                {
                    "figure_id": f"p2_missing_{len(out)}",
                    "png_path": None,
                    "caption": item.get("caption"),
                    "page": page,
                    "bbox": None,
                    "related_clause": None,
                    "is_inside_table": False,
                    "confidence": "pdffigures2",
                    "detector": "pdffigures2",
                    "status": FIG_FAILED,
                    "status_reason": "pdffigures2 reported the figure but produced no renderURL",
                }
            )
            continue

        png = Path(str(render))
        status, reason = FIG_OK, ""
        if not png.exists():
            status, reason = FIG_FAILED, f"PNG listed in metadata but absent at {png.name}"
        bbox = item.get("figBoundary")
        bbox_t = _pdf_norm_bbox(bbox) if bbox else None
        if status == FIG_OK and bbox_t:
            area = (bbox_t[2] - bbox_t[0]) * (bbox_t[3] - bbox_t[1])
            if area < min_area or area > max_area_frac * 595 * 842:
                status, reason = FIG_SUSPECT, f"implausible area {area:.0f}pt2"

        out.append(
            {
                "figure_id": f"p{page}_p2_{len(out) + 1}",
                "figure_number": item.get("name"),
                "png_path": str(png),
                "caption": item.get("caption"),
                "page": int(page),
                "bbox": [round(v, 2) for v in bbox_t] if bbox_t else None,
                "graphic_bbox": None,
                "related_clause": None,
                "is_inside_table": False,
                # Always lower than caption-anchored: on codebooks this detector
                # was measured at 0% recall and low precision.
                "confidence": "pdffigures2_low",
                "detector": "pdffigures2",
                "status": status,
                "status_reason": reason,
            }
        )
    return out


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


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop pdffigures2 hits that overlap a caption-anchored figure.

    Caption-anchored wins ties, since it was measured far more reliable on
    codebooks and it crops the caption into the image.
    """
    order = {"pycaption": 0, "pdffigures2": 1}
    ranked = sorted(
        records, key=lambda r: (order.get(r.get("detector", ""), 9), r.get("status") == FIG_FAILED)
    )
    kept: list[dict[str, Any]] = []
    for r in ranked:
        bb = r.get("bbox")
        if bb:
            dup = False
            for k in kept:
                kbb = k.get("bbox")
                if kbb and k.get("page") == r.get("page") and _iou(tuple(bb), tuple(kbb)) >= 0.5:
                    dup = True
                    break
            if dup:
                continue
        kept.append(r)
    return kept