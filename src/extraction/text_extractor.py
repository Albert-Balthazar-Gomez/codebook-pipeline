"""Phase 2: hierarchical text extraction.

Design notes
------------
A naive full-page text dump is explicitly out of scope. This module instead:

1.  Finds and removes running headers/footers, which otherwise get spliced into
    the middle of clauses ("IS 456 : 2000" appearing inside clause text).
2.  Detects clause headings with a per-standard anchored regex, and records how
    each one was found so downstream phases can distrust the weak ones.
3.  Rebuilds the section tree from clause-number depth, cross-checked against
    heading font size.
4.  Orders text correctly on two-column pages, which codebooks frequently are.

Every clause carries `confidence` ∈ {regex, heuristic, ocr, preamble}. A clause
is never emitted with an invented number: if no anchored match occurred, the
boundary is recorded as heuristic and the number is left null.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pdfplumber

from ..common.config import Config
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

# Confidence values.
#   regex+font  - number matched AND font/bold/indent cues agree. Trusted.
#   regex_only  - number matched but the typographic cues did not confirm it.
#   heuristic   - boundary inferred without an anchored number match.
#   ambiguous   - regex and font cues actively disagree. Forces DEGRADED.
#   ocr         - text came from Phase 8.
#   preamble    - unnumbered front matter.
CONF_REGEX_FONT = "regex+font"
CONF_REGEX_ONLY = "regex_only"
CONF_HEURISTIC = "heuristic"
CONF_AMBIGUOUS = "ambiguous"
CONF_OCR = "ocr"
CONF_PREAMBLE = "preamble"
CONF_ORDER = (
    CONF_REGEX_FONT, CONF_REGEX_ONLY, CONF_HEURISTIC,
    CONF_AMBIGUOUS, CONF_OCR, CONF_PREAMBLE,
)


@dataclass
class Line:
    text: str
    top: float
    bottom: float
    x0: float
    x1: float
    page: int
    font_size: float = 0.0
    is_bold: bool = False
    column: int = 0
    method: str = CONF_REGEX_FONT
    # Indent bucket derived from x0, used as a third hierarchy cue.
    indent_level: int = 0
    # True when the line sits inside a ruled grid. Table text is emitted once by
    # Phase 3, so it must not also become clause headings or clause body: a cell
    # reading "0.36 0.43" otherwise parses as clause 0.36 titled "0.43", and the
    # same content ends up duplicated in clauses.jsonl and tables.jsonl.
    in_table: bool = False


def _is_bold(fontname: str | None) -> bool:
    if not fontname:
        return False
    f = fontname.lower()
    return "bold" in f or "black" in f or "heavy" in f


def _median(values: Iterable[float], default: float = 0.0) -> float:
    vals = [v for v in values if v and v > 0]
    return statistics.median(vals) if vals else default


# --------------------------------------------------------------------------- #
# Running headers / footers
# --------------------------------------------------------------------------- #
_DIGITS = re.compile(r"\d+")
_ALPHA = re.compile(r"[A-Za-z]")
# A line that is nothing but a clause number, e.g. "26.5.1.1" or "26.5.1.1.".
_PURE_CLAUSE_NO = re.compile(r"^\s*\u00a7?\s*\d+(?:\.\d+){0,4}\s*\.?\s*$")


def _normalise_head(text: str) -> str:
    """Collapse a running head to a page-invariant key.

    Page numbers are the reason raw comparison fails: "Page 12 of 90" and
    "Page 13 of 90" are the same running head. Every digit run becomes "#".
    """
    text = text.strip().lower()
    text = _DIGITS.sub("#", text)
    text = re.sub(r"[^a-z#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_running_heads(
    pdf: pdfplumber.PDF, header_band: float, footer_band: float, min_fraction: float
) -> tuple[set[str], dict[str, int]]:
    """Return normalised running-head strings plus their page counts."""
    header_hits: Counter[str] = Counter()
    footer_hits: Counter[str] = Counter()
    n_pages = len(pdf.pages)
    if n_pages == 0:
        return set(), {}

    for page in pdf.pages:
        h = page.height
        # Split on horizontal position too: a two-column page can put running
        # text in the outer margin at any x, so band membership alone is the
        # filter, not x.
        top_cut = h * header_band
        bot_cut = h * (1.0 - footer_band)
        for line in page.extract_text_lines(x_tolerance_ratio=0.12) or []:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            mid = (line["top"] + line["bottom"]) / 2
            if mid <= top_cut:
                header_hits[_normalise_head(text)] += 1
            elif mid >= bot_cut:
                footer_hits[_normalise_head(text)] += 1

    threshold = max(2, int(n_pages * min_fraction))
    heads = {k for k, c in header_hits.items() if c >= threshold}
    heads |= {k for k, c in footer_hits.items() if c >= threshold}
    # A string covering the whole page is not a running head, it is a box frame.
    heads = {h for h in heads if h}
    counts = {**dict(header_hits), **dict(footer_hits)}
    return heads, counts


def _is_page_number_only(text: str) -> bool:
    return bool(re.fullmatch(r"[\d\s.,\-–—/ofivxlc]*", text.strip(), re.IGNORECASE))


# --------------------------------------------------------------------------- #
# Column ordering
# --------------------------------------------------------------------------- #
def _detect_gutter(page: pdfplumber.page.Page) -> tuple[float, float] | None:
    """Find the two-column gutter of a codebook page.

    Three approaches were tried, and the failure modes are worth recording:

    * 60 coarse bins (trimitdown-pdf's scheme): IS 456's gutter is ~10-18pt on a
      571pt page, i.e. 1-2 bins, so it fell under the minimum run length and the
      page was read straight across the gutter.
    * Merged word intervals at exact resolution: the opposite failure. Any
      full-width element -- a spanning figure caption, a rule -- bridges the
      gutter, so the merged coverage had *no* gap at all and detection returned
      None on 8 of 114 pages.

    So: fine 2pt bins, and a run is accepted when coverage is merely *low*
    rather than exactly zero. The non-zero threshold is what lets a page survive
    its own full-width captions.
    """
    try:
        words = page.extract_words()
    except Exception:
        return None
    if len(words) < 60:
        return None
    page_x0, _, page_x1, _ = page.bbox
    width = page_x1 - page_x0
    if width <= 0:
        return None

    bin_w = 2.0
    nbins = max(8, int(width / bin_w))
    counts = [0] * nbins
    for w in words:
        a = max(0, min(nbins - 1, int((w["x0"] - page_x0) / width * nbins)))
        b = max(0, min(nbins - 1, int((w["x1"] - page_x0) / width * nbins)))
        for i in range(a, b + 1):
            counts[i] += 1

    # Allow a couple of full-width words to intrude into the gutter.
    threshold = max(0, int(0.005 * len(words)))
    lo, hi = 0.30, 0.70
    best_start, best_run = None, 0
    run = 0
    for i, c in enumerate(counts + [10**9]):  # sentinel closes a trailing run
        if c <= threshold:
            run += 1
            continue
        start = i - run
        if run:
            centre = (start + run / 2) / nbins
            if lo <= centre <= hi and run > best_run:
                best_run, best_start = run, start
        run = 0
    if best_start is None:
        return None

    gap_px = best_run / nbins * width
    if gap_px < 6.0:
        return None
    centre = (best_start + best_run / 2) / nbins * width + page_x0

    # Word mass on each side, measured the same way (summed widths).
    left_mass = sum(w["x1"] - w["x0"] for w in words if (w["x0"] + w["x1"]) / 2 < centre)
    right_mass = sum(w["x1"] - w["x0"] for w in words if (w["x0"] + w["x1"]) / 2 >= centre)
    total = left_mass + right_mass
    if not total:
        return None
    if left_mass < 0.20 * total or right_mass < 0.20 * total:
        return None
    return (centre - gap_px / 2, centre + gap_px / 2)


def _extract_page_lines(
    page: pdfplumber.page.Page, gutter: tuple[float, float] | None, x_tol_ratio: float
) -> list[tuple[dict, int]]:
    """Text lines with their column index.

    When a gutter exists, the page is filtered to each column and extracted
    separately. Sorting afterwards is not equivalent: `extract_text_lines` groups
    by vertical position across the full page width, so on a two-column page both
    columns' text has already fused into one string by the time it is returned and
    no amount of reordering can separate it again. Filtering first is the only way
    to stop the fused line from forming.
    """
    if gutter is None:
        return [(item, 0) for item in (page.extract_text_lines(x_tolerance_ratio=x_tol_ratio) or [])]

    gx0, gx1 = gutter
    mid = (gx0 + gx1) / 2

    def left(obj):
        return (obj["x0"] + obj["x1"]) / 2 < mid

    def right(obj):
        return (obj["x0"] + obj["x1"]) / 2 >= mid

    out: list[tuple[dict, int]] = []
    try:
        lp = page.filter(left)
        rp = page.filter(right)
    except Exception:
        return [(item, 0) for item in (page.extract_text_lines(x_tolerance_ratio=x_tol_ratio) or [])]
    for item in (lp.extract_text_lines(x_tolerance_ratio=x_tol_ratio) or []):
        out.append((item, 0))
    for item in (rp.extract_text_lines(x_tolerance_ratio=x_tol_ratio) or []):
        out.append((item, 1))
    return out


# --------------------------------------------------------------------------- #
# Line collection
# --------------------------------------------------------------------------- #
# A table-of-contents entry: clause number, title, trailing page number. The
# trailing page number is what distinguishes it from a real clause heading.
TOC_ENTRY = re.compile(r"^\s*\d+(?:\.\d+){0,4}\s+\S.*?\s+\d{1,4}\s*$")


def find_toc_pages(pdf: pdfplumber.PDF, min_ratio: float = 0.35) -> list[int]:
    """Pages dominated by table-of-contents entries.

    A codebook's contents pages look exactly like a clause list -- section
    numbers, titles, depths -- so without this the TOC is parsed as ~200 real
    depth-1 clauses and pollutes the whole tree. The discriminator is the
    trailing page number, which appears on TOC lines and essentially never on a
    clause heading.

    Measured on IS 456: contents pages score 0.55-0.82, body pages 0.00-0.02.
    """
    toc: list[int] = []
    for pno, page in enumerate(pdf.pages, start=1):
        lines = [
            (l.get("text") or "").strip()
            for l in (page.extract_text_lines(x_tolerance_ratio=0.12) or [])
        ]
        lines = [t for t in lines if t]
        if len(lines) < 4:
            continue
        hits = sum(1 for t in lines if TOC_ENTRY.match(t))
        if hits / len(lines) >= min_ratio:
            toc.append(pno)
    return toc


def detect_document_gutter(pdf: pdfplumber.PDF) -> tuple[float, float] | None:
    """Agree on one gutter for the whole document, from the per-page votes.

    A codebook is laid out consistently, so pages that carry a full-width figure
    or caption should not each re-derive the gutter and fail (their own content
    bridges the gap). Voting across the document and taking the median centre
    lets those pages inherit the layout the rest of the book agrees on, while a
    genuinely single-column book never reaches the consensus threshold and stays
    unsplit.
    """
    centres: list[float] = []
    gaps: list[float] = []
    for page in pdf.pages:
        g = _detect_gutter(page)
        if g is not None:
            centres.append((g[0] + g[1]) / 2)
            gaps.append(g[1] - g[0])
    if len(centres) < max(3, 0.20 * len(pdf.pages)):
        return None
    centres.sort()
    gaps.sort()
    centre = centres[len(centres) // 2]
    gap = max(6.0, gaps[len(gaps) // 2])
    return (centre - gap / 2, centre + gap / 2)


def _collect_lines(
    pdf: pdfplumber.PDF,
    running_heads: set[str],
    x_tol_ratio: float = 0.12,
    table_bboxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
    toc_pages: set[int] | None = None,
    caption_rx: list = None,
) -> tuple[list[Line], int]:
    """Return the page's lines plus how many pages had a column gutter."""
    lines: list[Line] = []
    columns_detected = 0
    n_caption_lines = 0
    doc_gutter = detect_document_gutter(pdf)
    for pno, page in enumerate(pdf.pages, start=1):
        if pno in (toc_pages or ()):
            continue
        # Per-page detection first; the document-level consensus covers pages
        # whose own full-width content would otherwise defeat detection.
        gutter = _detect_gutter(page)
        if gutter is None and doc_gutter is not None:
            words = page.extract_words() or []
            cx = doc_gutter[0] + (doc_gutter[1] - doc_gutter[0]) / 2
            if any((w["x0"] + w["x1"]) / 2 < cx for w in words) and any(
                (w["x0"] + w["x1"]) / 2 >= cx for w in words
            ):
                gutter = doc_gutter
        if gutter is not None:
            columns_detected += 1
        grids = (table_bboxes or {}).get(pno, [])
        raw = _extract_page_lines(page, gutter, x_tol_ratio)

        for item, column in raw:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            chars = item.get("chars") or []

            norm = _normalise_head(text)
            if norm in running_heads:
                continue
            # A bare page number in a margin band is furniture, not content.
            if _is_page_number_only(text) and len(text) <= 6:
                continue
            # Captions belong to Phase 4/3. Left in, a caption is duplicated into
            # the enclosing clause's body, which then reads as the clause
            # "referencing" its own figure -- a circular match that corrupts
            # Phase 5's caption-number signal.
            if caption_rx and any(rx.match(text) for rx in caption_rx):
                n_caption_lines += 1
                continue

            sizes = [c.get("size", 0) for c in chars if c.get("size")]
            lines.append(
                Line(
                    text=text,
                    top=float(item["top"]),
                    bottom=float(item["bottom"]),
                    x0=float(item["x0"]),
                    x1=float(item["x1"]),
                    page=pno,
                    font_size=_median(sizes),
                    is_bold=any(_is_bold(c.get("fontname")) for c in chars),
                    column=column,
                    in_table=_within_any(
                        (item["x0"] + item["x1"]) / 2,
                        (item["top"] + item["bottom"]) / 2,
                        grids,
                    ),
                )
            )

    lines.sort(key=lambda l: (l.page, l.column, l.top, l.x0))
    return lines, columns_detected, n_caption_lines


def _within_any(
    cx: float,
    cy: float,
    boxes: list[tuple[float, float, float, float]],
    pad: float = 2.0,
) -> bool:
    """Centre-point containment, matching pdfplumber's own cell assignment."""
    for x0, top, x1, bottom in boxes:
        if (x0 - pad) <= cx <= (x1 + pad) and (top - pad) <= cy <= (bottom + pad):
            return True
    return False


def find_ruled_grid_bboxes(
    pdf: pdfplumber.PDF, vertical_strategy: str = "lines", horizontal_strategy: str = "lines"
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Ruled-grid bboxes per page, used as exclusion zones.

    Every ruled grid counts, including ones Phase 3 will later reject with
    `is_real_table`: a layout frame's text is still layout, not prose, so it must
    not be promoted into clause headings either.
    """
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    settings = {"vertical_strategy": vertical_strategy, "horizontal_strategy": horizontal_strategy}
    for pno, page in enumerate(pdf.pages, start=1):
        try:
            grids = page.find_tables(settings)
        except Exception as exc:
            log.debug("find_tables failed on page %d: %s", pno, exc)
            grids = []
        if grids:
            out[pno] = [tuple(float(v) for v in g.bbox) for g in grids]
    return out


def _reading_order(lines: list[Line]) -> list[Line]:
    """Sort into reading order: column-major, then top-to-bottom.

    Sorting purely by `top` reads a two-column page across the gutter and fuses
    unrelated sentences, so column is the primary key.
    """
    return sorted(lines, key=lambda l: (l.page, l.column, l.top, l.x0))


# --------------------------------------------------------------------------- #
# Clause segmentation
# --------------------------------------------------------------------------- #
def _dehyphenate(parts: list[str]) -> str:
    """Join wrapped lines, undoing end-of-line hyphenation."""
    out: list[str] = []
    for i, part in enumerate(parts):
        part = part.rstrip()
        if not part:
            continue
        if out and out[-1].endswith("-"):
            # "interact-" + "ing of" -> "interacting of"
            if part[:1].islower():
                out[-1] = out[-1][:-1] + part
                continue
        out.append(part)
    text = " ".join(out)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


@dataclass
class _Heading:
    number: str | None
    depth: int
    title: str
    confidence: str
    line: Line
    reason: str = ""
    # Individual cue verdicts, kept so a disagreement can be reported rather
    # than silently resolved.
    cue_regex: bool = False
    cue_font: bool = False
    cue_indent: bool = False


# Strongest first. Used to decide which record wins when a clause number occurs
# more than once.
_CONF_RANK = {
    CONF_REGEX_FONT: 5,
    CONF_REGEX_ONLY: 4,
    CONF_HEURISTIC: 3,
    CONF_AMBIGUOUS: 2,
    CONF_OCR: 1,
    CONF_PREAMBLE: 0,
}


def _conf_rank(confidence: str) -> int:
    return _CONF_RANK.get(confidence, 0)


def _indent_levels(lines: list[Line], bucket_pt: float) -> dict[int, int]:
    """Map each distinct x0 to an indent level.

    Codebooks indent clause headings with nesting depth, so x0 carries the same
    information as the clause number's depth. Levels are relative to the most
    common x0, so a document with a wide left margin is handled the same as one
    flush to the edge.
    """
    if not lines:
        return {}
    counts: Counter[float] = Counter(round(l.x0 / bucket_pt) for l in lines)
    base = counts.most_common(1)[0][0]
    return {l.x0: max(0, round(l.x0 / bucket_pt) - base) for l in lines}


def _match_heading(
    line: Line,
    patterns: dict[str, re.Pattern[str]],
    max_depth: int,
    max_heading_chars: int,
    body_font: float,
    max_number: int = 999,
    min_font_ratio: float = 0.85,
    font_advantage_pt: float = 0.5,
) -> _Heading | None:
    """Decide whether a line opens a new clause, from three independent cues.

    Three guards reject things that merely *look* like headings:

    * Body prose can start with a clause number ("26.5.1 The reinforcement
      shall be..."), so an anchored regex match alone is not enough.
    * Clause numbers have a plausible range. IS 456 has sections 0-23, so "415",
      "900" and "IS 2116" are values and standard references, not clauses.
    * Footnotes and marginalia are set smaller than body text.

    The three cues (regex, font/bold, indentation) are recorded separately so
    that when regex and typography disagree the clause is labelled `ambiguous`
    instead of being quietly filed as fine.
    """
    text = line.text.strip()
    # A bare clause number is a legitimate heading ("26.5.1.1" alone on a line),
    # so the no-letters guard must exempt it.
    if not _ALPHA.search(text) and not _PURE_CLAUSE_NO.match(text):
        return None
    if body_font and line.font_size and line.font_size < body_font * min_font_ratio:
        return None  # footnote / marginal note

    for key in ("pattern", "pattern_number_only"):
        rx = patterns.get(key)
        if not rx:
            continue
        m = rx.match(text)
        if not m:
            continue
        number = m.group("number").strip().rstrip(".")
        leading = number.split(".")[0]
        if leading.isdigit() and int(leading) > max_number:
            return None
        title = text[m.end():].strip()
        depth = number.count(".") + 1
        if depth > max_depth:
            return None

        longer = len(title) > max_heading_chars
        cue_font = line.is_bold or (body_font and line.font_size >= body_font + font_advantage_pt)
        if longer and not cue_font:
            # Number is inline prose, not a boundary.
            return None

        # Indentation cue: headings normally sit further left than their body,
        # so a heading that is indented *more* than the surrounding body text
        # contradicts the typography.
        cue_indent = line.indent_level == 0

        if not cue_font:
            conf = CONF_REGEX_ONLY
            reason = "clause number matched but font is neither bold nor larger than body"
        elif longer and cue_font:
            conf = CONF_AMBIGUOUS
            reason = "font says heading but the title runs on as body text"
        else:
            conf = CONF_REGEX_FONT
            reason = "clause number matched and font/bold confirms a heading"
        if conf == CONF_REGEX_FONT and not cue_indent:
            reason += "; heading is indented, which is unusual"

        return _Heading(number, depth, title, conf, line, reason, True, cue_font, cue_indent)
    return None


def _build_tree(headings: list[_Heading]) -> list[dict[str, Any]]:
    """Assemble the clause tree from number depth.

    A clause's parent is the nearest preceding heading exactly one level
    shallower. If none exists (a document that starts mid-tree, or a scanned
    jump), the nearest preceding shallower heading is used instead and the
    clause is flagged, because a wrong parent_id silently corrupts every
    downstream chunk.
    """
    clauses: list[dict[str, Any]] = []
    stack: list[_Heading] = []  # open ancestors, shallowest first

    for h in headings:
        while stack and stack[-1].depth >= h.depth:
            stack.pop()

        parent = None
        parent_exact = True
        if stack:
            # Nearest open ancestor shallower than this clause.
            shallow = [a for a in stack if a.depth < h.depth]
            if shallow:
                cand = shallow[-1]
                if cand.depth == h.depth - 1:
                    parent = cand
                else:
                    parent = cand
                    parent_exact = False
        elif clauses:
            # Nothing open: fall back to the nearest preceding shallower clause.
            shallower = [c for c in reversed(clauses) if c["depth"] < h.depth]
            if shallower:
                parent = shallower[-1]
                parent_exact = False

        confidence = h.confidence
        if parent is not None and not parent_exact and confidence == CONF_REGEX_FONT:
            # The number is real but the tree position is inferred.
            confidence = CONF_REGEX_ONLY

        record = {
            "number": h.number,
            "depth": h.depth,
            "title": h.title,
            "confidence": confidence,
            "confidence_reason": h.reason + ("" if parent_exact else "; parent inferred (no exact-depth ancestor)"),
            "parent": parent,
            "heading_line": h.line,
        }
        clauses.append(record)
        stack.append(h)
    return clauses


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def audit_text_damage(lines: list[Line]) -> dict[str, Any]:
    """Measure evidence that the text layer is OCR-damaged.

    Some PDFs present a text layer that exists but is unusable: IS 456:2000 as
    distributed by most mirrors was produced by `itextpdf`, an OCR tool, and lost
    word spacing on a large fraction of tokens -- `DurabUity`, `Shapeand Size`,
    `InthedesipotionofconcretemixMreferstothemixlUldthe`. No x_tolerance setting
    can repair that, because the gap between the words is simply absent.

    It still counts towards the 50-character "has a text layer" test, so without
    this audit such a file is silently treated as clean and the damage only
    surfaces as mysteriously bad embeddings and unusable training data.

    Two signals are counted:
      damaged_internal_cap - a lowercase letter followed by an uppercase one
                             ("DurabUity"), the classic OCR mis-case artefact
      glued_long_token   - an alphabetic token long enough to be several words
    """
    internal_cap = re.compile(r"[a-z][A-Z]")
    toks: list[str] = []
    for line in lines:
        toks.extend(line.text.split())

    if not toks:
        return {"tokens": 0, "damaged_internal_cap": 0, "glued_long_token": 0,
                "damaged_pct": 0.0, "glued_pct": 0.0, "examples": []}

    caps = [t for t in toks if internal_cap.search(t)]
    glued = [t for t in toks if len(t) >= 22 and t.isalpha()]
    examples = sorted(set(caps + glued), key=len, reverse=True)[:8]
    return {
        "tokens": len(toks),
        "damaged_internal_cap": len(caps),
        "glued_long_token": len(glued),
        "damaged_pct": round(100.0 * len(caps) / len(toks), 2),
        "glued_pct": round(100.0 * len(glued) / len(toks), 2),
        "examples": examples,
    }


def extract_clauses(
    pdf_path: Path | str,
    config: Config,
    out_dir: Path | str,
    method: str = CONF_REGEX_FONT,
    pages: tuple[int, int] | None = None,
) -> tuple[PhaseResult, list[dict[str, Any]]]:
    """Extract the clause tree. Returns a PhaseResult and the clause records."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    heading_band = config.get("text_extraction.header_band", 0.10)
    footer_band = config.get("text_extraction.footer_band", 0.10)
    min_frac = config.get("text_extraction.running_head_min_page_fraction", 0.6)
    min_chars = config.get("text_extraction.min_clause_chars", 2)
    max_heading = config.get("text_extraction.max_heading_chars", 120)
    min_font_ratio = config.get("text_extraction.min_heading_font_ratio", 0.85)
    standard = config.get("codebook.standard", "is456")
    patterns = config.clause_regexes(standard)
    max_depth = config.standard_cfg(standard).get("clause", {}).get("max_depth", 4)
    max_number = config.standard_cfg(standard).get("clause", {}).get("max_number", 999)
    font_advantage_pt = config.get("text_extraction.heading_font_advantage_pt", 0.5)
    indent_bucket = config.get("text_extraction.indent_bucket_pt", 8.0)
    indent_min_cov = config.get("text_extraction.indent_min_coverage", 0.30)

    result = PhaseResult(phase="phase2_text", status=Status.SUCCESS)

    with pdfplumber.open(pdf_path) as pdf:
        n_pages_total = len(pdf.pages)
        result.counts["pages"] = n_pages_total

        # Phase 8 hook: pages with almost no characters are image-only.
        empty_pages: list[int] = []
        for pno, page in enumerate(pdf.pages, start=1):
            chars = len(page.chars)
            if chars < config.get("ocr.no_text_layer_char_threshold", 50):
                text = (page.extract_text() or "").strip()
                if len(text) < config.get("ocr.no_text_layer_char_threshold", 50):
                    empty_pages.append(pno)
        if empty_pages:
            result.counts["no_text_layer_pages"] = len(empty_pages)
            result.degrade(
                f"{len(empty_pages)} page(s) have no text layer "
                f"(pages {empty_pages[:20]}{'...' if len(empty_pages) > 20 else ''}); "
                "run Phase 8 OCR to recover them"
            )

        target_pages = None
        if pages:
            lo, hi = pages
            target_pages = set(range(lo, hi + 1))
            result.counts["page_range"] = [lo, hi]

        running_heads, head_counts = detect_running_heads(pdf, heading_band, footer_band, min_frac)
        result.counts["running_heads_detected"] = len(running_heads)

        toc_pages = set(find_toc_pages(pdf))
        if toc_pages:
            result.counts["toc_pages_excluded"] = len(toc_pages)
            result.degrade(
                f"table-of-contents pages {sorted(toc_pages)} excluded from clause "
                "detection; their entries end in a page number and would otherwise "
                "parse as ~200 depth-1 clauses"
            )

        # Exclusion zones: every ruled grid, so table text never becomes a clause.
        grids = find_ruled_grid_bboxes(
            pdf,
            config.get("table_extraction.vertical_strategy", "lines"),
            config.get("table_extraction.horizontal_strategy", "lines"),
        )
        result.counts["ruled_grid_bboxes"] = sum(len(v) for v in grids.values())

        caption_rx = [patterns[k] for k in ("figure_pattern", "table_pattern") if k in patterns]
        lines, n_two_col, n_caps = _collect_lines(
            pdf, running_heads, table_bboxes=grids, toc_pages=toc_pages,
            caption_rx=caption_rx,
        )
        result.counts["caption_lines_excluded"] = n_caps
        if target_pages is not None:
            lines = [l for l in lines if l.page in target_pages]
        result.counts["lines_kept"] = len(lines)
        result.counts["lines_in_tables"] = sum(1 for l in lines if l.in_table)
        result.counts["two_column_pages"] = n_two_col

        if not lines:
            result.status = Status.FAILED
            result.error = "no extractable text after removing running heads"
            return result, []

        # Body font is the page-level median; headings must exceed it.
        body_font = _median(l.font_size for l in lines if not l.is_bold)
        result.counts["body_font_size"] = round(body_font, 2)

        ordered = _reading_order(lines)

        # Pass C cue 3: indentation. Stamped onto every line so the heading
        # matcher can consult it. Only trusted when headings actually occupy
        # several indent levels -- a flat document would otherwise "confirm"
        # every heading at level 0.
        indents = _indent_levels(ordered, indent_bucket)
        distinct_indents = {v for v in indents.values()}
        indent_trusted = len(distinct_indents) >= 2
        result.counts["indent_levels"] = sorted(distinct_indents)
        result.counts["indent_cue_trusted"] = indent_trusted
        if indent_trusted:
            for ln in ordered:
                ln.indent_level = indents.get(ln.x0, 0)
        else:
            for ln in ordered:
                ln.indent_level = 0

        # ---- segment into headings + body runs -------------------------- #
        headings: list[_Heading] = []
        runs: list[tuple[int, list[Line]]] = []  # (page, lines) body text
        current: list[Line] = []
        current_page: int | None = None

        def flush() -> None:
            if current:
                runs.append((current_page or current[0].page, list(current)))
                current.clear()

        for line in ordered:
            # Table interiors are Phase 3's responsibility. They are neither
            # clause boundaries nor clause prose.
            if line.in_table:
                continue
            h = _match_heading(
                line, patterns, max_depth, max_heading, body_font,
                max_number=max_number, min_font_ratio=min_font_ratio,
                font_advantage_pt=font_advantage_pt,
            )
            if h is not None:
                flush()
                headings.append(h)
                current_page = line.page
                continue
            if current_page is None:
                current_page = line.page
            elif line.page != current_page:
                # A page break mid-clause: start a new run but keep the clause
                # open so the text concatenates rather than splitting.
                flush()
                current_page = line.page
            current.append(line)
        flush()

        if not headings:
            result.status = Status.FAILED
            result.error = (
                "no clause headings matched the standard's regex; "
                "the document may be scanned, or codebook.standard may be wrong"
            )
            result.counts["lines_without_headings"] = len(runs)
            return result, []

        tree = _build_tree(headings)

        # Attach body text to each clause. Body runs are consumed in order, so
        # each run belongs to the heading that most recently preceded it.
        body_by_heading: dict[int, list[str]] = defaultdict(list)
        run_idx = 0
        for h_i, h in enumerate(headings):
            # Consume runs until we reach one on a page/position at or after
            # this heading's line.
            start = h.line
            while run_idx < len(runs):
                r_page, r_lines = runs[run_idx]
                first = r_lines[0]
                is_after = (r_page, first.top) >= (start.page, start.top)
                if not is_after:
                    run_idx += 1
                    continue
                body_by_heading[h_i].extend(l.text for l in r_lines)
                run_idx += 1
                break

        # Any leading run before the first heading is front matter.
        preamble = ""
        if runs and headings:
            first_h = headings[0]
            lead = [l for (p, ls) in runs for l in ls if (p, l.top) < (first_h.line.page, first_h.line.top)]
            if lead:
                preamble = _dehyphenate([l.text for l in lead])

        # ---- materialise records ---------------------------------------- #
        by_number: dict[int, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []
        for i, entry in enumerate(tree):
            h = headings[i]
            parent = entry["parent"]
            title = entry["title"]
            body = _dehyphenate(body_by_heading.get(i, []))
            full = f"{h.number} {title}".strip() if h.number else title
            if body:
                full = f"{full}\n{body}" if full else body

            if len(full.strip()) < min_chars and not title:
                continue

            clause_id = h.number or f"heuristic.p{start_page_of(h.line)}.{i}"
            record = {
                "clause_id": clause_id,
                "parent_id": parent.number if parent is not None else None,
                "number": h.number,
                "depth": entry["depth"],
                "title": title,
                "text": full,
                "body": body,
                "page": h.line.page,
                "confidence": CONF_OCR if method == CONF_OCR else entry["confidence"],
                "confidence_reason": entry["confidence_reason"],
                "standard": standard,
                "bbox": [round(h.line.x0, 2), round(h.line.top, 2), round(h.line.x1, 2), round(h.line.bottom, 2)],
                "heading_font_size": round(h.line.font_size, 2),
                "heading_is_bold": h.line.is_bold,
                "word_count": len(full.split()),
                "extraction_method": method,
            }
            by_number[i] = record
            records.append(record)

        if preamble.strip():
            records.insert(
                0,
                {
                    "clause_id": "preamble",
                    "parent_id": None,
                    "number": None,
                    "depth": 0,
                    "title": "Front matter",
                    "text": preamble,
                    "body": preamble,
                    "page": records[0]["page"] if records else 1,
                    "confidence": CONF_PREAMBLE,
                    "confidence_reason": "text preceding the first numbered clause",
                    "standard": standard,
                    "bbox": None,
                    "heading_font_size": 0.0,
                    "heading_is_bold": False,
                    "word_count": len(preamble.split()),
                    "extraction_method": method,
                },
            )

        # ---- report ------------------------------------------------------- #
        depths = Counter(r["depth"] for r in records)
        confs = Counter(r["confidence"] for r in records)
        result.counts["clauses"] = len(records)
        result.counts["by_depth"] = dict(sorted(depths.items()))
        result.counts["by_confidence"] = dict(confs)
        result.counts["max_depth_seen"] = max(depths) if depths else 0
        result.counts["orphans"] = sum(1 for r in records if r["parent_id"] is None and r["depth"] > 1)
        result.counts["word_count"] = sum(r["word_count"] for r in records)

        if confs.get(CONF_AMBIGUOUS, 0):
            # Regex and typography actively disagree: the phase cannot claim to
            # have identified these confidently.
            result.degrade(
                f"{confs[CONF_AMBIGUOUS]} clause(s) are flagged 'ambiguous': the "
                "clause number matched but the typography contradicts it (font says "
                "heading, layout says body text). Their titles and boundaries need "
                "manual review"
            )
        if confs.get(CONF_REGEX_ONLY, 0):
            result.warn(
                f"{confs[CONF_REGEX_ONLY]} clause(s) matched the regex but were not "
                "typographically confirmed; treat their boundary and parent_id as inferred"
            )
        if result.counts["orphans"]:
            result.warn(
                f"{result.counts['orphans']} sub-clause(s) have no resolvable parent "
                "(document may start mid-tree, or the standard's regex does not match)"
            )
        if depths and min(depths) > 1:
            result.warn(
                f"no depth-1 clauses found (minimum depth {min(depths)}); "
                "check codebook.standard"
            )

        # ---- duplicate clause numbers ------------------------------------- #
        # Two records sharing a clause_id silently corrupt every downstream join
        # (chunks, links, vector payloads all key on it), so this is an error,
        # not a warning.
        seen: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        for rec in records:
            cid = rec["clause_id"]
            if cid in seen:
                duplicates.append(cid)
                # Keep the more trustworthy record.
                if _conf_rank(rec["confidence"]) > _conf_rank(seen[cid]["confidence"]):
                    idx = records.index(seen[cid])
                    records[idx] = rec
                    seen[cid] = rec
            else:
                seen[cid] = rec
        if duplicates:
            result.warn(
                f"{len(set(duplicates))} clause number(s) occur more than once "
                f"({', '.join(sorted(set(duplicates))[:8])}); the highest-confidence "
                "record wins. Usually OCR damage splitting a number, e.g. '26.1' read "
                "as '26' + '1'."
            )
        result.counts["duplicate_clause_ids"] = len(set(duplicates))

        # ---- text-layer damage audit --------------------------------------- #
        damage = audit_text_damage(ordered)
        result.counts["text_damage"] = damage
        if damage["tokens"] and damage["damaged_pct"] >= 0.5:
            result.degrade(
                f"text layer appears OCR-damaged: {damage['damaged_pct']}% of "
                f"{damage['tokens']} tokens have an internal capital (e.g. "
                f"{', '.join(repr(e) for e in damage['examples'][:3])}) and "
                f"{damage['glued_pct']}% are 22+ character runs. Word spacing is "
                "missing from the PDF itself and cannot be repaired by tolerance "
                "settings, so embeddings and generated Q/A will inherit the damage. "
                "Re-OCR from the page images (Phase 8) for a usable dataset."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    from ..common.jsonl import write_jsonl

    write_jsonl(out_dir / "text" / "clauses.jsonl", records)
    result.counts["output"] = str(out_dir / "text" / "clauses.jsonl")
    return result, records


def start_page_of(line: Line) -> int:
    return line.page