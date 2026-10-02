"""Phase 5a: clause <-> figure <-> table linking.

A figure or table is attached to the nearest clause heading that *precedes* it.
Direction matters: a figure belongs to the clause whose text introduces it, not
to whatever clause happens to come next.

Every link records the vertical gap it was decided on. That number is what makes
the linking auditable -- a 12pt gap and a 700pt gap are both "linked", but only
the first is trustworthy, and the spot-check in the validation step reads it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

# Vertical gaps (pt) deciding link confidence.
GAP_TIGHT = 150.0
GAP_LOOSE = 600.0


def _centre(bbox: list[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _contains(outer: list[float], inner_centre: tuple[float, float], pad: float = 2.0) -> bool:
    return (
        outer[0] - pad <= inner_centre[0] <= outer[2] + pad
        and outer[1] - pad <= inner_centre[1] <= outer[3] + pad
    )


def _find_clause(
    clauses: list[dict[str, Any]], page: int, top: float
) -> tuple[dict[str, Any] | None, float, str]:
    """Nearest preceding clause heading, with the gap and the rule that found it.

    Three tiers, tried in order:
      1. heading on the same page above the item
      2. last heading on an earlier page (item sits above the first heading of
         its own page, e.g. a figure in a wide margin at the top)
      3. nothing
    """
    best: dict[str, Any] | None = None
    best_gap = float("inf")
    for c in clauses:
        bbox = c.get("bbox")
        if not bbox or c.get("page") != page:
            continue
        if bbox[1] <= top:
            gap = top - bbox[3]
            if gap < best_gap:
                best, best_gap = c, gap
    if best is not None:
        return best, best_gap, "same_page"

    # No heading above on this page: use the latest heading before this page.
    earlier = [c for c in clauses if c.get("bbox") and c.get("page", 0) < page]
    if earlier:
        last = max(earlier, key=lambda c: (c["page"], c["bbox"][1]))
        return last, float("inf"), "previous_page"

    return None, float("inf"), "none"


def _clause_by_reference(fig, clauses, ordered):
    """Resolve a clause from an in-text cross-reference such as "in Fig. 7".

    A figure numbered to match a clause number ("Fig. 5" / clause 30.2.1) is not
    directly derivable, so this looks for the figure number quoted in a clause
    body that sits near the figure. It is deliberately conservative: it returns
    None unless the reference is unambiguous.
    """
    number = fig.get("figure_number")
    if not number:
        return None
    ref = str(number).strip().rstrip(".")
    if not ref:
        return None
    pat = re.compile(r"(?i)\bfig(?:ure)?\.?\s*[-–—]?\s*" + re.escape(ref) + r"\b")
    page = fig.get("page") or 0
    # Look on the figure's own page and the one before it: the introducing text
    # is at or just before the figure.
    window = [c for c in clauses if page - 1 <= (c.get("page") or 0) <= page]
    hits = [c for c in window if pat.search(c.get("body") or "") or pat.search(c.get("title") or "")]
    if len(hits) == 1:
        return hits[0]
    return None


def link_items(
    clauses: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    result: PhaseResult | None = None,
) -> tuple[PhaseResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach related_clause to every figure and clause_ref to every table."""
    result = result or PhaseResult(phase="phase5_linking", status=Status.SUCCESS)
    ordered = sorted(
        [c for c in clauses if c.get("bbox")],
        key=lambda c: (c.get("page", 0), c["bbox"][1]),
    )
    # Keyed by page: a bbox on page 2 must not be tested against a table on page 1,
    # which would mark every figure as "inside a table".
    table_boxes_by_page: dict[int, list[list[float]]] = {}
    for t in tables:
        if t.get("bbox"):
            table_boxes_by_page.setdefault(t.get("page") or 0, []).append(t["bbox"])

    # ---- figures: three independent signals, then agreement --------------- #
    for fig in figures:
        bbox = fig.get("bbox")
        centre = _centre(bbox)
        fig["is_inside_table"] = bool(
            centre
            and any(_contains(b, centre) for b in table_boxes_by_page.get(fig.get("page") or 0, []))
        )
        page = fig.get("page") or 0
        top = bbox[1] if bbox else 0.0

        signals: dict[str, Any] = {}

        # Signal 1 -- caption number. The document's own "Fig. 7" cross-reference
        # is the strongest evidence available when it can be resolved.
        signals["caption_number"] = _clause_by_reference(fig, clauses, ordered)

        # Signal 2 -- geometric: nearest preceding heading on the page.
        geo_clause, gap, rule = _find_clause(ordered, page, top)
        signals["proximity"] = geo_clause
        signals["gap_pt"] = None if gap == float("inf") else round(gap, 1)
        signals["rule"] = rule

        # Signal 3 -- inside a table (a figure there is almost always a diagram
        # drawn within a table cell and belongs to that table's clause).
        signals["inside_table"] = fig["is_inside_table"]

        candidates = {
            c["clause_id"]: c
            for c in (signals["caption_number"], geo_clause)
            if c is not None
        }

        if not candidates:
            fig["related_clause"] = None
            fig["link_confidence"] = "unlinked"
            fig["link_rule"] = "none"
            fig["link_gap_pt"] = None
            fig["link_signals"] = signals
            continue

        if len(candidates) == 1:
            clause = next(iter(candidates.values()))
            agree = True
        else:
            # Signals disagree: the caption points somewhere the geometry does
            # not support. Report it rather than silently preferring one.
            agree = False
            # Prefer the caption when it is backed by a nearby heading; otherwise
            # the geometric candidate, which at least is on the right page.
            clause = signals["caption_number"] or geo_clause

        fig["related_clause"] = clause["clause_id"]
        fig["link_gap_pt"] = signals["gap_pt"]
        fig["link_rule"] = signals["rule"]
        fig["signals_agree"] = agree
        fig["link_signals"] = signals

        if not agree:
            fig["link_confidence"] = "degraded"
        elif gap == float("inf"):
            fig["link_confidence"] = "low"
        elif gap <= GAP_TIGHT:
            fig["link_confidence"] = "high"
        elif gap <= GAP_LOOSE:
            fig["link_confidence"] = "medium"
        else:
            fig["link_confidence"] = "low"

    # ---- tables ------------------------------------------------------------- #
    for tab in tables:
        bbox = tab.get("bbox")
        page = tab.get("page") or 0
        top = bbox[1] if bbox else 0.0
        clause, gap, rule = _find_clause(ordered, page, top)
        if clause is None:
            tab["clause_ref"] = None
            tab["link_confidence"] = "unlinked"
            tab["link_rule"] = "none"
            continue
        tab["clause_ref"] = clause["clause_id"]
        tab["link_rule"] = rule
        tab["link_gap_pt"] = None if gap == float("inf") else round(gap, 1)
        tab["link_confidence"] = (
            "high" if gap <= GAP_TIGHT else ("medium" if gap <= GAP_LOOSE else "low")
        )
        if rule == "previous_page":
            tab["link_confidence"] = "low"

    figs_conf = [f.get("link_confidence") for f in figures]
    tabs_conf = [t.get("link_confidence") for t in tables]
    result.counts["figures"] = len(figures)
    result.counts["tables"] = len(tables)
    result.counts["fig_link_high"] = figs_conf.count("high")
    result.counts["fig_link_medium"] = figs_conf.count("medium")
    result.counts["fig_link_low"] = figs_conf.count("low")
    result.counts["fig_link_unlinked"] = figs_conf.count("unlinked")
    result.counts["tab_link_high"] = tabs_conf.count("high")
    result.counts["tab_link_medium"] = tabs_conf.count("medium")
    result.counts["tab_link_low"] = tabs_conf.count("low")
    result.counts["tab_link_unlinked"] = tabs_conf.count("unlinked")
    result.counts["fig_inside_table"] = sum(1 for f in figures if f.get("is_inside_table"))

    weak = [f for f in figures if f.get("link_confidence") in ("low", "unlinked")]
    weak_t = [t for t in tables if t.get("link_confidence") in ("low", "unlinked")]
    if weak:
        result.warn(
            f"{len(weak)} figure(s) linked with low/unlinked confidence: "
            + ", ".join(f"{f['figure_id']}({f.get('link_confidence')})" for f in weak[:10])
        )
    if weak_t:
        result.warn(
            f"{len(weak_t)} table(s) linked with low/unlinked confidence: "
            + ", ".join(f"{t['table_id']}({t.get('link_confidence')})" for t in weak_t[:10])
        )
    if any(f.get("link_rule") == "previous_page" for f in figures):
        result.degrade(
            "some figures sit above the first clause heading on their page and were "
            "linked to a heading on an earlier page"
        )
    return result, figures, tables


def spot_check_samples(
    figures: list[dict[str, Any]], clauses: list[dict[str, Any]], n: int = 10
) -> list[dict[str, Any]]:
    """Evenly spaced sample of figures, for manual verification.

    Spacing rather than "first N" so the sample spans the whole document instead
    of whichever page happened to extract best.
    """
    if not figures:
        return []
    step = max(1, len(figures) // n)
    picked = figures[::step][:n]
    by_id = {c["clause_id"]: c for c in clauses}
    out = []
    for f in picked:
        cid = f.get("related_clause")
        clause = by_id.get(cid)
        out.append(
            {
                "figure_id": f["figure_id"],
                "page": f.get("page"),
                "caption": f.get("caption"),
                "related_clause": cid,
                "linked_clause_title": clause.get("title") if clause else None,
                "linked_clause_page": clause.get("page") if clause else None,
                "link_gap_pt": f.get("link_gap_pt"),
                "link_confidence": f.get("link_confidence"),
            }
        )
    return out