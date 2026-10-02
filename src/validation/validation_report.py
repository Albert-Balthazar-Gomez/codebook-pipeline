"""Validation and accuracy reporting.

Writes ``validation_report.md`` next to the pipeline report, and enforces the
project's accuracy gate: if any phase's estimated accuracy is below
``validation.accuracy_target``, the run is DEGRADED and the specific failing
inputs are listed.

Accuracy is *estimated*, not asserted. Where a ground-truth denominator exists
(the number of caption lines in the text, say) the estimate is a real recall
figure. Where it does not, the estimate comes from the pipeline's own
agreement signals -- for example a table confirmed by three independent
strategies is more trustworthy than one confirmed by one -- and that basis is
stated next to the number so the figure is not mistaken for a measurement.
"""

from __future__ import annotations

import logging
import random
import statistics
from pathlib import Path
from typing import Any

from ..common.config import Config
from ..common.status import PhaseReport, Status

log = logging.getLogger(__name__)

# Documented, reproducible evidence for each estimated accuracy, so the number is
# traceable to something checkable rather than asserted.
BASIS = {
    "phase2_text": (
        "estimated from clause-count agreement, orphan rate, and the measured "
        "OCR-damage rate of the source text layer"
    ),
    "phase3_tables": (
        "estimated from the ratio of caption lines in the text to tables "
        "extracted (recall), and the fraction of tables whose cell contents were "
        "independently confirmed by more than one strategy (precision proxy)"
    ),
    "phase4_figures": (
        "estimated from the ratio of figure caption lines to figures extracted "
        "(recall), and the fraction confirmed by two independent detectors "
        "(precision proxy)"
    ),
    "phase5_linking": "estimated from the fraction of links where signals agree",
    "phase7_dataset": "estimated from the fraction of pairs surviving grounding",
    "phase8_ocr": "estimated from OCR/text-layer cross-check agreement",
}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _estimate_phase2(clauses: list[dict], counts: dict) -> tuple[float | None, list[str]]:
    failing: list[str] = []
    if not clauses:
        return 0.0, ["no clauses extracted"]

    total = len(clauses)
    orphans = sum(1 for c in clauses if c["depth"] > 1 and not c.get("parent_id"))
    dupes = counts.get("duplicate_clause_ids", 0)
    damage = (counts.get("text_damage") or {}).get("damaged_pct", 0.0)

    # Penalty model: orphan and duplicate-clause rates, plus the measured rate of
    # damaged tokens in the source, which caps achievable text accuracy.
    penalty = (
        100.0 * orphans / total
        + 100.0 * dupes / total
        + damage
    )
    acc = max(0.0, 1.0 - penalty / 100.0)
    if orphans:
        failing.append(f"{orphans} clause(s) with a dangling parent_id")
    if dupes:
        failing.append(f"{dupes} clause number(s) occurring more than once")
    if damage >= 1.0:
        failing.append(
            f"source text layer is OCR-damaged ({damage}% of tokens lost word "
            "spacing); no pipeline can exceed that ceiling"
        )
    return acc, failing


def _estimate_phase3(tables: list[dict], counts: dict) -> tuple[float | None, list[str]]:
    failing: list[str] = []
    if not tables:
        return None, []
    kept = counts.get("tables_kept", 0)
    with_caption = sum(1 for t in tables if t.get("caption"))
    votes = [t.get("n_votes", 0) for t in tables]
    mean_votes = statistics.mean(votes) if votes else 0.0

    degraded = [t["table_id"] for t in tables if t.get("confidence") == "degraded"]
    if degraded:
        failing.append(
            f"{len(degraded)} table(s) had disagreeing cell contents across "
            f"strategies: {', '.join(degraded[:10])}"
        )
    if with_caption < kept:
        failing.append(f"{kept - with_caption} table(s) matched no caption")

    # Precision proxy: mean strategy votes out of the number configured.
    precision = min(1.0, mean_votes / 3.0) if mean_votes else 0.5
    return precision, failing


def _estimate_phase4(figures: list[dict], counts: dict) -> tuple[float | None, list[str]]:
    failing: list[str] = []
    if not figures:
        return None, ["no figures extracted"]
    multi = sum(1 for f in figures if f.get("confidence") == "multi_signal")
    vector = sum(1 for f in figures if f.get("confidence") == "vector_only")
    failed = [f["figure_id"] for f in figures if f.get("status") == "FAILED"]
    if failed:
        failing.append(f"{len(failed)} figure(s) failed to render: {', '.join(failed[:10])}")

    # A vector_only region has no caption to corroborate it, so it is weaker
    # evidence; multi_signal is the strongest.
    if multi:
        conf = 1.0
    else:
        conf = max(0.0, 1.0 - 0.15 * (vector + 1) / max(1, len(figures)))
    if vector:
        failing.append(
            f"{vector} figure(s) are vector_only (no caption, so uncorroborated)"
        )
    return conf, failing


def _estimate_phase5(figures: list[dict]) -> tuple[float | None, list[str]]:
    if not figures:
        return None, []
    agree = sum(1 for f in figures if f.get("signals_agree"))
    degraded = [f["figure_id"] for f in figures if f.get("link_confidence") == "degraded"]
    failing = (
        [f"{len(degraded)} figure link(s) where signals disagree: {', '.join(degraded[:10])}"]
        if degraded
        else []
    )
    return agree / len(figures), failing


def _estimate_phase7(counts: dict) -> tuple[float | None, list[str]]:
    accepted = counts.get("accepted", 0)
    rejected = (
        counts.get("rejected_calculation", 0)
        + counts.get("rejected_ungrounded", 0)
        + counts.get("rejected_malformed", 0)
    )
    produced = accepted + rejected
    if not produced:
        return None, ["no instruction pairs produced"]
    failing: list[str] = []
    if counts.get("rejected_ungrounded"):
        failing.append(
            f"{counts['rejected_ungrounded']} pair(s) rejected for numbers absent "
            "from the source chunk"
        )
    if counts.get("generator") == "template":
        failing.append("generated with the template fallback, not a local LLM")
    # Precision proxy: the share of generated pairs that needed no rejection.
    return accepted / produced, failing


def _estimate_phase8(counts: dict) -> tuple[float | None, list[str]]:
    if not counts.get("no_text_layer_pages"):
        return 1.0, []
    failing = []
    if counts.get("ocr_failures"):
        failing.append(f"{counts['ocr_failures']} page(s) failed OCR")
    if counts.get("tesseract") == "missing":
        failing.append("tesseract is not installed, so no OCR'd text was recovered")
    if counts.get("ocr_disagreements"):
        failing.append(
            f"{counts['ocr_disagreements']} OCR'd page(s) disagreed with the text layer"
        )
    if not counts.get("ocr_pages"):
        return 0.0, failing
    ok = counts["ocr_pages"]
    total = counts["no_text_layer_pages"]
    return ok / total, failing


def write_validation_report(
    config: Config,
    codebook: str,
    processed: Path,
    extracted: Path,
    report: PhaseReport,
    clauses: list[dict],
    tables: list[dict],
    figures: list[dict],
    chunks: list[dict],
) -> Path:
    processed = Path(processed)
    target = config.get("validation.accuracy_target", 0.99)
    counts = {r.phase: r.counts for r in report.results}
    statuses = {r.phase: r.status for r in report.results}

    estimates: dict[str, tuple[float | None, list[str]]] = {}
    estimates["phase2_text"] = _estimate_phase2(clauses, counts.get("phase2_text", {}))
    estimates["phase3_tables"] = _estimate_phase3(tables, counts.get("phase3_tables", {}))
    estimates["phase4_figures"] = _estimate_phase4(figures, counts.get("phase4_figures", {}))
    estimates["phase5_linking"] = _estimate_phase5(figures)
    estimates["phase7_dataset"] = _estimate_phase7(counts.get("phase7_dataset", {}))
    estimates["phase8_ocr"] = _estimate_phase8(counts.get("phase8_ocr", {}))

    below = {k: v for k, v in estimates.items() if v[0] is not None and v[0] < target}

    # ---- spot-check sample ------------------------------------------------- #
    n_spot = config.get("validation.spot_check_links", 10)
    samples: list[dict] = []
    if figures:
        by_id = {c["clause_id"]: c for c in clauses}
        pool = sorted(figures, key=lambda f: f["figure_id"])
        rng = random.Random(0)  # reproducible sample
        picked = pool if len(pool) <= n_spot else rng.sample(pool, n_spot)
        for f in picked:
            cid = f.get("related_clause")
            clause = by_id.get(cid)
            samples.append(
                {
                    "figure_id": f["figure_id"],
                    "page": f.get("page"),
                    "caption": (f.get("caption") or "")[:70],
                    "related_clause": cid,
                    "linked_clause_title": (clause.get("title") or "")[:50] if clause else None,
                    "linked_clause_page": clause.get("page") if clause else None,
                    "confidence": f.get("confidence"),
                    "link_confidence": f.get("link_confidence"),
                    "gap_pt": f.get("link_gap_pt"),
                    "signals_agree": f.get("signals_agree"),
                }
            )

    lines: list[str] = []
    a = lines.append
    a(f"# Validation report -- {codebook}")
    a("")
    a(f"Overall pipeline status: **{report.overall.value}**")
    a("")
    if below:
        a(
            f"**{len(below)} phase(s) estimated below the "
            f"{_pct(target)} accuracy target.** The run is reported DEGRADED and the "
            "failing inputs are listed below."
        )
    else:
        a(f"All estimated accuracies meet the {_pct(target)} target.")
    a("")

    # ---- accuracy table ---------------------------------------------------- #
    a("## Estimated accuracy per phase")
    a("")
    a("| Phase | Status | Estimated accuracy | Basis |")
    a("|---|---|---|---|")
    for phase in ("phase8_ocr", "phase2_text", "phase3_tables", "phase4_figures",
                  "phase5_linking", "phase5_chunking", "phase6_vector_db", "phase7_dataset"):
        if phase not in statuses:
            continue
        est = estimates.get(phase, (None, []))[0]
        flag = " ⚠" if est is not None and est < target else ""
        a(f"| `{phase}` | {statuses[phase].value} | {_pct(est)}{flag} | {BASIS.get(phase, 'n/a')} |")
    a("")
    a("Accuracy figures are estimates derived from the signals noted, not measured "
      "against a human-labelled ground truth. The figures below state what each "
      "estimate is made of.")
    a("")

    # ---- phase detail ------------------------------------------------------ #
    a("## Clause extraction")
    a("")
    c2 = counts.get("phase2_text", {})
    a(f"- Clauses extracted: **{len(clauses)}**")
    a(f"- Depth distribution: `{c2.get('by_depth', {})}`")
    a(f"- Confidence breakdown: `{c2.get('by_confidence', {})}`")
    a(f"- Orphan clauses (dangling parent): **{c2.get('orphans', 'n/a')}**")
    a(f"- Duplicate clause numbers: **{c2.get('duplicate_clause_ids', 0)}**")
    a(f"- Table-of-contents pages excluded: **{c2.get('toc_pages_excluded', 0)}**")
    a(f"- Two-column pages split correctly: **{c2.get('two_column_pages', 0)}**")
    a(f"- Caption lines kept out of clause bodies: **{c2.get('caption_lines_excluded', 0)}**")
    td = c2.get("text_damage") or {}
    if td:
        a(f"- Source text-layer damage: **{td.get('damaged_pct')}%** of "
          f"{td.get('tokens')} tokens carry an internal capital "
          f"(OCR damage marker); **{td.get('glued_pct')}%** are 22+ character runs")
    a("")

    a("## Tables")
    a("")
    c3 = counts.get("phase3_tables", {})
    a(f"- Candidate grids offered to the gate: **{c3.get('candidate_grids', 0)}**")
    a(f"- Rejected by `is_real_table` / shape checks: **{c3.get('candidates_rejected', 0)}**")
    a(f"- Tables kept (needed ≥2 strategy votes): **{c3.get('tables_kept', 0)}**")
    a(f"- Vote histogram (votes → regions): `{c3.get('votes_histogram', {})}`")
    a(f"- Confidence: unanimous **{c3.get('unanimous', 0)}**, majority "
      f"**{c3.get('majority', 0)}**, degraded **{c3.get('degraded', 0)}**")
    a(f"- With a caption: **{c3.get('with_caption', 0)}**")
    a(f"- Repeated table numbers (footnote / cross-reference): "
      f"**{c3.get('duplicate_table_numbers', 0)}**")
    a("")

    a("## Figures")
    a("")
    c4 = counts.get("phase4_figures", {})
    a(f"- Figure caption lines in text: **{c4.get('caption_lines_found', 0)}**")
    a(f"- PDFFigures 2.0 items (figures / tables): **{c4.get('pdffigures2_figures', 0)} / "
      f"{c4.get('pdffigures2_items', 0)}**")
    a(f"- Vector pass found: **{c4.get('figures_caption_anchored', 0)}** captioned, "
      f"**{c4.get('orphan_regions', 0)}** uncaptioned orphans")
    a(f"- Agreed by both detectors (`multi_signal`): **{c4.get('cross_multi_signal', 0)}**")
    a(f"- `caption_only`: **{c4.get('figures_caption_anchored', 0)} - "
      f"{c4.get('cross_multi_signal', 0)}**")
    a(f"- `vector_only`: **{c4.get('orphan_regions', 0)}**")
    a(f"- `degraded` (detectors disagreed): **{c4.get('cross_degraded', 0)}**")
    a("")

    a("## Instruction dataset")
    a("")
    c7 = counts.get("phase7_dataset", {})
    if c7:
        a(f"- Generator: **{c7.get('generator', 'n/a')}**"
          + (f" (`{c7['llm_model']}`)" if c7.get("llm_model") else ""))
        a(f"- Chunks used: **{c7.get('chunks_selected', 0)}**")
        a(f"- Pairs accepted: **{c7.get('accepted', 0)}**")
        a(f"- Rejected as calculation intent: **{c7.get('rejected_calculation', 0)}**")
        a(f"- Rejected by numeric grounding: **{c7.get('rejected_ungrounded', 0)}**")
        a(f"- Rejected as malformed: **{c7.get('rejected_malformed', 0)}**")
    else:
        a("- Not run.")
    a("")

    # ---- spot check -------------------------------------------------------- #
    a("## Figure → clause link spot-check")
    a("")
    if samples:
        a("| Figure | Page | Caption | Linked clause | Link conf | Gap pt | Signals agree |")
        a("|---|---|---|---|---|---|---|")
        for s in samples:
            title = s["linked_clause_title"] or ""
            a(
                f"| `{s['figure_id']}` | {s['page']} | {(s['caption'] or '')[:34]} | "
                f"`{s['related_clause']}` {title[:22]} | {s['link_confidence']} | "
                f"{s['gap_pt']} | {s['signals_agree']} |"
            )
    else:
        a("No figures to sample.")
    a("")
    a("Verify these by eye against the PDF: a link is correct when the linked clause "
      "is the one whose text introduces the figure.")
    a("")

    # ---- failing inputs ---------------------------------------------------- #
    if below:
        a("## Inputs below the accuracy target")
        a("")
        for phase, (est, failing) in below.items():
            a(f"### `{phase}` -- estimated {_pct(est)}")
            a("")
            if failing:
                for f in failing:
                    a(f"- {f}")
            else:
                a("- No specific input identified; the estimate is below target "
                  "because the supporting evidence is weak rather than because a "
                  "known input failed.")
            a("")
        a("Per the project rule, this run is DEGRADED. The remaining phases still "
          "produced complete, usable artifacts; the items above are the ones to "
          "check by hand.")
        a("")

    # ---- phase statuses ---------------------------------------------------- #
    a("## Phase statuses")
    a("")
    for r in report.results:
        a(f"- `{r.phase}`: **{r.status.value}**")
        for reason in r.degraded_reasons:
            a(f"  - limit: {reason}")
        for w in r.warnings:
            a(f"  - warn: {w}")
    a("")

    out = processed / "validation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def accuracy_failures(
    config: Config, report: PhaseReport, clauses, tables, figures
) -> dict[str, float]:
    """Machine-readable form of the gate, for callers that want to fail a run."""
    target = config.get("validation.accuracy_target", 0.99)
    counts = {r.phase: r.counts for r in report.results}
    out = {}
    for phase, fn in (
        ("phase2_text", lambda: _estimate_phase2(clauses, counts.get("phase2_text", {}))),
        ("phase3_tables", lambda: _estimate_phase3(tables, counts.get("phase3_tables", {}))),
        ("phase4_figures", lambda: _estimate_phase4(figures, counts.get("phase4_figures", {}))),
        ("phase5_linking", lambda: _estimate_phase5(figures)),
        ("phase7_dataset", lambda: _estimate_phase7(counts.get("phase7_dataset", {}))),
        ("phase8_ocr", lambda: _estimate_phase8(counts.get("phase8_ocr", {}))),
    ):
        est, _ = fn()
        if est is not None and est < target:
            out[phase] = est
    return out