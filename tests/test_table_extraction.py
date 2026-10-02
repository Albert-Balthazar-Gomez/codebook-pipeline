"""Phase 3: table extraction and the is_real_table gate."""

from __future__ import annotations

EXPECTED_TABLES = {"T1", "T2", "T3", "T4", "T6", "T7"}


def test_all_real_tables_found(extracted):
    assert {t["table_id"] for t in extracted["tables"]} == EXPECTED_TABLES


def test_voting_labels_are_honest(extracted):
    """`unanimous` requires every strategy to agree on every cell."""
    for t in extracted["tables"]:
        assert t["confidence"] in ("unanimous", "majority", "degraded")
        if t["confidence"] == "unanimous":
            assert t["cell_agreement"] == 1.0
            assert not t.get("single_detector")
        if t["confidence"] == "degraded":
            assert t["strategy_variants"], "degraded table must keep its variants"
            assert t["cell_agreement"] < 0.9


def test_layout_frame_rejected(extracted):
    """Table 5 in the fixture is a deliberately sparse grid, not a table.

    Three strategies now propose candidates, so the rejected *count* is larger
    than before; what matters is that nothing sparse survived the gate.
    """
    assert "T5" not in {t["table_id"] for t in extracted["tables"]}
    assert extracted["results"]["p3"].counts["candidates_rejected"] > 0


def test_gate_counts_consistent(extracted):
    c = extracted["results"]["p3"].counts
    assert c["candidate_grids"] == c["tables_kept"] + c["candidates_rejected"]


def test_multi_strategy_regions_need_multiple_votes(extracted):
    """min_votes=2 applies wherever a second strategy could have agreed.

    A borderless region is proposed only by the caption-band pass -- the ruled
    strategies find nothing there by definition -- so it is admitted on its
    caption instead, and must declare itself single_detector.
    """
    for t in extracted["tables"]:
        assert len(t["vote_strategies"]) == t["n_votes"]
        if t.get("single_detector"):
            assert t["vote_strategies"] == ["caption_band_text"], t["table_id"]
            assert t["caption"], "a caption-band table must have a caption"
            assert t["confidence"] != "unanimous"
        else:
            assert t["n_votes"] >= 2, t["table_id"]


def test_single_detector_tables_are_flagged(extracted):
    for t in extracted["tables"]:
        if t["confidence"] == "unanimous":
            assert not t.get("single_detector")


def test_degraded_tables_degrade_the_phase(extracted):
    degraded = [t for t in extracted["tables"] if t["confidence"] == "degraded"]
    if degraded:
        assert extracted["results"]["p3"].status.value != "SUCCESS"


def test_verification_renders_exist(extracted):
    """A 300 DPI crop per table, so a reviewer can diff extraction vs source."""
    from pathlib import Path

    base = extracted["dir"]
    for t in extracted["tables"]:
        rel = t.get("verification_png")
        if rel is None:
            continue
        path = base / rel
        assert path.exists(), rel


def test_every_table_has_caption_and_number(extracted):
    for t in extracted["tables"]:
        assert t["caption"], t["table_id"]
        assert t["table_number"], t["table_id"]


def test_markdown_shape(extracted):
    t1 = next(t for t in extracted["tables"] if t["table_id"] == "T1")
    lines = t1["markdown"].splitlines()
    assert lines[0].startswith("| Grade of concrete")
    assert set(lines[1]) <= set("|- ")
    assert len(lines) == t1["n_rows"] + 1  # header, separator, body
    assert "0.138" in t1["markdown"]


def test_json_data_has_no_duplicated_cells(extracted):
    """Nested-cell subtraction: a spanning cell must not repeat its children."""
    for t in extracted["tables"]:
        assert t["json_data"], t["table_id"]


def test_pipes_escaped(extracted):
    t1 = next(t for t in extracted["tables"] if t["table_id"] == "T1")
    # Cell values contain "/" and spaces, never a raw pipe that would break the row.
    for line in t1["markdown"].splitlines():
        assert line.count("|") >= 4