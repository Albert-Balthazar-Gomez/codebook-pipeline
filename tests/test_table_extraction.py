"""Phase 3: table extraction and the is_real_table gate."""

from __future__ import annotations

EXPECTED_TABLES = {"T1", "T2", "T3", "T4", "T6", "T7"}


def test_all_real_tables_found(extracted):
    assert {t["table_id"] for t in extracted["tables"]} == EXPECTED_TABLES


def test_layout_frame_rejected(extracted):
    """Table 5 in the fixture is a deliberately sparse grid, not a table."""
    assert "T5" not in {t["table_id"] for t in extracted["tables"]}
    assert extracted["results"]["p3"].counts["candidates_rejected"] == 1


def test_gate_counts_consistent(extracted):
    c = extracted["results"]["p3"].counts
    assert c["candidate_grids"] == c["tables_kept"] + c["candidates_rejected"]


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