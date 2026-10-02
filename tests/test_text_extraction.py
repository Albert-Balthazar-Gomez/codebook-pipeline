"""Phase 2: clause extraction.

The fixture's clause list is known exactly, so these are precision/recall
assertions rather than smoke tests.
"""

from __future__ import annotations

EXPECTED_CLAUSES = {
    "26", "26.1", "26.5", "26.5.1", "26.5.1.1", "26.5.1.2",
    "27", "27.1", "27.2", "27.2.1", "27.3", "27.3.1", "27.3.1.1", "27.3.1.2", "27.3.1.3",
    "28", "28.1", "28.2", "28.2.1", "28.3", "28.3.1",
    "29", "29.1", "29.2", "29.2.1",
    "30", "30.1", "30.2", "30.2.1",
    "31", "31.1", "31.2", "31.2.1",
}


def test_clause_recall_and_precision(extracted):
    found = {c["clause_id"] for c in extracted["clauses"]}
    assert found == EXPECTED_CLAUSES, (
        f"missing={EXPECTED_CLAUSES - found} unexpected={found - EXPECTED_CLAUSES}"
    )


def test_no_orphans(extracted):
    ids = {c["clause_id"] for c in extracted["clauses"]}
    for c in extracted["clauses"]:
        if c["depth"] > 1:
            assert c["parent_id"] in ids, f"{c['clause_id']} has dangling parent {c['parent_id']}"


def test_depth_distribution(extracted):
    by_depth = extracted["results"]["p2"].counts["by_depth"]
    assert by_depth == {1: 6, 2: 14, 3: 8, 4: 5}


def test_table_cells_are_not_clauses(extracted):
    """A cell reading "0.36 0.43" must not become clause 0.36 titled 0.43."""
    numbers = [c["clause_id"] for c in extracted["clauses"]]
    assert not any(n.startswith("0.") for n in numbers), numbers


def test_running_heads_removed(extracted):
    text = " ".join(c["text"] for c in extracted["clauses"])
    assert "INDIAN STANDARD IS 456" not in text
    assert "BIS New Delhi" not in text


def test_every_clause_has_confidence(extracted):
    for c in extracted["clauses"]:
        assert c["confidence"] in ("regex", "heuristic", "ocr", "preamble")
        assert c["confidence_reason"]


def test_regex_and_heuristic_differ(extracted):
    """Bold headings must be `regex`; run-in body text must be `heuristic`."""
    by_id = {c["clause_id"]: c for c in extracted["clauses"]}
    assert by_id["27.3.1"]["confidence"] == "regex"       # bold, short title
    assert by_id["27.3.1.1"]["confidence"] == "heuristic"  # run-in body text