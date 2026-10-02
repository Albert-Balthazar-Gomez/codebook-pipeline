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
    from src.extraction.text_extractor import CONF_ORDER

    for c in extracted["clauses"]:
        assert c["confidence"] in CONF_ORDER
        assert c["confidence_reason"]


def test_font_cue_splits_regex_plus_font_from_regex_only(extracted):
    """Bold headings are `regex+font`; run-in body text only matches the regex."""
    by_id = {c["clause_id"]: c for c in extracted["clauses"]}
    assert by_id["27.3.1"]["confidence"] == "regex+font"    # bold, short title
    assert by_id["27.3.1.1"]["confidence"] == "regex_only"  # plain body text


def test_ambiguous_clauses_degrade_the_phase(extracted):
    """`ambiguous` means the cues disagree, which must surface as DEGRADED."""
    from src.extraction.text_extractor import CONF_AMBIGUOUS

    result = extracted["results"]["p2"]
    ambiguous = [c for c in extracted["clauses"] if c["confidence"] == CONF_AMBIGUOUS]
    if ambiguous:
        assert result.status.value != "SUCCESS"
        assert any("ambiguous" in r for r in result.degraded_reasons)


def test_indent_cue_is_measured(extracted):
    counts = extracted["results"]["p2"].counts
    assert "indent_levels" in counts
    assert "indent_cue_trusted" in counts