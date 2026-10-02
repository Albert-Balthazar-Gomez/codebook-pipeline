"""Phase 5: linking and chunking."""

from __future__ import annotations

# Ground truth: which clause each fixture figure actually belongs to.
EXPECTED_LINKS = {
    "p2_f1": "26.5.1.1",   # Fig. 1 under 26.5.1.1
    "p3_f1": "26.5.1.2",   # Fig. 2 under 26.5.1.2
    "p5_f1": "27.3.1.3",   # Fig. 3 referenced by 27.3.1.3
    "p9_f1": "30.2.1",     # Fig. 5 referenced by 30.2.1
}


def test_figure_clause_links_are_correct(extracted):
    """Manual spot-check: all 4 figures, 100% expected."""
    by_id = {f["figure_id"]: f for f in extracted["figures"]}
    for fig_id, clause in EXPECTED_LINKS.items():
        assert by_id[fig_id]["related_clause"] == clause, fig_id


def test_all_links_high_confidence(extracted):
    """0% error rate on the fixture; the project gate is >20%."""
    for f in extracted["figures"]:
        assert f["link_confidence"] == "high", (f["figure_id"], f["link_gap_pt"])


def test_link_signals_are_recorded_and_agree(extracted):
    """Multi-signal linking: the caption signal must never be fabricated."""
    for f in extracted["figures"]:
        s = f["link_signals"]
        assert {"caption_number", "proximity", "inside_table"} <= set(s)
        assert f["signals_agree"] is True, (f["figure_id"], s)


def test_captions_are_not_duplicated_into_clause_bodies(extracted):
    """A caption left in a clause body creates a circular figure self-reference."""
    counts = extracted["results"]["p2"].counts
    assert counts.get("caption_lines_excluded", 0) > 0
    for c in extracted["clauses"]:
        assert "Typical Arrangement of Reinforcement" not in c["body"]


def test_link_gaps_are_small(extracted):
    for f in extracted["figures"]:
        assert f["link_gap_pt"] is not None
        assert f["link_gap_pt"] < 200, f["figure_id"]


def test_tables_linked_to_clause(extracted):
    ids = {c["clause_id"] for c in extracted["clauses"]}
    for t in extracted["tables"]:
        assert t["clause_ref"] in ids, t["table_id"]


def test_chunk_count_and_types(extracted):
    c = extracted["results"]["p5_chunk"].counts
    assert c["chunks_text"] == len(EXPECTED_LINKS) + 29
    assert c["chunks_table"] == 6
    assert c["chunks_figure"] == 4
    assert c["chunks_total"] == c["chunks_text"] + c["chunks_table"] + c["chunks_figure"]


def test_every_chunk_has_embed_text(extracted):
    for chunk in extracted["chunks"]:
        if chunk["type"] == "text":
            assert chunk["text"].strip()
        else:
            assert chunk["embed_text"].strip(), chunk["chunk_id"]


def test_chunk_ids_unique(extracted):
    ids = [c["chunk_id"] for c in extracted["chunks"]]
    assert len(ids) == len(set(ids))


def test_figure_chunks_carry_png(extracted):
    figs = [c for c in extracted["chunks"] if c["type"] == "figure"]
    assert len(figs) == 4
    for c in figs:
        assert c["png_path"] and c["caption"]


def test_split_clauses_keep_parent(extracted, config):
    """A clause over the word budget must split and retain parent_clause_id."""
    from src.processing.chunker import _chunk_clause

    long_clause = {
        "clause_id": "99.1",
        "parent_id": "99",
        "depth": 2,
        "title": "Long",
        "text": "\n\n".join(["word " * 30 for _ in range(12)]),
        "page": 1,
        "confidence": "regex",
    }
    parts = _chunk_clause(long_clause, max_words=100)
    assert len(parts) > 1
    assert all(p["parent_clause_id"] == "99" for p in parts)
    assert all(p["n_parts"] == len(parts) for p in parts)
    assert [p["part"] for p in parts] == list(range(1, len(parts) + 1))
    assert all(len(p["text"].split()) <= 100 for p in parts)


def test_no_mid_sentence_cut(extracted):
    """Every part of a split clause ends on sentence-ending punctuation."""
    from src.processing.chunker import _chunk_clause

    text = " ".join(["This is a sentence about reinforcement." for _ in range(40)])
    parts = _chunk_clause(
        {"clause_id": "9.9", "parent_id": None, "depth": 2, "title": "T", "text": text,
         "page": 1, "confidence": "regex"},
        max_words=60,
    )
    assert len(parts) > 1
    for p in parts[:-1]:
        assert p["text"].rstrip().endswith(".")