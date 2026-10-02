"""Phase 7: local generators. No API, no key, no network.

The template generator is exercised end to end. The Qwen path is exercised
through an injected fake model, so the guards and record format are verified
without downloading 3.4 GB of weights.
"""

from __future__ import annotations

import json

import pytest

from src.common.config import deep_merge, load_config
from src.dataset_generation.instruction_generator import (
    GROUNDING_RULE,
    generate_dataset,
    is_grounded,
    looks_like_calculation,
    template_pairs,
)

TEXT_CHUNK = {
    "chunk_id": "text:26.5.1.1",
    "type": "text",
    "clause_id": "26.5.1.1",
    "page": 26,
    "text": "26.5.1.1 Development length is the length of bar needed beyond the "
            "critical section. Bars bundled together must be increased by 0. "
            "The tension side requires 40 times the diameter.",
    "body": "Development length is the length of bar needed beyond the critical "
            "section. Bars bundled together must be increased by 0. The tension "
            "side requires 40 times the diameter.",
    "confidence": "regex+font",
    "word_count": 32,
}

TABLE_CHUNK = {
    "chunk_id": "table:T1",
    "type": "table",
    "clause_ref": "22.7",
    "page": 28,
    "caption": "Table 1 Permissible Limit for Solids",
    "markdown": "| Grade | Limit |\n| --- | --- |\n| M20 | 0.60 |",
    "json_data": [["Grade", "Limit"], ["M20", "0.60"]],
    "embed_text": "22.7 Table 1 Permissible Limit for Solids | Grade | Limit | M20 | 0.60 |",
    "word_count": 12,
}

FIGURE_CHUNK = {
    "chunk_id": "figure:p2_f1",
    "type": "figure",
    "figure_id": "p2_f1",
    "related_clause": "26.5.1.1",
    "caption": "Fig. 1 Typical Arrangement of Reinforcement",
    "png_path": "figures/p2_f1.png",
    "embed_text": "26.5.1.1 Fig. 1 Typical Arrangement of Reinforcement",
    "word_count": 8,
}


def _config(**over):
    cfg = load_config()
    cfg.data = deep_merge(cfg.data, {"dataset_generation": {"generator": "template"}})
    cfg.data["dataset_generation"].update(over)
    return cfg


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "q",
    [
        "Calculate the moment of resistance",
        "What is the required area of steel?",
        "How much water is used?",
        "Determine the depth of the neutral axis",
        "Compute the design shear stress",
    ],
)
def test_calculation_intent_rejected(q):
    assert looks_like_calculation(q)


@pytest.mark.parametrize(
    "q",
    [
        "What is the limiting moment of resistance factor for M25?",
        "Which grade of concrete has fck of 30 MPa?",
        "What does clause 26.5.1.1 require?",
    ],
)
def test_legitimate_question_kept(q):
    assert not looks_like_calculation(q)


def test_invented_number_rejected():
    ok, why = is_grounded("The factor is 0.183 for M25.", "M25 factor 0.175")
    assert not ok and "0.183" in why


def test_number_from_source_kept():
    assert is_grounded("Use 25 MPa for M25.", "Grade M25 fck 25 MPa")[0]


def test_grounding_rule_verbatim():
    assert "Answer only from the provided text" in GROUNDING_RULE
    assert "See {figure_id}" in GROUNDING_RULE


# --------------------------------------------------------------------------- #
# Template generator
# --------------------------------------------------------------------------- #
def test_template_figure_is_a_pointer_not_a_description():
    """No invented description of an image; the spec requires a See {id} pointer."""
    pairs = template_pairs(FIGURE_CHUNK, 3)
    assert len(pairs) == 1
    assert pairs[0]["output"] == "See p2_f1"


def test_template_table_uses_its_json():
    pairs = template_pairs(TABLE_CHUNK, 3)
    answers = " ".join(p["output"] for p in pairs)
    assert "Permissible Limit for Solids" in answers
    assert "Grade" in answers and "M20" in answers


def test_template_text_extracts_a_sentence():
    pairs = template_pairs(TEXT_CHUNK, 3)
    assert pairs
    for p in pairs:
        assert p["instruction"] and p["output"]
        # Every answer must be a real span of the source, not a paraphrase.
        assert p["output"][:40].lower() in TEXT_CHUNK["body"].lower() or p["output"][:20] in TEXT_CHUNK["body"]


def test_template_dataset_records_are_valid():
    chunks = [TEXT_CHUNK, TABLE_CHUNK, FIGURE_CHUNK]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        result = generate_dataset(chunks, _config(max_chunks=10), Path(tmp))
        out = Path(tmp) / "instruction_dataset.jsonl"
        records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert records, "template generator produced nothing"
    for r in records:
        assert r["generator"] == "template"
        assert r["source_chunk_id"]
    assert result.counts["generator"] == "template"


def test_template_path_degrades_the_phase():
    """Falling back to templates must be visible, not silent."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        result = generate_dataset([TEXT_CHUNK], _config(), Path(tmp))
    assert result.status.value == "DEGRADED"
    assert any("template" in d for d in result.degraded_reasons)


# --------------------------------------------------------------------------- #
# Qwen path, with the model injected
# --------------------------------------------------------------------------- #
def test_qwen_path_uses_local_llm_and_applies_guards(monkeypatch, tmp_path):
    import src.dataset_generation.instruction_generator as ig

    class FakeLLM:
        model_name = "Qwen/Qwen3-1.7B"

        def __init__(self, config):
            self.config = config

        def load(self):
            return None

        def generate(self, prompt):
            return json.dumps(
                [
                    {"instruction": "What does this clause require?", "output": "40 times the diameter."},
                    {"instruction": "Calculate the bar length", "output": "40 mm"},
                    {"instruction": "What is the limit?", "output": "It is 0.183"},
                ]
            )

    monkeypatch.setattr(ig, "LocalLLM", FakeLLM)
    result = ig.generate_dataset(
        [TEXT_CHUNK], _config(generator="qwen", max_chunks=1), tmp_path
    )
    records = [
        json.loads(l)
        for l in (tmp_path / "instruction_dataset.jsonl").read_text().splitlines()
        if l.strip()
    ]
    # Only the grounded, non-calculation pair survives.
    assert len(records) == 1
    assert records[0]["generator"] == "qwen3-4b"
    assert result.counts["rejected_calculation"] == 1
    assert result.counts["rejected_ungrounded"] == 1


def test_llm_load_failure_falls_back_to_templates(monkeypatch, tmp_path):
    import src.dataset_generation.instruction_generator as ig

    class BoomLLM:
        model_name = "Qwen/Qwen3-1.7B"

        def __init__(self, config):
            pass

        def load(self):
            raise RuntimeError("not enough memory to load the model")

    monkeypatch.setattr(ig, "LocalLLM", BoomLLM)
    result = ig.generate_dataset([TEXT_CHUNK], _config(generator="qwen"), tmp_path)
    assert result.counts["generator"] == "template"
    assert any("unavailable" in w for w in result.warnings)
    assert result.counts["records"] > 0


def test_no_api_key_is_ever_consulted():
    """Phase 7 is local-only by construction; nothing reads a key env var."""
    import inspect

    import src.dataset_generation.instruction_generator as ig

    src = inspect.getsource(ig)
    for forbidden in ("GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                      "google.genai", "requests.post", "api_key_env"):
        assert forbidden not in src, forbidden