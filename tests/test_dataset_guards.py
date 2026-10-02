"""Phase 7 guards: calculation refusal and numeric grounding.

No API calls. These assert the filters that keep a fabricated number out of the
dataset even when the model produces one.
"""

from __future__ import annotations

import pytest

from src.dataset_generation.instruction_generator import (
    _parse_json_array,
    is_grounded,
    looks_like_calculation,
)

CALC = [
    "Calculate the moment of resistance for M25 concrete",
    "Compute the design shear stress",
    "What is the required area of steel?",
    "How much water is used per cubic metre?",
    "Determine the depth of the neutral axis",
    "Find the value of Mu for a 200x400 beam",
    "What is the design size of the column?",
    "Work out the required spacing of stirrups",
]
SAFE = [
    "What is the limiting moment of resistance factor for M25?",
    "Which grade of concrete has fck of 30 MPa?",
    "What does clause 26.5.1.1 require?",
    "When shall vertical stirrups be provided?",
    "Which table gives the minimum cement content?",
    "What is meant by characteristic strength?",
]


@pytest.mark.parametrize("q", CALC)
def test_calculation_intent_rejected(q):
    assert looks_like_calculation(q), q


@pytest.mark.parametrize("q", SAFE)
def test_legitimate_question_kept(q):
    assert not looks_like_calculation(q), q


def test_grounded_answer_passes():
    src = "Grade M25 fck 25 MPa Limiting Mu/fck bd2 = 0.175"
    assert is_grounded("The factor is 0.175 for M25.", src)[0]


def test_invented_number_rejected():
    src = "Grade M25 fck 25 MPa Limiting Mu/fck bd2 = 0.175"
    ok, why = is_grounded("The factor is 0.183 for M25.", src)
    assert not ok and "0.183" in why


def test_number_absent_from_source_rejected():
    src = "Grade M25 fck 25 MPa"
    assert not is_grounded("Use 35 MPa.", src)[0]


def test_answer_without_numbers_passes():
    src = "The tensile strength of concrete is ignored in design."
    assert is_grounded("The tensile strength of concrete is ignored.", src)[0]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('[{"instruction":"a","output":"b"}]', 1),
        ('```json\n[{"instruction":"a","output":"b"}]\n```', 1),
        ('Here you go: [{"instruction":"a","output":"b"}] hope that helps', 1),
        ('[{"instruction":"a",}]', 0),          # malformed
        ('no json at all', 0),
        ('[]', 0),
        ('[{"instruction":"a with } brace","output":"c"}]', 1),  # braces in strings
    ],
)
def test_json_array_parsing(raw, expected):
    assert len(_parse_json_array(raw)) == expected


def test_grounding_rule_verbatim_present():
    """The spec's hard constraint must appear exactly as written."""
    from src.dataset_generation.instruction_generator import GROUNDING_RULE

    assert "Answer only from the provided text" in GROUNDING_RULE
    assert "See {figure_id}" in GROUNDING_RULE