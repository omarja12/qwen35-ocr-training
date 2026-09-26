"""The four-cell `[UNREADABLE]` matrix.

The cell that matters most is the one a looser implementation would get wrong:
a prediction that *contains* the marker but says more than the marker has not
honoured the contract. Treating it as a pass would let hallucination hide behind
a compliant-looking prefix (research R-008).
"""

import pytest

from ocr_eval.normalise.tables import UNREADABLE_MARKER
from ocr_eval.score.contract import (
    CORRECT_REFUSAL,
    FALSE_REFUSAL,
    HALLUCINATION,
    SCORED,
    carries_metrics,
    classify,
    is_marker,
)

LEGIBLE = "بسم الله الرحمن الرحيم"
M = UNREADABLE_MARKER


@pytest.mark.parametrize(
    "reference,prediction,expected",
    [
        (M, M, CORRECT_REFUSAL),
        (M, "some invented text", HALLUCINATION),
        (LEGIBLE, M, FALSE_REFUSAL),
        (LEGIBLE, "بسم الله الرحمن الرحيم", SCORED),
    ],
)
def test_the_four_cells(reference, prediction, expected):
    assert classify(reference, prediction) == expected


def test_marker_must_stand_alone():
    assert is_marker(M)
    assert is_marker(f"  {M}\n")           # surrounding whitespace is stripped
    assert not is_marker(f"{M} — possibly a stamp?")
    assert not is_marker(f"Here is the text: {M}")
    assert not is_marker("[unreadable]")   # case-sensitive
    assert not is_marker("UNREADABLE")


def test_marker_with_a_suffix_on_a_probe_page_is_hallucination():
    # The specific case R-008 calls out. A model that annotates its refusal has
    # produced text where none exists.
    assert classify(M, f"{M} — possibly a stamp?") == HALLUCINATION


def test_probe_outcomes_carry_no_metrics():
    assert carries_metrics(CORRECT_REFUSAL) is False
    assert carries_metrics(HALLUCINATION) is False


def test_false_refusal_carries_metrics_and_is_counted_separately():
    # It is a wrong transcription of a legible page, so FR-001 requires it in
    # the aggregate; FR-010 requires the separate count. Both, not either.
    assert carries_metrics(FALSE_REFUSAL) is True
    assert carries_metrics(SCORED) is True


def test_empty_prediction_on_a_legible_page_is_scored_not_refused():
    # An empty prediction is a real model behaviour, scored as total loss.
    assert classify(LEGIBLE, "") == SCORED
