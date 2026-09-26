"""Reading order: matching determinism, and the two numbers it produces.

The assertion this file exists for is the null one. A page with fewer than two
matched lines has no order to get right or wrong, and reporting a free `1.0`
there would inflate every aggregate it entered — so it reports `null`, never
`1.0`.
"""

import pytest

from ocr_eval.metrics.order import measure

MIN_SIM = 0.5

A = "بسم الله الرحمن الرحيم"
B = "الحمد لله رب العالمين"
C = "هذا نص تجريبي للتقييم"


def test_perfect_page_has_perfect_order_and_zero_corrected_cer():
    text = "\n".join([A, B, C])
    result = measure(text, text, MIN_SIM)
    assert result.matched_lines == 3
    assert result.reading_order_accuracy == 1.0
    assert result.order_corrected_cer == 0.0


def test_shuffled_page_is_recognised_as_a_sequencing_failure():
    """The signature FR-012a asks a reader to be able to spot."""
    reference = "\n".join([A, B, C])
    prediction = "\n".join([C, A, B])
    result = measure(reference, prediction, MIN_SIM)

    # Every line was read correctly...
    assert result.matched_lines == 3
    assert result.order_corrected_cer == 0.0
    # ...but they came out in the wrong sequence.
    assert result.reading_order_accuracy < 1.0


def test_shuffled_page_strict_cer_is_high_while_corrected_is_zero():
    from ocr_eval.metrics.rates import score_level

    reference = "\n".join([A, B, C])
    prediction = "\n".join([C, A, B])

    strict = score_level(reference, prediction)
    order = measure(reference, prediction, MIN_SIM)

    assert strict.cer > 0.3, "a reordered page should look badly wrong as produced"
    assert order.order_corrected_cer == 0.0, "but perfectly right once re-sequenced"


def test_fewer_than_two_matched_lines_reports_null_never_one():
    result = measure(A, A, MIN_SIM)
    assert result.matched_lines == 1
    assert result.reading_order_accuracy is None


def test_no_matching_lines_reports_null():
    result = measure(A, "completely unrelated latin text here", MIN_SIM)
    assert result.reading_order_accuracy is None


def test_empty_prediction_reports_null_order():
    result = measure("\n".join([A, B]), "", MIN_SIM)
    assert result.reading_order_accuracy is None
    assert result.pred_lines == 0


def test_matching_is_deterministic_under_identical_lines():
    """Ties must resolve identically on every run, or SC-003 fails."""
    reference = "\n".join([A, A, A, B])
    prediction = "\n".join([A, B, A, A])
    first = measure(reference, prediction, MIN_SIM)
    for _ in range(20):
        assert measure(reference, prediction, MIN_SIM).to_dict() == first.to_dict()


def test_line_counts_ignore_blank_lines():
    result = measure(f"{A}\n\n\n{B}", f"{A}\n{B}\n\n", MIN_SIM)
    assert result.ref_lines == 2
    assert result.pred_lines == 2


def test_unmatched_prediction_lines_are_kept_not_dropped():
    """Dropping spurious output would turn a hallucinated line into a free pass."""
    reference = "\n".join([A, B])
    with_extra = "\n".join([A, B, "سطر إضافي لم يكن موجودا في الأصل"])
    clean = measure(reference, reference, MIN_SIM)
    noisy = measure(reference, with_extra, MIN_SIM)
    assert noisy.order_corrected_cer > clean.order_corrected_cer


def test_similarity_floor_stops_unrelated_lines_pairing():
    reference = "\n".join([A, B])
    prediction = "\n".join(["xxxxxxxxxxxx", "yyyyyyyyyyyy"])
    result = measure(reference, prediction, MIN_SIM)
    assert result.matched_lines == 0


def test_slightly_misread_and_misordered_lines_still_pair():
    """Similarity matching, not exact matching — a line can be both."""
    reference = "\n".join([A, B])
    prediction = "\n".join([B.replace("رب", "رپ"), A.replace("الرحيم", "الرحىم")])
    result = measure(reference, prediction, MIN_SIM)
    assert result.matched_lines == 2
    assert result.reading_order_accuracy == 0.0  # both lines swapped


def test_serialised_order_block_has_exactly_the_contract_keys():
    # contracts/page-result.schema.json closes this object with
    # additionalProperties:false, so an extra key here fails the contract.
    result = measure("\n".join([A, B]), "\n".join([A, B]), MIN_SIM)
    assert set(result.to_dict()) == {
        "order_corrected_cer",
        "reading_order_accuracy",
        "matched_lines",
        "ref_lines",
        "pred_lines",
    }
