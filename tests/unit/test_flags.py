"""Generation flags.

Both thresholds on each flag are conjunctions, and the tests below are mostly
about the *second* condition — the one that stops the flag firing constantly on
short pages, which is what would make it useless.
"""

import pytest

from ocr_eval.config import load_config
from ocr_eval.metrics.flags import (
    EMPTY_PREDICTION,
    REPETITION,
    RUNAWAY_LENGTH,
    detect,
    is_repetitive,
    is_runaway,
    length_ratio,
)

CONFIG = load_config()
RATIO = CONFIG.runaway_length_ratio          # 2.0
EXCESS = CONFIG.runaway_min_char_excess      # 100
MIN_LEN = CONFIG.repetition_min_line_len     # 10
MIN_OCC = CONFIG.repetition_min_occurrences  # 5
MIN_SHARE = CONFIG.repetition_min_char_share  # 0.30

LINE = "هذا سطر عربي طويل بما يكفي"  # comfortably longer than MIN_LEN


# --- runaway ---------------------------------------------------------------

def test_runaway_needs_both_the_ratio_and_the_character_excess():
    reference = "a" * 200
    assert is_runaway(reference, "a" * 400, RATIO, EXCESS) is True


def test_a_short_page_does_not_trip_runaway_on_the_ratio_alone():
    """12 characters becoming 24 is double the length and means nothing."""
    reference = "a" * 12
    prediction = "a" * 24
    assert prediction == "a" * 24 and len(prediction) >= RATIO * len(reference)
    assert is_runaway(reference, prediction, RATIO, EXCESS) is False


def test_a_long_page_slightly_over_does_not_trip_runaway_on_excess_alone():
    reference = "a" * 1000
    prediction = "a" * 1150  # +150 chars, but nowhere near double
    assert len(prediction) - len(reference) >= EXCESS
    assert is_runaway(reference, prediction, RATIO, EXCESS) is False


def test_exactly_at_both_thresholds_fires():
    reference = "a" * 100
    prediction = "a" * 200  # exactly 2.0x and exactly +100
    assert is_runaway(reference, prediction, RATIO, EXCESS) is True


# --- repetition ------------------------------------------------------------

def test_repetition_needs_enough_occurrences():
    prediction = "\n".join([LINE] * (MIN_OCC - 1))
    assert is_repetitive(prediction, MIN_LEN, MIN_OCC, MIN_SHARE) is False

    prediction = "\n".join([LINE] * MIN_OCC)
    assert is_repetitive(prediction, MIN_LEN, MIN_OCC, MIN_SHARE) is True


def test_repetition_ignores_short_lines():
    """Short repeated lines are normal in tables and headers."""
    prediction = "\n".join(["ص ١"] * 20)
    assert is_repetitive(prediction, MIN_LEN, MIN_OCC, MIN_SHARE) is False


def test_one_repeated_line_in_a_long_page_is_not_repetition():
    filler = "\n".join(f"سطر مختلف رقم {i} مع نص إضافي هنا" for i in range(120))
    prediction = filler + "\n" + "\n".join([LINE] * MIN_OCC)
    assert is_repetitive(prediction, MIN_LEN, MIN_OCC, MIN_SHARE) is False


def test_repetition_matches_lines_that_differ_only_in_diacritics():
    """Lines are compared skeleton-normalised."""
    vowelled = "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
    plain = "الحمد لله رب العالمين"
    prediction = "\n".join([vowelled, plain] * 4)
    assert is_repetitive(prediction, MIN_LEN, MIN_OCC, MIN_SHARE) is True


def test_empty_prediction_is_not_repetitive():
    assert is_repetitive("", MIN_LEN, MIN_OCC, MIN_SHARE) is False


# --- length ratio and detect() --------------------------------------------

def test_length_ratio_never_divides_by_zero():
    assert length_ratio("", "abcd") == 4.0
    assert length_ratio("abcd", "") == 0.0
    assert length_ratio("abcd", "abcd") == 1.0


def test_detect_returns_sorted_flags():
    reference = "a" * 150
    prediction = "\n".join([LINE] * 12)
    flags = detect(reference, prediction, CONFIG)
    assert flags == sorted(flags)


def test_empty_prediction_flag():
    assert detect("a" * 50, "   ", CONFIG) == [EMPTY_PREDICTION]


def test_a_clean_page_carries_no_flags():
    assert detect(LINE, LINE, CONFIG) == []


def test_flags_do_not_exclude_a_page_from_the_aggregate():
    """The flag says where an error rate came from; it does not remove the page."""
    from ocr_eval.io.manifest import ReferencePage
    from ocr_eval.score.page import score_page

    reference = "a" * 150
    prediction = "a" * 400
    page = ReferencePage(page_id="p", split="s", kind="synthetic", reference_text=reference)
    result = score_page(page, reference, prediction, CONFIG)

    assert RUNAWAY_LENGTH in result.flags
    assert result.carries_metrics, "a flagged page stays in the aggregate"
    assert result.metrics["strict"].cer > 0
