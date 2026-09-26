"""Per-page generation flags.

A flag never removes a page from an aggregate. It tells a reader **where an
error rate came from** — a CER of 3.0 is arithmetic until you know one page
repeated the same line forty times, at which point it is a diagnosis
(FR-011, research R-010).

Both thresholds on each flag are deliberately conjunctions. A ratio alone fires
constantly on short pages; a character count alone fires on every long page that
runs slightly over.
"""

from __future__ import annotations

from collections import Counter

from ocr_eval.normalise.levels import skeleton

RUNAWAY_LENGTH = "runaway_length"
REPETITION = "repetition"
EMPTY_PREDICTION = "empty_prediction"

FLAG_NAMES = (RUNAWAY_LENGTH, REPETITION, EMPTY_PREDICTION)


def length_ratio(reference: str, prediction: str) -> float:
    """len(prediction) / max(len(reference), 1), at strict level."""
    return len(prediction) / max(len(reference), 1)


def is_runaway(reference: str, prediction: str, ratio: float, min_excess: int) -> bool:
    """Grossly disproportionate output length."""
    return (
        len(prediction) >= ratio * len(reference)
        and len(prediction) - len(reference) >= min_excess
    )


def is_repetitive(
    prediction: str,
    min_line_len: int,
    min_occurrences: int,
    min_char_share: float,
) -> bool:
    """One line emitted over and over, inflating length without adding content.

    Lines are compared skeleton-normalised so that a model repeating the same
    line with slightly different diacritics still trips the flag.
    """
    if not prediction.strip():
        return False

    lines = [skeleton(line) for line in prediction.split("\n")]
    candidates = [line for line in lines if len(line) >= min_line_len]
    if not candidates:
        return False

    total_chars = max(len(prediction), 1)
    for line, count in Counter(candidates).items():
        if count < min_occurrences:
            continue
        # Characters contributed by the repeats beyond the first occurrence.
        if (len(line) * count) / total_chars >= min_char_share:
            return True
    return False


def detect(reference: str, prediction: str, config) -> list[str]:
    """Every flag that fires, sorted so the artifact is order-independent."""
    flags: list[str] = []

    if not prediction.strip():
        flags.append(EMPTY_PREDICTION)

    if is_runaway(
        reference,
        prediction,
        config.runaway_length_ratio,
        config.runaway_min_char_excess,
    ):
        flags.append(RUNAWAY_LENGTH)

    if is_repetitive(
        prediction,
        config.repetition_min_line_len,
        config.repetition_min_occurrences,
        config.repetition_min_char_share,
    ):
        flags.append(REPETITION)

    return sorted(flags)
