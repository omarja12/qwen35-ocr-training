"""The `[UNREADABLE]` contract and its four-cell classification.

The declared behaviour for an unreadable image is that the model emits the
single fixed marker and nothing else (FR-010). Scoring against that contract is
what separates the two failures people conflate:

    reference      prediction        outcome            counted in
    -----------    --------------    ---------------    -----------------------
    [UNREADABLE]   [UNREADABLE]      correct_refusal    probe set
    [UNREADABLE]   anything else     hallucination      probe set
    legible text   [UNREADABLE]      false_refusal      legible set, separately
    legible text   anything else     scored             legible set, CER/WER/EM

A false refusal carries metrics **and** its own count: it is a wrong
transcription of a legible page, so FR-001 requires it in the aggregate and
FR-010 requires it counted apart. Hallucination and false refusal are never
summed — they are different failures with different fixes (research R-008).
"""

from __future__ import annotations

from ocr_eval.normalise.tables import UNREADABLE_MARKER

SCORED = "scored"
CORRECT_REFUSAL = "correct_refusal"
HALLUCINATION = "hallucination"
FALSE_REFUSAL = "false_refusal"

OUTCOMES = (SCORED, CORRECT_REFUSAL, HALLUCINATION, FALSE_REFUSAL)

# Outcomes that carry no metrics: a probe page has no reference text to measure
# a distance against.
PROBE_OUTCOMES = (CORRECT_REFUSAL, HALLUCINATION)


def is_marker(text: str) -> bool:
    """True when the text *is* the marker and nothing else.

    Leading and trailing whitespace is stripped; everything else must match
    exactly, case included. Requiring the marker to stand alone is what makes
    the contract checkable — a model that writes
    "[UNREADABLE] — possibly a stamp?" has not honoured it, and treating that as
    a pass would let hallucination hide behind a compliant-looking prefix.
    """
    return text.strip() == UNREADABLE_MARKER


def classify(reference: str, prediction: str) -> str:
    """Place one page in the four-cell matrix."""
    reference_is_marker = is_marker(reference)
    prediction_is_marker = is_marker(prediction)

    if reference_is_marker:
        return CORRECT_REFUSAL if prediction_is_marker else HALLUCINATION
    return FALSE_REFUSAL if prediction_is_marker else SCORED


def carries_metrics(outcome: str) -> bool:
    """Whether a page with this outcome contributes to CER/WER/exact match."""
    return outcome not in PROBE_OUTCOMES
