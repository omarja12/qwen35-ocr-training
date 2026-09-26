"""CER, WER, exact match, and corpus aggregation.

Pure functions over strings. Everything that produces a number in a report is
here or in a sibling module, which is what keeps those numbers trivially
testable — the constitution requires a unit test for every one of them.

Edit counts and denominators travel alongside the rates because corpus
aggregation is a **sum**, never an average of averages (research R-005). Micro
and macro are both reported: micro is the corpus-level truth, macro is dominated
by short pages and reveals failures clustering on them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

from ocr_eval.metrics.distance import levenshtein, levenshtein_sequences


def tokenise(text: str) -> list[str]:
    """Words are runs of non-whitespace, empty tokens discarded.

    `str.split()` with no argument splits on runs of Unicode whitespace and
    drops empties, which is the rule verbatim. No punctuation splitting, no
    morphological segmentation — that would import an Arabic NLP stack into an
    air-gapped image to slightly improve a secondary metric (research R-006).
    """
    return text.split()


@dataclass(frozen=True)
class LevelMetrics:
    """One normalisation level's numbers for one page."""

    cer: float
    wer: float
    exact_match: bool
    char_edits: int
    ref_chars: int
    word_edits: int
    ref_words: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_level(reference: str, prediction: str) -> LevelMetrics:
    """Score one page at one already-normalised level.

    Both denominators are guaranteed non-zero by `validate`, which rejects any
    non-probe reference that normalises to empty at any level — a
    punctuation-only reference empties at skeleton, a diacritics-only one at
    diacritic-insensitive, and either would divide by zero here (research
    R-009). The assertions below state that contract rather than defending
    against it: if one ever fires, the validator has a hole and silently
    returning 0.0 would hide it.
    """
    ref_chars = len(reference)
    ref_words_list = tokenise(reference)
    ref_words = len(ref_words_list)

    if ref_chars == 0 or ref_words == 0:
        raise ValueError(
            "reference normalised to empty; validate should have rejected this "
            "page upstream (research R-009)"
        )

    char_edits = levenshtein(reference, prediction)
    word_edits = levenshtein_sequences(ref_words_list, tokenise(prediction))

    return LevelMetrics(
        cer=char_edits / ref_chars,
        wer=word_edits / ref_words,
        exact_match=reference == prediction,
        char_edits=char_edits,
        ref_chars=ref_chars,
        word_edits=word_edits,
        ref_words=ref_words,
    )


@dataclass(frozen=True)
class Aggregate:
    """Micro and macro for one rate over a set of pages."""

    micro: float
    macro: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def aggregate_rate(edits: Sequence[int], denominators: Sequence[int]) -> Aggregate:
    """Micro is sum(edits)/sum(denominator); macro is the mean of per-page rates."""
    if not edits:
        return Aggregate(micro=0.0, macro=0.0)
    total_denominator = sum(denominators)
    micro = sum(edits) / total_denominator if total_denominator else 0.0
    per_page = [e / d for e, d in zip(edits, denominators) if d]
    macro = sum(per_page) / len(per_page) if per_page else 0.0
    return Aggregate(micro=micro, macro=macro)


def aggregate_level(metrics: Iterable[LevelMetrics]) -> dict[str, object]:
    """Aggregate one level across pages into the run-summary shape."""
    items = list(metrics)
    if not items:
        return {
            "cer": Aggregate(0.0, 0.0).to_dict(),
            "wer": Aggregate(0.0, 0.0).to_dict(),
            "exact_match_rate": 0.0,
        }
    cer = aggregate_rate(
        [m.char_edits for m in items], [m.ref_chars for m in items]
    )
    wer = aggregate_rate(
        [m.word_edits for m in items], [m.ref_words for m in items]
    )
    exact = sum(1 for m in items if m.exact_match) / len(items)
    return {
        "cer": cer.to_dict(),
        "wer": wer.to_dict(),
        "exact_match_rate": exact,
    }
