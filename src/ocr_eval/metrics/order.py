"""Reading order: recognition failure told apart from sequencing failure.

FR-012a is the actual requirement — a reader must be able to distinguish a model
that cannot read from one that reads fine and orders badly, because the two need
different fixes. Three numbers side by side do that:

    high CER + low order-corrected CER + low order accuracy  -> sequencing
    high CER + high order-corrected CER                      -> recognition

The headline CER is always computed on the model's raw output, order errors
included, because that is what a downstream consumer actually receives (FR-012).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ocr_eval.metrics.distance import levenshtein, similarity
from ocr_eval.normalise.levels import skeleton


@dataclass(frozen=True)
class OrderResult:
    order_corrected_cer: float
    reading_order_accuracy: float | None
    matched_lines: int
    ref_lines: int
    pred_lines: int
    # Kept in memory for micro-averaging at split level and deliberately NOT
    # serialised: contracts/page-result.schema.json closes the `order` object
    # with additionalProperties:false, so emitting them would make every page
    # row fail its own contract. The summary is written from these objects in
    # the same process, so the aggregate never has to reconstruct edits from a
    # rounded rate.
    char_edits: int = 0
    ref_chars: int = 0
    # Concordant/total matched pairs, for micro-averaging accuracy across pages.
    concordant_pairs: int = 0
    total_pairs: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "order_corrected_cer": self.order_corrected_cer,
            "reading_order_accuracy": self.reading_order_accuracy,
            "matched_lines": self.matched_lines,
            "ref_lines": self.ref_lines,
            "pred_lines": self.pred_lines,
        }


def _non_empty_lines(text: str) -> list[str]:
    """The page element is a non-empty line of the transcription (research R-007)."""
    return [line for line in text.split("\n") if line.strip()]


def measure(reference: str, prediction: str, min_similarity: float) -> OrderResult:
    """Match lines, then report the two order numbers."""
    ref_lines = _non_empty_lines(reference)
    pred_lines = _non_empty_lines(prediction)

    if not ref_lines or not pred_lines:
        edits = levenshtein(reference, prediction)
        return OrderResult(
            order_corrected_cer=_cer(edits, reference),
            reading_order_accuracy=None,
            matched_lines=0,
            ref_lines=len(ref_lines),
            pred_lines=len(pred_lines),
            char_edits=edits,
            ref_chars=len(reference),
        )

    ref_keys = [skeleton(line) for line in ref_lines]
    pred_keys = [skeleton(line) for line in pred_lines]

    # Similarity matrix over skeleton-normalised lines, so a line that is both
    # misordered and slightly misread still pairs with its counterpart.
    #
    # The length-difference test before each distance call is an exact
    # short-circuit, not an approximation: edit distance is at least the
    # difference in length, so when that difference alone already drops
    # similarity below the floor, the pair cannot qualify and the distance need
    # not be computed. This is the page's hot loop — O(ref_lines x pred_lines)
    # — and at 50 lines a side it removes most of the work without changing a
    # single reported number.
    candidates: list[tuple[float, int, int]] = []
    for ri, ref_key in enumerate(ref_keys):
        ref_len = len(ref_key)
        for pi, pred_key in enumerate(pred_keys):
            longest = max(ref_len, len(pred_key))
            if longest and (abs(ref_len - len(pred_key)) / longest) > (1.0 - min_similarity):
                continue
            score = similarity(ref_key, pred_key)
            if score >= min_similarity:
                candidates.append((score, ri, pi))

    # Greedy, with a total order on the sort key so ties resolve identically on
    # every run. Determinism here is not a nicety: SC-003 requires byte-identical
    # output, and an unstable match would change the order-corrected CER.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched: list[tuple[int, int]] = []
    used_ref: set[int] = set()
    used_pred: set[int] = set()
    for _score, ri, pi in candidates:
        if ri in used_ref or pi in used_pred:
            continue
        used_ref.add(ri)
        used_pred.add(pi)
        matched.append((ri, pi))

    rebuilt = _rebuild(pred_lines, matched, used_pred)
    concordant, total = _concordance(matched)
    edits = levenshtein(reference, rebuilt)

    return OrderResult(
        order_corrected_cer=_cer(edits, reference),
        reading_order_accuracy=(concordant / total) if total else None,
        matched_lines=len(matched),
        ref_lines=len(ref_lines),
        pred_lines=len(pred_lines),
        char_edits=edits,
        ref_chars=len(reference),
        concordant_pairs=concordant,
        total_pairs=total,
    )


def _rebuild(
    pred_lines: list[str],
    matched: list[tuple[int, int]],
    used_pred: set[int],
) -> str:
    """Matched prediction lines in reference order, then the unmatched ones.

    Scoring this rebuilt text measures recognition with sequencing removed.
    Unmatched prediction lines are kept, in their original order, because
    dropping them would turn spurious output into a free pass.
    """
    in_reference_order = [pred_lines[pi] for _ri, pi in sorted(matched)]
    leftovers = [line for i, line in enumerate(pred_lines) if i not in used_pred]
    return "\n".join(in_reference_order + leftovers)


def _concordance(matched: list[tuple[int, int]]) -> tuple[int, int]:
    """Concordant and total unordered pairs of matched lines.

    Kendall concordance, which the caller maps to [0, 1]. Fewer than two matched
    lines yields `(0, 0)` and the caller reports `None`, **never 1.0**: a page
    with one matched line has no order to get right or wrong, and a free 1.0
    would inflate every aggregate it entered.

    Returning the raw counts rather than the ratio is what lets the split-level
    figure be a micro-average over all matched pairs in the split, instead of a
    mean of per-page means weighted by nothing in particular.
    """
    pairs = list(combinations(matched, 2))
    concordant = sum(1 for (ra, pa), (rb, pb) in pairs if (ra - rb) * (pa - pb) > 0)
    return concordant, len(pairs)


def _cer(edits: int, reference: str) -> float:
    return edits / len(reference) if reference else 0.0
