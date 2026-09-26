"""Per-split aggregates. Read at a glance (FR-005).

Two rules in this module are guarantees rather than conveniences:

**No cross-split total key may exist.** FR-013 forbids merging real scans with
synthetic pages into one headline, and the strongest form of that guarantee is
that the merged number does not exist anywhere in the artifact, so it cannot be
quoted by accident (research R-013).

**A split with no scoreable page reports `null`, never `0`.** Probe pages are
excluded from CER/WER/exact-match entirely, so a probe-only split has no
denominator at all. Writing `0.0` there would read as a *perfect* score on the
one split containing nothing scoreable — a defaulted number that changes a
reported result, which constitution principle IV forbids in as many words
(research R-018).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from ocr_eval.aggregate import breakdown as breakdown_module
from ocr_eval.metrics.flags import FLAG_NAMES
from ocr_eval.metrics.rates import aggregate_level
from ocr_eval.normalise.tables import LEVELS
from ocr_eval.score import contract as contract_module
from ocr_eval.score.page import PageResult

SYNTHETIC_ONLY_NOTE = (
    "No gold_scans split is present in this run. Every number here is measured "
    "on synthetic pages and overstates real-world quality. A gold set of "
    "manually verified real scanned pages does not yet exist; until it does, "
    "these figures are an upper bound, not an estimate."
)


def _split_summary(
    split: str,
    results: Sequence[PageResult],
    min_pages_per_breakdown_row: int,
) -> dict[str, object]:
    scored = [r for r in results if r.carries_metrics]

    outcomes = Counter(r.outcome for r in results)
    probe_pages = sum(
        1 for r in results if not contract_module.carries_metrics(r.outcome)
    )
    legible_pages = len(scored)

    contract_block = {
        "correct_refusal": outcomes.get(contract_module.CORRECT_REFUSAL, 0),
        "hallucination": outcomes.get(contract_module.HALLUCINATION, 0),
        "false_refusal": outcomes.get(contract_module.FALSE_REFUSAL, 0),
        "legible_pages": legible_pages,
        "probe_pages": probe_pages,
        # null rather than 0 when the denominator is empty. The two rates are
        # never summed: they are different failures with different fixes.
        "hallucination_rate": (
            outcomes.get(contract_module.HALLUCINATION, 0) / probe_pages
            if probe_pages
            else None
        ),
        "false_refusal_rate": (
            outcomes.get(contract_module.FALSE_REFUSAL, 0) / legible_pages
            if legible_pages
            else None
        ),
    }

    flag_counts = Counter(flag for r in results for flag in r.flags)
    flags_block = {name: flag_counts.get(name, 0) for name in FLAG_NAMES}

    # Pages with a metrics block but no usable order signal: fewer than two
    # matched lines, so reading_order_accuracy is null for them. Probe pages are
    # not counted — they carry no order block at all, and counting them would
    # make this number a mix of "could not be measured" and "was not measured".
    pages_without_order_signal = sum(
        1 for r in scored if r.order is None or r.order.reading_order_accuracy is None
    )

    if not scored:
        return {
            "split": split,
            "pages": len(results),
            "scored_pages": 0,
            "headline": None,
            "metrics": None,
            "order": {
                "order_corrected_cer_micro": None,
                "reading_order_accuracy": None,
                "pages_without_order_signal": 0,
            },
            "contract": contract_block,
            "flags": flags_block,
            "breakdowns": [],
        }

    per_level = {
        level: aggregate_level([r.metrics[level] for r in scored]) for level in LEVELS
    }
    strict = per_level["strict"]

    order_edits = sum(r.order.char_edits for r in scored if r.order)
    order_chars = sum(r.order.ref_chars for r in scored if r.order)
    concordant = sum(r.order.concordant_pairs for r in scored if r.order)
    total_pairs = sum(r.order.total_pairs for r in scored if r.order)

    return {
        "split": split,
        "pages": len(results),
        "scored_pages": len(scored),
        "headline": {
            "level": "strict",
            "cer": strict["cer"]["micro"],
            "wer": strict["wer"]["micro"],
            "exact_match_rate": strict["exact_match_rate"],
        },
        "metrics": per_level,
        "order": {
            "order_corrected_cer_micro": (
                order_edits / order_chars if order_chars else None
            ),
            "reading_order_accuracy": (
                concordant / total_pairs if total_pairs else None
            ),
            "pages_without_order_signal": pages_without_order_signal,
        },
        "contract": contract_block,
        "flags": flags_block,
        "breakdowns": breakdown_module.build_all(scored, min_pages_per_breakdown_row),
    }


def build_summary(
    results: Iterable[PageResult],
    run_id: str,
    model_version: str,
    tool_version: str,
    policy: dict[str, object],
    min_pages_per_breakdown_row: int,
) -> dict[str, object]:
    """The whole of summary.json."""
    materialised = list(results)

    by_split: dict[str, list[PageResult]] = {}
    for result in materialised:
        by_split.setdefault(result.split, []).append(result)

    splits = [
        _split_summary(split, by_split[split], min_pages_per_breakdown_row)
        for split in sorted(by_split)
    ]

    notes: list[str] = []
    if "gold_scans" not in by_split:
        # Verbatim, every time it applies. The caveat outlives this run.
        notes.append(SYNTHETIC_ONLY_NOTE)

    return {
        "run_id": run_id,
        "model_version": model_version,
        "tool_version": tool_version,
        "policy": policy,
        "splits": splits,
        "notes": notes,
    }
