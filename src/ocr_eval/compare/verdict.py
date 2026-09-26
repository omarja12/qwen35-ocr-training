"""PROMOTE, REJECT or INCONCLUSIVE, from a stated rule.

SC-006 wants a reviewer to act without opening the underlying data, which means
the rule has to be **stated**, not felt. Every threshold comes from config and
is echoed into the artifact beside the verdict, so the decision is
reconstructable later.

Making gold-set regression an automatic reject encodes the project's actual
risk: synthetic gains that do not survive real scans. A single-threshold rule on
CER alone would promote a checkpoint that reads 0.5 points better and
hallucinates twice as often — the trade the spec calls the most dangerous
failure mode (research R-014).
"""

from __future__ import annotations

from dataclasses import dataclass

PROMOTE = "PROMOTE"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"

GOLD_SPLIT = "gold_scans"


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reasons: list[str]
    thresholds: dict[str, object]


def _pp(value: float) -> float:
    """Percentage points from a rate."""
    return value * 100.0


def _headline_cer(split: dict | None) -> float | None:
    if split is None or split.get("headline") is None:
        return None
    return split["headline"]["cer"]


def decide(baseline, candidate, primary_split: str, config) -> Verdict:
    """Apply the rule. `verdict_reasons` is never empty."""
    thresholds = {
        "promote_min_cer_improvement_pp": config.promote_min_cer_improvement_pp,
        "reject_on_any_gold_regression": config.reject_on_any_gold_regression,
        "reject_hallucination_worsening_pp": config.reject_hallucination_worsening_pp,
        "reject_false_refusal_worsening_pp": config.reject_false_refusal_worsening_pp,
    }
    reasons: list[str] = []
    rejected = False

    # --- REJECT conditions, evaluated first and all of them reported --------
    if config.reject_on_any_gold_regression:
        base_gold = _headline_cer(baseline.split(GOLD_SPLIT))
        cand_gold = _headline_cer(candidate.split(GOLD_SPLIT))
        if base_gold is not None and cand_gold is not None:
            if cand_gold > base_gold:
                rejected = True
                reasons.append(
                    f"REJECT: strict CER on {GOLD_SPLIT} worsened by "
                    f"{_pp(cand_gold - base_gold):.2f}pp "
                    f"({base_gold:.4f} -> {cand_gold:.4f}). Any gold-set "
                    f"regression rejects: synthetic gains that do not survive "
                    f"real scans are the project's actual risk"
                )
        elif base_gold is None and cand_gold is None:
            reasons.append(
                f"note: no {GOLD_SPLIT} split in either run, so the gold-set "
                f"regression rule could not be applied. Until a gold set exists, "
                f"every verdict here rests on synthetic pages alone"
            )

    for label, key, limit_key in (
        ("hallucination", "hallucination_rate", "reject_hallucination_worsening_pp"),
        ("false-refusal", "false_refusal_rate", "reject_false_refusal_worsening_pp"),
    ):
        limit = thresholds[limit_key]
        worsening = _worst_rate_change(baseline, candidate, key)
        if worsening is not None and _pp(worsening) > limit:
            rejected = True
            reasons.append(
                f"REJECT: {label} rate worsened by {_pp(worsening):.2f}pp "
                f"(limit {limit:.2f}pp)"
            )

    # --- PROMOTE condition --------------------------------------------------
    base_primary = _headline_cer(baseline.split(primary_split))
    cand_primary = _headline_cer(candidate.split(primary_split))

    if base_primary is None or cand_primary is None:
        reasons.append(
            f"primary split {primary_split!r} has no strict headline in "
            f"{'baseline' if base_primary is None else 'candidate'}, so the "
            f"PROMOTE rule could not be applied"
        )
        if rejected:
            return Verdict(REJECT, reasons, thresholds)
        return Verdict(INCONCLUSIVE, reasons, thresholds)

    improvement_pp = _pp(base_primary - cand_primary)
    required = config.promote_min_cer_improvement_pp
    direction = "improved" if improvement_pp > 0 else "worsened"
    reasons.append(
        f"strict micro CER on {primary_split} {direction} by "
        f"{abs(improvement_pp):.2f}pp ({base_primary:.4f} -> {cand_primary:.4f}); "
        f">= {required:.2f}pp improvement required for PROMOTE"
    )

    if rejected:
        return Verdict(REJECT, reasons, thresholds)

    if improvement_pp >= required:
        return Verdict(PROMOTE, reasons, thresholds)

    reasons.append(
        f"INCONCLUSIVE: no reject condition fired, but the improvement did not "
        f"reach the {required:.2f}pp bar"
    )
    return Verdict(INCONCLUSIVE, reasons, thresholds)


def _worst_rate_change(baseline, candidate, key: str) -> float | None:
    """The largest worsening of a contract rate across shared splits."""
    worst: float | None = None
    for split in sorted(baseline.splits & candidate.splits):
        before = baseline.split(split)["contract"][key]
        after = candidate.split(split)["contract"][key]
        if before is None or after is None:
            continue
        change = after - before
        if worst is None or change > worst:
            worst = change
    return worst
