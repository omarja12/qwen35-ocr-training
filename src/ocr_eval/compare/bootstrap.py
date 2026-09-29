"""Paired bootstrap over per-page CER deltas.

So that a 0.4-point improvement whose interval straddles zero is visibly not a
result. Cheap — the per-page deltas are already computed — needs nothing beyond
`random.Random(seed)`, and is deterministic because the seed is fixed and
recorded (research R-014).

`resamples` and `seed` come from config and are echoed into the artifact.
contracts/comparison.schema.json records them rather than pinning them, so a
tuned config cannot produce output that fails its own contract.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Sequence

from ocr_eval.config import BOOTSTRAP_CONFIDENCE


@dataclass(frozen=True)
class Interval:
    split: str
    delta_cer: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def paired_bootstrap(
    split: str,
    deltas: Sequence[float],
    resamples: int,
    seed: int,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Interval:
    """Percentile interval on the mean per-page CER delta."""
    if not deltas:
        return Interval(split, 0.0, 0.0, 0.0, confidence, resamples, seed)

    observed = sum(deltas) / len(deltas)

    # Seeded per call, so the interval for a given split does not depend on how
    # many splits were processed before it.
    # rng.choices draws the whole resample in C. It consumes the random stream
    # differently from the per-draw randrange loop used before tool 0.2.0, so
    # intervals from 0.1.0 artifacts do not reproduce under the same seed.
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, int(tail * resamples))
    high_index = min(resamples - 1, int((1.0 - tail) * resamples))

    return Interval(
        split=split,
        delta_cer=observed,
        ci_low=means[low_index],
        ci_high=means[high_index],
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
