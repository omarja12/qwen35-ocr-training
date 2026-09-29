"""The paired bootstrap is reproducible, and says when a difference is not a result."""

import pytest

from ocr_eval.compare.bootstrap import paired_bootstrap
from ocr_eval.io.writer import canonicalise

SEED = 20260916
RESAMPLES = 10000


def test_reproducible_under_the_default_seed_and_resamples():
    deltas = [-0.05, -0.02, 0.01, -0.03, -0.04, 0.02, -0.01, -0.06]
    first = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    second = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    assert first.to_dict() == second.to_dict()


def test_a_different_seed_gives_a_different_interval():
    """Seed sensitivity, demonstrated where it is actually observable.

    A continuous sample is needed here. With only eight discrete delta values
    the resampled means take so few distinct values that the 2.5% and 97.5%
    percentiles land on the same numbers whatever the seed — see
    `test_a_small_discrete_sample_has_seed_independent_endpoints` below, which
    pins that behaviour so nobody mistakes it for the seed being ignored.
    """
    import random

    rng = random.Random(1)
    deltas = [rng.gauss(-0.02, 0.05) for _ in range(60)]
    a = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    b = paired_bootstrap("s", deltas, RESAMPLES, SEED + 1)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_a_small_discrete_sample_has_seed_independent_endpoints():
    """Not a bug, and worth pinning so it is not mistaken for one.

    Eight distinct deltas resampled 10,000 times produce a coarse empirical
    distribution whose percentile endpoints are stable across seeds. The mean is
    still the mean of the data, and reproducibility is unaffected.

    Compared at the six places an artifact carries: two resamples that are equal
    in decimal can differ around 1e-19 in binary depending on which values were
    drawn, and no report can show that.
    """
    deltas = [-0.05, -0.02, 0.01, -0.03, -0.04, 0.02, -0.01, -0.06]
    a = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    b = paired_bootstrap("s", deltas, RESAMPLES, SEED + 1)
    assert canonicalise([a.ci_low, a.ci_high]) == canonicalise([b.ci_low, b.ci_high])
    assert a.delta_cer == b.delta_cer == pytest.approx(sum(deltas) / len(deltas))


def test_seed_and_resamples_are_recorded_in_the_artifact():
    """Recorded rather than pinned, so a tuned config still validates."""
    interval = paired_bootstrap("s", [-0.01, -0.02], 500, 42)
    row = interval.to_dict()
    assert row["resamples"] == 500
    assert row["seed"] == 42
    assert row["confidence"] == 0.95


def test_a_clear_improvement_does_not_straddle_zero():
    deltas = [-0.10] * 40
    interval = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    assert interval.delta_cer == pytest.approx(-0.10)
    assert not interval.ci_low <= 0.0 <= interval.ci_high
    assert interval.ci_high < 0


def test_a_marginal_difference_straddles_zero_and_is_reported_as_such():
    """A 0.4-point improvement that straddles zero is visibly not a result."""
    deltas = [-0.05, 0.05, -0.04, 0.04, -0.06, 0.06, -0.004, 0.004] * 5
    interval = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    assert interval.ci_low <= 0.0 <= interval.ci_high


def test_interval_brackets_the_observed_mean():
    deltas = [-0.05, -0.02, 0.01, -0.03]
    interval = paired_bootstrap("s", deltas, RESAMPLES, SEED)
    assert interval.ci_low <= interval.delta_cer <= interval.ci_high


def test_no_deltas_is_a_flat_interval_not_a_crash():
    interval = paired_bootstrap("s", [], RESAMPLES, SEED)
    assert interval.delta_cer == 0.0
    assert interval.ci_low == interval.ci_high == 0.0


def test_uses_only_the_standard_library_random():
    """No dependency beyond random.Random(seed) — constitution III."""
    import inspect

    from ocr_eval.compare import bootstrap

    source = inspect.getsource(bootstrap)
    assert "import random" in source
    for forbidden in ("numpy", "scipy", "pandas"):
        assert forbidden not in source
