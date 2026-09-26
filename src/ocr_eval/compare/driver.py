"""Assemble the comparison artifact and write it.

**Exit code is 0 for all three verdicts.** A REJECT is a successful comparison,
not a failed command — scripts read the verdict from the artifact, not from
`$?`. Only an incommensurable pair is a non-zero exit, and that is exit 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ocr_eval import __version__
from ocr_eval.compare import diff as diff_module
from ocr_eval.compare import verdict as verdict_module
from ocr_eval.compare.bootstrap import paired_bootstrap
from ocr_eval.errors import EXIT_OK
from ocr_eval.io import writer
from ocr_eval.report import markdown as markdown_module

# Lower is better for every metric here except exact_match_rate. The sign that
# means "better" therefore differs per metric, which is exactly why direction is
# stated as a word in the artifact rather than left to the reader.
LOWER_IS_BETTER = {
    "cer": True,
    "wer": True,
    "exact_match_rate": False,
    "hallucination_rate": True,
    "false_refusal_rate": True,
}


def _direction(metric: str, delta: float) -> str:
    if delta == 0:
        return "unchanged"
    better_when_lower = LOWER_IS_BETTER[metric]
    improved = delta < 0 if better_when_lower else delta > 0
    return "better" if improved else "worse"


def _metric_delta(metric: str, before: float, after: float) -> dict[str, object]:
    delta = after - before
    return {
        "metric": metric,
        "baseline": before,
        "candidate": after,
        "delta": delta,
        "direction": _direction(metric, delta),
    }


def _split_metrics(before: dict, after: dict) -> list[dict[str, object]]:
    """Only metrics defined on BOTH sides.

    A split whose `scored_pages` is 0 has a null headline (research R-018), so
    it contributes its contract rates and no CER delta computed from a null.
    """
    metrics: list[dict[str, object]] = []

    if before.get("headline") is not None and after.get("headline") is not None:
        for metric in ("cer", "wer", "exact_match_rate"):
            metrics.append(
                _metric_delta(metric, before["headline"][metric], after["headline"][metric])
            )

    for metric in ("hallucination_rate", "false_refusal_rate"):
        left = before["contract"][metric]
        right = after["contract"][metric]
        if left is not None and right is not None:
            metrics.append(_metric_delta(metric, left, right))

    return metrics


def build_comparison(baseline, candidate, primary_split: str, config) -> dict:
    """The whole of comparison.json."""
    deltas = diff_module.page_deltas(baseline, candidate)
    regressions = diff_module.regressions(deltas)
    improvements = diff_module.improvements(deltas)

    splits: list[dict[str, object]] = []
    bootstrap: list[dict[str, object]] = []
    for name in sorted(baseline.splits & candidate.splits):
        before = baseline.split(name)
        after = candidate.split(name)
        splits.append(
            {
                "split": name,
                "pages": before["pages"],
                "metrics": _split_metrics(before, after),
            }
        )
        split_deltas = [d.delta_cer for d in deltas if d.split == name]
        if split_deltas:
            bootstrap.append(
                paired_bootstrap(
                    name,
                    split_deltas,
                    resamples=config.bootstrap_resamples,
                    seed=config.bootstrap_seed,
                ).to_dict()
            )

    decision = verdict_module.decide(baseline, candidate, primary_split, config)

    return {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "baseline_model_version": baseline.model_version,
        "candidate_model_version": candidate.model_version,
        "policy_version": baseline.policy_version,
        "tool_version": __version__,
        "primary_split": primary_split,
        "verdict": decision.verdict,
        "verdict_reasons": decision.reasons,
        "thresholds": decision.thresholds,
        "splits": splits,
        "bootstrap": bootstrap,
        "regressions": [d.to_dict() for d in regressions],
        "improvements": [d.to_dict() for d in improvements],
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        # Always empty in a written artifact: an unequal page set is the fourth
        # incommensurability condition, so the command exits 3 instead of
        # producing this file. Present so a consumer can assert emptiness.
        "only_in_baseline": [],
        "only_in_candidate": [],
    }


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    config,
    out: str | Path | None = None,
    primary_split: str = "synthetic_heldout",
    top: int = 25,
    fmt: str = "md",
) -> int:
    baseline = diff_module.load_run(baseline_dir)
    candidate = diff_module.load_run(candidate_dir)

    # Before any computation: refusing late would mean reporting a number that
    # should never have been computed.
    diff_module.assert_commensurable(baseline, candidate)

    comparison = build_comparison(baseline, candidate, primary_split, config)

    truncated = dict(comparison)
    truncated["regressions"] = comparison["regressions"][:top]
    truncated["improvements"] = comparison["improvements"][:top]

    rendered = markdown_module.render_comparison(truncated)

    if out:
        target = Path(out)
        if target.suffix:
            writer.write_text(target, rendered if fmt == "md" else writer.dumps(truncated))
        else:
            writer.ensure_dir(target)
            writer.write_json(target / "comparison.json", truncated)
            writer.write_text(target / "comparison.md", rendered)
            print(f"written to: {target}", file=sys.stderr)
    else:
        print(writer.dumps(truncated) if fmt == "json" else rendered)

    print(
        f"verdict: {comparison['verdict']}  "
        f"({comparison['regression_count']} regression(s), "
        f"{comparison['improvement_count']} improvement(s))",
        file=sys.stderr,
    )
    return EXIT_OK
