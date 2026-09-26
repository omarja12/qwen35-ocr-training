"""Comparing two runs, and refusing to compare two that are not comparable."""

import json

import pytest

from ocr_eval.compare.driver import build_comparison, compare_runs
from ocr_eval.compare.diff import assert_commensurable, load_run
from ocr_eval.config import load_config
from ocr_eval.errors import EXIT_INCOMMENSURABLE, IncommensurableError
from ocr_eval.score.run import score_run
from ocr_eval.score.subset import parse_filter
from tests.conftest import FIXTURES, load_schema

jsonschema = pytest.importorskip("jsonschema")

MANIFEST = FIXTURES / "manifest.jsonl"


def make_run(out, predictions, model_version, **kwargs):
    assert score_run(
        manifest_path=MANIFEST,
        predictions_path=FIXTURES / "predictions" / predictions,
        model_version=model_version,
        out_dir=out,
        config=load_config(),
        **kwargs,
    ) == 0
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("compare")
    baseline = make_run(base / "baseline", "baseline", "demo-baseline")
    candidate = make_run(base / "candidate", "finetuned", "demo-ft")
    return baseline, candidate


def test_comparison_validates_against_its_schema(runs):
    baseline, candidate = runs
    comparison = build_comparison(
        load_run(baseline), load_run(candidate), "synthetic_heldout", load_config()
    )
    jsonschema.validate(comparison, load_schema("comparison.schema.json"))


def test_each_metric_states_a_direction_as_a_word(runs):
    baseline, candidate = runs
    comparison = build_comparison(
        load_run(baseline), load_run(candidate), "synthetic_heldout", load_config()
    )
    found = 0
    for split in comparison["splits"]:
        for metric in split["metrics"]:
            assert metric["direction"] in ("better", "worse", "unchanged")
            found += 1
    assert found, "no metrics were compared at all"


def test_lower_cer_is_better_and_higher_exact_match_is_better(runs):
    """The sign that means 'better' differs per metric — hence the word."""
    baseline, candidate = runs
    comparison = build_comparison(
        load_run(baseline), load_run(candidate), "synthetic_heldout", load_config()
    )
    synthetic = next(s for s in comparison["splits"] if s["split"] == "synthetic_heldout")
    by_metric = {m["metric"]: m for m in synthetic["metrics"]}

    cer = by_metric["cer"]
    assert cer["delta"] < 0 and cer["direction"] == "better"

    exact = by_metric["exact_match_rate"]
    if exact["delta"] > 0:
        assert exact["direction"] == "better"


def test_regressions_are_listed_worst_first(runs):
    baseline, candidate = runs
    # Compare in the other direction so there are regressions to list.
    comparison = build_comparison(
        load_run(candidate), load_run(baseline), "synthetic_heldout", load_config()
    )
    deltas = [r["delta_cer"] for r in comparison["regressions"]]
    assert deltas, "the reversed comparison should show regressions"
    assert deltas == sorted(deltas, reverse=True)
    assert comparison["regression_count"] == len(comparison["regressions"])


def test_verdict_reasons_are_never_empty(runs):
    baseline, candidate = runs
    for a, b in ((baseline, candidate), (candidate, baseline)):
        comparison = build_comparison(
            load_run(a), load_run(b), "synthetic_heldout", load_config()
        )
        assert comparison["verdict"] in ("PROMOTE", "REJECT", "INCONCLUSIVE")
        assert comparison["verdict_reasons"], "an INCONCLUSIVE verdict must state why"


def test_a_probe_split_contributes_contract_rates_and_no_cer_delta(runs):
    """Research R-018's knock-on: no delta is ever computed from a null."""
    baseline, candidate = runs
    comparison = build_comparison(
        load_run(baseline), load_run(candidate), "synthetic_heldout", load_config()
    )
    probe = next(s for s in comparison["splits"] if s["split"] == "probe")
    names = {m["metric"] for m in probe["metrics"]}
    assert "cer" not in names
    assert "hallucination_rate" in names


def test_exit_code_is_zero_for_every_verdict(runs, tmp_path):
    """A REJECT is a successful comparison. Scripts read the artifact, not $?."""
    baseline, candidate = runs
    assert compare_runs(baseline, candidate, config=load_config(),
                        out=tmp_path / "a") == 0
    assert compare_runs(candidate, baseline, config=load_config(),
                        out=tmp_path / "b") == 0


# --- the commensurability guard -------------------------------------------

def test_differing_policy_version_exits_3(runs, tmp_path):
    baseline, candidate = runs
    tampered = _clone_with(tmp_path / "policy", candidate, policy_version="9.9")
    with pytest.raises(IncommensurableError) as excinfo:
        assert_commensurable(load_run(baseline), load_run(tampered))
    assert excinfo.value.exit_code == EXIT_INCOMMENSURABLE
    assert any("policy_version" in p for p in excinfo.value.problems)


def test_differing_corpus_digest_exits_3(runs, tmp_path):
    baseline, candidate = runs
    tampered = _clone_with(tmp_path / "digest", candidate, corpus_manifest_digest="sha256:0")
    with pytest.raises(IncommensurableError) as excinfo:
        assert_commensurable(load_run(baseline), load_run(tampered))
    assert any("corpus_manifest_digest" in p for p in excinfo.value.problems)


def test_differing_split_set_exits_3(tmp_path):
    baseline = make_run(tmp_path / "all", "baseline", "a")
    partial = make_run(tmp_path / "some", "baseline", "b",
                       conditions=parse_filter("split=synthetic_heldout"))
    with pytest.raises(IncommensurableError) as excinfo:
        assert_commensurable(load_run(baseline), load_run(partial))
    assert any("split set differs" in p for p in excinfo.value.problems)


def test_differing_page_set_exits_3(tmp_path):
    """The case FR-018 makes reachable: every other guard passes.

    Two runs over the same corpus at the same policy version, restricted to
    different page subsets *within the same splits* — so policy_version,
    corpus_manifest_digest and split set all match and only the page set
    differs. Comparing them would report the difference between two populations
    as the difference between two models.
    """
    ids_a = ["syn_clean_001", "syn_clean_002", "gold_scan_001", "probe_blank_001"]
    ids_b = ["syn_clean_001", "syn_clean_003", "gold_scan_001", "probe_blank_001"]

    run_a = make_run(tmp_path / "subset_a", "baseline", "a", page_ids=ids_a)
    run_b = make_run(tmp_path / "subset_b", "baseline", "b", page_ids=ids_b)

    loaded_a, loaded_b = load_run(run_a), load_run(run_b)
    # Everything except the page set matches — that is the whole point.
    assert loaded_a.policy_version == loaded_b.policy_version
    assert (loaded_a.manifest["corpus_manifest_digest"]
            == loaded_b.manifest["corpus_manifest_digest"])
    assert loaded_a.splits == loaded_b.splits

    with pytest.raises(IncommensurableError) as excinfo:
        assert_commensurable(loaded_a, loaded_b)
    assert excinfo.value.exit_code == EXIT_INCOMMENSURABLE
    joined = " ".join(excinfo.value.problems)
    assert "page set differs" in joined
    # The offending ids are named, not just counted.
    assert "syn_clean_002" in joined and "syn_clean_003" in joined


def test_there_is_no_warn_and_continue_path(runs, tmp_path):
    """The guard runs before any computation, and raises rather than warning."""
    baseline, candidate = runs
    tampered = _clone_with(tmp_path / "nowarn", candidate, policy_version="9.9")
    with pytest.raises(IncommensurableError):
        compare_runs(baseline, tampered, config=load_config(), out=tmp_path / "out")
    assert not (tmp_path / "out").exists(), "nothing may be written on a refusal"


def _clone_with(destination, source, **manifest_overrides):
    """Copy a run directory, tampering with manifest fields."""
    import shutil

    shutil.copytree(source, destination)
    manifest_file = destination / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for key, value in manifest_overrides.items():
        if key == "policy_version":
            manifest["policy"]["policy_version"] = value
        else:
            manifest[key] = value
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
    )
    return destination
