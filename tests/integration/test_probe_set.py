"""Hallucination on unreadable input, and the line between it and false refusal.

The two failures are never summed. Inventing text on a blank page and refusing a
readable one need different fixes, and a combined "contract violation rate"
would hide which one a checkpoint actually has.
"""

import json

import pytest

from ocr_eval.config import load_config
from ocr_eval.score.run import score_run
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("probe") / "baseline"
    assert score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "baseline",
        model_version="probe-test",
        out_dir=out,
        config=load_config(),
    ) == 0
    return out


def summary_of(run_dir):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def pages_of(run_dir):
    return {
        json.loads(line)["page_id"]: json.loads(line)
        for line in (run_dir / "pages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_probe_pages_are_excluded_from_cer_wer_and_exact_match(baseline_run):
    summary = summary_of(baseline_run)
    probe = next(s for s in summary["splits"] if s["split"] == "probe")

    assert probe["pages"] == 4
    assert probe["scored_pages"] == 0
    assert probe["headline"] is None
    assert probe["metrics"] is None

    for row in pages_of(baseline_run).values():
        if row["outcome"] in ("correct_refusal", "hallucination"):
            assert "metrics" not in row


def test_hallucination_rate_is_a_single_percentage_for_the_checkpoint(baseline_run):
    """SC-007."""
    probe = next(s for s in summary_of(baseline_run)["splits"] if s["split"] == "probe")
    contract = probe["contract"]

    assert contract["probe_pages"] == 4
    assert contract["hallucination"] == 2
    assert contract["hallucination_rate"] == pytest.approx(0.5)


def test_false_refusals_appear_in_cer_and_in_their_own_count(baseline_run):
    """FR-001 wants the page in the aggregate; FR-010 wants the separate count."""
    pages = pages_of(baseline_run)
    page = pages["syn_false_refusal_001"]

    assert page["outcome"] == "false_refusal"
    assert "metrics" in page, "a false refusal is a wrong transcription of a legible page"
    assert page["metrics"]["strict"]["cer"] > 0

    synthetic = next(
        s for s in summary_of(baseline_run)["splits"] if s["split"] == "synthetic_heldout"
    )
    assert synthetic["contract"]["false_refusal"] == 1
    assert synthetic["contract"]["false_refusal_rate"] is not None
    # The page is inside the scored population, not beside it.
    assert synthetic["scored_pages"] == synthetic["pages"]


def test_hallucination_and_false_refusal_are_never_summed(baseline_run):
    for split in summary_of(baseline_run)["splits"]:
        contract = split["contract"]
        assert "contract_violation_rate" not in contract
        assert "total_violations" not in contract
        # Separate denominators, so a sum would be meaningless as well as wrong.
        assert set(contract) == {
            "correct_refusal", "hallucination", "false_refusal",
            "legible_pages", "probe_pages",
            "hallucination_rate", "false_refusal_rate",
        }


def test_rates_are_null_not_zero_when_their_denominator_is_empty(baseline_run):
    summary = summary_of(baseline_run)
    probe = next(s for s in summary["splits"] if s["split"] == "probe")
    synthetic = next(s for s in summary["splits"] if s["split"] == "synthetic_heldout")

    # No legible pages in the probe split.
    assert probe["contract"]["legible_pages"] == 0
    assert probe["contract"]["false_refusal_rate"] is None

    # No probe pages in the synthetic split.
    assert synthetic["contract"]["probe_pages"] == 0
    assert synthetic["contract"]["hallucination_rate"] is None


def test_a_marker_with_a_suffix_counts_as_hallucination(baseline_run):
    """The fixture carries this case deliberately (research R-008)."""
    page = pages_of(baseline_run)["probe_photo_004"]
    assert page["outcome"] == "hallucination"


def test_all_four_contract_cells_have_a_fixture(baseline_run):
    outcomes = {row["outcome"] for row in pages_of(baseline_run).values()}
    assert outcomes == {"scored", "correct_refusal", "hallucination", "false_refusal"}


def test_every_flag_has_a_fixture(baseline_run):
    fired = {flag for row in pages_of(baseline_run).values() for flag in row["flags"]}
    assert {"runaway_length", "repetition", "empty_prediction"} <= fired


def test_the_finetuned_set_improves_the_hallucination_rate(tmp_path):
    """The measurement has to be able to move, or it is not measuring anything."""
    out = tmp_path / "finetuned"
    assert score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "finetuned",
        model_version="probe-test-ft",
        out_dir=out,
        config=load_config(),
    ) == 0
    probe = next(s for s in summary_of(out)["splits"] if s["split"] == "probe")
    assert probe["contract"]["hallucination_rate"] == pytest.approx(0.25)


def test_summary_markdown_shows_the_two_figures_visibly_apart(baseline_run):
    text = (baseline_run / "summary.md").read_text(encoding="utf-8")
    assert "hallucination" in text
    assert "false refusal" in text
    assert "never summed" in text
