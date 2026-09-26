"""Scoring the fixture corpus produces a complete, contract-valid run directory."""

import json

import pytest

from ocr_eval.config import load_config
from ocr_eval.score.run import score_run

jsonschema = pytest.importorskip("jsonschema")

from tests.conftest import load_schema  # noqa: E402

EXPECTED_FILES = {
    "manifest.json",
    "pages.jsonl",
    "summary.json",
    "summary.md",
    "worst_pages.md",
    "worst_pages.html",
    "run.meta.json",
}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from tests.conftest import FIXTURES

    out = tmp_path_factory.mktemp("runs") / "baseline"
    code = score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "baseline",
        model_version="test-baseline",
        out_dir=out,
        config=load_config(),
    )
    assert code == 0
    return out


def read_pages(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "pages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_every_expected_file_is_written(run_dir):
    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES


def test_page_rows_validate_against_their_schema(run_dir):
    schema = load_schema("page-result.schema.json")
    rows = read_pages(run_dir)
    assert rows
    for row in rows:
        jsonschema.validate(row, schema)


def test_summary_validates_against_its_schema(run_dir):
    schema = load_schema("run-summary.schema.json")
    jsonschema.validate(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
                        schema)


def test_rows_are_sorted_by_page_id_in_codepoint_order(run_dir):
    ids = [row["page_id"] for row in read_pages(run_dir)]
    assert ids == sorted(ids)


def test_no_cross_split_total_key_exists_anywhere(run_dir):
    """FR-013's guarantee is the absence of the number, not a caveat about it."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"run_id", "model_version", "tool_version", "policy",
                            "splits", "notes"}
    for forbidden in ("total", "overall", "all", "combined", "aggregate"):
        assert forbidden not in summary


def test_probe_split_reports_null_not_zero(run_dir):
    """Research R-018. A zero here would read as a perfect score."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    probe = next(s for s in summary["splits"] if s["split"] == "probe")

    assert probe["scored_pages"] == 0
    assert probe["headline"] is None
    assert probe["metrics"] is None
    assert probe["order"]["order_corrected_cer_micro"] is None
    assert probe["breakdowns"] == []

    # The numbers such a split genuinely has are still reported in full.
    assert probe["contract"]["probe_pages"] == 4
    assert probe["contract"]["hallucination_rate"] is not None
    assert probe["pages"] == 4


def test_scoreable_split_reports_a_real_headline(run_dir):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    synthetic = next(s for s in summary["splits"] if s["split"] == "synthetic_heldout")
    assert synthetic["headline"]["level"] == "strict"
    assert synthetic["headline"]["cer"] > 0
    assert synthetic["scored_pages"] == synthetic["pages"]


def test_probe_pages_carry_no_metrics_and_legible_pages_do(run_dir):
    for row in read_pages(run_dir):
        if row["outcome"] in ("correct_refusal", "hallucination"):
            assert "metrics" not in row, row["page_id"]
        else:
            assert "metrics" in row, row["page_id"]
            assert set(row["metrics"]) == {"strict", "diacritic_insensitive", "skeleton"}


def test_false_refusal_is_in_the_aggregate_and_counted_separately(run_dir):
    rows = {row["page_id"]: row for row in read_pages(run_dir)}
    page = rows["syn_false_refusal_001"]
    assert page["outcome"] == "false_refusal"
    assert "metrics" in page, "a false refusal is a wrong transcription and belongs in CER"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    synthetic = next(s for s in summary["splits"] if s["split"] == "synthetic_heldout")
    assert synthetic["contract"]["false_refusal"] == 1
    assert synthetic["contract"]["false_refusal_rate"] is not None


def test_manifest_records_what_a_stranger_needs(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for key in ("run_id", "model_version", "corpus_manifest_digest",
                "predictions_digest", "pages_scored", "policy", "config",
                "tool_version"):
        assert key in manifest, key
    assert manifest["policy"]["unreadable_marker"] == "[UNREADABLE]"
    # Every threshold in force, echoed verbatim.
    assert manifest["config"]["bootstrap_seed"] == 20260916


def test_run_id_is_derived_from_inputs_not_from_a_timestamp(run_dir):
    from ocr_eval.score.run import derive_run_id

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == derive_run_id(
        manifest["model_version"],
        manifest["corpus_manifest_digest"],
        manifest["predictions_digest"],
    )


def test_existing_run_directory_is_refused_with_exit_4(run_dir):
    from ocr_eval.errors import EXIT_OUTPUT_COLLISION, OutputCollisionError
    from tests.conftest import FIXTURES

    with pytest.raises(OutputCollisionError) as excinfo:
        score_run(
            manifest_path=FIXTURES / "manifest.jsonl",
            predictions_path=FIXTURES / "predictions" / "baseline",
            model_version="test-baseline",
            out_dir=run_dir,
            config=load_config(),
        )
    assert excinfo.value.exit_code == EXIT_OUTPUT_COLLISION


def test_force_allows_overwriting(run_dir):
    from tests.conftest import FIXTURES

    code = score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "baseline",
        model_version="test-baseline",
        out_dir=run_dir,
        config=load_config(),
        force=True,
    )
    assert code == 0


def test_summary_markdown_shows_all_three_levels_with_strict_labelled_raw(run_dir):
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "strict (raw, unnormalised)" in text
    assert "diacritic-insensitive" in text
    assert "skeleton" in text


def test_synthetic_only_caveat_is_absent_when_gold_scans_are_present(run_dir):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert any(s["split"] == "gold_scans" for s in summary["splits"])
    assert summary["notes"] == []


def test_synthetic_only_caveat_appears_when_gold_scans_are_absent(tmp_path):
    from ocr_eval.score.subset import parse_filter
    from tests.conftest import FIXTURES

    out = tmp_path / "synthetic-only"
    score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "baseline",
        model_version="test-baseline",
        out_dir=out,
        config=load_config(),
        conditions=parse_filter("split=synthetic_heldout"),
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["notes"], "a synthetic-only run must say so, verbatim, every time"
    assert "overstates real-world quality" in summary["notes"][0]
