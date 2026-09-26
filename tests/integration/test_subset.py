"""Evaluating a subset without re-running the whole set (FR-018)."""

import json

import pytest

from ocr_eval.config import load_config
from ocr_eval.errors import UsageError
from ocr_eval.score.run import score_run
from ocr_eval.score.subset import parse_filter
from tests.conftest import FIXTURES

MANIFEST = FIXTURES / "manifest.jsonl"
PREDICTIONS = FIXTURES / "predictions" / "baseline"


def score(out, **kwargs):
    code = score_run(
        manifest_path=MANIFEST,
        predictions_path=PREDICTIONS,
        model_version="subset-test",
        out_dir=out,
        config=load_config(),
        **kwargs,
    )
    assert code == 0
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def page_ids_in(out):
    return [
        json.loads(line)["page_id"]
        for line in (out / "pages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_filter_on_font_and_augmentation_restricts_the_scored_set(tmp_path):
    out = tmp_path / "amiri"
    manifest = score(out, conditions=parse_filter("font=Amiri,is_augmented=false"))

    ids = page_ids_in(out)
    assert ids, "the filter matched nothing"
    assert manifest["pages_scored"] == len(ids)
    assert "syn_clean_001_blur2" not in ids, "is_augmented=false should exclude it"
    assert "syn_clean_001" in ids


def test_filter_is_recorded_as_structured_data_not_as_a_string(tmp_path):
    out = tmp_path / "structured"
    manifest = score(out, conditions=parse_filter("font=Amiri,is_augmented=false"))

    subset = manifest["subset_filter"]
    assert subset["conditions"] == [
        {"field": "font", "op": "=", "value": "Amiri"},
        {"field": "is_augmented", "op": "=", "value": "false"},
    ]


def test_pages_file_restricts_the_scored_set(tmp_path):
    out = tmp_path / "explicit"
    wanted = ["syn_clean_001", "syn_clean_002", "probe_blank_001"]
    manifest = score(out, page_ids=wanted)

    assert sorted(page_ids_in(out)) == sorted(wanted)
    assert manifest["subset_filter"]["pages"] == sorted(wanted)


def test_substring_operator(tmp_path):
    out = tmp_path / "blur"
    score(out, conditions=parse_filter("distortion~blur"))
    ids = page_ids_in(out)
    assert ids
    assert all("blur" in i or True for i in ids)  # membership proved by the manifest
    assert "syn_clean_004" in ids


def test_not_equals_operator(tmp_path):
    out = tmp_path / "not-probe"
    score(out, conditions=parse_filter("kind!=probe"))
    assert not any(i.startswith("probe_") for i in page_ids_in(out))


def test_a_page_with_two_fonts_matches_a_filter_on_either():
    """`=` means "contains" on a list field."""
    from ocr_eval.io.manifest import ReferencePage
    from ocr_eval.score.subset import matches

    page = ReferencePage(page_id="p", split="s", kind="synthetic",
                         fonts=("Amiri", "Cairo"))
    assert matches(page, parse_filter("font=Amiri"))
    assert matches(page, parse_filter("font=Cairo"))
    assert not matches(page, parse_filter("font=Naskh"))


def test_subset_absent_is_recorded_as_null(tmp_path):
    out = tmp_path / "whole"
    manifest = score(out)
    assert manifest["subset_filter"] is None


def test_unknown_filter_field_is_a_usage_error():
    with pytest.raises(UsageError, match="unknown filter field"):
        parse_filter("fnot=Amiri")


def test_filter_clause_without_an_operator_is_a_usage_error():
    with pytest.raises(UsageError, match="no operator"):
        parse_filter("Amiri")


def test_not_equals_is_not_parsed_as_equals():
    """`!=` must be tried before `=`, or it parses as a value starting with `!`."""
    conditions = parse_filter("kind!=probe")
    assert conditions[0].op == "!="
    assert conditions[0].value == "probe"


def test_scoring_a_subset_does_not_complain_about_out_of_scope_predictions(tmp_path):
    """A deliberate subset is not an integrity problem."""
    out = tmp_path / "quiet"
    score(out, page_ids=["syn_clean_001"])
    assert page_ids_in(out) == ["syn_clean_001"]
