"""`report` re-cuts an existing run without re-reading the corpus."""

import json

import pytest

from ocr_eval.config import load_config
from ocr_eval.errors import UsageError
from ocr_eval.report.driver import rank_rows, render_report
from ocr_eval.score.run import score_run
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("report") / "run"
    assert score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "baseline",
        model_version="report-test",
        out_dir=out,
        config=load_config(),
    ) == 0
    return out


def pages_of(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "pages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_report_reads_only_the_run_artifact_never_the_corpus(run_dir, tmp_path, monkeypatch):
    """Proved by making the corpus unreachable, not by inspection.

    Any attempt to open a path under the fixture corpus raises, so a report that
    quietly re-read the references would fail loudly here.
    """
    import builtins

    real_open = builtins.open
    corpus = str(FIXTURES.resolve())

    def guarded(file, *args, **kwargs):
        if corpus in str(file):
            raise AssertionError(f"report re-read the corpus: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    assert render_report(run_dir, worst=5, out=tmp_path / "r.md") == 0


def test_rerendering_with_a_different_worst_cannot_disagree_with_the_run(run_dir):
    pages = pages_of(run_dir)
    top_three = rank_rows(pages, 3)
    top_ten = rank_rows(pages, 10)

    # The same ranking, just cut at a different depth.
    assert [p["page_id"] for p in top_three] == [p["page_id"] for p in top_ten[:3]]
    # And the numbers come straight from the artifact.
    for row in top_ten:
        original = next(p for p in pages if p["page_id"] == row["page_id"])
        assert row["metrics"]["strict"]["cer"] == original["metrics"]["strict"]["cer"]


def test_ranking_is_worst_first_with_ties_broken_by_page_id(run_dir):
    rows = rank_rows(pages_of(run_dir), 50)
    keys = [(-r["metrics"]["strict"]["cer"], r["page_id"]) for r in rows]
    assert keys == sorted(keys)


def test_breakdown_filter_selects_one_dimension(run_dir, tmp_path):
    out = tmp_path / "font-only.md"
    render_report(run_dir, breakdown="font", out=out)
    text = out.read_text(encoding="utf-8")
    assert "By font" in text
    assert "By distortion" not in text


def test_split_filter_selects_one_split(run_dir, tmp_path):
    out = tmp_path / "gold.md"
    render_report(run_dir, split="gold_scans", out=out)
    text = out.read_text(encoding="utf-8")
    assert "`gold_scans`" in text
    assert "Split: `probe`" not in text


def test_unknown_split_is_a_usage_error_naming_what_exists(run_dir):
    with pytest.raises(UsageError, match="no split named"):
        render_report(run_dir, split="nonexistent")


def test_json_format_is_valid_json(run_dir, tmp_path):
    out = tmp_path / "r.json"
    render_report(run_dir, fmt="json", worst=5, out=out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in payload and "worst_pages" in payload
    assert len(payload["worst_pages"]) <= 5


def test_html_format_is_self_contained(run_dir, tmp_path):
    out = tmp_path / "r.html"
    render_report(run_dir, fmt="html", out=out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    # Air-gapped: nothing may be fetched at view time.
    for forbidden in ("http://", "https://", "<script"):
        assert forbidden not in text


def test_worst_table_says_why_it_has_no_side_by_side_text(run_dir, tmp_path):
    """The limitation is stated in the output, not left to be discovered."""
    out = tmp_path / "r.md"
    render_report(run_dir, worst=3, out=out)
    text = out.read_text(encoding="utf-8")
    assert "never re-reads the corpus" in text
    assert "worst_pages.md" in text, "it should point at where the text does live"


def test_a_missing_run_directory_is_a_usage_error(tmp_path):
    with pytest.raises(UsageError, match="not a run directory"):
        render_report(tmp_path / "nope")


def test_an_incomplete_run_directory_names_the_missing_file(tmp_path):
    incomplete = tmp_path / "half"
    incomplete.mkdir()
    (incomplete / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(UsageError, match="pages.jsonl is missing"):
        render_report(incomplete)


def test_score_already_wrote_the_side_by_side_view(run_dir):
    """FR-017 is satisfied by `score`, which has the corpus in hand (T054)."""
    markdown = (run_dir / "worst_pages.md").read_text(encoding="utf-8")
    assert "**Reference**" in markdown and "**Prediction**" in markdown
    assert "Codepoint differences" in markdown

    html = (run_dir / "worst_pages.html").read_text(encoding="utf-8")
    assert 'dir="rtl"' in html and 'lang="ar"' in html


def test_the_worst_font_and_distortion_are_nameable_from_one_report(run_dir):
    """SC-005, end to end.

    The first row of each table must be a real category name. This can fail for
    a reason that is not a code bug: with `min_pages_per_breakdown_row` at 5, a
    corpus where no font reaches five pages folds every font into `__sparse__`,
    and the report then names nothing. The fixture is sized so each font clears
    the floor, which is what makes the criterion demonstrable at all.
    """
    import json

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    synthetic = next(s for s in summary["splits"] if s["split"] == "synthetic_heldout")
    by_dimension = {b["dimension"]: b for b in synthetic["breakdowns"]}

    for dimension in ("font", "distortion"):
        rows = by_dimension[dimension]["rows"]
        assert rows, f"no {dimension} rows at all"
        worst = rows[0]
        assert worst["key"] != "__sparse__", (
            f"the worst {dimension} is unnameable: every category fell below "
            f"min_pages_per_breakdown_row, so SC-005 cannot be met on this corpus"
        )
        # Worst first, so the first row really is the worst.
        assert worst["cer_micro"] == max(r["cer_micro"] for r in rows)
