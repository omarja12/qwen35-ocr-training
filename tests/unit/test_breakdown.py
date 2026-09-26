"""Breakdown ranking, the __sparse__ fold, and the is_partition label."""

import pytest

from ocr_eval.aggregate.breakdown import SPARSE_KEY, build, build_all
from ocr_eval.config import load_config
from ocr_eval.io.manifest import ReferencePage
from ocr_eval.score.page import score_page

CONFIG = load_config()
GOOD = "بسم الله الرحمن الرحيم"


def make(page_id, fonts=(), distortions=(), source=None, errors=0):
    """A scored page whose CER rises with `errors`."""
    page = ReferencePage(
        page_id=page_id, split="s", kind="synthetic",
        reference_text=GOOD, fonts=tuple(fonts),
        distortions=tuple(distortions), source=source,
    )
    prediction = ("x" * errors) + GOOD[errors:] if errors else GOOD
    return score_page(page, GOOD, prediction, CONFIG)


def test_rows_rank_worst_first_by_cer_micro():
    results = (
        [make(f"a{i}", fonts=["Good"], errors=0) for i in range(5)]
        + [make(f"b{i}", fonts=["Bad"], errors=6) for i in range(5)]
    )
    rows = build(results, "font", min_pages=5)["rows"]
    assert [r["key"] for r in rows] == ["Bad", "Good"]
    assert rows[0]["cer_micro"] > rows[1]["cer_micro"]


def test_ties_are_broken_by_key_ascending():
    results = (
        [make(f"a{i}", fonts=["Zeta"], errors=2) for i in range(5)]
        + [make(f"b{i}", fonts=["Alpha"], errors=2) for i in range(5)]
    )
    rows = build(results, "font", min_pages=5)["rows"]
    assert [r["key"] for r in rows] == ["Alpha", "Zeta"]


def test_a_page_with_two_fonts_contributes_to_both_rows():
    results = [make(f"p{i}", fonts=["Amiri", "Cairo"], errors=1) for i in range(5)]
    breakdown = build(results, "font", min_pages=5)

    assert {r["key"] for r in breakdown["rows"]} == {"Amiri", "Cairo"}
    counted = sum(r["pages"] for r in breakdown["rows"])
    assert counted == 10, "5 pages, each counted under both of its fonts"
    assert counted > len(results)
    # Labelled explicitly so nobody reads the table as a partition.
    assert breakdown["is_partition"] is False


def test_single_valued_dimension_is_labelled_a_partition():
    results = [make(f"p{i}", source="doc_a") for i in range(5)]
    assert build(results, "source", min_pages=5)["is_partition"] is True


def test_a_dimension_missing_from_some_pages_is_not_a_partition():
    results = [make(f"p{i}", source="doc_a") for i in range(5)]
    results += [make("orphan")]  # no source at all
    assert build(results, "source", min_pages=1)["is_partition"] is False


def test_sparse_categories_fold_instead_of_topping_the_ranking():
    """One bad page must not crown itself the worst font."""
    results = [make(f"good{i}", fonts=["Amiri"], errors=1) for i in range(6)]
    results.append(make("rare", fonts=["OneOffFont"], errors=15))

    rows = build(results, "font", min_pages=5)["rows"]
    keys = [r["key"] for r in rows]

    assert "OneOffFont" not in keys, "a one-page category must not rank"
    assert SPARSE_KEY in keys
    assert "Amiri" in keys


def test_the_sparse_row_counts_a_page_once_even_with_two_sparse_keys():
    results = [make("p1", fonts=["RareA", "RareB"], errors=3)]
    rows = build(results, "font", min_pages=5)["rows"]
    sparse = next(r for r in rows if r["key"] == SPARSE_KEY)
    assert sparse["pages"] == 1


def test_all_three_dimensions_are_built_in_a_fixed_order():
    results = [make(f"p{i}", fonts=["Amiri"], distortions=["blur"], source="d") for i in range(5)]
    assert [b["dimension"] for b in build_all(results, 5)] == ["font", "distortion", "source"]


def test_probe_pages_are_excluded_from_breakdowns():
    from ocr_eval.normalise.tables import UNREADABLE_MARKER

    probe_page = ReferencePage(page_id="probe1", split="probe", kind="probe",
                               reference_text=UNREADABLE_MARKER, fonts=())
    probe = score_page(probe_page, UNREADABLE_MARKER, UNREADABLE_MARKER, CONFIG)
    results = [make(f"p{i}", fonts=["Amiri"]) for i in range(5)] + [probe]

    rows = build(results, "font", min_pages=5)["rows"]
    assert sum(r["pages"] for r in rows) == 5, "a probe page has no CER to contribute"


def test_row_shape_matches_the_contract():
    results = [make(f"p{i}", fonts=["Amiri"]) for i in range(5)]
    row = build(results, "font", min_pages=5)["rows"][0]
    assert set(row) == {
        "key", "pages", "cer_micro", "cer_macro", "wer_micro", "exact_match_rate"
    }
