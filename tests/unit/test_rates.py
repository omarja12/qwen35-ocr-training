"""CER, WER, exact match and aggregation.

The zero-denominator case gets a test of the *guard* rather than a comment
saying it cannot happen. The guard lives upstream in `validate`, so the test
reaches through it: a punctuation-only reference and a diacritics-only reference
are both non-empty raw and both empty once normalised, and both must be stopped
before anything divides by them (research R-009).
"""

import json

import pytest

from ocr_eval.metrics.rates import (
    aggregate_level,
    aggregate_rate,
    score_level,
    tokenise,
)
from ocr_eval.score.validate import validate


def test_tokenise_splits_on_whitespace_runs_and_drops_empties():
    assert tokenise("  alif   baa\t\ttaa \n") == ["alif", "baa", "taa"]
    assert tokenise("") == []
    assert tokenise("   ") == []


def test_perfect_page():
    m = score_level("بسم الله", "بسم الله")
    assert m.cer == 0.0
    assert m.wer == 0.0
    assert m.exact_match is True
    assert m.char_edits == 0
    assert m.ref_chars == len("بسم الله")
    assert m.ref_words == 2


def test_single_character_substitution():
    m = score_level("abcd", "abxd")
    assert m.char_edits == 1
    assert m.ref_chars == 4
    assert m.cer == 0.25
    assert m.exact_match is False


def test_empty_prediction_is_total_loss_not_an_error():
    # A real model behaviour: scored, not skipped.
    m = score_level("abcd", "")
    assert m.char_edits == 4
    assert m.cer == 1.0
    assert m.wer == 1.0


def test_cer_can_exceed_one_on_a_runaway_prediction():
    m = score_level("abc", "abc" + "x" * 100)
    assert m.cer > 1.0


def test_micro_and_macro_differ_on_a_mixed_length_page_set():
    # One short page scored badly, one long page scored well. Micro follows the
    # long page because it has most of the characters; macro follows neither,
    # which is the point of reporting both (research R-005).
    short = score_level("abcd", "xxxx")          # 4 edits / 4 chars  = 1.00
    long_ = score_level("a" * 100, "a" * 99 + "z")  # 1 edit / 100 chars = 0.01

    cer = aggregate_rate(
        [short.char_edits, long_.char_edits], [short.ref_chars, long_.ref_chars]
    )
    assert cer.micro == pytest.approx(5 / 104)
    assert cer.macro == pytest.approx((1.00 + 0.01) / 2)
    assert cer.macro > cer.micro


def test_micro_is_a_sum_not_an_average_of_averages():
    pages = [score_level("ab", "xb"), score_level("a" * 50, "a" * 50)]
    cer = aggregate_rate([p.char_edits for p in pages], [p.ref_chars for p in pages])
    assert cer.micro == pytest.approx(1 / 52)


def test_aggregate_level_shape_and_exact_match_rate():
    pages = [
        score_level("abc", "abc"),
        score_level("abc", "abd"),
    ]
    agg = aggregate_level(pages)
    assert set(agg) == {"cer", "wer", "exact_match_rate"}
    assert set(agg["cer"]) == {"micro", "macro"}
    assert agg["exact_match_rate"] == 0.5


def test_aggregate_of_no_pages_does_not_divide_by_zero():
    agg = aggregate_level([])
    assert agg["cer"]["micro"] == 0.0
    assert agg["exact_match_rate"] == 0.0


# --- the zero-denominator guard -------------------------------------------

def test_score_level_refuses_an_empty_reference_rather_than_returning_zero():
    # If this ever returns a number instead of raising, the validator has a hole
    # and every page behind that hole reports a silently wrong CER.
    with pytest.raises(ValueError, match="normalised to empty"):
        score_level("", "anything")


@pytest.mark.parametrize(
    "label,reference,level",
    [
        ("punctuation only", "«...»؟!", "skeleton"),
        ("diacritics only", "ًَّ", "diacritic_insensitive"),
    ],
)
def test_validate_rejects_references_that_normalise_to_empty(
    tmp_path, label, reference, level
):
    corpus = tmp_path / "corpus"
    (corpus / "refs").mkdir(parents=True)
    (corpus / "preds").mkdir(parents=True)

    (corpus / "refs" / "p1.txt").write_text(reference, encoding="utf-8", newline="")
    (corpus / "preds" / "p1.txt").write_text("anything", encoding="utf-8", newline="")
    (corpus / "manifest.jsonl").write_text(
        json.dumps(
            {
                "page_id": "p1",
                "split": "synthetic_heldout",
                "kind": "synthetic",
                "reference_path": "refs/p1.txt",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )

    result = validate(corpus / "manifest.jsonl", corpus / "preds")
    assert not result.ok, f"{label} reference was accepted"
    joined = " ".join(result.problems)
    assert "normalises to empty" in joined
    # The message names the level that emptied it: "your reference is empty"
    # about a reference full of visible characters is not actionable.
    assert level in joined
