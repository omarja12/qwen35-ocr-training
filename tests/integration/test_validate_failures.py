"""Every hard failure in research R-009, and the rule that they all get reported.

The second assertion in most of these tests matters more than the first: the
constitution requires a validation failure to report **every** problem found,
not the first. A validator that stops at the first error turns fixing a corpus
into one round-trip per defect.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ocr_eval.normalise.tables import UNREADABLE_MARKER
from ocr_eval.score.validate import validate

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def build_corpus(tmp_path, rows, predictions, refs=None):
    """Write a throwaway corpus and prediction set, return their paths."""
    corpus = tmp_path / "corpus"
    (corpus / "refs").mkdir(parents=True, exist_ok=True)
    preds = corpus / "preds"
    preds.mkdir(parents=True, exist_ok=True)

    for name, text in (refs or {}).items():
        (corpus / "refs" / f"{name}.txt").write_text(text, encoding="utf-8", newline="")
    for name, text in predictions.items():
        (preds / f"{name}.txt").write_text(text, encoding="utf-8", newline="")

    manifest = corpus / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
        newline="",
    )
    return manifest, preds


def page(page_id, **overrides):
    row = {
        "page_id": page_id,
        "split": "synthetic_heldout",
        "kind": "synthetic",
        "reference_text": "بسم الله الرحمن الرحيم",
    }
    row.update(overrides)
    # Passing None means "leave this key out", not "set it to null". The
    # distinction matters: a row carrying both reference_text and
    # reference_path is invalid even when one of them is null, because JSON
    # Schema's `required` counts a present key whatever its value — and the
    # reader deliberately matches the schema on that point.
    return {k: v for k, v in row.items() if v is not None}


def test_manifest_page_with_no_prediction(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1"), page("p2")], {"p1": "x"})
    result = validate(manifest, preds)
    assert not result.ok
    joined = " ".join(result.problems)
    assert "p2" in joined and "no prediction" in joined
    # Never scored as perfect or as zero.
    assert "gap" in joined


def test_prediction_with_no_manifest_page(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1")], {"p1": "x", "ghost": "y"})
    result = validate(manifest, preds)
    assert not result.ok
    assert any("ghost" in p and "no such page" in p for p in result.problems)


def test_non_utf8_reference_file(tmp_path):
    manifest, preds = build_corpus(
        tmp_path,
        [page("p1", reference_text=None, reference_path="refs/p1.txt")],
        {"p1": "x"},
    )
    # A lone 0xFF byte is not valid UTF-8 in any position.
    (tmp_path / "corpus" / "refs" / "p1.txt").write_bytes(b"\xff\xfe bad bytes")
    result = validate(manifest, preds)
    assert not result.ok
    assert any("not valid UTF-8" in p for p in result.problems)


def test_empty_reference_on_a_legible_page(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1", reference_text="   ")], {"p1": "x"})
    result = validate(manifest, preds)
    assert not result.ok
    assert any("empty or whitespace-only" in p for p in result.problems)
    # The "both sides empty" edge case is unreachable because of this rule.
    assert any("corpus defect" in p for p in result.problems)


def test_probe_page_whose_reference_is_not_the_marker(tmp_path):
    manifest, preds = build_corpus(
        tmp_path,
        [page("p1", kind="probe", split="probe", reference_text="actual text here")],
        {"p1": UNREADABLE_MARKER},
    )
    result = validate(manifest, preds)
    assert not result.ok
    assert any("must be exactly" in p and UNREADABLE_MARKER in p for p in result.problems)


def test_missing_required_field(tmp_path):
    manifest, preds = build_corpus(
        tmp_path, [{"page_id": "p1", "kind": "synthetic", "reference_text": "x y"}], {"p1": "x"}
    )
    result = validate(manifest, preds)
    assert not result.ok
    assert any("missing required field" in p and "split" in p for p in result.problems)


def test_unknown_kind(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1", kind="photocopy")], {"p1": "x"})
    result = validate(manifest, preds)
    assert not result.ok
    assert any("unknown kind" in p for p in result.problems)


def test_duplicate_page_id_is_not_last_wins(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1"), page("p1")], {"p1": "x"})
    result = validate(manifest, preds)
    assert not result.ok
    assert any("duplicate page_id" in p for p in result.problems)


def test_both_reference_forms_given(tmp_path):
    manifest, preds = build_corpus(
        tmp_path,
        [page("p1", reference_path="refs/p1.txt")],
        {"p1": "x"},
        refs={"p1": "بسم الله"},
    )
    result = validate(manifest, preds)
    assert not result.ok
    assert any("exactly one of" in p for p in result.problems)


def test_pages_id_absent_from_manifest(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1")], {"p1": "x"})
    result = validate(manifest, preds, page_ids=["p1", "nonexistent"])
    assert not result.ok
    assert any("nonexistent" in p and "not in the manifest" in p for p in result.problems)


def test_every_problem_is_reported_in_one_run_not_just_the_first(tmp_path):
    """The rule that makes the whole command worth running once."""
    rows = [
        page("dup"),
        page("dup"),                                  # duplicate id
        page("empty_ref", reference_text="   "),      # empty reference
        page("no_pred"),                              # missing prediction
        page("bad_kind", kind="photocopy"),           # unknown kind
        {"page_id": "no_split", "kind": "synthetic", "reference_text": "x y"},
    ]
    manifest, preds = build_corpus(tmp_path, rows, {"dup": "x", "empty_ref": "y", "ghost": "z"})
    result = validate(manifest, preds)

    assert not result.ok
    joined = " ".join(result.problems)
    for expected in (
        "duplicate page_id",
        "empty or whitespace-only",
        "no prediction",
        "unknown kind",
        "missing required field",
        "no such page",
    ):
        assert expected in joined, f"{expected!r} was not reported"
    assert len(result.problems) >= 6


def test_cli_exits_1_and_writes_nothing_to_stdout(tmp_path):
    """Exit code 1, explanation on stderr, stdout clean."""
    manifest, preds = build_corpus(tmp_path, [page("p1"), page("p2")], {"p1": "x"})
    proc = subprocess.run(
        [sys.executable, "-m", "ocr_eval", "validate",
         "--manifest", str(manifest), "--predictions", str(preds)],
        capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "p2" in proc.stderr


def test_cli_json_mode_still_exits_nonzero(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1"), page("p2")], {"p1": "x"})
    proc = subprocess.run(
        [sys.executable, "-m", "ocr_eval", "validate", "--json",
         "--manifest", str(manifest), "--predictions", str(preds)],
        capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["problems"]


def test_clean_corpus_passes(tmp_path):
    manifest, preds = build_corpus(tmp_path, [page("p1"), page("p2")], {"p1": "x", "p2": "y"})
    result = validate(manifest, preds)
    assert result.ok, result.problems
    assert result.pages_in_scope == 2
