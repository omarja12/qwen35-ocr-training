"""Prediction set reader.

Two forms, auto-detected from whether the path is a directory:

  - a directory of `<page_id>.txt`
  - a JSONL file of `{page_id, text}` rows

Both produce the same in-memory mapping. The form used is recorded in the run
manifest, because "which shape was the input" is part of what a stranger needs
to reconstruct the run (constitution II).

Decoding is strict UTF-8 and a decode error names the file. Exactly one trailing
newline is stripped, symmetrically with references (research R-002).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ocr_eval.io.writer import read_text


@dataclass
class PredictionSet:
    """Predictions for one model version over one corpus."""

    texts: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    form: str = ""
    digest: str = ""


def _digest(texts: dict[str, str]) -> str:
    """SHA-256 over sorted page_id + text.

    Deliberately independent of file layout: the same predictions supplied as a
    directory and as a JSONL must produce the same digest, or two runs over
    identical content would look incommensurable to `compare`.
    """
    hasher = hashlib.sha256()
    for page_id in sorted(texts):
        hasher.update(page_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(texts[page_id].encode("utf-8"))
        hasher.update(b"\x00")
    return "sha256:" + hasher.hexdigest()


def read_predictions(path: str | Path) -> PredictionSet:
    """Read a prediction set from either supported form."""
    source = Path(path)
    result = PredictionSet()

    if not source.exists():
        result.problems.append(f"predictions path does not exist: {source}")
        return result

    if source.is_dir():
        result.form = "directory"
        _read_directory(source, result)
    else:
        result.form = "jsonl"
        _read_jsonl(source, result)

    result.digest = _digest(result.texts)
    return result


def _read_directory(source: Path, result: PredictionSet) -> None:
    files = sorted(source.glob("*.txt"))
    if not files:
        result.problems.append(f"no *.txt prediction files found in {source}")
        return
    for item in files:
        page_id = item.stem
        try:
            result.texts[page_id] = read_text(item)
        except UnicodeDecodeError as exc:
            result.problems.append(f"{item} is not valid UTF-8: {exc}")
        except OSError as exc:
            result.problems.append(f"cannot read {item}: {exc}")


def _read_jsonl(source: Path, result: PredictionSet) -> None:
    try:
        raw = source.read_bytes()
    except OSError as exc:
        result.problems.append(f"cannot read {source}: {exc}")
        return

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        result.problems.append(f"{source} is not valid UTF-8: {exc}")
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        label = f"{source.name}:{line_number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            result.problems.append(f"{label}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict) or "page_id" not in row or "text" not in row:
            result.problems.append(
                f"{label}: each row must be an object with page_id and text"
            )
            continue
        page_id = row["page_id"]
        if page_id in result.texts:
            result.problems.append(f"{label}: duplicate page_id {page_id!r}")
            continue
        if not isinstance(row["text"], str):
            result.problems.append(f"{label}: text must be a string")
            continue
        result.texts[page_id] = row["text"]

    if not result.texts and not result.problems:
        result.problems.append(f"{source} contains no prediction rows")
