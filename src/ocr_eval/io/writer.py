"""The only place anything is serialised.

SC-003 requires two runs over identical inputs to produce byte-identical files.
That guarantee is only enforceable if there is exactly one writer to enforce it
in, which is why every artifact in the harness goes through this module and
nothing else calls `json.dump` or `open(..., "w")`.

The same argument now applies to permissions. The training image runs as an
arbitrary non-root UID assigned by OpenShift, so everything the harness creates
is made group-writable **explicitly** rather than inherited from the ambient
umask — a run directory created under `umask 077` is unreadable to the next pod,
and that surfaces as a confusing permission error in an unrelated job rather
than as a design mistake here (research R-017, constitution principle V).

The one file deliberately outside the byte-identity guarantee is
`run.meta.json`: wall-clock, host and elapsed genuinely vary per run, so they
are segregated into a file nothing compares rather than being allowed to break
the comparison of everything else (research R-011).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

DIR_MODE = 0o775
FILE_MODE = 0o664

FLOAT_PLACES = 6

# chmod on Windows only toggles the read-only bit; the group-writable rule is a
# Linux/OpenShift concern. The calls are skipped rather than allowed to raise.
_POSIX = os.name == "posix"


def round_float(value: float) -> float:
    """Every float in every artifact passes through here.

    Six decimal places, applied once, so that two runs cannot differ in the
    seventeenth digit of a repr. `-0.0` is folded to `0.0`: it compares equal to
    zero but serialises differently, which is exactly the kind of difference
    byte-identity would catch and a reader would not understand.
    """
    rounded = round(float(value), FLOAT_PLACES)
    if rounded == 0.0:
        return 0.0
    return rounded


def canonicalise(obj: Any) -> Any:
    """Recursively apply the float rule to a structure about to be written."""
    if isinstance(obj, float):
        return round_float(obj)
    if isinstance(obj, dict):
        return {key: canonicalise(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalise(item) for item in obj]
    return obj


def dumps(obj: Any) -> str:
    """The canonical JSON form: sorted keys, real UTF-8, two-space indent.

    `sort_keys` removes insertion order from the output, which is what lets a
    parallel run (`--jobs > 1`) reorder execution without reordering output.
    `ensure_ascii=False` keeps Arabic readable in the artifact instead of
    escaping it to \\uXXXX — the worst-pages report exists to be read by a human.
    """
    return json.dumps(
        canonicalise(obj),
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        separators=(",", ": "),
    )


def dumps_line(obj: Any) -> str:
    """One JSONL row: canonical, but compact and on a single line."""
    return json.dumps(
        canonicalise(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) group-writable, explicitly."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    _chmod(directory, DIR_MODE)
    return directory


def _chmod(path: Path, mode: int) -> None:
    if not _POSIX:
        return
    try:
        os.chmod(path, mode)
    except OSError as exc:  # pragma: no cover - filesystem-dependent
        print(
            f"warning: could not set mode {oct(mode)} on {path}: {exc}",
            file=sys.stderr,
        )


def write_text(path: str | Path, text: str) -> Path:
    """Write text as UTF-8 without BOM, LF endings, one trailing newline.

    `newline=""` stops Python translating "\\n" to "\\r\\n" on Windows, which
    would make the same inputs produce different bytes on different platforms
    and break SC-003 across the two targets plan.md names.
    """
    target = Path(path)
    ensure_dir(target.parent)
    body = text if text.endswith("\n") else text + "\n"
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(body)
    _chmod(target, FILE_MODE)
    return target


def write_json(path: str | Path, obj: Any) -> Path:
    """Write one JSON document in canonical form."""
    return write_text(path, dumps(obj))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Write JSONL. Callers sort first; this does not reorder."""
    body = "\n".join(dumps_line(row) for row in rows)
    return write_text(path, body)


def read_text(path: str | Path) -> str:
    """Read UTF-8 strictly, stripping exactly one trailing newline.

    The single I/O rule from the normalisation policy, applied symmetrically to
    references and predictions: it is an artifact of how text files are written
    rather than of what the model produced. No other whitespace is touched, and
    `trailing_newline_stripped: true` is recorded in every manifest so a reader
    knows it happened (research R-002).

    A UTF-8 BOM is stripped if present — a BOM is a byte-order artifact of the
    editor that wrote the file, and leaving it in would count as a character
    error on the first line of every page produced by that editor.
    """
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")  # strict: a decode error names the file upstream
    if text.startswith("﻿"):
        text = text[1:]
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text
