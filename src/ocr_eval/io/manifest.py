"""Corpus manifest reader.

One JSONL row per reference page, read against
contracts/corpus-manifest.schema.json. Structural problems are **collected**
rather than raised on the first one: the constitution requires a validation
failure to report every problem found, because fixing a corpus one error per run
turns an afternoon into a week.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_KINDS = ("synthetic", "scan", "probe")

_KNOWN_FIELDS = {
    "page_id",
    "split",
    "kind",
    "reference_text",
    "reference_path",
    "image_path",
    "fonts",
    "distortions",
    "source",
    "is_augmented",
}


@dataclass(frozen=True)
class ReferencePage:
    """One page whose correct text is known."""

    page_id: str
    split: str
    kind: str
    reference_text: str | None = None
    reference_path: str | None = None
    image_path: str | None = None
    fonts: tuple[str, ...] = ()
    distortions: tuple[str, ...] = ()
    source: str | None = None
    is_augmented: bool = False

    @property
    def is_probe(self) -> bool:
        return self.kind == "probe"


@dataclass
class ManifestReadResult:
    pages: list[ReferencePage] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    digest: str = ""


def _as_sorted_tuple(value: object, row_label: str, field_name: str,
                     problems: list[str]) -> tuple[str, ...]:
    """Sort list fields at read time.

    A breakdown key must not depend on the order someone happened to write the
    JSON, or the same corpus authored twice would produce two different
    groupings of the same pages.
    """
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        problems.append(f"{row_label}: {field_name} must be a list of strings")
        return ()
    return tuple(sorted(value))


def read_manifest(path: str | Path) -> ManifestReadResult:
    """Read and structurally check a corpus manifest.

    Returns every page it could parse plus every problem it found. Callers
    decide whether to stop; `validate` always does.
    """
    manifest_path = Path(path)
    result = ManifestReadResult()

    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        result.problems.append(f"cannot read manifest {manifest_path}: {exc}")
        return result

    result.digest = "sha256:" + hashlib.sha256(raw).hexdigest()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        result.problems.append(f"manifest {manifest_path} is not valid UTF-8: {exc}")
        return result

    seen: dict[str, int] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        label = f"{manifest_path.name}:{line_number}"

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            result.problems.append(f"{label}: not valid JSON ({exc.msg})")
            continue

        if not isinstance(row, dict):
            result.problems.append(f"{label}: row must be a JSON object")
            continue

        unknown = sorted(set(row) - _KNOWN_FIELDS)
        if unknown:
            result.problems.append(
                f"{label}: unknown field(s) {', '.join(unknown)}"
            )

        missing = [f for f in ("page_id", "split", "kind") if not row.get(f)]
        if missing:
            result.problems.append(
                f"{label}: missing required field(s) {', '.join(missing)}"
            )
            continue

        page_id = row["page_id"]
        label = f"{label} ({page_id})"

        kind = row["kind"]
        if kind not in VALID_KINDS:
            result.problems.append(
                f"{label}: unknown kind {kind!r}; expected one of "
                f"{', '.join(VALID_KINDS)}"
            )
            continue

        has_text = "reference_text" in row
        has_path = "reference_path" in row
        if has_text == has_path:
            result.problems.append(
                f"{label}: exactly one of reference_text / reference_path is "
                f"required ({'both given' if has_text else 'neither given'})"
            )
            continue

        if page_id in seen:
            result.problems.append(
                f"{label}: duplicate page_id, first seen at line {seen[page_id]}. "
                "A duplicate is a hard failure, not a last-wins overwrite"
            )
            continue
        seen[page_id] = line_number

        result.pages.append(
            ReferencePage(
                page_id=page_id,
                split=row["split"],
                kind=kind,
                reference_text=row.get("reference_text"),
                reference_path=row.get("reference_path"),
                image_path=row.get("image_path"),
                fonts=_as_sorted_tuple(row.get("fonts"), label, "fonts", result.problems),
                distortions=_as_sorted_tuple(
                    row.get("distortions"), label, "distortions", result.problems
                ),
                source=row.get("source"),
                is_augmented=bool(row.get("is_augmented", False)),
            )
        )

    if not result.pages and not result.problems:
        result.problems.append(f"manifest {manifest_path} contains no rows")

    return result
