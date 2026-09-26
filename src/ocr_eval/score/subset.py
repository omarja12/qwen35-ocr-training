"""Subset selection: evaluate part of a corpus without re-running the whole set.

Deliberately small. This is a filter, not a query language (FR-018): there is no
OR, because two runs express that more honestly than one run with a clause
nobody reads carefully.

The parsed filter is retained as **structured data** rather than as the original
string, so the run manifest records the scope in a form a later reader can
inspect field by field instead of re-parsing prose (SC-008).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ocr_eval.errors import UsageError
from ocr_eval.io.manifest import ReferencePage

# Singular here, plural in the manifest: a page has `fonts`, but you filter on
# "has this font". The mapping is explicit so the difference is a rule rather
# than a surprise.
LIST_FIELDS = {"font": "fonts", "distortion": "distortions"}
SCALAR_FIELDS = {"split", "kind", "source", "is_augmented"}
FILTER_FIELDS = tuple(sorted(SCALAR_FIELDS | set(LIST_FIELDS)))
OPERATORS = ("!=", "~", "=")


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_filter(expression: str) -> list[Condition]:
    """Parse `field<op>value[,field<op>value...]` into conditions."""
    conditions: list[Condition] = []
    for clause in expression.split(","):
        clause = clause.strip()
        if not clause:
            continue
        for op in OPERATORS:  # "!=" before "=", or "!=" parses as "=" with a stray "!"
            if op in clause:
                field, _, value = clause.partition(op)
                field = field.strip()
                value = value.strip()
                if field not in FILTER_FIELDS:
                    raise UsageError(
                        f"unknown filter field {field!r}; expected one of "
                        f"{', '.join(FILTER_FIELDS)}"
                    )
                if not value:
                    raise UsageError(f"filter clause {clause!r} has an empty value")
                conditions.append(Condition(field=field, op=op, value=value))
                break
        else:
            raise UsageError(
                f"filter clause {clause!r} has no operator; expected one of "
                f"{', '.join(OPERATORS)}"
            )
    if not conditions:
        raise UsageError("filter expression is empty")
    return conditions


def _page_values(page: ReferencePage, field: str) -> list[str]:
    """Every value of `field` on `page`, as strings."""
    if field in LIST_FIELDS:
        return list(getattr(page, LIST_FIELDS[field]))
    if field == "is_augmented":
        return ["true" if page.is_augmented else "false"]
    value = getattr(page, field, None)
    return [] if value is None else [str(value)]


def _matches_condition(page: ReferencePage, condition: Condition) -> bool:
    values = _page_values(page, condition.field)
    if condition.op == "=":
        # On a list field this means "contains", which is why a page with two
        # fonts satisfies a filter on either of them.
        return condition.value in values
    if condition.op == "!=":
        return condition.value not in values
    if condition.op == "~":
        return any(condition.value in value for value in values)
    raise UsageError(f"unknown operator {condition.op!r}")


def matches(page: ReferencePage, conditions: Sequence[Condition]) -> bool:
    """Conditions are ANDed."""
    return all(_matches_condition(page, c) for c in conditions)


def read_pages_file(path: str | Path) -> list[str]:
    """Read a `--pages` id list, one per line. Blank lines and # comments skipped."""
    pages_path = Path(path)
    if not pages_path.exists():
        raise UsageError(f"--pages file not found: {pages_path}")
    try:
        text = pages_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"cannot read --pages file {pages_path}: {exc}") from None

    ids: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    if not ids:
        raise UsageError(f"--pages file {pages_path} lists no page ids")
    return ids


def select(
    pages: Iterable[ReferencePage],
    page_ids: Sequence[str] | None = None,
    conditions: Sequence[Condition] | None = None,
) -> list[ReferencePage]:
    """Apply both restrictions, preserving manifest order."""
    selected = list(pages)
    if page_ids is not None:
        wanted = set(page_ids)
        selected = [p for p in selected if p.page_id in wanted]
    if conditions:
        selected = [p for p in selected if matches(p, conditions)]
    return selected


def describe(
    page_ids: Sequence[str] | None,
    conditions: Sequence[Condition] | None,
) -> dict[str, object] | None:
    """The structured record of a run's scope, for the manifest."""
    if not page_ids and not conditions:
        return None
    record: dict[str, object] = {}
    if page_ids:
        record["pages"] = sorted(page_ids)
    if conditions:
        record["conditions"] = [c.to_dict() for c in conditions]
    return record
