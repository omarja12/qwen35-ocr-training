"""Error rates broken down by font, distortion and source, ranked worst-first.

An aggregate number tells you that something is wrong, not what to fix. The data
phase recorded which fonts and which distortions produced each image precisely so
this breakdown would be possible later (FR-006).

Rows are sorted worst-first so the worst-performing category is the **first row**
and needs no further analysis (FR-007, SC-005).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from ocr_eval.metrics.rates import aggregate_level
from ocr_eval.score.page import PageResult
from ocr_eval.score.subset import _page_values

DIMENSIONS = ("font", "distortion", "source")

SPARSE_KEY = "__sparse__"


def _row(key: str, results: Sequence[PageResult]) -> dict[str, object]:
    strict = aggregate_level(r.metrics["strict"] for r in results)
    return {
        "key": key,
        "pages": len(results),
        "cer_micro": strict["cer"]["micro"],
        "cer_macro": strict["cer"]["macro"],
        "wer_micro": strict["wer"]["micro"],
        "exact_match_rate": strict["exact_match_rate"],
    }


def build(
    results: Iterable[PageResult],
    dimension: str,
    min_pages: int,
) -> dict[str, object]:
    """One dimension of one split, ranked."""
    scored = [r for r in results if r.carries_metrics]

    grouped: dict[str, list[PageResult]] = defaultdict(list)
    contributions = 0
    pages_with_a_key = 0
    for result in scored:
        keys = _page_values(result, dimension)
        if keys:
            pages_with_a_key += 1
        for key in keys:
            grouped[key].append(result)
            contributions += 1

    # A page with two fonts contributes to both rows, so the row page counts
    # sum to more than the split total. Labelled explicitly so nobody reads the
    # table as a partition — and computed rather than hard-coded, because a
    # corpus where every page has exactly one font really is one.
    is_partition = contributions == len(scored) and pages_with_a_key == len(scored)

    # Categories below the floor fold into one row instead of topping the
    # ranking on a single bad page.
    sparse: list[PageResult] = []
    rows: list[dict[str, object]] = []
    for key, group in grouped.items():
        if len(group) < min_pages:
            sparse.extend(group)
        else:
            rows.append(_row(key, group))

    if sparse:
        # De-duplicate: a page with two sparse fonts would otherwise be counted
        # twice inside the fold-up row.
        unique = list({id(r): r for r in sparse}.values())
        rows.append(_row(SPARSE_KEY, unique))

    # Worst first; ties broken by key so the order is total and reproducible.
    rows.sort(key=lambda row: (-row["cer_micro"], row["key"]))

    return {"dimension": dimension, "is_partition": is_partition, "rows": rows}


def build_all(
    results: Iterable[PageResult],
    min_pages: int,
) -> list[dict[str, object]]:
    """Every dimension, in a fixed order."""
    materialised = list(results)
    return [build(materialised, dimension, min_pages) for dimension in DIMENSIONS]
