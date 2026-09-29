"""Whole-set integrity checking, before a single page is scored.

Expensive work is preceded by a cheap integrity check, and the check runs over
the inputs **as a whole** rather than page by page, so that one run tells you
everything that is wrong with the corpus (constitution IV, research R-009).

The line this module draws is the one the constitution draws: **input
integrity** stops the run; **model behaviour** is scored and flagged. A missing
prediction is a gap and stops everything. An empty prediction on a legible page
is a real thing the model did — it is scored as total loss and flagged, never
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ocr_eval.io.manifest import ManifestReadResult, ReferencePage, read_manifest
from ocr_eval.io.predictions import PredictionSet, read_predictions
from ocr_eval.io.writer import read_text
from ocr_eval.normalise.levels import all_levels
from ocr_eval.normalise.tables import LEVELS, UNREADABLE_MARKER
from ocr_eval.score.subset import Condition, select


@dataclass
class ValidationResult:
    """What validation found, and the resolved inputs if it found nothing."""

    problems: list[str] = field(default_factory=list)
    pages: list[ReferencePage] = field(default_factory=list)
    references: dict[str, str] = field(default_factory=dict)
    predictions: PredictionSet | None = None
    manifest_digest: str = ""
    pages_in_scope: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def resolve_reference(
    page: ReferencePage,
    corpus_root: Path,
    problems: list[str],
) -> str | None:
    """Get a page's ground truth, from inline text or from a file."""
    if page.reference_text is not None:
        return page.reference_text

    assert page.reference_path is not None  # manifest reader guarantees one of the two
    reference_file = corpus_root / page.reference_path
    try:
        return read_text(reference_file)
    except UnicodeDecodeError as exc:
        problems.append(f"{page.page_id}: reference {reference_file} is not valid UTF-8: {exc}")
    except OSError as exc:
        problems.append(f"{page.page_id}: cannot read reference {reference_file}: {exc}")
    return None


def _check_reference_content(
    page: ReferencePage,
    reference: str,
    problems: list[str],
) -> None:
    """The rules that make a zero denominator unreachable."""
    if page.is_probe:
        # A probe page carrying real text is a corpus defect: it would be scored
        # under the contract matrix as though it had no readable content.
        if reference.strip() != UNREADABLE_MARKER:
            problems.append(
                f"{page.page_id}: kind is 'probe' so the reference must be exactly "
                f"{UNREADABLE_MARKER!r}, found {reference.strip()[:60]!r}"
            )
        return

    if not reference.strip():
        problems.append(
            f"{page.page_id}: reference is empty or whitespace-only on a legible "
            "page. This is a corpus defect, not a score of zero"
        )
        return

    # Non-empty raw, but empty once normalised, divides CER by zero. A reference
    # of nothing but punctuation empties at skeleton (step 6 drops every P* and
    # S*); one of nothing but diacritics empties at diacritic-insensitive. The
    # level that emptied it is named, because "your reference is empty" about a
    # reference that visibly contains characters is not an actionable message.
    normalised = all_levels(reference)
    for level in LEVELS:
        if not normalised[level].strip():
            problems.append(
                f"{page.page_id}: reference normalises to empty at level "
                f"'{level}' (raw reference is {reference.strip()[:40]!r}). "
                "Every level must leave at least one character, or CER has no "
                "denominator"
            )


def validate(
    manifest_path: str | Path,
    predictions_path: str | Path,
    corpus_root: str | Path | None = None,
    page_ids: Sequence[str] | None = None,
    conditions: Sequence[Condition] | None = None,
) -> ValidationResult:
    """Check that a corpus and a prediction set line up. Reports everything."""
    result = ValidationResult()
    manifest_file = Path(manifest_path)
    root = Path(corpus_root) if corpus_root else manifest_file.parent

    manifest: ManifestReadResult = read_manifest(manifest_file)
    result.problems.extend(manifest.problems)
    result.manifest_digest = manifest.digest

    predictions = read_predictions(predictions_path)
    result.problems.extend(predictions.problems)
    result.predictions = predictions

    known_ids = {page.page_id for page in manifest.pages}

    # A --pages id that is not in the manifest is a mistake worth stopping for:
    # silently scoring the subset that happened to match would report a number
    # over a different set of pages than the one that was asked for.
    if page_ids:
        for page_id in page_ids:
            if page_id not in known_ids:
                result.problems.append(
                    f"--pages lists {page_id!r}, which is not in the manifest"
                )

    in_scope = select(manifest.pages, page_ids, conditions)
    result.pages = in_scope
    result.pages_in_scope = len(in_scope)

    if manifest.pages and not in_scope:
        result.problems.append(
            "the subset selection matched no pages; nothing would be scored"
        )

    scope_ids = {page.page_id for page in in_scope}

    for page in in_scope:
        reference = resolve_reference(page, root, result.problems)
        if reference is None:
            continue
        result.references[page.page_id] = reference
        _check_reference_content(page, reference, result.problems)

        if page.page_id not in predictions.texts:
            result.problems.append(
                f"{page.page_id}: in the manifest but has no prediction. "
                "Reported as a gap, never scored as perfect or as zero"
            )

    # The mirror check. Restricted to pages the manifest knows about at all, so
    # a deliberate subset run is not drowned in "not in scope" noise for every
    # page it intentionally left out.
    for page_id in sorted(predictions.texts):
        if page_id not in known_ids:
            result.problems.append(
                f"{page_id}: prediction supplied but no such page in the manifest"
            )
        elif page_id not in scope_ids and not page_ids and not conditions:
            result.problems.append(
                f"{page_id}: prediction supplied but the page is not in scope"
            )

    return result
