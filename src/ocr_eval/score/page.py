"""Score one page: all three levels, the order block, the contract outcome.

Edit counts and denominators are kept alongside the rates so that corpus
aggregation is a sum rather than an average of averages (research R-005).

The emitted shape is the whole of contracts/page-result.schema.json even where
the story that gives a field meaning lands later. Slicing the schema by user
story would leave the earliest and most important deliverable writing artifacts
that fail their own contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ocr_eval.io.manifest import ReferencePage
from ocr_eval.metrics import flags as flag_module
from ocr_eval.metrics import order as order_module
from ocr_eval.metrics.rates import LevelMetrics, score_level
from ocr_eval.normalise.levels import all_levels
from ocr_eval.normalise.tables import LEVELS
from ocr_eval.score import contract as contract_module


@dataclass
class PageResult:
    """The outcome for a single page. One line of pages.jsonl."""

    page_id: str
    split: str
    kind: str
    outcome: str
    flags: list[str]
    length_ratio: float
    fonts: tuple[str, ...] = ()
    distortions: tuple[str, ...] = ()
    source: str | None = None
    is_augmented: bool = False
    metrics: dict[str, LevelMetrics] | None = None
    order: order_module.OrderResult | None = None

    @property
    def carries_metrics(self) -> bool:
        return self.metrics is not None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "page_id": self.page_id,
            "split": self.split,
            "kind": self.kind,
            "fonts": list(self.fonts),
            "distortions": list(self.distortions),
            "source": self.source,
            "is_augmented": self.is_augmented,
            "outcome": self.outcome,
            "flags": list(self.flags),
            "length_ratio": self.length_ratio,
        }
        # Absent, not null: a probe page has no reference text to measure a
        # distance against, and an explicit null would invite a reader to treat
        # it as a zero.
        if self.metrics is not None:
            row["metrics"] = {
                level: self.metrics[level].to_dict() for level in LEVELS
            }
        if self.order is not None:
            row["order"] = self.order.to_dict()
        return row


def score_page(
    page: ReferencePage,
    reference: str,
    prediction: str,
    config,
) -> PageResult:
    """Score one page against one prediction."""
    outcome = contract_module.classify(reference, prediction)

    result = PageResult(
        page_id=page.page_id,
        split=page.split,
        kind=page.kind,
        outcome=outcome,
        flags=flag_module.detect(reference, prediction, config),
        length_ratio=flag_module.length_ratio(reference, prediction),
        fonts=page.fonts,
        distortions=page.distortions,
        source=page.source,
        is_augmented=page.is_augmented,
    )

    if not contract_module.carries_metrics(outcome):
        # correct_refusal and hallucination: probe pages, excluded from
        # CER/WER/exact-match aggregates entirely (research R-008).
        return result

    normalised_reference = all_levels(reference)
    normalised_prediction = all_levels(prediction)
    result.metrics = {
        level: score_level(normalised_reference[level], normalised_prediction[level])
        for level in LEVELS
    }
    result.order = order_module.measure(
        reference, prediction, config.line_match_min_similarity
    )
    return result
