"""The worst-scoring pages, side by side, so a human can categorise the failure.

SC-009 wants at least 90% of surfaced pages to be assignable to a failure
category without consulting the raw files. Reference and prediction side by side
does most of that. The **codepoint difference list** does the rest: rendered
Arabic alone cannot distinguish a composed form from a decomposed one, or a
presentation form from a base letter, so two visually identical lines can differ
and a reviewer would have no way to see it (research R-016).
"""

from __future__ import annotations

import unicodedata
from typing import Mapping, Sequence

from ocr_eval.score.page import PageResult

MAX_DIFFERENCES = 40


def rank(results: Sequence[PageResult], limit: int) -> list[PageResult]:
    """Worst first, by strict CER. Ties broken by page_id so the order is total."""
    scored = [r for r in results if r.carries_metrics]
    scored.sort(key=lambda r: (-r.metrics["strict"].cer, r.page_id))
    return scored[:limit]


def codepoint_differences(
    reference: str, prediction: str, limit: int = MAX_DIFFERENCES
) -> list[str]:
    """The first `limit` positions where the two strings differ, named.

    Deliberately a plain positional walk rather than an alignment: after the
    first insertion everything shifts, which is itself the thing worth seeing.
    """
    lines: list[str] = []
    for index in range(max(len(reference), len(prediction))):
        ref_char = reference[index] if index < len(reference) else None
        pred_char = prediction[index] if index < len(prediction) else None
        if ref_char == pred_char:
            continue
        lines.append(
            f"  {index:>5}  {_describe(ref_char)}  ->  {_describe(pred_char)}"
        )
        if len(lines) >= limit:
            lines.append(f"  ... truncated at {limit} differences")
            break
    return lines


def _describe(char: str | None) -> str:
    if char is None:
        return "(end of string)"
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "<unnamed>"
    if char == "\n":
        name = "LINE FEED"
    return f"U+{ord(char):04X} {name}"


def render_markdown(
    results: Sequence[PageResult],
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    limit: int,
) -> str:
    worst = rank(results, limit)
    lines: list[str] = ["# Worst-scoring pages", ""]

    if not worst:
        lines.append("No scoreable pages in this run.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"The {len(worst)} worst page(s) by strict CER, worst first. Reference and "
        "prediction are shown side by side, followed by the exact codepoints "
        "that differ — rendered Arabic alone cannot tell a composed form from a "
        "decomposed one."
    )
    lines.append("")

    for position, result in enumerate(worst, start=1):
        strict = result.metrics["strict"]
        lines.append(f"## {position}. `{result.page_id}`")
        lines.append("")
        lines.append(f"- split: `{result.split}` · kind: `{result.kind}`")
        lines.append(
            f"- fonts: {', '.join(result.fonts) or '—'} · "
            f"distortions: {', '.join(result.distortions) or '—'}"
        )
        headline = f"- **strict CER {strict.cer:.4f}**"
        if result.order is not None:
            headline += (
                f" · order-corrected CER "
                f"{result.order.order_corrected_cer:.4f}"
            )
            accuracy = result.order.reading_order_accuracy
            headline += (
                f" · reading order "
                f"{'—' if accuracy is None else f'{accuracy * 100:.1f}%'}"
            )
        lines.append(headline)
        lines.append(f"- outcome: `{result.outcome}`")
        if result.flags:
            lines.append(f"- flags: {', '.join(f'`{f}`' for f in result.flags)}")
        lines.append(f"- length ratio: {result.length_ratio:.4f}")
        lines.append("")

        reference = references.get(result.page_id, "")
        prediction = predictions.get(result.page_id, "")

        lines.append("**Reference**")
        lines.append("")
        lines.append("```text")
        lines.append(reference)
        lines.append("```")
        lines.append("")
        lines.append("**Prediction**")
        lines.append("")
        lines.append("```text")
        lines.append(prediction)
        lines.append("```")
        lines.append("")

        differences = codepoint_differences(reference, prediction)
        if differences:
            lines.append("**Codepoint differences**")
            lines.append("")
            lines.append("```text")
            lines.append("  index  reference            ->  prediction")
            lines.extend(differences)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)
