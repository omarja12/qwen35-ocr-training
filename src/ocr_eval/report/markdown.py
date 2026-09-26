"""Human-readable summary, rendered from summary.json.

Two presentation rules here are requirements, not taste:

  - all three normalisation levels appear together, and the strict score is
    never absent and never subordinate (FR-003a, SC-004);
  - strict is labelled **"strict (raw, unnormalised)"** rather than bare
    "strict". FR-002 asks for a raw score alongside every normalised one. Strict
    *is* that score, but a reader scanning for the word "raw" would otherwise
    conclude it is missing.
"""

from __future__ import annotations

LEVEL_LABELS = {
    "strict": "strict (raw, unnormalised)",
    "diacritic_insensitive": "diacritic-insensitive",
    "skeleton": "skeleton",
}


def _fmt(value: float | None, places: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def render_summary(summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation summary — {summary['model_version']}")
    lines.append("")
    lines.append(f"- **run_id**: `{summary['run_id']}`")
    lines.append(f"- **tool version**: {summary['tool_version']}")
    lines.append(f"- **policy version**: {summary['policy']['policy_version']}")
    lines.append(
        f"- **unreadable marker**: `{summary['policy']['unreadable_marker']}`"
    )
    lines.append("")
    lines.append(
        "Headline figures are **strict**: every difference counts. The looser "
        "levels exist to locate errors, never to replace the headline."
    )
    lines.append("")

    for split in summary["splits"]:
        lines.extend(_render_split(split))

    if summary["notes"]:
        lines.append("## Notes")
        lines.append("")
        for note in summary["notes"]:
            lines.append(f"> {note}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "There is deliberately no cross-split total anywhere in this report. "
        "Real scans and synthetic pages are never merged into one headline "
        "(FR-013)."
    )
    lines.append("")
    return "\n".join(lines)


def _render_split(split: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Split: `{split['split']}`")
    lines.append("")
    lines.append(
        f"{split['pages']} page(s), {split['scored_pages']} scoreable."
    )
    lines.append("")

    if split["headline"] is None:
        # A probe-only split. Saying why, in words, beats an empty cell that a
        # reader has to interpret — and beats a zero, which would read as a
        # perfect score (research R-018).
        lines.append(
            "**No scoreable pages in this split.** Every page here is a probe "
            "with no reference text to measure a distance against, so CER, WER "
            "and exact match are reported as `null` rather than as `0` — a zero "
            "would read as a perfect score."
        )
        lines.append("")
        lines.extend(_render_contract(split["contract"]))
        lines.extend(_render_flags(split["flags"]))
        return lines

    headline = split["headline"]
    lines.append(
        f"**Strict CER {_fmt(headline['cer'])}** · "
        f"WER {_fmt(headline['wer'])} · "
        f"exact match {_pct(headline['exact_match_rate'])}"
    )
    lines.append("")

    lines.append("| Level | CER (micro) | CER (macro) | WER (micro) | Exact match |")
    lines.append("|---|---|---|---|---|")
    for level, label in LEVEL_LABELS.items():
        block = split["metrics"][level]
        marker = "**" if level == "strict" else ""
        lines.append(
            f"| {marker}{label}{marker} "
            f"| {marker}{_fmt(block['cer']['micro'])}{marker} "
            f"| {_fmt(block['cer']['macro'])} "
            f"| {_fmt(block['wer']['micro'])} "
            f"| {_pct(block['exact_match_rate'])} |"
        )
    lines.append("")

    order = split["order"]
    lines.append("### Recognition or ordering?")
    lines.append("")
    lines.append(
        f"- headline CER (as produced): **{_fmt(headline['cer'])}**"
    )
    lines.append(
        f"- order-corrected CER: {_fmt(order['order_corrected_cer_micro'])}"
    )
    lines.append(
        f"- reading-order accuracy: {_pct(order['reading_order_accuracy'])}"
    )
    lines.append(
        f"- pages without an order signal: {order['pages_without_order_signal']}"
    )
    lines.append("")
    lines.append(
        "A headline CER well above the order-corrected CER means the text was "
        "recognised and sequenced wrongly. Both high means recognition itself "
        "is failing. The two need different fixes."
    )
    lines.append("")

    lines.extend(_render_contract(split["contract"]))
    lines.extend(_render_flags(split["flags"]))
    lines.extend(_render_breakdowns(split["breakdowns"]))
    return lines


def _render_contract(contract: dict) -> list[str]:
    lines = ["### `[UNREADABLE]` contract", ""]
    lines.append("| Outcome | Count | Rate |")
    lines.append("|---|---|---|")
    lines.append(
        f"| correct refusal | {contract['correct_refusal']} | — |"
    )
    lines.append(
        f"| **hallucination** | {contract['hallucination']} "
        f"| {_pct(contract['hallucination_rate'])} of {contract['probe_pages']} probe page(s) |"
    )
    lines.append(
        f"| **false refusal** | {contract['false_refusal']} "
        f"| {_pct(contract['false_refusal_rate'])} of {contract['legible_pages']} legible page(s) |"
    )
    lines.append("")
    lines.append(
        "Hallucination and false refusal are separate counts over separate "
        "denominators and are never summed. Inventing text on a blank page and "
        "refusing a readable one are different failures with different fixes."
    )
    lines.append("")
    return lines


def _render_flags(flags: dict) -> list[str]:
    if not any(flags.values()):
        return []
    lines = ["### Flags", ""]
    for name, count in sorted(flags.items()):
        if count:
            lines.append(f"- `{name}`: {count} page(s)")
    lines.append("")
    lines.append(
        "A flag never removes a page from an aggregate. It says where an error "
        "rate came from."
    )
    lines.append("")
    return lines


VERDICT_BLURB = {
    "PROMOTE": "The candidate beats the baseline by the stated margin and no reject condition fired.",
    "REJECT": "At least one reject condition fired. A REJECT is a successful comparison, not a failed command.",
    "INCONCLUSIVE": "No reject condition fired, but the improvement did not reach the bar.",
}

METRIC_LABELS = {
    "cer": "CER (strict)",
    "wer": "WER (strict)",
    "exact_match_rate": "exact match",
    "hallucination_rate": "hallucination rate",
    "false_refusal_rate": "false-refusal rate",
}


def render_comparison(comparison: dict) -> str:
    """The comparison, rendered so a reviewer can act without opening the data."""
    lines: list[str] = []
    lines.append(
        f"# {comparison['candidate_model_version']} vs "
        f"{comparison['baseline_model_version']}"
    )
    lines.append("")
    lines.append(f"## Verdict: **{comparison['verdict']}**")
    lines.append("")
    lines.append(VERDICT_BLURB.get(comparison["verdict"], ""))
    lines.append("")
    for reason in comparison["verdict_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| baseline run | `{comparison['baseline_run_id']}` |")
    lines.append(f"| candidate run | `{comparison['candidate_run_id']}` |")
    lines.append(f"| policy version | {comparison['policy_version']} |")
    lines.append(f"| primary split | `{comparison['primary_split']}` |")
    lines.append(
        f"| regressions / improvements | {comparison['regression_count']} / "
        f"{comparison['improvement_count']} |"
    )
    lines.append("")

    for split in comparison["splits"]:
        lines.append(f"## Split: `{split['split']}`")
        lines.append("")
        if not split["metrics"]:
            lines.append(
                "No metric is defined on both sides for this split — it has no "
                "scoreable pages, so there is nothing to take a delta of."
            )
            lines.append("")
            continue
        lines.append("| Metric | Baseline | Candidate | Delta | Direction |")
        lines.append("|---|---|---|---|---|")
        for metric in split["metrics"]:
            label = METRIC_LABELS.get(metric["metric"], metric["metric"])
            lines.append(
                f"| {label} | {metric['baseline']:.4f} | {metric['candidate']:.4f} "
                f"| {metric['delta']:+.4f} | **{metric['direction']}** |"
            )
        lines.append("")
        lines.append(
            "*Direction is a word because the sign that means 'better' differs "
            "per metric: lower is better for CER, higher for exact match.*"
        )
        lines.append("")

    if comparison["bootstrap"]:
        lines.append("## Is the difference a result?")
        lines.append("")
        lines.append("| Split | Mean delta CER | 95% interval | Straddles zero? |")
        lines.append("|---|---|---|---|")
        for entry in comparison["bootstrap"]:
            straddles = entry["ci_low"] <= 0.0 <= entry["ci_high"]
            lines.append(
                f"| `{entry['split']}` | {entry['delta_cer']:+.4f} "
                f"| [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] "
                f"| {'**yes — not a result**' if straddles else 'no'} |"
            )
        lines.append("")
        first = comparison["bootstrap"][0]
        lines.append(
            f"*Paired bootstrap, {first['resamples']} resamples, seed "
            f"{first['seed']}, {first['confidence'] * 100:.0f}% interval.*"
        )
        lines.append("")

    if comparison["regressions"]:
        lines.append("## Regressions")
        lines.append("")
        lines.append(
            f"{comparison['regression_count']} page(s) got worse; the worst "
            f"{len(comparison['regressions'])} are listed."
        )
        lines.append("")
        lines.append("| Page | Split | Baseline CER | Candidate CER | Delta | Outcome |")
        lines.append("|---|---|---|---|---|---|")
        for row in comparison["regressions"]:
            lines.append(
                f"| `{row['page_id']}` | `{row['split']}` "
                f"| {row['baseline_cer']:.4f} | {row['candidate_cer']:.4f} "
                f"| {row['delta_cer']:+.4f} | {row['outcome_changed'] or '—'} |"
            )
        lines.append("")

    if comparison["improvements"]:
        lines.append("## Improvements")
        lines.append("")
        lines.append("| Page | Split | Baseline CER | Candidate CER | Delta |")
        lines.append("|---|---|---|---|---|")
        for row in comparison["improvements"]:
            lines.append(
                f"| `{row['page_id']}` | `{row['split']}` "
                f"| {row['baseline_cer']:.4f} | {row['candidate_cer']:.4f} "
                f"| {row['delta_cer']:+.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_breakdowns(breakdowns: list) -> list[str]:
    if not breakdowns:
        return []
    lines = ["### Breakdowns", ""]
    for breakdown in breakdowns:
        rows = breakdown["rows"]
        if not rows:
            continue
        lines.append(f"#### By {breakdown['dimension']}")
        lines.append("")
        if not breakdown["is_partition"]:
            lines.append(
                "*Not a partition: a page carrying two values contributes to "
                "both rows, so these page counts sum to more than the split "
                "total.*"
            )
            lines.append("")
        lines.append("| | Key | Pages | CER (micro) | CER (macro) | WER (micro) | Exact |")
        lines.append("|---|---|---|---|---|---|---|")
        for rank, row in enumerate(rows, start=1):
            worst = " ⬅ worst" if rank == 1 else ""
            lines.append(
                f"| {rank} | `{row['key']}`{worst} | {row['pages']} "
                f"| {_fmt(row['cer_micro'])} | {_fmt(row['cer_macro'])} "
                f"| {_fmt(row['wer_micro'])} | {_pct(row['exact_match_rate'])} |"
            )
        lines.append("")
    return lines
