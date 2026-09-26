"""Re-render human-readable output from an existing run, without rescoring.

**Reads `pages.jsonl` and `summary.json` only.** Never re-reads the corpus,
never recomputes a metric — which is what lets a report be re-cut with a
different `--worst` or a single breakdown dimension in under a second, and what
makes it impossible for a report to disagree with the run it came from.

One consequence is worth stating plainly, because it looks like a gap until you
see the reason. `report` cannot show reference and prediction text side by side:
that text lives in the corpus, and page rows deliberately do not carry it
(contracts/page-result.schema.json closes each row with
`additionalProperties: false`). The side-by-side view is produced by `score`,
which has the corpus in hand, and written to `worst_pages.md` / `worst_pages.html`
in the run directory. What `report` re-cuts is the **ranking** — which pages were
worst, by how much, with which flags — which is the part a different `--worst`
actually changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ocr_eval.errors import EXIT_OK, UsageError
from ocr_eval.io import writer
from ocr_eval.report import markdown as markdown_module


def _load(run_dir: Path) -> tuple[dict, list[dict]]:
    if not run_dir.is_dir():
        raise UsageError(f"not a run directory: {run_dir}")

    summary_file = run_dir / "summary.json"
    pages_file = run_dir / "pages.jsonl"
    for required in (summary_file, pages_file):
        if not required.exists():
            raise UsageError(
                f"{run_dir} is not a run directory: {required.name} is missing"
            )

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    pages = [
        json.loads(line)
        for line in pages_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summary, pages


def rank_rows(pages: list[dict], limit: int, split: str | None = None) -> list[dict]:
    """Worst first by strict CER, ties by page_id, from the artifact alone."""
    scored = [p for p in pages if "metrics" in p]
    if split:
        scored = [p for p in scored if p["split"] == split]
    scored.sort(key=lambda p: (-p["metrics"]["strict"]["cer"], p["page_id"]))
    return scored[:limit]


def render_worst_table(pages: list[dict], limit: int, split: str | None) -> str:
    rows = rank_rows(pages, limit, split)
    lines = ["# Worst-scoring pages (re-cut from the run artifact)", ""]
    if not rows:
        lines.append("No scoreable pages in this run.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"The {len(rows)} worst page(s) by strict CER. Reference and prediction "
        "text are not shown here: `report` reads only the run artifact and never "
        "re-reads the corpus, so it cannot disagree with the run it came from. "
        "For the side-by-side view see `worst_pages.md` in the run directory."
    )
    lines.append("")
    lines.append(
        "| | Page | Split | Strict CER | Order-corrected | Outcome | Flags |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for position, row in enumerate(rows, start=1):
        order = row.get("order") or {}
        corrected = order.get("order_corrected_cer")
        lines.append(
            f"| {position} | `{row['page_id']}` | `{row['split']}` "
            f"| {row['metrics']['strict']['cer']:.4f} "
            f"| {'—' if corrected is None else f'{corrected:.4f}'} "
            f"| `{row['outcome']}` "
            f"| {', '.join(row['flags']) or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_report(
    run_dir: str | Path,
    worst: int = 50,
    breakdown: str = "all",
    split: str | None = None,
    fmt: str = "md",
    out: str | Path | None = None,
) -> int:
    run_path = Path(run_dir)
    summary, pages = _load(run_path)

    if split and not any(s["split"] == split for s in summary["splits"]):
        available = ", ".join(sorted(s["split"] for s in summary["splits"]))
        raise UsageError(f"no split named {split!r} in this run. Available: {available}")

    filtered = dict(summary)
    if split:
        filtered["splits"] = [s for s in summary["splits"] if s["split"] == split]

    if breakdown != "all":
        filtered["splits"] = [
            {**s, "breakdowns": [b for b in s["breakdowns"] if b["dimension"] == breakdown]}
            for s in filtered["splits"]
        ]

    if fmt == "json":
        rendered = writer.dumps(
            {"summary": filtered, "worst_pages": rank_rows(pages, worst, split)}
        )
    else:
        rendered = (
            markdown_module.render_summary(filtered)
            + "\n"
            + render_worst_table(pages, worst, split)
        )
        if fmt == "html":
            rendered = _minimal_html(rendered)

    if out:
        writer.write_text(out, rendered)
        print(f"written to: {out}", file=sys.stderr)
    else:
        print(rendered)
    return EXIT_OK


def _minimal_html(markdown_text: str) -> str:
    """A readable HTML wrapper with no external anything (air-gapped)."""
    import html as html_escape

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Evaluation report</title>\n"
        "<style>:root{color-scheme:light dark}"
        "body{font-family:system-ui,sans-serif;max-width:1000px;margin:0 auto;"
        "padding:24px;line-height:1.5}"
        "pre{white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace;"
        "font-size:0.9rem}</style>\n</head>\n<body>\n<pre>"
        + html_escape.escape(markdown_text)
        + "</pre>\n</body>\n</html>\n"
    )
