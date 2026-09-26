"""RTL-correct side-by-side worst pages.

The text cells carry `dir="rtl" lang="ar"` so Arabic renders the way a reviewer
reads it. The codepoint list beside them stays LTR and monospaced, because it is
the part that answers the question rendering cannot: two visually identical
lines can differ in composition, and only the codepoints show it (research
R-016).

Self-contained by construction — inline CSS, no external stylesheet, no font
CDN, no script. The harness runs air-gapped, and an HTML file that silently
degrades because it cannot reach a network is worse than one that never tried.
"""

from __future__ import annotations

import html as html_escape
from typing import Mapping, Sequence

from ocr_eval.report.worst_pages import codepoint_differences, rank
from ocr_eval.score.page import PageResult

_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0 auto; padding: 24px; max-width: 1100px; line-height: 1.5;
}
h1 { font-size: 1.5rem; }
h2 { font-size: 1.1rem; margin-top: 2.5rem; border-top: 1px solid #8884; padding-top: 1rem; }
.meta { font-size: 0.85rem; opacity: 0.8; margin: 0.25rem 0 0.75rem; }
.meta code { font-size: 0.85em; }
table.sbs { width: 100%; border-collapse: collapse; table-layout: fixed; }
table.sbs th { text-align: start; font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: 0.05em; opacity: 0.7; padding-bottom: 0.4rem; }
table.sbs td {
  vertical-align: top; width: 50%; padding: 0.75rem;
  border: 1px solid #8884; border-radius: 4px;
  white-space: pre-wrap; word-break: break-word;
  font-size: 1.05rem;
}
td.arabic { direction: rtl; text-align: right; }
pre.codepoints {
  direction: ltr; text-align: left; overflow-x: auto;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 0.8rem; background: #8881; padding: 0.75rem; border-radius: 4px;
}
.flag { display: inline-block; padding: 0.1rem 0.4rem; border: 1px solid #8886;
  border-radius: 999px; font-size: 0.75rem; margin-inline-end: 0.3rem; }
.note { font-size: 0.9rem; opacity: 0.85; }
"""


def _esc(text: str) -> str:
    return html_escape.escape(text, quote=True)


def render(
    results: Sequence[PageResult],
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    limit: int,
) -> str:
    worst = rank(results, limit)

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Worst-scoring pages</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<h1>Worst-scoring pages</h1>",
    ]

    if not worst:
        parts.append("<p>No scoreable pages in this run.</p>")
        parts.extend(["</body>", "</html>", ""])
        return "\n".join(parts)

    parts.append(
        f'<p class="note">The {len(worst)} worst page(s) by strict CER, worst '
        "first. Text cells are RTL; the codepoint list is not, because it is "
        "the part that shows what rendering hides.</p>"
    )

    for position, result in enumerate(worst, start=1):
        strict = result.metrics["strict"]
        parts.append(f"<h2>{position}. <code>{_esc(result.page_id)}</code></h2>")

        meta = [
            f"split <code>{_esc(result.split)}</code>",
            f"kind <code>{_esc(result.kind)}</code>",
            f"fonts {_esc(', '.join(result.fonts) or '—')}",
            f"distortions {_esc(', '.join(result.distortions) or '—')}",
            f"<strong>strict CER {strict.cer:.4f}</strong>",
        ]
        if result.order is not None:
            meta.append(f"order-corrected {result.order.order_corrected_cer:.4f}")
        meta.append(f"outcome <code>{_esc(result.outcome)}</code>")
        meta.append(f"length ratio {result.length_ratio:.4f}")
        parts.append(f'<p class="meta">{" · ".join(meta)}</p>')

        if result.flags:
            badges = "".join(
                f'<span class="flag">{_esc(flag)}</span>' for flag in result.flags
            )
            parts.append(f"<p>{badges}</p>")

        reference = references.get(result.page_id, "")
        prediction = predictions.get(result.page_id, "")
        parts.extend(
            [
                '<table class="sbs">',
                "<tr><th>Reference</th><th>Prediction</th></tr>",
                "<tr>",
                f'<td class="arabic" dir="rtl" lang="ar">{_esc(reference)}</td>',
                f'<td class="arabic" dir="rtl" lang="ar">{_esc(prediction)}</td>',
                "</tr>",
                "</table>",
            ]
        )

        differences = codepoint_differences(reference, prediction)
        if differences:
            body = "\n".join(
                ["  index  reference            ->  prediction"] + differences
            )
            parts.append(f'<pre class="codepoints">{_esc(body)}</pre>')

    parts.extend(["</body>", "</html>", ""])
    return "\n".join(parts)
