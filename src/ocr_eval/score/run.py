"""Orchestrate a scoring run and write the run directory.

A run directory is written once and never mutated. `score` refuses to overwrite
one (exit 4) unless `--force`, because overwriting silently would destroy the
provenance of whatever was there.

Everything that genuinely varies per run — wall-clock, host, elapsed — is
segregated into `run.meta.json`, the one file outside the byte-identity
guarantee. That segregation is what lets SC-003 be an assertion about every
other file rather than an aspiration (research R-011).
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Sequence

from ocr_eval import __version__
from ocr_eval.aggregate.summary import build_summary
from ocr_eval.errors import (
    EXIT_OK,
    InputIntegrityError,
    OutputCollisionError,
    UsageError,
    format_problems,
)
from ocr_eval.io import writer
from ocr_eval.metrics.distance import ENGINE
from ocr_eval.normalise.tables import policy_object
from ocr_eval.report import markdown as markdown_module
from ocr_eval.report import worst_pages as worst_pages_module
from ocr_eval.score.page import score_page
from ocr_eval.score.subset import Condition, describe
from ocr_eval.score.validate import validate


def derive_run_id(model_version: str, manifest_digest: str, predictions_digest: str) -> str:
    """Derive a run id from what was scored, never from a timestamp.

    A timestamp would make two identical runs produce two different ids, which
    is precisely the kind of incidental variation SC-003 exists to remove.
    """
    hasher = hashlib.sha256()
    for part in (model_version, manifest_digest, predictions_digest):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in model_version)[:40]
    return f"{slug}-{hasher.hexdigest()[:12]}"


def score_run(
    manifest_path: str | Path,
    predictions_path: str | Path,
    model_version: str,
    out_dir: str | Path,
    config,
    corpus_root: str | Path | None = None,
    run_id: str | None = None,
    page_ids: Sequence[str] | None = None,
    conditions: Sequence[Condition] | None = None,
    worst: int = 50,
    jobs: int = 1,
    force: bool = False,
    no_html: bool = False,
) -> int:
    """Score a prediction set and write a run directory. Returns an exit code."""
    started = time.time()
    out = Path(out_dir)
    if jobs < 1:
        raise UsageError(f"--jobs must be at least 1, got {jobs}")

    # Checked before any work, so a collision costs no time.
    if out.exists() and any(out.iterdir()) and not force:
        raise OutputCollisionError(
            f"run directory already exists and is not empty: {out}\n"
            "Run directories are written once and never mutated. Pass --force to "
            "overwrite, or choose another --out."
        )

    # 1. Validate the whole input set. Any integrity problem stops the run
    #    before a single page is scored (FR-016, constitution IV).
    result = validate(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        corpus_root=corpus_root,
        page_ids=page_ids,
        conditions=conditions,
    )
    if not result.ok:
        raise InputIntegrityError(
            f"validation failed: {len(result.problems)} problem(s). Nothing was scored.",
            problems=format_problems(result.problems, limit=100),
        )

    assert result.predictions is not None

    # 2. Score every page in scope, across `jobs` worker processes if asked.
    pages = result.pages
    work = (
        pages,
        [result.references[p.page_id] for p in pages],
        [result.predictions.texts[p.page_id] for p in pages],
        repeat(config),
    )
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            chunk = max(1, len(pages) // (jobs * 4))
            page_results = list(pool.map(score_page, *work, chunksize=chunk))
    else:
        page_results = list(map(score_page, *work))
    # Sorted by page_id in codepoint order, independent of the order work
    # completed. This is what lets --jobs reorder execution without reordering
    # output (research R-011).
    page_results.sort(key=lambda r: r.page_id)

    resolved_run_id = run_id or derive_run_id(
        model_version, result.manifest_digest, result.predictions.digest
    )

    policy = policy_object()
    summary = build_summary(
        page_results,
        run_id=resolved_run_id,
        model_version=model_version,
        tool_version=__version__,
        policy=policy,
        min_pages_per_breakdown_row=config.min_pages_per_breakdown_row,
    )

    manifest = {
        "run_id": resolved_run_id,
        "model_version": model_version,
        "corpus_manifest_digest": result.manifest_digest,
        "predictions_digest": result.predictions.digest,
        "predictions_form": result.predictions.form,
        "pages_scored": len(page_results),
        "subset_filter": describe(page_ids, conditions),
        "policy": policy,
        "config": config.echo(),
        "config_source": config.source,
        "tool_version": __version__,
        "distance_engine": ENGINE,
    }

    # 3. Write the run directory.
    writer.ensure_dir(out)
    writer.write_json(out / "manifest.json", manifest)
    writer.write_jsonl(out / "pages.jsonl", [r.to_dict() for r in page_results])
    writer.write_json(out / "summary.json", summary)
    writer.write_text(out / "summary.md", markdown_module.render_summary(summary))

    writer.write_text(
        out / "worst_pages.md",
        worst_pages_module.render_markdown(page_results, result.references,
                                           result.predictions.texts, worst),
    )
    if not no_html:
        from ocr_eval.report import html as html_module

        writer.write_text(
            out / "worst_pages.html",
            html_module.render(page_results, result.references,
                               result.predictions.texts, worst),
        )

    # run.meta.json: the only file outside the byte-identity guarantee.
    writer.write_json(
        out / "run.meta.json",
        {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - started, 3),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "jobs": jobs,
        },
    )

    _print_headline(summary, out)
    return EXIT_OK


def _print_headline(summary: dict, out: Path) -> None:
    """The strict headline per split, plus the contract figures."""
    print(f"run_id: {summary['run_id']}")
    print(f"written to: {out}")
    for split in summary["splits"]:
        name = split["split"]
        headline = split["headline"]
        if headline is None:
            print(
                f"  {name}: {split['pages']} page(s), none scoreable "
                f"(probe pages carry no reference text to measure against)"
            )
        else:
            print(
                f"  {name}: strict CER {headline['cer']:.4f}  "
                f"WER {headline['wer']:.4f}  "
                f"exact {headline['exact_match_rate']:.4f}  "
                f"({split['scored_pages']} scored)"
            )
        contract = split["contract"]
        if contract["probe_pages"]:
            rate = contract["hallucination_rate"]
            print(
                f"      hallucination {contract['hallucination']}/"
                f"{contract['probe_pages']} ({rate:.4f})"
            )
        if contract["false_refusal"]:
            print(
                f"      false refusals {contract['false_refusal']} "
                f"(counted separately from hallucination)"
            )
    for note in summary["notes"]:
        print(f"note: {note}")
