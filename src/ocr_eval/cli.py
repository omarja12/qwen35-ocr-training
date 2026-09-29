"""Command line interface: validate | score | compare | report.

Stdout/stderr discipline, enforced here so no subcommand has to remember it:

  - machine-readable output goes to stdout, and is valid JSON whenever --json or
    --format json is given;
  - progress, warnings and errors go to stderr;
  - a non-zero exit writes its explanation to stderr and nothing to stdout, so a
    caller parsing stdout never has to guard against an error arriving in the
    stream it is reading.

No command attempts network access, ever (FR-015). There is nothing to disable:
the harness never loads the model, so no code path here has a reason to open a
socket, and tests/unit/test_no_network.py asserts that no module imports a
network client.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocr_eval import __version__
from ocr_eval.errors import EXIT_OK, OcrEvalError, UsageError, fail, format_problems
from ocr_eval.normalise.tables import POLICY_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-eval",
        description=(
            "Score OCR predictions against a reference corpus, compare "
            "checkpoints, and measure hallucination on unreadable input."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ocr-eval {__version__} (policy {POLICY_VERSION})",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # --- validate ---------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="check a corpus manifest and a prediction set line up, without scoring",
    )
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--predictions", required=True)
    validate_parser.add_argument("--corpus-root")
    validate_parser.add_argument("--pages")
    validate_parser.add_argument("--filter", dest="filter_expr")
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.set_defaults(handler=_cmd_validate)

    # --- score ------------------------------------------------------------
    score_parser = subparsers.add_parser(
        "score", help="produce a run directory from a corpus and a prediction set"
    )
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--model-version", required=True)
    score_parser.add_argument("--out", required=True)
    score_parser.add_argument("--corpus-root")
    score_parser.add_argument("--run-id")
    score_parser.add_argument("--config")
    score_parser.add_argument("--pages")
    score_parser.add_argument("--filter", dest="filter_expr")
    score_parser.add_argument("--worst", type=int, default=50)
    score_parser.add_argument("--jobs", type=int, default=1)
    score_parser.add_argument("--force", action="store_true")
    score_parser.add_argument("--no-html", action="store_true")
    score_parser.set_defaults(handler=_cmd_score)

    # --- compare ----------------------------------------------------------
    compare_parser = subparsers.add_parser(
        "compare", help="turn two run directories into a promote-or-reject decision"
    )
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--out")
    compare_parser.add_argument("--config")
    compare_parser.add_argument("--primary-split", default="synthetic_heldout")
    compare_parser.add_argument("--top", type=int, default=25)
    compare_parser.add_argument("--format", dest="fmt", choices=("md", "json"), default="md")
    compare_parser.set_defaults(handler=_cmd_compare)

    # --- report -----------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report", help="re-render human-readable output from an existing run"
    )
    report_parser.add_argument("--run", required=True)
    report_parser.add_argument("--worst", type=int, default=50)
    report_parser.add_argument(
        "--breakdown", choices=("font", "distortion", "source", "all"), default="all"
    )
    report_parser.add_argument("--split")
    report_parser.add_argument(
        "--format", dest="fmt", choices=("md", "html", "json"), default="md"
    )
    report_parser.add_argument("--out")
    report_parser.set_defaults(handler=_cmd_report)

    return parser


def _subset_args(args: argparse.Namespace):
    """Resolve --pages and --filter into (page_ids, conditions)."""
    from ocr_eval.score.subset import parse_filter, read_pages_file

    page_ids = read_pages_file(args.pages) if args.pages else None
    conditions = parse_filter(args.filter_expr) if args.filter_expr else None
    return page_ids, conditions


def _cmd_validate(args: argparse.Namespace) -> int:
    from ocr_eval.io.writer import dumps
    from ocr_eval.score.validate import validate

    page_ids, conditions = _subset_args(args)
    result = validate(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        corpus_root=args.corpus_root,
        page_ids=page_ids,
        conditions=conditions,
    )

    if result.ok:
        if args.json:
            print(dumps({"ok": True, "pages_in_scope": result.pages_in_scope, "problems": []}))
        else:
            print(f"OK: {result.pages_in_scope} page(s) in scope, every one with a prediction.")
        return EXIT_OK

    if args.json:
        # Findings on stdout when asked for machine-readably, but the exit code
        # still says failure so a pipeline gate cannot miss it.
        print(dumps({"ok": False, "pages_in_scope": result.pages_in_scope,
                     "problems": result.problems}))
        print(f"{len(result.problems)} problem(s) found.", file=sys.stderr)
        return 1

    raise OcrEvalError(
        f"validation failed: {len(result.problems)} problem(s) in "
        f"{result.pages_in_scope} page(s) in scope",
        problems=format_problems(result.problems, limit=100),
    )


def _cmd_score(args: argparse.Namespace) -> int:
    from ocr_eval.config import load_config
    from ocr_eval.score.run import score_run

    page_ids, conditions = _subset_args(args)
    config = load_config(args.config)
    return score_run(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        model_version=args.model_version,
        out_dir=args.out,
        corpus_root=args.corpus_root,
        run_id=args.run_id,
        config=config,
        page_ids=page_ids,
        conditions=conditions,
        worst=args.worst,
        jobs=args.jobs,
        force=args.force,
        no_html=args.no_html,
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    from ocr_eval.compare.driver import compare_runs
    from ocr_eval.config import load_config

    config = load_config(args.config)
    return compare_runs(
        baseline_dir=args.baseline,
        candidate_dir=args.candidate,
        out=args.out,
        config=config,
        primary_split=args.primary_split,
        top=args.top,
        fmt=args.fmt,
    )


def _cmd_report(args: argparse.Namespace) -> int:
    from ocr_eval.report.driver import render_report

    return render_report(
        run_dir=args.run,
        worst=args.worst,
        breakdown=args.breakdown,
        split=args.split,
        fmt=args.fmt,
        out=args.out,
    )


def _force_utf8_streams() -> None:
    """Print UTF-8 whatever the console's default encoding is.

    Windows consoles still default to a legacy code page (cp1252 here), which
    cannot encode Arabic — so printing a report to stdout raised
    UnicodeEncodeError and the command died with a traceback. Every artifact the
    harness writes is already UTF-8 through io/writer.py; this makes the
    terminal path agree with the file path.

    plan.md targets Linux and Windows both, so this is a correctness fix rather
    than a convenience: a report that cannot be printed on one of the two target
    platforms is not a report.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "handler", None):
        parser.print_help(sys.stderr)
        return 2

    try:
        return args.handler(args)
    except OcrEvalError as exc:
        return fail(exc)
    except FileNotFoundError as exc:
        return fail(UsageError(f"file not found: {exc.filename or exc}"))
    except BrokenPipeError:  # pragma: no cover - `| head` on the reader's side
        return EXIT_OK
