"""Typed failures and the single exit-code mapping.

One mapping, used by every command, defined once here so that `cli.py` never
invents a code and no caller has to remember which number means what
(contracts/cli.md).

The distinction the constitution draws (principle IV) is between **input
integrity**, which stops the run, and **measured system behaviour**, which is
recorded and flagged. Nothing in this module describes model behaviour: an empty
prediction and a hallucination are data, not errors.
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

EXIT_OK = 0
EXIT_INPUT_INTEGRITY = 1
EXIT_USAGE = 2
EXIT_INCOMMENSURABLE = 3
EXIT_OUTPUT_COLLISION = 4


class OcrEvalError(Exception):
    """Base for every failure that should end the process with a message.

    A non-zero exit always writes a human-readable explanation to stderr and
    nothing to stdout, so a caller parsing stdout as JSON never has to guard
    against an error message arriving in the stream it is reading.
    """

    exit_code = EXIT_INPUT_INTEGRITY

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        # Populated when a check finds more than one thing wrong. The
        # constitution requires reporting every problem found, not the first:
        # fixing a corpus one error per run is how a data pipeline becomes a
        # day's work instead of an afternoon's.
        self.problems: list[str] = list(problems or ())

    def render(self) -> str:
        if not self.problems:
            return self.message
        lines = [self.message, ""]
        lines.extend(f"  - {p}" for p in self.problems)
        lines.append("")
        lines.append(f"{len(self.problems)} problem(s) found.")
        return "\n".join(lines)


class InputIntegrityError(OcrEvalError):
    """Malformed, missing or mismatched input. Never scored, never imputed."""


class UsageError(OcrEvalError):
    """Bad flags, unreadable path, missing required argument."""

    exit_code = EXIT_USAGE


class IncommensurableError(OcrEvalError):
    """Two runs that must not be compared.

    Differing policy_version, corpus_manifest_digest, split set or page set.
    There is no warn-and-continue path: a warning gets ignored the one time it
    matters (research R-004).
    """

    exit_code = EXIT_INCOMMENSURABLE


class OutputCollisionError(OcrEvalError):
    """The run directory already exists and --force was not given.

    Run directories are written once and never mutated, so overwriting one
    silently would destroy the provenance of whatever was there.
    """

    exit_code = EXIT_OUTPUT_COLLISION


def fail(error: OcrEvalError) -> int:
    """Write an error's explanation to stderr and return its exit code."""
    print(error.render(), file=sys.stderr)
    return error.exit_code


def format_problems(problems: Iterable[str], limit: int = 20) -> list[str]:
    """Truncate a problem list for display, saying how many were hidden."""
    items = list(problems)
    if len(items) <= limit:
        return items
    hidden = len(items) - limit
    return items[:limit] + [f"... and {hidden} more"]
