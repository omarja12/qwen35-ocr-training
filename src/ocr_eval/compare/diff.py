"""Per-page deltas between two runs, and the commensurability guard.

A comparison only exists when the two runs are commensurable on all four
counts: same `policy_version`, same `corpus_manifest_digest`, same split set,
same **page set**. Anything else exits 3 naming the mismatch. There is no
warn-and-continue path — a warning gets ignored the one time it matters
(research R-004).

The page-set condition is the one FR-018 makes reachable. Two runs produced with
different `--pages` or `--filter` arguments over the same corpus share a policy
version, share a manifest digest and share a split set; every other guard passes,
and comparing them anyway reports the difference between two populations as if
it were the difference between two models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ocr_eval.errors import IncommensurableError, UsageError


@dataclass
class Run:
    """One loaded run directory."""

    path: Path
    manifest: dict
    summary: dict
    pages: dict[str, dict]

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def model_version(self) -> str:
        return self.manifest["model_version"]

    @property
    def policy_version(self) -> str:
        return self.manifest["policy"]["policy_version"]

    @property
    def splits(self) -> set[str]:
        return {s["split"] for s in self.summary["splits"]}

    def split(self, name: str) -> dict | None:
        for entry in self.summary["splits"]:
            if entry["split"] == name:
                return entry
        return None


def load_run(path: str | Path) -> Run:
    """Read a run directory. Never re-reads the corpus."""
    run_path = Path(path)
    if not run_path.is_dir():
        raise UsageError(f"not a run directory: {run_path}")

    def _read(name: str) -> dict:
        target = run_path / name
        if not target.exists():
            raise UsageError(f"{run_path} is not a run directory: {name} is missing")
        return json.loads(target.read_text(encoding="utf-8"))

    pages_file = run_path / "pages.jsonl"
    if not pages_file.exists():
        raise UsageError(f"{run_path} is not a run directory: pages.jsonl is missing")

    pages = {}
    for line in pages_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pages[row["page_id"]] = row

    return Run(
        path=run_path,
        manifest=_read("manifest.json"),
        summary=_read("summary.json"),
        pages=pages,
    )


def assert_commensurable(baseline: Run, candidate: Run) -> None:
    """The four conditions. Any mismatch is exit 3, naming what differed."""
    problems: list[str] = []

    if baseline.policy_version != candidate.policy_version:
        problems.append(
            f"policy_version differs: baseline {baseline.policy_version!r} vs "
            f"candidate {candidate.policy_version!r}. Two results produced under "
            f"different scoring rules are not comparable; re-score one of them."
        )

    base_digest = baseline.manifest["corpus_manifest_digest"]
    cand_digest = candidate.manifest["corpus_manifest_digest"]
    if base_digest != cand_digest:
        problems.append(
            f"corpus_manifest_digest differs: {base_digest} vs {cand_digest}. "
            f"The two runs are over different corpora."
        )

    if baseline.splits != candidate.splits:
        only_base = sorted(baseline.splits - candidate.splits)
        only_cand = sorted(candidate.splits - baseline.splits)
        problems.append(
            f"split set differs: only in baseline {only_base}, "
            f"only in candidate {only_cand}."
        )

    only_in_baseline = sorted(set(baseline.pages) - set(candidate.pages))
    only_in_candidate = sorted(set(candidate.pages) - set(baseline.pages))
    if only_in_baseline or only_in_candidate:
        problems.append(
            f"page set differs: {len(only_in_baseline)} page(s) only in baseline, "
            f"{len(only_in_candidate)} only in candidate. This is what --pages and "
            f"--filter produce (FR-018): the digests and splits match, but the two "
            f"runs cover different populations."
        )
        for page_id in only_in_baseline[:20]:
            problems.append(f"  only in baseline: {page_id}")
        for page_id in only_in_candidate[:20]:
            problems.append(f"  only in candidate: {page_id}")

    if problems:
        raise IncommensurableError(
            "refusing to compare incommensurable runs", problems=problems
        )


@dataclass(frozen=True)
class PageDelta:
    page_id: str
    split: str
    baseline_cer: float
    candidate_cer: float
    delta_cer: float
    outcome_changed: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def page_deltas(baseline: Run, candidate: Run) -> list[PageDelta]:
    """Join on page_id. Probe pages carry no CER and are compared by outcome only."""
    deltas: list[PageDelta] = []
    for page_id in sorted(baseline.pages):
        before = baseline.pages[page_id]
        after = candidate.pages[page_id]

        before_cer = before.get("metrics", {}).get("strict", {}).get("cer")
        after_cer = after.get("metrics", {}).get("strict", {}).get("cer")

        outcome_changed = (
            f"{before['outcome']} -> {after['outcome']}"
            if before["outcome"] != after["outcome"]
            else None
        )

        if before_cer is None or after_cer is None:
            # A probe page. It has no distance to measure, so it appears only
            # when its contract outcome changed — which is the thing worth
            # seeing about a probe page anyway.
            if outcome_changed:
                deltas.append(
                    PageDelta(page_id, before["split"], 0.0, 0.0, 0.0, outcome_changed)
                )
            continue

        deltas.append(
            PageDelta(
                page_id=page_id,
                split=before["split"],
                baseline_cer=before_cer,
                candidate_cer=after_cer,
                delta_cer=after_cer - before_cer,
                outcome_changed=outcome_changed,
            )
        )
    return deltas


def regressions(deltas: Sequence[PageDelta]) -> list[PageDelta]:
    """Pages that got worse, worst delta first, ties by page_id."""
    worse = [d for d in deltas if d.delta_cer > 0]
    worse.sort(key=lambda d: (-d.delta_cer, d.page_id))
    return worse


def improvements(deltas: Sequence[PageDelta]) -> list[PageDelta]:
    """Pages that got better, biggest improvement first, ties by page_id."""
    better = [d for d in deltas if d.delta_cer < 0]
    better.sort(key=lambda d: (d.delta_cer, d.page_id))
    return better
