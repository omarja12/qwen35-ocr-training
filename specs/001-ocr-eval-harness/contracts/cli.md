# Contract: Command Line Interface

**Feature**: `001-ocr-eval-harness` | **Version**: 1.0 | **Date**: 2026-09-16

The harness is a CLI over a library. This file is the contract: anything here is
depended upon by users and by the project's scripts, and changing it is a breaking
change.

Invocation, equivalently:

```bash
ocr-eval <command> [options]          # installed console script
python -m ocr_eval <command> [options]  # from a checkout, PYTHONPATH=src
```

Global: `--version` prints `ocr-eval <tool_version> (policy <policy_version>)` and
exits 0.

---

## Exit codes

One mapping, used by every command.

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Input integrity failure — missing predictions, duplicate ids, bad UTF-8, empty reference on a legible page, a legible reference that normalises to empty at any level, unknown `kind`, subset id not in the manifest. Every problem found is listed, not just the first (FR-016). |
| `2` | Usage error — bad flags, unreadable path, missing required argument. `argparse` default. |
| `3` | Incommensurable comparison — differing `policy_version`, `corpus_manifest_digest`, split set, or page set (research R-004). |
| `4` | Output collision — the run directory already exists and `--force` was not given. |

A non-zero exit always writes a human-readable explanation to stderr. Nothing is
written to stdout on failure.

---

## `ocr-eval validate`

Check that a corpus manifest and a prediction set line up, without scoring anything.
Exists so the corpus can be checked in seconds, before a scoring run or as a data
pipeline gate.

```bash
ocr-eval validate \
  --manifest <path> \
  --predictions <path> \
  [--corpus-root <dir>] \
  [--pages <file>] [--filter <expr>] \
  [--json]
```

| Flag | Required | Meaning |
|---|---|---|
| `--manifest` | yes | Corpus manifest JSONL. See `corpus-manifest.schema.json`. |
| `--predictions` | yes | Directory of `<page_id>.txt`, or a JSONL of `{page_id, text}`. Form auto-detected from whether the path is a directory. |
| `--corpus-root` | no | Base for relative `reference_path`. Defaults to the manifest's directory. |
| `--pages` | no | File of page ids, one per line, to restrict the check to (FR-018). |
| `--filter` | no | Metadata filter — see **Filter expressions** below. |
| `--json` | no | Machine-readable findings on stdout instead of prose. |

**Output**: on success, a one-line count to stdout and exit 0. On failure, every
problem grouped by kind, and exit 1.

---

## `ocr-eval score`

Produce a run directory. The core command (User Story 1).

```bash
ocr-eval score \
  --manifest <path> \
  --predictions <path> \
  --model-version <string> \
  --out <run_dir> \
  [--corpus-root <dir>] \
  [--run-id <string>] \
  [--config <path>] \
  [--pages <file>] [--filter <expr>] \
  [--worst <N>] [--jobs <N>] [--force] [--no-html]
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--model-version` | yes | — | Checkpoint identifier recorded in the manifest. The thing a reader needs six months later (SC-008). |
| `--out` | yes | — | Run directory to create. Refuses to overwrite (exit 4) unless `--force`. |
| `--run-id` | no | derived | Derived from `model_version` + input digests when omitted. Never derived from a timestamp. |
| `--config` | no | `configs/eval/default.json` | Thresholds. Echoed in full into `manifest.json`. JSON rather than YAML so the harness needs no parser beyond the stdlib. |
| `--worst` | no | `50` | How many worst-scoring pages to surface (FR-017). |
| `--jobs` | no | `1` | Worker processes. May reorder execution; may not reorder output (research R-011). |
| `--no-html` | no | off | Skip `worst_pages.html`. |

**Behaviour**

1. Runs the full `validate` check first. Any integrity problem stops the run before
   a single page is scored (FR-016).
2. Scores every page in scope at all three normalisation levels.
3. Writes the run directory described in [data-model.md](../data-model.md).
4. Prints the strict headline per split to stdout, and the `[UNREADABLE]` contract
   figures.

**Guarantee**: two `score` invocations with identical inputs, config and version
produce byte-identical output for every file except `run.meta.json` (SC-003).

---

## `ocr-eval compare`

Two runs to a decision (User Story 2).

```bash
ocr-eval compare \
  --baseline <run_dir> \
  --candidate <run_dir> \
  [--out <path>] \
  [--config <path>] \
  [--primary-split <name>] \
  [--top <N>] \
  [--format md|json]
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--baseline` | yes | — | The run being compared against — the untrained baseline, or the previous best. |
| `--candidate` | yes | — | The new run. |
| `--out` | no | stdout | Where to write. `comparison.json` plus `comparison.md` when a directory is given. |
| `--primary-split` | no | `synthetic_heldout` | Which split the PROMOTE rule reads (research R-014). |
| `--top` | no | `25` | How many regressions and improvements to list. |

**Behaviour**

1. Refuses incommensurable runs — differing `policy_version`, differing
   `corpus_manifest_digest`, a different split set, or a different **page set** —
   with exit 3 naming the mismatch. There is no warn-and-continue path. The page-set
   check matters because `--pages` and `--filter` make two runs over the same corpus
   at the same policy version routine (FR-018); their digests match and their splits
   match, and comparing them anyway would average two different populations into one
   delta. Unmatched ids are named on stderr, truncated to the first 20 per side.
2. States, per split, each headline metric on both sides, the delta, and the
   direction **as a word** (`better` / `worse` / `unchanged`).
3. Lists the pages that regressed, worst delta first (FR-008 scenario 2).
4. Emits a verdict of `PROMOTE`, `REJECT` or `INCONCLUSIVE` with every rule that
   fired, and a paired bootstrap interval on the CER delta.

**Exit code is 0 for all three verdicts.** A `REJECT` is a successful comparison,
not a failed command. Scripts read the verdict from the artifact, not from `$?`.

---

## `ocr-eval report`

Re-render human-readable output from an existing run, without rescoring
(User Story 3).

```bash
ocr-eval report \
  --run <run_dir> \
  [--worst <N>] \
  [--breakdown font|distortion|source|all] \
  [--split <name>] \
  [--format md|html|json] \
  [--out <path>]
```

Reads `pages.jsonl` and `summary.json` only. Never re-reads the corpus, never
recomputes a metric — so a report can be re-cut with a different `--worst` or a
single breakdown dimension in under a second, and cannot disagree with the run it
came from.

Breakdown rows are ranked worst-first, so the worst-performing font and distortion
are each the first row of their table (FR-007, SC-005).

---

## Filter expressions

`--filter` restricts a run to a subset without re-running the whole set (FR-018).
Deliberately small — this is a filter, not a query language.

```text
<field><op><value>[,<field><op><value>...]
```

- Fields: `split`, `kind`, `source`, `font`, `distortion`, `is_augmented`.
- Operators: `=` (equals), `!=` (not equals), `~` (substring).
- `font` and `distortion` are list fields; `=` means "contains".
- Comma-separated conditions are ANDed. There is no OR — use two runs.

```bash
--filter 'split=gold_scans'
--filter 'font=Amiri,is_augmented=false'
--filter 'distortion~blur'
```

The parsed filter is recorded in `manifest.json` as structured data, so the run's
scope is reconstructable from the artifact (SC-008).

---

## Stdout / stderr discipline

- Machine-readable output goes to stdout, always valid JSON when `--json` or
  `--format json` is given.
- Progress, warnings and errors go to stderr.
- No colour codes when stdout is not a TTY.
- No network access is attempted by any command, ever (FR-015).
