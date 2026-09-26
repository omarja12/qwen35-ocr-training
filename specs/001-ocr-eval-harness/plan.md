# Implementation Plan: OCR Evaluation Harness

**Branch**: `001-ocr-eval-harness` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-ocr-eval-harness/spec.md`

## Summary

A CPU-only, fully offline Python library and CLI that scores pre-existing OCR
predictions against a reference corpus and produces reports a reader can act on
without opening the underlying data.

The harness never loads the model. It consumes predictions that already exist on
disk, which is what lets User Story 1 be built and used today with no cluster
access — and what lets the same harness score any other OCR system used as a
comparator.

Four commands: `validate` (check the corpus and predictions line up, loudly),
`score` (produce an immutable run directory), `compare` (two runs to a
promote/reject recommendation), `report` (re-render human-readable output from a run
without rescoring).

Every result is reported at all three normalisation levels the spec fixes — strict,
diacritic-insensitive, skeleton — with strict as the headline, alongside an
order-corrected CER and a reading-order accuracy figure so that recognition failure
is distinguishable from sequencing failure. The `[UNREADABLE]` contract is scored as
a four-cell classification, keeping hallucination and false refusal apart. Reports
are byte-identical across runs; everything that genuinely varies per run is
segregated into a file outside that guarantee.

**The artifact shape is indivisible.** The schemas in `contracts/` mark `flags`,
`length_ratio`, the `contract` block including its two rates, and the `order` block
as required. Any implementation that writes a run directory must therefore emit all
of them, even where the story that gives them meaning lands later. Concretely: the
first slice to produce a run directory owns the full shape; later work deepens the
semantics — probe fixtures, threshold tuning, the reporting surface — without
changing it. Slicing the schema by story instead would leave the earliest and most
important deliverable emitting artifacts that fail their own contract.

## Technical Context

**Language/Version**: Python 3.10+ (3.11 is the target, matching the training image)

**Primary Dependencies**: None required at runtime — stdlib only. `rapidfuzz` is used
for edit distance when importable, with a bundled pure-Python Myers bit-parallel
fallback that produces identical numbers (research R-001). `pytest` for tests only.

**Storage**: Files. JSONL for per-page results, JSON for summary / manifest /
comparison, Markdown and HTML for human-readable reports. No database.

**Testing**: `pytest`. Unit tests on the normalisation tables and metrics; golden-file
tests that assert byte-identity (SC-003); a fixture corpus of ~20 synthetic pages
committed to the repo so the suite runs with no external data.

**Target Platform**: Ordinary CPU workstation — Linux and Windows both. No GPU, no
network access of any kind (FR-015).

**Project Type**: Single project — library plus CLI.

**Performance Goals**: 1,000 pages scored in under 5 minutes single-process (SC-002).
Expected actual is well under one minute; the budget exists so nobody skips
evaluating. SC-002 says "an ordinary workstation", which is not reproducible on its
own, so the measurement records the machine it was taken on — CPU model, core count,
RAM — alongside the elapsed time. A number without its machine is not a number a
stranger can check (constitution II).

**Constraints**: Offline, no telemetry, no model loading. Byte-identical reports for
identical inputs, 100% of the time (SC-003). Every added third-party wheel is a
build risk for the air-gapped image, so the default is to add none. Every directory
and file the harness creates is explicitly group-writable (`0775` / `0664`), because
the image runs as an arbitrary non-root UID (constitution principle V, research
R-017) — permissions are set explicitly rather than inherited from a umask.

**Scale/Scope**: Tens of thousands of pages per corpus; roughly 50 lines per page;
reports routinely read six months after the run that produced them (SC-008).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` **v1.0.0**, ratified 2026-09-16.

| Principle | Verdict | Evidence |
|---|---|---|
| **I. Measurement Before Claims** | **PASS** | This feature is the mechanism the principle depends on. Strict is the single headline field (FR-003a); the cross-split aggregate key does not exist in the summary schema; `compare` is the only path to a promote/reject verdict; hallucination and false refusal are separate counts. |
| **II. Reproducible by a Stranger** | **PASS** | `manifest.json` embeds model version, input digests, full config, tool and policy versions (R-011). Byte-identity is a tested guarantee, not an aspiration. `compare` refuses across policy versions rather than warning. |
| **III. Air-Gapped by Construction** | **PASS** | Zero required runtime dependencies. `rapidfuzz` is optional, already transitively present via `jiwer`, and cannot change a number (R-001). No network path exists to close — the harness never loads the model (R-012). |
| **IV. Fail Loudly, Never Silently** | **PASS** | Whole-set validation before any page is scored, reporting every problem rather than the first (R-009). The integrity-vs-behaviour split the principle names is the same split the design draws: a missing prediction stops the run; an empty prediction is scored and flagged. |
| **V. The Image Carries Software Only** | **PASS, with one constraint now binding** | No weights, data or credentials involved. **New**: outputs must be group-writable for an arbitrary non-root UID — see R-017 and the Constraints line above. The committed `tests/fixtures/mini_corpus/` and `tests/golden/expected/` fall under the principle's explicit exception for small fixed test fixtures; they are hand-built inputs and pinned expectations, not run output. |
| **VI. Specify Software, Not Operations** | **PASS** | The harness is exactly the class of work the principle assigns to Spec Kit. No operational procedure is specified here. FR-010a's cross-feature consequence is recorded where dataset preparation will see it, as the principle requires. |
| **Delivery and Technology Constraints** | **PASS** | CPU-only by construction, so judging a model never waits on the cluster. No dependency added to the training image. |
| **Development Workflow and Quality Gates** | **PASS** | Every metric, normalisation and validation rule has a unit test; the determinism guarantee is asserted by running twice and comparing bytes (`tests/golden/test_byte_identity.py`); the suite runs offline against committed fixtures. |

**Post-Phase-1 re-check**: **PASS, no violations.** The design predates the
constitution but was derived from the same source rules, so principles I–IV and VI
were already satisfied without change.

Ratification did tighten one thing that had been invisible: principle V's
arbitrary-UID rule reaches the harness, which writes run directories. Permissions are
now set explicitly in `io/writer.py` rather than left to the ambient umask — a run
directory created under a umask of `077` would be unreadable to the next pod, and
that failure would surface as a confusing permission error rather than as a design
mistake. Recorded as research R-017.

Complexity Tracking is empty because there is nothing to justify.

## Project Structure

### Documentation (this feature)

```text
specs/001-ocr-eval-harness/
├── plan.md                        # This file
├── spec.md                        # Feature specification
├── research.md                    # Phase 0 output — 18 decisions (R-017 added at
│                                  # ratification, R-018 by /speckit-analyze)
├── data-model.md                  # Phase 1 output — entities and invariants
├── quickstart.md                  # Phase 1 output — runnable validation scenarios
├── checklists/
│   └── requirements.md            # Spec quality checklist (already passing)
├── contracts/                     # Phase 1 output
│   ├── cli.md                     # Command surface, exit codes, flags
│   ├── normalisation-policy.md    # The three levels, exactly, versioned
│   ├── corpus-manifest.schema.json
│   ├── page-result.schema.json
│   ├── run-summary.schema.json
│   └── comparison.schema.json
└── tasks.md                       # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
pyproject.toml                     # new — src layout, console script, optional extras

src/ocr_eval/
├── __init__.py
├── __main__.py                    # python -m ocr_eval
├── cli.py                         # argparse; validate | score | compare | report
├── config.py                      # thresholds, defaults, config file load + echo
├── errors.py                      # typed failures; one exit-code mapping
├── io/
│   ├── __init__.py
│   ├── manifest.py                # corpus manifest reader + schema validation
│   ├── predictions.py             # directory-of-txt and JSONL readers
│   └── writer.py                  # the only place JSON/JSONL/text is written;
│                                  # owns byte-identity AND 0775/0664 permissions
├── normalise/
│   ├── __init__.py
│   ├── tables.py                  # codepoint tables; POLICY_VERSION
│   └── levels.py                  # strict | diacritic_insensitive | skeleton
├── metrics/
│   ├── __init__.py
│   ├── distance.py                # rapidfuzz path + pure-Python Myers fallback
│   ├── rates.py                   # CER / WER / exact match; micro and macro
│   ├── order.py                   # line matching, order-corrected CER, order accuracy
│   └── flags.py                   # runaway length, repetition
├── score/
│   ├── __init__.py
│   ├── validate.py                # whole-set integrity checks (fail loud)
│   ├── page.py                    # score one page at all three levels
│   ├── run.py                     # orchestrate a run, write the run directory
│   └── contract.py                # [UNREADABLE] four-cell classification
├── aggregate/
│   ├── __init__.py
│   ├── summary.py                 # per-split headline + aggregates
│   └── breakdown.py               # by font, distortion, source; ranked worst-first
├── compare/
│   ├── __init__.py
│   ├── diff.py                    # per-page deltas, regressions
│   ├── bootstrap.py               # paired bootstrap, fixed seed
│   └── verdict.py                 # PROMOTE | REJECT | INCONCLUSIVE
└── report/
    ├── __init__.py
    ├── markdown.py                # summary + breakdowns for terminal reading
    ├── worst_pages.py             # side-by-side + codepoint diff list
    └── html.py                    # RTL-correct side-by-side

configs/eval/
├── default.json                   # thresholds; echoed into every run manifest
│                                  # JSON, not YAML — PyYAML would be a runtime dep
└── README.md

tests/
├── fixtures/
│   └── mini_corpus/               # ~20 pages: clean, misordered, probe, runaway
├── unit/
│   ├── test_normalise.py          # per-level codepoint behaviour
│   ├── test_distance.py           # rapidfuzz path == fallback path
│   ├── test_rates.py
│   ├── test_order.py
│   ├── test_flags.py
│   └── test_contract.py           # the four-cell matrix
├── integration/
│   ├── test_validate_failures.py  # every hard failure in research R-009
│   ├── test_score_run.py
│   ├── test_compare.py
│   └── test_subset.py             # FR-018
└── golden/
    ├── test_byte_identity.py      # SC-003 — run twice, compare bytes
    └── expected/                  # committed golden artifacts
```

**Structure Decision**: Single project, src layout. `src/` already exists in the repo
(holding only `.gitkeep`) and `configs/` is already the established home for
configuration, so the harness lands in the directories the repo already declared
rather than inventing new top-level ones.

The module split follows the seams the spec draws rather than a generic
models/services/cli split. `normalise/` is isolated because FR-003 and FR-004 make its
tables a versioned artifact whose changes invalidate old comparisons. `metrics/` is
pure functions over strings, which keeps the parts that produce every number in the
report trivially testable. `io/writer.py` is the single place anything is serialised,
because SC-003's byte-identity guarantee is only enforceable if there is exactly one
writer to enforce it in — and the same argument now applies to the group-writable
permission rule, which is why both live there.

**Consequence for the training image**: `Containerfile` already copies `src/` to
`/opt/ocr-training/src/`, so the harness ships inside the image without any change to
the build. It is not pip-installed there — the image installs `requirements.txt`
only — so in-pod invocation is `PYTHONPATH=/opt/ocr-training/src python -m ocr_eval`,
not the `ocr-eval` console script. The console script exists for workstation use,
where `pip install -e .` is available. Worth stating because the two environments
invoke the same code differently, and the difference is otherwise discovered in a pod.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. No entries.
