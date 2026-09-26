---

description: "Task list for the OCR Evaluation Harness"
---

# Tasks: OCR Evaluation Harness

**Input**: Design documents from `specs/001-ocr-eval-harness/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md,
`.specify/memory/constitution.md` v1.0.0

**Tests**: **Included and mandatory.** Not a stylistic choice — the constitution's
Development Workflow section requires a unit test for every metric, transformation or
validation rule that produces a reported number, and requires any determinism guarantee
to be asserted by running twice and comparing. SC-003 states that guarantee. Test tasks
below are therefore first-class, not optional.

**Organization**: Grouped by user story. US1 is the MVP and, as the spec's own priority
rationale says, the only part that needs no cluster access.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete work)
- **[Story]**: US1–US4, mapping to the user stories in spec.md
- Exact file paths are given in every task

## Path Conventions

Single project, src layout, per plan.md: `src/ocr_eval/`, `tests/`, `configs/eval/` at
the repository root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Make the package importable, installable and testable. Nothing here scores
anything.

- [X] T001 Create `pyproject.toml` at repo root: setuptools src layout, `requires-python = ">=3.10"`, console script `ocr-eval = ocr_eval.cli:main`, **zero required runtime dependencies**, optional extras `fast = ["rapidfuzz"]` and `test = ["pytest"]` (constitution III: every added wheel is a build risk)
- [X] T002 [P] Create package skeleton `src/ocr_eval/{__init__.py,__main__.py}` plus empty packages `io/`, `normalise/`, `metrics/`, `score/`, `aggregate/`, `compare/`, `report/`, each with `__init__.py`; `__main__.py` delegates to `cli:main` so `python -m ocr_eval` and the console script share one entry point
- [X] T003 [P] Create `configs/eval/default.json` holding every threshold verbatim from research: `runaway_length_ratio: 2.0`, `runaway_min_char_excess: 100`, `repetition_min_line_len: 10`, `repetition_min_occurrences: 5`, `repetition_min_char_share: 0.30`, `line_match_min_similarity: 0.5`, `min_pages_per_breakdown_row: 5`, `promote_min_cer_improvement_pp: 0.5`, `reject_on_any_gold_regression: true`, `reject_hallucination_worsening_pp: 1.0`, `reject_false_refusal_worsening_pp: 1.0`, `bootstrap_resamples: 10000`, `bootstrap_seed: 20260916`
- [X] T004 [P] *(no-op — verified 2026-09-16: no artifact contains `default.yaml`; plan.md and contracts/cli.md already specify `.json`. Kept for the record rather than silently dropped.)* Correct the stale config-format references: `specs/001-ocr-eval-harness/plan.md` and `specs/001-ocr-eval-harness/quickstart.md` both say `configs/eval/default.yaml`; change to `.json`. YAML would require PyYAML at runtime, which contradicts the zero-dependency gate the same plan asserts (constitution III)
- [X] T005 [P] Create `configs/eval/README.md` explaining that every key is echoed verbatim into each run's `manifest.json`, so changing a threshold changes reported results and must be deliberate (FR-014, SC-008)
- [X] T006 [P] Add `[tool.pytest.ini_options]` to `pyproject.toml` (`testpaths = ["tests"]`) and create the `tests/{unit,integration,golden,fixtures}/` tree with `__init__.py` where needed; add `runs/`, `*.egg-info/`, `.pytest_cache/` to `.gitignore` (constitution V: generated artifacts are not committed)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The primitives every command depends on — serialisation, normalisation,
distance, input readers, and the fail-loud validator.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Core primitives

- [X] T007 Implement `src/ocr_eval/errors.py`: typed exceptions and the single exit-code mapping from contracts/cli.md — `0` success, `1` input integrity, `2` usage, `3` incommensurable comparison, `4` output collision; every non-zero exit writes its explanation to stderr and nothing to stdout
- [X] T008 [P] Implement `src/ocr_eval/io/writer.py` as the only serialisation point: JSON with `sort_keys=True`, `ensure_ascii=False`, `indent=2`, LF endings, UTF-8 without BOM, one trailing newline; all floats through a single 6-decimal-place helper; directories created `0775` and files `0664` set explicitly, never inherited from the umask (research R-011, R-017; constitution II and V)
- [X] T009 [P] Implement `src/ocr_eval/normalise/tables.py`: `POLICY_VERSION = "1.0"`, the diacritic set (`U+064B`–`U+065F`, `U+0670`, `U+06D6`–`U+06DC`, `U+06DF`–`U+06E4`, `U+06E7`, `U+06E8`, `U+06EA`–`U+06ED`, `U+08D3`–`U+08E1`, `U+08E3`–`U+08FF`, **excluding `U+08E2` which is a format character**), the letter-folding table, and the digit-folding ranges `U+0660`–`U+0669` and `U+06F0`–`U+06F9`, exactly as contracts/normalisation-policy.md states
- [X] T010 Implement `src/ocr_eval/normalise/levels.py`: `strict` applies **zero** transformations; `diacritic_insensitive` removes the diacritic set then any remaining `Mn` mark on an Arabic base; `skeleton` runs NFKC → strip all `Mn` → drop `U+0640` → letter fold → digit fold → drop categories `P*`/`S*` → collapse whitespace runs to one `U+0020` and strip. Step order is part of the contract and must not be reordered (depends on T009)
- [X] T011 [P] Implement `src/ocr_eval/metrics/distance.py`: `rapidfuzz.distance.Levenshtein` when importable, else a bundled pure-Python Myers bit-parallel fallback; both paths return identical integers, and the selected path is reported for the run manifest but never changes a number (research R-001)
- [X] T012 [P] Implement `src/ocr_eval/metrics/rates.py`: per-page CER `char_edits / ref_chars` and WER `word_edits / ref_words`, exact match, and corpus aggregation as **micro** (`sum(edits)/sum(denominator)`) and **macro** (mean of per-page rates); tokenisation is runs of non-whitespace after that level's normalisation, empty tokens discarded (research R-005, R-006)

### Input readers and validation

- [X] T013 [P] Implement `src/ocr_eval/io/manifest.py` against contracts/corpus-manifest.schema.json: required `page_id`, `split`, `kind`; `kind` is one of `synthetic | scan | probe`; exactly one of `reference_text` / `reference_path`; `fonts` and `distortions` sorted at read time so breakdown keys never depend on authoring order; `is_augmented` defaults `false`
- [X] T014 [P] Implement `src/ocr_eval/io/predictions.py`: accept a directory of `<page_id>.txt` or a JSONL of `{page_id, text}`, auto-detected by whether the path is a directory; decode UTF-8 strictly; strip exactly one trailing newline, symmetrically with references (research R-002); record which form was used
- [X] T015 Implement `src/ocr_eval/config.py`: load `configs/eval/default.json`, allow `--config` override, expose thresholds as a frozen object, and echo the full resolved config for embedding in `manifest.json` (depends on T003)
- [X] T016 Implement `src/ocr_eval/score/validate.py` reporting **every** problem found rather than the first: manifest page with no prediction; prediction with no manifest page; non-UTF-8 file; empty or whitespace-only reference where `kind != "probe"`; **a `kind != "probe"` reference that is non-empty raw but normalises to empty at any of the three levels, naming the level that emptied it** — punctuation-only empties at skeleton, diacritics-only at diacritic-insensitive, and either would divide CER by zero; `kind == "probe"` whose reference is not exactly `[UNREADABLE]`; missing required field or unknown `kind`; duplicate `page_id`; `--pages` id absent from the manifest (research R-009, FR-016, constitution IV) (depends on T010, T013, T014)
- [X] T017 Implement the subset selector in `src/ocr_eval/score/validate.py` or a sibling module: `--pages <file>` id list, and `--filter` over fields `split`, `kind`, `source`, `font`, `distortion`, `is_augmented` with operators `=`, `!=`, `~`, comma-ANDed, no OR; `font` and `distortion` are list fields where `=` means "contains"; the parsed filter is retained as structured data for the run manifest (FR-018)

### CLI shell

- [X] T018 Implement `src/ocr_eval/cli.py`: argparse with subcommands `validate | score | compare | report`, `--version` printing `ocr-eval <tool_version> (policy <policy_version>)`, machine output to stdout and all progress/errors to stderr, no colour when stdout is not a TTY, and no network call anywhere (FR-015) (depends on T007)
- [X] T019 Wire the `validate` subcommand end to end in `src/ocr_eval/cli.py` with the flags in contracts/cli.md (`--manifest`, `--predictions`, `--corpus-root`, `--pages`, `--filter`, `--json`) (depends on T016, T017, T018)

### Fixtures and foundational tests

- [X] T020 Build `tests/fixtures/mini_corpus/` — about 20 pages with `manifest.jsonl`, `refs/`, and two prediction sets `predictions/baseline/` and `predictions/finetuned/`; must include clean pages, `syn_shuffled_*` pages whose prediction is the correct text with lines reordered, a runaway-repetition page, a diacritics-differing page, a presentation-form page, and probe pages. Small, fixed, hand-built test input — the case constitution V explicitly permits committing
- [X] T021 [P] Unit tests `tests/unit/test_normalise.py`: assert each level per codepoint class — strict changes nothing; `diacritic_insensitive` drops vowelling but keeps tatweel, digit forms and punctuation; skeleton folds presentation forms, alef and yeh variants, teh marbuta and digits, and collapses whitespace. Include the NFD case `U+0627 U+0653` vs `U+0622`
- [X] T022 [P] Unit tests `tests/unit/test_distance.py`: the rapidfuzz path and the pure-Python fallback return identical integers across the shared vector set, including empty strings, pure-insertion and pure-deletion cases
- [X] T023 [P] Unit tests `tests/unit/test_rates.py`: micro and macro differ as expected on a mixed short/long page set; and assert the zero-denominator guard is real rather than assumed — a punctuation-only reference and a diacritics-only reference each reach `validate` and are rejected there (exit 1, naming the level that emptied them), so `rates.py` is never handed `ref_chars == 0` or `ref_words == 0`. Test the guard, not the hope that one exists
- [X] T024 [P] Integration tests `tests/integration/test_validate_failures.py`: one test per hard failure in T016, each asserting exit code 1 **and** that all seeded problems are named in one run, not just the first

**Checkpoint**: `ocr-eval validate` works. The corpus can be checked in seconds, and every
primitive that produces a number is under test.

---

## Phase 3: User Story 1 — Score a checkpoint against a reference set (Priority: P1) 🎯 MVP

**Goal**: Turn a corpus manifest plus a prediction set into an immutable run directory
carrying CER, WER and exact-match overall and per page, at all three normalisation levels.

**Independent Test**: Run it against the untrained base model's predictions on a few
hundred pages. It produces a baseline error rate — the line every future run must beat.

**Scope note**: US1 emits the full artifact shape, including the `order` and `contract`
objects, because contracts/page-result.schema.json and run-summary.schema.json require
those keys. US4 later deepens probe-set semantics and adds the generation flags; it does
not change this shape.

### Tests for User Story 1

- [X] T025 [P] [US1] Unit tests `tests/unit/test_order.py`: greedy matching is deterministic under the sort key `(-similarity, ref_index, pred_index)`; a shuffled-line page yields high strict CER, near-zero order-corrected CER and low order accuracy; a page with fewer than two matched lines reports `reading_order_accuracy: null`, **never `1.0`**
- [X] T026 [P] [US1] Unit tests `tests/unit/test_contract.py`: the four-cell matrix — marker/marker is `correct_refusal`, marker-reference/other-text is `hallucination`, legible-reference/marker is `false_refusal`, otherwise `scored`; the marker matches only when the stripped prediction equals exactly `[UNREADABLE]`, so `"[UNREADABLE] — possibly a stamp?"` is a hallucination
- [X] T027 [P] [US1] Integration test `tests/integration/test_score_run.py`: scoring the fixture corpus produces every file in the run directory, and each artifact validates against its schema in `specs/001-ocr-eval-harness/contracts/`; assert explicitly that the probe split's `summary.json` entry carries `scored_pages: 0` with `headline` and `metrics` **`null` rather than `0`**, a full `contract` block, and `breakdowns: []` (research R-018)
- [X] T028 [P] [US1] Integration test `tests/integration/test_subset.py`: `--filter 'font=Amiri,is_augmented=false'` and `--pages` each restrict the scored set, and `manifest.json` records the parsed filter as structured data (FR-018)
- [X] T029 [P] [US1] Golden test `tests/golden/test_byte_identity.py`: score the fixture corpus twice into different directories; every file except `run.meta.json` is byte-identical (SC-003, constitution II)

### Implementation for User Story 1

- [X] T030 [P] [US1] Implement `src/ocr_eval/score/contract.py`: the `[UNREADABLE]` classification returning `scored | correct_refusal | hallucination | false_refusal`; probe outcomes carry no metrics, `false_refusal` carries metrics **and** is counted separately (research R-008, FR-010)
- [X] T031 [US1] Implement `src/ocr_eval/metrics/order.py`: split both sides into non-empty lines keeping original indices; similarity matrix over skeleton-normalised lines; greedy matching at `>= line_match_min_similarity` with the deterministic sort key; order-corrected CER by re-emitting matched lines in reference order followed by unmatched prediction lines in original order; reading-order accuracy as the fraction of matched line pairs whose relative order agrees, `null` below two matches (research R-007) (depends on T010, T011)
- [X] T032 [US1] Implement `src/ocr_eval/score/page.py`: score one page at all three levels, keeping `char_edits`/`ref_chars`/`word_edits`/`ref_words` alongside the rates so aggregation is a sum, and attach the `order` block and the contract outcome (depends on T010, T012, T030, T031)
- [X] T033 [US1] Implement `src/ocr_eval/aggregate/summary.py`: per-split `scored_pages` (outcome `scored` or `false_refusal`), headline (strict micro CER, WER, exact-match), all three levels micro and macro, the `order` aggregate, and the `contract` counts. **No cross-split total key may exist** — its absence is the FR-013 guarantee. When `scored_pages == 0` — the probe split — emit `headline: null`, `metrics: null`, `order.order_corrected_cer_micro: null` and `breakdowns: []`, never `0`: a zero CER reads as a perfect score on the split with nothing to score, which is the defaulted number constitution IV forbids; `contract` and `flags` are still reported in full (research R-018). Emit the verbatim synthetic-only caveat into `notes` when no `gold_scans` split is present (research R-013)
- [X] T034 [US1] Implement `src/ocr_eval/score/run.py`: run `validate` first and abort before scoring anything if it fails; score every page in scope; write `manifest.json`, `pages.jsonl` sorted by `page_id` in codepoint order, `summary.json`, and `run.meta.json` holding wall-clock, host and elapsed — the only file outside the byte-identity guarantee; refuse an existing run directory with exit 4 unless `--force` (depends on T008, T016, T032, T033)
- [X] T035 [US1] Populate `manifest.json` in `src/ocr_eval/score/run.py` with `run_id` (derived from `model_version` plus input digests, **never from a timestamp**), `model_version`, `corpus_manifest_digest`, `predictions_digest` (SHA-256 over sorted `page_id` + text, independent of file layout), `pages_scored`, `subset_filter`, the embedded policy object including `unreadable_marker`, the full config, and `tool_version` (SC-008, FR-010a)
- [X] T036 [US1] Implement `src/ocr_eval/report/markdown.py`: render `summary.md` from `summary.json` showing all three levels together with strict as the headline, never absent and never subordinate (FR-003a, SC-004); a split whose `headline` is `null` renders as an explicit "no scoreable pages (n probe pages)" line rather than a blank cell or a dash, so the reason is on the page and not inferred (research R-018)
- [X] T037 [US1] Implement `src/ocr_eval/aggregate/breakdown.py` sufficiently for the run to emit the required `breakdowns` key, ranked by `cer_micro` descending with ties broken by `key` ascending; US3 adds the reporting surface on top (depends on T033)
- [X] T038 [US1] Wire the `score` subcommand in `src/ocr_eval/cli.py` with every flag from contracts/cli.md (`--manifest`, `--predictions`, `--model-version`, `--out`, `--corpus-root`, `--run-id`, `--config`, `--pages`, `--filter`, `--worst`, `--jobs`, `--force`, `--no-html`); `--jobs` may reorder execution but must not reorder output (depends on T034, T018)

**Checkpoint**: A checkpoint can be scored and the result read. The baseline number exists.

---

## Phase 4: User Story 2 — Compare against baseline and previous best (Priority: P2)

**Goal**: Turn two run directories into a promote-or-reject decision a reviewer can act on
without opening the underlying data.

**Independent Test**: Score two checkpoints separately, then compare the two reports. It
says which is better, by how much, and on which pages the new one got worse.

**Depends on US1**: `compare` consumes run directories, so US1 must be able to produce them.

### Tests for User Story 2

- [X] T039 [P] [US2] Integration test `tests/integration/test_compare.py`: comparing two fixture runs lists regressions worst-first and states a direction word for each metric
- [X] T040 [P] [US2] Integration test in the same file: comparing runs with differing `policy_version`, differing `corpus_manifest_digest`, a different split set, or **a different page set** exits **3** and names the mismatch — there is no warn-and-continue path (research R-004). The page-set case is the one FR-018 makes reachable: score the same corpus twice under different `--pages` subsets, so `policy_version`, `corpus_manifest_digest` and split set all match and only the page set differs; assert exit 3 and that the unmatched ids appear on stderr
- [X] T041 [P] [US2] Unit test `tests/unit/test_bootstrap.py`: the paired bootstrap is reproducible under the default `seed = 20260916` and `resamples = 10000`, and an interval straddling zero is reported as such; also run it with a **non-default** `bootstrap_seed` and `bootstrap_resamples` from config and assert the comparison artifact still validates against contracts/comparison.schema.json — the schema records these two values, it does not pin them

### Implementation for User Story 2

- [X] T042 [P] [US2] Implement `src/ocr_eval/compare/diff.py`: join two runs on `page_id`, compute `PageDelta` records `{page_id, split, baseline_cer, candidate_cer, delta_cer, outcome_changed}`, and sort regressions by `delta_cer` descending then `page_id`, improvements ascending then `page_id`
- [X] T043 [P] [US2] Implement `src/ocr_eval/compare/bootstrap.py`: paired bootstrap over per-page CER deltas using `random.Random(seed)` only, emitting `delta_cer`, `ci_low`, `ci_high` at a fixed 95%, with `resamples` and `seed` **read from config** (`bootstrap_resamples`, `bootstrap_seed`) and echoed into the artifact — the schema records whatever was in force rather than pinning the defaults, so a tuned config cannot produce output that fails its own contract
- [X] T044 [US2] Implement `src/ocr_eval/compare/verdict.py`: **REJECT** if gold-set strict CER worsens at all, or hallucination rate worsens by more than `reject_hallucination_worsening_pp`, or false-refusal rate worsens by more than `reject_false_refusal_worsening_pp`; **PROMOTE** if no reject condition holds and strict micro CER on the primary split improves by at least `promote_min_cer_improvement_pp`; otherwise **INCONCLUSIVE**. `verdict_reasons` is never empty — an INCONCLUSIVE verdict states why (research R-014)
- [X] T045 [US2] Implement the comparison renderer in `src/ocr_eval/report/markdown.py`: per split, each headline metric on both sides with delta and a **direction stated as a word** (`better`/`worse`/`unchanged`), because the sign meaning "better" differs per metric; emit a `metricDelta` only for metrics non-null on both sides, so a split with `scored_pages == 0` contributes its contract rates and no CER delta computed from a null (research R-018) (depends on T042, T043, T044)
- [X] T046 [US2] Wire the `compare` subcommand in `src/ocr_eval/cli.py` (`--baseline`, `--candidate`, `--out`, `--config`, `--primary-split`, `--top`, `--format`), enforcing the commensurability guard before any computation and **exiting 0 for all three verdicts** — a REJECT is a successful comparison, so scripts read the verdict from the artifact, not from `$?` (depends on T045, T018)

**Checkpoint**: Promote-or-reject decisions are mechanical and defensible.

---

## Phase 5: User Story 3 — Break errors down by cause (Priority: P3)

**Goal**: Say *why* an error rate is what it is — which font, which distortion, which
pages — so effort goes to the biggest problem rather than the most visible one.

**Independent Test**: Run a breakdown on any scored set and confirm the worst-performing
font can be named from the report alone.

### Tests for User Story 3

- [X] T047 [P] [US3] Unit test `tests/unit/test_breakdown.py`: rows rank worst-first by `cer_micro` with ties broken by `key`; a page carrying two fonts contributes to both rows so counts exceed the split total, and the artifact sets `is_partition: false` to say so; categories below `min_pages_per_breakdown_row` (default 5) fold into a single `__sparse__` row instead of topping the ranking on one bad page
- [X] T048 [P] [US3] Integration test `tests/integration/test_report.py`: `report` reads only `pages.jsonl` and `summary.json`, never the corpus, and re-rendering with a different `--worst` cannot disagree with the run it came from

### Implementation for User Story 3

- [X] T049 [US3] Complete `src/ocr_eval/aggregate/breakdown.py` for all three dimensions — `font`, `distortion`, `source` — with `__sparse__` folding and the `is_partition` flag (FR-006, FR-007) (depends on T037)
- [X] T050 [P] [US3] Implement `src/ocr_eval/report/worst_pages.py`: the worst-scoring pages with page id, split, fonts, distortions, strict CER, order-corrected CER and flags, then reference and prediction side by side, then a **codepoint difference list** rendering the first 40 differences as `U+XXXX NAME` (SC-009)
- [X] T051 [P] [US3] Implement `src/ocr_eval/report/html.py`: the same side-by-side with `dir="rtl" lang="ar"` on text cells, because rendered Arabic alone cannot distinguish composed from decomposed forms or presentation forms from base letters (research R-016)
- [X] T052 [US3] Extend `src/ocr_eval/report/markdown.py` to render the ranked breakdown tables into `summary.md` and as standalone output (depends on T049)
- [X] T053 [US3] Wire the `report` subcommand in `src/ocr_eval/cli.py` (`--run`, `--worst`, `--breakdown font|distortion|source|all`, `--split`, `--format md|html|json`, `--out`) (depends on T050, T051, T052, T018)
- [X] T054 [US3] Emit `worst_pages.md` and `worst_pages.html` from `score` by default, honouring `--worst` and `--no-html` (depends on T050, T051, T034)

**Checkpoint**: The worst font and worst distortion are each the first row of their table.

---

## Phase 6: User Story 4 — Detect hallucination on unreadable input (Priority: P3)

**Goal**: Measure how often the model invents text where none exists — the failure mode
specific to using a language model for OCR, and the one no aggregate error rate reveals.

**Independent Test**: Run the probe set through any checkpoint and count how often it
produced text where none exists.

### Tests for User Story 4

- [X] T055 [P] [US4] Unit test `tests/unit/test_flags.py`: `runaway_length` fires only when `len(pred) >= 2.0 * len(ref)` **and** `len(pred) - len(ref) >= 100`, so short pages do not trip it constantly; `repetition` fires when a skeleton-normalised line of length `>= 10` occurs `>= 5` times **and** those repeats are `>= 30%` of prediction characters
- [X] T056 [P] [US4] Integration test `tests/integration/test_probe_set.py`: probe pages are excluded from CER/WER/exact-match aggregates entirely, while false refusals on legible pages appear in CER **and** in their own separate count

### Implementation for User Story 4

- [X] T057 [P] [US4] Implement `src/ocr_eval/metrics/flags.py` with both flags above plus `empty_prediction`; flagged pages stay in the aggregate — the flag tells a reader where an error rate came from, it does not exclude the page (FR-011, research R-010)
- [X] T058 [US4] Attach flags and `length_ratio` (`len(pred) / max(len(ref), 1)` at strict level) to every `PageResult` in `src/ocr_eval/score/page.py`, with `flags` sorted (depends on T057, T032)
- [X] T059 [US4] Add the contract aggregate to `src/ocr_eval/aggregate/summary.py`: `hallucination_rate = hallucination / probe_pages` and `false_refusal_rate = false_refusal / legible_pages`, each `null` rather than `0` when its denominator is empty, and the two never summed (FR-010, SC-007) (depends on T033)
- [X] T060 [US4] Extend the probe fixtures in `tests/fixtures/mini_corpus/` to cover all four contract cells plus a runaway page and a repetition page, so every branch has a fixture (depends on T020)
- [X] T061 [US4] Report the contract figures and the flag counts in `summary.md` via `src/ocr_eval/report/markdown.py`, with hallucination and false refusal visibly separate (depends on T059, T036)
- [X] T062 [US4] Record the FR-010a cross-feature dependency in `README.md`: the dataset preparation work must emit the exact `[UNREADABLE]` marker as the training target, or this measurement is meaningless; the marker is written into every run manifest so a mismatch is at least visible (constitution VI)

**Checkpoint**: Every checkpoint reports a hallucination rate and a separate false-refusal rate.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T063 [P] Create `scripts/make_synthetic_eval_set.py` generating an N-page corpus with predictions, used by the performance scenario in quickstart.md and by nothing else
- [X] T064 Verify SC-002 by scoring 1,000 generated pages in under 5 minutes single-process on an ordinary workstation, and record the measured time in `configs/eval/README.md`
- [X] T065 [P] Add an "Evaluating a checkpoint" section to `README.md` covering both invocation paths: `ocr-eval` on a workstation after `pip install -e .`, and `PYTHONPATH=/opt/ocr-training/src python -m ocr_eval` inside a pod, where the package ships via the Containerfile's `COPY src/` but is never pip-installed
- [X] T066 [P] Add `pytest` to the test extra and confirm the whole suite runs offline with no external data, as the constitution's workflow section requires
- [X] T067 Walk every scenario in `specs/001-ocr-eval-harness/quickstart.md` end to end and correct any drift between the guide and the built CLI
- [X] T068 [P] Verify no module imports a network client anywhere under `src/ocr_eval/` (FR-015, constitution III) and add a test asserting it in `tests/unit/test_no_network.py`
- [ ] T069 **BLOCKED — not verifiable on this machine (Windows).** `tests/unit/test_writer.py::test_permissions_are_set_explicitly_not_inherited_from_the_umask` implements the check and skips off POSIX; it must be run once on a Linux host to close this. Confirm on a Linux host that a run directory written under `umask 077` is still group-readable and group-writable, proving T008's explicit `chmod` rather than umask inheritance (research R-017, constitution V)
- [ ] T070 Run `/speckit-analyze` across spec.md, plan.md and tasks.md and resolve anything it flags before implementation is called done

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: depends on Setup — **blocks every user story**
- **US1 (Phase 3)**: depends on Foundational
- **US2, US3, US4 (Phases 4–6)**: depend on Foundational **and on US1**
- **Polish (Phase 7)**: depends on the stories you intend to ship

### User Story Dependencies — stated honestly

The template's usual "stories are independent" does not hold cleanly here, and pretending
otherwise would mislead whoever schedules this:

- **US1 (P1)** is genuinely independent once Foundational is done.
- **US2, US3 and US4 all consume run directories that only US1 can produce.** Each is
  independently *testable* against committed fixture runs, but none is independently
  *deliverable* before US1. This is the spec's own reasoning for making US1 P1: nothing
  else in the project can be judged without it.
- **US2, US3 and US4 are independent of each other** and can proceed in parallel once US1
  lands. US4 touches `score/page.py` and `aggregate/summary.py`, which US1 also owns, so
  sequence US4 after US1 is merged rather than alongside it.

### Parallel Opportunities

- Phase 1: T002–T006 all parallel after T001
- Phase 2: T008, T009, T011, T013, T014 are parallel; T021–T024 are parallel once their
  subjects exist
- Phase 3: T025–T029 parallel; T030 parallel with them; T031–T038 largely sequential,
  sharing `score/` and `aggregate/`
- Phases 4–6: parallel across stories once US1 is merged
- Phase 7: T063, T065, T066, T068 parallel

---

## Parallel Example: Foundational primitives

```bash
# Four independent modules, four different files, no shared state:
Task: "Implement src/ocr_eval/io/writer.py with byte-identity and 0775/0664 permissions"
Task: "Implement src/ocr_eval/normalise/tables.py with POLICY_VERSION and codepoint tables"
Task: "Implement src/ocr_eval/metrics/distance.py with rapidfuzz plus pure-Python fallback"
Task: "Implement src/ocr_eval/io/manifest.py against contracts/corpus-manifest.schema.json"
```

---

## Implementation Strategy

### MVP — US1 only (T001–T038)

1. Phase 1 Setup
2. Phase 2 Foundational — blocks everything
3. Phase 3 US1
4. **STOP and VALIDATE**: score the untrained base model's predictions and read the result

That baseline number is the deliverable. It needs no GPU, no cluster access and no
infrastructure that is currently blocked, which is exactly why the spec made it P1.

### Incremental delivery

1. Setup + Foundational → `ocr-eval validate` works, the corpus is checkable
2. + US1 → checkpoints get scored (**MVP**)
3. + US2 → promote-or-reject decisions become mechanical
4. + US3 → the worst font and worst distortion get named
5. + US4 → hallucination and false refusal get measured

### A caveat that outlives this task list

The gold set of real scanned pages does not exist yet. Every number this harness produces
until it does is measured on synthetic pages and **overstates real-world quality**. T033
makes the harness say so in every affected report, which contains the problem but does not
solve it. Building that set is separate work and is the largest available improvement to
the credibility of these numbers.

---

## Notes

- `[P]` means a different file with no dependency on incomplete work
- Every task names its file path; constraints from data-model.md and the contracts are
  quoted verbatim so they are not left to implementation-time discretion
- Tests are mandatory here by constitution, not by preference — write them before the
  implementation they cover
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
