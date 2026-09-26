# Phase 1 Data Model: OCR Evaluation Harness

**Feature**: `001-ocr-eval-harness` | **Date**: 2026-09-16

Entities as the spec names them, given concrete fields, relationships and
invariants. Machine-checkable shapes live in [contracts/](./contracts/); this file
is the reasoning behind them.

---

## Entity map

```text
CorpusManifest  1 ──── * ReferencePage
                              │
ReferencePage   1 ──── 1 Prediction        (per PredictionSet)
PredictionSet   1 ──── 1 ModelVersion

EvaluationRun   1 ──── 1 CorpusManifest (subset thereof)
                1 ──── 1 PredictionSet
                1 ──── 1 NormalisationPolicy
                1 ──── * PageResult
                1 ──── 1 RunSummary  ──── * Breakdown
                                     ──── * SplitSummary

Comparison      1 ──── 2 EvaluationRun
                1 ──── * PageDelta
```

---

## ReferencePage

A page image whose correct text is known. One row of the corpus manifest.

| Field | Type | Required | Notes |
|---|---|---|---|
| `page_id` | string | yes | Unique across the manifest. Stable across corpus rebuilds — it is the join key for every artifact and every comparison. |
| `split` | string | yes | e.g. `synthetic_heldout`, `gold_scans`, `probe`. Headline metrics are reported per split and never merged (FR-013). |
| `kind` | enum | yes | `synthetic` \| `scan` \| `probe`. Drives scoring path, not presentation. |
| `reference_text` | string | one of | Inline ground truth. |
| `reference_path` | string | one of | Path relative to corpus root. Exactly one of the two must be present. |
| `image_path` | string | no | Recorded for provenance and for the worst-pages report; never read during scoring. |
| `fonts` | string[] | no | Fonts used to render the page. Empty for real scans. Drives the font breakdown (FR-006). |
| `distortions` | string[] | no | Degradations applied. Empty for originals. |
| `source` | string | no | Source document the page came from. |
| `is_augmented` | bool | no | Default `false`. Lets originals be evaluated apart from their variants. |

**Invariants**

- `page_id` is unique. A duplicate is a hard failure, not a last-wins overwrite.
- Exactly one of `reference_text` / `reference_path` is present.
- `kind == "probe"` implies the reference is exactly `[UNREADABLE]`. A probe page
  carrying real text is a corpus defect and stops the run.
- `kind != "probe"` implies the reference is non-empty and not whitespace-only.
  This is what makes the "both sides empty" edge case unreachable (research R-009).
- `kind != "probe"` further implies the reference is non-empty **after normalisation
  at every level**. A reference of nothing but punctuation empties at skeleton, one of
  nothing but diacritics empties at diacritic-insensitive; either is a corpus defect
  and stops the run, naming the level that emptied it. This is what makes a zero CER
  or WER denominator unreachable rather than merely unlikely (research R-009).
- `fonts` and `distortions` are sorted at read time, so a breakdown key does not
  depend on the order someone wrote the JSON.

---

## Prediction

What one model version produced for one reference page.

| Field | Type | Required | Notes |
|---|---|---|---|
| `page_id` | string | yes | Joins to exactly one `ReferencePage`. |
| `text` | string | yes | May be empty — an empty prediction on a legible page is a real model behaviour and is scored as total loss, not skipped. |

Supplied either as a directory of `<page_id>.txt` files or as a JSONL file of
`{page_id, text}` rows. Both forms produce the same in-memory object; the form used
is recorded in the run manifest.

**Invariants**

- Every manifest page in scope has exactly one prediction, and every prediction
  joins to a page in scope. Either gap is a hard failure listing the offending ids
  (FR-001 scenario 3).
- Files must decode as UTF-8. A decode error names the file and stops the run.
- Exactly one trailing newline is stripped at read time, symmetrically on both
  reference and prediction (research R-002). Nothing else is touched.

---

## NormalisationPolicy

Not data the user supplies — a versioned property of the harness, recorded in every
artifact so that no two numbers are ever compared under different rules (FR-004).

| Field | Type | Notes |
|---|---|---|
| `policy_version` | string | `"1.0"`. Bumped by any change to the tables. |
| `levels` | string[] | Always `["strict", "diacritic_insensitive", "skeleton"]`. |
| `headline_level` | string | Always `"strict"` (FR-003a). |
| `trailing_newline_stripped` | bool | Always `true`; recorded because it is the one I/O rule that touches text. |
| `unreadable_marker` | string | `"[UNREADABLE]"`. Recorded so a mismatch with the training targets is visible (FR-010a). |

**Invariant**: `compare` refuses to run across differing `policy_version` values.

---

## EvaluationRun

One scoring of one prediction set over one reference set. Materialised as a
directory, written once, never mutated.

```text
runs/<run_id>/
├── manifest.json        # everything needed to reproduce the run
├── pages.jsonl          # one PageResult per line, sorted by page_id
├── summary.json         # RunSummary
├── summary.md           # human-readable, rendered from summary.json
├── worst_pages.md       # side-by-side, terminal-readable
├── worst_pages.html     # side-by-side, RTL-correct
└── run.meta.json        # wall-clock, host, elapsed — OUTSIDE byte-identity
```

`manifest.json` fields:

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | User-supplied or derived from `model_version` + input digests. Never from a timestamp. |
| `model_version` | string | Checkpoint identifier. Free text, but the thing SC-008 depends on. |
| `corpus_manifest_digest` | string | SHA-256 of the manifest file as read. |
| `predictions_digest` | string | SHA-256 over sorted `page_id` + text. Independent of file layout. |
| `pages_scored` | int | After subset filtering. |
| `subset_filter` | object\|null | The filter expression, if the run was a subset (FR-018). |
| `policy` | NormalisationPolicy | Embedded whole, not referenced. |
| `config` | object | Every threshold in force, echoed in full. |
| `tool_version` | string | Harness version. |

**Invariants**

- Every file except `run.meta.json` is byte-identical across two runs on identical
  inputs (SC-003).
- A run directory that already exists is never overwritten; `score` exits non-zero
  unless `--force` is given.
- `manifest.json` is self-contained — reading it alone tells you which model, which
  data and which rules produced the run (SC-008).
- The run directory is created `0775` and its files `0664`, set explicitly rather
  than inherited from the umask, so a run written inside a pod stays readable to the
  next arbitrary non-root UID (research R-017, constitution principle V).

---

## PageResult

The outcome for a single page. One line of `pages.jsonl`.

| Field | Type | Notes |
|---|---|---|
| `page_id` | string | |
| `split`, `kind` | string | Copied from the manifest so per-page rows filter without a join. |
| `fonts`, `distortions`, `source`, `is_augmented` | — | Same reason. |
| `outcome` | enum | `scored` \| `correct_refusal` \| `hallucination` \| `false_refusal` (research R-008). |
| `metrics.<level>` | object | One per normalisation level. Fields below. |
| `order` | object | `order_corrected_cer` (strict level), `reading_order_accuracy` (nullable), `matched_lines`, `ref_lines`, `pred_lines`. |
| `flags` | string[] | Sorted. `runaway_length`, `repetition`, `empty_prediction`. |
| `length_ratio` | float | `len(pred) / max(len(ref), 1)`, strict level. |

Each `metrics.<level>` object:

| Field | Type | Notes |
|---|---|---|
| `cer` | float | `char_edits / ref_chars`. |
| `wer` | float | `word_edits / ref_words`. |
| `exact_match` | bool | |
| `char_edits`, `ref_chars` | int | Kept so micro-averaging is a sum, never an average of averages. |
| `word_edits`, `ref_words` | int | Same. |

**Invariants**

- Rows sorted by `page_id`, codepoint order (research R-011).
- `outcome` in `{correct_refusal, hallucination}` implies `metrics` is absent — a
  probe page has no reference text to measure distance against.
- `outcome == "false_refusal"` implies `metrics` is present. The page is a wrong
  transcription of a legible page and belongs in CER (FR-001) *and* in the separate
  false-refusal count (FR-010).
- `reading_order_accuracy` is `null`, never `1.0`, when fewer than two lines matched.
- Floats serialise at 6 decimal places through one shared helper.

---

## RunSummary

Aggregates, per split. Read at a glance (FR-005).

| Field | Type | Notes |
|---|---|---|
| `run_id`, `model_version`, `policy`, `tool_version` | — | Echoed so the summary stands alone. |
| `splits` | SplitSummary[] | Sorted by split name. |
| `notes` | string[] | e.g. the verbatim gold-set warning when no `gold_scans` split is present (research R-013). |

There is deliberately **no** cross-split total key (FR-013). Its absence, rather
than a null, is the guarantee.

`SplitSummary`:

| Field | Type | Notes |
|---|---|---|
| `split`, `pages` | string, int | |
| `scored_pages` | int | Pages carrying metrics — outcome `scored` or `false_refusal`. The denominator behind every rate on this split, published so it is visible. |
| `headline` | object\|null | Strict micro CER, WER, exact-match rate. The figure quoted when one number is quoted (FR-003a). `null` only when `scored_pages == 0`. |
| `metrics.<level>.{micro,macro}` | object\|null | All three levels, micro and macro (research R-005). `null` under the same condition, never independently of `headline`. |
| `order` | object | `order_corrected_cer_micro` (nullable), `reading_order_accuracy` (micro over matched pairs, nullable), `pages_without_order_signal`. |
| `contract` | object | `correct_refusal`, `hallucination`, `false_refusal`, `hallucination_rate`, `false_refusal_rate`. Always reported in full. |
| `flags` | object | Count per flag name. Always reported in full. |
| `breakdowns` | Breakdown[] | By font, distortion, source. Empty when `scored_pages == 0`. |

**Invariant**: a split with no scoreable page — the probe split is the case that
occurs — reports `null`, never `0`, for its aggregates. A zero would read as a
perfect score on the split that contains nothing to score, which is the defaulted
number constitution principle IV forbids. `contract` and `flags` are still reported
in full, because those are the numbers such a split genuinely has (research R-018).

---

## Breakdown

One dimension of a split, ranked (FR-006, FR-007).

| Field | Type | Notes |
|---|---|---|
| `dimension` | enum | `font` \| `distortion` \| `source`. |
| `rows` | object[] | `{key, pages, cer_micro, cer_macro, wer_micro, exact_match_rate}`. |

**Invariants**

- Rows sorted by `cer_micro` descending, ties broken by `key` — worst first, so the
  worst-performing font is the first row and needs no further analysis (FR-007, SC-005).
- A page with multiple fonts contributes to every font's row. The `pages` counts
  therefore sum to more than the split total; the schema labels this explicitly so
  nobody reads it as a partition.
- Keys with fewer pages than `min_pages_per_breakdown_row` (default 5) are folded
  into a single `__sparse__` row rather than topping the ranking on one bad page.

---

## Comparison

The difference between two runs (FR-008).

| Field | Type | Notes |
|---|---|---|
| `baseline_run_id`, `candidate_run_id` | string | |
| `policy_version` | string | Identical in both, or the comparison does not exist. |
| `verdict` | enum | `PROMOTE` \| `REJECT` \| `INCONCLUSIVE` (research R-014). |
| `verdict_reasons` | string[] | Every rule that fired, in rule order. |
| `thresholds` | object | The rule's thresholds as applied. |
| `splits` | object[] | Per split: each headline metric, both values, delta, direction. |
| `bootstrap` | object | Per split: `delta_cer`, `ci_low`, `ci_high`, `resamples`, `seed`. |
| `regressions` | PageDelta[] | Pages that got worse, sorted by `delta_cer` descending. |
| `improvements` | PageDelta[] | Same, ascending. |
| `only_in_baseline`, `only_in_candidate` | string[] | Page ids. Non-empty means the runs are not over the same set and the comparison fails. |

`PageDelta`: `{page_id, split, baseline_cer, candidate_cer, delta_cer, outcome_changed}`.

**Invariants**

- Both runs share `policy_version`, `corpus_manifest_digest`, split set **and page
  set**, or `compare` exits 3 naming the mismatch (research R-004). The page-set
  condition is the one FR-018 makes reachable: two subset runs over the same corpus
  pass every other guard.
- `only_in_baseline` and `only_in_candidate` are therefore always empty in a written
  artifact. They exist so a consumer can assert emptiness; the ids themselves go to
  stderr with the exit 3.
- `verdict_reasons` is never empty — `INCONCLUSIVE` states why it is inconclusive.
- Direction is stated as a word (`better` / `worse` / `unchanged`), never left for
  the reader to infer from the sign of a delta (FR-008, SC-006).

---

## Gold set and probe set

Not separate schemas — both are `split` values over `ReferencePage`, which is what
keeps them from ever being merged into a headline by accident.

- **Gold set**: `split: "gold_scans"`, `kind: "scan"`. Manually verified real scans,
  never used for training. **It does not exist yet.** Until it does, every summary
  carries the verbatim note that measurement is synthetic-only and overstates
  real-world quality.
- **Probe set**: `split: "probe"`, `kind: "probe"`, reference exactly
  `[UNREADABLE]`. Excluded from CER/WER/exact-match aggregates; scored only under
  the contract matrix.
