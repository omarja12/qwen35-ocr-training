# Evaluation thresholds

`default.json` holds every tunable number the harness uses. There are no others:
a threshold that is not in this file is not a threshold, it is a constant in the
code and changing it requires a `policy_version` bump.

## Changing a value changes a result

Every key here is echoed **verbatim** into each run's `manifest.json` and into
every comparison artifact (FR-014, SC-008). That is the point: someone reading a
report six months from now can see the thresholds that produced it without
having to guess which version of this file was on disk at the time.

The corollary is that editing a value silently changes what the harness reports.
Two runs scored under different configs are not directly comparable even though
nothing in the run directory looks different at a glance — so change a value
deliberately, and re-score rather than mixing old and new runs.

`--config <path>` overrides this file wholesale for a single run.

## JSON, not YAML

PyYAML would be a runtime dependency, and the harness has none by design — every
added wheel is a build risk for the air-gapped image (constitution III). JSON
parses from the standard library.

## The keys

### Generation flags (FR-011, research R-010)

| Key | Default | Meaning |
|---|---|---|
| `runaway_length_ratio` | `2.0` | A prediction flags as `runaway_length` when it is at least this many times the reference length… |
| `runaway_min_char_excess` | `100` | …**and** exceeds it by at least this many characters. Both conditions, so a 12-character reference does not trip the flag on a 24-character prediction. |
| `repetition_min_line_len` | `10` | Lines shorter than this are ignored when looking for repetition — short repeated lines are normal in tables and headers. |
| `repetition_min_occurrences` | `5` | A line must recur at least this many times to count. |
| `repetition_min_char_share` | `0.3` | …and those repeats must be at least this share of the prediction's characters, so one repeated line in a long page is not a flag. |

A flag never removes a page from an aggregate. It tells a reader where an error
rate came from.

### Reading order (FR-012, research R-007)

| Key | Default | Meaning |
|---|---|---|
| `line_match_min_similarity` | `0.5` | Two lines pair only above this skeleton-normalised similarity. Lowering it flatters the order-corrected CER by matching unrelated lines. |

### Breakdowns (FR-006, FR-007)

| Key | Default | Meaning |
|---|---|---|
| `min_pages_per_breakdown_row` | `5` | Categories with fewer pages fold into a single `__sparse__` row instead of topping the ranking on one bad page. |

⚠️ **This threshold interacts with corpus size, and the failure is quiet.** If no
font reaches five pages, every font folds into `__sparse__`, the first row of the
table names nothing, and SC-005 — "the worst-performing font can be named from a
single report" — silently cannot be met. The table still renders; it just answers
no question. On a small corpus or a narrow subset, either lower this value or
accept that the breakdown is not usable yet. `tests/fixtures/mini_corpus/` is
sized so each font and two of the three distortions clear the floor, which is what
makes the criterion testable at all.

### The promote/reject rule (SC-006, research R-014)

| Key | Default | Meaning |
|---|---|---|
| `promote_min_cer_improvement_pp` | `0.5` | Strict micro CER on the primary split must improve by at least this many percentage points to PROMOTE. |
| `reject_on_any_gold_regression` | `true` | Any worsening of gold-set strict CER is an automatic REJECT. Encodes the project's real risk: synthetic gains that do not survive real scans. |
| `reject_hallucination_worsening_pp` | `1.0` | Hallucination rate worsening by more than this rejects, whatever CER did. |
| `reject_false_refusal_worsening_pp` | `1.0` | Same, for false refusals on legible pages. |

### The bootstrap interval (research R-014)

| Key | Default | Meaning |
|---|---|---|
| `bootstrap_resamples` | `10000` | Resamples in the paired bootstrap over per-page CER deltas. |
| `bootstrap_seed` | `20260916` | Fixed so the interval is reproducible. Recorded in the artifact; the schema records these two values rather than pinning them, so a tuned config still produces valid output. |

The confidence level is fixed at 95% and is not configurable.

## Measured performance (SC-002)

The budget is 1,000 pages scored in under 5 minutes single-process, so that
evaluation is never the reason someone skips evaluating. Measurements are
recorded here as they are taken, with the machine attached — a number without
its machine is not a number a stranger can check (constitution II).

| Date | Pages | Elapsed | Budget | Machine |
|---|---|---|---|---|
| 2026-09-16 | 1,000 | 148 s | 300 s | Intel64 Family 6 Model 61 (4 cores), Windows 10, Python 3.14.7, pure-Python distance engine |
| 2026-09-16 | 1,000 | 183 s | 300 s | as above, measured while the test suite was running |
| 2026-09-16 | 1,000 | 201 s | 300 s | as above, two measurements running concurrently |

**PASS**, with roughly a third of the budget spare in the worst case.

Three things this number depends on, all of which make it pessimistic:

- **No `rapidfuzz`.** The pure-Python Myers fallback was the active path. Installing
  the `fast` extra should cut this substantially; the numbers are identical either
  way, only the time changes.
- **Machine contention.** The 183 s and 201 s figures were taken while other work
  occupied the same four cores. The 148 s figure is the cleanest of the three.
- **Single process.** `--jobs` was 1.

The spread between runs on the *same* machine is the reason the machine is recorded
beside the time. A number without its machine is not a number a stranger can check
(constitution II) — and, on this evidence, a number without its load is not either.

The corpus was generated by `scripts/make_synthetic_eval_set.py --pages 1000`
(50 lines per page), which is the only thing that script is for.
