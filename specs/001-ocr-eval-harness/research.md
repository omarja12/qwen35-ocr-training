# Phase 0 Research: OCR Evaluation Harness

**Feature**: `001-ocr-eval-harness` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

Every decision below closes a Technical Context unknown. Each is stated as
Decision / Rationale / Alternatives considered, and each names the requirement it
serves. Nothing here re-opens a question the spec already settled.

---

## R-001: Edit distance engine

**Decision**: Compute Levenshtein distance with `rapidfuzz.distance.Levenshtein`
when it is importable, and fall back to a bundled pure-Python Myers bit-parallel
implementation when it is not. Both paths are exercised by the same test vectors
and must agree exactly.

**Rationale**: Edit distance is an exact integer quantity — there is no accuracy
difference between the two paths, only speed, so the fallback cannot change a
reported number (FR-014). `rapidfuzz` is already present transitively via `jiwer`,
which `requirements.txt` already lists, so the fast path costs the air-gapped image
nothing new. The fallback means the harness runs on a bare laptop with no wheels
installed, which matters because User Story 1 exists specifically to be buildable
while cluster access is blocked.

**Alternatives considered**:
- `jiwer` directly — rejected. It wraps transformation pipelines we do not want; we
  need our own three-level normalisation, our own tokenisation, and raw edit
  distance *counts* per page (not just rates) so aggregates can be micro-averaged.
  Keeping `jiwer` for the training loop and not for the harness costs nothing.
- `python-Levenshtein` — rejected. Another dependency to freeze into the image for
  no capability `rapidfuzz` does not already provide.
- `difflib.SequenceMatcher` — rejected. It computes a similarity ratio on a
  longest-matching-block heuristic, not true edit distance; it is not the metric.

---

## R-002: "Raw" and "strict" are the same score

**Decision**: The strict level applies **zero** text transformation. FR-002's
required raw score and FR-003's strict headline score are therefore one number,
reported once and labelled `strict`. The report states explicitly that strict is
untransformed.

**Rationale**: FR-003 defines strict as counting every difference including Unicode
representation form, which is exactly what "unnormalised" means. Emitting a
separate identical column labelled `raw` would invite the reader to believe two
things were measured. One number, clearly labelled, satisfies both requirements.

**One documented exception**: a single trailing newline is stripped from every
reference and prediction *at file-read time*, because it is an artifact of how text
files are written, not of what the model produced. This is an I/O rule, not a
scoring rule, it applies identically to both sides, and it is recorded in the run
manifest. No other whitespace is touched at strict level.

**Alternatives considered**: reporting `raw` and `strict` as separate columns —
rejected as misleading. Stripping all trailing whitespace — rejected; that is
scoring normalisation smuggled in through the reader.

---

## R-003: The three normalisation levels, defined exactly

**Decision**: Three levels, applied as ordered pipelines. Order is part of the
definition because these operations do not commute.

**Level 1 — `strict`** (headline, FR-003a)

1. (nothing)

**Level 2 — `diacritic_insensitive`**

1. Remove Arabic combining marks: `U+064B`–`U+065F`, `U+0670`, `U+06D6`–`U+06DC`,
   `U+06DF`–`U+06E4`, `U+06E7`, `U+06E8`, `U+06EA`–`U+06ED`, `U+08D3`–`U+08E1`,
   `U+08E3`–`U+08FF`.
2. Remove any remaining character of Unicode category `Mn` whose base is Arabic —
   this catches NFD-decomposed forms such as `U+0622` written as `U+0627 U+0653`.

Nothing else changes. Tatweel, letter variants, digit forms, punctuation,
whitespace and representation form all still count, exactly as FR-003 requires
("as strict, but differences in vowelling are ignored").

**Level 3 — `skeleton`**

1. NFKC (folds presentation forms `U+FB50`–`U+FDFF` and `U+FE70`–`U+FEFF` back to
   base letters, and ligatures to their parts).
2. Remove all combining marks (category `Mn`).
3. Remove tatweel `U+0640`.
4. Letter folding table (below).
5. Digit folding: `U+0660`–`U+0669` and `U+06F0`–`U+06F9` to ASCII `0`–`9`.
6. Remove all characters of category `P*` and `S*` (punctuation and symbols).
7. Collapse each run of whitespace to a single `U+0020`; strip both ends.

**Letter folding table (skeleton only)**

| From | To | Note |
|---|---|---|
| `U+0622 U+0623 U+0625 U+0671 U+0672 U+0673 U+0675` | `U+0627` | alef variants to bare alef |
| `U+0624` | `U+0648` | hamza-on-waw to waw |
| `U+0626` | `U+064A` | hamza-on-yeh to yeh |
| `U+0621` | (removed) | standalone hamza is not part of the rasm |
| `U+0649` | `U+064A` | alef maksura to yeh |
| `U+06CC` | `U+064A` | Persian yeh to Arabic yeh |
| `U+06A9` | `U+0643` | keheh to Arabic kaf |
| `U+0629` | `U+0647` | teh marbuta to heh |
| `U+06C0 U+06C1 U+06C2` | `U+0647` | heh variants to heh |

**Rationale**: The spec fixes *what* each level means; the harness has to fix *how*,
or two runs on two machines are not comparable. The tables above are the standard
Arabic rasm reduction used in Arabic information retrieval. Every transformation is
information-destroying in a direction the spec explicitly authorises at that level,
and never at a level below it.

**Alternatives considered**:
- `camel-tools` normalisation — rejected. A heavy dependency with its own data
  files to ship into an air-gapped image, for a table small enough to read in one
  screen. Its choices would also be invisible to the reader of a report, which
  FR-004 forbids.
- Applying NFC at strict level to "be fair" — rejected. FR-003 says representation
  differences count at strict. A model emitting presentation forms is producing
  output a downstream consumer will have to deal with, and that must show up.

---

## R-004: Policy versioning and cross-report safety

**Decision**: The normalisation tables carry a `policy_version` string (starting
`"1.0"`). It is written into every run manifest and every report. `compare` refuses
to run when the two runs differ on any of four things — `policy_version`,
reference-set digest, split set, or **page set** — and exits 3 with a message naming
the mismatch.

**Rationale**: FR-004 requires knowing which rules produced a number; FR-008
compares two reports. The only way those two requirements are jointly safe is for
comparison to be a hard gate, not a warning. A warning gets ignored the one time it
matters.

The page set is the fourth condition because FR-018 makes it the reachable one. Two
runs produced with different `--pages` or `--filter` arguments over the same corpus
share a policy version, share a manifest digest and share a split set; every other
guard passes, and the comparison silently reports the difference between two
different populations as if it were the difference between two models. A subset is
exactly the case an engineer reaches for when iterating quickly, so this is the guard
most likely to be needed and the one whose absence would be hardest to notice in the
output.

**Alternatives considered**: warn-and-continue — rejected; it produces a
plausible-looking comparison of two incommensurable numbers, which is the exact
class of silent error FR-016 is written against.

---

## R-005: Aggregation — micro and macro

**Decision**: Report both, headline is micro.

- **Micro CER** = `sum(edit_distance) / sum(len(reference))` across pages.
- **Macro CER** = mean of per-page CER.

Same for WER. Exact-match rate is a page count ratio and has only one form.

**Rationale**: Micro is the corpus-level truth and is what a downstream consumer
experiences; macro is dominated by short pages and reveals when failures cluster on
them. Reporting only one hides a real effect: a model that is fine on long pages and
catastrophic on short ones looks healthy under micro alone. Both are cheap.

**Per-page CER denominator**: `len(reference)` after the level's normalisation. If a
reference normalises to the empty string the page is a data error and the run fails
(see R-009), so the denominator is never zero.

---

## R-006: Word error rate tokenisation

**Decision**: Tokenise on runs of Unicode whitespace, after the level's
normalisation, discarding empty tokens. No punctuation splitting, no morphological
segmentation.

**Rationale**: The spec already flags WER-for-Arabic as a working assumption with
CER as the primary metric. A whitespace token is the one definition a reader can
reconstruct without consulting us, which matters for SC-008. At skeleton level
punctuation has already been removed, so skeleton WER is the closest thing to a
content-word metric the harness offers.

**Alternatives considered**: morphological tokenisation — rejected. It imports a
large Arabic NLP stack into an air-gapped image to make a secondary metric slightly
better, and makes the number unreconstructable by a reader.

---

## R-007: Reading order — segmentation, matching, and the two CERs

**Decision**: The page element is a **non-empty line** of the reference
transcription. FR-012 is then implemented as:

1. Split reference and prediction into non-empty lines, keeping original indices.
2. Build a similarity matrix over **skeleton-normalised** lines using normalised
   Levenshtein similarity (`1 - dist/max(len)`).
3. **Greedy deterministic matching**: take all pairs with similarity `>= 0.5`,
   sort by `(-similarity, ref_index, pred_index)`, and accept a pair when neither
   side is already matched. The sort key makes ties resolve identically every run.
4. **Order-corrected CER**: rebuild the prediction as matched lines emitted in
   *reference* order, followed by unmatched prediction lines in their original
   order, joined by `\n`. Score that rebuilt text normally. This measures
   recognition with sequencing removed.
5. **Reading-order accuracy**: over all unordered pairs of matched lines `(a, b)`,
   the fraction where `sign(ref_a - ref_b) == sign(pred_a - pred_b)` — Kendall
   concordance mapped to `[0, 1]`. Pages with fewer than two matched lines report
   `null` and are excluded from the aggregate rather than scoring a free 1.0.
6. The headline CER is unchanged and computed on the model's raw output (FR-012).

**Rationale**: FR-012a is the actual requirement — a reader must be able to tell a
model that cannot read from a model that reads fine and orders badly. The three
numbers side by side do that: high CER + low order-corrected CER + low order
accuracy is a sequencing problem; high CER in both is a recognition problem.
Similarity matching rather than exact matching is necessary because a line can be
both misordered and slightly misread. The `0.5` floor stops unrelated lines from
being paired, which would flatter the order-corrected number.

**Cost**: `O(R x P)` short-line comparisons per page. At 50 lines a side that is
2,500 comparisons of ~60-character strings — microseconds. Well inside SC-002.

**Alternatives considered**:
- Hungarian / optimal assignment — rejected for v1. It needs SciPy (a heavy
  air-gapped dependency) or a hand-rolled O(n^3); greedy on a similarity floor
  differs from optimal only on pathological pages, and the choice is recorded in the
  manifest so it can be revisited without invalidating old reports.
- Bag-of-lines CER (sort both sides) — rejected. It destroys the order signal we are
  specifically trying to measure.
- Paragraph or block segmentation — deferred. Lines are what the corpus annotation
  provides today; the segmentation function is isolated behind one interface so a
  block-level unit can replace it later.

---

## R-008: The `[UNREADABLE]` contract and its classification matrix

**Decision**: A prediction "is the marker" when its text, with leading and trailing
whitespace stripped, equals exactly `[UNREADABLE]` — case-sensitive, nothing else
on the page. Pages are then classified:

| Reference | Prediction | Outcome | Counted in |
|---|---|---|---|
| `[UNREADABLE]` | `[UNREADABLE]` | `correct_refusal` | probe set |
| `[UNREADABLE]` | anything else | `hallucination` | probe set |
| legible text | `[UNREADABLE]` | `false_refusal` | legible set, **reported separately** |
| legible text | anything else | `scored` | legible set, CER/WER/EM |

`hallucination_rate = hallucination / (correct_refusal + hallucination)`.
`false_refusal_rate = false_refusal / legible_pages`.

Probe pages are **excluded** from CER/WER/exact-match aggregates entirely — a probe
page has no reference text to measure distance against. False refusals **are**
included in CER (they are a wrong transcription of a legible page) *and* counted
separately, because FR-010 requires the separate count and FR-001 requires the page
to appear in the aggregate.

**Rationale**: FR-010 spells out three of these four cells; the fourth is the normal
case. Requiring the marker to stand alone is what makes the contract checkable — a
model that writes "`[UNREADABLE]` — possibly a stamp?" has not honoured it, and
treating that as a pass would let hallucination hide inside a compliant-looking
prefix.

**Cross-feature consequence (FR-010a)**: the dataset preparation work must emit this
exact marker as the training target for unreadable images. The harness writes the
expected marker string into every run manifest so the mismatch is at least visible
if it happens.

---

## R-009: Failing loudly — what counts as malformed input

**Decision**: `score` validates the whole input set **before** scoring any page, and
exits non-zero listing every problem found (not just the first). Hard failures:

- a manifest page with no matching prediction, or a prediction with no manifest page
  (FR-001 scenario 3 — the gap is reported, never scored as 0 or 100%);
- any file that is not valid UTF-8;
- a reference that is empty or whitespace-only and is *not* marked as a probe page;
- a reference on a non-probe page that is non-empty raw but **normalises to empty at
  any of the three levels** — a reference of nothing but punctuation and symbols
  empties at skeleton, where step 6 removes every `P*` and `S*` character; one of
  nothing but diacritics empties at diacritic-insensitive. The check runs each level
  and names the level that emptied it;
- a manifest row missing a required field, or carrying an unknown `kind`;
- two manifest rows sharing a `page_id`;
- a `--pages` subset file naming an id absent from the manifest.

Not failures, but flagged per page and counted in the summary: empty *prediction*
for a legible page (a real model behaviour, scored as total loss), runaway-length
predictions, false refusals.

**Rationale**: FR-016. The distinction that matters is input integrity (our problem,
stop) versus model behaviour (the thing being measured, score it and flag it). The
edge case "reference and prediction both empty" resolves here: an empty reference on
a legible page is a corpus defect and the run stops, so the ambiguous cell never
arises.

The normalises-to-empty rule is what makes the zero-denominator case unreachable
rather than merely unlikely. Without it a legible page whose reference is `«...»`
produces `ref_chars = 0` at skeleton level, and CER divides by zero on a page the
raw-emptiness check waved through. Catching it at validation keeps the guarantee
where the rest of the input contract already lives — one whole-set check, every
problem named — instead of forcing every metric to carry a defensive branch. Both
denominators are covered by the one rule: tokens are runs of non-whitespace after
normalisation, so a non-empty normalised reference always yields at least one word.

**Validation is also its own command** (`ocr-eval validate`) so the corpus can be
checked without waiting for a scoring run.

---

## R-010: Runaway generation detection

**Decision**: Two independent deterministic flags, thresholds configurable and
recorded in the manifest:

- `runaway_length` — `len(pred) >= 2.0 * len(ref)` **and** `len(pred) - len(ref) >= 100`
  characters. The second clause stops short pages from tripping it constantly.
- `repetition` — some skeleton-normalised line of length `>= 10` occurs `>= 5` times
  in the prediction **and** those repeats account for `>= 30%` of prediction
  characters.

Flagged pages are listed in the summary with their ratios, and counted; they are not
excluded from the aggregate.

**Rationale**: FR-011 asks for a flag, not an exclusion — the errors are real and
belong in the error rate, but a reader needs to know that a chunk of the CER comes
from one page repeating a line 400 times rather than from a broad recognition
failure. Two separate signals because the failure has two shapes: bounded-but-inflated
output, and a stuck decode loop.

**Alternatives considered**: a single length-ratio threshold — rejected; it misses
a loop that repeats within roughly the right output length. Perplexity or
entropy-based detection — rejected; it would require loading the model, which
breaks the "score existing predictions on CPU" design (R-012).

---

## R-011: Byte-identical reports (SC-003)

**Decision**: Report artifacts contain no wall-clock time, no host name, no absolute
paths, and no unordered iteration. Specifically:

- JSON written with `sort_keys=True`, `ensure_ascii=False`, `indent=2`, LF endings,
  UTF-8 without BOM, and a trailing newline.
- All floats rounded to 6 decimal places at serialisation, formatted through one
  shared helper.
- Per-page rows sorted by `page_id` with a plain codepoint sort, independent of
  locale and independent of the order work completed.
- Paths recorded relative to the corpus root and the run root.
- Wall-clock time, host, and elapsed seconds go in `run.meta.json`, which is
  explicitly **outside** the byte-identity guarantee and is never read by `compare`.
- `--jobs N` may reorder execution; it may not reorder output.

**Rationale**: SC-003 says byte-identical, 100% of the time. A timestamp inside the
report would make that impossible, so provenance that genuinely varies per run is
segregated into a file that is not part of the comparison surface. The run manifest
still carries everything SC-008 needs to reconstruct the run six months later — model
id, input digests, config, policy version, tool version.

---

## R-012: The harness never loads the model

**Decision**: `score` consumes predictions that already exist on disk. Generating
them is out of scope, as the spec assumes.

**Rationale**: Stated in the spec's Assumptions, and it buys three things: the
harness is usable with no GPU and no cluster access (the reason User Story 1 is P1),
the same harness scores any OCR system used as a comparator, and evaluation stays
CPU-only per SC-002. It also keeps FR-015 (no internet) trivially true — there is no
weight download path to close.

---

## R-013: Splits are never merged

**Decision**: Every manifest row carries a `split` (e.g. `synthetic_heldout`,
`gold_scans`, `probe`). The summary reports headline metrics **per split**. When a
run covers more than one split the summary contains no cross-split total — the key
is simply absent, not zero, not averaged.

**Rationale**: FR-013 forbids merging real scans with synthetic pages into one
headline. The strongest form of that guarantee is that the merged number does not
exist anywhere in the artifact, so it cannot be quoted by accident.

**Note carried from the spec**: the gold set does not exist yet. Until it does,
every report is synthetic-only and overstates real-world quality. The summary prints
that sentence verbatim when no `gold_scans` split is present.

---

## R-014: Promote-or-reject recommendation (SC-006)

**Decision**: `compare` emits one of `PROMOTE`, `REJECT`, `INCONCLUSIVE` from a
deterministic rule over strict-level micro metrics, with all thresholds in config
and echoed into the comparison artifact:

- **REJECT** if any of: gold-set strict CER worsens at all; hallucination rate
  worsens by more than 1.0 percentage point; false-refusal rate worsens by more than
  1.0 percentage point.
- **PROMOTE** if no reject condition holds **and** strict micro CER on the primary
  held-out split improves by at least 0.5 percentage points absolute.
- **INCONCLUSIVE** otherwise.

Alongside it, a **paired bootstrap** over per-page CER deltas (10,000 resamples,
fixed seed `20260916`, recorded) gives a 95% interval on the difference, so a
0.4-point improvement that straddles zero is visibly not a result.

**Rationale**: SC-006 wants a reviewer to act without opening the data, which means
the rule has to be stated, not felt. Making gold-set regression an automatic reject
encodes the project's actual risk: synthetic gains that do not survive real scans.
The bootstrap is cheap (per-page deltas are already computed), needs no dependency
beyond `random.Random(seed)`, and is deterministic.

**Alternatives considered**: a single-threshold rule on CER alone — rejected; it
would promote a checkpoint that reads 0.5 points better and hallucinates twice as
often, which is the trade the spec calls the most dangerous failure mode.

---

## R-015: CLI, packaging and dependencies

**Decision**: A single installable package `ocr_eval` under `src/`, exposing an
`ocr-eval` console script and also runnable as `python -m ocr_eval`. CLI built on
stdlib `argparse` with four subcommands: `validate`, `score`, `compare`, `report`.
Runtime dependencies: **none required** (stdlib only); `rapidfuzz` optional and used
when present. Test dependency: `pytest`.

**Rationale**: FR-015 and the README's air-gapped rules make every added wheel a
build risk. `argparse` costs nothing and is already understood by anyone reading the
code. `python -m` support means the harness runs from a checkout with
`PYTHONPATH=src` even where nothing can be installed — which is the realistic
situation on a locked-down workstation.

**`report` exists separately from `score`** so the human-readable rendering can be
regenerated, re-sorted, or re-rendered with a different "worst N" without rescoring.

**Alternatives considered**: Typer/Click — rejected, a dependency for cosmetics.
Vendoring the whole thing as one script — rejected; the normalisation tables, the
metrics and the reporting want separate tests, and SC-009 makes the report renderer
substantial in its own right.

---

## R-016: Displaying Arabic side by side (FR-017, SC-009)

**Decision**: The worst-pages artifact is written as HTML with
`dir="rtl" lang="ar"` on the text cells, plus a Markdown twin for terminal reading.
Each entry shows: page id, split, fonts, distortions, strict CER, order-corrected
CER, flags, then reference and prediction in adjacent cells, then a **character
difference list** naming each differing codepoint as `U+XXXX NAME` for the first 40
differences.

**Rationale**: SC-009 wants 90% of surfaced pages categorisable without opening raw
files. Rendered Arabic alone does not achieve that for the confusable cases the spec
lists — composed versus decomposed forms, presentation forms, missing diacritics all
look identical or near-identical on screen. The codepoint list is what makes those
categorisable by eye. RTL markup is required or the reference and prediction will
not line up visually at all.

**Alternatives considered**: terminal-only output — rejected; terminals handle
bidirectional Arabic inconsistently, and a reviewer comparing two RTL strings in a
shell is exactly the situation SC-009 is trying to avoid.

---

## R-017: Output permissions for an arbitrary non-root UID

**Decision**: `io/writer.py` sets permissions explicitly on everything it creates —
`0775` on directories, `0664` on files — rather than inheriting the ambient umask.
On Windows the calls are no-ops and the tests skip accordingly.

**Rationale**: Constitution principle V requires application-written paths to be
group-writable, because the training image runs as an arbitrary non-root UID assigned
by OpenShift. A run directory created under a umask of `077` is unreadable to the next
pod, and that failure surfaces as a confusing permission error in an unrelated job
rather than as a design mistake here. The cost is two `os.chmod` calls in the one
module that already exists to centralise how things are written.

**Scope**: run directories, the artifacts inside them, and comparison output. The
harness creates nothing else.

**Alternatives considered**:
- Rely on the process umask — rejected. It makes correctness depend on how the
  process was launched, which is exactly the class of failure that only appears in
  the cluster.
- Set the umask globally at CLI start-up — rejected. It is a process-wide side effect
  from a library, and it would silently affect anything else running in-process.

**Recorded after the constitution was ratified on 2026-09-16.** The rest of this
design predates it and needed no change; this is the one constraint ratification made
visible.

---

## R-018: A split with no scoreable page reports `null`, not zero

**Decision**: `SplitSummary` gains a `scored_pages` count — pages whose outcome is
`scored` or `false_refusal`, the ones that carry metrics. When `scored_pages` is `0`,
`headline` and `metrics` are `null`, `order.order_corrected_cer_micro` is `null`, and
`breakdowns` is empty. `contract`, `flags` and `pages` are still reported in full,
because those are exactly the numbers such a split does have. The nulls occur in that
case and in no other; `headline` and `metrics` are never null independently of each
other.

**Rationale**: R-008 excludes probe pages from CER, WER and exact-match entirely — a
page with no reference text has no distance to measure — and R-013 gives the probe
set its own `split`. Together those two decisions produce a split whose every
aggregate denominator is zero. The schema previously required `headline.cer` as a
number, which left an implementer with no legal option but to write `0.0`, and a
`0.0` CER reads as a *perfect* score on the one split that contains nothing scoreable.
That is a defaulted number that changes a reported result, which constitution
principle IV forbids in as many words.

`null` rather than absence keeps SC-004 intact: the strict headline is never
*omitted* from a split that has one. And `null` rather than `0` is the idiom these
schemas already use three times — `reading_order_accuracy` below two matched lines,
`hallucination_rate` with no probe pages, `false_refusal_rate` with no legible pages.
This is the same rule applied one level up.

`scored_pages` exists so the null explains itself. A reader who sees a null headline
and a `scored_pages` of 0 needs no further lookup; a null with no denominator beside
it looks like a bug.

**Alternatives considered**:
- Omit probe splits from `splits[]` entirely — rejected. The probe split's contract
  counts are the SC-007 deliverable, and moving them to a top-level key outside the
  per-split structure would make the one figure the spec calls the most dangerous
  failure mode the only figure not reported per split.
- Report `headline.cer` as `0.0` with a note — rejected. A note is not a guarantee;
  the number is still there to be quoted, and R-013 already establishes that the
  strongest form of "do not quote this" is for the number not to exist.

**Recorded 2026-09-16 from `/speckit-analyze` finding F1**, which found the conflict
between this schema and R-008 before any code was written.

---

## Resolved Technical Context summary

| Unknown | Resolution |
|---|---|
| Language/version | Python 3.10+ (3.11 target, matching the training image) |
| Distance engine | `rapidfuzz` if present, pure-Python Myers fallback (R-001) |
| Normalisation | Three explicit pipelines, versioned tables (R-003, R-004) |
| Storage | Files only — JSONL per-page, JSON summary, Markdown/HTML report |
| Testing | `pytest`, with golden-file tests for byte-identity |
| Platform | CPU, Linux and Windows, fully offline (R-012, FR-015) |
| Project type | Library + CLI, single project |
| Performance | 1,000 pages under 5 minutes single-process (SC-002) |
| Scale | Tens of thousands of pages per corpus; ~50 lines per page |
