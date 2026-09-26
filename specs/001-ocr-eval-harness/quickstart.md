# Quickstart: OCR Evaluation Harness

**Feature**: `001-ocr-eval-harness` | **Date**: 2026-09-16

Runnable scenarios that prove the feature works end to end. Each maps to a user
story or success criterion in [spec.md](./spec.md). This is a validation and run
guide — implementation detail belongs in `tasks.md`, the command surface in
[contracts/cli.md](./contracts/cli.md).

SC-001 is the bar for this page: an engineer who has never used the harness should
get from here to a read result in under 15 minutes.

---

## Prerequisites

- Python 3.10 or newer. No GPU, no cluster access, no network.
- A checkout of this repository.
- Nothing else. `rapidfuzz` makes scoring faster when present but changes no number
  (research R-001), so the harness runs on a bare interpreter.

```bash
# from the repository root
pip install -e .                 # installs the ocr-eval console script
# or, where nothing can be installed:
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH = "src"
python -m ocr_eval --version
```

Expected: `ocr-eval 0.1.0 (policy 1.0)`.

---

## Scenario 1 — Score a checkpoint (User Story 1, FR-001)

The fixture corpus in `tests/fixtures/mini_corpus/` ships with the repo — about
twenty pages covering clean text, misordered lines, probe images and a runaway
prediction — so this runs with no external data.

```bash
ocr-eval score \
  --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/baseline/ \
  --model-version Qwen3.5-VL-base \
  --out runs/demo-baseline
```

**Expected outcome**

- Exit code 0.
- `runs/demo-baseline/` containing `manifest.json`, `pages.jsonl`, `summary.json`,
  `summary.md`, `worst_pages.md`, `worst_pages.html`, `run.meta.json`.
- stdout shows a strict headline per split, plus the `[UNREADABLE]` contract counts.
- `summary.md` shows all three normalisation levels together, with the strict figure
  as the headline (SC-004) — never a report where strict is absent or nested below
  another level.

**What to check by eye**

```bash
cat runs/demo-baseline/summary.md
head -1 runs/demo-baseline/pages.jsonl | python -m json.tool
```

Every page has a per-page CER, WER and exact-match at each level (FR-001), and the
run's `manifest.json` names the model, the input digests and the policy version, so
the run is reconstructable later (SC-008).

---

## Scenario 2 — Determinism (SC-003, FR-014)

```bash
ocr-eval score --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/baseline/ \
  --model-version Qwen3.5-VL-base --out runs/demo-repeat

# every file except run.meta.json must be byte-identical
diff -r --exclude=run.meta.json runs/demo-baseline runs/demo-repeat && echo "IDENTICAL"
```

**Expected outcome**: `IDENTICAL`, every time. Wall-clock time, host and elapsed
seconds live in `run.meta.json` precisely so they cannot break this guarantee
(research R-011).

---

## Scenario 3 — A missing prediction is reported, not scored (FR-001 scenario 3, FR-016)

```bash
mv tests/fixtures/mini_corpus/predictions/baseline/syn_000142.txt /tmp/
ocr-eval validate \
  --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/baseline/
echo "exit: $?"
mv /tmp/syn_000142.txt tests/fixtures/mini_corpus/predictions/baseline/
```

**Expected outcome**: exit 1, stderr naming `syn_000142` as missing. The page is
never silently scored as perfect or as zero. `score` performs the same check before
it scores anything, so a corpus problem costs seconds rather than a whole run.

---

## Scenario 4 — Compare two checkpoints (User Story 2, FR-008, SC-006)

```bash
ocr-eval score --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/finetuned/ \
  --model-version qwen35-ocr-ft-v003-step4000 --out runs/demo-ft

ocr-eval compare \
  --baseline runs/demo-baseline \
  --candidate runs/demo-ft \
  --out runs/demo-compare
```

**Expected outcome**

- Exit 0, and `runs/demo-compare/comparison.{json,md}`.
- Each headline metric with both values, the delta and a direction stated as a word
  (`better` / `worse` / `unchanged`) — not a bare sign the reader has to interpret.
- A list of the specific pages that regressed, worst first.
- One of `PROMOTE`, `REJECT`, `INCONCLUSIVE` with every rule that fired, plus a 95%
  bootstrap interval on the CER delta.

Read the verdict from the artifact, not from `$?` — a `REJECT` is a successful
comparison and still exits 0.

**Also check the guard works:**

```bash
# comparing runs over different corpora must refuse, not produce a plausible number
ocr-eval compare --baseline runs/demo-baseline --candidate runs/some-other-corpus-run
echo "exit: $?"      # expect 3, with the mismatch named
```

The case most likely to catch you is subtler, because every obvious field matches —
two runs over the **same** corpus at the same policy version, scored over different
subsets:

Use `--pages` with id lists that span every split, so the split sets stay identical
and the **page set** is the only thing that differs:

```bash
printf '%s\n' syn_clean_001 syn_clean_002 gold_scan_001 probe_blank_001 > /tmp/set_a.txt
printf '%s\n' syn_clean_001 syn_clean_003 gold_scan_001 probe_blank_001 > /tmp/set_b.txt

ocr-eval score --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/baseline \
  --model-version demo-a --pages /tmp/set_a.txt --out runs/subset-a

ocr-eval score --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/finetuned \
  --model-version demo-b --pages /tmp/set_b.txt --out runs/subset-b

ocr-eval compare --baseline runs/subset-a --candidate runs/subset-b
echo "exit: $?"      # expect 3 — same policy, same corpus digest, same splits,
                     # different page set; the offending ids are named on stderr
```

Comparing those two would report the difference between two populations as if it were
the difference between two models (research R-004).

> A filter such as `--filter 'font=Amiri'` is also refused, but for a different
> reason: probe pages carry no fonts, so filtering on one drops the `probe` split
> entirely and the **split-set** check fires first. That still protects you — it just
> does not demonstrate the page-set guard, which is the one FR-018 makes reachable in
> ordinary use.

---

## Scenario 5 — Break errors down by cause (User Story 3, FR-006, SC-005)

```bash
ocr-eval report --run runs/demo-baseline --breakdown all
```

**Expected outcome**: tables for font, distortion and source, each ranked worst
first, so the worst-performing font is the first row and can be named from the
report alone. Categories with fewer than five pages are folded into a `__sparse__`
row rather than topping the ranking on one bad page.

```bash
ocr-eval report --run runs/demo-baseline --worst 20 --format html --out /tmp/worst.html
```

Opens as reference and prediction side by side with correct RTL rendering, plus a
codepoint difference list (`U+XXXX NAME`) for the first 40 differences. That list is
what makes the confusable cases categorisable — composed vs decomposed forms,
presentation forms and missing diacritics look identical on screen (SC-009).

---

## Scenario 6 — Recognition failure vs ordering failure (FR-012a, SC-004a)

The fixture corpus includes `syn_shuffled_*` — pages whose prediction is the correct
text with its lines shuffled.

```bash
python -c "
import json
for line in open('runs/demo-baseline/pages.jsonl', encoding='utf-8'):
    r = json.loads(line)
    if r['page_id'].startswith('syn_shuffled'):
        print(r['page_id'], r['metrics']['strict']['cer'],
              r['order']['order_corrected_cer'], r['order']['reading_order_accuracy'])
"
```

**Expected outcome**: a high strict CER, a near-zero order-corrected CER, and a low
reading-order accuracy. That combination is the signature of a model that reads
correctly and sequences wrongly — a different fix from a model that cannot read, and
the reader should be able to tell which within a minute.

---

## Scenario 7 — Hallucination on unreadable input (User Story 4, FR-010, SC-007)

Probe pages (`kind: probe`, reference exactly `[UNREADABLE]`) are already in the
fixture corpus.

```bash
python -c "
import json
s = json.load(open('runs/demo-baseline/summary.json', encoding='utf-8'))
for sp in s['splits']:
    print(sp['split'], sp['contract'])
"
```

**Expected outcome**: for the `probe` split, counts of `correct_refusal` and
`hallucination` and a single `hallucination_rate`. For legible splits, a
`false_refusal_rate` reported **separately** — the two are never summed, because a
model that invents text and a model that refuses legible pages need opposite fixes.

Probe pages carry no `metrics` object at all: there is no reference text to measure
distance against.

---

## Scenario 8 — Evaluate a subset without re-running everything (FR-018)

```bash
ocr-eval score --manifest tests/fixtures/mini_corpus/manifest.jsonl \
  --predictions tests/fixtures/mini_corpus/predictions/baseline/ \
  --model-version Qwen3.5-VL-base \
  --filter 'font=Amiri,is_augmented=false' \
  --out runs/demo-amiri-originals
```

**Expected outcome**: only matching pages scored; `manifest.json` records the parsed
filter as structured data, so the run's scope is reconstructable from the artifact
alone. `--pages <file>` does the same for an explicit id list.

---

## Scenario 9 — Performance (SC-002)

```bash
python scripts/make_synthetic_eval_set.py --pages 1000 --out /tmp/perf_corpus
time ocr-eval score --manifest /tmp/perf_corpus/manifest.jsonl \
  --predictions /tmp/perf_corpus/predictions/ \
  --model-version perf-check --out /tmp/perf_run
```

**Expected outcome**: under 5 minutes single-process on an ordinary workstation.
Expected actual is well under a minute; the budget exists so that evaluation is
never the reason someone skips evaluating.

---

## Running the test suite

```bash
pytest                       # whole suite, no external data required
pytest tests/golden          # SC-003 byte-identity only
pytest tests/unit/test_normalise.py -v   # the three levels, per codepoint
```

The suite runs fully offline. Unit tests cover the normalisation tables, the metrics
and the contract matrix; integration tests cover every hard failure listed in
research R-009; golden tests assert byte-identity against committed artifacts.

---

## Known limitation to state out loud

**The gold set of real scanned pages does not exist yet.** Until it does, every
number produced here is measured on synthetic pages and **overstates real-world
quality**. The harness prints this verbatim in any summary with no `gold_scans`
split present, and it is not a warning to suppress — building that set is the single
largest improvement available to the credibility of these numbers.
