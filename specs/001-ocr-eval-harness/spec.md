# Feature Specification: OCR Evaluation Harness

**Feature Branch**: `001-ocr-eval-harness`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "An evaluation harness for the Arabic document OCR model: measure how good a checkpoint is (CER/WER/exact match), compare checkpoints against each other and against the untrained baseline, break errors down by failure category and by font, and detect hallucination on unreadable input."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Score a checkpoint against a reference set (Priority: P1)

An engineer has a set of page images with known correct text, and a set of predictions
produced by some version of the model. They run the harness and get back a single report
saying how wrong the model was, overall and per page.

**Why this priority**: Nothing else in the project can be judged without this. Training
loss going down is not evidence that OCR quality went up. Until this exists, no one can
answer "is the model any good?" — and there is no way to know whether a training run was
worth the four H200 GPUs it consumed. It is also the only deliverable here that needs no
cluster access at all, so it can be built while infrastructure access is still blocked.

**Independent Test**: Run it against the untrained base model's predictions on a few
hundred pages. It should produce a baseline error rate. That number alone is useful — it
is the line every future run must beat.

**Acceptance Scenarios**:

1. **Given** a folder of reference texts and a matching folder of predictions, **When** the
   engineer runs the harness, **Then** it reports character error rate, word error rate and
   exact-match rate, both overall and for each individual page.
2. **Given** the same inputs run twice, **When** the results are compared, **Then** the
   numbers are identical.
3. **Given** a prediction file that is missing for one image, **When** the harness runs,
   **Then** it reports the gap explicitly rather than silently scoring that page as perfect
   or as zero.

---

### User Story 2 - Compare a checkpoint against the baseline and the previous best (Priority: P2)

An engineer has just finished a training run. They want to know: is this better than the
untrained model, and is it better than the best checkpoint we had before?

**Why this priority**: A single error rate in isolation means very little. Decisions —
promote this model, or throw the run away — are always comparative. This is what turns the
harness from a measurement tool into a decision tool.

**Independent Test**: Score two different checkpoints separately, then ask the harness to
compare the two reports. It should say which is better, by how much, and on which pages
the new one got worse.

**Acceptance Scenarios**:

1. **Given** two completed evaluation reports, **When** the engineer compares them,
   **Then** the harness states the change in each headline metric and its direction.
2. **Given** a newer checkpoint that is worse than the older one on some pages, **When**
   compared, **Then** those specific pages are listed as regressions.
3. **Given** a comparison, **When** the engineer reads it, **Then** they can tell without
   further analysis whether to promote the new checkpoint.

---

### User Story 3 - Break errors down by cause (Priority: P3)

An engineer has an error rate but wants to know *why* it is what it is. They break the
results down by font, by the distortion applied to the image, and by failure category, so
that effort goes to the biggest problem rather than the most visible one.

**Why this priority**: An aggregate number tells you that something is wrong, not what to
fix. The data phase deliberately recorded which fonts and which distortions produced each
image specifically so this breakdown would be possible later — that metadata is currently
unused. It was also anticipated during data construction that some fonts may actively harm
training; this is how those fonts get identified and their images removed.

**Independent Test**: Run a breakdown on any scored set and confirm the worst-performing
font can be named from the report alone.

**Acceptance Scenarios**:

1. **Given** a scored evaluation where images carry font metadata, **When** the engineer
   requests a breakdown, **Then** error rates are reported per font, ranked worst first.
2. **Given** the same evaluation, **When** broken down by distortion type, **Then** the
   engineer can see whether a particular degradation is disproportionately damaging.
3. **Given** a scored evaluation, **When** the engineer asks for the worst pages, **Then**
   they receive them sorted by error rate with reference and prediction side by side, so
   failures can be categorised by eye.

---

### User Story 4 - Detect hallucination on unreadable input (Priority: P3)

An engineer feeds the model images that contain no readable text — blank pages, heavy
blur, visual noise, photographs with no writing — and checks whether the model invents
text anyway.

**Why this priority**: This is the failure mode specific to using a language model for
OCR, and the most dangerous one, because invented text looks completely plausible. A model
that confidently transcribes a blank page will corrupt a document archive silently. No
aggregate error rate on normal pages will ever reveal this.

**Independent Test**: Run the probe set through any checkpoint and count how often it
produced text where none exists.

**Acceptance Scenarios**:

1. **Given** a set of images known to contain no readable text, **When** evaluated, **Then**
   the harness reports how often the model produced text anyway.
2. **Given** the project's declared policy for unreadable input, **When** the model's
   response does not match that policy, **Then** it is counted as a failure.
3. **Given** a prediction far longer than its reference, **When** scored, **Then** it is
   flagged as possible runaway generation rather than only contributing to the error rate.

### Edge Cases

- Reference and prediction are both empty — does that count as a perfect score or is it
  excluded from the aggregate?
- The model emits conversational framing ("Here is the text I can see:") around an
  otherwise correct transcription.
- Prediction and reference are visually identical but differ in Unicode representation
  (composed vs decomposed forms, or presentation forms instead of base letters).
- The reference carries full diacritics and the prediction carries none, or the reverse.
- All the correct text is present but in the wrong reading order — a structural failure
  that character-level scoring reports as near-total failure.
- A page mixes Arabic, Latin script and digits, each with different correctness conventions.
- Extremely long pages at the top end of the length distribution.
- A prediction that repeats a line many times, inflating length without adding content.
- An image legitimately containing no text, evaluated as if it were a normal page.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST report character error rate, word error rate and exact-match rate,
  both as an aggregate across the evaluation set and for each individual page.
- **FR-002**: System MUST always report a raw, unnormalised score alongside any normalised
  score. Normalisation that is aggressive enough to hide real errors must never be the only
  number a reader sees.
- **FR-003**: System MUST report every result under three normalisation levels
  simultaneously, so that no single choice of rules can conceal a class of error:
  - **Strict** (the headline number): every difference counts — diacritics, tatweel,
    Arabic and Persian letter variants, punctuation, digit forms, whitespace and Unicode
    representation form.
  - **Diacritic-insensitive**: as strict, but differences in vowelling are ignored.
  - **Skeleton**: consonantal letters and their order only; diacritics, tatweel, variant
    letter forms and Unicode representation are all ignored.
- **FR-003a**: The strict score MUST be the figure quoted whenever a single number is
  reported. The other two exist to locate errors, never to replace it.
- **FR-004**: System MUST record, in every report, exactly which normalisation policy
  produced the numbers, so that two reports are never compared under different rules.
- **FR-005**: System MUST produce both a per-page result set suitable for sorting and
  filtering, and a summary suitable for reading at a glance.
- **FR-006**: System MUST break error rates down by font, by distortion type and by source
  of the page, using metadata already carried by the corpus.
- **FR-007**: System MUST rank the breakdown so the worst-performing category is
  identifiable without further analysis.
- **FR-008**: System MUST compare two evaluation reports and state which is better, by how
  much, and which specific pages got worse.
- **FR-009**: System MUST support a probe set of images containing no readable text and
  report how often the model produced text regardless.
- **FR-010**: The declared behaviour for an unreadable image is that the model emits the
  single fixed marker `[UNREADABLE]` and nothing else. System MUST score against that
  contract: the marker on an unreadable image is a pass; any other text on an unreadable
  image is a hallucination; the marker on a legible image is a false refusal, and MUST be
  counted and reported separately from hallucination.
- **FR-010a**: The training targets MUST use this same marker. A mismatch between the
  scoring contract and the training targets invalidates the measurement.
- **FR-011**: System MUST flag predictions whose length is grossly disproportionate to
  their reference, as an indicator of runaway or repeated generation.
- **FR-012**: System MUST report reading order both ways, because the two answers serve
  different purposes:
  - The headline character error rate MUST be computed on the output **as the model
    produced it**, order errors included, since that is what a downstream consumer receives.
  - An **order-corrected** character error rate MUST be reported alongside it, measuring
    recognition quality independently of sequencing.
  - A **reading-order accuracy** figure MUST be reported, stating how often page elements
    appeared in the correct sequence.
- **FR-012a**: Together these MUST allow a reader to distinguish a model that cannot
  recognise text from one that recognises it correctly but orders it wrongly. The two
  require different fixes.
- **FR-013**: System MUST evaluate a held-out set of genuine scanned pages separately from
  synthetic pages, and never merge the two into one headline number.
- **FR-014**: System MUST produce identical results for identical inputs, and record the
  configuration that produced any given report.
- **FR-015**: System MUST run with no access to the public internet.
- **FR-016**: System MUST fail loudly and stop on malformed, missing or mismatched input
  rather than silently scoring it.
- **FR-017**: System MUST present the worst-scoring pages with reference and prediction
  side by side, so a human can categorise the failure.
- **FR-018**: Users MUST be able to evaluate a subset without re-running the whole set.

### Key Entities *(include if feature involves data)*

- **Reference page**: A page image whose correct text is known. Carries the ground-truth
  transcription, the fonts used, any distortion applied, the source document it came from,
  and whether it is synthetic or a real scan.
- **Prediction**: What one version of the model produced for one reference page. Belongs to
  exactly one model version.
- **Evaluation run**: One scoring of one model version over one reference set under one
  normalisation policy. Identified uniquely, and carrying enough configuration to be
  repeated later by someone who was not there.
- **Page result**: The outcome for a single page — its error rates, whether it matched
  exactly, and any flags raised such as suspected runaway generation.
- **Summary report**: Aggregate outcomes for a run, plus the breakdowns by font, distortion
  and source.
- **Comparison**: The difference between two evaluation runs, including which pages
  improved and which regressed.
- **Gold set**: A small, manually verified set of genuinely difficult real scanned pages —
  small fonts, low contrast, rotation, tables, stamps, multi-column, degraded scans, mixed
  scripts. Held separate and never used for training.
- **Hallucination probe set**: Images known to contain no readable text, used only to
  detect invented output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An engineer who has never used the harness can score a checkpoint and read
  the result within 15 minutes of starting, using only the written instructions.
- **SC-002**: Scoring 1,000 pages completes in under 5 minutes on an ordinary workstation,
  so that evaluation is never the reason someone skips evaluating.
- **SC-003**: Two runs over identical inputs produce byte-identical reports, 100% of the time.
- **SC-004**: Every report shows all three normalisation levels together; no report exists
  in which the strict score is absent or subordinate.
- **SC-004a**: For any evaluated checkpoint, a reader can tell within one minute whether its
  errors are recognition errors or ordering errors, from the report alone.
- **SC-005**: The worst-performing font and the worst-performing distortion type can each
  be named from a single report without further analysis.
- **SC-006**: A comparison between two checkpoints yields a promote-or-reject recommendation
  that a reviewer can act on without opening the underlying data.
- **SC-007**: The hallucination rate on the probe set is reported as a single percentage
  for every evaluated checkpoint, with false refusals on legible pages reported separately.
- **SC-008**: Someone who was not present for the training run can read a report six months
  later and reconstruct which model, which data and which scoring rules produced it.
- **SC-009**: At least 90% of the worst-scoring pages surfaced for human review can be
  assigned a failure category by a reviewer without needing to consult the raw files.

## Assumptions

- The model's expected output is a full-page transcription in correct reading order, as
  recommended in the project report and as the corpus annotation already provides. Should
  the team instead choose a bounding-box or per-element output, the scoring rules change
  and this specification needs revisiting.
- Three decisions previously open on the project were settled while writing this
  specification, on 16 September 2026: the normalisation policy (FR-003), the contract for
  unreadable input (FR-010) and the treatment of reading-order errors (FR-012). The
  unreadable-input decision in particular constrains work outside this feature — the
  training targets must adopt the same marker.
- The harness scores predictions that already exist; generating them by running the model
  is a separate concern. This keeps evaluation usable without GPU access and lets the same
  harness score output from any source, including other OCR systems used as comparators.
- Every corpus page carries the font and distortion metadata recorded during the data phase.
- A gold set of real scanned pages does **not** yet exist and must be built. The public
  documents that were the visual reference for font selection are the natural source. Until
  it exists, all measurement is against synthetic pages only, and will overstate real-world
  quality.
- Evaluation runs on ordinary CPU hardware, not on the GPU cluster.
- The corpus separates original images from their augmented variants, so either can be
  evaluated alone.
- Word error rate is meaningful for Arabic at the whitespace-token level. This is a working
  assumption; if it proves misleading, character error rate remains the primary metric.
