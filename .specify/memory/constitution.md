<!--
SYNC IMPACT REPORT — scratch material for reviewing this amendment.
Delete this comment block before committing the amended constitution.

Version change: none (template placeholders) → 1.0.0
Bump rationale: initial ratification. The file previously contained only the
unmodified Spec Kit scaffold — no principle had ever been adopted, so there is no
prior version to increment from.

Modified principles:
  [PRINCIPLE_1_NAME] → I. Measurement Before Claims
  [PRINCIPLE_2_NAME] → II. Reproducible by a Stranger
  [PRINCIPLE_3_NAME] → III. Air-Gapped by Construction
  [PRINCIPLE_4_NAME] → IV. Fail Loudly, Never Silently
  [PRINCIPLE_5_NAME] → V. The Image Carries Software Only
  (added)           → VI. Specify Software, Not Operations

Added sections:
  [SECTION_2_NAME]  → Delivery and Technology Constraints
  [SECTION_3_NAME]  → Development Workflow and Quality Gates

Removed sections: none.

Sources the principles were derived from (not invented here):
  README.md "Rules this layout exists to enforce"
  Containerfile header (software only; air-gapped at runtime)
  scripts/build_and_push.sh (refuses 'latest', refuses credential-shaped strings,
    refuses an undersized disk)
  scripts/preflight.py (fail before four H200s are committed to a broken run)
  specs/001-ocr-eval-harness/spec.md FR-013, FR-014, FR-015, FR-016, SC-003, SC-008

Deferred items / TODOs: none.

Follow-up affecting existing artifacts:
  specs/001-ocr-eval-harness/plan.md records that no ratified constitution existed
  and substituted project-derived gates. Those gates are now principles I-V and the
  plan's Constitution Check should be re-run against this file. The substituted
  gates and these principles agree, so no verdict is expected to change.
-->

# Qwen3.5 Arabic OCR Training Constitution

## Core Principles

### I. Measurement Before Claims

No claim about model quality may be made from training loss, sample outputs, or
impression. A quality claim MUST cite a scored evaluation run that another person can
re-run from its recorded manifest.

- The strict, unnormalised score is the headline. Any single number quoted about a
  checkpoint MUST be that score. Looser normalisations exist to locate errors and MUST
  NOT be quoted in place of it.
- Held-out real scans and synthetic pages MUST NOT be merged into one headline figure.
  Where both are measured, they are reported separately.
- A decision to promote or discard a checkpoint MUST be comparative — against the
  untrained baseline, or against the current best — never against an absolute number
  alone.
- Hallucination on unreadable input MUST be measured for any checkpoint proposed for
  promotion, and reported separately from refusals on legible pages.

**Rationale**: This project fine-tunes a language model to read documents. Falling loss
is not evidence that OCR improved, and a language model asked to read a blank page will
confidently invent text that no aggregate error rate reveals. Four H200 GPUs per run
make an unmeasured run an expensive way to learn nothing.

### II. Reproducible by a Stranger

Every artifact that carries a number MUST also carry what produced it: model version,
input digests, configuration, and tool and policy versions. The test is whether someone
who was not present can reconstruct the run six months later from the artifact alone.

- Identical inputs MUST produce identical outputs. Where a report is intended to be
  byte-comparable, wall-clock time, host names and absolute paths MUST be kept out of
  it and recorded separately.
- Container images MUST be tagged with a version. `latest` is refused by the build
  script and MUST NOT be reintroduced.
- Each experiment MUST record the image digest it ran under.
- Scoring rules are versioned. Two results produced under different rule versions MUST
  NOT be compared; tooling MUST refuse rather than warn.

**Rationale**: An unreproducible result is an anecdote. The cost of recording provenance
is a few fields at write time; the cost of not recording it is re-running the experiment
or, worse, trusting a number nobody can defend.

### III. Air-Gapped by Construction

The cluster cannot reach the public internet. Code MUST be written so that this is never
discovered at run time.

- No component may attempt network access at run time. This includes weight downloads,
  dataset fetches, telemetry, and package installation.
- `pip install` at pod startup is forbidden. Dependencies are installed at image build
  time and frozen: once the environment is proven, `requirements.lock.txt` is generated
  and the image builds from it.
- The base image MUST be one the cluster already pulls successfully. Public registry
  paths MUST NOT be introduced.
- Every added third-party dependency is a build risk and MUST be justified against what
  it replaces. Tools that can be stdlib-only SHOULD be stdlib-only.
- Optional CUDA-compiled kernels stay disabled until the plain image trains
  successfully, because they are the most common air-gapped build failure.

**Rationale**: Every one of these rules exists because breaking it fails late — inside a
pod, on a cluster, after a queue wait, with no useful error.

### IV. Fail Loudly, Never Silently

Work that cannot be done correctly MUST stop with a non-zero exit and a message naming
the cause. Guessing, defaulting, or skipping is forbidden where the guess could change a
result.

- Expensive work MUST be preceded by a cheap integrity check. Inputs are validated as a
  whole before any of them is processed.
- A validation failure MUST report every problem found, not only the first.
- Missing, malformed, or mismatched input MUST NOT be scored, imputed, or silently
  dropped. A missing prediction is reported as a gap, never as a perfect or zero score.
- The distinction is between input integrity, which stops the run, and measured system
  behaviour, which is recorded and flagged. Model misbehaviour is data, not an error.

**Rationale**: The failures this project can least afford are the quiet ones — a
corrupted subset scored as if it were fine, a metric computed over the wrong set. A loud
failure costs minutes; a silent one costs the credibility of every number downstream.

### V. The Image Carries Software Only

Model weights, datasets and credentials MUST NOT enter the container image or the
repository. They arrive at run time from object storage and from cluster secrets.

- No credential may be committed, pasted into a chat, or written into a build context.
  The build script refuses credential-shaped strings and that check MUST NOT be weakened
  to make a build pass.
- Paths written by the application MUST be group-writable: the image runs as an
  arbitrary non-root UID.
- Generated artifacts — runs, reports, checkpoints — MUST NOT be committed to the
  repository unless they are small, fixed test fixtures.

**Rationale**: An image with weights in it is too large to move and too dangerous to
share; an image with a credential in it is a disclosure that survives every later fix,
because layers are permanent.

### VI. Specify Software, Not Operations

Spec Kit governs software that does not yet exist and has real requirements — the
evaluation harness, the dataset preparation pipeline, the dataset validator. It MUST NOT
be used for operational procedure.

- Software with requirements gets a spec before an implementation, and a plan before
  code.
- Connecting to a host, inspecting a cluster, building an image or staging a model are
  commands to run, not features to specify. They belong in the operational notes.
- A decision that constrains work outside its own feature MUST be recorded where the
  affected work will see it, not only in the feature that discovered it.

**Rationale**: Specifying operations produces ceremony around a shell command. Not
specifying real software produces the opposite failure — a harness whose scoring rules
live only in the head of whoever wrote it.

## Delivery and Technology Constraints

**Hardware split.** Training runs on the GPU cluster; evaluation, data preparation and
validation MUST run on ordinary CPU hardware. No tool required to judge a model may
depend on cluster access, or judging will be skipped whenever the cluster is busy.

**Dependency policy.** `requirements.txt` is a starting point, not a lock file. Once the
image is proven on a pod, `pip freeze | sort > requirements.lock.txt` and the build
switches to the lock file. Torch and torchvision are not reinstalled over a vendor CUDA
image.

**Build environment.** The image is built on VM01, not a laptop. The build script's
guards — disk space, tag, credential scan — are preconditions, not advisories.

**Verification.** `scripts/preflight.py` is the gate between a built image and a
committed GPU run: imports, GPU count, environment. It runs before training, every time.

## Development Workflow and Quality Gates

**Order of work.** Specify, clarify, plan, generate tasks, implement. `/speckit-analyze`
checks the three artifacts agree before implementation begins.

**The planning gate.** Every plan MUST include a Constitution Check evaluated against
this file. A violation MUST be either removed or justified in the plan's Complexity
Tracking table, naming the simpler alternative rejected and why. An unjustified violation
blocks the plan.

**Testing.** Every metric, transformation or validation rule that produces a reported
number MUST have a unit test. Test suites MUST run offline with no external data —
fixtures live in the repository. Where an output carries a determinism guarantee, a test
MUST assert it by running twice and comparing.

**Review.** A change is reviewable when it states which principle it serves or which one
it strains. Reviewers check compliance with this constitution alongside correctness.

**Runtime guidance.** `README.md` holds the operational rules for building and running
the image. Feature-level design lives under `specs/`. Where either disagrees with this
constitution, this constitution wins and the other document is corrected.

## Governance

This constitution supersedes conflicting practice elsewhere in the project. Where a
README, a script comment or a habit contradicts it, the contradiction is a defect in the
other document.

**Amendment procedure.** Amendments are made through `/speckit-constitution`, which
rewrites this file, records a Sync Impact Report and sets the version. An amendment MUST
state what changed and why. Principles are removed or redefined only with an explicit
rationale recorded in that report.

**Versioning policy.** Semantic versioning of governance:

- **MAJOR** — a principle is removed, or redefined in a way that would invalidate a plan
  previously approved under it.
- **MINOR** — a principle or section is added, or its guidance materially expanded.
- **PATCH** — clarification, wording, or non-semantic refinement.

**Compliance review.** The Constitution Check in each plan is the routine enforcement
point. When a principle is found to be blocking correct work rather than protecting it,
the response is to amend this file, not to route around it quietly.

**Version**: 1.0.0 | **Ratified**: 2026-09-16 | **Last Amended**: 2026-09-16
