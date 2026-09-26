# qwen35-ocr-training

The air-gapped training image for the Arabic OCR fine-tune. This is **Step 6**
of Part III in `../OCR_Project_Report_EN.pdf`.

The image carries software only. Model weights, the dataset and credentials all
arrive at run time — from MinIO and from an OpenShift `Secret`.

## Before this will build

| Prerequisite | Why | Status |
|---|---|---|
| SSH access to VM01 | the image is built there, not on a laptop | needed |
| VM01 disk extension (+50 GB) | build layers alone exceed the ~10 GB free | needed |
| The internal base image path | the cluster cannot pull from docker.io | needed |
| The registry token | to push | held by the team |

Find the base image from a GPU deployment that already runs here:

```bash
oc get deployment <A_RUNNING_GPU_DEPLOYMENT> \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
```

## Build and push (on VM01)

```bash
export BASE_IMAGE='<the image path from above>'
export REGISTRY='<registry host>' REGISTRY_ORG='<organisation>' QUAY_USER='<robot account>'
export QUAY_TOKEN='<token>'          # from a password manager
./scripts/build_and_push.sh v001
```

The script refuses to run if the disk is too small, if `latest` is used as a
tag, or if a credential-shaped string is found in the build context.

## Check it inside a pod

```bash
python /opt/ocr-training/scripts/preflight.py          # imports + 4 GPUs + env
python /opt/ocr-training/scripts/preflight.py --gpus 1 # a single-GPU look
```

## Rules this layout exists to enforce

- No model weights, no dataset, no credentials in the image.
- No `pip install` at pod startup — freeze `requirements.lock.txt` once the
  environment is proven, then build from it.
- Versioned tags only. Record the digest with each experiment.
- Optional CUDA kernels (`liger-kernel`, `flash-linear-attention`) stay commented
  out until the plain image trains. They are the usual air-gapped build failure.
- The image must run as an arbitrary non-root UID, so paths are group-writable.

---

## Training

The full fine-tune of Qwen3.5-9B-Base on the ~4.1M-page `OCR-Data/ocr_data`
dataset, from Hugging Face to a scored checkpoint, is in
[`docs/TRAINING.md`](docs/TRAINING.md). The pieces:

| | |
|---|---|
| `scripts/object_store.py` | Hugging Face -> MinIO -> PVC, SHA-256 verified end to end |
| `python -m ocr_train.prepare` | tar shards -> ms-swift `train.jsonl` / `val.jsonl` + an `ocr-eval` test corpus with `[UNREADABLE]` probes |
| `scripts/train.sh` | `swift sft` full fine-tune on 4 GPUs, restart-safe |
| `python -m ocr_train.predict` | a checkpoint's transcriptions, in the form `ocr-eval score` reads |
| `deploy/openshift/` | one Job per step, plus the workspace PVC |

---

## Evaluating a checkpoint

The evaluation harness (`ocr-eval`) scores predictions that already exist on
disk. It never loads the model, needs no GPU and needs no cluster access, so a
checkpoint can be judged while everything else is still blocked.

### Two invocation paths, same code

On a workstation, after `pip install -e .`:

```bash
ocr-eval score \
  --manifest corpus/manifest.jsonl \
  --predictions preds/ft-v003/ \
  --model-version qwen35-ocr-ft-v003-step4000 \
  --out runs/ft-v003
```

Inside a pod the package is **not** pip-installed — the Containerfile copies
`src/` to `/opt/ocr-training/src`, nothing more — so the console script does not
exist there and the invocation is:

```bash
PYTHONPATH=/opt/ocr-training/src python -m ocr_eval score ...
```

Worth stating because the difference is otherwise discovered in a pod.

### The four commands

| Command | What it does |
|---|---|
| `ocr-eval validate` | check a corpus and a prediction set line up, in seconds, before scoring or as a data pipeline gate |
| `ocr-eval score` | produce an immutable run directory: per-page results, summary, worst-pages report |
| `ocr-eval compare` | two runs to a PROMOTE / REJECT / INCONCLUSIVE recommendation |
| `ocr-eval report` | re-render a run's output without rescoring |

Thresholds live in `configs/eval/default.json` and are echoed verbatim into
every run manifest. Full walkthrough:
[`specs/001-ocr-eval-harness/quickstart.md`](specs/001-ocr-eval-harness/quickstart.md).

### Two things to know before trusting a number

**The gold set does not exist yet.** Until a set of manually verified real
scanned pages is built, every figure is measured on synthetic pages and
**overstates real-world quality**. The harness prints that caveat verbatim in
any report lacking a `gold_scans` split, which contains the problem without
solving it.

**The `[UNREADABLE]` marker is a cross-feature contract.** ⚠️ Dataset
preparation **must** emit exactly `[UNREADABLE]` as the training target for
unreadable images. The harness scores against that exact string: the marker on
an unreadable image is a pass, any other text is a hallucination, and the marker
on a legible image is a false refusal counted separately. If the training
targets use a different string — or a natural-language refusal — the
hallucination measurement is meaningless, and it will look like a *result*
rather than a bug. The expected marker is written into every run manifest as
`policy.unreadable_marker` so a mismatch is at least visible when someone goes
looking. (FR-010a; this is the decision recorded here because the work it
constrains lives outside the feature that discovered it.)

### Running the tests

```bash
pip install -e ".[test]"
pytest
```

The suite runs offline against committed fixtures. No network, no external data,
no setup step beyond the install.

---

## Using spec-kit in this project

Spec-kit is installed and set up here. Two different things, with two different scopes:

**The `specify` CLI is global and permanent.** It lives at `C:\Users\HP\.local\bin\specify.exe`,
which is on your persistent user PATH. It works in any terminal, after any reboot. Nothing to
reinstall.

**The `/speckit-*` skills are scoped to THIS folder.** They live in `.claude/skills/` here, so
they only appear when Claude Code's working directory is `qwen35-ocr-training`. Open Claude
Code anywhere else and they will not be listed.

### To use them in a future session

Either open the folder in VS Code (**File -> Open Folder ->** `qwen35-ocr-training`) and start
Claude Code there, or from a terminal:

```bash
cd ~/Documents/ocr_followup/qwen35-ocr-training
claude
```

Then the skills are available:

| Skill | What it does |
|---|---|
| `/speckit-constitution` | project principles (run once) |
| `/speckit-specify` | write the spec |
| `/speckit-clarify` | structured questions to remove ambiguity - run before plan |
| `/speckit-plan` | technical implementation plan |
| `/speckit-tasks` | actionable task list |
| `/speckit-implement` | build it |
| `/speckit-analyze` | consistency check across spec/plan/tasks |
| `/speckit-checklist` | quality checklist for the requirements |

### To add spec-kit to a different project later

```bash
specify init <new-project-name> --integration claude
# or, inside an existing folder:
specify init --here --integration claude
```

### To update spec-kit

```bash
specify self upgrade          # or: uv tool upgrade specify-cli
```

### What to use it for here

Use it for software that does not exist yet and has real requirements - the **evaluation
harness** (CER/WER, Arabic normalisation policy, gold set, hallucination tests, per-font
breakdown), the **dataset preparation pipeline**, the **dataset validator**.

Do not use it for operational steps. Connecting to VM01, `oc get pods`, `podman build` and
staging the model are commands to run, not features to specify - those live in
`../COMMANDS.txt`.
