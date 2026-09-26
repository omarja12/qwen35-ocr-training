# qwen35-ocr-training

Fine-tunes **Qwen3.5-9B-Base** to transcribe Arabic document images, on the
~4.1M-page `OCR-Data/ocr_data` dataset, on **one server with 4 GPUs (H200) and
no internet access**. Then scores each checkpoint against the untrained base
model with the evaluation harness in this repository.

This README is the how-to. [`docs/TRAINING.md`](docs/TRAINING.md) has the
reasoning behind every setting, the measured numbers and what was tested.

---

## Running it on an air-gapped server

An air-gapped server cannot download anything, so the work splits in two:

```
  OUTSIDE (a machine with internet)             INSIDE (the GPU server, no internet)
  ---------------------------------             ------------------------------------
  1. download dataset + model weights  ──┐
  2. build the software image          ──┼──>   3. load image, place data + weights
                                         │      4. prepare the data        (CPU, ~1-2 h)
     carried across by MinIO, or by      │      5. smoke run, 50 steps     (4 GPU, <1 h)
     disk / USB / approved transfer   ───┘      6. full training run       (4 GPU, ~2 days, measured in 5)
                                                7. predict + score + compare
```

Nothing inside ever reaches the internet: offline mode is forced
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and every package is already in
the image. This whole chain was run end to end with the network blocked.

There are two ways to run the inside half. Pick the one that matches your server:

| | **A. OpenShift / Kubernetes cluster** | **B. A plain Linux server** |
|---|---|---|
| runs as | one Job per step (`deploy/openshift/`) | `podman run` / `docker run` commands |
| data arrives through | MinIO (S3) | MinIO **or** a copied directory |
| restart after a crash | automatic (Job retries) | re-run the same command |

### What you need

| | Size | Notes |
|---|---|---|
| 4 NVIDIA GPUs, ~141 GB each (H200) | | 80 GB GPUs need `DEEPSPEED=zero3`, see Troubleshooting |
| Server RAM | 256 GB | |
| Disk on the server | **1.5 TB** | extracted pages ~430 GB, model 19 GB, 2 checkpoints ~130 GB each; route B without MinIO also stores the raw tars (+430 GB) until step 4 finishes |
| Hugging Face token | | read access to the gated `OCR-Data/ocr_data` dataset; used **outside only** |
| A container runtime | | podman or docker with the NVIDIA container toolkit (route B), or the cluster (route A) |

---

### Step 1 - outside: download the dataset and the model

On any machine with internet and enough disk. Never write the token into a file
or a command that is saved in shell history:

```bash
pip install -U huggingface_hub boto3
read -s HF_TOKEN && export HF_TOKEN        # paste the token, press Enter
```

**If you have MinIO** (a copy of it that the server can reach), stream straight
into it. Every file is checked against Hugging Face's SHA-256 on the way:

```bash
export AWS_ENDPOINT_URL=https://<minio-host>  AWS_ACCESS_KEY_ID=<key>
read -s AWS_SECRET_ACCESS_KEY && export AWS_SECRET_ACCESS_KEY

python scripts/object_store.py hf-to-s3 --repo OCR-Data/ocr_data --repo-type dataset \
    --dest s3://<data-bucket>/ocr_data --include 'data/*.tar' 'data_aug/*.tar'
python scripts/object_store.py hf-to-s3 --repo Qwen/Qwen3.5-9B-Base \
    --dest s3://<model-bucket>/Qwen3.5-9B-Base
```

**If you move files by disk instead**, download to a directory and carry it over:

```bash
hf download OCR-Data/ocr_data --repo-type dataset --include 'data/*.tar' --include 'data_aug/*.tar' \
    --local-dir ./transfer/ocr_data          # 428 GB, 412 tar files
hf download Qwen/Qwen3.5-9B-Base --local-dir ./transfer/Qwen3.5-9B-Base     # 19 GB
```

Both commands can be re-run after an interruption; they skip finished files.

### Step 2 - outside: build the software image

The image holds code and Python packages only - no data, no weights, no
passwords. Its base must be a PyTorch + CUDA image that works with the server's
NVIDIA driver. On an OpenShift cluster, use one the cluster already runs:

```bash
oc get deployment <A_RUNNING_GPU_DEPLOYMENT> -o jsonpath='{.spec.template.spec.containers[0].image}'
```

**Route A (cluster with a registry)** - on VM01:

```bash
export BASE_IMAGE='<base image>' REGISTRY='<registry host>' REGISTRY_ORG='<org>' QUAY_USER='<robot account>'
read -s QUAY_TOKEN && export QUAY_TOKEN
./scripts/build_and_push.sh v001        # prints the image digest: note it, every run records it
```

The script refuses to build if the disk is too small, if the tag is `latest`,
or if anything that looks like a password is in the build context.

**Route B (plain server, no registry)** - build, save to a file, carry it over:

```bash
podman build --build-arg BASE_IMAGE='<base image>' -t qwen35-ocr-train:v001 -f Containerfile .
podman run --rm qwen35-ocr-train:v001 -c 'python /opt/ocr-training/scripts/preflight.py --imports-only'
podman save qwen35-ocr-train:v001 | gzip > qwen35-ocr-train-v001.tar.gz
# on the server:
podman load -i qwen35-ocr-train-v001.tar.gz
```

(`docker` works the same way with the same arguments.)

---

### Steps 3-7, route A: OpenShift cluster

Each file in `deploy/openshift/` is one step. First replace every `REPLACE_...`
value in them (registry, tag, image digest, bucket names, storage class).

```bash
# 3. storage + MinIO credentials (from a file outside the repository)
oc apply -f deploy/openshift/pvc.yaml
oc create secret generic minio-credentials --from-env-file=<path-to-minio.env>
#    minio.env contains: AWS_ENDPOINT_URL=...  AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
oc apply -f deploy/openshift/stage-model-job.yaml && oc logs -f job/ocr-stage-model

# 4. prepare the data
oc apply -f deploy/openshift/prepare-job.yaml && oc logs -f job/ocr-prepare

# 5. smoke run: in train-job.yaml uncomment MAX_STEPS=50, set SAVE_STEPS=25 and a
#    throwaway OUTPUT_DIR + Job name, then
oc apply -f deploy/openshift/train-job.yaml && oc logs -f job/qwen35-9b-ocr-v001

# 6. full run: remove MAX_STEPS, set a fresh OUTPUT_DIR + Job name, apply again

# 7. predict + score: once with MODEL=the base model, once with MODEL=a checkpoint
oc apply -f deploy/openshift/predict-job.yaml && oc logs -f job/ocr-predict-v001-step32000
```

### Steps 3-7, route B: plain server

Everything lives under one directory on the server, mounted into the container
as `/workspace`. Set this once per shell:

```bash
WS=/data/ocr-workspace                     # any disk with 1.5 TB free
IMG=qwen35-ocr-train:v001
RUN="podman run --rm --device nvidia.com/gpu=all --shm-size=64g -v $WS:/workspace --env-file $WS/run.env $IMG -c"
#    docker: replace `--device nvidia.com/gpu=all` with `--gpus all`
mkdir -p $WS/models $WS/raw $WS/tmp $WS/cache
printf 'HF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\nIMAGE_REF=%s\n' "$IMG" > $WS/run.env
```

**3. Put the data and weights in place.** If you carried them by disk:

```bash
cp -r /media/transfer/Qwen3.5-9B-Base $WS/models/
cp -r /media/transfer/ocr_data        $WS/raw/          # holds data/ and data_aug/
```

If they are in MinIO, add the MinIO variables to `$WS/run.env`
(`AWS_ENDPOINT_URL=`, `AWS_ACCESS_KEY_ID=`, `AWS_SECRET_ACCESS_KEY=`, one per
line; keep the file readable by you only: `chmod 600 $WS/run.env`) and stage the
weights:

```bash
$RUN 'python /opt/ocr-training/scripts/object_store.py s3-to-dir \
        --src s3://<model-bucket>/Qwen3.5-9B-Base --dest /workspace/models/Qwen3.5-9B-Base'
```

**4. Prepare the data** (CPU only; about 1-2 hours with 16 cores). Use
`--source /workspace/raw/ocr_data` for copied files, or
`--source s3://<data-bucket>/ocr_data` for MinIO:

```bash
$RUN 'python -m ocr_train.prepare --source /workspace/raw/ocr_data \
        --out /workspace/data/prepared --workers 16'
```

Read `$WS/data/prepared/prepare_report.json` afterwards: it lists every page
that was dropped and why. Once it looks right, `$WS/raw` can be deleted.

**5. Smoke run - 50 steps, do not skip.** It checks the 4 GPUs, memory,
saving and resuming, and measures how long the real run will take:

```bash
$RUN 'MODEL_PATH=/workspace/models/Qwen3.5-9B-Base DATA_DIR=/workspace/data/prepared \
      OUTPUT_DIR=/workspace/checkpoints/smoke MAX_STEPS=50 SAVE_STEPS=25 \
      /opt/ocr-training/scripts/train.sh' 2>&1 | tee $WS/smoke.log
```

Check: `preflight passed`, the loss goes down, `checkpoints/smoke/checkpoint-50`
exists. The log's `train_speed(s/it)` x ~31,800 steps = the full run in seconds.

**6. The full run** (days - run it inside `tmux` or `screen` so it survives
logging out):

```bash
tmux new -s train
$RUN 'MODEL_PATH=/workspace/models/Qwen3.5-9B-Base DATA_DIR=/workspace/data/prepared \
      OUTPUT_DIR=/workspace/checkpoints/qwen35-9b-ocr-v001 \
      /opt/ocr-training/scripts/train.sh' 2>&1 | tee -a $WS/train-v001.log
```

If the server reboots or the run crashes, **run exactly the same command
again**: it continues from the newest saved checkpoint (one every 1,000 steps).
Use a new `OUTPUT_DIR` only when you want a new experiment.

**7. Predict, score, compare.** Run the predict + score block for the base model
and for a checkpoint:

```bash
predict() {   # predict <model dir inside /workspace> <run name>
  $RUN "set -e; for i in 0 1 2 3; do
          CUDA_VISIBLE_DEVICES=\$i python -m ocr_train.predict --model $1 \
            --manifest /workspace/data/prepared/eval/manifest.jsonl --corpus-root /workspace/data/prepared \
            --out /workspace/predictions/$2 --num-shards 4 --shard-index \$i & done; wait
        python -m ocr_eval score --manifest /workspace/data/prepared/eval/manifest.jsonl \
          --corpus-root /workspace/data/prepared --predictions /workspace/predictions/$2 \
          --model-version $2 --out /workspace/runs/$2"
}
predict /workspace/models/Qwen3.5-9B-Base                                  qwen35-9b-base
predict /workspace/checkpoints/qwen35-9b-ocr-v001/checkpoint-32000        qwen35-9b-ocr-v001-step32000

$RUN 'python -m ocr_eval compare --baseline /workspace/runs/qwen35-9b-base \
        --candidate /workspace/runs/qwen35-9b-ocr-v001-step32000'
```

The verdict is PROMOTE, REJECT or INCONCLUSIVE. Reports (Markdown and HTML) are
in `$WS/runs/<name>/`.

---

### Where things end up

| Path under the workspace | What |
|---|---|
| `data/prepared/train.jsonl`, `val.jsonl` | training and validation rows |
| `data/prepared/eval/manifest.jsonl` | 4,000 held-out pages + 200 `[UNREADABLE]` probes |
| `data/prepared/prepare_report.json` | dataset fingerprint, dropped pages, size statistics |
| `checkpoints/<run>/checkpoint-N/` | saved model (usable by predict) + optimizer state |
| `checkpoints/<run>/logging.jsonl`, `runs/` | loss curve (TensorBoard reads `runs/`) |
| `checkpoints/<run>/args.json`, `run_starts.jsonl` | every setting, image digest, each (re)start |
| `predictions/<name>/`, `runs/<name>/` | transcriptions and their scores |

### Settings you may want to change

All are environment variables read by `scripts/train.sh` (set them in the Job's
`env:` or in front of the command). Defaults are for 4x H200.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_STEPS` | unset | stop after N steps (smoke run) |
| `EPOCHS` | 1 | passes over the data (one pass = every page 4x: original + 3 augmentations) |
| `LR` | 1e-5 | learning rate |
| `PER_DEVICE_BATCH` x `GRAD_ACCUM` | 4 x 8 | with 4 GPUs: 128 pages per optimizer step |
| `DEEPSPEED` | zero2 | `zero3` if GPU memory runs out |
| `SAVE_STEPS`, `SAVE_TOTAL_LIMIT` | 1000, 2 | checkpoint frequency and how many to keep |
| `IMAGE_MAX_TOKEN_NUM` | 1280 | image resolution budget; predict's `--image-max-tokens` must match |
| `FREEZE_VIT` | true | keep the vision encoder frozen |
| `EXTRA_ARGS` | empty | any other `swift sft` flag, passed through |

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `PREFLIGHT FAILED: device count == 4` | the container does not see the GPUs: check `--device nvidia.com/gpu=all` / `--gpus all`, or the Job's `nvidia.com/gpu: "4"` |
| `PREFLIGHT FAILED: MODEL_PATH` / `TRAIN_DATA` | the path is wrong or step 3/4 did not finish |
| CUDA out of memory | `DEEPSPEED=zero3`, or `PER_DEVICE_BATCH=2 GRAD_ACCUM=16` (same effective batch) |
| Much slower than the estimate | the image lacks `flash-linear-attention` (preflight lists it as absent): add it to `requirements.txt` and rebuild - see the note there |
| `Bus error` or dataloader crashes | shared memory too small: `--shm-size=64g` (route B) / the `dshm` volume (route A) |
| Disk full during a save | lower `SAVE_TOTAL_LIMIT`, or move old checkpoints off the disk |
| An attempted download / connection error | something tried to reach the internet: the offline variables in `run.env` / the Job are missing |
| `prepare` stops with "unusable samples" | more than 0.1 % of pages failed to decode; read `prepare_report.json`, re-copy the named shards |
| predict refuses: "different settings" | the output directory holds predictions made with other settings; use a new `--out` |

### Rules this layout exists to enforce

- No model weights, no dataset, no credentials in the image or in git.
- No `pip install` at container start: everything is installed at build time.
  Once the image has trained successfully, freeze `requirements.lock.txt`
  (`pip freeze | sort`) and build from it.
- Versioned image tags only, never `latest`; every run records the image digest.
- Optional CUDA kernels stay out until the plain image has trained once - they
  are the usual cause of failed air-gapped builds.
- The image runs as an arbitrary non-root user, so written paths are group-writable.

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

## Design documents

- `specs/001-ocr-eval-harness/` - the specification, plan and contracts of the evaluation harness.
- `.specify/memory/constitution.md` - the project rules the code and docs refer to as
  "constitution I-VI" (measurement before claims, reproducibility, air-gapped by
  construction, fail loudly, software-only image, specify software not operations).

These were written with [spec-kit](https://github.com/github/spec-kit). Its Claude Code
commands (`/speckit-*`) are not kept in the repository; to use them again, run
`specify init --here --integration claude` in a checkout.
