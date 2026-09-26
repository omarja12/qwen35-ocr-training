# Training Qwen3.5-9B on the 4.1M-page OCR dataset

End to end, from Hugging Face to a scored checkpoint, on the air-gapped
OpenShift cluster (4x H200). Every step is a script or a Job in this repository;
nothing is typed into a pod by hand.

| Step | Where | What | Code |
|---|---|---|---|
| 0 | machine with internet | dataset + weights -> MinIO, SHA-256 verified | `scripts/object_store.py hf-to-s3` |
| 1 | VM01 | build and push the image | `scripts/build_and_push.sh` |
| 2 | cluster, CPU | weights MinIO -> PVC | `deploy/openshift/stage-model-job.yaml` |
| 3 | cluster, CPU | shards -> `train.jsonl`, `val.jsonl`, eval corpus | `deploy/openshift/prepare-job.yaml` (`ocr_train.prepare`) |
| 4 | cluster, 4 GPU | smoke run, 50 steps | `deploy/openshift/train-job.yaml` with `MAX_STEPS=50` |
| 5 | cluster, 4 GPU | full fine-tune | `deploy/openshift/train-job.yaml` (`scripts/train.sh`) |
| 6 | cluster, 4 GPU | predict + score, base model and checkpoint | `deploy/openshift/predict-job.yaml` (`ocr_train.predict`, `ocr-eval`) |
| 7 | any CPU | promote or reject | `ocr-eval compare` |

## The numbers this plan is built on

Measured on this repository's data, not assumed:

| | |
|---|---|
| Dataset | `OCR-Data/ocr_data`: 103 original shards (116 GB, PNG) + 309 augmented shards (312 GB, JPEG) = ~4.1M pages. Augmented shard `N_augK` holds variants of original shard `N`'s pages. |
| Sequence length (real `qwen3_5` template, 1,020 pages) | mean 869 tokens, p50 932, p99 1,894, max 2,270 |
| of which image tokens | mean 614, p99 1,036, max 1,092 (budget 1,280: no page downscaled) |
| of which trained tokens | mean 204 (transcription + end marker; prompt and image are masked) |
| Tokens per epoch | ~4.07M pages x 869 = ~3.5B |
| Optimizer steps per epoch | ~4.07M / 128 = ~31,800 |

Compute estimate, to be replaced by the smoke run's measured step time:
6 x 9B x 3.5B tokens = ~1.9e20 FLOPs; at 30-40 % utilisation of 4x H200
(~1.2-1.6 PFLOP/s) that is roughly **35-45 hours per epoch**. The pure-PyTorch
fallback for Qwen3.5's linear-attention layers (no `flash-linear-attention` in
the image) can make it several times longer - which is why step 4 measures it.

## Step 0 - mirror to MinIO (outside the cluster)

On a machine that reaches both Hugging Face and MinIO:

```bash
pip install huggingface_hub boto3
export HF_TOKEN=...                      # read access to the gated dataset; never in a file or a command line
export AWS_ENDPOINT_URL=https://<minio>  AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...

python scripts/object_store.py hf-to-s3 --repo OCR-Data/ocr_data --repo-type dataset \
    --dest s3://<data-bucket>/ocr_data --include 'data/*.tar' 'data_aug/*.tar'
python scripts/object_store.py hf-to-s3 --repo Qwen/Qwen3.5-9B-Base --dest s3://<model-bucket>/Qwen3.5-9B-Base
```

428 GB + 19 GB. Each file's bytes are hashed while streaming and compared to
Hugging Face's published SHA-256 before the object is put in place; the hash is
stored on the object and re-checked when the model is staged in step 2. Re-running
skips everything already mirrored.

## Step 1 - the image

As in the main README (`scripts/build_and_push.sh v002`). Record the digest it
prints: every Job below takes it as `IMAGE_REF`. `requirements.txt` pins
`ms-swift==4.5.3` and `transformers==5.16.1`, the versions this pipeline was
verified with.

## Step 2 - cluster setup and model staging

```bash
oc apply -f deploy/openshift/pvc.yaml           # set storageClassName first
# credentials from a file outside the repo, not typed into the shell history:
oc create secret generic minio-credentials --from-env-file=<path-to-minio.env>
#   minio.env holds AWS_ENDPOINT_URL=..., AWS_ACCESS_KEY_ID=..., AWS_SECRET_ACCESS_KEY=...
oc apply -f deploy/openshift/stage-model-job.yaml
```

Every `REPLACE_*` in the Job files must be filled in: registry, organisation,
tag, digest, bucket names.

## Step 3 - prepare the data

```bash
oc apply -f deploy/openshift/prepare-job.yaml && oc logs -f job/ocr-prepare
```

`python -m ocr_train.prepare` streams every shard from MinIO (the tars are never
stored), decodes every image, and writes under `/workspace/data/prepared`:

- `train.jsonl` - ms-swift rows, every group except 103, shuffled, with 0.5 %
  `[UNREADABLE]` negatives mixed in (blank, noise and obliterated pages; see
  `src/ocr_train/unreadable.py`). The dataset has no unreadable pages, and without
  them the model would learn to invent text for a blank page.
- `val.jsonl` - 1,000 held-out pages for the loss curve during training.
- `eval/manifest.jsonl` - 4,000 held-out pages (originals and augmented, tagged)
  plus 200 `[UNREADABLE]` probes, in `ocr-eval`'s manifest format.
- `prepare_report.json` - the dataset fingerprint, every dropped sample and why,
  and size statistics. **Read it before training.** The job fails on its own if
  more than 0.1 % of samples are unusable.

Group 103 (original shard 103 and its three augmented variants) is held out as a
whole, so no page, nor any augmented copy of it, is both trained on and
evaluated.

The prompt the model is trained and evaluated with is in `src/ocr_train/records.py`.

## Step 4 - smoke run (do not skip)

In `train-job.yaml`, uncomment `MAX_STEPS=50`, set `SAVE_STEPS=25` and a
throwaway `OUTPUT_DIR`, apply, and check:

1. `preflight passed`, 4 GPUs listed; note whether the optional kernels are present.
2. Loss is logged every 10 steps and falls; no NaN.
3. `checkpoint-25` and `checkpoint-50` exist.
4. Delete the pod mid-run: the Job's retry logs `resume_from` in `run_starts.jsonl`
   and continues from the checkpoint.
5. `train_speed(s/it)` in the log x 31,800 steps = the full run's wall time.
   Peak memory per GPU is in `nvidia-smi`; if it is near 141 GB, set `DEEPSPEED=zero3`.

If the measured time is too long, the options in order of effort:
`flash-linear-attention` in the image (Triton, no compile), then
`causal-conv1d` + `flash-attn` with `--padding_free true` (EXTRA_ARGS) to stop
paying for padding.

## Step 5 - the full run

Remove `MAX_STEPS`, set a fresh `OUTPUT_DIR` and Job name (`...-v001`), apply.

- Progress: `oc logs -f`, `OUTPUT_DIR/logging.jsonl`, TensorBoard files in `OUTPUT_DIR/runs`.
- Checkpoints every 1,000 steps (~70-80 min at the estimate above), the newest 2 kept.
- Eviction or crash: the Job retries up to 3 times, each resuming from the
  newest complete checkpoint. `run_starts.jsonl` records every start.
- Settings: `OUTPUT_DIR/args.json` (ms-swift's full argument set) and
  `run_starts.jsonl` (image digest, dataset fingerprint, env).

## Step 6 - predict and score

Twice, with `predict-job.yaml`:

1. `MODEL=/workspace/models/Qwen3.5-9B-Base`, `NAME=qwen35-9b-base` - the baseline.
2. `MODEL=.../checkpoint-<N>`, `NAME=qwen35-9b-ocr-v001-step<N>`.

Each writes `/workspace/predictions/<NAME>/<page_id>.txt` and scores them into
`/workspace/runs/<NAME>`.

## Step 7 - decide

```bash
ocr-eval compare --baseline runs/qwen35-9b-base --candidate runs/qwen35-9b-ocr-v001-step<N>
```

The headline is the strict CER on `synthetic_heldout`, reported separately from
the probe set's hallucination rate. Until a gold set of real scans exists, every
number here is on synthetic pages and overstates real-world quality (the report
says so itself).

## Why these settings

| Choice | Reason | Change it when |
|---|---|---|
| Full fine-tune, not LoRA | 4.1M examples is enough data to use the capacity; LoRA saves ~1/3 of compute, not most of it | memory or storage forbids it |
| Vision tower frozen (`FREEZE_VIT=true`), aligner + LLM trained | the pretrained ViT already sees Arabic script; training it too costs memory and risks forgetting | the eval shows glyph-level misreads that persist after training |
| DeepSpeed ZeRO-2 | a 9B full fine-tune fits in ~55 GB/GPU of state on 141 GB H200s, and ZeRO-2 is faster than ZeRO-3 | GPU memory runs out -> `zero3` |
| lr 1e-5, cosine, 2 % warmup, global batch 128, 1 epoch | standard for full fine-tuning a 9B model; one epoch already shows every page 4 times (original + 3 augmentations) | the val loss is still falling steeply at the end |
| `IMAGE_MAX_TOKEN_NUM=1280` | largest page measured is 1,092 tokens, so pages keep native resolution; small Arabic diacritics need it | pages get larger (higher dpi) |
| `max_length 4096`, `truncation_strategy delete` | 1.8x the longest sample measured; an over-long page is dropped rather than cut, because a truncated target teaches stopping mid-page | never lower it below the report's p99 |
| no packing / padding-free in v1 | both need `flash-attn`, a compiled kernel kept out until the plain image trains | after step 4, for speed |
| 0.5 % `[UNREADABLE]` negatives | teaches refusal without teaching it on legible pages; the probe set measures both | the eval shows false refusals (lower) or hallucination (higher) |

## Testing the pipeline without a GPU

`tests/unit/test_ocr_train.py` and `tests/integration/test_prepare.py` run in
the normal offline test suite.

The full chain was also run on a 4-core, 16 GB CPU VM (26 Sep 2026) with
`Qwen/Qwen3.5-0.8B-Base` - same architecture, processor and `qwen3_5` template as
the 9B - on 2,000 real pages rebuilt into mini shards, with the network blocked
(proxy pointed at a dead port, `HF_HUB_OFFLINE=1`), using
`EXTRA_ARGS="--use_cpu true" DEEPSPEED="" NPROC_PER_NODE=1`:

| Stage | Result |
|---|---|
| `ocr_train.prepare` | 2,000 pages, 0 dropped; held-out group never in `train.jsonl`; manifest accepted by `ocr-eval` |
| template check | prompt and image tokens masked; trained span is `<think>\n\n</think>\n\n` + transcription + `<|im_end|>`; negatives train on exactly `[UNREADABLE]` |
| `scripts/train.sh`, full fine-tune | loss logged, eval ran, `checkpoint-2` saved with optimizer state |
| eviction + resume | process tree killed after `checkpoint-2`; the identical command resumed at step 3 with the learning-rate schedule continuing (1e-5, 7.5e-6 -> 2.5e-6) and saved `checkpoint-4` |
| `ocr_train.predict` | base model and `checkpoint-4`, 22 pages each; re-run skips finished pages; changed settings refused |
| `ocr-eval score` + `compare` | both runs scored; compare produced a verdict |
| S3 paths (local S3 server standing in for MinIO) | `prepare --source s3://...` byte-identical to the local-disk run, same fingerprint; `hf-to-s3` of a real HF repo with LFS weights, hashes stored; `s3-to-dir` restored them bit-exact; both re-runs skipped every file; an object altered after mirroring was refused (exit 1, nothing left at the target) |

Not covered by that run, and what step 4 above must show on the cluster:
DeepSpeed (CPU runs without it), multi-GPU, the 9B model's memory and speed.
Resume was exercised with only the aligner trainable (`--freeze_llm true`):
reloading a full 0.8B optimizer state exceeds the VM's 14 GB memory limit, a
constraint the 256 GiB training pod does not have. The CER from that run (~0.93,
4 steps, 48-token outputs) says nothing about quality.

One thing the run caught: generation returns the template's empty-thinking
prefix in front of every answer. `ocr_train.predict` removes exactly that string
and nothing else; without it every correct `[UNREADABLE]` would have been scored
as a hallucination.
