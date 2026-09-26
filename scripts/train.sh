#!/usr/bin/env bash
# Full fine-tune of Qwen3.5-9B-Base on the prepared OCR data, with ms-swift.
#
# Run inside the GPU pod (deploy/openshift/train-job.yaml does exactly this):
#     export MODEL_PATH=/workspace/models/Qwen3.5-9B-Base
#     export DATA_DIR=/workspace/data/prepared            # output of `python -m ocr_train.prepare`
#     export OUTPUT_DIR=/workspace/checkpoints/qwen35-9b-ocr-v001
#     export IMAGE_REF=<registry>/<org>/qwen35-ocr-train@sha256:...   # recorded, constitution II
#     ./scripts/train.sh
#
# Restart-safe: if OUTPUT_DIR already holds a complete checkpoint, training
# resumes from the newest one (model, optimizer, scheduler, data position). A
# Job that is evicted and restarted therefore continues rather than starting over.
#
# Every setting below is an environment variable with a default, so a run is
# described entirely by its env block - which is written to OUTPUT_DIR/run_starts.jsonl
# next to ms-swift's own args.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the staged Qwen3.5-9B-Base directory}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the prepare output (holds train.jsonl, val.jsonl)}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the checkpoint directory of this run}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"        # ms-swift launches torchrun with this many ranks
# Qwen3.5 sees a page as one token per 32x32 px. 1280 tokens (~1.3 MP) keeps the
# dataset's 96-dpi pages at native resolution (p99 ~1050 tokens); predict.py must
# use the same value.
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1280}"
TUNER_TYPE="${TUNER_TYPE:-full}"
# ZeRO-2 fits a 9B full fine-tune on 4x H200 (~55 GB/GPU of weights, grads and
# optimizer state) and is faster than ZeRO-3. zero3 if memory ever runs out;
# empty to run without DeepSpeed (CPU smoke test only).
DEEPSPEED="${DEEPSPEED-zero2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
EPOCHS="${EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"                        # 4 GPUs x 4 x 8 = 128 pages per optimizer step
LR="${LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
MAX_LENGTH="${MAX_LENGTH:-4096}"                     # longest prepared sample measured: 2270 tokens
FREEZE_VIT="${FREEZE_VIT:-true}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"           # ~130 GB each with optimizer state
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"          # >= 1: ms-swift keeps workers persistent
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
MAX_STEPS="${MAX_STEPS:-}"                           # set for a smoke run; empty = full epochs
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Air-gapped: an attempted download must fail at once, not hang (constitution III).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export USE_HF=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for path in "${MODEL_PATH}/config.json" "${DATA_DIR}/train.jsonl" "${DATA_DIR}/val.jsonl" \
            "${DATA_DIR}/prepare_report.json"; do
    [[ -f "$path" ]] || { echo "ERROR: missing ${path}" >&2; exit 1; }
done

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    MODEL_PATH="$MODEL_PATH" TRAIN_DATA="${DATA_DIR}/train.jsonl" VAL_DATA="${DATA_DIR}/val.jsonl" \
        python "${REPO_ROOT}/scripts/preflight.py" --gpus "${NPROC_PER_NODE}"
fi

mkdir -p "$OUTPUT_DIR"

# Newest checkpoint that finished saving. trainer_state.json is written after the
# weights and optimizer state, so a checkpoint cut off mid-save is skipped.
RESUME=""
if [[ "${NO_RESUME:-0}" != "1" ]]; then
    while read -r ckpt; do
        if [[ -f "${ckpt}/trainer_state.json" ]]; then RESUME="$ckpt"; break; fi
    done < <(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n -r)
fi

export TUNER_TYPE DEEPSPEED TORCH_DTYPE ATTN_IMPL EPOCHS PER_DEVICE_BATCH GRAD_ACCUM LR WARMUP_RATIO \
       MAX_LENGTH FREEZE_VIT MAX_STEPS EXTRA_ARGS
python - "$OUTPUT_DIR" "$DATA_DIR" "$RESUME" <<'PY'
import json, os, sys, time
out, data, resume = sys.argv[1:4]
keys = ("IMAGE_REF", "MODEL_PATH", "NPROC_PER_NODE", "IMAGE_MAX_TOKEN_NUM", "TUNER_TYPE", "DEEPSPEED",
        "TORCH_DTYPE", "ATTN_IMPL", "EPOCHS", "PER_DEVICE_BATCH", "GRAD_ACCUM", "LR", "WARMUP_RATIO",
        "MAX_LENGTH", "FREEZE_VIT", "MAX_STEPS", "EXTRA_ARGS")
report = json.load(open(os.path.join(data, "prepare_report.json"), encoding="utf-8"))
entry = {
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "resume_from": resume or None,
    "dataset_fingerprint": report["dataset_fingerprint"],
    "train_rows": report["train_rows"],
    "env": {k: os.environ.get(k, "unset") for k in keys},
}
with open(os.path.join(out, "run_starts.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry) + "\n")
print(json.dumps(entry, indent=1))
PY

args=(
    --model "$MODEL_PATH" --model_type qwen3_5 --template qwen3_5
    --tuner_type "$TUNER_TYPE" --torch_dtype "$TORCH_DTYPE" --attn_impl "$ATTN_IMPL"
    --freeze_vit "$FREEZE_VIT" --freeze_aligner false --freeze_llm false
    --dataset "${DATA_DIR}/train.jsonl" --val_dataset "${DATA_DIR}/val.jsonl" --split_dataset_ratio 0
    # A page longer than max_length is dropped, never cut: a truncated target
    # teaches the model to stop mid-page.
    --max_length "$MAX_LENGTH" --truncation_strategy delete
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_BATCH" --per_device_eval_batch_size "$PER_DEVICE_BATCH"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR" --lr_scheduler_type cosine --warmup_ratio "$WARMUP_RATIO"
    --weight_decay 0.1 --max_grad_norm 1.0
    --gradient_checkpointing true
    --dataloader_num_workers "$DATALOADER_WORKERS" --dataset_num_proc "$DATASET_NUM_PROC"
    --load_from_cache_file true
    --eval_strategy steps --eval_steps "$EVAL_STEPS"
    --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT"
    --logging_steps "$LOGGING_STEPS" --report_to tensorboard
    --output_dir "$OUTPUT_DIR" --add_version false
    --seed 42 --data_seed 42
)
[[ -n "$DEEPSPEED" ]] && args+=(--deepspeed "$DEEPSPEED")
[[ -n "$MAX_STEPS" ]] && args+=(--max_steps "$MAX_STEPS")
[[ -n "$RESUME" ]] && args+=(--resume_from_checkpoint "$RESUME")
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

echo "==> swift sft ${args[*]}"
exec swift sft "${args[@]}"
