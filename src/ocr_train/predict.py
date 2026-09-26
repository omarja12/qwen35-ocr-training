"""A checkpoint's transcriptions of an ocr-eval corpus, one `<page_id>.txt` per page.

    python -m ocr_train.predict \
        --model /workspace/checkpoints/qwen35-9b-ocr-v001/checkpoint-32000 \
        --manifest /workspace/data/prepared/eval/manifest.jsonl \
        --corpus-root /workspace/data/prepared \
        --out /workspace/predictions/v001-step32000

    ocr-eval score --manifest .../eval/manifest.jsonl --predictions /workspace/predictions/v001-step32000 \
        --model-version qwen35-9b-ocr-v001-step32000 --out runs/v001-step32000

Run it on the untouched base model too: a checkpoint is promoted against the
untrained baseline, never against an absolute number (constitution I).

One process per GPU splits the corpus: `CUDA_VISIBLE_DEVICES=$i ... --num-shards 4
--shard-index $i`, all writing into the same --out. Pages whose file already
exists are skipped, so a killed job re-runs with the same command.

Prompt, chat template and image budget are the training ones: the prompt comes
from `ocr_train.records`, the template from the checkpoint via ms-swift, and
--image-max-tokens must equal the IMAGE_MAX_TOKEN_NUM the run was trained with.
Output is written as generated, with one exception: the qwen3_5 template's
empty-thinking prefix (`<think>\n\n</think>\n\n`). Training puts it in front of
every target and inference emits it in front of every answer; it is template
framing, not transcription. Left in, it would add characters to every page's
CER and turn every correct `[UNREADABLE]` into a hallucination. Only that exact
string is removed. A think block with anything in it, or any other deviation, is
kept - the harness's job is to see what the model did (constitution IV).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ocr_train.records import PROMPT


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="checkpoint directory or base model directory")
    ap.add_argument("--manifest", required=True, type=Path, help="ocr-eval corpus manifest")
    ap.add_argument("--corpus-root", required=True, type=Path, help="image_path in the manifest is relative to this")
    ap.add_argument("--out", required=True, type=Path, help="directory of <page_id>.txt (ocr-eval's directory form)")
    ap.add_argument("--image-max-tokens", type=int, default=1280, help="must match training's IMAGE_MAX_TOKEN_NUM")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--splits", default=None, help="comma-separated manifest splits to predict (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N pages of this shard only (smoke test)")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return ap.parse_args(argv)


def strip_prefix(text: str, prefix: str) -> str:
    """Remove the template's exact non-thinking prefix; leave anything else alone."""
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


def load_pages(manifest: Path, splits: set[str] | None) -> list[dict]:
    pages, problems = [], []
    with open(manifest, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if splits and row.get("split") not in splits:
                continue
            if not row.get("image_path"):
                problems.append(f"line {n}: page {row.get('page_id')} has no image_path")
                continue
            pages.append(row)
    if problems:
        raise SystemExit("manifest rows cannot be predicted:\n  " + "\n  ".join(problems))
    return sorted(pages, key=lambda r: r["page_id"])


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    # Read by qwen_vl_utils when ms-swift builds the processor, so it must be set
    # before the import below. A different value from training changes how many
    # tokens every page is shown with.
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_tokens)

    splits = {s for s in args.splits.split(",")} if args.splits else None
    pages = load_pages(args.manifest, splits)[args.shard_index::args.num_shards]
    if args.limit is not None:
        pages = pages[: args.limit]
    missing = [p["page_id"] for p in pages if not (args.corpus_root / p["image_path"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing under {args.corpus_root}, e.g. {missing[:5]}")
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pages if not (args.out / f"{p['page_id']}.txt").exists()]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(pages)} pages, "
          f"{len(pages) - len(todo)} already done, {len(todo)} to predict", flush=True)

    meta = {
        "model": str(args.model),
        "manifest": str(args.manifest),
        "prompt": PROMPT,
        "image_max_tokens": args.image_max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "dtype": args.dtype,
        "removed_prefix": "template non_thinking_prefix, exact match only",
        "image_ref": os.environ.get("IMAGE_REF", "unset"),
    }
    meta_path = args.out / "predict_meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = {k for k in meta if k != "image_ref" and previous.get(k) != meta[k]}
        if changed:
            raise SystemExit(f"{args.out} holds predictions made with different settings ({sorted(changed)}); "
                             "use a new --out rather than mixing them")
    elif args.shard_index == 0:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if not todo:
        return 0

    import torch
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    engine = TransformersEngine(args.model, model_type="qwen3_5", torch_dtype=getattr(torch, args.dtype),
                                max_batch_size=args.batch_size)
    config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    prefix = engine.template.template_meta.non_thinking_prefix or ""
    started, done = time.time(), 0
    chunk = max(args.batch_size * 4, 1)
    for i in range(0, len(todo), chunk):
        batch = todo[i: i + chunk]
        requests = [InferRequest(messages=[{"role": "user", "content": "<image>" + PROMPT}],
                                 images=[str(args.corpus_root / p["image_path"])]) for p in batch]
        responses = engine.infer(requests, config, use_tqdm=False)
        for page, response in zip(batch, responses):
            target = args.out / f"{page['page_id']}.txt"
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(strip_prefix(response.choices[0].message.content, prefix), encoding="utf-8")
            tmp.replace(target)
        done += len(batch)
        rate = done / (time.time() - started)
        print(f"{done}/{len(todo)} pages, {rate:.2f} pages/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
