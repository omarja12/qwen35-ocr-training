"""Pull the first N samples of one OCR-Data/ocr_data shard, not the whole ~1 GB tar.

The tar is streamed and reading stops after N samples, so only about N x 150 KB
is downloaded. The dataset is gated, so a Hugging Face token is needed:

    $env:HF_TOKEN = "hf_..."
    python fetch_hf_samples.py --n 1000

Output: data/<name>.png + <name>.json next to this script (git-ignored:
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--n", type=int, default=1000, help="samples to keep")
parser.add_argument("--shard", default="data/shard_001.tar")
parser.add_argument("--out", type=Path, default=HERE / "data")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

seen: set[str] = set()
path = f"datasets/OCR-Data/ocr_data/{args.shard}"
with HfFileSystem().open(path, "rb") as remote, tarfile.open(fileobj=remote, mode="r|") as tar:
    for member in tar:
        if not member.isfile():
            continue
        name = Path(member.name).name  # flatten: never write outside --out
        key = name.split(".", 1)[0]
        if key not in seen:
            if len(seen) == args.n:
                break
            seen.add(key)
            if len(seen) % 100 == 0:
                print(f"{len(seen)} samples", flush=True)
        (args.out / name).write_bytes(tar.extractfile(member).read())

size = sum(f.stat().st_size for f in args.out.iterdir())
print(f"done: {len(seen)} samples, {size / 1e6:.0f} MB in {args.out}")
