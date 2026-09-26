"""OCR-Data/ocr_data tar shards -> ms-swift training JSONL + an ocr-eval test corpus.

    python -m ocr_train.prepare --source s3://ocr-data/ocr_data --out /workspace/data/prepared
    python -m ocr_train.prepare --source /mnt/ocr_data          --out /workspace/data/prepared

`--source` is a directory or an s3:// prefix holding `data/shard_NNN.tar` and
`data_aug/shard_NNN_augK.tar`. MinIO is reached through boto3's own environment
variables (AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_CA_BUNDLE), so no credential ever appears on a command line.

Tars are streamed, never stored: each sample's image is written straight to
`<out>/images/<shard>/`, so the PVC holds the dataset once, not twice.

Output, under --out:

  train.jsonl                 ms-swift rows, shuffled, with [UNREADABLE] negatives mixed in
  val.jsonl                   small in-training eval set from the held-out groups
  eval/manifest.jsonl         ocr-eval corpus: held-out pages + probe pages
  prepare_report.json         counts, every dropped sample and why, image/text size stats
  images/, parts/             the extracted pages; per-shard results (for resuming)

The held-out split is drawn on shard *groups* — original shard N together with
its augmented variants — because data_aug/shard_N_augK.tar holds variants of the
pages in data/shard_N.tar. Holding out a shard file instead of a group would put
the same page in training and in evaluation.

Resumable: a shard whose `parts/<shard>.done.json` exists is not re-read. A pod
killed half way re-runs with the same command and picks up where it stopped.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import random
import statistics
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path, PurePosixPath

from ocr_eval.normalise.tables import UNREADABLE_MARKER
from ocr_train.records import (
    Shard,
    annotation_problems,
    estimate_image_tokens,
    page_id,
    parse_shard,
    training_record,
)

IMAGE_EXTS = ("png", "jpg", "jpeg")
UNREADABLE_DIR = "images/_unreadable"


# --------------------------------------------------------------------------- sources


def _s3_client():
    import boto3
    from botocore.config import Config

    # MinIO serves buckets by path, not by virtual host.
    return boto3.client("s3", config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10}))


def _split_s3(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    return bucket, prefix.rstrip("/")


def list_shards(source: str, pattern: str) -> list[tuple[Shard, str, int]]:
    """(shard, locator, size) for every shard tar under `source`, sorted by name."""
    found: list[tuple[str, str, int]] = []
    if source.startswith("s3://"):
        bucket, prefix = _split_s3(source)
        pages = _s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".tar"):
                    found.append((PurePosixPath(obj["Key"]).stem, f"s3://{bucket}/{obj['Key']}", obj["Size"]))
    else:
        root = Path(source)
        if not root.is_dir():
            raise SystemExit(f"--source {source}: not a directory and not an s3:// prefix")
        for path in root.rglob("*.tar"):
            found.append((path.stem, str(path), path.stat().st_size))

    # Every tar is checked, not only those the --shards glob selects: a stray file
    # under --source is a sign the source is not what it is believed to be.
    shards, bad = [], []
    for stem, locator, size in found:
        try:
            shard = parse_shard(stem)
        except ValueError as exc:
            bad.append(str(exc))
            continue
        if fnmatch.fnmatch(stem, pattern):
            shards.append((shard, locator, size))
    if bad:
        raise SystemExit("unrecognised tar files under --source:\n  " + "\n  ".join(sorted(bad)))
    stems = [s.stem for s, _, _ in shards]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        raise SystemExit(f"the same shard appears more than once under --source: {dupes}")
    return sorted(shards, key=lambda item: item[0].stem)


def _open(locator: str):
    if locator.startswith("s3://"):
        bucket, key = _split_s3(locator)
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"]
    return open(locator, "rb")


# --------------------------------------------------------------------------- one shard


def _check_image(data: bytes, verify: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        if verify == "full":
            image.load()  # a truncated file fails here, not 30 hours into training
        return image.size


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def process_shard(shard: Shard, locator: str, out: Path, verify: str) -> dict:
    """Extract one shard. Returns its stats; the samples go to parts/<stem>.jsonl."""
    done = out / "parts" / f"{shard.stem}.done.json"
    if done.exists():
        stats = json.loads(done.read_text(encoding="utf-8"))
        stats["resumed"] = True
        return stats

    started = time.time()
    img_dir = out / "images" / shard.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    problems: list[dict] = []
    pending: dict[str, dict] = {}

    def finish(key: str, sample: dict) -> None:
        if "image" not in sample or "json" not in sample:
            missing = "image" if "image" not in sample else "json"
            problems.append({"key": key, "problem": f"{missing} missing from shard"})
            return
        ext, data = sample["image"]
        try:
            ann = json.loads(sample["json"])
        except ValueError as exc:
            problems.append({"key": key, "problem": f"annotation is not valid JSON: {exc}"})
            return
        found = annotation_problems(ann)
        try:
            width, height = _check_image(data, verify)
        except Exception as exc:  # noqa: BLE001 - any decode failure disqualifies the page
            found.append(f"image does not decode: {type(exc).__name__}: {exc}")
        if found:
            problems.extend({"key": key, "problem": p} for p in found)
            return
        rel = f"images/{shard.stem}/{key}.{ext}"
        (out / rel).write_bytes(data)
        meta = ann.get("meta") or {}
        augmentation = meta.get("augmentation") or {}
        rows.append(json.dumps({
            "page_id": page_id(shard.stem, key),
            "image": rel,
            "target": ann["markdown"],
            "group": shard.group,
            "aug": shard.aug,
            "width": width,
            "height": height,
            "font": meta.get("page_font"),
            "template": meta.get("template"),
            "augmentation": augmentation.get("name") if isinstance(augmentation, dict) else None,
        }, ensure_ascii=False))

    with closing(_open(locator)) as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name).name  # flatten: never write outside --out
            key, _, ext = name.partition(".")
            ext = ext.lower()
            kind = "image" if ext in IMAGE_EXTS else "json" if ext == "json" else None
            if kind is None:
                problems.append({"key": key, "problem": f"unexpected member {name}"})
                continue
            # Samples are contiguous in a WebDataset tar, so a new key means the
            # previous sample is complete.
            for other in [k for k in pending if k != key]:
                finish(other, pending.pop(other))
            sample = pending.setdefault(key, {})
            if kind in sample:
                problems.append({"key": key, "problem": f"duplicate {kind} member {name}"})
                continue
            payload = tar.extractfile(member).read()
            sample[kind] = (ext, payload) if kind == "image" else payload
            if "image" in sample and "json" in sample:
                finish(key, pending.pop(key))
    for key, sample in pending.items():
        finish(key, sample)

    _write_atomic(out / "parts" / f"{shard.stem}.jsonl", "".join(r + "\n" for r in rows))
    stats = {
        "shard": shard.stem,
        "group": shard.group,
        "aug": shard.aug,
        "samples_ok": len(rows),
        "problems": problems,
        "seconds": round(time.time() - started, 1),
    }
    # Written last: its existence is what marks the shard complete.
    _write_atomic(done, json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


# --------------------------------------------------------------------------- merge


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": ordered[-1],
            "mean": round(statistics.fmean(ordered), 1)}


def _read_parts(out: Path, shards: list[Shard]) -> list[dict]:
    rows = []
    for shard in shards:
        with open(out / "parts" / f"{shard.stem}.jsonl", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows


def _unreadable_job(seed: int, name: str, out: Path, source: str | None) -> tuple[str, str]:
    from ocr_train.unreadable import make_unreadable

    kind = make_unreadable(seed, name, out / UNREADABLE_DIR / f"{name}.png",
                           out / source if source else None)
    return name, kind


def _make_negatives(pool, seed, prefix, count, sources, out) -> list[dict]:
    rng = random.Random(f"{seed}:{prefix}")
    picks = [rng.choice(sources)["image"] if sources else None for _ in range(count)]
    names = [f"{prefix}_{i:07d}" for i in range(count)]
    futures = [pool.submit(_unreadable_job, seed, n, out, s) for n, s in zip(names, picks)]
    kinds = dict(f.result() for f in futures)
    return [{"page_id": n, "image": f"{UNREADABLE_DIR}/{n}.png", "target": UNREADABLE_MARKER,
             "kind": kinds[n]} for n in names]


def _write_jsonl(path: Path, records) -> int:
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _manifest_row(row: dict) -> dict:
    return {
        "page_id": row["page_id"],
        "split": "synthetic_heldout",
        "kind": "synthetic",
        "reference_text": row["target"],
        "image_path": row["image"],
        "fonts": [row["font"]] if row.get("font") else [],
        "distortions": [row["augmentation"]] if row.get("augmentation") else [],
        "source": row.get("template") or "ocr_data",
        "is_augmented": row["aug"] > 0,
    }


def _probe_row(neg: dict) -> dict:
    return {
        "page_id": f"probe_{neg['kind']}_{neg['page_id'].rsplit('_', 1)[1]}",
        "split": "probe",
        "kind": "probe",
        "reference_text": UNREADABLE_MARKER,
        "image_path": neg["image"],
        "fonts": [],
        "distortions": [],
        "source": f"synthetic_{neg['kind']}",
    }


# --------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="directory or s3://bucket/prefix holding the shard tars")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image-root", default=None,
                    help="absolute path the training pod mounts --out at (default: --out resolved)")
    ap.add_argument("--shards", default="shard_*", help="glob on shard names, e.g. 'shard_00[1-2]*' for a smoke run")
    ap.add_argument("--holdout-groups", default="103", help="comma-separated shard group numbers kept out of training")
    ap.add_argument("--val-size", type=int, default=1000, help="held-out pages in val.jsonl (in-training eval)")
    ap.add_argument("--test-size", type=int, default=4000, help="held-out pages in eval/manifest.jsonl")
    ap.add_argument("--probe-count", type=int, default=200, help="[UNREADABLE] probe pages in eval/manifest.jsonl")
    ap.add_argument("--unreadable-fraction", type=float, default=0.005,
                    help="[UNREADABLE] negatives added to train/val, as a fraction of their size")
    ap.add_argument("--image-max-tokens", type=int, default=1280,
                    help="IMAGE_MAX_TOKEN_NUM the run will use; only feeds the size report")
    ap.add_argument("--verify", choices=("full", "header"), default="full",
                    help="full decodes every image (catches truncation); header reads the size only")
    ap.add_argument("--max-drop-fraction", type=float, default=0.001,
                    help="fail if more than this fraction of samples is unusable")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    image_root = Path(args.image_root) if args.image_root else out.resolve()
    if not image_root.is_absolute():
        raise SystemExit(f"--image-root must be absolute, got {image_root}")
    holdout = {int(g) for g in args.holdout_groups.split(",") if g.strip()}

    listing = list_shards(args.source, args.shards)
    if not listing:
        raise SystemExit(f"no shard tars matching {args.shards!r} under {args.source}")
    groups = {s.group for s, _, _ in listing}
    if not holdout or not holdout <= groups or holdout == groups:
        raise SystemExit(f"--holdout-groups {sorted(holdout)} must be a non-empty proper subset of the "
                         f"shard groups present ({min(groups)}..{max(groups)}, {len(groups)} groups)")
    print(f"{len(listing)} shards, {sum(size for _, _, size in listing) / 1e9:.1f} GB, "
          f"{len(groups)} groups; holding out {sorted(holdout)}", flush=True)

    all_stats, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_shard, s, loc, out, args.verify): s for s, loc, _ in listing}
        for i, future in enumerate(as_completed(futures), 1):
            shard = futures[future]
            try:
                stats = future.result()
            except Exception as exc:  # noqa: BLE001 - reported with the others below
                failures.append(f"{shard.stem}: {type(exc).__name__}: {exc}")
                print(f"[{i}/{len(listing)}] {shard.stem} FAILED: {exc}", flush=True)
                continue
            all_stats.append(stats)
            timing = " (already done)" if stats.get("resumed") else ", %ss" % stats["seconds"]
            print(f"[{i}/{len(listing)}] {shard.stem}: {stats['samples_ok']} ok, "
                  f"{len(stats['problems'])} problems{timing}", flush=True)
        if failures:
            print("\nPREPARE FAILED - these shards could not be read (re-run to retry only them):", file=sys.stderr)
            for line in failures:
                print(f"  {line}", file=sys.stderr)
            return 1

        shards = [s for s, _, _ in listing]
        rows = _read_parts(out, shards)
        n_problems = sum(len(s["problems"]) for s in all_stats)
        drop_fraction = n_problems / max(1, n_problems + len(rows))
        if drop_fraction > args.max_drop_fraction:
            print(f"\nPREPARE FAILED - {n_problems} unusable samples ({drop_fraction:.4%}) exceeds "
                  f"--max-drop-fraction {args.max_drop_fraction:.4%}; see parts/*.done.json", file=sys.stderr)
            return 1

        train = [r for r in rows if r["group"] not in holdout]
        heldout = sorted((r for r in rows if r["group"] in holdout), key=lambda r: r["page_id"])
        if len(heldout) < args.val_size + args.test_size:
            print(f"\nPREPARE FAILED - held-out groups have {len(heldout)} pages, fewer than "
                  f"--val-size {args.val_size} + --test-size {args.test_size}", file=sys.stderr)
            return 1
        random.Random(f"{args.seed}:heldout").shuffle(heldout)
        val, test = heldout[: args.val_size], heldout[args.val_size: args.val_size + args.test_size]

        print("generating [UNREADABLE] negatives ...", flush=True)
        n_train_neg = round(len(train) * args.unreadable_fraction)
        n_val_neg = round(len(val) * args.unreadable_fraction)
        train_neg = _make_negatives(pool, args.seed, "unreadable_train", n_train_neg, train, out)
        val_neg = _make_negatives(pool, args.seed, "unreadable_val", n_val_neg, heldout, out)
        probes = _make_negatives(pool, args.seed, "unreadable_probe", args.probe_count, heldout, out)

    def swift_rows(items):
        return (training_record(str(image_root / r["image"]), r["target"]) for r in items)

    train_all = train + train_neg
    random.Random(f"{args.seed}:train").shuffle(train_all)
    n_train = _write_jsonl(out / "train.jsonl", swift_rows(train_all))
    val_all = val + val_neg
    random.Random(f"{args.seed}:val").shuffle(val_all)
    n_val = _write_jsonl(out / "val.jsonl", swift_rows(val_all))
    manifest = sorted([_manifest_row(r) for r in test] + [_probe_row(p) for p in probes],
                      key=lambda m: m["page_id"])
    n_manifest = _write_jsonl(out / "eval" / "manifest.jsonl", manifest)

    fingerprint = hashlib.sha256()
    for shard, _, size in listing:
        fingerprint.update(f"{shard.stem}:{size}\n".encode())
    fingerprint.update(json.dumps({k: str(v) for k, v in sorted(vars(args).items())
                                   if k not in ("out", "workers", "source", "image_root")}).encode())
    trained = train + val
    report = {
        "dataset_fingerprint": "sha256:" + fingerprint.hexdigest(),
        "source": args.source,
        "image_root": str(image_root),
        "settings": {k: v for k, v in vars(args).items() if k not in ("out", "source")} | {"out": str(out)},
        "shards": len(listing),
        "holdout_groups": sorted(holdout),
        "samples_ok": len(rows),
        "samples_dropped": n_problems,
        "dropped_by_reason": _count_reasons(all_stats),
        "train_rows": n_train,
        "train_unreadable": n_train_neg,
        "val_rows": n_val,
        "val_unreadable": n_val_neg,
        "eval_manifest_rows": n_manifest,
        "eval_probe_rows": len(probes),
        "heldout_pages_unused": len(heldout) - len(val) - len(test),
        "originals_in_train": sum(1 for r in train if r["aug"] == 0),
        "augmented_in_train": sum(1 for r in train if r["aug"] > 0),
        "target_chars": _percentiles([len(r["target"]) for r in trained]),
        "image_tokens_estimate": _percentiles(
            [estimate_image_tokens(r["width"], r["height"], args.image_max_tokens) for r in trained]),
        "pages_downscaled_by_image_budget": sum(
            1 for r in trained
            if estimate_image_tokens(r["width"], r["height"], 10**9) > args.image_max_tokens),
        "problems": {s["shard"]: s["problems"] for s in sorted(all_stats, key=lambda s: s["shard"])
                     if s["problems"]},
    }
    _write_atomic(out / "prepare_report.json", json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\ntrain.jsonl {n_train} rows ({n_train_neg} [UNREADABLE]), val.jsonl {n_val} rows, "
          f"eval/manifest.jsonl {n_manifest} rows ({len(probes)} probes); "
          f"{n_problems} samples dropped - see {out / 'prepare_report.json'}")
    return 0


def _count_reasons(all_stats: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for stats in all_stats:
        for problem in stats["problems"]:
            reason = problem["problem"].split(":")[0]
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    sys.exit(main())
