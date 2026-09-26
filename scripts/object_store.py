#!/usr/bin/env python3
"""Move the dataset and the base model across the air gap, through MinIO.

Two directions, run in two places:

  hf-to-s3   on a machine WITH internet: Hugging Face repo -> MinIO, streamed
  s3-to-dir  inside the cluster:         MinIO -> a directory on the PVC

    # outside (HF_TOKEN needed for the gated dataset; never on the command line):
    python scripts/object_store.py hf-to-s3 --repo OCR-Data/ocr_data --repo-type dataset \
        --dest s3://ocr-data/ocr_data --include 'data/*.tar' 'data_aug/*.tar'
    python scripts/object_store.py hf-to-s3 --repo Qwen/Qwen3.5-9B-Base --dest s3://models/Qwen3.5-9B-Base

    # inside (the dataset is streamed by `python -m ocr_train.prepare`, only weights are staged):
    python scripts/object_store.py s3-to-dir --src s3://models/Qwen3.5-9B-Base \
        --dest /workspace/models/Qwen3.5-9B-Base

MinIO is reached through boto3's standard environment: AWS_ENDPOINT_URL,
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and, for an internal CA, AWS_CA_BUNDLE.

Integrity, end to end. Hugging Face publishes a SHA-256 for every LFS file.
hf-to-s3 hashes the bytes it streams, refuses to upload anything that does not
match, and stores the hash on the object as metadata. s3-to-dir recomputes it
after download and deletes the file on a mismatch. A corrupted 5 GB weight shard
therefore fails in the staging job, not as NaNs in step 3,000 of training.

Both directions skip files already present with the same size (and hash, where
known), so an interrupted transfer re-runs with the same command.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 64 * 1024 * 1024
SHA_KEY = "sha256"  # stored as x-amz-meta-sha256


def s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10}))


def split_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SystemExit(f"expected an s3:// URI, got {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.strip("/")


class HashingReader:
    """File-like wrapper that hashes and counts what boto3 reads through it."""

    def __init__(self, raw):
        self.raw, self.sha, self.size = raw, hashlib.sha256(), 0

    def read(self, n=-1):
        data = self.raw.read(n)
        self.sha.update(data)
        self.size += len(data)
        return data


def _head(client, bucket: str, key: str) -> dict | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def hf_to_s3(args) -> int:
    from boto3.s3.transfer import TransferConfig
    from huggingface_hub import HfApi, HfFileSystem

    info = HfApi().repo_info(args.repo, repo_type=args.repo_type, revision=args.revision, files_metadata=True)
    files = [s for s in info.siblings
             if not args.include or any(fnmatch.fnmatch(s.rfilename, p) for p in args.include)]
    if not files:
        raise SystemExit(f"no files in {args.repo} match {args.include}")
    bucket, prefix = split_s3(args.dest)
    client, fs = s3_client(), HfFileSystem()
    root = f"{'datasets/' if args.repo_type == 'dataset' else ''}{args.repo}@{info.sha}"
    transfer = TransferConfig(multipart_chunksize=CHUNK, max_concurrency=4)
    print(f"{args.repo}@{info.sha}: {len(files)} files, {sum(f.size or 0 for f in files) / 1e9:.1f} GB -> {args.dest}")

    def one(sibling) -> str:
        key = f"{prefix}/{sibling.rfilename}" if prefix else sibling.rfilename
        expected = sibling.lfs.sha256 if sibling.lfs else None
        head = _head(client, bucket, key)
        if head and head["ContentLength"] == sibling.size and \
                (expected is None or head.get("Metadata", {}).get(SHA_KEY) == expected):
            return f"skip  {sibling.rfilename}"
        with fs.open(f"{root}/{sibling.rfilename}", "rb", block_size=CHUNK) as remote:
            reader = HashingReader(remote)
            # Upload to a temporary key and copy into place only once the hash is
            # known good: a half-transferred or corrupted object never sits at the
            # real key where s3-to-dir or prepare would pick it up.
            tmp_key = key + ".partial"
            client.upload_fileobj(reader, bucket, tmp_key, Config=transfer)
        digest = reader.sha.hexdigest()
        if reader.size != sibling.size or (expected and digest != expected):
            client.delete_object(Bucket=bucket, Key=tmp_key)
            raise RuntimeError(f"{sibling.rfilename}: got {reader.size} bytes sha256 {digest}, "
                               f"expected {sibling.size} bytes sha256 {expected}")
        client.copy(
            {"Bucket": bucket, "Key": tmp_key}, bucket, key,
            ExtraArgs={"Metadata": {SHA_KEY: digest}, "MetadataDirective": "REPLACE"},
            Config=transfer,
        )
        client.delete_object(Bucket=bucket, Key=tmp_key)
        return f"done  {sibling.rfilename} ({reader.size / 1e9:.2f} GB)"

    return _run_all(one, files, args.workers, lambda s: s.rfilename)


def s3_to_dir(args) -> int:
    from boto3.s3.transfer import TransferConfig

    bucket, prefix = split_s3(args.src)
    client = s3_client()
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/" if prefix else ""):
        objects.extend(o for o in page.get("Contents", []) if not o["Key"].endswith((".partial", "/")))
    if not objects:
        raise SystemExit(f"nothing under {args.src}")
    dest = Path(args.dest)
    transfer = TransferConfig(multipart_chunksize=CHUNK, max_concurrency=8)
    print(f"{args.src}: {len(objects)} objects, {sum(o['Size'] for o in objects) / 1e9:.1f} GB -> {dest}")

    def one(obj) -> str:
        rel = obj["Key"][len(prefix):].lstrip("/") if prefix else obj["Key"]
        target = dest / rel
        if not target.resolve().is_relative_to(dest.resolve()):
            raise RuntimeError(f"refusing to write outside --dest: {obj['Key']}")
        expected = _head(client, bucket, obj["Key"]).get("Metadata", {}).get(SHA_KEY)
        if target.exists() and target.stat().st_size == obj["Size"] and \
                (expected is None or _sha256(target) == expected):
            return f"skip  {rel}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        client.download_file(bucket, obj["Key"], str(tmp), Config=transfer)
        if expected and _sha256(tmp) != expected:
            tmp.unlink()
            raise RuntimeError(f"{rel}: sha256 mismatch after download")
        tmp.replace(target)
        return f"done  {rel}" + ("" if expected else "  (no sha256 on the object: size-checked only)")

    return _run_all(one, objects, args.workers, lambda o: o["Key"])


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK):
            sha.update(block)
    return sha.hexdigest()


def _run_all(fn, items, workers: int, name) -> int:
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001 - all failures are listed at the end
                failures.append(f"{name(futures[future])}: {type(exc).__name__}: {exc}")
                print(f"FAIL  {failures[-1]}", flush=True)
    if failures:
        print(f"\n{len(failures)} of {len(items)} transfers failed (re-run to retry only those):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"all {len(items)} files in place")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("hf-to-s3", help="Hugging Face repo -> MinIO (run where there is internet)")
    up.add_argument("--repo", required=True)
    up.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    up.add_argument("--revision", default=None, help="branch, tag or commit; recorded as the resolved commit")
    up.add_argument("--dest", required=True, help="s3://bucket/prefix")
    up.add_argument("--include", nargs="*", default=None, help="glob(s) on repo paths; default: every file")
    up.add_argument("--workers", type=int, default=4)
    down = sub.add_parser("s3-to-dir", help="MinIO -> local directory (run inside the cluster)")
    down.add_argument("--src", required=True, help="s3://bucket/prefix")
    down.add_argument("--dest", required=True)
    down.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    return hf_to_s3(args) if args.cmd == "hf-to-s3" else s3_to_dir(args)


if __name__ == "__main__":
    sys.exit(main())
