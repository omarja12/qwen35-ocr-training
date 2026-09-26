#!/usr/bin/env python3
"""Fail loudly, before four H200s are committed to a broken run.

    python scripts/preflight.py --imports-only   # during the image build
    python scripts/preflight.py                  # inside the GPU pod
"""
import argparse
import importlib
import os
import sys

MODULES = [
    "torch", "torchvision", "transformers", "swift", "peft",
    "datasets", "webdataset", "boto3", "PIL", "numpy",
]

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def check_imports() -> None:
    print("== imports ==")
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            check(name, True, getattr(mod, "__version__", ""))
        except Exception as exc:                       # noqa: BLE001
            check(name, False, f"{type(exc).__name__}: {exc}")


def check_gpus(expected: int) -> None:
    print(f"\n== gpus (expecting {expected}) ==")
    try:
        import torch
    except Exception as exc:                           # noqa: BLE001
        check("torch import", False, str(exc))
        return

    check("cuda available", torch.cuda.is_available())
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    check(f"device count == {expected}", count == expected, f"found {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"       gpu {i}: {props.name}  {props.total_memory / 1024**3:.0f} GiB")


def check_env() -> None:
    print("\n== environment ==")
    # Offline flags turn an unexpected Internet call into an obvious failure
    # rather than a pod that hangs for minutes and then dies.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        check(var, os.environ.get(var) == "1", os.environ.get(var, "unset"))
    for var in ("MODEL_PATH", "TRAIN_DATA", "VAL_DATA"):
        path = os.environ.get(var)
        check(var, bool(path and os.path.exists(path)), path or "unset")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports-only", action="store_true",
                    help="skip GPU and env checks (used during the image build)")
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()

    check_imports()
    if not args.imports_only:
        check_gpus(args.gpus)
        check_env()

    print()
    if failures:
        print(f"PREFLIGHT FAILED: {', '.join(failures)}")
        return 1
    print("preflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
