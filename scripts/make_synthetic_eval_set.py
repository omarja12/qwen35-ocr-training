#!/usr/bin/env python3
"""Generate an N-page synthetic corpus for the SC-002 performance scenario.

Used by the performance check in quickstart.md and by nothing else. This is not
a dataset builder: the text is filler, the pages carry no images, and the only
property that matters is that there are N of them at a realistic length.

    python scripts/make_synthetic_eval_set.py --pages 1000 --out /tmp/perf_corpus

Deterministic for a given --seed, so a timing run can be repeated.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LINES = [
    "بسم الله الرحمن الرحيم",
    "الحمد لله رب العالمين",
    "هذا نص تجريبي للتقييم",
    "الفصل الأول في المقدمة",
    "تقرير عن حالة الطقس اليوم",
    "كان الجو معتدلا في الصباح",
    "وانخفضت الحرارة في الليل",
    "خاتمة الفصل والملاحظات",
    "المادة الثانية من النظام",
    "وقد جرى تعديل هذا البند",
]

FONTS = ["Amiri", "Cairo", "Naskh", "Scheherazade"]
DISTORTIONS = [[], ["gaussian_blur"], ["jpeg_artifact"], ["rotate"],
               ["gaussian_blur", "jpeg_artifact"]]


def corrupt(text: str, rng: random.Random, rate: float) -> str:
    """Introduce a realistic sprinkling of character errors."""
    characters = list(text)
    for index, char in enumerate(characters):
        if char.strip() and rng.random() < rate:
            characters[index] = rng.choice("ىهةکﻻ")
    return "".join(characters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--lines-per-page", type=int, default=50)
    parser.add_argument("--error-rate", type=float, default=0.05)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.out)
    refs = root / "refs"
    preds = root / "predictions"
    refs.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(args.pages):
        page_id = f"perf_{index:06d}"
        reference = "\n".join(
            rng.choice(LINES) for _ in range(args.lines_per_page)
        )
        (refs / f"{page_id}.txt").write_text(
            reference + "\n", encoding="utf-8", newline=""
        )
        (preds / f"{page_id}.txt").write_text(
            corrupt(reference, rng, args.error_rate) + "\n",
            encoding="utf-8",
            newline="",
        )
        rows.append(
            {
                "page_id": page_id,
                "split": "synthetic_heldout",
                "kind": "synthetic",
                "reference_path": f"refs/{page_id}.txt",
                "fonts": [rng.choice(FONTS)],
                "distortions": rng.choice(DISTORTIONS),
                "source": f"filler_{index % 7}",
                "is_augmented": bool(index % 3 == 0),
            }
        )

    (root / "manifest.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="",
    )

    print(f"{args.pages} pages written to {root}")
    print(f"  manifest:    {root / 'manifest.jsonl'}")
    print(f"  predictions: {preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
