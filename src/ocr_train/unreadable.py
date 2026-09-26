"""Pages with no legible text, whose only correct transcription is `[UNREADABLE]`.

OCR-Data/ocr_data is synthetic and every page in it is legible, so a model
trained on it alone never sees the one case where the right answer is to refuse.
Asked to read a blank page it would then invent text — the hallucination the
evaluation harness measures on its probe set. These negatives are that case.

Three kinds, all unambiguous: nobody could read text off any of them.

  blank        paper-coloured page with faint grain
  noise        uniform pixel noise
  obliterated  a real page shrunk to ~16 px wide and blown back up

Every image is a pure function of (seed, name, source page), so a rebuilt
dataset contains byte-identical negatives (constitution II). PIL's own noise
generator is not seeded from Python, which is why noise comes from `randbytes`.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

KINDS = ("blank", "noise", "obliterated")


def _page_size(rng: random.Random) -> tuple[int, int]:
    return rng.randint(320, 1000), rng.randint(240, 1400)


def _blank(rng: random.Random) -> Image.Image:
    w, h = _page_size(rng)
    base = tuple(rng.randint(225, 255) for _ in range(3))
    page = Image.new("RGB", (w, h), base)
    grain = Image.frombytes("L", (w, h), rng.randbytes(w * h)).convert("RGB")
    return Image.blend(page, grain, alpha=rng.uniform(0.02, 0.08))


def _noise(rng: random.Random) -> Image.Image:
    w, h = _page_size(rng)
    if rng.random() < 0.5:
        return Image.frombytes("L", (w, h), rng.randbytes(w * h)).convert("RGB")
    return Image.frombytes("RGB", (w, h), rng.randbytes(w * h * 3))


def _obliterated(rng: random.Random, source: Path) -> Image.Image:
    with Image.open(source) as src:
        page = src.convert("RGB")
    w, h = page.size
    tiny_w = rng.randint(10, 20)
    tiny = page.resize((tiny_w, max(1, round(h * tiny_w / w))), Image.Resampling.BILINEAR)
    return tiny.resize((w, h), Image.Resampling.BICUBIC)


def make_unreadable(seed: int, name: str, out: Path, source: Path | None = None) -> str:
    """Write the negative called `name` to `out` (PNG) and return its kind.

    `source` is the legible page an `obliterated` negative is made from; without
    one, the kind is drawn from the other two.
    """
    rng = random.Random(f"unreadable:{seed}:{name}")
    kinds = KINDS if source is not None else KINDS[:2]
    kind = kinds[rng.randrange(len(kinds))]
    if kind == "blank":
        image = _blank(rng)
    elif kind == "noise":
        image = _noise(rng)
    else:
        image = _obliterated(rng, source)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    image.save(tmp, format="PNG")
    tmp.replace(out)
    return kind
