"""What training, prediction and evaluation must agree on.

Standard library only, so the test suite and the data preparation CPU pod can
import it without the training stack.

The `[UNREADABLE]` marker is imported from the evaluation harness rather than
restated: the harness scores against that exact string (README, FR-010a), and a
second copy of it here is how the two would drift apart.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ocr_eval.normalise.tables import UNREADABLE_MARKER

# The one instruction the model is trained and evaluated with. Changing it after
# training has started changes what every checkpoint is being asked to do, so
# predict.py reads it from here and never takes it from the command line.
PROMPT = (
    "Transcribe all visible text in this image exactly, as Markdown, preserving "
    "reading order and line breaks. Return only the transcription. If the image "
    f"contains no legible text, return exactly {UNREADABLE_MARKER}"
)

# Qwen3.5 vision geometry: 16 px patches merged 2x2, so one LLM token per 32x32 px.
IMAGE_FACTOR = 32

_SHARD_RE = re.compile(r"^shard_(\d+)(?:_aug(\d+))?$")


@dataclass(frozen=True)
class Shard:
    """One tar of OCR-Data/ocr_data, identified by its file stem."""

    stem: str  # shard_001 or shard_001_aug2
    group: int  # 1 - an original and its augmented variants share a group
    aug: int  # 0 for the original, 1..3 for data_aug variants


def parse_shard(stem: str) -> Shard:
    """`shard_017_aug2` -> Shard(group=17, aug=2). Anything else is an error.

    The group number is what the held-out split is drawn on: augmented shard N
    holds variants of the pages in original shard N, so splitting by shard file
    instead of by group would put the same page in train and in validation.
    """
    match = _SHARD_RE.match(stem)
    if not match:
        raise ValueError(f"not an OCR-Data shard name: {stem!r}")
    return Shard(stem=stem, group=int(match.group(1)), aug=int(match.group(2) or 0))


def page_id(shard_stem: str, key: str) -> str:
    """Stable, filesystem-safe and unique across the whole dataset.

    Sample keys restart in every shard, so the key alone is not unique.
    """
    return f"{shard_stem}__{key}"


def training_record(image_path: str, target: str) -> dict:
    """One ms-swift SFT row. The `<image>` tag is where the page is inserted."""
    return {
        "messages": [
            {"role": "user", "content": "<image>" + PROMPT},
            {"role": "assistant", "content": target},
        ],
        "images": [image_path],
    }


def annotation_problems(ann: object) -> list[str]:
    """Every reason this annotation cannot become a training target."""
    if not isinstance(ann, dict):
        return ["annotation is not a JSON object"]
    problems = []
    markdown = ann.get("markdown")
    if not isinstance(markdown, str):
        problems.append("markdown missing or not a string")
    elif not markdown.strip():
        problems.append("markdown empty")
    elif markdown.strip() == UNREADABLE_MARKER:
        # A legible page whose target is the refusal marker would teach refusal.
        problems.append("markdown equals the unreadable marker")
    meta = ann.get("meta")
    if meta is not None and not isinstance(meta, dict):
        problems.append("meta is not an object")
    return problems


def estimate_image_tokens(width: int, height: int, max_tokens: int, min_tokens: int = 4) -> int:
    """LLM tokens one page costs after qwen_vl_utils' smart_resize.

    Used for the preparation report only (how many pages the image budget
    downscales), never to decide what is trained on.
    """
    f = IMAGE_FACTOR
    max_pixels, min_pixels = max_tokens * f * f, min_tokens * f * f
    h_bar = max(f, round(height / f) * f)
    w_bar = max(f, round(width / f) * f)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / f) * f
        w_bar = math.ceil(width * beta / f) * f
    return (h_bar // f) * (w_bar // f)
