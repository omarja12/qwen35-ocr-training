"""The pieces training, prediction and evaluation must agree on (ocr_train.records)."""

import pytest

from ocr_eval.normalise.tables import UNREADABLE_MARKER
from ocr_train.predict import strip_prefix
from ocr_train.records import (
    PROMPT,
    annotation_problems,
    estimate_image_tokens,
    page_id,
    parse_shard,
    training_record,
)


@pytest.mark.parametrize("stem, group, aug", [
    ("shard_001", 1, 0),
    ("shard_103", 103, 0),
    ("shard_017_aug2", 17, 2),
])
def test_parse_shard(stem, group, aug):
    shard = parse_shard(stem)
    assert (shard.stem, shard.group, shard.aug) == (stem, group, aug)


@pytest.mark.parametrize("stem", ["shard_x", "data_001", "shard_001_aug", "shard_001.tar", ""])
def test_parse_shard_rejects_other_names(stem):
    with pytest.raises(ValueError):
        parse_shard(stem)


def test_original_and_augmented_share_a_group():
    # The held-out split is drawn on groups; this is what keeps a page and its
    # augmented copies on the same side of it.
    assert parse_shard("shard_042").group == parse_shard("shard_042_aug3").group


def test_page_id_is_unique_across_shards_and_filename_safe():
    ids = {page_id("shard_001", "sample_0000000"), page_id("shard_002", "sample_0000000"),
           page_id("shard_001_aug1", "sample_0000000_01")}
    assert len(ids) == 3
    assert not any("/" in i for i in ids)


def test_prompt_names_the_exact_marker():
    # The harness scores refusals against this exact string; the prompt must ask for it verbatim.
    assert UNREADABLE_MARKER in PROMPT


def test_training_record_shape():
    record = training_record("/data/images/shard_001/a.png", "نص")
    assert record == {
        "messages": [
            {"role": "user", "content": "<image>" + PROMPT},
            {"role": "assistant", "content": "نص"},
        ],
        "images": ["/data/images/shard_001/a.png"],
    }


@pytest.mark.parametrize("ann, expected", [
    ({"markdown": "نص"}, []),
    ({"markdown": ""}, ["markdown empty"]),
    ({"markdown": "  \n"}, ["markdown empty"]),
    ({}, ["markdown missing or not a string"]),
    ({"markdown": 3}, ["markdown missing or not a string"]),
    ({"markdown": UNREADABLE_MARKER}, ["markdown equals the unreadable marker"]),
    ({"markdown": "نص", "meta": "x"}, ["meta is not an object"]),
    ([], ["annotation is not a JSON object"]),
])
def test_annotation_problems(ann, expected):
    assert annotation_problems(ann) == expected


def test_annotation_problems_reports_every_problem():
    assert len(annotation_problems({"markdown": "", "meta": "x"})) == 2


def test_image_tokens_typical_page():
    # 726x1020, the dataset's common A4-at-96-dpi size: 23 x 32 merged patches.
    assert estimate_image_tokens(726, 1020, max_tokens=1280) == 23 * 32


def test_image_tokens_respects_budget():
    assert estimate_image_tokens(4000, 4000, max_tokens=1280) <= 1280
    assert estimate_image_tokens(4000, 4000, max_tokens=10**9) == 125 * 125


def test_image_tokens_tiny_image_gets_minimum():
    assert estimate_image_tokens(10, 10, max_tokens=1280) >= 4


# --- ocr_train.predict: the one thing it removes from model output ---------------

PREFIX = "<think>\n\n</think>\n\n"


@pytest.mark.parametrize("raw, expected", [
    (PREFIX + UNREADABLE_MARKER, UNREADABLE_MARKER),          # the refusal must survive intact
    (PREFIX + "نص\nسطر", "نص\nسطر"),
    ("نص", "نص"),                                              # no prefix: untouched
    ("<think>\nplanning\n</think>\n\nنص", "<think>\nplanning\n</think>\n\nنص"),  # real thinking is data
    (PREFIX + PREFIX + "نص", PREFIX + "نص"),                  # only one, exact
    (" " + PREFIX + "نص", " " + PREFIX + "نص"),
])
def test_strip_prefix_removes_only_the_exact_template_prefix(raw, expected):
    assert strip_prefix(raw, PREFIX) == expected


def test_strip_prefix_without_a_template_prefix_is_a_no_op():
    assert strip_prefix(PREFIX + "نص", "") == PREFIX + "نص"
