"""ocr_train.prepare on tiny synthetic shards: split, negatives, failures, determinism.

Builds its own tar shards in tmp_path, so it needs Pillow but no network and no
dataset.
"""

import io
import json
import tarfile

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ocr_eval.io.manifest import read_manifest  # noqa: E402
from ocr_eval.normalise.tables import UNREADABLE_MARKER  # noqa: E402
from ocr_train import prepare  # noqa: E402

PAGES_PER_SHARD = 12


def _png(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64 + seed % 7, 90), (255 - seed % 50, 250, 245)).save(buf, "PNG")
    return buf.getvalue()


def _add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _annotation(text, aug=None):
    return json.dumps({"markdown": text, "meta": {"page_font": "amiri.ttf", "template": "letter",
                                                  "augmentation": aug}}, ensure_ascii=False).encode()


def build_source(root, groups=(1, 2, 3), defects=True):
    (root / "data").mkdir(parents=True)
    (root / "data_aug").mkdir()
    for g in groups:
        for aug in (0, 1):
            stem = f"shard_{g:03d}" + (f"_aug{aug}" if aug else "")
            folder = "data_aug" if aug else "data"
            with tarfile.open(root / folder / f"{stem}.tar", "w") as tar:
                for i in range(PAGES_PER_SHARD):
                    key = f"sample_{i:07d}" + (f"_{aug:02d}" if aug else "")
                    ext = "jpg" if aug else "png"
                    text = f"صفحة {g}-{i}"
                    image = _png(g * 100 + i)
                    if defects and g == 1 and aug == 0:
                        if i == 0:
                            text = ""                      # empty target
                        if i == 1:
                            image = image[:40]             # truncated image
                    _add(tar, f"{key}.{ext}", image)
                    if defects and g == 1 and aug == 0 and i == 2:
                        continue                           # annotation missing
                    _add(tar, f"{key}.json", _annotation(text, {"name": "blur"} if aug else None))
    return root


def run(source, out, *extra):
    return prepare.main([
        "--source", str(source), "--out", str(out), "--image-root", "/workspace/data/prepared",
        "--holdout-groups", "3", "--val-size", "4", "--test-size", "10", "--probe-count", "5",
        "--unreadable-fraction", "0.1", "--workers", "2", "--max-drop-fraction", "0.1", *extra,
    ])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prep")
    source = build_source(tmp / "src")
    assert run(source, tmp / "out") == 0
    return tmp / "out", source


def test_held_out_group_never_reaches_training(prepared):
    out, _ = prepared
    train_images = [r["images"][0] for r in read_jsonl(out / "train.jsonl")]
    assert not any("/shard_003" in p for p in train_images)
    assert any("/shard_001_aug1/" in p for p in train_images)
    val_images = [r["images"][0] for r in read_jsonl(out / "val.jsonl")]
    assert all("/shard_003" in p or "/_unreadable/" in p for p in val_images)


def test_every_defect_is_dropped_and_reported(prepared):
    out, _ = prepared
    report = json.loads((out / "prepare_report.json").read_text(encoding="utf-8"))
    assert report["samples_dropped"] == 3
    reasons = {p["key"]: p["problem"] for p in report["problems"]["shard_001"]}
    assert reasons["sample_0000000"] == "markdown empty"
    assert reasons["sample_0000001"].startswith("image does not decode")
    assert reasons["sample_0000002"] == "json missing from shard"
    assert report["samples_ok"] == 6 * PAGES_PER_SHARD - 3


def test_training_rows_use_the_image_root_and_unreadable_marker(prepared):
    out, _ = prepared
    rows = read_jsonl(out / "train.jsonl")
    assert all(r["images"][0].startswith("/workspace/data/prepared/images/") for r in rows)
    refusals = [r for r in rows if r["messages"][1]["content"] == UNREADABLE_MARKER]
    report = json.loads((out / "prepare_report.json").read_text(encoding="utf-8"))
    assert len(refusals) == report["train_unreadable"] > 0
    assert len(rows) == report["train_rows"] == 4 * PAGES_PER_SHARD - 3 + report["train_unreadable"]


def test_eval_manifest_is_valid_for_ocr_eval(prepared):
    out, _ = prepared
    result = read_manifest(out / "eval" / "manifest.jsonl")
    assert result.problems == []
    kinds = [p.kind for p in result.pages]
    assert kinds.count("probe") == 5 and kinds.count("synthetic") == 10
    for page in result.pages:
        assert (out / page.image_path).is_file()
        if page.kind == "probe":
            assert page.reference_text == UNREADABLE_MARKER


def test_rerun_resumes_and_is_byte_identical(prepared, tmp_path):
    out, source = prepared
    before = {name: (out / name).read_bytes() for name in ("train.jsonl", "val.jsonl", "eval/manifest.jsonl")}
    assert run(source, out) == 0                           # every shard already done: resumed
    assert run(source, tmp_path / "fresh") == 0            # from scratch, elsewhere
    for name, data in before.items():
        assert (out / name).read_bytes() == data
        assert (tmp_path / "fresh" / name).read_bytes() == data
    probe = "images/_unreadable/unreadable_probe_0000000.png"
    assert (out / probe).read_bytes() == (tmp_path / "fresh" / probe).read_bytes()


def test_too_many_bad_samples_fails_the_run(tmp_path):
    source = build_source(tmp_path / "src")
    assert run(source, tmp_path / "out", "--max-drop-fraction", "0") == 1


def test_holdout_must_name_a_present_group(tmp_path):
    source = build_source(tmp_path / "src", defects=False)
    with pytest.raises(SystemExit, match="holdout"):
        prepare.main(["--source", str(source), "--out", str(tmp_path / "out"), "--holdout-groups", "9"])


def test_unrecognised_tar_names_stop_the_run(tmp_path):
    source = build_source(tmp_path / "src", groups=(1, 2), defects=False)
    (source / "data" / "stray.tar").write_bytes(b"")
    with pytest.raises(SystemExit, match="unrecognised"):
        run(source, tmp_path / "out")
