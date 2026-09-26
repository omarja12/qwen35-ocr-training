"""SC-003: two runs over identical inputs produce byte-identical reports.

The constitution requires a determinism guarantee to be asserted by running
twice and comparing, not by reasoning about why it should hold. That is what
this file does — it scores the same corpus into two directories and compares
bytes.

`run.meta.json` is excluded by design, and the exclusion is itself tested: the
fields that genuinely vary are segregated into one file precisely so they cannot
break the comparison of everything else (research R-011).
"""

import json

import pytest

from ocr_eval.config import load_config
from ocr_eval.score.run import score_run
from tests.conftest import FIXTURES

VOLATILE = "run.meta.json"


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("identity")
    first, second = base / "first", base / "second"
    # The second run is parallel, so this also proves --jobs cannot change a byte.
    for out, jobs in ((first, 1), (second, 2)):
        assert score_run(
            manifest_path=FIXTURES / "manifest.jsonl",
            predictions_path=FIXTURES / "predictions" / "baseline",
            model_version="identity-test",
            out_dir=out,
            config=load_config(),
            jobs=jobs,
        ) == 0
    return first, second


def test_every_file_except_run_meta_is_byte_identical(two_runs):
    first, second = two_runs
    names = {p.name for p in first.iterdir()}
    assert names == {p.name for p in second.iterdir()}

    compared = 0
    for name in sorted(names):
        if name == VOLATILE:
            continue
        a = (first / name).read_bytes()
        b = (second / name).read_bytes()
        assert a == b, f"{name} differs between two runs on identical inputs"
        compared += 1
    assert compared >= 6, "the comparison should cover the whole run directory"


def test_run_meta_is_the_only_file_allowed_to_differ(two_runs):
    """The exclusion is deliberate, so it is asserted rather than assumed."""
    first, second = two_runs
    a = json.loads((first / VOLATILE).read_text(encoding="utf-8"))
    b = json.loads((second / VOLATILE).read_text(encoding="utf-8"))
    # Same shape, and it carries exactly the things that cannot be deterministic.
    assert set(a) == set(b)
    assert {"completed_at", "elapsed_seconds", "host"} <= set(a)


def test_no_wall_clock_or_host_leaked_into_the_compared_files(two_runs):
    """Byte-identity would catch this, but the reason deserves its own test."""
    first, _ = two_runs
    import socket

    hostname = socket.gethostname()
    for path in sorted(first.iterdir()):
        if path.name == VOLATILE:
            continue
        text = path.read_text(encoding="utf-8")
        assert hostname not in text, f"{path.name} leaks the host name"
        assert str(first) not in text, f"{path.name} leaks an absolute path"


def test_files_end_with_exactly_one_newline_and_use_lf(two_runs):
    """LF endings and one trailing newline, on every platform."""
    first, _ = two_runs
    for path in sorted(first.iterdir()):
        raw = path.read_bytes()
        assert raw.endswith(b"\n"), f"{path.name} has no trailing newline"
        assert not raw.endswith(b"\n\n"), f"{path.name} has more than one"
        assert b"\r\n" not in raw, f"{path.name} contains CRLF"


def test_utf8_without_a_bom(two_runs):
    first, _ = two_runs
    for path in sorted(first.iterdir()):
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), path.name


def test_floats_are_serialised_at_six_decimal_places(two_runs):
    first, _ = two_runs
    for line in (first / "pages.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for value in _walk_floats(row):
            rendered = repr(value)
            if "." in rendered and "e" not in rendered:
                decimals = len(rendered.split(".")[1])
                assert decimals <= 6, f"{rendered} carries more than six places"


def _walk_floats(obj):
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_floats(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_floats(value)
