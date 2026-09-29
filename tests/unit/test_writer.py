"""The single serialisation point, and the permission rule it also owns.

SC-003's byte-identity guarantee is only enforceable because there is exactly
one writer. These tests pin the properties that guarantee depends on.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ocr_eval.io import writer

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="chmod is a no-op on Windows; the group-writable rule is a Linux/OpenShift concern",
)


def test_floats_are_rounded_to_six_places():
    assert writer.round_float(1 / 3) == 0.333333
    assert writer.round_float(2.0) == 2.0


def test_negative_zero_is_folded_to_zero():
    """It compares equal to zero but serialises differently — byte-identity bait."""
    assert writer.round_float(-0.0) == 0.0
    assert "-0.0" not in writer.dumps({"value": -0.0})


def test_canonicalise_reaches_nested_floats():
    payload = {"a": [{"b": 1 / 3}], "c": (2 / 3,)}
    assert writer.canonicalise(payload) == {"a": [{"b": 0.333333}], "c": [0.666667]}


def test_keys_are_sorted_so_insertion_order_cannot_leak():
    """This is what lets --jobs reorder execution without reordering output."""
    first = writer.dumps({"b": 1, "a": 2, "c": 3})
    second = writer.dumps({"c": 3, "a": 2, "b": 1})
    assert first == second
    assert first.index('"a"') < first.index('"b"') < first.index('"c"')


def test_arabic_is_not_escaped():
    """The worst-pages report exists to be read by a human."""
    rendered = writer.dumps({"text": "بسم الله"})
    assert "بسم الله" in rendered
    assert "\\u" not in rendered


def test_written_files_use_lf_and_end_with_exactly_one_newline(tmp_path):
    target = writer.write_text(tmp_path / "a.txt", "line one\nline two")
    raw = target.read_bytes()
    assert raw == b"line one\nline two\n"
    assert b"\r\n" not in raw


def test_an_existing_trailing_newline_is_not_doubled(tmp_path):
    target = writer.write_text(tmp_path / "b.txt", "already\n")
    assert target.read_bytes() == b"already\n"


def test_written_files_have_no_bom(tmp_path):
    target = writer.write_json(tmp_path / "c.json", {"k": "قيمة"})
    assert not target.read_bytes().startswith(b"\xef\xbb\xbf")


def test_jsonl_rows_are_one_per_line_and_compact(tmp_path):
    target = writer.write_jsonl(tmp_path / "d.jsonl", [{"b": 1, "a": 2}, {"a": 3}])
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == '{"a":2,"b":1}'
    assert json.loads(lines[1]) == {"a": 3}


def test_read_text_strips_exactly_one_trailing_newline(tmp_path):
    path = tmp_path / "r.txt"
    path.write_bytes(b"content\n\n")
    assert writer.read_text(path) == "content\n"

    path.write_bytes(b"content")
    assert writer.read_text(path) == "content"


def test_read_text_strips_a_bom(tmp_path):
    """A BOM is an editor artifact; left in, it is a character error on line 1."""
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello\n")
    assert writer.read_text(path) == "hello"


def test_read_text_handles_crlf_symmetrically(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"line\r\n")
    assert writer.read_text(path) == "line"


def test_read_text_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        writer.read_text(path)


# --- the arbitrary-UID rule (research R-017, constitution V) ---------------

@POSIX_ONLY
def test_directories_are_group_writable(tmp_path):
    created = writer.ensure_dir(tmp_path / "nested" / "deep")
    assert stat.S_IMODE(created.stat().st_mode) & 0o775 == 0o775


@POSIX_ONLY
def test_files_are_group_writable(tmp_path):
    target = writer.write_text(tmp_path / "f.txt", "x")
    assert target.stat().st_mode & stat.S_IWGRP


@POSIX_ONLY
def test_permissions_are_set_explicitly_not_inherited_from_the_umask(tmp_path):
    """The whole point of R-017.

    A run directory created under `umask 077` must still be readable by the next
    pod's arbitrary UID. Run in a subprocess so the umask change cannot leak
    into the rest of the suite.
    """
    script = (
        "import os, sys; os.umask(0o077);"
        "sys.path.insert(0, %r);"
        "from ocr_eval.io import writer;"
        "d = writer.ensure_dir(%r);"
        "f = writer.write_text(%r, 'x');"
        "import stat;"
        "print(oct(stat.S_IMODE(os.stat(d).st_mode)), oct(stat.S_IMODE(os.stat(f).st_mode)))"
    ) % (
        str(Path(__file__).resolve().parents[2] / "src"),
        str(tmp_path / "umasked"),
        str(tmp_path / "umasked" / "file.txt"),
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    dir_mode, file_mode = proc.stdout.split()
    assert dir_mode == "0o775", f"umask leaked into the directory mode: {dir_mode}"
    assert file_mode == "0o664", f"umask leaked into the file mode: {file_mode}"


def test_chmod_is_skipped_rather_than_raising_on_windows(tmp_path):
    """The calls are no-ops off POSIX; they must not become errors."""
    target = writer.write_text(tmp_path / "w.txt", "x")
    assert target.exists()
