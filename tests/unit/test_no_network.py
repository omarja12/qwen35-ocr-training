"""No module under src/ocr_eval reaches the network (FR-015, constitution III).

The cluster cannot reach the public internet, and the rule is that code is
written so this is never *discovered* at run time. A test is the only way that
rule stays true as modules are added — nobody reviews an import list twice.
"""

import ast
import socket
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "ocr_eval"

FORBIDDEN_MODULES = {
    "urllib", "urllib2", "urllib3", "http", "httplib", "requests", "httpx",
    "aiohttp", "ftplib", "telnetlib", "smtplib", "poplib", "imaplib",
    "xmlrpc", "websocket", "websockets", "boto3", "botocore", "minio",
    "huggingface_hub", "transformers", "torch", "datasets",
}

# socket is allowed in exactly one place and for exactly one reason.
SOCKET_ALLOWED = {"score/run.py"}


def python_files():
    return sorted(SRC.rglob("*.py"))


def test_there_are_modules_to_check():
    assert python_files(), "no source files found — the test would pass vacuously"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_module_imports_a_network_client(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(SRC).as_posix()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

    offending = imported & FORBIDDEN_MODULES
    assert not offending, f"{relative} imports {sorted(offending)}"

    if "socket" in imported:
        assert relative in SOCKET_ALLOWED, (
            f"{relative} imports socket. Only {sorted(SOCKET_ALLOWED)} may, and "
            f"only to read gethostname() into run.meta.json"
        )


def test_the_one_socket_use_is_gethostname_only():
    """Allowed because a host name is provenance, not a connection."""
    source = (SRC / "score" / "run.py").read_text(encoding="utf-8")
    assert "socket.gethostname()" in source
    for forbidden in ("socket.socket", "connect(", "urlopen", "getaddrinfo"):
        assert forbidden not in source


def test_scoring_the_fixture_corpus_opens_no_connection(tmp_path, monkeypatch):
    """Belt and braces: make connecting impossible and score anyway."""
    from ocr_eval.config import load_config
    from ocr_eval.score.run import score_run
    from tests.conftest import FIXTURES

    def refuse(*args, **kwargs):
        raise AssertionError("the harness attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert score_run(
        manifest_path=FIXTURES / "manifest.jsonl",
        predictions_path=FIXTURES / "predictions" / "baseline",
        model_version="offline-test",
        out_dir=tmp_path / "run",
        config=load_config(),
    ) == 0


def test_the_package_declares_no_required_runtime_dependencies():
    """Constitution III: every added wheel is a build risk."""
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in pyproject
