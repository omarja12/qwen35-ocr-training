"""Shared paths and helpers. No network, no external data — fixtures only."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mini_corpus"
CONTRACTS = REPO_ROOT / "specs" / "001-ocr-eval-harness" / "contracts"


@pytest.fixture(scope="session")
def mini_corpus():
    return FIXTURES


@pytest.fixture(scope="session")
def manifest_path():
    return FIXTURES / "manifest.jsonl"


@pytest.fixture(scope="session")
def baseline_predictions():
    return FIXTURES / "predictions" / "baseline"


@pytest.fixture(scope="session")
def finetuned_predictions():
    return FIXTURES / "predictions" / "finetuned"


@pytest.fixture(scope="session")
def config():
    from ocr_eval.config import load_config

    return load_config()


def load_schema(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
