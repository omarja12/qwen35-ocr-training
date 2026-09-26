"""Threshold configuration.

Every tunable number the harness uses lives in `configs/eval/default.json` and
is echoed **in full** into each run's manifest and every comparison artifact
(FR-014, SC-008). A reader six months later sees the thresholds that produced
the numbers without guessing which version of the file was on disk.

The object is frozen once resolved. A threshold that changed halfway through a
run would make the run's own artifact a lie about how it was produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ocr_eval.errors import UsageError

DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "runaway_length_ratio": 2.0,
        "runaway_min_char_excess": 100,
        "repetition_min_line_len": 10,
        "repetition_min_occurrences": 5,
        "repetition_min_char_share": 0.30,
        "line_match_min_similarity": 0.5,
        "min_pages_per_breakdown_row": 5,
        "promote_min_cer_improvement_pp": 0.5,
        "reject_on_any_gold_regression": True,
        "reject_hallucination_worsening_pp": 1.0,
        "reject_false_refusal_worsening_pp": 1.0,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260916,
    }
)

# The confidence level is fixed rather than configurable: R-014 fixes it at 95%
# and contracts/comparison.schema.json pins it with a const, unlike resamples
# and seed which the schema records rather than pins.
BOOTSTRAP_CONFIDENCE = 0.95

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "default.json"


@dataclass(frozen=True)
class Config:
    """Resolved thresholds, plus the provenance of where they came from."""

    values: Mapping[str, Any]
    source: str

    def __getattr__(self, name: str) -> Any:
        # Thresholds read as attributes at call sites (config.runaway_length_ratio)
        # without needing a field per key duplicated from the JSON.
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None

    def echo(self) -> dict[str, Any]:
        """The full resolved config, for embedding verbatim in an artifact."""
        return dict(self.values)

    def __reduce__(self):
        # MappingProxyType cannot be pickled, and `score --jobs` sends the config
        # to worker processes. Rebuild it on the far side from a plain copy.
        return (_frozen, (dict(self.values), self.source))


def _frozen(values: dict[str, Any], source: str) -> Config:
    return Config(values=MappingProxyType(values), source=source)


def load_config(path: str | Path | None = None) -> Config:
    """Load thresholds, optionally from a `--config` override.

    An unknown key is a hard failure rather than a silent ignore. A typo in a
    threshold name would otherwise leave the default in force while the artifact
    echoed the file the user wrote, which is a reproducibility claim that is not
    true (constitution II).
    """
    if path is None:
        if DEFAULT_CONFIG_PATH.exists():
            return _load_file(DEFAULT_CONFIG_PATH)
        # Running from an installed wheel with no repo checkout beside it.
        return _frozen(dict(DEFAULTS), "built-in defaults")

    config_path = Path(path)
    if not config_path.exists():
        raise UsageError(f"config file not found: {config_path}")
    return _load_file(config_path)


def _load_file(config_path: Path) -> Config:
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UsageError(f"cannot read config {config_path}: {exc}") from None

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UsageError(f"config {config_path} is not valid JSON: {exc.msg}") from None

    if not isinstance(loaded, dict):
        raise UsageError(f"config {config_path} must be a JSON object")

    unknown = sorted(set(loaded) - set(DEFAULTS))
    if unknown:
        raise UsageError(
            f"config {config_path} has unknown key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(DEFAULTS))}"
        )

    return _frozen({**DEFAULTS, **loaded}, str(config_path))
