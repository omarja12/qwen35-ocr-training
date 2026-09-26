"""The rapidfuzz path and the pure-Python fallback must agree exactly.

A fallback that disagreed with the fast path would make a run's numbers depend
on which wheels happened to be installed — the same corpus scored twice would
produce two different CERs and neither would be wrong on its own terms. That is
why the equality is asserted rather than assumed (research R-001).
"""

import itertools
import random

import pytest

from ocr_eval.metrics.distance import (
    ENGINE,
    _myers,
    levenshtein,
    levenshtein_sequences,
    similarity,
)

try:
    from rapidfuzz.distance import Levenshtein as _RF

    HAS_RAPIDFUZZ = True
except ImportError:
    _RF = None
    HAS_RAPIDFUZZ = False


def _dp(a, b):
    """Textbook dynamic-programming Levenshtein: the independent reference."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)
            )
        previous = current
    return previous[len(b)]


VECTORS = [
    ("", ""),
    ("", "abc"),
    ("abc", ""),
    ("abc", "abc"),
    ("abc", "abd"),
    ("kitten", "sitting"),
    ("flaw", "lawn"),
    ("a" * 40, ""),                      # pure deletion
    ("", "b" * 40),                      # pure insertion
    ("بسم الله", "بسم اللە"),
    ("الحمد لله رب العالمين", "الحمد لله رب العالميں"),
    ("١٢٣٤٥", "12345"),
]


@pytest.mark.parametrize("a,b", VECTORS)
def test_myers_matches_dp(a, b):
    assert _myers(a, b) == _dp(a, b)


def test_myers_matches_dp_exhaustively_on_a_small_alphabet():
    for la in range(5):
        for lb in range(5):
            for a in itertools.product("ab", repeat=la):
                for b in itertools.product("ab", repeat=lb):
                    a, b = "".join(a), "".join(b)
                    assert _myers(a, b) == _dp(a, b), (a, b)


def test_myers_matches_dp_on_random_arabic():
    alphabet = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ١٢٣abc"
    rng = random.Random(20260916)
    for _ in range(500):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50)))
        assert _myers(a, b) == _dp(a, b)


@pytest.mark.skipif(not HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
@pytest.mark.parametrize("a,b", VECTORS)
def test_rapidfuzz_path_equals_fallback_path(a, b):
    assert int(_RF.distance(a, b)) == _myers(a, b)


@pytest.mark.skipif(not HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_rapidfuzz_path_equals_fallback_path_on_random_input():
    alphabet = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي abc"
    rng = random.Random(20260916)
    for _ in range(300):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert int(_RF.distance(a, b)) == _myers(a, b)


def test_engine_is_reported_for_provenance():
    assert ENGINE in ("rapidfuzz", "pure-python-myers")


def test_token_sequence_distance():
    assert levenshtein_sequences(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert levenshtein_sequences(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert levenshtein_sequences([], ["a", "b"]) == 2
    assert levenshtein_sequences(["a", "b"], []) == 2
    # Whole words differ, not characters: one substitution, not several.
    assert levenshtein_sequences(["hello", "world"], ["hello", "there"]) == 1


def test_similarity_bounds():
    assert similarity("", "") == 1.0
    assert similarity("abc", "abc") == 1.0
    assert similarity("abc", "xyz") == 0.0
    assert 0.0 < similarity("abcd", "abxd") < 1.0


def test_levenshtein_public_entry_point_matches_dp():
    for a, b in VECTORS:
        assert levenshtein(a, b) == _dp(a, b)
