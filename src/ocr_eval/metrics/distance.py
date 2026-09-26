"""Levenshtein edit distance, with a fallback that cannot change a number.

`rapidfuzz` is used when it is importable and a bundled pure-Python Myers
bit-parallel implementation is used when it is not. Both paths return identical
integers — that equality is asserted by tests/unit/test_distance.py across a
shared vector set, because a fallback that quietly disagreed with the fast path
would make a run's numbers depend on which wheels happened to be installed.

The selected engine is recorded in the run manifest for provenance
(constitution II) and never affects a result (research R-001).
"""

from __future__ import annotations

from typing import Sequence

try:  # pragma: no cover - the branch taken depends on the environment
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein

    _rapidfuzz_distance = _RapidfuzzLevenshtein.distance
    ENGINE = "rapidfuzz"
except ImportError:  # pragma: no cover
    _rapidfuzz_distance = None
    ENGINE = "pure-python-myers"


def _myers(a: Sequence[str], b: Sequence[str]) -> int:
    """Myers' bit-parallel Levenshtein distance.

    O(len(b) * ceil(len(a)/word)) with Python's arbitrary-precision integers
    standing in for machine words, so the whole pattern lives in one "word"
    however long it is. Every intermediate is masked back to `m` bits because
    Python's `~` produces a negative number rather than wrapping.
    """
    m = len(a)
    if m == 0:
        return len(b)
    if len(b) == 0:
        return m

    full = (1 << m) - 1

    peq: dict[str, int] = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)

    vp = full
    vn = 0
    score = m
    top = 1 << (m - 1)

    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | vn
        xh = (((((eq & vp) + vp) & full) ^ vp) | eq)
        hp = vn | (~(xh | vp) & full)
        hn = vp & xh

        if hp & top:
            score += 1
        if hn & top:
            score -= 1

        hp = ((hp << 1) | 1) & full
        hn = (hn << 1) & full

        vp = (hn | (~(xv | hp) & full)) & full
        vn = hp & xv

    return score


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings, as an integer number of edits."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if _rapidfuzz_distance is not None:
        return int(_rapidfuzz_distance(a, b))
    return _myers(a, b)


def levenshtein_sequences(a: Sequence[str], b: Sequence[str]) -> int:
    """Edit distance over token sequences rather than characters.

    Words are hashable but not single characters, so the bit-parallel routine is
    fed through an alphabet mapping first. rapidfuzz accepts sequences directly.
    """
    if list(a) == list(b):
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if _rapidfuzz_distance is not None:
        return int(_rapidfuzz_distance(list(a), list(b)))

    # Map each distinct token to one private-use character so the same
    # bit-parallel routine applies. The mapping is per-call and order-dependent
    # only on first appearance, which does not affect the distance.
    alphabet: dict[str, str] = {}

    def encode(tokens: Sequence[str]) -> str:
        out = []
        for tok in tokens:
            if tok not in alphabet:
                alphabet[tok] = chr(0xE000 + len(alphabet))
            out.append(alphabet[tok])
        return "".join(out)

    return _myers(encode(a), encode(b))


def similarity(a: str, b: str) -> float:
    """Normalised similarity in [0, 1]: 1 - distance / max(len)."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - (levenshtein(a, b) / longest)
