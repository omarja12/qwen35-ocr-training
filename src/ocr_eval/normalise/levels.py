"""The three normalisation levels.

Every result is reported at all three simultaneously, so that no single choice
of rules can conceal a class of error (FR-003). `strict` is the headline and the
only figure quoted when one number is quoted (FR-003a).

**The step order inside each pipeline is part of the contract.** These
operations do not commute — folding letters before stripping marks gives
different output from the reverse — so the order below matches
contracts/normalisation-policy.md line for line and must not be rearranged for
tidiness.
"""

from __future__ import annotations

import unicodedata

from ocr_eval.normalise.tables import (
    DIACRITICS,
    DIGIT_FOLD,
    LETTER_FOLD,
    LEVELS,
    TATWEEL,
    is_arabic_base,
)


def strict(text: str) -> str:
    """Level 1 — no transformation at all.

    Zero operations. Everything counts: diacritics, tatweel, letter variants,
    punctuation, digit forms, whitespace and representation form.

    This is simultaneously the raw unnormalised score FR-002 asks for and the
    strict headline FR-003 asks for. They are the same number, which is why it
    is reported once (research R-002) — and why human-readable reports label it
    "strict (raw, unnormalised)" rather than bare "strict", so a reader scanning
    for the word "raw" does not conclude it is missing.
    """
    return text


def diacritic_insensitive(text: str) -> str:
    """Level 2 — as strict, but vowelling is ignored.

    Nothing else changes: tatweel, letter variants, digit forms, punctuation,
    whitespace and representation form all still count.
    """
    # Step 1: remove every codepoint in the diacritic set.
    out: list[str] = [ch for ch in text if ch not in DIACRITICS]

    # Step 2: remove any remaining combining mark sitting on an Arabic base.
    # This is what catches NFD-decomposed forms — U+0622 written as
    # U+0627 U+0653 — which step 1 cannot see because U+0653 is outside the
    # enumerated set.
    result: list[str] = []
    last_base: str | None = None
    for ch in out:
        if unicodedata.category(ch) == "Mn":
            if last_base is not None and is_arabic_base(last_base):
                continue
            result.append(ch)
        else:
            last_base = ch
            result.append(ch)
    return "".join(result)


def skeleton(text: str) -> str:
    """Level 3 — consonantal letters and their order only."""
    # 1. NFKC folds presentation forms back to base letters and decomposes
    #    ligatures into their parts.
    s = unicodedata.normalize("NFKC", text)

    # 2. Remove all combining marks.
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    # 3. Remove tatweel.
    s = s.replace(TATWEEL, "")

    # 4. Letter folding.
    s = "".join(LETTER_FOLD.get(ch, ch) for ch in s)

    # 5. Digit folding.
    s = "".join(DIGIT_FOLD.get(ch, ch) for ch in s)

    # 6. Remove punctuation and symbols.
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] not in ("P", "S"))

    # 7. Collapse each run of whitespace to a single space, and strip.
    #    str.split() with no argument splits on runs of Unicode whitespace and
    #    discards empty tokens, which is exactly this rule.
    return " ".join(s.split())


_FUNCTIONS = {
    "strict": strict,
    "diacritic_insensitive": diacritic_insensitive,
    "skeleton": skeleton,
}


def all_levels(text: str) -> dict[str, str]:
    """Apply every level, returning {level: normalised text}."""
    return {level: _FUNCTIONS[level](text) for level in LEVELS}
