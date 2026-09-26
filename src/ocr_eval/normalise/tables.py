"""Codepoint tables for the three normalisation levels.

This is a versioned contract, not an implementation detail
(contracts/normalisation-policy.md). Any change to any table here requires a
POLICY_VERSION bump: two reports produced under different versions are not
comparable, and `compare` refuses to put them side by side rather than warning.
"""

from __future__ import annotations

POLICY_VERSION = "1.0"

UNREADABLE_MARKER = "[UNREADABLE]"

LEVELS = ("strict", "diacritic_insensitive", "skeleton")
HEADLINE_LEVEL = "strict"

TATWEEL = "ـ"

# The diacritic set, exactly as the policy states it.
#
# U+08E2 is deliberately absent: it is a format character (category Cf), not a
# mark, and folding it here would silently delete a character that changes how
# the text is laid out.
_DIACRITIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x064B, 0x065F),  # fathatan .. the extended marks, incl. shadda and sukun
    (0x0670, 0x0670),  # superscript alef
    (0x06D6, 0x06DC),  # Quranic annotation marks
    (0x06DF, 0x06E4),  # Quranic annotation marks
    (0x06E7, 0x06E8),  # small yeh, small noon ghunna
    (0x06EA, 0x06ED),  # empty centre / low stop marks
    (0x08D3, 0x08E1),  # Arabic Extended-A marks
    (0x08E3, 0x08FF),  # Arabic Extended-A marks (U+08E2 excluded above)
)

DIACRITICS: frozenset[str] = frozenset(
    chr(cp)
    for low, high in _DIACRITIC_RANGES
    for cp in range(low, high + 1)
)

# Blocks whose characters count as an "Arabic base" for step 2 of the
# diacritic-insensitive pipeline. A combining mark is only dropped there when it
# sits on one of these; a mark on a Latin base is somebody else's script and is
# left alone.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Letter folding for the skeleton level: consonantal shape only.
LETTER_FOLD: dict[str, str] = {
    # alef variants to bare alef
    "آ": "ا",
    "أ": "ا",
    "إ": "ا",
    "ٱ": "ا",
    "ٲ": "ا",
    "ٳ": "ا",
    "ٵ": "ا",
    # hamza carriers to their carrier letter
    "ؤ": "و",  # hamza-on-waw to waw
    "ئ": "ي",  # hamza-on-yeh to yeh
    # standalone hamza is not part of the rasm
    "ء": "",
    # yeh variants
    "ى": "ي",  # alef maksura to yeh
    "ی": "ي",  # Persian yeh to Arabic yeh
    # kaf variants
    "ک": "ك",  # keheh to Arabic kaf
    # heh variants
    "ة": "ه",  # teh marbuta to heh
    "ۀ": "ه",
    "ہ": "ه",
    "ۂ": "ه",
}

# Digit folding: Arabic-Indic and Extended Arabic-Indic to ASCII.
DIGIT_FOLD: dict[str, str] = {}
for _base in (0x0660, 0x06F0):
    for _offset in range(10):
        DIGIT_FOLD[chr(_base + _offset)] = str(_offset)
del _base, _offset


def is_arabic_base(ch: str) -> bool:
    """True when `ch` belongs to an Arabic block."""
    cp = ord(ch)
    return any(low <= cp <= high for low, high in _ARABIC_RANGES)


def policy_object() -> dict[str, object]:
    """The policy block embedded whole into every run manifest and summary.

    Embedded rather than referenced so the artifact stands alone six months
    later (FR-004, SC-008), and so a mismatch between the scoring contract and
    the training targets is visible in the run itself (FR-010a).
    """
    return {
        "policy_version": POLICY_VERSION,
        "levels": list(LEVELS),
        "headline_level": HEADLINE_LEVEL,
        "trailing_newline_stripped": True,
        "unreadable_marker": UNREADABLE_MARKER,
    }
