"""Per-level codepoint behaviour.

The normalisation tables are a versioned contract, so these assertions are the
contract restated in executable form: if one of them changes, POLICY_VERSION
must change with it.
"""

import unicodedata

import pytest

from ocr_eval.normalise.levels import diacritic_insensitive, skeleton, strict
from ocr_eval.normalise.tables import DIACRITICS, POLICY_VERSION

ALEF = "ا"
YEH = "ي"
HEH = "ه"
KAF = "ك"
WAW = "و"
TATWEEL = "ـ"


def test_policy_version_is_pinned():
    # A change here without a deliberate bump would make old reports silently
    # incomparable to new ones.
    assert POLICY_VERSION == "1.0"


# --- strict ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "بِسْمِ اللهِ",           # diacritics
        "بســم",                  # tatweel
        "ﻻ حول",             # presentation form
        "الأرقام ١٢٣ و 123",       # mixed digit forms
        "  leading and trailing  ",
        "",
    ],
)
def test_strict_changes_nothing(text):
    assert strict(text) == text


# --- diacritic_insensitive -------------------------------------------------

def test_diacritic_insensitive_drops_vowelling():
    assert diacritic_insensitive("بِسْمِ اللهِ") == "بسم الله"


def test_diacritic_insensitive_keeps_tatweel_digits_and_punctuation():
    text = f"بس{TATWEEL}م ١٢٣، ودعنا"
    assert diacritic_insensitive(text) == text


def test_diacritic_insensitive_handles_nfd_decomposed_alef():
    # U+0622 (alef madda) written decomposed as U+0627 U+0653. U+0653 is not in
    # the enumerated diacritic set, so only step 2 catches it — that is exactly
    # what step 2 is for.
    decomposed = "آبجد"
    assert diacritic_insensitive(decomposed) == "ابجد"


def test_diacritic_insensitive_leaves_marks_on_non_arabic_bases():
    # A combining acute on a Latin e is somebody else's script.
    assert diacritic_insensitive("café") == "café"


def test_diacritic_set_excludes_the_format_character():
    # U+08E2 is category Cf, not Mn. Folding it would delete a character that
    # changes how the text is laid out.
    assert "࣢" not in DIACRITICS
    assert unicodedata.category("࣢") == "Cf"
    assert "࣡" in DIACRITICS
    assert "ࣣ" in DIACRITICS


# --- skeleton --------------------------------------------------------------

def test_skeleton_folds_presentation_forms():
    assert skeleton("ﻻ") == skeleton("لا")


def test_skeleton_folds_alef_variants():
    for variant in ("آ", "أ", "إ", "ٱ"):
        assert skeleton(variant + "بجد") == ALEF + "بجد"


def test_skeleton_folds_yeh_and_kaf_and_heh_variants():
    assert skeleton("ى") == YEH       # alef maksura
    assert skeleton("ی") == YEH       # Persian yeh
    assert skeleton("ک") == KAF       # keheh
    assert skeleton("ة") == HEH       # teh marbuta
    assert skeleton("ہ") == HEH
    assert skeleton("ؤ") == WAW       # hamza-on-waw


def test_skeleton_removes_standalone_hamza():
    assert skeleton("ءبجد") == "بجد"


def test_skeleton_folds_both_digit_ranges():
    assert skeleton("١٢٣") == "123"
    assert skeleton("۴۵۶") == "456"


def test_skeleton_removes_tatweel_punctuation_and_symbols():
    assert skeleton(f"بس{TATWEEL}م، (الله) ٪") == "بسم الله"


def test_skeleton_collapses_whitespace_and_strips():
    assert skeleton("  alif\t\t baa \n\n taa  ") == "alif baa taa"


def test_skeleton_is_idempotent():
    text = "الْحَمْدُ لِلَّهِ، ١٢٣ (ﻻ)"
    once = skeleton(text)
    assert skeleton(once) == once


def test_levels_are_progressively_looser():
    reference = "بِسْمِ اللهِ"
    prediction = "بسم الله"
    assert strict(reference) != strict(prediction)
    assert diacritic_insensitive(reference) == diacritic_insensitive(prediction)
    assert skeleton(reference) == skeleton(prediction)
