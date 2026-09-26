# Contract: Normalisation Policy

**policy_version**: `1.0` | **Date**: 2026-09-16

This is a versioned contract, not an implementation detail. Two reports produced
under different `policy_version` values are not comparable, and `compare` refuses to
put them side by side (exit 3).

**Any change to any table on this page requires a `policy_version` bump.** Existing
runs are not migrated — they are re-scored or left alone with their old version
recorded.

---

## Level 1 — `strict` (headline)

**Transformation: none.** Zero operations are applied to reference or prediction.

Everything counts: diacritics, tatweel, Arabic and Persian letter variants,
punctuation, digit forms, whitespace, and Unicode representation form (composed vs
decomposed, presentation forms vs base letters).

This is simultaneously the raw unnormalised score FR-002 requires and the strict
headline FR-003 requires — they are the same number, reported once (research R-002).
It is the figure quoted whenever a single number is quoted (FR-003a).

**Display label**: human-readable reports MUST render this level as
**"strict (raw, unnormalised)"**, not bare "strict". The machine field stays
`strict`. FR-002 asks for a raw score alongside every normalised one, and a reader
scanning for the word "raw" would otherwise conclude it is missing — the collapse is
correct, but it has to be visible rather than inferred from the research log.

**The single I/O rule**: one trailing newline is stripped from every reference and
every prediction as the file is read. It applies symmetrically to both sides, it is
an artifact of how text files are written rather than of what the model produced,
and it is recorded in every run manifest as `trailing_newline_stripped: true`. No
other whitespace is touched.

---

## Level 2 — `diacritic_insensitive`

As strict, but vowelling is ignored. Nothing else changes — tatweel, letter
variants, digit forms, punctuation, whitespace and representation form all still
count.

**Pipeline, in order:**

1. Remove every codepoint in the diacritic set below.
2. Remove any remaining character of Unicode general category `Mn` that follows an
   Arabic-block base character. This catches NFD-decomposed forms, e.g. `U+0622`
   written as `U+0627 U+0653`.

**Diacritic set**

| Range | Contents |
|---|---|
| `U+064B`–`U+065F` | fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun, and the extended marks |
| `U+0670` | superscript alef |
| `U+06D6`–`U+06DC` | Quranic annotation marks |
| `U+06DF`–`U+06E4` | Quranic annotation marks |
| `U+06E7`, `U+06E8` | small yeh, small noon ghunna |
| `U+06EA`–`U+06ED` | empty centre / low stop marks |
| `U+08D3`–`U+08E1` | Arabic Extended-A marks |
| `U+08E3`–`U+08FF` | Arabic Extended-A marks |

`U+08E2` is deliberately excluded — it is a format character (`Cf`), not a mark.

---

## Level 3 — `skeleton`

Consonantal letters and their order only. Diacritics, tatweel, variant letter forms
and Unicode representation are all ignored.

**Pipeline, in order.** The order is part of the contract — these operations do not
commute.

1. **NFKC.** Folds presentation forms (`U+FB50`–`U+FDFF`, `U+FE70`–`U+FEFF`) back to
   base letters and decomposes ligatures into their parts.
2. **Remove all combining marks** — every character of category `Mn`.
3. **Remove tatweel** `U+0640`.
4. **Letter folding**, per the table below.
5. **Digit folding**: `U+0660`–`U+0669` and `U+06F0`–`U+06F9` to ASCII `0`–`9`.
6. **Remove punctuation and symbols** — every character of category `P*` or `S*`.
7. **Collapse whitespace**: each run of Unicode whitespace to a single `U+0020`;
   strip both ends.

**Letter folding table**

| From | To | Note |
|---|---|---|
| `U+0622` `U+0623` `U+0625` `U+0671` `U+0672` `U+0673` `U+0675` | `U+0627` | alef variants to bare alef |
| `U+0624` | `U+0648` | hamza-on-waw to waw |
| `U+0626` | `U+064A` | hamza-on-yeh to yeh |
| `U+0621` | *(removed)* | standalone hamza is not part of the rasm |
| `U+0649` | `U+064A` | alef maksura to yeh |
| `U+06CC` | `U+064A` | Persian yeh to Arabic yeh |
| `U+06A9` | `U+0643` | keheh to Arabic kaf |
| `U+0629` | `U+0647` | teh marbuta to heh |
| `U+06C0` `U+06C1` `U+06C2` | `U+0647` | heh variants to heh |

---

## Tokenisation (all levels)

Words are runs of non-whitespace, split on Unicode whitespace after that level's
normalisation, with empty tokens discarded. No punctuation splitting, no
morphological segmentation (research R-006).

At skeleton level punctuation has already been removed by step 6, so skeleton WER is
the closest thing here to a content-word metric.

---

## What the policy does *not* do

Stated explicitly, because each is a plausible thing for a reader to assume:

- It does not apply NFC or NFKC at strict or diacritic-insensitive level. A model
  emitting presentation forms is producing output a downstream consumer must handle,
  and FR-003 requires that to show up in the headline.
- It does not strip conversational framing. A prediction reading
  "Here is the text I can see: ..." scores as wrong at every level, which is the
  intended answer — the framing is output the model should not have produced.
- It does not reorder anything. Reading order is measured separately and explicitly
  (FR-012), never normalised away.
- It does not transliterate between scripts. A page mixing Arabic, Latin and digits
  is scored as written.

---

## Versioning rules

| Change | Consequence |
|---|---|
| Add or remove a codepoint from any table | Bump `policy_version`. |
| Reorder pipeline steps | Bump `policy_version`. |
| Add a fourth level | Bump `policy_version`. |
| Change tokenisation | Bump `policy_version`. |
| Fix a bug that changes any produced number | Bump `policy_version`. |
| Documentation wording only | No bump. |

The version string is embedded in `manifest.json`, `summary.json` and every
comparison, so a report always carries the rules that produced it (FR-004, SC-008).
