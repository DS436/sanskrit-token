"""Line-level quality rules for the OCR-heavy web sources of corpus M1.

Sangraha's `verified/san` is OCR of printed books and Sanskrit Wikipedia is edited by
hand but quotes freely from other languages; neither had ever been filtered when Task 1
assembled corpus M1. The Task 1 review sampled 201 Sangraha lines and found the planned
thresholds partly inert and partly mis-placed, and the calibrated rules that replaced them
are recorded in docs/decisions.md, 2026-09-05, "Sangraha quality filter calibrated on a
sample; Hindi-line heuristic; source-side normalisation". This module is those rules and
nothing else: every function here is pure, so each threshold can be checked against a
hand-computed example rather than against a corpus.

The rules run in a fixed order, three of them on the **Devanagari source line** and three
on its SLP1 form, because each is only expressible on one side:

1. `has_latin` — a Devanagari line containing ASCII letters is a bibliography entry, a
   page header or a bilingual gloss. It has to be tested *before* transliteration: SLP1 is
   itself ASCII, so after conversion "no Latin letters" is vacuously false for every line.
2. `looks_hindi` — Hindi and Sanskrit share the script, and Sangraha's `san` directory
   contains Hindi commentary on Sanskrit texts. Two distinct function words out of
   `HINDI_MARKERS` is the heuristic; it is counted and reported, never silent.
3. `normalise_source` — strip the Devanagari signs `to_slp1` would pass through unchanged
   (Vedic accents U+0951/U+0952, the abbreviation sign `॰`, cantillation marks), so they do
   not later count against the line as "non-SLP1 characters". The danda is kept: it is a
   character the tokenizer must learn, and it maps to SLP1 `.`.
4. `slp1_letter_fraction` — how much of the line is Sanskrit phonemes rather than digits,
   punctuation and mojibake. Trailing punctuation is removed first so that a short line
   ending in a danda is not penalised for ending in a danda.
5. `real_words` — words with at least one letter. A danda is not a word; neither is a verse
   number. This is also the fix for the base splitter's one-word-line gap: `piece.split()`
   counts `राम ।` as two words, and it is one.
6. `mean_word_length` — a line of run-together OCR (one 60-character "word") or of
   scattered fragments (`k z a`) has a mean far outside the observed 7.97.

Then `is_clean_slp1`, which is applied to *every* corpus line, not only the web sources:
anything left outside the SLP1 alphabet, ASCII digits, ASCII punctuation and the space is
a character no arm's vocabulary should have to spend an id on. It is what removes the
`ﾱ` mojibake that reached DCS-derived text through the IAST conversion.

`passes_quality` runs 1-6 and `is_clean_slp1` in that order and names the first rule that
failed, so the caller can keep a per-rule drop count. `check_quality` is the same thing
returning the SLP1 text as well, so a caller does not transliterate twice.
"""

import string
import unicodedata
from dataclasses import dataclass
from typing import Final

from sanskrit_tok.encoding import to_slp1

__all__ = [
    "DANDA",
    "DOUBLE_DANDA",
    "HINDI_MARKERS",
    "MAX_MEAN_WORD_LENGTH",
    "MAX_WORDS",
    "MIN_LETTER_FRACTION",
    "MIN_MEAN_WORD_LENGTH",
    "MIN_WORDS",
    "QUALITY_RULES",
    "SLP1_LETTERS",
    "STRIPPED_DEVANAGARI_SIGNS",
    "QualityResult",
    "check_quality",
    "has_latin",
    "is_clean_slp1",
    "looks_hindi",
    "mean_word_length",
    "normalise_source",
    "passes_quality",
    "real_words",
    "slp1_letter_fraction",
]

DANDA: Final = "।"
DOUBLE_DANDA: Final = "॥"

#: The SLP1 alphabet: one ASCII character per Sanskrit phoneme, case-significant (`a` is
#: short *a* and `A` long *ā*; `S`, `z` and `s` are three sibilants). `M` (anusvāra) and
#: `H` (visarga) are included — they are phonemes, and `str.isalpha()` is true of them, so
#: this set agrees with `exclusion.letters_only` on what a letter is. `V` and `Z` are the
#: only ASCII letters SLP1 does not use.
SLP1_LETTERS: Final[frozenset[str]] = frozenset(
    "aAiIuUfFxXeEoOMHkKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshL"
)

#: The non-letter SLP1 signs, all of them ASCII punctuation: `'` avagraha, `.` danda
#: (`..` double danda), `~` candrabindu, `|` Vedic *ḻh*. Listed for the reader; the
#: membership test in `is_clean_slp1` is `string.punctuation`, which contains all four.
_SLP1_PUNCTUATION: Final = "'.~|"

#: Everything `is_clean_slp1` admits besides the SLP1 letters: ASCII digits (SLP1 writes
#: Devanagari digits as ASCII ones), ASCII punctuation and the space. A line is
#: newline-free by the time it is checked — the corpora are newline-delimited files.
_CLEAN_NON_LETTERS: Final[frozenset[str]] = frozenset(
    string.digits + string.punctuation + " "
)

#: Hindi function words, as whitespace tokens of the Devanagari source line. Every one is a
#: closed-class form that Sanskrit does not have: the postpositions and copulas Hindi uses
#: where Sanskrit uses a case ending, which is precisely why their presence is evidence
#: about the *language* rather than about the topic.
HINDI_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "का",
        "के",
        "की",
        "है",
        "हैं",
        "में",
        "से",
        "को",
        "और",
        "पर",
        "ने",
        "था",
        "थी",
        "थे",
        "हो",
    }
)

#: Calibrated on the review's 201-line sample (docs/decisions.md, 2026-09-05).
MIN_LETTER_FRACTION: Final = 0.85
MIN_WORDS: Final = 3
MAX_WORDS: Final = 40
MIN_MEAN_WORD_LENGTH: Final = 3.0
MAX_MEAN_WORD_LENGTH: Final = 14.0

#: The rule names `passes_quality` can return, in the order they are applied. A caller
#: keeping per-rule drop counts initialises its counters from this, so a rule that never
#: fires still appears in the manifest as a zero rather than as a missing key.
QUALITY_RULES: Final[tuple[str, ...]] = (
    "latin",
    "hindi",
    "letter_fraction",
    "n_words",
    "word_length",
    "non_slp1",
)

#: Unicode general categories that make a pass-through Devanagari character a *sign*
#: rather than a letter: combining marks (Mn, Mc), punctuation (Po), modifier letters (Lm)
#: and symbols. Base letters are never stripped — a letter SLP1 cannot express (candra e,
#: RHA) must reach `is_clean_slp1` and drop its line, not be silently deleted from a word.
_SIGN_CATEGORIES: Final[frozenset[str]] = frozenset({"Mn", "Mc", "Po", "Lm", "So", "Sk"})

#: A sign whose Unicode name contains this spells a vowel, so deleting it would change the
#: word rather than remove a notation from it. Such a sign is left in place and its line is
#: dropped by `is_clean_slp1` instead — the candra vowels are Hindi and English-loanword
#: notation that SLP1 has no phoneme for, and quietly turning `कॉ` into `क` would put a
#: word in the corpus that was never in the source.
_VOWEL_SIGN: Final = "VOWEL SIGN"

#: The Devanagari block and Devanagari Extended.
_DEVANAGARI_RANGES: Final[tuple[tuple[int, int], ...]] = ((0x0900, 0x097F), (0xA8E0, 0xA8FF))


def _devanagari_signs_passed_through() -> frozenset[str]:
    """Devanagari signs that survive `to_slp1` unchanged, dandas and vowel signs excepted.

    Computed rather than listed, because the set is a property of the transliteration
    table and would drift silently if it were copied out of it. A character qualifies when
    (a) `to_slp1(character, "devanagari")` returns the character itself — the scheme has no
    SLP1 spelling for it — (b) its Unicode category is a sign category, so no letter and no
    digit is ever in the set, and (c) it is not a vowel sign (see `_VOWEL_SIGN`). With
    `indic_transliteration` as pinned this is 30 characters, among them U+0951/U+0952 (the
    Vedic stress signs), U+0970 `॰` (the abbreviation sign), U+093C (nukta) and the
    U+A8E0-U+A8F1 cantillation marks; the dandas are not in it, since they transliterate to
    `.` and `..`.
    """
    signs: set[str] = set()
    for start, end in _DEVANAGARI_RANGES:
        for codepoint in range(start, end + 1):
            character = chr(codepoint)
            if character in (DANDA, DOUBLE_DANDA):
                continue
            if unicodedata.category(character) not in _SIGN_CATEGORIES:
                continue
            if _VOWEL_SIGN in unicodedata.name(character, ""):
                continue
            if to_slp1(character, "devanagari") == character:
                signs.add(character)
    return frozenset(signs)


STRIPPED_DEVANAGARI_SIGNS: Final[frozenset[str]] = _devanagari_signs_passed_through()


# ------------------------------------------------------------- the Devanagari-side rules


def has_latin(line: str) -> bool:
    """Whether `line` contains an ASCII letter.

    Only meaningful *before* transliteration: SLP1 is written in ASCII letters, so asking
    this of an SLP1 line always says yes. That asymmetry is the reason the rule order in
    `passes_quality` is fixed rather than a matter of taste.
    """
    return any("a" <= character <= "z" or "A" <= character <= "Z" for character in line)


def looks_hindi(line: str, min_markers: int = 2) -> bool:
    """Whether `line` has at least `min_markers` distinct `HINDI_MARKERS` tokens.

    A heuristic, and deliberately a blunt one. `का` also occurs in Sanskrit (the feminine
    nominative singular of *ka*), so one marker proves nothing; two distinct ones in a
    single line effectively do not happen in Sanskrit prose, while Hindi cannot manage a
    sentence without them. Matching is on whole whitespace tokens, so a Sanskrit word that
    merely *contains* these letters is not a marker.
    """
    found: set[str] = set()
    for token in line.split():
        if token in HINDI_MARKERS:
            found.add(token)
            if len(found) >= min_markers:
                return True
    return False


def normalise_source(devanagari_line: str) -> str:
    """`devanagari_line` with `STRIPPED_DEVANAGARI_SIGNS` removed, and nothing else.

    Not a normal form: no NFC/NFD, no case folding, no whitespace handling, no danda
    rewriting. The one job is to delete the marks that `to_slp1` would carry through into
    the SLP1 text, where they would count as non-SLP1 characters and drop the line. A Vedic
    accent is not a defect in the text; it is a notation the SLP1 alphabet has no room for.
    """
    if not any(character in STRIPPED_DEVANAGARI_SIGNS for character in devanagari_line):
        return devanagari_line
    return "".join(
        character
        for character in devanagari_line
        if character not in STRIPPED_DEVANAGARI_SIGNS
    )


# ------------------------------------------------------------------- the SLP1-side rules


def real_words(text: str) -> list[str]:
    """The whitespace tokens of `text` that contain at least one letter.

    A danda (`.` in SLP1, `।` in Devanagari), a verse number and a bare bracket are not
    words. `str.isalpha()` is the test, the same one `exclusion.letters_only` and
    `sangraha.has_devanagari_letter` use, so the notion of "letter" is one notion across
    the project and the function works on either side of transliteration.
    """
    return [token for token in text.split() if any(character.isalpha() for character in token)]


def mean_word_length(text: str) -> float:
    """Mean character length of `text`'s `real_words`; `0.0` when there are none.

    Characters, not letters: an attached danda or bracket is part of what the tokenizer
    will see. The observed mean on the Sangraha sample is 7.97, and the band this is
    thresholded against (`MIN_MEAN_WORD_LENGTH`-`MAX_MEAN_WORD_LENGTH`) is wide enough that
    only the two OCR failure modes fall outside it: run-together pages and letter-spaced
    fragments.
    """
    words = real_words(text)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def slp1_letter_fraction(text: str) -> float:
    """Fraction of `text`'s non-space characters that are SLP1 letters; `0.0` when empty.

    Trailing punctuation and whitespace are removed first. Without that, `tat .` would
    score 3/4 = 0.75 and be dropped for ending in a sentence terminator, which is the one
    thing every well-formed Sanskrit line does; the review found the uncalibrated rule
    failing exactly there. Internal punctuation still counts against the line, because a
    line that is mostly brackets and dots is page furniture.

    The denominator is letters + digits + every other non-space character — the "non-SLP1
    characters" of the decision entry — so the number is the share of the line that is
    Sanskrit phonemes.
    """
    core = text.rstrip(string.punctuation + string.whitespace)
    considered = [character for character in core if not character.isspace()]
    if not considered:
        return 0.0
    letters = sum(1 for character in considered if character in SLP1_LETTERS)
    return letters / len(considered)


def is_clean_slp1(text: str) -> bool:
    """Whether every character of `text` is SLP1, an ASCII digit, ASCII punctuation or a space.

    The last gate, applied to every line of every corpus and every held-out file rather
    than only to the web sources. Mojibake reaches SLP1 text by more than one route — a
    `ﾱ` survived DCS's IAST conversion — and a held-out line carrying it would put
    characters in the BPC denominator that no arm's vocabulary can spell.
    """
    return all(
        character in SLP1_LETTERS or character in _CLEAN_NON_LETTERS for character in text
    )


# ------------------------------------------------------------------------- the pipeline


@dataclass(frozen=True)
class QualityResult:
    """The verdict on one source line, with the SLP1 text the rules were applied to.

    `text_slp1` is `""` when the line was rejected before transliteration (`latin`,
    `hindi`); otherwise it is the text a caller should write to the corpus, so nothing
    transliterates the same line twice.
    """

    ok: bool
    rule: str | None
    text_slp1: str


def check_quality(devanagari_line: str) -> QualityResult:
    """Apply every rule to `devanagari_line`, in order, and report the first failure.

    The order is the one the decision entry fixes: the two source-side rules, then
    `normalise_source` and transliteration, then the three SLP1-side thresholds, then
    `is_clean_slp1`. Each rule is cheap relative to the one after it, so the ordering is
    also the fast one — `has_latin` rejects on the first ASCII letter, and transliteration,
    the expensive step, runs only on what survives the two cheap tests.
    """
    if has_latin(devanagari_line):
        return QualityResult(False, "latin", "")
    if looks_hindi(devanagari_line):
        return QualityResult(False, "hindi", "")
    text_slp1 = to_slp1(normalise_source(devanagari_line), "devanagari")
    if slp1_letter_fraction(text_slp1) < MIN_LETTER_FRACTION:
        return QualityResult(False, "letter_fraction", text_slp1)
    if not MIN_WORDS <= len(real_words(text_slp1)) <= MAX_WORDS:
        return QualityResult(False, "n_words", text_slp1)
    if not MIN_MEAN_WORD_LENGTH <= mean_word_length(text_slp1) <= MAX_MEAN_WORD_LENGTH:
        return QualityResult(False, "word_length", text_slp1)
    if not is_clean_slp1(text_slp1):
        return QualityResult(False, "non_slp1", text_slp1)
    return QualityResult(True, None, text_slp1)


def passes_quality(devanagari_line: str) -> tuple[bool, str | None]:
    """`check_quality` reduced to `(passed, failing rule name or None)`."""
    result = check_quality(devanagari_line)
    return result.ok, result.rule
