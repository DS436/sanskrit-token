"""Tests for the calibrated line-quality rules of `sanskrit_tok.data.quality`.

Every case here is hand-computed: a short line whose SLP1 form, letter count and word
count can be checked by reading, so a threshold that drifts fails a test rather than
quietly changing a corpus. The rules and their thresholds are docs/decisions.md,
2026-09-05, "Sangraha quality filter calibrated on a sample; Hindi-line heuristic;
source-side normalisation"; nothing here reads a corpus or the network.
"""

import pytest

from sanskrit_tok.data.quality import (
    MIN_LETTER_FRACTION,
    QUALITY_RULES,
    TYPOGRAPHIC_PUNCTUATION,
    check_quality,
    has_latin,
    is_clean_slp1,
    looks_hindi,
    mean_word_length,
    normalise_source,
    normalise_typographic_punctuation,
    passes_quality,
    real_words,
    slp1_letter_fraction,
)

#: U+FF71 HALFWIDTH KATAKANA LETTER A — the mojibake the review found in DCS-derived text.
#: It is a *letter* to Python, so only `is_clean_slp1` catches it.
MOJIBAKE = "ﾱ"


# ------------------------------------------------------------------------- rule 1: Latin


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("रामः वनं गच्छति ।", False),
        ("रामः vanam गच्छति ।", True),
        ("Ramah vanam gacchati", True),
        ("रामः १२३ ।", False),  # Devanagari digits are not Latin letters
        (f"रामः {MOJIBAKE} वनं", False),  # nor is halfwidth katakana
    ],
)
def test_has_latin(line: str, expected: bool) -> None:
    assert has_latin(line) is expected


# ------------------------------------------------------------------------- rule 2: Hindi


def test_looks_hindi_needs_two_distinct_markers() -> None:
    """One marker is not evidence; two distinct ones are."""
    # `का` is also the Sanskrit feminine nominative singular of *ka*, so alone it says
    # nothing — this is why the threshold is two.
    assert looks_hindi("का कन्या वनं गच्छति") is False
    # The same marker twice is still one marker.
    assert looks_hindi("का वनं का गच्छति") is False
    assert looks_hindi("राम का पुत्र है वनं गच्छति") is True


def test_looks_hindi_matches_whole_tokens_only() -> None:
    """A Sanskrit word that merely contains a marker's letters is not a marker.

    `केवलम्` begins with `के` and `हो` occurs inside `होता`; matching substrings would
    make the heuristic fire on ordinary Sanskrit.
    """
    assert looks_hindi("केवलम् होता वनं गच्छति") is False


def test_looks_hindi_threshold_is_a_parameter() -> None:
    assert looks_hindi("का कन्या वनं गच्छति", min_markers=1) is True


# ------------------------------------------------------- rule 3: source-side normalisation


def test_normalise_source_strips_vedic_accents_and_the_abbreviation_sign() -> None:
    """U+0951 and `॰` go; the danda and every letter stay."""
    assert normalise_source("अ॑ग्निमी॑ळे ॰ ।") == "अग्निमीळे  ।"


def test_normalise_source_keeps_matras_and_the_virama() -> None:
    """A vowel sign spells a vowel: deleting it would change the word, not de-notate it."""
    line = "रामः वनं गच्छति ।"
    assert normalise_source(line) == line


def test_normalise_source_is_identity_when_there_is_nothing_to_strip() -> None:
    assert normalise_source("") == ""
    assert normalise_source("नृपः नगरं गच्छति") == "नृपः नगरं गच्छति"


# --------------------------------------------------------------- rule 4: letter fraction


def test_slp1_letter_fraction_ignores_trailing_punctuation() -> None:
    """A short line ending in a danda must not be penalised for ending in a danda.

    `tat .` is three letters and one danda. Counting the danda gives 3/4 = 0.75, below the
    0.85 threshold, which would drop every short well-formed sentence in the corpus; the
    trailing strip is what the review added to stop that.
    """
    assert slp1_letter_fraction("tat .") == 1.0
    assert slp1_letter_fraction("tat .") >= MIN_LETTER_FRACTION


def test_slp1_letter_fraction_counts_digits_and_internal_punctuation() -> None:
    """`rAma 12` after the trailing danda goes: four letters of six characters."""
    assert slp1_letter_fraction("rAma 12 .") == pytest.approx(4 / 6)


def test_slp1_letter_fraction_of_nothing_is_zero() -> None:
    assert slp1_letter_fraction("") == 0.0
    assert slp1_letter_fraction("... ..") == 0.0


# ------------------------------------------------------- rules 5 and 6: words and lengths


def test_real_words_excludes_punctuation_only_tokens() -> None:
    """A danda is not a word — in either script.

    This is also the base splitter's one-word-line gap: `"राम ।".split()` is two tokens
    and one word.
    """
    assert real_words("rAmaH . vanam ..") == ["rAmaH", "vanam"]
    assert real_words("राम ।") == ["राम"]
    assert real_words("। ॥ ०१") == []


def test_mean_word_length_is_over_real_words() -> None:
    assert mean_word_length("rAmaH vanam") == 5.0
    # The danda is not averaged in: two words of five, not three tokens of 5, 5 and 1.
    assert mean_word_length("rAmaH vanam .") == 5.0
    assert mean_word_length(".") == 0.0


# -------------------------------------------------------------------- rule 7: clean SLP1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rAmaH vanaM gacCati.", True),
        ("kzatriyaH 12 ' .. |~", True),  # digits and every SLP1 sign are ASCII punctuation
        (f"rAmaH {MOJIBAKE}", False),
        ("rAmaH ṃ", False),  # an IAST character that never became SLP1
        ("rAmaH V", False),  # `V` and `Z` are the two ASCII letters SLP1 does not use
        ("", True),
    ],
)
def test_is_clean_slp1(text: str, expected: bool) -> None:
    assert is_clean_slp1(text) is expected


# ------------------------------------------------------------------------- the pipeline


def test_passes_quality_accepts_ordinary_sanskrit() -> None:
    assert passes_quality("नृपः नगरं गच्छति ।") == (True, None)
    assert check_quality("नृपः नगरं गच्छति ।").text_slp1 == "nfpaH nagaraM gacCati ."


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("रामः vanam गच्छति ।", "latin"),
        ("राम का पुत्र है वनं गच्छति", "hindi"),
        # 17 letters of 24 non-space characters = 0.708, below 0.85.
        ("रामः ३४५६७ वनं ८९ गच्छति", "letter_fraction"),
        # Two words and a danda: the danda is not the third word.
        ("रामः वनम् ।", "n_words"),
        # Three words of one character: mean 1.0, below the floor of 3.
        ("अ इ उ", "word_length"),
        (f"रामः {MOJIBAKE} वनं गच्छति", "non_slp1"),
    ],
)
def test_passes_quality_names_the_failing_rule(line: str, rule: str) -> None:
    assert passes_quality(line) == (False, rule)
    assert rule in QUALITY_RULES


def test_check_quality_carries_the_slp1_text_of_a_line_it_transliterated() -> None:
    """A source-side rejection has no SLP1 text; a later one does, so nothing converts twice."""
    assert check_quality("रामः vanam गच्छति ।").text_slp1 == ""
    assert check_quality(f"रामः {MOJIBAKE} वनं गच्छति").text_slp1 == f"rAmaH {MOJIBAKE} vanaM gacCati"


def test_vedic_accents_do_not_by_themselves_drop_a_line() -> None:
    """The whole point of `normalise_source`: an accented Ṛgvedic line is good Sanskrit."""
    assert passes_quality("अ॑ग्निमी॒ळे पुरोहितं यज्ञस्य देवमृत्विजम् ।") == (True, None)


# ------------------------------------------------- typographic punctuation normalisation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("‘tat’", "'tat'"),
        ("‚tat’", "'tat'"),
        ("“tat”", '"tat"'),
        ("„tat”", '"tat"'),
        ("a–b", "a-b"),
        ("a—b", "a-b"),
        ("a‒b", "a-b"),
        ("tat…", "tat..."),
        ("tat\u00a0api", "tat api"),  # non-breaking space
        ("tat api", "tat api"),
        ("", ""),
    ],
)
def test_normalise_typographic_punctuation(text: str, expected: str) -> None:
    assert normalise_typographic_punctuation(text) == expected


def test_normalise_typographic_punctuation_is_idempotent() -> None:
    once = normalise_typographic_punctuation("“tat—api…”")
    assert normalise_typographic_punctuation(once) == once


def test_normalise_typographic_punctuation_leaves_devanagari_and_slp1_alone() -> None:
    assert normalise_typographic_punctuation("नृपः नगरं गच्छति ।") == "नृपः नगरं गच्छति ।"
    assert normalise_typographic_punctuation("nfpaH nagaraM gacCati .") == "nfpaH nagaraM gacCati ."


def test_normalise_typographic_punctuation_makes_a_curly_quoted_line_clean() -> None:
    """The whole point: these lines were dropped by `is_clean_slp1` for their typesetting."""
    line = "“nfpaH nagaraM gacCati” — iti"
    assert not is_clean_slp1(line)
    assert is_clean_slp1(normalise_typographic_punctuation(line))


def test_normalise_typographic_punctuation_does_not_rescue_a_non_slp1_letter() -> None:
    """A candra vowel has no SLP1 phoneme; normalisation must not silently drop it."""
    line = normalise_typographic_punctuation("“kaॉ”")
    assert not is_clean_slp1(line)


def test_typographic_table_maps_only_punctuation() -> None:
    assert not any(character.isalpha() for character in TYPOGRAPHIC_PUNCTUATION)
    assert not any(replacement.isalpha() for replacement in TYPOGRAPHIC_PUNCTUATION.values())
