"""Tests for the intrinsic metrics: fertility, compression, parity.

Every expected number here is hand-computed from the fake tokenizers below, so the
tests pin the metric definitions rather than the implementation.
"""

import math

import pytest

from sanskrit_tok.metrics._ratio import RatioParts, token_ratio
from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility, fertility_against_reference
from sanskrit_tok.metrics.parity import parity
from sanskrit_tok.tokenizers.base import Tokenizer


class CharTokenizer:
    """One token per character, whitespace included. `len(encode(t)) == len(t)`."""

    name = "char"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class WordTokenizer:
    """One token per whitespace-delimited word. `len(encode(t)) == len(t.split())`."""

    name = "word"

    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


CHAR = CharTokenizer()
WORD = WordTokenizer()


# --- the Tokenizer protocol ------------------------------------------------------


@pytest.mark.parametrize("tokenizer", [CHAR, WORD])
def test_fakes_satisfy_the_tokenizer_protocol(tokenizer: Tokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(tokenizer.name, str)


def test_object_without_encode_is_not_a_tokenizer() -> None:
    class NotATokenizer:
        name = "nope"

    assert not isinstance(NotATokenizer(), Tokenizer)


# --- fertility -------------------------------------------------------------------


def test_fertility_hand_computed() -> None:
    """`"ab cde"` is 2 words; CharTokenizer gives 2 + 3 = 5 tokens, so 5/2 = 2.5."""
    result = fertility(CHAR, ["ab cde"])
    assert result["value"] == pytest.approx(2.5)
    assert result["n"] == 2
    assert result["unit"] == "tokens/word"
    assert result["per_word"] == [2, 3]


def test_fertility_encodes_each_word_independently_so_whitespace_is_not_counted() -> None:
    """6 characters, but only the 5 non-space ones are inside words."""
    assert sum(fertility(CHAR, ["ab cde"])["per_word"]) == 5


def test_fertility_pools_across_texts() -> None:
    """Pooled, not averaged per text: (2 + 3 + 1) tokens over 3 words."""
    result = fertility(CHAR, ["ab cde", "f"])
    assert result["value"] == pytest.approx(6 / 3)
    assert result["n"] == 3
    assert result["per_word"] == [2, 3, 1]


def test_fertility_is_one_for_a_word_tokenizer() -> None:
    result = fertility(WORD, ["ab cde", "f"])
    assert result["value"] == pytest.approx(1.0)
    assert result["n"] == 3
    assert result["per_word"] == [1, 1, 1]


def test_fertility_splits_on_arbitrary_whitespace() -> None:
    result = fertility(CHAR, ["  ab\tcde\n\nf  "])
    assert result["n"] == 3
    assert result["per_word"] == [2, 3, 1]


def test_fertility_counts_multibyte_characters_as_the_tokenizer_does() -> None:
    """Fertility counts tokens, not bytes: `ā` is one character, so one token."""
    result = fertility(CHAR, ["āb"])
    assert result["per_word"] == [2]
    assert result["value"] == pytest.approx(2.0)


def test_fertility_of_empty_input() -> None:
    """Nothing was measured at all, so `0.0` is not a lie about a ratio."""
    result = fertility(CHAR, [])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["unit"] == "tokens/word"
    assert result["per_word"] == []
    assert result["n_undefined"] == 0


def test_fertility_of_whitespace_only_text_has_no_words_and_is_undefined() -> None:
    """Real texts holding no words have no tokens-per-word ratio: `nan`, never `0.0`."""
    result = fertility(CHAR, ["", "   "])
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert result["per_word"] == []
    assert result["n_undefined"] == 0


def test_fertility_counts_words_that_encode_to_zero_tokens() -> None:
    """A vocabulary that deletes a word makes `value` optimistic; `n_undefined` shows it."""

    class DropsOneWord:
        name = "drops-one-word"

        def encode(self, text: str) -> list[int]:
            return [] if text == "drop" else [ord(character) for character in text]

    result = fertility(DropsOneWord(), ["ab drop cde"])
    assert result["per_word"] == [2, 0, 3]
    assert result["n"] == 3
    assert result["n_undefined"] == 1
    assert result["value"] == pytest.approx(5 / 3)


def test_fertility_n_undefined_is_zero_when_every_word_costs_tokens() -> None:
    assert fertility(CHAR, ["ab cde", "f"])["n_undefined"] == 0


def test_fertility_rejects_a_bare_str() -> None:
    """A `str` is a `Sequence[str]`, so this would otherwise be measured per character."""
    with pytest.raises(TypeError, match="single str"):
        fertility(CHAR, "ab cde")  # type: ignore[arg-type]


# --- fertility against a reference word count (the T4 primary denominator) --------


def test_fertility_against_reference_hand_computed() -> None:
    """The Experiment 03 case: `"abcd"` is one raw word; sandhi splitting makes it
    `"ab cd"`, which CharTokenizer costs 2 + 2 = 4 tokens. Divided by the RAW word count
    (1), that is 4.0 — the same denominator every unsplit arm is measured on
    (docs/decisions.md, "Fertility for split arms uses the raw word count as the primary
    denominator")."""
    result = fertility_against_reference(CHAR, ["ab cd"], ["abcd"])
    assert result["value"] == pytest.approx(4.0)
    assert result["n"] == 1
    assert result["unit"] == "tokens/reference word"
    assert result["per_text"] == [pytest.approx(4.0)]
    assert result["n_undefined"] == 0


def test_fertility_against_reference_is_nan_where_the_reference_has_no_words() -> None:
    """No reference word means no tokens-per-reference-word ratio: `nan`, never `0.0`."""
    result = fertility_against_reference(CHAR, ["ab cd"], ["   "])
    assert math.isnan(result["per_text"][0])
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert result["n_undefined"] == 1


def test_fertility_against_reference_pools_across_texts() -> None:
    """Pooled: (4 + 1) tokens over (1 + 2) reference words, not the mean of 4.0 and 0.5."""
    result = fertility_against_reference(CHAR, ["ab cd", "f"], ["abcd", "gh ij"])
    assert result["value"] == pytest.approx(5 / 3)
    assert result["n"] == 3
    assert result["per_text"] == [pytest.approx(4.0), pytest.approx(0.5)]


def test_fertility_against_reference_matches_plain_fertility_when_texts_are_the_reference() -> None:
    """With no splitting there is nothing to correct for, so the two agree exactly."""
    texts = ["ab cde", "f"]
    assert fertility_against_reference(CHAR, texts, texts)["value"] == pytest.approx(
        fertility(CHAR, texts)["value"]
    )


def test_fertility_against_reference_ignores_inter_word_whitespace() -> None:
    """Each word is encoded on its own, exactly as `fertility` does, so the inserted
    boundary spaces cost nothing and the numerator stays comparable across arms."""
    assert fertility_against_reference(CHAR, ["ab   cd"], ["abcd"])["value"] == pytest.approx(4.0)


def test_fertility_against_reference_of_empty_input() -> None:
    """Nothing measured at all, as opposed to text that happens to hold no words."""
    result = fertility_against_reference(CHAR, [], [])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["n_undefined"] == 0


def test_fertility_against_reference_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="aligned"):
        fertility_against_reference(CHAR, ["ab cd"], ["abcd", "ef"])


@pytest.mark.parametrize("bad_side", ["texts", "reference"])
def test_fertility_against_reference_rejects_a_bare_str(bad_side: str) -> None:
    with pytest.raises(TypeError, match="single str"):
        if bad_side == "texts":
            fertility_against_reference(CHAR, "ab cd", ["abcd"])  # type: ignore[arg-type]
        else:
            fertility_against_reference(CHAR, ["ab cd"], "abcd")  # type: ignore[arg-type]


# --- compression -----------------------------------------------------------------


def test_compression_hand_computed_ascii() -> None:
    """`"ab cde"` is 6 UTF-8 bytes and 6 CharTokenizer tokens, so 6/6 = 1.0."""
    result = compression(CHAR, ["ab cde"])
    assert result["value"] == pytest.approx(1.0)
    assert result["n"] == 6
    assert result["unit"] == "bytes/token"
    assert result["per_text"] == [pytest.approx(1.0)]


def test_compression_hand_computed_multibyte() -> None:
    """`"āb"` is 3 UTF-8 bytes (`ā` is 2) and 2 tokens, so 3/2 = 1.5."""
    result = compression(CHAR, ["āb"])
    assert result["value"] == pytest.approx(1.5)
    assert result["n"] == 2
    assert result["per_text"] == [pytest.approx(1.5)]


def test_compression_encodes_each_text_whole_so_whitespace_counts() -> None:
    """Unlike fertility, the space is both a byte and a token here."""
    assert compression(CHAR, ["ab cde"])["n"] == 6
    assert compression(WORD, ["ab cde"])["n"] == 2


def test_compression_pools_across_texts() -> None:
    """(6 + 3) bytes over (6 + 2) tokens = 1.125, not the mean of 1.0 and 1.5."""
    result = compression(CHAR, ["ab cde", "āb"])
    assert result["value"] == pytest.approx(9 / 8)
    assert result["n"] == 8
    assert result["per_text"] == [pytest.approx(1.0), pytest.approx(1.5)]


def test_compression_with_a_coarser_tokenizer() -> None:
    """WordTokenizer: 6 bytes over 2 tokens = 3.0."""
    result = compression(WORD, ["ab cde"])
    assert result["value"] == pytest.approx(3.0)
    assert result["n"] == 2
    assert result["per_text"] == [pytest.approx(3.0)]


def test_compression_of_empty_input() -> None:
    result = compression(CHAR, [])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["unit"] == "bytes/token"
    assert result["per_text"] == []


def test_compression_of_text_yielding_no_tokens_is_nan_not_zero() -> None:
    """`0.0` is a plausible bytes/token value, so an undefined ratio must not read as one."""
    result = compression(CHAR, [""])
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert math.isnan(result["per_text"][0])
    assert result["n_undefined"] == 1


def test_compression_marks_only_the_undefined_texts_as_nan() -> None:
    result = compression(CHAR, ["ab cde", ""])
    assert result["value"] == pytest.approx(1.0)
    assert result["per_text"][0] == pytest.approx(1.0)
    assert math.isnan(result["per_text"][1])
    assert result["n_undefined"] == 1


def test_compression_of_empty_input_is_zero_not_nan() -> None:
    """Nothing was measured, so there is no undefined ratio to report either."""
    result = compression(CHAR, [])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["n_undefined"] == 0


def test_compression_rejects_a_bare_str() -> None:
    with pytest.raises(TypeError, match="single str"):
        compression(CHAR, "ab cde")  # type: ignore[arg-type]


# --- parity ----------------------------------------------------------------------


def test_parity_hand_computed() -> None:
    """3 tokens for `"abc"` over 2 for `"ab"` = 1.5."""
    result = parity(CHAR, ["abc"], ["ab"])
    assert result["value"] == pytest.approx(1.5)
    assert result["n"] == 1
    assert result["unit"] == "token ratio"
    assert result["per_pair"] == [pytest.approx(1.5)]


def test_parity_defaults_the_pivot_tokenizer_to_the_first() -> None:
    assert parity(CHAR, ["abc"], ["ab"]) == parity(CHAR, ["abc"], ["ab"], pivot_tokenizer=CHAR)


def test_parity_with_a_different_tokenizer_on_each_side() -> None:
    """WordTokenizer on the left gives 3 tokens; CharTokenizer pivot gives 4. 3/4."""
    result = parity(WORD, ["ab cde fg"], ["abcd"], pivot_tokenizer=CHAR)
    assert result["value"] == pytest.approx(0.75)
    assert result["n"] == 1
    assert result["per_pair"] == [pytest.approx(0.75)]


def test_parity_pools_across_pairs() -> None:
    """(3 + 1) over (2 + 4) = 2/3, while the per-pair ratios are 1.5 and 0.25."""
    result = parity(CHAR, ["abc", "d"], ["ab", "efgh"])
    assert result["value"] == pytest.approx(4 / 6)
    assert result["n"] == 2
    assert result["per_pair"] == [pytest.approx(1.5), pytest.approx(0.25)]


def test_parity_of_identical_sides_is_one() -> None:
    result = parity(CHAR, ["abc", "de"], ["abc", "de"])
    assert result["value"] == pytest.approx(1.0)
    assert result["per_pair"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_parity_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="aligned"):
        parity(CHAR, ["abc", "de"], ["ab"])
    with pytest.raises(ValueError, match="aligned"):
        parity(CHAR, ["abc"], ["ab", "cd"])


def test_parity_of_empty_input() -> None:
    result = parity(CHAR, [], [])
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert result["unit"] == "token ratio"
    assert result["per_pair"] == []
    assert result["n_undefined"] == 0


def test_parity_with_no_pivot_tokens_is_nan_not_zero() -> None:
    """The whole pivot side is empty here, so the pooled ratio is undefined too."""
    result = parity(CHAR, ["abc"], [""])
    assert math.isnan(result["value"])
    assert result["n"] == 1
    assert math.isnan(result["per_pair"][0])
    assert result["n_undefined"] == 1


def test_parity_keeps_the_pooled_value_when_only_some_pivots_are_empty() -> None:
    """(3 + 2) over (0 + 4): one pair is undefined, the corpus-level ratio is not."""
    result = parity(CHAR, ["abc", "de"], ["", "fghi"])
    assert result["value"] == pytest.approx(5 / 4)
    assert math.isnan(result["per_pair"][0])
    assert result["per_pair"][1] == pytest.approx(0.5)
    assert result["n_undefined"] == 1
    assert result["source_tokens"] == 5
    assert result["pivot_tokens"] == 4


def test_parity_rejects_a_bare_str_on_either_side() -> None:
    """Both sides are checked: `parity(tok, ["abc"], "abc")` compared 1 text with 3."""
    with pytest.raises(TypeError, match="single str"):
        parity(CHAR, "abc", ["abc"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="single str"):
        parity(CHAR, ["abc"], "abc")  # type: ignore[arg-type]


# --- the shared ratio core -------------------------------------------------------


def test_ratio_parts_hand_computed() -> None:
    parts = RatioParts(source_counts=(3, 1), pivot_counts=(2, 4))
    assert parts.source_total == 4 and parts.pivot_total == 6
    assert parts.value == 4 / 6
    assert parts.per_pair == [1.5, 0.25]
    assert parts.n_undefined == 0


def test_ratio_parts_marks_undefined_pairs_as_nan() -> None:
    parts = RatioParts(source_counts=(3, 2), pivot_counts=(0, 4))
    assert math.isnan(parts.per_pair[0]) and parts.per_pair[1] == 0.5
    assert parts.n_undefined == 1
    assert parts.value == 5 / 4  # pooled ratio still defined


def test_token_ratio_uses_pivot_tokenizer_and_checks_lengths() -> None:
    parts = token_ratio(CharTokenizer(), ["abc"], ["ab"], pivot_tokenizer=WordTokenizer())
    assert parts.source_counts == (3,) and parts.pivot_counts == (1,)
    with pytest.raises(ValueError):
        token_ratio(CharTokenizer(), ["a", "b"], ["a"])


# --- purity ----------------------------------------------------------------------


def test_metrics_do_not_mutate_their_inputs() -> None:
    texts = ["ab cde", "āb"]
    pivot = ["abc", "d"]
    fertility(CHAR, texts)
    compression(CHAR, texts)
    parity(CHAR, texts, pivot)
    assert texts == ["ab cde", "āb"]
    assert pivot == ["abc", "d"]
