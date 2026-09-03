"""Tests for the intrinsic metrics: fertility, compression, parity.

Every expected number here is hand-computed from the fake tokenizers below, so the
tests pin the metric definitions rather than the implementation.
"""

import pytest

from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility
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
    result = fertility(CHAR, [])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["unit"] == "tokens/word"
    assert result["per_word"] == []


def test_fertility_of_whitespace_only_text_has_no_words() -> None:
    result = fertility(CHAR, ["", "   "])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["per_word"] == []


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


def test_compression_of_text_yielding_no_tokens_is_zero_not_an_error() -> None:
    result = compression(CHAR, [""])
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["per_text"] == [0.0]


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
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["unit"] == "token ratio"
    assert result["per_pair"] == []


def test_parity_with_no_pivot_tokens_is_zero_not_an_error() -> None:
    result = parity(CHAR, ["abc"], [""])
    assert result["value"] == 0.0
    assert result["n"] == 1
    assert result["per_pair"] == [0.0]


# --- purity ----------------------------------------------------------------------


def test_metrics_do_not_mutate_their_inputs() -> None:
    texts = ["ab cde", "āb"]
    pivot = ["abc", "d"]
    fertility(CHAR, texts)
    compression(CHAR, texts)
    parity(CHAR, texts, pivot)
    assert texts == ["ab cde", "āb"]
    assert pivot == ["abc", "d"]
