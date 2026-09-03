"""Fertility: tokens per whitespace-delimited word.

**Fertility is never the headline metric** (CLAUDE.md §2.1). Sanskrit words are long
because sandhi erases word boundaries and compounds fuse many stems into one orthographic
word, so a corpus of Sanskrit has fewer, longer words than its English translation and
fertility punishes exactly the property under study. Always compute it, always report it,
never lead with it: the headline numbers are tokens-per-proposition and bits-per-character.

Words are whitespace-delimited (`str.split()`), which is the standard definition and the
one that makes the number comparable with published fertility figures. It is also why the
metric says nothing about morphology: one sandhied Sanskrit word may carry what English
writes as a whole clause.
"""

from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer, require_texts

__all__ = ["fertility"]


def fertility(tokenizer: Tokenizer, texts: Sequence[str]) -> DetailedMetricResult:
    """Mean tokens per word, pooled over `texts`.

    Each word is encoded independently, so inter-word whitespace contributes no tokens and
    the count is unaffected by how a tokenizer treats space-prefixed pieces.

    Returns `value` = total tokens / total words, `n` = total words, `unit` =
    `"tokens/word"`, and `per_word` = the token count of each word in order across all
    texts (the distribution, for variance and for the long tail of compounds).

    Pooled, not averaged per text: long sentences weigh proportionally more. With no words
    at all (empty input, or whitespace-only texts), `value` is `0.0` and `n` is `0`.

    Raises `TypeError` if `texts` is a single `str` rather than a sequence of them.
    """
    require_texts(texts)
    per_word = [len(tokenizer.encode(word)) for text in texts for word in text.split()]
    total_words = len(per_word)
    total_tokens = sum(per_word)
    return {
        "value": total_tokens / total_words if total_words else 0.0,
        "n": total_words,
        "unit": "tokens/word",
        "per_word": per_word,
    }
