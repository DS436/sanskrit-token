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

import math
from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer, require_texts

__all__ = ["fertility"]


def fertility(tokenizer: Tokenizer, texts: Sequence[str]) -> DetailedMetricResult:
    """Mean tokens per word, pooled over `texts`.

    Each word is encoded independently, so inter-word whitespace contributes no tokens and
    the count is unaffected by how a tokenizer treats space-prefixed pieces.

    Returns `value` = total tokens / total words, `n` = total words, `unit` =
    `"tokens/word"`, `per_word` = the token count of each word in order across all texts
    (the distribution, for variance and for the long tail of compounds), and
    `n_undefined` = how many of those words encode to zero tokens.

    Pooled, not averaged per text: long sentences weigh proportionally more.

    Text with no words in it has no tokens-per-word ratio, so `value` is `nan` — matching
    `compression` and `parity`, where an undefined ratio is never `0.0`, since `0.0` is
    itself a plausible measurement (a tokenizer that drops its input entirely). The one
    exception is genuinely empty input, where nothing was measured at all: `texts == []`
    gives `value` `0.0`, `n` `0` and `n_undefined` `0`, while `["", "   "]` — real texts
    that happen to hold no words — gives `nan`.

    A word that encodes to zero tokens is counted in `n_undefined` but contributes a
    perfectly defined `0` to `per_word` and to the pooled numerator: unlike a ratio, a
    token count of zero is a measurement, not a missing one. The count is a diagnostic —
    a vocabulary that silently deletes a word (rare, but possible for a control character
    or an unmapped script) makes `value` optimistic, and nothing else in the result would
    show it.

    Raises `TypeError` if `texts` is a single `str` rather than a sequence of them.
    """
    require_texts(texts)
    per_word = [len(tokenizer.encode(word)) for text in texts for word in text.split()]
    total_words = len(per_word)
    total_tokens = sum(per_word)
    if total_words:
        value = total_tokens / total_words
    else:
        value = math.nan if len(texts) else 0.0
    return {
        "value": value,
        "n": total_words,
        "unit": "tokens/word",
        "per_word": per_word,
        "n_undefined": sum(1 for count in per_word if count == 0),
    }
