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

`fertility_against_reference` exists because sandhi splitting moves the denominator.
Splitting inserts whitespace, so "tokens per whitespace word" on split text counts a
different thing from the same metric on raw text and the two are not comparable
(docs/decisions.md, "Fertility for split arms uses the raw word count as the primary
denominator"). It divides the split text's tokens by the *raw* sentence's word count,
which puts every arm — split or not — over the same denominator.
"""

import math
from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer, require_texts

__all__ = ["fertility", "fertility_against_reference"]


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


def fertility_against_reference(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    reference_texts: Sequence[str],
) -> DetailedMetricResult:
    """Tokens of `texts[i]` per whitespace word of `reference_texts[i]`, pooled.

    The Experiment 03 primary fertility for the sandhi-split (`T4`) arms: `texts` is the
    split sentence the arm actually tokenizes and `reference_texts` is the raw sentence it
    was split from, so the denominator is the same raw word count every unsplit arm is
    measured over (docs/decisions.md, "Fertility for split arms uses the raw word count as
    the primary denominator"). Without it, splitting would flatter a `T4` arm twice: once
    by giving its subwords an easier job, and once by inflating the word count it is
    divided by. The two sequences must be index-aligned translations of nothing — they are
    the same sentence in two segmentations.

    The numerator is counted exactly as `fertility` counts it: each whitespace unit of
    `texts[i]` is encoded on its own and the counts are summed, so the boundaries splitting
    inserted cost nothing themselves and the numerator stays comparable with the raw arms'.

    Returns `value` = total tokens / total reference words, `n` = total reference words,
    `unit` = `"tokens/reference word"`, `per_text` = each text's ratio in order, and
    `n_undefined` = how many of those are `nan`. Pooled, not averaged per text.

    A reference sentence with no words has no ratio and contributes `nan`, never `0.0`;
    `value` is `nan` when no reference sentence has a word in it, and `0.0` only for
    genuinely empty input (`texts == []`), matching `fertility`'s contract exactly.

    Raises `TypeError` if either side is a single `str` rather than a sequence of them,
    and `ValueError` if the two sequences differ in length.
    """
    require_texts(texts)
    require_texts(reference_texts)
    if len(texts) != len(reference_texts):
        raise ValueError(
            "texts and reference_texts must be aligned pairs of equal length: "
            f"got {len(texts)} and {len(reference_texts)}"
        )
    per_text: list[float] = []
    total_tokens = 0
    total_words = 0
    for text, reference in zip(texts, reference_texts, strict=True):
        tokens = sum(len(tokenizer.encode(word)) for word in text.split())
        words = len(reference.split())
        total_tokens += tokens
        total_words += words
        per_text.append(tokens / words if words else math.nan)
    if total_words:
        value = total_tokens / total_words
    else:
        value = math.nan if len(texts) else 0.0
    return {
        "value": value,
        "n": total_words,
        "unit": "tokens/reference word",
        "per_text": per_text,
        "n_undefined": sum(1 for ratio in per_text if math.isnan(ratio)),
    }
