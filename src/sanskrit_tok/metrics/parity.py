"""Parity: the token-count ratio between two languages on the same content.

Petrov et al. 2023, "Language Model Tokenizers Introduce Unfairness Between Languages"
(NeurIPS 2023), measures tokenizer fairness on parallel text: because the two sides say
the same thing, any ratio away from 1.0 is a cost the tokenizer imposes on one language
rather than a difference in what was said. A ratio of 2.0 means the same content costs
twice as many tokens, and so twice the context window and twice the API price.

Content is held constant, not meaning-per-token, so parity is a fairness measure and not
a density measure. It is the FLORES-anchored diagnostic for RQ1; tokens-per-proposition
(`tpp.py`) is what tests whether Sanskrit's density survives tokenization.
"""

from collections.abc import Sequence

from sanskrit_tok.metrics._ratio import token_ratio
from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer

__all__ = ["parity"]


def parity(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    pivot_texts: Sequence[str],
    pivot_tokenizer: Tokenizer | None = None,
) -> DetailedMetricResult:
    """Token-count ratio of `texts` to `pivot_texts`, which must be aligned by index.

    `texts[i]` and `pivot_texts[i]` are translations of each other (FLORES devtest is
    aligned by line, per CLAUDE.md §5); the default pivot is `eng_Latn`, with `hin_Deva`
    reported alongside it. `pivot_tokenizer` defaults to `tokenizer`, which is the usual
    case of one tokenizer scored on two languages; pass a different one to compare two
    tokenizers on parallel text.

    Returns `value` = total tokens over `texts` / total tokens over `pivot_texts`, `n` =
    the number of aligned pairs, `unit` = `"token ratio"`, `per_pair` = the ratio for each
    pair in order, `n_undefined` = how many of those are `nan`, and `source_tokens` /
    `pivot_tokens` = the two totals behind `value`.

    Pooled, not averaged per pair, so `value` is generally not the mean of `per_pair`.
    A pair whose pivot side yields no tokens has no ratio and contributes `nan`, never
    `0.0`, which would be indistinguishable from a measured ratio; the pooled `value`
    stays defined as long as some pivot sentence yields a token, and is `nan` when none
    does. This is arithmetically `tpp` without the bootstrap: both are the shared ratio
    core of `_ratio.py`, differing in corpus and unit (CLAUDE.md §7).

    Raises `TypeError` if either side is a single `str` rather than a sequence of them,
    and `ValueError` if the two sequences differ in length.
    """
    parts = token_ratio(tokenizer, texts, pivot_texts, pivot_tokenizer)
    return {
        "value": parts.value,
        "n": len(parts.source_counts),
        "unit": "token ratio",
        "per_pair": parts.per_pair,
        "n_undefined": parts.n_undefined,
        "source_tokens": parts.source_total,
        "pivot_tokens": parts.pivot_total,
    }
