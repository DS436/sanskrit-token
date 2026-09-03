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

from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer, require_texts

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
    the number of aligned pairs, `unit` = `"token ratio"`, and `per_pair` = the ratio for
    each pair in order.

    Pooled, not averaged per pair, so `value` is generally not the mean of `per_pair`.
    A pair whose pivot side yields no tokens contributes `0.0` to `per_pair`; when the
    whole pivot side yields no tokens, `value` is `0.0`.

    Raises `TypeError` if either side is a single `str` rather than a sequence of them,
    and `ValueError` if the two sequences differ in length.
    """
    require_texts(texts)
    require_texts(pivot_texts)
    if len(texts) != len(pivot_texts):
        raise ValueError(
            "texts and pivot_texts must be aligned pairs of equal length: "
            f"got {len(texts)} and {len(pivot_texts)}"
        )
    pivot = tokenizer if pivot_tokenizer is None else pivot_tokenizer
    per_pair: list[float] = []
    total_tokens = 0
    total_pivot_tokens = 0
    for text, pivot_text in zip(texts, pivot_texts, strict=True):
        text_tokens = len(tokenizer.encode(text))
        pivot_text_tokens = len(pivot.encode(pivot_text))
        per_pair.append(text_tokens / pivot_text_tokens if pivot_text_tokens else 0.0)
        total_tokens += text_tokens
        total_pivot_tokens += pivot_text_tokens
    return {
        "value": total_tokens / total_pivot_tokens if total_pivot_tokens else 0.0,
        "n": len(per_pair),
        "unit": "token ratio",
        "per_pair": per_pair,
    }
