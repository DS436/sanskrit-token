"""Split a token-count ratio into a length factor and a density factor.

A TPP ratio (`tpp.py`) says how many tokens the Sanskrit side of a parallel corpus costs
per token of the English side, and nothing about *why*. Two entirely different worlds give
the same 0.6: a Sanskrit side that writes the same propositions in far fewer characters at
the same tokens-per-character, and a Sanskrit side of ordinary length whose tokenizer
happens to spend fewer tokens per character than the English one does. Experiment 02's
verse result — Itihāsa, the one corpus where Sanskrit stays below 1.0 under the matched
control — is exactly the place that distinction matters, because a 19th-century verse
translation is a plausible source of a long *English* side, and that is a fact about the
translator rather than about Sanskrit.

The identity is arithmetic and exact:

    tokens_sa / tokens_en
        = (chars_sa / chars_en) * ((tokens_sa / chars_sa) / (tokens_en / chars_en))
        = char_ratio * density_ratio

`char_ratio` is how much *text* the Sanskrit side spends on the same content;
`density_ratio` is how expensively the two tokenizers charge for a character of their own
side. `decompose_ratio` returns both factors, the six counts behind them, and the `tpp`
they multiply to, so `results.json` carries the attribution rather than an argument about
it. Callers assert `char_ratio * density_ratio == tpp` to within floating-point tolerance;
the two sides are not bit-identical because they are different orders of the same three
divisions.

Bytes are carried alongside characters because they are not the same question. SLP1 is
ASCII, so on the Sanskrit side of this project's corpora bytes and characters coincide;
Devanagari is three bytes per character and English is one, so a byte ratio and a character
ratio disagree by a factor of three on the same sentences. Both are recorded, and the
`variant` a caller stores beside them says which script the characters were counted in.

Pure and count-only: nothing here tokenizes, reads a file, or knows what a corpus is.
"""

import math

__all__ = ["DECOMPOSITION_KEYS", "decompose_ratio"]

#: The keys `decompose_ratio` returns, in order. Named so a test (and `results.json`'s
#: reader) can assert the shape without repeating the literal list.
DECOMPOSITION_KEYS: tuple[str, ...] = (
    "chars_sa",
    "chars_en",
    "bytes_sa",
    "bytes_en",
    "tokens_sa",
    "tokens_en",
    "tokens_per_char_sa",
    "tokens_per_char_en",
    "char_ratio",
    "density_ratio",
    "tpp",
)


def _divide(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or `nan` when the denominator is 0.

    `nan` rather than `0.0` for the same reason `_ratio.py` gives: zero is a plausible
    ratio and would read as a measurement, where an empty side has no ratio at all.
    """
    return numerator / denominator if denominator else math.nan


def decompose_ratio(
    *,
    chars_sa: int,
    chars_en: int,
    bytes_sa: int,
    bytes_en: int,
    tokens_sa: int,
    tokens_en: int,
) -> dict[str, float | int]:
    """The character/density decomposition of one side-to-side token ratio.

    All six inputs are *totals over a whole corpus*, not per-sentence values: the ratio
    being decomposed is the pooled `sum(tokens_sa) / sum(tokens_en)` that `tpp` reports,
    so a decomposition of per-pair means would not multiply back to it.

    Returns the six counts unchanged plus `tokens_per_char_sa`, `tokens_per_char_en`,
    `char_ratio` (`chars_sa / chars_en`), `density_ratio` (`tokens_per_char_sa /
    tokens_per_char_en`) and `tpp` (`tokens_sa / tokens_en`). Every derived value is `nan`
    where its denominator is zero, so an empty corpus side produces a readable entry rather
    than a `ZeroDivisionError` in the middle of a run.

    Raises `ValueError` on a negative count, which is not a measurement this can come from
    and would silently invert the reading of both factors.
    """
    counts = {
        "chars_sa": chars_sa,
        "chars_en": chars_en,
        "bytes_sa": bytes_sa,
        "bytes_en": bytes_en,
        "tokens_sa": tokens_sa,
        "tokens_en": tokens_en,
    }
    negative = sorted(name for name, value in counts.items() if value < 0)
    if negative:
        raise ValueError(f"counts must be non-negative; got negative {', '.join(negative)}")

    tokens_per_char_sa = _divide(tokens_sa, chars_sa)
    tokens_per_char_en = _divide(tokens_en, chars_en)
    return {
        **counts,
        "tokens_per_char_sa": tokens_per_char_sa,
        "tokens_per_char_en": tokens_per_char_en,
        "char_ratio": _divide(chars_sa, chars_en),
        "density_ratio": _divide(tokens_per_char_sa, tokens_per_char_en),
        "tpp": _divide(tokens_sa, tokens_en),
    }
