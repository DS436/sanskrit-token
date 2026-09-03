"""Tokens-per-proposition: the headline metric of this project.

Fertility asks how many tokens a *word* costs, which punishes Sanskrit for the very
property under study — sandhi and compounding pack into one orthographic word what English
spreads over a clause (CLAUDE.md §2.1). TPP asks how many tokens a *unit of meaning*
costs, by holding the meaning constant: on translation pairs, `texts[i]` and
`pivot_texts[i]` say the same thing, so `sum(tokens(sanskrit)) / sum(tokens(english))` is
the token cost of the same propositions expressed two ways (CLAUDE.md §7). Below 1.0 means
Sanskrit's density survived tokenization; above 1.0 means the tokenizer destroyed it.

Arithmetically this is `parity` — both are the shared ratio core in `_ratio.py` — and the
difference is what the number is being asked. Parity holds *content* constant on FLORES to
measure fairness; TPP holds *meaning* constant on translation corpora (Sāmayik prose
first, Itihāsa verse second, per CLAUDE.md §2.7) to measure density. TPP adds the bootstrap
CI, because it is the number a claim is made on.

The bootstrap is a paired one over the ratio of sums: each draw resamples pair *indices*
with replacement and recomputes `sum(source) / sum(pivot)` over the resampled pairs. Two
consequences are deliberate. Resampling indices keeps the two sides aligned — a draw takes
both halves of a pair or neither — so the interval reflects sentence-to-sentence variation
rather than noise from mismatching translations. And recomputing the ratio of sums, rather
than averaging the per-pair ratios, matches how `value` itself is computed: a mean of
ratios lets a two-token sentence count as heavily as a fifty-token one, which for a corpus
of verse quarters and long prose periods is a materially different number.

The draws reuse the token counts from `RatioParts`, so a 1000-draw bootstrap costs no
extra tokenization. Seed and draw count are returned so `results.json` records exactly
what produced the interval (CLAUDE.md §8).
"""

import math
from collections.abc import Sequence

import numpy as np

from sanskrit_tok.metrics._ratio import RatioParts, token_ratio
from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer

__all__ = ["tpp"]

#: Unit label carried into `results.json` and every figure axis.
UNIT = "tokens/proposition ratio"


def _bootstrap_ci(
    parts: RatioParts,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> tuple[float, float]:
    """Percentile CI of the ratio of sums, over `n_bootstrap` resamples of the pairs.

    Returns `(nan, nan)` when there is nothing to resample (no pairs, or `n_bootstrap` is
    0) and when every draw is undefined, i.e. every resample landed on pairs whose pivot
    side has no tokens. A draw that is individually undefined is dropped from the
    percentile rather than counted as zero.
    """
    n = len(parts.source_counts)
    if n == 0 or n_bootstrap <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    source = np.asarray(parts.source_counts, dtype=float)
    pivot = np.asarray(parts.pivot_counts, dtype=float)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    source_sums = source[indices].sum(axis=1)
    pivot_sums = pivot[indices].sum(axis=1)
    draws = np.divide(
        source_sums,
        pivot_sums,
        out=np.full(n_bootstrap, math.nan),
        where=pivot_sums != 0.0,
    )
    if not bool(np.isfinite(draws).any()):
        return math.nan, math.nan
    tail = (1.0 - ci) / 2.0
    low, high = np.nanpercentile(draws, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(low), float(high)


def tpp(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    pivot_texts: Sequence[str],
    pivot_tokenizer: Tokenizer | None = None,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> DetailedMetricResult:
    """Tokens-per-proposition of `texts` against `pivot_texts`, aligned by index.

    `texts[i]` and `pivot_texts[i]` are translations of each other; `texts` is the side
    whose cost is being measured (Sanskrit) and `pivot_texts` the side it is measured
    against (English, tokenized with o200k as the primary pivot). `pivot_tokenizer`
    defaults to `tokenizer`; pass a different one — the usual case here — to score each
    language with the tokenizer it would really be used with (CLAUDE.md §7).

    Returns `value` = total source tokens / total pivot tokens, `n` = the number of aligned
    pairs, `unit` = `"tokens/proposition ratio"`, `per_pair` = each pair's ratio in order,
    `n_undefined` = how many of those are `nan`, `source_tokens` / `pivot_tokens` = the two
    totals behind `value`, `ci_low` / `ci_high` = the percentile bootstrap interval, and
    `n_bootstrap` / `seed` = the settings that produced it.

    Pooled, not averaged per pair, so `value` is generally not the mean of `per_pair`. A
    pair whose pivot side yields no tokens has no ratio and contributes `nan`, never `0.0`;
    `value` stays defined as long as some pivot sentence yields a token, and is `nan` when
    none does. `ci_low` and `ci_high` are `nan` when `n_bootstrap` is 0 or every draw is
    undefined.

    Raises `TypeError` if either side is a single `str` rather than a sequence of them,
    and `ValueError` if the two sequences differ in length or if `ci` is not strictly
    between 0 and 1. `ci` is a confidence *level* (`0.95`), not a percentage (`95`) and
    not a tail probability (`0.05`); every out-of-range value produces an interval that
    looks plausible in `results.json` — `95` gives percentiles far outside `[0, 100]`,
    which `numpy` clamps to the extremes, and `0.05` gives a needle-thin interval that
    reads as a very precise measurement — so this is checked rather than trusted.
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci must be a confidence level strictly between 0 and 1, got {ci!r}")
    parts = token_ratio(tokenizer, texts, pivot_texts, pivot_tokenizer)
    ci_low, ci_high = _bootstrap_ci(parts, n_bootstrap, seed, ci)
    return {
        "value": parts.value,
        "n": len(parts.source_counts),
        "unit": UNIT,
        "per_pair": parts.per_pair,
        "n_undefined": parts.n_undefined,
        "source_tokens": parts.source_total,
        "pivot_tokens": parts.pivot_total,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
