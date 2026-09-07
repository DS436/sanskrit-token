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

__all__ = ["tpp", "tpp_from_parts", "tpp_paired_delta"]

#: Unit label carried into `results.json` and every figure axis.
UNIT = "tokens/proposition ratio"

#: Unit label for the *difference* between two TPP ratios (`tpp_paired_delta`).
UNIT_DELTA = "delta tokens/proposition ratio"


def _require_ci(ci: float) -> None:
    """`ci` must be a confidence *level* strictly inside `(0, 1)`; see `tpp`'s docstring."""
    if not 0 < ci < 1:
        raise ValueError(f"ci must be a confidence level strictly between 0 and 1, got {ci!r}")


def _resampled_ratios(parts: RatioParts, indices: np.ndarray, n_bootstrap: int) -> np.ndarray:
    """Ratio of sums for each row of `indices`, `nan` where the pivot side sums to zero.

    The one place a bootstrap draw is turned into a ratio, shared by the single-ratio CI
    (`_bootstrap_ci`) and the paired-difference CI (`_bootstrap_delta_ci`) so the two
    cannot drift in how they pool a resample or what they do with an undefined one.
    """
    source = np.asarray(parts.source_counts, dtype=float)
    pivot = np.asarray(parts.pivot_counts, dtype=float)
    source_sums = source[indices].sum(axis=1)
    pivot_sums = pivot[indices].sum(axis=1)
    ratios: np.ndarray = np.divide(
        source_sums,
        pivot_sums,
        out=np.full(n_bootstrap, math.nan),
        where=pivot_sums != 0.0,
    )
    return ratios


def _percentile_ci(draws: np.ndarray, ci: float) -> tuple[float, float]:
    """Two-sided percentile interval of `draws`, `(nan, nan)` if no draw is defined.

    Also shared by both bootstraps, for the same reason: `ci` is a confidence *level*, and
    turning it into a pair of percentiles in two places is two chances to get the tails
    wrong in one of them.
    """
    if not bool(np.isfinite(draws).any()):
        return math.nan, math.nan
    tail = (1.0 - ci) / 2.0
    low, high = np.nanpercentile(draws, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(low), float(high)


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
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    return _percentile_ci(_resampled_ratios(parts, indices, n_bootstrap), ci)


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
    _require_ci(ci)
    parts = token_ratio(tokenizer, texts, pivot_texts, pivot_tokenizer)
    return tpp_from_parts(parts, n_bootstrap=n_bootstrap, seed=seed, ci=ci)


def tpp_from_parts(
    parts: RatioParts,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> DetailedMetricResult:
    """`tpp` on token counts that have already been computed; the same dict, key for key.

    `tpp` is this function preceded by a `token_ratio` call, so everything its docstring
    says about pooling, undefined pairs, the bootstrap and `ci` holds here unchanged.
    The split exists for analyses that measure *parts* of a corpus: a length stratum
    (Experiment 02's `tpp_by_length`) is `tpp_from_parts(parts.subset(indices), ...)`,
    which reuses the counts from one `token_ratio` pass over the whole corpus instead of
    re-encoding it once per stratum — cheaper, and it guarantees the strata and the whole
    are the same measurement.

    Note that `seed` is used as given for every call, so two strata of the same corpus
    share a bootstrap seed. That is intended: the draws are independent resamples of
    different index sets, and reproducing a stratum's interval should need only the seed
    `results.json` records beside it.
    """
    _require_ci(ci)
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


def _bootstrap_delta_ci(
    counts_a: RatioParts,
    counts_b: RatioParts,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> tuple[float, float]:
    """Percentile CI of `ratio(a) - ratio(b)` over `n_bootstrap` joint resamples.

    One set of pair indices per draw, used on *both* sides: the two arms are measured on
    the same sentences, so a draw must take those sentences out of both or the difference
    picks up noise from two different corpora rather than from the two arms. A draw whose
    pivot side sums to zero on either side is `nan` and is dropped from the percentile.
    """
    n = len(counts_a.source_counts)
    if n == 0 or n_bootstrap <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    draws_a = _resampled_ratios(counts_a, indices, n_bootstrap)
    draws_b = _resampled_ratios(counts_b, indices, n_bootstrap)
    return _percentile_ci(draws_a - draws_b, ci)


def tpp_paired_delta(
    counts_a: RatioParts,
    counts_b: RatioParts,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float | int | str]:
    """The difference between two TPP ratios measured on the same pairs, with a paired CI.

    Experiment 03's success criterion is not "the split arm's TPP is low" but "the split
    arm's TPP is *lower than its matched raw arm's*, by more than sampling noise", so the
    quantity that needs an interval is the difference itself. `counts_a` is the side the
    delta is reported for (the `T4` split arm) and `counts_b` the side it is compared
    against (its matched `T1`/`T2` raw arm): `delta = ratio(a) - ratio(b)`, so a negative
    delta whose interval excludes 0 is the result H3 predicts.

    Both sides come from `token_ratio` over the *same* sentences of the same corpus,
    normally against the same English pivot, and the bootstrap resamples one set of pair
    indices for both (`_bootstrap_delta_ci`). Differencing two independently-resampled
    intervals would be a different, wider and wrong quantity: most of the variation is
    shared between the two arms, and pairing removes it.

    Returns `value` (= `delta`, so the CLAUDE.md §7 metric contract holds), `delta`,
    `value_a` and `value_b` (the two ratios behind it), `n` = the number of pairs,
    `unit`, `ci_low`/`ci_high` = the percentile interval, and `n_bootstrap`/`seed`/`ci` =
    the settings that produced it.

    `delta` is `nan` when either ratio is undefined (a pivot side with no tokens at all);
    `ci_low`/`ci_high` are `nan` when `n_bootstrap` is 0, there are no pairs, or every draw
    is undefined. Raises `ValueError` if the two sides hold different numbers of pairs —
    they would then not be the same sentences, and nothing here would be paired — or if
    `ci` is not a confidence level strictly between 0 and 1.
    """
    _require_ci(ci)
    if len(counts_a.source_counts) != len(counts_b.source_counts):
        raise ValueError(
            "counts_a and counts_b must cover the same pairs: got "
            f"{len(counts_a.source_counts)} and {len(counts_b.source_counts)}"
        )
    value_a, value_b = counts_a.value, counts_b.value
    ci_low, ci_high = _bootstrap_delta_ci(counts_a, counts_b, n_bootstrap, seed, ci)
    delta = value_a - value_b
    return {
        "value": delta,
        "delta": delta,
        "value_a": value_a,
        "value_b": value_b,
        "n": len(counts_a.source_counts),
        "unit": UNIT_DELTA,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci": ci,
    }
