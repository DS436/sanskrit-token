"""Rényi efficiency of a tokenizer's output distribution (CLAUDE.md §7).

Zouhar, Meister, Gastaldi, Du, Vieira, Sachan & Cotterell, "Tokenization and the Noiseless
Channel" (ACL 2023): treat a tokenized corpus as a channel and score the tokenizer by how
close its token-unigram distribution comes to using its support evenly. A tokenizer that
spends most of its mass on a handful of types is wasting the vocabulary it was given; the
paper reports Rényi efficiency at α > 1, which weights the head of the distribution more
heavily than Shannon entropy does, correlating better with downstream BLEU than
compression alone. This project measures it at α ∈ {2.5, 3} (outline; `renyi_alphas` in
each experiment's config).

**Secondary intrinsic, never a headline.** Cognetta, Zouhar, Moon & Okazaki, "Two
Counterexamples to Tokenization and the Noiseless Channel" (LREC-COLING 2024) construct
tokenizers whose Rényi efficiency is arbitrarily high while downstream performance is
unchanged or worse: appending never-used or trivially-split types reshapes the unigram
distribution without changing what the tokenizer does to the text. The number is therefore
reported alongside the others as a description of the token distribution, and the claims
of this project rest on tokens-per-proposition and BPC (CLAUDE.md §2.1).

**Normalisation: the observed support, `K` = the number of distinct types that actually
occur.** This follows the reference implementation, `tokenization-scorer`
(github.com/zouharvi/tokenization-scorer), verified 2026-09-07 at
`tokenization_scorer/metrics.py`, whose `get_prob_distribution` returns
`vocab_size = len(words_freqs)` — the length of the type-count table built from the text —
and whose `_renyi_efficiency` divides by `np.log2(vocab_size)`, with an optional `vocab`
keyword commented "override observed vocabulary size". Normalising by the *nominal*
vocabulary size instead makes the number depend on how many types the vocabulary holds but
never emits, which is exactly the quantity Cognetta et al. exploit; it is still reported,
as `efficiency_nominal`, because arms in this project are compared at matched vocabulary
sizes (CLAUDE.md §2.5) and a reader may want the version that charges an arm for the types
it never uses.

Pure, like every module here: it counts what a tokenizer emits and does no I/O.
"""

import math
from collections import Counter
from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer, require_texts

__all__ = ["renyi_efficiency"]


def renyi_efficiency(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    alpha: float,
    vocab_size: int | None = None,
) -> DetailedMetricResult:
    """Rényi efficiency at order `alpha` of the token unigram distribution over `texts`.

    Each text is encoded whole (as in `compression`, unlike `fertility`'s word-by-word
    encoding) and the token ids are pooled into one distribution: with `c_i` the count of
    type `i`, `N` the total tokens and `p_i = c_i / N`,

        entropy_bits = log2(sum_i p_i ** alpha) / (1 - alpha)
        value        = entropy_bits / log2(K)

    where `K` is the number of distinct types observed. `value` is the headline of this
    function and is in `[0, 1]`, 1 meaning every observed type is equiprobable. `n` is `N`,
    the token count, so it is the same denominator `compression` reports.

    `efficiency_nominal` is the same entropy over `log2(vocab_size)` when a `vocab_size` is
    given — the arm's full id space, including types it never emits — and `nan` otherwise,
    or when `vocab_size` is 1 or smaller and the denominator does not exist. See the module
    docstring for why the observed support, not this, is `value`.

    Degenerate corpora are `nan` rather than 0.0, and say so in `n_undefined` (1 when
    `value` is undefined, 0 otherwise), so a consumer can tell a nan-by-construction from a
    measurement: a corpus with one token type has `entropy_bits` 0.0 and `log2(1) = 0`
    underneath it, and a corpus that yields no tokens at all has no distribution to score.
    Genuinely empty input is the one case that is not undefined — nothing was measured, so
    `value` is 0.0, `n` 0, `n_types` 0 and `n_undefined` 0.

    Raises `ValueError` for `alpha <= 0` and for `alpha == 1` (the Shannon limit, where the
    formula above is 0/0; the outline fixes α ∈ {2.5, 3} and Shannon efficiency is out of
    scope). Raises `TypeError` if `texts` is a single `str` rather than a sequence of them.
    """
    require_texts(texts)
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if alpha == 1.0:
        raise ValueError(
            "alpha must not be 1: Renyi entropy at alpha=1 is the Shannon limit, where "
            "log2(sum p**alpha) / (1 - alpha) is 0/0. This project measures alpha in "
            "{2.5, 3} (CLAUDE.md §7); Shannon efficiency is out of scope."
        )

    counts: Counter[int] = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text))
    total = sum(counts.values())
    n_types = len(counts)

    if total == 0:
        # Nothing was measured at all (no texts) versus texts that yielded no tokens: the
        # first is a 0.0 the same way an empty sum is, the second an undefined ratio.
        undefined = bool(texts)
        return _result(
            value=math.nan if undefined else 0.0,
            n=0,
            entropy_bits=0.0,
            n_types=0,
            n_undefined=1 if undefined else 0,
            alpha=alpha,
            vocab_size=vocab_size,
            entropy_for_nominal=0.0,
        )

    mass = math.fsum((count / total) ** alpha for count in counts.values())
    # `+ 0.0` turns IEEE's `-0.0` (from `log2(1.0) / (1 - alpha)` on a single-type corpus)
    # into positive zero, so results.json never carries a signed zero for "no entropy".
    entropy_bits = math.log2(mass) / (1.0 - alpha) + 0.0
    denominator = math.log2(n_types) if n_types > 1 else 0.0

    return _result(
        value=entropy_bits / denominator if denominator > 0 else math.nan,
        n=total,
        entropy_bits=entropy_bits,
        n_types=n_types,
        n_undefined=0 if denominator > 0 else 1,
        alpha=alpha,
        vocab_size=vocab_size,
        entropy_for_nominal=entropy_bits,
    )


def _result(
    *,
    value: float,
    n: int,
    entropy_bits: float,
    n_types: int,
    n_undefined: int,
    alpha: float,
    vocab_size: int | None,
    entropy_for_nominal: float,
) -> DetailedMetricResult:
    """Assemble the result dict, including the nominal-vocabulary efficiency.

    Separate only so the two returns above cannot drift in which keys they populate.
    """
    nominal = math.nan
    if vocab_size is not None and vocab_size > 1:
        nominal = entropy_for_nominal / math.log2(vocab_size)
    return {
        "value": value,
        "n": n,
        "unit": "renyi efficiency",
        "entropy_bits": entropy_bits,
        "n_types": n_types,
        "n_undefined": n_undefined,
        "alpha": alpha,
        "vocab_size": vocab_size,
        "efficiency_nominal": nominal,
    }
