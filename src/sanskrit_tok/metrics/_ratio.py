"""The shared core of every token-count ratio metric.

`parity` (content held constant, FLORES) and `tpp` (meaning held constant, translation
corpora) are the same arithmetic — `sum(tokens(a)) / sum(tokens(b))` over index-aligned
pairs, per CLAUDE.md §7 — differing only in the corpus they are run on, the unit they
report, and the bootstrap CI TPP adds. Keeping one core means the two cannot drift apart
in how they pool, how they align, or what they do with an undefined pair.

`RatioParts` deliberately stores the raw per-pair *counts* rather than the ratios. The
counts are what a bootstrap needs: resampling pairs and recomputing a ratio of sums must
not re-encode the corpus, and it must not be approximated by resampling the per-pair
ratios, which weighs a two-token sentence as heavily as a fifty-token one.

Undefined pairs (a pivot side that yields no tokens) are `nan`, never `0.0`: a zero is a
plausible ratio and would silently drag down any mean or CI computed over the list. The
pooled `value` stays defined as long as the pivot total is non-zero, because a single
empty pivot sentence does not make the corpus-level ratio undefined.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from sanskrit_tok.tokenizers.base import Tokenizer, require_texts

__all__ = ["RatioParts", "token_ratio"]


@dataclass(frozen=True)
class RatioParts:
    """Per-pair token counts for two index-aligned sides of a parallel corpus.

    `source_counts[i]` and `pivot_counts[i]` are the token counts of the two halves of
    pair `i`. The two tuples are the same length; every derived number below is a pure
    function of them, so a caller can resample indices without touching a tokenizer.
    """

    source_counts: tuple[int, ...]
    pivot_counts: tuple[int, ...]

    @property
    def source_total(self) -> int:
        """Total tokens over the source side."""
        return sum(self.source_counts)

    @property
    def pivot_total(self) -> int:
        """Total tokens over the pivot side."""
        return sum(self.pivot_counts)

    @property
    def value(self) -> float:
        """Pooled ratio `source_total / pivot_total`, or `nan` if the pivot total is 0.

        Pooled, not averaged per pair, so this is generally not the mean of `per_pair`:
        long sentences weigh proportionally more, which is the intended behaviour for a
        corpus-level cost.
        """
        return self.source_total / self.pivot_total if self.pivot_total else math.nan

    @property
    def per_pair(self) -> list[float]:
        """Ratio of each pair in order, `nan` where the pair's pivot side has no tokens."""
        return [
            source / pivot if pivot else math.nan
            for source, pivot in zip(self.source_counts, self.pivot_counts, strict=True)
        ]

    @property
    def n_undefined(self) -> int:
        """How many pairs have an undefined ratio, i.e. how many `per_pair` are `nan`."""
        return sum(1 for pivot in self.pivot_counts if not pivot)


def token_ratio(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    pivot_texts: Sequence[str],
    pivot_tokenizer: Tokenizer | None = None,
) -> RatioParts:
    """Token counts for `texts` and `pivot_texts`, which must be aligned by index.

    `pivot_tokenizer` defaults to `tokenizer`, which is the usual case of one tokenizer
    scored on two languages; pass a different one to compare two tokenizers on parallel
    text (CLAUDE.md §7 requires TPP to accept a different tokenizer per side).

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
    return RatioParts(
        source_counts=tuple(len(tokenizer.encode(text)) for text in texts),
        pivot_counts=tuple(len(pivot.encode(text)) for text in pivot_texts),
    )
