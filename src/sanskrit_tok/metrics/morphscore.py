"""MorphScore: how well a tokenizer's token boundaries land on gold morpheme boundaries.

Arnett & Bergen's measure (*Why do language models perform worse for morphologically
complex languages?*, 2024), which is a precision/recall of *boundaries*, not of tokens:
each word is tokenised on its own, the character offsets where its tokens begin (excluding
0, which every word has) are compared against the gold morpheme boundaries inside it, and
the counts are pooled over the corpus. `value` is the F1, with `precision` and `recall` as
extras so over-segmentation and under-segmentation stay distinguishable (docs/decisions.md,
"MorphScore reports F1 as `value`, precision and recall as extras").

**The two exclusions, both from Arnett & Bergen.** A word the tokenizer emits as a single
token has no boundaries to be right or wrong about, and a word with no gold boundary inside
it has nothing to be scored against; both are excluded from precision and recall and
counted separately, because how much of a corpus a MorphScore was computed on is part of
the number. Without the first exclusion a tokenizer with a large enough vocabulary scores a
perfect precision by never splitting anything; without the second, one that never splits
scores a perfect *everything* on a corpus of monomorphemic words.

**Gold boundaries come from DCS** (CLAUDE.md §7), as character offsets into the word: the
segment boundaries the ingestion aligns into the sandhied surface, or the derived
stem/ending boundary, which is a heuristic and is labelled as one wherever it is reported
(docs/decisions.md, "Gold boundaries: segment boundaries from DCS…"). A word the ingestion
could not align carries `None` and is skipped — scoring it as a miss would charge the
tokenizer for the aligner's failure.

**`tolerance` exists because of sandhi.** Vowel sandhi fuses the last character of one
segment and the first of the next into a single surface character, so for 53% of
multi-segment DCS words there is no character offset that is *the* boundary: the aligner
picks one side, and which side it picks is a property of the alignment rather than of the
language (docs/decisions.md, "MorphScore on sandhied surfaces reports exact and
±1-character variants", and the correction entry that follows it, which establishes that
the error runs in *both* directions and so the tolerance must be symmetric). `tolerance=1`
counts a token boundary as matching a gold boundary within one character, each gold
boundary claimable at most once. The exact variant (`tolerance=0`, the default) is the one
paired arm comparisons use, since the offset convention is identical for every arm; the
tolerant variant bounds how much of an arm's exact score is the convention rather than the
tokenizer. Absolute MorphScore values from this corpus are not portable to studies on
languages without sandhi, and every table saying so is the point of recording both.
"""

import math
from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import (
    DetailedMetricResult,
    TokenizerWithSpans,
    require_texts,
)

__all__ = ["morphscore"]


def _token_boundaries(tokenizer: TokenizerWithSpans, word: str) -> tuple[list[int], int]:
    """The offsets where `word`'s tokens begin, 0 excluded, and how many tokens there were.

    0 is excluded because every word starts one and no tokenizer can get it wrong; counting
    it would add a free true positive to every word and inflate both precision and recall.
    """
    spans = tokenizer.spans(word)
    return [start for start, _ in spans if start != 0], len(spans)


def _match_count(token_boundaries: Sequence[int], gold: Sequence[int], tolerance: int) -> int:
    """How many token boundaries pair with a distinct gold boundary within `tolerance`.

    A one-to-one matching, so the count is simultaneously the true positives of precision
    and of recall: two token boundaries either side of one gold boundary cannot both claim
    it, and a tokenizer cannot buy recall by cutting everywhere.

    Greedy, nearest-first, token boundaries in ascending order, ties to the lower gold
    offset. Greedy is not optimal in general, but both sequences are sorted and tolerance
    is 0 or 1, so at most two token boundaries can ever contend for one gold boundary and
    the greedy choice is the optimal one; a larger tolerance would need a real assignment.
    """
    unclaimed = sorted(gold)
    matched = 0
    for boundary in sorted(token_boundaries):
        best: int | None = None
        best_distance = tolerance + 1
        for index, offset in enumerate(unclaimed):
            distance = abs(boundary - offset)
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            unclaimed.pop(best)
            matched += 1
    return matched


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean, `0.0` when both are zero (no matches is a score, not a missing one)."""
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def morphscore(
    tokenizer: TokenizerWithSpans,
    words: Sequence[str],
    gold_offsets: Sequence[Sequence[int] | None],
    *,
    tolerance: int = 0,
) -> DetailedMetricResult:
    """Boundary F1 of `tokenizer` against `gold_offsets`, pooled over `words`.

    `words[i]` is one whitespace-free word, tokenised on its own, and `gold_offsets[i]` its
    gold morpheme boundaries as character offsets into it — or `None` for a word with no
    gold segmentation, which is skipped. Offset 0 is dropped from the gold set (token
    boundaries never include it, so it could only depress recall) and duplicates are
    collapsed.

    A word is scored only if it has **at least two tokens** and **at least one gold
    boundary**; a word failing both tests is counted under both exclusions, so
    `n_excluded_single_token + n_excluded_single_morpheme` is not a partition and may
    exceed the number of words excluded.

    Returns `value` = pooled F1, `n` = words scored, `unit` = `"F1"`, and the extras
    `precision`, `recall`, `n_matched`, `n_token_boundaries`, `n_gold_boundaries`,
    `per_word_f1` (one F1 per scored word, in order), `tolerance`,
    `n_excluded_single_token`, `n_excluded_single_morpheme`, `n_skipped_unaligned` and
    `n_undefined`.

    **Pooled, not averaged.** Precision is `n_matched / n_token_boundaries` and recall is
    `n_matched / n_gold_boundaries` over the whole corpus, so a long compound weighs more
    than a two-morpheme word, and `value` is generally not the mean of `per_word_f1`.
    `per_word_f1` is the distribution, for variance and for finding the words an arm fails
    on; every entry is defined, since a scored word has at least one boundary on each side,
    so `n_undefined` is structurally 0 and is reported only to keep the
    `DetailedMetricResult` contract uniform.

    `value`, `precision` and `recall` are `nan` when no word could be scored — including
    for empty input. Unlike `fertility`, which reports `0.0` for genuinely empty input, an
    F1 of nothing is not zero and `0.0` is a perfectly plausible measured F1 (a tokenizer
    that never cuts where the morphology does), so the two must stay distinguishable.

    Raises `TypeError` if `words` is a single `str`, and `ValueError` if the two sequences
    differ in length or `tolerance` is negative.
    """
    require_texts(words)
    if len(words) != len(gold_offsets):
        raise ValueError(
            "words and gold_offsets must be aligned and of equal length: "
            f"got {len(words)} and {len(gold_offsets)}"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    per_word_f1: list[float] = []
    n_matched = 0
    n_token_boundaries = 0
    n_gold_boundaries = 0
    n_excluded_single_token = 0
    n_excluded_single_morpheme = 0
    n_skipped_unaligned = 0

    for word, offsets in zip(words, gold_offsets, strict=True):
        if offsets is None:
            n_skipped_unaligned += 1
            continue
        gold = sorted({offset for offset in offsets if offset != 0})
        boundaries, n_tokens = _token_boundaries(tokenizer, word)
        if n_tokens < 2:
            n_excluded_single_token += 1
        if not gold:
            n_excluded_single_morpheme += 1
        if n_tokens < 2 or not gold:
            continue

        matched = _match_count(boundaries, gold, tolerance)
        n_matched += matched
        n_token_boundaries += len(boundaries)
        n_gold_boundaries += len(gold)
        per_word_f1.append(_f1(matched / len(boundaries), matched / len(gold)))

    if per_word_f1:
        precision = n_matched / n_token_boundaries
        recall = n_matched / n_gold_boundaries
        value = _f1(precision, recall)
    else:
        precision = recall = value = math.nan

    return {
        "value": value,
        "n": len(per_word_f1),
        "unit": "F1",
        "precision": precision,
        "recall": recall,
        "per_word_f1": per_word_f1,
        "n_matched": n_matched,
        "n_token_boundaries": n_token_boundaries,
        "n_gold_boundaries": n_gold_boundaries,
        "n_excluded_single_token": n_excluded_single_token,
        "n_excluded_single_morpheme": n_excluded_single_morpheme,
        "n_skipped_unaligned": n_skipped_unaligned,
        "tolerance": tolerance,
        "n_undefined": sum(1 for score in per_word_f1 if math.isnan(score)),
    }
