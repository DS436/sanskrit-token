"""Tests for `sanskrit_tok.metrics.morphscore` and the `TokenizerWithSpans` protocol.

Every expected number is hand-computed from the fake spans tokenizer below, so the tests
pin the MorphScore definition (Arnett & Bergen: token boundaries against gold morpheme
boundaries, single-token and single-morpheme words excluded) rather than the
implementation. Everything here runs offline.

`SpanFake` is deliberately built from explicit *pieces* rather than from a real tokenizer:
a hand-computed precision of 0.5 is only checkable if the segmentation it is computed from
is written out in the test.
"""

import math
from collections.abc import Sequence

import pytest

from sanskrit_tok.metrics.morphscore import morphscore
from sanskrit_tok.tokenizers.base import Tokenizer, TokenizerWithSpans, spans_cover_text


class SpanFake:
    """Segments each word according to a `word -> pieces` table; spans are cumulative.

    A word not in the table is returned whole (one token), which is how the "excluded
    because it is a single token" cases are written.
    """

    name = "span_fake"

    def __init__(self, pieces: dict[str, Sequence[str]]) -> None:
        self._pieces = pieces
        for word, parts in pieces.items():
            if "".join(parts) != word:
                raise AssertionError(f"pieces for {word!r} do not concatenate back to it")

    def _parts(self, text: str) -> Sequence[str]:
        return self._pieces.get(text, [text])

    def encode(self, text: str) -> list[int]:
        return [len(part) for part in self._parts(text)]

    def spans(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        cursor = 0
        for part in self._parts(text):
            out.append((cursor, cursor + len(part)))
            cursor += len(part)
        return out


#: `tadapi` = `tad` + `api`: the gold boundary is at character 3 (docs/decisions.md,
#: "Gold boundaries: segment boundaries from DCS").
PERFECT = SpanFake({"tadapi": ["tad", "api"]})
SHIFTED = SpanFake({"tadapi": ["ta", "dapi"]})
OVER = SpanFake({"tadapi": ["t", "ad", "api"]})
OFF_BY_ONE = SpanFake({"tadapi": ["tada", "pi"]})


# --- the protocol ----------------------------------------------------------------


def test_span_fake_satisfies_both_protocols() -> None:
    assert isinstance(PERFECT, Tokenizer)
    assert isinstance(PERFECT, TokenizerWithSpans)


def test_a_plain_tokenizer_is_not_a_tokenizer_with_spans() -> None:
    class CountOnly:
        name = "count_only"

        def encode(self, text: str) -> list[int]:
            return [len(text)]

    assert isinstance(CountOnly(), Tokenizer)
    assert not isinstance(CountOnly(), TokenizerWithSpans)


def test_spans_cover_text_accepts_a_tiling() -> None:
    assert spans_cover_text("tadapi", [(0, 3), (3, 6)])


def test_spans_cover_text_ignores_whitespace() -> None:
    assert spans_cover_text("tad api", [(0, 3), (4, 7)])


def test_spans_cover_text_rejects_a_gap_over_a_letter() -> None:
    assert not spans_cover_text("tadapi", [(0, 3), (4, 6)])


def test_spans_cover_text_rejects_an_overlap() -> None:
    assert not spans_cover_text("tadapi", [(0, 4), (3, 6)])


def test_spans_cover_text_allows_a_span_containing_whitespace() -> None:
    """The invariant is about *non-whitespace* characters. Real vocabularies contain
    multi-word tokens (o200k has `" in the"`), whose span necessarily holds a space."""
    assert spans_cover_text("tad api", [(0, 7)])


def test_the_fake_spans_cover_every_word_they_segment() -> None:
    for word in ("tadapi", "rAmaH"):
        assert spans_cover_text(word, OVER.spans(word))


# --- hand-computed MorphScore ----------------------------------------------------


def test_exact_boundary_scores_one() -> None:
    """`tad|api` puts its only boundary at 3, which is the only gold boundary."""
    result = morphscore(PERFECT, ["tadapi"], [[3]])
    assert result["value"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["n"] == 1
    assert result["unit"] == "F1"
    assert result["n_matched"] == 1
    assert result["n_token_boundaries"] == 1
    assert result["n_gold_boundaries"] == 1
    assert result["per_word_f1"] == [pytest.approx(1.0)]
    assert result["tolerance"] == 0
    assert result["n_undefined"] == 0
    assert result["n_skipped_unaligned"] == 0
    assert result["n_excluded_single_token"] == 0
    assert result["n_excluded_single_morpheme"] == 0


def test_a_boundary_in_the_wrong_place_scores_zero() -> None:
    """`ta|dapi` puts its only boundary at 2; gold is 3, so nothing matches."""
    result = morphscore(SHIFTED, ["tadapi"], [[3]])
    assert result["value"] == pytest.approx(0.0)
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)
    assert result["n"] == 1
    assert result["n_matched"] == 0
    assert result["per_word_f1"] == [pytest.approx(0.0)]


def test_over_segmentation_keeps_recall_and_halves_precision() -> None:
    """`t|ad|api` has boundaries {1, 3}: one of two is gold, and the one gold boundary is
    found, so P = 1/2, R = 1/1 and F1 = 2 * 0.5 * 1 / 1.5 = 2/3."""
    result = morphscore(OVER, ["tadapi"], [[3]])
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(2 / 3)
    assert result["n_token_boundaries"] == 2
    assert result["n_gold_boundaries"] == 1
    assert result["n_matched"] == 1


def test_single_token_word_is_excluded() -> None:
    """`rAmaH` is not in the fake's table, so it is one token and cannot be scored."""
    result = morphscore(PERFECT, ["rAmaH"], [[4]])
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert result["n_excluded_single_token"] == 1
    assert result["n_excluded_single_morpheme"] == 0
    assert result["per_word_f1"] == []


def test_single_morpheme_word_is_excluded() -> None:
    """A word with no gold boundary inside it has nothing to score against."""
    result = morphscore(PERFECT, ["tadapi"], [[]])
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert result["n_excluded_single_morpheme"] == 1
    assert result["n_excluded_single_token"] == 0


def test_a_word_that_is_both_single_token_and_single_morpheme_counts_under_both() -> None:
    result = morphscore(PERFECT, ["rAmaH"], [[]])
    assert result["n"] == 0
    assert result["n_excluded_single_token"] == 1
    assert result["n_excluded_single_morpheme"] == 1


def test_unaligned_word_is_skipped() -> None:
    """`align_segments` returns `None` for a word it could not align; those words are
    skipped rather than scored as a miss."""
    result = morphscore(PERFECT, ["tadapi", "tadapi"], [None, [3]])
    assert result["n_skipped_unaligned"] == 1
    assert result["n"] == 1
    assert result["value"] == pytest.approx(1.0)


def test_a_mix_pools_over_words_not_over_per_word_f1() -> None:
    """Three scored words: `tad|api` (1 boundary, 1 match), `t|ad|api` (2, 1) and
    `ta|dapi` (1, 0), against one gold boundary each.

    Pooled: P = 2/4 = 0.5, R = 2/3, F1 = 2 * 0.5 * (2/3) / (0.5 + 2/3) = 4/7.
    The mean of the per-word F1s (1, 2/3, 0) is 5/9, which is deliberately different.
    """
    mixed = SpanFake(
        {
            "tadapi": ["tad", "api"],
            "tadapiX": ["t", "ad", "apiX"],
            "tadapiY": ["ta", "dapiY"],
        }
    )
    result = morphscore(mixed, ["tadapi", "tadapiX", "tadapiY"], [[3], [3], [3]])
    assert result["n"] == 3
    assert result["n_token_boundaries"] == 4
    assert result["n_gold_boundaries"] == 3
    assert result["n_matched"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["value"] == pytest.approx(4 / 7)
    assert result["per_word_f1"] == [
        pytest.approx(1.0),
        pytest.approx(2 / 3),
        pytest.approx(0.0),
    ]
    assert result["value"] != pytest.approx(5 / 9)


# --- the tolerant variant --------------------------------------------------------


def test_off_by_one_boundary_misses_exactly_and_matches_within_one() -> None:
    """`tada|pi` cuts at 4 where gold is 3 — the vowel-sandhi case of docs/decisions.md,
    "MorphScore on sandhied surfaces reports exact and +/-1-character variants"."""
    exact = morphscore(OFF_BY_ONE, ["tadapi"], [[3]])
    assert exact["value"] == pytest.approx(0.0)
    assert exact["precision"] == pytest.approx(0.0)
    assert exact["recall"] == pytest.approx(0.0)
    assert exact["tolerance"] == 0

    tolerant = morphscore(OFF_BY_ONE, ["tadapi"], [[3]], tolerance=1)
    assert tolerant["value"] == pytest.approx(1.0)
    assert tolerant["precision"] == pytest.approx(1.0)
    assert tolerant["recall"] == pytest.approx(1.0)
    assert tolerant["tolerance"] == 1
    assert tolerant["n_matched"] == 1


def test_tolerance_is_symmetric() -> None:
    """A cut one character *before* the gold boundary matches too; the correction entry
    in docs/decisions.md is explicit that the error runs in both directions."""
    early = SpanFake({"tadapi": ["ta", "dapi"]})
    assert morphscore(early, ["tadapi"], [[3]], tolerance=1)["value"] == pytest.approx(1.0)


def test_each_gold_boundary_is_matched_at_most_once() -> None:
    """`ta|d|api` has boundaries {2, 3}, both within one of the single gold boundary 3,
    but only one of them can claim it: P = 1/2, R = 1."""
    crowded = SpanFake({"tadapi": ["ta", "d", "api"]})
    result = morphscore(crowded, ["tadapi"], [[3]], tolerance=1)
    assert result["n_matched"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)


def test_two_offset_boundaries_can_match_two_offset_gold_boundaries() -> None:
    """Boundaries {3, 4} against gold {4, 5}, tolerance 1: 3 takes the nearest unclaimed
    gold (4) and 4 then takes 5, so both match and P = R = 1. Matching greedily from the
    left without releasing a claim is what makes this deterministic."""
    both = SpanFake({"tadapiXY": ["tad", "a", "piXY"]})
    result = morphscore(both, ["tadapiXY"], [[4, 5]], tolerance=1)
    assert result["n_matched"] == 2
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


# --- input validation ------------------------------------------------------------


def test_a_bare_string_is_rejected() -> None:
    with pytest.raises(TypeError):
        morphscore(PERFECT, "tadapi", [[3]])  # type: ignore[arg-type]


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        morphscore(PERFECT, ["tadapi"], [[3], [3]])


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError):
        morphscore(PERFECT, ["tadapi"], [[3]], tolerance=-1)


def test_empty_input_is_undefined_not_zero() -> None:
    result = morphscore(PERFECT, [], [])
    assert math.isnan(result["value"])
    assert math.isnan(result["precision"])
    assert math.isnan(result["recall"])
    assert result["n"] == 0


def test_gold_offset_zero_is_ignored() -> None:
    """Token boundaries never include 0 (every word starts one), so a gold 0 could never
    be matched and would only depress recall."""
    result = morphscore(PERFECT, ["tadapi"], [[0, 3]])
    assert result["n_gold_boundaries"] == 1
    assert result["value"] == pytest.approx(1.0)


def test_duplicate_gold_offsets_are_collapsed() -> None:
    result = morphscore(PERFECT, ["tadapi"], [[3, 3]])
    assert result["n_gold_boundaries"] == 1
    assert result["value"] == pytest.approx(1.0)
