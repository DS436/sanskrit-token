"""Tests for tokens-per-proposition, the headline metric of CLAUDE.md §2.1 and §7.

Every expected number is hand-computed from the fake tokenizers below (duplicated from
`tests/test_metrics.py` so this module stands alone). The bootstrap is pinned by its
seed, by its degenerate cases and by determinism, not by its exact percentiles.
"""

import math

import pytest

from sanskrit_tok.metrics._ratio import RatioParts, token_ratio
from sanskrit_tok.metrics.tpp import block_count, tpp, tpp_from_parts, tpp_paired_delta


class CharTokenizer:
    """One token per character, whitespace included. `len(encode(t)) == len(t)`."""

    name = "char"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class WordTokenizer:
    """One token per whitespace-delimited word. `len(encode(t)) == len(t.split())`."""

    name = "word"

    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


def test_tpp_hand_computed_value_and_keys() -> None:
    r = tpp(CharTokenizer(), ["abc", "a"], ["ab", "abcd"], n_bootstrap=200, seed=0)
    assert r["value"] == 4 / 6 and r["n"] == 2 and r["unit"] == "tokens/proposition ratio"
    assert r["per_pair"] == [1.5, 0.25] and r["n_undefined"] == 0
    assert r["source_tokens"] == 4 and r["pivot_tokens"] == 6
    assert r["ci_low"] <= r["value"] <= r["ci_high"]
    assert r["n_bootstrap"] == 200 and r["seed"] == 0


def test_tpp_identical_sides_has_degenerate_ci() -> None:
    r = tpp(CharTokenizer(), ["ab", "cde"], ["ab", "cde"], n_bootstrap=50, seed=1)
    assert r["value"] == 1.0 and r["ci_low"] == 1.0 and r["ci_high"] == 1.0


def test_tpp_is_deterministic_for_a_seed() -> None:
    a = tpp(CharTokenizer(), ["abc", "a", "abcd"], ["ab", "abcd", "a"], n_bootstrap=100, seed=7)
    b = tpp(CharTokenizer(), ["abc", "a", "abcd"], ["ab", "abcd", "a"], n_bootstrap=100, seed=7)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_tpp_zero_bootstrap_gives_nan_ci() -> None:
    r = tpp(CharTokenizer(), ["abc"], ["ab"], n_bootstrap=0)
    assert math.isnan(r["ci_low"]) and math.isnan(r["ci_high"])


def test_tpp_accepts_different_pivot_tokenizer() -> None:
    r = tpp(
        WordTokenizer(), ["ab cde fg"], ["abcd"], pivot_tokenizer=CharTokenizer(), n_bootstrap=0
    )
    assert r["value"] == 0.75


def test_tpp_different_seeds_give_different_intervals() -> None:
    """Otherwise the seed would be recorded in `results.json` without doing anything."""
    pairs = (["abc", "a", "abcd", "ab"], ["ab", "abcd", "a", "abcde"])
    a = tpp(CharTokenizer(), *pairs, n_bootstrap=40, seed=0)
    b = tpp(CharTokenizer(), *pairs, n_bootstrap=40, seed=1)
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_tpp_pools_rather_than_averaging_the_per_pair_ratios() -> None:
    """4/6 = 0.667, while the mean of the per-pair ratios 1.5 and 0.25 is 0.875."""
    r = tpp(CharTokenizer(), ["abc", "a"], ["ab", "abcd"], n_bootstrap=0)
    assert r["value"] == pytest.approx(4 / 6)
    assert r["value"] != pytest.approx(sum(r["per_pair"]) / 2)


def test_tpp_marks_undefined_pairs_as_nan_and_counts_them() -> None:
    r = tpp(CharTokenizer(), ["abc", "de"], ["", "fghi"], n_bootstrap=0)
    assert math.isnan(r["per_pair"][0])
    assert r["per_pair"][1] == pytest.approx(0.5)
    assert r["n_undefined"] == 1
    assert r["value"] == pytest.approx(5 / 4)


def test_tpp_with_an_entirely_empty_pivot_side_is_nan_throughout() -> None:
    r = tpp(CharTokenizer(), ["abc"], [""], n_bootstrap=10, seed=0)
    assert math.isnan(r["value"])
    assert math.isnan(r["ci_low"]) and math.isnan(r["ci_high"])
    assert r["n"] == 1 and r["n_undefined"] == 1


def test_tpp_narrower_ci_at_a_lower_confidence_level() -> None:
    pairs = (["abc", "a", "abcd", "ab"], ["ab", "abcd", "a", "abcde"])
    wide = tpp(CharTokenizer(), *pairs, n_bootstrap=500, seed=3, ci=0.99)
    narrow = tpp(CharTokenizer(), *pairs, n_bootstrap=500, seed=3, ci=0.50)
    assert wide["ci_high"] - wide["ci_low"] >= narrow["ci_high"] - narrow["ci_low"]


def test_tpp_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="aligned"):
        tpp(CharTokenizer(), ["abc", "de"], ["ab"], n_bootstrap=0)


def test_tpp_rejects_a_bare_str_on_either_side() -> None:
    with pytest.raises(TypeError, match="single str"):
        tpp(CharTokenizer(), "abc", ["abc"], n_bootstrap=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="single str"):
        tpp(CharTokenizer(), ["abc"], "abc", n_bootstrap=0)  # type: ignore[arg-type]


def test_tpp_rejects_a_confidence_level_outside_the_unit_interval() -> None:
    """`ci` is a level (`0.95`), not a percentage (`95`) or a tail probability."""
    for bad in (0.0, 1.0, 95.0, -0.5, math.nan):
        with pytest.raises(ValueError, match="confidence level"):
            tpp(CharTokenizer(), ["abc"], ["ab"], n_bootstrap=10, ci=bad)


# --- RatioParts.subset and tpp_from_parts: TPP over part of a corpus ------------------


def test_subset_keeps_the_two_sides_aligned_and_in_the_given_order() -> None:
    """A stratum takes both halves of a pair or neither, and follows the caller's order."""
    parts = RatioParts(source_counts=(1, 2, 3, 4), pivot_counts=(10, 20, 30, 40))
    picked = parts.subset([2, 0])
    assert picked.source_counts == (3, 1)
    assert picked.pivot_counts == (30, 10)


def test_subset_of_an_out_of_range_index_raises() -> None:
    with pytest.raises(IndexError):
        RatioParts(source_counts=(1,), pivot_counts=(2,)).subset([1])


def test_tpp_from_parts_equals_tpp_key_for_key_at_the_same_seed() -> None:
    """`tpp` is `token_ratio` then this, so the two must be the same dict — including the
    bootstrap bounds, which take the same RNG path and are compared exactly."""
    tokenizer = CharTokenizer()
    texts, pivot_texts = ["abc", "a", "abcd"], ["ab", "abcd", "a"]
    direct = tpp(tokenizer, texts, pivot_texts, n_bootstrap=100, seed=7, ci=0.9)
    delegated = tpp_from_parts(
        token_ratio(tokenizer, texts, pivot_texts), n_bootstrap=100, seed=7, ci=0.9
    )
    assert delegated == direct


def test_tpp_from_parts_on_a_subset_measures_only_those_pairs() -> None:
    parts = token_ratio(CharTokenizer(), ["abc", "a", "abcd"], ["ab", "abcd", "a"])
    result = tpp_from_parts(parts.subset([0, 2]), n_bootstrap=0)
    assert result["source_tokens"] == 7 and result["pivot_tokens"] == 3
    assert result["value"] == pytest.approx(7 / 3)
    assert result["n"] == 2


def test_tpp_from_parts_on_an_empty_subset_is_an_empty_undefined_measurement() -> None:
    """An empty length bin still gets an entry, and it must not read as a measured zero."""
    parts = token_ratio(CharTokenizer(), ["abc", "a"], ["ab", "abcd"])
    result = tpp_from_parts(parts.subset([]), n_bootstrap=100, seed=0)
    assert result["n"] == 0
    assert math.isnan(result["value"])
    assert math.isnan(result["ci_low"]) and math.isnan(result["ci_high"])
    assert result["per_pair"] == []


# --- tpp_paired_delta: the split-arm minus raw-arm difference (Experiment 03) --------


def _parts(source: tuple[int, ...], pivot: tuple[int, ...]) -> RatioParts:
    return RatioParts(source_counts=source, pivot_counts=pivot)


def test_tpp_paired_delta_hand_computed() -> None:
    """`a` is 6/4 = 1.5, `b` is 4/4 = 1.0, so the delta (split minus raw) is 0.5."""
    result = tpp_paired_delta(
        _parts((4, 2), (2, 2)),
        _parts((2, 2), (2, 2)),
        n_bootstrap=200,
        seed=0,
        ci=0.95,
    )
    assert result["value_a"] == pytest.approx(1.5)
    assert result["value_b"] == pytest.approx(1.0)
    assert result["delta"] == pytest.approx(0.5)
    assert result["value"] == result["delta"]
    assert result["n"] == 2
    assert result["unit"] == "delta tokens/proposition ratio"
    assert result["n_bootstrap"] == 200 and result["seed"] == 0 and result["ci"] == 0.95


def test_tpp_paired_delta_of_identical_sides_is_zero_with_a_degenerate_ci() -> None:
    """Nothing changed, so every resample gives the same ratio on both sides."""
    parts = _parts((4, 2, 7), (2, 2, 3))
    result = tpp_paired_delta(parts, parts, n_bootstrap=100, seed=3)
    assert result["delta"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_tpp_paired_delta_is_deterministic_for_a_seed() -> None:
    a, b = _parts((4, 2, 9), (2, 2, 3)), _parts((3, 2, 8), (2, 2, 3))
    first = tpp_paired_delta(a, b, n_bootstrap=100, seed=7)
    second = tpp_paired_delta(a, b, n_bootstrap=100, seed=7)
    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])


def test_tpp_paired_delta_ci_brackets_the_delta() -> None:
    result = tpp_paired_delta(
        _parts((4, 2, 9, 5), (2, 2, 3, 4)),
        _parts((3, 2, 8, 5), (2, 2, 3, 4)),
        n_bootstrap=500,
        seed=1,
    )
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]


def test_tpp_paired_delta_resamples_pairs_jointly() -> None:
    """The bootstrap is paired: a draw takes the same pair indices on both sides, so a
    pair-for-pair identical difference cannot pick up noise from mismatched draws."""
    a = _parts((4, 8, 12), (2, 2, 2))
    b = _parts((2, 4, 6), (2, 2, 2))  # exactly half of `a` on every pair
    result = tpp_paired_delta(a, b, n_bootstrap=200, seed=5)
    assert result["delta"] == pytest.approx(a.value - b.value)
    # every resample halves identically, so the interval is the delta's own spread only
    assert result["ci_low"] < result["ci_high"]


def test_tpp_paired_delta_zero_bootstrap_gives_nan_ci() -> None:
    result = tpp_paired_delta(_parts((4,), (2,)), _parts((2,), (2,)), n_bootstrap=0)
    assert math.isnan(result["ci_low"]) and math.isnan(result["ci_high"])


def test_tpp_paired_delta_with_an_empty_pivot_side_is_nan_throughout() -> None:
    result = tpp_paired_delta(_parts((4,), (0,)), _parts((2,), (0,)), n_bootstrap=50)
    assert math.isnan(result["delta"])
    assert math.isnan(result["ci_low"]) and math.isnan(result["ci_high"])


def test_tpp_paired_delta_length_mismatch_raises() -> None:
    """The two sides must be the same pairs, or the difference is not paired at all."""
    with pytest.raises(ValueError, match="same"):
        tpp_paired_delta(_parts((4, 2), (2, 2)), _parts((2,), (2,)), n_bootstrap=0)


def test_tpp_paired_delta_rejects_a_confidence_level_outside_the_unit_interval() -> None:
    for bad in (0.0, 1.0, 95.0, -0.5, math.nan):
        with pytest.raises(ValueError, match="confidence level"):
            tpp_paired_delta(_parts((4,), (2,)), _parts((2,), (2,)), n_bootstrap=10, ci=bad)


# --- block bootstrap: consecutive verses and document sentences are not i.i.d. ---------


def _autocorrelated_parts(n_blocks: int, block_length: int) -> RatioParts:
    """A series whose blocks are internally constant and differ from each other.

    Every pair inside block `b` has the same counts, so an i.i.d. resample of pairs sees
    `n_blocks * block_length` "independent" observations of `n_blocks` distinct values and
    reports a narrow interval; a block resample sees `n_blocks` observations and reports a
    wide one. That is exactly the dependence a corpus of consecutive verses has.
    """
    source: list[int] = []
    pivot: list[int] = []
    for index in range(n_blocks):
        source.extend([1 + 4 * index] * block_length)
        pivot.extend([4] * block_length)
    return RatioParts(source_counts=tuple(source), pivot_counts=tuple(pivot))


def test_block_count_counts_the_short_final_block() -> None:
    assert block_count(10, 5) == 2
    assert block_count(11, 5) == 3
    assert block_count(4, 5) == 1
    assert block_count(0, 5) == 0


def test_block_count_rejects_a_non_positive_block_length() -> None:
    for bad in (0, -3):
        with pytest.raises(ValueError, match="block_length"):
            block_count(10, bad)


def test_block_keys_are_absent_unless_a_block_length_is_asked_for() -> None:
    """Every pre-existing leaf of `results.json` must be unchanged by this addition."""
    parts = token_ratio(CharTokenizer(), ["abc", "a"], ["ab", "abcd"])
    plain = tpp_from_parts(parts, n_bootstrap=50, seed=0)
    assert not [key for key in plain if key.endswith("_block") or key == "n_blocks"]
    blocked = tpp_from_parts(parts, n_bootstrap=50, seed=0, block_length=1)
    assert set(blocked) - set(plain) == {
        "ci_low_block",
        "ci_high_block",
        "block_length",
        "n_blocks",
    }
    assert all(blocked[key] == plain[key] for key in plain)  # type: ignore[literal-required]


def test_block_bootstrap_is_wider_than_iid_on_a_perfectly_autocorrelated_series() -> None:
    parts = _autocorrelated_parts(n_blocks=12, block_length=10)
    result = tpp_from_parts(parts, n_bootstrap=500, seed=0, block_length=10)
    iid_width = result["ci_high"] - result["ci_low"]
    block_width = result["ci_high_block"] - result["ci_low_block"]
    assert block_width > iid_width
    assert result["block_length"] == 10 and result["n_blocks"] == 12


def test_block_length_one_reproduces_the_iid_interval_exactly() -> None:
    """Blocks of one pair *are* the i.i.d. resample: same generator, same shape, same seed."""
    parts = token_ratio(
        CharTokenizer(), ["abc", "a", "abcd", "ab", "e"], ["ab", "abcd", "a", "abcde", "ef"]
    )
    result = tpp_from_parts(parts, n_bootstrap=300, seed=5, block_length=1)
    assert result["ci_low_block"] == result["ci_low"]
    assert result["ci_high_block"] == result["ci_high"]
    assert result["n_blocks"] == 5


def test_a_block_length_at_or_above_n_collapses_to_the_point_estimate() -> None:
    """One block is one exchangeable unit: there is nothing to resample, and the interval
    says so rather than pretending to precision."""
    parts = token_ratio(CharTokenizer(), ["abc", "a", "abcd"], ["ab", "abcd", "a"])
    for block_length in (3, 4, 100):
        result = tpp_from_parts(parts, n_bootstrap=200, seed=0, block_length=block_length)
        assert result["ci_low_block"] == pytest.approx(result["value"])
        assert result["ci_high_block"] == pytest.approx(result["value"])
        assert result["n_blocks"] == 1


def test_block_bootstrap_keeps_the_two_sides_paired() -> None:
    """Both sides are resampled with the same block draw, so a ratio of identical sides
    stays exactly 1.0 in every draw."""
    parts = RatioParts(source_counts=(3, 1, 4, 1, 5), pivot_counts=(3, 1, 4, 1, 5))
    result = tpp_from_parts(parts, n_bootstrap=200, seed=1, block_length=2)
    assert result["ci_low_block"] == 1.0 and result["ci_high_block"] == 1.0


def test_block_bootstrap_is_deterministic_for_a_seed() -> None:
    parts = _autocorrelated_parts(n_blocks=6, block_length=4)
    a = tpp_from_parts(parts, n_bootstrap=100, seed=3, block_length=4)
    b = tpp_from_parts(parts, n_bootstrap=100, seed=3, block_length=4)
    assert (a["ci_low_block"], a["ci_high_block"]) == (b["ci_low_block"], b["ci_high_block"])


def test_block_bootstrap_on_an_empty_or_zero_draw_measurement_is_nan() -> None:
    empty = tpp_from_parts(
        RatioParts(source_counts=(), pivot_counts=()), n_bootstrap=100, seed=0, block_length=5
    )
    assert math.isnan(empty["ci_low_block"]) and math.isnan(empty["ci_high_block"])
    assert empty["n_blocks"] == 0
    none_drawn = tpp_from_parts(
        RatioParts(source_counts=(1, 2), pivot_counts=(1, 2)),
        n_bootstrap=0,
        seed=0,
        block_length=1,
    )
    assert math.isnan(none_drawn["ci_low_block"])


def test_block_bootstrap_rejects_a_non_positive_block_length_through_tpp() -> None:
    with pytest.raises(ValueError, match="block_length"):
        tpp(CharTokenizer(), ["abc"], ["ab"], n_bootstrap=10, block_length=0)


def test_tpp_forwards_block_length_to_tpp_from_parts() -> None:
    tokenizer = CharTokenizer()
    texts, pivot_texts = ["abc", "a", "abcd"], ["ab", "abcd", "a"]
    direct = tpp(tokenizer, texts, pivot_texts, n_bootstrap=100, seed=7, block_length=2)
    delegated = tpp_from_parts(
        token_ratio(tokenizer, texts, pivot_texts), n_bootstrap=100, seed=7, block_length=2
    )
    assert delegated == direct
