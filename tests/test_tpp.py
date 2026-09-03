"""Tests for tokens-per-proposition, the headline metric of CLAUDE.md §2.1 and §7.

Every expected number is hand-computed from the fake tokenizers below (duplicated from
`tests/test_metrics.py` so this module stands alone). The bootstrap is pinned by its
seed, by its degenerate cases and by determinism, not by its exact percentiles.
"""

import math

import pytest

from sanskrit_tok.metrics.tpp import tpp


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
