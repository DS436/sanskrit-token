"""Tests for Rényi efficiency (`sanskrit_tok.metrics.renyi`).

Every expected number is hand-computed from the fake tokenizer below, which returns a
fixed id list per text, so the token *type counts* are exact and the tests pin the
definition of the metric rather than the implementation. The reference the numbers are
checked against is Zouhar et al. 2023's `tokenization-scorer`, whose `_renyi_efficiency`
divides the Rényi entropy by `log2` of the *observed* number of distinct types.
"""

import math

import pytest

from sanskrit_tok.metrics.renyi import renyi_efficiency
from sanskrit_tok.tokenizers.base import Tokenizer


class FixedTokenizer:
    """Returns a fixed id list per text, so the pooled type counts are exact.

    Any text not in the table encodes to nothing, which is how the zero-token corpus
    below is built.
    """

    name = "fixed"

    def __init__(self, table: dict[str, list[int]]) -> None:
        self.table = table

    def encode(self, text: str) -> list[int]:
        return list(self.table.get(text, []))


#: Four types, one occurrence each: `p_i = 1/4`.
UNIFORM4 = FixedTokenizer({"a": [10, 11], "b": [12, 13]})
#: Two types with counts [3, 1].
SKEWED = FixedTokenizer({"a": [10, 10], "b": [10, 11]})
#: Two types, one occurrence each.
UNIFORM2 = FixedTokenizer({"a": [10, 11]})
#: One type, three occurrences: the support is 1, so `log2(K)` is 0.
SINGLE = FixedTokenizer({"a": [10, 10, 10]})
#: Every text encodes to nothing.
EMPTY_ENCODER = FixedTokenizer({})


def test_fake_satisfies_the_tokenizer_protocol() -> None:
    assert isinstance(UNIFORM4, Tokenizer)


# --- hand-computed values ----------------------------------------------------------


def test_uniform_four_types_alpha_2_is_perfectly_efficient() -> None:
    """counts [1,1,1,1]: sum p^2 = 4·(1/4)^2 = 1/4, H_2 = log2(1/4)/(1−2) = 2 bits,
    K = 4, value = 2 / log2(4) = 1.0."""
    result = renyi_efficiency(UNIFORM4, ["a", "b"], alpha=2.0)
    assert result["entropy_bits"] == pytest.approx(2.0)
    assert result["value"] == pytest.approx(1.0)
    assert result["n"] == 4
    assert result["n_types"] == 4
    assert result["unit"] == "renyi efficiency"
    assert result["n_undefined"] == 0
    assert result["alpha"] == 2.0


def test_skewed_two_types_alpha_2_hand_computed() -> None:
    """counts [3,1]: sum p^2 = 9/16 + 1/16 = 10/16, H_2 = −log2(10/16) = log2(1.6),
    K = 2 so log2(K) = 1 and value = H_2."""
    result = renyi_efficiency(SKEWED, ["a", "b"], alpha=2.0)
    assert result["entropy_bits"] == pytest.approx(0.678072, abs=1e-6)
    assert result["value"] == pytest.approx(0.678072, abs=1e-6)
    assert result["n"] == 4
    assert result["n_types"] == 2


def test_uniform_two_types_alpha_3_hand_computed() -> None:
    """counts [1,1] at α=3: sum p^3 = 2·(1/8) = 1/4, H_3 = log2(1/4)/(1−3) = 1.0,
    value = 1.0 / log2(2) = 1.0."""
    result = renyi_efficiency(UNIFORM2, ["a"], alpha=3.0)
    assert result["entropy_bits"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(1.0)
    assert result["n"] == 2
    assert result["n_types"] == 2


def test_single_type_has_zero_entropy_and_an_undefined_efficiency() -> None:
    """One type: sum p^α = 1, so H_α = 0; the denominator log2(1) is 0 too, so the
    efficiency is `nan` by construction and says so in `n_undefined`."""
    result = renyi_efficiency(SINGLE, ["a"], alpha=2.5)
    assert result["entropy_bits"] == 0.0
    assert math.isnan(result["value"])
    assert result["n"] == 3
    assert result["n_types"] == 1
    assert result["n_undefined"] == 1


def test_zero_entropy_is_positive_zero() -> None:
    """`log2(1)/(1−α)` is `-0.0` in IEEE arithmetic; `results.json` should not carry it."""
    assert math.copysign(1.0, renyi_efficiency(SINGLE, ["a"], alpha=3.0)["entropy_bits"]) == 1.0


# --- the nominal-vocabulary normalisation -------------------------------------------


def test_efficiency_nominal_divides_by_the_declared_vocabulary_size() -> None:
    """H_2 = 2 bits over log2(8) = 3, so 2/3 — a smaller number than the observed-support
    efficiency of 1.0, because only 4 of the 8 available types were used."""
    result = renyi_efficiency(UNIFORM4, ["a", "b"], alpha=2.0, vocab_size=8)
    assert result["efficiency_nominal"] == pytest.approx(2 / 3)
    assert result["value"] == pytest.approx(1.0)
    assert result["vocab_size"] == 8


def test_efficiency_nominal_is_nan_without_a_vocabulary_size() -> None:
    result = renyi_efficiency(UNIFORM4, ["a", "b"], alpha=2.0)
    assert math.isnan(result["efficiency_nominal"])
    assert result["vocab_size"] is None


def test_efficiency_nominal_is_nan_for_a_degenerate_vocabulary_size() -> None:
    """log2(1) is 0, so a one-token vocabulary has no nominal efficiency either."""
    result = renyi_efficiency(UNIFORM4, ["a", "b"], alpha=2.0, vocab_size=1)
    assert math.isnan(result["efficiency_nominal"])


# --- validation and edges ------------------------------------------------------------


def test_alpha_one_is_rejected_as_the_shannon_limit() -> None:
    with pytest.raises(ValueError) as excinfo:
        renyi_efficiency(UNIFORM4, ["a"], alpha=1.0)
    assert "alpha" in str(excinfo.value)


def test_alpha_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        renyi_efficiency(UNIFORM4, ["a"], alpha=0.0)


def test_negative_alpha_is_rejected() -> None:
    with pytest.raises(ValueError):
        renyi_efficiency(UNIFORM4, ["a"], alpha=-2.0)


def test_a_bare_str_is_rejected() -> None:
    with pytest.raises(TypeError):
        renyi_efficiency(UNIFORM4, "a b c", alpha=2.0)  # type: ignore[arg-type]


def test_empty_input_measures_nothing() -> None:
    result = renyi_efficiency(UNIFORM4, [], alpha=2.0)
    assert result["value"] == 0.0
    assert result["n"] == 0
    assert result["entropy_bits"] == 0.0
    assert result["n_types"] == 0
    assert result["n_undefined"] == 0


def test_texts_that_yield_no_tokens_are_undefined_rather_than_zero() -> None:
    """Not the same case as empty input: something was measured and had no support."""
    result = renyi_efficiency(EMPTY_ENCODER, ["a", "b"], alpha=2.0)
    assert math.isnan(result["value"])
    assert result["n"] == 0
    assert result["n_types"] == 0
    assert result["n_undefined"] == 1


def test_counts_are_pooled_across_texts_not_averaged_per_text() -> None:
    """The same four ids split one-per-text give the same uniform distribution."""
    pooled = renyi_efficiency(UNIFORM4, ["a", "b"], alpha=2.0)
    split = renyi_efficiency(
        FixedTokenizer({"w": [10], "x": [11], "y": [12], "z": [13]}),
        ["w", "x", "y", "z"],
        alpha=2.0,
    )
    assert split["value"] == pytest.approx(pooled["value"])
    assert split["n_types"] == pooled["n_types"] == 4


def test_alpha_two_point_five_matches_the_closed_form() -> None:
    """α is not restricted to integers: counts [3,1] at α=2.5, hand-computed as
    H = log2((3/4)^2.5 + (1/4)^2.5) / (1 − 2.5), over log2(2) = 1."""
    expected = math.log2(0.75**2.5 + 0.25**2.5) / (1 - 2.5)
    result = renyi_efficiency(SKEWED, ["a", "b"], alpha=2.5)
    assert result["entropy_bits"] == pytest.approx(expected)
    assert result["value"] == pytest.approx(expected)
