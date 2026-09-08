"""Tests for the character/density decomposition of a TPP ratio.

Every expected number is hand-computed. The one property that matters most is the identity
`char_ratio * density_ratio == tpp`: it is what lets Experiment 02's README say whether the
verse ratio lives in a shorter Sanskrit side or a longer English one, and if the two
factors did not multiply back to the measured ratio the attribution would be arbitrary.
"""

import math

import pytest

from sanskrit_tok.metrics.decomposition import DECOMPOSITION_KEYS, decompose_ratio


def test_decompose_ratio_hand_computed() -> None:
    result = decompose_ratio(
        chars_sa=100, chars_en=200, bytes_sa=100, bytes_en=200, tokens_sa=40, tokens_en=50
    )
    assert result["chars_sa"] == 100 and result["chars_en"] == 200
    assert result["bytes_sa"] == 100 and result["bytes_en"] == 200
    assert result["tokens_sa"] == 40 and result["tokens_en"] == 50
    assert result["tokens_per_char_sa"] == pytest.approx(0.4)
    assert result["tokens_per_char_en"] == pytest.approx(0.25)
    assert result["char_ratio"] == pytest.approx(0.5)
    assert result["density_ratio"] == pytest.approx(1.6)
    assert result["tpp"] == pytest.approx(0.8)


def test_decompose_ratio_returns_exactly_the_documented_keys() -> None:
    result = decompose_ratio(
        chars_sa=3, chars_en=4, bytes_sa=9, bytes_en=4, tokens_sa=1, tokens_en=2
    )
    assert tuple(result) == DECOMPOSITION_KEYS


@pytest.mark.parametrize(
    ("chars_sa", "chars_en", "tokens_sa", "tokens_en"),
    [
        (100, 200, 40, 50),
        (7, 3, 5, 11),
        (12345, 9876, 4321, 8765),
        (1, 1, 1, 1),
    ],
)
def test_the_two_factors_multiply_back_to_the_ratio(
    chars_sa: int, chars_en: int, tokens_sa: int, tokens_en: int
) -> None:
    """The identity the README's attribution rests on, to the tolerance run.py asserts."""
    result = decompose_ratio(
        chars_sa=chars_sa,
        chars_en=chars_en,
        bytes_sa=chars_sa,
        bytes_en=chars_en,
        tokens_sa=tokens_sa,
        tokens_en=tokens_en,
    )
    product = float(result["char_ratio"]) * float(result["density_ratio"])
    assert abs(product - float(result["tpp"])) < 1e-9


def test_bytes_are_carried_independently_of_characters() -> None:
    """Devanagari is three bytes per character and English one, so the byte ratio and the
    character ratio of the same sentences differ by a factor of three."""
    result = decompose_ratio(
        chars_sa=10, chars_en=20, bytes_sa=30, bytes_en=20, tokens_sa=5, tokens_en=8
    )
    assert result["char_ratio"] == pytest.approx(0.5)
    assert float(result["bytes_sa"]) / float(result["bytes_en"]) == pytest.approx(1.5)


def test_an_empty_side_gives_nan_factors_not_zeros() -> None:
    """A zero is a plausible ratio and would read as a measurement (`_ratio.py`'s rule)."""
    result = decompose_ratio(
        chars_sa=10, chars_en=0, bytes_sa=10, bytes_en=0, tokens_sa=5, tokens_en=0
    )
    assert math.isnan(float(result["char_ratio"]))
    assert math.isnan(float(result["tokens_per_char_en"]))
    assert math.isnan(float(result["density_ratio"]))
    assert math.isnan(float(result["tpp"]))
    assert result["tokens_per_char_sa"] == pytest.approx(0.5)


def test_a_negative_count_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        decompose_ratio(
            chars_sa=-1, chars_en=1, bytes_sa=1, bytes_en=1, tokens_sa=1, tokens_en=1
        )
