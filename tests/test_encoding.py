"""Tests for `sanskrit_tok.encoding`: SLP1 conversion and roundtrip identity."""

from pathlib import Path

import pytest

from sanskrit_tok.encoding import Script, from_slp1, roundtrip_ok, to_slp1

FIXTURE = Path(__file__).parent / "fixtures" / "devanagari_sample.txt"


def _fixture_lines() -> list[str]:
    return FIXTURE.read_text(encoding="utf-8").splitlines()


def _first_difference(expected: str, actual: str) -> str:
    """Describe the first differing character, for an actionable failure message."""
    for index, (want, got) in enumerate(zip(expected, actual, strict=False)):
        if want != got:
            return (
                f"index {index}: expected {want!r} (U+{ord(want):04X}), "
                f"got {got!r} (U+{ord(got):04X})"
            )
    if len(expected) != len(actual):
        return f"length differs: expected {len(expected)}, got {len(actual)}"
    return "no difference"


#: Line 10 mixes Latin text with Devanagari. It cannot roundtrip: SLP1 is written in
#: ASCII, so `from_slp1` reads "The" as the phoneme sequence T-h-e ('T' U+0054 becomes
#: 'थ' U+0925) and the sentence-final '.' as danda '।'. Kept in the fixture deliberately,
#: as the documented boundary of the roundtrip guarantee. `strict=True` so that the suite
#: fails if this ever starts passing.
_MIXED_SCRIPT_LINE = 10
_LINE_PARAMS = [
    pytest.param(
        n,
        marks=(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "fixture line 10 mixes Latin and Devanagari; SLP1 is an ASCII scheme, "
                    "so from_slp1 decodes Latin letters as phonemes ('T' U+0054 -> "
                    "'थ' U+0925) and '.' as danda. Inherent, not a bug."
                ),
            )
            if n == _MIXED_SCRIPT_LINE
            else ()
        ),
    )
    for n in range(1, 13)
]


@pytest.mark.parametrize("lineno", _LINE_PARAMS)
def test_devanagari_fixture_line_roundtrips(lineno: int) -> None:
    """Devanagari -> SLP1 -> Devanagari must be the identity on every fixture line."""
    line = _fixture_lines()[lineno - 1]
    actual = from_slp1(to_slp1(line, "devanagari"), "devanagari")
    assert actual == line, (
        f"fixture line {lineno} did not roundtrip: {line!r} -> {actual!r}; "
        f"{_first_difference(line, actual)}"
    )
    assert roundtrip_ok(line, "devanagari")


def test_fixture_has_twelve_lines() -> None:
    assert len(_fixture_lines()) == 12


def test_iast_to_slp1_known_string() -> None:
    assert to_slp1("rāmaḥ gacchati", "iast") == "rAmaH gacCati"


def test_slp1_to_iast_known_string() -> None:
    assert from_slp1("rAmaH gacCati", "iast") == "rāmaḥ gacchati"


def test_iast_roundtrip() -> None:
    assert roundtrip_ok("rāmaḥ gacchati", "iast")


def test_slp1_source_is_identity() -> None:
    assert to_slp1("rAmaH gacCati", "slp1") == "rAmaH gacCati"
    assert from_slp1("rAmaH gacCati", "slp1") == "rAmaH gacCati"
    assert roundtrip_ok("rAmaH gacCati", "slp1")


#: ASCII that SLP1 does not use: safe in both directions. SLP1 claims letters, digits,
#: `'` (avagraha), `.` (danda), `~` (candrabindu) and `|` (Vedic ḻh).
_NEUTRAL_ASCII = "!\"#$%&()*+,-/:;<=>?@[\\]^_{} \t\n"


@pytest.mark.parametrize("script", ["devanagari", "iast", "slp1"])
def test_non_slp1_ascii_passes_through_both_directions(script: Script) -> None:
    """Punctuation and whitespace outside the SLP1 alphabet are untouched both ways."""
    assert to_slp1(_NEUTRAL_ASCII, script) == _NEUTRAL_ASCII
    assert from_slp1(_NEUTRAL_ASCII, script) == _NEUTRAL_ASCII


def test_latin_digits_and_punctuation_pass_through_to_slp1() -> None:
    """Everything non-Devanagari survives the ingest direction unchanged."""
    text = "The word संस्कृतम् means Sanskrit (1.5%)."
    assert to_slp1(text, "devanagari") == "The word saMskftam means Sanskrit (1.5%)."


def test_from_slp1_decodes_ascii_and_so_does_not_roundtrip_latin() -> None:
    """Documented limitation: `from_slp1` cannot tell English from SLP1 phonemes."""
    assert from_slp1("The", "devanagari") == "थ्हे"
    assert from_slp1("123", "devanagari") == "१२३"
    assert from_slp1("'", "devanagari") == "ऽ"
    assert from_slp1(".", "devanagari") == "।"


def test_devanagari_digits_roundtrip() -> None:
    assert to_slp1("१२३", "devanagari") == "123"
    assert roundtrip_ok("१२३", "devanagari")


def test_empty_string() -> None:
    assert to_slp1("", "devanagari") == ""
    assert from_slp1("", "devanagari") == ""
    assert roundtrip_ok("", "devanagari")


def test_unknown_script_rejected() -> None:
    with pytest.raises(ValueError):
        to_slp1("rāmaḥ", "harvard-kyoto")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        from_slp1("rAmaH", "harvard-kyoto")  # type: ignore[arg-type]
