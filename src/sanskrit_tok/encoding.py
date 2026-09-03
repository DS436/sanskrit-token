"""Devanagari/IAST <-> SLP1 transliteration, with roundtrip guarantees.

SLP1 is the project's internal encoding (CLAUDE.md §2.3): every Sanskrit string is
converted to SLP1 on ingest and back to Devanagari or IAST only for display. SLP1 is
ASCII, one character per phoneme, and lossless for the Sanskrit phoneme inventory.

Passthrough is asymmetric, because SLP1 is itself written in ASCII:

- `to_slp1(text, "devanagari")` leaves every non-Devanagari character alone, so Latin
  words, digits, punctuation and whitespace embedded in a Devanagari line survive
  conversion. That guarantee is scoped to `source="devanagari"`.
- `to_slp1(text, "iast")` does **not** give it. IAST is itself a romanisation, so ASCII
  letters are read as phonemes in the forward direction too (`"The"` -> `"Te"`,
  `"Sanskrit"` -> `"sanskrit"`) and `|` is danda (-> `.`). Only ASCII outside the IAST
  alphabet passes through.
- `from_slp1` reads its input as SLP1, so ASCII characters that SLP1 uses are decoded
  rather than passed through: letters are phonemes, digits become Devanagari digits,
  `'` is avagraha, `.` is danda, `~` is candrabindu and `|` is Vedic `ḻh`. Only
  characters outside the SLP1 alphabet pass through untouched in both directions.

Consequence: a line mixing Latin text with Devanagari does **not** roundtrip, and cannot,
since `from_slp1` has no way to tell an English word from a run of SLP1 phonemes. Callers
that must preserve mixed-script text should keep the original string alongside the SLP1
form (CLAUDE.md §2.3), which is what the corpus loaders do.
"""

from typing import Final, Literal, get_args

from indic_transliteration import sanscript

__all__ = ["Script", "from_slp1", "roundtrip_ok", "to_slp1"]

Script = Literal["devanagari", "iast", "slp1"]

_SCHEMES: Final[dict[Script, str]] = {
    "devanagari": sanscript.DEVANAGARI,
    "iast": sanscript.IAST,
    "slp1": sanscript.SLP1,
}


def _scheme(script: Script) -> str:
    """Map a project script name to its `sanscript` scheme constant."""
    scheme = _SCHEMES.get(script)
    if scheme is None:
        valid = ", ".join(repr(name) for name in get_args(Script))
        raise ValueError(f"unknown script {script!r}; expected one of {valid}")
    return scheme


def to_slp1(text: str, source: Script) -> str:
    """Convert `text` from `source` script to SLP1.

    `source == "slp1"` is the identity.
    """
    scheme = _scheme(source)
    if source == "slp1":
        return text
    return str(sanscript.transliterate(text, scheme, sanscript.SLP1))


def from_slp1(text: str, target: Script) -> str:
    """Convert SLP1 `text` to `target` script.

    `target == "slp1"` is the identity.
    """
    scheme = _scheme(target)
    if target == "slp1":
        return text
    return str(sanscript.transliterate(text, sanscript.SLP1, scheme))


def roundtrip_ok(text: str, script: Script) -> bool:
    """True when `text` survives `script` -> SLP1 -> `script` unchanged."""
    return from_slp1(to_slp1(text, script), script) == text
