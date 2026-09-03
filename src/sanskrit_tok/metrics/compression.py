"""Compression: UTF-8 bytes per token.

Report this on **both** the SLP1 form and the original script of the same text
(CLAUDE.md §7). The two differ by construction and neither alone is honest: SLP1 is ASCII
at one byte per phoneme, while Devanagari costs three UTF-8 bytes per character and packs
an inherent vowel into each consonant sign, so byte counts are not comparable across
scripts. A tokenizer that looks efficient in bytes/token on Devanagari may only be
exploiting that inflated denominator.

This function does not transliterate. The caller passes whichever form it wants measured
and records which one it was; `sanskrit_tok.encoding` does the conversion.
"""

from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer, require_texts

__all__ = ["compression"]


def compression(tokenizer: Tokenizer, texts: Sequence[str]) -> DetailedMetricResult:
    """Mean UTF-8 bytes per token, pooled over `texts`.

    Each text is encoded whole, so whitespace and punctuation count on both sides of the
    ratio, unlike `fertility`, which encodes word by word.

    Returns `value` = total UTF-8 bytes / total tokens, `n` = total tokens, `unit` =
    `"bytes/token"`, and `per_text` = the bytes-per-token ratio of each text in order.

    Pooled, not averaged per text, so `value` is generally not the mean of `per_text`.
    A text that yields no tokens contributes `0.0` to `per_text`; when no text yields any
    token, `value` is `0.0` and `n` is `0`.

    Raises `TypeError` if `texts` is a single `str` rather than a sequence of them.
    """
    require_texts(texts)
    per_text: list[float] = []
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        text_bytes = len(text.encode("utf-8"))
        text_tokens = len(tokenizer.encode(text))
        per_text.append(text_bytes / text_tokens if text_tokens else 0.0)
        total_bytes += text_bytes
        total_tokens += text_tokens
    return {
        "value": total_bytes / total_tokens if total_tokens else 0.0,
        "n": total_tokens,
        "unit": "bytes/token",
        "per_text": per_text,
    }
