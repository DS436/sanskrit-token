"""Bits per character: the one language-model number this project compares across arms.

CLAUDE.md §2.2 forbids comparing perplexity across tokenizers, and the reason is the whole
point of Experiment 05. Perplexity is per *token*, and a token means something different in
every arm — a 64k BPE token is several characters, a `T7_byt5` token is one byte — so a
model with a coarser vocabulary can post a better perplexity while predicting the same text
strictly worse. Dividing the same total surprise by the number of SLP1 *characters* removes
the vocabulary from the denominator entirely: every arm is scored on the same text measured
the same way, and the number is directly comparable (docs/decisions.md, 2026-09-05,
"Experiment 05 model, evaluation and comparison protocol").

Two conventions this module fixes, because they are the kind of thing that silently differs
between two implementations of "BPC" and makes two numbers incomparable:

1. **The total is in nats and the result is in bits.** Cross-entropy from
   `torch.nn.functional.cross_entropy` is natural log, and BPC is base 2; the conversion
   happens here, once, rather than in each caller.
2. **Every token's surprise is charged, including the end-of-line token**, and the
   denominator is the characters of the text *without* its newlines. The EOS tokens are
   therefore paid for by the characters they separate. This inflates every arm's BPC by the
   same small constant — one token per line, on the same lines — so it cancels in every
   comparison the experiment actually makes, and the alternative (dropping the EOS nats)
   would let an arm win by being bad at predicting where a sentence ends.

Pure arithmetic only: nothing here imports `torch` or touches a file. The evaluator that
runs a model over held-out text and produces the token NLLs lives in `train.py`.
"""

import math
from collections.abc import Sequence

from sanskrit_tok.tokenizers.base import MetricResult

__all__ = ["BITS_PER_NAT", "BpcResult", "bits_per_char", "bpc_from_token_nll"]

#: Nats to bits: `1 / ln 2`. Named so the conversion is never re-typed as `1.4427`.
BITS_PER_NAT = 1.0 / math.log(2.0)


class BpcResult(MetricResult, total=False):
    """A BPC measurement: the CLAUDE.md §7 contract plus what BPC needs beside it.

    `value` is bits per character, `n` the character count it was divided by, and `unit`
    is always `"bits/char"`.

    `total_nats` is the summed negative log-likelihood the value came from and `n_tokens`
    how many token predictions went into that sum; both are recorded because they are what
    lets a later reader re-derive the number against a different denominator, and because
    `n_tokens / n` is the arm's characters-per-token on the evaluation text — the
    tokenization fact that BPC deliberately removes from the comparison.

    `bits_per_token` is the same total over `n_tokens`. It is the quantity closest to a
    perplexity (`exp(total_nats / n_tokens)` is exactly that), and it is reported *because*
    it is not comparable across arms: having it beside the BPC makes the difference between
    the two visible rather than tempting.

    `bits_per_byte` and `n_bytes` appear only when a byte count is supplied. SLP1 is ASCII,
    so on this project's internal encoding they equal `value` and `n`; they are carried
    anyway so that a number computed on Devanagari, where a character is three bytes, is
    not mistaken for one computed on SLP1.
    """

    total_nats: float
    n_tokens: int
    bits_per_token: float
    bits_per_byte: float
    n_bytes: int


def bits_per_char(total_nll_nats: float, n_chars: int) -> float:
    """`total_nll_nats` (natural log) spread over `n_chars` characters, in bits.

    Ten tokens each costing `ln 2` nats over twenty characters is `10 * ln 2 / ln 2 / 20`
    = 0.5 bits per character.

    Raises `ValueError` on a non-positive `n_chars` (there is no such thing as bits per
    character of nothing) and on a negative total (a negative log-likelihood cannot be
    negative; if one arrives, the caller summed something else).
    """
    if n_chars <= 0:
        raise ValueError(f"n_chars must be positive, got {n_chars}")
    if total_nll_nats < 0.0:
        raise ValueError(
            f"total_nll_nats must be non-negative, got {total_nll_nats}; a summed negative "
            "log-likelihood is a sum of non-negative terms"
        )
    return total_nll_nats * BITS_PER_NAT / n_chars


def bpc_from_token_nll(
    token_nll: Sequence[float], n_chars: int, n_bytes: int | None = None
) -> BpcResult:
    """BPC over `n_chars` characters from the per-token negative log-likelihoods.

    `token_nll` is one non-negative nat value per token *prediction* — every token of the
    held-out text, end-of-line tokens included (see the module docstring). Pass `n_bytes`
    to also get `bits_per_byte`; omit it and those two keys are absent rather than guessed,
    because assuming bytes equal characters is true for SLP1 and wrong for Devanagari.

    Raises `ValueError` on an empty `token_nll` — a BPC over no predictions is not zero,
    it does not exist — and, through `bits_per_char`, on a non-positive `n_chars`.
    """
    if len(token_nll) == 0:
        raise ValueError("token_nll is empty: a BPC over no token predictions is undefined")
    total_nats = float(math.fsum(token_nll))
    result: BpcResult = {
        "value": bits_per_char(total_nats, n_chars),
        "n": n_chars,
        "unit": "bits/char",
        "total_nats": total_nats,
        "n_tokens": len(token_nll),
        "bits_per_token": total_nats * BITS_PER_NAT / len(token_nll),
    }
    if n_bytes is not None:
        if n_bytes <= 0:
            raise ValueError(f"n_bytes must be positive when given, got {n_bytes}")
        result["bits_per_byte"] = total_nats * BITS_PER_NAT / n_bytes
        result["n_bytes"] = n_bytes
    return result
