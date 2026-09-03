"""Structural types shared by every tokenizer arm and every metric function.

`Tokenizer` is the only surface the metrics depend on, so a metric can be run against
anything that counts tokens: a `tokenizers.Tokenizer`, a `transformers` fast tokenizer,
`tiktoken`, or a byte-level fallback. `registry.py` returns objects satisfying it for
each arm T0..T7 (CLAUDE.md §6); the metrics never import the registry.

`MetricResult` is the metric contract of CLAUDE.md §7: every metric returns at least
`value`, `n` and `unit`. `DetailedMetricResult` adds the optional per-item distributions
that individual metrics attach on top of it.

`require_texts` is the one piece of validation every metric shares, and it exists because
`str` is itself a `Sequence[str]`: passing a single sentence where a corpus is expected
type-checks, runs, and silently measures the text one *character* at a time.
"""

from typing import Protocol, TypedDict, runtime_checkable

__all__ = ["DetailedMetricResult", "MetricResult", "Tokenizer", "require_texts"]


@runtime_checkable
class Tokenizer(Protocol):
    """Anything that can name itself and turn a string into token ids.

    `name` is the arm key (CLAUDE.md §6) and is what experiments record in `results.json`;
    only `len(encode(text))` matters to the metrics, never the id values themselves.
    """

    name: str

    def encode(self, text: str) -> list[int]: ...


class MetricResult(TypedDict):
    """The minimum every metric returns (CLAUDE.md §7).

    `value` is the metric, `n` its denominator or sample size (the meaning is per metric
    and stated in that metric's docstring), and `unit` a human-readable label carried into
    figures and `results.json`.
    """

    value: float
    n: int
    unit: str


class DetailedMetricResult(MetricResult, total=False):
    """`MetricResult` plus the per-item distribution and diagnostics a metric may attach.

    Every extra key is optional: a caller that only needs the headline number can consume
    this as a plain `MetricResult`. Each metric documents which keys it populates.

    Three groups. The `per_*` lists are the distributions, one entry per word, text or
    aligned pair. `n_undefined` counts how many of those entries are `nan` because the
    ratio does not exist (a zero denominator); it is reported rather than papered over,
    because an undefined item is a property of the corpus, not a measurement of zero.
    `ci_low`/`ci_high`/`n_bootstrap`/`seed` and `source_tokens`/`pivot_tokens` are the
    bootstrap interval and the two token totals behind a ratio metric such as `tpp`.
    """

    per_word: list[int]
    per_text: list[float]
    per_pair: list[float]
    n_undefined: int
    ci_low: float
    ci_high: float
    n_bootstrap: int
    seed: int
    source_tokens: int
    pivot_tokens: int


def require_texts(texts: object) -> None:
    """Raise `TypeError` if `texts` is a bare `str` rather than a sequence of texts.

    Every metric takes `(tokenizer, list[str])` (CLAUDE.md §7), and a `str` satisfies
    `Sequence[str]` structurally: `fertility(tok, "rAmaH gacCati")` would type-check, run,
    and return a number computed over single characters — a wrong answer rather than an
    error. The mistake is easy to make from a REPL or a one-sentence sanity check, and the
    resulting number is plausible enough to survive review, so the metrics reject it.

    Deliberately narrow: only `str` is rejected, because only `str` is silently wrong.
    Anything else that is not iterable of strings fails loudly on its own.
    """
    if isinstance(texts, str):
        raise TypeError(
            "expected a sequence of texts, got a single str; a str is itself a "
            "Sequence[str], so this would be measured one character at a time. "
            "Wrap it in a list: [text]"
        )
