"""Structural types shared by every tokenizer arm and every metric function.

`Tokenizer` is the only surface the metrics depend on, so a metric can be run against
anything that counts tokens: a `tokenizers.Tokenizer`, a `transformers` fast tokenizer,
`tiktoken`, or a byte-level fallback. `registry.py` returns objects satisfying it for
each arm T0..T7 (CLAUDE.md §6); the metrics never import the registry.

`MetricResult` is the metric contract of CLAUDE.md §7: every metric returns at least
`value`, `n` and `unit`. `DetailedMetricResult` adds the optional per-item distributions
that individual metrics attach on top of it.
"""

from typing import Protocol, TypedDict, runtime_checkable

__all__ = ["DetailedMetricResult", "MetricResult", "Tokenizer"]


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
    """`MetricResult` plus the per-item distribution a metric may attach.

    Every extra key is optional: a caller that only needs the headline number can consume
    this as a plain `MetricResult`. Each metric documents which key it populates.
    """

    per_word: list[int]
    per_text: list[float]
    per_pair: list[float]
