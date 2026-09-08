"""Structural types shared by every tokenizer arm and every metric function.

`Tokenizer` is the only surface the metrics depend on, so a metric can be run against
anything that counts tokens: a `tokenizers.Tokenizer`, a `transformers` fast tokenizer,
`tiktoken`, or a byte-level fallback. `registry.py` returns objects satisfying it for
each arm T0..T7 (CLAUDE.md §6); the metrics never import the registry.

`MetricResult` is the metric contract of CLAUDE.md §7: every metric returns at least
`value`, `n` and `unit`. `DetailedMetricResult` adds the optional per-item distributions
that individual metrics attach on top of it.

`TokenizerWithSpans` is `Tokenizer` plus character offsets, which MorphScore needs and no
count-based metric does (docs/decisions.md, "Token spans for MorphScore deferred to
Experiment 04"). It is a separate protocol rather than an extra method on `Tokenizer`
because not every arm can answer it: offsets require a *fast* tokenizer, and a slow
SentencePiece load can count tokens perfectly well while having no offsets at all.

`require_texts` is the one piece of validation every metric shares, and it exists because
`str` is itself a `Sequence[str]`: passing a single sentence where a corpus is expected
type-checks, runs, and silently measures the text one *character* at a time.
"""

from collections.abc import Sequence
from typing import Protocol, TypedDict, runtime_checkable

__all__ = [
    "DetailedMetricResult",
    "MetricResult",
    "Tokenizer",
    "TokenizerWithSpans",
    "require_texts",
    "spans_cover_text",
]


@runtime_checkable
class Tokenizer(Protocol):
    """Anything that can name itself and turn a string into token ids.

    `name` is the arm key (CLAUDE.md §6) and is what experiments record in `results.json`;
    only `len(encode(text))` matters to the metrics, never the id values themselves.
    """

    name: str

    def encode(self, text: str) -> list[int]: ...


@runtime_checkable
class TokenizerWithSpans(Tokenizer, Protocol):
    """A `Tokenizer` that can also say *where* in the text each of its tokens came from.

    `spans(text)` returns half-open character offsets `(start, end)` into `text`, one per
    token, in order and non-overlapping, whose union covers every non-whitespace character
    of `text` exactly once (`spans_cover_text` is that invariant, executable). Tokens that
    carry no characters of their own — a bare `Metaspace` marker, a token that is only the
    whitespace attached to the next word, a byte-level fragment that lands entirely inside
    a character an earlier token already covered — are dropped, so `len(spans(text))` is
    generally *not* `len(encode(text))` and must never be used as a token count.

    Offsets are characters, not bytes: MorphScore compares them against gold morpheme
    boundaries, which the DCS ingestion records as character indices into the SLP1 surface
    (docs/decisions.md, "Gold boundaries: segment boundaries from DCS"). A byte-level
    tokenizer's boundary that falls inside a multi-byte character therefore has to be
    rounded to a character boundary; `registry.py` documents the rule it uses.
    """

    def spans(self, text: str) -> list[tuple[int, int]]: ...


def spans_cover_text(text: str, spans: Sequence[tuple[int, int]]) -> bool:
    """Whether `spans` satisfies the `TokenizerWithSpans` contract on `text`.

    True when the spans are in order, non-overlapping, within bounds, and together cover
    every non-whitespace character of `text` exactly once. Whitespace is not required to
    be covered — the adapters trim it off the edge of a token — but a span is allowed to
    *contain* whitespace, because real vocabularies hold multi-word tokens.

    This is the invariant MorphScore depends on: a boundary set built from span starts is
    only a segmentation of the word if the spans tile it. Exposed rather than inlined so
    the same check runs against fakes offline and against every real arm under the network
    gate.
    """
    covered = [False] * len(text)
    cursor = 0
    for start, end in spans:
        if start < cursor or end < start or start < 0 or end > len(text):
            return False
        for index in range(start, end):
            covered[index] = True
        cursor = end
    return all(
        is_covered or character.isspace()
        for is_covered, character in zip(covered, text, strict=True)
    )


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

    Five groups. The `per_*` lists are the distributions, one entry per word, text or
    aligned pair. `n_undefined` counts how many of those entries are `nan` because the
    ratio does not exist (a zero denominator); it is reported rather than papered over,
    because an undefined item is a property of the corpus, not a measurement of zero.
    `ci_low`/`ci_high`/`n_bootstrap`/`seed` and `source_tokens`/`pivot_tokens` are the
    bootstrap interval and the two token totals behind a ratio metric such as `tpp`.
    `ci_low_block`/`ci_high_block`/`block_length`/`n_blocks` are `tpp`'s *second*,
    non-overlapping-block interval, present only when the caller asked for one: sentences
    of a verse corpus or of one document are not exchangeable, so the i.i.d. interval in
    `ci_low`/`ci_high` is too narrow for them (`metrics/tpp.py`).

    `entropy_bits`, `n_types`, `alpha`, `vocab_size` and `efficiency_nominal` belong to
    `renyi`, whose `value` is a ratio of two numbers a reader needs separately: the Rényi
    entropy of the token distribution and the support it was normalised by (`n_types`, the
    types actually observed). `alpha` records the order it was computed at, and
    `vocab_size`/`efficiency_nominal` the alternative normalisation by the arm's full id
    space — reported alongside, never as the headline (see that module's docstring).

    The last group belongs to `morphscore`, whose headline `value` is an F1 and therefore
    needs both of its components (`precision`, `recall`), the three pooled counts they are
    computed from (`n_matched`, `n_token_boundaries`, `n_gold_boundaries`), the
    `tolerance` the matching used, its own per-item distribution (`per_word_f1`), and the
    three populations it did *not* score: `n_excluded_single_token` and
    `n_excluded_single_morpheme` are Arnett & Bergen's exclusions and
    `n_skipped_unaligned` the words with no gold segmentation at all. Those three are
    reported rather than folded into `n`, because how much of a corpus a MorphScore was
    computed on is part of the number.
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
    ci_low_block: float
    ci_high_block: float
    block_length: int
    n_blocks: int
    precision: float
    recall: float
    per_word_f1: list[float]
    n_matched: int
    n_token_boundaries: int
    n_gold_boundaries: int
    n_excluded_single_token: int
    n_excluded_single_morpheme: int
    n_skipped_unaligned: int
    tolerance: int
    entropy_bits: float
    n_types: int
    alpha: float
    vocab_size: int | None
    efficiency_nominal: float


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
