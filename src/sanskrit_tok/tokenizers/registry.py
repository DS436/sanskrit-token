"""Arm name -> loaded tokenizer, for the off-the-shelf T0 arms (CLAUDE.md §6).

Experiments name a tokenizer by its arm key and never by a model id: `load_tokenizer`
turns `"T0_o200k"` into something the metrics can use, and records *which* concrete model
actually answered for that arm in `LoadedTokenizer.source_id`, so `results.json` always
says what was measured rather than what was asked for.

Two shims stand between the upstream libraries and `Tokenizer` (`base.py`):

* `TiktokenAdapter` — tiktoken's `Encoding.encode` accepts text containing special-token
  strings only if told what to do with them; `disallowed_special=()` makes `"<|endoftext|>"`
  in a corpus encode as ordinary text instead of raising.
* `HFAdapter` — a `transformers` tokenizer would otherwise wrap every string in BOS/EOS.
  Fertility encodes each word on its own (CLAUDE.md §7), so per-call special tokens would
  be counted once per *word* and inflate every number; `add_special_tokens=False` removes
  them.

**Token spans.** MorphScore needs to know which characters each token covers, not just
how many tokens there were, so every adapter here also implements `spans(text)` and
`LoadedTokenizer` carries whichever provider its loader could build. `LoadedTokenizer`
structurally satisfies `TokenizerWithSpans` (`base.py`) either way, since it always *has*
a `spans` method; `supports_spans` is the question worth asking, and `spans` on an arm
that has no provider raises rather than inventing offsets. The three adapter kinds get
there three different ways — `tokenizers` offsets,
`transformers` `return_offsets_mapping`, and a byte cursor over tiktoken's token bytes —
and all three are pushed through `_normalise_spans`, which is where the shared contract
(in order, non-overlapping, every non-whitespace character covered exactly once) is
actually enforced. A slow `transformers` tokenizer has no offsets at all; rather than
dropping the arm, `_FastReloadSpans` reloads the same model id with `use_fast=True` the
first time spans are asked for, and records the reload by suffixing `source_id` with
`+fast` (docs/decisions.md, "Token spans for MorphScore deferred to Experiment 04").

The T0 arms are "existing practice" (CLAUDE.md §2.5), never a controlled comparison: their
vocabulary sizes differ, so their fertilities are read as evidence about deployed
tokenizers, not as a matched experiment. `T3_*` (off-the-shelf Indic tokenizers) are the
same kind of arm, one family over. `T1_*`/`T2_*` are the opposite: trained from scratch by
this project at matched vocabulary sizes (CLAUDE.md §5), so they are read from a
`tokenizer.json` file on disk rather than downloaded, and are absent — `TokenizerUnavailable`
— until `tokenizers/train_bpe.py` / `train_unigram.py` (Task 4) have written one. `E1_*`
is the matched *English* control family (CLAUDE.md §6): the same algorithms at the same
vocabulary sizes, trained on the English side of the same corpus, so a TPP ratio against an
E1 arm holds algorithm, vocabulary size and training domain constant on both sides. It
comes in two halves — pair-matched (one English sentence per Sanskrit sentence) and
byte-matched (`_bm`, the English side cut to the Sanskrit corpus's byte count) — because
those two ways of saying "the same corpus" disagree by 48% of the text; see `E1_ARMS`. It
is file-backed for exactly the same reason as T1/T2 and loads the same way. `T4_*` is
the sandhi-split family (Experiment 03): the same two algorithms at the same two
vocabulary sizes as T1/T2, trained on the sandhi-split SLP1 corpus, and file-backed for
the same reason again. The throughput rule selected the full training corpus
(docs/decisions.md, "Splitter throughput measured"), so no matched `_sub` arms exist.

**Gated repositories.** `meta-llama/*` and `google/*` need an accepted licence and an
`HF_TOKEN`. Each HF arm therefore declares a list of candidate ids, tried in order, first
success wins: the official gated id, a mirror of the same tokenizer, then the previous
model generation and its mirror. `HF_TOKEN` is passed when the environment sets one. Any
load below the first candidate is a substitution and is recorded in `docs/decisions.md`
(CLAUDE.md §11), because the tokenizer measured is then not the one the arm is named for.

`T5_*`/`T6_*` are the morpheme-constrained families (Experiment 04), and they arrive with
ten more file-backed arms: every Experiment 04 arm — including the `T1`/`T2`/`T4` ones it
compares against — is trained on the DCS training split and carries a `_dcs` or
`_oracle_dcs` suffix to say so (`DCS_ARMS`). Same loader, same directory layout as every
other trained arm.

**`family` and `variant`.** Every `LoadedTokenizer` carries the arm-name prefix before its
first underscore (`"T0_gpt2"` -> `"T0"`), so an experiment or a metric summary can group or
filter arms without parsing the name itself; a file-backed arm also carries the
training-corpus suffix its name ends in (`"T5_morphbpe_raw_32k_dcs"` -> `"dcs"`, `""` for an
arm with none). `list_tokenizers(family=..., variant=...)` filters the registry the same
way, and both filters are needed for a family that spans experiments: `T4` is four
ByT5-split arms from Experiment 03 and four gold-split DCS arms from Experiment 04.

**`TokenizerUnavailable`.** A registered arm can still fail to produce a tokenizer this
run — every HF candidate is gated or unpublished, or a trained arm's `tokenizer.json` has
not been written yet. That is different from `KeyError` (the arm is not registered at
all): experiments catch `TokenizerUnavailable` specifically, log a WARNING, and record the
arm under `unavailable_arms` rather than aborting the whole run over one missing model.
"""

import functools
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DCS_ARMS",
    "E1_ARMS",
    "REGISTRY",
    "T0_GEMMA3_CANDIDATES",
    "T0_GPT2_CANDIDATES",
    "T0_LLAMA4_CANDIDATES",
    "T0_O200K_ENCODING",
    "T3_BRAHMIC131K_MODEL_ID",
    "T3_INDICSUPER_CANDIDATES",
    "T3_SARVAM_CANDIDATES",
    "T3_SUTRA_CANDIDATES",
    "T7_BYTE_VOCAB_SIZE",
    "ByteAdapter",
    "HFAdapter",
    "LoadedTokenizer",
    "TiktokenAdapter",
    "TokenizerUnavailable",
    "TokenizersAdapter",
    "list_tokenizers",
    "load_tokenizer",
    "trained_tokenizer_path",
]

logger = logging.getLogger(__name__)

#: tiktoken encoding behind `T0_o200k` (GPT-4o / o-series).
T0_O200K_ENCODING = "o200k_base"

#: Llama-4 Scout first; `unsloth` re-uploads the identical tokenizer without the licence
#: gate; the Llama-3 pair is the previous generation, a different tokenizer and therefore
#: a reportable substitution.
T0_LLAMA4_CANDIDATES: tuple[str, ...] = (
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "unsloth/llama-3-8b",
)

#: Same ordering for Gemma: gated Gemma-3, its mirror, then Gemma-2 and its mirror.
T0_GEMMA3_CANDIDATES: tuple[str, ...] = (
    "google/gemma-3-4b-it",
    "unsloth/gemma-3-4b-it",
    "google/gemma-2-2b",
    "unsloth/gemma-2-2b",
)

#: GPT-2 (2019, 50,257-token BPE), ungated: an older-generation English-centric arm, added
#: because the three arms above all have >=200k vocabularies (2026-09-03 decision log entry
#: "Add T0_gpt2 as an older-generation English-centric arm").
T0_GPT2_CANDIDATES: tuple[str, ...] = ("openai-community/gpt2",)

#: Off-the-shelf Indic tokenizers (CLAUDE.md §6, T3). Single published id each; `T3_sarvam`
#: and `T3_sutra` load through the same candidate-walking `_load_hf_arm` as every T0 arm.
T3_SARVAM_CANDIDATES: tuple[str, ...] = ("sarvamai/sarvam-1",)
T3_SUTRA_CANDIDATES: tuple[str, ...] = ("TWO/sutra-mlt256-v2",)

#: `T3_brahmic131k` ships a `tokenizer.json` but no guarantee of a `transformers`-style
#: config, so it is loaded by `_load_brahmic131k_arm`, which tries three adapter kinds in
#: order rather than three repository ids.
T3_BRAHMIC131K_MODEL_ID = "theschoolofai/BrahmicTokenizer-131K"

#: `T3_indicsuper` is not known to be publicly released as of 2026-09-03: none of these
#: three plausible ids resolves (`huggingface_hub.list_repo_files` returns
#: `RepositoryNotFoundError` for all three; see docs/decisions.md). Kept as candidates so
#: the arm starts loading the moment any of them is published, with no code change.
T3_INDICSUPER_CANDIDATES: tuple[str, ...] = (
    "krutrim-ai-labs/IndicSuperTokenizer",
    "ai4bharat/IndicSuperTokenizer",
    "ai4bharat/indic-super-tokenizer",
)


#: `T7_byt5` has no vocabulary file and no repository: its ids are the 256 possible UTF-8
#: byte values, so it is the tokenizer-free floor every subword arm is measured against
#: (docs/decisions.md, 2026-09-05, "Experiment 05 runs in two tracks"). The name keeps the
#: `byt5` label of CLAUDE.md §6 because ByT5 is the published model of this vocabulary, but
#: nothing is downloaded: `T7_byt5` is UTF-8 itself.
T7_BYTE_VOCAB_SIZE = 256


@dataclass
class LoadedTokenizer:
    """A tokenizer arm, ready to encode, carrying its own provenance.

    Satisfies the `Tokenizer` protocol (`base.py`), so it can be handed straight to any
    metric. `name` is the arm key (`"T0_llama4"`); `source_id` is the model id, tiktoken
    encoding name, or trained-tokenizer file path *actually* loaded, which may be a
    substitute for the arm's first choice; `vocab_size` is the full id space including
    added special tokens.

    `family` is the arm-name prefix before its first underscore (`"T0"`, `"T3"`, ...),
    computed once at load time so callers never re-derive it from `name`. `variant` is the
    other half of that split: the training-corpus label a file-backed arm's name ends in
    (`"dcs"`, `"oracle_dcs"`) and `""` for every arm whose name carries none. Family and
    variant together are what separate `T4_bpe_split_32k` (Experiment 03, ByT5-split
    parallel corpora) from `T4_bpe_split_32k_oracle_dcs` (Experiment 04, gold-split DCS),
    which are the same algorithm at the same vocabulary size on different text and must
    never be pooled. `attempted` is
    every candidate id (or, for a file-backed arm, the one file path) tried before the
    winner, winner last — a one-element tuple whenever an arm has only ever had one
    candidate. Both default to empty so hand-built fakes in tests need not set them.

    `_spans` is the arm's character-offset provider, or `None` for an arm that has none —
    a hand-built fake, or a loader tier that cannot produce offsets. `supports_spans` is
    the question a caller should ask before handing the arm to MorphScore; `spans` raises
    `TokenizerUnavailable` rather than returning something wrong when the answer is no,
    which is the same failure mode as an arm that could not be downloaded at all. An arm
    whose provider is a lazy fast reload reports `supports_spans` `True` before that
    reload has been attempted: whether it will succeed is not knowable without doing it,
    and the failure surfaces from `spans` itself.
    """

    name: str
    source_id: str
    vocab_size: int
    _encode: Callable[[str], list[int]] = field(repr=False)
    family: str = ""
    variant: str = ""
    attempted: tuple[str, ...] = ()
    _spans: Callable[[str], list[tuple[int, int]]] | None = field(default=None, repr=False)

    def encode(self, text: str) -> list[int]:
        """Token ids for `text`, with no special tokens added."""
        return self._encode(text)

    @property
    def supports_spans(self) -> bool:
        """Whether this arm can report character offsets (`TokenizerWithSpans`)."""
        return self._spans is not None

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Half-open character offsets of each token of `text` (`base.TokenizerWithSpans`).

        Raises `TokenizerUnavailable` if this arm has no offsets to give.
        """
        if self._spans is None:
            raise TokenizerUnavailable(
                f"{self.name}: this arm reports no character spans, so it cannot be scored "
                "with morphscore; it can still be counted with every other metric"
            )
        return self._spans(text)


class TokenizerUnavailable(RuntimeError):
    """A registered arm exists but could not be loaded this run.

    Two causes: every candidate in an HF arm's candidate list failed (`_load_hf_arm`,
    `_load_brahmic131k_arm`), or a file-backed arm's `tokenizer.json` (T1/T2/E1/T4, written
    by `tokenizers/train_*.py`) does not exist yet at `trained_tokenizer_path(name)`.
    Distinct from `KeyError`, which means the arm is not registered at all.

    Subclasses `RuntimeError` on purpose: the two loaders raised a bare `RuntimeError` for
    "no candidate loaded" before this class existed, and the tests pinning that behaviour
    (`pytest.raises(RuntimeError)`) still pass unchanged, since every `TokenizerUnavailable`
    *is* a `RuntimeError`. Callers that want to catch "unavailable this run" specifically —
    an experiment recording `unavailable_arms` rather than aborting — should catch this
    class rather than `RuntimeError`.
    """


def _family(name: str) -> str:
    """The arm-name prefix before its first underscore: `"T0_gpt2"` -> `"T0"`."""
    return name.split("_", 1)[0]


#: Arm-name suffixes that name a training corpus rather than an algorithm or a vocabulary
#: size, longest first so `_oracle_dcs` is recognised before the `_dcs` it ends with.
_VARIANT_SUFFIXES = ("_oracle_dcs", "_dcs")


def _variant(name: str) -> str:
    """The training-corpus label an arm name ends in: `"T5_morphbpe_raw_32k_dcs"` -> `"dcs"`.

    `""` for an arm with no such suffix, which is every arm predating Experiment 04. Only
    file-backed arms can carry one — an off-the-shelf arm is not trained on anything of
    this project's choosing — so only `_load_trained_arm` sets it.
    """
    for suffix in _VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return suffix[1:]
    return ""


def _normalise_spans(text: str, raw: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Turn one tokenizer's raw offsets into spans satisfying `base.spans_cover_text`.

    Three repairs, in this order, and every adapter goes through them so the contract has
    one implementation rather than three:

    1. **Clip to the previous span's end.** Raw offsets can overlap. A byte-level BPE may
       split one multi-byte character across two tokens, and both then report the same
       character range; the earlier token keeps the character and the later one collapses.
    2. **Trim whitespace off both edges.** `Metaspace` and byte-level pre-tokenizers
       attach the space *before* a word to that word's first token, so raw offsets tile
       the whole string including its spaces. Whitespace is not part of any morpheme and
       a boundary sitting on a space would be scored as a segmentation decision, so it is
       trimmed away. Whitespace *inside* a span survives: multi-word tokens are real.
    3. **Drop what is left empty.** A token that was only a space, only a `Metaspace`
       marker, or only the tail of a character an earlier token already covered carries
       no characters, so it has no span. This is why `len(spans(text))` is not a token
       count.

    Not a validator: it produces a conforming result from any input rather than rejecting
    a non-conforming one, which is what lets a single implementation serve three upstream
    libraries. The invariant itself is asserted in the tests, against `spans_cover_text`.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for raw_start, raw_end in raw:
        start = max(int(raw_start), cursor)
        end = min(int(raw_end), len(text))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        spans.append((start, end))
        cursor = end
    return spans


def _byte_boundaries_to_char_spans(
    text: str, token_byte_lengths: Sequence[int]
) -> list[tuple[int, int]]:
    """Character spans for a byte-level tokenizer that reports only token byte lengths.

    tiktoken has no offsets API: `decode_single_token_bytes` gives each token's bytes, and
    their lengths accumulate into a byte cursor over `text.encode("utf-8")`. Every byte
    position is then mapped back to a character position.

    **A boundary that falls inside a multi-byte character is rounded up to that
    character's end.** UTF-8 continuation bytes are not character boundaries, so a token
    that ends mid-character has no exact character offset; rounding up means the character
    belongs to whichever token holds its *first* byte, and the token that holds only its
    tail collapses to an empty span and is dropped by `_normalise_spans`. Rounding down
    instead would give two tokens the same character and break the covered-exactly-once
    half of the contract. It does not arise on SLP1, which is ASCII; it can on Devanagari,
    where every character is three bytes.
    """
    encoded = text.encode("utf-8")
    char_of_byte = [len(text)] * (len(encoded) + 1)
    is_char_start = [False] * (len(encoded) + 1)
    position = 0
    for index, character in enumerate(text):
        width = len(character.encode("utf-8"))
        is_char_start[position] = True
        for offset in range(width):
            char_of_byte[position + offset] = index
        position += width
    is_char_start[len(encoded)] = True

    def boundary(byte_index: int) -> int:
        if is_char_start[byte_index]:
            return char_of_byte[byte_index]
        return char_of_byte[byte_index] + 1

    raw: list[tuple[int, int]] = []
    cursor = 0
    for length in token_byte_lengths:
        if cursor >= len(encoded):
            break
        end = min(cursor + length, len(encoded))
        raw.append((boundary(cursor), boundary(end)))
        cursor = end
    return _normalise_spans(text, raw)


class TiktokenAdapter:
    """Wraps a `tiktoken.Encoding` as a plain `str -> list[int]` callable.

    Special-token strings occurring in a corpus are encoded as ordinary text rather than
    raising (`disallowed_special=()`).
    """

    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def __call__(self, text: str) -> list[int]:
        ids: list[int] = list(self._encoding.encode(text, disallowed_special=()))
        return ids

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans, via each token's byte length (`_byte_boundaries_to_char_spans`)."""
        ids = self._encoding.encode(text, disallowed_special=())
        lengths = [len(self._encoding.decode_single_token_bytes(token_id)) for token_id in ids]
        return _byte_boundaries_to_char_spans(text, lengths)


class HFAdapter:
    """Wraps a `transformers` tokenizer as a plain `str -> list[int]` callable.

    `add_special_tokens=False` because metrics encode individual words: BOS/EOS added per
    call would be counted once per word.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, text: str) -> list[int]:
        ids: list[int] = list(self._tokenizer.encode(text, add_special_tokens=False))
        return ids

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans from `return_offsets_mapping`, which only a *fast* tokenizer has.

        Raises `TokenizerUnavailable` on a slow (pure-Python) tokenizer: it can count
        tokens but has no offsets at all, and a caller that got an empty list back would
        read it as "this text has no tokens". `_FastReloadSpans` is the recovery path.
        """
        if not getattr(self._tokenizer, "is_fast", False):
            raise TokenizerUnavailable(
                "this transformers tokenizer loaded slow (pure-Python) and reports no "
                "character offsets; reload the same model id with use_fast=True"
            )
        encoding = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return _normalise_spans(text, encoding["offset_mapping"])


class TokenizersAdapter:
    """Wraps a raw `tokenizers.Tokenizer` (the Rust library, not `transformers`) as a
    plain `str -> list[int]` callable.

    Used for repos that ship a `tokenizer.json` but nothing `AutoTokenizer` can resolve
    (`_load_brahmic131k_arm`'s second tier) and for the file-backed T1/T2/E1 arms, which are
    always this exact type since they are loaded with `Tokenizer.from_file`.
    `add_special_tokens=False` for the same reason as `HFAdapter`: metrics encode
    individual words, so per-call special tokens would be counted once per word.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, text: str) -> list[int]:
        ids: list[int] = list(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return ids

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans from `Encoding.offsets`, which this library always provides."""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return _normalise_spans(text, encoding.offsets)


class ByteAdapter:
    """UTF-8 bytes as token ids: the `T7_byt5` arm's whole implementation.

    `__call__` is `list(text.encode("utf-8"))` — 256 possible ids, no vocabulary, no
    training corpus, nothing that can be unavailable.

    `spans` is the reason this is a class rather than a lambda. One *byte* is not a
    segmentation decision: two of the three bytes of a Devanagari character fall strictly
    inside it, and a boundary there would be scored by MorphScore as a claim about
    morphology that no tokenizer made. `_byte_boundaries_to_char_spans` therefore rounds
    every boundary up to a character edge, which collapses the continuation bytes into
    empty spans that `_normalise_spans` drops, leaving exactly one span per non-whitespace
    character. On SLP1, which is ASCII, one byte *is* one character and the two coincide.
    So `len(spans(text))` is a character count and `len(encode(text))` a byte count, and
    for Devanagari the second is three times the first.
    """

    def __call__(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def spans(self, text: str) -> list[tuple[int, int]]:
        """One span per non-whitespace character (see the class docstring)."""
        return _byte_boundaries_to_char_spans(text, [1] * len(text.encode("utf-8")))


def _hf_from_pretrained(model_id: str, token: str | None) -> Any:
    """`AutoTokenizer.from_pretrained`, imported lazily and monkeypatchable in tests.

    `transformers` is imported inside the call so that loading this module (and the
    tiktoken arm) does not pay for it. `token` is only passed when set, so an unset
    `HF_TOKEN` leaves the library's own credential discovery untouched.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"token": token} if token else {}
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def _hf_from_pretrained_fast(model_id: str, token: str | None) -> Any:
    """`AutoTokenizer.from_pretrained(..., use_fast=True)`, separately monkeypatchable.

    Deliberately *not* a keyword argument on `_hf_from_pretrained`: that function is the
    seam every existing test monkeypatches with a two-argument fake, and widening its
    signature would make those fakes silently wrong the day a caller started passing the
    third argument. Two names, two seams.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"token": token} if token else {}
    return AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)


class _FastReloadSpans:
    """Spans for an HF arm whose tokenizer loaded slow: reload it fast, once, on demand.

    `T0_gemma3` is the arm this exists for (docs/decisions.md, "Token spans for MorphScore
    deferred to Experiment 04"): a SentencePiece model can load as a slow tokenizer, which
    counts tokens correctly — so every Experiment 01-03 metric is unaffected — but reports
    no character offsets. Reloading is deferred to the first `spans` call so that an
    experiment which only counts tokens never pays for it, and cached afterwards.

    A successful reload is a change of what is being measured, so it is logged at WARNING
    and reported through `on_reload`, which suffixes the arm's `source_id` with `+fast`;
    `results.json` then says `"<model id>+fast"` and the substitution is visible in the
    results rather than only in the log.

    A failed reload raises `TokenizerUnavailable` from `spans` — the arm keeps working for
    every count-based metric and is unavailable only for MorphScore. `OSError` covers the
    hub failures, `ValueError` the "couldn't instantiate the backend tokenizer" a model
    with no fast converter raises, and `ImportError` the missing `sentencepiece`/`protobuf`
    conversion dependencies; anything else is a defect here and propagates.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        token: str | None,
        on_reload: Callable[[str], None],
    ) -> None:
        self._name = name
        self._model_id = model_id
        self._token = token
        self._on_reload = on_reload
        self._delegate: HFAdapter | None = None

    def __call__(self, text: str) -> list[tuple[int, int]]:
        if self._delegate is None:
            self._delegate = self._reload()
        return self._delegate.spans(text)

    def _reload(self) -> HFAdapter:
        try:
            tokenizer = _hf_from_pretrained_fast(self._model_id, self._token)
        except (OSError, ValueError, ImportError) as exc:
            raise TokenizerUnavailable(
                f"{self._name}: {self._model_id} loaded as a slow tokenizer and could not be "
                f"reloaded with use_fast=True, so it has no character spans: "
                f"{_first_line(str(exc))}"
            ) from exc
        if not getattr(tokenizer, "is_fast", False):
            raise TokenizerUnavailable(
                f"{self._name}: {self._model_id} is still a slow tokenizer after "
                "use_fast=True, so it has no character spans"
            )
        logger.warning(
            "%s: reloaded %s with use_fast=True for token spans; recorded as %s+fast",
            self._name,
            self._model_id,
            self._model_id,
        )
        self._on_reload(f"{self._model_id}+fast")
        return HFAdapter(tokenizer)


def _hf_spans_provider(
    loaded: LoadedTokenizer, adapter: HFAdapter, tokenizer: Any, model_id: str, token: str | None
) -> Callable[[str], list[tuple[int, int]]]:
    """The adapter's own offsets when `tokenizer` is fast, else a lazy fast reload of it."""
    if getattr(tokenizer, "is_fast", False):
        return adapter.spans

    def record(source_id: str) -> None:
        loaded.source_id = source_id

    return _FastReloadSpans(loaded.name, model_id, token, record)


def _hf_vocab_size(tokenizer: Any) -> int:
    """Size of the id space, added special tokens included.

    `len(tokenizer)` rather than `tokenizer.vocab_size`: the latter reports the base
    vocabulary only, so the Llama-4 Scout tokenizer says 200000 where its ids in fact run
    to 201134. The id space is what a model's embedding matrix has to cover, so it is the
    number `results.json` should carry.
    """
    return int(len(tokenizer))


def _first_line(text: str) -> str:
    """First non-empty line of an exception message; hub errors run to many paragraphs.

    The full text still reaches the WARNING log; this keeps the raised summary readable.
    """
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


def _load_tiktoken_arm(name: str, encoding_name: str) -> LoadedTokenizer:
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    adapter = TiktokenAdapter(encoding)
    logger.info("%s: loaded tiktoken encoding %s", name, encoding_name)
    return LoadedTokenizer(
        name=name,
        source_id=encoding_name,
        vocab_size=int(encoding.n_vocab),
        _encode=adapter,
        family=_family(name),
        attempted=(encoding_name,),
        _spans=adapter.spans,
    )


def _load_hf_arm(name: str, candidates: tuple[str, ...]) -> LoadedTokenizer:
    """First candidate that loads wins; every failure is logged at WARNING.

    Only `OSError` is caught, and that is deliberate: it is what "this repository is not
    available to you" looks like from this stack. `huggingface_hub`'s `GatedRepoError` and
    `RepositoryNotFoundError` derive from `HfHubHTTPError` -> `requests.HTTPError` ->
    `IOError` (= `OSError`), as do plain network failures, and `transformers` re-raises a
    repo it cannot resolve as `OSError("Can't load tokenizer for ...")`. Anything else — a
    `TypeError` from a signature change, an `ImportError` from a missing extra, a bug in
    `_hf_vocab_size` — is a defect in this repository, not an unavailable candidate, and
    must propagate rather than be silently retried against the next model id and then
    reported as "no candidate could be loaded".

    One known gap: `huggingface_hub.errors.EntryNotFoundError` (a repo that exists but is
    missing a requested file) is *not* an `OSError`. `transformers` converts that case
    into the `OSError` above before it reaches here; if a future version stops doing so,
    the arm will raise rather than fall through to its next candidate — which is the safe
    direction to fail, since a mirror missing its tokenizer files is not a licence gate.

    Raises `TokenizerUnavailable` (a `RuntimeError`) naming every candidate and its error
    when none loads, since an arm that silently disappears would leave a hole in the
    experiment's results table.
    """
    token = os.environ.get("HF_TOKEN")
    failures: list[str] = []
    attempted: list[str] = []
    for model_id in candidates:
        attempted.append(model_id)
        try:
            tokenizer = _hf_from_pretrained(model_id, token)
        except OSError as exc:
            logger.warning("%s: could not load %s: %s", name, model_id, exc)
            failures.append(f"{model_id}: {_first_line(str(exc))}")
            continue
        if model_id != candidates[0]:
            logger.warning(
                "%s: substituting %s for %s; record this in docs/decisions.md",
                name,
                model_id,
                candidates[0],
            )
        logger.info("%s: loaded %s", name, model_id)
        adapter = HFAdapter(tokenizer)
        loaded = LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=_hf_vocab_size(tokenizer),
            _encode=adapter,
            family=_family(name),
            attempted=tuple(attempted),
        )
        loaded._spans = _hf_spans_provider(loaded, adapter, tokenizer, model_id, token)
        return loaded
    raise TokenizerUnavailable(
        f"{name}: no candidate tokenizer could be loaded. Tried:\n  " + "\n  ".join(failures)
    )


def _load_brahmic131k_arm(name: str) -> LoadedTokenizer:
    """`T3_brahmic131k`: try three adapter kinds against one repository id, in order.

    Unlike every other HF arm, this one has no alternate repository to fall back to; what
    varies is *how* the repo's tokenizer files are read. `AutoTokenizer` (the same path
    every T0/T3 HF arm uses) wins in practice — the repo carries a plain `tokenizer.json` —
    but the two more permissive fallbacks stay in place per the exp02 plan, for a future
    repo revision that drops the `transformers`-compatible config:

    1. `AutoTokenizer.from_pretrained` (`HFAdapter`) — the normal path.
    2. `tokenizers.Tokenizer.from_pretrained` (`TokenizersAdapter`) — reads the same
       `tokenizer.json` directly, for repos `AutoTokenizer` cannot resolve (e.g. no
       `tokenizer_config.json`).
    3. Any `*.tiktoken` mergeable-ranks file in the repo, wrapped as a `tiktoken.Encoding`
       (`TiktokenAdapter`). Last resort: the pre-tokenizer regex is not published by a
       ranks-only file, so a generic GPT-2-style split pattern is assumed, which can
       under- or over-segment relative to the model's own tokenizer. Only exercised if
       tiers 1 and 2 both fail.

    `attempted` records the repository id once per tier tried, so it can hold repeats;
    that mirrors "one candidate, several adapters" rather than "several candidates".
    Raises `TokenizerUnavailable` naming every tier's error when all three fail.
    """
    model_id = T3_BRAHMIC131K_MODEL_ID
    token = os.environ.get("HF_TOKEN")
    attempted: list[str] = []
    failures: list[str] = []

    attempted.append(model_id)
    try:
        tokenizer = _hf_from_pretrained(model_id, token)
    except OSError as exc:
        logger.warning("%s: AutoTokenizer could not load %s: %s", name, model_id, exc)
        failures.append(f"AutoTokenizer({model_id}): {_first_line(str(exc))}")
    else:
        logger.info("%s: loaded %s via AutoTokenizer", name, model_id)
        adapter = HFAdapter(tokenizer)
        loaded = LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=_hf_vocab_size(tokenizer),
            _encode=adapter,
            family=_family(name),
            attempted=tuple(attempted),
        )
        loaded._spans = _hf_spans_provider(loaded, adapter, tokenizer, model_id, token)
        return loaded

    attempted.append(model_id)
    try:
        from tokenizers import Tokenizer as _RawTokenizer

        fast_tokenizer = _RawTokenizer.from_pretrained(model_id)
    except OSError as exc:
        logger.warning("%s: tokenizers.Tokenizer could not load %s: %s", name, model_id, exc)
        failures.append(f"tokenizers.Tokenizer({model_id}): {_first_line(str(exc))}")
    else:
        logger.info("%s: loaded %s via tokenizers.Tokenizer", name, model_id)
        raw_adapter = TokenizersAdapter(fast_tokenizer)
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=int(fast_tokenizer.get_vocab_size()),
            _encode=raw_adapter,
            family=_family(name),
            attempted=tuple(attempted),
            _spans=raw_adapter.spans,
        )

    attempted.append(model_id)
    try:
        encoding, ranks_file = _load_tiktoken_style_hub_file(model_id)
    except OSError as exc:
        logger.warning("%s: no usable .tiktoken file in %s: %s", name, model_id, exc)
        failures.append(f".tiktoken({model_id}): {_first_line(str(exc))}")
    else:
        logger.info("%s: loaded %s/%s via tiktoken.Encoding", name, model_id, ranks_file)
        tiktoken_adapter = TiktokenAdapter(encoding)
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=int(encoding.n_vocab),
            _encode=tiktoken_adapter,
            family=_family(name),
            attempted=tuple(attempted),
            _spans=tiktoken_adapter.spans,
        )

    raise TokenizerUnavailable(
        f"{name}: no adapter could load {model_id}. Tried:\n  " + "\n  ".join(failures)
    )


def _load_tiktoken_style_hub_file(model_id: str) -> tuple[Any, str]:
    """Last-resort tier for `_load_brahmic131k_arm`: a bare `.tiktoken` ranks file.

    Lists the repo's files, downloads the first one ending `.tiktoken`, and builds a
    `tiktoken.Encoding` from its mergeable ranks. Raises `OSError` (folded into the
    caller's failure message) if the repo has no such file or the hub call itself fails;
    `huggingface_hub`'s listing/download errors are `OSError` subclasses, matching every
    other candidate-walk in this module.

    The pre-tokenizer split pattern is not recoverable from a ranks-only file, so this
    assumes GPT-2's regex (`tiktoken.get_encoding("gpt2")._pat_str`) — a reasonable
    default for a BPE tokenizer, but not guaranteed to match the original model's own
    segmentation. Returns the encoding and the repo-relative filename used, for logging.
    """
    import huggingface_hub
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    repo_files = huggingface_hub.list_repo_files(model_id)
    ranks_files = [path for path in repo_files if path.endswith(".tiktoken")]
    if not ranks_files:
        raise OSError(f"{model_id} has no *.tiktoken file")
    ranks_file = ranks_files[0]
    local_path = huggingface_hub.hf_hub_download(model_id, ranks_file)
    mergeable_ranks = load_tiktoken_bpe(local_path)
    pat_str = tiktoken.get_encoding("gpt2")._pat_str  # noqa: SLF001 - no public accessor
    encoding = tiktoken.Encoding(
        name=f"{model_id}/{ranks_file}",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    return encoding, ranks_file


def _repo_root() -> Path:
    """The repository root, i.e. the parent of `src/` (three levels above this file)."""
    return Path(__file__).resolve().parents[3]


def _tokenizer_dir(root: Path) -> Path:
    """Where trained (T1/T2/E1/T4) tokenizers live: `$SANSKRIT_TOK_TOKENIZER_DIR`, defaulting to
    `outputs/tokenizers`, resolved against `root` when relative."""
    value = os.environ.get("SANSKRIT_TOK_TOKENIZER_DIR", "outputs/tokenizers")
    path = Path(value)
    return path if path.is_absolute() else root / path


def trained_tokenizer_path(name: str) -> Path:
    """Where a file-backed arm's `tokenizer.json` lives (or should be written to).

    `<tokenizer_dir>/<name>/tokenizer.json`, `tokenizer_dir` from `_tokenizer_dir`
    (`$SANSKRIT_TOK_TOKENIZER_DIR`, defaulting to `outputs/tokenizers` under the repo
    root). `tokenizers/train_bpe.py` and `train_unigram.py` (Task 4) import this so the
    file they write and the file `_load_trained_arm` reads can never drift apart.
    """
    return _tokenizer_dir(_repo_root()) / name / "tokenizer.json"


def _load_trained_arm(name: str) -> LoadedTokenizer:
    """Load a file-backed T1/T2/E1/T4 arm from its trained `tokenizer.json`.

    Raises `TokenizerUnavailable` naming the expected path if the file does not exist —
    the arm is registered (it is a known name) but has not been trained yet, which is
    exactly the condition this exception exists to distinguish from `KeyError`.
    """
    path = trained_tokenizer_path(name)
    if not path.exists():
        raise TokenizerUnavailable(
            f"{name}: trained tokenizer file not found at {path}; train it with "
            "sanskrit_tok.tokenizers.train_bpe / train_unigram, or set "
            "SANSKRIT_TOK_TOKENIZER_DIR to where it already lives"
        )
    from tokenizers import Tokenizer as _RawTokenizer

    tokenizer = _RawTokenizer.from_file(str(path))
    adapter = TokenizersAdapter(tokenizer)
    logger.info("%s: loaded trained tokenizer from %s", name, path)
    return LoadedTokenizer(
        name=name,
        source_id=str(path),
        vocab_size=int(tokenizer.get_vocab_size()),
        _encode=adapter,
        family=_family(name),
        variant=_variant(name),
        attempted=(str(path),),
        _spans=adapter.spans,
    )


def _load_byte_arm(name: str) -> LoadedTokenizer:
    """Load the byte-level `T7_byt5` arm, which is constructed rather than fetched.

    Never raises `TokenizerUnavailable`: there is no file to be missing and no repository
    to be gated, which is precisely what makes this arm the floor of every comparison —
    it is available on any machine, offline, at every vocabulary size, forever.
    """
    adapter = ByteAdapter()
    return LoadedTokenizer(
        name=name,
        source_id="bytes/utf-8",
        vocab_size=T7_BYTE_VOCAB_SIZE,
        _encode=adapter,
        family=_family(name),
        variant=_variant(name),
        attempted=("bytes",),
        _spans=adapter.spans,
    )


#: The matched English control family (CLAUDE.md §6), in two halves that differ only in how
#: much English text they saw. The **pair-matched** arms (`E1_bpe_32k`, ...) are trained on
#: the whole English side of the Sāmayik + Itihāsa training splits — the same *sentences*
#: the Sanskrit arms saw, one sentence per pair. The **byte-matched** arms (the `_bm`
#: suffix) are trained on a deterministic subsample of those lines cut to the Sanskrit
#: corpus's UTF-8 byte count, because the pair-matched English corpus is 16,554,871 bytes
#: against the Sanskrit corpus's 11,209,356 — 48% more text at the same vocabulary size,
#: which a reviewer can read as the reason the English side tokenizes more cheaply
#: (docs/decisions.md, 2026-09-08, "Byte-matched English control arms `E1_*_bm`"). Neither
#: is the control: they bracket it, one matching sentences and one matching bytes.
E1_ARMS: tuple[str, ...] = (
    "E1_bpe_32k",
    "E1_bpe_64k",
    "E1_bpe_128k",
    "E1_unigram_32k",
    "E1_unigram_64k",
    "E1_unigram_128k",
    "E1_bpe_32k_bm",
    "E1_bpe_64k_bm",
    "E1_bpe_128k_bm",
    "E1_unigram_32k_bm",
    "E1_unigram_64k_bm",
    "E1_unigram_128k_bm",
)


#: The fourteen Experiment 04 arms, every one trained by this project on the DCS training
#: split (docs/decisions.md, 2026-09-05, "Experiment 04: DCS is the gold source and the
#: first monolingual training corpus"). File-backed like T1/T2/E1/T4 and loaded by the same
#: loader; what the names say is which of the four DCS corpora each was trained on —
#: `_dcs` the sandhied text (or, for T5/T6, its marked form), `_oracle_dcs` the gold
#: segmentation. `T5`/`T6` are the morpheme-constrained (MorphBPE-hard) families, and their
#: matched unconstrained controls are the `T1_*_dcs` / `T4_*_oracle_dcs` arms beside them:
#: same corpus sentences, same vocabulary size, same trainer, no boundary marker. The
#: `T5_morphbpe_rawseg_*` pair is the same constraint marked at DCS's gold **segment**
#: boundaries only, so it is the arm whose result owes nothing to the stem heuristic
#: (docs/decisions.md, 2026-09-05, "Stem boundaries are heuristic ... gold-segment-only
#: constrained arm added").
DCS_ARMS: tuple[str, ...] = (
    "T1_bpe_raw_32k_dcs",
    "T1_bpe_raw_64k_dcs",
    "T2_unigram_raw_32k_dcs",
    "T2_unigram_raw_64k_dcs",
    "T4_bpe_split_32k_oracle_dcs",
    "T4_bpe_split_64k_oracle_dcs",
    "T4_unigram_split_32k_oracle_dcs",
    "T4_unigram_split_64k_oracle_dcs",
    "T5_morphbpe_raw_32k_dcs",
    "T5_morphbpe_raw_64k_dcs",
    "T5_morphbpe_rawseg_32k_dcs",
    "T5_morphbpe_rawseg_64k_dcs",
    "T6_morphbpe_split_32k_dcs",
    "T6_morphbpe_split_64k_dcs",
)


#: Arm key -> zero-argument loader. Loading is deferred: an experiment that needs one arm
#: does not download the others, and a gated arm fails only when it is actually asked for.
REGISTRY: dict[str, Callable[[], LoadedTokenizer]] = {
    "T0_o200k": functools.partial(_load_tiktoken_arm, "T0_o200k", T0_O200K_ENCODING),
    "T0_llama4": functools.partial(_load_hf_arm, "T0_llama4", T0_LLAMA4_CANDIDATES),
    "T0_gemma3": functools.partial(_load_hf_arm, "T0_gemma3", T0_GEMMA3_CANDIDATES),
    "T0_gpt2": functools.partial(_load_hf_arm, "T0_gpt2", T0_GPT2_CANDIDATES),
    "T3_sarvam": functools.partial(_load_hf_arm, "T3_sarvam", T3_SARVAM_CANDIDATES),
    "T3_sutra": functools.partial(_load_hf_arm, "T3_sutra", T3_SUTRA_CANDIDATES),
    "T3_brahmic131k": functools.partial(_load_brahmic131k_arm, "T3_brahmic131k"),
    "T3_indicsuper": functools.partial(_load_hf_arm, "T3_indicsuper", T3_INDICSUPER_CANDIDATES),
    "T1_bpe_raw_32k": functools.partial(_load_trained_arm, "T1_bpe_raw_32k"),
    "T1_bpe_raw_64k": functools.partial(_load_trained_arm, "T1_bpe_raw_64k"),
    "T1_bpe_raw_128k": functools.partial(_load_trained_arm, "T1_bpe_raw_128k"),
    "T2_unigram_raw_32k": functools.partial(_load_trained_arm, "T2_unigram_raw_32k"),
    "T2_unigram_raw_64k": functools.partial(_load_trained_arm, "T2_unigram_raw_64k"),
    "T2_unigram_raw_128k": functools.partial(_load_trained_arm, "T2_unigram_raw_128k"),
    **{name: functools.partial(_load_trained_arm, name) for name in E1_ARMS},
    "T4_bpe_split_32k": functools.partial(_load_trained_arm, "T4_bpe_split_32k"),
    "T4_bpe_split_64k": functools.partial(_load_trained_arm, "T4_bpe_split_64k"),
    "T4_unigram_split_32k": functools.partial(_load_trained_arm, "T4_unigram_split_32k"),
    "T4_unigram_split_64k": functools.partial(_load_trained_arm, "T4_unigram_split_64k"),
    "T7_byt5": functools.partial(_load_byte_arm, "T7_byt5"),
    **{name: functools.partial(_load_trained_arm, name) for name in DCS_ARMS},
}


def list_tokenizers(family: str | None = None, variant: str | None = None) -> list[str]:
    """Every registered arm key, sorted.

    `family` (e.g. `"T3"`) narrows to that name prefix; `variant` (`"dcs"`, `"oracle_dcs"`,
    or `""` for the arms carrying no corpus suffix) narrows to that training corpus. Both
    are needed together whenever a family spans experiments: `list_tokenizers(family="T4")`
    is eight arms, four trained on ByT5-split parallel text (Experiment 03, variant `""`)
    and four on gold-split DCS (Experiment 04, variant `"oracle_dcs"`).
    """
    names = [
        name
        for name in REGISTRY
        if (family is None or _family(name) == family)
        and (variant is None or _variant(name) == variant)
    ]
    return sorted(names)


def load_tokenizer(name: str) -> LoadedTokenizer:
    """Load the arm called `name`.

    Downloads on first use and reuses the Hugging Face / tiktoken caches afterwards; the
    result is not cached here, so callers that need an arm repeatedly should hold on to it.

    Raises `KeyError` naming the known arms if `name` is not registered, and `RuntimeError`
    if a registered arm has no loadable candidate.
    """
    try:
        loader = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown tokenizer arm {name!r}; known arms: {', '.join(list_tokenizers())}"
        ) from None
    return loader()
