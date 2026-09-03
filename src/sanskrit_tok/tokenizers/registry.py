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

The T0 arms are "existing practice" (CLAUDE.md §2.5), never a controlled comparison: their
vocabulary sizes differ, so their fertilities are read as evidence about deployed
tokenizers, not as a matched experiment.

**Gated repositories.** `meta-llama/*` and `google/*` need an accepted licence and an
`HF_TOKEN`. Each HF arm therefore declares a list of candidate ids, tried in order, first
success wins: the official gated id, a mirror of the same tokenizer, then the previous
model generation and its mirror. `HF_TOKEN` is passed when the environment sets one. Any
load below the first candidate is a substitution and is recorded in `docs/decisions.md`
(CLAUDE.md §11), because the tokenizer measured is then not the one the arm is named for.
"""

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "REGISTRY",
    "T0_GEMMA3_CANDIDATES",
    "T0_LLAMA4_CANDIDATES",
    "T0_O200K_ENCODING",
    "HFAdapter",
    "LoadedTokenizer",
    "TiktokenAdapter",
    "list_tokenizers",
    "load_tokenizer",
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


@dataclass
class LoadedTokenizer:
    """A tokenizer arm, ready to encode, carrying its own provenance.

    Satisfies the `Tokenizer` protocol (`base.py`), so it can be handed straight to any
    metric. `name` is the arm key (`"T0_llama4"`); `source_id` is the model id or tiktoken
    encoding name *actually* loaded, which may be a substitute for the arm's first choice;
    `vocab_size` is the full id space including added special tokens.
    """

    name: str
    source_id: str
    vocab_size: int
    _encode: Callable[[str], list[int]] = field(repr=False)

    def encode(self, text: str) -> list[int]:
        """Token ids for `text`, with no special tokens added."""
        return self._encode(text)


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


def _hf_from_pretrained(model_id: str, token: str | None) -> Any:
    """`AutoTokenizer.from_pretrained`, imported lazily and monkeypatchable in tests.

    `transformers` is imported inside the call so that loading this module (and the
    tiktoken arm) does not pay for it. `token` is only passed when set, so an unset
    `HF_TOKEN` leaves the library's own credential discovery untouched.
    """
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"token": token} if token else {}
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


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
    logger.info("%s: loaded tiktoken encoding %s", name, encoding_name)
    return LoadedTokenizer(
        name=name,
        source_id=encoding_name,
        vocab_size=int(encoding.n_vocab),
        _encode=TiktokenAdapter(encoding),
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

    Raises `RuntimeError` naming every candidate and its error when none loads, since an
    arm that silently disappears would leave a hole in the experiment's results table.
    """
    token = os.environ.get("HF_TOKEN")
    failures: list[str] = []
    for model_id in candidates:
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
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=_hf_vocab_size(tokenizer),
            _encode=HFAdapter(tokenizer),
        )
    raise RuntimeError(
        f"{name}: no candidate tokenizer could be loaded. Tried:\n  " + "\n  ".join(failures)
    )


#: Arm key -> zero-argument loader. Loading is deferred: an experiment that needs one arm
#: does not download the others, and a gated arm fails only when it is actually asked for.
REGISTRY: dict[str, Callable[[], LoadedTokenizer]] = {
    "T0_o200k": functools.partial(_load_tiktoken_arm, "T0_o200k", T0_O200K_ENCODING),
    "T0_llama4": functools.partial(_load_hf_arm, "T0_llama4", T0_LLAMA4_CANDIDATES),
    "T0_gemma3": functools.partial(_load_hf_arm, "T0_gemma3", T0_GEMMA3_CANDIDATES),
}


def list_tokenizers() -> list[str]:
    """Every registered arm key, sorted."""
    return sorted(REGISTRY)


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
