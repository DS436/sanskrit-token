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
tokenizers, not as a matched experiment. `T3_*` (off-the-shelf Indic tokenizers) are the
same kind of arm, one family over. `T1_*`/`T2_*` are the opposite: trained from scratch by
this project at matched vocabulary sizes (CLAUDE.md §5), so they are read from a
`tokenizer.json` file on disk rather than downloaded, and are absent — `TokenizerUnavailable`
— until `tokenizers/train_bpe.py` / `train_unigram.py` (Task 4) have written one. `E1_*`
is the matched *English* control family (CLAUDE.md §6): the same two algorithms at the
same two vocabulary sizes, trained on the English side of the same corpus, so a TPP ratio
against an E1 arm holds algorithm, vocabulary size and training domain constant on both
sides. It is file-backed for exactly the same reason as T1/T2 and loads the same way.

**Gated repositories.** `meta-llama/*` and `google/*` need an accepted licence and an
`HF_TOKEN`. Each HF arm therefore declares a list of candidate ids, tried in order, first
success wins: the official gated id, a mirror of the same tokenizer, then the previous
model generation and its mirror. `HF_TOKEN` is passed when the environment sets one. Any
load below the first candidate is a substitution and is recorded in `docs/decisions.md`
(CLAUDE.md §11), because the tokenizer measured is then not the one the arm is named for.

**`family`.** Every `LoadedTokenizer` carries the arm-name prefix before its first
underscore (`"T0_gpt2"` -> `"T0"`), so an experiment or a metric summary can group or
filter arms without parsing the name itself; `list_tokenizers(family=...)` filters the
registry the same way.

**`TokenizerUnavailable`.** A registered arm can still fail to produce a tokenizer this
run — every HF candidate is gated or unpublished, or a trained arm's `tokenizer.json` has
not been written yet. That is different from `KeyError` (the arm is not registered at
all): experiments catch `TokenizerUnavailable` specifically, log a WARNING, and record the
arm under `unavailable_arms` rather than aborting the whole run over one missing model.
"""

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "REGISTRY",
    "T0_GEMMA3_CANDIDATES",
    "T0_GPT2_CANDIDATES",
    "T0_LLAMA4_CANDIDATES",
    "T0_O200K_ENCODING",
    "T3_BRAHMIC131K_MODEL_ID",
    "T3_INDICSUPER_CANDIDATES",
    "T3_SARVAM_CANDIDATES",
    "T3_SUTRA_CANDIDATES",
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


@dataclass
class LoadedTokenizer:
    """A tokenizer arm, ready to encode, carrying its own provenance.

    Satisfies the `Tokenizer` protocol (`base.py`), so it can be handed straight to any
    metric. `name` is the arm key (`"T0_llama4"`); `source_id` is the model id, tiktoken
    encoding name, or trained-tokenizer file path *actually* loaded, which may be a
    substitute for the arm's first choice; `vocab_size` is the full id space including
    added special tokens.

    `family` is the arm-name prefix before its first underscore (`"T0"`, `"T3"`, ...),
    computed once at load time so callers never re-derive it from `name`. `attempted` is
    every candidate id (or, for a file-backed arm, the one file path) tried before the
    winner, winner last — a one-element tuple whenever an arm has only ever had one
    candidate. Both default to empty so hand-built fakes in tests need not set them.
    """

    name: str
    source_id: str
    vocab_size: int
    _encode: Callable[[str], list[int]] = field(repr=False)
    family: str = ""
    attempted: tuple[str, ...] = ()

    def encode(self, text: str) -> list[int]:
        """Token ids for `text`, with no special tokens added."""
        return self._encode(text)


class TokenizerUnavailable(RuntimeError):
    """A registered arm exists but could not be loaded this run.

    Two causes: every candidate in an HF arm's candidate list failed (`_load_hf_arm`,
    `_load_brahmic131k_arm`), or a file-backed arm's `tokenizer.json` (T1/T2/E1, written
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
        family=_family(name),
        attempted=(encoding_name,),
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
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=_hf_vocab_size(tokenizer),
            _encode=HFAdapter(tokenizer),
            family=_family(name),
            attempted=tuple(attempted),
        )
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
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=_hf_vocab_size(tokenizer),
            _encode=HFAdapter(tokenizer),
            family=_family(name),
            attempted=tuple(attempted),
        )

    attempted.append(model_id)
    try:
        from tokenizers import Tokenizer as _RawTokenizer

        fast_tokenizer = _RawTokenizer.from_pretrained(model_id)
    except OSError as exc:
        logger.warning("%s: tokenizers.Tokenizer could not load %s: %s", name, model_id, exc)
        failures.append(f"tokenizers.Tokenizer({model_id}): {_first_line(str(exc))}")
    else:
        logger.info("%s: loaded %s via tokenizers.Tokenizer", name, model_id)
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=int(fast_tokenizer.get_vocab_size()),
            _encode=TokenizersAdapter(fast_tokenizer),
            family=_family(name),
            attempted=tuple(attempted),
        )

    attempted.append(model_id)
    try:
        encoding, ranks_file = _load_tiktoken_style_hub_file(model_id)
    except OSError as exc:
        logger.warning("%s: no usable .tiktoken file in %s: %s", name, model_id, exc)
        failures.append(f".tiktoken({model_id}): {_first_line(str(exc))}")
    else:
        logger.info("%s: loaded %s/%s via tiktoken.Encoding", name, model_id, ranks_file)
        return LoadedTokenizer(
            name=name,
            source_id=model_id,
            vocab_size=int(encoding.n_vocab),
            _encode=TiktokenAdapter(encoding),
            family=_family(name),
            attempted=tuple(attempted),
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
    """Where trained (T1/T2/E1) tokenizers live: `$SANSKRIT_TOK_TOKENIZER_DIR`, defaulting to
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
    """Load a file-backed T1/T2/E1 arm from its trained `tokenizer.json`.

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
    logger.info("%s: loaded trained tokenizer from %s", name, path)
    return LoadedTokenizer(
        name=name,
        source_id=str(path),
        vocab_size=int(tokenizer.get_vocab_size()),
        _encode=TokenizersAdapter(tokenizer),
        family=_family(name),
        attempted=(str(path),),
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
    "T2_unigram_raw_32k": functools.partial(_load_trained_arm, "T2_unigram_raw_32k"),
    "T2_unigram_raw_64k": functools.partial(_load_trained_arm, "T2_unigram_raw_64k"),
    "E1_bpe_32k": functools.partial(_load_trained_arm, "E1_bpe_32k"),
    "E1_bpe_64k": functools.partial(_load_trained_arm, "E1_bpe_64k"),
    "E1_unigram_32k": functools.partial(_load_trained_arm, "E1_unigram_32k"),
    "E1_unigram_64k": functools.partial(_load_trained_arm, "E1_unigram_64k"),
}


def list_tokenizers(family: str | None = None) -> list[str]:
    """Every registered arm key, sorted; `family` (e.g. `"T3"`) narrows to that prefix."""
    names = REGISTRY if family is None else (n for n in REGISTRY if _family(n) == family)
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
