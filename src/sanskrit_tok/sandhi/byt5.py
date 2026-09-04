"""ByT5-Sanskrit sandhi splitter: Devanagari in, space-separated SLP1 segments out.

The model is `chronbmm/sanskrit5-multitask` (Nehrdich, Hellwig & Keutzer, EMNLP Findings
2024), run in its segmentation mode: the authors' inference code prefixes the input with
`"S "`, feeds IAST, and decodes unsandhied IAST segments within a 512-byte window
(docs/decisions.md, "Experiment 03 sandhi splitter"). The decoded string separates
segments with `SEGMENT_SEPARATOR` (`_`), not with spaces — verified against the real
checkpoint, and the reason `split` normalises before returning. This wrapper is the only
place in the project that knows any of that. Everything outside `sandhi/` speaks SLP1
(CLAUDE.md §2.3); IAST is an implementation detail of this file, and the two conversions
that hide it are `from_slp1(to_slp1(text, "devanagari"), "iast")` on the way in and
`to_slp1(output, "iast")` on the way out.

Four things this class does beyond calling `generate`:

* **Cache.** Every result is written to a `SplitCache` keyed by the sha256 of the input
  sentence, flushed per batch. Splitting the corpus takes hours on this hardware; nothing
  should ever be split twice, within a run or across runs.
* **Chunking.** An input whose IAST form exceeds `MAX_BYTES` would be silently truncated
  by the model. Such inputs are cut at danda/`|`/`.` boundaries (then spaces, then, as a
  last resort, the byte limit) into pieces that fit, split separately, and rejoined with
  a space. `stats["n_chunked"]` counts the inputs this happened to, so a run can report
  how much of its output went through a seam.
* **Ordering.** Batches are sorted by length so a batch pads to something near its own
  longest member rather than the corpus's; the input order is restored before returning.
  Repeated sentences inside one call are split once.
* **Laziness.** `torch` and `transformers` are imported inside the loader, not at module
  import, and the checkpoint is loaded on the first cache-missing `split`. A splitter
  whose sentences are all cached — the tokenizer-corpus builder's normal case — never
  loads 2.3 GB of weights, and the offline test suite constructs one in milliseconds.

`_generate_iast` is the seam: it takes already-prefixed IAST strings and returns decoded
IAST. Tests monkeypatch it; every other step above is real code under test.
"""

import logging
import time
from collections.abc import Sequence
from typing import Any

from sanskrit_tok.encoding import from_slp1, to_slp1
from sanskrit_tok.sandhi.cache import SplitCache

__all__ = [
    "CHUNK_MAX_BYTES",
    "MAX_BYTES",
    "MAX_OUTPUT_TOKENS",
    "SEGMENTATION_PREFIX",
    "SEGMENT_SEPARATOR",
    "SPLITTER_CANDIDATES",
    "SandhiSplitter",
    "normalise_segments",
    "pick_device",
]

logger = logging.getLogger(__name__)

#: Model ids tried in order, first success wins (the registry's convention, CLAUDE.md §6).
#: The first is the multitask ByT5-Sanskrit checkpoint the decisions log selected; the
#: second is the segmentation-only hackathon checkpoint, kept as a fallback. A load below
#: the first candidate is a substitution and must be recorded in `docs/decisions.md`.
SPLITTER_CANDIDATES: tuple[str, ...] = (
    "chronbmm/sanskrit5-multitask",
    "buddhist-nlp/byt5-sanskrit-analyzer-hackathon",
)

#: The multitask model's task tag for segmentation, from the authors' inference code
#: (`input_texts = ["S " + text]`, dharmamitra/byt5-sanskrit-analyzers).
SEGMENTATION_PREFIX = "S "

#: What the model puts between two segments it has unsandhied. The authors' inference
#: code post-processes it away; measured on `chronbmm/sanskrit5-multitask`, the raw decode
#: of `S <iast>` is `viśvāsa_kāraṇāt_eva_...`, so a bare `batch_decode` would leave a
#: whole sentence as one whitespace unit and every fertility and TPP number computed over
#: it would be wrong. `split` maps it to a space.
SEGMENT_SEPARATOR = "_"

#: The model's context in bytes (it is byte-level: one token per UTF-8 byte). Inputs
#: longer than this are chunked rather than truncated.
MAX_BYTES = 512

#: The budget a *chunk* is cut to. What the model tokenises is the chunk plus
#: `SEGMENTATION_PREFIX` plus an EOS token, so chunking to `MAX_BYTES` would hand the
#: tokeniser `MAX_BYTES + 3` bytes and `truncation=True` would silently drop the last
#: three — precisely the sentence-final material a segmenter is worst at recovering.
CHUNK_MAX_BYTES = MAX_BYTES - len(SEGMENTATION_PREFIX) - 1

#: Characters a chunk may end on, in preference over a bare space. Chunking runs on the
#: **IAST** form, where `from_slp1` has already rendered the Devanagari danda as `|`; `.`
#: is the SLP1 danda, kept because a caller may hand `chunk_text` SLP1 directly. Both are
#: sentence boundaries, so a chunk seam falls where a sandhi boundary would not have
#: crossed anyway. (Devanagari `।`/`॥` never reach this function and are not listed.)
_BOUNDARY_CHARS = ("|", ".")

#: Output cap, in generated tokens (= bytes). Segmentation output is *longer* than its
#: input — every restored boundary adds a separator byte and undone sandhi restores
#: elided phonemes — so capping generation at the input budget would truncate exactly the
#: longest sentences. The cap is therefore sized from the batch's own longest input:
#: `min(OUTPUT_TOKENS_FACTOR * longest_input_bytes + OUTPUT_TOKENS_SLACK,
#: MAX_OUTPUT_TOKENS)`, generous enough never to bind in practice and bounded so a
#: degenerate repeat loop cannot run forever.
OUTPUT_TOKENS_FACTOR = 2
OUTPUT_TOKENS_SLACK = 32
MAX_OUTPUT_TOKENS = 1024


def pick_device() -> str:
    """The best available torch device name: `"mps"`, else `"cuda"`, else `"cpu"`.

    Apple Silicon first because that is what this project's hardware is; the CUDA branch
    is here so the same code runs unchanged on a rented GPU (docs/decisions.md, "Splitter
    throughput rule"). Imports `torch` on call, not at module import.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _model_revision(model_id: str) -> str:
    """The Hugging Face commit sha for `model_id` (module-level so tests can patch it)."""
    from huggingface_hub import model_info

    return str(model_info(model_id).sha)


def _byte_prefix(text: str, limit: int) -> str:
    """The longest prefix of `text` that fits in `limit` UTF-8 bytes, cut on a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def chunk_text(text: str, limit: int = CHUNK_MAX_BYTES) -> list[str]:
    """Cut `text` into pieces of at most `limit` UTF-8 bytes, preferring sentence seams.

    Each piece is the longest prefix that fits, ending at the last `|`/`.` inside it;
    failing that at the last space; failing that at the byte limit itself (a word so long
    it cannot be broken cleanly — in practice only degenerate input). Returns `[text]`
    unchanged when it already fits, so the common case allocates nothing.

    The default `limit` is `CHUNK_MAX_BYTES`, not `MAX_BYTES`: what the model tokenises is
    the chunk plus its task prefix plus EOS.
    """
    if len(text.encode("utf-8")) <= limit:
        return [text]
    if not _byte_prefix(text, limit):
        raise ValueError(
            f"limit={limit} is smaller than the first character of the text "
            f"({len(text[0].encode('utf-8'))} bytes); no chunking is possible"
        )

    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest.encode("utf-8")) <= limit:
            chunks.append(rest)
            break
        head = _byte_prefix(rest, limit)
        cut = max((head.rfind(char) + 1 for char in _BOUNDARY_CHARS), default=0)
        if cut <= 0:
            space = head.rfind(" ")
            cut = space if space > 0 else len(head)
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    return [chunk for chunk in chunks if chunk]


def normalise_segments(iast: str) -> str:
    """Turn a raw decode into segments separated by exactly one space.

    The model marks a boundary with `SEGMENT_SEPARATOR` and emits a trailing one at the
    end of a sentence; it also emits ordinary spaces around material it did not analyse
    (numerals, transliterated English). Both become single spaces here, and the result is
    stripped, so "one whitespace unit per segment" holds for every downstream metric.
    """
    return " ".join(iast.replace(SEGMENT_SEPARATOR, " ").split())


class SandhiSplitter:
    """Reverses sandhi with ByT5-Sanskrit, caching every result.

    `split` takes Devanagari sentences and returns SLP1 with a space at every segment
    boundary the model found. Construction is cheap and offline: no import of `torch` or
    `transformers`, no network call, no weights — the checkpoint loads on the first
    sentence that misses the cache.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        cache: SplitCache | None = None,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        #: The candidates still to try, in order; `model_id` pins it to exactly one.
        self._candidates: tuple[str, ...] = (
            SPLITTER_CANDIDATES if model_id is None else (model_id,)
        )
        #: The id this splitter reports and will load first; updated if a later candidate
        #: is the one that actually loads.
        self.model_id: str = self._candidates[0]
        #: `None` until the model loads, at which point it becomes the resolved device.
        self.device: str | None = device
        #: True when generation on MPS failed and the model was moved to the CPU.
        self.device_fallback: bool = False
        self.cache = cache
        self.batch_size = batch_size
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._revision: str | None = None
        self._stats: dict[str, int | float] = {
            "n_calls": 0,
            "n_cache_hits": 0,
            "n_model": 0,
            "n_blank": 0,
            "n_duplicate": 0,
            "n_chunked": 0,
            "seconds_model": 0.0,
        }

    @property
    def stats(self) -> dict[str, int | float]:
        """Running counters, for a manifest or a benchmark. Live: mutated as work happens.

        * `n_calls` — input texts passed to `split`, summed over calls.
        * `n_cache_hits` — texts answered from the `SplitCache` (first occurrence only).
        * `n_model` — texts that required at least one generation.
        * `n_blank` — texts that were empty or whitespace and returned `""` untouched.
        * `n_duplicate` — texts identical to an earlier one *in the same call*, answered
          from that one's result rather than split twice.
        * `n_chunked` — texts whose IAST form exceeded `CHUNK_MAX_BYTES` and were cut into
          pieces (counted once per text, not once per piece).
        * `seconds_model` — wall time inside `_generate_iast`, excluding the one-off model
          load.

        The first five partition the input exactly:
        `n_calls == n_cache_hits + n_model + n_blank + n_duplicate`, which is what makes
        a manifest's "how much of this run was actually computed" line trustworthy.
        """
        return self._stats

    # -- identity -----------------------------------------------------------------

    @property
    def source_id(self) -> str:
        """`"<model id>@<commit sha>"` — what a manifest records as the splitter used.

        The sha is fetched from the Hub once and cached on the instance; any failure
        (offline, gated, renamed) degrades to `"unknown"` with a WARNING rather than
        aborting a multi-hour split over a metadata call.
        """
        if self._revision is None:
            try:
                self._revision = _model_revision(self.model_id)
            except Exception as error:  # noqa: BLE001 - any Hub failure is non-fatal here
                logger.warning(
                    "could not resolve the revision of %s (%s); recording 'unknown'",
                    self.model_id,
                    error,
                )
                self._revision = "unknown"
        return f"{self.model_id}@{self._revision}"

    # -- model loading ------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the checkpoint and its tokenizer on first use, trying each candidate."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        if self.device is None:
            self.device = pick_device()

        errors: dict[str, str] = {}
        for candidate in self._candidates:
            try:
                logger.info("loading sandhi splitter %s on %s", candidate, self.device)
                started = time.perf_counter()
                tokenizer: Any = AutoTokenizer.from_pretrained(candidate)
                model: Any = T5ForConditionalGeneration.from_pretrained(candidate)
            except Exception as error:  # noqa: BLE001 - try the next candidate
                errors[candidate] = f"{type(error).__name__}: {error}"
                logger.warning("candidate %s did not load (%s)", candidate, error)
                continue
            model.to(torch.device(self.device))
            model.eval()
            if candidate != self.model_id:
                # A substitution changes what `source_id` must report; drop any sha
                # already resolved for the candidate that did not load.
                self._revision = None
            self.model_id = candidate
            self._tokenizer = tokenizer
            self._model = model
            logger.info(
                "loaded %s in %.1f s on %s", candidate, time.perf_counter() - started, self.device
            )
            if candidate != self._candidates[0]:
                logger.warning(
                    "substituted %s for %s; record it in docs/decisions.md",
                    candidate,
                    self._candidates[0],
                )
            return
        raise RuntimeError(f"no sandhi splitter candidate could be loaded: {errors}")

    # -- generation ---------------------------------------------------------------

    def _generate_iast(self, batch: list[str]) -> list[str]:
        """Run the model over already-prefixed IAST strings, returning decoded IAST.

        The output length is capped per batch rather than at `MAX_BYTES`: segmentation
        output is longer than its input (see `OUTPUT_TOKENS_FACTOR`), so the input budget
        is the wrong cap for it and would truncate the longest sentences.

        This is the seam the tests replace wholesale, and the only step of `split` that
        is not exercised offline. On
        MPS a generation failure is retried once on the CPU — an MPS kernel gap should
        cost a run some throughput, not the run itself — and `device_fallback` records it.
        """
        import torch

        self._ensure_model()
        assert self._model is not None and self._tokenizer is not None

        try:
            return self._decode(batch, torch)
        except (RuntimeError, NotImplementedError) as error:
            if self.device != "mps":
                raise
            logger.warning("MPS generation failed (%s); falling back to the CPU", error)
            self._model.to(torch.device("cpu"))
            self.device = "cpu"
            self.device_fallback = True
            return self._decode(batch, torch)

    def _decode(self, batch: list[str], torch: Any) -> list[str]:
        assert self._model is not None and self._tokenizer is not None
        inputs = self._tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_BYTES,
        )
        inputs = {key: value.to(torch.device(str(self.device))) for key, value in inputs.items()}
        longest = max(len(text.encode("utf-8")) for text in batch)
        max_new_tokens = min(
            OUTPUT_TOKENS_FACTOR * longest + OUTPUT_TOKENS_SLACK, MAX_OUTPUT_TOKENS
        )
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=1
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [str(text) for text in decoded]

    # -- the public pipeline ------------------------------------------------------

    def split(self, texts: Sequence[str]) -> list[str]:
        """Split `texts` (Devanagari) and return SLP1 with spaces at segment boundaries.

        Every input is classified exactly once — blank, duplicate-within-this-call, cache
        hit, or model work — which is the partition `stats` documents. Blank inputs return
        `""` without a model call or a cache entry; cached inputs skip the model entirely,
        which is why a fully cached corpus never loads the weights.
        """
        self._stats["n_calls"] += len(texts)

        outputs: list[str | None] = [None] * len(texts)
        resolved: dict[str, str] = {}  # texts already answered from the cache
        pending: dict[str, list[int]] = {}  # cache-missing texts -> awaiting positions
        for index, text in enumerate(texts):
            if not text.strip():
                outputs[index] = ""
                self._stats["n_blank"] += 1
                continue
            if text in resolved:
                outputs[index] = resolved[text]
                self._stats["n_duplicate"] += 1
                continue
            if text in pending:
                pending[text].append(index)
                self._stats["n_duplicate"] += 1
                continue
            cached = None if self.cache is None else self.cache.get(text)
            if cached is not None:
                outputs[index] = cached
                resolved[text] = cached
                self._stats["n_cache_hits"] += 1
                continue
            pending[text] = [index]

        if pending:
            distinct = list(pending)
            for text, split_text in zip(distinct, self._split_uncached(distinct), strict=True):
                for index in pending[text]:
                    outputs[index] = split_text

        return [output if output is not None else "" for output in outputs]

    def _split_uncached(self, texts: list[str]) -> list[str]:
        """Transliterate, chunk, batch and generate for inputs the cache did not have.

        Cache writes happen **inside** the batch loop: as soon as the last chunk of a
        sentence comes back, that sentence is assembled, `put` and `flush`ed. A run killed
        after an hour therefore keeps its hour of work, minus at most the in-flight batch
        — the whole reason the cache is append-only jsonl rather than a dict dumped at the
        end (docs/decisions.md, "Splitter throughput rule": the split is a multi-hour
        background job).
        """
        self._stats["n_model"] += len(texts)

        # Flatten to chunks, remembering which chunk positions each sentence owns, so a
        # batch can span several sentences and a sentence can span several batches.
        owner_chunks: list[list[int]] = []
        prefixed: list[str] = []
        owner_of: list[int] = []
        for index, text in enumerate(texts):
            iast = from_slp1(to_slp1(text, "devanagari"), "iast")
            pieces = chunk_text(iast)
            if len(pieces) > 1:
                self._stats["n_chunked"] += 1
                logger.debug(
                    "chunked a %d-byte input into %d pieces",
                    len(iast.encode("utf-8")),
                    len(pieces),
                )
            owner_chunks.append(list(range(len(prefixed), len(prefixed) + len(pieces))))
            prefixed.extend(SEGMENTATION_PREFIX + piece for piece in pieces)
            owner_of.extend([index] * len(pieces))

        # Longest first: padding is per batch, so grouping similar lengths keeps a batch
        # from being padded out to the corpus's longest sentence. The order is an
        # efficiency detail and must not be observable, hence the scatter back below.
        order = sorted(range(len(prefixed)), key=lambda position: -len(prefixed[position]))
        decoded: list[str | None] = [None] * len(prefixed)
        outstanding = [len(chunks) for chunks in owner_chunks]
        outputs: list[str] = [""] * len(texts)

        for start in range(0, len(order), self.batch_size):
            positions = order[start : start + self.batch_size]
            batch = [prefixed[position] for position in positions]

            started = time.perf_counter()
            generated = self._generate_iast(batch)
            self._stats["seconds_model"] += time.perf_counter() - started

            finished: list[int] = []
            for position, output in zip(positions, generated, strict=True):
                decoded[position] = output
                owner = owner_of[position]
                outstanding[owner] -= 1
                if outstanding[owner] == 0:
                    finished.append(owner)

            for owner in finished:
                pieces = [decoded[position] or "" for position in owner_chunks[owner]]
                outputs[owner] = to_slp1(normalise_segments(" ".join(pieces)), "iast")
                if self.cache is not None:
                    self.cache.put(texts[owner], outputs[owner])
            if finished and self.cache is not None:
                self.cache.flush()

        return outputs

    def __repr__(self) -> str:
        return (
            f"SandhiSplitter(model_id={self.model_id!r}, device={self.device!r}, "
            f"batch_size={self.batch_size})"
        )
