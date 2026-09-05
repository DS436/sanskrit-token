"""Turning a line-per-sentence SLP1 corpus into the token stream a GPT trains on.

One corpus file, one tokenizer arm, one `.bin` of token ids plus a `.meta.json` recording
what produced it. Everything here is streaming: `track2_raw.txt` is 3.67 GB and 45.2M lines
(docs/decisions.md, 2026-09-05, "M1 built"), so no function in this module holds the corpus,
or its token ids, in memory at once — encoding appends to an open file in chunks and
training reads back through a `numpy` memmap.

Three decisions are baked in, and each of them is a number in `results.json` if got wrong:

**One EOS per line, with a dedicated id.** The corpora are one sentence per line and the
newline is *not* in the text (SLP1 has no use for it), so without a separator the model
would be trained on sentences run together with no boundary. The separator is a new id
equal to the arm's vocabulary size, so the logical vocabulary is `vocab_size + 1`; reusing
an existing token (`[UNK]`, or a byte value) would make the boundary indistinguishable from
real text and would differ per arm. `T7_byt5` has vocabulary 256 and therefore EOS 256.

**The byte and character counts are of the lines, not of the file.** `n_chars` sums
`len(line)` with the newline excluded and `n_bytes` sums `len(line.encode("utf-8"))` of the
same lines; blank lines are skipped entirely. `n_chars` is the denominator of every BPC
(`bpc.py`), and `n_bytes` is what "equal training bytes" means when arms are compared
(docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and comparison protocol"),
so both have to mean the same thing for every arm on the same file — which they do, since
neither depends on the tokenizer at all.

**The dtype follows the vocabulary.** `uint16` while the ids fit, `uint32` beyond; a 64k
arm plus its EOS fits comfortably, a hypothetical 200k one would not, and storing the
3.67 GB corpus as `int64` would cost four times the disk for no information.
"""

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from sanskrit_tok.tokenizers.registry import LoadedTokenizer

__all__ = [
    "CHUNK_LINES",
    "EncodedCorpus",
    "TokenisedText",
    "dtype_for",
    "encode_corpus",
    "ensure_encoded_corpus",
    "eos_id_for",
    "iter_batches",
    "load_encoded_corpus",
    "meta_path_for",
    "open_tokens",
    "tokenise_text",
]

logger = logging.getLogger(__name__)

#: How many lines `encode_corpus` encodes before flushing them to disk. Large enough that
#: the write syscall is not the bottleneck on 45M lines, small enough that the buffered ids
#: are a few megabytes rather than the whole corpus.
CHUNK_LINES = 50_000

#: The largest id `uint16` can hold. `dtype_for` keeps one id of headroom above the logical
#: vocabulary so that a caller adding one more special token later widens the dtype rather
#: than silently wrapping around.
_UINT16_MAX = 65_535


@dataclass(frozen=True)
class EncodedCorpus:
    """A `.bin` of token ids and everything needed to know what it is.

    `n_tokens` counts the EOS tokens; `n_chars`, `n_bytes` and `n_lines` describe the
    *source text* and are identical for every arm encoding the same file, which is what
    makes an equal-bytes comparison across arms meaningful.

    `eos_id` is `None` for a stream with no separators, in which case `logical_vocab` is
    just the arm's vocabulary. `source_sha256` is of the corpus file's bytes, so a cached
    `.bin` can be told apart from one built before the corpus was rebuilt.
    """

    bin_path: Path
    meta_path: Path
    arm: str
    source_path: Path
    source_sha256: str
    dtype: str
    n_tokens: int
    n_chars: int
    n_bytes: int
    n_lines: int
    eos_id: int | None
    logical_vocab: int

    @property
    def bytes_per_token(self) -> float:
        """Source UTF-8 bytes per token: the conversion an equal-bytes budget needs.

        Training length is measured in tokens (that is what a step consumes) but compared
        in bytes (that is what is identical across arms), and this ratio is the bridge.
        """
        return self.n_bytes / self.n_tokens

    @property
    def chars_per_token(self) -> float:
        """Source characters per token — the arm's compression on this corpus."""
        return self.n_chars / self.n_tokens

    def to_meta(self) -> dict[str, object]:
        """The `.meta.json` payload: every field, paths as strings."""
        meta = asdict(self)
        for key in ("bin_path", "meta_path", "source_path"):
            meta[key] = str(meta[key])
        return meta


@dataclass(frozen=True)
class TokenisedText:
    """A held-out text tokenised in memory for evaluation, with its own character counts.

    Evaluation texts are small (the largest is 1.3 MB) and are scored in one pass, so unlike
    a training corpus they are held as a list rather than memmapped. `n_chars` excludes
    newlines and `ids` includes one EOS per line, exactly as `encode_corpus` does, so a BPC
    is computed against the same convention the model was trained under.
    """

    ids: list[int]
    n_chars: int
    n_bytes: int
    n_lines: int


def eos_id_for(arm: LoadedTokenizer) -> int:
    """The end-of-line id for `arm`: one past its largest real id.

    `arm.vocab_size` is the full id space including added special tokens, so `vocab_size`
    itself is free by construction and the logical vocabulary becomes `vocab_size + 1`
    (256 + 1 for `T7_byt5`, 64000 + 1 for a 64k arm).
    """
    return arm.vocab_size


def dtype_for(logical_vocab: int) -> str:
    """`"uint16"` if the ids fit with one spare, `"uint32"` otherwise."""
    if logical_vocab <= 0:
        raise ValueError(f"logical_vocab must be positive, got {logical_vocab}")
    return "uint16" if logical_vocab + 1 <= _UINT16_MAX else "uint32"


def meta_path_for(bin_path: Path) -> Path:
    """`foo.bin` -> `foo.meta.json`."""
    return bin_path.with_suffix(".meta.json")


def _sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_lines(path: Path) -> Iterator[str]:
    """Non-blank lines of `path`, newline stripped.

    A line that is empty or only whitespace carries no text and would contribute an EOS
    with nothing before it, so it is skipped and not counted in `n_lines`.
    """
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n").rstrip("\r")
            if line.strip():
                yield line


def encode_corpus(
    arm: LoadedTokenizer,
    corpus_path: Path,
    out_path: Path,
    *,
    eos_id: int | None,
) -> EncodedCorpus:
    """Encode `corpus_path` with `arm` into `out_path` (a `.bin`) plus its `.meta.json`.

    One line at a time, `arm.encode(line)` followed by `eos_id` when one is given; ids are
    buffered `CHUNK_LINES` lines at a time and appended as raw little-endian `uint16` or
    `uint32` (`dtype_for`). Nothing is validated about the ids beyond the dtype: an arm that
    emitted an id outside its declared vocabulary would be a bug in the arm, and the
    assertion below catches it rather than writing a corpus a model cannot embed.

    Overwrites `out_path` unconditionally; `ensure_encoded_corpus` is the caching wrapper.
    """
    logical_vocab = arm.vocab_size if eos_id is None else max(arm.vocab_size, eos_id + 1)
    dtype_name = dtype_for(logical_vocab)
    dtype = np.dtype(dtype_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_tokens = n_chars = n_bytes = n_lines = 0
    buffer: list[int] = []
    with out_path.open("wb") as sink:

        def flush() -> None:
            if buffer:
                np.asarray(buffer, dtype=dtype).tofile(sink)
                buffer.clear()

        for line in _iter_lines(corpus_path):
            ids = arm.encode(line)
            if ids and (max(ids) >= arm.vocab_size or min(ids) < 0):
                raise ValueError(
                    f"{arm.name} emitted an id outside [0, {arm.vocab_size}) on a line of "
                    f"{corpus_path}; the encoded corpus would not fit the model's embedding"
                )
            buffer.extend(ids)
            n_tokens += len(ids)
            if eos_id is not None:
                buffer.append(eos_id)
                n_tokens += 1
            n_chars += len(line)
            n_bytes += len(line.encode("utf-8"))
            n_lines += 1
            if n_lines % CHUNK_LINES == 0:
                flush()
                logger.info("%s: encoded %d lines of %s", arm.name, n_lines, corpus_path.name)
        flush()

    if n_tokens == 0:
        raise ValueError(f"{corpus_path} produced no tokens; is it empty?")

    encoded = EncodedCorpus(
        bin_path=out_path,
        meta_path=meta_path_for(out_path),
        arm=arm.name,
        source_path=corpus_path,
        source_sha256=_sha256_file(corpus_path),
        dtype=dtype_name,
        n_tokens=n_tokens,
        n_chars=n_chars,
        n_bytes=n_bytes,
        n_lines=n_lines,
        eos_id=eos_id,
        logical_vocab=logical_vocab,
    )
    encoded.meta_path.write_text(
        json.dumps(encoded.to_meta(), indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "%s: %d tokens from %d lines (%d chars, %d bytes) -> %s",
        arm.name,
        n_tokens,
        n_lines,
        n_chars,
        n_bytes,
        out_path,
    )
    return encoded


def load_encoded_corpus(bin_path: Path) -> EncodedCorpus:
    """Read the `EncodedCorpus` recorded beside `bin_path`."""
    meta = json.loads(meta_path_for(bin_path).read_text(encoding="utf-8"))
    return EncodedCorpus(
        bin_path=Path(meta["bin_path"]),
        meta_path=Path(meta["meta_path"]),
        arm=meta["arm"],
        source_path=Path(meta["source_path"]),
        source_sha256=meta["source_sha256"],
        dtype=meta["dtype"],
        n_tokens=int(meta["n_tokens"]),
        n_chars=int(meta["n_chars"]),
        n_bytes=int(meta["n_bytes"]),
        n_lines=int(meta["n_lines"]),
        eos_id=None if meta["eos_id"] is None else int(meta["eos_id"]),
        logical_vocab=int(meta["logical_vocab"]),
    )


def ensure_encoded_corpus(
    arm: LoadedTokenizer,
    corpus_path: Path,
    out_path: Path,
    *,
    eos_id: int | None,
) -> EncodedCorpus:
    """`encode_corpus`, skipped when `out_path` already holds exactly this encoding.

    "Exactly this" is the arm name, the EOS id and the source file's sha256 — so a rebuilt
    corpus, a different arm or a change of separator all force a re-encode, while a sweep
    that runs three seeds of the same arm encodes once. A `.meta.json` that cannot be read
    is treated as absent rather than fatal.
    """
    meta_path = meta_path_for(out_path)
    if out_path.exists() and meta_path.exists():
        try:
            cached = load_encoded_corpus(out_path)
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning("%s: unreadable meta, re-encoding", meta_path)
        else:
            if (
                cached.arm == arm.name
                and cached.eos_id == eos_id
                and cached.source_sha256 == _sha256_file(corpus_path)
            ):
                logger.info("%s: reusing encoded corpus %s", arm.name, out_path)
                return cached
            logger.info("%s: encoded corpus %s is stale, re-encoding", arm.name, out_path)
    return encode_corpus(arm, corpus_path, out_path, eos_id=eos_id)


def open_tokens(corpus: EncodedCorpus) -> NDArray[np.integer]:
    """The token ids of `corpus` as a read-only memmap (nothing is loaded into RAM)."""
    return np.memmap(corpus.bin_path, dtype=np.dtype(corpus.dtype), mode="r")


def tokenise_text(
    arm: LoadedTokenizer,
    text_path: Path,
    *,
    eos_id: int | None,
    max_chars: int | None,
) -> TokenisedText:
    """Tokenise a held-out text for evaluation, one EOS per line, in memory.

    `max_chars` truncates at a **whole line**: lines are taken until the running character
    count reaches the cap, so the last line is included in full and `n_chars` may exceed the
    cap by less than one line. Truncating mid-line would charge the model for predicting the
    continuation of a sentence it was then not shown, and would make the cap's effect depend
    on the arm's tokenization.
    """
    ids: list[int] = []
    n_chars = n_bytes = n_lines = 0
    for line in _iter_lines(text_path):
        ids.extend(arm.encode(line))
        if eos_id is not None:
            ids.append(eos_id)
        n_chars += len(line)
        n_bytes += len(line.encode("utf-8"))
        n_lines += 1
        if max_chars is not None and n_chars >= max_chars:
            break
    if not ids:
        raise ValueError(f"{text_path} produced no tokens; is it empty?")
    return TokenisedText(ids=ids, n_chars=n_chars, n_bytes=n_bytes, n_lines=n_lines)


def iter_batches(
    bin_path: Path,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
    *,
    device: str | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Endlessly yield `(x, y)` batches of random `block_size` windows, nanoGPT style.

    `x` is `batch_size × block_size` token ids and `y` the same window shifted one token
    left, so every position predicts the next token. Windows start at uniformly random
    offsets and may overlap: this is sampling with replacement from the corpus, not an
    epoch, which is why "epochs" in the training curve is a ratio (`tokens_seen / n_tokens`)
    rather than a count of passes.

    Determinism is entirely `rng`'s: two iterators built from `numpy.random.default_rng(k)`
    with the same `k` yield identical batches, which is what makes a seed reproduce a run.

    Raises `ValueError` if the corpus is shorter than `block_size + 1` tokens — there is no
    window to draw, and the alternative is a silently empty or wrapped batch.
    """
    corpus = load_encoded_corpus(bin_path)
    tokens = open_tokens(corpus)
    n_tokens = len(tokens)
    if n_tokens < block_size + 1:
        raise ValueError(
            f"{bin_path} holds {n_tokens} tokens, fewer than the {block_size + 1} a "
            f"block_size of {block_size} needs; use a smaller block or a larger corpus"
        )
    high = n_tokens - block_size
    while True:
        starts = rng.integers(0, high, size=batch_size)
        x = torch.from_numpy(
            np.stack([tokens[start : start + block_size] for start in starts]).astype(np.int64)
        )
        y = torch.from_numpy(
            np.stack(
                [tokens[start + 1 : start + 1 + block_size] for start in starts]
            ).astype(np.int64)
        )
        if device is not None and device != "cpu":
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        yield x, y

