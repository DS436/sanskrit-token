"""Shared save/log helper for `train_bpe.py` and `train_unigram.py`.

Private (`_`-prefixed): this is not part of the tokenizers package's public surface, just
the handful of lines both trainers do identically after `tokenizer.train(...)` returns
(create the output directory, save `tokenizer.json`, log what happened), factored out so
the two trainer modules only differ in the model/trainer construction that is actually
specific to BPE vs. Unigram.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer

__all__ = ["iter_corpus_lines", "save_trained_tokenizer"]

logger = logging.getLogger(__name__)


def iter_corpus_lines(corpus_path: Path) -> Iterator[str]:
    """Every non-blank line of `corpus_path`, with its line break stripped.

    This exists because `Tokenizer.train([path])` hands each line to the pre-tokenizer with
    its trailing `\n` still attached, and `Metaspace` treats that newline as an ordinary
    character: a line-final word is learned as `word\n`, so 5–15% of a BPE vocabulary ends
    up as entries carrying a newline that no inference-time string can ever contain. They
    are dead ids, and they are not distributed evenly across arms — a constrained arm's
    marker `Split` shatters the line-final word and leaves it far fewer of them — so
    "matched vocabulary size" (CLAUDE.md §2.5) was off by about 7% of live entries in the
    unconstrained controls' disfavour (docs/decisions.md, 2026-09-05, "Trainers strip
    newlines; every trained arm is retrained and Experiments 02–04 re-run").

    Both trainers therefore call `train_from_iterator` over this rather than
    `train([path])`. That is the fix which leaves the *saved* tokenizer byte-identical in
    everything but its learned vocabulary: adding a `Split("\n")` to the pre-tokenizer
    would also change the pre-tokenizer the file records, and so change how the arm behaves
    at inference for no reason.

    A line that is empty once its break is removed is skipped: it contributes nothing, and
    an empty string handed to `train_from_iterator` is one more thing for the trainer to
    count as a sequence.
    """
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\r\n")
            if stripped:
                yield stripped


def save_trained_tokenizer(
    tokenizer: Tokenizer,
    out_dir: Path,
    corpus_path: Path,
    vocab_size: int,
    label: str,
) -> Path:
    """Write a trained `tokenizer` to `out_dir/tokenizer.json` and log the outcome.

    `label` is the caller's own name for its log lines (`"train_bpe"` / `"train_unigram"`).
    `vocab_size` is the *requested* size, logged alongside the trained tokenizer's actual
    `get_vocab_size()` so a shortfall (CLAUDE.md §11: e.g. Unigram settling on fewer
    pieces than requested) is visible in the log even before a caller inspects the
    written file.

    Creates `out_dir` (and parent directories) as needed. Returns the written path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    logger.info(
        "%s: %s -> %s (requested vocab_size=%d, actual=%d)",
        label,
        corpus_path,
        out_path,
        vocab_size,
        tokenizer.get_vocab_size(),
    )
    return out_path
