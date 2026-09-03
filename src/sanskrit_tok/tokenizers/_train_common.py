"""Shared save/log helper for `train_bpe.py` and `train_unigram.py`.

Private (`_`-prefixed): this is not part of the tokenizers package's public surface, just
the handful of lines both trainers do identically after `tokenizer.train(...)` returns
(create the output directory, save `tokenizer.json`, log what happened), factored out so
the two trainer modules only differ in the model/trainer construction that is actually
specific to BPE vs. Unigram.
"""

import logging
from pathlib import Path

from tokenizers import Tokenizer

__all__ = ["save_trained_tokenizer"]

logger = logging.getLogger(__name__)


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
