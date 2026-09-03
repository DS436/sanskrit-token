"""Train a Unigram-LM tokenizer on an SLP1 corpus (CLAUDE.md §6: T2 arms).

Same contract as `train_bpe.train_bpe`, with the Unigram model instead of BPE; see that
module's docstring for the `Metaspace` boundary rationale, which applies unchanged here.
"""

import logging
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

__all__ = ["train_unigram"]

logger = logging.getLogger(__name__)


def train_unigram(corpus_path: Path, vocab_size: int, out_dir: Path, *, seed: int = 0) -> Path:
    """Train a Unigram-LM tokenizer on `corpus_path` (one sentence per line).

    `models.Unigram()` with a `Metaspace` pre-tokenizer and matching decoder, trained with
    `trainers.UnigramTrainer(vocab_size=vocab_size, unk_token="[UNK]",
    special_tokens=["[UNK]"], show_progress=False)`.

    The EM-based Unigram trainer can settle on fewer pieces than `vocab_size` when the
    corpus does not support the full request; `tokenizer.get_vocab_size()` after training
    is the number to report as the arm's actual vocabulary (record any shortfall in
    `docs/decisions.md`, CLAUDE.md §11), never the requested `vocab_size` itself.

    `seed` is accepted only so the caller can record it against the reproducibility
    protocol (CLAUDE.md §8); HF `tokenizers`' `UnigramTrainer` exposes no random seed of
    its own, so this has no effect on the result.

    Writes `out_dir/tokenizer.json` (parent directories created as needed) and returns
    its path.
    """
    del seed  # UnigramTrainer has no exposed seed; accepted for the protocol only
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    # `tokenizers`' generated .pyi stubs leave every Trainer subclass's `__init__`
    # unannotated, so this call is untyped even under strict mypy; harmless (the run-time
    # signature is documented in the stub's docstring and matches what is passed here).
    trainer = trainers.UnigramTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        unk_token="[UNK]",
        special_tokens=["[UNK]"],
        show_progress=False,
    )

    tokenizer.train([str(corpus_path)], trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    logger.info(
        "train_unigram: %s -> %s (requested vocab_size=%d, actual=%d)",
        corpus_path,
        out_path,
        vocab_size,
        tokenizer.get_vocab_size(),
    )
    return out_path
