"""Train a BPE tokenizer on an SLP1 corpus (CLAUDE.md §6: T1 arms, `T1_bpe_raw_{32k,64k}`).

`pre_tokenizers.Metaspace` (and its matching `decoders.Metaspace`) means no merge ever
crosses a whitespace boundary, so this is the raw-BPE baseline the project's
morpheme-constrained arms (T5/T6) are compared against, not the proposed tokenizer
itself — CLAUDE.md §6 lists these arms as `T1_bpe_raw_{32k,64k}` on purpose.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from sanskrit_tok.tokenizers._train_common import save_trained_tokenizer

__all__ = ["train_bpe"]


def train_bpe(corpus_path: Path, vocab_size: int, out_dir: Path, *, seed: int = 0) -> Path:
    """Train a byte-pair-encoding tokenizer on `corpus_path` (one sentence per line).

    `models.BPE(unk_token="[UNK]")` with a `Metaspace` pre-tokenizer and matching decoder,
    trained with `trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"],
    show_progress=False)`.

    HF `tokenizers`' BPE trainer has no random step, so training is deterministic given
    the corpus and these settings; `seed` is accepted only so the caller can record it
    against the reproducibility protocol (CLAUDE.md §8) and is not otherwise used.

    Writes `out_dir/tokenizer.json` (parent directories created as needed, via
    `_train_common.save_trained_tokenizer`) and returns its path.
    """
    del seed  # deterministic trainer; accepted for the protocol only, see docstring
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    # `tokenizers`' generated .pyi stubs leave every Trainer subclass's `__init__`
    # unannotated, so this call is untyped even under strict mypy; harmless (the run-time
    # signature is documented in the stub's docstring and matches what is passed here).
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size, special_tokens=["[UNK]"], show_progress=False
    )

    tokenizer.train([str(corpus_path)], trainer=trainer)

    return save_trained_tokenizer(tokenizer, out_dir, corpus_path, vocab_size, "train_bpe")
