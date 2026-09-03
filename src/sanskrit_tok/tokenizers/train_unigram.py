"""Train a Unigram-LM tokenizer on an SLP1 corpus (CLAUDE.md §6: T2 arms).

Same contract as `train_bpe.train_bpe`, with the Unigram model instead of BPE; see that
module's docstring for the `Metaspace` boundary rationale, which applies unchanged here.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from sanskrit_tok.tokenizers._train_common import save_trained_tokenizer

__all__ = ["train_unigram"]


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
    protocol (CLAUDE.md §8); HF `tokenizers`' `UnigramTrainer` exposes no random seed
    parameter to set, so `seed` itself has no effect on the result.

    That is *not* the same as this function being bit-for-bit deterministic, unlike
    `train_bpe.train_bpe`. Verified empirically (2026-09-03 decision log,
    "`UnigramTrainer` is not bit-for-bit deterministic"): training twice in the same
    process on byte-identical input produces two `tokenizer.json` files that differ —
    the vocabulary *set* and the requested vs. actual size are stable across runs, but
    per-piece EM scores land a few floating-point ULPs apart (almost certainly
    multi-threaded summation order inside the Rust implementation), which changes
    tie-breaking in token-id assignment for near-equal-score pieces. Treat the written
    file as reproducible in *content* (what it tokenizes into, up to those ties), not in
    bytes; re-running this on the same corpus is not guaranteed to reproduce a
    previously-written `tokenizer.json` exactly.

    Writes `out_dir/tokenizer.json` (parent directories created as needed, via
    `_train_common.save_trained_tokenizer`) and returns its path.
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

    return save_trained_tokenizer(tokenizer, out_dir, corpus_path, vocab_size, "train_unigram")
