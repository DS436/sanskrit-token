"""Train a BPE tokenizer on an SLP1 corpus (CLAUDE.md §6: T1 arms, `T1_bpe_raw_{32k,64k}`).

`pre_tokenizers.Metaspace` (and its matching `decoders.Metaspace`) means no merge ever
crosses a whitespace boundary, so this is the raw-BPE baseline the project's
morpheme-constrained arms (T5/T6) are compared against, not the proposed tokenizer
itself — CLAUDE.md §6 lists these arms as `T1_bpe_raw_{32k,64k}` on purpose.

`boundary_marker` is what turns it into that proposed tokenizer: see `train_bpe`'s
docstring and `sanskrit_tok.tokenizers.morph_bpe`, which is this function with the marker
switched on.
"""

import logging
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from sanskrit_tok.tokenizers._train_common import iter_corpus_lines, save_trained_tokenizer

__all__ = ["train_bpe"]

logger = logging.getLogger(__name__)


class BoundaryMarkerLeakedError(RuntimeError):
    """A trained tokenizer's vocabulary contains the training-time boundary marker.

    It must not: the marker exists only to stop the trainer counting a pair that straddles
    a gold morpheme boundary, and it is absent from every string the tokenizer will ever be
    asked to encode. A vocabulary entry containing it would be an id that can never fire,
    and — worse — evidence that the `Split` pre-tokenizer did not remove what it was
    supposed to, i.e. that the constraint was not actually applied.
    """


def train_bpe(
    corpus_path: Path,
    vocab_size: int,
    out_dir: Path,
    *,
    seed: int = 0,
    boundary_marker: str | None = None,
) -> Path:
    """Train a byte-pair-encoding tokenizer on `corpus_path` (one sentence per line).

    `models.BPE(unk_token="[UNK]")` with a `Metaspace` pre-tokenizer and matching decoder,
    trained with `trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"],
    show_progress=False)`.

    **`boundary_marker`** — when given (the T5/T6 arms; `morph_bpe.BOUNDARY_MARKER`), the
    corpus is expected to carry that character at every gold morpheme boundary and the
    pre-tokenizer becomes `Sequence([Metaspace(), Split(marker, behavior="removed")])`. The
    `Split` cuts each `Metaspace` word into its morphemes and deletes the marker itself, so
    the trainer never *sees* a pair spanning a boundary and no such pair can become a merge
    rule — MorphBPE's hard constraint, implemented without a custom trainer
    (docs/decisions.md, 2026-09-05, "MorphBPE-hard implemented as boundary-marker
    pre-tokenisation; inference unchanged"). The trainer's `initial_alphabet` is left empty,
    so the alphabet comes only from what survives the `Split` and the marker cannot enter
    it; that this held is *checked* after training, not assumed, and a violation raises
    `BoundaryMarkerLeakedError`. Real text carries no markers, so at inference the `Split`
    matches nothing and the arm behaves as an ordinary Metaspace BPE with a constrained
    merge table.

    Note what the constraint is and is not. It binds merge *learning*: no merge rule is
    learned from an occurrence that crosses a gold boundary. It cannot bind merge
    *application*, because a BPE merge rule is a global character pair — a pair that is
    frequent in some other word, where no boundary falls between its two halves, still
    becomes a rule and can then apply across a boundary elsewhere. Measuring exactly that
    residue is what `morph_bpe.assert_no_cross_boundary_merges` is for.

    **Line breaks never reach the pre-tokenizer.** The corpus is streamed through
    `_train_common.iter_corpus_lines` and trained with `train_from_iterator`, not
    `train([path])`, because the latter leaves each line's `\n` attached and `Metaspace`
    learns line-final words as `word\n` — dead vocabulary entries, unevenly distributed
    across arms (see that function's docstring). The saved tokenizer's pre-tokenizer is
    unchanged by this, which is why the fix is here and not in the pre-tokenizer.

    HF `tokenizers`' BPE trainer has no random step, so training is deterministic given the
    corpus and these settings; `seed` is accepted only so the caller can record it against
    the reproducibility protocol (CLAUDE.md §8) and is not otherwise used.

    Writes `out_dir/tokenizer.json` (parent directories created as needed, via
    `_train_common.save_trained_tokenizer`) and returns its path.
    """
    del seed  # deterministic trainer; accepted for the protocol only, see docstring
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    if boundary_marker is None:
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(),
                pre_tokenizers.Split(boundary_marker, behavior="removed"),
            ]
        )
    tokenizer.decoder = decoders.Metaspace()
    # `tokenizers`' generated .pyi stubs leave every Trainer subclass's `__init__`
    # unannotated, so this call is untyped even under strict mypy; harmless (the run-time
    # signature is documented in the stub's docstring and matches what is passed here).
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        special_tokens=["[UNK]"],
        show_progress=False,
        initial_alphabet=[],
    )

    tokenizer.train_from_iterator(iter_corpus_lines(corpus_path), trainer=trainer)

    if boundary_marker is not None:
        _assert_marker_absent_from_vocab(tokenizer, boundary_marker, corpus_path)

    return save_trained_tokenizer(tokenizer, out_dir, corpus_path, vocab_size, "train_bpe")


def _assert_marker_absent_from_vocab(
    tokenizer: Tokenizer, boundary_marker: str, corpus_path: Path
) -> None:
    """Raise `BoundaryMarkerLeakedError` if any vocabulary entry contains the marker.

    Cheap (one pass over at most 64k strings) and the only direct evidence that the `Split`
    pre-tokenizer really removed the marker rather than, say, silently treating it as an
    isolated token; without it a mis-specified pre-tokenizer would produce an arm that
    looks trained and is not constrained.
    """
    leaked = sorted(token for token in tokenizer.get_vocab() if boundary_marker in token)
    if leaked:
        raise BoundaryMarkerLeakedError(
            f"{len(leaked)} vocabulary entry/entries trained from {corpus_path} contain the "
            f"boundary marker {boundary_marker!r} (first: {leaked[:3]!r}); the Split "
            "pre-tokenizer did not remove it, so the merge constraint was not applied"
        )

