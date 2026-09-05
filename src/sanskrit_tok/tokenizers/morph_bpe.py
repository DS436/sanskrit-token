"""Merge-constrained BPE: the T5/T6 arms (CLAUDE.md §6), and how to audit the constraint.

MorphBPE's hard constraint is "no merge may span a morpheme boundary". HF `tokenizers`
exposes no hook into merge selection, so this project applies the constraint one layer up,
in pre-tokenisation: the *training* corpus carries `BOUNDARY_MARKER` (U+001F) at every gold
boundary, and the pre-tokenizer is `Sequence([Metaspace(), Split(marker, "removed")])`.
Each `Metaspace` word is therefore cut into its gold morphemes before the trainer counts
anything, and the marker is deleted rather than tokenised, so no pair straddling a boundary
is ever counted and no such merge rule can be learned (docs/decisions.md, 2026-09-05,
"MorphBPE-hard implemented as boundary-marker pre-tokenisation; inference unchanged").
Evaluation text has no markers, so the `Split` matches nothing and the arm is a standard
Metaspace BPE with a constrained merge table — which is what MorphBPE reports.

**The constraint binds learning, not application, and `assert_no_cross_boundary_merges`
measures the gap.** A BPE merge rule is a global character pair. `D`+`A` may be frequent in
words where no boundary falls between the two, become a rule on that evidence, and then
apply inside `budDAya` whose gold boundary sits at index 4 — producing a token that spans a
boundary the trainer was never allowed to merge across. So the number this module reports
for a T5/T6 arm is a *residue*, expected to be well below the matching unconstrained arm's
and not expected to be zero on a real corpus; it is zero only when the corpus is small
enough that no other context supplies the crossing pair. Both numbers go into the arms'
`results.json` so the claim "the constraint did something" is a measurement rather than an
assertion about the code.

Where the gold boundaries come from is `sanskrit_tok.data.boundaries`, which is also where
`BOUNDARY_MARKER` is defined; it is imported here rather than redefined so the ingestion and
the trainer can never disagree about which character marks a boundary.
"""

import logging
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as RawTokenizer

from sanskrit_tok.data.boundaries import BOUNDARY_MARKER
from sanskrit_tok.tokenizers.train_bpe import train_bpe

__all__ = [
    "BOUNDARY_MARKER",
    "assert_no_cross_boundary_merges",
    "strip_markers",
    "train_morph_bpe",
]

logger = logging.getLogger(__name__)

#: How many offending tokens `assert_no_cross_boundary_merges` keeps as examples.
_MAX_EXAMPLES = 10


def train_morph_bpe(corpus_path: Path, vocab_size: int, out_dir: Path, *, seed: int = 0) -> Path:
    """Train a morpheme-constrained BPE on a **marked** corpus (the T5/T6 arms).

    Exactly `train_bpe(..., boundary_marker=BOUNDARY_MARKER)`: same model, same decoder,
    same trainer settings, same output layout, so a T5 arm and its matched T1 arm differ in
    the pre-tokenizer and nothing else. `corpus_path` must be a corpus written with markers
    (`GoldSentence.t5_marked` / `t6_marked`); given an unmarked corpus this trains an
    ordinary BPE and says nothing about it, which is why the arm configs name the marked
    corpus explicitly and `assert_no_cross_boundary_merges` is run afterwards.

    Same signature as `train_bpe`/`train_unigram` so `training.ALGO_TRAINERS` can hold all
    three interchangeably. Returns the written `tokenizer.json` path.
    """
    return train_bpe(
        corpus_path, vocab_size, out_dir, seed=seed, boundary_marker=BOUNDARY_MARKER
    )


def strip_markers(marked: str, marker: str = BOUNDARY_MARKER) -> tuple[str, list[int]]:
    """`(plain_text, boundary_offsets)` for one marked line.

    The offsets are character indices *into the returned plain text*, at the position each
    marker occupied — i.e. the index of the first character after the boundary. Consecutive
    markers, and a marker at either end, all collapse onto a position that no token span can
    strictly contain, so they cannot manufacture a violation.
    """
    out: list[str] = []
    offsets: list[int] = []
    for character in marked:
        if character == marker:
            offsets.append(len(out))
        else:
            out.append(character)
    return "".join(out), offsets


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def assert_no_cross_boundary_merges(
    tokenizer_json: Path,
    marked_corpus: Path,
    sample: int = 2000,
    *,
    marker: str = BOUNDARY_MARKER,
) -> dict[str, Any]:
    """Count tokens whose span strictly contains a gold boundary, on a strided sample.

    For each sampled line of `marked_corpus`: strip the markers, recording where they were
    (`strip_markers`); encode the resulting plain text with the tokenizer at
    `tokenizer_json`; and count the tokens whose character span `(start, end)` satisfies
    `start < position < end` for some boundary position. Strict containment is the whole
    definition — a token that *ends* at a boundary, or begins at one, respects it.

    The sample is **strided, not the first `sample` lines**: a DCS corpus is ordered by text,
    so a prefix would be one work. The file is counted once (a cheap scan), the stride is
    `max(1, n_lines // sample)`, and every `stride`-th line is taken, which is deterministic
    and spreads the sample over the whole corpus.

    Despite the name this *returns* the counts rather than raising: what counts as
    acceptable differs between the toy corpus in the tests (where 0 is the correct answer
    and the test asserts it) and a real arm (where the module docstring explains why a small
    residue is expected). The caller decides; every T5/T6 arm records the dict in its
    `results.json`, together with the same measurement on the matching unconstrained arm as
    a control.

    Returns `n_lines`, `sample`, `stride`, `n_sentences` (lines actually scored),
    `n_boundaries`, `n_tokens`, `n_violations`, `violation_rate` (violations per token),
    `boundary_marker`, and up to ten `examples` of an offending token in context.
    """
    tokenizer = RawTokenizer.from_file(str(tokenizer_json))
    n_lines = _count_lines(marked_corpus)
    stride = max(1, n_lines // sample) if sample > 0 else 1

    n_sentences = n_boundaries = n_tokens = n_violations = 0
    examples: list[dict[str, Any]] = []

    with marked_corpus.open(encoding="utf-8") as handle:
        kept = 0
        for index, raw_line in enumerate(line for line in handle if line.strip()):
            if kept >= sample:
                break
            if index % stride:
                continue
            kept += 1
            text, offsets = strip_markers(raw_line.rstrip("\n"), marker)
            positions = set(offsets)
            n_sentences += 1
            n_boundaries += len(positions)
            for start, end in tokenizer.encode(text).offsets:
                n_tokens += 1
                crossed = sorted(p for p in positions if start < p < end)
                if not crossed:
                    continue
                n_violations += 1
                if len(examples) < _MAX_EXAMPLES:
                    examples.append(
                        {
                            "token": text[start:end],
                            "text": text,
                            "span": [start, end],
                            "boundaries": crossed,
                        }
                    )

    report: dict[str, Any] = {
        "tokenizer_json": str(tokenizer_json),
        "marked_corpus": str(marked_corpus),
        "boundary_marker": marker,
        "n_lines": n_lines,
        "sample": sample,
        "stride": stride,
        "n_sentences": n_sentences,
        "n_boundaries": n_boundaries,
        "n_tokens": n_tokens,
        "n_violations": n_violations,
        "violation_rate": (n_violations / n_tokens) if n_tokens else 0.0,
        "examples": examples,
    }
    logger.info(
        "cross-boundary check %s on %s: %d/%d token(s) span a gold boundary (%.4f) over "
        "%d sentence(s), %d boundary/boundaries",
        tokenizer_json.parent.name,
        marked_corpus.name,
        n_violations,
        n_tokens,
        report["violation_rate"],
        n_sentences,
        n_boundaries,
    )
    return report
