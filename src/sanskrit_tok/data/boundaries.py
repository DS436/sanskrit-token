"""Gold morpheme boundaries for a DCS sentence, located in the sandhied SLP1 surface.

DCS gives, for each surface word, the sequence of *unsandhied* segments it was formed from
(the `Unsandhied=` values of a multiword range) plus each segment's lemma. It does not give
character positions: `pratītyajānāṃ` is annotated as `pratītya` + `jānām`, and where in the
sandhied string the join falls is left implicit. This module recovers those positions
(docs/decisions.md, 2026-09-05, "Gold boundaries: segment boundaries from DCS, stem/ending
boundary derived from the lemma and labelled heuristic"):

* **Primary — segment boundaries.** `align_segments` aligns the concatenated segments to
  the sandhied surface with `difflib.SequenceMatcher` and reports the surface index at
  which each segment after the first begins. Sandhi rewrites characters at the join
  (`...jānāṃ` for `...jānām`), so an exact search would fail on a large fraction of words;
  a similarity alignment does not, and a word whose alignment is implausible (ratio < 0.6)
  or whose offsets do not come out strictly increasing inside the word is reported as
  `None` rather than guessed at, and counted.
* **Secondary — the stem/ending boundary.** `stem_boundary` is the longest common prefix
  of a segment and its lemma, kept only when the stem is at least two characters and the
  ending is non-empty. This is a **heuristic**, not DCS annotation, and every table that
  uses it must say so.

`GoldSentence` packages both, in the two text variants Experiment 04 trains on: the
sandhied text (`t5_marked`) and the gold "oracle" split (`t6_marked`), each carrying
`BOUNDARY_MARKER` (U+001F) at the boundaries a constrained BPE may not merge across
(docs/decisions.md, "MorphBPE-hard implemented as boundary-marker pre-tokenisation").
Removing every marker from either marked string returns the corresponding plain string
exactly, which is what makes the marker safe as a training-time-only device.

**The `# text` line is the sentence.** A DCS sentence's `# text` is the authority on what
the sandhied text *is*; its token block occasionally reconstructs fewer surface words than
`# text` holds (11,339 sentences, 1.5%). Words are therefore taken from `# text`, matched to
the token block by `difflib` over the two word sequences, and a `# text` word with no
matching token word carries no segments and an alignment of `None`. That keeps the
training text faithful to the corpus and keeps `segment_offsets[i]` about `text_slp1`'s
i-th word by construction.

Pure: transliteration and string arithmetic only, no I/O.
"""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from sanskrit_tok.data.dcs import DcsSentence, sentence_is_human_verified
from sanskrit_tok.encoding import to_slp1

__all__ = [
    "BOUNDARY_MARKER",
    "GoldSentence",
    "MIN_ALIGN_RATIO",
    "MIN_STEM_LENGTH",
    "align_segments",
    "build_gold_sentence",
    "mark",
    "stem_boundary",
    "text_words_slp1",
]

#: U+001F UNIT SEPARATOR: the private boundary marker inserted into *training* text so a
#: BPE pre-tokenizer can split on it and no learned merge can span a gold boundary. It is
#: never present at inference time and never occurs in SLP1, IAST or Devanagari text.
#: `sanskrit_tok.tokenizers.morph_bpe` imports this rather than redefining it.
BOUNDARY_MARKER = "\x1f"

#: `difflib` similarity below which a word's segment alignment is rejected as implausible.
MIN_ALIGN_RATIO = 0.6

#: Shortest stem `stem_boundary` will report. One character is almost always a coincidence
#: (every lemma beginning with the same letter as its inflected form would qualify).
MIN_STEM_LENGTH = 2

#: Lemmas DCS writes for "no lemma"; `_` is the CoNLL-U empty cell.
_EMPTY_LEMMAS = frozenset({"", "_"})

_WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=1 << 20)
def _slp1(iast: str) -> str:
    """`to_slp1(iast, "iast")`, memoised.

    DCS is 750,660 sentences of 4.29M word tokens over a far smaller vocabulary, and each
    is transliterated several times (surface, unsandhied, lemma). The cache is what makes
    the whole ingestion a 92-second job rather than a transliteration-bound one.
    """
    return to_slp1(iast, "iast")


# ------------------------------------------------------------------------------ alignment


def _project(blocks: Sequence[tuple[int, int, int]], index: int, fallback: int) -> int:
    """Map an index in the concatenated segments to an index in the surface.

    `blocks` are `SequenceMatcher.get_matching_blocks()` over `(surface, concatenation)`,
    which are ordered and non-overlapping in both sequences. An index inside a matching
    block maps through it exactly; an index in a gap between blocks (a region sandhi
    rewrote) maps to the start of the next matched surface region, which keeps the whole
    map monotone non-decreasing — the property the strict-increase check below relies on.
    """
    for surface_start, concat_start, size in blocks:
        if concat_start <= index < concat_start + size:
            return surface_start + (index - concat_start)
        if concat_start > index:
            return surface_start
    return fallback


def align_segments(surface_slp1: str, segments_slp1: Sequence[str]) -> list[int] | None:
    """Surface offsets at which each segment after the first begins, or `None`.

    Returns a list of `len(segments) - 1` character indices into `surface_slp1`, excluding
    the trivial boundaries 0 and `len(surface_slp1)`; a single segment therefore yields
    `[]` (the word has no internal boundary), which is a *successful* alignment.

    `None` — the word carries no gold boundary and is counted as unaligned — when the
    inputs are empty, when the `difflib` similarity between the surface and the
    concatenated segments is below `MIN_ALIGN_RATIO`, or when the projected offsets are
    not strictly increasing inside `(0, len(surface_slp1))`. The last case is what rejects
    a segmentation that cannot physically fit the surface (two copies of `ab` against a
    surface holding one).

    **Where a vowel-sandhi-fused character lands** (docs/decisions.md, 2026-09-05,
    "Correction: the fused character does not consistently join the left segment"). When
    sandhi contracts the two vowels either side of a join into one surface character, that
    character belongs to both segments and no index is correct. Which side `difflib` gives
    it to is not uniform: the fused character joins the **left** segment, unless the sandhi
    output happens to equal the right segment's first character (`a` + `E` -> `E`,
    `a` + `A` -> `A`, `a` + `O` -> `O`), in which case `difflib` matches it to the right
    segment and the boundary lands **before** it. Pinned by regression tests:
    `rAmeti` ['rAma', 'iti'] -> [4]; `tatrEva` ['tatra', 'eva'] -> [5];
    `vacanenEkam` ['vacanena', 'Ekam'] -> [7]; `sAgacCat` ['sa', 'AgacCat'] -> [1].
    The convention is identical for every arm, so paired comparisons are unaffected;
    absolute MorphScore is not portable, which is why the metric also reports a symmetric
    +/-1-character tolerant variant.
    """
    if not surface_slp1 or not segments_slp1:
        return None
    concatenation = "".join(segments_slp1)
    if not concatenation:
        return None

    matcher = SequenceMatcher(None, surface_slp1, concatenation, autojunk=False)
    if matcher.ratio() < MIN_ALIGN_RATIO:
        return None
    if len(segments_slp1) == 1:
        return []

    blocks = matcher.get_matching_blocks()
    offsets: list[int] = []
    cursor = 0
    for segment in segments_slp1[:-1]:
        cursor += len(segment)
        offsets.append(_project(blocks, cursor, len(surface_slp1)))

    previous = 0
    for offset in offsets:
        if not previous < offset < len(surface_slp1):
            return None
        previous = offset
    return offsets


def stem_boundary(segment_slp1: str, lemma_slp1: str) -> int | None:
    """Length of the common prefix of segment and lemma — the derived stem/ending split.

    `None` when there is no lemma, when the prefix is shorter than `MIN_STEM_LENGTH`, or
    when it covers the whole segment (an uninflected form has no ending to split off).

    Heuristic, not DCS annotation, and it is wrong in both directions. It under-splits when
    the lemma's own vowel is rewritten by inflection — `stem_boundary("BAvAnAm", "BAva")` is
    **3**, not 4, because `BAva` ends in a short `a` where the stem has long `A` (the
    Experiment 04 brief's worked example says 4; the SLP1 strings say 3) — and it gives up
    entirely when reduplication or guṇa moves the stem, as in
    `stem_boundary("jagAda", "gad") is None`. Every table built on it says "heuristic".
    """
    if not segment_slp1 or lemma_slp1 in _EMPTY_LEMMAS:
        return None
    length = 0
    for segment_char, lemma_char in zip(segment_slp1, lemma_slp1, strict=False):
        if segment_char != lemma_char:
            break
        length += 1
    if length < MIN_STEM_LENGTH or length >= len(segment_slp1):
        return None
    return length


def mark(text: str, offsets: Sequence[int], marker: str = BOUNDARY_MARKER) -> str:
    """Insert `marker` at each of `offsets` (character indices into `text`).

    Offsets are sorted before insertion, so a caller may pass segment and stem boundaries
    in whatever order it computed them. Raises `ValueError` on an offset outside
    `[0, len(text)]`, which can only mean a projection bug upstream.
    """
    ordered = sorted(offsets)
    if ordered and not (0 <= ordered[0] and ordered[-1] <= len(text)):
        raise ValueError(f"offsets {ordered} out of range for a {len(text)}-character string")
    pieces: list[str] = []
    previous = 0
    for offset in ordered:
        pieces.append(text[previous:offset])
        previous = offset
    pieces.append(text[previous:])
    return marker.join(pieces)


# -------------------------------------------------------------------------- GoldSentence


@dataclass(frozen=True)
class GoldSentence:
    """One DCS sentence with its gold boundaries, in both text variants Exp04 trains on.

    * `text_slp1` — the sandhied sentence, words separated by single spaces.
    * `oracle_split_slp1` — the same sentence with every gold segment separated by a
      space: the `_oracle_dcs` arms' training text and MorphScore's split-side input.
    * `segment_offsets[i]` — boundaries inside `text_slp1`'s i-th word, relative to that
      word; `None` when the word could not be aligned (it is then excluded from MorphScore
      rather than scored against a guess).
    * `stem_offsets[i]` — the i-th word's derived stem/ending boundaries as **absolute**
      indices into `oracle_split_slp1` (the split text is where a stem boundary is
      unambiguous, because each segment stands alone there). Heuristic; see
      `stem_boundary`.
    * `t5_marked` / `t6_marked` — `text_slp1` and `oracle_split_slp1` with
      `BOUNDARY_MARKER` at the boundaries a constrained merge may not cross. `t5_marked`
      carries the segment boundaries *and* the stem boundaries projected into the sandhied
      surface; `t6_marked` carries only the stem boundaries, since the segments are
      already whitespace-separated there.
    * `n_words` / `n_words_aligned` — words in `text_slp1`, and how many have a non-`None`
      alignment.
    * `human_verified` — no token of the sentence carries `UnsandhiedReconstructed=True`.
    """

    text_slp1: str
    oracle_split_slp1: str
    segment_offsets: list[list[int] | None]
    stem_offsets: list[list[int]]
    t5_marked: str
    t6_marked: str
    n_words: int
    n_words_aligned: int
    human_verified: bool

    def to_dict(self) -> dict[str, Any]:
        """A plain dict for one jsonl line (`dataclasses.asdict`, no tuples or Paths)."""
        return asdict(self)


def _match_text_words_to_token_block(
    text_words: Sequence[str], token_words: Sequence[str]
) -> list[int | None]:
    """For each `# text` word, the index of the token-block word it is, or `None`.

    Only `difflib`'s `equal` blocks map: a `# text` word that the token block does not
    reproduce verbatim is not trusted to carry that block's segmentation, which is the
    "set alignment to `None` for mismatched words" rule.
    """
    mapping: list[int | None] = [None] * len(text_words)
    matcher = SequenceMatcher(None, list(text_words), list(token_words), autojunk=False)
    for tag, text_start, text_end, token_start, _token_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        for step in range(text_end - text_start):
            mapping[text_start + step] = token_start + step
    return mapping


def text_words_slp1(sentence: DcsSentence) -> list[str]:
    """The `# text` line as SLP1 words — exactly the words `GoldSentence` is built over.

    Whitespace is normalised to single spaces and each word is transliterated on its own,
    so word boundaries survive transliteration exactly; a word that transliterates to the
    empty string (punctuation-only) is dropped, which is why a caller wanting the word
    count must call this rather than `sentence.text_iast.split()`.

    Public so a caller can apply a `min_words` filter *before* paying for alignment: the
    ingestion drops 14,204 short sentences, and building their gold boundaries first is
    work thrown away. `_slp1` is memoised, so the second call inside
    `build_gold_sentence` is free.
    """
    return [
        slp1
        for slp1 in (_slp1(word) for word in _WHITESPACE.split(sentence.text_iast.strip()) if word)
        if slp1
    ]


def build_gold_sentence(sentence: DcsSentence) -> GoldSentence:
    """Convert one parsed `DcsSentence` into SLP1 with its gold boundaries located.

    Words come from `text_words_slp1`; they are matched to the token block by
    `_match_text_words_to_token_block`.
    """
    text_words = text_words_slp1(sentence)
    token_words = [_slp1(word.surface_iast) for word in sentence.words]
    mapping = _match_text_words_to_token_block(text_words, token_words)

    oracle_words: list[str] = []
    segment_offsets: list[list[int] | None] = []
    stem_offsets: list[list[int]] = []
    marked_words: list[str] = []
    oracle_cursor = 0

    for index, word in enumerate(text_words):
        token_index = mapping[index]
        segments, lemmas = _segments_and_lemmas(sentence, token_index, word)
        offsets = align_segments(word, segments) if token_index is not None else None
        segment_offsets.append(offsets)

        # Stem boundaries, absolute in `oracle_split_slp1`, and their projections into the
        # sandhied surface for `t5_marked`.
        segment_starts_surface = [0, *offsets] if offsets is not None else []
        segment_ends_surface = [*offsets, len(word)] if offsets is not None else []
        absolute_stems: list[int] = []
        projected_stems: list[int] = []
        segment_cursor = oracle_cursor
        for segment_index, (segment, lemma) in enumerate(zip(segments, lemmas, strict=True)):
            boundary = stem_boundary(segment, lemma)
            if boundary is not None:
                absolute_stems.append(segment_cursor + boundary)
                if offsets is not None:
                    projected = segment_starts_surface[segment_index] + boundary
                    if segment_starts_surface[segment_index] < projected < (
                        segment_ends_surface[segment_index]
                    ):
                        projected_stems.append(projected)
            segment_cursor += len(segment) + 1
        stem_offsets.append(absolute_stems)

        oracle_word = " ".join(segments)
        oracle_words.append(oracle_word)
        oracle_cursor += len(oracle_word) + 1

        word_marks = sorted({*(offsets or []), *projected_stems})
        marked_words.append(mark(word, word_marks))

    oracle_split_slp1 = " ".join(oracle_words)
    flat_stems = sorted(offset for word_stems in stem_offsets for offset in word_stems)
    return GoldSentence(
        text_slp1=" ".join(text_words),
        oracle_split_slp1=oracle_split_slp1,
        segment_offsets=segment_offsets,
        stem_offsets=stem_offsets,
        t5_marked=" ".join(marked_words),
        t6_marked=mark(oracle_split_slp1, flat_stems),
        n_words=len(text_words),
        n_words_aligned=sum(1 for offsets in segment_offsets if offsets is not None),
        human_verified=sentence_is_human_verified(sentence),
    )


def _segments_and_lemmas(
    sentence: DcsSentence, token_index: int | None, word: str
) -> tuple[list[str], list[str]]:
    """The SLP1 segments and lemmas of one `# text` word.

    A word with no matching token-block word is its own single segment with no lemma, so
    it passes through the oracle split unsplit and contributes no boundary.
    """
    if token_index is None:
        return [word], [""]
    tokens = sentence.words[token_index].tokens
    segments = [_slp1(token.unsandhied_iast) for token in tokens]
    lemmas = [_slp1(token.lemma_iast) if token.lemma_iast else "" for token in tokens]
    kept = [(segment, lemma) for segment, lemma in zip(segments, lemmas, strict=True) if segment]
    if not kept:
        return [word], [""]
    return [segment for segment, _ in kept], [lemma for _, lemma in kept]
