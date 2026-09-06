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
* **Secondary — the stem/ending boundary.** `stem_boundary` is **sandhi-aware and
  part-of-speech gated**, and it is a **heuristic**, not DCS annotation; every table that
  uses it must say so. The rule, in full (docs/decisions.md, 2026-09-05, "Stem boundaries
  are heuristic; sandhi-aware rule with part-of-speech exclusions", and the CORRECTION
  entry that pins the wording):

  1. A segment whose UPOS is in `NO_STEM_UPOS` — pronouns, particles, adverbs, both
     conjunction classes, adpositions and numerals — gets **no** stem boundary. These are
     the closed classes whose surface forms are suppletive (`yaH`/`yad`, `mama`/`mad`) or
     whose "ending" is not an ending, so a prefix rule can only invent a boundary.
  2. Otherwise take the longest common prefix of segment and lemma. If it covers the whole
     lemma, the surface preserves the lemma and the cut goes at `len(lemma)`
     (`vIram`/`vIra` -> 4, i.e. `vIra|m`).
  3. If it stops exactly one character short of the lemma's end, and both the lemma's final
     character and the surface character at that position are vowels, the inflection has
     **fused or lengthened the stem's final vowel** — `vIrAH`/`vIra`, `anuBAvena`/`anuBAva`
     — and the cut goes **before that vowel** (3 and 6), which is the consonantal body.
     Such a cut is flagged `fused`: it is inside the lemma by character count, but it is
     the linguistically motivated position, not an accident of the prefix.
  4. Anything else — the lemma and the surface diverge inside the consonantal body, as in
     `jagAda`/`gad` or `gacCati`/`gam` — yields **no boundary at all**. The old rule cut
     there anyway, at the prefix, which is what made 46.8% of its cuts fall inside the
     lemma; `stem_boundary_lcp` keeps that rule for the audit that measures it.

  Then the two long-standing guards: the stem must be at least `MIN_STEM_LENGTH`
  characters and the ending must be non-empty.

`GoldSentence` packages both, in the three marked text variants Experiment 04 trains on:
the sandhied text with segment *and* stem marks (`t5_marked`), the sandhied text with
**segment marks only** (`t5seg_marked`, the clean "MorphBPE with gold boundaries" arm,
which owes nothing to the stem heuristic) and the gold "oracle" split with stem marks
(`t6_marked`), each carrying `BOUNDARY_MARKER` (U+001F) at the boundaries a constrained BPE
may not merge across (docs/decisions.md, "MorphBPE-hard implemented as boundary-marker
pre-tokenisation").
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
    "IAST_ANUSVARA_NORMALISATION",
    "MIN_ALIGN_RATIO",
    "MIN_STEM_LENGTH",
    "NO_STEM_UPOS",
    "SLP1_VOWELS",
    "StemAudit",
    "align_segments",
    "build_gold_sentence",
    "mark",
    "normalise_iast",
    "stem_boundary",
    "stem_boundary_lcp",
    "stem_cut_inside_lemma",
    "stem_cut_is_fused",
    "stem_rule_audit",
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

#: SLP1's vowels, simple and compound, short and long. Case is significant: `a` is short
#: *a* and `A` is long *ā*, which is exactly the distinction `stem_cut_is_fused` turns on.
SLP1_VOWELS = frozenset("aAiIuUfFxXeEoO")

#: Universal parts of speech that get no stem boundary (docs/decisions.md, 2026-09-05,
#: "Stem boundaries are heuristic; sandhi-aware rule with part-of-speech exclusions"). Every
#: one of them is a closed class whose surface forms are suppletive (`yaH`/`yad`,
#: `mama`/`mad`, `dvau`/`dvi`) or invariant (`iti`, `ca`, `eva`), so a prefix rule applied to
#: them can only invent a boundary that is not there.
NO_STEM_UPOS = frozenset({"PRON", "PART", "ADV", "CCONJ", "SCONJ", "ADP", "NUM"})

_WHITESPACE = re.compile(r"\s+")


#: IAST anusvāra written with a dot *above* rather than below. DCS uses both, and
#: `indic_transliteration` maps only `ṃ` (U+1E43); `ṁ` (U+1E41) has no SLP1 spelling and is
#: passed through into the SLP1 text, where it is a character no tokenizer arm can spell.
#: 1,119 DCS sentences carried one. The two are the same phoneme in the same notation, so
#: normalising before conversion is a spelling fix, not a change of content
#: (docs/decisions.md, 2026-09-05, "Sangraha quality filter calibrated on a sample").
IAST_ANUSVARA_NORMALISATION: dict[str, str] = {"ṁ": "ṃ"}

_IAST_NORMALISATION_TABLE = str.maketrans(IAST_ANUSVARA_NORMALISATION)


def normalise_iast(iast: str) -> str:
    """`iast` with `ṁ` rewritten to `ṃ`; nothing else is touched."""
    return iast.translate(_IAST_NORMALISATION_TABLE)


@lru_cache(maxsize=1 << 20)
def _slp1(iast: str) -> str:
    """`normalise_iast` then `to_slp1(·, "iast")`, memoised.

    DCS is 750,660 sentences of 4.29M word tokens over a far smaller vocabulary, and each
    is transliterated several times (surface, unsandhied, lemma). The cache is what makes
    the whole ingestion a 92-second job rather than a transliteration-bound one.

    This is the project's single DCS IAST -> SLP1 conversion point, which is why the
    normalisation lives here: putting it in the ingestion script would leave every other
    caller of `build_gold_sentence` converting `ṁ` to itself.
    """
    return to_slp1(normalise_iast(iast), "iast")


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


def _common_prefix_length(segment_slp1: str, lemma_slp1: str) -> int:
    """How many leading characters `segment_slp1` and `lemma_slp1` share."""
    length = 0
    for segment_char, lemma_char in zip(segment_slp1, lemma_slp1, strict=False):
        if segment_char != lemma_char:
            break
        length += 1
    return length


def stem_boundary_lcp(segment_slp1: str, lemma_slp1: str) -> int | None:
    """The **old** rule: the longest common prefix of segment and lemma, with the guards.

    `None` when there is no lemma, when the prefix is shorter than `MIN_STEM_LENGTH`, or
    when it covers the whole segment (an uninflected form has no ending to split off).

    Superseded by `stem_boundary` and kept only so the ingestion can *measure* what it
    replaced: 46.8% of the cuts this rule makes on held-out DCS fall strictly inside the
    lemma (`vIr|AH` for lemma `vIra`, `anuBAv|ena` for `anuBAva`, `gac|Cati` for `gam`), so
    the T6 arms it used to mark were constrained by an inconsistent heuristic
    (docs/decisions.md, 2026-09-05, "Stem boundaries are heuristic"). Nothing in the
    pipeline calls this for a boundary any more; `stem_rule_audit` calls it for a number.
    """
    if not segment_slp1 or lemma_slp1 in _EMPTY_LEMMAS:
        return None
    length = _common_prefix_length(segment_slp1, lemma_slp1)
    if length < MIN_STEM_LENGTH or length >= len(segment_slp1):
        return None
    return length


def stem_cut_is_fused(segment_slp1: str, lemma_slp1: str, cut: int) -> bool:
    """Whether `cut` sits before a stem-final vowel the inflection fused or lengthened.

    True exactly when the cut stops one character short of the lemma's end, the lemma's
    final character is a vowel and the surface carries a vowel in its place: `vIrAH`/`vIra`
    at 3 (`a` + `as` -> `A`), `anuBAvena`/`anuBAva` at 6 (`a` + `ina` -> `ena`). No single
    character offset is *the* boundary in these words — the surface vowel belongs to the
    stem and to the ending at once — so the cut goes before it and carries this flag, and
    `stem_rule_audit` counts these separately from the cuts that fall inside the lemma's
    consonantal body, which are simply wrong.
    """
    return (
        cut == len(lemma_slp1) - 1
        and cut < len(segment_slp1)
        and bool(lemma_slp1)
        and lemma_slp1[-1] in SLP1_VOWELS
        and segment_slp1[cut] in SLP1_VOWELS
    )


def stem_cut_inside_lemma(segment_slp1: str, lemma_slp1: str, cut: int) -> bool:
    """Whether `cut` falls strictly inside the lemma — i.e. the stem is a *proper* prefix.

    The literal character-count test, `cut < len(lemma)`, deliberately counting the fused
    cuts of `stem_cut_is_fused` as inside: a caller measuring a rule reports both this
    fraction and the fused one, and subtracts if it wants the fraction of cuts that are
    inside the lemma *for no good reason*. `segment_slp1` is unused by the test itself and
    is in the signature so a caller passes the whole triple and cannot mix up which string
    is which.
    """
    del segment_slp1  # part of the triple for the caller's clarity; not needed by the test
    return cut < len(lemma_slp1)


def stem_boundary(segment_slp1: str, lemma_slp1: str, upos: str = "") -> int | None:
    """The stem/ending split: sandhi-aware, part-of-speech gated, never inside the body.

    See this module's docstring for the rule in full. In one line: closed-class parts of
    speech (`NO_STEM_UPOS`) get nothing; the cut goes at `len(lemma)` when the surface
    preserves the lemma, before the fused vowel when the lemma is vowel-final and the
    surface has a different vowel there, and nowhere at all when segment and lemma diverge
    inside the consonantal body.

    `upos` defaults to `""` so a caller with no part-of-speech annotation still gets the
    sandhi-aware behaviour; DCS always supplies one.

    Heuristic, not DCS annotation. What it fixes relative to `stem_boundary_lcp` is the
    third case — `stem_boundary("jagAda", "gad") is None` under both rules, but
    `stem_boundary("gacCati", "gam")` is `None` here where the LCP rule cut at 2. What it
    does *not* fix is that a fused cut is one character to the left of where a linguist
    would draw the morpheme boundary; that is flagged, counted and reported, never hidden.
    """
    if upos in NO_STEM_UPOS:
        return None
    if not segment_slp1 or lemma_slp1 in _EMPTY_LEMMAS:
        return None
    cut = _common_prefix_length(segment_slp1, lemma_slp1)
    if cut != len(lemma_slp1) and not stem_cut_is_fused(segment_slp1, lemma_slp1, cut):
        return None
    if cut < MIN_STEM_LENGTH or cut >= len(segment_slp1):
        return None
    return cut


@dataclass(frozen=True)
class StemAudit:
    """How many cuts one stem rule made on a corpus, and how many fell inside the lemma.

    `n_segments` is every segment the rule was offered (lemma or not, closed class or not),
    so `n_cuts / n_segments` says how much of the corpus the rule marks at all.
    `n_inside_lemma` is the literal `stem_cut_inside_lemma` count and `n_fused` the subset
    of those that `stem_cut_is_fused` explains; `n_inside_lemma - n_fused` is the number of
    cuts that fall inside the lemma's consonantal body.

    For `stem_boundary` that difference is **0 by construction, not by measurement**: the
    rule returns a cut only when `cut == len(lemma)` — outside the lemma, so not counted —
    or when `stem_cut_is_fused` flags it, so `n_inside_lemma == n_fused` identically and
    `fraction_inside_lemma_excluding_fused` can only ever be 0.0. Reading that 0 as evidence
    is circular. The informative number for the sandhi-aware rule is `fraction_fused` (how
    much of what it marks sits at a boundary no single offset gets right, 36.5% on the
    held-out split); the difference is a real measurement only for `stem_boundary_lcp`,
    which is free to cut anywhere and puts 21.4% of its cuts inside the lemma body.
    """

    n_segments: int = 0
    n_cuts: int = 0
    n_inside_lemma: int = 0
    n_fused: int = 0

    def to_dict(self) -> dict[str, Any]:
        """A manifest-ready dict, with the two fractions the decisions entry asks for."""
        return {
            "n_segments": self.n_segments,
            "n_cuts": self.n_cuts,
            "n_inside_lemma": self.n_inside_lemma,
            "n_fused": self.n_fused,
            "n_inside_lemma_excluding_fused": self.n_inside_lemma - self.n_fused,
            "fraction_inside_lemma": (
                self.n_inside_lemma / self.n_cuts if self.n_cuts else 0.0
            ),
            "fraction_fused": self.n_fused / self.n_cuts if self.n_cuts else 0.0,
            "fraction_inside_lemma_excluding_fused": (
                (self.n_inside_lemma - self.n_fused) / self.n_cuts if self.n_cuts else 0.0
            ),
        }


def stem_rule_audit(sentence: DcsSentence) -> dict[str, StemAudit]:
    """`{"sandhi_aware": ..., "lcp": ...}` — both rules' cut counts over one sentence.

    Runs on the same `(segment, lemma, upos)` triples `build_gold_sentence` derives its
    boundaries from, so the numbers describe the corpus as ingested rather than a
    re-derivation of it. The ingestion accumulates these over the held-out split and writes
    both dicts into the manifest. Both rules are audited because only the comparison is
    informative: that the sandhi-aware rule's cuts never fall inside the lemma body follows
    from its definition (see `StemAudit`), so what the manifest actually establishes is how
    much of the corpus each rule marks and how the LCP rule it replaced differs — 120,459
    cuts with 21.4% inside the lemma body against 90,878 cuts with none.
    """
    counters = {"sandhi_aware": StemAudit(), "lcp": StemAudit()}
    for word in sentence.words:
        for token in word.tokens:
            segment = _slp1(token.unsandhied_iast)
            lemma = _slp1(token.lemma_iast) if token.lemma_iast else ""
            if not segment:
                continue
            for name, cut in (
                ("sandhi_aware", stem_boundary(segment, lemma, token.upos)),
                ("lcp", stem_boundary_lcp(segment, lemma)),
            ):
                current = counters[name]
                counters[name] = StemAudit(
                    n_segments=current.n_segments + 1,
                    n_cuts=current.n_cuts + (cut is not None),
                    n_inside_lemma=current.n_inside_lemma
                    + (cut is not None and stem_cut_inside_lemma(segment, lemma, cut)),
                    n_fused=current.n_fused
                    + (cut is not None and stem_cut_is_fused(segment, lemma, cut)),
                )
    return counters


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
    * `t5_marked` / `t5seg_marked` / `t6_marked` — the three marked training texts, with
      `BOUNDARY_MARKER` at the boundaries a constrained merge may not cross. `t5_marked` is
      `text_slp1` with the segment boundaries *and* the stem boundaries projected into the
      sandhied surface; `t5seg_marked` is `text_slp1` with the **segment boundaries only**,
      so an arm trained on it is constrained by DCS annotation alone and owes nothing to the
      stem heuristic; `t6_marked` is `oracle_split_slp1` with only the stem boundaries,
      since the segments are already whitespace-separated there.
    * `n_words` / `n_words_aligned` — words in `text_slp1`, and how many have a non-`None`
      alignment.
    * `human_verified` — no token of the sentence carries `UnsandhiedReconstructed=True`.
    """

    text_slp1: str
    oracle_split_slp1: str
    segment_offsets: list[list[int] | None]
    stem_offsets: list[list[int]]
    t5_marked: str
    t5seg_marked: str
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
    segment_marked_words: list[str] = []
    oracle_cursor = 0

    for index, word in enumerate(text_words):
        token_index = mapping[index]
        segments, lemmas, tags = _segments_and_lemmas(sentence, token_index, word)
        offsets = align_segments(word, segments) if token_index is not None else None
        segment_offsets.append(offsets)

        # Stem boundaries, absolute in `oracle_split_slp1`, and their projections into the
        # sandhied surface for `t5_marked`.
        segment_starts_surface = [0, *offsets] if offsets is not None else []
        segment_ends_surface = [*offsets, len(word)] if offsets is not None else []
        absolute_stems: list[int] = []
        projected_stems: list[int] = []
        segment_cursor = oracle_cursor
        for segment_index, (segment, lemma, upos) in enumerate(
            zip(segments, lemmas, tags, strict=True)
        ):
            boundary = stem_boundary(segment, lemma, upos)
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
        segment_marked_words.append(mark(word, sorted(offsets or [])))

    oracle_split_slp1 = " ".join(oracle_words)
    flat_stems = sorted(offset for word_stems in stem_offsets for offset in word_stems)
    return GoldSentence(
        text_slp1=" ".join(text_words),
        oracle_split_slp1=oracle_split_slp1,
        segment_offsets=segment_offsets,
        stem_offsets=stem_offsets,
        t5_marked=" ".join(marked_words),
        t5seg_marked=" ".join(segment_marked_words),
        t6_marked=mark(oracle_split_slp1, flat_stems),
        n_words=len(text_words),
        n_words_aligned=sum(1 for offsets in segment_offsets if offsets is not None),
        human_verified=sentence_is_human_verified(sentence),
    )


def _segments_and_lemmas(
    sentence: DcsSentence, token_index: int | None, word: str
) -> tuple[list[str], list[str], list[str]]:
    """The SLP1 segments, lemmas and UPOS tags of one `# text` word.

    The UPOS tags come along because `stem_boundary` gates on them: a pronoun or a particle
    gets no stem boundary at all, and the tag is the only thing that says which is which.

    A word with no matching token-block word is its own single segment with no lemma and no
    tag, so it passes through the oracle split unsplit and contributes no boundary.
    """
    if token_index is None:
        return [word], [""], [""]
    tokens = sentence.words[token_index].tokens
    triples = [
        (
            _slp1(token.unsandhied_iast),
            _slp1(token.lemma_iast) if token.lemma_iast else "",
            token.upos,
        )
        for token in tokens
    ]
    kept = [triple for triple in triples if triple[0]]
    if not kept:
        return [word], [""], [""]
    return (
        [segment for segment, _, _ in kept],
        [lemma for _, lemma, _ in kept],
        [upos for _, _, upos in kept],
    )
