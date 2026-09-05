"""Experiment 04 — does forbidding merges across gold morpheme boundaries pay? (RQ4)

Two questions, one script, and they are not the same question.

**H3, MorphScore half** (outline §1): sandhi splitting before subword learning *raises
MorphScore* relative to a matched-vocabulary tokenizer trained on raw sandhied text. Here
that is tested with the *gold* split rather than a model's — the `T4_*_oracle_dcs` arms are
trained on DCS's own segmentation — so it is the upper bound on what splitting can buy,
measured on held-out DCS sentences with gold boundaries.

**H4, TPP half**: a morpheme-constrained vocabulary (`T5` and `T5seg` on raw text, `T6`
on gold-split text) buys tokens. `T5` is constrained on gold segment boundaries *and*
heuristic stem boundaries; `T5seg` on the gold segment boundaries alone, so it is the
arm whose result owes nothing to the stem heuristic. H4's own wording is about *tokens
to a reference bits-per-character*, which needs a trained language model and belongs to
Experiment 05; what this script can answer is whether the constrained arm spends fewer
tokens per unit of meaning than the unconstrained arm trained on the same sentences at
the same vocabulary size (docs/decisions.md, 2026-09-05, "Experiment 04 headline is the
paired TPP delta between constrained and unconstrained DCS arms; MorphScore is the
mechanism check").

**The headline is the paired delta, not the level.** Every arm here is trained on DCS,
which has no matched English side — `E1_bpe_64k` was trained on the English half of the
*parallel* corpora — so a TPP *level* against any English pivot mixes the arm's cost with a
corpus mismatch. The delta between two DCS arms measured on the same sentences against the
same English side does not: the English total cancels, and what is left is the difference
the constraint (or the splitting) made. Both pivots are reported as context and neither
carries the verdict.

**Three separate measurements of the constraint, deliberately kept apart.** MorphScore asks
whether the arm's boundaries land where the morphology does. The out-of-sample violation
audit asks whether the arm's *tokens* ever span a gold boundary on text it never saw — the
mechanism, since the marker constrains learning and not application (docs/decisions.md,
"Twelve DCS arms trained; the boundary-marker constraint halves cross-boundary tokens but
cannot reach zero"). TPP asks whether any of it is worth paying for. An arm can win one and
lose another, and the README reports all three.

**Fertility never leads** (CLAUDE.md §2.1) and, as in Experiment 03, is reported twice: over
the raw sentence's word count for every arm (the comparable denominator) and, for split
arms only, over the split text's own words.

Run it with `uv run python experiments/04_morph_constrained/run.py`. Relative paths in the
config are resolved against the repository root, so the working directory does not matter.
Every function below is pure or takes its I/O paths explicitly, so `tests/test_exp04.py` can
exercise the whole pipeline — the gold-boundary derivations, the aggregation, the plotting
and `run()` end to end — on synthetic data, offline.
"""

import argparse
import json
import logging
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sanskrit_tok.data.exclusion import (
    load_exclusion_hashes,
    sentence_hash_en,
    sentence_hash_slp1,
)
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.experiment import (
    ENGLISH_LANGUAGE,
    SANSKRIT_LANGUAGE,
    deletion_cost,
    exclusion_check_for,
    load_arms,
    load_config,
    load_corpus_entry,
    provenance,
    repo_root,
    resolve_path,
    select_aligned_indices,
    summarise_tpp,
    take_indices,
    text_invariants,
    tokenizer_file_sha256,
    tokenizer_sources,
    unavailable_caption,
    write_results,
)
from sanskrit_tok.metrics._ratio import token_ratio
from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility, fertility_against_reference
from sanskrit_tok.metrics.morphscore import morphscore
from sanskrit_tok.metrics.summary import summarise_metric
from sanskrit_tok.metrics.tpp import tpp, tpp_paired_delta
from sanskrit_tok.tokenizers.base import spans_cover_text
from sanskrit_tok.tokenizers.morph_bpe import (
    BOUNDARY_MARKER,
    assert_no_cross_boundary_merges,
    strip_markers,
)
from sanskrit_tok.tokenizers.registry import LoadedTokenizer

logger = logging.getLogger("exp04")

#: The two Sanskrit texts of a parallel corpus and the keys `results.json` stores them
#: under. `RAW` is the SLP1 transliteration of the corpus as written; `SPLIT` is the
#: ByT5-reconciled split text from `data/processed/split/`, which is the only split
#: available for parallel corpora — the split `_dcs` arms were *trained* on DCS's gold
#: split, and the README states the mismatch.
RAW = "raw_slp1"
SPLIT = "reconciled"

#: What an arm does with those texts, read off its name (`arm_text_kind`).
RAW_ARM = "raw"
SPLIT_ARM = "split"

#: The two boundary granularities, and only one of them is gold. `segment` is DCS's own
#: word/compound segmentation located in the sandhied surface — the primary, the one H3/H4's
#: MorphScore claims use, and the only one that is annotation. `stem` is the stem/ending
#: split *derived* from the segment and its lemma by the sandhi-aware rule in
#: `sanskrit_tok.data.boundaries.stem_boundary`: a **heuristic**, never called gold, and
#: labelled "heuristic stem" everywhere it is reported (docs/decisions.md, 2026-09-05,
#: "Stem boundaries are heuristic; sandhi-aware rule with part-of-speech exclusions").
GRANULARITY_SEGMENT = "segment"
GRANULARITY_STEM = "stem"

#: The two populations MorphScore is reported over. 68% of DCS's segmentation is
#: machine-generated, so the verified subset is the number to read when the question is
#: "is this measuring the tokenizer or the annotator?".
SUBSET_ALL = "all"
SUBSET_HUMAN = "human_verified"

#: `results.json`'s tolerance keys and the `morphscore(tolerance=)` each stands for. Exact
#: is Arnett & Bergen's definition and the one paired comparisons use; `pm1` bounds how much
#: of a score is the sandhi offset convention rather than the tokenizer (docs/decisions.md,
#: "MorphScore on sandhied surfaces reports exact and ±1-character variants").
TOLERANCES: dict[str, int] = {"exact": 0, "pm1": 1}

#: Which English side a TPP column is divided by. Neither is a matched control here — see
#: the module docstring — so both roles are context and the labels say which is which.
ROLE_CONTROLLED = "controlled"
ROLE_DEPLOYED = "deployed"

#: Arm-name suffixes naming a training corpus rather than an algorithm or a vocabulary
#: size, longest first so `_oracle_dcs` is recognised before the `_dcs` it ends with. The
#: same tuple the registry uses; duplicated rather than imported because the registry's copy
#: is private, and pinned against the registry by `tests/test_exp04.py`'s arm-name cases.
VARIANT_SUFFIXES: tuple[str, ...] = ("_oracle_dcs", "_dcs")

#: Which algorithms may be paired. `morphbpe` *is* BPE with a constrained merge table, so a
#: T5/T6 arm is matched against a T1/T4 arm; `unigram` is a different learner and pairing it
#: with either would put the algorithm and the constraint into one delta.
ALGORITHM_BASE: dict[str, str] = {"bpe": "bpe", "morphbpe": "bpe", "unigram": "unigram"}

FIGURE_STEM = "constraint_effects"
FIGURE_SUPTITLE = "Morpheme-constrained merges: paired TPP deltas and MorphScore"
FIGURE_SUBTITLE = (
    "left: TPP(a) − TPP(b) on parallel text, same sentences and same English side, "
    "95% paired bootstrap CI · right: boundary F1 on held-out DCS"
)
FIGURE_CAPTION = (
    "All `_dcs` arms are trained on the DCS training split; `oracle` arms are trained on "
    "DCS's own gold segmentation, which is an upper bound on what splitting can buy, while "
    "split arms are *evaluated* on ByT5-reconciled text. `T5` is constrained on gold "
    "segment boundaries and heuristic stem boundaries, `T5seg` on the gold segment "
    "boundaries alone, `T6` on heuristic stem boundaries inside the gold split. Arms marked "
    "provisional (*) are trained on the parallel corpora, not on DCS, and are shown for "
    "continuity with Experiments 02–03 only. MorphScore for raw arms uses DCS segment "
    "boundaries; for split arms it uses the **heuristic stem** boundary derived from the "
    "lemma, which is not annotation, so the two columns are not one series. Thin markers "
    "are the ±1-character tolerant variant. The English pivot is cross-corpus (E1 was "
    "trained on the parallel English side, not on DCS), which is why the deltas — where the "
    "English side cancels — carry the verdict and the levels do not."
)
FIGURE_Y_PAD_FRACTION = 0.15

# ------------------------------------------------------------------------- arm names


def strip_variant_suffix(name: str) -> tuple[str, str]:
    """`("T5_morphbpe_raw_32k", "dcs")` from `"T5_morphbpe_raw_32k_dcs"`.

    The variant is the training-corpus label; `""` for an arm that carries none, which is
    every off-the-shelf arm and every arm predating Experiment 04.
    """
    for suffix in VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return name, ""


def _fields(name: str) -> list[str]:
    return strip_variant_suffix(name)[0].split("_")


def arm_algorithm(name: str) -> str:
    """`"morphbpe"` from `"T5_morphbpe_raw_32k_dcs"`: the second field of the arm name."""
    fields = _fields(name)
    if len(fields) < 2:
        raise ValueError(f"arm name {name!r} is not <family>_<algorithm>_..._<vocab>")
    return fields[1]


def arm_vocab(name: str) -> str:
    """`"32k"` from `"T5_morphbpe_raw_32k_dcs"`: the last field before the variant suffix."""
    fields = _fields(name)
    if len(fields) < 2:
        raise ValueError(f"arm name {name!r} is not <family>_<algorithm>_..._<vocab>")
    return fields[-1]


def arm_text_kind(name: str) -> str:
    """Whether an arm is measured on the sandhied surface or on split text.

    Read off the name's text field (`T4_bpe_**split**_32k_oracle_dcs`). An arm whose name
    carries no such field — every off-the-shelf arm — is a `RAW_ARM`: it is applied to the
    corpus as written, which for Sanskrit is the sandhied surface.
    """
    fields = _fields(name)
    return SPLIT_ARM if len(fields) >= 3 and fields[2] == SPLIT_ARM else RAW_ARM


def arm_labels(name: str) -> dict[str, Any]:
    """`{"variant", "provisional", "oracle"}` — the three qualifiers every table must carry.

    `provisional` marks a from-scratch arm trained on the *parallel* corpora rather than on
    DCS (Experiments 02–03's arms, kept here for continuity); an off-the-shelf arm is not
    provisional, it is existing practice, and is not trained by this project at all.
    `oracle` marks an arm trained on DCS's gold segmentation rather than on a model's.
    """
    variant = strip_variant_suffix(name)[1]
    family = name.split("_", 1)[0]
    return {
        "variant": variant,
        "provisional": variant == "" and family in {"T1", "T2", "T4", "T5", "T6", "E1"},
        "oracle": variant == "oracle_dcs",
    }


def granularities_for(kind: str) -> tuple[str, ...]:
    """Which gold granularities are defined for an arm measured on `kind` text.

    A split arm sees the oracle segmentation as whitespace, so there is no segment boundary
    left inside a unit to score: only the stem granularity is defined for it. A raw arm has
    both, and the segment one is primary.
    """
    if kind == SPLIT_ARM:
        return (GRANULARITY_STEM,)
    return (GRANULARITY_SEGMENT, GRANULARITY_STEM)


def check_paired_arms(first: str, second: str) -> None:
    """Raise `ValueError` unless `[first, second]` is a controlled contrast.

    Three things must match, and each of them is a way the delta could silently stop being
    about the constraint: the **vocabulary size** (CLAUDE.md §2.5), the **algorithm family**
    (`morphbpe` is BPE plus the constraint and pairs with `bpe`; `unigram` pairs only with
    `unigram`), and the **training corpus** — both sides must be DCS-trained, or the delta
    would compare a DCS arm with a parallel-corpus arm and read as a corpus effect.
    """
    if arm_vocab(first) != arm_vocab(second):
        raise ValueError(
            f"paired arms [{first}, {second}] differ in vocab size: "
            f"{arm_vocab(first)} vs {arm_vocab(second)}"
        )
    bases = (
        ALGORITHM_BASE.get(arm_algorithm(first), arm_algorithm(first)),
        ALGORITHM_BASE.get(arm_algorithm(second), arm_algorithm(second)),
    )
    if bases[0] != bases[1]:
        raise ValueError(
            f"paired arms [{first}, {second}] differ in algorithm family: "
            f"{bases[0]} vs {bases[1]}"
        )
    variants = (strip_variant_suffix(first)[1], strip_variant_suffix(second)[1])
    if not all(variant in {"dcs", "oracle_dcs"} for variant in variants):
        raise ValueError(
            f"paired arms [{first}, {second}] differ in training corpus: variants "
            f"{variants[0]!r} and {variants[1]!r}; both sides of a controlled delta must "
            "be trained on the DCS split"
        )


def pair_key(first: str, second: str) -> str:
    """`results.json`'s `tpp_delta` key, in delta order: `"<a>/<b>"` for `TPP(a) − TPP(b)`."""
    return f"{first}/{second}"


def pair_label(first: str, second: str) -> str:
    """The figure's x-tick label for one pair: `"T6−T1 32k"`.

    The two family prefixes say which contrast it is (the constraint, the splitting, or
    both) and the shared vocabulary size says at which size; the rest of the arm names is
    what the two sides have in common and would only crowd the axis.
    """
    check_paired_arms(first, second)
    return f"{first.split('_', 1)[0]}−{second.split('_', 1)[0]} {arm_vocab(first)}"


def select_pairs(
    paired_deltas: Sequence[Sequence[str]], arms: Mapping[str, LoadedTokenizer]
) -> list[tuple[str, str]]:
    """The configured pairs whose *both* sides loaded, structure validated for all of them.

    Validation runs on every configured pair, available or not: a config typo pairing two
    vocabulary sizes is a defect that must fail even on a run where one of the arms happens
    to be missing. Availability is then filtered with a WARNING, so one untrained arm costs
    its own pairs and nothing else.
    """
    selected: list[tuple[str, str]] = []
    for pair in paired_deltas:
        names = list(pair)
        if len(names) != 2:
            raise ValueError(f"config['paired_deltas'] entry {names!r} must be [a, b]")
        first, second = names
        check_paired_arms(first, second)
        missing = [name for name in names if name not in arms]
        if missing:
            logger.warning(
                "paired delta %s is skipped: %s unavailable this run",
                pair_key(first, second),
                ", ".join(missing),
            )
            continue
        selected.append((first, second))
    return selected


# -------------------------------------------------------------- the DCS held-out split


@dataclass(frozen=True)
class GoldRecord:
    """One held-out DCS sentence: the two text forms, the gold boundaries, the marked text.

    A read-only mirror of `sanskrit_tok.data.boundaries.GoldSentence` as it was written to
    `heldout.jsonl`, plus the two identifiers. `segment_offsets[i]` is `None` for a word the
    ingestion could not align, which MorphScore skips rather than scoring against a guess.
    """

    sent_id: str
    text_id: int
    text_slp1: str
    oracle_split_slp1: str
    segment_offsets: list[list[int] | None]
    stem_offsets: list[list[int]]
    t5_marked: str
    t6_marked: str
    n_words: int
    n_words_aligned: int
    human_verified: bool

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "GoldRecord":
        return cls(
            sent_id=str(record["sent_id"]),
            text_id=int(record["text_id"]),
            text_slp1=str(record["text_slp1"]),
            oracle_split_slp1=str(record["oracle_split_slp1"]),
            segment_offsets=[
                None if offsets is None else [int(offset) for offset in offsets]
                for offsets in record["segment_offsets"]
            ],
            stem_offsets=[
                [int(offset) for offset in offsets] for offsets in record["stem_offsets"]
            ],
            t5_marked=str(record["t5_marked"]),
            t6_marked=str(record["t6_marked"]),
            n_words=int(record["n_words"]),
            n_words_aligned=int(record["n_words_aligned"]),
            human_verified=bool(record["human_verified"]),
        )


def read_gold_records(path: Path) -> list[GoldRecord]:
    """Every record of the DCS held-out jsonl, in file order.

    A missing file aborts naming the script that writes it: there is no fallback that would
    produce a meaningful MorphScore, and silently measuring nothing is worse than stopping.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no DCS held-out split at {path}. Run "
            "`uv run python experiments/04_morph_constrained/ingest_dcs.py` first."
        )
    records: list[GoldRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(GoldRecord.from_dict(json.loads(stripped)))
    return records


def human_verified(records: Sequence[GoldRecord]) -> list[GoldRecord]:
    """The sentences with no machine-generated segmentation in them."""
    return [record for record in records if record.human_verified]


def subsets(records: Sequence[GoldRecord]) -> dict[str, list[GoldRecord]]:
    """`{SUBSET_ALL: ..., SUBSET_HUMAN: ...}`, in the order tables print them."""
    return {SUBSET_ALL: list(records), SUBSET_HUMAN: human_verified(records)}


def sample_records(
    records: Sequence[GoldRecord], size: int | None, seed: int
) -> tuple[list[GoldRecord], int | None]:
    """`(records, sample_size)`: a fixed random subsample, or everything when `size` is None.

    Returns the size actually used so `results.json` can say which — a MorphScore over a
    tenth of the corpus and one over all of it are different measurements and the file must
    not leave the reader guessing. A `size` at or above the corpus is a no-op.
    """
    if size is None or size >= len(records):
        return list(records), None
    chosen = random.Random(seed).sample(range(len(records)), size)
    return [records[index] for index in sorted(chosen)], size


# -------------------------------------------------------- gold boundaries for MorphScore


def rebase_into_units(text: str, offsets: Sequence[int]) -> list[tuple[str, list[int]]]:
    """Split `text` on single spaces and re-base each absolute offset inside its unit.

    `stem_offsets` are absolute indices into `oracle_split_slp1`, where the whitespace units
    are the gold segments; MorphScore scores one unit at a time, so each offset has to be
    expressed relative to the unit that contains it. An offset landing on the space itself
    belongs to the unit that follows it, which is where the next segment begins.

    Raises `ValueError` on an offset past the end of `text`, which can only mean the record
    and its offsets came from different sentences.
    """
    units: list[tuple[str, list[int]]] = []
    starts: list[int] = []
    cursor = 0
    for unit in text.split(" "):
        units.append((unit, []))
        starts.append(cursor)
        cursor += len(unit) + 1
    for offset in offsets:
        if offset < 0 or offset > len(text):
            raise ValueError(f"offset {offset} out of range for a {len(text)}-character string")
        index = max(position for position, start in enumerate(starts) if start <= offset)
        unit, collected = units[index]
        local = offset - starts[index]
        if local > len(unit):  # the offset sat on the space before the next unit
            index += 1
            unit, collected = units[index]
            local = offset - starts[index]
        collected.append(local)
    return units


def _marked_word_offsets(marked_word: str) -> list[int]:
    """The boundary offsets a marked word carries, as indices into the unmarked word."""
    return strip_markers(marked_word, BOUNDARY_MARKER)[1]


def _projected_stem_offsets(record: GoldRecord) -> list[list[int] | None]:
    """Per surface word, the stem boundaries projected into the sandhied surface.

    `t5_marked` carries the union of the segment boundaries and the stem boundaries the
    ingestion could project into the surface, so the projected stems are exactly its markers
    minus the segment offsets — the ingestion never projects a stem onto a segment boundary
    (it requires the projection to lie strictly inside its segment), so the difference is
    exact rather than approximate.

    A word whose segment alignment failed carries `None`: no stem could be projected into
    it, and reporting `[]` would score the tokenizer as having missed nothing rather than
    skipping the word.
    """
    marked_words = record.t5_marked.split(" ")
    projected: list[list[int] | None] = []
    for marked, offsets in zip(marked_words, record.segment_offsets, strict=True):
        if offsets is None:
            projected.append(None)
            continue
        segment = set(offsets)
        projected.append(
            [offset for offset in _marked_word_offsets(marked) if offset not in segment]
        )
    return projected


def morphscore_inputs(
    records: Sequence[GoldRecord], kind: str, granularity: str
) -> tuple[list[str], list[list[int] | None]]:
    """`(words, gold_offsets)` for one arm kind at one granularity, pooled over `records`.

    Three combinations, and they score different populations of *units*, which is why every
    table names the granularity beside the number:

    * raw + `segment` — the whitespace words of the sandhied sentence against DCS's segment
      boundaries located inside them. The primary.
    * raw + `stem` — the same words against the stem/ending boundaries projected into the
      surface, available only where the word's segment alignment succeeded. Heuristic.
    * split + `stem` — each gold **segment** of the oracle split against the heuristic stem
      boundary inside it. The units are gold; the boundaries scored against are not, and a
      different unit population from the two above: a raw word may hold several segments, so
      the split rows have more units and shorter ones.

    raw + `segment` is what a split arm has no analogue of: in the oracle split the segment
    boundaries *are* whitespace, so there is nothing left inside a unit to find.
    """
    if granularity not in granularities_for(kind):
        raise ValueError(f"granularity {granularity!r} is not defined for a {kind!r} arm")
    words: list[str] = []
    gold: list[list[int] | None] = []
    for record in records:
        if kind == SPLIT_ARM:
            marked_units = record.t6_marked.split(" ")
            for marked in marked_units:
                plain, offsets = strip_markers(marked, BOUNDARY_MARKER)
                words.append(plain)
                gold.append(offsets)
            continue
        surface_words = record.text_slp1.split(" ")
        if granularity == GRANULARITY_SEGMENT:
            offsets_per_word: list[list[int] | None] = list(record.segment_offsets)
        else:
            offsets_per_word = _projected_stem_offsets(record)
        words.extend(surface_words)
        gold.extend(offsets_per_word)
    return words, gold


# ------------------------------------------------------------------ the spans contract


def check_spans_coverage(
    records: Sequence[GoldRecord],
    arms: Mapping[str, LoadedTokenizer],
    names: Sequence[str],
    n: int,
) -> dict[str, dict[str, dict[str, int]]]:
    """`arm -> text form -> {n, n_passed, n_failed}` over the first `n` held-out sentences.

    MorphScore's boundary set is the set of span starts, which is a segmentation of the word
    only if the spans tile it (`base.spans_cover_text`). That is an invariant of the span
    adapters, asserted in their own tests — but it is asserted there on a handful of strings,
    and this is the corpus the numbers come from, so it is re-checked here on both text forms
    and the counts go into `results.json`. An arm that fails has its MorphScore withheld.
    """
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    head = list(records[:n])
    for name in names:
        tokenizer = arms.get(name)
        if tokenizer is None or not tokenizer.supports_spans:
            continue
        per_form: dict[str, dict[str, int]] = {}
        for form, texts in (
            ("text_slp1", [record.text_slp1 for record in head]),
            ("oracle_split_slp1", [record.oracle_split_slp1 for record in head]),
        ):
            passed = sum(1 for text in texts if spans_cover_text(text, tokenizer.spans(text)))
            per_form[form] = {
                "n": len(texts),
                "n_passed": passed,
                "n_failed": len(texts) - passed,
            }
            if passed != len(texts):
                logger.warning(
                    "%s: spans do not cover %d/%d held-out sentence(s) of %s; its MorphScore "
                    "on that form is withheld",
                    name,
                    len(texts) - passed,
                    len(texts),
                    form,
                )
        coverage[name] = per_form
    return coverage


def _coverage_form(kind: str) -> str:
    """Which text form's coverage gates an arm of this kind."""
    return "oracle_split_slp1" if kind == SPLIT_ARM else "text_slp1"


# ------------------------------------------------------------------------- MorphScore


def compute_morphscore(
    records: Sequence[GoldRecord],
    arms: Mapping[str, LoadedTokenizer],
    names: Sequence[str],
    coverage: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """`arm -> granularity -> subset -> tolerance -> summary`: the mechanism check.

    The nesting is the four things that have to be said about every MorphScore number for it
    to mean anything, and flattening any of them into a key prefix would invite reading two
    of them as one series. Two arms are dropped rather than scored: one with no span
    provider at all (a slow tokenizer counts tokens correctly and has no offsets), and one
    whose spans failed the coverage check, whose F1 would be built from a boundary set that
    is not a segmentation.

    The inputs are computed once per `(kind, granularity)` and reused across arms and
    tolerances, since they depend on the gold annotation and not on the tokenizer.
    """
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    populations = subsets(records)
    inputs: dict[tuple[str, str, str], tuple[list[str], list[list[int] | None]]] = {
        (kind, granularity, subset_name): morphscore_inputs(subset_records, kind, granularity)
        for kind in (RAW_ARM, SPLIT_ARM)
        for granularity in granularities_for(kind)
        for subset_name, subset_records in populations.items()
    }

    for name in names:
        tokenizer = arms.get(name)
        if tokenizer is None:
            continue
        if not tokenizer.supports_spans:
            logger.warning(
                "%s: reports no character spans, so it is omitted from MorphScore (every "
                "count-based metric still covers it)",
                name,
            )
            continue
        kind = arm_text_kind(name)
        form = _coverage_form(kind)
        checked = coverage.get(name, {}).get(form)
        if checked is not None and checked.get("n_failed", 0):
            logger.warning(
                "%s: MorphScore aborted — spans coverage failed on %d/%d %s sentence(s)",
                name,
                checked["n_failed"],
                checked["n"],
                form,
            )
            continue

        arm_result: dict[str, dict[str, dict[str, Any]]] = {}
        for granularity in granularities_for(kind):
            subset_result: dict[str, dict[str, Any]] = {}
            for subset_name in populations:
                words, gold = inputs[(kind, granularity, subset_name)]
                tolerance_result: dict[str, Any] = {}
                for tolerance_name, tolerance in TOLERANCES.items():
                    raw = morphscore(tokenizer, words, gold, tolerance=tolerance)
                    entry: dict[str, Any] = dict(summarise_metric(raw))
                    for extra in (
                        "precision",
                        "recall",
                        "n_matched",
                        "n_token_boundaries",
                        "n_gold_boundaries",
                        "n_excluded_single_token",
                        "n_excluded_single_morpheme",
                        "n_skipped_unaligned",
                        "tolerance",
                    ):
                        entry[extra] = raw[extra]
                    entry["n_units"] = len(words)
                    entry["granularity"] = granularity
                    entry["heuristic"] = granularity == GRANULARITY_STEM
                    tolerance_result[tolerance_name] = entry
                subset_result[subset_name] = tolerance_result
                exact = tolerance_result["exact"]
                logger.info(
                    "%s / %s / %s: F1 %.4f (P %.4f, R %.4f) over %d unit(s)",
                    name,
                    granularity,
                    subset_name,
                    exact["value"],
                    exact["precision"],
                    exact["recall"],
                    exact["n"],
                )
            arm_result[granularity] = subset_result
        results[name] = arm_result
    return results


# ------------------------------------------------- MorphScore paired deltas with a CI


def per_sentence_boundary_counts(
    tokenizer: LoadedTokenizer,
    records: Sequence[GoldRecord],
    kind: str,
    granularity: str,
    tolerance: int,
) -> np.ndarray:
    """`(n_sentences, 3)` of `(matched, token boundaries, gold boundaries)` per sentence.

    MorphScore is a *pooled* F1 — one ratio over the whole corpus, not a mean of per-word
    scores — so a bootstrap over it has to resample the counts and re-pool, not resample the
    F1s. These three per-sentence sums are exactly what re-pooling needs, and they come from
    `morphscore` itself (called once per sentence) rather than from a reimplementation, so
    the resampled statistic and the reported level are the same function of the same inputs
    by construction.

    Sentences are the resampling unit rather than words because words inside one sentence
    are not independent: a formulaic line contributes several words that stand or fall
    together.
    """
    rows = np.zeros((len(records), 3), dtype=np.int64)
    for index, record in enumerate(records):
        words, gold = morphscore_inputs([record], kind, granularity)
        raw = morphscore(tokenizer, words, gold, tolerance=tolerance)
        rows[index] = (
            int(raw["n_matched"]),
            int(raw["n_token_boundaries"]),
            int(raw["n_gold_boundaries"]),
        )
    return rows


def pooled_boundary_f1(totals: np.ndarray) -> tuple[float, float, float]:
    """`(f1, precision, recall)` from a `(matched, token boundaries, gold boundaries)` sum."""
    matched, token_boundaries, gold_boundaries = (float(value) for value in totals)
    precision = matched / token_boundaries if token_boundaries else 0.0
    recall = matched / gold_boundaries if gold_boundaries else 0.0
    total = precision + recall
    return (2 * precision * recall / total if total else 0.0, precision, recall)


def morphscore_paired_delta(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, Any]:
    """`F1(a) − F1(b)` with a sentence-level paired bootstrap interval.

    Paired: each resample draws one set of sentence indices and re-pools *both* arms over
    it, so the sentence-to-sentence variation the two arms share cancels from the delta,
    exactly as in `tpp_paired_delta`. An interval excluding 0 is the claim "the constraint
    changed boundary F1"; the README reports it beside the levels, because a difference of
    +0.02 on 4,600 words and one on 40,000 are not the same evidence.

    Returns the delta, its interval, both arms' F1/precision/recall, and the bookkeeping
    (`n_sentences`, `n_bootstrap`, `seed`, `ci`).
    """
    if counts_a.shape != counts_b.shape:
        raise ValueError(
            f"paired MorphScore counts must cover the same sentences: {counts_a.shape} "
            f"vs {counts_b.shape}"
        )
    f1_a, precision_a, recall_a = pooled_boundary_f1(counts_a.sum(axis=0))
    f1_b, precision_b, recall_b = pooled_boundary_f1(counts_b.sum(axis=0))
    n_sentences = counts_a.shape[0]
    generator = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        chosen = generator.integers(0, n_sentences, n_sentences)
        draws[index] = (
            pooled_boundary_f1(counts_a[chosen].sum(axis=0))[0]
            - pooled_boundary_f1(counts_b[chosen].sum(axis=0))[0]
        )
    tail = (1.0 - ci) / 2.0
    low, high = np.percentile(draws, [100 * tail, 100 * (1 - tail)])
    return {
        "delta": f1_a - f1_b,
        "ci_low": float(low),
        "ci_high": float(high),
        "f1_a": f1_a,
        "f1_b": f1_b,
        "precision_a": precision_a,
        "precision_b": precision_b,
        "recall_a": recall_a,
        "recall_b": recall_b,
        "n_sentences": n_sentences,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci": ci,
        "unit": "F1",
    }


def compute_morphscore_deltas(
    records: Sequence[GoldRecord],
    arms: Mapping[str, LoadedTokenizer],
    contrasts: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """`"<a>/<b>" -> subset -> tolerance -> delta`: the clean MorphScore contrasts, with CIs.

    Only the contrasts that score the **same population of units** are here: `T5 − T1_dcs`
    and `T5seg − T1_dcs` at segment granularity (both raw arms, both scored on the sandhied
    surface's words) and `T6 − T4_oracle` at stem granularity (both split arms, both scored
    on the oracle split's segments). `T4_oracle − T1_dcs` is *not* a MorphScore contrast at
    all and is deliberately absent: a raw arm is scored on surface words and a split arm on
    gold segments, so their difference is a difference of populations.

    A contrast whose arms are not both available, or either of whose arms failed the spans
    coverage check, is skipped with a WARNING rather than reported from one side.

    The per-sentence counts are computed once per `(arm, granularity, tolerance)` over
    *all* records and the human-verified subset is then a row selection, so the two subsets
    cost one pass rather than two.
    """
    verified = np.array([record.human_verified for record in records], dtype=bool)
    cache: dict[tuple[str, str, int], np.ndarray] = {}

    def counts_for(name: str, granularity: str, tolerance: int) -> np.ndarray | None:
        key = (name, granularity, tolerance)
        if key in cache:
            return cache[key]
        tokenizer = arms.get(name)
        if tokenizer is None or not tokenizer.supports_spans:
            return None
        checked = coverage.get(name, {}).get(_coverage_form(arm_text_kind(name)))
        if checked is not None and checked.get("n_failed", 0):
            return None
        cache[key] = per_sentence_boundary_counts(
            tokenizer, records, arm_text_kind(name), granularity, tolerance
        )
        return cache[key]

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for contrast in contrasts:
        first, second = str(contrast["a"]), str(contrast["b"])
        granularity = str(contrast["granularity"])
        check_paired_arms(first, second)
        for name in (first, second):
            if granularity not in granularities_for(arm_text_kind(name)):
                raise ValueError(
                    f"morphscore_deltas entry {first}/{second}: granularity "
                    f"{granularity!r} is not defined for {name!r}"
                )
        subset_result: dict[str, dict[str, Any]] = {}
        for tolerance_name, tolerance in TOLERANCES.items():
            counts_a = counts_for(first, granularity, tolerance)
            counts_b = counts_for(second, granularity, tolerance)
            if counts_a is None or counts_b is None:
                logger.warning(
                    "MorphScore delta %s/%s is skipped: an arm is unavailable or failed "
                    "the spans coverage check",
                    first,
                    second,
                )
                break
            for subset_name, rows in (
                (SUBSET_ALL, slice(None)),
                (SUBSET_HUMAN, verified),
            ):
                entry = morphscore_paired_delta(
                    counts_a[rows], counts_b[rows], n_bootstrap=n_bootstrap, seed=seed, ci=ci
                )
                entry["arm_a"] = first
                entry["arm_b"] = second
                entry["granularity"] = granularity
                entry["heuristic"] = granularity == GRANULARITY_STEM
                entry["tolerance"] = tolerance
                subset_result.setdefault(subset_name, {})[tolerance_name] = entry
                logger.info(
                    "MorphScore delta %s − %s (%s, %s, %s): %+.4f [%+.4f, %+.4f]",
                    first,
                    second,
                    granularity,
                    subset_name,
                    tolerance_name,
                    entry["delta"],
                    entry["ci_low"],
                    entry["ci_high"],
                )
        if subset_result:
            results[pair_key(first, second)] = subset_result
    return results


# ---------------------------------------------------- in-domain compression on held-out


def compute_indomain_compression(
    records: Sequence[GoldRecord],
    arms: Mapping[str, LoadedTokenizer],
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """`arm -> summary`: bytes per token and tokens per raw word on **held-out DCS**.

    Every `_dcs` arm is trained on DCS and evaluated, in the TPP tables, on parallel corpora
    it has never seen. That domain mismatch is identical for both sides of every pair and so
    cannot manufacture a delta — but it can *shrink* one, and a reader is entitled to see
    what the same contrast costs in domain. This is that measurement: the same arms on the
    held-out DCS sentences, on whatever text form each arm tokenizes (the sandhied surface
    for a raw arm, the oracle split for a split one), with the raw sentence's whitespace-word
    count as the common fertility denominator.

    It is not a TPP: DCS has no English side. What it supports is the sentence "the
    constraint costs tokens in domain too", which the cross-corpus TPP deltas alone cannot.
    """
    texts = {
        RAW: [record.text_slp1 for record in records],
        SPLIT: [record.oracle_split_slp1 for record in records],
    }
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        tokenizer = arms.get(name)
        if tokenizer is None:
            continue
        variant = variant_for(name)
        raw = compression(tokenizer, texts[variant])
        entry: dict[str, Any] = dict(summarise_metric(raw))
        entry["variant"] = variant
        entry["n_tokens"] = int(raw["n"])
        entry["n_sentences"] = len(records)
        entry["fertility_vs_raw_words"] = summarise_metric(
            fertility_against_reference(tokenizer, texts[variant], texts[RAW])
            if variant == SPLIT
            else fertility(tokenizer, texts[RAW])
        )["value"]
        # `variant` here means the *text form* the arm tokenizes, as in the `compression`
        # block, so the arm's training-corpus label is carried under its own key rather
        # than colliding with it.
        labels = arm_labels(name)
        entry["corpus_variant"] = labels["variant"]
        entry["oracle"] = labels["oracle"]
        results[name] = entry
        logger.info(
            "%s: in-domain held-out DCS %.4f bytes/token over %d token(s), "
            "%.4f tokens/raw word",
            name,
            entry["value"],
            entry["n_tokens"],
            entry["fertility_vs_raw_words"],
        )
    return results


# ----------------------------------------------------- the out-of-sample violation audit


def write_marked_corpus(records: Sequence[GoldRecord], field: str, path: Path) -> Path:
    """Write one marked line per record, for the cross-boundary audit to read back.

    `assert_no_cross_boundary_merges` takes a corpus *file* — it is the same function the
    training script runs in-sample — so the held-out marked text has to be materialised. The
    file goes under the experiment's output directory, which is gitignored.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(str(getattr(record, field)) + "\n")
    return path


def violation_report(tokenizer_json: Path, marked_corpus: Path, sample: int) -> dict[str, Any]:
    """`assert_no_cross_boundary_merges` plus the per-**boundary** rate the README leads with.

    The shared function's own `violation_rate` is per *token*, which is the wrong
    denominator for comparing two arms: a constrained arm emits more tokens for the same
    text, so dividing by tokens flatters it twice — once for respecting the boundaries and
    once for being more verbose. `violations_per_boundary` divides by the gold boundaries in
    the audited text instead, which is the same denominator for both arms of a pair.

    Its numerator is offending *tokens*, not offending boundaries: a token spanning two gold
    boundaries counts once. The two coincide unless a single token swallows a whole
    morpheme, so the number is a close lower bound on the fraction of boundaries crossed,
    and the README says so rather than calling it a fraction.
    """
    report = dict(assert_no_cross_boundary_merges(tokenizer_json, marked_corpus, sample))
    boundaries = int(report["n_boundaries"])
    report["violations_per_boundary"] = (
        int(report["n_violations"]) / boundaries if boundaries else math.nan
    )
    return report


def compute_violations(
    arms: Mapping[str, LoadedTokenizer],
    violation_arms: Mapping[str, Sequence[str]],
    marked_corpora: Mapping[str, Path],
    sample: int,
) -> dict[str, dict[str, Any]]:
    """`arm -> report`: how often each arm's tokens span a gold boundary on *unseen* text.

    This is the mechanism the whole design rests on, and it is measured out of sample on
    purpose. The marker constrains *learning*: no merge rule is counted from a pair
    straddling a boundary, but a rule learned elsewhere still applies inside a marked word
    at inference, where the text carries no markers (docs/decisions.md, "Twelve DCS arms
    trained; the boundary-marker constraint halves cross-boundary tokens but cannot reach
    zero"). The training-time check in each arm's own `results.json` is in-sample; this one
    is on sentences no arm saw.

    An arm whose `tokenizer.json` is not on disk — an off-the-shelf arm, or a fake in a test
    — is recorded with a `skipped` reason rather than dropped, so the block says why a row
    is missing.
    """
    reports: dict[str, dict[str, Any]] = {}
    for kind, names in violation_arms.items():
        corpus = marked_corpora[kind]
        for name in names:
            tokenizer = arms.get(name)
            if tokenizer is None:
                logger.warning("%s: unavailable this run; no violation audit", name)
                reports[name] = {"skipped": "arm unavailable this run", "kind": kind}
                continue
            path = Path(tokenizer.source_id)
            if not path.is_file():
                logger.warning(
                    "%s: source %s is not a tokenizer.json on disk; no violation audit",
                    name,
                    tokenizer.source_id,
                )
                reports[name] = {
                    "skipped": f"source_id {tokenizer.source_id!r} is not a file",
                    "kind": kind,
                }
                continue
            report = violation_report(path, corpus, sample)
            report["kind"] = kind
            report["arm"] = name
            reports[name] = report
            logger.info(
                "%s: %d violating token(s) over %d gold boundary/boundaries "
                "(%.4f per boundary) on held-out %s text",
                name,
                report["n_violations"],
                report["n_boundaries"],
                report["violations_per_boundary"],
                kind,
            )
    return reports


# ---------------------------------------------------------------------- corpus wrangling


@dataclass(frozen=True)
class CorpusData:
    """One parallel corpus after loading, blank-index filtering and split-cache lookup."""

    name: str
    split: str
    n_total: int
    n_used: int
    raw_deva: list[str]
    texts: dict[str, list[str]]
    english: list[str]


def read_split_records(name: str, path: Path) -> list[dict[str, Any]]:
    """Every record of one corpus's split jsonl, in file order.

    A missing file aborts naming the script that writes it: falling back to raw text would
    silently measure a split arm on unsplit sentences, which produces a plausible number
    that means nothing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{name}: no split cache at {path}. Run "
            "`uv run python experiments/03_sandhi_split/split_corpora.py` first "
            "(it is resumable: everything already split is a cache hit)."
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def split_texts_for(
    name: str, records: Sequence[Mapping[str, Any]], raw_deva: Sequence[str]
) -> list[str]:
    """The reconciled split text of one corpus, aligned to `raw_deva` by the jsonl's `index`.

    Aborts — never falls back to raw text — on a missing sentence, a record count that does
    not match the evaluation set, or a cached `raw_deva` that is not the sentence at that
    index. Each is a different way the cache can be stale, and each would otherwise be
    invisible in the output.
    """
    by_index = {int(record["index"]): record for record in records}
    total = len(raw_deva)
    missing = [index for index in range(total) if index not in by_index]
    if missing:
        raise ValueError(
            f"{name}: {len(missing)} of {total} evaluation sentence(s) are missing from the "
            f"split cache (first missing index {missing[0]}); re-run "
            "experiments/03_sandhi_split/split_corpora.py."
        )
    if len(records) != total:
        raise ValueError(
            f"{name}: the split cache holds {len(records)} record(s) for {total} evaluation "
            "sentence(s); it was built from a different version of this corpus."
        )
    mismatched = [
        index
        for index in range(total)
        if str(by_index[index].get("raw_deva", "")).strip() != raw_deva[index].strip()
    ]
    if mismatched:
        raise ValueError(
            f"{name}: {len(mismatched)} of {total} cached sentence(s) do not match the corpus "
            f"at the same index (first at index {mismatched[0]}); the split cache is stale."
        )
    return [str(by_index[index]["output"]) for index in range(total)]


def prepare_corpus(entry: Mapping[str, Any], root: Path, split_dir: Path) -> CorpusData:
    """Load one `config["corpora"]` entry, filter it, and attach its cached split text."""
    name = str(entry["name"])
    corpus = load_corpus_entry(entry, root)
    sentences = {language: list(corpus.sentences[language]) for language in corpus.languages}
    total = len(corpus)
    indices = select_aligned_indices(sentences)
    if len(indices) < total:
        logger.warning(
            "%s: dropping %d/%d indices blank in at least one language",
            name,
            total - len(indices),
            total,
        )
    filtered = take_indices(sentences, indices)
    raw_deva = filtered[SANSKRIT_LANGUAGE]
    records = read_split_records(name, split_dir / f"{name}.jsonl")
    texts = {
        RAW: [to_slp1(text, "devanagari") for text in raw_deva],
        SPLIT: split_texts_for(name, records, raw_deva),
    }
    logger.info("%s: using %d/%d aligned sentences", name, len(indices), total)
    return CorpusData(
        name=name,
        split=str(entry["split"]),
        n_total=total,
        n_used=len(indices),
        raw_deva=raw_deva,
        texts=texts,
        english=filtered[ENGLISH_LANGUAGE],
    )


# ------------------------------------------------------------------------- TPP / metrics


def variant_for(name: str) -> str:
    """Which text variant an arm is measured on: `RAW` for a raw arm, `SPLIT` for a split one."""
    return SPLIT if arm_text_kind(name) == SPLIT_ARM else RAW


def compute_tpp(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    names: Sequence[str],
    pivots: Mapping[str, str],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """`corpus -> arm -> pivot -> summary`: TPP levels, reported as context only.

    Neither pivot is a matched control for a DCS-trained arm — `E1_bpe_64k` was trained on
    the English side of the parallel corpora and `T0_o200k` is a 200k general-domain
    vocabulary — so every entry records its `role` and the E1 entries additionally record
    `cross_corpus: true`. The verdict is read from `compute_tpp_delta`, where the English
    side cancels.
    """
    for pivot_name in sorted(set(pivots.values())):
        if pivot_name not in arms:
            logger.warning(
                "English pivot %s is unavailable this run; every TPP column against it is "
                "omitted from results.json",
                pivot_name,
            )
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        corpus_result: dict[str, dict[str, Any]] = {}
        for name in names:
            tokenizer = arms.get(name)
            if tokenizer is None:
                continue
            variant = variant_for(name)
            pivot_result: dict[str, Any] = {}
            for role, pivot_name in pivots.items():
                pivot_tokenizer = arms.get(pivot_name)
                if pivot_tokenizer is None:
                    continue
                raw = tpp(
                    tokenizer,
                    corpus.texts[variant],
                    corpus.english,
                    pivot_tokenizer,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    ci=ci,
                )
                summary = summarise_tpp(raw, ci=ci)
                summary["role"] = role
                summary["variant"] = variant
                summary["cross_corpus"] = role == ROLE_CONTROLLED
                summary.update(arm_labels(name))
                pivot_result[pivot_name] = summary
                logger.info(
                    "%s / %s / vs %s (%s): TPP %.3f [%.3f, %.3f]",
                    corpus.name,
                    name,
                    pivot_name,
                    role,
                    raw["value"],
                    raw["ci_low"],
                    raw["ci_high"],
                )
            corpus_result[name] = pivot_result
        results[corpus.name] = corpus_result
    return results


def compute_tpp_delta(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    pairs: Sequence[tuple[str, str]],
    pivot_name: str,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """`corpus -> "<a>/<b>" -> delta summary`: the experiment's headline.

    `TPP(a) − TPP(b)`, both against the same English side over the same sentences, with a
    paired bootstrap. The English total is identical on both sides and therefore cancels
    from the *sign* of the delta entirely, which is why a cross-corpus pivot can carry a
    controlled comparison here when it could not carry a level.

    **The deletion cost applies to exactly one kind of pair.** A `T5−T1` or `T6−T4` pair
    tokenizes the *same* text on both sides — only the vocabulary differs — so there is
    nothing to price. A `T4−T1` or `T6−T1` pair does not: the split side is measured on
    ByT5-reconciled text, which does not preserve every non-letter character, and an arm
    handed less text looks cheaper for a reason that is not tokenization
    (docs/decisions.md, "Reconciliation must preserve every non-letter character"). Those
    pairs carry `deletion_cost`; the others carry `null`, which reads as "not applicable",
    not as "zero".
    """
    pivot = arms.get(pivot_name)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        pair_results: dict[str, dict[str, Any]] = {}
        if pivot is None:
            logger.warning(
                "%s: every paired delta is skipped: pivot %s unavailable this run",
                corpus.name,
                pivot_name,
            )
            results[corpus.name] = pair_results
            continue
        for first, second in pairs:
            variant_a, variant_b = variant_for(first), variant_for(second)
            counts_a = token_ratio(arms[first], corpus.texts[variant_a], corpus.english, pivot)
            counts_b = token_ratio(arms[second], corpus.texts[variant_b], corpus.english, pivot)
            entry: dict[str, Any] = dict(
                tpp_paired_delta(counts_a, counts_b, n_bootstrap=n_bootstrap, seed=seed, ci=ci)
            )
            entry["arm_a"] = first
            entry["arm_b"] = second
            entry["variant_a"] = variant_a
            entry["variant_b"] = variant_b
            entry["pivot"] = pivot_name
            entry["pivot_cross_corpus"] = True
            entry["label"] = pair_label(first, second)
            saving = counts_b.source_total - counts_a.source_total
            entry["source_tokens_saved"] = saving
            if variant_a == variant_b:
                entry["deletion_cost"] = None
                entry["deletion_cost_tokens"] = None
                entry["deletion_cost_share_of_saving"] = None
            else:
                raw_texts = corpus.texts[RAW]
                out_texts = corpus.texts[SPLIT]
                reference_arm = arms[second] if variant_b == RAW else arms[first]
                cost = deletion_cost(
                    reference_arm, raw_texts, out_texts, source_tokens_saved=saving
                )
                entry["deletion_cost"] = cost
                entry["deletion_cost_tokens"] = cost["deletion_cost_tokens"]
                entry["deletion_cost_share_of_saving"] = cost["share_of_saving"]
            pair_results[pair_key(first, second)] = entry
            logger.info(
                "%s / %s: delta %+.4f [%+.4f, %+.4f] (a %.3f, b %.3f, vs %s)",
                corpus.name,
                pair_key(first, second),
                float(entry["delta"]),
                float(entry["ci_low"]),
                float(entry["ci_high"]),
                float(entry["value_a"]),
                float(entry["value_b"]),
                pivot_name,
            )
        results[corpus.name] = pair_results
    return results


def compute_fertility_compression(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    names: Sequence[str],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """`(fertility_primary, fertility_secondary, compression)`, each `corpus -> arm -> summary`.

    Reported, never headlined (CLAUDE.md §2.1). **Primary** puts every arm over the same
    denominator — the raw sentence's whitespace-word count — so a split arm's number stays
    comparable with a raw arm's; **secondary** is plain fertility on the split text, which
    has a different denominator and exists only to show what the primary is correcting for.
    Only the pooled `value` means the same thing across both tables: the two metrics attach
    different distributions, so `mean` and `std` describe per-word token counts for a raw arm
    and per-sentence ratios for a split arm. Compression is measured on whatever text the arm
    actually tokenizes, and every entry records that `variant`.
    """
    primary: dict[str, dict[str, dict[str, Any]]] = {}
    secondary: dict[str, dict[str, dict[str, Any]]] = {}
    comp: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        primary[corpus.name] = {}
        secondary[corpus.name] = {}
        comp[corpus.name] = {}
        for name in names:
            tokenizer = arms.get(name)
            if tokenizer is None:
                continue
            variant = variant_for(name)
            texts = corpus.texts[variant]
            if variant == SPLIT:
                primary_result = fertility_against_reference(tokenizer, texts, corpus.texts[RAW])
                secondary[corpus.name][name] = {
                    **summarise_metric(fertility(tokenizer, texts)),
                    "variant": variant,
                    "n_texts": len(texts),
                }
            else:
                primary_result = fertility(tokenizer, texts)
            primary[corpus.name][name] = {
                **summarise_metric(primary_result),
                "variant": variant,
                "n_texts": len(texts),
                **arm_labels(name),
            }
            comp[corpus.name][name] = {
                **summarise_metric(compression(tokenizer, texts)),
                "variant": variant,
            }
    return primary, secondary, comp


# ------------------------------------------------------------------------ provenance


def arm_sources(arms: Mapping[str, LoadedTokenizer]) -> dict[str, dict[str, Any]]:
    """`tokenizer_sources` with `variant` added and every file-backed arm actually hashed.

    Two gaps in the shared helper, both filled here rather than there because a change to
    `sanskrit_tok.experiment` would alter Experiments 01–03's output for no reason. Its
    `FILE_BACKED_FAMILIES` predates the constrained families, so a `T5`/`T6` arm — a
    `tokenizer.json` this project trained, whose only tie to a number in this file is its
    hash — would otherwise carry none; and `variant` is what separates
    `T4_bpe_split_64k_oracle_dcs` from `T4_bpe_split_64k`, which are the same algorithm at
    the same vocabulary size trained on different text.
    """
    sources = tokenizer_sources(arms)
    for name, entry in sources.items():
        entry["variant"] = arms[name].variant
        entry.update(arm_labels(name))
        if "sha256" not in entry:
            path = Path(arms[name].source_id)
            if path.is_file():
                try:
                    entry["sha256"] = tokenizer_file_sha256(path)
                except OSError as error:  # pragma: no cover - a race, not a code path
                    logger.warning("%s: could not hash %s (%s)", name, path, error)
    return sources


# ------------------------------------------------------------------------------ figure


def _delta_series(
    corpus_delta: Mapping[str, Any], pairs: Sequence[tuple[str, str]]
) -> tuple[list[str], list[float], tuple[list[float], list[float]]]:
    """`(labels, values, (lower errors, upper errors))` for one corpus's paired deltas."""
    labels: list[str] = []
    values: list[float] = []
    errors: tuple[list[float], list[float]] = ([], [])
    for first, second in pairs:
        entry = corpus_delta.get(pair_key(first, second))
        if entry is None:
            continue
        labels.append(str(entry.get("label", pair_label(first, second))))
        value = entry.get("delta")
        value = math.nan if value is None else float(value)
        values.append(value)
        low, high = entry.get("ci_low"), entry.get("ci_high")
        errors[0].append(value - float(low) if low is not None and math.isfinite(value) else 0.0)
        errors[1].append(float(high) - value if high is not None and math.isfinite(value) else 0.0)
    return labels, values, errors


def _morphscore_series(
    results: Mapping[str, Any], names: Sequence[str]
) -> tuple[list[str], list[float], list[float]]:
    """`(labels, exact F1, ±1 F1)` for the human-verified subset, one entry per scored arm.

    Each arm is shown at its *primary* granularity — segments for a raw arm, stems for a
    split arm — so the bar answers "how well do this arm's boundaries land on the best gold
    boundaries it has", not "how do two granularities compare". The caption says the two
    columns are not one series.
    """
    labels: list[str] = []
    exact: list[float] = []
    tolerant: list[float] = []
    for name in names:
        block = results.get("morphscore", {}).get(name)
        if not block:
            continue
        granularity = granularities_for(arm_text_kind(name))[0]
        entry = block.get(granularity, {}).get(SUBSET_HUMAN)
        if entry is None:
            continue
        star = "*" if arm_labels(name)["provisional"] else ""
        labels.append(f"{name}{star}")
        for key, target in (("exact", exact), ("pm1", tolerant)):
            value = entry.get(key, {}).get("value")
            target.append(math.nan if value is None else float(value))
    return labels, exact, tolerant


def _build_figure(results: Mapping[str, Any]) -> Any:
    """Build (but do not save or close) the Experiment 04 figure; returns the `Figure`.

    Left column, one row per parallel corpus in config order (prose first, CLAUDE.md §2.7):
    the eight paired TPP deltas with their 95% paired-bootstrap intervals against a dashed
    line at 0. Zero is the null here, not English parity: the question is whether the
    constrained arm costs fewer tokens than the arm it is matched against, and an interval
    that straddles 0 answers "no distinguishable difference".

    Right column, spanning every row: MorphScore F1 on the human-verified held-out subset,
    one bar per scored arm, each at its primary granularity, with a thin marker at the
    ±1-character tolerant value. The two panels answer different questions and share no axis
    on purpose — an arm can raise MorphScore and cost tokens, which is exactly the outcome
    the README has to be able to report.

    Split out from `make_figure` so tests can inspect the constructed `Figure` — its tick
    labels, its caption — before anything is written to disk.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on CI and over ssh
    import matplotlib.pyplot as plt

    config = results["config"]
    corpus_entries = list(config["corpora"])
    if not corpus_entries:
        raise ValueError("results['config']['corpora'] is empty; nothing to plot")
    pairs = [(str(pair[0]), str(pair[1])) for pair in config["paired_deltas"]]
    arm_names = [str(name) for name in config["arms_morphscore"]]

    rows = len(corpus_entries)
    figure = plt.figure(figsize=(13.0, max(2.6 * rows, 5.0)))
    grid = figure.add_gridspec(rows, 2, width_ratios=[1.15, 1.0])

    for index, entry in enumerate(corpus_entries):
        axes = figure.add_subplot(grid[index, 0])
        corpus_name = str(entry["name"])
        labels, values, errors = _delta_series(
            results.get("tpp_delta", {}).get(corpus_name, {}), pairs
        )
        positions = np.arange(len(labels), dtype=float)
        axes.errorbar(
            positions,
            values,
            yerr=list(errors),
            fmt="o",
            capsize=3,
            color="#2b6cb0",
            zorder=3,
        )
        axes.axhline(0.0, linestyle="--", color="gray", linewidth=1)
        finite = [
            bound
            for value, low, high in zip(values, errors[0], errors[1], strict=True)
            for bound in (value - low, value + high)
            if math.isfinite(value) and math.isfinite(value - low) and math.isfinite(value + high)
        ]
        if finite:
            span_low, span_high = min([*finite, 0.0]), max([*finite, 0.0])
            span = span_high - span_low
            pad = span * FIGURE_Y_PAD_FRACTION if span > 0 else 0.05
            axes.set_ylim(span_low - pad, span_high + pad)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        axes.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
        axes.set_ylabel("Δ TPP (a − b)", fontsize=8)
        axes.set_title(corpus_name, fontsize=9, loc="left")
        axes.spines[["top", "right"]].set_visible(False)

    panel = figure.add_subplot(grid[:, 1])
    labels, exact, tolerant = _morphscore_series(results, arm_names)
    positions = np.arange(len(labels), dtype=float)
    panel.barh(positions, exact, color="#c05621", height=0.6, label="exact", zorder=2)
    panel.plot(
        tolerant,
        positions,
        linestyle="none",
        marker="|",
        markersize=10,
        color="#1a202c",
        label="±1 character",
        zorder=3,
    )
    panel.set_yticks(positions)
    panel.set_yticklabels(labels, fontsize=6)
    panel.invert_yaxis()
    panel.set_xlabel("MorphScore F1, human-verified held-out DCS", fontsize=8)
    panel.set_title(
        "raw arms: gold segment boundaries · split arms: heuristic stem boundaries",
        fontsize=8,
        loc="left",
    )
    panel.spines[["top", "right"]].set_visible(False)
    if labels:
        panel.legend(fontsize=7, loc="lower right")

    caption = FIGURE_CAPTION
    omitted = unavailable_caption(results.get("unavailable_arms", {}))
    if omitted:
        caption = f"{caption}  {omitted}"
    figure.suptitle(FIGURE_SUPTITLE, fontsize=11)
    figure.text(0.5, 0.955, FIGURE_SUBTITLE, ha="center", fontsize=7)
    figure.text(0.01, 0.005, caption, fontsize=5.5, wrap=True)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    return figure


def make_figure(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Build the Experiment 04 figure (`_build_figure`) and save it as PDF and PNG."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figure = _build_figure(results)
    paths = [out_dir / f"{FIGURE_STEM}.pdf", out_dir / f"{FIGURE_STEM}.png"]
    for path in paths:
        figure.savefig(path, dpi=200)
    plt.close(figure)
    logger.info("wrote %s", " and ".join(str(path) for path in paths))
    return paths


# -------------------------------------------------------------------------------- main


def _all_arm_names(config: Mapping[str, Any]) -> list[str]:
    """Every arm this run touches, deduplicated: MorphScore, TPP, the pairs and the pivots."""
    names: set[str] = set()
    names.update(str(name) for name in config["arms_morphscore"])
    names.update(str(name) for name in config["arms_tpp"])
    for pair in config["paired_deltas"]:
        names.update(str(name) for name in pair)
    for contrast in config.get("morphscore_deltas", []):
        names.update({str(contrast["a"]), str(contrast["b"])})
    names.update(str(name) for name in config.get("arms_indomain", []))
    for group in config["violation_arms"].values():
        names.update(str(name) for name in group)
    names.update(str(name) for name in config["english_pivots"].values())
    return sorted(names)


def run(config: Mapping[str, Any], config_src: Path | None = None) -> dict[str, Any]:
    """Run the whole experiment from a parsed config; returns the results dict.

    Split out from `main` so the end-to-end path — the held-out split, the leakage checks,
    every metric, `results.json`, the figure — is exercisable from a test with a temporary
    config, fake tokenizers and synthetic data, offline.
    """
    root = repo_root()
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    n_bootstrap = int(config["n_bootstrap"])
    ci = float(config.get("ci", 0.95))

    corpora_config = list(config["corpora"])
    if not corpora_config:
        raise ValueError("config['corpora'] is empty; nothing to measure")
    arms_morphscore = [str(name) for name in config["arms_morphscore"]]
    arms_tpp = [str(name) for name in config["arms_tpp"]]
    paired_deltas = [list(pair) for pair in config["paired_deltas"]]
    violation_arms = {
        str(kind): [str(name) for name in names]
        for kind, names in config["violation_arms"].items()
    }
    pivots = {str(role): str(name) for role, name in config["english_pivots"].items()}

    heldout_path = resolve_path(str(config["dcs_heldout"]), root)
    split_dir = resolve_path(str(config["split_dir"]), root)
    out_dir = resolve_path(str(config["output_dir"]), root)
    exclusion_path = resolve_path(str(config["exclusion_path"]), root)
    exclusion_path_en = resolve_path(str(config["exclusion_path_en"]), root)

    all_records = read_gold_records(heldout_path)
    records, sample_size = sample_records(
        all_records, config.get("morphscore_sample"), seed
    )
    logger.info(
        "DCS held-out: %d sentence(s), %d human-verified; MorphScore over %d",
        len(all_records),
        sum(1 for record in all_records if record.human_verified),
        len(records),
    )

    corpora = [prepare_corpus(entry, root, split_dir) for entry in corpora_config]

    hashes = load_exclusion_hashes(exclusion_path)
    hashes_en = load_exclusion_hashes(exclusion_path_en)
    heldout_check = exclusion_check_for(
        [record.text_slp1 for record in all_records], hashes, sentence_hash_slp1
    )
    if heldout_check["n_missing"]:
        logger.warning(
            "%d/%d DCS held-out sentences are NOT in the exclusion list; the `_dcs` arms "
            "may have been trained on evaluation text (data/exclusion_hashes.txt is stale)",
            heldout_check["n_missing"],
            heldout_check["n"],
        )
    exclusion_check: dict[str, dict[str, int]] = {}
    exclusion_check_en: dict[str, dict[str, int]] = {}
    for corpus in corpora:
        report = exclusion_check_for(corpus.raw_deva, hashes)
        exclusion_check[corpus.name] = report
        if report["n_missing"]:
            logger.warning(
                "%s: %d/%d Sanskrit sentences are NOT in the exclusion list",
                corpus.name,
                report["n_missing"],
                report["n"],
            )
        report_en = exclusion_check_for(corpus.english, hashes_en, sentence_hash_en)
        exclusion_check_en[corpus.name] = report_en
        if report_en["n_missing"]:
            logger.warning(
                "%s: %d/%d English sentences are NOT in the English exclusion list; the E1 "
                "pivot could have trained on evaluation text",
                corpus.name,
                report_en["n_missing"],
                report_en["n"],
            )

    arms, unavailable_arms = load_arms(_all_arm_names(config))

    coverage = check_spans_coverage(
        records, arms, arms_morphscore, int(config.get("spans_coverage_sentences", 500))
    )
    morphscore_results = compute_morphscore(records, arms, arms_morphscore, coverage)
    morphscore_delta = compute_morphscore_deltas(
        records,
        arms,
        [dict(contrast) for contrast in config.get("morphscore_deltas", [])],
        coverage,
        n_bootstrap=int(config.get("morphscore_bootstrap", n_bootstrap)),
        seed=int(config.get("morphscore_bootstrap_seed", seed)),
        ci=ci,
    )
    indomain_compression = compute_indomain_compression(
        records, arms, [str(name) for name in config.get("arms_indomain", [])]
    )

    marked_corpora = {
        RAW_ARM: write_marked_corpus(
            all_records, "t5_marked", out_dir / "marked" / "heldout_t5_marked.txt"
        ),
        SPLIT_ARM: write_marked_corpus(
            all_records, "t6_marked", out_dir / "marked" / "heldout_t6_marked.txt"
        ),
    }
    violations = compute_violations(
        arms, violation_arms, marked_corpora, int(config.get("violation_sample", 2000))
    )

    tpp_results = compute_tpp(corpora, arms, arms_tpp, pivots, n_bootstrap, seed, ci)
    tpp_delta = compute_tpp_delta(
        corpora,
        arms,
        select_pairs(paired_deltas, arms),
        pivots[ROLE_CONTROLLED],
        n_bootstrap,
        seed,
        ci,
    )
    fertility_primary, fertility_secondary, compression_results = compute_fertility_compression(
        corpora, arms, arms_tpp
    )

    invariants: dict[str, Any] = {
        "dcs_heldout_oracle_vs_raw": text_invariants(
            [record.text_slp1 for record in all_records],
            [record.oracle_split_slp1 for record in all_records],
        )
    }
    for corpus in corpora:
        invariants[f"{corpus.name}_reconciled_vs_raw"] = text_invariants(
            corpus.texts[RAW], corpus.texts[SPLIT]
        )

    results: dict[str, Any] = {
        "experiment": str(config.get("experiment", out_dir.name)),
        **provenance(root),
        "config": dict(config),
        "tokenizer_sources": arm_sources(arms),
        "unavailable_arms": unavailable_arms,
        "dcs_heldout": {
            "path": str(heldout_path),
            "n": len(all_records),
            "n_human_verified": sum(1 for record in all_records if record.human_verified),
            "n_scored": len(records),
            "sample": sample_size,
            "sample_seed": seed if sample_size is not None else None,
            "n_words": sum(record.n_words for record in all_records),
            "n_words_aligned": sum(record.n_words_aligned for record in all_records),
            "n_segments": sum(
                len(record.oracle_split_slp1.split(" ")) for record in all_records
            ),
            "exclusion_check": heldout_check,
        },
        "corpora": {
            corpus.name: {
                "split": corpus.split,
                "n_total": corpus.n_total,
                "n_used": corpus.n_used,
            }
            for corpus in corpora
        },
        "exclusion_check": exclusion_check,
        "exclusion_check_en": exclusion_check_en,
        "spans_coverage": coverage,
        "morphscore": morphscore_results,
        "morphscore_delta": morphscore_delta,
        "compression_indomain": indomain_compression,
        "violations": violations,
        "tpp": tpp_results,
        "tpp_delta": tpp_delta,
        "fertility_primary": fertility_primary,
        "fertility_secondary": fertility_secondary,
        "compression": compression_results,
        "text_invariants": invariants,
    }

    write_results(results, out_dir, config_src)
    make_figure(results, out_dir)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
        help="experiment config YAML (default: the config.yaml beside this script)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    run(load_config(args.config), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
