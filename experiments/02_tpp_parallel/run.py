"""Experiment 02 — tokens-per-proposition on parallel text (RQ2).

Hypothesis (outline §1, H2): with a Sanskrit-native tokenizer, tokens-per-proposition
(TPP) on parallel corpora is lower for Sanskrit than English; with English-centric
tokenizers it is higher; the sign flips depending on tokenizer.

TPP is the headline metric of this project (CLAUDE.md §7): fertility asks how many
tokens a *word* costs, which punishes Sanskrit for the very density under study (sandhi,
compounding); TPP asks how many tokens the *same proposition* costs on aligned
translation pairs. This script computes TPP, with a paired bootstrap CI, for every
Sanskrit tokenizer arm against English on four corpora, prose before verse (CLAUDE.md
§2.7): Sāmayik test and test_ood (primary, prose), Itihāsa test (secondary, verse — meter
is a confound), FLORES devtest (tertiary). It also computes a Hindi-pivot TPP on FLORES
for the T0/T3 arms, and fertility/compression on the Sanskrit side of every corpus x arm
x script variant, reported but never headlined.

Two English sides, and they answer different questions:

* `tpp_controlled` — each trained Sanskrit arm against the `E1_*` arm that matches it on
  algorithm, vocabulary size and training corpus (`config["controlled_pairs"]`). This is
  the **controlled** comparison: algorithm, vocabulary size and domain fit are held equal
  on both sides, which is what H2 needs (docs/decisions.md, "Add a matched English
  control family E1"). Not everything is controlled even here — corpus size and diversity
  differ between the two languages' sides, and the English text is a translation.
* `tpp` — every arm against `T0_o200k` (primary) and `T0_llama4` (secondary), 200k-plus
  general-domain vocabularies. This is **deployed practice**: what today's tokenizers
  charge for Sanskrit, never a matched comparison (CLAUDE.md §2.5).

Run it with `uv run python experiments/02_tpp_parallel/run.py`. Relative paths in the
config are resolved against the repository root, so the working directory does not
matter. Every function below is pure or takes its I/O paths explicitly, so
`tests/test_exp02.py` can exercise the aggregation and plotting logic on synthetic data
without touching the network, a real corpus, or the Hugging Face cache.
"""

import argparse
import bisect
import logging
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash_en
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.experiment import (
    ENGLISH_LANGUAGE,
    HINDI_LANGUAGE,
    SANSKRIT_LANGUAGE,
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
    tokenizer_sources,
    unavailable_caption,
    write_results,
)
from sanskrit_tok.metrics._ratio import token_ratio
from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility
from sanskrit_tok.metrics.renyi import renyi_efficiency
from sanskrit_tok.metrics.summary import summarise_metric
from sanskrit_tok.metrics.tpp import tpp, tpp_from_parts
from sanskrit_tok.tokenizers.registry import LoadedTokenizer

logger = logging.getLogger("exp02")

#: Script variant names, matching exp01's constants.
ORIGINAL = "original"
SLP1 = "slp1"

#: Tokenizer-arm families whose T0/T3-style provisional flag the figure and README mark
#: with a `*`: trained from scratch on the parallel-corpus training splits, not yet on
#: the monolingual corpus (docs/decisions.md, "Provisional T1/T2 tokenizers...").
PROVISIONAL_FAMILIES = frozenset({"T1", "T2"})

#: Tokenizer-arm families eligible for the Hindi pivot (CLAUDE.md §7 resolution 4:
#: T0/T3 arms only, same tokenizer both sides).
HINDI_PIVOT_FAMILIES = frozenset({"T0", "T3"})

#: The script variant the controlled (T1/T2 vs E1) comparison reads. T1/T2 have no other
#: variant, and the English side is always the text as written.
CONTROLLED_VARIANT = SLP1

FIGURE_STEM = "tpp_by_arm"
#: The pivot and script variant the figure's main marker reads, so every arm — T0/T3
#: (which also have an `original` variant) and T1/T2 (slp1 only) — sits on the same
#: footing (config.yaml resolution 6).
FIGURE_PIVOT = "T0_o200k"
FIGURE_MAIN_VARIANT = SLP1
FIGURE_SECONDARY_VARIANT = ORIGINAL
#: The figure now carries two columns measured against *different* English sides, so the
#: suptitle names neither; each column header names its own pivot.
FIGURE_SUPTITLE = "Tokens per proposition (Sanskrit / English), 95% bootstrap CI"
#: Header over the figure's left-hand column, which holds the deployed-practice pivot.
FIGURE_DEPLOYED_TITLE = "Deployed practice: every arm vs English o200k (200k, general domain)"
#: Header over the figure's right-hand column, which holds the controlled comparison.
FIGURE_CONTROLLED_TITLE = (
    "Matched control: Sanskrit T1/T2 vs English E1 (same algorithm, vocab, training corpus)"
)
#: Static half of the caption; the omitted-arms half is built at plot time from
#: `results["unavailable_arms"]` (`unavailable_caption`) since it depends on the run.
FIGURE_CAPTION_PROVISIONAL = "* provisional: trained on parallel-corpus training splits"
#: Fraction of the SLP1-series y-range added above and below as headroom, and how far
#: inside the top/bottom edge an off-scale (clipped) secondary marker is drawn.
FIGURE_Y_PAD_FRACTION = 0.15
FIGURE_CLIP_INSET_FRACTION = 0.04

# ---------------------------------------------------------------------- corpus wrangling


@dataclass(frozen=True)
class CorpusData:
    """One corpus after loading and blank-index filtering, ready for the metrics.

    `sanskrit`/`hindi` hold both script variants (`hindi` is `None` for the two-language
    Sāmayik/Itihāsa corpora, which carry no Devanagari pivot); `english` is Latin-script
    only, since English has no SLP1 form to convert to.
    """

    name: str
    split: str
    n_total: int
    n_used: int
    sanskrit: dict[str, list[str]]
    english: list[str]
    hindi: dict[str, list[str]] | None


def prepare_corpus(entry: Mapping[str, Any], root: Path) -> CorpusData:
    """Load one `config["corpora"]` entry and filter it to aligned, non-blank indices."""
    name = str(entry["name"])
    split = str(entry["split"])
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

    sanskrit_original = filtered[SANSKRIT_LANGUAGE]
    sanskrit = {
        ORIGINAL: sanskrit_original,
        SLP1: [to_slp1(text, "devanagari") for text in sanskrit_original],
    }
    english = filtered[ENGLISH_LANGUAGE]
    hindi: dict[str, list[str]] | None = None
    if HINDI_LANGUAGE in filtered:
        hindi_original = filtered[HINDI_LANGUAGE]
        hindi = {
            ORIGINAL: hindi_original,
            SLP1: [to_slp1(text, "devanagari") for text in hindi_original],
        }

    logger.info("%s: using %d/%d aligned sentences", name, len(indices), total)
    return CorpusData(
        name=name,
        split=split,
        n_total=total,
        n_used=len(indices),
        sanskrit=sanskrit,
        english=english,
        hindi=hindi,
    )


# --------------------------------------------------------------------------- leakage


# ------------------------------------------------------------------------ tokenizers


def variants_for_family(family: str, script_variants: Mapping[str, Sequence[str]]) -> list[str]:
    """The script variants an arm's family is measured in, per `config["script_variants"]`.

    Raises `ValueError` naming the family and the configured keys when the family is not
    in `script_variants`. A bare `KeyError: 'T5'` from the middle of a two-minute run says
    nothing about which config key is short a family; measuring the arm in no variants at
    all — the other tempting fallback — would silently drop it from every table instead.
    """
    try:
        variants = script_variants[family]
    except KeyError:
        raise ValueError(
            f"unknown tokenizer family {family!r}: config['script_variants'] has "
            f"{sorted(script_variants)}"
        ) from None
    return list(variants)


# ------------------------------------------------------------------------- TPP / metrics


def compute_tpp(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    english_pivots: Sequence[str],
    script_variants: Mapping[str, Sequence[str]],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    """`corpus -> arm -> variant -> pivot -> summary`, per the brief's `results.json` shape.

    Arms absent from `arms` (unavailable this run) and pivots absent from `arms` are
    skipped rather than stored as `None`, which keeps the nested dict free of placeholders
    every consumer would have to check. An unavailable *pivot* is worth a WARNING even so —
    `unavailable_arms` records why it could not be loaded, but a missing pivot silently
    removes a whole column from every corpus and arm below, so it is logged once, here,
    before the loops rather than once per corpus x arm x variant.
    """
    for pivot_name in english_pivots:
        if pivot_name not in arms:
            logger.warning(
                "English pivot %s is unavailable this run; every TPP column against it "
                "is omitted from results.json",
                pivot_name,
            )

    results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    for corpus in corpora:
        corpus_result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for arm_name in sanskrit_arm_names:
            tokenizer = arms.get(arm_name)
            if tokenizer is None:
                continue
            variant_result: dict[str, dict[str, Any]] = {}
            for variant in variants_for_family(tokenizer.family, script_variants):
                sanskrit_texts = corpus.sanskrit[variant]
                pivot_result: dict[str, Any] = {}
                for pivot_name in english_pivots:
                    pivot_tokenizer = arms.get(pivot_name)
                    if pivot_tokenizer is None:
                        continue
                    raw = tpp(
                        tokenizer,
                        sanskrit_texts,
                        corpus.english,
                        pivot_tokenizer,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        ci=ci,
                    )
                    pivot_result[pivot_name] = summarise_tpp(raw, ci=ci)
                    logger.info(
                        "%s / %s / %s / vs %s: TPP %.3f [%.3f, %.3f]",
                        corpus.name,
                        arm_name,
                        variant,
                        pivot_name,
                        raw["value"],
                        raw["ci_low"],
                        raw["ci_high"],
                    )
                variant_result[variant] = pivot_result
            corpus_result[arm_name] = variant_result
        results[corpus.name] = corpus_result
    return results


def controlled_pair_key(sanskrit_arm: str, english_arm: str) -> str:
    """`results.json`'s `tpp_controlled` key for one matched pair: `"<sa>/<en>"`."""
    return f"{sanskrit_arm}/{english_arm}"


def select_controlled_pairs(
    controlled_pairs: Sequence[Sequence[str]],
    arms: Mapping[str, LoadedTokenizer],
) -> list[tuple[str, str]]:
    """The configured `(sanskrit_arm, english_arm)` pairs whose *both* sides loaded.

    A pair with an unavailable side is dropped with a WARNING rather than silently
    substituted: the whole point of the pair is that the two arms match on algorithm,
    vocabulary size and training corpus, so falling back to some other English arm would
    quietly turn the controlled comparison back into an uncontrolled one. Raises
    `ValueError` for a malformed entry (not exactly two names), since that is a config
    error, not a missing artifact.
    """
    selected: list[tuple[str, str]] = []
    for pair in controlled_pairs:
        names = list(pair)
        if len(names) != 2:
            raise ValueError(
                f"config['controlled_pairs'] entry {names!r} must be [sanskrit_arm, english_arm]"
            )
        sanskrit_arm, english_arm = names
        missing = [name for name in (sanskrit_arm, english_arm) if name not in arms]
        if missing:
            logger.warning(
                "controlled pair %s is skipped: %s unavailable this run",
                controlled_pair_key(sanskrit_arm, english_arm),
                ", ".join(missing),
            )
            continue
        selected.append((sanskrit_arm, english_arm))
    return selected


def compute_tpp_controlled(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    pairs: Sequence[tuple[str, str]],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """`corpus -> "<sa_arm>/<en_arm>" -> summary`: the controlled TPP (docs/decisions.md,
    "Add a matched English control family E1 for TPP").

    This is the headline comparison. Everything else in `tpp` divides a Sanskrit arm's
    token count by a general-domain, 200k-vocabulary English tokenizer's, so the ratio
    mixes the language effect H2 is about with vocabulary size and training domain. Here
    both sides come from the same algorithm at the same vocabulary size, trained on the
    two sides of the *same* sentences, so those two nuisance factors are held constant —
    at each corpus both sides are in-domain, or neither is.

    The Sanskrit side is read in `CONTROLLED_VARIANT` (SLP1 — the only variant T1/T2
    have) and the English side as written; the summaries carry the same enriched keys as
    `tpp`, so the two blocks read the same way.
    """
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        pair_results: dict[str, dict[str, Any]] = {}
        for sanskrit_arm, english_arm in pairs:
            raw = tpp(
                arms[sanskrit_arm],
                corpus.sanskrit[CONTROLLED_VARIANT],
                corpus.english,
                arms[english_arm],
                n_bootstrap=n_bootstrap,
                seed=seed,
                ci=ci,
            )
            key = controlled_pair_key(sanskrit_arm, english_arm)
            pair_results[key] = summarise_tpp(raw, ci=ci)
            logger.info(
                "%s / controlled %s: TPP %.3f [%.3f, %.3f]",
                corpus.name,
                key,
                raw["value"],
                raw["ci_low"],
                raw["ci_high"],
            )
        results[corpus.name] = pair_results
    return results


def compute_tpp_hindi(
    corpus: CorpusData | None,
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    script_variants: Mapping[str, Sequence[str]],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, Any]]:
    """`arm -> variant -> summary` for the Hindi pivot: T0/T3 arms only, same tokenizer
    scoring both sides (config.yaml resolution 4); `{}` if the corpus has no Hindi side.
    """
    if corpus is None or corpus.hindi is None:
        return {}
    results: dict[str, dict[str, Any]] = {}
    for arm_name in sanskrit_arm_names:
        tokenizer = arms.get(arm_name)
        if tokenizer is None or tokenizer.family not in HINDI_PIVOT_FAMILIES:
            continue
        variant_result: dict[str, Any] = {}
        for variant in variants_for_family(tokenizer.family, script_variants):
            raw = tpp(
                tokenizer,
                corpus.sanskrit[variant],
                corpus.hindi[variant],
                n_bootstrap=n_bootstrap,
                seed=seed,
                ci=ci,
            )
            variant_result[variant] = summarise_tpp(raw, ci=ci)
            logger.info(
                "%s / %s / %s vs Hindi: TPP %.3f [%.3f, %.3f]",
                corpus.name,
                arm_name,
                variant,
                raw["value"],
                raw["ci_low"],
                raw["ci_high"],
            )
        results[arm_name] = variant_result
    return results


def compute_fertility_compression(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    script_variants: Mapping[str, Sequence[str]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Fertility and compression of the Sanskrit side, `corpus -> arm -> variant -> summary`.

    Reported per CLAUDE.md §2.1/§7, never headlined: fertility punishes exactly the word
    density this project studies, so it lives in its own results key and the README's
    per-corpus tables only, never the summary paragraph.
    """
    fert: dict[str, dict[str, dict[str, Any]]] = {}
    comp: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        fert[corpus.name] = {}
        comp[corpus.name] = {}
        for arm_name in sanskrit_arm_names:
            tokenizer = arms.get(arm_name)
            if tokenizer is None:
                continue
            fert[corpus.name][arm_name] = {}
            comp[corpus.name][arm_name] = {}
            for variant in variants_for_family(tokenizer.family, script_variants):
                texts = corpus.sanskrit[variant]
                fert[corpus.name][arm_name][variant] = summarise_metric(
                    fertility(tokenizer, texts)
                )
                comp[corpus.name][arm_name][variant] = summarise_metric(
                    compression(tokenizer, texts)
                )
    return fert, comp


# ------------------------------------------------------------- length-stratified TPP

#: Inclusive lower bounds of the sentence-length bins, counted in *English* whitespace
#: words: `1-8`, `9-16`, `17-24`, `25-40`, `41+` (docs/decisions.md, 2026-09-07). Fixed
#: and absolute rather than per-corpus quantiles, so a verse line and a prose sentence of
#: the same length fall in the same bin and can be read against each other; quantile bins
#: would put Itihāsa's short lines and Sāmayik's long periods in bins that share a name
#: and nothing else. `config["length_bin_edges"]` overrides them.
LENGTH_BIN_EDGES_DEFAULT: tuple[int, ...] = (1, 9, 17, 25, 41)

#: A bin holding fewer pairs than this is flagged `sparse` in `results.json` and drawn
#: with a hollow marker; `config["length_sparse_below"]` overrides it.
LENGTH_SPARSE_BELOW_DEFAULT = 30


def length_bin_labels(edges: Sequence[int]) -> list[str]:
    """`results.json` keys and x-tick labels for `edges`: `["1-8", ..., "41+"]`.

    Every bin but the last is closed and named by its inclusive range; the last is
    open-ended, because sentence length has no upper bound and a stated one would stop
    being true the first time a corpus exceeded it. Raises `ValueError` on empty `edges`,
    which would otherwise produce a single unnamed bin holding the whole corpus.
    """
    bounds = list(edges)
    if not bounds:
        raise ValueError("length bin edges are empty; at least one edge is required")
    labels = [f"{low}-{high - 1}" for low, high in zip(bounds, bounds[1:], strict=False)]
    labels.append(f"{bounds[-1]}+")
    return labels


def assign_length_bin(n_words: int, edges: Sequence[int]) -> int:
    """Index of the bin `n_words` falls in; `edges` are sorted, inclusive lower bounds.

    A count below the first edge — with the default edges, that is a 0-word sentence —
    goes in bin 0 as well, rather than being dropped or given a bin of its own. The
    corpora reaching this function are already filtered to lines that are non-blank in
    every language (`prepare_corpus`), so the case does not arise in a real run; it is
    defined rather than raising so that one stray whitespace-only line cannot abort a
    long experiment over a sentence it would have contributed nothing to.
    """
    return max(bisect.bisect_right(list(edges), n_words) - 1, 0)


def select_length_pairs(
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    english_pivots: Sequence[str],
    controlled: Sequence[tuple[str, str]],
    script_variants: Mapping[str, Sequence[str]],
) -> list[tuple[str, str, str]]:
    """The `(sanskrit_arm, variant, english_arm)` triples the length strata are measured on.

    Two sets, and only the first is a controlled comparison. (a) Every matched pair from
    `select_controlled_pairs`, read in `CONTROLLED_VARIANT` — these are what the figure
    draws, because both sides share an algorithm, a vocabulary size and a training corpus,
    so a difference between two length bins is a difference in the language, not in the
    two tokenizers. (b) Every available Sanskrit arm against the primary English pivot,
    read in its own natural script (`original` where the family has one, otherwise its
    single variant): deployed practice, recorded in `results.json` and never plotted
    (CLAUDE.md §2.5).

    Both sets are keyed `"<sanskrit_arm>/<english_arm>"` and cannot collide, since an E1
    control arm and a T0 pivot never share a name.
    """
    pairs = [
        (sanskrit_arm, CONTROLLED_VARIANT, english_arm)
        for sanskrit_arm, english_arm in controlled
    ]
    if not english_pivots:
        return pairs
    pivot_name = english_pivots[0]
    if pivot_name not in arms:
        logger.warning(
            "primary English pivot %s is unavailable this run; the deployed-practice "
            "length strata are omitted from results.json",
            pivot_name,
        )
        return pairs
    for arm_name in sanskrit_arm_names:
        tokenizer = arms.get(arm_name)
        if tokenizer is None:
            continue
        variants = variants_for_family(tokenizer.family, script_variants)
        if not variants:
            continue
        variant = ORIGINAL if ORIGINAL in variants else variants[0]
        pairs.append((arm_name, variant, pivot_name))
    return pairs


def compute_tpp_by_length(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    pairs: Sequence[tuple[str, str, str]],
    edges: Sequence[int],
    n_bootstrap: int,
    seed: int,
    ci: float,
    *,
    sparse_below: int = LENGTH_SPARSE_BELOW_DEFAULT,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """`corpus -> "<sa_arm>/<en_arm>" -> bin label -> summary`: TPP within a length band.

    A corpus-level TPP compares a whole corpus with a whole corpus, and the two primary
    corpora differ in more than genre: Itihāsa is verse *and* short (a śloka quarter is
    eight syllables), Sāmayik is prose *and* long. Since Itihāsa is the one corpus where
    the matched control leaves Sanskrit below 1.0, "verse is denser" and "short sentences
    are denser" are two readings of the same number. Binning every corpus on the same
    fixed edges of the *English* side's whitespace word count separates them: a 9-16-word
    verse line and a 9-16-word prose sentence are then directly comparable.

    The bin variable is the English side deliberately. Sanskrit word count is the
    quantity under study — sandhi and compounding are what make it small — so binning on
    it would sort sentences by the very effect being measured.

    Token counts are computed **once** per (corpus, pair) with `token_ratio` and then
    subset per bin (`RatioParts.subset`), so the strata cost one tokenization pass rather
    than one per bin, and every stratum is by construction a subset of the same
    measurement the corpus-level number comes from.

    Each summary is `summarise_tpp` of that bin's TPP, plus `n_pairs` (how many aligned
    pairs the bin holds), `sparse` (`n_pairs < sparse_below`: too few to read as anything
    but a hint, drawn hollow in the figure), `mean_words_en` / `mean_words_sa` (the mean
    whitespace word count of the two sides over the bin's pairs — the English one says
    where inside the band the bin actually sits, the Sanskrit one is the word-count
    compression that motivates the whole project), and `variant` (which script variant of
    the Sanskrit side was read). An empty bin still gets an entry, with `n_pairs` 0, a
    `nan` value and `sparse` true: a missing key and a measured-nothing bin would
    otherwise be indistinguishable in `results.json`.
    """
    labels = length_bin_labels(edges)
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for corpus in corpora:
        english_words = [len(text.split()) for text in corpus.english]
        sanskrit_words = [len(text.split()) for text in corpus.sanskrit[ORIGINAL]]
        indices_by_bin: list[list[int]] = [[] for _ in labels]
        for index, n_words in enumerate(english_words):
            indices_by_bin[assign_length_bin(n_words, edges)].append(index)

        pair_results: dict[str, dict[str, dict[str, Any]]] = {}
        for sanskrit_arm, variant, english_arm in pairs:
            parts = token_ratio(
                arms[sanskrit_arm],
                corpus.sanskrit[variant],
                corpus.english,
                arms[english_arm],
            )
            bin_results: dict[str, dict[str, Any]] = {}
            for label, indices in zip(labels, indices_by_bin, strict=True):
                raw = tpp_from_parts(
                    parts.subset(indices), n_bootstrap=n_bootstrap, seed=seed, ci=ci
                )
                summary = summarise_tpp(raw, ci=ci)
                summary["n_pairs"] = len(indices)
                summary["sparse"] = len(indices) < sparse_below
                summary["mean_words_en"] = _mean_over(english_words, indices)
                summary["mean_words_sa"] = _mean_over(sanskrit_words, indices)
                summary["variant"] = variant
                bin_results[label] = summary
            key = controlled_pair_key(sanskrit_arm, english_arm)
            pair_results[key] = bin_results
            logger.info(
                "%s / %s (%s) by length: %s",
                corpus.name,
                key,
                variant,
                ", ".join(
                    f"{label} n={bin_results[label]['n_pairs']} "
                    f"TPP {bin_results[label]['value']:.3f}"
                    for label in labels
                ),
            )
        results[corpus.name] = pair_results
    return results


def _mean_over(values: Sequence[int], indices: Sequence[int]) -> float:
    """Mean of `values` at `indices`, `nan` for an empty selection.

    `nan` rather than `0.0` because an empty bin has no mean sentence length, and `0.0`
    is a length a reader would take for a measurement (`_ratio.py` makes the same choice
    for an undefined ratio).
    """
    if not indices:
        return math.nan
    return sum(values[index] for index in indices) / len(indices)


# ------------------------------------------------------------------------------- Rényi


def english_renyi_arm_names(
    english_pivots: Sequence[str],
    controlled_pairs: Sequence[Sequence[str]],
) -> list[str]:
    """The English arms the Rényi block measures: the pivots, then the controlled-pair
    English sides, deduplicated in that order.

    Both families are needed and neither subsumes the other. The pivots (`T0_o200k`,
    `T0_llama4`) are the English side of the deployed-practice TPP column, so their token
    distribution is what the Sanskrit arms' is being divided by there; the `E1_*` arms are
    the English side of the *controlled* column and never appear in `english_pivots` (see
    that key's comment in config.yaml). Config order is preserved so `results.json` lists
    them the way the config does.
    """
    names: list[str] = []
    for name in [*english_pivots, *(pair[1] for pair in controlled_pairs if len(pair) == 2)]:
        if name not in names:
            names.append(name)
    return names


def _renyi_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """`summarise_metric` plus the five keys Rényi efficiency needs kept alongside it.

    `summarise_metric` carries the `value`/`n`/`unit` contract and nothing else; the
    entropy, the support it was normalised by, the token total, the order, and the
    alternative nominal-vocabulary normalisation are all part of reading the number
    (`sanskrit_tok.metrics.renyi`), so they are recorded rather than recomputed.
    """
    summary: dict[str, Any] = dict(summarise_metric(raw))
    summary["entropy_bits"] = float(raw["entropy_bits"])
    summary["n_types"] = int(raw["n_types"])
    summary["n_tokens"] = int(raw["n"])
    summary["efficiency_nominal"] = float(raw["efficiency_nominal"])
    summary["vocab_size"] = raw["vocab_size"]
    summary["alpha"] = float(raw["alpha"])
    return summary


def compute_renyi(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    script_variants: Mapping[str, Sequence[str]],
    alphas: Sequence[float],
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    """Rényi efficiency of the Sanskrit side, `corpus -> arm -> variant -> alpha -> summary`.

    A secondary intrinsic, reported and never headlined: it can be raised without changing
    what the tokenizer does to the text (Cognetta et al. 2024), so it describes the token
    distribution rather than testing any claim of this project. Each arm is normalised by
    its own observed support, and `efficiency_nominal` additionally by its `vocab_size`.
    The alpha keys are `str(alpha)` because JSON object keys are strings.
    """
    results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    for corpus in corpora:
        corpus_result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for arm_name in sanskrit_arm_names:
            tokenizer = arms.get(arm_name)
            if tokenizer is None:
                continue
            variant_result: dict[str, dict[str, dict[str, Any]]] = {}
            for variant in variants_for_family(tokenizer.family, script_variants):
                texts = corpus.sanskrit[variant]
                alpha_result: dict[str, dict[str, Any]] = {}
                for alpha in alphas:
                    raw = renyi_efficiency(
                        tokenizer, texts, alpha=alpha, vocab_size=tokenizer.vocab_size
                    )
                    alpha_result[str(alpha)] = _renyi_summary(raw)
                    logger.info(
                        "%s / %s / %s / alpha=%s: Renyi efficiency %.4f "
                        "(H %.3f bits over %d types, %d tokens)",
                        corpus.name,
                        arm_name,
                        variant,
                        alpha,
                        raw["value"],
                        raw["entropy_bits"],
                        raw["n_types"],
                        raw["n"],
                    )
                variant_result[variant] = alpha_result
            corpus_result[arm_name] = variant_result
        results[corpus.name] = corpus_result
    return results


def compute_renyi_english(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    english_arm_names: Sequence[str],
    alphas: Sequence[float],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """The same, for the English side: `corpus -> arm -> alpha -> summary`.

    One level shallower than `compute_renyi` because English has no script variant — it is
    read as written. This exists so the Sanskrit numbers have something to be read against:
    an efficiency of 0.9 means nothing on its own, but the same arm family's efficiency on
    the English side of the *same* sentences is a reference point.
    """
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for corpus in corpora:
        corpus_result: dict[str, dict[str, dict[str, Any]]] = {}
        for arm_name in english_arm_names:
            tokenizer = arms.get(arm_name)
            if tokenizer is None:
                continue
            alpha_result: dict[str, dict[str, Any]] = {}
            for alpha in alphas:
                raw = renyi_efficiency(
                    tokenizer, corpus.english, alpha=alpha, vocab_size=tokenizer.vocab_size
                )
                alpha_result[str(alpha)] = _renyi_summary(raw)
                logger.info(
                    "%s / %s / English / alpha=%s: Renyi efficiency %.4f "
                    "(H %.3f bits over %d types, %d tokens)",
                    corpus.name,
                    arm_name,
                    alpha,
                    raw["value"],
                    raw["entropy_bits"],
                    raw["n_types"],
                    raw["n"],
                )
            corpus_result[arm_name] = alpha_result
        results[corpus.name] = corpus_result
    return results


# ------------------------------------------------------------------------------ figure


def arm_label(name: str, vocab_size: int) -> str:
    """X-tick label: arm name, a `*` for provisional (T1/T2) arms, vocab size in `Nk`.

    `T1_bpe_raw_32k` -> `"T1_bpe_raw_32k* (32k)"`; `T0_o200k` -> `"T0_o200k (200k)"`.
    """
    family = name.split("_", 1)[0]
    star = "*" if family in PROVISIONAL_FAMILIES else ""
    thousands = round(vocab_size / 1000)
    return f"{name}{star} ({thousands}k)"


def controlled_pair_label(sanskrit_arm: str, english_arm: str) -> str:
    """X-tick label for one controlled pair: `"T1_bpe_raw_32k* / E1_bpe_32k"`.

    The `*` marks the provisional Sanskrit arm, exactly as `arm_label` does; the English
    control arm carries none — it is trained on the same (parallel-corpus) text, but the
    caveat the star stands for is about the Sanskrit side's interim training corpus.
    Vocabulary sizes are omitted: within a pair they are equal by construction, which is
    the whole point, and both are named in the arm keys already.
    """
    star = "*" if sanskrit_arm.split("_", 1)[0] in PROVISIONAL_FAMILIES else ""
    return f"{sanskrit_arm}{star} / {english_arm}"


def _plot_controlled_panel(
    axes: Any,
    corpus_name: str,
    controlled: Mapping[str, Any],
    pairs: Sequence[Sequence[str]],
) -> None:
    """One right-column panel: controlled TPP for every matched pair, with its CI.

    `pairs` is `config["controlled_pairs"]` (config order); a pair with no entry for this
    corpus — because one of its arms was unavailable — is omitted from the x-axis rather
    than drawn as a gap. The dashed line at 1.0 is the same reference the left column
    uses, so the two columns are read against the same threshold.
    """
    corpus_controlled = controlled.get(corpus_name, {})
    labels: list[str] = []
    values: list[float] = []
    lower_err: list[float] = []
    upper_err: list[float] = []
    for pair in pairs:
        sanskrit_arm, english_arm = pair[0], pair[1]
        entry = corpus_controlled.get(controlled_pair_key(sanskrit_arm, english_arm))
        if entry is None:
            continue
        value = entry.get("value")
        labels.append(controlled_pair_label(sanskrit_arm, english_arm))
        if value is None:
            values.append(math.nan)
            lower_err.append(0.0)
            upper_err.append(0.0)
            continue
        ci_low, ci_high = entry.get("ci_low"), entry.get("ci_high")
        values.append(float(value))
        lower_err.append(float(value) - ci_low if ci_low is not None else 0.0)
        upper_err.append(ci_high - float(value) if ci_high is not None else 0.0)

    positions = list(range(len(labels)))
    axes.errorbar(
        positions,
        values,
        yerr=[lower_err, upper_err],
        fmt="o",
        capsize=3,
        color="#2f855a",
        zorder=3,
    )
    axes.axhline(1.0, linestyle="--", color="gray", linewidth=1)

    # 1.0 is the threshold every one of these panels is read against, so it is kept
    # inside the axes with headroom rather than left to land on the frame, where a CI
    # sitting just below it is indistinguishable from one sitting just above.
    finite = [
        bound
        for value, lower, upper in zip(values, lower_err, upper_err, strict=True)
        if not math.isnan(value)
        for bound in (value - lower, value + upper)
    ]
    if finite:
        span_low, span_high = min([*finite, 1.0]), max([*finite, 1.0])
        span = span_high - span_low
        pad = span * FIGURE_Y_PAD_FRACTION if span > 0 else max(span_high * 0.2, 0.1)
        axes.set_ylim(span_low - pad, span_high + pad)

    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=40, ha="right", fontsize=6)
    axes.set_xlim(-0.5, max(len(positions) - 0.5, 0.5))
    axes.set_ylabel("TPP ratio", fontsize=9)
    axes.set_title(corpus_name, fontsize=9, loc="left")
    axes.spines[["top", "right"]].set_visible(False)


def _build_tpp_figure(results: Mapping[str, Any]) -> Any:
    """Build (but do not save or close) the TPP figure; returns the `Figure`.

    Split out from `make_figure` so tests can inspect the constructed `Figure` — its
    axes' tick labels, its `.texts` (the caption) — before anything is written to disk or
    the figure is closed, rather than only being able to check that two files exist.

    One row per corpus in config order (prose first), and two columns whenever the run
    produced a controlled comparison (`figure_pivot_controlled` and a non-empty
    `tpp_controlled`; otherwise the left column alone, so a `results.json` from before the
    E1 arms existed still plots). **Right column — the controlled comparison, and the one
    to read first:** each matched pair's TPP with its CI, Sanskrit arm over the English
    arm that matches it on algorithm, vocabulary size and training corpus. **Left column —
    deployed practice:** every arm against `T0_o200k`, whose 200k general-domain
    vocabulary makes it a description of what today's tokenizers do, not a control.

    Each left panel plots, for every arm present in that corpus's `tpp` entry (unavailable
    arms have none and are silently omitted from the x-axis): the TPP of the SLP1 variant
    against `T0_o200k` as a point with a 95% bootstrap-CI error bar (every arm has an
    SLP1 variant, so this puts T0/T3/T1/T2 on the same footing), plus — for T0/T3 arms,
    which also carry an `original`-script variant — a second, thin marker at the same x
    position showing the un-transliterated number, so the transliteration effect is
    visible. A dashed line at 1.0 marks the sign flip TPP is testing for.

    Each panel's y-axis is scaled from the SLP1 series alone (its values and CI bounds,
    plus 1.0, plus headroom) rather than from every point on the panel: `T0_gpt2`'s
    original-script number is 4-8x every other arm's (CLAUDE.md §2.1 — an old,
    Devanagari-blind vocabulary falling back to near-byte-level segmentation), and
    letting it set the axis limits, as an unscaled scatter would, squeezes every other
    point — including the SLP1 sign flip this figure exists to show — into a sliver at
    the bottom. A secondary marker that falls outside the resulting y-range is instead
    drawn just inside the axis edge as a triangle pointing further off-scale, annotated
    with its true value (e.g. "▲ 6.56"), so the reader can see it exists and what it is
    without it distorting the rest of the panel.

    Provisional (T1/T2) arms carry a `*` in their tick label; the caption explains it,
    together with one sentence naming any arm omitted for being unavailable this run
    (`results["unavailable_arms"]`, `unavailable_caption`).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on CI and over ssh
    import matplotlib.pyplot as plt

    config = results["config"]
    corpus_entries = list(config["corpora"])
    arm_names = list(config["sanskrit_arms"])
    tpp_results = results["tpp"]
    sources = results["tokenizer_sources"]

    if not corpus_entries:
        raise ValueError("results['config']['corpora'] is empty; nothing to plot")

    controlled = results.get("tpp_controlled", {})
    controlled_pairs = list(config.get("controlled_pairs", []))
    show_controlled = bool(config.get("figure_pivot_controlled")) and bool(
        controlled and controlled_pairs
    )

    n_panels = len(corpus_entries)
    n_columns = 2 if show_controlled else 1
    figure, axes_grid = plt.subplots(
        n_panels,
        n_columns,
        figsize=(7.6 * n_columns if show_controlled else 9.5, 3.2 * n_panels),
        squeeze=False,
    )
    axes_list = [row[0] for row in axes_grid]

    if show_controlled:
        for row, entry in zip(axes_grid, corpus_entries, strict=True):
            _plot_controlled_panel(row[1], str(entry["name"]), controlled, controlled_pairs)

    for panel_index, (axes, entry) in enumerate(zip(axes_list, corpus_entries, strict=True)):
        corpus_name = str(entry["name"])
        corpus_tpp = tpp_results.get(corpus_name, {})
        available_arms = [name for name in arm_names if name in corpus_tpp]

        labels: list[str] = []
        main_values: list[float] = []
        lower_err: list[float] = []
        upper_err: list[float] = []
        secondary_x: list[int] = []
        secondary_values: list[float] = []

        for position, arm_name in enumerate(available_arms):
            vocab_size = int(sources[arm_name]["vocab_size"])
            labels.append(arm_label(arm_name, vocab_size))

            main_entry = corpus_tpp[arm_name].get(FIGURE_MAIN_VARIANT, {}).get(FIGURE_PIVOT)
            main_value = main_entry.get("value") if main_entry is not None else None
            if main_value is not None:
                value = float(main_value)
                ci_low = main_entry.get("ci_low") if main_entry is not None else None
                ci_high = main_entry.get("ci_high") if main_entry is not None else None
                main_values.append(value)
                lower_err.append(value - ci_low if ci_low is not None else 0.0)
                upper_err.append(ci_high - value if ci_high is not None else 0.0)
            else:
                main_values.append(math.nan)
                lower_err.append(0.0)
                upper_err.append(0.0)

            secondary_entry = (
                corpus_tpp[arm_name].get(FIGURE_SECONDARY_VARIANT, {}).get(FIGURE_PIVOT)
            )
            secondary_value = secondary_entry.get("value") if secondary_entry is not None else None
            if secondary_value is not None:
                secondary_x.append(position)
                secondary_values.append(float(secondary_value))

        positions = list(range(len(available_arms)))

        # Y-range from the SLP1 (main) series alone, so an off-scale original-script
        # point (chiefly T0_gpt2) cannot squash the rest of the panel.
        main_lows = [
            value - lower for value, lower in zip(main_values, lower_err, strict=True)
        ]
        main_highs = [
            value + upper for value, upper in zip(main_values, upper_err, strict=True)
        ]
        finite_lows = [value for value in main_lows if not math.isnan(value)]
        finite_highs = [value for value in main_highs if not math.isnan(value)]
        if finite_lows and finite_highs:
            span_low = min([*finite_lows, 1.0])
            span_high = max([*finite_highs, 1.0])
            span = span_high - span_low
            pad = span * FIGURE_Y_PAD_FRACTION if span > 0 else max(span_high * 0.2, 0.1)
            y_bottom, y_top = span_low - pad, span_high + pad
        else:
            y_bottom, y_top = 0.0, 2.0

        axes.errorbar(
            positions,
            main_values,
            yerr=[lower_err, upper_err],
            fmt="o",
            capsize=3,
            color="#2b6cb0",
            label="SLP1 (every arm)",
            zorder=3,
        )

        # Secondary (original-script) markers: plotted normally inside the SLP1-derived
        # range, clipped to the nearest edge with an annotated arrow otherwise.
        in_range_x, in_range_values = [], []
        clip_inset = (y_top - y_bottom) * FIGURE_CLIP_INSET_FRACTION
        for x, value in zip(secondary_x, secondary_values, strict=True):
            if value > y_top:
                clip_y = y_top - clip_inset
                axes.scatter(
                    [x], [clip_y], marker="^", s=70, color="#c05621", zorder=4
                )
                axes.annotate(
                    f"▲ {value:.2f}",
                    (x, clip_y),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=6,
                    color="#c05621",
                )
            elif value < y_bottom:
                clip_y = y_bottom + clip_inset
                axes.scatter(
                    [x], [clip_y], marker="v", s=70, color="#c05621", zorder=4
                )
                axes.annotate(
                    f"▼ {value:.2f}",
                    (x, clip_y),
                    textcoords="offset points",
                    xytext=(0, -4),
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="#c05621",
                )
            else:
                in_range_x.append(x)
                in_range_values.append(value)
        if in_range_values:
            axes.scatter(
                in_range_x,
                in_range_values,
                marker="_",
                s=90,
                linewidths=2,
                color="#c05621",
                label="original script (T0/T3 only)",
                zorder=2,
            )

        axes.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axes.set_ylim(y_bottom, y_top)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axes.set_xlim(-0.5, max(len(positions) - 0.5, 0.5))
        axes.set_ylabel("TPP ratio", fontsize=9)
        axes.set_title(corpus_name, fontsize=9, loc="left")
        axes.spines[["top", "right"]].set_visible(False)
        if panel_index == 0:
            axes.legend(fontsize=7, loc="best")

    caption = FIGURE_CAPTION_PROVISIONAL
    omitted = unavailable_caption(results.get("unavailable_arms", {}))
    if omitted:
        caption = f"{caption}  {omitted}"

    figure.suptitle(FIGURE_SUPTITLE, fontsize=11)
    figure.text(0.01, 0.005, caption, fontsize=7)
    if show_controlled:
        # Column headers rather than per-panel titles: each describes its whole column,
        # and every panel still carries its corpus name. They, not the suptitle, name the
        # English side a column is measured against — the two columns use different ones.
        figure.text(0.25, 0.955, FIGURE_DEPLOYED_TITLE, ha="center", fontsize=8)
        figure.text(0.75, 0.955, FIGURE_CONTROLLED_TITLE, ha="center", fontsize=8)
        figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    else:
        figure.text(0.5, 0.955, FIGURE_DEPLOYED_TITLE, ha="center", fontsize=8)
        figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    return figure


#: Stem of the length-stratified figure's two files.
LENGTH_FIGURE_STEM = "tpp_by_length"
LENGTH_FIGURE_SUPTITLE = (
    "Tokens per proposition by sentence length, matched control, 95% bootstrap CI"
)
#: Names the bin variable and the hollow marker; `{n}` is filled from `sparse_below`. The
#: bin variable belongs on the figure because "17-24" alone does not say 17-24 of what,
#: and the answer (English words, not Sanskrit ones) is the choice that makes the panels
#: comparable to each other.
LENGTH_FIGURE_FOOTNOTE = "bins by English whitespace word count; hollow = fewer than {n} pairs"
#: One colour per controlled pair, cycled; the figure carries at most four of them.
LENGTH_SERIES_COLORS = ("#2b6cb0", "#2f855a", "#c05621", "#6b46c1", "#b83280", "#2c7a7b")


def _length_series(
    entries: Mapping[str, Any],
    labels: Sequence[str],
) -> tuple[list[int], list[float], list[float], list[float], list[bool]]:
    """One pair's plottable points: x, y, lower/upper error, and the sparse flag per point.

    Bins with no entry, no value, or a non-finite one (an empty bin's `nan`, or the `null`
    the JSON sanitiser writes for it) are dropped rather than plotted at zero, so the line
    simply skips a length band the corpus has no sentences in.
    """
    xs: list[int] = []
    values: list[float] = []
    lower_err: list[float] = []
    upper_err: list[float] = []
    sparse: list[bool] = []
    for position, label in enumerate(labels):
        entry = entries.get(label)
        if entry is None:
            continue
        raw = entry.get("value")
        if raw is None or not math.isfinite(float(raw)):
            continue
        value = float(raw)
        ci_low, ci_high = entry.get("ci_low"), entry.get("ci_high")
        low = float(ci_low) if ci_low is not None and math.isfinite(float(ci_low)) else value
        high = float(ci_high) if ci_high is not None and math.isfinite(float(ci_high)) else value
        xs.append(position)
        values.append(value)
        lower_err.append(max(value - low, 0.0))
        upper_err.append(max(high - value, 0.0))
        sparse.append(bool(entry.get("sparse")))
    return xs, values, lower_err, upper_err, sparse


def _build_tpp_by_length_figure(results: Mapping[str, Any]) -> Any:
    """Build (but do not save or close) the length-stratified TPP figure; returns the `Figure`.

    One panel per corpus in config order (prose before verse, CLAUDE.md §2.7), sharing a
    y axis so the panels can be read against each other — which is the entire point:
    Itihāsa's verse lines and Sāmayik's prose sentences are plotted at the same x
    position when they hold the same number of English words, and the question the figure
    answers is whether the two panels' curves differ at equal length or only in where
    their sentences sit along the x axis.

    One line-and-marker series per **controlled** pair (`config["controlled_pairs"]`,
    labelled by `controlled_pair_label`), with its bootstrap CI as error bars. The
    deployed-practice series that `results["tpp_by_length"]` also carries are deliberately
    not drawn: an off-the-shelf 200k English vocabulary against a 32k Sanskrit one is a
    description of current practice, never a controlled comparison (CLAUDE.md §2.5), and
    on a figure about a *difference between length bands* it would read as one.

    A bin with fewer than `config["length_sparse_below"]` pairs is drawn with a hollow
    marker: at the corpora's sizes the longest band can hold a handful of sentences, whose
    ratio is real but whose interval is wide enough that a reader should not follow the
    line into it. The dashed line at 1.0 is the same sign-flip threshold the main figure
    uses. Raises `ValueError` if the config names no corpora; every other emptiness —
    no `tpp_by_length` block at all, no controlled pairs, no non-empty bin — draws the
    axes with the reference line and nothing on them rather than failing, so a
    `results.json` from before this task still plots.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on CI and over ssh
    import matplotlib.pyplot as plt

    config = results["config"]
    corpus_entries = list(config["corpora"])
    if not corpus_entries:
        raise ValueError("results['config']['corpora'] is empty; nothing to plot")

    edges = list(
        results.get("length_bin_edges")
        or config.get("length_bin_edges")
        or LENGTH_BIN_EDGES_DEFAULT
    )
    labels = length_bin_labels(edges)
    positions = list(range(len(labels)))
    sparse_below = int(config.get("length_sparse_below", LENGTH_SPARSE_BELOW_DEFAULT))
    by_length = results.get("tpp_by_length", {})
    controlled_pairs = list(config.get("controlled_pairs", []))

    n_panels = len(corpus_entries)
    figure, axes_grid = plt.subplots(
        n_panels,
        1,
        figsize=(8.0, 2.9 * n_panels),
        squeeze=False,
        sharey=True,
    )
    axes_list = [row[0] for row in axes_grid]

    handles: dict[str, Any] = {}
    finite_bounds: list[float] = []
    for axes, entry in zip(axes_list, corpus_entries, strict=True):
        corpus_name = str(entry["name"])
        corpus_by_length = by_length.get(corpus_name, {})
        for index, pair in enumerate(controlled_pairs):
            sanskrit_arm, english_arm = pair[0], pair[1]
            entries = corpus_by_length.get(controlled_pair_key(sanskrit_arm, english_arm))
            if not entries:
                continue
            xs, values, lower_err, upper_err, sparse = _length_series(entries, labels)
            if not xs:
                continue
            color = LENGTH_SERIES_COLORS[index % len(LENGTH_SERIES_COLORS)]
            label = controlled_pair_label(sanskrit_arm, english_arm)
            line = axes.plot(xs, values, color=color, linewidth=1.2, zorder=2)[0]
            handles.setdefault(label, line)
            # Two error bars rather than one: matplotlib takes a single face colour for a
            # whole series, and the hollow marker is what tells a reader which points rest
            # on too few sentences to lean on.
            for is_sparse in (False, True):
                selected = [position for position, flag in enumerate(sparse) if flag is is_sparse]
                if not selected:
                    continue
                axes.errorbar(
                    [xs[position] for position in selected],
                    [values[position] for position in selected],
                    yerr=[
                        [lower_err[position] for position in selected],
                        [upper_err[position] for position in selected],
                    ],
                    fmt="o",
                    linestyle="none",
                    capsize=3,
                    color=color,
                    markerfacecolor="none" if is_sparse else color,
                    zorder=3,
                )
            finite_bounds.extend(
                bound
                for value, low, high in zip(values, lower_err, upper_err, strict=True)
                for bound in (value - low, value + high)
            )

        axes.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, fontsize=8)
        axes.set_xlim(-0.4, len(positions) - 0.6 if len(positions) > 1 else 0.6)
        axes.set_ylabel("TPP ratio", fontsize=9)
        axes.set_xlabel("English words per sentence", fontsize=8)
        axes.set_title(corpus_name, fontsize=9, loc="left")
        axes.spines[["top", "right"]].set_visible(False)

    # Shared y range over every panel's points, with 1.0 always inside it: the panels are
    # only comparable if they are on the same scale, and the threshold is what each is
    # read against. Falls back to a plain range when no panel drew anything.
    if finite_bounds:
        span_low, span_high = min([*finite_bounds, 1.0]), max([*finite_bounds, 1.0])
        span = span_high - span_low
        pad = span * FIGURE_Y_PAD_FRACTION if span > 0 else max(span_high * 0.2, 0.1)
        axes_list[0].set_ylim(span_low - pad, span_high + pad)
    else:
        axes_list[0].set_ylim(0.0, 2.0)

    if handles:
        axes_list[0].legend(
            handles=list(handles.values()),
            labels=list(handles),
            fontsize=7,
            loc="best",
        )

    figure.suptitle(LENGTH_FIGURE_SUPTITLE, fontsize=11)
    figure.text(0.01, 0.005, LENGTH_FIGURE_FOOTNOTE.format(n=sparse_below), fontsize=7)
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    return figure


def make_figure(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Build both Experiment 02 figures and save each as PDF and PNG (CLAUDE.md §8).

    `tpp_by_arm` is the central per-arm figure (`_build_tpp_figure`); `tpp_by_length` is
    the length-stratified one (`_build_tpp_by_length_figure`). Both are built before
    either is written, so a failure in the second does not leave a half-refreshed output
    directory whose two figures came from different code.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        (FIGURE_STEM, _build_tpp_figure(results)),
        (LENGTH_FIGURE_STEM, _build_tpp_by_length_figure(results)),
    )
    paths: list[Path] = []
    for stem, figure in figures:
        stem_paths = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
        for path in stem_paths:
            figure.savefig(path, dpi=200)
        plt.close(figure)
        paths.extend(stem_paths)
    logger.info("wrote %s", " and ".join(str(path) for path in paths))
    return paths


# -------------------------------------------------------------------------------- main


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
    root = repo_root()
    config = load_config(args.config)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    n_bootstrap = int(config["n_bootstrap"])
    ci = float(config.get("ci", 0.95))

    corpora_config: list[dict[str, Any]] = list(config["corpora"])
    if not corpora_config:
        raise ValueError(f"{args.config}: 'corpora' is empty; nothing to measure")
    sanskrit_arm_names: list[str] = list(config["sanskrit_arms"])
    english_pivots: list[str] = list(config["english_pivots"])
    controlled_pairs: list[list[str]] = [list(pair) for pair in config.get("controlled_pairs", [])]
    hindi_pivot_corpus_name = str(config["hindi_pivot_corpus"])
    script_variants: dict[str, list[str]] = {
        family: list(variants) for family, variants in config["script_variants"].items()
    }
    out_dir = resolve_path(str(config["output_dir"]), root)
    exclusion_path = resolve_path(str(config["exclusion_path"]), root)
    exclusion_path_en = resolve_path(str(config["exclusion_path_en"]), root)

    corpora = [prepare_corpus(entry, root) for entry in corpora_config]

    hashes = load_exclusion_hashes(exclusion_path)
    hashes_en = load_exclusion_hashes(exclusion_path_en)
    exclusion_check: dict[str, dict[str, int]] = {}
    exclusion_check_en: dict[str, dict[str, int]] = {}
    for corpus in corpora:
        report = exclusion_check_for(corpus.sanskrit[ORIGINAL], hashes)
        exclusion_check[corpus.name] = report
        if report["n_missing"]:
            logger.warning(
                "%s: %d/%d Sanskrit sentences are NOT in the exclusion list "
                "(data/exclusion_hashes.txt may be stale)",
                corpus.name,
                report["n_missing"],
                report["n"],
            )
        # The same check for the English side, which is what kept the E1 control arms
        # away from this evaluation text (CLAUDE.md §2.4 applies to them as to T1/T2).
        report_en = exclusion_check_for(corpus.english, hashes_en, sentence_hash_en)
        exclusion_check_en[corpus.name] = report_en
        if report_en["n_missing"]:
            logger.warning(
                "%s: %d/%d English sentences are NOT in the English exclusion list "
                "(data/exclusion_hashes_en.txt may be stale; the E1 arms could have "
                "trained on evaluation text)",
                corpus.name,
                report_en["n_missing"],
                report_en["n"],
            )

    all_arm_names = sorted(
        set(sanskrit_arm_names)
        | set(english_pivots)
        | {name for pair in controlled_pairs for name in pair}
    )
    arms, unavailable_arms = load_arms(all_arm_names)

    tpp_results = compute_tpp(
        corpora, arms, sanskrit_arm_names, english_pivots, script_variants, n_bootstrap, seed, ci
    )
    controlled_selected = select_controlled_pairs(controlled_pairs, arms)
    tpp_controlled = compute_tpp_controlled(
        corpora,
        arms,
        controlled_selected,
        n_bootstrap,
        seed,
        ci,
    )
    hindi_corpus = next((c for c in corpora if c.name == hindi_pivot_corpus_name), None)
    if hindi_corpus is None:
        logger.warning(
            "hindi_pivot_corpus %r is not among the configured corpora; "
            "tpp_hindi will be empty",
            hindi_pivot_corpus_name,
        )
    tpp_hindi = compute_tpp_hindi(
        hindi_corpus, arms, sanskrit_arm_names, script_variants, n_bootstrap, seed, ci
    )
    length_bin_edges = [
        int(edge) for edge in config.get("length_bin_edges", LENGTH_BIN_EDGES_DEFAULT)
    ]
    tpp_by_length = compute_tpp_by_length(
        corpora,
        arms,
        select_length_pairs(
            arms, sanskrit_arm_names, english_pivots, controlled_selected, script_variants
        ),
        length_bin_edges,
        n_bootstrap,
        seed,
        ci,
        sparse_below=int(config.get("length_sparse_below", LENGTH_SPARSE_BELOW_DEFAULT)),
    )
    fertility_results, compression_results = compute_fertility_compression(
        corpora, arms, sanskrit_arm_names, script_variants
    )
    renyi_alphas = [float(alpha) for alpha in config["renyi_alphas"]]
    renyi_results = compute_renyi(corpora, arms, sanskrit_arm_names, script_variants, renyi_alphas)
    renyi_english_results = compute_renyi_english(
        corpora,
        arms,
        english_renyi_arm_names(english_pivots, controlled_pairs),
        renyi_alphas,
    )

    results: dict[str, Any] = {
        "experiment": str(config.get("experiment", out_dir.name)),
        **provenance(root),
        "config": config,
        "tokenizer_sources": tokenizer_sources(arms),
        "unavailable_arms": unavailable_arms,
        "corpora": {
            corpus.name: {"split": corpus.split, "n_total": corpus.n_total, "n_used": corpus.n_used}
            for corpus in corpora
        },
        "exclusion_check": exclusion_check,
        "exclusion_check_en": exclusion_check_en,
        "tpp_controlled": tpp_controlled,
        "tpp": tpp_results,
        "tpp_hindi": tpp_hindi,
        "tpp_by_length": tpp_by_length,
        "length_bin_edges": length_bin_edges,
        "fertility": fertility_results,
        "compression": compression_results,
        "renyi": renyi_results,
        "renyi_english": renyi_english_results,
    }

    write_results(results, out_dir, args.config)

    make_figure(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
