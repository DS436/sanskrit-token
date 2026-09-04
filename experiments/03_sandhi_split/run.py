"""Experiment 03 — does reversing sandhi before subword learning lower TPP? (RQ3)

Hypothesis (outline §1, H3, TPP/fertility half): a subword vocabulary learned on raw
Sanskrit spends its merges on the surface junctions sandhi creates rather than on
morphemes, so splitting sandhi (and, with this splitter, samāsa) *before* subword learning
should lower tokens-per-proposition and fertility relative to a matched raw-text arm at the
same algorithm and the same vocabulary size.

The comparison is matched twice over. Within a pair, `T4_bpe_split_32k` differs from
`T1_bpe_raw_32k` in exactly one thing — whether the training text was split first — and
both are measured on the same evaluation sentences. Across languages, each Sanskrit arm is
divided by the `E1_*` English arm that matches it on algorithm, vocabulary size and
training corpus (docs/decisions.md, "Add a matched English control family E1 for TPP"),
because the deployed 200k-vocabulary pivot mixes the language effect with vocabulary size
and domain fit. Both pivots are computed; only the controlled one carries the verdict.

**TPP leads; fertility never does** (CLAUDE.md §2.1). Fertility is additionally awkward
here: splitting inserts whitespace, so "tokens per whitespace word" changes denominator
between the two arms of a pair. It is therefore reported twice — primary over the *raw*
sentence's word count (`fertility_against_reference`, the same denominator for every arm),
secondary over the split text's own words (docs/decisions.md, "Fertility for split arms
uses the raw word count as the primary denominator").

**The split text comes from disk, never from the model.** `split_corpora.py` wrote
`data/processed/split/<corpus>.jsonl` with both variants per sentence: `output` (the
model's segmentation reconciled against the raw sentence — the primary text, since the
raw model output silently deletes ~13% of the characters and would credit T4 with tokens
saved by deleting content) and `output_model` (the raw model output, measured as a clearly
labelled secondary variant). A sentence missing from that file aborts the run: falling
back to raw text would quietly measure a T4 arm on unsplit sentences.

Run it with `uv run python experiments/03_sandhi_split/run.py`. Relative paths in the
config are resolved against the repository root, so the working directory does not matter.
Every function below is pure or takes its I/O paths explicitly, so `tests/test_exp03.py`
can exercise the whole pipeline — aggregation, plotting and `run()` end to end — on
synthetic data without touching the network, a real corpus or the splitter.
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

from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash_en
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
    tokenizer_sources,
    unavailable_caption,
    write_results,
)
from sanskrit_tok.metrics._ratio import token_ratio
from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility, fertility_against_reference
from sanskrit_tok.metrics.summary import summarise_metric
from sanskrit_tok.metrics.tpp import tpp, tpp_paired_delta
from sanskrit_tok.tokenizers.registry import LoadedTokenizer

logger = logging.getLogger("exp03")

#: The three Sanskrit texts of one corpus, and the keys `results.json` stores them under.
#: `RAW` is the SLP1 transliteration of the corpus as written — what the `T1`/`T2` arms
#: are measured on. `SPLIT` is the reconciled split text and is the primary variant for
#: the `T4` arms; `SPLIT_MODEL` is the splitter's raw output, kept as a secondary variant
#: because it is not a pure re-segmentation of its input (docs/decisions.md, "T4 text is
#: the model's segmentation reconciled against the raw sentence").
RAW = "raw_slp1"
SPLIT = "reconciled"
SPLIT_MODEL = "model_raw"

#: What an arm does with those texts. `RAW_ARM`s read `RAW` only; `SPLIT_ARM`s read
#: `SPLIT` (primary) and `SPLIT_MODEL` (secondary).
RAW_ARM = "raw"
SPLIT_ARM = "split"

#: Which English side a TPP column is divided by, recorded inside every summary so the
#: file says which of the two questions a number answers without consulting the config.
ROLE_CONTROLLED = "controlled"
ROLE_DEPLOYED = "deployed"

#: Per-corpus manifest keys copied into `splitter_stats`; the run-wide counters
#: (`n_chunked`, `n_model`, ...) live in the top-level `splitter` block instead.
MANIFEST_CORPUS_KEYS: tuple[str, ...] = (
    "n_sentences",
    "n_out",
    "n_units_raw",
    "n_units_out",
    "n_units_kept_verbatim",
    "n_units_replaced_inexact",
    "n_units_changed",
    "fraction_units_changed",
    "chars_raw",
    "chars_model",
    "chars_out",
    "char_retention_model",
    "char_retention_reconciled",
    "seconds",
)

FIGURE_STEM = "tpp_split_vs_raw"
FIGURE_SUPTITLE = (
    "Tokens per proposition vs the matched English control (E1), 95% bootstrap CI"
)
FIGURE_SUBTITLE = (
    "raw-text arms (T1/T2) vs sandhi-split arms (T4), matched on algorithm and vocabulary"
)
#: Static half of the caption; the splitter id and any omitted arms are appended at plot
#: time, since both depend on the run.
FIGURE_CAPTION_PROVISIONAL = (
    "Provisional: every arm is trained on the Sanskrit (or English) side of two parallel "
    "corpora, not the monolingual corpus. "
    "split = sandhi + compound (samāsa) splitting, ByT5-Sanskrit, reconciled against the "
    "raw sentence."
)
#: Fraction of the y-range added above and below as headroom, so a CI just below the 1.0
#: reference line is distinguishable from one just above it.
FIGURE_Y_PAD_FRACTION = 0.15
#: How far the raw and split markers of one pair sit either side of its x position.
FIGURE_MARKER_OFFSET = 0.12

# ---------------------------------------------------------------------- corpus wrangling


@dataclass(frozen=True)
class CorpusData:
    """One corpus after loading, blank-index filtering and split-cache lookup.

    `texts` holds the Sanskrit side in all three variants (`RAW`, `SPLIT`, `SPLIT_MODEL`),
    index-aligned with `english` and with `raw_deva`, the corpus as written (which is what
    the exclusion list hashes).
    """

    name: str
    split: str
    n_total: int
    n_used: int
    raw_deva: list[str]
    texts: dict[str, list[str]]
    english: list[str]


def read_split_records(name: str, path: Path) -> list[dict[str, Any]]:
    """Every record of one corpus's split jsonl, in file order.

    Raises `FileNotFoundError` naming the corpus and the command that writes the file: the
    runner never loads the splitter itself, so a missing file is a missing prerequisite,
    not something to recover from.
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
) -> dict[str, list[str]]:
    """`{SPLIT: [...], SPLIT_MODEL: [...]}` for one corpus, aligned to `raw_deva`.

    The jsonl's `index` is the position in the same aligned, non-blank sentence list this
    script evaluates (`split_corpora.eval_sentences` applies the identical filter), so the
    two are joined on it rather than on file order.

    Aborts — never falls back to raw text — on any of three mismatches, each naming the
    corpus and a count: a sentence with no cached split (the split job is unfinished for
    this corpus), a record count that differs from the evaluation set (the cache was built
    from a different corpus version), and a cached `raw_deva` that is not the sentence at
    that index (the cache is stale). Silently measuring a `T4` arm on unsplit or misaligned
    text would produce a plausible number that means nothing.
    """
    by_index: dict[int, Mapping[str, Any]] = {int(record["index"]): record for record in records}
    total = len(raw_deva)
    missing = [index for index in range(total) if index not in by_index]
    if missing:
        raise ValueError(
            f"{name}: {len(missing)} of {total} evaluation sentence(s) are missing from the "
            f"split cache (first missing index {missing[0]}); the split job has not finished "
            "for this corpus. Re-run experiments/03_sandhi_split/split_corpora.py — every "
            "sentence already split is a cache hit."
        )
    if len(records) != total:
        raise ValueError(
            f"{name}: the split cache holds {len(records)} record(s) for {total} evaluation "
            "sentence(s); it was built from a different version of this corpus. Delete "
            "the corpus jsonl and re-run split_corpora.py."
        )
    mismatched = [
        index
        for index in range(total)
        if str(by_index[index].get("raw_deva", "")).strip() != raw_deva[index].strip()
    ]
    if mismatched:
        raise ValueError(
            f"{name}: {len(mismatched)} of {total} cached sentence(s) do not match the corpus "
            f"at the same index (first at index {mismatched[0]}); the split cache is stale. "
            "Delete the corpus jsonl and re-run split_corpora.py."
        )
    return {
        SPLIT: [str(by_index[index]["output"]) for index in range(total)],
        SPLIT_MODEL: [str(by_index[index]["output_model"]) for index in range(total)],
    }


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
    texts = {RAW: [to_slp1(text, "devanagari") for text in raw_deva]}
    records = read_split_records(name, split_dir / f"{name}.jsonl")
    texts.update(split_texts_for(name, records, raw_deva))

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


# ------------------------------------------------------------------------ tokenizers


@dataclass(frozen=True)
class ArmSpec:
    """One tokenizer arm and the text variants it is measured on.

    `primary` is the variant its fertility, compression and `tpp_delta` are computed from;
    `variants` is every variant its TPP is computed on (one for a raw arm, two for a split
    arm, the second being the splitter's unreconciled output).
    """

    name: str
    kind: str
    variants: tuple[str, ...]
    primary: str


def arm_specs(arms_raw: Sequence[str], arms_split: Sequence[str]) -> list[ArmSpec]:
    """`ArmSpec`s for the configured arms, raw arms first (config order within each)."""
    specs = [ArmSpec(name, RAW_ARM, (RAW,), RAW) for name in arms_raw]
    specs += [ArmSpec(name, SPLIT_ARM, (SPLIT, SPLIT_MODEL), SPLIT) for name in arms_split]
    return specs


def arm_algorithm_and_vocab(name: str) -> tuple[str, str]:
    """`("bpe", "32k")` from `T1_bpe_raw_32k` or `T4_bpe_split_32k`.

    The arm-name grammar of CLAUDE.md §6 is `<family>_<algorithm>_<text>_<vocab>`, so the
    algorithm is the second field and the vocabulary size the last.
    """
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"arm name {name!r} is not <family>_<algorithm>_..._<vocab>")
    return parts[1], parts[-1]


def check_matched_pair(raw_arm: str, split_arm: str) -> None:
    """Raise `ValueError` unless the two arms match on algorithm and vocabulary size.

    A pair that differs in either is not a controlled comparison at all — the delta would
    conflate splitting with a vocabulary or algorithm change — and a config typo pairing
    `T1_bpe_raw_32k` with `T4_bpe_split_64k` is otherwise invisible in the output.
    """
    raw_parts = arm_algorithm_and_vocab(raw_arm)
    split_parts = arm_algorithm_and_vocab(split_arm)
    if raw_parts != split_parts:
        raise ValueError(
            f"matched pair [{raw_arm}, {split_arm}] is not matched: "
            f"algorithm/vocab {raw_parts} vs {split_parts}"
        )


def matched_pair_label(raw_arm: str, split_arm: str) -> str:
    """The figure's x-tick label for one pair: `"bpe 32k"`.

    A pair has one algorithm and one vocabulary size or it is not a pair at all, so the
    label is only well defined for a matched one and `check_matched_pair` is called first.
    The family prefixes are what the two markers *mean*, so the label carries neither: the
    legend says which marker is raw and which is split.
    """
    check_matched_pair(raw_arm, split_arm)
    algorithm, vocab = arm_algorithm_and_vocab(split_arm)
    return f"{algorithm} {vocab}"


def matched_pair_key(raw_arm: str, split_arm: str) -> str:
    """`results.json`'s `tpp_delta` key: `"<split arm>/<raw arm>"`, in delta order.

    The delta is the first minus the second, so the key reads in the same order as the
    subtraction — as `tpp_controlled`'s `"<sanskrit>/<english>"` keys do in Experiment 02.
    """
    return f"{split_arm}/{raw_arm}"


def select_matched_pairs(
    matched_pairs: Sequence[Sequence[str]],
    arms: Mapping[str, LoadedTokenizer],
    english_control: Mapping[str, str],
) -> list[tuple[str, str]]:
    """The configured `(raw_arm, split_arm)` pairs whose *both* sides loaded.

    Structure is validated for every configured pair, available or not (a malformed or
    mismatched pair is a config error and must fail even on a run where the arm happens to
    be missing); availability is then filtered with a WARNING. Both arms must also name the
    same `english_control` arm — the delta divides both sides by it, and two different
    denominators would make the difference meaningless.
    """
    selected: list[tuple[str, str]] = []
    for pair in matched_pairs:
        names = list(pair)
        if len(names) != 2:
            raise ValueError(
                f"config['matched_pairs'] entry {names!r} must be [raw_arm, split_arm]"
            )
        raw_arm, split_arm = names
        check_matched_pair(raw_arm, split_arm)
        controls = {english_control.get(raw_arm), english_control.get(split_arm)}
        if None in controls or len(controls) != 1:
            raise ValueError(
                f"matched pair [{raw_arm}, {split_arm}]: both arms must name the same "
                f"english_control arm, got {english_control.get(raw_arm)!r} and "
                f"{english_control.get(split_arm)!r}"
            )
        missing = [name for name in names if name not in arms]
        if missing:
            logger.warning(
                "matched pair %s is skipped: %s unavailable this run",
                matched_pair_key(raw_arm, split_arm),
                ", ".join(missing),
            )
            continue
        selected.append((raw_arm, split_arm))
    return selected


# ------------------------------------------------------------------------- TPP / metrics


def pivots_for(
    arm: str, english_control: Mapping[str, str], deployed_pivot: str
) -> list[tuple[str, str]]:
    """`[(pivot arm, role)]` for one Sanskrit arm: its matched control, then the deployed
    pivot. Raises `ValueError` if the arm has no `english_control` entry, since the
    controlled comparison is the one the verdict is read from and silently dropping it
    would leave a table that looks complete."""
    control = english_control.get(arm)
    if control is None:
        raise ValueError(
            f"config['english_control'] has no matched English arm for {arm!r}; "
            f"known: {sorted(english_control)}"
        )
    return [(control, ROLE_CONTROLLED), (deployed_pivot, ROLE_DEPLOYED)]


def compute_tpp(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    specs: Sequence[ArmSpec],
    english_control: Mapping[str, str],
    deployed_pivot: str,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    """`corpus -> arm -> variant -> pivot -> summary`.

    The variant level is what makes the file self-describing: a `T4` number measured on the
    reconciled text and the same arm's number on the splitter's raw output are different
    measurements of different text, and burying that distinction in a key prefix or a
    separate top-level block would invite reading them as one series. Raw arms have exactly
    one variant, so their entries are one level deeper than they strictly need to be and
    read the same way as everything else.

    Arms and pivots absent from `arms` are skipped rather than stored as `None`; an
    unavailable *pivot* is logged once, before the loops, because it silently removes a
    whole column from every corpus and arm below.
    """
    wanted_pivots = {deployed_pivot} | {
        english_control[spec.name] for spec in specs if spec.name in english_control
    }
    for pivot_name in sorted(wanted_pivots):
        if pivot_name not in arms:
            logger.warning(
                "English pivot %s is unavailable this run; every TPP column against it is "
                "omitted from results.json",
                pivot_name,
            )

    results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    for corpus in corpora:
        corpus_result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for spec in specs:
            tokenizer = arms.get(spec.name)
            if tokenizer is None:
                continue
            variant_result: dict[str, dict[str, dict[str, Any]]] = {}
            for variant in spec.variants:
                pivot_result: dict[str, dict[str, Any]] = {}
                for pivot_name, role in pivots_for(spec.name, english_control, deployed_pivot):
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
                    pivot_result[pivot_name] = summary
                    logger.info(
                        "%s / %s / %s / vs %s (%s): TPP %.3f [%.3f, %.3f]",
                        corpus.name,
                        spec.name,
                        variant,
                        pivot_name,
                        role,
                        raw["value"],
                        raw["ci_low"],
                        raw["ci_high"],
                    )
                variant_result[variant] = pivot_result
            corpus_result[spec.name] = variant_result
        results[corpus.name] = corpus_result
    return results


def compute_tpp_delta(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    pairs: Sequence[tuple[str, str]],
    english_control: Mapping[str, str],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """`corpus -> "<split arm>/<raw arm>" -> delta summary`: the experiment's headline.

    The delta is `TPP(split arm on the reconciled text) - TPP(raw arm on the raw text)`,
    both against the matched English control, over the same sentences, with a paired
    bootstrap (`tpp_paired_delta`). Success for H3's TPP half is a negative delta whose CI
    excludes 0 on prose; a split arm that merely sits below 1.0 says nothing about
    splitting, only about the language pair.

    Pairs whose control pivot is unavailable are skipped with a WARNING.
    """
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        pair_results: dict[str, dict[str, Any]] = {}
        for raw_arm, split_arm in pairs:
            pivot_name = english_control[raw_arm]
            pivot = arms.get(pivot_name)
            if pivot is None:
                logger.warning(
                    "%s: delta for %s is skipped: control pivot %s unavailable this run",
                    corpus.name,
                    matched_pair_key(raw_arm, split_arm),
                    pivot_name,
                )
                continue
            counts_split = token_ratio(
                arms[split_arm], corpus.texts[SPLIT], corpus.english, pivot
            )
            counts_raw = token_ratio(arms[raw_arm], corpus.texts[RAW], corpus.english, pivot)
            entry: dict[str, Any] = dict(
                tpp_paired_delta(
                    counts_split, counts_raw, n_bootstrap=n_bootstrap, seed=seed, ci=ci
                )
            )
            entry["split_arm"] = split_arm
            entry["raw_arm"] = raw_arm
            entry["pivot"] = pivot_name
            entry["label"] = matched_pair_label(raw_arm, split_arm)
            entry["split_variant"] = SPLIT
            saving = counts_raw.source_total - counts_split.source_total
            cost = deletion_cost(
                arms[raw_arm],
                corpus.texts[RAW],
                corpus.texts[SPLIT],
                source_tokens_saved=saving,
            )
            entry["deletion_cost"] = cost
            entry["deletion_cost_tokens"] = cost["deletion_cost_tokens"]
            entry["source_tokens_saved"] = saving
            entry["deletion_cost_share_of_saving"] = cost["share_of_saving"]
            pair_results[matched_pair_key(raw_arm, split_arm)] = entry
            logger.info(
                "%s / %s: delta %+.4f [%+.4f, %+.4f] (split %.3f, raw %.3f, vs %s); "
                "non-letter deletion cost %s token(s) of %d saved",
                corpus.name,
                matched_pair_key(raw_arm, split_arm),
                float(entry["delta"]),
                float(entry["ci_low"]),
                float(entry["ci_high"]),
                float(entry["value_a"]),
                float(entry["value_b"]),
                pivot_name,
                cost["deletion_cost_tokens"],
                int(saving),
            )
        results[corpus.name] = pair_results
    return results


def compute_fertility_compression(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    specs: Sequence[ArmSpec],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """`(fertility_primary, fertility_secondary, compression)`, each `corpus -> arm -> summary`.

    Reported, never headlined (CLAUDE.md §2.1). **Primary** puts every arm over the same
    denominator, the raw sentence's whitespace-word count: plain `fertility` for a raw arm,
    `fertility_against_reference` for a split arm, whose unit therefore reads
    `"tokens/reference word"` while the number remains comparable. The two metrics' own
    `n_undefined` counts are deliberately not carried into the summaries: they count
    different things (words that encoded to nothing, versus sentences whose reference had
    no words), and one key with two meanings across a table is worse than no key.

    **`value` is comparable across both; `mean` and `std` are not.** The two metrics
    attach different distributions, so `summarise_metric` summarises different lists: for a
    raw arm `mean`/`std` describe the spread of *per-word token counts* (`per_word`, one
    entry per word), and for a split arm the spread of *per-sentence ratios* (`per_text`,
    one entry per sentence). Only the pooled `value` — tokens over raw words, both ways —
    means the same thing in both rows, and it is the only one a table may compare. Each
    summary therefore records `distribution` (which list was summarised), `n` (the
    denominator, raw words in both cases) and `n_texts` (how many sentences the arm was
    measured over, which for a split arm is also the length of the summarised list).

    **Secondary** is plain
    `fertility` on the split text, i.e. tokens per *split* word — a different denominator,
    and the reason the primary exists (docs/decisions.md, "Fertility for split arms uses
    the raw word count as the primary denominator"). Raw arms have no secondary entry:
    their primary already is plain fertility on the text they tokenize, and storing the
    same number twice would invite reading the two tables as comparable columns.

    Compression is measured on whatever text the arm actually tokenizes; every entry
    records that `variant` explicitly.
    """
    primary: dict[str, dict[str, dict[str, Any]]] = {}
    secondary: dict[str, dict[str, dict[str, Any]]] = {}
    comp: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus in corpora:
        primary[corpus.name] = {}
        secondary[corpus.name] = {}
        comp[corpus.name] = {}
        for spec in specs:
            tokenizer = arms.get(spec.name)
            if tokenizer is None:
                continue
            texts = corpus.texts[spec.primary]
            if spec.kind == SPLIT_ARM:
                primary_result = fertility_against_reference(tokenizer, texts, corpus.texts[RAW])
                secondary[corpus.name][spec.name] = {
                    **summarise_metric(fertility(tokenizer, texts)),
                    "variant": spec.primary,
                    "n_texts": len(texts),
                }
            else:
                primary_result = fertility(tokenizer, texts)
            primary[corpus.name][spec.name] = {
                **summarise_metric(primary_result),
                "variant": spec.primary,
                "n_texts": len(texts),
            }
            comp[corpus.name][spec.name] = {
                **summarise_metric(compression(tokenizer, texts)),
                "variant": spec.primary,
            }
    return primary, secondary, comp


# ------------------------------------------------------------------- splitter statistics


def load_manifest(path: Path) -> dict[str, Any]:
    """The split manifest, or an abort naming the script that writes it.

    Without it `results.json` could not say which model and revision produced the text
    every `T4` number is measured on, which is exactly the provenance CLAUDE.md §2.9 and
    §11 exist to preserve — so a missing manifest stops the run rather than producing a
    file whose splitter fields read "unknown".
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no split manifest at {path}. Run "
            "`uv run python experiments/03_sandhi_split/split_corpora.py` first; it writes "
            "the manifest when every corpus is split."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(manifest).__name__}")
    return manifest


def splitter_provenance(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    """The top-level `splitter` block: which model, at which revision, on what, how lossily.

    The counters here (`n_chunked`, `n_model`, `n_cache_hits`) are run-wide over the whole
    split job — every evaluation corpus and the training corpus — not per corpus; the
    per-corpus numbers are in `splitter_stats`.
    """
    return {
        "source_id": str(manifest.get("splitter_source_id", "unknown")),
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
        "device": manifest.get("device"),
        "batch_size": manifest.get("batch_size"),
        "reconcile_threshold": manifest.get("reconcile_threshold"),
        "subset": manifest.get("subset"),
        "n_chunked": manifest.get("n_chunked"),
        "n_model": manifest.get("n_model"),
        "n_cache_hits": manifest.get("n_cache_hits"),
        "seconds": manifest.get("seconds"),
        "char_retention_definition": manifest.get("char_retention_definition"),
        "manifest_path": str(manifest_path),
        "manifest_git_commit": manifest.get("git_commit"),
        "manifest_timestamp": manifest.get("timestamp"),
    }


def _mean_units(texts: Sequence[str]) -> float | None:
    """Mean whitespace-unit count over `texts`, or `None` for an empty corpus."""
    if not texts:
        return None
    return float(np.mean([len(text.split()) for text in texts]))


def splitter_stats(
    corpora: Sequence[CorpusData], manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """`corpus -> descriptive statistics about what splitting did to that corpus`.

    Two sources, kept apart. `manifest` is what the split job recorded over the sentences
    it split (character retention before and after reconciliation, units kept verbatim,
    units replaced inexactly). The rest is computed here over the sentences this run
    actually evaluates, which is the same set but recomputed rather than trusted: mean
    whitespace units raw, reconciled and unreconciled, and the share of sentences whose
    unit count changed.

    `invariants` is `experiment.text_invariants` recomputed here over the evaluated
    sentences, and a corpus whose non-letter multiset is not preserved logs a WARNING: that
    is the condition under which a `T4` arm is being credited for deleted characters rather
    than for inserted boundaries, and it is not something to discover in a table.

    Raises `ValueError` if the manifest does not cover a corpus being evaluated — the two
    artifacts would then describe different runs, and a table combining them would be
    quietly wrong.
    """
    per_corpus = manifest.get("corpora", {})
    stats: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        entry = per_corpus.get(corpus.name)
        if entry is None:
            raise ValueError(
                f"{corpus.name}: the split manifest does not cover this corpus; it names "
                f"{sorted(per_corpus)}. The manifest and the corpus jsonl files must come "
                "from the same split run."
            )
        invariants = text_invariants(corpus.texts[RAW], corpus.texts[SPLIT])
        if not invariants["nonletter_multiset_preserved"]:
            logger.warning(
                "%s: the reconciled text does not preserve the non-letter character "
                "multiset (%d raw vs %d out; missing %s, added %s). Whatever is missing is "
                "content the split arm was not charged for: read deletion_cost_tokens "
                "before reading the delta",
                corpus.name,
                invariants["nonletter_chars_raw"],
                invariants["nonletter_chars_out"],
                dict(list(invariants["nonletter_missing"].items())[:6]),
                dict(list(invariants["nonletter_added"].items())[:6]),
            )
        raw_units = [len(text.split()) for text in corpus.texts[RAW]]
        split_units = [len(text.split()) for text in corpus.texts[SPLIT]]
        n_changed = sum(
            1 for raw, split in zip(raw_units, split_units, strict=True) if raw != split
        )
        n = len(raw_units)
        stats[corpus.name] = {
            "n_sentences_evaluated": n,
            "mean_units_raw": _mean_units(corpus.texts[RAW]),
            "mean_units_split": _mean_units(corpus.texts[SPLIT]),
            "mean_units_split_model": _mean_units(corpus.texts[SPLIT_MODEL]),
            "n_units_raw_evaluated": sum(raw_units),
            "n_units_split_evaluated": sum(split_units),
            "n_sentences_units_changed": n_changed,
            "fraction_units_changed": n_changed / n if n else math.nan,
            "invariants": invariants,
            "manifest": {key: entry[key] for key in MANIFEST_CORPUS_KEYS if key in entry},
        }
    return stats


# ------------------------------------------------------------------------------ figure


def _series(
    corpus_tpp: Mapping[str, Any],
    arm: str,
    variant: str,
    pivot: str,
) -> tuple[float, float, float]:
    """`(value, lower error, upper error)` for one arm's controlled TPP, `nan` if absent."""
    entry = corpus_tpp.get(arm, {}).get(variant, {}).get(pivot)
    if entry is None or entry.get("value") is None:
        return math.nan, 0.0, 0.0
    value = float(entry["value"])
    ci_low, ci_high = entry.get("ci_low"), entry.get("ci_high")
    return (
        value,
        value - ci_low if ci_low is not None else 0.0,
        ci_high - value if ci_high is not None else 0.0,
    )


def _build_figure(results: Mapping[str, Any]) -> Any:
    """Build (but do not save or close) the split-vs-raw figure; returns the `Figure`.

    One row per corpus in config order (prose first, CLAUDE.md §2.7), one x position per
    matched pair, two markers at each: the raw arm and the split arm, both as controlled
    TPP against the E1 arm they share, both with their 95% bootstrap CI. Reading the two
    markers against each other is the whole experiment — the dashed line at 1.0 is the
    English-parity reference, not the comparison, since an arm can sit either side of it
    for reasons that have nothing to do with splitting.

    The interval drawn is each arm's own; the interval on the *difference* is narrower
    (the two are paired) and lives in `tpp_delta` and the README table, so overlapping
    error bars here do not by themselves mean the delta is indistinguishable from zero.
    The caption says which splitter produced the split text and names any omitted arm.

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
    pairs = [(str(pair[0]), str(pair[1])) for pair in config["matched_pairs"]]
    english_control = dict(config["english_control"])
    tpp_results = results["tpp"]

    figure, axes_grid = plt.subplots(
        len(corpus_entries), 1, figsize=(7.2, 3.0 * len(corpus_entries)), squeeze=False
    )

    for panel_index, (row, entry) in enumerate(zip(axes_grid, corpus_entries, strict=True)):
        axes = row[0]
        corpus_name = str(entry["name"])
        corpus_tpp = tpp_results.get(corpus_name, {})

        labels: list[str] = []
        raw_values: list[float] = []
        raw_err: tuple[list[float], list[float]] = ([], [])
        split_values: list[float] = []
        split_err: tuple[list[float], list[float]] = ([], [])
        for raw_arm, split_arm in pairs:
            pivot = str(english_control.get(raw_arm, ""))
            if raw_arm not in corpus_tpp and split_arm not in corpus_tpp:
                continue
            labels.append(matched_pair_label(raw_arm, split_arm))
            value, lower, upper = _series(corpus_tpp, raw_arm, RAW, pivot)
            raw_values.append(value)
            raw_err[0].append(lower)
            raw_err[1].append(upper)
            value, lower, upper = _series(corpus_tpp, split_arm, SPLIT, pivot)
            split_values.append(value)
            split_err[0].append(lower)
            split_err[1].append(upper)

        positions = np.arange(len(labels), dtype=float)
        axes.errorbar(
            positions - FIGURE_MARKER_OFFSET,
            raw_values,
            yerr=list(raw_err),
            fmt="o",
            capsize=3,
            color="#2b6cb0",
            label="raw text (T1/T2)",
            zorder=3,
        )
        axes.errorbar(
            positions + FIGURE_MARKER_OFFSET,
            split_values,
            yerr=list(split_err),
            fmt="s",
            capsize=3,
            color="#c05621",
            label="sandhi-split (T4)",
            zorder=3,
        )
        axes.axhline(1.0, linestyle="--", color="gray", linewidth=1)

        # Every bound that goes into the y-range must be finite, not merely non-`nan`: an
        # arm whose pivot side tokenized to nothing gives an infinite ratio, and one `inf`
        # among the bounds makes `set_ylim` either raise or collapse the panel.
        finite: list[float] = []
        for values, errors in ((raw_values, raw_err), (split_values, split_err)):
            for value, lower, upper in zip(values, errors[0], errors[1], strict=True):
                bounds = (value - lower, value + upper)
                if math.isfinite(value) and all(math.isfinite(bound) for bound in bounds):
                    finite.extend(bounds)
        if finite:
            span_low, span_high = min([*finite, 1.0]), max([*finite, 1.0])
            span = span_high - span_low
            pad = span * FIGURE_Y_PAD_FRACTION if span > 0 else max(span_high * 0.2, 0.1)
            axes.set_ylim(span_low - pad, span_high + pad)

        axes.set_xticks(positions)
        axes.set_xticklabels(labels, fontsize=8)
        axes.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
        axes.set_ylabel("TPP ratio (Sanskrit / English)", fontsize=9)
        axes.set_title(corpus_name, fontsize=9, loc="left")
        axes.spines[["top", "right"]].set_visible(False)
        if panel_index == 0:
            axes.legend(fontsize=7, loc="best")

    splitter_id = str(results.get("splitter", {}).get("source_id", "unknown"))
    caption = f"{FIGURE_CAPTION_PROVISIONAL} Splitter: {splitter_id}."
    omitted = unavailable_caption(results.get("unavailable_arms", {}))
    if omitted:
        caption = f"{caption}  {omitted}"

    figure.suptitle(FIGURE_SUPTITLE, fontsize=11)
    figure.text(0.5, 0.955, FIGURE_SUBTITLE, ha="center", fontsize=8)
    figure.text(0.01, 0.005, caption, fontsize=6, wrap=True)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    return figure


def make_figure(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Build the split-vs-raw figure (`_build_figure`) and save it as PDF and PNG."""
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


def run(config: Mapping[str, Any], config_src: Path | None = None) -> dict[str, Any]:
    """Run the whole experiment from a parsed config; returns the results dict.

    Split out from `main` so the end-to-end path — corpora, leakage checks, every metric,
    `results.json`, the figure — is exercisable from a test with a temporary config, fake
    tokenizers and a synthetic split cache, offline.
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
    arms_raw = [str(name) for name in config["arms_raw"]]
    arms_split = [str(name) for name in config["arms_split"]]
    matched_pairs = [list(pair) for pair in config["matched_pairs"]]
    english_control = {str(key): str(value) for key, value in config["english_control"].items()}
    deployed_pivot = str(config["deployed_pivot"])

    split_dir = resolve_path(str(config["split_dir"]), root)
    manifest_path = resolve_path(
        str(config.get("split_manifest_path", split_dir / "manifest.json")), root
    )
    out_dir = resolve_path(str(config["output_dir"]), root)
    exclusion_path = resolve_path(str(config["exclusion_path"]), root)
    exclusion_path_en = resolve_path(str(config["exclusion_path_en"]), root)

    manifest = load_manifest(manifest_path)
    corpora = [prepare_corpus(entry, root, split_dir) for entry in corpora_config]

    hashes = load_exclusion_hashes(exclusion_path)
    hashes_en = load_exclusion_hashes(exclusion_path_en)
    exclusion_check: dict[str, dict[str, int]] = {}
    exclusion_check_en: dict[str, dict[str, int]] = {}
    for corpus in corpora:
        report = exclusion_check_for(corpus.raw_deva, hashes)
        exclusion_check[corpus.name] = report
        if report["n_missing"]:
            logger.warning(
                "%s: %d/%d Sanskrit sentences are NOT in the exclusion list "
                "(data/exclusion_hashes.txt may be stale)",
                corpus.name,
                report["n_missing"],
                report["n"],
            )
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

    specs = arm_specs(arms_raw, arms_split)
    # `pivots_for` also validates that every arm has a matched control, so a config short a
    # mapping fails here rather than after the corpora and the tokenizers are loaded.
    all_arm_names = sorted(
        {spec.name for spec in specs}
        | {deployed_pivot}
        | {
            pivot
            for spec in specs
            for pivot, _ in pivots_for(spec.name, english_control, deployed_pivot)
        }
    )
    arms, unavailable_arms = load_arms(all_arm_names)

    tpp_results = compute_tpp(
        corpora,
        arms,
        specs,
        english_control,
        deployed_pivot,
        n_bootstrap,
        seed,
        ci,
    )
    tpp_delta = compute_tpp_delta(
        corpora,
        arms,
        select_matched_pairs(matched_pairs, arms, english_control),
        english_control,
        n_bootstrap,
        seed,
        ci,
    )
    fertility_primary, fertility_secondary, compression_results = compute_fertility_compression(
        corpora, arms, specs
    )

    splitter = splitter_provenance(manifest, manifest_path)
    results: dict[str, Any] = {
        "experiment": str(config.get("experiment", out_dir.name)),
        **provenance(root),
        "config": dict(config),
        "splitter": splitter,
        "tokenizer_sources": tokenizer_sources(arms, splitter["source_id"]),
        "unavailable_arms": unavailable_arms,
        "corpora": {
            corpus.name: {"split": corpus.split, "n_total": corpus.n_total, "n_used": corpus.n_used}
            for corpus in corpora
        },
        "exclusion_check": exclusion_check,
        "exclusion_check_en": exclusion_check_en,
        "splitter_stats": splitter_stats(corpora, manifest),
        "tpp": tpp_results,
        "tpp_delta": tpp_delta,
        "fertility_primary": fertility_primary,
        "fertility_secondary": fertility_secondary,
        "compression": compression_results,
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
