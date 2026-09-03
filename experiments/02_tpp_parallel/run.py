"""Experiment 02 — tokens-per-proposition on parallel text (RQ2).

Hypothesis (outline §1, H2): with a Sanskrit-native tokenizer, tokens-per-proposition
(TPP) on parallel corpora is lower for Sanskrit than English; with English-centric
tokenizers it is higher; the sign flips depending on tokenizer.

TPP is the headline metric of this project (CLAUDE.md §7): fertility asks how many
tokens a *word* costs, which punishes Sanskrit for the very density under study (sandhi,
compounding); TPP asks how many tokens the *same proposition* costs on aligned
translation pairs. This script computes TPP, with a paired bootstrap CI, for every
Sanskrit tokenizer arm against English (o200k primary pivot, Llama-4 secondary) on four
corpora, prose before verse (CLAUDE.md §2.7): Sāmayik test and test_ood (primary, prose),
Itihāsa test (secondary, verse — meter is a confound), FLORES devtest (tertiary). It also
computes a Hindi-pivot TPP on FLORES for the T0/T3 arms, and fertility/compression on the
Sanskrit side of every corpus x arm x script variant, reported but never headlined.

Run it with `uv run python experiments/02_tpp_parallel/run.py`. Relative paths in the
config are resolved against the repository root, so the working directory does not
matter. Every function below is pure or takes its I/O paths explicitly, so
`tests/test_exp02.py` can exercise the aggregation and plotting logic on synthetic data
without touching the network, a real corpus, or the Hugging Face cache.
"""

import argparse
import json
import logging
import math
import random
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash
from sanskrit_tok.data.flores import ParallelCorpus, load_jsonl, save_jsonl
from sanskrit_tok.data.itihasa import load_itihasa
from sanskrit_tok.data.samayik import load_samayik
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility
from sanskrit_tok.metrics.summary import summarise_metric
from sanskrit_tok.metrics.tpp import tpp
from sanskrit_tok.tokenizers.registry import LoadedTokenizer, TokenizerUnavailable, load_tokenizer

logger = logging.getLogger("exp02")

#: Script variant names, matching exp01's constants.
ORIGINAL = "original"
SLP1 = "slp1"

#: Languages every corpus in this experiment carries.
SANSKRIT_LANGUAGE = "san_Deva"
ENGLISH_LANGUAGE = "eng_Latn"
HINDI_LANGUAGE = "hin_Deva"

#: Tokenizer-arm families whose T0/T3-style provisional flag the figure and README mark
#: with a `*`: trained from scratch on the parallel-corpus training splits, not yet on
#: the monolingual corpus (docs/decisions.md, "Provisional T1/T2 tokenizers...").
PROVISIONAL_FAMILIES = frozenset({"T1", "T2"})

#: Tokenizer-arm families eligible for the Hindi pivot (CLAUDE.md §7 resolution 4:
#: T0/T3 arms only, same tokenizer both sides).
HINDI_PIVOT_FAMILIES = frozenset({"T0", "T3"})

FIGURE_STEM = "tpp_by_arm"
#: The pivot and script variant the figure's main marker reads, so every arm — T0/T3
#: (which also have an `original` variant) and T1/T2 (slp1 only) — sits on the same
#: footing (config.yaml resolution 6).
FIGURE_PIVOT = "T0_o200k"
FIGURE_MAIN_VARIANT = SLP1
FIGURE_SECONDARY_VARIANT = ORIGINAL
FIGURE_SUPTITLE = "Tokens per proposition: Sanskrit vs English (o200k), 95% bootstrap CI"
FIGURE_CAPTION = "* provisional: trained on parallel-corpus training splits"

#: `DetailedMetricResult` keys `summarise_metric` drops (it keeps only value/n/unit/
#: distribution/mean/std); the brief requires them copied from the raw `tpp` result into
#: the stored summary by hand.
_TPP_EXTRA_KEYS: tuple[str, ...] = (
    "ci_low",
    "ci_high",
    "n_undefined",
    "n_bootstrap",
    "seed",
    "source_tokens",
    "pivot_tokens",
)


# ------------------------------------------------------------------- paths and config


def repo_root() -> Path:
    """The repository root, i.e. the parent of `experiments/`."""
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    """Resolve a config path: absolute ones as given, relative ones against `root`."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict[str, Any]:
    """Read the experiment's YAML config (`yaml.safe_load` only)."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(config).__name__}")
    return config


def git_commit(root: Path) -> str:
    """`git rev-parse HEAD`, or `"unknown"` (logged at WARNING) if that fails."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("could not read the git commit (%s); recording 'unknown'", error)
        return "unknown"
    return completed.stdout.strip() or "unknown"


def sanitize_json(value: Any) -> Any:
    """Recursively replace `nan`/`inf` floats with `None`, so `results.json` is strict.

    `json.dump(..., allow_nan=False)` raises on a non-finite float rather than emitting
    the non-standard `NaN`/`Infinity` tokens JSON forbids; several metrics in this
    experiment produce `nan` on purpose (an undefined per-pair ratio, a bootstrap CI over
    zero draws), so the tree is sanitised once, here, before it is ever written.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


# ---------------------------------------------------------------------- corpus wrangling


def select_aligned_indices(sentences: Mapping[str, Sequence[str]]) -> list[int]:
    """Indices whose sentence is non-blank in *every* language (exp01's helper, copied:
    each experiment script is self-contained, per its existing convention)."""
    per_language = {language: len(values) for language, values in sentences.items()}
    lengths = set(per_language.values())
    if len(lengths) > 1:
        raise ValueError(f"languages differ in length: {per_language}")
    total = lengths.pop() if lengths else 0
    return [
        index
        for index in range(total)
        if all(sentences[language][index].strip() for language in sentences)
    ]


def take_indices(
    sentences: Mapping[str, Sequence[str]], indices: Sequence[int]
) -> dict[str, list[str]]:
    """Keep `indices`, in order, from every language at once."""
    return {
        language: [values[index] for index in indices] for language, values in sentences.items()
    }


def load_corpus_entry(entry: Mapping[str, Any], root: Path) -> ParallelCorpus:
    """Dispatch one `config["corpora"]` entry to its loader, by `entry["loader"]`."""
    loader = str(entry["loader"])
    split = str(entry["split"])
    if loader == "samayik":
        return load_samayik(split)  # type: ignore[arg-type]
    if loader == "itihasa":
        return load_itihasa(split)  # type: ignore[arg-type]
    if loader == "flores":
        jsonl_path = resolve_path(str(entry["jsonl"]), root)
        if jsonl_path.exists():
            return load_jsonl(jsonl_path, name="flores200", split=split)
        from sanskrit_tok.data.flores import load_flores

        logger.info("%s not found; downloading FLORES-200 %s", jsonl_path, split)
        corpus = load_flores((SANSKRIT_LANGUAGE, HINDI_LANGUAGE, ENGLISH_LANGUAGE), split)
        save_jsonl(corpus, jsonl_path)
        return corpus
    raise ValueError(f"corpus {entry.get('name')!r}: unknown loader {loader!r}")


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


def exclusion_check_for(sentences: Sequence[str], hashes: frozenset[str]) -> dict[str, int]:
    """`{"n": len(sentences), "n_missing": how many hash to something not in `hashes`}`.

    Every Sanskrit evaluation sentence used by this experiment is expected to be in
    `data/exclusion_hashes.txt` (CLAUDE.md §2.4): a non-zero `n_missing` means this
    corpus's Sanskrit side (or some of it) was not included when the exclusion list was
    built, which is worth a WARNING but not an abort — this experiment reads evaluation
    text, it does not train anything, so there is nothing here for a missed hash to leak
    into.
    """
    n_missing = sum(1 for sentence in sentences if sentence_hash(sentence) not in hashes)
    return {"n": len(sentences), "n_missing": n_missing}


# ------------------------------------------------------------------------ tokenizers


def load_arms(
    names: Sequence[str],
) -> tuple[dict[str, LoadedTokenizer], dict[str, str]]:
    """Load every arm in `names`, once each; split into loaded and `unavailable_arms`."""
    loaded: dict[str, LoadedTokenizer] = {}
    unavailable: dict[str, str] = {}
    for name in names:
        if name in loaded or name in unavailable:
            continue
        try:
            loaded[name] = load_tokenizer(name)
        except TokenizerUnavailable as error:
            logger.warning("%s: unavailable this run: %s", name, error)
            unavailable[name] = str(error)
    return loaded, unavailable


def variants_for_family(family: str, script_variants: Mapping[str, Sequence[str]]) -> list[str]:
    """The script variants an arm's family is measured in, per `config["script_variants"]`."""
    return list(script_variants[family])


# ------------------------------------------------------------------------- TPP / metrics


def _enrich_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """`summarise_metric(raw)` plus the bootstrap/undefined-count keys it drops.

    `summarise_metric` (CLAUDE.md §7 contract) keeps only `value`/`n`/`unit`/
    `distribution`/`mean`/`std`; `tpp`'s `ci_low`, `ci_high`, `n_undefined`,
    `n_bootstrap`, `seed`, `source_tokens` and `pivot_tokens` are the caller's to record
    alongside it (`summary.py`'s own docstring says as much), which is what this does.
    """
    summary = dict(summarise_metric(raw))
    for key in _TPP_EXTRA_KEYS:
        summary[key] = raw[key]
    return summary


def compute_tpp(
    corpora: Sequence[CorpusData],
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    english_pivots: Sequence[str],
    script_variants: Mapping[str, Sequence[str]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    """`corpus -> arm -> variant -> pivot -> summary`, per the brief's `results.json` shape.

    Arms absent from `arms` (unavailable this run) and pivots absent from `arms` are
    silently skipped — `unavailable_arms` already records why, and this keeps the nested
    dict free of `None` placeholders that every consumer would otherwise have to check.
    """
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
                    )
                    pivot_result[pivot_name] = _enrich_summary(raw)
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


def compute_tpp_hindi(
    corpus: CorpusData | None,
    arms: Mapping[str, LoadedTokenizer],
    sanskrit_arm_names: Sequence[str],
    script_variants: Mapping[str, Sequence[str]],
    n_bootstrap: int,
    seed: int,
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
            )
            variant_result[variant] = _enrich_summary(raw)
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


# ------------------------------------------------------------------------------ figure


def arm_label(name: str, vocab_size: int) -> str:
    """X-tick label: arm name, a `*` for provisional (T1/T2) arms, vocab size in `Nk`.

    `T1_bpe_raw_32k` -> `"T1_bpe_raw_32k* (32k)"`; `T0_o200k` -> `"T0_o200k (200k)"`.
    """
    family = name.split("_", 1)[0]
    star = "*" if family in PROVISIONAL_FAMILIES else ""
    thousands = round(vocab_size / 1000)
    return f"{name}{star} ({thousands}k)"


def make_figure(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Four panels stacked vertically, one per corpus in config order (prose first).

    Each panel plots, for every arm present in that corpus's `tpp` entry (unavailable
    arms have none and are silently omitted from the x-axis): the TPP of the SLP1
    variant against `T0_o200k` as a point with a 95% bootstrap-CI error bar (every arm
    has an SLP1 variant, so this puts T0/T3/T1/T2 on the same footing), plus — for T0/T3
    arms, which also carry an `original`-script variant — a second, thin marker at the
    same x position showing the un-transliterated number, so the transliteration effect
    is visible. A dashed line at 1.0 marks the sign flip TPP is testing for. Provisional
    (T1/T2) arms carry a `*` in their tick label; the caption explains it.
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

    out_dir.mkdir(parents=True, exist_ok=True)
    n_panels = len(corpus_entries)
    figure, axes_grid = plt.subplots(n_panels, 1, figsize=(9.5, 3.2 * n_panels), squeeze=False)
    axes_list = [row[0] for row in axes_grid]

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
            if main_entry is not None and main_entry["value"] is not None:
                value = float(main_entry["value"])
                ci_low = main_entry.get("ci_low")
                ci_high = main_entry.get("ci_high")
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
            if secondary_entry is not None and secondary_entry["value"] is not None:
                secondary_x.append(position)
                secondary_values.append(float(secondary_entry["value"]))

        positions = list(range(len(available_arms)))
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
        if secondary_values:
            axes.scatter(
                secondary_x,
                secondary_values,
                marker="_",
                s=90,
                linewidths=2,
                color="#c05621",
                label="original script (T0/T3 only)",
                zorder=2,
            )
        axes.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axes.set_xlim(-0.5, max(len(positions) - 0.5, 0.5))
        axes.set_ylabel("TPP ratio", fontsize=9)
        axes.set_title(corpus_name, fontsize=9, loc="left")
        axes.spines[["top", "right"]].set_visible(False)
        if panel_index == 0:
            axes.legend(fontsize=7, loc="best")

    figure.suptitle(FIGURE_SUPTITLE, fontsize=11)
    figure.text(0.01, 0.005, FIGURE_CAPTION, fontsize=7)
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))

    paths = [out_dir / f"{FIGURE_STEM}.pdf", out_dir / f"{FIGURE_STEM}.png"]
    for path in paths:
        figure.savefig(path, dpi=200)
    plt.close(figure)
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

    corpora_config: list[dict[str, Any]] = list(config["corpora"])
    if not corpora_config:
        raise ValueError(f"{args.config}: 'corpora' is empty; nothing to measure")
    sanskrit_arm_names: list[str] = list(config["sanskrit_arms"])
    english_pivots: list[str] = list(config["english_pivots"])
    hindi_pivot_corpus_name = str(config["hindi_pivot_corpus"])
    script_variants: dict[str, list[str]] = {
        family: list(variants) for family, variants in config["script_variants"].items()
    }
    out_dir = resolve_path(str(config["output_dir"]), root)
    exclusion_path = resolve_path(str(config["exclusion_path"]), root)

    corpora = [prepare_corpus(entry, root) for entry in corpora_config]

    hashes = load_exclusion_hashes(exclusion_path)
    exclusion_check: dict[str, dict[str, int]] = {}
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

    all_arm_names = sorted(set(sanskrit_arm_names) | set(english_pivots))
    arms, unavailable_arms = load_arms(all_arm_names)

    tpp_results = compute_tpp(
        corpora, arms, sanskrit_arm_names, english_pivots, script_variants, n_bootstrap, seed
    )
    hindi_corpus = next((c for c in corpora if c.name == hindi_pivot_corpus_name), None)
    if hindi_corpus is None:
        logger.warning(
            "hindi_pivot_corpus %r is not among the configured corpora; "
            "tpp_hindi will be empty",
            hindi_pivot_corpus_name,
        )
    tpp_hindi = compute_tpp_hindi(
        hindi_corpus, arms, sanskrit_arm_names, script_variants, n_bootstrap, seed
    )
    fertility_results, compression_results = compute_fertility_compression(
        corpora, arms, sanskrit_arm_names, script_variants
    )

    results: dict[str, Any] = {
        "experiment": str(config.get("experiment", out_dir.name)),
        "git_commit": git_commit(root),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "tokenizer_sources": {
            name: {
                "source_id": tokenizer.source_id,
                "vocab_size": tokenizer.vocab_size,
                "family": tokenizer.family,
                "attempted": list(tokenizer.attempted),
            }
            for name, tokenizer in arms.items()
        },
        "unavailable_arms": unavailable_arms,
        "corpora": {
            corpus.name: {"split": corpus.split, "n_total": corpus.n_total, "n_used": corpus.n_used}
            for corpus in corpora
        },
        "exclusion_check": exclusion_check,
        "tpp": tpp_results,
        "tpp_hindi": tpp_hindi,
        "fertility": fertility_results,
        "compression": compression_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(results), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    logger.info("wrote %s", results_path)

    shutil.copyfile(args.config, out_dir / "config.yaml")
    logger.info("copied %s to %s", args.config, out_dir / "config.yaml")

    make_figure(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
