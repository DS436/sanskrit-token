"""Experiment 01 — baseline penalty (RQ1): what off-the-shelf tokenizers cost Sanskrit.

Hypothesis (CLAUDE.md §10): English-centric tokenizers produce fertility > 5 on Sanskrit,
and the Sanskrit/Hindi parity ratio is > 1.5 on identical FLORES content, because sandhi
merges what Hindi keeps separate.

What this script does, for each T0 arm × language × script variant on FLORES-200 devtest:
fertility and compression; and for each arm, parity of `san_Deva` against each pivot.
The three arms have different vocabulary sizes, so `results.json` records each one's
`vocab_size` and the whole table is *existing practice*, never a controlled comparison
(CLAUDE.md §2.5). Parity is the number to read first; fertility is reported because it is
the standard number in the literature, not because it is the headline (CLAUDE.md §2.1).

Run it with `uv run python experiments/01_baseline_penalty/run.py`. Relative paths in the
config are resolved against the repository root, so the working directory does not matter.

Every function below is pure or takes its I/O paths explicitly, so `tests/test_exp01.py`
can exercise the aggregation, filtering and plotting on synthetic data without touching
the network, the corpus, or the Hugging Face cache.
"""

import argparse
import json
import logging
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sanskrit_tok.data.flores import ParallelCorpus, load_flores, load_jsonl, save_jsonl
from sanskrit_tok.encoding import roundtrip_ok, to_slp1
from sanskrit_tok.metrics.compression import compression
from sanskrit_tok.metrics.fertility import fertility
from sanskrit_tok.metrics.parity import parity
from sanskrit_tok.tokenizers.base import DetailedMetricResult, Tokenizer
from sanskrit_tok.tokenizers.registry import load_tokenizer

logger = logging.getLogger("exp01")

#: The distribution keys a metric may attach on top of the `MetricResult` contract
#: (CLAUDE.md §7). They are summarised into `mean`/`std` and dropped: 1012 sentences ×
#: three arms × two scripts would otherwise put roughly a million floats in `results.json`.
DISTRIBUTION_KEYS: tuple[str, ...] = ("per_word", "per_text", "per_pair")

#: The script variant every language has. Devanagari languages get `"slp1"` as well.
ORIGINAL = "original"
SLP1 = "slp1"

#: Language whose sentences the SLP1 roundtrip assertion is run over. Devanagari-specific:
#: `roundtrip_ok(..., "devanagari")` is meaningless for `eng_Latn`.
ROUNDTRIP_LANGUAGE = "san_Deva"

#: Numerator of every parity ratio; the pivots in `config["pivots"]` are the denominators.
#: Deliberately a separate constant from `ROUNDTRIP_LANGUAGE` even though the two currently
#: hold the same value: one names the language this experiment measures the cost *of*, the
#: other names a transliteration check on Devanagari input. Overridable per config.
PARITY_SOURCE = "san_Deva"

FIGURE_STEM = "fertility_by_language"
FIGURE_YLABEL = "Fertility (tokens per whitespace word)"
FIGURE_TITLE = (
    "FLORES-200 devtest: fertility by language (original script)\n"
    "Reported for comparability with the literature; read parity and BPC first"
)

_LATIN = re.compile(r"[A-Za-z]")
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")


# ------------------------------------------------------------------- paths and config


def repo_root() -> Path:
    """The repository root, i.e. the parent of `experiments/`.

    Derived from this file's location rather than from the working directory, so
    `uv run python experiments/01_baseline_penalty/run.py` behaves identically from
    anywhere in the repo.
    """
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
    """`git rev-parse HEAD`, or `"unknown"` (logged at WARNING) if that fails.

    Recorded in `results.json` so every number can be traced to the code that produced it
    (CLAUDE.md §8). A missing commit is not fatal: it must not stop an experiment run in a
    tarball or a worktree without git.
    """
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


# ---------------------------------------------------------------------- corpus wrangling


def get_corpus(
    languages: Sequence[str],
    split: str,
    jsonl_path: Path,
) -> ParallelCorpus:
    """Load the aligned corpus from `jsonl_path`, downloading it once if it is absent.

    The jsonl is the offline path (`data/raw/`, gitignored): the first run downloads
    FLORES-200 and saves it, and every later run — and every experiment — reads the same
    bytes, so the corpus cannot drift between runs.
    """
    if jsonl_path.exists():
        return load_jsonl(jsonl_path, split=split)
    logger.info("%s not found; downloading FLORES-200 %s", jsonl_path, split)
    corpus = load_flores(languages, split, cache_dir=jsonl_path.parent)
    save_jsonl(corpus, jsonl_path)
    return corpus


def select_languages(corpus: ParallelCorpus, languages: Sequence[str]) -> dict[str, list[str]]:
    """The requested languages, in the requested order, from a possibly wider corpus."""
    missing = [language for language in languages if language not in corpus.sentences]
    if missing:
        raise KeyError(
            f"corpus {corpus.name}/{corpus.split} has no sentences for {missing}; "
            f"it holds {list(corpus.languages)}"
        )
    return {language: list(corpus.sentences[language]) for language in languages}


def select_aligned_indices(sentences: Mapping[str, Sequence[str]]) -> list[int]:
    """Indices whose sentence is non-blank in *every* language.

    Filtering the shared index rather than each language separately is what keeps the
    languages aligned; parity and TPP are meaningless the moment they are not.
    """
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


def roundtrip_report(sentences: Sequence[str], max_examples: int = 5) -> dict[str, Any]:
    """Check Devanagari -> SLP1 -> Devanagari identity over `sentences`.

    SLP1 is ASCII, so a sentence containing Latin letters cannot roundtrip by
    construction: `from_slp1` has no way to tell an English word from a run of SLP1
    phonemes (see `sanskrit_tok.encoding`, and the 2026-09-03 decision log entry). Those
    failures are expected, and are counted apart from the rest.

    Failures are bucketed twice, because the obvious cut is not tight enough:

    - `failures_pure_devanagari` counts the failures with no Latin **letter** in them.
      That is not the same as "no ASCII": FLORES writes years and quantities with ASCII
      digits, which `from_slp1` renders as Devanagari digits, so most of this bucket is
      digits rather than anything to do with Devanagari.
    - `failures_no_ascii_alnum` narrows it to failures with no ASCII letter or digit at
      all. What is left is dominated by punctuation SLP1 claims as phonemes (ASCII `.`
      and `|` standing in for danda) plus the odd nukta; `examples_no_ascii_alnum` lists
      up to `max_examples` of them verbatim so the cause can be read off directly.

    Reporting, not asserting: these failures must not stop the experiment. Every metric
    below is computed on the original script *and* on SLP1 and both are reported, and the
    corpus stores the original script alongside (CLAUDE.md §2.3), so nothing downstream
    depends on the reverse direction being lossless.
    """
    failures = [
        (index, sentence)
        for index, sentence in enumerate(sentences)
        if not roundtrip_ok(sentence, "devanagari")
    ]
    with_latin = [item for item in failures if _LATIN.search(item[1])]
    no_latin = [item for item in failures if not _LATIN.search(item[1])]
    no_ascii_alnum = [item for item in no_latin if not _ASCII_ALNUM.search(item[1])]

    def examples(items: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
        return [{"index": index, "sentence": sentence} for index, sentence in items[:max_examples]]

    return {
        "n": len(sentences),
        "failures": len(failures),
        "failures_with_latin": len(with_latin),
        "failures_pure_devanagari": len(no_latin),
        "failures_no_ascii_alnum": len(no_ascii_alnum),
        "examples_pure_devanagari": examples(no_latin),
        "examples_no_ascii_alnum": examples(no_ascii_alnum),
    }


def script_variants(sentences: Sequence[str], language: str) -> dict[str, list[str]]:
    """`{"original": ...}` for every language, plus `{"slp1": ...}` for Devanagari ones.

    Both are kept because the two answer different questions: the original script is what
    a deployed tokenizer actually sees, and SLP1 is this project's internal encoding
    (CLAUDE.md §2.3), so the pair separates "Sanskrit is expensive" from "Devanagari bytes
    are expensive".
    """
    variants = {ORIGINAL: list(sentences)}
    if language.endswith("_Deva"):
        variants[SLP1] = [to_slp1(sentence, "devanagari") for sentence in sentences]
    return variants


# ------------------------------------------------------------------------- aggregation


def summarise_metric(result: DetailedMetricResult | Mapping[str, Any]) -> dict[str, Any]:
    """A metric result with its per-item distribution replaced by that list's mean and std.

    Returns `value`, `n` and `unit` unchanged (the CLAUDE.md §7 contract), plus
    `distribution` naming the key that was summarised (or `None` when the metric attached
    none), `mean` and `std`. `std` is the population standard deviation, matching
    `numpy.std`'s default; both are `0.0` for an empty or absent distribution.

    `mean` is not redundant with `value`: the metrics pool their numerator and denominator
    over the whole corpus, so for compression and parity the mean of the per-item ratios
    is a different — and, for a per-sentence sense of spread, more useful — number.
    """
    summary: dict[str, Any] = {
        "value": float(result["value"]),
        "n": int(result["n"]),
        "unit": str(result["unit"]),
        "distribution": None,
        "mean": 0.0,
        "std": 0.0,
    }
    for key in DISTRIBUTION_KEYS:
        raw: Any = result.get(key)
        if raw is None:
            continue
        values: list[float] = [float(item) for item in raw]
        summary["distribution"] = key
        if values:
            summary["mean"] = statistics.fmean(values)
            summary["std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        break
    return summary


def compute_metrics(
    tokenizers: Sequence[Tokenizer],
    variants: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Fertility and compression for every tokenizer × language × script variant.

    Nested `tokenizer -> language -> variant -> metric`, which is the shape
    `results.json` carries and the figure reads.
    """
    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for tokenizer in tokenizers:
        per_language: dict[str, dict[str, dict[str, Any]]] = {}
        for language, language_variants in variants.items():
            per_variant: dict[str, dict[str, Any]] = {}
            for variant, texts in language_variants.items():
                per_variant[variant] = {
                    "fertility": summarise_metric(fertility(tokenizer, texts)),
                    "compression": summarise_metric(compression(tokenizer, texts)),
                }
                logger.info(
                    "%s / %s / %s: fertility %.3f, compression %.3f bytes/token",
                    tokenizer.name,
                    language,
                    variant,
                    per_variant[variant]["fertility"]["value"],
                    per_variant[variant]["compression"]["value"],
                )
            per_language[language] = per_variant
        metrics[tokenizer.name] = per_language
    return metrics


def _parity_row(
    tokenizer: Tokenizer,
    variants: Mapping[str, Mapping[str, Sequence[str]]],
    source_language: str,
    source_variant: str,
    pivot: str,
) -> dict[str, Any]:
    """One parity row: the summarised metric plus what was compared against what.

    The pivot is always scored in its original script — `eng_Latn` has no SLP1 form — so
    only the source side varies. Naming the three operands in the row itself means the
    dict key never has to be parsed to interpret the number.
    """
    if pivot not in variants:
        raise KeyError(f"no sentences for the parity pivot {pivot!r}")
    row: dict[str, Any] = {
        "pivot": pivot,
        "source_language": source_language,
        "source_variant": source_variant,
        **summarise_metric(
            parity(
                tokenizer,
                variants[source_language][source_variant],
                variants[pivot][ORIGINAL],
            )
        ),
    }
    label = source_language if source_variant == ORIGINAL else f"{source_language}(SLP1)"
    logger.info("%s: parity %s/%s = %.3f", tokenizer.name, label, pivot, row["value"])
    return row


def compute_parity(
    tokenizers: Sequence[Tokenizer],
    variants: Mapping[str, Mapping[str, Sequence[str]]],
    pivots: Sequence[str],
    source_language: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Parity of `source_language` against each pivot, on identical FLORES content.

    One row per pivot in the original script, plus a `"<default pivot>__slp1"` row that
    scores the SLP1 form of the source against the same (Latin) pivot. The default pivot
    is `pivots[0]`, `eng_Latn` per CLAUDE.md §7; the SLP1 row is skipped when the source
    language has no SLP1 variant.

    Each row carries `pivot` and `source_variant` alongside the summarised metric, so the
    key never has to be parsed to know what was compared.
    """
    if source_language not in variants:
        raise KeyError(f"no sentences for the parity source language {source_language!r}")
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for tokenizer in tokenizers:
        rows: dict[str, dict[str, Any]] = {
            pivot: _parity_row(tokenizer, variants, source_language, ORIGINAL, pivot)
            for pivot in pivots
        }
        if pivots and SLP1 in variants[source_language]:
            default_pivot = pivots[0]
            rows[f"{default_pivot}__{SLP1}"] = _parity_row(
                tokenizer, variants, source_language, SLP1, default_pivot
            )
        results[tokenizer.name] = rows
    return results


# ------------------------------------------------------------------------------ figure


def make_figure(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Grouped bars of original-script fertility: one group per language, one bar per arm.

    Written as both `.pdf` and `.png` (CLAUDE.md §8) into `out_dir`, which is created if
    needed; returns the two paths in that order. The legend title says these arms are
    existing practice with unmatched vocabularies, so the chart cannot be misread as a
    controlled comparison, and neither the figure title nor the axis label calls fertility
    the headline number (CLAUDE.md §2.1).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on CI and over ssh
    import matplotlib.pyplot as plt

    metrics = results["metrics"]
    tokenizer_names = list(metrics)
    if not tokenizer_names:
        raise ValueError("no tokenizers in results['metrics']; nothing to plot")
    languages = list(metrics[tokenizer_names[0]])

    out_dir.mkdir(parents=True, exist_ok=True)
    width = 0.8 / len(tokenizer_names)
    positions = range(len(languages))

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for index, name in enumerate(tokenizer_names):
        offset = (index - (len(tokenizer_names) - 1) / 2) * width
        values = [
            metrics[name][language][ORIGINAL]["fertility"]["value"] for language in languages
        ]
        bars = axes.bar([position + offset for position in positions], values, width, label=name)
        axes.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    axes.set_xticks(list(positions))
    axes.set_xticklabels(languages)
    axes.set_xlabel("FLORES-200 devtest language (original script)")
    axes.set_ylabel(FIGURE_YLABEL)
    axes.set_title(FIGURE_TITLE, fontsize=10)
    axes.legend(title="Off-the-shelf arm (existing practice; vocabularies differ)", fontsize=8)
    axes.margins(y=0.18)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

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

    # Nothing here is stochastic, but the seed is set and recorded anyway so that adding
    # any sampling later cannot quietly make a run irreproducible (CLAUDE.md §8). `torch`
    # is not a dependency until Experiment 05, so it is not seeded here.
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    languages: list[str] = list(config["languages"])
    if not languages:
        raise ValueError(f"{args.config}: 'languages' is empty; nothing to measure")
    pivots: list[str] = list(config["pivots"])
    split = str(config["split"])
    parity_source = str(config.get("parity_source", PARITY_SOURCE))
    jsonl_path = resolve_path(str(config["flores_jsonl"]), root)
    out_dir = resolve_path(str(config["output_dir"]), root)

    corpus = get_corpus(languages, split, jsonl_path)
    sentences = select_languages(corpus, languages)
    total = len(next(iter(sentences.values())))

    roundtrip = roundtrip_report(sentences[ROUNDTRIP_LANGUAGE])
    if roundtrip["failures"]:
        logger.warning(
            "SLP1 roundtrip: %d/%d %s sentences did not roundtrip "
            "(%d contain Latin letters and cannot, by construction; %d have no Latin "
            "letter; of those, %d have no ASCII letter or digit at all)",
            roundtrip["failures"],
            roundtrip["n"],
            ROUNDTRIP_LANGUAGE,
            roundtrip["failures_with_latin"],
            roundtrip["failures_pure_devanagari"],
            roundtrip["failures_no_ascii_alnum"],
        )
        for example in roundtrip["examples_no_ascii_alnum"]:
            logger.warning(
                "  roundtrip failure with no ASCII letter or digit, at index %d: %r",
                example["index"],
                example["sentence"],
            )

    indices = select_aligned_indices(sentences)
    if len(indices) < total:
        logger.warning(
            "dropping %d of %d indices that are blank in at least one language",
            total - len(indices),
            total,
        )
    sentences = take_indices(sentences, indices)
    logger.info("using %d aligned sentences across %s", len(indices), ", ".join(languages))

    variants = {
        language: script_variants(texts, language) for language, texts in sentences.items()
    }

    tokenizers = [load_tokenizer(name) for name in config["tokenizers"]]
    metrics = compute_metrics(tokenizers, variants)
    parity_rows = compute_parity(tokenizers, variants, pivots, parity_source)

    results: dict[str, Any] = {
        "experiment": str(config.get("experiment", out_dir.name)),
        "git_commit": git_commit(root),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "corpus": {
            "name": corpus.name,
            "split": corpus.split,
            # Repo-relative when it is inside the repo, so `results.json` stays portable;
            # absolute otherwise, since a path outside the repo has no relative form.
            "path": str(
                jsonl_path.relative_to(root) if jsonl_path.is_relative_to(root) else jsonl_path
            ),
        },
        "seed": seed,
        "n_sentences_total": total,
        "n_sentences_used": len(indices),
        # Vocabulary sizes differ across these arms, so this table is existing practice,
        # not a controlled comparison (CLAUDE.md §2.5).
        "tokenizer_sources": {
            tokenizer.name: {
                "source_id": tokenizer.source_id,
                "vocab_size": tokenizer.vocab_size,
            }
            for tokenizer in tokenizers
        },
        "roundtrip": roundtrip,
        "roundtrip_failures": roundtrip["failures"],
        "metrics": metrics,
        "parity": parity_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    logger.info("wrote %s", results_path)

    shutil.copyfile(args.config, out_dir / "config.yaml")
    logger.info("copied %s to %s", args.config, out_dir / "config.yaml")

    make_figure(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
