"""Helpers every experiment script shares: config paths, provenance, results writing.

`experiments/` holds scripts, not a package — the directory names start with digits, so
they cannot be imported — and until Experiment 03 the three scripts under it each carried
their own copy of the same six helpers. The copies had already begun to drift (exp01
wrote `results.json` with `allow_nan` left at its permissive default; exp02 added a
sanitiser and a strict dump), which is the failure mode this module exists to end: one
implementation, one set of tests, imported by every script (docs/decisions.md, "Add torch
as a dependency; shared experiment helpers move to `sanskrit_tok.experiment`").

Nothing here is experiment-specific. The rule for what belongs is CLAUDE.md §8's split
between pure functions and I/O: these are the *plumbing* every experiment repeats —
resolving a config path against the repository root so the working directory cannot
change a number, recording which commit produced a file, turning a metric tree into
strict JSON. Metrics live in `sanskrit_tok.metrics`; corpus loading lives in
`sanskrit_tok.data`; the experiment's own aggregation stays in its `run.py`.

`git_commit`/`git_dirty` stay in `sanskrit_tok.provenance`, which owns the subprocess
handling and its own tests; `provenance()` here is the thin composition of the two with a
timestamp, in the key order `results.json` stores them in.

`load_corpus_entry` and `SOURCE_LOADERS` are here for the same reason as the rest: three
scripts (exp02's `run.py` and `train_tokenizers.py`, exp03's `split_corpora.py`) name the
same corpora in the same YAML shape, and a fourth copy of the dispatch is a fourth chance
for two of them to end up measuring and training on different text.
"""

import json
import logging
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sanskrit_tok.metrics.summary import summarise_metric
from sanskrit_tok.provenance import git_commit, git_dirty

if TYPE_CHECKING:
    from sanskrit_tok.data.parallel import ParallelCorpus

__all__ = [
    "ENGLISH_LANGUAGE",
    "ENGLISH_SOURCE_LOADERS",
    "HINDI_LANGUAGE",
    "SANSKRIT_LANGUAGE",
    "SANSKRIT_SOURCE_LOADERS",
    "SOURCE_LOADERS",
    "TPP_EXTRA_KEYS",
    "collect_sources",
    "load_config",
    "load_corpus_entry",
    "provenance",
    "repo_root",
    "resolve_path",
    "sanitize_json",
    "select_aligned_indices",
    "summarise_tpp",
    "take_indices",
    "write_results",
]

logger = logging.getLogger(__name__)

#: `DetailedMetricResult` keys `summarise_metric` drops (it keeps only value/n/unit/
#: distribution/mean/std) and that `summarise_tpp` copies back into the stored summary.
#: `ci` (the nominal confidence level, e.g. 0.95) is not itself a key `tpp()` returns —
#: only `ci_low`/`ci_high` are — so it is not in this tuple; `summarise_tpp` sets it
#: explicitly from the value the caller used.
TPP_EXTRA_KEYS: tuple[str, ...] = (
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
    """The repository root: the parent of `src/`, and so of `experiments/` and `data/`.

    Derived from this file's location rather than from the working directory, so
    `uv run python experiments/01_baseline_penalty/run.py` behaves identically from
    anywhere in the repo — and, more to the point, so that two runs from two different
    directories cannot resolve `data/raw/...` to two different corpora.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a config path: absolute ones as given, relative ones against `root`.

    `root` defaults to `repo_root()`, which is what every config in this repository is
    written against; it is a parameter chiefly so tests can resolve against `tmp_path`.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root() if root is None else root) / path


def load_config(path: Path) -> dict[str, Any]:
    """Read an experiment's YAML config (`yaml.safe_load` only).

    Raises `ValueError` when the document is not a mapping — an empty file parses to
    `None` and a bare list to `list`, both of which would otherwise fail much later with
    a `TypeError` deep inside the run.
    """
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(config).__name__}")
    return config


def provenance(root: Path | None = None) -> dict[str, object]:
    """`{"git_commit", "git_dirty", "timestamp"}` for a `results.json`, in that order.

    The key order matters only cosmetically — it is the order the three experiment
    scripts already stored them in, so `**provenance()` drops into their results dicts
    without reordering the file — but the WARNING does not: a dirty tree means
    `git_commit` names the *parent* commit, not the code that ran, so a run whose numbers
    might not be reproducible from that commit says so once, here, rather than in each
    caller (`sanskrit_tok.provenance` explains the contract in full).

    `root` defaults to `repo_root()`; the timestamp is ISO-8601 UTC to the second.
    """
    root = repo_root() if root is None else root
    dirty = git_dirty(root)
    if dirty:
        logger.warning(
            "the working tree has uncommitted changes; git_commit names the parent "
            "commit, not the code that ran (recording git_dirty=true)"
        )
    return {
        "git_commit": git_commit(root),
        "git_dirty": dirty,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ------------------------------------------------------------------------ corpus loading

#: FLORES-200 language codes, the names every corpus in this project keys its sentences by.
SANSKRIT_LANGUAGE = "san_Deva"
ENGLISH_LANGUAGE = "eng_Latn"
HINDI_LANGUAGE = "hin_Deva"


def load_corpus_entry(entry: Mapping[str, Any], root: Path) -> "ParallelCorpus":
    """Dispatch one config `corpora`/`eval_corpora` entry to its loader, by `entry["loader"]`.

    The shape is the one `experiments/02_tpp_parallel/config.yaml` established:
    `{name, loader, split}` plus `jsonl` for FLORES, whose devtest file is downloaded and
    cached on first use and read from disk afterwards. Imported lazily so that loading this
    module does not pull in `datasets`.
    """
    from sanskrit_tok.data.flores import load_jsonl, save_jsonl
    from sanskrit_tok.data.itihasa import load_itihasa
    from sanskrit_tok.data.samayik import load_samayik

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


def _train_sentences(loader: str, language: str) -> list[str]:
    """One language of one corpus's `train` split, through the same dispatch as everything
    else — so "the Sāmayik training sentences" has exactly one definition in this repo."""
    corpus = load_corpus_entry(
        {"name": f"{loader}_train", "loader": loader, "split": "train"}, repo_root()
    )
    return list(corpus.sentences[language])


#: Training-corpus source name -> a loader for that split's Sanskrit sentences. Prose
#: before verse (CLAUDE.md §7): Sāmayik first, and the order is binding — it decides which
#: of two duplicate sentences is the one kept.
SANSKRIT_SOURCE_LOADERS: dict[str, Callable[[], list[str]]] = {
    "samayik_train": lambda: _train_sentences("samayik", SANSKRIT_LANGUAGE),
    "itihasa_train": lambda: _train_sentences("itihasa", SANSKRIT_LANGUAGE),
}

#: The English side of the very same `ParallelCorpus` objects, for the matched E1 control
#: arms: the two corpora are the two sides of exactly the same sentences.
ENGLISH_SOURCE_LOADERS: dict[str, Callable[[], list[str]]] = {
    "samayik_train_en": lambda: _train_sentences("samayik", ENGLISH_LANGUAGE),
    "itihasa_train_en": lambda: _train_sentences("itihasa", ENGLISH_LANGUAGE),
}


#: Both sides, which is what `train_tokenizers.py` resolves an arm's `sources` against.
SOURCE_LOADERS: dict[str, Callable[[], list[str]]] = {
    **SANSKRIT_SOURCE_LOADERS,
    **ENGLISH_SOURCE_LOADERS,
}


def collect_sources(
    names: Sequence[str], loaders: Mapping[str, Callable[[], list[str]]] | None = None
) -> dict[str, list[str]]:
    """Load the sentences for each named source, in config order.

    `loaders` defaults to `SOURCE_LOADERS`; `split_corpora.py` passes
    `SANSKRIT_SOURCE_LOADERS` so an English source name in its `train_sources` is an error
    rather than five hours of a sandhi splitter reading English.
    """
    table = SOURCE_LOADERS if loaders is None else loaders
    missing = [name for name in names if name not in table]
    if missing:
        raise KeyError(f"unknown corpus source(s) {missing}; known: {list(table)}")
    return {name: table[name]() for name in names}


# ---------------------------------------------------------------------- corpus wrangling


def select_aligned_indices(sentences: Mapping[str, Sequence[str]]) -> list[int]:
    """Indices whose sentence is non-blank in *every* language.

    Filtering the shared index rather than each language separately is what keeps the
    languages aligned; parity and TPP are meaningless the moment they are not. Raises
    `ValueError` if the languages differ in length, since nothing downstream can align
    corpora that were never the same length to begin with.

    Takes the languages as a mapping rather than a `ParallelCorpus` because both callers
    have already turned their corpus into one (exp01 after `select_languages`, exp02
    inside `prepare_corpus`), and because the mapping is what makes the helper testable
    with four hand-written strings.
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


# --------------------------------------------------------------------- results writing


def sanitize_json(value: Any) -> Any:
    """Make a results tree strict-JSON safe, without mutating it.

    `json.dump(..., allow_nan=False)` raises on a non-finite float rather than emitting
    the non-standard `NaN`/`Infinity` tokens JSON forbids; several metrics produce `nan`
    on purpose (an undefined per-pair ratio, a bootstrap CI over zero draws), so the tree
    is sanitised once, here, before it is ever written. `tuple` becomes `list` and `Path`
    becomes `str` for the same reason: both are things a results dict legitimately holds
    and neither is JSON.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


def write_results(
    results: Mapping[str, object],
    out_dir: Path,
    config_src: Path | None = None,
) -> Path:
    """Write `out_dir/results.json` (and a copy of the config), returning the results path.

    CLAUDE.md §2.9: every experiment writes a `results.json` and a `config.yaml` to its
    output dir. The JSON is sanitised (`sanitize_json`) and then dumped strictly
    (`allow_nan=False`), so a non-finite float that slipped past the sanitiser is a loud
    failure rather than a file no strict JSON parser will read; `ensure_ascii=False` keeps
    Devanagari readable and `indent=2` keeps the diff between two runs legible.

    `config_src` is copied byte-for-byte to `out_dir/config.yaml` — comments included —
    so the file beside a number is the file that produced it; pass `None` for an output
    directory that has no config of its own. `out_dir` is created if it does not exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(results), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    logger.info("wrote %s", results_path)

    if config_src is not None:
        config_dest = out_dir / "config.yaml"
        shutil.copyfile(config_src, config_dest)
        logger.info("copied %s to %s", config_src, config_dest)
    return results_path


# ------------------------------------------------------------------------ TPP summaries


def summarise_tpp(raw: Mapping[str, Any], *, ci: float) -> dict[str, Any]:
    """`summarise_metric(raw)` plus the bootstrap/undefined-count keys it drops.

    `summarise_metric` (CLAUDE.md §7 contract) keeps only `value`/`n`/`unit`/
    `distribution`/`mean`/`std`; `tpp`'s `ci_low`, `ci_high`, `n_undefined`,
    `n_bootstrap`, `seed`, `source_tokens` and `pivot_tokens` are the caller's to record
    alongside it (`summary.py`'s own docstring says as much), which is what this does.
    `ci` (the nominal confidence level the caller passed to `tpp()`, e.g. `0.95`) is not
    part of `tpp()`'s own return value — only the resulting `ci_low`/`ci_high` bounds are
    — so it is set here from the argument, next to the bounds it produced, rather than
    copied from `raw`.
    """
    summary: dict[str, Any] = dict(summarise_metric(raw))
    for key in TPP_EXTRA_KEYS:
        summary[key] = raw[key]
    summary["ci"] = ci
    return summary
