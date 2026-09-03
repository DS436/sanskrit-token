"""Train the provisional T1/T2 tokenizers on SLP1 (exp02 Task 4).

Builds one SLP1 training corpus from the Sanskrit side of the Sāmayik and Itihāsa
*training* splits (prose before verse, CLAUDE.md §7), asserted against
`data/exclusion_hashes.txt` (CLAUDE.md §2.4: no evaluation leakage), and trains the four
`T1_bpe_raw_{32k,64k}` / `T2_unigram_raw_{32k,64k}` arms (CLAUDE.md §6) from it at matched
vocabulary sizes (CLAUDE.md §2.5).

These arms are **provisional**: the monolingual corpus this project will eventually train
on (DCS, GRETIL, Wikipedia; milestone M1) is not assembled yet, so training text is the
Sanskrit sides of two parallel corpora instead — see `docs/decisions.md`, "Provisional
T1/T2 tokenizers trained on the Sanskrit sides of Sāmayik and Itihāsa training splits".
Every table these arms appear in must say so.

Each trained tokenizer is written exactly where `sanskrit_tok.tokenizers.registry
.trained_tokenizer_path` expects it, so `load_tokenizer("T1_bpe_raw_32k")` (etc.) works
immediately afterwards with no further wiring. A `results.json` and a copy of the config
are written alongside it, in the same directory (CLAUDE.md §2.9).

Run it with `uv run python experiments/02_tpp_parallel/train_tokenizers.py`. Relative
paths in the config are resolved against the repository root, so the working directory
does not matter.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer as RawTokenizer

from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash
from sanskrit_tok.data.itihasa import load_itihasa
from sanskrit_tok.data.samayik import load_samayik
from sanskrit_tok.provenance import git_commit, git_dirty
from sanskrit_tok.tokenizers.corpus import build_training_corpus
from sanskrit_tok.tokenizers.registry import trained_tokenizer_path
from sanskrit_tok.tokenizers.train_bpe import train_bpe
from sanskrit_tok.tokenizers.train_unigram import train_unigram

logger = logging.getLogger("train_tokenizers")

#: `tokenizers.yaml`'s `sources` entries -> a loader for that split's Sanskrit
#: (Devanagari) sentences. Prose before verse (CLAUDE.md §7): Sāmayik listed first.
SOURCE_LOADERS: dict[str, Callable[[], list[str]]] = {
    "samayik_train": lambda: load_samayik("train").sentences["san_Deva"],
    "itihasa_train": lambda: load_itihasa("train").sentences["san_Deva"],
}

#: `tokenizers.yaml`'s `arms[].algo` -> the trainer function for that algorithm.
ALGO_TRAINERS: dict[str, Callable[..., Path]] = {
    "bpe": train_bpe,
    "unigram": train_unigram,
}


# ------------------------------------------------------------------- paths and config


def repo_root() -> Path:
    """The repository root, i.e. the parent of `experiments/`."""
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    """Resolve a config path: absolute ones as given, relative ones against `root`."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict[str, Any]:
    """Read the training config (`yaml.safe_load` only)."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(config).__name__}")
    return config


# --------------------------------------------------------------------------- corpus


def collect_sources(names: Sequence[str]) -> dict[str, list[str]]:
    """Load the Devanagari Sanskrit sentences for each named source, in config order."""
    missing = [name for name in names if name not in SOURCE_LOADERS]
    if missing:
        raise KeyError(f"unknown corpus source(s) {missing}; known: {list(SOURCE_LOADERS)}")
    return {name: SOURCE_LOADERS[name]() for name in names}


def filter_leaked_sentences(
    sources: Mapping[str, Sequence[str]], exclusion: frozenset[str]
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Drop sentences that collide with the evaluation exclusion list before training.

    Real corpora are not perfectly split: a handful of Sāmayik train sentences are
    formulaic course-material lines ("पाठान्ता: प्रश्ना:", "end-of-lesson questions")
    that recur verbatim in its own dev/test/test_ood splits, and a handful of Itihāsa
    train sentences are famous, oft-quoted ślokas that recur verbatim in its dev/test
    splits — not a split-alignment bug, just how these corpora were assembled upstream
    (verified by inspection: 2026-09-03 decision log entry "T1/T2 training run: filtered
    218 leaked sentences before the corpus build; no vocab shortfall").

    `build_training_corpus`'s own `assert_not_excluded` call (CLAUDE.md §2.4) is a hard
    stop for exactly this condition — a safety net, not the removal mechanism — so this
    runs first and drops the colliding sentences, logging a WARNING with the count per
    source. After this filter, that assertion is expected to pass trivially.

    Returns `(filtered_sources, dropped_counts)`: `dropped_counts` names *every* source
    passed in, including the ones with nothing dropped (value `0`), so a caller building a
    manifest never has to special-case a source that had no leakage.
    """
    filtered: dict[str, list[str]] = {}
    dropped_counts: dict[str, int] = {}
    for name, texts in sources.items():
        kept = [text for text in texts if sentence_hash(text) not in exclusion]
        dropped = len(texts) - len(kept)
        dropped_counts[name] = dropped
        if dropped:
            logger.warning(
                "%s: dropped %d/%d sentence(s) that collide with the evaluation "
                "exclusion list (formulaic/repeated text recurring across splits, not a "
                "split-alignment bug; see docs/decisions.md)",
                name,
                dropped,
                len(texts),
            )
        filtered[name] = kept
    return filtered, dropped_counts


def ensure_training_corpus(
    sources: Mapping[str, Sequence[str]],
    corpus_path: Path,
    exclusion: frozenset[str],
) -> dict[str, object]:
    """Build the SLP1 training corpus, or reuse it if it is already current.

    "Current" means `corpus_path` exists, a `manifest.json` sits beside it, and the
    manifest's recorded `sha256` matches the file's actual bytes right now — i.e. nothing
    has touched the corpus file since that manifest was written. Any mismatch (missing
    manifest, hand-edited corpus, stale sha256) triggers a full rebuild rather than
    trusting a corpus that might not be what the manifest describes.

    On a rebuild, `sources` (the *raw*, unfiltered sentences from `collect_sources`) is
    first run through `filter_leaked_sentences`, and the manifest `build_training_corpus`
    returns is extended with two keys before being written and returned: `n_in_raw` (the
    per-source count of `sources` as given, before any filtering) and `n_leaked_dropped`
    (the per-source count `filter_leaked_sentences` dropped). `n_in` — already in the
    manifest `build_training_corpus` returns — stays the *post*-filter count, i.e. what was
    actually fed to `build_training_corpus`.
    """
    manifest_path = corpus_path.parent / "manifest.json"
    if corpus_path.exists() and manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        if recorded.get("sha256") == actual_sha256:
            logger.info(
                "reusing existing training corpus %s (sha256 matches %s)",
                corpus_path,
                manifest_path,
            )
            return dict(recorded)
        logger.info("%s exists but does not match %s; rebuilding", corpus_path, manifest_path)

    n_in_raw = {name: len(texts) for name, texts in sources.items()}
    filtered, dropped_counts = filter_leaked_sentences(sources, exclusion)

    manifest = build_training_corpus(filtered, corpus_path, exclusion)
    manifest["n_in_raw"] = n_in_raw
    manifest["n_leaked_dropped"] = dropped_counts

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", manifest_path)
    return manifest


# ---------------------------------------------------------------------------- training


def train_arm(arm: Mapping[str, Any], corpus_path: Path, seed: int) -> dict[str, Any]:
    """Train one arm from `tokenizers.yaml`'s `arms` list and write its `tokenizer.json`.

    Writes to `trained_tokenizer_path(name).parent` — the exact location
    `sanskrit_tok.tokenizers.registry.load_tokenizer` reads from — so the file this run
    produces and the file the registry loads afterwards can never drift apart.

    Returns the per-arm portion of `results.json`: the run-wide keys (`git_commit`,
    `git_dirty`, `timestamp`, `config`, `manifest`) are filled in by the caller, since
    they are identical across every arm in one run.
    """
    name = str(arm["name"])
    algo = str(arm["algo"])
    vocab_size = int(arm["vocab_size"])
    if algo not in ALGO_TRAINERS:
        raise ValueError(f"{name}: unknown algo {algo!r}; expected one of {list(ALGO_TRAINERS)}")

    out_dir = trained_tokenizer_path(name).parent
    trainer = ALGO_TRAINERS[algo]

    start = time.monotonic()
    tokenizer_path = trainer(corpus_path, vocab_size, out_dir, seed=seed)
    train_seconds = time.monotonic() - start

    actual_vocab_size = int(RawTokenizer.from_file(str(tokenizer_path)).get_vocab_size())
    if actual_vocab_size != vocab_size:
        # Expected for Unigram on a small corpus (CLAUDE.md §11: record the deviation).
        logger.warning(
            "%s: requested vocab_size=%d, trainer settled on %d",
            name,
            vocab_size,
            actual_vocab_size,
        )

    logger.info(
        "%s: trained in %.1fs, vocab_size=%d, wrote %s",
        name,
        train_seconds,
        actual_vocab_size,
        tokenizer_path,
    )
    return {
        "arm": name,
        "algo": algo,
        "requested_vocab_size": vocab_size,
        "vocab_size": actual_vocab_size,
        "train_seconds": train_seconds,
        "seed": seed,
        "corpus_path": str(corpus_path),
        "tokenizer_path": str(tokenizer_path),
    }


def write_arm_outputs(name: str, results: Mapping[str, Any], config_path: Path) -> Path:
    """Write `results.json` and a copy of the config beside the trained tokenizer.

    Both land in `trained_tokenizer_path(name).parent` (CLAUDE.md §2.9: every experiment
    writes a `results.json` and a `config.yaml` to its output dir). Returns the
    `results.json` path.
    """
    out_dir = trained_tokenizer_path(name).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    logger.info("wrote %s", results_path)

    shutil.copyfile(config_path, out_dir / "config.yaml")
    logger.info("copied %s to %s", config_path, out_dir / "config.yaml")
    return results_path


# -------------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("tokenizers.yaml"),
        help="tokenizer training config YAML (default: tokenizers.yaml beside this script)",
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
    corpus_path = resolve_path(str(config["corpus_path"]), root)
    exclusion_path = resolve_path(str(config["exclusion_path"]), root)
    source_names: list[str] = list(config["sources"])
    arms: list[dict[str, Any]] = list(config["arms"])
    if not arms:
        raise ValueError(f"{args.config}: 'arms' is empty; nothing to train")

    exclusion = load_exclusion_hashes(exclusion_path)
    logger.info("loaded %d exclusion hashes from %s", len(exclusion), exclusion_path)

    sources = collect_sources(source_names)
    for source_name, texts in sources.items():
        logger.info("source %s: %d sentence(s) loaded", source_name, len(texts))

    # `ensure_training_corpus` owns the leakage filter (`filter_leaked_sentences`) itself
    # on a rebuild, and records both the raw and post-filter per-source counts in the
    # manifest it returns/writes — see that function's docstring.
    manifest = ensure_training_corpus(sources, corpus_path, exclusion)
    logger.info(
        "training corpus: n_in_raw=%s n_leaked_dropped=%s n_in=%s n_out=%d n_dedup_removed=%d",
        manifest.get("n_in_raw"),
        manifest.get("n_leaked_dropped"),
        manifest["n_in"],
        manifest["n_out"],
        manifest["n_dedup_removed"],
    )

    commit = git_commit(root)
    dirty = git_dirty(root)
    if dirty:
        logger.warning(
            "the working tree has uncommitted changes; git_commit names the parent "
            "commit, not the code that ran (recording git_dirty=true)"
        )
    for arm in arms:
        name = str(arm["name"])
        logger.info("training %s ...", name)
        payload = train_arm(arm, corpus_path, seed)
        payload["git_commit"] = commit
        payload["git_dirty"] = dirty
        payload["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        payload["config"] = config
        payload["manifest"] = manifest
        write_arm_outputs(name, payload, args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
