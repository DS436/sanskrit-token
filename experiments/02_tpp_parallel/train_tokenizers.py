"""Train the provisional T1/T2 tokenizers and the matched English control E1 (exp02 Tasks 4, 6).

Builds one SLP1 training corpus from the Sanskrit side of the Sāmayik and Itihāsa
*training* splits (prose before verse, CLAUDE.md §7), asserted against
`data/exclusion_hashes.txt` (CLAUDE.md §2.4: no evaluation leakage), and trains the four
`T1_bpe_raw_{32k,64k}` / `T2_unigram_raw_{32k,64k}` arms (CLAUDE.md §6) from it at matched
vocabulary sizes (CLAUDE.md §2.5).

It then does the same thing one language over. The `E1_*` arms (`side: en` in
`tokenizers.yaml`) are trained on the **English side of the same training splits**, with
the same trainers and the same two vocabulary sizes, so that TPP measured against an E1
arm holds algorithm, vocabulary size and training domain constant on both sides of the
ratio, and whatever gap remains is about the languages (docs/decisions.md, "Add a matched
English control family E1 for TPP"). The English corpus is *not* transliterated
(`identity_transform`) and is asserted against its own list,
`data/exclusion_hashes_en.txt`, hashed with `sentence_hash_en`.

The T1/T2 arms are **provisional**: the monolingual corpus this project will eventually
train on (DCS, GRETIL, Wikipedia; milestone M1) is not assembled yet, so training text is
the Sanskrit sides of two parallel corpora instead — see `docs/decisions.md`, "Provisional
T1/T2 tokenizers trained on the Sanskrit sides of Sāmayik and Itihāsa training splits".
Every table these arms appear in must say so.

Each trained tokenizer is written exactly where `sanskrit_tok.tokenizers.registry
.trained_tokenizer_path` expects it, so `load_tokenizer("T1_bpe_raw_32k")` (etc.) works
immediately afterwards with no further wiring. A `results.json` and a copy of the config
are written alongside it, in the same directory (CLAUDE.md §2.9).

An arm whose `tokenizer.json` already exists is **skipped**, so adding an arm to
`tokenizers.yaml` does not retrain (and, for Unigram, silently replace — see
`docs/decisions.md`, "`UnigramTrainer` is not bit-for-bit deterministic") the arms whose
numbers are already in a `results.json`. Pass `--retrain` to train every configured arm
regardless.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer as RawTokenizer

from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash, sentence_hash_en
from sanskrit_tok.data.itihasa import load_itihasa
from sanskrit_tok.data.samayik import load_samayik
from sanskrit_tok.provenance import git_commit, git_dirty
from sanskrit_tok.tokenizers.corpus import build_training_corpus, identity_transform, slp1_transform
from sanskrit_tok.tokenizers.registry import trained_tokenizer_path
from sanskrit_tok.tokenizers.train_bpe import train_bpe
from sanskrit_tok.tokenizers.train_unigram import train_unigram

logger = logging.getLogger("train_tokenizers")

#: The two sides of the parallel corpora an arm can be trained on: `sa` (Sanskrit, the
#: T1/T2 arms) and `en` (English, the E1 control arms). `tokenizers.yaml` names one per
#: arm under `side`; an arm without one is Sanskrit, which is what every arm was before
#: the E1 family existed.
SANSKRIT_SIDE = "sa"
ENGLISH_SIDE = "en"
SIDES = (SANSKRIT_SIDE, ENGLISH_SIDE)

#: `tokenizers.yaml`'s `sources`/`english_sources` entries -> a loader for that split's
#: sentences on that side. Prose before verse (CLAUDE.md §7): Sāmayik listed first. The
#: `_en` loaders read `eng_Latn` off the very same `ParallelCorpus` the `sa` ones read
#: `san_Deva` from, so the two corpora are the two sides of exactly the same sentences.
SOURCE_LOADERS: dict[str, Callable[[], list[str]]] = {
    "samayik_train": lambda: load_samayik("train").sentences["san_Deva"],
    "itihasa_train": lambda: load_itihasa("train").sentences["san_Deva"],
    "samayik_train_en": lambda: load_samayik("train").sentences["eng_Latn"],
    "itihasa_train_en": lambda: load_itihasa("train").sentences["eng_Latn"],
}

#: `tokenizers.yaml`'s `arms[].algo` -> the trainer function for that algorithm. Both
#: sides use the same trainers with the same settings; only the corpus differs.
ALGO_TRAINERS: dict[str, Callable[..., Path]] = {
    "bpe": train_bpe,
    "unigram": train_unigram,
}


# ------------------------------------------------------------------- paths and config


@dataclass(frozen=True)
class SideSpec:
    """The corpus recipe for one side of the parallel corpora (`side_spec`).

    Everything that differs between training a Sanskrit arm and an English control arm:
    where the corpus and its manifest go, which exclusion list guards it, which sources
    feed it, and the transform/hash pair the text is written and checked with.
    """

    side: str
    corpus_path: Path
    manifest_path: Path
    exclusion_path: Path
    source_names: list[str]
    transform: Callable[[str], str]
    hash_fn: Callable[[str], str]


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
    """Load the sentences for each named source, in config order.

    Sanskrit sources (`samayik_train`, `itihasa_train`) yield Devanagari; the `_en`
    sources yield the English side of the same splits, as written.
    """
    missing = [name for name in names if name not in SOURCE_LOADERS]
    if missing:
        raise KeyError(f"unknown corpus source(s) {missing}; known: {list(SOURCE_LOADERS)}")
    return {name: SOURCE_LOADERS[name]() for name in names}


def arm_side(arm: Mapping[str, Any]) -> str:
    """Which side of the parallel corpora an arm trains on: `"sa"` (default) or `"en"`.

    Defaults to Sanskrit for an arm with no `side` key — every arm predating the E1
    family is Sanskrit, and defaulting keeps an older `tokenizers.yaml` working. An
    unrecognised value raises rather than silently training an arm on the wrong corpus,
    which would be invisible in the results.
    """
    side = str(arm.get("side", SANSKRIT_SIDE))
    if side not in SIDES:
        raise ValueError(f"{arm.get('name')}: unknown side {side!r}; expected one of {list(SIDES)}")
    return side


def select_arms_to_train(arms: Sequence[Mapping[str, Any]], retrain: bool) -> list[dict[str, Any]]:
    """The configured arms that still need training: those with no `tokenizer.json` yet.

    Adding an arm to `tokenizers.yaml` (the four `E1_*` arms, say) must not retrain the
    arms whose numbers a `results.json` already reports: a Unigram retrain is not
    bit-for-bit reproducible (docs/decisions.md, "`UnigramTrainer` is not bit-for-bit
    deterministic"), so it would silently replace the artifact behind a published number.
    `retrain=True` (the `--retrain` flag) overrides and returns every arm.
    """
    selected: list[dict[str, Any]] = []
    for arm in arms:
        name = str(arm["name"])
        path = trained_tokenizer_path(name)
        if not retrain and path.exists():
            logger.info("%s: already trained at %s; skipping (use --retrain to force)", name, path)
            continue
        selected.append(dict(arm))
    return selected


def filter_leaked_sentences(
    sources: Mapping[str, Sequence[str]],
    exclusion: frozenset[str],
    hash_fn: Callable[[str], str] = sentence_hash,
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

    `hash_fn` must be the function `exclusion` was built with — `sentence_hash_en` for the
    English (E1) side, whose list is `data/exclusion_hashes_en.txt`.
    """
    filtered: dict[str, list[str]] = {}
    dropped_counts: dict[str, int] = {}
    for name, texts in sources.items():
        kept = [text for text in texts if hash_fn(text) not in exclusion]
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
    *,
    manifest_path: Path | None = None,
    transform: Callable[[str], str] | None = None,
    hash_fn: Callable[[str], str] = sentence_hash,
) -> dict[str, object]:
    """Build the training corpus, or reuse it if it is already current.

    "Current" means `corpus_path` exists, its manifest sits beside it, and the manifest's
    recorded `sha256` matches the file's actual bytes right now — i.e. nothing has touched
    the corpus file since that manifest was written. Any mismatch (missing manifest,
    hand-edited corpus, stale sha256) triggers a full rebuild rather than trusting a
    corpus that might not be what the manifest describes.

    `manifest_path` defaults to `manifest.json` beside `corpus_path`; the English (E1)
    corpus passes `manifest_en.json` explicitly, since both corpora live in
    `data/processed/` and would otherwise overwrite each other's manifest.
    `transform`/`hash_fn` are passed straight through to `build_training_corpus`
    (`identity_transform` + `sentence_hash_en` for English) and `hash_fn` is also used for
    the leakage filter below, so both must match the list `exclusion` came from.

    On a rebuild, `sources` (the *raw*, unfiltered sentences from `collect_sources`) is
    first run through `filter_leaked_sentences`, and the manifest `build_training_corpus`
    returns is extended with two keys before being written and returned: `n_in_raw` (the
    per-source count of `sources` as given, before any filtering) and `n_leaked_dropped`
    (the per-source count `filter_leaked_sentences` dropped). `n_in` — already in the
    manifest `build_training_corpus` returns — stays the *post*-filter count, i.e. what was
    actually fed to `build_training_corpus`.
    """
    manifest_path = corpus_path.parent / "manifest.json" if manifest_path is None else manifest_path
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
    filtered, dropped_counts = filter_leaked_sentences(sources, exclusion, hash_fn)

    manifest = build_training_corpus(
        filtered, corpus_path, exclusion, transform=transform, hash_fn=hash_fn
    )
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
        "side": arm_side(arm),
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


def side_spec(side: str, config: Mapping[str, Any], root: Path) -> SideSpec:
    """Everything that differs between the Sanskrit and English training corpora.

    Sanskrit reads `corpus_path`/`sources`/`exclusion_path` and writes `manifest.json`
    beside its corpus, transliterating to SLP1 and hashing with `sentence_hash`. English
    reads `english_corpus_path`/`english_sources`/`english_exclusion_path`, writes
    `manifest_en.json`, keeps its text as written (`identity_transform`) and hashes with
    `sentence_hash_en`.
    """
    if side == SANSKRIT_SIDE:
        corpus_path = resolve_path(str(config["corpus_path"]), root)
        return SideSpec(
            side=side,
            corpus_path=corpus_path,
            manifest_path=corpus_path.parent / "manifest.json",
            exclusion_path=resolve_path(str(config["exclusion_path"]), root),
            source_names=list(config["sources"]),
            transform=slp1_transform,
            hash_fn=sentence_hash,
        )
    corpus_path = resolve_path(str(config["english_corpus_path"]), root)
    return SideSpec(
        side=side,
        corpus_path=corpus_path,
        manifest_path=corpus_path.parent / "manifest_en.json",
        exclusion_path=resolve_path(str(config["english_exclusion_path"]), root),
        source_names=list(config["english_sources"]),
        transform=identity_transform,
        hash_fn=sentence_hash_en,
    )


def build_side_corpus(spec: SideSpec) -> dict[str, object]:
    """Load one side's training sources and build (or reuse) its training corpus."""
    exclusion = load_exclusion_hashes(spec.exclusion_path)
    logger.info(
        "%s: loaded %d exclusion hashes from %s", spec.side, len(exclusion), spec.exclusion_path
    )

    sources = collect_sources(spec.source_names)
    for source_name, texts in sources.items():
        logger.info("source %s: %d sentence(s) loaded", source_name, len(texts))

    # `ensure_training_corpus` owns the leakage filter (`filter_leaked_sentences`) itself
    # on a rebuild, and records both the raw and post-filter per-source counts in the
    # manifest it returns/writes — see that function's docstring.
    manifest = ensure_training_corpus(
        sources,
        spec.corpus_path,
        exclusion,
        manifest_path=spec.manifest_path,
        transform=spec.transform,
        hash_fn=spec.hash_fn,
    )
    logger.info(
        "%s training corpus: n_in_raw=%s n_leaked_dropped=%s n_in=%s n_out=%d n_dedup_removed=%d",
        spec.side,
        manifest.get("n_in_raw"),
        manifest.get("n_leaked_dropped"),
        manifest["n_in"],
        manifest["n_out"],
        manifest["n_dedup_removed"],
    )
    return manifest


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
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="train every configured arm, including ones whose tokenizer.json already exists",
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
    arms: list[dict[str, Any]] = list(config["arms"])
    if not arms:
        raise ValueError(f"{args.config}: 'arms' is empty; nothing to train")

    to_train = select_arms_to_train(arms, args.retrain)
    if not to_train:
        logger.info("every configured arm is already trained; nothing to do (--retrain forces)")
        return 0

    # Only build the corpus for a side that actually has an arm to train, so adding the
    # English arms does not re-read (or rebuild) the Sanskrit corpus, and vice versa.
    manifests: dict[str, dict[str, object]] = {}
    for side in SIDES:
        if any(arm_side(arm) == side for arm in to_train):
            manifests[side] = build_side_corpus(side_spec(side, config, root))

    commit = git_commit(root)
    dirty = git_dirty(root)
    if dirty:
        logger.warning(
            "the working tree has uncommitted changes; git_commit names the parent "
            "commit, not the code that ran (recording git_dirty=true)"
        )
    for arm in to_train:
        name = str(arm["name"])
        side = arm_side(arm)
        spec = side_spec(side, config, root)
        logger.info("training %s (side=%s) ...", name, side)
        payload = train_arm(arm, spec.corpus_path, seed)
        payload["git_commit"] = commit
        payload["git_dirty"] = dirty
        payload["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        payload["config"] = config
        payload["manifest"] = manifests[side]
        write_arm_outputs(name, payload, args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
