"""Shared machinery for training a family of tokenizer arms from a YAML config.

This is what `experiments/02_tpp_parallel/train_tokenizers.py` and
`experiments/04_morph_constrained/train_tokenizers.py` have in common: decide which
configured arms still need training, build (or reuse) the corpus each one trains on, run
the trainer, and write a `results.json` beside the `tokenizer.json` it produced
(CLAUDE.md §2.9). It lives in the package rather than in either script so the two
experiments cannot drift apart on any of it — an arm skipped in one and retrained in the
other, or two subtly different definitions of "the corpus is already current", would show
up only as numbers that do not reconcile.

What is *not* here is anything about a particular corpus: Experiment 02's sides (Sanskrit,
English, sandhi-split) and Experiment 03's split cache stay in the exp02 script, and
Experiment 04's DCS jsonl stays in the exp04 script. `SideSpec` is the seam — a plain
recipe the caller fills in — and `train_arm`'s `extra` is the other one, carrying whatever
per-experiment label (`side`, `corpus`) belongs in that experiment's `results.json`.

**Two corpus builders, for two shapes of source.** `ensure_training_corpus` takes the
sources in memory (`Mapping[str, Sequence[str]]`) and is what the parallel corpora need:
they are small, and the leakage filter, the two deduplications and the per-source manifest
counts all want the whole list. `build_streamed_corpus` takes an iterator of
`(check_text, write_text)` pairs and never holds the input: Experiment 04 builds five
corpora from a 685,805-line, 439 MB jsonl, and materialising that as a list of dicts is
several gigabytes for no benefit.
"""

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as RawTokenizer

from sanskrit_tok.data.exclusion import LeakageError, sentence_hash
from sanskrit_tok.experiment import write_results
from sanskrit_tok.tokenizers.corpus import (
    build_training_corpus,
    deduplicate_sources,
    filter_leaked_sentences,
)
from sanskrit_tok.tokenizers.morph_bpe import train_morph_bpe
from sanskrit_tok.tokenizers.registry import trained_tokenizer_path
from sanskrit_tok.tokenizers.train_bpe import train_bpe
from sanskrit_tok.tokenizers.train_unigram import train_unigram

__all__ = [
    "ALGO_TRAINERS",
    "SideSpec",
    "build_arg_parser",
    "build_streamed_corpus",
    "corpus_is_current",
    "ensure_training_corpus",
    "select_arms_to_train",
    "train_arm",
    "write_arm_results",
    "write_corpus_manifest",
]

logger = logging.getLogger(__name__)

#: A config's `arms[].algo` -> the trainer for that algorithm. Every trainer has the same
#: signature `(corpus_path, vocab_size, out_dir, *, seed) -> Path`, so `train_arm` does not
#: care which one it is calling. `morph_bpe` is `train_bpe` with the boundary-marker
#: pre-tokenizer switched on (docs/decisions.md, 2026-09-05, "MorphBPE-hard implemented as
#: boundary-marker pre-tokenisation"); there is no `morph_unigram`, because the constraint
#: is a statement about *merges* and Unigram has none.
ALGO_TRAINERS: dict[str, Callable[..., Path]] = {
    "bpe": train_bpe,
    "unigram": train_unigram,
    "morph_bpe": train_morph_bpe,
}

#: How many offending line numbers a streamed leakage failure names before it summarises.
_MAX_NAMED_OFFENDERS = 5


@dataclass(frozen=True)
class SideSpec:
    """The corpus recipe for one side of a parallel corpus (Experiment 02's `side_spec`).

    Everything that differs between training a raw Sanskrit arm, an English control arm and
    a sandhi-split arm: where the corpus and its manifest go, which exclusion list guards
    it, which sources feed it, and the transform/hash pair the text is written and checked
    with.

    `precheck` is the one side-specific *precondition*, run on the filtered sources just
    before a rebuild (and only then — a corpus that is already current is not rebuilt, so
    its precondition is not re-imposed). Only `sa_split` has one: it verifies that every
    sentence about to be transformed is in the split cache, so a missing split is one error
    naming the count rather than a `MissingSplitError` on the first of 117,720 sentences.
    """

    side: str
    corpus_path: Path
    manifest_path: Path
    exclusion_path: Path
    source_names: list[str]
    transform: Callable[[str], str]
    hash_fn: Callable[[str], str]
    precheck: Callable[[Mapping[str, Sequence[str]]], None] | None = None


# ------------------------------------------------------------------------ corpus currency


def corpus_is_current(corpus_path: Path, manifest_path: Path) -> dict[str, Any] | None:
    """The recorded manifest if `corpus_path` is exactly what `manifest_path` describes.

    "Current" means the corpus file exists, its manifest exists, and the manifest's
    recorded `sha256` matches the file's actual bytes right now — i.e. nothing has touched
    the corpus since that manifest was written. Any mismatch (missing manifest, hand-edited
    corpus, stale digest) returns `None` and the caller rebuilds, rather than trusting a
    corpus that might not be what the manifest says it is.
    """
    if not (corpus_path.exists() and manifest_path.exists()):
        return None
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if recorded.get("sha256") != actual_sha256:
        logger.info("%s exists but does not match %s; rebuilding", corpus_path, manifest_path)
        return None
    logger.info("reusing existing corpus %s (sha256 matches %s)", corpus_path, manifest_path)
    return dict(recorded)


def write_corpus_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    """Write a corpus manifest as pretty JSON, creating its directory if needed."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", manifest_path)


def ensure_training_corpus(
    sources: Mapping[str, Sequence[str]],
    corpus_path: Path,
    exclusion: frozenset[str],
    *,
    manifest_path: Path | None = None,
    transform: Callable[[str], str] | None = None,
    hash_fn: Callable[[str], str] = sentence_hash,
    precheck: Callable[[Mapping[str, Sequence[str]]], None] | None = None,
) -> dict[str, Any]:
    """Build the training corpus from in-memory sources, or reuse it if already current.

    Currency is `corpus_is_current`'s definition. `manifest_path` defaults to
    `manifest.json` beside `corpus_path`; the English (E1) corpus passes `manifest_en.json`
    explicitly, since both corpora live in `data/processed/` and would otherwise overwrite
    each other's manifest. `transform`/`hash_fn` are passed straight through to
    `build_training_corpus` (`identity_transform` + `sentence_hash_en` for English) and
    `hash_fn` is also used for the leakage filter below, so both must match the list
    `exclusion` came from. `precheck`, when given, runs on the *filtered* sources on a
    rebuild only, immediately before the build.

    On a rebuild, `sources` (the *raw*, unfiltered sentences) goes through
    `filter_leaked_sentences` and then `deduplicate_sources` — the same selection
    `experiments/03_sandhi_split/split_corpora.py` splits, so the split side can never be
    asked to transform a sentence the split run did not split (`sanskrit_tok.tokenizers
    .corpus`'s module docstring, "Two deduplications, and why neither is redundant"). The
    written corpus is unaffected: `build_training_corpus` still deduplicates on the
    *transformed* text.

    The manifest `build_training_corpus` returns is extended with three keys before being
    written and returned: `n_in_raw` (the per-source count of `sources` as given, before any
    filtering), `n_leaked_dropped` (what `filter_leaked_sentences` dropped) and
    `n_original_dedup_removed` (what `deduplicate_sources` then dropped). `n_in` stays the
    count actually fed to `build_training_corpus`, i.e. post-filter *and* post-selection.
    """
    manifest_path = corpus_path.parent / "manifest.json" if manifest_path is None else manifest_path
    recorded = corpus_is_current(corpus_path, manifest_path)
    if recorded is not None:
        return recorded

    n_in_raw = {name: len(texts) for name, texts in sources.items()}
    filtered, dropped_counts = filter_leaked_sentences(sources, exclusion, hash_fn)
    selected = deduplicate_sources(filtered)
    if precheck is not None:
        precheck(selected)

    manifest = build_training_corpus(
        selected, corpus_path, exclusion, transform=transform, hash_fn=hash_fn
    )
    manifest["n_in_raw"] = n_in_raw
    manifest["n_leaked_dropped"] = dropped_counts
    manifest["n_original_dedup_removed"] = {
        name: len(filtered[name]) - len(selected[name]) for name in filtered
    }
    write_corpus_manifest(manifest, manifest_path)
    return dict(manifest)


def build_streamed_corpus(
    records: Iterable[tuple[str, str]],
    out_path: Path,
    exclusion: frozenset[str],
    *,
    hash_fn: Callable[[str], str],
    label: str,
    marker: str | None = None,
    log_every: int = 100_000,
) -> dict[str, Any]:
    """Write one training corpus from a stream, asserting it leak-free as it goes.

    Each record is `(check_text, write_text)`: `check_text` is what the exclusion list is
    consulted about — for Experiment 04 always the *sandhied* SLP1 sentence, so all four
    DCS corpora are checked against exactly the same string regardless of what they write —
    and `write_text` is the line that lands in the corpus. Blank `write_text` is dropped
    (counted as `n_blank`), and exact repeats of an already-written line are dropped
    (`n_dedup_removed`).

    The leakage check (CLAUDE.md §2.4) is not `assert_not_excluded`, which needs the whole
    sequence in memory; it is the same test applied per record, accumulating offending
    **1-based record numbers** and raising `LeakageError` at the end so one run reports every
    collision rather than only the first. `n_checked` records how many sentences were
    actually tested, so "0 leaked" is distinguishable from "nothing was checked".

    `marker`, when given, is counted (`n_markers`, total marker characters written) and used
    for `n_out_plain`: how many distinct lines there would be if every marker were stripped
    first. A marked corpus and its unmarked twin are only a matched pair when that number
    equals the twin's `n_out`, and a gap means two differently-annotated copies of one
    sentence — worth seeing in the manifest rather than assuming away.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    seen_plain: set[str] = set()
    offending: list[int] = []
    n_in = n_blank = n_dedup = n_markers = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for index, (check_text, write_text) in enumerate(records, start=1):
            n_in = index
            if hash_fn(check_text) in exclusion:
                offending.append(index)
            line = write_text.strip()
            if not line:
                n_blank += 1
                continue
            if line in seen:
                n_dedup += 1
                continue
            seen.add(line)
            if marker is not None:
                n_markers += line.count(marker)
                seen_plain.add(line.replace(marker, ""))
            handle.write(line + "\n")
            if index % log_every == 0:
                logger.info("%s: %d record(s) read, %d line(s) written", label, index, len(seen))

    if offending:
        named = offending[:_MAX_NAMED_OFFENDERS]
        remaining = len(offending) - _MAX_NAMED_OFFENDERS
        suffix = "" if remaining <= 0 else f" (+{remaining} more)"
        raise LeakageError(
            f"{label}: {len(offending)} sentence(s) match the evaluation exclusion list; "
            f"offending record numbers (1-based): {named}{suffix}"
        )

    manifest: dict[str, Any] = {
        "label": label,
        "n_in": n_in,
        "n_out": len(seen),
        "n_blank": n_blank,
        "n_dedup_removed": n_dedup,
        "n_checked": n_in,
        "n_leaked": 0,
        "exclusion_hashes": len(exclusion),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    if marker is not None:
        manifest["boundary_marker"] = marker
        manifest["n_markers"] = n_markers
        manifest["n_out_plain"] = len(seen_plain)
    logger.info(
        "%s: %d record(s) in, %d line(s) written, %d exact-dup removed, %d blank, wrote %s",
        label,
        n_in,
        len(seen),
        n_dedup,
        n_blank,
        out_path,
    )
    return manifest


# ------------------------------------------------------------------ arm selection and training


def select_arms_to_train(
    arms: Sequence[Mapping[str, Any]], retrain: bool
) -> list[dict[str, Any]]:
    """The configured arms that still need training: those with no `tokenizer.json` yet.

    Adding an arm to a config must not retrain the arms whose numbers a `results.json`
    already reports: a Unigram retrain is not bit-for-bit reproducible (docs/decisions.md,
    "`UnigramTrainer` is not bit-for-bit deterministic"), so it would silently replace the
    artifact behind a published number. `retrain=True` (the `--retrain` flag) overrides and
    returns every arm.
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


def train_arm(
    arm: Mapping[str, Any],
    corpus_path: Path,
    seed: int,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one configured arm and write its `tokenizer.json`; return its results payload.

    Writes to `trained_tokenizer_path(name).parent` — the exact location
    `sanskrit_tok.tokenizers.registry.load_tokenizer` reads from — so the file this run
    produces and the file the registry loads afterwards can never drift apart.

    `extra` is the calling experiment's own per-arm labels (`{"side": ...}` for Experiment
    02, `{"corpus": ...}` for Experiment 04); it lands immediately after `algo` so a
    payload's shape is stable across runs of one experiment.

    Returns the per-arm portion of `results.json`: the run-wide keys (`git_commit`,
    `git_dirty`, `timestamp`, `config`, `manifest`) are filled in by the caller, since they
    are identical across every arm in one run.
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
        **dict(extra or {}),
        "requested_vocab_size": vocab_size,
        "vocab_size": actual_vocab_size,
        "train_seconds": train_seconds,
        "seed": seed,
        "corpus_path": str(corpus_path),
        "tokenizer_path": str(tokenizer_path),
    }


def write_arm_results(
    payload: Mapping[str, Any], name: str, config_src: Path | None = None
) -> Path:
    """Write one arm's `results.json` (and a copy of its config) beside its tokenizer.

    CLAUDE.md §2.9: every experiment writes a `results.json` and a `config.yaml` to its
    output dir, and for a trained arm that dir is the one holding the `tokenizer.json` the
    results describe.
    """
    return write_results(payload, trained_tokenizer_path(name).parent, config_src)


def build_arg_parser(description: str, default_config: Path) -> argparse.ArgumentParser:
    """The `--config` / `--retrain` parser both training scripts expose."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="tokenizer training config YAML (default: tokenizers.yaml beside this script)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="train every configured arm, including ones whose tokenizer.json already exists",
    )
    return parser
