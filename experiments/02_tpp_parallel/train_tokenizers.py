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
import functools
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as RawTokenizer

from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash, sentence_hash_en
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.experiment import (
    collect_sources,
    load_config,
    provenance,
    repo_root,
    resolve_path,
    write_results,
)
from sanskrit_tok.sandhi.cache import SplitCache
from sanskrit_tok.sandhi.reconcile import DEFAULT_THRESHOLD, reconcile
from sanskrit_tok.tokenizers.corpus import (
    build_training_corpus,
    deduplicate_sources,
    filter_leaked_sentences,
    identity_transform,
    slp1_transform,
)
from sanskrit_tok.tokenizers.registry import trained_tokenizer_path
from sanskrit_tok.tokenizers.train_bpe import train_bpe
from sanskrit_tok.tokenizers.train_unigram import train_unigram

logger = logging.getLogger("train_tokenizers")

#: The sides of the parallel corpora an arm can be trained on: `sa` (raw Sanskrit, the
#: T1/T2 arms), `en` (English, the E1 control arms) and `sa_split` (sandhi-split Sanskrit,
#: the T4 arms). `tokenizers.yaml` names one per arm under `side`; an arm without one is
#: Sanskrit, which is what every arm was before the E1 family existed.
SANSKRIT_SIDE = "sa"
ENGLISH_SIDE = "en"

#: The sandhi-split Sanskrit side (Experiment 03): the same Devanagari sentences as `sa`,
#: written to the corpus as the *split* SLP1 text rather than the raw SLP1 text. It is a
#: third side rather than a flag on `sa` because it has its own corpus file, its own
#: manifest and its own precondition (a populated split cache); the exclusion list, the
#: sources and the hash function are `sa`'s.
SPLIT_SIDE = "sa_split"
SIDES = (SANSKRIT_SIDE, ENGLISH_SIDE, SPLIT_SIDE)

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

    Everything that differs between training a raw Sanskrit arm, an English control arm
    and a sandhi-split arm: where the corpus and its manifest go, which exclusion list
    guards it, which sources feed it, and the transform/hash pair the text is written and
    checked with.

    `precheck` is the one side-specific *precondition*, run on the filtered sources just
    before a rebuild (and only then — a corpus that is already current is not rebuilt, so
    its precondition is not re-imposed). Only `sa_split` has one: it verifies that every
    sentence about to be transformed is in the split cache, so a missing split is one
    error naming the count rather than a `MissingSplitError` on the first of 117,720
    sentences.
    """

    side: str
    corpus_path: Path
    manifest_path: Path
    exclusion_path: Path
    source_names: list[str]
    transform: Callable[[str], str]
    hash_fn: Callable[[str], str]
    precheck: Callable[[Mapping[str, Sequence[str]]], None] | None = None


# ------------------------------------------------------------------ the sandhi-split side


class MissingSplitError(RuntimeError):
    """A sentence the `sa_split` corpus needs has not been sandhi-split yet.

    Splitting is a five-hour job against a 2.3 GB model and belongs to
    `experiments/03_sandhi_split/split_corpora.py`, never to this script: loading the
    model here would turn "train four tokenizers" into an overnight run nobody asked for,
    and would do it silently. So the split text is read from the cache only, and a gap is
    this error — which is also why it names how many sentences are missing rather than
    just the first one.
    """


@functools.cache
def _split_cache(path: Path) -> SplitCache:
    """The split cache at `path`, loaded once per process.

    `side_spec` is called once per arm, and the cache file is ~137k lines; without this the
    four T4 arms would re-read and re-parse the whole thing four times over.
    """
    return SplitCache(path)


class SplitTransform:
    """Devanagari -> reconciled split SLP1, read from the split cache. Never loads a model.

    The cache holds what the splitter's model actually emitted (SLP1, one whitespace unit
    per segment). What the T4 corpus is trained on is that output *reconciled* against the
    raw sentence — the model drops punctuation and the occasional loanword, and training on
    text with content deleted would bias every T4 number in the experiment's favour
    (docs/decisions.md, "T4 text is the model's segmentation reconciled against the raw
    sentence"). Reconciliation is pure and cheap, so it is recomputed here rather than
    cached, and `threshold` can change without re-splitting anything.
    """

    def __init__(self, cache_path: Path, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.cache_path = cache_path
        self.threshold = threshold

    def __call__(self, text: str) -> str:
        cached = _split_cache(self.cache_path).get(text)
        if cached is None:
            raise MissingSplitError(
                f"sentence not in the sandhi split cache {self.cache_path}: {text[:60]!r}. "
                "Run experiments/03_sandhi_split/split_corpora.py first; this script never "
                "loads the splitter model."
            )
        return reconcile(to_slp1(text, "devanagari"), cached, threshold=self.threshold).text


def resolve_split_threshold(config: Mapping[str, Any], manifest_path: Path) -> float:
    """The reconciliation threshold, single-sourced from the split manifest.

    The manifest records what the corpus was actually split and reconciled with, so it
    wins; `tokenizers.yaml`'s `split_reconcile_threshold` is a declaration of intent and is
    *checked* against it rather than trusted. A disagreement raises: reconciling the
    training corpus at a different threshold from the one the manifest and the evaluation
    jsonls carry would give the T4 arms a training text nothing else in the experiment
    shares, and the only symptom would be numbers that quietly do not add up.

    Falls back to the config value (then `DEFAULT_THRESHOLD`) when no manifest exists yet —
    the state the repository is in before the split run has finished.
    """
    declared = config.get("split_reconcile_threshold")
    if not manifest_path.exists():
        return DEFAULT_THRESHOLD if declared is None else float(declared)
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("reconcile_threshold")
    if recorded is None:
        return DEFAULT_THRESHOLD if declared is None else float(declared)
    if declared is not None and float(declared) != float(recorded):
        raise ValueError(
            f"split_reconcile_threshold in the config is {float(declared)} but "
            f"{manifest_path} records {float(recorded)}; the corpus was split and "
            "reconciled at the manifest's value. Fix the config, or re-run "
            "experiments/03_sandhi_split/split_corpora.py at the intended threshold."
        )
    return float(recorded)


def split_cache_precheck(cache_path: Path) -> Callable[[Mapping[str, Sequence[str]]], None]:
    """A `SideSpec.precheck` asserting every sentence in `sources` is in the split cache.

    Raises `MissingSplitError` naming how many of how many sentences are missing and which
    cache file was consulted, so the fix ("finish the split run") is obvious from the one
    line and does not require re-running the build to discover a second missing sentence.

    The set checked is `deduplicate_sources(sources)` — the same selection
    `split_corpora.py` splits and `ensure_training_corpus` feeds the builder — so "the
    cache covers the corpus" means exactly what it says.
    """

    def check(sources: Mapping[str, Sequence[str]]) -> None:
        cache = _split_cache(cache_path)
        # Deduplicated the same way the split run selected its sentences, so the count is
        # over exactly the set `split_corpora.py` was supposed to have split. Idempotent:
        # `ensure_training_corpus` has already applied the same selection.
        texts = [text for source in deduplicate_sources(sources).values() for text in source]
        missing = sum(1 for text in texts if cache.get(text) is None)
        if missing:
            raise MissingSplitError(
                f"{missing} of {len(texts)} training sentence(s) are not in the sandhi "
                f"split cache {cache_path} ({len(cache)} entry/entries). Run "
                "experiments/03_sandhi_split/split_corpora.py to completion first."
            )
        logger.info(
            "split cache %s covers all %d training sentence(s)", cache_path, len(texts)
        )

    return check


# --------------------------------------------------------------------------- corpus


def arm_side(arm: Mapping[str, Any]) -> str:
    """Which side an arm trains on: `"sa"` (default), `"en"` or `"sa_split"`.

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


def ensure_training_corpus(
    sources: Mapping[str, Sequence[str]],
    corpus_path: Path,
    exclusion: frozenset[str],
    *,
    manifest_path: Path | None = None,
    transform: Callable[[str], str] | None = None,
    hash_fn: Callable[[str], str] = sentence_hash,
    precheck: Callable[[Mapping[str, Sequence[str]]], None] | None = None,
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
    `precheck`, when given, runs on the *filtered* sources on a rebuild only, immediately
    before the build — the `sa_split` side uses it to fail once, with a count, if the
    sandhi split cache does not cover the corpus.

    On a rebuild, `sources` (the *raw*, unfiltered sentences from `collect_sources`) goes
    through `filter_leaked_sentences` and then `deduplicate_sources` — the same selection
    `experiments/03_sandhi_split/split_corpora.py` splits, so the split side can never be
    asked to transform a sentence the split run did not split (`sanskrit_tok.tokenizers
    .corpus`'s module docstring, "Two deduplications, and why neither is redundant": two
    Devanagari spellings can share one SLP1 form). The written corpus is unaffected:
    `build_training_corpus` still deduplicates on the *transformed* text, so those two
    spellings still collapse to one line.

    The manifest `build_training_corpus` returns is extended with three keys before being
    written and returned: `n_in_raw` (the per-source count of `sources` as given, before
    any filtering), `n_leaked_dropped` (what `filter_leaked_sentences` dropped) and
    `n_original_dedup_removed` (what `deduplicate_sources` then dropped). `n_in` — already
    in the manifest `build_training_corpus` returns — stays the count actually fed to
    `build_training_corpus`, i.e. post-filter *and* post-selection.
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


# -------------------------------------------------------------------------------- main


def side_spec(side: str, config: Mapping[str, Any], root: Path) -> SideSpec:
    """Everything that differs between the Sanskrit and English training corpora.

    Sanskrit reads `corpus_path`/`sources`/`exclusion_path` and writes `manifest.json`
    beside its corpus, transliterating to SLP1 and hashing with `sentence_hash`. English
    reads `english_corpus_path`/`english_sources`/`english_exclusion_path`, writes
    `manifest_en.json`, keeps its text as written (`identity_transform`) and hashes with
    `sentence_hash_en`. The sandhi-split side reads the *same* sources, exclusion list and
    hash function as Sanskrit — the sentences must be identical for the T1/T4 comparison to
    be matched — but writes `split_corpus_path` and `manifest_split.json` and transforms
    through the split cache (`SplitTransform`) instead of transliterating.
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
    if side == SPLIT_SIDE:
        corpus_path = resolve_path(str(config["split_corpus_path"]), root)
        cache_path = resolve_path(str(config["split_cache_path"]), root)
        manifest_path = resolve_path(
            str(config.get("split_manifest_path", "data/processed/split/manifest.json")), root
        )
        threshold = resolve_split_threshold(config, manifest_path)
        return SideSpec(
            side=side,
            corpus_path=corpus_path,
            manifest_path=corpus_path.parent / "manifest_split.json",
            exclusion_path=resolve_path(str(config["exclusion_path"]), root),
            source_names=list(config["sources"]),
            transform=SplitTransform(cache_path, threshold),
            hash_fn=sentence_hash,
            precheck=split_cache_precheck(cache_path),
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
        precheck=spec.precheck,
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

    # One provenance reading for the whole run (one `git` call, one WARNING if the tree
    # is dirty), but a per-arm timestamp: the arms are written minutes apart and each
    # `results.json` records when its own tokenizer was produced.
    run_provenance = provenance(root)
    for arm in to_train:
        name = str(arm["name"])
        side = arm_side(arm)
        spec = side_spec(side, config, root)
        logger.info("training %s (side=%s) ...", name, side)
        payload = train_arm(arm, spec.corpus_path, seed)
        payload.update(run_provenance)
        payload["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        payload["config"] = config
        payload["manifest"] = manifests[side]
        # Beside the tokenizer it describes (CLAUDE.md §2.9): `results.json` and a copy of
        # the config land in the same directory `train_arm` just wrote `tokenizer.json` to.
        write_results(payload, trained_tokenizer_path(name).parent, args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
