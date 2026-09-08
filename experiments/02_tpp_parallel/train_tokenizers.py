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

The `E1_*_bm` arms (`side: en_bm`) are the same control cut a second way. Matching the two
sides sentence-for-sentence leaves the English corpus 48% larger in bytes (16,554,871
against 11,209,356), because English spells out what sandhi and compounding pack into one
Sanskrit word — so "the same corpus" can mean the same sentences or the same amount of
text, and the two disagree by a third of the English side. The `_bm` corpus
(`data/processed/tok_train_en_bm.txt`) is a deterministic subsample of the pair-matched
one, cut to the Sanskrit corpus's byte count (docs/decisions.md, 2026-09-08, "Byte-matched
English control arms"). Both halves are trained and both are reported; neither is *the*
control, they bracket it.

Every family is trained at 32k, 64k and 128k (CLAUDE.md §2.5: matched vocabulary sizes
within a comparison), so that a controlled result can be checked for whether it is an
artefact of the vocabulary size it was measured at.

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

The corpus-currency, arm-selection and per-arm-results logic lives in
`sanskrit_tok.tokenizers.training`, shared with
`experiments/04_morph_constrained/train_tokenizers.py` so the two experiments cannot drift
apart on any of it. What stays here is what is specific to the parallel corpora: the four
sides, the byte-matched subsample, the split cache, and the reconciliation threshold.

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
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sanskrit_tok.data.exclusion import (
    LeakageError,
    load_exclusion_hashes,
    sentence_hash,
    sentence_hash_en,
)
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.experiment import (
    collect_sources,
    load_config,
    provenance,
    repo_root,
    resolve_path,
)
from sanskrit_tok.sandhi.cache import SplitCache
from sanskrit_tok.sandhi.reconcile import DEFAULT_THRESHOLD, reconcile
from sanskrit_tok.tokenizers.corpus import (
    byte_matched_prefix,
    deduplicate_sources,
    identity_transform,
    slp1_transform,
)
from sanskrit_tok.tokenizers.training import (
    SideSpec,
    build_arg_parser,
    corpus_is_current,
    ensure_training_corpus,
    select_arms_to_train,
    train_arm,
    write_arm_results,
    write_corpus_manifest,
)

logger = logging.getLogger("train_tokenizers")

#: The sides of the parallel corpora an arm can be trained on: `sa` (raw Sanskrit, the
#: T1/T2 arms), `en` (English, the pair-matched E1 control arms), `en_bm` (the same English
#: cut to the Sanskrit corpus's byte count, the `E1_*_bm` arms) and `sa_split`
#: (sandhi-split Sanskrit, the T4 arms). `tokenizers.yaml` names one per arm under `side`;
#: an arm without one is Sanskrit, which is what every arm was before the E1 family existed.
SANSKRIT_SIDE = "sa"
ENGLISH_SIDE = "en"

#: The byte-matched English side (`E1_*_bm`): the same English corpus cut to the Sanskrit
#: corpus's UTF-8 byte count. It is a fourth side rather than a flag on `en` because it has
#: its own corpus file, its own manifest and its own precondition (the pair-matched English
#: corpus must exist first, since it is what gets subsampled); the exclusion list, the
#: sources and the hash function are `en`'s.
ENGLISH_BM_SIDE = "en_bm"

#: The sandhi-split Sanskrit side (Experiment 03): the same Devanagari sentences as `sa`,
#: written to the corpus as the *split* SLP1 text rather than the raw SLP1 text. It is a
#: third side rather than a flag on `sa` because it has its own corpus file, its own
#: manifest and its own precondition (a populated split cache); the exclusion list, the
#: sources and the hash function are `sa`'s.
SPLIT_SIDE = "sa_split"
SIDES = (SANSKRIT_SIDE, ENGLISH_SIDE, ENGLISH_BM_SIDE, SPLIT_SIDE)

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
    """Which side an arm trains on: `"sa"` (default), `"en"`, `"en_bm"` or `"sa_split"`.

    Defaults to Sanskrit for an arm with no `side` key — every arm predating the E1
    family is Sanskrit, and defaulting keeps an older `tokenizers.yaml` working. An
    unrecognised value raises rather than silently training an arm on the wrong corpus,
    which would be invisible in the results.
    """
    side = str(arm.get("side", SANSKRIT_SIDE))
    if side not in SIDES:
        raise ValueError(f"{arm.get('name')}: unknown side {side!r}; expected one of {list(SIDES)}")
    return side


# -------------------------------------------------------------------------------- main


def side_spec(side: str, config: Mapping[str, Any], root: Path) -> SideSpec:
    """Everything that differs between the Sanskrit, English and split training corpora.

    Sanskrit reads `corpus_path`/`sources`/`exclusion_path` and writes `manifest.json`
    beside its corpus, transliterating to SLP1 and hashing with `sentence_hash`. English
    reads `english_corpus_path`/`english_sources`/`english_exclusion_path`, writes
    `manifest_en.json`, keeps its text as written (`identity_transform`) and hashes with
    `sentence_hash_en`. The byte-matched English side is that side with one path changed —
    `english_bm_corpus_path` and `manifest_en_bm.json` — since it is the same text,
    the same exclusion list and the same hash function, only less of it; what fills that
    corpus is `build_byte_matched_corpus`, not `ensure_training_corpus`.
    The sandhi-split side reads the *same* sources, exclusion list and
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
    if side == ENGLISH_BM_SIDE:
        corpus_path = resolve_path(str(config["english_bm_corpus_path"]), root)
        return SideSpec(
            side=side,
            corpus_path=corpus_path,
            manifest_path=corpus_path.parent / "manifest_en_bm.json",
            exclusion_path=resolve_path(str(config["english_exclusion_path"]), root),
            source_names=list(config["english_sources"]),
            transform=identity_transform,
            hash_fn=sentence_hash_en,
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


# ------------------------------------------------------------ the byte-matched English side


def corpus_size(path: Path) -> dict[str, int]:
    """`n_lines`, `bytes` and `chars` of one training corpus file.

    Bytes and characters are both recorded because they answer different questions and
    differ by side: SLP1 is ASCII, so the Sanskrit corpus's two numbers coincide, while an
    English corpus written in UTF-8 differs from its character count wherever a
    non-ASCII character occurs. The byte-matched corpus is cut on *bytes*, so the character
    ratio it lands on is a measurement rather than a target, and both belong in the
    manifest.
    """
    data = path.read_bytes()
    text = data.decode("utf-8")
    return {
        "n_lines": text.count("\n"),
        "bytes": len(data),
        "chars": len(text),
    }


def training_corpus_stats(config: Mapping[str, Any], root: Path) -> dict[str, dict[str, int]]:
    """Size of every training corpus that exists, keyed by side (`sa`, `en`, `en_bm`).

    Recorded in each arm's `results.json` so that a number read off a trained arm can be
    read against how much text the arm actually saw — which is the whole point of the
    byte-matched family. A corpus that has not been built yet is simply absent, since this
    runs for whichever sides the run touched.
    """
    paths = {
        SANSKRIT_SIDE: resolve_path(str(config["corpus_path"]), root),
        ENGLISH_SIDE: resolve_path(str(config["english_corpus_path"]), root),
        ENGLISH_BM_SIDE: resolve_path(str(config["english_bm_corpus_path"]), root),
    }
    return {side: corpus_size(path) for side, path in paths.items() if path.exists()}


def build_byte_matched_corpus(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Build (or reuse) the byte-matched English corpus, and return its manifest.

    Derived from the *pair-matched* English corpus rather than from the sources: it must be
    a subsample of exactly the text the `E1_*` arms train on, so that the only difference
    between an `E1_bpe_32k` and an `E1_bpe_32k_bm` is how much of that text was kept.
    The pair-matched corpus is therefore built first (or reused, if current), then cut to
    the Sanskrit corpus's byte count by `byte_matched_prefix` — a shuffle at `seed` and the
    prefix that first reaches the target.

    The Sanskrit corpus is the reference and must exist; it is the corpus whose size is
    being matched, and reading its byte count from the file rather than from a config
    constant means the two can never disagree.

    The leakage assertion (CLAUDE.md §2.4) is re-run on the selected lines against
    `english_exclusion_path`, even though every line comes from an already-filtered corpus:
    the check is cheap, and "this corpus was checked" is a property each corpus should
    carry itself rather than inherit by argument. `n_checked` and `n_leaked` record it.
    """
    spec = side_spec(ENGLISH_BM_SIDE, config, root)
    source_path = resolve_path(str(config["english_corpus_path"]), root)
    reference_path = resolve_path(str(config["corpus_path"]), root)
    seed = int(config.get("byte_match_seed", config.get("seed", 0)))

    recorded = corpus_is_current(spec.corpus_path, spec.manifest_path)
    if recorded is not None:
        return recorded

    english_manifest = build_side_corpus(side_spec(ENGLISH_SIDE, config, root))
    if not reference_path.exists():
        raise FileNotFoundError(
            f"the byte-matched English corpus is cut to the size of {reference_path}, which "
            "does not exist; build the Sanskrit training corpus first (train any T1/T2 arm)"
        )
    reference = corpus_size(reference_path)
    source = corpus_size(source_path)

    lines = source_path.read_text(encoding="utf-8").splitlines()
    selected = byte_matched_prefix(lines, reference["bytes"], seed=seed)

    exclusion = load_exclusion_hashes(spec.exclusion_path)
    leaked = [line for line in selected if spec.hash_fn(line) in exclusion]
    if leaked:
        raise LeakageError(
            f"{spec.side}: {len(leaked)} selected line(s) match the evaluation exclusion "
            f"list {spec.exclusion_path}; the pair-matched corpus it was cut from is stale"
        )

    spec.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.corpus_path.open("w", encoding="utf-8") as handle:
        for line in selected:
            handle.write(line + "\n")
    written = corpus_size(spec.corpus_path)

    manifest: dict[str, Any] = {
        "label": spec.side,
        "source_corpus": str(source_path),
        "source": source,
        "reference_corpus": str(reference_path),
        "reference": reference,
        "target_bytes": reference["bytes"],
        "seed": seed,
        "n_out": written["n_lines"],
        "bytes": written["bytes"],
        "chars": written["chars"],
        "bytes_ratio_to_reference": written["bytes"] / reference["bytes"],
        "chars_ratio_to_reference": written["chars"] / reference["chars"],
        "lines_ratio_to_source": (
            written["n_lines"] / source["n_lines"] if source["n_lines"] else 0.0
        ),
        "n_checked": len(selected),
        "n_leaked": 0,
        "exclusion_hashes": len(exclusion),
        "source_manifest": dict(english_manifest),
        "sha256": hashlib.sha256(spec.corpus_path.read_bytes()).hexdigest(),
    }
    write_corpus_manifest(manifest, spec.manifest_path)
    logger.info(
        "%s: %d of %d line(s) kept, %d bytes (%.4f x the Sanskrit corpus's %d), %d chars",
        spec.side,
        written["n_lines"],
        source["n_lines"],
        written["bytes"],
        manifest["bytes_ratio_to_reference"],
        reference["bytes"],
        written["chars"],
    )
    return manifest


def build_corpus_for_side(side: str, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The corpus for `side`, built or reused: `en_bm` is derived, every other side is built.

    One dispatch point so `main` does not have to know that one of the four sides is a
    subsample of another rather than an assembly of sources.
    """
    if side == ENGLISH_BM_SIDE:
        return build_byte_matched_corpus(config, root)
    return dict(build_side_corpus(side_spec(side, config, root)))


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
    parser = build_arg_parser(__doc__ or "", Path(__file__).resolve().with_name("tokenizers.yaml"))
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
    manifests: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        if any(arm_side(arm) == side for arm in to_train):
            manifests[side] = build_corpus_for_side(side, config, root)

    # One reading of every training corpus's size, shared by every arm this run trains, so
    # a `results.json` says how much text its arm saw next to what the arm cost
    # (docs/decisions.md, 2026-09-08, "Byte-matched English control arms").
    corpus_stats = training_corpus_stats(config, root)

    # One provenance reading for the whole run (one `git` call, one WARNING if the tree
    # is dirty), but a per-arm timestamp: the arms are written minutes apart and each
    # `results.json` records when its own tokenizer was produced.
    run_provenance = provenance(root)
    for arm in to_train:
        name = str(arm["name"])
        side = arm_side(arm)
        spec = side_spec(side, config, root)
        logger.info("training %s (side=%s) ...", name, side)
        payload = train_arm(arm, spec.corpus_path, seed, extra={"side": side})
        payload.update(run_provenance)
        payload["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        payload["config"] = config
        payload["manifest"] = manifests[side]
        payload["training_corpus_sizes"] = corpus_stats
        # Beside the tokenizer it describes (CLAUDE.md §2.9): `results.json` and a copy of
        # the config land in the same directory `train_arm` just wrote `tokenizer.json` to.
        write_arm_results(payload, name, args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
