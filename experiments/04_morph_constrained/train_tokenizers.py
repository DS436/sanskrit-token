"""Train the twelve DCS-trained Experiment 04 arms, T5/T6 included (exp04 Task 3).

Four training corpora are streamed out of `data/processed/dcs/train.jsonl` — the sandhied
text, its gold "oracle" segmentation, and the boundary-marked form of each — and twelve arms
are trained from them at matched 32k/64k vocabulary sizes (CLAUDE.md §2.5): the T5/T6
morpheme-constrained arms and, from the same sentences with the same trainer, the
unconstrained T1/T2/T4 arms they are compared against. `tokenizers.yaml` is the authority on
which corpus each arm trains on and why; this script is the plumbing.

Everything reusable is `sanskrit_tok.tokenizers.training`, shared with
`experiments/02_tpp_parallel/train_tokenizers.py`: skip-if-trained (`--retrain` overrides),
corpus currency by sha256 against a manifest, `train_arm`, and the per-arm `results.json`
written beside the `tokenizer.json` it describes (CLAUDE.md §2.9). What is local to this
script is the DCS corpus source, which cannot use the shared `ensure_training_corpus`: that
one takes its sentences in memory, and `train.jsonl` is 720,510 lines and 437 MB, so the
corpora are built with `build_streamed_corpus` from an iterator that parses one line at a
time and holds only the written-line dedup set.

**Leakage.** Every sentence is checked against `data/exclusion_hashes.txt` on its *sandhied*
SLP1 text via `sentence_hash_slp1` — the same string for all four corpora, so no corpus can
be checked more loosely than another (CLAUDE.md §2.4). The ingestion already dropped every
held-out collision, so this is expected to pass with zero; the manifest records how many
sentences were actually checked, so "clean" is distinguishable from "not checked".

**The constraint audit.** After training, every BPE arm with a `check_corpus` is measured
with `morph_bpe.assert_no_cross_boundary_merges` and the result goes into its
`results.json`. The T5/T6 arms are measured against their own marked training text; the
unconstrained T1/T4 BPE arms are measured against the same marked corpus as the control, so
each constrained number is reported next to what an unconstrained tokenizer scores on
identical text. See `morph_bpe`'s module docstring for why the constrained number is a
residue rather than a guaranteed zero.

Run it with `uv run python experiments/04_morph_constrained/train_tokenizers.py`. Relative
paths in the config are resolved against the repository root, so the working directory does
not matter. Training all twelve arms from scratch is a long job (the Unigram arms
especially); it is resumable, since an arm whose `tokenizer.json` exists is skipped.
"""

import argparse
import json
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sanskrit_tok.data.boundaries import BOUNDARY_MARKER
from sanskrit_tok.data.exclusion import load_exclusion_hashes, sentence_hash_slp1
from sanskrit_tok.experiment import load_config, provenance, repo_root, resolve_path
from sanskrit_tok.tokenizers.morph_bpe import assert_no_cross_boundary_merges
from sanskrit_tok.tokenizers.training import (
    build_arg_parser,
    build_streamed_corpus,
    corpus_is_current,
    select_arms_to_train,
    train_arm,
    write_arm_results,
    write_corpus_manifest,
)

logger = logging.getLogger("exp04_train_tokenizers")

#: The `GoldSentence` field every corpus is checked against, whatever it writes: the
#: sandhied SLP1 sentence, which is what `data/exclusion_hashes.txt` holds for DCS.
CHECK_FIELD = "text_slp1"


@dataclass(frozen=True)
class CorpusSpec:
    """One DCS-derived training corpus: which field it writes, where, and whether it is marked."""

    name: str
    field: str
    corpus_path: Path
    manifest_path: Path
    marked: bool

    @property
    def marker(self) -> str | None:
        """The boundary marker to count in this corpus, or `None` for an unmarked one."""
        return BOUNDARY_MARKER if self.marked else None


def corpus_specs(config: Mapping[str, Any], root: Path) -> dict[str, CorpusSpec]:
    """`tokenizers.yaml`'s `corpora` block as `CorpusSpec`s, paths resolved against `root`."""
    specs: dict[str, CorpusSpec] = {}
    for name, entry in dict(config["corpora"]).items():
        specs[str(name)] = CorpusSpec(
            name=str(name),
            field=str(entry["field"]),
            corpus_path=resolve_path(str(entry["corpus_path"]), root),
            manifest_path=resolve_path(str(entry["manifest_path"]), root),
            marked=bool(entry["marked"]),
        )
    return specs


def iter_dcs_records(jsonl_path: Path, field: str) -> Iterator[tuple[str, str]]:
    """Stream `(sandhied SLP1, field value)` from a DCS split's jsonl, one line at a time.

    The first element is what the exclusion list is consulted about and the second is what
    is written to the corpus; for `dcs_raw` they are the same string. Never materialises the
    file, which is why this is a generator and not a list comprehension.
    """
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield str(record[CHECK_FIELD]), str(record[field])


def ensure_dcs_corpus(
    spec: CorpusSpec, jsonl_path: Path, exclusion: frozenset[str]
) -> dict[str, Any]:
    """Build `spec`'s corpus from `jsonl_path`, or reuse it if it is already current.

    Currency is `corpus_is_current`'s sha256-against-manifest test, the same one the
    Experiment 02 corpora use, so a corpus that has been touched since its manifest was
    written is rebuilt rather than trusted.
    """
    recorded = corpus_is_current(spec.corpus_path, spec.manifest_path)
    if recorded is not None:
        return recorded

    logger.info("building %s from %s field %r", spec.corpus_path, jsonl_path, spec.field)
    manifest = build_streamed_corpus(
        iter_dcs_records(jsonl_path, spec.field),
        spec.corpus_path,
        exclusion,
        hash_fn=sentence_hash_slp1,
        label=spec.name,
        marker=spec.marker,
    )
    manifest["source_jsonl"] = str(jsonl_path)
    manifest["field"] = spec.field
    manifest["corpus_path"] = str(spec.corpus_path)
    write_corpus_manifest(manifest, spec.manifest_path)
    return manifest


def needed_corpora(arms: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every corpus the selected arms train on *or* are audited against, in config order.

    The audit corpus matters as much as the training one: `T1_bpe_raw_32k_dcs` trains on
    `dcs_raw` and is checked against `dcs_raw_marked`, so building only what is trained on
    would leave the control measurement with nothing to read.
    """
    names: list[str] = []
    for arm in arms:
        for key in ("corpus", "check_corpus"):
            name = arm.get(key)
            if name is not None and str(name) not in names:
                names.append(str(name))
    return names


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
    sample = int(config.get("cross_boundary_sample", 2000))
    arms: list[dict[str, Any]] = list(config["arms"])
    if not arms:
        raise ValueError(f"{args.config}: 'arms' is empty; nothing to train")

    to_train = select_arms_to_train(arms, args.retrain)
    if not to_train:
        logger.info("every configured arm is already trained; nothing to do (--retrain forces)")
        return 0

    specs = corpus_specs(config, root)
    jsonl_path = resolve_path(str(config["dcs_train_jsonl"]), root)
    exclusion = load_exclusion_hashes(resolve_path(str(config["exclusion_path"]), root))
    logger.info("loaded %d exclusion hash(es)", len(exclusion))

    manifests: dict[str, dict[str, Any]] = {}
    for name in needed_corpora(to_train):
        manifests[name] = ensure_dcs_corpus(specs[name], jsonl_path, exclusion)
        logger.info(
            "corpus %s: n_in=%s n_out=%s n_dedup_removed=%s n_markers=%s",
            name,
            manifests[name].get("n_in"),
            manifests[name].get("n_out"),
            manifests[name].get("n_dedup_removed"),
            manifests[name].get("n_markers"),
        )

    # One provenance reading for the whole run, but a per-arm timestamp: the arms are
    # written minutes apart and each `results.json` records when its own tokenizer landed.
    run_provenance = provenance(root)
    for arm in to_train:
        name = str(arm["name"])
        corpus_name = str(arm["corpus"])
        spec = specs[corpus_name]
        logger.info("training %s (corpus=%s) ...", name, corpus_name)
        payload = train_arm(
            arm,
            spec.corpus_path,
            seed,
            extra={
                "corpus": corpus_name,
                "boundary_marker": BOUNDARY_MARKER if str(arm["algo"]) == "morph_bpe" else None,
            },
        )
        check_name = arm.get("check_corpus")
        if check_name is not None:
            payload["cross_boundary_check"] = assert_no_cross_boundary_merges(
                Path(str(payload["tokenizer_path"])),
                specs[str(check_name)].corpus_path,
                sample,
            )
        payload.update(run_provenance)
        payload["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        payload["config"] = config
        payload["manifest"] = manifests[corpus_name]
        write_arm_results(payload, name, args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
