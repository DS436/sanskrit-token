"""Ingest the Digital Corpus of Sanskrit into SLP1 sentences with gold boundaries.

Experiment 04, Task 1. Downloads the pinned sparse checkout (`sanskrit_tok.data.dcs`),
parses every `.conllu` file, converts IAST to SLP1, locates each surface word's gold
segment boundaries in the sandhied surface and its derived stem/ending boundaries
(`sanskrit_tok.data.boundaries`), and writes

    data/processed/dcs/train.jsonl      one GoldSentence per line, + sent_id/text_id/text_iast
    data/processed/dcs/heldout.jsonl    the same, for the held-out texts
    data/processed/dcs/manifest.json    commit, counts, alignment rate, provenance

Three rules the numbers depend on (docs/decisions.md, 2026-09-05):

* **Held out by whole text.** 5% of `text_id`s, chosen with `random.Random(seed)` over the
  sorted unique ids; every sentence of a held-out text goes to `heldout.jsonl`. Splitting
  by sentence would put two ślokas of one work on both sides of the wall.
* **No leakage.** A sentence whose sandhied SLP1 form hashes into `data/exclusion_hashes.txt`
  is dropped from *both* splits and counted. The DCS held-out hashes that a previous run of
  this script contributed to that list are subtracted from it first, so re-running does not
  progressively delete the held-out split it created (`_previous_dcs_hashes`). Separately, a
  training sentence whose text is identical to a held-out sentence's is dropped too: DCS's
  formulaic lines recur verbatim across texts, so a whole-text split alone does not keep the
  two sides disjoint (`n_dropped_heldout_duplicate`).
* **`# text` is the sentence.** Words come from the `# text` line; a word the token block
  does not reconstruct carries no gold segmentation and is counted as unaligned
  (`n_sentences_text_mismatch` counts the sentences where this happens).

Three invariants are checked rather than assumed, because each one failing silently would
show up only as a MorphScore that looks too good: every file must carry a `## text_id:`
header (`read_text_ids`; a `-1` pseudo-text would route unrelated files together), every
sentence's own `text_id` must equal its file header's (routing is by header), and no
`text_id` may reach both splits (`assert_splits_disjoint`, before the manifest is written).
All three raise `IngestError`.

The exclusion list itself is regenerated afterwards, with `dcs_heldout` appended:

    uv run python experiments/04_morph_constrained/ingest_dcs.py --config .../dcs.yaml
    uv run python experiments/02_tpp_parallel/build_exclusion.py

770,731 sentences over 16,051 files in ~92 s on a laptop (transliteration is memoised);
progress is logged every 50 files per pass. The 1.3 GB checkout it downloads on a cold
run is the slow part, so run that one with `nohup` and poll.
"""

import argparse
import json
import logging
import random
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

from sanskrit_tok.data.boundaries import build_gold_sentence, text_words_slp1
from sanskrit_tok.data.dcs import (
    DCS_COMMIT,
    DCS_LICENCE,
    DCS_REPO,
    conllu_files,
    download_dcs,
    iter_conllu_sentences,
    read_text_id,
    text_matches_token_block,
)
from sanskrit_tok.data.exclusion import (
    EXCLUSION_PATH,
    load_exclusion_hashes,
    sentence_hash_slp1,
)
from sanskrit_tok.experiment import load_config, provenance, repo_root, resolve_path, sanitize_json

logger = logging.getLogger("ingest_dcs")

#: How often the file loop logs progress.
_PROGRESS_EVERY = 50

TRAIN_FILENAME = "train.jsonl"
HELDOUT_FILENAME = "heldout.jsonl"
MANIFEST_FILENAME = "manifest.json"

#: What `read_text_id` returns for a file with no `## text_id:` header. Such a file is
#: fatal here rather than routed as a `-1` pseudo-text: every file carrying it would be
#: lumped into one imaginary text and either held out together or trained on together,
#: which is precisely the whole-text split this experiment depends on being wrong.
MISSING_TEXT_ID = -1

#: How many offending paths an error names before it stops listing them.
_MAX_NAMED = 5


class IngestError(RuntimeError):
    """A corpus invariant the split depends on does not hold; the ingestion must stop."""


def assign_heldout_texts(
    text_ids: Iterable[int], *, fraction: float, seed: int
) -> set[int]:
    """The `text_id`s held out: `fraction` of the sorted unique ids, sampled with `seed`.

    Deterministic in the *set* of ids, not their order — the caller scans files in
    directory order and must get the same split whatever that order is. The count is
    floored, but never below one: a corpus small enough to floor to zero still needs a
    held-out set for anything downstream to evaluate on.
    """
    unique = sorted(set(text_ids))
    if not unique:
        return set()
    count = min(len(unique), max(1, int(len(unique) * fraction)))
    return set(random.Random(seed).sample(unique, count))


def read_text_ids(paths: Sequence[Path]) -> dict[Path, int]:
    """Every file's `## text_id`, or raise `IngestError` naming the files that have none.

    The held-out split is by *text*, and a file whose header the parser could not read
    would be assigned `MISSING_TEXT_ID`; all such files would then share one pseudo-text
    and be routed together, silently. A missing header means the checkout or the corpus
    format changed, so the ingestion stops instead of guessing.
    """
    text_id_of = {path: read_text_id(path) for path in paths}
    missing = [path for path, text_id in text_id_of.items() if text_id == MISSING_TEXT_ID]
    if missing:
        named = ", ".join(str(path) for path in missing[:_MAX_NAMED])
        remaining = len(missing) - _MAX_NAMED
        suffix = "" if remaining <= 0 else f" (+{remaining} more)"
        raise IngestError(
            f"{len(missing)} file(s) have no '## text_id:' header and cannot be routed to a "
            f"split: {named}{suffix}"
        )
    return text_id_of


def assert_splits_disjoint(train_texts: set[int], heldout_texts: set[int]) -> None:
    """Raise `IngestError` unless no `text_id` reached both splits.

    The two passes route by file, and every sentence is checked against its own file's
    header, so this cannot fail as the code stands — which is the point: it is the
    assertion that says so, and it runs before the manifest is written rather than being
    rediscovered when a MorphScore number looks too good.
    """
    shared = sorted(train_texts & heldout_texts)
    if shared:
        raise IngestError(
            f"{len(shared)} text_id(s) reached both splits, so the held-out set leaks into "
            f"training: {shared[:_MAX_NAMED]}"
        )


def _previous_dcs_hashes(out_dir: Path) -> frozenset[str]:
    """Hashes of a previous run's held-out sentences, so re-running stays idempotent.

    `build_exclusion.py` writes DCS's own held-out sentences into
    `data/exclusion_hashes.txt`. Without this subtraction, the *second* run of this script
    would find every held-out sentence "already excluded" and drop the entire split.
    """
    heldout = out_dir / HELDOUT_FILENAME
    if not heldout.exists():
        return frozenset()
    hashes: set[str] = set()
    with heldout.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                hashes.add(sentence_hash_slp1(json.loads(line)["text_slp1"]))
    logger.info("subtracting %d hashes contributed by a previous DCS ingestion", len(hashes))
    return frozenset(hashes)


def ingest(
    *,
    conllu_dir: Path,
    out_dir: Path,
    excluded: frozenset[str],
    heldout_fraction: float,
    seed: int,
    min_words: int,
    files: Sequence[Path] | None = None,
    repo: str = DCS_REPO,
    commit: str = DCS_COMMIT,
) -> dict[str, Any]:
    """Parse every file, write both splits and `manifest.json`; return the manifest.

    Two passes, in this order, because the held-out split is itself an exclusion list.
    A file belongs to exactly one text, so the first pass reads only the held-out texts'
    files and the second only the rest; together they read each file once.

    1. **Held out.** Every sentence of a held-out `text_id`, minus the `min_words` and
       `excluded` drops. Their hashes are kept.
    2. **Train.** Everything else, minus the same two drops *and* any sentence whose SLP1
       text hashes to a held-out sentence. That last drop is not bookkeeping: DCS is full
       of formulaic lines (`sUta uvAca`, `fzaya UcuH`) that recur verbatim across texts,
       and 5,305 training sentences of the full corpus collide with a held-out one. Since
       `build_exclusion.py` puts the held-out hashes into `data/exclusion_hashes.txt`,
       leaving them in the training split would be a leak by CLAUDE.md §2.4's own
       definition and would fail Task 3's `assert_not_excluded` outright.

    `files` defaults to every `.conllu` under `conllu_dir`, sorted; it is a parameter so
    the tests can run the whole pipeline over a fixture.
    """
    paths = list(conllu_files(conllu_dir) if files is None else files)
    text_id_of = read_text_ids(paths)
    heldout_ids = assign_heldout_texts(
        text_id_of.values(), fraction=heldout_fraction, seed=seed
    )
    logger.info(
        "%d file(s); %d text(s) held out at fraction %.3f, seed %d",
        len(paths),
        len(heldout_ids),
        heldout_fraction,
        seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "n_dropped_excluded": 0,
        "n_dropped_min_words": 0,
        "n_dropped_heldout_duplicate": 0,
        "n_sentences_text_mismatch": 0,
        "n_words": 0,
        "n_words_aligned": 0,
        "n_sentences_human_verified": 0,
    }
    per_split: dict[str, dict[str, Any]] = {
        "train": {"n_sentences": 0, "texts": set()},
        "heldout": {"n_sentences": 0, "texts": set()},
    }
    started = time.monotonic()
    heldout_hashes: set[str] = set()

    def run_pass(split: str, handle: TextIO, pass_paths: Sequence[Path]) -> None:
        for index, path in enumerate(pass_paths, start=1):
            file_text_id = text_id_of[path]
            for sentence in iter_conllu_sentences(path):
                # Routing is by the file's header; the split is only trustworthy if every
                # sentence in the file really belongs to that text.
                if sentence.text_id != file_text_id:
                    raise IngestError(
                        f"{path}: sentence {sentence.sent_id!r} declares text_id "
                        f"{sentence.text_id} but the file header says {file_text_id}; it "
                        f"was about to be routed to the {split!r} split on the header's word"
                    )
                # `min_words` before alignment: 14,204 sentences are dropped here, and
                # building their gold boundaries first is work thrown away.
                if len(text_words_slp1(sentence)) < min_words:
                    counts["n_dropped_min_words"] += 1
                    continue
                gold = build_gold_sentence(sentence)
                digest = sentence_hash_slp1(gold.text_slp1)
                if digest in excluded:
                    counts["n_dropped_excluded"] += 1
                    continue
                if split == "train" and digest in heldout_hashes:
                    counts["n_dropped_heldout_duplicate"] += 1
                    continue

                record = {
                    "sent_id": sentence.sent_id,
                    "text_id": sentence.text_id,
                    "text_iast": sentence.text_iast,
                    **gold.to_dict(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if split == "heldout":
                    heldout_hashes.add(digest)

                per_split[split]["n_sentences"] += 1
                per_split[split]["texts"].add(sentence.text_id)
                counts["n_words"] += gold.n_words
                counts["n_words_aligned"] += gold.n_words_aligned
                counts["n_sentences_human_verified"] += int(gold.human_verified)
                counts["n_sentences_text_mismatch"] += int(not text_matches_token_block(sentence))

            if index % _PROGRESS_EVERY == 0 or index == len(pass_paths):
                logger.info(
                    "%s: %d/%d files, %d sentences kept, %.1f s elapsed",
                    split,
                    index,
                    len(pass_paths),
                    per_split[split]["n_sentences"],
                    time.monotonic() - started,
                )

    heldout_paths = [path for path in paths if text_id_of[path] in heldout_ids]
    train_paths = [path for path in paths if text_id_of[path] not in heldout_ids]
    with (out_dir / HELDOUT_FILENAME).open("w", encoding="utf-8") as heldout_handle:
        run_pass("heldout", heldout_handle, heldout_paths)
    with (out_dir / TRAIN_FILENAME).open("w", encoding="utf-8") as train_handle:
        run_pass("train", train_handle, train_paths)

    assert_splits_disjoint(per_split["train"]["texts"], per_split["heldout"]["texts"])

    kept = per_split["train"]["n_sentences"] + per_split["heldout"]["n_sentences"]
    manifest: dict[str, Any] = {
        "repo": repo,
        "commit": commit,
        "licence": DCS_LICENCE,
        "conllu_dir": str(conllu_dir),
        "n_files": len(paths),
        "n_texts": len(per_split["train"]["texts"] | per_split["heldout"]["texts"]),
        "heldout_fraction": heldout_fraction,
        "seed": seed,
        "min_words": min_words,
        "splits": {
            name: {
                "n_sentences": split["n_sentences"],
                "n_texts": len(split["texts"]),
                "path": str(out_dir / (TRAIN_FILENAME if name == "train" else HELDOUT_FILENAME)),
            }
            for name, split in per_split.items()
        },
        "n_sentences_kept": kept,
        **{
            key: counts[key]
            for key in (
                "n_dropped_excluded",
                "n_dropped_min_words",
                "n_dropped_heldout_duplicate",
            )
        },
        "n_sentences_text_mismatch": counts["n_sentences_text_mismatch"],
        "n_words": counts["n_words"],
        "n_words_aligned": counts["n_words_aligned"],
        "alignment_rate": (
            counts["n_words_aligned"] / counts["n_words"] if counts["n_words"] else 0.0
        ),
        "n_sentences_human_verified": counts["n_sentences_human_verified"],
        "human_verified_fraction": (
            counts["n_sentences_human_verified"] / kept if kept else 0.0
        ),
        "wall_seconds": round(time.monotonic() - started, 1),
        **provenance(),
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(sanitize_json(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", manifest_path)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "dcs.yaml",
        help="YAML config (default: dcs.yaml beside this script)",
    )
    args = parser.parse_args()

    root = repo_root()
    config = load_config(args.config)
    raw_dir = resolve_path(str(config["raw_dir"]), root)
    out_dir = resolve_path(str(config["out_dir"]), root)
    exclusion_path = resolve_path(str(config.get("exclusion_path", EXCLUSION_PATH)), root)
    commit = str(config["commit"])

    conllu_dir = download_dcs(raw_dir, commit)

    excluded = load_exclusion_hashes(exclusion_path) - _previous_dcs_hashes(out_dir)
    logger.info("%d exclusion hashes in force (from %s)", len(excluded), exclusion_path)

    manifest = ingest(
        conllu_dir=conllu_dir,
        out_dir=out_dir,
        excluded=frozenset(excluded),
        heldout_fraction=float(config["heldout_fraction"]),
        seed=int(config["seed"]),
        min_words=int(config["min_words"]),
        repo=str(config["repo"]),
        commit=commit,
    )
    logger.info(
        "done: train %d sentences / %d texts, heldout %d / %d; dropped %d excluded, "
        "%d too short, %d duplicating a held-out sentence; alignment %.4f; "
        "human-verified %.4f",
        manifest["splits"]["train"]["n_sentences"],
        manifest["splits"]["train"]["n_texts"],
        manifest["splits"]["heldout"]["n_sentences"],
        manifest["splits"]["heldout"]["n_texts"],
        manifest["n_dropped_excluded"],
        manifest["n_dropped_min_words"],
        manifest["n_dropped_heldout_duplicate"],
        manifest["alignment_rate"],
        manifest["human_verified_fraction"],
    )


if __name__ == "__main__":
    main()
