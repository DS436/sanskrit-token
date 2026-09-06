"""Build `data/exclusion_hashes.txt` and `data/exclusion_hashes_en.txt` (CLAUDE.md §2.4).

Hashes both sides of every evaluation split named in `docs/decisions.md` ("Experiment 02
corpora and pivots"): FLORES devtest, Sāmayik dev/test/test_ood, and Itihāsa dev/test.
Sāmayik train and Itihāsa train are deliberately excluded — they are training text, not
evaluation text, and hashing them would make every tokenizer trained on them fail its own
leakage check.

Two lists, one per side, because the two tokenizer families train on different text:

* `data/exclusion_hashes.txt` — the Sanskrit side, hashed with `sentence_hash` (sha256 of
  the SLP1 form), guarding the `T1_*`/`T2_*` arms.
* `data/exclusion_hashes_en.txt` — the English side of the *same* sentences, hashed with
  `sentence_hash_en` (sha256 of the stripped text, no transliteration), guarding the
  matched English control arms `E1_*` (docs/decisions.md, "Add a matched English control
  family E1 for TPP").

**The DCS held-out split.** Experiment 04 trains on DCS and evaluates MorphScore on a
held-out 5% of its *texts*, so those sentences are evaluation text and belong in the
Sanskrit list (docs/decisions.md, 2026-09-05, "DCS is the gold source and the first
monolingual training corpus"). DCS is annotated in IAST and stored in SLP1, so its
sentences are hashed with `sentence_hash_slp1` rather than `sentence_hash` — the same
digest of the same SLP1 bytes, without a lossy Devanagari round-trip — via
`build_exclusion_list`'s per-source `hash_fns`. There is no English side, so `dcs_heldout`
appears in the Sanskrit list only.

They are read from `data/processed/dcs/heldout.jsonl`, which
`experiments/04_morph_constrained/ingest_dcs.py` writes and which is gitignored.

**The Sangraha held-out split.** Experiment 05's Track 2 trains on a corpus that is 93.6%
Sangraha, so it holds 2,000 Sangraha lines out of its own training sample as an in-domain
evaluation set (`experiments/05_lm_training/build_corpus.py --sangraha-heldout`;
docs/decisions.md, 2026-09-06, "Track 2 gets an in-domain held-out set"). Those lines are
evaluation text from the moment they are drawn, so they belong in the Sanskrit list on the
same footing as the DCS split: already SLP1, hashed with `sentence_hash_slp1`, monolingual
and therefore absent from the English list. They are read from
`data/processed/lm/heldout_sangraha.txt`, which is gitignored.

**Two guards, because a leakage list must never weaken silently.**

* If `heldout.jsonl` is absent (a fresh clone that has not run the ingestion) this script
  **fails**. Regenerating the list without DCS would drop 30k hashes from the committed
  file, and the committed file is the authority every training script asserts against.
  `--allow-missing-dcs` writes the list without that source, for someone who genuinely
  wants a DCS-free list and has said so. `heldout_sangraha.txt` is guarded the same way,
  by `--allow-missing-sangraha`.
* Whatever the sources, the list about to be written must be a **superset** of the one
  already on disk. Any hash the committed file holds and the new one does not means an
  evaluation sentence has stopped being excluded — a corpus loader that silently returned
  a short split, a renamed source, a half-run ingestion. `--allow-shrink` is the escape
  hatch, and it exists so that removing a source is a deliberate, recorded act.

Config-free and idempotent: it reuses the corpus loaders directly (downloading any split
that is not already cached under `data/raw/`) and overwrites both lists with the freshly
computed hashes. Run with:

    uv run python experiments/02_tpp_parallel/build_exclusion.py
"""

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sanskrit_tok.data.exclusion import (
    EXCLUSION_PATH,
    EXCLUSION_PATH_EN,
    build_exclusion_list,
    hash_sources,
    load_exclusion_hashes,
    sentence_hash,
    sentence_hash_en,
    sentence_hash_slp1,
)
from sanskrit_tok.data.flores import ParallelCorpus, load_jsonl
from sanskrit_tok.data.itihasa import load_itihasa
from sanskrit_tok.data.samayik import load_samayik
from sanskrit_tok.experiment import repo_root

logger = logging.getLogger("build_exclusion")

#: This repository's root, from which the data paths below are resolved.
REPO_ROOT = repo_root()

#: Order matches the `exclusion.py` header contract: `flores_devtest`, `samayik_dev`,
#: `samayik_test`, `samayik_test_ood`, `itihasa_dev`, `itihasa_test`. Both lists use it,
#: so the two files' `# sources:` headers name the same splits in the same order.
SOURCE_ORDER = (
    "flores_devtest",
    "samayik_dev",
    "samayik_test",
    "samayik_test_ood",
    "itihasa_dev",
    "itihasa_test",
)

#: The Sanskrit list carries two more sources, appended after `itihasa_test`: the DCS
#: held-out split and the Sangraha one. Both monolingual, so neither has an English
#: counterpart.
DCS_HELDOUT_SOURCE = "dcs_heldout"
SANGRAHA_HELDOUT_SOURCE = "sangraha_heldout"

#: Written by `experiments/04_morph_constrained/ingest_dcs.py`; gitignored.
DCS_HELDOUT_PATH = REPO_ROOT / "data" / "processed" / "dcs" / "heldout.jsonl"

#: Written by `experiments/05_lm_training/build_corpus.py --sangraha-heldout`; gitignored.
#: One SLP1 sentence per line, already normalised — it was drawn out of a built corpus.
SANGRAHA_HELDOUT_PATH = REPO_ROOT / "data" / "processed" / "lm" / "heldout_sangraha.txt"

SANSKRIT_LANGUAGE = "san_Deva"
ENGLISH_LANGUAGE = "eng_Latn"


class MissingDcsHeldoutError(RuntimeError):
    """`heldout.jsonl` is absent, so the list would be written without its DCS hashes."""


class MissingSangrahaHeldoutError(RuntimeError):
    """`heldout_sangraha.txt` is absent, so the list would lose its Sangraha hashes."""


class ExclusionShrinkError(RuntimeError):
    """The list about to be written drops hashes the committed one already holds."""


def load_dcs_heldout(
    path: Path = DCS_HELDOUT_PATH, *, allow_missing: bool = False
) -> list[str]:
    """The sandhied SLP1 text of every DCS held-out sentence.

    Raises `MissingDcsHeldoutError` when `path` does not exist. Regenerating the Sanskrit
    list without it silently removes 30,150 evaluation hashes, and a leakage guard that
    quietly gets weaker is worse than one that refuses to run — so the fresh-clone case is
    a hard stop with instructions, not a warning.

    `allow_missing=True` (the `--allow-missing-dcs` flag) returns `[]` with a loud warning
    instead: the source is then left out of the list entirely rather than contributing an
    empty set of hashes, which is the same thing but silent.
    """
    if not path.exists():
        message = (
            f"{path} not found, so the Sanskrit exclusion list would be written WITHOUT the "
            "DCS held-out hashes. Run experiments/04_morph_constrained/ingest_dcs.py first, "
            "or keep the committed data/exclusion_hashes.txt, or pass --allow-missing-dcs "
            "if you really want a DCS-free list."
        )
        if not allow_missing:
            raise MissingDcsHeldoutError(message)
        logger.warning("--allow-missing-dcs: %s", message)
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text_slp1"] for line in handle if line.strip()]


def load_sangraha_heldout(
    path: Path = SANGRAHA_HELDOUT_PATH, *, allow_missing: bool = False
) -> list[str]:
    """The SLP1 text of every held-out Sangraha line, one per line of `path`.

    Raises `MissingSangrahaHeldoutError` when `path` does not exist, for the same reason
    `load_dcs_heldout` does: regenerating the list without it silently drops 2,000
    evaluation hashes, and the committed list is the authority every training script
    asserts against. `--allow-missing-sangraha` returns `[]` with a warning instead.
    """
    if not path.exists():
        message = (
            f"{path} not found, so the Sanskrit exclusion list would be written WITHOUT the "
            "Sangraha held-out hashes. Run experiments/05_lm_training/build_corpus.py "
            "--sangraha-heldout first, or keep the committed data/exclusion_hashes.txt, or "
            "pass --allow-missing-sangraha if you really want a Sangraha-free list."
        )
        if not allow_missing:
            raise MissingSangrahaHeldoutError(message)
        logger.warning("--allow-missing-sangraha: %s", message)
        return []
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def assert_superset_of_committed(
    new_hashes: set[str], path: Path, *, allow_shrink: bool = False
) -> None:
    """Raise `ExclusionShrinkError` if `path` holds hashes `new_hashes` does not.

    A missing `path` (nothing committed yet) passes. Every dropped hash is an evaluation
    sentence that would stop being excluded, so the default is to refuse; `allow_shrink`
    (the `--allow-shrink` flag) downgrades it to a warning for a deliberate removal.
    """
    if not path.exists():
        return
    lost = load_exclusion_hashes(path) - new_hashes
    if not lost:
        return
    message = (
        f"{path}: the list about to be written drops {len(lost)} hash(es) the committed "
        f"file holds, so {len(lost)} evaluation sentence(s) would stop being excluded "
        f"(e.g. {sorted(lost)[:3]}). Pass --allow-shrink if this removal is deliberate."
    )
    if not allow_shrink:
        raise ExclusionShrinkError(message)
    logger.warning("--allow-shrink: %s", message)


def load_corpora() -> dict[str, ParallelCorpus]:
    """Every evaluation split, keyed by source name — loaded once, read from twice."""
    flores_path = REPO_ROOT / "data" / "raw" / "flores" / "devtest.jsonl"
    return {
        "flores_devtest": load_jsonl(flores_path, name="flores200", split="devtest"),
        "samayik_dev": load_samayik("dev"),
        "samayik_test": load_samayik("test"),
        "samayik_test_ood": load_samayik("test_ood"),
        "itihasa_dev": load_itihasa("dev"),
        "itihasa_test": load_itihasa("test"),
    }


def collect_sources(
    corpora: dict[str, ParallelCorpus], language: str
) -> dict[str, list[str]]:
    """One language's sentences from every evaluation split, in `SOURCE_ORDER`."""
    return {name: list(corpora[name].sentences[language]) for name in SOURCE_ORDER}


def write_list(
    sources: Mapping[str, Sequence[str]],
    path: Path,
    *,
    hash_fn: Callable[[str], str],
    hash_fns: Mapping[str, Callable[[str], str]] | None = None,
    allow_shrink: bool = False,
) -> int:
    """Check the superset guard, then write the list; return the number of hashes written.

    The hashes are computed once by `hash_sources` and compared with the committed file
    *before* `build_exclusion_list` overwrites it, so a refusal leaves the committed list
    exactly as it was.
    """
    new_hashes = hash_sources(sources, hash_fn=hash_fn, hash_fns=hash_fns)
    assert_superset_of_committed(new_hashes, path, allow_shrink=allow_shrink)
    return build_exclusion_list(sources, path, hash_fn=hash_fn, hash_fns=hash_fns)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-dcs",
        action="store_true",
        help="write the Sanskrit list without the DCS held-out hashes when "
        "data/processed/dcs/heldout.jsonl is absent (default: fail)",
    )
    parser.add_argument(
        "--allow-missing-sangraha",
        action="store_true",
        help="write the Sanskrit list without the Sangraha held-out hashes when "
        "data/processed/lm/heldout_sangraha.txt is absent (default: fail)",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="write a list even if it drops hashes the committed one holds (default: fail)",
    )
    args = parser.parse_args()

    corpora = load_corpora()
    dcs_heldout = load_dcs_heldout(allow_missing=args.allow_missing_dcs)
    sangraha_heldout = load_sangraha_heldout(allow_missing=args.allow_missing_sangraha)

    for language, path, hash_fn in (
        (SANSKRIT_LANGUAGE, REPO_ROOT / EXCLUSION_PATH, sentence_hash),
        (ENGLISH_LANGUAGE, REPO_ROOT / EXCLUSION_PATH_EN, sentence_hash_en),
    ):
        sources = collect_sources(corpora, language)
        hash_fns: dict[str, Callable[[str], str]] = {}
        if language == SANSKRIT_LANGUAGE and dcs_heldout:
            sources[DCS_HELDOUT_SOURCE] = dcs_heldout
            hash_fns[DCS_HELDOUT_SOURCE] = sentence_hash_slp1
        if language == SANSKRIT_LANGUAGE and sangraha_heldout:
            sources[SANGRAHA_HELDOUT_SOURCE] = sangraha_heldout
            hash_fns[SANGRAHA_HELDOUT_SOURCE] = sentence_hash_slp1
        count = write_list(
            sources, path, hash_fn=hash_fn, hash_fns=hash_fns, allow_shrink=args.allow_shrink
        )
        size = path.stat().st_size
        logger.info(
            "%s: wrote %d unique exclusion hashes to %s (%d bytes)", language, count, path, size
        )


if __name__ == "__main__":
    main()
