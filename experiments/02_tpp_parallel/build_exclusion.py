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

Config-free and idempotent: it reuses the corpus loaders directly (downloading any split
that is not already cached under `data/raw/`) and overwrites both lists with the freshly
computed hashes. Run with:

    uv run python experiments/02_tpp_parallel/build_exclusion.py
"""

import logging

from sanskrit_tok.data.exclusion import (
    EXCLUSION_PATH,
    EXCLUSION_PATH_EN,
    build_exclusion_list,
    sentence_hash,
    sentence_hash_en,
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

SANSKRIT_LANGUAGE = "san_Deva"
ENGLISH_LANGUAGE = "eng_Latn"


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    corpora = load_corpora()

    for language, path, hash_fn in (
        (SANSKRIT_LANGUAGE, REPO_ROOT / EXCLUSION_PATH, sentence_hash),
        (ENGLISH_LANGUAGE, REPO_ROOT / EXCLUSION_PATH_EN, sentence_hash_en),
    ):
        sources = collect_sources(corpora, language)
        count = build_exclusion_list(sources, path, hash_fn=hash_fn)
        size = path.stat().st_size
        logger.info(
            "%s: wrote %d unique exclusion hashes to %s (%d bytes)", language, count, path, size
        )


if __name__ == "__main__":
    main()
