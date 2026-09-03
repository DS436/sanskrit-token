"""Build `data/exclusion_hashes.txt` (CLAUDE.md §2.4: no evaluation leakage).

Hashes the Sanskrit side of every evaluation split named in `docs/decisions.md`
("Experiment 02 corpora and pivots"): FLORES devtest, Sāmayik dev/test/test_ood, and
Itihāsa dev/test. Sāmayik train and Itihāsa train are deliberately excluded — they are
training text, not evaluation text, and hashing them would make every tokenizer trained
on them fail its own leakage check.

Config-free and idempotent: it reuses the corpus loaders directly (downloading any split
that is not already cached under `data/raw/`) and overwrites `data/exclusion_hashes.txt`
with the freshly computed list. Run with:

    uv run python experiments/02_tpp_parallel/build_exclusion.py
"""

import logging
from pathlib import Path

from sanskrit_tok.data.exclusion import EXCLUSION_PATH, build_exclusion_list
from sanskrit_tok.data.flores import load_jsonl
from sanskrit_tok.data.itihasa import load_itihasa
from sanskrit_tok.data.samayik import load_samayik

logger = logging.getLogger("build_exclusion")

#: `experiments/02_tpp_parallel/build_exclusion.py` -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Order matches the `exclusion.py` header contract: `flores_devtest`, `samayik_dev`,
#: `samayik_test`, `samayik_test_ood`, `itihasa_dev`, `itihasa_test`.
SOURCE_ORDER = (
    "flores_devtest",
    "samayik_dev",
    "samayik_test",
    "samayik_test_ood",
    "itihasa_dev",
    "itihasa_test",
)


def collect_sources() -> dict[str, list[str]]:
    """Load every evaluation split and return its Sanskrit sentences, keyed by source name."""
    flores_path = REPO_ROOT / "data" / "raw" / "flores" / "devtest.jsonl"
    flores = load_jsonl(flores_path, name="flores200", split="devtest")

    return {
        "flores_devtest": flores.sentences["san_Deva"],
        "samayik_dev": load_samayik("dev").sentences["san_Deva"],
        "samayik_test": load_samayik("test").sentences["san_Deva"],
        "samayik_test_ood": load_samayik("test_ood").sentences["san_Deva"],
        "itihasa_dev": load_itihasa("dev").sentences["san_Deva"],
        "itihasa_test": load_itihasa("test").sentences["san_Deva"],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sources = collect_sources()
    # Reorder to SOURCE_ORDER explicitly rather than relying on dict literal order above.
    ordered_sources = {name: sources[name] for name in SOURCE_ORDER}
    path = REPO_ROOT / EXCLUSION_PATH
    count = build_exclusion_list(ordered_sources, path)
    size = path.stat().st_size
    logger.info("wrote %d unique exclusion hashes to %s (%d bytes)", count, path, size)


if __name__ == "__main__":
    main()
