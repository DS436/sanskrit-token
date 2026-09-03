"""Itihāsa corpus loader: śloka-verse Sanskrit-English parallel text (Rāmāyaṇa/Mahābhārata).

Itihāsa (CLAUDE.md §5, §2.7: secondary parallel corpus — meter is a confound, prose comes
first) is downloaded from the pinned commit of its GitHub repository, split by split, and
cached as jsonl under `data/raw/itihasa/` (gitignored) so later runs never touch the
network. All three splits live at `data/{split}.sn` / `data/{split}.en`.

License: the repository carries no LICENSE file at the pinned commit (verified via the
GitHub API tree, 2026-09-03); recorded as unspecified in `data/README.md`, with the
dataset described in Aralikatte et al., WAT 2021.

Reproducibility follows the raw-URL pin, not a per-file sha256: `ITIHASA_COMMIT` is
baked into every download URL, the same mechanism `samayik.py` uses. `flores.py` uses a
checksum-verified variant of this download mechanism for its tarball source; this
loader's files have no published per-file checksum to verify against, so the commit pin
alone is the reproducibility guarantee.
"""

import logging
from pathlib import Path
from typing import Literal

from sanskrit_tok.data.parallel import (
    ParallelCorpus,
    download_file,
    load_jsonl,
    read_aligned_files,
    save_jsonl,
)

__all__ = ["ITIHASA_COMMIT", "ITIHASA_DOWNLOAD_TIMEOUT_S", "ITIHASA_REPO", "load_itihasa"]

logger = logging.getLogger(__name__)

ITIHASA_REPO = "rahular/itihasa"

#: Pinned commit on `main`, observed 2026-09-03. Pinning the commit in every raw URL is
#: the reproducibility mechanism (CLAUDE.md §5); there is no separate per-file sha256 pin.
ITIHASA_COMMIT = "37df077a80c83dbba1598afbdb079674b9bf6daa"

#: Seconds to wait on each file download before giving up.
ITIHASA_DOWNLOAD_TIMEOUT_S = 120

Split = Literal["train", "dev", "test"]

#: Repo-relative path (without extension) for each split's Sanskrit/English pair.
_SPLIT_PATHS: dict[str, str] = {
    "train": "data/train",
    "dev": "data/dev",
    "test": "data/test",
}

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "itihasa"


def _raw_url(repo_relative_path: str) -> str:
    return f"https://raw.githubusercontent.com/{ITIHASA_REPO}/{ITIHASA_COMMIT}/{repo_relative_path}"


def load_itihasa(split: Split, cache_dir: Path | None = None) -> ParallelCorpus:
    """Load one Itihāsa split as a `ParallelCorpus` of `("san_Deva", "eng_Latn")`.

    Downloads `data/{split}.sn`/`.en` from the pinned commit on first use, caches the
    aligned pairs as `cache_dir/{split}.jsonl`, and reuses that jsonl on every later call
    — no network access once the cache is warm. `cache_dir` defaults to
    `data/raw/itihasa/` resolved against the repo root. Logs the dropped-pair count
    (empty lines after stripping) at INFO.
    """
    if split not in _SPLIT_PATHS:
        raise ValueError(
            f"unknown Itihāsa split {split!r}; expected one of {list(_SPLIT_PATHS)}"
        )
    resolved_cache_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    jsonl_path = resolved_cache_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        logger.info("reusing cached Itihāsa %s at %s", split, jsonl_path)
        return load_jsonl(jsonl_path, name="itihasa", split=split)

    repo_relative = _SPLIT_PATHS[split]
    sn_path = resolved_cache_dir / f"{split}.sn"
    en_path = resolved_cache_dir / f"{split}.en"
    if not sn_path.exists():
        logger.info("downloading Itihāsa %s (Sanskrit) to %s", split, sn_path)
        download_file(_raw_url(f"{repo_relative}.sn"), sn_path, timeout=ITIHASA_DOWNLOAD_TIMEOUT_S)
    if not en_path.exists():
        logger.info("downloading Itihāsa %s (English) to %s", split, en_path)
        download_file(_raw_url(f"{repo_relative}.en"), en_path, timeout=ITIHASA_DOWNLOAD_TIMEOUT_S)

    corpus, dropped = read_aligned_files(
        {"san_Deva": sn_path, "eng_Latn": en_path}, name="itihasa", split=split
    )
    logger.info("Itihāsa %s: %d aligned pairs, %d dropped", split, len(corpus), dropped)
    save_jsonl(corpus, jsonl_path)
    return corpus
