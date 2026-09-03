"""Itihāsa corpus loader: śloka-verse Sanskrit-English parallel text (Rāmāyaṇa/Mahābhārata).

Itihāsa (CLAUDE.md §5, §2.7: secondary parallel corpus — meter is a confound, prose comes
first) is downloaded from the pinned commit of its GitHub repository, split by split, and
cached as jsonl under `data/raw/itihasa/` (gitignored) so later runs never touch the
network. All three splits live at `data/{split}.sn` / `data/{split}.en`.

License: the repository carries no LICENSE file at the pinned commit (verified via the
GitHub API tree, 2026-09-03); recorded as unspecified in `data/README.md`, with the
dataset described in Aralikatte et al., WAT 2021.

Reproducibility follows the raw-URL pin, not a per-file sha256: `ITIHASA_COMMIT` is
baked into every download URL, the same mechanism `samayik.py` and `flores.py` use.
"""

import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Literal

from sanskrit_tok.data.parallel import (
    ParallelCorpus,
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


def _download(url: str, destination: Path) -> None:
    """Download `url` to `destination`, atomically: a sibling `.part` file plus `os.replace`.

    `destination` either does not exist or holds a complete download; an interrupted
    transfer can never be mistaken for a cached file on the next run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=ITIHASA_DOWNLOAD_TIMEOUT_S) as response,
            part.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)


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
        _download(_raw_url(f"{repo_relative}.sn"), sn_path)
    if not en_path.exists():
        logger.info("downloading Itihāsa %s (English) to %s", split, en_path)
        _download(_raw_url(f"{repo_relative}.en"), en_path)

    corpus, dropped = read_aligned_files(
        {"san_Deva": sn_path, "eng_Latn": en_path}, name="itihasa", split=split
    )
    logger.info("Itihāsa %s: %d aligned pairs, %d dropped", split, len(corpus), dropped)
    save_jsonl(corpus, jsonl_path)
    return corpus
