"""Sāmayik corpus loader: modern-register Sanskrit-English prose parallel text.

Sāmayik (CLAUDE.md §5: primary parallel corpus for TPP) is downloaded from the pinned
commit of its GitHub repository, split by split, and cached as jsonl under
`data/raw/samayik/` (gitignored) so later runs never touch the network. `train`, `dev`
and `test` come from `data/final_data/`; `test_ood` is the Mann Ki Baat out-of-domain
split, `data/mkb/mkb.sa` / `data/mkb/mkb.en` (the sibling `mkb.txt` is not an aligned
sentence pair and is ignored).

License: the repository carries no LICENSE file at the pinned commit (verified via the
GitHub API tree, 2026-09-03); recorded as unspecified in `data/README.md`, with the
dataset described in Aralikatte et al., LREC-COLING 2024.

Reproducibility follows the raw-URL pin, not a per-file sha256: `SAMAYIK_COMMIT` is
baked into every download URL, so a re-run always fetches the exact bytes this project
was built against. `flores.py` uses a checksum-verified variant of this download
mechanism for its tarball source; this loader's files have no published per-file
checksum to verify against, so the commit pin alone is the reproducibility guarantee.
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

__all__ = ["SAMAYIK_COMMIT", "SAMAYIK_DOWNLOAD_TIMEOUT_S", "SAMAYIK_REPO", "load_samayik"]

logger = logging.getLogger(__name__)

SAMAYIK_REPO = "ayushbits/Saamayik"

#: Pinned commit on `main`, observed 2026-09-03 via
#: `gh api repos/ayushbits/Saamayik/commits/main --jq .sha`. Pinning the commit in every
#: raw URL is the reproducibility mechanism (CLAUDE.md §5); there is no separate per-file
#: sha256 pin for these files.
SAMAYIK_COMMIT = "f87d54903e9f8540bda63efeb2c203e368195523"

#: Seconds to wait on each file download before giving up.
SAMAYIK_DOWNLOAD_TIMEOUT_S = 120

Split = Literal["train", "dev", "test", "test_ood"]

#: Repo-relative path (without extension) for each split's Sanskrit/English pair.
_SPLIT_PATHS: dict[str, str] = {
    "train": "data/final_data/train",
    "dev": "data/final_data/dev",
    "test": "data/final_data/test",
    "test_ood": "data/mkb/mkb",
}

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "samayik"


def _raw_url(repo_relative_path: str) -> str:
    return f"https://raw.githubusercontent.com/{SAMAYIK_REPO}/{SAMAYIK_COMMIT}/{repo_relative_path}"


def load_samayik(split: Split, cache_dir: Path | None = None) -> ParallelCorpus:
    """Load one Sāmayik split as a `ParallelCorpus` of `("san_Deva", "eng_Latn")`.

    Downloads `data/final_data/{split}.sa`/`.en` (or, for `test_ood`, `data/mkb/mkb.sa`/
    `.en`) from the pinned commit on first use, caches the aligned pairs as
    `cache_dir/{split}.jsonl`, and reuses that jsonl on every later call — no network
    access once the cache is warm. `cache_dir` defaults to `data/raw/samayik/` resolved
    against the repo root. Logs the dropped-pair count (empty lines after stripping) at
    INFO.
    """
    if split not in _SPLIT_PATHS:
        raise ValueError(
            f"unknown Sāmayik split {split!r}; expected one of {list(_SPLIT_PATHS)}"
        )
    resolved_cache_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    jsonl_path = resolved_cache_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        logger.info("reusing cached Sāmayik %s at %s", split, jsonl_path)
        return load_jsonl(jsonl_path, name="samayik", split=split)

    repo_relative = _SPLIT_PATHS[split]
    sa_path = resolved_cache_dir / f"{split}.sa"
    en_path = resolved_cache_dir / f"{split}.en"
    if not sa_path.exists():
        logger.info("downloading Sāmayik %s (Sanskrit) to %s", split, sa_path)
        download_file(_raw_url(f"{repo_relative}.sa"), sa_path, timeout=SAMAYIK_DOWNLOAD_TIMEOUT_S)
    if not en_path.exists():
        logger.info("downloading Sāmayik %s (English) to %s", split, en_path)
        download_file(_raw_url(f"{repo_relative}.en"), en_path, timeout=SAMAYIK_DOWNLOAD_TIMEOUT_S)

    corpus, dropped = read_aligned_files(
        {"san_Deva": sa_path, "eng_Latn": en_path}, name="samayik", split=split
    )
    logger.info("Sāmayik %s: %d aligned pairs, %d dropped", split, len(corpus), dropped)
    save_jsonl(corpus, jsonl_path)
    return corpus
