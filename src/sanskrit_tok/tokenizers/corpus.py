"""Build the SLP1 training corpus for the T1/T2 tokenizers, asserted leak-free.

CLAUDE.md §2.4: no evaluation leakage. `build_training_corpus` checks every source's
*original* Devanagari sentences against the committed exclusion list before any
conversion happens — `assert_not_excluded` hashes its own input (`sentence_hash`, which
converts to SLP1 internally), so the check is exactly the same hash the exclusion list
was built from, even though the file this function writes is already SLP1.
"""

import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from sanskrit_tok.data.exclusion import assert_not_excluded
from sanskrit_tok.encoding import to_slp1

__all__ = ["build_training_corpus"]

logger = logging.getLogger(__name__)


def build_training_corpus(
    sources: Mapping[str, Sequence[str]],
    out_path: Path,
    exclusion: frozenset[str],
) -> dict[str, object]:
    """Assemble one SLP1 training corpus from several sources of Devanagari sentences.

    For each `(name, texts)` in `sources`, in iteration order: asserts `texts` (the
    *original* Devanagari) against `exclusion` via `assert_not_excluded(texts, exclusion,
    label=name)`, which raises `LeakageError` if any sentence hashes to a committed
    evaluation sentence. Converts every text to SLP1 (`to_slp1(text, "devanagari")`),
    strips it, and drops it if empty after stripping.

    The cleaned, non-empty sentences from every source are then exact-deduplicated
    *across* sources, keeping the first occurrence (source order, then within-source
    order) and dropping every later exact repeat, and written one per line to `out_path`
    (parent directories created as needed).

    Returns a manifest: `n_in` (per-source input count, before any cleaning or
    deduplication), `n_out` (lines written), `n_dedup_removed` (cleaned non-empty
    sentences dropped as an exact duplicate of an earlier one — this does not count lines
    dropped for being empty), `exclusion_hashes` (size of `exclusion`), and `sha256` of
    the written file's bytes.
    """
    n_in: dict[str, int] = {}
    seen: set[str] = set()
    cleaned_total = 0
    unique: list[str] = []

    for name, texts in sources.items():
        n_in[name] = len(texts)
        assert_not_excluded(texts, exclusion, label=name)
        for text in texts:
            slp1 = to_slp1(text, "devanagari").strip()
            if not slp1:
                continue
            cleaned_total += 1
            if slp1 in seen:
                continue
            seen.add(slp1)
            unique.append(slp1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for line in unique:
            handle.write(line + "\n")

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "n_in": n_in,
        "n_out": len(unique),
        "n_dedup_removed": cleaned_total - len(unique),
        "exclusion_hashes": len(exclusion),
        "sha256": sha256,
    }
    logger.info(
        "training corpus: %d sentence(s) in across %d source(s), %d written, "
        "%d exact-dup removed, wrote %s",
        sum(n_in.values()),
        len(n_in),
        manifest["n_out"],
        manifest["n_dedup_removed"],
        out_path,
    )
    return manifest
