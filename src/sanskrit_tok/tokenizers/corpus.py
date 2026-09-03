"""Build the tokenizer training corpus for the T1/T2 (and E1) arms, asserted leak-free.

CLAUDE.md §2.4: no evaluation leakage. `build_training_corpus` checks every source's
*original* sentences against the committed exclusion list before any conversion happens —
`assert_not_excluded` hashes its own input with the same `hash_fn` the exclusion list was
built from, so the check matches the committed hashes even though the file this function
writes may already be transliterated.

Two knobs make one builder serve both sides of the parallel corpora:

* `transform` — what each sentence is converted to before being written. The default,
  `slp1_transform`, is the Sanskrit path (CLAUDE.md §2.3: internal encoding is SLP1);
  `identity_transform` is the English path for the `E1_*` control arms, whose text must
  reach the trainer exactly as written.
* `hash_fn` — how a sentence is hashed for the leakage assertion: `sentence_hash`
  (SLP1) for Sanskrit, `sentence_hash_en` (stripped text) for English. They agree on
  pure ASCII but not on a sentence carrying any Devanagari, so an English corpus checked
  with the Sanskrit hash can pass while leaking; the caller must pass the function its
  exclusion list was built with.
"""

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sanskrit_tok.data.exclusion import assert_not_excluded, sentence_hash
from sanskrit_tok.encoding import to_slp1

__all__ = ["build_training_corpus", "identity_transform", "slp1_transform"]

logger = logging.getLogger(__name__)


def slp1_transform(text: str) -> str:
    """Devanagari -> SLP1, the default corpus transform (CLAUDE.md §2.3)."""
    return to_slp1(text, "devanagari")


def identity_transform(text: str) -> str:
    """`text` unchanged — the English (`E1_*`) corpus transform.

    A named function rather than a lambda so a config or a caller can refer to it, and so
    a corpus built without transliteration says so in a traceback.
    """
    return text


def build_training_corpus(
    sources: Mapping[str, Sequence[str]],
    out_path: Path,
    exclusion: frozenset[str],
    *,
    transform: Callable[[str], str] | None = None,
    hash_fn: Callable[[str], str] = sentence_hash,
) -> dict[str, object]:
    """Assemble one training corpus from several sources of sentences.

    For each `(name, texts)` in `sources`, in iteration order: asserts `texts` (the
    *original*, untransformed text) against `exclusion` via `assert_not_excluded(texts,
    exclusion, label=name, hash_fn=hash_fn)`, which raises `LeakageError` if any sentence
    hashes to a committed evaluation sentence. Applies `transform` to every text
    (defaulting to `slp1_transform`, i.e. Devanagari -> SLP1), strips it, and drops it if
    empty after stripping.

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
    convert = slp1_transform if transform is None else transform
    n_in: dict[str, int] = {}
    seen: set[str] = set()
    cleaned_total = 0
    unique: list[str] = []

    for name, texts in sources.items():
        n_in[name] = len(texts)
        assert_not_excluded(texts, exclusion, label=name, hash_fn=hash_fn)
        for text in texts:
            cleaned = convert(text).strip()
            if not cleaned:
                continue
            cleaned_total += 1
            if cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)

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
