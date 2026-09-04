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

**Two deduplications, and why neither is redundant.** `select_training_sentences` (and the
`deduplicate_sources` it is built on) drops repeats of the **original** text;
`build_training_corpus` drops repeats of the **transformed** text. They differ because
transliteration is many-to-one: `जयमुदीरयेत्॥` (one U+0965 double danda) and
`जयमुदीरयेत्।।` (two U+0964 single dandas) are two distinct sentences and one SLP1 string.

Experiment 03 is where that stops being pedantry. Its sandhi splitter is fed Devanagari and
its cache is keyed on Devanagari, so the two spellings are two sentences to split; the
corpus is written in SLP1, so they are one line to train on. Selecting on the SLP1 form
would split only one of them and leave the other missing from the cache when the corpus was
built — a `MissingSplitError` five hours after the mistake, which is exactly what happened
in review. So the *selection* — the set that gets split, shared by
`experiments/03_sandhi_split/split_corpora.py` and `train_tokenizers.py` so the two can
never disagree — deduplicates on the original, and the corpus keeps its own deduplication
on the transformed text on top.
"""

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sanskrit_tok.data.exclusion import assert_not_excluded, sentence_hash
from sanskrit_tok.encoding import to_slp1

__all__ = [
    "build_training_corpus",
    "deduplicate_sources",
    "filter_leaked_sentences",
    "identity_transform",
    "select_training_sentences",
    "slp1_transform",
]

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


def filter_leaked_sentences(
    sources: Mapping[str, Sequence[str]],
    exclusion: frozenset[str],
    hash_fn: Callable[[str], str] = sentence_hash,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Drop sentences that collide with the evaluation exclusion list before training.

    Real corpora are not perfectly split: a handful of Sāmayik train sentences are
    formulaic course-material lines ("पाठान्ता: प्रश्ना:", "end-of-lesson questions")
    that recur verbatim in its own dev/test/test_ood splits, and a handful of Itihāsa
    train sentences are famous, oft-quoted ślokas that recur verbatim in its dev/test
    splits — not a split-alignment bug, just how these corpora were assembled upstream
    (verified by inspection: 2026-09-03 decision log entry "T1/T2 training run: filtered
    218 leaked sentences before the corpus build; no vocab shortfall").

    `build_training_corpus`'s own `assert_not_excluded` call (CLAUDE.md §2.4) is a hard
    stop for exactly this condition — a safety net, not the removal mechanism — so this
    runs first and drops the colliding sentences, logging a WARNING with the count per
    source. After this filter, that assertion is expected to pass trivially.

    Returns `(filtered_sources, dropped_counts)`: `dropped_counts` names *every* source
    passed in, including the ones with nothing dropped (value `0`), so a caller building a
    manifest never has to special-case a source that had no leakage.

    `hash_fn` must be the function `exclusion` was built with — `sentence_hash_en` for the
    English (E1) side, whose list is `data/exclusion_hashes_en.txt`.
    """
    filtered: dict[str, list[str]] = {}
    dropped_counts: dict[str, int] = {}
    for name, texts in sources.items():
        kept = [text for text in texts if hash_fn(text) not in exclusion]
        dropped = len(texts) - len(kept)
        dropped_counts[name] = dropped
        if dropped:
            logger.warning(
                "%s: dropped %d/%d sentence(s) that collide with the evaluation "
                "exclusion list (formulaic/repeated text recurring across splits, not a "
                "split-alignment bug; see docs/decisions.md)",
                name,
                dropped,
                len(texts),
            )
        filtered[name] = kept
    return filtered, dropped_counts


def deduplicate_sources(sources: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Drop blanks and exact repeats of the **stripped original text**, across sources.

    First occurrence wins, in source order then within-source order, and the per-source
    structure is preserved so a caller can still report `n_in` per source. See this
    module's docstring for why this deduplication is on the original text and why
    `build_training_corpus` deduplicates again on the transformed text.
    """
    seen: set[str] = set()
    result: dict[str, list[str]] = {}
    for name, texts in sources.items():
        kept: list[str] = []
        for text in texts:
            stripped = text.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            kept.append(text)
        result[name] = kept
    return result


def select_training_sentences(
    sources: Mapping[str, Sequence[str]],
    exclusion: frozenset[str],
    hash_fn: Callable[[str], str] = sentence_hash,
) -> list[str]:
    """The exact sentences a training corpus is built from, flat and in corpus order.

    Leakage filter, then blank and exact-repeat removal on the stripped original text.
    This is the single definition of "the training sentences", used by
    `experiments/03_sandhi_split/split_corpora.py` to decide what to hand the sandhi
    splitter and by `train_tokenizers.py` to decide what to feed the corpus builder and
    what the split cache must cover. One function rather than two implementations because
    the two must agree exactly: a sentence in one set and not the other is either model
    time spent on text nothing reads, or a `MissingSplitError` five hours later.
    """
    filtered, _ = filter_leaked_sentences(sources, exclusion, hash_fn)
    return [text for texts in deduplicate_sources(filtered).values() for text in texts]


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
