"""Evaluation-leakage exclusion list (CLAUDE.md §2.4: no evaluation leakage).

SIGHUM, Hackathon and DCS-2018 are all subsets of DCS, and the tokenizer/parallel-text
evaluation splits used across this project overlap in ways that are easy to lose track of
by hand. Instead, every evaluation sentence's SLP1 form is hashed once and recorded in
`data/exclusion_hashes.txt`; any script that assembles training text for a tokenizer or an
LM asserts the training text against this list before training starts, via
`assert_not_excluded`.

Hashing is over the SLP1 form (`to_slp1(text.strip(), "devanagari")`), not the original
script, so a sentence that appears in both Devanagari and a romanised form still collides
to the same hash. The list is committed to git (`data/exclusion_hashes.txt`); it holds
only sha256 digests of public evaluation text, never the text itself.

**The English side.** The matched English control arms (`E1_*`, CLAUDE.md §6) are trained
on the *English* side of the same parallel corpora, so they need their own exclusion list
over the *English* evaluation sentences: `data/exclusion_hashes_en.txt`, built with
`sentence_hash_en` (sha256 of the stripped text, no transliteration — `to_slp1` is for
Devanagari and would garble English). Every function here that hashes takes a `hash_fn`
so one code path serves both lists, and the caller must pass the one its list was built
with: the two agree on pure ASCII (transliteration leaves Latin alone) but diverge on any
sentence carrying Devanagari, so checking an English list with the Sanskrit hash would
pass a leak through unnoticed.
"""

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from sanskrit_tok.encoding import to_slp1

__all__ = [
    "EXCLUSION_PATH",
    "EXCLUSION_PATH_EN",
    "LeakageError",
    "assert_not_excluded",
    "build_exclusion_list",
    "load_exclusion_hashes",
    "sentence_hash",
    "sentence_hash_en",
]

logger = logging.getLogger(__name__)

#: Resolved against the repo root at call time (not import time), so callers running from
#: any working directory still find the committed file.
EXCLUSION_PATH = Path("data/exclusion_hashes.txt")

#: The English counterpart, for the `E1_*` control arms (`sentence_hash_en`).
EXCLUSION_PATH_EN = Path("data/exclusion_hashes_en.txt")

_HEADER_HASH_LINE = "# sha256 of the SLP1 form of every evaluation sentence, one per line"
_HEADER_HASH_LINE_EN = (
    "# sha256 of the stripped text of every English evaluation sentence, one per line"
)
_HEADER_HASH_LINE_OTHER = "# sha256 of every evaluation sentence, one per line"

#: How many offending indices `assert_not_excluded` names before it stops listing them.
_MAX_NAMED_OFFENDERS = 5


def sentence_hash(text: str) -> str:
    """sha256 of the SLP1 form of `text.strip()`, as hex.

    `text` is assumed to be Devanagari (the script every evaluation corpus in this project
    stores its Sanskrit in); stripping first means leading/trailing whitespace differences
    between sources never produce different hashes for the same sentence.
    """
    return hashlib.sha256(to_slp1(text.strip(), "devanagari").encode("utf-8")).hexdigest()


def sentence_hash_en(text: str) -> str:
    """sha256 of `text.strip()` as written, as hex — the English (`E1_*`) counterpart.

    No transliteration: `to_slp1` maps a Devanagari phoneme inventory and has nothing to
    say about Latin script, so the English side is hashed verbatim. Stripping first, for
    the same reason as `sentence_hash`: whitespace differences between two copies of the
    same sentence must not produce two hashes.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


#: Which header line describes which hash function, so a written list says on its first
#: line what its digests are *of*. An unrecognised `hash_fn` gets the generic line rather
#: than a wrong one.
_HEADER_FOR_HASH_FN: dict[Callable[[str], str], str] = {
    sentence_hash: _HEADER_HASH_LINE,
    sentence_hash_en: _HEADER_HASH_LINE_EN,
}


def build_exclusion_list(
    sources: Mapping[str, Sequence[str]],
    path: Path,
    *,
    hash_fn: Callable[[str], str] = sentence_hash,
) -> int:
    """Hash every sentence in `sources` and write the sorted, deduplicated list to `path`.

    `sources` maps a source name (e.g. `"samayik_test"`) to its sentences. Two header
    comment lines are written first: the hash-format line for `hash_fn` (SLP1 for
    `sentence_hash`, stripped English text for `sentence_hash_en`), then
    `# sources: <name>=<count>, ...` in the order `sources` was given, recording how many
    sentences each source contributed (before deduplication, so the counts are honest about
    what was hashed even when sources overlap). Returns the number of unique hashes written.
    """
    hashes: set[str] = set()
    for sentences in sources.values():
        for sentence in sentences:
            hashes.add(hash_fn(sentence))

    source_summary = ", ".join(f"{name}={len(sentences)}" for name, sentences in sources.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(_HEADER_FOR_HASH_FN.get(hash_fn, _HEADER_HASH_LINE_OTHER) + "\n")
        handle.write(f"# sources: {source_summary}\n")
        for digest in sorted(hashes):
            handle.write(digest + "\n")

    logger.info("wrote %d unique exclusion hashes to %s", len(hashes), path)
    return len(hashes)


def load_exclusion_hashes(path: Path) -> frozenset[str]:
    """Read a file written by `build_exclusion_list`, skipping `#` comment lines."""
    hashes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            hashes.add(stripped)
    return frozenset(hashes)


class LeakageError(RuntimeError):
    """Raised when text destined for training overlaps a committed evaluation sentence."""


def assert_not_excluded(
    texts: Sequence[str],
    hashes: frozenset[str],
    *,
    label: str,
    hash_fn: Callable[[str], str] = sentence_hash,
) -> None:
    """Raise `LeakageError` if any of `texts` hashes to an entry in `hashes`.

    `label` identifies what was being checked (e.g. `"T1_bpe_raw_32k training text"`) so
    the error is actionable without re-deriving what call site failed. Names up to
    `_MAX_NAMED_OFFENDERS` offending indices; passes silently for a clean list.

    `hash_fn` must be the same function `hashes` was built with — `sentence_hash_en` for
    an English list. Mismatching them does not error: the Sanskrit hash transliterates
    first, so on any sentence carrying Devanagari it simply never matches, and the leak
    this check exists to catch passes through.
    """
    offending = [index for index, text in enumerate(texts) if hash_fn(text) in hashes]
    if not offending:
        return
    named = offending[:_MAX_NAMED_OFFENDERS]
    remaining = len(offending) - _MAX_NAMED_OFFENDERS
    suffix = "" if remaining <= 0 else f" (+{remaining} more)"
    raise LeakageError(
        f"{label}: {len(offending)} sentence(s) match the evaluation exclusion list; "
        f"offending indices: {named}{suffix}"
    )
