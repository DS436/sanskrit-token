"""Assemble the Experiment 05 LM training corpora and their held-out evaluation texts.

Experiment 05 (phase A), Task 1. Writes, under `data/processed/lm/` (gitignored):

    track1_raw.txt      DCS training sentences, sandhied, one per line
    track1_split.txt    the gold (oracle) split of the *same* sentences, same order
    track2_raw.txt      corpus M1: DCS + the parallel training sides + Sangraha + Wikipedia
    track2_sample.txt   (--sample) M1 cut to the Track 2 token budget
    heldout_*.txt       the evaluation texts, raw and ByT5-reconciled/oracle split
    <corpus>.manifest.json, manifest.json

Track 1 is the matched comparison: every arm's LM sees the same sentences, raw arms the
sandhied text and split arms the oracle split, so the tokenizer is the only variable.
Track 2 is the scale run on the enlarged monolingual corpus (docs/decisions.md, 2026-09-05,
"Experiment 05 runs in two tracks" and "M1 monolingual corpus").

**Line i of `track1_raw.txt` and line i of `track1_split.txt` are the same sentence.**
Both files are written in lockstep from one pass over `data/processed/dcs/train.jsonl`, and
every drop — either leakage layer, the deduplication, an empty field — removes the pair,
never one half of it. Deduplicating the two files independently would silently misalign
them, which is the failure that would make "the same data, differently tokenised" false.

**Two leakage layers, on every training line** (CLAUDE.md §2.4). First the sha256 of the
SLP1 line against `data/exclusion_hashes.txt`; then the 24-letter shingle index of every
evaluation set — FLORES devtest, Sāmayik dev/test/test_ood, Itihāsa dev/test and the DCS
held-out split — via the shared `build_evaluation_shingle_index`. A hit **drops and counts**
the line rather than aborting the build: Sangraha and Wikipedia are new sources that have
never been filtered before, and Wikipedia in particular overlaps FLORES by construction
(FLORES's Sanskrit is translated Wikipedia-style prose), so hits are the expected case and
an abort would simply mean no corpus. The per-source counts overlap, as they do in the DCS
ingestion: a line quoting two evaluation sets is counted under both.

**The two web sources pass a calibrated quality filter first** (docs/decisions.md,
2026-09-05, "Sangraha quality filter calibrated on a sample"). Sangraha is OCR of printed
books and Wikipedia quotes freely from other languages; `sanskrit_tok.data.quality` drops a
line for Latin letters, for two distinct Hindi function words, for too low a share of SLP1
letters, for too few or too many real words, for an implausible mean word length, or for a
character SLP1 cannot spell — and the manifest records the count per rule per source. The
last of those rules, `is_clean_slp1`, is applied on its own to **every** line of every
corpus and every held-out file, DCS included: a character no arm's vocabulary can spell
would otherwise sit in the bits-per-character denominator.

**Deduplication is exact-line, first occurrence kept, across sources in config order.**
The digest is a 64-bit blake2b of the line rather than the line itself, because Track 2 has
tens of millions of lines and a set of the strings would not fit in memory; at 10^8 lines
the expected number of distinct lines lost to a digest collision is about 3·10^-4, which is
recorded in the manifest (`dedup_digest_bits`) rather than described as exact.

Run:

    uv run python experiments/05_lm_training/build_corpus.py --config \\
        experiments/05_lm_training/corpus.yaml

It downloads ~4.2 GB of Sangraha parquet on a cold run and streams several billion
characters through transliteration, so run it with `nohup` and poll the log.

Then, separately and in minutes, the Track 2 training sample:

    uv run python experiments/05_lm_training/build_corpus.py --config \\
        experiments/05_lm_training/corpus.yaml --sample

`--sample` rebuilds nothing. It subsets the `track2_raw.txt` that is already on disk, using
the per-source line ranges its manifest records, so the sample inherits every filter above
unchanged (`build_sample`).
"""

import argparse
import hashlib
import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sanskrit_tok.data.exclusion import (
    EXCLUSION_PATH,
    SHINGLE_K,
    build_evaluation_shingle_index,
    letters_only,
    load_exclusion_hashes,
    sentence_hash_slp1,
    shingles,
)
from sanskrit_tok.data.quality import (
    QUALITY_RULES,
    check_quality,
    is_clean_slp1,
)
from sanskrit_tok.data.sangraha import (
    SANGRAHA_LICENCE,
    SANGRAHA_REPO,
    SANGRAHA_REVISION,
    iter_sangraha_lines,
    resolve_sanskrit_files,
    sangraha_file_sizes,
)
from sanskrit_tok.data.wikipedia_sa import (
    WIKIPEDIA_CONFIG,
    WIKIPEDIA_LICENCE,
    WIKIPEDIA_REPO,
    WIKIPEDIA_REVISION,
    iter_wikipedia_lines,
)
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.experiment import (
    SANSKRIT_LANGUAGE,
    SANSKRIT_SOURCE_LOADERS,
    load_config,
    load_corpus_entry,
    provenance,
    repo_root,
    resolve_path,
    sanitize_json,
    select_aligned_indices,
    take_indices,
)
from sanskrit_tok.tokenizers.registry import load_tokenizer

logger = logging.getLogger("build_corpus")

#: Every source key `track2_sources` may name, in the order M1 is assembled in: DCS first
#: (the gold corpus), then the parallel training sides (prose before verse, CLAUDE.md §7),
#: then the two web sources. The order is binding — it decides which copy of a duplicated
#: line is the one kept.
SOURCE_KEYS: tuple[str, ...] = (
    "dcs_train",
    "samayik_train_sa",
    "itihasa_train_sa",
    "sangraha_verified_san",
    "wikipedia_sa",
)

#: Parallel-corpus source key -> the `SANSKRIT_SOURCE_LOADERS` key it reads.
_PARALLEL_TRAIN_LOADERS: dict[str, str] = {
    "samayik_train_sa": "samayik_train",
    "itihasa_train_sa": "itihasa_train",
}

#: The sources `source_lines` yields **Devanagari** for, and the sources the calibrated
#: quality filter is applied to. The two lists coincide by design: the filter's first two
#: rules (`latin`, `hindi`) can only be asked of the source script, and these are the two
#: sources — OCR of printed books, and an encyclopedia that quotes other languages — the
#: review calibrated it on.
DEVANAGARI_SOURCES: frozenset[str] = frozenset({"sangraha_verified_san", "wikipedia_sa"})

TRACK1_RAW = "track1_raw"
TRACK1_SPLIT = "track1_split"
TRACK2_RAW = "track2_raw"
TRACK2_SAMPLE = "track2_sample"

MANIFEST_FILENAME = "manifest.json"

#: Bits of the blake2b digest deduplication compares lines by (see the module docstring).
DEDUP_DIGEST_BITS = 64

#: Lines between progress lines in the log.
_PROGRESS_EVERY = 1_000_000


class CorpusError(RuntimeError):
    """A corpus invariant the LM comparison depends on does not hold; the build stops."""


# ------------------------------------------------------------------------- primitives


@dataclass(frozen=True)
class CorpusStats:
    """What a written corpus file is: its line, character and byte counts, and its digest.

    `n_chars` and `n_bytes` exclude the line terminators, because they are what the BPC
    denominator will be: bits per SLP1 *character* of the text, not of the file framing.
    """

    n_out: int
    n_chars: int
    n_bytes: int
    sha256: str


class _Sink:
    """A corpus file being written, accumulating its own statistics as it goes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self._digest = hashlib.sha256()
        self.n_out = 0
        self.n_chars = 0
        self.n_bytes = 0

    def write(self, line: str) -> None:
        self._handle.write(line + "\n")
        encoded = line.encode("utf-8")
        self._digest.update(encoded + b"\n")
        self.n_out += 1
        self.n_chars += len(line)
        self.n_bytes += len(encoded)

    def close(self) -> CorpusStats:
        self._handle.close()
        return CorpusStats(
            n_out=self.n_out,
            n_chars=self.n_chars,
            n_bytes=self.n_bytes,
            sha256=self._digest.hexdigest(),
        )


def write_lines(path: Path, lines: Iterable[str]) -> CorpusStats:
    """Write `lines` to `path`, one per line, and return the file's statistics."""
    sink = _Sink(path)
    for line in lines:
        sink.write(line)
    return sink.close()


def normalise_line(text: str) -> str:
    """`text` with every run of whitespace collapsed to one space, and stripped.

    The corpora are newline-delimited files, so a line may not contain a line break; and a
    sentence that differs from another only in its spacing is the same sentence for
    deduplication. Collapsing here is what makes both true of every source at once.
    """
    return " ".join(text.split())


def line_digest(line: str) -> int:
    """The 64-bit blake2b digest of `line`, as an int — the deduplication key."""
    return int.from_bytes(
        hashlib.blake2b(line.encode("utf-8"), digest_size=DEDUP_DIGEST_BITS // 8).digest(),
        "big",
    )


def matching_eval_sources(
    text_slp1: str, indices: Mapping[str, frozenset[str]], k: int = SHINGLE_K
) -> list[str]:
    """The evaluation sources `text_slp1` near-duplicates, by the 24-letter shingle rule.

    Exactly `has_shingle_overlap(text_slp1, index, k)` asked of every index, but with the
    line's windows computed once and tested by set intersection instead of once per index:
    Track 2 streams billions of characters past this function, and rebuilding the window
    set seven times per line is the difference between minutes and hours. The short-text
    branch is the same deliberate floor `has_shingle_overlap` documents — a line with fewer
    than `k` letters has no window to offer and is compared by exact letter equality.
    """
    letters = letters_only(text_slp1)
    if not letters:
        return []
    if len(letters) < k:
        return [name for name, index in indices.items() if letters in index]
    windows = shingles(letters, k)
    return [name for name, index in indices.items() if not windows.isdisjoint(index)]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Every non-blank line of a jsonl file, parsed, streaming."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record: dict[str, Any] = json.loads(line)
                yield record


# ------------------------------------------------------------------------- filtering


@dataclass
class FilterCounts:
    """Why lines did not reach a corpus, per source and in total.

    `n_dropped_quality` and `n_dropped_quality_per_rule` are the calibrated web-source
    filter (docs/decisions.md, 2026-09-05, "Sangraha quality filter calibrated on a
    sample"); they are zero for the sources that are not passed through it. Its last rule
    is `non_slp1`, so a web-source line dropped for an unspellable character is counted
    there and **not** in `n_dropped_non_slp1`, which is the same test applied on its own to
    the sources that skip the quality filter. Nothing is counted twice.
    """

    n_in: dict[str, int]
    n_dropped_quality: dict[str, int]
    n_dropped_quality_per_rule: dict[str, dict[str, int]]
    n_dropped_empty: dict[str, int]
    n_dropped_non_slp1: dict[str, int]
    n_dropped_hash: dict[str, int]
    n_dropped_shingle: dict[str, int]
    n_dropped_shingle_per_source: dict[str, int]
    n_dedup_removed: dict[str, int]

    @classmethod
    def for_sources(cls, sources: Sequence[str]) -> "FilterCounts":
        return cls(
            n_in=dict.fromkeys(sources, 0),
            n_dropped_quality=dict.fromkeys(sources, 0),
            n_dropped_quality_per_rule={
                source: dict.fromkeys(QUALITY_RULES, 0) for source in sources
            },
            n_dropped_empty=dict.fromkeys(sources, 0),
            n_dropped_non_slp1=dict.fromkeys(sources, 0),
            n_dropped_hash=dict.fromkeys(sources, 0),
            n_dropped_shingle=dict.fromkeys(sources, 0),
            n_dropped_shingle_per_source={},
            n_dedup_removed=dict.fromkeys(sources, 0),
        )

    def to_manifest(self) -> dict[str, Any]:
        """The manifest fields, totals alongside the per-source breakdowns."""
        per_rule_total = {
            rule: sum(rules[rule] for rules in self.n_dropped_quality_per_rule.values())
            for rule in QUALITY_RULES
        }
        return {
            "n_in": dict(self.n_in),
            "n_in_total": sum(self.n_in.values()),
            "n_dropped_quality": sum(self.n_dropped_quality.values()),
            "n_dropped_quality_per_source": dict(self.n_dropped_quality),
            "n_dropped_quality_per_rule": per_rule_total,
            "n_dropped_quality_per_source_per_rule": {
                source: dict(rules)
                for source, rules in self.n_dropped_quality_per_rule.items()
            },
            "n_dropped_empty": sum(self.n_dropped_empty.values()),
            "n_dropped_empty_per_source": dict(self.n_dropped_empty),
            "n_dropped_non_slp1": sum(self.n_dropped_non_slp1.values()),
            "n_dropped_non_slp1_per_source": dict(self.n_dropped_non_slp1),
            "n_dropped_hash": sum(self.n_dropped_hash.values()),
            "n_dropped_hash_per_source": dict(self.n_dropped_hash),
            "n_dropped_shingle": sum(self.n_dropped_shingle.values()),
            "n_dropped_shingle_per_input_source": dict(self.n_dropped_shingle),
            "n_dropped_shingle_per_source": dict(self.n_dropped_shingle_per_source),
            "n_dedup_removed": sum(self.n_dedup_removed.values()),
            "n_dedup_removed_per_source": dict(self.n_dedup_removed),
            "dedup_digest_bits": DEDUP_DIGEST_BITS,
        }


class _Filter:
    """The two leakage layers plus exact-line deduplication, applied line by line.

    One instance owns one output corpus's `seen` set, so Track 1 and Track 2 deduplicate
    independently (the same DCS sentence belongs in both) while every source inside Track 2
    shares one set.
    """

    def __init__(
        self,
        *,
        excluded: frozenset[str],
        indices: Mapping[str, frozenset[str]],
        shingle_k: int,
        counts: FilterCounts,
    ) -> None:
        self._excluded = excluded
        self._indices = indices
        self._shingle_k = shingle_k
        self._counts = counts
        self._seen: set[int] = set()

    def accept(self, item: str, source: str) -> str | None:
        """The SLP1 line `item` contributes to the corpus, or `None` with a drop counted.

        `item` is the **Devanagari source line** for a source in `DEVANAGARI_SOURCES` and
        already-normalised SLP1 for the rest. The web sources are the ones the quality
        filter was calibrated for and the ones that arrive un-transliterated, so the two
        facts travel together: `check_quality` does the transliteration, so a line is never
        transliterated twice and a line rejected by a source-side rule is never
        transliterated at all.
        """
        self._counts.n_in[source] += 1
        if source in DEVANAGARI_SOURCES:
            result = check_quality(item)
            if not result.ok:
                assert result.rule is not None
                self._counts.n_dropped_quality[source] += 1
                self._counts.n_dropped_quality_per_rule[source][result.rule] += 1
                return None
            line = normalise_line(result.text_slp1)
        else:
            line = item
        return line if self._keep_slp1(line, source) else None

    def keep(self, line: str, source: str) -> bool:
        """Whether `line` (already normalised SLP1) goes into the corpus; counts if not.

        The `n_in`-counting entry point for the SLP1-only sources; `accept` is the general
        one. Track 1 uses this, because it filters on the raw field of a record whose split
        field it also has to write.
        """
        self._counts.n_in[source] += 1
        return self._keep_slp1(line, source)

    def _keep_slp1(self, line: str, source: str) -> bool:
        """The leakage layers and the deduplication, on an SLP1 line already counted in."""
        if not line:
            self._counts.n_dropped_empty[source] += 1
            return False
        if not is_clean_slp1(line):
            self._counts.n_dropped_non_slp1[source] += 1
            return False
        if sentence_hash_slp1(line) in self._excluded:
            self._counts.n_dropped_hash[source] += 1
            return False
        matched = matching_eval_sources(line, self._indices, self._shingle_k)
        if matched:
            self._counts.n_dropped_shingle[source] += 1
            for name in matched:
                per_source = self._counts.n_dropped_shingle_per_source
                per_source[name] = per_source.get(name, 0) + 1
            return False
        digest = line_digest(line)
        if digest in self._seen:
            self._counts.n_dedup_removed[source] += 1
            return False
        self._seen.add(digest)
        return True


# --------------------------------------------------------------------------- sources


def _dcs_train_lines(config: Mapping[str, Any], root: Path) -> Iterator[str]:
    path = resolve_path(str(config["dcs"]["train_jsonl"]), root)
    for record in iter_jsonl(path):
        yield normalise_line(str(record["text_slp1"]))


def _parallel_train_lines(key: str) -> Iterator[str]:
    loader_key = _PARALLEL_TRAIN_LOADERS[key]
    for sentence in SANSKRIT_SOURCE_LOADERS[loader_key]():
        yield normalise_line(to_slp1(sentence, "devanagari"))


#: Process-lifetime memo for `sangraha_settings`, keyed by the settings it resolves from.
#: Resolving `files: null` is a Hub call, and it is asked for twice — once to read, once to
#: write the manifest — an hour apart. Caching it means the manifest cannot disagree with
#: what was read, and a network blip an hour into the build cannot lose the build.
_SANGRAHA_SETTINGS_CACHE: dict[str, tuple[Path, dict[str, Any]]] = {}


def sangraha_settings(config: Mapping[str, Any], root: Path) -> tuple[Path, dict[str, Any]]:
    """Sangraha's cache directory and its resolved read settings, memoised per process.

    `files` is resolved here rather than inside the reader, so the manifest can record the
    file list the build **actually read** — names, not the config's `null` — and so an
    out-of-prefix name in the config is a `ValueError` before a single byte is downloaded
    (`resolve_sanskrit_files` -> `assert_verified_sanskrit_files`).
    """
    settings = dict(config.get("sangraha") or {})
    key = json.dumps({"settings": sanitize_json(settings), "root": str(root)}, sort_keys=True)
    cached = _SANGRAHA_SETTINGS_CACHE.get(key)
    if cached is not None:
        return cached
    cache_dir = resolve_path(str(settings.get("cache_dir", "data/raw/sangraha")), root)
    files = settings.get("files")
    max_files = settings.get("max_files")
    resolved = resolve_sanskrit_files(
        files=None if files is None else [str(name) for name in files],
        revision=str(settings.get("revision", SANGRAHA_REVISION)),
        max_files=None if max_files is None else int(max_files),
    )
    resolved_settings = (
        cache_dir,
        {
            "revision": str(settings.get("revision", SANGRAHA_REVISION)),
            "max_files": None if max_files is None else int(max_files),
            "files": resolved,
        },
    )
    _SANGRAHA_SETTINGS_CACHE[key] = resolved_settings
    return resolved_settings


def _sangraha_lines(config: Mapping[str, Any], root: Path) -> Iterator[str]:
    cache_dir, settings = sangraha_settings(config, root)
    yield from iter_sangraha_lines(
        cache_dir, files=settings["files"], revision=settings["revision"]
    )


def _wikipedia_lines(config: Mapping[str, Any], root: Path) -> Iterator[str]:
    settings = dict(config.get("wikipedia") or {})
    cache_dir = resolve_path(str(settings.get("cache_dir", "data/raw/wikipedia_sa")), root)
    yield from iter_wikipedia_lines(
        cache_dir, revision=str(settings.get("revision", WIKIPEDIA_REVISION))
    )


def source_lines(key: str, config: Mapping[str, Any], root: Path) -> Iterator[str]:
    """One named source's lines, streaming, in corpus order.

    SLP1 for the corpus sources, **Devanagari** for the two in `DEVANAGARI_SOURCES`: their
    quality rules are only expressible before transliteration (an SLP1 line trivially
    "contains Latin letters"), so `_Filter.accept` transliterates them, once, after the
    source-side rules have had their say.
    """
    if key == "dcs_train":
        return _dcs_train_lines(config, root)
    if key in _PARALLEL_TRAIN_LOADERS:
        return _parallel_train_lines(key)
    if key == "sangraha_verified_san":
        return _sangraha_lines(config, root)
    if key == "wikipedia_sa":
        return _wikipedia_lines(config, root)
    raise CorpusError(f"unknown corpus source {key!r}; known: {list(SOURCE_KEYS)}")


def source_provenance(key: str, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """What a source is, for the manifest: repository, revision, licence, files read."""
    if key == "dcs_train":
        return {"path": str(config["dcs"]["train_jsonl"]), "licence": "CC BY 4.0"}
    if key in _PARALLEL_TRAIN_LOADERS:
        return {"loader": _PARALLEL_TRAIN_LOADERS[key], "language": SANSKRIT_LANGUAGE}
    if key == "sangraha_verified_san":
        cache_dir, settings = sangraha_settings(config, root)
        return {
            "repo": SANGRAHA_REPO,
            "revision": settings["revision"],
            "subset": "verified/san",
            "licence": SANGRAHA_LICENCE,
            "max_files": settings["max_files"],
            "cache_dir": str(cache_dir),
            # The resolved list, with the size of each file as read: "verified only" is
            # then a fact recorded in the manifest, not a promise made by a loader.
            "files": sangraha_file_sizes(cache_dir, settings["files"]),
            "n_files": len(settings["files"]),
        }
    if key == "wikipedia_sa":
        settings = dict(config.get("wikipedia") or {})
        return {
            "repo": WIKIPEDIA_REPO,
            "revision": str(settings.get("revision", WIKIPEDIA_REVISION)),
            "config": WIKIPEDIA_CONFIG,
            "licence": WIKIPEDIA_LICENCE,
        }
    raise CorpusError(f"unknown corpus source {key!r}; known: {list(SOURCE_KEYS)}")


# ---------------------------------------------------------------------------- tracks


def build_track1(
    config: Mapping[str, Any],
    root: Path,
    out_dir: Path,
    *,
    excluded: frozenset[str],
    indices: Mapping[str, frozenset[str]],
    shingle_k: int,
) -> dict[str, tuple[CorpusStats, FilterCounts]]:
    """Write `track1_raw.txt` and `track1_split.txt` from one pass over the DCS train set.

    The pair is written in lockstep: a record survives only if both of its fields do, so
    line i of the two files is the same sentence. Filtering is on the **raw** form, since
    that is what the exclusion list and the shingle indices are built from; the split form
    of a leaked sentence is just as leaked.
    """
    train_jsonl = resolve_path(str(config["dcs"]["train_jsonl"]), root)
    raw_field = str(config["dcs"].get("raw_field", "text_slp1"))
    split_field = str(config["dcs"].get("split_field", "oracle_split_slp1"))

    counts = FilterCounts.for_sources(["dcs_train"])
    line_filter = _Filter(
        excluded=excluded, indices=indices, shingle_k=shingle_k, counts=counts
    )
    raw_sink = _Sink(out_dir / f"{TRACK1_RAW}.txt")
    split_sink = _Sink(out_dir / f"{TRACK1_SPLIT}.txt")
    for index, record in enumerate(iter_jsonl(train_jsonl), start=1):
        raw = normalise_line(str(record[raw_field]))
        split = normalise_line(str(record[split_field]))
        if not split:
            # Counted as empty on the raw side too, so the two files' `n_in` agree.
            counts.n_in["dcs_train"] += 1
            counts.n_dropped_empty["dcs_train"] += 1
            continue
        if not line_filter.keep(raw, "dcs_train"):
            continue
        raw_sink.write(raw)
        split_sink.write(split)
        if index % _PROGRESS_EVERY == 0:
            logger.info("track 1: %d records read, %d kept", index, raw_sink.n_out)
    raw_stats = raw_sink.close()
    split_stats = split_sink.close()
    if raw_stats.n_out != split_stats.n_out:
        raise CorpusError(
            f"track 1 alignment: {raw_stats.n_out} raw lines but {split_stats.n_out} split "
            "lines; the two files must be the same sentences in the same order"
        )
    logger.info("track 1: %d sentence pair(s) written", raw_stats.n_out)
    return {TRACK1_RAW: (raw_stats, counts), TRACK1_SPLIT: (split_stats, counts)}


def build_track2(
    config: Mapping[str, Any],
    root: Path,
    out_dir: Path,
    *,
    excluded: frozenset[str],
    indices: Mapping[str, frozenset[str]],
    shingle_k: int,
) -> tuple[CorpusStats, FilterCounts, dict[str, list[int]]]:
    """Write `track2_raw.txt`: every source in config order, deduplicated across all of them.

    Returns the file's statistics, the drop counts, and the half-open line range each
    source occupies in the written file. The ranges are what makes `--sample` possible
    without a second 65-minute pass: the sources are written contiguously in config order,
    so `[start, end)` per source is all a sampler needs to know which line came from where.
    """
    sources = [str(name) for name in config["track2_sources"]]
    unknown = [name for name in sources if name not in SOURCE_KEYS]
    if unknown:
        raise CorpusError(f"unknown corpus source(s) {unknown}; known: {list(SOURCE_KEYS)}")

    counts = FilterCounts.for_sources(sources)
    line_filter = _Filter(
        excluded=excluded, indices=indices, shingle_k=shingle_k, counts=counts
    )
    sink = _Sink(out_dir / f"{TRACK2_RAW}.txt")
    ranges: dict[str, list[int]] = {}
    for source in sources:
        started = time.monotonic()
        first = sink.n_out
        for index, item in enumerate(source_lines(source, config, root), start=1):
            line = line_filter.accept(item, source)
            if line is not None:
                sink.write(line)
            if index % _PROGRESS_EVERY == 0:
                logger.info(
                    "track 2 / %s: %d lines read, %d kept in total, %.0f s",
                    source,
                    index,
                    sink.n_out,
                    time.monotonic() - started,
                )
        ranges[source] = [first, sink.n_out]
        logger.info(
            "track 2 / %s: %d lines read, %d kept (lines %d-%d), %d in total, %.0f s",
            source,
            counts.n_in[source],
            sink.n_out - first,
            first,
            sink.n_out,
            sink.n_out,
            time.monotonic() - started,
        )
    stats = sink.close()
    logger.info("track 2: %d line(s) written", stats.n_out)
    return stats, counts, ranges


# --------------------------------------------------------------------- held-out text


def _heldout_pair_is_writable(raw: str, split: str, drops: dict[str, int], name: str) -> bool:
    """Whether a held-out raw/split pair goes to disk; counts the reason if not.

    The only two rules that apply to evaluation text. **Empty**: a pair with an empty half
    is not a sentence. **Non-SLP1**: a character outside the SLP1 alphabet would sit in the
    bits-per-character denominator without any arm's vocabulary being able to spell it, so
    every arm would pay for it in `[UNK]`s or bytes and the BPC comparison would be partly
    a comparison of how each vocabulary handles mojibake. Neither leakage layer applies and
    nothing is deduplicated: a sentence occurring twice in the evaluation text is two
    sentences to evaluate on, and dropping one would silently change the denominator.

    The two halves are kept in lockstep, as they are in Track 1: a pair is written or not.
    """
    if not raw or not split:
        drops[f"{name}_empty"] = drops.get(f"{name}_empty", 0) + 1
        return False
    if not is_clean_slp1(raw) or not is_clean_slp1(split):
        drops[f"{name}_non_slp1"] = drops.get(f"{name}_non_slp1", 0) + 1
        return False
    return True


def _dcs_heldout(
    config: Mapping[str, Any], root: Path, out_dir: Path, drops: dict[str, int]
) -> dict[str, CorpusStats]:
    """`heldout_dcs.txt` and `heldout_dcs_split.txt`, in held-out order."""
    path = resolve_path(str(config["dcs"]["heldout_jsonl"]), root)
    raw_field = str(config["dcs"].get("raw_field", "text_slp1"))
    split_field = str(config["dcs"].get("split_field", "oracle_split_slp1"))
    raw_sink = _Sink(out_dir / "heldout_dcs.txt")
    split_sink = _Sink(out_dir / "heldout_dcs_split.txt")
    for record in iter_jsonl(path):
        raw = normalise_line(str(record[raw_field]))
        split = normalise_line(str(record[split_field]))
        if not _heldout_pair_is_writable(raw, split, drops, "dcs"):
            continue
        raw_sink.write(raw)
        split_sink.write(split)
    return {"dcs": raw_sink.close(), "dcs_split": split_sink.close()}


def _parallel_heldout(
    entry: Mapping[str, Any], root: Path, out_dir: Path, drops: dict[str, int]
) -> dict[str, CorpusStats]:
    """One parallel evaluation corpus's raw and split held-out text, aligned by index.

    The raw side is the Sanskrit of `data/processed/split/<name>.jsonl` and the split side
    is its ByT5-reconciled `output`, so line i of the two files is the same sentence by
    construction. When the entry carries a `corpus` mapping the jsonl's `raw_deva` is
    checked against the corpus loader's aligned Sanskrit sentences — the same alignment
    Experiment 03 split and Experiment 02 measured (`select_aligned_indices`) — because a
    split file that has drifted from its corpus would produce an evaluation set whose two
    halves are different sentences, and nothing downstream could tell.
    """
    name = str(entry["name"])
    records = list(iter_jsonl(resolve_path(str(entry["split_jsonl"]), root)))
    expected = list(range(len(records)))
    if [int(record["index"]) for record in records] != expected:
        raise CorpusError(
            f"{name}: the split jsonl's indices are not 0..{len(records) - 1} in order, so "
            "its alignment with the corpus cannot be established"
        )

    corpus_entry = entry.get("corpus")
    if corpus_entry is not None:
        corpus = load_corpus_entry(dict(corpus_entry), root)
        sentences = {language: list(corpus.sentences[language]) for language in corpus.languages}
        aligned = take_indices(sentences, select_aligned_indices(sentences))[SANSKRIT_LANGUAGE]
        if len(aligned) != len(records):
            raise CorpusError(
                f"{name}: alignment mismatch — the split jsonl has {len(records)} sentence(s) "
                f"but the corpus has {len(aligned)} aligned one(s)"
            )
        for index, (record, sentence) in enumerate(zip(records, aligned, strict=True)):
            if str(record["raw_deva"]) != sentence:
                raise CorpusError(
                    f"{name}: alignment mismatch at index {index} — the split jsonl's "
                    "raw_deva is not the corpus's sentence at that index"
                )

    raw_sink = _Sink(out_dir / f"heldout_{name}.txt")
    split_sink = _Sink(out_dir / f"heldout_{name}_split.txt")
    for record in records:
        raw = normalise_line(str(record["raw_slp1"]))
        split = normalise_line(str(record["output"]))
        if not _heldout_pair_is_writable(raw, split, drops, name):
            continue
        raw_sink.write(raw)
        split_sink.write(split)
    return {name: raw_sink.close(), f"{name}_split": split_sink.close()}


# ---------------------------------------------------------------------- token counts


def count_tokens(path: Path, encode: Callable[[str], list[int]]) -> int:
    """Total tokens `encode` produces over the lines of `path`, streaming.

    This is the "≈ tokens available" figure for a track: how much a given arm's vocabulary
    turns the corpus into. It is per line, with no end-of-sequence token, because the LM
    packing adds its own (`sanskrit_tok.lm.data`).
    """
    total = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            total += len(encode(line.rstrip("\n")))
            if index % _PROGRESS_EVERY == 0:
                logger.info("%s: %d lines, %d tokens so far", path.name, index, total)
    return total


# ---------------------------------------------------------------------- track 2 sample


def _source_at(index: int, ranges: Sequence[tuple[str, int, int]]) -> str:
    """The source that produced line `index` of `track2_raw.txt`."""
    for source, start, end in ranges:
        if start <= index < end:
            return source
    raise CorpusError(f"line {index} of {TRACK2_RAW}.txt falls in no source's range")


def sampling_probability(budget_bytes: int, kept_bytes: int, sampled_bytes: int) -> float:
    """The per-line keep probability that brings the sample to `budget_bytes`.

    `kept_bytes` is what the kept-in-full sources already contribute and `sampled_bytes`
    what the sampled sources hold in total; the answer is the share of the latter the
    budget still has room for, clamped to 1.0.
    """
    if sampled_bytes <= 0:
        return 0.0
    remaining = budget_bytes - kept_bytes
    if remaining <= 0:
        raise CorpusError(
            f"the kept-in-full sources are {kept_bytes} bytes, which already exceeds the "
            f"{budget_bytes}-byte sample budget; raise track2_sample_bytes"
        )
    return min(1.0, remaining / sampled_bytes)


def build_sample(config: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Write `track2_sample.txt`: Track 2 cut to `track2_sample_bytes`.

    Track 2 is 1.02B tokens and the outline budgets ~200M (docs/decisions.md, 2026-09-05,
    "Track 2 budget"). The sample keeps **every** line of the small, clean sources — DCS,
    Sāmayik, Itihāsa, Wikipedia — and takes a uniform random subset of the sampled ones
    (Sangraha) until the byte budget is reached. Because it is a subset of the file Track 2
    already wrote, it inherits both leakage layers, the deduplication and the quality
    filter, and every arm sees exactly the same bytes.

    **Uniform means Bernoulli, not a truncated prefix.** Each sampled line is kept with
    probability `sampling_probability`, drawn from `random.Random(track2_sample_seed)` in
    file order. Stopping at the budget instead would keep the *front* of Sangraha — one
    particular set of scanned books — and call it a sample of the corpus. The achieved byte
    count therefore lands near the budget rather than on it (the relative error is
    ~1/sqrt(n) over millions of lines), and it is the achieved count, with the achieved
    token count from a streaming encode, that the manifest records.

    Two passes over `track2_raw.txt`: the first measures each source's bytes, which is what
    fixes the probability; the second writes. Both stream.
    """
    resolved_root = repo_root() if root is None else root
    out_dir = resolve_path(str(config["out_dir"]), resolved_root)
    corpus_path = out_dir / f"{TRACK2_RAW}.txt"
    track2_manifest_path = out_dir / f"{TRACK2_RAW}.manifest.json"
    if not corpus_path.exists() or not track2_manifest_path.exists():
        raise CorpusError(
            f"{corpus_path} and its manifest must exist before --sample can subset them; "
            "run the build without --sample first"
        )
    track2_manifest = json.loads(track2_manifest_path.read_text(encoding="utf-8"))
    raw_ranges = track2_manifest.get("source_line_ranges")
    if not raw_ranges:
        raise CorpusError(
            f"{track2_manifest_path} has no source_line_ranges; it predates --sample and "
            f"{TRACK2_RAW}.txt must be rebuilt before it can be sampled by source"
        )
    ranges: list[tuple[str, int, int]] = [
        (str(source), int(bounds[0]), int(bounds[1])) for source, bounds in raw_ranges.items()
    ]

    budget = int(config["track2_sample_bytes"])
    seed = int(config.get("track2_sample_seed", 0))
    sampled_sources = frozenset(
        str(name) for name in config.get("track2_sample_sampled_sources", [])
    )
    unknown = sorted(sampled_sources - {source for source, _, _ in ranges})
    if unknown:
        raise CorpusError(
            f"track2_sample_sampled_sources names {unknown}, which {TRACK2_RAW}.txt does "
            "not contain"
        )

    bytes_by_source: dict[str, int] = dict.fromkeys((source for source, _, _ in ranges), 0)
    lines_by_source: dict[str, int] = dict(bytes_by_source)
    with corpus_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            source = _source_at(index, ranges)
            bytes_by_source[source] += len(line.rstrip("\n").encode("utf-8"))
            lines_by_source[source] += 1
    kept_bytes = sum(
        count for source, count in bytes_by_source.items() if source not in sampled_sources
    )
    sampled_bytes = sum(
        count for source, count in bytes_by_source.items() if source in sampled_sources
    )
    probability = sampling_probability(budget, kept_bytes, sampled_bytes)
    logger.info(
        "sample: budget %d bytes; %d kept in full, %d available to sample -> p=%.6f",
        budget,
        kept_bytes,
        sampled_bytes,
        probability,
    )

    rng = random.Random(seed)
    sink = _Sink(out_dir / f"{TRACK2_SAMPLE}.txt")
    out_lines: dict[str, int] = dict.fromkeys(bytes_by_source, 0)
    out_bytes: dict[str, int] = dict.fromkeys(bytes_by_source, 0)
    with corpus_path.open(encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            source = _source_at(index, ranges)
            if source in sampled_sources and rng.random() >= probability:
                continue
            line = raw.rstrip("\n")
            sink.write(line)
            out_lines[source] += 1
            out_bytes[source] += len(line.encode("utf-8"))
            if sink.n_out % _PROGRESS_EVERY == 0:
                logger.info("sample: %d line(s), %d bytes", sink.n_out, sink.n_bytes)
    stats = sink.close()

    entry: dict[str, Any] = {
        "name": TRACK2_SAMPLE,
        "path": str(out_dir / f"{TRACK2_SAMPLE}.txt"),
        "sampled_from": str(corpus_path),
        "sampled_from_sha256": track2_manifest.get("sha256"),
        "track2_sample_bytes": budget,
        "seed": seed,
        "sampled_sources": sorted(sampled_sources),
        "keep_probability": probability,
        "composition": {
            source: {
                "n_in": lines_by_source[source],
                "n_in_bytes": bytes_by_source[source],
                "n_out": out_lines[source],
                "n_bytes": out_bytes[source],
                "sampled": source in sampled_sources,
            }
            for source, _, _ in ranges
        },
        **asdict(stats),
        "sources": {
            source: source_provenance(source, config, resolved_root)
            for source, _, _ in ranges
        },
    }
    arm_name = config.get("token_count_arm")
    if arm_name:
        entry["token_count_arm"] = str(arm_name)
        entry["n_tokens"] = count_tokens(
            out_dir / f"{TRACK2_SAMPLE}.txt", load_tokenizer(str(arm_name)).encode
        )
        entry["bytes_per_token"] = (
            stats.n_bytes / entry["n_tokens"] if entry["n_tokens"] else None
        )
    entry.update(provenance(resolved_root))
    (out_dir / f"{TRACK2_SAMPLE}.manifest.json").write_text(
        json.dumps(sanitize_json(entry), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "%s: %d line(s), %d bytes, %s tokens under %s",
        TRACK2_SAMPLE,
        stats.n_out,
        stats.n_bytes,
        entry.get("n_tokens", "n/a"),
        arm_name or "n/a",
    )
    return entry


# ----------------------------------------------------------------------------- build


def build(
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    shingle_indices: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Write every corpus, every held-out text and every manifest; return the manifest.

    `shingle_indices` defaults to `build_evaluation_shingle_index` over the configured
    FLORES devtest jsonl plus the DCS held-out split this build has just written; it is a
    parameter so a test can plant an index without loading four corpora, exactly as
    `experiments/04_morph_constrained/ingest_dcs.py` takes it.
    """
    resolved_root = repo_root() if root is None else root
    out_dir = resolve_path(str(config["out_dir"]), resolved_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    shingle_k = int(config.get("shingle_k", SHINGLE_K))
    exclusion_path = resolve_path(str(config.get("exclusion_path", EXCLUSION_PATH)), resolved_root)
    excluded = load_exclusion_hashes(exclusion_path)
    logger.info("%d exclusion hashes in force (from %s)", len(excluded), exclusion_path)
    started = time.monotonic()

    heldout_drops: dict[str, int] = {}
    heldout: dict[str, CorpusStats] = _dcs_heldout(
        config, resolved_root, out_dir, heldout_drops
    )
    for entry in config.get("heldout_parallel", []):
        heldout.update(_parallel_heldout(dict(entry), resolved_root, out_dir, heldout_drops))
    logger.info(
        "held-out texts: %s",
        ", ".join(f"{name}={stats.n_out}" for name, stats in heldout.items()),
    )

    if shingle_indices is None:
        raw_field = str(config["dcs"].get("raw_field", "text_slp1"))
        heldout_jsonl = resolve_path(str(config["dcs"]["heldout_jsonl"]), resolved_root)
        heldout_texts = (
            normalise_line(str(record[raw_field])) for record in iter_jsonl(heldout_jsonl)
        )
        shingle_indices = build_evaluation_shingle_index(
            resolve_path(str(config["flores_devtest_jsonl"]), resolved_root),
            shingle_k,
            extra_sources={"dcs_heldout": heldout_texts},
        )
    logger.info(
        "shingle index over %d source(s): %s",
        len(shingle_indices),
        ", ".join(f"{name}={len(index)}" for name, index in shingle_indices.items()),
    )

    corpora: dict[str, dict[str, Any]] = {}
    track1 = build_track1(
        config,
        resolved_root,
        out_dir,
        excluded=excluded,
        indices=shingle_indices,
        shingle_k=shingle_k,
    )
    track2_stats, track2_counts, track2_ranges = build_track2(
        config,
        resolved_root,
        out_dir,
        excluded=excluded,
        indices=shingle_indices,
        shingle_k=shingle_k,
    )

    arm_name = config.get("token_count_arm")
    encode: Callable[[str], list[int]] | None = None
    if arm_name:
        encode = load_tokenizer(str(arm_name)).encode

    entries: list[tuple[str, CorpusStats, FilterCounts]] = [
        (TRACK1_RAW, track1[TRACK1_RAW][0], track1[TRACK1_RAW][1]),
        (TRACK1_SPLIT, track1[TRACK1_SPLIT][0], track1[TRACK1_SPLIT][1]),
        (TRACK2_RAW, track2_stats, track2_counts),
    ]
    for name, stats, counts in entries:
        path = out_dir / f"{name}.txt"
        entry_manifest: dict[str, Any] = {
            "name": name,
            "path": str(path),
            **counts.to_manifest(),
            **asdict(stats),
            "sources": {
                source: source_provenance(source, config, resolved_root)
                for source in counts.n_in
            },
            "shingle_k": shingle_k,
            "exclusion_path": str(exclusion_path),
        }
        if name == TRACK2_RAW:
            entry_manifest["source_line_ranges"] = {
                source: list(bounds) for source, bounds in track2_ranges.items()
            }
        if name == TRACK1_SPLIT:
            entry_manifest["aligned_with"] = TRACK1_RAW
            entry_manifest["field"] = str(config["dcs"].get("split_field", "oracle_split_slp1"))
        if encode is not None:
            entry_manifest["token_count_arm"] = str(arm_name)
            entry_manifest["n_tokens"] = count_tokens(path, encode)
            logger.info("%s: %d tokens under %s", name, entry_manifest["n_tokens"], arm_name)
        corpora[name] = entry_manifest
        (out_dir / f"{name}.manifest.json").write_text(
            json.dumps(sanitize_json(entry_manifest), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    manifest: dict[str, Any] = {
        "out_dir": str(out_dir),
        "corpora": corpora,
        "heldout": {
            name: {"path": str(out_dir / f"heldout_{name}.txt"), **asdict(stats)}
            for name, stats in heldout.items()
        },
        "heldout_dropped": dict(sorted(heldout_drops.items())),
        "shingle_k": shingle_k,
        "shingle_index_sizes": {name: len(index) for name, index in shingle_indices.items()},
        "n_exclusion_hashes": len(excluded),
        "wall_seconds": round(time.monotonic() - started, 1),
        **provenance(resolved_root),
    }
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(sanitize_json(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", out_dir / MANIFEST_FILENAME)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "corpus.yaml",
        help="YAML config (default: corpus.yaml beside this script)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "do not rebuild: subset the existing track2_raw.txt to track2_sample.txt at "
            "the config's track2_sample_bytes budget"
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.sample:
        entry = build_sample(config)
        logger.info(
            "%s: %d lines, %d bytes, %s tokens; per source %s",
            TRACK2_SAMPLE,
            entry["n_out"],
            entry["n_bytes"],
            entry.get("n_tokens", "n/a"),
            ", ".join(
                f"{source}={composition['n_out']}/{composition['n_in']}"
                for source, composition in entry["composition"].items()
            ),
        )
        return 0

    manifest = build(config)
    for name, entry in manifest["corpora"].items():
        logger.info(
            "%s: %d lines, %d chars, %d bytes, %s tokens; dropped %d by hash, %d by "
            "shingle (%s), %d duplicates",
            name,
            entry["n_out"],
            entry["n_chars"],
            entry["n_bytes"],
            entry.get("n_tokens", "n/a"),
            entry["n_dropped_hash"],
            entry["n_dropped_shingle"],
            ", ".join(
                f"{source}={count}"
                for source, count in entry["n_dropped_shingle_per_source"].items()
            )
            or "none",
            entry["n_dedup_removed"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
