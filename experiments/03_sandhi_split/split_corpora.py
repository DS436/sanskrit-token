"""Sandhi-split every Experiment 03 corpus once, into one shared cache (exp03 Task 3).

This is the expensive step of the experiment: `chronbmm/sanskrit5-multitask` at the
measured 7.09 sentences/s on MPS over ~137k sentences, i.e. about five and a half hours
(docs/decisions.md, "Splitter throughput measured"). It is therefore written to be
launched in the background and left alone:

    mkdir -p outputs/03_sandhi_split
    nohup uv run python experiments/03_sandhi_split/split_corpora.py \\
        > outputs/03_sandhi_split/split.log 2>&1 &

**Resumability.** Every model result is appended to `SplitCache` and flushed per batch, so
a kill at hour four loses at most the in-flight batch; re-running the script re-reads the
cache, skips everything already split, and picks up where it stopped. Nothing else in the
project ever calls the model: the tokenizer-corpus builder (`train_tokenizers.py`, side
`sa_split`) reads this same cache and fails loudly if a sentence is missing from it.

**What gets split.** The four Experiment 02 evaluation corpora's Sanskrit side, filtered
to the same aligned, non-blank indices `run.py` evaluates, and then the tokenizer training
corpus's Sanskrit side — *after* the exclusion filter and the exact dedup, so leaked
sentences are never handed to the model and the sentence set is exactly the one behind
`data/processed/tok_train_slp1.txt`.

**Two outputs per sentence.** The cache stores what the model said; the per-corpus
`data/processed/split/<corpus>.jsonl` stores the raw Devanagari, its SLP1 form, the raw
model output (`output_model`) and the reconciled text (`output`, `sandhi.reconcile`).
Reconciliation is cheap, deterministic and pure, so the jsonl is recomputed from the cache
on every run and the threshold can be changed without re-splitting anything.

**What the manifest is for.** `data/processed/split/manifest.json` records the model and
its revision, the device, the counts per corpus, and — the reason the reconciliation
exists — character retention before and after it, per corpus, pooled.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sanskrit_tok.data.exclusion import load_exclusion_hashes
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.experiment import (
    SANSKRIT_LANGUAGE,
    SANSKRIT_SOURCE_LOADERS,
    TextInvariants,
    collect_sources,
    load_config,
    load_corpus_entry,
    provenance,
    repo_root,
    resolve_path,
    sanitize_json,
    select_aligned_indices,
    take_indices,
    text_invariants,
)
from sanskrit_tok.sandhi import SandhiSplitter, SplitCache
from sanskrit_tok.sandhi.reconcile import DEFAULT_THRESHOLD, reconcile
from sanskrit_tok.tokenizers.corpus import select_training_sentences

logger = logging.getLogger("split_corpora")

#: The name the training corpus is reported under in the manifest and its jsonl file.
TRAIN_CORPUS_NAME = "train"

class SupportsSplit(Protocol):
    """What `split_corpus` needs from a splitter: a batch size, `split`, and `stats`.

    A `Protocol` rather than `SandhiSplitter` itself so the offline tests can drive the
    whole file-writing and accounting path with a fake that returns canned strings and
    never imports torch.
    """

    batch_size: int

    @property
    def stats(self) -> dict[str, int | float]: ...

    def split(self, texts: Sequence[str]) -> list[str]: ...


# ------------------------------------------------------------------------------ config


@dataclass(frozen=True)
class SplitConfig:
    """`split.yaml`, parsed and with every path resolved against the repository root."""

    model_id: str
    device: str
    batch_size: int
    cache_path: Path
    manifest_path: Path
    split_dir: Path
    exclusion_path: Path
    eval_corpora: list[dict[str, Any]]
    train_sources: list[str]
    subset: tuple[int, int] | None
    reconcile_threshold: float
    progress_every: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], root: Path) -> "SplitConfig":
        """Build a config from a parsed YAML mapping, validating what it can up front.

        An unknown `train_sources` entry raises here, before the model is loaded and hours
        of work are spent, rather than at the moment the loader is looked up.
        """
        train_sources = [str(name) for name in mapping["train_sources"]]
        unknown = [name for name in train_sources if name not in SANSKRIT_SOURCE_LOADERS]
        if unknown:
            raise KeyError(
                f"unknown train_sources entry/entries {unknown}; known: "
                f"{list(SANSKRIT_SOURCE_LOADERS)}"
            )
        subset_raw = mapping.get("subset")
        subset = (
            None if subset_raw is None else (int(subset_raw["n"]), int(subset_raw.get("seed", 0)))
        )
        return cls(
            model_id=str(mapping["model_id"]),
            device=str(mapping.get("device", "auto")),
            batch_size=int(mapping.get("batch_size", 16)),
            cache_path=resolve_path(str(mapping["cache_path"]), root),
            manifest_path=resolve_path(str(mapping["manifest_path"]), root),
            split_dir=resolve_path(str(mapping["split_dir"]), root),
            exclusion_path=resolve_path(str(mapping["exclusion_path"]), root),
            eval_corpora=[dict(entry) for entry in mapping["eval_corpora"]],
            train_sources=train_sources,
            subset=subset,
            reconcile_threshold=float(mapping.get("reconcile_threshold", DEFAULT_THRESHOLD)),
            progress_every=int(mapping.get("progress_every", 500)),
        )


# ------------------------------------------------------------------- corpus preparation


def eval_sentences(entry: Mapping[str, Any], root: Path) -> list[str]:
    """The Sanskrit sentences of one evaluation corpus, in evaluation order.

    Filtered to the indices that are non-blank in *every* language, exactly as
    `experiments/02_tpp_parallel/run.py` does before it measures anything: splitting a
    sentence Experiment 02 will never evaluate would be hours of model time spent on text
    no number is computed from.
    """
    corpus = load_corpus_entry(entry, root)
    sentences = {language: list(corpus.sentences[language]) for language in corpus.languages}
    indices = select_aligned_indices(sentences)
    if len(indices) < len(corpus):
        logger.warning(
            "%s: dropping %d/%d indices blank in at least one language",
            entry.get("name"),
            len(corpus) - len(indices),
            len(corpus),
        )
    return take_indices(sentences, indices)[SANSKRIT_LANGUAGE]


def select_subset(texts: Sequence[str], n: int, seed: int) -> list[str]:
    """A deterministic uniform subset of `n` sentences, kept in corpus order.

    Only ever used if the throughput rule's subset branch fires (it did not:
    docs/decisions.md, "Splitter throughput measured"). Deterministic for a seed so the
    same subset can be selected again by the matched raw `_sub` arms — the whole point of
    the rule is that the split and unsplit arms see the same sentences. Corpus order is
    preserved so the subset reads like a sample of the corpus rather than a shuffle of it.
    """
    if n >= len(texts):
        return list(texts)
    indices = sorted(random.Random(seed).sample(range(len(texts)), n))
    return [texts[index] for index in indices]


# ------------------------------------------------------------------------- the split run


@dataclass(frozen=True)
class CorpusSplitReport:
    """What one corpus's split run produced, for the manifest.

    `char_retention_model` is `chars_model / chars_raw` and `char_retention_reconciled` is
    `chars_out / chars_raw`, both **pooled** over the corpus (summed numerators over
    summed denominators, not a mean of per-sentence ratios — docs/decisions.md,
    "CORRECTION: splitter character retention is 87.3%") and both over non-space
    characters. The pair is the measurement that justifies reconciliation: the first
    should sit well below 1, the second at or just above it.

    `invariants` is `experiment.text_invariants` over the raw and reconciled text of this
    corpus. Its `nonletter_multiset_preserved` is the pass/fail check the final review
    added: retention of 0.9895 looked healthy while punctuation fused to words was being
    deleted wholesale, and only an exact multiset comparison catches that
    (docs/decisions.md, "Reconciliation must preserve every non-letter character").

    `n_units_kept_verbatim` and `n_units_replaced_inexact` are the pooled honesty counters
    retention cannot supply: retention reads 1.000 whether the model re-segmented a
    sentence faithfully or dropped one word and rewrote another by the same number of
    characters. The first counts what reconciliation had to put back, the second the
    replacements whose similarity was below 1.0 — where the model changed characters
    rather than only inserting boundaries.
    """

    name: str
    path: str
    n_sentences: int
    n_out: int
    n_units_raw: int
    n_units_out: int
    n_units_kept_verbatim: int
    n_units_replaced_inexact: int
    n_units_changed: int
    fraction_units_changed: float
    chars_raw: int
    chars_model: int
    chars_out: int
    char_retention_model: float
    char_retention_reconciled: float
    invariants: TextInvariants
    seconds: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def split_corpus(
    name: str,
    texts: Sequence[str],
    splitter: SupportsSplit,
    out_path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    progress_every: int = 500,
) -> CorpusSplitReport:
    """Split `texts` in batches, reconcile each result, and write `out_path`.

    One JSON object per line, `{"index", "raw_deva", "raw_slp1", "output_model",
    "output"}`, in corpus order. The file is written to a sibling `.tmp` and moved into
    place only once the corpus is complete, so a run killed mid-corpus can never leave
    behind a short file that looks finished; the cache, not this file, is what makes the
    re-run cheap.

    Progress is logged every `progress_every` sentences with the running rate and an ETA,
    because the only view onto a five-hour background job is its log.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    started = time.perf_counter()
    raw_slp1_texts: list[str] = []
    model_texts: list[str] = []
    out_texts: list[str] = []
    n_units_changed = 0
    n_units_raw = n_units_out = n_units_verbatim = n_units_inexact = 0
    chars_raw = chars_model = chars_out = 0
    written = 0

    with tmp_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(texts), splitter.batch_size):
            batch = list(texts[start : start + splitter.batch_size])
            outputs = splitter.split(batch)
            for offset, (raw_deva, output_model) in enumerate(zip(batch, outputs, strict=True)):
                raw_slp1 = to_slp1(raw_deva, "devanagari")
                result = reconcile(raw_slp1, output_model, threshold=threshold)
                handle.write(
                    json.dumps(
                        {
                            "index": start + offset,
                            "raw_deva": raw_deva,
                            "raw_slp1": raw_slp1,
                            "output_model": output_model,
                            "output": result.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                raw_slp1_texts.append(raw_slp1)
                model_texts.append(output_model)
                out_texts.append(result.text)
                if result.n_units_out != result.n_units_raw:
                    n_units_changed += 1
                n_units_raw += result.n_units_raw
                n_units_out += result.n_units_out
                n_units_verbatim += result.n_units_kept_verbatim
                n_units_inexact += result.n_units_replaced_inexact
                chars_raw += result.chars_raw
                chars_model += result.chars_model
                chars_out += result.chars_out

            done = min(start + splitter.batch_size, len(texts))
            if progress_every and done % progress_every < splitter.batch_size:
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (len(texts) - done) / rate if rate else float("nan")
                logger.info(
                    "%s: %d/%d sentences, %.2f sentences/s, ETA %.1f min",
                    name,
                    done,
                    len(texts),
                    rate,
                    remaining / 60,
                )

    os.replace(tmp_path, out_path)
    invariants = text_invariants(raw_slp1_texts, out_texts, model_texts)
    if not invariants["letters_out_subset_of_raw_union_model"]:
        logger.warning(
            "%s: the reconciled text contains letters present in NEITHER the raw sentence "
            "nor the model output; reconciliation is inventing text",
            name,
        )
    if not invariants["nonletter_multiset_preserved"]:
        logger.warning(
            "%s: reconciliation did not preserve the non-letter character multiset "
            "(%d raw vs %d out); missing %s, added %s",
            name,
            invariants["nonletter_chars_raw"],
            invariants["nonletter_chars_out"],
            dict(list(invariants["nonletter_missing"].items())[:6]),
            dict(list(invariants["nonletter_added"].items())[:6]),
        )
    seconds = time.perf_counter() - started
    logger.info(
        "%s: wrote %d record(s) to %s in %.1f s (retention %.4f model, %.4f reconciled; "
        "%d unit(s) kept verbatim, %d replaced inexactly, of %d)",
        name,
        written,
        out_path,
        seconds,
        _ratio(chars_model, chars_raw),
        _ratio(chars_out, chars_raw),
        n_units_verbatim,
        n_units_inexact,
        n_units_raw,
    )
    return CorpusSplitReport(
        name=name,
        path=str(out_path),
        n_sentences=len(texts),
        n_out=written,
        n_units_raw=n_units_raw,
        n_units_out=n_units_out,
        n_units_kept_verbatim=n_units_verbatim,
        n_units_replaced_inexact=n_units_inexact,
        n_units_changed=n_units_changed,
        fraction_units_changed=_ratio(n_units_changed, len(texts)),
        chars_raw=chars_raw,
        chars_model=chars_model,
        chars_out=chars_out,
        char_retention_model=_ratio(chars_model, chars_raw),
        char_retention_reconciled=_ratio(chars_out, chars_raw),
        invariants=invariants,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------- manifest


def build_manifest(
    *,
    config: SplitConfig,
    splitter_source_id: str,
    device: str,
    stats: Mapping[str, int | float],
    reports: Sequence[CorpusSplitReport],
    seconds: float,
    root: Path,
) -> dict[str, Any]:
    """The split run's manifest: what model split what, on what, and how lossily.

    `splitter_source_id` is `"<model id>@<revision>"`; `revision` is split back out of it
    so a reader does not have to parse the composite. `n_sentences`/`n_out` are per corpus
    (the plan's wording); the run-wide splitter counters — cache hits, model calls,
    chunked inputs — come straight from `SandhiSplitter.stats`.
    """
    model_id, _, revision = splitter_source_id.partition("@")
    manifest: dict[str, Any] = {
        "model_id": model_id or config.model_id,
        "revision": revision or "unknown",
        "splitter_source_id": splitter_source_id,
        "device": device,
        "batch_size": config.batch_size,
        "subset": None
        if config.subset is None
        else {"n": config.subset[0], "seed": config.subset[1]},
        "reconcile_threshold": config.reconcile_threshold,
        "cache_path": str(config.cache_path),
        "n_sentences": {report.name: report.n_sentences for report in reports},
        "n_out": {report.name: report.n_out for report in reports},
        "n_cache_hits": int(stats.get("n_cache_hits", 0)),
        "n_model": int(stats.get("n_model", 0)),
        "n_chunked": int(stats.get("n_chunked", 0)),
        "seconds": seconds,
        "corpora": {
            report.name: {
                "path": report.path,
                "n_sentences": report.n_sentences,
                "n_out": report.n_out,
                "n_units_raw": report.n_units_raw,
                "n_units_out": report.n_units_out,
                "n_units_kept_verbatim": report.n_units_kept_verbatim,
                "n_units_replaced_inexact": report.n_units_replaced_inexact,
                "n_units_changed": report.n_units_changed,
                "fraction_units_changed": report.fraction_units_changed,
                "chars_raw": report.chars_raw,
                "chars_model": report.chars_model,
                "chars_out": report.chars_out,
                "char_retention_model": report.char_retention_model,
                "char_retention_reconciled": report.char_retention_reconciled,
                "invariants": report.invariants,
                "seconds": report.seconds,
            }
            for report in reports
        },
        "splitter_stats": dict(stats),
        "char_retention_definition": (
            "non-space SLP1 characters in the output, summed over sentences, divided by "
            "non-space SLP1 characters in the raw SLP1 input, summed over the same "
            "sentences (pooled, not a mean of per-sentence ratios)"
        ),
    }
    manifest.update(provenance(root))
    return manifest


# -------------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("split.yaml"),
        help="split config YAML (default: split.yaml beside this script)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="split at most this many sentences per corpus (smoke runs only; the manifest "
        "records the truncated counts, so a limited run's manifest is not the real one)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    root = repo_root()
    config = SplitConfig.from_mapping(load_config(args.config), root)

    cache = SplitCache(config.cache_path)
    splitter = SandhiSplitter(
        model_id=config.model_id,
        device=None if config.device == "auto" else config.device,
        cache=cache,
        batch_size=config.batch_size,
    )
    logger.info(
        "splitter %s, device=%s, batch_size=%d, cache %s (%d entry/entries)",
        config.model_id,
        config.device,
        config.batch_size,
        config.cache_path,
        len(cache),
    )

    started = time.perf_counter()
    reports: list[CorpusSplitReport] = []

    for entry in config.eval_corpora:
        name = str(entry["name"])
        texts = eval_sentences(entry, root)
        if args.limit is not None:
            texts = texts[: args.limit]
        logger.info("%s: %d Sanskrit sentence(s) to split", name, len(texts))
        reports.append(
            split_corpus(
                name,
                texts,
                splitter,
                config.split_dir / f"{name}.jsonl",
                threshold=config.reconcile_threshold,
                progress_every=config.progress_every,
            )
        )

    exclusion = load_exclusion_hashes(config.exclusion_path)
    logger.info("loaded %d exclusion hashes from %s", len(exclusion), config.exclusion_path)
    sources = collect_sources(config.train_sources, SANSKRIT_SOURCE_LOADERS)
    for name, texts in sources.items():
        logger.info("source %s: %d sentence(s) loaded", name, len(texts))
    train_texts = select_training_sentences(sources, exclusion)
    logger.info(
        "training corpus: %d distinct, non-leaked sentence(s) after the exp02 pipeline",
        len(train_texts),
    )
    if config.subset is not None:
        train_texts = select_subset(train_texts, *config.subset)
        logger.warning(
            "subset rule applied: splitting %d sentence(s) with seed %d",
            len(train_texts),
            config.subset[1],
        )
    if args.limit is not None:
        train_texts = train_texts[: args.limit]
    reports.append(
        split_corpus(
            TRAIN_CORPUS_NAME,
            train_texts,
            splitter,
            config.split_dir / f"{TRAIN_CORPUS_NAME}.jsonl",
            threshold=config.reconcile_threshold,
            progress_every=config.progress_every,
        )
    )

    cache.close()
    seconds = time.perf_counter() - started
    manifest = build_manifest(
        config=config,
        splitter_source_id=splitter.source_id,
        device=str(splitter.device),
        stats=splitter.stats,
        reports=reports,
        seconds=seconds,
        root=root,
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(sanitize_json(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    config.manifest_path.write_text(payload, encoding="utf-8")
    logger.info("wrote %s", config.manifest_path)
    # A second, timestamped copy. `manifest.json` is what everything downstream reads and is
    # therefore last-write-wins, and the cache makes re-runs cheap enough that it *will* be
    # overwritten — which is how the 9.4-hour first run's throughput, chunking and device
    # facts were lost to a 72-second cache-hit re-run that recorded `n_model: 1`. The
    # per-run copies are the record of what each run actually did.
    stamped = config.manifest_path.with_name(
        f"{config.manifest_path.stem}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    stamped.write_text(payload, encoding="utf-8")
    logger.info("wrote %s", stamped)
    logger.info(
        "done in %.1f min: %d sentence(s) split, %d from cache, %d chunked",
        seconds / 60,
        int(splitter.stats.get("n_model", 0)),
        int(splitter.stats.get("n_cache_hits", 0)),
        int(splitter.stats.get("n_chunked", 0)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
