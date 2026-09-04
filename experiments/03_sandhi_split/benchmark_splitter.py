"""Measure `SandhiSplitter` throughput, to decide how much corpus can be split.

The splitter is the cost centre of Experiment 03: a 2.3 GB byte-level seq2seq model on a
machine with no CUDA GPU. Which corpus it can be run over is therefore an empirical
question, and `docs/decisions.md` ("Splitter throughput rule") fixes the rule in advance so
the answer is not chosen after seeing the results: benchmark N sentences on the best
available device; if the projected time for the 117,720-sentence training corpus is at
most `FULL_CORPUS_BUDGET_H`, split all of it, otherwise split a seed-0 subset sized to
`SUBSET_BUDGET_H` and train the matched raw arms on the same subset. The 19,197-sentence
evaluation set is always split in full; it is projected here so its cost is known too.

This script measures, prints and records — it does not act on the rule. The choice it
implies is written into `docs/decisions.md` by hand and applied in Task 3's `split.yaml`.

The cache is a throwaway in a temporary directory: a benchmark must measure the model, not
a cache hit, and must never write into `data/processed/split/`, whose contents the real
split run is responsible for.

    uv run python experiments/03_sandhi_split/benchmark_splitter.py --n 200 --device mps
"""

import argparse
import json
import logging
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sanskrit_tok.data.samayik import load_samayik
from sanskrit_tok.encoding import from_slp1, to_slp1
from sanskrit_tok.experiment import provenance, repo_root, sanitize_json
from sanskrit_tok.sandhi import SandhiSplitter, SplitCache
from sanskrit_tok.sandhi.byt5 import SEGMENTATION_PREFIX, chunk_text, pick_device

logger = logging.getLogger("benchmark_splitter")

#: Sentences in the tokenizer training corpus (Sāmayik + Itihāsa training splits, post
#: exclusion filter) and in the four evaluation corpora, from Experiment 02's results.
TRAIN_CORPUS_SENTENCES = 117_720
EVAL_CORPUS_SENTENCES = 19_197

#: The two budgets of the throughput rule, in hours.
FULL_CORPUS_BUDGET_H = 8.0
SUBSET_BUDGET_H = 6.0

DEFAULT_OUTPUT_DIR = "outputs/03_sandhi_split"
N_EXAMPLES = 10

#: Written beside the number so `benchmark.json` states what was divided by what.
RETENTION_DEFINITION = (
    "non-space SLP1 characters in the split output, summed over sentences, divided by "
    "non-space SLP1 characters in the raw SLP1 input, summed over the same sentences "
    "(pooled, not a mean of per-sentence ratios)"
)


class RecordingSplitter(SandhiSplitter):
    """A splitter that keeps the model's raw decoded IAST for every chunk it generated.

    The examples table shows what the model actually emitted. Re-deriving that column from
    the stored SLP1 would show what the *pipeline* emitted after normalisation, which is a
    different string — the separators are already spaces by then — so the benchmark hooks
    the seam instead. Keyed by the prefixed chunk, which is what the seam receives.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        cache: SplitCache | None = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_id=model_id, device=device, cache=cache, batch_size=batch_size)
        self.raw_decodes: dict[str, str] = {}

    def _generate_iast(self, batch: list[str]) -> list[str]:
        outputs = super()._generate_iast(batch)
        self.raw_decodes.update(zip(batch, outputs, strict=True))
        return outputs

    def raw_iast(self, text: str) -> str:
        """The raw decode(s) for `text`, rebuilt from the same chunks `split` fed the model."""
        iast = from_slp1(to_slp1(text, "devanagari"), "iast")
        pieces = [SEGMENTATION_PREFIX + piece for piece in chunk_text(iast)]
        return " ".join(self.raw_decodes.get(piece, "") for piece in pieces)


def nonspace_length(text: str) -> int:
    return len("".join(text.split()))


def load_sentences(n: int) -> list[str]:
    """The first `n` non-blank Sanskrit (Devanagari) sentences of the Sāmayik test split."""
    corpus = load_samayik("test")
    sentences = [text for text in corpus.sentences["san_Deva"] if text.strip()]
    if len(sentences) < n:
        logger.warning("only %d sentences available, requested %d", len(sentences), n)
    return sentences[:n]


def whitespace_units(text: str) -> int:
    return len(text.split())


def projected_hours(sentences: int, seconds_per_sentence: float) -> float:
    return sentences * seconds_per_sentence / 3600.0


def implied_choice(train_hours: float) -> dict[str, Any]:
    """Apply the throughput rule to a projection: full corpus, or a subset sized to fit.

    Returned as data rather than acted on, so `benchmark.json` records the rule's own
    arithmetic (including the subset size it implies) next to the measurement it came from.
    """
    if train_hours <= FULL_CORPUS_BUDGET_H:
        return {
            "decision": "full",
            "train_hours": train_hours,
            "budget_hours": FULL_CORPUS_BUDGET_H,
            "subset_n": None,
        }
    per_sentence_hours = train_hours / TRAIN_CORPUS_SENTENCES
    return {
        "decision": "subset",
        "train_hours": train_hours,
        "budget_hours": FULL_CORPUS_BUDGET_H,
        "subset_n": int(SUBSET_BUDGET_H / per_sentence_hours),
        "subset_budget_hours": SUBSET_BUDGET_H,
        "subset_seed": 0,
    }


def build_examples(
    splitter: RecordingSplitter,
    inputs: Sequence[str],
    outputs: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Before/after rows: Devanagari in, the model's raw IAST, the stored SLP1, unit counts.

    `iast_model_output` is the decode captured at the seam — separators and all — not a
    back-conversion of the SLP1 column, so the two columns together show what the
    normalisation step actually did.
    """
    return [
        {
            "devanagari": source,
            "iast_model_output": splitter.raw_iast(source),
            "slp1_split": output,
            "words_raw": whitespace_units(source),
            "words_split": whitespace_units(output),
        }
        for source, output in list(zip(inputs, outputs, strict=True))[:limit]
    ]


def run(n: int, device: str, batch_size: int, out_dir: Path) -> dict[str, Any]:
    sentences = load_sentences(n)
    resolved_device = pick_device() if device == "auto" else device
    logger.info(
        "benchmarking %d sentences on %s (batch_size=%d)",
        len(sentences),
        resolved_device,
        batch_size,
    )

    with tempfile.TemporaryDirectory(prefix="sandhi-benchmark-") as tmp:
        cache = SplitCache(Path(tmp) / "cache.jsonl")
        splitter = RecordingSplitter(device=resolved_device, cache=cache, batch_size=batch_size)
        started = time.perf_counter()
        outputs = splitter.split(sentences)
        wall_seconds = time.perf_counter() - started
        # After the split, not before: a candidate substitution during loading changes
        # which model `source_id` must name.
        source_id = splitter.source_id
        stats = dict(splitter.stats)
        effective_device = splitter.device
        device_fallback = splitter.device_fallback
        examples = build_examples(splitter, sentences, outputs, N_EXAMPLES)
        cache.close()

    per_sentence = wall_seconds / len(sentences)
    sentences_per_second = len(sentences) / wall_seconds
    changed = [
        whitespace_units(out) > whitespace_units(src)
        for src, out in zip(sentences, outputs, strict=True)
    ]
    train_hours = projected_hours(TRAIN_CORPUS_SENTENCES, per_sentence)
    raw_slp1 = [to_slp1(sentence, "devanagari") for sentence in sentences]
    retention = sum(nonspace_length(text) for text in outputs) / sum(
        nonspace_length(text) for text in raw_slp1
    )

    results: dict[str, Any] = {
        **provenance(),
        # Top level, not only inside `config`: `benchmark.json` is a copy of one device's
        # run, and a reader must not have to guess which.
        "device": resolved_device,
        "config": {
            "n": len(sentences),
            "device_requested": device,
            "device_resolved": resolved_device,
            "device_effective": effective_device,
            "device_fallback_to_cpu": device_fallback,
            "batch_size": batch_size,
            "corpus": "samayik/test (san_Deva)",
        },
        "splitter_source_id": source_id,
        "splitter_stats": stats,
        "throughput": {
            "wall_seconds": wall_seconds,
            "seconds_per_sentence": per_sentence,
            "sentences_per_second": sentences_per_second,
            "model_seconds": stats["seconds_model"],
        },
        "projection_hours": {
            "train_corpus": train_hours,
            "train_corpus_sentences": TRAIN_CORPUS_SENTENCES,
            "eval_corpora": projected_hours(EVAL_CORPUS_SENTENCES, per_sentence),
            "eval_corpora_sentences": EVAL_CORPUS_SENTENCES,
        },
        "throughput_rule": implied_choice(train_hours),
        "segmentation": {
            "fraction_words_increased": sum(changed) / len(changed),
            "mean_words_raw": statistics.fmean(whitespace_units(s) for s in sentences),
            "mean_words_split": statistics.fmean(whitespace_units(o) for o in outputs),
            "n_chunked": stats["n_chunked"],
            # The model does not merely re-segment: it normalises, drops punctuation and
            # sometimes drops a transliterated loanword. This is how much of the input
            # survives it, and Experiment 03 must report it per corpus.
            "char_retention_nonspace": retention,
            "char_retention_nonspace_definition": RETENTION_DEFINITION,
        },
        "examples": examples,
        # Every output, not just the ten shown: two runs on two devices are compared by
        # diffing this list, which is the only way "mps agrees with cpu" is checkable
        # after the fact rather than asserted.
        "split_outputs": list(outputs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(results, out_dir / f"benchmark_{resolved_device}.json")

    # `benchmark.json` is the file the plan names, and the file the decisions entry is read
    # against, so it must hold the run the throughput rule was actually applied to — the
    # best device available, which is the device the corpus split will use. A `--device
    # cpu` comparison run writes its own per-device file and leaves this one alone; last
    # write must not win.
    applied_device = pick_device()
    if resolved_device == applied_device:
        _write(results, out_dir / "benchmark.json")
    else:
        logger.info(
            "benchmark.json left unchanged: this run is on %s, but the throughput rule "
            "applies to %s (the best available device)",
            resolved_device,
            applied_device,
        )
    return results


def _write(results: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(results), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    logger.info("wrote %s", path)


def report(results: dict[str, Any]) -> None:
    config = results["config"]
    throughput = results["throughput"]
    projection = results["projection_hours"]
    rule = results["throughput_rule"]
    segmentation = results["segmentation"]

    print(f"\nsplitter        {results['splitter_source_id']}")
    print(f"device          {config['device_effective']} (requested {config['device_requested']})")
    print(f"sentences       {config['n']}  batch_size {config['batch_size']}")
    print(
        f"throughput      {throughput['sentences_per_second']:.3f} sentences/s "
        f"({throughput['seconds_per_sentence']:.3f} s/sentence, "
        f"{throughput['wall_seconds']:.1f} s wall)"
    )
    print(
        f"projection      train {projection['train_corpus']:.2f} h "
        f"({projection['train_corpus_sentences']:,} sentences), "
        f"eval {projection['eval_corpora']:.2f} h "
        f"({projection['eval_corpora_sentences']:,} sentences)"
    )
    print(
        f"rule            {rule['decision']} "
        f"(budget {rule['budget_hours']} h; subset_n {rule['subset_n']})"
    )
    print(
        f"segmentation    {segmentation['fraction_words_increased']:.1%} of sentences gained "
        f"whitespace units; mean {segmentation['mean_words_raw']:.2f} -> "
        f"{segmentation['mean_words_split']:.2f}; n_chunked {int(segmentation['n_chunked'])}"
    )
    print(
        f"retention       {segmentation['char_retention_nonspace']:.4f} "
        f"(non-space SLP1 characters kept, pooled)"
    )
    print("\nexamples")
    for index, example in enumerate(results["examples"], start=1):
        print(f"\n[{index}] deva  {example['devanagari']}")
        print(f"    iast  {example['iast_model_output']}")
        print(f"    slp1  {example['slp1_split']}")
        print(f"    words {example['words_raw']} -> {example['words_split']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=200, help="sentences to split (default 200)")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="torch device; 'auto' picks mps, then cuda, then cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="generation batch size")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / DEFAULT_OUTPUT_DIR,
        help=f"where to write benchmark.json (default {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    report(run(args.n, args.device, args.batch_size, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
