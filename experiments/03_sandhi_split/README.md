# Experiment 03 — Sandhi-split arms (T4): does reversing sandhi first recover token efficiency?

**Hypothesis (H3, TPP/fertility half).** Sanskrit's whitespace words are long because
sandhi welds them together; a subword vocabulary learned on that welded text spends its
merges on surface junctions rather than on morphemes. Reversing sandhi *before* subword
learning should therefore lower **tokens-per-proposition** and fertility for the `T4`
arms relative to the matched raw-text `T1`/`T2` arms at the same algorithm and the same
vocabulary size, measured against the matched English control `E1`.

**Success.** On Sāmayik test (prose first, CLAUDE.md §7), the split arm's controlled TPP
is below its matched raw arm's and the paired-bootstrap CI on the difference excludes 0.
Fertility is reported both ways and never leads (CLAUDE.md §2.1; docs/decisions.md,
"Fertility for split arms uses the raw word count as the primary denominator").

**What this experiment cannot test.** MorphScore and the gold-segmentation upper bound
(ablation A3) need DCS morpheme boundaries, which Experiment 04 ingests; both are deferred
there (docs/decisions.md, "Experiment 03 tests the TPP/fertility half of H3"). The `T4`
arms remain **provisional** for the same reason `T1`/`T2` are: they are trained on the
Sanskrit sides of two parallel corpora, not on the monolingual corpus of milestone M1.

## Status

| Task | State |
|---|---|
| 1. Shared experiment helpers (`sanskrit_tok.experiment`) | done |
| 2. `SandhiSplitter` + cache + throughput benchmark | done |
| 3. Split the corpora, train the `T4` arms | not started |
| 4. Runner, `results.json`, figure, results table | not started |

Numbers land in Task 4; this file carries only the splitter facts so far.

## The splitter

`sanskrit_tok.sandhi.SandhiSplitter` wraps `chronbmm/sanskrit5-multitask` in segmentation
mode (`"S "` prefix, IAST in and out, 512-byte window). Devanagari goes in, SLP1 with one
whitespace unit per segment comes out, and every result is cached in an append-only jsonl
`SplitCache` so no sentence is ever split twice.

Three empirical facts from the Task 2 benchmark, all recorded in `docs/decisions.md`:

* The raw decode separates segments with `_`, not with a space
  (`viśvāsa_kāraṇāt_eva_…`). The wrapper normalises it; a bare `batch_decode` would leave
  a whole sentence as a single whitespace unit.
* The model **splits compounds as well as sandhi** (`द्वितीयमुद्रायां` →
  `dvitīya mudrāyām`). The `T4` arms therefore measure "sandhi *and* samāsa splitting",
  which the write-up must say wherever it names them.
* Its output is **not a pure re-segmentation**: it normalises, drops sentence punctuation
  and occasionally drops a transliterated loanword, keeping 88.5% of the raw SLP1
  non-space characters over the 200 benchmarked sentences (9.06 → 9.64 whitespace units;
  97 gained, 54 lost, 49 unchanged). Task 4 reports this per corpus.

**Throughput** (200 Sāmayik-test sentences, batch 16, M3 Pro, torch 2.14.0):
7.12 sentences/s on `mps`, 2.52 on `cpu`, **byte-identical output on both**. Projected on
`mps`: 4.59 h for the 117,720-sentence training corpus and 0.75 h for the 19,197
evaluation sentences. 4.59 h is inside the rule's 8 h budget, so **the full training
corpus is split** and no `_sub` arms are needed (`docs/decisions.md`, "Splitter throughput
measured"; `outputs/03_sandhi_split/benchmark_{mps,cpu}.json`).

## Running it

```bash
# throughput benchmark (throwaway cache; never writes data/processed/split/)
uv run python experiments/03_sandhi_split/benchmark_splitter.py --n 200 --device mps
```

**Expected runtime.** Benchmark: minutes. Task 3's corpus split: hours, in the background
and resumable through the cache. Task 4's runner: minutes.
