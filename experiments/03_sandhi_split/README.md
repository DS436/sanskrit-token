# Experiment 03 — Sandhi-split arms (T4): does reversing sandhi first recover token efficiency?

**Hypothesis (outline §1, RQ3/H3, verbatim):** "Sandhi splitting before subword learning
raises MorphScore and lowers fertility relative to a matched-vocabulary tokenizer trained
on raw sandhied text."

**What this experiment tests.** The TPP/fertility half of H3 only. MorphScore and the
gold-segmentation upper bound (ablation A3) need DCS morpheme boundaries, which
Experiment 04 ingests for the `T5`/`T6` arms anyway; both are deferred there
(`docs/decisions.md`, "Experiment 03 tests the TPP/fertility half of H3; MorphScore moves
to Experiment 04"). So no result here settles H3 in full: it settles whether splitting
first buys token efficiency, not whether it buys morphological alignment.

**And the headline metric is TPP, not the fertility H3 names.** Fertility asks how many
tokens a *word* costs, which punishes Sanskrit for exactly the density under study, so it
is always computed, always reported and never led with (CLAUDE.md §2.1). Splitting makes
it doubly awkward: inserting whitespace changes the denominator, so fertility is reported
twice — primary over the **raw** sentence's word count (the same denominator for every
arm) and secondary over the split text's own words (`docs/decisions.md`, "Fertility for
split arms uses the raw word count as the primary denominator").

**Success (pre-registered).** On Sāmayik test — prose first (CLAUDE.md §2.7) — each split
arm's controlled TPP is **below** its matched raw arm's, and the paired-bootstrap CI on
the difference **excludes 0**. The difference is the claim, not the level: an arm sitting
below 1.0 says something about the language pair and the English pivot, not about
splitting. Both arms of a pair are divided by the same `E1_*` control (same algorithm,
vocabulary size and training corpus), so the pivot cancels out of the sign.

**Scope.** Four matched pairs, `T1`/`T2` (raw) against `T4` (split) at BPE and Unigram,
32k and 64k. Every arm is **provisional** in the same sense as Experiment 02's: trained on
the Sanskrit (or English) side of two parallel corpora, not on the monolingual corpus of
milestone M1. `T4` measures **sandhi *and* compound (samāsa) splitting**, because the
splitter does both — see **The splitter** below; every table and the paper must name it
that way.

**Run:** `uv run python experiments/03_sandhi_split/run.py` (after the corpus split and
the `T4` training of Task 3).

**Runtime:** `<pending run>`.

---

## Status

| Task | State |
|---|---|
| 1. Shared experiment helpers (`sanskrit_tok.experiment`) | done |
| 2. `SandhiSplitter` + cache + throughput benchmark | done |
| 3. Split the corpora, train the `T4` arms | corpus split running; `T4` arms not yet trained |
| 4. Runner, `results.json`, figure, results table | code and tests done; numbers pending |

Every number below lands when Task 4 Part B runs the experiment for real against the
finished split cache and the trained `T4` arms.

## Summary

`<pending run>` — leads with the controlled TPP delta on Sāmayik test (split minus raw,
with its paired-bootstrap CI), then the other three corpora, then what the fertility pair
says, and last the verdict on the TPP half of H3 scoped to the arms that exist.

## Results

TPP is the ratio Sanskrit tokens / English tokens over aligned sentence pairs, against the
matched English control `E1_*`; **Δ** is split minus raw with a paired bootstrap over the
same sentence pairs (a negative Δ whose CI excludes 0 is the H3 prediction). Fertility
columns are the primary form — tokens over the **raw** sentence's word count for both arms
— and never lead. All arms provisional (`*`).

### Sāmayik test (prose, primary)

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | fertility raw | fertility split |
|---|---|---|---|---|---|
| bpe 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| bpe 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |

### Sāmayik test_ood (prose, out-of-domain for both sides)

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | fertility raw | fertility split |
|---|---|---|---|---|---|
| bpe 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| bpe 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |

### Itihāsa test (verse, secondary — meter is a confound)

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | fertility raw | fertility split |
|---|---|---|---|---|---|
| bpe 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| bpe 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |

### FLORES devtest (tertiary)

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | fertility raw | fertility split |
|---|---|---|---|---|---|
| bpe 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| bpe 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 32k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |
| unigram 64k | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` | `<pending run>` |

**Deployed practice.** Every arm is also measured against `T0_o200k` (200k, general
domain) and stored under the same `tpp` key. That is a description of what today's
tokenizers charge for Sanskrit, never a controlled comparison (CLAUDE.md §2.5), and the
delta above is not read from it. `<pending run>`.

**Secondary variant.** Every `T4` number above is measured on the **reconciled** split
text. The splitter's unreconciled output is measured too, under `model_raw`, and reported
here as a robustness check: `<pending run>`. It is expected to look *better* than the
reconciled variant, because the model deletes characters (see below) — which is precisely
why it is not the primary.

**Secondary fertility.** Tokens per *split* word, a different denominator from every other
number in this file: `<pending run>`.

## Figure

`outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` — one row per corpus (prose first),
one x position per matched pair, two markers at each (raw arm, split arm) with their 95%
bootstrap CIs, against the matched English control, with a dashed reference line at 1.0.
The error bars drawn are each arm's own; the interval on the **difference** is narrower,
because the two arms are paired on the same sentences, and is the one the verdict uses.

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
  and occasionally drops a transliterated loanword, keeping **87.3%** of the raw SLP1
  non-space characters over the 200 benchmarked sentences (pooled;
  `segmentation.char_retention_nonspace` in `benchmark_mps.json`, which carries the
  definition). Whitespace units go 9.06 → 9.64 (97 sentences gained, 54 lost, 49
  unchanged).

That last fact is why the primary `T4` text is the model's segmentation **reconciled**
against the raw sentence (`sandhi.reconcile`; `docs/decisions.md`, "T4 text is the model's
segmentation reconciled against the raw sentence"): measured on the raw output, TPP would
credit `T4` with every token the *splitter* saved by deleting a word. `results.json`'s
`splitter_stats` reports, per corpus, mean whitespace units raw vs split, the share of
sentences whose unit count changed, and the manifest's character retention before and
after reconciliation: `<pending run>`.

**Throughput** (200 Sāmayik-test sentences, batch 16, M3 Pro, torch 2.14.0):
7.09 sentences/s on `mps`, 2.52 on `cpu`, **byte-identical output on both**. Projected on
`mps`: 4.61 h for the 117,720-sentence training corpus and 0.75 h for the 19,197
evaluation sentences. 4.61 h is inside the rule's 8 h budget, so **the full training
corpus is split** and no `_sub` arms are needed (`docs/decisions.md`, "Splitter throughput
measured"; `outputs/03_sandhi_split/benchmark_{mps,cpu}.json`). Actual wall-clock for the
full split run: `<pending run>`.

## Caveats

* **Provisional arms.** Every `T1`/`T2`/`T4`/`E1` arm is trained on the Sanskrit or
  English side of two parallel corpora (117,720 Sanskrit sentences), not on the
  monolingual corpus of milestone M1. Absolute levels will move when that corpus exists;
  the matched *differences* are the more durable quantity.
* **The split text is not the input text.** The splitter rewrites as well as re-segments
  (87.3% raw character retention before reconciliation). Reconciliation restores the
  deletions, but a `T4` number is still measured over text a model has touched, and the
  reconciliation threshold (0.6) is a judgement call recorded in the manifest.
* **Domain fit.** Sāmayik test is in-domain for the training corpus on both sides;
  `test_ood` and FLORES are out-of-domain for both. Experiment 02 showed how much of an
  apparent Sanskrit advantage a mismatched English pivot can manufacture, which is why
  every number here is read against `E1_*` and not against `T0_o200k`.
* **Verse.** Itihāsa is śloka: meter constrains word choice, and its English side is a
  19th-century verse translation. Prose comes first for that reason (CLAUDE.md §2.7).
* **Hindi is not applicable here.** The Hindi pivot needs one tokenizer scoring both
  sides (Experiment 02's `T0`/`T3` arms). A Sanskrit-trained `T1`/`T4` arm scoring Hindi
  would measure transfer, not parity, so no Hindi column exists in this experiment.
* **MorphScore is deferred**, with the gold-segmentation upper bound, to Experiment 04.
  Nothing here shows whether `T4`'s boundaries are *morphological*, only what they cost.

## Files

| Path | What |
|---|---|
| `config.yaml` | corpora, arms, matched pairs, control mapping, bootstrap settings |
| `run.py` | the runner: TPP, the paired delta, both fertilities, compression, the figure |
| `split.yaml`, `split_corpora.py` | Task 3's corpus split (multi-hour, background, resumable) |
| `benchmark_splitter.py` | Task 2's throughput and behaviour benchmark |
| `outputs/03_sandhi_split/results.json` | every number, with provenance and config |
| `outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` | the figure |
