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

**And the headline metric is TPP, not the fertility H3 names.** Fertility is tokens per
whitespace-delimited word; TPP is tokens per unit of meaning. Fertility here is always
computed, always reported and never led with (CLAUDE.md §2.1). Splitting makes it doubly
awkward: inserting whitespace changes the denominator, so fertility is reported twice —
primary over the **raw** sentence's word count (the same denominator for every arm) and
secondary over the split text's own words (`docs/decisions.md`, "Fertility for split arms
uses the raw word count as the primary denominator").

**Every `T1`/`T2`/`T4`/`E1` arm below is marked `*` — provisional**, per the plan's Global
Constraints: trained on the Sanskrit (or English) side of two parallel corpora, not on the
monolingual corpus of milestone M1.

**Scope.** Four matched pairs, `T1`/`T2` (raw) against `T4` (split) at BPE and Unigram, 32k
and 64k, each against the English control that matches it on algorithm, vocabulary size and
training corpus. `T4` measures **sandhi *and* compound (samāsa) splitting**, because the
splitter does both — see **The splitter** below; every table and the paper must name it
that way.

**Run:** `uv run python experiments/03_sandhi_split/run.py` (55 s, no network, the splitter
model is never loaded — the 9.4 h of model time is Task 3's, spent once).

---

## Status

| Task | State |
|---|---|
| 1. Shared experiment helpers | done |
| 2. `SandhiSplitter` + cache + throughput benchmark | done |
| 3. Split the corpora, train the `T4` arms | done |
| 4. Runner, `results.json`, figure, this file | done — **re-run twice after review** |

Every number here is from the run of 2026-09-05 at commit `6c3be51`, clean tree
(`results.json` records `git_dirty: false`). Two earlier versions of this file are
superseded; see **What changed, and why**.

---

## What changed, and why

Two review findings, in order, each of which shrank the result.

**1. Attached punctuation was being deleted.** `sandhi.reconcile` protected punctuation only
when it stood as its own whitespace unit; punctuation *fused* to a word — `karoti.`,
`"tadA,` — was deleted with the unit it belonged to. The split arms were credited with
tokens saved by removing characters rather than by inserting boundaries. Reconciliation now
peels each unit into a non-letter prefix, a letter core and a non-letter suffix, aligns only
the core, and re-attaches the affixes without adding whitespace. That withdrew the original
claim that `T4_bpe_split_64k` reached a controlled TPP of **0.976** on Sāmayik test — the
first arm in the project below its matched English control on prose.

**2. Raw hyphens are boundaries the raw arm already pays for.** A hyphenated compound
(`parAmarSa-dUraBAzA`) is *already* segmented in the source; converting the hyphen to
whitespace credits `T4` with a boundary it did not find. Reconciliation now splits each unit's
core on `-`, aligns the parts independently, and rejoins them with the original hyphens, so
only boundaries found *inside* a part become spaces (`docs/decisions.md`, "Raw hyphens are
pre-existing boundaries"). The deletion-cost column was also wrong: it stripped letters as
well as non-letters, so it measured garbling by removed anusvāras rather than deletion. It
now prices non-letters only, and its share is undefined when the saving is not positive.

The corrected effect on Sāmayik test_ood is roughly **half** what the previous version
reported (−0.025 to −0.069, against −0.057 to −0.105), almost all of the difference being
hyphens that are no longer credited.

---

## Results

Δ is `TPP(split arm on reconciled text) − TPP(raw arm on raw text)`, both divided by the same
English control, over the same sentences, with a paired bootstrap (1,000 resamples, seed 0).
Negative is the predicted direction. **A TPP below 1.0 says nothing about splitting** — it is
a fact about the language pair; only Δ speaks to H3.

`non-letter deletion cost` is the re-tokenised price of the non-letter characters the
reconciled text still lacks: tokens the *raw* arm spends on them, i.e. how much of the saving
is bought by missing content rather than by better segmentation. Never priced at mean
bytes-per-token — punctuation costs ≈0.5 tokens/character under these arms and letters ≈0.2,
so an average is wrong by a factor of two in the flattering direction. The share is left
blank where the saving is not positive, since a share of a saving that does not exist is not
a number.

### Controlled TPP

#### samayik_test

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.077 | 1.065 | **-0.0119** | [-0.0158, -0.0074] | +32 | +439 | +7% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.027 | 1.021 | **-0.0053** | [-0.0096, -0.0006] | +29 | +183 | +16% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.148 | 1.101 | **-0.0470** | [-0.0524, -0.0417] | +20 | +1724 | +1% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.096 | 1.056 | **-0.0401** | [-0.0455, -0.0347] | +19 | +1437 | +1% |

#### samayik_test_ood

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.091 | 1.061 | **-0.0302** | [-0.0328, -0.0273] | +185 | +3095 | +6% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.067 | 1.041 | **-0.0254** | [-0.0278, -0.0226] | +167 | +2454 | +7% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.161 | 1.099 | **-0.0616** | [-0.0649, -0.0584] | +159 | +6431 | +2% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.146 | 1.077 | **-0.0690** | [-0.0719, -0.0658] | +165 | +6876 | +2% |

#### itihasa_test

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 0.653 | 0.661 | **+0.0081** | [+0.0069, +0.0093] | +222 | -3139 | — |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 0.607 | 0.624 | **+0.0171** | [+0.0160, +0.0183] | +99 | -6451 | — |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 0.663 | 0.669 | **+0.0056** | [+0.0043, +0.0068] | -107 | -2241 | — |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 0.626 | 0.637 | **+0.0114** | [+0.0101, +0.0125] | -176 | -4466 | — |

#### flores_devtest

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.139 | 1.103 | **-0.0366** | [-0.0411, -0.0320] | +78 | +1140 | +7% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.138 | 1.102 | **-0.0356** | [-0.0401, -0.0313] | +79 | +1031 | +8% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.207 | 1.132 | **-0.0751** | [-0.0805, -0.0700] | +81 | +2415 | +3% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.219 | 1.144 | **-0.0753** | [-0.0819, -0.0695] | +82 | +2263 | +4% |

### Fertility — never the headline (CLAUDE.md §2.1)

Primary denominator is the **raw** sentence's whitespace-word count, identical for every
arm. The secondary column divides a split arm's tokens by the *split* text's own word
count and is not comparable with any raw arm.

| corpus | matched pair | fertility raw → split (raw-word denom.) | split arm, secondary (split-word denom.) |
|---|---|---:|---:|
| samayik_test | T1_bpe_raw_32k* → T4_bpe_split_32k* | 1.696 → 1.694 | 1.459 |
| samayik_test | T1_bpe_raw_64k* → T4_bpe_split_64k* | 1.525 → 1.533 | 1.320 |
| samayik_test | T2_unigram_raw_32k* → T4_unigram_split_32k* | 1.801 → 1.742 | 1.500 |
| samayik_test | T2_unigram_raw_64k* → T4_unigram_split_64k* | 1.680 → 1.633 | 1.406 |
| samayik_test_ood | T1_bpe_raw_32k* → T4_bpe_split_32k* | 2.340 → 2.275 | 1.859 |
| samayik_test_ood | T1_bpe_raw_64k* → T4_bpe_split_64k* | 2.156 → 2.105 | 1.720 |
| samayik_test_ood | T2_unigram_raw_32k* → T4_unigram_split_32k* | 2.532 → 2.397 | 1.959 |
| samayik_test_ood | T2_unigram_raw_64k* → T4_unigram_split_64k* | 2.388 → 2.244 | 1.834 |
| itihasa_test | T1_bpe_raw_32k* → T4_bpe_split_32k* | 1.941 → 1.965 | 1.448 |
| itihasa_test | T1_bpe_raw_64k* → T4_bpe_split_64k* | 1.748 → 1.797 | 1.325 |
| itihasa_test | T2_unigram_raw_32k* → T4_unigram_split_32k* | 2.022 → 2.039 | 1.503 |
| itihasa_test | T2_unigram_raw_64k* → T4_unigram_split_64k* | 1.877 → 1.911 | 1.408 |
| flores_devtest | T1_bpe_raw_32k* → T4_bpe_split_32k* | 2.082 → 2.023 | 1.773 |
| flores_devtest | T1_bpe_raw_64k* → T4_bpe_split_64k* | 1.935 → 1.882 | 1.650 |
| flores_devtest | T2_unigram_raw_32k* → T4_unigram_split_32k* | 2.280 → 2.146 | 1.881 |
| flores_devtest | T2_unigram_raw_64k* → T4_unigram_split_64k* | 2.150 → 2.025 | 1.775 |

### Secondary variant: the splitter's raw output (`model_raw`)

Not content-preserving — it is what the model emitted before reconciliation, and it
deletes words. Reported so the reconciliation's effect is visible, never as a result.

| corpus | split arm | controlled TPP on reconciled text | on raw model output |
|---|---|---:|---:|
| samayik_test | T4_bpe_split_32k* | 1.065 | 0.834 |
| samayik_test | T4_bpe_split_64k* | 1.021 | 0.819 |
| samayik_test | T4_unigram_split_32k* | 1.101 | 0.872 |
| samayik_test | T4_unigram_split_64k* | 1.056 | 0.849 |
| samayik_test_ood | T4_bpe_split_32k* | 1.061 | 0.854 |
| samayik_test_ood | T4_bpe_split_64k* | 1.041 | 0.842 |
| samayik_test_ood | T4_unigram_split_32k* | 1.099 | 0.885 |
| samayik_test_ood | T4_unigram_split_64k* | 1.077 | 0.870 |
| itihasa_test | T4_bpe_split_32k* | 0.661 | 0.552 |
| itihasa_test | T4_bpe_split_64k* | 0.624 | 0.529 |
| itihasa_test | T4_unigram_split_32k* | 0.669 | 0.571 |
| itihasa_test | T4_unigram_split_64k* | 0.637 | 0.551 |
| flores_devtest | T4_bpe_split_32k* | 1.103 | 0.965 |
| flores_devtest | T4_bpe_split_64k* | 1.102 | 0.966 |
| flores_devtest | T4_unigram_split_32k* | 1.132 | 0.988 |
| flores_devtest | T4_unigram_split_64k* | 1.144 | 1.000 |

### Deployed practice (not a controlled comparison)

`results.json`'s `tpp` block also carries every arm against `T0_o200k`. That is a
**deployed-practice column only**: a 200k-vocabulary general-domain tokenizer trained on
different data at a different vocabulary size, so a ratio against it holds nothing constant
and cannot be read as evidence about splitting. It is here because "what does this cost
against what people actually deploy" is a legitimate question, not because it is a control.

---

## The splitter, and what it did to each corpus

`chronbmm/sanskrit5-multitask@c0d2ada54f3d19903149425aa888a203601423f8`, segmentation mode,
MPS. The 9.4-hour run of 2026-09-04/05 is recorded in `data/processed/split/manifest.json`,
with the original run's throughput and chunking facts in
`data/processed/split/manifest_original_run.json`; every run also writes a timestamped copy.

| corpus | mean units raw | mean units split | letter retention | non-letter missing (gross) | added (gross) | of |
|---|---:|---:|---:|---:|---:|---:|
| samayik_test | 9.59 | 11.14 | 1.0139 | 134 | 12 | 8,704 |
| samayik_test_ood | 11.82 | 14.47 | 1.0096 | 397 | 14 | 18,017 |
| itihasa_test | 11.16 | 15.15 | 1.0207 | 1,806 | 3 | 44,544 |
| flores_devtest | 16.77 | 19.13 | 1.0059 | 80 | 6 | 3,735 |

Missing and added are **gross** — per-sentence differences summed, so a danda deleted in one
sentence and one invented in another count as two faults rather than cancelling to zero.
Letter retention above 1.0 is expected and is the splitter doing its job: reversing sandhi
restores elided phonemes (`prARina Agatya` → `prARinaH Agatya`).

**The non-letter multiset is still not preserved, and the runner says so on every corpus.**
Attached punctuation and hyphens are now safe. What remains, in order of size:

- **avagraha `'`** (1,711 of Itihāsa's 1,806): `vaMSo'yaM` → `vaMSaH ayam`. The mark exists to
  record an elided vowel; sandhi reversal restores the vowel *as a letter*, so the mark is
  consumed rather than deleted. Licensed — but its re-tokenised cost is still in the deletion
  column above.
- **unconverted Devanagari signs** (`ॉ`, `ॅ`, `़`): the known SLP1 coverage gap
  (`docs/decisions.md`, "Report the Hindi SLP1 variant as approximate").
- **a residue of dandas, colons and commas** the model drops mid-sentence rather than at a
  word edge.
- **characters the model *adds*** (3–14 per corpus, chiefly combining marks and `/`). These
  are paid for by the split arm, so they push the comparison against `T4`.

On these corpora the deletion column is 1–8% of the saving wherever a saving exists, so no Δ
below is materially a deletion artefact — but it is not zero, and it is reported beside every
delta rather than argued away.

---

## Verdict — the TPP/fertility half of H3

**Sandhi-and-compound splitting buys a real but small token saving on prose, and costs tokens
on verse.** Precisely:

- On **Sāmayik test** (primary prose) all four pairs move the predicted way with CIs clear of
  0, but the effects are small: −0.005 to −0.047, and the BPE 64k pair's CI reaches
  −0.0006, i.e. it barely clears zero.
- On **Sāmayik test_ood** and **FLORES**, all eight pairs move the predicted way, Δ −0.025 to
  −0.075, every CI clear of 0, deletion cost ≤8% of the saving.
- On **Itihāsa** (verse), all four pairs move the **wrong** way (+0.006 to +0.017, CIs clear
  of 0). Splitting is not free, and on metrical text it does not pay.
- **No arm is below its matched English control on prose.** The lowest split TPP on prose is
  1.021 (`T4_bpe_split_64k*`, Sāmayik test). Sanskrit still costs more tokens per proposition
  than matched-vocabulary English on every prose corpus here.
- **Fertility moves with TPP** and never leads: down on prose and FLORES, up on verse.

H3's TPP half is **supported on prose and contradicted on verse**, at an effect size an order
of magnitude smaller than the first version of this file reported.

---

## Limitations

1. **Two rounds of correction, both found by review rather than by the pipeline.** The first
   version reported an effect that was mostly deleted punctuation; the second credited `T4`
   with hyphen boundaries already present in the source. `text_invariants` and the
   re-tokenised deletion-cost column now run on every corpus, but the general lesson stands:
   a *retention* statistic looked healthy through both.
2. **Orthographic normalisation is an uncontrolled confound.** The splitter does not only
   insert boundaries: it rewrites anusvāra `M` → `m`, `:` → `H`, restores visargas, and
   normalises inflected endings. Some of the TPP saving may be the split text being *more
   regular*, not better segmented. Experiment 04's gold-split `_oracle` arm separates the two.
3. **Boundary markers are the hypothesised mechanism, not a measured one.** Under Metaspace
   pre-tokenisation every word starts with `▁`, so splitting a compound into three words adds
   two boundary markers and *should* cost tokens; that the total still falls means the subword
   merges are getting cheaper by more than the markers cost. Plausible, not measured.
4. **The English side is unchanged**, so Δ mixes "splitting helps Sanskrit" with "the English
   control is a fixed reference". Intended, but Δ is not a symmetric quantity.
5. **In-domain training text.** The `T4` arms are trained on the Sanskrit side of the same two
   corpora whose test splits they are evaluated on (disjoint sentences, exclusion-list
   enforced). Sāmayik test_ood exists because of this.
6. **Verse is an adverse pair, and it is not just meter.** Itihāsa's positive Δ could be meter
   constraining word choice, the splitter being weaker on Epic register, or the 1,711 avagraha
   consumptions. Undiagnosed.
7. **Avagraha is licensed but not free.** It is the largest remaining non-letter difference and
   is concentrated in exactly the corpus that behaves adversely.
8. **Provisional arms.** Absolute TPP levels will move when milestone M1 lands; the matched
   deltas should be more stable, but that is an assumption.
9. **Corpus sizes differ slightly.** The raw training corpus is 117,720 lines and the split one
   117,642, from the identical 117,721 sentences. 0.07%, below the noise floor, but stated.
10. **MorphScore and the gold-split upper bound are absent**, deferred to Experiment 04.

---

## Files

- `run.py`, `config.yaml` — the runner and its config.
- `split_corpora.py`, `split.yaml` — the splitting job (Task 3); writes `data/processed/split/`.
- `benchmark_splitter.py` — the throughput benchmark (Task 2).
- `outputs/03_sandhi_split/results.json` — every number above, with provenance.
- `outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` — the figure.
