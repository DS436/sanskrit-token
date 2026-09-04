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
words a sentence costs; TPP asks how many tokens a unit of meaning costs. Fertility here
is always computed, always reported and never led with (CLAUDE.md §2.1). Splitting makes
it doubly awkward: inserting whitespace changes the denominator, so fertility is reported
twice — primary over the **raw** sentence's word count (the same denominator for every
arm) and secondary over the split text's own words (`docs/decisions.md`, "Fertility for
split arms uses the raw word count as the primary denominator").

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
| 4. Runner, `results.json`, figure, this file | done — **re-run after the punctuation fix** |

Every number here is from the corrected re-run of 2026-09-05. The previous version of this
file reported a result that was mostly an artefact of reconciliation deleting punctuation;
see **What changed, and why** and **Limitations**.

---

## What changed, and why

The first version of this experiment reported that `T4_bpe_split_64k` reached a controlled
TPP of **0.976** on Sāmayik test — the first arm in the project below its matched English
control on prose. That result no longer exists.

`sandhi.reconcile` protected punctuation only when it stood as its own whitespace unit.
Punctuation *fused* to a word — `karoti.`, `"tadA,` — was deleted along with the unit it
was attached to, on 79% of such units on Sāmayik test and 87% on Itihāsa. The split arms
were being credited with tokens saved by deleting characters, not by inserting boundaries,
and priced by re-tokenising, those deletions accounted for 77–110% of the Sāmayik-test
delta (`docs/decisions.md`, "Reconciliation must preserve every non-letter character").

Reconciliation now peels each unit into a non-letter prefix, a letter core and a non-letter
suffix, aligns only the core, and re-attaches the affixes to the first and last output
segments with no added whitespace — so restoring them cannot manufacture a boundary token
either. With the fix, `T4_bpe_split_64k` on Sāmayik test is **1.027**, its delta is
**+0.0002 [−0.0042, +0.0052]**, and the claim it supported is withdrawn.

---

## Results — controlled TPP against the matched English control

The headline. Δ is `TPP(split arm on reconciled text) − TPP(raw arm on raw text)`, both
divided by the same English control, over the same sentences, with a paired bootstrap
(1,000 resamples, seed 0). Negative is the predicted direction. **A TPP below 1.0 says
nothing about splitting** — it is a fact about the language pair; only Δ speaks to H3.

`deletion cost` is the re-tokenised price of whatever characters the reconciled text still
lacks: tokens the *raw* arm spends on them, i.e. how much of the saving is bought by
missing content rather than by better segmentation. Never priced at mean bytes-per-token —
punctuation costs ≈0.5 tokens/character under these arms and letters ≈0.2, so an average is
wrong by a factor of two in the flattering direction. A **negative** deletion cost means
removing those characters would have made the raw text tokenise *worse*, so they were not
subsidising the split arm at all.

#### samayik_test

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | deletion cost (tok) | Sanskrit tokens saved | share of saving |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.077 | 1.067 | **-0.0100** | [-0.0142, -0.0052] | -2105 | +367 | -574% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.027 | 1.027 | **+0.0002** | [-0.0042, +0.0052] | -2247 | -8 | n/a |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.148 | 1.105 | **-0.0434** | [-0.0492, -0.0378] | -1860 | +1592 | -117% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.096 | 1.062 | **-0.0344** | [-0.0401, -0.0290] | -1489 | +1232 | -121% |

#### samayik_test_ood

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | deletion cost (tok) | Sanskrit tokens saved | share of saving |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.091 | 1.026 | **-0.0649** | [-0.0683, -0.0617] | +310 | +6659 | +5% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.067 | 1.010 | **-0.0565** | [-0.0597, -0.0534] | -178 | +5461 | -3% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.161 | 1.064 | **-0.0972** | [-0.1009, -0.0933] | +874 | +10141 | +9% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.146 | 1.041 | **-0.1052** | [-0.1088, -0.1014] | +987 | +10486 | +9% |

#### itihasa_test

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | deletion cost (tok) | Sanskrit tokens saved | share of saving |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 0.653 | 0.662 | **+0.0090** | [+0.0078, +0.0102] | -15724 | -3484 | +451% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 0.607 | 0.625 | **+0.0180** | [+0.0169, +0.0192] | -18084 | -6770 | +267% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 0.663 | 0.670 | **+0.0066** | [+0.0053, +0.0078] | -9974 | -2636 | +378% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 0.626 | 0.638 | **+0.0121** | [+0.0108, +0.0132] | -10435 | -4745 | +220% |

#### flores_devtest

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | deletion cost (tok) | Sanskrit tokens saved | share of saving |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.139 | 1.103 | **-0.0365** | [-0.0412, -0.0320] | -215 | +1136 | -19% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.138 | 1.103 | **-0.0345** | [-0.0391, -0.0300] | -288 | +1000 | -29% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.207 | 1.133 | **-0.0737** | [-0.0791, -0.0686] | +156 | +2373 | +7% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.219 | 1.144 | **-0.0752** | [-0.0819, -0.0692] | +177 | +2258 | +8% |

**Reading the four corpora together.** Sāmayik test_ood and FLORES move the predicted way
in all eight pairs, with CIs clear of 0 and deletion cost explaining at most 9% of the
saving. Sāmayik test moves the predicted way in three of four pairs, but the effects are
small (−0.010 to −0.043) and the BPE 64k pair shows nothing at all. Itihāsa moves the
**wrong** way in all four pairs, with CIs clear of 0: on verse, splitting costs tokens.

### Deployed practice (not a controlled comparison)

`results.json`'s `tpp` block also carries every arm against `T0_o200k` and `T0_llama4`.
Those are **deployed-practice columns only**: the pivots are 200k-vocabulary
general-domain tokenizers trained on different data at a different vocabulary size, so a
ratio against them holds nothing constant and cannot be read as evidence about splitting.
They are here because "what does this cost against what people actually deploy" is a
legitimate question, not because they are a control.

---

## The splitter, and what it did to each corpus

`chronbmm/sanskrit5-multitask@c0d2ada54f3d19903149425aa888a203601423f8`, segmentation mode,
MPS. The 9.4-hour run of 2026-09-04/05 is recorded in `data/processed/split/manifest.json`
and, for the original run's throughput and chunking facts,
`data/processed/split/manifest_original_run.json`; per-run copies are timestamped.

| corpus | mean units raw | mean units split | letter retention | non-letter multiset preserved |
|---|---:|---:|---:|---|
| samayik_test | 9.59 | 11.31 | 1.0138 | no |
| samayik_test_ood | 11.82 | 15.20 | 1.0095 | no |
| itihasa_test | 11.16 | 15.18 | 1.0207 | no |
| flores_devtest | 16.77 | 19.26 | 1.0057 | no |

Letter retention above 1.0 is expected and is the splitter doing its job: reversing sandhi
restores elided phonemes (`prARina Agatya` → `prARinaH Agatya`).

**The non-letter multiset is still not preserved, and the runner says so on every corpus.**
Affix punctuation is now safe; what remains is *mid-word* marks, and they are a different
phenomenon:

- **hyphens inside compounds** (3,083 on Sāmayik test_ood): `parAmarSa-dUraBAzA` →
  `parAmarSa dUra BAzA`. The splitter turns a compound boundary marked by a hyphen into a
  boundary marked by a space. Forcing the hyphen back would corrupt the segmentation.
- **avagraha** `'` (1,711 on Itihāsa): `vaMSo'yaM` → `vaMSaH ayam`. The mark exists
  precisely to record an elided vowel; sandhi reversal restores the vowel *as a letter*, so
  the mark is consumed rather than deleted.
- **unconverted Devanagari signs** (`ॉ`, `ॅ`, `़`): the known SLP1 coverage gap
  (`docs/decisions.md`, "Report the Hindi SLP1 variant as approximate").
- **characters the model *adds*** — chiefly `/` (434 on Itihāsa) and quotation marks. These
  are paid for by the split arm, so they push the comparison against `T4`, not for it.

Whether these should be preserved is a question the multiset check cannot answer; what it
can do is force the question, which is why the deletion-cost column is reported beside every
delta. On these corpora that column is small (≤9% of the saving on test_ood and FLORES) or
negative, so the deltas above are not deletion artefacts.

---

## Verdict — the TPP/fertility half of H3

**Withdrawn and restated.** The previous verdict ("all four pairs meet the pre-registered
criterion on prose; the first arm below its matched English control") was an artefact of
deleted punctuation and is withdrawn.

**Restated on the corrected numbers: sandhi-and-compound splitting buys a real but small
token saving on prose, buys nothing at 64k BPE on in-domain prose, and costs tokens on
verse.** Precisely:

- On **out-of-domain prose** (Sāmayik test_ood) and **FLORES**, all eight matched pairs move
  the predicted way, Δ −0.034 to −0.105, every CI clear of 0, deletion cost ≤9% of the
  saving. This is the strongest form of the result.
- On **in-domain prose** (Sāmayik test), three of four pairs move the predicted way but
  small (−0.010 to −0.043); **`T4_bpe_split_64k` shows no effect** (+0.0002, CI spans 0).
- On **verse** (Itihāsa), all four pairs move the **wrong** way (+0.007 to +0.018, CIs clear
  of 0). Splitting is not free, and on metrical text it does not pay.
- **No arm is below its matched English control on prose.** The lowest split TPP on prose is
  1.010 (`T4_bpe_split_64k*`, test_ood). Sanskrit still costs more tokens per proposition
  than matched-vocabulary English on every prose corpus here.

So H3's TPP half is **partially supported, with a corpus-dependent sign**, and the effect is
an order of magnitude smaller than the withdrawn one.

---

## Limitations

1. **The punctuation episode.** The first published version of this experiment reported an
   effect that was mostly reconciliation deleting characters. It was found by a review, not
   by the pipeline, which is why `text_invariants` and the re-tokenised deletion-cost column
   are now mandatory for anything measured on transformed text
   (`docs/decisions.md`, "Reconciliation must preserve every non-letter character"). The
   general lesson: a transformation's *retention* statistic (0.9895, healthy-looking) did not
   detect a deletion that was worth 100% of the effect size.
2. **Orthographic normalisation is an uncontrolled confound.** The splitter does not only
   insert boundaries: it rewrites anusvāra `M` → `m`, `:` → `H`, restores visargas, and
   normalises inflected endings. 45.7% of raw whitespace units come back changed by more
   than a boundary. Some of the TPP saving may be the split text being *more regular*, not
   better segmented. Experiment 04's gold-split `_oracle` arm separates the two.
3. **Boundary markers are the hypothesised mechanism, not a measured one.** Under Metaspace
   pre-tokenisation every word starts with `▁`, so splitting a compound into three words adds
   two boundary markers and *should* cost tokens; that the total still falls means the
   sub-word merges are getting cheaper by more than the markers cost. That is a plausible
   mechanism, not something this experiment measured.
4. **The English side is unchanged.** Only the Sanskrit side is split, so Δ mixes "splitting
   helps Sanskrit" with "the English control is a fixed reference". That is the intended
   design, but it means Δ is not a symmetric quantity.
5. **In-domain training text.** The `T4` arms are trained on the Sanskrit side of the same
   two corpora whose test splits they are evaluated on (disjoint sentences, exclusion-list
   enforced). Sāmayik test_ood exists precisely because of this, and it is where the effect
   is strongest — which is reassuring, but the in-domain/out-of-domain gap is not explained.
6. **Verse is an adverse pair, and it is not just meter.** Itihāsa's positive Δ could be
   meter constraining word choice, or the splitter being weaker on Epic register, or both.
   Undiagnosed.
7. **Provisional arms.** Every `T1`/`T2`/`T4`/`E1` arm is trained on parallel-corpus training
   splits, not the monolingual corpus of milestone M1. Absolute TPP levels will move when M1
   lands; the matched deltas should be more stable, but that is an assumption.
8. **Corpus sizes differ slightly.** The raw training corpus is 117,720 lines and the split
   one 117,639, from the identical 117,721 sentences — reconciliation is marginally more
   many-to-one than transliteration. 0.07%, far below the noise floor, but stated.
9. **MorphScore and the gold-split upper bound are absent**, deferred to Experiment 04. Nothing
   here says whether splitting improves *morphological* alignment, which is the other half of H3.

---

## Files

- `run.py`, `config.yaml` — the runner and its config.
- `split_corpora.py`, `split.yaml` — the splitting job (Task 3); writes
  `data/processed/split/`.
- `benchmark_splitter.py` — the throughput benchmark (Task 2).
- `outputs/03_sandhi_split/results.json` — every number above, with provenance.
- `outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` — the figure.
