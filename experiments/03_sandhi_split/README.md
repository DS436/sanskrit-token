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

> **Re-run 2026-09-05 at commit `b7302a8`, clean tree.** Every trained arm was retrained
> after the trainers stopped letting line breaks reach the pre-tokenizer
> (`docs/decisions.md`, 2026-09-05, "Trainers strip newlines", and the CORRECTION entry for
> this experiment). Every number below moved; **no controlled delta changed direction and
> no CI changed its relation to zero.** The two changes are in the *deployed-practice*
> `tpp` block, where `T1_bpe_raw_32k`* and `T4_bpe_split_32k`* against `T0_o200k` on
> Sāmayik test now sit below 1.0 with CIs excluding it. Note also that the
> `E1_unigram_64k` control trains to 62,896 pieces rather than 64,000 (Experiment 02's
> README, footnote ‡).

---

## Status

| Task | State |
|---|---|
| 1. Shared experiment helpers | done |
| 2. `SandhiSplitter` + cache + throughput benchmark | done |
| 3. Split the corpora, train the `T4` arms | done |
| 4. Runner, `results.json`, figure, this file | done — **re-run three times after review** |

Every number here is from the run of 2026-09-05 at commit `b7302a8`, clean tree
(`results.json` records `git_dirty: false`), after every trained arm was retrained with
line breaks stripped before pre-tokenisation. Four earlier versions of this file are
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
reported (−0.025 to −0.077, against −0.057 to −0.105), almost all of the difference being
hyphens that are no longer credited.

**4. Line breaks were reaching the pre-tokenizer, so 5–15% of every BPE vocabulary was
dead.** `Tokenizer.train([path])` leaves each line's `\n` attached and `Metaspace` learns
line-final words as `word\n` — ids no inference-time string can ever produce. The waste was
not even across arms, so "matched vocabulary size" (CLAUDE.md §2.5) was off by about 7% of
live entries. Both trainers now stream the corpus through `iter_corpus_lines` and call
`train_from_iterator`, and every arm in the project was retrained
(`docs/decisions.md`, "Trainers strip newlines"). Effect on this experiment's conclusions:
none. Every delta kept its direction and its CI-versus-zero status; the largest movement in
a controlled delta is Sāmayik test_ood Unigram 64k, −0.0688 → −0.0773.

**3. The rejoin duplicated letters when the model kept a hyphenated compound whole.** `log-in`
came back as `login-in` (27 records: `plag-ins` → `plagins-ins`, `Super-G` → `SuperG-G`). The
model's segments are now split on hyphens too. The bug deleted nothing and preserved every
non-letter, so both existing invariants passed it; a third, letter containment, was added, and
the exact hyphen-count invariant is what actually pins the class. Effect on the results: none
visible beyond the fourth decimal — no row changed direction or CI-vs-zero status.

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
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.084 | 1.070 | **-0.0132** | [-0.0175, -0.0086] | +33 | +473 | +7% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.035 | 1.030 | **-0.0053** | [-0.0099, -0.0005] | +32 | +179 | +18% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.142 | 1.099 | **-0.0432** | [-0.0487, -0.0376] | +16 | +1592 | +1% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.107 | 1.066 | **-0.0406** | [-0.0462, -0.0351] | +19 | +1445 | +1% |

#### samayik_test_ood

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.086 | 1.056 | **-0.0295** | [-0.0323, -0.0268] | +182 | +3000 | +6% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.061 | 1.036 | **-0.0247** | [-0.0272, -0.0219] | +162 | +2359 | +7% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.163 | 1.098 | **-0.0642** | [-0.0673, -0.0610] | +170 | +6734 | +3% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.156 | 1.079 | **-0.0773** | [-0.0803, -0.0741] | +175 | +7694 | +2% |

#### itihasa_test

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 0.645 | 0.655 | **+0.0092** | [+0.0080, +0.0104] | +223 | -3548 | — |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 0.598 | 0.616 | **+0.0181** | [+0.0170, +0.0193] | +72 | -6802 | — |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 0.658 | 0.664 | **+0.0059** | [+0.0047, +0.0072] | -84 | -2412 | — |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 0.623 | 0.634 | **+0.0116** | [+0.0104, +0.0128] | -145 | -4656 | — |

#### flores_devtest

| matched pair (raw → split) | control | TPP raw | TPP split | Δ | 95% CI | non-letter deletion cost | Sanskrit tokens saved | share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| T1_bpe_raw_32k* → T4_bpe_split_32k* | E1_bpe_32k* | 1.143 | 1.106 | **-0.0366** | [-0.0409, -0.0322] | +78 | +1129 | +7% |
| T1_bpe_raw_64k* → T4_bpe_split_64k* | E1_bpe_64k* | 1.144 | 1.108 | **-0.0354** | [-0.0399, -0.0309] | +81 | +1014 | +8% |
| T2_unigram_raw_32k* → T4_unigram_split_32k* | E1_unigram_32k* | 1.206 | 1.133 | **-0.0728** | [-0.0786, -0.0675] | +82 | +2354 | +3% |
| T2_unigram_raw_64k* → T4_unigram_split_64k* | E1_unigram_64k* | 1.219 | 1.142 | **-0.0766** | [-0.0825, -0.0708] | +82 | +2309 | +4% |

### Fertility — never the headline (CLAUDE.md §2.1)

Primary denominator is the **raw** sentence's whitespace-word count, identical for every
arm. The secondary column divides a split arm's tokens by the *split* text's own word
count and is not comparable with any raw arm.

| corpus | matched pair | fertility raw → split (raw-word denom.) | split arm, secondary (split-word denom.) |
|---|---|---:|---:|
| samayik_test | T1_bpe_raw_32k* → T4_bpe_split_32k* | 1.653 → 1.649 | 1.420 |
| samayik_test | T1_bpe_raw_64k* → T4_bpe_split_64k* | 1.488 → 1.497 | 1.289 |
| samayik_test | T2_unigram_raw_32k* → T4_unigram_split_32k* | 1.800 → 1.746 | 1.504 |
| samayik_test | T2_unigram_raw_64k* → T4_unigram_split_64k* | 1.685 → 1.638 | 1.411 |
| samayik_test_ood | T1_bpe_raw_32k* → T4_bpe_split_32k* | 2.305 → 2.243 | 1.832 |
| samayik_test_ood | T1_bpe_raw_64k* → T4_bpe_split_64k* | 2.120 → 2.071 | 1.692 |
| samayik_test_ood | T2_unigram_raw_32k* → T4_unigram_split_32k* | 2.548 → 2.407 | 1.967 |
| samayik_test_ood | T2_unigram_raw_64k* → T4_unigram_split_64k* | 2.405 → 2.244 | 1.833 |
| itihasa_test | T1_bpe_raw_32k* → T4_bpe_split_32k* | 1.903 → 1.930 | 1.422 |
| itihasa_test | T1_bpe_raw_64k* → T4_bpe_split_64k* | 1.713 → 1.765 | 1.300 |
| itihasa_test | T2_unigram_raw_32k* → T4_unigram_split_32k* | 2.041 → 2.060 | 1.518 |
| itihasa_test | T2_unigram_raw_64k* → T4_unigram_split_64k* | 1.903 → 1.939 | 1.429 |
| flores_devtest | T1_bpe_raw_32k* → T4_bpe_split_32k* | 2.070 → 2.011 | 1.763 |
| flores_devtest | T1_bpe_raw_64k* → T4_bpe_split_64k* | 1.924 → 1.872 | 1.641 |
| flores_devtest | T2_unigram_raw_32k* → T4_unigram_split_32k* | 2.288 → 2.157 | 1.892 |
| flores_devtest | T2_unigram_raw_64k* → T4_unigram_split_64k* | 2.155 → 2.027 | 1.777 |

### Secondary variant: the splitter's raw output (`model_raw`)

Not content-preserving — it is what the model emitted before reconciliation, and it
deletes words. Reported so the reconciliation's effect is visible, never as a result.

| corpus | split arm | controlled TPP on reconciled text | on raw model output |
|---|---|---:|---:|
| samayik_test | T4_bpe_split_32k* | 1.070 | 0.860 |
| samayik_test | T4_bpe_split_64k* | 1.030 | 0.845 |
| samayik_test | T4_unigram_split_32k* | 1.099 | 0.869 |
| samayik_test | T4_unigram_split_64k* | 1.066 | 0.855 |
| samayik_test_ood | T4_bpe_split_32k* | 1.056 | 0.863 |
| samayik_test_ood | T4_bpe_split_64k* | 1.036 | 0.851 |
| samayik_test_ood | T4_unigram_split_32k* | 1.098 | 0.884 |
| samayik_test_ood | T4_unigram_split_64k* | 1.079 | 0.871 |
| itihasa_test | T4_bpe_split_32k* | 0.655 | 0.558 |
| itihasa_test | T4_bpe_split_64k* | 0.616 | 0.533 |
| itihasa_test | T4_unigram_split_32k* | 0.664 | 0.563 |
| itihasa_test | T4_unigram_split_64k* | 0.634 | 0.540 |
| flores_devtest | T4_bpe_split_32k* | 1.106 | 0.973 |
| flores_devtest | T4_bpe_split_64k* | 1.108 | 0.976 |
| flores_devtest | T4_unigram_split_32k* | 1.133 | 0.988 |
| flores_devtest | T4_unigram_split_64k* | 1.142 | 0.998 |
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

| corpus | mean units raw | mean units split | letter retention | non-letter missing (gross) | added (gross) | of | letters contained |
|---|---:|---:|---:|---:|---:|---:|---|
| samayik_test | 9.59 | 11.14 | 1.0138 | 134 | 12 | 8,704 | yes |
| samayik_test_ood | 11.82 | 14.47 | 1.0096 | 397 | 15 | 18,017 | yes |
| itihasa_test | 11.16 | 15.15 | 1.0207 | 1,806 | 3 | 44,544 | yes |
| flores_devtest | 16.77 | 19.13 | 1.0058 | 80 | 5 | 3,735 | yes |

Missing and added are **gross** — per-sentence differences summed, so a danda deleted in one
sentence and one invented in another count as two faults rather than cancelling to zero.
Letter retention above 1.0 has two components, and only one of them is the splitter doing
its job: reversing sandhi restores elided phonemes (`prARina Agatya` → `prARinaH Agatya`),
and separately the splitter **normalises orthography** — anusvāra `M` → `m`, `:` → `H`,
restored visargas — which also changes the letter count without reflecting any segmentation
work. The ratio cannot separate the two; Limitation 2 is about the second.

**Letters contained** is the third invariant: per sentence, no letter appears in the output
more often than the raw text and the model output together hold it. It holds on every corpus.
Read it as a floor, not a certificate — the budget has to be the *sum* of the two sources
because the output legitimately mixes them, which makes it loose (see `experiment.py`).

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

On these corpora the deletion column is **1–18%** of the saving wherever a saving exists, and
**≤8% on every pair whose saving exceeds 1,000 tokens**. The 18% is Sāmayik test BPE 64k,
which is also the pair with the smallest saving (179 tokens) and the Δ closest to zero — the
two facts belong together and are stated together. No Δ below is materially a deletion
artefact, but the column is not zero and is reported beside every delta rather than argued
away.

---

## Verdict — the TPP/fertility half of H3

**Sandhi-and-compound splitting buys a real but small token saving on prose, and costs tokens
on verse.** Precisely:

- On **Sāmayik test** (primary prose) all four pairs move the predicted way with CIs clear of
  0, but the effects are small: −0.005 to −0.043, and the BPE 64k pair's CI reaches −0.0005,
  i.e. it barely clears zero.
- On **Sāmayik test_ood** and **FLORES**, all eight pairs move the predicted way, Δ −0.025 to
  −0.077, every CI clear of 0, deletion cost ≤8% of the saving.
- On **Itihāsa** (verse), all four pairs move the **wrong** way (+0.006 to +0.018, CIs clear
  of 0). Splitting is not free, and on metrical text it does not pay.
- **No arm is below its matched English control on prose.** The lowest split TPP on prose is
  1.030 (`T4_bpe_split_64k*`, Sāmayik test). Sanskrit still costs more tokens per proposition
  than matched-vocabulary English on every prose corpus here.
- **Fertility moves with TPP** and never leads: down on 11 of the 12 prose/FLORES pairs, up
  on the Sāmayik-test BPE 64k pair (1.488 → 1.497) and on all four verse pairs.

H3's TPP half is **supported on prose and contradicted on verse**, at an effect size an order
of magnitude smaller than the first version of this file reported.

---

## Limitations

1. **Three rounds of correction, all found by review rather than by the pipeline.** The first
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
   constraining word choice or the splitter being weaker on Epic register. It is *not* the
   avagraha consumptions: their re-tokenised cost pushes the comparison the other way, so if
   anything they flatter `T4` there. Undiagnosed.
7. **Avagraha is licensed but not free.** It is the largest remaining non-letter difference and
   is concentrated in exactly the corpus that behaves adversely.
8. **Provisional arms.** Absolute TPP levels will move when milestone M1 lands; the matched
   deltas should be more stable, but that is an assumption.
9. **Corpus sizes differ slightly.** The raw training corpus is 117,720 lines and the split one
   117,642, from the identical 117,721 sentences. 0.07%, below the noise floor, but stated.
   Separately, the `E1_unigram_64k` control trains to 62,896 pieces rather than the requested
   64,000: the EM trainer settles below the request on this corpus once the `word\n`
   candidates are gone. 1.7% off matched size, on the English side of one pair.
10. **MorphScore and the gold-split upper bound are absent**, deferred to Experiment 04.

---

## Files

- `run.py`, `config.yaml` — the runner and its config.
- `split_corpora.py`, `split.yaml` — the splitting job (Task 3); writes `data/processed/split/`.
- `benchmark_splitter.py` — the throughput benchmark (Task 2).
- `outputs/03_sandhi_split/results.json` — every number above, with provenance.
- `outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` — the figure.
- [`results/03_sandhi_split/`](../../results/03_sandhi_split/) — the tracked copy of that
  `results.json`, its `config.yaml` and both figures, committed so the numbers can be read
  without a run (`outputs/` itself is gitignored). See
  [`results/README.md`](../../results/README.md).
