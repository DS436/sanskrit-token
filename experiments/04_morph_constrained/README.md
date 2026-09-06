# Experiment 04 — Morpheme-constrained merges (T5, T5seg, T6): MorphScore and paired TPP deltas

**Hypothesis (outline §1, RQ3/H3, verbatim):** "Sandhi splitting before subword learning
raises MorphScore and lowers fertility relative to a matched-vocabulary tokenizer trained
on raw sandhied text."

**Hypothesis (outline §1, RQ4/H4, verbatim):** "A model trained with the constrained
tokenizer reaches a reference bits-per-character with fewer training tokens than the BPE
control, in line with the 25–29% speedups reported for Hungarian and English."

## What this experiment can and cannot test

**H3's MorphScore half, in full.** Experiment 03 settled H3's TPP/fertility half and
deferred MorphScore here, because MorphScore needs gold morpheme boundaries and only DCS
has them (`docs/decisions.md`, "Experiment 03 tests the TPP/fertility half of H3;
MorphScore moves to Experiment 04"). This file reports it, on held-out DCS sentences, for
twenty arms.

**H4's TPP half only.** H4 as written is about *tokens to a reference bits-per-character*,
which needs a trained language model; that is Experiment 05. What is answerable here is the
question one layer below it: does a morpheme-constrained vocabulary spend **fewer tokens per
unit of meaning** than the unconstrained vocabulary trained on the same sentences at the
same vocabulary size? If it does not, the training-efficiency claim has to come from
somewhere other than token count. It does not (below), and **that is a result about token
count, not a refutation of H4** — a constrained tokenizer could still reach a reference BPC
in fewer tokens by making each token easier to predict. Experiment 05 decides.

**One boundary source is gold and one is not.** DCS's **segment** boundaries are
annotation. The **stem/ending** boundary inside a segment is *derived* from the segment and
its lemma by a heuristic rule (`sanskrit_tok.data.boundaries.stem_boundary`), and this file
calls it "heuristic stem" everywhere, never "gold". That distinction is why there are three
constrained arms rather than two:

| arm | constrained on | depends on the stem heuristic? |
|---|---|---|
| `T5_morphbpe_raw_{32k,64k}_dcs` | gold segment ∪ heuristic stem, in the sandhied surface | yes |
| `T5_morphbpe_rawseg_{32k,64k}_dcs` | **gold segment only**, in the sandhied surface | **no** |
| `T6_morphbpe_split_{32k,64k}_dcs` | heuristic stem, inside the gold split | yes |

`T5seg` is the clean "MorphBPE with gold boundaries" condition and, as it turns out, the
arm that carries the result.

**The headline is the paired delta, never the level.** Every arm here is trained on DCS,
and DCS has no matched English side: `E1_bpe_64k` was trained on the English half of the
*parallel* corpora. A TPP *level* against it therefore mixes the arm's cost with a corpus
mismatch, and is labelled **cross-corpus** everywhere it appears. The delta between two DCS
arms measured on the same sentences against the same English side does not: the English
total is identical on both sides and cancels (`docs/decisions.md`, "Experiment 04 headline
is the paired TPP delta between constrained and unconstrained DCS arms; MorphScore is the
mechanism check").

**Fertility never leads** (CLAUDE.md §2.1). It is computed, reported, and — as in Experiment
03 — reported twice: primary over the **raw** sentence's word count, the same denominator
for every arm, and secondary over the split text's own words for split arms only.

**Run:** `uv run python experiments/04_morph_constrained/run.py` — **3–4 min**, no
network beyond the cached off-the-shelf tokenizers, at commit `eb53842` with a clean tree
(`results.json` records `git_dirty: false`).

---

## Status

| Task | State |
|---|---|
| 1. DCS ingestion with aligned gold boundaries | done |
| 2. Token spans and MorphScore | done |
| 3. MorphBPE constraint and the fourteen `_dcs` arms | done |
| 4. Runner, `results.json`, figure, this file | done |
| 5. Review fixes: leakage filter, newline-stripped retrain, stem rule, `T5seg` | done |

Every number below is from `outputs/04_morph_constrained/results.json`, run 2026-09-05
at commit `eb53842`, clean tree (a re-run of the `ca25390` run at the pair-label fix: every
one of its 9,366 metric leaves is bit-identical, only the figure's tick labels and
`tpp_delta`'s `label` fields changed). No arm was unavailable (`unavailable_arms: {}`);
`T0_gemma3` resolved to `unsloth/gemma-3-4b-it`, the substitution already recorded in
`docs/decisions.md`.

---

## What changed since the first version of this file

Four review findings, all of which invalidated every trained arm, so all four landed
together with one retrain (`docs/decisions.md`, 2026-09-05, three entries and their two
CORRECTIONs).

1. **Near-duplicate evaluation text was in the DCS training corpus.** The Mahābhārata and
   Rāmāyaṇa are DCS texts *and* the Itihāsa parallel corpus, segmented into sentences
   differently by each, so 19% of Itihāsa test verses sat verbatim — as letter strings —
   inside a DCS training sentence that hashed to something else. The ingestion now drops a
   training sentence when any 24-letter window of its letter-normalised form occurs in any
   evaluation sentence. **34,705 sentences dropped** (train 720,510 → 685,805). The
   per-source counts overlap — one training sentence can match several evaluation sets, and
   is dropped once — so they sum to 35,659 against that deduplicated total of 34,705:
   20,611 matched Itihāsa test, 11,460 Itihāsa dev, 3,581 the DCS held-out split, 5 Sāmayik
   dev, 2 Sāmayik test-OOD, 0 FLORES devtest, 0 Sāmayik test. Residual check: of 400
   randomly sampled Itihāsa test lines, **0** now occur as letter-substrings of
   `tok_train_dcs_raw.txt`; the review measured 19% of Itihāsa test verses occurring
   verbatim in the pre-filter corpus. The filter has a floor, and it is not symmetric: a
   training sentence with fewer than 24 letters has no window to offer and is matched only
   by exact letters-only equality, so 387 of the 51,098 such lines in
   `tok_train_dcs_raw.txt` (0.06% of its 652,889 written lines) still occur verbatim inside
   an Itihāsa test verse — the 0/400 residual check tests the opposite direction, whole
   evaluation verses inside training sentences, and says nothing about this one.
2. **Line breaks were reaching the pre-tokenizer.** 5–15% of every BPE vocabulary was
   `word\n` entries that no inference-time string can produce, unevenly distributed across
   arms. Every arm in the project is retrained; all 26 now carry **0** newline-bearing
   entries.
3. **The stem rule cut inside the lemma half the time.** The LCP rule made 120,459 cuts on
   the held-out split, 50.4% of them strictly inside the lemma and only 29.1 pp of those
   explicable as a fused stem-final vowel. The rule is now part-of-speech gated and refuses
   to cut inside the lemma's consonantal body: 90,878 cuts, **36.5% of them flagged fused
   cuts** one character short of the lemma's end. Its 0.0000 not-fused-inside-lemma
   fraction is definitional rather than a measurement — `stem_boundary` returns a cut only
   at `len(lemma)` or at a flagged fused position, so `n_inside_lemma == n_fused` by
   construction — and the 36.5% is the informative figure, against the LCP rule's 21.4% of
   cuts falling strictly inside the lemma body, which *was* a measurement.
4. **`T5seg` was missing.** The old `T5`/`T6` arms were constrained partly by that
   heuristic, so "MorphBPE with gold boundaries" had never actually been run. It has now,
   and it is the arm that changes the verdict.

---

## The arms

Fourteen arms trained on the DCS training split (685,805 sentences), matched at 32k and 64k:

| Key | Text trained on | Constrained? |
|---|---|---|
| `T1_bpe_raw_{32k,64k}_dcs` | sandhied surface | no |
| `T2_unigram_raw_{32k,64k}_dcs` | sandhied surface | no |
| `T4_bpe_split_{32k,64k}_oracle_dcs` | DCS **gold** segmentation (`oracle`) | no |
| `T4_unigram_split_{32k,64k}_oracle_dcs` | DCS **gold** segmentation (`oracle`) | no |
| `T5_morphbpe_raw_{32k,64k}_dcs` | sandhied surface, marked at segment + heuristic stem | **yes** |
| `T5_morphbpe_rawseg_{32k,64k}_dcs` | sandhied surface, marked at **gold segments only** | **yes** |
| `T6_morphbpe_split_{32k,64k}_dcs` | gold split, marked at heuristic stem (**proposed**) | **yes** |

Every arm reached its full requested vocabulary. Plus, for context and marked as such in
every table: **provisional** arms `T1_bpe_raw_64k*`, `T2_unigram_raw_64k*`,
`T4_bpe_split_64k*` (trained on the *parallel* corpora, Experiments 02–03), and **existing
practice** `T0_o200k`, `T0_gemma3`, `T3_sarvam`.

## The evaluation sets

| Set | n | Note |
|---|---|---|
| DCS held-out | **30,150** sentences over 13 texts | 160,178 surface words, 214,234 gold segments |
| — human-verified subset | **6,952** (23.06%) | no `UnsandhiedReconstructed=True` token anywhere in the sentence |
| Sāmayik test (prose) | 2,417 pairs | primary parallel corpus |
| Sāmayik test-OOD (prose) | 4,047 pairs | |
| Itihāsa test (verse) | 11,721 pairs | secondary; meter is a confound |
| FLORES-200 devtest | 1,012 pairs | parity anchor |

**Two alignment rates, over two populations.** `align_segments` locates a word's gold
segment boundaries in the sandhied surface, and fails on a small minority. On the **DCS
held-out split** — the population every MorphScore number here is computed over — 156,475
of 160,178 surface words align, **97.69%**. Over the **whole ingested corpus** (train and
held-out together, 4,087,443 words) the rate is **97.47%**
(`data/processed/dcs/manifest.json`). The two are different denominators and neither is a
rounding of the other; the 97.69% is the one that bounds what MorphScore measures.

**MorphScore ran over the whole held-out split** — all 30,150 sentences, no subsampling
(`results.json`'s `dcs_heldout.sample` is `null`).

**No leakage, in two layers.** All 30,150 held-out sentences hash into
`data/exclusion_hashes.txt` (`n_missing: 0`), which is what kept them out of every `_dcs`
arm's training corpus; all four parallel corpora are fully present in both the Sanskrit and
the English exclusion lists (`n_missing: 0` in all eight checks). On top of that, the
shingle filter dropped 34,705 near-duplicate training sentences (above).

**Spans coverage.** Every one of the twenty arms tiles both text forms exactly on the
first 500 held-out sentences (`spans_cover_text`, 500/500 passed, 0 failed, on `text_slp1`
and on `oracle_split_slp1`). No arm's MorphScore was withheld.

---

## Result 1 — the constraint works, out of sample

Each arm re-tokenizes the **held-out** marked text it never saw, and a token whose span
strictly contains a boundary is a violation. **`violations per boundary` is the mechanism
number**: violating tokens divided by the boundaries in the audited text — the same
denominator for both arms of a pair. Its numerator is offending *tokens*, so a token
swallowing two boundaries counts once, which makes it a close lower bound on the fraction of
boundaries crossed rather than that fraction exactly. The per-token rate is shown beside it
because it is what `assert_no_cross_boundary_merges` reports natively, and because it is the
*flattering* denominator: a constrained arm emits more tokens for the same text.

**Two denominators, because "did it keep its constraint" and "did the constraint
generalise" are different questions.** The `segment ∪ stem` set (135,718 boundaries) is what
`T5` was constrained on; the `segment only` set (52,988) is what `T5seg` was constrained on
and is a subset of it. Every raw arm is measured against both.

### Gold segment boundaries only (52,988 boundaries) — the annotation-based number

| vocab | `T1_dcs` | `T5_dcs` | `T5seg_dcs` |
|---|---|---|---|
| 32k | 0.4283 | 0.2728 (−36.3%) | **0.2508 (−41.4%)** |
| 64k | 0.4732 | 0.2876 (−39.2%) | **0.2662 (−43.7%)** |

### Segment ∪ heuristic stem (135,718 boundaries) — what `T5` was constrained on

| vocab | `T1_dcs` | `T5_dcs` | `T5seg_dcs` |
|---|---|---|---|
| 32k | 0.7069 | **0.4737 (−33.0%)** | 0.6572 (−7.0%) |
| 64k | 0.7224 | **0.4871 (−32.6%)** | 0.6731 (−6.8%) |

`T5seg`'s small reduction here is the honest reading of a question it was not asked: half
this denominator is stem boundaries it was never constrained on, and it respects them barely
better than the unconstrained control does. The constraint does not generalise from
segments to stems.

### Heuristic stem boundaries inside the gold split (90,878 boundaries)

| vocab | `T4_oracle_dcs` | `T6_dcs` |
|---|---|---|
| 32k | 0.9637 | **0.5371 (−44.3%)** |
| 64k | 0.9806 | **0.5514 (−43.8%)** |

Counts behind all three tables (30,150 sentences, stride 1, so every sentence audited):

| arm | boundary set | violating tokens | boundaries | per boundary | tokens | per token |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | segment ∪ stem | 95,935 | 135,718 | 0.7069 | 281,661 | 0.3406 |
| `T1_bpe_raw_64k_dcs` | segment ∪ stem | 98,047 | 135,718 | 0.7224 | 255,873 | 0.3832 |
| `T5_morphbpe_raw_32k_dcs` | segment ∪ stem | 64,295 | 135,718 | **0.4737** | 336,232 | 0.1912 |
| `T5_morphbpe_raw_64k_dcs` | segment ∪ stem | 66,107 | 135,718 | **0.4871** | 316,049 | 0.2092 |
| `T5_morphbpe_rawseg_32k_dcs` | segment ∪ stem | 89,194 | 135,718 | 0.6572 | 293,390 | 0.3040 |
| `T5_morphbpe_rawseg_64k_dcs` | segment ∪ stem | 91,346 | 135,718 | 0.6731 | 269,037 | 0.3395 |
| `T1_bpe_raw_32k_dcs` | segment only | 22,696 | 52,988 | 0.4283 | 281,661 | 0.0806 |
| `T1_bpe_raw_64k_dcs` | segment only | 25,076 | 52,988 | 0.4732 | 255,873 | 0.0980 |
| `T5_morphbpe_raw_32k_dcs` | segment only | 14,457 | 52,988 | 0.2728 | 336,232 | 0.0430 |
| `T5_morphbpe_raw_64k_dcs` | segment only | 15,240 | 52,988 | 0.2876 | 316,049 | 0.0482 |
| `T5_morphbpe_rawseg_32k_dcs` | segment only | 13,290 | 52,988 | **0.2508** | 293,390 | 0.0453 |
| `T5_morphbpe_rawseg_64k_dcs` | segment only | 14,106 | 52,988 | **0.2662** | 269,037 | 0.0524 |
| `T4_bpe_split_32k_oracle_dcs` | stem in split | 87,578 | 90,878 | 0.9637 | 264,020 | 0.3317 |
| `T4_bpe_split_64k_oracle_dcs` | stem in split | 89,119 | 90,878 | 0.9806 | 244,408 | 0.3646 |
| `T6_morphbpe_split_32k_dcs` | stem in split | 48,808 | 90,878 | **0.5371** | 332,018 | 0.1470 |
| `T6_morphbpe_split_64k_dcs` | stem in split | 50,113 | 90,878 | **0.5514** | 317,515 | 0.1578 |

**The residue is expected and is the point.** A BPE merge rule is a global character pair:
the marker stops the *trainer* counting a pair that straddles a boundary, but a rule learned
elsewhere still applies inside a word at inference, where the text has no markers. Task 3
measured this in sample (`docs/decisions.md`, "the boundary-marker constraint halves
cross-boundary tokens but cannot reach zero"); this is the same measurement on sentences no
arm saw, and the reduction holds out of sample at the same magnitude.

---

## Result 2 — MorphScore (the H3 half)

`value` is boundary F1, exclusions per Arnett & Bergen: a word emitted as one token and a
word with no boundary inside it are both excluded and counted. Raw arms are scored on the
sandhied surface's whitespace words against **gold segment** boundaries (primary); split
arms are scored on each gold segment against the **heuristic stem** boundary inside it. Raw
arms additionally carry the heuristic stem granularity, projected into the surface where the
word's alignment succeeded. **Precision and recall are reported beside every F1**, because
the constraint moves them very differently.

### Human-verified subset (6,952 sentences) — the number to read

| arm | granularity | F1 exact | P | R | F1 ±1 | P ±1 | R ±1 | units scored |
|---|---|---|---|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | gold segment | 0.4469 | 0.3511 | 0.6147 | 0.5733 | 0.4503 | 0.7885 | 4,573 |
| `T1_bpe_raw_64k_dcs` | gold segment | 0.4649 | 0.3809 | 0.5965 | 0.5951 | 0.4875 | 0.7635 | 4,241 |
| `T2_unigram_raw_32k_dcs` | gold segment | 0.3846 | 0.3026 | 0.5275 | 0.5697 | 0.4483 | 0.7814 | 4,894 |
| `T2_unigram_raw_64k_dcs` | gold segment | 0.3724 | 0.3074 | 0.4721 | 0.5677 | 0.4687 | 0.7198 | 4,844 |
| `T5_morphbpe_raw_32k_dcs` | gold segment | 0.4858 | 0.3680 | 0.7147 | 0.6397 | 0.4845 | 0.9411 | 5,060 |
| `T5_morphbpe_raw_64k_dcs` | gold segment | 0.5004 | 0.3934 | 0.6875 | 0.6613 | 0.5198 | 0.9085 | 5,043 |
| `T5_morphbpe_rawseg_32k_dcs` | gold segment | **0.5448** | 0.4225 | 0.7667 | 0.6681 | 0.5181 | 0.9402 | 5,056 |
| `T5_morphbpe_rawseg_64k_dcs` | gold segment | **0.5792** | 0.4705 | 0.7530 | 0.7098 | 0.5766 | 0.9228 | 5,035 |
| `T1_bpe_raw_64k*` | gold segment | 0.4282 | 0.3272 | 0.6195 | 0.5447 | 0.4162 | 0.7881 | 4,755 |
| `T2_unigram_raw_64k*` | gold segment | 0.3777 | 0.2811 | 0.5756 | 0.5409 | 0.4025 | 0.8243 | 4,864 |
| `T0_o200k` | gold segment | 0.1302 | 0.0833 | 0.2978 | 0.4276 | 0.2736 | 0.9783 | 5,111 |
| `T0_gemma3` | gold segment | 0.1214 | 0.0780 | 0.2741 | 0.4289 | 0.2754 | 0.9680 | 5,091 |
| `T3_sarvam` | gold segment | 0.1472 | 0.0891 | 0.4225 | 0.3479 | 0.2106 | 0.9985 | 5,116 |
| `T1_bpe_raw_32k_dcs` | heuristic stem | 0.1065 | 0.0879 | 0.1352 | 0.4522 | 0.3731 | 0.5739 | 10,035 |
| `T1_bpe_raw_64k_dcs` | heuristic stem | 0.0961 | 0.0806 | 0.1190 | 0.4336 | 0.3636 | 0.5369 | 7,912 |
| `T2_unigram_raw_32k_dcs` | heuristic stem | 0.3906 | 0.3310 | 0.4764 | 0.5769 | 0.4889 | 0.7035 | 14,699 |
| `T2_unigram_raw_64k_dcs` | heuristic stem | 0.3982 | 0.3497 | 0.4622 | 0.5776 | 0.5073 | 0.6706 | 14,121 |
| `T5_morphbpe_raw_32k_dcs` | heuristic stem | 0.3222 | 0.2681 | 0.4035 | 0.6106 | 0.5081 | 0.7648 | 17,218 |
| `T5_morphbpe_raw_64k_dcs` | heuristic stem | 0.3314 | 0.2841 | 0.3977 | 0.6208 | 0.5322 | 0.7449 | 16,814 |
| `T5_morphbpe_rawseg_32k_dcs` | heuristic stem | 0.1006 | 0.0818 | 0.1305 | 0.4320 | 0.3514 | 0.5604 | 9,993 |
| `T5_morphbpe_rawseg_64k_dcs` | heuristic stem | 0.0892 | 0.0739 | 0.1125 | 0.4044 | 0.3352 | 0.5098 | 7,982 |
| `T4_bpe_split_32k_oracle_dcs` | heuristic stem | 0.1376 | 0.1220 | 0.1579 | 0.5092 | 0.4514 | 0.5840 | 6,702 |
| `T4_bpe_split_64k_oracle_dcs` | heuristic stem | 0.1247 | 0.1132 | 0.1388 | 0.5028 | 0.4565 | 0.5595 | 4,309 |
| `T4_unigram_split_32k_oracle_dcs` | heuristic stem | 0.4921 | 0.4514 | 0.5408 | 0.7111 | 0.6523 | 0.7814 | 15,740 |
| `T4_unigram_split_64k_oracle_dcs` | heuristic stem | 0.5045 | 0.4778 | 0.5344 | 0.7175 | 0.6795 | 0.7600 | 15,311 |
| `T6_morphbpe_split_32k_dcs` | heuristic stem | 0.4159 | 0.3773 | 0.4633 | 0.7006 | 0.6356 | 0.7804 | 18,701 |
| `T6_morphbpe_split_64k_dcs` | heuristic stem | 0.4302 | 0.4017 | 0.4630 | 0.7183 | 0.6707 | 0.7731 | 18,174 |
| `T1_bpe_raw_64k*` | heuristic stem | 0.1605 | 0.1298 | 0.2101 | 0.4855 | 0.3927 | 0.6357 | 13,353 |
| `T2_unigram_raw_64k*` | heuristic stem | 0.3424 | 0.2717 | 0.4627 | 0.5604 | 0.4448 | 0.7573 | 15,056 |
| `T4_bpe_split_64k*` | heuristic stem | 0.1599 | 0.1386 | 0.1890 | 0.5052 | 0.4378 | 0.5970 | 9,751 |
| `T0_o200k` | heuristic stem | 0.2784 | 0.1873 | 0.5422 | 0.4547 | 0.3059 | 0.8855 | 17,779 |
| `T0_gemma3` | heuristic stem | 0.2762 | 0.1869 | 0.5287 | 0.4506 | 0.3050 | 0.8628 | 17,693 |
| `T3_sarvam` | heuristic stem | 0.2586 | 0.1626 | 0.6320 | 0.3907 | 0.2456 | 0.9549 | 17,971 |

### All held-out sentences (30,150)

| arm | granularity | F1 exact | P | R | F1 ±1 | units scored |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | gold segment | 0.4682 | 0.3806 | 0.6082 | 0.6494 | 35,003 |
| `T1_bpe_raw_64k_dcs` | gold segment | 0.4829 | 0.4128 | 0.5816 | 0.6744 | 32,743 |
| `T2_unigram_raw_32k_dcs` | gold segment | 0.4680 | 0.3768 | 0.6176 | 0.6164 | 37,040 |
| `T2_unigram_raw_64k_dcs` | gold segment | 0.4575 | 0.3889 | 0.5555 | 0.6123 | 36,661 |
| `T5_morphbpe_raw_32k_dcs` | gold segment | 0.5047 | 0.3858 | 0.7296 | 0.6638 | 38,682 |
| `T5_morphbpe_raw_64k_dcs` | gold segment | 0.5286 | 0.4195 | 0.7146 | 0.6982 | 38,595 |
| `T5_morphbpe_rawseg_32k_dcs` | gold segment | **0.5568** | 0.4420 | 0.7521 | 0.7124 | 38,654 |
| `T5_morphbpe_rawseg_64k_dcs` | gold segment | **0.5888** | 0.4898 | 0.7381 | 0.7583 | 38,532 |
| `T1_bpe_raw_64k*` | gold segment | 0.4328 | 0.3457 | 0.5783 | 0.6023 | 34,482 |
| `T2_unigram_raw_64k*` | gold segment | 0.4327 | 0.3411 | 0.5916 | 0.5912 | 36,075 |
| `T0_o200k` | gold segment | 0.1425 | 0.0917 | 0.3195 | 0.4298 | 38,893 |
| `T0_gemma3` | gold segment | 0.1500 | 0.0970 | 0.3307 | 0.4353 | 38,852 |
| `T3_sarvam` | gold segment | 0.1532 | 0.0929 | 0.4353 | 0.3504 | 38,899 |
| `T1_bpe_raw_32k_dcs` | heuristic stem | 0.0959 | 0.0758 | 0.1305 | 0.3598 | 46,837 |
| `T1_bpe_raw_64k_dcs` | heuristic stem | 0.0917 | 0.0744 | 0.1194 | 0.3355 | 39,309 |
| `T2_unigram_raw_32k_dcs` | heuristic stem | 0.3774 | 0.3068 | 0.4901 | 0.5296 | 66,226 |
| `T2_unigram_raw_64k_dcs` | heuristic stem | 0.3835 | 0.3239 | 0.4699 | 0.5398 | 64,362 |
| `T5_morphbpe_raw_32k_dcs` | heuristic stem | 0.3002 | 0.2358 | 0.4130 | 0.5221 | 75,127 |
| `T5_morphbpe_raw_64k_dcs` | heuristic stem | 0.3109 | 0.2513 | 0.4075 | 0.5323 | 73,376 |
| `T5_morphbpe_rawseg_32k_dcs` | heuristic stem | 0.0845 | 0.0654 | 0.1193 | 0.3433 | 47,792 |
| `T5_morphbpe_rawseg_64k_dcs` | heuristic stem | 0.0761 | 0.0602 | 0.1032 | 0.3063 | 41,002 |
| `T4_bpe_split_32k_oracle_dcs` | heuristic stem | 0.1082 | 0.0949 | 0.1257 | 0.4421 | 26,245 |
| `T4_bpe_split_64k_oracle_dcs` | heuristic stem | 0.0935 | 0.0837 | 0.1060 | 0.4146 | 16,591 |
| `T4_unigram_split_32k_oracle_dcs` | heuristic stem | 0.5432 | 0.5028 | 0.5906 | 0.7379 | 69,630 |
| `T4_unigram_split_64k_oracle_dcs` | heuristic stem | 0.5674 | 0.5411 | 0.5964 | 0.7544 | 67,932 |
| `T6_morphbpe_split_32k_dcs` | heuristic stem | 0.4370 | 0.3959 | 0.4877 | 0.6676 | 86,259 |
| `T6_morphbpe_split_64k_dcs` | heuristic stem | 0.4523 | 0.4214 | 0.4880 | 0.6846 | 83,528 |
| `T1_bpe_raw_64k*` | heuristic stem | 0.1253 | 0.0985 | 0.1725 | 0.3951 | 53,245 |
| `T2_unigram_raw_64k*` | heuristic stem | 0.3479 | 0.2727 | 0.4804 | 0.5243 | 62,656 |
| `T4_bpe_split_64k*` | heuristic stem | 0.1451 | 0.1269 | 0.1692 | 0.4668 | 35,890 |
| `T0_o200k` | heuristic stem | 0.2493 | 0.1626 | 0.5341 | 0.4104 | 78,846 |
| `T0_gemma3` | heuristic stem | 0.2500 | 0.1637 | 0.5291 | 0.4120 | 78,627 |
| `T3_sarvam` | heuristic stem | 0.2310 | 0.1416 | 0.6275 | 0.3474 | 79,184 |

### The paired deltas, with sentence-level bootstrap intervals

Only three contrasts score the **same population of units** and so are comparisons between
tokenizers rather than between corpora. Each carries a paired bootstrap over sentences
(1,000 resamples, seed 0): one set of sentence indices is drawn and both arms are re-pooled
over it, so the sentence-to-sentence variation the two arms share cancels.

**Human-verified subset (6,952 sentences)**

| contrast | granularity | vocab | ΔF1 exact | 95% CI | ΔF1 ±1 | 95% CI |
|---|---|---|---|---|---|---|
| `T5_dcs` − `T1_dcs` | gold segment | 32k | **+0.0389** | [+0.0302, +0.0482] | +0.0664 | [+0.0589, +0.0739] |
| `T5_dcs` − `T1_dcs` | gold segment | 64k | **+0.0355** | [+0.0250, +0.0458] | +0.0662 | [+0.0576, +0.0752] |
| `T5seg_dcs` − `T1_dcs` | gold segment | 32k | **+0.0979** | [+0.0908, +0.1047] | +0.0948 | [+0.0881, +0.1014] |
| `T5seg_dcs` − `T1_dcs` | gold segment | 64k | **+0.1143** | [+0.1063, +0.1224] | +0.1147 | [+0.1069, +0.1223] |
| `T6_dcs` − `T4_oracle_dcs` | heuristic stem | 32k | **+0.2783** | [+0.2686, +0.2878] | +0.1914 | [+0.1816, +0.2021] |
| `T6_dcs` − `T4_oracle_dcs` | heuristic stem | 64k | **+0.3055** | [+0.2938, +0.3169] | +0.2155 | [+0.2015, +0.2292] |

**All 30,150 held-out sentences**

| contrast | granularity | vocab | ΔF1 exact | 95% CI | ΔF1 ±1 | 95% CI |
|---|---|---|---|---|---|---|
| `T5_dcs` − `T1_dcs` | gold segment | 32k | +0.0365 | [+0.0335, +0.0397] | +0.0143 | [+0.0119, +0.0169] |
| `T5_dcs` − `T1_dcs` | gold segment | 64k | +0.0458 | [+0.0424, +0.0494] | +0.0238 | [+0.0211, +0.0267] |
| `T5seg_dcs` − `T1_dcs` | gold segment | 32k | +0.0886 | [+0.0860, +0.0910] | +0.0629 | [+0.0609, +0.0651] |
| `T5seg_dcs` − `T1_dcs` | gold segment | 64k | +0.1059 | [+0.1029, +0.1090] | +0.0839 | [+0.0815, +0.0865] |
| `T6_dcs` − `T4_oracle_dcs` | heuristic stem | 32k | +0.3289 | [+0.3247, +0.3332] | +0.2254 | [+0.2202, +0.2310] |
| `T6_dcs` − `T4_oracle_dcs` | heuristic stem | 64k | +0.3587 | [+0.3538, +0.3636] | +0.2700 | [+0.2633, +0.2769] |

Every interval excludes 0 in both populations, at both sizes, under both tolerances.

**How the constraint moves the score: recall, mostly.** For `T5` − `T1_dcs` at 64k on the
human-verified subset, precision goes 0.3809 → 0.3934 (+0.013) while recall goes
0.5965 → 0.6875 (+0.091): the constraint **raises boundary recall at essentially unchanged
precision** — it finds more of the real boundaries without cutting more indiscriminately.
`T5seg` is the arm where precision moves too (0.3809 → 0.4705, +0.090, alongside recall
0.5965 → 0.7530), which is what makes it the best-aligned arm in the experiment rather than
merely a more eager one. `T6` − `T4_oracle` at 64k raises both (P 0.1132 → 0.4017,
R 0.1388 → 0.4630), but against the heuristic stem boundary its own training text was marked
with, so read it as "the constraint was applied", not as an independent validation.

**`T2` (Unigram) trails BPE on gold segment boundaries and beats it by a wide margin on
heuristic stem boundaries**, and the gap between the two populations is itself the finding.
On **gold segment** F1, `T2_unigram_raw_32k_dcs` scores 0.4680 over all 30,150 sentences
against `T1_bpe_raw_32k_dcs`'s 0.4682 — a dead heat — and 0.4575 against 0.4829 at 64k, a
loss; on the **human-verified** subset it loses clearly at both sizes (0.3846 vs 0.4469 at
32k, 0.3724 vs 0.4649 at 64k), and the gap widens by 0.06–0.09 F1 when the machine-generated
sentences are removed. On **heuristic stem** boundaries it is the reverse and the margin is
enormous (0.3906 vs 0.1065 at 32k human-verified). Read that as a statement about the
boundary sets, not about the learners: Unigram's pieces land near stem/ending splits that
BPE's merges swallow, and the all-sentences population is 68% segmentation produced by DCS's
own segmenter, which Unigram's segmentation resembles more than BPE's does — which is
precisely why the human-verified subset is the number to read. **No `T2` number is a
controlled comparison with any `T5`/`T6` arm anyway** — Unigram has no merges, so the
constraint cannot be applied to it and the pairing rules refuse it.

**"Splitting raises MorphScore" is *not* cleanly testable here**, and no number in this file
should be read as testing it. A raw arm is scored on surface *words* and a split arm on gold
*segments*: they are different units of different lengths, so `T4_oracle` − `T1_dcs` compares
two populations, not two tokenizers. The gold segment granularity, which would be the fair
one, does not exist for a split arm at all: in the oracle split the segment boundaries *are*
whitespace.

---

## Result 3 — the paired TPP deltas (the H4 half)

`Δ = TPP(a) − TPP(b)`, both arms on the same sentences against the same English side
(`E1_bpe_64k`, cross-corpus), 95% paired bootstrap over 1,000 resamples, seed 0. **Negative
is the result H4's TPP half predicts.** Prose first.

### Sāmayik test (prose, 2,417 pairs) — the primary corpus

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) | tokens saved | deletion cost |
|---|---|---|---|---|---|---|
| `T5_morphbpe_raw_32k_dcs` − `T1_bpe_raw_32k_dcs` | **+0.0693** | [+0.0636, +0.0747] | 1.805 | 1.736 | −2,334 | n/a (same text) |
| `T5_morphbpe_raw_64k_dcs` − `T1_bpe_raw_64k_dcs` | **+0.0837** | [+0.0782, +0.0896] | 1.725 | 1.642 | −2,820 | n/a (same text) |
| `T5_morphbpe_rawseg_32k_dcs` − `T1_bpe_raw_32k_dcs` | **+0.0040** | [+0.0009, +0.0071] | 1.740 | 1.736 | −135 | n/a (same text) |
| `T5_morphbpe_rawseg_64k_dcs` − `T1_bpe_raw_64k_dcs` | **+0.0022** | **[−0.0010, +0.0054]** | 1.644 | 1.642 | −73 | n/a (same text) |
| `T4_bpe_split_32k_oracle_dcs` − `T1_bpe_raw_32k_dcs` | **−0.0267** | [−0.0325, −0.0207] | 1.709 | 1.736 | +901 | 88 tok (9.8% of saving) |
| `T4_bpe_split_64k_oracle_dcs` − `T1_bpe_raw_64k_dcs` | **−0.0199** | [−0.0253, −0.0142] | 1.622 | 1.642 | +670 | 83 tok (12.4% of saving) |
| `T6_morphbpe_split_32k_dcs` − `T4_bpe_split_32k_oracle_dcs` | **+0.0728** | [+0.0669, +0.0787] | 1.782 | 1.709 | −2,453 | n/a (same text) |
| `T6_morphbpe_split_64k_dcs` − `T4_bpe_split_64k_oracle_dcs` | **+0.0959** | [+0.0902, +0.1013] | 1.718 | 1.622 | −3,233 | n/a (same text) |
| `T6_morphbpe_split_32k_dcs` − `T1_bpe_raw_32k_dcs` | **+0.0461** | [+0.0394, +0.0524] | 1.782 | 1.736 | −1,552 | 88 tok (saving is negative) |
| `T6_morphbpe_split_64k_dcs` − `T1_bpe_raw_64k_dcs` | **+0.0760** | [+0.0698, +0.0826] | 1.718 | 1.642 | −2,563 | 83 tok (saving is negative) |

**`T5seg` at 64k is the only constrained delta in this experiment whose CI includes 0.** Its
point estimate is +0.0022 on a level of 1.642 — a **0.13%** token cost — against `T5`'s
+5.1% and `T6`'s +4.6% at the same size on the same sentences.

### Sāmayik test-OOD (prose, 4,047 pairs)

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) |
|---|---|---|---|---|
| `T5` − `T1` 32k | +0.0885 | [+0.0857, +0.0914] | 1.384 | 1.295 |
| `T5` − `T1` 64k | +0.0931 | [+0.0902, +0.0960] | 1.309 | 1.216 |
| `T5seg` − `T1` 32k | **+0.0172** | [+0.0155, +0.0191] | 1.312 | 1.295 |
| `T5seg` − `T1` 64k | **+0.0134** | [+0.0116, +0.0151] | 1.229 | 1.216 |
| `T4_oracle` − `T1` 32k | **−0.0115** | [−0.0145, −0.0081] | 1.284 | 1.295 |
| `T4_oracle` − `T1` 64k | **−0.0071** | [−0.0103, −0.0039] | 1.209 | 1.216 |
| `T6` − `T4_oracle` 32k | +0.0885 | [+0.0855, +0.0916] | 1.372 | 1.284 |
| `T6` − `T4_oracle` 64k | +0.0987 | [+0.0956, +0.1017] | 1.307 | 1.209 |
| `T6` − `T1` 32k | +0.0770 | [+0.0734, +0.0805] | 1.372 | 1.295 |
| `T6` − `T1` 64k | +0.0916 | [+0.0880, +0.0951] | 1.307 | 1.216 |

Deletion cost on the two split-vs-raw pairs: 290 tokens (26.4% of a 1,099-token saving) at
32k and 268 (39.2% of 683) at 64k — a quarter to two-fifths of the gold-splitting saving on
this corpus is characters the reconciled text no longer contains, not boundaries the split
found.

### Itihāsa test (verse, 11,721 pairs) — meter is a confound

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) |
|---|---|---|---|---|
| `T5` − `T1` 32k | +0.1400 | [+0.1384, +0.1415] | 0.885 | 0.745 |
| `T5` − `T1` 64k | +0.1547 | [+0.1530, +0.1563] | 0.837 | 0.683 |
| `T5seg` − `T1` 32k | +0.0444 | [+0.0435, +0.0452] | 0.790 | 0.745 |
| `T5seg` − `T1` 64k | +0.0482 | [+0.0473, +0.0492] | 0.731 | 0.683 |
| `T4_oracle` − `T1` 32k | +0.0100 | [+0.0080, +0.0123] | 0.756 | 0.745 |
| `T4_oracle` − `T1` 64k | +0.0239 | [+0.0220, +0.0259] | 0.706 | 0.683 |
| `T6` − `T4_oracle` 32k | +0.1362 | [+0.1344, +0.1381] | 0.892 | 0.756 |
| `T6` − `T4_oracle` 64k | +0.1470 | [+0.1452, +0.1489] | 0.853 | 0.706 |
| `T6` − `T1` 32k | +0.1462 | [+0.1444, +0.1482] | 0.892 | 0.745 |
| `T6` − `T1` 64k | **+0.1709** | [+0.1689, +0.1728] | 0.853 | 0.683 |

**Every interval in this table now excludes 0**, including `T4_oracle` − `T1` at 32k, which
straddled it before the shingle filter removed 20,611 near-duplicate Itihāsa test verses
from the training corpus. That is the one conclusion in this experiment that the leakage fix
changed, and it changed it in the direction that makes gold splitting look *worse* on verse
— consistent with the contamination having flattered the arms that memorised whole words.

### FLORES-200 devtest (1,012 pairs)

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) |
|---|---|---|---|---|
| `T5` − `T1` 32k | +0.1017 | [+0.0962, +0.1078] | 1.445 | 1.344 |
| `T5` − `T1` 64k | +0.1002 | [+0.0949, +0.1057] | 1.364 | 1.264 |
| `T5seg` − `T1` 32k | +0.0111 | [+0.0084, +0.0140] | 1.355 | 1.344 |
| `T5seg` − `T1` 64k | +0.0065 | [+0.0036, +0.0096] | 1.270 | 1.264 |
| `T4_oracle` − `T1` 32k | **−0.0338** | [−0.0389, −0.0285] | 1.310 | 1.344 |
| `T4_oracle` − `T1` 64k | **−0.0339** | [−0.0391, −0.0289] | 1.230 | 1.264 |
| `T6` − `T4_oracle` 32k | +0.0981 | [+0.0924, +0.1038] | 1.408 | 1.310 |
| `T6` − `T4_oracle` 64k | +0.1134 | [+0.1073, +0.1196] | 1.343 | 1.230 |
| `T6` − `T1` 32k | +0.0644 | [+0.0582, +0.0703] | 1.408 | 1.344 |
| `T6` − `T1` 64k | +0.0794 | [+0.0736, +0.0855] | 1.343 | 1.264 |

Deletion cost: 87 tokens (9.0% of a 968-token saving) at 32k and 88 (9.0% of 973) at 64k.

### In domain: the same contrasts on held-out DCS

The TPP deltas above are measured out of domain. Held-out DCS has no English side, so it
carries no TPP — but it carries token counts, and they are the same contrasts on text of
the kind these arms were trained on.

| arm | text form | bytes/token | tokens | tokens / raw word |
|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | sandhied | 4.670 | 281,661 | 1.758 |
| `T1_bpe_raw_64k_dcs` | sandhied | 5.141 | 255,873 | 1.597 |
| `T2_unigram_raw_32k_dcs` | sandhied | 4.074 | 322,870 | 2.016 |
| `T2_unigram_raw_64k_dcs` | sandhied | 4.344 | 302,808 | 1.890 |
| `T4_bpe_split_32k_oracle_dcs` | oracle split | 5.295 | 264,020 | 1.648 |
| `T4_bpe_split_64k_oracle_dcs` | oracle split | **5.720** | **244,408** | **1.526** |
| `T4_unigram_split_32k_oracle_dcs` | oracle split | 4.204 | 332,519 | 2.076 |
| `T4_unigram_split_64k_oracle_dcs` | oracle split | 4.341 | 322,022 | 2.010 |
| `T5_morphbpe_raw_32k_dcs` | sandhied | 3.912 | 336,232 | 2.099 |
| `T5_morphbpe_raw_64k_dcs` | sandhied | 4.162 | 316,049 | 1.973 |
| `T5_morphbpe_rawseg_32k_dcs` | sandhied | 4.483 | 293,390 | 1.832 |
| `T5_morphbpe_rawseg_64k_dcs` | sandhied | 4.889 | 269,037 | 1.680 |
| `T6_morphbpe_split_32k_dcs` | oracle split | 4.211 | 332,018 | 2.073 |
| `T6_morphbpe_split_64k_dcs` | oracle split | 4.403 | 317,515 | 1.982 |

Side by side with the same contrasts' token cost on Sāmayik test:

| contrast | 32k in domain | 32k Sāmayik test | 64k in domain | 64k Sāmayik test |
|---|---|---|---|---|
| `T5` − `T1_dcs` | **+19.4%** | +4.0% | **+23.5%** | +5.1% |
| `T5seg` − `T1_dcs` | **+4.2%** | +0.2% | **+5.1%** | +0.1% |
| `T6` − `T4_oracle_dcs` | **+25.8%** | +4.3% | **+29.9%** | +5.9% |
| `T6` − `T1_dcs` | **+17.9%** | +2.7% | **+24.1%** | +4.6% |
| `T4_oracle` − `T1_dcs` | −6.3% | −1.5% | −4.5% | −1.2% |

**Domain mismatch shrinks the constraint's cost; it does not create it.** In domain the
constraint costs four to five times what it costs on the out-of-domain parallel corpora,
in the same direction, at every size and for every contrast. `T5seg` is the cheapest arm in
both settings by a wide margin: +5.1% in domain against `T5`'s +23.5% at 64k.

---

## TPP levels — context, not a comparison

Neither pivot is a matched control for a DCS-trained arm. `E1_bpe_64k` is **cross-corpus**;
`T0_o200k` is deployed practice at a 200k general-domain vocabulary. Both columns are
reported so the file is self-contained, and neither carries a verdict. The provisional
arms sit far below the DCS arms for a reason that has nothing to do with the method: they
were trained on the very corpora they are evaluated on.

### Sāmayik test

| arm | vs `E1_bpe_64k` (cross-corpus) | vs `T0_o200k` (deployed) |
|---|---|---|
| `T1_bpe_raw_32k_dcs` | 1.736 [1.707, 1.766] | 1.487 [1.463, 1.514] |
| `T1_bpe_raw_64k_dcs` | 1.642 [1.615, 1.672] | 1.406 [1.382, 1.433] |
| `T2_unigram_raw_32k_dcs` | 1.925 [1.893, 1.959] | 1.649 [1.621, 1.679] |
| `T2_unigram_raw_64k_dcs` | 1.851 [1.820, 1.884] | 1.586 [1.557, 1.616] |
| `T4_bpe_split_32k_oracle_dcs` | 1.709 [1.681, 1.740] | 1.464 [1.440, 1.491] |
| `T4_bpe_split_64k_oracle_dcs` | **1.622** [1.593, 1.652] | **1.389** [1.365, 1.416] |
| `T4_unigram_split_32k_oracle_dcs` | 1.942 [1.910, 1.975] | 1.664 [1.635, 1.693] |
| `T4_unigram_split_64k_oracle_dcs` | 1.885 [1.854, 1.918] | 1.615 [1.586, 1.644] |
| `T5_morphbpe_raw_32k_dcs` | 1.805 [1.778, 1.834] | 1.546 [1.522, 1.573] |
| `T5_morphbpe_raw_64k_dcs` | 1.725 [1.699, 1.754] | 1.478 [1.455, 1.504] |
| `T5_morphbpe_rawseg_32k_dcs` | 1.740 [1.712, 1.769] | 1.490 [1.465, 1.517] |
| `T5_morphbpe_rawseg_64k_dcs` | 1.644 [1.617, 1.673] | 1.408 [1.384, 1.434] |
| `T6_morphbpe_split_32k_dcs` | 1.782 [1.755, 1.811] | 1.526 [1.502, 1.551] |
| `T6_morphbpe_split_64k_dcs` | 1.718 [1.692, 1.746] | 1.472 [1.448, 1.496] |
| `T1_bpe_raw_64k*` | 1.035 [1.021, 1.049] | 0.887 [0.875, 0.899] |
| `T4_bpe_split_64k*` | 1.030 [1.016, 1.043] | 0.882 [0.869, 0.895] |

### Sāmayik test-OOD / Itihāsa test / FLORES devtest, 64k arms only

| arm | OOD vs E1 | OOD vs o200k | Itihāsa vs E1 | Itihāsa vs o200k | FLORES vs E1 | FLORES vs o200k |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_64k_dcs` | 1.216 | 1.201 | 0.683 | 0.528 | 1.264 | 1.348 |
| `T2_unigram_raw_64k_dcs` | 1.380 | 1.363 | 0.787 | 0.609 | 1.498 | 1.599 |
| `T4_bpe_split_64k_oracle_dcs` | **1.209** | **1.194** | 0.706 | 0.546 | **1.230** | **1.312** |
| `T4_unigram_split_64k_oracle_dcs` | 1.423 | 1.405 | 0.882 | 0.682 | 1.466 | 1.564 |
| `T5_morphbpe_raw_64k_dcs` | 1.309 | 1.293 | 0.837 | 0.647 | 1.364 | 1.455 |
| `T5_morphbpe_rawseg_64k_dcs` | 1.229 | 1.214 | 0.731 | 0.565 | 1.270 | 1.355 |
| `T6_morphbpe_split_64k_dcs` | 1.307 | 1.291 | 0.853 | 0.660 | 1.343 | 1.433 |
| `T1_bpe_raw_64k*` | 1.061 | 1.048 | 0.598 | 0.462 | 1.144 | 1.220 |
| `T4_bpe_split_64k*` | 1.036 | 1.024 | 0.616 | 0.476 | 1.108 | 1.182 |

Full intervals for every corpus and every arm are in `results.json` under `tpp`.

---

## Fertility and compression — reported, never led with

Fertility primary is tokens per **raw** whitespace word for every arm, so the column is
comparable across raw and split arms; a split arm's unit label reads
`tokens/reference word` and its secondary number (tokens per *split* word) is beside it.
Compression is UTF-8 bytes per token on the SLP1 text each arm actually tokenizes.

### Sāmayik test

| arm | fertility (tokens / raw word) | fertility secondary | bytes/token |
|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | 2.507 | — | 3.013 |
| `T1_bpe_raw_64k_dcs` | 2.370 | — | 3.185 |
| `T2_unigram_raw_32k_dcs` | 2.782 | — | 2.716 |
| `T2_unigram_raw_64k_dcs` | 2.675 | — | 2.825 |
| `T4_bpe_split_32k_oracle_dcs` | 2.484 | 2.140 | 3.150 |
| `T4_bpe_split_64k_oracle_dcs` | **2.358** | 2.031 | **3.319** |
| `T4_unigram_split_32k_oracle_dcs` | 2.823 | 2.431 | 2.772 |
| `T4_unigram_split_64k_oracle_dcs` | 2.740 | 2.360 | 2.856 |
| `T5_morphbpe_raw_32k_dcs` | 2.607 | — | 2.897 |
| `T5_morphbpe_raw_64k_dcs` | 2.492 | — | 3.031 |
| `T5_morphbpe_rawseg_32k_dcs` | 2.513 | — | 3.006 |
| `T5_morphbpe_rawseg_64k_dcs` | 2.373 | — | 3.181 |
| `T6_morphbpe_split_32k_dcs` | 2.590 | 2.231 | 3.022 |
| `T6_morphbpe_split_64k_dcs` | 2.497 | 2.151 | 3.134 |
| `T1_bpe_raw_64k*` | 1.488 | — | 5.053 |
| `T4_bpe_split_64k*` | 1.497 | 1.289 | 5.229 |

### 64k arms, all four corpora (fertility / bytes-per-token)

| arm | Sāmayik test | Sāmayik OOD | Itihāsa test | FLORES devtest |
|---|---|---|---|---|
| `T1_bpe_raw_64k_dcs` | 2.370 / 3.185 | 2.429 / 3.812 | 1.956 / 4.832 | 2.127 / 3.859 |
| `T2_unigram_raw_64k_dcs` | 2.675 / 2.825 | 2.758 / 3.358 | 2.256 / 4.188 | 2.523 / 3.255 |
| `T4_bpe_split_64k_oracle_dcs` | 2.358 / 3.319 | 2.415 / 3.954 | 2.024 / 4.922 | 2.077 / 4.045 |
| `T4_unigram_split_64k_oracle_dcs` | 2.740 / 2.856 | 2.844 / 3.358 | 2.526 / 3.943 | 2.476 / 3.392 |
| `T5_morphbpe_raw_64k_dcs` | 2.492 / 3.031 | 2.615 / 3.541 | 2.399 / 3.939 | 2.296 / 3.576 |
| `T5_morphbpe_rawseg_64k_dcs` | 2.373 / 3.181 | 2.456 / 3.770 | 2.094 / 4.513 | 2.138 / 3.840 |
| `T6_morphbpe_split_64k_dcs` | 2.497 / 3.134 | 2.612 / 3.656 | 2.445 / 4.074 | 2.269 / 3.703 |

**The constraint costs compression, and how much depends on which constraint.** At 64k on
Sāmayik test, `T5` gets 3.031 bytes/token against `T1_dcs`'s 3.185 (−4.8%) and `T6` gets
3.134 against `T4_oracle`'s 3.319 (−5.6%); `T5seg` gets 3.181, which is `T1_dcs`'s number to
within 0.1%. On Itihāsa the gaps widen (`T5` 3.939 vs 4.832, −18.5%; `T5seg` 4.513, −6.6%).
That is the same fact as the positive TPP deltas seen from the character side: forbidding
merges across boundaries removes the long merges a compressor wants, and forbidding them
across *gold segment* boundaries alone removes far fewer of them than forbidding them across
segment ∪ heuristic stem.

---

## Text invariants

Every transformation this experiment measures on is recorded, per CLAUDE.md and the
Experiment 03 finding that made it necessary.

| transformation | letter retention |
|---|---|
| DCS held-out: raw → oracle split | 1.0262 |
| Sāmayik test: raw → ByT5-reconciled | 1.0138 |
| Sāmayik OOD: raw → ByT5-reconciled | 1.0096 |
| Itihāsa test: raw → ByT5-reconciled | 1.0207 |
| FLORES devtest: raw → ByT5-reconciled | 1.0058 |

**The DCS oracle split is a re-analysis, not a re-segmentation**, so two of these checks do
not apply to it in the sense they were written for. Its letter retention is 1.0262 because
reversing sandhi legitimately restores elided phonemes, and
`letters_out_subset_of_raw_union_model` is `false` for the same reason — the check asks
whether the output invents letters neither source supplies, and DCS's unsandhied forms
*do* supply letters the surface does not. This is why the oracle split is used for
**training** the `T4_oracle`/`T6` arms and for MorphScore, and never as a text an arm is
credited with compressing. The reconciled rows are Experiment 03's, unchanged, and are what
the deletion-cost column in the TPP tables prices. Full per-corpus counts (missing, added,
gross) are in `results.json` under `text_invariants`.

---

## Verdicts

**H3, MorphScore half — the constraint half is SUPPORTED; the splitting half is NOT TESTED
here.** Forbidding merges across **gold segment boundaries (`T5seg`)** raises boundary F1 on
the human-verified held-out subset by **+0.1143 [+0.1063, +0.1224]** at 64k and
**+0.0979 [+0.0908, +0.1047]** at 32k over the matched unconstrained control, and raises
precision and recall together; forbidding them across **segment ∪ heuristic stem (`T5`)**
raises F1 by +0.0355 [+0.0250, +0.0458] at 64k, essentially all of it recall; and forbidding
them across **heuristic stem boundaries inside the gold split (`T6`)** raises stem F1 by
+0.3055 [+0.2938, +0.3169] at 64k over `T4_oracle` — but against the heuristic stem
boundary its own training text was marked with, so that row reads as "the constraint was
applied", not as an independent validation (Result 2). Every interval excludes 0 at both
sizes in both populations. Whether *sandhi splitting alone* raises MorphScore, which is what H3
literally asserts, cannot be read off these tables: the only granularity a split arm has is
scored over a different population of units (gold segments) from the raw arm's (surface
words), so the comparison would be between two corpora rather than between two tokenizers.

**H4, TPP half — NOT SUPPORTED, but the cost is much smaller than the first version of this
file reported, and for the clean arm it is not distinguishable from zero.** No constrained
delta is negative on any corpus at any size. But their sizes differ by an order of
magnitude, and the difference is exactly the heuristic stem constraint:

- **`T5seg` (gold segment boundaries only)** costs **+0.0022 TPP [−0.0010, +0.0054]** on
  Sāmayik test at 64k — a 0.13% token cost whose CI **includes 0**, the only constrained
  delta in the experiment of which that is true — and +0.0040 [+0.0009, +0.0071] at 32k. On
  the other three corpora it is +0.0065 to +0.0482, all CIs clear of 0, and in domain it
  costs +5.1% tokens at 64k.
- **`T5` (segment ∪ heuristic stem)** costs +0.0837 [+0.0782, +0.0896] at 64k on the same
  corpus, 5.1% more tokens, and +23.5% in domain.
- **`T6` (the proposed arm)** costs +0.0760 [+0.0698, +0.0826] against `T1_dcs` at 64k, 4.6%
  more tokens out of domain and +24.1% in domain.

The only negative deltas anywhere are **gold splitting without the constraint** —
`T4_oracle` − `T1_dcs` — which saves a small amount on prose (−0.0267 / −0.0199 on Sāmayik
test at 32k / 64k, −0.0115 / −0.0071 on OOD), saves on FLORES (−0.0338 / −0.0339), and is
adverse on verse at both sizes. That reproduces Experiment 03's finding with a gold splitter
in place of a model one, and it is the upper bound on what splitting can buy.

**The honest summary.** The constraint does what it is supposed to do to the segmentation,
and its price is almost entirely attributable to the *heuristic* half of the boundary set,
not to the gold half. Constrained on DCS's own segment annotation alone, MorphScore rises
most (+0.11 F1 at 64k, the largest gain in the experiment) and the token cost falls to
something a bootstrap cannot distinguish from zero on the primary prose corpus. That makes
`T5seg` — not `T6` — the arm Experiment 05 should carry forward, and it makes "is the
constraint worth its cost?" a live question rather than a closed one.

**What this does not settle.** H4 is a claim about tokens to a reference
bits-per-character, not about tokens per proposition. A vocabulary that costs 5% more tokens
can still reach a reference BPC sooner if its tokens are correspondingly easier to predict —
which is exactly the trade a morphologically-aligned vocabulary would be expected to make.
Experiment 05 measures BPC and tokens-to-reference-loss, and it is the experiment that
decides H4.

---

## Caveats

1. **Domain mismatch shrinks the constraint's cost; it does not create it.** Every arm in
   the paired deltas is trained on DCS — classical Sanskrit across 258 texts — and evaluated
   on modern prose (Sāmayik), epic verse (Itihāsa) and translated news (FLORES). The
   mismatch is identical for both sides of every pair, so it cannot produce a delta. The
   in-domain table in Result 3 shows the same contrasts on held-out DCS: the constraint's
   token cost is **four to five times larger** in domain (`T5` +23.5% vs +5.1% at 64k;
   `T5seg` +5.1% vs +0.1%). The out-of-domain numbers are the *conservative* ones.
2. **Oracle split at training, ByT5 split at evaluation.** The `T4_oracle`/`T6` arms are
   trained on DCS's *gold* segmentation and evaluated on the *ByT5-reconciled* split of the
   parallel corpora, which is the only split those corpora have. A split arm therefore meets
   a segmentation at evaluation that is not the one it learned on. This is a real limitation
   of the TPP half and is stated in the decisions log; MorphScore has no such mismatch,
   since it is computed on the DCS held-out oracle split. `T5` and `T5seg` are unaffected —
   they tokenize the sandhied surface on both sides.
3. **Stem boundaries are a heuristic, and `T5seg` exists because of it.** The stem/ending
   boundary is derived from the segment and its lemma by a part-of-speech-gated,
   sandhi-aware rule that refuses to cut inside the lemma's consonantal body
   (`docs/decisions.md`, 2026-09-05, "Stem boundaries are heuristic", and its CORRECTION).
   On the held-out split it makes 90,878 cuts, **36.5% of them one character short of the
   lemma's end at a fused stem-final vowel**, flagged as such. Its 0 cuts inside the lemma
   body is a property of the rule's definition, not a measurement of it: `stem_boundary`
   returns a cut only at `len(lemma)` or at a flagged fused position, so the audit's
   `n_inside_lemma` and `n_fused` are the same number by construction. What is measured is
   the 36.5%, and that the LCP rule it replaced put 21.4% of its cuts strictly inside the
   lemma body. Every stem-granularity number is labelled heuristic. `T5seg` is the arm no claim of which depends on it.
4. **The fused-sandhi boundary convention shifts boundaries by up to one character.** Vowel
   sandhi fuses two characters into one in 53.4% of multi-segment words, so no character
   offset is *the* boundary; `align_segments` places it on one side or the other depending
   on the alignment, and the direction is not uniform (`docs/decisions.md`, "Correction: the
   fused sandhi character does not consistently join the left segment"). The convention is
   **identical for every arm**, so paired comparisons are unaffected; the absolute level is
   not portable to a published MorphScore on a language without sandhi. The ±1 columns bound
   the effect.
5. **±1 on split text is not the sandhi correction.** Segments in the oracle split are
   whitespace-separated, so sandhi cannot displace a boundary there. The ±1 column for a
   split arm is a sensitivity bound on the stem heuristic and nothing else.
6. **68% of DCS's segmentation is machine-generated.** Only 6,952 of 30,150 held-out
   sentences (23.06%) carry no `UnsandhiedReconstructed=True` token. Both populations are
   reported above with their own bootstrap intervals, and every paired delta keeps its sign
   and its CI-versus-zero status across them, so the conclusions are not an artefact of the
   annotator's own segmenter. The *levels* do differ between populations — `T5seg` 64k
   segment F1 is 0.5792 verified against 0.5888 over all — and the verified subset is the
   one to read.
7. **The English pivot is cross-corpus.** `E1_bpe_64k` was trained on the English side of
   the parallel corpora, not on anything DCS-sized or DCS-like. It cancels from the paired
   deltas, which is why they carry the verdict; it does not cancel from the levels, which is
   why the levels do not. `E1_unigram_64k` additionally trains to 62,896 pieces rather than
   64,000 (Experiment 02's README, footnote ‡), but it is not a pivot here.
8. **Verse is a confound.** Itihāsa is śloka: meter constrains word choice, so its numbers
   are reported second everywhere and no conclusion rests on them. They are also where the
   constraint's cost is largest (+0.1709 for `T6` − `T1_dcs` at 64k). **The contamination
   caveat that stood here is withdrawn**: after the shingle filter, 0 of 400 sampled Itihāsa
   test lines occur as letter-substrings of the DCS training corpus.
9. **The constrained arms cost compression**, by 0.1% (`T5seg`) to 18.5% (`T5` on verse)
   bytes per token depending on corpus and pair. That is not a side effect to be tuned away;
   it is the same measurement as the TPP deltas.
10. **The violation rate's numerator is tokens, its denominator boundaries.** A token
    spanning two gold boundaries counts once, so `violations_per_boundary` is a close lower
    bound on the fraction of boundaries crossed rather than that fraction exactly. It is
    still the right rate for comparing two arms, because a per-token rate rewards the
    constrained arm for being more verbose.
11. **`T2` (Unigram) is never a controlled comparison for a constrained arm.** The
    constraint is a statement about merges and Unigram has none, so `T2`/`T4_unigram` rows
    are context. Result 2 states where they beat BPE and where they do not, and why the two
    populations disagree.

---

## Artifacts

```
outputs/04_morph_constrained/
├── results.json                # every number above, strict JSON
├── config.yaml                 # byte copy of the config that produced it
├── constraint_effects.pdf/.png # the figure
├── run.log                     # the run's own log
└── marked/                     # held-out t5/t5seg/t6 marked text, written for the audit
```

`results.json` keys: `experiment`, `git_commit`, `git_dirty`, `timestamp`, `config`,
`tokenizer_sources`, `unavailable_arms`, `dcs_heldout`, `corpora`, `exclusion_check`,
`exclusion_check_en`, `spans_coverage`, `morphscore`, `morphscore_delta`,
`compression_indomain`, `violations`, `tpp`, `tpp_delta`, `fertility_primary`,
`fertility_secondary`, `compression`, `text_invariants`.

`outputs/` is gitignored. `results.json`, `config.yaml` and both figures from the run this
file reports are tracked at [`results/04_morph_constrained/`](../../results/04_morph_constrained/)
(`run.log` and `marked/` are not) — see [`results/README.md`](../../results/README.md).
