# Experiment 04 — Morpheme-constrained merges (T5, T6): MorphScore and paired TPP deltas

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
eighteen arms.

**H4's TPP half only.** H4 as written is about *tokens to a reference bits-per-character*,
which needs a trained language model; that is Experiment 05. What is answerable here is the
question one layer below it: does a morpheme-constrained vocabulary spend **fewer tokens per
unit of meaning** than the unconstrained vocabulary trained on the same sentences at the
same vocabulary size? If it does not, the training-efficiency claim has to come from
somewhere other than token count. It does not (below), and **that is a result about token
count, not a refutation of H4** — a constrained tokenizer could still reach a reference BPC
in fewer tokens by making each token easier to predict. Experiment 05 decides.

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

**Run:** `uv run python experiments/04_morph_constrained/run.py` — **2 min 53 s**, no
network beyond the cached off-the-shelf tokenizers, at commit `7f49aaa` with a clean tree
(`results.json` records `git_dirty: false`).

---

## Status

| Task | State |
|---|---|
| 1. DCS ingestion with aligned gold boundaries | done |
| 2. Token spans and MorphScore | done |
| 3. MorphBPE constraint and the twelve `_dcs` arms | done |
| 4. Runner, `results.json`, figure, this file | done |

Every number below is from `outputs/04_morph_constrained/results.json`, run 2026-09-05
09:35:30 UTC at commit `7f49aaa`, clean tree. No arm was unavailable
(`unavailable_arms: {}`); `T0_gemma3` resolved to `unsloth/gemma-3-4b-it`, the substitution
already recorded in `docs/decisions.md`.

---

## The arms

Twelve arms trained on the DCS training split (720,510 sentences), matched at 32k and 64k:

| Key | Text trained on | Constrained? |
|---|---|---|
| `T1_bpe_raw_{32k,64k}_dcs` | sandhied surface | no |
| `T2_unigram_raw_{32k,64k}_dcs` | sandhied surface | no |
| `T4_bpe_split_{32k,64k}_oracle_dcs` | DCS **gold** segmentation (`oracle`) | no |
| `T4_unigram_split_{32k,64k}_oracle_dcs` | DCS **gold** segmentation (`oracle`) | no |
| `T5_morphbpe_raw_{32k,64k}_dcs` | sandhied surface, boundary-marked | **yes** |
| `T6_morphbpe_split_{32k,64k}_dcs` | gold split, boundary-marked (**proposed**) | **yes** |

Plus, for context and marked as such in every table: **provisional** arms `T1_bpe_raw_64k*`,
`T2_unigram_raw_64k*`, `T4_bpe_split_64k*` (trained on the *parallel* corpora, Experiments
02–03), and **existing practice** `T0_o200k`, `T0_gemma3`, `T3_sarvam`.

## The evaluation sets

| Set | n | Note |
|---|---|---|
| DCS held-out | **30,150** sentences over 13 texts | 160,178 surface words (97.69% with an aligned gold segmentation), 214,234 gold segments |
| — human-verified subset | **6,952** (23.06%) | no `UnsandhiedReconstructed=True` token anywhere in the sentence |
| Sāmayik test (prose) | 2,417 pairs | primary parallel corpus |
| Sāmayik test-OOD (prose) | 4,047 pairs | |
| Itihāsa test (verse) | 11,721 pairs | secondary; meter is a confound |
| FLORES-200 devtest | 1,012 pairs | parity anchor |

**MorphScore ran over the whole held-out split** — all 30,150 sentences, no subsampling
(`results.json`'s `dcs_heldout.sample` is `null`), because the full pass costs under a
second per arm.

**No leakage.** All 30,150 held-out sentences hash into `data/exclusion_hashes.txt`
(`n_missing: 0`), which is what kept them out of every `_dcs` arm's training corpus; all four
parallel corpora are fully present in both the Sanskrit and the English exclusion lists
(`n_missing: 0` in all eight checks).

**Spans coverage.** Every one of the eighteen arms tiles both text forms exactly on the
first 500 held-out sentences (`spans_cover_text`, 500/500 passed, 0 failed, on `text_slp1`
and on `oracle_split_slp1`). No arm's MorphScore was withheld.

---

## Result 1 — the constraint works, out of sample

Each arm re-tokenizes the **held-out** marked text it never saw, and a token whose span
strictly contains a gold boundary is a violation. `violations per boundary` is the
comparable rate: violating tokens divided by the gold boundaries in the audited text — the
same denominator for both arms of a pair. Its numerator is offending *tokens*, so a token
swallowing two boundaries counts once, which makes it a close lower bound on the fraction of
boundaries crossed rather than that fraction exactly. The per-token rate is shown beside it
because it is what `assert_no_cross_boundary_merges` reports natively, and because it is the
*flattering* denominator: a constrained arm emits more tokens for the same text.

| vocab | raw text: `T1_dcs` | raw text: `T5_dcs` | reduction | oracle-split: `T4_oracle_dcs` | oracle-split: `T6_dcs` | reduction |
|---|---|---|---|---|---|---|
| 32k | **0.7182** | **0.4645** | −35.3% | **0.9394** | **0.4563** | −51.4% |
| 64k | **0.7373** | **0.4872** | −33.9% | **0.9634** | **0.4905** | −49.1% |

Counts behind them (30,150 sentences, stride 1, so every sentence audited):

| arm | text | violating tokens | gold boundaries | per boundary | tokens | per token |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | raw | 117,124 | 163,074 | 0.7182 | 292,446 | 0.4005 |
| `T1_bpe_raw_64k_dcs` | raw | 120,234 | 163,074 | 0.7373 | 266,150 | 0.4518 |
| `T5_morphbpe_raw_32k_dcs` | raw | 75,741 | 163,074 | **0.4645** | 350,875 | 0.2159 |
| `T5_morphbpe_raw_64k_dcs` | raw | 79,457 | 163,074 | **0.4872** | 329,401 | 0.2412 |
| `T4_bpe_split_32k_oracle_dcs` | split | 113,154 | 120,459 | 0.9394 | 272,625 | 0.4151 |
| `T4_bpe_split_64k_oracle_dcs` | split | 116,055 | 120,459 | 0.9634 | 251,289 | 0.4618 |
| `T6_morphbpe_split_32k_dcs` | split | 54,969 | 120,459 | **0.4563** | 351,021 | 0.1566 |
| `T6_morphbpe_split_64k_dcs` | split | 59,084 | 120,459 | **0.4905** | 334,987 | 0.1764 |

**Read the two halves of the table separately.** The raw audit is against segment ∪
projected-stem boundaries (`t5_marked`, 163,074 of them) and the split audit against stem
boundaries only (`t6_marked`, 120,459), because that is what each marked form carries. Raw
and split rows share no denominator; only the within-column comparison is a comparison.

**The residue is expected and is the point.** A BPE merge rule is a global character pair:
the marker stops the *trainer* counting a pair that straddles a boundary, but a rule learned
elsewhere still applies inside a word at inference, where the text has no markers. Task 3
measured this in sample (`docs/decisions.md`, "the boundary-marker constraint halves
cross-boundary tokens but cannot reach zero"); this is the same measurement on sentences no
arm saw, and the reduction holds out of sample at the same magnitude.

---

## Result 2 — MorphScore (the H3 half)

`value` is boundary F1, exclusions per Arnett & Bergen: a word emitted as one token and a
word with no gold boundary are both excluded and counted. Raw arms are scored on the
sandhied surface's whitespace words against **segment** boundaries (primary); split arms are
scored on each gold **segment** against the **stem/ending** boundary inside it, which is a
lemma-LCP **heuristic** and is labelled as one. Raw arms additionally carry the stem
granularity, projected into the surface where the word's alignment succeeded.

### Human-verified subset (6,952 sentences) — the number to read

| arm | granularity | F1 exact | P | R | F1 ±1 | words scored |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | segment | 0.4496 | 0.3512 | 0.6245 | 0.5693 | 4,620 |
| `T1_bpe_raw_64k_dcs` | segment | **0.4640** | 0.3779 | 0.6008 | 0.5896 | 4,319 |
| `T2_unigram_raw_32k_dcs` | segment | 0.4004 | 0.3177 | 0.5411 | 0.5662 | 4,862 |
| `T2_unigram_raw_64k_dcs` | segment | 0.3905 | 0.3270 | 0.4847 | 0.5695 | 4,803 |
| `T5_morphbpe_raw_32k_dcs` | segment | 0.4761 | 0.3572 | 0.7140 | 0.6266 | 5,064 |
| `T5_morphbpe_raw_64k_dcs` | segment | **0.4870** | 0.3795 | 0.6794 | 0.6452 | 5,045 |
| `T1_bpe_raw_64k*` | segment | 0.4306 | 0.3290 | 0.6230 | 0.5476 | 4,755 |
| `T2_unigram_raw_64k*` | segment | 0.3803 | 0.2841 | 0.5747 | 0.5447 | 4,878 |
| `T0_o200k` | segment | 0.1302 | 0.0833 | 0.2978 | 0.4276 | 5,111 |
| `T0_gemma3` | segment | 0.1214 | 0.0780 | 0.2741 | 0.4289 | 5,091 |
| `T3_sarvam` | segment | 0.1472 | 0.0891 | 0.4225 | 0.3479 | 5,116 |
| `T1_bpe_raw_32k_dcs` | stem (heuristic) | 0.1504 | 0.1255 | 0.1877 | 0.5015 | 13,564 |
| `T1_bpe_raw_64k_dcs` | stem (heuristic) | **0.1310** | 0.1116 | 0.1585 | 0.4847 | 10,938 |
| `T2_unigram_raw_32k_dcs` | stem (heuristic) | 0.3092 | 0.2652 | 0.3705 | 0.5159 | 18,680 |
| `T2_unigram_raw_64k_dcs` | stem (heuristic) | 0.2886 | 0.2586 | 0.3264 | 0.4896 | 17,854 |
| `T5_morphbpe_raw_32k_dcs` | stem (heuristic) | 0.3894 | 0.3274 | 0.4804 | 0.6688 | 22,455 |
| `T5_morphbpe_raw_64k_dcs` | stem (heuristic) | **0.3966** | 0.3436 | 0.4689 | 0.6805 | 21,633 |
| `T4_bpe_split_32k_oracle_dcs` | stem (heuristic) | 0.1929 | 0.1707 | 0.2217 | 0.5721 | 9,454 |
| `T4_bpe_split_64k_oracle_dcs` | stem (heuristic) | **0.1751** | 0.1586 | 0.1953 | 0.5703 | 6,395 |
| `T4_unigram_split_32k_oracle_dcs` | stem (heuristic) | 0.3700 | 0.3168 | 0.4448 | 0.6481 | 9,515 |
| `T4_unigram_split_64k_oracle_dcs` | stem (heuristic) | 0.3752 | 0.3324 | 0.4307 | 0.6333 | 7,114 |
| `T6_morphbpe_split_32k_dcs` | stem (heuristic) | 0.5213 | 0.4767 | 0.5751 | 0.7686 | 25,119 |
| `T6_morphbpe_split_64k_dcs` | stem (heuristic) | **0.5354** | 0.5034 | 0.5717 | 0.7863 | 23,719 |
| `T4_bpe_split_64k*` | stem (heuristic) | 0.1934 | 0.1672 | 0.2292 | 0.5374 | 12,257 |
| `T0_o200k` | stem (heuristic) | 0.2815 | 0.1924 | 0.5244 | 0.4872 | 24,392 |
| `T0_gemma3` | stem (heuristic) | 0.2816 | 0.1930 | 0.5211 | 0.4799 | 23,792 |
| `T3_sarvam` | stem (heuristic) | 0.2536 | 0.1621 | 0.5825 | 0.4198 | 25,494 |

### All held-out sentences (30,150) — the same picture, slightly higher

| arm | granularity | F1 exact | F1 ±1 | words scored |
|---|---|---|---|---|
| `T1_bpe_raw_32k_dcs` | segment | 0.4521 | 0.6303 | 35,641 |
| `T1_bpe_raw_64k_dcs` | segment | 0.4635 | 0.6525 | 33,840 |
| `T2_unigram_raw_32k_dcs` | segment | 0.4896 | 0.6306 | 36,907 |
| `T2_unigram_raw_64k_dcs` | segment | 0.4826 | 0.6269 | 36,579 |
| `T5_morphbpe_raw_32k_dcs` | segment | 0.4821 | 0.6482 | 38,740 |
| `T5_morphbpe_raw_64k_dcs` | segment | **0.5032** | 0.6794 | 38,612 |
| `T1_bpe_raw_64k*` | segment | 0.4347 | 0.6046 | 34,441 |
| `T2_unigram_raw_64k*` | segment | 0.4315 | 0.5924 | 36,122 |
| `T0_o200k` | segment | 0.1425 | 0.4298 | 38,893 |
| `T0_gemma3` | segment | 0.1500 | 0.4353 | 38,852 |
| `T3_sarvam` | segment | 0.1532 | 0.3504 | 38,899 |
| `T1_bpe_raw_32k_dcs` | stem (heuristic) | 0.1241 | 0.4224 | 60,836 |
| `T1_bpe_raw_64k_dcs` | stem (heuristic) | 0.1133 | 0.3974 | 51,942 |
| `T2_unigram_raw_32k_dcs` | stem (heuristic) | 0.2884 | 0.4592 | 80,235 |
| `T2_unigram_raw_64k_dcs` | stem (heuristic) | 0.2776 | 0.4465 | 77,681 |
| `T5_morphbpe_raw_32k_dcs` | stem (heuristic) | 0.3639 | 0.5887 | 93,260 |
| `T5_morphbpe_raw_64k_dcs` | stem (heuristic) | 0.3679 | 0.5947 | 90,257 |
| `T4_bpe_split_32k_oracle_dcs` | stem (heuristic) | 0.1681 | 0.5276 | 38,110 |
| `T4_bpe_split_64k_oracle_dcs` | stem (heuristic) | 0.1571 | 0.5074 | 25,150 |
| `T4_unigram_split_32k_oracle_dcs` | stem (heuristic) | 0.3828 | 0.6301 | 37,943 |
| `T4_unigram_split_64k_oracle_dcs` | stem (heuristic) | 0.4047 | 0.6282 | 29,564 |
| `T6_morphbpe_split_32k_dcs` | stem (heuristic) | 0.5469 | 0.7479 | 109,128 |
| `T6_morphbpe_split_64k_dcs` | stem (heuristic) | **0.5570** | 0.7625 | 103,351 |
| `T4_bpe_split_64k*` | stem (heuristic) | 0.1726 | 0.5033 | 42,335 |
| `T0_o200k` | stem (heuristic) | 0.2525 | 0.4457 | 100,416 |
| `T0_gemma3` | stem (heuristic) | 0.2528 | 0.4460 | 99,222 |
| `T3_sarvam` | stem (heuristic) | 0.2299 | 0.3754 | 102,247 |

### The two clean comparisons, and the one that is not clean

Only two of these rows form a matched pair over the **same population of units**:

| contrast | granularity | 32k | 64k |
|---|---|---|---|
| `T5_dcs` − `T1_dcs` (constraint on raw) | segment | 0.4761 − 0.4496 = **+0.027** | 0.4870 − 0.4640 = **+0.023** |
| `T5_dcs` − `T1_dcs` (constraint on raw) | stem | 0.3894 − 0.1504 = **+0.239** | 0.3966 − 0.1310 = **+0.266** |
| `T6_dcs` − `T4_oracle_dcs` (constraint on split) | stem | 0.5213 − 0.1929 = **+0.328** | 0.5354 − 0.1751 = **+0.360** |

**"Splitting raises MorphScore" is *not* cleanly testable here**, and no number in this file
should be read as testing it. A raw arm is scored on 43,635 surface *words* and a split arm
on 50,268 *segments*; they are different units of different lengths, so `T4_oracle` −
`T1_dcs` (0.1751 − 0.1310 at 64k) compares two populations, not two tokenizers. The segment
granularity, which would be the fair one, does not exist for a split arm at all: in the
oracle split the segment boundaries *are* whitespace.

---

## Result 3 — the paired TPP deltas (the H4 half)

`Δ = TPP(a) − TPP(b)`, both arms on the same sentences against the same English side
(`E1_bpe_64k`, cross-corpus), 95% paired bootstrap over 1,000 resamples, seed 0. **Negative
is the result H4's TPP half predicts.** Prose first.

### Sāmayik test (prose, 2,417 pairs) — the primary corpus

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) | tokens saved | deletion cost |
|---|---|---|---|---|---|---|
| `T5_morphbpe_raw_32k_dcs` − `T1_bpe_raw_32k_dcs` | **+0.0752** | [+0.0687, +0.0806] | 1.772 | 1.697 | −2,616 | n/a (same text) |
| `T5_morphbpe_raw_64k_dcs` − `T1_bpe_raw_64k_dcs` | **+0.0810** | [+0.0751, +0.0867] | 1.688 | 1.607 | −2,820 | n/a (same text) |
| `T4_bpe_split_32k_oracle_dcs` − `T1_bpe_raw_32k_dcs` | **−0.0291** | [−0.0342, −0.0236] | 1.668 | 1.697 | +1,012 | 89 tok (8.8% of saving) |
| `T4_bpe_split_64k_oracle_dcs` − `T1_bpe_raw_64k_dcs` | **−0.0178** | [−0.0229, −0.0124] | 1.589 | 1.607 | +620 | 84 tok (13.5% of saving) |
| `T6_morphbpe_split_32k_dcs` − `T4_bpe_split_32k_oracle_dcs` | **+0.0883** | [+0.0814, +0.0945] | 1.757 | 1.668 | −3,074 | n/a (same text) |
| `T6_morphbpe_split_64k_dcs` − `T4_bpe_split_64k_oracle_dcs` | **+0.1025** | [+0.0955, +0.1088] | 1.692 | 1.589 | −3,567 | n/a (same text) |
| `T6_morphbpe_split_32k_dcs` − `T1_bpe_raw_32k_dcs` | **+0.0593** | [+0.0519, +0.0659] | 1.757 | 1.697 | −2,062 | 89 tok (saving is negative) |
| `T6_morphbpe_split_64k_dcs` − `T1_bpe_raw_64k_dcs` | **+0.0847** | [+0.0772, +0.0920] | 1.692 | 1.607 | −2,947 | 84 tok (saving is negative) |

### Sāmayik test-OOD (prose, 4,047 pairs)

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) |
|---|---|---|---|---|
| `T5` − `T1` 32k | +0.0999 | [+0.0967, +0.1030] | 1.400 | 1.300 |
| `T5` − `T1` 64k | +0.0953 | [+0.0922, +0.0985] | 1.317 | 1.222 |
| `T4_oracle` − `T1` 32k | **−0.0117** | [−0.0148, −0.0083] | 1.288 | 1.300 |
| `T4_oracle` − `T1` 64k | **−0.0091** | [−0.0122, −0.0059] | 1.213 | 1.222 |
| `T6` − `T4_oracle` 32k | +0.0989 | [+0.0956, +0.1024] | 1.387 | 1.288 |
| `T6` − `T4_oracle` 64k | +0.1139 | [+0.1106, +0.1172] | 1.327 | 1.213 |
| `T6` − `T1` 32k | +0.0873 | [+0.0836, +0.0909] | 1.387 | 1.300 |
| `T6` − `T1` 64k | **+0.1048** | [+0.1011, +0.1081] | 1.327 | 1.222 |

Deletion cost on the two split-vs-raw pairs: 287 tokens (25.4% of a 1,130-token saving) at
32k and 281 (31.8% of 884) at 64k — a third of the gold-splitting saving on this corpus is
characters the reconciled text no longer contains, not boundaries the split found.

### Itihāsa test (verse, 11,721 pairs) — meter is a confound

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) |
|---|---|---|---|---|
| `T5` − `T1` 32k | +0.1478 | [+0.1461, +0.1496] | 0.915 | 0.768 |
| `T5` − `T1` 64k | +0.1615 | [+0.1599, +0.1632] | 0.865 | 0.704 |
| `T4_oracle` − `T1` 32k | +0.0019 | [−0.0001, +0.0041] | 0.770 | 0.768 |
| `T4_oracle` − `T1` 64k | +0.0138 | [+0.0120, +0.0159] | 0.717 | 0.704 |
| `T6` − `T4_oracle` 32k | +0.1605 | [+0.1585, +0.1626] | 0.930 | 0.770 |
| `T6` − `T4_oracle` 64k | +0.1739 | [+0.1719, +0.1760] | 0.891 | 0.717 |
| `T6` − `T1` 32k | +0.1624 | [+0.1606, +0.1643] | 0.930 | 0.768 |
| `T6` − `T1` 64k | **+0.1877** | [+0.1859, +0.1897] | 0.891 | 0.704 |

`T4_oracle` − `T1` at 32k is the only interval in the whole table that includes 0.

### FLORES-200 devtest (1,012 pairs)

| contrast | Δ TPP | 95% CI | TPP(a) | TPP(b) |
|---|---|---|---|---|
| `T5` − `T1` 32k | +0.1155 | [+0.1092, +0.1218] | 1.462 | 1.347 |
| `T5` − `T1` 64k | +0.1039 | [+0.0984, +0.1097] | 1.371 | 1.267 |
| `T4_oracle` − `T1` 32k | **−0.0325** | [−0.0379, −0.0273] | 1.314 | 1.347 |
| `T4_oracle` − `T1` 64k | **−0.0370** | [−0.0424, −0.0319] | 1.230 | 1.267 |
| `T6` − `T4_oracle` 32k | +0.1092 | [+0.1033, +0.1151] | 1.423 | 1.314 |
| `T6` − `T4_oracle` 64k | +0.1273 | [+0.1205, +0.1342] | 1.357 | 1.230 |
| `T6` − `T1` 32k | +0.0767 | [+0.0699, +0.0833] | 1.423 | 1.347 |
| `T6` − `T1` 64k | **+0.0903** | [+0.0834, +0.0974] | 1.357 | 1.267 |

Deletion cost: 87 tokens (9.2% of a 942-token saving) at 32k and 86 (8.0% of 1,073) at 64k.

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
| `T1_bpe_raw_32k_dcs` | 1.697 [1.670, 1.727] | 1.502 [1.476, 1.529] |
| `T1_bpe_raw_64k_dcs` | 1.607 [1.580, 1.636] | 1.422 [1.397, 1.448] |
| `T2_unigram_raw_32k_dcs` | 1.860 [1.828, 1.893] | 1.645 [1.616, 1.676] |
| `T2_unigram_raw_64k_dcs` | 1.786 [1.756, 1.818] | 1.580 [1.552, 1.610] |
| `T4_bpe_split_32k_oracle_dcs` | 1.668 [1.641, 1.698] | 1.476 [1.451, 1.503] |
| `T4_bpe_split_64k_oracle_dcs` | **1.589** [1.562, 1.620] | **1.406** [1.381, 1.432] |
| `T4_unigram_split_32k_oracle_dcs` | 1.742 [1.711, 1.776] | 1.541 [1.514, 1.572] |
| `T4_unigram_split_64k_oracle_dcs` | 1.674 [1.644, 1.708] | 1.481 [1.455, 1.511] |
| `T5_morphbpe_raw_32k_dcs` | 1.772 [1.746, 1.800] | 1.568 [1.545, 1.594] |
| `T5_morphbpe_raw_64k_dcs` | 1.688 [1.663, 1.715] | 1.493 [1.470, 1.518] |
| `T6_morphbpe_split_32k_dcs` | 1.757 [1.730, 1.785] | 1.554 [1.529, 1.578] |
| `T6_morphbpe_split_64k_dcs` | 1.692 [1.667, 1.719] | 1.496 [1.473, 1.520] |
| `T1_bpe_raw_64k*` | 1.027 [1.014, 1.040] | 0.908 [0.896, 0.921] |
| `T4_bpe_split_64k*` | 1.021 [1.008, 1.035] | 0.903 [0.891, 0.916] |

### Sāmayik test-OOD / Itihāsa test / FLORES devtest, 64k arms only

| arm | OOD vs E1 | OOD vs o200k | Itihāsa vs E1 | Itihāsa vs o200k | FLORES vs E1 | FLORES vs o200k |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_64k_dcs` | 1.222 | 1.221 | 0.704 | 0.547 | 1.267 | 1.367 |
| `T2_unigram_raw_64k_dcs` | 1.337 | 1.335 | 0.765 | 0.595 | 1.461 | 1.577 |
| `T4_bpe_split_64k_oracle_dcs` | **1.213** | **1.212** | 0.717 | 0.557 | **1.230** | **1.327** |
| `T4_unigram_split_64k_oracle_dcs` | 1.273 | 1.271 | 0.743 | 0.577 | 1.326 | 1.430 |
| `T5_morphbpe_raw_64k_dcs` | 1.317 | 1.316 | 0.865 | 0.672 | 1.371 | 1.479 |
| `T6_morphbpe_split_64k_dcs` | 1.327 | 1.326 | 0.891 | 0.692 | 1.357 | 1.464 |
| `T1_bpe_raw_64k*` | 1.067 | 1.066 | 0.607 | 0.472 | 1.138 | 1.227 |
| `T4_bpe_split_64k*` | 1.041 | 1.040 | 0.624 | 0.485 | 1.102 | 1.189 |

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
| `T1_bpe_raw_32k_dcs` | 2.531 | — | 2.983 |
| `T1_bpe_raw_64k_dcs` | 2.396 | — | 3.151 |
| `T2_unigram_raw_32k_dcs` | 2.776 | — | 2.723 |
| `T2_unigram_raw_64k_dcs` | 2.665 | — | 2.835 |
| `T4_bpe_split_32k_oracle_dcs` | 2.504 | 2.157 | 3.125 |
| `T4_bpe_split_64k_oracle_dcs` | **2.386** | 2.055 | **3.281** |
| `T4_unigram_split_32k_oracle_dcs` | 2.615 | 2.253 | 2.992 |
| `T4_unigram_split_64k_oracle_dcs` | 2.514 | 2.165 | 3.114 |
| `T5_morphbpe_raw_32k_dcs` | 2.644 | — | 2.857 |
| `T5_morphbpe_raw_64k_dcs` | 2.517 | — | 3.000 |
| `T6_morphbpe_split_32k_dcs` | 2.637 | 2.271 | 2.968 |
| `T6_morphbpe_split_64k_dcs` | 2.539 | 2.187 | 3.082 |
| `T1_bpe_raw_64k*` | 1.525 | — | 4.933 |
| `T4_bpe_split_64k*` | 1.533 | 1.320 | 5.105 |

### 64k arms, all four corpora (fertility / bytes-per-token)

| arm | Sāmayik test | Sāmayik OOD | Itihāsa test | FLORES devtest |
|---|---|---|---|---|
| `T1_bpe_raw_64k_dcs` | 2.396 / 3.151 | 2.470 / 3.748 | 2.026 / 4.663 | 2.156 / 3.807 |
| `T2_unigram_raw_64k_dcs` | 2.665 / 2.835 | 2.701 / 3.428 | 2.204 / 4.287 | 2.488 / 3.301 |
| `T4_bpe_split_64k_oracle_dcs` | 2.386 / 3.281 | 2.452 / 3.895 | 2.066 / 4.822 | 2.101 / 3.999 |
| `T4_unigram_split_64k_oracle_dcs` | 2.514 / 3.114 | 2.572 / 3.712 | 2.140 / 4.656 | 2.265 / 3.710 |
| `T5_morphbpe_raw_64k_dcs` | 2.517 / 3.000 | 2.663 / 3.477 | 2.491 / 3.793 | 2.333 / 3.518 |
| `T6_morphbpe_split_64k_dcs` | 2.539 / 3.082 | 2.682 / 3.560 | 2.567 / 3.881 | 2.318 / 3.624 |

**The constraint costs compression, consistently.** At 64k on Sāmayik test, `T5` gets 3.000
bytes/token against `T1_dcs`'s 3.151 (−4.8%) and `T6` gets 3.082 against `T4_oracle`'s 3.281
(−6.1%); on Itihāsa the gap is wider (3.793 vs 4.663, −18.7%). That is the same fact as the
positive TPP deltas, seen from the character side: forbidding merges across gold boundaries
removes exactly the long merges a compressor wants.

---

## Text invariants

Every transformation this experiment measures on is recorded, per CLAUDE.md and the
Experiment 03 finding that made it necessary.

| transformation | letter retention | non-letter multiset preserved | missing (gross) | added (gross) |
|---|---|---|---|---|
| DCS held-out: raw → oracle split | 1.0262 | no | 2,471 | 105 |
| Sāmayik test: raw → ByT5-reconciled | 1.0138 | no | 134 | 12 |
| Sāmayik OOD: raw → ByT5-reconciled | 1.0096 | no | 397 | 15 |
| Itihāsa test: raw → ByT5-reconciled | 1.0207 | no | 1,806 | 3 |
| FLORES devtest: raw → ByT5-reconciled | 1.0058 | no | 80 | 5 |

**The DCS oracle split is a re-analysis, not a re-segmentation**, so two of these checks do
not apply to it in the sense they were written for. Its letter retention is 1.0262 because
reversing sandhi legitimately restores elided phonemes, and
`letters_out_subset_of_raw_union_model` is `false` for the same reason — the check asks
whether the output invents letters neither source supplies, and DCS's unsandhied forms
*do* supply letters the surface does not. This is why the oracle split is used for
**training** the `T4_oracle`/`T6` arms and for MorphScore, and never as a text an arm is
credited with compressing. The reconciled rows are Experiment 03's, unchanged, and are what
the deletion-cost column in the TPP tables prices.

---

## Verdicts

**H3, MorphScore half — the constraint half is SUPPORTED; the splitting half is NOT TESTED
here.** Forbidding merges across gold boundaries raises boundary F1 on the human-verified
held-out subset in both matched comparisons, at both vocabulary sizes: `T5` over `T1_dcs` by
+0.023 at segment granularity and +0.266 at stem granularity (64k), and `T6` over
`T4_oracle_dcs` by +0.360 at stem granularity (64k) — a 3.1× increase. Whether *sandhi
splitting alone* raises MorphScore, which is what H3 literally asserts, cannot be read off
these tables: the only granularity a split arm has is scored over a different population of
units (gold segments) from the raw arm's (surface words), so the comparison would be between
two corpora rather than between two tokenizers.

**H4, TPP half — NOT SUPPORTED, decisively and in the wrong direction.** The proposed method
costs tokens rather than saving them on all four corpora at both sizes. `T6` − `T1_dcs` on
Sāmayik test is **+0.0847** (64k) and **+0.0593** (32k), CIs [+0.0772, +0.0920] and
[+0.0519, +0.0659] — 5.3% and 3.5% *more* tokens per unit of meaning. The constraint alone
is the same story (`T5` − `T1_dcs`, +0.0810 at 64k) and so is the constraint applied on top
of gold splitting (`T6` − `T4_oracle`, +0.1025 at 64k). Not one of the 24 constrained deltas
in this experiment (six constrained pairs on four corpora) is negative, and not one CI includes 0. The only negative deltas anywhere
are **gold splitting without the constraint** — `T4_oracle` − `T1_dcs` — which saves a small
amount on prose (−0.0291 / −0.0178 on Sāmayik test at 32k / 64k, −0.0117 / −0.0091 on OOD)
and on FLORES (−0.0325 / −0.0370), and is adverse on verse. That reproduces Experiment 03's
finding with a gold splitter in place of a model one, and it is the upper bound on what
splitting can buy.

**What this does not settle.** H4 is a claim about tokens to a reference bits-per-character,
not about tokens per proposition. A vocabulary that costs 5% more tokens can still reach a
reference BPC sooner if its tokens are correspondingly easier to predict — which is exactly
the trade a morphologically-aligned vocabulary would be expected to make, and exactly what
the MorphScore and violation results say `T6` has bought. Experiment 05 measures BPC and
tokens-to-reference-loss, and it is the experiment that decides H4. The honest summary of
Experiment 04 is: **the constraint does what it is supposed to do to the segmentation, and
it is not free.**

---

## Caveats

1. **Domain mismatch: DCS trains, parallel corpora evaluate.** Every arm in the paired
   deltas is trained on DCS — classical Sanskrit across 258 texts — and evaluated on modern
   prose (Sāmayik), epic verse (Itihāsa) and translated news (FLORES). The mismatch is
   identical for both sides of every pair, so it cannot produce a delta, but it does mean the
   absolute TPP levels are not what these arms would achieve in domain.
2. **Oracle split at training, ByT5 split at evaluation.** The `T4_oracle`/`T6` arms are
   trained on DCS's *gold* segmentation and evaluated on the *ByT5-reconciled* split of the
   parallel corpora, which is the only split those corpora have. A split arm therefore meets
   a segmentation at evaluation that is not the one it learned on. This is a real limitation
   of the TPP half and is stated in the decisions log; MorphScore has no such mismatch,
   since it is computed on the DCS held-out oracle split.
3. **Stem boundaries are a heuristic.** The stem/ending boundary is the longest common
   prefix of a segment and its lemma, kept only when ≥ 2 characters and shorter than the
   segment. Every stem-granularity number is labelled heuristic and no claim rests on it
   alone; the segment granularity is primary and is where the H3 claim is read.
4. **The fused-sandhi boundary convention shifts boundaries by up to one character.** Vowel
   sandhi fuses two characters into one in 53.4% of multi-segment words, so no character
   offset is *the* boundary; `align_segments` places it on one side or the other depending
   on the alignment, and the direction is not uniform (`docs/decisions.md`, "Correction: the
   fused sandhi character does not consistently join the left segment"). The convention is
   **identical for every arm**, so paired comparisons are unaffected; the absolute level is
   not portable to a published MorphScore on a language without sandhi. The ±1 column bounds
   the effect: it roughly doubles every raw arm's exact score.
5. **±1 on split text is not the sandhi correction.** Segments in the oracle split are
   whitespace-separated, so sandhi cannot displace a boundary there. The ±1 column for a
   split arm is a sensitivity bound on the LCP stem heuristic and nothing else; the earlier
   decision that split-text MorphScore is "exact only" governs the claims, and the tolerant
   column is reported as extra information.
6. **68% of DCS's segmentation is machine-generated.** Only 6,952 of 30,150 held-out
   sentences (23.06%) carry no `UnsandhiedReconstructed=True` token. Both populations are
   reported above; the conclusions are unchanged between them (e.g. `T6` 64k stem F1 0.5354
   verified vs 0.5570 over all, `T4_oracle` 64k 0.1751 vs 0.1571), so the effect is not an
   artefact of the annotator's own tokenizer-like segmenter.
7. **The English pivot is cross-corpus.** `E1_bpe_64k` was trained on the English side of
   the parallel corpora, not on anything DCS-sized or DCS-like. It cancels from the paired
   deltas, which is why they carry the verdict; it does not cancel from the levels, which is
   why the levels do not.
8. **Verse is a confound.** Itihāsa is śloka: meter constrains word choice, so its numbers
   are reported second everywhere and no conclusion rests on them. They are also where the
   constraint's cost is largest (+0.1877 for `T6` − `T1_dcs` at 64k).
9. **The constrained arms cost compression**, by roughly 5–20% bytes per token depending on corpus
   and pair. That is not a side effect to be tuned away; it is the same measurement as the
   TPP deltas.
10. **The violation rate's numerator is tokens, its denominator boundaries.** A token
    spanning two gold boundaries counts once, so `violations_per_boundary` is a close lower
    bound on the fraction of boundaries crossed rather than that fraction exactly. It is
    still the right rate for comparing two arms, because a per-token rate rewards the
    constrained arm for being more verbose.
11. **The twelve arm `results.json` files record `git_dirty: true`.** They were written
    during Task 3 with uncommitted work in the tree, so their `git_commit` (`e2a8f73`) names
    the parent commit rather than the code that produced them. The arms themselves are
    pinned by the sha256 of each `tokenizer.json` in this experiment's
    `tokenizer_sources`, which is the tie between a number here and the artifact that
    produced it. This file's own run is at `7f49aaa` with a clean tree.

---

## Artifacts

```
outputs/04_morph_constrained/
├── results.json                # every number above, strict JSON
├── config.yaml                 # byte copy of the config that produced it
├── constraint_effects.pdf/.png # the figure
├── run.log                     # the run's own log
└── marked/                     # held-out t5/t6 marked text, written for the audit
```

`results.json` keys: `experiment`, `git_commit`, `git_dirty`, `timestamp`, `config`,
`tokenizer_sources`, `unavailable_arms`, `dcs_heldout`, `corpora`, `exclusion_check`,
`exclusion_check_en`, `spans_coverage`, `morphscore`, `violations`, `tpp`, `tpp_delta`,
`fertility_primary`, `fertility_secondary`, `compression`, `text_invariants`.
