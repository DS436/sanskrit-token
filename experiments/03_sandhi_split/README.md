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

Within the primary table, **only `value` is comparable between a raw arm and a split
arm.** The two are computed by different metrics attaching different distributions, so
`results.json`'s `mean`/`std` summarise different lists: per-*word* token counts for a raw
arm (`distribution: per_word`, one entry per word) and per-*sentence* ratios for a split
arm (`distribution: per_text`, one entry per sentence). `value` is tokens over raw words
in both cases and is the number the tables show; `n` is that shared denominator (raw
words) and `n_texts` the number of sentences measured, so the summarised list's length is
recorded either way. Read a `mean` column across the two families and it is comparing a
mean over words with a mean over sentences.

**Success (pre-registered) — met on prose.** The criterion was: on Sāmayik test, each
split arm's controlled TPP is below its matched raw arm's and the paired-bootstrap CI on
the difference excludes 0. **All four pairs meet it**, with deltas of −0.051 to −0.083 and
every CI clear of 0. Across all four corpora, 15 of 16 matched pairs move the predicted
way with the CI excluding 0; the sixteenth (Itihāsa, unigram 64k) moves the *wrong* way,
also with its CI excluding 0.

**Scope.** Four matched pairs, `T1`/`T2` (raw) against `T4` (split) at BPE and Unigram,
32k and 64k. Every arm is **provisional** in the same sense as Experiment 02's: trained on
the Sanskrit (or English) side of two parallel corpora, not on the monolingual corpus of
milestone M1. `T4` measures **sandhi *and* compound (samāsa) splitting**, because the
splitter does both — see **The splitter** below; every table and the paper must name it
that way.

**Run:** `uv run python experiments/03_sandhi_split/run.py`

**Runtime:** 55 s wall-clock (2026-09-05, commit `d6ddeee`, clean tree) with every corpus,
tokenizer and split jsonl already on disk — no network, and the splitter model is never
loaded. That is the whole point of the cache: the 9.4 h of model time is Task 3's, spent
once. Producing the inputs took that 9.4 h split run plus 17.0 s of `T4` tokenizer
training (2.1 / 2.6 / 7.5 / 4.8 s for BPE 32k, BPE 64k, Unigram 32k, Unigram 64k).

---

## Status

| Task | State |
|---|---|
| 1. Shared experiment helpers (`sanskrit_tok.experiment`) | done |
| 2. `SandhiSplitter` + cache + throughput benchmark | done |
| 3. Split the corpora, train the `T4` arms | done (`d6ddeee`) |
| 4. Runner, `results.json`, figure, results table | done |

All arms loaded (`unavailable_arms` is empty) and all four corpora passed both leakage
checks with `n_missing: 0` on the Sanskrit and the English side.

## Summary

**On Sāmayik test — the primary prose corpus — sandhi-splitting the training text lowers
tokens-per-proposition for every matched pair, and the paired CI excludes 0 every time.**
Against the matched English control `E1`, BPE 32k goes 1.077 → 1.003 (Δ **−0.0739**
[−0.0800, −0.0675]), BPE 64k 1.027 → 0.976 (Δ −0.0505 [−0.0567, −0.0447]), Unigram 32k
1.148 → 1.065 (Δ −0.0833 [−0.0901, −0.0765]) and Unigram 64k 1.096 → 1.036 (Δ −0.0606
[−0.0675, −0.0543]). Nothing about the English side changed between the two columns: the
same `E1` arm scores the same English sentences, so the entire move is on the Sanskrit
side, and the delta is a measurement of what splitting the *training* text bought.

**One arm now sits below parity on prose under the matched control.**
`T4_bpe_split_64k` reads **0.976 [0.963, 0.990]** on Sāmayik test — the whole interval
below 1.0. That is the criterion Experiment 02 pre-registered for H2 and failed with the
raw-subword arms, where every matched pair on this corpus sat above 1.0. It is one arm of
four, on one corpus, from a provisional tokenizer, so it is a foothold and not a result
about Sanskrit; but it is the first time in this project that a Sanskrit-native tokenizer
has beaten its matched English control on prose.

**The direction is near-universal, and its one exception is informative.** Fifteen of the
sixteen matched pairs across four corpora show a negative delta whose CI excludes 0. The
exception is Itihāsa `unigram 64k`: **+0.0090 [+0.0076, +0.0104]** — splitting made it
*worse*, significantly. Itihāsa is verse, its raw Unigram arms were already the strongest
on that corpus, and a 64k Unigram vocabulary learned on split text apparently spends
capacity it cannot use there. No pair anywhere has a CI containing 0, so nothing here is
undecided; one thing is simply decided the other way.

**Fertility moves with TPP, and never leads.** Over the same raw-word denominator, the
split arms are cheaper on 15 of 16 arm-corpus pairs — on Sāmayik test 1.70 → 1.60 (BPE
32k) down to 1.68 → 1.60 (Unigram 64k) — with the same Itihāsa `unigram 64k` exception
(1.877 → 1.904). Read the secondary fertility (tokens per *split* word: 1.33 on Sāmayik
test for BPE 32k) as a different measurement, not an improvement: splitting multiplied the
words.

**Against deployed practice the picture is the same and larger.** Every split arm is
cheaper than its raw twin against `T0_o200k` on all four corpora — on Sāmayik test
`T4_bpe_split_64k` costs 0.863 [0.850, 0.877] English tokens per proposition against
`T1_bpe_raw_64k`'s 0.908 — but that comparison mixes language with a 200k general-domain
vocabulary and is never the one the verdict is read from (CLAUDE.md §2.5).

## Results

TPP is Sanskrit tokens / English tokens over aligned sentence pairs, against the matched
English control `E1_*`; **Δ** is split minus raw with a paired bootstrap over the same
sentence pairs (1000 draws, seed 0). Fertility columns are the primary form — tokens over
the **raw** sentence's word count for both arms — and never lead. All arms provisional.

### Sāmayik test (prose, primary) — 2,417 sentences

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | Δ CI excludes 0 | fert raw | fert split |
|---|---|---|---|---|---:|---:|
| bpe 32k | 1.077 [1.063, 1.091] | 1.003 [0.989, 1.017] | **−0.0739** [−0.0800, −0.0675] | yes | 1.70 | 1.60 |
| bpe 64k | 1.027 [1.014, 1.040] | **0.976** [0.963, 0.990] | **−0.0505** [−0.0567, −0.0447] | yes | 1.52 | 1.47 |
| unigram 32k | 1.148 [1.134, 1.164] | 1.065 [1.051, 1.080] | **−0.0833** [−0.0901, −0.0765] | yes | 1.80 | 1.68 |
| unigram 64k | 1.096 [1.082, 1.111] | 1.036 [1.023, 1.050] | **−0.0606** [−0.0675, −0.0543] | yes | 1.68 | 1.60 |

`T4_bpe_split_64k`'s interval is the only one entirely below 1.0; `T4_bpe_split_32k`
straddles it (1.003 [0.989, 1.017]) and the two Unigram split arms remain above.

### Sāmayik test_ood (prose, out-of-domain for both sides) — 4,047 sentences

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | Δ CI excludes 0 | fert raw | fert split |
|---|---|---|---|---|---:|---:|
| bpe 32k | 1.091 [1.078, 1.103] | **0.938** [0.927, 0.949] | **−0.1529** [−0.1569, −0.1487] | yes | 2.34 | 2.01 |
| bpe 64k | 1.067 [1.054, 1.078] | **0.923** [0.912, 0.934] | **−0.1435** [−0.1475, −0.1396] | yes | 2.16 | 1.87 |
| unigram 32k | 1.161 [1.146, 1.174] | 0.988 [0.976, 1.000] | **−0.1726** [−0.1768, −0.1682] | yes | 2.53 | 2.16 |
| unigram 64k | 1.146 [1.132, 1.159] | **0.976** [0.964, 0.988] | **−0.1702** [−0.1743, −0.1659] | yes | 2.39 | 2.03 |

The largest deltas in the experiment, and three of the four split arms fall entirely below
1.0. Read that with care: `test_ood` is Mann Ki Baat transcripts, out-of-domain for both
sides, and the splitter changed the whitespace-unit count of 89% of its sentences — the
most of any prose corpus here.

### Itihāsa test (verse, secondary — meter is a confound) — 11,721 sentences

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | Δ CI excludes 0 | fert raw | fert split |
|---|---|---|---|---|---:|---:|
| bpe 32k | 0.653 [0.650, 0.657] | 0.626 [0.622, 0.630] | **−0.0275** [−0.0290, −0.0261] | yes | 1.94 | 1.86 |
| bpe 64k | 0.607 [0.604, 0.610] | 0.597 [0.594, 0.601] | **−0.0099** [−0.0113, −0.0085] | yes | 1.75 | 1.72 |
| unigram 32k | 0.663 [0.660, 0.667] | 0.657 [0.653, 0.660] | **−0.0064** [−0.0078, −0.0049] | yes | 2.02 | 2.00 |
| unigram 64k | 0.626 [0.622, 0.629] | 0.635 [0.631, 0.638] | **+0.0090** [+0.0076, +0.0104] | yes (**wrong way**) | 1.88 | 1.90 |

Every arm here sits far below 1.0, split or not, and that says nothing about splitting:
Itihāsa is śloka against a 19th-century English verse translation, so the denominator is
inflated for reasons that have nothing to do with tokenization. What splitting bought is
the delta column, and it is the smallest of any corpus — and negative for three pairs,
positive for the fourth.

### FLORES devtest (tertiary) — 1,012 sentences

| Pair | raw TPP [95% CI] | split TPP [95% CI] | Δ [95% CI] | Δ CI excludes 0 | fert raw | fert split |
|---|---|---|---|---|---:|---:|
| bpe 32k | 1.139 [1.128, 1.150] | 1.064 [1.052, 1.076] | **−0.0750** [−0.0803, −0.0699] | yes | 2.08 | 1.95 |
| bpe 64k | 1.138 [1.126, 1.149] | 1.061 [1.048, 1.073] | **−0.0763** [−0.0819, −0.0711] | yes | 1.94 | 1.81 |
| unigram 32k | 1.207 [1.194, 1.220] | 1.116 [1.104, 1.127] | **−0.0914** [−0.0982, −0.0848] | yes | 2.28 | 2.11 |
| unigram 64k | 1.219 [1.206, 1.232] | 1.128 [1.116, 1.140] | **−0.0911** [−0.0981, −0.0848] | yes | 2.15 | 2.00 |

Consistent deltas, and every arm still above 1.0: Wikipedia-register text is out-of-domain
for a tokenizer trained on Sāmayik and Itihāsa, and splitting does not close that gap.

### Deployed practice (vs `T0_o200k`, 200k general-domain)

Reported, never a controlled comparison (CLAUDE.md §2.5): this pivot differs from the
Sanskrit arms in vocabulary size and training domain as well as language.

| arm | Sāmayik test | Sāmayik test_ood | Itihāsa test | FLORES devtest |
|---|---|---|---|---|
| T1_bpe_raw_32k | 1.009 [0.995, 1.023] | 1.156 [1.143, 1.169] | 0.524 [0.521, 0.526] | 1.320 [1.304, 1.335] |
| T4_bpe_split_32k | 0.940 [0.926, 0.954] | 0.994 [0.982, 1.005] | 0.502 [0.499, 0.504] | 1.233 [1.218, 1.248] |
| T1_bpe_raw_64k | 0.908 [0.896, 0.921] | 1.066 [1.053, 1.077] | 0.472 [0.469, 0.474] | 1.227 [1.212, 1.241] |
| T4_bpe_split_64k | 0.863 [0.850, 0.877] | 0.922 [0.911, 0.933] | 0.464 [0.461, 0.466] | 1.145 [1.130, 1.159] |
| T2_unigram_raw_32k | 1.070 [1.056, 1.086] | 1.251 [1.236, 1.265] | 0.546 [0.543, 0.548] | 1.445 [1.427, 1.462] |
| T4_unigram_split_32k | 0.993 [0.978, 1.007] | 1.065 [1.052, 1.077] | 0.540 [0.537, 0.543] | 1.336 [1.318, 1.352] |
| T2_unigram_raw_64k | 0.999 [0.985, 1.014] | 1.180 [1.166, 1.193] | 0.506 [0.504, 0.509] | 1.363 [1.346, 1.379] |
| T4_unigram_split_64k | 0.944 [0.929, 0.958] | 1.005 [0.993, 1.017] | 0.514 [0.511, 0.516] | 1.261 [1.244, 1.276] |

Every split arm beats its raw twin here on every corpus, Itihāsa `unigram 64k` included —
the one place the controlled and deployed pivots disagree, and a reminder that a ratio is
only as meaningful as its denominator.

### Secondary variant: the splitter's raw output — **not content-preserving**

The primary `T4` numbers are measured on the **reconciled** text. The table below adds the
same arms measured on the splitter's *unreconciled* output, which keeps only 87.1% of the
raw non-space characters: it drops sentence punctuation, normalises, and sometimes deletes
a transliterated loanword outright. Controlled TPP, `reconciled / model_raw`:

| arm | Sāmayik test | Sāmayik test_ood | Itihāsa test | FLORES devtest |
|---|---|---|---|---|
| T4_bpe_split_32k | 1.003 / *0.819* | 0.938 / *0.839* | 0.626 / *0.532* | 1.064 / *0.949* |
| T4_bpe_split_64k | 0.976 / *0.804* | 0.923 / *0.824* | 0.597 / *0.511* | 1.061 / *0.946* |
| T4_unigram_split_32k | 1.065 / *0.888* | 0.988 / *0.891* | 0.657 / *0.562* | 1.116 / *0.997* |
| T4_unigram_split_64k | 1.036 / *0.871* | 0.976 / *0.881* | 0.635 / *0.548* | 1.128 / *1.009* |

**The italicised column is not a result.** It is 0.09–0.18 lower than the primary
everywhere, and much of that is text the splitter deleted: on Sāmayik test it would have
put every arm below 1.0 and made the headline look four times stronger. It is printed only
to show the size of the bias reconciliation removes (`docs/decisions.md`, "T4 text is the
model's segmentation reconciled against the raw sentence, not the raw model output").

## Figure

`outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` — one row per corpus (prose first),
one x position per matched pair, two markers at each (raw arm circle, split arm square)
with their 95% bootstrap CIs against the matched English control, dashed reference at 1.0.
The error bars drawn are each arm's own; the interval on the **difference** is narrower,
because the two arms are paired on the same sentences, and is what the Δ column reports.
The Itihāsa panel is where the two markers cross for `unigram 64k`.

## The splitter, and what it did to each corpus

`sanskrit_tok.sandhi.SandhiSplitter` wraps `chronbmm/sanskrit5-multitask`
(`@c0d2ada54f3d19903149425aa888a203601423f8`) in segmentation mode (`"S "` prefix, IAST in
and out, 512-byte window), on `mps`. Devanagari goes in, SLP1 with one whitespace unit per
segment comes out, and every result is cached, so this experiment loaded no model at all.
The manifest beside the split text is the **re-run**'s (`n_model: 1`, `n_cache_hits:
136,789`, `n_chunked: 0`): after the reconciliation fix of `9c5a195`, `split_corpora.py`
was run again to regenerate every jsonl from the cache, so all but one sentence was a cache
hit and its counters describe that pass, not the model work. The original 9.4 h run
generated 136,652 sentences and chunked **433** of them — those are the numbers that
describe what the model actually did.

| corpus | sentences | mean units raw | mean units split | sentences with a changed unit count | retention, raw model output | retention, reconciled | units kept verbatim | units replaced by an inexact window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sāmayik test | 2,417 | 9.59 | 11.49 | 67.4% | 0.8737 | 0.9907 | 19.9% | 35.4% |
| Sāmayik test_ood | 4,047 | 11.82 | 15.68 | 89.5% | 0.9208 | 0.9782 | 9.9% | 43.3% |
| Itihāsa test | 11,721 | 11.16 | 15.86 | 97.7% | 0.8694 | 0.9890 | 13.2% | 51.1% |
| FLORES devtest | 1,012 | 16.77 | 19.62 | 79.5% | 0.9242 | 0.9958 | 11.1% | 29.1% |

Retention is pooled non-space SLP1 characters out over characters in. Over all 136,918
sentences the
split job processed (these four corpora plus the training corpus), pooled retention is
0.8706 before reconciliation and 0.9895 after, and 45.7% of raw whitespace units were
replaced by an inexact window — 46.0% over the four evaluation corpora above.

**An "inexact window" is expected, not an error.** Reconciliation replaces a raw
whitespace unit with the model segments that align to it, and records the replacement as
*inexact* when the aligned segments' concatenation is not character-identical to the raw
unit. That is precisely what genuine sandhi reversal looks like: `tadapi` → `tad api`
restores the elided vowel, so the characters legitimately differ, and the replacement
scores below 1.0. The 45.7% therefore counts **units where the model did more than insert
a boundary** — the union of correct sandhi reversal and unwanted rewriting — and it cannot
tell the two apart. It is reported because retention alone cannot: a sentence can read
1.000 retention while one word was dropped and another rewritten to the same length. What
it bounds is how much of the split text is *not* a verbatim re-slicing of the input, which
on this splitter is a lot, and every `T4` number should be read as measured on such text.

**Two training-corpus line counts, and they differ.** The raw corpus
`data/processed/tok_train_slp1.txt` has **117,720** lines; the split corpus
`data/processed/tok_train_slp1_split.txt` has **117,608**. Both come from the same 117,721
selected sentences; each is deduplicated on its own transformed text, and splitting
collapses 113 more pairs into one line than transliteration alone does. So a matched pair
is trained on the same *sentences* but not on exactly the same number of *lines* — a
0.1% difference, recorded here because "matched" should mean what it says.

**Throughput.** The full split run took **9.40 h** on `mps` for 136,918 sentences (4.05
sentences/s sustained), against the 200-sentence benchmark's 7.09 s/s and its 5.4 h
projection — optimistic by 74%, because rate tracks sentence length and the benchmark saw
only short Sāmayik-test prose (`docs/decisions.md`, Task 3's throughput entry).

## Verdict — the TPP/fertility half of H3

**Supported on prose, with one significant counterexample on verse, for provisional arms.**

* **Sāmayik test (primary):** 4/4 pairs negative, all CIs excluding 0. The pre-registered
  criterion is met. `T4_bpe_split_64k` additionally falls entirely below 1.0.
* **Sāmayik test_ood:** 4/4 negative, all CIs excluding 0, and the largest deltas here.
* **Itihāsa test:** 3/4 negative with CIs excluding 0; `unigram 64k` **positive**
  (+0.0090 [+0.0076, +0.0104]) — splitting significantly hurt that arm.
* **FLORES devtest:** 4/4 negative, all CIs excluding 0.
* Fertility over the shared raw-word denominator moves the same way on 15 of 16
  arm-corpus pairs, with the same exception. Fertility does not lead this verdict; it
  agrees with it.

**An arm sitting below 1.0 is not the finding.** Every Itihāsa arm is at 0.60–0.66 and
none of that is attributable to splitting; the level reflects the language pair, the
register and the translation in the denominator. Only the *difference* between two arms
that share a denominator and differ in one thing — whether their training text was split —
supports a claim about splitting, which is why the delta column carries the verdict and
the level does not.

**What this does not show.** Whether `T4`'s boundaries are morphological (MorphScore,
Experiment 04), how much of the gain survives on a monolingual training corpus, or whether
a downstream model trained on these tokenizations is better in any way. It also does not
separate sandhi splitting from compound splitting: this splitter does both.

## Caveats

* **Provisional arms.** Every `T1`/`T2`/`T4`/`E1` arm is trained on the Sanskrit or
  English side of two parallel corpora (117,720 / 117,608 lines), not on the monolingual
  corpus of milestone M1. Absolute levels will move when that corpus exists; the matched
  *differences* are the more durable quantity.
* **The split text is not the input text.** The splitter rewrites as well as re-segments
  (87.1% pooled raw character retention). Reconciliation restores the deletions (98.95%),
  but a `T4` number is still measured over text a model has touched, on 45.7% of units by
  more than a boundary insertion, and the reconciliation threshold (0.6) is a judgement
  call recorded in the manifest.
* **Sandhi *and* samāsa.** `T4` is not "sandhi splitting" — the model splits compounds
  too, so every claim must say "sandhi and compound splitting".
* **Domain fit.** Sāmayik test is in-domain for the training corpus on both sides;
  `test_ood` and FLORES are out-of-domain for both. Experiment 02 showed how much of an
  apparent Sanskrit advantage a mismatched English pivot can manufacture, which is why
  every verdict number here is read against `E1_*` and not against `T0_o200k`.
* **Verse.** Itihāsa is śloka: meter constrains word choice, and its English side is a
  19th-century verse translation. Prose comes first for that reason (CLAUDE.md §2.7), and
  the one adverse pair lives here.
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
| `split.yaml`, `split_corpora.py` | Task 3's corpus split (9.4 h, background, resumable) |
| `benchmark_splitter.py` | Task 2's throughput and behaviour benchmark |
| `outputs/03_sandhi_split/results.json` | every number, with provenance and config |
| `outputs/03_sandhi_split/tpp_split_vs_raw.{pdf,png}` | the figure |
