# Experiment 05 — language-model training and bits per character

**Status: phase A complete — corpora built, the training pipeline written, and the sweep
validated end to end on this laptop's MPS. No research LM has been trained yet: the real
sweep is 51 runs and needs a rented GPU ("How to run on a rented GPU", below).**

## Hypothesis

**H4, verbatim from the outline (`docs/paper_outline.md` §1, RQ4):**

> A model trained with the constrained tokenizer reaches a reference bits-per-character
> with fewer training tokens than the BPE control, in line with the 25–29% speedups
> reported for Hungarian and English.

If sandhi-splitting and morpheme-constrained
merges recover token efficiency that a plain subword tokenizer destroys, then a language
model trained on the **same text** under those tokenizers should reach a lower **bits per
character** on held-out Sanskrit at equal training bytes than one trained under
`T1_bpe_raw_64k_dcs`, and should reach any given BPC after fewer bytes, tokens and FLOPs.

BPC is the headline because it is the only quantity that is comparable across
vocabularies: it is the model's total cross-entropy divided by the number of SLP1
*characters* of the evaluation text. **Perplexity is never reported** (CLAUDE.md §2.2) —
it is per token, and the arms have different tokens.

**What Experiment 04 leads us to expect.** The constrained arms pay a token handicap on
the same text, and it is almost entirely the *heuristic* half of the boundary set that
costs: on held-out DCS at 64k, `T5_morphbpe_raw` costs **+23.5%** tokens and
`T6_morphbpe_split` **+24.1%**, while `T5_morphbpe_rawseg` — constrained on DCS's gold
segment annotation only, no stem heuristic — costs **+5.1%** and its out-of-domain TPP
delta on the primary prose corpus is the one delta in that experiment whose bootstrap
interval includes zero. `T5seg` also gained the most MorphScore (+0.11 F1). Experiment 04's
own conclusion was that **`T5seg`, not `T6`, is the arm to carry forward** — and that H4 is
a live question, because a vocabulary that costs 5% more tokens can still reach a reference
BPC sooner if its tokens are correspondingly easier to predict. That is the trade this
experiment measures. At equal bytes the handicap becomes more steps and more FLOPs, both of
which are recorded per run; equal bytes is the primary comparison and equal tokens the
secondary one.

## The two tracks

Tokenizers are **not** retrained here. Every arm is one of the fixed `_dcs` arms trained in
Experiment 04, so the LM data is the only thing that changes within a track
(docs/decisions.md, 2026-09-05, "Experiment 05 runs in two tracks").

| | Track 1 — matched | Track 2 — scale |
|---|---|---|
| Corpus | DCS training sentences | corpus M1 (DCS + Sāmayik/Itihāsa training sides + Sangraha verified Sanskrit + Sanskrit Wikipedia) |
| Raw arms see | `track1_raw.txt` (sandhied) | `track2_sample.txt` (the ~200 M-token cut of `track2_raw.txt`) |
| Split arms see | `track1_split.txt` (the gold split of the *same* sentences, same order) | — (no gold split exists at this scale) |
| Arms | `T1_bpe_raw_64k_dcs`, `T2_unigram_raw_64k_dcs`, `T4_bpe_split_64k_oracle_dcs`, `T5_morphbpe_rawseg_64k_dcs`, `T5_morphbpe_raw_64k_dcs`, `T6_morphbpe_split_64k_dcs`, `T7_byt5` | the five raw ones |
| Sizes | 50M | 50M and 125M |
| Seeds | 3 | 3 |

Line *i* of `track1_raw.txt` and line *i* of `track1_split.txt` are the same sentence:
both files are written in one lockstep pass and every drop removes the pair.

## Evaluation sets

**Track 1's in-domain set** is `heldout_dcs.txt`, the DCS held-out split — 13 whole texts,
never trained on. **Track 2's is `heldout_sangraha.txt`**: 2,000 Sangraha lines drawn out
of its own training sample, because that sample is 93.6% Sangraha and a DCS-only held-out
set would make every Track 2 number a transfer measurement. On Track 2, `heldout_dcs` is
reported as **transfer** rather than in-domain (docs/decisions.md, 2026-09-06). Both tracks
also report the four out-of-domain parallel sets: the Sanskrit side of Sāmayik test,
Sāmayik test_ood, Itihāsa test and FLORES devtest.

Each parallel set has a `_split` twin (oracle split for DCS, ByT5-reconciled for the
parallel corpora) for the split arms, so every arm is evaluated on the text shape it was
trained on. **Every arm's BPC is nevertheless divided by the RAW twin's character count.**
The split text is 2.1–6.3% longer (undoing sandhi inserts spaces), which is the size of the
effect this experiment is measuring, so dividing by it would hand `T4_oracle` and `T6` a
mechanical discount worth as much as the result. `curve.jsonl` and `results.json` record
`total_nats`, `n_tokens`, `n_chars_scored` (what was predicted) and `n_chars_denominator`
(the raw twin) per set, so any normalisation is re-derivable without retraining
(docs/decisions.md, 2026-09-06, "BPC is bits per character of the RAW held-out text").

| held-out set | lines | SLP1 chars | dropped (non-SLP1) |
|---|---:|---:|---:|
| `heldout_dcs` | 30,137 | 1,314,581 | 0 |
| `heldout_sangraha` | 2,000 | 152,302 | — (drawn from a built corpus) |
| `heldout_samayik_test` | 2,386 | 173,029 | 31 of 2,417 |
| `heldout_samayik_test_ood` | 3,935 | 425,976 | 112 of 4,047 |
| `heldout_itihasa_test` | 11,674 | 1,228,945 | 47 of 11,721 |
| `heldout_flores_devtest` | 917 | 124,799 | 95 of 1,012 |

A parallel pair is dropped when either half carries a character outside SLP1 — after
typographic punctuation has been normalised to ASCII, so the residue is candra vowels of
Hindi loanwords and zero-width joiners rather than curly quotes (docs/decisions.md,
2026-09-06, "Held-out texts rebuilt ..."). Experiment 05's OOD evaluation sets are
therefore still proper subsets of Experiments 02 and 03's, by 1.3–9.4% rather than the
4.8–13.9% before the rebuild; see `data/README.md`.

## Leakage

Every training line passes **both** layers (CLAUDE.md §2.4): the sha256 of its SLP1 form
against `data/exclusion_hashes.txt`, and the 24-letter shingle index of every evaluation
source. A hit drops and counts the line; the per-source counts are in the manifests. See
`data/README.md` for the numbers.

`heldout_sangraha.txt` is drawn from the training sample rather than from an external
corpus, so it is filtered the other way round: the 2,000 drawn lines are removed from
`track2_sample.txt`, **and so is every remaining sample line sharing a 24-letter shingle
with one of them** — 5,755 of them, which is what Sangraha's repeated scans of the same
printed passages look like. Their `sentence_hash_slp1` digests then go into
`data/exclusion_hashes.txt` as the `sangraha_heldout` source, so no future corpus can pick
them up either.

## What Task 1 built

| Corpus | Lines | SLP1 chars | Bytes | Tokens (`T1_bpe_raw_64k_dcs`) |
|---|---|---|---|---|
| `track1_raw.txt` | 652,007 | 31,535,529 | 31,535,529 | 6,157,847 |
| `track1_split.txt` | 652,007 | 33,489,418 | 33,489,418 | 6,503,434 |
| `track2_raw.txt` (M1) | 32,164,442 | 2,504,067,955 | 2,504,067,955 | 669,967,173 |
| `track2_sample.txt` | 9,854,981 | 754,169,567 | 754,169,567 | 199,877,664 |

The sample is 7,755 lines smaller than as first built (9,862,736): the 2,000 lines held out
as `heldout_sangraha.txt` and the 5,755 near-duplicates of them.

70.1 min warm plus ~4 min for the sample; `data/README.md` has the per-source counts, the
licences, the sample's composition and every drop. Track 2's sample is ~24× Track 1 by
characters, so the two tracks answer different questions: Track 1 is the controlled
comparison at ~6M tokens multi-epoch, Track 2 the scale run at ~200M.

Chars equal bytes in every row because `quality.is_clean_slp1` is applied to every written
line: the corpora are pure ASCII.

Three caveats to carry into any Track 2 number. Sangraha's verified Sanskrit is OCR of
printed books; the calibrated quality filter (docs/decisions.md, 2026-09-05) removes 26.9%
of its lines but the residual failure modes — character substitution, run-together words —
are not filtered. 14.5% of what survived that filter was an exact duplicate of an earlier
line, which the deduplication removed but which says something about what the rest looks
like. And the sample is 93.6% Sangraha by bytes, so a Track 2 BPC number is largely a
number about OCR'd print.

**What Track 2 actually measures, stated as a limitation.** A hand read of sixty sample
lines found Pali, a chapter-and-verse-numbered Bible translation, and residual OCR salad
that no rule in `quality.py` can catch — the filter can require Devanagari, plausible word
lengths and SLP1-spellable characters, and none of those distinguishes Sanskrit from a
closely related Indic language in the same script. Track 2 is therefore best described as
**Devanagari Indic web and book text dominated by Sanskrit**, not as curated Sanskrit, and
every Track 2 BPC should be read that way. Track 1 is the controlled corpus; Track 2 tests
whether the Track 1 ordering survives two orders of magnitude more, and noisier, text.

## Training protocol

From the outline (`docs/paper_outline.md` §6) and the decision entry that fixed the details
(docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and comparison protocol"):

- **Architecture:** decoder-only GPT-2 style, nanoGPT's `model.py` vendored byte-for-byte
  at commit `f08abb45bd22` under `src/sanskrit_tok/lm/model.py` with its MIT licence.
- **Sizes: the labels "50M" and "125M" follow the GPT-2 convention of counting parameters
  *with* embeddings; the transformer bodies are 25.2M and 84.9M.** 8 layers × 512 × 8 heads
  gives 25,174,528 non-embedding parameters and **58,499,584 total** with a 64k vocabulary;
  12 × 768 × 12 gives 84,953,856 and **134,941,440 total**. The body is what is held fixed
  across arms — that is the point of counting it separately — while the embedding table
  grows with the vocabulary, so `T7_byt5` at the same size is 25.9M / 86.0M total.
  `results.json` records both counts and says which one `flops_est = 6 × params × tokens`
  used (the total).
- **Context** 1024 tokens; AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, linear
  warmup then cosine decay to 10% of the peak rate; bf16 on CUDA, float32 on MPS and CPU.
- **Data:** identical text for every arm within a track; the tokenizer is the only
  variable. One EOS per line, id = the arm's `vocab_size`.
- **Seeds:** three (CLAUDE.md §2.6), `random` / `numpy` / `torch` all set from the config
  and recorded; mean ± sample standard deviation reported.
- **Comparison:** at equal training **bytes**, with tokens and FLOPs recorded, and
  secondarily at equal tokens. **BPC is the headline; perplexity is never reported**
  (CLAUDE.md §2.2).
- **Tokens/bytes/FLOPs to reference BPC:** the reference is `T1_bpe_raw_64k_dcs`'s **best**
  in-domain BPC on its own curve, *at the same track and size*, matched seed by seed;
  `null` when an arm never reaches it. Best rather than final, because Track 1 is eight
  epochs over a 31 MB corpus and a baseline that overfits in its last epoch would otherwise
  hand every other arm an easier target. Both the final and the best BPC of every arm on
  every set are reported (`final_bpc`, `best_bpc`, `best_step`); the **headline table is
  final BPC**.
- **Evaluation cadence:** every 25 steps at 50M and every 100 at 125M, so a crossing is
  resolved to a few percent of the budget rather than to a tenth of it; ~1.6% of the FLOPs.
  Note that the arms have **different step counts** (the byte budget buys a denser arm more
  tokens), so a fixed `warmup_steps` is a different *fraction* of each arm's schedule —
  12% of `T4_oracle`'s 736 steps and 2.5% of `T7_byt5`'s 3,929. The schedules are matched
  on bytes, which is the protocol, not on shape.

## The equal-bytes budget

The budget is one number per track — the **raw** corpus's bytes × `epochs` — and every arm
in that track spends that same number of bytes on the text it actually trains on. Each arm
converts it into its own token budget with its own bytes-per-token, measured on the encoded
corpus:

    max_tokens = round(max_bytes / bytes_per_token)

Track 1 is 8 × 31,535,529 = **252,284,232 bytes**; Track 2's sample is **754,169,567
bytes**, one pass. The conversion is what the experiment is about: a denser vocabulary
turns the same text into more tokens, so at equal bytes it takes more steps and more FLOPs.
Measured (Track 1, from the sweep's own dry run):

| arm | text | bytes/token | tokens at 252,284,232 bytes | vs `T1` |
|---|---|---:|---:|---:|
| `T1_bpe_raw_64k_dcs` | raw | 4.631 | 54,478,832 | — |
| `T2_unigram_raw_64k_dcs` | raw | 4.002 | 63,044,940 | +15.7% |
| `T4_bpe_split_64k_oracle_dcs` | split | 5.227 | 48,264,454 | −11.4% |
| `T5_morphbpe_rawseg_64k_dcs` | raw | 4.439 | 56,830,864 | +4.3% |
| `T5_morphbpe_raw_64k_dcs` | raw | 3.880 | 65,029,911 | +19.4% |
| `T6_morphbpe_split_64k_dcs` | split | 4.142 | 60,908,210 | +11.8% |
| `T7_byt5` | raw | 0.980 | 257,500,288 | +373% |

Two things this table makes explicit. **`T7_byt5` dominates the sweep's cost**: at ~1 byte
per token the equal-bytes protocol hands it four to five times the tokens of a 64k arm.
That is the correct treatment — it sees the same text — and it is why the byte arm's rows
are the long ones in the projection below. And a **split arm's bytes are bytes of the split
text**, which is ~6% larger than the raw corpus because undoing sandhi inserts spaces; the
same byte figure therefore buys a split arm ~7.5 passes over the sentences rather than 8.
Equal bytes of the *raw* text is the alternative and is deliberately not what is used
(docs/decisions.md, 2026-09-06, "Experiment 05 sweep: the byte budget, the shape a split
arm is scored on, and what a crossing means").

## What the sweep writes

Per run, under `outputs/05_lm_training/<sweep>/<track>/<size>/<arm>/seed<k>/`:
`results.json` (final BPC per evaluation set with its `total_nats`, `n_tokens`,
`n_chars_scored` and `n_chars_denominator`, plus params, tokens, bytes, FLOPs, throughput,
hardware, config, provenance), `config.yaml`, `curve.jsonl` (one row per evaluation: step,
tokens, bytes, FLOPs, loss, lr, BPC per set, and a `bpc_detail` block with the same five
numbers behind each BPC) and `sweep_run.json` (the plan hash — which now includes the
tokenizer file's sha256, so a retrained vocabulary re-runs its runs — the evaluation-set
labels, and the budget arithmetic: `max_bytes`, `bytes_per_token`, `max_tokens`). `ckpt.pt`
is deleted once the results are final unless `keep_checkpoints: true`.

Per sweep, in `outputs/05_lm_training/<sweep>/`: `results.json` (per track × size × arm:
final and best BPC as mean ± sd over seeds for every set, the seed-mean curve, and the
bytes/tokens/FLOPs to the reference threshold), `bpc_curves.{pdf,png}` (BPC against training
bytes, one panel per track × size, shaded ± 1 sd, dashed line at the reference's **best**
in-domain BPC) and `bpc_final.{pdf,png}` (final BPC per arm per evaluation role, with
vocabulary and the `oracle` / `provisional` caveats on the labels).

Everything under `outputs/` is gitignored; the numbers quoted in this file are the record.
The one exception is the smoke sweep, whose aggregated `results.json`, `config.yaml` and
figures are tracked at [`results/05_lm_training/smoke/`](../../results/05_lm_training/smoke/)
and labelled there as a pipeline validation rather than a result; see
[`results/README.md`](../../results/README.md).

## Smoke sweep — pipeline validation, NOT a result

Re-ran 2026-09-06 on this laptop (Apple MPS, float32, torch 2.14.0,
macOS-26.6.2-arm64) at commit `b13f6e0`:
`--sweep smoke`, three arms, one seed, 300 steps of the **`smoke` model — two layers, 128
wide, 0.47–8.6M parameters** — on `track1_raw.txt`, evaluating every 100 steps on
`heldout_dcs` (capped at 200,000 characters) and `heldout_samayik_test` (173 KB, so
uncapped in practice). Total wall clock **89 s of training** (42.4 + 42.4 + 4.2 s), plus
~1 min of first-time corpus encoding per arm.

| arm | vocab | params (total) | tokens | bytes seen | BPC `heldout_dcs` | BPC `heldout_samayik_test` | tok/s (before → after) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `T1_bpe_raw_64k_dcs` | 64,001 | 8,626,816 | 614,400 | 2,845,205 | **2.8259** | 4.6140 | 12,352 → **14,504** |
| `T5_morphbpe_rawseg_64k_dcs` | 64,001 | 8,626,816 | 614,400 | 2,727,452 | **2.8588** | 4.6544 | 12,301 → **14,492** |
| `T7_byt5` | 257 | 467,584 | 614,400 | 601,954 | **3.7855** | 4.4280 | 155,252 → **147,952** |

**Every BPC is bit-identical to the run before the fix wave**, which is the expected result
and the check worth having: all three of these are raw arms, so the raw-character
denominator is the denominator they already had, and `forward_logits` is the vendored
forward with a discarded loss removed, not a different computation. The gain is throughput:
**+17.4%** and **+17.8%** on the two 64k arms, from not computing a `batch × 256 × 64,064`
softmax that was thrown away. `T7_byt5` loses 4.7%, which is measurement noise on a 4-second
run whose 320-column softmax was never the bottleneck.

BPC falls monotonically in every arm (`T1` 3.5228 → 2.8744 → 2.8446 → 2.8259; `T7` 8.2361 →
3.9841 → 3.8198 → 3.7855), the aggregation produced both figures, and `T5seg` and `T7`
never reach `T1`'s best in-domain BPC, so their `to_reference` entries are `null` — the
undefined branch is exercised by real data, not only by a test. On these monotone curves
the best BPC *is* the final one, so the threshold change is not exercised here; the
aggregation tests cover the overfitting case that motivates it.

**None of these numbers say anything about tokenizers.** The model is three orders of
magnitude smaller than the research sizes, the budget is 300 steps, and — because the smoke
sweep is budgeted in *steps* rather than bytes — the three arms did not even see the same
amount of text (2.85 MB, 2.73 MB and 0.60 MB respectively). `T7_byt5`'s 155k tokens/s is
likewise an artefact of the `smoke` size: with a 320-row embedding its softmax is free,
while the 64k arms spend most of their time there. At 50M and 125M the transformer body
dominates and that gap will close.

## Projected cost of the real sweep

`sweep.py --dry-run --tokens-per-s 12352` (the pre-`forward_logits` MPS rate on the 64k
smoke arm, kept so the two projections are comparable; the measured rate is now 14,504),
`--gpu-speedup 50`:

```
projection at 12,352 tokens/s; the x50 column is an assumption about a rented GPU, not a measurement
track   size  arm                          seed         tokens     B/tok    PFLOPs   h @ rate    h @ x50
--------------------------------------------------------------------------------------------------------
track1  50M   T1_bpe_raw_64k_dcs              0     54,478,832    4.631       19.1       1.23       0.02
track1  50M   T2_unigram_raw_64k_dcs          0     63,044,940    4.002*      22.1       1.42       0.03
track1  50M   T4_bpe_split_64k_oracle_dcs     0     48,264,454    5.227*      16.9       1.09       0.02
track1  50M   T5_morphbpe_rawseg_64k_dcs      0     56,830,864    4.439       19.9       1.28       0.03
track1  50M   T5_morphbpe_raw_64k_dcs         0     65,029,911    3.880*      22.8       1.46       0.03
track1  50M   T6_morphbpe_split_64k_dcs       0     60,908,210    4.142*      21.4       1.37       0.03
track1  50M   T7_byt5                         0    257,500,288    0.980       40.0       5.79       0.12
--------------------------------------------------------------------------------------------------------
TOTAL track1 (21 runs)                       21  1,818,172,497               486.9      40.89       0.82

track2  50M   T1_bpe_raw_64k_dcs              0    209,588,934    3.598*      73.6       4.71       0.09
track2  50M   T2_unigram_raw_64k_dcs          0    234,073,294    3.222*      82.2       5.26       0.11
track2  50M   T5_morphbpe_rawseg_64k_dcs      0    216,622,970    3.481*      76.0       4.87       0.10
track2  50M   T5_morphbpe_raw_64k_dcs         0    228,375,865    3.302*      80.2       5.14       0.10
track2  50M   T7_byt5                         0    764,080,981    0.987*     118.6      17.18       0.34
track2  125M  T1_bpe_raw_64k_dcs              0    209,588,934    3.598*     169.7       4.71       0.09
track2  125M  T2_unigram_raw_64k_dcs          0    234,073,294    3.222*     189.5       5.26       0.11
track2  125M  T5_morphbpe_rawseg_64k_dcs      0    216,622,970    3.481*     175.4       4.87       0.10
track2  125M  T5_morphbpe_raw_64k_dcs         0    228,375,865    3.302*     184.9       5.14       0.10
track2  125M  T7_byt5                         0    764,080,981    0.987*     394.2      17.18       0.34
--------------------------------------------------------------------------------------------------------
TOTAL track2 (30 runs)                       30  9,916,452,264             4,632.6     223.01       4.46

GRAND TOTAL (51 runs)                        51 11,734,624,761             5,119.5     263.89       5.28

* bytes/token estimated by sampling the corpus (no cached encoding yet)
```

(Only seed 0 of each arm is shown; the three seeds are identical rows and the totals cover
all 51 runs. `*` marks a bytes-per-token estimated by tokenising one line in 1,000 rather
than read from a cached encoding; the exact figure is measured when the sweep encodes the
corpus.)

**Read the two hour columns with suspicion, and prefer the FLOPs.** The 264 h "at rate"
figure applies a token rate measured on an **0.47–8.6M-parameter** model to models 6–15×
larger, so it is a floor, not an estimate. The ×50 column is an assumption, stated as one.
The physically grounded cross-check is the FLOPs column: **5,120 PFLOPs = 5.12 × 10¹⁸
FLOPs** of *training* (Track 1 487 PFLOPs, Track 2 4,633). An A100-80GB does 312 TFLOP/s in
bf16; at 40% model-FLOPs-utilisation that is 1.25 × 10¹⁴ FLOP/s.

**Evaluation is not free and is now in the estimate.** Five (Track 1) or six (Track 2)
held-out sets totalling 3.3–3.4 MB, scored every 25 steps at 50M and every 100 at 125M, is
a further **941 PFLOPs — +18% on the training total**, forward-only at `2 × N × tokens`.
It falls almost entirely on `T7_byt5`, which evaluates 3.4M tokens 158 times because a byte
vocabulary gives it four times the steps of a 64k arm.

| | PFLOPs (train + eval) | A100 @ 40% MFU | A100 @ 30% | H100 @ 40% |
|---|---:|---:|---:|---:|
| Track 1 (21 runs, 50M) | 487 + 158 | ~1.4 h | ~1.9 h | ~0.45 h |
| Track 2 (30 runs, 50M + 125M) | 4,633 + 783 | ~12.0 h | ~16.1 h | ~3.8 h |
| whole sweep (51 runs) | 5,120 + 941 | **~13.5 h** | **~18.0 h** | ~4.3 h |

**The realistic budget is 18–26 A100-hours; the 11.4 h / 13.5 h figures are a FLOP floor,
not a schedule.** 40% MFU is what a well-tuned large model reaches; a 50M model at batch 16
with a 64,064-wide softmax and no `torch.compile` or fused kernels will not, and 20–30% is
the honest band — which is 18 to 27 hours. Add data transfer, the first encode of five
Track 2 corpora (0.4–1.5 GB of `.bin` each), checkpoint I/O and 51 process starts. **Rent
for a day and expect change**, or budget half that on an H100.

## How to run on a rented GPU

```bash
git clone <this repo> && cd sanskrit-token
uv sync                       # torch, tokenizers, matplotlib, everything pinned
```

**Then get the data onto the box.** The LM corpora and the trained tokenizer arms are
gitignored — they are large and derived — so `git clone` gives you the code and nothing to
train on. Two options:

**(a) Copy them (minutes, recommended).** From a machine that has already built them:

```bash
rsync -av --exclude 'track2_raw.txt' data/processed/lm/  gpu:sanskrit-token/data/processed/lm/
rsync -av outputs/tokenizers/                            gpu:sanskrit-token/outputs/tokenizers/
# Optional, and safe: the encoded token streams, so the box does not re-tokenise 755 MB.
rsync -av outputs/05_lm_training/encoded/                gpu:sanskrit-token/outputs/05_lm_training/encoded/
```

Copying `outputs/05_lm_training/encoded/` is **safe to do**: `ensure_encoded_corpus`
validates every cached `.bin` against the corpus's path, byte count and sha256 *and* against
the tokenizer file's own sha256 (`arm_fingerprint`) before using it, and re-encodes if any
of them disagrees. A stale cache cannot silently train a model on the wrong tokens.

| what | size | needed for |
|---|---:|---|
| `data/processed/lm/track1_raw.txt` | 32 MB | Track 1, raw arms |
| `data/processed/lm/track1_split.txt` | 34 MB | Track 1, split arms |
| `data/processed/lm/track2_sample.txt` | 764 MB | Track 2 (all arms) |
| `data/processed/lm/heldout_*.txt` (11 files) | 7.0 MB | every evaluation |
| manifests (`*.manifest.json`, `manifest.json`) | 30 KB | corpus byte counts |
| **subtotal `data/processed/lm/` minus `track2_raw.txt`** | **809 MB** | |
| the six file-backed tokenizer arms | 26 MB | every arm but `T7_byt5` |
| (`outputs/tokenizers/` in full, if simpler) | 84 MB | |

`data/processed/lm/track2_raw.txt` (2.4 GB) is the *unsampled* M1 corpus and is **not**
needed: the sweep reads `track2_sample.txt`.

**(b) Rebuild from the raw sources (hours).** Only if you cannot copy. This re-downloads
~4.2 GB of Sangraha parquet and re-runs both leakage layers; the tokenizer arms have to be
retrained first, from Experiment 04:

```bash
uv run python experiments/04_morph_constrained/ingest_dcs.py       # ~2 min
uv run python experiments/04_morph_constrained/train_tokenizers.py # tokenizer arms
nohup uv run python experiments/05_lm_training/build_corpus.py \
    --config experiments/05_lm_training/corpus.yaml \
    > outputs/05_lm_training/build_corpus.log 2>&1 &                # ~70 min warm
uv run python experiments/05_lm_training/build_corpus.py \
    --config experiments/05_lm_training/corpus.yaml --sample        # ~4 min
uv run python experiments/05_lm_training/build_corpus.py \
    --config experiments/05_lm_training/corpus.yaml --sangraha-heldout   # ~4 min
uv run python experiments/02_tpp_parallel/build_exclusion.py        # ~1 min
```

The last two are Track 2's in-domain held-out set and the exclusion list that keeps it out
of every future corpus. `--sangraha-heldout` refuses to run twice, because a second draw
would hold out a second set while the exclusion list named only one.

Rebuilding is **not** byte-identical to what this repository reports — the Sangraha and
Wikipedia dataset revisions are pinned in `corpus.yaml`, but the ByT5 split cache the
`_split` held-out twins come from is Experiment 03 output — so the manifests' `sha256`
values are the check to run afterwards.

**Then check the box, smoke it, look at the plan, and run it — in that order.** Each step
is cheap and each one has failed for somebody:

```bash
# 1. Is there a GPU, and what is it?
nvidia-smi

# 2. Does torch agree, and will `dtype: auto` give bf16? (Ampere or newer; see below.)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), \
    torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

# 3. ~5 minutes: the whole pipeline end to end on this box. NOT a result.
uv run python experiments/05_lm_training/run.py --sweep smoke

# 4. The plan, at a rate you now believe. Writes sweep_plan.json; trains nothing.
uv run python experiments/05_lm_training/run.py --sweep sweep --dry-run \
    --tokens-per-s 100000        # a plausible A100 rate for a 50M model

# 5. The real thing. Detached, because it is a working day.
mkdir -p outputs/05_lm_training/sweep
nohup uv run python experiments/05_lm_training/run.py --sweep sweep \
    > outputs/05_lm_training/sweep/run.log 2>&1 &
tail -f outputs/05_lm_training/sweep/run.log
```

The sweep logs its resolved device and dtype on its first line — check that it says
`device cuda, dtype bfloat16` before walking away.

It is resumable: re-running the identical command skips every run whose `results.json` is
already there under an unchanged plan, so a killed job needs no argument changes.
Aggregation and both figures run automatically at the end, or on their own with
`uv run python experiments/05_lm_training/aggregate.py --config experiments/05_lm_training/sweep.yaml`
over a partially finished sweep.

**Disk.** ~810 MB of corpora, plus the encoded token streams the sweep caches under
`outputs/05_lm_training/encoded/`: ~3.5 GB (Track 2's five `.bin` files are 0.4–1.5 GB
each; `T7_byt5`'s is the large one, since a byte per token stored as `uint16` is 2 bytes).
Checkpoints are pruned as runs finish, so peak checkpoint use is one run's model plus its
two AdamW moments: **~1.0 GB at 50M** (85.2M total parameters with a 64k vocabulary) and
**~1.6 GB at 125M** (134.2M). **Budget 8 GB** beyond the environment.

**VRAM and batch size.** `batch_size × grad_accum × block_size` is **65,536 tokens per
step** at both sizes and that product is what the byte budget is spent in — change the
split, never the product, or the runs stop being comparable to the ones reported here. The
memory that matters is the logits tensor: `batch × 1024 × 64,064`, which at batch 16 is
~2 GB in bf16 before the cross-entropy's own working memory.

| card | 50M | 125M |
|---|---|---|
| 40 GB (A100-40, A6000) | `batch_size: 16, grad_accum: 4` (as shipped) | `batch_size: 8, grad_accum: 8` (as shipped) |
| 80 GB (A100-80, H100) | `batch_size: 32, grad_accum: 2` | `batch_size: 16, grad_accum: 4` |
| 24 GB (4090, L4) | `batch_size: 8, grad_accum: 8` | `batch_size: 4, grad_accum: 16` |

`device: auto` picks CUDA over MPS and `dtype: auto` selects bf16 on CUDA, so neither needs
changing on a GPU box. **`device: auto` will not fall through to the CPU**: `sweep.yaml`
does not set `allow_cpu`, so a box whose driver did not come up aborts in the first second
rather than starting a 5.1-exaFLOP job on a CPU (`lm/config.resolve_device`). An explicitly
named `cuda` that torch reports as unavailable raises for the same reason.

**Ampere or newer.** `dtype: auto` takes bf16 only where `torch.cuda.is_bf16_supported()`,
which means SM 8.0+ (A100, A10, A6000, L4, 4090, H100); on anything older it silently falls
back to fp16 with a `GradScaler`, which this project has never validated on this model and
which can NaN a loss into a missing number rather than an error. Do not rent a V100 or a T4.

**A note on the VRAM numbers below.** The logits tensor is materialised in **fp32 even under
bf16 autocast**: `masked_cross_entropy` writes `-inf` into the padding columns and then calls
`F.cross_entropy`, which promotes. At 50M with `batch_size: 16` that is `16 × 1024 × 64,064 ×
4 B` ≈ **4.2 GB** for the logits alone, and with its gradient and the autocast bf16 copy the
peak is **~14 GB**; at 125M with `batch_size: 8` it is ~2.1 GB of logits and **~9–11 GB**
peak. Both fit a 24 GB card with room, which is why the table below starts there.

## Files

```
corpus.yaml        what build_corpus.py builds and from where, incl. the sample budget
build_corpus.py    Task 1: the corpora, the held-out texts, the sample and their manifests
                   (--sample, --heldout-only, --sangraha-heldout)
sweep.yaml         Task 3: the real 51-run grid — arms, sizes, seeds, byte budgets
smoke.yaml         Task 3: the three-arm 300-step MPS validation (not a result)
sweep.py           enumerate / budget / train / prune, resumable; --dry-run projections
aggregate.py       per-seed runs -> results.json + bpc_curves.* + bpc_final.*
run.py             --sweep smoke|sweep: sweep.py then aggregate.py
```

The pipeline itself is `src/sanskrit_tok/lm/`: `model.py` (vendored nanoGPT), `data.py`
(corpus → memmapped token ids), `bpc.py` (the pure metric), `config.py`, `train.py`.

## How to run here

The smoke sweep (~5 min on MPS, including first-time corpus encoding):

```bash
uv run python experiments/05_lm_training/run.py --sweep smoke
```

Corpus assembly, if `data/processed/lm/` is empty, is under "How to run on a rented GPU",
option (b) above.

## Status

| step | state |
|---|---|
| Task 1 — M1 corpus, Track 1/2 corpora, held-out texts, leakage filtering | done |
| Task 2 — vendored nanoGPT, corpus encoding, BPC evaluation, training loop | done |
| Task 3 — sweep runner, aggregation, figures, MPS smoke validation | done |
| Phase-A review fixes — raw BPC denominator, strict device, `forward_logits`, best-curve reference, Sangraha held-out set | done |
| `data/exclusion_hashes.txt` regenerated with `sangraha_heldout` | done |
| **the real sweep (51 runs)** | **not run — needs a GPU (18–26 A100-hours)** |
| Experiment 05 write-up | blocked on the sweep |

**The exclusion list, for the record.** `heldout_sangraha.txt` exists, `track2_sample.txt`
no longer contains it, and `data/exclusion_hashes.txt` **has** been regenerated with the
`sangraha_heldout` source; its header names that source with 2,000 hashes. The regeneration
needed `--allow-shrink`, because the committed list also held 35 stale DCS hashes that no
source produces any more: the DCS held-out split was re-ingested on 2026-09-05 with the
`ṁ` → `ṃ` normalisation, so those 35 sentences are still excluded, under their new
spellings. The evidence, measured before the file was touched: all 27,487 committed
parallel-corpus hashes were still produced (27,487 of 27,487), all 35 lost hashes fell in
the DCS-only region, and DCS gained 22 new ones. Nothing stopped being evaluation text. The
override is recorded in `docs/decisions.md` under 2026-09-06, "Exclusion list regenerated
with `--allow-shrink` once, to add `sangraha_heldout`". The command that was run:

```bash
uv run python experiments/02_tpp_parallel/build_exclusion.py --allow-shrink
```
