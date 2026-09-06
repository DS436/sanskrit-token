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

In-domain (primary): `heldout_dcs.txt`, the DCS held-out split — 13 whole texts, never
trained on. Out-of-domain: the Sanskrit side of Sāmayik test, Sāmayik test_ood, Itihāsa
test and FLORES devtest. Each has a `_split` twin (oracle split for DCS, ByT5-reconciled
for the parallel corpora) for the split arms, so every arm is evaluated on the text shape
it was trained on.

A pair is dropped when either half carries a character outside SLP1, which costs the
out-of-domain sets 116 / 564 / 108 / 48 sentences (Sāmayik test, test_ood, FLORES, Itihāsa
test) and DCS none — mostly curly quotation marks and en-dashes, not mojibake. Experiment
05's OOD evaluation sets are therefore proper subsets of Experiments 02 and 03's; see
`data/README.md` and the open question in `.superpowers/sdd/exp05-task-1-report.md`.

## Leakage

Every training line passes **both** layers (CLAUDE.md §2.4): the sha256 of its SLP1 form
against `data/exclusion_hashes.txt`, and the 24-letter shingle index of all seven
evaluation sources. A hit drops and counts the line; the per-source counts are in the
manifests. See `data/README.md` for the numbers.

## What Task 1 built

| Corpus | Lines | SLP1 chars | Bytes | Tokens (`T1_bpe_raw_64k_dcs`) |
|---|---|---|---|---|
| `track1_raw.txt` | 652,007 | 31,535,529 | 31,535,529 | 6,157,847 |
| `track1_split.txt` | 652,007 | 33,489,418 | 33,489,418 | 6,503,434 |
| `track2_raw.txt` (M1) | 32,164,442 | 2,504,067,955 | 2,504,067,955 | 669,967,173 |
| `track2_sample.txt` | 9,862,736 | 755,048,553 | 755,048,553 | 200,096,083 |

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

## Training protocol

From the outline (`docs/paper_outline.md` §6) and the decision entry that fixed the details
(docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and comparison protocol"):

- **Architecture:** decoder-only GPT-2 style, nanoGPT's `model.py` vendored byte-for-byte
  at commit `f08abb45bd22` under `src/sanskrit_tok/lm/model.py` with its MIT licence.
- **Sizes:** 50M and 125M **non-embedding** parameters (8 layers × 512 × 8 heads; 12 × 768
  × 12). Non-embedding, so the transformer body is identical across arms and only the
  embedding table grows with the vocabulary; `results.json` records both counts and says
  which one `flops_est = 6 × params × tokens` used (the total).
- **Context** 1024 tokens; AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, linear
  warmup then cosine decay to 10% of the peak rate; bf16 on CUDA, float32 on MPS and CPU.
- **Data:** identical text for every arm within a track; the tokenizer is the only
  variable. One EOS per line, id = the arm's `vocab_size`.
- **Seeds:** three (CLAUDE.md §2.6), `random` / `numpy` / `torch` all set from the config
  and recorded; mean ± sample standard deviation reported.
- **Comparison:** at equal training **bytes**, with tokens and FLOPs recorded, and
  secondarily at equal tokens. **BPC is the headline; perplexity is never reported**
  (CLAUDE.md §2.2).
- **Tokens/bytes/FLOPs to reference BPC:** the reference is `T1_bpe_raw_64k_dcs`'s final
  in-domain BPC *at the same track and size*, matched seed by seed; `null` when an arm
  never reaches it.

## The equal-bytes budget

The budget is one number per track — the **raw** corpus's bytes × `epochs` — and every arm
in that track spends that same number of bytes on the text it actually trains on. Each arm
converts it into its own token budget with its own bytes-per-token, measured on the encoded
corpus:

    max_tokens = round(max_bytes / bytes_per_token)

Track 1 is 8 × 31,535,529 = **252,284,232 bytes**; Track 2's sample is **755,048,553
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
`results.json` (final BPC per evaluation set, params, tokens, bytes, FLOPs, throughput,
hardware, config, provenance), `config.yaml`, `curve.jsonl` (one row per evaluation:
step, tokens, bytes, FLOPs, loss, lr, BPC per set) and `sweep_run.json` (the plan hash and
the budget arithmetic: `max_bytes`, `bytes_per_token`, `max_tokens`). `ckpt.pt` is deleted
once the results are final unless `keep_checkpoints: true`.

Per sweep, in `outputs/05_lm_training/<sweep>/`: `results.json` (per track × size × arm:
BPC mean ± sd over seeds for every set, the seed-mean curve, and the bytes/tokens/FLOPs to
the reference BPC), `bpc_curves.{pdf,png}` (BPC against training bytes, one panel per track
× size, shaded ± 1 sd, dashed line at the reference's final BPC) and `bpc_final.{pdf,png}`
(final BPC per arm per evaluation role, with vocabulary and the `oracle` / `provisional`
caveats on the labels).

Everything under `outputs/` is gitignored; the numbers quoted in this file are the record.

## Smoke sweep — pipeline validation, NOT a result

Ran 2026-09-06 on this laptop (Apple MPS, float32, torch 2.14.0,
macOS-26.6.2-arm64) at commit `4b4f05a`:
`--sweep smoke`, three arms, one seed, 300 steps of the **`smoke` model — two layers, 128
wide, 0.43–8.6M parameters** — on `track1_raw.txt`, evaluating every 100 steps on
`heldout_dcs` (capped at 200,000 characters) and `heldout_samayik_test` (173 KB, so
uncapped in practice). Total wall clock **104 s of training** (49.7 + 49.9 + 4.0 s), plus
~1 min of first-time corpus encoding per arm.

| arm | vocab | params (total) | tokens | bytes seen | BPC `heldout_dcs` | BPC `heldout_samayik_test` | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `T1_bpe_raw_64k_dcs` | 64,001 | 8,626,816 | 614,400 | 2,845,205 | **2.8259** | 4.6140 | 12,352 |
| `T5_morphbpe_rawseg_64k_dcs` | 64,001 | 8,626,816 | 614,400 | 2,727,452 | **2.8588** | 4.6544 | 12,301 |
| `T7_byt5` | 257 | 467,584 | 614,400 | 601,954 | **3.7855** | 4.4280 | 155,252 |

BPC falls monotonically in every arm (`T1` 3.5228 → 2.8744 → 2.8446 → 2.8259; `T7` 8.2361 →
3.9841 → 3.8198 → 3.7855), the aggregation produced both figures, and `T5seg` and `T7`
never reach `T1`'s final in-domain BPC, so their `to_reference` entries are `null` — the
undefined branch is exercised by real data, not only by a test.

**None of these numbers say anything about tokenizers.** The model is three orders of
magnitude smaller than the research sizes, the budget is 300 steps, and — because the smoke
sweep is budgeted in *steps* rather than bytes — the three arms did not even see the same
amount of text (2.85 MB, 2.73 MB and 0.60 MB respectively). `T7_byt5`'s 155k tokens/s is
likewise an artefact of the `smoke` size: with a 320-row embedding its softmax is free,
while the 64k arms spend most of their time there. At 50M and 125M the transformer body
dominates and that gap will close.

## Projected cost of the real sweep

`sweep.py --dry-run --tokens-per-s 12352` (this laptop's measured end-to-end MPS rate on
the 64k smoke arm), `--gpu-speedup 50`:

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

track2  50M   T1_bpe_raw_64k_dcs              0    209,803,778    3.599*      73.6       4.72       0.09
track2  50M   T2_unigram_raw_64k_dcs          0    234,234,174    3.223*      82.2       5.27       0.11
track2  50M   T5_morphbpe_rawseg_64k_dcs      0    216,927,037    3.481*      76.1       4.88       0.10
track2  50M   T5_morphbpe_raw_64k_dcs         0    228,923,477    3.298*      80.4       5.15       0.10
track2  50M   T7_byt5                         0    764,833,609    0.987*     118.7      17.20       0.34
track2  125M  T1_bpe_raw_64k_dcs              0    209,803,778    3.599*     169.9       4.72       0.09
track2  125M  T2_unigram_raw_64k_dcs          0    234,234,174    3.223*     189.6       5.27       0.11
track2  125M  T5_morphbpe_rawseg_64k_dcs      0    216,927,037    3.481*     175.6       4.88       0.10
track2  125M  T5_morphbpe_raw_64k_dcs         0    228,923,477    3.298*     185.3       5.15       0.10
track2  125M  T7_byt5                         0    764,833,609    0.987*     394.6      17.20       0.34
--------------------------------------------------------------------------------------------------------
TOTAL track2 (30 runs)                       30  9,928,332,450             4,638.4     223.27       4.47

GRAND TOTAL (51 runs)                        51 11,746,504,947             5,125.3     264.16       5.28

* bytes/token estimated by sampling the corpus (no cached encoding yet)
```

(Only seed 0 of each arm is shown; the three seeds are identical rows and the totals cover
all 51 runs. `*` marks a bytes-per-token estimated by tokenising one line in 1,000 rather
than read from a cached encoding; the exact figure is measured when the sweep encodes the
corpus.)

**Read the two hour columns with suspicion, and prefer the FLOPs.** The 264 h "at rate"
figure applies a token rate measured on a **0.43–8.6M-parameter** model to models 6–15×
larger, so it is a floor, not an estimate. The ×50 column is an assumption, stated as one.
The physically grounded cross-check is the FLOPs column: **5,125 PFLOPs = 5.13 × 10¹⁸
FLOPs** in total (Track 1 487 PFLOPs, Track 2 4,638). An A100-80GB does 312 TFLOP/s in
bf16; at a realistic 40% model-FLOPs-utilisation that is 1.25 × 10¹⁴ FLOP/s, so:

| | PFLOPs | A100 @ 40% MFU | H100 @ 40% MFU |
|---|---:|---:|---:|
| Track 1 (21 runs, 50M) | 487 | **~1.1 h** | ~0.35 h |
| Track 2 (30 runs, 50M + 125M) | 4,638 | **~10.3 h** | ~3.2 h |
| whole sweep (51 runs) | 5,125 | **~11.4 h** | ~3.6 h |

Call it **one A100-day including data transfer, encoding and evaluation overhead**, or
under half that on an H100. Evaluation is not in the FLOPs column: five held-out sets
totalling ~3.3 MB, scored every 100 (Track 1) or 250 (Track 2) steps, adds roughly 5%.

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
```

| what | size | needed for |
|---|---:|---|
| `data/processed/lm/track1_raw.txt` | 32 MB | Track 1, raw arms |
| `data/processed/lm/track1_split.txt` | 34 MB | Track 1, split arms |
| `data/processed/lm/track2_sample.txt` | 765 MB | Track 2 (all arms) |
| `data/processed/lm/heldout_*.txt` (10 files) | 6.8 MB | every evaluation |
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
```

Rebuilding is **not** byte-identical to what this repository reports — the Sangraha and
Wikipedia dataset revisions are pinned in `corpus.yaml`, but the ByT5 split cache the
`_split` held-out twins come from is Experiment 03 output — so the manifests' `sha256`
values are the check to run afterwards.

**Then look at the plan, and run it:**

```bash
uv run python experiments/05_lm_training/run.py --sweep sweep --dry-run \
    --tokens-per-s 100000        # a plausible A100 rate for a 50M model
nohup uv run python experiments/05_lm_training/run.py --sweep sweep \
    > outputs/05_lm_training/sweep/run.log 2>&1 &
```

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

`device: auto` picks CUDA over MPS over CPU and `dtype: auto` selects bf16 on CUDA, so
neither needs changing on a GPU box.

## Files

```
corpus.yaml        what build_corpus.py builds and from where, incl. the sample budget
build_corpus.py    Task 1: the corpora, the held-out texts, the sample and their manifests
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
| **the real sweep (51 runs)** | **not run — needs a GPU (~1 A100-day)** |
| Experiment 05 write-up | blocked on the sweep |
