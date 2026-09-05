# Experiment 05 — language-model training and bits per character

**Status: phase A, Task 1 complete (corpora built). No LM has been trained yet.**
Tasks 2 (training pipeline) and 3 (sweep, aggregation, MPS smoke run) follow; the real
sweep needs a GPU (see below).

## Hypothesis

H4, the BPC half of the research question. If sandhi-splitting and morpheme-constrained
merges recover token efficiency that a plain subword tokenizer destroys, then a language
model trained on the **same text** under those tokenizers should reach a lower **bits per
character** on held-out Sanskrit at equal training bytes than one trained under
`T1_bpe_raw_64k_dcs`, and should reach any given BPC after fewer bytes, tokens and FLOPs.

BPC is the headline because it is the only quantity that is comparable across
vocabularies: it is the model's total cross-entropy divided by the number of SLP1
*characters* of the evaluation text. **Perplexity is never reported** (CLAUDE.md §2.2) —
it is per token, and the arms have different tokens.

Experiment 04's caveat travels with this one: the constrained arms (T5, T6) pay a 24–33%
token handicap on the same text, so at equal bytes they see more tokens and more compute.
Equal bytes is the primary comparison, equal tokens the secondary one, and both are
recorded per run.

## The two tracks

Tokenizers are **not** retrained here. Every arm is one of the fixed `_dcs` arms trained in
Experiment 04, so the LM data is the only thing that changes within a track
(docs/decisions.md, 2026-09-05, "Experiment 05 runs in two tracks").

| | Track 1 — matched | Track 2 — scale |
|---|---|---|
| Corpus | DCS training sentences | corpus M1 (DCS + Sāmayik/Itihāsa training sides + Sangraha verified Sanskrit + Sanskrit Wikipedia) |
| Raw arms see | `track1_raw.txt` (sandhied) | `track2_raw.txt` |
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

## Leakage

Every training line passes **both** layers (CLAUDE.md §2.4): the sha256 of its SLP1 form
against `data/exclusion_hashes.txt`, and the 24-letter shingle index of all seven
evaluation sources. A hit drops and counts the line; the per-source counts are in the
manifests. See `data/README.md` for the numbers.

## What Task 1 built

| Corpus | Lines | SLP1 chars | Bytes | Tokens (`T1_bpe_raw_64k_dcs`) |
|---|---|---|---|---|
| `track1_raw.txt` | 652,889 | 31,605,778 | 31,609,800 | 6,172,903 |
| `track1_split.txt` | 652,889 | 33,562,815 | 33,566,131 | 6,518,519 |
| `track2_raw.txt` (M1) | 45,207,395 | 3,664,892,180 | 3,671,928,463 | 1,015,272,319 |

64.6 min warm; `data/README.md` has the per-source counts, the licences and every drop.
Track 2 is ~165× Track 1 by characters, so the two tracks answer different questions: Track
1 is the controlled comparison at ~6M tokens multi-epoch, Track 2 the scale run at ~1B.

Two caveats to carry into any Track 2 number. Sangraha's verified Sanskrit is OCR of
printed books, with the usual extraction damage and no cleaning applied; and 20.1% of its
lines were exact duplicates of an earlier line, which the deduplication removed but which
says something about what the rest looks like.

## Files

```
corpus.yaml        what build_corpus.py builds and from where
build_corpus.py    Task 1: the corpora, the held-out texts and their manifests
```

Outputs land in `data/processed/lm/` (gitignored); the manifests are the record.

## How to run

Corpus assembly (~4.2 GB of Sangraha parquet on a cold run, ~70 min warm):

```
mkdir -p outputs/05_lm_training
nohup uv run python experiments/05_lm_training/build_corpus.py \
    --config experiments/05_lm_training/corpus.yaml \
    > outputs/05_lm_training/build_corpus.log 2>&1 &
```

**A GPU is needed for the real sweep.** Experiments 01–04 run on this laptop; Track 1 is
seven arms × three seeds and Track 2 is five arms × two sizes × three seeds, which is not
a laptop-sized job. Phase A validates the whole pipeline on MPS with a `smoke`
configuration (2 layers, 128 wide, a few hundred steps) so that the paid GPU run is one
command. Instructions and the projected hours land here with Task 3.
