# Experiment 05 — language-model training and bits per character

**Status: phase A, Task 1 complete — corpora built, quality-filtered, and the Track 2
sample cut to its ~200 M-token budget. No LM has been trained yet.** Tasks 2 (training
pipeline) and 3 (sweep, aggregation, MPS smoke run) follow; the real sweep needs a GPU
(see below).

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

## Files

```
corpus.yaml        what build_corpus.py builds and from where, incl. the sample budget
build_corpus.py    Task 1: the corpora, the held-out texts, the sample and their manifests
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

Then the Track 2 sample (~4 min; it subsets the file the build just wrote, so it must run
after it and is invalidated whenever `track2_raw.txt` is rebuilt):

```
uv run python experiments/05_lm_training/build_corpus.py \
    --config experiments/05_lm_training/corpus.yaml --sample
```

**A GPU is needed for the real sweep.** Experiments 01–04 run on this laptop; Track 1 is
seven arms × three seeds and Track 2 is five arms × two sizes × three seeds, which is not
a laptop-sized job. Phase A validates the whole pipeline on MPS with a `smoke`
configuration (2 layers, 128 wide, a few hundred steps) so that the paid GPU run is one
command. Instructions and the projected hours land here with Task 3.
