# Data provenance

Every corpus used by this project is recorded below; raw and processed data are gitignored
and never committed. Fill in the download date, version/commit, license and any filtering
applied at the moment a source is first fetched.

| Name | Source URL | Download date | Version/commit | License | Filtering applied |
|------|------------|---------------|----------------|---------|-------------------|
| Digital Corpus of Sanskrit (DCS) | https://github.com/OliverHellwig/sanskrit (`dcs/data/conllu/files` only) | 2026-09-05 | commit `8aeed5a1343d0e48b64eb32af8c00e8c6eb29359` (head of `master`, authored 2026-08-24) | **CC BY 4.0**, stated at `dcs/data/readme.md` in the source repo: "The data of the DCS and any data in child directories are licensed under the Creative Common BY 4.0 (CC BY 4.0) license." Cite: Oliver Hellwig, *Digital Corpus of Sanskrit (DCS)*, 2010-2024. | Blobless sparse checkout (`git clone --filter=blob:none --no-checkout` + `sparse-checkout set dcs/data/conllu/files` + `checkout <commit>`; **not** `--depth 1`, which cannot be pinned) into `data/raw/dcs/repo` — 1.3 GB on disk, 16,051 `.conllu` files across 271 texts. Ingested by `experiments/04_morph_constrained/ingest_dcs.py` to `data/processed/dcs/` (467 MB, gitignored) in 126 s: IAST -> SLP1 per word (with `ṁ` normalised to `ṃ` first — `boundaries.normalise_iast`), gold segment boundaries located in the sandhied surface by `difflib` alignment, heuristic stem/ending boundaries from the lemma. Every number below is `data/processed/dcs/manifest.json` as of the 2026-09-05 re-ingestion. **715,011 sentences / 271 texts kept**: train 684,874 / 258 texts, held out 30,137 / 13 texts (5% of `text_id`s, `random.Random(0)`; Śivasūtravārtika, Bhāvaprakāśa, Skandapurāṇa (Revākhaṇḍa), Rasataraṅgiṇī, Nirukta, Kātyāyanasmṛti, Śyainikaśāstra, Śāṅkhāyanāraṇyaka, Devīmāhātmya, Kauṣītakibrāhmaṇa, Vaitānasūtra, Aitareya-Āraṇyaka, Vasiṣṭhadharmasūtra). Dropped: 14,204 sentences with fewer than two words, 776 colliding with an existing evaluation sentence, 954 whose SLP1 form still carried a character outside the SLP1 alphabet after normalisation (halfwidth-katakana mojibake from the source scans — dropped from **both** splits, since the held-out text is the bits-per-character denominator), 5,305 training sentences whose text duplicates a held-out sentence verbatim, and 34,695 training sentences near-duplicating an evaluation sentence by the 24-letter shingle filter (`itihasa_test` 20,608, `itihasa_dev` 11,458, `dcs_heldout` 3,576, `samayik_dev` 5, `samayik_test_ood` 2, `flores_devtest` 0; counts overlap). 4,077,626 surface words, **97.47% with an aligned gold segmentation**; **33.15% of sentences human-verified** (no `UnsandhiedReconstructed=True` token) — the rest of the segmentation is machine-generated and every MorphScore table reports the two separately. `# text` disagrees with the reconstructed token block on 11,054 sentences (1.5%); those words carry no gold boundary and are counted as unaligned. The held-out sentences are in `data/exclusion_hashes.txt` (source `dcs_heldout`). **Five tokenizer training corpora** are streamed from `train.jsonl` by `experiments/04_morph_constrained/train_tokenizers.py` into `data/processed/` (gitignored, manifest with per-corpus counts and sha256 beside each): `tok_train_dcs_raw.txt` (`text_slp1`, 652,889 lines), `tok_train_dcs_oracle_split.txt` (`oracle_split_slp1`, 647,896), `tok_train_dcs_raw_marked.txt` (`t5_marked`, 653,863 lines, 3,244,038 U+001F boundary markers), `tok_train_dcs_raw_segmarked.txt` (`t5seg_marked`, 653,445 lines, 1,291,655 markers) and `tok_train_dcs_split_marked.txt` (`t6_marked`, 648,302 lines, 2,123,574 markers). Each is exact-deduplicated on the written line and every one of its input sentences is checked against `data/exclusion_hashes.txt` on its sandhied SLP1 form (`sentence_hash_slp1`): **0 leaked, 685,805 checked, for all five**. Those five corpora and the twelve `_dcs` tokenizer arms trained on them were built from the **previous** ingestion, whose training split held 685,805 sentences; they were deliberately not rebuilt after the re-ingestion (docs/decisions.md, 2026-09-05, the DCS re-ingestion CORRECTION), so their `n_in` is 685,805 while the table above says 684,874. |
| SIGHUM | Krishna et al. 2017 release | not yet downloaded | | | |
| Hackathon | Krishnan et al. 2020 release | not yet downloaded | | | |
| DCS-2018 | Hellwig & Nehrdich 2018 release | not yet downloaded | | | |
| UoH corpus + SandhiKosh | https://sanskrit.uohyd.ac.in/Corpus/ | not yet downloaded | | | |
| Itihāsa | https://github.com/rahular/itihasa | 2026-09-03 | commit `37df077a80c83dbba1598afbdb079674b9bf6daa` | unspecified (repo has no LICENSE file; dataset described in Aralikatte et al., WAT 2021) | `data/{train,dev,test}.sn`/`.en` downloaded verbatim to `data/raw/itihasa/`; train 75,161 / dev 6,148 / test 11,721 aligned pairs; 0 empty pairs dropped in every split |
| Sāmayik | https://github.com/ayushbits/Saamayik (arXiv 2305.14004) | 2026-09-03 | commit `f87d54903e9f8540bda63efeb2c203e368195523` | unspecified (repo has no LICENSE file; dataset described in Aralikatte et al., LREC-COLING 2024) | `data/final_data/{train,dev,test}.sa`/`.en` and `data/mkb/mkb.sa`/`.en` (out-of-domain `test_ood`; sibling `mkb.txt` ignored, not an aligned pair) downloaded verbatim to `data/raw/samayik/`; train 43,493 / dev 2,416 / test 2,417 / test_ood 4,047 aligned pairs; 0 empty pairs dropped in every split |
| Sangraha (verified Sanskrit) | HF `ai4bharat/sangraha`, `verified/san/data-{0..10}.parquet` (https://huggingface.co/datasets/ai4bharat/sangraha) | 2026-09-05 | dataset revision `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8` | **CC BY 4.0** (dataset card); AI4Bharat, *Sangraha*, cite Khan et al. 2024 (IndicLLMSuite) | **Only `verified/san/`**: 11 parquet files, 4,182,068,434 bytes (3.9 GB on disk) in `data/raw/sangraha/`, **908,066 documents** (mostly `type: pdf`, i.e. OCR'd print — the extraction carries the usual OCR damage and no cleaning beyond the line rules below is applied). The sibling `synthetic/san_Deva/wiki_*.parquet` files are **machine-translated Wikipedia and are excluded by the loader** (`sanskrit_tok.data.sangraha.verified_sanskrit_files`), as is `unverified/`. Documents are cut into lines by `documents_to_lines`: split on `\n`, then after every danda `।`/`॥` (the danda stays on the line it ends), internal whitespace collapsed, and a line dropped unless it has ≥2 **real** words (a token with at least one letter — a bare danda or verse number is not a word, `quality.real_words`) and ≥1 Devanagari **letter** (digits, dandas and combining signs do not count) — **50,602,793 lines**. Every line then passes the calibrated quality filter (`sanskrit_tok.data.quality`, docs/decisions.md 2026-09-05) **on the Devanagari source**, which also does the SLP1 conversion: **13,617,929 dropped (26.9%)** — `latin` 659,030, `hindi` 460,704, `letter_fraction` 1,359,829, `n_words` 6,928,270, `word_length` 3,965,033, `non_slp1` 245,063. Into `track2_raw.txt` (corpus M1) after both leakage layers and exact-line dedup: **101 dropped by sha256**, **313,606 by the 24-letter shingle filter**, **5,352,602 as duplicates** (14.5% of what survived the quality filter — OCR page furniture and formulaic lines recur across scans), leaving **31,318,555 lines**. |
| Sanskrit Wikipedia | HF `wikimedia/wikipedia`, config `20231101.sa`, `train-00000-of-00001.parquet` | 2026-09-05 | dataset revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | **CC BY-SA 3.0** (with GFDL), per the dataset card | 23,850,161 bytes in `data/raw/wikipedia_sa/`, **12,156 articles**. Boilerplate stripped per article before line splitting (`wikipedia_sa.strip_boilerplate`): lines starting `==`, a line repeating the article's own title, and raw lines shorter than 20 characters (section headings in this dump are bare lines, so the length rule is what removes them); then the same danda/newline line rules as Sangraha — **338,112 lines**. The same quality filter as Sangraha, on the Devanagari source: **84,297 dropped (24.9%)** — `latin` 8,863, `hindi` 333, `letter_fraction` 8,379, `n_words` 35,613, `word_length` 11,776, `non_slp1` 19,333. Into `track2_raw.txt` after both leakage layers and dedup: **0 dropped by sha256**, **1,185 by the shingle filter**, **168,852 as duplicates**, leaving **83,778 lines**. |
| SAHAAYAK 2023 | arXiv 2307.00021 release | not yet downloaded | | | |
| FLORES-200 | https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (official NLLB tarball; the HF ids `openlanguagedata/flores_plus`, `facebook/flores` and `Muennighoff/flores200` all failed, see `docs/decisions.md`) | 2026-09-03 | `flores200_dataset.tar.gz`, sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6` | CC BY-SA 4.0 | `devtest` split only; `san_Deva`, `hin_Deva`, `eng_Latn` extracted verbatim (source script, no normalisation) to `data/raw/flores/devtest.jsonl`; 1012 aligned sentences |
| IN22-Gen | AI4Bharat | not yet downloaded | | | |
| ByT5-Sanskrit (sandhi splitter) | HF `chronbmm/sanskrit5-multitask` (the plan's placeholder id `chronbmm/byt5-sanskrit` does not resolve; see `docs/decisions.md`, "Experiment 03 sandhi splitter") | 2026-09-05 | revision `c0d2ada54f3d19903149425aa888a203601423f8` | unspecified (no LICENSE and no model card on the repo; model described in Nehrdich, Hellwig & Keutzer, EMNLP Findings 2024) | Segmentation mode only (`"S "` prefix, IAST in and out, greedy, 512-byte window). **Model weights are not data and are not committed**; what is derived from it is. Run of 2026-09-04/05 on MPS, 9.40 h wall: 136,918 sentences submitted, 136,652 generated (128 within-call duplicates, 138 cache hits on the first pass, 1 on the re-run), 433 inputs chunked at the byte limit — the Sanskrit side of `samayik_test` (2,417), `samayik_test_ood` (4,047), `itihasa_test` (11,721), `flores_devtest` (1,012) and the 117,721-sentence tokenizer training selection (post exclusion filter and original-text dedup). Cached in `data/processed/split/cache.jsonl` (59 MB, gitignored) keyed by sha256 of the input; per-corpus `<corpus>.jsonl` carry the raw model output and the reconciled text, `manifest.json` the counts. Every run also writes a timestamped `manifest_<UTC>.json`, and **the most recent of those is the authoritative record of what a given run did** — `manifest.json` is last-write-wins and a cheap cache-hit re-run overwrites it (`manifest_original_run.json` recovers the first 9.4 h run's facts from its log). Pooled non-space character retention **0.8706** for the raw model output and **0.9895** after `sandhi.reconcile`; 15.5% of raw whitespace units kept verbatim, 45.7% replaced by a window scoring below 1.0. |
| DharmaBench | per-paper release | not yet downloaded | | | |
| IndicParam | per-paper release | not yet downloaded | | | |

## Experiment 05 LM corpora (`data/processed/lm/`, gitignored)

Built by `experiments/05_lm_training/build_corpus.py` in 70.1 min (warm cache), manifests
beside each file (`<corpus>.manifest.json`) and a combined `manifest.json`.

| Corpus | Sources | Lines | SLP1 chars | Bytes | Tokens (`T1_bpe_raw_64k_dcs`) |
|---|---|---|---|---|---|
| `track1_raw.txt` | DCS train, sandhied (`text_slp1`) | 652,007 | 31,535,529 | 31,535,529 | 6,157,847 |
| `track1_split.txt` | the **same sentences in the same order**, gold split (`oracle_split_slp1`) | 652,007 | 33,489,418 | 33,489,418 | 6,503,434 |
| `track2_raw.txt` (corpus M1) | DCS train + Sāmayik train + Itihāsa train + Sangraha verified + Wikipedia | 32,164,442 | 2,504,067,955 | 2,504,067,955 | 669,967,173 |
| `track2_sample.txt` | a subset of `track2_raw.txt` at the ~200 M-token budget | 9,862,736 | 755,048,553 | 755,048,553 | 200,096,083 |

**SLP1 chars and Bytes are equal in every row**, and that is a property of the corpora
rather than a copied column: `quality.is_clean_slp1` is applied to every line of every
corpus and every held-out file, so what reaches disk is SLP1 letters, ASCII digits, ASCII
punctuation and spaces — pure ASCII, one byte per character. (They differed in the first
build, before that rule existed.) The counts exclude the newline that terminates each line,
so the files on disk are `n_bytes + n_out` bytes long.

**The Track 2 sample.** Track 2 is 670 M tokens and the outline budgets ~200 M
(docs/decisions.md, 2026-09-05, "Track 2 budget"), so `build_corpus.py --sample` subsets
the already-written `track2_raw.txt` — inheriting both leakage layers, the deduplication
and the quality filter by construction — using the half-open per-source line ranges the
Track 2 manifest records. DCS, Sāmayik, Itihāsa and Wikipedia (48,131,611 bytes together)
are kept **in full**; Sangraha is thinned by a per-line Bernoulli draw at *p* = 0.287820
from `random.Random(0)`, uniform over the whole of it rather than a prefix. Composition of
`track2_sample.txt` (lines out / lines in, bytes out):

| Source | Lines | of | Bytes | Sampled |
|---|---|---|---|---|
| `dcs_train` | 652,007 | 652,007 | 31,535,529 | no |
| `samayik_train_sa` | 38,584 | 38,584 | 2,776,499 | no |
| `itihasa_train_sa` | 71,518 | 71,518 | 7,515,478 | no |
| `sangraha_verified_san` | 9,016,849 | 31,318,555 | 706,916,942 | **yes** |
| `wikipedia_sa` | 83,778 | 83,778 | 6,304,105 | no |
| total | 9,862,736 | 32,164,442 | 755,048,553 | |

`track2_sample_bytes: 755000000` in `corpus.yaml` was calibrated, not predicted: a first
run at 926,000,000 bytes measured 246,005,664 tokens (3.7638 bytes/token overall, 4.7332
over the kept-in-full sources, 3.7220 over Sangraha), and the budget was set once from
those measurements. Both runs are recorded in `corpus.yaml` and `docs/decisions.md`.

Held-out evaluation texts, raw and split, one line per sentence: `heldout_dcs` 30,137,
`heldout_samayik_test` 2,301, `heldout_samayik_test_ood` 3,483, `heldout_itihasa_test`
11,673, `heldout_flores_devtest` 904. The four parallel ones come from
`data/processed/split/<corpus>.jsonl` (raw `raw_slp1`, split `output`), and each file's
`raw_deva` is checked sentence-by-sentence against its loader's aligned Sanskrit before
anything is written — all four passed, so line *i* of a raw file and of its `_split` twin
is the same sentence.

Held-out text is neither deduplicated nor leakage-filtered — it *is* the evaluation set —
but a pair is dropped when either half is empty or fails `is_clean_slp1`, since a character
no arm's vocabulary can spell would otherwise sit in the bits-per-character denominator.
**`samayik_test` 116 pairs dropped, `samayik_test_ood` 564, `flores_devtest` 108,
`itihasa_test` 48, `heldout_dcs` 0.** These are mostly not mojibake: curly quotation marks
(`‘ ’ “ ”`), en-dashes and the candra vowels `ॉ`/`ऑ` pass through `to_slp1` unchanged. The
consequence is that Experiment 05's out-of-domain evaluation sets are **proper subsets** of
the sets Experiments 02 and 03 measured on (Sāmayik test_ood loses 13.9%), which is a
research decision for the orchestrator, not an engineering one — see the open question in
`.superpowers/sdd/exp05-task-1-report.md`.

**Leakage (CLAUDE.md §2.4), both layers on every training line.** Track 1: 0 dropped by
sha256, 0 by shingle (DCS train was already filtered at ingestion), 32,867 exact duplicates.
Track 2: **311 by sha256** (Sāmayik train 170, Itihāsa train 40, Sangraha 101,
Wikipedia 0), **320,320 by the 24-letter shingle filter**, and 5,554,868 exact duplicates;
2,266 further lines (Sāmayik train 2,005, Itihāsa train 261) were dropped by
`is_clean_slp1` applied on its own to the sources that do not go through the quality filter.
Shingle drops by evaluation source (counts overlap — a line quoting two sets is counted
under both): `itihasa_test` 130,534, `dcs_heldout` 106,922, `itihasa_dev` 80,549,
`samayik_dev` 4,653, `samayik_test` 4,448, `samayik_test_ood` 2,142, `flores_devtest` 84.
The Track 2 figures are far below the first build's because the quality filter now removes
most of the offending Sangraha lines before either leakage layer sees them.
Deduplication compares 64-bit blake2b digests rather than the strings (32M lines), so it is
exact to within a collision probability of ~1e-4 lines, recorded as `dedup_digest_bits`.
