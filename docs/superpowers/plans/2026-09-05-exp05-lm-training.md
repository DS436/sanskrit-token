# Experiment 05 (phase A): M1 Corpus and LM Training Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the M1 monolingual corpus with full leakage filtering, and build a GPU-agnostic LM training and BPC evaluation pipeline validated end to end on this machine (MPS) with a smoke sweep, so that the real Experiment 05 sweep is a single command on a rented GPU.

**Architecture:** `data/sangraha.py` and `data/wikipedia_sa.py` loaders; `experiments/05_lm_training/build_corpus.py` writes Track 1 (DCS raw / DCS oracle-split) and Track 2 (M1) corpora with manifests; `sanskrit_tok/lm/` holds vendored nanoGPT `model.py`, `data.py` (tokenise a corpus with a registry arm into memmapped token ids; byte mode for T7), `bpc.py` (pure metric), `train.py` (config-driven training loop with periodic BPC eval and a results jsonl), `sweep.py` (runs a grid of arm × size × seed, resumable), `aggregate.py` (results.json with mean ± std, tokens-to-reference-BPC, figure).

**Tech Stack:** Python 3.11, `uv`, `torch` (MPS/CUDA/CPU), `numpy`, `datasets`/`pyarrow` for parquet, HF `tokenizers`, `matplotlib`, `pytest`, `ruff`, `mypy --strict`.

Repo: /Users/devanshsharma/Desktop/Project/sanskrit-token, `main`, `origin`. Experiments 01–04 complete. Patterns: `src/sanskrit_tok/experiment.py`, `experiments/04_morph_constrained/{ingest_dcs,run}.py`, `src/sanskrit_tok/data/{dcs,exclusion,parallel}.py`, `src/sanskrit_tok/tokenizers/registry.py` (`load_tokenizer`, `LoadedTokenizer.encode`, file-backed `_dcs` arms).

## Global Constraints

- `uv`-managed; new deps via `pyproject.toml` + `uv lock` (`pyarrow` if not transitively present). Type hints; `ruff` and `mypy --strict` clean per experiment dir; pure metric in `lm/bpc.py`; I/O only in `data/`, `lm/data.py`, `lm/train.py`, `experiments/`.
- YAML config resolved against repo root; `pathlib`; `logging`; strict-JSON `results.json` via `write_results` with provenance, hardware (`torch.__version__`, device name), throughput (tokens/s), config and seed.
- Internal encoding SLP1; original script kept in raw jsonl only.
- **BPC is the headline (bits per SLP1 character); perplexity is never reported.** Comparisons at equal training bytes; tokens and FLOPs recorded. Three seeds minimum in real configs; the smoke config may use one.
- No leakage: every LM training line asserted against `data/exclusion_hashes.txt` (sha256) AND the 24-letter shingle index of all evaluation sets (parallel dev/test splits, FLORES devtest, DCS held-out); drops recorded per source.
- Arm names exactly: Track 1 `T1_bpe_raw_64k_dcs`, `T2_unigram_raw_64k_dcs`, `T4_bpe_split_64k_oracle_dcs`, `T5_morphbpe_rawseg_64k_dcs`, `T5_morphbpe_raw_64k_dcs`, `T6_morphbpe_split_64k_dcs`, `T7_byt5`; Track 2 the five raw ones. `T7_byt5` is a byte-level vocabulary (256 + `<eos>`), registered in the registry as an arm with `family "T7"` whose `encode` returns UTF-8 byte values.
- Vendored nanoGPT: `src/sanskrit_tok/lm/model.py` with the MIT licence text and origin URL/commit in the header; only `model.py` is vendored.
- `docs/decisions.md` append-only, date 2026-09-05. Commit prefixes `data:`, `lm:`, `exp05:`, `docs:`; commits end with a blank line then `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Tests offline, < 60 s, tiny fixtures; the smoke sweep is NOT a test (it is run once and its results recorded under `outputs/05_lm_training/smoke/`).
- Long jobs via `nohup` + polling; everything resumable.

---

### Task 1: M1 corpus assembly and Track 1/2 LM corpora

**Files:**
- Create: `src/sanskrit_tok/data/sangraha.py`, `src/sanskrit_tok/data/wikipedia_sa.py`, `experiments/05_lm_training/build_corpus.py`, `experiments/05_lm_training/corpus.yaml`, `tests/test_lm_corpus.py`, `tests/fixtures/sangraha_mini.parquet` (write it in-test or as a tiny committed parquet, ≤ 5 KB), `experiments/05_lm_training/README.md` (hypothesis, tracks, status, "GPU needed for real sweep")
- Modify: `data/README.md` (Sangraha, Wikipedia rows), `docs/decisions.md`

**Interfaces:**
- `sangraha.py`: `SANGRAHA_REPO = "ai4bharat/sangraha"`, `SANGRAHA_REVISION` (pin the dataset revision sha via `huggingface_hub.dataset_info(...).sha`), `list_verified_sanskrit_files() -> list[str]` (files under the `verified` split for Sanskrit — inspect the repo tree: the path is `verified/san/*.parquet` or similar; if `verified` has no Sanskrit, report and fall back to nothing — never use `synthetic`), `iter_sangraha_sanskrit(cache_dir) -> Iterator[str]` (download each parquet with `huggingface_hub.hf_hub_download` to `data/raw/sangraha/`, read with `pyarrow`, yield the text column's documents), `documents_to_lines(doc) -> list[str]` (split on danda `।`/`॥` and newlines, strip, drop lines with < 2 whitespace words or without Devanagari letters).
- `wikipedia_sa.py`: `WIKIPEDIA_REPO = "wikimedia/wikipedia"`, config `20231101.sa`, single parquet; same iterator shape; strip wiki boilerplate minimally (headings, `==` markers); same line splitting.
- `build_corpus.py` (`--config corpus.yaml`): sources with keys `dcs_train` (from `data/processed/dcs/train.jsonl`: `text_slp1` and `oracle_split_slp1`), `samayik_train_sa`, `itihasa_train_sa` (Devanagari → SLP1), `sangraha_verified_san`, `wikipedia_sa` (Devanagari → SLP1); outputs `data/processed/lm/track1_raw.txt` (DCS raw, one sentence per line), `data/processed/lm/track1_split.txt` (DCS oracle split, same sentences same order), `data/processed/lm/track2_raw.txt` (all sources, raw SLP1), each with a manifest (`n_in` per source, `n_dropped_hash`, `n_dropped_shingle` per eval source, `n_dedup_removed`, `n_out`, `n_chars`, `n_bytes`, `sha256`, provenance) and `data/processed/lm/heldout_*.txt` evaluation texts: `heldout_dcs.txt` (DCS held-out `text_slp1`), `heldout_dcs_split.txt` (oracle split), and the four parallel eval Sanskrit sides raw (`heldout_samayik_test.txt`, `heldout_samayik_test_ood.txt`, `heldout_itihasa_test.txt`, `heldout_flores_devtest.txt`) plus their ByT5-reconciled split forms (`heldout_*_split.txt`) from `data/processed/split/*.jsonl`. Leakage: assert both layers on every training line; record drops. Exact dedup within each output. Streaming throughout.

- [ ] Step 1: failing tests: `documents_to_lines` on a fixture document (danda splitting, short-line drop, Devanagari check); Sangraha parquet reading via an in-test parquet; wikipedia boilerplate stripping; `build_corpus` on a tmp config with mini sources producing the three corpora + heldout files + manifests, with a planted leaked line dropped by hash and another by shingle.
- [ ] Step 2: implement; run for real in the background (Sangraha download size unknown — report it); report manifests: per-source counts, drops, `n_out`, chars, bytes, and the token count of each corpus under `T1_bpe_raw_64k_dcs` (stream-encode; this is the "≈ tokens available" number for the tracks). Update `data/README.md`; decisions entry with the real numbers.
- [ ] Step 3: lint/type/test clean. Commit `data: M1 monolingual corpus and Experiment 05 LM corpora with two-layer leakage filtering`.

---

### Task 2: LM pipeline — vendored nanoGPT, data packing, BPC, training loop

**Files:**
- Create: `src/sanskrit_tok/lm/model.py` (vendored nanoGPT, MIT header), `src/sanskrit_tok/lm/data.py`, `src/sanskrit_tok/lm/bpc.py`, `src/sanskrit_tok/lm/train.py`, `src/sanskrit_tok/lm/config.py`, `tests/test_lm.py`, `tests/fixtures/lm_mini_corpus.txt` (≈ 300 SLP1 lines synthetic)
- Modify: `src/sanskrit_tok/tokenizers/registry.py` (`T7_byt5` byte arm), `src/sanskrit_tok/lm/__init__.py`, `tests/test_registry.py`, `pyproject.toml` if a dep is missing

**Interfaces:**
- `data.py`: `encode_corpus(arm: LoadedTokenizer, corpus_path: Path, out_path: Path, *, eos_id: int | None) -> EncodedCorpus` writing a `uint16`/`uint32` `.bin` memmap (dtype by vocab size) plus `.meta.json` (`n_tokens`, `n_chars`, `n_bytes`, `n_lines`, `dtype`, `arm`, `sha256` of source); one EOS per line (use the arm's `[UNK]`? No: append a dedicated EOS id = `vocab_size` and tell the model `vocab_size + 1`; for T7 EOS = 256); `EncodedCorpus` dataclass; `iter_batches(bin_path, block_size, batch_size, rng) -> Iterator[tuple[Tensor, Tensor]]` random windows (nanoGPT style).
- `bpc.py` (pure): `bits_per_char(total_nll_nats: float, n_chars: int) -> float`; `bpc_from_token_nll(token_nll: Sequence[float], n_chars: int) -> DetailedMetricResult` (`value` BPC, `n` chars, `unit "bits/char"`, `total_nats`, `n_tokens`, `bits_per_token`); tests hand-computed (e.g. 10 tokens each nll ln2 over 20 chars → 0.5 bits/char).
- `train.py`: `TrainConfig` (from `config.py`: `arm`, `track`, `size` (`smoke|50M|125M` → n_layer/n_embd/n_head), `block_size 1024`, `batch_size`, `grad_accum`, `max_steps` or `max_tokens`, `lr`, `warmup`, `weight_decay`, `seed`, `device auto|mps|cuda|cpu`, `dtype auto`, `eval_every`, `eval_sets` (names → heldout paths), `eval_max_chars`), `train(config) -> Path` writing `outputs/05_lm_training/<track>/<size>/<arm>/seed<k>/{results.json, config.yaml, curve.jsonl, ckpt.pt}`: sets all seeds; builds the model with `vocab_size = arm vocab + 1`; counts non-embedding and total params; logs tokens/s; every `eval_every` steps computes BPC on each eval set by full-sequence NLL over the held-out text tokenised with the same arm (chunked at block_size with stride = block_size, counting every character once; clamp `eval_max_chars` for smoke) and appends to `curve.jsonl` (`step, tokens_seen, bytes_seen, flops_est (6*N*tokens), bpc per set, loss, lr, elapsed`); resumable from `ckpt.pt`; `results.json` has final BPC per set, curve summary, params, throughput, hardware, provenance.
- `T7_byt5` registry arm: `family "T7"`, `vocab_size 256`, `encode(text) = list(text.encode("utf-8"))`, spans supported (one byte per char in ASCII SLP1).

- [ ] Step 1: failing tests: bpc hand cases; `encode_corpus` on the mini corpus with a tiny in-test trained tokenizer and with `T7_byt5` (token counts, EOS per line, dtype); `iter_batches` shapes/determinism by seed; `train` with `size: smoke`, 5 steps, cpu, on the mini corpus writes results.json/curve.jsonl with finite BPC and resumes from ckpt for 5 more steps (step counter continues); `T7_byt5` in the registry.
- [ ] Step 2: implement; vendor `model.py` from github.com/karpathy/nanoGPT (record the commit sha in the header); run one real smoke training on MPS: `T1_bpe_raw_64k_dcs`, Track 1, `size smoke`, 200 steps, batch 8, block 256, eval every 50 on `heldout_dcs` capped at 200k chars; report tokens/s on MPS vs cpu and the BPC curve (must decrease).
- [ ] Step 3: lint/type/test clean. Commit `lm: vendored nanoGPT, corpus encoding, BPC evaluation and resumable training loop`.

---

### Task 3: Sweep, aggregation, figure, smoke sweep on MPS

**Files:**
- Create: `experiments/05_lm_training/{sweep.yaml,smoke.yaml,sweep.py,aggregate.py,run.py}`, `tests/test_exp05.py`
- Modify: `experiments/05_lm_training/README.md`, `docs/decisions.md`

**Interfaces:**
- `sweep.yaml`: `tracks: {track1: {corpus_raw, corpus_split, arms: [the seven], sizes: [50M]}, track2: {corpus_raw, arms: [the five raw], sizes: [50M, 125M]}}`, `seeds: [0, 1, 2]`, `block_size 1024`, per-size `batch_size`/`grad_accum`/`lr`/`max_tokens` (equal training BYTES across arms: express the budget as bytes of the corpus × epochs, converted to tokens per arm from the encoded meta), `eval_every`, `eval_sets` (in-domain `heldout_dcs` (+ `_split` for split arms), out-of-domain four parallel sets (+ `_split` for split arms)), `output_dir`. `smoke.yaml`: three arms (`T1_bpe_raw_64k_dcs`, `T5_morphbpe_rawseg_64k_dcs`, `T7_byt5`), `size smoke`, 1 seed, 300 steps, block 256, eval sets capped.
- `sweep.py`: `--config` → enumerates runs, skips completed (results.json present with matching config hash), encodes each corpus × arm once (cached `.bin`), runs `train` sequentially, logs progress; `--dry-run` prints the plan with estimated tokens, FLOPs and, given a `tokens_per_s` argument, hours.
- `aggregate.py` → `outputs/05_lm_training/<sweep-name>/results.json`: per track × size × arm: final BPC per eval set mean ± std over seeds, params, tokens, bytes, FLOPs; tokens/bytes/FLOPs-to-reference-BPC (reference = `T1_bpe_raw_64k_dcs` final in-domain BPC at that track/size; per seed then mean ± std; undefined → null); figure `bpc_curves.{pdf,png}`: BPC vs training bytes per arm (mean over seeds, shaded std), one panel per track/size, dashed reference line; a second figure `bpc_final.{pdf,png}`: bars of final in-domain and OOD BPC per arm with error bars.
- `run.py`: `--sweep smoke|sweep` convenience wrapper: `sweep.py` then `aggregate.py`.
- README: H4 (BPC half) and outline §6 protocol; both tracks; the 24–33% token handicap caveat for T6/T5 from Exp04; "how to run on a rented GPU" (uv sync; `uv run python experiments/05_lm_training/run.py --sweep sweep`; expected hours from `--dry-run`); smoke results table (BPC per arm, not a scientific result); status.

- [ ] Step 1: failing tests: sweep enumeration/skip logic on a tmp output dir; equal-bytes → per-arm token budget conversion; aggregation with synthetic per-seed results (mean ± std, reference crossing, undefined case); figures on synthetic data.
- [ ] Step 2: implement; run the smoke sweep on MPS for real (three arms, 300 steps each) and aggregate; run `sweep.py --dry-run --config sweep.yaml --tokens-per-s <measured MPS rate>` and report the projected hours on MPS and, scaled by a stated factor, on an A100 (assume 50× the MPS token rate; label it an assumption).
- [ ] Step 3: README filled (smoke table, projections, GPU instructions); decisions entry; lint/type/test clean. Commit `exp05: sweep runner, aggregation, figures and MPS smoke sweep`.
