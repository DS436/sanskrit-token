# Paper 1a (RQ1–RQ2 measurement paper) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A compilable workshop-length manuscript, `paper/1a/main.tex`, whose every number is generated from `results/` by script, reporting RQ1 (the penalty under deployed tokenizers) and RQ2 (tokens-per-proposition with the matched English control), plus two cheap additions reviewers of tokenizer papers expect: Rényi efficiency and a length-stratified TPP analysis.

**Scope decision (2026-09-07):** paper 1a is Experiments 01 and 02 only. The sandhi-split and morpheme-constrained arms (Exp03/04) and LM training (Exp05) are held for paper 1b, so 1b keeps its method novelty. 1a's thesis: the Sanskrit penalty under deployed tokenizers is large (RQ1), but the "Sanskrit is cheaper per proposition" sign flip seen against o200k disappears under a matched English control (RQ2) — the flip was the English pivot's domain handicap, not Sanskrit's density. Only verse stays below 1.0, and meter is the leading suspect.

Repo: `<repo>`, `main`, `origin`. Patterns: `src/sanskrit_tok/metrics/{compression,tpp,_ratio}.py`, `experiments/02_tpp_parallel/run.py`, `tests/test_tpp.py`, `tests/test_exp02.py`, `src/sanskrit_tok/experiment.py` (`summarise_tpp`, `write_results`).

## Global Constraints

- `uv` only; no ad-hoc `pip install`. Type hints; `ruff check` and `mypy --strict` clean on touched files. Pure metric functions in `metrics/`; I/O only in `experiments/` and `scripts/`.
- Metric contract: `(tokenizer, list[str])` → dict with at least `value`, `n`, `unit`; `require_texts` on input; unit tests against hand-computed examples; tests offline and < 60 s.
- Internal encoding SLP1. Fertility never the headline. Perplexity never appears. No "NASA chose Sanskrit" anywhere. Off-the-shelf arms are "deployed practice", never a controlled comparison.
- Every number in the paper comes from `results/*/results.json` via `scripts/paper_tables.py`; no hand-typed numbers in `.tex`.
- `docs/decisions.md` append-only, date 2026-09-07. Commit prefixes `metric:`, `exp02:`, `paper:`, `docs:`; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Research decisions (orchestrator, binding)

- **D1 Rényi efficiency.** `metrics/renyi.py`: `renyi_efficiency(tokenizer, texts, *, alpha)`; unigram distribution over the pooled token ids of `texts`; `H_α = log2(Σ p^α)/(1−α)` bits; `value` = `H_α / log2(n_types)` with `n_types` the observed support, matching Zouhar et al.'s reference implementation (`zouharvi/tokenization-scorer`, verify by reading its source); also return `entropy_bits`, `n_types`, `n_tokens`, and `efficiency_nominal = H_α / log2(vocab_size)` when a `vocab_size` kwarg is given. α ∈ {2.5, 3}. Docstring states it can be gamed (Cognetta et al. 2024) and is reported as a secondary intrinsic, never a headline.
- **D2 Length-stratified TPP.** Bins on the **English** whitespace word count of each pair, fixed across corpora so verse lines can be read against prose sentences of the same length: `[1,8], [9,16], [17,24], [25,40], [41,∞)`. Per corpus × bin: ratio of sums with the paired bootstrap, `n` pairs, mean English and Sanskrit word counts; bins with `n < 30` carry `sparse: true` and hollow markers. Computed for the four controlled pairs and for every Sanskrit arm against `T0_o200k` (SLP1 variant for T1/T2, original script for T0/T3).
- **D3 Paper.** ACL style files (`acl.sty`, `acl_natbib.bst` from `acl-org/acl-style-files`), `\aclfinalcopy` toggle, target 4 pages + appendix (short-paper form; trim later). Title working: *Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per Proposition*. Bibliography entries only from `docs/paper_outline.md` §12; any field not given there is marked `note={verify}`; no invented DOIs.

---

### Task 1: Rényi efficiency metric + Exp02 integration
**Files:** create `src/sanskrit_tok/metrics/renyi.py`, `tests/test_renyi.py`; modify `experiments/02_tpp_parallel/run.py` (`compute_renyi`, results keys `renyi` = corpus→arm→variant→alpha→summary on the Sanskrit side; `renyi_english` = corpus→arm→alpha→summary for `english_pivots` + every E1 arm named in `controlled_pairs`), `experiments/02_tpp_parallel/config.yaml` (`renyi_alphas: [2.5, 3.0]`), `tests/test_exp02.py`, `docs/decisions.md`.
- [x] Failing tests (hand-computed): uniform distribution → efficiency 1.0; single type → entropy 0, efficiency nan (log2(1)=0) reported as nan with `n_undefined`-style flag; a 3-type skewed example with the closed-form value; α=1 rejected (`ValueError`) since the Rényi limit is Shannon and not this function's business; empty input → value 0.0, n 0; `str` input → `TypeError`.
- [x] Implement; integrate; run Exp02 tests; ruff + mypy.

### Task 2: Length-stratified TPP + figure
**Files:** modify `src/sanskrit_tok/metrics/_ratio.py` (`RatioParts.subset(indices)`), `src/sanskrit_tok/metrics/tpp.py` (`tpp_from_parts(parts, *, n_bootstrap, seed, ci)` returning the same dict as `tpp`; `tpp` delegates to it), `experiments/02_tpp_parallel/run.py` (`length_bins`, `compute_tpp_by_length`, results key `tpp_by_length` = corpus→"sa_arm/en_arm"→bin_label→summary+`n`,`sparse`,`mean_words_en`,`mean_words_sa`, plus `length_bin_edges` recorded once; figure `tpp_by_length.{pdf,png}`: one panel per corpus, x = bin, y = TPP, one line per controlled pair, dashed y=1.0, hollow marker when sparse), `experiments/02_tpp_parallel/config.yaml` (`length_bin_edges: [1, 9, 17, 25, 41]`), `tests/test_tpp.py`, `tests/test_exp02.py`, `docs/decisions.md`.
- [x] Failing tests: `subset` preserves alignment; `tpp_from_parts` equals `tpp` on the same inputs with the same seed; bin assignment at the edges (8→bin 1, 9→bin 2, 41→last); sparse flag; figure writes both formats.
- [x] Implement; ruff + mypy.

### Task 3: Re-run Exp02, refresh snapshot, update Exp02 README
- [x] `uv run python experiments/02_tpp_parallel/run.py` at a clean tree; numeric diff of every pre-existing leaf against `results/02_tpp_parallel/results.json` must be zero; copy to `results/02_tpp_parallel/`; update `results/README.md` row; append the Rényi and by-length tables to `experiments/02_tpp_parallel/README.md` with two sentences of reading each.

### Task 4: Paper tables/figures scripts + LaTeX
**Files:** create `scripts/paper_tables.py` (reads `results/01_baseline_penalty/results.json`, `results/02_tpp_parallel/results.json`; writes `paper/1a/tables/*.tex` booktabs tables: fertility+compression by arm (FLORES), parity Sa/En and Sa/Hi, TPP deployed (o200k pivot) per corpus, TPP controlled per corpus, Rényi, TPP by length (controlled pairs)), `scripts/paper_figures.py` (`paper/1a/figures/`: parity bars; the "same Sanskrit arm, two English denominators" figure; TPP by length), `tests/test_paper_tables.py` (runs both scripts on the tracked snapshot, asserts every table file exists and every numeric cell matches the JSON source to the printed precision), `paper/1a/{main.tex,refs.bib,acl.sty,acl_natbib.bst,Makefile,README.md}`.
- [x] Draft sections per outline §9, condensed to short-paper form: Introduction (two-claims separation, Briggs 1985 once), Background (sandhi, samāsa, SLP1 for an NLP audience), Method (fertility vs parity vs TPP definitions; matched E1 control; bootstrap), Setup (corpora, arms, leakage control), Results (RQ1, RQ2 deployed, RQ2 controlled, verse, length, Rényi), Limitations (verse/meter, translation length bias, Dutt's English, T3 as deployed practice, single-language control), Conclusion. Pre-registered predictions stated with the outline date and their outcomes. Compile with `tectonic` (install via `brew install tectonic` if absent; if that fails, report and leave `Makefile` targeting `latexmk`).

### Task 5: Whole-branch review (orchestrator + Opus reviewer)
- [ ] Every claim in `main.tex` traced to a table cell or a `results.json` key; no fertility-led sentence; no cross-tokenizer perplexity; no "NASA".

---

## Revision wave 1 (2026-09-08): five reviewer objections, answered with measurements

Each item answers an objection a reviewer of the 1a draft would raise against the RQ2
result, and each is a *measurement* rather than a caveat. Research decisions D1–D8 of this
wave are recorded in `docs/decisions.md` (2026-09-08).

- [x] **R1 Byte-matched English control `E1_*_bm`.** The pair-matched control saw 16,554,871 bytes of English against the Sanskrit corpus's 11,209,356 — 48% more text at the same vocabulary size. Six new arms trained on a deterministic subsample of `tok_train_en.txt` cut to the Sanskrit byte count (`data/processed/tok_train_en_bm.txt` + manifest; `random.Random(0)` shuffle, prefix to the target).
- [x] **R2 128k arms on both sides.** `T1_bpe_raw_128k`, `T2_unigram_raw_128k`, `E1_bpe_128k`, `E1_unigram_128k` and the two `_bm` twins, so the controlled result can be checked against the vocabulary size it was measured at.
- [x] **R3 `T7_byt5` as a tokenizer-free reference.** Added to `sanskrit_arms` and as the controlled pair `[T7_byt5, T7_byt5]` — the ratio of the two sides' UTF-8 bytes, drawn as a dotted reference line on the controlled panel.
- [x] **R4 Side decomposition.** `results.json["side_decomposition"]`: every controlled and deployed pair factorised into `char_ratio` × `density_ratio` = `tpp`, so "the verse ratio" can be attributed to a shorter Sanskrit side or a longer English one.
- [x] **R5 Block bootstrap.** `tpp_from_parts(..., block_length=50)`: `ci_low_block`/`ci_high_block`/`block_length`/`n_blocks` beside the i.i.d. interval in every TPP summary, because consecutive verses and consecutive document sentences are not exchangeable.
- [x] **R6 Train the ten new arms**, without retraining any existing one (`select_arms_to_train` skips a trained arm; all 26 existing `tokenizer.json` sha256s verified unchanged). 32s wall-clock; `E1_unigram_128k` settles at 62,896 and `E1_unigram_{64k,128k}_bm` at 50,659, recorded rather than padded.
- [x] **R7 Re-run Exp02 at a clean tree** in a detached worktree, refresh `results/02_tpp_parallel/`, verify every pre-existing leaf is unchanged. Done at commit `40fd9c8`, 4m27s, exit 0; all 17,776 pre-existing leaves reappear unchanged (only the provenance keys and the `<repo>`-scrubbed tokenizer paths differ).
- [x] **R8 Update the Exp02 README, `results/README.md`, CLAUDE.md §6 and the decision log.**
