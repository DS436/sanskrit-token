# Experiment 04: Morpheme-Constrained Tokenizers (T5, T6) and MorphScore — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer RQ4 (TPP/MorphScore half) and the MorphScore half of RQ3: does forbidding merges across gold morpheme boundaries (T5 on raw, T6 on gold-split text) raise MorphScore and lower tokens-per-proposition relative to matched unconstrained arms trained on the same DCS corpus?

**Architecture:** Ingest DCS CoNLL-U into SLP1 sentences with gold segment boundaries aligned into the sandhied surface and a derived stem/ending boundary; add token spans to the tokenizer protocol and a MorphScore metric; implement the hard MorphBPE constraint as boundary-marker pre-tokenisation; train twelve `_dcs` arms on the DCS training split with the shared trainer and exclusion assertion; run one experiment script for MorphScore on DCS held-out and paired TPP deltas on the parallel corpora.

**Tech Stack:** Python 3.11, `uv`, HF `tokenizers`, `transformers`, `tiktoken`, `numpy`, `matplotlib`, `pytest`, `ruff`, `mypy --strict`, `git` sparse checkout.

Repo: /Users/devanshsharma/Desktop/Project/sanskrit-token, branch `main`, remote `origin`. Experiments 01–03 complete. Established patterns: `src/sanskrit_tok/experiment.py` (helpers), `experiments/03_sandhi_split/run.py`, `experiments/02_tpp_parallel/train_tokenizers.py` (sides, `select_training_sentences`, skip-if-trained), `src/sanskrit_tok/tokenizers/{corpus,registry,_train_common,train_bpe,train_unigram}.py`, `src/sanskrit_tok/data/exclusion.py`, `src/sanskrit_tok/sandhi/reconcile.py`.

## Global Constraints

- Python 3.11+, `uv`-managed; add deps only via `pyproject.toml` + `uv lock`.
- Type hints; `uv run ruff check .` and `uv run mypy src experiments/04_morph_constrained` clean. Pure metrics; I/O only in `data/`, `tokenizers/train_*.py`/`morph_bpe.py`, `experiment.py`, `experiments/`.
- YAML config resolved against repo root; `pathlib`; `logging`; `results.json` strict JSON via `write_results` with provenance and file hashes; both figure formats.
- Internal encoding SLP1 (DCS is IAST: convert with `to_slp1(text, "iast")`; store the IAST original alongside).
- **TPP paired deltas are the headline; MorphScore is the mechanism check; fertility never leads.** Prose before verse. Matched vocab 32k/64k. `_dcs` arms are trained on the same DCS split with the same trainer; provisional arms and existing-practice arms are labelled as such with vocab sizes.
- Registry names exactly as the decisions entry lists them. MorphScore: `value` = F1, `precision`/`recall` extras; exclude single-token words and single-morpheme words; report primary (segment boundaries) and secondary (derived stem/ending, labelled heuristic).
- No leakage: DCS held-out sentences added to `data/exclusion_hashes.txt` (committed); DCS training asserted against the full list; parallel-corpus eval sentences checked present.
- Every text transformation measured records `text_invariants` and a re-tokenised non-letter deletion cost (helpers exist in `sanskrit_tok.experiment`).
- `docs/decisions.md` append-only, date 2026-09-05. Commit prefixes `data:`, `metric:`, `tok:`, `exp04:`, `docs:`; commits end with a blank line then `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Tests offline, < 60 s, tiny fixtures (a mini CoNLL-U file written from the DCS sample in the brief). Long jobs via `nohup` + polling, resumable.

---

### Task 1: DCS ingestion with aligned gold boundaries

**Files:**
- Create: `src/sanskrit_tok/data/dcs.py`, `src/sanskrit_tok/data/boundaries.py`, `experiments/04_morph_constrained/ingest_dcs.py`, `experiments/04_morph_constrained/dcs.yaml`, `tests/test_dcs.py`, `tests/fixtures/dcs_mini.conllu`
- Modify: `data/README.md` (DCS row: pinned commit, date, licence CC BY 4.0, counts, filtering), `data/exclusion_hashes.txt` (+ DCS held-out; regenerate through `build_exclusion.py` extended with a `dcs_heldout` source), `experiments/02_tpp_parallel/build_exclusion.py`, `docs/decisions.md`

**Interfaces:**
- `dcs.py`: `DCS_REPO = "OliverHellwig/sanskrit"`, `DCS_COMMIT` (pin the `master` head SHA at download), `DCS_CONLLU_DIR = "dcs/data/conllu/files"`; `download_dcs(dest: Path, commit: str) -> Path` using `git clone --filter=blob:none --no-checkout` + `git sparse-checkout set dcs/data/conllu/files` + `git checkout <commit>` into `data/raw/dcs/` (gitignored), resumable; `iter_conllu_sentences(path) -> Iterator[DcsSentence]`; `@dataclass(frozen=True) class DcsToken: id: int; surface_iast: str; lemma_iast: str; upos: str; feats: str; unsandhied_iast: str; reconstructed: bool; is_cpd: bool`; `@dataclass(frozen=True) class DcsWord: surface_iast: str; tokens: tuple[DcsToken, ...]` (one per multiword range or single token); `@dataclass(frozen=True) class DcsSentence: sent_id: int; text_id: int; text_name: str; chapter: str; text_iast: str; words: tuple[DcsWord, ...]`; `sentence_is_human_verified(s) -> bool` (no token `reconstructed`).
- `boundaries.py` (pure): `align_segments(surface_slp1: str, segments_slp1: Sequence[str]) -> list[int] | None` — boundary offsets (character indices in the surface where a new segment begins, excluding 0 and len) via `difflib.SequenceMatcher` over the concatenation, `None` if the alignment ratio < 0.6 or offsets are non-monotone; `stem_boundary(segment_slp1: str, lemma_slp1: str) -> int | None` — LCP length if ≥ 2 and < len(segment), else `None`; `mark(text: str, offsets: Sequence[int], marker: str = "\x1f") -> str`; `GoldSentence` dataclass: `text_slp1` (sandhied), `oracle_split_slp1` (segments joined by single spaces, words joined by single spaces), `segment_offsets: list[list[int] | None]` per word (into the sandhied word), `stem_offsets: list[list[int]]` per word (into the oracle-split segments, absolute in `oracle_split_slp1`), `t5_marked` (sandhied text with `\x1f` at aligned segment offsets and at aligned stem offsets where alignment succeeded), `t6_marked` (oracle split with `\x1f` at stem offsets), `n_words`, `n_words_aligned`, `human_verified: bool`.
- `ingest_dcs.py`: `--config dcs.yaml` (`repo`, `commit`, `raw_dir: data/raw/dcs`, `out_dir: data/processed/dcs`, `heldout_fraction: 0.05`, `seed: 0`, `min_words: 2`); downloads, parses every `.conllu`, converts IAST→SLP1 (`to_slp1(x, "iast")`), builds `GoldSentence`s, assigns held-out by whole text (`text_id`) with the seed, drops sentences whose sandhied SLP1 hashes into the existing exclusion list (record count), writes `train.jsonl` and `heldout.jsonl` (one `GoldSentence` per line plus `sent_id`, `text_id`, `text_iast`), and `manifest.json` (commit, counts per split, texts per split, `n_dropped_excluded`, `n_words`, `n_words_aligned`, alignment rate, human-verified fraction, provenance). Progress logs every 50 files. Then extend `build_exclusion.py` with source `dcs_heldout` (sandhied SLP1 of held-out sentences; hash via `sentence_hash` of the Devanagari form? — no: DCS is IAST; add `sentence_hash_slp1(text_slp1)` = sha256 of the SLP1 string, and note that `sentence_hash` for Devanagari inputs is equivalent) and regenerate `data/exclusion_hashes.txt`.

- [ ] Step 1: write `tests/fixtures/dcs_mini.conllu` from the sample in the brief (the three Acintyastava sentences) and failing tests: parser yields 3 sentences with correct multiword grouping (`pratītyajānāṃ` → 2 tokens; `asamajñānam` → 3), `reconstructed` flags, `is_cpd`; `align_segments("tadapi", ["tad","api"]) == [3]`; alignment `None` on garbage; `stem_boundary("BAvAnAm","BAva") == 4`; `mark`; `GoldSentence` construction for the fixture with expected `oracle_split_slp1`; held-out assignment deterministic by text; exclusion drop.
- [ ] Step 2: implement; run the ingestion for real in the background (clone is ~hundreds of MB); report manifest numbers; regenerate and COMMIT the exclusion list; update `data/README.md`; decisions entry with the actual commit SHA, counts, alignment rate, human-verified fraction.
- [ ] Step 3: lint/type/test clean. Commit `data: ingest DCS CoNLL-U with aligned gold boundaries and held-out split`.

---

### Task 2: Token spans and MorphScore

**Files:**
- Create: `src/sanskrit_tok/metrics/morphscore.py`, `tests/test_morphscore.py`
- Modify: `src/sanskrit_tok/tokenizers/base.py` (`TokenizerWithSpans` Protocol: `spans(text) -> list[tuple[int, int]]` character offsets, half-open, in order, non-overlapping, covering every non-space character), `src/sanskrit_tok/tokenizers/registry.py` (`LoadedTokenizer.spans` implemented for: `tokenizers` file-backed arms via `encode(...).offsets`; HF fast via `return_offsets_mapping=True`; tiktoken via cumulative UTF-8 byte decoding of token bytes mapped to char offsets; `T0_gemma3` reloaded with `use_fast=True` when spans are requested — record the substitution), `tests/test_registry.py`, `docs/decisions.md`

**Interfaces:**
- `morphscore.py`: `morphscore(tokenizer: TokenizerWithSpans, words: Sequence[str], gold_offsets: Sequence[Sequence[int]]) -> DetailedMetricResult` — for each word with ≥ 2 tokens AND ≥ 1 gold boundary: token boundaries = set of span starts excluding 0; precision = |token ∩ gold| / |token|, recall = |token ∩ gold| / |gold|, pooled over words; `value` = F1, extras `precision`, `recall`, `n` = words scored, `n_excluded_single_token`, `n_excluded_single_morpheme`, `per_word_f1`, `n_undefined`; docstring cites Arnett & Bergen and states the exclusions. Tests hand-computed with a fake spans tokenizer: word `tadapi` gold [3], tokens `tad|api` → P=R=F1=1; tokens `ta|dapi` → 0; tokens `t|ad|api` → P=0.5, R=1; single-token word excluded; single-morpheme word excluded; a mix pooled correctly.
- Span adapters tested offline with a fake and, for `tokenizers` file-backed arms, with an in-test trained tokenizer; network-gated for HF/tiktoken (tiktoken test checks spans cover `"hello world"` exactly).

- [ ] Step 1: failing tests; Step 2: implement; Step 3: lint/type/test clean. Commit `metric: token spans protocol and MorphScore`.

---

### Task 3: MorphBPE constraint and the twelve `_dcs` arms

**Files:**
- Create: `src/sanskrit_tok/tokenizers/morph_bpe.py`, `experiments/04_morph_constrained/tokenizers.yaml`, `experiments/04_morph_constrained/train_tokenizers.py` (thin: reuses `experiments/02_tpp_parallel/train_tokenizers.py` machinery by import — move whatever must be shared into `sanskrit_tok.tokenizers.training` first), `tests/test_morph_bpe.py`
- Modify: `src/sanskrit_tok/tokenizers/registry.py` (twelve `_dcs` arms; `family` from prefix; `variant` field `"dcs"`/`"oracle_dcs"`), `src/sanskrit_tok/tokenizers/train_bpe.py` (accept an optional `boundary_marker: str | None`; when set, pre-tokenizer = `Sequence([Metaspace(), Split(marker, behavior="removed")])`, and the trainer's `initial_alphabet` must not contain the marker), `tests/test_registry.py`, `tests/test_train_tokenizers.py`, `docs/decisions.md`

**Interfaces:**
- `morph_bpe.py`: `BOUNDARY_MARKER = "\x1f"`, `train_morph_bpe(corpus_path, vocab_size, out_dir, *, seed=0) -> Path` = `train_bpe(..., boundary_marker=BOUNDARY_MARKER)`; `assert_no_cross_boundary_merges(tokenizer_json: Path, marked_corpus: Path, sample: int = 2000) -> dict` — tokenises a sample of marked sentences with markers removed and checks that no token span crosses a marker position; returns counts (must be 0 violations; used as a test and recorded in the arm's results.json).
- `tokenizers.yaml`: four corpora built from `data/processed/dcs/train.jsonl`: `dcs_raw` (`text_slp1`), `dcs_oracle_split` (`oracle_split_slp1`), `dcs_raw_marked` (`t5_marked`), `dcs_split_marked` (`t6_marked`); exclusion asserted on the SLP1 sandhied text via `sentence_hash_slp1`; arms: `T1_bpe_raw_{32k,64k}_dcs` (bpe, dcs_raw), `T2_unigram_raw_{32k,64k}_dcs` (unigram, dcs_raw), `T4_bpe_split_{32k,64k}_oracle_dcs` (bpe, dcs_oracle_split), `T4_unigram_split_{32k,64k}_oracle_dcs` (unigram, dcs_oracle_split), `T5_morphbpe_raw_{32k,64k}_dcs` (morph_bpe, dcs_raw_marked), `T6_morphbpe_split_{32k,64k}_dcs` (morph_bpe, dcs_split_marked); `output_dir: outputs/tokenizers`.
- Each arm's `results.json` records the corpus manifest, `boundary_marker`, and for T5/T6 the `assert_no_cross_boundary_merges` result.

- [ ] Step 1: failing tests: marker pre-tokenizer never yields a token containing the marker; a toy marked corpus (`tad\x1fapi`, `rAma\x1fH`) trained at vocab 40 never merges across the marker while the unmarked control does; registry lists the twelve names with correct families/variants; yaml parsing.
- [ ] Step 2: implement; train all twelve arms (report corpus sizes, per-arm seconds, vocab, and the cross-boundary check); decisions entry.
- [ ] Step 3: lint/type/test clean. Commit `tok: MorphBPE boundary-marker constraint and twelve DCS-trained arms`.

---

### Task 4: Experiment 04 runner, results, figure, README

**Files:**
- Create: `experiments/04_morph_constrained/run.py`, `experiments/04_morph_constrained/config.yaml`, `tests/test_exp04.py`
- Modify: `experiments/04_morph_constrained/README.md`, `docs/decisions.md` if deviating

**Interfaces:**
- `config.yaml`: `dcs_heldout: data/processed/dcs/heldout.jsonl`; the four parallel corpora prose-first with `split_dir: data/processed/split`; `arms_morphscore`: all twelve `_dcs` arms + provisional `T1_bpe_raw_64k`, `T2_unigram_raw_64k`, `T4_bpe_split_64k` + existing practice `T0_o200k`, `T0_gemma3`, `T3_sarvam`; `arms_tpp`: the twelve `_dcs` arms; `paired_deltas`: `[[T5_morphbpe_raw_32k_dcs, T1_bpe_raw_32k_dcs], [T5_morphbpe_raw_64k_dcs, T1_bpe_raw_64k_dcs], [T4_bpe_split_32k_oracle_dcs, T1_bpe_raw_32k_dcs], [T4_bpe_split_64k_oracle_dcs, T1_bpe_raw_64k_dcs], [T6_morphbpe_split_32k_dcs, T4_bpe_split_32k_oracle_dcs], [T6_morphbpe_split_64k_dcs, T4_bpe_split_64k_oracle_dcs], [T6_morphbpe_split_32k_dcs, T1_bpe_raw_32k_dcs], [T6_morphbpe_split_64k_dcs, T1_bpe_raw_64k_dcs]]`; `english_pivots: {controlled: E1_bpe_64k, deployed: T0_o200k}` (E1 labelled cross-corpus); `n_bootstrap: 1000`, `seed: 0`, `ci: 0.95`; `output_dir: outputs/04_morph_constrained`.
- `run.py`: MorphScore on DCS held-out for every `arms_morphscore` arm: raw arms tokenise `text_slp1` (words = sandhied words, gold = aligned segment offsets; words with `None` alignment skipped and counted); split arms tokenise `oracle_split_slp1` (words = segments... no: words = oracle-split words joined, gold = stem offsets; and ALSO segment-level MorphScore on `text_slp1` is undefined for split arms — report primary MorphScore for raw arms on segment boundaries, and for split arms on stem boundaries within segments, clearly labelled; secondary for raw arms = stem offsets projected into the surface where aligned); both on all held-out and on the human-verified subset. TPP on the parallel corpora: raw `_dcs` arms on `raw_slp1`, split `_dcs` arms on the ByT5-reconciled `output`; vs E1 (cross-corpus, labelled) and o200k; `tpp_paired_delta` for every configured pair per corpus with deletion cost where the two sides tokenise different text; fertility (raw-word denominator) and compression; `text_invariants` for the oracle split vs raw on DCS held-out.
- Figure `constraint_effects.{pdf,png}`: left column per corpus = paired TPP deltas (8 pairs, CI, dashed 0); right = MorphScore F1 (primary) per arm on DCS held-out (human-verified subset), grouped by family, vocab in labels.
- README: H4 (TPP half) and H3 (MorphScore half) verbatim; what Exp04 can and cannot test (BPC in Exp05); results tables; summary leading with T6−T1_dcs deltas on Sāmayik test; caveats (DCS domain vs parallel eval domain, oracle split at train vs ByT5 split at eval, heuristic stem boundaries, DCS segmentation partly machine-generated with the human-verified subset numbers, E1 cross-corpus, verse).

- [ ] Step 1: failing tests (synthetic held-out jsonl, fake spans tokenizers, figure, pair validation, human-verified filter); Step 2: implement, run for real, fill README; Step 3: lint/type/test clean. Commit `exp04: morpheme-constrained tokenizers, MorphScore and paired TPP deltas`.
