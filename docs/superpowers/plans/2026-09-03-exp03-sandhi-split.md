# Experiment 03: Sandhi-Split Arms (T4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the TPP/fertility half of RQ3 — does reversing sandhi before subword learning (T4 arms) lower tokens-per-proposition and fertility relative to matched raw arms (T1/T2), under the matched English control (E1)?

**Architecture:** Extract the duplicated experiment helpers into `sanskrit_tok.experiment`; wrap ByT5-Sanskrit as a cached `SandhiSplitter`; split the evaluation sets and the tokenizer training corpus (full or subset per the throughput rule); train `T4_bpe_split_{32k,64k}` and `T4_unigram_split_{32k,64k}` on the split SLP1 corpus with the same trainer, exclusion and settings as T1/T2; run one experiment script producing `results.json`, the figure and a README.

**Tech Stack:** Python 3.11, `uv`, `torch` (CPU/MPS), `transformers` (`T5ForConditionalGeneration`), HF `tokenizers`, `numpy`, `matplotlib`, `pytest`, `ruff`, `mypy --strict`.

Repo: /Users/devanshsharma/Desktop/Project/sanskrit-token, branch `main`, remote `origin`. Experiments 01–02 complete. Read `experiments/02_tpp_parallel/run.py`, `experiments/02_tpp_parallel/train_tokenizers.py`, `src/sanskrit_tok/tokenizers/{corpus,registry}.py`, `src/sanskrit_tok/data/exclusion.py`, `src/sanskrit_tok/encoding.py` for the established patterns.

## Global Constraints

- Python 3.11+, `uv`-managed; add `torch` to `pyproject.toml` (`torch>=2.4,<3`) and `uv lock`; never `pip install` into the project env.
- Type hints everywhere; `uv run ruff check .` and `uv run mypy src experiments/<dir>` clean per experiment directory. Pure functions for metrics; I/O only in `data/`, `sandhi/`, `tokenizers/train_*.py`, `experiment.py`, and `experiments/`.
- Config via YAML resolved against the repo root; `pathlib`; `logging`; every results writer records `git_commit`, `git_dirty`, config, timestamp; strict JSON (NaN → null).
- Internal encoding SLP1; the splitter's IAST is an implementation detail hidden inside `sandhi/`.
- **TPP is the headline; fertility never leads.** Prose before verse (Sāmayik test, Sāmayik test_ood, Itihāsa test, FLORES devtest). Matched vocab 32k/64k. Provisional arms (T1/T2/E1/T4) carry `*` and the training-corpus caveat. Existing-practice arms show vocab sizes. Never equate fewer words with fewer tokens.
- No leakage: every training text asserted against `data/exclusion_hashes.txt` on the ORIGINAL Devanagari before splitting or transliteration; every evaluated sentence checked to be in the exclusion set.
- Splitter results cached under `data/processed/split/` (gitignored) keyed by sha256 of the Devanagari input; manifests record model id, revision, device, throughput.
- Registry names exactly: `T4_bpe_split_32k`, `T4_bpe_split_64k`, `T4_unigram_split_32k`, `T4_unigram_split_64k` (family `T4`), plus `T1_bpe_raw_32k_sub` etc. only if the subset rule fires.
- `docs/decisions.md` append-only, CLAUDE.md §11 format, date 2026-09-03. Commit prefixes `metric:`, `tok:`, `data:`, `exp03:`, `docs:`; commits end with a blank line then `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Tests offline by default, < 60 s, fake models for the splitter; network/model tests under `SANSKRIT_TOK_NETWORK_TESTS`.
- Long-running steps (splitting, training) are launched with `nohup ... &` writing a log file, polled with short `sleep` loops, and must be resumable (the cache makes re-runs skip finished work).

---

### Task 1: Shared experiment helpers in `sanskrit_tok.experiment`

**Files:**
- Create: `src/sanskrit_tok/experiment.py`, `tests/test_experiment.py`
- Modify: `experiments/01_baseline_penalty/run.py`, `experiments/02_tpp_parallel/run.py`, `experiments/02_tpp_parallel/train_tokenizers.py`, `experiments/02_tpp_parallel/build_exclusion.py` (only if it duplicates a helper), `src/sanskrit_tok/provenance.py` (fold into `experiment.py` or re-export; keep `provenance` importable), `tests/test_exp01.py`, `tests/test_exp02.py`, `tests/test_provenance.py`

**Interfaces:**
- Produces in `experiment.py`: `repo_root() -> Path` (parent of `src/`), `resolve_path(value: str | Path, root: Path | None = None) -> Path`, `load_config(path: Path) -> dict[str, Any]` (`yaml.safe_load`, must be a mapping), `provenance() -> dict[str, object]` (`git_commit`, `git_dirty`, `timestamp` ISO-8601 UTC), `select_aligned_indices(corpus: ParallelCorpus) -> list[int]` (indices where every language is non-blank after strip), `take_indices(texts, indices) -> list[str]`, `sanitize_json(obj) -> object` (NaN/inf → None, tuples → lists, Path → str), `write_results(results: Mapping[str, object], out_dir: Path, config_src: Path | None) -> Path` (strict `json.dump(allow_nan=False)` after sanitising, `indent=2`, copies the config file as `config.yaml` when given, returns the results path), `summarise_tpp(raw: Mapping[str, Any], *, ci: float) -> dict[str, object]` (moved from exp02 `_enrich_summary`).
- Every script imports these and deletes its local copies. Behaviour must be identical.

- [ ] Step 1: failing tests in `tests/test_experiment.py` for each helper (tmp git repo for provenance; NaN sanitising; `write_results` strict JSON and config copy; `select_aligned_indices` with a blank-line fixture; `load_config` rejecting a non-mapping YAML).
- [ ] Step 2: implement; rewire the three scripts; move/adjust existing tests that referenced the local helpers.
- [ ] Step 3: re-run Experiments 01 and 02 (`uv run python experiments/01_baseline_penalty/run.py`, `uv run python experiments/02_tpp_parallel/run.py`) and compare every metric leaf against the pre-refactor `results.json` files (ignore timestamp/git fields): zero differences.
- [ ] Step 4: `uv run pytest -q`, `uv run ruff check .`, `uv run mypy src experiments/01_baseline_penalty`, `uv run mypy src experiments/02_tpp_parallel` clean. Commit `exp03: extract shared experiment helpers into sanskrit_tok.experiment`.

---

### Task 2: `SandhiSplitter` wrapper with cache, torch dependency, throughput benchmark

**Files:**
- Create: `src/sanskrit_tok/sandhi/byt5.py`, `src/sanskrit_tok/sandhi/cache.py`, `tests/test_sandhi.py`, `experiments/03_sandhi_split/benchmark_splitter.py`, `experiments/03_sandhi_split/README.md` (replace placeholder with hypothesis + status; numbers come in Task 4)
- Modify: `pyproject.toml` (+`torch>=2.4,<3`; `uv lock`), `src/sanskrit_tok/sandhi/__init__.py` (export `SandhiSplitter`), `docs/decisions.md`

**Interfaces:**
- `cache.py`: `class SplitCache` backed by one jsonl file per cache (`{"key": sha256, "input": devanagari, "output": slp1_split}` per line), `get(text) -> str | None`, `put(text, output)`, `__len__`, `flush()`; key = `hashlib.sha256(text.strip().encode()).hexdigest()`; append-only writes, loaded fully on open, tolerant of a truncated last line (log WARNING, drop it).
- `byt5.py`: `SPLITTER_CANDIDATES = ("chronbmm/sanskrit5-multitask", "buddhist-nlp/byt5-sanskrit-analyzer-hackathon")`, `SEGMENTATION_PREFIX = "S "`, `MAX_BYTES = 512`; `class SandhiSplitter` with `__init__(self, model_id: str | None = None, device: str | None = None, cache: SplitCache | None = None, batch_size: int = 16)`, `pick_device() -> str` (`"mps"` if `torch.backends.mps.is_available()`, else `"cuda"` if available, else `"cpu"`), `split(texts: Sequence[str]) -> list[str]` returning SLP1 with spaces at every segment boundary; pipeline per text: Devanagari → `to_slp1` → `from_slp1(..., "iast")` → prefix → model.generate (`max_length=MAX_BYTES`, `num_beams=1`) → decoded IAST → `to_slp1(..., "iast")`; batching sorts by length for padding efficiency and restores order; cache hits skip the model; inputs longer than `MAX_BYTES` bytes in IAST are split on danda/`|`/`.` boundaries into chunks under the limit, processed separately and rejoined with a space (record `n_chunked`); `source_id` property = `f"{model_id}@{revision}"` via `huggingface_hub.model_info(...).sha` (fallback `"unknown"`); `stats` dict (`n_calls`, `n_cache_hits`, `n_model`, `n_chunked`, `seconds_model`).
- `benchmark_splitter.py`: `--n 200 --device auto|cpu|mps --batch-size 16`; loads the first N Sanskrit sentences of Sāmayik test (Devanagari), runs the splitter with a throwaway cache, prints sentences/s, projected hours for 117,720 and for 19,197 sentences, ten before/after examples (Devanagari input, IAST model output, SLP1 split output), and the fraction of sentences where the number of whitespace units increased; writes `outputs/03_sandhi_split/benchmark.json`.

- [ ] Step 1: failing tests (offline): `SplitCache` roundtrip, truncated-last-line tolerance, key normalisation; `SandhiSplitter.split` with a fake model object (monkeypatch `_generate_iast(batch) -> list[str]`) returning known IAST strings, asserting SLP1 output, order restoration after length sorting, cache hit on the second call (`stats["n_model"]` unchanged), and chunking of an over-long input. Model-loading test network-gated.
- [ ] Step 2: add torch, `uv lock`, `uv sync`; implement; run the benchmark for real on `mps` and `cpu` (N=200 each; if `mps` errors, record and use cpu); paste the throughput, projection, examples, and whether compounds are split (look at the examples: does `munipuṃgavam`-style compounding come out as two segments?).
- [ ] Step 3: append a decisions entry recording measured throughput per device, the resulting full-vs-subset decision per the "Splitter throughput rule", and the compound-splitting observation. Lint/type/test clean. Commit `data: ByT5-Sanskrit sandhi splitter with cache and throughput benchmark`.

---

### Task 3: Split the corpora and train the T4 arms

**Files:**
- Create: `experiments/03_sandhi_split/split_corpora.py`, `experiments/03_sandhi_split/split.yaml`, `tests/test_split_corpora.py`
- Modify: `experiments/02_tpp_parallel/tokenizers.yaml` (add T4 arms with `side: sa_split`, `corpus_path: data/processed/tok_train_slp1_split.txt`; and `_sub` raw arms only if the subset rule fired), `experiments/02_tpp_parallel/train_tokenizers.py` (new side `sa_split` whose corpus builder applies `transform = split_then_slp1` using the cache — the model is not loaded if every sentence is cached; subset support via `subset: {n: int, seed: int}` in `split.yaml` applied identically to the raw `_sub` arms), `src/sanskrit_tok/tokenizers/registry.py` (T4 arms, and `_sub` arms if needed, file-backed, `family` from prefix; `_sub` arms keep family `T1`/`T2` with `variant: "sub"`), `tests/test_registry.py`, `tests/test_train_tokenizers.py`, `docs/decisions.md`, `data/README.md` (row for the split cache: model id/revision, date, counts)

**Interfaces:**
- `split.yaml`: `model_id`, `device: auto`, `batch_size: 16`, `cache_path: data/processed/split/cache.jsonl`, `eval_corpora` (the four Exp02 corpora with loader/split), `train_sources: [samayik_train, itihasa_train]`, `subset: null | {n: int, seed: 0}`, `manifest_path: data/processed/split/manifest.json`.
- `split_corpora.py`: `main()`; splits every eval corpus's Sanskrit side and the training sources' Sanskrit side (post exclusion filter, so leaked sentences are never even split), all through the shared cache; resumable (re-run skips cached); writes the manifest (`model_id`, `revision`, `device`, `n_sentences` per corpus, `n_cache_hits`, `n_chunked`, `seconds`, `subset`, `git_commit`, `git_dirty`); logs progress every 500 sentences. Must be launched in the background (`nohup uv run python experiments/03_sandhi_split/split_corpora.py > outputs/03_sandhi_split/split.log 2>&1 &`) and polled.
- After splitting: `uv run python experiments/02_tpp_parallel/train_tokenizers.py` trains only the new arms (skip-if-trained already exists); the split-corpus manifest records `splitter_source_id`, `n_out`, and the fraction of sentences whose whitespace-unit count changed.

**Addendum (after Task 2):** the splitter output is not content-preserving (see decisions entry "T4 text is the model's segmentation reconciled against the raw sentence"). Add `src/sanskrit_tok/sandhi/reconcile.py` with `reconcile(raw_slp1: str, model_slp1: str, *, threshold: float = 0.6) -> ReconcileResult` (`dataclass`: `text: str`, `n_units_raw: int`, `n_units_out: int`, `n_units_kept_verbatim: int`, `chars_raw: int`, `chars_model: int`, `chars_out: int`): walk raw whitespace units in order; for each, greedily extend a window of model segments while `difflib.SequenceMatcher(None, raw_unit, "".join(window)).ratio()` increases and stays ≥ threshold; on success replace the unit by the window's segments (space-joined) and advance; otherwise keep the unit verbatim (always verbatim for units with no SLP1 letters: punctuation, digits, Latin). Tests: pure sandhi reversal (`tadapi` → `tad api`), compound split, dropped punctuation restored, dropped loanword restored, a model output that is empty (everything verbatim), and retention counts. `split_corpora.py` stores BOTH the raw model output and the reconciled text in the cache/jsonl (`output_model`, `output`), the corpus builder uses the reconciled text, and the manifest records mean character retention before/after per corpus. The `tests/test_sandhi.py` fake-splitter tests stay as they are.

- [ ] Step 1: failing tests: `split.yaml` parsing; the corpus transform composes split + SLP1 through a fake splitter; subset selection is deterministic for a seed and applied identically to raw and split sides; registry lists the four T4 arms (and `_sub` arms when configured) with correct families; manifest keys.
- [ ] Step 2: implement; run `split_corpora.py` in the background to completion (report total wall time, cache size, `n_chunked`); build the split corpus; train the T4 arms (and `_sub` arms if applicable); report per-arm vocab, seconds, and token counts for one Sāmayik-test sentence raw vs split under T1 vs T4.
- [ ] Step 3: decisions entry with actual numbers; data/README row; lint/type/test clean. Commit `tok: split corpora with ByT5-Sanskrit and train T4 arms`.

---

### Task 4: Experiment 03 runner, results, figure, README

**Files:**
- Create: `experiments/03_sandhi_split/run.py`, `experiments/03_sandhi_split/config.yaml`, `tests/test_exp03.py`
- Modify: `src/sanskrit_tok/metrics/fertility.py` (add `fertility_against_reference(tokenizer, texts, reference_texts) -> DetailedMetricResult`: tokens of `texts[i]` over whitespace words of `reference_texts[i]`, `unit = "tokens/reference word"`, `per_text` ratios with NaN where the reference has no words, `n` = reference word total, `n_undefined`), `tests/test_metrics.py`, `experiments/03_sandhi_split/README.md`, `docs/decisions.md` if deviating

**Interfaces:**
- `config.yaml`: the four corpora (prose first); `split_cache_path`; `arms_raw: [T1_bpe_raw_32k, T1_bpe_raw_64k, T2_unigram_raw_32k, T2_unigram_raw_64k]` (or `_sub` variants if the subset rule fired — then the matched comparison uses those), `arms_split: [T4_bpe_split_32k, T4_bpe_split_64k, T4_unigram_split_32k, T4_unigram_split_64k]`, `matched_pairs` (raw arm ↔ split arm at equal vocab and algorithm), `english_control: {T4_bpe_split_32k: E1_bpe_32k, ...}` and the same for raw arms, `deployed_pivot: T0_o200k`, `n_bootstrap: 1000`, `seed: 0`, `ci: 0.95`, `output_dir: outputs/03_sandhi_split`.
- `run.py`: loads corpora; Sanskrit side raw (SLP1) and split (from cache; abort with a clear error naming the missing sentence count if any evaluation sentence is not cached); leakage check both sides as in Exp02; for each corpus: TPP of every raw arm on raw text and every split arm on split text vs E1 (controlled) and vs o200k (deployed); `tpp_delta` per matched pair = split − raw with a paired bootstrap CI (resample pairs, recompute both ratios, difference); fertility primary (`fertility_against_reference(split_arm, split_texts, raw_texts)` for split arms, plain `fertility` for raw arms) and secondary (`fertility` on split text); compression on split vs raw; splitter descriptive stats per corpus (mean whitespace units raw vs split, fraction changed, `n_chunked`). `results.json` keys: provenance, config, `tokenizer_sources` (with sha256 and `splitter_source_id`), `corpora`, `exclusion_check`, `splitter_stats`, `tpp` (`corpus → arm → pivot → summary`), `tpp_delta` (`corpus → pair → {delta, ci_low, ci_high}`), `fertility_primary`, `fertility_secondary`, `compression`.
- Figure `tpp_split_vs_raw.{pdf,png}`: one row per corpus (prose first); x = the four matched pairs; two markers per pair (raw arm, split arm) with CI, controlled TPP vs E1 on the y-axis; dashed 1.0 line; caption states provisional status and the splitter id.
- README: H3 verbatim from the outline; what this experiment can and cannot test (MorphScore deferred, per decisions); success = split arms below raw arms with the delta CI excluding 0, on prose; results table per corpus (pairs × {raw TPP, split TPP, delta [CI], fertility primary raw/split}); summary leading with the controlled TPP delta on Sāmayik test; existing caveats (provisional, domain fit, verse meter, Hindi not applicable here); the splitter's compound behaviour and throughput.

- [ ] Step 1: failing tests: `fertility_against_reference` hand-computed (`["ab cd"]` split from reference `["abcd"]` under CharTokenizer: 4 tokens / 1 reference word = 4.0; NaN when the reference is blank); `tpp_delta` helper on synthetic counts with a fixed seed (identical sides → delta 0 with CI [0, 0]); `make_figure` on a synthetic results dict; the missing-cache abort.
- [ ] Step 2: implement; run for real; fill the README with every number from `results.json`; lint/type/test clean. Commit `exp03: sandhi-split TPP runner, results and figure`.
