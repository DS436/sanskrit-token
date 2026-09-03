# Experiment 02: Tokens-per-Proposition on Parallel Text — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer RQ2 — on parallel Sanskrit–English text, does tokens-per-proposition (TPP) fall below 1.0 for a Sanskrit-native tokenizer and sit above 1.0 for English-centric ones — with bootstrap CIs, on prose first.

**Architecture:** Refactor the ratio metrics onto one shared core and add `tpp` with a paired bootstrap; extend the registry with `T0_gpt2`, the off-the-shelf Indic arms (T3) and file-backed trained arms (T1/T2); add Sāmayik and Itihāsa loaders sharing the existing `ParallelCorpus`; build the leakage exclusion list; train provisional T1/T2 tokenizers on SLP1; run one experiment script that writes `results.json` and the central TPP figure.

**Tech Stack:** Python 3.11, `uv`, HF `tokenizers` (training + loading), `transformers`, `tiktoken`, `numpy`, `matplotlib`, `pytest`, `ruff`, `mypy --strict`.

Repo: /Users/devanshsharma/Desktop/Project/sanskrit-token, branch `main`, remote `origin` (github.com/DS436/sanskrit-token). Experiment 01 is complete; read `experiments/01_baseline_penalty/run.py` for the established patterns (config resolution, `results.json` shape, figure function).

## Global Constraints

- Python 3.11+, `uv`-managed; never `pip install` into the project env; add deps to `pyproject.toml` and `uv lock`.
- Type hints everywhere; `uv run ruff check .` and `uv run mypy src experiments` clean. Pure functions for metrics; I/O only in `src/sanskrit_tok/data/`, `src/sanskrit_tok/tokenizers/train_*.py`, and `experiments/`.
- Config via YAML; relative paths resolve against the repo root; `pathlib` everywhere; `logging` not print.
- Internal encoding is SLP1 via `sanskrit_tok.encoding.to_slp1(text, "devanagari")`; original script stored alongside.
- **TPP is the headline metric.** Fertility is computed and reported but never leads a table, figure title, or README summary.
- **Prose before verse:** Sāmayik (test, test_ood) precedes Itihāsa precedes FLORES in every table, figure and config list.
- Matched vocabulary sizes 32k and 64k for trained arms; off-the-shelf arms are "existing practice", never a controlled comparison; their vocab sizes are always reported next to their numbers.
- Perplexity is never mentioned as a cross-tokenizer comparison.
- No evaluation leakage: tokenizer training text is asserted against `data/exclusion_hashes.txt` (sha256 of the SLP1 form of every evaluation sentence).
- Undefined per-item ratios are `float("nan")`, never `0.0`; every metric records `n_undefined`.
- Registry arm names must match CLAUDE.md §6 exactly: `T0_gpt2` (new), `T3_sarvam`, `T3_sutra`, `T3_indicsuper`, `T3_brahmic131k`, `T1_bpe_raw_32k`, `T1_bpe_raw_64k`, `T2_unigram_raw_32k`, `T2_unigram_raw_64k`.
- No "NASA chose Sanskrit" claim anywhere. Do not commit raw data (`data/raw/`, `data/processed/`, `outputs/` are gitignored). `data/exclusion_hashes.txt` IS committed.
- `docs/decisions.md` is append-only, `## YYYY-MM-DD — Title` + `Why:`/`Alternatives:`/`Reversible:`. Record every substitution, filter and deviation. Today's date is 2026-09-03.
- Tests offline by default, < 60 s, tiny fixtures in `tests/fixtures/`; network tests only under `SANSKRIT_TOK_NETWORK_TESTS`.
- Commit prefixes: `metric:`, `tok:`, `data:`, `exp02:`, `docs:`; every commit ends with a blank line then `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: Shared ratio core, NaN sentinels, `summary.py`, and `tpp`

**Files:**
- Create: `src/sanskrit_tok/metrics/_ratio.py`, `src/sanskrit_tok/metrics/summary.py`, `src/sanskrit_tok/metrics/tpp.py`, `tests/test_tpp.py`, `tests/test_summary.py`
- Modify: `src/sanskrit_tok/tokenizers/base.py` (extend `DetailedMetricResult`), `src/sanskrit_tok/metrics/parity.py`, `src/sanskrit_tok/metrics/compression.py`, `experiments/01_baseline_penalty/run.py` (delete local `summarise_metric`, import from `sanskrit_tok.metrics.summary`), `tests/test_metrics.py`, `tests/test_exp01.py` (move summarise tests to `tests/test_summary.py`)

**Interfaces:**
- Consumes: `Tokenizer` Protocol, `DetailedMetricResult`, `require_texts` from `sanskrit_tok.tokenizers.base`.
- Produces:
  - `_ratio.py`: `@dataclass(frozen=True) class RatioParts: source_counts: tuple[int, ...]; pivot_counts: tuple[int, ...]` with properties `source_total`, `pivot_total`, `value` (source_total / pivot_total, `nan` if pivot_total == 0), `per_pair` (list of `s/p`, `nan` where `p == 0`), `n_undefined`; and `token_ratio(tokenizer, texts, pivot_texts, pivot_tokenizer=None) -> RatioParts` (validates with `require_texts`, raises `ValueError` on length mismatch, defaults `pivot_tokenizer` to `tokenizer`).
  - `summary.py`: `summarise_values(values: Sequence[float]) -> dict[str, float | int | None]` returning `{"mean", "std", "n", "n_undefined"}` (nan-aware via `numpy.nanmean`/`nanstd` with `ddof=0`; `mean`/`std` are `None` when no finite values); `summarise_metric(result: Mapping[str, object]) -> dict[str, object]` moved verbatim in behaviour from `run.py` but using `summarise_values` and emitting `None` (not 0.0) when no distribution.
  - `tpp.py`: `tpp(tokenizer, texts, pivot_texts, pivot_tokenizer=None, *, n_bootstrap: int = 1000, seed: int = 0, ci: float = 0.95) -> DetailedMetricResult` with keys `value`, `n` (pairs), `unit = "tokens/proposition ratio"`, `per_pair`, `n_undefined`, `source_tokens`, `pivot_tokens`, `ci_low`, `ci_high`, `n_bootstrap`, `seed`. Bootstrap: `rng = numpy.random.default_rng(seed)`; for each draw sample `n` pair indices with replacement, compute ratio of sums from `RatioParts` counts (no re-encoding); `ci_low`/`ci_high` are the `(1-ci)/2` and `1-(1-ci)/2` percentiles; both `nan` if `n_bootstrap == 0` or every draw undefined.
  - `base.py`: `DetailedMetricResult` gains `n_undefined: int`, `ci_low: float`, `ci_high: float`, `n_bootstrap: int`, `seed: int`, `source_tokens: int`, `pivot_tokens: int` (all `total=False`).
  - `parity.py`: reimplemented on `token_ratio`; same public signature; `per_pair` now NaN for undefined pairs; adds `n_undefined`, `source_tokens`, `pivot_tokens`; docstring unchanged in intent.
  - `compression.py`: `per_text` entries `nan` where a text produces zero tokens; adds `n_undefined`. Overall `value` is `nan` (not 0.0) when total tokens is 0 but at least one text was given; still `0.0, n 0` for empty input.

- [ ] **Step 1: Write failing tests for `_ratio.py`** in `tests/test_metrics.py` (append):

```python
import math
from sanskrit_tok.metrics._ratio import RatioParts, token_ratio

def test_ratio_parts_hand_computed() -> None:
    parts = RatioParts(source_counts=(3, 1), pivot_counts=(2, 4))
    assert parts.source_total == 4 and parts.pivot_total == 6
    assert parts.value == 4 / 6
    assert parts.per_pair == [1.5, 0.25]
    assert parts.n_undefined == 0

def test_ratio_parts_marks_undefined_pairs_as_nan() -> None:
    parts = RatioParts(source_counts=(3, 2), pivot_counts=(0, 4))
    assert math.isnan(parts.per_pair[0]) and parts.per_pair[1] == 0.5
    assert parts.n_undefined == 1
    assert parts.value == 5 / 4  # pooled ratio still defined

def test_token_ratio_uses_pivot_tokenizer_and_checks_lengths() -> None:
    parts = token_ratio(CharTokenizer(), ["abc"], ["ab"], pivot_tokenizer=WordTokenizer())
    assert parts.source_counts == (3,) and parts.pivot_counts == (1,)
    with pytest.raises(ValueError):
        token_ratio(CharTokenizer(), ["a", "b"], ["a"])
```

- [ ] **Step 2: Run** `uv run pytest tests/test_metrics.py -q` → FAIL (`ModuleNotFoundError: sanskrit_tok.metrics._ratio`).

- [ ] **Step 3: Implement `_ratio.py`** exactly per Interfaces. Use `math.nan`. `per_pair` is a `list[float]`.

- [ ] **Step 4: Re-implement `parity.py` on `token_ratio`** and update its tests: the existing hand-computed parity cases stay; change the undefined-pivot test to assert `math.isnan(per_pair[i])` and `n_undefined == 1`; the overall `value` when the whole pivot side is empty is `nan` and `n` stays the pair count. Update `compression.py`: `per_text` NaN for zero-token texts, `n_undefined`; update its tests likewise.

- [ ] **Step 5: Write failing tests for `summary.py`** in new `tests/test_summary.py`:

```python
import math
from sanskrit_tok.metrics.summary import summarise_metric, summarise_values

def test_summarise_values_is_nan_aware() -> None:
    s = summarise_values([1.0, math.nan, 3.0])
    assert s == {"mean": 2.0, "std": 1.0, "n": 3, "n_undefined": 1}

def test_summarise_values_all_undefined_gives_none() -> None:
    assert summarise_values([math.nan]) == {"mean": None, "std": None, "n": 1, "n_undefined": 1}
    assert summarise_values([]) == {"mean": None, "std": None, "n": 0, "n_undefined": 0}

def test_summarise_metric_replaces_distribution_with_summary() -> None:
    out = summarise_metric({"value": 2.5, "n": 2, "unit": "tokens/word", "per_word": [2, 3]})
    assert out["distribution"] == "per_word" and out["mean"] == 2.5 and out["std"] == 0.5
    assert "per_word" not in out

def test_summarise_metric_without_distribution_uses_none() -> None:
    out = summarise_metric({"value": 1.0, "n": 1, "unit": "x"})
    assert out["distribution"] is None and out["mean"] is None and out["std"] is None
```

- [ ] **Step 6: Run** `uv run pytest tests/test_summary.py -q` → FAIL. Implement `summary.py`. Move the existing `summarise_metric` tests out of `tests/test_exp01.py` (delete them there), make `run.py` import `summarise_metric` from `sanskrit_tok.metrics.summary`, delete the local copy and its TypedDict. Run `uv run pytest -q` → PASS.

- [ ] **Step 7: Write failing tests for `tpp`** in new `tests/test_tpp.py` (reuse `CharTokenizer`/`WordTokenizer` by importing them from `tests/test_metrics.py` or duplicating the two tiny classes — duplicating is fine):

```python
import math
from sanskrit_tok.metrics.tpp import tpp

def test_tpp_hand_computed_value_and_keys() -> None:
    r = tpp(CharTokenizer(), ["abc", "a"], ["ab", "abcd"], n_bootstrap=200, seed=0)
    assert r["value"] == 4 / 6 and r["n"] == 2 and r["unit"] == "tokens/proposition ratio"
    assert r["per_pair"] == [1.5, 0.25] and r["n_undefined"] == 0
    assert r["source_tokens"] == 4 and r["pivot_tokens"] == 6
    assert r["ci_low"] <= r["value"] <= r["ci_high"]
    assert r["n_bootstrap"] == 200 and r["seed"] == 0

def test_tpp_identical_sides_has_degenerate_ci() -> None:
    r = tpp(CharTokenizer(), ["ab", "cde"], ["ab", "cde"], n_bootstrap=50, seed=1)
    assert r["value"] == 1.0 and r["ci_low"] == 1.0 and r["ci_high"] == 1.0

def test_tpp_is_deterministic_for_a_seed() -> None:
    a = tpp(CharTokenizer(), ["abc", "a", "abcd"], ["ab", "abcd", "a"], n_bootstrap=100, seed=7)
    b = tpp(CharTokenizer(), ["abc", "a", "abcd"], ["ab", "abcd", "a"], n_bootstrap=100, seed=7)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])

def test_tpp_zero_bootstrap_gives_nan_ci() -> None:
    r = tpp(CharTokenizer(), ["abc"], ["ab"], n_bootstrap=0)
    assert math.isnan(r["ci_low"]) and math.isnan(r["ci_high"])

def test_tpp_accepts_different_pivot_tokenizer() -> None:
    r = tpp(WordTokenizer(), ["ab cde fg"], ["abcd"], pivot_tokenizer=CharTokenizer(), n_bootstrap=0)
    assert r["value"] == 0.75
```

- [ ] **Step 8: Run** → FAIL. Implement `tpp.py` per Interfaces (docstring: cite CLAUDE.md §7 and note it is the headline metric; explain ratio-of-sums bootstrap). Run `uv run pytest -q` → PASS.

- [ ] **Step 9: Re-run Experiment 01** (`uv run python experiments/01_baseline_penalty/run.py`) and confirm `metrics` and `parity` values in `outputs/01_baseline_penalty/results.json` are unchanged (FLORES has no undefined items); note that `mean`/`std` for compression are unchanged too. Paste the comparison in the report.

- [ ] **Step 10: Lint/type/test** all clean. **Commit:** `metric: shared ratio core, NaN sentinels, summary module and tpp with bootstrap`.

---

### Task 2: Registry — `T0_gpt2`, T3 Indic arms, file-backed T1/T2 arms, `TokenizerUnavailable`

**Files:**
- Modify: `src/sanskrit_tok/tokenizers/registry.py`, `tests/test_registry.py`, `experiments/01_baseline_penalty/config.yaml` (add `T0_gpt2`), `experiments/01_baseline_penalty/README.md` (add the GPT-2 row/columns after re-run), `docs/decisions.md` (append)

**Interfaces:**
- Consumes: `LoadedTokenizer`, existing adapters and candidate-loop pattern in `registry.py`.
- Produces:
  - `class TokenizerUnavailable(RuntimeError)` — raised by `load_tokenizer` when an arm exists in the registry but cannot be loaded (no candidate loads, or a trained-tokenizer file is missing). Experiments catch it, log WARNING, and record the arm under `unavailable_arms`.
  - `LoadedTokenizer` gains `attempted: tuple[str, ...]` (every candidate id tried, winner last) and `family: str` (one of `"T0"`, `"T1"`, `"T2"`, `"T3"`).
  - New arms and candidate lists (first success wins):
    - `T0_gpt2`: `openai-community/gpt2` (HF fast).
    - `T3_sarvam`: `sarvamai/sarvam-1`.
    - `T3_sutra`: `TWO/sutra-mlt256-v2`.
    - `T3_brahmic131k`: `theschoolofai/BrahmicTokenizer-131K`. Inspect the repo files first (`huggingface_hub.list_repo_files`); if `AutoTokenizer` cannot load it, add a `TokenizersAdapter` using `tokenizers.Tokenizer.from_pretrained(id)` (`encode(text).ids`, `add_special_tokens=False`), and if that fails too, try any `tiktoken`-style `.tiktoken` file with `tiktoken.Encoding` built from its mergeable ranks. Whatever works, `source_id` records the id and the adapter kind.
    - `T3_indicsuper`: candidates `krutrim-ai-labs/IndicSuperTokenizer`, `ai4bharat/IndicSuperTokenizer`, `ai4bharat/indic-super-tokenizer`. If none loads, `load_tokenizer("T3_indicsuper")` raises `TokenizerUnavailable` listing the attempted ids; record in `docs/decisions.md` ("T3_indicsuper unavailable: not publicly released as of 2026-09-03").
    - `T1_bpe_raw_32k`, `T1_bpe_raw_64k`, `T2_unigram_raw_32k`, `T2_unigram_raw_64k`: load `tokenizers.Tokenizer.from_file(TOKENIZER_DIR / name / "tokenizer.json")` where `TOKENIZER_DIR = Path(os.environ.get("SANSKRIT_TOK_TOKENIZER_DIR", "outputs/tokenizers"))` resolved against the repo root (`Path(__file__).resolve().parents[3]`); `source_id` = the file path as a string; `vocab_size = tokenizer.get_vocab_size()`; `encode(text) -> tokenizer.encode(text, add_special_tokens=False).ids`; raise `TokenizerUnavailable` with the expected path if missing. Expose `trained_tokenizer_path(name) -> Path` for Task 4 to write to.
  - `list_tokenizers()` returns all names sorted; add `list_tokenizers(family="T3")` filter.
- Every HF arm: `add_special_tokens=False`; loading each candidate inside `except OSError` only (already the pattern).

- [ ] **Step 1: Failing tests** (append to `tests/test_registry.py`): `TokenizerUnavailable` raised for a trained arm when `SANSKRIT_TOK_TOKENIZER_DIR` points at an empty `tmp_path` and the message contains the expected path; a trained arm loads from a `tokenizer.json` you build in the test with `tokenizers` (`models.BPE`, trainer on ten short SLP1 lines, vocab 50) and returns `family == "T1"`, `vocab_size == 50`, ints from `encode`; `list_tokenizers()` contains all 13 names sorted and `list_tokenizers(family="T3")` returns exactly the four T3 names; candidate-loop `attempted` records the failed ids then the winner (with the existing fake-loader pattern); `T0_gpt2` has `family == "T0"`. Network-gated: `T0_gpt2` vocab 50257; each T3 arm either loads with `vocab_size > 30000` or raises `TokenizerUnavailable`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** Run all three T3 loaders and `T0_gpt2` once for real; paste `source_id`, adapter kind, `vocab_size`, and token counts for `संस्कृतम्` / `Sanskrit` / the SLP1 string `saMskftam` per arm in the report. Append decisions entries for any substitution and for `T3_indicsuper` if unavailable.
- [ ] **Step 5:** Add `T0_gpt2` to `experiments/01_baseline_penalty/config.yaml`; re-run Experiment 01; update the README tables with the GPT-2 numbers (fertility per language, parity Sa/En, Sa/Hi, Sa(SLP1)/En) and one sentence on whether the 50k-vocab arm changes the fertility > 5 verdict. Keep parity leading the summary.
- [ ] **Step 6:** Lint/type/test clean. **Commit:** `tok: add T0_gpt2, T3 Indic arms and file-backed T1/T2 arms with TokenizerUnavailable`.

---

### Task 3: Sāmayik and Itihāsa loaders, shared `parallel.py`, exclusion hashes

**Files:**
- Create: `src/sanskrit_tok/data/parallel.py`, `src/sanskrit_tok/data/samayik.py`, `src/sanskrit_tok/data/itihasa.py`, `src/sanskrit_tok/data/exclusion.py`, `tests/test_parallel_loaders.py`, `tests/test_exclusion.py`, `tests/fixtures/samayik_mini/{test.sa,test.en}`, `tests/fixtures/itihasa_mini/{test.sn,test.en}`
- Modify: `src/sanskrit_tok/data/flores.py` (import `ParallelCorpus`, `save_jsonl`, `load_jsonl` from `parallel.py` and re-export them so existing imports keep working), `data/README.md` (Sāmayik and Itihāsa rows), `data/exclusion_hashes.txt` (generated, committed), `docs/decisions.md`

**Interfaces:**
- Consumes: `ParallelCorpus` (moved), `to_slp1`.
- Produces:
  - `parallel.py`: `ParallelCorpus`, `save_jsonl`, `load_jsonl` (moved verbatim from `flores.py`), plus `def read_aligned_files(paths: Mapping[str, Path], *, name: str, split: str) -> tuple[ParallelCorpus, int]` that reads one plain-text file per language (one sentence per line, `\n` only), strips each line, drops any index where any language is empty after strip, and returns the corpus plus the number of dropped indices. Raises `ValueError` if line counts differ before dropping.
  - `samayik.py`: `SAMAYIK_REPO = "ayushbits/Saamayik"`, `SAMAYIK_COMMIT` (pin the current `main` SHA you observe via `gh api repos/ayushbits/Saamayik/commits/main --jq .sha` and record it), `load_samayik(split: Literal["train","dev","test","test_ood"], cache_dir: Path | None = None) -> ParallelCorpus` with languages `("san_Deva", "eng_Latn")`. Files: `data/final_data/{split}.sa` and `.en` for train/dev/test; for `test_ood` inspect `data/mkb/` in the repo and use its Sanskrit/English files (record the exact names). Download via `https://raw.githubusercontent.com/ayushbits/Saamayik/<SAMAYIK_COMMIT>/<path>` with `urlopen(timeout=120)`, write atomically, cache under `cache_dir` (default `data/raw/samayik/`), and save `cache_dir/{split}.jsonl`; reuse the jsonl if present. Log the dropped-pair count.
  - `itihasa.py`: same shape, `ITIHASA_REPO = "rahular/itihasa"`, `ITIHASA_COMMIT` pinned, files `data/{split}.sn` / `.en` for train/dev/test, languages `("san_Deva", "eng_Latn")`, cache `data/raw/itihasa/`.
  - `exclusion.py`: `EXCLUSION_PATH = Path("data/exclusion_hashes.txt")` (resolved against repo root at call time, not import time); `sentence_hash(text: str) -> str` = `hashlib.sha256(to_slp1(text.strip(), "devanagari").encode("utf-8")).hexdigest()`; `build_exclusion_list(sources: Mapping[str, Sequence[str]], path: Path) -> int` writes two header comment lines (`# sha256 of the SLP1 form of every evaluation sentence, one per line` and `# sources: <name>=<count>, ...`) then sorted unique hashes, returns the number of hashes; `load_exclusion_hashes(path: Path) -> frozenset[str]` (skips `#` lines); `class LeakageError(RuntimeError)`; `assert_not_excluded(texts: Sequence[str], hashes: frozenset[str], *, label: str) -> None` raising `LeakageError` naming up to 5 offending indices and `label`.

- [ ] **Step 1: Failing tests.** `tests/test_parallel_loaders.py`: `read_aligned_files` on the two mini fixtures (write 4 lines each yourself, one pair with an empty Sanskrit line) returns 3 pairs and `dropped == 1`; mismatched line counts raise `ValueError`; `save_jsonl`/`load_jsonl` still importable from `sanskrit_tok.data.flores` (regression). `tests/test_exclusion.py`: `sentence_hash` is stable and script-normalised (`sentence_hash("रामः") == sentence_hash(" रामः ")`); `build_exclusion_list` writes headers + sorted unique hashes and returns the count; `load_exclusion_hashes` round-trips; `assert_not_excluded` raises `LeakageError` naming the index for an excluded sentence and passes for a clean list. Network-gated: `load_samayik("test")` has 2417 pairs (± the dropped count), `load_itihasa("test")` ~11,722 pairs.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Download all splits for real: Sāmayik train/dev/test/test_ood and Itihāsa train/dev/test into `data/raw/` (gitignored); report pair counts and dropped counts per split.
- [ ] **Step 5:** Build the exclusion list from the Sanskrit side of: FLORES devtest (`data/raw/flores/devtest.jsonl`), Sāmayik dev, test, test_ood, Itihāsa dev, test. Write `data/exclusion_hashes.txt` with `build_exclusion_list` (a small script under `experiments/02_tpp_parallel/build_exclusion.py` with `main()`, config-free, is fine) and commit the file. Report the total count and file size.
- [ ] **Step 6:** Update `data/README.md` rows (download date, pinned commit, licence from each repo's LICENSE file — record "unspecified" if there is none, filtering = "dropped N empty pairs"). Append decisions entries for anything unexpected (e.g. duplicate lines in Sāmayik are kept, as the authors intend).
- [ ] **Step 7:** Lint/type/test clean. **Commit:** `data: add Sāmayik and Itihāsa loaders, shared parallel corpus module, exclusion hashes`.

---

### Task 4: Train provisional T1/T2 tokenizers on SLP1

**Files:**
- Create: `src/sanskrit_tok/tokenizers/corpus.py`, `src/sanskrit_tok/tokenizers/train_bpe.py`, `src/sanskrit_tok/tokenizers/train_unigram.py`, `experiments/02_tpp_parallel/train_tokenizers.py`, `experiments/02_tpp_parallel/tokenizers.yaml`, `tests/test_train_tokenizers.py`, `tests/fixtures/slp1_corpus_mini.txt` (200 lines of SLP1 you generate from the Sāmayik mini fixture plus made-up Sanskrit-like lines — no real evaluation sentences)
- Modify: `docs/decisions.md`

**Interfaces:**
- Consumes: `load_samayik("train")`, `load_itihasa("train")`, `to_slp1`, `load_exclusion_hashes`, `assert_not_excluded`, `trained_tokenizer_path`.
- Produces:
  - `corpus.py`: `build_training_corpus(sources: Mapping[str, Sequence[str]], out_path: Path, exclusion: frozenset[str]) -> dict[str, object]` — for each source, asserts `assert_not_excluded(texts, exclusion, label=name)` on the Devanagari text, converts to SLP1, strips, drops empty, exact-dedups across all sources preserving first occurrence, writes one line per sentence to `out_path`, and returns a manifest `{"n_in": {name: int}, "n_out": int, "n_dedup_removed": int, "exclusion_hashes": len(exclusion), "sha256": <of the file>}`.
  - `train_bpe.py`: `train_bpe(corpus_path: Path, vocab_size: int, out_dir: Path, *, seed: int = 0) -> Path` — HF `tokenizers`: `models.BPE(unk_token="[UNK]")`, `pre_tokenizers.Metaspace()`, `decoders.Metaspace()`, `trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"], show_progress=False)`; saves `out_dir/tokenizer.json` and returns its path. (`tokenizers` training is deterministic given the corpus; `seed` is recorded in the manifest for the protocol.)
  - `train_unigram.py`: same signature; `models.Unigram()`, `trainers.UnigramTrainer(vocab_size=vocab_size, unk_token="[UNK]", special_tokens=["[UNK]"], show_progress=False)`, same pre-tokenizer/decoder.
  - `tokenizers.yaml`: `corpus_path: data/processed/tok_train_slp1.txt`, `sources: [samayik_train, itihasa_train]`, `exclusion_path: data/exclusion_hashes.txt`, `arms: [{name: T1_bpe_raw_32k, algo: bpe, vocab_size: 32000}, {name: T1_bpe_raw_64k, algo: bpe, vocab_size: 64000}, {name: T2_unigram_raw_32k, algo: unigram, vocab_size: 32000}, {name: T2_unigram_raw_64k, algo: unigram, vocab_size: 64000}]`, `seed: 0`, `output_dir: outputs/tokenizers`.
  - `train_tokenizers.py`: `--config` (default sibling `tokenizers.yaml`); builds the corpus (skips if `corpus_path` exists and its sha256 matches `manifest.json` next to it), trains every arm into `output_dir/<name>/tokenizer.json`, and writes `output_dir/<name>/results.json` (`git_commit`, `timestamp`, config, manifest, `vocab_size` actual, `train_seconds`) and `output_dir/<name>/config.yaml`. Logic in functions, `main()` under `__main__`.

- [ ] **Step 1: Failing tests** in `tests/test_train_tokenizers.py`: `build_training_corpus` dedups and raises `LeakageError` when a source contains an excluded sentence; `train_bpe` and `train_unigram` on the 200-line fixture with `vocab_size=300` write `tokenizer.json`, the trained tokenizer loads through `load_tokenizer("T1_bpe_raw_32k")` when `SANSKRIT_TOK_TOKENIZER_DIR` is monkeypatched to the tmp dir (copy the file into `<tmp>/T1_bpe_raw_32k/tokenizer.json`), `encode` returns ints, no token string other than `[UNK]` contains a space, and every non-initial token lacks the `▁` prefix (Metaspace boundary marker only at word starts).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run `uv run python experiments/02_tpp_parallel/train_tokenizers.py` for real; report corpus manifest numbers, per-arm `train_seconds`, actual vocab sizes, and the token counts for `saMskftam` and a full SLP1 sentence from Sāmayik test under each arm. Append a decisions entry with the actual corpus size and any deviation.
- [ ] **Step 5:** Lint/type/test clean. **Commit:** `tok: train provisional T1/T2 BPE and Unigram tokenizers on SLP1 with leakage assertion`.

---

### Task 5: Experiment 02 runner, results, central figure, README

**Files:**
- Create: `experiments/02_tpp_parallel/run.py`, `experiments/02_tpp_parallel/config.yaml`, `tests/test_exp02.py`
- Modify: `experiments/02_tpp_parallel/README.md` (replace placeholder), `docs/decisions.md` if deviating

**Interfaces:**
- Consumes: `tpp`, `fertility`, `compression`, `summarise_metric`, `load_tokenizer`, `TokenizerUnavailable`, `list_tokenizers`, `load_samayik`, `load_itihasa`, `load_jsonl` (FLORES), `to_slp1`, `load_exclusion_hashes`, `sentence_hash`.
- Produces: `outputs/02_tpp_parallel/{results.json,config.yaml,tpp_by_arm.pdf,tpp_by_arm.png}`.

`config.yaml`:
```yaml
corpora:               # prose before verse; order is binding for tables and figure panels
  - {name: samayik_test, loader: samayik, split: test}
  - {name: samayik_test_ood, loader: samayik, split: test_ood}
  - {name: itihasa_test, loader: itihasa, split: test}
  - {name: flores_devtest, loader: flores, split: devtest, jsonl: data/raw/flores/devtest.jsonl}
sanskrit_arms: [T0_o200k, T0_llama4, T0_gemma3, T0_gpt2, T3_sarvam, T3_sutra, T3_indicsuper, T3_brahmic131k, T1_bpe_raw_32k, T1_bpe_raw_64k, T2_unigram_raw_32k, T2_unigram_raw_64k]
english_pivots: [T0_o200k, T0_llama4]      # o200k is primary
hindi_pivot_corpus: flores_devtest         # Sa/Hi under the same tokenizer, T0/T3 arms only
script_variants: {T0: [original, slp1], T3: [original, slp1], T1: [slp1], T2: [slp1]}
n_bootstrap: 1000
seed: 0
exclusion_path: data/exclusion_hashes.txt
output_dir: outputs/02_tpp_parallel
```

`run.py` (functions, `main()` under `__main__`; follow `experiments/01_baseline_penalty/run.py` conventions for path resolution, git commit, logging, config copy):
1. Load config; load each corpus; filter blank-aligned indices; record `n_used`.
2. Leakage sanity: every Sanskrit evaluation sentence's `sentence_hash` must be in the exclusion set; record `exclusion_check: {corpus: {"n": ..., "n_missing": ...}}` and log WARNING if any missing (do not abort).
3. Load arms via `load_tokenizer`; on `TokenizerUnavailable` log WARNING and add to `unavailable_arms` with the message.
4. For each corpus × Sanskrit arm × script variant (per `family`) × English pivot: `tpp(arm, sanskrit_texts, english_texts, pivot_tokenizer=pivot, n_bootstrap, seed)`; store `summarise_metric` output (value, n, ci_low, ci_high, n_undefined, mean, std, source_tokens, pivot_tokens).
5. For `hindi_pivot_corpus` and each T0/T3 arm × variant: `tpp(arm, sanskrit, hindi)` with the same tokenizer both sides (Hindi transliterated with the same `to_slp1` for the `slp1` variant, labelled approximate as in Exp01).
6. Fertility and compression of the Sanskrit side per corpus × arm × variant (summarised).
7. `results.json` keys: `experiment`, `git_commit`, `timestamp`, `config`, `tokenizer_sources` (`arm → {source_id, vocab_size, family, attempted}`), `unavailable_arms`, `corpora` (`name → {split, n_total, n_used}`), `exclusion_check`, `tpp` (`corpus → arm → variant → pivot → summary`), `tpp_hindi` (`arm → variant → summary`), `fertility`, `compression`.
8. `make_figure(results, out_dir)`: four panels stacked vertically in config corpus order; x = arms in config order (unavailable arms omitted, noted in the caption text below the panel), y = TPP Sanskrit (slp1 variant) vs English o200k, point with CI error bars; dashed horizontal line at 1.0; x tick labels include vocab size in thousands (e.g. `T1_bpe_raw_32k (32k)`, `T0_o200k (200k)`); suptitle `Tokens per proposition: Sanskrit vs English (o200k), 95% bootstrap CI`; the T1/T2 tick labels carry a `*` and the caption says `* provisional: trained on parallel-corpus training splits`. Save PDF and PNG.
9. README: hypothesis H2 verbatim from the outline §1; success = at least one Sanskrit-native arm below 1.0 on Sāmayik with CI excluding 1.0, and T0 arms above 1.0; expected runtime; then a results table per corpus (rows = arms with vocab size, columns = TPP vs o200k with CI, TPP vs Llama-4, fertility) in config order, prose first; a summary paragraph that leads with TPP and states plainly whether the sign flip occurred; the existing-practice caveat; the provisional-T1/T2 caveat; the Hindi-SLP1 approximation caveat.

- [ ] **Step 1: Failing tests** in `tests/test_exp02.py`: `make_figure` on a synthetic results dict with two corpora and three arms (one flagged provisional, one unavailable) writes both files; the arm-label helper renders `T1_bpe_raw_32k` as `T1_bpe_raw_32k* (32k)` and `T0_o200k` as `T0_o200k (200k)`; the exclusion-check helper counts missing hashes correctly on three sentences with two in the set; the per-family variant selection returns `["slp1"]` for `T1` and `["original", "slp1"]` for `T0`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `run.py`. **Step 4:** Run the experiment for real; paste the full TPP tables (value, CI) for all corpora and the Hindi table in the report, plus runtime. **Step 5:** Fill the README. **Step 6:** Lint/type/test clean. **Commit:** `exp02: tokens-per-proposition runner, results and central figure`.

---

### Task 6: Matched English control arms E1 and controlled TPP

**Files:**
- Modify: `src/sanskrit_tok/tokenizers/registry.py` (add `E1_bpe_32k`, `E1_bpe_64k`, `E1_unigram_32k`, `E1_unigram_64k` as file-backed arms via the existing trained-arm loader; `family == "E1"`), `experiments/02_tpp_parallel/tokenizers.yaml` (add an `english_corpus_path: data/processed/tok_train_en.txt`, `english_sources: [samayik_train_en, itihasa_train_en]`, and the four E1 arms with `side: en`; existing arms get `side: sa`), `experiments/02_tpp_parallel/train_tokenizers.py` (build the English corpus from `ParallelCorpus.sentences["eng_Latn"]` of the same training splits — no SLP1 conversion, `str.strip()` only, exact dedup, and assert against an English exclusion set built the same way: sha256 of the stripped English eval sentences from the same six evaluation splits, written to `data/exclusion_hashes_en.txt` and committed), `src/sanskrit_tok/data/exclusion.py` (add `sentence_hash_en(text) -> str` = sha256 of `text.strip()`; `build_exclusion_list` takes a `hash_fn` parameter defaulting to `sentence_hash`), `experiments/02_tpp_parallel/build_exclusion.py` (also writes the English list), `experiments/02_tpp_parallel/config.yaml` (`english_pivots: [E1_bpe_32k, E1_bpe_64k, E1_unigram_32k, E1_unigram_64k, T0_o200k, T0_llama4]`; add `controlled_pairs: [[T1_bpe_raw_32k, E1_bpe_32k], [T1_bpe_raw_64k, E1_bpe_64k], [T2_unigram_raw_32k, E1_unigram_32k], [T2_unigram_raw_64k, E1_unigram_64k]]`; `figure_pivot_controlled: true`), `experiments/02_tpp_parallel/run.py` (compute TPP for every Sanskrit arm × every English pivot as now; add a `tpp_controlled` block keyed by pair name `T1_bpe_raw_32k/E1_bpe_32k` etc.; the figure gains a second column of panels showing the controlled TPP per pair with CI and the 1.0 line, prose first; `tokenizer_sources` gains `sha256` for file-backed arms; every results writer records `git_dirty`), `experiments/02_tpp_parallel/README.md` (controlled TPP table becomes the first table; deployed-practice tables follow; verdict rewritten per the decisions entry "Experiment 02 verdict reframed"), `CLAUDE.md` §6 (add row `E1_bpe_{32k,64k}`, `E1_unigram_{32k,64k}` | Matched English control trained on the English side of the same corpus), `tests/test_registry.py`, `tests/test_train_tokenizers.py`, `tests/test_exclusion.py`, `tests/test_exp02.py`.

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: E1 arms in the registry; `data/exclusion_hashes_en.txt`; `results.json` keys `tpp_controlled` (`pair → variant → summary`) and `git_dirty`; `tokenizer_sources[arm]["sha256"]` for file-backed arms.

- [ ] Step 1: failing tests — registry lists 16 arms and `list_tokenizers(family="E1")` has four; `sentence_hash_en` is whitespace-insensitive and differs from `sentence_hash`; `build_exclusion_list(..., hash_fn=sentence_hash_en)` works; the English corpus builder raises `LeakageError` on an excluded English sentence; `train_tokenizers` config parsing handles `side`; run.py helper that pairs arms reads `controlled_pairs` and skips a pair when either side is unavailable; figure test asserts the controlled column exists.
- [ ] Step 2: implement; build the English exclusion list (commit it); train the four E1 arms for real (report corpus manifest and vocab sizes); re-run the experiment; fill the README with the controlled table first; update CLAUDE.md §6.
- [ ] Step 3: lint/type/test clean. Commit `exp02: matched English control arms E1 and controlled TPP`.
