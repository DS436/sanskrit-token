# CLAUDE.md — Sanskrit Tokenization Research Project

This file gives a coding agent the context needed to work on this project without re-deriving decisions. Read it fully before touching code. The full research design is in `docs/paper_outline.md` (copy of the outline delivered separately). When this file and the outline disagree, this file wins for engineering decisions and the outline wins for research decisions.

---

## 0. Delegation policy

The top-level session orchestrates; it does not implement. All coding, tests, scripts, configs, lint/type fixes, and boilerplate docs go to an Opus subagent (`Agent` tool with `model: "opus"`). Load and follow the project skill `delegating-to-opus` (`.claude/skills/delegating-to-opus/SKILL.md`) before any implementation work. The orchestrator keeps research-design decisions, review, and final verification.

---

## 1. What this project is

We are testing whether Sanskrit's morphological density (case, number, person, tense fused into endings; clause-length compounds) represents **latent token efficiency that current subword tokenizers destroy**, and whether a tokenizer that (a) reverses sandhi before subword learning and (b) forbids BPE merges across gold morpheme boundaries recovers it.

Output is a research paper targeting ACL/EMNLP. Every experiment must be reproducible from this repo with one command.

### The two claims — keep them separate in code and prose
- **Claim A (true, cite, don't prove):** Sanskrit is information-dense per *word*.
- **Claim B (the thing we test):** whether that density survives tokenization, measured per *unit of meaning*, not per word.

Never write code, plots, or docs that conflate "fewer words" with "fewer tokens."

---

## 2. Non-negotiable constraints

1. **Fertility (tokens/word) is never the headline metric.** Sanskrit words are long because of sandhi and compounding; fertility punishes that. Always compute it, always report it, never lead with it. The headline is **tokens-per-proposition (TPP)** on parallel text and **bits-per-character (BPC)** from LM training.
2. **Perplexity is never compared across tokenizers.** Different vocabularies make perplexity incomparable. Use BPC or bits-per-byte.
3. **Internal encoding is SLP1.** All Sanskrit text is converted to SLP1 on ingest and back to Devanagari/IAST only for display. Use `indic_transliteration` (`sanscript`). Store the original script alongside.
4. **No evaluation leakage.** SIGHUM, Hackathon, DCS-2018 test sets are all subsets of DCS. Any tokenizer or LM training set must exclude every evaluation sentence. Enforce with a hash-based exclusion list in `data/exclusion_hashes.txt`; every training script must assert against it.
5. **Matched vocabulary sizes** for any trained-from-scratch tokenizer comparison (32k and 64k). Off-the-shelf tokenizers (Llama-4, Sarvam, etc.) are reported as "existing practice," never as controlled comparisons.
6. **Three seeds minimum** for anything involving model training. Report mean ± std.
7. **Prose before verse.** Sāmayik (prose) is the primary parallel corpus for TPP; Itihāsa (verse) is secondary. Meter is a confound.
8. **Do not add the "NASA chose Sanskrit" claim** to any doc, README, or comment. Briggs 1985 (AI Magazine) is the only acceptable historical citation.
9. **Every experiment writes a `results.json` and a `config.yaml`** to its output dir. No numbers live only in notebooks or stdout.

---

## 3. Repository layout

```
sanskrit-tok/
├── CLAUDE.md                  # this file
├── README.md
├── pyproject.toml
├── docs/
│   ├── paper_outline.md
│   └── decisions.md           # append-only log of design decisions with dates
├── data/
│   ├── raw/                   # downloaded, untouched
│   ├── processed/             # SLP1-converted, split, deduped
│   ├── exclusion_hashes.txt   # sha256 of every eval sentence (SLP1 form)
│   └── README.md              # provenance + license for every source
├── src/sanskrit_tok/
│   ├── encoding.py            # Devanagari/IAST <-> SLP1, roundtrip-tested
│   ├── data/                  # loaders, one module per corpus
│   ├── sandhi/                # ByT5-Sanskrit wrapper, TransLIST wrapper
│   ├── tokenizers/
│   │   ├── train_bpe.py
│   │   ├── train_unigram.py
│   │   ├── morph_bpe.py       # merge-constrained BPE (adapted from MorphBPE)
│   │   └── registry.py        # name -> loader for every arm T0..T7
│   ├── metrics/
│   │   ├── fertility.py
│   │   ├── compression.py
│   │   ├── renyi.py
│   │   ├── morphscore.py
│   │   ├── parity.py
│   │   └── tpp.py             # tokens-per-proposition
│   ├── lm/                    # nanoGPT-style training, BPC eval
│   └── eval/                  # downstream: segmentation, DharmaBench, IndicParam
├── experiments/
│   ├── 01_baseline_penalty/   # RQ1: T0 fertility + parity on FLORES
│   ├── 02_tpp_parallel/       # RQ2: TPP on Sāmayik/Itihāsa/FLORES
│   ├── 03_sandhi_split/       # RQ3: T4 arms
│   ├── 04_morph_constrained/  # RQ4: T5, T6 arms
│   └── 05_lm_training/        # BPC, tokens-to-reference-loss
├── tests/
└── outputs/                   # gitignored; results.json + figures per experiment
```

Each `experiments/NN_name/` has `run.py`, `config.yaml`, and a `README.md` stating the hypothesis, what "success" looks like, and the expected runtime.

---

## 4. Environment

- Python 3.11+, managed with `uv` (`uv sync`). Pin everything in `pyproject.toml`.
- Core deps: `tokenizers`, `transformers`, `sentencepiece`, `indic-transliteration`, `datasets`, `numpy`, `pandas`, `pyyaml`, `pytest`, `matplotlib`.
- LM training: `torch`; nanoGPT vendored into `src/sanskrit_tok/lm/` with attribution.
- GPU is optional for experiments 01–04; required for 05.
- Never `pip install` ad hoc; add to `pyproject.toml` and re-sync.

---

## 5. Data sources

| Name | Where | Format | Status |
|------|-------|--------|--------|
| Digital Corpus of Sanskrit (DCS) | https://github.com/OliverHellwig/sanskrit (conllu exports) | CoNLL-U with lemma + morph | **Primary gold morpheme source.** Segmented forms partly machine-generated; track which subsets are human-verified. |
| SIGHUM | Krishna et al. 2017 release | SLP1 | Eval only. DCS subset. |
| Hackathon | Krishnan et al. 2020 release | SLP1 | Eval only. DCS subset. |
| DCS-2018 | Hellwig & Nehrdich 2018 release | SLP1 | Eval only. |
| UoH corpus + SandhiKosh | https://sanskrit.uohyd.ac.in/Corpus/ | Devanagari | Apply Dave et al. 2021 pruning. |
| Itihāsa | https://github.com/rahular/itihasa | Sa–En verse, 93k | Secondary parallel (verse). |
| Sāmayik | arXiv 2305.14004 release | En–Sa prose, 53k | **Primary parallel.** |
| SAHAAYAK 2023 | arXiv 2307.00021 release | Sa–Hi, 1.5M | Hindi pivot. |
| FLORES-200 | HF `facebook/flores` | devtest; use `san_Deva`, `hin_Deva`, `eng_Latn` | **Parity anchor.** |
| IN22-Gen | AI4Bharat | 22 Indic + English | Secondary parity. |
| ByT5-Sanskrit | HF `chronbmm/byt5-sanskrit` (verify exact id) | model | Sandhi-splitting oracle. |
| DharmaBench, IndicParam | per-paper releases | eval | Downstream only. |

`data/README.md` must record download date, commit/version, license, and any filtering applied, for each source. Do not commit raw data.

---

## 6. Tokenizer arms (must match `registry.py` names exactly)

| Key | Description |
|-----|-------------|
| `T0_llama4`, `T0_gemma3`, `T0_o200k` | Off-the-shelf English-centric |
| `T1_bpe_raw_{32k,64k}` | BPE on raw sandhied Sanskrit |
| `T2_unigram_raw_{32k,64k}` | Unigram on raw Sanskrit |
| `T3_sarvam`, `T3_sutra`, `T3_indicsuper`, `T3_brahmic131k` | Off-the-shelf Indic |
| `T4_bpe_split_{32k,64k}`, `T4_unigram_split_{32k,64k}` | Sandhi-split then subword |
| `T5_morphbpe_raw_{32k,64k}` | Morpheme-constrained merges on raw |
| `T5_morphbpe_rawseg_{32k,64k}` | Merges constrained on gold **segment** boundaries only, on raw |
| `T6_morphbpe_split_{32k,64k}` | Sandhi-split + morpheme-constrained (**proposed**) |
| `T7_byt5` | Byte-level, no subword |
| `E1_bpe_{32k,64k}`, `E1_unigram_{32k,64k}` | Matched English control trained on the English side of the same corpus |

Ablation suffixes: `_deva` / `_slp1` / `_iast` for script; `_oracle` for gold splits.

---

## 7. Metric contracts

Every metric function takes `(tokenizer, list[str])` and returns a dict with at minimum `{"value": float, "n": int, "unit": str}`. Metric modules must have unit tests against hand-computed examples.

- `fertility.py`: tokens per whitespace-delimited word. Also return per-word distribution.
- `compression.py`: UTF-8 bytes per token, computed on SLP1 *and* on the original script; report both.
- `renyi.py`: Rényi efficiency at α ∈ {2.5, 3}, per Zouhar et al. 2023. Docstring must note it can be gamed (Cognetta et al. 2024).
- `morphscore.py`: precision and recall of token boundaries vs. gold morpheme boundaries, per Arnett & Bergen. Exclude single-token words and single-morpheme words. Input gold boundaries come from DCS.
- `parity.py`: tokens(lang_a) / tokens(lang_b) on aligned FLORES sentences. Default pivot `eng_Latn`; also `hin_Deva`.
- `tpp.py`: tokens-per-proposition. For aligned pairs `(s, e)`: `sum(tokens(s)) / sum(tokens(e))`, plus per-pair distribution and bootstrap CI. Must accept different tokenizers for each side.

---

## 8. Coding conventions

- Type hints everywhere. `ruff` and `mypy --strict` clean.
- Pure functions for metrics; I/O only in `data/` and `experiments/`.
- Config via YAML; no hard-coded paths. Use `pathlib`.
- Log with `logging`, not print. Every experiment logs its git commit hash and config into `results.json`.
- Seeds: set `random`, `numpy`, `torch` from config; record in results.
- Figures: `matplotlib`, saved as both `.pdf` and `.png`, generated by a script, never by hand.
- Tests run in < 60 s on CPU with tiny fixtures in `tests/fixtures/`.
- Commit messages: `exp01: ...`, `metric: ...`, `data: ...`, `tok: ...`, `lm: ...`, `docs: ...`.

---

## 9. Sanskrit glossary for the agent

- **Sandhi** — euphonic combination at word boundaries (`tat + api → tadapi`). Erases whitespace. Types: vowel, visarga, consonant. Rules are deterministic forward, ambiguous backward.
- **Samāsa** — compounding; multiple stems fused into one word with only the final member inflected.
- **Vibhakti** — case endings (8 cases). Encode grammatical role, replacing English prepositions.
- **Dhātu / prātipadika** — verbal root / nominal stem.
- **Śloka** — verse form, 4 × 8 syllables. Meter constrains word choice. Itihāsa is śloka.
- **SLP1** — ASCII transliteration scheme, one char per phoneme, lossless. Preferred by neural Sanskrit models.
- **IAST** — diacritic romanisation (`ā, ṣ, ṃ`). Human-readable; use for display.
- **DCS** — Digital Corpus of Sanskrit, the gold-annotated corpus.

---

## 10. First task — Experiment 01: baseline penalty (RQ1)

**Hypothesis:** English-centric tokenizers produce fertility > 5 on Sanskrit, and the Sanskrit/Hindi parity ratio is > 1.5 on identical FLORES content, because sandhi merges what Hindi keeps separate.

**Steps:**
1. Implement `encoding.py` with roundtrip tests (Devanagari → SLP1 → Devanagari must be identity on FLORES `san_Deva`).
2. Implement `data/flores.py` loading `san_Deva`, `hin_Deva`, `eng_Latn` devtest, aligned by index.
3. Implement `fertility.py`, `compression.py`, `parity.py` with tests.
4. Implement `registry.py` for `T0_o200k` (via `tiktoken`) and `T0_llama4`, `T0_gemma3` (via `transformers`; gated models may need a token — fall back to Llama-3 / Gemma-2 tokenizers and record the substitution in `decisions.md`).
5. `experiments/01_baseline_penalty/run.py`: for each T0 tokenizer × each language × {original script, SLP1}, compute fertility, compression, and parity vs. `eng_Latn` and vs. `hin_Deva`. Write `results.json` and one figure: grouped bars of fertility by language per tokenizer.
6. `README.md` in that dir summarising the numbers in two sentences.

**Done when:** `uv run python experiments/01_baseline_penalty/run.py` produces `outputs/01_baseline_penalty/results.json` and `fertility_by_language.pdf`, tests pass, and `decisions.md` has an entry.

Do not start Experiment 02 until 01 is merged.

---

## 11. Decision log protocol

`docs/decisions.md` is append-only. Format:

```
## 2026-09-03 — Use SLP1 as internal encoding
Why: CharSS 2024 and ByT5-Sanskrit both report SLP1 outperforms Devanagari for neural models; it is lossless and ASCII.
Alternatives: IAST (diacritics complicate tokenizer training), Devanagari (byte-level fragmentation).
Reversible: yes, at cost of re-running tokenizer training.
```

Record every substitution, dataset filter, and hyperparameter deviation from the outline.
