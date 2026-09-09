# sanskrit-tok

**Does Sanskrit's morphological density survive tokenization?**

Sanskrit fuses case, number, person and tense into word endings and chains whole clauses
into single compounds, so it is genuinely information-dense *per word* — that is a settled
claim about the language (call it **Claim A**; we cite it, we do not re-prove it). The
question this repository asks is a different one (**Claim B**): does that density survive
being cut into subword tokens? Fewer words is not the same as fewer tokens. Sandhi erases
the whitespace a subword learner uses as its cheapest boundary signal, so a tokenizer may
be spending its budget re-discovering morphology that was already marked in the language.
We measure the cost **per unit of meaning** on parallel text (tokens-per-proposition) and,
where a model is trained, in **bits per character** — never in perplexity, which is not
comparable across vocabularies, and never leading with fertility (tokens per word), which
punishes Sanskrit for the very compounding under test. Then we ask whether reversing sandhi
before subword learning, and forbidding BPE merges across gold morpheme boundaries,
recovers anything.

Short answer so far: **the naive version of the claim does not survive contact with a
matched control, and the interventions buy less than advertised** — but they buy something,
and exactly which half of the boundary set costs tokens is now measurable.

## Results

Every number below is from the committed snapshot in [`results/`](results/README.md); no
run is needed to read them. Experiments 01 and 02 are also written up as a paper draft in
[`paper/1a/`](paper/1a/README.md), whose every table, figure and prose number is generated
from that same snapshot, and which builds to a PDF with `cd paper/1a && make all` (see
[`paper/1a/README.md`](paper/1a/README.md#build) for the review, final and arXiv targets).

| # | Hypothesis | Verdict | Details | Figure |
|---|---|---|---|---|
| 01 | English-centric tokenizers penalise Sanskrit against English and Hindi | Half refuted: Sa/En parity 1.77–2.19, but Sa/Hi only 1.33–1.35 | [README](experiments/01_baseline_penalty/README.md) | [fertility_by_language](results/01_baseline_penalty/fertility_by_language.png) |
| 02 | A Sanskrit-native tokenizer beats English on tokens-per-proposition | Refuted at 32k and 64k under matched control: TPP 1.030–1.142 on Sāmayik prose; the penalty shrinks with vocabulary and the 128k BPE pair reads 0.983 in domain | [README](experiments/02_tpp_parallel/README.md) | [tpp_by_arm](results/02_tpp_parallel/tpp_by_arm.png) |
| 03 | Splitting sandhi before subword learning lowers tokens-per-proposition | Supported on prose (−0.005 to −0.077), adverse on verse | [README](experiments/03_sandhi_split/README.md) | [tpp_split_vs_raw](results/03_sandhi_split/tpp_split_vs_raw.png) |
| 04 | Morpheme-constrained merges raise MorphScore and lower tokens-per-proposition | MorphScore +0.114; no constrained arm saves tokens, so H4's TPP half fails | [README](experiments/04_morph_constrained/README.md) | [constraint_effects](results/04_morph_constrained/constraint_effects.png) |
| 05 | The constrained tokenizer reaches a reference BPC with fewer tokens | Not yet run: pipeline validated on CPU/MPS, sweep needs a GPU | [README](experiments/05_lm_training/README.md) | [smoke bpc_final](results/05_lm_training/smoke/bpc_final.png) — *not a result* |

A little more detail on each, in the order the argument runs:

1. **Baseline penalty.** On identical FLORES-200 devtest content, Sanskrit costs 1.77–2.19×
   as many tokens as its English translation across three modern ≥200k-vocabulary
   tokenizers — but only 1.33–1.35× as many as its *Hindi* translation. So sandhi does not
   buy Sanskrit a large penalty over another Devanagari language. Fertility on Sanskrit is
   3.11–3.88, not the >5 the hypothesis predicted, except for GPT-2's old 50k vocabulary
   meeting three-byte UTF-8 (12.49), which is a fact about that vocabulary, not about
   Sanskrit.
2. **Tokens per proposition.** Trained Sanskrit BPE/Unigram arms look like they beat English
   (0.887 against `o200k`) until you train the *matched* English control — same algorithm,
   same vocabulary size, same corpus. Then every prose pair at 32k and 64k pieces is above
   parity (1.030–1.142), under a pair-matched control and a byte-matched one alike. The
   apparent Sanskrit advantage was the English pivot's domain handicap. The penalty shrinks
   as the vocabulary grows: at 128k the BPE pair reads 0.983 [0.971, 0.997] on in-domain
   Sāmayik prose while staying above parity out of domain (1.025) and on FLORES (1.116), so
   the negative result is scoped to the sizes it was measured at. Itihāsa verse stays below
   1.0 throughout, and verse has a meter confound.
3. **Sandhi splitting.** Splitting first (ByT5-Sanskrit, reconciled against the raw
   sentence) gives a real but small controlled saving on prose — Δ TPP −0.005 to −0.077,
   CIs clear of zero — and *costs* tokens on verse (+0.006 to +0.018). No arm gets below its
   matched English control on prose. Two review rounds cut the original headline in half
   twice: the first version was measuring deleted punctuation, the second was taking credit
   for hyphens the source already segmented.
4. **Morpheme constraints.** Forbidding merges across DCS's **gold segment** boundaries
   raises boundary F1 by **+0.114 [+0.106, +0.122]** at 64k with a token cost whose CI
   includes zero (+0.002 TPP on Sāmayik test). Adding the *heuristic* stem boundary raises
   the cost by an order of magnitude (+0.084) for a third of the F1. So the constraint's
   price is almost entirely the heuristic half of the boundary set, not the gold half — and
   the arm to carry forward is `T5seg`, not the originally proposed `T6`.
5. **Language-model training.** Prepared, not run. See below.

## What this does not show yet

- **Experiment 05 has not been run.** The corpora, the vendored nanoGPT pipeline, the
  bits-per-character evaluation and the sweep runner all exist and have been validated end
  to end, but the real sweep is 51 runs and needs a rented GPU (~18–26 A100-hours). Until it
  runs, **H4 is untested**: a vocabulary that costs 5% more tokens can still reach a
  reference BPC sooner if its tokens are easier to predict, and nothing here rules that in
  or out. The `results/05_lm_training/smoke/` numbers are a pipeline check on an 8.6 M-parameter
  model for 300 steps and must not be cited.
- **H3's MorphScore half is only half testable.** Whether *sandhi splitting alone* raises
  MorphScore cannot be read off these tables: a split arm's only granularity is scored over
  a different population of units (gold segments) than a raw arm's (surface words), so the
  comparison would be between two corpora rather than between two tokenizers. What is
  tested, and supported, is the *constraint* half.
- **Every trained arm in Experiments 02 and 03 is provisional** (marked `*` in those files):
  trained on the Sanskrit side of two parallel corpora, not on the monolingual corpus.
  Experiment 04's `_dcs` arms are trained on DCS and are the ones Experiment 05 uses.
- **The stem/ending boundary is a heuristic, not gold.** DCS annotates segment boundaries;
  the boundary *inside* a segment is derived from the lemma by a rule, and 36% of its cuts
  are flagged as fused. Every table names it "heuristic stem" and never "gold".
- **Verse is adverse and undiagnosed.** Splitting and constraining both cost tokens on
  Itihāsa. Meter is the leading suspect, per the project's own rule that prose comes first.
- **Two off-the-shelf arms are ungated mirrors.** `T0_llama4` and `T0_gemma3` load
  `unsloth/*` re-uploads because the official repositories are gated; the mirrors cannot be
  byte-compared against the originals, and `results.json` always records the id that
  actually loaded.

## Reproduce

### Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/). Everything else is pinned:
  `uv sync` installs the locked environment. Never `pip install` into it.
- **No Hugging Face token is needed.** Gated tokenizer repositories are substituted with
  ungated `unsloth` mirrors and the substitution is recorded in `docs/decisions.md` and in
  each `results.json`. Setting `HF_TOKEN` switches them back with no code change.
- **Downloads happen automatically**, into gitignored `data/raw/`. Sizes worth knowing
  before you start: the DCS sparse checkout is **~1.3 GB** (Experiments 04 and 05) and
  Sangraha verified Sanskrit is **~4.2 GB** of parquet (Experiment 05 only). FLORES,
  Itihāsa and Sāmayik are small.
- **One step is slow and is not on the critical path for reading results.** Experiment 03's
  sandhi splitting ran the ByT5-Sanskrit model over 136,918 sentences in **9.4 hours on an
  M3 Pro** (MPS, ~4 sentences/s). It is cached, so it is paid once.
- **A GPU is required for Experiment 05 only.** Experiments 01–04 are CPU work.

```bash
git clone https://github.com/DS436/sanskrit-token && cd sanskrit-token
uv sync
uv run pytest -q          # < 60 s, tiny fixtures, no network
```

### One command per experiment, in order

```bash
# 01 — baseline penalty. ~13 s warm; a few minutes cold (FLORES + tokenizer downloads).
uv run python experiments/01_baseline_penalty/run.py

# 02 — tokens per proposition. Train the matched English controls once (~9 s), then run.
uv run python experiments/02_tpp_parallel/train_tokenizers.py
uv run python experiments/02_tpp_parallel/run.py                    # ~2 min

# 03 — sandhi split. The splitting job is the 9.4 h one; the runner itself is 55 s.
uv run python experiments/03_sandhi_split/split_corpora.py          # ~9.4 h, cached
uv run python experiments/03_sandhi_split/run.py                    # ~55 s

# 04 — morpheme-constrained merges. Ingestion downloads the 1.3 GB DCS checkout.
uv run python experiments/04_morph_constrained/ingest_dcs.py        # ~2 min after download
uv run python experiments/04_morph_constrained/train_tokenizers.py
uv run python experiments/04_morph_constrained/run.py               # 3–4 min

# 05 — LM training. Corpus build first (downloads ~4.2 GB of Sangraha), then the sweep.
uv run python experiments/05_lm_training/build_corpus.py --config experiments/05_lm_training/corpus.yaml
uv run python experiments/05_lm_training/run.py --sweep sweep       # GPU, ~18–26 A100-hours
```

Each writes `results.json`, a copy of its `config.yaml` and its figures to
`outputs/<experiment>/`, which is gitignored — the committed copies live in
[`results/`](results/README.md), so a re-run never clobbers the published snapshot. Every
`results.json` records the git commit, whether the tree was dirty, the full config and the
seeds.

**Experiment 05 on a rented GPU.** Do not start there. Read
[`experiments/05_lm_training/README.md`](experiments/05_lm_training/README.md) first: it has
the box check, the smoke sweep to run before spending money, the dry-run planner, the
`rsync` list (809 MB of corpora and 26 MB of tokenizers, so the box does not re-download
4.2 GB), and an honest budget — 18–26 A100-hours, not the 11.4 h FLOP floor.

## Method notes

**SLP1 is the internal encoding.** All Sanskrit text is converted to SLP1 — a lossless
one-character-per-phoneme ASCII scheme — on ingest, and back to Devanagari or IAST only for
display; the original script is kept alongside and the Devanagari→SLP1→Devanagari roundtrip
is tested on FLORES. This is why byte counts and character counts coincide in the corpus
tables.

**Leakage is excluded in two layers.** SIGHUM, Hackathon and DCS-2018 are all subsets of
DCS, so an evaluation sentence can reach a training corpus by several routes. Every training
line is checked against `data/exclusion_hashes.txt` (sha256 of every evaluation sentence in
SLP1 form) *and* against a 24-letter shingle filter that catches near-duplicates — the
Mahābhārata is both a DCS text and the Itihāsa parallel corpus, sentence-split differently
by each, which the hash layer alone missed for 34,705 sentences. Both layers run on every
line of every corpus and their counts are reported in `data/README.md`.

**Every design decision is logged.** [`docs/decisions.md`](docs/decisions.md) is
append-only: why SLP1, why each substitution, every dataset filter, and — importantly —
every **CORRECTION**, where a later run invalidated an earlier claim. Three of this
project's headline numbers shrank under review; the log says so and by how much.
[`docs/paper_outline.md`](docs/paper_outline.md) is the research design.

**How the work was done.** Implementation was delegated to Opus subagents working from
written plans (`docs/superpowers/plans/`), while the orchestrating session kept the
research-design decisions, the review and the final verification — the process is stated in
[`CLAUDE.md`](CLAUDE.md) §0. Every experiment was then reviewed adversarially, and the
review is where the interesting failures were found: deleted punctuation credited as
compression, pre-existing hyphens credited as discovered boundaries, and line breaks leaking
into the pre-tokenizer so that 5–15% of every BPE vocabulary was dead entries.

## How to cite

Use [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{sharma2026sanskrittok,
  author    = {Sharma, Devansh},
  title     = {Recovering Latent Compression: Sandhi-Aware, Morpheme-Constrained
               Tokenization for Sanskrit Language Modelling},
  year      = {2026},
  url       = {https://github.com/DS436/sanskrit-token},
  license   = {MIT}
}
```

## Licence

Code is **MIT** ([`LICENSE`](LICENSE)). No corpus is distributed in this repository —
`data/raw/` and `data/processed/` are gitignored — with one deliberate exception: the test
fixture `tests/fixtures/dcs_mini.conllu` is three sentences copied verbatim from the
Digital Corpus of Sanskrit (CC BY 4.0), attributed in
[`data/README.md`](data/README.md#test-fixtures). Every other data source keeps its own
licence, listed per row in [`data/README.md`](data/README.md) along with its download date
and pinned revision.

## Acknowledgements

This project is entirely dependent on corpora and models built by other people, none of
which are redistributed here:

- **Digital Corpus of Sanskrit** — Oliver Hellwig, *DCS*, 2010–2024. The gold morpheme
  source; CC BY 4.0. Also Hellwig & Nehrdich 2018.
- **Sāmayik** — the primary En–Sa prose parallel corpus (Maheshwari et al., LREC-COLING
  2024, `2024.lrec-main.1245`; arXiv 2305.14004).
- **Itihāsa** — Aralikatte et al., *Itihāsa: A large-scale corpus for Sanskrit to English
  translation*, WAT 2021. The secondary, verse, parallel corpus.
- **FLORES-200** — NLLB Team et al., 2022. The parity anchor; CC BY-SA 4.0.
- **Sangraha** — AI4Bharat, verified Sanskrit split (Khan et al., *IndicLLMSuite*, 2024);
  CC BY 4.0. The bulk of the language-model corpus.
- **Sanskrit Wikipedia** — Wikimedia contributors, `20231101.sa` dump; CC BY-SA 3.0.
- **ByT5-Sanskrit** — Nehrdich, Hellwig & Keutzer, EMNLP Findings 2024
  (`chronbmm/sanskrit5-multitask`). The sandhi- and compound-splitting oracle.

Methodological debts: Zouhar et al. 2023 (Rényi efficiency) and Cognetta et al. 2024 (how it
can be gamed); Arnett & Bergen (MorphScore); Asgari et al. 2025 (MorphBPE); Petrov et al.
2023 and Ahia et al. 2023 (tokenizer inequity); Briggs 1985 (*Knowledge representation in
Sanskrit and artificial intelligence*, AI Magazine 6(1)) for the historical framing. The
language-model trainer is nanoGPT, vendored with attribution in `src/sanskrit_tok/lm/`.
Full reading list: [`docs/paper_outline.md`](docs/paper_outline.md) §12.
