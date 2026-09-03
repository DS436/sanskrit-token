# Recovering Latent Compression: Sandhi-Aware, Morpheme-Constrained Tokenization for Sanskrit Language Modelling

**Working paper outline and experimental design — v0.1, September 2026**

---

## 0. One-paragraph thesis

Sanskrit encodes more meaning per word than English because case, number, person, tense, mood and voice are fused into inflectional endings, and compounding (samāsa) packs clauses into single words. Standard subword tokenizers destroy this density: they were fit on English-dominant corpora, fragment Devanagari at the byte level, and are blind to sandhi, which phonologically erases word boundaries. The result is that Sanskrit pays a heavy "language tax" in tokens despite being linguistically compact. This paper tests whether a tokenizer that (a) reverses sandhi before subword learning and (b) forbids BPE merges across gold morpheme boundaries recovers that latent compression — measured not as tokens per word, which penalises long words, but as **tokens per unit of meaning** on parallel text, and validated by language-model training at small scale.

**The claim being tested is not "Sanskrit is efficient." It is "Sanskrit's efficiency is real but currently invisible to the tooling, and here is how much of it can be recovered."**

---

## 1. Research questions and hypotheses

| # | Question | Falsifiable hypothesis |
|---|----------|------------------------|
| RQ1 | How large is the tokenization penalty for Sanskrit under English-centric and Indic-specialised tokenizers? | H1: English-centric tokenizers produce fertility >5 on Sanskrit; Indic-specialised tokenizers reduce it but remain worse than Hindi on the same tokenizer, because sandhi merges what Hindi keeps separate. |
| RQ2 | When measured per unit of meaning rather than per word, does Sanskrit require fewer tokens than English for the same content? | H2: With a Sanskrit-native tokenizer, tokens-per-proposition on parallel corpora is lower for Sanskrit than English. With English-centric tokenizers it is higher. The sign flips depending on tokenizer. |
| RQ3 | Does sandhi-aware pre-tokenization improve tokenizer quality beyond what a Sanskrit-trained BPE/Unigram achieves alone? | H3: Sandhi splitting before subword learning raises MorphScore and lowers fertility relative to a matched-vocabulary tokenizer trained on raw sandhied text. |
| RQ4 | Does morpheme-constrained merging (MorphBPE-style) using Digital Corpus of Sanskrit gold segmentations improve language-model training efficiency? | H4: A model trained with the constrained tokenizer reaches a reference bits-per-character with fewer training tokens than the BPE control, in line with the 25–29% speedups reported for Hungarian and English. |
| RQ5 | Do intrinsic tokenizer improvements transfer to downstream Sanskrit tasks? | H5: Gains are largest on morphology-sensitive tasks (segmentation, morphological tagging) and smallest on knowledge-heavy tasks. Expect plateau effects. |

RQ2 is the headline result. RQ3 and RQ4 are the method contribution. RQ5 is the honesty check reviewers will demand.

---

## 2. Positioning against prior work

Four literatures exist. Nobody has joined the third and fourth for Sanskrit. That junction is the contribution.

### 2.1 Tokenizer inequity across languages
- Petrov et al. 2023 — "Language model tokenizers introduce unfairness between languages." Introduces parity ratio.
- Ahia et al. 2023 — cost and "language tax" framing.
- IndicGenBench (Singh et al. 2024) — fertility from 4.1 (Pashto) to 19.9 (Tibetan); high fertility means fewer in-context examples fit.
- MUTANT / IndicSuperTokenizer (2025) — LLaMA-4 tokenizer reaches fertility 10.5 on Oriya.
- Arnett & Bergen 2025 — "Explaining and mitigating crosslingual tokenizer inequities."

**Use for:** motivation, RQ1 baseline numbers. Do not re-prove; cite.

### 2.2 Indic-specialised tokenizers (crowded — do not compete here)
- Sarvam-1 (2024): 1.4–2.1 tokens/word on Indic scripts.
- SUTRA (Bendale et al. 2024): 256k vocab.
- IndicSuperTokenizer (2025): +39.5% fertility over LLaMA-4, +18% over Sutra, 44% inference throughput gain.
- MUTANT-Indic (2025/26): two-stage subword + multi-word recipe.
- BrahmicTokenizer-131K (2026): drop-in o200k replacement; evaluated on FLORES-200 and IN22-Gen.
- Paramanu (2024): monolingual Indic LMs under 400M params with novel tokenization.

**Use for:** the strongest baseline arm (T3). Key differentiator: these cover 22 scheduled languages but treat Sanskrit as one more Devanagari language. None handle sandhi. Check each for Sanskrit in training data.

### 2.3 Sanskrit word segmentation (mature — reuse as tooling, not as contribution)
- Hellwig 2015 — RNN joint compound + sandhi splitting, ~93% accuracy.
- Krishna et al. 2017 — SIGHUM dataset.
- Hellwig & Nehrdich 2018 (EMNLP) — char-level rCNN, DCS 2018 dataset.
- Aralikatte et al. 2018 (EMNLP) — seq2(seq)2 sandhi splitting.
- Krishnan et al. 2020 — Hackathon dataset.
- Krishna et al. 2020 — graph-based structured prediction.
- Dave et al. 2021 — two-stage RNN, data-driven only.
- Sandhan et al. 2022 — TransLIST, lexicon + transformer, SOTA on SIGHUM.
- Nehrdich, Hellwig, Keutzer 2024 (EMNLP Findings) — ByT5-Sanskrit; beats rcNN-SS by a wide margin on DCS 2018 and Hackathon, near TransLIST on SIGHUM, no lexicon needed.
- CharSS 2024 — character-level transformer; notes SLP1 encoding outperforms Devanagari for neural models.
- Sandarśana (ACM Computing Surveys 2025) — survey of Sanskrit computational linguistics.

**Use for:** the sandhi-splitting oracle in T4/T6. ByT5-Sanskrit is the pragmatic choice (no lexicon pipeline, robust off-distribution). Note DCS segmented forms are themselves partly machine-generated by the 2018 model — see threats.

### 2.4 Morphology-aware tokenization for LM training (active — Sanskrit absent)
- MorphPiece (Jabbar 2023/24) — English only; MorphGPT converges like a 6× larger model.
- MorphBPE (Asgari et al. 2025) — constrains merges to respect morpheme boundaries, inference unchanged; 29% fewer tokens to reference loss for Hungarian, 25% for English; tested on English, Russian, Hungarian, Arabic up to 1B params. **This is the method to adapt.**
- Brahma et al. 2025 — Hindi/Marathi linguist-informed segmentation + CBPE: up to 14% perplexity drop.
- Morpheus (Turkish, 2026) — morphology-aware tokenizer has *higher* fertility (1.73 vs 1.51) but *lower* BPC (1.425 vs 1.436). Fertility and quality decouple.
- DaMorph (Danish 2025), SKMT (Slovak 2026), MoVoC-Tok (Amharic/Tigrinya 2025).
- "Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE" (2025) — Unigram beats BPE on morphologically rich languages. **Implication: Unigram must be a baseline, not just BPE.**
- Emergent-mind synthesis: gains plateau once basic morphological alignment is reached; further improvement is limited by data and model capacity.

**Use for:** method (T5), metrics, expected effect sizes, and the plateau caveat.

### 2.5 Historical framing (handle with care)
- Briggs 1985, *AI Magazine*, "Knowledge Representation in Sanskrit and Artificial Intelligence" — argues Pāṇinian śābdabodha analysis resembles semantic-network representations. Legitimate; cite once in intro as intellectual lineage.
- Do **not** reference the "NASA chose Sanskrit" claim that grew around it. Reviewers will discount the whole paper.
- Pāṇini's Aṣṭādhyāyī as a generative rule system: cite Kiparsky, Scharf, Huet, Kulkarni (Sanskrit Heritage Reader). Frame as "why sandhi is formally specifiable and therefore reversible," not as "ancient wisdom."

---

## 3. Data

### 3.1 Sanskrit monolingual (tokenizer training + LM pretraining)
| Corpus | Size | Notes |
|--------|------|-------|
| Digital Corpus of Sanskrit (DCS, Hellwig 2010–) | ~650k sentences, ~4.5M word references, ~175k unique words, ~250 texts spanning 3,000 years | Gold lemma + morphology tags. Segmented forms partly auto-generated. Prose and verse. **Primary source for gold morphemes.** |
| UoH corpus + SandhiKosh | 62k train / 15.5k test sandhi instances | Apply Dave et al. 2021 pruning. |
| SIGHUM (Krishna et al. 2017) | 97k / 3k / 4.2k | DCS subset with Sanskrit Heritage Reader candidates. |
| Hackathon (Krishnan et al. 2020) | 90k / 10.3k / 10k | DCS subset. |
| Normalized SWS+MP dataset (LRE 2024) | DCS-derived | Adds compound constituency and type. |
| MITRA Buddhist corpus (2026) | Part of 4.4B-token multilingual set | Sanskrit share unknown; request. |
| GRETIL, Sanskrit Wikipedia, Wikisource | Unquantified; assemble | Needed to reach pretraining scale. |

**Realistic pretraining budget:** expect low hundreds of millions of tokens of clean Sanskrit at best. This forces small models and makes the tokens-to-reference-loss metric (RQ4) more valuable, not less.

### 3.2 Parallel corpora (for tokens-per-meaning, RQ2)
| Corpus | Pairs | Register | Role |
|--------|-------|----------|------|
| Itihāsa (Aralikatte et al. 2021) | 93k Sanskrit–English | Verse (Rāmāyaṇa, Mahābhārata); Dutt 1890s translation | Largest Sa–En. **Verse confound — see §7.** |
| Sāmayik (2023) | 53k English–Sanskrit | Contemporary prose | **Preferred for RQ2** — no meter, modern register. |
| SAHAAYAK 2023 | 1.5M Sanskrit–Hindi | Multi-domain | Sa–Hi parity; Hindi is the natural control (same script, no sandhi, less fusion). |
| FLORES-200 devtest (san_Deva) | ~1k sentences × 200 languages | Wikipedia prose | **Parity anchor**: identical content across Sanskrit, Hindi, English and 197 others. Use for Petrov-style parity ratio. |
| IN22-Gen | 22 Indic languages, identical sentence sets | | Secondary parity check. |
| IWLV-Ramayana (2026) | Sarga-aligned, multi-language | Chapter-level only | Not usable for sentence-level counts. |

### 3.3 Evaluation benchmarks (RQ5)
- Segmentation: SIGHUM, Hackathon, DCS 2018 test splits (compare to ByT5-Sanskrit, TransLIST numbers).
- DharmaBench (IJCNLP-AACL 2025): six Sanskrit classification/detection tasks on Buddhist texts.
- IndicParam (2025/26): Sanskrit subset + Sanskrit–English code-mixed set; labels distinguish knowledge vs. purely linguistic questions — **use the linguistic subset**.
- Itihāsa / Sāmayik MT: BLEU, chrF (expect small deltas; report anyway).
- Vedic dependency parsing (from ByT5-Sanskrit paper) if time permits.

---

## 4. Experimental arms (tokenizer conditions)

All trained-from-scratch tokenizers use matched vocabulary sizes (32k and 64k) and matched training data. Report both Devanagari and SLP1 encodings (see ablation A1).

| Arm | Tokenizer | Purpose |
|-----|-----------|---------|
| **T0** | Off-the-shelf English-centric: Llama-4, Gemma-3, GPT o200k_base | Establishes the penalty (RQ1). Control, not competitor. |
| **T1** | BPE trained on raw (sandhied) Sanskrit | Fair baseline. Any method must beat this at matched vocab. |
| **T2** | Unigram trained on raw Sanskrit | Required given 2025 evidence that Unigram dominates BPE on rich morphology. |
| **T3** | Indic-specialised off-the-shelf: Sarvam-1, SUTRA, IndicSuperTokenizer, BrahmicTokenizer-131K | Strongest existing practice. Note vocab-size mismatch in reporting. |
| **T4** | Sandhi-split pre-tokenization (ByT5-Sanskrit) → BPE and → Unigram | Isolates the effect of restoring word boundaries. |
| **T5** | MorphBPE-style constrained merges using DCS gold morpheme boundaries, on raw text | Isolates the effect of morpheme constraints. |
| **T6** | T4 + T5: sandhi-split, then morpheme-constrained merging | **Proposed method.** |
| **T7** | ByT5 byte-level (no subword) | Tokenizer-free floor; also the strongest existing Sanskrit model. |

### Ablations
- **A1 Script:** Devanagari vs. SLP1 vs. IAST. CharSS and others report SLP1 helps neural models; test whether it helps subword tokenizers. Also the Ukrainian transliteration result suggests lossless romanisation alone can cut fertility.
- **A2 Vocab size:** 16k / 32k / 64k / 128k for T1, T2, T6.
- **A3 Segmentation quality:** T6 with ByT5-Sanskrit splits vs. TransLIST splits vs. DCS gold splits (upper bound). Quantifies how much a noisy oracle costs.
- **A4 Constraint strictness:** hard boundary (no cross-morpheme merges) vs. soft penalty.
- **A5 Register:** train tokenizer on verse-only vs. prose-only vs. mixed; evaluate cross-register.

---

## 5. Metrics

### 5.1 Intrinsic (cheap; run on every arm)
| Metric | Definition | Why |
|--------|-----------|-----|
| Fertility | tokens per whitespace word | Standard; but **misleading for Sanskrit** because sandhi makes "words" long. Report but do not headline. |
| Compression | UTF-8 bytes per token | Vocab-size sensitive; report at matched vocab only. |
| Rényi efficiency (α=2.5 and 3) | Zouhar et al. 2023 | Report as diagnostic only; Cognetta et al. 2024 show it can be gamed. |
| MorphScore precision / recall | Arnett & Bergen 2024/25; token boundaries vs. DCS gold morpheme boundaries | Direct test of H3/H5. Precision vs. recall tells you over- vs. under-segmentation. |
| Vocabulary utilisation | fraction of vocab used on held-out text | Detects wasted capacity. |
| Parity ratio | Petrov et al. 2023; tokens(Sa) / tokens(En) and tokens(Sa) / tokens(Hi) on FLORES-200 | Cross-lingual fairness, identical content. |

### 5.2 The headline metric: tokens per proposition (TPP)
On a parallel corpus, for aligned pair (s, e):

TPP_ratio = tokens(s under tokenizer X) / tokens(e under tokenizer Y)

- Report Sanskrit under each arm vs. English under its native tokenizer (o200k or Llama-4).
- Report on Sāmayik (prose, primary), Itihāsa (verse, secondary), FLORES-200 (Wikipedia, tertiary).
- **Control for translation verbosity** (§7): compute also on Sa→En and En→Sa machine translations from the same system, and on the Hindi pivot.
- The paper's central figure: TPP_ratio across arms T0→T6, showing the crossover from >1 (Sanskrit costs more) to <1 (Sanskrit costs less).

### 5.3 Extrinsic (expensive; run on T1, T2, T3-best, T6, T7)
- Language modelling: **bits per character** (not perplexity — perplexity is not comparable across tokenizers with different vocabularies). Also bits per byte.
- Tokens-to-reference-BPC (MorphBPE protocol): training tokens needed to hit a fixed BPC. This is the RQ4 result.
- Downstream (§3.3). Fine-tune identical architectures; three seeds; report mean ± std.

---

## 6. Model training protocol

- Architecture: decoder-only GPT-2-style, nanoGPT-based, following Regional TinyStories methodology.
- Sizes: 50M and 125M parameters. Add 350M only if compute allows; the effect direction at 50M/125M is the result.
- Matched **parameters** (embedding size varies with vocab) and, separately, matched **FLOPs**. Report both; they disagree at small scale.
- Context: 1024 tokens.
- Data: identical raw text for all arms; the tokenizer is the only variable.
- Seeds: 3 per condition minimum.
- Encoding: SLP1 internally; convert on output.
- Compute estimate: ~14 tokenizer arms × 2 vocab sizes for intrinsic (trivial); ~5 arms × 2 sizes × 3 seeds = 30 LM runs at 50–125M on ~200M tokens. Feasible on a single A100 over weeks, or a small cluster over days.

---

## 7. Threats to validity (write this section early; reviewers will)

1. **Verse and meter.** Itihāsa is śloka — eight syllables per quarter. Meter constrains word choice and inflates compounding. Prose (Sāmayik) must be the primary RQ2 corpus; verse is a secondary register.
2. **Translation length bias.** Translations are systematically longer than sources (explicitation). Dutt's 1890s English is verbose. Mitigations: use both directions; use Hindi pivot; use FLORES where all languages are translations of the same English source.
3. **DCS segmentation noise.** DCS segmented forms are generated by the Hellwig–Nehrdich 2018 model; TransLIST could not use DCS10k because half lacked gold inflections. Use only gold-verified subsets for MorphScore; report oracle (A3) to bound the effect.
4. **Data leakage.** SIGHUM, Hackathon, and DCS 2018 are all DCS subsets. Tokenizer and LM training data must exclude all evaluation splits at the sentence level, and ideally at the text level.
5. **Fertility ≠ quality.** Morpheus shows better BPC with worse fertility. Do not headline fertility. Headline TPP and BPC.
6. **Plateau.** Prior work finds morphological gains saturate; beyond alignment, data volume dominates. Sanskrit is data-poor. Frame expected gains as modest and report null results.
7. **Scale generalisation.** Results at 125M may not hold at 7B. Say so. MorphBPE held to 1B, which is the citable ceiling.
8. **Tokenizer training data mismatch** for T3 baselines (different corpora, vocab sizes). Report as "existing practice," not as controlled comparison.
9. **Low-resource evaluation noise.** IndicParam: best model gets 58% on extremely-low-resource Indic. Variance is high; three seeds is the minimum.

---

## 8. Predicted outcomes (pre-register these)

- T0 fertility on Sanskrit: 5–12. Hindi under the same tokenizer: 2–4. Gap attributable to sandhi and compounding.
- T3 (Indic-specialised) closes most of the gap vs. Hindi but not all.
- T4 (sandhi-split) alone: fertility drops; MorphScore recall rises; BPC roughly flat or slightly worse (boundaries add tokens).
- T5 (morpheme-constrained) alone: MorphScore precision rises sharply; fertility may rise slightly; BPC improves; tokens-to-reference-BPC drops 10–25%.
- T6: best MorphScore; TPP_ratio on Sāmayik crosses below 1.0 against English o200k; BPC best among subword arms; T7 (ByT5) competitive on BPC but at much higher sequence length and compute.
- Downstream: segmentation and morphological tagging improve clearly; DharmaBench and IndicParam knowledge tasks move within noise.

If T6 does not cross TPP < 1.0 on prose, the paper's finding becomes "the density is real at the word level but does not survive subword tokenization even with morphological constraints," which is still publishable and arguably more interesting.

---

## 9. Paper structure

1. Introduction (1.5 pp) — the two-claims separation; Briggs lineage; contributions list.
2. Background (2 pp) — Sanskrit morphology and sandhi for an NLP audience; why boundaries are formally recoverable.
3. Related work (1.5 pp) — the four literatures, §2 above.
4. Method (2 pp) — sandhi-aware pre-tokenization; morpheme-constrained merging; TPP metric definition.
5. Experimental setup (2 pp) — data, arms, metrics, protocol.
6. Results (3 pp) — RQ1 → RQ5 in order; central TPP crossover figure; BPC and tokens-to-loss table; downstream table.
7. Analysis (1.5 pp) — where the gains come from (A1–A5 ablations); error analysis on sandhi types (vowel, visarga, consonant); compound-length effects.
8. Threats to validity (0.5 pp).
9. Conclusion (0.5 pp).
Appendix — full tables, hyperparameters, corpus assembly, SLP1 mapping.

Target: ACL / EMNLP main or Findings; or the Sanskrit Computational Linguistics Symposium (ISCLS) for an early version. Preprint on arXiv once RQ1–RQ2 are done; those alone are a workshop paper.

---

## 10. Milestones

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| M1 (weeks 1–3) | Corpus assembly: DCS, Sāmayik, Itihāsa, FLORES san_Deva; SLP1 conversion; leakage-safe splits | Data access |
| M2 (weeks 3–5) | RQ1 + RQ2 intrinsic results across T0–T3. **This is a standalone workshop paper.** | M1 |
| M3 (weeks 5–8) | ByT5-Sanskrit sandhi-splitting pipeline; T4 tokenizers; MorphScore against DCS gold | M1 |
| M4 (weeks 8–11) | MorphBPE adaptation (subclass HF BPE trainer, override merge scoring); T5, T6 | M3 |
| M5 (weeks 11–16) | LM training runs; BPC; tokens-to-reference-loss | M4, compute |
| M6 (weeks 16–20) | Downstream fine-tuning and evaluation | M5 |
| M7 (weeks 20–24) | Ablations A1–A5; writing | M6 |

---

## 11. Immediate next actions

1. Download DCS and confirm which subsets have human-verified segmentation.
2. Download Sāmayik, Itihāsa, FLORES-200 san_Deva/hin_Deva/eng_Latn devtest.
3. Run T0 fertility and parity on FLORES for Sanskrit vs. Hindi vs. English with Llama-4 / Gemma-3 / o200k. This is a one-afternoon experiment and gives the first figure of the paper.
4. Get ByT5-Sanskrit running for segmentation; measure its accuracy on a DCS gold subset to bound oracle noise.
5. Read MorphBPE's code (HF Tokenizers subclass) and confirm the merge-constraint hook is reusable.
6. Check whether Sarvam, SUTRA, IndicSuperTokenizer included Sanskrit in tokenizer training; email authors if unclear.

---

## 12. Reading list (grouped; author–year keys used above)

**Inequity and evaluation**
- Petrov, La Malfa, Torr, Bibi 2023. Language model tokenizers introduce unfairness between languages. NeurIPS.
- Ahia et al. 2023. Do all languages cost the same?
- Zouhar et al. 2023. Tokenization and the noiseless channel (Rényi efficiency). ACL.
- Cognetta et al. 2024. Rényi efficiency can be gamed.
- Arnett & Bergen 2024/25. MorphScore; Evaluating morphological alignment of tokenizers in 70 languages; Explaining and mitigating crosslingual tokenizer inequities.
- Ali et al. 2024. Tokenizer choice for LLM training: negligible or crucial? ACL Findings.
- Singh et al. 2024. IndicGenBench.
- Stop Taking Tokenizers for Granted (2026). Survey.
- TokEval (2026). Evaluation suite.
- Parity-Aware BPE (ACL 2026).

**Indic tokenizers**
- Sarvam-1 (2024); SUTRA (Bendale et al. 2024); IndicSuperTokenizer (2025); MUTANT (2025/26); BrahmicTokenizer-131K (2026); Paramanu (2024); Regional TinyStories (2025); Multilingual Tokenization through the Lens of Indian Languages (2025); Brahma et al. 2025 (Hindi/Marathi CBPE).

**Sanskrit segmentation and resources**
- Hellwig 2010–. Digital Corpus of Sanskrit.
- Hellwig 2015; Hellwig & Nehrdich 2018; Krishna et al. 2017, 2020; Krishnan et al. 2020; Aralikatte et al. 2018; Dave et al. 2021; Sandhan et al. 2022 (TransLIST); Nehrdich, Hellwig, Keutzer 2024 (ByT5-Sanskrit); CharSS 2024; Normalized SWS+MP dataset (LRE 2024); Sandarśana survey (ACM CSUR 2025).
- Aralikatte et al. 2021. Itihāsa. WAT.
- Sāmayik 2023.
- SAHAAYAK 2023.
- DharmaBench 2025; IndicParam 2025/26; MITRA 2026.
- Scharf & Hyman 2011. Linguistic issues in encoding Sanskrit.
- Hyman 2008. From Pāṇinian sandhi to finite-state calculus.
- Huet; Kulkarni; Goyal & Huet 2013 (Sanskrit Heritage Reader).

**Morphology-aware tokenization**
- Jabbar 2023/24 (MorphPiece); Asgari et al. 2025 (MorphBPE); Morpheus 2026 (Turkish); DaMorph 2025 (Danish); SKMT 2026 (Slovak); MoVoC-Tok 2025; Rethinking Tokenization for Rich Morphology: Unigram vs BPE (2025); Limisiewicz et al. 2024 (MYTE).

**Lineage**
- Briggs 1985. Knowledge representation in Sanskrit and artificial intelligence. AI Magazine 6(1).
