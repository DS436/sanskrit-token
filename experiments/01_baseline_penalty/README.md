# Experiment 01 — Baseline penalty (RQ1)

**Hypothesis (CLAUDE.md §10):** English-centric tokenizers produce fertility > 5 on
Sanskrit, and the Sanskrit/Hindi parity ratio is > 1.5 on identical FLORES content,
because sandhi merges what Hindi keeps separate.

**Success:** `outputs/01_baseline_penalty/results.json`, a copy of `config.yaml`, and
`fertility_by_language.{pdf,png}` exist, carrying fertility, compression and parity for
each T0 arm × language × script variant, with the git commit and config recorded.

**Expected runtime:** ~13 s on CPU once `data/raw/flores/devtest.jsonl` and the Hugging
Face tokenizer caches are warm; a few minutes on a cold cache (one FLORES download, two
tokenizer downloads).

**Run:** `uv run python experiments/01_baseline_penalty/run.py`

---

## Summary

**On identical FLORES-200 devtest content, Sanskrit costs 1.77–2.19× as many tokens as its
English translation (`T0_gemma3` 1.77, `T0_o200k` 2.09, `T0_llama4` 2.19) but only
1.32–1.35× as many as its Hindi translation.** Against the hypothesis as written: Sa/En
parity > 1.5 holds; **Sa/Hi parity > 1.5 is refuted** (1.32–1.35), so sandhi does not buy
Sanskrit a large token penalty over another Devanagari language under these tokenizers;
and **fertility > 5 on Sanskrit is refuted** (3.11–3.88).

The gap between the two ratios is the point of the experiment. Over these 1012 sentences
Sanskrit is written in 16,975 whitespace words against English's 21,901 and Hindi's 25,643,
because sandhi and compounding fuse into one word what the other two languages write as
several. Dividing by that smaller word count makes the **fertility ratio overstate the
penalty**: Sa/En by fertility is 2.31–2.73 against a true parity of 1.77–2.19, and Sa/Hi by
fertility is 1.66–1.74 against a true parity of 1.32–1.35 — high enough that reading the
hypothesis off fertility would have *falsely confirmed* the Sa/Hi > 1.5 clause that parity
refutes. Fertility counts tokens per word; parity counts tokens per unit of meaning, which
is the quantity this project is about (CLAUDE.md §1, §2.1). Fewer words is not fewer
tokens, in either direction.

## Fertility (tokens per whitespace word), original script

Reported for comparability with the literature; **not** the headline metric, and not a
penalty measure. Sanskrit's denominator is its word count, which sandhi and compounding
make small (16,975 words here, against 21,901 English and 25,643 Hindi), so a high number
in this column mixes "the tokenizer segments badly" with "the language packs more into a
word" — and inflates any cross-language ratio taken from it. Compare it against the parity
table below, never in place of it.

| Arm | `san_Deva` | `hin_Deva` | `eng_Latn` | `san_Deva` (SLP1) | `hin_Deva` (SLP1) |
|---|---|---|---|---|---|
| `T0_o200k`  | 3.75 | 2.23 | 1.42 | 3.50 | 2.56 |
| `T0_llama4` | 3.88 | 2.34 | 1.42 | 3.56 | 2.57 |
| `T0_gemma3` | 3.11 | 1.79 | 1.35 | 3.50 | 2.52 |

Per-word standard deviation on `san_Deva` is large (1.67–2.05), i.e. the mean hides a long
tail of compounds — the distribution, not the mean, is what a compound-aware tokenizer has
to fix.

## Parity (Sanskrit tokens ÷ pivot tokens), 1012 aligned sentences

The headline of this experiment. `Sa(SLP1)/En` scores the SLP1 form of the Sanskrit side
against the same Latin-script English pivot.

| Arm | Sa/En | Sa/Hi | Sa(SLP1)/En |
|---|---|---|---|
| `T0_o200k`  | 2.09 | 1.33 | 2.18 |
| `T0_llama4` | 2.19 | 1.32 | 2.24 |
| `T0_gemma3` | 1.77 | 1.35 | 2.19 |

Two things to note. First, Sa/Hi is stable at ~1.33 across all three arms while Sa/En
ranges over 1.77–2.19: the English-relative penalty is largely a property of the
tokenizer, the Hindi-relative one a property of the language pair. Second, `T0_gemma3` is
the best of the three on Devanagari (Sa/En 1.77) but the *worst* once the Sanskrit side is
transliterated to SLP1 (2.19 vs 1.77) — its 262k vocabulary covers Devanagari well and
gains nothing from romanisation, which is a caution against assuming SLP1 is free.

## Compression (UTF-8 bytes per token), original script

`san_Deva` 5.86–7.21, `hin_Deva` 7.55–9.49, `eng_Latn` 4.87–4.92. Devanagari is 3 UTF-8
bytes per character, so these are not comparable across scripts; the SLP1 column
(`san_Deva` 2.31–2.38, `hin_Deva` 2.27–2.31) is the like-for-like one and puts Sanskrit
and Hindi within 3% of each other.

## Caveats

- **These three arms are existing practice, not a controlled comparison.** Their
  vocabularies differ (`T0_o200k` 200019, `T0_llama4` 201135, `T0_gemma3` 262145 ids), so
  differences between arms confound vocabulary size with tokenizer quality. Matched-vocab
  comparisons start at T1/T2 (CLAUDE.md §2.5). `results.json` records each arm's
  `source_id` and `vocab_size`.
- **Two arms load ungated mirrors** (`unsloth/Llama-4-Scout-17B-16E-Instruct`,
  `unsloth/gemma-3-4b-it`) because the official repos are gated and this machine has no
  `HF_TOKEN`; see `docs/decisions.md`.
- **SLP1 roundtrip:** 317/1012 `san_Deva` sentences do not survive
  Devanagari → SLP1 → Devanagari. 92 contain Latin letters and cannot, by construction;
  of the remaining 225, 136 contain ASCII digits (which come back as Devanagari digits)
  and 89 contain no ASCII letter or digit at all — those are almost entirely ASCII `.`
  and `|` used as danda, which SLP1 claims as phonemes, plus one nukta case. None of this
  affects the numbers above: every metric is computed on the original script and on SLP1
  separately, and the corpus stores the original script alongside (CLAUDE.md §2.3).
- **No sentences were dropped:** all 1012 FLORES devtest indices are non-blank in all
  three languages (`n_sentences_used` = `n_sentences_total` = 1012).
