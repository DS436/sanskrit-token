# Experiment 01 — Baseline penalty (RQ1)

**Hypothesis (CLAUDE.md §10):** English-centric tokenizers produce fertility > 5 on
Sanskrit, and the Sanskrit/Hindi parity ratio is > 1.5 on identical FLORES content,
because sandhi merges what Hindi keeps separate.

**Success:** `outputs/01_baseline_penalty/results.json`, a copy of `config.yaml`, and
`fertility_by_language.{pdf,png}` exist, carrying fertility, compression and parity for
each T0 arm × language × script variant, with the git commit and config recorded.

**Expected runtime:** ~13 s on CPU once `data/raw/flores/devtest.jsonl` and the Hugging
Face tokenizer caches are warm; a few minutes on a cold cache (one FLORES download, three
Hugging Face tokenizer downloads plus `T0_o200k`'s tiktoken BPE file).

**Run:** `uv run python experiments/01_baseline_penalty/run.py`

**Committed snapshot:** `outputs/` is gitignored; the `results.json`, `config.yaml` and
figures of the run this file reports are tracked at
[`results/01_baseline_penalty/`](../../results/01_baseline_penalty/) — see
[`results/README.md`](../../results/README.md). Re-running writes to `outputs/` and leaves
the snapshot untouched.

---

## Summary

**On identical FLORES-200 devtest content, Sanskrit costs 1.77–7.86× as many tokens as its
English translation (`T0_gemma3` 1.77, `T0_o200k` 2.09, `T0_llama4` 2.19, `T0_gpt2` 7.86)
but only 1.06–1.35× as many as its Hindi translation (`T0_gpt2` 1.06, `T0_llama4` 1.33,
`T0_o200k` 1.33, `T0_gemma3` 1.35).** Against the hypothesis as written: Sa/En parity > 1.5
holds, and holds harder once an older, smaller-vocabulary arm is in the mix; **Sa/Hi
parity > 1.5 is refuted** (1.06–1.35, refuted more decisively than before), so sandhi does
not buy Sanskrit a large token penalty over another Devanagari language under any of these
four tokenizers. **Fertility > 5 on Sanskrit is refuted for every ≥200k-vocab arm**
(3.11–3.88) **but confirmed for `T0_gpt2`'s 50,257-token vocabulary on raw Devanagari**
(12.49) — the same arm's SLP1 variant stays low (3.97), so the >5 result is a property of
an old, Devanagari-blind vocabulary meeting three-byte-per-character UTF-8, not new
evidence about Sanskrit; see the `T0_gpt2` paragraph below.

The gap between the parity and fertility ratios is the point of the experiment. Over these
1012 sentences Sanskrit is written in 16,975 whitespace words against English's 21,901 and
Hindi's 25,643, because sandhi and compounding fuse into one word what the other two
languages write as several. Dividing by that smaller word count makes the **fertility
ratio overstate the penalty**: Sa/En by fertility is 2.31–8.38 against a true parity of
1.77–7.86, and Sa/Hi by fertility is 1.60–1.74 against a true parity of 1.06–1.35 — high
enough that reading the hypothesis off fertility would have *falsely confirmed* the Sa/Hi
> 1.5 clause that parity refutes, for all four arms. Fertility counts tokens per word;
parity counts tokens per unit of meaning, which is the quantity this project is about
(CLAUDE.md §1, §2.1). Fewer words is not fewer tokens, in either direction.

## Fertility (tokens per whitespace word), original script

Reported for comparability with the literature; **not** the headline metric, and not a
penalty measure. Sanskrit's denominator is its word count, which sandhi and compounding
make small (16,975 words here, against 21,901 English and 25,643 Hindi), so a high number
in this column mixes "the tokenizer segments badly" with "the language packs more into a
word" — and inflates any cross-language ratio taken from it. Compare it against the parity
table below, never in place of it.

| Arm | `san_Deva` | `hin_Deva` | `eng_Latn` | `san_Deva` (SLP1) | `hin_Deva` (SLP1) † |
|---|---|---|---|---|---|
| `T0_o200k`  | 3.75 | 2.23 | 1.42 | 3.50 | 2.56 |
| `T0_llama4` | 3.88 | 2.34 | 1.42 | 3.56 | 2.57 |
| `T0_gemma3` | 3.11 | 1.79 | 1.35 | 3.50 | 2.52 |
| `T0_gpt2`   | 12.49 | 7.82 | 1.49 | 3.97 | 2.85 |

† **Approximate — Hindi is outside SLP1's inventory.** SLP1 encodes the Sanskrit phoneme
set, so transliterating Hindi is a lossy approximation rather than a change of script; see
the SLP1-coverage caveat below before reading this column, and never quote it as a
measurement of Hindi. The `san_Deva` (SLP1) column is affected far less, but not zero.

**`T0_gpt2` is the outlier this table exists to show.** Its 50,257-token vocabulary
(2019-era, English-only training data) has essentially no dedicated Devanagari coverage,
so raw `san_Deva` text falls back to something close to byte-level segmentation: fertility
12.49, more than triple the next-worst arm, and above the pre-registered >5 threshold that
the three ≥200k-vocab arms all fall under. `eng_Latn` fertility for the same arm (1.49) is
unremarkable — the effect is specific to a script the vocabulary was never trained on, not
a general property of an older tokenizer. Its `san_Deva` (SLP1) fertility (3.97) sits in
the same band as the other three arms' SLP1 columns, because SLP1 is ASCII and every one of
these vocabularies covers ASCII well; see the parity section for the token-count version of
this same swing.

Per-word standard deviation on `san_Deva` is large (1.67–7.40, the top of that range being
`T0_gpt2`), i.e. the mean hides a long tail of compounds — the distribution, not the mean,
is what a compound-aware tokenizer has to fix.

## Parity (Sanskrit tokens ÷ pivot tokens), 1012 aligned sentences

The headline of this experiment. `Sa(SLP1)/En` scores the SLP1 form of the Sanskrit side
against the same Latin-script English pivot.

| Arm | Sa/En | Sa/Hi | Sa(SLP1)/En |
|---|---|---|---|
| `T0_o200k`  | 2.09 | 1.33 | 2.18 |
| `T0_llama4` | 2.19 | 1.32 | 2.24 |
| `T0_gemma3` | 1.77 | 1.35 | 2.19 |
| `T0_gpt2`   | 7.86 | 1.06 | 2.52 |

Two things to note. First, Sa/Hi is stable at ~1.06–1.35 across all four arms — narrower,
in fact, once `T0_gpt2` (1.06) is included — while Sa/En ranges far more widely, 1.77–7.86:
the English-relative penalty is sensitive to the tokenizer (chiefly its vocabulary's
Devanagari coverage), the Hindi-relative one is closer to a fixed property of the language
pair regardless of tokenizer. Second, SLP1 transliteration cuts in opposite directions
depending on whether an arm's vocabulary already covers Devanagari. `T0_gemma3` is the
best of the three ≥200k-vocab arms on raw Devanagari (Sa/En 1.77) and *loses* the most from
romanising among them: SLP1 costs it +0.41 (1.77 → 2.19), against +0.09 for `T0_o200k`
(2.09 → 2.18) and +0.05 for `T0_llama4` (2.19 → 2.24) — its 262k vocabulary covers
Devanagari well and gains nothing from romanisation, a caution against assuming SLP1 is
free. `T0_gpt2` is the opposite case and by far the largest swing of any arm: raw-script
Sa/En is 7.86 (its 50k vocabulary has no real Devanagari coverage — see the fertility
section), and SLP1 collapses that to 2.52, a **−5.34** change, larger in magnitude than the
other three arms' SLP1 deltas combined. In other words, SLP1 is close to free (or a small
loss) for a tokenizer whose vocabulary already spans Devanagari, and a large win for one
that does not — the same conclusion the fertility table points to, now in the metric that
actually measures token cost.

## Compression (UTF-8 bytes per token), original script

`san_Deva` 1.63–7.21, `hin_Deva` 1.68–9.49, `eng_Latn` 4.87–4.92. Devanagari is 3 UTF-8
bytes per character, so these are not comparable across scripts; the SLP1 column
(`san_Deva` 2.05–2.38, `hin_Deva` 2.02–2.31) is the closer to like-for-like of the two, and
under this approximate transliteration it puts Sanskrit and Hindi within a few percent of
each other. That gap is not a measurement of Hindi: half its sentences carry a nukta SLP1
cannot encode, so the Hindi denominator is built from strings SLP1 only partly represents
(see the SLP1-coverage caveat).

**The low end of both original-script ranges is `T0_gpt2`** (`san_Deva` 1.63, `hin_Deva`
1.68 bytes/token) — well under half of the next-lowest arm — which is the same fact as its
high fertility read from the other direction: its vocabulary has so little dedicated
Devanagari coverage that most tokens cover only a fraction of one three-byte character. Its
`eng_Latn` compression (4.88) and both SLP1 columns (2.05, 2.02) sit in the same band as
the other three arms, because those inputs are the ASCII text its vocabulary was actually
trained on.

## Caveats

- **These four arms are existing practice, not a controlled comparison.** Their
  vocabularies differ by more than 5×: `T0_gpt2` 50257, `T0_o200k` 200019, `T0_llama4`
  201135, `T0_gemma3` 262145 ids. Differences between arms confound vocabulary size (and,
  for `T0_gpt2`, tokenizer generation and training-data era) with tokenizer quality, and
  `T0_gpt2`'s numbers above are the clearest illustration of that confound in this whole
  table. Matched-vocab comparisons start at T1/T2 (CLAUDE.md §2.5). `results.json` records
  each arm's `source_id` and `vocab_size`.
- **Two arms load ungated mirrors** (`unsloth/Llama-4-Scout-17B-16E-Instruct`,
  `unsloth/gemma-3-4b-it`) because the official repos are gated and this machine has no
  `HF_TOKEN`; see `docs/decisions.md`. `T0_gpt2` (`openai-community/gpt2`) is ungated and
  loads directly — no substitution to record.
- **SLP1 roundtrip:** 317/1012 `san_Deva` sentences do not survive
  Devanagari → SLP1 → Devanagari. 92 contain Latin letters and cannot, by construction;
  of the remaining 225, 136 contain ASCII digits (which come back as Devanagari digits)
  and 89 contain no ASCII letter or digit at all — those are almost entirely ASCII `.`
  and `|` used as danda, which SLP1 claims as phonemes, plus one nukta case (that is one
  nukta sentence *in this bucket*; the corpus holds 40, the rest of which also carry Latin
  letters or ASCII digits — see the SLP1-coverage caveat below). None of this
  affects the numbers above: every metric is computed on the original script and on SLP1
  separately, and the corpus stores the original script alongside (CLAUDE.md §2.3).
- **SLP1 coverage — the `hin_Deva` (SLP1) column is approximate.** SLP1 encodes the
  Sanskrit phoneme inventory, so applying it to Hindi is an approximation, not a
  transliteration, and it fails in two ways. Nukta consonants (`क़ ज़ ड़ ढ़ फ़`) have no SLP1
  phoneme and `sanscript` emits the sign as a literal ASCII `0` (`क़` → `k0a`); signs
  outside the scheme, chiefly candra-o and candra-e (`ॉ ऑ ॅ`), pass through unconverted, so
  the "ASCII" SLP1 string still holds raw three-byte Devanagari (`डॉक्टर` → `qaॉkwara`).
  On FLORES devtest (`slp1_coverage` in `results.json`): **`hin_Deva` 513/1012 sentences
  contain a nukta and 256/1012 SLP1 strings still contain a non-ASCII character**;
  `san_Deva` is affected too but far less, at **40/1012 and 100/1012**. Both effects
  inflate the SLP1 byte count and change how a tokenizer segments the string, so the
  `hin_Deva` SLP1 fertility and compression numbers are indicative only. Everything in the
  original-script columns, and every parity number, is unaffected: parity's SLP1 row scores
  the *Sanskrit* side against a Latin English pivot and never touches Hindi.
- **No sentences were dropped:** all 1012 FLORES devtest indices are non-blank in all
  three languages (`n_sentences_used` = `n_sentences_total` = 1012).
