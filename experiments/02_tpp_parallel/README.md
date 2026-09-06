# Experiment 02 — Tokens per proposition on parallel text (RQ2)

**Hypothesis (outline §1, H2, verbatim):** With a Sanskrit-native tokenizer,
tokens-per-proposition on parallel corpora is lower for Sanskrit than English. With
English-centric tokenizers it is higher. The sign flips depending on tokenizer.

**Success (pre-registered, as reframed):** at least one Sanskrit-native arm below 1.0 on
Sāmayik with its CI excluding 1.0 **against the matched English control `E1_*`** — same
algorithm, same vocabulary size, same training corpus. The comparison against the T0
(English-centric) arms cannot establish it: those arms differ from the trained Sanskrit
arms in vocabulary size and training domain as well as language, so a flip there is not
attributable to language (`docs/decisions.md`, "Experiment 02 verdict reframed…").

**Scope:** this experiment can only test the Sanskrit-native arms that exist, which today
are the raw-subword baselines `T1_bpe_raw_{32k,64k}` and `T2_unigram_raw_{32k,64k}`. The
sandhi-split (`T4_*`) and sandhi-split + morpheme-constrained (`T6_*`) arms — the ones
this project proposes — are Experiments 03 and 04 and are not built yet, so no result
here settles H2 in general.

**Run:** `uv run python experiments/02_tpp_parallel/run.py`

**Committed snapshot:** `outputs/` is gitignored; the `results.json`, `config.yaml` and
figures of the run this file reports are tracked at
[`results/02_tpp_parallel/`](../../results/02_tpp_parallel/) — see
[`results/README.md`](../../results/README.md). Re-running writes to `outputs/` and leaves
the snapshot untouched.

> **Re-run 2026-09-05 at commit `b7302a8`, clean tree.** Every trained arm was retrained
> after the trainers stopped letting line breaks reach the pre-tokenizer, so every number
> on this page moved slightly and four of them are new
> (`docs/decisions.md`, 2026-09-05, "Trainers strip newlines" and the CORRECTION entry for
> this experiment). **No conclusion changed in the controlled comparison.** Two rows of the
> *deployed-practice* tables did: `T1_bpe_raw_32k`* against `T0_o200k` and against
> `T0_llama4` on Sāmayik test now sit below 1.0 with their CIs excluding it, where before
> they straddled it. `E1_unigram_64k` now trains to **62,896** pieces rather than 64,000,
> which is recorded below and in the decision log.

**Runtime:** 2m09s wall-clock (113.0s user, 88% CPU) on this machine with every tokenizer
cache, corpus jsonl and trained `tokenizer.json` already warm — no network access,
essentially unchanged from the 2m07s of the pre-E1 run. `english_pivots` stays at the two
deployed arms; the controlled comparison adds only the sixteen `controlled_pairs`
measurements (four pairs x four corpora). A cold run additionally pays
for four Sāmayik/Itihāsa split downloads (already cached under `data/raw/`), the tiktoken
download for `T0_o200k` and Hugging Face downloads for six more Sanskrit arms (all cached;
the four T1/T2 arms and the four E1 pivots are local files, and `T3_indicsuper` resolves
to nothing — see below). Training the four E1 arms first (a one-off,
`uv run python experiments/02_tpp_parallel/train_tokenizers.py`) took 8.8s wall-clock.

---

## Summary

**Against the matched English control, the Sāmayik sign flip disappears.** On Sāmayik test
— the primary prose corpus this hypothesis is pre-registered against — every one of the
four matched pairs (all four Sanskrit sides are raw-subword baselines; see **Scope**
above) costs *more* tokens per proposition in Sanskrit than in English, with
every 95% CI entirely above 1.0: `T1_bpe_raw_64k`*/`E1_bpe_64k` **1.035 [1.021, 1.049]**,
up to `T2_unigram_raw_32k`*/`E1_unigram_32k` 1.142 [1.128, 1.157]. The same Sanskrit arm
measured against the deployed 200k-vocabulary `T0_o200k` reads 0.887 [0.875, 0.899] —
below parity. Nothing about the Sanskrit side changed between those two numbers; the
English side did. `E1_bpe_64k` spends 33,702 tokens on the English half of Sāmayik test
where `T0_o200k` spends 39,339 (14.3% fewer), because it too was trained on this domain.
That denominator accounts for the whole move: rescaling the measured 0.887 by
39,339/33,702 gives 1.035, exactly the measured controlled value. The apparent Sanskrit
advantage was the English pivot's handicap.

**The controlled numbers on the other three corpora, in the same direction.** Sāmayik
`test_ood` (Mann Ki Baat transcripts, out-of-domain for *both* sides) 1.061–1.163, all four
CIs above 1.0; FLORES devtest (out-of-domain for both sides) 1.143–1.219, likewise. Only
**Itihāsa test stays below 1.0 under the control** — 0.598 [0.594, 0.601] to 0.658
[0.655, 0.661], all four CIs excluding 1.0 — a smaller flip than the 0.46–0.55 the same
arms show against `T0_o200k`, but a decisive one. Itihāsa is verse: meter, not
tokenization, is the leading suspect for why Sanskrit looks compact there (CLAUDE.md
§2.7), and its English side is a 19th-century verse translation whose own verbosity sits
in the denominator. So the one surviving flip is on the one corpus whose confounds this
experiment was already told not to trust.

**Deployed practice is still the robust finding.** Against the tokenizers people actually
use, Sanskrit costs 1.8–2.9 English tokens per proposition on prose and Wikipedia text
(`T0_o200k` 1.835 [1.813, 1.858] on Sāmayik test, up to `T3_sarvam` 2.899 [2.865, 2.929]
on FLORES), for every off-the-shelf arm, English-centric and Indic alike. Those rows are
byte-identical across every run of this experiment: no off-the-shelf arm was retrained.
The trained `T1`/`T2` rows in the same tables did move with the newline fix, and two of
them crossed 1.0: `T1_bpe_raw_32k`* against `T0_o200k` on Sāmayik test went from
1.009 [0.995, 1.023] to **0.984 [0.970, 0.998]**, and against `T0_llama4` from
0.991 [0.977, 1.005] to **0.966 [0.952, 0.980]**. Both are deployed-practice columns, not
controls, so neither changes a verdict — a Sanskrit arm beating a 200k general-domain
English tokenizer is a statement about vocabulary size and domain, which is exactly why
the E1 table exists.

**Verdict, scoped to the arms that exist: under matched conditions, *raw subword*
training on Sanskrit does not by itself bring TPP below English on prose.** Read the arm
column before reading the verdict. The only Sanskrit-native tokenizers built so far are
`T1_*` (BPE on raw, sandhied, compounded text) and `T2_*` (Unigram on the same) — subword
learners given no morphological information whatsoever, which CLAUDE.md §6 lists as the
project's *baselines*, not its proposal. The proposed arms are not built yet:
**`T4_*`** (reverse sandhi first, then learn subwords) is Experiment 03 and **`T6_*`**
(sandhi-split *and* forbid merges across gold morpheme boundaries — the tokenizer this
project actually proposes) is Experiment 04. Neither has been trained, let alone measured.
So what follows is a result about T1/T2, not a verdict on H2.

For T1/T2, the domain-fit question this experiment was carrying is now settled in one
direction on prose. Domain fit says a tokenizer trained on a corpus is cheap on that
corpus's held-out split and no cheaper anywhere else. E1 shares the training domain with
T1/T2 corpus-for-corpus, so at each corpus both sides of the ratio now have the same
home-field advantage or the same lack of it: in-domain on Sāmayik test and Itihāsa test,
out-of-domain on `test_ood` and FLORES. The Sāmayik-test flip did not survive that
equalisation — it vanished, and it vanished by the amount the English side gained. That is
the signature of domain fit rather than a language effect, which is what the matched
control was added to distinguish (`docs/decisions.md`, "Add a matched English control
family E1 for TPP"). The Itihāsa flip *did* survive the control, in exactly the register
CLAUDE.md §2.7 says not to lead with; it is a lead to follow on verse, not a prose result.

**What this leaves open — and it is most of the hypothesis.** H2 is about a
Sanskrit-*native* tokenizer recovering density an English-centric one destroys, and the
mechanism this project proposes for recovering it is precisely the thing T1/T2 lack: a raw
BPE tokenizer is handed `tadapi` as one opaque string and has no way to know it is `tat +
api`, so it cannot spend one token per morpheme however much Sanskrit text it sees. That
T1/T2 do not beat a matched English control on prose is evidence that scale-and-vocabulary
alone do not recover the density — it says nothing about whether sandhi-splitting (T4) or
merge-constrained training (T6) will. Those are the next two experiments, and this result
is the baseline they have to beat: **1.035 on Sāmayik test against `E1_bpe_64k`** is the
number T4 and T6 must push below 1.0 with a CI excluding it.

Two further caveats keep even the T1/T2 result provisional. First, both sides are trained
on ~116–118k sentences of the same two parallel corpora, so the control equalises domain
*fit* but not corpus *size or diversity*, and neither side has seen a real monolingual
corpus. **M1, the monolingual retrain** (T1/T2 on DCS, GRETIL, Wikipedia) moves the
Sanskrit side out of every evaluation domain — a harder test than this one, not an easier
one. Second, TPP against a translated pivot inherits the translator's verbosity; a
Sanskrit–English pair is not a controlled propositional unit, only the best available
proxy. Read alongside Experiment 01, what is settled is: Sanskrit's word-level density is
real, it does not survive an English-centric tokenizer, and matched-size raw subword
training on Sanskrit does not by itself buy it back on prose.

---

## The controlled comparison (matched English control, `E1_*`)

Each row is one matched pair: a trained Sanskrit arm over the English arm sharing its
algorithm, its vocabulary size and its training corpus (the two sides of the same Sāmayik
+ Itihāsa training splits). This is the controlled comparison and the one the verdict
reads from; the tables after it are deployed practice. Sanskrit is scored in SLP1 (T1/T2
have no other variant), English as written. Values are TPP with a 95% bootstrap CI; prose
first (CLAUDE.md §2.7). `*` marks the provisional Sanskrit arms (see caveats).

| Matched pair (Sanskrit / English) | Sāmayik test (prose) | Sāmayik test_ood (prose, OOD) | Itihāsa test (verse) | FLORES devtest |
|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` (32k) | 1.084 [1.070, 1.098] | 1.086 [1.072, 1.098] | **0.645 [0.642, 0.649]** | 1.143 [1.131, 1.154] |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` (64k) | 1.035 [1.021, 1.049] | 1.061 [1.048, 1.073] | **0.598 [0.594, 0.601]** | 1.144 [1.131, 1.155] |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` (32k) | 1.142 [1.128, 1.157] | 1.163 [1.148, 1.176] | **0.658 [0.655, 0.661]** | 1.206 [1.194, 1.219] |
| `T2_unigram_raw_64k`* / `E1_unigram_64k`‡ (64k) | 1.107 [1.092, 1.121] | 1.156 [1.142, 1.169] | **0.623 [0.619, 0.626]** | 1.219 [1.206, 1.232] |

Per corpus: **Sāmayik test** — 0 pairs below 1.0, 4 above (CIs exclude 1.0), 0 straddling.
**Sāmayik test_ood** — 0 below, 4 above, 0 straddling. **Itihāsa test** — 4 below (CIs
exclude 1.0), 0 above. **FLORES devtest** — 0 below, 4 above, 0 straddling. `n` is 2417 /
4047 / 11721 / 1012 pairs respectively and `n_undefined` is 0 throughout: no English
sentence encoded to zero tokens under any control arm.

Why the same Sanskrit arms read differently here than in the tables below: the control
arms are cheaper on English than the deployed pivot, because they share its domain.
`E1_bpe_64k` uses 33,702 tokens on Sāmayik test's English side against `T0_o200k`'s 39,339
(14.3% fewer) and 374,789 against 484,898 on Itihāsa test (22.7% fewer). A ratio whose
denominator shrinks by a tenth to a fifth moves accordingly, and that movement is
vocabulary and domain, not language — which is the entire reason this table exists.

**‡ `E1_unigram_64k` is 62,896 pieces, not 64,000.** The EM-based Unigram trainer settles
on fewer pieces than requested when the corpus does not support the full vocabulary, and
stripping line breaks removed ~1,100 `word\n` candidates it had been counting
(`docs/decisions.md`, 2026-09-05, CORRECTION for Experiment 02). The matched-size rule
(CLAUDE.md §2.5) is therefore satisfied to within 1.7% on that one pair, and the shortfall
runs in the direction that makes the English pivot *dearer*, i.e. it pushes the
`T2_unigram_raw_64k`* ratio **down**; the measured ratio is still above 1.0 with its CI
excluding it, so the verdict does not turn on it.

`results.json` stores these under `tpp_controlled` as `corpus -> "<sa_arm>/<en_arm>" ->
summary`, each summary carrying the same keys as a `tpp` entry (`value`, `n`, `unit`,
`distribution`, `mean`, `std`, `ci_low`, `ci_high`, `ci`, `n_bootstrap`, `seed`,
`n_undefined`, `source_tokens`, `pivot_tokens`).

---

## Deployed practice: results by corpus

These four tables are **deployed practice**, not a controlled comparison: every column
divides a Sanskrit arm's tokens by those of a 200k-vocabulary, general-domain English
tokenizer, so vocabulary size and training domain vary alongside language (CLAUDE.md
§2.5). They answer "what does Sanskrit cost under the tokenizers people ship?" — the
controlled question is answered by the E1 table above. Vocabulary sizes differ across
arms; `*` marks the two provisional T1/T2 families (trained on ~118k parallel-corpus
sentences, not the monolingual corpus — see the caveat below).
"SLP1" is the transliterated variant every arm has and the one the figure and this
verdict read from; "original" is the untransliterated script, available for T0/T3 only.
Fertility is reported in the last column for completeness (CLAUDE.md §2.1) — it is not
part of the verdict above and should not be read as a second measurement of the same
claim: it counts tokens per whitespace *word*, which sandhi and compounding make
short for Sanskrit, the opposite bias from TPP.

### Sāmayik test (prose, primary, n=2417)

| Arm | TPP Sa/En vs o200k (SLP1) | TPP Sa/En vs Llama-4 (SLP1) | TPP Sa/En vs o200k (original) | Fertility (SLP1) |
|---|---|---|---|---|
| `T0_o200k` (200k) | 1.835 [1.813, 1.858] | 1.802 [1.779, 1.824] | 1.90 [1.87, 1.93] | 3.15 |
| `T0_llama4` (201k) | 1.882 [1.858, 1.905] | 1.848 [1.825, 1.871] | 1.99 [1.96, 2.01] | 3.21 |
| `T0_gemma3` (262k) | 1.831 [1.809, 1.854] | 1.798 [1.776, 1.821] | 1.65 [1.63, 1.68] | 3.13 |
| `T0_gpt2` (50k) | 2.123 [2.096, 2.151] | 2.085 [2.058, 2.113] | 6.56 [6.47, 6.65] | 3.57 |
| `T3_sarvam` (68k) | 2.416 [2.386, 2.447] | 2.373 [2.342, 2.404] | 1.81 [1.78, 1.83] | 4.08 |
| `T3_sutra` (256k) | 1.930 [1.905, 1.955] | 1.895 [1.870, 1.921] | 1.76 [1.74, 1.78] | 3.26 |
| `T3_brahmic131k` (131k) | 1.873 [1.850, 1.896] | 1.839 [1.816, 1.862] | 1.90 [1.87, 1.92] | 3.22 |
| `T1_bpe_raw_32k`* (32k) | **0.984 [0.970, 0.998]** | 0.966 [0.952, 0.980] | — | 1.65 |
| `T1_bpe_raw_64k`* (64k) | **0.887 [0.875, 0.899]** | 0.871 [0.859, 0.883] | — | 1.49 |
| `T2_unigram_raw_32k`* (32k) | 1.069 [1.055, 1.085] | 1.050 [1.036, 1.066] | — | 1.80 |
| `T2_unigram_raw_64k`* (64k) | 1.002 [0.988, 1.017] | 0.984 [0.970, 0.998] | — | 1.69 |

Below 1.0 (CI excludes): `T1_bpe_raw_64k`* and, since the newline-stripped retrain,
`T1_bpe_raw_32k`*. Above 1.0 (CI excludes): every T0/T3 arm and `T2_unigram_raw_32k`*.
Straddling 1.0: `T2_unigram_raw_64k`* (1.002 [0.988, 1.017]).

### Sāmayik test_ood (prose, primary, out-of-domain, n=4047)

| Arm | TPP Sa/En vs o200k (SLP1) | TPP Sa/En vs Llama-4 (SLP1) | TPP Sa/En vs o200k (original) | Fertility (SLP1) |
|---|---|---|---|---|
| `T0_o200k` (200k) | 1.933 [1.911, 1.954] | 1.905 [1.883, 1.925] | 1.96 [1.93, 1.98] | 3.98 |
| `T0_llama4` (201k) | 1.985 [1.962, 2.005] | 1.955 [1.933, 1.976] | 2.06 [2.04, 2.09] | 4.04 |
| `T0_gemma3` (262k) | 1.949 [1.927, 1.970] | 1.920 [1.898, 1.942] | 1.68 [1.66, 1.69] | 3.98 |
| `T0_gpt2` (50k) | 2.258 [2.231, 2.281] | 2.224 [2.199, 2.248] | 6.99 [6.90, 7.07] | 4.58 |
| `T3_sarvam` (68k) | 2.556 [2.526, 2.585] | 2.518 [2.489, 2.547] | 1.80 [1.78, 1.82] | 5.17 |
| `T3_sutra` (256k) | 2.057 [2.033, 2.081] | 2.027 [2.003, 2.049] | 1.83 [1.81, 1.85] | 4.16 |
| `T3_brahmic131k` (131k) | 1.972 [1.949, 1.993] | 1.943 [1.921, 1.964] | 1.95 [1.93, 1.97] | 4.05 |
| `T1_bpe_raw_32k`* (32k) | 1.139 [1.126, 1.152] | 1.123 [1.109, 1.134] | — | 2.31 |
| `T1_bpe_raw_64k`* (64k) | 1.048 [1.036, 1.059] | 1.032 [1.020, 1.044] | — | 2.12 |
| `T2_unigram_raw_32k`* (32k) | 1.260 [1.244, 1.274] | 1.241 [1.226, 1.255] | — | 2.55 |
| `T2_unigram_raw_64k`* (64k) | 1.189 [1.174, 1.202] | 1.171 [1.157, 1.184] | — | 2.40 |

Below 1.0: none. Above 1.0 (CI excludes): all eleven available arms. Straddling: none.

### Itihāsa test (verse, secondary — meter is a confound, n=11721)

| Arm | TPP Sa/En vs o200k (SLP1) | TPP Sa/En vs Llama-4 (SLP1) | TPP Sa/En vs o200k (original) | Fertility (SLP1) |
|---|---|---|---|---|
| `T0_o200k` (200k) | 1.105 [1.100, 1.110] | 1.087 [1.082, 1.092] | 1.15 [1.15, 1.16] | 4.19 |
| `T0_llama4` (201k) | 1.124 [1.119, 1.129] | 1.105 [1.100, 1.110] | 1.24 [1.23, 1.24] | 4.23 |
| `T0_gemma3` (262k) | 1.095 [1.090, 1.100] | 1.077 [1.072, 1.082] | 1.00 [0.99, 1.00]† | 4.12 |
| `T0_gpt2` (50k) | 1.249 [1.243, 1.254] | 1.228 [1.223, 1.233] | 4.05 [4.03, 4.06] | 4.67 |
| `T3_sarvam` (68k) | 1.425 [1.419, 1.432] | 1.402 [1.396, 1.408] | 1.12 [1.11, 1.12] | 5.28 |
| `T3_sutra` (256k) | 1.169 [1.164, 1.174] | 1.150 [1.145, 1.155] | 1.09 [1.09, 1.10] | 4.33 |
| `T3_brahmic131k` (131k) | 1.121 [1.115, 1.126] | 1.102 [1.097, 1.107] | 1.15 [1.15, 1.16] | 4.25 |
| `T1_bpe_raw_32k`* (32k) | **0.513 [0.511, 0.516]** | 0.505 [0.502, 0.507] | — | 1.90 |
| `T1_bpe_raw_64k`* (64k) | **0.462 [0.460, 0.464]** | 0.454 [0.452, 0.457] | — | 1.71 |
| `T2_unigram_raw_32k`* (32k) | **0.551 [0.548, 0.553]** | 0.542 [0.539, 0.544] | — | 2.04 |
| `T2_unigram_raw_64k`* (64k) | **0.513 [0.511, 0.516]** | 0.505 [0.502, 0.507] | — | 1.90 |

† `T0_gemma3`'s original-script CI is [0.9909, 1.0000] — essentially parity, straddling
1.0 by 0.00005; rounds to 1.00 above. Below 1.0 (SLP1, CI excludes): all four T1/T2 arms.
Above 1.0 (SLP1, CI excludes): all seven T0/T3 arms. Straddling: none in SLP1 (the
`T0_gemma3` near-parity case above is in the *original*-script column only, not part of
this SLP1-based classification).

### FLORES devtest (Wikipedia, tertiary, n=1012)

| Arm | TPP Sa/En vs o200k (SLP1) | TPP Sa/En vs Llama-4 (SLP1) | TPP Sa/En vs o200k (original) | Fertility (SLP1) |
|---|---|---|---|---|
| `T0_o200k` (200k) | 2.183 [2.159, 2.207] | 2.169 [2.145, 2.192] | 2.09 [2.07, 2.11] | 3.50 |
| `T0_llama4` (201k) | 2.256 [2.230, 2.280] | 2.241 [2.216, 2.265] | 2.20 [2.18, 2.22] | 3.56 |
| `T0_gemma3` (262k) | 2.208 [2.184, 2.230] | 2.194 [2.170, 2.215] | 1.79 [1.77, 1.81] | 3.50 |
| `T0_gpt2` (50k) | 2.534 [2.502, 2.562] | 2.518 [2.486, 2.544] | 7.91 [7.81, 8.00] | 3.97 |
| `T3_sarvam` (68k) | 2.899 [2.865, 2.929] | 2.881 [2.847, 2.911] | 1.88 [1.86, 1.90] | 4.58 |
| `T3_sutra` (256k) | 2.316 [2.288, 2.342] | 2.301 [2.273, 2.327] | 1.86 [1.84, 1.88] | 3.66 |
| `T3_brahmic131k` (131k) | 2.244 [2.219, 2.269] | 2.230 [2.205, 2.254] | 2.09 [2.06, 2.11] | 3.57 |
| `T1_bpe_raw_32k`* (32k) | 1.312 [1.297, 1.327] | 1.304 [1.288, 1.318] | — | 2.07 |
| `T1_bpe_raw_64k`* (64k) | 1.220 [1.204, 1.234] | 1.212 [1.197, 1.226] | — | 1.92 |
| `T2_unigram_raw_32k`* (32k) | 1.450 [1.432, 1.467] | 1.441 [1.423, 1.458] | — | 2.29 |
| `T2_unigram_raw_64k`* (64k) | 1.366 [1.349, 1.382] | 1.357 [1.341, 1.373] | — | 2.15 |

Below 1.0: none. Above 1.0 (CI excludes): all eleven available arms. Straddling: none.

## Hindi pivot (FLORES devtest only, T0/T3 arms, same tokenizer both sides)

Sa/Hi under the *original* script is close to parity or slightly above for every arm
(1.06–1.41); under SLP1 it drops below 1.0 for every arm (0.91–0.95) — read that drop as
a transliteration artifact, not a real result. It is driven by Hindi characters that fall
outside SLP1's Sanskrit-only inventory, which inflates the Hindi token count sitting in
this ratio's *denominator* and mechanically depresses it (mechanism and exact counts in
the caveat below). The original-script column is the one to trust. Experiment 01's own
Sa/Hi finding (1.06–1.35) is likewise an original-script number — it never measured an
SLP1 Hindi pivot, so there is no earlier SLP1 result to compare this one against.

| Arm | Sa/Hi (original) | Sa/Hi (SLP1, approximate) |
|---|---|---|
| `T0_o200k` | 1.328 [1.315, 1.342] | 0.907 [0.898, 0.917] |
| `T0_llama4` | 1.325 [1.312, 1.339] | 0.919 [0.910, 0.929] |
| `T0_gemma3` | 1.353 [1.339, 1.368] | 0.914 [0.905, 0.924] |
| `T0_gpt2` | 1.060 [1.050, 1.071] | 0.922 [0.913, 0.933] |
| `T3_sarvam` | 1.405 [1.390, 1.423] | 0.946 [0.936, 0.955] |
| `T3_sutra` | 1.413 [1.398, 1.429] | 0.946 [0.937, 0.957] |
| `T3_brahmic131k` | 1.327 [1.314, 1.341] | 0.924 [0.915, 0.934] |

`T1_bpe_raw_*`/`T2_unigram_raw_*` are excluded from this table by design (config
resolution: Hindi pivot is T0/T3 arms only), and `T3_indicsuper` is absent because it is
unavailable this run (see below).

## The central figure

`tpp_by_arm.pdf` / `.png`: one row per corpus in the order above (prose first) and two
columns.

The two columns are measured against *different* English sides, so the suptitle names
neither ("Tokens per proposition (Sanskrit / English), 95% bootstrap CI") and each column
carries a header naming its own pivot.

The **right column is the controlled comparison** and the one to read first: for each
corpus, the four matched pairs (Sanskrit arm over its `E1_*` twin) as points with 95%
bootstrap-CI error bars, against the same dashed 1.0 line. Its header names what is held
constant — "Matched control: Sanskrit T1/T2 vs English E1 (same algorithm, vocab, training
corpus)". Read across the four rows, it shows three panels sitting entirely above 1.0 and
only Itihāsa (verse) below.

The **left column is deployed practice** ("Deployed practice: every arm vs English o200k
(200k, general domain)"): the SLP1-variant TPP of every available arm against `T0_o200k`,
as a point with its 95% bootstrap-CI error bar, plus a thin marker at
the same x-position for the *original*-script variant where the arm has one (T0/T3), and
the same dashed line at 1.0. Provisional (T1/T2) arms carry a `*` in their x-tick label;
the caption explains it. The two columns share no y-axis: the left one spans the T0/T3
arms' 1.8–2.9, the right one the narrow band the controlled pairs occupy, so each is
scaled to be readable rather than to be compared by eye across the page — compare the
numbers, not the marker heights.

Each left panel's y-axis is scaled from the SLP1 series alone, not from every point on the
panel: `T0_gpt2`'s original-script number is 4–8x every other arm's (its old,
Devanagari-blind vocabulary falls back to near-byte-level segmentation on raw Sanskrit —
see Experiment 01), and letting it set the axis would squeeze the sign flip this figure
exists to show into a sliver at the bottom. Any original-script marker that falls outside
that range is drawn as a triangle just inside the axis edge, pointing further off-scale
and annotated with its true value (e.g. "▲ 6.56"), rather than distorting the panel. The
caption also names any arm omitted from a panel for being unavailable this run
(built from `results.json`'s `unavailable_arms`, e.g. "`T3_indicsuper` omitted (no
candidate tokenizer could be loaded)").

## Caveats

- **T0/T3 arms are existing practice, not a controlled comparison** (CLAUDE.md §2.5):
  their vocabularies span 50k (`T0_gpt2`) to 262k (`T0_gemma3`) ids; `results.json`
  records each arm's `source_id` and `vocab_size` under `tokenizer_sources`.
- **T1/T2 arms are provisional.** They are trained on the Sanskrit sides of the Sāmayik
  and Itihāsa *training* splits only (`data/processed/manifest.json`: 118,654 raw
  sentences → 218 dropped for colliding with the evaluation exclusion list → 716 exact
  duplicates removed → **117,720 training sentences**), not the monolingual corpus (DCS,
  GRETIL, Wikipedia) planned for milestone M1; they will be retrained once that corpus is
  assembled (`docs/decisions.md`, "Provisional T1/T2 tokenizers..."). Separately,
  `UnigramTrainer` (the `T2_*` arms) is not bit-for-bit deterministic across runs on
  identical input — the vocabulary content is reproducible, the exact `tokenizer.json`
  bytes are not (`docs/decisions.md`, "`UnigramTrainer` is not bit-for-bit
  deterministic..."); the numbers above are from the `tokenizer.json` files committed to
  `outputs/tokenizers/` at the time of this run, not necessarily reproducible bit-for-bit
  from a fresh training run on the same recipe.
- **The E1 control arms are matched on what they are matched on, and no more.** Same two
  algorithms, same 32k/64k vocabulary sizes, same trainer settings, same sentences —
  trained on the English side of the very splits whose Sanskrit side trained T1/T2
  (`data/processed/manifest_en.json`: 118,654 raw sentences → 680 dropped for colliding
  with `data/exclusion_hashes_en.txt` → 1,847 exact duplicates removed → **116,127
  training sentences**, against the Sanskrit corpus's 117,720). The English corpus is not
  transliterated; its exclusion list is the English side of the same six evaluation
  splits, hashed with `sentence_hash_en` (27,686 unique hashes,
  `data/exclusion_hashes_en.txt`, committed). What is *not* matched: English and Sanskrit
  differ in what a "sentence" of ~116k of them contains, and both corpora are small and
  single-domain by tokenizer standards, so this control equalises domain fit, not corpus
  size or diversity. `E1_unigram_*` inherits the same Unigram non-determinism caveat as
  `T2_*`; `results.json` records the sha256 of every E1 `tokenizer.json` under
  `tokenizer_sources`, as it does for T1/T2.
- **The English side of a TPP ratio is a translation.** Sāmayik and Itihāsa pair Sanskrit
  with an English rendering, and a translator's verbosity lands in the denominator — most
  visibly for Itihāsa, whose English is a 19th-century verse translation. TPP is the least
  bad available proxy for "cost per proposition", not a measurement of propositions.
- **`T3_indicsuper` is unavailable this run.** All three candidate repository ids
  (`krutrim-ai-labs/IndicSuperTokenizer`, `ai4bharat/IndicSuperTokenizer`,
  `ai4bharat/indic-super-tokenizer`) fail to resolve on the Hugging Face Hub — not
  publicly released as of 2026-09-03 (`docs/decisions.md`). It is omitted from every
  table and the figure; `results.json`'s `unavailable_arms` records the three failures
  verbatim.
- **Hindi SLP1 is not trustworthy — mechanism and direction.** SLP1 encodes the Sanskrit
  phoneme inventory; Hindi carries characters outside it. Of the 1012 FLORES Hindi
  sentences, 513 contain a nukta consonant (क़ ज़ ड़ ढ़ फ़, and the other precomposed nukta
  letters), which `sanscript` renders as a literal ASCII `'0'` rather than a phoneme, and
  256 of the resulting SLP1 strings still contain raw, unconverted Devanagari (signs such
  as candra-o, `ॉ`, that fall outside the scheme) — Experiment 01's `slp1_coverage`
  measured these same two counts on these same FLORES `hin_Deva` sentences. Both effects
  inflate the Hindi token count, which sits in the *denominator* of the Sa/Hi ratio, so
  they mechanically depress the SLP1 numbers below their true value (0.91–0.95 above);
  this is not evidence that Sanskrit costs fewer tokens than Hindi under these
  tokenizers. The **original**-script Sa/Hi column (1.06–1.41) is the faithful
  measurement and the one to read. Experiment 01's own Sa/Hi finding (1.06–1.35) is
  likewise an original-script number — it never measured an SLP1 Hindi pivot.
- **Fertility is reported, never headlined** (CLAUDE.md §2.1, §7): it counts tokens per
  whitespace word, which sandhi and compounding make artificially short for Sanskrit —
  the opposite bias from TPP — so it appears only in each table's last column and this
  one-line note, never in the summary above.
- **Every stored TPP summary carries its bootstrap settings.** Each `results.json["tpp"][corpus][arm][variant][pivot]`
  (and the equivalent `tpp_controlled[corpus][pair]` and `tpp_hindi` entries) is
  `{value, n, unit, distribution, mean, std,
  ci_low, ci_high, ci, n_bootstrap, seed, n_undefined, source_tokens, pivot_tokens}` —
  `ci` is the nominal confidence level the bootstrap targeted (`0.95` throughout this
  run, from `config.yaml`'s `ci` key), so a reader of `results.json` alone can tell what
  `ci_low`/`ci_high` are a CI *of* without cross-referencing `config.yaml`.
- **No evaluation leakage detected, on either side.** `exclusion_check` in
  `results.json`: every Sanskrit sentence used by this experiment (2417 + 4047 + 11721 +
  1012 = 19,197 total) hashes to an entry already in `data/exclusion_hashes.txt`;
  `n_missing` is 0 for all four corpora. **`exclusion_check_en`** does the same for the
  English side of those same 19,197 pairs against `data/exclusion_hashes_en.txt`, hashed
  with `sentence_hash_en`: `n_missing` is 0 for all four corpora there too. That second
  check is the one that matters for the control arms — the English list is what kept the
  `E1_*` arms away from this evaluation text, and a missed hash there would mean an E1 arm
  had trained on a sentence it is now being evaluated on. At training time the same list
  dropped 680 of 118,654 English training sentences for colliding with it (675 Sāmayik,
  5 Itihāsa) before any tokenizer saw them — the same filter-then-assert path the Sanskrit
  corpus uses, with the English hash function.
- **`results.json` is strict JSON.** A handful of per-pair TPP ratios and, in principle,
  a bootstrap CI can be undefined (`nan`) when a pivot sentence yields zero tokens;
  `sanitize_json` replaces every `nan`/`inf` float with `null` before writing, and the
  file is written with `json.dump(..., allow_nan=False)` so no non-standard `NaN`/
  `Infinity` token can ever land in it. On this run no such undefined case occurred
  (`n_undefined` is 0 throughout), but the sanitiser runs regardless.
