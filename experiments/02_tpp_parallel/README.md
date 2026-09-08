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

> **Re-run 2026-09-08 at commit `40fd9c8`, clean tree.** This run answers five reviewer
> objections with measurements rather than caveats, and one of them changes a headline.
> **(1) A byte-matched English control.** The pair-matched `E1_*` arms saw 16,554,871 bytes
> of English against the Sanskrit corpus's 11,209,356 — the same sentences, 48% more text —
> so six `E1_*_bm` arms were trained on a subsample of that English corpus cut to the
> Sanskrit byte count. Every controlled ratio moves by at most 0.025 and none crosses 1.0
> that did not already: the objection is real and its size is small. **(2) 128k arms on both
> sides**, and here the verdict does move: `T1_bpe_raw_128k`*/`E1_bpe_128k` reads **0.983
> [0.971, 0.997]** on Sāmayik test, the first matched pair below 1.0 on prose with its CI
> excluding it (0.978 [0.965, 0.991] against the byte-matched control) — while the same pair
> stays above 1.0 on out-of-domain prose (1.025 [1.013, 1.037]) and on FLORES. **(3) The
> byte-level arm `T7_byt5`** is now measured as a tokenizer-free reference, including as a
> controlled pair against itself (a ratio of UTF-8 bytes). **(4) `side_decomposition`**
> factorises every ratio into a character ratio and a density ratio, which settles where the
> verse result lives: Itihāsa's Sanskrit side is 0.596 of its English side in *characters*,
> and its density ratio (1.00–1.10) is the same as prose's. **(5) A block bootstrap**
> (`block_length: 50`) runs beside the i.i.d. one; on Itihāsa it widens every controlled
> interval by 2.4–2.6x, and on Sāmayik test_ood it turns `T1_bpe_raw_128k`*/`E1_bpe_128k`
> from 1.025 [1.013, 1.037] into [0.999, 1.050], which straddles 1.0.
> **No pre-existing number moved:** all 17,776 leaves of the previous snapshot's
> `results.json` reappear at the same path with the same value (numeric to within 1e-9,
> strings and booleans exactly), with only `git_commit`, `git_dirty`, `timestamp` and the
> `<repo>`-scrubbed tokenizer paths allowed to differ; the only new top-level key is
> `side_decomposition`, the only new config keys are `block_length` and
> `length_strata_skip_arms`, and the only keys added inside an existing summary are
> `ci_low_block`, `ci_high_block`, `block_length` and `n_blocks`. `config["sanskrit_arms"]`
> gains three arms in place (the old list is still a subsequence of the new one).

> **Re-run 2026-09-07 at commit `5820ee9`, clean tree.** The length strata are now
> measured under **two** stratifications, not one: `tpp_by_length` (bins on the English
> side's word count, unchanged) and the new `tpp_by_length_sa` / `length_bin_edges_sa`
> (bins on the Sanskrit side's, `[1, 6, 11, 16, 26]`). Every stratum records which side it
> was binned on (`bin_on`), and `tpp_by_length.pdf` / `.png` becomes two rows of
> per-corpus panels — one row per bin variable — each panel scaled to its own non-sparse
> bins. **No pre-existing number moved:** every one of the 11,469 leaves of the previous
> snapshot's `results.json` reappears at the same path with the same value (numeric to
> within 1e-9, strings and booleans exactly), with only `git_commit`, `git_dirty` and
> `timestamp` allowed to differ; the only new top-level keys are `tpp_by_length_sa` and
> `length_bin_edges_sa`, the only new config key is `length_bin_edges_sa`, and the only
> key added inside an existing block is `bin_on` on each `tpp_by_length` stratum. What
> did change is the *reading*: the within-prose length gradient reverses sign under the
> mirror stratification, so it cannot be read as a density effect — see "TPP by sentence
> length" below and `docs/decisions.md`, 2026-09-07.

> **Re-run 2026-09-07 at commit `70d9219`, clean tree.** This run adds two blocks to
> `results.json` and one figure, and changes nothing that was already there: `renyi` /
> `renyi_english` (Rényi efficiency at α ∈ {2.5, 3}) and `tpp_by_length` /
> `length_bin_edges` (TPP stratified by English sentence length), drawn as
> `tpp_by_length.pdf` / `.png`. **No pre-existing number moved.** Every one of the 3,452
> leaves of the previous snapshot's `results.json` reappears at the same path with the
> same value — numeric leaves equal to within 1e-9, strings and booleans exactly — with
> only `git_commit`, `git_dirty` and `timestamp` allowed to differ, and the only new
> top-level keys are the four named above (`config` likewise gains only `renyi_alphas`,
> `length_bin_edges` and `length_sparse_below`). The two sections at the end of this page
> report the new numbers; every table above them is unchanged.

> **Re-run 2026-09-05 at commit `b7302a8`, clean tree.** Every trained arm was retrained
> after the trainers stopped letting line breaks reach the pre-tokenizer, so every number
> on this page moved slightly and four of them are new
> (`docs/decisions.md`, 2026-09-05, "Trainers strip newlines" and the CORRECTION entry for
> this experiment). **No conclusion changed in the controlled comparison.** Two rows of the
> *deployed-practice* tables did: `T1_bpe_raw_32k`* against `T0_o200k` and against
> `T0_llama4` on Sāmayik test now sit below 1.0 with their CIs excluding it, where before
> they straddled it. `E1_unigram_64k` now trains to **62,896** pieces rather than 64,000,
> which is recorded below and in the decision log.

**Runtime:** 4m27s wall-clock on this machine with every tokenizer cache, corpus jsonl and
trained `tokenizer.json` already warm — 3m33s before this wave, 3m19s before the mirror
stratification, and 2m09s for the 2026-09-05 run, which measured the same TPP tables
without any of the additions. This wave's extra minute is the ten new arms (three more
Sanskrit arms x 4 corpora x fertility/compression/Rényi/TPP, and nine more controlled pairs
x 4 corpora), the `side_decomposition` pass (one extra tokenization per pair per corpus),
and the second bootstrap, which doubles the resampling cost of every TPP summary. The
length strata did **not** grow with the new arms: `length_strata_skip_arms` keeps the 128k
arms and `T7_byt5` out of them (see "TPP by sentence length"). `english_pivots` stays at the
two deployed arms. Training the ten new arms first (a one-off,
`uv run python experiments/02_tpp_parallel/train_tokenizers.py`) took 32s wall-clock,
including building the byte-matched English corpus; the twelve arms this config had already
trained were skipped untouched, and the sha256 of every one of the 26 pre-existing
`tokenizer.json` files was verified unchanged afterwards.

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

**The 2026-09-08 wave: the control survives a byte-matched denominator, and fails at 128k.**
Two of the five additions bear on the verdict. Cutting the English control's training text
to the Sanskrit corpus's byte count — the reviewer's objection, since the pair-matched
control had seen 48% more bytes — moves every controlled ratio by at most 0.025 and flips
none of them: on Sāmayik test the four original pairs read 1.081–1.133 against the
byte-matched control where they read 1.084–1.142 against the pair-matched one. Raising the
vocabulary to 128k does flip one: `T1_bpe_raw_128k`*/`E1_bpe_128k` reads **0.983 [0.971,
0.997]** on Sāmayik test (0.978 [0.965, 0.991] byte-matched), the first matched pair below
1.0 on prose with its interval excluding it. It is one corpus and one algorithm — the same
pair reads 1.025 [1.013, 1.037] on out-of-domain prose and 1.116 [1.103, 1.127] on FLORES,
and under the block bootstrap the out-of-domain number becomes [0.999, 1.050] — so what the
128k row establishes is that the prose result is **not** stable in vocabulary size, not that
Sanskrit wins at 128k. Read "raw subword training does not bring TPP below English on prose"
as holding at 32k and 64k, and as contested at 128k on in-domain prose only.

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
`n_undefined`, `source_tokens`, `pivot_tokens`, and since 2026-09-08 `ci_low_block`,
`ci_high_block`, `block_length`, `n_blocks`).

**This table is the 32k/64k pair-matched half of the comparison.** The 2026-09-08 wave added
nine more controlled pairs to the same block: the same four Sanskrit arms against the
byte-matched control (**Byte-matched control**, below), both algorithms at 128k against both
controls (**128k**), and `T7_byt5` against itself (**Byte-level reference**). Where each of
these ratios comes from — a shorter Sanskrit side or a denser tokenizer — is in **Where the
verse ratio lives**, and the second, block-resampled interval for every one of them is in
**Block bootstrap**.

---

## Byte-matched control (`E1_*_bm`)

The pair-matched `E1_*` arms are trained on the English side of the same sentences the
Sanskrit arms are trained on — and on **48% more bytes**: `data/processed/tok_train_en.txt`
is 16,554,871 bytes (16,512,997 characters, 116,127 lines) against
`tok_train_slp1.txt`'s 11,209,356 bytes (11,201,170 characters, 117,720 lines; SLP1 is
ASCII, so its bytes and characters coincide). A reviewer can therefore read the control's
cheaper tokenization as the extra text rather than as the language. The `E1_*_bm` arms cut
that reading off: they are the same trainers at the same vocabulary sizes on
`data/processed/tok_train_en_bm.txt`, a deterministic subsample of the *same* English
corpus — `random.Random(0)` shuffle, then the prefix whose written size first reaches the
Sanskrit corpus's byte count — which is **78,624 lines, 11,209,371 bytes (1.0000013x the
Sanskrit corpus), 11,180,926 characters (0.998x)**, i.e. 67.7% of the English lines
(`data/processed/manifest_en_bm.json`). Neither control is *the* control; they bracket it,
one matching sentences and one matching bytes.

| Matched pair (Sanskrit / English) | Control | Sāmayik test | Sāmayik test_ood | Itihāsa test | FLORES devtest |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | pair-matched | 1.084 [1.070, 1.098] | 1.086 [1.072, 1.098] | 0.645 [0.642, 0.649] | 1.143 [1.131, 1.154] |
| `T1_bpe_raw_32k`* / `E1_bpe_32k_bm` | byte-matched | 1.081 [1.067, 1.096] | 1.085 [1.072, 1.097] | 0.645 [0.641, 0.648] | 1.139 [1.127, 1.150] |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | pair-matched | 1.035 [1.021, 1.049] | 1.061 [1.048, 1.073] | 0.598 [0.594, 0.601] | 1.144 [1.131, 1.155] |
| `T1_bpe_raw_64k`* / `E1_bpe_64k_bm` | byte-matched | 1.030 [1.017, 1.043] | 1.055 [1.042, 1.066] | 0.597 [0.593, 0.600] | 1.131 [1.119, 1.142] |
| `T1_bpe_raw_128k`* / `E1_bpe_128k` | pair-matched | 0.983 [0.971, 0.997] | 1.025 [1.013, 1.037] | 0.557 [0.554, 0.560] | 1.116 [1.103, 1.127] |
| `T1_bpe_raw_128k`* / `E1_bpe_128k_bm` | byte-matched | 0.978 [0.965, 0.991] | 1.015 [1.002, 1.026] | 0.555 [0.552, 0.559] | 1.103 [1.091, 1.115] |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | pair-matched | 1.142 [1.128, 1.157] | 1.163 [1.148, 1.176] | 0.658 [0.655, 0.661] | 1.206 [1.194, 1.219] |
| `T2_unigram_raw_32k`* / `E1_unigram_32k_bm` | byte-matched | 1.133 [1.118, 1.148] | 1.159 [1.145, 1.173] | 0.651 [0.647, 0.654] | 1.209 [1.196, 1.222] |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | pair-matched | 1.107 [1.092, 1.121] | 1.156 [1.142, 1.169] | 0.623 [0.619, 0.626] | 1.219 [1.206, 1.232] |
| `T2_unigram_raw_64k`* / `E1_unigram_64k_bm` | byte-matched | 1.088 [1.073, 1.102] | 1.138 [1.124, 1.150] | 0.613 [0.610, 0.617] | 1.194 [1.181, 1.206] |
| `T2_unigram_raw_128k`* / `E1_unigram_128k` | pair-matched | 1.051 [1.037, 1.065] | 1.084 [1.071, 1.096] | 0.586 [0.584, 0.590] | 1.145 [1.132, 1.157] |
| `T2_unigram_raw_128k`* / `E1_unigram_128k_bm` | byte-matched | 1.033 [1.019, 1.047] | 1.067 [1.054, 1.079] | 0.578 [0.575, 0.581] | 1.122 [1.110, 1.133] |

**Did any controlled ratio cross 1.0 under the byte-matched control? No — not one.** Every
pair that was above 1.0 on prose stays above it with its CI excluding it, every Itihāsa pair
stays below, and the single pair that is below 1.0 on Sāmayik test (`T1_bpe_raw_128k`*, see
the next section) is below it under both controls. **How far did the ratios move?** By 0.0006
to 0.0249 in absolute TPP over the 24 pairs, median 0.0080, and downward in 23 of the 24
(the byte-matched control spends slightly *more* tokens on the same English text — a
vocabulary trained on two thirds of the corpus is a little worse at it — which lifts the
denominator and lowers the ratio). The largest single move is
`T2_unigram_raw_64k`*/`E1_unigram_64k` on FLORES, 1.2187 -> 1.1938; the smallest is
`T1_bpe_raw_32k`* on Itihāsa, 0.6454 -> 0.6448; the one upward move is
`T2_unigram_raw_32k`* on FLORES, 1.2060 -> 1.2090. The objection is real, and on this corpus
pair it is worth under one percent of TPP.

---

## 128k: the same comparison at a third vocabulary size

Every family is now trained at 32k, 64k and 128k, so a controlled result can be checked
against the vocabulary size it was measured at rather than asserted to be independent of it.
Rows are corpora; columns are the six trained pairs; `E1` is the pair-matched control and
`E1_bm` the byte-matched one.

| Corpus | Control | BPE 32k | BPE 64k | BPE 128k | Unigram 32k | Unigram 64k | Unigram 128k |
|---|---|---|---|---|---|---|---|
| Sāmayik test | E1 | 1.084 [1.070, 1.098] | 1.035 [1.021, 1.049] | 0.983 [0.971, 0.997] | 1.142 [1.128, 1.157] | 1.107 [1.092, 1.121] | 1.051 [1.037, 1.065] |
| Sāmayik test | E1_bm | 1.081 [1.067, 1.096] | 1.030 [1.017, 1.043] | 0.978 [0.965, 0.991] | 1.133 [1.118, 1.148] | 1.088 [1.073, 1.102] | 1.033 [1.019, 1.047] |
| Sāmayik test_ood | E1 | 1.086 [1.072, 1.098] | 1.061 [1.048, 1.073] | 1.025 [1.013, 1.037] | 1.163 [1.148, 1.176] | 1.156 [1.142, 1.169] | 1.084 [1.071, 1.096] |
| Sāmayik test_ood | E1_bm | 1.085 [1.072, 1.097] | 1.055 [1.042, 1.066] | 1.015 [1.002, 1.026] | 1.159 [1.145, 1.173] | 1.138 [1.124, 1.150] | 1.067 [1.054, 1.079] |
| Itihāsa test | E1 | 0.645 [0.642, 0.649] | 0.598 [0.594, 0.601] | 0.557 [0.554, 0.560] | 0.658 [0.655, 0.661] | 0.623 [0.619, 0.626] | 0.586 [0.584, 0.590] |
| Itihāsa test | E1_bm | 0.645 [0.641, 0.648] | 0.597 [0.593, 0.600] | 0.555 [0.552, 0.559] | 0.651 [0.647, 0.654] | 0.613 [0.610, 0.617] | 0.578 [0.575, 0.581] |
| FLORES devtest | E1 | 1.143 [1.131, 1.154] | 1.144 [1.131, 1.155] | 1.116 [1.103, 1.127] | 1.206 [1.194, 1.219] | 1.219 [1.206, 1.232] | 1.145 [1.132, 1.157] |
| FLORES devtest | E1_bm | 1.139 [1.127, 1.150] | 1.131 [1.119, 1.142] | 1.103 [1.091, 1.115] | 1.209 [1.196, 1.222] | 1.194 [1.181, 1.206] | 1.122 [1.110, 1.133] |

**Read the BPE columns and the Unigram columns differently, because only BPE is
size-matched at 128k.** `T1_bpe_raw_128k`, `E1_bpe_128k` and `E1_bpe_128k_bm` all train to
exactly 128,000 pieces. The Unigram trainer does not: `T2_unigram_raw_128k` reaches 128,000
but `E1_unigram_128k` settles at **62,896** (the same ceiling `E1_unigram_64k` hit) and
`E1_unigram_128k_bm` at **50,659** (the same as `E1_unigram_64k_bm`), so the "128k" Unigram
pairs are really 128k Sanskrit against a ~63k/~51k English control. That shortfall makes the
English side *dearer* and therefore pushes those ratios **down**, so the Unigram 128k
numbers (1.051 / 1.033 on Sāmayik test) are, if anything, generous to Sanskrit and still sit
above 1.0.

**The BPE progression is monotone and it crosses.** On Sāmayik test the pair-matched BPE
ratio falls 1.084 -> 1.035 -> 0.983 as the vocabulary quadruples, and the 128k interval
[0.971, 0.997] excludes 1.0; the byte-matched twin does the same, 1.081 -> 1.030 -> 0.978.
On out-of-domain prose the same progression stops just above parity (1.086 -> 1.061 ->
1.025), on Itihāsa it deepens an already-large gap (0.645 -> 0.598 -> 0.557), and on FLORES
it stays well above 1.0 (1.143 -> 1.144 -> 1.116). So the prose verdict is **not** stable in
vocabulary size: it holds at 32k and 64k, and at 128k it holds only out of domain.

---

## Byte-level reference (`T7_byt5`)

`T7_byt5` is UTF-8 itself: 256 ids, no vocabulary, no training corpus, nothing that can be
unavailable. Measured on both sides of the same aligned pairs it is not a tokenizer
comparison at all but the **ratio of the two sides' bytes** — the amount of text each
language spends on the same propositions, before any tokenizer has an opinion. It is drawn
as a dotted reference line on each controlled panel and is the line every matched pair should
be read against: a pair above it is spending more tokens than the raw text ratio alone would
buy.

| Corpus | TPP (bytes Sa / bytes En) | 95% block CI | Sanskrit SLP1 bytes | English bytes |
|---|---|---|---|---|
| Sāmayik test | 1.027 [1.016, 1.039] | [1.015, 1.037] | 176,222 | 171,585 |
| Sāmayik test_ood | 1.016 [1.005, 1.027] | [0.996, 1.037] | 442,971 | 435,846 |
| Itihāsa test | 0.595 [0.592, 0.598] | [0.588, 0.603] | 1,236,059 | 2,077,934 |
| FLORES devtest | 1.059 [1.049, 1.067] | [1.039, 1.078] | 139,831 | 132,096 |

The Sanskrit side is scored in SLP1 (ASCII, one byte per character); the controlled TPP in
the first column is *identical* to `bytes_sa / bytes_en` from `side_decomposition`, which is
the check the verification runs. Two readings. On the three prose/Wikipedia corpora the byte
ratio is just above parity (1.016–1.059), and every 32k/64k matched pair sits **above** it —
subword tokenization costs Sanskrit more than the text ratio does. On Itihāsa the byte ratio
is 0.595 and the matched pairs sit *at or just above* it (0.555–0.658): essentially the whole
verse result is already present in the raw byte counts, before tokenization.

---

## Where the verse ratio lives: the side decomposition

`results.json["side_decomposition"]` factorises every ratio exactly:
`tpp = char_ratio x density_ratio`, where `char_ratio = chars_sa / chars_en` (how much text
each side spends on the same propositions) and `density_ratio = tokens_per_char_sa /
tokens_per_char_en` (how expensively each tokenizer charges for a character of its own
side). The identity holds to within 1e-9 on all 108 entries. Sanskrit is read in SLP1, and
the entry records the variant.

| Matched pair | Sāmayik char_ratio | Sāmayik density_ratio | Sāmayik TPP | Itihāsa char_ratio | Itihāsa density_ratio | Itihāsa TPP |
|---|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 1.028 | 1.054 | 1.084 | 0.596 | 1.083 | 0.645 |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 1.028 | 1.006 | 1.035 | 0.596 | 1.003 | 0.598 |
| `T1_bpe_raw_32k`* / `E1_bpe_32k_bm` | 1.028 | 1.051 | 1.081 | 0.596 | 1.082 | 0.645 |
| `T1_bpe_raw_64k`* / `E1_bpe_64k_bm` | 1.028 | 1.002 | 1.030 | 0.596 | 1.001 | 0.597 |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 1.028 | 1.111 | 1.142 | 0.596 | 1.104 | 0.658 |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 1.028 | 1.076 | 1.107 | 0.596 | 1.045 | 0.623 |
| `T2_unigram_raw_32k`* / `E1_unigram_32k_bm` | 1.028 | 1.102 | 1.133 | 0.596 | 1.092 | 0.651 |
| `T2_unigram_raw_64k`* / `E1_unigram_64k_bm` | 1.028 | 1.058 | 1.088 | 0.596 | 1.029 | 0.613 |

**The verse crossing sits entirely in the character ratio, not the density ratio.** Itihāsa's
Sanskrit side is **0.596** of its English side in characters (1,235,972 against 2,073,532),
while Sāmayik test's is **1.028** (175,772 against 170,943) — a factor of 1.725 between the
two corpora. The density ratios, by contrast, are the same in both: 1.00–1.11 on Sāmayik and
1.00–1.10 on Itihāsa, pair for pair within 0.032. **So Itihāsa's sub-1.0 TPP is a statement
about how much text Dutt's 19th-century verse translation spends, not about Sanskrit
tokenizing more densely there than in prose** — which is what CLAUDE.md §2.7 warns about
verse, now measured rather than suspected. The same arithmetic reads the prose result the
other way: on Sāmayik test the Sanskrit side is 2.8% longer in characters *and* 0.2–11%
dearer per character, and both factors push the ratio above 1.0.

---

## Block bootstrap (`block_length: 50`)

The i.i.d. bootstrap resamples single sentence pairs, which assumes they are exchangeable.
They are not: Itihāsa test is consecutive verses of one epic, FLORES devtest is consecutive
sentences of the documents it was drawn from, and Sāmayik's splits are as released. Every
TPP summary therefore carries a second interval from a **non-overlapping block bootstrap** —
50 consecutive pairs per block, the final short block kept, blocks drawn with replacement and
concatenated to the corpus's own length — under `ci_low_block` / `ci_high_block`, with
`block_length` and `n_blocks` beside them. `ci_low` / `ci_high` remain the i.i.d. interval,
so nothing already reported has moved.

| Corpus | pairs | blocks | mean i.i.d. width | mean block width | ratio (min–max) |
|---|---|---|---|---|---|
| Sāmayik test | 2,417 | 49 | 0.0277 | 0.0268 | 0.90–1.01 |
| Sāmayik test_ood | 4,047 | 81 | 0.0254 | 0.0562 | 1.79–2.37 |
| Itihāsa test | 11,721 | 235 | 0.0064 | 0.0158 | 2.40–2.59 |
| FLORES devtest | 1,012 | 21 | 0.0237 | 0.0577 | 2.11–2.64 |

**The dependence is real everywhere except Sāmayik test.** On Itihāsa every controlled
interval widens by 2.40–2.59x, on FLORES by 2.11–2.64x and on Sāmayik test_ood by
1.79–2.37x; on Sāmayik test the block intervals are 0.90–1.01x the i.i.d. ones, i.e. no
wider and in eleven of thirteen cases marginally narrower, which is what a corpus with no
usable local correlation looks like at 49 blocks (a narrower block interval is noise in the
estimate of the variance, not evidence of less variance, and it is reported rather than
hidden). Only one verdict changes: `T1_bpe_raw_128k`*/`E1_bpe_128k` on Sāmayik test_ood is
1.025 [1.013, 1.037] i.i.d. and [0.999, 1.050] under blocks, so it straddles 1.0 there. Every
Itihāsa pair still excludes 1.0 by a wide margin under blocks (highest block bound 0.667),
and the Sāmayik test 128k BPE crossing survives ([0.971, 0.997]).

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
| `T1_bpe_raw_128k`* (128k) | **0.818 [0.807, 0.830]** | 0.804 [0.792, 0.815] | — | 1.37 |
| `T2_unigram_raw_128k`* (128k) | **0.952 [0.939, 0.966]** | 0.935 [0.921, 0.949] | — | 1.60 |
| `T7_byt5` (256) | 4.480 [4.424, 4.537] | 4.399 [4.344, 4.458] | 10.82 [10.67, 10.96] | 6.69 |

Below 1.0 (CI excludes): `T1_bpe_raw_64k`*, `T1_bpe_raw_32k`* (since the newline-stripped
retrain) and, added by this wave, `T1_bpe_raw_128k`* and `T2_unigram_raw_128k`*. Above 1.0
(CI excludes): every T0/T3 arm, `T2_unigram_raw_32k`* and `T7_byt5` (4.480 — a byte
vocabulary is the most expensive thing on the page, which is what makes it a floor and not
a proposal). Straddling 1.0: `T2_unigram_raw_64k`* (1.002 [0.988, 1.017]).

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
| `T1_bpe_raw_128k`* (128k) | **0.977 [0.966, 0.988]** | 0.963 [0.951, 0.973] | — | 1.98 |
| `T2_unigram_raw_128k`* (128k) | 1.115 [1.101, 1.127] | 1.098 [1.085, 1.110] | — | 2.26 |
| `T7_byt5` (256) | 4.576 [4.522, 4.627] | 4.508 [4.455, 4.557] | 11.34 [11.21, 11.47] | 8.34 |

Below 1.0 (CI excludes): `T1_bpe_raw_128k`* (0.977 [0.966, 0.988]), the only arm below
parity on this corpus and new in this wave — deployed practice, not a control. Above 1.0
(CI excludes): every other available arm, `T7_byt5` included. Straddling: none.

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
| `T1_bpe_raw_128k`* (128k) | **0.425 [0.423, 0.427]** | 0.418 [0.416, 0.420] | — | 1.57 |
| `T2_unigram_raw_128k`* (128k) | **0.484 [0.481, 0.486]** | 0.476 [0.473, 0.478] | — | 1.79 |
| `T7_byt5` (256) | 2.549 [2.538, 2.561] | 2.507 [2.496, 2.518] | 6.57 [6.54, 6.60] | 8.54 |

† `T0_gemma3`'s original-script CI is [0.9909, 1.0000] — essentially parity, straddling
1.0 by 0.00005; rounds to 1.00 above. Below 1.0 (SLP1, CI excludes): all six T1/T2 arms.
Above 1.0 (SLP1, CI excludes): all seven T0/T3 arms and `T7_byt5`. Straddling: none in SLP1 (the
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
| `T1_bpe_raw_128k`* (128k) | 1.145 [1.130, 1.158] | 1.137 [1.123, 1.150] | — | 1.80 |
| `T2_unigram_raw_128k`* (128k) | 1.283 [1.268, 1.298] | 1.275 [1.259, 1.290] | — | 2.02 |
| `T7_byt5` (256) | 5.203 [5.142, 5.259] | 5.170 [5.110, 5.226] | 12.91 [12.75, 13.05] | 7.29 |

Below 1.0: none. Above 1.0 (CI excludes): all fourteen available arms, `T7_byt5` included.
Straddling: none.

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

The **right column is the controlled comparison** and the one to read first: one x position
per Sanskrit arm (six of them, the two algorithms at three vocabulary sizes), carrying two
points with 95% bootstrap-CI error bars — **filled** against the pair-matched `E1_*` control,
**hollow** against the byte-matched `E1_*_bm` twin — plus a dotted **byte reference line** at
that corpus's `T7_byt5` ratio, against the same dashed 1.0 line. A pair above the dotted line
is spending more tokens than the two sides' raw byte ratio alone would buy. The legend on the
first panel names the three series; it is not repeated on the others. Its header names what is held
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

## Rényi efficiency (secondary intrinsic; can be gamed)

Rényi efficiency (Zouhar, Meister, Gastaldi, Du, Vieira, Sachan & Cotterell, "Tokenization
and the Noiseless Channel", ACL 2023) scores a tokenizer by how evenly its token-unigram
distribution over a text uses the support it actually touches, at α > 1 so the head of the
distribution weighs more than Shannon entropy would. `results.json`'s `renyi` records it
per corpus x arm x script variant at α ∈ {2.5, 3} (`renyi_alphas` in `config.yaml`).

**Normalisation: the observed support.** The value divides the Rényi entropy by
`log2(K)`, where `K` is the number of *distinct types that actually occur* in that text —
the convention of the authors' own `tokenization-scorer`, whose `get_prob_distribution`
sets `vocab_size = len(words_freqs)`. Normalising by the nominal vocabulary instead would
charge an arm for types it never emits, which is precisely the quantity Cognetta, Zouhar,
Moon & Okazaki ("Two Counterexamples to Tokenization and the Noiseless Channel",
LREC-COLING 2024) show can be manipulated; that variant is still stored alongside, as
`efficiency_nominal`, because this project's trained arms are compared at matched
vocabulary sizes.

**This is not a headline and it cannot rank arms on its own.** Cognetta et al. construct
tokenizers whose Rényi efficiency rises arbitrarily while the tokenization of the text —
and downstream performance — is unchanged or worse. The verdicts on this page rest on TPP,
and the project's headline metrics remain TPP and BPC (CLAUDE.md §2.1, §7).

Rows are the `sanskrit_arms` of `config.yaml` in order; `*` marks the provisional T1/T2
arms. Each row is read from the `original` script variant where the arm has one (T0/T3)
and from `slp1` otherwise (T1/T2 have no other), so the **variant column is part of the
row**: a T1 number and a T0 number are not measured on the same string, and the two
families are not comparable down a column. `T3_indicsuper` is unavailable this run and is
omitted here as everywhere else.

| Arm | Variant | Sāmayik test α=2.5 | Sāmayik test α=3.0 | Sāmayik test_ood α=2.5 | Sāmayik test_ood α=3.0 | Itihāsa test α=2.5 | Itihāsa test α=3.0 | FLORES devtest α=2.5 | FLORES devtest α=3.0 |
|---|---|---|---|---|---|---|---|---|---|
| `T0_o200k` | original | 0.581 | 0.556 | 0.589 | 0.568 | 0.627 | 0.605 | 0.604 | 0.578 |
| `T0_llama4` | original | 0.575 | 0.549 | 0.589 | 0.569 | 0.630 | 0.609 | 0.615 | 0.591 |
| `T0_gemma3` | original | 0.568 | 0.536 | 0.531 | 0.500 | 0.561 | 0.530 | 0.527 | 0.492 |
| `T0_gpt2` | original | 0.270 | 0.252 | 0.282 | 0.261 | 0.433 | 0.403 | 0.318 | 0.295 |
| `T3_sarvam` | original | 0.570 | 0.541 | 0.538 | 0.510 | 0.535 | 0.506 | 0.532 | 0.499 |
| `T3_sutra` | original | 0.580 | 0.549 | 0.559 | 0.530 | 0.592 | 0.564 | 0.544 | 0.510 |
| `T3_brahmic131k` | original | 0.580 | 0.555 | 0.589 | 0.568 | 0.626 | 0.605 | 0.604 | 0.578 |
| `T1_bpe_raw_32k`* | slp1 | 0.615 | 0.564 | 0.609 | 0.562 | 0.689 | 0.646 | 0.725 | 0.692 |
| `T1_bpe_raw_64k`* | slp1 | 0.589 | 0.539 | 0.604 | 0.557 | 0.655 | 0.609 | 0.715 | 0.679 |
| `T2_unigram_raw_32k`* | slp1 | 0.493 | 0.456 | 0.505 | 0.474 | 0.491 | 0.457 | 0.528 | 0.492 |
| `T2_unigram_raw_64k`* | slp1 | 0.485 | 0.449 | 0.493 | 0.463 | 0.463 | 0.430 | 0.525 | 0.491 |
| `T1_bpe_raw_128k`* | slp1 | 0.572 | 0.522 | 0.606 | 0.559 | 0.631 | 0.585 | 0.707 | 0.668 |
| `T2_unigram_raw_128k`* | slp1 | 0.478 | 0.443 | 0.487 | 0.458 | 0.442 | 0.410 | 0.526 | 0.492 |
| `T7_byt5` | original | 0.328 | 0.311 | 0.318 | 0.302 | 0.353 | 0.335 | 0.316 | 0.300 |
| `T7_byt5` | slp1 | 0.562 | 0.535 | 0.556 | 0.526 | 0.576 | 0.547 | 0.537 | 0.509 |

The English side, `renyi_english`: the two deployed pivots, the twelve `E1_*` control arms
(both halves of the family) and `T7_byt5`, on the English half of the same four corpora (English text has no script
variant, so there is one number per arm x corpus x α).

| Arm | Sāmayik test α=2.5 | Sāmayik test α=3.0 | Sāmayik test_ood α=2.5 | Sāmayik test_ood α=3.0 | Itihāsa test α=2.5 | Itihāsa test α=3.0 | FLORES devtest α=2.5 | FLORES devtest α=3.0 |
|---|---|---|---|---|---|---|---|---|
| `T0_o200k` | 0.490 | 0.461 | 0.466 | 0.441 | 0.424 | 0.397 | 0.483 | 0.455 |
| `T0_llama4` | 0.492 | 0.463 | 0.468 | 0.443 | 0.429 | 0.402 | 0.484 | 0.456 |
| `E1_bpe_32k` | 0.507 | 0.468 | 0.509 | 0.478 | 0.424 | 0.394 | 0.556 | 0.517 |
| `E1_bpe_64k` | 0.492 | 0.455 | 0.490 | 0.460 | 0.410 | 0.382 | 0.537 | 0.499 |
| `E1_unigram_32k` | 0.508 | 0.475 | 0.488 | 0.458 | 0.428 | 0.401 | 0.544 | 0.514 |
| `E1_unigram_64k` | 0.500 | 0.466 | 0.468 | 0.438 | 0.418 | 0.393 | 0.532 | 0.500 |
| `E1_bpe_128k` | 0.486 | 0.449 | 0.480 | 0.451 | 0.405 | 0.377 | 0.527 | 0.490 |
| `E1_unigram_128k` | 0.500 | 0.466 | 0.468 | 0.438 | 0.418 | 0.393 | 0.532 | 0.500 |
| `E1_bpe_32k_bm` | 0.507 | 0.469 | 0.509 | 0.478 | 0.424 | 0.395 | 0.557 | 0.518 |
| `E1_bpe_64k_bm` | 0.493 | 0.456 | 0.492 | 0.462 | 0.411 | 0.382 | 0.540 | 0.502 |
| `E1_bpe_128k_bm` | 0.487 | 0.451 | 0.483 | 0.453 | 0.406 | 0.378 | 0.530 | 0.493 |
| `E1_unigram_32k_bm` | 0.509 | 0.475 | 0.489 | 0.459 | 0.426 | 0.400 | 0.543 | 0.513 |
| `E1_unigram_64k_bm` | 0.503 | 0.469 | 0.475 | 0.446 | 0.420 | 0.394 | 0.535 | 0.504 |
| `E1_unigram_128k_bm` | 0.503 | 0.469 | 0.475 | 0.446 | 0.420 | 0.394 | 0.535 | 0.504 |
| `T7_byt5` | 0.533 | 0.511 | 0.529 | 0.508 | 0.519 | 0.499 | 0.554 | 0.535 |

**Reading it.** Two ends of the table are stable across all four corpora and both α.
`T0_gpt2` is the lowest arm in five of the eight columns (0.252–0.433) — the same arm whose
Devanagari-blind vocabulary falls back to near-byte-level segmentation in Experiment 01,
where a handful of byte types carry most of the mass — and the arm that undercuts it in the
other three is `T7_byt5` read in the *original* script (0.316 on FLORES at α=2.5, 0.353 and
0.335 on Itihāsa), which is the same phenomenon taken to its limit: a Devanagari character
is three UTF-8 bytes, so a byte vocabulary on Devanagari emits a small, extremely skewed set
of types. Read in SLP1 the same arm scores 0.509–0.576, mid-table. The Unigram family is the lowest of
the trained arms everywhere (`T2_*` 0.430–0.528), below every T0/T3 arm but `T0_gpt2` —
the lone exception being `T2_unigram_raw_32k`* edging `T0_gemma3` on FLORES (0.528 vs
0.527 at α=2.5, 0.492 vs 0.492 at α=3). The
top is `T1_bpe_raw_32k`* (0.562–0.725) in seven of the eight columns. What sits between
them is not stable: the T1 arms, `T0_o200k`, `T0_llama4` and `T3_brahmic131k` fall within
about 0.03 of each other on the two Sāmayik corpora, and raising α from 2.5 to 3 — which
lowers every value by 0.019 to 0.051 — is enough to reshuffle them, handing the top of
Sāmayik test_ood to `T0_llama4` (0.569) over `T1_bpe_raw_32k`* (0.562). Only on the two
corpora where the T1 arms lead by a margin (Itihāsa, FLORES) does the order survive α.

**And why it settles nothing.** Within the BPE family the number moves *against* TPP: on
Sāmayik test `T1_bpe_raw_32k`* scores 0.615 to `T1_bpe_raw_64k`*'s 0.589, while the 64k arm
is the cheaper of the two per proposition against its own matched control (1.035 [1.021,
1.049] versus 1.084 [1.070, 1.098]). Between families the two happen to agree — the T1
arms beat the T2 arms on both — but an intrinsic that ranks 32k above 64k where the
controlled measurement ranks them the other way is exactly the "can be gamed" caveat in
operation, not a tie-breaker. Read it as a description of the token distribution, and let
TPP (and, once Experiment 05 runs, BPC) decide.

## TPP by sentence length (two stratifications: English-binned and Sanskrit-binned)

The controlled TPP of a corpus is one number over sentences of very different lengths, and
Sāmayik's prose sentences and Itihāsa's verse lines are not the same length. `tpp_by_length`
re-measures each of the four matched pairs inside five **fixed, absolute** bins of the
English side's whitespace word count — 1-8, 9-16, 17-24, 25-40, 41+ (`length_bin_edges`) —
so that a verse line and a prose sentence of the same English length land in the same bin
and can be read against each other across corpora.

**Which side the bins are cut on biases the answer, so both sides are reported.** Within a
corpus, a sentence's length on either side is its content plus noise — here, how wordy this
particular translator happened to be. Selecting pairs by a high *English* word count
therefore preferentially selects positive noise on the **denominator** of the
Sanskrit/English token ratio, so that ratio falls as the English bin rises even if nothing
about density changes with length. Binning on the *Sanskrit* side has the mirror-image
bias: the noise then sits in the numerator, and the ratio rises with the bin. Neither
stratification is the truth; together they bracket it. **A gradient that keeps its sign
under both is a length effect; a gradient that changes sign between them is selection.**
The cross-corpus comparison at a fixed bin — verse against prose — is less exposed, since
both corpora undergo the same selection, but it is not immune either, so it too is read
under both. The bins' own means show the mechanism directly: across Sāmayik test's English
bins the English side's mean length rises 5.5 → 46.4 words while the Sanskrit side's rises
only 5.5 → 24.5; across its Sanskrit bins the Sanskrit side rises 4.0 → 28.0 while the
English side rises only 5.9 → 30.8. Each bin variable stretches its own side about twice as
far as the other.

`tpp_by_length_sa` therefore repeats the entire measurement — same pairs, same tokenizers,
same bootstrap — on bins of the Sanskrit side's whitespace word count in the original
script: 1-5, 6-10, 11-15, 16-25, 26+ (`length_bin_edges_sa`), chosen so each bin holds a
share of each corpus comparable to its English counterpart (Sāmayik test 23/40/23/12/1 %
against the English bins' 34/41/16/8/0.5 %; test_ood 14/37/26/18/5 % against 13/39/27/17/4 %;
Itihāsa 2/57/30/9/2 % against 0.1/4/34/48/14 %; FLORES 0.5/15/31/45/8 % against
0.8/24/46/28/2 %). Every summary records which side it was binned on (`bin_on`).

The strata cover the eight 32k/64k matched pairs — the four `E1_*` and the four `E1_*_bm`
ones — and not the 128k arms or `T7_byt5`, which `config["length_strata_skip_arms"]` keeps
out of both stratifications (the tables would otherwise be half as long again to answer a
question about length gradients that a third vocabulary size does not bear on). Those arms
are measured at corpus level like every other arm. In `tpp_by_length.pdf` a pair measured
against a byte-matched control is drawn **dashed**, since it lies within a hundredth of its
pair-matched twin and colour alone does not separate the two.

Cells are `TPP [95% bootstrap CI] (pairs)` — the **i.i.d.** interval, as before; the block
interval is in `results.json` beside it (`ci_low_block`/`ci_high_block`); `†` marks a bin with fewer than 30 pairs
(`length_sparse_below`), drawn hollow in `tpp_by_length.pdf`, whose two rows of panels are
these two stratifications (each panel scaled to its own non-sparse bins, with an off-scale
bin drawn as an annotated triangle at the panel edge). Corpora in config order, prose before
verse (CLAUDE.md §2.7). Under both stratifications the per-bin `n` sum to each corpus's
`n_used`, and the pooled corpus value in `tpp_controlled` lies between the smallest and
largest populated bin.

### Binned on the English side

#### Sāmayik test (prose, primary, n=2417)

| Matched pair | 1-8 words | 9-16 words | 17-24 words | 25-40 words | 41+ words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 1.225 [1.191, 1.260] (835) | 1.101 [1.080, 1.122] (995) | 1.035 [1.006, 1.064] (389) | 0.992 [0.957, 1.025] (187) | 0.920 [0.835, 1.006] (11)† |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 1.196 [1.165, 1.227] (835) | 1.056 [1.036, 1.077] (995) | 0.979 [0.950, 1.006] (389) | 0.934 [0.900, 0.965] (187) | 0.851 [0.762, 0.940] (11)† |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 1.304 [1.269, 1.339] (835) | 1.170 [1.146, 1.195] (995) | 1.083 [1.052, 1.116] (389) | 1.026 [0.986, 1.061] (187) | 0.955 [0.853, 1.057] (11)† |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 1.280 [1.248, 1.314] (835) | 1.136 [1.113, 1.159] (995) | 1.049 [1.016, 1.079] (389) | 0.981 [0.941, 1.016] (187) | 0.910 [0.818, 1.006] (11)† |

#### Sāmayik test_ood (prose, primary, out-of-domain, n=4047)

| Matched pair | 1-8 words | 9-16 words | 17-24 words | 25-40 words | 41+ words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 1.245 [1.181, 1.313] (515) | 1.142 [1.121, 1.164] (1565) | 1.103 [1.080, 1.125] (1108) | 1.036 [1.009, 1.059] (704) | 0.951 [0.904, 0.995] (155) |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 1.234 [1.172, 1.298] (515) | 1.118 [1.097, 1.139] (1565) | 1.078 [1.054, 1.100] (1108) | 1.010 [0.984, 1.033] (704) | 0.927 [0.882, 0.970] (155) |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 1.338 [1.272, 1.413] (515) | 1.223 [1.199, 1.247] (1565) | 1.180 [1.155, 1.205] (1108) | 1.111 [1.083, 1.137] (704) | 1.013 [0.960, 1.063] (155) |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 1.348 [1.283, 1.420] (515) | 1.222 [1.199, 1.246] (1565) | 1.174 [1.149, 1.198] (1108) | 1.098 [1.069, 1.124] (704) | 1.001 [0.950, 1.049] (155) |

#### Itihāsa test (verse, secondary — meter is a confound, n=11721)

| Matched pair | 1-8 words | 9-16 words | 17-24 words | 25-40 words | 41+ words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 8.944 [3.647, 21.501] (9)† | 0.984 [0.964, 1.002] (419) | 0.760 [0.755, 0.764] (3945) | 0.609 [0.605, 0.612] (5658) | 0.588 [0.579, 0.598] (1690) |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 7.944 [3.061, 19.601] (9)† | 0.927 [0.908, 0.946] (419) | 0.705 [0.701, 0.710] (3945) | 0.562 [0.559, 0.566] (5658) | 0.546 [0.536, 0.555] (1690) |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 9.833 [4.104, 23.800] (9)† | 0.996 [0.975, 1.016] (419) | 0.774 [0.769, 0.779] (3945) | 0.622 [0.619, 0.626] (5658) | 0.596 [0.587, 0.605] (1690) |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 8.889 [3.350, 22.228] (9)† | 0.951 [0.933, 0.971] (419) | 0.734 [0.729, 0.739] (3945) | 0.589 [0.585, 0.592] (5658) | 0.564 [0.555, 0.573] (1690) |

#### FLORES devtest (Wikipedia, tertiary, n=1012)

| Matched pair | 1-8 words | 9-16 words | 17-24 words | 25-40 words | 41+ words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 1.165 [1.000, 1.336] (8)† | 1.155 [1.127, 1.181] (244) | 1.148 [1.132, 1.165] (464) | 1.136 [1.117, 1.155] (280) | 1.091 [1.045, 1.139] (16)† |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 1.206 [1.047, 1.375] (8)† | 1.164 [1.136, 1.191] (244) | 1.151 [1.133, 1.170] (464) | 1.130 [1.111, 1.150] (280) | 1.102 [1.057, 1.144] (16)† |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 1.228 [1.086, 1.378] (8)† | 1.212 [1.181, 1.244] (244) | 1.211 [1.191, 1.231] (464) | 1.204 [1.183, 1.226] (280) | 1.143 [1.100, 1.182] (16)† |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 1.286 [1.104, 1.462] (8)† | 1.237 [1.209, 1.266] (244) | 1.227 [1.209, 1.248] (464) | 1.209 [1.188, 1.231] (280) | 1.132 [1.082, 1.179] (16)† |

### Binned on the Sanskrit side

#### Sāmayik test (prose, primary, n=2417)

| Matched pair | 1-5 Sanskrit words | 6-10 Sanskrit words | 11-15 Sanskrit words | 16-25 Sanskrit words | 26+ Sanskrit words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 0.924 [0.885, 0.965] (566) | 1.063 [1.039, 1.087] (973) | 1.099 [1.074, 1.125] (562) | 1.156 [1.128, 1.184] (287) | 1.280 [1.196, 1.374] (29)† |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 0.894 [0.858, 0.934] (566) | 1.013 [0.991, 1.037] (973) | 1.050 [1.027, 1.074] (562) | 1.101 [1.073, 1.130] (287) | 1.206 [1.123, 1.304] (29)† |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 0.986 [0.944, 1.030] (566) | 1.117 [1.091, 1.144] (973) | 1.148 [1.123, 1.176] (562) | 1.228 [1.195, 1.263] (287) | 1.371 [1.278, 1.492] (29)† |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 0.968 [0.927, 1.013] (566) | 1.075 [1.050, 1.101] (973) | 1.117 [1.093, 1.144] (562) | 1.188 [1.156, 1.222] (287) | 1.319 [1.227, 1.433] (29)† |

#### Sāmayik test_ood (prose, primary, out-of-domain, n=4047)

| Matched pair | 1-5 Sanskrit words | 6-10 Sanskrit words | 11-15 Sanskrit words | 16-25 Sanskrit words | 26+ Sanskrit words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 0.769 [0.727, 0.816] (571) | 0.998 [0.979, 1.018] (1500) | 1.101 [1.081, 1.122] (1046) | 1.178 [1.151, 1.206] (738) | 1.268 [1.212, 1.326] (192) |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 0.759 [0.717, 0.806] (571) | 0.978 [0.959, 0.997] (1500) | 1.076 [1.056, 1.097] (1046) | 1.147 [1.122, 1.174] (738) | 1.238 [1.183, 1.295] (192) |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 0.820 [0.775, 0.870] (571) | 1.067 [1.044, 1.087] (1500) | 1.183 [1.160, 1.208] (1046) | 1.260 [1.231, 1.291] (738) | 1.361 [1.301, 1.427] (192) |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 0.815 [0.770, 0.866] (571) | 1.065 [1.045, 1.085] (1500) | 1.174 [1.153, 1.198] (1046) | 1.249 [1.222, 1.279] (738) | 1.349 [1.290, 1.413] (192) |

#### Itihāsa test (verse, secondary — meter is a confound, n=11721)

| Matched pair | 1-5 Sanskrit words | 6-10 Sanskrit words | 11-15 Sanskrit words | 16-25 Sanskrit words | 26+ Sanskrit words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 0.608 [0.579, 0.638] (225) | 0.630 [0.626, 0.634] (6699) | 0.643 [0.637, 0.649] (3458) | 0.674 [0.665, 0.685] (1077) | 0.712 [0.689, 0.736] (262) |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 0.559 [0.534, 0.587] (225) | 0.580 [0.576, 0.584] (6699) | 0.600 [0.594, 0.605] (3458) | 0.626 [0.617, 0.635] (1077) | 0.670 [0.646, 0.695] (262) |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 0.576 [0.549, 0.606] (225) | 0.639 [0.635, 0.643] (6699) | 0.665 [0.659, 0.671] (3458) | 0.687 [0.678, 0.697] (1077) | 0.722 [0.700, 0.745] (262) |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 0.540 [0.514, 0.568] (225) | 0.602 [0.598, 0.606] (6699) | 0.633 [0.627, 0.639] (3458) | 0.653 [0.643, 0.663] (1077) | 0.685 [0.664, 0.708] (262) |

#### FLORES devtest (Wikipedia, tertiary, n=1012)

| Matched pair | 1-5 Sanskrit words | 6-10 Sanskrit words | 11-15 Sanskrit words | 16-25 Sanskrit words | 26+ Sanskrit words |
|---|---|---|---|---|---|
| `T1_bpe_raw_32k`* / `E1_bpe_32k` | 1.000 [0.868, 1.228] (5)† | 1.073 [1.043, 1.105] (153) | 1.101 [1.082, 1.122] (311) | 1.157 [1.143, 1.172] (457) | 1.223 [1.182, 1.269] (86) |
| `T1_bpe_raw_64k`* / `E1_bpe_64k` | 0.967 [0.864, 1.119] (5)† | 1.075 [1.044, 1.109] (153) | 1.098 [1.078, 1.119] (311) | 1.160 [1.144, 1.175] (457) | 1.225 [1.185, 1.267] (86) |
| `T2_unigram_raw_32k`* / `E1_unigram_32k` | 1.138 [0.875, 1.398] (5)† | 1.132 [1.098, 1.167] (153) | 1.164 [1.142, 1.188] (311) | 1.221 [1.203, 1.238] (457) | 1.287 [1.242, 1.334] (86) |
| `T2_unigram_raw_64k`* / `E1_unigram_64k` | 1.095 [0.903, 1.275] (5)† | 1.148 [1.112, 1.187] (153) | 1.172 [1.150, 1.196] (311) | 1.237 [1.218, 1.253] (457) | 1.295 [1.248, 1.346] (86) |

**(1) Does the within-prose length gradient keep its sign under Sanskrit binning? No — it
reverses, in every corpus and every pair.** Under English bins Sāmayik test's ratio falls
monotonically with length (`T1_bpe_raw_32k`*: 1.225 → 1.101 → 1.035 → 0.992 → 0.920†) and
three of the four pairs drop below 1.0 by the 25-40 bin. Under Sanskrit bins the same pairs
rise just as monotonically (`T1_bpe_raw_32k`*: 0.924 → 1.063 → 1.099 → 1.156 → 1.280†), and
the only bin below 1.0 is now the *shortest* one — 0.894 to 0.986 at 1-5 Sanskrit words,
where the English gradient's shortest bin was its most adverse. Sāmayik test_ood does the
same (English 1.245 → 0.951; Sanskrit 0.769 → 1.268), so does FLORES (English 1.165 →
1.091; Sanskrit 1.000† → 1.223), and so does Itihāsa (English 0.984 → 0.588; Sanskrit
0.608 → 0.712). Sixteen within-corpus gradients, sixteen sign reversals: the length
gradient tracks the bin variable, which is what selection on the binned side looks like and
what a genuine density-by-length effect does not. Nothing in these tables supports "Sanskrit
gets relatively cheaper as sentences get longer", and the previous version of this section,
which read the English-binned fall as a finding about long prose, was reading the
stratification rather than the corpus.

**(2) At bins where both corpora are populated, is verse still below prose under both
stratifications? Yes, in all 28 comparisons.** Three English bins hold ≥ 30 pairs in both
Itihāsa test and Sāmayik test (9-16, 17-24, 25-40) and four Sanskrit bins do (1-5, 6-10,
11-15, 16-25); that is 7 bins × 4 matched pairs. In every one the verse corpus sits below
the prose corpus measured against the same matched control. The **smallest** gap is 0.117
(English 9-16, `T1_bpe_raw_32k`*: 0.984 verse against 1.101 prose) and the **largest** is
0.541 (Sanskrit 16-25, `T2_unigram_raw_32k`*: 0.687 against 1.228); within the English rows
the range is 0.117-0.404, within the Sanskrit rows 0.316-0.541. Sāmayik test_ood gives the
same picture over its own jointly populated bins (English 0.158-0.510, Sanskrit
0.161-0.663). The two stratifications differ on one secondary point: under Sanskrit binning
Itihāsa is below 1.0 in all four jointly populated bins with every CI excluding it, while
under English binning the 9-16 bin's two 32k arms straddle the line (0.984 [0.964, 1.002]
and 0.996 [0.975, 1.016]).

**(3) What this does and does not settle.** The verse-prose separation is the part that
survives both stratifications — it holds at every jointly populated bin of either, with a
gap of at least 0.117 TPP, so it is not an artefact of Itihāsa's sentences being longer or
shorter than Sāmayik's on either side; the within-corpus length gradient is the part that
does not survive, and no claim about TPP changing with sentence length should be made from
these tables. Even the surviving separation is bracketed rather than isolated: a bin equates
the two corpora on one side and leaves the other free (at 25-40 English words Itihāsa's
Sanskrit side averages 10.4 whitespace words against Sāmayik's 17.9; at 6-10 Sanskrit words
Itihāsa's English side averages 26.2 against Sāmayik's 10.7), which is consistent with the
confounds this experiment already flags for Itihāsa — meter on the Sanskrit side, a
19th-century verse translation on the English side — but nothing here measures meter, and
these bins test length only.

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
- **The Unigram trainer does not reach every requested vocabulary size, and the 128k
  Unigram pairs are therefore not size-matched.** `E1_unigram_64k` and `E1_unigram_128k`
  both settle at **62,896** pieces, and `E1_unigram_64k_bm` and `E1_unigram_128k_bm` both at
  **50,659**: the EM trainer stops when the corpus does not support more. So
  `T2_unigram_raw_128k`* (a full 128,000) is measured against a ~63k / ~51k English control,
  which violates CLAUDE.md §2.5 for that one row. The shortfall makes the English side
  *dearer* and therefore pushes those ratios **down**, i.e. towards Sanskrit's favour, and
  they are still above 1.0 on all three prose/Wikipedia corpora, so nothing in the verdict
  turns on it. The BPE 128k pair *is* matched: 128,000 on both sides, both controls.
- **The byte-matched control is matched on bytes, not on sentences.** `E1_*_bm` sees
  78,624 of the 116,127 English lines (67.7%) at 11,209,371 bytes (1.0000013x the Sanskrit
  corpus). Cutting on bytes necessarily unmatches the sentence count, exactly as matching
  sentences unmatched the byte count; the two controls bracket the question rather than
  settling it, and both are reported for every pair. The subsample is uniform over lines at
  `random.Random(0)`, so it is reproducible and carries no length preference of its own.
- **`T7_byt5` is a reference, not a proposal.** A byte vocabulary is the most expensive arm
  in every deployed table (2.5–5.2 tokens per English token) and its interest is entirely
  that it has no vocabulary at all: measured on both sides it reports the two sides' byte
  ratio, which is the text-length component of every other row on the page.
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
  ci_low, ci_high, ci, n_bootstrap, seed, n_undefined, source_tokens, pivot_tokens,
  ci_low_block, ci_high_block, block_length, n_blocks}` — the last four added by this wave,
  and `ci_low`/`ci_high` still the i.i.d. interval every table above reports —
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
