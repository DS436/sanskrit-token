# LinkedIn series: "Does Sanskrit's density survive tokenization?"

Twelve posts, two a week (Tuesday and Thursday, morning IST), one experiment per pair of posts, then the GPU run and a wrap-up. Every number below is in the repo's `results.json` files and the media in `docs/outreach/media/` is rendered from them by `scripts/social_figures.py`; the media `manifest.json` records the exact values plotted, so a post can be checked against it before it goes out.

## Ground rules for every post

- Lead with the question or the surprise, not the method. Three facts maximum. One takeaway. One link (the repo).
- Never headline fertility (tokens per word). Never write "fewer words" as if it meant "fewer tokens". Never mention NASA.
- Say "provisional" where the tokenizers were trained on the parallel training splits, and "oracle" or "heuristic" where the README does. The negatives are the honest part of the story; do not soften them.
- Do not claim the paper's conclusion before Experiment 05 runs. Until then the series is "what we've measured so far".
- Alt text on every image (supplied below). Hashtags at the end, five at most: #NLP #Tokenization #Sanskrit #LLM #MachineLearning.
- Reply to comments with numbers from the README, not from memory.

## Before post 1: make the repository public

The series only works if readers can open the repo and see the numbers. Before post 1 goes out:

1. `LICENSE` at the root (MIT for the code; each data source keeps its own licence, listed in `data/README.md`).
2. A visitor-facing `README.md`: what the project asks, a results table with one line per experiment linking to its README and figure, how to reproduce (`uv sync`, one command per experiment), what needs a download and what needs a GPU, how to cite.
3. A tracked `results/` snapshot of every experiment's `results.json`, `config.yaml` and figures (the live `outputs/` folder stays gitignored), so results are viewable without running anything.
4. Secrets and raw-data check: no tokens, no `data/raw` or `data/processed` files in history.
5. Flip visibility to public and put the repo link in post 1's first comment.

Post 2 then points at the decision log and the results folder as the proof that the numbers are checkable.

## Calendar

| # | Day | Topic | Media |
|---|---|---|---|
| 1 | Week 1 Tue | The question | `00_sandhi_card` |
| 2 | Week 1 Thu | How the work is run (agent orchestration, review loops, no-leakage rules) | `00_arms_card` |
| 3 | Week 2 Tue | Exp 01: the language tax | `01_language_tax`, `01_parity` |
| 4 | Week 2 Thu | Exp 01: the metric that would have lied | `01_fertility_vs_parity` |
| 5 | Week 3 Tue | Exp 02: the sign flip that wasn't | `02_flip_vs_control` |
| 6 | Week 3 Thu | Exp 02: prose, verse, and domain | `02_by_corpus` |
| 7 | Week 4 Tue | Exp 03: what sandhi splitting buys | `03_split_deltas`, `03_example_card` |
| 8 | Week 4 Thu | Exp 03: the result review caught | `03_artefact` |
| 9 | Week 5 Tue | Exp 04: alignment versus compression | `04_scatter` |
| 10 | Week 5 Thu | Exp 04: leakage you can't hash | `04_leakage` |
| 11 | When the GPU run starts | Exp 05: what bits-per-character will settle | dry-run table screenshot |
| 12 | When Exp 05 aggregates | Exp 05 result and what the paper will claim | `bpc_final` from the sweep |

## Post drafts

### 1. The question

Sanskrit packs case, number, person, tense and whole compounds into single words. Sandhi then erases the spaces between them: tat + api becomes tadapi.

Linguists call that density. Tokenizers call it a problem: modern LLM tokenizers were fit on English-heavy text and fragment Devanagari at the byte level.

So here is the question I've spent the last weeks on: does Sanskrit's density survive tokenization, and can a tokenizer built for Sanskrit recover it?

Two claims, kept apart on purpose:
- Claim A: Sanskrit is dense per word. True; we cite it, we don't test it.
- Claim B: that density survives into tokens, measured per unit of meaning on parallel text. That is the experiment.

Everything is open: five experiments, every number reproducible with one command, every design decision logged with its date and the alternative rejected. The repository is public as of today; link in the first comment, results folder included so you can check any number without running anything.

Over the next few weeks I'll post each experiment as it stands, including the two that came out against the hypothesis.

Alt text: A card showing "tat + api → tadapi" in Devanagari and SLP1, explaining that sandhi removes the space between words.

### 2. How the work is run

A note on process before the results, because it shaped them.

The project runs as one orchestrating model that plans, decides, and reviews, with every line of code written by a separate model in an isolated context. Each task gets a fresh reviewer that reads only the diff, and nothing merges with an open finding.

Three rules did most of the work:
1. No evaluation sentence may appear in any training set. Two layers enforce it: exact hashes and 24-letter shingles. The shingle layer later caught leakage the hashes had missed (post 10).
2. Fertility is computed and reported but never the headline (post 4 shows why).
3. Every deviation goes in an append-only decision log, dated, with the alternative that was rejected.

The tokenizer ladder we test is in the image: off-the-shelf English tokenizers, raw BPE and Unigram trained on Sanskrit, off-the-shelf Indic tokenizers, sandhi-split variants, morpheme-constrained variants, a byte-level floor, and a matched English control trained on the English side of the same corpus. That last one turned out to matter most.

Alt text: A table listing tokenizer arms T0 through T7 and the E1 English control with one-line descriptions and vocabulary sizes.

### 3. Exp 01: the language tax

First measurement: how much do today's tokenizers charge Sanskrit?

On identical FLORES sentences in Sanskrit, Hindi and English:
- GPT-2's tokenizer needs about 12.5 tokens per Sanskrit word. The 200k-vocabulary tokenizers in Llama 4, Gemma 3 and GPT-4o need 3 to 4.
- Per identical sentence, Sanskrit costs 1.8 to 2.2 times as many tokens as English under the modern tokenizers, and 7.9 times under GPT-2.
- Sanskrit costs about 1.3 times as many tokens as Hindi, same script, no sandhi.

So the tax is real and it shrank a lot between tokenizer generations. The pre-registered prediction of "more than 5 tokens per word" was calibrated on the older generation; it holds for GPT-2 and fails for everything current.

Next post: why the per-word number in the first chart is the wrong one to headline, even though it is the one everyone quotes.

Alt text: Grouped bars of tokens per word for Sanskrit, Hindi and English under GPT-2 and GPT-4o's tokenizer, and a second chart of Sanskrit-to-English and Sanskrit-to-Hindi token ratios per tokenizer with a line at 1.0.

### 4. Exp 01: the metric that would have lied

Tokens per word is the standard "tokenizer fairness" number. For Sanskrit it points the wrong way.

Sanskrit writes the same content in fewer, longer words: on our 1,012 FLORES sentences, 16,975 words against Hindi's 25,643. Divide tokens by that smaller word count and the ratio inflates.

The hypothesis said Sanskrit should cost more than 1.5 times Hindi's tokens. By tokens per word, the ratio is 1.7. By tokens on identical content, it is 1.3. The per-word number would have confirmed a prediction the real count refutes.

That is the whole reason this project measures tokens per proposition on parallel text and treats fertility as a diagnostic. Fewer words is not fewer tokens, in either direction.

Alt text: Chart comparing the Sanskrit-to-Hindi ratio computed from tokens per word against the ratio computed from tokens on identical sentences, with the 1.5 threshold marked; the per-word ratio is above it and the true ratio below.

### 5. Exp 02: the sign flip that wasn't

The headline experiment: train a tokenizer on Sanskrit and count tokens per proposition against English on parallel prose.

First result: a 64k BPE trained on Sanskrit brought Sanskrit below English, 0.89 tokens for every English token under GPT-4o's tokenizer. The sign flip the hypothesis predicted.

Then the control. We trained the same algorithm, same vocabulary size, on the English side of the same corpus. Against that matched English tokenizer, Sanskrit costs 1.03 to 1.14 tokens per English token. Above parity on every prose corpus.

The flip was the English tokenizer being out of its domain, not Sanskrit being cheaper. Every "language X is more efficient" comparison that uses a general-purpose tokenizer on one side is exposed to this.

Verdict as it stands: raw subword training does not recover the density on prose. The sandhi-split and morpheme-constrained tokenizers were still untested at this point. They are the next two posts.

Alt text: Two bars for the same Sanskrit tokenizer on Sāmayik prose: 0.89 against GPT-4o's tokenizer, 1.03 against the matched English control, with a dashed line at 1.0.

### 6. Exp 02: prose, verse, and domain

The same experiment across four corpora tells a second story.

- Contemporary prose (Sāmayik): 1.03 to 1.14.
- Out-of-domain prose: 1.06 to 1.16.
- FLORES (Wikipedia register): 1.14 to 1.22.
- Verse (Itihāsa, the epics): 0.61 to 0.66.

Verse is the only place Sanskrit falls below English, and verse is a confound: the śloka meter constrains word choice and inflates compounding, and the English translations of the epics are famously wordy. We report it and do not build on it.

Also visible: the gap is smallest on the text closest to the tokenizers' training data. Domain fit predicts this pattern about as well as any claim about the language does, which is why the control in post 5 was necessary.

Alt text: Controlled tokens-per-proposition for four tokenizer pairs across four corpora, prose first, all above 1.0 except the verse corpus.

### 7. Exp 03: what sandhi splitting buys

Now the first piece of the proposed method: reverse sandhi before training the tokenizer.

We ran ByT5-Sanskrit, the current best segmenter, over 136,651 sentences (9.4 hours on a laptop GPU). It splits sandhi and also compounds, and it rewrites some spelling, so the split text is reconciled against the original so that nothing but boundaries changes. The example card shows one sentence at each stage.

Result, against the matched English control, split minus unsplit:
- Contemporary prose: 0.005 to 0.043 fewer tokens per English token.
- Out-of-domain prose and FLORES: 0.03 to 0.08 fewer.
- Verse: 0.006 to 0.017 more.

Real, small, and consistent on prose. Still no arm below its matched English control. The next post is about how this number was almost twice as large.

Alt text: Paired differences in tokens per proposition, split minus raw, for four tokenizer pairs across four corpora, with confidence intervals; negative on prose, positive on verse. A second image shows one Sanskrit sentence in Devanagari, in SLP1, and after sandhi splitting.

### 8. Exp 03: the result review caught

The first run of this experiment produced the number I wanted: a sandhi-split tokenizer at 0.976, below the matched English control on prose. First time in the project.

The whole-branch review re-tokenised the characters the splitter had deleted and priced them. Punctuation attached to words, `karoti.` and `"tadā,`, had been dropped and not restored. Those characters cost tokens. They explained 77 to 110 percent of the gain.

Fixed the reconciliation, retrained, re-ran: 1.03. A second round found hyphens marking compound boundaries were being credited as discovered boundaries. Fixed that too.

Two lessons I'm keeping:
- Character retention (99 percent!) said nothing was wrong. Only re-tokenising the deleted characters showed the cost, because punctuation is expensive per character and letters are cheap.
- A review that can run code and recompute the number is worth more than one that reads the prose.

The decision log has both corrections, dated, with the withdrawn value.

Alt text: Three bars: the withdrawn first-run value 0.976 in red, the corrected value after punctuation was restored, and the unsplit baseline, with an annotation that deleted characters explained most of the original gain.

### 9. Exp 04: alignment versus compression

The second piece of the method: forbid BPE merges that cross gold morpheme boundaries, using the Digital Corpus of Sanskrit.

We measured two things per tokenizer against a matched baseline trained on the same corpus: MorphScore (do token boundaries land on morpheme boundaries?) and tokens per proposition.

- Constrain on gold segment boundaries only: MorphScore up 0.11, token cost unchanged (confidence interval includes zero).
- Add a heuristic stem/ending boundary: MorphScore up less, and 8 percent more tokens.
- The full proposed method (split, then constrain): 8 percent more tokens.
- Gold splitting alone, no constraint: 2 percent fewer.

So the constraint does what it is meant to do to the segmentation, and it is free when the boundaries are real. The token cost came from the heuristic half of the boundary set, not from the gold half. Whether alignment pays off in training efficiency, which is what the MorphBPE paper claims, is the GPU experiment.

Alt text: Scatter of four constrained tokenizers: x is change in tokens per proposition, y is change in MorphScore; the gold-segment-only tokenizer sits at zero cost and highest alignment gain.

### 10. Exp 04: leakage you can't hash

A process post, because this one nearly went into the paper.

The Digital Corpus of Sanskrit contains the Mahābhārata and Rāmāyaṇa. So does the Itihāsa test set we evaluate on. We excluded every evaluation sentence from training by hash, as the project rules require.

Review sampled 400 test verses and searched the training corpus for their letter strings: 19 percent were there verbatim, 36 percent by 24-letter prefix. DCS segments verses differently from Itihāsa, so the hashes matched 776 sentences and missed the rest. Every verse number for those tokenizers was contaminated, and asymmetrically, because unconstrained tokenizers memorise whole words.

Fix: a second exclusion layer that drops any training sentence sharing a 24-letter run of letters with any evaluation sentence. It removed 34,705 training sentences (20,611 of them Itihāsa test overlaps). Residual after the fix: 0 of 400.

If your evaluation set is a subset of a public corpus, exact-match exclusion is not enough. Shingle it.

Alt text: Left panel: percentage of verse test sentences found inside the training corpus before and after the shingle filter; right panel: training sentences dropped per evaluation source.

### 11. Exp 05: what bits-per-character will settle

Everything so far counted tokens. The last experiment trains small language models with each tokenizer and measures bits per character on held-out text, which is comparable across vocabularies where perplexity is not.

Fifty-one runs: seven tokenizers on a matched corpus at 50M parameters, five at 50M and 125M on a 200M-token corpus, three seeds each, equal training bytes. About a day of A100 time, rented.

The honest expectation going in: the proposed tokenizer starts with a 24 to 33 percent in-domain token handicap, about the size of the speedup the hypothesis predicts. The one to watch is the gold-segment-only constraint, which cost nothing in tokens.

Results in a week or two. The sweep is one command; the pipeline was validated on a laptop first, including a review that found the byte-level baseline was being scored against padded vocabulary rows.

Alt text: Screenshot of the sweep's dry-run table: runs, tokens, estimated FLOPs and hours per track.

### 12. Exp 05 result and what the paper will claim

(Draft after aggregation. Structure: one sentence on the result; the BPC bar chart; whether any arm reached the baseline's loss with fewer bytes; what the paper will and won't claim; link to the preprint when it exists.)

## Media checklist before posting

1. Open `docs/outreach/media/manifest.json` and confirm each number in the post text matches the plotted value.
2. Check the figure title makes no claim the README's verdict paragraph doesn't make.
3. Attach the 1600×900 image for feed posts; the `_sq` square variant if posting from mobile.
4. Put the repo link in the first comment, not the post body.
