# LinkedIn plan: "Does Sanskrit's density survive tokenization?"

Two posts, not a series. One now, while the paper is written but not yet announced, and
one on the day it appears on arXiv. Every number in either post is in a committed
`results/*/results.json` or in `paper/1a/`, and the media in `docs/outreach/media/` is
rendered from those files by `scripts/social_figures.py`; `manifest.json` records the
exact values plotted, so a post can be checked against it before it goes out.

**Why two and not twelve.** The twelve-post plan in the appendix was written before the
2026-09-08 revision wave (byte-matched control, 128k vocabulary arms, the verse
decomposition, block bootstrap). A serial narrative about results that were still moving
would have meant correcting myself in public, one post at a time. Two posts, both written
after the results settled, say the same thing without that exposure: one about the
question and the measurement trap, which cannot move, and one about the finished result.

## Ground rules for both posts

- Lead with the question or the surprise, not the method. Three facts maximum. One
  takeaway. One link (the repository).
- **No post may state a number that is not in `results/` or in `paper/1a/`.** Ranges are
  read off the results file or off `paper/1a/tables/numbers.tex`, never rounded from
  memory or recomputed by hand.
- **A post that names a result links the repository**, so a reader can check the number
  without running anything. In the post body when the post is the only one out (post 1);
  in the first comment when the arXiv link owns the body (post 2).
- Never headline fertility (tokens per word). Post 1 may use it as the metric that
  misleads, which is the paper's own framing, but it is never the finding.
- Never write "fewer words" as if it meant "fewer tokens". Never mention perplexity, which
  is not comparable across vocabularies. Never mention NASA.
- Off-the-shelf tokenizers are deployed practice, never a controlled comparison. Say
  "deployed", not "generic baseline", and never present a Sanskrit arm beating one as
  evidence about the language.
- Prose before verse. The verse number is a lead, not a finding, and any post that gives
  it says so.
- Do not soften the negatives. The result is negative about the baselines and says nothing
  about the method this project proposes; post 2 states both.
- Alt text on every image (supplied below). At most two hashtags: #NLP #Tokenization.
- Reply to comments with numbers from `experiments/*/README.md` and `paper/1a/`, not from
  memory. If a commenter is right about something, say so and link the decision log.

## Before post 1: make the repository public

Post 1 names results and links the repository, so the repository has to be readable first:

1. `LICENSE` at the root (MIT for the code; each data source keeps its own licence, listed
   in `data/README.md`).
2. A visitor-facing `README.md`: what the project asks, a results table with one line per
   experiment linking to its README and figure, how to reproduce (`uv sync`, one command
   per experiment), what needs a download and what needs a GPU, how to cite.
3. The tracked `results/` snapshot of every experiment's `results.json`, `config.yaml` and
   figures (the live `outputs/` folder stays gitignored), so results are viewable without
   running anything.
4. Secrets and raw-data check: no tokens, no `data/raw` or `data/processed` files in
   history.
5. Flip visibility to public.

Post 2 additionally needs the arXiv identifier, and `CITATION.cff` and the root
`README.md` updated with it (`paper/1a/SUBMISSION.md`, "After submission").

## Two-post plan

### Post 1: the question and the measurement trap

**When:** now, before the paper is announced. It states only Experiment 01 numbers, which
are final and cannot move.

**Media:** `docs/outreach/media/01_fertility_vs_parity.png` (square variant
`01_fertility_vs_parity_sq.png` if posting from mobile). Verified cell by cell against
`results/01_baseline_penalty/results.json` on 2026-09-09: all eight plotted values match
exactly.

**Alt text:**

> Grouped bar chart. For each of four deployed tokenizers, GPT-2, o200k, Llama-4 and
> Gemma-3, a red bar gives the Sanskrit-over-Hindi ratio implied by tokens per word and a
> green bar gives the token ratio measured on identical FLORES-200 content. A dashed line
> marks the pre-registered threshold of 1.5. Every red bar sits above the line, between
> 1.60 and 1.74; every green bar sits below it, between 1.06 and 1.35.

**Post text:**

Sanskrit fuses case, number, person and tense into word endings, and chains what English
writes as several clauses into a single compound. It is dense per word.

Whether that density survives tokenization is a different question, and it has to be asked
per unit of meaning. Tokens per word divides by a denominator that Sanskrit's own grammar
shrinks.

Here is what that costs you. On 1,012 identical FLORES-200 sentences I put one question to
four deployed tokenizers, is Sanskrit more expensive than Hindi in tokens, and answered it
two ways. Read off tokens per word, Sanskrit costs 1.60 to 1.74 times Hindi. Counted on
the identical content, 1.06 to 1.35 times. My pre-registered threshold was 1.5. The
per-word reading clears it for all four tokenizers. The count on identical content clears
it for none. The metric everyone quotes would have confirmed a hypothesis that the correct
measurement refutes, for every tokenizer I tested.

Hindi is the control worth having here: same script, but no productive sandhi between
words and much less fusion inside them.

The code and every result are public, so any number above can be checked without running
anything: github.com/DS436/sanskrit-token

A paper is on the way.

#NLP #Tokenization

**Does not claim:** anything about whether a Sanskrit-trained tokenizer recovers the
density, or about what any token count buys a language model.

### Post 2: the result, on the day the paper is announced

**When:** the day arXiv announces the paper, not the day it is submitted. arXiv assigns an
identifier on acceptance into the queue but only lists the paper on the next mailing
cycle, roughly one business day and weekdays only (`paper/1a/SUBMISSION.md`).

**Media:** `docs/outreach/media/02_flip_vs_control.png` (square variant
`02_flip_vs_control_sq.png`). Regenerated on 2026-09-09 to add the byte-matched control as
a third bar; the two-bar version predated it. All nine plotted values verified against
`results/02_tpp_parallel/results.json`.

**Alt text:**

> Bar chart of tokens per proposition for the Sanskrit BPE tokenizer T1_bpe_raw_64k on
> Sāmayik test prose, with 95% confidence intervals. Against the deployed English
> tokenizer o200k the value is 0.887, below the dashed parity line at 1.0. Against the
> pair-matched English control E1_bpe_64k it is 1.035, and against the byte-matched
> control E1_bpe_64k_bm it is 1.030; both sit above the line.

**Link placement:** the arXiv link goes in the post body, the repository link in the first
comment. Post the comment yourself, immediately, so it sits at the top of the thread.

**Post text:**

The paper is up: [ARXIV LINK]

It is a negative result, and the negative is the point.

Train a BPE tokenizer on Sanskrit, count tokens per proposition against English on
contemporary prose, and score it against a deployed English tokenizer: 0.887, below
English. That looks like Sanskrit's density surviving tokenization.

It does not survive the control. Train the same algorithm at the same vocabulary size on
the English side of the same corpus, and the same Sanskrit token counts read 1.035, above
English. The Sanskrit numerator never moved. The whole difference is the English
denominator, because the control had been trained on this domain and the deployed
tokenizer had not.

Two things the revision added.

A byte-matched control, built because the first control had seen 48 percent more training
text than the Sanskrit side. Cutting it to the Sanskrit corpus's byte count moves every
ratio by at most 0.025 and changes no verdict.

And a decomposition. Tokens per proposition factorises exactly into a character-length
ratio and a tokens-per-character ratio, and the second stays near 1. So what a matched
tokenizer preserves is length, and Sanskrit prose has no length advantage over English in
this encoding, 1.028. Verse does, 0.596, which is a lead rather than a finding.

One qualification I will not bury: at the largest vocabulary I tested, 128,000 pieces, the
matched BPE pair does cross below parity on in-domain prose, 0.983. It stays above parity
out of domain.

So this is a result about the baselines. The method this project proposes, reversing
sandhi before subword learning and forbidding merges across gold morpheme boundaries, is
untested here. That is the follow-up.

#NLP #Tokenization

**Does not claim:** that Sanskrit is or is not more token-efficient in general, that the
proposed tokenizer would do better, or anything about downstream model quality.

## Media checklist before posting

1. Open `docs/outreach/media/manifest.json` and confirm each number in the post text
   matches the plotted value, and that `source_files` names the `results/` file the number
   was read from.
2. Check the figure title makes no claim the experiment's README verdict paragraph does
   not make.
3. Attach the 1600x900 image for feed posts; the `_sq` square variant if posting from
   mobile.
4. Post 1: repository link in the body. Post 2: arXiv link in the body, repository link in
   the first comment.

## Appendix: longer series, held in reserve

**These twelve drafts predate the 2026-09-08 revision wave and are not ready to post.**
They were written against the results as they stood before the byte-matched control, the
128k vocabulary arms, the verse decomposition and the block bootstrap. Posts 5 and 6 are
the ones the wave touched, and they are **incomplete rather than wrong**: the domain-fit
explanation they give survived the revision intact, but neither mentions the byte-matched
control, neither says the controlled ratio depends on vocabulary size and crosses below
parity at 128k on in-domain prose, and neither has the decomposition that locates the
verse result in the character ratio rather than in tokenization. Post 3's framing of the
tokenizer arms as a "language tax" also needs the deployed-practice wording the ground
rules above now require.

Any of these would need checking line by line against `paper/1a/main.tex` and the relevant
`experiments/*/README.md` before use. They are kept because the process posts (2, 8, 10)
and the experiment 03 and 04 material are still accurate and may be worth a second wave
after the paper lands.

#### 1. The question

Sanskrit packs case, number, person, tense and whole compounds into single words. Sandhi then erases the spaces between them: tat + api becomes tadapi.

Linguists call that density. Tokenizers call it a problem: modern LLM tokenizers were fit on English-heavy text and fragment Devanagari at the byte level.

So here is the question I've spent the last weeks on: does Sanskrit's density survive tokenization, and can a tokenizer built for Sanskrit recover it?

Two claims, kept apart on purpose:
- Claim A: Sanskrit is dense per word. True; we cite it, we don't test it.
- Claim B: that density survives into tokens, measured per unit of meaning on parallel text. That is the experiment.

Everything is open: five experiments, every number reproducible with one command, every design decision logged with its date and the alternative rejected. The repository is public as of today; link in the first comment, results folder included so you can check any number without running anything.

Over the next few weeks I'll post each experiment as it stands, including the two that came out against the hypothesis.

Alt text: A card showing "tat + api → tadapi" in Devanagari and SLP1, explaining that sandhi removes the space between words.

#### 2. How the work is run

A note on process before the results, because it shaped them.

The project runs as one orchestrating model that plans, decides, and reviews, with every line of code written by a separate model in an isolated context. Each task gets a fresh reviewer that reads only the diff, and nothing merges with an open finding.

Three rules did most of the work:
1. No evaluation sentence may appear in any training set. Two layers enforce it: exact hashes and 24-letter shingles. The shingle layer later caught leakage the hashes had missed (post 10).
2. Fertility is computed and reported but never the headline (post 4 shows why).
3. Every deviation goes in an append-only decision log, dated, with the alternative that was rejected.

The tokenizer ladder we test is in the image: off-the-shelf English tokenizers, raw BPE and Unigram trained on Sanskrit, off-the-shelf Indic tokenizers, sandhi-split variants, morpheme-constrained variants, a byte-level floor, and a matched English control trained on the English side of the same corpus. That last one turned out to matter most.

Alt text: A table listing tokenizer arms T0 through T7 and the E1 English control with one-line descriptions and vocabulary sizes.

#### 3. Exp 01: the language tax

First measurement: how much do today's tokenizers charge Sanskrit?

On identical FLORES sentences in Sanskrit, Hindi and English:
- GPT-2's tokenizer needs about 12.5 tokens per Sanskrit word. The 200k-vocabulary tokenizers in Llama 4, Gemma 3 and GPT-4o need 3 to 4.
- Per identical sentence, Sanskrit costs 1.8 to 2.2 times as many tokens as English under the modern tokenizers, and 7.9 times under GPT-2.
- Sanskrit costs about 1.3 times as many tokens as Hindi, same script, no sandhi.

So the tax is real and it shrank a lot between tokenizer generations. The pre-registered prediction of "more than 5 tokens per word" was calibrated on the older generation; it holds for GPT-2 and fails for everything current.

Next post: why the per-word number in the first chart is the wrong one to headline, even though it is the one everyone quotes.

Alt text: Grouped bars of tokens per word for Sanskrit, Hindi and English under GPT-2 and GPT-4o's tokenizer, and a second chart of Sanskrit-to-English and Sanskrit-to-Hindi token ratios per tokenizer with a line at 1.0.

#### 4. Exp 01: the metric that would have lied

Tokens per word is the standard "tokenizer fairness" number. For Sanskrit it points the wrong way.

Sanskrit writes the same content in fewer, longer words: on our 1,012 FLORES sentences, 16,975 words against Hindi's 25,643. Divide tokens by that smaller word count and the ratio inflates.

The hypothesis said Sanskrit should cost more than 1.5 times Hindi's tokens. By tokens per word, the ratio is 1.7. By tokens on identical content, it is 1.3. The per-word number would have confirmed a prediction the real count refutes.

That is the whole reason this project measures tokens per proposition on parallel text and treats fertility as a diagnostic. Fewer words is not fewer tokens, in either direction.

Alt text: Chart comparing the Sanskrit-to-Hindi ratio computed from tokens per word against the ratio computed from tokens on identical sentences, with the 1.5 threshold marked; the per-word ratio is above it and the true ratio below.

#### 5. Exp 02: the sign flip that wasn't

The headline experiment: train a tokenizer on Sanskrit and count tokens per proposition against English on parallel prose.

First result: a 64k BPE trained on Sanskrit brought Sanskrit below English, 0.89 tokens for every English token under GPT-4o's tokenizer. The sign flip the hypothesis predicted.

Then the control. We trained the same algorithm, same vocabulary size, on the English side of the same corpus. Against that matched English tokenizer, Sanskrit costs 1.03 to 1.14 tokens per English token. Above parity on every prose corpus.

The flip was the English tokenizer being out of its domain, not Sanskrit being cheaper. Every "language X is more efficient" comparison that uses a general-purpose tokenizer on one side is exposed to this.

Verdict as it stands: raw subword training does not recover the density on prose. The sandhi-split and morpheme-constrained tokenizers were still untested at this point. They are the next two posts.

Alt text: Two bars for the same Sanskrit tokenizer on Sāmayik prose: 0.89 against GPT-4o's tokenizer, 1.03 against the matched English control, with a dashed line at 1.0.

#### 6. Exp 02: prose, verse, and domain

The same experiment across four corpora tells a second story.

- Contemporary prose (Sāmayik): 1.03 to 1.14.
- Out-of-domain prose: 1.06 to 1.16.
- FLORES (Wikipedia register): 1.14 to 1.22.
- Verse (Itihāsa, the epics): 0.61 to 0.66.

Verse is the only place Sanskrit falls below English, and verse is a confound: the śloka meter constrains word choice and inflates compounding, and the English translations of the epics are famously wordy. We report it and do not build on it.

Also visible: the gap is smallest on the text closest to the tokenizers' training data. Domain fit predicts this pattern about as well as any claim about the language does, which is why the control in post 5 was necessary.

Alt text: Controlled tokens-per-proposition for four tokenizer pairs across four corpora, prose first, all above 1.0 except the verse corpus.

#### 7. Exp 03: what sandhi splitting buys

Now the first piece of the proposed method: reverse sandhi before training the tokenizer.

We ran ByT5-Sanskrit, the current best segmenter, over 136,651 sentences (9.4 hours on a laptop GPU). It splits sandhi and also compounds, and it rewrites some spelling, so the split text is reconciled against the original so that nothing but boundaries changes. The example card shows one sentence at each stage.

Result, against the matched English control, split minus unsplit:
- Contemporary prose: 0.005 to 0.043 fewer tokens per English token.
- Out-of-domain prose and FLORES: 0.03 to 0.08 fewer.
- Verse: 0.006 to 0.017 more.

Real, small, and consistent on prose. Still no arm below its matched English control. The next post is about how this number was almost twice as large.

Alt text: Paired differences in tokens per proposition, split minus raw, for four tokenizer pairs across four corpora, with confidence intervals; negative on prose, positive on verse. A second image shows one Sanskrit sentence in Devanagari, in SLP1, and after sandhi splitting.

#### 8. Exp 03: the result review caught

The first run of this experiment produced the number I wanted: a sandhi-split tokenizer at 0.976, below the matched English control on prose. First time in the project.

The whole-branch review re-tokenised the characters the splitter had deleted and priced them. Punctuation attached to words, `karoti.` and `"tadā,`, had been dropped and not restored. Those characters cost tokens. They explained 77 to 110 percent of the gain.

Fixed the reconciliation, retrained, re-ran: 1.03. A second round found hyphens marking compound boundaries were being credited as discovered boundaries. Fixed that too.

Two lessons I'm keeping:
- Character retention (99 percent!) said nothing was wrong. Only re-tokenising the deleted characters showed the cost, because punctuation is expensive per character and letters are cheap.
- A review that can run code and recompute the number is worth more than one that reads the prose.

The decision log has both corrections, dated, with the withdrawn value.

Alt text: Three bars: the withdrawn first-run value 0.976 in red, the corrected value after punctuation was restored, and the unsplit baseline, with an annotation that deleted characters explained most of the original gain.

#### 9. Exp 04: alignment versus compression

The second piece of the method: forbid BPE merges that cross gold morpheme boundaries, using the Digital Corpus of Sanskrit.

We measured two things per tokenizer against a matched baseline trained on the same corpus: MorphScore (do token boundaries land on morpheme boundaries?) and tokens per proposition.

- Constrain on gold segment boundaries only: MorphScore up 0.11, token cost unchanged (confidence interval includes zero).
- Add a heuristic stem/ending boundary: MorphScore up less, and 8 percent more tokens.
- The full proposed method (split, then constrain): 8 percent more tokens.
- Gold splitting alone, no constraint: 2 percent fewer.

So the constraint does what it is meant to do to the segmentation, and it is free when the boundaries are real. The token cost came from the heuristic half of the boundary set, not from the gold half. Whether alignment pays off in training efficiency, which is what the MorphBPE paper claims, is the GPU experiment.

Alt text: Scatter of four constrained tokenizers: x is change in tokens per proposition, y is change in MorphScore; the gold-segment-only tokenizer sits at zero cost and highest alignment gain.

#### 10. Exp 04: leakage you can't hash

A process post, because this one nearly went into the paper.

The Digital Corpus of Sanskrit contains the Mahābhārata and Rāmāyaṇa. So does the Itihāsa test set we evaluate on. We excluded every evaluation sentence from training by hash, as the project rules require.

Review sampled 400 test verses and searched the training corpus for their letter strings: 19 percent were there verbatim, 36 percent by 24-letter prefix. DCS segments verses differently from Itihāsa, so the hashes matched 776 sentences and missed the rest. Every verse number for those tokenizers was contaminated, and asymmetrically, because unconstrained tokenizers memorise whole words.

Fix: a second exclusion layer that drops any training sentence sharing a 24-letter run of letters with any evaluation sentence. It removed 34,705 training sentences (20,611 of them Itihāsa test overlaps). Residual after the fix: 0 of 400.

If your evaluation set is a subset of a public corpus, exact-match exclusion is not enough. Shingle it.

Alt text: Left panel: percentage of verse test sentences found inside the training corpus before and after the shingle filter; right panel: training sentences dropped per evaluation source.

#### 11. Exp 05: what bits-per-character will settle

Everything so far counted tokens. The last experiment trains small language models with each tokenizer and measures bits per character on held-out text, which is comparable across vocabularies where perplexity is not.

Fifty-one runs: seven tokenizers on a matched corpus at 50M parameters, five at 50M and 125M on a 200M-token corpus, three seeds each, equal training bytes. About a day of A100 time, rented.

The honest expectation going in: the proposed tokenizer starts with a 24 to 33 percent in-domain token handicap, about the size of the speedup the hypothesis predicts. The one to watch is the gold-segment-only constraint, which cost nothing in tokens.

Results in a week or two. The sweep is one command; the pipeline was validated on a laptop first, including a review that found the byte-level baseline was being scored against padded vocabulary rows.

Alt text: Screenshot of the sweep's dry-run table: runs, tokens, estimated FLOPs and hours per track.

#### 12. Exp 05 result and what the paper will claim

(Draft after aggregation. Structure: one sentence on the result; the BPC bar chart; whether any arm reached the baseline's loss with fewer bytes; what the paper will and won't claim; link to the preprint when it exists.)
