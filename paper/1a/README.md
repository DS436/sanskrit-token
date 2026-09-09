# Paper 1a — the measurement paper (RQ1 and RQ2)

**Status: review wave 1 complete, results included; final claims trace done.** The
structural and textual changes an ARR-style review asked for are in (Related Work as §2, an
unnumbered Limitations section after the Conclusion, the Rényi subsection and the
pre-registration table in appendices, a table of translation direction and the sign of its
bias per corpus), and the five
`% TODO(wave1-results)` markers are gone: the byte-matched control and the 128k vocabulary
sweep are in §6.3, the byte-level reference arm `T7_byt5` in §5, the verse side
decomposition in §6.4, and the block-resampled intervals in §3 and Appendix A.2. The
acknowledgements print in final mode only. A final adversarial claims trace then checked
every measured sentence against the snapshot and tightened the quantifiers it found
over-broad (which arms carry bootstrap intervals, which corpora the deployed-practice
statement covers, which trained arm is cheapest against the deployed pivot, and which rows
change a verdict under the block bootstrap).

**The central finding, as the paper now states it.** Against a matched English control the
prose flip disappears at 32k and 64k pieces, under a pair-matched control and a
byte-matched one alike (the byte-matched family subsamples the English training text to the
Sanskrit corpus's byte count; every controlled ratio moves by at most 0.025 and no verdict
changes). The controlled penalty shrinks with vocabulary size: at 128k the size-matched BPE
pair reads just below parity on in-domain prose and stays above it out of domain and on
FLORES, so the negative result is scoped to the sizes it was measured at. Tokens per
proposition factorises exactly into a character ratio and a tokens-per-character ratio, and
the second is near 1 for every matched pair, so the verse crossing lives in the character
ratio, which these corpora cannot attribute to meter rather than to a verbose 19th-century
English translation.

**Format: long paper.** 19 pages in final mode and 20 in review mode; the body runs to
eight pages, with the Conclusion ending on page 8 and the Limitations section, which ARR
excludes from that limit, starting on page 9 alongside the References. arXiv imposes no
limit. `placeins` puts a
`\FloatBarrier` before the bibliography, so no body float is deferred into the references,
and the float parameters are relaxed in the preamble because the defaults pushed a figure
several pages past its first reference. To hold the body to eight pages against the new
results, two things moved to the appendix: the deployed-practice table (§6.2 keeps one
sentence of macros and points at Appendix A, which already carried the same rows for all
four corpora with a second English pivot) and the parity table (Figure 1 draws the same
numbers in the body). Fertility and compression sit in the appendix, where fertility
belongs: it is reported and never led with.

**Two-sided length strata, in the appendix.** §6.5 bins each pair twice, once on the
English side's word count and once on the Sanskrit side's, because binning selects the
binned side's wordiness into the bin and so tilts the ratio in a known direction. Both
stratifications are two halves of a single table rather than a facing pair, because the
section's whole argument is that neither is read alone; every within-corpus gradient
reverses between them, which is why the section draws no length claim, while the
verse-below-prose separation survives both. That separation is now read as the character
ratio of §6.4 surviving a control for length on either side. The table and its figure sit
in Appendix A.1 with the all-corpora versions; nothing in them is bolded, and a bin holding
fewer than ten pairs prints its dagger alone.

*Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per
Proposition.* Scope is Experiments 01 and 02 only. The proposed sandhi-aware and
morpheme-constrained tokenizers and the language-model training are deliberately out of
scope and are named once, in the Conclusion, as follow-up work.

## Build

```bash
cd paper/1a && make all       # regenerate tables and figures, then compile main.pdf
make pdf                      # final (de-anonymised) PDF -> main.pdf
make review                   # review (anonymised) PDF   -> main_review.pdf
make arxiv                    # final PDF + arxiv/ + paper1a_arxiv.tar.gz
make clean                    # remove build products
```

`make all` runs `scripts/paper_tables.py` and `scripts/paper_figures.py` from the
repository root and then `make pdf`. The compile uses `tectonic` when it is on `PATH` and
falls back to `latexmk -pdf`; the first `tectonic` run needs network access, because it
fetches the TeX packages it needs. `make clean` removes the build products and leaves the
generated tables and figures alone.

`main.pdf` is always the final PDF: `make review` compiles into a scratch output directory
and copies the result to `main_review.pdf`, so the anonymised build can never end up in
`main.pdf` by accident.

## Mode switch

`mode.tex` is the one file that decides which version is built. It loads `acl.sty` with
either `[final]` or `[review]` and sets a `\iffinalmode` switch that `main.tex` reads in
two places: the Acknowledgements section, which prints only in final mode (`acl.sty`
anonymises the author block but does not suppress that section), and the last sentence of
§5, which gives `https://github.com/DS436/sanskrit-token` in final mode and says "an
anonymised repository" in review mode.

`mode.tex` is tracked in **final** mode, so a plain `tectonic main.tex` produces the
de-anonymised paper. `make review` overwrites it, builds, and restores it, so the tracked
state is never left anonymised. The two states are written by the `mode-final` and
`mode-review` recipes in the `Makefile`; do not edit `mode.tex` by hand for a one-off
build, run the target.

Anonymity is checked mechanically, and both checks are part of the release routine:

```bash
pdftotext main.pdf - | grep -c -E "Sharma|github.com/DS436"        # must be >= 2
pdftotext main_review.pdf - | grep -c -i -E "sharma|gmail|github"  # must be 0
```

## arXiv bundle

`make arxiv` builds final mode with `--keep-intermediates` so that `main.bbl` exists, then
packs `arxiv/` and `paper1a_arxiv.tar.gz` with `main.tex`, `mode.tex`, `main.bbl`,
`acl.sty`, `acl_natbib.bst`, `refs.bib`, `tables/*.tex`, `figures/*.pdf` (the PNG copies
are left out) and a `00README.txt` naming the main file.

arXiv runs pdflatex and does **not** run BibTeX, which is why the compiled `main.bbl` is
shipped. `refs.bib` is shipped as well only because a standalone `tectonic main.tex` *does*
re-run BibTeX when the `.aux` names a `\bibdata`, and fails without it; arXiv itself
ignores the `.bib`. The bundle was verified by extracting the tarball into an empty
directory and compiling it there with nothing else present.

Every field the arXiv web form asks for is written out, paste-ready, in
[`SUBMISSION.md`](SUBMISSION.md), including the abstract as plain text with the macros
already expanded and diffed against the PDF.

### Upload steps

`SUBMISSION.md` is the single source of truth for every value the web form asks for. Do not
copy any of those values into this file; read them there when filling the form in.

1. `make clean && make arxiv`, then upload `paper1a_arxiv.tar.gz`. This is the only
   artefact to upload; the PDF is not submitted, because arXiv compiles the source itself.
2. Take the title and the author list from the "Title" and "Authors" sections of
   `SUBMISSION.md`, and the abstract from its "Abstract" section, which is already
   expanded from the macros and diffed against the compiled PDF.
3. Take the primary category and the cross-list decision from the "Primary category" and
   "Cross-list" sections of `SUBMISSION.md`, the licence from its "License" section, the
   comments string from its "Comments" section, and leave blank the fields its
   "MSC class / ACM class" and "Journal reference and DOI" sections say to leave blank.
4. No BibTeX run is needed on arXiv's side. arXiv runs pdflatex only, and the bundle ships
   the compiled `main.bbl` for exactly that reason.
5. Check the arXiv-generated PDF, not `main.pdf`, on the submission preview before
   announcing, in particular that the author block, the acknowledgements and the repository
   URL are all present. `main.pdf` is the reference to compare it against.
6. Re-run `make clean && make arxiv` and re-upload after any edit to the paper, the tables
   or the figures. The tarball is a build product and goes stale silently.

## Where the numbers come from

Nothing in `main.tex` contains a typed-in measurement.

- Every table under `tables/` is written by `scripts/paper_tables.py` from
  `results/01_baseline_penalty/results.json` and `results/02_tpp_parallel/results.json`,
  the tracked snapshot described in [`results/README.md`](../../results/README.md).
- Every figure under `figures/` is written by `scripts/paper_figures.py` from the same two
  files, as both `.pdf` and `.png`.
- Every number in the prose is a macro from `tables/numbers.tex`, also generated. The list
  of macros is the `MACROS` constant in `scripts/paper_tables.py`.
- `tests/test_paper_tables.py` runs both generators against the snapshot, parses the
  numbers back out of `parity.tex`, `tpp_controlled.tex` and the two length tables and
  checks them against the JSON at the printed precision, and asserts that every `\num...`
  macro `main.tex` uses is one the generator emits. It recomputes each of §6.5's macros
  from the JSON, and fails if either ordering that section asserts in words (every
  gradient reverses; verse sits below prose at every jointly populated bin) stops holding.
  It also checks that the thin bin is suppressed in the body and kept in the appendix,
  that no length cell is bolded, that the Hindi table's SLP1 artefact column is gone, that
  every undersized control is daggered, and that only the twelve reported matched pairs
  reach a table: the snapshot carries pairs whose prose is not written yet, and
  `CONTROLLED_PAIRS` in `scripts/paper_tables.py` (with `LENGTH_PAIRS` for the four the
  strata cover, and its twin in `scripts/paper_figures.py`) is the list that decides what
  is printed. It
  also fails if the committed tables are stale. The wave-1 additions have their own
  checks: the byte-matched moves, the vocabulary sweep, the side decomposition and the
  block intervals are each recomputed from the JSON, and a prose-claims test asserts the
  ordinal statements the manuscript makes in words (no verdict changes between the two
  controls; the BPE ratio falls with vocabulary size on every corpus, strictly at every
  step in seven of the eight sequences; the 128k BPE pair is below 1.0 in domain under
  both controls; the density ratios stay in the band §6.4 quotes; `T7_byt5` is exactly the
  two sides' byte ratio; and no Itihāsa block interval is narrower than its i.i.d. one).

  The training-corpus byte counts §6.3 and the Limitations section quote are constants in
  `scripts/paper_tables.py`, like the training-split sizes: the corpora live under the
  gitignored `data/processed/` and the tracked snapshot records no corpus size, so the
  generator cannot read them. A test greps `experiments/02_tpp_parallel/README.md` for
  each of them.

Both generators read `results/` and never write to it.

## A note on the bibliography style

`acl.sty` issues `\bibliographystyle{acl_natbib}` itself, so `main.tex` must not repeat
it; a second `\bibstyle` in the `.aux` makes BibTeX reject the file.

## Style files

`acl.sty` and `acl_natbib.bst` were fetched on 2026-09-07 from
<https://github.com/acl-org/acl-style-files>, `master` at commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`. The most recent commit touching `acl.sty`
itself at that point was `24272820f6c457a34a6ac61d74b7488466175e41` (2025-08-19). The
stock `acl_latex_template.tex` was fetched from the same commit but is not kept here: it
is unmodified upstream boilerplate with placeholder authors, is included by nothing, and
would not compile in place because it cites `.bib` files this directory does not have.

## Bibliography

`refs.bib` carries `note = {verify}` on every entry whose bibliographic details could not
be confirmed from the project's own design document, `data/README.md`, or the arXiv record
of the work. No DOI, page range or volume number is stated unless it was read off one of
those sources. On 2026-09-07 the five remaining `verify` entries were checked against a
primary record (ACL Anthology, AAAI's AI Magazine, CiNii Research, the Sarvam-1 model
card) and the note removed; the header comment of `refs.bib` names the source used for
each. The AI Magazine record publishes a first page and no range, so only the first page
is recorded.

On 2026-09-08, before the arXiv posting, three more entries were checked. `maheshwari-etal-2024-samayik`
gained the Anthology spelling of its sixth author, plus month, address, publisher, pages
and URL; `arnett-etal-2025-inequities` was confirmed to be a NeurIPS 2025 main-conference
paper rather than a preprint, so its booktitle stands, and the OpenReview URL was added;
`shravan-2026-brahmic` had its arXiv identifier confirmed to resolve, so the eprint is
kept beside the model card. The `refs.bib` header comment records the URL read for each.

On 2026-09-08 the Related Work section added eighteen entries, every one read off a
primary record before being cited: the ACL Anthology BibTeX for Mielke et al. (2019),
Bugliarello et al. (2020), Rust et al. (2021), Limisiewicz et al. (2023), Bostrom and
Durrett (2020), Gowda and May (2020), Schmidt et al. (2024), Goldman et al. (2024), Ali et
al. (2024), Uzan et al. (2024), Koppel and Ordan (2011), Graham et al. (2020), Klein and
Tsarfaty (2020) and Arnett and Bergen (2025); and the Crossref DOI record, the publisher's
own deposited metadata, for Coupé et al. (2019), Pellegrino et al. (2011), Volansky et al.
(2015) and Toraman et al. (2023), whose publisher pages refuse automated requests. Each
entry carries the URL it was verified against in a comment above it.

The Sāmayik row of `data/README.md` used to attribute the dataset to "Aralikatte et al.,
LREC-COLING 2024", which is the Itihāsa author list pasted into the wrong row. That cell
now reads "Maheshwari et al.", matching the Anthology record `2024.lrec-main.1245` and the
paper's bibliography.
