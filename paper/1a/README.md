# Paper 1a — the measurement paper (RQ1 and RQ2)

**Status: ready for arXiv; workshop venue to be chosen.** Every section is written, §5.5
(tokens per proposition by sentence length) and §5.6 (Rényi efficiency) included. The
acknowledgements are written and print in final mode only; no `% TODO` or `% verify`
marker is left in `main.tex`.

**Format: long paper.** 15 pages in total, in both modes. The body still runs to eight
pages excluding references and appendix: Conclusion and Limitations both end on page 8,
against the eight-page limit for a long paper at ACL venues, and what sits above
References at the top of page 9 is Table 5 floated over from page 8, not body prose.
arXiv imposes no limit. Writing §5.5 and adding its
figure used the slack that was left, so the fertility and compression table moved to the
appendix (Table 13, beside the full-precision version of itself that already sat there),
where fertility belongs anyway: it is reported and never led with. The length strata for
all four corpora moved with it, into Appendix A.1; the body tabulates only the two primary
corpora. A four-page short version would additionally move Table 2 (deployed-practice TPP)
and Tables 4 and 5 (the length strata), leaving Table 1 (parity), Table 3 (the matched
control) and the figures in the body.

**Two-sided length strata.** §5.5 bins each pair twice, once on the English side's word
count and once on the Sanskrit side's (Tables 4 and 5, Figure 3), because binning selects
the binned side's wordiness into the bin and so tilts the ratio in a known direction;
every within-corpus gradient reverses between the two, which is why the section draws no
length claim from them, while the verse-below-prose separation survives both.

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
§4, which gives `https://github.com/DS436/sanskrit-token` in final mode and says "an
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

### Upload steps

1. `make clean && make arxiv`, then upload `paper1a_arxiv.tar.gz`.
2. Primary category **cs.CL** (Computation and Language). No cross-list is needed.
3. License: **CC BY 4.0**, the most permissive of the offered set and the one that lets the
   tables and figures be reused with attribution.
4. Title and authors as in the PDF. Paste the abstract from the PDF itself, not from
   `main.tex`, because the prose numbers are macros: `pdftotext -f 1 -l 1 main.pdf -` and
   take the abstract block, then strip the line breaks.
5. Comments field: `Code and results: https://github.com/DS436/sanskrit-token`.
6. Check the arXiv-generated PDF against `main.pdf` before announcing, in particular that
   the author block, the acknowledgements and the repository URL are all present.

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
  numbers back out of `parity.tex`, `tpp_controlled.tex` and the three length tables and
  checks them against the JSON at the printed precision, and asserts that every `\num...`
  macro `main.tex` uses is one the generator emits. It recomputes each of §5.5's macros
  from the JSON, and fails if either ordering that section asserts in words (every
  gradient reverses; verse sits below prose at every jointly populated bin) stops holding.
  It also fails if the committed tables are stale.

Both generators read `results/` and never write to it.

## A note on the bibliography style

`acl.sty` issues `\bibliographystyle{acl_natbib}` itself, so `main.tex` must not repeat
it; a second `\bibstyle` in the `.aux` makes BibTeX reject the file.

## Style files

`acl.sty`, `acl_natbib.bst` and `acl_latex_template.tex` (the last kept for reference and
never compiled) were fetched on 2026-09-07 from
<https://github.com/acl-org/acl-style-files>, `master` at commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`. The most recent commit touching `acl.sty`
itself at that point was `24272820f6c457a34a6ac61d74b7488466175e41` (2025-08-19).

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

`data/README.md` attributes the Sāmayik dataset to "Aralikatte et al., LREC-COLING 2024",
which is the Itihāsa author list pasted into the wrong row. The primary record says
Maheshwari et al.; the paper's bibliography is right and that data row is not. It is left
for a `data:` commit rather than fixed here.
