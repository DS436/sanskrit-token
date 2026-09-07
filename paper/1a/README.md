# Paper 1a — the measurement paper (RQ1 and RQ2)

**Status: draft.** The prose is written except for §5.5 (tokens per proposition by
sentence length), which carries a `% TODO(1b-of-task)` marker and a visible draft line in
the PDF: its table is generated and final, and what is missing is the reading of it.
§5.6 (Rényi efficiency) is written.

**Format: long paper.** The body runs to eight pages excluding references and appendix
(References begins partway down page 8), against the eight-page limit for a long paper at
ACL venues; arXiv imposes no limit. Nothing is cut to fit, but there is now no slack: the
§5.5 prose still to be written will push the body over the ACL limit, and something will
have to move to the appendix at that point. A four-page short version would move Tables 2
and 3 (fertility and compression; deployed-practice TPP) to the appendix, leaving Table 1
(parity), Table 4 (the matched control) and the two main figures in the body.

*Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per
Proposition.* Scope is Experiments 01 and 02 only. The proposed sandhi-aware and
morpheme-constrained tokenizers and the language-model training are deliberately out of
scope and are named once, in the Conclusion, as follow-up work.

## Build

```bash
cd paper/1a && make all      # regenerate tables and figures, then compile
```

`make all` runs `scripts/paper_tables.py` and `scripts/paper_figures.py` from the
repository root and then `make pdf`. `make pdf` uses `tectonic` when it is on `PATH` and
falls back to `latexmk -pdf`; the first `tectonic` run needs network access, because it
fetches the TeX packages it needs. `make clean` removes the build artefacts and leaves the
generated tables and figures alone.

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
  numbers back out of `parity.tex` and `tpp_controlled.tex` and checks them against the
  JSON at the printed precision, and asserts that every `\num...` macro `main.tex` uses is
  one the generator emits. It also fails if the committed tables are stale.

Both generators read `results/` and never write to it.

## Review and final

The document is in **review** mode: `\usepackage[review]{acl}`, which anonymises the
author block and adds line numbers. For the camera-ready, change that one line to
`\usepackage[final]{acl}` and reveal the repository URL, which is presently a comment
beside the "anonymised repository" sentence at the end of §4.

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
