# arXiv submission sheet, paper 1a

This file holds every field the arXiv web submission form asks for, in the form it should
be pasted. The upload itself is `paper1a_arxiv.tar.gz`, produced by `make arxiv` in this
directory (see [`README.md`](README.md), section "arXiv bundle", for what goes into it).
Nothing below needs retyping or reformatting: each fenced block is the literal value for
one form field. The numbers in the abstract are LaTeX macros in `main.tex`, so the block
here has already been expanded from `tables/numbers.tex` and checked word for word against
the compiled PDF.

## Title

```
Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per Proposition
```

## Authors

```
Devansh Sharma
```

arXiv accepts one author per line or a comma-separated list; with a single author either
form gives the same result. The affiliation field can be left blank for an independent
researcher. The PDF's author block already states "Independent researcher" and the contact
address, so nothing is lost by leaving the form field empty.

## Abstract

```
Sanskrit fuses case, number, person and tense into word endings and chains clauses into compounds, so it is information-dense per word. Whether that density survives subword tokenization is a separate question, to be asked per unit of meaning rather than per word. On identical FLORES-200 devtest content, Sanskrit costs 1.774-2.187 times the English tokens under deployed tokenizers with vocabularies of 200,019 ids or more, but only 1.325-1.353 times the Hindi tokens. Against a deployed English tokenizer, Sanskrit-trained BPE arms then look cheaper per proposition than English on contemporary prose (0.887). Against a matched English control, the same algorithm and vocabulary trained on the English side of the same corpus, that flip disappears: at 32,000 and 64,000 pieces all 8 matched pairs, each size-matched arm against both a pair-matched and a byte-matched control, sit above 1.0 on prose with 95% intervals excluding it. The gap closes as the vocabulary grows: at 128,000 pieces the BPE pair reads 0.983 in domain while staying above parity out of domain (1.025) and on FLORES (1.116). The ratio factorises into a character-length ratio and a tokens-per-character ratio, the second near 1 throughout: what survives matched tokenization is character-level length, which Sanskrit prose lacks over English in SLP1 (1.028) and Sanskrit verse has (0.596). The robust statement is about deployed practice: on contemporary prose and on FLORES, with the Sanskrit side in SLP1 against the deployed o200k English pivot, Sanskrit costs 1.831-2.899 English tokens per proposition under the tokenizers people actually ship. Code, the results snapshot and every table here are public.
```

256 words, one paragraph, no hard line breaks.

How it was produced, and how to reproduce the check:

- Every `\num...` macro was expanded to its literal value from `tables/numbers.tex`. There
  are eighteen of them in the abstract.
- All markup was stripped: `\emph{cheaper}` to `cheaper`, `\texttt{o200k}` to `o200k`,
  `\%` to `%`, `~` and `\,` to a plain space.
- Every `--` was written as the plain ASCII hyphen `-` rather than an en dash. The en dash
  is typographically correct for a numeric range, but the ASCII hyphen survives every
  encoding path the form and the arXiv listing put the text through, so it is the safer
  choice here.
- The block contains no `\`, no `{`, no `}` and no `$`. Verify by extracting the block and
  running `grep -c '[\\{}$]'` on it, which must print `0`.
- Ground truth for the wording is the compiled PDF, not `main.tex`. The block was diffed
  word by word against `pdftotext -f 1 -l 1 main.pdf -` after normalising the `fl`
  ligature, the en dashes and the line-break hyphenation. Result: 256 words on both sides,
  zero differing words, and the two texts are identical as strings once whitespace and
  hyphens are removed.

## Comments

```
8 pages, 4 figures, 18 tables. Code, data pipeline and the full results snapshot: https://github.com/DS436/sanskrit-token
```

The counts are read off the final PDF. The body runs to 8 pages: the Conclusion ends on
page 8, and Limitations and the References start on page 9. The full PDF is 19 pages
including Limitations, References and Appendices A to D. There are 4 figures and 18 tables
in total, counting those in the appendices.

## Primary category

```
cs.CL
```

Computation and Language. The paper is a tokenizer measurement study on parallel text and
sits squarely in that category.

## Cross-list

No cross-list is warranted.

The obvious candidate is `cs.LG`, and it is not defensible from this paper's content. No
model is trained, no learning method is proposed or analysed, and the paper states
explicitly (Section 2, "What tokenizer metrics predict", and Section 6.5) that it claims
nothing about what a token count buys downstream. A `cs.LG` cross-list would advertise a
machine-learning contribution that is not here. `cs.IT` is similarly out of scope: the
paper measures token counts and character ratios and deliberately makes no
information-theoretic claim. Listing in `cs.CL` alone is accurate, and padding the list
would only dilute it.

## MSC class / ACM class

Both fields are optional and should be left blank. They are legacy classification schemes
for mathematics and for older computing literature, and `cs.CL` already carries the
subject information a reader needs.

## License

```
CC BY 4.0
```

Choose `CC BY 4.0` (Creative Commons Attribution 4.0). It matches the MIT licence on the
repository's code, and it lets the tables and figures be reused with attribution, which is
what a paper whose whole artefact story is a public results snapshot wants.

The alternative, for the record, is arXiv's own non-exclusive licence to distribute, which
grants arXiv the right to distribute the paper but reserves everything else. It is the
right choice only if a publisher later requires it; nothing here does.

## Journal reference and DOI

Leave both blank at submission. They are for work already published elsewhere. If this
paper is accepted at a venue later, the fields are filled in afterwards through the
"journal ref" update on the arXiv abstract page, which does not require a new version.

## Pre-upload checklist

- [ ] The bundle is current. Re-run after any edit to the paper, the tables or the
      figures: `cd paper/1a && make clean && make arxiv`
- [ ] The final PDF is the de-anonymised build:
      `pdftotext paper/1a/main.pdf - | grep -c -E "Sharma|github.com/DS436"` returns at
      least 2 (it currently returns 2)
- [ ] The bundle compiles standalone:
      `d=$(mktemp -d) && tar xzf paper/1a/paper1a_arxiv.tar.gz -C "$d" && (cd "$d" && tectonic main.tex)`
- [ ] The repository is public and the link in the paper resolves:
      `curl -o /dev/null -s -w '%{http_code}\n' https://github.com/DS436/sanskrit-token`
      returns `200`
- [ ] The Acknowledgements section is present in the final build:
      `pdftotext paper/1a/main.pdf - | grep -c '^Acknowledgements$'` returns 1
- [ ] No `% TODO` remains in the source: `grep -c '% TODO' paper/1a/main.tex` returns 0
- [ ] The arXiv-generated PDF matches `main.pdf` on the author block, the Acknowledgements
      and the repository URL, checked on the submission preview before announcing

## After submission

arXiv assigns the identifier as soon as the submission is accepted into the queue, but
announces it on the next mailing cycle, which is roughly one business day and runs on
weekdays only. The paper is not publicly listed until that cycle.

A submission can be replaced with a new version at any time, and every earlier version
stays permanently visible at its own `vN` URL. Version 1 is therefore a public record, not
a draft, so the pre-upload checklist above is worth completing in full.

Once the identifier is known, add it in two places: the root `CITATION.cff`, and the root
`README.md`.
