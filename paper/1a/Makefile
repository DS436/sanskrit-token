# Build paper/1a. Every table and figure is generated from the tracked results snapshot in
# `results/`; nothing in this directory is edited by hand except main.tex and refs.bib.
#
#   make all     regenerate tables and figures, then compile the final PDF
#   make pdf     compile the final (de-anonymised) PDF into main.pdf
#   make review  compile the review (anonymised) PDF into main_review.pdf
#   make arxiv   build final, pack arxiv/ and paper1a_arxiv.tar.gz for upload
#   make clean   remove build artefacts (not the generated tables or figures)
#
# `mode.tex` selects the build mode. It is tracked in final mode, so a plain
# `tectonic main.tex` gives the de-anonymised paper; `make review` overwrites it for one
# build and restores it afterwards.

REPO := $(realpath ../..)
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all tables figures pdf review arxiv clean mode-final mode-review

all: tables figures pdf

tables:
	cd $(REPO) && uv run python scripts/paper_tables.py

figures:
	cd $(REPO) && uv run python scripts/paper_figures.py

# The two mode files. Kept as recipes rather than tracked variants so that mode.tex has
# exactly one tracked state (final) and `make review` is self-restoring.
mode-final:
	@printf '%s\n' \
	  '% Build mode for paper/1a. `make pdf` and `make arxiv` build with this file as tracked;' \
	  '% `make review` overwrites it, builds main_review.pdf, and restores it. The tracked' \
	  '% default is final, so a plain `tectonic main.tex` produces the de-anonymised paper.' \
	  '%' \
	  '% \finalmodetrue  -> author block shown, acknowledgements printed, repository URL given.' \
	  '% \finalmodefalse -> anonymised author block, line numbers, no acknowledgements, no URL.' \
	  '\newif\iffinalmode' \
	  '\finalmodetrue' \
	  '\usepackage[final]{acl}' > mode.tex

mode-review:
	@printf '%s\n' \
	  '% Build mode for paper/1a: review. Written by `make review`, which restores the' \
	  '% tracked final-mode file afterwards. Do not commit this state.' \
	  '\newif\iffinalmode' \
	  '\finalmodefalse' \
	  '\usepackage[review]{acl}' > mode.tex

pdf: mode-final
ifdef TECTONIC
	tectonic --keep-intermediates main.tex
else
	latexmk -pdf main.tex
endif

# Built into a scratch output directory so that main.pdf is never left holding the
# anonymised build: main.pdf is always the final PDF, whatever was run last.
review: mode-review
	mkdir -p review_build
ifdef TECTONIC
	tectonic --outdir review_build main.tex
else
	latexmk -pdf -outdir=review_build main.tex
endif
	cp review_build/main.pdf main_review.pdf
	rm -rf review_build
	@$(MAKE) --no-print-directory mode-final
	@echo "wrote main_review.pdf; mode.tex restored to final, main.pdf untouched"

# arXiv wants the .bbl, not the .bib: its pdflatex run does not invoke BibTeX. refs.bib is
# shipped as well, because a standalone `tectonic main.tex` does run BibTeX when the .aux
# names a \bibdata, and would fail without it. The PNG copies of the figures are left out.
arxiv: pdf
	rm -rf arxiv paper1a_arxiv.tar.gz
	mkdir -p arxiv/tables arxiv/figures
	cp main.tex mode.tex main.bbl acl.sty acl_natbib.bst refs.bib arxiv/
	cp tables/*.tex arxiv/tables/
	cp figures/*.pdf arxiv/figures/
	@printf '%s\n' \
	  'Main file: main.tex' \
	  '' \
	  'Compiles with pdflatex. The bibliography is shipped pre-compiled as main.bbl, so' \
	  'no BibTeX run is required; refs.bib is included only so that the source also' \
	  'builds under engines that re-run BibTeX. acl.sty issues \bibliographystyle itself.' \
	  '' \
	  'main.tex reads mode.tex, which selects final mode (de-anonymised, no line numbers).' \
	  > arxiv/00README.txt
	tar czf paper1a_arxiv.tar.gz -C arxiv .
	@echo "wrote paper1a_arxiv.tar.gz"

clean:
	rm -rf review_build
	rm -rf arxiv paper1a_arxiv.tar.gz
	rm -f main.pdf main_review.pdf main.aux main.log main.out main.bbl main.blg \
	      main.fls main.fdb_latexmk main.synctex.gz main.xdv
