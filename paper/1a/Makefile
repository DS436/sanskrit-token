# Build paper/1a. Every table and figure is generated from the tracked results snapshot in
# `results/`; nothing in this directory is edited by hand except main.tex and refs.bib.
#
#   make all    regenerate tables and figures, then compile
#   make pdf    compile only
#   make clean  remove build artefacts (not the generated tables or figures)

REPO := $(realpath ../..)
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all tables figures pdf clean

all: tables figures pdf

tables:
	cd $(REPO) && uv run python scripts/paper_tables.py

figures:
	cd $(REPO) && uv run python scripts/paper_figures.py

pdf:
ifdef TECTONIC
	tectonic main.tex
else
	latexmk -pdf main.tex
endif

clean:
	rm -f main.pdf main.aux main.log main.out main.bbl main.blg main.fls \
	      main.fdb_latexmk main.synctex.gz
