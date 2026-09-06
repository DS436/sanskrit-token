"""Sangraha loader: AI4Bharat's verified Sanskrit web/PDF text, the bulk of corpus M1.

Sangraha (HF `ai4bharat/sangraha`, CC BY 4.0, ungated) is the enlarged monolingual corpus
Experiment 05's Track 2 trains on (docs/decisions.md, 2026-09-05, "M1 monolingual corpus").
Only the **verified** Sanskrit split is used: `verified/san/data-*.parquet`, human-written
text extracted from PDFs and web pages. The sibling `synthetic/san_Deva/wiki_*.parquet`
files are machine-translated Wikipedia and are **excluded by name** — training a Sanskrit
LM on machine translation would measure the translator, not the language — and
`unverified/` is excluded for the same reason the dataset card separates it.

`SANGRAHA_REVISION` pins the dataset revision, as `samayik.py` and `itihasa.py` pin a
commit: every download URL carries it, so a re-run fetches the exact bytes this project was
built against. Files land under `data/raw/sangraha/` (gitignored) and a file already there
is read without touching the network at all, which is what makes the corpus build resumable
and every test offline.

Parquet is read a batch at a time (`ParquetFile.iter_batches`) rather than whole: one file
is ~380 MB compressed and ~350M characters decoded, and the corpus builder streams
throughout, so nothing here may hold a file in memory.

**The text is OCR.** `type` is `pdf` for most documents and the extraction carries the
usual OCR damage (broken conjuncts, dropped diacritics). That is recorded here and in
`data/README.md` because it bounds what a bits-per-character number on this corpus means;
no cleaning beyond the line rules below is applied, since any threshold would be a research
decision rather than an engineering one.
"""

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from sanskrit_tok.data.quality import real_words

__all__ = [
    "DANDA",
    "DOUBLE_DANDA",
    "MIN_WORDS_PER_LINE",
    "SANGRAHA_LICENCE",
    "SANGRAHA_REPO",
    "SANGRAHA_REVISION",
    "SANGRAHA_TEXT_COLUMN",
    "VERIFIED_SANSKRIT_PREFIX",
    "assert_verified_sanskrit_files",
    "documents_to_lines",
    "has_devanagari_letter",
    "iter_sangraha_lines",
    "iter_sangraha_sanskrit",
    "list_verified_sanskrit_files",
    "resolve_sanskrit_files",
    "sangraha_file_path",
    "sangraha_file_sizes",
    "verified_sanskrit_files",
]

logger = logging.getLogger(__name__)

SANGRAHA_REPO = "ai4bharat/sangraha"

#: Revision sha of the dataset repository, observed 2026-09-05 via
#: `huggingface_hub.HfApi().dataset_info("ai4bharat/sangraha").sha`. Changing it means
#: re-downloading and rebuilding the M1 corpus.
SANGRAHA_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"

SANGRAHA_LICENCE = "CC BY 4.0"

#: The only directory this loader will read. `synthetic/` (machine-translated) and
#: `unverified/` are excluded by construction, not by a flag a caller could flip.
VERIFIED_SANSKRIT_PREFIX = "verified/san/"

#: The document column of a Sangraha parquet (the others are `doc_id` and `type`).
SANGRAHA_TEXT_COLUMN = "text"

#: Rows decoded at a time. Small enough that peak memory is a batch, not a row group.
_BATCH_SIZE = 500

DANDA = "।"
DOUBLE_DANDA = "॥"

#: A line with fewer whitespace-delimited words than this has no word boundary to tokenise
#: around and no proposition to count — the same floor `experiments/04_morph_constrained`
#: applies to DCS sentences (`min_words: 2`).
MIN_WORDS_PER_LINE = 2

#: The Devanagari block. A line with no *letter* from it (digits and dandas are not
#: letters) is page furniture — a page number, a Latin caption, a rule of dashes.
_DEVANAGARI_START = "ऀ"
_DEVANAGARI_END = "ॿ"


def has_devanagari_letter(text: str) -> bool:
    """Whether `text` contains at least one letter of the Devanagari block.

    `str.isalpha()` is what separates a letter from the block's digits (०-९), its dandas
    and its combining signs, so a line of verse numbering does not pass for text.
    """
    return any(
        _DEVANAGARI_START <= character <= _DEVANAGARI_END and character.isalpha()
        for character in text
    )


def _split_on_danda(text: str) -> list[str]:
    """`text` cut after every danda, the danda staying with the line it ends.

    Sanskrit's sentence terminator is the danda, and Sangraha's documents are whole pages
    with the sentence structure inside the line rather than at its end. Keeping the danda
    attached matters for what is trained on: it is a character the tokenizer must learn,
    and dropping it would make these lines differ from DCS's and the parallel corpora's,
    which keep their punctuation.
    """
    pieces: list[str] = []
    start = 0
    for position, character in enumerate(text):
        if character in (DANDA, DOUBLE_DANDA):
            pieces.append(text[start : position + 1])
            start = position + 1
    pieces.append(text[start:])
    return pieces


def documents_to_lines(document: str) -> list[str]:
    """One document split into corpus lines, in order.

    Newlines first, then dandas (`।`, `॥`), which is the splitting rule Experiment 05's
    plan fixes. Each candidate has its internal whitespace collapsed to single spaces —
    the corpus files are newline-delimited, so "one sentence per line" has to be true by
    construction, not by hope — and is then kept only if it has at least
    `MIN_WORDS_PER_LINE` words and at least one Devanagari letter.
    """
    lines: list[str] = []
    for raw_line in document.split("\n"):
        for piece in _split_on_danda(raw_line):
            # `real_words`, not `piece.split()`: a danda is not a word, so `राम ।` is a
            # one-word line and does not reach the corpus (docs/decisions.md, 2026-09-05,
            # "Sangraha quality filter calibrated on a sample", rule 5). Counting the danda
            # was the base splitter's one-word-line gap.
            words = real_words(piece)
            if len(words) < MIN_WORDS_PER_LINE:
                continue
            candidate = " ".join(piece.split())
            if not has_devanagari_letter(candidate):
                continue
            lines.append(candidate)
    return lines


# ------------------------------------------------------------------------ repo files


def verified_sanskrit_files(repo_files: Iterable[str]) -> list[str]:
    """The `verified/san/*.parquet` entries of `repo_files`, sorted.

    Pure, so the selection rule is testable without the network: this is the function that
    guarantees `synthetic/` never reaches the corpus.
    """
    return sorted(
        name
        for name in repo_files
        if name.startswith(VERIFIED_SANSKRIT_PREFIX) and name.endswith(".parquet")
    )


def list_verified_sanskrit_files(revision: str = SANGRAHA_REVISION) -> list[str]:
    """The verified Sanskrit parquet paths at `revision`, from the Hub's file listing.

    Network access, once per build. `verified_sanskrit_files` does the selecting.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(SANGRAHA_REPO, repo_type="dataset", revision=revision)
    selected = verified_sanskrit_files(files)
    logger.info(
        "Sangraha %s: %d verified Sanskrit file(s) of %d in the repository",
        revision[:12],
        len(selected),
        len(files),
    )
    return selected


def assert_verified_sanskrit_files(files: Iterable[str]) -> list[str]:
    """`files` unchanged, or `ValueError` naming every entry outside `verified/san/`.

    `verified_sanskrit_files` guarantees the *default* file list contains no machine
    translation, but a caller may pass `files` explicitly — `corpus.yaml` has a `files:`
    key — and nothing stopped that list from naming `synthetic/san_Deva/wiki_0.parquet`.
    Training a Sanskrit LM on machine-translated Wikipedia would measure the translator,
    and it would do so silently, so an out-of-prefix name is an error rather than a
    warning. The offenders are named in the message: a config typo is what this catches
    in practice, and a message that says only "invalid file list" would not help.
    """
    names = list(files)
    outside = [
        name
        for name in names
        if not (name.startswith(VERIFIED_SANSKRIT_PREFIX) and name.endswith(".parquet"))
    ]
    if outside:
        raise ValueError(
            f"{len(outside)} Sangraha file(s) are not verified Sanskrit parquet under "
            f"{VERIFIED_SANSKRIT_PREFIX!r} and will not be read: {outside}"
        )
    return names


def resolve_sanskrit_files(
    *,
    files: Sequence[str] | None = None,
    revision: str = SANGRAHA_REVISION,
    max_files: int | None = None,
) -> list[str]:
    """The file list a read will actually use: validated, and truncated by `max_files`.

    `files` defaults to `list_verified_sanskrit_files(revision)` (one network call). A
    caller-supplied list is checked by `assert_verified_sanskrit_files` and otherwise kept
    in the order given. `max_files` takes whole files from the front, so two runs with the
    same cap see the same documents.

    Separated from `iter_sangraha_sanskrit` so a build can record *what it read* — the
    resolved names, not the config's `null` — in its manifest before reading anything.
    """
    names = (
        list_verified_sanskrit_files(revision)
        if files is None
        else assert_verified_sanskrit_files(files)
    )
    return names if max_files is None else names[:max_files]


def sangraha_file_path(cache_dir: Path, name: str) -> Path:
    """Where `name` lives (or will live) under `cache_dir`, mirroring the repo layout."""
    return cache_dir / name


def sangraha_file_sizes(cache_dir: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    """`{"name", "n_bytes"}` for each of `names`, for the manifest's provenance.

    `n_bytes` is `None` for a file not on disk — the sizes are recorded after the read, so
    in a completed build every entry has one, and a `None` says the build did not get that
    far rather than that the file is empty.
    """
    return [
        {
            "name": name,
            "n_bytes": (
                sangraha_file_path(cache_dir, name).stat().st_size
                if sangraha_file_path(cache_dir, name).exists()
                else None
            ),
        }
        for name in names
    ]


def _ensure_downloaded(
    cache_dir: Path, name: str, revision: str = SANGRAHA_REVISION
) -> Path:
    """`name` on disk, downloading it from the pinned revision only if it is missing."""
    path = sangraha_file_path(cache_dir, name)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s from %s@%s", name, SANGRAHA_REPO, revision[:12])
    return Path(
        hf_hub_download(
            SANGRAHA_REPO,
            name,
            repo_type="dataset",
            revision=revision,
            local_dir=str(cache_dir),
        )
    )


def iter_sangraha_sanskrit(
    cache_dir: Path,
    *,
    files: Sequence[str] | None = None,
    revision: str = SANGRAHA_REVISION,
    max_files: int | None = None,
) -> Iterator[str]:
    """Every verified Sanskrit document, file by file and batch by batch.

    `files` defaults to `list_verified_sanskrit_files(revision)` (one network call); pass
    it to work offline from an already-populated `cache_dir`, and it is validated against
    `VERIFIED_SANSKRIT_PREFIX` (`resolve_sanskrit_files`). `max_files` truncates the list,
    which is how a shortened build is configured — it takes whole files from the front, so
    two runs with the same cap see the same documents.
    """
    import pyarrow.parquet as pq

    names = resolve_sanskrit_files(files=files, revision=revision, max_files=max_files)
    for name in names:
        path = _ensure_downloaded(cache_dir, name, revision)
        parquet = pq.ParquetFile(path)
        logger.info("reading %s (%d documents)", name, parquet.metadata.num_rows)
        for batch in parquet.iter_batches(
            batch_size=_BATCH_SIZE, columns=[SANGRAHA_TEXT_COLUMN]
        ):
            for document in batch.column(SANGRAHA_TEXT_COLUMN).to_pylist():
                if document:
                    yield str(document)


def iter_sangraha_lines(
    cache_dir: Path,
    *,
    files: Sequence[str] | None = None,
    revision: str = SANGRAHA_REVISION,
    max_files: int | None = None,
) -> Iterator[str]:
    """`iter_sangraha_sanskrit` flattened to corpus lines by `documents_to_lines`."""
    for document in iter_sangraha_sanskrit(
        cache_dir, files=files, revision=revision, max_files=max_files
    ):
        yield from documents_to_lines(document)
