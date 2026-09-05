"""Sanskrit Wikipedia loader: the small, clean half of corpus M1.

`wikimedia/wikipedia`, config `20231101.sa`, one 23.8 MB parquet, CC BY-SA 3.0 (with GFDL;
the dataset card's terms). It is two orders of magnitude smaller than Sangraha's verified
Sanskrit but it is *written* Sanskrit rather than OCR, so it is worth its own row in the
corpus (docs/decisions.md, 2026-09-05, "M1 monolingual corpus").

The parquet's `text` is already plain text — the wiki markup has been rendered away — so
the boilerplate that survives is structural rather than syntactic: a section heading on its
own line, the article title repeated as the first line, and the stub/category tails that
every short article ends in. `strip_boilerplate` removes those three shapes and then the
line rules of `sangraha.documents_to_lines` apply unchanged, so both halves of M1 are cut
into lines by exactly one implementation.

The `==` heading rule fires on nothing in the 20231101.sa dump (headings there are bare
lines, caught by the length rule instead). It is kept because it costs nothing and a
future dump that leaves the markers in would otherwise put `== इतिहासः ==` into the
training corpus.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

from sanskrit_tok.data.sangraha import documents_to_lines as _lines_of

__all__ = [
    "MIN_BOILERPLATE_LINE_CHARS",
    "WIKIPEDIA_CONFIG",
    "WIKIPEDIA_LICENCE",
    "WIKIPEDIA_PARQUET",
    "WIKIPEDIA_REPO",
    "WIKIPEDIA_REVISION",
    "documents_to_lines",
    "iter_wikipedia_lines",
    "iter_wikipedia_sanskrit",
    "strip_boilerplate",
    "wikipedia_file_path",
]

logger = logging.getLogger(__name__)

WIKIPEDIA_REPO = "wikimedia/wikipedia"

#: Revision sha of the dataset repository, observed 2026-09-05 via
#: `huggingface_hub.HfApi().dataset_info("wikimedia/wikipedia").sha`.
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"

#: The Sanskrit dump: 12,156 articles in a single parquet shard.
WIKIPEDIA_CONFIG = "20231101.sa"
WIKIPEDIA_PARQUET = f"{WIKIPEDIA_CONFIG}/train-00000-of-00001.parquet"

WIKIPEDIA_LICENCE = "CC BY-SA 3.0 (and GFDL), per the dataset card"

#: A newline-delimited line shorter than this is a heading, a category tail or a stub
#: marker rather than a sentence. Applied to the document's *raw* lines, before danda
#: splitting, so a long line of short sentences is not touched by it.
MIN_BOILERPLATE_LINE_CHARS = 20

_HEADING_MARKER = "=="

_BATCH_SIZE = 500

_TITLE_COLUMN = "title"
_TEXT_COLUMN = "text"


def _is_boilerplate(line: str, title: str | None) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(_HEADING_MARKER):
        return True
    if title is not None and stripped == title.strip():
        return True
    return len(stripped) < MIN_BOILERPLATE_LINE_CHARS


def strip_boilerplate(document: str, *, title: str | None = None) -> str:
    """`document` without its heading lines, title line and very short lines.

    Returns the surviving lines rejoined with newlines, so the result is still a document
    `documents_to_lines` can split; the three rules are deliberately applied to raw lines
    only, since a heading is a line and a sentence is not.
    """
    return "\n".join(
        line for line in document.split("\n") if not _is_boilerplate(line, title)
    )


def documents_to_lines(document: str, *, title: str | None = None) -> list[str]:
    """One article's corpus lines: boilerplate stripped, then the shared line rules."""
    return _lines_of(strip_boilerplate(document, title=title))


def wikipedia_file_path(cache_dir: Path) -> Path:
    """Where the Sanskrit parquet lives (or will live) under `cache_dir`."""
    return cache_dir / WIKIPEDIA_PARQUET


def _ensure_downloaded(cache_dir: Path, revision: str = WIKIPEDIA_REVISION) -> Path:
    """The parquet on disk, downloading it from the pinned revision only if missing."""
    path = wikipedia_file_path(cache_dir)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s from %s@%s", WIKIPEDIA_PARQUET, WIKIPEDIA_REPO, revision[:12])
    return Path(
        hf_hub_download(
            WIKIPEDIA_REPO,
            WIKIPEDIA_PARQUET,
            repo_type="dataset",
            revision=revision,
            local_dir=str(cache_dir),
        )
    )


def _iter_articles(
    cache_dir: Path, revision: str = WIKIPEDIA_REVISION
) -> Iterator[tuple[str, str]]:
    """`(title, text)` for every article, a batch at a time."""
    import pyarrow.parquet as pq

    path = _ensure_downloaded(cache_dir, revision)
    parquet = pq.ParquetFile(path)
    logger.info("reading %s (%d articles)", WIKIPEDIA_PARQUET, parquet.metadata.num_rows)
    for batch in parquet.iter_batches(
        batch_size=_BATCH_SIZE, columns=[_TITLE_COLUMN, _TEXT_COLUMN]
    ):
        titles: Sequence[str | None] = batch.column(_TITLE_COLUMN).to_pylist()
        texts: Sequence[str | None] = batch.column(_TEXT_COLUMN).to_pylist()
        for title, text in zip(titles, texts, strict=True):
            if text:
                yield str(title or ""), str(text)


def iter_wikipedia_sanskrit(
    cache_dir: Path, *, revision: str = WIKIPEDIA_REVISION
) -> Iterator[str]:
    """Every article's plain text, in dump order (the loader's document shape)."""
    for _, text in _iter_articles(cache_dir, revision):
        yield text


def iter_wikipedia_lines(
    cache_dir: Path, *, revision: str = WIKIPEDIA_REVISION
) -> Iterator[str]:
    """Every article flattened to corpus lines, its own title used to drop the title line."""
    for title, text in _iter_articles(cache_dir, revision):
        yield from documents_to_lines(text, title=title)
