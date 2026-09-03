"""Aligned parallel-corpus core: `ParallelCorpus`, jsonl I/O, plain-text pair loading.

Every parallel-text loader in this project (`flores.py`, `samayik.py`, `itihasa.py`) needs
the same three things: a container that guarantees every language is aligned by index
(`ParallelCorpus`), an offline jsonl cache (`save_jsonl`/`load_jsonl`), and a way to turn
one plain-text file per language into an aligned, cleaned corpus (`read_aligned_files`).
Centralising them here means the alignment invariant and the empty-line policy are
enforced once, not reimplemented per corpus.

`ParallelCorpus`, `save_jsonl` and `load_jsonl` were moved here verbatim (same behaviour)
from `flores.py`, which now imports and re-exports them so existing `from
sanskrit_tok.data.flores import ParallelCorpus` call sites keep working (CLAUDE.md §8:
no silent breakage of an established interface).
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CORPUS_NAME",
    "DEFAULT_SPLIT",
    "ParallelCorpus",
    "load_jsonl",
    "read_aligned_files",
    "save_jsonl",
]

logger = logging.getLogger(__name__)

#: Default corpus name, also assumed by `load_jsonl` (the jsonl stores sentences only).
CORPUS_NAME = "flores200"

#: Default evaluation split. FLORES-200 devtest is 1012 sentences.
DEFAULT_SPLIT = "devtest"


@dataclass(frozen=True)
class ParallelCorpus:
    """Sentences in several languages, aligned by index.

    `sentences[lang][i]` is the same content in every language, which is what makes
    parity and tokens-per-proposition well defined. The invariant that every language
    list has the same length is enforced at construction, not assumed by callers.
    """

    name: str
    split: str
    languages: tuple[str, ...]
    sentences: dict[str, list[str]]

    def __post_init__(self) -> None:
        if not self.languages:
            raise ValueError("a ParallelCorpus needs at least one language")
        missing = [language for language in self.languages if language not in self.sentences]
        if missing:
            raise ValueError(f"sentences missing for declared language(s): {missing}")
        undeclared = sorted(set(self.sentences) - set(self.languages))
        if undeclared:
            raise ValueError(f"sentences given for undeclared language(s): {undeclared}")
        lengths = {language: len(self.sentences[language]) for language in self.languages}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"aligned languages must be of equal length, got {lengths}")

    def __len__(self) -> int:
        return len(self.sentences[self.languages[0]])


# --------------------------------------------------------------------------- jsonl I/O


def save_jsonl(corpus: ParallelCorpus, path: Path) -> None:
    """Write `corpus` as one JSON object per line: `id` plus one key per language.

    Devanagari is written unescaped so the file stays readable; parent directories are
    created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(len(corpus)):
            record: dict[str, object] = {"id": index}
            for language in corpus.languages:
                record[language] = corpus.sentences[language][index]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("wrote %d aligned sentences to %s", len(corpus), path)


def load_jsonl(
    path: Path,
    *,
    name: str = CORPUS_NAME,
    split: str = DEFAULT_SPLIT,
) -> ParallelCorpus:
    """Read a jsonl written by `save_jsonl`.

    The languages are taken, in order, from the keys of the first record. `name` and
    `split` are not stored in the file (it holds sentences only) and default to the
    FLORES devtest values; pass them explicitly for any other corpus.
    """
    languages: tuple[str, ...] | None = None
    sentences: dict[str, list[str]] = {}
    index = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"{path}: line {line_number} has no 'id' field")
            record_languages = tuple(key for key in record if key != "id")
            if languages is None:
                languages = record_languages
                sentences = {language: [] for language in languages}
            elif record_languages != languages:
                raise ValueError(
                    f"{path}: line {line_number} has languages {record_languages}, "
                    f"expected {languages}"
                )
            if record["id"] != index:
                raise ValueError(
                    f"{path}: line {line_number} has id {record['id']!r}, expected {index}"
                )
            for language in languages:
                sentences[language].append(str(record[language]))
            index += 1
    if languages is None:
        raise ValueError(f"{path} is empty: no records to load")
    logger.info("loaded %d aligned sentences from %s", index, path)
    return ParallelCorpus(name=name, split=split, languages=languages, sentences=sentences)


# ------------------------------------------------------------------- plain-text pairs


def read_aligned_files(
    paths: Mapping[str, Path], *, name: str, split: str
) -> tuple[ParallelCorpus, int]:
    """Read one plain-text file per language (one sentence per line) into a `ParallelCorpus`.

    Lines are split on `\\n` only and each is `str.strip()`-ped; no Unicode normalisation
    is applied. An index is dropped from every language when any language's line is empty
    after stripping — a blank line in one file with content in the others is not a usable
    aligned pair, so it is dropped rather than kept misaligned or padded. Returns the
    cleaned corpus and the number of dropped indices.

    Raises `ValueError` if the files do not all have the same number of lines before any
    dropping happens — a line-count mismatch means the files are not aligned at all, which
    dropping empty lines cannot fix and must not paper over.
    """
    languages = tuple(paths)
    raw: dict[str, list[str]] = {
        language: path.read_text(encoding="utf-8").split("\n") for language, path in paths.items()
    }
    # A trailing newline produces one trailing empty element per file; drop it before the
    # length check so a normally-terminated file isn't reported as having an extra line.
    for lines in raw.values():
        if lines and lines[-1] == "":
            lines.pop()

    lengths = {language: len(lines) for language, lines in raw.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{name}/{split}: mismatched line count across languages: {lengths}")

    n = next(iter(lengths.values())) if lengths else 0
    stripped = {language: [line.strip() for line in lines] for language, lines in raw.items()}

    keep_indices = [
        index
        for index in range(n)
        if all(stripped[language][index] != "" for language in languages)
    ]
    dropped = n - len(keep_indices)

    sentences = {
        language: [stripped[language][index] for index in keep_indices] for language in languages
    }
    corpus = ParallelCorpus(name=name, split=split, languages=languages, sentences=sentences)
    logger.info(
        "%s/%s: read %d aligned pairs from plain text, dropped %d empty pair(s)",
        name,
        split,
        len(corpus),
        dropped,
    )
    return corpus, dropped
