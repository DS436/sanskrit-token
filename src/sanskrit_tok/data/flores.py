"""FLORES-200 loader: aligned devtest sentences, with an offline jsonl path.

FLORES-200 is the parity anchor of this project (CLAUDE.md §5): the same 1012 sentences
are translated into every language, so `tokens(san_Deva) / tokens(eng_Latn)` on this
corpus compares tokenizers on identical content rather than on whatever each language
happens to talk about.

The dataset is served from several places and none of them is reliable on its own, so
`load_flores` tries four sources in order and uses the first that works:

1. `openlanguagedata/flores_plus` on the Hugging Face Hub — the maintained release, but
   gated, so it needs an accepted licence and an `HF_TOKEN`.
2. `facebook/flores` — the original upload; now gated too, and a loading-script dataset,
   which `datasets` 5.x no longer executes at all.
3. `Muennighoff/flores200` — a community mirror; also a loading-script dataset.
4. The official tarball `flores200_dataset.tar.gz` from `dl.fbaipublicfiles.com`, which
   needs no Hub account and no extra dependency: one plain-text file per language under
   `flores200_dataset/<split>/<lang>.<split>`, one sentence per line, aligned by line
   number across languages.

Sentences are kept in their **source script** (`san_Deva` and `hin_Deva` in Devanagari,
`eng_Latn` in Latin). Per CLAUDE.md §2.3 the original script is what is stored; callers
that want the SLP1 form convert with `sanskrit_tok.encoding.to_slp1` at the point of use,
so both variants stay available to the experiments.

`save_jsonl` / `load_jsonl` are the offline path. Download once, save to
`data/raw/flores/devtest.jsonl` (gitignored), and reuse it on every later run so that
experiments and tests never depend on the network.
"""

import contextlib
import json
import logging
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "CORPUS_NAME",
    "DEFAULT_SPLIT",
    "FLORES_TARBALL_URL",
    "ParallelCorpus",
    "load_flores",
    "load_jsonl",
    "save_jsonl",
]

logger = logging.getLogger(__name__)

#: Official FLORES-200 release, mirrored by the NLLB project. ~25 MB gzipped.
FLORES_TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

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


# -------------------------------------------------------------------------- source I/O


def _split_language_code(language: str) -> tuple[str, str]:
    """`"san_Deva"` -> `("san", "Deva")`, i.e. ISO 639-3 and ISO 15924."""
    iso_639_3, _, iso_15924 = language.partition("_")
    if not iso_639_3 or not iso_15924:
        raise ValueError(
            f"language code {language!r} is not of the form '<iso639-3>_<iso15924>', "
            "e.g. 'san_Deva'"
        )
    return iso_639_3, iso_15924


def _cache_str(cache_dir: Path | None) -> str | None:
    return str(cache_dir) if cache_dir is not None else None


def _ordered_text(dataset: Any, field: str, language: str) -> list[str]:
    """Pull `field` out of a `datasets.Dataset`, ordered by `id` when there is one."""
    if field not in dataset.column_names:
        raise KeyError(f"no {field!r} column for {language}, got {dataset.column_names}")
    if "id" in dataset.column_names:
        dataset = dataset.sort("id")
    rows = [str(value) for value in dataset[field]]
    if not rows:
        raise ValueError(f"no rows for {language}")
    return rows


def _load_flores_plus(
    languages: Sequence[str], split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    """Source 1: `openlanguagedata/flores_plus` (gated; one config per language)."""
    from datasets import load_dataset

    sentences: dict[str, list[str]] = {}
    for language in languages:
        iso_639_3, iso_15924 = _split_language_code(language)
        dataset = load_dataset(
            "openlanguagedata/flores_plus",
            language,
            split=split,
            cache_dir=_cache_str(cache_dir),
        )
        # A config can carry several script variants; keep only the requested one.
        dataset = dataset.filter(
            lambda row, iso=iso_639_3, script=iso_15924: (
                row["iso_639_3"] == iso and row["iso_15924"] == script
            )
        )
        sentences[language] = _ordered_text(dataset, "text", language)
    return sentences


def _load_facebook_flores(
    languages: Sequence[str], split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    """Source 2: `facebook/flores`, a loading-script dataset (`datasets` < 4 only)."""
    from datasets import load_dataset

    sentences: dict[str, list[str]] = {}
    for language in languages:
        dataset = load_dataset(
            "facebook/flores",
            language,
            split=split,
            cache_dir=_cache_str(cache_dir),
            trust_remote_code=True,
        )
        sentences[language] = _ordered_text(dataset, "sentence", language)
    return sentences


def _load_muennighoff_flores200(
    languages: Sequence[str], split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    """Source 3: the `Muennighoff/flores200` community mirror."""
    from datasets import load_dataset

    sentences: dict[str, list[str]] = {}
    for language in languages:
        dataset = load_dataset(
            "Muennighoff/flores200",
            language,
            split=split,
            cache_dir=_cache_str(cache_dir),
        )
        sentences[language] = _ordered_text(dataset, "sentence", language)
    return sentences


def _load_flores_tarball(
    languages: Sequence[str], split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    """Source 4: the official tarball, downloaded with the standard library only.

    Only the requested `<split>/<lang>.<split>` members are read, and they are read
    through `extractfile` rather than extracted, so no archive path ever touches the
    filesystem (no path-traversal surface). The archive itself is cached in `cache_dir`
    when one is given, and thrown away with a temporary directory when it is not.
    """
    wanted = set(languages)
    sentences: dict[str, list[str]] = {}
    with contextlib.ExitStack() as stack:
        if cache_dir is None:
            work_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            work_dir = cache_dir
            work_dir.mkdir(parents=True, exist_ok=True)
        archive = work_dir / "flores200_dataset.tar.gz"
        if archive.exists():
            logger.info("reusing cached FLORES-200 archive at %s", archive)
        else:
            logger.info("downloading %s to %s", FLORES_TARBALL_URL, archive)
            urllib.request.urlretrieve(FLORES_TARBALL_URL, archive)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                if len(path.parts) < 2 or path.parts[-2] != split:
                    continue
                language = path.name.removesuffix(f".{split}")
                if language not in wanted or language in sentences:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                sentences[language] = handle.read().decode("utf-8").splitlines()
    absent = sorted(wanted - set(sentences))
    if absent:
        raise KeyError(f"{FLORES_TARBALL_URL} has no {split} file for {absent}")
    return {language: sentences[language] for language in languages}


#: `(name, loader)` in the order `load_flores` attempts them. The name is what gets
#: logged and what is recorded in `data/README.md` as the provenance of the download.
_SourceLoader = Callable[[Sequence[str], str, Path | None], dict[str, list[str]]]

_SOURCES: tuple[tuple[str, _SourceLoader], ...] = (
    ("openlanguagedata/flores_plus", _load_flores_plus),
    ("facebook/flores", _load_facebook_flores),
    ("Muennighoff/flores200", _load_muennighoff_flores200),
    ("dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", _load_flores_tarball),
)


def load_flores(
    languages: Sequence[str],
    split: str = DEFAULT_SPLIT,
    cache_dir: Path | None = None,
) -> ParallelCorpus:
    """Load aligned FLORES-200 sentences for `languages`, e.g. `["san_Deva", "eng_Latn"]`.

    Sources are tried in the order documented in the module docstring and the first that
    yields every requested language wins; the one used is logged at INFO. Raises
    `RuntimeError` naming every source and its failure if none of them work.
    """
    requested = tuple(languages)
    if not requested:
        raise ValueError("load_flores needs at least one language")

    failures: list[str] = []
    for source_name, loader in _SOURCES:
        try:
            # Construction is inside the `try` on purpose: a source that returns only
            # some of the languages, or ragged ones, is a failed source and should fall
            # through to the next rather than abort the whole load.
            corpus = ParallelCorpus(
                name=CORPUS_NAME,
                split=split,
                languages=requested,
                sentences=loader(requested, split, cache_dir),
            )
        except Exception as error:  # each source fails in its own way; try the next
            logger.info(
                "FLORES source %s unavailable: %s: %s",
                source_name,
                type(error).__name__,
                error,
            )
            failures.append(f"  {source_name}: {type(error).__name__}: {error}")
            continue
        logger.info(
            "loaded FLORES-200 %s (%d sentences, %s) from %s",
            split,
            len(corpus),
            ", ".join(requested),
            source_name,
        )
        return corpus

    raise RuntimeError(
        f"could not load FLORES-200 {split} for {list(requested)}; attempted:\n"
        + "\n".join(failures)
    )
