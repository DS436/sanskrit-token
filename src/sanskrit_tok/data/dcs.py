"""Digital Corpus of Sanskrit: sparse checkout and CoNLL-U parsing.

DCS (CLAUDE.md §5) is this project's only source of gold morpheme boundaries and the
corpus every Experiment 04 arm is trained on (docs/decisions.md, 2026-09-05, "DCS is the
gold source and the first monolingual training corpus"). Licence **CC BY 4.0**, stated in
`dcs/data/readme.md` of the source repository.

**Download.** The repository is 1.9 GB and only `dcs/data/conllu/files` is wanted, so the
checkout is blobless and sparse: `git clone --filter=blob:none --no-checkout`, then
`git sparse-checkout set dcs/data/conllu/files`, then `git checkout <commit>`. `--depth 1`
is deliberately *not* used — a shallow clone cannot be checked out at an arbitrary pinned
commit, and pinning is the reproducibility mechanism this project uses everywhere else
(`itihasa.py`, `samayik.py`). `download_dcs` is resumable and idempotent: an existing
checkout already at `commit` is returned untouched.

**Format.** One file per chapter, four `##` document lines at the top (`text`, `text_id`,
`chapter`, `chapter_id`) and then a run of sentences, each introduced by `# text = <IAST>`
followed by `# sent_id` and the token block. A *surface word* is either

* a multiword range line `n-m<TAB>surface<TAB>_...`, whose surface is the sandhied word and
  whose unsandhied segments are the token lines `n` through `m`; or
* a token line whose id no range covers, which is a one-segment word.

`Unsandhied=` and `UnsandhiedReconstructed=True` live in MISC (column 10);
`UnsandhiedReconstructed=True` marks a *machine-generated* segmentation, so a sentence with
none of them is human-verified (`sentence_is_human_verified`) and is the subset every
MorphScore table reports separately. `Case=Cpd` in FEATS marks a compound member
(`is_cpd`). Parsing is defensive: FEATS is sometimes an empty cell rather than `_`, and
`# text` occasionally holds words the token block does not reconstruct
(`text_matches_token_block`, 11,339 sentences, 1.5%) — both are recorded, never fatal.

Everything here keeps DCS's own IAST; conversion to SLP1 is
`sanskrit_tok.data.boundaries`, so the raw parse stays exactly what the file said
(CLAUDE.md §2.3: store the original script alongside).
"""

import logging
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DCS_COMMIT",
    "DCS_CONLLU_DIR",
    "DCS_LICENCE",
    "DCS_REPO",
    "DcsSentence",
    "DcsToken",
    "DcsWord",
    "conllu_files",
    "download_dcs",
    "iter_conllu_sentences",
    "read_text_id",
    "sentence_is_human_verified",
    "text_matches_token_block",
]

logger = logging.getLogger(__name__)

DCS_REPO = "OliverHellwig/sanskrit"

#: Head of `master`, observed and pinned 2026-09-05 (authored 2026-08-24). Every checkout
#: this project makes is detached at this commit; changing it means re-running the
#: ingestion and re-training every `_dcs` arm.
DCS_COMMIT = "8aeed5a1343d0e48b64eb32af8c00e8c6eb29359"

#: The only path the sparse checkout materialises.
DCS_CONLLU_DIR = "dcs/data/conllu/files"

#: Stated in `dcs/data/readme.md` at `DCS_COMMIT`.
DCS_LICENCE = "CC BY 4.0"

_CLONE_URL = f"https://github.com/{DCS_REPO}.git"

_DOC_PREFIX = "## "
_TEXT_PREFIX = "# text = "
_SENT_ID_PREFIX = "# sent_id = "

_MISC_UNSANDHIED = "Unsandhied"
_MISC_RECONSTRUCTED = "UnsandhiedReconstructed"
_FEAT_CPD = "Case=Cpd"

#: CoNLL-U's empty cell.
_EMPTY_CELL = "_"

#: Minimum columns a token line must have for the fields this project reads (MISC is 10).
_N_COLUMNS = 10


@dataclass(frozen=True)
class DcsToken:
    """One token line: a single unsandhied segment of a surface word, with its analysis."""

    id: int
    surface_iast: str
    lemma_iast: str
    upos: str
    feats: str
    unsandhied_iast: str
    reconstructed: bool
    is_cpd: bool


@dataclass(frozen=True)
class DcsWord:
    """One surface word: a multiword range and its tokens, or a lone token line."""

    surface_iast: str
    tokens: tuple[DcsToken, ...]


@dataclass(frozen=True)
class DcsSentence:
    """One sentence: its `# text` line, its document metadata, and its surface words.

    `sent_id` is a **string**, not an integer: 741,395 DCS sentences carry a plain numeric
    id but 29,336 (4.0%, almost all of them in the syntactically annotated Vedic subset —
    Ṛgveda, Atharvaveda, the brāhmaṇas and saṃhitās) carry a sub-sentence id of the form
    `746028_1`, with no bare `746028` anywhere beside it. Parsing the id as an integer
    would silently drop exactly those sentences, which are also the bulk of the
    human-verified subset MorphScore reports on.
    """

    sent_id: str
    text_id: int
    text_name: str
    chapter: str
    text_iast: str
    words: tuple[DcsWord, ...]


def sentence_is_human_verified(sentence: DcsSentence) -> bool:
    """True when no token carries `UnsandhiedReconstructed=True`.

    DCS's segmented forms are partly machine-generated (CLAUDE.md §5); this is the flag
    that separates the two, and MorphScore is reported on the whole held-out set and on
    this subset (docs/decisions.md, "DCS is the gold source...").
    """
    return all(not token.reconstructed for word in sentence.words for token in word.tokens)


def text_matches_token_block(sentence: DcsSentence) -> bool:
    """True when `# text`'s whitespace words are exactly the reconstructed surface words.

    False for the 1.5% of DCS sentences whose token block omits a word `# text` holds.
    The sentence is kept either way; the omitted words simply carry no gold segmentation.
    """
    return sentence.text_iast.split() == [word.surface_iast for word in sentence.words]


# ------------------------------------------------------------------------------ download


def _git(*args: str) -> str:
    """Run `git` with `args`, returning stdout; raises `CalledProcessError` on failure."""
    logger.info("git %s", " ".join(args))
    completed = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _head_commit(repo_dir: Path) -> str | None:
    """`git rev-parse HEAD` in `repo_dir`, or `None` if it is not a usable checkout."""
    if not (repo_dir / ".git").exists():
        return None
    try:
        return _git("-C", str(repo_dir), "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return None


def download_dcs(dest: Path, commit: str = DCS_COMMIT) -> Path:
    """Sparse-checkout `DCS_CONLLU_DIR` at `commit` under `dest/repo`; return that directory.

    Resumable: a checkout already at `commit` whose CoNLL-U directory exists is returned
    without touching the network, so re-running the ingestion is free. `dest` is under
    `data/raw/`, which is gitignored (CLAUDE.md §5: do not commit raw data).
    """
    repo_dir = dest / "repo"
    conllu_dir = repo_dir / DCS_CONLLU_DIR
    if conllu_dir.is_dir() and _head_commit(repo_dir) == commit:
        logger.info("DCS already checked out at %s (%s)", commit, conllu_dir)
        return conllu_dir

    dest.mkdir(parents=True, exist_ok=True)
    if _head_commit(repo_dir) is None:
        logger.info("cloning %s (blobless, no checkout) into %s", _CLONE_URL, repo_dir)
        _git("clone", "--filter=blob:none", "--no-checkout", _CLONE_URL, str(repo_dir))
    _git("-C", str(repo_dir), "sparse-checkout", "set", DCS_CONLLU_DIR)
    if _head_commit(repo_dir) != commit:
        _git("-C", str(repo_dir), "fetch", "--filter=blob:none", "origin", commit)
    _git("-C", str(repo_dir), "checkout", commit)

    if not conllu_dir.is_dir():
        raise RuntimeError(f"{conllu_dir} missing after checkout of {commit}")
    return conllu_dir


def conllu_files(conllu_dir: Path) -> list[Path]:
    """Every `.conllu` file under `conllu_dir`, sorted — the ingestion's file order."""
    return sorted(conllu_dir.rglob("*.conllu"))


# -------------------------------------------------------------------------------- parsing


def _cell(value: str) -> str:
    """A CoNLL-U cell, with the empty marker `_` normalised to the empty string."""
    return "" if value == _EMPTY_CELL else value


def _parse_misc(misc: str) -> dict[str, str]:
    """MISC (`Key=Value|Key=Value`) as a dict; keys without `=` are ignored."""
    fields: dict[str, str] = {}
    for item in misc.split("|"):
        key, separator, value = item.partition("=")
        if separator:
            fields[key] = value
    return fields


def _parse_token(columns: Sequence[str]) -> DcsToken:
    surface = columns[1]
    misc = _parse_misc(columns[9])
    return DcsToken(
        id=int(columns[0]),
        surface_iast=surface,
        lemma_iast=_cell(columns[2]),
        upos=_cell(columns[3]),
        feats=_cell(columns[5]),
        unsandhied_iast=misc.get(_MISC_UNSANDHIED) or surface,
        reconstructed=misc.get(_MISC_RECONSTRUCTED) == "True",
        is_cpd=_FEAT_CPD in columns[5],
    )


class _SentenceBuilder:
    """Accumulates one sentence's surface words while its token block is being read."""

    def __init__(self) -> None:
        self.words: list[DcsWord] = []
        self._range_surface: str | None = None
        self._range_end: int | None = None
        self._range_tokens: list[DcsToken] = []

    def _close_range(self) -> None:
        if self._range_surface is not None and self._range_tokens:
            self.words.append(
                DcsWord(surface_iast=self._range_surface, tokens=tuple(self._range_tokens))
            )
        self._range_surface = None
        self._range_end = None
        self._range_tokens = []

    def add_range(self, columns: Sequence[str]) -> None:
        self._close_range()
        _, _, end = columns[0].partition("-")
        if not end.isdigit():  # pragma: no cover - malformed range id
            logger.warning("skipping malformed range id %r", columns[0])
            return
        self._range_end = int(end)
        self._range_surface = columns[1]
        self._range_tokens = []

    def add_token(self, columns: Sequence[str]) -> None:
        token = _parse_token(columns)
        if self._range_end is not None and token.id <= self._range_end:
            self._range_tokens.append(token)
            if token.id == self._range_end:
                self._close_range()
            return
        self._close_range()
        self.words.append(DcsWord(surface_iast=token.surface_iast, tokens=(token,)))

    def finish(self) -> tuple[DcsWord, ...]:
        self._close_range()
        return tuple(self.words)


def iter_conllu_sentences(path: Path) -> Iterator[DcsSentence]:
    """Yield every sentence of one `.conllu` file, in file order.

    Document metadata (`## text`, `## text_id`, `## chapter`) is carried into every
    sentence that follows it. A sentence with no `# sent_id` is skipped with a warning
    rather than given a fabricated id.
    """
    text_name = ""
    text_id = -1
    chapter = ""

    text_iast: str | None = None
    sent_id: str | None = None
    builder = _SentenceBuilder()

    def emit() -> Iterator[DcsSentence]:
        if text_iast is None:
            return
        if not sent_id:
            logger.warning("%s: sentence %r has no sent_id; skipped", path, text_iast[:40])
            return
        yield DcsSentence(
            sent_id=sent_id,
            text_id=text_id,
            text_name=text_name,
            chapter=chapter,
            text_iast=text_iast,
            words=builder.finish(),
        )

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(_DOC_PREFIX):
                key, _, value = line[len(_DOC_PREFIX) :].partition(":")
                value = value.strip()
                if key == "text":
                    text_name = value
                elif key == "chapter":
                    chapter = value
                elif key == "text_id":
                    text_id = int(value) if value.isdigit() else -1
                continue
            if line.startswith(_TEXT_PREFIX):
                yield from emit()
                text_iast = line[len(_TEXT_PREFIX) :]
                sent_id = None
                builder = _SentenceBuilder()
                continue
            if line.startswith(_SENT_ID_PREFIX):
                sent_id = line[len(_SENT_ID_PREFIX) :].strip() or None
                continue
            if line.startswith("#") or not line.strip():
                continue

            columns = line.split("\t")
            if len(columns) < _N_COLUMNS:
                logger.warning("%s: skipping short token line %r", path, line[:60])
                continue
            token_id = columns[0]
            if "-" in token_id:
                builder.add_range(columns)
            elif "." in token_id:
                continue  # CoNLL-U empty node; DCS does not use them for segmentation.
            else:
                builder.add_token(columns)

    yield from emit()


def read_text_id(path: Path) -> int:
    """The `## text_id:` of one file, read from its header alone.

    The ingestion needs every file's text id before it can assign the held-out split, and
    a full parse of 16,051 files to learn 271 ids would double the work; one file is one
    chapter of one text, so the header is enough.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(_DOC_PREFIX):
                break
            key, _, value = line[len(_DOC_PREFIX) :].partition(":")
            if key == "text_id":
                stripped = value.strip()
                return int(stripped) if stripped.isdigit() else -1
    logger.warning("%s: no ## text_id header", path)
    return -1
