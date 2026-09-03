"""Tests for `sanskrit_tok.data.flores`.

Everything here runs offline against `tests/fixtures/flores_mini.jsonl`, except the one
test that exercises the real download; that is skipped unless `SANSKRIT_TOK_NETWORK_TESTS`
is set in the environment (plan Global Constraints: no network in tests).
"""

import hashlib
import io
import json
import os
import tarfile
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from sanskrit_tok.data.flores import (
    ParallelCorpus,
    load_flores,
    load_jsonl,
    save_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "flores_mini.jsonl"
LANGUAGES = ("san_Deva", "hin_Deva", "eng_Latn")

NETWORK_TESTS = pytest.mark.skipif(
    not os.environ.get("SANSKRIT_TOK_NETWORK_TESTS"),
    reason="set SANSKRIT_TOK_NETWORK_TESTS=1 to exercise the FLORES download",
)


def _mini() -> ParallelCorpus:
    return load_jsonl(FIXTURE)


# --------------------------------------------------------------------------- dataclass


def test_parallel_corpus_reports_its_length() -> None:
    corpus = ParallelCorpus(
        name="toy",
        split="devtest",
        languages=("a", "b"),
        sentences={"a": ["one", "two"], "b": ["eins", "zwei"]},
    )
    assert len(corpus) == 2


def test_parallel_corpus_is_frozen() -> None:
    corpus = _mini()
    with pytest.raises(AttributeError):
        corpus.split = "dev"  # type: ignore[misc]


def test_parallel_corpus_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ParallelCorpus(
            name="toy",
            split="devtest",
            languages=("a", "b"),
            sentences={"a": ["one", "two"], "b": ["eins"]},
        )


def test_parallel_corpus_rejects_a_language_with_no_sentences() -> None:
    with pytest.raises(ValueError, match="missing"):
        ParallelCorpus(
            name="toy",
            split="devtest",
            languages=("a", "b"),
            sentences={"a": ["one"]},
        )


def test_parallel_corpus_rejects_sentences_for_an_undeclared_language() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        ParallelCorpus(
            name="toy",
            split="devtest",
            languages=("a",),
            sentences={"a": ["one"], "b": ["eins"]},
        )


def test_parallel_corpus_rejects_an_empty_language_tuple() -> None:
    with pytest.raises(ValueError, match="at least one language"):
        ParallelCorpus(name="toy", split="devtest", languages=(), sentences={})


# ------------------------------------------------------------------------------ jsonl


def test_load_jsonl_returns_three_aligned_languages() -> None:
    corpus = _mini()
    assert len(corpus) == 3
    assert corpus.languages == LANGUAGES
    assert set(corpus.sentences) == set(LANGUAGES)
    assert all(len(corpus.sentences[lang]) == 3 for lang in LANGUAGES)


def test_load_jsonl_preserves_alignment_and_text() -> None:
    corpus = _mini()
    assert corpus.sentences["san_Deva"][0] == "सः ग्रामं गच्छति।"
    assert corpus.sentences["hin_Deva"][0] == "वह गाँव जाता है।"
    assert corpus.sentences["eng_Latn"][0] == "He goes to the village."
    assert corpus.sentences["eng_Latn"][2] == "The river flows down from the mountain."


def test_load_jsonl_defaults_name_and_split_and_accepts_overrides() -> None:
    assert _mini().name == "flores200"
    assert _mini().split == "devtest"
    override = load_jsonl(FIXTURE, name="mini", split="dev")
    assert (override.name, override.split) == ("mini", "dev")


def test_save_jsonl_then_load_jsonl_roundtrips(tmp_path: Path) -> None:
    corpus = _mini()
    path = tmp_path / "nested" / "devtest.jsonl"
    save_jsonl(corpus, path)
    assert load_jsonl(path) == corpus


def test_save_jsonl_writes_one_object_per_line_with_an_id(tmp_path: Path) -> None:
    path = tmp_path / "devtest.jsonl"
    save_jsonl(_mini(), path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == [0, 1, 2]
    assert list(records[0]) == ["id", *LANGUAGES]
    assert records[1]["eng_Latn"] == "The children study at the school."


def test_save_jsonl_keeps_devanagari_unescaped(tmp_path: Path) -> None:
    path = tmp_path / "devtest.jsonl"
    save_jsonl(_mini(), path)
    assert "सः ग्रामं गच्छति।" in path.read_text(encoding="utf-8")


def test_load_jsonl_rejects_a_line_whose_languages_differ(tmp_path: Path) -> None:
    path = tmp_path / "ragged.jsonl"
    path.write_text(
        '{"id": 0, "a": "one", "b": "eins"}\n{"id": 1, "a": "two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)


def test_load_jsonl_rejects_an_out_of_order_id(tmp_path: Path) -> None:
    path = tmp_path / "shuffled.jsonl"
    path.write_text('{"id": 0, "a": "one"}\n{"id": 7, "a": "two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_jsonl(path)


def test_load_jsonl_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_jsonl(path)


# ------------------------------------------------------------------- source fallback


def _stub_source(
    sentences: dict[str, list[str]] | None,
) -> object:
    """A loader that either returns `sentences` or raises, matching the source signature."""

    def loader(
        languages: Sequence[str], split: str, cache_dir: Path | None
    ) -> dict[str, list[str]]:
        if sentences is None:
            raise OSError("stub source unavailable")
        return sentences

    return loader


def test_load_flores_uses_the_first_source_that_succeeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from sanskrit_tok.data import flores

    monkeypatch.setattr(
        flores,
        "_SOURCES",
        (
            ("first", _stub_source(None)),
            ("second", _stub_source({"a": ["one"]})),
            ("third", _stub_source({"a": ["never used"]})),
        ),
    )
    with caplog.at_level("INFO", logger=flores.__name__):
        corpus = load_flores(["a"])
    assert corpus.sentences == {"a": ["one"]}
    assert "second" in caplog.text


def test_load_flores_falls_through_a_source_that_returns_only_some_languages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial source is a failed source: it must not abort the whole load."""
    from sanskrit_tok.data import flores

    monkeypatch.setattr(
        flores,
        "_SOURCES",
        (
            ("partial", _stub_source({"a": ["one"]})),  # "b" is missing
            ("complete", _stub_source({"a": ["one"], "b": ["eins"]})),
        ),
    )
    corpus = load_flores(["a", "b"])
    assert corpus.sentences == {"a": ["one"], "b": ["eins"]}


def test_load_flores_falls_through_a_source_that_returns_ragged_languages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sanskrit_tok.data import flores

    monkeypatch.setattr(
        flores,
        "_SOURCES",
        (
            ("ragged", _stub_source({"a": ["one", "two"], "b": ["eins"]})),
            ("complete", _stub_source({"a": ["one"], "b": ["eins"]})),
        ),
    )
    assert load_flores(["a", "b"]).sentences == {"a": ["one"], "b": ["eins"]}


def test_load_flores_raises_listing_every_attempted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sanskrit_tok.data import flores

    monkeypatch.setattr(
        flores,
        "_SOURCES",
        (("first", _stub_source(None)), ("second", _stub_source(None))),
    )
    with pytest.raises(RuntimeError) as excinfo:
        load_flores(["a"])
    message = str(excinfo.value)
    assert "first" in message
    assert "second" in message
    assert "stub source unavailable" in message


def test_load_flores_declares_four_sources_in_the_documented_order() -> None:
    from sanskrit_tok.data import flores

    assert [name for name, _ in flores._SOURCES] == [
        "openlanguagedata/flores_plus",
        "facebook/flores",
        "Muennighoff/flores200",
        "dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    ]


def test_load_flores_rejects_an_empty_language_list() -> None:
    with pytest.raises(ValueError, match="at least one language"):
        load_flores([])


# --------------------------------------------------------------- tarball download path

TARBALL_LANGUAGES = ("xxa_Test", "xxb_Test")


def _tiny_tarball(sentences: dict[str, list[str]], split: str = "devtest") -> bytes:
    """A stand-in for flores200_dataset.tar.gz: `<split>/<lang>.<split>` text members."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for language, lines in sentences.items():
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            info = tarfile.TarInfo(f"flores200_dataset/{split}/{language}.{split}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[float | None]:
    """Make `urlopen` return `payload`; returns the list of timeouts it was called with."""
    timeouts: list[float | None] = []

    def fake_urlopen(url: str, timeout: float | None = None) -> io.BytesIO:
        timeouts.append(timeout)
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return timeouts


def _refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any download attempt fail loudly, to prove a cached archive was reused."""

    def fake_urlopen(url: str, timeout: float | None = None) -> io.BytesIO:
        raise AssertionError("the network must not be touched here")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


@pytest.fixture
def tarball(monkeypatch: pytest.MonkeyPatch) -> Iterator[bytes]:
    """A tiny tarball, with the module's expected hash pointed at it."""
    from sanskrit_tok.data import flores

    payload = _tiny_tarball(
        {
            "xxa_Test": ["first a", "second a"],
            "xxb_Test": ["first b", "second b"],
        }
    )
    monkeypatch.setattr(
        flores, "FLORES200_TARBALL_SHA256", hashlib.sha256(payload).hexdigest()
    )
    yield payload


def test_tarball_download_is_verified_and_installed_atomically(
    tarball: bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sanskrit_tok.data import flores

    timeouts = _serve(monkeypatch, tarball)
    sentences = flores._load_flores_tarball(TARBALL_LANGUAGES, "devtest", tmp_path)

    assert sentences == {
        "xxa_Test": ["first a", "second a"],
        "xxb_Test": ["first b", "second b"],
    }
    archive = tmp_path / "flores200_dataset.tar.gz"
    assert archive.read_bytes() == tarball
    assert list(tmp_path.glob("*.part")) == [], "the .part file must not survive"
    assert timeouts == [flores.FLORES_DOWNLOAD_TIMEOUT_S]


def test_a_verified_cached_archive_is_reused_without_downloading(
    tarball: bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sanskrit_tok.data import flores

    (tmp_path / "flores200_dataset.tar.gz").write_bytes(tarball)
    _refuse(monkeypatch)
    sentences = flores._load_flores_tarball(("xxa_Test",), "devtest", tmp_path)
    assert sentences["xxa_Test"] == ["first a", "second a"]


def test_a_corrupt_cached_archive_is_discarded_and_downloaded_again(
    tarball: bytes,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sanskrit_tok.data import flores

    archive = tmp_path / "flores200_dataset.tar.gz"
    archive.write_bytes(b"a truncated download, or a captive-portal login page")
    _serve(monkeypatch, tarball)

    with caplog.at_level("WARNING", logger=flores.__name__):
        sentences = flores._load_flores_tarball(("xxa_Test",), "devtest", tmp_path)

    assert sentences["xxa_Test"] == ["first a", "second a"]
    assert archive.read_bytes() == tarball
    assert "expected" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_a_download_whose_hash_mismatches_raises_and_installs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sanskrit_tok.data import flores

    _serve(monkeypatch, _tiny_tarball({"xxa_Test": ["first a"]}))
    # FLORES200_TARBALL_SHA256 is left at the real archive's hash, so this cannot match.
    with pytest.raises(ValueError, match="sha256"):
        flores._load_flores_tarball(("xxa_Test",), "devtest", tmp_path)

    assert not (tmp_path / "flores200_dataset.tar.gz").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_tarball_lines_split_on_newline_only() -> None:
    from sanskrit_tok.data import flores

    # U+2028 LINE SEPARATOR: `str.splitlines` breaks on it, the FLORES format does not.
    payload = f"one{chr(0x2028)}still one\ntwo\n".encode()
    assert flores._tarball_lines(payload) == [f"one{chr(0x2028)}still one", "two"]


def test_tarball_lines_keep_an_intentional_blank_line() -> None:
    from sanskrit_tok.data import flores

    assert flores._tarball_lines(b"one\n\nthree\n") == ["one", "", "three"]


def test_tarball_reports_a_language_the_archive_does_not_have(
    tarball: bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sanskrit_tok.data import flores

    _serve(monkeypatch, tarball)
    with pytest.raises(KeyError, match="zzz_Test"):
        flores._load_flores_tarball(("xxa_Test", "zzz_Test"), "devtest", tmp_path)


# ---------------------------------------------------------------------------- network


@NETWORK_TESTS
def test_load_flores_downloads_the_real_devtest() -> None:
    corpus = load_flores(LANGUAGES)
    assert len(corpus) == 1012
    assert corpus.languages == LANGUAGES
