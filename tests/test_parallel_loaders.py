"""Tests for `sanskrit_tok.data.parallel` and the Sāmayik/Itihāsa loaders.

Offline tests run against `tests/fixtures/samayik_mini/` and `tests/fixtures/itihasa_mini/`,
each four lines with one empty Sanskrit line, matching the network-gate pattern used in
`tests/test_flores.py`. The live-download tests are skipped unless
`SANSKRIT_TOK_NETWORK_TESTS` is set.
"""

import io
import os
import urllib.request
from pathlib import Path

import pytest

from sanskrit_tok.data.flores import ParallelCorpus, load_jsonl, save_jsonl
from sanskrit_tok.data.parallel import download_file, read_aligned_files

FIXTURES = Path(__file__).parent / "fixtures"
SAMAYIK_MINI = FIXTURES / "samayik_mini"
ITIHASA_MINI = FIXTURES / "itihasa_mini"

NETWORK_TESTS = pytest.mark.skipif(
    not os.environ.get("SANSKRIT_TOK_NETWORK_TESTS"),
    reason="set SANSKRIT_TOK_NETWORK_TESTS=1 to exercise the real download",
)


# -------------------------------------------------------------------------- download_file


def test_download_file_lands_atomically_with_no_part_file_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"aligned sentence bytes"

    def fake_urlopen(url: str, timeout: float | None = None) -> io.BytesIO:
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    destination = tmp_path / "nested" / "file.sa"
    download_file("https://example.invalid/file.sa", destination, timeout=5)

    assert destination.read_bytes() == payload
    assert list(tmp_path.rglob("*.part")) == []


class _RaisingStream(io.BytesIO):
    """A response whose `read` fails partway through, to exercise the interrupted path."""

    def read(self, *args: object, **kwargs: object) -> bytes:
        raise OSError("connection reset mid-copy")


def test_download_file_leaves_nothing_behind_when_the_stream_fails_mid_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_urlopen(url: str, timeout: float | None = None) -> _RaisingStream:
        return _RaisingStream(b"partial")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    destination = tmp_path / "file.sa"
    with pytest.raises(OSError, match="mid-copy"):
        download_file("https://example.invalid/file.sa", destination, timeout=5)

    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


# --------------------------------------------------------------- read_aligned_files


def test_read_aligned_files_drops_the_pair_with_an_empty_sanskrit_line() -> None:
    corpus, dropped = read_aligned_files(
        {"san_Deva": SAMAYIK_MINI / "test.sa", "eng_Latn": SAMAYIK_MINI / "test.en"},
        name="samayik",
        split="test",
    )
    assert dropped == 1
    assert len(corpus) == 3
    assert corpus.sentences["san_Deva"] == ["सः गच्छति।", "बालकः पठति।", "नदी वहति।"]
    assert corpus.sentences["eng_Latn"] == ["He goes.", "The boy reads.", "The river flows."]
    assert corpus.name == "samayik"
    assert corpus.split == "test"


def test_read_aligned_files_on_itihasa_mini_fixture() -> None:
    corpus, dropped = read_aligned_files(
        {"san_Deva": ITIHASA_MINI / "test.sn", "eng_Latn": ITIHASA_MINI / "test.en"},
        name="itihasa",
        split="test",
    )
    assert dropped == 1
    assert len(corpus) == 3


def test_read_aligned_files_raises_on_mismatched_line_counts(tmp_path: Path) -> None:
    sa = tmp_path / "a.sa"
    en = tmp_path / "a.en"
    sa.write_text("one\ntwo\n", encoding="utf-8")
    en.write_text("one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line count"):
        read_aligned_files({"san_Deva": sa, "eng_Latn": en}, name="x", split="test")


def test_read_aligned_files_strips_whitespace() -> None:
    corpus, dropped = read_aligned_files(
        {"san_Deva": SAMAYIK_MINI / "test.sa", "eng_Latn": SAMAYIK_MINI / "test.en"},
        name="samayik",
        split="test",
    )
    for language in corpus.languages:
        for sentence in corpus.sentences[language]:
            assert sentence == sentence.strip()


# ------------------------------------------------------ flores.py regression re-exports


def test_parallel_corpus_still_importable_from_flores() -> None:
    from sanskrit_tok.data.parallel import ParallelCorpus as MovedParallelCorpus

    assert ParallelCorpus is MovedParallelCorpus


def test_save_and_load_jsonl_still_importable_from_flores() -> None:
    from sanskrit_tok.data.parallel import load_jsonl as moved_load_jsonl
    from sanskrit_tok.data.parallel import save_jsonl as moved_save_jsonl

    assert save_jsonl is moved_save_jsonl
    assert load_jsonl is moved_load_jsonl


# --------------------------------------------------------------------- load_samayik


@NETWORK_TESTS
def test_load_samayik_test_split_has_expected_pair_count(tmp_path: Path) -> None:
    from sanskrit_tok.data.samayik import load_samayik

    corpus = load_samayik("test", cache_dir=tmp_path)
    assert 2417 - 20 <= len(corpus) <= 2417


@NETWORK_TESTS
def test_load_samayik_reuses_cached_jsonl(tmp_path: Path) -> None:
    from sanskrit_tok.data.samayik import load_samayik

    first = load_samayik("dev", cache_dir=tmp_path)
    assert (tmp_path / "dev.jsonl").exists()
    second = load_samayik("dev", cache_dir=tmp_path)
    assert first == second


# --------------------------------------------------------------------- load_itihasa


@NETWORK_TESTS
def test_load_itihasa_test_split_has_expected_pair_count(tmp_path: Path) -> None:
    from sanskrit_tok.data.itihasa import load_itihasa

    corpus = load_itihasa("test", cache_dir=tmp_path)
    assert 11722 - 200 <= len(corpus) <= 11722
