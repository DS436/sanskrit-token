"""Tests for `sanskrit_tok.data.flores`.

Everything here runs offline against `tests/fixtures/flores_mini.jsonl`, except the one
test that exercises the real download; that is skipped unless `SANSKRIT_TOK_NETWORK_TESTS`
is set in the environment (plan Global Constraints: no network in tests).
"""

import json
import os
from collections.abc import Sequence
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


# ---------------------------------------------------------------------------- network


@NETWORK_TESTS
def test_load_flores_downloads_the_real_devtest() -> None:
    corpus = load_flores(LANGUAGES)
    assert len(corpus) == 1012
    assert corpus.languages == LANGUAGES
