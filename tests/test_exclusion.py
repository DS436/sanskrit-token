"""Tests for `sanskrit_tok.data.exclusion` (CLAUDE.md §2.4: no evaluation leakage)."""

from pathlib import Path

import pytest

from sanskrit_tok.data.exclusion import (
    LeakageError,
    assert_not_excluded,
    build_exclusion_list,
    load_exclusion_hashes,
    sentence_hash,
    sentence_hash_en,
)


def test_sentence_hash_is_stable() -> None:
    assert sentence_hash("रामः") == sentence_hash("रामः")


def test_sentence_hash_is_script_normalised_by_stripping_whitespace() -> None:
    assert sentence_hash("रामः") == sentence_hash(" रामः ")


def test_sentence_hash_differs_for_different_text() -> None:
    assert sentence_hash("रामः") != sentence_hash("सीता")


def test_build_exclusion_list_writes_headers_and_sorted_unique_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exclusion_hashes.txt"
    count = build_exclusion_list(
        {"a": ["रामः", "सीता"], "b": ["रामः"]},  # duplicate hash across sources
        path,
    )
    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# sha256 of the SLP1 form")
    assert lines[1].startswith("# sources:")
    assert "a=2" in lines[1]
    assert "b=1" in lines[1]
    hashes = lines[2:]
    assert len(hashes) == 2
    assert hashes == sorted(hashes)
    assert len(set(hashes)) == 2


def test_load_exclusion_hashes_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "exclusion_hashes.txt"
    build_exclusion_list({"a": ["रामः", "सीता"]}, path)
    loaded = load_exclusion_hashes(path)
    assert loaded == frozenset({sentence_hash("रामः"), sentence_hash("सीता")})


def test_load_exclusion_hashes_skips_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "exclusion_hashes.txt"
    path.write_text("# comment one\n# comment two\nabc123\n", encoding="utf-8")
    assert load_exclusion_hashes(path) == frozenset({"abc123"})


def test_assert_not_excluded_raises_naming_the_offending_index() -> None:
    hashes = frozenset({sentence_hash("रामः")})
    with pytest.raises(LeakageError) as excinfo:
        assert_not_excluded(["सीता", "रामः"], hashes, label="tokenizer training text")
    message = str(excinfo.value)
    assert "tokenizer training text" in message
    assert "1" in message


def test_assert_not_excluded_names_up_to_five_offending_indices() -> None:
    hashes = frozenset({sentence_hash("रामः")})
    texts = ["रामः"] * 7
    with pytest.raises(LeakageError) as excinfo:
        assert_not_excluded(texts, hashes, label="x")
    message = str(excinfo.value)
    for index in range(5):
        assert str(index) in message


def test_assert_not_excluded_passes_for_a_clean_list() -> None:
    hashes = frozenset({sentence_hash("रामः")})
    assert assert_not_excluded(["सीता", "गच्छति"], hashes, label="clean") is None


# ---------------------------------------------------- English control list (E1 arms)


def test_sentence_hash_en_is_stable_and_whitespace_insensitive() -> None:
    assert sentence_hash_en("Rama goes to the forest") == sentence_hash_en(
        "  Rama goes to the forest\n"
    )


def test_sentence_hash_en_differs_for_different_text() -> None:
    assert sentence_hash_en("Rama goes") != sentence_hash_en("Sita speaks")


def test_sentence_hash_en_does_not_transliterate() -> None:
    """`sentence_hash` runs its input through Devanagari->SLP1 first; `sentence_hash_en`
    never does. On pure ASCII the two happen to agree, because transliteration passes
    Latin characters through untouched — which is exactly why the English list must not
    lean on that accident: the moment a sentence carries any Devanagari (a quoted term,
    say) the two part ways, and only `sentence_hash_en` still describes the text as the
    English tokenizer will see it."""
    assert sentence_hash_en("Rama goes") == sentence_hash("Rama goes")
    assert sentence_hash_en("रामः") != sentence_hash("रामः")


def test_build_exclusion_list_accepts_an_english_hash_fn(tmp_path: Path) -> None:
    path = tmp_path / "exclusion_hashes_en.txt"
    count = build_exclusion_list(
        {"a": ["Rama goes", "Sita speaks"], "b": ["Rama goes"]},
        path,
        hash_fn=sentence_hash_en,
    )
    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "English" in lines[0]
    assert lines[1].startswith("# sources: a=2, b=1")
    assert lines[2:] == sorted({sentence_hash_en("Rama goes"), sentence_hash_en("Sita speaks")})


def test_assert_not_excluded_accepts_an_english_hash_fn() -> None:
    hashes = frozenset({sentence_hash_en("Rama goes")})
    with pytest.raises(LeakageError) as excinfo:
        assert_not_excluded(
            ["Sita speaks", "Rama goes"],
            hashes,
            label="E1 training text",
            hash_fn=sentence_hash_en,
        )
    assert "E1 training text" in str(excinfo.value)


def test_assert_not_excluded_with_the_english_hash_fn_passes_for_a_clean_list() -> None:
    hashes = frozenset({sentence_hash_en("Rama goes")})
    assert (
        assert_not_excluded(
            ["Sita speaks"], hashes, label="clean", hash_fn=sentence_hash_en
        )
        is None
    )
