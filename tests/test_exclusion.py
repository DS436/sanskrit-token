"""Tests for `sanskrit_tok.data.exclusion` (CLAUDE.md §2.4: no evaluation leakage)."""

from pathlib import Path

import pytest

from sanskrit_tok.data.exclusion import (
    SHINGLE_K,
    LeakageError,
    assert_not_excluded,
    build_exclusion_list,
    build_shingle_index,
    has_shingle_overlap,
    letters_only,
    load_exclusion_hashes,
    sentence_hash,
    sentence_hash_en,
    shingles,
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


# ------------------------------------------------------- the near-duplicate shingle layer

#: One Itihāsa-style verse line in SLP1, with the danda and verse numbering a DCS copy of
#: the same text would punctuate differently. 60 letters, so it has shingles.
VERSE = "Darmakzetre kurukzetre samavetA yuyutsavaH | mAmakAH pARqavAScEva kimakurvata saMjaya ||"

#: The second half of the same verse as DCS might carry it: a different sentence by any
#: hash, the same letters in the same order.
HALF_VERSE = "mAmakAH pARqavAScEva kimakurvata saMjaya"

#: Unrelated Sanskrit of comparable length.
OTHER = "yadA yadA hi Darmasya glAnirBavati BArata | aByutTAnamaDarmasya tadAtmAnaM sfjAmyaham ||"


def test_letters_only_keeps_letters_and_case() -> None:
    assert letters_only("tat | tvam 2 asi ||") == "tattvamasi"
    assert letters_only("rAmaH") == "rAmaH"  # SLP1 case is contrastive, so it survives


def test_letters_only_of_empty_or_punctuation_is_empty() -> None:
    assert letters_only("") == ""
    assert letters_only("|| 12 ||") == ""


def test_shingles_are_every_window_and_none_when_too_short() -> None:
    assert shingles("abcdef", k=4) == {"abcd", "bcde", "cdef"}
    assert shingles("abc", k=4) == set()


def test_shingles_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        shingles("abcdef", k=0)


def test_a_half_verse_overlaps_the_full_verse_it_came_from() -> None:
    """The Itihāsa/DCS case: same letters, different sentence boundaries and punctuation."""
    index = build_shingle_index([VERSE])
    assert len(letters_only(HALF_VERSE)) >= SHINGLE_K
    assert has_shingle_overlap(HALF_VERSE, index)


def test_two_unrelated_sentences_do_not_overlap() -> None:
    index = build_shingle_index([VERSE])
    assert not has_shingle_overlap(OTHER, index)


def test_a_short_sentence_matches_only_by_exact_letters() -> None:
    short = "sUta uvAca"
    assert len(letters_only(short)) < SHINGLE_K
    index = build_shingle_index([short])
    assert has_shingle_overlap("sUta uvAca ||", index)  # same letters, different punctuation
    assert not has_shingle_overlap("fzaya UcuH", index)


def test_a_short_sentence_is_not_caught_inside_a_long_one() -> None:
    """The deliberate floor of the filter: sub-`k` text has no window to offer."""
    index = build_shingle_index([VERSE])
    assert not has_shingle_overlap("saMjaya", index)


def test_a_short_evaluation_sentence_never_collides_with_a_window() -> None:
    index = build_shingle_index(["sUta uvAca", VERSE])
    assert all(len(entry) in {SHINGLE_K, len(letters_only("sUta uvAca"))} for entry in index)


def test_empty_text_overlaps_nothing() -> None:
    assert not has_shingle_overlap("|| 4 ||", build_shingle_index([VERSE]))
    assert build_shingle_index(["", "|| ||"]) == frozenset()
