"""DCS CoNLL-U parsing and gold-boundary construction (Experiment 04, Task 1).

Every expectation here is hand-computed from `tests/fixtures/dcs_mini.conllu`, which is a
byte-for-byte copy of the first three sentences of the real DCS file
`Acintyastava/Acintyastava-0000-Acintyastava, 1-8248.conllu` (the third sentence truncated
after its last token line, so the fixture also exercises the `# text` / token-block
mismatch path that 1.5% of real DCS sentences take).

No network and no `data/raw/` access: the fixture is the corpus.
"""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

from sanskrit_tok.data.boundaries import (
    BOUNDARY_MARKER,
    GoldSentence,
    align_segments,
    build_gold_sentence,
    mark,
    stem_boundary,
)
from sanskrit_tok.data.dcs import (
    DCS_COMMIT,
    DCS_CONLLU_DIR,
    DCS_REPO,
    DcsSentence,
    DcsToken,
    DcsWord,
    iter_conllu_sentences,
    sentence_is_human_verified,
)
from sanskrit_tok.data.exclusion import sentence_hash, sentence_hash_slp1
from sanskrit_tok.encoding import to_slp1

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "dcs_mini.conllu"
INGEST_PY = REPO_ROOT / "experiments" / "04_morph_constrained" / "ingest_dcs.py"
DCS_YAML = REPO_ROOT / "experiments" / "04_morph_constrained" / "dcs.yaml"

MARK = BOUNDARY_MARKER


def _load_ingest_module() -> ModuleType:
    """`experiments/04_morph_constrained/` starts with a digit and cannot be imported;
    load `ingest_dcs.py` by path, as `tests/test_split_corpora.py` does for exp03."""
    spec = importlib.util.spec_from_file_location("exp04_ingest_dcs", INGEST_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest_module() -> ModuleType:
    return _load_ingest_module()


@pytest.fixture(scope="module")
def sentences() -> list[DcsSentence]:
    return list(iter_conllu_sentences(FIXTURE))


# --------------------------------------------------------------------------- constants


def test_module_constants_pin_the_source() -> None:
    assert DCS_REPO == "OliverHellwig/sanskrit"
    assert DCS_CONLLU_DIR == "dcs/data/conllu/files"
    assert len(DCS_COMMIT) == 40
    assert all(character in "0123456789abcdef" for character in DCS_COMMIT)


# ------------------------------------------------------------------------------ parser


def test_parser_yields_every_sentence_with_its_document_metadata(
    sentences: list[DcsSentence],
) -> None:
    assert len(sentences) == 3
    # `sent_id` is a string: 4% of DCS ids are sub-sentence ids like `746028_1`.
    assert [s.sent_id for s in sentences] == ["555547", "555548", "555549"]
    for sentence in sentences:
        assert sentence.text_id == 415
        assert sentence.text_name == "Acintyastava"
        assert sentence.chapter == "Acintyastava, 1"


def test_parser_reads_the_sandhied_text_line(sentences: list[DcsSentence]) -> None:
    assert sentences[0].text_iast == "pratītyajānāṃ bhāvānāṃ naiḥsvābhāvyaṃ jagāda yaḥ"
    assert sentences[2].text_iast == "yathā tvayā mahāyāne dharmanairātmyam ātmanā"


def test_multiword_ranges_group_their_tokens_into_one_surface_word(
    sentences: list[DcsSentence],
) -> None:
    first = sentences[0]
    assert [word.surface_iast for word in first.words] == [
        "pratītyajānāṃ",
        "bhāvānāṃ",
        "naiḥsvābhāvyaṃ",
        "jagāda",
        "yaḥ",
    ]
    # `1-2` groups two tokens; every other word here is a single token line.
    assert [len(word.tokens) for word in first.words] == [2, 1, 1, 1, 1]
    assert [token.unsandhied_iast for token in first.words[0].tokens] == ["pratītya", "jānām"]

    second = sentences[1]
    assert [word.surface_iast for word in second.words] == [
        "taṃ",
        "namāmy",
        "asamajñānam",
        "acintyam",
        "anidarśanam",
    ]
    assert [len(word.tokens) for word in second.words] == [1, 1, 3, 2, 2]
    assert [token.unsandhied_iast for token in second.words[2].tokens] == ["a", "sama", "jñānam"]


def test_token_fields_come_from_the_right_columns(sentences: list[DcsSentence]) -> None:
    token = sentences[0].words[1].tokens[0]
    assert token.id == 3
    assert token.surface_iast == "bhāvānāṃ"
    assert token.lemma_iast == "bhāva"
    assert token.upos == "NOUN"
    assert token.feats == "Case=Gen|Gender=Masc|Number=Plur"
    assert token.unsandhied_iast == "bhāvānām"
    assert token.reconstructed is True
    assert token.is_cpd is False


def test_reconstructed_flag_is_read_from_misc(sentences: list[DcsSentence]) -> None:
    # `jagāda` carries no `UnsandhiedReconstructed`; its neighbours do.
    jagada = sentences[0].words[3].tokens[0]
    assert jagada.unsandhied_iast == "jagāda"
    assert jagada.reconstructed is False
    assert sentences[0].words[4].tokens[0].reconstructed is True


def test_case_cpd_marks_a_compound_member(sentences: list[DcsSentence]) -> None:
    compound = sentences[1].words[2]
    assert [token.is_cpd for token in compound.tokens] == [False, True, False]


def test_empty_feats_cell_parses_as_empty_not_as_a_crash(sentences: list[DcsSentence]) -> None:
    # `a	a	PART	_		_	_	_	...` — the FEATS cell is empty rather than `_`.
    particle = sentences[1].words[2].tokens[0]
    assert particle.upos == "PART"
    assert particle.feats == ""


def test_sentence_is_human_verified_is_false_when_any_token_is_reconstructed(
    sentences: list[DcsSentence],
) -> None:
    assert [sentence_is_human_verified(s) for s in sentences] == [False, False, False]


def test_sentence_is_human_verified_is_true_when_no_token_is_reconstructed() -> None:
    token = DcsToken(
        id=1,
        surface_iast="jagāda",
        lemma_iast="gad",
        upos="VERB",
        feats="Tense=Past",
        unsandhied_iast="jagāda",
        reconstructed=False,
        is_cpd=False,
    )
    sentence = DcsSentence(
        sent_id="1",
        text_id=1,
        text_name="T",
        chapter="C",
        text_iast="jagāda",
        words=(DcsWord(surface_iast="jagāda", tokens=(token,)),),
    )
    assert sentence_is_human_verified(sentence) is True


# --------------------------------------------------------------------------- alignment


def test_align_segments_locates_a_sandhi_boundary() -> None:
    assert align_segments("tadapi", ["tad", "api"]) == [3]


def test_align_segments_handles_a_final_character_changed_by_sandhi() -> None:
    # `pratItyajAnAM` <- `pratItya` + `jAnAm`: identical but for the last character.
    assert align_segments("pratItyajAnAM", ["pratItya", "jAnAm"]) == [8]


def test_align_segments_locates_two_boundaries_in_a_compound() -> None:
    assert align_segments("asamajYAnam", ["a", "sama", "jYAnam"]) == [1, 5]


def test_align_segments_returns_empty_for_a_single_segment() -> None:
    assert align_segments("BAvAnAM", ["BAvAnAm"]) == []


def test_align_segments_returns_none_on_garbage() -> None:
    assert align_segments("abcdef", ["zzz", "yyy"]) is None


def test_align_segments_returns_none_on_empty_input() -> None:
    assert align_segments("", ["a"]) is None
    assert align_segments("tadapi", []) is None


def test_align_segments_rejects_non_monotone_offsets() -> None:
    # Two identical segments against a surface that holds only one copy: the second
    # segment cannot start after the first without leaving the surface.
    assert align_segments("ab", ["ab", "ab"]) is None


# ------------------------------------------------------------------------ stem boundary


def test_stem_boundary_is_the_longest_common_prefix_with_the_lemma() -> None:
    # `BAvAnAm` / `BAva`: the common prefix is `BAv`, not `BAva` — the lemma ends in a
    # short `a` where the inflected stem has a long `A` (the Exp04 brief's worked example
    # says 4; the SLP1 strings say 3, and the definition is the longest common prefix).
    assert stem_boundary("BAvAnAm", "BAva") == 3
    assert stem_boundary("jYAnam", "jYAna") == 5
    assert stem_boundary("yaH", "yad") == 2
    assert stem_boundary("namAmi", "nam") == 3


def test_stem_boundary_is_none_when_the_prefix_is_too_short() -> None:
    assert stem_boundary("jAnAm", "ja") is None
    assert stem_boundary("jagAda", "gad") is None


def test_stem_boundary_is_none_when_the_ending_would_be_empty() -> None:
    assert stem_boundary("sama", "sama") is None
    assert stem_boundary("yaTA", "yaTA") is None


def test_stem_boundary_is_none_for_an_unknown_lemma() -> None:
    assert stem_boundary("BAvAnAm", "_") is None
    assert stem_boundary("BAvAnAm", "") is None


# -------------------------------------------------------------------------------- mark


def test_mark_inserts_the_marker_at_each_offset() -> None:
    assert mark("tadapi", [3]) == f"tad{MARK}api"
    assert mark("asamajYAnam", [1, 5]) == f"a{MARK}sama{MARK}jYAnam"
    assert mark("tadapi", []) == "tadapi"


def test_mark_rejects_an_out_of_range_offset() -> None:
    with pytest.raises(ValueError):
        mark("tad", [9])


# ------------------------------------------------------------------------ GoldSentence


def test_gold_sentence_for_the_first_fixture_sentence(sentences: list[DcsSentence]) -> None:
    gold = build_gold_sentence(sentences[0])
    assert gold.text_slp1 == "pratItyajAnAM BAvAnAM nEHsvABAvyaM jagAda yaH"
    assert gold.oracle_split_slp1 == "pratItya jAnAm BAvAnAm nEHsvABAvyam jagAda yaH"
    assert gold.segment_offsets == [[8], [], [], [], []]
    assert gold.stem_offsets == [[5], [18], [34], [], [45]]
    assert gold.t5_marked == (
        f"pratI{MARK}tya{MARK}jAnAM BAv{MARK}AnAM nEHsvABAvya{MARK}M jagAda ya{MARK}H"
    )
    assert gold.t6_marked == (
        f"pratI{MARK}tya jAnAm BAv{MARK}AnAm nEHsvABAvya{MARK}m jagAda ya{MARK}H"
    )
    assert gold.n_words == 5
    assert gold.n_words_aligned == 5
    assert gold.human_verified is False


def test_gold_sentence_for_the_compound_fixture_sentence(sentences: list[DcsSentence]) -> None:
    gold = build_gold_sentence(sentences[1])
    assert gold.text_slp1 == "taM namAmy asamajYAnam acintyam anidarSanam"
    assert gold.oracle_split_slp1 == "tam namAmi a sama jYAnam a cintyam a nidarSanam"
    assert gold.segment_offsets == [[], [], [1, 5], [1], [1]]
    assert gold.stem_offsets == [[2], [7], [23], [31], [46]]
    assert gold.n_words == 5
    assert gold.n_words_aligned == 5


def test_gold_sentence_marks_words_absent_from_the_token_block_as_unaligned(
    sentences: list[DcsSentence],
) -> None:
    # `# text` has five words; the (truncated) token block reconstructs only the first
    # three, which is exactly the shape of the ~1.4% of real DCS sentences that mismatch.
    gold = build_gold_sentence(sentences[2])
    assert gold.text_slp1 == "yaTA tvayA mahAyAne DarmanErAtmyam AtmanA"
    assert gold.oracle_split_slp1 == gold.text_slp1
    assert gold.segment_offsets == [[], [], [], None, None]
    assert gold.stem_offsets == [[], [8], [18], [], []]
    assert gold.n_words == 5
    assert gold.n_words_aligned == 3
    assert gold.t5_marked == f"yaTA tva{MARK}yA mahAyAn{MARK}e DarmanErAtmyam AtmanA"


def test_gold_sentence_markers_are_removable_without_trace(sentences: list[DcsSentence]) -> None:
    for sentence in sentences:
        gold = build_gold_sentence(sentence)
        assert gold.t5_marked.replace(MARK, "") == gold.text_slp1
        assert gold.t6_marked.replace(MARK, "") == gold.oracle_split_slp1


def test_gold_sentence_offsets_agree_with_the_marked_strings(
    sentences: list[DcsSentence],
) -> None:
    for sentence in sentences:
        gold = build_gold_sentence(sentence)
        flat = sorted(offset for word in gold.stem_offsets for offset in word)
        assert gold.t6_marked == mark(gold.oracle_split_slp1, flat)


def test_gold_sentence_is_json_serialisable(sentences: list[DcsSentence]) -> None:
    gold = build_gold_sentence(sentences[0])
    restored = GoldSentence(**json.loads(json.dumps(gold.to_dict())))
    assert restored == gold


# ------------------------------------------------------------------------- held-out split


def test_heldout_texts_are_deterministic_and_whole(ingest_module: ModuleType) -> None:
    assign_heldout_texts = ingest_module.assign_heldout_texts
    ids = [7, 3, 11, 2, 5, 13, 1, 17, 19, 23]
    first = assign_heldout_texts(ids, fraction=0.2, seed=0)
    second = assign_heldout_texts(list(reversed(ids)), fraction=0.2, seed=0)
    assert first == second
    assert len(first) == 2
    assert first <= set(ids)


def test_heldout_fraction_of_zero_still_yields_one_text(ingest_module: ModuleType) -> None:
    assign_heldout_texts = ingest_module.assign_heldout_texts
    assert len(assign_heldout_texts([1, 2, 3], fraction=0.0, seed=0)) == 1


# ---------------------------------------------------------------------------- exclusion


def test_sentence_hash_slp1_agrees_with_sentence_hash_on_devanagari() -> None:
    devanagari = "तदपि"
    assert sentence_hash(devanagari) == sentence_hash_slp1(to_slp1(devanagari, "devanagari"))


def test_sentence_hash_slp1_strips_before_hashing() -> None:
    assert sentence_hash_slp1("  tadapi \n") == sentence_hash_slp1("tadapi")


def test_ingest_drops_sentences_already_in_the_exclusion_list(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    ingest = ingest_module.ingest
    excluded = {sentence_hash_slp1("taM namAmy asamajYAnam acintyam anidarSanam")}
    out_dir = tmp_path / "dcs"
    manifest = ingest(
        conllu_dir=FIXTURE.parent,
        out_dir=out_dir,
        excluded=frozenset(excluded),
        heldout_fraction=0.05,
        seed=0,
        min_words=2,
        files=[FIXTURE],
    )
    assert manifest["n_dropped_excluded"] == 1
    written = [
        json.loads(line)
        for path in (out_dir / "train.jsonl", out_dir / "heldout.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(written) == 2
    assert all("namAmy" not in record["text_slp1"] for record in written)
    assert {record["sent_id"] for record in written} == {"555547", "555549"}
    assert manifest["n_sentences_text_mismatch"] == 1
    assert manifest["n_words"] == 10
    assert manifest["n_words_aligned"] == 8


def test_ingest_drops_short_sentences(tmp_path: Path, ingest_module: ModuleType) -> None:
    ingest = ingest_module.ingest
    manifest = ingest(
        conllu_dir=FIXTURE.parent,
        out_dir=tmp_path / "dcs",
        excluded=frozenset(),
        heldout_fraction=0.05,
        seed=0,
        min_words=6,
        files=[FIXTURE],
    )
    assert manifest["n_dropped_min_words"] == 3
    assert manifest["splits"]["train"]["n_sentences"] == 0


def test_ingest_writes_both_splits_and_a_manifest(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    ingest = ingest_module.ingest
    out_dir = tmp_path / "dcs"
    manifest = ingest(
        conllu_dir=FIXTURE.parent,
        out_dir=out_dir,
        excluded=frozenset(),
        heldout_fraction=1.0,
        seed=0,
        min_words=2,
        files=[FIXTURE],
    )
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "heldout.jsonl").exists()
    # The single fixture text is the whole corpus, so a fraction of 1.0 holds it all out.
    assert manifest["splits"]["heldout"]["n_sentences"] == 3
    assert manifest["splits"]["heldout"]["n_texts"] == 1
    assert manifest["splits"]["train"]["n_sentences"] == 0
    assert manifest["human_verified_fraction"] == 0.0
    assert manifest["alignment_rate"] == pytest.approx(13 / 15)
    record = json.loads((out_dir / "heldout.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["text_id"] == 415
    assert record["text_iast"] == "pratītyajānāṃ bhāvānāṃ naiḥsvābhāvyaṃ jagāda yaḥ"
    assert set(record) == {
        "sent_id",
        "text_id",
        "text_iast",
        "text_slp1",
        "oracle_split_slp1",
        "segment_offsets",
        "stem_offsets",
        "t5_marked",
        "t6_marked",
        "n_words",
        "n_words_aligned",
        "human_verified",
    }


def test_ingest_drops_training_sentences_that_duplicate_a_heldout_sentence(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    """A whole-text split alone does not make the two sides disjoint.

    DCS repeats formulaic lines verbatim across texts (`sUta uvAca`), so a training text
    can hold a sentence identical to one in a held-out text. Here the same fixture is
    presented twice under two `text_id`s: whichever is held out, the other's three
    sentences are duplicates and must not reach `train.jsonl`.
    """
    twin = tmp_path / "twin.conllu"
    twin.write_text(
        FIXTURE.read_text(encoding="utf-8").replace("## text_id: 415", "## text_id: 999"),
        encoding="utf-8",
    )
    out_dir = tmp_path / "dcs"
    manifest = ingest_module.ingest(
        conllu_dir=tmp_path,
        out_dir=out_dir,
        excluded=frozenset(),
        heldout_fraction=0.5,
        seed=0,
        min_words=2,
        files=[FIXTURE, twin],
    )
    assert manifest["splits"]["heldout"]["n_sentences"] == 3
    assert manifest["splits"]["train"]["n_sentences"] == 0
    assert manifest["n_dropped_heldout_duplicate"] == 3
    assert (out_dir / "train.jsonl").read_text(encoding="utf-8") == ""


def test_random_seed_zero_is_what_assign_heldout_texts_uses(ingest_module: ModuleType) -> None:
    assign_heldout_texts = ingest_module.assign_heldout_texts
    ids = list(range(100))
    expected = set(random.Random(0).sample(sorted(ids), 5))
    assert assign_heldout_texts(ids, fraction=0.05, seed=0) == expected
