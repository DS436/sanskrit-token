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
    stem_boundary_lcp,
    stem_cut_inside_lemma,
    stem_cut_is_fused,
    stem_rule_audit,
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
from sanskrit_tok.data.exclusion import (
    build_exclusion_list,
    build_shingle_index,
    sentence_hash,
    sentence_hash_slp1,
)
from sanskrit_tok.encoding import to_slp1

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "dcs_mini.conllu"
INGEST_PY = REPO_ROOT / "experiments" / "04_morph_constrained" / "ingest_dcs.py"
DCS_YAML = REPO_ROOT / "experiments" / "04_morph_constrained" / "dcs.yaml"
BUILD_EXCLUSION_PY = REPO_ROOT / "experiments" / "02_tpp_parallel" / "build_exclusion.py"

MARK = BOUNDARY_MARKER


def _load_module(name: str, path: Path) -> ModuleType:
    """Load an experiment script by path: both experiment directories start with a digit
    and cannot be imported, the pattern `tests/test_split_corpora.py` established."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest_module() -> ModuleType:
    return _load_module("exp04_ingest_dcs", INGEST_PY)


@pytest.fixture(scope="module")
def build_exclusion_module() -> ModuleType:
    return _load_module("exp02_build_exclusion", BUILD_EXCLUSION_PY)


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


# The offsets `align_segments` actually produces where vowel sandhi contracts two
# characters into one. The earlier decisions entry claimed the fused character always joins
# the *left* segment; it does not (docs/decisions.md, 2026-09-05, "Correction: the fused
# character does not consistently join the left segment"). The rule these four pin: the
# fused character joins the left segment **unless** the sandhi output equals the right
# segment's first character (a+E -> E, a+A -> A, a+O -> O), in which case `difflib` matches
# it to the right segment and the boundary lands before it. The bias is identical across
# arms, so paired comparisons are unaffected; absolute MorphScore is not portable, which is
# why MorphScore also reports a symmetric +/-1-character tolerant variant.
FUSED_SANDHI_CASES = [
    # (surface, segments, expected offsets, which side the fused character joined)
    ("rAmeti", ["rAma", "iti"], [4], "left: a + i -> e, and `e` is not `iti`'s first char"),
    ("tatrEva", ["tatra", "eva"], [5], "left: a + e -> E, and `E` is not `eva`'s first char"),
    ("vacanenEkam", ["vacanena", "Ekam"], [7], "right: a + E -> E == `Ekam`'s first char"),
    ("sAgacCat", ["sa", "AgacCat"], [1], "right: a + A -> A == `AgacCat`'s first char"),
    # No contraction at all: `tat` + `api` is a consonant sandhi, the concatenation and the
    # surface are the same length, and the boundary is unambiguous.
    ("tadapi", ["tat", "api"], [3], "no contraction"),
]


@pytest.mark.parametrize(("surface", "segments", "expected", "side"), FUSED_SANDHI_CASES)
def test_align_segments_pins_where_a_fused_sandhi_character_lands(
    surface: str, segments: list[str], expected: list[int], side: str
) -> None:
    assert align_segments(surface, segments) == expected, side


def test_fused_character_joins_the_left_segment_only_sometimes() -> None:
    """The claim these tests correct, stated as an assertion.

    If the fused character always joined the left segment, every offset below would equal
    the first segment's length. Two of the four do not, which is the whole point.
    """
    joins_left = {
        surface: align_segments(surface, segments) == [len(segments[0])]
        for surface, segments, _expected, _side in FUSED_SANDHI_CASES[:4]
    }
    assert joins_left == {
        "rAmeti": True,
        "tatrEva": True,
        "vacanenEkam": False,
        "sAgacCat": False,
    }


# ------------------------------------------------------------------------ stem boundary


def test_stem_boundary_cuts_at_the_lemma_when_the_surface_preserves_it() -> None:
    assert stem_boundary("vIram", "vIra") == 4  # `vIra|m`
    assert stem_boundary("jYAnam", "jYAna") == 5
    assert stem_boundary("namAmi", "nam") == 3


def test_stem_boundary_cuts_before_a_fused_stem_final_vowel() -> None:
    # No character offset is *the* boundary in these: the surface vowel is the stem's final
    # vowel and the ending's first at once, so the cut goes before it and is flagged fused.
    assert stem_boundary("vIrAH", "vIra") == 3  # `vIr|AH`, from `vIra` + `as`
    assert stem_boundary("anuBAvena", "anuBAva") == 6  # `anuBAv|ena`, from `anuBAva` + `ina`
    assert stem_cut_is_fused("vIrAH", "vIra", 3)
    assert stem_cut_is_fused("anuBAvena", "anuBAva", 6)
    assert stem_cut_inside_lemma("vIrAH", "vIra", 3)  # inside by character count, and flagged
    assert not stem_cut_is_fused("vIram", "vIra", 4)
    assert not stem_cut_inside_lemma("vIram", "vIra", 4)


def test_stem_boundary_is_none_when_the_cut_would_fall_inside_the_lemma_body() -> None:
    # The defect the sandhi-aware rule removes: the LCP rule cut at the prefix anyway.
    assert stem_boundary("gacCati", "gam") is None
    assert stem_boundary_lcp("gacCati", "gam") == 2
    assert stem_boundary("jagAda", "gad") is None


def test_stem_boundary_is_none_for_the_closed_class_parts_of_speech() -> None:
    assert stem_boundary("yaH", "yad", "PRON") is None
    assert stem_boundary("mama", "mad", "PRON") is None
    # UPOS is what decides these two: the string pair alone would yield a cut.
    assert stem_boundary("nityam", "nitya") == 5
    assert stem_boundary("nityam", "nitya", "ADV") is None
    assert stem_boundary("dvau", "dvi") == 2
    assert stem_boundary("dvau", "dvi", "NUM") is None


def test_stem_boundary_is_none_when_the_prefix_is_too_short() -> None:
    assert stem_boundary("jAnAm", "ja") is None
    assert stem_boundary("jagAda", "gad") is None


def test_stem_boundary_is_none_when_the_ending_would_be_empty() -> None:
    assert stem_boundary("sama", "sama") is None
    assert stem_boundary("yaTA", "yaTA") is None


def test_stem_boundary_is_none_for_an_unknown_lemma() -> None:
    assert stem_boundary("BAvAnAm", "_") is None
    assert stem_boundary("BAvAnAm", "") is None


def test_stem_boundary_lcp_is_the_old_rule_unchanged() -> None:
    # Kept only so the ingestion can measure what the sandhi-aware rule replaced.
    assert stem_boundary_lcp("BAvAnAm", "BAva") == 3
    assert stem_boundary_lcp("yaH", "yad") == 2
    assert stem_boundary_lcp("sama", "sama") is None


def test_stem_rule_audit_counts_both_rules(sentences: list[DcsSentence]) -> None:
    audit = stem_rule_audit(sentences[0])
    assert set(audit) == {"sandhi_aware", "lcp"}
    # The sandhi-aware rule makes fewer cuts and none of them fall inside the lemma body.
    assert audit["sandhi_aware"].n_cuts < audit["lcp"].n_cuts
    assert audit["sandhi_aware"].to_dict()["n_inside_lemma_excluding_fused"] == 0
    assert audit["lcp"].to_dict()["n_inside_lemma_excluding_fused"] > 0


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
    # `yaH` / `yad` no longer carries a stem cut: the lemma is consonant-final and the
    # surface diverges inside its body, which the sandhi-aware rule declines to guess at.
    assert gold.stem_offsets == [[5], [18], [34], [], []]
    assert gold.t5_marked == (
        f"pratI{MARK}tya{MARK}jAnAM BAv{MARK}AnAM nEHsvABAvya{MARK}M jagAda yaH"
    )
    assert gold.t5seg_marked == f"pratItya{MARK}jAnAM BAvAnAM nEHsvABAvyaM jagAda yaH"
    assert gold.t6_marked == (
        f"pratI{MARK}tya jAnAm BAv{MARK}AnAm nEHsvABAvya{MARK}m jagAda yaH"
    )
    assert gold.n_words == 5
    assert gold.n_words_aligned == 5
    assert gold.human_verified is False


def test_gold_sentence_for_the_compound_fixture_sentence(sentences: list[DcsSentence]) -> None:
    gold = build_gold_sentence(sentences[1])
    assert gold.text_slp1 == "taM namAmy asamajYAnam acintyam anidarSanam"
    assert gold.oracle_split_slp1 == "tam namAmi a sama jYAnam a cintyam a nidarSanam"
    assert gold.segment_offsets == [[], [], [1, 5], [1], [1]]
    assert gold.stem_offsets == [[], [7], [23], [], [46]]
    assert gold.t5seg_marked == (
        f"taM namAmy a{MARK}sama{MARK}jYAnam a{MARK}cintyam a{MARK}nidarSanam"
    )
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
    assert gold.stem_offsets == [[], [], [18], [], []]
    assert gold.n_words == 5
    assert gold.n_words_aligned == 3
    assert gold.t5_marked == f"yaTA tvayA mahAyAn{MARK}e DarmanErAtmyam AtmanA"
    # Segment-only marking leaves a sentence with no internal segment boundary untouched.
    assert gold.t5seg_marked == gold.text_slp1


def test_gold_sentence_markers_are_removable_without_trace(sentences: list[DcsSentence]) -> None:
    for sentence in sentences:
        gold = build_gold_sentence(sentence)
        assert gold.t5_marked.replace(MARK, "") == gold.text_slp1
        assert gold.t5seg_marked.replace(MARK, "") == gold.text_slp1
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
        "t5seg_marked",
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


def test_ingest_drops_training_sentences_near_duplicating_an_evaluation_sentence(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    """The near-duplicate layer: same letters, different sentence boundaries and hash.

    The fixture's first sentence, re-punctuated and split in two the way a parallel corpus
    would carry it, hashes to something else entirely — and shares far more than 24 letters
    with it, so the shingle filter catches what the hash cannot.
    """

    evaluation = "pratItyajAnAM BAvAnAM || nEHsvABAvyaM jagAda yaH ||"
    out_dir = tmp_path / "dcs"
    manifest = ingest_module.ingest(
        conllu_dir=FIXTURE.parent,
        out_dir=out_dir,
        excluded=frozenset(),
        heldout_fraction=0.0,  # floors to one text; the fixture is one text, so nothing trains
        seed=0,
        min_words=2,
        files=[FIXTURE],
        shingle_indices={"parallel_eval": build_shingle_index([evaluation])},
    )
    # Everything is held out here, so the *train* filter has nothing to act on; what this
    # asserts is that the index reached the manifest and the held-out split was not filtered.
    assert manifest["shingle_k"] == 24
    assert manifest["n_dropped_shingle"] == 0
    assert manifest["splits"]["heldout"]["n_sentences"] == 3
    assert set(manifest["n_dropped_shingle_per_source"]) == {"parallel_eval", "dcs_heldout"}
    assert manifest["shingle_index_sizes"]["parallel_eval"] > 0


def test_ingest_shingle_filter_drops_a_repunctuated_training_sentence(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    """The same fixture under a second `text_id`, with the first held out: the training
    copy is dropped by the *hash* layer, and a re-punctuated evaluation sentence drops the
    rest by shingle overlap."""
    twin = tmp_path / "twin.conllu"
    twin.write_text(
        FIXTURE.read_text(encoding="utf-8")
        .replace("## text_id: 415", "## text_id: 999")
        .replace("pratītyajānāṃ", "pratītyajānāṁ"),  # a different hash, the same letters
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
        shingle_indices={},  # only the DCS held-out split guards, and that is the point
    )
    assert manifest["splits"]["heldout"]["n_sentences"] == 3
    assert manifest["n_dropped_shingle"] >= 1
    assert manifest["n_dropped_shingle_per_source"]["dcs_heldout"] >= 1


def test_ingest_records_both_stem_rules(tmp_path: Path, ingest_module: ModuleType) -> None:
    manifest = ingest_module.ingest(
        conllu_dir=FIXTURE.parent,
        out_dir=tmp_path / "dcs",
        excluded=frozenset(),
        heldout_fraction=1.0,
        seed=0,
        min_words=2,
        files=[FIXTURE],
    )
    audit = manifest["stem_rule"]
    assert set(audit) == {"lcp", "sandhi_aware"}
    assert audit["sandhi_aware"]["n_inside_lemma_excluding_fused"] == 0
    assert audit["lcp"]["fraction_inside_lemma_excluding_fused"] > 0


def test_random_seed_zero_is_what_assign_heldout_texts_uses(ingest_module: ModuleType) -> None:
    assign_heldout_texts = ingest_module.assign_heldout_texts
    ids = list(range(100))
    expected = set(random.Random(0).sample(sorted(ids), 5))
    assert assign_heldout_texts(ids, fraction=0.05, seed=0) == expected


# ------------------------------------------------------------- ingestion split invariants

_SYNTHETIC_HEADER = """## text: {name}
## text_id: {text_id}
## chapter: {name}, 1
## chapter_id: {text_id}
"""

_SYNTHETIC_SENTENCE = """# text = {text}
# sent_id = {sent_id}
1\ttad\ttad\tPRON\t_\t_\t_\t_\t_\tUnsandhied=tad
2\tapi\tapi\tADV\t_\t_\t_\t_\t_\tUnsandhied=api

"""


def _write_conllu(path: Path, blocks: list[str]) -> Path:
    """A minimal two-word-per-sentence CoNLL-U file assembled from header/sentence blocks."""
    path.write_text("".join(blocks), encoding="utf-8")
    return path


def _synthetic_text(directory: Path, name: str, text_id: int, texts: list[str]) -> Path:
    """One well-formed file: a header and one sentence per entry of `texts`."""
    blocks = [_SYNTHETIC_HEADER.format(name=name, text_id=text_id)]
    blocks += [
        _SYNTHETIC_SENTENCE.format(text=text, sent_id=f"{text_id}{index}")
        for index, text in enumerate(texts, start=1)
    ]
    return _write_conllu(directory / f"{name}.conllu", blocks)


def test_ingest_accepts_a_well_formed_two_file_corpus(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    """The baseline the three invariant tests below deviate from, one at a time."""
    first = _synthetic_text(tmp_path, "alpha", 1, ["tad api", "tad api ca"])
    second = _synthetic_text(tmp_path, "beta", 2, ["tvam eva"])
    out_dir = tmp_path / "dcs"
    manifest = ingest_module.ingest(
        conllu_dir=tmp_path,
        out_dir=out_dir,
        excluded=frozenset(),
        heldout_fraction=0.5,
        seed=0,
        min_words=2,
        files=[first, second],
    )
    assert manifest["n_sentences_kept"] == 3
    assert manifest["n_texts"] == 2
    assert manifest["splits"]["train"]["n_texts"] == 1
    assert manifest["splits"]["heldout"]["n_texts"] == 1
    assert (out_dir / "manifest.json").exists()


def test_ingest_rejects_a_sentence_whose_text_id_differs_from_its_file_header(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    """Routing is by the file's `## text_id`, so a sentence declaring another one is fatal.

    A file holding a second `## text_id:` block is the shape this can really take: the
    parser carries the new id into the sentences that follow it, while `read_text_id` only
    ever sees the first, so those sentences would be routed on the wrong text's word.
    """
    good = _synthetic_text(tmp_path, "alpha", 1, ["tad api"])
    smuggled = _write_conllu(
        tmp_path / "beta.conllu",
        [
            _SYNTHETIC_HEADER.format(name="beta", text_id=2),
            _SYNTHETIC_SENTENCE.format(text="tvam eva", sent_id="21"),
            _SYNTHETIC_HEADER.format(name="gamma", text_id=3),
            _SYNTHETIC_SENTENCE.format(text="sa eva", sent_id="31"),
        ],
    )
    with pytest.raises(ingest_module.IngestError) as error:
        ingest_module.ingest(
            conllu_dir=tmp_path,
            out_dir=tmp_path / "dcs",
            excluded=frozenset(),
            heldout_fraction=0.5,
            seed=0,
            min_words=2,
            files=[good, smuggled],
        )
    message = str(error.value)
    assert str(smuggled) in message
    assert "text_id 3" in message and "header says 2" in message


def test_read_text_ids_rejects_a_file_with_no_text_id_header(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    """No `-1` pseudo-text: files with no header would all be routed together, silently."""
    good = _synthetic_text(tmp_path, "alpha", 1, ["tad api"])
    headerless = _write_conllu(
        tmp_path / "beta.conllu",
        ["## text: beta\n", _SYNTHETIC_SENTENCE.format(text="tvam eva", sent_id="21")],
    )
    assert ingest_module.read_text_ids([good]) == {good: 1}
    with pytest.raises(ingest_module.IngestError) as error:
        ingest_module.read_text_ids([good, headerless])
    assert str(headerless) in str(error.value)
    assert "text_id" in str(error.value)


def test_ingest_fails_on_a_headerless_file_before_writing_anything(
    tmp_path: Path, ingest_module: ModuleType
) -> None:
    headerless = _write_conllu(
        tmp_path / "beta.conllu",
        ["## text: beta\n", _SYNTHETIC_SENTENCE.format(text="tvam eva", sent_id="21")],
    )
    out_dir = tmp_path / "dcs"
    with pytest.raises(ingest_module.IngestError):
        ingest_module.ingest(
            conllu_dir=tmp_path,
            out_dir=out_dir,
            excluded=frozenset(),
            heldout_fraction=0.5,
            seed=0,
            min_words=2,
            files=[headerless],
        )
    assert not (out_dir / "manifest.json").exists()


def test_assert_splits_disjoint_rejects_a_text_in_both_splits(ingest_module: ModuleType) -> None:
    ingest_module.assert_splits_disjoint({1, 2}, {3, 4})
    ingest_module.assert_splits_disjoint(set(), set())
    with pytest.raises(ingest_module.IngestError) as error:
        ingest_module.assert_splits_disjoint({1, 2, 3}, {3, 4})
    assert "[3]" in str(error.value)


# ------------------------------------------------------- exclusion-list regeneration guards


def _write_heldout(path: Path, texts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"text_slp1": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )
    return path


def test_load_dcs_heldout_reads_the_sandhied_slp1_text(
    tmp_path: Path, build_exclusion_module: ModuleType
) -> None:
    path = _write_heldout(tmp_path / "dcs" / "heldout.jsonl", ["tadapi", "tvameva"])
    assert build_exclusion_module.load_dcs_heldout(path) == ["tadapi", "tvameva"]


def test_load_dcs_heldout_fails_when_the_file_is_absent(
    tmp_path: Path, build_exclusion_module: ModuleType
) -> None:
    """A fresh clone must not quietly regenerate the list 30k hashes lighter."""
    missing = tmp_path / "dcs" / "heldout.jsonl"
    with pytest.raises(build_exclusion_module.MissingDcsHeldoutError) as error:
        build_exclusion_module.load_dcs_heldout(missing)
    assert "ingest_dcs.py" in str(error.value)
    assert "--allow-missing-dcs" in str(error.value)


def test_load_dcs_heldout_returns_empty_when_missing_is_explicitly_allowed(
    tmp_path: Path, build_exclusion_module: ModuleType
) -> None:
    assert build_exclusion_module.load_dcs_heldout(
        tmp_path / "dcs" / "heldout.jsonl", allow_missing=True
    ) == []


def test_exclusion_list_may_not_shrink_without_allow_shrink(
    tmp_path: Path, build_exclusion_module: ModuleType
) -> None:
    """A leakage guard that silently stops covering a sentence is worse than a broken run."""
    committed = tmp_path / "exclusion_hashes.txt"
    build_exclusion_list({"a": ["tadapi", "tvameva"]}, committed, hash_fn=sentence_hash_slp1)
    before = committed.read_text(encoding="utf-8")

    superset = {sentence_hash_slp1(text) for text in ("tadapi", "tvameva", "sEva")}
    build_exclusion_module.assert_superset_of_committed(superset, committed)

    shrunk = {sentence_hash_slp1("tadapi")}
    with pytest.raises(build_exclusion_module.ExclusionShrinkError) as error:
        build_exclusion_module.write_list(
            {"a": ["tadapi"]}, committed, hash_fn=sentence_hash_slp1
        )
    assert "1 hash" in str(error.value)
    assert "--allow-shrink" in str(error.value)
    # The refusal must leave the committed list exactly as it was.
    assert committed.read_text(encoding="utf-8") == before

    build_exclusion_module.assert_superset_of_committed(shrunk, committed, allow_shrink=True)
    count = build_exclusion_module.write_list(
        {"a": ["tadapi"]}, committed, hash_fn=sentence_hash_slp1, allow_shrink=True
    )
    assert count == 1
    assert build_exclusion_module.assert_superset_of_committed(superset, tmp_path / "nope") is None
