"""Tests for the Experiment 04 runner (`experiments/04_morph_constrained/run.py`).

Everything here is offline and synthetic. The DCS held-out set is three hand-written
`GoldSentence` records whose segment, stem and marked forms are consistent with each other
by construction, so the gold-boundary derivations can be asserted against arithmetic rather
than against the real corpus. The parallel corpora are four Devanagari sentences with a
hand-written split cache, exactly as `tests/test_exp03.py` builds them. The tokenizers are
fakes: one emits a token per non-space character (so a token count is a length and every
character offset is a token boundary), one emits a single token per text (so it has no
boundaries at all).

What is pinned is the *derivation* — which words MorphScore is asked about, which gold
offsets go with them, how a stem offset inside a segment is re-based, which pairs are legal,
what happens when spans do not cover the text — plus the plotting and the end-to-end wiring.
The experiment's numbers come from actually running it.

`run.py` is not importable as a package module (`experiments/` holds scripts, not a
package), so it is loaded by path, as `tests/test_exp01.py`, `test_exp02.py` and
`test_exp03.py` do.
"""

import importlib.util
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from sanskrit_tok import experiment as experiment_module
from sanskrit_tok.data.exclusion import (
    build_exclusion_list,
    sentence_hash_en,
    sentence_hash_slp1,
)
from sanskrit_tok.data.parallel import ParallelCorpus
from sanskrit_tok.experiment import ENGLISH_LANGUAGE, SANSKRIT_LANGUAGE
from sanskrit_tok.metrics.summary import DISTRIBUTION_KEYS
from sanskrit_tok.tokenizers.registry import TokenizerUnavailable
from sanskrit_tok.tokenizers.train_bpe import train_bpe

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "experiments" / "04_morph_constrained" / "run.py"
CONFIG_YAML = REPO_ROOT / "experiments" / "04_morph_constrained" / "config.yaml"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp04_run", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run_module()

MARKER = "\x1f"

#: Three synthetic DCS held-out records. Every field is consistent with the others: the
#: markers in `t5_marked` are the segment offsets plus the stem offsets projected into the
#: sandhied surface, the markers in `t5seg_marked` are the segment offsets alone, and the
#: markers in `t6_marked` are the stem offsets as absolute indices into
#: `oracle_split_slp1`, which is exactly the invariant the real ingestion maintains.
#:
#: Record 1 (verified): `tadapi` = `tat|api` with a stem boundary inside `tat` at 2, and
#: `satyam` with a stem boundary at 2. Record 2 (unverified): `rAmaH` = `rAma|H`, and
#: `gacCati`, whose alignment failed (`None`). Record 3 (verified): `gacCati` with no
#: boundary at all, and `vanam` = `van|am`.
GOLD_RECORDS: list[dict[str, Any]] = [
    {
        "sent_id": "1",
        "text_id": 10,
        "text_slp1": "tadapi satyam",
        "oracle_split_slp1": "tat api satyam",
        "segment_offsets": [[3], []],
        "stem_offsets": [[2], [10]],
        "t5_marked": f"ta{MARKER}d{MARKER}api sa{MARKER}tyam",
        "t5seg_marked": f"tad{MARKER}api satyam",
        "t6_marked": f"ta{MARKER}t api sa{MARKER}tyam",
        "n_words": 2,
        "n_words_aligned": 2,
        "human_verified": True,
    },
    {
        "sent_id": "2",
        "text_id": 10,
        "text_slp1": "rAmaH gacCati",
        "oracle_split_slp1": "rAmaH gacCati",
        "segment_offsets": [[], None],
        "stem_offsets": [[4], []],
        "t5_marked": f"rAma{MARKER}H gacCati",
        "t5seg_marked": "rAmaH gacCati",
        "t6_marked": f"rAma{MARKER}H gacCati",
        "n_words": 2,
        "n_words_aligned": 1,
        "human_verified": False,
    },
    {
        "sent_id": "3",
        "text_id": 11,
        "text_slp1": "gacCati vanam",
        "oracle_split_slp1": "gacCati vanam",
        "segment_offsets": [[], []],
        "stem_offsets": [[], [11]],
        "t5_marked": f"gacCati van{MARKER}am",
        "t5seg_marked": "gacCati vanam",
        "t6_marked": f"gacCati van{MARKER}am",
        "n_words": 2,
        "n_words_aligned": 2,
        "human_verified": True,
    },
]

#: Four Devanagari sentences and their English side, standing in for one parallel corpus.
SANSKRIT = ["रामः गच्छति", "सीता वदति", "तदपि सत्यम्", "विश्वासजनकम् वचः"]
ENGLISH = ["Rama goes", "Sita speaks", "that too is true", "a trust-inspiring word"]
RAW_SLP1 = ["rAmaH gacCati", "sItA vadati", "tadapi satyam", "viSvAsajanakam vacaH"]
SPLIT_OUTPUT = ["rAmaH gacCati", "sItA vadati", "tad api satyam", "viSvAsa janakam vacaH"]
SPLIT_MODEL_OUTPUT = ["rAmaH gacCati", "sItA vadati", "tad api satyam", "viSvAsa janakam"]

RAW_ARMS = ["T1_bpe_raw_32k_dcs", "T5_morphbpe_raw_32k_dcs"]
SPLIT_ARMS = ["T4_bpe_split_32k_oracle_dcs", "T6_morphbpe_split_32k_dcs"]
PAIRS = [
    ["T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_32k_dcs"],
    ["T4_bpe_split_32k_oracle_dcs", "T1_bpe_raw_32k_dcs"],
    ["T6_morphbpe_split_32k_dcs", "T4_bpe_split_32k_oracle_dcs"],
    ["T6_morphbpe_split_32k_dcs", "T1_bpe_raw_32k_dcs"],
]


# --- fakes ----------------------------------------------------------------------------


def _char_arm(name: str) -> Any:
    """A `LoadedTokenizer` with one token per character and one span per non-space one.

    Every character offset inside a word is therefore a token boundary, which makes the
    pooled MorphScore hand-computable: recall is 1 and precision is
    `n_gold / (len(word) - 1)`.
    """
    return run.LoadedTokenizer(
        name=name,
        source_id=f"{name}.json",
        vocab_size=32000,
        _encode=lambda text: [0] * len(text),
        family=name.split("_", 1)[0],
        variant=run.strip_variant_suffix(name)[1],
        attempted=(f"{name}.json",),
        _spans=lambda text: [
            (index, index + 1) for index, char in enumerate(text) if not char.isspace()
        ],
    )


def _whole_word_arm(name: str) -> Any:
    """A `LoadedTokenizer` emitting one token covering the whole text: no boundaries."""
    return run.LoadedTokenizer(
        name=name,
        source_id=f"{name}.json",
        vocab_size=32000,
        _encode=lambda text: [0],
        family=name.split("_", 1)[0],
        variant=run.strip_variant_suffix(name)[1],
        attempted=(f"{name}.json",),
        _spans=lambda text: [(0, len(text))] if text.strip() else [],
    )


def _spanless_arm(name: str) -> Any:
    """A `LoadedTokenizer` with no span provider — countable, not MorphScore-able."""
    return run.LoadedTokenizer(
        name=name,
        source_id=f"{name}.json",
        vocab_size=32000,
        _encode=lambda text: [0] * len(text),
        family=name.split("_", 1)[0],
        variant=run.strip_variant_suffix(name)[1],
        attempted=(f"{name}.json",),
    )


def _broken_spans_arm(name: str) -> Any:
    """A `LoadedTokenizer` whose spans cover only the first character: contract violated."""
    return run.LoadedTokenizer(
        name=name,
        source_id=f"{name}.json",
        vocab_size=32000,
        _encode=lambda text: [0] * len(text),
        family=name.split("_", 1)[0],
        variant=run.strip_variant_suffix(name)[1],
        attempted=(f"{name}.json",),
        _spans=lambda text: [(0, 1)] if text else [],
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _split_records() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "raw_deva": SANSKRIT[index],
            "raw_slp1": RAW_SLP1[index],
            "output_model": SPLIT_MODEL_OUTPUT[index],
            "output": SPLIT_OUTPUT[index],
        }
        for index in range(len(SANSKRIT))
    ]


def _records() -> list[Any]:
    return [run.GoldRecord.from_dict(record) for record in GOLD_RECORDS]


# --- arm names ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("T5_morphbpe_raw_32k_dcs", ("T5_morphbpe_raw_32k", "dcs")),
        ("T4_bpe_split_64k_oracle_dcs", ("T4_bpe_split_64k", "oracle_dcs")),
        ("T1_bpe_raw_64k", ("T1_bpe_raw_64k", "")),
        ("T0_o200k", ("T0_o200k", "")),
    ],
)
def test_strip_variant_suffix_separates_the_training_corpus_label(
    name: str, expected: tuple[str, str]
) -> None:
    assert run.strip_variant_suffix(name) == expected


@pytest.mark.parametrize(
    ("name", "algorithm", "vocab"),
    [
        ("T1_bpe_raw_32k_dcs", "bpe", "32k"),
        ("T5_morphbpe_raw_64k_dcs", "morphbpe", "64k"),
        ("T4_unigram_split_32k_oracle_dcs", "unigram", "32k"),
        ("T6_morphbpe_split_64k_dcs", "morphbpe", "64k"),
    ],
)
def test_arm_algorithm_and_vocab_read_past_the_variant_suffix(
    name: str, algorithm: str, vocab: str
) -> None:
    assert run.arm_algorithm(name) == algorithm
    assert run.arm_vocab(name) == vocab


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("T1_bpe_raw_32k_dcs", run.RAW_ARM),
        ("T5_morphbpe_raw_64k_dcs", run.RAW_ARM),
        ("T4_bpe_split_32k_oracle_dcs", run.SPLIT_ARM),
        ("T6_morphbpe_split_64k_dcs", run.SPLIT_ARM),
        ("T0_o200k", run.RAW_ARM),  # off-the-shelf arms tokenize the sandhied surface
        ("T3_sarvam", run.RAW_ARM),
    ],
)
def test_arm_text_kind_says_which_text_an_arm_is_measured_on(name: str, kind: str) -> None:
    assert run.arm_text_kind(name) == kind


def test_arm_labels_separate_provisional_from_dcs_and_oracle() -> None:
    assert run.arm_labels("T6_morphbpe_split_64k_dcs") == {
        "variant": "dcs",
        "provisional": False,
        "oracle": False,
    }
    assert run.arm_labels("T4_bpe_split_64k_oracle_dcs")["oracle"] is True
    assert run.arm_labels("T1_bpe_raw_64k")["provisional"] is True
    assert run.arm_labels("T0_o200k")["provisional"] is False


# --- pair validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_32k_dcs"),  # constraint alone
        ("T4_bpe_split_64k_oracle_dcs", "T1_bpe_raw_64k_dcs"),  # gold splitting alone
        ("T6_morphbpe_split_32k_dcs", "T4_bpe_split_32k_oracle_dcs"),  # constraint on split
        ("T6_morphbpe_split_64k_dcs", "T1_bpe_raw_64k_dcs"),  # the proposed method
    ],
)
def test_check_paired_arms_accepts_the_four_configured_contrasts(
    first: str, second: str
) -> None:
    """`morphbpe` is BPE plus the constraint, so it pairs with `bpe` at the same vocab."""
    run.check_paired_arms(first, second)


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        ("T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_64k_dcs", "vocab"),
        ("T5_morphbpe_raw_32k_dcs", "T2_unigram_raw_32k_dcs", "algorithm"),
        ("T6_morphbpe_split_32k_dcs", "T1_bpe_raw_32k", "training corpus"),
    ],
)
def test_check_paired_arms_rejects_an_uncontrolled_pair(
    first: str, second: str, match: str
) -> None:
    """A pair differing in vocabulary, in algorithm family or in training corpus would put
    that difference and the constraint into the same delta, invisibly."""
    with pytest.raises(ValueError, match=match):
        run.check_paired_arms(first, second)


def test_pair_key_reads_in_delta_order() -> None:
    assert (
        run.pair_key("T6_morphbpe_split_32k_dcs", "T1_bpe_raw_32k_dcs")
        == "T6_morphbpe_split_32k_dcs/T1_bpe_raw_32k_dcs"
    )


def test_pair_label_names_the_two_families_and_the_shared_vocab() -> None:
    assert run.pair_label("T6_morphbpe_split_32k_dcs", "T1_bpe_raw_32k_dcs") == "T6−T1 32k"
    assert (
        run.pair_label("T6_morphbpe_split_64k_dcs", "T4_bpe_split_64k_oracle_dcs")
        == "T6−T4 64k"
    )


def test_select_pairs_keeps_a_pair_whose_both_sides_loaded() -> None:
    arms = {name: _char_arm(name) for name in ("T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_32k_dcs")}
    assert run.select_pairs([PAIRS[0]], arms) == [
        ("T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_32k_dcs")
    ]


def test_select_pairs_skips_a_pair_with_an_untrained_arm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arms = {"T1_bpe_raw_32k_dcs": _char_arm("T1_bpe_raw_32k_dcs")}
    with caplog.at_level("WARNING", logger="exp04"):
        assert run.select_pairs([PAIRS[0]], arms) == []
    assert "T5_morphbpe_raw_32k_dcs" in " ".join(r.getMessage() for r in caplog.records)


def test_select_pairs_validates_structure_even_for_unavailable_arms() -> None:
    with pytest.raises(ValueError, match="vocab"):
        run.select_pairs([["T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_64k_dcs"]], {})


def test_select_pairs_rejects_a_malformed_pair() -> None:
    with pytest.raises(ValueError, match="paired_deltas"):
        run.select_pairs([["T5_morphbpe_raw_32k_dcs"]], {})


# --- gold records and the human-verified filter ---------------------------------------


def test_read_gold_records_reads_every_field(tmp_path: Path) -> None:
    path = tmp_path / "heldout.jsonl"
    _write_jsonl(path, GOLD_RECORDS)
    records = run.read_gold_records(path)
    assert len(records) == 3
    assert records[0].text_slp1 == "tadapi satyam"
    assert records[0].segment_offsets == [[3], []]
    assert records[1].segment_offsets[1] is None
    assert records[2].human_verified is True


def test_read_gold_records_aborts_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest_dcs.py"):
        run.read_gold_records(tmp_path / "absent.jsonl")


def test_human_verified_keeps_only_the_sentences_with_no_reconstructed_token() -> None:
    subset = run.human_verified(_records())
    assert [record.sent_id for record in subset] == ["1", "3"]


def test_subsets_names_both_populations_and_reports_their_sizes() -> None:
    subsets = run.subsets(_records())
    assert list(subsets) == [run.SUBSET_ALL, run.SUBSET_HUMAN]
    assert len(subsets[run.SUBSET_ALL]) == 3
    assert len(subsets[run.SUBSET_HUMAN]) == 2


# --- gold boundary derivation ---------------------------------------------------------


def test_segment_gold_pairs_each_surface_word_with_its_aligned_offsets() -> None:
    words, gold = run.morphscore_inputs(_records(), run.RAW_ARM, run.GRANULARITY_SEGMENT)
    assert words == ["tadapi", "satyam", "rAmaH", "gacCati", "gacCati", "vanam"]
    assert gold == [[3], [], [], None, [], []]


def test_stem_gold_for_raw_arms_projects_into_the_surface_and_drops_segment_offsets() -> None:
    """The projected stems are exactly `t5_marked`'s markers minus the segment offsets; a
    word whose alignment failed carries `None`, not an empty list, so MorphScore skips it
    instead of scoring it as a miss."""
    words, gold = run.morphscore_inputs(_records(), run.RAW_ARM, run.GRANULARITY_STEM)
    assert words == ["tadapi", "satyam", "rAmaH", "gacCati", "gacCati", "vanam"]
    assert gold == [[2], [2], [4], None, [], [3]]


def test_stem_gold_for_split_arms_rebases_each_offset_inside_its_segment() -> None:
    """The split-side unit is the oracle segment, so an absolute offset of 10 in
    `tat api satyam` becomes offset 2 inside `satyam`."""
    words, gold = run.morphscore_inputs(_records(), run.SPLIT_ARM, run.GRANULARITY_STEM)
    assert words == [
        "tat",
        "api",
        "satyam",
        "rAmaH",
        "gacCati",
        "gacCati",
        "vanam",
    ]
    assert gold == [[2], [], [2], [4], [], [], [3]]


def test_split_arms_have_no_segment_granularity() -> None:
    """Segment boundaries are whitespace in the oracle split, so there is nothing to score."""
    assert run.granularities_for(run.SPLIT_ARM) == (run.GRANULARITY_STEM,)
    assert run.granularities_for(run.RAW_ARM) == (
        run.GRANULARITY_SEGMENT,
        run.GRANULARITY_STEM,
    )


def test_rebase_into_segments_is_exact_at_a_segment_edge() -> None:
    """A stem offset at the first character after a space belongs to the *next* segment."""
    units = run.rebase_into_units("ab cd ef", [3, 4, 7])
    assert units == [("ab", []), ("cd", [0, 1]), ("ef", [1])]


def test_rebase_into_segments_rejects_an_offset_past_the_end() -> None:
    with pytest.raises(ValueError, match="out of range"):
        run.rebase_into_units("ab cd", [99])


# --- MorphScore aggregation -----------------------------------------------------------


def test_compute_morphscore_pools_over_the_right_words_and_is_hand_computable() -> None:
    """The char-level arm makes every offset a boundary, so the pooled numbers are
    arithmetic: on the human-verified segment granularity only `tadapi` is scored (the
    other three verified words have no gold boundary), giving 1 match out of 5 token
    boundaries and 1 gold boundary."""
    arms = {"T1_bpe_raw_32k_dcs": _char_arm("T1_bpe_raw_32k_dcs")}
    result = run.compute_morphscore(_records(), arms, ["T1_bpe_raw_32k_dcs"], {})
    entry = result["T1_bpe_raw_32k_dcs"][run.GRANULARITY_SEGMENT][run.SUBSET_HUMAN]["exact"]
    assert entry["n_matched"] == 1
    assert entry["n_token_boundaries"] == 5
    assert entry["n_gold_boundaries"] == 1
    assert entry["precision"] == pytest.approx(0.2)
    assert entry["recall"] == pytest.approx(1.0)
    assert entry["value"] == pytest.approx(2 * 0.2 / 1.2)
    assert entry["n"] == 1


def test_compute_morphscore_reports_both_tolerances_and_both_subsets() -> None:
    arms = {"T1_bpe_raw_32k_dcs": _char_arm("T1_bpe_raw_32k_dcs")}
    result = run.compute_morphscore(_records(), arms, ["T1_bpe_raw_32k_dcs"], {})
    block = result["T1_bpe_raw_32k_dcs"]
    assert set(block) == {run.GRANULARITY_SEGMENT, run.GRANULARITY_STEM}
    assert set(block[run.GRANULARITY_SEGMENT]) == {run.SUBSET_ALL, run.SUBSET_HUMAN}
    assert set(block[run.GRANULARITY_SEGMENT][run.SUBSET_ALL]) == {"exact", "pm1"}
    assert block[run.GRANULARITY_SEGMENT][run.SUBSET_ALL]["pm1"]["tolerance"] == 1


def test_compute_morphscore_summarises_the_per_word_f1_distribution() -> None:
    """`summary.DISTRIBUTION_KEYS` must know `per_word_f1`, or every MorphScore row in
    `results.json` reports `distribution: null` and no spread at all."""
    assert "per_word_f1" in DISTRIBUTION_KEYS
    arms = {"T1_bpe_raw_32k_dcs": _char_arm("T1_bpe_raw_32k_dcs")}
    result = run.compute_morphscore(_records(), arms, ["T1_bpe_raw_32k_dcs"], {})
    entry = result["T1_bpe_raw_32k_dcs"][run.GRANULARITY_SEGMENT][run.SUBSET_ALL]["exact"]
    assert entry["distribution"] == "per_word_f1"
    assert entry["mean"] is not None


def test_compute_morphscore_uses_split_units_for_a_split_arm() -> None:
    arms = {"T6_morphbpe_split_32k_dcs": _char_arm("T6_morphbpe_split_32k_dcs")}
    result = run.compute_morphscore(_records(), arms, ["T6_morphbpe_split_32k_dcs"], {})
    block = result["T6_morphbpe_split_32k_dcs"]
    assert set(block) == {run.GRANULARITY_STEM}
    # four segments carry a stem boundary across all three records: tat, satyam, rAmaH, vanam
    assert block[run.GRANULARITY_STEM][run.SUBSET_ALL]["exact"]["n_gold_boundaries"] == 4


def test_compute_morphscore_skips_an_arm_with_no_spans(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arms = {"T7_byt5": _spanless_arm("T7_byt5")}
    with caplog.at_level("WARNING", logger="exp04"):
        result = run.compute_morphscore(_records(), arms, ["T7_byt5"], {})
    assert "T7_byt5" not in result
    assert "spans" in " ".join(record.getMessage() for record in caplog.records)


def test_compute_morphscore_aborts_an_arm_whose_spans_do_not_cover_the_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed coverage check means the boundary set is not a segmentation of the word, so
    the F1 would be a number with no meaning; the arm is dropped and the failure recorded."""
    arms = {"T1_bpe_raw_32k_dcs": _broken_spans_arm("T1_bpe_raw_32k_dcs")}
    coverage = run.check_spans_coverage(_records(), arms, ["T1_bpe_raw_32k_dcs"], n=3)
    assert coverage["T1_bpe_raw_32k_dcs"]["text_slp1"]["n_failed"] == 3
    with caplog.at_level("WARNING", logger="exp04"):
        result = run.compute_morphscore(_records(), arms, ["T1_bpe_raw_32k_dcs"], coverage)
    assert "T1_bpe_raw_32k_dcs" not in result
    assert "coverage" in " ".join(record.getMessage() for record in caplog.records)


def test_check_spans_coverage_passes_for_a_conforming_arm() -> None:
    arms = {"T1_bpe_raw_32k_dcs": _char_arm("T1_bpe_raw_32k_dcs")}
    coverage = run.check_spans_coverage(_records(), arms, ["T1_bpe_raw_32k_dcs"], n=3)
    entry = coverage["T1_bpe_raw_32k_dcs"]
    assert entry["text_slp1"] == {"n": 3, "n_passed": 3, "n_failed": 0}
    assert entry["oracle_split_slp1"] == {"n": 3, "n_passed": 3, "n_failed": 0}


# --- the out-of-sample violation audit ------------------------------------------------


@pytest.fixture
def toy_tokenizer(tmp_path: Path) -> Path:
    """A real (tiny) Metaspace BPE, so the audit runs against genuine token offsets."""
    corpus = tmp_path / "toy.txt"
    corpus.write_text("\n".join(["ab cd ef"] * 100) + "\n", encoding="utf-8")
    return train_bpe(corpus, 40, tmp_path / "toy_tok")


def test_violation_report_trims_whitespace_before_the_containment_test(
    toy_tokenizer: Path, tmp_path: Path
) -> None:
    """`Metaspace` attaches the space *before* a word to that word's first token, so the raw
    span of `cd` in `ab cd` is `(2, 5)` and strictly contains offset 3 — the first character
    of `cd`. Trimming the whitespace off the span makes it `(3, 5)`, which does not."""
    marked = tmp_path / "marked.txt"
    marked.write_text(f"ab {MARKER}cd\n", encoding="utf-8")
    report = run.violation_report(toy_tokenizer, marked, sample=10)
    assert report["n_boundaries"] == 1
    assert report["n_violations"] == 0


def test_violation_report_still_counts_a_boundary_inside_a_token(
    toy_tokenizer: Path, tmp_path: Path
) -> None:
    """The trim must not make the audit blind: a marker in the middle of `cd` is crossed."""
    marked = tmp_path / "marked.txt"
    marked.write_text(f"ab c{MARKER}d\n", encoding="utf-8")
    report = run.violation_report(toy_tokenizer, marked, sample=10)
    assert report["n_boundaries"] == 1
    assert report["n_violations"] == 1
    assert report["violations_per_boundary"] == pytest.approx(1.0)


def test_violation_report_is_per_boundary_not_per_token(
    toy_tokenizer: Path, tmp_path: Path
) -> None:
    """The headline mechanism number divides by gold boundaries, not by tokens: a corpus
    with twice the tokens and the same boundaries must not look twice as good."""
    marked = tmp_path / "marked.txt"
    marked.write_text(f"ab c{MARKER}d ef ef ef\n", encoding="utf-8")
    report = run.violation_report(toy_tokenizer, marked, sample=10)
    assert report["violations_per_boundary"] == pytest.approx(1.0)
    assert report["violation_rate"] < 1.0  # the function's own per-token rate


def test_compute_violations_records_a_skip_for_an_arm_with_no_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    marked = {run.RAW_ARM: tmp_path / "raw.txt"}
    marked[run.RAW_ARM].write_text(f"ta{MARKER}dapi\n", encoding="utf-8")
    arms = {"T1_bpe_raw_32k_dcs": _char_arm("T1_bpe_raw_32k_dcs")}
    with caplog.at_level("WARNING", logger="exp04"):
        result = run.compute_violations(arms, {run.RAW_ARM: ["T1_bpe_raw_32k_dcs"]}, marked, 10)
    assert "skipped" in result["T1_bpe_raw_32k_dcs"]


def test_write_marked_corpus_writes_one_line_per_record(tmp_path: Path) -> None:
    path = run.write_marked_corpus(_records(), "t6_marked", tmp_path / "m.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [record["t6_marked"] for record in GOLD_RECORDS]


# --- the figure -----------------------------------------------------------------------


def _synthetic_results() -> dict[str, Any]:
    def delta(value: float) -> dict[str, Any]:
        return {
            "delta": value,
            "value": value,
            "ci_low": value - 0.01,
            "ci_high": value + 0.01,
            "label": "T5−T1 32k",
        }

    def morph(value: float) -> dict[str, Any]:
        return {
            run.SUBSET_HUMAN: {
                "exact": {"value": value, "n": 10},
                "pm1": {"value": value + 0.05, "n": 10},
            },
            run.SUBSET_ALL: {
                "exact": {"value": value, "n": 30},
                "pm1": {"value": value + 0.05, "n": 30},
            },
        }

    keys = [run.pair_key(first, second) for first, second in PAIRS]
    return {
        "config": {
            "corpora": [{"name": "corpus_a"}, {"name": "corpus_b"}],
            "paired_deltas": [list(pair) for pair in PAIRS],
            "arms_morphscore": [*RAW_ARMS, *SPLIT_ARMS],
            "english_pivots": {"controlled": "E1_bpe_64k", "deployed": "T0_o200k"},
        },
        "unavailable_arms": {"T2_unigram_raw_64k_dcs": "T2_unigram_raw_64k_dcs: not found"},
        "tpp_delta": {
            "corpus_a": {key: delta(-0.02 * index) for index, key in enumerate(keys)},
            "corpus_b": {key: delta(0.01 * index) for index, key in enumerate(keys)},
        },
        "morphscore": {
            "T1_bpe_raw_32k_dcs": {run.GRANULARITY_SEGMENT: morph(0.30)},
            "T5_morphbpe_raw_32k_dcs": {run.GRANULARITY_SEGMENT: morph(0.45)},
            "T4_bpe_split_32k_oracle_dcs": {run.GRANULARITY_STEM: morph(0.35)},
            "T6_morphbpe_split_32k_dcs": {run.GRANULARITY_STEM: morph(0.60)},
        },
    }


def test_make_figure_writes_a_pdf_and_a_png(tmp_path: Path) -> None:
    paths = run.make_figure(_synthetic_results(), tmp_path)
    assert [path.name for path in paths] == ["constraint_effects.pdf", "constraint_effects.png"]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_build_figure_has_one_delta_row_per_corpus_and_one_morphscore_panel() -> None:
    import matplotlib.pyplot as plt

    figure = run._build_figure(_synthetic_results())
    try:
        assert len(figure.axes) == 3  # two corpus rows plus the MorphScore panel
        labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert labels[0] == "T5−T1 32k"
        assert len(labels) == len(PAIRS)
    finally:
        plt.close(figure)


def test_build_figure_caption_states_every_label_the_numbers_depend_on() -> None:
    import matplotlib.pyplot as plt

    figure = run._build_figure(_synthetic_results())
    try:
        texts = " ".join(text.get_text() for text in figure.texts).lower()
        assert "oracle" in texts
        assert "heuristic" in texts
        assert "provisional" in texts
        assert "cross-corpus" in texts
        assert "t2_unigram_raw_64k_dcs" in texts  # the omitted arm is named
    finally:
        plt.close(figure)


def test_build_figure_rejects_a_results_dict_with_no_corpora() -> None:
    results = _synthetic_results()
    results["config"]["corpora"] = []
    with pytest.raises(ValueError, match="corpora"):
        run._build_figure(results)


# --- the shipped config ---------------------------------------------------------------


def test_shipped_config_pairs_are_all_controlled() -> None:
    """Every pair in the config the real run uses must be matched on vocabulary, algorithm
    family and training corpus — checked here so a config typo fails in CI, not after an
    hour of bootstrapping."""
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    assert len(config["paired_deltas"]) == 10
    for first, second in config["paired_deltas"]:
        run.check_paired_arms(first, second)


def test_shipped_config_morphscore_deltas_are_controlled_and_same_population() -> None:
    """Every MorphScore contrast must be matched the same way a TPP pair is *and* score the
    same population of units: a raw arm at segment granularity against a raw arm, a split
    arm at stem granularity against a split arm."""
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    contrasts = config["morphscore_deltas"]
    assert len(contrasts) == 6
    for contrast in contrasts:
        run.check_paired_arms(contrast["a"], contrast["b"])
        kinds = {run.arm_text_kind(contrast["a"]), run.arm_text_kind(contrast["b"])}
        assert len(kinds) == 1
        assert contrast["granularity"] in run.granularities_for(kinds.pop())
    # `T4_oracle − T1_dcs` compares two populations of units and must never be listed.
    assert not [
        contrast
        for contrast in contrasts
        if run.arm_text_kind(contrast["a"]) != run.arm_text_kind(contrast["b"])
    ]


def test_shipped_config_indomain_arms_are_all_dcs_trained() -> None:
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    assert len(config["arms_indomain"]) == 14
    for name in config["arms_indomain"]:
        assert run.strip_variant_suffix(name)[1] in {"dcs", "oracle_dcs"}


def test_shipped_config_lists_prose_before_verse() -> None:
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    names = [entry["name"] for entry in config["corpora"]]
    assert names.index("samayik_test") < names.index("itihasa_test")


# --- end to end, offline --------------------------------------------------------------


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A complete, synthetic Experiment 04 environment under `tmp_path`."""
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(heldout, GOLD_RECORDS)

    corpus_names = ["corpus_a", "corpus_b"]
    split_dir = tmp_path / "split"
    for name in corpus_names:
        _write_jsonl(split_dir / f"{name}.jsonl", _split_records())

    exclusion = tmp_path / "exclusion.txt"
    exclusion_en = tmp_path / "exclusion_en.txt"
    build_exclusion_list({"corpus_a": SANSKRIT}, exclusion)
    build_exclusion_list({"corpus_a": ENGLISH}, exclusion_en, hash_fn=sentence_hash_en)

    config: dict[str, Any] = {
        "experiment": "04_morph_constrained_test",
        "dcs_heldout": str(heldout),
        "corpora": [
            {"name": name, "loader": "samayik", "split": "test"} for name in corpus_names
        ],
        "split_dir": str(split_dir),
        "arms_morphscore": [*RAW_ARMS, *SPLIT_ARMS],
        "arms_tpp": [*RAW_ARMS, *SPLIT_ARMS],
        "paired_deltas": [list(pair) for pair in PAIRS],
        "morphscore_deltas": [
            {
                "a": "T5_morphbpe_raw_32k_dcs",
                "b": "T1_bpe_raw_32k_dcs",
                "granularity": run.GRANULARITY_SEGMENT,
            },
            {
                "a": "T6_morphbpe_split_32k_dcs",
                "b": "T4_bpe_split_32k_oracle_dcs",
                "granularity": run.GRANULARITY_STEM,
            },
        ],
        "morphscore_bootstrap": 25,
        "morphscore_bootstrap_seed": 0,
        "arms_indomain": [*RAW_ARMS, *SPLIT_ARMS],
        "violation_arms": {
            run.RAW_ARM: RAW_ARMS,
            "rawseg": RAW_ARMS,
            run.SPLIT_ARM: SPLIT_ARMS,
        },
        "violation_sample": 10,
        "spans_coverage_sentences": 3,
        "english_pivots": {"controlled": "E1_bpe_64k", "deployed": "T0_o200k"},
        "n_bootstrap": 25,
        "seed": 0,
        "ci": 0.95,
        "exclusion_path": str(exclusion),
        "exclusion_path_en": str(exclusion_en),
        "output_dir": str(tmp_path / "out"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def fake_corpus(entry: Mapping[str, Any], root: Path) -> ParallelCorpus:
        return ParallelCorpus(
            name=str(entry["name"]),
            split=str(entry["split"]),
            languages=(SANSKRIT_LANGUAGE, ENGLISH_LANGUAGE),
            sentences={
                SANSKRIT_LANGUAGE: list(SANSKRIT),
                ENGLISH_LANGUAGE: list(ENGLISH),
            },
        )

    monkeypatch.setattr(run, "load_corpus_entry", fake_corpus)
    monkeypatch.setattr(experiment_module, "load_tokenizer", lambda name: _char_arm(name))
    yield {"config": config, "config_path": config_path, "out_dir": tmp_path / "out"}


def test_run_end_to_end_writes_strict_json_a_config_and_both_figures(
    experiment: dict[str, Any],
) -> None:
    def reject_non_finite(token: str) -> float:
        raise AssertionError(f"results.json is not strict JSON: {token}")

    assert run.main(["--config", str(experiment["config_path"])]) == 0

    out_dir: Path = experiment["out_dir"]
    assert (out_dir / "config.yaml").exists()
    for name in ("constraint_effects.pdf", "constraint_effects.png"):
        assert (out_dir / name).stat().st_size > 0

    results = json.loads(
        (out_dir / "results.json").read_text(encoding="utf-8"),
        parse_constant=reject_non_finite,
    )
    for key in (
        "experiment",
        "git_commit",
        "git_dirty",
        "timestamp",
        "config",
        "tokenizer_sources",
        "unavailable_arms",
        "corpora",
        "dcs_heldout",
        "exclusion_check",
        "exclusion_check_en",
        "morphscore",
        "morphscore_delta",
        "compression_indomain",
        "spans_coverage",
        "violations",
        "tpp",
        "tpp_delta",
        "fertility_primary",
        "fertility_secondary",
        "compression",
        "text_invariants",
    ):
        assert key in results, key

    assert results["dcs_heldout"]["n"] == 3
    assert results["dcs_heldout"]["n_human_verified"] == 2
    assert results["dcs_heldout"]["sample"] is None
    # eight configured entries: four pairs on each of two corpora
    assert len(results["tpp_delta"]["corpus_a"]) == len(PAIRS)
    delta = results["tpp_delta"]["corpus_a"][run.pair_key(*PAIRS[0])]
    assert delta["delta"] == pytest.approx(delta["value_a"] - delta["value_b"])
    # the constraint-only pair tokenizes the same text on both sides, so no deletion cost
    assert delta["deletion_cost"] is None
    # the split-vs-raw pair does not, so it has one
    split_delta = results["tpp_delta"]["corpus_a"][run.pair_key(*PAIRS[1])]
    assert split_delta["deletion_cost"] is not None
    assert set(results["tpp"]["corpus_a"]["T6_morphbpe_split_32k_dcs"]) == {
        "E1_bpe_64k",
        "T0_o200k",
    }
    assert results["exclusion_check"]["corpus_a"] == {"n": 4, "n_missing": 0}
    assert results["exclusion_check_en"]["corpus_a"] == {"n": 4, "n_missing": 0}
    assert results["tokenizer_sources"]["T6_morphbpe_split_32k_dcs"]["variant"] == "dcs"
    assert results["text_invariants"]["dcs_heldout_oracle_vs_raw"]["letter_chars_raw"] > 0

    # the MorphScore paired deltas: both subsets, both tolerances, delta = F1(a) - F1(b)
    ms_delta = results["morphscore_delta"][
        run.pair_key("T5_morphbpe_raw_32k_dcs", "T1_bpe_raw_32k_dcs")
    ]
    assert set(ms_delta) == {run.SUBSET_ALL, run.SUBSET_HUMAN}
    entry = ms_delta[run.SUBSET_ALL]["exact"]
    assert entry["delta"] == pytest.approx(entry["f1_a"] - entry["f1_b"])
    assert entry["ci_low"] <= entry["ci_high"]
    assert entry["n_sentences"] == 3
    assert entry["granularity"] == run.GRANULARITY_SEGMENT
    assert ms_delta[run.SUBSET_HUMAN]["exact"]["n_sentences"] == 2

    # in-domain compression on the held-out split, on the text form each arm tokenizes
    indomain = results["compression_indomain"]
    assert set(indomain) == {*RAW_ARMS, *SPLIT_ARMS}
    assert indomain["T1_bpe_raw_32k_dcs"]["variant"] == run.RAW
    assert indomain["T6_morphbpe_split_32k_dcs"]["variant"] == run.SPLIT
    assert indomain["T1_bpe_raw_32k_dcs"]["n_tokens"] > 0

    # the violation audit keys: the arm's own boundary set keeps the bare name, a second
    # denominator is suffixed with the boundary set it was measured against
    assert "T1_bpe_raw_32k_dcs" in results["violations"]
    assert "T1_bpe_raw_32k_dcs@rawseg" in results["violations"]
    assert results["violations"]["T1_bpe_raw_32k_dcs@rawseg"]["arm"] == "T1_bpe_raw_32k_dcs"


def test_run_checks_the_heldout_sentences_are_in_the_exclusion_list(
    experiment: dict[str, Any], tmp_path: Path
) -> None:
    """The DCS held-out sentences are what every `_dcs` arm was kept away from; if they are
    not in the committed list the arms may have trained on them."""
    build_exclusion_list(
        {"dcs_heldout": [record["text_slp1"] for record in GOLD_RECORDS], "corpus_a": SANSKRIT},
        tmp_path / "exclusion.txt",
        hash_fn=sentence_hash_slp1,
    )
    results = run.run(experiment["config"], experiment["config_path"])
    assert results["dcs_heldout"]["exclusion_check"]["n_missing"] == 0


def test_run_records_an_untrained_arm_as_unavailable_rather_than_aborting(
    experiment: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def load(name: str) -> Any:
        if name == "T6_morphbpe_split_32k_dcs":
            raise TokenizerUnavailable(f"{name}: trained tokenizer file not found")
        return _char_arm(name)

    monkeypatch.setattr(experiment_module, "load_tokenizer", load)
    results = run.run(experiment["config"], experiment["config_path"])
    assert "T6_morphbpe_split_32k_dcs" in results["unavailable_arms"]
    assert "T6_morphbpe_split_32k_dcs" not in results["morphscore"]
    assert run.pair_key(*PAIRS[3]) not in results["tpp_delta"]["corpus_a"]
    assert not math.isnan(
        results["tpp"]["corpus_a"]["T1_bpe_raw_32k_dcs"]["E1_bpe_64k"]["value"]
    )


def test_run_aborts_when_a_corpus_is_not_in_the_split_cache(
    experiment: dict[str, Any], tmp_path: Path
) -> None:
    (tmp_path / "split" / "corpus_b.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="corpus_b"):
        run.run(experiment["config"], experiment["config_path"])


def test_run_measures_a_whole_word_arm_without_crashing(
    experiment: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every word is one token, so every MorphScore is undefined (`nan`) — which must reach
    `results.json` as `null` rather than as a non-standard JSON token."""
    monkeypatch.setattr(experiment_module, "load_tokenizer", lambda name: _whole_word_arm(name))
    results = run.run(experiment["config"], experiment["config_path"])
    entry = results["morphscore"]["T1_bpe_raw_32k_dcs"][run.GRANULARITY_SEGMENT][
        run.SUBSET_ALL
    ]["exact"]
    assert math.isnan(entry["value"])
    assert entry["n_excluded_single_token"] > 0
    assert json.loads(
        (experiment["out_dir"] / "results.json").read_text(encoding="utf-8")
    )["morphscore"]["T1_bpe_raw_32k_dcs"][run.GRANULARITY_SEGMENT][run.SUBSET_ALL]["exact"][
        "value"
    ] is None


def test_violation_key_names_the_boundary_set_only_when_it_is_not_the_arms_own() -> None:
    assert run.violation_key(run.RAW_ARM, "T1_bpe_raw_32k_dcs") == "T1_bpe_raw_32k_dcs"
    assert (
        run.violation_key("rawseg", "T1_bpe_raw_32k_dcs") == "T1_bpe_raw_32k_dcs@rawseg"
    )
    assert (
        run.violation_key(run.SPLIT_ARM, "T6_morphbpe_split_64k_dcs")
        == "T6_morphbpe_split_64k_dcs"
    )


def test_violation_fields_name_a_gold_sentence_field_for_every_boundary_set() -> None:
    assert run.VIOLATION_FIELDS == {
        run.RAW_ARM: "t5_marked",
        "rawseg": "t5seg_marked",
        run.SPLIT_ARM: "t6_marked",
    }


def test_shipped_config_audits_every_raw_arm_against_both_raw_boundary_sets() -> None:
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    groups = config["violation_arms"]
    assert set(groups) == set(run.VIOLATION_FIELDS)
    assert groups["raw"] == groups["rawseg"]
    assert "T5_morphbpe_rawseg_64k_dcs" in groups["raw"]
