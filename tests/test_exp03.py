"""Tests for the Experiment 03 runner (`experiments/03_sandhi_split/run.py`).

Everything here is offline and synthetic. The corpora are four hand-written Devanagari
sentences, the split cache is a jsonl this module writes into `tmp_path`, the tokenizers
are fakes whose token counts are lengths, and the splitter is never imported at all — the
runner reads its output from disk by design, so nothing here needs a 2.3 GB model or the
network. What is pinned is the aggregation, the abort conditions, the plotting and the
end-to-end wiring, not the experiment's numbers (those come from actually running it).

`run.py` is not importable as a package module (`experiments/` holds scripts, not a
package), so it is loaded by path, exactly as `tests/test_exp01.py` and `test_exp02.py` do.
"""

import importlib.util
import json
import math
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from sanskrit_tok.data.exclusion import build_exclusion_list, sentence_hash_en
from sanskrit_tok.data.parallel import ParallelCorpus
from sanskrit_tok.experiment import ENGLISH_LANGUAGE, SANSKRIT_LANGUAGE

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "experiments" / "03_sandhi_split" / "run.py"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp03_run", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run_module()

#: Four Devanagari sentences and their English side, standing in for one corpus split.
SANSKRIT = [
    "रामः गच्छति",
    "सीता वदति",
    "तदपि सत्यम्",
    "विश्वासजनकम् वचः",
]
ENGLISH = [
    "Rama goes",
    "Sita speaks",
    "that too is true",
    "a trust-inspiring word",
]
#: What a splitter would have produced for those four, in SLP1: `output` is reconciled
#: (the primary variant) and `output_model` the unreconciled model output, which here
#: drops a word from the last sentence — the lossiness reconciliation exists to undo.
SPLIT_OUTPUT = ["rAmaH gacCati", "sItA vadati", "tad api satyam", "viSvAsa janakam vacaH"]
SPLIT_MODEL_OUTPUT = ["rAmaH gacCati", "sItA vadati", "tad api satyam", "viSvAsa janakam"]


# --- fakes ---------------------------------------------------------------------------


def _char_arm(name: str, family: str | None = None) -> Any:
    """A `LoadedTokenizer` emitting one id per character, so a token count is a length."""
    return run.LoadedTokenizer(
        name=name,
        source_id=f"{name}.json",
        vocab_size=32000,
        _encode=lambda text: [0] * len(text),
        family=family if family is not None else name.split("_", 1)[0],
        attempted=(f"{name}.json",),
    )


def _corpus(name: str = "corpus_a", split: str = "test") -> Any:
    """One four-sentence `CorpusData` with all three Sanskrit variants."""
    return run.CorpusData(
        name=name,
        split=split,
        n_total=len(SANSKRIT),
        n_used=len(SANSKRIT),
        raw_deva=list(SANSKRIT),
        texts={
            run.RAW: ["rAmaH gacCati", "sItA vadati", "tadapi satyam", "viSvAsajanakam vacaH"],
            run.SPLIT: list(SPLIT_OUTPUT),
            run.SPLIT_MODEL: list(SPLIT_MODEL_OUTPUT),
        },
        english=list(ENGLISH),
    )


def _split_records() -> list[dict[str, Any]]:
    from sanskrit_tok.encoding import to_slp1

    return [
        {
            "index": index,
            "raw_deva": SANSKRIT[index],
            "raw_slp1": to_slp1(SANSKRIT[index], "devanagari"),
            "output_model": SPLIT_MODEL_OUTPUT[index],
            "output": SPLIT_OUTPUT[index],
        }
        for index in range(len(SANSKRIT))
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _manifest(corpus_names: list[str]) -> dict[str, Any]:
    return {
        "model_id": "fake/splitter",
        "revision": "deadbeef",
        "splitter_source_id": "fake/splitter@deadbeef",
        "device": "cpu",
        "batch_size": 16,
        "reconcile_threshold": 0.6,
        "subset": None,
        "n_cache_hits": 0,
        "n_model": 4 * len(corpus_names),
        "n_chunked": 1,
        "seconds": 1.5,
        "git_commit": "0" * 40,
        "git_dirty": False,
        "timestamp": "2026-09-04T00:00:00+00:00",
        "corpora": {
            name: {
                "path": f"{name}.jsonl",
                "n_sentences": len(SANSKRIT),
                "n_out": len(SANSKRIT),
                "n_units_raw": 8,
                "n_units_out": 10,
                "n_units_kept_verbatim": 1,
                "n_units_replaced_inexact": 2,
                "n_units_changed": 2,
                "fraction_units_changed": 0.5,
                "chars_raw": 100,
                "chars_model": 87,
                "chars_out": 100,
                "char_retention_model": 0.87,
                "char_retention_reconciled": 1.0,
                "seconds": 0.5,
            }
            for name in corpus_names
        },
    }


# --- arm names, matched pairs and their labels ----------------------------------------


def test_arm_algorithm_and_vocab_reads_both_families() -> None:
    assert run.arm_algorithm_and_vocab("T1_bpe_raw_32k") == ("bpe", "32k")
    assert run.arm_algorithm_and_vocab("T4_unigram_split_64k") == ("unigram", "64k")


def test_arm_algorithm_and_vocab_rejects_a_name_that_is_not_an_arm() -> None:
    with pytest.raises(ValueError, match="arm name"):
        run.arm_algorithm_and_vocab("T7_byt5")


def test_matched_pair_label_names_the_algorithm_and_the_vocab_only() -> None:
    """The family prefixes are what the two markers mean, so the label carries neither."""
    assert run.matched_pair_label("T1_bpe_raw_32k", "T4_bpe_split_32k") == "bpe 32k"
    assert run.matched_pair_label("T2_unigram_raw_64k", "T4_unigram_split_64k") == "unigram 64k"


def test_matched_pair_key_reads_in_delta_order() -> None:
    """The delta is split minus raw, so the key names the split arm first."""
    assert (
        run.matched_pair_key("T1_bpe_raw_32k", "T4_bpe_split_32k")
        == "T4_bpe_split_32k/T1_bpe_raw_32k"
    )


def test_check_matched_pair_accepts_a_matched_pair() -> None:
    run.check_matched_pair("T2_unigram_raw_32k", "T4_unigram_split_32k")


@pytest.mark.parametrize(
    ("raw_arm", "split_arm"),
    [
        ("T1_bpe_raw_32k", "T4_bpe_split_64k"),  # vocabulary differs
        ("T1_bpe_raw_32k", "T4_unigram_split_32k"),  # algorithm differs
    ],
)
def test_check_matched_pair_rejects_an_unmatched_pair(raw_arm: str, split_arm: str) -> None:
    """A pair differing in vocabulary or algorithm would put the splitting effect and a
    vocabulary change into the same delta, invisibly."""
    with pytest.raises(ValueError, match="not matched"):
        run.check_matched_pair(raw_arm, split_arm)


def _control() -> dict[str, str]:
    return {"T1_bpe_raw_32k": "E1_bpe_32k", "T4_bpe_split_32k": "E1_bpe_32k"}


def test_select_matched_pairs_keeps_a_pair_whose_both_sides_loaded() -> None:
    arms = {name: _char_arm(name) for name in ("T1_bpe_raw_32k", "T4_bpe_split_32k")}
    assert run.select_matched_pairs(
        [["T1_bpe_raw_32k", "T4_bpe_split_32k"]], arms, _control()
    ) == [("T1_bpe_raw_32k", "T4_bpe_split_32k")]


def test_select_matched_pairs_skips_a_pair_with_an_untrained_arm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arms = {"T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k")}
    with caplog.at_level("WARNING", logger="exp03"):
        pairs = run.select_matched_pairs(
            [["T1_bpe_raw_32k", "T4_bpe_split_32k"]], arms, _control()
        )
    assert pairs == []
    assert "T4_bpe_split_32k" in " ".join(record.getMessage() for record in caplog.records)


def test_select_matched_pairs_validates_structure_even_for_unavailable_arms() -> None:
    """A config typo must fail on a run where the arm happens to be missing, too."""
    with pytest.raises(ValueError, match="not matched"):
        run.select_matched_pairs([["T1_bpe_raw_32k", "T4_bpe_split_64k"]], {}, _control())


def test_select_matched_pairs_rejects_a_malformed_pair() -> None:
    with pytest.raises(ValueError, match="matched_pairs"):
        run.select_matched_pairs([["T1_bpe_raw_32k"]], {}, _control())


def test_select_matched_pairs_rejects_a_pair_with_two_different_controls() -> None:
    """The delta divides both sides by the control arm; two denominators, no delta."""
    control = {"T1_bpe_raw_32k": "E1_bpe_32k", "T4_bpe_split_32k": "E1_unigram_32k"}
    with pytest.raises(ValueError, match="same"):
        run.select_matched_pairs([["T1_bpe_raw_32k", "T4_bpe_split_32k"]], {}, control)


def test_pivots_for_puts_the_control_first_and_labels_both_roles() -> None:
    assert run.pivots_for("T4_bpe_split_32k", _control(), "T0_o200k") == [
        ("E1_bpe_32k", run.ROLE_CONTROLLED),
        ("T0_o200k", run.ROLE_DEPLOYED),
    ]


def test_pivots_for_rejects_an_arm_with_no_matched_control() -> None:
    with pytest.raises(ValueError, match="english_control"):
        run.pivots_for("T4_unigram_split_64k", _control(), "T0_o200k")


def test_arm_specs_give_raw_arms_one_variant_and_split_arms_two() -> None:
    specs = run.arm_specs(["T1_bpe_raw_32k"], ["T4_bpe_split_32k"])
    assert [spec.name for spec in specs] == ["T1_bpe_raw_32k", "T4_bpe_split_32k"]
    assert specs[0].variants == (run.RAW,) and specs[0].primary == run.RAW
    assert specs[1].variants == (run.SPLIT, run.SPLIT_MODEL) and specs[1].primary == run.SPLIT


# --- the split cache: alignment and the abort conditions ------------------------------


def test_split_texts_for_returns_both_variants_in_corpus_order() -> None:
    texts = run.split_texts_for("corpus_a", _split_records(), SANSKRIT)
    assert texts[run.SPLIT] == SPLIT_OUTPUT
    assert texts[run.SPLIT_MODEL] == SPLIT_MODEL_OUTPUT


def test_split_texts_for_joins_on_index_not_file_order() -> None:
    """The jsonl is written in corpus order, but the join is on `index` regardless."""
    records = list(reversed(_split_records()))
    assert run.split_texts_for("corpus_a", records, SANSKRIT)[run.SPLIT] == SPLIT_OUTPUT


def test_split_texts_for_aborts_naming_the_corpus_and_the_missing_count() -> None:
    """The headline abort: the split job has not finished for this corpus. Falling back to
    raw text would silently measure a T4 arm on unsplit sentences."""
    records = [record for record in _split_records() if record["index"] != 2]
    with pytest.raises(ValueError) as excinfo:
        run.split_texts_for("samayik_test", records, SANSKRIT)
    message = str(excinfo.value)
    assert "samayik_test" in message
    assert "1 of 4" in message
    assert "index 2" in message
    assert "split_corpora.py" in message


def test_split_texts_for_aborts_on_a_record_count_mismatch() -> None:
    records = _split_records() + [dict(_split_records()[0])]
    with pytest.raises(ValueError, match="different version"):
        run.split_texts_for("samayik_test", records, SANSKRIT)


def test_split_texts_for_aborts_when_a_cached_sentence_is_not_the_corpus_sentence() -> None:
    """A stale cache aligns index-for-index with the wrong text, which no metric notices."""
    records = _split_records()
    records[1] = {**records[1], "raw_deva": "अन्यत् वाक्यम्"}
    with pytest.raises(ValueError, match="stale"):
        run.split_texts_for("samayik_test", records, SANSKRIT)


def test_read_split_records_names_the_script_that_writes_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="split_corpora.py"):
        run.read_split_records("samayik_test", tmp_path / "samayik_test.jsonl")


def test_read_split_records_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "corpus_a.jsonl"
    _write_jsonl(path, _split_records())
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(run.read_split_records("corpus_a", path)) == len(SANSKRIT)


def test_load_manifest_aborts_when_it_is_absent(tmp_path: Path) -> None:
    """Without the manifest, results.json could not say which model produced the text every
    T4 number is measured on."""
    with pytest.raises(FileNotFoundError, match="split_corpora.py"):
        run.load_manifest(tmp_path / "manifest.json")


# --- splitter statistics --------------------------------------------------------------


def test_splitter_stats_combines_manifest_fields_with_recomputed_ones() -> None:
    corpus = _corpus()
    stats = run.splitter_stats([corpus], _manifest([corpus.name]))[corpus.name]
    assert stats["n_sentences_evaluated"] == 4
    assert stats["mean_units_raw"] == pytest.approx(2.0)  # 2, 2, 2, 2
    assert stats["mean_units_split"] == pytest.approx(2.5)  # 2, 2, 3, 3
    assert stats["mean_units_split_model"] == pytest.approx(2.25)  # 2, 2, 3, 2
    assert stats["n_sentences_units_changed"] == 2
    assert stats["fraction_units_changed"] == pytest.approx(0.5)
    # the manifest's own accounting is carried through, not recomputed
    assert stats["manifest"]["char_retention_model"] == pytest.approx(0.87)
    assert stats["manifest"]["char_retention_reconciled"] == pytest.approx(1.0)
    assert stats["manifest"]["n_units_kept_verbatim"] == 1


def test_splitter_stats_aborts_when_the_manifest_does_not_cover_a_corpus() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        run.splitter_stats([_corpus("itihasa_test")], _manifest(["samayik_test"]))


def test_splitter_provenance_carries_the_source_id_and_the_run_wide_counters(
    tmp_path: Path,
) -> None:
    block = run.splitter_provenance(_manifest(["corpus_a"]), tmp_path / "manifest.json")
    assert block["source_id"] == "fake/splitter@deadbeef"
    assert block["revision"] == "deadbeef"
    assert block["n_chunked"] == 1
    assert block["manifest_path"].endswith("manifest.json")


# --- TPP, the delta, fertility and compression ----------------------------------------


def _arms() -> dict[str, Any]:
    return {
        name: _char_arm(name)
        for name in ("T1_bpe_raw_32k", "T4_bpe_split_32k", "E1_bpe_32k", "T0_o200k")
    }


def _specs() -> list[Any]:
    return run.arm_specs(["T1_bpe_raw_32k"], ["T4_bpe_split_32k"])


def test_compute_tpp_is_keyed_corpus_arm_variant_pivot() -> None:
    results = run.compute_tpp(
        [_corpus()],
        _arms(),
        _specs(),
        _control(),
        "T0_o200k",
        n_bootstrap=20,
        seed=0,
        ci=0.95,
    )
    assert list(results) == ["corpus_a"]
    assert set(results["corpus_a"]) == {"T1_bpe_raw_32k", "T4_bpe_split_32k"}
    assert set(results["corpus_a"]["T1_bpe_raw_32k"]) == {run.RAW}
    assert set(results["corpus_a"]["T4_bpe_split_32k"]) == {run.SPLIT, run.SPLIT_MODEL}
    summary = results["corpus_a"]["T4_bpe_split_32k"][run.SPLIT]["E1_bpe_32k"]
    assert summary["role"] == run.ROLE_CONTROLLED
    assert results["corpus_a"]["T4_bpe_split_32k"][run.SPLIT]["T0_o200k"]["role"] == (
        run.ROLE_DEPLOYED
    )
    for key in ("value", "n", "unit", "ci_low", "ci_high", "ci", "source_tokens", "pivot_tokens"):
        assert key in summary


def test_compute_tpp_measures_each_arm_on_its_own_text() -> None:
    """These fakes emit one id per character, so the recorded totals are exactly the
    character lengths of the variant each arm was given."""
    corpus = _corpus()
    results = run.compute_tpp(
        [corpus], _arms(), _specs(), _control(), "T0_o200k", n_bootstrap=0, seed=0, ci=0.95
    )
    raw = results["corpus_a"]["T1_bpe_raw_32k"][run.RAW]["E1_bpe_32k"]
    split = results["corpus_a"]["T4_bpe_split_32k"][run.SPLIT]["E1_bpe_32k"]
    model = results["corpus_a"]["T4_bpe_split_32k"][run.SPLIT_MODEL]["E1_bpe_32k"]
    assert raw["source_tokens"] == sum(len(text) for text in corpus.texts[run.RAW])
    assert split["source_tokens"] == sum(len(text) for text in corpus.texts[run.SPLIT])
    assert model["source_tokens"] == sum(len(text) for text in corpus.texts[run.SPLIT_MODEL])
    assert raw["pivot_tokens"] == sum(len(text) for text in corpus.english)


def test_compute_tpp_warns_once_for_an_unavailable_pivot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arms = {name: _char_arm(name) for name in ("T1_bpe_raw_32k", "T4_bpe_split_32k", "E1_bpe_32k")}
    with caplog.at_level("WARNING", logger="exp03"):
        results = run.compute_tpp(
            [_corpus(), _corpus("corpus_b")],
            arms,
            _specs(),
            _control(),
            "T0_o200k",
            n_bootstrap=0,
            seed=0,
            ci=0.95,
        )
    warnings = [record for record in caplog.records if "T0_o200k" in record.getMessage()]
    assert len(warnings) == 1
    assert set(results["corpus_a"]["T1_bpe_raw_32k"][run.RAW]) == {"E1_bpe_32k"}


def test_compute_tpp_delta_is_split_minus_raw_against_the_shared_control() -> None:
    corpus = _corpus()
    results = run.compute_tpp_delta(
        [corpus],
        _arms(),
        [("T1_bpe_raw_32k", "T4_bpe_split_32k")],
        _control(),
        n_bootstrap=50,
        seed=0,
        ci=0.95,
    )
    entry = results["corpus_a"]["T4_bpe_split_32k/T1_bpe_raw_32k"]
    english = sum(len(text) for text in corpus.english)
    expected_split = sum(len(text) for text in corpus.texts[run.SPLIT]) / english
    expected_raw = sum(len(text) for text in corpus.texts[run.RAW]) / english
    assert entry["value_a"] == pytest.approx(expected_split)
    assert entry["value_b"] == pytest.approx(expected_raw)
    assert entry["delta"] == pytest.approx(expected_split - expected_raw)
    assert entry["pivot"] == "E1_bpe_32k"
    assert entry["label"] == "bpe 32k"
    assert entry["split_variant"] == run.SPLIT
    assert entry["ci_low"] <= entry["delta"] <= entry["ci_high"]


def test_compute_tpp_delta_skips_a_pair_whose_control_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arms = {name: _char_arm(name) for name in ("T1_bpe_raw_32k", "T4_bpe_split_32k")}
    with caplog.at_level("WARNING", logger="exp03"):
        results = run.compute_tpp_delta(
            [_corpus()],
            arms,
            [("T1_bpe_raw_32k", "T4_bpe_split_32k")],
            _control(),
            n_bootstrap=0,
            seed=0,
            ci=0.95,
        )
    assert results["corpus_a"] == {}
    assert "E1_bpe_32k" in " ".join(record.getMessage() for record in caplog.records)


def test_fertility_primary_uses_the_raw_word_count_for_split_arms() -> None:
    """Every arm over the same denominator: the raw sentence's whitespace-word count
    (docs/decisions.md, "Fertility for split arms uses the raw word count as the primary
    denominator"). With one-token-per-character fakes, the split arm's primary fertility is
    the split text's non-space character count over the raw text's word count."""
    corpus = _corpus()
    primary, secondary, comp = run.compute_fertility_compression([corpus], _arms(), _specs())
    raw_words = sum(len(text.split()) for text in corpus.texts[run.RAW])
    split_chars = sum(len(word) for text in corpus.texts[run.SPLIT] for word in text.split())
    split_words = sum(len(text.split()) for text in corpus.texts[run.SPLIT])

    split_primary = primary["corpus_a"]["T4_bpe_split_32k"]
    assert split_primary["unit"] == "tokens/reference word"
    assert split_primary["n"] == raw_words
    assert split_primary["value"] == pytest.approx(split_chars / raw_words)
    assert split_primary["variant"] == run.SPLIT

    # the secondary is the same numerator over the split text's own words — a different
    # denominator, which is exactly why the primary exists
    split_secondary = secondary["corpus_a"]["T4_bpe_split_32k"]
    assert split_secondary["unit"] == "tokens/word"
    assert split_secondary["value"] == pytest.approx(split_chars / split_words)

    # a raw arm has no secondary entry: its primary already is plain fertility
    assert primary["corpus_a"]["T1_bpe_raw_32k"]["unit"] == "tokens/word"
    assert "T1_bpe_raw_32k" not in secondary["corpus_a"]
    assert comp["corpus_a"]["T4_bpe_split_32k"]["variant"] == run.SPLIT


def test_compression_is_measured_on_the_text_the_arm_tokenizes() -> None:
    corpus = _corpus()
    _, _, comp = run.compute_fertility_compression([corpus], _arms(), _specs())
    raw_bytes = sum(len(text.encode("utf-8")) for text in corpus.texts[run.RAW])
    raw_tokens = sum(len(text) for text in corpus.texts[run.RAW])
    assert comp["corpus_a"]["T1_bpe_raw_32k"]["value"] == pytest.approx(raw_bytes / raw_tokens)


def test_tokenizer_sources_hashes_files_and_names_the_splitter_for_t4(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "tokenizer.json"
    path.write_bytes(b'{"model": "fake"}')
    arms = {
        "T4_bpe_split_32k": run.LoadedTokenizer(
            name="T4_bpe_split_32k",
            source_id=str(path),
            vocab_size=32000,
            _encode=lambda text: [0],
            family="T4",
            attempted=(str(path),),
        ),
        "T0_o200k": _char_arm("T0_o200k"),
    }
    sources = run.tokenizer_sources(arms, "fake/splitter@deadbeef")
    assert sources["T4_bpe_split_32k"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # the splitter is as much a part of a T4 arm's provenance as its own tokenizer.json
    assert sources["T4_bpe_split_32k"]["splitter_source_id"] == "fake/splitter@deadbeef"
    assert "sha256" not in sources["T0_o200k"]
    assert "splitter_source_id" not in sources["T0_o200k"]


# --- leakage check --------------------------------------------------------------------


def test_exclusion_check_counts_missing_hashes() -> None:
    from sanskrit_tok.data.exclusion import sentence_hash

    hashes = frozenset({sentence_hash(SANSKRIT[0]), sentence_hash(SANSKRIT[1])})
    assert run.exclusion_check_for(SANSKRIT, hashes) == {"n": 4, "n_missing": 2}


def test_exclusion_check_uses_the_english_hash_for_the_english_side() -> None:
    hashes = frozenset({sentence_hash_en(text) for text in ENGLISH})
    assert run.exclusion_check_for(ENGLISH, hashes, sentence_hash_en) == {"n": 4, "n_missing": 0}


# --- the figure -----------------------------------------------------------------------


def _synthetic_results() -> dict[str, Any]:
    """Two corpora, one matched pair present and one whose split arm never loaded."""

    def entry(value: float, low: float, high: float) -> dict[str, Any]:
        return {"value": value, "ci_low": low, "ci_high": high, "n": 4, "unit": "x"}

    def corpus_block(raw: float, split: float) -> dict[str, Any]:
        return {
            "T1_bpe_raw_32k": {run.RAW: {"E1_bpe_32k": entry(raw, raw - 0.05, raw + 0.05)}},
            "T4_bpe_split_32k": {
                run.SPLIT: {"E1_bpe_32k": entry(split, split - 0.05, split + 0.05)},
                run.SPLIT_MODEL: {"E1_bpe_32k": entry(split - 0.1, split - 0.15, split - 0.05)},
            },
        }

    return {
        "config": {
            "corpora": [{"name": "corpus_a"}, {"name": "corpus_b"}],
            "matched_pairs": [
                ["T1_bpe_raw_32k", "T4_bpe_split_32k"],
                ["T2_unigram_raw_64k", "T4_unigram_split_64k"],
            ],
            "english_control": {
                "T1_bpe_raw_32k": "E1_bpe_32k",
                "T4_bpe_split_32k": "E1_bpe_32k",
                "T2_unigram_raw_64k": "E1_unigram_64k",
                "T4_unigram_split_64k": "E1_unigram_64k",
            },
        },
        "splitter": {"source_id": "fake/splitter@deadbeef"},
        "unavailable_arms": {
            "T4_unigram_split_64k": "T4_unigram_split_64k: trained tokenizer file not found",
            "T2_unigram_raw_64k": "T2_unigram_raw_64k: trained tokenizer file not found",
        },
        "tpp": {
            "corpus_a": corpus_block(1.10, 0.95),
            "corpus_b": corpus_block(0.62, 0.58),
        },
    }


def test_make_figure_writes_a_pdf_and_a_png(tmp_path: Path) -> None:
    paths = run.make_figure(_synthetic_results(), tmp_path)
    assert [path.name for path in paths] == ["tpp_split_vs_raw.pdf", "tpp_split_vs_raw.png"]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_make_figure_creates_a_missing_output_directory(tmp_path: Path) -> None:
    paths = run.make_figure(_synthetic_results(), tmp_path / "nested" / "outputs")
    assert all(path.exists() for path in paths)


def test_build_figure_has_one_row_per_corpus_and_pair_labels() -> None:
    import matplotlib.pyplot as plt

    figure = run._build_figure(_synthetic_results())
    try:
        assert len(figure.axes) == 2  # one row per corpus, single column
        labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert labels == ["bpe 32k"]  # the unavailable pair leaves no gap
        legend_labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert any("raw" in label for label in legend_labels)
        assert any("split" in label for label in legend_labels)
    finally:
        plt.close(figure)


def test_build_figure_caption_states_the_splitter_and_what_split_means() -> None:
    import matplotlib.pyplot as plt

    figure = run._build_figure(_synthetic_results())
    try:
        texts = " ".join(text.get_text() for text in figure.texts)
        assert "fake/splitter@deadbeef" in texts
        assert "provisional" in texts.lower()
        assert "compound" in texts  # the splitter splits samāsa as well as sandhi
        assert "reconciled" in texts
        assert "T4_unigram_split_64k" in texts  # the omitted arm is named
    finally:
        plt.close(figure)


def test_build_figure_rejects_a_results_dict_with_no_corpora() -> None:
    results = _synthetic_results()
    results["config"]["corpora"] = []
    with pytest.raises(ValueError, match="corpora"):
        run._build_figure(results)


# --- end to end, offline --------------------------------------------------------------


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A complete, synthetic Experiment 03 environment: config, split cache, manifest,
    exclusion lists, fake corpora and fake tokenizers, all under `tmp_path`."""
    corpus_names = ["corpus_a", "corpus_b"]
    split_dir = tmp_path / "split"
    for name in corpus_names:
        _write_jsonl(split_dir / f"{name}.jsonl", _split_records())
    (split_dir / "manifest.json").write_text(
        json.dumps(_manifest(corpus_names)), encoding="utf-8"
    )

    exclusion = tmp_path / "exclusion.txt"
    exclusion_en = tmp_path / "exclusion_en.txt"
    build_exclusion_list({"corpus_a": SANSKRIT}, exclusion)
    build_exclusion_list({"corpus_a": ENGLISH}, exclusion_en, hash_fn=sentence_hash_en)

    config: dict[str, Any] = {
        "experiment": "03_sandhi_split_test",
        "corpora": [
            {"name": name, "loader": "samayik", "split": "test"} for name in corpus_names
        ],
        "split_cache_path": str(split_dir),
        "split_manifest_path": str(split_dir / "manifest.json"),
        "arms_raw": ["T1_bpe_raw_32k"],
        "arms_split": ["T4_bpe_split_32k"],
        "matched_pairs": [["T1_bpe_raw_32k", "T4_bpe_split_32k"]],
        "english_control": _control(),
        "deployed_pivot": "T0_o200k",
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
    monkeypatch.setattr(run, "load_tokenizer", lambda name: _char_arm(name))
    yield {"config": config, "config_path": config_path, "out_dir": tmp_path / "out"}


def test_run_end_to_end_writes_strict_json_a_config_and_both_figures(
    experiment: dict[str, Any],
) -> None:
    """The whole runner, offline: `main` parses the config, every metric is computed over
    the synthetic corpora, and the three artifacts CLAUDE.md §2.9 requires land in the
    output directory."""

    def reject_non_finite(token: str) -> float:
        raise AssertionError(f"results.json is not strict JSON: {token}")

    assert run.main(["--config", str(experiment["config_path"])]) == 0

    out_dir: Path = experiment["out_dir"]
    assert (out_dir / "config.yaml").exists()
    for name in ("tpp_split_vs_raw.pdf", "tpp_split_vs_raw.png"):
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
        "splitter",
        "tokenizer_sources",
        "unavailable_arms",
        "corpora",
        "exclusion_check",
        "exclusion_check_en",
        "splitter_stats",
        "tpp",
        "tpp_delta",
        "fertility_primary",
        "fertility_secondary",
        "compression",
    ):
        assert key in results, key
    assert results["splitter"]["source_id"] == "fake/splitter@deadbeef"
    assert set(results["corpora"]) == {"corpus_a", "corpus_b"}
    # both variants of the split arm are measured, against both pivots
    variants = results["tpp"]["corpus_a"]["T4_bpe_split_32k"]
    assert set(variants) == {run.SPLIT, run.SPLIT_MODEL}
    assert set(variants[run.SPLIT]) == {"E1_bpe_32k", "T0_o200k"}
    delta = results["tpp_delta"]["corpus_a"]["T4_bpe_split_32k/T1_bpe_raw_32k"]
    assert delta["delta"] == pytest.approx(delta["value_a"] - delta["value_b"])
    assert results["fertility_primary"]["corpus_a"]["T4_bpe_split_32k"]["unit"] == (
        "tokens/reference word"
    )
    assert results["splitter_stats"]["corpus_a"]["mean_units_split"] == pytest.approx(2.5)
    # the evaluation text is in both exclusion lists, so nothing leaked into any arm
    assert results["exclusion_check"]["corpus_a"] == {"n": 4, "n_missing": 0}
    assert results["exclusion_check_en"]["corpus_a"] == {"n": 4, "n_missing": 0}


def test_run_aborts_when_a_corpus_is_not_in_the_split_cache(
    experiment: dict[str, Any], tmp_path: Path
) -> None:
    """The abort that matters most in practice: the split job is still running."""
    (tmp_path / "split" / "corpus_b.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="corpus_b"):
        run.run(experiment["config"], experiment["config_path"])


def test_run_records_an_untrained_arm_as_unavailable_rather_than_aborting(
    experiment: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One missing arm must not cost the whole run; it is recorded and its columns are
    simply absent (this is how a T4 arm behaves before it has been trained)."""

    def load(name: str) -> Any:
        if name == "T4_bpe_split_32k":
            raise run.TokenizerUnavailable(f"{name}: trained tokenizer file not found")
        return _char_arm(name)

    monkeypatch.setattr(run, "load_tokenizer", load)
    results = run.run(experiment["config"], experiment["config_path"])
    assert "T4_bpe_split_32k" in results["unavailable_arms"]
    assert "T4_bpe_split_32k" not in results["tpp"]["corpus_a"]
    assert results["tpp_delta"]["corpus_a"] == {}
    assert not math.isnan(results["tpp"]["corpus_a"]["T1_bpe_raw_32k"][run.RAW]["E1_bpe_32k"]
                          ["value"])
