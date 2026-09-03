"""Tests for the Experiment 02 runner (`experiments/02_tpp_parallel/run.py`).

Everything here is offline and synthetic: a hand-built results dict for the figure, tiny
sentence lists for the exclusion-check helper, and the config's own `script_variants`
mapping for the per-family variant selector. No corpus, tokenizer download, or the
network is touched, so these tests pin the aggregation, filtering and plotting logic
rather than the experiment's numbers (real numbers are exercised by actually running the
experiment, per the task brief).

`run.py` is not importable as a package module (`experiments/` holds scripts, not a
package), so it is loaded by path, exactly as `tests/test_exp01.py` does.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "experiments" / "02_tpp_parallel" / "run.py"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp02_run", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run_module()

#: The config's `script_variants` mapping, copied verbatim from `config.yaml` so this
#: test pins the same object the real run reads, without parsing YAML.
SCRIPT_VARIANTS = {
    "T0": ["original", "slp1"],
    "T3": ["original", "slp1"],
    "T1": ["slp1"],
    "T2": ["slp1"],
}


# --- per-family script variant selection ------------------------------------------


def test_variants_for_family_t0_has_both_scripts() -> None:
    assert run.variants_for_family("T0", SCRIPT_VARIANTS) == ["original", "slp1"]


def test_variants_for_family_t1_is_slp1_only() -> None:
    assert run.variants_for_family("T1", SCRIPT_VARIANTS) == ["slp1"]


def test_variants_for_family_t3_matches_t0() -> None:
    assert run.variants_for_family("T3", SCRIPT_VARIANTS) == ["original", "slp1"]


def test_variants_for_family_t2_is_slp1_only() -> None:
    assert run.variants_for_family("T2", SCRIPT_VARIANTS) == ["slp1"]


# --- exclusion-check helper ---------------------------------------------------------


def test_exclusion_check_counts_missing_hashes() -> None:
    sentences = ["रामः", "सीता", "लक्ष्मणः"]
    hashes = frozenset({run.sentence_hash("रामः"), run.sentence_hash("सीता")})
    report = run.exclusion_check_for(sentences, hashes)
    assert report == {"n": 3, "n_missing": 1}


def test_exclusion_check_all_present_is_zero_missing() -> None:
    sentences = ["रामः", "सीता"]
    hashes = frozenset({run.sentence_hash("रामः"), run.sentence_hash("सीता")})
    assert run.exclusion_check_for(sentences, hashes) == {"n": 2, "n_missing": 0}


def test_exclusion_check_none_present_is_all_missing() -> None:
    sentences = ["रामः", "सीता"]
    hashes: frozenset[str] = frozenset()
    assert run.exclusion_check_for(sentences, hashes) == {"n": 2, "n_missing": 2}


# --- arm-label helper (figure x-tick labels) ----------------------------------------


def test_arm_label_marks_t1_provisional_with_vocab_in_thousands() -> None:
    assert run.arm_label("T1_bpe_raw_32k", 32000) == "T1_bpe_raw_32k* (32k)"


def test_arm_label_marks_t2_provisional_too() -> None:
    assert run.arm_label("T2_unigram_raw_64k", 64000) == "T2_unigram_raw_64k* (64k)"


def test_arm_label_t0_carries_no_star() -> None:
    assert run.arm_label("T0_o200k", 200019) == "T0_o200k (200k)"


def test_arm_label_t3_carries_no_star() -> None:
    assert run.arm_label("T3_sarvam", 68096) == "T3_sarvam (68k)"


# --- JSON NaN sanitiser (strict-JSON resolution) ------------------------------------


def test_sanitize_json_turns_nan_and_inf_into_none() -> None:
    sanitized = run.sanitize_json({"a": math.nan, "b": math.inf, "c": [1.0, math.nan]})
    assert sanitized == {"a": None, "b": None, "c": [1.0, None]}
    # Must survive a strict json.dump (allow_nan=False) with no exception.
    text = json.dumps(sanitized, allow_nan=False)
    assert json.loads(text) == sanitized


def test_sanitize_json_leaves_ordinary_values_alone() -> None:
    assert run.sanitize_json({"value": 1.5, "n": 3, "unit": "x", "nested": {"y": [1, 2]}}) == {
        "value": 1.5,
        "n": 3,
        "unit": "x",
        "nested": {"y": [1, 2]},
    }


# --- the central figure --------------------------------------------------------------


def _synthetic_results() -> dict[str, object]:
    """Two corpora, three arms: one T0 (both variants), one provisional T1 (slp1 only),
    one arm listed in `sanskrit_arms` but absent from `tpp` because it is unavailable."""

    def entry(value: float, lo: float, hi: float) -> dict[str, object]:
        return {"value": value, "ci_low": lo, "ci_high": hi, "n": 5, "unit": "x"}

    return {
        "config": {
            "corpora": [{"name": "corpus_a"}, {"name": "corpus_b"}],
            "sanskrit_arms": ["T0_o200k", "T1_bpe_raw_32k", "T3_missing"],
        },
        "tokenizer_sources": {
            "T0_o200k": {"source_id": "o200k_base", "vocab_size": 200019, "family": "T0"},
            "T1_bpe_raw_32k": {
                "source_id": "outputs/.../tokenizer.json",
                "vocab_size": 32000,
                "family": "T1",
            },
        },
        "unavailable_arms": {"T3_missing": "no candidate tokenizer could be loaded"},
        "tpp": {
            "corpus_a": {
                "T0_o200k": {
                    "original": {"T0_o200k": entry(2.1, 1.9, 2.3)},
                    "slp1": {"T0_o200k": entry(1.8, 1.6, 2.0)},
                },
                "T1_bpe_raw_32k": {
                    "slp1": {"T0_o200k": entry(0.9, 0.8, 1.0)},
                },
            },
            "corpus_b": {
                "T0_o200k": {
                    "original": {"T0_o200k": entry(2.5, 2.3, 2.7)},
                    "slp1": {"T0_o200k": entry(2.0, 1.8, 2.2)},
                },
                "T1_bpe_raw_32k": {
                    "slp1": {"T0_o200k": entry(0.95, 0.85, 1.05)},
                },
            },
        },
    }


def test_make_figure_writes_a_pdf_and_a_png(tmp_path: Path) -> None:
    paths = run.make_figure(_synthetic_results(), tmp_path)
    assert [path.name for path in paths] == ["tpp_by_arm.pdf", "tpp_by_arm.png"]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_make_figure_creates_a_missing_output_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "outputs"
    paths = run.make_figure(_synthetic_results(), out_dir)
    assert all(path.exists() for path in paths)
