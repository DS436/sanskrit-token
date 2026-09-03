"""Tests for the Experiment 01 runner (`experiments/01_baseline_penalty/run.py`).

Everything here is offline and synthetic: fake tokenizers, a handful of hand-written
sentences, and a tiny results dict. The runner's own I/O (FLORES, the Hugging Face hub)
is never touched, so these tests pin the aggregation, filtering and plotting logic rather
than the experiment's numbers.

`run.py` is not importable as a package module (`experiments/` holds scripts, not a
package), so it is loaded by path.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "experiments" / "01_baseline_penalty" / "run.py"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp01_run", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run_module()


class CharTokenizer:
    """One token per character, whitespace included."""

    name = "char"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class WordTokenizer:
    """One token per whitespace-delimited word."""

    name = "word"

    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


CHAR = CharTokenizer()
WORD = WordTokenizer()


# --- summarise_metric ------------------------------------------------------------


def test_summarise_metric_replaces_the_distribution_with_its_mean_and_std() -> None:
    result = run.summarise_metric(
        {"value": 2.0, "n": 3, "unit": "tokens/word", "per_word": [1, 2, 3]}
    )
    assert result["value"] == 2.0
    assert result["n"] == 3
    assert result["unit"] == "tokens/word"
    assert result["distribution"] == "per_word"
    assert result["mean"] == pytest.approx(2.0)
    # Population std of [1, 2, 3] is sqrt(2/3).
    assert result["std"] == pytest.approx(0.816496580927726)
    assert "per_word" not in result


def test_summarise_metric_handles_per_text_and_per_pair() -> None:
    from_text = run.summarise_metric(
        {"value": 1.5, "n": 2, "unit": "bytes/token", "per_text": [1.0, 2.0]}
    )
    assert from_text["distribution"] == "per_text"
    assert from_text["mean"] == pytest.approx(1.5)
    assert from_text["std"] == pytest.approx(0.5)

    from_pair = run.summarise_metric(
        {"value": 3.0, "n": 2, "unit": "token ratio", "per_pair": [2.0, 4.0]}
    )
    assert from_pair["distribution"] == "per_pair"
    assert from_pair["mean"] == pytest.approx(3.0)
    assert from_pair["std"] == pytest.approx(1.0)


def test_summarise_metric_on_an_empty_or_absent_distribution_reports_none_not_zero() -> None:
    """`0.0` is a plausible ratio, so it would read as a measurement rather than as none."""
    empty = run.summarise_metric({"value": 0.0, "n": 0, "unit": "tokens/word", "per_word": []})
    assert empty["mean"] is None
    assert empty["std"] is None
    assert empty["distribution"] == "per_word"

    bare = run.summarise_metric({"value": 1.0, "n": 1, "unit": "tokens/word"})
    assert bare["distribution"] is None
    assert bare["mean"] is None
    assert bare["std"] is None


def test_summarise_metric_none_survives_the_json_round_trip_as_null() -> None:
    bare = run.summarise_metric({"value": 1.0, "n": 1, "unit": "tokens/word"})
    assert json.loads(json.dumps(bare)) == bare
    assert '"mean": null' in json.dumps(bare)


def test_summarise_metric_of_a_single_item_has_zero_std_but_a_real_mean() -> None:
    """Zero std is a genuine measurement here, unlike the empty case above."""
    single = run.summarise_metric({"value": 2.0, "n": 1, "unit": "tokens/word", "per_word": [2]})
    assert single["mean"] == pytest.approx(2.0)
    assert single["std"] == 0.0


def test_summarise_metric_output_is_json_serialisable() -> None:
    summary = run.summarise_metric(
        {"value": 2.0, "n": 3, "unit": "tokens/word", "per_word": [1, 2, 3]}
    )
    assert json.loads(json.dumps(summary)) == summary


# --- blank-line filtering --------------------------------------------------------


def test_select_aligned_indices_drops_every_index_blank_in_any_language() -> None:
    sentences = {
        "a": ["one", "", "three", "four"],
        "b": ["uno", "dos", "   ", "cuatro"],
    }
    assert run.select_aligned_indices(sentences) == [0, 3]


def test_select_aligned_indices_keeps_everything_when_nothing_is_blank() -> None:
    sentences = {"a": ["x", "y"], "b": ["p", "q"]}
    assert run.select_aligned_indices(sentences) == [0, 1]


def test_take_indices_preserves_alignment() -> None:
    sentences = {"a": ["one", "", "three"], "b": ["uno", "dos", "tres"]}
    taken = run.take_indices(sentences, [0, 2])
    assert taken == {"a": ["one", "three"], "b": ["uno", "tres"]}


# --- roundtrip reporting ---------------------------------------------------------


def test_roundtrip_report_separates_latin_failures_from_the_rest() -> None:
    sentences = [
        "नमस्ते",  # roundtrips
        "नमस्ते World",  # fails, and contains Latin: expected by construction
        "मासः 4",  # fails on the ASCII digit, which comes back as Devanagari '४'
        "क़",  # fails with no ASCII letter or digit at all: the interesting case
    ]
    report = run.roundtrip_report(sentences)
    assert report["n"] == 4
    assert report["failures"] == 3
    assert report["failures_with_latin"] == 1
    # "no Latin letter" still admits the ASCII digit, which is why the tighter bucket
    # exists: on FLORES it is the difference between 225 failures and 89.
    assert report["failures_pure_devanagari"] == 2
    assert report["failures_no_ascii_alnum"] == 1
    assert report["examples_pure_devanagari"] == [
        {"index": 2, "sentence": "मासः 4"},
        {"index": 3, "sentence": "क़"},
    ]
    assert report["examples_no_ascii_alnum"] == [{"index": 3, "sentence": "क़"}]


def test_roundtrip_report_caps_the_examples_it_lists() -> None:
    report = run.roundtrip_report(["क़"] * 20, max_examples=5)
    assert report["failures_pure_devanagari"] == 20
    assert report["failures_no_ascii_alnum"] == 20
    assert len(report["examples_pure_devanagari"]) == 5
    assert len(report["examples_no_ascii_alnum"]) == 5


def test_roundtrip_report_on_clean_devanagari_reports_no_failures() -> None:
    report = run.roundtrip_report(["नमस्ते", "रामः गच्छति।", "संस्कृतम्"])
    assert report["failures"] == 0
    assert report["failures_no_ascii_alnum"] == 0
    assert report["examples_pure_devanagari"] == []
    assert report["examples_no_ascii_alnum"] == []


# --- SLP1 coverage ---------------------------------------------------------------


def test_slp1_coverage_counts_nukta_and_leaked_devanagari_separately() -> None:
    """The two counts overlap only partly: neither bucket contains the other.

    `"क़"` carries a nukta and transliterates to ASCII (`"k0a"`), so it is counted as a
    nukta sentence but not as a non-ASCII one. `"डॉक्टर"` has no nukta, but candra-o is
    outside SLP1 and passes through unconverted, so it is counted the other way round.
    """
    original = [
        "नमस्ते",  # clean: inside SLP1's inventory
        "क़ानून",  # nukta -> literal ASCII '0'
        "डॉक्टर",  # candra-o leaks through as raw Devanagari
    ]
    slp1 = [run.to_slp1(text, "devanagari") for text in original]
    # Pin the two failure modes, so the test fails if `sanscript` changes behaviour.
    assert slp1[1].startswith("k0")
    assert "ॉ" in slp1[2]

    report = run.slp1_coverage(original, slp1)
    assert report["n"] == 3
    assert report["n_with_nukta"] == 1
    assert report["n_non_ascii_after_slp1"] == 1
    assert report["examples_non_ascii"] == [{"index": 2, "slp1": slp1[2]}]


def test_slp1_coverage_counts_precomposed_nukta_letters_too() -> None:
    """`ढ़` exists both as U+095D and as ढ + U+093C; both must count."""
    precomposed = "\u095d"
    decomposed = "\u0922\u093c"
    report = run.slp1_coverage(
        [precomposed, decomposed, "क"],
        ["ignored", "ignored", "ka"],
    )
    assert report["n_with_nukta"] == 2


def test_slp1_coverage_on_clean_sanskrit_reports_nothing() -> None:
    original = ["नमस्ते", "रामः गच्छति"]
    slp1 = [run.to_slp1(text, "devanagari") for text in original]
    report = run.slp1_coverage(original, slp1)
    assert report == {
        "n": 2,
        "n_with_nukta": 0,
        "n_non_ascii_after_slp1": 0,
        "examples_non_ascii": [],
    }


def test_slp1_coverage_caps_the_examples_it_lists() -> None:
    report = run.slp1_coverage(["x"] * 20, ["ॉ"] * 20, max_examples=5)
    assert report["n_non_ascii_after_slp1"] == 20
    assert len(report["examples_non_ascii"]) == 5


def test_slp1_coverage_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same corpus"):
        run.slp1_coverage(["a", "b"], ["a"])


def test_slp1_coverage_output_is_json_serialisable() -> None:
    report = run.slp1_coverage(["क़"], ["ॉ"])
    assert json.loads(json.dumps(report)) == report


# --- script variants -------------------------------------------------------------


def test_script_variants_adds_slp1_for_devanagari_languages_only() -> None:
    deva = run.script_variants(["नमस्ते"], "san_Deva")
    assert list(deva) == ["original", "slp1"]
    assert deva["original"] == ["नमस्ते"]
    assert deva["slp1"] == ["namaste"]

    latin = run.script_variants(["Hello"], "eng_Latn")
    assert list(latin) == ["original"]
    assert latin["original"] == ["Hello"]


# --- metric and parity computation -----------------------------------------------


def test_compute_metrics_covers_every_tokenizer_language_and_variant() -> None:
    variants = {
        "san_Deva": {"original": ["नम ते"], "slp1": ["nama te"]},
        "eng_Latn": {"original": ["a bc"]},
    }
    metrics = run.compute_metrics([WORD], variants)
    assert list(metrics) == ["word"]
    assert list(metrics["word"]) == ["san_Deva", "eng_Latn"]
    assert list(metrics["word"]["san_Deva"]) == ["original", "slp1"]
    assert list(metrics["word"]["eng_Latn"]) == ["original"]
    # WordTokenizer emits one token per word, so fertility is exactly 1.0 everywhere.
    assert metrics["word"]["eng_Latn"]["original"]["fertility"]["value"] == pytest.approx(1.0)
    assert metrics["word"]["eng_Latn"]["original"]["fertility"]["n"] == 2
    # "a bc" is 4 UTF-8 bytes over 2 tokens.
    assert metrics["word"]["eng_Latn"]["original"]["compression"]["value"] == pytest.approx(2.0)
    assert "per_word" not in metrics["word"]["eng_Latn"]["original"]["fertility"]


def test_compute_parity_covers_each_pivot_plus_the_slp1_source() -> None:
    variants = {
        "san_Deva": {"original": ["abcd"], "slp1": ["ab"]},
        "hin_Deva": {"original": ["abc"], "slp1": ["abc"]},
        "eng_Latn": {"original": ["ab"]},
    }
    parity = run.compute_parity([CHAR], variants, ["eng_Latn", "hin_Deva"], "san_Deva")
    assert list(parity) == ["char"]
    assert list(parity["char"]) == ["eng_Latn", "hin_Deva", "eng_Latn__slp1"]
    # CharTokenizer: 4 chars of Sanskrit over 2 chars of English.
    assert parity["char"]["eng_Latn"]["value"] == pytest.approx(2.0)
    assert parity["char"]["eng_Latn"]["pivot"] == "eng_Latn"
    assert parity["char"]["eng_Latn"]["source_variant"] == "original"
    assert parity["char"]["hin_Deva"]["value"] == pytest.approx(4 / 3)
    # The SLP1 Sanskrit side is 2 chars against the same 2-char English pivot.
    assert parity["char"]["eng_Latn__slp1"]["value"] == pytest.approx(1.0)
    assert parity["char"]["eng_Latn__slp1"]["source_variant"] == "slp1"
    assert "per_pair" not in parity["char"]["eng_Latn"]


def test_compute_parity_skips_the_slp1_row_when_the_source_has_no_slp1_variant() -> None:
    variants = {"san_Deva": {"original": ["abcd"]}, "eng_Latn": {"original": ["ab"]}}
    parity = run.compute_parity([CHAR], variants, ["eng_Latn"], "san_Deva")
    assert list(parity["char"]) == ["eng_Latn"]


# --- the figure ------------------------------------------------------------------


def _synthetic_results() -> dict[str, object]:
    def cell(value: float) -> dict[str, object]:
        return {
            "fertility": {"value": value, "n": 10, "unit": "tokens/word", "mean": value},
            "compression": {"value": 3.0, "n": 10, "unit": "bytes/token", "mean": 3.0},
        }

    return {
        "metrics": {
            "T0_o200k": {
                "san_Deva": {"original": cell(6.5), "slp1": cell(4.0)},
                "hin_Deva": {"original": cell(3.5), "slp1": cell(2.5)},
                "eng_Latn": {"original": cell(1.3)},
            },
            "T0_llama4": {
                "san_Deva": {"original": cell(5.5), "slp1": cell(3.8)},
                "hin_Deva": {"original": cell(3.0), "slp1": cell(2.4)},
                "eng_Latn": {"original": cell(1.2)},
            },
        }
    }


def test_make_figure_writes_a_pdf_and_a_png(tmp_path: Path) -> None:
    paths = run.make_figure(_synthetic_results(), tmp_path)
    assert [path.name for path in paths] == [
        "fertility_by_language.pdf",
        "fertility_by_language.png",
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_make_figure_creates_a_missing_output_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "outputs"
    paths = run.make_figure(_synthetic_results(), out_dir)
    assert all(path.exists() for path in paths)


def test_make_figure_labels_the_y_axis_by_the_metric_and_never_calls_it_the_headline() -> None:
    assert run.FIGURE_YLABEL == "Fertility (tokens per whitespace word)"
    assert "headline" not in run.FIGURE_TITLE.lower()


def test_make_figure_needs_at_least_one_tokenizer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no tokenizers"):
        run.make_figure({"metrics": {}}, tmp_path)


# --- config resolution -----------------------------------------------------------


def test_resolve_path_makes_relative_config_paths_repo_relative(tmp_path: Path) -> None:
    assert run.resolve_path("a/b.json", tmp_path) == tmp_path / "a" / "b.json"


def test_resolve_path_leaves_absolute_paths_alone(tmp_path: Path) -> None:
    absolute = tmp_path / "already" / "absolute.json"
    assert run.resolve_path(str(absolute), tmp_path / "elsewhere") == absolute


def test_repo_root_is_the_parent_of_the_experiments_directory() -> None:
    assert run.repo_root() == REPO_ROOT
    assert (run.repo_root() / "experiments").is_dir()
