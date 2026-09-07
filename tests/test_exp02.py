"""Tests for the Experiment 02 runner (`experiments/02_tpp_parallel/run.py`).

Everything here is offline and synthetic: a hand-built results dict for the figure, tiny
sentence lists for the exclusion-check helper, and the config's own `script_variants`
mapping for the per-family variant selector. No corpus, tokenizer download, or the
network is touched, so these tests pin the aggregation, filtering and plotting logic
rather than the experiment's numbers (real numbers are exercised by actually running the
experiment, per the task brief).

`run.py` is not importable as a package module (`experiments/` holds scripts, not a
package), so it is loaded by path, exactly as `tests/test_exp01.py` does.

The JSON sanitiser and the TPP summary enricher this script used to define are now
`sanskrit_tok.experiment`'s `sanitize_json` and `summarise_tpp`; the arm loader, the
`tokenizer_sources` provenance block, the file hash, the leakage check and the
unavailable-arm caption followed them there when Experiment 03 needed the same five. All
of their tests live in `tests/test_experiment.py`; what remains here is this script's own
wiring.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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


def test_variants_for_family_unknown_family_names_itself_and_the_config_keys() -> None:
    """A bare `KeyError: 'T5'` from mid-run says nothing about which config key is short."""
    with pytest.raises(ValueError) as excinfo:
        run.variants_for_family("T5", SCRIPT_VARIANTS)
    message = str(excinfo.value)
    assert "T5" in message
    assert "script_variants" in message
    for family in SCRIPT_VARIANTS:
        assert family in message


# --- a fake arm, shared by the tests below -------------------------------------------


def _fake_arm(name: str, source_id: str, family: str) -> object:
    """A `LoadedTokenizer` whose `encode` is never called; only its provenance is read."""
    return run.LoadedTokenizer(
        name=name,
        source_id=source_id,
        vocab_size=32000,
        _encode=lambda text: [len(text)],
        family=family,
        attempted=(source_id,),
    )


# --- arm-label helper (figure x-tick labels) ----------------------------------------


def test_arm_label_marks_t1_provisional_with_vocab_in_thousands() -> None:
    assert run.arm_label("T1_bpe_raw_32k", 32000) == "T1_bpe_raw_32k* (32k)"


def test_arm_label_marks_t2_provisional_too() -> None:
    assert run.arm_label("T2_unigram_raw_64k", 64000) == "T2_unigram_raw_64k* (64k)"


def test_arm_label_t0_carries_no_star() -> None:
    assert run.arm_label("T0_o200k", 200019) == "T0_o200k (200k)"


def test_arm_label_t3_carries_no_star() -> None:
    assert run.arm_label("T3_sarvam", 68096) == "T3_sarvam (68k)"


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


def test_build_tpp_figure_omits_unavailable_arm_and_captions_it() -> None:
    """`T3_missing` sits in `config["sanskrit_arms"]` but has no `tpp` entry (it is in
    `unavailable_arms` instead): it must not appear on any panel's x-axis, and the
    caption must name it, built straight from `results["unavailable_arms"]`."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_figure(_synthetic_results())
    try:
        for axes in figure.axes:
            tick_labels = [label.get_text() for label in axes.get_xticklabels()]
            assert not any("T3_missing" in label for label in tick_labels)
            # the two arms that DO have data are still there, in config order
            assert any("T0_o200k" in label for label in tick_labels)
            assert any("T1_bpe_raw_32k" in label for label in tick_labels)

        caption_texts = [text.get_text() for text in figure.texts]
        assert any("T3_missing" in text for text in caption_texts)
        assert any("provisional" in text for text in caption_texts)
    finally:
        plt.close(figure)


# --- unavailable English pivot -------------------------------------------------------


def _one_corpus() -> object:
    """One two-sentence corpus with an SLP1 Sanskrit side and an English side."""
    return run.CorpusData(
        name="corpus_a",
        split="test",
        n_total=2,
        n_used=2,
        sanskrit={run.SLP1: ["rAmaH gacCati", "sItA vadati"]},
        english=["Rama goes", "Sita speaks"],
        hindi=None,
    )


def test_compute_tpp_warns_once_per_unavailable_pivot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing pivot removes a whole column from every corpus and arm below, so it is
    logged once, before the loops — not once per corpus x arm x variant, and not silently."""
    arms = {
        "T0_o200k": _fake_arm("T0_o200k", "o200k_base", "T0"),
        "T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", "tokenizer.json", "T1"),
    }
    with caplog.at_level("WARNING", logger="exp02"):
        results = run.compute_tpp(
            [_one_corpus(), _one_corpus()],  # two corpora: the warning must not repeat
            arms,
            ["T1_bpe_raw_32k"],
            ["T0_o200k", "T0_llama4"],
            {"T1": ["slp1"]},
            n_bootstrap=10,
            seed=0,
            ci=0.95,
        )
    warnings = [record for record in caplog.records if "T0_llama4" in record.getMessage()]
    assert len(warnings) == 1
    assert "unavailable" in warnings[0].getMessage()
    # the available pivot is still measured; the unavailable one leaves no placeholder
    pivots = results["corpus_a"]["T1_bpe_raw_32k"]["slp1"]
    assert set(pivots) == {"T0_o200k"}


def test_compute_tpp_does_not_warn_when_every_pivot_is_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arms = {
        "T0_o200k": _fake_arm("T0_o200k", "o200k_base", "T0"),
        "T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", "tokenizer.json", "T1"),
    }
    with caplog.at_level("WARNING", logger="exp02"):
        run.compute_tpp(
            [_one_corpus()],
            arms,
            ["T1_bpe_raw_32k"],
            ["T0_o200k"],
            {"T1": ["slp1"]},
            n_bootstrap=10,
            seed=0,
            ci=0.95,
        )
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


# --- controlled TPP: T1/T2 against the matched English control E1 ---------------------


def _controlled_arms() -> dict[str, object]:
    return {
        "T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", "sa.json", "T1"),
        "E1_bpe_32k": _fake_arm("E1_bpe_32k", "en.json", "E1"),
    }


def test_controlled_pair_key_joins_the_two_arm_names() -> None:
    assert run.controlled_pair_key("T1_bpe_raw_32k", "E1_bpe_32k") == "T1_bpe_raw_32k/E1_bpe_32k"


def test_select_controlled_pairs_keeps_pairs_whose_both_sides_loaded() -> None:
    pairs = run.select_controlled_pairs(
        [["T1_bpe_raw_32k", "E1_bpe_32k"]],
        _controlled_arms(),  # type: ignore[arg-type]
    )
    assert pairs == [("T1_bpe_raw_32k", "E1_bpe_32k")]


def test_select_controlled_pairs_skips_a_pair_with_an_unavailable_side(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An untrained E1 arm (or an unavailable Sanskrit arm) must drop its pair rather than
    silently pairing the Sanskrit arm against something else."""
    with caplog.at_level("WARNING", logger="exp02"):
        pairs = run.select_controlled_pairs(
            [
                ["T1_bpe_raw_32k", "E1_bpe_32k"],
                ["T1_bpe_raw_64k", "E1_bpe_64k"],  # neither side loaded
                ["T1_bpe_raw_32k", "E1_unigram_32k"],  # English side missing
            ],
            _controlled_arms(),  # type: ignore[arg-type]
        )
    assert pairs == [("T1_bpe_raw_32k", "E1_bpe_32k")]
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "E1_bpe_64k" in messages
    assert "E1_unigram_32k" in messages


def test_select_controlled_pairs_rejects_a_malformed_pair() -> None:
    with pytest.raises(ValueError, match="controlled_pairs"):
        run.select_controlled_pairs(
            [["T1_bpe_raw_32k"]],
            _controlled_arms(),  # type: ignore[arg-type]
        )


def test_compute_tpp_controlled_is_keyed_by_corpus_then_pair() -> None:
    results = run.compute_tpp_controlled(
        [_one_corpus()],
        _controlled_arms(),  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
        n_bootstrap=10,
        seed=0,
        ci=0.95,
    )
    assert list(results) == ["corpus_a"]
    summary = results["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]
    for key in (
        "value",
        "n",
        "unit",
        "distribution",
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "ci",
        "n_bootstrap",
        "seed",
        "n_undefined",
        "source_tokens",
        "pivot_tokens",
    ):
        assert key in summary
    assert summary["n"] == 2
    assert summary["ci"] == 0.95


def _char_arm(name: str, family: str) -> object:
    """A `LoadedTokenizer` emitting one id per character, so a token count is a length."""
    return run.LoadedTokenizer(
        name=name,
        source_id=f"{name}.json",
        vocab_size=32000,
        _encode=lambda text: [0] * len(text),
        family=family,
        attempted=(f"{name}.json",),
    )


def test_compute_tpp_controlled_reads_the_slp1_sanskrit_side_and_raw_english() -> None:
    """The Sanskrit arm scores SLP1 (T1/T2 have no other variant) and the control arm
    scores the English side as written — these two fakes emit one id per character, so
    the recorded token counts are exactly the character lengths of those two sides."""
    corpus = _one_corpus()
    results = run.compute_tpp_controlled(
        [corpus],
        {
            "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
            "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
        },  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
        n_bootstrap=0,
        seed=0,
        ci=0.95,
    )
    summary = results["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]
    expected_source = sum(len(text) for text in corpus.sanskrit[run.SLP1])  # type: ignore[attr-defined]
    expected_pivot = sum(len(text) for text in corpus.english)  # type: ignore[attr-defined]
    assert summary["source_tokens"] == expected_source
    assert summary["pivot_tokens"] == expected_pivot
    assert summary["value"] == expected_source / expected_pivot


# --- the controlled column of the central figure --------------------------------------


def _synthetic_results_with_control() -> dict[str, object]:
    results = _synthetic_results()
    config = results["config"]
    assert isinstance(config, dict)
    config["controlled_pairs"] = [["T1_bpe_raw_32k", "E1_bpe_32k"]]
    config["figure_pivot_controlled"] = True
    sources = results["tokenizer_sources"]
    assert isinstance(sources, dict)
    sources["E1_bpe_32k"] = {"source_id": "en.json", "vocab_size": 32000, "family": "E1"}
    results["tpp_controlled"] = {
        "corpus_a": {
            "T1_bpe_raw_32k/E1_bpe_32k": {
                "value": 0.85,
                "ci_low": 0.80,
                "ci_high": 0.90,
                "n": 5,
                "unit": "x",
            }
        },
        "corpus_b": {
            "T1_bpe_raw_32k/E1_bpe_32k": {
                "value": 1.15,
                "ci_low": 1.10,
                "ci_high": 1.20,
                "n": 5,
                "unit": "x",
            }
        },
    }
    return results


def test_controlled_pair_label_names_both_arms_and_marks_the_provisional_side() -> None:
    assert run.controlled_pair_label("T1_bpe_raw_32k", "E1_bpe_32k") == (
        "T1_bpe_raw_32k* / E1_bpe_32k"
    )


def test_build_tpp_figure_adds_a_controlled_column() -> None:
    """Two columns, one row per corpus: the left column is unchanged (Sanskrit vs o200k),
    the right column is the matched control (T1/T2 vs E1)."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_figure(_synthetic_results_with_control())
    try:
        assert len(figure.axes) == 4  # two corpora x (deployed, controlled)
        right_labels = [
            label.get_text()
            for axes in figure.axes[1::2]
            for label in axes.get_xticklabels()
        ]
        assert any("E1_bpe_32k" in label for label in right_labels)
        left_labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert not any("E1_bpe_32k" in label for label in left_labels)
        texts = [text.get_text() for text in figure.texts]
        assert any(run.FIGURE_CONTROLLED_TITLE in text for text in texts)
        assert any(run.FIGURE_DEPLOYED_TITLE in text for text in texts)
        assert any(run.FIGURE_SUPTITLE in text for text in texts)
        # the two columns use different English sides, so the suptitle names neither
        assert "o200k" not in run.FIGURE_SUPTITLE
    finally:
        plt.close(figure)


def test_build_tpp_figure_stays_single_column_without_controlled_results() -> None:
    """The controlled column appears only when the run produced one, so a results.json
    from before this task still plots."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_figure(_synthetic_results())
    try:
        assert len(figure.axes) == 2
        # the single column is still labelled with the pivot it is measured against
        texts = [text.get_text() for text in figure.texts]
        assert any(run.FIGURE_DEPLOYED_TITLE in text for text in texts)
        assert not any(run.FIGURE_CONTROLLED_TITLE in text for text in texts)
    finally:
        plt.close(figure)


# --- Rényi efficiency (secondary intrinsic) -------------------------------------------


def test_english_renyi_arm_names_is_pivots_then_controlled_english_sides() -> None:
    assert run.english_renyi_arm_names(
        ["T0_o200k", "T0_llama4"],
        [["T1_bpe_raw_32k", "E1_bpe_32k"], ["T2_unigram_raw_32k", "E1_unigram_32k"]],
    ) == ["T0_o200k", "T0_llama4", "E1_bpe_32k", "E1_unigram_32k"]


def test_english_renyi_arm_names_deduplicates_a_repeated_control_arm() -> None:
    """Two Sanskrit arms may be matched to the same English control; it is measured once."""
    assert run.english_renyi_arm_names(
        ["T0_o200k"],
        [["T1_bpe_raw_32k", "E1_bpe_32k"], ["T1_bpe_raw_64k", "E1_bpe_32k"]],
    ) == ["T0_o200k", "E1_bpe_32k"]


def test_english_renyi_arm_names_deduplicates_a_pivot_used_as_a_control() -> None:
    assert run.english_renyi_arm_names(
        ["T0_o200k"], [["T1_bpe_raw_32k", "T0_o200k"]]
    ) == ["T0_o200k"]


def test_english_renyi_arm_names_without_controlled_pairs_is_just_the_pivots() -> None:
    assert run.english_renyi_arm_names(["T0_o200k", "T0_llama4"], []) == [
        "T0_o200k",
        "T0_llama4",
    ]


def test_compute_renyi_is_keyed_by_corpus_arm_variant_then_alpha() -> None:
    results = run.compute_renyi(
        [_one_corpus()],
        {"T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1")},  # type: ignore[arg-type]
        ["T1_bpe_raw_32k"],
        {"T1": ["slp1"]},
        [2.5, 3.0],
    )
    assert set(results) == {"corpus_a"}
    assert set(results["corpus_a"]) == {"T1_bpe_raw_32k"}
    assert set(results["corpus_a"]["T1_bpe_raw_32k"]) == {"slp1"}
    entry = results["corpus_a"]["T1_bpe_raw_32k"]["slp1"]
    assert set(entry) == {"2.5", "3.0"}  # JSON object keys are strings
    for alpha, summary in entry.items():
        for key in (
            "value",
            "n",
            "unit",
            "entropy_bits",
            "n_types",
            "n_tokens",
            "efficiency_nominal",
            "vocab_size",
            "alpha",
        ):
            assert key in summary
        assert summary["alpha"] == float(alpha)
        assert summary["unit"] == "renyi efficiency"
        # `_char_arm` emits one id per character, all of them 0: one type, so the
        # efficiency is undefined by construction and the entropy is zero.
        assert summary["n_tokens"] == sum(len(text) for text in _one_corpus().sanskrit["slp1"])  # type: ignore[attr-defined]
        assert summary["vocab_size"] == 32000


def test_compute_renyi_skips_an_unavailable_arm_without_a_placeholder() -> None:
    results = run.compute_renyi(
        [_one_corpus()],
        {"T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1")},  # type: ignore[arg-type]
        ["T1_bpe_raw_32k", "T2_unigram_raw_32k"],
        {"T1": ["slp1"], "T2": ["slp1"]},
        [2.5],
    )
    assert set(results["corpus_a"]) == {"T1_bpe_raw_32k"}


def test_compute_renyi_english_is_keyed_by_corpus_arm_then_alpha() -> None:
    corpus = _one_corpus()
    results = run.compute_renyi_english(
        [corpus],
        {"T0_o200k": _char_arm("T0_o200k", "T0")},  # type: ignore[arg-type]
        ["T0_o200k", "T0_llama4"],  # the second arm did not load this run
        [2.5, 3.0],
    )
    assert set(results["corpus_a"]) == {"T0_o200k"}
    entry = results["corpus_a"]["T0_o200k"]
    assert set(entry) == {"2.5", "3.0"}
    assert entry["2.5"]["n_tokens"] == sum(len(text) for text in corpus.english)  # type: ignore[attr-defined]


def test_compute_renyi_measures_the_real_distribution_of_a_two_type_arm() -> None:
    """One id per character, split by whether it is a space: the SLP1 side of `_one_corpus`
    is 24 characters with 2 spaces, so the counts are [22, 2] and the efficiency at α=2 is
    hand-computable as log2((22/24)^2 + (2/24)^2) / (1 - 2) over log2(2) = 1."""
    import math

    arm = run.LoadedTokenizer(
        name="fake",
        source_id="fake.json",
        vocab_size=4,
        _encode=lambda text: [1 if c == " " else 0 for c in text],
        family="T1",
        attempted=("fake.json",),
    )
    results = run.compute_renyi(
        [_one_corpus()], {"fake": arm}, ["fake"], {"T1": ["slp1"]}, [2.0]
    )
    summary = results["corpus_a"]["fake"]["slp1"]["2.0"]
    expected = math.log2((22 / 24) ** 2 + (2 / 24) ** 2) / (1 - 2.0)
    assert summary["n_tokens"] == 24
    assert summary["n_types"] == 2
    assert summary["entropy_bits"] == pytest.approx(expected)
    assert summary["value"] == pytest.approx(expected)
    assert summary["efficiency_nominal"] == pytest.approx(expected / 2)  # log2(4) = 2
