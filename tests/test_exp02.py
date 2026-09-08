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
    "T7": ["original", "slp1"],
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
    assert [path.name for path in paths] == [
        "tpp_by_arm.pdf",
        "tpp_by_arm.png",
        "tpp_by_length.pdf",
        "tpp_by_length.png",
    ]
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


# --- length-stratified TPP -------------------------------------------------------------


def test_length_bin_labels_names_closed_ranges_and_an_open_last_bin() -> None:
    assert run.length_bin_labels(run.LENGTH_BIN_EDGES_DEFAULT) == [
        "1-8",
        "9-16",
        "17-24",
        "25-40",
        "41+",
    ]


def test_length_bin_labels_rejects_empty_edges() -> None:
    with pytest.raises(ValueError, match="edge"):
        run.length_bin_labels([])


@pytest.mark.parametrize(
    ("n_words", "expected"),
    [
        (0, 0),  # below the first edge: bin 0, not a bin of its own
        (1, 0),
        (8, 0),
        (9, 1),
        (16, 1),
        (17, 2),
        (24, 2),
        (25, 3),
        (40, 3),
        (41, 4),
        (200, 4),
    ],
)
def test_assign_length_bin_at_every_edge(n_words: int, expected: int) -> None:
    assert run.assign_length_bin(n_words, run.LENGTH_BIN_EDGES_DEFAULT) == expected


def _length_corpus() -> object:
    """Four pairs whose English sides fall in bins 0, 0, 1 and 2, leaving 3 and 4 empty."""
    return run.CorpusData(
        name="corpus_a",
        split="test",
        n_total=4,
        n_used=4,
        sanskrit={
            run.ORIGINAL: ["rAmaH", "sItA", "gacCati", "vadati"],
            run.SLP1: ["rAmaH", "sItA", "gacCati", "vadati"],
        },
        english=["aa bb", "cc dd", " ".join(["w"] * 10), " ".join(["w"] * 20)],
        hindi=None,
    )


def _length_bins(
    sparse_below: int = 2,
    *,
    bin_on: str = run.BIN_ON_ENGLISH,
    edges: object = None,
) -> dict[str, dict[str, object]]:
    results = run.compute_tpp_by_length(
        [_length_corpus()],  # type: ignore[list-item]
        {
            "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
            "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
        },  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", run.SLP1, "E1_bpe_32k")],
        run.LENGTH_BIN_EDGES_DEFAULT if edges is None else edges,  # type: ignore[arg-type]
        n_bootstrap=0,
        seed=0,
        ci=0.95,
        sparse_below=sparse_below,
        bin_on=bin_on,  # type: ignore[arg-type]
    )
    return results["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]


def test_compute_tpp_by_length_is_keyed_by_corpus_pair_and_bin_label() -> None:
    bins = _length_bins()
    assert list(bins) == ["1-8", "9-16", "17-24", "25-40", "41+"]
    assert [bins[label]["n_pairs"] for label in bins] == [2, 1, 1, 0, 0]
    # every pair of the corpus lands in exactly one bin
    assert sum(int(bins[label]["n_pairs"]) for label in bins) == 4


def test_compute_tpp_by_length_value_is_the_bins_hand_computed_ratio_of_sums() -> None:
    """Bin `1-8` holds the first two pairs; these fakes emit one id per character, so the
    ratio is (len("rAmaH") + len("sItA")) / (len("aa bb") + len("cc dd")) = 9 / 10."""
    bins = _length_bins()
    assert bins["1-8"]["source_tokens"] == 9
    assert bins["1-8"]["pivot_tokens"] == 10
    assert bins["1-8"]["value"] == pytest.approx(0.9)
    assert bins["1-8"]["mean_words_en"] == pytest.approx(2.0)
    assert bins["1-8"]["mean_words_sa"] == pytest.approx(1.0)
    assert bins["1-8"]["variant"] == run.SLP1


def test_compute_tpp_by_length_flags_bins_below_the_sparse_threshold() -> None:
    bins = _length_bins(sparse_below=2)
    assert bins["1-8"]["sparse"] is False  # two pairs, threshold two
    assert bins["9-16"]["sparse"] is True  # one pair
    assert all(bins[label]["sparse"] is True for label in ("9-16", "17-24", "25-40", "41+"))


def test_compute_tpp_by_length_gives_an_empty_bin_a_nan_entry_not_a_missing_key() -> None:
    import math

    empty = _length_bins()["41+"]
    assert empty["n_pairs"] == 0
    assert empty["sparse"] is True
    assert math.isnan(float(empty["value"]))  # type: ignore[arg-type]
    assert math.isnan(float(empty["ci_low"]))  # type: ignore[arg-type]
    assert math.isnan(float(empty["ci_high"]))  # type: ignore[arg-type]
    assert math.isnan(float(empty["mean_words_en"]))  # type: ignore[arg-type]
    assert math.isnan(float(empty["mean_words_sa"]))  # type: ignore[arg-type]


def test_compute_tpp_by_length_encodes_each_side_once_not_once_per_bin() -> None:
    """The counts come from one `token_ratio` pass and are subset per bin; re-encoding
    per bin would multiply the run's cost by the number of bins."""
    calls: list[str] = []

    def counting(text: str) -> list[int]:
        calls.append(text)
        return [0] * len(text)

    arm = run.LoadedTokenizer(
        name="both_sides",
        source_id="both.json",
        vocab_size=32000,
        _encode=counting,
        family="T1",
        attempted=("both.json",),
    )
    run.compute_tpp_by_length(
        [_length_corpus()],  # type: ignore[list-item]
        {"both_sides": arm},
        [("both_sides", run.SLP1, "both_sides")],
        run.LENGTH_BIN_EDGES_DEFAULT,
        n_bootstrap=0,
        seed=0,
        ci=0.95,
    )
    assert len(calls) == 8  # four Sanskrit sentences + four English, once each


# --- the mirror stratification: binning on the Sanskrit side --------------------------


def _two_sided_corpus() -> object:
    """Four pairs the two stratifications group differently.

    English word counts 2, 2, 10, 20 put the pairs in English bins 0, 0, 1, 2; Sanskrit
    word counts 12, 2, 3, 7 put the *same* pairs in Sanskrit bins 2, 0, 0, 1. The per-bin
    counts happen to match, so a test that only counted pairs would pass under either
    stratification; what differs is which pair is in which bin.
    """
    return run.CorpusData(
        name="corpus_a",
        split="test",
        n_total=4,
        n_used=4,
        sanskrit={
            run.ORIGINAL: [" ".join(["sa"] * 12), "rAmaH sItA", "a b c", " ".join(["va"] * 7)],
            run.SLP1: [" ".join(["sa"] * 12), "rAmaH sItA", "a b c", " ".join(["va"] * 7)],
        },
        english=["aa bb", "cc dd", " ".join(["w"] * 10), " ".join(["w"] * 20)],
        hindi=None,
    )


def _two_sided_bins(bin_on: str, edges: object) -> dict[str, dict[str, object]]:
    results = run.compute_tpp_by_length(
        [_two_sided_corpus()],  # type: ignore[list-item]
        {
            "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
            "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
        },  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", run.SLP1, "E1_bpe_32k")],
        edges,  # type: ignore[arg-type]
        n_bootstrap=0,
        seed=0,
        ci=0.95,
        sparse_below=2,
        bin_on=bin_on,  # type: ignore[arg-type]
    )
    return results["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]


def test_compute_tpp_by_length_bins_on_the_sanskrit_side_when_asked() -> None:
    """`bin_on="sanskrit"` assigns bins from the Sanskrit side's word count, not the
    English one: the 12-word Sanskrit sentence is in the top populated bin even though
    its English side is the shortest of the four."""
    bins = _two_sided_bins(run.BIN_ON_SANSKRIT, run.LENGTH_BIN_EDGES_SA_DEFAULT)
    assert list(bins) == ["1-5", "6-10", "11-15", "16-25", "26+"]
    assert [bins[label]["n_pairs"] for label in bins] == [2, 1, 1, 0, 0]
    # bin "11-15" holds exactly the 12-word Sanskrit sentence, whose English side is 2 words
    assert bins["11-15"]["mean_words_sa"] == pytest.approx(12.0)
    assert bins["11-15"]["mean_words_en"] == pytest.approx(2.0)
    # the same pair under English binning sits in the *lowest* bin instead
    english_bins = _two_sided_bins(run.BIN_ON_ENGLISH, run.LENGTH_BIN_EDGES_DEFAULT)
    assert english_bins["1-8"]["n_pairs"] == 2
    assert english_bins["1-8"]["mean_words_sa"] == pytest.approx(7.0)  # 12 and 2 words


def test_compute_tpp_by_length_records_which_side_it_binned_on() -> None:
    """Every summary carries `bin_on`, so a `results.json` reader can tell the two
    stratifications apart without consulting which key they were stored under."""
    english = _two_sided_bins(run.BIN_ON_ENGLISH, run.LENGTH_BIN_EDGES_DEFAULT)
    sanskrit = _two_sided_bins(run.BIN_ON_SANSKRIT, run.LENGTH_BIN_EDGES_SA_DEFAULT)
    assert all(summary["bin_on"] == run.BIN_ON_ENGLISH for summary in english.values())
    assert all(summary["bin_on"] == run.BIN_ON_SANSKRIT for summary in sanskrit.values())


def test_the_two_stratifications_partition_the_same_corpus_into_the_same_totals() -> None:
    """Stratifying differently must not change what is being stratified: both binnings
    cover every pair exactly once, and their per-bin token counts sum to the same two
    corpus totals — so a difference between the two rows is a difference in grouping."""
    english = _two_sided_bins(run.BIN_ON_ENGLISH, run.LENGTH_BIN_EDGES_DEFAULT)
    sanskrit = _two_sided_bins(run.BIN_ON_SANSKRIT, run.LENGTH_BIN_EDGES_SA_DEFAULT)
    for key in ("n_pairs", "source_tokens", "pivot_tokens"):
        assert sum(int(summary[key]) for summary in english.values()) == sum(  # type: ignore[arg-type]
            int(summary[key]) for summary in sanskrit.values()  # type: ignore[arg-type]
        )
    assert sum(int(summary["n_pairs"]) for summary in english.values()) == 4  # type: ignore[arg-type]


def test_compute_tpp_by_length_rejects_an_unknown_bin_on() -> None:
    """Falling back to the English side would make a run that measured one stratification
    twice indistinguishable from one that measured both."""
    with pytest.raises(ValueError, match="bin_on"):
        _two_sided_bins("hindi", run.LENGTH_BIN_EDGES_SA_DEFAULT)


def test_sanskrit_bin_labels_name_the_configured_edges() -> None:
    assert run.length_bin_labels(run.LENGTH_BIN_EDGES_SA_DEFAULT) == [
        "1-5",
        "6-10",
        "11-15",
        "16-25",
        "26+",
    ]


def test_select_length_pairs_takes_controlled_pairs_and_the_primary_pivot() -> None:
    """Set (a) reads the controlled variant; set (b) reads each family's own script —
    `original` for T0, its only variant for T1 — against the first English pivot."""
    arms = {
        "T0_o200k": _fake_arm("T0_o200k", "o200k_base", "T0"),
        "T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", "sa.json", "T1"),
        "E1_bpe_32k": _fake_arm("E1_bpe_32k", "en.json", "E1"),
    }
    pairs = run.select_length_pairs(
        arms,  # type: ignore[arg-type]
        ["T0_o200k", "T1_bpe_raw_32k"],
        ["T0_o200k", "T0_llama4"],
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
        SCRIPT_VARIANTS,
    )
    assert pairs == [
        ("T1_bpe_raw_32k", run.CONTROLLED_VARIANT, "E1_bpe_32k"),
        ("T0_o200k", run.ORIGINAL, "T0_o200k"),
        ("T1_bpe_raw_32k", run.SLP1, "T0_o200k"),
    ]


def test_select_length_pairs_drops_the_deployed_set_when_the_pivot_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="exp02"):
        pairs = run.select_length_pairs(
            {"T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", "sa.json", "T1")},  # type: ignore[arg-type]
            ["T1_bpe_raw_32k"],
            ["T0_o200k"],
            [],
            SCRIPT_VARIANTS,
        )
    assert pairs == []
    assert any("T0_o200k" in record.getMessage() for record in caplog.records)


# --- the length-stratified figure ------------------------------------------------------


def _length_bin_entry(value: float, lo: float, hi: float, n_pairs: int) -> dict[str, object]:
    return {
        "value": value,
        "ci_low": lo,
        "ci_high": hi,
        "n": n_pairs,
        "unit": "x",
        "n_pairs": n_pairs,
        "sparse": n_pairs < 30,
        "mean_words_en": 5.0,
        "mean_words_sa": 3.0,
        "variant": "slp1",
    }


def _empty_length_bin_entry() -> dict[str, object]:
    nan = float("nan")
    entry = _length_bin_entry(nan, nan, nan, 0)
    entry["mean_words_en"] = nan
    entry["mean_words_sa"] = nan
    return entry


def _synthetic_results_with_length() -> dict[str, object]:
    """Two corpora x two stratifications, the four cases the figure has to handle.

    English-binned: `corpus_a` has three populated bins (one of them sparse) and two
    empty ones; `corpus_b`'s bins are all empty. Sanskrit-binned: `corpus_a` sits on a
    visibly different range — so the two rows' panels cannot share a y-axis — and its
    sparse bin is planted far off that range, which must be drawn at the panel edge
    rather than being allowed to set the scale; `corpus_b` is again all empty.
    """
    results = _synthetic_results()
    config = results["config"]
    assert isinstance(config, dict)
    config["controlled_pairs"] = [["T1_bpe_raw_32k", "E1_bpe_32k"]]
    config["length_sparse_below"] = 30
    labels = run.length_bin_labels(run.LENGTH_BIN_EDGES_DEFAULT)
    labels_sa = run.length_bin_labels(run.LENGTH_BIN_EDGES_SA_DEFAULT)
    results["length_bin_edges"] = list(run.LENGTH_BIN_EDGES_DEFAULT)
    results["length_bin_edges_sa"] = list(run.LENGTH_BIN_EDGES_SA_DEFAULT)
    results["tpp_by_length"] = {
        "corpus_a": {
            "T1_bpe_raw_32k/E1_bpe_32k": {
                labels[0]: _length_bin_entry(0.90, 0.85, 0.95, 120),
                labels[1]: _length_bin_entry(0.95, 0.90, 1.00, 80),
                labels[2]: _length_bin_entry(1.20, 0.90, 1.50, 4),  # sparse: hollow
                labels[3]: _empty_length_bin_entry(),
                labels[4]: _empty_length_bin_entry(),
            },
            # deployed practice: recorded in results.json, never drawn (CLAUDE.md §2.5)
            "T1_bpe_raw_32k/T0_o200k": {
                label: _length_bin_entry(2.0, 1.9, 2.1, 120) for label in labels
            },
        },
        "corpus_b": {
            "T1_bpe_raw_32k/E1_bpe_32k": {label: _empty_length_bin_entry() for label in labels}
        },
    }
    results["tpp_by_length_sa"] = {
        "corpus_a": {
            "T1_bpe_raw_32k/E1_bpe_32k": {
                labels_sa[0]: _length_bin_entry(1.30, 1.25, 1.35, 120),
                labels_sa[1]: _length_bin_entry(1.35, 1.30, 1.40, 80),
                # planted extreme, sparse: off the range the two dense bins set
                labels_sa[2]: _length_bin_entry(9.00, 3.00, 21.00, 4),
                labels_sa[3]: _empty_length_bin_entry(),
                labels_sa[4]: _empty_length_bin_entry(),
            }
        },
        "corpus_b": {
            "T1_bpe_raw_32k/E1_bpe_32k": {label: _empty_length_bin_entry() for label in labels_sa}
        },
    }
    return results


def test_build_tpp_by_length_figure_has_a_row_per_stratification() -> None:
    """Two rows (English bins, then Sanskrit bins) x one column per corpus, each row
    ticked and labelled with its own bin variable."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_by_length_figure(_synthetic_results_with_length())
    try:
        assert len(figure.axes) == 4  # 2 stratifications x 2 corpora
        english_panel, sanskrit_panel = figure.axes[0], figure.axes[2]
        assert [label.get_text() for label in english_panel.get_xticklabels()] == [
            "1-8",
            "9-16",
            "17-24",
            "25-40",
            "41+",
        ]
        assert [label.get_text() for label in sanskrit_panel.get_xticklabels()] == [
            "1-5",
            "6-10",
            "11-15",
            "16-25",
            "26+",
        ]
        assert "English" in english_panel.get_xlabel()
        assert "Sanskrit" in sanskrit_panel.get_xlabel()
        # both panels of a row name their corpus
        assert english_panel.get_title(loc="left") == "corpus_a"
        assert sanskrit_panel.get_title(loc="left") == "corpus_a"
        # the row headers state the bin variable
        texts = [text.get_text() for text in figure.texts]
        assert any("ENGLISH" in text for text in texts)
        assert any("SANSKRIT" in text for text in texts)
    finally:
        plt.close(figure)


def test_build_tpp_by_length_figure_legends_the_controlled_pair_once() -> None:
    import matplotlib.pyplot as plt

    figure = run._build_tpp_by_length_figure(_synthetic_results_with_length())
    try:
        assert len(figure.legends) == 1
        legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
        # only the controlled pair; the deployed-pivot series is results.json-only
        assert legend_labels == [run.controlled_pair_label("T1_bpe_raw_32k", "E1_bpe_32k")]
        assert not any("T0_o200k" in label for label in legend_labels)
        footnote = [text.get_text() for text in figure.texts]
        assert any("English whitespace word count" in text for text in footnote)
        assert any("30" in text for text in footnote)
    finally:
        plt.close(figure)


def test_build_tpp_by_length_figure_scales_each_panel_to_its_own_data() -> None:
    """The panels no longer share one y-axis: `corpus_a`'s two stratifications sit on
    different ranges, and each is tight around its own non-sparse bins."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_by_length_figure(_synthetic_results_with_length())
    try:
        english_limits = figure.axes[0].get_ylim()
        sanskrit_limits = figure.axes[2].get_ylim()
        assert english_limits != sanskrit_limits
        # English panel: dense bins span 0.85-1.00, so the top stays well under the
        # sparse 1.20-1.50 bin; the Sanskrit panel's top stays far under its 9.00 one.
        assert english_limits[1] < 1.2
        assert sanskrit_limits[1] < 2.0
        assert sanskrit_limits[0] > 1.0 - 0.5  # tight around 1.25-1.40, 1.0 kept inside
    finally:
        plt.close(figure)


def test_build_tpp_by_length_figure_marks_an_off_scale_bin_at_the_edge() -> None:
    """The planted 9.00 [3.00, 21.00] bin is drawn as a triangle at the panel edge with
    its value annotated, rather than silently dropped or allowed to flatten the panel."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_by_length_figure(_synthetic_results_with_length())
    try:
        panel = figure.axes[2]
        annotations = [text.get_text() for text in panel.texts]
        assert any("9.00" in text for text in annotations)
        assert any(text.startswith("▲") for text in annotations)
        # and nothing off-scale is drawn on a panel whose bins all fit
        assert not figure.axes[1].texts
    finally:
        plt.close(figure)


def test_build_tpp_by_length_figure_survives_a_corpus_whose_bins_are_all_empty() -> None:
    """`corpus_b` has an entry for every bin and a value for none; the panel must draw
    its reference line and no points rather than raising or producing an empty y range."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_by_length_figure(_synthetic_results_with_length())
    try:
        bottom, top = figure.axes[1].get_ylim()
        assert bottom < 1.0 < top  # the sign-flip threshold stays inside the panel
    finally:
        plt.close(figure)


def test_build_tpp_by_length_figure_without_any_length_results_still_plots() -> None:
    """A results.json from before this task carries no `tpp_by_length`; the figure is
    then empty axes rather than a crash."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_by_length_figure(_synthetic_results())
    try:
        assert len(figure.axes) == 4
        assert figure.axes[0].get_ylim() == (0.0, 2.0)
    finally:
        plt.close(figure)


def test_make_figure_writes_the_length_figure_too(tmp_path: Path) -> None:
    paths = run.make_figure(_synthetic_results_with_length(), tmp_path)
    length_paths = [path for path in paths if path.stem == run.LENGTH_FIGURE_STEM]
    assert [path.name for path in length_paths] == ["tpp_by_length.pdf", "tpp_by_length.png"]
    for path in length_paths:
        assert path.exists() and path.stat().st_size > 0


# --- the byte-matched control, the byte reference and the side decomposition ------------


def _decomposition_corpus() -> object:
    """One corpus whose two sides have hand-countable characters and bytes.

    The Sanskrit side is given in both variants: `original` is Devanagari (three UTF-8
    bytes per character) and `slp1` its ASCII transliteration, so a test can tell a
    character count from a byte count.
    """
    return run.CorpusData(
        name="corpus_a",
        split="test",
        n_total=2,
        n_used=2,
        sanskrit={
            run.ORIGINAL: ["रामः", "सीता"],
            run.SLP1: ["rAmaH", "sItA"],
        },
        english=["Rama", "Sita speaks"],
        hindi=None,
    )


def _byte_arm(name: str) -> object:
    """A `LoadedTokenizer` emitting one id per UTF-8 byte, like the real `T7_byt5`."""
    return run.LoadedTokenizer(
        name=name,
        source_id="bytes/utf-8",
        vocab_size=256,
        _encode=lambda text: list(text.encode("utf-8")),
        family="T7",
        attempted=("bytes",),
    )


def test_arm_label_writes_a_sub_thousand_vocabulary_in_full() -> None:
    """`T7_byt5` has 256 ids; "0k" would read as a bug rather than as a byte vocabulary."""
    assert run.arm_label("T7_byt5", 256) == "T7_byt5 (256)"


def test_select_controlled_pairs_accepts_the_same_arm_on_both_sides() -> None:
    """The byte reference is `T7_byt5` over `T7_byt5` — a ratio of byte counts, which is
    the one controlled row that owes nothing to any vocabulary."""
    arms = {"T7_byt5": _byte_arm("T7_byt5")}
    assert run.select_controlled_pairs(
        [["T7_byt5", "T7_byt5"]],
        arms,  # type: ignore[arg-type]
    ) == [("T7_byt5", "T7_byt5")]


def test_is_byte_reference_pair_only_for_the_byte_arm_on_both_sides() -> None:
    assert run.is_byte_reference_pair("T7_byt5", "T7_byt5")
    assert not run.is_byte_reference_pair("T7_byt5", "E1_bpe_32k")
    assert not run.is_byte_reference_pair("T1_bpe_raw_32k", "T1_bpe_raw_32k")


def test_compute_side_decomposition_is_keyed_by_corpus_and_pair_and_multiplies_back() -> None:
    corpus = _decomposition_corpus()
    arms = {
        "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
        "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
    }
    results = run.compute_side_decomposition(
        [corpus],  # type: ignore[list-item]
        arms,  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
    )
    entry = results["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]
    assert entry["variant"] == run.SLP1 and entry["n"] == 2
    # SLP1 is ASCII: "rAmaH" + "sItA" is 9 characters and 9 bytes; the English side is
    # "Rama" + "Sita speaks" = 15 of each. Both fakes emit one id per character.
    assert entry["chars_sa"] == 9 and entry["bytes_sa"] == 9
    assert entry["chars_en"] == 15 and entry["bytes_en"] == 15
    assert entry["tokens_sa"] == 9 and entry["tokens_en"] == 15
    assert entry["char_ratio"] == pytest.approx(9 / 15)
    assert entry["density_ratio"] == pytest.approx(1.0)
    assert entry["tpp"] == pytest.approx(9 / 15)
    assert abs(entry["char_ratio"] * entry["density_ratio"] - entry["tpp"]) < 1e-9


def test_side_decomposition_of_the_byte_arm_is_the_ratio_of_bytes() -> None:
    """The check the experiment's verification runs: a byte arm's token count *is* its byte
    count, so its `tpp` must equal `bytes_sa / bytes_en` exactly."""
    corpus = _decomposition_corpus()
    results = run.compute_side_decomposition(
        [corpus],  # type: ignore[list-item]
        {"T7_byt5": _byte_arm("T7_byt5")},  # type: ignore[arg-type]
        [("T7_byt5", "T7_byt5")],
    )
    entry = results["corpus_a"]["T7_byt5/T7_byt5"]
    assert entry["tpp"] == pytest.approx(entry["bytes_sa"] / entry["bytes_en"])


def test_side_decomposition_reads_the_original_script_when_asked() -> None:
    """`variant` is part of the entry because the character counts depend on it: the same
    sentences in Devanagari are three bytes per character and fewer characters."""
    corpus = _decomposition_corpus()
    results = run.compute_side_decomposition(
        [corpus],  # type: ignore[list-item]
        {"T7_byt5": _byte_arm("T7_byt5")},  # type: ignore[arg-type]
        [("T7_byt5", "T7_byt5")],
        variant=run.ORIGINAL,
    )
    entry = results["corpus_a"]["T7_byt5/T7_byt5"]
    assert entry["variant"] == run.ORIGINAL
    assert entry["chars_sa"] == 8  # रामः + सीता
    assert entry["bytes_sa"] == 24  # three UTF-8 bytes per Devanagari character


def test_decomposition_pairs_takes_controlled_then_the_primary_pivot() -> None:
    arms = {
        "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
        "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
        "T0_o200k": _char_arm("T0_o200k", "T0"),
        "T7_byt5": _byte_arm("T7_byt5"),
    }
    pairs = run.decomposition_pairs(
        arms,  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", "E1_bpe_32k"), ("T7_byt5", "T7_byt5")],
        ["T1_bpe_raw_32k", "T7_byt5", "T3_missing"],
        ["T0_o200k", "T0_llama4"],
    )
    assert pairs == [
        ("T1_bpe_raw_32k", "E1_bpe_32k"),
        ("T7_byt5", "T7_byt5"),
        ("T1_bpe_raw_32k", "T0_o200k"),
        ("T7_byt5", "T0_o200k"),
    ]


def test_decomposition_pairs_without_an_available_pivot_is_the_controlled_set() -> None:
    arms = {"T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1")}
    assert run.decomposition_pairs(
        arms,  # type: ignore[arg-type]
        [],
        ["T1_bpe_raw_32k"],
        ["T0_o200k"],
    ) == []


# --- the block bootstrap, as wired into every stored summary ---------------------------


def test_compute_tpp_controlled_adds_the_block_interval_when_asked() -> None:
    corpus = _one_corpus()
    arms = _controlled_arms()
    with_block = run.compute_tpp_controlled(
        [corpus],
        arms,  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
        n_bootstrap=20,
        seed=0,
        ci=0.95,
        block_length=1,
    )["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]
    without = run.compute_tpp_controlled(
        [corpus],
        arms,  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
        n_bootstrap=20,
        seed=0,
        ci=0.95,
    )["corpus_a"]["T1_bpe_raw_32k/E1_bpe_32k"]

    assert set(with_block) - set(without) == set(run.BLOCK_KEYS)
    assert all(with_block[key] == without[key] for key in without)
    # blocks of one pair are the i.i.d. resample itself
    assert with_block["ci_low_block"] == with_block["ci_low"]
    assert with_block["block_length"] == 1 and with_block["n_blocks"] == 2


def test_compute_tpp_and_hindi_and_length_carry_the_block_keys() -> None:
    arms = {
        "T0_o200k": _char_arm("T0_o200k", "T0"),
        "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
    }
    deployed = run.compute_tpp(
        [_one_corpus()],
        arms,  # type: ignore[arg-type]
        ["T1_bpe_raw_32k"],
        ["T0_o200k"],
        {"T1": ["slp1"]},
        n_bootstrap=10,
        seed=0,
        ci=0.95,
        block_length=2,
    )
    summary = deployed["corpus_a"]["T1_bpe_raw_32k"]["slp1"]["T0_o200k"]
    assert all(key in summary for key in run.BLOCK_KEYS)

    strata = run.compute_tpp_by_length(
        [_decomposition_corpus()],  # type: ignore[list-item]
        arms,  # type: ignore[arg-type]
        [("T1_bpe_raw_32k", run.SLP1, "T0_o200k")],
        [1, 3],
        n_bootstrap=10,
        seed=0,
        ci=0.95,
        block_length=2,
    )
    for bins in strata["corpus_a"]["T1_bpe_raw_32k/T0_o200k"].values():
        assert all(key in bins for key in run.BLOCK_KEYS)


def test_summarise_leaves_a_summary_untouched_without_a_block_interval() -> None:
    """Every leaf of a `results.json` written before the block bootstrap must be
    reproducible key-for-key by a run that does not configure one."""
    raw = {
        "value": 1.0,
        "n": 2,
        "unit": "tokens/proposition ratio",
        "per_pair": [1.0, 1.0],
        "n_undefined": 0,
        "source_tokens": 4,
        "pivot_tokens": 4,
        "ci_low": 1.0,
        "ci_high": 1.0,
        "n_bootstrap": 10,
        "seed": 0,
    }
    assert not [key for key in run.summarise(raw, 0.95) if key in run.BLOCK_KEYS]


# --- length strata: skipped arms --------------------------------------------------------


def test_select_length_pairs_skips_the_configured_arms_on_both_sides() -> None:
    arms = {
        "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
        "T1_bpe_raw_128k": _char_arm("T1_bpe_raw_128k", "T1"),
        "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
        "E1_bpe_128k": _char_arm("E1_bpe_128k", "E1"),
        "T0_o200k": _char_arm("T0_o200k", "T0"),
        "T7_byt5": _byte_arm("T7_byt5"),
    }
    pairs = run.select_length_pairs(
        arms,  # type: ignore[arg-type]
        ["T1_bpe_raw_32k", "T1_bpe_raw_128k", "T7_byt5"],
        ["T0_o200k"],
        [
            ("T1_bpe_raw_32k", "E1_bpe_32k"),
            ("T1_bpe_raw_128k", "E1_bpe_128k"),
            ("T7_byt5", "T7_byt5"),
        ],
        SCRIPT_VARIANTS,
        ["T1_bpe_raw_128k", "E1_bpe_128k", "T7_byt5"],
    )
    assert pairs == [
        ("T1_bpe_raw_32k", run.SLP1, "E1_bpe_32k"),
        ("T1_bpe_raw_32k", run.SLP1, "T0_o200k"),
    ]


def test_select_length_pairs_skips_nothing_by_default() -> None:
    arms = {
        "T1_bpe_raw_32k": _char_arm("T1_bpe_raw_32k", "T1"),
        "E1_bpe_32k": _char_arm("E1_bpe_32k", "E1"),
        "T0_o200k": _char_arm("T0_o200k", "T0"),
    }
    pairs = run.select_length_pairs(
        arms,  # type: ignore[arg-type]
        ["T1_bpe_raw_32k"],
        ["T0_o200k"],
        [("T1_bpe_raw_32k", "E1_bpe_32k")],
        SCRIPT_VARIANTS,
    )
    assert pairs == [
        ("T1_bpe_raw_32k", run.SLP1, "E1_bpe_32k"),
        ("T1_bpe_raw_32k", run.SLP1, "T0_o200k"),
    ]


# --- the controlled column, with both English controls and the byte reference -----------


def test_controlled_panel_groups_pairs_each_sanskrit_arm_with_both_controls() -> None:
    groups = run.controlled_panel_groups(
        [
            ["T1_bpe_raw_32k", "E1_bpe_32k"],
            ["T2_unigram_raw_32k", "E1_unigram_32k"],
            ["T1_bpe_raw_32k", "E1_bpe_32k_bm"],
            ["T7_byt5", "T7_byt5"],
        ]
    )
    assert groups == [
        ("T1_bpe_raw_32k", "E1_bpe_32k", "E1_bpe_32k_bm"),
        ("T2_unigram_raw_32k", "E1_unigram_32k", None),
    ]


def test_controlled_panel_groups_keeps_an_arm_with_only_a_byte_matched_control() -> None:
    assert run.controlled_panel_groups([["T1_bpe_raw_32k", "E1_bpe_32k_bm"]]) == [
        ("T1_bpe_raw_32k", None, "E1_bpe_32k_bm")
    ]


def _synthetic_results_with_both_controls() -> dict[str, object]:
    results = _synthetic_results_with_control()
    config = results["config"]
    assert isinstance(config, dict)
    config["controlled_pairs"] = [
        ["T1_bpe_raw_32k", "E1_bpe_32k"],
        ["T1_bpe_raw_32k", "E1_bpe_32k_bm"],
        ["T7_byt5", "T7_byt5"],
    ]
    sources = results["tokenizer_sources"]
    assert isinstance(sources, dict)
    sources["E1_bpe_32k_bm"] = {"source_id": "en_bm.json", "vocab_size": 32000, "family": "E1"}
    controlled = results["tpp_controlled"]
    assert isinstance(controlled, dict)
    for corpus_name, value in (("corpus_a", 0.9), ("corpus_b", 1.2)):
        controlled[corpus_name]["T1_bpe_raw_32k/E1_bpe_32k_bm"] = {
            "value": value,
            "ci_low": value - 0.05,
            "ci_high": value + 0.05,
            "n": 5,
            "unit": "x",
        }
        controlled[corpus_name]["T7_byt5/T7_byt5"] = {
            "value": 0.75,
            "ci_low": 0.75,
            "ci_high": 0.75,
            "n": 5,
            "unit": "x",
        }
    return results


def test_build_tpp_figure_draws_both_controls_at_one_x_position_with_a_byte_line() -> None:
    """Two markers per Sanskrit arm (filled = pair-matched, hollow = byte-matched) and a
    dotted reference line at the byte ratio, so the reviewer's objection is readable off
    the panel rather than only off the table."""
    import matplotlib.pyplot as plt

    figure = run._build_tpp_figure(_synthetic_results_with_both_controls())
    try:
        controlled_axes = figure.axes[1]
        # one x position, not two: both controls sit on the same Sanskrit arm
        assert [label.get_text() for label in controlled_axes.get_xticklabels()] == [
            "T1_bpe_raw_32k* / E1_bpe_32k"
        ]
        byte_lines = [
            line
            for line in controlled_axes.get_lines()
            if line.get_linestyle() == ":" and line.get_ydata()[0] == 0.75
        ]
        assert byte_lines, "the byte reference line is missing"
        legend = controlled_axes.get_legend()
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert run.CONTROLLED_LABEL in legend_labels
        assert run.CONTROLLED_BM_LABEL in legend_labels
        assert run.BYTE_REFERENCE_LABEL in legend_labels
    finally:
        plt.close(figure)
