"""Tests for `scripts/social_figures.py`.

The renderer is a script, not a package module, so it is loaded by path. Every figure is
rendered against a tiny synthetic `Sources`, which is what keeps the test fast and
independent of `outputs/`; a further test renders the set against the real results if
they are present, and skips otherwise.

The layout guard is the point of most of this file: after each render, every text artist
the figure actually paints is measured with the real renderer, and the test fails if two
of them overlap or if any of them falls off the canvas.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "social_figures.py"

#: Pixels two text boxes may overlap by before the guard calls it a collision. Glyph
#: bounding boxes carry a little side bearing, so a hairline touch is not ink on ink.
OVERLAP_TOLERANCE_PX = 2.0


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("social_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load()

ALL_FIGURES = [spec.name for spec in sf.FIGURES]


def _interval(value: float, half: float = 0.01) -> dict[str, float]:
    return {"value": value, "ci_low": value - half, "ci_high": value + half}


def _delta(value: float, half: float = 0.004) -> dict[str, float]:
    return {"delta": value, "ci_low": value - half, "ci_high": value + half}


def synthetic_sources() -> Any:
    """A `Sources` holding every field any figure reads, with plausible values.

    The numbers are made up but the shapes are the real ones, so a figure that reaches
    for a field the experiments do not publish fails here rather than at render time.
    """

    def fertility(value: float) -> dict[str, Any]:
        return {"original": {"fertility": {"value": value}}}

    arms = {"T0_gpt2": 12.0, "T0_o200k": 4.0, "T0_llama4": 4.4, "T0_gemma3": 3.6}
    exp01 = {
        "metrics": {
            arm: {
                "san_Deva": fertility(value),
                "hin_Deva": fertility(value / 1.6),
                "eng_Latn": fertility(1.5),
            }
            for arm, value in arms.items()
        },
        "parity": {
            arm: {"eng_Latn": {"value": value / 1.6}, "hin_Deva": {"value": 1.06 + index * 0.1}}
            for index, (arm, value) in enumerate(arms.items())
        },
        "tokenizer_sources": {
            "T0_gpt2": {"vocab_size": 50257},
            "T0_o200k": {"vocab_size": 200019},
            "T3_sarvam": {"vocab_size": 68096},
        },
    }

    corpora = ("samayik_test", "samayik_test_ood", "itihasa_test", "flores_devtest")
    exp02 = {
        "tpp": {"samayik_test": {"T1_bpe_raw_64k": {"slp1": {"T0_o200k": _interval(0.887)}}}},
        "tpp_controlled": {
            corpus: {
                key: _interval(0.65 if corpus == "itihasa_test" else 1.05 + index * 0.03)
                for index, (key, _) in enumerate(sf.MATCHED_PAIRS)
            }
            for corpus in corpora
        },
        "tokenizer_sources": {"T3_brahmic131k": {"vocab_size": 131072}},
    }
    exp03 = {
        "tpp_delta": {
            corpus: {
                key: _delta(0.012 if corpus == "itihasa_test" else -0.02 - index * 0.015)
                for index, (key, _) in enumerate(sf.SPLIT_PAIRS)
            }
            for corpus in corpora
        },
        "tpp": {
            "samayik_test": {
                "T4_bpe_split_64k": {"reconciled": {"E1_bpe_64k": _interval(1.030)}},
                "T1_bpe_raw_64k": {"raw_slp1": {"E1_bpe_64k": _interval(1.035)}},
            }
        },
    }
    pairs = (
        ("T5_morphbpe_rawseg_64k_dcs/T1_bpe_raw_64k_dcs", 0.002, 0.115),
        ("T5_morphbpe_raw_64k_dcs/T1_bpe_raw_64k_dcs", 0.084, 0.041),
        ("T6_morphbpe_split_64k_dcs/T4_bpe_split_64k_oracle_dcs", 0.096, 0.305),
    )
    exp04 = {
        "tpp_delta": {
            "samayik_test": {
                **{key: _delta(tpp) for key, tpp, _ in pairs},
                "T4_bpe_split_64k_oracle_dcs/T1_bpe_raw_64k_dcs": _delta(-0.020),
            }
        },
        "morphscore_delta": {
            key: {"human_verified": {"exact": _delta(morph, 0.006)}} for key, _, morph in pairs
        },
    }
    dcs_manifest = {
        "n_dropped_shingle_per_source": {
            "itihasa_test": 20608,
            "itihasa_dev": 11458,
            "dcs_heldout": 3576,
            "samayik_dev": 5,
            "samayik_test_ood": 2,
            "samayik_test": 0,
            "flores_devtest": 0,
        }
    }
    split_records = [
        {
            "index": sf.EXAMPLE_INDEX,
            "raw_deva": "अनेन वयं पाठस्यान्तमागतवन्तः ।",
            "raw_slp1": "anena vayaM pAWasyAntamAgatavantaH .",
            "output": "anena vayam pAWasya antam AgatavantaH .",
            "output_model": "anena vayam pAWasya antam AgatavantaH /",
        }
    ]
    return sf.Sources(
        exp01=exp01,
        exp02=exp02,
        exp03=exp03,
        exp04=exp04,
        dcs_manifest=dcs_manifest,
        split_records=split_records,
    )


def _draw(name: str, square: bool) -> tuple[Any, dict[str, Any]]:
    """Render one figure exactly as `render` does, and hand back the live figure."""
    sf.apply_style()
    spec = next(item for item in sf.FIGURES if item.name == name)
    figure = sf.plt.figure(figsize=sf.SQUARE_SIZE if square else sf.WIDE_SIZE)
    plotted = spec.draw(figure, square, synthetic_sources())
    sf.draw_footer(figure, square)
    return figure, plotted


@pytest.mark.parametrize("name", ALL_FIGURES)
@pytest.mark.parametrize("square", [False, True])
def test_render_on_synthetic_results(tmp_path: Path, name: str, square: bool) -> None:
    """Each figure draws from the synthetic dict and reports the numbers it plotted."""
    figure, plotted = _draw(name, square)
    try:
        out = tmp_path / f"{name}{'_sq' if square else ''}.png"
        figure.savefig(out, dpi=50)
    finally:
        sf.plt.close(figure)
    assert out.exists() and out.stat().st_size > 0
    assert plotted, "every figure must report the values it plotted"


@pytest.mark.parametrize("name", ALL_FIGURES)
@pytest.mark.parametrize("square", [False, True])
def test_no_text_overlaps_and_nothing_is_clipped(tmp_path: Path, name: str, square: bool) -> None:
    """The layout guard: no two text artists collide, and none leaves the canvas.

    The figure is rendered to a temp dir at the real dpi first, so the extents measured
    here are the extents of the PNG a reader would open, not of an undrawn figure.
    """
    figure, _ = _draw(name, square)
    try:
        figure.savefig(tmp_path / f"{name}{'_sq' if square else ''}.png", dpi=sf.DPI)
        clashes = sf.text_overlaps(figure, OVERLAP_TOLERANCE_PX)
        outside = sf.outside_canvas(figure, OVERLAP_TOLERANCE_PX)
    finally:
        sf.plt.close(figure)
    assert not clashes, f"overlapping text in {name}{'_sq' if square else ''}: {clashes}"
    assert not outside, f"text outside the canvas in {name}{'_sq' if square else ''}: {outside}"


@pytest.mark.parametrize("name", ALL_FIGURES)
@pytest.mark.parametrize("square", [False, True])
def test_titles_stay_within_three_lines(name: str, square: bool) -> None:
    """A headline that needs a fourth line has to shrink instead."""
    figure, _ = _draw(name, square)
    try:
        lines = sf.title_line_count(figure)
    finally:
        sf.plt.close(figure)
    assert 1 <= lines <= 3, f"{name}{'_sq' if square else ''} has {lines} title lines"


def test_wrap_measured_never_exceeds_the_width_it_was_given() -> None:
    """Every wrapped line fits, measured by the same renderer that draws it."""
    sf.apply_style()
    figure = sf.plt.figure(figsize=sf.WIDE_SIZE)
    try:
        text = "Constraining merges on gold segment boundaries raises alignment at no cost"
        lines = sf.wrap_measured(figure, text, 2.5, 14.0)
        assert len(lines) > 1
        assert " ".join(lines) == text
        for line in lines:
            assert sf.text_inches(figure, line, 14.0) <= 2.5
    finally:
        sf.plt.close(figure)


def test_overlap_guard_catches_a_deliberate_collision() -> None:
    """The guard is worth trusting only if it fails on text that really does collide."""
    sf.apply_style()
    figure = sf.plt.figure(figsize=sf.WIDE_SIZE)
    try:
        figure.text(0.1, 0.5, "overlapping", fontsize=20)
        figure.text(0.1, 0.5, "overlapping", fontsize=20)
        assert sf.text_overlaps(figure, OVERLAP_TOLERANCE_PX)
        figure.text(-0.4, 0.5, "off the canvas", fontsize=20)
        assert sf.outside_canvas(figure, OVERLAP_TOLERANCE_PX)
    finally:
        sf.plt.close(figure)


def test_language_tax_reads_the_fertility_it_was_given() -> None:
    """The plotted numbers are the source's numbers, not anything the script invents."""
    figure, plotted = _draw("01_language_tax", False)
    sf.plt.close(figure)
    assert plotted["fertility_original_script"]["T0_gpt2"]["san_Deva"] == 12.0
    assert plotted["fertility_original_script"]["T0_o200k"]["eng_Latn"] == 1.5


def test_group_split_units_groups_a_sandhi_split_word() -> None:
    """One raw word covered by three split units is one group of three."""
    groups = sf.group_split_units(
        "anena vayaM pAWasyAntamAgatavantaH .",
        "anena vayam pAWasya antam AgatavantaH .",
    )
    assert [len(g) for g in groups] == [1, 1, 3, 1]
    assert groups[2] == ["pAWasya", "antam", "AgatavantaH"]


def test_every_figure_declares_a_claim_and_its_sources() -> None:
    names = [spec.name for spec in sf.FIGURES]
    assert len(names) == len(set(names))
    for spec in sf.FIGURES:
        assert spec.claim.strip()
        assert spec.sources and all(source.strip() for source in spec.sources)


@pytest.mark.skipif(
    not (REPO_ROOT / "outputs" / "01_baseline_penalty" / "results.json").exists(),
    reason="outputs/ is absent; the real-data render cannot run",
)
def test_render_every_figure_on_the_real_results(tmp_path: Path) -> None:
    """With `outputs/` present, every figure renders and the manifest records its values."""
    sf.apply_style()
    src = sf.load_sources(REPO_ROOT)
    plotted = {spec.name: sf.render(spec, src, tmp_path) for spec in sf.FIGURES}
    for spec in sf.FIGURES:
        for suffix in ("", "_sq"):
            path = tmp_path / f"{spec.name}{suffix}.png"
            assert path.exists()
            assert path.stat().st_size <= sf.SIZE_BUDGET_KB * 1024
    sf.write_manifest(tmp_path, REPO_ROOT, src, plotted)
    sf.write_readme(tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "README.md").exists()


@pytest.mark.skipif(
    not (REPO_ROOT / "outputs" / "01_baseline_penalty" / "results.json").exists(),
    reason="outputs/ is absent; the real-data render cannot run",
)
@pytest.mark.parametrize("name", ALL_FIGURES)
def test_real_results_render_without_text_collisions(name: str) -> None:
    """The same guard, against the numbers the committed PNGs were rendered from."""
    sf.apply_style()
    src = sf.load_sources(REPO_ROOT)
    spec = next(item for item in sf.FIGURES if item.name == name)
    for suffix, size in (("", sf.WIDE_SIZE), ("_sq", sf.SQUARE_SIZE)):
        figure = sf.plt.figure(figsize=size)
        try:
            spec.draw(figure, suffix == "_sq", src)
            sf.draw_footer(figure, suffix == "_sq")
            clashes = sf.text_overlaps(figure, OVERLAP_TOLERANCE_PX)
            outside = sf.outside_canvas(figure, OVERLAP_TOLERANCE_PX)
        finally:
            sf.plt.close(figure)
        assert not clashes, f"overlapping text in {name}{suffix}: {clashes}"
        assert not outside, f"text outside the canvas in {name}{suffix}: {outside}"
