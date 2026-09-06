"""Tests for `scripts/social_figures.py`.

The renderer is a script, not a package module, so it is loaded by path. Two figures are
rendered against a tiny synthetic `Sources`, which is what keeps the test fast and
independent of `outputs/`; a third test renders one figure against the real results if
they are present, and skips otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "social_figures.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("social_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sf = _load()


def _interval(value: float, half: float = 0.01) -> dict[str, float]:
    return {"value": value, "ci_low": value - half, "ci_high": value + half}


def synthetic_sources() -> object:
    """A `Sources` holding only the fields the two tested figures read."""

    def fertility(value: float) -> dict[str, object]:
        return {"original": {"fertility": {"value": value}}}

    exp01 = {
        "metrics": {
            "T0_gpt2": {
                "san_Deva": fertility(12.0),
                "hin_Deva": fertility(7.0),
                "eng_Latn": fertility(1.5),
            },
            "T0_o200k": {
                "san_Deva": fertility(4.0),
                "hin_Deva": fertility(2.0),
                "eng_Latn": fertility(1.4),
            },
        }
    }
    exp02 = {
        "tpp": {
            "samayik_test": {
                "T1_bpe_raw_64k": {"slp1": {"T0_o200k": _interval(0.9)}},
            }
        },
        "tpp_controlled": {"samayik_test": {"T1_bpe_raw_64k/E1_bpe_64k": _interval(1.05)}},
    }
    return sf.Sources(exp01=exp01, exp02=exp02)


@pytest.mark.parametrize("name", ["01_language_tax", "02_flip_vs_control"])
@pytest.mark.parametrize("square", [False, True])
def test_render_on_synthetic_results(tmp_path: Path, name: str, square: bool) -> None:
    """Each figure draws from the synthetic dict and reports the numbers it plotted."""
    sf.apply_style()
    spec = next(item for item in sf.FIGURES if item.name == name)
    figure = sf.plt.figure(figsize=sf.SQUARE_SIZE if square else sf.WIDE_SIZE)
    try:
        plotted = spec.draw(figure, square, synthetic_sources())
        sf.draw_footer(figure)
        out = tmp_path / f"{name}.png"
        figure.savefig(out, dpi=50)
    finally:
        sf.plt.close(figure)
    assert out.exists() and out.stat().st_size > 0
    assert plotted, "every figure must report the values it plotted"


def test_language_tax_reads_the_fertility_it_was_given() -> None:
    """The plotted numbers are the source's numbers, not anything the script invents."""
    sf.apply_style()
    spec = next(item for item in sf.FIGURES if item.name == "01_language_tax")
    figure = sf.plt.figure(figsize=sf.WIDE_SIZE)
    try:
        plotted = spec.draw(figure, False, synthetic_sources())
    finally:
        sf.plt.close(figure)
    assert plotted["fertility_original_script"]["T0_gpt2"]["san_Deva"] == 12.0
    assert plotted["fertility_original_script"]["T0_o200k"]["eng_Latn"] == 1.4


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
