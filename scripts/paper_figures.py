"""Render the figures of `paper/1a` from the tracked results snapshot.

Every figure is written from `results/01_baseline_penalty/results.json` and
`results/02_tpp_parallel/results.json`; no value is typed into this script.

* `figures/parity.{pdf,png}` — the RQ1 penalty. Grouped bars of the measured parity
  ratios (Sanskrit tokens over English or Hindi tokens, same tokenizer both sides, same
  FLORES-200 sentences), with the ratio that fertility would have implied drawn as a faint
  second marker on each bar. The gap between the two is the point: fertility divides by a
  word count that sandhi and compounding make small, so it overstates the penalty.
* `figures/denominator.{pdf,png}` — the RQ2 result. The same four Sanskrit arms, measured
  against two different English denominators: the deployed 200k-vocabulary tokenizer and
  the matched English control. Nothing about the Sanskrit side changes between the two
  series.
* `figures/length.{pdf,png}` — the controlled ratio for each matched pair, stratified by
  the number of words in the English side of the pair. Skipped, with a log line, if the
  snapshot has no `tpp_by_length` key.

The style mirrors `scripts/social_figures.py`, which is not imported: that module inserts
`src/` on `sys.path` and switches the matplotlib backend at import time, so it is not
importable without side effects. The palette, the font stack and the axis conventions are
copied from it deliberately, so both figure sets look like one project.

Usage::

    uv run python scripts/paper_figures.py
    uv run python scripts/paper_figures.py --results-dir results --out-dir paper/1a/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("paper_figures")

# --------------------------------------------------------------------------------------
# Style, mirrored from scripts/social_figures.py
# --------------------------------------------------------------------------------------

#: Sanskrit-native arms and the controlled comparison generally.
ACCENT = "#136F63"
#: Steps of the accent, for the four matched pairs on one panel.
ACCENT_RAMP = ("#0C4A42", "#136F63", "#3E9A8B", "#8CC6BC")
#: Baselines, off-the-shelf arms, and anything that is context rather than result.
NEUTRAL = "#8A9199"
NEUTRAL_DARK = "#4A5158"
NEUTRAL_LIGHT = "#C3C8CD"
#: Measurement artefacts and the metric that would have misled us.
MUTED_RED = "#B0413E"
INK = "#1A1A1A"
SUBTLE_INK = "#5A6068"
GRID = "#DEE2E5"

SANS_STACK = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
MONO_STACK = ("Menlo", "DejaVu Sans Mono", "Consolas", "Courier New")

DPI = 200


def _installed_font(candidates: Sequence[str]) -> str | None:
    """Return the first installed family from `candidates`, else None."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def mono_family() -> str:
    """A monospace family for arm names, which read best fixed-width."""
    return _installed_font(MONO_STACK) or "monospace"


def apply_style() -> None:
    """The one style both figures share."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": list(SANS_STACK),
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": NEUTRAL_LIGHT,
            "xtick.color": SUBTLE_INK,
            "ytick.color": SUBTLE_INK,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def style_axes(ax: Axes, ylabel: str | None = None) -> None:
    """Horizontal grid only, behind the data, and no chartjunk."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(length=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9)


def save(fig: Figure, out_dir: Path, stem: str) -> list[Path]:
    """Write both formats, as CLAUDE.md §8 requires."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------------------
# Snapshot access
# --------------------------------------------------------------------------------------

T0_ARMS = ("T0_o200k", "T0_llama4", "T0_gemma3", "T0_gpt2")
#: Arms whose vocabulary is the size current deployments actually ship.
LARGE_VOCAB_T0 = ("T0_o200k", "T0_llama4", "T0_gemma3")

CORPUS_LABELS: dict[str, str] = {
    "samayik_test": "Sāmayik test\n(prose, primary)",
    "samayik_test_ood": "Sāmayik test_ood\n(prose, out of domain)",
    "itihasa_test": "Itihāsa test\n(verse)",
    "flores_devtest": "FLORES devtest\n(Wikipedia)",
}


def load_results(results_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the two snapshot files. Read-only: nothing here writes to `results/`."""
    exp01_path = results_dir / "01_baseline_penalty" / "results.json"
    exp02_path = results_dir / "02_tpp_parallel" / "results.json"
    for path in (exp01_path, exp02_path):
        if not path.exists():
            raise FileNotFoundError(f"missing snapshot file {path}")
    exp01: dict[str, Any] = json.loads(exp01_path.read_text())
    exp02: dict[str, Any] = json.loads(exp02_path.read_text())
    return exp01, exp02


def _fertility(exp01: dict[str, Any], arm: str, language: str) -> float:
    return float(exp01["metrics"][arm][language]["original"]["fertility"]["value"])


def _errbars(value: float, low: float, high: float) -> tuple[list[float], list[float]]:
    """Asymmetric error offsets, clamped so matplotlib never sees a negative."""
    return ([max(value - low, 0.0)], [max(high - value, 0.0)])


# --------------------------------------------------------------------------------------
# Figure 1: the RQ1 penalty, and what fertility would have said instead
# --------------------------------------------------------------------------------------


def figure_parity(exp01: dict[str, Any]) -> Figure:
    """Measured parity per deployed arm, with the fertility-derived ratio beside it."""
    arms = [arm for arm in T0_ARMS if arm in exp01["parity"]]
    measured = {
        "Sa / En": [float(exp01["parity"][arm]["eng_Latn"]["value"]) for arm in arms],
        "Sa / Hi": [float(exp01["parity"][arm]["hin_Deva"]["value"]) for arm in arms],
    }
    implied = {
        "Sa / En": [
            _fertility(exp01, arm, "san_Deva") / _fertility(exp01, arm, "eng_Latn")
            for arm in arms
        ],
        "Sa / Hi": [
            _fertility(exp01, arm, "san_Deva") / _fertility(exp01, arm, "hin_Deva")
            for arm in arms
        ],
    }

    # The axis is scaled from the arms whose vocabulary size is current practice. GPT-2's
    # 50k vocabulary falls back to near byte-level segmentation on Devanagari and would
    # otherwise squash every other bar; its bars are clipped and annotated instead.
    in_scale = [arm for arm in arms if arm in LARGE_VOCAB_T0]
    ceiling = max(
        max(implied[series][arms.index(arm)] for arm in in_scale) for series in implied
    )
    top = ceiling * 1.22

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    width = 0.36
    positions = list(range(len(arms)))
    colours = {"Sa / En": ACCENT, "Sa / Hi": NEUTRAL_DARK}
    for offset, (series, values) in zip((-0.5, 0.5), measured.items(), strict=True):
        centres = [position + offset * width for position in positions]
        heights = [min(value, top) for value in values]
        ax.bar(centres, heights, width=width * 0.92, color=colours[series], label=series,
               zorder=2)
        for centre, value, shown in zip(centres, values, heights, strict=True):
            clipped = value > top
            ax.annotate(
                f"{value:.2f}",
                (centre, shown),
                textcoords="offset points",
                xytext=(0, -4 if clipped else 3),
                ha="center",
                va="top" if clipped else "baseline",
                fontsize=7.5,
                color="white" if clipped else SUBTLE_INK,
            )
        for centre, value in zip(centres, implied[series], strict=True):
            clipped = value > top
            ax.scatter(
                [centre],
                [min(value, top)],
                marker="^" if clipped else "D",
                s=15 if clipped else 13,
                facecolor="white",
                edgecolor=MUTED_RED if clipped else NEUTRAL,
                linewidth=1.0,
                zorder=3,
            )
            if clipped:
                ax.annotate(
                    f"{value:.2f}",
                    (centre, top),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color=MUTED_RED,
                )

    ax.axhline(1.0, color=NEUTRAL_LIGHT, linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate("parity", (-0.62, 1.0), textcoords="offset points", xytext=(0, 3),
                fontsize=7.5, color=SUBTLE_INK)
    ax.set_xticks(positions)
    ax.set_xticklabels([arm.replace("T0_", "") for arm in arms],
                       fontsize=8.5, family=mono_family())
    ax.set_xlim(-0.65, len(arms) - 0.35)
    ax.set_ylim(0, top)
    style_axes(ax, "tokens per unit of content")
    ax.scatter([], [], marker="D", s=13, facecolor="white", edgecolor=NEUTRAL,
               linewidth=1.0, label="ratio implied by fertility")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.55, -0.10))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 2: the same Sanskrit arm, two English denominators
# --------------------------------------------------------------------------------------


def figure_denominator(exp02: dict[str, Any]) -> Figure:
    """Every matched pair, against the deployed English pivot and against its control."""
    pairs: list[tuple[str, str]] = [
        (str(pair[0]), str(pair[1])) for pair in exp02["config"]["controlled_pairs"]
    ]
    corpora = list(exp02["tpp_controlled"].keys())
    pivot = str(exp02["config"]["english_pivots"][0])

    fig, axes = plt.subplots(1, len(corpora), figsize=(7.1, 3.1), sharey=True)
    for ax, corpus in zip(axes, corpora, strict=True):
        for index, (sa_arm, en_arm) in enumerate(pairs):
            deployed = exp02["tpp"][corpus][sa_arm]["slp1"][pivot]
            controlled = exp02["tpp_controlled"][corpus][f"{sa_arm}/{en_arm}"]
            for offset, node, colour, marker in (
                (-0.17, deployed, NEUTRAL, "o"),
                (0.17, controlled, ACCENT, "s"),
            ):
                value = float(node["value"])
                ax.errorbar(
                    [index + offset],
                    [value],
                    yerr=_errbars(value, float(node["ci_low"]), float(node["ci_high"])),
                    fmt=marker,
                    markersize=4.0,
                    color=colour,
                    ecolor=colour,
                    elinewidth=1.1,
                    capsize=2.0,
                    zorder=3,
                )
        ax.axhline(1.0, color=NEUTRAL_LIGHT, linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(
            [sa_arm.replace("_raw", "").replace("T1_", "").replace("T2_", "")
             for sa_arm, _ in pairs],
            fontsize=7.0, rotation=40, ha="right", family=mono_family(),
        )
        ax.set_xlim(-0.6, len(pairs) - 0.4)
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=8, color=SUBTLE_INK, pad=6)
        style_axes(ax)
    axes[0].set_ylabel("tokens per proposition (Sa / En)", fontsize=9)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NEUTRAL, markersize=4.0,
                   label=f"vs deployed English ({pivot.replace('T0_', '')})"),
        Line2D([], [], marker="s", linestyle="none", color=ACCENT, markersize=4.0,
                   label="vs matched English control (E1)"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Figure 3: length-stratified TPP, when the snapshot carries it
# --------------------------------------------------------------------------------------


def figure_length(exp02: dict[str, Any]) -> Figure | None:
    """Controlled TPP by English sentence length, or None if not measured yet.

    Restricted to the matched pairs, for the same reason as the table: a length breakdown
    of the uncontrolled deployed-practice ratios answers a question this paper does not
    ask.
    """
    if "tpp_by_length" not in exp02:
        return None
    strata: dict[str, Any] = exp02["tpp_by_length"]
    pairs = [f"{pair[0]}/{pair[1]}" for pair in exp02["config"]["controlled_pairs"]]
    corpora = list(strata.keys())
    bins = list(strata[corpora[0]][pairs[0]].keys())

    fig, axes = plt.subplots(1, len(corpora), figsize=(7.1, 3.1), sharey=True)
    axes_list = list(axes) if len(corpora) > 1 else [axes]
    for ax, corpus in zip(axes_list, corpora, strict=True):
        for index, pair in enumerate(pairs):
            nodes = [strata[corpus][pair].get(name) for name in bins]
            xs = [position for position, node in enumerate(nodes) if node is not None]
            values = [float(nodes[position]["value"]) for position in xs]
            lows = [float(nodes[position]["ci_low"]) for position in xs]
            highs = [float(nodes[position]["ci_high"]) for position in xs]
            ax.errorbar(
                xs,
                values,
                yerr=[
                    [max(v - lo, 0.0) for v, lo in zip(values, lows, strict=True)],
                    [max(hi - v, 0.0) for v, hi in zip(values, highs, strict=True)],
                ],
                fmt="o-",
                markersize=3.0,
                linewidth=1.0,
                color=ACCENT_RAMP[index % len(ACCENT_RAMP)],
                ecolor=ACCENT_RAMP[index % len(ACCENT_RAMP)],
                elinewidth=0.9,
                capsize=1.5,
                label=pair.split("/")[0] if corpus == corpora[0] else None,
            )
        ax.axhline(1.0, color=NEUTRAL_LIGHT, linestyle="--", linewidth=0.9)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, fontsize=7.0, rotation=40, ha="right")
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=8, color=SUBTLE_INK, pad=6)
        style_axes(ax)
    axes_list[0].set_ylabel("tokens per proposition (Sa / En)", fontsize=9)
    fig.legend(fontsize=7, loc="lower center", ncol=len(pairs), bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="the tracked results snapshot to read (default: results/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "paper" / "1a" / "figures",
        help="where the figures go (default: paper/1a/figures)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    exp01, exp02 = load_results(args.results_dir)
    out_dir: Path = args.out_dir
    apply_style()

    for path in save(figure_parity(exp01), out_dir, "parity"):
        LOGGER.info("wrote %s", path)
    for path in save(figure_denominator(exp02), out_dir, "denominator"):
        LOGGER.info("wrote %s", path)

    length = figure_length(exp02)
    if length is None:  # pragma: no cover - only for a snapshot predating the analysis
        LOGGER.info("skipping figures/length: the snapshot has no `tpp_by_length` key yet")
    else:
        for path in save(length, out_dir, "length"):
            LOGGER.info("wrote %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
