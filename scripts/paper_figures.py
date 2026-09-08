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
* `figures/length.{pdf,png}` — the controlled ratio for each matched pair by sentence
  length, in two rows: bins cut on the English side's word count, then on the Sanskrit
  side's. Which side the bins are cut on biases the ratio in that side's direction, so
  neither row is read alone. Skipped, with a log line, if the snapshot has no
  `tpp_by_length` key.

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
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

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

#: The matched pairs this paper's figures draw, in reporting order. Named here for the
#: same reason `paper_tables.CONTROLLED_PAIRS` is: the snapshot carries pairs whose prose
#: is not written yet, and a figure must not draw a series the manuscript does not discuss.
CONTROLLED_PAIRS: tuple[str, ...] = (
    "T1_bpe_raw_32k/E1_bpe_32k",
    "T1_bpe_raw_64k/E1_bpe_64k",
    "T2_unigram_raw_32k/E1_unigram_32k",
    "T2_unigram_raw_64k/E1_unigram_64k",
)

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


def _axis_break(ax: Axes, centre: float, top: float, width: float, span: float) -> None:
    """Draw the two-stroke break glyph across the top of a clipped bar.

    Without it a truncated bar reads as its drawn height, which is the one way this
    figure could mislead: GPT-2's Sanskrit-over-English bar is drawn at the axis ceiling
    and its real value is several times that. `span` is the height of one stroke, set by
    the caller from the final axis range.
    """
    half = width / 2.0
    for offset in (-span * 0.9, span * 0.9):
        base = top - span * 4.2 + offset
        ax.add_patch(
            Polygon(
                [
                    (centre - half, base),
                    (centre, base + span),
                    (centre + half, base),
                    (centre + half, base + span * 1.1),
                    (centre, base + span * 2.1),
                    (centre - half, base + span * 1.1),
                ],
                closed=True,
                facecolor="white",
                edgecolor="white",
                linewidth=0.0,
                zorder=4,
            )
        )
        ax.plot(
            [centre - half, centre, centre + half],
            [base, base + span, base],
            color=INK,
            linewidth=0.7,
            zorder=5,
            clip_on=False,
        )


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
    #: Bar centres whose value runs past the axis, collected while the bars are drawn and
    #: given their break glyph once the axis range is final.
    broken: list[float] = []
    width = 0.36
    positions = list(range(len(arms)))
    colours = {"Sa / En": ACCENT, "Sa / Hi": NEUTRAL_DARK}
    for offset, (series, values) in zip((-0.5, 0.5), measured.items(), strict=True):
        centres = [position + offset * width for position in positions]
        heights = [min(value, top) for value in values]
        ax.bar(centres, heights, width=width * 0.92, color=colours[series], label=series,
               zorder=2)
        broken.extend(
            centre
            for centre, value in zip(centres, values, strict=True)
            if value > top
        )
        for centre, value, shown in zip(centres, values, heights, strict=True):
            clipped = value > top
            ax.annotate(
                f"{value:.2f}",
                (centre, shown),
                textcoords="offset points",
                # A clipped bar's number is set below the break glyph, not against the
                # ceiling, where the glyph would strike through it.
                xytext=(0, -22 if clipped else 3),
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
    for centre in broken:
        _axis_break(ax, centre, top, width * 0.92, top * 0.018)
    style_axes(ax, "ratio of token counts on the same content")
    ax.scatter([], [], marker="D", s=13, facecolor="white", edgecolor=NEUTRAL,
               linewidth=1.0, label="same ratio implied by fertility")
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
    configured = {f"{a}/{b}" for a, b in exp02["config"]["controlled_pairs"]}
    pairs: list[tuple[str, str]] = [
        (pair.split("/")[0], pair.split("/")[1])
        for pair in CONTROLLED_PAIRS
        if pair in configured
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


LENGTH_CORPORA = ("samayik_test", "itihasa_test")
#: One-line panel titles: the two-line forms of `CORPUS_LABELS` are too tall for a
#: four-panel figure sized to one column.
LENGTH_CORPUS_TITLES: dict[str, str] = {
    "samayik_test": "Sāmayik test (prose)",
    "itihasa_test": "Itihāsa test (verse)",
}
#: The two stratifications, as (results key, what the bins count). Both are drawn because
#: binning on one side biases the ratio in that side's direction; the pair brackets it.
LENGTH_STRATA: tuple[tuple[str, str], ...] = (
    ("tpp_by_length", "English words per sentence"),
    ("tpp_by_length_sa", "Sanskrit words per sentence"),
)
#: One marker shape per matched pair, so the four series separate in greyscale and in
#: print, where four steps of one hue do not. `^` and `v` are reserved for off-scale marks.
LENGTH_MARKERS: tuple[str, ...] = ("o", "s", "D", "P")
#: Headroom above and below a panel's own data, as a fraction of its span.
LENGTH_Y_PAD = 0.10
#: How far left of its triangle an off-scale number is set in the last bins, in bin
#: widths. A number set flush against the triangle there is struck through by the tall
#: error bar the neighbouring in-range point carries into the reserved rows.
LENGTH_LABEL_GAP = 0.42
#: Height of one reserved off-scale annotation row, as a fraction of a panel's data span.
#: Each off-scale arm gets a row of its own beyond the data band, so a triangle and its
#: number never land on another arm's markers.
LENGTH_OFF_SCALE_ROW = 0.085


def _length_ylim(
    series: list[tuple[list[float], list[float], list[float], list[bool]]],
) -> tuple[float, float]:
    """A panel's own y range: the CI bounds of its non-sparse bins, and 1.0.

    Sparse bins are excluded from the scale, not from the plot. Itihāsa's nine-pair
    shortest English bin sits at 8.9 with an interval reaching 21.5, and letting it set
    the scale flattens every other bin of the figure into the reference line. 1.0 stays
    inside the range because it is the threshold every panel is read against.
    """
    dense: list[float] = []
    every: list[float] = []
    for _values, lows, highs, sparse in series:
        for low, high, is_sparse in zip(lows, highs, sparse, strict=True):
            every.extend((low, high))
            if not is_sparse:
                dense.extend((low, high))
    bounds = dense or every
    if not bounds:  # pragma: no cover - a panel with no points at all
        return 0.0, 2.0
    low, high = min([*bounds, 1.0]), max([*bounds, 1.0])
    span = high - low
    pad = span * LENGTH_Y_PAD if span > 0 else max(abs(high) * 0.2, 0.1)
    return low - pad, high + pad


def _off_scale(
    ax: Axes,
    x: float,
    y: float,
    value: float,
    colour: str,
    *,
    above: bool,
    to_left: bool,
) -> None:
    """Mark an off-scale value in the panel's reserved margin, with its number beside it.

    `y` is the middle of the row reserved for this arm, which lies beyond the data band,
    so the annotation cannot collide with a plotted marker or with another arm's. The
    number is set beside the triangle rather than under it, and to its left with a bin
    width's clearance in the last bins, where a label to the right would run off the
    panel and one flush against the triangle would meet a neighbouring error bar.

    Mirrors `_draw_off_scale` in `experiments/02_tpp_parallel/run.py`: a dropped point and
    a bin the corpus has no sentences in would otherwise look identical.
    """
    ax.scatter(
        [x],
        [y],
        marker="^" if above else "v",
        s=16,
        edgecolors=colour,
        facecolors="none",
        linewidths=0.8,
        zorder=5,
    )
    ax.annotate(
        f"{value:.2f}",
        (x - LENGTH_LABEL_GAP if to_left else x, y),
        textcoords="offset points",
        xytext=(0.0 if to_left else 3.5, 0.0),
        ha="right" if to_left else "left",
        va="center",
        fontsize=5.0,
        color=colour,
        zorder=5,
    )


def _data_band(ax: Axes, bottom: float, top: float) -> Rectangle:
    """An invisible rectangle over a panel's data band, used to clip the data to it.

    The reserved off-scale rows sit outside this band. Without the clip a confidence
    interval that runs past the panel's scale would draw its whisker up through those
    rows and across an annotation, which is the one collision the reserved rows cannot
    prevent on their own.
    """
    left, right = ax.get_xlim()
    band = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(band)
    return band


def figure_length(exp02: dict[str, Any]) -> Figure | None:
    """The controlled ratio by sentence length, under both stratifications.

    Rows are the two stratifications (English-side bins, then Sanskrit-side bins) and
    columns the two primary corpora, prose before verse (CLAUDE.md §2.7). Only the matched
    pairs are drawn, for the same reason the tables carry only those: an off-the-shelf
    200k English vocabulary against a 32k Sanskrit one is deployed practice, never a
    controlled comparison, and on a figure about differences between length bands it would
    read as one. Returns None if the snapshot predates the analysis.
    """
    if "tpp_by_length" not in exp02:
        return None
    strata_keys = [(key, axis) for key, axis in LENGTH_STRATA if key in exp02]
    configured = {f"{a}/{b}" for a, b in exp02["config"]["controlled_pairs"]}
    pairs = [pair for pair in CONTROLLED_PAIRS if pair in configured]
    corpora = [corpus for corpus in LENGTH_CORPORA if corpus in exp02[strata_keys[0][0]]]

    # One column wide, and short enough that the figure can share a page with the
    # full-width length tables it is read against: taller than this and LaTeX defers
    # it two pages past its first reference.
    fig, axes = plt.subplots(
        len(strata_keys), len(corpora), figsize=(3.3, 2.8), squeeze=False
    )
    for row, (key, axis_label) in enumerate(strata_keys):
        for column, corpus in enumerate(corpora):
            ax = axes[row][column]
            entries = exp02[key][corpus]
            bins = list(entries[pairs[0]].keys())
            drawn: list[tuple[list[float], list[float], list[float], list[bool]]] = []
            for pair in pairs:
                nodes = [entries[pair].get(name) for name in bins]
                values = [float(node["value"]) for node in nodes if node is not None]
                lows = [float(node["ci_low"]) for node in nodes if node is not None]
                highs = [float(node["ci_high"]) for node in nodes if node is not None]
                sparse = [bool(node.get("sparse")) for node in nodes if node is not None]
                drawn.append((values, lows, highs, sparse))
            bottom, top = _length_ylim(drawn)
            # A bin where more than one arm runs off the panel gets one note instead of a
            # stack of triangles: four numbers stacked in a corner is the least legible
            # thing this figure can draw, and the values are in the length tables anyway.
            crowded = {
                position
                for position in range(len(bins))
                if sum(
                    1
                    for values, *_rest in drawn
                    if position < len(values) and not bottom <= values[position] <= top
                )
                > 1
            }
            # One reserved row per arm that runs off the panel, on the side it runs off,
            # so the triangles and their numbers sit beyond the data rather than on it.
            rows_above: dict[int, int] = {}
            rows_below: dict[int, int] = {}
            for index, (values, *_rest) in enumerate(drawn):
                for position, value in enumerate(values):
                    if position in crowded:
                        continue
                    if value > top:
                        rows_above.setdefault(index, len(rows_above))
                    elif value < bottom:
                        rows_below.setdefault(index, len(rows_below))
            row_height = (top - bottom) * LENGTH_OFF_SCALE_ROW
            ax.set_ylim(
                bottom - row_height * len(rows_below), top + row_height * len(rows_above)
            )
            ax.set_xlim(-0.6, len(bins) - 0.4)
            band = _data_band(ax, bottom, top)
            # Ticks are chosen over the data band and fixed, so that the reserved rows
            # do not coarsen the scale and no tick is drawn inside them.
            ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(bottom, top)
            ax.set_yticks([tick for tick in ticks if bottom <= tick <= top])

            for index, (pair, (values, lows, highs, sparse)) in enumerate(
                zip(pairs, drawn, strict=True)
            ):
                colour = ACCENT_RAMP[index % len(ACCENT_RAMP)]
                marker = LENGTH_MARKERS[index % len(LENGTH_MARKERS)]
                nudge = (index - (len(pairs) - 1) / 2) * 0.22
                line_x: list[float] = []
                line_y: list[float] = []
                for position, value in enumerate(values):
                    if position in crowded:
                        continue
                    if not bottom <= value <= top:
                        above = value > top
                        rank = rows_above[index] if above else rows_below[index]
                        edge = (
                            top + row_height * (rank + 0.5)
                            if above
                            else bottom - row_height * (rank + 0.5)
                        )
                        _off_scale(
                            ax,
                            position + nudge,
                            edge,
                            value,
                            colour,
                            above=above,
                            # A number set to the right of a triangle in the last bins
                            # would run off the panel, so those are set to its left.
                            to_left=position > len(bins) - 3,
                        )
                        continue
                    line_x.append(position)
                    line_y.append(value)
                    container = ax.errorbar(
                        [position],
                        [value],
                        yerr=_errbars(value, lows[position], highs[position]),
                        fmt=marker,
                        markersize=2.6,
                        markerfacecolor="none" if sparse[position] else colour,
                        color=colour,
                        ecolor=colour,
                        elinewidth=0.8,
                        capsize=1.2,
                        zorder=3,
                    )
                    for whisker in (*container.lines[1], *container.lines[2]):
                        whisker.set_clip_path(band)
                ax.plot(line_x, line_y, linewidth=0.9, color=colour, zorder=2,
                        marker=marker, markersize=2.6, markerfacecolor=colour,
                        label=_pair_label(pair) if (row, column) == (0, 0) else None)
            for position in sorted(crowded):
                _crowded_note(
                    ax,
                    bins[position],
                    int(entries[pairs[0]][bins[position]]["n_pairs"]),
                    left=_free_corner_is_left(drawn, bottom, top),
                )
            ax.axhline(1.0, color=NEUTRAL_LIGHT, linestyle="--", linewidth=0.8, zorder=1)
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(bins, fontsize=6.0, rotation=40, ha="right")
            ax.tick_params(axis="y", labelsize=6.0)
            if row == 0:
                ax.set_title(LENGTH_CORPUS_TITLES.get(corpus, corpus),
                             fontsize=7.0, color=SUBTLE_INK, pad=4)
            ax.set_xlabel(axis_label, fontsize=7.0, labelpad=2)
            if column == 0:
                ax.set_ylabel("TPP (Sa / En)", fontsize=7.0)
            style_axes(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.0, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.07), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    return fig


def _free_corner_is_left(
    series: list[tuple[list[float], list[float], list[float], list[bool]]],
    bottom: float,
    top: float,
) -> bool:
    """True when a panel's plotted points leave more room at its top-left than top-right.

    The panels of this figure are monotone within a row, so the corner above the low end
    of the curves is the one an annotation can occupy without landing on data.
    """
    halves: list[list[float]] = [[], []]
    for values, *_rest in series:
        for position, value in enumerate(values):
            if bottom <= value <= top:
                halves[0 if position < len(values) / 2 else 1].append(value)
    if not halves[0] or not halves[1]:  # pragma: no cover - every panel has both halves
        return True
    return sum(halves[0]) / len(halves[0]) < sum(halves[1]) / len(halves[1])


def _crowded_note(ax: Axes, name: str, n_pairs: int, left: bool) -> None:
    """Name a bin whose arms all run off the panel, instead of drawing four triangles.

    `left` puts the note in the corner with the most headroom, which the caller works out
    from where the panel's own curves sit; a note over the curves is no better than the
    stack of triangles it replaces.
    """
    label = name.replace("-", "\u2013")
    ax.text(
        0.03 if left else 0.97,
        0.985,
        f"{label}: n={n_pairs}, off scale\n(values in the length tables)",
        transform=ax.transAxes,
        fontsize=4.6,
        color=SUBTLE_INK,
        ha="left" if left else "right",
        va="top",
        linespacing=1.15,
        zorder=6,
    )


def _pair_label(pair: str) -> str:
    """`T1_bpe_raw_32k/E1_bpe_32k` -> `BPE 32k`, as the tables abbreviate it."""
    sanskrit = pair.split("/")[0]
    algorithm = "BPE" if "_bpe_" in sanskrit else "Unigram"
    return f"{algorithm} {sanskrit.rsplit('_', 1)[-1]}"


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
