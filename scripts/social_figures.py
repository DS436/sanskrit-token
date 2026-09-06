"""Render the LinkedIn-ready figure set for the Sanskrit tokenization project.

Every number plotted here is read out of a committed results file at render time; no
value is typed into this script except (a) the two figures that tell the story of a
*withdrawn* number, where the withdrawn value and the review's bound come from
`docs/decisions.md` and are labelled as such on the figure and in `manifest.json`, and
(b) the descriptive arm ladder, whose off-the-shelf vocabulary sizes are still read from
the experiments' `tokenizer_sources`.

Framing rules this module obeys (CLAUDE.md §1, §2):

* Fertility (tokens per whitespace word) is drawn, but only as the metric that would have
  misled us; the headline figures are tokens-per-proposition against a *matched* English
  control. `01_fertility_vs_parity.png` exists precisely to say so.
* "Fewer words" is never presented as "fewer tokens".
* No claim about Sanskrit's historical use in computing appears anywhere.

Usage::

    uv run python scripts/social_figures.py --out docs/outreach/media

Writes, for every figure, a 1600x900 PNG at dpi 200 and a 1080x1080 `_sq` variant, plus
`README.md` and `manifest.json` recording the exact numbers plotted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import-path plumbing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sanskrit_tok.encoding import from_slp1  # noqa: E402
from sanskrit_tok.provenance import git_commit, git_dirty  # noqa: E402

LOGGER = logging.getLogger("social_figures")

# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------

#: Sanskrit-native / proposed arms, and the controlled comparison generally.
ACCENT = "#136F63"
#: Steps of the accent, for the four matched pairs on one panel.
ACCENT_RAMP = ("#0C4A42", "#136F63", "#3E9A8B", "#8CC6BC")
#: Baselines, off-the-shelf arms, and anything that is context rather than result.
NEUTRAL = "#8A9199"
NEUTRAL_DARK = "#4A5158"
NEUTRAL_LIGHT = "#C3C8CD"
#: Withdrawn results and measurement artefacts.
MUTED_RED = "#B0413E"
MUTED_RED_LIGHT = "#D99693"
INK = "#1A1A1A"
SUBTLE_INK = "#5A6068"
GRID = "#DEE2E5"
HIGHLIGHT_FILL = "#DCEDE9"

FOOTER = "github.com/DS436/sanskrit-token · numbers from results.json"

WIDE_SIZE = (8.0, 4.5)  # 1600 x 900 at dpi 200
SQUARE_SIZE = (5.4, 5.4)  # 1080 x 1080 at dpi 200
DPI = 200

#: Point sizes, per aspect ratio. The square canvas is 1080 px wide against the wide
#: canvas's 1600, so every size on it is smaller in points as well as in inches.
TITLE_PT = 26.0
TITLE_SQ_PT = 20.0
TITLE_MIN_PT = 17.0
TITLE_SQ_MIN_PT = 13.5
SUBTITLE_PT = 15.5
SUBTITLE_SQ_PT = 12.5
AXIS_PT = 15.0
AXIS_SQ_PT = 12.0
TICK_PT = 13.5
TICK_SQ_PT = 10.5
LEGEND_PT = 12.5
LEGEND_SQ_PT = 10.0
VALUE_PT = 11.5
VALUE_SQ_PT = 9.5
NOTE_PT = 11.5
NOTE_SQ_PT = 9.5
FOOTER_PT = 10.0
FOOTER_SQ_PT = 9.0

#: The most title lines a figure may carry; the title shrinks until it fits.
MAX_TITLE_LINES = 2
MAX_TITLE_LINES_SQ = 3

MARGIN_X = 0.045
RIGHT_EDGE = 0.975

SANS_STACK = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
MONO_STACK = ("Menlo", "DejaVu Sans Mono", "Consolas", "Courier New")
DEVANAGARI_CANDIDATES = (
    "Kohinoor Devanagari",
    "Devanagari Sangam MN",
    "Noto Sans Devanagari",
    "Nirmala UI",
    "ITF Devanagari",
    "Devanagari MT",
)


def _installed_font(candidates: Sequence[str]) -> str | None:
    """Return the first installed font family from `candidates`, else None."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def devanagari_family() -> str:
    """A family that can render Devanagari, else the sans fallback with a warning."""
    found = _installed_font(DEVANAGARI_CANDIDATES)
    if found is None:  # pragma: no cover - depends on the host's fonts
        LOGGER.warning(
            "no Devanagari font found among %s; Devanagari rows will render as boxes",
            ", ".join(DEVANAGARI_CANDIDATES),
        )
        return SANS_STACK[-1]
    return found


def mono_family() -> str:
    """A monospace family for SLP1, which is ASCII and reads best fixed-width."""
    return _installed_font(MONO_STACK) or "monospace"


def apply_style() -> None:
    """Set the one style every figure in this set shares."""
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


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """Every committed artefact the figures read, plus each file's sha256."""

    exp01: dict[str, Any] = field(default_factory=dict)
    exp02: dict[str, Any] = field(default_factory=dict)
    exp03: dict[str, Any] = field(default_factory=dict)
    exp04: dict[str, Any] = field(default_factory=dict)
    dcs_manifest: dict[str, Any] = field(default_factory=dict)
    split_records: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


RELATIVE_SOURCES: dict[str, str] = {
    "exp01": "outputs/01_baseline_penalty/results.json",
    "exp02": "outputs/02_tpp_parallel/results.json",
    "exp03": "outputs/03_sandhi_split/results.json",
    "exp04": "outputs/04_morph_constrained/results.json",
    "dcs_manifest": "data/processed/dcs/manifest.json",
    "split_records": "data/processed/split/samayik_test.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(root: Path) -> Sources:
    """Load every committed artefact the figures read, recording each file's sha256."""
    payload: dict[str, Any] = {}
    files: dict[str, str] = {}
    for key, relative in RELATIVE_SOURCES.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"required source {relative} is missing; run the experiment that writes it"
            )
        files[relative] = _sha256(path)
        if path.suffix == ".jsonl":
            payload[key] = [json.loads(line) for line in path.read_text().splitlines() if line]
        else:
            payload[key] = json.loads(path.read_text())
    return Sources(files=files, **payload)


def dig(obj: Any, *path: str) -> Any:
    """Walk a nested mapping, raising a KeyError that names the full path on a miss."""
    cursor = obj
    for step in path:
        if not isinstance(cursor, dict) or step not in cursor:
            raise KeyError(f"missing {'.'.join(path)} (failed at {step!r})")
        cursor = cursor[step]
    return cursor


def num(obj: Any, *path: str) -> float:
    """`dig`, coerced to float."""
    return float(dig(obj, *path))


@dataclass(frozen=True)
class Interval:
    """A plotted value with its bootstrap interval, as stored in every results.json."""

    value: float
    low: float
    high: float

    @property
    def err(self) -> tuple[float, float]:
        """Asymmetric error offsets, clamped so matplotlib never sees a negative."""
        return (max(self.value - self.low, 0.0), max(self.high - self.value, 0.0))

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high}


def interval(obj: Any, *path: str, key: str = "value") -> Interval:
    """Read `{key, ci_low, ci_high}` from a results node."""
    node = dig(obj, *path)
    return Interval(float(node[key]), float(node["ci_low"]), float(node["ci_high"]))


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------

# Every figure is divided into horizontal bands, measured in inches from the canvas edge
# and only then converted to figure fractions, so a 1600x900 canvas and a 1080x1080 one
# get the same physical spacing rather than the same proportional spacing:
#
#     top margin | title | subtitle | [legend] | axes | ticks | [x label] | [note] | footer
#
# Nothing is placed by eye: the title shrinks until it fits `MAX_TITLE_LINES`, tick labels
# are wrapped to the width of the group they sit under, and the bottom bands are sized
# from the number of lines their text actually wraps to. `text_overlaps` and
# `outside_canvas` re-measure the result and are asserted on in the tests.

TOP_MARGIN_IN = 0.22
TITLE_LEAD = 1.16
SUBTITLE_LEAD = 1.30
TITLE_SUBTITLE_GAP_IN = 0.09
HEADER_BOTTOM_IN = 0.17
FOOTER_BASELINE_IN = 0.15
FOOTER_GAP_IN = 0.15
NOTE_LEAD = 1.34
NOTE_GAP_IN = 0.16
TICK_GAP_IN = 0.11
TICK_LEAD = 1.26
XLABEL_GAP_IN = 0.06
XLABEL_LEAD = 1.24
LEGEND_LEAD = 1.55
LEGEND_GAP_IN = 0.10
MIN_AXES_IN = 0.75


def _size(fig: Any) -> tuple[float, float]:
    width, height = fig.get_size_inches()
    return float(width), float(height)


def _renderer(fig: Any) -> Any:
    """The Agg renderer, which can measure text before anything has been drawn."""
    return fig.canvas.get_renderer()


def text_inches(
    fig: Any, text: str, pt: float, family: str | None = None, weight: str = "normal"
) -> float:
    """Width of `text` in inches, measured by the real renderer at the real font."""
    if not text:
        return 0.0
    kwargs: dict[str, Any] = {"fontsize": pt, "fontweight": weight}
    if family is not None:
        kwargs["family"] = family
    probe = fig.text(0.0, -1.0, text, **kwargs)
    width = float(probe.get_window_extent(renderer=_renderer(fig)).width)
    probe.remove()
    return width / float(fig.dpi)


def text_width(fig: Any, text: str, fontsize: float, family: str, weight: str = "normal") -> float:
    """Width of `text` as a fraction of the figure width."""
    return text_inches(fig, text, fontsize, family, weight) / _size(fig)[0]


def content_width_in(fig: Any) -> float:
    """The width available to figure-level text, between the two side margins."""
    return (RIGHT_EDGE - MARGIN_X) * _size(fig)[0]


def wrap_measured(
    fig: Any,
    text: str,
    max_in: float,
    pt: float,
    family: str | None = None,
    weight: str = "normal",
) -> list[str]:
    """Wrap `text` so every line measures at most `max_in` inches at `pt`.

    `textwrap` wraps on a character budget, so the budget is seeded from the measured
    average glyph advance of this exact string in this exact font and then tightened
    until every produced line really fits.
    """
    text = text.strip()
    if not text:
        return []
    if text_inches(fig, text, pt, family, weight) <= max_in:
        return [text]
    per_char = text_inches(fig, text, pt, family, weight) / len(text)
    columns = max(4, int(max_in / per_char))
    while columns > 4:
        lines = textwrap.wrap(text, width=columns, break_long_words=False, break_on_hyphens=False)
        if all(text_inches(fig, line, pt, family, weight) <= max_in for line in lines):
            return lines
        columns -= 1
    return [text]


def fit_lines(
    fig: Any,
    text: str,
    max_in: float,
    start_pt: float,
    min_pt: float,
    max_lines: int,
    weight: str = "normal",
) -> tuple[float, list[str]]:
    """Largest size in [min_pt, start_pt] whose wrap fits in `max_lines`, and that wrap.

    Among the sizes that fit, one that does not leave a one-word orphan on the last line
    wins, so a title never ends on a stray "4".
    """
    fitting: tuple[float, list[str]] | None = None
    size = start_pt
    while size >= min_pt:
        lines = wrap_measured(fig, text, max_in, size, None, weight)
        if len(lines) <= max_lines:
            if fitting is None:
                fitting = (size, lines)
            if not _orphaned(lines):
                return size, lines
        size -= 0.5
    if fitting is not None:
        return fitting
    return min_pt, wrap_measured(fig, text, max_in, min_pt, None, weight)


def _orphaned(lines: Sequence[str]) -> bool:
    """True when the last line is a stub next to the lines above it."""
    return len(lines) > 1 and len(lines[-1]) < 0.35 * max(len(line) for line in lines)


def wrap_tick_labels(fig: Any, labels: Sequence[str], pt: float, max_in: float) -> list[str]:
    """Re-wrap tick labels to the width of one group, keeping any author's line breaks."""
    wrapped: list[str] = []
    for label in labels:
        lines: list[str] = []
        for segment in label.split("\n"):
            lines.extend(wrap_measured(fig, segment, max_in, pt) or [""])
        wrapped.append("\n".join(lines))
    return wrapped


def draw_header(fig: Any, title: str, subtitle: str | None, square: bool) -> float:
    """Draw the title and optional subtitle; return the figure-fraction y left below."""
    width, height = _size(fig)
    del width
    max_in = content_width_in(fig)
    title_pt, title_lines = fit_lines(
        fig,
        title,
        max_in,
        TITLE_SQ_PT if square else TITLE_PT,
        TITLE_SQ_MIN_PT if square else TITLE_MIN_PT,
        MAX_TITLE_LINES_SQ if square else MAX_TITLE_LINES,
        weight="bold",
    )
    sub_pt = SUBTITLE_SQ_PT if square else SUBTITLE_PT
    fig._social_title_lines = len(title_lines)  # noqa: SLF001 - read back by the tests
    y = 1.0 - TOP_MARGIN_IN / height
    for line in title_lines:
        fig.text(MARGIN_X, y, line, fontsize=title_pt, fontweight="bold", va="top", color=INK)
        y -= (title_pt * TITLE_LEAD / 72.0) / height
    if subtitle:
        y -= TITLE_SUBTITLE_GAP_IN / height
        for line in wrap_measured(fig, subtitle, max_in, sub_pt):
            fig.text(MARGIN_X, y, line, fontsize=sub_pt, va="top", color=SUBTLE_INK)
            y -= (sub_pt * SUBTITLE_LEAD / 72.0) / height
    return y - HEADER_BOTTOM_IN / height


def title_line_count(fig: Any) -> int:
    """How many lines the header's title wrapped to, as recorded by `draw_header`."""
    return int(getattr(fig, "_social_title_lines", 0))


def footer_pt(square: bool) -> float:
    return FOOTER_SQ_PT if square else FOOTER_PT


def draw_footer(fig: Any, square: bool = False) -> None:
    """The one-line provenance footer every figure carries."""
    _, height = _size(fig)
    fig.text(
        MARGIN_X,
        FOOTER_BASELINE_IN / height,
        FOOTER,
        fontsize=footer_pt(square),
        color=SUBTLE_INK,
        va="baseline",
    )


def footer_band(fig: Any, square: bool) -> float:
    """Height in inches reserved at the bottom for the provenance footer."""
    return FOOTER_BASELINE_IN + footer_pt(square) / 72.0 + FOOTER_GAP_IN


def legend_band(fig: Any, rows: int, square: bool = False) -> float:
    """Vertical space, in figure fraction, to reserve above the axes for a legend."""
    _, height = _size(fig)
    if rows <= 0:
        return 0.0
    pt = LEGEND_SQ_PT if square else LEGEND_PT
    return (rows * pt * LEGEND_LEAD / 72.0 + LEGEND_GAP_IN) / height


def place_legend(fig: Any, ax: Any, top: float, ncol: int, square: bool = False) -> None:
    """Put the legend in the band reserved between the header and the axes."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(MARGIN_X, top),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize=LEGEND_SQ_PT if square else LEGEND_PT,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.5,
        borderaxespad=0.0,
    )


def pick(square: bool, wide: float, narrow: float) -> float:
    """Choose a layout constant per aspect ratio."""
    return narrow if square else wide


def tick_slot_in(square: bool, left: float, x_span: float, right: float = RIGHT_EDGE) -> float:
    """Inches one x data unit occupies, which is the room a tick label really has.

    Grouped bar charts pad their x limits so the reference-line label has somewhere to
    go, so the axes width divided by the number of groups overstates the space under
    each tick by that padding. This divides by the data span instead.
    """
    width = (SQUARE_SIZE if square else WIDE_SIZE)[0]
    return (right - left) * width / x_span


def style_axes(
    ax: Any,
    *,
    tick_pt: float = TICK_PT,
    grid_axis: str = "y",
) -> None:
    ax.tick_params(labelsize=tick_pt, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(NEUTRAL_LIGHT)


def reference_line(ax: Any, y: float, label: str, pt: float = VALUE_PT) -> None:
    """Dashed reference line at 1.0 or 0, labelled in the right-hand margin."""
    ax.axhline(y, color=NEUTRAL_DARK, linestyle=(0, (5, 4)), linewidth=1.4, zorder=1)
    ax.annotate(
        label,
        xy=(0.997, y),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="center",
        fontsize=pt,
        color=NEUTRAL_DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )


def group_xlim(ax: Any, n_groups: int, right_pad: float = 1.15) -> None:
    """Leave room on the right of a grouped bar chart for the reference-line label."""
    ax.set_xlim(-0.62, n_groups - 1 + right_pad)


def bar_values(
    ax: Any,
    bars: Iterable[Any],
    values: Sequence[float],
    fmt: str = "{:.2f}",
    *,
    dy: float = 4.0,
    fontsize: float = VALUE_PT,
) -> None:
    """Print each bar's value just outside its end."""
    for rect, value in zip(bars, values, strict=True):
        above = value >= 0
        ax.annotate(
            fmt.format(value),
            xy=(rect.get_x() + rect.get_width() / 2.0, value),
            xytext=(0, dy if above else -dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=fontsize,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )


@dataclass
class Panel:
    """One figure's reserved bands, and the axes that fills the space between them.

    Built by `build_panel`, which draws the header, measures every band and only then
    creates the axes. `finish` draws the note in the band already reserved for it, so a
    two-line footnote can never land on the x-axis label.
    """

    fig: Any
    square: bool
    ax: Any
    top: float
    legend_top: float
    bottom: float
    left: float
    right: float
    tick_labels: tuple[str, ...]
    tick_pt: float
    axis_pt: float
    value_pt: float
    legend_pt: float
    note_pt: float
    note_lines: tuple[str, ...]
    note_colour: str
    note_top: float
    xlabel_lines: tuple[str, ...]

    def xticks(self, positions: Sequence[float]) -> None:
        """Apply the pre-wrapped tick labels at `positions`."""
        self.ax.set_xticks(list(positions))
        self.ax.set_xticklabels(list(self.tick_labels))
        self.ax.tick_params(axis="x", labelsize=self.tick_pt)

    def style(self, *, ylabel: str | None = None, grid_axis: str = "y") -> None:
        """Grid, spines, tick sizing, and a y label sized to fit the axes height."""
        style_axes(self.ax, tick_pt=self.tick_pt, grid_axis=grid_axis)
        if ylabel:
            self.set_ylabel(ylabel)
        if self.xlabel_lines:
            self.ax.set_xlabel(
                "\n".join(self.xlabel_lines),
                fontsize=self.axis_pt,
                labelpad=6,
                linespacing=1.2,
            )

    def set_ylabel(self, text: str) -> None:
        """A rotated y label, shrunk until its longest line fits the axes height."""
        height_in = self.ax.get_position().height * _size(self.fig)[1]
        pt = self.axis_pt
        while pt > 8.0:
            longest = max(
                (text_inches(self.fig, line, pt) for line in text.split("\n")), default=0.0
            )
            if longest <= height_in * 0.98:
                break
            pt -= 0.5
        self.ax.set_ylabel(text, fontsize=pt, labelpad=6, linespacing=1.15)

    def legend(self, ncol: int) -> None:
        place_legend(self.fig, self.ax, self.legend_top, ncol, self.square)

    def reference_line(self, y: float, label: str) -> None:
        reference_line(self.ax, y, label, self.value_pt)

    def finish(self) -> None:
        """Draw the note into the band reserved for it, above the footer."""
        if not self.note_lines:
            return
        height = _size(self.fig)[1]
        step = (self.note_pt * NOTE_LEAD / 72.0) / height
        y = self.note_top
        for line in self.note_lines:
            self.fig.text(
                MARGIN_X, y, line, fontsize=self.note_pt, va="top", color=self.note_colour
            )
            y -= step


def build_panel(
    fig: Any,
    square: bool,
    *,
    title: str,
    subtitle: str | None = None,
    tick_labels: Sequence[str] = (),
    groups: int | None = None,
    tick_slot_inches: float | None = None,
    legend_rows: int = 0,
    xlabel: str | None = None,
    note: str | None = None,
    note_colour: str = SUBTLE_INK,
    left: float = 0.12,
    right: float = RIGHT_EDGE,
    axes: bool = True,
    tick_pt: float | None = None,
    note_pt: float | None = None,
) -> Panel:
    """Draw the header, reserve every band, and add the axes into what is left."""
    _, height = _size(fig)
    tick_size = tick_pt if tick_pt is not None else pick(square, TICK_PT, TICK_SQ_PT)
    axis_size = pick(square, AXIS_PT, AXIS_SQ_PT)
    value_size = pick(square, VALUE_PT, VALUE_SQ_PT)
    legend_size = pick(square, LEGEND_PT, LEGEND_SQ_PT)
    note_size = note_pt if note_pt is not None else pick(square, NOTE_PT, NOTE_SQ_PT)

    header_bottom = draw_header(fig, title, subtitle, square)
    legend_top = header_bottom
    top = header_bottom - legend_band(fig, legend_rows, square)

    n_groups = groups if groups is not None else max(len(tick_labels), 1)
    if tick_slot_inches is None:
        tick_slot_inches = (right - left) * _size(fig)[0] / n_groups
    # Shrink the tick font a little, if that is enough to keep every label to the number
    # of lines its author wrote; below that floor, wrapping to a third line is preferred
    # to type nobody can read.
    target = max((label.count("\n") + 1 for label in tick_labels), default=1)
    floor = max(tick_size - 2.5, 9.0)
    wrapped = tuple(wrap_tick_labels(fig, tick_labels, tick_size, tick_slot_inches * 0.94))
    while tick_size > floor and max((w.count("\n") + 1 for w in wrapped), default=0) > target:
        tick_size -= 0.5
        wrapped = tuple(wrap_tick_labels(fig, tick_labels, tick_size, tick_slot_inches * 0.94))
    tick_lines = max((label.count("\n") + 1 for label in wrapped), default=0)

    xlabel_lines = (
        tuple(wrap_measured(fig, xlabel, (right - left) * _size(fig)[0], axis_size))
        if xlabel
        else ()
    )
    note_lines = tuple(wrap_measured(fig, note, content_width_in(fig), note_size)) if note else ()

    cursor = footer_band(fig, square)
    note_top = cursor + len(note_lines) * note_size * NOTE_LEAD / 72.0
    if note_lines:
        cursor = note_top + NOTE_GAP_IN
    if xlabel_lines:
        cursor += len(xlabel_lines) * axis_size * XLABEL_LEAD / 72.0 + XLABEL_GAP_IN
    if tick_lines:
        cursor += tick_lines * tick_size * TICK_LEAD / 72.0 + TICK_GAP_IN
    bottom = cursor / height

    ax = None
    if axes:
        ax = fig.add_axes(
            (left, bottom, right - left, max(top - bottom, MIN_AXES_IN / height))
        )
    return Panel(
        fig=fig,
        square=square,
        ax=ax,
        top=top,
        legend_top=legend_top,
        bottom=bottom,
        left=left,
        right=right,
        tick_labels=wrapped,
        tick_pt=tick_size,
        axis_pt=axis_size,
        value_pt=value_size,
        legend_pt=legend_size,
        note_pt=note_size,
        note_lines=note_lines,
        note_colour=note_colour,
        note_top=note_top / height,
        xlabel_lines=xlabel_lines,
    )


# --------------------------------------------------------------------------------------
# Overlap guard
# --------------------------------------------------------------------------------------


def _visible_tick_labels(ax: Any) -> list[Any]:
    """Tick labels that are really drawn: the ones whose tick sits inside the view."""
    labels: list[Any] = []
    for axis in (ax.xaxis, ax.yaxis):
        low, high = sorted(axis.get_view_interval())
        slack = (high - low) * 1e-9
        for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
            loc = tick.get_loc()
            if loc is None or not low - slack <= loc <= high + slack:
                continue
            labels.extend(label for label in (tick.label1, tick.label2) if label.get_visible())
    return labels


def text_artists(fig: Any) -> list[Any]:
    """Every text artist `fig` actually paints, once each.

    `Figure.findobj` also returns tick labels for tick positions outside the axes' view
    limits, which matplotlib creates but never draws; those are filtered out here so the
    overlap guard reports only ink a reader can see.
    """
    artists: list[Any] = list(fig.texts)
    for legend in fig.legends:
        artists.extend(legend.get_texts())
    for ax in fig.axes:
        artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        artists.extend(_visible_tick_labels(ax))
        legend = ax.get_legend()
        if legend is not None:
            artists.extend(legend.get_texts())
    seen: set[int] = set()
    unique: list[Any] = []
    for artist in artists:
        if id(artist) in seen:
            continue
        seen.add(id(artist))
        unique.append(artist)
    return unique


def text_boxes(fig: Any) -> list[tuple[Any, Any]]:
    """Every visible, non-empty text artist in `fig`, with its device-space extent."""
    fig.canvas.draw()
    renderer = _renderer(fig)
    boxes: list[tuple[Any, Any]] = []
    for artist in text_artists(fig):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        extent = artist.get_window_extent(renderer=renderer)
        if extent.width <= 0 or extent.height <= 0:
            continue
        boxes.append((artist, extent))
    return boxes


def _intersects(a: Any, b: Any, tol: float) -> bool:
    return bool(
        a.x0 + tol < b.x1 - tol
        and b.x0 + tol < a.x1 - tol
        and a.y0 + tol < b.y1 - tol
        and b.y0 + tol < a.y1 - tol
    )


def text_overlaps(fig: Any, tol: float = 2.0) -> list[tuple[str, str]]:
    """Pairs of text artists whose ink boxes overlap by more than `tol` pixels."""
    boxes = text_boxes(fig)
    clashes: list[tuple[str, str]] = []
    for i, (first, box_a) in enumerate(boxes):
        for second, box_b in boxes[i + 1 :]:
            if _intersects(box_a, box_b, tol):
                clashes.append((first.get_text(), second.get_text()))
    return clashes


def outside_canvas(fig: Any, tol: float = 2.0) -> list[str]:
    """Text artists that fall outside the canvas by more than `tol` pixels."""
    boxes = text_boxes(fig)
    width, height = fig.get_size_inches()
    right = width * fig.dpi
    upper = height * fig.dpi
    return [
        artist.get_text()
        for artist, box in boxes
        if box.x0 < -tol or box.y0 < -tol or box.x1 > right + tol or box.y1 > upper + tol
    ]


# --------------------------------------------------------------------------------------
# Text-card layout
# --------------------------------------------------------------------------------------


def fit_fontsize(
    fig: Any, text: str, family: str, start: float, limit: float, weight: str = "normal"
) -> float:
    """Largest size <= `start` at which `text` fits within `limit` of the figure width."""
    return fit_mixed(fig, [(text, family)], start, limit, weight)


def fit_mixed(
    fig: Any,
    parts: Sequence[tuple[str, str]],
    start: float,
    limit: float,
    weight: str = "normal",
) -> float:
    """`fit_fontsize` for a line whose pieces use different families."""
    size = start
    while size > 6.0:
        total = sum(text_width(fig, text, size, family, weight) for text, family in parts)
        total += (len(parts) - 1) * text_width(fig, "  ", size, SANS_STACK[-1]) / 2.0
        if total <= limit:
            break
        size -= 0.5
    return size


@dataclass
class Flow:
    """Top-down text cursor in figure coordinates, for the explainer cards.

    `floor` is the figure fraction the cursor must not cross; `fits` reports whether the
    next block still has room, so a card can shrink rather than write over its footer.
    """

    fig: Any
    y: float
    floor: float = 0.0

    @property
    def height(self) -> float:
        return _size(self.fig)[1]

    def gap(self, inches: float) -> None:
        self.y -= inches / self.height

    def advance(self, pt: float, lines: int = 1) -> None:
        self.y -= lines * (pt * 1.35 / 72.0) / self.height

    def fits(self, pt: float, lines: int = 1) -> bool:
        return self.y - lines * (pt * 1.35 / 72.0) / self.height >= self.floor

    def line(self, text: str, pt: float, **kwargs: Any) -> None:
        self.fig.text(MARGIN_X, self.y, text, fontsize=pt, va="top", **kwargs)
        self.advance(pt)

    def paragraph(self, text: str, pt: float, **kwargs: Any) -> None:
        for wrapped in wrap_measured(self.fig, text, content_width_in(self.fig), pt):
            self.line(wrapped, pt, **kwargs)

    def mixed_line(self, parts: Sequence[tuple[str, str]], pt: float, **kwargs: Any) -> None:
        """One line whose pieces use different families (Devanagari fonts lack `→`)."""
        x = MARGIN_X
        gap = text_width(self.fig, "  ", pt, SANS_STACK[-1]) / 2.0
        for text, family in parts:
            self.fig.text(x, self.y, text, fontsize=pt, family=family, va="top", **kwargs)
            x += text_width(self.fig, text, pt, family) + gap
        self.advance(pt)


# --------------------------------------------------------------------------------------
# Small domain helpers
# --------------------------------------------------------------------------------------

CORPUS_LABELS: dict[str, str] = {
    "samayik_test": "Sāmayik test\n(prose)",
    "samayik_test_ood": "Sāmayik OOD\n(prose)",
    "itihasa_test": "Itihāsa\n(verse)",
    "flores_devtest": "FLORES",
}

CORPUS_LABELS_SQ: dict[str, str] = {
    "samayik_test": "Sāmayik\ntest",
    "samayik_test_ood": "Sāmayik\nOOD",
    "itihasa_test": "Itihāsa\n(verse)",
    "flores_devtest": "FLORES",
}

MATCHED_PAIRS: tuple[tuple[str, str], ...] = (
    ("T1_bpe_raw_32k/E1_bpe_32k", "BPE 32k"),
    ("T1_bpe_raw_64k/E1_bpe_64k", "BPE 64k"),
    ("T2_unigram_raw_32k/E1_unigram_32k", "Unigram 32k"),
    ("T2_unigram_raw_64k/E1_unigram_64k", "Unigram 64k"),
)

SPLIT_PAIRS: tuple[tuple[str, str], ...] = (
    ("T4_bpe_split_32k/T1_bpe_raw_32k", "BPE 32k"),
    ("T4_bpe_split_64k/T1_bpe_raw_64k", "BPE 64k"),
    ("T4_unigram_split_32k/T2_unigram_raw_32k", "Unigram 32k"),
    ("T4_unigram_split_64k/T2_unigram_raw_64k", "Unigram 64k"),
)

#: The withdrawn Experiment 03 headline and the bound review put on it. Both are quoted
#: from `docs/decisions.md`, "Reconciliation must preserve every non-letter character";
#: they exist in no results.json because the run that produced them was withdrawn.
WITHDRAWN_TPP = 0.976
WITHDRAWN_SOURCE = (
    'docs/decisions.md, 2026-09-05 "Reconciliation must preserve every non-letter character"'
)
DELETION_SHARE_TEXT = "77–110% of the gain"

#: Itihāsa-test leakage into DCS train before the shingle filter, and the residual after,
#: quoted from `docs/decisions.md`, "Near-duplicate leakage filter".
LEAKAGE_BEFORE_VERBATIM = 19.0
LEAKAGE_BEFORE_PREFIX = 36.0
LEAKAGE_AFTER = 0.0
LEAKAGE_SOURCE = 'docs/decisions.md, 2026-09-05 "Near-duplicate leakage filter"'

#: Fixed record of `data/processed/split/samayik_test.jsonl` used by the example card.
EXAMPLE_INDEX = 42

LEAKAGE_SOURCE_LABELS: dict[str, str] = {
    "itihasa_test": "Itihāsa test",
    "itihasa_dev": "Itihāsa dev",
    "dcs_heldout": "DCS held-out",
    "samayik_dev": "Sāmayik dev",
    "samayik_test_ood": "Sāmayik OOD",
    "samayik_test": "Sāmayik test",
    "flores_devtest": "FLORES devtest",
}


def letters_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isalpha())


def group_split_units(raw: str, split: str) -> list[list[str]]:
    """Group the split text's units under the raw whitespace unit each came from.

    Greedy by letter count: a raw unit consumes split units until their letters cover its
    own. Sandhi reversal adds letters (it restores elided vowels), never removes them, so
    the cover is reached on or after the correct unit.
    """
    raw_units = raw.split()
    split_units = split.split()
    groups: list[list[str]] = []
    cursor = 0
    for unit in raw_units:
        needed = len(letters_only(unit))
        group: list[str] = []
        covered = 0
        while cursor < len(split_units) and (not group or covered < needed):
            piece = split_units[cursor]
            cursor += 1
            group.append(piece)
            covered += len(letters_only(piece))
            if needed == 0:
                break
        groups.append(group)
    if cursor < len(split_units) and groups:
        groups[-1].extend(split_units[cursor:])
    return groups


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_language_tax(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    arms = ("T0_gpt2", "T0_o200k")
    panel = build_panel(
        fig,
        square,
        title="Older tokenizers charge Sanskrit 12 tokens per word; modern ones about 4",
        subtitle="Fertility is reported, not the headline (see next figure)",
        tick_labels=("GPT-2\n(50k vocabulary)", "o200k / GPT-4o\n(200k vocabulary)"),
        legend_rows=1,
        tick_slot_inches=tick_slot_in(square, pick(square, 0.145, 0.20), 2.1),
        left=pick(square, 0.145, 0.20),
    )
    ax = panel.ax
    langs = (("san_Deva", "Sanskrit"), ("hin_Deva", "Hindi"), ("eng_Latn", "English"))
    colours = (ACCENT, NEUTRAL, NEUTRAL_LIGHT)

    values: dict[str, dict[str, float]] = {}
    width = 0.24
    positions = list(range(len(arms)))
    for offset, ((lang, label), colour) in enumerate(zip(langs, colours, strict=True)):
        heights = [
            num(src.exp01, "metrics", arm, lang, "original", "fertility", "value") for arm in arms
        ]
        for arm, height in zip(arms, heights, strict=True):
            values.setdefault(arm, {})[lang] = height
        xs = [p + (offset - 1) * width for p in positions]
        bars = ax.bar(xs, heights, width=width * 0.92, color=colour, label=label, zorder=2)
        bar_values(ax, bars, heights, "{:.1f}", fontsize=panel.value_pt)

    panel.xticks(positions)
    panel.style(ylabel="Tokens\nper word")
    ax.set_ylim(0, max(v for arm in values.values() for v in arm.values()) * 1.16)
    ax.set_xlim(-0.55, len(arms) - 1 + 0.55)
    panel.legend(ncol=3)
    panel.finish()
    return {"fertility_original_script": values}


def fig_parity(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    arms = ("T0_gpt2", "T0_o200k", "T0_llama4", "T0_gemma3")
    panel = build_panel(
        fig,
        square,
        title="Sanskrit costs 1.8–7.9× English tokens, but only 1.1–1.4× Hindi",
        subtitle="Identical FLORES-200 devtest sentences, 1,012 pairs",
        tick_labels=("GPT-2", "o200k", "Llama-4", "Gemma-3"),
        legend_rows=2 if square else 1,
        tick_slot_inches=tick_slot_in(
            square, pick(square, 0.155, 0.22), 3 + pick(square, 1.55, 1.75) + 0.62
        ),
        left=pick(square, 0.155, 0.22),
    )
    ax = panel.ax
    series = (("eng_Latn", "Sanskrit / English", ACCENT), ("hin_Deva", "Sanskrit / Hindi", NEUTRAL))

    values: dict[str, dict[str, float]] = {}
    width = 0.34
    positions = list(range(len(arms)))
    for offset, (pivot, label, colour) in enumerate(series):
        heights = [num(src.exp01, "parity", arm, pivot, "value") for arm in arms]
        for arm, height in zip(arms, heights, strict=True):
            values.setdefault(arm, {})[pivot] = height
        xs = [p + (offset - 0.5) * width for p in positions]
        bars = ax.bar(xs, heights, width=width * 0.9, color=colour, label=label, zorder=2)
        bar_values(ax, bars, heights, fontsize=panel.value_pt)

    panel.xticks(positions)
    panel.style(ylabel="Tokens,\nSanskrit ÷ pivot")
    ax.set_ylim(0, max(v for arm in values.values() for v in arm.values()) * 1.18)
    group_xlim(ax, len(arms), right_pad=pick(square, 1.55, 1.75))
    panel.reference_line(1.0, "parity (1.0)")
    panel.legend(ncol=1 if square else 2)
    panel.finish()
    return {"parity": values}


def fig_fertility_vs_parity(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    arms = ("T0_gpt2", "T0_o200k", "T0_llama4", "T0_gemma3")
    panel = build_panel(
        fig,
        square,
        title="Counting tokens per word would have falsely confirmed the hypothesis",
        subtitle="Sanskrit against Hindi; the pre-registered threshold was 1.5",
        tick_labels=("GPT-2", "o200k", "Llama-4", "Gemma-3"),
        legend_rows=2,
        tick_slot_inches=tick_slot_in(
            square, pick(square, 0.155, 0.215), 3 + pick(square, 2.05, 2.35) + 0.62
        ),
        left=pick(square, 0.155, 0.215),
    )
    ax = panel.ax

    fert_ratio = [
        num(src.exp01, "metrics", arm, "san_Deva", "original", "fertility", "value")
        / num(src.exp01, "metrics", arm, "hin_Deva", "original", "fertility", "value")
        for arm in arms
    ]
    parity = [num(src.exp01, "parity", arm, "hin_Deva", "value") for arm in arms]

    width = 0.34
    positions = list(range(len(arms)))
    bars_a = ax.bar(
        [p - width / 2 for p in positions],
        fert_ratio,
        width=width * 0.9,
        color=MUTED_RED,
        label="Fertility ratio (tokens per word) — misleading",
        zorder=2,
    )
    bar_values(ax, bars_a, fert_ratio, fontsize=panel.value_pt)
    bars_b = ax.bar(
        [p + width / 2 for p in positions],
        parity,
        width=width * 0.9,
        color=ACCENT,
        label="Token ratio on identical content",
        zorder=2,
    )
    bar_values(ax, bars_b, parity, fontsize=panel.value_pt)

    panel.xticks(positions)
    panel.style(ylabel="Sanskrit\n÷ Hindi")
    ax.set_ylim(0, max(fert_ratio + parity) * 1.18)
    group_xlim(ax, len(arms), right_pad=pick(square, 2.05, 2.35))
    panel.reference_line(1.5, "threshold (1.5)")
    panel.legend(ncol=1)
    panel.finish()
    return {
        "arms": list(arms),
        "fertility_ratio_san_over_hin": dict(zip(arms, fert_ratio, strict=True)),
        "parity_san_over_hin": dict(zip(arms, parity, strict=True)),
    }


def fig_flip_vs_control(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    panel = build_panel(
        fig,
        square,
        title="Below English against a generic tokenizer, above it against a matched one",
        subtitle="Sāmayik test prose, arm T1_bpe_raw_64k, 2,417 pairs, 95% CI",
        tick_labels=(
            "vs T0_o200k\n(generic English-centric 200k)",
            "vs E1_bpe_64k\n(matched English control)",
        ),
        tick_slot_inches=tick_slot_in(square, pick(square, 0.175, 0.245), 2.65),
        left=pick(square, 0.175, 0.245),
    )
    ax = panel.ax
    generic = interval(src.exp02, "tpp", "samayik_test", "T1_bpe_raw_64k", "slp1", "T0_o200k")
    controlled = interval(src.exp02, "tpp_controlled", "samayik_test", "T1_bpe_raw_64k/E1_bpe_64k")
    heights = [generic.value, controlled.value]
    errors = list(zip(generic.err, controlled.err, strict=True))
    bars = ax.bar(
        [0, 1],
        heights,
        width=0.42,
        color=[NEUTRAL, ACCENT],
        yerr=errors,
        capsize=7,
        error_kw={"ecolor": INK, "elinewidth": 1.4, "capthick": 1.4},
        zorder=2,
    )
    bar_values(ax, bars, heights, "{:.3f}", dy=13.0, fontsize=panel.value_pt)
    panel.xticks([0, 1])
    panel.style(ylabel="Tokens per\nproposition")
    ax.set_ylim(0.80, 1.12)
    ax.set_xlim(-0.7, 1.95)
    panel.reference_line(1.0, "English = 1.0")
    panel.finish()
    return {
        "vs_T0_o200k": generic.as_dict(),
        "vs_E1_bpe_64k_matched_control": controlled.as_dict(),
    }


def _grouped_intervals(
    panel: Panel,
    corpora: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    lookup: Callable[[str, str], Interval],
) -> dict[str, dict[str, dict[str, float]]]:
    """Draw one CI-barred group per corpus, one bar per matched pair; return the values."""
    ax = panel.ax
    values: dict[str, dict[str, dict[str, float]]] = {}
    width = 0.74 / len(pairs)
    positions = list(range(len(corpora)))
    for offset, ((key, label), colour) in enumerate(zip(pairs, ACCENT_RAMP, strict=True)):
        heights: list[float] = []
        errors: list[tuple[float, float]] = []
        for corpus in corpora:
            point = lookup(corpus, key)
            values.setdefault(corpus, {})[key] = point.as_dict()
            heights.append(point.value)
            errors.append(point.err)
        xs = [p + (offset - (len(pairs) - 1) / 2) * width for p in positions]
        ax.bar(
            xs,
            heights,
            width=width * 0.88,
            color=colour,
            label=label,
            yerr=list(zip(*errors, strict=True)),
            capsize=3,
            error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0},
            zorder=2,
        )
    panel.xticks(positions)
    return values


def fig_by_corpus(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    corpora = ("samayik_test", "samayik_test_ood", "itihasa_test", "flores_devtest")
    labels = CORPUS_LABELS_SQ if square else CORPUS_LABELS
    panel = build_panel(
        fig,
        square,
        title="Sanskrit costs more tokens than English on prose, fewer on verse",
        subtitle="Tokens per proposition against a matched English control",
        tick_labels=[labels[c] for c in corpora],
        legend_rows=2 if square else 1,
        tick_slot_inches=tick_slot_in(
            square, pick(square, 0.175, 0.245), 3 + pick(square, 1.55, 1.80) + 0.62
        ),
        left=pick(square, 0.175, 0.245),
    )
    values = _grouped_intervals(
        panel,
        corpora,
        MATCHED_PAIRS,
        lambda corpus, key: interval(src.exp02, "tpp_controlled", corpus, key),
    )
    panel.style(ylabel="Tokens per\nproposition")
    panel.ax.set_ylim(0.0, 1.34)
    group_xlim(panel.ax, len(corpora), right_pad=pick(square, 1.55, 1.80))
    panel.reference_line(1.0, "English = 1.0")
    panel.legend(ncol=2 if square else 4)
    panel.finish()
    return {"tpp_controlled": values}


def fig_split_deltas(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    corpora = ("samayik_test", "samayik_test_ood", "flores_devtest", "itihasa_test")
    labels = CORPUS_LABELS_SQ if square else CORPUS_LABELS
    panel = build_panel(
        fig,
        square,
        title="Sandhi splitting saves a few percent on prose and costs tokens on verse",
        subtitle="Split arm minus its raw twin; tokens per proposition, 95% CI",
        tick_labels=[labels[c] for c in corpora],
        legend_rows=2 if square else 1,
        note="Below the line, splitting saves tokens.",
        tick_slot_inches=tick_slot_in(
            square, pick(square, 0.185, 0.255), 3 + pick(square, 1.65, 1.90) + 0.62
        ),
        left=pick(square, 0.185, 0.255),
    )
    values = _grouped_intervals(
        panel,
        corpora,
        SPLIT_PAIRS,
        lambda corpus, key: interval(src.exp03, "tpp_delta", corpus, key, key="delta"),
    )
    panel.style(ylabel="Δ tokens per\nproposition")
    panel.ax.set_ylim(-0.098, 0.048)
    group_xlim(panel.ax, len(corpora), right_pad=pick(square, 1.65, 1.90))
    panel.reference_line(0.0, "no change (0)")
    panel.legend(ncol=2 if square else 4)
    panel.finish()
    return {"tpp_delta": values}


def fig_artefact(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    panel = build_panel(
        fig,
        square,
        title="A result that review caught before it was published",
        subtitle="Sāmayik test prose, against the matched control E1_bpe_64k",
        tick_labels=(
            ["WITHDRAWN\nfirst run", "After the fix\nT4 split 64k", "Reference\nT1 raw 64k"]
            if square
            else [
                "WITHDRAWN first run\n(punctuation dropped)",
                "After the fix\nT4_bpe_split_64k",
                "Reference, no splitting\nT1_bpe_raw_64k",
            ]
        ),
        note=f"Review re-tokenised the deleted characters: {DELETION_SHARE_TEXT}.",
        note_colour=MUTED_RED,
        tick_slot_inches=tick_slot_in(square, pick(square, 0.175, 0.245), 3.75),
        left=pick(square, 0.175, 0.245),
    )
    ax = panel.ax
    fixed = interval(
        src.exp03, "tpp", "samayik_test", "T4_bpe_split_64k", "reconciled", "E1_bpe_64k"
    )
    raw = interval(src.exp03, "tpp", "samayik_test", "T1_bpe_raw_64k", "raw_slp1", "E1_bpe_64k")
    heights = [WITHDRAWN_TPP, fixed.value, raw.value]
    bars = ax.bar([0, 1, 2], heights, width=0.44, color=[MUTED_RED, ACCENT, NEUTRAL], zorder=2)
    bar_values(ax, bars, heights, "{:.3f}", dy=6.0, fontsize=panel.value_pt)
    panel.xticks([0, 1, 2])
    panel.style(ylabel="Tokens per\nproposition")
    ax.set_ylim(0.945, 1.075)
    ax.set_xlim(-0.7, 3.05)
    panel.reference_line(1.0, "English = 1.0")
    panel.finish()
    return {
        "withdrawn_first_run": {"value": WITHDRAWN_TPP, "source": WITHDRAWN_SOURCE},
        "after_fix_T4_bpe_split_64k": fixed.as_dict(),
        "raw_arm_T1_bpe_raw_64k": raw.as_dict(),
        "deletion_cost_share_of_gain": DELETION_SHARE_TEXT,
    }


def _example_record(src: Sources) -> dict[str, Any]:
    for record in src.split_records:
        if int(record["index"]) == EXAMPLE_INDEX:
            return record
    raise KeyError(f"record index {EXAMPLE_INDEX} not in samayik_test.jsonl")


def fig_example_card(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    record = _example_record(src)
    raw_deva = str(record["raw_deva"])
    raw_slp1 = str(record["raw_slp1"])
    split = str(record["output"])
    model = str(record["output_model"])

    panel = build_panel(
        fig,
        square,
        title="Sandhi splitting turns one written word into three",
        subtitle=f"Sāmayik test, sentence #{EXAMPLE_INDEX}",
        note=(
            "ByT5-Sanskrit splits sandhi and compounds; punctuation is preserved by "
            f"reconciliation — the model emitted “{model.split()[-1]}” where the source "
            f"had “{raw_slp1.split()[-1]}”."
        ),
        axes=False,
    )
    deva = devanagari_family()
    mono = mono_family()
    label_pt = pick(square, 12.0, 10.5)
    limit = 0.90

    # Three labelled rows share whatever the header and the note left behind. Each row's
    # size is capped by the width of its own text and by an equal share of that band; the
    # slack the width cap leaves over is then spread across the gaps between the rows.
    height = _size(fig)[1]
    groups = group_split_units(raw_slp1, split)
    pieces: list[tuple[str, bool]] = [(unit, len(group) > 1) for group in groups for unit in group]
    rows: tuple[tuple[str, str, str], ...] = (
        ("Devanagari (raw)", raw_deva, deva),
        ("SLP1 (raw)", raw_slp1, mono),
        ("Reconciled split — ByT5-Sanskrit", split, mono),
    )
    available = (panel.top - panel.note_top - NOTE_GAP_IN / height) * height
    label_in = label_pt * 1.35 / 72.0 + 0.02
    cap = min(
        pick(square, 22.0, 15.0),
        max(9.0, (available / 3.0 - label_in - 0.10) * 72.0 / 1.35),
    )
    sizes = [fit_fontsize(fig, text, family, cap, limit) for _, text, family in rows]
    used = sum(label_in + size * 1.35 / 72.0 for size in sizes)
    row_gap = max(pick(square, 0.12, 0.08), min((available - used) / 2.0, pick(square, 0.45, 0.62)))
    lead_in = max((available - used - 2.0 * row_gap) / 2.0, 0.0)

    flow = Flow(fig, panel.top, floor=panel.note_top + NOTE_GAP_IN / height)
    flow.gap(lead_in)
    for (label, text, family), size in zip(rows[:2], sizes[:2], strict=True):
        flow.line(label, label_pt, color=SUBTLE_INK)
        flow.gap(0.02)
        flow.line(text, size, family=family, color=INK)
        flow.gap(row_gap)

    # Third row: the reconciled split, with the units that came from one raw word marked.
    flow.line(rows[2][0], label_pt, color=SUBTLE_INK)
    flow.gap(0.02)
    size = sizes[2]
    space = text_width(fig, "  ", size, mono) / 2.0
    x = MARGIN_X
    for unit, highlighted in pieces:
        fig.text(
            x,
            flow.y,
            unit,
            fontsize=size,
            family=mono,
            va="top",
            color=ACCENT if highlighted else INK,
            bbox=(
                {"facecolor": HIGHLIGHT_FILL, "edgecolor": "none", "boxstyle": "square,pad=0.12"}
                if highlighted
                else None
            ),
        )
        x += text_width(fig, unit, size, mono) + space
    flow.advance(size)
    panel.finish()
    return {
        "index": EXAMPLE_INDEX,
        "raw_deva": raw_deva,
        "raw_slp1": raw_slp1,
        "output_model": model,
        "output_reconciled": split,
        "highlighted_units": [unit for unit, flag in pieces if flag],
    }


def fig_scatter(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    t4 = interval(
        src.exp04,
        "tpp_delta",
        "samayik_test",
        "T4_bpe_split_64k_oracle_dcs/T1_bpe_raw_64k_dcs",
        key="delta",
    )
    panel = build_panel(
        fig,
        square,
        title="Constraining merges on gold segment boundaries raises alignment at no token cost",
        subtitle="64k arms; tokens on Sāmayik, MorphScore on DCS",
        xlabel=(
            "Δ tokens per proposition"
            if square
            else "Δ tokens per proposition   (right = costs more)"
        ),
        note=(
            f"T4 − T1 (gold splitting alone): {t4.value:+.3f} tokens per proposition; "
            "no matching MorphScore delta is published for that pair."
        ),
        groups=6,
        tick_labels=("−0.02", "0.00", "0.02", "0.04", "0.06", "0.08"),
        left=pick(square, 0.155, 0.215),
    )
    ax = panel.ax

    points = (
        ("T5seg − T1", "T5_morphbpe_rawseg_64k_dcs/T1_bpe_raw_64k_dcs", ACCENT, (11, -5)),
        ("T5 − T1", "T5_morphbpe_raw_64k_dcs/T1_bpe_raw_64k_dcs", NEUTRAL_DARK, (11, 7)),
        (
            "T6 − T4",
            "T6_morphbpe_split_64k_dcs/T4_bpe_split_64k_oracle_dcs",
            ACCENT_RAMP[2],
            (-11, 9) if square else (11, -5),
        ),
    )
    values: dict[str, Any] = {}
    for label, key, colour, offset in points:
        tpp = interval(src.exp04, "tpp_delta", "samayik_test", key, key="delta")
        morph = interval(
            src.exp04, "morphscore_delta", key, "human_verified", "exact", key="delta"
        )
        values[label] = {"tpp_delta": tpp.as_dict(), "morphscore_delta": morph.as_dict()}
        ax.errorbar(
            tpp.value,
            morph.value,
            xerr=[[tpp.err[0]], [tpp.err[1]]],
            yerr=[[morph.err[0]], [morph.err[1]]],
            fmt="o",
            markersize=pick(square, 10, 8),
            color=colour,
            ecolor=colour,
            elinewidth=1.3,
            capsize=3,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(tpp.value, morph.value),
            xytext=offset,
            textcoords="offset points",
            fontsize=panel.legend_pt,
            color=colour,
            fontweight="bold",
            ha="right" if offset[0] < 0 else "left",
            va="center",
        )

    values["T4 − T1"] = {
        "tpp_delta": t4.as_dict(),
        "morphscore_delta": None,
        "note": "no matching MorphScore paired delta is published for this pair",
    }
    ax.axvline(t4.value, color=NEUTRAL, linestyle=(0, (3, 3)), linewidth=1.3, zorder=1)
    ax.axvline(0.0, color=NEUTRAL_DARK, linestyle=(0, (5, 4)), linewidth=1.4, zorder=1)
    ax.annotate(
        "no token cost (0)",
        xy=(0.0, 0.985),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=panel.value_pt,
        color=NEUTRAL_DARK,
    )
    panel.style(grid_axis="both")
    panel.set_ylabel("Δ MorphScore F1")
    ax.set_xlim(-0.035, pick(square, 0.132, 0.122))
    ax.set_ylim(-0.02, pick(square, 0.42, 0.46))
    panel.finish()
    return values


def fig_leakage(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    panel = build_panel(
        fig,
        square,
        title="Hash-exact exclusion missed a fifth of the verse test set; shingles caught it",
        subtitle=None if square else "Itihāsa test verses inside the DCS training corpus",
        axes=False,
    )
    width, height = _size(fig)
    top = panel.top
    floor = footer_band(fig, square) / height
    title_pt = panel.axis_pt - 1
    label_pt = panel.axis_pt - 2.5
    tick_pt = panel.tick_pt - 1.5

    before_after = [
        ("Verbatim in train", LEAKAGE_BEFORE_VERBATIM, MUTED_RED),
        ("Shares a 24-letter prefix", LEAKAGE_BEFORE_PREFIX, MUTED_RED_LIGHT),
        ("After the filter", LEAKAGE_AFTER, ACCENT),
    ]
    # Both panels stand on the same baseline, so the taller demand of the two — the
    # before/after panel's wrapped ticks, or the per-source panel's ticks plus x label —
    # sets it.
    left_x0, left_w = (0.235, 0.685) if square else (0.115, 0.30)
    left_ticks = wrap_tick_labels(
        fig, [label for label, _, _ in before_after], tick_pt, left_w * width / 3.0 * 0.94
    )
    left_lines = max(label.count("\n") + 1 for label in left_ticks)
    left_need = left_lines * tick_pt * TICK_LEAD / 72.0 + TICK_GAP_IN
    right_need = (
        tick_pt * TICK_LEAD / 72.0
        + TICK_GAP_IN
        + label_pt * XLABEL_LEAD / 72.0
        + XLABEL_GAP_IN
    )
    title_need = (title_pt * 1.35 + 7.0) / 72.0

    if square:
        # Stacked panels: the per-source panel needs the taller box, it has seven rows.
        usable = max((top - floor) * height, 1.0)
        right_h = 0.44 * usable
        right_y = floor + right_need / height
        left_bottom = right_y + (right_h + 0.30 + left_need) / height
        left = fig.add_axes(
            (left_x0, left_bottom, left_w, max(top - left_bottom, 0.08))
        )
        right = fig.add_axes((0.305, right_y, 0.615, right_h / height))
    else:
        base = floor + max(left_need, right_need) / height
        box = max(top - base - title_need / height, 0.10)
        left = fig.add_axes((left_x0, base, left_w, box))
        right = fig.add_axes((0.615, base, 0.355, box))

    heights = [value for _, value, _ in before_after]
    bars = left.bar(
        range(len(before_after)),
        heights,
        width=0.55,
        color=[colour for _, _, colour in before_after],
        zorder=2,
    )
    bar_values(left, bars, heights, "{:.0f}%", fontsize=panel.value_pt - 1)
    left.set_xticks(range(len(before_after)))
    left.set_xticklabels(left_ticks)
    left.tick_params(labelsize=tick_pt, length=0)
    left.grid(axis="y", color=GRID, linewidth=0.9)
    left.set_axisbelow(True)
    left.spines["left"].set_visible(False)
    left.set_ylabel("% of Itihāsa\ntest verses", fontsize=label_pt, labelpad=5, linespacing=1.15)
    left.set_ylim(0, 48)
    left.set_yticks([0, 20, 40])
    if not square:
        left.set_title("Before / after the filter", fontsize=title_pt, color=INK, pad=7)

    per_source = dict(dig(src.dcs_manifest, "n_dropped_shingle_per_source"))
    order = [key for key in LEAKAGE_SOURCE_LABELS if key in per_source]
    counts = [float(per_source[key]) for key in order]
    ys = list(range(len(order)))[::-1]
    colours = [MUTED_RED if key.startswith("itihasa") else NEUTRAL for key in order]
    right.barh(ys, counts, height=0.6, color=colours, zorder=2)
    for y, count in zip(ys, counts, strict=True):
        right.annotate(
            f"{int(count):,}",
            xy=(count, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=tick_pt - 1,
            color=INK,
        )
    right.set_yticks(ys)
    right.set_yticklabels([LEAKAGE_SOURCE_LABELS[key] for key in order])
    right.set_ylim(-0.7, len(order) - 0.3)
    right.tick_params(labelsize=tick_pt - 1, length=0)
    right.grid(axis="x", color=GRID, linewidth=0.9)
    right.set_axisbelow(True)
    right.spines["left"].set_visible(False)
    right.set_xlim(0, (max(counts) or 1.0) * 1.5)
    right.set_xticks([0, 10000, 20000])
    right.set_xticklabels(["0", "10k", "20k"])
    right.set_xlabel("DCS training sentences dropped", fontsize=label_pt, labelpad=5)
    if not square:
        right.set_title("Dropped per evaluation source", fontsize=title_pt, color=INK, pad=7)

    return {
        "itihasa_test_overlap_percent": {
            "verbatim_before": LEAKAGE_BEFORE_VERBATIM,
            "shared_24_letter_prefix_before": LEAKAGE_BEFORE_PREFIX,
            "after_filter": LEAKAGE_AFTER,
            "after_filter_note": "0 of 400 sampled Itihasa test lines",
            "source": LEAKAGE_SOURCE,
        },
        "n_dropped_shingle_per_source": {key: int(per_source[key]) for key in order},
    }


def fig_sandhi_card(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    del src
    panel = build_panel(
        fig, square, title="Two Sanskrit words are written as one", axes=False
    )
    deva = devanagari_family()
    mono = mono_family()
    sans = SANS_STACK[-1]
    slp1_line = "tat + api  →  tadapi"
    parts = [
        (from_slp1("tat", "devanagari"), deva),
        ("+", sans),
        (from_slp1("api", "devanagari"), deva),
        ("→", sans),
        (from_slp1("tadapi", "devanagari"), deva),
    ]
    deva_line = " ".join(text for text, _ in parts)

    # Size the two display lines first, then spread whatever is left over the four gaps,
    # so the card fills its canvas instead of stacking against the title.
    height = _size(fig)[1]
    floor = footer_band(fig, square) / height
    deva_pt = fit_mixed(fig, parts, pick(square, 34.0, 29.0), 0.86)
    slp1_pt = fit_fontsize(fig, slp1_line, mono, pick(square, 24.0, 21.0), 0.88)
    body_pt = pick(square, 16.0, 14.0)
    paragraphs = [
        wrap_measured(fig, text, content_width_in(fig), body_pt)
        for text in (
            "Sandhi erases the space; compounding fuses stems.",
            "So “tokens per word” punishes Sanskrit — we measure tokens per proposition "
            "on parallel text.",
        )
    ]
    used = (deva_pt + slp1_pt + body_pt * sum(len(p) for p in paragraphs)) * 1.35 / 72.0
    weights = (1.0, 0.6, 1.4, 0.5)
    slack = max((panel.top - floor) * height - used, 0.0)
    gaps = [slack * w / sum(weights) for w in weights]

    flow = Flow(fig, panel.top, floor=floor)
    flow.gap(gaps[0])
    flow.mixed_line(parts, deva_pt, color=ACCENT)
    flow.gap(gaps[1])
    flow.line(slp1_line, slp1_pt, family=mono, color=INK)
    flow.gap(gaps[2])
    for index, lines in enumerate(paragraphs):
        if index:
            flow.gap(gaps[3])
        for line in lines:
            flow.line(line, body_pt, color=INK)
    return {"example": {"slp1": "tat + api -> tadapi", "devanagari": deva_line}}


ARM_LADDER: tuple[tuple[str, str, str], ...] = (
    ("T0", "Off-the-shelf, English-centric", "{T0}"),
    ("T1 / T2", "BPE / Unigram on raw Sanskrit", "32k, 64k"),
    ("T3", "Off-the-shelf Indic", "{T3}"),
    ("T4", "Sandhi-split, then subword", "32k, 64k"),
    ("T5", "Merges forbidden at gold boundaries", "32k, 64k"),
    ("T6", "Split + merge-constrained (proposed)", "32k, 64k"),
    ("T7", "Byte-level, no subword vocabulary", "256"),
    ("E1", "Matched English control", "32k, 64k"),
)


def _vocab_sizes(src: Sources) -> dict[str, int]:
    """Every arm's vocabulary size, as recorded by the experiments that loaded it."""
    sizes: dict[str, int] = {}
    for results in (src.exp01, src.exp02):
        for arm, meta in dict(results.get("tokenizer_sources", {})).items():
            if isinstance(meta, dict) and "vocab_size" in meta:
                sizes[arm] = int(meta["vocab_size"])
    return sizes


def _family_range(sizes: dict[str, int], prefix: str) -> str:
    """`50k – 262k` over every loaded arm of one off-the-shelf family."""
    found = [size for arm, size in sizes.items() if arm.startswith(prefix)]
    if not found:
        return "n/a"
    low, high = min(found) // 1000, max(found) // 1000
    return f"{low}k" if low == high else f"{low}k – {high}k"


def fig_arms_card(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    panel = build_panel(
        fig,
        square,
        title="The tokenizer ladder: each arm changes one thing",
        subtitle="Trained arms are matched at 32k and 64k pieces",
        axes=False,
    )
    sizes = _vocab_sizes(src)
    substitutions = {"T0": _family_range(sizes, "T0_"), "T3": _family_range(sizes, "T3_")}
    rows = [
        (key, description, template.format(**substitutions))
        for key, description, template in ARM_LADDER
    ]

    width, height = _size(fig)
    x_desc = pick(square, 0.145, 0.175)
    x_vocab = pick(square, 0.66, 0.70)
    key_pt = pick(square, 14.0, 11.5)
    # The description column runs from `x_desc` to `x_vocab`; shrink until it fits.
    room_in = (x_vocab - x_desc - 0.02) * width
    body_pt = pick(square, 13.0, 10.5)
    while body_pt > 7.0 and max(
        text_inches(fig, description, body_pt) for _, description, _ in rows
    ) > room_in:
        body_pt -= 0.5
    head_pt = body_pt - 2.0

    floor = footer_band(fig, square) / height
    head_y = panel.top - (head_pt * 1.4 / 72.0) / height
    band = max(head_y - (0.14 / height) - floor, 0.12)
    step = band / len(rows)

    for x, label in ((MARGIN_X, "ARM"), (x_desc, "WHAT IT CHANGES"), (x_vocab, "VOCABULARY")):
        fig.text(x, head_y, label, fontsize=head_pt, color=SUBTLE_INK, va="bottom")
    y = head_y - 0.55 * step
    for key, description, vocab in rows:
        colour = ACCENT if key == "T6" else INK
        fig.add_artist(
            Line2D(
                (MARGIN_X, RIGHT_EDGE),
                (y + step * 0.45, y + step * 0.45),
                color=GRID,
                linewidth=0.8,
                transform=fig.transFigure,
            )
        )
        fig.text(MARGIN_X, y, key, fontsize=key_pt, fontweight="bold", color=colour, va="center")
        fig.text(x_desc, y, description, fontsize=body_pt, color=colour, va="center")
        fig.text(x_vocab, y, vocab, fontsize=body_pt - 0.5, color=SUBTLE_INK, va="center")
        y -= step
    return {
        "rows": [{"arm": r[0], "changes": r[1], "vocabulary": r[2]} for r in rows],
        "loaded_vocab_sizes": sizes,
    }


# --------------------------------------------------------------------------------------
# Registry, rendering, docs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """One figure: its file stem, the claim it makes, and the fields it reads."""

    name: str
    claim: str
    sources: tuple[str, ...]
    draw: Callable[[Any, bool, Sources], dict[str, Any]]


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        "00_sandhi_card",
        "Sandhi writes two Sanskrit words as one, which is why tokens per word is the "
        "wrong unit.",
        ("illustration; Devanagari generated by sanskrit_tok.encoding.from_slp1",),
        fig_sandhi_card,
    ),
    FigureSpec(
        "00_arms_card",
        "Every tokenizer arm in the study, and the one thing each changes.",
        (
            "outputs/01_baseline_penalty/results.json: tokenizer_sources[*].vocab_size",
            "outputs/02_tpp_parallel/results.json: tokenizer_sources[*].vocab_size",
        ),
        fig_arms_card,
    ),
    FigureSpec(
        "01_language_tax",
        "GPT-2 spends about 12 tokens per Sanskrit word; a modern 200k vocabulary spends "
        "about 4. Fertility is reported, never the headline.",
        (
            "outputs/01_baseline_penalty/results.json: "
            "metrics[T0_gpt2|T0_o200k][san_Deva|hin_Deva|eng_Latn].original.fertility.value",
        ),
        fig_language_tax,
    ),
    FigureSpec(
        "01_parity",
        "On identical FLORES sentences Sanskrit costs 1.8-7.9x as many tokens as English "
        "but only 1.1-1.4x as many as Hindi.",
        ("outputs/01_baseline_penalty/results.json: parity[T0_*][eng_Latn|hin_Deva].value",),
        fig_parity,
    ),
    FigureSpec(
        "01_fertility_vs_parity",
        "The Sanskrit/Hindi fertility ratio clears the pre-registered 1.5 threshold for "
        "every arm; the token ratio on identical content clears it for none.",
        (
            "outputs/01_baseline_penalty/results.json: "
            "metrics[T0_*][san_Deva|hin_Deva].original.fertility.value; "
            "parity[T0_*][hin_Deva].value",
        ),
        fig_fertility_vs_parity,
    ),
    FigureSpec(
        "02_flip_vs_control",
        "The same Sanskrit arm sits below English against a generic 200k tokenizer and "
        "above it against a matched English control; the control is the honest comparison.",
        (
            "outputs/02_tpp_parallel/results.json: "
            "tpp.samayik_test.T1_bpe_raw_64k.slp1.T0_o200k.{value,ci_low,ci_high}; "
            "tpp_controlled.samayik_test['T1_bpe_raw_64k/E1_bpe_64k'].{value,ci_low,ci_high}",
        ),
        fig_flip_vs_control,
    ),
    FigureSpec(
        "02_by_corpus",
        "Against matched English controls Sanskrit costs 3-22% more tokens on prose and "
        "on FLORES, and 34-40% fewer on verse.",
        (
            "outputs/02_tpp_parallel/results.json: "
            "tpp_controlled[corpus][pair].{value,ci_low,ci_high} for the four matched pairs",
        ),
        fig_by_corpus,
    ),
    FigureSpec(
        "03_split_deltas",
        "Reversing sandhi before subword learning lowers tokens per proposition by "
        "0.005-0.077 on prose and FLORES and raises it by 0.006-0.018 on verse; every "
        "interval excludes zero.",
        (
            "outputs/03_sandhi_split/results.json: "
            "tpp_delta[corpus][pair].{delta,ci_low,ci_high} for the four matched pairs",
        ),
        fig_split_deltas,
    ),
    FigureSpec(
        "03_artefact",
        "The first Experiment 03 headline (0.976, the first arm below its matched English "
        "control on prose) was an artefact of dropped punctuation and was withdrawn; the "
        "corrected value is above 1.0.",
        (
            "docs/decisions.md 2026-09-05 'Reconciliation must preserve every non-letter "
            "character': the withdrawn 0.976 and the 77-110% bound",
            "outputs/03_sandhi_split/results.json: "
            "tpp.samayik_test.T4_bpe_split_64k.reconciled.E1_bpe_64k and "
            "tpp.samayik_test.T1_bpe_raw_64k.raw_slp1.E1_bpe_64k",
        ),
        fig_artefact,
    ),
    FigureSpec(
        "03_example_card",
        "One Samayik test sentence in raw Devanagari, raw SLP1 and reconciled split: one "
        "written word becomes three, and the sentence-final punctuation survives.",
        (f"data/processed/split/samayik_test.jsonl: record index {EXAMPLE_INDEX}",),
        fig_example_card,
    ),
    FigureSpec(
        "04_scatter",
        "Constraining merges on gold segment boundaries alone (T5seg) buys the largest "
        "MorphScore gain of the raw arms at a token cost whose CI includes zero.",
        (
            "outputs/04_morph_constrained/results.json: "
            "tpp_delta.samayik_test[pair].{delta,ci_low,ci_high}; "
            "morphscore_delta[pair].human_verified.exact.{delta,ci_low,ci_high}",
        ),
        fig_scatter,
    ),
    FigureSpec(
        "04_leakage",
        "Hash-exact exclusion left 19% of Itihasa test verses verbatim in DCS training "
        "text; the 24-letter shingle filter removed 20,608 training sentences and left "
        "0 of 400 sampled verses.",
        (
            "docs/decisions.md 2026-09-05 'Near-duplicate leakage filter': 19% / 36% "
            "before, 0 of 400 after",
            "data/processed/dcs/manifest.json: n_dropped_shingle_per_source",
        ),
        fig_leakage,
    ),
)

SIZE_BUDGET_KB = 600.0


def render(spec: FigureSpec, src: Sources, out_dir: Path) -> dict[str, Any]:
    """Render one figure in both aspect ratios; return the numbers it plotted."""
    plotted: dict[str, Any] = {}
    for suffix, size in (("", WIDE_SIZE), ("_sq", SQUARE_SIZE)):
        fig = plt.figure(figsize=size)
        try:
            square = suffix == "_sq"
            plotted = spec.draw(fig, square, src)
            draw_footer(fig, square)
            path = out_dir / f"{spec.name}{suffix}.png"
            fig.savefig(path, dpi=DPI, facecolor="white")
            size_kb = path.stat().st_size / 1024
            if size_kb > SIZE_BUDGET_KB:
                LOGGER.warning("%s is %.0f KB, above the 600 KB budget", path.name, size_kb)
            LOGGER.info("wrote %s (%.0f KB)", path.name, size_kb)
        finally:
            plt.close(fig)
    return plotted


def write_manifest(
    out_dir: Path, root: Path, src: Sources, plotted: dict[str, dict[str, Any]]
) -> None:
    """Record the exact numbers plotted, so a post can be checked against them."""
    payload = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "source_files": src.files,
        "figures": {
            spec.name: {
                "claim": spec.claim,
                "sources": list(spec.sources),
                "files": [f"{spec.name}.png", f"{spec.name}_sq.png"],
                "values": plotted[spec.name],
            }
            for spec in FIGURES
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


README_TAIL = """
## Two numbers that are in no results.json

`03_artefact.png` plots the **withdrawn** Experiment 03 headline (0.976) and the
77-110% bound review put on it. Both come from `docs/decisions.md`, 2026-09-05,
"Reconciliation must preserve every non-letter character". No results.json holds them,
because the run that produced them was withdrawn; the two bars beside the withdrawn one
are read from the current `outputs/03_sandhi_split/results.json`.

`04_leakage.png` plots the 19% / 36% before-filter overlap and the 0-of-400 residual from
`docs/decisions.md`, 2026-09-05, "Near-duplicate leakage filter". The right-hand panel is
read from `data/processed/dcs/manifest.json`.

## One pair that could not be plotted on both axes

`04_scatter.png` wants a MorphScore paired delta for every arm it plots.
`outputs/04_morph_constrained/results.json` publishes MorphScore deltas for `T5-T1`,
`T5seg-T1` and `T6-T4` only, so `T6` appears as `T6-T4` (its own published pair) rather
than as `T6-T1`, and `T4_bpe_split_64k_oracle_dcs - T1_bpe_raw_64k_dcs` appears as a
labelled vertical line on the x-axis with no y value. `T6-T1`'s TPP delta is published
(+0.076) but has no MorphScore counterpart, so it is not plotted at all.
"""


def write_readme(out_dir: Path) -> None:
    """The index a reader of the media directory needs: file, claim, source fields."""
    lines = [
        "# Outreach media",
        "",
        "Generated by `uv run python scripts/social_figures.py --out docs/outreach/media`.",
        "Do not edit these PNGs by hand: re-run the script.",
        "",
        "Every figure exists at 1600x900 (`<name>.png`) and 1080x1080 (`<name>_sq.png`),",
        "both at dpi 200. `manifest.json` records the exact numbers plotted, so any post",
        "built from these images can be checked against it.",
        "",
        "Framing rules these figures obey (CLAUDE.md sections 1 and 2): fertility appears",
        "only as the metric that would have misled us, never as a headline; \"fewer words\"",
        "is never presented as \"fewer tokens\"; the headline comparisons are tokens per",
        "proposition against a *matched* English control.",
        "",
        "| File | Claim | Source fields |",
        "| --- | --- | --- |",
    ]
    for spec in FIGURES:
        sources = "<br>".join(s.replace("|", "\\|") for s in spec.sources)
        lines.append(f"| `{spec.name}.png` | {spec.claim} | {sources} |")
    lines.append(README_TAIL)
    (out_dir / "README.md").write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the outreach figure set.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/outreach/media"),
        help="directory to write the PNGs, README.md and manifest.json into",
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root holding outputs/ and data/"
    )
    parser.add_argument("--only", action="append", default=None, help="render only this figure")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_style()

    root: Path = args.root
    out_dir: Path = args.out if args.out.is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    src = load_sources(root)
    selected = [spec for spec in FIGURES if not args.only or spec.name in args.only]
    plotted: dict[str, dict[str, Any]] = {}
    for spec in selected:
        plotted[spec.name] = render(spec, src, out_dir)

    if len(selected) == len(FIGURES):
        write_manifest(out_dir, root, src, plotted)
        write_readme(out_dir)
        LOGGER.info("wrote README.md and manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
