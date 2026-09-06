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

TITLE_PT = 28.0
SUBTITLE_PT = 17.0
SUBTITLE_SQ_PT = 15.0
AXIS_PT = 18.0
TICK_PT = 15.0
LEGEND_PT = 14.0
VALUE_PT = 13.0
FOOTER_PT = 11.0

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


def _size(fig: Any) -> tuple[float, float]:
    width, height = fig.get_size_inches()
    return float(width), float(height)


def wrap(text: str, width_inches: float, fontsize: float) -> list[str]:
    """Greedy wrap sized from the figure width, so square variants wrap more."""
    # 0.52 em is a good average glyph advance for this sans stack at these sizes.
    per_line = max(12, int((width_inches * 0.94) / (0.52 * fontsize / 72.0)))
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > per_line and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_header(fig: Any, title: str, subtitle: str | None, square: bool) -> float:
    """Draw the title and optional subtitle; return the figure-fraction y left below."""
    width, height = _size(fig)
    sub_pt = SUBTITLE_SQ_PT if square else SUBTITLE_PT
    y = 1.0 - 0.30 / height
    for line in wrap(title, width, TITLE_PT):
        fig.text(MARGIN_X, y, line, fontsize=TITLE_PT, fontweight="bold", va="top", color=INK)
        y -= (TITLE_PT * 1.18 / 72.0) / height
    if subtitle:
        y -= 0.06 / height
        for line in wrap(subtitle, width, sub_pt):
            fig.text(MARGIN_X, y, line, fontsize=sub_pt, va="top", color=SUBTLE_INK)
            y -= (sub_pt * 1.28 / 72.0) / height
    return y - 0.10 / height


def draw_bottom_note(
    fig: Any, text: str, pt: float, colour: str = SUBTLE_INK, y0: float = 0.075
) -> None:
    """A wrapped note anchored just above the provenance footer."""
    width, height = _size(fig)
    lines = wrap(text, width, pt)
    step = (pt * 1.35 / 72.0) / height
    y = y0 + step * (len(lines) - 1)
    for line in lines:
        fig.text(MARGIN_X, y, line, fontsize=pt, va="baseline", color=colour)
        y -= step


def draw_footer(fig: Any) -> None:
    """The one-line provenance footer every figure carries."""
    fig.text(MARGIN_X, 0.022, FOOTER, fontsize=FOOTER_PT, color=SUBTLE_INK, va="bottom")


def legend_band(fig: Any, rows: int) -> float:
    """Vertical space, in figure fraction, to reserve above the axes for a legend."""
    _, height = _size(fig)
    return (rows * LEGEND_PT * 1.6 / 72.0 + 0.10) / height


def place_legend(fig: Any, ax: Any, top: float, ncol: int) -> None:
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
        fontsize=LEGEND_PT,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.5,
        borderaxespad=0.0,
    )


def pick(square: bool, wide: float, narrow: float) -> float:
    """Choose a layout constant per aspect ratio."""
    return narrow if square else wide


def main_axes(
    fig: Any,
    top: float,
    *,
    left: float = 0.12,
    right: float = RIGHT_EDGE,
    tick_lines: int = 1,
    xlabel: bool = False,
) -> Any:
    """A single axes filling the space under the header, above ticks and footer."""
    _, height = _size(fig)
    bottom_in = 0.40 + tick_lines * 0.23 + (0.36 if xlabel else 0.0)
    bottom = bottom_in / height
    return fig.add_axes((left, bottom, right - left, max(top - bottom, 0.12)))


def style_axes(ax: Any, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.tick_params(labelsize=TICK_PT, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(NEUTRAL_LIGHT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_PT, labelpad=8, linespacing=1.15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_PT, labelpad=8)


def reference_line(ax: Any, y: float, label: str) -> None:
    """Dashed reference line at 1.0 or 0, labelled in the right-hand margin."""
    ax.axhline(y, color=NEUTRAL_DARK, linestyle=(0, (5, 4)), linewidth=1.4, zorder=1)
    ax.annotate(
        label,
        xy=(0.997, y),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="center",
        fontsize=VALUE_PT,
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


# --------------------------------------------------------------------------------------
# Text-card layout
# --------------------------------------------------------------------------------------


def text_width(fig: Any, text: str, fontsize: float, family: str, weight: str = "normal") -> float:
    """Width of `text` as a fraction of the figure width, via the real renderer."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0.0, -1.0, text, fontsize=fontsize, family=family, fontweight=weight)
    width = float(probe.get_window_extent(renderer=renderer).width)
    probe.remove()
    return width / float(fig.get_window_extent(renderer=renderer).width)


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
    """Top-down text cursor in figure coordinates, for the explainer cards."""

    fig: Any
    y: float

    @property
    def height(self) -> float:
        return _size(self.fig)[1]

    def gap(self, inches: float) -> None:
        self.y -= inches / self.height

    def line(self, text: str, pt: float, **kwargs: Any) -> None:
        self.fig.text(MARGIN_X, self.y, text, fontsize=pt, va="top", **kwargs)
        self.y -= (pt * 1.35 / 72.0) / self.height

    def paragraph(self, text: str, pt: float, **kwargs: Any) -> None:
        for wrapped in wrap(text, _size(self.fig)[0], pt):
            self.line(wrapped, pt, **kwargs)

    def mixed_line(self, parts: Sequence[tuple[str, str]], pt: float, **kwargs: Any) -> None:
        """One line whose pieces use different families (Devanagari fonts lack `→`)."""
        x = MARGIN_X
        gap = text_width(self.fig, "  ", pt, SANS_STACK[-1]) / 2.0
        for text, family in parts:
            self.fig.text(x, self.y, text, fontsize=pt, family=family, va="top", **kwargs)
            x += text_width(self.fig, text, pt, family) + gap
        self.y -= (pt * 1.35 / 72.0) / self.height


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
    top = draw_header(
        fig,
        "Older tokenizers charge Sanskrit 12 tokens per word; modern ones about 4",
        "Fertility is reported, not the headline (see next figure)",
        square,
    )
    ax = main_axes(
        fig, top - legend_band(fig, 1), left=pick(square, 0.13, 0.20), tick_lines=2
    )
    arms = ("T0_gpt2", "T0_o200k")
    arm_labels = ("GPT-2\n(50k vocabulary)", "o200k / GPT-4o\n(200k vocabulary)")
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
        bar_values(ax, bars, heights, "{:.1f}")

    ax.set_xticks(positions)
    ax.set_xticklabels(arm_labels)
    style_axes(ax, ylabel="Tokens\nper word")
    ax.set_ylim(0, max(v for arm in values.values() for v in arm.values()) * 1.16)
    ax.set_xlim(-0.55, len(arms) - 1 + 0.55)
    place_legend(fig, ax, top, ncol=3)
    return {"fertility_original_script": values}


def fig_parity(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Sanskrit costs 1.8–7.9× English tokens, but only 1.1–1.4× Hindi",
        "Identical FLORES-200 devtest sentences, 1,012 pairs",
        square,
    )
    ax = main_axes(
        fig, top - legend_band(fig, 1 if not square else 2), left=pick(square, 0.14, 0.25)
    )
    arms = ("T0_gpt2", "T0_o200k", "T0_llama4", "T0_gemma3")
    labels = ("GPT-2", "o200k", "Llama-4", "Gemma-3")
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
        bar_values(ax, bars, heights, fontsize=pick(square, VALUE_PT, 11.0))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    style_axes(ax, ylabel="Tokens,\nSanskrit ÷ pivot")
    ax.tick_params(axis="x", labelsize=TICK_PT - (4 if square else 0))
    ax.set_ylim(0, max(v for arm in values.values() for v in arm.values()) * 1.16)
    group_xlim(ax, len(arms), right_pad=pick(square, 1.05, 1.60))
    reference_line(ax, 1.0, "parity (1.0)")
    place_legend(fig, ax, top, ncol=1 if square else 2)
    return {"parity": values}


def fig_fertility_vs_parity(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Counting tokens per word would have falsely confirmed the hypothesis",
        "Sanskrit against Hindi; the pre-registered threshold was 1.5",
        square,
    )
    ax = main_axes(fig, top - legend_band(fig, 2), left=pick(square, 0.14, 0.23))
    arms = ("T0_gpt2", "T0_o200k", "T0_llama4", "T0_gemma3")
    labels = ("GPT-2", "o200k", "Llama-4", "Gemma-3")

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
    bar_values(ax, bars_a, fert_ratio, fontsize=pick(square, VALUE_PT, 11.0))
    bars_b = ax.bar(
        [p + width / 2 for p in positions],
        parity,
        width=width * 0.9,
        color=ACCENT,
        label="Token ratio on identical content",
        zorder=2,
    )
    bar_values(ax, bars_b, parity, fontsize=pick(square, VALUE_PT, 11.0))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    style_axes(ax, ylabel="Sanskrit\n÷ Hindi")
    ax.tick_params(axis="x", labelsize=TICK_PT - (4 if square else 0))
    ax.set_ylim(0, max(fert_ratio + parity) * 1.18)
    group_xlim(ax, len(arms), right_pad=pick(square, 1.35, 2.60))
    reference_line(ax, 1.5, "threshold (1.5)")
    place_legend(fig, ax, top, ncol=1)
    return {
        "arms": list(arms),
        "fertility_ratio_san_over_hin": dict(zip(arms, fert_ratio, strict=True)),
        "parity_san_over_hin": dict(zip(arms, parity, strict=True)),
    }


def fig_flip_vs_control(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Below English against a generic tokenizer, above it against a matched one",
        "Sāmayik test prose, arm T1_bpe_raw_64k, 2,417 pairs, 95% CI",
        square,
    )
    ax = main_axes(fig, top, left=pick(square, 0.16, 0.27), tick_lines=2)
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
    bar_values(ax, bars, heights, "{:.3f}", dy=13.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "vs T0_o200k\n(generic 200k)"
            if square
            else "vs T0_o200k\n(generic English-centric 200k)",
            "vs E1_bpe_64k\n(matched control)"
            if square
            else "vs E1_bpe_64k\n(matched English control)",
        ]
    )
    ax.tick_params(axis="x", labelsize=TICK_PT - (3 if square else 1))
    style_axes(ax, ylabel="Tokens per\nproposition")
    ax.set_ylim(0.80, 1.12)
    ax.set_xlim(-0.7, 1.9)
    reference_line(ax, 1.0, "English = 1.0")
    return {
        "vs_T0_o200k": generic.as_dict(),
        "vs_E1_bpe_64k_matched_control": controlled.as_dict(),
    }


def _grouped_intervals(
    ax: Any,
    corpora: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    lookup: Callable[[str, str], Interval],
    square: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    """Draw one CI-barred group per corpus, one bar per matched pair; return the values."""
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
    labels = CORPUS_LABELS_SQ if square else CORPUS_LABELS
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[c] for c in corpora])
    ax.tick_params(axis="x", labelsize=TICK_PT - (5 if square else 1))
    return values


def fig_by_corpus(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Sanskrit costs more tokens than English on prose, fewer on verse",
        "Tokens per proposition against a matched English control",
        square,
    )
    ax = main_axes(
        fig,
        top - legend_band(fig, 1 if not square else 2),
        left=pick(square, 0.16, 0.27),
        tick_lines=2,
    )
    corpora = ("samayik_test", "samayik_test_ood", "itihasa_test", "flores_devtest")
    values = _grouped_intervals(
        ax,
        corpora,
        MATCHED_PAIRS,
        lambda corpus, key: interval(src.exp02, "tpp_controlled", corpus, key),
        square,
    )
    style_axes(ax, ylabel="Tokens per\nproposition")
    ax.tick_params(axis="x", labelsize=TICK_PT - (5 if square else 1))
    ax.set_ylim(0.0, 1.34)
    group_xlim(ax, len(corpora), right_pad=pick(square, 1.30, 1.85))
    reference_line(ax, 1.0, "English = 1.0")
    place_legend(fig, ax, top, ncol=2 if square else 4)
    return {"tpp_controlled": values}


def fig_split_deltas(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Sandhi splitting saves a few percent on prose and costs tokens on verse",
        "Split arm minus its raw twin; tokens per proposition, 95% CI",
        square,
    )
    ax = main_axes(
        fig,
        top - legend_band(fig, 1 if not square else 2),
        left=pick(square, 0.17, 0.28),
        tick_lines=2,
    )
    corpora = ("samayik_test", "samayik_test_ood", "flores_devtest", "itihasa_test")
    values = _grouped_intervals(
        ax,
        corpora,
        SPLIT_PAIRS,
        lambda corpus, key: interval(src.exp03, "tpp_delta", corpus, key, key="delta"),
        square,
    )
    style_axes(ax, ylabel="Δ tokens per\nproposition")
    ax.tick_params(axis="x", labelsize=TICK_PT - (5 if square else 1))
    ax.set_ylim(-0.098, 0.048)
    group_xlim(ax, len(corpora), right_pad=pick(square, 1.45, 1.95))
    reference_line(ax, 0.0, "no change (0)")
    if not square:
        ax.annotate(
            "below the line = splitting saves tokens",
            xy=(0.012, 0.05),
            xycoords="axes fraction",
            fontsize=VALUE_PT,
            color=SUBTLE_INK,
        )
    place_legend(fig, ax, top, ncol=2 if square else 4)
    return {"tpp_delta": values}


def fig_artefact(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "A result that review caught before it was published",
        "Sāmayik test prose, against the matched control E1_bpe_64k",
        square,
    )
    ax = main_axes(fig, top, left=pick(square, 0.16, 0.27), tick_lines=4)
    fixed = interval(
        src.exp03, "tpp", "samayik_test", "T4_bpe_split_64k", "reconciled", "E1_bpe_64k"
    )
    raw = interval(src.exp03, "tpp", "samayik_test", "T1_bpe_raw_64k", "raw_slp1", "E1_bpe_64k")
    heights = [WITHDRAWN_TPP, fixed.value, raw.value]
    labels = (
        [
            "WITHDRAWN\nfirst run",
            "After the fix\nT4 split 64k",
            "Reference\nT1 raw 64k",
        ]
        if square
        else [
            "WITHDRAWN first run\n(punctuation dropped)",
            "After the fix\nT4_bpe_split_64k",
            "Reference, no splitting\nT1_bpe_raw_64k",
        ]
    )
    bars = ax.bar([0, 1, 2], heights, width=0.44, color=[MUTED_RED, ACCENT, NEUTRAL], zorder=2)
    bar_values(ax, bars, heights, "{:.3f}", dy=6.0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=TICK_PT - 2)
    style_axes(ax, ylabel="Tokens per\nproposition")
    ax.set_ylim(0.945, 1.075)
    ax.set_xlim(-0.7, 2.95)
    reference_line(ax, 1.0, "English = 1.0")
    draw_bottom_note(
        fig,
        f"Review re-tokenised the deleted characters: {DELETION_SHARE_TEXT}.",
        VALUE_PT,
        MUTED_RED,
    )
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

    top = draw_header(
        fig,
        "Sandhi splitting turns one written word into three",
        f"Sāmayik test, sentence #{EXAMPLE_INDEX}",
        square,
    )
    deva = devanagari_family()
    mono = mono_family()
    label_pt = 13.0 if not square else 12.0
    body_pt = 22.0 if not square else 15.0
    limit = 0.90

    flow = Flow(fig, top)
    for label, text, family in (
        ("Devanagari (raw)", raw_deva, deva),
        ("SLP1 (raw)", raw_slp1, mono),
    ):
        flow.line(label, label_pt, color=SUBTLE_INK)
        flow.gap(0.02)
        size = fit_fontsize(fig, text, family, body_pt, limit)
        flow.line(text, size, family=family, color=INK)
        flow.gap(0.12)

    # Third row: the reconciled split, with the units that came from one raw word marked.
    groups = group_split_units(raw_slp1, split)
    pieces: list[tuple[str, bool]] = [(unit, len(group) > 1) for group in groups for unit in group]
    flow.line("Reconciled split — ByT5-Sanskrit", label_pt, color=SUBTLE_INK)
    flow.gap(0.02)
    size = fit_fontsize(fig, split, mono, body_pt, limit)
    space = text_width(fig, "  ", size, mono) / 2.0
    x = MARGIN_X
    height = _size(fig)[1]
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
    flow.y -= (size * 1.35 / 72.0) / height
    draw_bottom_note(
        fig,
        "ByT5-Sanskrit splits sandhi and compounds; punctuation is preserved by "
        f"reconciliation — the model emitted “{model.split()[-1]}” where the source "
        f"had “{raw_slp1.split()[-1]}”.",
        label_pt,
    )
    return {
        "index": EXAMPLE_INDEX,
        "raw_deva": raw_deva,
        "raw_slp1": raw_slp1,
        "output_model": model,
        "output_reconciled": split,
        "highlighted_units": [unit for unit, flag in pieces if flag],
    }


def fig_scatter(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Constraining merges on gold segment boundaries raises alignment at no token cost",
        "64k arms; tokens on Sāmayik, MorphScore on DCS",
        square,
    )
    ax = main_axes(
        fig,
        top,
        left=pick(square, 0.17, 0.26),
        tick_lines=int(pick(square, 2, 3)),
        xlabel=True,
    )

    points = (
        (
            "T5seg − T1",
            "T5_morphbpe_rawseg_64k_dcs/T1_bpe_raw_64k_dcs",
            ACCENT,
            (12, -6) if not square else (12, -6),
        ),
        (
            "T5 − T1",
            "T5_morphbpe_raw_64k_dcs/T1_bpe_raw_64k_dcs",
            NEUTRAL_DARK,
            (12, 6) if not square else (12, 8),
        ),
        (
            "T6 − T4",
            "T6_morphbpe_split_64k_dcs/T4_bpe_split_64k_oracle_dcs",
            ACCENT_RAMP[2],
            (12, -4) if not square else (-12, 10),
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
            markersize=11,
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
            fontsize=LEGEND_PT,
            color=colour,
            fontweight="bold",
            ha="right" if offset[0] < 0 else "left",
        )

    t4 = interval(
        src.exp04,
        "tpp_delta",
        "samayik_test",
        "T4_bpe_split_64k_oracle_dcs/T1_bpe_raw_64k_dcs",
        key="delta",
    )
    values["T4 − T1"] = {
        "tpp_delta": t4.as_dict(),
        "morphscore_delta": None,
        "note": "no matching MorphScore paired delta is published for this pair",
    }
    ax.axvline(t4.value, color=NEUTRAL, linestyle=(0, (3, 3)), linewidth=1.3, zorder=1)
    draw_bottom_note(
        fig,
        f"T4 − T1 (gold splitting alone): {t4.value:+.3f} tokens per proposition; "
        "no matching MorphScore delta is published for that pair.",
        VALUE_PT - (2 if square else 0),
    )

    ax.axvline(0.0, color=NEUTRAL_DARK, linestyle=(0, (5, 4)), linewidth=1.4, zorder=1)
    ax.annotate(
        "no token cost (0)",
        xy=(0.0, 0.98 if square else 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(7, 0),
        textcoords="offset points",
        ha="left",
        va="top" if square else "bottom",
        fontsize=VALUE_PT,
        color=NEUTRAL_DARK,
    )
    ax.grid(color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_PT, length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        "Δ tokens per proposition" if square else "Δ tokens per proposition   (right = costs more)",
        fontsize=AXIS_PT,
        labelpad=8,
    )
    ax.set_ylabel("Δ MorphScore\nF1", fontsize=AXIS_PT, labelpad=8, linespacing=1.15)
    ax.set_xlim(-0.035, pick(square, 0.135, 0.125))
    ax.set_ylim(-0.015, pick(square, 0.40, 0.44))
    return values


def fig_leakage(fig: Any, square: bool, src: Sources) -> dict[str, Any]:
    top = draw_header(
        fig,
        "Hash-exact exclusion missed a fifth of the verse test set; shingles caught it",
        None if square else "Itihāsa test verses inside the DCS training corpus",
        square,
    )
    height = _size(fig)[1]
    title_pt = AXIS_PT - 3
    label_pt = AXIS_PT - 5
    if square:
        # Stacked panels: the per-source panel needs the taller box, it has seven rows.
        usable = max(top - 0.075, 0.20)
        right = fig.add_axes((0.30, 0.075 + 0.185 * usable, 0.62, 0.385 * usable))
        left = fig.add_axes((0.22, 0.075 + 0.775 * usable, 0.70, 0.19 * usable))
    else:
        base = 0.62 / height
        left = fig.add_axes((0.105, base + 0.14, 0.295, max(top - base - 0.21, 0.10)))
        right = fig.add_axes((0.615, base + 0.115, 0.355, max(top - base - 0.185, 0.10)))

    before_after = [
        ("Verbatim\nin train", LEAKAGE_BEFORE_VERBATIM, MUTED_RED),
        ("Shares a 24-\nletter prefix", LEAKAGE_BEFORE_PREFIX, MUTED_RED_LIGHT),
        ("After the\nfilter", LEAKAGE_AFTER, ACCENT),
    ]
    heights = [value for _, value, _ in before_after]
    bars = left.bar(
        range(len(before_after)),
        heights,
        width=0.55,
        color=[colour for _, _, colour in before_after],
        zorder=2,
    )
    bar_values(left, bars, heights, "{:.0f}%")
    left.set_xticks(range(len(before_after)))
    left.set_xticklabels([label for label, _, _ in before_after])
    left.tick_params(labelsize=TICK_PT - 4, length=0)
    left.grid(axis="y", color=GRID, linewidth=0.9)
    left.set_axisbelow(True)
    left.spines["left"].set_visible(False)
    left.set_ylabel("% of Itihāsa\ntest verses", fontsize=label_pt, labelpad=6, linespacing=1.15)
    left.set_ylim(0, 46)
    left.set_yticks([0, 20, 40])
    if not square:
        left.set_title("Before / after the filter", fontsize=title_pt, color=INK, pad=8)

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
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=TICK_PT - 5.5,
            color=INK,
        )
    right.set_yticks(ys)
    right.set_yticklabels([LEAKAGE_SOURCE_LABELS[key] for key in order])
    right.set_ylim(-0.7, len(order) - 0.3)
    right.tick_params(labelsize=TICK_PT - 5.5, length=0)
    right.grid(axis="x", color=GRID, linewidth=0.9)
    right.set_axisbelow(True)
    right.spines["left"].set_visible(False)
    right.set_xlim(0, (max(counts) or 1.0) * 1.45)
    right.set_xticks([0, 10000, 20000])
    right.set_xticklabels(["0", "10k", "20k"])
    right.set_xlabel("DCS training sentences dropped", fontsize=label_pt, labelpad=6)
    if not square:
        right.set_title(
            "Dropped per evaluation source", fontsize=title_pt, color=INK, pad=8
        )

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
    top = draw_header(fig, "Two Sanskrit words are written as one", None, square)
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

    flow = Flow(fig, top)
    flow.gap(0.10)
    size = fit_mixed(fig, parts, 36.0 if not square else 27.0, 0.86)
    flow.mixed_line(parts, size, color=ACCENT)
    flow.gap(0.18)
    size = fit_fontsize(fig, slp1_line, mono, 26.0 if not square else 20.0, 0.88)
    flow.line(slp1_line, size, family=mono, color=INK)
    flow.gap(0.34)
    body_pt = 17.0 if not square else 14.5
    flow.paragraph("Sandhi erases the space; compounding fuses stems.", body_pt, color=INK)
    flow.gap(0.14)
    flow.paragraph(
        "So “tokens per word” punishes Sanskrit — we measure tokens per proposition "
        "on parallel text.",
        body_pt,
        color=INK,
    )
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
    top = draw_header(
        fig,
        "The tokenizer ladder: each arm changes one thing",
        "Trained arms are matched at 32k and 64k pieces",
        square,
    )
    sizes = _vocab_sizes(src)
    substitutions = {"T0": _family_range(sizes, "T0_"), "T3": _family_range(sizes, "T3_")}
    rows = [
        (key, description, template.format(**substitutions))
        for key, description, template in ARM_LADDER
    ]

    height = _size(fig)[1]
    key_pt = pick(square, 15.0, 12.0)
    body_pt = pick(square, 14.0, 11.0)
    step = max(top - 0.42 / height, 0.10) / len(rows)
    x_desc = pick(square, 0.145, 0.175)
    x_vocab = pick(square, 0.66, 0.70)

    y = top - 0.30 / height
    head_pt = body_pt - 3
    for x, label in ((MARGIN_X, "ARM"), (x_desc, "WHAT IT CHANGES"), (x_vocab, "VOCABULARY")):
        fig.text(x, y + 0.036, label, fontsize=head_pt, color=SUBTLE_INK, va="bottom")
    for key, description, vocab in rows:
        colour = ACCENT if key == "T6" else INK
        fig.add_artist(
            Line2D(
                (MARGIN_X, RIGHT_EDGE),
                (y + 0.028, y + 0.028),
                color=GRID,
                linewidth=0.8,
                transform=fig.transFigure,
            )
        )
        fig.text(MARGIN_X, y, key, fontsize=key_pt, fontweight="bold", color=colour, va="center")
        fig.text(x_desc, y, description, fontsize=body_pt, color=colour, va="center")
        fig.text(x_vocab, y, vocab, fontsize=body_pt - 1, color=SUBTLE_INK, va="center")
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
            plotted = spec.draw(fig, suffix == "_sq", src)
            draw_footer(fig)
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
