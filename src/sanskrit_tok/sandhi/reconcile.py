"""Align a splitter's segmentation back onto the sentence it was given.

`chronbmm/sanskrit5-multitask` in segmentation mode does not re-segment its input, it
*rewrites* it: over 200 benchmarked Sāmayik-test sentences the output keeps 87.3% of the
raw non-space characters, because the model normalises (restoring an elided visarga),
drops sentence punctuation, and on modern-register prose sometimes drops a transliterated
loanword outright (docs/decisions.md, "ByT5-Sanskrit segmentation: `_` separators,
compounds split too" and its 87.3% correction).

That matters because tokens-per-proposition is the headline metric of this project
(CLAUDE.md §2.1). TPP measured on the raw model output would credit the `T4` arms with
every token the *splitter* saved by deleting a word — a measurement of the segmenter's
lossiness dressed up as a property of Sanskrit. So the T4 text is the model's segmentation
**reconciled** against the raw sentence (docs/decisions.md, "T4 text is the model's
segmentation reconciled against the raw sentence, not the raw model output"):

* every raw whitespace unit that aligns to one or more model segments is replaced by
  those segments — this is exactly where legitimate sandhi and compound reversal lives,
  and the only place characters are allowed to change;
* every raw unit that aligns to nothing survives verbatim, as does every unit with no
  letter in it at all (punctuation, digits), so deletions are undone;
* the segment cursor only advances on a successful alignment, so a dropped word does not
  knock the rest of the sentence out of alignment.

The similarity is `difflib.SequenceMatcher(...).ratio()` on the concatenated window,
thresholded at `DEFAULT_THRESHOLD`. Both variants are kept downstream — the raw model
output is stored beside the reconciled text and reported as a clearly labelled secondary
variant — so nothing here hides what the model actually said.

Pure functions only: no I/O, no model, no cache (CLAUDE.md §8).
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = ["DEFAULT_THRESHOLD", "ReconcileResult", "reconcile"]

#: Similarity a window of model segments must reach before it may replace a raw unit.
#: 0.6 is `difflib`'s own conventional "close enough" cutoff and, on this model's output,
#: sits between a genuine sandhi/compound split (`tadapi` -> `tad api`, 1.0; a three-piece
#: compound, 0.94) and an accidental resemblance between two unrelated words. It is a
#: parameter because it is a judgement call, and the value used is recorded in the split
#: manifest.
DEFAULT_THRESHOLD = 0.6


@dataclass(frozen=True)
class ReconcileResult:
    """The reconciled text and the counters a corpus-level report is pooled from.

    `chars_*` count **non-space** characters, the same definition the benchmark's
    `char_retention_nonspace` uses (docs/decisions.md, "CORRECTION: splitter character
    retention is 87.3%"), so `chars_model / chars_raw` and `chars_out / chars_raw` summed
    over a corpus are retention before and after reconciliation.

    `chars_out` can legitimately *exceed* `chars_raw`: reversing sandhi restores elided
    phonemes (`viSvAsakAraRAdeva` -> `viSvAsa kAraRAt eva` puts back a `t`). What must not
    happen is `chars_out` falling far below `chars_raw`, which is the deletion this module
    exists to undo.
    """

    text: str
    n_units_raw: int
    n_units_out: int
    n_units_kept_verbatim: int
    chars_raw: int
    chars_model: int
    chars_out: int


def _non_space_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _has_letter(unit: str) -> bool:
    """Whether a raw unit carries any letter, i.e. whether it is worth aligning.

    A unit of punctuation or digits has nothing a segmenter could have split, and the
    model routinely drops it, so it is kept verbatim without consuming a segment. Latin
    loanwords *are* letters and do go through alignment — they are handled by the
    threshold failing, which keeps them verbatim anyway, which is the same outcome by a
    route that still lets a genuinely segmented Latin string through.
    """
    return any(char.isalpha() for char in unit)


def _best_window(unit: str, segments: list[str], start: int, threshold: float) -> int:
    """How many segments from `start` best match `unit`; `0` when none does.

    Extends the window one segment at a time for as long as the similarity keeps
    *increasing*, and remembers the largest similarity seen that also cleared
    `threshold`. Both halves matter. Stopping at the first sub-threshold ratio would lose
    every multi-segment compound — `viSvAsakAraRAdeva` scores 0.58 against its first
    segment alone and only reaches 0.84 and 0.94 as the window grows — while ignoring the
    threshold would let an unrelated segment swallow a unit the model had dropped.
    """
    best_size = 0
    best_ratio = threshold
    previous_ratio = -1.0
    for size in range(1, len(segments) - start + 1):
        window = "".join(segments[start : start + size])
        ratio = SequenceMatcher(None, unit, window).ratio()
        if ratio <= previous_ratio:
            break
        previous_ratio = ratio
        if ratio >= best_ratio:
            best_ratio = ratio
            best_size = size
    return best_size


def reconcile(
    raw_slp1: str, model_slp1: str, *, threshold: float = DEFAULT_THRESHOLD
) -> ReconcileResult:
    """Rebuild `raw_slp1` using `model_slp1`'s segmentation wherever the two agree.

    Both arguments are SLP1: `raw_slp1` is the source sentence, `model_slp1` is what the
    splitter returned for it (segments separated by single spaces — `SandhiSplitter.split`
    has already normalised the model's `_` separators away). Whitespace units of the raw
    sentence are walked in order and each is either replaced by the model segments it
    aligns to or kept as written; the output units are joined with single spaces, so the
    result has exactly one whitespace unit per segment and can be counted directly by
    every downstream metric.

    Raising `threshold` makes replacement stricter and the output closer to `raw_slp1`;
    at 1.0 only a byte-identical window may replace a unit.
    """
    raw_units = raw_slp1.split()
    segments = model_slp1.split()

    out_units: list[str] = []
    kept_verbatim = 0
    cursor = 0

    for unit in raw_units:
        size = (
            _best_window(unit, segments, cursor, threshold)
            if _has_letter(unit) and cursor < len(segments)
            else 0
        )
        if size == 0:
            out_units.append(unit)
            kept_verbatim += 1
            continue
        out_units.extend(segments[cursor : cursor + size])
        cursor += size

    text = " ".join(out_units)
    return ReconcileResult(
        text=text,
        n_units_raw=len(raw_units),
        n_units_out=len(out_units),
        n_units_kept_verbatim=kept_verbatim,
        chars_raw=_non_space_chars(raw_slp1),
        chars_model=_non_space_chars(model_slp1),
        chars_out=_non_space_chars(text),
    )
