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
  knock the rest of the sentence out of alignment;
* and a window is only given to the unit with the better claim on it — a window that
  resembles the *next* raw unit at least as much is refused, so a dropped word cannot be
  "repaired" by stealing its neighbour's segments and duplicating it.

The similarity is `difflib.SequenceMatcher(...).ratio()` on the concatenated window,
thresholded at `DEFAULT_THRESHOLD`. Both variants are kept downstream — the raw model
output is stored beside the reconciled text and reported as a clearly labelled secondary
variant — so nothing here hides what the model actually said.

Pure functions only: no I/O, no model, no cache (CLAUDE.md §8).
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = ["DEFAULT_THRESHOLD", "MAX_WINDOW_SEGMENTS", "ReconcileResult", "reconcile"]

#: Similarity a window of model segments must reach before it may replace a raw unit.
#: 0.6 is `difflib`'s own conventional "close enough" cutoff and, on this model's output,
#: sits between a genuine sandhi/compound split (`tadapi` -> `tad api`, 1.0; a three-piece
#: compound, 0.94) and an accidental resemblance between two unrelated words. It is a
#: parameter because it is a judgement call, and the value used is recorded in the split
#: manifest.
DEFAULT_THRESHOLD = 0.6

#: Where the window scan starts giving up. The scan inside it is exhaustive — every window
#: size is scored and the best-scoring eligible one wins — rather than a monotone extension
#: that stops at the first dip, because similarity is not monotone in window size: a
#: compound's ratio can fall as a partly-matching segment is added and rise again when the
#: segment completing it arrives.
#:
#: It is a *starting* bound, not a hard one. `_best_window` raises it while the window at
#: the bound is still the best match tried, so a long compound is followed to its end, and
#: refuses the unit outright if the best window still ends at the bound with segments to
#: spare. A hard cap would quietly delete the tail of any compound longer than it — the
#: precise failure this module exists to prevent — while the soft one still stops a
#: degenerate output, whose ratio stops improving immediately, in O(cap) per unit.
MAX_WINDOW_SEGMENTS = 8


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

    The two unit counters are the honesty checks retention cannot provide.
    `n_units_kept_verbatim` counts what the model did not account for (its deletions, and
    every punctuation/digit unit). `n_units_replaced_inexact` counts the replacements whose
    similarity was below 1.0 — the model did not merely re-segment those characters, it
    changed them, whether by restoring an elided phoneme (legitimate) or by rewriting the
    word (not). Retention can read 1.000 while both are high, so both are reported per
    corpus.
    """

    text: str
    n_units_raw: int
    n_units_out: int
    n_units_kept_verbatim: int
    n_units_replaced_inexact: int
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


def _best_window(
    unit: str,
    segments: list[str],
    start: int,
    threshold: float,
    next_unit: str | None,
) -> tuple[int, float]:
    """The best window of segments from `start` for `unit`, as `(size, ratio)`; `(0, 0.0)`
    when none is eligible.

    Window sizes are scored with `SequenceMatcher(...).ratio()` on the concatenated
    segments and the highest-scoring *eligible* one wins (ties go to the shorter window).
    A window is eligible when:

    * its similarity to `unit` is at least `threshold` — the threshold gates eligibility,
      it does not stop the scan. A compound scores 0.58 against its first segment alone
      and only clears 0.6 at two segments, so a scan that stopped at the first
      sub-threshold ratio would lose every compound split; and
    * it resembles `unit` at least as much as it resembles `next_unit`, the next
      letter-bearing raw unit. Without that lookahead a dropped word is "repaired" by
      stealing the following word's segments and the sentence acquires a duplicate:
      `rAmaH rAmam vadati` with `rAmaH` dropped becomes `rAmam rAmam vadati`, a case error
      introduced by this pipeline, with character retention reading a reassuring 1.000.
      The comparison is `>=`, not `>`: an exact tie means the two raw units are the same
      word (`tacca tacca` against `tat ca tat ca`), and the walk should give the window to
      the one that comes first and let the next unit take the next window, not refuse both.

    **The cap protects, it never truncates.** `MAX_WINDOW_SEGMENTS` exists so a degenerate
    model output cannot make one raw unit swallow a sentence, but a fixed cap silently
    *creates* the deletion this module exists to undo: a ten-member compound whose ratio is
    still climbing at eight segments would be replaced by its first eight and the last two
    dropped, with retention showing nothing wrong. So the cap is raised whenever the window
    at the cap is the best-scoring one tried so far and segments remain — the scan follows a
    genuinely improving match as far as it goes — and if the best eligible window still ends
    exactly at the cap with segments left over, the unit is kept **verbatim** rather than
    replaced by a truncated window. Refusing to reconcile is always safe; truncating is not.
    """
    best_size = 0
    best_ratio = 0.0
    best_any_ratio = 0.0
    available = len(segments) - start
    cap = min(MAX_WINDOW_SEGMENTS, available)

    size = 0
    while size < cap:
        size += 1
        window = "".join(segments[start : start + size])
        ratio = SequenceMatcher(None, unit, window).ratio()
        if (
            ratio >= threshold
            and ratio > best_ratio
            and (
                next_unit is None
                or ratio >= SequenceMatcher(None, next_unit, window).ratio()
            )
        ):
            best_size = size
            best_ratio = ratio
        if ratio > best_any_ratio:
            best_any_ratio = ratio
            if size == cap and cap < available:
                cap += 1

    if best_size and best_size == cap and cap < available:
        return 0, 0.0
    return best_size, best_ratio


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

    # The next *letter-bearing* unit after each position, for the lookahead guard: a
    # punctuation unit between two words never competes for a window, so skipping to the
    # next real word is what makes the guard fire on `rAmaH , rAmam` as well.
    next_letter_unit: list[str | None] = [None] * len(raw_units)
    following: str | None = None
    for index in range(len(raw_units) - 1, -1, -1):
        next_letter_unit[index] = following
        if _has_letter(raw_units[index]):
            following = raw_units[index]

    out_units: list[str] = []
    kept_verbatim = 0
    replaced_inexact = 0
    cursor = 0

    for index, unit in enumerate(raw_units):
        size, ratio = (
            _best_window(unit, segments, cursor, threshold, next_letter_unit[index])
            if _has_letter(unit) and cursor < len(segments)
            else (0, 0.0)
        )
        if size == 0:
            out_units.append(unit)
            kept_verbatim += 1
            continue
        out_units.extend(segments[cursor : cursor + size])
        cursor += size
        if ratio < 1.0:
            replaced_inexact += 1

    text = " ".join(out_units)
    return ReconcileResult(
        text=text,
        n_units_raw=len(raw_units),
        n_units_out=len(out_units),
        n_units_kept_verbatim=kept_verbatim,
        n_units_replaced_inexact=replaced_inexact,
        chars_raw=_non_space_chars(raw_slp1),
        chars_model=_non_space_chars(model_slp1),
        chars_out=_non_space_chars(text),
    )
