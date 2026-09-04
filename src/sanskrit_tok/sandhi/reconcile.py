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

* every raw whitespace unit is peeled into a non-letter prefix, a letter core and a
  non-letter suffix, and only the **core** is aligned; the prefix and suffix are re-attached
  to the first and last output segments with no added whitespace, so attached punctuation
  (`karoti.`, `"tadA,`) survives a replacement instead of being deleted with the unit it
  was fused to (docs/decisions.md, "Reconciliation must preserve every non-letter
  character"). This is what makes the non-letter multiset an invariant of the function;
* a core that aligns to one or more model segments is replaced by those segments — this is
  exactly where legitimate sandhi and compound reversal lives, and the only place
  characters are allowed to change;
* every raw unit whose core aligns to nothing survives verbatim, as does every unit with no
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

__all__ = [
    "DEFAULT_THRESHOLD",
    "HYPHEN",
    "MAX_WINDOW_SEGMENTS",
    "ReconcileResult",
    "reconcile",
]

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

#: The character that already marks a boundary in the raw text. Parts either side of it are
#: aligned separately and rejoined with it, so a hyphenated compound cannot be turned into
#: whitespace-separated words and counted as a boundary the splitter found.
HYPHEN = "-"


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


def _lstrip_nonletters(text: str) -> str:
    """`text` without its leading non-letter run (`"'Cancel"` -> `"Cancel"`)."""
    index = 0
    while index < len(text) and not text[index].isalpha():
        index += 1
    return text[index:]


def _rstrip_nonletters(text: str) -> str:
    """`text` without its trailing non-letter run."""
    end = len(text)
    while end > 0 and not text[end - 1].isalpha():
        end -= 1
    return text[:end]


def _peel(unit: str) -> tuple[str, str, str]:
    """Split `unit` into `(non-letter prefix, letter core, non-letter suffix)`.

    Only the outermost runs are peeled, so a hyphen or apostrophe *between* letters stays
    in the core, where it belongs — that is the splitter's business, not this function's.
    A unit with no letters at all comes back as `("", "", unit)` and never reaches
    alignment; `_has_letter` gates that case before this is called.
    """
    start = 0
    end = len(unit)
    while start < end and not unit[start].isalpha():
        start += 1
    while end > start and not unit[end - 1].isalpha():
        end -= 1
    return unit[:start], unit[start:end], unit[end:]


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
    * it resembles `unit` at least as much as it resembles `next_unit`, what the next raw
      unit would match (its letter core, or the unit itself when it has no letters).
      Without that lookahead a dropped word is "repaired" by
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
    available = len(segments) - start
    if available <= 0:
        return 0, 0.0

    # The window has to *start* on a segment this unit has a better claim to than the next
    # one does. Without it, a unit whose own segments the model dropped can open its window
    # on the next unit's segment and score well on the combination — `tadapi "2020.` with
    # `tadapi` dropped matches the two-segment window `2020tadapi` at 0.75 and emits `2020`
    # twice, once here and once when the digit unit is kept verbatim. Checking the whole
    # window is not enough, because the combination can beat each part.
    if next_unit is not None:
        head = segments[start]
        if (
            SequenceMatcher(None, unit, head).ratio()
            < SequenceMatcher(None, next_unit, head).ratio()
        ):
            return 0, 0.0

    best_size = 0
    best_ratio = 0.0
    best_any_ratio = 0.0
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
    sentence are walked in order and each is either replaced by the model segments its
    letter core aligns to — with its non-letter prefix and suffix re-attached — or kept as
    written; the output units are joined with single spaces, so the result has exactly one
    whitespace unit per segment and can be counted directly by every downstream metric.

    **Invariant:** the multiset of non-letter, non-space characters in the output equals
    that of `raw_slp1`. `sanskrit_tok.experiment.text_invariants` checks it corpus-wide,
    and every experiment measuring on this text records the result.

    Raising `threshold` makes replacement stricter and the output closer to `raw_slp1`;
    at 1.0 only a byte-identical window may replace a unit.
    """
    raw_units = raw_slp1.split()
    # Model segments are split on hyphens as well as whitespace. The model sometimes keeps a
    # hyphenated compound as one segment (`log-in`), and since the raw hyphens are
    # re-inserted at the part joins below, aligning a part against a segment that still
    # carries its own hyphen duplicated the letters either side of it: `log-in` came back as
    # `login-in`. Treating the model's hyphens as separators is consistent with the rest of
    # this function, which already regards them as disposable.
    segments = [
        piece for chunk in model_slp1.split() for piece in chunk.split(HYPHEN) if piece
    ]

    # What the next raw unit would match, for the lookahead guard: its letter core if it
    # has one, else the unit as written. A unit with no letters is kept verbatim and never
    # advances the cursor, so without it in the guard an earlier word could absorb *its*
    # segment and the sentence would end up with the text twice — `2020` duplicated rather
    # than dropped, which breaks the non-letter invariant just as surely. Punctuation costs
    # nothing to include: its similarity to a window of letters is ~0, so it never blocks a
    # legitimate replacement.
    next_unit_key: list[str | None] = [None] * len(raw_units)
    following: str | None = None
    for index in range(len(raw_units) - 1, -1, -1):
        next_unit_key[index] = following
        unit_at = raw_units[index]
        following = _peel(unit_at)[1] if _has_letter(unit_at) else unit_at

    out_units: list[str] = []
    kept_verbatim = 0
    replaced_inexact = 0
    cursor = 0

    for index, unit in enumerate(raw_units):
        prefix, core, suffix = _peel(unit) if _has_letter(unit) else ("", "", unit)

        # Hyphen-separated parts are aligned independently. A hyphen in the raw text is a
        # boundary the raw arm *already pays a token for*, so turning it into whitespace
        # would credit T4 with a boundary it did not discover (docs/decisions.md, "Raw
        # hyphens are pre-existing boundaries"). Each part gets its own window and the
        # parts are rejoined with the original hyphens, so only boundaries the model found
        # *inside* a part become spaces.
        parts = core.split(HYPHEN) if core else []
        rendered: list[str] = []
        matched_any = False
        inexact_here = False
        for part in parts:
            if not part:
                rendered.append("")
                continue
            size, ratio = (
                _best_window(part, segments, cursor, threshold, next_unit_key[index])
                if cursor < len(segments)
                else (0, 0.0)
            )
            if size == 0:
                rendered.append(part)
                continue
            matched_any = True
            if ratio < 1.0:
                inexact_here = True
            # No hyphen can appear here: the segments were split on them above, and the raw
            # hyphens are re-inserted at the part joins below, which is what keeps the
            # hyphen count exactly equal to the raw text's.
            rendered.append(" ".join(segments[cursor : cursor + size]))
            cursor += size

        if not matched_any:
            out_units.append(unit)
            kept_verbatim += 1
            # A unit with no letters is emitted as written, but the model may have emitted
            # it too (it keeps numerals, it drops punctuation). If the segment at the
            # cursor is that unit, consume it: leaving it there lets the *next* word open
            # its window on an orphan segment and swallow it, putting the text in twice
            # (`2020 rAmaH` -> `2020 2020 rAmaH`). A letter-bearing unit kept verbatim
            # never advances the cursor — that is the dropped-word case, and its segment
            # genuinely does not exist.
            if (
                not core
                and cursor < len(segments)
                and SequenceMatcher(None, unit, segments[cursor]).ratio() >= threshold
            ):
                cursor += 1
            continue

        # The raw affix is authoritative. The model emits its own punctuation, and a
        # segment that already carries a quote would otherwise be given the raw one too
        # (`'Cancel'` against the segment `'Cancel` -> `''Cancel'`), so the first and last
        # rendered pieces are stripped of their own outer non-letters first. No whitespace
        # is added: restoring an affix must not manufacture a boundary, and therefore a
        # token, that the raw sentence did not have.
        rendered[0] = prefix + _lstrip_nonletters(rendered[0])
        rendered[-1] = _rstrip_nonletters(rendered[-1]) + suffix
        out_units.append(HYPHEN.join(rendered))
        if inexact_here:
            replaced_inexact += 1

    text = " ".join(out_units)
    return ReconcileResult(
        text=text,
        n_units_raw=len(raw_units),
        n_units_out=len(text.split()),
        n_units_kept_verbatim=kept_verbatim,
        n_units_replaced_inexact=replaced_inexact,
        chars_raw=_non_space_chars(raw_slp1),
        chars_model=_non_space_chars(model_slp1),
        chars_out=_non_space_chars(text),
    )
