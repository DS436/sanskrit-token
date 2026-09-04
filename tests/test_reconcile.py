"""Tests for `sanskrit_tok.sandhi.reconcile` (exp03 Task 3 addendum).

The splitter's output is not a re-segmentation of its input: it normalises, drops
sentence punctuation and sometimes drops a transliterated loanword outright, keeping only
87.3% of the raw non-space characters (docs/decisions.md, "CORRECTION: splitter character
retention is 87.3%"). Measuring TPP on that output would credit T4 with tokens saved by
deleting content, so the T4 text is the model's segmentation *reconciled* against the raw
sentence: every raw whitespace unit either aligns to model segments (and is replaced by
them — this is where legitimate sandhi and compound reversal happens) or survives
verbatim.

Everything here is a hand-computed example in SLP1; no model, no network, no fixtures.
"""

from sanskrit_tok.sandhi.reconcile import ReconcileResult, reconcile


def test_pure_sandhi_reversal_takes_the_models_segments() -> None:
    """`tadapi` -> `tad api`: one raw unit, two segments, no characters lost."""
    result = reconcile("tadapi", "tad api")

    assert result.text == "tad api"
    assert result.n_units_raw == 1
    assert result.n_units_out == 2
    assert result.n_units_kept_verbatim == 0


def test_compound_split_takes_all_three_segments() -> None:
    """`viSvAsakAraRAdeva` -> `viSvAsa kAraRAt eva`: the window has to grow past two.

    At one segment the similarity is 0.58 — *below* the 0.6 threshold — and only rises
    above it at two and again at three. A window that stopped at the first sub-threshold
    ratio would keep this unit verbatim and silently lose every compound split.
    """
    result = reconcile("viSvAsakAraRAdeva", "viSvAsa kAraRAt eva")

    assert result.text == "viSvAsa kAraRAt eva"
    assert result.n_units_out == 3
    assert result.n_units_kept_verbatim == 0


def test_dropped_punctuation_is_restored() -> None:
    """The model drops the danda; the raw unit carries no letter, so it stays verbatim."""
    result = reconcile("tadapi .", "tad api")

    assert result.text == "tad api ."
    assert result.n_units_raw == 2
    assert result.n_units_kept_verbatim == 1


def test_dropped_loanword_is_restored() -> None:
    """`eklips iti progrAmar` with `eklips` missing from the model output.

    The observed failure mode on Sāmayik's modern prose (docs/decisions.md): the model
    drops a transliterated loanword. The unit aligns to nothing above threshold, so it is
    kept and the segment cursor does not advance — the two following units still match.
    """
    result = reconcile("eklips iti progrAmar", "iti progrAmar")

    assert result.text == "eklips iti progrAmar"
    assert result.n_units_kept_verbatim == 1
    assert result.n_units_out == 3


def test_empty_model_output_keeps_every_unit_verbatim() -> None:
    result = reconcile("tadapi ca", "")

    assert result.text == "tadapi ca"
    assert result.n_units_raw == 2
    assert result.n_units_out == 2
    assert result.n_units_kept_verbatim == 2


def test_blank_raw_input_is_empty_everywhere() -> None:
    result = reconcile("   ", "tad api")

    assert result.text == ""
    assert result.n_units_raw == 0
    assert result.n_units_out == 0
    assert result.chars_raw == 0


def test_character_counts_are_non_space_and_reconciliation_restores_retention() -> None:
    """Retention is the pooled non-space ratio the manifest reports per corpus."""
    result = reconcile("eklips iti progrAmar", "iti progrAmar")

    assert result.chars_raw == len("eklipsitiprogrAmar")
    assert result.chars_model == len("itiprogrAmar")
    assert result.chars_out == len("eklipsitiprogrAmar")
    assert result.chars_model < result.chars_raw
    assert result.chars_out == result.chars_raw


def test_sandhi_reversal_may_add_characters() -> None:
    """Undone sandhi restores elided phonemes, so `chars_out` can exceed `chars_raw`.

    The observed case (docs/decisions.md): `प्राणिन आगत्य` comes back with its visarga
    put back, `prARinaH Agatya`. Retention above 1.0 is therefore normal and not a bug.
    """
    result = reconcile("prARina Agatya", "prARinaH Agatya")

    assert result.text == "prARinaH Agatya"
    assert result.chars_out == len("prARinaHAgatya")
    assert result.chars_out > result.chars_raw


def test_result_is_a_frozen_dataclass() -> None:
    result = reconcile("tadapi", "tad api")

    assert isinstance(result, ReconcileResult)


def test_a_digit_unit_is_verbatim_and_does_not_consume_a_segment() -> None:
    """A unit with no letter at all is verbatim by construction, and the segment cursor
    stays put so the letter-bearing unit after it still finds its own segments."""
    result = reconcile("2020 tadapi", "tad api")

    assert result.text == "2020 tad api"
    assert result.n_units_kept_verbatim == 1


def test_a_higher_threshold_keeps_a_loose_match_verbatim() -> None:
    loose = reconcile("gacCati", "gacCatu", threshold=0.6)
    strict = reconcile("gacCati", "gacCatu", threshold=0.99)

    assert loose.text == "gacCatu"
    assert strict.text == "gacCati"
    assert strict.n_units_kept_verbatim == 1


def test_units_are_joined_with_single_spaces() -> None:
    result = reconcile("tadapi\t\nca", "tad  api   ca")

    assert result.text == "tad api ca"


# ------------------------------------------------- the neighbour guard (Part A review)


def test_a_window_that_matches_the_next_unit_better_is_refused() -> None:
    """`rAmaH rAmam vadati` with `rAmaH` dropped must not be repaired by stealing `rAmam`.

    Without a lookahead the greedy walk hands `rAmam` (similarity 0.80) to `rAmaH`, then
    hands it to `rAmam` as well, and the sentence comes out as `rAmam rAmam vadati`: a
    *case error introduced by the pipeline*, with character retention reading a reassuring
    1.000. A window may only be taken by the current unit if it resembles that unit more
    than it resembles the next letter-bearing one.
    """
    result = reconcile("rAmaH rAmam vadati", "rAmam vadati")

    assert result.text == "rAmaH rAmam vadati"
    assert result.n_units_kept_verbatim == 1


def test_the_neighbour_guard_holds_for_a_dropped_nominative() -> None:
    result = reconcile("devaH devam paSyati", "devam paSyati")

    assert result.text == "devaH devam paSyati"
    assert result.n_units_kept_verbatim == 1


def test_the_neighbour_guard_holds_for_two_forms_of_one_verb() -> None:
    result = reconcile("gacCati gacCatu ca", "gacCatu ca")

    assert result.text == "gacCati gacCatu ca"
    assert result.n_units_kept_verbatim == 1


def test_a_genuine_repeated_word_survives_a_dropped_copy() -> None:
    """Both copies of a genuinely repeated word are kept when the model emits only one."""
    result = reconcile("tataH tataH rAjA", "tataH rAjA")

    assert result.text == "tataH tataH rAjA"
    assert result.n_units_raw == 3
    assert result.n_units_out == 3


def test_the_guard_does_not_block_an_exact_match_before_a_similar_word() -> None:
    """The guard is `>`, not `>=` against nothing: an exact match still wins its window."""
    result = reconcile("rAmaH rAmam", "rAmaH rAmam")

    assert result.text == "rAmaH rAmam"
    assert result.n_units_kept_verbatim == 0


# ------------------------------------------------------ inexact replacements (review 3)


def test_an_exact_replacement_is_not_counted_as_inexact() -> None:
    result = reconcile("tadapi", "tad api")

    assert result.n_units_replaced_inexact == 0


def test_a_replacement_below_ratio_one_is_counted_as_inexact() -> None:
    """`prARina` -> `prARinaH` changes characters: a normalisation, not a re-segmentation.

    Counting these is what separates "the model split this word" from "the model rewrote
    this word", which retention alone cannot distinguish.
    """
    result = reconcile("prARina Agatya", "prARinaH Agatya")

    assert result.n_units_replaced_inexact == 1
    assert result.n_units_kept_verbatim == 0


def test_verbatim_units_are_not_counted_as_inexact() -> None:
    result = reconcile("tadapi ca", "")

    assert result.n_units_replaced_inexact == 0
    assert result.n_units_kept_verbatim == 2


def test_result_fields_include_the_inexact_counter() -> None:
    result = reconcile("tadapi", "tad api")

    assert set(vars(result)) == {
        "text",
        "n_units_raw",
        "n_units_out",
        "n_units_kept_verbatim",
        "n_units_replaced_inexact",
        "chars_raw",
        "chars_model",
        "chars_out",
    }


# ------------------------------------------------------- bounded exhaustive scan (minor)


def test_the_window_scan_is_bounded() -> None:
    """A unit may absorb at most `MAX_WINDOW_SEGMENTS` segments, so a pathological model
    output cannot make one raw unit swallow the whole sentence."""
    from sanskrit_tok.sandhi.reconcile import MAX_WINDOW_SEGMENTS

    assert MAX_WINDOW_SEGMENTS == 8


def test_the_best_window_is_the_highest_ratio_not_the_last_increasing_one() -> None:
    """The scan is exhaustive within the bound: a window whose similarity dips and then
    recovers is still found, which a monotone extension would have missed."""
    result = reconcile("viSvAsakAraRAdeva", "viSvAsa kAraRAt eva")

    assert result.text == "viSvAsa kAraRAt eva"
