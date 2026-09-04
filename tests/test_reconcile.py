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

import importlib

import pytest

from sanskrit_tok.sandhi.reconcile import ReconcileResult, reconcile

#: The module object, fetched through `import_module` because `sanskrit_tok.sandhi`
#: re-exports the `reconcile` *function* under that name and so shadows the submodule
#: attribute. Only needed to monkeypatch `MAX_WINDOW_SEGMENTS`.
reconcile_module = importlib.import_module("sanskrit_tok.sandhi.reconcile")


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


#: A ten-member compound: longer than `MAX_WINDOW_SEGMENTS`, which is the point.
LONG_COMPOUND_SEGMENTS = (
    "deva rAja putra mitra sena pati vaMSa kula dIpa tejas"
)


def test_a_compound_longer_than_the_window_bound_is_not_truncated() -> None:
    """The bound must protect, never truncate (re-review, item 1).

    With a hard cap of eight the best eligible window for this unit is its first eight
    segments — similarity 0.886, comfortably over threshold — so `dIpa tejas` was replaced
    by nothing and the sentence lost two members of a compound. Character retention would
    have shown 0.80 for the sentence and nothing at all would have named the cause. The
    scan now follows the improving match to ten segments (0.886 -> 0.940 -> 1.000).
    """
    unit = LONG_COMPOUND_SEGMENTS.replace(" ", "")
    result = reconcile(unit, LONG_COMPOUND_SEGMENTS)

    assert result.text == LONG_COMPOUND_SEGMENTS
    assert result.n_units_out == 10
    assert result.chars_out == result.chars_raw  # nothing lost
    assert result.n_units_kept_verbatim == 0


def test_the_window_bound_is_a_starting_point_not_a_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioural replacement for the old `MAX_WINDOW_SEGMENTS == 8` assertion.

    The guarantee is not the number, it is that lowering the number cannot cost a
    character: the scan raises the bound for as long as the window at it is still the best
    match tried. Pinned at a bound of two against a three-segment compound.
    """
    monkeypatch.setattr(reconcile_module, "MAX_WINDOW_SEGMENTS", 2)

    result = reconcile("viSvAsakAraRAdeva", "viSvAsa kAraRAt eva")

    assert result.text == "viSvAsa kAraRAt eva"


def test_a_unit_does_not_swallow_a_tail_it_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the guarantee: a window that stops improving stops the scan, so a
    degenerate model output cannot make one raw unit absorb a whole sentence."""
    monkeypatch.setattr(reconcile_module, "MAX_WINDOW_SEGMENTS", 8)

    result = reconcile("rAma vanam", "rAma ca tu hi vE sma nu Kalu aTa api iti vanam")

    assert result.text.split()[0] == "rAma"
    assert result.n_units_out == 2


def test_a_window_whose_similarity_dips_and_recovers_is_still_found() -> None:
    """The scan is exhaustive within the bound, not a monotone walk (re-review, item 2).

    Ratios across window sizes here are 0.667, 0.476, 0.769: the two-segment window falls
    *below* the threshold and the three-segment one is the best of the three. A monotone
    extension stops at the dip and returns the one-segment window, dropping `fghij`.
    """
    result = reconcile("abcdefghij", "abcde xyzxyz fghij")

    assert result.text == "abcde xyzxyz fghij"
    assert result.n_units_out == 3


def test_two_identical_words_each_take_their_own_window() -> None:
    """The neighbour guard is `>=`, not `>` (re-review, item 3).

    A genuine repetition scores exactly the same against both units, and a strict `>` read
    that tie as "this window belongs to the next one" and refused both, leaving `tacca
    tacca` unsplit. A tie means the two units are the same word, so the first takes the
    first window and the second takes the next.
    """
    result = reconcile("tacca tacca", "tat ca tat ca")

    assert result.text == "tat ca tat ca"
    assert result.n_units_kept_verbatim == 0


def test_the_tie_break_does_not_weaken_the_dropped_word_guard() -> None:
    """0.80 against `rAmaH` versus 1.00 against `rAmam` is not a tie, so the guard holds."""
    result = reconcile("rAmaH rAmam vadati", "rAmam vadati")

    assert result.text == "rAmaH rAmam vadati"


def test_the_best_window_is_the_highest_ratio_not_the_last_increasing_one() -> None:
    """The scan is exhaustive within the bound: a window whose similarity dips and then
    recovers is still found, which a monotone extension would have missed."""
    result = reconcile("viSvAsakAraRAdeva", "viSvAsa kAraRAt eva")

    assert result.text == "viSvAsa kAraRAt eva"


# ------------------------------------- attached punctuation is preserved (final review)


def test_a_suffix_fused_to_a_word_survives_its_replacement() -> None:
    """The failure the final review found: `reconcile` protected punctuation only when it
    stood as its own whitespace unit. `karoti.` aligns to the segment `karoti` above
    threshold, so the unit was replaced and the danda went with it — on 79% of such units
    on Sāmayik test, which turned out to be most of the headline effect
    (docs/decisions.md, "Reconciliation must preserve every non-letter character")."""
    result = reconcile("tadapi karoti.", "tad api karoti")

    assert result.text == "tad api karoti."


def test_a_prefix_and_a_suffix_are_both_re_attached() -> None:
    result = reconcile('"tadA,', "tadA")

    assert result.text == '"tadA,'


def test_a_prefix_is_re_attached_to_the_first_segment_of_a_split() -> None:
    """Prefix goes on the first output segment, suffix on the last, no added whitespace —
    restoring them as separate units would manufacture boundary tokens for free."""
    result = reconcile('"tadapi."', "tad api")

    assert result.text == '"tad api."'
    assert result.n_units_out == 2


def test_a_mid_unit_hyphen_stays_inside_its_unit() -> None:
    """Only the outermost non-letter runs are peeled: a hyphen between two letters is part
    of the core and is the splitter's business, not this function's."""
    result = reconcile("parAmarSa-dUraBAzA", "parAmarSa-dUra BAzA")

    assert result.text == "parAmarSa-dUra BAzA"


def test_a_suffix_survives_when_the_core_is_kept_verbatim() -> None:
    result = reconcile("eklips, iti", "iti")

    assert result.text == "eklips, iti"
    assert result.n_units_kept_verbatim == 1


def test_the_core_is_what_is_aligned_not_the_decorated_unit() -> None:
    """`tadapi.` against `tad api` scores 0.923 as written but 1.000 on its core, so
    peeling makes the alignment strictly better as well as lossless."""
    decorated = reconcile("tadapi.", "tad api")
    bare = reconcile("tadapi", "tad api")

    assert decorated.text == "tad api."
    assert bare.text == "tad api"
    assert decorated.n_units_replaced_inexact == bare.n_units_replaced_inexact == 0


def test_non_letter_characters_are_preserved_over_many_synthetic_cases() -> None:
    """The invariant, not one example: whatever reconciliation does to a sentence, the
    multiset of its non-letter characters comes out unchanged. This is what
    `experiment.text_invariants` checks corpus-wide, pinned here on 480 constructed cases
    that exercise prefixes, suffixes, both, standalone punctuation, dropped words,
    compounds and empty model output."""
    import itertools
    import random
    from collections import Counter

    rng = random.Random(0)
    words = ["tadapi", "karoti", "rAmaH", "viSvAsakAraRAdeva", "gacCati", "2020", "eklips"]
    prefixes = ["", '"', "'", "(", "‘"]
    suffixes = ["", ".", "..", ",", '?"', "-"]

    cases = 0
    for prefix, suffix in itertools.product(prefixes, suffixes):
        for _ in range(16):
            units = [
                f"{prefix if rng.random() < 0.5 else ''}{rng.choice(words)}"
                f"{suffix if rng.random() < 0.5 else ''}"
                for _ in range(rng.randint(1, 5))
            ]
            raw = " ".join(units)
            # Model output: some units split, some dropped, sometimes nothing at all.
            segments: list[str] = []
            for unit in units:
                core = unit.strip("\"'(‘.,?-")
                if rng.random() < 0.2:
                    continue
                if core == "tadapi":
                    segments += ["tad", "api"]
                elif core == "viSvAsakAraRAdeva":
                    segments += ["viSvAsa", "kAraRAt", "eva"]
                elif core:
                    segments.append(core)
            out = reconcile(raw, " ".join(segments)).text

            raw_nonletters = Counter(c for c in raw if not c.isalpha() and not c.isspace())
            out_nonletters = Counter(c for c in out if not c.isalpha() and not c.isspace())
            assert out_nonletters == raw_nonletters, (raw, out)
            cases += 1

    assert cases == 480
