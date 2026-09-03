"""Tests for `sanskrit_tok.metrics.summary`: the nan-aware summaries `results.json` carries.

Every expected number is hand-computed. These tests moved out of `tests/test_exp01.py`
when `summarise_metric` moved out of the Experiment 01 runner, so they also pin the
schema `results.json` has carried since Experiment 01: the same keys, the same `None`
rather than `0.0` when there is nothing to average.
"""

import json
import math
import warnings

import pytest

from sanskrit_tok.metrics.summary import summarise_metric, summarise_values

# --- summarise_values ------------------------------------------------------------


def test_summarise_values_is_nan_aware() -> None:
    s = summarise_values([1.0, math.nan, 3.0])
    assert s == {"mean": 2.0, "std": 1.0, "n": 3, "n_undefined": 1}


def test_summarise_values_all_undefined_gives_none() -> None:
    assert summarise_values([math.nan]) == {"mean": None, "std": None, "n": 1, "n_undefined": 1}
    assert summarise_values([]) == {"mean": None, "std": None, "n": 0, "n_undefined": 0}


def test_summarise_values_counts_every_item_in_n_including_the_undefined_ones() -> None:
    """`n` is how many items there were; `n_undefined` says how many had no value."""
    s = summarise_values([2.0, math.nan, math.nan])
    assert s["n"] == 3
    assert s["n_undefined"] == 2
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == 0.0


def test_summarise_values_std_is_the_population_std() -> None:
    """Population std of [1, 2, 3] is sqrt(2/3), not the sample std sqrt(1)."""
    assert summarise_values([1.0, 2.0, 3.0])["std"] == pytest.approx(0.816496580927726)


def test_summarise_values_does_not_warn_on_an_all_nan_input() -> None:
    """`numpy.nanmean` of an all-NaN slice warns; a metric summary must stay quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert summarise_values([math.nan, math.nan])["mean"] is None


# --- summarise_metric ------------------------------------------------------------


def test_summarise_metric_replaces_distribution_with_summary() -> None:
    out = summarise_metric({"value": 2.5, "n": 2, "unit": "tokens/word", "per_word": [2, 3]})
    assert out["distribution"] == "per_word" and out["mean"] == 2.5 and out["std"] == 0.5
    assert "per_word" not in out


def test_summarise_metric_without_distribution_uses_none() -> None:
    out = summarise_metric({"value": 1.0, "n": 1, "unit": "x"})
    assert out["distribution"] is None and out["mean"] is None and out["std"] is None


def test_summarise_metric_replaces_the_distribution_with_its_mean_and_std() -> None:
    result = summarise_metric({"value": 2.0, "n": 3, "unit": "tokens/word", "per_word": [1, 2, 3]})
    assert result["value"] == 2.0
    assert result["n"] == 3
    assert result["unit"] == "tokens/word"
    assert result["distribution"] == "per_word"
    assert result["mean"] == pytest.approx(2.0)
    # Population std of [1, 2, 3] is sqrt(2/3).
    assert result["std"] == pytest.approx(0.816496580927726)
    assert "per_word" not in result


def test_summarise_metric_handles_per_text_and_per_pair() -> None:
    from_text = summarise_metric(
        {"value": 1.5, "n": 2, "unit": "bytes/token", "per_text": [1.0, 2.0]}
    )
    assert from_text["distribution"] == "per_text"
    assert from_text["mean"] == pytest.approx(1.5)
    assert from_text["std"] == pytest.approx(0.5)

    from_pair = summarise_metric(
        {"value": 3.0, "n": 2, "unit": "token ratio", "per_pair": [2.0, 4.0]}
    )
    assert from_pair["distribution"] == "per_pair"
    assert from_pair["mean"] == pytest.approx(3.0)
    assert from_pair["std"] == pytest.approx(1.0)


def test_summarise_metric_ignores_undefined_items_when_averaging() -> None:
    """NaN marks a ratio that does not exist, so it must not enter the mean as a zero."""
    result = summarise_metric(
        {"value": 1.5, "n": 3, "unit": "token ratio", "per_pair": [1.0, math.nan, 3.0]}
    )
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(1.0)


def test_summarise_metric_on_an_empty_or_absent_distribution_reports_none_not_zero() -> None:
    """`0.0` is a plausible ratio, so it would read as a measurement rather than as none."""
    empty = summarise_metric({"value": 0.0, "n": 0, "unit": "tokens/word", "per_word": []})
    assert empty["mean"] is None
    assert empty["std"] is None
    assert empty["distribution"] == "per_word"

    bare = summarise_metric({"value": 1.0, "n": 1, "unit": "tokens/word"})
    assert bare["distribution"] is None
    assert bare["mean"] is None
    assert bare["std"] is None


def test_summarise_metric_none_survives_the_json_round_trip_as_null() -> None:
    bare = summarise_metric({"value": 1.0, "n": 1, "unit": "tokens/word"})
    assert json.loads(json.dumps(bare)) == bare
    assert '"mean": null' in json.dumps(bare)


def test_summarise_metric_of_a_single_item_has_zero_std_but_a_real_mean() -> None:
    """Zero std is a genuine measurement here, unlike the empty case above."""
    single = summarise_metric({"value": 2.0, "n": 1, "unit": "tokens/word", "per_word": [2]})
    assert single["mean"] == pytest.approx(2.0)
    assert single["std"] == 0.0


def test_summarise_metric_output_is_json_serialisable() -> None:
    summary = summarise_metric({"value": 2.0, "n": 3, "unit": "tokens/word", "per_word": [1, 2, 3]})
    assert json.loads(json.dumps(summary)) == summary


def test_summarise_metric_keeps_the_results_json_schema_stable() -> None:
    """The key set is what every `results.json` written so far carries; do not widen it."""
    summary = summarise_metric(
        {"value": 2.0, "n": 3, "unit": "tokens/word", "per_word": [1, 2, 3], "n_undefined": 0}
    )
    assert set(summary) == {"value", "n", "unit", "distribution", "mean", "std"}
