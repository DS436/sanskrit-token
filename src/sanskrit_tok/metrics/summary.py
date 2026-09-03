"""Nan-aware summaries of a metric's per-item distribution, for `results.json`.

Every experiment writes a `results.json` (CLAUDE.md §9), and a distribution is far too
large to put in it: 1012 FLORES sentences × several arms × two scripts would be roughly a
million floats. `summarise_metric` therefore replaces the distribution with its mean and
standard deviation, keeping the `value`/`n`/`unit` contract of CLAUDE.md §7 unchanged.

Two rules matter more than the arithmetic. First, an undefined item is `nan`, never `0.0`
(see `_ratio.py`), so every reduction here is nan-aware: an undefined ratio is skipped,
not averaged in as a zero. Second, when there is nothing to average — no distribution, an
empty one, or one with no defined item in it — `mean` and `std` are `None`, which JSON
writes as `null`. `0.0` is a plausible fertility, bytes-per-token or parity value, so it
would be indistinguishable in `results.json` from a real measurement of zero.
"""

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import numpy as np

__all__ = ["DISTRIBUTION_KEYS", "MetricSummary", "summarise_metric", "summarise_values"]

#: The distribution keys a metric may attach on top of the `MetricResult` contract
#: (CLAUDE.md §7), in the order `summarise_metric` looks for them. A metric attaches at
#: most one, so the order only decides which wins if that ever stops being true.
DISTRIBUTION_KEYS: tuple[str, ...] = ("per_word", "per_text", "per_pair")


class MetricSummary(TypedDict):
    """What `results.json` carries in place of a metric's full per-item distribution.

    `value`, `n` and `unit` are the CLAUDE.md §7 contract, passed through unchanged.
    `distribution` names the per-item key that was summarised, or is `None` when the
    metric attached none. `mean` and `std` summarise that key's values, and are `None`
    — not `0.0` — whenever there were no values to summarise, because a mean over zero
    items does not exist and writing `0.0` for it puts a number into `results.json` that
    reads as a measurement.
    """

    value: float
    n: int
    unit: str
    distribution: str | None
    mean: float | None
    std: float | None


def summarise_values(values: Sequence[float]) -> dict[str, float | int | None]:
    """Mean and population std of `values`, ignoring the `nan`s among them.

    Returns `mean`, `std`, `n` = how many items there were in total, and `n_undefined` =
    how many of those were `nan`. `n` counts the undefined items too: the distribution had
    that many entries, and hiding the undefined ones would make `n` disagree with the
    length of the list it summarises.

    `std` is the population standard deviation (`ddof=0`, `numpy`'s default), so a single
    defined item gives `0.0` — a real measurement, unlike the `None` below.

    `mean` and `std` are `None` when no item is defined (an empty sequence, or all `nan`).
    `numpy.nanmean` returns `nan` with a RuntimeWarning for an all-`nan` slice; this
    returns `None` instead, quietly, because `None` survives JSON as `null` while `nan`
    does not survive it at all.
    """
    array = np.asarray(values, dtype=float)
    defined = ~np.isnan(array)
    n_undefined = int(array.size - int(defined.sum()))
    if not bool(defined.any()):
        return {"mean": None, "std": None, "n": int(array.size), "n_undefined": n_undefined}
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "n": int(array.size),
        "n_undefined": n_undefined,
    }


def summarise_metric(result: Mapping[str, Any]) -> MetricSummary:
    """A metric result with its per-item distribution replaced by that list's mean and std.

    Returns `value`, `n` and `unit` unchanged (the CLAUDE.md §7 contract), plus
    `distribution` naming the key that was summarised (or `None` when the metric attached
    none), `mean` and `std`. `std` is the population standard deviation, matching
    `numpy.std`'s default, and is `0.0` for a single item.

    `mean` and `std` are `None` when there is nothing to average — the metric attached no
    distribution, attached an empty one, or attached one whose every item is undefined.
    `0.0` would be a lie in all three cases, and a silent one: it is a plausible value for
    a ratio, so an empty corpus would show up in `results.json` as a measured zero rather
    than as a missing measurement. `None` survives the JSON round trip as `null`.

    `mean` is not redundant with `value`: the metrics pool their numerator and denominator
    over the whole corpus, so for compression, parity and TPP the mean of the per-item
    ratios is a different — and, for a per-sentence sense of spread, more useful — number.

    The key set is deliberately fixed and narrow, so that a `results.json` diff across runs
    compares numbers rather than schemas; diagnostics a metric attaches beyond its
    distribution (`n_undefined`, the bootstrap CI) are the caller's to record alongside it.
    """
    summary: MetricSummary = {
        "value": float(result["value"]),
        "n": int(result["n"]),
        "unit": str(result["unit"]),
        "distribution": None,
        "mean": None,
        "std": None,
    }
    for key in DISTRIBUTION_KEYS:
        raw: Any = result.get(key)
        if raw is None:
            continue
        summary["distribution"] = key
        stats = summarise_values([float(item) for item in raw])
        mean, std = stats["mean"], stats["std"]
        summary["mean"] = None if mean is None else float(mean)
        summary["std"] = None if std is None else float(std)
        break
    return summary
