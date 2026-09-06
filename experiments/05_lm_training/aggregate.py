"""Experiment 05's aggregation: per-seed runs into one `results.json` and two figures.

    uv run python experiments/05_lm_training/aggregate.py \
        --config experiments/05_lm_training/sweep.yaml

Reads every `<output_dir>/<track>/<size>/<arm>/seed<k>/results.json` the sweep wrote, plus
the `curve.jsonl` beside it, and writes one `results.json` in `<output_dir>` with, per
track × size × arm: the final bits-per-character on every evaluation set as mean ± standard
deviation over seeds, the parameter and compute counts, the mean training curve, and the
training bytes, tokens and FLOPs at which the arm first reached the reference arm's final
in-domain BPC.

**The reference is `T1_bpe_raw_64k_dcs`'s final in-domain BPC at the same track and size,
matched seed by seed** (docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and
comparison protocol"). Seed 0 of an arm is compared against seed 0 of the reference, and the
three crossings are then averaged — not the other way round, which would mix a seed's
trajectory with another seed's target. An arm that never reaches the reference has `null`
for that seed and is counted in `n_defined` rather than quietly dropped from the mean.

**Bits per character, never perplexity** (CLAUDE.md §2.2). The one subtlety the numbers
carry is that a split arm is scored on the `_split` twin of each held-out set — the same
sentences with sandhi undone — whose character count is ~6% larger than the raw text's.
The denominators therefore differ slightly between raw and split arms, so `n_chars` is
recorded for every set of every arm and the figures say which shape each arm was scored on.

Standard deviations are the sample standard deviation (ddof 1) and are `null` for a
single-seed group, because one number has no spread; the smoke sweep is such a group and
its figures are drawn without a band.
"""

import argparse
import json
import logging
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sanskrit_tok.experiment import load_config, provenance, repo_root, resolve_path, write_results

logger = logging.getLogger("exp05.aggregate")

EXPERIMENT_NAME = "05_lm_training"

#: The arm whose final in-domain BPC every crossing is measured against.
DEFAULT_REFERENCE_ARM = "T1_bpe_raw_64k_dcs"

#: The base name of the in-domain evaluation set. A split arm's is this plus `_split`.
IN_DOMAIN_BASE = "heldout_dcs"

#: The role given to the in-domain set in `bpc[...]["role"]`; every other set's role is its
#: base name, so a raw arm's `heldout_itihasa_test` and a split arm's
#: `heldout_itihasa_test_split` are recognisably the same evaluation, on two text shapes.
IN_DOMAIN_ROLE = "in_domain"

CURVE_FIGURE_STEM = "bpc_curves"
FINAL_FIGURE_STEM = "bpc_final"

#: Arms whose reported boundaries or splits are not fully gold, per Experiment 04. The
#: oracle arms use gold sandhi splits that no inference-time system provides; the
#: morpheme-constrained arms rest partly on the heuristic stem rule (docs/decisions.md,
#: 2026-09-05, "Stem boundaries are heuristic").
_PROVISIONAL_FAMILIES = ("T5_morphbpe", "T6_morphbpe")


def base_set_name(name: str) -> str:
    """`heldout_dcs_split` -> `heldout_dcs`; a raw set's name is unchanged."""
    return name[: -len("_split")] if name.endswith("_split") else name


def eval_role(name: str) -> str:
    """The comparison slot an evaluation set fills, independent of text shape."""
    base = base_set_name(name)
    return IN_DOMAIN_ROLE if base == IN_DOMAIN_BASE else base


def is_split_set(name: str) -> bool:
    return name.endswith("_split")


def arm_note(arm: str) -> str:
    """A short label for a figure: what has to be said beside this arm's number.

    `oracle` marks an arm trained on gold sandhi splits, which is an upper bound and not a
    deployable system; `provisional` marks the morpheme-constrained arms, whose stem
    boundaries come from a heuristic rule rather than from annotation.
    """
    notes = []
    if "oracle" in arm:
        notes.append("oracle")
    if arm.startswith(_PROVISIONAL_FAMILIES):
        notes.append("provisional")
    return "+".join(notes)


def vocab_label(logical_vocab: int | None) -> str:
    """`64001` -> `64k`, `257` -> `257 (bytes)`, unknown -> `?`."""
    if logical_vocab is None:
        return "?"
    if logical_vocab <= 512:
        return f"{logical_vocab} (bytes)"
    return f"{round(logical_vocab / 1000)}k"


@dataclass(frozen=True)
class RunRecord:
    """One seed's `results.json` and `curve.jsonl`."""

    run_dir: Path
    track: str
    size: str
    arm: str
    seed: int
    results: Mapping[str, Any]
    curve: list[dict[str, Any]]
    sidecar: Mapping[str, Any]

    @property
    def in_domain_set(self) -> str:
        """The evaluation set this arm's in-domain BPC is on: raw or `_split`."""
        for name in self.results.get("bpc") or {}:
            if eval_role(str(name)) == IN_DOMAIN_ROLE:
                return str(name)
        return f"{IN_DOMAIN_BASE}_split" if is_split_arm_name(self.arm) else IN_DOMAIN_BASE

    def final_bpc(self, name: str) -> float | None:
        entry = (self.results.get("bpc") or {}).get(name)
        if not entry:
            return None
        value = entry.get("value")
        return None if value is None else float(value)


def is_split_arm_name(arm: str) -> bool:
    """`sweep.is_split_arm` without importing a sibling script (same rule, same tests)."""
    return "_split_" in f"_{arm}_"


def _read_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_runs(output_dir: Path) -> list[RunRecord]:
    """Every finished run under `output_dir`, in track → size → arm → seed order.

    A directory with no `results.json` is skipped silently: it is a run that has not
    happened yet, or one that is happening right now, and an aggregation over what exists
    is the point — a partially finished sweep still produces a figure.
    """
    records: list[RunRecord] = []
    for results_path in sorted(output_dir.glob("*/*/*/seed*/results.json")):
        run_dir = results_path.parent
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - a half-written results.json
            logger.warning("skipping unreadable %s", results_path)
            continue
        sidecar_path = run_dir / "sweep_run.json"
        sidecar = (
            json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar_path.exists()
            else {}
        )
        track, size, arm, seed_dir = run_dir.parts[-4:]
        records.append(
            RunRecord(
                run_dir=run_dir,
                track=str(results.get("track") or track),
                size=str(results.get("size") or size),
                arm=str(results.get("arm") or arm),
                seed=int(results.get("seed", seed_dir.removeprefix("seed") or 0)),
                results=results,
                curve=_read_curve(run_dir / "curve.jsonl"),
                sidecar=sidecar,
            )
        )
    records.sort(key=lambda record: (record.track, record.size, record.arm, record.seed))
    return records


def mean_std(values: Sequence[float | None]) -> dict[str, Any]:
    """`{mean, std, n, n_defined, values}` over `values`, `None`s kept and counted.

    `None` is a real answer here — an arm that never reached the reference BPC — so it is
    preserved in `values` and excluded from the moments rather than turned into a zero.
    `std` is the sample standard deviation and is `None` for fewer than two defined
    values, because one measurement has no spread to report.
    """
    defined = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.fmean(defined) if defined else None,
        "std": statistics.stdev(defined) if len(defined) > 1 else None,
        "n": len(values),
        "n_defined": len(defined),
        "values": [None if value is None else float(value) for value in values],
    }


def crossing_row(
    curve: Sequence[Mapping[str, Any]], eval_name: str, threshold: float
) -> dict[str, Any] | None:
    """The first curve row whose `eval_name` BPC is at or below `threshold`.

    The curve is in step order, so "first" is the earliest point in training; the row
    carries the tokens, bytes and FLOPs the run had spent by then, which is what
    "tokens to reference BPC" means.
    """
    for row in curve:
        value = (row.get("bpc") or {}).get(eval_name)
        if value is None or not math.isfinite(float(value)):
            continue
        if float(value) <= threshold:
            return dict(row)
    return None


def _mean_curve(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """The seed-mean curve of a group, aligned by step.

    Every seed of a group runs the same configuration and therefore evaluates at the same
    steps; a step missing from one seed is averaged over the seeds that have it, and `n`
    records how many that was.
    """
    by_step: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        for row in record.curve:
            by_step.setdefault(int(row["step"]), []).append(dict(row))
    curve: list[dict[str, Any]] = []
    for step in sorted(by_step):
        rows = by_step[step]
        names = sorted({name for row in rows for name in (row.get("bpc") or {})})
        curve.append(
            {
                "step": step,
                "n": len(rows),
                "tokens_seen": int(statistics.fmean(row["tokens_seen"] for row in rows)),
                "bytes_seen_est": int(
                    statistics.fmean(row.get("bytes_seen_est", 0) for row in rows)
                ),
                "flops_est": statistics.fmean(float(row.get("flops_est", 0.0)) for row in rows),
                "bpc": {
                    name: {
                        key: value
                        for key, value in mean_std(
                            [(row.get("bpc") or {}).get(name) for row in rows]
                        ).items()
                        if key in {"mean", "std", "n_defined"}
                    }
                    for name in names
                },
            }
        )
    return curve


def _identical(records: Sequence[RunRecord], *path: str) -> Any:
    """A value that should be the same across seeds, taken from the first run that has it."""
    for record in records:
        node: Any = record.results
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            return node
    return None


def aggregate(
    records: Sequence[RunRecord], reference_arm: str = DEFAULT_REFERENCE_ARM
) -> dict[str, Any]:
    """Group `records` by track × size × arm and summarise each group over its seeds."""
    groups: dict[tuple[str, str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.track, record.size, record.arm), []).append(record)

    # The reference's final in-domain BPC per (track, size, seed): the target every arm at
    # that track and size is timed against.
    reference: dict[tuple[str, str, int], float] = {}
    for (track, size, arm), members in groups.items():
        if arm != reference_arm:
            continue
        for record in members:
            value = record.final_bpc(record.in_domain_set)
            if value is not None:
                reference[(track, size, record.seed)] = value

    out: list[dict[str, Any]] = []
    for (track, size, arm), members in sorted(groups.items()):
        members = sorted(members, key=lambda record: record.seed)
        in_domain = members[0].in_domain_set
        set_names = sorted(
            {name for record in members for name in (record.results.get("bpc") or {})},
            key=lambda name: (eval_role(name) != IN_DOMAIN_ROLE, name),
        )
        bpc: dict[str, Any] = {}
        for name in set_names:
            summary = mean_std([record.final_bpc(name) for record in members])
            entry = (members[0].results.get("bpc") or {}).get(name) or {}
            summary["role"] = eval_role(name)
            summary["split_text"] = is_split_set(name)
            summary["n_chars"] = entry.get("n")
            summary["unit"] = entry.get("unit", "bits/char")
            bpc[name] = summary

        thresholds: list[float | None] = [
            reference.get((track, size, record.seed)) for record in members
        ]
        crossings = [
            crossing_row(record.curve, in_domain, threshold) if threshold is not None else None
            for record, threshold in zip(members, thresholds, strict=True)
        ]
        to_reference = {
            "reference_arm": reference_arm,
            "eval_set": in_domain,
            "threshold_bpc": mean_std(thresholds),
            "tokens": mean_std(
                [None if row is None else row.get("tokens_seen") for row in crossings]
            ),
            "bytes": mean_std(
                [None if row is None else row.get("bytes_seen_est") for row in crossings]
            ),
            "flops": mean_std(
                [None if row is None else row.get("flops_est") for row in crossings]
            ),
            "steps": mean_std([None if row is None else row.get("step") for row in crossings]),
            "n_defined": sum(1 for row in crossings if row is not None),
        }

        out.append(
            {
                "track": track,
                "size": size,
                "arm": arm,
                "is_reference": arm == reference_arm,
                "is_split_arm": is_split_arm_name(arm),
                "in_domain_set": in_domain,
                "n_seeds": len(members),
                "seeds": [record.seed for record in members],
                "logical_vocab": _identical(members, "model", "logical_vocab"),
                "params": _identical(members, "params"),
                "corpus": _identical(members, "corpus"),
                "budget": {
                    "max_bytes": members[0].sidecar.get("max_bytes"),
                    "bytes_per_token": members[0].sidecar.get("bytes_per_token"),
                    "max_tokens": members[0].sidecar.get("max_tokens"),
                },
                "steps": mean_std([record.results.get("steps") for record in members]),
                "tokens_seen": mean_std(
                    [record.results.get("tokens_seen") for record in members]
                ),
                "bytes_seen_est": mean_std(
                    [record.results.get("bytes_seen_est") for record in members]
                ),
                "flops_est": mean_std([record.results.get("flops_est") for record in members]),
                "tokens_per_s": mean_std(
                    [
                        (record.results.get("throughput") or {}).get("tokens_per_s")
                        for record in members
                    ]
                ),
                "bpc": bpc,
                "to_reference": to_reference,
                "curve": _mean_curve(members),
                "run_dirs": [str(record.run_dir) for record in members],
            }
        )

    return {
        "experiment": EXPERIMENT_NAME,
        "reference_arm": reference_arm,
        "n_runs": len(records),
        "groups": out,
    }


# --------------------------------------------------------------------------- figures


def _panels(results: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (track, size) panels, in the order the groups were written."""
    seen: list[tuple[str, str]] = []
    for group in results["groups"]:
        key = (str(group["track"]), str(group["size"]))
        if key not in seen:
            seen.append(key)
    return seen


def build_curve_figure(results: Mapping[str, Any]) -> Any:
    """BPC against training bytes, one panel per track × size, one line per arm.

    The x axis is *bytes*, not steps or tokens, because bytes is the axis on which the arms
    are matched: at any x, every arm has seen the same amount of text. The shaded band is
    ± one standard deviation over seeds (absent for a single-seed sweep), and the dashed
    horizontal line is the reference arm's final in-domain BPC — the level whose crossing
    `to_reference` reports.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs over ssh on a rented box
    import matplotlib.pyplot as plt

    panels = _panels(results)
    figure = plt.figure(figsize=(6.5 * max(len(panels), 1), 4.6))
    reference_arm = str(results.get("reference_arm", DEFAULT_REFERENCE_ARM))
    for index, (track, size) in enumerate(panels):
        axes = figure.add_subplot(1, max(len(panels), 1), index + 1)
        for group in results["groups"]:
            if (group["track"], group["size"]) != (track, size):
                continue
            curve = group.get("curve") or []
            name = str(group["in_domain_set"])
            xs = [row["bytes_seen_est"] for row in curve if name in (row.get("bpc") or {})]
            means = [
                row["bpc"][name]["mean"]
                for row in curve
                if name in (row.get("bpc") or {})
            ]
            points = [(x, y) for x, y in zip(xs, means, strict=True) if y is not None]
            if not points:
                continue
            xs = [point[0] for point in points]
            means = [point[1] for point in points]
            note = arm_note(str(group["arm"]))
            label = f"{group['arm']}{' (' + note + ')' if note else ''}"
            (line,) = axes.plot(xs, means, marker="o", markersize=3, linewidth=1.4, label=label)
            stds = [
                (row["bpc"][name].get("std") or 0.0)
                for row in curve
                if name in (row.get("bpc") or {}) and row["bpc"][name]["mean"] is not None
            ]
            if any(std > 0 for std in stds):
                axes.fill_between(
                    xs,
                    [mean - std for mean, std in zip(means, stds, strict=True)],
                    [mean + std for mean, std in zip(means, stds, strict=True)],
                    alpha=0.15,
                    color=line.get_color(),
                    linewidth=0,
                )
            if group["arm"] == reference_arm:
                final = group["bpc"].get(name, {}).get("mean")
                if final is not None:
                    axes.axhline(
                        float(final),
                        linestyle="--",
                        linewidth=1.0,
                        color="gray",
                        zorder=1,
                    )
        axes.set_title(f"{track} · {size}", fontsize=10, loc="left")
        axes.set_xlabel("training bytes seen (estimated)", fontsize=8)
        axes.set_ylabel("in-domain BPC (bits/char)", fontsize=8)
        axes.tick_params(labelsize=7)
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(fontsize=6.5, loc="upper right")
    figure.suptitle(
        "Experiment 05 — bits per character against training bytes, mean ± sd over seeds",
        fontsize=11,
    )
    figure.text(
        0.01,
        0.005,
        "Dashed line: the reference arm's final in-domain BPC. Split arms are scored on the "
        "`_split` twin of the held-out set (same sentences, ~6% more characters), so their "
        "denominators differ slightly from the raw arms'. Perplexity is never compared "
        "(CLAUDE.md §2.2).",
        fontsize=5.5,
        wrap=True,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    return figure


def build_final_figure(results: Mapping[str, Any]) -> Any:
    """Final BPC per arm, one panel per (track × size) × evaluation role.

    Bars are the seed mean with a ± one standard deviation error bar; the tick label carries
    the arm's vocabulary and, where one applies, the `oracle` / `provisional` caveat that
    Experiment 04 attached to it. Roles rather than set names label the columns, so a raw
    arm's `heldout_itihasa_test` and a split arm's `heldout_itihasa_test_split` are compared
    in the same panel while `n_chars` in `results.json` records that the denominators differ.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = _panels(results)
    roles: list[str] = []
    for group in results["groups"]:
        for entry in group["bpc"].values():
            if entry["role"] not in roles:
                roles.append(entry["role"])
    roles.sort(key=lambda role: (role != IN_DOMAIN_ROLE, role))
    columns = max(len(roles), 1)
    figure = plt.figure(figsize=(3.4 * columns, 3.4 * max(len(panels), 1)))
    grid = figure.add_gridspec(max(len(panels), 1), columns)

    for row_index, (track, size) in enumerate(panels):
        members = [
            group
            for group in results["groups"]
            if (group["track"], group["size"]) == (track, size)
        ]
        labels = [
            f"{group['arm']} [{vocab_label(group.get('logical_vocab'))}]"
            + (f" {arm_note(str(group['arm']))}" if arm_note(str(group["arm"])) else "")
            for group in members
        ]
        for column, role in enumerate(roles):
            axes = figure.add_subplot(grid[row_index, column])
            values: list[float] = []
            errors: list[float] = []
            for group in members:
                entry = next(
                    (item for item in group["bpc"].values() if item["role"] == role), None
                )
                values.append(math.nan if entry is None or entry["mean"] is None else entry["mean"])
                errors.append(0.0 if entry is None or entry.get("std") is None else entry["std"])
            positions = np.arange(len(members), dtype=float)
            axes.barh(
                positions,
                values,
                xerr=errors,
                color="#2b6cb0" if role == IN_DOMAIN_ROLE else "#c05621",
                height=0.62,
                error_kw={"elinewidth": 0.9, "capsize": 2},
                zorder=2,
            )
            axes.set_yticks(positions)
            axes.set_yticklabels(labels if column == 0 else ["" for _ in members], fontsize=6)
            axes.invert_yaxis()
            axes.set_xlabel("BPC (bits/char)", fontsize=7)
            axes.tick_params(labelsize=6)
            role_label = "in-domain (DCS held out)" if role == IN_DOMAIN_ROLE else role
            axes.set_title(f"{track} · {size} · {role_label}", fontsize=7, loc="left")
            axes.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Experiment 05 — final bits per character per arm, mean ± sd over seeds", fontsize=11
    )
    figure.text(
        0.01,
        0.004,
        "`oracle`: trained on gold sandhi splits, an upper bound rather than a deployable "
        "system. `provisional`: morpheme constraints resting partly on the heuristic stem "
        "rule (Experiment 04). Lower is better.",
        fontsize=5.5,
        wrap=True,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))
    return figure


def make_figures(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Write `bpc_curves` and `bpc_final` as PDF and PNG; returns the four paths."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for stem, builder in (
        (CURVE_FIGURE_STEM, build_curve_figure),
        (FINAL_FIGURE_STEM, build_final_figure),
    ):
        figure = builder(results)
        for suffix in (".pdf", ".png"):
            path = out_dir / f"{stem}{suffix}"
            figure.savefig(path, dpi=200)
            paths.append(path)
        plt.close(figure)
    logger.info("wrote %s", ", ".join(str(path) for path in paths))
    return paths


# ------------------------------------------------------------------------------ main


def run(
    output_dir: Path, reference_arm: str, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Aggregate `output_dir`, write its `results.json` and both figures."""
    records = load_runs(output_dir)
    if not records:
        raise SystemExit(f"{output_dir}: no results.json under <track>/<size>/<arm>/seed*/")
    results = aggregate(records, reference_arm)
    results["output_dir"] = str(output_dir)
    results["config"] = dict(config or {})
    results.update(provenance())
    write_results(results, output_dir)
    make_figures(results, output_dir)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("sweep.yaml"),
        help="the sweep config whose output_dir is aggregated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="aggregate this directory instead of the config's output_dir",
    )
    parser.add_argument(
        "--reference-arm",
        default=None,
        help=f"arm the crossings are measured against (default: the config's, else "
        f"{DEFAULT_REFERENCE_ARM})",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    config = load_config(args.config)
    output_dir = args.output_dir or resolve_path(str(config["output_dir"]), repo_root())
    reference = args.reference_arm or str(config.get("reference_arm", DEFAULT_REFERENCE_ARM))
    run(output_dir, reference, config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
