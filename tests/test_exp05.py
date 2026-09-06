"""Tests for the Experiment 05 sweep runner, aggregation and figures.

Everything here is offline, synthetic and second-scale. No model is trained and no corpus
is encoded: the sweep is exercised through its *plan* (which runs exist, which corpus and
which evaluation sets each one gets, what its byte budget converts to in tokens, which runs
are already done) and the aggregation through hand-written `results.json` / `curve.jsonl`
trees whose means, standard deviations and reference crossings are arithmetic a reader can
check by eye.

Three things are deliberately pinned harder than the rest, because each of them silently
changes a published number rather than raising:

- **Split arms get split text.** `T4_bpe_split_64k_oracle_dcs` and
  `T6_morphbpe_split_64k_dcs` train on `track1_split.txt` and are evaluated on the `_split`
  twin of every held-out set. An arm evaluated on the other shape would be reported beside
  the raw arms as if it were comparable.
- **The byte budget is one number per track/size**, and each arm converts it to tokens with
  its own bytes-per-token on its own text (docs/decisions.md, 2026-09-06).
- **The reference crossing matches seeds by index** and is `null`, not omitted and not
  silently dropped from the mean, when an arm never reaches the reference BPC.

`sweep.py` and `aggregate.py` are scripts, not package modules (`experiments/` holds no
package), so they are loaded by path as every other experiment test does.
"""

import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from sanskrit_tok.lm.config import MODEL_SIZES
from sanskrit_tok.tokenizers.registry import LoadedTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "05_lm_training"
SWEEP_PY = EXP_DIR / "sweep.py"
AGGREGATE_PY = EXP_DIR / "aggregate.py"
RUN_PY = EXP_DIR / "run.py"
SWEEP_YAML = EXP_DIR / "sweep.yaml"
SMOKE_YAML = EXP_DIR / "smoke.yaml"

TRACK1_ARMS = [
    "T1_bpe_raw_64k_dcs",
    "T2_unigram_raw_64k_dcs",
    "T4_bpe_split_64k_oracle_dcs",
    "T5_morphbpe_rawseg_64k_dcs",
    "T5_morphbpe_raw_64k_dcs",
    "T6_morphbpe_split_64k_dcs",
    "T7_byt5",
]
TRACK2_ARMS = [
    "T1_bpe_raw_64k_dcs",
    "T2_unigram_raw_64k_dcs",
    "T5_morphbpe_rawseg_64k_dcs",
    "T5_morphbpe_raw_64k_dcs",
    "T7_byt5",
]
OOD_SETS = [
    "heldout_samayik_test",
    "heldout_samayik_test_ood",
    "heldout_itihasa_test",
    "heldout_flores_devtest",
]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load("exp05_sweep", SWEEP_PY)
aggregate = _load("exp05_aggregate", AGGREGATE_PY)


# ------------------------------------------------------------------ synthetic fixtures


def _write_corpora(root: Path) -> None:
    """A miniature `data/processed/lm/` tree: two Track 1 corpora, one Track 2, ten evals."""
    lm_dir = root / "data" / "processed" / "lm"
    lm_dir.mkdir(parents=True)
    (lm_dir / "track1_raw.txt").write_text("tadapi\nrAmaH\n", encoding="utf-8")
    (lm_dir / "track1_split.txt").write_text("tat api\nrAmaH\n", encoding="utf-8")
    (lm_dir / "track2_sample.txt").write_text("tadapi\nrAmaH\nvanam\n", encoding="utf-8")
    for name in ["heldout_dcs", *OOD_SETS]:
        (lm_dir / f"{name}.txt").write_text("gacCati\n", encoding="utf-8")
        (lm_dir / f"{name}_split.txt").write_text("gacCa ti\n", encoding="utf-8")


def _sweep_config(root: Path) -> dict[str, Any]:
    """A two-track sweep config over the miniature corpora, small enough to read."""
    return {
        "name": "testsweep",
        "output_dir": "outputs/test/sweep",
        "cache_dir": "outputs/test/encoded",
        "eval_dir": "data/processed/lm",
        "reference_arm": "T1_bpe_raw_64k_dcs",
        "seeds": [0, 1, 2],
        "device": "auto",
        "dtype": "auto",
        "keep_checkpoints": False,
        "eval_sets": {"in_domain": "heldout_dcs", "ood": list(OOD_SETS)},
        "sizes": {
            "50M": {"batch_size": 16, "grad_accum": 4, "lr": 6.0e-4, "eval_every": 100},
            "125M": {"batch_size": 8, "grad_accum": 8, "lr": 6.0e-4, "eval_every": 250},
        },
        "tracks": {
            "track1": {
                "corpus_raw": "data/processed/lm/track1_raw.txt",
                "corpus_split": "data/processed/lm/track1_split.txt",
                "epochs": 8,
                "arms": list(TRACK1_ARMS),
                "sizes": ["50M"],
            },
            "track2": {
                "corpus_raw": "data/processed/lm/track2_sample.txt",
                "epochs": 1,
                "arms": list(TRACK2_ARMS),
                "sizes": ["50M", "125M"],
            },
        },
    }


@pytest.fixture()
def sweep_root(tmp_path: Path) -> Iterator[Path]:
    _write_corpora(tmp_path)
    yield tmp_path


def _char_tokenizer(name: str, vocab_size: int = 64000) -> LoadedTokenizer:
    """One token per non-space character, so a token count is a length."""
    return LoadedTokenizer(
        name=name,
        source_id=f"fake/{name}",
        vocab_size=vocab_size,
        _encode=lambda text: [1] * len(text.replace(" ", "")),
    )


# ------------------------------------------------------------------------- enumeration


def test_is_split_arm_separates_the_two_split_arms_from_the_five_raw_ones() -> None:
    assert sweep.is_split_arm("T4_bpe_split_64k_oracle_dcs")
    assert sweep.is_split_arm("T6_morphbpe_split_64k_dcs")
    for arm in TRACK2_ARMS:
        assert not sweep.is_split_arm(arm), arm


def test_enumerate_runs_covers_every_track_size_arm_and_seed(sweep_root: Path) -> None:
    specs = sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
    # Track 1: 7 arms x 1 size x 3 seeds. Track 2: 5 arms x 2 sizes x 3 seeds.
    assert len(specs) == 7 * 1 * 3 + 5 * 2 * 3
    assert {(spec.track, spec.size) for spec in specs} == {
        ("track1", "50M"),
        ("track2", "50M"),
        ("track2", "125M"),
    }
    assert {spec.seed for spec in specs} == {0, 1, 2}
    assert {spec.arm for spec in specs if spec.track == "track1"} == set(TRACK1_ARMS)
    assert {spec.arm for spec in specs if spec.track == "track2"} == set(TRACK2_ARMS)
    # One directory per run, all distinct.
    assert len({spec.run_dir for spec in specs}) == len(specs)


def test_split_arms_train_on_split_text_and_are_evaluated_on_split_held_out_sets(
    sweep_root: Path,
) -> None:
    specs = sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
    by_arm = {spec.arm: spec for spec in specs if spec.track == "track1" and spec.seed == 0}

    split = by_arm["T6_morphbpe_split_64k_dcs"]
    assert split.corpus.name == "track1_split.txt"
    assert split.in_domain_set == "heldout_dcs_split"
    assert set(split.eval_sets) == {"heldout_dcs_split", *(f"{n}_split" for n in OOD_SETS)}
    assert all(path.name.endswith("_split.txt") for path in split.eval_sets.values())

    raw = by_arm["T1_bpe_raw_64k_dcs"]
    assert raw.corpus.name == "track1_raw.txt"
    assert raw.in_domain_set == "heldout_dcs"
    assert set(raw.eval_sets) == {"heldout_dcs", *OOD_SETS}


def test_track2_has_no_split_arm_and_every_arm_reads_the_sample(sweep_root: Path) -> None:
    specs = [s for s in sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
             if s.track == "track2"]
    assert not any(sweep.is_split_arm(spec.arm) for spec in specs)
    assert {spec.corpus.name for spec in specs} == {"track2_sample.txt"}


def test_run_dir_is_track_size_arm_seed(sweep_root: Path) -> None:
    spec = next(
        s
        for s in sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
        if s.track == "track2" and s.size == "125M" and s.arm == "T7_byt5" and s.seed == 2
    )
    assert spec.run_dir.parts[-4:] == ("track2", "125M", "T7_byt5", "seed2")


# ------------------------------------------------------------- equal bytes -> tokens


def test_max_bytes_is_the_raw_corpus_bytes_times_epochs(sweep_root: Path) -> None:
    config = _sweep_config(sweep_root)
    specs = sweep.enumerate_runs(config, sweep_root)
    # "tadapi" + "rAmaH" = 11 bytes of text; the newlines are not part of it.
    track1 = [spec for spec in specs if spec.track == "track1"]
    assert {spec.max_bytes for spec in track1} == {11 * 8}
    # The split arm gets the SAME byte figure, although its own text is longer.
    split = next(spec for spec in track1 if sweep.is_split_arm(spec.arm))
    assert split.max_bytes == 11 * 8
    # "tadapi" + "rAmaH" + "vanam" = 16 bytes, one epoch.
    track2 = [spec for spec in specs if spec.track == "track2"]
    assert {spec.max_bytes for spec in track2} == {16 * 1}


def test_token_budget_converts_bytes_with_the_arms_own_bytes_per_token() -> None:
    # 252,284,232 bytes at 4.6309 bytes/token is 54,478,... tokens; the conversion rounds.
    assert sweep.token_budget(1000, 4.0) == 250
    assert sweep.token_budget(1000, 3.0) == 333
    assert sweep.token_budget(1001, 3.0) == 334
    with pytest.raises(ValueError):
        sweep.token_budget(1000, 0.0)


def test_a_denser_arm_gets_fewer_tokens_for_the_same_bytes() -> None:
    """The whole point of the equal-bytes budget: more tokens per byte, more tokens."""
    plain = sweep.token_budget(1_000_000, 4.63)
    constrained = sweep.token_budget(1_000_000, 3.50)
    assert constrained > plain


def test_corpus_bytes_reads_the_manifest_when_there_is_one(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text("abc\nde\n", encoding="utf-8")
    assert sweep.corpus_bytes(corpus) == 5
    # A manifest beside it wins, so a 3.7 GB corpus is not re-read on every invocation.
    (tmp_path / "c.manifest.json").write_text(json.dumps({"n_bytes": 999}), encoding="utf-8")
    assert sweep.corpus_bytes(corpus) == 999


def test_sample_bytes_per_token_strides_the_corpus_and_counts_the_eos(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text("abcd\nzz\nefgh\nyy\n", encoding="utf-8")
    arm = _char_tokenizer("fake")
    # stride 2 takes lines 0 and 2: 8 bytes, 8 characters + 2 EOS = 10 tokens.
    value, n_lines = sweep.sample_bytes_per_token(arm, corpus, stride=2)
    assert n_lines == 2
    assert value == pytest.approx(8 / 10)


# --------------------------------------------------------------------- skip / resume


def _finish(spec: Any, *, plan_hash: str | None = None) -> None:
    """Write the two files that make a run count as finished."""
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "results.json").write_text("{}", encoding="utf-8")
    (spec.run_dir / "sweep_run.json").write_text(
        json.dumps({"plan_hash": plan_hash if plan_hash is not None else spec.plan_hash()}),
        encoding="utf-8",
    )


def test_pending_runs_skips_a_finished_run(sweep_root: Path) -> None:
    specs = sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
    _finish(specs[0])
    pending = sweep.pending_runs(specs)
    assert specs[0] not in pending
    assert len(pending) == len(specs) - 1


def test_a_run_with_results_but_no_sidecar_is_not_treated_as_finished(
    sweep_root: Path,
) -> None:
    specs = sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
    specs[0].run_dir.mkdir(parents=True)
    (specs[0].run_dir / "results.json").write_text("{}", encoding="utf-8")
    assert specs[0] in sweep.pending_runs(specs)


def test_a_changed_plan_reruns_the_run(sweep_root: Path) -> None:
    specs = sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)
    _finish(specs[0], plan_hash="a-hash-from-a-different-config")
    assert specs[0] in sweep.pending_runs(specs)


def test_the_plan_hash_moves_with_the_hyperparameters(sweep_root: Path) -> None:
    config = _sweep_config(sweep_root)
    before = sweep.enumerate_runs(config, sweep_root)[0].plan_hash()
    config["sizes"]["50M"]["lr"] = 3.0e-4
    after = sweep.enumerate_runs(config, sweep_root)[0].plan_hash()
    assert before != after


def test_prune_checkpoint_deletes_unless_keeping(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"weights")
    assert sweep.prune_checkpoint(tmp_path, keep=True) is False
    assert ckpt.exists()
    assert sweep.prune_checkpoint(tmp_path, keep=False) is True
    assert not ckpt.exists()
    # Idempotent: a second prune of a directory with no checkpoint is not an error.
    assert sweep.prune_checkpoint(tmp_path, keep=False) is False


# ------------------------------------------------------------------------- dry run


def _plan_rows() -> list[Any]:
    return [
        sweep.PlanRow(
            track="track1",
            size="50M",
            arm="T1_bpe_raw_64k_dcs",
            seed=seed,
            bytes_per_token=4.0,
            max_bytes=4000,
            max_tokens=1000,
            params=1_000_000,
            flops_est=6.0 * 1_000_000 * 1000,
            estimated=False,
        )
        for seed in (0, 1)
    ] + [
        sweep.PlanRow(
            track="track2",
            size="125M",
            arm="T7_byt5",
            seed=0,
            bytes_per_token=1.0,
            max_bytes=4000,
            max_tokens=4000,
            params=2_000_000,
            flops_est=6.0 * 2_000_000 * 4000,
            estimated=True,
        )
    ]


def test_dry_run_table_lists_every_run_with_both_hour_columns() -> None:
    table = sweep.format_dry_run_table(_plan_rows(), tokens_per_s=1000.0, gpu_speedup=50.0)
    assert table.count("T1_bpe_raw_64k_dcs") == 2
    assert "T7_byt5" in table
    # 1000 tokens at 1000 tok/s is 1 second = 0.000278 h; 4000 tokens is 4 s.
    assert "track1" in table and "track2" in table
    assert "x50" in table or "×50" in table
    assert "assumption" in table.lower()


def test_dry_run_table_totals_per_track_and_overall() -> None:
    table = sweep.format_dry_run_table(_plan_rows(), tokens_per_s=1000.0, gpu_speedup=50.0)
    lines = [line for line in table.splitlines() if "TOTAL" in line.upper()]
    # one per track plus the grand total
    assert len(lines) == 3
    assert any("track1" in line for line in lines)
    assert any("track2" in line for line in lines)


def test_dry_run_hours_scale_with_the_token_rate() -> None:
    rows = _plan_rows()
    fast = sweep.total_hours(rows, tokens_per_s=2000.0)
    slow = sweep.total_hours(rows, tokens_per_s=1000.0)
    assert slow == pytest.approx(2 * fast)
    assert slow == pytest.approx((1000 + 1000 + 4000) / 1000.0 / 3600.0)


def test_estimated_rows_are_marked_in_the_table() -> None:
    table = sweep.format_dry_run_table(_plan_rows(), tokens_per_s=1000.0, gpu_speedup=50.0)
    byt5_line = next(line for line in table.splitlines() if "T7_byt5" in line)
    exact_line = next(line for line in table.splitlines() if "T1_bpe_raw_64k_dcs" in line)
    assert "*" in byt5_line
    assert "*" not in exact_line


# ------------------------------------------------------------------------ aggregation


def _run_results(
    *,
    track: str,
    size: str,
    arm: str,
    seed: int,
    bpc: dict[str, float],
    n_chars: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "05_lm_training",
        "arm": arm,
        "track": track,
        "size": size,
        "seed": seed,
        "steps": 2,
        "tokens_seen": 1000,
        "bytes_seen_est": 4000,
        "flops_est": 6.0e9,
        "params": {"non_embedding": 500, "total": 1000, "flops_params_kind": "total"},
        "model": {"logical_vocab": 257 if arm == "T7_byt5" else 64001},
        "corpus": {"bytes_per_token": 4.0, "n_bytes": 4000, "n_tokens": 1000},
        "bpc": {
            name: {
                "value": value,
                "n": (n_chars or {}).get(name, 100),
                "unit": "bits/char",
            }
            for name, value in bpc.items()
        },
        "throughput": {"tokens_per_s": 100.0, "elapsed_s": 10.0},
        "hardware": {"device": "cpu", "device_name": "test"},
    }


def _write_run(
    out_dir: Path,
    *,
    track: str,
    size: str,
    arm: str,
    seed: int,
    final_bpc: dict[str, float],
    curve: list[dict[str, Any]] | None = None,
    n_chars: dict[str, int] | None = None,
) -> Path:
    run_dir = out_dir / track / size / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            _run_results(
                track=track, size=size, arm=arm, seed=seed, bpc=final_bpc, n_chars=n_chars
            )
        ),
        encoding="utf-8",
    )
    rows = curve if curve is not None else []
    (run_dir / "curve.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run_dir


def _curve(points: list[tuple[int, float]], name: str = "heldout_dcs") -> list[dict[str, Any]]:
    """`[(bytes_seen, bpc), ...]` as curve rows with matching token and FLOP counts."""
    return [
        {
            "step": index,
            "tokens_seen": byts // 4,
            "bytes_seen_est": byts,
            "flops_est": 6.0e6 * byts,
            "bpc": {name: value},
        }
        for index, (byts, value) in enumerate(points)
    ]


def test_aggregate_reports_mean_and_std_over_seeds(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    for seed, value in zip((0, 1, 2), (3.0, 4.0, 5.0), strict=True):
        _write_run(
            out,
            track="track1",
            size="50M",
            arm="T1_bpe_raw_64k_dcs",
            seed=seed,
            final_bpc={"heldout_dcs": value},
            curve=_curve([(100, value + 1.0), (200, value)]),
        )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    group = results["groups"][0]
    assert group["arm"] == "T1_bpe_raw_64k_dcs"
    assert group["n_seeds"] == 3
    assert group["seeds"] == [0, 1, 2]
    entry = group["bpc"]["heldout_dcs"]
    assert entry["mean"] == pytest.approx(4.0)
    assert entry["std"] == pytest.approx(1.0)  # sample std of 3,4,5
    assert entry["values"] == [3.0, 4.0, 5.0]
    assert entry["role"] == "in_domain"
    assert entry["n_chars"] == 100


def test_a_single_seed_has_no_standard_deviation(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    _write_run(
        out,
        track="track1",
        size="smoke",
        arm="T1_bpe_raw_64k_dcs",
        seed=0,
        final_bpc={"heldout_dcs": 2.85},
        curve=_curve([(100, 2.85)]),
    )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    entry = results["groups"][0]["bpc"]["heldout_dcs"]
    assert entry["mean"] == pytest.approx(2.85)
    assert entry["std"] is None
    assert entry["n"] == 1


def test_split_arms_are_reported_under_their_own_in_domain_set_and_role(
    tmp_path: Path,
) -> None:
    out = tmp_path / "sweep"
    _write_run(
        out,
        track="track1",
        size="50M",
        arm="T6_morphbpe_split_64k_dcs",
        seed=0,
        final_bpc={"heldout_dcs_split": 2.5, "heldout_samayik_test_split": 3.5},
        curve=_curve([(100, 2.5)], name="heldout_dcs_split"),
        n_chars={"heldout_dcs_split": 106},
    )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    group = results["groups"][0]
    assert group["is_split_arm"] is True
    assert group["in_domain_set"] == "heldout_dcs_split"
    assert group["bpc"]["heldout_dcs_split"]["role"] == "in_domain"
    assert group["bpc"]["heldout_dcs_split"]["n_chars"] == 106
    assert group["bpc"]["heldout_samayik_test_split"]["role"] == "heldout_samayik_test"


def test_reference_crossing_uses_the_reference_arms_own_seed(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    # Reference finals: 3.0 (seed 0) and 2.0 (seed 1).
    for seed, value in zip((0, 1), (3.0, 2.0), strict=True):
        _write_run(
            out,
            track="track1",
            size="50M",
            arm="T1_bpe_raw_64k_dcs",
            seed=seed,
            final_bpc={"heldout_dcs": value},
            curve=_curve([(100, 5.0), (200, value)]),
        )
    # The candidate reaches 3.0 at 100 bytes and 2.0 at 200 bytes, in both seeds.
    for seed in (0, 1):
        _write_run(
            out,
            track="track1",
            size="50M",
            arm="T5_morphbpe_raw_64k_dcs",
            seed=seed,
            final_bpc={"heldout_dcs": 1.5},
            curve=_curve([(100, 3.0), (200, 2.0), (300, 1.5)]),
        )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    group = next(g for g in results["groups"] if g["arm"] == "T5_morphbpe_raw_64k_dcs")
    crossing = group["to_reference"]
    assert crossing["threshold_bpc"]["values"] == [3.0, 2.0]
    assert crossing["bytes"]["values"] == [100, 200]
    assert crossing["bytes"]["mean"] == pytest.approx(150.0)
    assert crossing["tokens"]["values"] == [25, 50]
    assert crossing["n_defined"] == 2


def test_reference_crossing_is_null_when_the_reference_is_never_reached(
    tmp_path: Path,
) -> None:
    out = tmp_path / "sweep"
    _write_run(
        out,
        track="track1",
        size="50M",
        arm="T1_bpe_raw_64k_dcs",
        seed=0,
        final_bpc={"heldout_dcs": 2.0},
        curve=_curve([(100, 3.0), (200, 2.0)]),
    )
    _write_run(
        out,
        track="track1",
        size="50M",
        arm="T2_unigram_raw_64k_dcs",
        seed=0,
        final_bpc={"heldout_dcs": 2.5},
        curve=_curve([(100, 3.5), (200, 2.5)]),
    )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    group = next(g for g in results["groups"] if g["arm"] == "T2_unigram_raw_64k_dcs")
    crossing = group["to_reference"]
    assert crossing["bytes"]["values"] == [None]
    assert crossing["bytes"]["mean"] is None
    assert crossing["n_defined"] == 0


def test_the_reference_arm_crosses_itself_at_its_own_final_point(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    _write_run(
        out,
        track="track1",
        size="50M",
        arm="T1_bpe_raw_64k_dcs",
        seed=0,
        final_bpc={"heldout_dcs": 2.0},
        curve=_curve([(100, 3.0), (200, 2.0)]),
    )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    group = results["groups"][0]
    assert group["is_reference"] is True
    assert group["to_reference"]["bytes"]["values"] == [200]


def test_a_missing_reference_arm_leaves_every_crossing_null(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    _write_run(
        out,
        track="track2",
        size="50M",
        arm="T7_byt5",
        seed=0,
        final_bpc={"heldout_dcs": 2.0},
        curve=_curve([(100, 2.0)]),
    )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    group = results["groups"][0]
    assert group["to_reference"]["bytes"]["mean"] is None
    assert group["to_reference"]["threshold_bpc"]["values"] == [None]


def test_groups_carry_the_mean_curve_the_figure_is_drawn_from(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    for seed, offset in zip((0, 1), (0.0, 1.0), strict=True):
        _write_run(
            out,
            track="track1",
            size="50M",
            arm="T1_bpe_raw_64k_dcs",
            seed=seed,
            final_bpc={"heldout_dcs": 2.0 + offset},
            curve=_curve([(100, 3.0 + offset), (200, 2.0 + offset)]),
        )
    results = aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")
    curve = results["groups"][0]["curve"]
    assert [row["bytes_seen_est"] for row in curve] == [100, 200]
    assert curve[0]["bpc"]["heldout_dcs"]["mean"] == pytest.approx(3.5)
    assert curve[0]["bpc"]["heldout_dcs"]["std"] == pytest.approx(math.sqrt(0.5))


def test_load_runs_ignores_a_directory_with_no_results(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    _write_run(
        out,
        track="track1",
        size="50M",
        arm="T1_bpe_raw_64k_dcs",
        seed=0,
        final_bpc={"heldout_dcs": 2.0},
        curve=_curve([(100, 2.0)]),
    )
    (out / "track1" / "50M" / "T2_unigram_raw_64k_dcs" / "seed0").mkdir(parents=True)
    assert [record.arm for record in aggregate.load_runs(out)] == ["T1_bpe_raw_64k_dcs"]


# --------------------------------------------------------------------------- figures


def _aggregated(tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / "sweep"
    for arm, base in (("T1_bpe_raw_64k_dcs", 3.0), ("T5_morphbpe_raw_64k_dcs", 2.8)):
        for seed in (0, 1):
            _write_run(
                out,
                track="track1",
                size="50M",
                arm=arm,
                seed=seed,
                final_bpc={"heldout_dcs": base + 0.1 * seed, "heldout_samayik_test": base + 1.0},
                curve=_curve([(100, base + 1.0), (200, base + 0.1 * seed)]),
            )
    return aggregate.aggregate(aggregate.load_runs(out), reference_arm="T1_bpe_raw_64k_dcs")


def test_make_figures_writes_both_figures_as_pdf_and_png(tmp_path: Path) -> None:
    results = _aggregated(tmp_path)
    paths = aggregate.make_figures(results, tmp_path / "figs")
    names = sorted(path.name for path in paths)
    assert names == [
        "bpc_curves.pdf",
        "bpc_curves.png",
        "bpc_final.pdf",
        "bpc_final.png",
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_the_curve_figure_has_one_panel_per_track_and_size(tmp_path: Path) -> None:
    results = _aggregated(tmp_path)
    figure = aggregate.build_curve_figure(results)
    try:
        assert len(figure.axes) == 1
        titles = [axes.get_title(loc="left") for axes in figure.axes]
        assert any("track1" in title and "50M" in title for title in titles)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_the_bar_figure_labels_arms_with_their_vocabulary(tmp_path: Path) -> None:
    results = _aggregated(tmp_path)
    figure = aggregate.build_final_figure(results)
    try:
        labels = [
            text.get_text()
            for axes in figure.axes
            for text in axes.get_xticklabels() + axes.get_yticklabels()
        ]
        joined = " ".join(labels)
        assert "T1_bpe_raw_64k_dcs" in joined
        assert "64k" in joined or "64001" in joined
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_arm_label_marks_the_oracle_and_provisional_arms() -> None:
    assert "oracle" in aggregate.arm_note("T4_bpe_split_64k_oracle_dcs")
    assert "provisional" in aggregate.arm_note("T6_morphbpe_split_64k_dcs")
    assert aggregate.arm_note("T1_bpe_raw_64k_dcs") == ""


# ---------------------------------------------------------------- the real configs


def test_the_real_sweep_config_enumerates_the_planned_grid() -> None:
    config = yaml.safe_load(SWEEP_YAML.read_text(encoding="utf-8"))
    specs = sweep.enumerate_runs(config, REPO_ROOT)
    assert config["seeds"] == [0, 1, 2]
    assert config["reference_arm"] == "T1_bpe_raw_64k_dcs"
    assert config["tracks"]["track1"]["arms"] == TRACK1_ARMS
    assert config["tracks"]["track2"]["arms"] == TRACK2_ARMS
    assert len(specs) == 7 * 3 + 5 * 2 * 3
    for spec in specs:
        assert spec.settings["block_size"] == 1024
        assert spec.max_bytes is not None and spec.max_bytes > 0
        assert len(spec.eval_sets) == 5


def test_the_real_smoke_config_is_three_arms_one_seed_and_capped_evaluation() -> None:
    config = yaml.safe_load(SMOKE_YAML.read_text(encoding="utf-8"))
    specs = sweep.enumerate_runs(config, REPO_ROOT)
    assert config["seeds"] == [0]
    assert len(specs) == 3
    assert {spec.arm for spec in specs} == {
        "T1_bpe_raw_64k_dcs",
        "T5_morphbpe_rawseg_64k_dcs",
        "T7_byt5",
    }
    for spec in specs:
        assert spec.size == "smoke"
        assert spec.settings["max_steps"] == 300
        assert spec.settings["block_size"] == 256
        assert spec.settings["batch_size"] == 8
        assert spec.settings["eval_every"] == 100
        assert spec.settings["eval_max_chars"] == 200_000
        assert set(spec.eval_sets) == {"heldout_dcs", "heldout_samayik_test"}
        # A step budget, not a byte budget: the smoke run is a pipeline test.
        assert spec.max_bytes is None


def test_every_size_named_by_the_real_configs_exists() -> None:
    for path in (SWEEP_YAML, SMOKE_YAML):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for track in config["tracks"].values():
            for size in track["sizes"]:
                assert size in MODEL_SIZES, f"{path.name}: {size}"
                assert size in config["sizes"], f"{path.name}: {size}"


def test_train_mapping_is_accepted_by_train_config(sweep_root: Path) -> None:
    from sanskrit_tok.lm.config import TrainConfig

    spec = sweep.enumerate_runs(_sweep_config(sweep_root), sweep_root)[0]
    config = TrainConfig.from_mapping(spec.train_mapping(max_tokens=1234), sweep_root)
    assert config.arm == spec.arm
    assert config.max_tokens == 1234
    assert config.seed == spec.seed
    assert config.run_dir == spec.run_dir


def test_run_py_exposes_both_sweep_names() -> None:
    module = _load("exp05_run", RUN_PY)
    assert set(module.SWEEPS) == {"smoke", "sweep"}
    assert module.SWEEPS["smoke"].name == "smoke.yaml"
    assert module.SWEEPS["sweep"].name == "sweep.yaml"
