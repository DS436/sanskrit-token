"""Experiment 05, one command: run a sweep and aggregate it.

    uv run python experiments/05_lm_training/run.py --sweep smoke   # ~10 min on this laptop
    uv run python experiments/05_lm_training/run.py --sweep sweep   # a GPU job; see README

`--sweep smoke` is the pipeline validation (three arms, one seed, 300 steps at the `smoke`
model size) and is **not a result**: two layers 128 wide, trained on a few hundred thousand
tokens, says only that the plumbing works end to end. `--sweep sweep` is the real grid —
Track 1's seven arms and Track 2's five, three seeds each — and wants a rented GPU.

Both are resumable: re-running the same command skips every run whose `results.json` is
already there under the same plan (`sweep.py`), so an interrupted job is restarted with the
identical command line. Long runs belong under `nohup`:

    nohup uv run python experiments/05_lm_training/run.py --sweep sweep \\
        > outputs/05_lm_training/sweep/run.log 2>&1 &

`--dry-run` forwards to `sweep.py`'s projection table and trains nothing.
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from sanskrit_tok.experiment import load_config, repo_root, resolve_path

logger = logging.getLogger("exp05.run")

HERE = Path(__file__).resolve().parent

#: The two sweeps this experiment defines, by the name `--sweep` takes.
SWEEPS: dict[str, Path] = {
    "smoke": HERE / "smoke.yaml",
    "sweep": HERE / "sweep.yaml",
}


def _load(name: str, path: Path) -> ModuleType:
    """Load a sibling script by path: `experiments/` holds scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable for real files
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweep",
        choices=sorted(SWEEPS),
        default="sweep",
        help="which sweep to run (default: sweep, the real grid)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the projection table and exit; train nothing",
    )
    parser.add_argument(
        "--tokens-per-s",
        type=float,
        default=17_300.0,
        help="dry run only: the token rate the hour projection assumes",
    )
    parser.add_argument(
        "--gpu-speedup",
        type=float,
        default=50.0,
        help="dry run only: the assumed rented-GPU speedup for the second column",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    config_path = SWEEPS[args.sweep]
    sweep = _load("exp05_sweep_module", HERE / "sweep.py")
    if args.dry_run:
        return int(
            sweep.main(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--tokens-per-s",
                    str(args.tokens_per_s),
                    "--gpu-speedup",
                    str(args.gpu_speedup),
                    "--log-level",
                    args.log_level,
                ]
            )
        )

    config = load_config(config_path)
    sweep.run_sweep(config, config_path)

    aggregate = _load("exp05_aggregate_module", HERE / "aggregate.py")
    output_dir = resolve_path(str(config["output_dir"]), repo_root())
    reference = str(config.get("reference_arm", aggregate.DEFAULT_REFERENCE_ARM))
    aggregate.run(output_dir, reference, config)
    logger.info("done: %s", output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
