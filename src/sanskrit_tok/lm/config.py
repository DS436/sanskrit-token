"""What a training run is: model size, optimisation budget, device, and where it writes.

`TrainConfig` is the whole of a run's identity. It is built from a YAML mapping (or from a
sweep's generated one), copied verbatim into the run's `config.yaml`, and recorded inside
its `results.json`, so the file beside a bits-per-character number always says what produced
it (CLAUDE.md §9).

The three model sizes are fixed here rather than in a config file because they are a
research decision, not a knob: 50M and 125M are the two sizes Experiment 05 reports, counted
as *non-embedding* parameters so the transformer body is identical across arms while the
embedding table grows with the vocabulary (docs/decisions.md, 2026-09-05, "Experiment 05
model, evaluation and comparison protocol"). `smoke` is not a research size — two layers,
128 wide — and exists only to prove the pipeline runs end to end on this machine's MPS
before any GPU is rented.

`resolve_device` and `resolve_dtype` are the reason "runs on my laptop" and "runs on a
rented A100" are the same command: `device: auto` takes CUDA if there is one, then MPS, then
CPU, and `dtype: auto` picks bfloat16 only where it is known good. MPS gets float32 —
half-precision on Metal is not something this project has validated, and a silently NaN'd
BPC is worse than a slower smoke run.

**Neither resolution ever silently downgrades.** A run that asked for `cuda` on a box with
no CUDA used to fall back to CPU with a warning; on a rented GPU that turns a mis-set
`CUDA_VISIBLE_DEVICES` into a sweep that appears to be running and finishes in three weeks.
`resolve_device` now raises, and `auto` is allowed to land on CPU only when the config says
`allow_cpu: true` — which the smoke sweep sets and the real sweep does not
(docs/decisions.md, 2026-09-06, "Sweep hardening before paid GPU time").
"""

import logging
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sanskrit_tok.experiment import resolve_path

__all__ = [
    "MODEL_SIZES",
    "ModelSize",
    "TrainConfig",
    "VOCAB_PAD_MULTIPLE",
    "device_display_name",
    "resolve_device",
    "resolve_dtype",
    "round_up_vocab",
]

logger = logging.getLogger(__name__)

#: nanoGPT pads GPT-2's 50,257-token vocabulary to 50,304 for the same reason: matrix
#: shapes that are multiples of 64 hit faster kernels. The padding rows are real parameters
#: that get gradient from the softmax denominator, but no id at or above the *logical*
#: vocabulary is ever emitted or scored, so they cost throughput, not correctness.
VOCAB_PAD_MULTIPLE = 64


@dataclass(frozen=True)
class ModelSize:
    """One row of `MODEL_SIZES`: the transformer body and the context it is trained at."""

    name: str
    n_layer: int
    n_embd: int
    n_head: int
    block_size: int


#: The three configurations Experiment 05 uses. `smoke` is a pipeline test, not a result.
MODEL_SIZES: dict[str, ModelSize] = {
    "smoke": ModelSize("smoke", n_layer=2, n_embd=128, n_head=4, block_size=256),
    "50M": ModelSize("50M", n_layer=8, n_embd=512, n_head=8, block_size=1024),
    "125M": ModelSize("125M", n_layer=12, n_embd=768, n_head=12, block_size=1024),
}


def round_up_vocab(logical_vocab: int, multiple: int = VOCAB_PAD_MULTIPLE) -> int:
    """The embedding row count for a logical vocabulary: rounded up to `multiple`.

    257 (a byte arm plus its EOS) becomes 320; 64,001 becomes 64,064.
    """
    if logical_vocab <= 0:
        raise ValueError(f"logical_vocab must be positive, got {logical_vocab}")
    return -(-logical_vocab // multiple) * multiple


def resolve_device(name: str, *, allow_cpu: bool = False) -> str:
    """Turn `auto|cuda|mps|cpu` into the device string this machine will actually use.

    `auto` prefers CUDA, then MPS, then CPU; an explicit `cpu` is honoured, since asking
    for it is a choice rather than an accident.

    Raises `ValueError` on an unknown name, and `RuntimeError` in the two cases that used
    to be warnings:

    * a named device that this machine does not have (`cuda` on a box with no CUDA). The
      old behaviour was a silent fall back to CPU, which on rented hardware means a sweep
      that looks like it is running and is a hundred times too slow to finish.
    * `auto` resolving to CPU when `allow_cpu` is false. The smoke sweep sets
      `allow_cpu: true` because a CPU smoke run is still a valid pipeline test; the real
      sweep does not, so a GPU box whose driver did not come up stops immediately.
    """
    if name not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"unknown device {name!r}; expected one of auto, cuda, mps, cpu")
    if name == "cpu":
        return "cpu"
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    if name == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        if allow_cpu:
            logger.warning("device auto found no accelerator; running on cpu (allow_cpu)")
            return "cpu"
        raise RuntimeError(
            "device 'auto' found neither CUDA nor MPS and would run on the CPU. Set "
            "`allow_cpu: true` in the config if that is what you want (the smoke sweep "
            "does); the real sweep deliberately does not, because a CPU run of it would "
            "not finish."
        )
    if (name == "cuda" and has_cuda) or (name == "mps" and has_mps):
        return name
    raise RuntimeError(
        f"device {name!r} was requested but torch reports it unavailable "
        f"(cuda={has_cuda}, mps={has_mps}). Fix the environment, or set device to 'auto' "
        "or 'cpu' deliberately — this used to fall back to the CPU silently."
    )


def resolve_dtype(name: str, device: str) -> str:
    """Turn `auto|bfloat16|float16|float32` into the autocast dtype for `device`.

    `auto` is bfloat16 on a CUDA device that supports it, float16 on one that does not, and
    **float32 everywhere else**. MPS in particular: reduced precision on Metal is not
    validated by this project and a NaN'd loss would be discovered as a missing number in
    `results.json` rather than as an error, so the smoke path is the safe one. An explicit
    request for reduced precision on MPS or CPU is honoured but warned about.
    """
    if name not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of auto, bfloat16, float16, float32"
        )
    if name == "auto":
        if device == "cuda":
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"
    if name != "float32" and device != "cuda":
        logger.warning(
            "dtype %s requested on %s: reduced precision outside CUDA is untested here",
            name,
            device,
        )
    return name


@dataclass
class TrainConfig:
    """Everything one training run needs, and everything `results.json` records about it.

    `corpus` is the training text (one SLP1 sentence per line) and `eval_sets` maps a name
    to a held-out text scored at every evaluation; both are resolved against the repository
    root by `from_mapping`. `cache_dir` holds the `.bin` token streams, which are shared
    across seeds of the same arm and corpus, so it is deliberately outside `run_dir`.

    `block_size` defaults to the size's own (256 for `smoke`, 1024 for the two real sizes)
    when the mapping does not name one. Exactly one of `max_steps` and `max_tokens` need be
    given; when both are, the smaller resulting step budget wins, which is what makes
    "stop at 200 steps or 10M tokens, whichever comes first" expressible.

    `eval_raw_sets` maps an evaluation set's name to the **raw twin** of its text: for a
    split arm, `heldout_dcs_split` -> `heldout_dcs.txt`. Its character count is the BPC
    denominator (`bpc.py`, convention 3). A name absent from it is its own raw twin, which
    is every set of every raw arm.

    `eval_max_chars` truncates every held-out text at a whole line. It exists for the smoke
    runs; a real run leaves it `None` and scores the whole set.

    `allow_cpu` lets `device: auto` resolve to the CPU; without it that resolution raises
    (`resolve_device`).
    """

    arm: str
    track: str
    size: str
    corpus: Path
    out_dir: Path
    cache_dir: Path
    eval_sets: dict[str, Path] = field(default_factory=dict)
    eval_raw_sets: dict[str, Path] = field(default_factory=dict)
    block_size: int = 0
    batch_size: int = 8
    grad_accum: int = 1
    max_steps: int | None = None
    max_tokens: int | None = None
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0
    bias: bool = False
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"
    allow_cpu: bool = False
    eval_every: int = 50
    eval_batch_size: int = 8
    eval_max_chars: int | None = None
    log_every: int = 10
    resume: bool = True

    def __post_init__(self) -> None:
        if self.size not in MODEL_SIZES:
            raise ValueError(
                f"unknown size {self.size!r}; expected one of {', '.join(MODEL_SIZES)}"
            )
        if self.block_size <= 0:
            self.block_size = MODEL_SIZES[self.size].block_size
        if self.max_steps is None and self.max_tokens is None:
            raise ValueError("give max_steps or max_tokens: a run needs a budget to stop at")
        for name, value in (
            ("batch_size", self.batch_size),
            ("grad_accum", self.grad_accum),
            ("eval_every", self.eval_every),
            ("eval_batch_size", self.eval_batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def model_size(self) -> ModelSize:
        """The `ModelSize` row this run's `size` names."""
        return MODEL_SIZES[self.size]

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimiser step: batch × accumulation × context."""
        return self.batch_size * self.grad_accum * self.block_size

    @property
    def run_dir(self) -> Path:
        """`<out_dir>/<track>/<size>/<arm>/seed<k>` — one directory per run, per the plan."""
        return self.out_dir / self.track / self.size / self.arm / f"seed{self.seed}"

    def total_steps(self) -> int:
        """The step budget: `max_steps`, or `max_tokens` converted, or the smaller of both."""
        budgets = []
        if self.max_steps is not None:
            budgets.append(int(self.max_steps))
        if self.max_tokens is not None:
            budgets.append(-(-int(self.max_tokens) // self.tokens_per_step))
        return min(budgets)

    def to_dict(self) -> dict[str, Any]:
        """A YAML/JSON-safe mapping of every field, paths as strings."""
        return {
            "arm": self.arm,
            "track": self.track,
            "size": self.size,
            "corpus": str(self.corpus),
            "out_dir": str(self.out_dir),
            "cache_dir": str(self.cache_dir),
            "eval_sets": {name: str(path) for name, path in self.eval_sets.items()},
            "eval_raw_sets": {
                name: str(path) for name, path in self.eval_raw_sets.items()
            },
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "lr": self.lr,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "grad_clip": self.grad_clip,
            "dropout": self.dropout,
            "bias": self.bias,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "allow_cpu": self.allow_cpu,
            "eval_every": self.eval_every,
            "eval_batch_size": self.eval_batch_size,
            "eval_max_chars": self.eval_max_chars,
            "log_every": self.log_every,
            "resume": self.resume,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], root: Path | None = None) -> "TrainConfig":
        """Build a `TrainConfig` from a YAML mapping, resolving paths against `root`.

        Unknown keys are rejected rather than ignored: a typo in a sweep config would
        otherwise silently train at the default learning rate and be discovered only when
        the numbers looked odd. `root` defaults to the repository root, and is a parameter
        so tests can resolve against `tmp_path` (`sanskrit_tok.experiment.resolve_path`).
        """
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"unknown config keys: {', '.join(unknown)}; known keys: "
                f"{', '.join(sorted(known))}"
            )
        for required in ("arm", "track", "size", "corpus"):
            if required not in mapping:
                raise ValueError(f"config is missing required key {required!r}")

        values: dict[str, Any] = dict(mapping)
        values["corpus"] = resolve_path(values["corpus"], root)
        values["out_dir"] = resolve_path(
            values.get("out_dir", "outputs/05_lm_training"), root
        )
        values["cache_dir"] = resolve_path(
            values.get("cache_dir", "outputs/05_lm_training/encoded"), root
        )
        values["eval_sets"] = {
            name: resolve_path(path, root)
            for name, path in (values.get("eval_sets") or {}).items()
        }
        values["eval_raw_sets"] = {
            name: resolve_path(path, root)
            for name, path in (values.get("eval_raw_sets") or {}).items()
        }
        unknown_raw = sorted(set(values["eval_raw_sets"]) - set(values["eval_sets"]))
        if unknown_raw:
            raise ValueError(
                f"eval_raw_sets names sets that eval_sets does not: {', '.join(unknown_raw)}"
            )
        return cls(**values)


def device_display_name(device: str) -> str:
    """A human-readable name for `device`, for `results.json`'s `hardware` block.

    The GPU model on CUDA, the literal `"mps"` on Metal (Torch exposes no model name for
    it), and the host CPU otherwise — enough to tell two runs' hardware apart when their
    throughput numbers differ.
    """
    if device == "cuda":
        return str(torch.cuda.get_device_name(0))
    if device == "mps":
        return "mps"
    return platform.processor() or platform.machine()
