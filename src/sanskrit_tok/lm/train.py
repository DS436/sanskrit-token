"""The training loop, the BPC evaluator, and the `results.json` a run leaves behind.

One function does the work — `train(config) -> Path` — and everything else here exists so
that its output is comparable to another arm's. The loop itself is nanoGPT's, kept
deliberately plain: AdamW with betas (0.9, 0.95), a cosine schedule after a linear warmup,
gradient clipping at 1.0, autocast only where the device is known to support it.

**Evaluation is the part with decisions in it.** A held-out text is tokenised exactly as the
training corpus was — one EOS after every line — and the resulting id stream is prefixed
with a single EOS acting as a document-start marker, so that the very first real token has a
context and is scored like every other. The stream is then cut into non-overlapping
`block_size` windows; the last window is padded and its padding masked out with
`ignore_index=-1`. Every token of the held-out text is therefore predicted exactly once and
counted exactly once, and the total surprise is divided by the text's SLP1 characters
(`bpc.py`). The EOS tokens are inside that total: they are one token per line on the same
lines for every arm, a constant handicap that cancels in every comparison, and excluding
them would reward an arm for being bad at sentence boundaries.

**Resume is not a convenience.** A Track 2 run is hours long and this project's numbers must
survive a laptop lid closing, so the checkpoint holds the optimiser state, the step counter,
the tokens and bytes consumed, the elapsed seconds, and the batch sampler's own RNG state —
resuming continues the same stream of batches, not a fresh one. `curve.jsonl` is truncated
back to the checkpoint's step on resume, so a run that died between an eval and a checkpoint
does not leave two rows for one step.

**Parameters are counted twice, on purpose.** `non_embedding` excludes both `wte` and `wpe`,
which is what makes 50M mean the same transformer body for a 256-id byte arm and a 64k BPE
arm (docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and comparison
protocol"); `total` includes them and is what the `6 * N * tokens` FLOPs estimate uses.
Both are in `results.json`, along with which one the estimate used.
"""

import argparse
import json
import logging
import math
import platform
import random
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from sanskrit_tok.experiment import load_config, provenance, write_results
from sanskrit_tok.lm.bpc import BpcResult, bpc_from_token_nll
from sanskrit_tok.lm.config import (
    TrainConfig,
    device_display_name,
    resolve_device,
    resolve_dtype,
    round_up_vocab,
)
from sanskrit_tok.lm.data import (
    EncodedCorpus,
    ensure_encoded_corpus,
    eos_id_for,
    iter_batches,
    tokenise_text,
)
from sanskrit_tok.lm.model import GPT, GPTConfig
from sanskrit_tok.tokenizers.registry import LoadedTokenizer, load_tokenizer

__all__ = [
    "EXPERIMENT_NAME",
    "ParameterCounts",
    "build_model",
    "count_parameters",
    "evaluate_bpc",
    "learning_rate_at",
    "main",
    "seed_everything",
    "train",
]

logger = logging.getLogger(__name__)

#: The experiment these runs belong to; recorded in every `results.json`.
EXPERIMENT_NAME = "05_lm_training"

#: The target id `torch.nn.functional.cross_entropy` ignores, as vendored nanoGPT's
#: `forward` hard-codes it. Used to mask the padding of the final evaluation window.
IGNORE_INDEX = -1

_TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class ParameterCounts:
    """The three parameter counts `results.json` records, and what each is for.

    `non_embedding` is the transformer body: everything except the token embedding `wte`
    and the position embedding `wpe`. It is the number the model sizes are defined by, so
    that "50M" means the same capacity for every arm however large its vocabulary.

    `token_embedding` and `position_embedding` are the two tables, and `embedding` their
    sum. `total` is `non_embedding + embedding`, and equals `sum(p.numel())` over the model
    — nanoGPT ties `wte` and `lm_head` to one tensor, so the token embedding is counted
    once, not twice.

    Note this is deliberately *not* nanoGPT's own `get_num_params(non_embedding=True)`,
    which subtracts only `wpe` and therefore counts `wte` as body.
    """

    non_embedding: int
    token_embedding: int
    position_embedding: int

    @property
    def embedding(self) -> int:
        return self.token_embedding + self.position_embedding

    @property
    def total(self) -> int:
        return self.non_embedding + self.embedding

    def to_dict(self) -> dict[str, int | str]:
        return {
            "non_embedding": self.non_embedding,
            "token_embedding": self.token_embedding,
            "position_embedding": self.position_embedding,
            "embedding": self.embedding,
            "total": self.total,
            "flops_params_kind": "total",
        }


def seed_everything(seed: int) -> None:
    """Seed `random`, `numpy` and `torch` (plus CUDA and MPS) and ask for determinism.

    CLAUDE.md §8: seeds come from the config and are recorded in the results. The batch
    sampler does *not* draw from the global numpy state — it has its own `Generator`, seeded
    from the same number, so that resuming can restore its position exactly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: GPT) -> ParameterCounts:
    """Split `model`'s parameters into body, token embedding and position embedding."""
    # `nn.ModuleDict` is typed as holding `Module`, so the two embedding tables need a
    # cast to be read as embeddings; vendored nanoGPT carries no annotations of its own.
    wte = cast(torch.nn.Embedding, model.transformer.wte)
    wpe = cast(torch.nn.Embedding, model.transformer.wpe)
    token_embedding = int(wte.weight.numel())
    position_embedding = int(wpe.weight.numel())
    total = int(sum(p.numel() for p in model.parameters()))
    return ParameterCounts(
        non_embedding=total - token_embedding - position_embedding,
        token_embedding=token_embedding,
        position_embedding=position_embedding,
    )


def build_model(config: TrainConfig, logical_vocab: int) -> GPT:
    """A `GPT` sized by `config`, with its embedding padded up from `logical_vocab`.

    The model's `vocab_size` is the padded row count (`round_up_vocab`); the logical
    vocabulary — the arm's ids plus one EOS — is what the data contains and what
    `results.json` reports. The extra rows are never a target and never an input.
    """
    size = config.model_size
    model_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=round_up_vocab(logical_vocab),
        n_layer=size.n_layer,
        n_head=size.n_head,
        n_embd=size.n_embd,
        dropout=config.dropout,
        bias=config.bias,
    )
    return GPT(model_config)  # type: ignore[no-untyped-call]


def learning_rate_at(step: int, config: TrainConfig, total_steps: int) -> float:
    """Linear warmup to `lr`, then cosine decay to `lr * min_lr_ratio` at `total_steps`.

    `step` is 1-based (the rate used for the step about to be taken). A warmup longer than
    the whole run degenerates to pure warmup, which is what a five-step test wants.
    """
    minimum = config.lr * config.min_lr_ratio
    if step <= config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)
    decay_steps = max(1, total_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / decay_steps)
    return minimum + 0.5 * (1.0 + math.cos(math.pi * progress)) * (config.lr - minimum)


def _autocast(device: str, dtype_name: str) -> AbstractContextManager[Any]:
    """Autocast context for `device`, or a no-op when running in float32."""
    if dtype_name == "float32":
        return nullcontext()
    return torch.autocast(device_type=device, dtype=_TORCH_DTYPES[dtype_name])


@torch.no_grad()
def evaluate_bpc(
    model: GPT,
    arm: LoadedTokenizer,
    text_path: Path,
    *,
    eos_id: int,
    block_size: int,
    batch_size: int,
    device: str,
    dtype_name: str,
    max_chars: int | None,
) -> BpcResult:
    """Bits per character of `model` on the held-out text at `text_path`.

    The text is tokenised with `arm` and one EOS per line, prefixed with one more EOS as a
    document-start marker (that prefix is context, never a target), and cut into
    non-overlapping windows of `block_size`. Every token of the text is a target exactly
    once; the final window is padded with EOS on the input side and `IGNORE_INDEX` on the
    target side so the padding contributes nothing. The summed nats go to `bpc_from_token_nll`
    over the text's character and byte counts — see the module docstring for why the EOS
    tokens are charged to those characters.
    """
    tokenised = tokenise_text(arm, text_path, eos_id=eos_id, max_chars=max_chars)
    stream = np.asarray([eos_id, *tokenised.ids], dtype=np.int64)
    n_targets = len(stream) - 1
    n_windows = -(-n_targets // block_size)
    padded_length = n_windows * block_size + 1
    inputs = np.full(padded_length, eos_id, dtype=np.int64)
    targets = np.full(padded_length, IGNORE_INDEX, dtype=np.int64)
    inputs[: len(stream)] = stream
    targets[: len(stream)] = stream

    x_all = torch.from_numpy(inputs[:-1]).view(n_windows, block_size)
    y_all = torch.from_numpy(targets[1:]).view(n_windows, block_size)

    was_training = model.training
    model.eval()
    total_nats = 0.0
    counted = 0
    for start in range(0, n_windows, batch_size):
        x = x_all[start : start + batch_size].to(device)
        y = y_all[start : start + batch_size].to(device)
        valid = int((y != IGNORE_INDEX).sum().item())
        if valid == 0:
            continue
        with _autocast(device, dtype_name):
            _, loss = model(x, y)
        # nanoGPT's forward reduces with `mean` over the non-ignored targets, so the sum of
        # nats for this batch is that mean times how many targets it averaged over.
        total_nats += float(loss.item()) * valid
        counted += valid
    if was_training:
        model.train()

    if counted != n_targets:
        raise AssertionError(
            f"scored {counted} targets but the text has {n_targets} tokens; the windowing "
            "dropped or double-counted a token"
        )
    # The model gives one mean per batch, not one nat value per token, so the sequence
    # handed to `bpc_from_token_nll` is the exact total spread evenly over the tokens: its
    # *sum* — the only thing BPC uses — is the measured total, and `n_tokens` is right.
    mean_nats = total_nats / counted
    return bpc_from_token_nll(
        [mean_nats] * counted, tokenised.n_chars, n_bytes=tokenised.n_bytes
    )


def _batch_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """The sampler `Generator`'s internal state, JSON-free but `torch.save`-able."""
    return dict(rng.bit_generator.state)


def _restore_batch_rng(rng: np.random.Generator, state: dict[str, Any]) -> None:
    rng.bit_generator.state = state


def _truncate_curve(curve_path: Path, last_step: int) -> None:
    """Drop `curve.jsonl` rows past `last_step`, so a resume cannot duplicate a step.

    A run that was killed between writing an evaluation row and writing its checkpoint has
    rows the checkpoint does not know about; on resume those steps are retrained and would
    be logged twice.
    """
    if not curve_path.exists():
        return
    kept = [
        line
        for line in curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["step"] <= last_step
    ]
    curve_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def train(config: TrainConfig) -> Path:
    """Train one arm on one corpus and write its run directory; returns the results path.

    Writes `results.json`, `config.yaml`, `curve.jsonl` and `ckpt.pt` under
    `config.run_dir`. Resumes from an existing `ckpt.pt` when `config.resume` is set (the
    default): a checkpoint already at the step budget re-evaluates and rewrites the results
    without taking a step, which is what makes a sweep restartable.
    """
    device = resolve_device(config.device)
    dtype_name = resolve_dtype(config.dtype, device)
    total_steps = config.total_steps()
    seed_everything(config.seed)

    arm = load_tokenizer(config.arm)
    eos_id = eos_id_for(arm)
    corpus = ensure_encoded_corpus(
        arm,
        config.corpus,
        config.cache_dir / f"{config.corpus.stem}.{config.arm}.bin",
        eos_id=eos_id,
    )
    logger.info(
        "%s: %s -> %d tokens (%.3f bytes/token), device %s, dtype %s",
        config.arm,
        config.corpus.name,
        corpus.n_tokens,
        corpus.bytes_per_token,
        device,
        dtype_name,
    )

    model = build_model(config, corpus.logical_vocab).to(device)
    params = count_parameters(model)
    optimizer: torch.optim.Optimizer = model.configure_optimizers(  # type: ignore[no-untyped-call]
        config.weight_decay, config.lr, (config.beta1, config.beta2), device
    )
    # Only CUDA ever runs in float16 here (`resolve_dtype`), and a disabled scaler is a
    # pass-through on every path; naming a device the scaler does not support would raise
    # even when disabled, so it is told "cpu" whenever it is off.
    scaler = torch.amp.GradScaler(
        device if device == "cuda" else "cpu", enabled=dtype_name == "float16"
    )
    batch_rng = np.random.default_rng(config.seed)

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "curve.jsonl"
    ckpt_path = run_dir / "ckpt.pt"

    step = 0
    tokens_seen = 0
    elapsed_before = 0.0
    resumed_from = 0
    if config.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_batch_rng(batch_rng, checkpoint["batch_rng_state"])
        step = int(checkpoint["step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        elapsed_before = float(checkpoint["elapsed_s"])
        resumed_from = step
        _truncate_curve(curve_path, step)
        logger.info("resumed %s at step %d (%d tokens seen)", ckpt_path, step, tokens_seen)

    batches = iter_batches(
        corpus.bin_path, config.block_size, config.batch_size, batch_rng, device=device
    )

    def evaluate_all() -> dict[str, BpcResult]:
        return {
            name: evaluate_bpc(
                model,
                arm,
                path,
                eos_id=eos_id,
                block_size=config.block_size,
                batch_size=config.eval_batch_size,
                device=device,
                dtype_name=dtype_name,
                max_chars=config.eval_max_chars,
            )
            for name, path in config.eval_sets.items()
        }

    started = time.perf_counter()
    last_loss = float("nan")
    latest_bpc: dict[str, BpcResult] = {}
    evaluated = False

    def log_curve(at_step: int, loss: float, lr: float) -> dict[str, BpcResult]:
        nonlocal latest_bpc, evaluated
        bpc = evaluate_all()
        latest_bpc = bpc
        evaluated = True
        elapsed = elapsed_before + (time.perf_counter() - started)
        row: dict[str, Any] = {
            "step": at_step,
            "tokens_seen": tokens_seen,
            "bytes_seen": int(round(tokens_seen * corpus.bytes_per_token)),
            "epochs": tokens_seen / corpus.n_tokens,
            "flops_est": 6.0 * params.total * tokens_seen,
            "loss_train": loss if math.isfinite(loss) else None,
            "lr": lr,
            "elapsed_s": elapsed,
            "tokens_per_s": tokens_seen / elapsed if elapsed > 0 else 0.0,
            "bpc": {name: result["value"] for name, result in bpc.items()},
        }
        with curve_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        logger.info(
            "step %d/%d loss %.4f bpc %s (%.0f tok/s)",
            at_step,
            total_steps,
            loss,
            {name: round(value, 4) for name, value in row["bpc"].items()},
            row["tokens_per_s"],
        )
        return bpc

    def save_checkpoint(at_step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": at_step,
                "tokens_seen": tokens_seen,
                "elapsed_s": elapsed_before + (time.perf_counter() - started),
                "batch_rng_state": _batch_rng_state(batch_rng),
                "config": config.to_dict(),
                "logical_vocab": corpus.logical_vocab,
            },
            ckpt_path,
        )

    if step == 0:
        log_curve(0, last_loss, 0.0)

    while step < total_steps:
        lr = learning_rate_at(step + 1, config, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(config.grad_accum):
            x, y = next(batches)
            with _autocast(device, dtype_name):
                _, loss = model(x, y)
                loss = loss / config.grad_accum
            scaler.scale(loss).backward()
            accumulated += float(loss.item())
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += config.tokens_per_step
        last_loss = accumulated
        if step % config.log_every == 0:
            logger.debug("step %d loss %.4f lr %.2e", step, last_loss, lr)
        if step % config.eval_every == 0 or step == total_steps:
            log_curve(step, last_loss, lr)
            save_checkpoint(step)

    if not evaluated:
        # Resumed at (or past) the budget with nothing left to train: score and record it
        # anyway, so a re-run of a finished sweep still produces a results.json. The flag,
        # rather than `if not latest_bpc`, so a run with no eval sets does not evaluate
        # (nothing) a second time and save a redundant checkpoint.
        latest_bpc = evaluate_all()
        save_checkpoint(step)

    elapsed = elapsed_before + (time.perf_counter() - started)
    bytes_seen = int(round(tokens_seen * corpus.bytes_per_token))
    curve_summary = _summarise_curve(curve_path, list(config.eval_sets))

    results: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "arm": config.arm,
        "track": config.track,
        "size": config.size,
        "seed": config.seed,
        "steps": step,
        "resumed_from_step": resumed_from,
        "total_steps": total_steps,
        "tokens_seen": tokens_seen,
        "bytes_seen": bytes_seen,
        "epochs": tokens_seen / corpus.n_tokens,
        "flops_est": 6.0 * params.total * tokens_seen,
        "loss_train_final": last_loss,
        "bpc": {name: dict(result) for name, result in latest_bpc.items()},
        "curve_summary": curve_summary,
        "params": params.to_dict(),
        "model": {
            "n_layer": config.model_size.n_layer,
            "n_embd": config.model_size.n_embd,
            "n_head": config.model_size.n_head,
            "block_size": config.block_size,
            "logical_vocab": corpus.logical_vocab,
            "model_vocab_size": round_up_vocab(corpus.logical_vocab),
            "dropout": config.dropout,
            "bias": config.bias,
        },
        "corpus": _corpus_summary(corpus),
        "throughput": {
            "elapsed_s": elapsed,
            "tokens_per_s": tokens_seen / elapsed if elapsed > 0 else 0.0,
            "steps_per_s": step / elapsed if elapsed > 0 else 0.0,
        },
        "hardware": {
            "torch": torch.__version__,
            "device": device,
            "device_name": device_display_name(device),
            "dtype": dtype_name,
            "platform": platform.platform(),
        },
        "config": config.to_dict(),
        **provenance(),
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8"
    )
    return write_results(results, run_dir)


def _corpus_summary(corpus: EncodedCorpus) -> dict[str, Any]:
    """The encoded-corpus facts `results.json` needs to compare arms at equal bytes."""
    return {
        "path": str(corpus.source_path),
        "sha256": corpus.source_sha256,
        "n_tokens": corpus.n_tokens,
        "n_chars": corpus.n_chars,
        "n_bytes": corpus.n_bytes,
        "n_lines": corpus.n_lines,
        "bytes_per_token": corpus.bytes_per_token,
        "chars_per_token": corpus.chars_per_token,
        "eos_id": corpus.eos_id,
        "dtype": corpus.dtype,
    }


def _summarise_curve(curve_path: Path, eval_names: list[str]) -> dict[str, Any]:
    """Per evaluation set, the lowest BPC on the curve and the step it happened at."""
    rows = [
        json.loads(line)
        for line in curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary: dict[str, Any] = {}
    for name in eval_names:
        scored = [
            (row["bpc"][name], row["step"])
            for row in rows
            if name in row.get("bpc", {}) and math.isfinite(row["bpc"][name])
        ]
        if not scored:
            summary[name] = {"min_bpc": None, "step": None, "n_points": 0}
            continue
        best_value, best_step = min(scored)
        summary[name] = {
            "min_bpc": best_value,
            "step": best_step,
            "n_points": len(scored),
            "first_bpc": scored[0][0],
            "last_bpc": scored[-1][0],
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    """`python -m sanskrit_tok.lm.train --config run.yaml`: train one run from a YAML file."""
    parser = argparse.ArgumentParser(description="Train one Experiment 05 language model.")
    parser.add_argument("--config", type=Path, required=True, help="run config YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = TrainConfig.from_mapping(load_config(args.config))
    results_path = train(config)
    logger.info("wrote %s", results_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
