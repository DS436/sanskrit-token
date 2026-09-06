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
counted exactly once, and the total surprise is divided by the SLP1 characters of the
**raw** twin of that text (`bpc.py`). The EOS tokens are inside that total: they are one
token per line on the same lines for every arm, a constant handicap that cancels in every
comparison, and excluding them would reward an arm for being bad at sentence boundaries.

**The BPC denominator is the raw held-out text, for a split arm as much as a raw one.** A
split arm predicts the sandhi-split twin of a held-out set, which carries 2.1-6.3% more
characters than the raw text because undoing sandhi inserts spaces; dividing its nats by
that larger count would hand it a discount the size of the effect being measured. So
`evaluate_bpc` takes the raw twin's path beside the text it scores, measures it over
*exactly the lines that were scored* (which matters when `eval_max_chars` truncates), and
divides by that. `n_chars_scored` and `n_chars_denominator` are both recorded, per set, in
`curve.jsonl` and in `results.json`, so the difference is in the record rather than in the
number (docs/decisions.md, 2026-09-06, "BPC is bits per character of the RAW held-out
text for every arm").

**The softmax runs over the logical vocabulary, in training and in evaluation alike.** The
embedding is padded up to a multiple of 64 for kernel shapes (`config.round_up_vocab`), so
a byte arm's 257 ids live in a 320-row table and 63 rows exist that no id ever names. Left
alone they still sit in the softmax denominator, and a freshly initialised model then
scores `ln(320)` nats per token instead of `ln(257)` — 3.7% of probability mass leaked to
ids that cannot occur, straight into the reported BPC, and by an amount that depends on how
far each arm's vocabulary happens to sit from a multiple of 64. `_masked_logits` sets those
columns to `-inf` before the loss, so the distribution is over the logical vocabulary
exactly. Training is masked too: the padded rows are never targets, but without the mask
they receive gradient through the denominator and the model spends capacity pushing down
ids it will never be scored on. The cost is that the loss is computed here rather than
taken from vendored nanoGPT's `forward`. `forward_logits` therefore runs the transformer
stack and `lm_head` itself and never asks the model for a loss — the vendored `forward`
with `targets` would compute a full unmasked cross-entropy over every position and then
throw it away, which at a 64k vocabulary is 10-20% of the step (docs/decisions.md,
2026-09-06, "Sweep hardening before paid GPU time"). `model.py` stays byte-for-byte
nanoGPT; the fast path lives here.

**Resume is not a convenience.** A Track 2 run is hours long and this project's numbers must
survive a laptop lid closing, so the checkpoint holds the optimiser state, the step counter,
the tokens and bytes consumed, the elapsed seconds, the batch sampler's own RNG state and
torch's global generator states (CPU, and CUDA/MPS where they exist) — resuming continues
the same stream of batches and the same stream of every other random draw, not a fresh one.
`curve.jsonl` is truncated back to the checkpoint's step on resume, and emptied on a start
that is *not* a resume, so neither a crash between an eval and a checkpoint nor a re-used
run directory can leave two rows for one step.

**Parameters are counted twice, on purpose.** `non_embedding` excludes both `wte` and `wpe`,
which is what makes 50M mean the same transformer body for a 256-id byte arm and a 64k BPE
arm (docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and comparison
protocol"); `total` includes them and is what the `6 * N * tokens` FLOPs estimate uses.
Both are in `results.json`, along with which one the estimate used.
"""

import argparse
import io
import json
import logging
import math
import platform
import random
import sys
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sanskrit_tok.experiment import load_config, provenance, write_results
from sanskrit_tok.lm.bpc import BpcResult, bpc_from_total
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
    measure_text,
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
    "forward_logits",
    "learning_rate_at",
    "main",
    "masked_cross_entropy",
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
    with _vendor_stdout_to_log():
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


@contextmanager
def _vendor_stdout_to_log() -> Iterator[None]:
    """Route the vendored model's `print`s into this module's logger.

    `model.py` is nanoGPT byte-for-byte and prints its parameter count and its optimiser
    grouping to stdout. CLAUDE.md §8 wants logging, not prints, and a sweep that redirects
    stdout to a file otherwise interleaves those lines with nothing to say which run they
    belong to. The file itself is not edited — its sha256 is the provenance record — so the
    stream is captured here instead.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        yield
    for line in buffer.getvalue().splitlines():
        if line.strip():
            logger.info("nanoGPT: %s", line.strip())


def forward_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """`model`'s logits for every position of `idx`, without computing any loss.

    Exactly vendored nanoGPT's `GPT.forward(idx, targets)` up to and including `lm_head`,
    minus the `F.cross_entropy` it would then compute over the padded vocabulary and this
    module would discard (`masked_cross_entropy` computes the one that is used). At 64k
    that discarded softmax is a `batch x 1024 x 64,064` reduction on every forward, in
    training and in evaluation alike.

    The `cast`s are because `nn.ModuleDict` is typed as holding bare `Module`s and the
    vendored file carries no annotations of its own; the attribute names are nanoGPT's.
    """
    block_size = int(model.config.block_size)
    _, t = idx.size()
    if t > block_size:
        raise ValueError(f"sequence of length {t} exceeds the block size {block_size}")
    transformer = model.transformer
    wte = cast(torch.nn.Embedding, transformer.wte)
    wpe = cast(torch.nn.Embedding, transformer.wpe)
    drop = cast(torch.nn.Module, transformer.drop)
    ln_f = cast(torch.nn.Module, transformer.ln_f)
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
    x: torch.Tensor = drop(wte(idx) + wpe(pos))
    for block in cast(torch.nn.ModuleList, transformer.h):
        x = block(x)
    return cast(torch.Tensor, model.lm_head(ln_f(x)))


def _masked_logits(logits: torch.Tensor, logical_vocab: int) -> torch.Tensor:
    """`logits` with every column at or above `logical_vocab` set to `-inf`, in place.

    The columns are the embedding's padding rows (`config.round_up_vocab`): real
    parameters that no id ever names. `-inf` gives them exactly zero probability, so the
    softmax normalises over the logical vocabulary and their gradient is zero rather than
    the small negative pressure an unmasked denominator applies.

    In place, deliberately: at 50M with a 64k vocabulary one logit tensor is over a
    gigabyte and a masked copy would double the peak. It is safe because the only producer
    of `logits` is the tied `lm_head` linear, whose backward saves its *input* and weight,
    never its output; nothing else holds this tensor.
    """
    if logical_vocab < logits.size(-1):
        logits[..., logical_vocab:] = float("-inf")
    return logits


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, logical_vocab: int, *, reduction: str = "mean"
) -> torch.Tensor:
    """Cross-entropy over the logical vocabulary only, ignoring `IGNORE_INDEX` targets.

    The same reduction semantics as `torch.nn.functional.cross_entropy`: `"mean"` averages
    over the non-ignored targets (what a training step wants) and `"sum"` totals them (what
    an evaluator wants, since a mean would have to be multiplied back out).
    """
    masked = _masked_logits(logits, logical_vocab)
    return F.cross_entropy(
        masked.view(-1, masked.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


@torch.no_grad()
def evaluate_bpc(
    model: GPT,
    arm: LoadedTokenizer,
    text_path: Path,
    *,
    eos_id: int,
    logical_vocab: int,
    block_size: int,
    batch_size: int,
    device: str,
    dtype_name: str,
    max_chars: int | None,
    raw_text_path: Path | None = None,
) -> BpcResult:
    """Bits per character of `model` on the held-out text at `text_path`.

    The text is tokenised with `arm` and one EOS per line, prefixed with one more EOS as a
    document-start marker (that prefix is context, never a target), and cut into
    non-overlapping windows of `block_size`. Every token of the text is a target exactly
    once; the final window is padded with EOS on the input side and `IGNORE_INDEX` on the
    target side so the padding contributes nothing. The summed nats go to `bpc_from_total`
    over the text's character and byte counts — see the module docstring for why the EOS
    tokens are charged to those characters.

    `raw_text_path` is the **raw twin** of `text_path` — for a split arm's
    `heldout_dcs_split.txt`, that is `heldout_dcs.txt` — and its characters are the
    denominator. It defaults to `text_path`, which is correct for every raw arm because
    there the two are one file. The twin is measured over exactly as many lines as were
    scored, so a `max_chars` truncation cuts both sides at the same sentence; the two files
    are written in lockstep line by line (`build_corpus.py`), and a line-count mismatch is
    an error rather than a silently rescaled BPC.

    The loss is computed here, from `forward_logits`, rather than taken from the model's own
    `forward`: the distribution has to be over `logical_vocab` ids and not over the padded
    embedding's rows (`masked_cross_entropy`, and the module docstring). `reduction="sum"`
    gives the batch's exact total, so nothing is multiplied back out of a mean.
    """
    tokenised = tokenise_text(arm, text_path, eos_id=eos_id, max_chars=max_chars)
    raw_path = text_path if raw_text_path is None else raw_text_path
    if raw_path == text_path:
        denominator = tokenised.n_chars
        denominator_bytes = tokenised.n_bytes
    else:
        extent = measure_text(raw_path, max_lines=tokenised.n_lines)
        if extent.n_lines != tokenised.n_lines:
            raise ValueError(
                f"{raw_path} has {extent.n_lines} line(s) where {text_path} scored "
                f"{tokenised.n_lines}: the raw twin and the text scored are not aligned, "
                "so the BPC denominator would be of different sentences"
            )
        denominator = extent.n_chars
        denominator_bytes = extent.n_bytes
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
            # Never `model(x, y)`: its own loss is a mean over the *padded* vocabulary and
            # computing it to throw it away is 10-20% of the forward at 64k.
            logits = forward_logits(model, x)
            nats = masked_cross_entropy(logits, y, logical_vocab, reduction="sum")
        total_nats += float(nats.item())
        counted += valid
    if was_training:
        model.train()

    if counted != n_targets:
        raise AssertionError(
            f"scored {counted} targets but the text has {n_targets} tokens; the windowing "
            "dropped or double-counted a token"
        )
    return bpc_from_total(
        total_nats,
        counted,
        denominator,
        n_bytes=denominator_bytes,
        n_chars_scored=tokenised.n_chars,
    )


def _batch_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """The sampler `Generator`'s internal state, JSON-free but `torch.save`-able."""
    return dict(rng.bit_generator.state)


def _restore_batch_rng(rng: np.random.Generator, state: dict[str, Any]) -> None:
    rng.bit_generator.state = state


def _torch_rng_states() -> dict[str, Any]:
    """Every torch generator state this process has, for the checkpoint.

    The batch sampler has its own `Generator` and is restored separately, but torch's
    global stream is what dropout and any future initialisation draw from, so a resume that
    restored only the batch RNG would still diverge from an uninterrupted run the moment
    dropout was switched on. Captured on CPU (`torch.get_rng_state` returns a CPU
    `ByteTensor`), plus the CUDA and MPS streams where the platform has them.
    """
    states: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        states["mps"] = torch.mps.get_rng_state()
    return states


def _restore_torch_rng(states: dict[str, Any] | None) -> None:
    """Put back what `_torch_rng_states` saved, skipping devices this machine lacks.

    `torch.load(..., map_location=device)` moves every tensor in the checkpoint, generator
    states included, so each is sent back to the CPU before being set — `set_rng_state`
    wants a CPU `ByteTensor` and would otherwise raise on an MPS resume. A checkpoint
    written before these states existed passes `None` and simply keeps the seeded stream.
    """
    if not states:
        return
    if "cpu" in states:
        torch.set_rng_state(states["cpu"].cpu())
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])
    if (
        "mps" in states
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(states["mps"].cpu())


def _clear_curve(curve_path: Path) -> None:
    """Empty `curve.jsonl`, for a start that is not a resume.

    A run directory can be re-used — a config edited and re-run with `resume: false`, or a
    checkpoint deleted to start over — and appending to the previous run's curve would
    produce a file with two rows for step 0 and a `curve_summary` mixing two trajectories.
    The file is emptied rather than deleted so it always exists for `_summarise_curve`.
    """
    curve_path.write_text("", encoding="utf-8")


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
    device = resolve_device(config.device, allow_cpu=config.allow_cpu)
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

    logical_vocab = corpus.logical_vocab
    model = build_model(config, logical_vocab).to(device)
    params = count_parameters(model)
    with _vendor_stdout_to_log():
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
        _restore_torch_rng(checkpoint.get("torch_rng_state"))
        step = int(checkpoint["step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        elapsed_before = float(checkpoint["elapsed_s"])
        resumed_from = step
        _truncate_curve(curve_path, step)
        logger.info("resumed %s at step %d (%d tokens seen)", ckpt_path, step, tokens_seen)
    else:
        # Not a resume: whatever is in this run directory belongs to a previous run.
        _clear_curve(curve_path)

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
                logical_vocab=logical_vocab,
                block_size=config.block_size,
                batch_size=config.eval_batch_size,
                device=device,
                dtype_name=dtype_name,
                max_chars=config.eval_max_chars,
                raw_text_path=config.eval_raw_sets.get(name),
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
            "bytes_seen_est": int(round(tokens_seen * corpus.bytes_per_token)),
            "epochs": tokens_seen / corpus.n_tokens,
            "flops_est": 6.0 * params.total * tokens_seen,
            "loss_train": loss if math.isfinite(loss) else None,
            "lr": lr,
            "elapsed_s": elapsed,
            "tokens_per_s": tokens_seen / elapsed if elapsed > 0 else 0.0,
            "bpc": {name: result["value"] for name, result in bpc.items()},
            # The arithmetic behind each of those values, so that a BPC can be re-derived
            # against a different denominator without re-running the model. `n_chars_scored`
            # is the text the model predicted and `n_chars_denominator` the raw twin it was
            # divided by; they differ only for a split arm (`evaluate_bpc`).
            "bpc_detail": {
                name: {
                    "total_nats": result["total_nats"],
                    "n_tokens": result["n_tokens"],
                    "n_chars_scored": result["n_chars_scored"],
                    "n_chars_denominator": result["n_chars_denominator"],
                    "bpc": result["value"],
                }
                for name, result in bpc.items()
            },
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
                "torch_rng_state": _torch_rng_states(),
                "config": config.to_dict(),
                "logical_vocab": logical_vocab,
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
                # As in `evaluate_bpc`: the vendored forward's own loss is over the padded
                # vocabulary, so it is never computed and the masked one is used instead.
                # The padded rows are never targets, but an unmasked softmax denominator
                # still sends them gradient, so train and eval have to agree on this.
                logits = forward_logits(model, x)
                loss = masked_cross_entropy(logits, y, logical_vocab) / config.grad_accum
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
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
    # An *estimate*: batches are random windows of the corpus, so the exact source bytes
    # behind them are not tracked. `tokens_seen` times the corpus's overall bytes-per-token
    # is unbiased over the run and is what the equal-bytes budget is set against, but it is
    # not a count of bytes actually read, hence the name.
    bytes_seen_est = int(round(tokens_seen * corpus.bytes_per_token))
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
        "bytes_seen_est": bytes_seen_est,
        "epochs": tokens_seen / corpus.n_tokens,
        "flops_est": 6.0 * params.total * tokens_seen,
        "loss_train_final": last_loss,
        "bpc": {name: dict(result) for name, result in latest_bpc.items()},
        "bpc_denominator_sets": {
            name: str(config.eval_raw_sets.get(name, path))
            for name, path in config.eval_sets.items()
        },
        "curve_summary": curve_summary,
        "params": params.to_dict(),
        "model": {
            "n_layer": config.model_size.n_layer,
            "n_embd": config.model_size.n_embd,
            "n_head": config.model_size.n_head,
            "block_size": config.block_size,
            "logical_vocab": logical_vocab,
            "model_vocab_size": round_up_vocab(logical_vocab),
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
