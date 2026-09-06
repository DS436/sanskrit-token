"""Experiment 05's sweep runner: enumerate the grid, budget it in bytes, train it, prune it.

    uv run python experiments/05_lm_training/sweep.py --config experiments/05_lm_training/sweep.yaml
    uv run python experiments/05_lm_training/sweep.py --config .../sweep.yaml \
        --dry-run --tokens-per-s 17300

One process, one run at a time, resumable: a run whose `results.json` is present *and*
whose recorded plan hash matches the config is skipped, so an interrupted sweep is restarted
with the same command and picks up where it stopped.

Three things here are research decisions rather than plumbing, and each of them is a number
in every `results.json` the sweep writes.

**The budget is bytes, not tokens or steps.** Arms are compared at equal training *bytes*
(docs/decisions.md, 2026-09-05, "Experiment 05 model, evaluation and comparison protocol"):
the models see the same text, and the tokenizer decides only how that text is cut up. The
budget is therefore one number per track — `epochs × the raw corpus's bytes` — which each
arm converts into its own token budget through its own bytes-per-token
(`token_budget`). A denser vocabulary spends the same bytes as *more* tokens and therefore
more compute, which is the handicap Experiment 04 measured (24–33% for T5/T6) and is
reported, not corrected for.

**A split arm's budget is bytes of the split text.** `track1_split.txt` is the same
sentences as `track1_raw.txt` with sandhi undone, which inserts spaces and makes it ~6%
larger. The budget number is the same for every arm in a track (it comes from the raw
corpus), so a split arm trains on that many bytes *of its own text* — the same amount of
text-as-the-model-sees-it, one epoch's worth less of the underlying sentences. The
alternative, equal bytes of the *raw* text, would give split arms ~6% more training signal
per byte and is not what is used.

**`T4_bpe_split_64k_oracle_dcs` and `T6_morphbpe_split_64k_dcs` are evaluated on the
`_split` twin of every held-out set** — the same sentences, split the same way the training
text was — **and scored against the RAW twin's character count**. The split text carries
2.1-6.3% more characters (the inserted spaces), so dividing by it would hand the split arms
a discount the size of the effect being measured; every arm's bits-per-character therefore
has the same denominator per evaluation set. `_eval_sets_for` returns both paths and the
run config carries both (docs/decisions.md, 2026-09-06, "BPC is bits per character of the
RAW held-out text for every arm").

**The device is resolved and checked before anything is encoded.** `run_sweep` logs the
device and dtype it will use and lets `resolve_device` raise when the requested device is
absent, or when `auto` would land on the CPU without `allow_cpu: true` — a 51-run sweep
should stop in the first second on a box whose GPU did not come up, not encode 755 MB and
then take three weeks.

Checkpoints are deleted once a run's `results.json` is final (`keep_checkpoints: true`
keeps them): the sweep is 51 runs and a 125M checkpoint is ~1.5 GB, while nothing
downstream reads one. Resumption of an *unfinished* run still works — the checkpoint is
only removed after the results are written.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from sanskrit_tok.experiment import load_config, provenance, repo_root, resolve_path
from sanskrit_tok.lm.config import MODEL_SIZES, TrainConfig, resolve_device, resolve_dtype
from sanskrit_tok.lm.data import (
    arm_fingerprint,
    ensure_encoded_corpus,
    eos_id_for,
    load_encoded_corpus,
)
from sanskrit_tok.lm.train import build_model, count_parameters, train
from sanskrit_tok.tokenizers.registry import LoadedTokenizer, load_tokenizer

logger = logging.getLogger("exp05.sweep")

#: The arm every "tokens/bytes/FLOPs to reference BPC" figure is measured against.
DEFAULT_REFERENCE_ARM = "T1_bpe_raw_64k_dcs"

#: Where the held-out evaluation texts live when a config does not say otherwise.
DEFAULT_EVAL_DIR = "data/processed/lm"

#: The suffix that turns a held-out set (or a Track 1 corpus) into its sandhi-split twin.
SPLIT_SUFFIX = "_split"

#: The groups a config's `eval_sets` may name, and the label each gives its sets. Order is
#: the order the sets are evaluated and reported in: the in-domain set first, then the
#: same-language-different-register transfer sets, then the parallel out-of-domain ones.
EVAL_SET_LABELS = ("in_domain", "transfer", "ood")

#: `TrainConfig` fields a sweep config may set per size or per track. Everything else about
#: a run — its arm, corpus, evaluation sets, seed and token budget — the sweep derives, so
#: allowing them here would let a config silently contradict the grid.
SETTING_KEYS = frozenset(
    {
        "block_size",
        "batch_size",
        "grad_accum",
        "max_steps",
        "lr",
        "min_lr_ratio",
        "warmup_steps",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "dropout",
        "bias",
        "device",
        "dtype",
        "eval_every",
        "eval_batch_size",
        "eval_max_chars",
        "log_every",
        "resume",
        "allow_cpu",
    }
)

#: Settings a config may also give once at the top level, next to `seeds`, because they
#: describe the machine rather than the experiment.
GLOBAL_SETTING_KEYS = ("device", "dtype", "allow_cpu")

#: One line in every `stride` is tokenised when a corpus has no cached encoding and a
#: dry run needs its bytes-per-token. 1 in 1,000 of Track 2 is ~9,900 lines: seconds to
#: tokenise, and uniform over the file rather than over its first source.
DEFAULT_ESTIMATE_STRIDE = 1000

#: The assumed ratio between this laptop's MPS token rate and a rented A100's. An
#: assumption, labelled as one wherever it is printed: it is not measured anywhere in this
#: project, and it ignores that the ratio itself grows with model size.
DEFAULT_GPU_SPEEDUP = 50.0


def is_split_arm(name: str) -> bool:
    """Whether `name` is one of the arms trained on sandhi-split text.

    `T4_bpe_split_64k_oracle_dcs` and `T6_morphbpe_split_64k_dcs` are; the five raw arms —
    including `T5_morphbpe_rawseg_64k_dcs`, whose *constraint* comes from segments but
    whose *text* is sandhied — are not. Matched as a whole name component so that
    `rawseg` and `raw` cannot be caught by a substring.
    """
    return "_split_" in f"_{name}_"


@cache
def tokenizer_digest(arm: str) -> str | None:
    """The sha256 of `arm`'s tokenizer file, or `None` when the arm has no file.

    Cached: `plan_hash` is called once per run and the sweep has 51 of them over seven
    arms, and hashing a 2 MB tokenizer file 51 times is pure waste.
    """
    return arm_fingerprint(load_tokenizer(arm))[2]


def corpus_bytes(path: Path) -> int:
    """UTF-8 bytes of `path`'s non-blank lines, newlines excluded.

    The same convention as `lm/data.py`'s `n_bytes`, so a budget of "8 epochs" is 8 times
    what one pass over the corpus actually encodes. `build_corpus.py` records this number
    in the manifest beside every corpus, which is read in preference to re-scanning: Track
    2 is 755 MB and the sweep asks this question on every invocation.
    """
    manifest = path.with_suffix("").with_suffix(".manifest.json")
    if manifest.name == path.name:  # pragma: no cover - a path with no suffix
        manifest = path.with_name(f"{path.name}.manifest.json")
    if manifest.exists():
        try:
            recorded = json.loads(manifest.read_text(encoding="utf-8")).get("n_bytes")
        except (json.JSONDecodeError, OSError):  # pragma: no cover - unreadable manifest
            recorded = None
        if isinstance(recorded, int) and recorded > 0:
            return recorded
    total = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n").rstrip("\r")
            if line.strip():
                total += len(line.encode("utf-8"))
    return total


def token_budget(max_bytes: int, bytes_per_token: float) -> int:
    """`max_bytes` of text expressed in one arm's tokens, rounded to the nearest.

    This is the whole of the equal-bytes protocol in one line: every arm is given the same
    number of bytes and converts it with its own compression, so the arm that spends fewer
    bytes per token trains on more tokens.
    """
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be positive, got {bytes_per_token}")
    return int(round(max_bytes / bytes_per_token))


def sample_bytes_per_token(
    arm: LoadedTokenizer, corpus: Path, stride: int = DEFAULT_ESTIMATE_STRIDE
) -> tuple[float, int]:
    """Estimate `arm`'s bytes-per-token on `corpus` from every `stride`-th line.

    Returns `(bytes_per_token, n_lines_sampled)`. One EOS is counted per sampled line,
    exactly as `encode_corpus` writes one, so the estimate is of the same quantity the
    cached `.meta.json` reports rather than of a slightly different one. Used only by
    `--dry-run`; a real run always encodes the corpus and uses the exact figure.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    n_bytes = 0
    n_tokens = 0
    n_lines = 0
    with corpus.open(encoding="utf-8") as handle:
        index = 0
        for raw in handle:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if index % stride == 0:
                n_bytes += len(line.encode("utf-8"))
                n_tokens += len(arm.encode(line)) + 1
                n_lines += 1
            index += 1
    if n_tokens == 0:
        raise ValueError(f"{corpus}: sampling every {stride} lines produced no tokens")
    return n_bytes / n_tokens, n_lines


@dataclass
class RunSpec:
    """One planned run: which arm, on which text, at which size, with which seed.

    Everything except the token budget, which needs the corpus encoded before it can be
    computed (`token_budget`). `settings` are the `TrainConfig` knobs the config chose;
    `max_bytes` is the equal-bytes budget, or `None` for a step-budgeted run (the smoke
    sweep). `corpus_n_bytes` is the corpus's own size, carried so the plan hash changes
    when the corpus is rebuilt.
    """

    track: str
    size: str
    arm: str
    seed: int
    corpus: Path
    corpus_n_bytes: int
    eval_sets: dict[str, Path]
    eval_raw_sets: dict[str, Path]
    eval_labels: dict[str, str]
    in_domain_set: str
    settings: dict[str, Any]
    out_dir: Path
    cache_dir: Path
    max_bytes: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        """`<out_dir>/<track>/<size>/<arm>/seed<k>`, matching `TrainConfig.run_dir`."""
        return self.out_dir / self.track / self.size / self.arm / f"seed{self.seed}"

    @property
    def bin_path(self) -> Path:
        """The cached token stream for this arm on this corpus, as `train` names it."""
        return self.cache_dir / f"{self.corpus.stem}.{self.arm}.bin"

    @property
    def is_split(self) -> bool:
        return is_split_arm(self.arm)

    def train_mapping(self, max_tokens: int | None) -> dict[str, Any]:
        """The `TrainConfig` mapping for this run, with `max_tokens` filled in."""
        mapping: dict[str, Any] = {
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
            "seed": self.seed,
            **self.settings,
        }
        if max_tokens is not None:
            mapping["max_tokens"] = max_tokens
        return mapping

    def plan_hash(self) -> str:
        """A digest of everything about this run that is fixed before encoding.

        Covers the arm, the text, the evaluation sets and their raw twins, every
        hyperparameter and the byte budget, plus the corpus's own size so that a rebuilt
        corpus invalidates a finished run. The token budget is *not* in it — it is derived
        from this budget and the arm's encoding, both of which are recorded in
        `sweep_run.json` beside the results.

        **The tokenizer's own sha256 is in it too.** An arm name is not its contents: a
        retrained `T1_bpe_raw_64k_dcs` is a different vocabulary under the same key, and
        without its digest a resumed sweep would skip every run of it and report numbers
        from the old one. `arm_fingerprint` returns `None` for an arm with no file behind
        it (`T7_byt5`, a HuggingFace id), which is recorded as `null` rather than omitted.
        """
        payload = {
            "track": self.track,
            "size": self.size,
            "arm": self.arm,
            "tokenizer_sha256": tokenizer_digest(self.arm),
            "seed": self.seed,
            "corpus": str(self.corpus),
            "corpus_n_bytes": self.corpus_n_bytes,
            "eval_sets": {name: str(path) for name, path in sorted(self.eval_sets.items())},
            "eval_raw_sets": {
                name: str(path) for name, path in sorted(self.eval_raw_sets.items())
            },
            "settings": {key: self.settings[key] for key in sorted(self.settings)},
            "max_bytes": self.max_bytes,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _settings_for(
    config: Mapping[str, Any], track_config: Mapping[str, Any], size: str
) -> dict[str, Any]:
    """Merge the config's setting layers for one size: defaults < size < track."""
    merged: dict[str, Any] = {}
    merged.update(dict(config.get("defaults") or {}))
    for key in GLOBAL_SETTING_KEYS:
        if key in config:
            merged[key] = config[key]
    sizes = config.get("sizes") or {}
    if size not in sizes:
        raise ValueError(f"config has no settings for size {size!r}")
    merged.update(dict(sizes[size]))
    merged.update(dict(track_config.get("settings") or {}))
    unknown = sorted(set(merged) - SETTING_KEYS)
    if unknown:
        raise ValueError(
            f"unknown run settings: {', '.join(unknown)}; allowed: "
            f"{', '.join(sorted(SETTING_KEYS))}"
        )
    merged.setdefault("block_size", MODEL_SIZES[size].block_size)
    return merged


def _eval_sets_for(
    config: Mapping[str, Any],
    track_config: Mapping[str, Any],
    arm: str,
    root: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, str], str]:
    """`(sets, raw twins, labels, in_domain_name)` — `_split` twins for a split arm.

    The second mapping is keyed by the *same* names as the first and always points at the
    **raw** file: `heldout_dcs_split` -> `heldout_dcs.txt` for a split arm, and a set's own
    path for a raw one. Its character count is the BPC denominator every arm on that set
    shares (`lm/train.evaluate_bpc`).

    The third return value labels each set `in_domain`, `transfer` or `ood`, from the
    config's own grouping. Track 2 uses all three: `heldout_sangraha` is in-domain (the
    sample is 93.6% Sangraha), `heldout_dcs` is **transfer** there rather than in-domain —
    the DCS sentences it holds out are curated literary Sanskrit and the corpus is OCR'd
    web and book text — and the parallel sets are out-of-domain (docs/decisions.md,
    2026-09-06, "Track 2 gets an in-domain held-out set").

    `in_domain` may name one set or a list; the first is the one the reference crossing is
    measured on.
    """
    spec = dict(track_config.get("eval_sets") or config.get("eval_sets") or {})
    if "in_domain" not in spec:
        raise ValueError("config's eval_sets needs an 'in_domain' entry")
    unknown = sorted(set(spec) - set(EVAL_SET_LABELS))
    if unknown:
        raise ValueError(
            f"unknown eval_sets group(s): {', '.join(unknown)}; allowed: "
            f"{', '.join(EVAL_SET_LABELS)}"
        )
    eval_dir = resolve_path(
        str(track_config.get("eval_dir") or config.get("eval_dir") or DEFAULT_EVAL_DIR), root
    )
    suffix = SPLIT_SUFFIX if is_split_arm(arm) else ""
    grouped: dict[str, list[str]] = {}
    for label in EVAL_SET_LABELS:
        value = spec.get(label)
        if value is None:
            grouped[label] = []
        elif isinstance(value, list):
            grouped[label] = [str(name) for name in value]
        else:
            grouped[label] = [str(value)]
    if not grouped["in_domain"]:
        raise ValueError("config's eval_sets 'in_domain' is empty")
    names = [name for label in EVAL_SET_LABELS for name in grouped[label]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"eval_sets names {duplicates} under more than one label")
    sets = {f"{name}{suffix}": eval_dir / f"{name}{suffix}.txt" for name in names}
    raw_sets = {f"{name}{suffix}": eval_dir / f"{name}.txt" for name in names}
    labels = {
        f"{name}{suffix}": label
        for label in EVAL_SET_LABELS
        for name in grouped[label]
    }
    return sets, raw_sets, labels, f"{grouped['in_domain'][0]}{suffix}"


def enumerate_runs(config: Mapping[str, Any], root: Path | None = None) -> list[RunSpec]:
    """Every run this config describes, in track → size → arm → seed order.

    Raises rather than guessing on a config that names an unknown size, gives a split arm
    to a track with no split corpus, or leaves a run with neither a byte budget nor a step
    budget — each of which would otherwise surface as a strange number hours later.
    """
    where = repo_root() if root is None else root
    seeds = [int(seed) for seed in config.get("seeds") or [0]]
    out_dir = resolve_path(str(config["output_dir"]), where)
    cache_dir = resolve_path(
        str(config.get("cache_dir") or "outputs/05_lm_training/encoded"), where
    )
    specs: list[RunSpec] = []
    for track, track_config in (config.get("tracks") or {}).items():
        corpus_raw = resolve_path(str(track_config["corpus_raw"]), where)
        corpus_split = (
            resolve_path(str(track_config["corpus_split"]), where)
            if track_config.get("corpus_split")
            else None
        )
        raw_bytes = corpus_bytes(corpus_raw)
        split_bytes = corpus_bytes(corpus_split) if corpus_split is not None else 0
        epochs = track_config.get("epochs")
        # The budget comes from the *raw* corpus for every arm in the track, so that one
        # number describes the track; a split arm then spends it on its own longer text.
        max_bytes = int(round(float(epochs) * raw_bytes)) if epochs is not None else None
        for size in track_config["sizes"]:
            if size not in MODEL_SIZES:
                raise ValueError(f"{track}: unknown size {size!r}")
            settings = _settings_for(config, track_config, str(size))
            if max_bytes is None and settings.get("max_steps") is None:
                raise ValueError(
                    f"{track}/{size}: give the track an 'epochs' budget or the size a "
                    "'max_steps'; a run needs something to stop at"
                )
            for arm in track_config["arms"]:
                arm = str(arm)
                if is_split_arm(arm) and corpus_split is None:
                    raise ValueError(
                        f"{track}: {arm} is a split arm but the track has no corpus_split"
                    )
                corpus = corpus_split if is_split_arm(arm) else corpus_raw
                assert corpus is not None
                eval_sets, eval_raw_sets, eval_labels, in_domain = _eval_sets_for(
                    config, track_config, arm, where
                )
                for seed in seeds:
                    specs.append(
                        RunSpec(
                            track=str(track),
                            size=str(size),
                            arm=arm,
                            seed=seed,
                            corpus=corpus,
                            corpus_n_bytes=split_bytes if is_split_arm(arm) else raw_bytes,
                            eval_sets=eval_sets,
                            eval_raw_sets=eval_raw_sets,
                            eval_labels=eval_labels,
                            in_domain_set=in_domain,
                            settings=dict(settings),
                            out_dir=out_dir,
                            cache_dir=cache_dir,
                            max_bytes=max_bytes,
                        )
                    )
    return specs


def is_complete(spec: RunSpec) -> bool:
    """Whether `spec` has already been run under exactly this plan.

    Both files must be there: `results.json` alone could have been written by an earlier,
    differently configured sweep, and `sweep_run.json` records the plan hash that says
    which one. A sidecar that cannot be read counts as absent — re-running is cheap
    relative to reporting a number produced by an unknown config.
    """
    if not (spec.run_dir / "results.json").exists():
        return False
    sidecar = spec.run_dir / "sweep_run.json"
    if not sidecar.exists():
        return False
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("plan_hash")
    except (json.JSONDecodeError, OSError):  # pragma: no cover - unreadable sidecar
        return False
    return bool(recorded == spec.plan_hash())


def pending_runs(specs: Sequence[RunSpec]) -> list[RunSpec]:
    """The runs of `specs` that are not already finished under this plan."""
    return [spec for spec in specs if not is_complete(spec)]


def prune_checkpoint(run_dir: Path, *, keep: bool) -> bool:
    """Delete `run_dir/ckpt.pt` unless `keep`; returns whether a file was removed.

    Called only once `results.json` is written, so a killed sweep still resumes from the
    checkpoint of the run it was in the middle of. A 125M checkpoint is model plus AdamW
    state — about 1.5 GB — and 51 of them is not something to keep by accident.
    """
    ckpt = run_dir / "ckpt.pt"
    if keep or not ckpt.exists():
        return False
    ckpt.unlink()
    logger.info("pruned %s", ckpt)
    return True


# ------------------------------------------------------------------------------ dry run


@dataclass(frozen=True)
class PlanRow:
    """One row of the `--dry-run` table: a run and what it is projected to cost."""

    track: str
    size: str
    arm: str
    seed: int
    bytes_per_token: float
    max_bytes: int | None
    max_tokens: int
    params: int
    flops_est: float
    estimated: bool


def total_hours(rows: Sequence[PlanRow], tokens_per_s: float) -> float:
    """Wall-clock hours for `rows` at a flat `tokens_per_s`."""
    if tokens_per_s <= 0:
        raise ValueError(f"tokens_per_s must be positive, got {tokens_per_s}")
    return sum(row.max_tokens for row in rows) / tokens_per_s / 3600.0


def format_dry_run_table(
    rows: Sequence[PlanRow], *, tokens_per_s: float, gpu_speedup: float = DEFAULT_GPU_SPEEDUP
) -> str:
    """The projection table: one line per run, a subtotal per track, and a grand total.

    Two hour columns. The first is at the rate the caller measured (`--tokens-per-s`); the
    second is that rate multiplied by `--gpu-speedup`, which is **an assumption**, not a
    measurement — this project has never run this model on a datacentre GPU, and the ratio
    would in any case grow with model size, because the laptop's disadvantage is largest
    where the matmuls are largest. The FLOPs column is `6 × total parameters × tokens`
    (Kaplan et al.'s forward+backward rule) and is the number to re-derive an estimate from
    if the assumed speedup is not believed.

    A `*` on a row's bytes-per-token marks a figure *estimated* by sampling the corpus
    rather than read from a cached encoding: the run's real token budget will differ
    slightly.
    """
    header = (
        f"{'track':<7} {'size':<5} {'arm':<28} {'seed':>4} {'tokens':>14} "
        f"{'B/tok':>9} {'PFLOPs':>9} {'h @ rate':>10} {'h @ x' + str(int(gpu_speedup)):>10}"
    )
    lines = [
        f"projection at {tokens_per_s:,.0f} tokens/s; the x{gpu_speedup:g} column is an "
        "assumption about a rented GPU, not a measurement",
        header,
        "-" * len(header),
    ]

    def hours(tokens: int) -> tuple[float, float]:
        base = tokens / tokens_per_s / 3600.0
        return base, base / gpu_speedup

    def summary(label: str, subset: Sequence[PlanRow]) -> str:
        tokens = sum(row.max_tokens for row in subset)
        flops = sum(row.flops_est for row in subset)
        slow, fast = hours(tokens)
        return (
            f"{label:<42} {len(subset):>4} {tokens:>14,} {'':>9} {flops / 1e15:>9,.1f} "
            f"{slow:>10,.2f} {fast:>10,.2f}"
        )

    for track in dict.fromkeys(row.track for row in rows):  # tracks in plan order
        subset = [row for row in rows if row.track == track]
        for row in subset:
            slow, fast = hours(row.max_tokens)
            mark = "*" if row.estimated else " "
            lines.append(
                f"{row.track:<7} {row.size:<5} {row.arm:<28} {row.seed:>4} "
                f"{row.max_tokens:>14,} {row.bytes_per_token:>8.3f}{mark} "
                f"{row.flops_est / 1e15:>9,.1f} {slow:>10,.2f} {fast:>10,.2f}"
            )
        lines.append("-" * len(header))
        lines.append(summary(f"TOTAL {track} ({len(subset)} runs)", subset))
        lines.append("")
    lines.append(summary(f"GRAND TOTAL ({len(rows)} runs)", rows))
    lines.append("")
    lines.append("* bytes/token estimated by sampling the corpus (no cached encoding yet)")
    return "\n".join(lines)


def _params_for(spec: RunSpec, logical_vocab: int, cache: dict[tuple[str, int], int]) -> int:
    """Total parameters for `spec`'s size at `logical_vocab`, built once per combination."""
    key = (spec.size, logical_vocab)
    if key not in cache:
        config = TrainConfig.from_mapping(spec.train_mapping(max_tokens=1))
        model = build_model(config, logical_vocab)
        cache[key] = count_parameters(model).total
        del model
    return cache[key]


def _cached_bytes_per_token(spec: RunSpec) -> float | None:
    """The exact bytes-per-token from a cached encoding, or `None` if there is none.

    The cache is validated cheaply — the encoding must be of this corpus and record the
    same byte count — rather than by re-hashing a 755 MB file, because this is only ever
    used to make a projection. `ensure_encoded_corpus` does the full check before any run.
    """
    if not (spec.bin_path.exists() and spec.bin_path.with_suffix(".meta.json").exists()):
        return None
    try:
        cached = load_encoded_corpus(spec.bin_path)
    except (json.JSONDecodeError, KeyError, OSError):  # pragma: no cover - unreadable meta
        return None
    if cached.source_path != spec.corpus or cached.n_bytes != spec.corpus_n_bytes:
        return None
    return cached.bytes_per_token


def plan_rows(specs: Sequence[RunSpec], *, stride: int = DEFAULT_ESTIMATE_STRIDE) -> list[PlanRow]:
    """Turn `specs` into `PlanRow`s, using cached encodings where they exist.

    One tokenizer load and at most one corpus sample per (track, arm) pair; one model build
    per (size, vocabulary) pair. Everything else is arithmetic.
    """
    bpt: dict[tuple[str, str], tuple[float, bool]] = {}
    vocab: dict[str, int] = {}
    params: dict[tuple[str, int], int] = {}
    rows: list[PlanRow] = []
    for spec in specs:
        key = (spec.track, spec.arm)
        if key not in bpt:
            arm = load_tokenizer(spec.arm)
            vocab[spec.arm] = eos_id_for(arm) + 1
            exact = _cached_bytes_per_token(spec)
            if exact is not None:
                bpt[key] = (exact, False)
            else:
                logger.info(
                    "%s/%s: no cached encoding; sampling 1 line in %d of %s",
                    spec.track,
                    spec.arm,
                    stride,
                    spec.corpus.name,
                )
                sampled, n_lines = sample_bytes_per_token(arm, spec.corpus, stride)
                logger.info("%s: %.4f bytes/token over %d lines", spec.arm, sampled, n_lines)
                bpt[key] = (sampled, True)
        bytes_per_token, estimated = bpt[key]
        if spec.max_bytes is not None:
            max_tokens = token_budget(spec.max_bytes, bytes_per_token)
        else:
            config = TrainConfig.from_mapping(spec.train_mapping(max_tokens=None))
            max_tokens = config.total_steps() * config.tokens_per_step
        total_params = _params_for(spec, vocab[spec.arm], params)
        rows.append(
            PlanRow(
                track=spec.track,
                size=spec.size,
                arm=spec.arm,
                seed=spec.seed,
                bytes_per_token=bytes_per_token,
                max_bytes=spec.max_bytes,
                max_tokens=max_tokens,
                params=total_params,
                flops_est=6.0 * total_params * max_tokens,
                estimated=estimated,
            )
        )
    return rows


# --------------------------------------------------------------------------- running


def _budget_for(spec: RunSpec) -> tuple[int | None, float | None]:
    """Encode `spec`'s corpus if needed and convert its byte budget into tokens."""
    if spec.max_bytes is None:
        return None, None
    arm = load_tokenizer(spec.arm)
    corpus = ensure_encoded_corpus(arm, spec.corpus, spec.bin_path, eos_id=eos_id_for(arm))
    bytes_per_token = corpus.bytes_per_token
    return token_budget(spec.max_bytes, bytes_per_token), bytes_per_token


def run_one(spec: RunSpec, *, keep_checkpoints: bool) -> Path:
    """Train one run and write its `sweep_run.json`; returns the results path.

    The sidecar is written *after* the results, and holds the budget arithmetic that
    `results.json` has no field for — the byte budget, the arm's measured bytes-per-token
    and the token budget they produced — plus the plan hash that lets a later sweep skip
    this run.
    """
    max_tokens, bytes_per_token = _budget_for(spec)
    config = TrainConfig.from_mapping(spec.train_mapping(max_tokens))
    started = time.perf_counter()
    results_path = train(config)
    elapsed = time.perf_counter() - started
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "sweep_run.json").write_text(
        json.dumps(
            {
                "plan_hash": spec.plan_hash(),
                "track": spec.track,
                "size": spec.size,
                "arm": spec.arm,
                "seed": spec.seed,
                "corpus": str(spec.corpus),
                "corpus_n_bytes": spec.corpus_n_bytes,
                "in_domain_set": spec.in_domain_set,
                "eval_labels": dict(spec.eval_labels),
                "eval_raw_sets": {
                    name: str(path) for name, path in spec.eval_raw_sets.items()
                },
                "is_split_arm": spec.is_split,
                "max_bytes": spec.max_bytes,
                "bytes_per_token": bytes_per_token,
                "max_tokens": max_tokens,
                "elapsed_s": elapsed,
                **provenance(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prune_checkpoint(spec.run_dir, keep=keep_checkpoints)
    return results_path


def resolve_sweep_device(config: Mapping[str, Any]) -> tuple[str, str]:
    """The `(device, dtype)` every run of `config` will use, or a `RuntimeError`.

    Called before anything is encoded or trained. `resolve_device` raises when the named
    device is unavailable, and when `auto` would land on the CPU without `allow_cpu: true`;
    the smoke sweep sets that flag and the real sweep deliberately does not, so a GPU box
    whose driver did not come up fails in the first second rather than after encoding a
    755 MB corpus (docs/decisions.md, 2026-09-06, "Sweep hardening before paid GPU time").
    """
    device = resolve_device(
        str(config.get("device", "auto")), allow_cpu=bool(config.get("allow_cpu", False))
    )
    return device, resolve_dtype(str(config.get("dtype", "auto")), device)


def run_sweep(
    config: Mapping[str, Any], config_src: Path | None = None, root: Path | None = None
) -> list[Path]:
    """Run every pending run of `config` in order; returns the results paths written."""
    device, dtype = resolve_sweep_device(config)
    logger.info("device %s, dtype %s (allow_cpu=%s)", device, dtype, config.get("allow_cpu", False))
    specs = enumerate_runs(config, root)
    pending = pending_runs(specs)
    out_dir = resolve_path(str(config["output_dir"]), repo_root() if root is None else root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if config_src is not None:
        (out_dir / "config.yaml").write_text(
            config_src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    keep = bool(config.get("keep_checkpoints", False))
    logger.info("%d runs planned, %d pending", len(specs), len(pending))
    written: list[Path] = []
    for index, spec in enumerate(pending, start=1):
        logger.info(
            "[%d/%d] %s/%s/%s/seed%d",
            index,
            len(pending),
            spec.track,
            spec.size,
            spec.arm,
            spec.seed,
        )
        written.append(run_one(spec, keep_checkpoints=keep))
    logger.info("sweep finished: %d runs trained this invocation", len(written))
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("sweep.yaml"),
        help="sweep config YAML (default: the sweep.yaml beside this script)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan with token, FLOP and hour projections; train nothing",
    )
    parser.add_argument(
        "--tokens-per-s",
        type=float,
        default=17_300.0,
        help="token rate the hour projection assumes (default: this laptop's MPS rate)",
    )
    parser.add_argument(
        "--gpu-speedup",
        type=float,
        default=DEFAULT_GPU_SPEEDUP,
        help="assumed rented-GPU speedup for the second hour column (default: 50)",
    )
    parser.add_argument(
        "--estimate-stride",
        type=int,
        default=DEFAULT_ESTIMATE_STRIDE,
        help="dry run only: sample one line in N when a corpus has no cached encoding",
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
    if args.dry_run:
        specs = enumerate_runs(config)
        rows = plan_rows(specs, stride=args.estimate_stride)
        print(
            format_dry_run_table(
                rows, tokens_per_s=args.tokens_per_s, gpu_speedup=args.gpu_speedup
            )
        )
        out_dir = resolve_path(str(config["output_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sweep_plan.json").write_text(
            json.dumps(
                {
                    "tokens_per_s": args.tokens_per_s,
                    "gpu_speedup": args.gpu_speedup,
                    "rows": [
                        {
                            "track": row.track,
                            "size": row.size,
                            "arm": row.arm,
                            "seed": row.seed,
                            "bytes_per_token": row.bytes_per_token,
                            "bytes_per_token_estimated": row.estimated,
                            "max_bytes": row.max_bytes,
                            "max_tokens": row.max_tokens,
                            "params": row.params,
                            "flops_est": row.flops_est,
                        }
                        for row in rows
                    ],
                    **provenance(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("wrote %s", out_dir / "sweep_plan.json")
        return 0
    run_sweep(config, args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
