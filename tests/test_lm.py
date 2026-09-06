"""Tests for `sanskrit_tok.lm`: BPC arithmetic, corpus packing, batching, training.

Everything here is offline and CPU-only. The tokenizer arms used are `T7_byt5` (no files,
no network) and tiny `tokenizers`-trained BPE models written into `tmp_path` and picked up
through `SANSKRIT_TOK_TOKENIZER_DIR`, the same mechanism `tests/test_registry.py` uses.

The training tests run a `smoke`-size model (2 layers, 128 wide) for a handful of steps on
the 300-line fixture corpus, which is seconds on CPU; they assert the *plumbing* — files
written, fields present, BPC finite, the step counter continuing across a resume — never
that the loss reached any particular value, which five steps cannot establish.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from sanskrit_tok.lm.bpc import bits_per_char, bpc_from_token_nll, bpc_from_total
from sanskrit_tok.lm.config import MODEL_SIZES, TrainConfig, resolve_device, round_up_vocab
from sanskrit_tok.lm.data import (
    EncodedCorpus,
    arm_fingerprint,
    dtype_for,
    encode_corpus,
    ensure_encoded_corpus,
    eos_id_for,
    iter_batches,
    load_encoded_corpus,
    open_tokens,
    tokenise_text,
)
from sanskrit_tok.lm.train import (
    IGNORE_INDEX,
    build_model,
    evaluate_bpc,
    masked_cross_entropy,
    seed_everything,
    train,
)
from sanskrit_tok.tokenizers.registry import LoadedTokenizer, load_tokenizer

FIXTURES = Path(__file__).parent / "fixtures"
MINI_CORPUS = FIXTURES / "lm_mini_corpus.txt"

LN2 = math.log(2.0)


# ------------------------------------------------------------------------------- bpc


def test_bits_per_char_hand_computed() -> None:
    """Ten tokens each costing one bit (ln 2 nats) over twenty characters is 0.5 bits/char."""
    assert bits_per_char(10 * LN2, 20) == pytest.approx(0.5)


def test_bits_per_char_is_zero_for_a_perfect_model() -> None:
    assert bits_per_char(0.0, 20) == 0.0


def test_bits_per_char_rejects_a_zero_character_count() -> None:
    with pytest.raises(ValueError):
        bits_per_char(1.0, 0)


def test_bpc_from_token_nll_hand_computed() -> None:
    result = bpc_from_token_nll([LN2] * 10, 20)

    assert result["value"] == pytest.approx(0.5)
    assert result["n"] == 20
    assert result["unit"] == "bits/char"
    assert result["total_nats"] == pytest.approx(10 * LN2)
    assert result["n_tokens"] == 10
    assert result["bits_per_token"] == pytest.approx(1.0)
    assert "bits_per_byte" not in result


def test_bpc_from_token_nll_reports_bits_per_byte_when_given_a_byte_count() -> None:
    result = bpc_from_token_nll([LN2] * 10, 20, n_bytes=40)

    assert result["bits_per_byte"] == pytest.approx(0.25)
    assert result["n_bytes"] == 40
    assert result["value"] == pytest.approx(0.5)


def test_bpc_from_token_nll_rejects_an_empty_token_sequence() -> None:
    with pytest.raises(ValueError):
        bpc_from_token_nll([], 20)


# ------------------------------------------------------------------------------ dtype


def test_dtype_is_uint16_for_a_byte_vocabulary() -> None:
    assert dtype_for(257) == "uint16"


def test_dtype_is_uint16_for_a_64k_vocabulary() -> None:
    assert dtype_for(64001) == "uint16"


def test_dtype_widens_to_uint32_when_the_vocabulary_will_not_fit() -> None:
    assert dtype_for(70000) == "uint32"
    assert dtype_for(65535) == "uint32"


# --------------------------------------------------------------------- encode_corpus


def _mini_lines() -> list[str]:
    return MINI_CORPUS.read_text(encoding="utf-8").splitlines()


def test_eos_id_for_a_byte_arm_is_256() -> None:
    assert eos_id_for(load_tokenizer("T7_byt5")) == 256


def test_encode_corpus_with_the_byte_arm_counts_bytes_plus_one_eos_per_line(
    tmp_path: Path,
) -> None:
    arm = load_tokenizer("T7_byt5")
    lines = _mini_lines()

    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=eos_id_for(arm))

    assert encoded.n_lines == len(lines)
    assert encoded.n_chars == sum(len(line) for line in lines)
    assert encoded.n_bytes == sum(len(line.encode("utf-8")) for line in lines)
    assert encoded.n_tokens == encoded.n_bytes + len(lines)
    assert encoded.eos_id == 256
    assert encoded.logical_vocab == 257
    assert encoded.dtype == "uint16"


def test_encode_corpus_writes_one_eos_at_the_end_of_every_line(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    lines = _mini_lines()

    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=256)
    tokens = open_tokens(encoded)

    cursor = 0
    for line in lines[:20]:
        width = len(line.encode("utf-8"))
        assert list(tokens[cursor : cursor + width]) == list(line.encode("utf-8"))
        assert tokens[cursor + width] == 256
        cursor += width + 1
    assert int((tokens[:] == 256).sum()) == len(lines)


def test_encode_corpus_writes_a_meta_json_beside_the_bin(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")

    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=256)

    meta = json.loads(encoded.meta_path.read_text(encoding="utf-8"))
    assert encoded.meta_path == tmp_path / "mini.meta.json"
    for key in (
        "arm",
        "dtype",
        "eos_id",
        "logical_vocab",
        "n_bytes",
        "n_chars",
        "n_lines",
        "n_tokens",
        "source_path",
        "source_sha256",
    ):
        assert key in meta
    assert meta["arm"] == "T7_byt5"
    assert len(meta["source_sha256"]) == 64


def test_load_encoded_corpus_round_trips_the_meta(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    written = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=256)

    assert load_encoded_corpus(tmp_path / "mini.bin") == written


def test_ensure_encoded_corpus_reuses_a_matching_bin(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    out = tmp_path / "mini.bin"
    first = ensure_encoded_corpus(arm, MINI_CORPUS, out, eos_id=256)
    stamp = out.stat().st_mtime_ns

    second = ensure_encoded_corpus(arm, MINI_CORPUS, out, eos_id=256)

    assert second == first
    assert out.stat().st_mtime_ns == stamp


def test_ensure_encoded_corpus_re_encodes_when_the_source_changed(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    source = tmp_path / "corpus.txt"
    source.write_text("rAmaH gacCati\n", encoding="utf-8")
    out = tmp_path / "c.bin"
    first = ensure_encoded_corpus(arm, source, out, eos_id=256)

    source.write_text("rAmaH gacCati vanam\n", encoding="utf-8")
    second = ensure_encoded_corpus(arm, source, out, eos_id=256)

    assert second.n_tokens > first.n_tokens
    assert second.source_sha256 != first.source_sha256


def test_encode_corpus_with_a_trained_arm_matches_a_direct_encode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path / "toks"))
    _write_tiny_bpe_tokenizer(tmp_path / "toks" / "T1_bpe_raw_32k" / "tokenizer.json")
    arm = load_tokenizer("T1_bpe_raw_32k")
    lines = _mini_lines()
    expected = sum(len(arm.encode(line)) for line in lines) + len(lines)

    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=eos_id_for(arm))

    assert encoded.n_tokens == expected
    assert encoded.eos_id == arm.vocab_size
    assert encoded.logical_vocab == arm.vocab_size + 1
    tokens = open_tokens(encoded)
    assert int(tokens.max()) <= arm.vocab_size


def test_encode_corpus_widens_the_dtype_for_a_large_vocabulary(tmp_path: Path) -> None:
    arm = LoadedTokenizer(
        name="T_fake",
        source_id="fake",
        vocab_size=70_000,
        _encode=lambda text: [len(text) % 70_000],
        family="T_fake",
    )
    source = tmp_path / "c.txt"
    source.write_text("rAmaH\ngacCati\n", encoding="utf-8")

    encoded = encode_corpus(arm, source, tmp_path / "c.bin", eos_id=eos_id_for(arm))

    assert encoded.dtype == "uint32"
    assert open_tokens(encoded).dtype == np.uint32


def test_encode_corpus_without_an_eos_writes_only_the_arm_ids(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")

    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=None)

    assert encoded.eos_id is None
    assert encoded.logical_vocab == 256
    assert encoded.n_tokens == encoded.n_bytes


def test_encode_corpus_skips_blank_lines(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    source = tmp_path / "c.txt"
    source.write_text("rAmaH\n\n   \ngacCati\n", encoding="utf-8")

    encoded = encode_corpus(arm, source, tmp_path / "c.bin", eos_id=256)

    assert encoded.n_lines == 2
    assert encoded.n_chars == len("rAmaH") + len("gacCati")


def test_bytes_per_token_is_the_corpus_byte_to_token_ratio(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=256)

    assert encoded.bytes_per_token == pytest.approx(encoded.n_bytes / encoded.n_tokens)


# ------------------------------------------------------------------------ tokenise_text


def test_tokenise_text_appends_an_eos_per_line_and_counts_characters() -> None:
    arm = load_tokenizer("T7_byt5")

    tokenised = tokenise_text(arm, MINI_CORPUS, eos_id=256, max_chars=None)

    lines = _mini_lines()
    assert tokenised.n_lines == len(lines)
    assert tokenised.n_chars == sum(len(line) for line in lines)
    assert len(tokenised.ids) == tokenised.n_bytes + len(lines)
    assert tokenised.ids[-1] == 256


def test_tokenise_text_truncates_at_whole_lines() -> None:
    arm = load_tokenizer("T7_byt5")
    lines = _mini_lines()

    tokenised = tokenise_text(arm, MINI_CORPUS, eos_id=256, max_chars=100)

    assert 0 < tokenised.n_lines < len(lines)
    assert tokenised.n_chars <= 100 + len(lines[tokenised.n_lines - 1])
    assert tokenised.n_chars == sum(len(line) for line in lines[: tokenised.n_lines])


# ------------------------------------------------------------------------- iter_batches


def _encode_mini(tmp_path: Path) -> EncodedCorpus:
    arm = load_tokenizer("T7_byt5")
    return encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=256)


def test_iter_batches_yields_the_requested_shapes(tmp_path: Path) -> None:
    encoded = _encode_mini(tmp_path)
    batches = iter_batches(
        encoded.bin_path, block_size=16, batch_size=4, rng=np.random.default_rng(0)
    )

    x, y = next(batches)

    assert tuple(x.shape) == (4, 16)
    assert tuple(y.shape) == (4, 16)


def test_iter_batches_targets_are_the_inputs_shifted_by_one(tmp_path: Path) -> None:
    encoded = _encode_mini(tmp_path)
    tokens = open_tokens(encoded)
    batches = iter_batches(
        encoded.bin_path, block_size=8, batch_size=2, rng=np.random.default_rng(1)
    )

    x, y = next(batches)

    assert x[:, 1:].tolist() == y[:, :-1].tolist()
    assert int(x.max()) <= 256
    assert int(tokens.max()) == 256


def test_iter_batches_is_deterministic_given_the_same_seed(tmp_path: Path) -> None:
    encoded = _encode_mini(tmp_path)
    first = [
        next(iter_batches(encoded.bin_path, 16, 4, np.random.default_rng(7)))[0].tolist()
        for _ in range(1)
    ]
    second = [
        next(iter_batches(encoded.bin_path, 16, 4, np.random.default_rng(7)))[0].tolist()
        for _ in range(1)
    ]

    assert first == second


def test_iter_batches_differs_across_seeds(tmp_path: Path) -> None:
    encoded = _encode_mini(tmp_path)
    a = next(iter_batches(encoded.bin_path, 16, 4, np.random.default_rng(7)))[0].tolist()
    b = next(iter_batches(encoded.bin_path, 16, 4, np.random.default_rng(8)))[0].tolist()

    assert a != b


def test_iter_batches_rejects_a_corpus_shorter_than_the_block(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")
    source = tmp_path / "c.txt"
    source.write_text("ab\n", encoding="utf-8")
    encoded = encode_corpus(arm, source, tmp_path / "c.bin", eos_id=256)

    with pytest.raises(ValueError):
        next(iter_batches(encoded.bin_path, 64, 2, np.random.default_rng(0)))


# ------------------------------------------------------------------------------ config


def test_model_sizes_carry_the_three_documented_configurations() -> None:
    assert set(MODEL_SIZES) == {"smoke", "50M", "125M"}
    assert (MODEL_SIZES["smoke"].n_layer, MODEL_SIZES["smoke"].n_embd) == (2, 128)
    assert (MODEL_SIZES["50M"].n_layer, MODEL_SIZES["50M"].n_embd) == (8, 512)
    assert (MODEL_SIZES["125M"].n_layer, MODEL_SIZES["125M"].n_embd) == (12, 768)
    assert MODEL_SIZES["smoke"].block_size == 256
    assert MODEL_SIZES["125M"].block_size == 1024


def test_round_up_vocab_pads_to_a_multiple_of_64() -> None:
    assert round_up_vocab(257) == 320
    assert round_up_vocab(64001) == 64064
    assert round_up_vocab(64) == 64


def test_resolve_device_returns_cpu_when_asked() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_train_config_from_mapping_resolves_paths_against_the_root(tmp_path: Path) -> None:
    config = TrainConfig.from_mapping(
        {
            "arm": "T7_byt5",
            "track": "track1",
            "size": "smoke",
            "corpus": "corpus.txt",
            "eval_sets": {"heldout": "heldout.txt"},
            "out_dir": "outputs/05_lm_training",
            "cache_dir": "outputs/05_lm_training/encoded",
            "max_steps": 5,
        },
        root=tmp_path,
    )

    assert config.corpus == tmp_path / "corpus.txt"
    assert config.eval_sets["heldout"] == tmp_path / "heldout.txt"
    assert config.block_size == 256  # from the `smoke` size
    assert config.run_dir == tmp_path / "outputs/05_lm_training/track1/smoke/T7_byt5/seed0"


def test_train_config_round_trips_through_its_own_dict(tmp_path: Path) -> None:
    """`to_dict` is what `results.json` and `config.yaml` record, so it has to be enough to
    rebuild the run: a field missing from it would be a run that cannot be reproduced."""
    config = TrainConfig.from_mapping(
        {
            "arm": "T7_byt5",
            "track": "track1",
            "size": "smoke",
            "corpus": "corpus.txt",
            "eval_sets": {"heldout": "heldout.txt"},
            "max_steps": 5,
            "seed": 3,
            "grad_accum": 2,
        },
        root=tmp_path,
    )

    assert TrainConfig.from_mapping(config.to_dict(), root=tmp_path) == config


def test_train_config_rejects_an_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="learnin_rate"):
        TrainConfig.from_mapping(
            {
                "arm": "T7_byt5",
                "track": "t",
                "size": "smoke",
                "corpus": "c.txt",
                "max_steps": 1,
                "learnin_rate": 1e-3,
            },
            root=tmp_path,
        )


def test_train_config_rejects_an_unknown_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_mapping(
            {"arm": "T7_byt5", "track": "t", "size": "huge", "corpus": "c.txt", "max_steps": 1},
            root=tmp_path,
        )


def test_train_config_requires_a_step_or_token_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_mapping(
            {"arm": "T7_byt5", "track": "t", "size": "smoke", "corpus": "c.txt"},
            root=tmp_path,
        )


# ------------------------------------------------------------------------------- train


def _smoke_config(
    tmp_path: Path,
    *,
    max_steps: int,
    track: str = "test",
    eval_every: int = 5,
    warmup_steps: int = 2,
    resume: bool = True,
) -> TrainConfig:
    return TrainConfig.from_mapping(
        {
            "arm": "T7_byt5",
            "track": track,
            "size": "smoke",
            "corpus": str(MINI_CORPUS),
            "eval_sets": {"mini": str(MINI_CORPUS)},
            "out_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "encoded"),
            "block_size": 32,
            "batch_size": 4,
            "eval_batch_size": 4,
            "eval_every": eval_every,
            "eval_max_chars": 2000,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "lr": 1e-3,
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
            "resume": resume,
        },
        root=tmp_path,
    )


def test_train_writes_results_curve_and_checkpoint(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=5)

    results_path = train(config)

    run_dir = config.run_dir
    assert results_path == run_dir / "results.json"
    assert (run_dir / "curve.jsonl").exists()
    assert (run_dir / "ckpt.pt").exists()
    assert (run_dir / "config.yaml").exists()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["steps"] == 5
    assert math.isfinite(results["bpc"]["mini"]["value"])
    assert results["bpc"]["mini"]["unit"] == "bits/char"
    assert results["params"]["total"] > results["params"]["non_embedding"] > 0
    assert results["params"]["embedding"] > 0
    assert results["tokens_seen"] == 5 * 4 * 32
    assert results["bytes_seen_est"] > 0
    assert results["flops_est"] > 0
    assert results["throughput"]["tokens_per_s"] > 0
    assert results["hardware"]["device"] == "cpu"
    assert results["seed"] == 0
    assert "git_commit" in results
    assert results["curve_summary"]["mini"]["min_bpc"] > 0


def test_train_curve_has_a_row_at_step_zero_and_at_every_eval(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=10)

    train(config)

    rows = [
        json.loads(line)
        for line in (config.run_dir / "curve.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [0, 5, 10]
    for row in rows:
        assert math.isfinite(row["bpc"]["mini"])
        for key in ("tokens_seen", "bytes_seen_est", "epochs", "flops_est", "lr", "elapsed_s"):
            assert key in row
    assert rows[-1]["tokens_per_s"] > 0


def test_train_resumes_from_its_checkpoint_and_continues_the_step_counter(
    tmp_path: Path,
) -> None:
    first = _smoke_config(tmp_path, max_steps=5)
    train(first)
    rows_before = [
        json.loads(line)
        for line in (first.run_dir / "curve.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    second = _smoke_config(tmp_path, max_steps=10)
    results_path = train(second)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (second.run_dir / "curve.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert results["steps"] == 10
    assert results["resumed_from_step"] == 5
    assert [row["step"] for row in rows] == [0, 5, 10]
    assert rows_before == rows[: len(rows_before)]
    assert results["tokens_seen"] == 10 * 4 * 32


def test_train_is_a_no_op_when_the_checkpoint_already_reached_the_budget(
    tmp_path: Path,
) -> None:
    config = _smoke_config(tmp_path, max_steps=5)
    train(config)
    first = json.loads((config.run_dir / "results.json").read_text(encoding="utf-8"))

    train(config)
    second = json.loads((config.run_dir / "results.json").read_text(encoding="utf-8"))

    assert second["steps"] == first["steps"] == 5
    assert second["resumed_from_step"] == 5


def test_train_records_the_logical_vocabulary_and_its_padded_model_vocabulary(
    tmp_path: Path,
) -> None:
    config = _smoke_config(tmp_path, max_steps=5)

    results = json.loads(train(config).read_text(encoding="utf-8"))

    assert results["model"]["logical_vocab"] == 257
    assert results["model"]["model_vocab_size"] == 320
    assert results["model"]["n_layer"] == 2
    assert results["model"]["block_size"] == 32


def _write_tiny_bpe_tokenizer(path: Path, vocab_size: int = 120) -> None:
    """A small SLP1 BPE `tokenizer.json` at `path`, trained on the mini corpus fixture."""
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers import models, pre_tokenizers, trainers

    tokenizer = RawTokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(_mini_lines(), trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


# ------------------------------------------------------------------- padded-vocab masking


def _random_byte_text(path: Path, *, n_lines: int = 40, width: int = 60, seed: int = 1) -> Path:
    """A file of pseudo-random ASCII letters — text with nothing in it to learn."""
    rng = random.Random(seed)
    lines = [
        "".join(chr(rng.randint(ord("a"), ord("z"))) for _ in range(width))
        for _ in range(n_lines)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_evaluate_bpc_softmaxes_over_the_logical_vocabulary_not_the_padded_one(
    tmp_path: Path,
) -> None:
    """With uniform logits the score is `ln(257)`, not `ln(320)`: the 63 pad rows are gone.

    The output weights are zeroed so the model's distribution is exactly uniform over
    whatever the softmax normalises across, which turns "did the masking happen" into an
    exact arithmetic assertion rather than a comparison against a noisy random init.
    `wte` is tied to `lm_head` in nanoGPT, so zeroing it zeroes the logits.
    """
    arm = load_tokenizer("T7_byt5")
    text = _random_byte_text(tmp_path / "rand.txt")
    config = _smoke_config(tmp_path, max_steps=1)
    model = build_model(config, 257)
    with torch.no_grad():
        model.transformer.wte.weight.zero_()

    masked = evaluate_bpc(
        model,
        arm,
        text,
        eos_id=256,
        logical_vocab=257,
        block_size=32,
        batch_size=4,
        device="cpu",
        dtype_name="float32",
        max_chars=None,
    )
    unmasked = evaluate_bpc(
        model,
        arm,
        text,
        eos_id=256,
        logical_vocab=320,
        block_size=32,
        batch_size=4,
        device="cpu",
        dtype_name="float32",
        max_chars=None,
    )

    assert masked["total_nats"] / masked["n_tokens"] == pytest.approx(math.log(257), abs=1e-6)
    assert unmasked["total_nats"] / unmasked["n_tokens"] == pytest.approx(
        math.log(320), abs=1e-6
    )


def test_masking_removes_exactly_the_padded_rows_from_a_freshly_initialised_model(
    tmp_path: Path,
) -> None:
    """A real init scores `ln(320)` unmasked and `ln(257)` masked, up to one shared offset.

    A freshly initialised `GPT` is *near* uniform, not uniform: its logits have a spread of
    a few tenths, which moves the per-token score off `ln(V)` by ~0.03 nats. That offset is
    a property of the initialisation and is identical in both calls, so what is asserted to
    1e-3 is the *difference* the masking makes — which is exactly `ln(320) - ln(257)`, the
    probability mass the padded rows were taking.
    """
    arm = load_tokenizer("T7_byt5")
    text = _random_byte_text(tmp_path / "rand.txt")
    config = _smoke_config(tmp_path, max_steps=1)
    seed_everything(0)
    model = build_model(config, 257)

    def nats_per_token(logical_vocab: int) -> float:
        result = evaluate_bpc(
            model,
            arm,
            text,
            eos_id=256,
            logical_vocab=logical_vocab,
            block_size=32,
            batch_size=4,
            device="cpu",
            dtype_name="float32",
            max_chars=None,
        )
        return result["total_nats"] / result["n_tokens"]

    masked, unmasked = nats_per_token(257), nats_per_token(320)

    assert unmasked - masked == pytest.approx(math.log(320) - math.log(257), abs=1e-3)
    assert masked == pytest.approx(math.log(257), abs=0.05)
    assert masked < unmasked


def test_masked_cross_entropy_ignores_the_padding_target(tmp_path: Path) -> None:
    """`IGNORE_INDEX` positions contribute nothing, and the pad columns get no probability."""
    logits = torch.zeros(1, 3, 320)
    targets = torch.tensor([[5, IGNORE_INDEX, 7]])

    total = masked_cross_entropy(logits.clone(), targets, 257, reduction="sum")
    mean = masked_cross_entropy(logits.clone(), targets, 257, reduction="mean")

    assert float(total) == pytest.approx(2 * math.log(257), abs=1e-5)
    assert float(mean) == pytest.approx(math.log(257), abs=1e-5)


def test_evaluate_bpc_scores_a_stream_that_is_not_a_multiple_of_the_block(
    tmp_path: Path,
) -> None:
    """Every target scored once, padding scored never — against a hand-run log-softmax.

    The text is chosen so the token stream does not fill a whole number of `block_size`
    windows, which is the case the padded final window exists for. The expected total is
    computed here by running the model over the same windows and summing the log-softmax of
    each target, so the assertion is against arithmetic rather than against the evaluator's
    own reduction.
    """
    arm = load_tokenizer("T7_byt5")
    text = tmp_path / "short.txt"
    text.write_text("rAmaH gacCati vanam\ntataH kim\n", encoding="utf-8")
    lines = ["rAmaH gacCati vanam", "tataH kim"]
    block_size = 8
    config = _smoke_config(tmp_path, max_steps=1)
    seed_everything(0)
    model = build_model(config, 257)
    model.eval()

    ids: list[int] = []
    for line in lines:
        ids.extend(line.encode("utf-8"))
        ids.append(256)
    n_chars = sum(len(line) for line in lines)
    assert len(ids) % block_size != 0, "the fixture must not fill whole windows"

    stream = [256, *ids]
    n_windows = -(-len(ids) // block_size)
    padded = stream + [256] * (n_windows * block_size + 1 - len(stream))
    x = torch.tensor(padded[:-1]).view(n_windows, block_size)
    y_full = padded[1:]
    expected = 0.0
    with torch.no_grad():
        logits, _ = model(x, torch.zeros_like(x))
    log_probs = torch.log_softmax(logits[..., :257].float(), dim=-1)
    for window in range(n_windows):
        for position in range(block_size):
            flat = window * block_size + position
            if flat >= len(ids):  # padding: not a target
                continue
            expected -= float(log_probs[window, position, y_full[flat]])

    result = evaluate_bpc(
        model,
        arm,
        text,
        eos_id=256,
        logical_vocab=257,
        block_size=block_size,
        batch_size=4,
        device="cpu",
        dtype_name="float32",
        max_chars=None,
    )

    assert result["n_tokens"] == len(ids)
    assert result["n"] == n_chars
    assert result["total_nats"] == pytest.approx(expected, rel=1e-5)
    assert result["value"] == pytest.approx(expected / math.log(2.0) / n_chars, rel=1e-6)


# ---------------------------------------------------------------------- curve handling


def _curve_steps(config: TrainConfig) -> list[int]:
    return [
        json.loads(line)["step"]
        for line in (config.run_dir / "curve.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_two_fresh_runs_into_one_directory_leave_only_the_second_runs_curve(
    tmp_path: Path,
) -> None:
    config = _smoke_config(tmp_path, max_steps=3, eval_every=3, resume=False)

    train(config)
    train(config)

    assert _curve_steps(config) == [0, 3]


def test_a_start_with_no_checkpoint_clears_a_leftover_curve(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=3, eval_every=3)
    train(config)
    (config.run_dir / "ckpt.pt").unlink()

    train(config)

    assert _curve_steps(config) == [0, 3]


def test_resume_leaves_no_duplicate_curve_rows_even_after_extra_rows(tmp_path: Path) -> None:
    """A crash between an eval row and its checkpoint leaves rows the checkpoint disowns."""
    config = _smoke_config(tmp_path, max_steps=5, eval_every=5)
    train(config)
    with (config.run_dir / "curve.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 7, "bpc": {"mini": 9.0}}) + "\n")

    train(_smoke_config(tmp_path, max_steps=10, eval_every=5))

    assert _curve_steps(config) == [0, 5, 10]


def test_a_resumed_run_reproduces_the_uninterrupted_curve(tmp_path: Path) -> None:
    """Ten steps in one process and 5 + 5 in two produce the same numbers.

    `warmup_steps` is set above the step budget so the learning-rate schedule is pure
    warmup and therefore does not depend on `total_steps`: without that, stopping at 5 with
    `max_steps: 5` would compress the cosine decay into five steps and the two runs would
    legitimately differ (the Task 2 report's resume case (a)). With it, the only thing that
    can make the trajectories differ is a resume that failed to restore state.
    """
    whole = _smoke_config(tmp_path, max_steps=10, eval_every=5, warmup_steps=50, track="whole")
    train(whole)
    reference = [
        json.loads(line)
        for line in (whole.run_dir / "curve.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    train(_smoke_config(tmp_path, max_steps=5, eval_every=5, warmup_steps=50, track="split"))
    resumed_config = _smoke_config(
        tmp_path, max_steps=10, eval_every=5, warmup_steps=50, track="split"
    )
    results = json.loads(train(resumed_config).read_text(encoding="utf-8"))
    resumed_curve = (resumed_config.run_dir / "curve.jsonl").read_text(encoding="utf-8")
    resumed = [json.loads(line) for line in resumed_curve.splitlines() if line.strip()]

    assert results["resumed_from_step"] == 5
    assert [row["step"] for row in resumed] == [row["step"] for row in reference] == [0, 5, 10]
    for row, expected_row in zip(resumed, reference, strict=True):
        assert row["bpc"]["mini"] == pytest.approx(expected_row["bpc"]["mini"], abs=1e-6)
        assert row["tokens_seen"] == expected_row["tokens_seen"]
        assert row["lr"] == pytest.approx(expected_row["lr"], rel=1e-12)
        if expected_row["loss_train"] is not None:
            assert row["loss_train"] == pytest.approx(expected_row["loss_train"], abs=1e-6)


def test_the_checkpoint_carries_the_torch_generator_state(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_steps=3, eval_every=3)

    train(config)

    checkpoint = torch.load(config.run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert "torch_rng_state" in checkpoint
    assert "cpu" in checkpoint["torch_rng_state"]
    assert isinstance(checkpoint["torch_rng_state"]["cpu"], torch.Tensor)


# ------------------------------------------------------------- encoded-corpus cache key


def test_ensure_encoded_corpus_re_encodes_when_the_tokenizer_file_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same arm name, same corpus, different `tokenizer.json` — the cache must not be reused."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path / "toks"))
    tokenizer_path = tmp_path / "toks" / "T1_bpe_raw_32k" / "tokenizer.json"
    _write_tiny_bpe_tokenizer(tokenizer_path, vocab_size=120)
    source = tmp_path / "c.txt"
    source.write_text("rAmaH gacCati vanam\ntataH kim\n", encoding="utf-8")
    out = tmp_path / "c.bin"
    first_arm = load_tokenizer("T1_bpe_raw_32k")
    first = ensure_encoded_corpus(first_arm, source, out, eos_id=eos_id_for(first_arm))

    _write_tiny_bpe_tokenizer(tokenizer_path, vocab_size=80)
    second_arm = load_tokenizer("T1_bpe_raw_32k")
    second = ensure_encoded_corpus(second_arm, source, out, eos_id=eos_id_for(second_arm))

    assert second.arm_sha256 != first.arm_sha256
    assert second.arm_vocab_size != first.arm_vocab_size
    assert second.n_tokens != first.n_tokens
    assert second.source_sha256 == first.source_sha256


def test_encoded_corpus_meta_records_the_arm_fingerprint(tmp_path: Path) -> None:
    arm = load_tokenizer("T7_byt5")

    encoded = encode_corpus(arm, MINI_CORPUS, tmp_path / "mini.bin", eos_id=256)

    meta = json.loads(encoded.meta_path.read_text(encoding="utf-8"))
    assert meta["arm_source_id"] == "bytes/utf-8"
    assert meta["arm_vocab_size"] == 256
    assert meta["arm_sha256"] is None
    assert arm_fingerprint(arm) == ("bytes/utf-8", 256, None)


def test_ensure_encoded_corpus_re_encodes_a_meta_without_a_fingerprint(tmp_path: Path) -> None:
    """A `.meta.json` from before the fingerprint existed cannot prove its arm; re-encode."""
    arm = load_tokenizer("T7_byt5")
    out = tmp_path / "mini.bin"
    encoded = ensure_encoded_corpus(arm, MINI_CORPUS, out, eos_id=256)
    meta = json.loads(encoded.meta_path.read_text(encoding="utf-8"))
    for key in ("arm_source_id", "arm_vocab_size", "arm_sha256"):
        del meta[key]
    encoded.meta_path.write_text(json.dumps(meta), encoding="utf-8")
    stamp = out.stat().st_mtime_ns

    reloaded = ensure_encoded_corpus(arm, MINI_CORPUS, out, eos_id=256)

    assert reloaded.arm_source_id == "bytes/utf-8"
    assert out.stat().st_mtime_ns != stamp


def test_bpc_from_total_matches_the_per_token_form() -> None:
    from_total = bpc_from_total(10 * LN2, 10, 20, n_bytes=40)
    from_tokens = bpc_from_token_nll([LN2] * 10, 20, n_bytes=40)

    assert from_total["value"] == pytest.approx(from_tokens["value"])
    assert from_total["bits_per_token"] == pytest.approx(from_tokens["bits_per_token"])
    assert from_total["bits_per_byte"] == pytest.approx(from_tokens["bits_per_byte"])
    assert from_total["n_tokens"] == 10


def test_bpc_from_total_rejects_a_zero_token_count() -> None:
    with pytest.raises(ValueError):
        bpc_from_total(1.0, 0, 20)
