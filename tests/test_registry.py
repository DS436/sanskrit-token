"""Tests for `sanskrit_tok.tokenizers.registry`: the T0 arms and their adapters.

Everything here runs offline against fake tokenizer objects, except: the `T0_o200k` test,
which needs tiktoken's BPE file (skipped when it cannot be fetched and is not cached), and
the two Hugging Face arms, which pull real model repositories and so are skipped unless
`SANSKRIT_TOK_NETWORK_TESTS` is set (plan Global Constraints: no network in tests).
"""

import os
from pathlib import Path
from typing import Any

import pytest

from sanskrit_tok.tokenizers.base import Tokenizer
from sanskrit_tok.tokenizers.registry import (
    REGISTRY,
    T0_GEMMA3_CANDIDATES,
    T0_GPT2_CANDIDATES,
    T0_LLAMA4_CANDIDATES,
    T3_BRAHMIC131K_MODEL_ID,
    T3_INDICSUPER_CANDIDATES,
    T3_SARVAM_CANDIDATES,
    T3_SUTRA_CANDIDATES,
    HFAdapter,
    LoadedTokenizer,
    TiktokenAdapter,
    TokenizerUnavailable,
    list_tokenizers,
    load_tokenizer,
    trained_tokenizer_path,
)

ARMS = ("T0_gemma3", "T0_llama4", "T0_o200k")

#: Every arm the registry must carry after this task (CLAUDE.md §6, exp02 plan Tasks 2/6).
ALL_ARMS = (
    "E1_bpe_32k",
    "E1_bpe_64k",
    "E1_unigram_32k",
    "E1_unigram_64k",
    "T0_gemma3",
    "T0_gpt2",
    "T0_llama4",
    "T0_o200k",
    "T1_bpe_raw_32k",
    "T1_bpe_raw_64k",
    "T2_unigram_raw_32k",
    "T2_unigram_raw_64k",
    "T3_brahmic131k",
    "T3_indicsuper",
    "T3_sarvam",
    "T3_sutra",
    "T4_bpe_split_32k",
    "T4_bpe_split_64k",
    "T4_unigram_split_32k",
    "T4_unigram_split_64k",
)

T3_ARMS = ("T3_brahmic131k", "T3_indicsuper", "T3_sarvam", "T3_sutra")

#: The sandhi-split family (CLAUDE.md §6, exp03 plan Task 3): the same two algorithms at
#: the same two vocabulary sizes as T1/T2, trained on the sandhi-split SLP1 corpus. The
#: throughput rule selected the FULL training corpus (docs/decisions.md, "Splitter
#: throughput measured"), so there are no `_sub` arms.
T4_ARMS = (
    "T4_bpe_split_32k",
    "T4_bpe_split_64k",
    "T4_unigram_split_32k",
    "T4_unigram_split_64k",
)

#: The matched English control family (CLAUDE.md §6, docs/decisions.md "Add a matched
#: English control family E1 for TPP"): same algorithms and vocabulary sizes as T1/T2,
#: trained on the English side of the same corpus.
E1_ARMS = ("E1_bpe_32k", "E1_bpe_64k", "E1_unigram_32k", "E1_unigram_64k")

TRAINED_ARMS = (
    "T1_bpe_raw_32k",
    "T1_bpe_raw_64k",
    "T2_unigram_raw_32k",
    "T2_unigram_raw_64k",
    *E1_ARMS,
    *T4_ARMS,
)

NETWORK_TESTS = pytest.mark.skipif(
    not os.environ.get("SANSKRIT_TOK_NETWORK_TESTS"),
    reason="set SANSKRIT_TOK_NETWORK_TESTS=1 to exercise the Hugging Face downloads",
)

#: One Sanskrit word in Devanagari ("saMskftam") and its English gloss, used as a smoke
#: test that every arm encodes both scripts to a non-empty id sequence.
DEVANAGARI = "संस्कृतम्"
ENGLISH = "Sanskrit"


# --------------------------------------------------------------------------- dataclass


def test_loaded_tokenizer_delegates_encode_to_its_callable() -> None:
    tok = LoadedTokenizer(
        name="T_fake",
        source_id="fake/id",
        vocab_size=7,
        _encode=lambda text: [len(text)],
    )
    assert tok.encode("abcd") == [4]
    assert tok.name == "T_fake"
    assert tok.source_id == "fake/id"
    assert tok.vocab_size == 7


def test_loaded_tokenizer_satisfies_the_tokenizer_protocol() -> None:
    tok = LoadedTokenizer(name="T_fake", source_id="fake/id", vocab_size=1, _encode=lambda _: [0])
    assert isinstance(tok, Tokenizer)


# --------------------------------------------------------------------------- adapters


class _FakeTiktoken:
    """Records the keyword arguments `TiktokenAdapter` passes through."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.calls.append((text, kwargs))
        return [1, 2, 3]


class _FakeHF:
    """Records the keyword arguments `HFAdapter` passes through."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.calls.append((text, kwargs))
        return [4, 5]


def test_tiktoken_adapter_disallows_special_tokens() -> None:
    fake = _FakeTiktoken()
    adapter = TiktokenAdapter(fake)
    assert adapter("hello") == [1, 2, 3]
    assert fake.calls == [("hello", {"disallowed_special": ()})]


def test_hf_adapter_suppresses_special_tokens() -> None:
    fake = _FakeHF()
    adapter = HFAdapter(fake)
    assert adapter("hello") == [4, 5]
    assert fake.calls == [("hello", {"add_special_tokens": False})]


# --------------------------------------------------------------------------- registry


def test_list_tokenizers_returns_every_arm_name_sorted() -> None:
    assert list_tokenizers() == sorted(ALL_ARMS)


def test_list_tokenizers_filters_by_family() -> None:
    assert list_tokenizers(family="T3") == sorted(T3_ARMS)


def test_registry_carries_twenty_arms() -> None:
    assert len(list_tokenizers()) == 20


def test_list_tokenizers_filters_the_english_control_family() -> None:
    assert list_tokenizers(family="E1") == sorted(E1_ARMS)


def test_list_tokenizers_filters_the_sandhi_split_family() -> None:
    assert list_tokenizers(family="T4") == sorted(T4_ARMS)


def test_no_subset_arms_are_registered() -> None:
    """The throughput rule selected the full training corpus, so the matched-subset arms
    it would otherwise have required (`T1_bpe_raw_32k_sub` and friends) do not exist."""
    assert not [name for name in REGISTRY if name.endswith("_sub")]


def test_registry_keys_are_exactly_the_listed_arms() -> None:
    assert sorted(REGISTRY) == sorted(ALL_ARMS)


def test_load_tokenizer_rejects_an_unknown_name_and_says_what_is_known() -> None:
    with pytest.raises(KeyError) as excinfo:
        load_tokenizer("nope")
    message = str(excinfo.value)
    assert "nope" in message
    for arm in ALL_ARMS:
        assert arm in message


def test_candidate_lists_start_with_the_official_gated_ids() -> None:
    assert T0_LLAMA4_CANDIDATES[0] == "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    assert T0_GEMMA3_CANDIDATES[0] == "google/gemma-3-4b-it"


# ------------------------------------------------------------------- candidate walking


def test_hf_arms_try_candidates_in_order_and_take_the_first_that_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first two ids fail (as gated ids do without a token); the third wins."""
    import sanskrit_tok.tokenizers.registry as registry

    attempted: list[str] = []

    def fake_from_pretrained(model_id: str, token: str | None) -> Any:
        attempted.append(model_id)
        if model_id in T0_LLAMA4_CANDIDATES[:2]:
            raise OSError(f"{model_id} is gated")
        return _FakeHF()

    monkeypatch.setattr(registry, "_hf_from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(registry, "_hf_vocab_size", lambda _: 128256)

    tok = load_tokenizer("T0_llama4")

    assert attempted == list(T0_LLAMA4_CANDIDATES[:3])
    assert tok.name == "T0_llama4"
    assert tok.source_id == T0_LLAMA4_CANDIDATES[2]
    assert tok.vocab_size == 128256
    assert tok.encode("hello") == [4, 5]
    # `attempted` records the failed ids then the winner, in the order they were tried.
    assert tok.attempted == tuple(T0_LLAMA4_CANDIDATES[:3])
    assert tok.family == "T0"


def test_hf_arms_raise_with_every_error_when_no_candidate_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sanskrit_tok.tokenizers.registry as registry

    def always_fails(model_id: str, token: str | None) -> Any:
        raise OSError(f"{model_id} is gated")

    monkeypatch.setattr(registry, "_hf_from_pretrained", always_fails)

    with pytest.raises(RuntimeError) as excinfo:
        load_tokenizer("T0_gemma3")
    message = str(excinfo.value)
    for candidate in T0_GEMMA3_CANDIDATES:
        assert candidate in message
    assert "gated" in message


def test_hf_loader_passes_hf_token_only_when_the_environment_sets_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sanskrit_tok.tokenizers.registry as registry

    seen: list[str | None] = []

    def fake_from_pretrained(model_id: str, token: str | None) -> Any:
        seen.append(token)
        return _FakeHF()

    monkeypatch.setattr(registry, "_hf_from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(registry, "_hf_vocab_size", lambda _: 10)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    load_tokenizer("T0_gemma3")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    load_tokenizer("T0_gemma3")

    assert seen == [None, "hf_secret"]


# ------------------------------------------------------------------------------ arms


def _skip_if_tiktoken_is_offline() -> LoadedTokenizer:
    """Skip only on a download/cache failure; any other error is a real bug.

    `OSError` covers the whole "the BPE file is not here and cannot be fetched" family
    (`requests`' `HTTPError`/`ConnectionError` derive from it, as do plain filesystem
    failures). Catching bare `Exception` would turn any defect in `_load_tiktoken_arm`
    into a green skip.
    """
    try:
        return load_tokenizer("T0_o200k")
    except OSError as exc:  # pragma: no cover - depends on cache/network state
        pytest.skip(f"tiktoken could not fetch its BPE file: {exc}")


def test_t0_o200k_loads_and_encodes() -> None:
    tok = _skip_if_tiktoken_is_offline()
    assert tok.name == "T0_o200k"
    assert tok.source_id == "o200k_base"
    assert tok.vocab_size > 100_000
    ids = tok.encode("hello world")
    assert ids
    assert all(isinstance(i, int) for i in ids)
    assert tok.encode(DEVANAGARI)
    assert tok.encode(ENGLISH)


@NETWORK_TESTS
@pytest.mark.parametrize(
    ("arm", "candidates"),
    [("T0_llama4", T0_LLAMA4_CANDIDATES), ("T0_gemma3", T0_GEMMA3_CANDIDATES)],
)
def test_hf_arms_load_from_a_declared_candidate(arm: str, candidates: tuple[str, ...]) -> None:
    tok = load_tokenizer(arm)
    assert tok.name == arm
    assert tok.source_id in candidates
    assert tok.vocab_size > 1_000
    assert tok.encode(DEVANAGARI)
    assert tok.encode(ENGLISH)
    assert isinstance(tok, Tokenizer)


# --------------------------------------------------------------------- T0_gpt2 (offline)


def test_t0_gpt2_candidates_and_family() -> None:
    assert T0_GPT2_CANDIDATES == ("openai-community/gpt2",)


def test_t0_gpt2_has_family_t0(monkeypatch: pytest.MonkeyPatch) -> None:
    import sanskrit_tok.tokenizers.registry as registry

    monkeypatch.setattr(registry, "_hf_from_pretrained", lambda model_id, token: _FakeHF())
    monkeypatch.setattr(registry, "_hf_vocab_size", lambda _: 50257)

    tok = load_tokenizer("T0_gpt2")

    assert tok.family == "T0"
    assert tok.source_id == "openai-community/gpt2"
    assert tok.attempted == ("openai-community/gpt2",)


# ------------------------------------------------------------------------ TokenizerUnavailable


def test_trained_arm_raises_tokenizer_unavailable_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))

    with pytest.raises(TokenizerUnavailable) as excinfo:
        load_tokenizer("T1_bpe_raw_32k")

    expected_path = trained_tokenizer_path("T1_bpe_raw_32k")
    assert str(expected_path) in str(excinfo.value)


def test_tokenizer_unavailable_is_a_runtime_error() -> None:
    assert issubclass(TokenizerUnavailable, RuntimeError)


def test_hf_arm_with_no_candidate_raises_tokenizer_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sanskrit_tok.tokenizers.registry as registry

    def always_fails(model_id: str, token: str | None) -> Any:
        raise OSError(f"{model_id} is gated")

    monkeypatch.setattr(registry, "_hf_from_pretrained", always_fails)

    with pytest.raises(TokenizerUnavailable):
        load_tokenizer("T0_gemma3")


# ------------------------------------------------------------------------- trained (T1/T2)


def _write_tiny_bpe_tokenizer(path: Path, vocab_size: int = 50) -> None:
    """A ten-line SLP1 toy corpus, trained to a `models.BPE` tokenizer.json at `path`."""
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers import models, pre_tokenizers, trainers

    lines = [
        "rAmaH gacCati vanam",
        "kfzRa uvAca",
        "devI vadati",
        "nftyati bAlakaH",
        "gajaH calati",
        "sUryaH udayati",
        "candraH BAti",
        "nadI vahati",
        "vfkzaH tizWati",
        "pakzI uqqIyate",
    ]
    tokenizer = RawTokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tokenizer.train_from_iterator(lines, trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def test_trained_arm_loads_from_tokenizer_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    path = tmp_path / "T1_bpe_raw_32k" / "tokenizer.json"
    _write_tiny_bpe_tokenizer(path, vocab_size=50)

    tok = load_tokenizer("T1_bpe_raw_32k")

    assert tok.family == "T1"
    assert tok.vocab_size == 50
    assert tok.source_id == str(path)
    assert tok.attempted == (str(path),)
    ids = tok.encode("rAmaH")
    assert ids
    assert all(isinstance(i, int) for i in ids)


def test_trained_tokenizer_path_resolves_under_the_tokenizer_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    assert trained_tokenizer_path("T2_unigram_raw_64k") == tmp_path / "T2_unigram_raw_64k" / (
        "tokenizer.json"
    )


@pytest.mark.parametrize("arm", TRAINED_ARMS)
def test_every_trained_arm_is_registered(arm: str) -> None:
    assert arm in REGISTRY


@pytest.mark.parametrize("arm", E1_ARMS)
def test_english_control_arm_loads_from_tokenizer_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str
) -> None:
    """E1 is file-backed exactly like T1/T2 — same loader, same directory layout — so it
    is absent until `train_tokenizers.py` has written its `tokenizer.json`."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    path = tmp_path / arm / "tokenizer.json"
    _write_tiny_bpe_tokenizer(path, vocab_size=50)

    tok = load_tokenizer(arm)

    assert tok.family == "E1"
    assert tok.vocab_size == 50
    assert tok.source_id == str(path)
    assert tok.encode("Rama goes")


@pytest.mark.parametrize("arm", T4_ARMS)
def test_sandhi_split_arm_loads_from_tokenizer_json_with_family_t4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str
) -> None:
    """T4 is file-backed exactly like T1/T2/E1: same loader, same directory layout, so it
    is absent until `train_tokenizers.py` has written its `tokenizer.json`."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    path = tmp_path / arm / "tokenizer.json"
    _write_tiny_bpe_tokenizer(path, vocab_size=50)

    tok = load_tokenizer(arm)

    assert tok.family == "T4"
    assert tok.vocab_size == 50
    assert tok.source_id == str(path)
    assert tok.attempted == (str(path),)
    assert tok.encode("tad api")


@pytest.mark.parametrize("arm", T4_ARMS)
def test_untrained_sandhi_split_arm_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    with pytest.raises(TokenizerUnavailable) as excinfo:
        load_tokenizer(arm)
    assert str(trained_tokenizer_path(arm)) in str(excinfo.value)


@pytest.mark.parametrize("arm", E1_ARMS)
def test_untrained_english_control_arm_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    with pytest.raises(TokenizerUnavailable) as excinfo:
        load_tokenizer(arm)
    assert str(trained_tokenizer_path(arm)) in str(excinfo.value)


# --------------------------------------------------------------------------------- T3 arms


def test_t3_candidate_constants() -> None:
    assert T3_SARVAM_CANDIDATES == ("sarvamai/sarvam-1",)
    assert T3_SUTRA_CANDIDATES == ("TWO/sutra-mlt256-v2",)
    assert T3_BRAHMIC131K_MODEL_ID == "theschoolofai/BrahmicTokenizer-131K"
    assert T3_INDICSUPER_CANDIDATES == (
        "krutrim-ai-labs/IndicSuperTokenizer",
        "ai4bharat/IndicSuperTokenizer",
        "ai4bharat/indic-super-tokenizer",
    )


@NETWORK_TESTS
@pytest.mark.parametrize("arm", ["T0_gpt2"])
def test_t0_gpt2_loads_for_real(arm: str) -> None:
    tok = load_tokenizer(arm)
    assert tok.vocab_size == 50257
    assert tok.family == "T0"
    assert tok.encode(DEVANAGARI)
    assert tok.encode(ENGLISH)


@NETWORK_TESTS
@pytest.mark.parametrize("arm", T3_ARMS)
def test_each_t3_arm_loads_or_is_reported_unavailable(arm: str) -> None:
    try:
        tok = load_tokenizer(arm)
    except TokenizerUnavailable as exc:
        pytest.skip(f"{arm}: not available ({exc})")
        return
    assert tok.vocab_size > 30_000
    assert tok.family == "T3"
    assert tok.encode(DEVANAGARI)
    assert tok.encode(ENGLISH)
