"""Tests for `sanskrit_tok.tokenizers.registry`: the T0 arms and their adapters.

Everything here runs offline against fake tokenizer objects, except: the `T0_o200k` test,
which needs tiktoken's BPE file (skipped when it cannot be fetched and is not cached), and
the two Hugging Face arms, which pull real model repositories and so are skipped unless
`SANSKRIT_TOK_NETWORK_TESTS` is set (plan Global Constraints: no network in tests).
"""

import os
from typing import Any

import pytest

from sanskrit_tok.tokenizers.base import Tokenizer
from sanskrit_tok.tokenizers.registry import (
    REGISTRY,
    T0_GEMMA3_CANDIDATES,
    T0_LLAMA4_CANDIDATES,
    HFAdapter,
    LoadedTokenizer,
    TiktokenAdapter,
    list_tokenizers,
    load_tokenizer,
)

ARMS = ("T0_gemma3", "T0_llama4", "T0_o200k")

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


def test_list_tokenizers_returns_the_three_arm_names_sorted() -> None:
    assert list_tokenizers() == sorted(ARMS)


def test_registry_keys_are_exactly_the_listed_arms() -> None:
    assert sorted(REGISTRY) == sorted(ARMS)


def test_load_tokenizer_rejects_an_unknown_name_and_says_what_is_known() -> None:
    with pytest.raises(KeyError) as excinfo:
        load_tokenizer("nope")
    message = str(excinfo.value)
    assert "nope" in message
    for arm in ARMS:
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
    try:
        return load_tokenizer("T0_o200k")
    except Exception as exc:  # pragma: no cover - depends on cache/network state
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
