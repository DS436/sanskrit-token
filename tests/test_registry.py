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

from sanskrit_tok.tokenizers.base import Tokenizer, TokenizerWithSpans, spans_cover_text
from sanskrit_tok.tokenizers.registry import (
    E1_ARMS as REGISTRY_E1_ARMS,
)
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
    _variant,
    list_tokenizers,
    load_tokenizer,
    trained_tokenizer_path,
)

ARMS = ("T0_gemma3", "T0_llama4", "T0_o200k")

#: The twelve Experiment 04 arms, all trained on the DCS training split (docs/decisions.md,
#: 2026-09-05, "Experiment 04: DCS is the gold source ..."). `_dcs` marks the corpus,
#: `_oracle_dcs` marks the gold-segmented variant of it; `T5`/`T6` are the
#: morpheme-constrained (MorphBPE-hard) families.
T5_ARMS = (
    "T5_morphbpe_raw_32k_dcs",
    "T5_morphbpe_raw_64k_dcs",
    "T5_morphbpe_rawseg_32k_dcs",
    "T5_morphbpe_rawseg_64k_dcs",
)
T6_ARMS = ("T6_morphbpe_split_32k_dcs", "T6_morphbpe_split_64k_dcs")
DCS_ARMS = (
    "T1_bpe_raw_32k_dcs",
    "T1_bpe_raw_64k_dcs",
    "T2_unigram_raw_32k_dcs",
    "T2_unigram_raw_64k_dcs",
    "T4_bpe_split_32k_oracle_dcs",
    "T4_bpe_split_64k_oracle_dcs",
    "T4_unigram_split_32k_oracle_dcs",
    "T4_unigram_split_64k_oracle_dcs",
    *T5_ARMS,
    *T6_ARMS,
)

#: Arm -> the `variant` label its `LoadedTokenizer` must carry.
DCS_VARIANTS = {
    name: ("oracle_dcs" if name.endswith("_oracle_dcs") else "dcs") for name in DCS_ARMS
}

#: Every arm the registry must carry after this task (CLAUDE.md §6, exp02 plan Tasks 2/6).
ALL_ARMS = (
    *REGISTRY_E1_ARMS,
    "T0_gemma3",
    "T0_gpt2",
    "T0_llama4",
    "T0_o200k",
    "T1_bpe_raw_32k",
    "T1_bpe_raw_64k",
    "T1_bpe_raw_128k",
    "T2_unigram_raw_32k",
    "T2_unigram_raw_64k",
    "T2_unigram_raw_128k",
    "T3_brahmic131k",
    "T3_indicsuper",
    "T3_sarvam",
    "T3_sutra",
    "T4_bpe_split_32k",
    "T4_bpe_split_64k",
    "T4_unigram_split_32k",
    "T4_unigram_split_64k",
    "T7_byt5",
    *DCS_ARMS,
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
#: English control family E1 for TPP" and, for the `_bm` half, 2026-09-08 "Byte-matched
#: English control arms"): same algorithms and vocabulary sizes as T1/T2, trained on the
#: English side of the same corpus — the whole of it for the pair-matched arms, a
#: byte-count-matched subsample of it for the `_bm` arms. Imported from the registry rather
#: than retyped, so this file cannot disagree with it about which arms the family has; the
#: membership itself is pinned by `test_the_english_control_family_has_both_halves`.
E1_ARMS = REGISTRY_E1_ARMS

TRAINED_ARMS = (
    "T1_bpe_raw_32k",
    "T1_bpe_raw_64k",
    "T1_bpe_raw_128k",
    "T2_unigram_raw_32k",
    "T2_unigram_raw_64k",
    "T2_unigram_raw_128k",
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


def test_registry_carries_thirty_five_arms() -> None:
    assert len(list_tokenizers()) == 45


def test_list_tokenizers_filters_the_english_control_family() -> None:
    assert list_tokenizers(family="E1") == sorted(E1_ARMS)


def test_the_english_control_family_has_both_halves_at_three_sizes() -> None:
    """Six pair-matched arms and six byte-matched twins (CLAUDE.md §6): a `_bm` arm exists
    for every pair-matched one, and neither half carries a size the other lacks."""
    pair_matched = [name for name in E1_ARMS if not name.endswith("_bm")]
    byte_matched = [name for name in E1_ARMS if name.endswith("_bm")]
    assert sorted(pair_matched) == sorted(
        [
            "E1_bpe_32k",
            "E1_bpe_64k",
            "E1_bpe_128k",
            "E1_unigram_32k",
            "E1_unigram_64k",
            "E1_unigram_128k",
        ]
    )
    assert sorted(byte_matched) == sorted(f"{name}_bm" for name in pair_matched)


def test_a_byte_matched_control_arm_carries_no_corpus_variant_suffix() -> None:
    """`_bm` names how much English an arm saw, not which corpus it came from, so it must
    not be read as a training-corpus variant the way `_dcs` is."""
    assert all(_variant(name) == "" for name in E1_ARMS)


def test_list_tokenizers_filters_the_sandhi_split_family() -> None:
    """`T4` now spans both the Experiment 03 arms (ByT5-split parallel corpora) and the
    Experiment 04 oracle arms (gold-split DCS); the `variant` filter separates them."""
    assert list_tokenizers(family="T4") == sorted(
        (*T4_ARMS, *[n for n in DCS_ARMS if n.startswith("T4_")])
    )
    assert list_tokenizers(family="T4", variant="") == sorted(T4_ARMS)


def test_list_tokenizers_filters_the_morpheme_constrained_families() -> None:
    assert list_tokenizers(family="T5") == sorted(T5_ARMS)
    assert list_tokenizers(family="T6") == sorted(T6_ARMS)


def test_list_tokenizers_filters_by_variant() -> None:
    assert list_tokenizers(variant="oracle_dcs") == sorted(
        n for n in DCS_ARMS if n.endswith("_oracle_dcs")
    )
    assert list_tokenizers(variant="dcs") == sorted(
        n for n in DCS_ARMS if not n.endswith("_oracle_dcs")
    )


@pytest.mark.parametrize("arm", DCS_ARMS)
def test_every_dcs_arm_is_registered(arm: str) -> None:
    assert arm in REGISTRY


@pytest.mark.parametrize("arm", DCS_ARMS)
def test_dcs_arm_loads_from_tokenizer_json_with_its_family_and_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str
) -> None:
    """The twelve Experiment 04 arms are file-backed exactly like T1/T2/E1/T4: same
    loader, same directory layout, absent until `experiments/04_morph_constrained/
    train_tokenizers.py` has written a `tokenizer.json`."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    path = tmp_path / arm / "tokenizer.json"
    _write_tiny_bpe_tokenizer(path, vocab_size=50)

    tok = load_tokenizer(arm)

    assert tok.family == arm.split("_", 1)[0]
    assert tok.variant == DCS_VARIANTS[arm]
    assert tok.source_id == str(path)
    assert tok.supports_spans
    assert tok.encode("tad api")


# --------------------------------------------------------------------------- T7 bytes


def test_byt5_arm_is_registered_with_a_byte_vocabulary() -> None:
    tok = load_tokenizer("T7_byt5")

    assert tok.name == "T7_byt5"
    assert tok.family == "T7"
    assert tok.variant == ""
    assert tok.vocab_size == 256
    assert tok.source_id == "bytes/utf-8"
    assert tok.attempted == ("bytes",)


def test_byt5_arm_encodes_utf8_byte_values() -> None:
    tok = load_tokenizer("T7_byt5")

    assert tok.encode("tad api") == list(b"tad api")
    assert tok.encode(DEVANAGARI) == list(DEVANAGARI.encode("utf-8"))
    assert max(tok.encode(DEVANAGARI)) < 256


def test_byt5_arm_needs_no_files_and_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike every other arm, `T7_byt5` is a rule, not an artefact: no `tokenizer.json`
    to find and no repository to download, so it loads under any tokenizer directory."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", "/nonexistent")

    assert load_tokenizer("T7_byt5").vocab_size == 256


def test_byt5_arm_reports_one_span_per_character() -> None:
    tok = load_tokenizer("T7_byt5")
    text = "tad api"

    spans = tok.spans(text)

    assert tok.supports_spans
    assert spans_cover_text(text, spans)
    assert spans == [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]


def test_byt5_arm_groups_the_bytes_of_a_multibyte_character_into_one_span() -> None:
    """Three UTF-8 bytes make one Devanagari character, and a boundary inside a character
    is not a segmentation decision, so the character is one span rather than three."""
    tok = load_tokenizer("T7_byt5")

    spans = tok.spans(DEVANAGARI)

    assert spans_cover_text(DEVANAGARI, spans)
    assert spans == [(index, index + 1) for index in range(len(DEVANAGARI))]
    assert len(tok.encode(DEVANAGARI)) == 3 * len(DEVANAGARI)


def test_a_non_dcs_arm_has_an_empty_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    _write_tiny_bpe_tokenizer(tmp_path / "T4_bpe_split_32k" / "tokenizer.json", vocab_size=50)

    assert load_tokenizer("T4_bpe_split_32k").variant == ""


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


# ------------------------------------------------------------------------------ spans


#: The Experiment 04 spans sanity sentence: one sandhi-fused word (`tadapi` = `tad` +
#: `api`) and two unfused ones, in SLP1 and in the Devanagari it round-trips from.
SPANS_SLP1 = "tadapi rAmaH gacCati"
SPANS_DEVANAGARI = "तदपि रामः गच्छति"

#: Every arm whose spans the Experiment 04 plan names, across all three adapter kinds.
SPANS_ARMS = (
    "T0_o200k",
    "T0_gemma3",
    "T0_llama4",
    "T3_sarvam",
    "T1_bpe_raw_64k",
    "T4_bpe_split_64k",
)


class _FakeFastHF:
    """A `transformers` fast tokenizer, as far as `HFAdapter.spans` can tell.

    `__call__` is the offsets path (`return_offsets_mapping=True`); `encode` is the id
    path. The offsets deliberately include the space before `world`, which is what a
    byte-level fast tokenizer really returns and what the adapter has to trim.
    """

    is_fast = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((text, kwargs))
        return {"input_ids": [4, 5], "offset_mapping": [(0, 5), (5, 11)]}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [4, 5]


def test_loaded_tokenizer_without_spans_reports_it_and_raises() -> None:
    tok = LoadedTokenizer(name="T_fake", source_id="fake/id", vocab_size=1, _encode=lambda _: [0])
    assert tok.supports_spans is False
    with pytest.raises(TokenizerUnavailable):
        tok.spans("abc")


def test_loaded_tokenizer_with_spans_satisfies_the_spans_protocol() -> None:
    tok = LoadedTokenizer(
        name="T_fake",
        source_id="fake/id",
        vocab_size=1,
        _encode=lambda _: [0],
        _spans=lambda text: [(0, len(text))],
    )
    assert tok.supports_spans is True
    assert isinstance(tok, TokenizerWithSpans)
    assert tok.spans("abc") == [(0, 3)]


def test_hf_adapter_spans_trims_whitespace_and_passes_the_offsets_flag() -> None:
    fake = _FakeFastHF()
    adapter = HFAdapter(fake)
    assert adapter.spans("hello world") == [(0, 5), (6, 11)]
    assert fake.calls == [
        ("hello world", {"add_special_tokens": False, "return_offsets_mapping": True})
    ]


def test_hf_adapter_spans_refuses_a_slow_tokenizer() -> None:
    class _SlowHF:
        is_fast = False

        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1]

    with pytest.raises(TokenizerUnavailable):
        HFAdapter(_SlowHF()).spans("hello")


def test_slow_hf_arm_reloads_fast_for_spans_and_records_the_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`T0_gemma3` is the arm this path exists for (docs/decisions.md, "Token spans for
    MorphScore deferred to Experiment 04"): a slow SentencePiece load can count tokens but
    cannot report offsets, so the first `spans` call reloads the same id with
    `use_fast=True` and suffixes `source_id` with `+fast`."""
    import sanskrit_tok.tokenizers.registry as registry

    class _SlowHF:
        is_fast = False

        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1]

    reloaded: list[str] = []

    def fake_fast(model_id: str, token: str | None) -> Any:
        reloaded.append(model_id)
        return _FakeFastHF()

    monkeypatch.setattr(registry, "_hf_from_pretrained", lambda model_id, token: _SlowHF())
    monkeypatch.setattr(registry, "_hf_from_pretrained_fast", fake_fast)
    monkeypatch.setattr(registry, "_hf_vocab_size", lambda _: 256000)

    tok = load_tokenizer("T0_gemma3")
    assert tok.source_id == T0_GEMMA3_CANDIDATES[0]
    assert tok.supports_spans is True

    assert tok.spans("hello world") == [(0, 5), (6, 11)]
    assert reloaded == [T0_GEMMA3_CANDIDATES[0]]
    assert tok.source_id == f"{T0_GEMMA3_CANDIDATES[0]}+fast"

    # The reload happens once and is cached.
    assert tok.spans("hello world") == [(0, 5), (6, 11)]
    assert reloaded == [T0_GEMMA3_CANDIDATES[0]]


def test_slow_hf_arm_reports_spans_unavailable_when_the_fast_reload_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sanskrit_tok.tokenizers.registry as registry

    class _SlowHF:
        is_fast = False

        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1]

    def fails(model_id: str, token: str | None) -> Any:
        raise ValueError("Couldn't instantiate the backend tokenizer")

    monkeypatch.setattr(registry, "_hf_from_pretrained", lambda model_id, token: _SlowHF())
    monkeypatch.setattr(registry, "_hf_from_pretrained_fast", fails)
    monkeypatch.setattr(registry, "_hf_vocab_size", lambda _: 256000)

    tok = load_tokenizer("T0_gemma3")
    with pytest.raises(TokenizerUnavailable) as excinfo:
        tok.spans("hello")
    assert "backend tokenizer" in str(excinfo.value)
    # Counting tokens still works; only spans are unavailable.
    assert tok.encode("hello") == [1]


def test_tokenizers_adapter_spans_trims_the_metaspace_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `Metaspace` pre-tokenizer attaches the preceding space to the following token, so
    its raw offsets tile the whole string; the adapter trims that space away."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    path = tmp_path / "T1_bpe_raw_32k" / "tokenizer.json"
    _write_tiny_bpe_tokenizer(path, vocab_size=80)

    tok = load_tokenizer("T1_bpe_raw_32k")

    text = "rAmaH gacCati"
    assert tok.supports_spans is True
    spans = tok.spans(text)
    assert spans_cover_text(text, spans)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)


def test_tiktoken_adapter_spans_cover_hello_world() -> None:
    tok = _skip_if_tiktoken_is_offline()
    assert tok.supports_spans is True
    spans = tok.spans("hello world")
    assert spans_cover_text("hello world", spans)
    assert spans == [(0, 5), (6, 11)]


def test_tiktoken_adapter_spans_cover_multibyte_text() -> None:
    """Devanagari is three UTF-8 bytes per character, so the byte cursor and the character
    cursor diverge; the map from one to the other is what this pins."""
    tok = _skip_if_tiktoken_is_offline()
    assert spans_cover_text(SPANS_DEVANAGARI, tok.spans(SPANS_DEVANAGARI))


def test_tiktoken_span_of_a_token_ending_mid_character_extends_to_the_character_end() -> None:
    """A byte-level token can end inside a multi-byte character. The byte->character map
    rounds such a boundary *up* to the character's end, so the character is covered once
    (by the earlier token) and the later token's span is empty and dropped."""
    from sanskrit_tok.tokenizers.registry import _byte_boundaries_to_char_spans

    text = "अb"  # 3 bytes + 1 byte
    # Three "tokens": bytes [0,1), [1,3), [3,4) — the first two split the Devanagari char.
    spans = _byte_boundaries_to_char_spans(text, [1, 2, 1])
    assert spans == [(0, 1), (1, 2)]
    assert spans_cover_text(text, spans)


@NETWORK_TESTS
@pytest.mark.parametrize("arm", SPANS_ARMS)
@pytest.mark.parametrize("text", [SPANS_SLP1, SPANS_DEVANAGARI])
def test_every_spans_arm_covers_the_sanity_sentence(arm: str, text: str) -> None:
    try:
        tok = load_tokenizer(arm)
    except TokenizerUnavailable as exc:
        pytest.skip(f"{arm}: not available ({exc})")
        return
    assert tok.supports_spans is True
    spans = tok.spans(text)
    assert spans, f"{arm} produced no spans"
    assert spans_cover_text(text, spans), f"{arm} spans do not cover {text!r}: {spans}"
