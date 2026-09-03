"""Tests for `sanskrit_tok.tokenizers.corpus`, `train_bpe`, `train_unigram` (exp02 Task 4).

TDD per the task brief: `build_training_corpus` builds the SLP1 training file with a
leakage assertion and exact dedup across sources; `train_bpe`/`train_unigram` train HF
`tokenizers` arms from it, and the result must load through the registry's file-backed T1
arm. Everything here runs offline: `tests/fixtures/slp1_corpus_mini.txt` is 200 SLP1 lines
(three real Sāmayik-mini sentences plus 197 made-up ones, no real evaluation sentence —
see `test_slp1_corpus_mini_fixture_is_not_in_the_exclusion_list`).
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from sanskrit_tok.data.exclusion import (
    EXCLUSION_PATH,
    LeakageError,
    assert_not_excluded,
    load_exclusion_hashes,
    sentence_hash,
)
from sanskrit_tok.encoding import from_slp1
from sanskrit_tok.tokenizers.corpus import build_training_corpus
from sanskrit_tok.tokenizers.registry import load_tokenizer
from sanskrit_tok.tokenizers.train_bpe import train_bpe
from sanskrit_tok.tokenizers.train_unigram import train_unigram

FIXTURE = Path(__file__).parent / "fixtures" / "slp1_corpus_mini.txt"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOKENIZERS_PY = (
    REPO_ROOT / "experiments" / "02_tpp_parallel" / "train_tokenizers.py"
)


def _load_train_tokenizers_module() -> ModuleType:
    """`train_tokenizers.py` is a script under `experiments/`, not an importable
    package module (same convention as `tests/test_exp01.py`), so load it by path."""
    spec = importlib.util.spec_from_file_location("exp02_train_tokenizers", TRAIN_TOKENIZERS_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_tokenizers = _load_train_tokenizers_module()


def _fixture_line_count() -> int:
    return len(FIXTURE.read_text(encoding="utf-8").splitlines())


# --------------------------------------------------------------------- build_training_corpus


def test_build_training_corpus_dedups_across_sources_and_writes_manifest(
    tmp_path: Path,
) -> None:
    sources = {
        "a": ["रामः गच्छति", "सीता वदति"],
        "b": ["रामः गच्छति", "बालकः पठति"],  # first sentence duplicates source "a"
    }
    out_path = tmp_path / "corpus.txt"

    manifest = build_training_corpus(sources, out_path, frozenset())

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # 4 sentences in, 1 exact duplicate removed
    assert manifest["n_in"] == {"a": 2, "b": 2}
    assert manifest["n_out"] == 3
    assert manifest["n_dedup_removed"] == 1
    assert manifest["exclusion_hashes"] == 0
    assert manifest["sha256"] == hashlib.sha256(out_path.read_bytes()).hexdigest()


def test_build_training_corpus_preserves_first_occurrence_order(tmp_path: Path) -> None:
    sources = {
        "a": ["रामः गच्छति", "सीता वदति"],
        "b": ["सीता वदति", "बालकः पठति"],
    }
    out_path = tmp_path / "corpus.txt"

    build_training_corpus(sources, out_path, frozenset())

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == lines[0]  # first source's first sentence stays first
    assert len(lines) == 3


def test_build_training_corpus_drops_empty_lines_without_counting_them_as_dedup(
    tmp_path: Path,
) -> None:
    sources = {"a": ["रामः", "   ", ""]}
    out_path = tmp_path / "corpus.txt"

    manifest = build_training_corpus(sources, out_path, frozenset())

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["rAmaH"]
    assert manifest["n_in"] == {"a": 3}
    assert manifest["n_out"] == 1
    assert manifest["n_dedup_removed"] == 0


def test_build_training_corpus_raises_leakage_error_for_excluded_sentence(
    tmp_path: Path,
) -> None:
    excluded = frozenset({sentence_hash("रामः गच्छति")})
    sources = {"a": ["रामः गच्छति", "सीता वदति"]}

    with pytest.raises(LeakageError):
        build_training_corpus(sources, tmp_path / "corpus.txt", excluded)


def test_build_training_corpus_checks_exclusion_on_devanagari_not_slp1(
    tmp_path: Path,
) -> None:
    """The exclusion hash is of the SLP1 form, but the assertion is fed Devanagari text —
    `assert_not_excluded`/`sentence_hash` convert internally, so a source given SLP1
    strings directly would (correctly) never match a Devanagari-derived hash."""
    excluded = frozenset({sentence_hash("रामः गच्छति")})
    sources = {"a": ["not the excluded sentence"]}

    manifest = build_training_corpus(sources, tmp_path / "corpus.txt", excluded)

    assert manifest["n_out"] == 1


# --------------------------------------------------------------- filter_leaked_sentences


def test_filter_leaked_sentences_drops_only_colliding_texts() -> None:
    excluded = frozenset({sentence_hash("रामः गच्छति")})
    sources = {
        "a": ["रामः गच्छति", "सीता वदति"],
        "b": ["बालकः पठति"],
    }

    filtered = train_tokenizers.filter_leaked_sentences(sources, excluded)

    assert filtered == {"a": ["सीता वदति"], "b": ["बालकः पठति"]}


def test_filter_leaked_sentences_is_a_no_op_for_a_clean_exclusion_set() -> None:
    sources = {"a": ["रामः गच्छति", "सीता वदति"]}

    filtered = train_tokenizers.filter_leaked_sentences(sources, frozenset())

    assert filtered == sources


# ------------------------------------------------------------------------------ train_bpe


def test_train_bpe_writes_tokenizer_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "arm"
    path = train_bpe(FIXTURE, vocab_size=300, out_dir=out_dir, seed=0)

    assert path == out_dir / "tokenizer.json"
    assert path.exists()


def test_train_unigram_writes_tokenizer_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "arm"
    path = train_unigram(FIXTURE, vocab_size=300, out_dir=out_dir, seed=0)

    assert path == out_dir / "tokenizer.json"
    assert path.exists()


@pytest.mark.parametrize("trainer", [train_bpe, train_unigram])
def test_trained_tokenizer_has_no_token_string_with_a_literal_space(
    tmp_path: Path, trainer: object
) -> None:
    from tokenizers import Tokenizer as RawTokenizer

    path = trainer(FIXTURE, vocab_size=300, out_dir=tmp_path / "arm", seed=0)  # type: ignore[operator]
    raw = RawTokenizer.from_file(str(path))
    vocab = raw.get_vocab()

    assert all(" " not in token for token in vocab if token != "[UNK]")


@pytest.mark.parametrize("trainer", [train_bpe, train_unigram])
def test_trained_tokenizer_metaspace_boundary_only_at_word_start(
    tmp_path: Path, trainer: object
) -> None:
    """Encoding a single word (no internal whitespace) should place the Metaspace `▁`
    boundary marker on the first token only — CLAUDE.md's Sanskrit glossary spells
    "saMskftam" in SLP1, so it doubles as this project's canonical single-word probe."""
    from tokenizers import Tokenizer as RawTokenizer

    path = trainer(FIXTURE, vocab_size=300, out_dir=tmp_path / "arm", seed=0)  # type: ignore[operator]
    raw = RawTokenizer.from_file(str(path))

    encoded = raw.encode("saMskftam", add_special_tokens=False)
    tokens = encoded.tokens

    assert tokens
    assert all(isinstance(i, int) for i in encoded.ids)
    assert not any(token.startswith("▁") for token in tokens[1:])


@pytest.mark.parametrize("trainer", [train_bpe, train_unigram])
def test_trained_tokenizer_loads_through_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainer: object
) -> None:
    trained_path = trainer(  # type: ignore[operator]
        FIXTURE, vocab_size=300, out_dir=tmp_path / "src", seed=0
    )

    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    dest = tmp_path / "T1_bpe_raw_32k" / "tokenizer.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(trained_path.read_bytes())

    tok = load_tokenizer("T1_bpe_raw_32k")
    ids = tok.encode("saMskftam")

    assert ids
    assert all(isinstance(i, int) for i in ids)


# ----------------------------------------------------------------------------- fixture


def test_slp1_corpus_mini_fixture_has_two_hundred_lines() -> None:
    assert _fixture_line_count() == 200


def test_slp1_corpus_mini_fixture_is_not_in_the_exclusion_list() -> None:
    """The fixture (three real Sāmayik-mini sentences plus 197 made-up SLP1 lines) must
    contain no real evaluation sentence: its Devanagari form must clear the same
    exclusion assertion tokenizer training uses."""
    hashes = load_exclusion_hashes(EXCLUSION_PATH)
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    devanagari = [from_slp1(line, "devanagari") for line in lines]

    assert_not_excluded(devanagari, hashes, label="slp1_corpus_mini fixture")
