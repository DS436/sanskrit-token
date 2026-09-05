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
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from sanskrit_tok.data.exclusion import (
    EXCLUSION_PATH,
    LeakageError,
    assert_not_excluded,
    load_exclusion_hashes,
    sentence_hash,
    sentence_hash_en,
)
from sanskrit_tok.encoding import from_slp1, to_slp1
from sanskrit_tok.tokenizers.corpus import build_training_corpus, filter_leaked_sentences
from sanskrit_tok.tokenizers.registry import load_tokenizer
from sanskrit_tok.tokenizers.train_bpe import train_bpe
from sanskrit_tok.tokenizers.train_unigram import train_unigram

FIXTURE = Path(__file__).parent / "fixtures" / "slp1_corpus_mini.txt"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOKENIZERS_PY = (
    REPO_ROOT / "experiments" / "02_tpp_parallel" / "train_tokenizers.py"
)
TOKENIZERS_YAML = REPO_ROOT / "experiments" / "02_tpp_parallel" / "tokenizers.yaml"


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
    # "सीता वदति" is source "a"'s second sentence and source "b"'s first; the written
    # corpus must keep the *first* occurrence (source "a", position 1) and drop "b"'s
    # repeat, not the other way around.
    sources = {
        "a": ["रामः गच्छति", "सीता वदति"],
        "b": ["सीता वदति", "बालकः पठति"],
    }
    out_path = tmp_path / "corpus.txt"

    build_training_corpus(sources, out_path, frozenset())

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == to_slp1("रामः गच्छति", "devanagari")
    assert lines.count(to_slp1("सीता वदति", "devanagari")) == 1
    assert lines == [
        to_slp1("रामः गच्छति", "devanagari"),
        to_slp1("सीता वदति", "devanagari"),
        to_slp1("बालकः पठति", "devanagari"),
    ]


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

    filtered, dropped_counts = filter_leaked_sentences(sources, excluded)

    assert filtered == {"a": ["सीता वदति"], "b": ["बालकः पठति"]}
    assert dropped_counts == {"a": 1, "b": 0}


def test_filter_leaked_sentences_is_a_no_op_for_a_clean_exclusion_set() -> None:
    sources = {"a": ["रामः गच्छति", "सीता वदति"]}

    filtered, dropped_counts = filter_leaked_sentences(sources, frozenset())

    assert filtered == sources
    assert dropped_counts == {"a": 0}


# ------------------------------------------------- English control side (E1, exp02 Task 6)


def test_build_training_corpus_can_skip_transliteration_for_english(tmp_path: Path) -> None:
    """The E1 corpus is the English side of the same splits and must reach the trainer as
    written: `to_slp1` would mangle it (CLAUDE.md §2.3 applies to Sanskrit text only)."""
    from sanskrit_tok.tokenizers.corpus import identity_transform

    sources = {"a": ["Rama goes to the forest", "Sita speaks"]}
    out_path = tmp_path / "corpus_en.txt"

    manifest = build_training_corpus(
        sources,
        out_path,
        frozenset(),
        transform=identity_transform,
        hash_fn=sentence_hash_en,
    )

    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "Rama goes to the forest",
        "Sita speaks",
    ]
    assert manifest["n_out"] == 2


def test_build_training_corpus_english_raises_leakage_error(tmp_path: Path) -> None:
    from sanskrit_tok.tokenizers.corpus import identity_transform

    excluded = frozenset({sentence_hash_en("Rama goes to the forest")})
    sources = {"a": ["Rama goes to the forest", "Sita speaks"]}

    with pytest.raises(LeakageError):
        build_training_corpus(
            sources,
            tmp_path / "corpus_en.txt",
            excluded,
            transform=identity_transform,
            hash_fn=sentence_hash_en,
        )


def test_build_training_corpus_forwards_the_hash_fn(tmp_path: Path) -> None:
    """Guards the reason `hash_fn` exists at all. The default hash transliterates
    Devanagari to SLP1 first, so for any sentence carrying Devanagari it lands somewhere
    an English exclusion list (built with `sentence_hash_en`) does not have — the corpus
    would then be built while leaking. Passing the list's own hash function catches it."""
    from sanskrit_tok.tokenizers.corpus import identity_transform

    text = "The verse रामः गच्छति opens the chapter"
    excluded = frozenset({sentence_hash_en(text)})

    missed = build_training_corpus(
        {"a": [text]}, tmp_path / "default_hash.txt", excluded, transform=identity_transform
    )
    assert missed["n_out"] == 1  # the Sanskrit hash never matches this English list

    with pytest.raises(LeakageError):
        build_training_corpus(
            {"a": [text]},
            tmp_path / "english_hash.txt",
            excluded,
            transform=identity_transform,
            hash_fn=sentence_hash_en,
        )


def test_the_exp02_script_uses_the_shared_training_machinery() -> None:
    """The corpus-currency, arm-selection and per-arm-results logic moved into
    `sanskrit_tok.tokenizers.training` so Experiment 04 could reuse it; this script must go
    on using *that* implementation rather than keeping a second copy (its own tests moved
    to `tests/test_tokenizer_training.py`)."""
    from sanskrit_tok.tokenizers import training

    assert train_tokenizers.ensure_training_corpus is training.ensure_training_corpus
    assert train_tokenizers.select_arms_to_train is training.select_arms_to_train
    assert train_tokenizers.train_arm is training.train_arm
    assert train_tokenizers.SideSpec is training.SideSpec


def test_english_train_split_loaders_are_registered() -> None:
    from sanskrit_tok.experiment import ENGLISH_SOURCE_LOADERS, SOURCE_LOADERS

    assert "samayik_train_en" in SOURCE_LOADERS
    assert "itihasa_train_en" in SOURCE_LOADERS
    assert set(ENGLISH_SOURCE_LOADERS) == {"samayik_train_en", "itihasa_train_en"}


def test_arm_side_defaults_to_sanskrit() -> None:
    assert train_tokenizers.arm_side({"name": "T1_bpe_raw_32k", "algo": "bpe"}) == "sa"


def test_arm_side_reads_the_configured_side() -> None:
    assert train_tokenizers.arm_side({"name": "E1_bpe_32k", "side": "en"}) == "en"
    assert train_tokenizers.arm_side({"name": "T1_bpe_raw_32k", "side": "sa"}) == "sa"


def test_arm_side_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="side"):
        train_tokenizers.arm_side({"name": "E1_bpe_32k", "side": "de"})


def test_tokenizers_yaml_declares_the_english_control_arms_and_sides() -> None:
    config = yaml.safe_load(TOKENIZERS_YAML.read_text(encoding="utf-8"))

    assert config["english_corpus_path"] == "data/processed/tok_train_en.txt"
    assert config["english_sources"] == ["samayik_train_en", "itihasa_train_en"]
    assert config["english_exclusion_path"] == "data/exclusion_hashes_en.txt"

    by_name = {arm["name"]: arm for arm in config["arms"]}
    for name in ("T1_bpe_raw_32k", "T1_bpe_raw_64k", "T2_unigram_raw_32k", "T2_unigram_raw_64k"):
        assert by_name[name]["side"] == "sa"
    assert by_name["E1_bpe_32k"] == {
        "name": "E1_bpe_32k",
        "algo": "bpe",
        "vocab_size": 32000,
        "side": "en",
    }
    assert by_name["E1_bpe_64k"]["vocab_size"] == 64000
    assert by_name["E1_unigram_32k"]["algo"] == "unigram"
    assert by_name["E1_unigram_64k"]["vocab_size"] == 64000
    assert all(by_name[name]["side"] == "en" for name in by_name if name.startswith("E1_"))


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
    """The Metaspace `▁` boundary marker must open every word and never appear inside
    one — CLAUDE.md's Sanskrit glossary spells "saMskftam" in SLP1, so it doubles as
    this project's canonical single-word probe. A tokenizer trained without a Metaspace
    pre-tokenizer would fail both halves of this: a single word would start with an
    ordinary character-or-merge token instead of `▁`, and a two-word input would produce
    at most one `▁`-prefixed token (or none) instead of one per word."""
    from tokenizers import Tokenizer as RawTokenizer

    path = trainer(FIXTURE, vocab_size=300, out_dir=tmp_path / "arm", seed=0)  # type: ignore[operator]
    raw = RawTokenizer.from_file(str(path))

    single = raw.encode("saMskftam", add_special_tokens=False)
    assert single.tokens
    assert all(isinstance(i, int) for i in single.ids)
    assert single.tokens[0].startswith("▁")
    assert not any(token.startswith("▁") for token in single.tokens[1:])

    two_words = raw.encode("saMskftam gacCati", add_special_tokens=False)
    assert two_words.tokens
    assert two_words.tokens[0].startswith("▁")
    boundary_tokens = [token for token in two_words.tokens if token.startswith("▁")]
    assert len(boundary_tokens) >= 2  # one boundary marker per word, at minimum


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


# ------------------------------------------------------- the sandhi-split side (exp03)


def _split_config(tmp_path: Path) -> dict[str, object]:
    """A minimal `tokenizers.yaml`-shaped mapping for the `sa_split` side."""
    return {
        "corpus_path": "data/processed/tok_train_slp1.txt",
        "split_corpus_path": "data/processed/tok_train_slp1_split.txt",
        "split_cache_path": "data/processed/split/cache.jsonl",
        "split_manifest_path": "data/processed/split/manifest.json",
        "split_reconcile_threshold": 0.6,
        "sources": ["samayik_train", "itihasa_train"],
        "exclusion_path": "data/exclusion_hashes.txt",
    }


def test_arm_side_accepts_the_sandhi_split_side() -> None:
    assert train_tokenizers.arm_side({"name": "T4_bpe_split_32k", "side": "sa_split"}) == "sa_split"


def test_side_spec_for_the_split_side_points_at_the_split_corpus(tmp_path: Path) -> None:
    spec = train_tokenizers.side_spec("sa_split", _split_config(tmp_path), tmp_path)

    assert spec.side == "sa_split"
    assert spec.corpus_path == tmp_path / "data" / "processed" / "tok_train_slp1_split.txt"
    assert spec.manifest_path == tmp_path / "data" / "processed" / "manifest_split.json"
    assert spec.exclusion_path == tmp_path / "data" / "exclusion_hashes.txt"
    assert spec.source_names == ["samayik_train", "itihasa_train"]
    assert spec.hash_fn is sentence_hash


def test_split_side_transform_reconciles_the_cached_model_output(tmp_path: Path) -> None:
    """The transform is cache-only: it never constructs a splitter, never imports torch,
    and returns the *reconciled* text, not the raw model output."""
    from sanskrit_tok.sandhi.cache import SplitCache

    cache_path = tmp_path / "data" / "processed" / "split" / "cache.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache = SplitCache(cache_path)
    cache.put("तदपि ।", "tad api")  # the model drops the danda
    cache.close()

    spec = train_tokenizers.side_spec("sa_split", _split_config(tmp_path), tmp_path)

    assert spec.transform("तदपि ।") == "tad api ."


def test_split_side_transform_raises_naming_the_sentence_when_the_cache_misses(
    tmp_path: Path,
) -> None:
    (tmp_path / "data" / "processed" / "split").mkdir(parents=True)
    spec = train_tokenizers.side_spec("sa_split", _split_config(tmp_path), tmp_path)

    with pytest.raises(train_tokenizers.MissingSplitError) as excinfo:
        spec.transform("तदपि ।")
    assert "split_corpora.py" in str(excinfo.value)


def test_split_side_precheck_names_the_number_of_uncached_sentences(tmp_path: Path) -> None:
    from sanskrit_tok.sandhi.cache import SplitCache

    cache_path = tmp_path / "data" / "processed" / "split" / "cache.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache = SplitCache(cache_path)
    cache.put("तदपि", "tad api")
    cache.close()

    spec = train_tokenizers.side_spec("sa_split", _split_config(tmp_path), tmp_path)
    assert spec.precheck is not None

    spec.precheck({"a": ["तदपि"]})  # fully cached: no error

    with pytest.raises(train_tokenizers.MissingSplitError) as excinfo:
        spec.precheck({"a": ["तदपि", "रामः गच्छति", "सीता वदति"]})
    message = str(excinfo.value)
    assert "2" in message and "3" in message
    assert str(cache_path) in message


def test_sanskrit_and_english_sides_have_no_precheck(tmp_path: Path) -> None:
    config = dict(_split_config(tmp_path))
    config.update(
        {
            "english_corpus_path": "data/processed/tok_train_en.txt",
            "english_sources": ["samayik_train_en"],
            "english_exclusion_path": "data/exclusion_hashes_en.txt",
        }
    )

    assert train_tokenizers.side_spec("sa", config, tmp_path).precheck is None
    assert train_tokenizers.side_spec("en", config, tmp_path).precheck is None


def test_tokenizers_yaml_declares_the_four_split_arms() -> None:
    config = yaml.safe_load(TOKENIZERS_YAML.read_text(encoding="utf-8"))

    assert config["split_corpus_path"] == "data/processed/tok_train_slp1_split.txt"
    assert config["split_cache_path"] == "data/processed/split/cache.jsonl"

    by_name = {arm["name"]: arm for arm in config["arms"]}
    assert by_name["T4_bpe_split_32k"] == {
        "name": "T4_bpe_split_32k",
        "algo": "bpe",
        "vocab_size": 32000,
        "side": "sa_split",
    }
    assert by_name["T4_bpe_split_64k"]["vocab_size"] == 64000
    assert by_name["T4_unigram_split_32k"]["algo"] == "unigram"
    assert by_name["T4_unigram_split_64k"]["vocab_size"] == 64000
    assert all(by_name[name]["side"] == "sa_split" for name in by_name if name.startswith("T4_"))
    assert not [name for name in by_name if name.endswith("_sub")]


# --------------------------------------------- shared training-sentence selection (fix 1)


#: Two Devanagari spellings that transliterate to the same SLP1 string: `॥` (U+0965, the
#: double danda) and `।।` (two U+0964 single dandas) both become `..`. Itihāsa's train
#: split really does contain such a pair, and it is what broke the first cut of the split
#: pipeline: `split_corpora.py` deduped on SLP1 and so split only one of them, while the
#: cache is keyed on the *Devanagari*, so the other was missing when the corpus was built.
SAME_SLP1_PAIR = ("जयमुदीरयेत्॥", "जयमुदीरयेत्।।")


def test_the_same_slp1_pair_really_does_differ_in_devanagari_only() -> None:
    first, second = SAME_SLP1_PAIR
    assert first != second
    assert to_slp1(first, "devanagari") == to_slp1(second, "devanagari")


def test_select_training_sentences_keeps_both_devanagari_spellings() -> None:
    """The selection deduplicates on the *original* text, so every sentence whose split
    the corpus builder will later ask for is in the set that gets split."""
    from sanskrit_tok.tokenizers.corpus import select_training_sentences

    first, second = SAME_SLP1_PAIR
    selected = select_training_sentences({"a": [first, second]}, frozenset())

    assert selected == [first, second]


def test_select_training_sentences_drops_exact_repeats_and_leaks_and_blanks() -> None:
    from sanskrit_tok.tokenizers.corpus import select_training_sentences

    sources = {
        "a": ["रामः गच्छति", "सीता वदति", "   ", "रामः गच्छति"],
        "b": ["रामः गच्छति", "बालकः पठति"],  # duplicate across sources
    }
    exclusion = frozenset({sentence_hash("सीता वदति")})

    selected = select_training_sentences(sources, exclusion)

    assert selected == ["रामः गच्छति", "बालकः पठति"]


def test_the_split_side_precheck_accepts_both_devanagari_spellings(tmp_path: Path) -> None:
    """The regression the reviewer found, pinned end to end: both spellings are cached
    (because the selection both sides share put both of them in), so the precheck passes."""
    from sanskrit_tok.sandhi.cache import SplitCache

    cache_path = tmp_path / "data" / "processed" / "split" / "cache.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache = SplitCache(cache_path)
    for text in SAME_SLP1_PAIR:
        cache.put(text, "jayam udIrayet")
    cache.close()

    spec = train_tokenizers.side_spec("sa_split", _split_config(tmp_path), tmp_path)
    assert spec.precheck is not None

    spec.precheck({"a": list(SAME_SLP1_PAIR)})


def test_the_split_side_precheck_uses_the_same_selection_as_the_splitter(
    tmp_path: Path,
) -> None:
    """A cache holding only the SLP1-deduped half of the pair must now be reported as
    short — the failure mode that used to surface only after a five-hour split run."""
    from sanskrit_tok.sandhi.cache import SplitCache

    cache_path = tmp_path / "data" / "processed" / "split" / "cache.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache = SplitCache(cache_path)
    cache.put(SAME_SLP1_PAIR[0], "jayam udIrayet")
    cache.close()

    spec = train_tokenizers.side_spec("sa_split", _split_config(tmp_path), tmp_path)
    assert spec.precheck is not None

    with pytest.raises(train_tokenizers.MissingSplitError, match="1 of 2"):
        spec.precheck({"a": list(SAME_SLP1_PAIR)})


# ------------------------------------------------ single-sourced reconcile threshold


def _write_split_manifest(tmp_path: Path, threshold: float) -> Path:
    path = tmp_path / "data" / "processed" / "split" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reconcile_threshold": threshold}), encoding="utf-8")
    return path


def test_the_split_threshold_comes_from_the_split_manifest(tmp_path: Path) -> None:
    """The manifest is the record of what the corpus was actually split and reconciled
    with; `tokenizers.yaml` must not be able to silently disagree with it."""
    _write_split_manifest(tmp_path, 0.6)
    config = dict(_split_config(tmp_path))
    del config["split_reconcile_threshold"]

    spec = train_tokenizers.side_spec("sa_split", config, tmp_path)

    assert spec.transform.threshold == 0.6


def test_a_threshold_disagreement_between_config_and_manifest_is_an_error(
    tmp_path: Path,
) -> None:
    _write_split_manifest(tmp_path, 0.6)
    config = dict(_split_config(tmp_path))
    config["split_reconcile_threshold"] = 0.75

    with pytest.raises(ValueError, match="0.75"):
        train_tokenizers.side_spec("sa_split", config, tmp_path)


def test_the_config_threshold_is_used_when_no_manifest_exists_yet(tmp_path: Path) -> None:
    config = dict(_split_config(tmp_path))
    config["split_reconcile_threshold"] = 0.75

    spec = train_tokenizers.side_spec("sa_split", config, tmp_path)

    assert spec.transform.threshold == 0.75
