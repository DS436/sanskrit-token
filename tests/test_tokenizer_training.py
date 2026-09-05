"""Tests for `sanskrit_tok.tokenizers.training`, the shared tokenizer-training machinery.

These were `tests/test_train_tokenizers.py`'s tests for `ensure_training_corpus` and
`select_arms_to_train` back when both lived in `experiments/02_tpp_parallel/
train_tokenizers.py`; Experiment 04 needs the same corpus-currency, arm-selection and
per-arm-results logic against a different corpus source, so the machinery moved into the
package and its tests moved with it. What stays in `tests/test_train_tokenizers.py` is what
is still specific to Experiment 02: its sides, its split transform, its YAML.

`build_streamed_corpus` is the new piece: Experiment 04 builds four corpora from a
720,510-line, 437 MB jsonl, which cannot be held in memory as `ensure_training_corpus`'s
`Mapping[str, Sequence[str]]` requires, so it streams `(check_text, write_text)` pairs
instead. Everything here is offline and tiny.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from sanskrit_tok.data.exclusion import LeakageError, sentence_hash, sentence_hash_en
from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.tokenizers.corpus import identity_transform
from sanskrit_tok.tokenizers.training import (
    ALGO_TRAINERS,
    build_streamed_corpus,
    corpus_is_current,
    ensure_training_corpus,
    select_arms_to_train,
    train_arm,
    write_corpus_manifest,
)

#: Two Devanagari spellings (U+0965 vs. two U+0964) that transliterate to one SLP1 string.
SAME_SLP1_PAIR = ("जयमुदीरयेत्॥", "जयमुदीरयेत्।।")

MARKER = "\x1f"


# ---------------------------------------------------------------- ensure_training_corpus


def test_ensure_training_corpus_records_raw_and_leaked_counts_in_the_manifest(
    tmp_path: Path,
) -> None:
    excluded = frozenset({sentence_hash("रामः गच्छति")})
    sources = {
        "a": ["रामः गच्छति", "सीता वदति"],  # first sentence is leaked, dropped before build
        "b": ["बालकः पठति"],
    }
    corpus_path = tmp_path / "corpus.txt"

    manifest = ensure_training_corpus(sources, corpus_path, excluded)

    assert manifest["n_in"] == {"a": 1, "b": 1}
    assert manifest["n_in_raw"] == {"a": 2, "b": 1}
    assert manifest["n_leaked_dropped"] == {"a": 1, "b": 0}

    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["n_in_raw"] == {"a": 2, "b": 1}
    assert written["n_leaked_dropped"] == {"a": 1, "b": 0}


def test_ensure_training_corpus_writes_the_manifest_it_is_given(tmp_path: Path) -> None:
    """The Sanskrit and English corpora live side by side in `data/processed/`, so the
    English one needs its own manifest name rather than overwriting the Sanskrit one."""
    corpus_path = tmp_path / "tok_train_en.txt"
    manifest_path = tmp_path / "manifest_en.json"

    manifest = ensure_training_corpus(
        {"a": ["Rama goes", "Sita speaks"]},
        corpus_path,
        frozenset(),
        manifest_path=manifest_path,
        transform=identity_transform,
        hash_fn=sentence_hash_en,
    )

    assert manifest_path.exists()
    assert not (tmp_path / "manifest.json").exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["n_out"] == manifest["n_out"]


def test_ensure_training_corpus_runs_the_precheck_before_building(tmp_path: Path) -> None:
    calls: list[int] = []

    def precheck(sources: object) -> None:
        calls.append(1)

    ensure_training_corpus(
        {"a": ["रामः गच्छति"]},
        tmp_path / "corpus.txt",
        frozenset(),
        manifest_path=tmp_path / "manifest.json",
        precheck=precheck,
    )

    assert calls == [1]


def test_ensure_training_corpus_deduplicates_on_the_original_before_transforming(
    tmp_path: Path,
) -> None:
    """The corpus builder is fed the *selected* sentences, so the split side's transform
    is never asked for a sentence the split run did not split. The written corpus is
    unchanged: the builder still dedups on the transformed text."""
    transformed: list[str] = []

    def transform(text: str) -> str:
        transformed.append(text)
        return to_slp1(text, "devanagari")

    first, second = SAME_SLP1_PAIR
    manifest = ensure_training_corpus(
        {"a": [first, second, first]},
        tmp_path / "corpus.txt",
        frozenset(),
        manifest_path=tmp_path / "manifest.json",
        transform=transform,
    )

    assert transformed == [first, second]
    assert manifest["n_out"] == 1
    assert (tmp_path / "corpus.txt").read_text(encoding="utf-8").splitlines() == [
        to_slp1(first, "devanagari")
    ]


# --------------------------------------------------------------------- corpus currency


def test_corpus_is_current_returns_the_manifest_when_the_sha256_matches(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tadapi\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_corpus_manifest({"n_out": 1, "sha256": _sha256(corpus)}, manifest_path)

    assert corpus_is_current(corpus, manifest_path) == {"n_out": 1, "sha256": _sha256(corpus)}


def test_corpus_is_current_rejects_a_corpus_edited_since_its_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tadapi\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_corpus_manifest({"sha256": _sha256(corpus)}, manifest_path)
    corpus.write_text("rAmaH\n", encoding="utf-8")

    assert corpus_is_current(corpus, manifest_path) is None


def test_corpus_is_current_is_none_without_a_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tadapi\n", encoding="utf-8")

    assert corpus_is_current(corpus, tmp_path / "manifest.json") is None


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ build_streamed_corpus


def _records(pairs: list[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    yield from pairs


def test_build_streamed_corpus_writes_every_line_and_counts_them(tmp_path: Path) -> None:
    out = tmp_path / "corpus.txt"

    manifest = build_streamed_corpus(
        _records([("tadapi", "tadapi"), ("rAmaH", "rAmaH")]),
        out,
        frozenset(),
        hash_fn=sentence_hash,
        label="toy",
    )

    assert out.read_text(encoding="utf-8").splitlines() == ["tadapi", "rAmaH"]
    assert manifest["n_in"] == 2
    assert manifest["n_out"] == 2
    assert manifest["n_dedup_removed"] == 0
    assert manifest["n_checked"] == 2
    assert manifest["sha256"] == _sha256(out)


def test_build_streamed_corpus_deduplicates_on_the_written_line(tmp_path: Path) -> None:
    out = tmp_path / "corpus.txt"

    manifest = build_streamed_corpus(
        _records([("a", "tadapi"), ("b", "tadapi"), ("c", "rAmaH")]),
        out,
        frozenset(),
        hash_fn=sentence_hash,
        label="toy",
    )

    assert out.read_text(encoding="utf-8").splitlines() == ["tadapi", "rAmaH"]
    assert manifest["n_dedup_removed"] == 1


def test_build_streamed_corpus_counts_boundary_markers_and_plain_duplicates(
    tmp_path: Path,
) -> None:
    """`n_markers` is what makes a marked corpus auditable; `n_out_plain` says how many
    lines would survive if the markers were stripped first, so a marked corpus that is not
    line-for-line matched with its unmarked twin is visible rather than assumed away."""
    out = tmp_path / "corpus.txt"

    manifest = build_streamed_corpus(
        _records([("a", f"tad{MARKER}api"), ("b", f"ta{MARKER}dapi"), ("c", f"rAma{MARKER}H")]),
        out,
        frozenset(),
        hash_fn=sentence_hash,
        label="toy",
        marker=MARKER,
    )

    assert manifest["n_markers"] == 3
    assert manifest["n_out"] == 3
    # `tad|api` and `ta|dapi` are two marked lines and one plain line.
    assert manifest["n_out_plain"] == 2


def test_build_streamed_corpus_raises_leakage_error_naming_the_line_numbers(
    tmp_path: Path,
) -> None:
    excluded = frozenset({sentence_hash("रामः गच्छति")})

    with pytest.raises(LeakageError, match="toy"):
        build_streamed_corpus(
            _records([("सीता वदति", "sItA vadati"), ("रामः गच्छति", "rAmaH gacCati")]),
            tmp_path / "corpus.txt",
            excluded,
            hash_fn=sentence_hash,
            label="toy",
        )


def test_build_streamed_corpus_drops_blank_written_lines(tmp_path: Path) -> None:
    out = tmp_path / "corpus.txt"

    manifest = build_streamed_corpus(
        _records([("a", "tadapi"), ("b", "   "), ("c", "")]),
        out,
        frozenset(),
        hash_fn=sentence_hash,
        label="toy",
    )

    assert out.read_text(encoding="utf-8").splitlines() == ["tadapi"]
    assert manifest["n_out"] == 1
    assert manifest["n_blank"] == 2


# ------------------------------------------------------------- arm selection and training


def test_select_arms_to_train_skips_an_already_trained_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    trained = tmp_path / "T1_bpe_raw_32k" / "tokenizer.json"
    trained.parent.mkdir(parents=True)
    trained.write_text("{}", encoding="utf-8")
    arms = [{"name": "T1_bpe_raw_32k"}, {"name": "E1_bpe_32k"}]

    selected = select_arms_to_train(arms, retrain=False)

    assert [arm["name"] for arm in selected] == ["E1_bpe_32k"]


def test_select_arms_to_train_retrains_everything_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path))
    trained = tmp_path / "T1_bpe_raw_32k" / "tokenizer.json"
    trained.parent.mkdir(parents=True)
    trained.write_text("{}", encoding="utf-8")
    arms = [{"name": "T1_bpe_raw_32k"}, {"name": "E1_bpe_32k"}]

    selected = select_arms_to_train(arms, retrain=True)

    assert [arm["name"] for arm in selected] == ["T1_bpe_raw_32k", "E1_bpe_32k"]


def test_algo_trainers_cover_the_three_algorithms() -> None:
    assert set(ALGO_TRAINERS) == {"bpe", "unigram", "morph_bpe"}


def test_train_arm_writes_the_tokenizer_and_reports_its_actual_vocab_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path / "tok"))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["tadapi rAmaH"] * 40) + "\n", encoding="utf-8")

    payload = train_arm({"name": "T1_bpe_raw_32k", "algo": "bpe", "vocab_size": 40}, corpus, 0)

    assert payload["arm"] == "T1_bpe_raw_32k"
    assert payload["requested_vocab_size"] == 40
    assert payload["vocab_size"] <= 40
    assert (tmp_path / "tok" / "T1_bpe_raw_32k" / "tokenizer.json").exists()
    assert payload["train_seconds"] >= 0.0


def test_train_arm_puts_extra_keys_straight_after_the_algo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`side` (exp02) and `corpus` (exp04) are per-experiment labels the shared trainer
    knows nothing about; they are passed in and land in a stable place in the payload."""
    monkeypatch.setenv("SANSKRIT_TOK_TOKENIZER_DIR", str(tmp_path / "tok"))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["tadapi rAmaH"] * 40) + "\n", encoding="utf-8")

    payload = train_arm(
        {"name": "T1_bpe_raw_32k", "algo": "bpe", "vocab_size": 40},
        corpus,
        0,
        extra={"side": "sa"},
    )

    assert list(payload)[:3] == ["arm", "algo", "side"]


def test_train_arm_rejects_an_unknown_algo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="algo"):
        train_arm({"name": "X", "algo": "nope", "vocab_size": 10}, tmp_path / "c.txt", 0)
