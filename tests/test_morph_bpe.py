"""Tests for `sanskrit_tok.tokenizers.morph_bpe` and the Experiment 04 arm config (Task 3).

The hard MorphBPE constraint is implemented as boundary-marker pre-tokenisation
(docs/decisions.md, 2026-09-05, "MorphBPE-hard implemented as boundary-marker
pre-tokenisation; inference unchanged"): the training text carries U+001F at every gold
boundary and the pre-tokenizer is `Sequence([Metaspace(), Split(marker, "removed")])`, so
the trainer never *counts* a pair that straddles a boundary and the marker itself never
reaches the model.

Everything here is offline and tiny: the toy corpus is the two words from the plan's Step 1
(`tad\\x1fapi`, `rAma\\x1fH`) and vocabularies are 40, so a full train is milliseconds.
"""

import json
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer as RawTokenizer

from sanskrit_tok.data.boundaries import BOUNDARY_MARKER as GOLD_MARKER
from sanskrit_tok.tokenizers.morph_bpe import (
    BOUNDARY_MARKER,
    assert_no_cross_boundary_merges,
    train_morph_bpe,
)
from sanskrit_tok.tokenizers.train_bpe import train_bpe

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP04_TOKENIZERS_YAML = REPO_ROOT / "experiments" / "04_morph_constrained" / "tokenizers.yaml"

#: The plan's Step 1 toy corpus: the two words `tad|api` and `rAma|H`, each with one gold
#: boundary, plus an unmarked third word. Repeated so the trainer has something to count;
#: in the unmarked control the pair straddling each boundary is the most frequent one in
#: the corpus and is merged into the whole word immediately.
#:
#: The third word is not decoration. HF `tokenizers` feeds a training *file* to the trainer
#: with each line's trailing newline attached, so the last word of a line only ever appears
#: as `word\n` and its merges are learned for that string rather than for the bare word.
#: (True of every arm this repository has trained — `T1_bpe_raw_32k` has 1,501 vocabulary
#: entries containing a newline — and harmless, since it is identical across arms.) Putting
#: `iti` last keeps the two words under test away from that edge.
TOY_LINES = ["tad\x1fapi rAma\x1fH iti"] * 100


def _write_toy_corpus(path: Path, *, marked: bool) -> Path:
    lines = TOY_LINES if marked else [line.replace(BOUNDARY_MARKER, "") for line in TOY_LINES]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------ the marker itself


def test_boundary_marker_is_the_one_the_gold_boundaries_use() -> None:
    """One definition, imported — not a second copy that could drift from the ingestion."""
    assert BOUNDARY_MARKER == GOLD_MARKER == "\x1f"


# --------------------------------------------------------------- the marker never reaches the vocab


def test_marker_pretokenizer_never_puts_the_marker_in_the_vocabulary(tmp_path: Path) -> None:
    corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)

    path = train_morph_bpe(corpus, 40, tmp_path / "arm")

    vocab = RawTokenizer.from_file(str(path)).get_vocab()
    assert not [token for token in vocab if BOUNDARY_MARKER in token]


def test_saved_tokenizer_json_does_not_contain_the_marker_anywhere_in_its_vocab(
    tmp_path: Path,
) -> None:
    corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)

    path = train_morph_bpe(corpus, 40, tmp_path / "arm")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert not [token for token in saved["model"]["vocab"] if BOUNDARY_MARKER in token]


def test_marker_trained_tokenizer_encodes_marker_free_text_as_a_normal_bpe(
    tmp_path: Path,
) -> None:
    """At inference the text has no markers, so the `Split` is a no-op and the arm is an
    ordinary Metaspace BPE with a constrained merge table."""
    corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)
    path = train_morph_bpe(corpus, 40, tmp_path / "arm")

    encoding = RawTokenizer.from_file(str(path)).encode("tadapi")

    assert [("tadapi")[start:end] for start, end in encoding.offsets] == ["tad", "api"]


# ---------------------------------------------------------------- the constraint itself


def test_toy_marked_corpus_has_no_cross_boundary_merges(tmp_path: Path) -> None:
    corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)
    path = train_morph_bpe(corpus, 40, tmp_path / "arm")

    report = assert_no_cross_boundary_merges(path, corpus, sample=100)

    assert report["n_violations"] == 0
    assert report["n_boundaries"] == 200  # two per line, 100 lines
    assert report["n_sentences"] == 100


def test_the_unmarked_control_does_merge_across_the_boundary(tmp_path: Path) -> None:
    """The control is what makes the test above mean something: same corpus, same vocab
    size, no marker — and the boundary is exactly what BPE merges first."""
    control_corpus = _write_toy_corpus(tmp_path / "plain.txt", marked=False)
    marked_corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)
    path = train_bpe(control_corpus, 40, tmp_path / "control")

    report = assert_no_cross_boundary_merges(path, marked_corpus, sample=100)

    assert report["n_violations"] > 0


def test_train_bpe_without_a_boundary_marker_is_unchanged(tmp_path: Path) -> None:
    """The new keyword is opt-in: omitted, `train_bpe` is the plain Metaspace BPE the T1
    and T4 arms were trained with, byte for byte."""
    corpus = _write_toy_corpus(tmp_path / "plain.txt", marked=False)

    default = train_bpe(corpus, 40, tmp_path / "default")
    explicit_none = train_bpe(corpus, 40, tmp_path / "none", boundary_marker=None)

    assert default.read_bytes() == explicit_none.read_bytes()


# --------------------------------------------------- assert_no_cross_boundary_merges contract


def test_cross_boundary_report_counts_tokens_and_reports_a_rate(tmp_path: Path) -> None:
    corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)
    path = train_morph_bpe(corpus, 40, tmp_path / "arm")

    report = assert_no_cross_boundary_merges(path, corpus, sample=100)

    assert report["n_tokens"] > 0
    assert report["violation_rate"] == 0.0
    assert report["boundary_marker"] == BOUNDARY_MARKER
    assert report["sample"] == 100


def test_cross_boundary_report_strides_the_corpus_when_it_is_larger_than_the_sample(
    tmp_path: Path,
) -> None:
    corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)
    path = train_morph_bpe(corpus, 40, tmp_path / "arm")

    report = assert_no_cross_boundary_merges(path, corpus, sample=10)

    assert report["n_sentences"] == 10
    assert report["stride"] == 10
    assert report["n_lines"] == 100


def test_cross_boundary_report_names_its_violations(tmp_path: Path) -> None:
    control_corpus = _write_toy_corpus(tmp_path / "plain.txt", marked=False)
    marked_corpus = _write_toy_corpus(tmp_path / "marked.txt", marked=True)
    path = train_bpe(control_corpus, 40, tmp_path / "control")

    report = assert_no_cross_boundary_merges(path, marked_corpus, sample=100)

    examples = report["examples"]
    assert isinstance(examples, list) and examples
    assert {"token", "text", "boundaries"} <= set(examples[0])


# --------------------------------------------------------------------- the Experiment 04 config

#: The twelve arms Experiment 04 trains, and the corpus each is trained on (the plan's
#: Task 3 interface list, CLAUDE.md §6).
EXPECTED_ARMS = {
    "T1_bpe_raw_32k_dcs": ("bpe", "dcs_raw", 32000),
    "T1_bpe_raw_64k_dcs": ("bpe", "dcs_raw", 64000),
    "T2_unigram_raw_32k_dcs": ("unigram", "dcs_raw", 32000),
    "T2_unigram_raw_64k_dcs": ("unigram", "dcs_raw", 64000),
    "T4_bpe_split_32k_oracle_dcs": ("bpe", "dcs_oracle_split", 32000),
    "T4_bpe_split_64k_oracle_dcs": ("bpe", "dcs_oracle_split", 64000),
    "T4_unigram_split_32k_oracle_dcs": ("unigram", "dcs_oracle_split", 32000),
    "T4_unigram_split_64k_oracle_dcs": ("unigram", "dcs_oracle_split", 64000),
    "T5_morphbpe_raw_32k_dcs": ("morph_bpe", "dcs_raw_marked", 32000),
    "T5_morphbpe_raw_64k_dcs": ("morph_bpe", "dcs_raw_marked", 64000),
    "T6_morphbpe_split_32k_dcs": ("morph_bpe", "dcs_split_marked", 32000),
    "T6_morphbpe_split_64k_dcs": ("morph_bpe", "dcs_split_marked", 64000),
}

#: Which DCS `GoldSentence` field each configured corpus is written from.
EXPECTED_CORPUS_FIELDS = {
    "dcs_raw": "text_slp1",
    "dcs_oracle_split": "oracle_split_slp1",
    "dcs_raw_marked": "t5_marked",
    "dcs_split_marked": "t6_marked",
}


@pytest.fixture(scope="module")
def exp04_config() -> dict[str, object]:
    return dict(yaml.safe_load(EXP04_TOKENIZERS_YAML.read_text(encoding="utf-8")))


def test_exp04_config_declares_the_four_dcs_corpora(exp04_config: dict[str, object]) -> None:
    corpora = exp04_config["corpora"]
    assert isinstance(corpora, dict)
    assert set(corpora) == set(EXPECTED_CORPUS_FIELDS)
    for name, field in EXPECTED_CORPUS_FIELDS.items():
        assert corpora[name]["field"] == field
        assert corpora[name]["corpus_path"] == f"data/processed/tok_train_{name}.txt"


def test_exp04_config_declares_the_twelve_arms(exp04_config: dict[str, object]) -> None:
    arms = exp04_config["arms"]
    assert isinstance(arms, list)
    declared = {
        str(arm["name"]): (str(arm["algo"]), str(arm["corpus"]), int(arm["vocab_size"]))
        for arm in arms
    }
    assert declared == EXPECTED_ARMS


def test_exp04_config_gives_every_bpe_arm_a_marked_corpus_to_check_against(
    exp04_config: dict[str, object],
) -> None:
    """The constrained arms are checked against their own training text; the unconstrained
    BPE arms are checked against the matching *marked* corpus as the control."""
    arms = {str(arm["name"]): arm for arm in exp04_config["arms"]}
    assert arms["T5_morphbpe_raw_32k_dcs"]["check_corpus"] == "dcs_raw_marked"
    assert arms["T6_morphbpe_split_32k_dcs"]["check_corpus"] == "dcs_split_marked"
    assert arms["T1_bpe_raw_32k_dcs"]["check_corpus"] == "dcs_raw_marked"
    assert arms["T4_bpe_split_64k_oracle_dcs"]["check_corpus"] == "dcs_split_marked"
    assert "check_corpus" not in arms["T2_unigram_raw_32k_dcs"]


def test_exp04_config_points_at_the_dcs_training_split_and_the_exclusion_list(
    exp04_config: dict[str, object],
) -> None:
    assert exp04_config["dcs_train_jsonl"] == "data/processed/dcs/train.jsonl"
    assert exp04_config["exclusion_path"] == "data/exclusion_hashes.txt"
    assert exp04_config["output_dir"] == "outputs/tokenizers"
