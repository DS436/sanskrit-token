"""Tests for `experiments/03_sandhi_split/split_corpora.py` (exp03 Task 3).

The script itself is a multi-hour background job against a 2.3 GB model; everything
testable about it is the plumbing around that call, and all of it runs offline here
against a fake splitter: config parsing, the sentence set it feeds the model (the raw
training corpus's *own* set, so the split corpus is line-for-line the same sentences),
deterministic subset selection, the per-corpus jsonl carrying both the raw model output
and the reconciled text, and the manifest's keys.

`split_corpora.py` is a script under `experiments/`, whose directory name starts with a
digit and so cannot be imported as a module; it is loaded by path, the same way
`tests/test_train_tokenizers.py` loads `train_tokenizers.py`.
"""

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from sanskrit_tok.encoding import to_slp1
from sanskrit_tok.tokenizers.corpus import build_training_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CORPORA_PY = REPO_ROOT / "experiments" / "03_sandhi_split" / "split_corpora.py"
SPLIT_YAML = REPO_ROOT / "experiments" / "03_sandhi_split" / "split.yaml"


def _load_split_corpora_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp03_split_corpora", SPLIT_CORPORA_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


split_corpora = _load_split_corpora_module()


class FakeSplitter:
    """Stands in for `SandhiSplitter`: same three members `split_corpus` touches.

    `split` upper-cases nothing and knows no Sanskrit — it looks each Devanagari sentence
    up in a dict of canned SLP1 outputs, so a test can pin exactly what the "model" said
    and therefore exactly what reconciliation had to repair.
    """

    def __init__(self, outputs: dict[str, str], batch_size: int = 2) -> None:
        self._outputs = outputs
        self.batch_size = batch_size
        self.calls: list[list[str]] = []
        self.stats: dict[str, int | float] = {
            "n_calls": 0,
            "n_cache_hits": 0,
            "n_model": 0,
            "n_chunked": 0,
            "seconds_model": 0.0,
        }

    def split(self, texts: Sequence[str]) -> list[str]:
        self.calls.append(list(texts))
        self.stats["n_calls"] += len(texts)
        self.stats["n_model"] += len(texts)
        return [self._outputs.get(text, "") for text in texts]


# ------------------------------------------------------------------------ split.yaml


def test_split_yaml_parses_into_a_config_with_resolved_paths(tmp_path: Path) -> None:
    config = split_corpora.SplitConfig.from_mapping(
        yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8")), tmp_path
    )

    assert config.model_id == "chronbmm/sanskrit5-multitask"
    assert config.device == "auto"
    assert config.batch_size == 16
    assert config.cache_path == tmp_path / "data" / "processed" / "split" / "cache.jsonl"
    assert config.manifest_path == tmp_path / "data" / "processed" / "split" / "manifest.json"
    assert config.split_dir == tmp_path / "data" / "processed" / "split"
    assert config.exclusion_path == tmp_path / "data" / "exclusion_hashes.txt"
    assert config.train_sources == ["samayik_train", "itihasa_train"]
    assert config.subset is None
    assert config.progress_every == 500
    assert config.reconcile_threshold == 0.6


def test_split_yaml_names_the_four_exp02_evaluation_corpora() -> None:
    config = yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8"))
    exp02 = yaml.safe_load(
        (REPO_ROOT / "experiments" / "02_tpp_parallel" / "config.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert [entry["name"] for entry in config["eval_corpora"]] == [
        entry["name"] for entry in exp02["corpora"]
    ]
    assert [entry["loader"] for entry in config["eval_corpora"]] == [
        entry["loader"] for entry in exp02["corpora"]
    ]
    assert [entry["split"] for entry in config["eval_corpora"]] == [
        entry["split"] for entry in exp02["corpora"]
    ]


def test_split_config_rejects_an_unknown_training_source() -> None:
    mapping = yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8"))
    mapping["train_sources"] = ["nope"]

    with pytest.raises(KeyError, match="nope"):
        split_corpora.SplitConfig.from_mapping(mapping, REPO_ROOT)


def test_split_config_accepts_a_subset() -> None:
    mapping = yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8"))
    mapping["subset"] = {"n": 100, "seed": 0}

    config = split_corpora.SplitConfig.from_mapping(mapping, REPO_ROOT)

    assert config.subset == (100, 0)


# --------------------------------------------------------------------- subset selection


def test_subset_selection_is_deterministic_for_a_seed() -> None:
    texts = [f"s{i}" for i in range(50)]

    first = split_corpora.select_subset(texts, 10, 0)
    second = split_corpora.select_subset(texts, 10, 0)
    other_seed = split_corpora.select_subset(texts, 10, 1)

    assert first == second
    assert len(first) == 10
    assert first != other_seed
    assert set(first) <= set(texts)


def test_subset_selection_keeps_corpus_order() -> None:
    texts = [f"s{i}" for i in range(50)]

    chosen = split_corpora.select_subset(texts, 10, 0)

    assert chosen == [text for text in texts if text in set(chosen)]


def test_subset_larger_than_the_corpus_returns_everything() -> None:
    texts = ["a", "b", "c"]

    assert split_corpora.select_subset(texts, 99, 0) == texts


# ----------------------------------------------------- the training sentence selection


def test_the_script_uses_the_shared_training_sentence_selection() -> None:
    """There is exactly one definition of "the training sentences" (fix 1).

    The first cut of this script had its own copy that deduplicated on SLP1, while the
    split cache is keyed on Devanagari and the corpus builder transforms every Devanagari
    sentence it is given — so a sentence whose SLP1 form collided with an earlier one was
    never split and turned into a `MissingSplitError` hours later. Both callers now go
    through `tokenizers.corpus.select_training_sentences`.
    """
    from sanskrit_tok.tokenizers.corpus import select_training_sentences

    assert split_corpora.select_training_sentences is select_training_sentences
    assert not hasattr(split_corpora, "training_sentences")


def test_the_selection_keeps_two_devanagari_spellings_of_one_slp1_form() -> None:
    """`॥` and `।।` transliterate identically, so both must reach the splitter."""
    from sanskrit_tok.tokenizers.corpus import select_training_sentences

    pair = ["जयमुदीरयेत्॥", "जयमुदीरयेत्।।"]
    assert to_slp1(pair[0], "devanagari") == to_slp1(pair[1], "devanagari")

    assert split_corpora.select_training_sentences({"a": pair}, frozenset()) == pair
    assert select_training_sentences({"a": pair}, frozenset()) == pair


def test_the_selected_sentences_still_build_the_same_corpus(tmp_path: Path) -> None:
    """The selection is a superset of the corpus lines, never a different set: the corpus
    builder's own transformed-text dedup collapses what transliteration merges."""
    sources = {
        "a": ["रामः गच्छति", "सीता वदति", "  "],
        "b": ["रामः गच्छति", "बालकः पठति"],  # first duplicates source "a"
    }

    selected = split_corpora.select_training_sentences(sources, frozenset())
    build_training_corpus({"all": selected}, tmp_path / "corpus.txt", frozenset())
    lines = (tmp_path / "corpus.txt").read_text(encoding="utf-8").splitlines()

    assert selected == ["रामः गच्छति", "सीता वदति", "बालकः पठति"]
    assert [to_slp1(text, "devanagari").strip() for text in selected] == lines


def test_split_config_rejects_an_english_source_as_a_training_source() -> None:
    """`train_sources` resolves against the *Sanskrit* loaders only: an English source name
    here would be hours of a sandhi splitter reading English."""
    mapping = yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8"))
    mapping["train_sources"] = ["samayik_train_en"]

    with pytest.raises(KeyError, match="samayik_train_en"):
        split_corpora.SplitConfig.from_mapping(mapping, REPO_ROOT)


# ------------------------------------------------------------------------ split_corpus


def test_split_corpus_writes_both_the_model_output_and_the_reconciled_text(
    tmp_path: Path,
) -> None:
    # "तदपि" is SLP1 "tadapi"; the fake model returns the split form but drops the danda.
    texts = ["तदपि ।"]
    splitter = FakeSplitter({texts[0]: "tad api"})
    out_path = tmp_path / "mini.jsonl"

    report = split_corpora.split_corpus("mini", texts, splitter, out_path)

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert set(records[0]) == {"index", "raw_deva", "raw_slp1", "output_model", "output"}
    assert records[0]["index"] == 0
    assert records[0]["raw_deva"] == "तदपि ।"
    assert records[0]["raw_slp1"] == to_slp1("तदपि ।", "devanagari")
    assert records[0]["output_model"] == "tad api"
    assert records[0]["output"] == "tad api ."
    assert report.n_sentences == 1
    assert report.n_out == 1


def test_split_corpus_batches_at_the_splitters_batch_size(tmp_path: Path) -> None:
    texts = [f"वाक्य{i}" for i in range(5)]
    splitter = FakeSplitter({text: "vAkya" for text in texts}, batch_size=2)

    split_corpora.split_corpus("mini", texts, splitter, tmp_path / "mini.jsonl")

    assert [len(call) for call in splitter.calls] == [2, 2, 1]


def test_split_corpus_reports_pooled_character_retention(tmp_path: Path) -> None:
    """`chars_model / chars_raw` before reconciliation, `chars_out / chars_raw` after.

    Pooled over the corpus, not a mean of per-sentence ratios (docs/decisions.md,
    "CORRECTION: splitter character retention is 87.3%").
    """
    texts = ["तदपि ।"]
    splitter = FakeSplitter({texts[0]: "tad api"})

    report = split_corpora.split_corpus("mini", texts, splitter, tmp_path / "mini.jsonl")

    assert report.chars_raw == len("tadapi.")
    assert report.chars_model == len("tadapi")
    assert report.chars_out == len("tadapi.")
    assert report.char_retention_model == pytest.approx(6 / 7)
    assert report.char_retention_reconciled == pytest.approx(1.0)


def test_split_corpus_reports_the_verbatim_and_inexact_unit_counters(tmp_path: Path) -> None:
    """Retention alone can read 1.000 while the model dropped a word and rewrote another,
    so both counters are pooled per corpus (review item 3)."""
    texts = ["तदपि ।", "प्राणिन आगत्य"]
    splitter = FakeSplitter({"तदपि ।": "tad api", "प्राणिन आगत्य": "prARinaH Agatya"})

    report = split_corpora.split_corpus("mini", texts, splitter, tmp_path / "mini.jsonl")

    assert report.n_units_kept_verbatim == 1  # the danda the model dropped
    assert report.n_units_replaced_inexact == 1  # prARina -> prARinaH, a normalisation
    assert report.n_units_raw == 4
    assert report.n_units_out == 5


def test_split_corpus_reports_the_fraction_of_sentences_whose_unit_count_changed(
    tmp_path: Path,
) -> None:
    texts = ["तदपि", "रामः"]
    splitter = FakeSplitter({"तदपि": "tad api", "रामः": "rAmaH"})

    report = split_corpora.split_corpus("mini", texts, splitter, tmp_path / "mini.jsonl")

    assert report.n_units_changed == 1
    assert report.fraction_units_changed == pytest.approx(0.5)


def test_split_corpus_replaces_a_partial_file_from_an_earlier_run(tmp_path: Path) -> None:
    """A run killed mid-corpus leaves nothing that could be mistaken for a finished file:
    the jsonl is written to a temporary name and moved into place only when the corpus is
    complete. The cache (not this file) is what makes the re-run cheap."""
    out_path = tmp_path / "mini.jsonl"
    out_path.write_text('{"index": 99}\n', encoding="utf-8")
    texts = ["तदपि"]
    splitter = FakeSplitter({"तदपि": "tad api"})

    split_corpora.split_corpus("mini", texts, splitter, out_path)

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [record["index"] for record in records] == [0]
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------- manifest


def _report(name: str) -> object:
    return split_corpora.CorpusSplitReport(
        name=name,
        path="data/processed/split/x.jsonl",
        n_sentences=3,
        n_out=3,
        n_units_changed=1,
        fraction_units_changed=1 / 3,
        n_units_raw=30,
        n_units_out=32,
        n_units_kept_verbatim=2,
        n_units_replaced_inexact=3,
        chars_raw=30,
        chars_model=27,
        chars_out=31,
        char_retention_model=0.9,
        char_retention_reconciled=31 / 30,
        seconds=1.5,
    )


def test_manifest_carries_every_documented_key(tmp_path: Path) -> None:
    config = split_corpora.SplitConfig.from_mapping(
        yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8")), tmp_path
    )
    stats = {"n_calls": 6, "n_cache_hits": 2, "n_model": 4, "n_chunked": 1, "seconds_model": 3.0}

    manifest = split_corpora.build_manifest(
        config=config,
        splitter_source_id="chronbmm/sanskrit5-multitask@abc123",
        device="mps",
        stats=stats,
        reports=[_report("samayik_test"), _report("train")],
        seconds=12.5,
        root=REPO_ROOT,
    )

    for key in (
        "model_id",
        "revision",
        "splitter_source_id",
        "device",
        "batch_size",
        "subset",
        "reconcile_threshold",
        "n_sentences",
        "n_out",
        "n_cache_hits",
        "n_model",
        "n_chunked",
        "seconds",
        "corpora",
        "splitter_stats",
        "git_commit",
        "git_dirty",
        "timestamp",
    ):
        assert key in manifest, key

    assert manifest["model_id"] == "chronbmm/sanskrit5-multitask"
    assert manifest["revision"] == "abc123"
    assert manifest["device"] == "mps"
    assert manifest["subset"] is None
    assert manifest["n_cache_hits"] == 2
    assert manifest["n_chunked"] == 1
    assert manifest["n_sentences"] == {"samayik_test": 3, "train": 3}
    assert manifest["n_out"] == {"samayik_test": 3, "train": 3}
    assert manifest["seconds"] == 12.5


def test_manifest_records_retention_before_and_after_reconciliation(tmp_path: Path) -> None:
    config = split_corpora.SplitConfig.from_mapping(
        yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8")), tmp_path
    )

    manifest = split_corpora.build_manifest(
        config=config,
        splitter_source_id="m@rev",
        device="cpu",
        stats={"n_cache_hits": 0, "n_model": 3, "n_chunked": 0},
        reports=[_report("samayik_test")],
        seconds=1.0,
        root=REPO_ROOT,
    )

    corpus = manifest["corpora"]["samayik_test"]  # type: ignore[index]
    assert corpus["char_retention_model"] == pytest.approx(0.9)
    assert corpus["char_retention_reconciled"] == pytest.approx(31 / 30)
    assert corpus["fraction_units_changed"] == pytest.approx(1 / 3)
    assert corpus["n_units_raw"] == 30
    assert corpus["n_units_out"] == 32
    assert corpus["n_units_kept_verbatim"] == 2
    assert corpus["n_units_replaced_inexact"] == 3


def test_manifest_is_strict_json_serialisable(tmp_path: Path) -> None:
    config = split_corpora.SplitConfig.from_mapping(
        yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8")), tmp_path
    )

    manifest = split_corpora.build_manifest(
        config=config,
        splitter_source_id="m@rev",
        device="cpu",
        stats={"n_cache_hits": 0, "n_model": 0, "n_chunked": 0},
        reports=[_report("samayik_test")],
        seconds=1.0,
        root=REPO_ROOT,
    )

    json.dumps(manifest, allow_nan=False)
