"""Tests for the M1 corpus loaders and the Experiment 05 corpus builder.

Everything here is offline. The two Hugging Face parquet sources are stood up as tiny
in-test parquet files written with `pyarrow` into `tmp_path` (the brief's
`sangraha_mini.parquet`, built rather than committed, so no binary lands in git): both
loaders read a file that is already in their cache directory without touching the
network, which is the same code path a warm run takes.

What is pinned is the *derivation*, not any real number: how a document becomes lines
(danda splitting, the two-word floor, the Devanagari-letter test, wiki boilerplate), and
how `build_corpus.py` turns five sources into three corpora — which lines the two leakage
layers drop and under which evaluation source they are counted, which duplicates go, that
Track 1's raw and split files stay line-for-line the same sentences, and that a
parallel held-out corpus whose split jsonl has drifted out of alignment with its loader is
an error rather than a silently misaligned evaluation set.

`build_corpus.py` is not importable as a package module (`experiments/` holds scripts, not
a package), so it is loaded by path, as `tests/test_exp01.py` onwards do.
"""

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from sanskrit_tok.data import sangraha, wikipedia_sa
from sanskrit_tok.data.exclusion import (
    SHINGLE_K,
    build_exclusion_list,
    build_shingle_index,
    has_shingle_overlap,
    sentence_hash_slp1,
)
from sanskrit_tok.data.parallel import ParallelCorpus

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_PY = REPO_ROOT / "experiments" / "05_lm_training" / "build_corpus.py"
CONFIG_YAML = REPO_ROOT / "experiments" / "05_lm_training" / "corpus.yaml"


def _load_build_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp05_build_corpus", BUILD_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_corpus = _load_build_module()


# ------------------------------------------------------------------- line splitting


def test_documents_to_lines_splits_on_danda_and_keeps_it() -> None:
    """A danda ends a line and stays attached to it; a trailing fragment obeys the rules."""
    document = "रामः वनं गच्छति। सीता गृहे तिष्ठति॥ अन्यत्"
    assert sangraha.documents_to_lines(document) == [
        "रामः वनं गच्छति।",
        "सीता गृहे तिष्ठति॥",
    ]


def test_documents_to_lines_splits_on_newlines_and_collapses_whitespace() -> None:
    """Newlines separate lines; runs of whitespace inside one collapse to a single space.

    Collapsing is what makes "one sentence per line" true by construction: the corpus
    files these lines are written to are newline-delimited, so a line may not carry one.
    """
    document = "रामः  वनं\tगच्छति\n\nसीता गृहे तिष्ठति "
    assert sangraha.documents_to_lines(document) == [
        "रामः वनं गच्छति",
        "सीता गृहे तिष्ठति",
    ]


@pytest.mark.parametrize(
    "document",
    [
        "रामः",  # one word
        "hello world there",  # no Devanagari letter
        "१२३ ४५६",  # Devanagari digits are not letters
        "। ॥",  # punctuation only
        "",
    ],
)
def test_documents_to_lines_drops_unusable_lines(document: str) -> None:
    assert sangraha.documents_to_lines(document) == []


# ------------------------------------------------------------------------- Sangraha


def _write_parquet(path: Path, columns: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def test_verified_sanskrit_files_takes_only_verified_sanskrit() -> None:
    """`verified/san/` only: never `synthetic` (machine-translated) and never another language."""
    files = [
        "README.md",
        "synthetic/san_Deva/wiki_0.parquet",
        "unverified/san/data-0.parquet",
        "verified/hin/data-0.parquet",
        "verified/san/data-1.parquet",
        "verified/san/data-0.parquet",
    ]
    assert sangraha.verified_sanskrit_files(files) == [
        "verified/san/data-0.parquet",
        "verified/san/data-1.parquet",
    ]


def test_iter_sangraha_sanskrit_reads_a_cached_parquet(tmp_path: Path) -> None:
    """A file already in the cache directory is read as-is, with no network access."""
    name = "verified/san/data-0.parquet"
    _write_parquet(
        tmp_path / name,
        {
            "doc_id": ["a", "b"],
            "text": ["रामः वनं गच्छति।", "सीता गृहे तिष्ठति॥"],
            "type": ["pdf", "pdf"],
        },
    )
    documents = list(sangraha.iter_sangraha_sanskrit(tmp_path, files=[name]))
    assert documents == ["रामः वनं गच्छति।", "सीता गृहे तिष्ठति॥"]


def test_iter_sangraha_lines_flattens_documents(tmp_path: Path) -> None:
    name = "verified/san/data-0.parquet"
    _write_parquet(
        tmp_path / name,
        {
            "doc_id": ["a"],
            "text": ["रामः वनं गच्छति।\nसीता गृहे तिष्ठति॥ एकम्"],
            "type": ["pdf"],
        },
    )
    assert list(sangraha.iter_sangraha_lines(tmp_path, files=[name])) == [
        "रामः वनं गच्छति।",
        "सीता गृहे तिष्ठति॥",
    ]


# ------------------------------------------------------------------------ Wikipedia


def test_wikipedia_strip_boilerplate() -> None:
    """Headings, a line repeating the article title and very short lines all go."""
    document = "\n".join(
        [
            "== इतिहासः ==",
            "श्रीलङ्का",
            "अस्य द्वीपराष्ट्रस्य राजधानी कोलम्बो अस्ति।",
            "लघु पङ्क्तिः",
        ]
    )
    kept = wikipedia_sa.strip_boilerplate(document, title="श्रीलङ्का")
    assert kept == "अस्य द्वीपराष्ट्रस्य राजधानी कोलम्बो अस्ति।"


def test_wikipedia_documents_to_lines_applies_both_filters() -> None:
    document = "== इतिहासः ==\nरामायणकाले स्वर्णलङ्का इति उल्लिखितम्। रावणस्य राजधानी आसीत्।"
    assert wikipedia_sa.documents_to_lines(document, title="श्रीलङ्का") == [
        "रामायणकाले स्वर्णलङ्का इति उल्लिखितम्।",
        "रावणस्य राजधानी आसीत्।",
    ]


def test_iter_wikipedia_lines_reads_a_cached_parquet(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / wikipedia_sa.WIKIPEDIA_CONFIG / "train-00000-of-00001.parquet",
        {
            "id": ["1"],
            "url": ["https://sa.wikipedia.org/wiki/x"],
            "title": ["श्रीलङ्का"],
            "text": [
                "श्रीलङ्का\nअस्य द्वीपराष्ट्रस्य राजधानी कोलम्बो अस्ति। लङ्कायाः उल्लेखः भवति।"
            ],
        },
    )
    assert list(wikipedia_sa.iter_wikipedia_lines(tmp_path)) == [
        "अस्य द्वीपराष्ट्रस्य राजधानी कोलम्बो अस्ति।",
        "लङ्कायाः उल्लेखः भवति।",
    ]


# --------------------------------------------------------------- the shingle helper


def test_matching_eval_sources_agrees_with_has_shingle_overlap() -> None:
    """The per-source helper is `has_shingle_overlap`, asked of each index at once."""
    long_eval = "aBiDarmakoSam pravakzyAmi SAstram"
    short_eval = "rAmaH vanam"
    indices = {
        "long": build_shingle_index([long_eval]),
        "short": build_shingle_index([short_eval]),
    }
    cases = [
        "atra aBiDarmakoSam pravakzyAmi SAstram iti",  # long window shared
        "rAmaH vanam",  # short, exact letters
        "sItA gfhe tizWati",  # neither
    ]
    for text in cases:
        expected = sorted(
            name
            for name, index in indices.items()
            if has_shingle_overlap(text, index, SHINGLE_K)
        )
        assert sorted(build_corpus.matching_eval_sources(text, indices, SHINGLE_K)) == expected
    assert build_corpus.matching_eval_sources(cases[0], indices, SHINGLE_K) == ["long"]
    assert build_corpus.matching_eval_sources(cases[1], indices, SHINGLE_K) == ["short"]
    assert build_corpus.matching_eval_sources(cases[2], indices, SHINGLE_K) == []


# ---------------------------------------------------------------- the corpus builder

#: DCS training records: a keeper, its exact duplicate, a second keeper, one sentence
#: whose hash is in the exclusion list, and one that near-duplicates the planted
#: evaluation sentence without hashing to it.
_DCS_TRAIN: list[dict[str, str]] = [
    {"text_slp1": "rAmaH vanaM gacCati", "oracle_split_slp1": "rAmaH vanam gacCati"},
    {"text_slp1": "rAmaH vanaM gacCati", "oracle_split_slp1": "rAmaH vanam gacCati"},
    {"text_slp1": "sItA gfhe tizWati", "oracle_split_slp1": "sItA gfhe tizWati"},
    {"text_slp1": "leaked by hash sentence", "oracle_split_slp1": "leaked by hash sentence"},
    {
        "text_slp1": "atra aBiDarmakoSam pravakzyAmi SAstram iti",
        "oracle_split_slp1": "atra aBiDarma koSam pravakzyAmi SAstram iti",
    },
]

#: The evaluation sentence the fifth training record quotes; its letters are a 31-letter
#: run inside that record's, so the 24-letter windows overlap while the hashes do not.
_PLANTED_EVAL = "aBiDarmakoSam pravakzyAmi SAstram"

_DCS_HELDOUT: list[dict[str, str]] = [
    {"text_slp1": "guruH CAtrAn pAWayati", "oracle_split_slp1": "guruH CAtrAn pAWayati"},
    {"text_slp1": "vAyuH vahati", "oracle_split_slp1": "vAyuH vahati"},
]

#: The four parallel evaluation corpora, as `data/processed/split/<name>.jsonl` stores
#: them: the raw Devanagari and its SLP1, the model output and the reconciled split.
_PARALLEL: dict[str, list[tuple[str, str, str]]] = {
    "samayik_test": [("गुरुः पाठयति।", "guruH pAWayati.", "guruH pAWayati")],
    "samayik_test_ood": [("वायुः वहति।", "vAyuH vahati.", "vAyuH vahati")],
    "itihasa_test": [("रामः वनम्।", "rAmaH vanam.", "rAmaH vanam")],
    "flores_devtest": [("सीता गृहे।", "sItA gfhe.", "sItA gfhe")],
}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class _FakeTokenizer:
    """One token per non-space character, so a token count is a length."""

    name = "fake"
    vocab_size = 8

    def encode(self, text: str) -> list[int]:
        return [1] * len(text.replace(" ", ""))


@pytest.fixture
def corpus_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A full tiny corpus tree plus the config that builds it, with loaders stubbed."""
    _write_jsonl(tmp_path / "dcs" / "train.jsonl", list(_DCS_TRAIN))
    _write_jsonl(tmp_path / "dcs" / "heldout.jsonl", list(_DCS_HELDOUT))
    for name, rows in _PARALLEL.items():
        _write_jsonl(
            tmp_path / "split" / f"{name}.jsonl",
            [
                {
                    "index": index,
                    "raw_deva": deva,
                    "raw_slp1": slp1,
                    "output_model": split,
                    "output": split,
                }
                for index, (deva, slp1, split) in enumerate(rows)
            ],
        )
    build_exclusion_list(
        {"planted": [_DCS_TRAIN[3]["text_slp1"]]},
        tmp_path / "exclusion_hashes.txt",
        hash_fn=sentence_hash_slp1,
    )
    _write_parquet(
        tmp_path / "sangraha" / "verified" / "san" / "data-0.parquet",
        {
            "doc_id": ["a"],
            "text": ["नृपः नगरं गच्छति। गुरुः छात्रान् पाठयति।"],
            "type": ["pdf"],
        },
    )
    _write_parquet(
        tmp_path / "wikipedia" / wikipedia_sa.WIKIPEDIA_CONFIG / "train-00000-of-00001.parquet",
        {
            "id": ["1"],
            "url": ["https://sa.wikipedia.org/wiki/x"],
            "title": ["श्रीलङ्का"],
            "text": ["श्रीलङ्का\n== इतिहासः ==\nअस्य द्वीपराष्ट्रस्य राजधानी कोलम्बो अस्ति।"],
        },
    )

    # The two parallel training sides, and the loader the alignment cross-check uses.
    monkeypatch.setattr(
        build_corpus,
        "SANSKRIT_SOURCE_LOADERS",
        {
            "samayik_train": lambda: ["नृपः नगरं गच्छति।", "जलं पिबति नरः।"],
            "itihasa_train": lambda: ["रामः सीतां पश्यति।"],
        },
    )

    def fake_load_corpus_entry(entry: dict[str, Any], root: Path) -> ParallelCorpus:
        rows = _PARALLEL[str(entry["name"])]
        return ParallelCorpus(
            name=str(entry["name"]),
            split=str(entry["split"]),
            languages=("san_Deva", "eng_Latn"),
            sentences={
                "san_Deva": [deva for deva, _, _ in rows],
                "eng_Latn": ["english"] * len(rows),
            },
        )

    monkeypatch.setattr(build_corpus, "load_corpus_entry", fake_load_corpus_entry)
    monkeypatch.setattr(build_corpus, "load_tokenizer", lambda name: _FakeTokenizer())

    config: dict[str, Any] = {
        "out_dir": str(tmp_path / "lm"),
        "exclusion_path": str(tmp_path / "exclusion_hashes.txt"),
        "shingle_k": SHINGLE_K,
        "flores_devtest_jsonl": str(tmp_path / "flores_devtest.jsonl"),
        "dcs": {
            "train_jsonl": str(tmp_path / "dcs" / "train.jsonl"),
            "heldout_jsonl": str(tmp_path / "dcs" / "heldout.jsonl"),
        },
        "sangraha": {
            "cache_dir": str(tmp_path / "sangraha"),
            "files": ["verified/san/data-0.parquet"],
        },
        "wikipedia": {"cache_dir": str(tmp_path / "wikipedia")},
        "track2_sources": [
            "dcs_train",
            "samayik_train_sa",
            "itihasa_train_sa",
            "sangraha_verified_san",
            "wikipedia_sa",
        ],
        "heldout_parallel": [
            {
                "name": name,
                "split_jsonl": str(tmp_path / "split" / f"{name}.jsonl"),
                "corpus": {"name": name, "loader": "stub", "split": "test"},
            }
            for name in _PARALLEL
        ],
        "token_count_arm": "fake",
    }
    return {"tmp_path": tmp_path, "config": config}


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_build_corpus_writes_three_corpora_and_heldout_files(
    corpus_case: dict[str, Any],
) -> None:
    tmp_path = corpus_case["tmp_path"]
    indices = {"flores_devtest": build_shingle_index([_PLANTED_EVAL])}
    manifest = build_corpus.build(
        corpus_case["config"], root=tmp_path, shingle_indices=indices
    )
    out = tmp_path / "lm"

    # Track 1: the two leaked records and the duplicate are gone; the split file is the
    # same sentences in the same order.
    assert _read_lines(out / "track1_raw.txt") == [
        "rAmaH vanaM gacCati",
        "sItA gfhe tizWati",
    ]
    assert _read_lines(out / "track1_split.txt") == [
        "rAmaH vanam gacCati",
        "sItA gfhe tizWati",
    ]

    # Track 2: DCS first, then the parallel training sides, then Sangraha (whose first
    # line duplicates the first Sāmayik one and is removed), then Wikipedia.
    assert _read_lines(out / "track2_raw.txt") == [
        "rAmaH vanaM gacCati",
        "sItA gfhe tizWati",
        "nfpaH nagaraM gacCati.",
        "jalaM pibati naraH.",
        "rAmaH sItAM paSyati.",
        "guruH CAtrAn pAWayati.",
        "asya dvIparAzwrasya rAjaDAnI kolambo asti.",
    ]

    # Held-out evaluation texts, raw and split, one line per sentence.
    assert _read_lines(out / "heldout_dcs.txt") == [
        "guruH CAtrAn pAWayati",
        "vAyuH vahati",
    ]
    assert _read_lines(out / "heldout_dcs_split.txt") == [
        "guruH CAtrAn pAWayati",
        "vAyuH vahati",
    ]
    for name, rows in _PARALLEL.items():
        assert _read_lines(out / f"heldout_{name}.txt") == [slp1 for _, slp1, _ in rows]
        assert _read_lines(out / f"heldout_{name}_split.txt") == [
            split for _, _, split in rows
        ]

    assert set(manifest["heldout"]) == {"dcs", "dcs_split"} | {
        f"{name}{suffix}" for name in _PARALLEL for suffix in ("", "_split")
    }


def test_build_corpus_manifest_records_every_drop(corpus_case: dict[str, Any]) -> None:
    tmp_path = corpus_case["tmp_path"]
    indices = {"flores_devtest": build_shingle_index([_PLANTED_EVAL])}
    manifest = build_corpus.build(
        corpus_case["config"], root=tmp_path, shingle_indices=indices
    )

    track1 = manifest["corpora"]["track1_raw"]
    assert track1["n_in"] == {"dcs_train": 5}
    assert track1["n_dropped_hash"] == 1
    assert track1["n_dropped_shingle"] == 1
    assert track1["n_dropped_shingle_per_source"] == {"flores_devtest": 1}
    assert track1["n_dedup_removed"] == 1
    assert track1["n_out"] == 2
    assert track1["n_chars"] == len("rAmaH vanaM gacCati") + len("sItA gfhe tizWati")
    assert track1["n_bytes"] == track1["n_chars"]  # SLP1 is ASCII
    assert len(track1["sha256"]) == 64
    assert track1["n_tokens"] == sum(
        len(line.replace(" ", "")) for line in ("rAmaH vanaM gacCati", "sItA gfhe tizWati")
    )
    assert track1["token_count_arm"] == "fake"

    track2 = manifest["corpora"]["track2_raw"]
    assert track2["n_in"] == {
        "dcs_train": 5,
        "samayik_train_sa": 2,
        "itihasa_train_sa": 1,
        "sangraha_verified_san": 2,
        "wikipedia_sa": 1,
    }
    assert track2["n_dedup_removed"] == 2
    assert track2["n_dedup_removed_per_source"] == {
        "dcs_train": 1,
        "samayik_train_sa": 0,
        "itihasa_train_sa": 0,
        "sangraha_verified_san": 1,
        "wikipedia_sa": 0,
    }
    assert track2["n_out"] == 7
    assert "git_commit" in manifest and "timestamp" in manifest

    # A manifest per corpus lands beside it, holding that corpus's entry.
    on_disk = json.loads(
        (tmp_path / "lm" / "track2_raw.manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk["n_out"] == track2["n_out"]
    assert on_disk["sha256"] == track2["sha256"]


def test_build_corpus_rejects_a_misaligned_parallel_heldout(
    corpus_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A split jsonl that has drifted from its loader is fatal, not silently misaligned."""
    tmp_path = corpus_case["tmp_path"]

    def drifted(entry: dict[str, Any], root: Path) -> ParallelCorpus:
        return ParallelCorpus(
            name=str(entry["name"]),
            split=str(entry["split"]),
            languages=("san_Deva",),
            sentences={"san_Deva": ["अन्यत् वाक्यम्।"]},
        )

    monkeypatch.setattr(build_corpus, "load_corpus_entry", drifted)
    with pytest.raises(build_corpus.CorpusError, match="alignment"):
        build_corpus.build(corpus_case["config"], root=tmp_path, shingle_indices={})


def test_corpus_yaml_matches_the_builder(corpus_case: dict[str, Any]) -> None:
    """The committed config names the sources and outputs the builder knows."""
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    assert set(config["track2_sources"]) <= set(build_corpus.SOURCE_KEYS)
    assert config["track2_sources"][0] == "dcs_train"
    assert [entry["name"] for entry in config["heldout_parallel"]] == list(_PARALLEL)
    assert config["token_count_arm"] == "T1_bpe_raw_64k_dcs"


def test_write_lines_reports_chars_bytes_and_digest(tmp_path: Path) -> None:
    """The streaming writer's totals are of the file it wrote, newlines excluded."""
    path = tmp_path / "corpus.txt"

    def lines() -> Iterator[str]:
        yield "rAmaH vanam"
        yield "sItA gfhe"

    stats = build_corpus.write_lines(path, lines())
    assert stats.n_out == 2
    assert stats.n_chars == len("rAmaH vanam") + len("sItA gfhe")
    assert stats.n_bytes == stats.n_chars
    assert path.read_text(encoding="utf-8") == "rAmaH vanam\nsItA gfhe\n"
    assert len(stats.sha256) == 64
