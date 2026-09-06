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
from sanskrit_tok.data.quality import QUALITY_RULES

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


def test_documents_to_lines_does_not_count_a_danda_as_a_word() -> None:
    """`राम ।` is one word, not two: the base splitter's one-word-line gap.

    `piece.split()` counts the danda, so a two-word floor let one-word lines through
    (docs/decisions.md, 2026-09-05, rule 5). `real_words` is the fix, and it is the same
    function the quality filter's word-count rule uses.
    """
    assert sangraha.documents_to_lines("रामः ।") == []
    assert sangraha.documents_to_lines("रामः ॥ १२ ॥") == []
    assert sangraha.documents_to_lines("रामः वनम् ।") == ["रामः वनम् ।"]


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


@pytest.mark.parametrize(
    "name",
    [
        "synthetic/san_Deva/wiki_0.parquet",  # machine translation
        "unverified/san/data-0.parquet",
        "verified/hin/data-0.parquet",  # another language
        "verified/san/data-0.txt",  # not a parquet
        "verified/san_Deva/data-0.parquet",  # the directory that does not exist
    ],
)
def test_assert_verified_sanskrit_files_rejects_anything_outside_the_prefix(name: str) -> None:
    """A caller-supplied `files` list is validated, not trusted.

    `verified_sanskrit_files` only guarantees the *default* list; `corpus.yaml` has a
    `files:` key, and a name pointing at `synthetic/` would put machine-translated
    Wikipedia into a Sanskrit LM's training data without a word in any log.
    """
    with pytest.raises(ValueError, match=name):
        sangraha.assert_verified_sanskrit_files(["verified/san/data-0.parquet", name])


def test_assert_verified_sanskrit_files_accepts_and_preserves_order() -> None:
    files = ["verified/san/data-1.parquet", "verified/san/data-0.parquet"]
    assert sangraha.assert_verified_sanskrit_files(files) == files


def test_iter_sangraha_sanskrit_validates_the_file_list(tmp_path: Path) -> None:
    """The reader refuses the same list, so no caller can bypass the check."""
    with pytest.raises(ValueError, match="synthetic"):
        list(
            sangraha.iter_sangraha_sanskrit(
                tmp_path, files=["synthetic/san_Deva/wiki_0.parquet"]
            )
        )


def test_resolve_sanskrit_files_applies_max_files() -> None:
    files = [f"verified/san/data-{index}.parquet" for index in range(4)]
    assert sangraha.resolve_sanskrit_files(files=files, max_files=2) == files[:2]


def test_sangraha_file_sizes_reports_what_is_on_disk(tmp_path: Path) -> None:
    name = "verified/san/data-0.parquet"
    _write_parquet(tmp_path / name, {"doc_id": ["a"], "text": ["रामः वनं गच्छति।"], "type": ["pdf"]})
    sizes = sangraha.sangraha_file_sizes(tmp_path, [name, "verified/san/data-9.parquet"])
    assert sizes[0]["name"] == name
    assert sizes[0]["n_bytes"] == (tmp_path / name).stat().st_size
    assert sizes[1]["n_bytes"] is None


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


# ------------------------------------------------------------------- the quality filter


def test_build_corpus_drops_web_lines_by_rule_and_counts_each(
    corpus_case: dict[str, Any],
) -> None:
    """Each quality rule's drop is counted under its own name, per source.

    The four planted Sangraha documents fail one rule each, in the order the rules run;
    the fifth is ordinary Sanskrit and survives. DCS and the parallel sides are SLP1
    already and are not passed through the filter at all, so their counters stay zero.
    """
    tmp_path = corpus_case["tmp_path"]
    _write_parquet(
        tmp_path / "sangraha" / "verified" / "san" / "data-0.parquet",
        {
            "doc_id": ["a", "b", "c", "d", "e"],
            "text": [
                "रामः vanam गच्छति।",  # latin
                "राम का पुत्र है वनं गच्छति।",  # hindi
                "रामः ३४५६७ वनं ८९ गच्छति।",  # letter_fraction
                "रामः ﾱ वनं गच्छति।",  # non_slp1
                "गुरुः छात्रान् पाठयति।",  # kept
            ],
            "type": ["pdf"] * 5,
        },
    )
    manifest = build_corpus.build(
        corpus_case["config"],
        root=tmp_path,
        shingle_indices={"flores_devtest": build_shingle_index([_PLANTED_EVAL])},
    )
    track2 = manifest["corpora"]["track2_raw"]
    assert track2["n_dropped_quality_per_source_per_rule"]["sangraha_verified_san"] == {
        "latin": 1,
        "hindi": 1,
        "letter_fraction": 1,
        "n_words": 0,
        "word_length": 0,
        "non_slp1": 1,
    }
    assert track2["n_dropped_quality_per_source"]["sangraha_verified_san"] == 4
    assert track2["n_dropped_quality"] == 4
    assert track2["n_dropped_quality_per_source"]["dcs_train"] == 0
    assert set(track2["n_dropped_quality_per_rule"]) == set(QUALITY_RULES)
    assert _read_lines(tmp_path / "lm" / "track2_raw.txt")[-2] == "guruH CAtrAn pAWayati."


def test_build_corpus_drops_an_unspellable_slp1_line_from_every_track(
    corpus_case: dict[str, Any], tmp_path: Path
) -> None:
    """A DCS record carrying a character SLP1 cannot spell reaches no corpus and no held-out
    file, and is counted as `non_slp1` rather than as an empty line."""
    case_path = corpus_case["tmp_path"]
    _write_jsonl(
        case_path / "dcs" / "train.jsonl",
        [
            {"text_slp1": "rAmaH vanaM gacCati", "oracle_split_slp1": "rAmaH vanam gacCati"},
            {"text_slp1": "sItA ﾱ gfhe", "oracle_split_slp1": "sItA ﾱ gfhe"},
        ],
    )
    _write_jsonl(
        case_path / "dcs" / "heldout.jsonl",
        [
            {"text_slp1": "guruH CAtrAn pAWayati", "oracle_split_slp1": "guruH CAtrAn pAWayati"},
            {"text_slp1": "vAyuH ﾱ vahati", "oracle_split_slp1": "vAyuH ﾱ vahati"},
        ],
    )
    manifest = build_corpus.build(
        corpus_case["config"], root=case_path, shingle_indices={}
    )
    track1 = manifest["corpora"]["track1_raw"]
    assert track1["n_dropped_non_slp1_per_source"] == {"dcs_train": 1}
    assert track1["n_dropped_empty"] == 0
    assert _read_lines(case_path / "lm" / "track1_raw.txt") == ["rAmaH vanaM gacCati"]
    assert _read_lines(case_path / "lm" / "track1_split.txt") == ["rAmaH vanam gacCati"]
    assert manifest["corpora"]["track2_raw"]["n_dropped_non_slp1_per_source"]["dcs_train"] == 1

    # The held-out text is not deduplicated and not leakage-filtered, but it is cleaned:
    # a character no vocabulary can spell would sit in the bits-per-character denominator.
    assert manifest["heldout_dropped"]["dcs_non_slp1"] == 1
    assert _read_lines(case_path / "lm" / "heldout_dcs.txt") == ["guruH CAtrAn pAWayati"]
    assert _read_lines(case_path / "lm" / "heldout_dcs_split.txt") == ["guruH CAtrAn pAWayati"]


def test_build_corpus_records_the_resolved_sangraha_files(
    corpus_case: dict[str, Any],
) -> None:
    """The manifest names the files the build read, with their sizes."""
    tmp_path = corpus_case["tmp_path"]
    manifest = build_corpus.build(
        corpus_case["config"], root=tmp_path, shingle_indices={}
    )
    sangraha_provenance = manifest["corpora"]["track2_raw"]["sources"][
        "sangraha_verified_san"
    ]
    assert sangraha_provenance["n_files"] == 1
    assert sangraha_provenance["files"] == [
        {
            "name": "verified/san/data-0.parquet",
            "n_bytes": (
                tmp_path / "sangraha" / "verified" / "san" / "data-0.parquet"
            ).stat().st_size,
        }
    ]
    assert sangraha_provenance["subset"] == "verified/san"


def test_build_corpus_rejects_a_synthetic_sangraha_file_in_the_config(
    corpus_case: dict[str, Any],
) -> None:
    """The `files:` key cannot smuggle machine translation into the corpus."""
    config = dict(corpus_case["config"])
    config["sangraha"] = {
        **config["sangraha"],
        "files": ["synthetic/san_Deva/wiki_0.parquet"],
    }
    with pytest.raises(ValueError, match="synthetic"):
        build_corpus.build(config, root=corpus_case["tmp_path"], shingle_indices={})


# -------------------------------------------------------------------- the Track 2 sample


def test_sampling_probability_is_the_share_of_the_budget_left_over() -> None:
    """Hand-computed: 100 bytes of budget, 20 already spent, 160 available -> 0.5."""
    assert build_corpus.sampling_probability(100, 20, 160) == 0.5
    # Never above 1: a budget larger than the corpus keeps all of it.
    assert build_corpus.sampling_probability(1000, 20, 160) == 1.0
    assert build_corpus.sampling_probability(1000, 20, 0) == 0.0
    with pytest.raises(build_corpus.CorpusError, match="already exceeds"):
        build_corpus.sampling_probability(10, 20, 160)


def _sample_case(corpus_case: dict[str, Any], n_documents: int) -> dict[str, Any]:
    """`corpus_case` with a Sangraha file of `n_documents` distinct one-line documents."""
    tmp_path = corpus_case["tmp_path"]
    _write_parquet(
        tmp_path / "sangraha" / "verified" / "san" / "data-0.parquet",
        {
            "doc_id": [str(index) for index in range(n_documents)],
            "text": [f"गुरुः छात्रान् पाठयति {index} वारम्।" for index in range(n_documents)],
            "type": ["pdf"] * n_documents,
        },
    )
    return corpus_case


def test_build_sample_keeps_every_unsampled_source_and_thins_sangraha(
    corpus_case: dict[str, Any],
) -> None:
    """Half the Sangraha bytes, all of everything else, and the same file twice over."""
    tmp_path = _sample_case(corpus_case, 40)["tmp_path"]
    config = dict(corpus_case["config"])
    manifest = build_corpus.build(config, root=tmp_path, shingle_indices={})
    track2 = manifest["corpora"]["track2_raw"]
    ranges = track2["source_line_ranges"]
    assert ranges["dcs_train"][0] == 0
    assert ranges["wikipedia_sa"][1] == track2["n_out"]

    composition_bytes = {}
    lines = _read_lines(tmp_path / "lm" / "track2_raw.txt")
    for source, (start, end) in ranges.items():
        composition_bytes[source] = sum(
            len(line.encode("utf-8")) for line in lines[start:end]
        )
    kept = sum(
        count for source, count in composition_bytes.items()
        if source != "sangraha_verified_san"
    )
    config["track2_sample_bytes"] = kept + composition_bytes["sangraha_verified_san"] // 2
    config["track2_sample_sampled_sources"] = ["sangraha_verified_san"]
    config["track2_sample_seed"] = 0

    entry = build_corpus.build_sample(config, root=tmp_path)
    assert entry["keep_probability"] == pytest.approx(0.5, abs=0.01)
    for source, composition in entry["composition"].items():
        if source == "sangraha_verified_san":
            assert 0 < composition["n_out"] < composition["n_in"]
        else:
            assert composition["n_out"] == composition["n_in"]
    assert entry["n_out"] == sum(
        composition["n_out"] for composition in entry["composition"].values()
    )
    assert entry["n_tokens"] > 0
    assert entry["sampled_from_sha256"] == track2["sha256"]

    # Every sampled line is a line of the corpus it was sampled from, and the seed makes
    # the sample reproducible.
    sample_lines = _read_lines(tmp_path / "lm" / "track2_sample.txt")
    assert set(sample_lines) <= set(lines)
    first = entry["sha256"]
    assert build_corpus.build_sample(config, root=tmp_path)["sha256"] == first


def test_build_sample_needs_a_corpus_with_source_ranges(
    corpus_case: dict[str, Any],
) -> None:
    """`--sample` refuses to guess: no corpus, or a manifest predating the ranges, is fatal."""
    tmp_path = corpus_case["tmp_path"]
    config = dict(corpus_case["config"])
    config["track2_sample_bytes"] = 10_000
    with pytest.raises(build_corpus.CorpusError, match="run the build without"):
        build_corpus.build_sample(config, root=tmp_path)

    build_corpus.build(config, root=tmp_path, shingle_indices={})
    manifest_path = tmp_path / "lm" / "track2_raw.manifest.json"
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    del stale["source_line_ranges"]
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(build_corpus.CorpusError, match="source_line_ranges"):
        build_corpus.build_sample(config, root=tmp_path)


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


# ------------------------------------------------ held-out rebuild and typographic ASCII


def test_build_heldout_rewrites_the_heldout_files_and_leaves_the_corpora_alone(
    corpus_case: dict[str, Any],
) -> None:
    """`--heldout-only`: the evaluation text is rebuilt, M1 is not, the manifest agrees."""
    tmp_path = corpus_case["tmp_path"]
    manifest = build_corpus.build(corpus_case["config"], root=tmp_path, shingle_indices={})
    out = tmp_path / "lm"
    track1_sha = manifest["corpora"]["track1_raw"]["sha256"]
    (out / "track1_raw.txt").write_text("not rebuilt\n", encoding="utf-8")
    (out / "heldout_dcs.txt").unlink()

    entry = build_corpus.build_heldout(corpus_case["config"], root=tmp_path)

    assert _read_lines(out / "heldout_dcs.txt") == ["guruH CAtrAn pAWayati", "vAyuH vahati"]
    assert (out / "track1_raw.txt").read_text(encoding="utf-8") == "not rebuilt\n"
    patched = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert patched["corpora"]["track1_raw"]["sha256"] == track1_sha
    assert patched["heldout"] == entry["heldout"]
    assert patched["heldout_dropped"] == entry["heldout_dropped"]
    assert "heldout_rebuilt" in patched


def test_heldout_text_keeps_a_line_whose_only_defect_was_its_typesetting(
    corpus_case: dict[str, Any],
) -> None:
    """A curly-quoted, em-dashed held-out line survives as ASCII instead of being dropped."""
    tmp_path = corpus_case["tmp_path"]
    heldout_path = tmp_path / "dcs" / "heldout.jsonl"
    records = [json.loads(line) for line in _read_lines(heldout_path) if line.strip()]
    records.append(
        {
            "text_slp1": "“vAyuH vahati” — iti…",
            "oracle_split_slp1": "“vAyuH vahati” — iti…",
        }
    )
    _write_jsonl(heldout_path, records)

    build_corpus.build_heldout(corpus_case["config"], root=tmp_path)

    lines = _read_lines(tmp_path / "lm" / "heldout_dcs.txt")
    assert lines[-1] == '"vAyuH vahati" - iti...'


# ------------------------------------------------ Track 2's own in-domain held-out set


def _sampled_case(corpus_case: dict[str, Any], n_documents: int = 60) -> dict[str, Any]:
    """Build the tiny corpora and take a full-budget Track 2 sample of them."""
    tmp_path = _sample_case(corpus_case, n_documents)["tmp_path"]
    config = dict(corpus_case["config"])
    build_corpus.build(config, root=tmp_path, shingle_indices={})
    config["track2_sample_bytes"] = 10_000_000  # far above the corpus: keep everything
    config["track2_sample_sampled_sources"] = ["sangraha_verified_san"]
    config["track2_sample_seed"] = 0
    build_corpus.build_sample(config, root=tmp_path)
    return {"tmp_path": tmp_path, "config": config}


def test_sample_line_ranges_are_the_cumulative_composition() -> None:
    ranges = build_corpus.sample_line_ranges(
        {"a": {"n_out": 3}, "b": {"n_out": 0}, "c": {"n_out": 2}}
    )
    assert ranges == [("a", 0, 3), ("b", 3, 3), ("c", 3, 5)]


def test_sangraha_heldout_leaves_the_sample_without_its_lines_or_their_near_duplicates(
    corpus_case: dict[str, Any],
) -> None:
    """The drawn lines go, and so does anything sharing a 24-letter shingle with them.

    Track 2's sample is 93.6% Sangraha in production, so without this the scale track has
    no in-domain evaluation at all (docs/decisions.md, 2026-09-06).
    """
    case = _sampled_case(corpus_case)
    tmp_path, config = case["tmp_path"], case["config"]
    sample_path = tmp_path / "lm" / "track2_sample.txt"
    before = _read_lines(sample_path)

    record = build_corpus.build_sangraha_heldout(config, n=5, seed=0, root=tmp_path)

    heldout = _read_lines(tmp_path / "lm" / "heldout_sangraha.txt")
    after = _read_lines(sample_path)
    assert len(heldout) == 5 == record["n_heldout"]
    assert set(heldout) <= set(before)
    # Nothing held out is still in the sample, by line or by shingle.
    assert not set(heldout) & set(after)
    index = build_shingle_index(heldout, build_corpus.SHINGLE_K)
    for line in after:
        assert not has_shingle_overlap(line, index, build_corpus.SHINGLE_K)
    assert len(after) == len(before) - record["n_removed_selected"] - record["n_removed_shingle"]

    # The sample's manifest describes the file that is now on disk.
    manifest = json.loads((tmp_path / "lm" / "track2_sample.manifest.json").read_text("utf-8"))
    assert manifest["n_out"] == len(after)
    assert manifest["n_bytes"] == sum(len(line.encode("utf-8")) for line in after)
    assert manifest["sangraha_heldout"]["seed"] == 0
    assert manifest["sangraha_heldout"]["n_removed_selected"] == 5
    assert sum(entry["n_out"] for entry in manifest["composition"].values()) == len(after)

    # ... and so does the top-level one, under the key the sweep reads.
    top = json.loads((tmp_path / "lm" / "manifest.json").read_text("utf-8"))
    assert top["heldout"]["sangraha"]["n_out"] == 5
    assert top["heldout"]["sangraha"]["path"].endswith("heldout_sangraha.txt")


def test_sangraha_heldout_refuses_to_draw_a_second_time(
    corpus_case: dict[str, Any],
) -> None:
    """Two draws would hold out two sets while the exclusion list named only one."""
    case = _sampled_case(corpus_case)
    build_corpus.build_sangraha_heldout(case["config"], n=3, seed=0, root=case["tmp_path"])
    with pytest.raises(build_corpus.CorpusError, match="already exists"):
        build_corpus.build_sangraha_heldout(case["config"], n=3, seed=0, root=case["tmp_path"])


def test_sangraha_heldout_needs_a_sample_and_enough_sangraha_lines(
    corpus_case: dict[str, Any],
) -> None:
    tmp_path = corpus_case["tmp_path"]
    with pytest.raises(build_corpus.CorpusError, match="run the build with --sample"):
        build_corpus.build_sangraha_heldout(corpus_case["config"], n=1, root=tmp_path)
    case = _sampled_case(corpus_case)
    with pytest.raises(build_corpus.CorpusError, match="only"):
        build_corpus.build_sangraha_heldout(
            case["config"], n=10_000, root=case["tmp_path"]
        )


def test_a_later_corpus_build_filters_against_the_sangraha_held_out_lines(
    corpus_case: dict[str, Any],
) -> None:
    """Once drawn, the held-out lines are an evaluation source like any other."""
    case = _sampled_case(corpus_case)
    build_corpus.build_sangraha_heldout(case["config"], n=5, seed=0, root=case["tmp_path"])
    heldout_path = case["tmp_path"] / "lm" / "heldout_sangraha.txt"
    assert heldout_path.exists()
    # `build` picks the file up as an extra shingle source when it builds its own index.
    captured: dict[str, Any] = {}

    def _capture(flores_path: Path, k: int, *, extra_sources: Any = None) -> dict[str, Any]:
        captured["extra"] = dict(extra_sources or {})
        return {}

    original = build_corpus.build_evaluation_shingle_index
    build_corpus.build_evaluation_shingle_index = _capture  # type: ignore[assignment]
    try:
        build_corpus.build(case["config"], root=case["tmp_path"])
    finally:
        build_corpus.build_evaluation_shingle_index = original  # type: ignore[assignment]
    assert "sangraha_heldout" in captured["extra"]
    assert len(list(captured["extra"]["sangraha_heldout"])) == 5
