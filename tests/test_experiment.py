"""Tests for `sanskrit_tok.experiment`, the helpers every experiment script shares.

These pin the behaviour three scripts previously each carried their own copy of
(`experiments/01_baseline_penalty/run.py`, `experiments/02_tpp_parallel/run.py`,
`experiments/02_tpp_parallel/train_tokenizers.py`): config-path resolution against the
repository root, YAML loading, git provenance, blank-index filtering, JSON sanitising,
the `results.json`/`config.yaml` writer, arm loading, the `tokenizer_sources` provenance
block, and the leakage check and caption that go with them. Several of them started life
in `tests/test_exp01.py` and `tests/test_exp02.py` and moved here with the code, the last
five when Experiment 03 turned out to need the same helpers verbatim.

`provenance` runs `git` for real in a throwaway repository, for the reason
`tests/test_provenance.py` gives: the subprocess call is the thing under test.
"""

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sanskrit_tok.data.exclusion import sentence_hash, sentence_hash_en
from sanskrit_tok.experiment import (
    exclusion_check_for,
    load_arms,
    load_config,
    provenance,
    repo_root,
    resolve_path,
    sanitize_json,
    select_aligned_indices,
    summarise_tpp,
    take_indices,
    tokenizer_file_sha256,
    tokenizer_sources,
    unavailable_caption,
    write_results,
)
from sanskrit_tok.tokenizers.registry import LoadedTokenizer, TokenizerUnavailable

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- paths and config -------------------------------------------------------------


def test_repo_root_is_the_parent_of_the_experiments_directory() -> None:
    assert repo_root() == REPO_ROOT
    assert (repo_root() / "experiments").is_dir()
    assert (repo_root() / "src" / "sanskrit_tok").is_dir()


def test_resolve_path_makes_relative_config_paths_root_relative(tmp_path: Path) -> None:
    assert resolve_path("a/b.json", tmp_path) == tmp_path / "a" / "b.json"


def test_resolve_path_leaves_absolute_paths_alone(tmp_path: Path) -> None:
    absolute = tmp_path / "already" / "absolute.json"
    assert resolve_path(str(absolute), tmp_path / "elsewhere") == absolute


def test_resolve_path_defaults_to_the_repository_root() -> None:
    assert resolve_path("data/raw") == REPO_ROOT / "data" / "raw"


def test_resolve_path_accepts_a_path_as_well_as_a_string(tmp_path: Path) -> None:
    assert resolve_path(Path("a/b.json"), tmp_path) == tmp_path / "a" / "b.json"


def test_load_config_reads_a_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 0\nlanguages: [san_Deva, eng_Latn]\n", encoding="utf-8")
    assert load_config(path) == {"seed": 0, "languages": ["san_Deva", "eng_Latn"]}


def test_load_config_rejects_a_yaml_document_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_config(path)


def test_load_config_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_config(path)


# --- provenance -------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return completed.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny git repository with one committed file and a clean working tree."""
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


def test_provenance_records_commit_dirty_and_timestamp(repo: Path) -> None:
    recorded = provenance(repo)
    assert list(recorded) == ["git_commit", "git_dirty", "timestamp"]
    assert recorded["git_commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert recorded["git_dirty"] is False


def test_provenance_timestamp_is_iso_8601_utc_to_the_second(repo: Path) -> None:
    from datetime import datetime

    stamp = str(provenance(repo)["timestamp"])
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    assert parsed.microsecond == 0


def test_provenance_reports_a_dirty_tree(repo: Path) -> None:
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    assert provenance(repo)["git_dirty"] is True


def test_provenance_defaults_to_the_repository_root() -> None:
    """No argument means this repository, which is the only root the scripts ever pass."""
    assert len(str(provenance()["git_commit"])) in (7, 40)


# --- blank-line filtering ---------------------------------------------------------


def test_select_aligned_indices_drops_every_index_blank_in_any_language() -> None:
    sentences = {
        "a": ["one", "", "three", "four"],
        "b": ["uno", "dos", "   ", "cuatro"],
    }
    assert select_aligned_indices(sentences) == [0, 3]


def test_select_aligned_indices_keeps_everything_when_nothing_is_blank() -> None:
    assert select_aligned_indices({"a": ["x", "y"], "b": ["p", "q"]}) == [0, 1]


def test_select_aligned_indices_of_nothing_is_empty() -> None:
    assert select_aligned_indices({}) == []


def test_select_aligned_indices_rejects_misaligned_languages() -> None:
    with pytest.raises(ValueError, match="languages differ in length"):
        select_aligned_indices({"a": ["x", "y"], "b": ["p"]})


def test_take_indices_preserves_alignment() -> None:
    sentences = {"a": ["one", "", "three"], "b": ["uno", "dos", "tres"]}
    assert take_indices(sentences, [0, 2]) == {"a": ["one", "three"], "b": ["uno", "tres"]}


# --- JSON sanitising --------------------------------------------------------------


def test_sanitize_json_turns_nan_and_inf_into_none() -> None:
    sanitized = sanitize_json({"a": math.nan, "b": math.inf, "c": [1.0, math.nan]})
    assert sanitized == {"a": None, "b": None, "c": [1.0, None]}
    # Must survive a strict json.dump (allow_nan=False) with no exception.
    assert json.loads(json.dumps(sanitized, allow_nan=False)) == sanitized


def test_sanitize_json_leaves_ordinary_values_alone() -> None:
    assert sanitize_json({"value": 1.5, "n": 3, "unit": "x", "nested": {"y": [1, 2]}}) == {
        "value": 1.5,
        "n": 3,
        "unit": "x",
        "nested": {"y": [1, 2]},
    }


def test_sanitize_json_turns_tuples_into_lists() -> None:
    assert sanitize_json({"attempted": ("a", "b")}) == {"attempted": ["a", "b"]}


def test_sanitize_json_turns_paths_into_strings(tmp_path: Path) -> None:
    assert sanitize_json({"path": tmp_path}) == {"path": str(tmp_path)}


# --- results writing --------------------------------------------------------------


def test_write_results_writes_strict_json_and_returns_its_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs" / "99_demo"
    path = write_results({"value": math.nan, "n": 2}, out_dir)
    assert path == out_dir / "results.json"
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert text.endswith("\n")
    assert json.loads(text) == {"value": None, "n": 2}


def test_write_results_indents_by_two_and_keeps_unicode(tmp_path: Path) -> None:
    path = write_results({"sentence": "अथ", "nested": {"a": 1}}, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "अथ" in text
    assert '\n  "nested": {\n    "a": 1\n  }' in text


def test_write_results_copies_the_config_byte_for_byte(tmp_path: Path) -> None:
    config_src = tmp_path / "src.yaml"
    config_src.write_text("seed: 0\n# a comment the copy must keep\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    write_results({"a": 1}, out_dir, config_src)
    assert (out_dir / "config.yaml").read_bytes() == config_src.read_bytes()


def test_write_results_without_a_config_writes_no_config_yaml(tmp_path: Path) -> None:
    write_results({"a": 1}, tmp_path)
    assert not (tmp_path / "config.yaml").exists()


def test_write_results_does_not_mutate_the_results_it_is_given(tmp_path: Path) -> None:
    results: dict[str, Any] = {"value": math.nan}
    write_results(results, tmp_path)
    assert math.isnan(results["value"])


# --- TPP summary enrichment -------------------------------------------------------


def test_summarise_tpp_copies_bootstrap_keys_and_sets_ci() -> None:
    raw = {
        "value": 1.5,
        "n": 2,
        "unit": "tokens/proposition ratio",
        "per_pair": [1.0, 2.0],
        "n_undefined": 0,
        "source_tokens": 3,
        "pivot_tokens": 2,
        "ci_low": 1.2,
        "ci_high": 1.8,
        "n_bootstrap": 1000,
        "seed": 0,
    }
    summary = summarise_tpp(raw, ci=0.95)
    for key in (
        "ci_low",
        "ci_high",
        "n_undefined",
        "n_bootstrap",
        "seed",
        "source_tokens",
        "pivot_tokens",
    ):
        assert summary[key] == raw[key]
    assert summary["ci"] == 0.95
    # summarise_metric's own contract still holds: value/n/unit passed through, and the
    # per_pair distribution is reduced to distribution/mean/std rather than kept whole.
    assert summary["value"] == 1.5
    assert summary["n"] == 2
    assert summary["unit"] == "tokens/proposition ratio"
    assert summary["distribution"] == "per_pair"
    assert "per_pair" not in summary


def test_summarise_tpp_key_set_is_exactly_what_results_json_stores() -> None:
    """The stored key set is frozen: `results.json`'s shape must not drift (CLAUDE.md §9)."""
    raw = {
        "value": 1.0,
        "n": 1,
        "unit": "tokens/proposition ratio",
        "per_pair": [1.0],
        "n_undefined": 0,
        "source_tokens": 1,
        "pivot_tokens": 1,
        "ci_low": 1.0,
        "ci_high": 1.0,
        "n_bootstrap": 10,
        "seed": 0,
    }
    assert set(summarise_tpp(raw, ci=0.95)) == {
        "value",
        "n",
        "unit",
        "distribution",
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "n_undefined",
        "n_bootstrap",
        "seed",
        "source_tokens",
        "pivot_tokens",
        "ci",
    }


# --- arm loading ------------------------------------------------------------------


def _fake_arm(name: str, source_id: str, family: str) -> LoadedTokenizer:
    """A `LoadedTokenizer` whose `encode` is never called; only its provenance is read."""
    return LoadedTokenizer(
        name=name,
        source_id=source_id,
        vocab_size=32000,
        _encode=lambda text: [len(text)],
        family=family,
        attempted=(source_id,),
    )


def test_load_arms_returns_what_loaded_and_why_the_rest_did_not(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One unavailable arm must not cost a run that measures a dozen: it is recorded, with
    a WARNING, and every other arm still loads."""
    import sanskrit_tok.experiment as experiment

    def load(name: str) -> LoadedTokenizer:
        if name == "T4_bpe_split_32k":
            raise TokenizerUnavailable(f"{name}: trained tokenizer file not found")
        return _fake_arm(name, f"{name}.json", name.split("_", 1)[0])

    monkeypatch.setattr(experiment, "load_tokenizer", load)
    with caplog.at_level("WARNING", logger="sanskrit_tok.experiment"):
        loaded, unavailable = load_arms(["T1_bpe_raw_32k", "T4_bpe_split_32k", "T1_bpe_raw_32k"])
    assert set(loaded) == {"T1_bpe_raw_32k"}
    assert "trained tokenizer file not found" in unavailable["T4_bpe_split_32k"]
    assert "T4_bpe_split_32k" in " ".join(record.getMessage() for record in caplog.records)


def test_load_arms_loads_each_name_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A name repeated across an experiment's arm and pivot lists must not be downloaded
    (or read off disk) twice."""
    import sanskrit_tok.experiment as experiment

    calls: list[str] = []

    def load(name: str) -> LoadedTokenizer:
        calls.append(name)
        return _fake_arm(name, f"{name}.json", "T1")

    monkeypatch.setattr(experiment, "load_tokenizer", load)
    load_arms(["T1_bpe_raw_32k", "T1_bpe_raw_32k"])
    assert calls == ["T1_bpe_raw_32k"]


def test_load_arms_does_not_swallow_a_defect(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TokenizerUnavailable` means a missing artifact; anything else is a bug here and
    must propagate rather than be reported as an unavailable arm."""
    import sanskrit_tok.experiment as experiment

    def load(name: str) -> LoadedTokenizer:
        raise TypeError("signature changed upstream")

    monkeypatch.setattr(experiment, "load_tokenizer", load)
    with pytest.raises(TypeError):
        load_arms(["T1_bpe_raw_32k"])


# --- tokenizer_sources: provenance of each loaded arm -----------------------------


def test_tokenizer_sources_hashes_file_backed_arms(tmp_path: Path) -> None:
    """T1/T2/T4/E1 arms live in gitignored `outputs/` and Unigram training is not
    bit-reproducible, so the sha256 of the exact `tokenizer.json` is the only tie between
    a number in `results.json` and the artifact behind it."""
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b'{"model": "fake"}')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    sources = tokenizer_sources({"T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", str(path), "T1")})
    assert sources["T1_bpe_raw_32k"]["sha256"] == expected
    assert sources["T1_bpe_raw_32k"]["source_id"] == str(path)
    assert sources["T1_bpe_raw_32k"]["vocab_size"] == 32000
    assert sources["T1_bpe_raw_32k"]["family"] == "T1"
    assert sources["T1_bpe_raw_32k"]["attempted"] == [str(path)]


def test_tokenizer_sources_hashes_t2_arms_too(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b'{"model": "unigram"}')
    sources = tokenizer_sources(
        {"T2_unigram_raw_64k": _fake_arm("T2_unigram_raw_64k", str(path), "T2")}
    )
    assert sources["T2_unigram_raw_64k"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_tokenizer_sources_omits_sha256_for_hub_backed_arms() -> None:
    """A T0/T3 `source_id` is a model id, not a path; there is no file here to hash."""
    sources = tokenizer_sources({"T0_o200k": _fake_arm("T0_o200k", "o200k_base", "T0")})
    assert "sha256" not in sources["T0_o200k"]


def test_tokenizer_sources_survives_an_unreadable_file(tmp_path: Path) -> None:
    """The numbers are already computed by then; a missing file must not abort the write."""
    sources = tokenizer_sources(
        {"T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", str(tmp_path / "gone.json"), "T1")}
    )
    assert "sha256" not in sources["T1_bpe_raw_32k"]
    assert sources["T1_bpe_raw_32k"]["source_id"].endswith("gone.json")


def test_tokenizer_sources_names_the_splitter_for_split_trained_arms(tmp_path: Path) -> None:
    """A T4 arm's training text is a splitter's output, so the model and revision that
    produced it are part of that arm's provenance (Experiment 03)."""
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b'{"model": "fake"}')
    sources = tokenizer_sources(
        {
            "T4_bpe_split_32k": _fake_arm("T4_bpe_split_32k", str(path), "T4"),
            "T1_bpe_raw_32k": _fake_arm("T1_bpe_raw_32k", str(path), "T1"),
        },
        "chronbmm/sanskrit5-multitask@c0d2ada5",
    )
    assert sources["T4_bpe_split_32k"]["splitter_source_id"] == (
        "chronbmm/sanskrit5-multitask@c0d2ada5"
    )
    # a raw arm was trained on unsplit text: the key would be a false provenance claim
    assert "splitter_source_id" not in sources["T1_bpe_raw_32k"]


def test_tokenizer_sources_without_a_splitter_id_adds_no_key(tmp_path: Path) -> None:
    """Experiment 02 has no split arms and passes nothing; nothing changes."""
    sources = tokenizer_sources(
        {"T4_bpe_split_32k": _fake_arm("T4_bpe_split_32k", str(tmp_path / "t.json"), "T4")}
    )
    assert "splitter_source_id" not in sources["T4_bpe_split_32k"]


def test_tokenizer_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    path.write_bytes(b"x" * (3 * (1 << 20) + 7))  # spans several read chunks
    assert tokenizer_file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# --- the leakage check ------------------------------------------------------------


def test_exclusion_check_counts_missing_hashes() -> None:
    sentences = ["रामः", "सीता", "लक्ष्मणः"]
    hashes = frozenset({sentence_hash("रामः"), sentence_hash("सीता")})
    assert exclusion_check_for(sentences, hashes) == {"n": 3, "n_missing": 1}


def test_exclusion_check_all_present_is_zero_missing() -> None:
    sentences = ["रामः", "सीता"]
    hashes = frozenset({sentence_hash(text) for text in sentences})
    assert exclusion_check_for(sentences, hashes) == {"n": 2, "n_missing": 0}


def test_exclusion_check_none_present_is_all_missing() -> None:
    assert exclusion_check_for(["रामः", "सीता"], frozenset()) == {"n": 2, "n_missing": 2}


def test_exclusion_check_uses_the_english_hash_for_the_english_side() -> None:
    """`exclusion_check_en` verifies the list that kept the E1 control arms away from the
    evaluation text, so it must hash the way that list was built (`sentence_hash_en`)."""
    sentences = ["Rama goes", "The verse रामः गच्छति opens the chapter"]
    hashes = frozenset({sentence_hash_en(text) for text in sentences})
    assert exclusion_check_for(sentences, hashes, sentence_hash_en) == {"n": 2, "n_missing": 0}
    # the Sanskrit hash transliterates first, so the Devanagari-bearing sentence misses
    assert exclusion_check_for(sentences, hashes) == {"n": 2, "n_missing": 1}


# --- the unavailable-arm caption --------------------------------------------------


def test_unavailable_caption_lists_every_omitted_arm() -> None:
    caption = unavailable_caption(
        {"T3_indicsuper": "T3_indicsuper: no candidate tokenizer could be loaded. Tried:\n  a\n  b"}
    )
    assert "T3_indicsuper" in caption
    assert "omitted" in caption
    # the redundant "T3_indicsuper: " prefix and the per-candidate "Tried:" list are
    # trimmed, so the per-candidate detail does not leak into the one-line caption.
    assert "Tried" not in caption


def test_unavailable_caption_names_every_arm_in_sorted_order() -> None:
    caption = unavailable_caption({"T4_z": "T4_z: gone", "T1_a": "T1_a: gone"})
    assert caption.index("T1_a") < caption.index("T4_z")


def test_unavailable_caption_is_empty_when_nothing_is_unavailable() -> None:
    assert unavailable_caption({}) == ""


# --------------------------------------------------------------- text invariants (exp03)


def test_text_invariants_reports_a_preserved_multiset() -> None:
    from sanskrit_tok.experiment import text_invariants

    stats = text_invariants(["tadapi karoti."], ["tad api karoti."])

    assert stats["nonletter_multiset_preserved"] is True
    assert stats["nonletter_chars_raw"] == 1
    assert stats["nonletter_chars_out"] == 1
    assert stats["letter_chars_raw"] == len("tadapikaroti")
    assert stats["letter_chars_out"] == len("tadapikaroti")
    assert stats["letter_retention"] == pytest.approx(1.0)


def test_text_invariants_detects_a_dropped_punctuation_mark() -> None:
    """The regression the check exists for: a danda deleted with the word it was fused to."""
    from sanskrit_tok.experiment import text_invariants

    stats = text_invariants(["tadapi karoti."], ["tad api karoti"])

    assert stats["nonletter_multiset_preserved"] is False
    assert stats["nonletter_chars_raw"] == 1
    assert stats["nonletter_chars_out"] == 0


def test_text_invariants_is_pooled_and_ignores_whitespace() -> None:
    from sanskrit_tok.experiment import text_invariants

    stats = text_invariants(["a.", "b,"], ["a  .", "b\t,"])

    assert stats["nonletter_multiset_preserved"] is True
    assert stats["nonletter_chars_raw"] == 2
    assert stats["letter_chars_raw"] == 2


def test_text_invariants_letter_retention_tracks_the_splitters_rewriting() -> None:
    """Letters legitimately change (restored visargas, normalised anusvāra), so they get a
    ratio rather than an equality; non-letters get the equality."""
    from sanskrit_tok.experiment import text_invariants

    stats = text_invariants(["prARina Agatya"], ["prARinaH Agatya"])

    assert stats["nonletter_multiset_preserved"] is True
    assert stats["letter_retention"] == pytest.approx(14 / 13)


def test_text_invariants_rejects_mismatched_lengths() -> None:
    from sanskrit_tok.experiment import text_invariants

    with pytest.raises(ValueError, match="same length"):
        text_invariants(["a"], ["a", "b"])


def test_write_results_does_not_leave_a_truncated_file_when_sanitising_fails(
    tmp_path: Path,
) -> None:
    """Sanitising happens before the file is opened, so a failure leaves the previous
    results.json intact instead of replacing it with an empty one."""

    class Unserialisable:
        def __repr__(self) -> str:  # pragma: no cover - only for the failure message
            return "<boom>"

    from sanskrit_tok.experiment import write_results

    (tmp_path / "results.json").write_text('{"previous": 1}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        write_results({"bad": Unserialisable()}, tmp_path)

    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == {
        "previous": 1
    }
