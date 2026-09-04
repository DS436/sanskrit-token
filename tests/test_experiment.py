"""Tests for `sanskrit_tok.experiment`, the helpers every experiment script shares.

These pin the behaviour three scripts previously each carried their own copy of
(`experiments/01_baseline_penalty/run.py`, `experiments/02_tpp_parallel/run.py`,
`experiments/02_tpp_parallel/train_tokenizers.py`): config-path resolution against the
repository root, YAML loading, git provenance, blank-index filtering, JSON sanitising and
the `results.json`/`config.yaml` writer. Several of them started life in
`tests/test_exp01.py` and `tests/test_exp02.py` and moved here with the code.

`provenance` runs `git` for real in a throwaway repository, for the reason
`tests/test_provenance.py` gives: the subprocess call is the thing under test.
"""

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sanskrit_tok.experiment import (
    load_config,
    provenance,
    repo_root,
    resolve_path,
    sanitize_json,
    select_aligned_indices,
    summarise_tpp,
    take_indices,
    write_results,
)

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
