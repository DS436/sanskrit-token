"""Tests for `sanskrit_tok.provenance`, the shared git provenance for `results.json`.

These run `git` for real, in a throwaway repository built under `tmp_path`, because the
thing under test *is* the subprocess call and its output parsing: mocking `subprocess.run`
would pin this module's own string handling and prove nothing about what `git status
--porcelain` actually prints. The repository is created with `init`/`add`/`commit` and
never touches the developer's own repo or global config (`user.name`/`user.email` are set
locally, and `commit.gpgsign` is disabled, so the test passes on a machine whose global
git config would otherwise refuse or sign the commit).
"""

import subprocess
from pathlib import Path

import pytest

from sanskrit_tok.provenance import git_commit, git_dirty

#: A 40-character lowercase hex sha1, i.e. what `git rev-parse HEAD` prints.
_SHA_LENGTH = 40


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


def test_git_commit_is_the_head_sha(repo: Path) -> None:
    commit = git_commit(repo)
    assert len(commit) == _SHA_LENGTH
    assert commit == _git(repo, "rev-parse", "HEAD").strip()


def test_git_dirty_is_false_on_a_clean_tree(repo: Path) -> None:
    assert git_dirty(repo) is False


def test_git_dirty_is_true_after_modifying_a_tracked_file(repo: Path) -> None:
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    assert git_dirty(repo) is True


def test_git_dirty_is_true_for_an_untracked_file(repo: Path) -> None:
    """`--porcelain` lists untracked paths too, and they are part of what ran."""
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert git_dirty(repo) is True


def test_git_dirty_is_true_for_a_staged_but_uncommitted_change(repo: Path) -> None:
    (repo / "tracked.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    assert git_dirty(repo) is True


def test_git_dirty_returns_to_false_after_committing(repo: Path) -> None:
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "second")
    assert git_dirty(repo) is False


def test_outside_a_repository_commit_is_unknown_and_dirty_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run from a tarball must record its ignorance, not crash (CLAUDE.md §8).

    `git` walks upward until it finds a `.git`, so the directory is fenced off with
    `GIT_CEILING_DIRECTORIES` — otherwise a repository anywhere above the temp directory
    would make this pass or fail for the wrong reason. Both calls exit non-zero here
    (`CalledProcessError`), the other failure branch from the missing-`git` test below.
    """
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    assert git_commit(outside) == "unknown"
    assert git_dirty(outside) is False


def test_commit_and_dirty_warn_when_git_cannot_run(
    repo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`git` missing from `PATH` is recorded, at WARNING, not raised."""

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", boom)
    with caplog.at_level("WARNING"):
        assert git_commit(repo) == "unknown"
        assert git_dirty(repo) is False
    assert "no git here" in caplog.text
