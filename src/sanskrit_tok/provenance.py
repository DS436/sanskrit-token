"""Git provenance for `results.json`: which commit produced a number, and was it clean.

Every experiment records its git commit so a number can be traced back to the code that
produced it (CLAUDE.md §8). A commit alone is not enough: an experiment is routinely run
before the code that changed it is committed, so `results.json` can name a parent commit
while containing keys that commit's code could not have written (docs/decisions.md,
"Record `git_dirty` and file hashes in every results.json"). `git_dirty` makes that
visible instead of silent — `True` means the working tree had uncommitted changes when
the run started, so `git_commit` is a lower bound on the code that ran, not a description
of it.

Neither helper is fatal on failure. An experiment must still run from a tarball, a
worktree without git, or a machine with no `git` on `PATH`: `git_commit` records
`"unknown"` and `git_dirty` records `False`, both with a WARNING.
"""

import logging
import subprocess
from pathlib import Path

__all__ = ["git_commit", "git_dirty"]

logger = logging.getLogger(__name__)


def _run_git(args: list[str], root: Path) -> str | None:
    """`git <args>` in `root`, or `None` (logged at WARNING) if it cannot be run."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("could not run `git %s` in %s (%s)", " ".join(args), root, error)
        return None
    return completed.stdout


def git_commit(root: Path) -> str:
    """`git rev-parse HEAD` in `root`, or `"unknown"` (logged at WARNING) if that fails."""
    output = _run_git(["rev-parse", "HEAD"], root)
    if output is None:
        logger.warning("recording git_commit='unknown'")
        return "unknown"
    return output.strip() or "unknown"


def git_dirty(root: Path) -> bool:
    """Whether `root`'s working tree has uncommitted changes.

    `git status --porcelain` prints one line per modified, staged, deleted or untracked
    path and nothing at all for a clean tree, so a non-empty output means dirty. Returns
    `False` (logged at WARNING) when git cannot be run: an unknown state is recorded as
    clean rather than crashing a run, and the accompanying `git_commit` will be
    `"unknown"` in that case, which is the honest signal.
    """
    output = _run_git(["status", "--porcelain"], root)
    if output is None:
        logger.warning("recording git_dirty=False")
        return False
    return bool(output.strip())
