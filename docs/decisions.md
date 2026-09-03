# Decision log

Append-only. Newest entries at the bottom. Format per CLAUDE.md §11.

## 2026-09-03 — Use SLP1 as internal encoding
Why: CharSS 2024 and ByT5-Sanskrit both report SLP1 outperforms Devanagari for neural models; it is lossless and ASCII.
Alternatives: IAST (diacritics complicate tokenizer training), Devanagari (byte-level fragmentation).
Reversible: yes, at cost of re-running tokenizer training.

## 2026-09-03 — Manage environment with uv installed via `python3 -m pip install --user uv`
Why: `uv` was not present on the machine. `pip install --user` bootstraps the tool itself into `~/.local/bin` (already on PATH) without touching the project environment, which stays fully `uv`-managed from `pyproject.toml` + `uv.lock`. Installed version: uv 0.12.9.
Alternatives: the astral install script `curl -LsSf https://astral.sh/uv/install.sh | sh` (equivalent result, extra network trust); Homebrew (adds a package manager dependency).
Reversible: yes, `pip uninstall uv` and reinstall by any other method; no project files depend on how uv was obtained.

## 2026-09-03 — Add `.DS_Store` to `.gitignore`
Why: macOS Finder writes `.DS_Store` into every directory it opens; several were picked up by `git add -A` during the initial scaffold. They carry no project information and would churn every commit.
Alternatives: a global `~/.gitignore_global` (does not travel with the repo, so other contributors would still commit them).
Reversible: yes, delete the line.
