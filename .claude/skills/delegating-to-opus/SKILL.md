---
name: delegating-to-opus
description: Use when a task involves writing or editing code, tests, scripts, configs, or boilerplate docs, running tests or linters, fixing ruff/mypy errors, or any other implementation work — before the orchestrating session writes a single line itself.
---

# Delegating to Opus

## Overview

The top-level session is the **orchestrator**: it plans, makes research and
engineering decisions, briefs, reviews, and verifies. It does not implement.
Every implementation-level task is dispatched to an **Opus subagent**
(`Agent` tool, `model: "opus"`, `subagent_type: "general-purpose"`).

**Doing it yourself because it is small is a violation, not an optimisation.**
The orchestrator's context is reserved for judgement; implementation tokens
belong in a subagent.

## What to delegate vs. keep

| Delegate to Opus | Keep in the orchestrator |
|---|---|
| Writing / editing any `.py`, test, script, YAML, `pyproject.toml` | Research-design choices: which metric, which corpus, what counts as leakage |
| Writing tests, running `pytest`, `ruff`, `mypy`, fixing what they report | Reviewing the subagent's diff against CLAUDE.md constraints |
| Data download / conversion / dedup scripts | Deciding the *content* of a `decisions.md` entry (writing it is delegable) |
| Plotting scripts, `results.json` writers, README summaries from given numbers | Anything needing the user's judgement |
| Debugging a failing test with a clear repro | Final verification before reporting done |
| Refactors, renames, boilerplate, `__init__.py` scaffolding | Interpreting results and deciding next experiment |

If a task mixes both, split it: decide first, then delegate the doing.

## Dispatch recipe

A dispatch prompt IS these five parts, in this order:

1. **Goal** — one sentence, what exists when done.
2. **Files** — exact paths to create or edit; working directory.
3. **Constraints** — paste the relevant CLAUDE.md rules verbatim (SLP1 internal encoding, metric contract shape, exclusion-hash assertion, type hints, ruff + `mypy --strict`, tests < 60 s on CPU). Do not say "see CLAUDE.md"; the subagent may not read it.
4. **Definition of done** — the exact commands that must pass, e.g. `uv run pytest tests/test_fertility.py` and `uv run ruff check src`.
5. **Report format** — files touched, commands run with output, anything left undone.

Independent tasks go out as multiple `Agent` calls in one message. After the
subagent returns: read its diff, run the definition-of-done commands yourself
or read their output verbatim, and only then report to the user.

**REQUIRED SUB-SKILL:** for multi-task plans use
superpowers:subagent-driven-development; for 2+ independent tasks use
superpowers:dispatching-parallel-agents.

## Rationalisations (all mean: dispatch to Opus)

| Excuse | Reality |
|---|---|
| "It's a small, fully specified function" | Small and specified is the *easiest* thing to delegate. Brief it in 30 seconds. |
| "The user is in a hurry" | One Opus dispatch is ~1 minute. Your own implementation plus self-review is not faster and burns orchestrator context. |
| "Spawning adds latency" | Latency is the subagent's cost. Context is yours. Context is the scarce one. |
| "I'd have to verify its output anyway" | Yes. Verifying is your job. Writing is not. |
| "I've already started writing it" | Stop. Hand what you have to the subagent as part of the brief. |
| "The brief would be longer than the code" | Then the brief is the work. Write it. |
| "Opus might get the Sanskrit details wrong" | Put the details in Constraints. Review for them. |

## Red flags — STOP and dispatch

- You are about to call `Write`/`Edit` on a `.py`, test, or config file.
- You are running `pytest`/`ruff`/`mypy` to fix errors yourself.
- You are thinking "just this one file".
- The word "quick" or "tiny" is in your reasoning.

**All of these mean: write the five-part brief and call `Agent` with `model: "opus"`.**

## Common mistakes

- **Omitting `model: "opus"`** — the subagent inherits the orchestrator's model. Always set it.
- **Brief says "follow CLAUDE.md"** — paste the rules; the subagent starts with an empty context.
- **No definition of done** — the subagent stops at "it compiles". Give it the commands.
- **Trusting the report** — read the diff. Run the commands. The report is a claim.
