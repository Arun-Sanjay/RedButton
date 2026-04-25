# CLAUDE.md

This file is read by Claude Code at session start. It defines how to work in this repository.

## Project

**Name:** RedButton
**Status:** Setup phase. Spec exists at PROJECT.md; no production code yet.
**Spec location:** PROJECT.md (read end-to-end before any implementation begins).

## Working Style

This project is directed by a two-tier system:
- A high-level prompter (Claude.ai) writes the prompts.
- Claude Code (you) executes them.

The prompts are designed assuming you can handle long, multi-step tasks. Trust the prompts to specify what matters; ask only when something material is genuinely unclear.

When working in this repo:
- Tests are written alongside the code, not after.
- Commits happen at logical task boundaries with descriptive messages. No "Claude" trailers in commit messages.
- Pre-commit hook runs the test suite and linter; do not bypass.
- Push to origin after every committed task unless told otherwise.

## Environment

**Language:** Python (system Python 3.14 detected at scaffold time; ruff/pyproject target floor is 3.11).
**Package manager:** pip, with a project virtualenv at `.venv/`.
**Test runner:** pytest.
**Linter/formatter:** ruff (handles both lint and format).

Always activate the project's virtualenv before running any commands that depend on project deps:

    source .venv/bin/activate

The `.venv` is at the repo root and is gitignored.

## Hard Constraints

These are non-negotiable. Flag immediately if you see them creeping in:
- Do not commit secrets, .env files, or credentials.
- Do not add dependencies without explicit approval.
- Do not modify the spec document (PROJECT.md) without explicit approval.
- Do not skip tests to "get it working" — failing tests block commits.

## Subagent Roster

Empty. Subagents will be added when the project needs specialized roles. Defined in `.claude/agents/`.

## Skills

Empty. Skills will be added when task patterns repeat. Defined in `.claude/skills/`.

## Recent State

This section tracks current build state. Update at the end of each significant task.

**Last completed:** Initial scaffold — CLAUDE.md, `.claude/` tree, Python project skeleton (`pyproject.toml`, `src/redbutton/`, `tests/`), pre-commit hook.
**Next up:** Awaiting first task prompt. PROJECT.md should be read in full before any implementation begins.
**Known issues:** None at this time.

## When to Stop and Ask

Stop and ask the user before:
- Installing any package not already in the dependency list.
- Modifying CLAUDE.md or anything in `.claude/` that wasn't created in this task.
- Running anything destructive (rm -rf, force push, dropping databases).
- Making architectural decisions that weren't specified.

Do NOT stop and ask for:
- Standard implementation choices within a specified task.
- Style decisions consistent with existing code.
- Test coverage decisions for code you're writing.
