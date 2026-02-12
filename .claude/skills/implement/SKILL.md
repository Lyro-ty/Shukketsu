---
name: implement
description: Execute an implementation plan using subagent-driven development with parallel task batching, automated testing, and incremental commits
user-invocable: true
---

# Subagent-Driven Implementation

Execute an implementation plan document by spawning parallel subagents for independent tasks, with automated verification after each step.

## Workflow

1. **Read the plan** — Ask the user which plan document to execute (check `docs/plans/` if not specified). Parse all tasks, their dependencies, and batch groupings (parallel vs sequential).

2. **Create task list** — Use TaskCreate to register every task from the plan. Set up dependency chains with `addBlockedBy`/`addBlocks` so sequential ordering is enforced.

3. **Execute in batches** — For each batch of independent tasks:
   - Mark tasks as `in_progress`
   - Spawn one Task subagent per independent task using `subagent_type: "general-purpose"` with `run_in_background: true`
   - Each subagent receives: the task description, relevant file paths, coding conventions from CLAUDE.md, and instructions to write code + tests
   - Wait for all subagents in the batch to complete before moving to the next batch

4. **Verify after each task** — After each subagent completes:
   ```bash
   ruff check . --fix && ruff format .
   python3 -m pytest tests/unit/ -v
   ```
   If tests fail, debug and fix before proceeding.

5. **Commit each task** — After verification passes for a task, commit with a descriptive message:
   - Stage only the files changed by that task (`git add <specific-files>`)
   - Commit message format: `feat(<scope>): <description>` or `test(<scope>): <description>`
   - Include `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

6. **Final verification** — After all tasks are complete, run the full suite:
   ```bash
   ruff check . --fix && ruff format .
   python3 -m mypy .
   python3 -m pytest tests/unit/ -v
   ```

7. **Report results** — Summarize: total tasks completed, final test count, any issues encountered, and files changed.

## Rules

- Always use `python3 -m pytest` (never bare `pytest`)
- Stage specific files per commit, never `git add .`
- If a subagent fails, investigate before retrying — don't brute-force
- Respect task dependencies — never start a blocked task
- Each subagent should write both implementation code AND unit tests
- Follow all conventions in CLAUDE.md (type hints, Google docstrings, error handling, etc.)
