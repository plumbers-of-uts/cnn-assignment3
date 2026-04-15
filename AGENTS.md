# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Application code. Entry point: `src/main.py` exposing `main()`.
- Modules live under `src/<module>/` (e.g., `src/demo/`, `src/week1/`). With `pythonpath = ["src"]`, import modules as `import demo.main`.
- `tests/`: Pytest suite (e.g., `tests/test_main.py`, `tests/test_demo.py`).
- Key config: `pyproject.toml` (poethepoet tasks), `ruff.toml` (lint/format), `mypy.ini` (type checking), `.pre-commit-config.yaml` (hooks), `uv.lock` (deps).

## Build, Test, and Development Commands
- Install deps: `uv sync`.
- Run app: `uv run poe run` (alias for `python src/main.py`).
- Run a module: `uv run python src/demo/main.py`.
- Test: `uv run poe test`; Coverage: `uv run poe test-cov`.
- Lint/Format: `uv run poe lint` / `uv run poe format`.
- Type check: `uv run poe type-check`.
- All checks: `uv run poe all-checks`.
- Pre-commit: `uv run pre-commit install`, then `uv run poe pre-commit`.

## Agent Quality Gate
- After implementing or modifying functionality, follow `.agent/rules/python-quality-gate.md`.
- Required minimum checks before reporting completion: `uv run poe lint` and `uv run poe type-check`.
- If either check fails, do not mark work complete; fix introduced issues first and clearly report unrelated pre-existing failures.

## Coding Style & Naming Conventions
- Python 3.12 (>=3.12,<3.13); strict type hints enforced by `mypy`.
- Ruff manages linting and formatting; line length 100, double quotes, spaces for indent.
- Imports sorted by Ruff (isort); first‑party package root is `src`.
- Naming: modules/files `snake_case.py`; functions/variables `snake_case`; classes `PascalCase`.

## Testing Guidelines
- Framework: `pytest`; coverage measured with `--cov=src`.
- Place tests in `tests/` as `test_*.py` with functions `test_*`.
- Run tests via `uv run poe test` or `uv run pytest tests/`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat:`, `fix:`, `chore:`). Keep subjects <72 chars, imperative mood.
- Before pushing, run `uv run poe all-checks` and ensure tests pass.
- PRs should include purpose, a concise summary of changes, relevant logs/output, and linked issues (e.g., `Closes #123`). Request at least one review.

## Security & Configuration Tips
- Requirements: Python `>=3.12,<3.13` and `uv` (`pip install uv`).
- Don’t commit secrets or large data; follow `.gitignore`.
- Manage deps with `uv add <pkg>` / `uv remove <pkg>`; commit `pyproject.toml` and `uv.lock`.

<!-- OMA:START — managed by oh-my-agent. Do not edit this block manually. -->

# oh-my-agent

## Architecture

- **SSOT**: `.agents/` directory (do not modify directly)
- **Response language**: Follows `language` in `.agents/oma-config.yaml`
- **Skills**: `.agents/skills/` (domain specialists)
- **Workflows**: `.agents/workflows/` (multi-step orchestration)
- **Subagents**: Same-vendor native dispatch via Codex custom agents in `.codex/agents/{name}.toml`; cross-vendor fallback via `oma agent:spawn`

## Per-Agent Dispatch

1. Resolve `target_vendor_for_agent` from `.agents/oma-config.yaml`.
2. If `target_vendor_for_agent === current_runtime_vendor`, use the runtime's native subagent path.
3. If vendors differ, or native subagents are unavailable, use `oma agent:spawn` for that agent only.

## Workflows

Execute by naming the workflow in your prompt. Keywords are auto-detected via hooks.

| Workflow | File | Description |
|----------|------|-------------|
| orchestrate | `orchestrate.md` | Parallel subagents + Review Loop |
| work | `work.md` | Step-by-step with remediation loop |
| ultrawork | `ultrawork.md` | 5-Phase Gate Loop (11 reviews) |
| plan | `plan.md` | PM task breakdown |
| brainstorm | `brainstorm.md` | Design-first ideation |
| review | `review.md` | QA audit |
| debug | `debug.md` | Root cause + minimal fix |
| scm | `scm.md` | SCM + Git operations + Conventional Commits |

To execute: read and follow `.agents/workflows/{name}.md` step by step.

## Auto-Detection

Hooks: `UserPromptSubmit` (keyword detection), `PreToolUse`, `Stop` (persistent mode)
Keywords defined in `.agents/hooks/core/triggers.json` (multi-language).
Persistent workflows (orchestrate, ultrawork, work) block termination until complete.
Deactivate: say "workflow done".

## Rules

1. **Do not modify `.agents/` files** — SSOT protection
2. Workflows execute via keyword detection or explicit naming — never self-initiated
3. Response language follows `.agents/oma-config.yaml`

## Project Rules

Read the relevant file from `.agents/rules/` when working on matching code.

| Rule | File | Scope |
|------|------|-------|
| backend | `.agents/rules/backend.md` | on request |
| commit | `.agents/rules/commit.md` | on request |
| database | `.agents/rules/database.md` | **/*.{sql,prisma} |
| debug | `.agents/rules/debug.md` | on request |
| design | `.agents/rules/design.md` | on request |
| dev-workflow | `.agents/rules/dev-workflow.md` | on request |
| frontend | `.agents/rules/frontend.md` | **/*.{tsx,jsx,css,scss} |
| i18n-guide | `.agents/rules/i18n-guide.md` | always |
| infrastructure | `.agents/rules/infrastructure.md` | **/*.{tf,tfvars,hcl} |
| mobile | `.agents/rules/mobile.md` | **/*.{dart,swift,kt} |
| python-quality-gate | `.agents/rules/python-quality-gate.md` | on request |
| quality | `.agents/rules/quality.md` | on request |

<!-- OMA:END -->
