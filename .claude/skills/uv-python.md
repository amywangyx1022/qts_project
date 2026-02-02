# UV Python Execution

## Overview
Always use `uv` for Python project management and execution instead of `python` or `pip` directly.

## Rules

### Project Initialization
- If no `pyproject.toml` exists, run `uv init` before doing anything else
- Never use `python -m venv` or `virtualenv`

### Installing Dependencies
- Use `uv add <package>` to add dependencies (not `pip install`)
- Use `uv add --dev <package>` for dev dependencies
- Never use `pip install` or `pip`

### Running Code
- Use `uv run python script.py` instead of `python script.py`
- Use `uv run <command>` for any Python tools (e.g., `uv run pytest`, `uv run ruff`, `uv run mypy`)
- Never call `python` or `python3` directly

### Summary
| Instead of | Use |
|------------|-----|
| `python -m venv .venv` | `uv init` |
| `pip install package` | `uv add package` |
| `python script.py` | `uv run python script.py` |
| `python -m pytest` | `uv run pytest` |