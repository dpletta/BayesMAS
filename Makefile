# BayesMAS — every target runs through `uv run`.
.DEFAULT_GOAL := help
.PHONY: help setup sync test lint fmt typecheck check clean
UV ?= uv

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## pin Python 3.12 and install the locked environment
	$(UV) python pin 3.12
	$(UV) sync --all-groups

sync: ## reproduce the locked environment exactly
	$(UV) sync --locked --all-groups

test: ## unit + integration tests
	$(UV) run --locked pytest

lint: ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## ruff autofix + format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## mypy --strict over src/
	$(UV) run mypy

check: lint typecheck test ## CI entry point

clean: ## remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
