.PHONY: install install-dev run report test-framework format check lint test collect

# Optional overrides:  make test-framework MODEL=Qwen/Qwen3.5-27B SUITE=tools
MODEL ?=
SUITE ?=
_MODEL = $(if $(MODEL),--model $(MODEL),)
_SUITE = $(if $(SUITE),--suite $(SUITE),)

# Prefer uv when available (uv-created venvs have no `pip`); fall back to pip.
PIP := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip" || echo "pip")

install:        ## editable install with dev tools (pytest, ruff)
	$(PIP) install -e ".[dev]"

install-dev: install

# Default mode: capability report (pass several MODEL= for a comparison table).
run:
	quick-model-tests $(_MODEL)

report: run

# Pass/fail CI gate: the pytest suite.
test-framework:
	quick-model-tests --test-framework $(_SUITE) $(_MODEL)

# Auto-fix lint + format the tree.
format:
	ruff check --fix .
	ruff format .

# Lint + format gate (what CI runs on every PR). Modifies nothing.
check:
	ruff check .
	ruff format --check .

lint: check

# Offline: imports + test collection only, no API calls.
collect:
	pytest --collect-only -q

# Full suite (needs CSCS_SERVING_API / QMT_API_KEY set).
test:
	pytest
