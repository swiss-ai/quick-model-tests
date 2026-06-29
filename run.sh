#!/usr/bin/env bash
# Bootstrap + run quick-model-tests against a hosted API.
#
# By default this prints the capability report (the human-readable view). Pass
# --test-framework to run the pytest pass/fail gate instead (for CI gating).
#
# Remote (the common case):
#   export CSCS_SERVING_API=...   # bearer token
#   curl -fsSL https://raw.githubusercontent.com/swiss-ai/quick-model-tests/main/run.sh | bash
#
# Scoped:
#   curl -fsSL .../run.sh | bash -s -- --suite tools,streaming --model <id>
#
# Inside a checkout (skip the git install, use the local tree):
#   bash run.sh --local --suite core
#
# Flags: --suite a,b  --model ID  --base-url URL  --junit PATH  --local
# Config via env: QMT_API_BASE, QMT_API_KEY|CSCS_SERVING_API,
#                 QMT_MODEL, QMT_TIMEOUT.
# See SPEC.md section 3.
set -euo pipefail

REPO="${QMT_REPO:-https://github.com/swiss-ai/quick-model-tests}"
REF="${QMT_REF:-main}"
LOCAL=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --local) LOCAL=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -z "${QMT_API_KEY:-}" ] && [ -z "${CSCS_SERVING_API:-}" ]; then
  echo "error: set CSCS_SERVING_API (or QMT_API_KEY) to your bearer token" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
VENV="$(mktemp -d)/venv"
cleanup() { rm -rf "$(dirname "$VENV")"; }
trap cleanup EXIT

echo "Setting up test environment..."
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install --quiet --upgrade pip

if [ "$LOCAL" -eq 1 ] || [ -f "./pyproject.toml" ]; then
  pip install --quiet -e ".[dev]"
else
  pip install --quiet "quick_model_tests[dev] @ git+${REPO}@${REF}"
fi

echo "Running quick-model-tests..."
# Default to the capability report (the --detail view). Pass --test-framework
# through ARGS to run the pytest pass/fail gate instead (CI gating).
set +e
quick-model-tests --detail "${ARGS[@]}"
status=$?
set -e
exit "$status"
