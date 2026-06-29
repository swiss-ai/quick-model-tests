# quick-model-tests

Lightweight tests that are 100% deterministic to prove the model (and system around it) will work in production. Designed to be prereq to evals and benchmarks.

## Quickstart

```bash
export CSCS_SERVING_API=...   # your bearer token
curl -fsSL https://raw.githubusercontent.com/swiss-ai/quick-model-tests/main/run.sh | bash
```

Scope to specific areas and pick a model:

```bash
curl -fsSL https://raw.githubusercontent.com/swiss-ai/quick-model-tests/main/run.sh | bash -s -- --model swiss-ai/Apertus-8B-Instruct-2509
```

Every check is **deterministic** (status codes, response schema, token counts,
substring / regex / closed-set membership) — there is intentionally no
LLM-as-judge. Semantic/quality evaluation belongs in LLM evals, not in a
functional pass/fail gate.

## Usage

One command. It runs the deterministic suites against a model and prints a
**✔/✗ capability table**, exiting non-zero on any failure.

```bash
quick-model-tests                              # run ALL checks, default model
quick-model-tests --model swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools
quick-model-tests --capability tools           # just one capability
quick-model-tests --model A --model B          # compare models (table)
quick-model-tests --model A --model B --detail # + per-model failure reasons
quick-model-tests --json                       # machine-readable
```

Status per check: `✔` pass · `✗` an assertion failed (a real gap) · `⚠` the
check errored (e.g. the server returned an HTTP error) · `–` skipped. The
capabilities are: `core` · `streaming` · `tools` · `multimodal` · `multiturn` ·
`reasoning` · `robustness` (· `perf`, opt-in).

Local development:

```bash
uv venv && source .venv/bin/activate   # or python -m venv .venv && source .venv/bin/activate
make install                      # editable install (auto-detects `uv pip` / `pip`)
make run MODEL=Qwen/Qwen3.5-27B   # run the checks
make format                       # ruff auto-fix + format
make check                        # ruff lint + format check (the CI PR gate)
```

Every PR runs `ruff check` + `ruff format --check` (see `.github/workflows/ci.yml`);
keep the tree clean with `make format` before pushing.

## Configuration

| Env var | Default |
|---------|---------|
| `QMT_API_BASE` | `https://api.swissai.svc.cscs.ch/v1` |
| `QMT_API_KEY` (or `CSCS_SERVING_API`) | — (required) |
| `QMT_MODEL` | `swiss-ai/Apertus-8B-Instruct-2509` |

## Suites

`core` · `streaming` · `tools` · `multimodal` · `multiturn` · `reasoning` ·
`robustness` · `perf` (opt-in).

Only `core` is implemented today; the rest are stubs. **See
[`SPEC.md`](./SPEC.md)** for the full specification, test catalog, open
questions to probe against the live API, and implementation milestones — it is
written so another engineer can take over the build from it directly.
