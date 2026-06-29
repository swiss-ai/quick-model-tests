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

## Two modes

`quick-model-tests` defaults to a **capability report** (a quick diagnostic). The
**pass/fail gate** (pytest) is opt-in via `--test-framework` — and that's what
`run.sh` / CI use.

```bash
quick-model-tests                                 # capability report, default model
quick-model-tests --model swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools
quick-model-tests --model A --model B             # comparison table (capabilities x models)
quick-model-tests --json                          # machine-readable report

quick-model-tests --test-framework                # run the pass/fail suite (CI gate)
quick-model-tests --test-framework --suite tools  # scope the gate to a suite
```

Report status per row: `yes` works · `no` absent / server rejected the feature ·
`broken` offered but misbehaves (server bug) · `error` the probe couldn't run
(auth/connection). The gate *fails loudly* on broken paths; the report inventories
them.

Local development:

```bash
uv venv && source .venv/bin/activate   # or python -m venv .venv && source .venv/bin/activate
make install                      # editable install (auto-detects `uv pip` / `pip`)
make run MODEL=Qwen/Qwen3.5-27B   # capability report
make test-framework SUITE=core    # the pass/fail gate
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
