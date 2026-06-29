"""Run the deterministic suites and render the result as a capability table.

The pytest suites in `suites/` ARE the capability checks -- there is no separate
set of probes. Running quick-model-tests executes them (against the configured
model) and prints a per-check ✔/✗/⚠ table, exiting non-zero on any failure. It
can be scoped to one capability (`--capability tools`) or compared across models
(`--model A --model B`).

Status per check, derived from the pytest outcome:
    ✔ pass    the deterministic check passed
    ✗ fail    an assertion failed (a real gap in the model/endpoint)
    ⚠ broken  the test errored (e.g. the server returned an HTTP error mid-check)
    – skip    the check was skipped (capability not applicable)
"""

import contextlib
import io
import json
import os
from dataclasses import dataclass

from .config import Config

PASS, FAIL, BROKEN, SKIP = "pass", "fail", "broken", "skip"
_ICON = {PASS: "✔", FAIL: "✗", BROKEN: "⚠", SKIP: "–"}

_SUITES_DIR = os.path.join(os.path.dirname(__file__), "suites")


@dataclass
class Result:
    name: str  # check name, e.g. "tools_parallel"
    status: str
    detail: str = ""


def _check_name(nodeid: str) -> str:
    """ ".../tools.py::test_tools_parallel" -> "tools_parallel"."""
    func = nodeid.split("::")[-1]
    return func[len("test_") :] if func.startswith("test_") else func


def _crash_message(report) -> str:
    lr = getattr(report, "longrepr", None)
    crash = getattr(lr, "reprcrash", None)
    msg = (
        crash.message
        if crash is not None and getattr(crash, "message", None)
        else str(lr or "")
    )
    lines = [ln for ln in msg.strip().splitlines() if ln.strip()]
    return (lines[0] if lines else "")[:160]


def _skip_reason(report) -> str:
    lr = report.longrepr
    if isinstance(lr, tuple) and len(lr) == 3:
        return lr[2].replace("Skipped: ", "")[:160]
    return ""


class _Collector:
    """pytest plugin: record one Result per test (insertion-ordered)."""

    def __init__(self):
        self._by_name = {}

    def pytest_runtest_logreport(self, report):
        name = _check_name(report.nodeid)
        if report.when == "setup":
            if report.outcome == "failed":  # fixture error -> broken
                self._by_name[name] = Result(name, BROKEN, _crash_message(report))
            elif report.outcome == "skipped":
                self._by_name.setdefault(name, Result(name, SKIP, _skip_reason(report)))
        elif report.when == "call":
            if report.outcome == "passed":
                self._by_name[name] = Result(name, PASS, "")
            elif report.outcome == "failed":
                msg = _crash_message(report)
                low = msg.lower()
                status = (
                    FAIL
                    if low.startswith("assert") or "assertionerror" in low
                    else BROKEN
                )
                self._by_name[name] = Result(name, status, msg)
            elif report.outcome == "skipped":
                self._by_name.setdefault(name, Result(name, SKIP, _skip_reason(report)))

    @property
    def results(self):
        return list(self._by_name.values())


def run_checks(config: Config, capability: str = None, junit: str = None) -> list:
    """Run the suites against `config`'s model; return a list of Result."""
    os.environ["QMT_MODEL"] = config.model
    os.environ["QMT_API_BASE"] = config.api_base
    if config.api_key:
        os.environ["QMT_API_KEY"] = config.api_key

    args = [_SUITES_DIR, "-o", "python_files=*.py", "-p", "no:cacheprovider", "-q"]
    args += ["-m", capability if capability else "not perf"]
    if junit:
        args += [f"--junitxml={junit}"]

    import pytest  # required at runtime now (the default command runs the suites)

    collector = _Collector()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        pytest.main(args, plugins=[collector])
    return collector.results


def _exit_code(results) -> int:
    return 1 if any(r.status in (FAIL, BROKEN) for r in results) else 0


def report(
    config: Config, capability: str = None, as_json: bool = False, junit: str = None
) -> int:
    results = run_checks(config, capability, junit)
    if as_json:
        print(
            json.dumps(
                {
                    "model": config.model,
                    "api_base": config.api_base,
                    "checks": [r.__dict__ for r in results],
                },
                indent=2,
            )
        )
        return _exit_code(results)

    print(f"Capability checks for {config.model}")
    print(f"  endpoint: {config.api_base}\n")
    if not results:
        print("  (no checks ran -- unknown --capability, or no API key?)")
        return 1
    w = max(len(r.name) for r in results)
    for r in results:
        print(f"  {_ICON[r.status]} {r.name:<{w}}  {r.detail}")
    n_pass = sum(1 for r in results if r.status == PASS)
    n_fail = sum(1 for r in results if r.status in (FAIL, BROKEN))
    print(f"\n  {len(results)} checks; {n_pass} passed, {n_fail} failed/broken")
    return _exit_code(results)


def report_compare(
    configs: list, capability: str = None, as_json: bool = False, detail: bool = False
) -> int:
    """Compare >=2 models. Transposed table (checks down, models across) with one
    glyph per cell; columns M1/M2/... keep it narrow, full ids in a legend. With
    detail=True, failure reasons are listed as per-model footnotes."""
    runs = [(c.model, {r.name: r for r in run_checks(c, capability)}) for c in configs]
    names = []
    for _, res in runs:
        for n in res:
            if n not in names:
                names.append(n)

    failed = any(r.status in (FAIL, BROKEN) for _, res in runs for r in res.values())

    if as_json:
        print(
            json.dumps(
                {
                    "api_base": configs[0].api_base,
                    "models": [m for m, _ in runs],
                    "checks": {
                        n: {
                            m: (res[n].__dict__ if n in res else None)
                            for m, res in runs
                        }
                        for n in names
                    },
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    if not names:
        print("(no checks ran -- unknown --capability, or no API key?)")
        return 1
    cols = [f"M{i + 1}" for i in range(len(runs))]
    w = max(len(n) for n in names)
    print(f"Capability comparison ({configs[0].api_base})\n")
    print(f"| {'Check':<{w}} | " + " | ".join(cols) + " |")
    print(f"|{'-' * (w + 2)}|" + "----|" * len(cols))
    for n in names:
        cells = []
        for _, res in runs:
            r = res.get(n)
            cells.append(_ICON.get(r.status, " ") if r else " ")
        cells = [f"{c:<{len(col)}}" for c, col in zip(cells, cols)]
        print(f"| {n:<{w}} | " + " | ".join(cells) + " |")
    print("\nLegend: ✔ pass · ✗ fail · ⚠ broken · – skip")
    for i, (m, _) in enumerate(runs):
        print(f"M{i + 1} = {m}")

    if detail:
        print("\nFailure details (non-pass checks):")
        for i, (m, res) in enumerate(runs):
            fails = [res[n] for n in names if n in res and res[n].status != PASS]
            print(f"\nM{i + 1} = {m}")
            if not fails:
                print("  (all checks passed)")
            for r in fails:
                print(f"  {_ICON[r.status]} {r.name:<{w}}  {r.detail}")
    return 1 if failed else 0
