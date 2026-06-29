"""`quick-model-tests` console entrypoint.

Two modes:
  - default: print the capability report for the model(s) (a quick diagnostic;
    pass several --model for a comparison table).
  - `--test-framework`: run the pytest pass/fail suite (the CI gate). Also
    implied by `--suite` or `--junit`.

Flags are exported as APERTUS_* env vars so configuration resolution lives in one
place (config.Config), whether invoked via run.sh, the console script, or pytest
directly. See SPEC.md sections 3 and 8.
"""

import argparse
import os
import sys

_SUITES = [
    "core",
    "streaming",
    "tools",
    "multimodal",
    "multiturn",
    "reasoning",
    "robustness",
    "perf",
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="quick-model-tests")
    p.add_argument(
        "--test-framework",
        action="store_true",
        help="run the pytest pass/fail suite (CI gate) instead of the default "
        "capability report",
    )
    p.add_argument("--suite", help="comma-separated subset, e.g. tools,streaming")
    p.add_argument(
        "--model",
        action="append",
        help="model id; repeat (default capability mode) for a comparison table",
    )
    p.add_argument("--base-url")
    p.add_argument("--junit", metavar="PATH")
    p.add_argument(
        "--capabilities",
        action="store_true",
        help="force the capability report (the default; kept for explicitness)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="in capability mode, emit the report as JSON",
    )
    p.add_argument(
        "--detail",
        action="store_true",
        help="in a multi-model comparison, also list the failure reasons "
        "(per-model footnotes below the table)",
    )
    args, passthrough = p.parse_known_args(argv)

    models = args.model or []
    if len(models) == 1:
        os.environ["QMT_MODEL"] = models[0]
    if args.base_url:
        os.environ["QMT_API_BASE"] = args.base_url

    # The pass/fail gate runs only when asked (--test-framework), or implied by
    # --suite / --junit. Everything else defaults to the capability report.
    run_gate = (
        args.test_framework or args.suite or args.junit
    ) and not args.capabilities

    if not run_gate:
        import dataclasses

        from .capabilities import report, report_compare
        from .config import Config

        base = Config.from_env()
        if not base.api_key:
            p.error("no API key (set CSCS_SERVING_API or QMT_API_KEY)")
        if len(models) > 1:
            cfgs = [dataclasses.replace(base, model=m) for m in models]
            return report_compare(cfgs, as_json=args.json, detail=args.detail)
        return report(base, as_json=args.json)

    if len(models) > 1:
        p.error("multiple --model is only supported in the capability report")

    # Point pytest at the INSTALLED suites dir so `quick-model-tests` runs from any
    # cwd (the curl|bash flow), not only a source checkout. Markers and the
    # default filter are applied here because pyproject.toml is not on disk when
    # the package is pip-installed.
    import quick_model_tests

    suites_dir = os.path.join(os.path.dirname(quick_model_tests.__file__), "suites")
    # `-o python_files=*.py`: suite modules are named core.py/tools.py (not
    # test_*.py); pyproject sets this for the checkout flow, but it is absent when
    # pip-installed, so inject it here too.
    pytest_args = [suites_dir, "-ra", "-o", "python_files=*.py", *passthrough]

    if args.suite:
        wanted = [s.strip() for s in args.suite.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in _SUITES]
        if unknown:
            p.error(f"unknown suite(s): {', '.join(unknown)}")
        pytest_args += ["-m", " or ".join(wanted)]
    elif not any(a == "-m" or a.startswith("-m") for a in passthrough):
        pytest_args += ["-m", "not perf"]  # default: everything except perf
    if args.junit:
        pytest_args += [f"--junitxml={args.junit}"]

    import pytest  # imported here so --help works without pytest installed

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
