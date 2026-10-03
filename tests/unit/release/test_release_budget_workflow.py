"""CI build and release-budget wiring tests."""

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"
PYTEST_DIAGNOSTIC_WRAPPER = (
    "import faulthandler, pytest; "
    "diagnostics = type('CiSummaryDiagnostics', (), "
    "{'pytest_sessionfinish': lambda self, session, exitstatus: "
    "faulthandler.dump_traceback_later(120, repeat=True)})(); "
    "raise SystemExit(pytest.main(plugins=[diagnostics]))"
)


def _load_ci() -> dict[str, object]:
    return yaml.load(CI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_ci_pins_uv_and_uses_the_native_build_frontend():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    jobs = _load_ci()["jobs"]
    uv_steps = [
        step
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]

    assert uv_steps
    assert all(step.get("with", {}).get("version") == "0.11.28" for step in uv_steps)
    assert 'uv build --wheel --out-dir "$RUNNER_TEMP/openmed-wheel"' in workflow
    assert "uv run --with build python -m build" not in workflow
    assert "uv run --no-project --with build python -m build" not in workflow


def test_make_build_targets_use_uv_without_ephemeral_frontend_dependencies():
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert makefile.count("\t$(UV) build\n") == 2
    assert "--with build" not in makefile


def test_build_job_enforces_and_uploads_release_budgets():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    build_index = workflow.index("- name: Build package")
    size_index = workflow.index(
        "- name: Enforce wheel size budget and record language-extra footprints"
    )
    import_index = workflow.index("- name: Enforce core import budget")
    upload_index = workflow.index("- name: Upload build artifacts")

    assert build_index < size_index < import_index < upload_index
    assert "python scripts/release/check_size_budget.py" in workflow
    assert "python scripts/release/check_import_budget.py" in workflow
    assert "size-budget-report.json" in workflow
    assert "--gate-file" not in workflow


def test_sdk_compatibility_matches_advertised_python_and_os_support():
    jobs = _load_ci()["jobs"]
    lane = jobs["sdk-compatibility"]
    matrix = lane["strategy"]["matrix"]
    versions = ["3.10", "3.11", "3.12", "3.13"]
    assert matrix == {
        "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "python-version": versions,
    }
    assert lane["strategy"]["fail-fast"] == "false"
    assert int(lane["timeout-minutes"]) <= 30
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    prefix = "Programming Language :: Python :: "
    assert (
        sorted(
            c.removeprefix(prefix)
            for c in project["classifiers"]
            if c.startswith(prefix) and c.removeprefix(prefix).startswith("3.")
        )
        == versions
    )
    assert project["requires-python"] == ">=3.10"
    # CLI imports dependency-report parsing even in a minimal installation.
    # Development extras used to hide this missing Python 3.10 backport.
    assert "tomli>=2.0; python_version < '3.11'" in project["dependencies"]
    assert {
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    } <= set(project["classifiers"])
    docs = (ROOT / "docs/testing.md").read_text(encoding="utf-8")
    assert all(version in docs for version in versions)
    assert "12 jobs" in docs and "sdk-compatibility-gate" in docs


def test_installed_lane_is_mandatory_offline_and_covers_privacy_sentinels():
    jobs = _load_ci()["jobs"]
    lane = jobs["sdk-compatibility"]
    assert all(value == "1" for value in lane["env"].values())
    commands = "\n".join(step.get("run", "") for step in lane["steps"])
    for fragment in (
        "--frozen",
        "--no-emit-project",
        "--extra cli --extra service --extra mcp",
        "tests/unit/service/test_brief_surfaces.py",
        "tests/unit/ner/test_infer.py",
        "tests/unit/test_no_raw_text_logging.py",
        "--artifacts",
        "--constraints",
        "--report sdk-compatibility.json",
    ):
        assert fragment in commands
    assert "sdk-compatibility-gate" in jobs["build"]["needs"]
    assert "needs.sdk-compatibility-gate.result == 'success'" in jobs["build"]["if"]
    gate = jobs["sdk-compatibility-gate"]
    assert gate["needs"] == "sdk-compatibility" and gate["if"] == "always()"
    assert 'test "$MATRIX_RESULT" = success' in gate["steps"][0]["run"]


def _assert_full_suite_diagnostics(lane):
    assert lane.get("continue-on-error", "false") == "false"
    steps = lane["steps"]
    test_step = next(step for step in steps if step.get("name") == "Test with pytest")
    assert test_step["timeout-minutes"] == "45"
    assert test_step.get("continue-on-error", "false") == "false"
    assert "if" not in test_step
    # Exact argv retains default full-suite discovery and rejects selection,
    # early-stop flags, shell success overrides, and changes to coverage.
    assert shlex.split(test_step["run"]) == [
        "uv",
        "run",
        "--frozen",
        "python",
        "-c",
        PYTEST_DIAGNOSTIC_WRAPPER,
        "--cov=openmed",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--tb=line",
        "--show-capture=no",
        "--color=no",
        "--code-highlight=no",
        "-ra",
        "--junitxml=pytest-results.xml",
    ]
    artifact = next(
        step for step in steps if step.get("name") == "Upload test diagnostics"
    )
    assert artifact["if"] == "always()"
    assert artifact["uses"] == (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    )
    assert artifact["with"]["name"] == (
        "pytest-${{ matrix.os }}-py${{ matrix.python-version }}-${{ github.run_attempt }}"
    )
    assert artifact["with"]["path"].splitlines() == [
        "pytest-results.xml",
        "coverage.xml",
    ]
    assert artifact["with"]["if-no-files-found"] == "warn"
    assert steps.index(test_step) < steps.index(artifact)


def test_full_suite_diagnostics_preserve_discovery_and_failure_semantics():
    _assert_full_suite_diagnostics(_load_ci()["jobs"]["test"])
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = project["tool"]["pytest"]["ini_options"]
    assert options["testpaths"] == ["tests"]
    assert not options.get("addopts")


@pytest.mark.parametrize("exit_code", [0, 1, 2, 3, 4, 5])
def test_ci_pytest_diagnostic_wrapper_preserves_arguments_and_exit_status(
    monkeypatch, exit_code
):
    steps = _load_ci()["jobs"]["test"]["steps"]
    step = next(step for step in steps if step.get("name") == "Test with pytest")
    argv = shlex.split(step["run"])
    wrapper = argv[argv.index("-c") + 1]
    pytest_args = argv[argv.index("-c") + 2 :]
    observed = []

    def dump_traceback_later(timeout, *, repeat):
        observed.append(("diagnostics", timeout, repeat))

    def main(*, plugins):
        observed.append(("pytest", sys.argv[1:]))
        assert len(plugins) == 1
        # Pytest cancels ordinary watchdogs on test errors. Arm this one at
        # session finish so it survives into error-summary rendering.
        plugins[0].pytest_sessionfinish(session=None, exitstatus=exit_code)
        return exit_code

    monkeypatch.setitem(
        sys.modules,
        "faulthandler",
        SimpleNamespace(dump_traceback_later=dump_traceback_later),
    )
    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=main))
    monkeypatch.setattr(sys, "argv", ["-c", *pytest_args])
    with pytest.raises(SystemExit) as result:
        exec(compile(wrapper, "<ci-pytest-diagnostics>", "exec"), {})
    assert result.value.code == exit_code
    assert observed == [("pytest", pytest_args), ("diagnostics", 120, True)]


@pytest.mark.parametrize(
    "suffix",
    ["-k smoke", "-m 'not slow'", "-x", "--maxfail=1", "--ignore=tests", "|| true"],
)
def test_full_suite_diagnostics_contract_rejects_reduced_or_masked_runs(suffix):
    lane = _load_ci()["jobs"]["test"]
    step = next(
        step for step in lane["steps"] if step.get("name") == "Test with pytest"
    )
    step["run"] += " " + suffix
    with pytest.raises(AssertionError):
        _assert_full_suite_diagnostics(lane)


@pytest.mark.parametrize(
    ("step_name", "field", "value"),
    [
        ("Test with pytest", "timeout-minutes", "360"),
        ("Test with pytest", "continue-on-error", "true"),
        ("Test with pytest", "if", "false"),
        ("Upload test diagnostics", "if", "success()"),
        ("Upload test diagnostics", "uses", "actions/upload-artifact@v7"),
    ],
)
def test_full_suite_diagnostics_contract_rejects_unbounded_or_optional_steps(
    step_name, field, value
):
    lane = _load_ci()["jobs"]["test"]
    step = next(step for step in lane["steps"] if step.get("name") == step_name)
    step[field] = value
    with pytest.raises(AssertionError):
        _assert_full_suite_diagnostics(lane)


def test_full_suite_diagnostics_contract_rejects_job_failure_override():
    lane = _load_ci()["jobs"]["test"]
    lane["continue-on-error"] = "true"
    with pytest.raises(AssertionError):
        _assert_full_suite_diagnostics(lane)
