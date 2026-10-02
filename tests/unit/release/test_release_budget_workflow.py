"""CI build and release-budget wiring tests."""

from pathlib import Path

import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"


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
