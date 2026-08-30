"""CI build and release-budget wiring tests."""

from pathlib import Path

import yaml

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
        if step.get("uses") == "astral-sh/setup-uv@v8.3.2"
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
