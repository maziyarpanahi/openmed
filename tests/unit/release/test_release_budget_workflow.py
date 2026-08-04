"""CI wiring tests for release budgets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


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
