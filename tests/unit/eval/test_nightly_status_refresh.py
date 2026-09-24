"""Nightly status freshness and synthetic-control publication checks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.status.generate_nightly_control import build_control_report
from scripts.status.generate_status import render_status_page, validate_nightly_report

REVISION = "3405de48a2d1e9651353dada64761bdd3fb89568"
NOW = datetime(2026, 9, 24, 21, 30, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def control_report():
    """Build one real harness control over the committed golden corpus."""
    return build_control_report(REVISION, now=NOW)


def test_control_report_is_synthetic_hashed_and_never_claims_model_quality(
    control_report,
) -> None:
    report = control_report

    assert report.fixture_count > 0
    assert report.suite == "nightly-synthetic-control"
    assert report.model_name == "synthetic-empty-detector-v1"
    assert report.metrics["leakage"]["overall"] == 1.0
    assert report.metadata["synthetic"] is True
    assert report.metadata["source_rights"] == "OpenMed generated synthetic fixtures"
    assert report.metadata["license_id"] == "Apache-2.0"
    assert report.metadata["fixture_source"] == "openmed/eval/golden/fixtures"
    assert report.metadata["source_revision"] == REVISION
    assert report.metadata["reproducibility_hash"].startswith("sha256:")
    assert "not a clinical" in report.metadata["limitations"]
    validate_nightly_report(report, max_age=timedelta(hours=36), now=NOW)

    page = render_status_page(
        manifest_rows=[], baseline_store={}, nightly_report=report
    )
    assert "Latest Synthetic Harness Validation" in page
    assert report.generated_at in page
    assert "does not measure OpenMed model performance" in page


@pytest.mark.parametrize("age_hours", [37, 200])
def test_stale_control_report_is_rejected(age_hours: int, control_report) -> None:
    report = replace(
        control_report,
        generated_at=(NOW - timedelta(hours=age_hours))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    )
    with pytest.raises(ValueError, match="stale"):
        validate_nightly_report(report, max_age=timedelta(hours=36), now=NOW)


def test_malformed_or_non_synthetic_control_is_rejected(control_report) -> None:
    report = control_report
    for metadata in (
        {**report.metadata, "synthetic": False},
        {**report.metadata, "source_rights": "unknown"},
        {**report.metadata, "fixture_set_hash": "bad"},
        {**report.metadata, "model_revision": "unknown"},
        {**report.metadata, "config_revision": "unknown"},
        {**report.metadata, "limitations": ""},
        {**report.metadata, "reproducibility_hash": "bad"},
    ):
        with pytest.raises(ValueError):
            validate_nightly_report(
                replace(report, metadata=metadata),
                max_age=timedelta(hours=36),
                now=NOW,
            )


def test_pages_schedule_runs_control_renderer_and_public_deployment() -> None:
    workflow = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "schedule:" in workflow
    assert "scripts/status/generate_nightly_control.py" in workflow
    assert "scripts/status/generate_status.py" in workflow
    assert "--nightly-report docs/status/evidence/nightly-control.json" in workflow
    assert "actions/deploy-pages@" in workflow
