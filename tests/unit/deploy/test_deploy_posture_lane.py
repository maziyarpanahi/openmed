"""Offline render lane: shipped Helm and Compose files against the contract.

The lane renders the chart with ``helm template`` and parses the Compose
bundle, evaluates every workload against the production posture contract and
requires each target to produce exactly its recorded findings.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from openmed.deploy import posture as dp  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / "deploy" / "posture" / "lane.yaml"
FIXTURES = ROOT / "deploy" / "posture" / "helm"
HELM = shutil.which("helm")
needs_helm = pytest.mark.skipif(
    HELM is None, reason="helm is not installed; the Helm render lane needs it"
)


def _lane() -> dp.PostureLane:
    return dp.load_lane(LANE)


@pytest.fixture(scope="module")
def full_report() -> dp.LaneReport:
    if HELM is None:
        pytest.skip("helm is not installed; the Helm render lane needs it")
    return dp.run_lane(_lane(), helm=HELM)


def _describe(result: dp.TargetResult) -> str:
    return json.dumps(
        {
            "unexpected": [f.to_dict() for f in result.unexpected],
            "stale": [e.to_dict() for e in result.stale],
        },
        indent=2,
    )


def test_compose_lane_targets_match_their_recorded_findings() -> None:
    report = dp.run_lane(_lane(), platforms=[dp.Platform.COMPOSE])

    assert {result.target.target_id for result in report.results} >= {
        "compose-bundle",
        "compose-invalid-dev-profile",
    }
    for result in report.results:
        assert result.passed, f"{result.target.target_id}: {_describe(result)}"


@needs_helm
def test_helm_lane_targets_match_their_recorded_findings(
    full_report: dp.LaneReport,
) -> None:
    results = [
        result
        for result in full_report.results
        if result.target.platform is dp.Platform.HELM
    ]

    ids = {result.target.target_id for result in results}
    assert {"helm-default", "helm-ci", "helm-journey", "helm-production"} <= ids
    for result in results:
        assert result.passed, f"{result.target.target_id}: {_describe(result)}"


@needs_helm
def test_production_example_passes_and_negative_controls_fail_closed(
    full_report: dp.LaneReport,
) -> None:
    results = {result.target.target_id: result for result in full_report.results}

    assert results["helm-production"].evaluation.gated == ()
    for result in full_report.results:
        if result.target.purpose is dp.TargetPurpose.NEGATIVE_CONTROL:
            assert result.evaluation.gated, result.target.target_id
    assert full_report.exit_code == dp.EXIT_LANE_PASSED


@needs_helm
def test_lane_report_is_value_free_and_deterministic(
    full_report: dp.LaneReport,
) -> None:
    first = full_report.to_json()
    second = dp.run_lane(_lane(), helm=HELM).to_json()
    text = full_report.to_text()

    assert first == second
    for forbidden in (
        "openmed-service-auth",
        "openmed-journey-database",
        "openmed-hf-token",
        "api.example.test",
        "gateway.example.test",
        "clinic-ui.example",
        "openmed_journey_api_only",
        "postgresql://",
        "/var/",
        "/root/",
        str(ROOT),
    ):
        assert forbidden not in first
        assert forbidden not in text


def test_lane_rules_hold_for_every_target() -> None:
    lane = _lane()
    for target in lane.targets:
        if target.purpose is dp.TargetPurpose.NEGATIVE_CONTROL:
            assert any(
                entry.disposition is dp.Disposition.NEGATIVE_CONTROL
                for entry in target.expectations
            )
        for entry in target.expectations:
            if entry.disposition is dp.Disposition.KNOWN:
                assert entry.owner_pr is not None and entry.issue is not None
            if entry.disposition is dp.Disposition.FAIL_CLOSED:
                assert entry.check == "auth.enabled"


def test_every_negative_fixture_is_exercised_and_carries_no_secrets() -> None:
    lane = _lane()
    used = {path.name for target in lane.targets for path in target.values}
    for fixture in sorted(FIXTURES.glob("*.yaml")):
        assert fixture.name in used, fixture.name
        values = yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}
        for entry in values.get("extraEnv") or []:
            name = entry["name"]
            if name.endswith(("_API_KEYS", "_JWKS", "_DSN", "_PASSWORD", "_TOKEN")):
                assert "value" not in entry, f"{fixture.name}: {name}"
                assert "secretKeyRef" in entry["valueFrom"]


def test_ci_runs_the_render_lane() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    )
    steps = [
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    ]
    assert any("python -m openmed.deploy.posture" in run for run in steps)
    triggers = workflow[True]["pull_request"]["paths"]
    for path in (
        "deploy/posture/**",
        "deploy/openmed-compose.yaml",
        "openmed/deploy/posture.py",
        "openmed/service/security_profile.py",
    ):
        assert path in triggers
