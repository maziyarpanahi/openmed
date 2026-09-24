"""Tests for the v3.1 agent deployment-assurance pack."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from openmed.compliance.assurance.agent_workflows import (
    UNVALIDATED_LIMITATION,
    VALIDATED_CLAIM,
    AgentAssurancePackError,
    ValidationReference,
    build_agent_assurance_pack,
    build_agent_workflow_assurance_pack,
)
from openmed.eval.suites.agent_release import UNAUTHORIZED_ACTION_ESCAPE

_FIXTURE_PATH = Path("tests/fixtures/eval/agent_release.py")
_SOURCE_REVISION = "a" * 40


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "agent_release_fixtures_for_assurance", _FIXTURE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack(*, passing: bool = True):
    fixtures = _fixtures()
    report = (
        fixtures.passing_report() if passing else fixtures.critical_failure_report()
    )
    return build_agent_assurance_pack(
        source_revision=_SOURCE_REVISION,
        artifact_digest=fixtures.digest("deployment-artifact"),
        gate_report=report,
    )


def test_pack_is_exactly_bound_and_byte_deterministic() -> None:
    first = _pack()
    second = _pack()

    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert first.pack_digest == second.pack_digest
    assert json.loads(first.to_json()) == first.to_dict()
    assert first.source_revision == _SOURCE_REVISION
    assert first.to_dict()["artifact_digest"].startswith("sha256:")
    assert first.to_dict()["gate_report_digest"].startswith("sha256:")


def test_each_claim_has_passing_gate_evidence_or_is_a_labeled_limitation() -> None:
    pack = _pack()
    payload = pack.to_dict()
    validations = {item["metric"]: item for item in payload["validation_references"]}

    for claim in payload["claims"]:
        if claim["classification"] == VALIDATED_CLAIM:
            reference = validations[claim["release_gate"]]
            assert reference["passed"] is True
            assert claim["evidence_digest"] == reference["evidence_digest"]
            assert claim["gate_documentation"].endswith(
                "v3.1-agent-gates.md#pre-registered-thresholds"
            )
        else:
            assert claim["classification"] == UNVALIDATED_LIMITATION
            assert claim["release_gate"] is None
            assert claim["evidence_digest"] is None


def test_failed_gate_becomes_limitation_instead_of_an_assurance_claim() -> None:
    payload = _pack(passing=False).to_dict()
    authority = next(
        item for item in payload["claims"] if item["claim_id"] == "authority_boundary"
    )
    validation = next(
        item
        for item in payload["validation_references"]
        if item["metric"] == UNAUTHORIZED_ACTION_ESCAPE
    )

    assert payload["release_decision"] == "NOT_READY"
    assert authority["classification"] == UNVALIDATED_LIMITATION
    assert authority["release_gate"] is None
    assert "not validated" in authority["statement"]
    assert validation["passed"] is False


def test_pack_contains_required_operator_and_review_sections() -> None:
    payload = _pack().to_dict()

    assert payload["deployment_profile"]["deployment_mode"] == "local_first"
    assert payload["approval_semantics"]["states"]
    assert payload["incident_evidence_checklist"]
    assert payload["threat_and_limitation_summary"]
    assert payload["change_log"] == [
        {
            "change": "initial_v3.1_deployment_and_review_contract",
            "pack_version": "3.1",
        }
    ]
    assert payload["clinician_review_protocol"]
    assert all(
        row["classification"] == UNVALIDATED_LIMITATION
        for row in payload["jurisdiction_support_matrix"]
    )


def test_markdown_states_clinical_and_certification_boundaries() -> None:
    markdown = _pack().to_markdown()

    assert "does not provide certification" in markdown
    assert "autonomous clinical" in markdown
    assert "site-specific clinical governance" in markdown
    assert f"Source revision: `{_SOURCE_REVISION}`" in markdown
    assert "Clinician-review protocol" in markdown
    assert "decline, abstention, or escalation" in markdown


@pytest.mark.parametrize(
    ("source_revision", "artifact_digest", "message"),
    [
        ("patient-name", "sha256:" + ("a" * 64), "source_revision"),
        (_SOURCE_REVISION, "secret-token", "artifact_digest"),
    ],
)
def test_pack_rejects_non_digest_inputs(
    source_revision: str,
    artifact_digest: str,
    message: str,
) -> None:
    with pytest.raises(AgentAssurancePackError, match=message):
        build_agent_assurance_pack(
            source_revision=source_revision,
            artifact_digest=artifact_digest,
            gate_report=_fixtures().passing_report(),
        )


def test_workflow_builder_alias_has_the_same_contract() -> None:
    fixtures = _fixtures()
    report = fixtures.passing_report()
    kwargs = {
        "source_revision": _SOURCE_REVISION,
        "artifact_digest": fixtures.digest("deployment-artifact"),
        "gate_report": report,
    }

    assert build_agent_workflow_assurance_pack(**kwargs).to_json() == (
        build_agent_assurance_pack(**kwargs).to_json()
    )


def test_passing_validation_reference_requires_evidence_digest() -> None:
    with pytest.raises(AgentAssurancePackError, match="required_for_passed_gate"):
        ValidationReference(
            metric=UNAUTHORIZED_ACTION_ESCAPE,
            passed=True,
            reason_code="threshold_satisfied",
            evidence_digest=None,
            limitations=("synthetic_policy_matrix",),
        )


def test_generated_pack_has_no_raw_workflow_content_fields() -> None:
    encoded = _pack().to_json()

    for forbidden in (
        "patient",
        "prompt",
        "tool_arguments",
        "reviewer_identity",
        "credential",
    ):
        assert forbidden not in encoded.lower()
