"""Actual cohort, registry review and file projection with synthetic local inputs."""

from __future__ import annotations

import json
import socket
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.structured.cohort import (
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)
from openmed.structured.registry import (
    NAACCRFieldMapping,
    RegistryAssignmentAuthorization,
    RegistryCaseState,
    RegistryDefinition,
    RegistryExportAuthorization,
    RegistryFactBinding,
    RegistryFieldRule,
    RegistryWorkflowPolicy,
    adjudicate_registry_case,
    assign_registry_case,
    begin_registry_review,
    build_registry_export,
    complete_registry_review,
    materialize_registry_cases,
    parse_naaccr_dictionary,
    version_registry_definition,
    write_naaccr_xml,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("corrected", [False, True])
def test_synthetic_registry_to_xml_uses_existing_governance_and_stays_offline(
    tmp_path, monkeypatch, corrected
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("offline registry projection attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    cohort = save_cohort_definition(
        PhenotypeDefinition.load(
            Path(__file__).resolve().parents[1]
            / "fixtures/cohort/phenotypes/diabetes_on_metformin.json"
        )
    )
    subject = "patient_aaaaaaaaaaaaaaaa"
    membership = CohortMembership(
        patient_key=subject,
        state=MembershipState.MET,
        criteria=tuple(
            CriterionMembership(
                criterion_id=name,
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        f"evidence_{suffix * 16}",
                        f"fact_{suffix * 16}",
                        "window_aaaaaaaaaaaaaaaa",
                    ),
                ),
            )
            for name, suffix in (("has-diabetes", "a"), ("has-metformin", "b"))
        ),
    )
    execution = build_cohort_execution(
        cohort,
        source_snapshot=CohortSourceSnapshot(
            "snapshot_aaaaaaaaaaaaaaaa",
            "sha256:" + "1" * 64,
            "journey-snapshot-v1",
            ("synthetic",),
        ),
        vocabulary_digest="sha256:" + "2" * 64,
        policy_digest="sha256:" + "3" * 64,
        evaluator_version="synthetic-naaccr-1.0",
        memberships=(membership,),
    ).value
    assert execution is not None
    version = version_registry_definition(
        RegistryDefinition(
            registry_id="registry_aaaaaaaaaaaaaaaa",
            cohort_definition_version_id=cohort.version_id,
            cohort_definition_digest=cohort.definition_digest,
            fields=(
                RegistryFieldRule(
                    "condition", "condition", True, ("active", "corrected")
                ),
            ),
            workflow=RegistryWorkflowPolicy(
                "synthetic_naaccr",
                "1.0.0",
                "ownerscope_aaaaaaaaaaaaaaaa",
                privacy_policy_digest="sha256:" + "4" * 64,
                export_policy_digest="sha256:" + "5" * 64,
            ),
            definition_version="1.0.0",
        )
    )
    fact = ClinicalFact(
        "fact_aaaaaaaaaaaaaaaa",
        subject,
        "condition",
        {"code": "SYN01", "surface": "PRIVATE SYNTHETIC SURFACE"},
        "corrected" if corrected else "active",
        ("evidence_aaaaaaaaaaaaaaaa",),
        canonical_digest({"synthetic": "naaccr"}),
    )
    binding = RegistryFactBinding(fact, ("fact_cccccccccccccccc",) if corrected else ())
    materialized = materialize_registry_cases(
        version, execution, (binding,), created_at="2026-01-02T00:00:00Z"
    ).value
    assert materialized is not None
    case = materialized.cases[0]
    if corrected:
        assert case.state is RegistryCaseState.REVIEW_REQUIRED
        case = assign_registry_case(
            case,
            version,
            authorization=RegistryAssignmentAuthorization(
                "authorization_bbbbbbbbbbbbbbbb",
                version.definition.workflow.owner_scope_id,
                version.version_id,
                version.definition.workflow.digest,
                "synthetic_review_queue",
                True,
            ),
            assigned_at="2026-01-02T00:00:30Z",
        ).value
        assert case is not None and case.state is RegistryCaseState.ASSIGNED
        case = begin_registry_review(
            case, version, occurred_at="2026-01-02T00:01:00Z"
        ).value
        assert case is not None
        case = complete_registry_review(
            case,
            version,
            approved=True,
            occurred_at="2026-01-02T00:02:00Z",
            reason_code="synthetic_review",
        ).value
        assert (
            case is not None and case.state is RegistryCaseState.ADJUDICATION_REQUIRED
        )
        case = adjudicate_registry_case(
            case,
            version,
            approved=True,
            occurred_at="2026-01-02T00:03:00Z",
            reason_code="synthetic_adjudication",
            decision_digest=canonical_digest({"synthetic": "approved"}),
        ).value
        assert case is not None
    assert case.state is RegistryCaseState.EXPORT_READY
    before = case.to_json()
    policy = version.definition.workflow
    authorization = RegistryExportAuthorization(
        "authorization_aaaaaaaaaaaaaaaa",
        version.version_id,
        policy.privacy_policy_digest,
        policy.export_policy_digest,
        True,
    )
    envelope = build_registry_export(
        version, (case,), authorization=authorization, created_at="2026-01-02T00:04:00Z"
    ).value
    assert envelope is not None
    dictionary = parse_naaccr_dictionary(
        b"""<NaaccrDictionary xmlns="http://naaccr.org/naaccrxml" dictionaryUri="urn:synthetic:naaccr:integration" specificationVersion="1.8"><ItemDefs><ItemDef naaccrId="opaquePatient" naaccrNum="9500" length="64" parentXmlElement="Patient"/><ItemDef naaccrId="syntheticCondition" naaccrNum="9501" length="5" parentXmlElement="Tumor" dataType="alphanumeric"/></ItemDefs></NaaccrDictionary>"""
    )
    calls = []

    def resolver(looked_up_case, field_id, fact_id):
        calls.append((looked_up_case.case_id, field_id, fact_id))
        return fact

    report = write_naaccr_xml(
        (case,),
        version,
        authorization=authorization,
        export_envelope=envelope,
        dictionary=dictionary,
        field_map=(NAACCRFieldMapping("condition", "syntheticCondition", ("code",)),),
        resolver=resolver,
        output_path=tmp_path / "protected.xml",
        patient_key_item="opaquePatient",
        patient_key_secret=b"synthetic-integration-secret-32-bytes",
    )
    assert calls == [(case.case_id, "condition", fact.fact_id)]
    assert report.losses == () and report.file_written
    payload = (tmp_path / "protected.xml").read_bytes()
    root = ET.fromstring(payload)
    ns = {"n": "http://naaccr.org/naaccrxml"}
    assert root.find("n:Patient/n:Tumor/n:Item", ns).text == "SYN01"
    assert b"PRIVATE SYNTHETIC SURFACE" not in payload
    assert subject.encode() not in payload
    assert "PRIVATE SYNTHETIC SURFACE" not in json.dumps(report.to_dict())
    assert case.to_json() == before
    assert report.to_dict()["submitted"] is False
    assert report.to_dict()["edits_validated"] is False
