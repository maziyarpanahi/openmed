"""Actual passive exporters feed an offline labeling and approval boundary."""

from copy import deepcopy

import pytest

from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.clinical.context import (
    CERTAIN,
    NEGATED,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
)
from openmed.clinical.exporters import to_fhir
from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.interop.fhir import (
    FHIRWriteLabelPolicy,
    normalize_proposed_resource,
    validate_proposed_resource,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind,label,system,code",
    [
        ("Observation", "LAB", "LOINC", "synthetic-observation"),
        ("Condition", "CONDITION", "ICD10CM", "synthetic-condition"),
        ("AllergyIntolerance", "ALLERGEN", "RXNORM", "12345"),
    ],
)
@pytest.mark.parametrize("negated", [False, True])
def test_exporter_round_trip_to_protected_preview_boundary(
    kind, label, system, code, negated
):
    text = "synthetic finding"
    span = GroundedSpan(
        text=text,
        start=0,
        end=len(text),
        canonical_label=label,
        candidates=(
            Candidate(
                system=system,
                code=code,
                display=text,
                score=0.99,
                source="synthetic",
                matched_alias=text,
                match_kind="exact",
                vocab_version="synthetic-v1",
            ),
        ),
        assertion=ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=NEGATED if negated else "affirmed",
            experiencer=PATIENT_EXPERIENCER,
        ),
        metadata={"value": 42, "unit": "mg"},
    )
    exported = to_fhir(span, resource=kind, subject_reference="Patient/synthetic")
    assert exported is not None
    before = deepcopy(exported)
    assert validate_proposed_resource(exported)
    normalized = normalize_proposed_resource(exported)
    assert not validate_proposed_resource(normalized.resource)
    # A consumer may preview only after an empty validation result.
    preview_calls = []
    if not validate_proposed_resource(normalized.resource):
        preview_calls.append(normalized.resource)
    assert len(preview_calls) == 1
    allowed = {"meta", "status", "verificationStatus"}
    assert {k: v for k, v in before.items() if k not in allowed} == {
        k: v for k, v in normalized.resource.items() if k not in allowed
    }
    assert exported == before
    path = "status" if kind == "Observation" else "verificationStatus"
    if negated:
        assert normalized.resource[path] == before[path]


@pytest.mark.integration
@pytest.mark.parametrize(
    "role,allowed",
    [
        ("role:org.openmed/attester", True),
        ("role:org.openmed/workflow-operator", False),
    ],
)
def test_existing_single_use_approval_receipt_controls_status_exception(role, allowed):
    # Token verification/action binding is owned by the approval subsystem.
    policy = FHIRWriteLabelPolicy(
        attesting_roles=frozenset({"role:org.openmed/attester"})
    )
    payload = normalize_proposed_resource(
        {"resourceType": "Observation", "status": "final"}
    ).resource
    payload["status"] = (
        "final"  # Explicit clinician-supplied final state, never automatic.
    )
    key = b"synthetic-local-approval-key-32-bytes"
    action_digest = "sha256:" + "a" * 64
    token = ApprovalTokenSigner(key, clock=lambda: 10).issue(
        action_digest=action_digest,
        reviewer_role=role,
        expires_at=20,
        nonce="nonce_" + "1" * 32,
    )
    verifier = ApprovalTokenVerifier(
        key, InMemoryApprovalNonceStore(), clock=lambda: 10
    )
    authorization = verifier.consume_authorization(
        token, action_digest=action_digest, reviewer_role=role
    )
    findings = validate_proposed_resource(
        payload, policy=policy, approval_authorization=authorization
    )
    assert (not findings) == allowed
