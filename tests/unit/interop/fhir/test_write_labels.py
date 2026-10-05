"""Synthetic offline origin/status policy and diagnostic leakage controls."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.interop.fhir.write_labels import (
    FHIRWriteLabelPolicy,
    WriteLabelError,
    normalize_proposed_resource,
    validate_proposed_resource,
)

TYPES = ("Observation", "Condition", "AllergyIntolerance")
SYSTEMS = {
    "Condition": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
    "AllergyIntolerance": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
}
ROLE = "role:org.openmed/attester"
CANARY = "synthetic-private-Élodie-张三-555-0199"


def resource(kind, status=None):
    result = {
        "resourceType": kind,
        "code": {
            "coding": [{"system": "urn:synthetic", "code": "example"}],
            "text": CANARY,
        },
        "subject" if kind != "AllergyIntolerance" else "patient": {
            "reference": "Patient/synthetic"
        },
        "encounter": {"reference": "Encounter/synthetic"},
        "extension": [
            {
                "url": "urn:synthetic:source",
                "valueReference": {"reference": "DocumentReference/synthetic"},
            }
        ],
        "meta": {
            "versionId": "synthetic-version",
            "profile": ["urn:synthetic:profile"],
        },
    }
    if kind == "Observation":
        result["valueString"] = CANARY
    else:
        result["clinicalStatus"] = {
            "coding": [{"system": "urn:synthetic:clinical", "code": "active"}]
        }
    set_status(result, status or ("final" if kind == "Observation" else "confirmed"))
    return result


def set_status(payload, status):
    kind = payload["resourceType"]
    if kind == "Observation":
        payload["status"] = status
    else:
        payload["verificationStatus"] = {
            "coding": [{"system": SYSTEMS[kind], "code": status}]
        }


def receipt(role=ROLE):
    return ApprovalReceipt("sha256:" + "a" * 64, role, "sha256:" + "b" * 64, 10, 20)


@pytest.mark.parametrize("kind", TYPES)
def test_round_trip_changes_only_meta_and_status_and_is_idempotent(kind):
    original = resource(kind)
    before = deepcopy(original)
    result = normalize_proposed_resource(original)
    assert original == before
    assert validate_proposed_resource(original)  # Passive exporter cannot pass.
    assert not validate_proposed_resource(result.resource)
    exempt = {"meta", "status", "verificationStatus"}
    assert {k: v for k, v in original.items() if k not in exempt} == {
        k: v for k, v in result.resource.items() if k not in exempt
    }
    assert result.resource["meta"]["versionId"] == "synthetic-version"
    again = normalize_proposed_resource(json.loads(json.dumps(result.resource)))
    assert again.resource == result.resource
    assert again.findings == ()
    assert CANARY not in repr(result)
    assert CANARY not in json.dumps([f.to_dict() for f in result.findings])
    assert all(
        set(asdict(f)) == {"resource_type", "path", "code"} for f in result.findings
    )
    result.resource["code"]["text"] = "changed"
    assert original == before  # No shared nested clinical data.


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("role", [None, "role:org.openmed/reviewer", ROLE])
def test_final_assertions_require_consumed_receipt_and_configured_attesting_role(
    kind, role
):
    policy = FHIRWriteLabelPolicy(attesting_roles=frozenset({ROLE}))
    payload = normalize_proposed_resource(resource(kind), policy=policy).resource
    set_status(payload, "final" if kind == "Observation" else "confirmed")
    findings = validate_proposed_resource(
        payload, policy=policy, approval_receipt=receipt(role) if role else None
    )
    assert bool(findings) == (role != ROLE)
    assert validate_proposed_resource(
        payload, approval_receipt=receipt()
    )  # Empty role policy.
    # Even an attesting policy never upgrades a proposal at normalization.
    normalized = normalize_proposed_resource(payload, policy=policy)
    assert not validate_proposed_resource(normalized.resource)


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("field", ["security", "tag"])
@pytest.mark.parametrize(
    "mutation", ["missing", "system", "code", "split-pair", "empty", "malformed"]
)
def test_origin_label_negative_controls_fail_closed(kind, field, mutation):
    payload = normalize_proposed_resource(resource(kind)).resource
    if mutation == "missing":
        del payload["meta"][field]
    elif mutation in {"system", "code"}:
        payload["meta"][field][0][mutation] = CANARY
    elif mutation == "split-pair":
        correct = payload["meta"][field][0]
        payload["meta"][field] = [
            {"system": correct["system"], "code": "wrong"},
            {"system": "wrong", "code": correct["code"]},
        ]
    elif mutation == "empty":
        payload["meta"][field] = []
    else:
        payload["meta"][field] = [CANARY]
    findings = validate_proposed_resource(payload, approval_receipt=receipt())
    assert findings
    assert CANARY not in repr(findings)
    if mutation != "missing":
        before = deepcopy(payload)
        with pytest.raises(WriteLabelError) as error:
            normalize_proposed_resource(payload)
        assert payload == before
        assert CANARY not in str(error.value)
        assert CANARY not in repr(error.value.findings)


def test_custom_labels_are_exact_and_preserve_other_metadata():
    policy = FHIRWriteLabelPolicy(
        security_system="urn:synthetic:origin",
        security_code="auto",
        tag_system="urn:synthetic:openmed",
        tag_code="proposed",
    )
    payload = normalize_proposed_resource(
        resource("Observation"), policy=policy
    ).resource
    payload["meta"]["security"].append(
        {"system": "urn:synthetic:confidentiality", "code": "restricted"}
    )
    assert not validate_proposed_resource(payload, policy=policy)
    assert validate_proposed_resource(payload)  # Default pair must not match.
    assert normalize_proposed_resource(payload, policy=policy).resource == payload


@pytest.mark.parametrize(
    "kind,status",
    [
        ("Observation", "registered"),
        ("Observation", "preliminary"),
        ("Observation", "cancelled"),
        ("Observation", "entered-in-error"),
        ("Condition", "refuted"),
        ("Condition", "differential"),
        ("Condition", "unconfirmed"),
        ("Condition", "provisional"),
        ("Condition", "entered-in-error"),
        ("AllergyIntolerance", "refuted"),
        ("AllergyIntolerance", "unconfirmed"),
        ("AllergyIntolerance", "entered-in-error"),
    ],
)
def test_tentative_and_negative_states_never_become_positive(kind, status):
    original = resource(kind, status)
    result = normalize_proposed_resource(original)
    path = "status" if kind == "Observation" else "verificationStatus"
    assert result.resource[path] == original[path]
    assert not validate_proposed_resource(result.resource)


@pytest.mark.parametrize("status", ["final", "amended", "corrected"])
def test_all_final_observation_states_need_attestation(status):
    payload = normalize_proposed_resource(resource("Observation")).resource
    set_status(payload, status)
    assert validate_proposed_resource(payload)[0].code == "attestation_required"
    assert normalize_proposed_resource(payload).resource["status"] == "preliminary"


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("malformed", [CANARY, None, [], {"coding": []}])
def test_malformed_status_is_not_silently_reinterpreted(kind, malformed):
    payload = resource(kind)
    payload["status" if kind == "Observation" else "verificationStatus"] = malformed
    assert validate_proposed_resource(payload)
    with pytest.raises(WriteLabelError) as error:
        normalize_proposed_resource(payload)
    assert CANARY not in repr(error.value.findings)


@pytest.mark.parametrize("kind", TYPES)
def test_missing_status_becomes_provisional_but_validation_rejects_it(kind):
    payload = resource(kind)
    del payload["status" if kind == "Observation" else "verificationStatus"]
    assert validate_proposed_resource(payload)
    assert not validate_proposed_resource(normalize_proposed_resource(payload).resource)


@pytest.mark.parametrize("kind", ["Condition", "AllergyIntolerance"])
def test_status_system_and_multiple_codings_fail_closed(kind):
    for codings in [
        [{"system": "urn:wrong", "code": "confirmed"}],
        [
            {"system": SYSTEMS[kind], "code": "unconfirmed"},
            {"system": SYSTEMS[kind], "code": "confirmed"},
        ],
    ]:
        payload = resource(kind)
        payload["verificationStatus"] = {"coding": codings}
        with pytest.raises(WriteLabelError):
            normalize_proposed_resource(payload)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"resourceType": CANARY},
        {"resourceType": []},
        {"resourceType": "Bundle", "entry": []},
    ],
)
def test_unsupported_shapes_never_echo_untrusted_resource_type(payload):
    findings = validate_proposed_resource(payload)
    assert findings[0].resource_type == "Resource"
    with pytest.raises(WriteLabelError) as error:
        normalize_proposed_resource(payload)
    assert CANARY not in repr(error.value.findings)


@pytest.mark.parametrize("meta", [None, CANARY, [], 42])
def test_invalid_meta_fails_without_leakage(meta):
    payload = resource("Observation")
    payload["meta"] = meta
    with pytest.raises(WriteLabelError) as error:
        normalize_proposed_resource(payload)
    assert error.value.findings[0].code == "invalid_meta"
    assert CANARY not in repr(error.value.findings)


def test_embedded_proposals_cannot_bypass_labeling():
    payload = resource("Observation")
    payload["contained"] = [resource("Condition")]
    with pytest.raises(WriteLabelError) as error:
        normalize_proposed_resource(payload)
    assert error.value.findings[0].code == "nested_resources_unsupported"


def test_plain_role_or_receipt_mapping_cannot_attest():
    policy = FHIRWriteLabelPolicy(attesting_roles=frozenset({ROLE}))
    payload = normalize_proposed_resource(resource("Observation")).resource
    payload["status"] = "final"
    for invalid in [ROLE, {"reviewer_role": ROLE}, CANARY]:
        assert validate_proposed_resource(
            payload, policy=policy, approval_receipt=invalid
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"security_code": ""},
        {"security_system": None},
        {"tag_code": " padded "},
        {"attesting_roles": {ROLE}},
        {"attesting_roles": frozenset({None})},
    ],
)
def test_invalid_configuration_uses_constant_errors(kwargs):
    with pytest.raises(ValueError, match="^invalid_write_label_policy$"):
        FHIRWriteLabelPolicy(**kwargs)
