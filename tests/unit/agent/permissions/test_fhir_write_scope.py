"""Synthetic offline negative controls for FHIR write authorization."""

from __future__ import annotations

import copy
import json
import traceback
from typing import Any

import pytest

from openmed.agent.correlation import RunId
from openmed.agent.permissions.access_tickets import (
    AccessTicket,
    AccessTicketExpiredError,
    AccessTicketRequest,
    RecordSelector,
    ToolAction,
)
from openmed.agent.permissions.fhir_write_scope import (
    FHIRWriteScopeError,
    FHIRWriteScopeVerifier,
)

KEY = b"synthetic-local-selector-key-32-bytes"
PATIENT_KIND = "selector:org.example/patient@1.0.0"
ENCOUNTER_KIND = "selector:org.example/encounter@1.0.0"
BASE = "https://synthetic.example/fhir"


def selector(value: str, kind: str = PATIENT_KIND) -> RecordSelector:
    return RecordSelector.from_value(kind=kind, value=value, key=KEY)


def authority() -> tuple[AccessTicket, AccessTicketRequest]:
    scope = (selector("patient-a"), selector("encounter-a", ENCOUNTER_KIND))
    action = ToolAction(
        tool="tool:org.example/fhir@1.0.0", action="action:org.example/write@1.0.0"
    )
    ticket = AccessTicket(
        run_id=RunId.parse("run_" + "1" * 32),
        purpose="purpose:org.example/update@1.0.0",
        permitted_data_classes=("data:org.example/observations@1.0.0",),
        record_selectors=scope,
        permitted_tool_actions=(action,),
        expires_at=100,
    )
    request = AccessTicketRequest(
        run_id=ticket.run_id,
        purpose=ticket.purpose,
        projection=ticket.permitted_data_classes,
        record_selectors=scope,
        tool_action=action,
    )
    return ticket, request


class Resolver:
    def __init__(self, result: Any = None):
        self.result = (
            result if result is not None else [authority()[0].record_selectors]
        )
        self.calls = []

    def resolve(self, reference, *, resource):
        self.calls.append((reference, resource))
        return self.result


def verifier(resolver=None, **kwargs):
    return FHIRWriteScopeVerifier(
        selector_key=KEY,
        patient_selector_kind=PATIENT_KIND,
        encounter_selector_kind=ENCOUNTER_KIND,
        server_base=BASE,
        resolver=resolver or Resolver(),
        **kwargs,
    )


def observation():
    return {
        "resourceType": "Observation",
        "id": "synthetic-observation",
        "subject": {"reference": "Patient/patient-a"},
        "encounter": {"reference": "Encounter/encounter-a"},
    }


def verify(payload, resolver=None):
    ticket, request = authority()
    return verifier(resolver).verify(payload, ticket, request, now=99)


def denied(payload, code, resolver=None):
    with pytest.raises(FHIRWriteScopeError) as caught:
        verify(payload, resolver)
    assert code in {f.code for f in caught.value.findings}
    return caught.value


@pytest.mark.parametrize(
    "reference",
    ["Patient/patient-a", BASE + "/Patient/patient-a", "Patient/patient-a/_history/7"],
)
def test_in_scope_create_and_update_pass_unchanged(reference):
    payload = observation()
    payload["subject"]["reference"] = reference
    before = copy.deepcopy(payload)
    assert verify(payload) is None
    assert payload == before


@pytest.mark.parametrize(
    "field,reference",
    [
        ("subject", "Patient/patient-b"),
        ("encounter", "Encounter/encounter-b"),
    ],
)
def test_direct_wrong_scope_is_rejected(field, reference):
    payload = observation()
    payload[field] = {"reference": reference}
    error = denied(payload, "selector_out_of_scope")
    assert error.findings[0].path == "Resource." + field


def test_wrong_patient_has_member_is_rejected():
    payload = observation()
    payload["hasMember"] = [{"reference": "Observation/other"}]
    error = denied(
        payload,
        "selector_out_of_scope",
        Resolver([(selector("patient-b"), selector("encounter-a", ENCOUNTER_KIND))]),
    )
    assert error.findings[0].path == "Resource.hasMember[0]"


@pytest.mark.parametrize(
    "node,code",
    [
        (
            {
                "identifier": {
                    "system": "synthetic-system",
                    "value": "synthetic-private",
                }
            },
            "identifier_only_reference",
        ),
        (
            {"reference": "https://foreign.example/fhir/Patient/patient-a"},
            "foreign_base_reference",
        ),
        ({"reference": BASE + "-foreign/Patient/patient-a"}, "foreign_base_reference"),
        (
            {"reference": "//synthetic.example/fhir/Patient/patient-a"},
            "foreign_base_reference",
        ),
        ({"reference": "Patient/patient-a?secret=value"}, "invalid_reference"),
        ({"reference": " Patient/patient-a"}, "invalid_reference"),
        ({"reference": ""}, "invalid_reference"),
        ({"display": "synthetic-private"}, "invalid_reference"),
        ({}, "invalid_reference"),
        (
            {"reference": "Patient/patient-a", "type": "Encounter"},
            "ambiguous_reference",
        ),
    ],
)
def test_reference_failure_codes(node, code):
    payload = observation()
    payload["subject"] = node
    denied(payload, code)


@pytest.mark.parametrize(
    "result,code",
    [
        ([], "unresolvable_reference"),
        ([authority()[0].record_selectors] * 2, "ambiguous_reference"),
        ([()], "scope_evidence_missing"),
        ([(selector("patient-a"),)], "scope_evidence_missing"),
        (
            [(selector("patient-a"), selector("encounter-b", ENCOUNTER_KIND))],
            "selector_out_of_scope",
        ),
        ([(selector("patient-a"), selector("patient-b"))], "scope_evidence_missing"),
        ([("synthetic-private",)], "scope_evidence_missing"),
        ({"synthetic-private": "bad"}, "scope_evidence_missing"),
    ],
)
def test_noncompartment_resolver_fails_closed(result, code):
    payload = observation()
    payload["basedOn"] = [{"reference": "ServiceRequest/synthetic-order"}]
    denied(payload, code, Resolver(result))


def transaction():
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": "urn:uuid:00000000-0000-4000-8000-000000000001",
                "resource": {"resourceType": "Patient"},
                "request": {"method": "POST", "url": "Patient"},
            },
            {
                "fullUrl": "urn:uuid:00000000-0000-4000-8000-000000000002",
                "resource": observation(),
                "request": {
                    "method": "PUT",
                    "url": "Observation/synthetic-observation",
                },
            },
        ],
    }


def test_transaction_internal_urns_pass_with_resolved_scope():
    payload = transaction()
    payload["entry"][1]["resource"]["subject"] = {
        "reference": "urn:uuid:00000000-0000-4000-8000-000000000001"
    }
    before = copy.deepcopy(payload)
    resolver = Resolver()
    verify(payload, resolver)
    assert payload == before
    assert resolver.calls[0][1] is payload["entry"][0]["resource"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_transaction_missing_or_ambiguous_urn_fails_closed(duplicate):
    payload = transaction()
    payload["entry"][1]["resource"]["subject"] = {
        "reference": "urn:uuid:00000000-0000-4000-8000-000000000001"
    }
    if duplicate:
        payload["entry"].append(copy.deepcopy(payload["entry"][0]))
    else:
        payload["entry"].pop(0)
    denied(payload, "ambiguous_reference" if duplicate else "unresolvable_reference")


def test_contained_reference_and_nested_wrong_patient():
    payload = observation()
    contained = observation()
    contained["id"] = "contained-observation"
    payload["contained"] = [contained]
    payload["hasMember"] = [{"reference": "#contained-observation"}]
    resolver = Resolver()
    verify(payload, resolver)
    assert resolver.calls[0][1] is contained
    contained["subject"] = {"reference": "Patient/patient-b"}
    error = denied(payload, "selector_out_of_scope", resolver)
    assert any(f.path == "Resource.contained[0].subject" for f in error.findings)


def test_fragment_cannot_resolve_in_another_entry():
    payload = transaction()
    payload["entry"][0]["resource"]["contained"] = [
        {"resourceType": "Observation", "id": "synthetic-local"}
    ]
    payload["entry"][1]["resource"]["hasMember"] = [{"reference": "#synthetic-local"}]
    denied(payload, "unresolvable_reference")


def test_unknown_profile_reference_and_codeable_reference_are_checked():
    payload = observation()
    payload["extension"] = [
        {"valueReference": {"identifier": {"value": "synthetic-secret"}}}
    ]
    payload["customSensitiveKey"] = {"reference": "Patient/patient-b"}
    payload["reason"] = [{"reference": {"reference": "Patient/patient-b"}}]
    error = denied(payload, "selector_out_of_scope")
    assert any(f.code == "identifier_only_reference" for f in error.findings)
    assert "customSensitiveKey" not in str(error)


def test_resolver_errors_never_leak_into_evidence_or_traceback():
    sentinel = "SYNTHETIC-PRIVATE-患者-123"

    class Broken:
        def resolve(self, reference, *, resource):
            raise RuntimeError(sentinel)

    payload = observation()
    payload["hasMember"] = [{"reference": "Observation/" + sentinel}]
    payload["privateKey" + sentinel] = {"identifier": {"value": sentinel}}
    payload["text"] = {"div": sentinel}
    # An invalid reference will not call the resolver; use a valid resource id.
    payload["hasMember"][0]["reference"] = "Observation/synthetic-secret"
    with pytest.raises(FHIRWriteScopeError) as caught:
        verify(payload, Broken())
    error = caught.value
    outputs = [
        str(error),
        repr(error),
        repr(error.findings),
        json.dumps([f.to_dict() for f in error.findings]),
        "".join(traceback.format_exception(error)),
    ]
    for output in outputs:
        assert sentinel not in output
        assert "synthetic-secret" not in output
    assert error.__cause__ is None
    assert error.__context__ is None


def test_expired_ticket_denied_before_resolving():
    resolver = Resolver()
    payload = observation()
    payload["hasMember"] = [{"reference": "Observation/synthetic"}]
    with pytest.raises(AccessTicketExpiredError):
        verifier(resolver).verify(payload, *authority(), now=100)
    assert not resolver.calls


@pytest.mark.parametrize(
    "mutation", ["cycle", "deep", "bad-entry", "bad-request", "non-write"]
)
def test_malformed_payload_fails_with_controlled_codes(mutation):
    payload = observation()
    if mutation == "cycle":
        payload["private"] = payload
    elif mutation == "deep":
        current = payload
        for _ in range(66):
            current["private"] = {}
            current = current["private"]
    else:
        payload = transaction()
        if mutation == "bad-entry":
            payload["entry"] = [None]
        elif mutation == "bad-request":
            payload["entry"][0]["request"] = None
        else:
            payload["entry"][0]["request"]["method"] = "DELETE"
    denied(payload, "invalid_payload")


def test_internal_patient_identity_cannot_be_overridden_by_resolver():
    payload = transaction()
    payload["entry"][0]["resource"]["id"] = "patient-b"
    payload["entry"][1]["resource"]["subject"] = {
        "reference": payload["entry"][0]["fullUrl"]
    }
    resolver = Resolver()
    denied(payload, "selector_out_of_scope", resolver)
    assert not resolver.calls


def test_known_field_names_in_unrelated_datatypes_are_not_references():
    payload = transaction()
    payload["entry"][1]["resource"]["component"] = [
        {"code": {"text": "synthetic"}, "valueString": "synthetic"}
    ]
    payload["entry"][1]["resource"]["identifier"] = [
        {"system": "synthetic", "value": "synthetic-id"}
    ]
    verify(payload)


@pytest.mark.parametrize(
    "reference",
    ["urn:uuid:invalid", "Patient/patient-a#fragment", "Patient/patient-a/extra"],
)
def test_malformed_reference_syntax_rejected(reference):
    payload = observation()
    payload["subject"] = {"reference": reference}
    denied(payload, "invalid_reference")


def test_repeated_contained_identity_is_ambiguous():
    payload = observation()
    payload["contained"] = [{"resourceType": "Observation", "id": "synthetic"}] * 2
    payload["hasMember"] = [{"reference": "#synthetic"}]
    denied(payload, "ambiguous_reference")


@pytest.mark.parametrize("method", [[], {}, None, 123])
def test_malformed_transaction_method_is_a_value_free_denial(method):
    payload = transaction()
    payload["entry"][0]["request"]["method"] = method
    denied(payload, "invalid_payload")


def test_internal_encounter_and_has_member_urns_pass_unchanged():
    payload = transaction()
    encounter_url = "urn:uuid:00000000-0000-4000-8000-000000000003"
    member_url = "urn:uuid:00000000-0000-4000-8000-000000000004"
    payload["entry"].extend(
        [
            {
                "fullUrl": encounter_url,
                "resource": {
                    "resourceType": "Encounter",
                    "id": "encounter-a",
                    "subject": {"reference": "Patient/patient-a"},
                },
                "request": {"method": "POST", "url": "Encounter"},
            },
            {
                "fullUrl": member_url,
                "resource": observation(),
                "request": {"method": "POST", "url": "Observation"},
            },
        ]
    )
    # New observations may omit ids; an internal URN still needs resolver scope.
    payload["entry"][3]["resource"].pop("id")
    target = payload["entry"][1]["resource"]
    target["encounter"] = {"reference": encounter_url}
    target["hasMember"] = [{"reference": member_url}]
    original = copy.deepcopy(payload)
    resolver = Resolver()
    verify(payload, resolver)
    assert payload == original
    assert resolver.calls == [(member_url, payload["entry"][3]["resource"])]


def test_duplicate_relative_target_is_ambiguous():
    payload = transaction()
    entry = {
        "resource": {"resourceType": "Patient", "id": "patient-a"},
        "request": {"method": "PUT", "url": "Patient/patient-a"},
    }
    payload["entry"].extend([entry, copy.deepcopy(entry)])
    denied(payload, "ambiguous_reference")


def test_invalid_internal_patient_id_fails_without_encoding_exception():
    payload = transaction()
    payload["entry"][0]["resource"]["id"] = "synthetic-invalid-\ud800"
    payload["entry"][1]["resource"]["subject"] = {
        "reference": payload["entry"][0]["fullUrl"]
    }
    error = denied(payload, "invalid_reference")
    assert "synthetic-invalid" not in str(error)
