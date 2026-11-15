"""Offline write-scope integration boundary before side-effect previews."""

from __future__ import annotations

import copy

import pytest

from openmed.agent import RunId
from openmed.agent.permissions import (
    AccessTicket,
    AccessTicketRequest,
    FHIRWriteScopeError,
    FHIRWriteScopeVerifier,
    RecordSelector,
    ToolAction,
    preview_with_fhir_write_scope,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("wrong", [None, "encounter", "hasMember"])
def test_authorized_ticket_cannot_preview_wrong_patient_write(wrong):
    key = b"synthetic-preview-selector-key-material"
    patient_kind = "selector:org.example/patient@1.0.0"
    encounter_kind = "selector:org.example/encounter@1.0.0"

    def selector(value, kind):
        return RecordSelector.from_value(kind=kind, value=value, key=key)

    scope = (
        selector("patient-a", patient_kind),
        selector("encounter-a", encounter_kind),
    )
    action = ToolAction(
        tool="tool:org.example/write@1.0.0", action="action:org.example/update@1.0.0"
    )
    ticket = AccessTicket(
        run_id=RunId.parse("run_" + "1" * 32),
        purpose="purpose:org.example/care@1.0.0",
        permitted_data_classes=("data:org.example/observation@1.0.0",),
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

    class Resolver:
        def resolve(self, reference, *, resource):
            return [(selector("patient-b", patient_kind), scope[1])]

    guard = FHIRWriteScopeVerifier(
        selector_key=key,
        patient_selector_kind=patient_kind,
        encounter_selector_kind=encounter_kind,
        server_base="https://synthetic.example/fhir",
        resolver=Resolver(),
    )
    payload = {
        "resourceType": "Observation",
        "subject": {"reference": "Patient/patient-a"},
        "encounter": {"reference": "Encounter/encounter-a"},
    }
    if wrong == "encounter":
        payload["encounter"]["reference"] = "Encounter/encounter-b"
    elif wrong:
        payload["hasMember"] = [{"reference": "Observation/synthetic-other"}]
    original = copy.deepcopy(payload)
    previews = []
    expected = object()

    def preview():
        previews.append(payload)
        return expected

    if wrong:
        with pytest.raises(FHIRWriteScopeError):
            preview_with_fhir_write_scope(
                payload, ticket, request, guard, preview, now=99
            )
        assert previews == []
    else:
        assert (
            preview_with_fhir_write_scope(
                payload, ticket, request, guard, preview, now=99
            )
            is expected
        )
        assert previews == [payload]
    assert payload == original
