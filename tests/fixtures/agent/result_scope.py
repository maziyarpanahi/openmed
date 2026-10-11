"""Synthetic local result-authority fixtures; no model or transport required."""

from dataclasses import replace
from typing import Any

from openmed.agent.artifact_reference import ArtifactKind, ArtifactReference
from openmed.agent.correlation import RunId
from openmed.agent.permissions.access_tickets import (
    AccessTicket,
    AccessTicketRequest,
    RecordSelector,
    ToolAction,
)
from openmed.agent.permissions.result_scope import (
    ResultScope,
    ToolResultPage,
    ToolResultRecord,
)

PURPOSE = "purpose:org.example/care-summary@1.0.0"
DATA_CLASS = "data:org.example/clinical-text@1.0.0"


def selector(kind: str, value: str = "synthetic-one") -> RecordSelector:
    """Create a synthetic keyed selector for one scope dimension."""

    return RecordSelector.from_value(
        kind=f"selector:org.example/{kind}@1.0.0", value=value, key=b"s" * 32
    )


def binding() -> tuple[AccessTicket, AccessTicketRequest, ResultScope]:
    """Return one purpose ticket, narrow request, and expected resource scope."""

    scope = ResultScope(
        *(selector(kind) for kind in ("patient", "encounter", "namespace", "snapshot"))
    )
    action = ToolAction("tool:org.example/read@1.0.0", "action:org.example/read@1.0.0")
    ticket = AccessTicket(
        RunId.generate(), PURPOSE, (DATA_CLASS,), scope.selectors(), (action,), 100
    )
    request = AccessTicketRequest(
        ticket.run_id, PURPOSE, (DATA_CLASS,), scope.selectors(), action
    )
    return ticket, request, scope


def schema() -> dict[str, Any]:
    """Return an annotated nested output schema with closed structured fields."""

    def field(kind: str, **extra: Any) -> dict[str, Any]:
        return {
            "type": kind,
            "x-openmed-purpose": PURPOSE,
            "x-openmed-minimum-data": "required",
            "x-openmed-data-class": DATA_CLASS,
            **extra,
        }

    return {
        "type": "object",
        "x-openmed-purpose": PURPOSE,
        "properties": {
            "observations": field(
                "array",
                items={
                    "type": "object",
                    "properties": {"text": field("string")},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
        },
        "required": ["observations"],
        "additionalProperties": False,
    }


def record(scope: ResultScope, index: int = 1) -> ToolResultRecord:
    """Return a typed record retaining a synthetic original artifact identity."""

    evidence = ArtifactReference(
        f"art_{index:032x}",
        ArtifactKind.EVIDENCE,
        "org.example.evidence.v1",
        f"{index:064x}",
        100,
    )
    return ToolResultRecord(
        scope, evidence, {"observations": [{"text": "synthetic finding"}]}
    )


def pages(scope: ResultScope) -> tuple[ToolResultPage, ...]:
    """Return two pages, including an independently scoped nested resource."""

    parent = replace(record(scope), children=(record(scope, 2),))
    return (
        ToolResultPage(scope, (parent,), 0, False),
        ToolResultPage(scope, (record(scope, 3),), 1, True),
    )
