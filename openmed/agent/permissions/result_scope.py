"""Authorize complete typed tool-result batches before downstream exposure.

Trusted local adapters supply scope and evidence from authoritative resource
metadata, never from model assertions. Rejected payloads remain with the caller;
this module stores no quarantine content and emits only controlled codes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

from openmed.agent.artifact_reference import ArtifactReference
from openmed.agent.tools.data_projection import DataProjectionPlan, plan_data_projection

from .access_tickets import (
    AccessTicket,
    AccessTicketRequest,
    AccessTicketVerifier,
    RecordSelector,
)

_T = TypeVar("_T")
_MAX_PAGES = 64
_MAX_NODES = 10_000
_MAX_DEPTH = 32


class ResultQuarantineCode(str, Enum):
    """Closed vocabulary for result authorization failures."""

    INVALID_BINDING = "invalid_binding"
    MISSING_SCOPE = "missing_scope"
    AMBIGUOUS_SCOPE = "ambiguous_scope"
    SCOPE_MISMATCH = "scope_mismatch"
    MISSING_EVIDENCE = "missing_evidence"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    INVALID_RESULT = "invalid_result"
    INCOMPLETE_PAGES = "incomplete_pages"
    PROJECTION_MISMATCH = "projection_mismatch"
    OVERBROAD_OUTPUT = "overbroad_output"
    MISSING_FIELD = "missing_field"
    INVALID_FIELD = "invalid_field"
    RESULT_LIMIT = "result_limit"
    READ_FAILED = "read_failed"


class ResultQuarantinedError(ValueError):
    """Fail closed without retaining a result, field name, or provider message."""

    def __init__(self, code: ResultQuarantineCode) -> None:
        if type(code) is not ResultQuarantineCode:
            raise ValueError("invalid_quarantine_code")
        self.code = code
        super().__init__(code.value)

    def to_dict(self) -> dict[str, str]:
        """Return content-free evidence suitable for an action trace."""

        return {
            "schema_version": "openmed.agent.result_quarantine.v1",
            "reason_code": self.code.value,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ResultScope:
    """Four independently bound opaque selectors for a resource result.

    Args:
        patient: Patient selector derived by the trusted local adapter.
        encounter: Encounter selector derived independently from the resource.
        namespace: Source namespace selector; never a credential or endpoint.
        snapshot: Selector for the exact evidence snapshot, including its version.
    """

    patient: RecordSelector
    encounter: RecordSelector
    namespace: RecordSelector
    snapshot: RecordSelector

    def selectors(self) -> tuple[RecordSelector, ...]:
        """Return unambiguous selectors without inferring or resolving identity."""

        values = (self.patient, self.encounter, self.namespace, self.snapshot)
        if any(type(value) is not RecordSelector for value in values):
            raise ResultQuarantinedError(ResultQuarantineCode.MISSING_SCOPE)
        if len({value.kind for value in values}) != len(values):
            raise ResultQuarantinedError(ResultQuarantineCode.AMBIGUOUS_SCOPE)
        return values

    def __repr__(self) -> str:
        """Return a representation without selector metadata."""

        return "ResultScope(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ToolResultRecord:
    """One protected record and any independently scoped nested records.

    Args:
        scope: Independently established scope; absent evidence fails closed.
        evidence: Original evidence identity, not an attestation of projected data.
        fields: Structured values under the reviewed per-record output schema.
        children: Nested resources, each requiring its own scope and evidence.
    """

    scope: ResultScope | None
    evidence: ArtifactReference | None
    fields: dict[str, Any]
    children: tuple[ToolResultRecord, ...] = ()

    def __repr__(self) -> str:
        """Return a representation without source values or evidence metadata."""

        return "ToolResultRecord(<protected>)"


@dataclass(frozen=True, slots=True, repr=False)
class ToolResultPage:
    """A locally collected page with explicit scope and completion metadata.

    Args:
        scope: Page scope, independently verified rather than inherited.
        records: Typed records from this page.
        index: Zero-based page index in the complete local batch.
        final: Whether the adapter established this is the last source page.
    """

    scope: ResultScope | None
    records: tuple[ToolResultRecord, ...]
    index: int = 0
    final: bool = True

    def __repr__(self) -> str:
        """Return a representation without page content."""

        return "ToolResultPage(<protected>)"


def _binding(
    ticket: AccessTicket | None,
    request: AccessTicketRequest,
    verifier: AccessTicketVerifier,
    scope: ResultScope,
    schema: Mapping[str, Any],
    now: int | None,
) -> DataProjectionPlan:
    if type(verifier) is not AccessTicketVerifier:
        raise ResultQuarantinedError(ResultQuarantineCode.INVALID_BINDING)
    verifier.verify(ticket, request, now=now)
    if type(scope) is not ResultScope or set(scope.selectors()) != set(
        request.record_selectors
    ):
        raise ResultQuarantinedError(ResultQuarantineCode.INVALID_BINDING)
    # Derive from the narrower request, never from all classes in the ticket.
    return plan_data_projection(
        schema,
        workflow_purpose=request.purpose,
        granted_data_classes=request.projection,
    )


def authorize_tool_results(
    ticket: AccessTicket | None,
    request: AccessTicketRequest,
    verifier: AccessTicketVerifier,
    *,
    scope: ResultScope,
    schema: Mapping[str, Any],
    pages: tuple[ToolResultPage, ...],
    now: int | None = None,
) -> tuple[ToolResultPage, ...]:
    """Release a projected copy only after the entire result batch is authorized.

    Args:
        ticket: Active purpose-bound data authority.
        request: Exact run, purpose, selectors and data classes of this read.
        verifier: Existing local ticket verifier, with an injectable clock.
        scope: Trusted expected scope, exactly covering the request selectors.
        schema: Reviewed per-record output schema using minimum-data annotations.
        pages: Complete, ordered local batch; no lazy or network pagination.
        now: Optional deterministic verification time.

    Returns:
        Fresh structured fields with original scope and evidence identity.

    Raises:
        ResultQuarantinedError: For mismatched, ambiguous, overbroad or malformed
            output. No page or partial record is released on failure.
        AccessTicketError: If current data authority is invalid or expired.
        DataProjectionError: If the declared field projection is unauthorized.
    """

    schema = deepcopy(schema)
    plan = _binding(ticket, request, verifier, scope, schema, now)
    if type(pages) is not tuple or not pages:
        raise ResultQuarantinedError(ResultQuarantineCode.INCOMPLETE_PAGES)
    if len(pages) > _MAX_PAGES:
        raise ResultQuarantinedError(ResultQuarantineCode.RESULT_LIMIT)
    budget = [0]
    evidence_ids: set[str] = set()
    projected: list[ToolResultPage] = []
    for index, page in enumerate(pages):
        if type(page) is not ToolResultPage:
            raise ResultQuarantinedError(ResultQuarantineCode.INVALID_RESULT)
        if (
            type(page.index) is not int
            or page.index != index
            or type(page.final) is not bool
            or page.final != (index == len(pages) - 1)
        ):
            raise ResultQuarantinedError(ResultQuarantineCode.INCOMPLETE_PAGES)
        _scope_matches(page.scope, scope)
        projected.append(
            ToolResultPage(
                page.scope,
                _records(page.records, scope, schema, plan, evidence_ids, budget, 0),
                index,
                page.final,
            )
        )
    # A runtime clock is rechecked after validation, before downstream exposure.
    verifier.verify(ticket, request, now=now)
    return tuple(projected)


def dispatch_with_authorized_results(
    ticket: AccessTicket | None,
    request: AccessTicketRequest,
    verifier: AccessTicketVerifier,
    *,
    scope: ResultScope,
    schema: Mapping[str, Any],
    read: Callable[[], tuple[ToolResultPage, ...]],
    consume: Callable[[tuple[ToolResultPage, ...]], _T],
    now: int | None = None,
) -> _T:
    """Guard a local read and invoke the next step only with authorized results.

    Args:
        ticket: Active data-access ticket.
        request: Exact authority used for the read.
        verifier: Local verifier with the runtime clock.
        scope: Trusted expected patient, encounter, namespace and snapshot.
        schema: Reviewed minimum-data output schema.
        read: Local adapter returning the complete typed batch.
        consume: Next step, responsible for separate privacy/injection screening.
        now: Optional deterministic time; runtime callers should use the clock.

    Returns:
        The next step's return value after complete result authorization.

    Raises:
        ResultQuarantinedError: If a read fails or results cannot be authorized.
        AccessTicketError: If ticket authority fails before or after the read.
        DataProjectionError: If the output contract exceeds the request.
    """

    if not callable(read) or not callable(consume):
        raise ResultQuarantinedError(ResultQuarantineCode.INVALID_BINDING)
    # Pin the reviewed contract before the adapter can mutate caller-owned data.
    schema = deepcopy(schema)
    _binding(ticket, request, verifier, scope, schema, now)
    try:
        pages = read()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ResultQuarantinedError(ResultQuarantineCode.READ_FAILED) from None
    authorized = authorize_tool_results(
        ticket, request, verifier, scope=scope, schema=schema, pages=pages, now=now
    )
    return consume(authorized)


def _scope_matches(actual: ResultScope | None, expected: ResultScope) -> None:
    if type(actual) is not ResultScope:
        raise ResultQuarantinedError(ResultQuarantineCode.MISSING_SCOPE)
    actual.selectors()
    if actual != expected:
        raise ResultQuarantinedError(ResultQuarantineCode.SCOPE_MISMATCH)


def _count(budget: list[int], depth: int) -> None:
    budget[0] += 1
    if budget[0] > _MAX_NODES or depth > _MAX_DEPTH:
        raise ResultQuarantinedError(ResultQuarantineCode.RESULT_LIMIT)


def _records(
    records: tuple[ToolResultRecord, ...],
    scope: ResultScope,
    schema: Mapping[str, Any],
    plan: DataProjectionPlan,
    evidence_ids: set[str],
    budget: list[int],
    depth: int,
) -> tuple[ToolResultRecord, ...]:
    if type(records) is not tuple:
        raise ResultQuarantinedError(ResultQuarantineCode.INVALID_RESULT)
    result = []
    for record in records:
        _count(budget, depth)
        if type(record) is not ToolResultRecord:
            raise ResultQuarantinedError(ResultQuarantineCode.INVALID_RESULT)
        _scope_matches(record.scope, scope)
        if type(record.evidence) is not ArtifactReference:
            raise ResultQuarantinedError(ResultQuarantineCode.MISSING_EVIDENCE)
        # Even identical repeated IDs are ambiguous record custody in this batch.
        if record.evidence.artifact_id in evidence_ids:
            raise ResultQuarantinedError(ResultQuarantineCode.AMBIGUOUS_EVIDENCE)
        evidence_ids.add(record.evidence.artifact_id)
        fields = _project(record.fields, schema, plan, "", budget, depth)
        children = _records(
            record.children, scope, schema, plan, evidence_ids, budget, depth + 1
        )
        result.append(
            ToolResultRecord(
                record.scope, record.evidence, cast(dict, fields), children
            )
        )
    return tuple(result)


def _project(
    value: Any,
    schema: Mapping[str, Any],
    plan: DataProjectionPlan,
    path: str,
    budget: list[int],
    depth: int,
) -> Any:
    _count(budget, depth)
    kind = schema.get("type")
    if kind == "object":
        if type(value) is not dict:
            raise ResultQuarantinedError(ResultQuarantineCode.INVALID_FIELD)
        if any(type(name) is not str for name in value):
            raise ResultQuarantinedError(ResultQuarantineCode.INVALID_FIELD)
        properties = schema.get("properties", {})
        if set(value) - set(properties):
            raise ResultQuarantinedError(ResultQuarantineCode.OVERBROAD_OUTPUT)
        if set(properties) - set(value):
            raise ResultQuarantinedError(ResultQuarantineCode.MISSING_FIELD)
        result = {}
        for name, child_schema in properties.items():
            child_path = f"{path}/{name}"
            if child_path not in plan.field_paths:
                raise ResultQuarantinedError(ResultQuarantineCode.PROJECTION_MISMATCH)
            result[name] = _project(
                value[name], child_schema, plan, child_path, budget, depth + 1
            )
        return result
    if kind == "array":
        if type(value) is not list or type(schema.get("items")) is not dict:
            raise ResultQuarantinedError(ResultQuarantineCode.INVALID_FIELD)
        return [
            _project(item, schema["items"], plan, f"{path}/*", budget, depth + 1)
            for item in value
        ]
    allowed_types = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
    }
    if type(kind) is not str or type(value) not in allowed_types.get(kind, ()):
        raise ResultQuarantinedError(ResultQuarantineCode.INVALID_FIELD)
    if type(value) is float and not math.isfinite(value):
        raise ResultQuarantinedError(ResultQuarantineCode.INVALID_FIELD)
    return value


__all__ = [
    "ResultQuarantineCode",
    "ResultQuarantinedError",
    "ResultScope",
    "ToolResultPage",
    "ToolResultRecord",
    "authorize_tool_results",
    "dispatch_with_authorized_results",
]
