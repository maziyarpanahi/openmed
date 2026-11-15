"""Offline, value-free scope verification before FHIR write previews."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit

from openmed.clinical.exporters.fhir.reference_types import REFERENCE_TARGET_ALLOWLISTS
from openmed.interop.fhir.reference_integrity import iter_fhir_elements

from .access_tickets import (
    AccessTicket,
    AccessTicketRequest,
    AccessTicketVerifier,
    RecordSelector,
)

_T = TypeVar("_T")

__all__ = [
    "FHIRWriteScopeError",
    "FHIRWriteScopeFinding",
    "FHIRWriteScopeVerifier",
    "ReferenceScopeResolver",
    "preview_with_fhir_write_scope",
]

# Only source-controlled FHIR field names may enter diagnostics. Unknown profile
# fields still get inspected, but receive positional paths without copying keys.
_REFERENCE_FIELDS = frozenset(
    part
    for version in REFERENCE_TARGET_ALLOWLISTS.values()
    for fields in version.values()
    for path in fields
    for part in path.split(".")
) | frozenset({"valueReference", "focus", "beneficiary", "supportingInformation"})
_PATH_KEYS = _REFERENCE_FIELDS | frozenset(
    {
        "entry",
        "resource",
        "contained",
        "extension",
        "modifierExtension",
        "reference",
        "identifier",
        "resourceType",
        "fullUrl",
        "type",
        "request",
        "component",
        "item",
        "agent",
        "entity",
        "participant",
    }
)
_IDENTITY = re.compile(
    r"([A-Z][A-Za-z0-9]*)/([A-Za-z0-9.-]{1,64})"
    r"(?:/_history/[A-Za-z0-9.-]{1,64})?"
)
_CODES = frozenset(
    {
        "invalid_payload",
        "invalid_configuration",
        "identifier_only_reference",
        "invalid_reference",
        "foreign_base_reference",
        "unresolvable_reference",
        "ambiguous_reference",
        "scope_evidence_missing",
        "selector_out_of_scope",
    }
)


@dataclass(frozen=True)
class FHIRWriteScopeFinding:
    """A controlled reason code and value-free FHIR element path."""

    path: str
    code: str

    def to_dict(self) -> dict[str, str]:
        """Return only the structural path and controlled reason code."""
        return {"path": self.path, "code": self.code}


class FHIRWriteScopeError(ValueError):
    """Reject a proposed write with paths and codes only."""

    def __init__(self, findings: tuple[FHIRWriteScopeFinding, ...]) -> None:
        self.findings = findings
        super().__init__("; ".join(f"{item.path}: {item.code}" for item in findings))


class ReferenceScopeResolver(Protocol):
    """Resolve one reference to candidate scopes without network requirements."""

    def resolve(
        self, reference: str, *, resource: Mapping[str, Any] | None
    ) -> Sequence[tuple[RecordSelector, ...]]:
        """Return zero, one or multiple candidate scopes for a reference.

        Args:
            reference: Transient reference text; never log or retain it in evidence.
            resource: Exact local transaction/contained target, or None for an
                existing server resource. Local presence alone is not authority.

        Returns:
            Candidate scopes made only of keyed RecordSelectors. Zero candidates
            means unresolved; multiple candidates mean ambiguous. Each accepted
            scope must contain a patient and, when the ticket is encounter-bound,
            an encounter selector. A resolver must not trust proposed identifiers
            or ownership claims without independently establishing their scope.
        """
        ...


class FHIRWriteScopeVerifier:
    """Verify complete ticket authority and every write-payload reference.

    Selector kinds are caller-owned contracts, matching the ticket issuer. Direct
    Patient/Encounter references hash the unversioned resource id (not its URL).
    Patient/Encounter URN targets with ids use the same keyed mapping. All other
    targets use the resolver. Payloads, references and tickets are not retained;
    the local selector key is private verifier configuration.
    """

    def __init__(
        self,
        *,
        selector_key: bytes,
        patient_selector_kind: str,
        encounter_selector_kind: str,
        server_base: str,
        resolver: ReferenceScopeResolver,
        ticket_verifier: AccessTicketVerifier | None = None,
    ) -> None:
        """Configure caller-owned selector contracts and the local resolver."""
        invalid = False
        try:
            RecordSelector.from_value(
                kind=patient_selector_kind, value="", key=selector_key
            )
            RecordSelector.from_value(
                kind=encounter_selector_kind, value="", key=selector_key
            )
            base = urlsplit(server_base)
            if (
                base.scheme not in {"http", "https"}
                or not base.netloc
                or base.username
                or base.password
                or base.query
                or base.fragment
                or patient_selector_kind == encounter_selector_kind
                or not callable(resolver.resolve)
                or (
                    ticket_verifier is not None
                    and type(ticket_verifier) is not AccessTicketVerifier
                )
            ):
                raise ValueError
        except Exception:
            invalid = True
        if invalid:
            _deny("invalid_configuration", "Resource")
        self._key = selector_key
        self._patient_kind = patient_selector_kind
        self._encounter_kind = encounter_selector_kind
        self._base = server_base.rstrip("/")
        self._resolver = resolver
        self._tickets = ticket_verifier or AccessTicketVerifier()

    def __repr__(self) -> str:
        """Return a value-free representation."""
        return "FHIRWriteScopeVerifier(<redacted>)"

    def verify(
        self,
        payload: Mapping[str, Any],
        ticket: AccessTicket | None,
        request: AccessTicketRequest,
        *,
        now: int | None = None,
    ) -> None:
        """Reject before preview unless the active ticket covers every reference.

        Args:
            payload: Proposed create/update resource or transaction Bundle.
                Never mutated, serialized, logged or included in the result.
            ticket: Authority issued for the active local run.
            request: Actual run, purpose, projection and write tool/action.
            now: Optional integer clock value for deterministic validation.

        Raises:
            AccessTicketError: If full ticket authorization fails.
            FHIRWriteScopeError: If any reference cannot be placed in scope.
                Findings, messages and repr contain only paths and codes.
        """
        active = self._tickets.verify(ticket, request, now=now)
        if not any(s.kind == self._patient_kind for s in active.record_selectors):
            _deny("scope_evidence_missing", "Resource")
        _validate_payload(payload)
        root = "Bundle" if payload.get("resourceType") == "Bundle" else "Resource"
        elements = tuple(iter_fhir_elements(payload, root, path_keys=_PATH_KEYS))
        resources = [(root, payload)] + [
            (path, node)
            for path, _, node in elements
            if isinstance(node, Mapping) and "resourceType" in node
        ]
        reference_paths = {}
        for resource_path, resource in resources:
            fields = set()
            for version in REFERENCE_TARGET_ALLOWLISTS.values():
                for field_path in version.get(resource["resourceType"], {}):
                    fields.add(field_path)
            reference_paths[resource_path] = fields
        resource_paths = sorted(reference_paths, key=len, reverse=True)
        urns: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        if root == "Bundle":
            for entry in payload["entry"]:
                full_url = entry.get("fullUrl")
                if isinstance(full_url, str) and full_url.startswith("urn:uuid:"):
                    urns[full_url].append(entry["resource"])
        findings = []
        for path, field, node in elements:
            # The resource-specific contracts distinguish Reference fields from
            # unrelated primitive/backbone fields with the same name.
            owner_path = next(p for p in resource_paths if path.startswith(p + "."))
            relative_path = re.sub(r"\[\d+\]", "", path[len(owner_path) + 1 :])
            hinted = (
                relative_path in reference_paths[owner_path]
                or field == "valueReference"
            )
            # CodeableReference.reference is unwrapped by the shared walker.
            if isinstance(node, Mapping):
                if isinstance(node.get("reference"), Mapping):
                    continue
                is_reference = (
                    "reference" in node
                    or ("resourceType" not in node and "identifier" in node)
                    or (hinted and "concept" not in node)
                )
                if not is_reference:
                    continue
            elif hinted and not isinstance(node, list):
                findings.append(FHIRWriteScopeFinding(path, "invalid_reference"))
                continue
            else:
                continue
            code = self._check(node, path, active, resources, urns)
            if code:
                findings.append(FHIRWriteScopeFinding(path, code))
        if findings:
            raise FHIRWriteScopeError(tuple(findings)) from None

    def _check(
        self,
        node: Mapping[str, Any],
        path: str,
        ticket: AccessTicket,
        resources: list[tuple[str, Mapping[str, Any]]],
        urns: Mapping[str, list[Mapping[str, Any]]],
    ) -> str | None:
        reference = node.get("reference")
        if reference is None and "identifier" in node:
            return "identifier_only_reference"
        if (
            not isinstance(reference, str)
            or not reference
            or reference != reference.strip()
        ):
            return "invalid_reference"
        target = None
        identity = None
        if reference.startswith("urn:uuid:"):
            if (
                re.fullmatch(
                    r"urn:uuid:[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                    reference,
                )
                is None
            ):
                return "invalid_reference"
            candidates = urns.get(reference, [])
            if len(candidates) != 1:
                return "ambiguous_reference" if candidates else "unresolvable_reference"
            target = candidates[0]
            if target.get("resourceType") in {"Patient", "Encounter"} and isinstance(
                target.get("id"), str
            ):
                identity = target["resourceType"], target["id"]
        elif reference.startswith("#"):
            # Fragments are scoped to the nearest containing top-level resource,
            # never to a similarly named contained resource in another entry.
            owners = [
                (p, r)
                for p, r in resources
                if path.startswith(p + ".") and ".contained[" not in p
            ]
            owner = max(owners, key=lambda item: len(item[0]))[1]
            candidates = [
                r
                for p, r in resources
                if ".contained[" in p
                and r.get("id") == reference[1:]
                and any(r is c for c in owner.get("contained", []))
            ]
            if len(candidates) != 1:
                return "ambiguous_reference" if candidates else "unresolvable_reference"
            target = candidates[0]
        else:
            relative = reference
            if ":" in reference or reference.startswith("//"):
                if not reference.startswith(self._base + "/"):
                    return "foreign_base_reference"
                relative = reference[len(self._base) + 1 :]
            match = _IDENTITY.fullmatch(relative)
            if not match:
                return "invalid_reference"
            identity = match.group(1), match.group(2)
            candidates = [
                r
                for p, r in resources
                if ".contained[" not in p
                and (r.get("resourceType"), r.get("id")) == identity
            ]
            if len(candidates) > 1:
                return "ambiguous_reference"
            if candidates:
                target = candidates[0]
        target_type = identity[0] if identity else (target or {}).get("resourceType")
        if target_type and node.get("type") not in (
            None,
            target_type,
            "http://hl7.org/fhir/" + target_type,
        ):
            return "ambiguous_reference"
        if identity and identity[0] in {"Patient", "Encounter"}:
            if _IDENTITY.fullmatch("/".join(identity)) is None:
                return "invalid_reference"
            kind = (
                self._patient_kind if identity[0] == "Patient" else self._encounter_kind
            )
            selector = RecordSelector.from_value(
                kind=kind, value=identity[1], key=self._key
            )
            return (
                None if selector in ticket.record_selectors else "selector_out_of_scope"
            )
        try:
            scopes = self._resolver.resolve(reference, resource=target)
            if not isinstance(scopes, (list, tuple)):
                return "scope_evidence_missing"
            if not scopes:
                return "unresolvable_reference"
            if len(scopes) != 1:
                return "ambiguous_reference"
            scope = scopes[0]
            if not isinstance(scope, tuple) or any(
                type(s) is not RecordSelector for s in scope
            ):
                return "scope_evidence_missing"
            kinds = [s.kind for s in scope]
            if (
                kinds.count(self._patient_kind) != 1
                or (
                    any(s.kind == self._encounter_kind for s in ticket.record_selectors)
                    and kinds.count(self._encounter_kind) != 1
                )
                or kinds.count(self._encounter_kind) > 1
            ):
                return "scope_evidence_missing"
            if not set(scope).issubset(ticket.record_selectors):
                return "selector_out_of_scope"
        except Exception:
            return "unresolvable_reference"
        return None


def _deny(code: str, path: str) -> None:
    assert code in _CODES
    raise FHIRWriteScopeError((FHIRWriteScopeFinding(path, code),)) from None


def _validate_payload(payload: Any) -> None:
    # Reject cycles/deep or non-JSON input before recursion; never echo source.
    nodes = 0

    def visit(node: Any, depth: int, ancestors: set[int]) -> None:
        nonlocal nodes
        nodes += 1
        if depth > 64 or nodes > 100_000:
            _deny("invalid_payload", "Resource")
        if isinstance(node, (Mapping, list)):
            if id(node) in ancestors:
                _deny("invalid_payload", "Resource")
            branch = ancestors | {id(node)}
            children: Iterable[Any]
            if isinstance(node, Mapping):
                if "resourceType" in node and (
                    not isinstance(node["resourceType"], str)
                    or re.fullmatch(r"[A-Z][A-Za-z0-9]*", node["resourceType"]) is None
                ):
                    _deny("invalid_payload", "Resource")
                if "contained" in node and (
                    not isinstance(node["contained"], list)
                    or any(
                        not isinstance(c, Mapping) or "resourceType" not in c
                        for c in node["contained"]
                    )
                ):
                    _deny("invalid_payload", "Resource.contained")
                if any(type(k) is not str for k in node):
                    _deny("invalid_payload", "Resource")
                children = node.values()
            else:
                children = node
            for child in children:
                visit(child, depth + 1, branch)
        elif node is not None and type(node) not in (str, int, float, bool):
            _deny("invalid_payload", "Resource")

    visit(payload, 0, set())
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("resourceType"), str
    ):
        _deny("invalid_payload", "Resource")
    if payload["resourceType"] == "Bundle":
        entries = payload.get("entry")
        if (
            payload.get("type") != "transaction"
            or not isinstance(entries, list)
            or not entries
        ):
            _deny("invalid_payload", "Bundle.entry")
        for entry in entries:
            if (
                not isinstance(entry, Mapping)
                or not isinstance(entry.get("resource"), Mapping)
                or "resourceType" not in entry["resource"]
            ):
                _deny("invalid_payload", "Bundle.entry")
            request = entry.get("request")
            if (
                not isinstance(request, Mapping)
                or not isinstance(request.get("method"), str)
                or request["method"] not in {"POST", "PUT"}
            ):
                _deny("invalid_payload", "Bundle.entry.request")


def preview_with_fhir_write_scope(
    payload: Mapping[str, Any],
    ticket: AccessTicket | None,
    request: AccessTicketRequest,
    verifier: FHIRWriteScopeVerifier,
    preview: Callable[[], _T],
    *,
    now: int | None = None,
) -> _T:
    """Verify write references before invoking the caller's local preview.

    Args:
        payload: Proposed write; the callback must preview this exact payload.
        ticket: Active purpose-bound authority.
        request: Actual run and requested write operation.
        verifier: Configured local write-scope verifier.
        preview: No-argument callback that renders the preview after verification.
        now: Optional deterministic timestamp for ticket verification.

    Returns:
        The preview callback's unchanged result.

    Raises:
        FHIRWriteScopeError: On invalid configuration or rejected references.
        AccessTicketError: On invalid or expired ticket authority.
    """
    if type(verifier) is not FHIRWriteScopeVerifier or not callable(preview):
        _deny("invalid_configuration", "Resource")
    verifier.verify(payload, ticket, request, now=now)
    return preview()
