"""Offline assembly of bounded, approval-bound FHIR R4 transactions.

This mechanical boundary neither authenticates approval nor executes writes.
The caller must verify approval for the exact returned bytes before dispatch.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from urllib.parse import parse_qsl
from uuid import UUID, uuid5

from .validation import validation_result
from .versions import SUPPORTED_RESOURCE_TYPES

__all__ = [
    "ApprovedWriteEntry",
    "AssembledTransaction",
    "TransactionApproval",
    "TransactionAssemblyError",
    "TransactionLimits",
    "TransactionReviewerRole",
    "assemble_transaction",
]

_NAMESPACE = UUID("3ae0fa9d-6e94-5ad4-94bb-3e0b80b68f93")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9.\-]{1,64}\Z")
_QUERY = re.compile(r"[A-Za-z0-9._~!$&'()*+,;=:@%\-]+\Z")
_PARAMETER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:\-]*\Z")
_SYSTEM = "https://openmed.dev/fhir/sid/"
_WRITE_RESOURCE_TYPES = SUPPORTED_RESOURCE_TYPES - {"Bundle", "Provenance"}


class TransactionAssemblyError(ValueError):
    """Controlled, value-free rejection code; never contains input content."""


class TransactionReviewerRole(str, Enum):
    """Controlled review categories, without an individual reviewer identity."""

    CLINICAL_REVIEWER = "clinical-reviewer"
    DATA_STEWARD = "data-steward"
    OPERATOR = "operator"


class ApprovedWriteEntry(Protocol):
    """Minimal caller-owned write contract, independent of planning adapters.

    Attributes:
        resource: Exact JSON-native R4 resource to write; never redacted here.
        kind: ``create`` or ``update``.
        conditional_predicate: Encoded search query without a leading ``?``;
            used as POST ifNoneExist or a conditional PUT URL.
        expected_version: Bare R4 version ID for PUT ifMatch, or None.
            Direct PUT requires resource.id; conditional PUT may omit it.
    """

    @property
    def resource(self) -> Mapping[str, Any]: ...

    @property
    def kind(self) -> str: ...

    @property
    def conditional_predicate(self) -> str | None: ...

    @property
    def expected_version(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class TransactionLimits:
    """Caller-declared inclusive limits, counting the appended Provenance."""

    max_entries: int
    max_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.max_entries, self.max_bytes)
        ):
            raise TransactionAssemblyError("invalid_limits")


@dataclass(frozen=True, slots=True)
class TransactionApproval:
    """Trusted review metadata supplied after the caller's approval checks.

    Digests use ``sha256:<64 lowercase hex>``. Evidence references are only
    canonical ``urn:uuid`` or ``urn:sha256`` commitments, never source text,
    filesystem paths, authenticated URLs, or credentials. Low-entropy evidence
    should use keyed commitments. These values alone do not prove approval.
    """

    action_digest: str
    receipt_digest: str
    reviewer_role: TransactionReviewerRole
    evidence_references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _is_digest(self.action_digest) or not _is_digest(self.receipt_digest):
            raise TransactionAssemblyError("invalid_approval_digest")
        if type(self.reviewer_role) is not TransactionReviewerRole:
            raise TransactionAssemblyError("invalid_reviewer_role")
        if (
            type(self.evidence_references) is not tuple
            or not self.evidence_references
            or any(
                not _is_evidence_reference(value) for value in self.evidence_references
            )
            or len(set(self.evidence_references)) != len(self.evidence_references)
        ):
            raise TransactionAssemblyError("invalid_evidence_references")


@dataclass(frozen=True, slots=True)
class AssembledTransaction:
    """Immutable exact bytes and SHA-256 commitment, with a safe representation."""

    serialized: bytes = field(repr=False)
    bundle_digest: str
    entry_count: int

    @property
    def bundle(self) -> dict[str, Any]:
        """Return a detached payload for protected local preview/validation.

        Mutating this copy does not change the approved bytes. Dispatch must
        use serialized, or reassemble and obtain approval for the new digest.
        """
        return json.loads(self.serialized)


def assemble_transaction(
    entries: Sequence[ApprovedWriteEntry],
    *,
    approval: TransactionApproval,
    limits: TransactionLimits,
    clock: Callable[[], datetime],
) -> AssembledTransaction:
    """Assemble one atomic R4 transaction without I/O or implicit splitting.

    Args:
        entries: Ordered caller-approved creates and updates using the protocol.
        approval: Verified action/receipt commitments and controlled metadata.
        limits: Inclusive total entry count and canonical UTF-8 byte limits.
        clock: Injected aware datetime provider; retain the returned instant
            when rebuilding the identical reviewed transaction.

    Returns:
        Canonical JSON bytes (sorted keys, compact separators, UTF-8), their
        SHA-256 digest, and total entry count. Approval must bind this digest.

    Raises:
        TransactionAssemblyError: For malformed input, unsupported R4 shapes,
            duplicate targets, unsafe metadata, or exceeded limits. Errors
            contain only fixed codes, never resource or predicate content.
    """
    if (
        type(approval) is not TransactionApproval
        or type(limits) is not TransactionLimits
    ):
        raise TransactionAssemblyError("invalid_assembly_metadata")
    if (
        not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes))
        or not entries
    ):
        raise TransactionAssemblyError("invalid_entries")
    if len(entries) + 1 > limits.max_entries:
        raise TransactionAssemblyError("entry_limit_exceeded")

    # Snapshot each protocol property once. Do not retain caller-owned mappings
    # or copy extra fields from a planner, approval token, or evidence object.
    prepared: list[dict[str, Any]] = []
    logical_ids: set[str] = set()
    requests: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            resource = entry.resource
            kind = entry.kind
            predicate = entry.conditional_predicate
            version = entry.expected_version
        except Exception:
            raise TransactionAssemblyError("invalid_entry") from None
        resource = _snapshot_resource(resource)
        resource_type = resource.get("resourceType")
        if type(resource_type) is not str or resource_type not in _WRITE_RESOURCE_TYPES:
            raise TransactionAssemblyError("unsupported_resource_type")
        resource_id = resource.get("id")
        if "id" in resource:
            if type(resource_id) is not str or not _ID.fullmatch(resource_id):
                raise TransactionAssemblyError("invalid_resource_id")
            logical_id = f"{resource_type}/{resource_id}"
            if logical_id in logical_ids:
                raise TransactionAssemblyError("duplicate_resource_target")
            logical_ids.add(logical_id)
        request = _request(resource_type, resource_id, kind, predicate, version)
        if kind == "update" or predicate is not None:
            # Different version preconditions cannot justify writing the same
            # target twice in one atomic transaction.
            target_key = f"{request['method']}:{request['url']}:{predicate or ''}"
            if target_key in requests:
                raise TransactionAssemblyError("duplicate_request_target")
            requests.add(target_key)
        resource_digest = _digest(
            _serialize({"resource": resource, "request": request}, limits.max_bytes)
        )
        urn = "urn:uuid:" + str(
            uuid5(
                _NAMESPACE,
                f"write:{approval.action_digest}:{index}:{resource_digest}",
            )
        )
        prepared.append({"fullUrl": urn, "resource": resource, "request": request})

    reference_map = {
        f"{entry['resource']['resourceType']}/{entry['resource']['id']}": entry[
            "fullUrl"
        ]
        for entry in prepared
        if "id" in entry["resource"]
    }
    for prepared_entry in prepared:
        _rewrite_references(prepared_entry["resource"], reference_map)
    recorded = _recorded(clock)
    provenance = _provenance(prepared, approval, recorded)
    provenance_urn = "urn:uuid:" + str(
        uuid5(
            _NAMESPACE,
            "provenance:" + _digest(_serialize(provenance, limits.max_bytes)),
        )
    )
    prepared.append(
        {
            "fullUrl": provenance_urn,
            "resource": provenance,
            "request": {"method": "POST", "url": "Provenance"},
        }
    )
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": prepared}
    serialized = _serialize(bundle, limits.max_bytes)
    try:
        valid = validation_result(bundle, "R4").valid
    except Exception:
        raise TransactionAssemblyError("invalid_r4_structure") from None
    if not valid:
        raise TransactionAssemblyError("invalid_r4_structure")
    return AssembledTransaction(serialized, _digest(serialized), len(prepared))


def _request(
    resource_type: str,
    resource_id: str | None,
    kind: str,
    predicate: str | None,
    version: str | None,
) -> dict[str, str]:
    if type(kind) is not str or kind not in {"create", "update"}:
        raise TransactionAssemblyError("unsupported_write_kind")
    if predicate is not None:
        if (
            type(predicate) is not str
            or len(predicate) > 4096
            or not _QUERY.fullmatch(predicate)
            or re.search(r"%(?![0-9A-Fa-f]{2})", predicate)
        ):
            raise TransactionAssemblyError("invalid_conditional_predicate")
        try:
            pairs = parse_qsl(
                predicate,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=64,
                errors="strict",
            )
        except (ValueError, UnicodeError):
            raise TransactionAssemblyError("invalid_conditional_predicate") from None
        if not pairs or any(
            not _PARAMETER.fullmatch(key)
            or not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            for key, value in pairs
        ):
            raise TransactionAssemblyError("invalid_conditional_predicate")
    if version is not None and (type(version) is not str or not _ID.fullmatch(version)):
        raise TransactionAssemblyError("invalid_expected_version")
    if kind == "create":
        if version is not None:
            raise TransactionAssemblyError("create_version_not_supported")
        request = {"method": "POST", "url": resource_type}
        if predicate is not None:
            request["ifNoneExist"] = predicate
        return request
    if predicate is None and resource_id is None:
        raise TransactionAssemblyError("update_target_required")
    request = {
        "method": "PUT",
        "url": f"{resource_type}?{predicate}"
        if predicate is not None
        else f"{resource_type}/{resource_id}",
    }
    if version is not None:
        request["ifMatch"] = f'W/"{version}"'
    return request


def _snapshot_resource(resource: Any) -> dict[str, Any]:
    def check(node: Any, depth: int = 0) -> None:
        if depth > 64:
            raise TransactionAssemblyError("invalid_resource_json")
        if type(node) is dict:
            for key, value in node.items():
                if type(key) is not str:
                    raise TransactionAssemblyError("invalid_resource_json")
                check(value, depth + 1)
        elif type(node) is list:
            for value in node:
                check(value, depth + 1)
        elif node is not None and type(node) not in (str, int, float, bool):
            raise TransactionAssemblyError("invalid_resource_json")

    try:
        if not isinstance(resource, Mapping):
            raise TransactionAssemblyError("invalid_resource_json")
        resource = dict(resource)
        check(resource)
        return json.loads(json.dumps(resource, allow_nan=False, ensure_ascii=False))
    except Exception:
        raise TransactionAssemblyError("invalid_resource_json") from None


def _serialize(value: Any, max_bytes: int) -> bytes:
    parts: list[bytes] = []
    size = 0
    try:
        for part in json.JSONEncoder(
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).iterencode(value):
            encoded = part.encode("utf-8")
            size += len(encoded)
            if size > max_bytes:
                raise TransactionAssemblyError("size_limit_exceeded")
            parts.append(encoded)
    except (UnicodeError, TypeError, ValueError) as exc:
        if isinstance(exc, TransactionAssemblyError):
            raise
        raise TransactionAssemblyError("invalid_resource_json") from None
    return b"".join(parts)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _rewrite_references(node: Any, references: dict[str, str]) -> None:
    if isinstance(node, dict):
        reference = node.get("reference")
        if isinstance(reference, str) and reference in references:
            node["reference"] = references[reference]
        for value in node.values():
            _rewrite_references(value, references)
    elif isinstance(node, list):
        for value in node:
            _rewrite_references(value, references)


def _is_digest(value: Any) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _is_evidence_reference(value: Any) -> bool:
    if type(value) is not str:
        return False
    if value.startswith("urn:sha256:"):
        return _is_digest(value[4:])
    if not value.startswith("urn:uuid:"):
        return False
    try:
        return str(UUID(value[9:])) == value[9:]
    except ValueError:
        return False


def _recorded(clock: Callable[[], datetime]) -> str:
    try:
        instant = clock()
        if type(instant) is not datetime or instant.utcoffset() is None:
            raise ValueError
        return (
            instant.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    except Exception:
        raise TransactionAssemblyError("invalid_clock") from None


def _provenance(
    entries: list[dict[str, Any]], approval: TransactionApproval, recorded: str
) -> dict[str, Any]:
    def entity(system: str, value: str) -> dict[str, Any]:
        return {
            "role": "source",
            "what": {"identifier": {"system": _SYSTEM + system, "value": value}},
        }

    return {
        "resourceType": "Provenance",
        "target": [{"reference": entry["fullUrl"]} for entry in entries],
        "recorded": recorded,
        "extension": [
            {
                "url": "https://openmed.dev/fhir/StructureDefinition/reviewer-role",
                "valueCode": approval.reviewer_role.value,
            }
        ],
        "activity": {
            "coding": [
                {
                    "system": "https://openmed.dev/fhir/CodeSystem/audit-activity",
                    "code": "approved-write",
                }
            ]
        },
        "agent": [
            {
                "who": {
                    "type": "Device",
                    "identifier": {"system": _SYSTEM + "software", "value": "openmed"},
                },
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                            "code": "author",
                        }
                    ]
                },
            }
        ],
        "entity": [
            entity("approval-receipt-digest", approval.receipt_digest),
            entity("action-digest", approval.action_digest),
            *(
                entity("evidence-reference", reference)
                for reference in approval.evidence_references
            ),
        ],
    }
