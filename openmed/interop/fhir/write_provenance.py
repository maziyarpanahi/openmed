"""Offline, value-free field lineage gate for proposed FHIR writes.

The caller supplies verified evidence and an approval verifier. This module
does not issue tokens, contact a FHIR server, or dispatch a write.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from openmed.agent.approvals.side_effect_preview import (
    WriteIntent,
    WriteKind,
    render_side_effect_preview,
)
from openmed.interop.fhir.conditional_writes import (
    ConditionalWriteKind,
    ConditionalWritePlan,
)

__all__ = [
    "ApprovalBinding",
    "EvidenceSpan",
    "FieldLineage",
    "ProvenanceError",
    "ProvenanceManifest",
    "ProvenanceRecord",
    "require_provenance_target",
    "require_write_provenance",
    "write_provenance_digest",
]

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_HANDLE = re.compile(r"res_[0-9a-f]{32}\Z")
_PATH = re.compile(r"[a-z][A-Za-z0-9]*(?:\.[a-z][A-Za-z0-9]*)*\Z")
_REFERENCE = re.compile(r"([A-Z][A-Za-z0-9]{0,63})/([A-Za-z0-9.\-]{1,64})\Z")
_TARGET_DOMAIN = b"openmed.fhir.provenance-target.v1\x00"
_ACTION_DOMAIN = b"openmed.fhir.write-provenance.v1\x00"


class ProvenanceError(ValueError):
    """Value-free rejection of an incomplete or unapproved write."""


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """A keyed source digest and half-open offsets, without source text."""

    digest: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not _is_digest(self.digest):
            raise ProvenanceError("invalid_evidence_digest")
        if type(self.start) is not int or type(self.end) is not int:
            raise ProvenanceError("invalid_evidence_span")
        if self.start < 0 or self.end <= self.start:
            raise ProvenanceError("invalid_evidence_span")


@dataclass(frozen=True, slots=True)
class FieldLineage:
    """Evidence and transformation chain for one changed schema field."""

    resource_handle: str
    field_path: str
    evidence: tuple[EvidenceSpan, ...]
    step_digests: tuple[str, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if type(self.resource_handle) is not str or not _HANDLE.fullmatch(
            self.resource_handle
        ):
            raise ProvenanceError("invalid_resource_handle")
        if (
            type(self.field_path) is not str
            or len(self.field_path) > 160
            or not _PATH.fullmatch(self.field_path)
        ):
            raise ProvenanceError("invalid_field_path")
        if (
            type(self.evidence) is not tuple
            or not self.evidence
            or any(type(span) is not EvidenceSpan for span in self.evidence)
            or len(set(self.evidence)) != len(self.evidence)
        ):
            raise ProvenanceError("invalid_evidence_chain")
        if (
            type(self.step_digests) is not tuple
            or not self.step_digests
            or any(not _is_digest(item) for item in self.step_digests)
            or len(set(self.step_digests)) != len(self.step_digests)
        ):
            raise ProvenanceError("invalid_transformation_chain")
        if not _is_digest(self.policy_digest):
            raise ProvenanceError("invalid_policy_digest")


@dataclass(frozen=True, slots=True)
class ApprovalBinding:
    """Value-free receipt bound to the complete write-provenance digest."""

    action_digest: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if not _is_digest(self.action_digest) or not _is_digest(self.receipt_digest):
            raise ProvenanceError("invalid_approval_binding")


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """One field's links to evidence, approval, and a FHIR target commitment."""

    resource_handle: str
    field_path: str
    evidence: tuple[EvidenceSpan, ...]
    step_digests: tuple[str, ...]
    policy_digest: str
    approval_receipt_digest: str
    target_digest: str
    plan_key: str


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Deterministically ordered, content-free write provenance packet."""

    action_digest: str
    records: tuple[ProvenanceRecord, ...]


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _target_digest(reference: str, resource_type: str, secret: bytes) -> str:
    if type(reference) is not str:
        raise ProvenanceError("invalid_provenance_target")
    match = _REFERENCE.fullmatch(reference)
    if match is None or match.group(1) != resource_type:
        try:
            urn = UUID(reference.removeprefix("urn:uuid:"))
        except (ValueError, AttributeError):
            raise ProvenanceError("invalid_provenance_target") from None
        if not reference.startswith("urn:uuid:") or str(urn) != reference[9:]:
            raise ProvenanceError("invalid_provenance_target")
    return hmac.new(
        secret, _TARGET_DOMAIN + reference.encode("ascii"), hashlib.sha256
    ).hexdigest()


def require_provenance_target(
    record: ProvenanceRecord,
    reference: str,
    resource_type: str,
    *,
    secret: bytes,
) -> None:
    """Confirm a prospective FHIR ``Provenance.target`` matches its field link.

    Call before committing the transaction containing the target reference.
    The reference is never retained or included in an exception.
    """

    if type(record) is not ProvenanceRecord:
        raise ProvenanceError("invalid_provenance_record")
    if type(secret) is not bytes or len(secret) < 32:
        raise ProvenanceError("invalid_provenance_key")
    digest = _target_digest(reference, resource_type, secret)
    if not hmac.compare_digest(digest, record.target_digest):
        raise ProvenanceError("provenance_target_mismatch")


def require_write_provenance(
    intent: WriteIntent,
    plans: Mapping[str, ConditionalWritePlan],
    lineage: Sequence[FieldLineage],
    targets: Mapping[str, str],
    approval: ApprovalBinding,
    *,
    secret: bytes,
    verify_approval: Callable[[ApprovalBinding], bool],
) -> ProvenanceManifest:
    """Reject missing or mismatched lineage before an authorized commit.

    Args:
        intent: The exact proposed resource writes reviewed by a human.
        plans: One conditional plan per resource handle.
        lineage: One source and transformation chain per changed field.
        targets: FHIR ``ResourceType/id`` or transaction ``urn:uuid`` targets.
            References remain in caller memory and are committed with HMAC.
        approval: Receipt bound to the complete write-provenance digest.
        secret: The private preview and target-commitment key (at least 32 bytes).
        verify_approval: Caller-owned, local verifier that checks the token's
            decision, action binding, expiry, and single use before returning True.

    Returns:
        A value-free manifest ordered by resource handle and field path.

    Raises:
        ProvenanceError: If any field, target, plan, or approval is incomplete.
    """

    if type(approval) is not ApprovalBinding or not callable(verify_approval):
        raise ProvenanceError("invalid_approval_binding")

    action_digest, records = _prepare(intent, plans, lineage, targets, secret)
    if not hmac.compare_digest(action_digest, approval.action_digest):
        raise ProvenanceError("approval_action_mismatch")

    try:
        verified = verify_approval(approval)
    except Exception:
        raise ProvenanceError("approval_verification_failed") from None
    if verified is not True:
        raise ProvenanceError("approval_verification_failed")
    return ProvenanceManifest(
        action_digest=action_digest,
        records=tuple(
            replace(record, approval_receipt_digest=approval.receipt_digest)
            for record in records
        ),
    )


def write_provenance_digest(
    intent: WriteIntent,
    plans: Mapping[str, ConditionalWritePlan],
    lineage: Sequence[FieldLineage],
    targets: Mapping[str, str],
    *,
    secret: bytes,
) -> str:
    """Build the content-free digest to display and bind in an approval token.

    Use the same arguments again at the pre-commit gate. A changed resource,
    plan, source chain, policy, or target produces a different digest.
    """

    digest, _ = _prepare(intent, plans, lineage, targets, secret)
    return digest


def _prepare(
    intent: WriteIntent,
    plans: Mapping[str, ConditionalWritePlan],
    lineage: Sequence[FieldLineage],
    targets: Mapping[str, str],
    secret: bytes,
) -> tuple[str, tuple[ProvenanceRecord, ...]]:
    if type(intent) is not WriteIntent:
        raise ProvenanceError("invalid_write_intent")
    if type(secret) is not bytes or len(secret) < 32:
        raise ProvenanceError("invalid_provenance_key")
    if not isinstance(plans, Mapping) or not isinstance(targets, Mapping):
        raise ProvenanceError("invalid_write_bindings")
    if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
        raise ProvenanceError("invalid_lineage")

    preview = render_side_effect_preview(intent, secret=secret)

    handles = {write.handle for write in intent.writes}
    if set(plans) != handles or set(targets) != handles:
        raise ProvenanceError("incomplete_write_bindings")
    changed: set[tuple[str, str]] = set()
    by_handle = {write.handle: write for write in intent.writes}
    for resource in preview.resources:
        write = by_handle[resource.handle]
        plan = plans[resource.handle]
        if (
            type(plan) is not ConditionalWritePlan
            or write.kind not in (WriteKind.CREATE, WriteKind.UPDATE)
            or plan.kind.value != write.kind.value
            or plan.resource_type != write.resource_type
        ):
            raise ProvenanceError("write_plan_mismatch")
        if not resource.fields:
            raise ProvenanceError("empty_write")
        changed.update((resource.handle, field.path) for field in resource.fields)

    records: list[ProvenanceRecord] = []
    seen: set[tuple[str, str]] = set()
    for link in lineage:
        if type(link) is not FieldLineage:
            raise ProvenanceError("invalid_lineage")
        key = (link.resource_handle, link.field_path)
        if key in seen or key not in changed:
            raise ProvenanceError("unexpected_field_lineage")
        seen.add(key)
        write = by_handle[link.resource_handle]
        records.append(
            ProvenanceRecord(
                resource_handle=link.resource_handle,
                field_path=link.field_path,
                evidence=link.evidence,
                step_digests=link.step_digests,
                policy_digest=link.policy_digest,
                approval_receipt_digest="",
                target_digest=_target_digest(
                    targets[link.resource_handle], write.resource_type, secret
                ),
                plan_key=plans[link.resource_handle].idempotency_key,
            )
        )
    if seen != changed:
        raise ProvenanceError("missing_field_lineage")

    ordered = tuple(
        sorted(records, key=lambda item: (item.resource_handle, item.field_path))
    )
    canonical = json.dumps(
        {
            "preview": preview.digest,
            "fields": [
                {
                    "handle": record.resource_handle,
                    "path": record.field_path,
                    "evidence": [
                        [span.digest, span.start, span.end] for span in record.evidence
                    ],
                    "steps": record.step_digests,
                    "policy": record.policy_digest,
                    "target": record.target_digest,
                    "plan": record.plan_key,
                }
                for record in ordered
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hmac.new(
        secret, _ACTION_DOMAIN + canonical.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return digest, ordered
