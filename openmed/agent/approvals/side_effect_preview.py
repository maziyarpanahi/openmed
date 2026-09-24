"""Offline, value-free previews for explicitly typed clinical write intents.

The caller owns fresh reads, approval-token validation, and atomic target-system
preconditions. This module never dispatches a write or records clinical values.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
from uuid import UUID

_RESOURCE_TYPE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_RESOURCE_HANDLE = re.compile(r"res_[0-9a-f]{32}\Z")
_FIELD_PATH = re.compile(r"[a-z][A-Za-z0-9]*(?:\.[a-z][A-Za-z0-9]*)*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DOMAIN = b"openmed.agent.side-effect-preview.v1\x00"
_MISSING = object()


class PreviewError(ValueError):
    """A value-free validation or stale-preview failure."""


class WriteKind(str, Enum):
    """Closed set of resource write interactions."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class WorkflowState(str, Enum):
    """Public workflow lifecycle states displayed in a preview."""

    PROPOSED = "proposed"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, repr=False)
class ResourceWrite:
    """One in-memory resource mutation with a random, nonclinical handle.

    ``before`` and ``after`` are flat mappings of trusted schema field paths to
    JSON values. Field paths must be schema labels, never payload-derived keys.
    Nested values are compared as atomic values. All input values are copied
    into immutable mappings and hidden from representations.
    """

    kind: WriteKind
    resource_type: str
    handle: str
    before: Mapping[str, Any] | None = field(repr=False)
    after: Mapping[str, Any] | None = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not WriteKind:
            raise PreviewError("invalid_write_kind")
        if type(self.resource_type) is not str or not _RESOURCE_TYPE.fullmatch(
            self.resource_type
        ):
            raise PreviewError("invalid_resource_type")
        if type(self.handle) is not str or not _RESOURCE_HANDLE.fullmatch(self.handle):
            raise PreviewError("invalid_resource_handle")
        if (self.before is None) != (self.kind is WriteKind.CREATE):
            raise PreviewError("invalid_before_state")
        if (self.after is None) != (self.kind is WriteKind.DELETE):
            raise PreviewError("invalid_after_state")
        _canonical_fields(self.before)
        _canonical_fields(self.after)
        object.__setattr__(self, "before", _freeze_fields(self.before))
        object.__setattr__(self, "after", _freeze_fields(self.after))

    def __repr__(self) -> str:
        return "ResourceWrite(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class WriteIntent:
    """An action-scoped set of proposed writes and workflow transition."""

    action_id: str
    writes: tuple[ResourceWrite, ...]
    workflow_before: WorkflowState
    workflow_after: WorkflowState

    def __post_init__(self) -> None:
        try:
            action = UUID(self.action_id)
        except (ValueError, TypeError, AttributeError):
            raise PreviewError("invalid_action_id") from None
        if str(action) != self.action_id or action.version != 4:
            raise PreviewError("invalid_action_id")
        if type(self.writes) is not tuple or not self.writes:
            raise PreviewError("invalid_writes")
        if any(type(write) is not ResourceWrite for write in self.writes):
            raise PreviewError("invalid_writes")
        if len({write.handle for write in self.writes}) != len(self.writes):
            raise PreviewError("duplicate_resource_handle")
        if (
            type(self.workflow_before) is not WorkflowState
            or type(self.workflow_after) is not WorkflowState
        ):
            raise PreviewError("invalid_workflow_state")

    def __repr__(self) -> str:
        return "WriteIntent(<redacted>)"


@dataclass(frozen=True, slots=True)
class FieldPreview:
    """A trusted field path with redacted before and after value states."""

    path: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class ResourcePreview:
    """Value-free preview of one resource mutation."""

    kind: WriteKind
    resource_type: str
    handle: str
    fields: tuple[FieldPreview, ...]


@dataclass(frozen=True, slots=True)
class SideEffectPreview:
    """Deterministic review view and keyed commitment to the full write intent."""

    digest: str
    resources: tuple[ResourcePreview, ...]
    workflow_before: WorkflowState
    workflow_after: WorkflowState


def _canonical_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is list or type(value) is tuple:
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise PreviewError("invalid_value")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    raise PreviewError("invalid_value")


def _canonical_fields(fields: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if fields is None:
        return None
    if not isinstance(fields, Mapping):
        raise PreviewError("invalid_fields")
    if any(
        type(path) is not str or not _FIELD_PATH.fullmatch(path) or len(path) > 160
        for path in fields
    ):
        raise PreviewError("invalid_field_path")
    ordered = {path: _canonical_value(fields[path]) for path in sorted(fields)}
    try:
        json.dumps(ordered, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        raise PreviewError("invalid_value") from None
    return ordered


def _freeze_fields(fields: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if fields is None:
        return None

    def freeze(value: Any) -> Any:
        if type(value) is dict:
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if type(value) is list:
            return tuple(freeze(item) for item in value)
        return value

    canonical = _canonical_fields(fields)
    assert canonical is not None
    return MappingProxyType({path: freeze(value) for path, value in canonical.items()})


def _key(secret: bytes) -> bytes:
    if type(secret) is not bytes or len(secret) < 32:
        raise PreviewError("invalid_preview_key")
    return secret


def render_side_effect_preview(
    intent: WriteIntent, *, secret: bytes
) -> SideEffectPreview:
    """Render a deterministic redacted preview without performing I/O.

    The secret must be stable for review and dispatch and remain private. It
    prevents low-entropy clinical values from being guessed from the digest.
    The caller must present the typed paths and opaque handles with its trusted
    local resource mapping when a reviewer inspects this result.
    """

    if type(intent) is not WriteIntent:
        raise PreviewError("invalid_intent")
    key = _key(secret)
    writes: list[dict[str, Any]] = []
    resources: list[ResourcePreview] = []
    for write in intent.writes:
        before = _canonical_fields(write.before)
        after = _canonical_fields(write.after)
        paths = sorted(set(before or ()) | set(after or ()))
        fields = tuple(
            FieldPreview(
                path=path,
                before="absent" if before is None or path not in before else "redacted",
                after="absent" if after is None or path not in after else "redacted",
            )
            for path in paths
            if (before or {}).get(path, _MISSING) != (after or {}).get(path, _MISSING)
        )
        resources.append(
            ResourcePreview(write.kind, write.resource_type, write.handle, fields)
        )
        writes.append(
            {
                "kind": write.kind.value,
                "resource_type": write.resource_type,
                "handle": write.handle,
                "before": before,
                "after": after,
            }
        )
    payload = {
        "action_id": intent.action_id,
        "writes": writes,
        "workflow_before": intent.workflow_before.value,
        "workflow_after": intent.workflow_after.value,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hmac.new(key, _DOMAIN + canonical.encode(), hashlib.sha256).hexdigest()
    return SideEffectPreview(
        digest, tuple(resources), intent.workflow_before, intent.workflow_after
    )


def require_current_preview(
    intent: WriteIntent,
    *,
    observed_before: tuple[Mapping[str, Any] | None, ...],
    observed_workflow_state: WorkflowState,
    approved_digest: str,
    secret: bytes,
) -> None:
    """Reject execution if a fresh read or proposed action differs from review.

    ``observed_before`` must come from a fresh target-system read in write order.
    A successful return authorizes no write on its own: validate the separate
    approval token and use an atomic version precondition at the target system.
    """

    if type(intent) is not WriteIntent:
        raise PreviewError("invalid_intent")
    if type(observed_before) is not tuple or len(observed_before) != len(intent.writes):
        raise PreviewError("invalid_observed_state")
    if type(observed_workflow_state) is not WorkflowState:
        raise PreviewError("invalid_observed_state")
    if type(approved_digest) is not str or not _DIGEST.fullmatch(approved_digest):
        raise PreviewError("invalid_approved_digest")
    current = WriteIntent(
        action_id=intent.action_id,
        writes=tuple(
            ResourceWrite(
                write.kind,
                write.resource_type,
                write.handle,
                observed,
                write.after,
            )
            for write, observed in zip(intent.writes, observed_before, strict=True)
        ),
        workflow_before=observed_workflow_state,
        workflow_after=intent.workflow_after,
    )
    if not hmac.compare_digest(
        render_side_effect_preview(current, secret=secret).digest, approved_digest
    ):
        raise PreviewError("stale_preview")


__all__ = [
    "FieldPreview",
    "PreviewError",
    "ResourcePreview",
    "ResourceWrite",
    "SideEffectPreview",
    "WorkflowState",
    "WriteIntent",
    "WriteKind",
    "render_side_effect_preview",
    "require_current_preview",
]
