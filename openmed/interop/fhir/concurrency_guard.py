"""Offline optimistic concurrency checks for explicit FHIR R4 updates.

The caller supplies read evidence and performs any HTTP request. No resource
payload, server response body, or identifier is retained by guard results.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_VERSION = re.compile(r"[A-Za-z0-9.-]{1,64}\Z")
_MISSING = object()


class FHIRWriteConflict(Exception):
    """A value-free conflict requiring a fresh read and human review."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in {
            "stale_evidence",
            "server_conflict",
            "precondition_failed",
            "precondition_required",
        }:
            raise ValueError("conflict reason is invalid")
        self.reason_code = reason_code
        super().__init__(f"FHIR update conflict: {reason_code}")


@dataclass(frozen=True, slots=True, repr=False)
class VersionEvidence:
    """Version and timestamp observed on the same FHIR resource read.

    Both values are sensitive request metadata and omitted from representations.
    A timestamp alone is never an atomic write precondition.
    """

    version_id: str = field(repr=False)
    last_updated: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.version_id) is not str or not _VERSION.fullmatch(self.version_id):
            raise ValueError("FHIR version evidence is invalid")
        if type(self.last_updated) is not str or not self.last_updated:
            raise ValueError("FHIR last-updated evidence is invalid")
        try:
            parsed = datetime.fromisoformat(self.last_updated.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("FHIR last-updated evidence is invalid") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("FHIR last-updated evidence is invalid")

    @classmethod
    def from_resource(cls, resource: Mapping[str, Any]) -> VersionEvidence:
        """Extract both required fields from one in-memory FHIR resource."""

        if not isinstance(resource, Mapping):
            raise TypeError("FHIR resource must be a mapping")
        meta = resource.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("FHIR version evidence is missing")
        return cls(meta.get("versionId"), meta.get("lastUpdated"))

    def __repr__(self) -> str:
        return "VersionEvidence(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class UpdatePrecondition:
    """A request header to apply to one explicitly authorized FHIR update."""

    if_match: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.if_match) is not str or not re.fullmatch(
            r'W/"[A-Za-z0-9.-]{1,64}"', self.if_match
        ):
            raise ValueError("FHIR update precondition is invalid")

    def __repr__(self) -> str:
        return "UpdatePrecondition(<redacted>)"


def guard_update(
    expected: VersionEvidence, observed: VersionEvidence
) -> UpdatePrecondition:
    """Require current evidence to match the original read before a PUT.

    The caller must obtain ``observed`` from a fresh read immediately before
    sending the request and send ``if_match`` as its ``If-Match`` header. The
    server's atomic version check closes the race after that read.
    """

    if not isinstance(expected, VersionEvidence) or not isinstance(
        observed, VersionEvidence
    ):
        raise TypeError("FHIR update requires version evidence")
    if expected != observed:
        raise FHIRWriteConflict("stale_evidence")
    return UpdatePrecondition(f'W/"{expected.version_id}"')


def require_no_server_conflict(status_code: int) -> None:
    """Map FHIR write conflict statuses without reading response content."""

    if type(status_code) is not int or not 100 <= status_code <= 599:
        raise ValueError("FHIR response status is invalid")
    if status_code == 409:
        raise FHIRWriteConflict("server_conflict")
    if status_code == 412:
        raise FHIRWriteConflict("precondition_failed")
    if status_code == 428:
        raise FHIRWriteConflict("precondition_required")


@dataclass(frozen=True, slots=True)
class ThreeWayConflictSummary:
    """Count-only comparison with no resource paths or clinical values."""

    server_changes: int
    proposed_changes: int
    overlapping_changes: int
    divergent_changes: int
    already_applied_changes: int

    def __post_init__(self) -> None:
        values = (
            self.server_changes,
            self.proposed_changes,
            self.overlapping_changes,
            self.divergent_changes,
            self.already_applied_changes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("FHIR conflict summary is invalid")


def summarize_conflict(
    base: Mapping[str, Any],
    current: Mapping[str, Any],
    proposed: Mapping[str, Any],
) -> ThreeWayConflictSummary:
    """Compare three in-memory resource snapshots without exposing content.

    Mapping fields are compared recursively; arrays are atomic because list
    positions and contents can shift. ``meta`` is excluded from clinical
    change counts. This report never merges or approves clinical content.
    """

    resources = (base, current, proposed)
    if any(not isinstance(resource, Mapping) for resource in resources):
        raise TypeError("FHIR resources must be mappings")
    identity = (base.get("resourceType"), base.get("id"))
    if not all(type(value) is str and value for value in identity) or any(
        (item.get("resourceType"), item.get("id")) != identity for item in resources
    ):
        raise ValueError("FHIR resource identity is invalid")

    def changes(
        left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> set[tuple[str, ...]]:
        result: set[tuple[str, ...]] = set()

        def walk(before: Any, after: Any, path: tuple[str, ...]) -> None:
            if isinstance(before, Mapping) and isinstance(after, Mapping):
                for key in before.keys() | after.keys():
                    if type(key) is not str:
                        raise ValueError("FHIR resource field is invalid")
                    if not path and key == "meta":
                        continue
                    walk(
                        before.get(key, _MISSING),
                        after.get(key, _MISSING),
                        (*path, key),
                    )
            elif before != after:
                result.add(path)

        walk(left, right, ())
        return result

    server = changes(base, current)
    proposal = changes(base, proposed)
    overlap = {
        path
        for path in proposal
        if any(
            path[: len(server_path)] == server_path or server_path[: len(path)] == path
            for server_path in server
        )
    }

    def value_at(resource: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = resource
        for key in path:
            if not isinstance(value, Mapping):
                return _MISSING
            value = value.get(key, _MISSING)
        return value

    already_applied = sum(
        value_at(current, path) == value_at(proposed, path) for path in overlap
    )
    return ThreeWayConflictSummary(
        server_changes=len(server),
        proposed_changes=len(proposal),
        overlapping_changes=len(overlap),
        divergent_changes=len(overlap) - already_applied,
        already_applied_changes=already_applied,
    )
