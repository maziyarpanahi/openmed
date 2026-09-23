"""Deterministic forward migrations for value-free clinical review packets."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openmed.structured.store import StoreResult, StoreState

from .review_transitions import (
    CLINICAL_REVIEW_COMPATIBILITY_POLICY,
    CLINICAL_REVIEW_PACKET_SCHEMA_VERSION,
    ClinicalReviewError,
    ClinicalReviewPacket,
)

SUPPORTED_REVIEW_PACKET_VERSIONS = ("1.0.0", "1.1.0")


@dataclass(frozen=True, slots=True)
class ReviewPacketMigrationReport:
    """Value-free field-level report for one forward migration."""

    source_version: str
    target_version: str
    added_fields: tuple[str, ...]
    preserved_transition_count: int
    lossy: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return counts and field names only."""

        return {
            "added_fields": list(self.added_fields),
            "lossy": self.lossy,
            "preserved_transition_count": self.preserved_transition_count,
            "source_version": self.source_version,
            "target_version": self.target_version,
        }


@dataclass(frozen=True, slots=True)
class ReviewPacketMigration:
    """A migrated packet and its value-free report."""

    packet: ClinicalReviewPacket
    report: ReviewPacketMigrationReport


def migrate_review_packet(
    payload: Mapping[str, Any],
    *,
    target_version: str = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION,
) -> StoreResult[ReviewPacketMigration]:
    """Migrate a supported packet forward without external access.

    Version 1.1 adds an explicit compatibility policy, extension container,
    and redundant transition-id list for append-only integrity checks.
    """

    source_version = str(payload.get("schema_version") or "")
    if target_version != CLINICAL_REVIEW_PACKET_SCHEMA_VERSION:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "review_packet_target_unsupported"
        )
    if source_version not in SUPPORTED_REVIEW_PACKET_VERSIONS:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "review_packet_source_unsupported"
        )
    migrated = copy.deepcopy(dict(payload))
    added: list[str] = []
    transitions = migrated.get("transitions", ())
    if not isinstance(transitions, (list, tuple)):
        return StoreResult.outcome(StoreState.FAILURE, "review_packet_invalid")
    if source_version == "1.0.0":
        defaults = {
            "compatibility_policy": CLINICAL_REVIEW_COMPATIBILITY_POLICY,
            "extensions": {},
            "transition_ids": [
                item.get("event_id")
                for item in transitions
                if isinstance(item, Mapping)
            ],
        }
        for name, value in defaults.items():
            if name not in migrated:
                migrated[name] = value
                added.append(name)
        migrated["schema_version"] = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION
        for transition in transitions:
            if not isinstance(transition, Mapping):
                return StoreResult.outcome(StoreState.FAILURE, "review_packet_invalid")
            if transition.get("schema_version") not in {"1.0.0", "1.1.0"}:
                return StoreResult.outcome(
                    StoreState.UNSUPPORTED, "review_transition_source_unsupported"
                )
            if isinstance(transition, dict):
                transition["schema_version"] = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION
                transition.setdefault(
                    "compatibility_policy", CLINICAL_REVIEW_COMPATIBILITY_POLICY
                )
            else:
                normalized = dict(transition)
                normalized["schema_version"] = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION
                normalized.setdefault(
                    "compatibility_policy", CLINICAL_REVIEW_COMPATIBILITY_POLICY
                )
                transitions = [
                    normalized if item is transition else item for item in transitions
                ]
        migrated["transitions"] = transitions
    try:
        packet = ClinicalReviewPacket.from_dict(migrated)
    except ClinicalReviewError:
        return StoreResult.outcome(StoreState.FAILURE, "review_packet_invalid")
    report = ReviewPacketMigrationReport(
        source_version=source_version,
        target_version=target_version,
        added_fields=tuple(sorted(added)),
        preserved_transition_count=len(packet.transitions),
    )
    return StoreResult.success(ReviewPacketMigration(packet=packet, report=report))


__all__ = [
    "SUPPORTED_REVIEW_PACKET_VERSIONS",
    "ReviewPacketMigration",
    "ReviewPacketMigrationReport",
    "migrate_review_packet",
]
