"""Deterministic, lossless migrations for clinical review packet mappings."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

REVIEW_PACKET_SCHEMA_VERSION: Final = 2
REVIEW_PACKET_MIGRATION_REPORT_SCHEMA_VERSION: Final = 1
SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS: Final = (1, 2)


class ReviewPacketMigrationError(ValueError):
    """Base error for an unsupported or malformed packet migration."""


class LossyReviewPacketMigrationError(ReviewPacketMigrationError):
    """Raised when migration would remove or overwrite packet information."""


@dataclass(frozen=True, order=True)
class ReviewPacketMigrationChange:
    """One value-free field operation applied during migration."""

    field_path: str
    operation: str

    def __post_init__(self) -> None:
        if not self.field_path.startswith("/"):
            raise ValueError("migration change field_path must be absolute")
        if self.operation not in {"add", "replace"}:
            raise ValueError("unsupported migration change operation")

    def to_dict(self) -> dict[str, str]:
        """Return the field path and operation without field values."""

        return {"field_path": self.field_path, "operation": self.operation}


@dataclass(frozen=True)
class ReviewPacketMigrationReport:
    """Field-level, value-free report for one forward migration."""

    source_version: int
    target_version: int
    changes: tuple[ReviewPacketMigrationChange, ...]
    schema_version: int = REVIEW_PACKET_MIGRATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_PACKET_MIGRATION_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported review packet migration report version")
        object.__setattr__(self, "changes", tuple(sorted(set(self.changes))))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report that contains no packet values."""

        return {
            "schema_version": self.schema_version,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "changes": [change.to_dict() for change in self.changes],
        }


@dataclass(frozen=True)
class ReviewPacketMigrationResult:
    """Migrated packet plus its safe report.

    Packet content is hidden from ``repr`` so accidental exception/log output
    does not expose review material.
    """

    packet: Mapping[str, Any] = field(repr=False)
    report: ReviewPacketMigrationReport

    def __iter__(self) -> Iterator[Any]:
        """Allow explicit unpacking as ``packet, report``."""

        yield self.packet
        yield self.report


def migrate_review_packet(
    packet: Mapping[str, Any],
    *,
    target_version: int = REVIEW_PACKET_SCHEMA_VERSION,
) -> ReviewPacketMigrationResult:
    """Migrate a review packet forward without network or external state.

    Version 2 adds ``safety.privacy_scan_required``. The migration never
    overwrites an existing packet field: an incompatible ``safety`` field or a
    pre-existing false/non-boolean requirement is rejected as potentially
    lossy. Backward migrations are also rejected.
    """

    if not isinstance(packet, Mapping):
        raise TypeError("review packet must be a mapping")
    source_version = _packet_version(packet.get("schema_version"))
    target = _target_version(target_version)

    if source_version > target:
        raise LossyReviewPacketMigrationError(
            "backward review packet migrations are not supported"
        )

    try:
        migrated: dict[str, Any] = copy.deepcopy(dict(packet))
    except Exception:
        raise ReviewPacketMigrationError(
            "review packet could not be copied for migration"
        ) from None

    changes: list[ReviewPacketMigrationChange] = []
    version = source_version
    while version < target:
        if version == 1:
            _migrate_v1_to_v2(migrated, changes)
            version = 2
            continue
        raise ReviewPacketMigrationError(
            "review packet has no supported forward migration path"
        )

    if target == 2:
        _validate_v2(migrated)

    report = ReviewPacketMigrationReport(
        source_version=source_version,
        target_version=target,
        changes=tuple(changes),
    )
    return ReviewPacketMigrationResult(packet=migrated, report=report)


def _migrate_v1_to_v2(
    packet: dict[str, Any], changes: list[ReviewPacketMigrationChange]
) -> None:
    safety = packet.get("safety")
    if "safety" not in packet:
        packet["safety"] = {"privacy_scan_required": True}
        changes.append(ReviewPacketMigrationChange("/safety", "add"))
    else:
        if not isinstance(safety, Mapping):
            raise LossyReviewPacketMigrationError(
                "review packet migration would overwrite an incompatible safety field"
            )
        safe_copy = copy.deepcopy(dict(safety))
        if "privacy_scan_required" not in safe_copy:
            safe_copy["privacy_scan_required"] = True
            changes.append(
                ReviewPacketMigrationChange("/safety/privacy_scan_required", "add")
            )
        elif safe_copy["privacy_scan_required"] is not True:
            raise LossyReviewPacketMigrationError(
                "review packet migration would overwrite an incompatible safety field"
            )
        packet["safety"] = safe_copy

    packet["schema_version"] = 2
    changes.append(ReviewPacketMigrationChange("/schema_version", "replace"))


def _validate_v2(packet: Mapping[str, Any]) -> None:
    safety = packet.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("privacy_scan_required") is not True
    ):
        raise ReviewPacketMigrationError(
            "review packet schema version 2 requires the privacy scan safety field"
        )


def _packet_version(value: Any) -> int:
    if type(value) is not int or value not in SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS:
        raise ReviewPacketMigrationError(
            "review packet schema version is missing or unsupported"
        )
    return value


def _target_version(value: Any) -> int:
    if type(value) is not int or value not in SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS:
        raise ReviewPacketMigrationError(
            "target review packet schema version is unsupported"
        )
    return value


__all__ = [
    "REVIEW_PACKET_MIGRATION_REPORT_SCHEMA_VERSION",
    "REVIEW_PACKET_SCHEMA_VERSION",
    "SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS",
    "LossyReviewPacketMigrationError",
    "ReviewPacketMigrationChange",
    "ReviewPacketMigrationError",
    "ReviewPacketMigrationReport",
    "ReviewPacketMigrationResult",
    "migrate_review_packet",
]
