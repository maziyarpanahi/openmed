"""Deterministic schema compatibility evaluation for versioned agent artifacts.

Run summaries, handoff packets, policy matrices, and evidence references evolve
independently. This module checks whether an artifact schema version satisfies
caller-declared compatibility ranges without inspecting artifact payloads or
performing network lookups.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .artifact_reference import ArtifactKind

COMPATIBILITY_SCHEMA_VERSION: Final = "openmed.agent.schema_compatibility.v1"

_SEMVER_RE: Final = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

_ORDERED_FIELDS: Final = (
    "artifact_kind",
    "incoming_version",
    "outcome",
    "reason_code",
    "min_version",
    "max_version",
)


class CompatibilityOutcome(str, Enum):
    """Closed vocabulary of compatibility evaluation outcomes."""

    COMPATIBLE = "compatible"
    UPGRADE_REQUIRED = "upgrade-required"
    DOWNGRADE_REQUIRED = "downgrade-required"
    UNSUPPORTED = "unsupported"


class CompatibilityReason(str, Enum):
    """Stable machine-readable reason codes for compatibility decisions."""

    EXACT_MATCH = "exact_match"
    WITHIN_RANGE = "within_range"
    UPGRADE_REQUIRED = "upgrade_required"
    DOWNGRADE_REQUIRED = "downgrade_required"
    PRERELEASE_UNSUPPORTED = "prerelease_unsupported"
    MALFORMED_VERSION = "malformed_version"
    UNKNOWN_ARTIFACT_KIND = "unknown_artifact_kind"
    INVALID_RANGE = "invalid_range"


class SchemaCompatibilityError(ValueError):
    """Raised when schema compatibility inputs fail contract validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SemVer:
    """Strict Semantic Version 2.0.0 model supporting total order precedence."""

    major: int
    minor: int
    patch: int
    prerelease: tuple[str | int, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: Any) -> SemVer:
        """Parse a strict SemVer 2.0.0 string.

        Args:
            value: Version string to validate.

        Returns:
            Immutable parsed SemVer instance.

        Raises:
            SchemaCompatibilityError: If value is not a valid SemVer 2.0.0 string.
        """
        if type(value) is not str or len(value) > 128:
            raise SchemaCompatibilityError("malformed_version", "version")

        match = _SEMVER_RE.fullmatch(value)
        if match is None:
            raise SchemaCompatibilityError("malformed_version", "version")

        major = int(match.group("major"))
        minor = int(match.group("minor"))
        patch = int(match.group("patch"))

        prerelease_raw = match.group("prerelease")
        if prerelease_raw:
            parts: list[str | int] = []
            for part in prerelease_raw.split("."):
                if part.isdigit():
                    if len(part) > 1 and part.startswith("0"):
                        raise SchemaCompatibilityError("malformed_version", "version")
                    parts.append(int(part))
                else:
                    parts.append(part)
            prerelease = tuple(parts)
        else:
            prerelease = ()

        build_raw = match.group("buildmetadata")
        build = tuple(build_raw.split(".")) if build_raw else ()

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build,
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Return True if value can be parsed as a valid SemVer 2.0.0."""
        try:
            cls.parse(value)
            return True
        except (SchemaCompatibilityError, TypeError, ValueError):
            return False

    @property
    def is_prerelease(self) -> bool:
        """Return True if this version contains pre-release identifiers."""
        return len(self.prerelease) > 0

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += "-" + ".".join(str(p) for p in self.prerelease)
        if self.build:
            base += "+" + ".".join(self.build)
        return base

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented

        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)
        if self_tuple != other_tuple:
            return self_tuple < other_tuple

        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and not other.prerelease:
            return False

        for a, b in zip(self.prerelease, other.prerelease):
            if a == b:
                continue
            is_a_int = isinstance(a, int)
            is_b_int = isinstance(b, int)
            if is_a_int and is_b_int:
                return a < b
            if is_a_int and not is_b_int:
                return True
            if not is_a_int and is_b_int:
                return False
            return str(a) < str(b)

        return len(self.prerelease) < len(other.prerelease)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return not (self <= other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return not (self < other)


@dataclass(frozen=True, slots=True)
class SchemaRange:
    """Caller-declared inclusive minimum and maximum supported schema versions.

    Args:
        min_version: Inclusive lower bound semantic version.
        max_version: Inclusive upper bound semantic version.
        allow_prerelease: Whether pre-release schema versions are permitted.
    """

    min_version: str
    max_version: str
    allow_prerelease: bool = False

    def __post_init__(self) -> None:
        min_semver = SemVer.parse(self.min_version)
        max_semver = SemVer.parse(self.max_version)
        if min_semver > max_semver:
            raise SchemaCompatibilityError("invalid_range", "min_version")
        if type(self.allow_prerelease) is not bool:
            raise SchemaCompatibilityError("invalid_type", "allow_prerelease")


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Deterministic result of an artifact schema compatibility check.

    Contains only bounded categorical metadata and version identifiers. No
    artifact content or arbitrary payload is ever accepted or serialized.

    Args:
        outcome: The compatibility verdict.
        reason_code: Stable machine-readable reason code.
        artifact_kind: Normalized artifact category string.
        incoming_version: Version submitted for evaluation.
        min_version: Minimum supported version declared by caller, if known.
        max_version: Maximum supported version declared by caller, if known.
    """

    outcome: CompatibilityOutcome
    reason_code: str
    artifact_kind: str
    incoming_version: str
    min_version: str | None = None
    max_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, CompatibilityOutcome):
            raise SchemaCompatibilityError("invalid_outcome", "outcome")
        if type(self.reason_code) is not str or not self.reason_code:
            raise SchemaCompatibilityError("invalid_reason", "reason_code")
        if type(self.artifact_kind) is not str:
            raise SchemaCompatibilityError("invalid_kind", "artifact_kind")
        if type(self.incoming_version) is not str:
            raise SchemaCompatibilityError("invalid_version", "incoming_version")

    @property
    def is_compatible(self) -> bool:
        """Return True if the artifact schema is compatible."""
        return self.outcome == CompatibilityOutcome.COMPATIBLE

    def to_dict(self) -> dict[str, str]:
        """Return deterministic metadata-only fields."""
        values: dict[str, str | None] = {
            "artifact_kind": self.artifact_kind,
            "incoming_version": self.incoming_version,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "min_version": self.min_version,
            "max_version": self.max_version,
        }
        return {
            field: str(values[field])
            for field in _ORDERED_FIELDS
            if values[field] is not None
        }

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompatibilityResult:
        """Build and validate a result from a strict mapping."""
        if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
            raise SchemaCompatibilityError("not_a_mapping")

        allowed = frozenset(_ORDERED_FIELDS)
        unknown = set(data) - allowed
        if unknown:
            raise SchemaCompatibilityError("unknown_field")

        required = {"outcome", "reason_code", "artifact_kind", "incoming_version"}
        if required - set(data):
            raise SchemaCompatibilityError("missing_field")

        try:
            outcome = CompatibilityOutcome(data["outcome"])
        except ValueError:
            raise SchemaCompatibilityError("invalid_outcome", "outcome") from None

        return cls(
            outcome=outcome,
            reason_code=data["reason_code"],
            artifact_kind=data["artifact_kind"],
            incoming_version=data["incoming_version"],
            min_version=data.get("min_version"),
            max_version=data.get("max_version"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> CompatibilityResult:
        """Build and validate a result from a JSON payload."""
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            raise SchemaCompatibilityError("malformed_json") from None
        return cls.from_dict(data)


def check_schema_compatibility(
    artifact_kind: ArtifactKind | str,
    incoming_version: str,
    *,
    min_version: str | None = None,
    max_version: str | None = None,
    supported_range: SchemaRange | None = None,
    allow_prerelease: bool = False,
) -> CompatibilityResult:
    """Evaluate schema compatibility for an artifact without inspecting content.

    Args:
        artifact_kind: Closed artifact category.
        incoming_version: Incoming semantic version string.
        min_version: Inclusive minimum supported version (optional if supported_range is passed).
        max_version: Inclusive maximum supported version (optional if supported_range is passed).
        supported_range: Optional pre-constructed SchemaRange.
        allow_prerelease: Whether pre-release schema versions are permitted.

    Returns:
        Deterministic CompatibilityResult with stable outcome and reason code.
    """
    kind_str: str
    if isinstance(artifact_kind, ArtifactKind):
        kind_str = artifact_kind.value
    elif type(artifact_kind) is str:
        try:
            kind_str = ArtifactKind(artifact_kind).value
        except ValueError:
            return CompatibilityResult(
                outcome=CompatibilityOutcome.UNSUPPORTED,
                reason_code=CompatibilityReason.UNKNOWN_ARTIFACT_KIND.value,
                artifact_kind=artifact_kind,
                incoming_version=incoming_version
                if type(incoming_version) is str
                else "",
                min_version=min_version,
                max_version=max_version,
            )
    else:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.UNKNOWN_ARTIFACT_KIND.value,
            artifact_kind=str(artifact_kind),
            incoming_version=incoming_version if type(incoming_version) is str else "",
            min_version=min_version,
            max_version=max_version,
        )

    if type(incoming_version) is not str:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.MALFORMED_VERSION.value,
            artifact_kind=kind_str,
            incoming_version="",
            min_version=min_version,
            max_version=max_version,
        )

    try:
        incoming_semver = SemVer.parse(incoming_version)
    except SchemaCompatibilityError:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.MALFORMED_VERSION.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=min_version,
            max_version=max_version,
        )

    effective_min = min_version
    effective_max = max_version
    effective_allow_prerelease = allow_prerelease

    if supported_range is not None:
        effective_min = supported_range.min_version
        effective_max = supported_range.max_version
        effective_allow_prerelease = supported_range.allow_prerelease

    if effective_min is None or effective_max is None:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.INVALID_RANGE.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    try:
        min_semver = SemVer.parse(effective_min)
        max_semver = SemVer.parse(effective_max)
    except SchemaCompatibilityError:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.MALFORMED_VERSION.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    if min_semver > max_semver:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.INVALID_RANGE.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    if incoming_semver.is_prerelease and not effective_allow_prerelease:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UNSUPPORTED,
            reason_code=CompatibilityReason.PRERELEASE_UNSUPPORTED.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    if incoming_semver == min_semver or incoming_semver == max_semver:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.COMPATIBLE,
            reason_code=CompatibilityReason.EXACT_MATCH.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    if min_semver < incoming_semver < max_semver:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.COMPATIBLE,
            reason_code=CompatibilityReason.WITHIN_RANGE.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    if incoming_semver < min_semver:
        return CompatibilityResult(
            outcome=CompatibilityOutcome.UPGRADE_REQUIRED,
            reason_code=CompatibilityReason.UPGRADE_REQUIRED.value,
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            min_version=effective_min,
            max_version=effective_max,
        )

    return CompatibilityResult(
        outcome=CompatibilityOutcome.DOWNGRADE_REQUIRED,
        reason_code=CompatibilityReason.DOWNGRADE_REQUIRED.value,
        artifact_kind=kind_str,
        incoming_version=incoming_version,
        min_version=effective_min,
        max_version=effective_max,
    )


class SchemaCompatibilityMatrix:
    """Multi-artifact policy matrix declaring supported schema ranges per kind.

    Fails closed when queried for an unregistered or unknown artifact kind.
    """

    def __init__(
        self,
        ranges: Mapping[ArtifactKind | str, SchemaRange],
    ) -> None:
        self._ranges: dict[str, SchemaRange] = {}
        for kind_key, range_val in ranges.items():
            if isinstance(kind_key, ArtifactKind):
                key = kind_key.value
            elif type(kind_key) is str:
                try:
                    key = ArtifactKind(kind_key).value
                except ValueError:
                    raise SchemaCompatibilityError(
                        "unknown_artifact_kind", "artifact_kind"
                    )
            else:
                raise SchemaCompatibilityError("unknown_artifact_kind", "artifact_kind")

            if not isinstance(range_val, SchemaRange):
                raise SchemaCompatibilityError("invalid_range", "range")
            self._ranges[key] = range_val

    def check(
        self,
        artifact_kind: ArtifactKind | str,
        incoming_version: str,
    ) -> CompatibilityResult:
        """Check compatibility against declared range for the artifact kind."""
        kind_str: str
        if isinstance(artifact_kind, ArtifactKind):
            kind_str = artifact_kind.value
        elif type(artifact_kind) is str:
            try:
                kind_str = ArtifactKind(artifact_kind).value
            except ValueError:
                return CompatibilityResult(
                    outcome=CompatibilityOutcome.UNSUPPORTED,
                    reason_code=CompatibilityReason.UNKNOWN_ARTIFACT_KIND.value,
                    artifact_kind=artifact_kind,
                    incoming_version=incoming_version
                    if type(incoming_version) is str
                    else "",
                )
        else:
            return CompatibilityResult(
                outcome=CompatibilityOutcome.UNSUPPORTED,
                reason_code=CompatibilityReason.UNKNOWN_ARTIFACT_KIND.value,
                artifact_kind=str(artifact_kind),
                incoming_version=incoming_version
                if type(incoming_version) is str
                else "",
            )

        supported_range = self._ranges.get(kind_str)
        if supported_range is None:
            return CompatibilityResult(
                outcome=CompatibilityOutcome.UNSUPPORTED,
                reason_code=CompatibilityReason.UNKNOWN_ARTIFACT_KIND.value,
                artifact_kind=kind_str,
                incoming_version=incoming_version
                if type(incoming_version) is str
                else "",
            )

        return check_schema_compatibility(
            artifact_kind=kind_str,
            incoming_version=incoming_version,
            supported_range=supported_range,
        )


__all__ = [
    "COMPATIBILITY_SCHEMA_VERSION",
    "CompatibilityOutcome",
    "CompatibilityReason",
    "CompatibilityResult",
    "SchemaCompatibilityError",
    "SchemaCompatibilityMatrix",
    "SchemaRange",
    "SemVer",
    "check_schema_compatibility",
]
