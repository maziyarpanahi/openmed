"""Deterministic retention planning for aggregate-only audit artifacts.

The public input contract is intentionally narrow: an artifact has an opaque
identifier, a creation timestamp, a disposition, and numeric counts. The
scrubber never returns the identifier or timestamp. It returns only aggregate
counts, safe disposition names, and SHA-256 fingerprints that let a caller
verify the retained set after applying the deletion plan in its own store.

The implementation is local-only and side-effect free. It does not delete
files or make network calls; storage-specific deletion remains the caller's
responsibility.
"""

from __future__ import annotations

import hmac
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Final, cast

from openmed.core.audit import stable_hash

__all__ = [
    "AUDIT_RETENTION_FORMAT",
    "AUDIT_RETENTION_VERSION",
    "MAX_AUDIT_RETENTION_ARTIFACTS",
    "MAX_AUDIT_RETENTION_COUNT",
    "MAX_AUDIT_RETENTION_JSON_BYTES",
    "MAX_AUDIT_RETENTION_METRICS",
    "MAX_AUDIT_RETENTION_RULES",
    "AuditArtifact",
    "AuditArtifactRecord",
    "AuditRetentionPolicy",
    "AuditRetentionReport",
    "DeletionFingerprint",
    "RetainedArtifactSummary",
    "RetentionPolicy",
    "RetentionReport",
    "RetentionRule",
    "artifact_set_fingerprint",
    "scrub",
    "scrub_audit_artifacts",
    "verify_remaining_artifacts",
]

AUDIT_RETENTION_FORMAT = "openmed.audit-retention"
AUDIT_RETENTION_VERSION = 1
MAX_AUDIT_RETENTION_ARTIFACTS: Final = 10_000
MAX_AUDIT_RETENTION_COUNT: Final = (1 << 63) - 1
MAX_AUDIT_RETENTION_JSON_BYTES: Final = 8_388_608
MAX_AUDIT_RETENTION_METRICS: Final = 1_024
MAX_AUDIT_RETENTION_RULES: Final = 256

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_NAME_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
_ACTIONS: Final = frozenset({"delete", "retain"})
_DELETION_REASONS: Final = frozenset({"age_expired"})
_RETENTION_REASONS: Final = frozenset({"disposition_hold", "within_retention"})
_MAX_ARTIFACT_ID_CHARS: Final = 1_024
_MAX_TIMESTAMP_CHARS: Final = 64
_MISSING = object()
_RAW_INPUT_FIELDS: Final = frozenset(
    {
        "document",
        "original",
        "original_text",
        "patient_id",
        "path",
        "record_id",
        "source",
        "source_path",
        "surface",
        "text",
        "value",
    }
)
_ARTIFACT_FIELD_ALIASES: Final = {
    "artifact_id": ("artifact_id", "id", "key"),
    "created_at": ("created_at", "created", "timestamp"),
    "disposition": ("disposition",),
    "counts": ("counts", "counters", "metrics"),
    "count": ("count", "event_count"),
}
_ARTIFACT_FIELDS: Final = frozenset(
    field for aliases in _ARTIFACT_FIELD_ALIASES.values() for field in aliases
)


def _is_digest(value: Any) -> bool:
    return type(value) is str and bool(_DIGEST_RE.fullmatch(value))


def _require_digest(value: Any, field_name: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{field_name} must be a sha256:<hex> digest")
    return value


def _safe_name(value: Any, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a safe identifier")
    normalized = value.strip().lower()
    if not _SAFE_NAME_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a safe identifier")
    return normalized


def _coerce_datetime(value: Any, *, field_name: str) -> datetime:
    if type(value) is datetime:
        parsed = value
    elif type(value) is date:
        parsed = datetime.combine(value, datetime.min.time())
    elif type(value) is str and value.strip():
        text = value.strip()
        if len(text) > _MAX_TIMESTAMP_CHARS:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except (OverflowError, ValueError):
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from None
    else:
        raise TypeError(f"{field_name} must be an ISO-8601 timestamp")

    try:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001 - tzinfo callbacks are caller-controlled.
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from None


def _isoformat(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _coerce_counts(value: Any) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("audit artifact counts must be a mapping")

    values = _snapshot_mapping(
        value,
        field_name="audit artifact counts",
        max_items=MAX_AUDIT_RETENTION_METRICS,
    )
    normalized: dict[str, int] = {}
    total = 0
    for key, count in values.items():
        name = _safe_name(key, field_name="audit artifact count name")
        if name in normalized:
            raise ValueError("audit artifact count names must be unique")
        if type(count) is not int or not 0 <= count <= MAX_AUDIT_RETENTION_COUNT:
            raise ValueError("audit artifact counts must be bounded integers")
        total += count
        if total > MAX_AUDIT_RETENTION_COUNT:
            raise ValueError("audit artifact count total exceeds the supported limit")
        normalized[name] = count
    return tuple(sorted(normalized.items()))


def _snapshot_mapping(
    data: Mapping[Any, Any],
    *,
    field_name: str,
    max_items: int,
) -> dict[str, Any]:
    try:
        items = list(itertools.islice(data.items(), max_items + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise ValueError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise ValueError(f"{field_name} exceeds the supported item limit")

    result: dict[str, Any] = {}
    for item in items:
        if type(item) not in {tuple, list} or len(item) != 2:
            raise ValueError(f"{field_name} contains an invalid entry")
        key, value = item
        if type(key) is not str:
            raise ValueError(f"{field_name} keys must be strings")
        if key in result:
            raise ValueError(f"{field_name} keys must be unique")
        result[key] = value
    return result


def _aliased_value(data: Mapping[str, Any], field_name: str) -> Any:
    aliases = _ARTIFACT_FIELD_ALIASES[field_name]
    matches = [alias for alias in aliases if alias in data]
    if len(matches) > 1:
        raise ValueError("audit artifact fields must not use duplicate aliases")
    return _MISSING if not matches else data[matches[0]]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("retention report JSON fields must be unique")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("retention report JSON numbers must be finite")


@dataclass(frozen=True)
class RetentionRule:
    """One disposition rule for an aggregate-only audit artifact.

    A ``delete`` rule requires a non-negative ``max_age``. A ``retain`` rule
    may omit ``max_age`` and acts as an indefinite hold, which is useful for
    legal or incident-response holds.
    """

    max_age: timedelta | None
    action: str = "delete"

    def __post_init__(self) -> None:
        action = _safe_name(self.action, field_name="retention action")
        if action not in _ACTIONS:
            raise ValueError("retention action must be delete or retain")
        if self.max_age is not None:
            if type(self.max_age) is not timedelta:
                raise TypeError("retention max_age must be a timedelta or None")
            if self.max_age < timedelta(0):
                raise ValueError("retention max_age must not be negative")
        if action == "delete" and self.max_age is None:
            raise ValueError("delete retention rules require max_age")
        if action == "retain" and self.max_age is not None:
            raise ValueError("retain retention rules must not specify max_age")
        object.__setattr__(self, "action", action)

    @classmethod
    def days(cls, days: int, *, action: str = "delete") -> "RetentionRule":
        """Construct a rule from a whole number of retention days."""

        if type(days) is not int or not 0 <= days <= timedelta.max.days:
            raise ValueError("retention days must be a non-negative integer")
        return cls(max_age=timedelta(days=days), action=action)

    @property
    def max_age_seconds(self) -> int | float | None:
        """Return the exact retention duration in JSON-safe seconds."""

        if self.max_age is None:
            return None
        seconds = self.max_age.total_seconds()
        return int(seconds) if seconds.is_integer() else seconds

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, deterministic rule representation."""

        return {
            "action": self.action,
            "max_age_seconds": self.max_age_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetentionRule":
        """Load a serialized retention rule without accepting unknown fields."""

        if not isinstance(data, Mapping):
            raise ValueError("retention rule has missing or unknown fields")
        values = _snapshot_mapping(
            data,
            field_name="retention rule",
            max_items=2,
        )
        if set(values) != {
            "action",
            "max_age_seconds",
        }:
            raise ValueError("retention rule has missing or unknown fields")
        seconds = values["max_age_seconds"]
        if seconds is None:
            max_age = None
        elif type(seconds) is int and 0 <= seconds <= timedelta.max.total_seconds():
            max_age = timedelta(seconds=seconds)
        elif (
            type(seconds) is float
            and math.isfinite(seconds)
            and 0 <= seconds <= timedelta.max.total_seconds()
        ):
            max_age = timedelta(seconds=seconds)
        else:
            raise ValueError("retention max_age_seconds must be non-negative")
        return cls(max_age=max_age, action=values["action"])


@dataclass(frozen=True)
class AuditRetentionPolicy:
    """Explicit age and disposition rules for audit artifacts."""

    rules: Mapping[str, RetentionRule | timedelta]
    default_rule: RetentionRule | timedelta | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rules, Mapping):
            raise ValueError("retention policy rules must be a non-empty mapping")
        rules = _snapshot_mapping(
            self.rules,
            field_name="retention policy rules",
            max_items=MAX_AUDIT_RETENTION_RULES,
        )
        if not rules:
            raise ValueError("retention policy rules must be a non-empty mapping")

        normalized: dict[str, RetentionRule] = {}
        for disposition, rule in rules.items():
            name = _safe_name(disposition, field_name="retention disposition")
            if name in normalized:
                raise ValueError("retention policy dispositions must be unique")
            if type(rule) is timedelta:
                rule = RetentionRule(max_age=rule)
            if type(rule) is not RetentionRule:
                raise TypeError(
                    "retention policy rules must contain RetentionRule values"
                )
            normalized[name] = rule

        default = self.default_rule
        if type(default) is timedelta:
            default = RetentionRule(max_age=default)
        if default is not None and type(default) is not RetentionRule:
            raise TypeError("default retention rule must be a RetentionRule or None")

        object.__setattr__(
            self, "rules", MappingProxyType(dict(sorted(normalized.items())))
        )
        object.__setattr__(self, "default_rule", default)

    def rule_for(self, disposition: str) -> RetentionRule:
        """Return the explicit rule for a disposition or fail closed."""

        name = _safe_name(disposition, field_name="retention disposition")
        rule = self.rules.get(name, self.default_rule)
        if rule is None:
            raise ValueError(
                "retention policy has no rule for the artifact disposition"
            )
        return cast(RetentionRule, rule)

    @property
    def fingerprint(self) -> str:
        """Return the deterministic policy fingerprint."""

        return stable_hash(
            {
                "format": f"{AUDIT_RETENTION_FORMAT}.policy",
                "rules": self.to_dict()["rules"],
                "default_rule": self.to_dict()["default_rule"],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only safe disposition names and age/action settings."""

        rules = cast(Mapping[str, RetentionRule], self.rules)
        default_rule = cast(RetentionRule | None, self.default_rule)
        return {
            "rules": {
                disposition: rule.to_dict()
                for disposition, rule in sorted(rules.items())
            },
            "default_rule": (
                default_rule.to_dict() if default_rule is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditRetentionPolicy":
        """Load a policy from its deterministic safe representation."""

        if not isinstance(data, Mapping):
            raise ValueError("retention policy has missing or unknown fields")
        values = _snapshot_mapping(
            data,
            field_name="retention policy",
            max_items=2,
        )
        if set(values) != {"rules", "default_rule"}:
            raise ValueError("retention policy has missing or unknown fields")
        rules = values["rules"]
        if not isinstance(rules, Mapping):
            raise TypeError("retention policy rules must be a mapping")
        rule_values = _snapshot_mapping(
            rules,
            field_name="retention policy rules",
            max_items=MAX_AUDIT_RETENTION_RULES,
        )
        parsed_rules = {
            disposition: RetentionRule.from_dict(rule)
            for disposition, rule in rule_values.items()
            if isinstance(rule, Mapping)
        }
        if len(parsed_rules) != len(rule_values):
            raise ValueError("retention policy rules contain invalid entries")
        default = values["default_rule"]
        if default is not None and not isinstance(default, Mapping):
            raise TypeError("default retention rule must be an object or null")
        return cls(
            rules=parsed_rules,
            default_rule=(
                RetentionRule.from_dict(default) if default is not None else None
            ),
        )


@dataclass(frozen=True)
class AuditArtifact:
    """Opaque counts-only audit artifact accepted by the scrubber.

    ``artifact_id`` is used only in memory to calculate a fingerprint. It is
    never present in a retention report. Callers should use a locally scoped
    opaque identifier rather than a patient, encounter, or source identifier.
    """

    artifact_id: str
    created_at: datetime | date | str
    disposition: str
    counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            type(self.artifact_id) is not str
            or not self.artifact_id.strip()
            or len(self.artifact_id) > _MAX_ARTIFACT_ID_CHARS
        ):
            raise ValueError("audit artifact identifier must be a non-empty string")
        object.__setattr__(
            self,
            "created_at",
            _coerce_datetime(self.created_at, field_name="audit artifact created_at"),
        )
        object.__setattr__(
            self,
            "disposition",
            _safe_name(self.disposition, field_name="audit artifact disposition"),
        )
        object.__setattr__(
            self,
            "counts",
            MappingProxyType(dict(_coerce_counts(self.counts))),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AuditArtifact":
        """Create an artifact while rejecting known raw-content fields."""

        if not isinstance(data, Mapping):
            raise TypeError("audit artifact must be a mapping or AuditArtifact")
        values = _snapshot_mapping(
            data,
            field_name="audit artifact",
            max_items=len(_ARTIFACT_FIELDS),
        )
        if set(values) - _ARTIFACT_FIELDS or set(values) & _RAW_INPUT_FIELDS:
            raise ValueError("audit artifacts must contain counts only")

        artifact_id = _aliased_value(values, "artifact_id")
        created_at = _aliased_value(values, "created_at")
        disposition = _aliased_value(values, "disposition")
        if artifact_id is _MISSING or created_at is _MISSING or disposition is _MISSING:
            raise ValueError("audit artifact requires id, created_at, and disposition")

        counts = _aliased_value(values, "counts")
        count = _aliased_value(values, "count")
        if counts is not _MISSING and count is not _MISSING:
            raise ValueError(
                "audit artifact count fields must not use duplicate aliases"
            )
        if counts is _MISSING:
            counts = {} if count is _MISSING else {"total": count}
        return cls(
            artifact_id=artifact_id,
            created_at=created_at,
            disposition=disposition,
            counts=counts,
        )

    @property
    def count_total(self) -> int:
        """Return the aggregate count without exposing metric names."""

        return sum(self.counts.values())

    @property
    def metric_count(self) -> int:
        """Return the number of count buckets."""

        return len(self.counts)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe summary without the opaque input fields."""

        return {
            "artifact_fingerprint": self.fingerprint,
            "count_total": self.count_total,
            "disposition": self.disposition,
            "metric_count": self.metric_count,
        }

    @property
    def fingerprint(self) -> str:
        """Return the deterministic, content-binding artifact fingerprint."""

        return stable_hash(
            {
                "format": f"{AUDIT_RETENTION_FORMAT}.artifact",
                "artifact_id": self.artifact_id,
                "created_at": _isoformat(cast(datetime, self.created_at)),
                "disposition": self.disposition,
                "counts": dict(self.counts),
            }
        )


def _coerce_artifact(value: AuditArtifact | Mapping[str, Any]) -> AuditArtifact:
    if type(value) is AuditArtifact:
        return cast(AuditArtifact, value)
    return AuditArtifact.from_mapping(cast(Mapping[str, Any], value))


def _materialize_artifacts(
    artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
) -> tuple[AuditArtifact, ...]:
    if isinstance(artifacts, (str, bytes, Mapping)):
        raise TypeError("audit artifacts must be an iterable of counts-only artifacts")
    try:
        inputs = list(
            itertools.islice(iter(artifacts), MAX_AUDIT_RETENTION_ARTIFACTS + 1)
        )
    except Exception:  # noqa: BLE001 - iterables are caller-controlled protocols.
        raise ValueError("audit artifacts could not be read") from None
    if len(inputs) > MAX_AUDIT_RETENTION_ARTIFACTS:
        raise ValueError("audit artifacts exceed the supported item limit")

    values: list[AuditArtifact] = []
    seen_ids: set[str] = set()
    for item in inputs:
        try:
            artifact = _coerce_artifact(item)
        except (TypeError, ValueError):
            raise
        except Exception:  # noqa: BLE001 - mappings may execute caller code.
            raise ValueError("audit artifact could not be read") from None
        if artifact.artifact_id in seen_ids:
            raise ValueError("audit artifact identifiers must be unique")
        seen_ids.add(artifact.artifact_id)
        values.append(artifact)
    return tuple(sorted(values, key=lambda item: item.fingerprint))


def _manifest_fingerprint(artifact_fingerprints: Iterable[str]) -> str:
    return stable_hash(
        {
            "format": f"{AUDIT_RETENTION_FORMAT}.set",
            "artifacts": sorted(artifact_fingerprints),
        }
    )


def artifact_set_fingerprint(
    artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
) -> str:
    """Return a deterministic fingerprint for a counts-only artifact set."""

    normalized = _materialize_artifacts(artifacts)
    return _manifest_fingerprint(item.fingerprint for item in normalized)


@dataclass(frozen=True)
class DeletionFingerprint:
    """Privacy-safe evidence for one artifact selected for deletion."""

    artifact_fingerprint: str
    disposition: str
    age_seconds: int
    reason: str

    def __post_init__(self) -> None:
        _require_digest(self.artifact_fingerprint, "deletion artifact_fingerprint")
        if (
            type(self.age_seconds) is not int
            or not 0 <= self.age_seconds <= MAX_AUDIT_RETENTION_COUNT
        ):
            raise ValueError("deletion age_seconds must be a bounded integer")
        object.__setattr__(
            self,
            "disposition",
            _safe_name(self.disposition, field_name="deletion disposition"),
        )
        object.__setattr__(
            self,
            "reason",
            _safe_name(self.reason, field_name="deletion reason"),
        )
        if self.reason not in _DELETION_REASONS:
            raise ValueError("deletion reason is not supported")

    def to_dict(self) -> dict[str, Any]:
        """Return the safe deletion evidence representation."""

        return {
            "age_seconds": self.age_seconds,
            "artifact_fingerprint": self.artifact_fingerprint,
            "disposition": self.disposition,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeletionFingerprint":
        """Load one deletion fingerprint from a safe mapping."""

        fields = {"age_seconds", "artifact_fingerprint", "disposition", "reason"}
        if not isinstance(data, Mapping):
            raise ValueError("deletion fingerprint has missing or unknown fields")
        values = _snapshot_mapping(
            data,
            field_name="deletion fingerprint",
            max_items=len(fields),
        )
        if set(values) != fields:
            raise ValueError("deletion fingerprint has missing or unknown fields")
        return cls(
            artifact_fingerprint=values["artifact_fingerprint"],
            disposition=values["disposition"],
            age_seconds=values["age_seconds"],
            reason=values["reason"],
        )


@dataclass(frozen=True)
class RetainedArtifactSummary:
    """Aggregate-only summary for one retained artifact."""

    artifact_fingerprint: str
    disposition: str
    age_seconds: int
    count_total: int
    metric_count: int
    reason: str

    def __post_init__(self) -> None:
        _require_digest(self.artifact_fingerprint, "retained artifact_fingerprint")
        if (
            type(self.age_seconds) is not int
            or not 0 <= self.age_seconds <= MAX_AUDIT_RETENTION_COUNT
        ):
            raise ValueError("retained age_seconds must be a bounded integer")
        if (
            type(self.count_total) is not int
            or not 0 <= self.count_total <= MAX_AUDIT_RETENTION_COUNT
        ):
            raise ValueError("retained count_total must be a bounded integer")
        if (
            type(self.metric_count) is not int
            or not 0 <= self.metric_count <= MAX_AUDIT_RETENTION_METRICS
        ):
            raise ValueError("retained metric_count must be a bounded integer")
        object.__setattr__(
            self,
            "disposition",
            _safe_name(self.disposition, field_name="retained disposition"),
        )
        object.__setattr__(
            self,
            "reason",
            _safe_name(self.reason, field_name="retained reason"),
        )
        if self.reason not in _RETENTION_REASONS:
            raise ValueError("retained reason is not supported")

    def to_dict(self) -> dict[str, Any]:
        """Return the safe retained-artifact summary."""

        return {
            "age_seconds": self.age_seconds,
            "artifact_fingerprint": self.artifact_fingerprint,
            "count_total": self.count_total,
            "disposition": self.disposition,
            "metric_count": self.metric_count,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetainedArtifactSummary":
        """Load one retained summary from a safe mapping."""

        fields = {
            "age_seconds",
            "artifact_fingerprint",
            "count_total",
            "disposition",
            "metric_count",
            "reason",
        }
        if not isinstance(data, Mapping):
            raise ValueError("retained artifact summary has missing or unknown fields")
        values = _snapshot_mapping(
            data,
            field_name="retained artifact summary",
            max_items=len(fields),
        )
        if set(values) != fields:
            raise ValueError("retained artifact summary has missing or unknown fields")
        return cls(
            artifact_fingerprint=values["artifact_fingerprint"],
            disposition=values["disposition"],
            age_seconds=values["age_seconds"],
            count_total=values["count_total"],
            metric_count=values["metric_count"],
            reason=values["reason"],
        )


@dataclass(frozen=True)
class AuditRetentionReport:
    """Deterministic, integrity-checked evidence for a retention pass."""

    as_of: str
    policy_fingerprint: str
    input_fingerprint: str
    remaining_fingerprint: str
    deletion_fingerprint: str
    input_artifact_count: int
    retained_artifact_count: int
    deleted_artifact_count: int
    retained_artifacts: tuple[RetainedArtifactSummary, ...] = ()
    deleted_artifacts: tuple[DeletionFingerprint, ...] = ()

    def __post_init__(self) -> None:
        as_of = _coerce_datetime(self.as_of, field_name="retention report as_of")
        object.__setattr__(self, "as_of", _isoformat(as_of))
        for name in (
            "policy_fingerprint",
            "input_fingerprint",
            "remaining_fingerprint",
            "deletion_fingerprint",
        ):
            _require_digest(getattr(self, name), f"retention report {name}")
        for name in (
            "input_artifact_count",
            "retained_artifact_count",
            "deleted_artifact_count",
        ):
            value = getattr(self, name)
            if (
                type(value) is not int
                or not 0 <= value <= MAX_AUDIT_RETENTION_ARTIFACTS
            ):
                raise ValueError(f"retention report {name} must be bounded")

        if type(self.retained_artifacts) not in {list, tuple} or type(
            self.deleted_artifacts
        ) not in {list, tuple}:
            raise TypeError("retention report artifact collections must be sequences")
        retained = tuple(self.retained_artifacts)
        deleted = tuple(self.deleted_artifacts)
        if len(retained) + len(deleted) > MAX_AUDIT_RETENTION_ARTIFACTS:
            raise ValueError("retention report contains too many artifact summaries")
        if not all(type(item) is RetainedArtifactSummary for item in retained):
            raise TypeError(
                "retention report retained_artifacts must contain summaries"
            )
        if not all(type(item) is DeletionFingerprint for item in deleted):
            raise TypeError(
                "retention report deleted_artifacts must contain fingerprints"
            )
        if len(retained) != self.retained_artifact_count:
            raise ValueError("retention report retained count does not match summaries")
        if len(deleted) != self.deleted_artifact_count:
            raise ValueError(
                "retention report deleted count does not match fingerprints"
            )
        if self.input_artifact_count != (
            self.retained_artifact_count + self.deleted_artifact_count
        ):
            raise ValueError("retention report artifact counts do not balance")

        retained = tuple(sorted(retained, key=lambda item: item.artifact_fingerprint))
        deleted = tuple(sorted(deleted, key=lambda item: item.artifact_fingerprint))
        fingerprints = [
            *(item.artifact_fingerprint for item in retained),
            *(item.artifact_fingerprint for item in deleted),
        ]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("retention report artifact fingerprints must be unique")
        expected_remaining = _manifest_fingerprint(
            item.artifact_fingerprint for item in retained
        )
        expected_deleted = stable_hash(
            {
                "format": f"{AUDIT_RETENTION_FORMAT}.deletions",
                "artifacts": [item.to_dict() for item in deleted],
            }
        )
        expected_input = _manifest_fingerprint(fingerprints)
        if self.remaining_fingerprint != expected_remaining:
            raise ValueError("retention report remaining fingerprint does not match")
        if self.deletion_fingerprint != expected_deleted:
            raise ValueError("retention report deletion fingerprint does not match")
        if self.input_fingerprint != expected_input:
            raise ValueError("retention report input fingerprint does not match")
        object.__setattr__(self, "retained_artifacts", retained)
        object.__setattr__(self, "deleted_artifacts", deleted)

    @property
    def integrity_digest(self) -> str:
        """Return the digest binding the complete safe report payload."""

        return stable_hash(self._payload())

    @property
    def deleted_fingerprints(self) -> tuple[str, ...]:
        """Return deleted artifact fingerprints in canonical order."""

        return tuple(item.artifact_fingerprint for item in self.deleted_artifacts)

    @property
    def deletion_fingerprints(self) -> tuple[str, ...]:
        """Compatibility alias for :attr:`deleted_fingerprints`."""

        return self.deleted_fingerprints

    @property
    def remaining_artifact_set_fingerprint(self) -> str:
        """Compatibility alias for the retained-set fingerprint."""

        return self.remaining_fingerprint

    def _payload(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "deleted_artifact_count": self.deleted_artifact_count,
            "deleted_artifacts": [item.to_dict() for item in self.deleted_artifacts],
            "deletion_fingerprint": self.deletion_fingerprint,
            "format": AUDIT_RETENTION_FORMAT,
            "input_artifact_count": self.input_artifact_count,
            "input_fingerprint": self.input_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "remaining_fingerprint": self.remaining_fingerprint,
            "retained_artifact_count": self.retained_artifact_count,
            "retained_artifacts": [item.to_dict() for item in self.retained_artifacts],
            "version": AUDIT_RETENTION_VERSION,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete aggregate-only report with integrity digest."""

        return {**self._payload(), "integrity_digest": self.integrity_digest}

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report deterministically without raw values."""

        if indent is not None and (type(indent) is not int or not 0 <= indent <= 8):
            raise ValueError("retention report JSON indentation is invalid")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    def verify_remaining_artifacts(
        self,
        artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
    ) -> bool:
        """Verify that supplied artifacts are exactly the retained artifact set."""

        try:
            return artifact_set_fingerprint(artifacts) == self.remaining_fingerprint
        except Exception:  # noqa: BLE001 - verification must fail closed.
            return False

    def verify_remaining(
        self,
        artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
    ) -> bool:
        """Compatibility alias for :meth:`verify_remaining_artifacts`."""

        return self.verify_remaining_artifacts(artifacts)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditRetentionReport":
        """Load and integrity-check a safe retention report."""

        expected = {
            "as_of",
            "deleted_artifact_count",
            "deleted_artifacts",
            "deletion_fingerprint",
            "format",
            "input_artifact_count",
            "input_fingerprint",
            "integrity_digest",
            "policy_fingerprint",
            "remaining_fingerprint",
            "retained_artifact_count",
            "retained_artifacts",
            "version",
        }
        if not isinstance(data, Mapping):
            raise ValueError("retention report has missing or unknown fields")
        values = _snapshot_mapping(
            data,
            field_name="retention report",
            max_items=len(expected),
        )
        if set(values) != expected:
            raise ValueError("retention report has missing or unknown fields")
        if (
            type(values["format"]) is not str
            or values["format"] != AUDIT_RETENTION_FORMAT
        ):
            raise ValueError("retention report format is not supported")
        if (
            type(values["version"]) is not int
            or values["version"] != AUDIT_RETENTION_VERSION
        ):
            raise ValueError("retention report version is not supported")
        integrity_digest = _require_digest(
            values["integrity_digest"], "retention report integrity_digest"
        )

        retained = values["retained_artifacts"]
        deleted = values["deleted_artifacts"]
        if type(retained) is not list or type(deleted) is not list:
            raise TypeError("retention report artifact collections must be lists")
        if len(retained) + len(deleted) > MAX_AUDIT_RETENTION_ARTIFACTS:
            raise ValueError("retention report contains too many artifact summaries")
        if not all(isinstance(item, Mapping) for item in retained + deleted):
            raise TypeError(
                "retention report artifact collections must contain objects"
            )
        report = cls(
            as_of=values["as_of"],
            policy_fingerprint=values["policy_fingerprint"],
            input_fingerprint=values["input_fingerprint"],
            remaining_fingerprint=values["remaining_fingerprint"],
            deletion_fingerprint=values["deletion_fingerprint"],
            input_artifact_count=values["input_artifact_count"],
            retained_artifact_count=values["retained_artifact_count"],
            deleted_artifact_count=values["deleted_artifact_count"],
            retained_artifacts=tuple(
                RetainedArtifactSummary.from_dict(item) for item in retained
            ),
            deleted_artifacts=tuple(
                DeletionFingerprint.from_dict(item) for item in deleted
            ),
        )
        if not hmac.compare_digest(report.integrity_digest, integrity_digest):
            raise ValueError("retention report integrity digest mismatch")
        try:
            is_canonical = report.to_dict() == values
        except Exception:  # noqa: BLE001 - mappings may execute caller code.
            is_canonical = False
        if not is_canonical:
            raise ValueError("retention report is not the canonical representation")
        return report

    @classmethod
    def from_json(cls, payload: str) -> "AuditRetentionReport":
        """Load a strict JSON retention report."""

        if type(payload) is not str:
            raise TypeError("retention report JSON payload must be a string")
        try:
            encoded_size = len(payload.encode("utf-8"))
        except UnicodeError:
            raise ValueError("invalid retention report JSON payload") from None
        if encoded_size > MAX_AUDIT_RETENTION_JSON_BYTES:
            raise ValueError("retention report JSON payload exceeds the size limit")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (RecursionError, TypeError, ValueError):
            raise ValueError("invalid retention report JSON payload") from None
        if not isinstance(decoded, Mapping):
            raise ValueError("retention report JSON payload must be an object")
        return cls.from_dict(decoded)


def _coerce_policy(
    policy: AuditRetentionPolicy | Mapping[str, RetentionRule | timedelta],
) -> AuditRetentionPolicy:
    if type(policy) is AuditRetentionPolicy:
        return policy
    if isinstance(policy, Mapping):
        return AuditRetentionPolicy(policy)
    raise TypeError("retention policy must be an AuditRetentionPolicy or mapping")


def scrub_audit_artifacts(
    artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
    policy: AuditRetentionPolicy | Mapping[str, RetentionRule | timedelta],
    *,
    as_of: datetime | date | str | None = None,
    now: datetime | date | str | None = None,
) -> AuditRetentionReport:
    """Plan deterministic retention for a counts-only artifact iterable.

    ``as_of`` is required so repeated evaluations are reproducible. ``now`` is
    accepted as a compatibility alias, but both names cannot be supplied.
    The function only returns a safe deletion/retention report; it does not
    mutate the input iterable or perform storage or network operations.
    """

    if as_of is not None and now is not None:
        raise ValueError("supply only one retention evaluation timestamp")
    evaluation_value = as_of if as_of is not None else now
    if evaluation_value is None:
        raise ValueError("retention evaluation timestamp is required")

    evaluation_time = _coerce_datetime(
        evaluation_value,
        field_name="retention evaluation timestamp",
    )
    selected_policy = _coerce_policy(policy)
    normalized = _materialize_artifacts(artifacts)
    retained: list[RetainedArtifactSummary] = []
    deleted: list[DeletionFingerprint] = []

    for artifact in normalized:
        rule = selected_policy.rule_for(artifact.disposition)
        created_at = cast(datetime, artifact.created_at)
        if created_at > evaluation_time:
            raise ValueError(
                "audit artifact created_at must not follow the evaluation timestamp"
            )
        age = evaluation_time - created_at
        age_seconds = int(age.total_seconds())
        expired = (
            rule.action == "delete" and rule.max_age is not None and age >= rule.max_age
        )
        if expired:
            deleted.append(
                DeletionFingerprint(
                    artifact_fingerprint=artifact.fingerprint,
                    disposition=artifact.disposition,
                    age_seconds=age_seconds,
                    reason="age_expired",
                )
            )
            continue

        reason = "disposition_hold" if rule.action == "retain" else "within_retention"
        retained.append(
            RetainedArtifactSummary(
                artifact_fingerprint=artifact.fingerprint,
                disposition=artifact.disposition,
                age_seconds=age_seconds,
                count_total=artifact.count_total,
                metric_count=artifact.metric_count,
                reason=reason,
            )
        )

    retained.sort(key=lambda item: item.artifact_fingerprint)
    deleted.sort(key=lambda item: item.artifact_fingerprint)
    return AuditRetentionReport(
        as_of=_isoformat(evaluation_time),
        policy_fingerprint=selected_policy.fingerprint,
        input_fingerprint=_manifest_fingerprint(
            item.fingerprint for item in normalized
        ),
        remaining_fingerprint=_manifest_fingerprint(
            item.artifact_fingerprint for item in retained
        ),
        deletion_fingerprint=stable_hash(
            {
                "format": f"{AUDIT_RETENTION_FORMAT}.deletions",
                "artifacts": [item.to_dict() for item in deleted],
            }
        ),
        input_artifact_count=len(normalized),
        retained_artifact_count=len(retained),
        deleted_artifact_count=len(deleted),
        retained_artifacts=tuple(retained),
        deleted_artifacts=tuple(deleted),
    )


def verify_remaining_artifacts(
    report: AuditRetentionReport,
    artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
) -> bool:
    """Verify an artifact iterable against a retention report's remaining set."""

    if type(report) is not AuditRetentionReport:
        raise TypeError("retention report must be an AuditRetentionReport")
    return report.verify_remaining_artifacts(artifacts)


AuditArtifactRecord = AuditArtifact
RetentionPolicy = AuditRetentionPolicy
RetentionReport = AuditRetentionReport
scrub = scrub_audit_artifacts
