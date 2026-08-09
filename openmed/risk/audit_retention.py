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

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

from openmed.core.audit import stable_hash

__all__ = [
    "AUDIT_RETENTION_FORMAT",
    "AUDIT_RETENTION_VERSION",
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

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
_ACTIONS = frozenset({"delete", "retain"})
_MISSING = object()
_RAW_INPUT_FIELDS = frozenset(
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


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _require_digest(value: Any, field_name: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{field_name} must be a sha256:<hex> digest")
    return value


def _safe_name(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a safe identifier")
    normalized = value.strip().lower()
    if not _SAFE_NAME_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a safe identifier")
    return normalized


def _coerce_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from None
    else:
        raise TypeError(f"{field_name} must be an ISO-8601 timestamp")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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

    normalized: dict[str, int] = {}
    for key, count in value.items():
        name = _safe_name(key, field_name="audit artifact count name")
        if type(count) is not int or count < 0:
            raise ValueError("audit artifact counts must be non-negative integers")
        normalized[name] = count
    return tuple(sorted(normalized.items()))


def _mapping_value(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return _MISSING


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
            if not isinstance(self.max_age, timedelta):
                raise TypeError("retention max_age must be a timedelta or None")
            if self.max_age < timedelta(0):
                raise ValueError("retention max_age must not be negative")
        if action == "delete" and self.max_age is None:
            raise ValueError("delete retention rules require max_age")
        object.__setattr__(self, "action", action)

    @classmethod
    def days(cls, days: int, *, action: str = "delete") -> "RetentionRule":
        """Construct a rule from a whole number of retention days."""

        if type(days) is not int or days < 0:
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

        if not isinstance(data, Mapping) or set(data) != {
            "action",
            "max_age_seconds",
        }:
            raise ValueError("retention rule has missing or unknown fields")
        seconds = data["max_age_seconds"]
        if seconds is None:
            max_age = None
        elif (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and seconds >= 0
        ):
            max_age = timedelta(seconds=seconds)
        else:
            raise ValueError("retention max_age_seconds must be non-negative")
        return cls(max_age=max_age, action=data["action"])


@dataclass(frozen=True)
class AuditRetentionPolicy:
    """Explicit age and disposition rules for audit artifacts."""

    rules: Mapping[str, RetentionRule | timedelta]
    default_rule: RetentionRule | timedelta | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rules, Mapping) or not self.rules:
            raise ValueError("retention policy rules must be a non-empty mapping")

        normalized: dict[str, RetentionRule] = {}
        for disposition, rule in self.rules.items():
            name = _safe_name(disposition, field_name="retention disposition")
            if isinstance(rule, timedelta):
                rule = RetentionRule(max_age=rule)
            if not isinstance(rule, RetentionRule):
                raise TypeError(
                    "retention policy rules must contain RetentionRule values"
                )
            normalized[name] = rule

        default = self.default_rule
        if isinstance(default, timedelta):
            default = RetentionRule(max_age=default)
        if default is not None and not isinstance(default, RetentionRule):
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
        return rule

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

        return {
            "rules": {
                disposition: rule.to_dict()
                for disposition, rule in sorted(self.rules.items())
            },
            "default_rule": (
                self.default_rule.to_dict() if self.default_rule is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditRetentionPolicy":
        """Load a policy from its deterministic safe representation."""

        if not isinstance(data, Mapping) or set(data) != {"rules", "default_rule"}:
            raise ValueError("retention policy has missing or unknown fields")
        rules = data["rules"]
        if not isinstance(rules, Mapping):
            raise TypeError("retention policy rules must be a mapping")
        parsed_rules = {
            disposition: RetentionRule.from_dict(rule)
            for disposition, rule in rules.items()
            if isinstance(disposition, str) and isinstance(rule, Mapping)
        }
        if len(parsed_rules) != len(rules):
            raise ValueError("retention policy rules contain invalid entries")
        default = data["default_rule"]
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
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
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
        if any(isinstance(key, str) and key in _RAW_INPUT_FIELDS for key in data):
            raise ValueError("audit artifacts must contain counts only")

        artifact_id = _mapping_value(data, "artifact_id", "id", "key")
        created_at = _mapping_value(data, "created_at", "created", "timestamp")
        disposition = _mapping_value(data, "disposition")
        if artifact_id is _MISSING or created_at is _MISSING or disposition is _MISSING:
            raise ValueError("audit artifact requires id, created_at, and disposition")

        counts = _mapping_value(data, "counts", "counters", "metrics")
        if counts is _MISSING:
            count = _mapping_value(data, "count", "event_count")
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
                "created_at": _isoformat(self.created_at),
                "disposition": self.disposition,
                "counts": dict(self.counts),
            }
        )


def _coerce_artifact(value: AuditArtifact | Mapping[str, Any]) -> AuditArtifact:
    if isinstance(value, AuditArtifact):
        return value
    return AuditArtifact.from_mapping(value)


def _materialize_artifacts(
    artifacts: Iterable[AuditArtifact | Mapping[str, Any]],
) -> tuple[AuditArtifact, ...]:
    try:
        values = tuple(_coerce_artifact(item) for item in artifacts)
    except TypeError:
        raise TypeError(
            "audit artifacts must be an iterable of counts-only artifacts"
        ) from None
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
        if type(self.age_seconds) is not int or self.age_seconds < 0:
            raise ValueError("deletion age_seconds must be a non-negative integer")
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
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("deletion fingerprint has missing or unknown fields")
        return cls(
            artifact_fingerprint=data["artifact_fingerprint"],
            disposition=data["disposition"],
            age_seconds=data["age_seconds"],
            reason=data["reason"],
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
        if type(self.age_seconds) is not int or self.age_seconds < 0:
            raise ValueError("retained age_seconds must be a non-negative integer")
        if type(self.count_total) is not int or self.count_total < 0:
            raise ValueError("retained count_total must be a non-negative integer")
        if type(self.metric_count) is not int or self.metric_count < 0:
            raise ValueError("retained metric_count must be a non-negative integer")
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
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("retained artifact summary has missing or unknown fields")
        return cls(
            artifact_fingerprint=data["artifact_fingerprint"],
            disposition=data["disposition"],
            age_seconds=data["age_seconds"],
            count_total=data["count_total"],
            metric_count=data["metric_count"],
            reason=data["reason"],
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
        _coerce_datetime(self.as_of, field_name="retention report as_of")
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
            if type(value) is not int or value < 0:
                raise ValueError(f"retention report {name} must be non-negative")

        retained = tuple(self.retained_artifacts)
        deleted = tuple(self.deleted_artifacts)
        if not all(isinstance(item, RetainedArtifactSummary) for item in retained):
            raise TypeError(
                "retention report retained_artifacts must contain summaries"
            )
        if not all(isinstance(item, DeletionFingerprint) for item in deleted):
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
        expected_remaining = _manifest_fingerprint(
            item.artifact_fingerprint for item in retained
        )
        expected_deleted = stable_hash(
            {
                "format": f"{AUDIT_RETENTION_FORMAT}.deletions",
                "artifacts": [item.to_dict() for item in deleted],
            }
        )
        expected_input = _manifest_fingerprint(
            [
                *(item.artifact_fingerprint for item in retained),
                *(item.artifact_fingerprint for item in deleted),
            ]
        )
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
        except (TypeError, ValueError):
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
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("retention report has missing or unknown fields")
        if data["format"] != AUDIT_RETENTION_FORMAT:
            raise ValueError("retention report format is not supported")
        if data["version"] != AUDIT_RETENTION_VERSION:
            raise ValueError("retention report version is not supported")
        integrity_digest = _require_digest(
            data["integrity_digest"], "retention report integrity_digest"
        )
        payload = {key: data[key] for key in expected if key != "integrity_digest"}
        if stable_hash(payload) != integrity_digest:
            raise ValueError("retention report integrity digest mismatch")

        retained = data["retained_artifacts"]
        deleted = data["deleted_artifacts"]
        if not isinstance(retained, list) or not isinstance(deleted, list):
            raise TypeError("retention report artifact collections must be lists")
        if not all(isinstance(item, Mapping) for item in retained + deleted):
            raise TypeError(
                "retention report artifact collections must contain objects"
            )
        report = cls(
            as_of=data["as_of"],
            policy_fingerprint=data["policy_fingerprint"],
            input_fingerprint=data["input_fingerprint"],
            remaining_fingerprint=data["remaining_fingerprint"],
            deletion_fingerprint=data["deletion_fingerprint"],
            input_artifact_count=data["input_artifact_count"],
            retained_artifact_count=data["retained_artifact_count"],
            deleted_artifact_count=data["deleted_artifact_count"],
            retained_artifacts=tuple(
                RetainedArtifactSummary.from_dict(item) for item in retained
            ),
            deleted_artifacts=tuple(
                DeletionFingerprint.from_dict(item) for item in deleted
            ),
        )
        if report.to_dict() != dict(data):
            raise ValueError("retention report is not the canonical representation")
        return report

    @classmethod
    def from_json(cls, payload: str) -> "AuditRetentionReport":
        """Load a strict JSON retention report."""

        if type(payload) is not str:
            raise TypeError("retention report JSON payload must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            raise ValueError("invalid retention report JSON payload") from None
        if not isinstance(decoded, Mapping):
            raise ValueError("retention report JSON payload must be an object")
        return cls.from_dict(decoded)


def _coerce_policy(
    policy: AuditRetentionPolicy | Mapping[str, RetentionRule | timedelta],
) -> AuditRetentionPolicy:
    if isinstance(policy, AuditRetentionPolicy):
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
        age = evaluation_time - artifact.created_at
        age_seconds = max(0, int(age.total_seconds()))
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

    if not isinstance(report, AuditRetentionReport):
        raise TypeError("retention report must be an AuditRetentionReport")
    return report.verify_remaining_artifacts(artifacts)


AuditArtifactRecord = AuditArtifact
RetentionPolicy = AuditRetentionPolicy
RetentionReport = AuditRetentionReport
scrub = scrub_audit_artifacts
