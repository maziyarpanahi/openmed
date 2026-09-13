"""Deterministic, privacy-safe comparisons for policy migrations.

Policy files are configuration, but a small change to one protection action can
change the behaviour of an entire de-identification run.  This module turns a
pair of policy versions into a stable, reviewable diff.  It understands
OpenMed :class:`~openmed.core.policy.PolicyProfile` objects and JSON-like
mappings, and does not include arbitrary policy values in its reports or error
messages.

The checker treats a migration that weakens a protection as incompatible.  A
caller can inspect the report first and then pass its deterministic,
report-bound acknowledgement token to :func:`check_policy_migration` after a
human has reviewed the change.  No network or external service is involved.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.policy import PolicyProfile, load_policy
from openmed.core.redaction_strength import ACTION_STRENGTH_ORDER, action_strength

MAX_POLICY_MIGRATION_DEPTH: Final = 32
MAX_POLICY_MIGRATION_ITEMS: Final = 10_000
MAX_POLICY_MIGRATION_JSON_BYTES: Final = 1_048_576
MAX_POLICY_MIGRATION_STRING_CHARS: Final = 65_536

_MAX_POLICY_MIGRATION_INTEGER: Final = (1 << 63) - 1
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_RE: Final = re.compile(r"^openmed-ack:[0-9a-f]{64}$")
_HASHED_PATH_RE: Final = re.compile(r"^<sha256:[0-9a-f]{12}>$")
_CHANGE_KINDS: Final = frozenset(
    {"action", "boolean", "collection", "metadata", "numeric", "structural"}
)
_CHANGE_REASON_RULES: Final = {
    "boolean protection setting became stronger": ("boolean", "stricter"),
    "boolean protection setting became weaker": ("boolean", "incompatible"),
    "boolean protection setting is unchanged": ("boolean", "compatible"),
    "boolean protection setting changed type": ("boolean", "incompatible"),
    "boolean protection setting changed without a known safe direction": (
        "boolean",
        "incompatible",
    ),
    "effective redaction action is unchanged": ("action", "compatible"),
    "non-behavioural policy metadata changed": ("metadata", "compatible"),
    "numeric protection threshold became stronger": ("numeric", "stricter"),
    "numeric protection threshold became weaker": ("numeric", "incompatible"),
    "numeric protection threshold changed type": ("numeric", "incompatible"),
    "numeric protection threshold is unchanged": ("numeric", "compatible"),
    "numeric protection threshold was added or removed": (
        "numeric",
        "incompatible",
    ),
    "numeric protection setting changed without a known safe direction": (
        "numeric",
        "incompatible",
    ),
    "policy behaviour changed without a compatible protection mapping": (
        "structural",
        "incompatible",
    ),
    "policy schema version changed": ("structural", "incompatible"),
    "protected collection changed in both directions": (
        "collection",
        "incompatible",
    ),
    "protected collection gained coverage": ("collection", "stricter"),
    "protected collection lost coverage": ("collection", "incompatible"),
    "protection cascade order changed": ("collection", "incompatible"),
    "protection collection changed type": ("collection", "incompatible"),
    "protection collection order is compatible": ("collection", "compatible"),
    "protection collection changed without a known safe direction": (
        "collection",
        "incompatible",
    ),
    "protection setting was added or removed": ("structural", "incompatible"),
    "redaction action became stronger": ("action", "stricter"),
    "redaction action became weaker": ("action", "incompatible"),
    "redaction action is missing or unsupported": ("action", "incompatible"),
}
_CHANGE_REASONS: Final = frozenset(_CHANGE_REASON_RULES)

__all__ = [
    "ChangeClassification",
    "MigrationClassification",
    "MAX_POLICY_MIGRATION_DEPTH",
    "MAX_POLICY_MIGRATION_ITEMS",
    "MAX_POLICY_MIGRATION_JSON_BYTES",
    "MAX_POLICY_MIGRATION_STRING_CHARS",
    "PolicyChange",
    "PolicyMigrationAcknowledgementRequired",
    "PolicyMigrationApprovalRequired",
    "PolicyMigrationError",
    "PolicyMigrationReport",
    "acknowledgement_token_for",
    "check_policy_migration",
    "compare_policies",
    "compare_policy_versions",
    "enforce_policy_migration",
    "policy_migration_report",
    "validate_policy_migration",
]


class MigrationClassification(str, Enum):
    """Overall relationship between two policy versions."""

    COMPATIBLE = "compatible"
    STRICTER = "stricter"
    INCOMPATIBLE = "incompatible"


ChangeClassification = MigrationClassification


class PolicyMigrationError(ValueError):
    """Base class for safe policy-migration validation errors."""


class PolicyMigrationAcknowledgementRequired(PolicyMigrationError):
    """Raised when an unacknowledged migration weakens a protection."""

    def __init__(self, report: "PolicyMigrationReport") -> None:
        self.report = report
        super().__init__(
            "policy migration requires a human acknowledgement before it can be applied"
        )


PolicyMigrationApprovalRequired = PolicyMigrationAcknowledgementRequired


@dataclass(frozen=True)
class PolicyChange:
    """One privacy-safe change in a policy comparison.

    ``before`` and ``after`` are deliberately limited to safe configuration
    scalars such as actions, booleans, and numeric protection thresholds.
    Arbitrary strings and containers are represented by a digest or a count;
    source policy values are never copied into this object.
    """

    path: tuple[str, ...]
    kind: str
    classification: MigrationClassification
    before_present: bool
    after_present: bool
    before: Any = None
    after: Any = None
    before_digest: str | None = None
    after_digest: str | None = None
    reason: str = ""
    weakens_protection: bool = False

    def __post_init__(self) -> None:
        if type(self.path) is not tuple or not _is_safe_report_path(self.path):
            raise PolicyMigrationError("policy change path is invalid")
        if type(self.kind) is not str or self.kind not in _CHANGE_KINDS:
            raise PolicyMigrationError("policy change kind is invalid")
        if type(self.classification) is not MigrationClassification:
            raise PolicyMigrationError("policy change classification is invalid")
        if (
            type(self.before_present) is not bool
            or type(self.after_present) is not bool
        ):
            raise PolicyMigrationError("policy change presence flags are invalid")
        if type(self.weakens_protection) is not bool:
            raise PolicyMigrationError("policy change weakening flag is invalid")
        if self.weakens_protection != (
            self.classification is MigrationClassification.INCOMPATIBLE
        ):
            raise PolicyMigrationError("policy change weakening state is inconsistent")
        if type(self.reason) is not str or self.reason not in _CHANGE_REASONS:
            raise PolicyMigrationError("policy change reason is invalid")
        if _CHANGE_REASON_RULES[self.reason] != (
            self.kind,
            self.classification.value,
        ):
            raise PolicyMigrationError("policy change reason is inconsistent")

        before = _normalize_report_value(
            self.before,
            present=self.before_present,
            kind=self.kind,
        )
        after = _normalize_report_value(
            self.after,
            present=self.after_present,
            kind=self.kind,
        )
        _validate_presence_digest(
            present=self.before_present,
            digest=self.before_digest,
            field_name="before",
        )
        _validate_presence_digest(
            present=self.after_present,
            digest=self.after_digest,
            field_name="after",
        )
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)

    @property
    def path_key(self) -> str:
        """Return the safe dotted path used in reports and review output."""

        return ".".join(self.path) if self.path else "$"

    @property
    def change_type(self) -> str:
        """Backward-compatible descriptive alias for :attr:`kind`."""

        return self.kind

    @property
    def is_weakening(self) -> bool:
        """Return whether the change reduces or may reduce protection."""

        return self.weakens_protection

    @property
    def before_value(self) -> Any:
        """Return the safe before value, if one is available."""

        return self.before

    @property
    def after_value(self) -> Any:
        """Return the safe after value, if one is available."""

        return self.after

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-value-free change record."""

        return {
            "path": list(self.path),
            "key": self.path_key,
            "kind": self.kind,
            "classification": self.classification.value,
            "before_present": self.before_present,
            "after_present": self.after_present,
            "before": _report_value_to_dict(self.before),
            "after": _report_value_to_dict(self.after),
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "reason": self.reason,
            "weakens_protection": self.weakens_protection,
        }


@dataclass(frozen=True)
class PolicyMigrationReport:
    """Stable result of comparing two policy versions.

    The report contains content digests rather than policy payloads.  For an
    incompatible report, :attr:`acknowledgement_token` is a deterministic token
    bound to both digests and the safe change paths.  It is intended to be
    copied only after a human review of the report.
    """

    before_digest: str
    after_digest: str
    classification: MigrationClassification
    changes: tuple[PolicyChange, ...] = ()
    acknowledged: bool = False
    acknowledgement_token: str | None = None

    def __post_init__(self) -> None:
        if type(self.before_digest) is not str or not _DIGEST_RE.fullmatch(
            self.before_digest
        ):
            raise PolicyMigrationError("before_digest is invalid")
        if type(self.after_digest) is not str or not _DIGEST_RE.fullmatch(
            self.after_digest
        ):
            raise PolicyMigrationError("after_digest is invalid")
        if type(self.classification) is not MigrationClassification:
            raise PolicyMigrationError("migration classification is invalid")
        if type(self.acknowledged) is not bool:
            raise PolicyMigrationError("migration acknowledgement state is invalid")
        if type(self.changes) not in {list, tuple} or any(
            type(change) is not PolicyChange for change in self.changes
        ):
            raise PolicyMigrationError("migration changes are invalid")

        ordered = tuple(sorted(self.changes, key=_change_sort_key))
        object.__setattr__(self, "changes", ordered)
        if self.classification is not _overall_classification(ordered):
            raise PolicyMigrationError("migration classification is inconsistent")
        if not ordered and self.before_digest != self.after_digest:
            raise PolicyMigrationError("unchanged migration digests are inconsistent")
        if ordered and self.before_digest == self.after_digest:
            raise PolicyMigrationError("changed migration digests are inconsistent")

        needs_acknowledgement = any(change.weakens_protection for change in ordered)
        expected_token = (
            _acknowledgement_token(
                self.before_digest,
                self.after_digest,
                ordered,
            )
            if needs_acknowledgement
            else None
        )
        if self.acknowledgement_token != expected_token:
            raise PolicyMigrationError("migration acknowledgement token is invalid")
        if self.acknowledgement_token is not None and not _TOKEN_RE.fullmatch(
            self.acknowledgement_token
        ):
            raise PolicyMigrationError("migration acknowledgement token is invalid")
        if self.acknowledged and not needs_acknowledgement:
            raise PolicyMigrationError(
                "migration acknowledgement state is inconsistent"
            )

    @property
    def requires_acknowledgement(self) -> bool:
        """Return whether a human acknowledgement is required."""

        return any(change.weakens_protection for change in self.changes)

    @property
    def requires_approval(self) -> bool:
        """Alias for :attr:`requires_acknowledgement`."""

        return self.requires_acknowledgement

    @property
    def approved(self) -> bool:
        """Return whether the report can be applied under the migration gate."""

        return not self.requires_acknowledgement or self.acknowledged

    @property
    def is_compatible(self) -> bool:
        """Return whether the overall classification is compatible."""

        return self.classification is MigrationClassification.COMPATIBLE

    @property
    def is_stricter(self) -> bool:
        """Return whether the migration only strengthens protections."""

        return self.classification is MigrationClassification.STRICTER

    @property
    def is_incompatible(self) -> bool:
        """Return whether the migration contains an incompatible change."""

        return self.classification is MigrationClassification.INCOMPATIBLE

    @property
    def weakened_changes(self) -> tuple[PolicyChange, ...]:
        """Return changes that require human acknowledgement."""

        return tuple(change for change in self.changes if change.weakens_protection)

    @property
    def protection_changes(self) -> tuple[PolicyChange, ...]:
        """Return changes that affect protection semantics."""

        return tuple(
            change
            for change in self.changes
            if change.kind in {"action", "boolean", "numeric", "collection"}
        )

    @property
    def summary(self) -> dict[str, int]:
        """Return deterministic aggregate counts for the migration."""

        return {
            "changes": len(self.changes),
            "compatible_changes": sum(
                change.classification is MigrationClassification.COMPATIBLE
                for change in self.changes
            ),
            "stricter_changes": sum(
                change.classification is MigrationClassification.STRICTER
                for change in self.changes
            ),
            "incompatible_changes": sum(
                change.classification is MigrationClassification.INCOMPATIBLE
                for change in self.changes
            ),
            "weakened_changes": len(self.weakened_changes),
        }

    @property
    def ack_token(self) -> str | None:
        """Short alias for :attr:`acknowledgement_token`."""

        return self.acknowledgement_token

    def with_acknowledgement(self, token: str) -> "PolicyMigrationReport":
        """Return an acknowledged copy only when the token matches this report."""

        expected = self.acknowledgement_token
        if expected is None or not _token_matches(token, expected):
            raise PolicyMigrationAcknowledgementRequired(self)
        return replace(self, acknowledged=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without raw policy values."""

        return {
            "schema_version": 1,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "classification": self.classification.value,
            "approved": self.approved,
            "acknowledged": self.acknowledged,
            "acknowledgement_required": self.requires_acknowledgement,
            "acknowledgement_token": self.acknowledgement_token,
            "summary": self.summary,
            "changes": [change.to_dict() for change in self.changes],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report as deterministic JSON."""

        if indent is not None and (type(indent) is not int or not 0 <= indent <= 8):
            raise PolicyMigrationError("report JSON indentation is invalid")
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        """Render a compact review summary containing no policy payloads."""

        lines = [
            "## Policy Migration Review",
            "",
            f"- Classification: **{self.classification.value}**",
            f"- Changes: {len(self.changes)}",
            f"- Acknowledgement required: {'yes' if self.requires_acknowledgement else 'no'}",
            f"- Approved: {'yes' if self.approved else 'no'}",
            "",
            "| Path | Kind | Classification | Reason |",
            "|---|---|---|---|",
        ]
        if not self.changes:
            lines.append("| — | — | compatible | no changes |")
        else:
            lines.extend(
                f"| `{change.path_key}` | {change.kind} | "
                f"{change.classification.value} | {change.reason} |"
                for change in self.changes
            )
        return "\n".join(lines)


def compare_policy_versions(
    before: PolicyProfile | Mapping[str, Any] | str | Path,
    after: PolicyProfile | Mapping[str, Any] | str | Path,
    *,
    acknowledgement_token: str | None = None,
    require_acknowledgement: bool = False,
) -> PolicyMigrationReport:
    """Compare two local policy versions.

    Args:
        before: An OpenMed ``PolicyProfile``, mapping, JSON string, or local
            JSON path representing the old policy.
        after: The corresponding new policy representation.
        acknowledgement_token: The report-bound token returned on an
            incompatible report.  It is checked without being included in any
            error message or report field supplied by the caller.
        require_acknowledgement: Raise when the migration weakens protection
            and the token is absent or does not match.

    Returns:
        A deterministic :class:`PolicyMigrationReport`.

    Raises:
        PolicyMigrationAcknowledgementRequired: If enforcement is requested
            for an unacknowledged weakening.
        PolicyMigrationError: If either input is not a supported local policy
            representation.
    """

    before_payload = _load_policy_input(before)
    after_payload = _load_policy_input(after)
    before_digest = _digest(before_payload)
    after_digest = _digest(after_payload)
    changes = _compare_payloads(before_payload, after_payload)
    classification = _overall_classification(changes)
    expected_token = _acknowledgement_token(
        before_digest,
        after_digest,
        changes,
    )
    needs_ack = any(change.weakens_protection for change in changes)
    acknowledged = bool(
        needs_ack
        and acknowledgement_token is not None
        and _token_matches(acknowledgement_token, expected_token)
    )
    report = PolicyMigrationReport(
        before_digest=before_digest,
        after_digest=after_digest,
        classification=classification,
        changes=changes,
        acknowledged=acknowledged,
        acknowledgement_token=expected_token if needs_ack else None,
    )
    if (
        require_acknowledgement
        and report.requires_acknowledgement
        and not report.approved
    ):
        raise PolicyMigrationAcknowledgementRequired(report)
    return report


def check_policy_migration(
    before: PolicyProfile | Mapping[str, Any] | str | Path,
    after: PolicyProfile | Mapping[str, Any] | str | Path,
    *,
    acknowledgement_token: str | None = None,
) -> PolicyMigrationReport:
    """Compare policies and enforce acknowledgement for weakening changes."""

    return compare_policy_versions(
        before,
        after,
        acknowledgement_token=acknowledgement_token,
        require_acknowledgement=True,
    )


def acknowledgement_token_for(report: PolicyMigrationReport) -> str | None:
    """Return the safe, report-bound token required for an incompatible diff."""

    if type(report) is not PolicyMigrationReport:
        raise TypeError("report must be a PolicyMigrationReport")
    return report.acknowledgement_token


compare_policies = compare_policy_versions
enforce_policy_migration = check_policy_migration
policy_migration_report = compare_policy_versions
validate_policy_migration = check_policy_migration


_ACTION_VALUES = frozenset(ACTION_STRENGTH_ORDER)
_ACTION_CONTAINER_KEYS = frozenset(
    {
        "action",
        "actions",
        "policy_label_actions",
        "redaction_action",
        "redaction_actions",
        "protection_action",
        "protection_actions",
        "rules",
        "protections",
    }
)
_ACTION_FIELD_KEYS = frozenset(
    {
        "action",
        "default_action",
        "minimum_action",
        "redaction",
        "redaction_action",
        "protection",
        "protection_action",
        "strategy",
    }
)
_STRONG_TRUE_KEYS = frozenset(
    {
        "enabled",
        "enforce",
        "fail_closed",
        "mandatory",
        "protected",
        "required",
        "safety_sweep_mandatory",
        "strict_no_leak",
    }
)
_WEAK_TRUE_KEYS = frozenset(
    {
        "allow_default_fallthrough",
        "allow_unredacted",
        "include_raw",
        "keep_mapping",
        "raw_values",
        "reversible_id",
        "retain_original",
        "retain_mapping",
        "store_original",
    }
)
_STRONGER_HIGHER_KEYS = frozenset(
    {
        "k",
        "l",
        "min_k",
        "min_l",
        "minimum_k",
        "minimum_l",
        "protection_level",
        "severity",
        "strength",
        "target_k",
        "target_l",
    }
)
_STRONGER_LOWER_KEYS = frozenset(
    {
        "delta",
        "epsilon",
        "max_false_negative",
        "max_risk",
        "max_suppression_rate",
        "risk_budget",
        "retention",
        "retention_days",
        "threshold",
        "window",
    }
)
_PROTECTION_COLLECTION_KEYS = frozenset(
    {
        "cascade",
        "excluded_labels",
        "forced_cascade_tiers",
        "ignored_labels",
        "protected_labels",
        "redact_labels",
        "required_labels",
        "sensitive_labels",
    }
)
_SCHEMA_VERSION_KEYS = frozenset({"schema_version", "version"})
_METADATA_KEYS = frozenset(
    {
        "comment",
        "description",
        "display_name",
        "metadata",
        "name",
        "source",
        "title",
    }
)
_PROTECTION_CONTEXT_KEYS = frozenset(
    {
        "action",
        "actions",
        "cascade",
        "detector",
        "detectors",
        "forced_cascade_tiers",
        "labels",
        "pii",
        "privacy",
        "protected",
        "protection",
        "protections",
        "recall_floors",
        "redact",
        "redaction",
        "redactions",
        "required",
        "rules",
        "safety",
        "security",
        "sensitive",
        "threshold",
    }
)
_SAFE_PATH_KEYS = (
    _ACTION_CONTAINER_KEYS
    | _ACTION_FIELD_KEYS
    | _STRONG_TRUE_KEYS
    | _WEAK_TRUE_KEYS
    | _STRONGER_HIGHER_KEYS
    | _STRONGER_LOWER_KEYS
    | _PROTECTION_COLLECTION_KEYS
    | _SCHEMA_VERSION_KEYS
    | _METADATA_KEYS
    | _PROTECTION_CONTEXT_KEYS
)
_MISSING = object()


def _is_safe_report_path(path: tuple[str, ...]) -> bool:
    if len(path) > MAX_POLICY_MIGRATION_DEPTH:
        return False
    return all(
        type(segment) is str
        and (
            segment in CANONICAL_LABELS
            or segment.lower() in _SAFE_PATH_KEYS
            or _HASHED_PATH_RE.fullmatch(segment) is not None
        )
        for segment in path
    )


def _normalize_report_value(value: Any, *, present: bool, kind: str) -> Any:
    if not present:
        if value is not None:
            raise PolicyMigrationError("absent policy change value is invalid")
        return None
    if value is None or (type(value) is str and value == "<redacted>"):
        return value
    if kind == "action" and type(value) is str and value in ACTION_STRENGTH_ORDER:
        return value
    if kind == "boolean" and type(value) is bool:
        return value
    if kind == "numeric" and type(value) in {int, float}:
        if type(value) is int and abs(value) > _MAX_POLICY_MIGRATION_INTEGER:
            raise PolicyMigrationError("policy change numeric value is invalid")
        if type(value) is float and not math.isfinite(value):
            raise PolicyMigrationError("policy change numeric value is invalid")
        return value
    if type(value) is dict and set(value) in ({"count"}, {"key_count"}):
        count = next(iter(value.values()))
        if type(count) is not int or not 0 <= count <= MAX_POLICY_MIGRATION_ITEMS:
            raise PolicyMigrationError("policy change aggregate value is invalid")
        return MappingProxyType(dict(value))
    raise PolicyMigrationError("policy change value is invalid")


def _report_value_to_dict(value: Any) -> Any:
    return dict(value) if isinstance(value, MappingProxyType) else value


def _validate_presence_digest(
    *,
    present: bool,
    digest: str | None,
    field_name: str,
) -> None:
    if present:
        if type(digest) is not str or not _DIGEST_RE.fullmatch(digest):
            raise PolicyMigrationError(f"{field_name} digest is invalid")
    elif digest is not None:
        raise PolicyMigrationError(f"absent {field_name} digest is invalid")


def _load_policy_input(
    value: PolicyProfile | Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    payload: Any
    if type(value) is PolicyProfile:
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = value
    elif isinstance(value, Path):
        payload = _read_policy_path(value)
    elif type(value) is str:
        try:
            input_size = len(value.encode("utf-8"))
        except UnicodeError:
            raise PolicyMigrationError("policy input is invalid") from None
        if input_size > MAX_POLICY_MIGRATION_JSON_BYTES:
            raise PolicyMigrationError("policy input exceeds the supported size limit")
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            payload = _parse_policy_json(stripped)
        else:
            candidate = Path(value)
            try:
                is_file = candidate.is_file()
            except (OSError, ValueError):
                is_file = False
            if is_file:
                payload = _read_policy_path(candidate)
            else:
                try:
                    payload = load_policy(value).to_dict()
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    raise PolicyMigrationError(
                        "policy input must be a mapping, JSON document, local JSON path, or policy name"
                    ) from None
    else:
        raise PolicyMigrationError(
            "policy input must be a PolicyProfile, mapping, JSON document, or path"
        )

    if not isinstance(payload, Mapping):
        raise PolicyMigrationError("policy document must contain an object")
    try:
        normalized = _normalize_json_value(payload)
        if (
            len(_canonical_json(normalized).encode("utf-8"))
            > MAX_POLICY_MIGRATION_JSON_BYTES
        ):
            raise ValueError("normalized policy exceeds the supported size limit")
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise PolicyMigrationError(
            "policy document contains an unsupported value"
        ) from None
    if type(normalized) is not dict:
        raise PolicyMigrationError("policy document must contain an object")
    return normalized


def _read_policy_path(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_POLICY_MIGRATION_JSON_BYTES + 1)
    except OSError:
        raise PolicyMigrationError("policy JSON could not be read") from None
    if len(payload) > MAX_POLICY_MIGRATION_JSON_BYTES:
        raise PolicyMigrationError("policy input exceeds the supported size limit")
    return _parse_policy_json(payload)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate policy JSON field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite policy JSON number")


def _parse_policy_json(value: str | bytes) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise PolicyMigrationError("policy JSON could not be parsed") from None


def _normalize_json_value(
    value: Any,
    *,
    depth: int = 0,
    item_budget: list[int] | None = None,
    text_budget: list[int] | None = None,
) -> Any:
    if item_budget is None:
        item_budget = [MAX_POLICY_MIGRATION_ITEMS]
    if text_budget is None:
        text_budget = [MAX_POLICY_MIGRATION_JSON_BYTES]
    if depth > MAX_POLICY_MIGRATION_DEPTH or item_budget[0] <= 0:
        raise ValueError("policy document exceeds the supported complexity limit")
    item_budget[0] -= 1

    if isinstance(value, Mapping):
        items = list(itertools.islice(value.items(), item_budget[0] + 1))
        if len(items) > item_budget[0]:
            raise ValueError("policy document exceeds the supported item limit")
        normalized: dict[str, Any] = {}
        for key, item in items:
            if type(key) is not str:
                raise TypeError("policy keys must be strings")
            if len(key) > MAX_POLICY_MIGRATION_STRING_CHARS:
                raise ValueError("policy key exceeds the supported length limit")
            text_budget[0] -= len(key.encode("utf-8"))
            if text_budget[0] < 0:
                raise ValueError("policy text exceeds the supported size limit")
            if key in normalized:
                raise ValueError("policy keys must be unique")
            normalized[key] = _normalize_json_value(
                item,
                depth=depth + 1,
                item_budget=item_budget,
                text_budget=text_budget,
            )
        return {key: normalized[key] for key in sorted(normalized)}
    if type(value) in {list, tuple, set, frozenset}:
        items = list(itertools.islice(value, item_budget[0] + 1))
        if len(items) > item_budget[0]:
            raise ValueError("policy document exceeds the supported item limit")
        normalized_items = [
            _normalize_json_value(
                item,
                depth=depth + 1,
                item_budget=item_budget,
                text_budget=text_budget,
            )
            for item in items
        ]
        if type(value) in {list, tuple}:
            return normalized_items
        return sorted(normalized_items, key=_canonical_json)
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        if len(value) > MAX_POLICY_MIGRATION_STRING_CHARS:
            raise ValueError("policy string exceeds the supported length limit")
        text_budget[0] -= len(value.encode("utf-8"))
        if text_budget[0] < 0:
            raise ValueError("policy text exceeds the supported size limit")
        return value
    if type(value) is int:
        if abs(value) > _MAX_POLICY_MIGRATION_INTEGER:
            raise ValueError("policy integer exceeds the supported range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("policy numbers must be finite")
        return value
    raise TypeError("policy value type is unsupported")


def _compare_payloads(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[PolicyChange, ...]:
    before_values = dict(_flatten(before))
    after_values = dict(_flatten(after))
    changes: list[PolicyChange] = []
    for path in sorted(set(before_values) | set(after_values), key=_path_sort_key):
        before_value = before_values.get(path, _MISSING)
        after_value = after_values.get(path, _MISSING)
        if before_value is not _MISSING and after_value is not _MISSING:
            if _canonical_json(before_value) == _canonical_json(after_value):
                continue
        kind, classification, weakens, reason = _classify_change(
            path,
            before_value,
            after_value,
            before,
            after,
        )
        changes.append(
            PolicyChange(
                path=_safe_path(path),
                kind=kind,
                classification=classification,
                before_present=before_value is not _MISSING,
                after_present=after_value is not _MISSING,
                before=_safe_value(path, before_value, kind),
                after=_safe_value(path, after_value, kind),
                before_digest=(
                    None if before_value is _MISSING else _digest(before_value)
                ),
                after_digest=None if after_value is _MISSING else _digest(after_value),
                reason=reason,
                weakens_protection=weakens,
            )
        )
    return tuple(sorted(changes, key=_change_sort_key))


def _flatten(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        if not value:
            return [(path, {})]
        flattened: list[tuple[tuple[str, ...], Any]] = []
        for key in sorted(value):
            flattened.extend(_flatten(value[key], (*path, key)))
        return flattened
    if isinstance(value, list):
        return [(path, value)]
    return [(path, value)]


def _classify_change(
    path: tuple[str, ...],
    before: Any,
    after: Any,
    before_payload: Mapping[str, Any],
    after_payload: Mapping[str, Any],
) -> tuple[str, MigrationClassification, bool, str]:
    before_present = before is not _MISSING
    after_present = after is not _MISSING
    key = _leaf_key(path)
    if key in _SCHEMA_VERSION_KEYS:
        return (
            "structural",
            MigrationClassification.INCOMPATIBLE,
            True,
            "policy schema version changed",
        )
    action_path = _is_action_path(path)
    if action_path and (
        _looks_like_action(before)
        or _looks_like_action(after)
        or not before_present
        or not after_present
    ):
        return _classify_action_change(
            path,
            before,
            after,
            before_payload,
            after_payload,
        )

    if isinstance(before, bool) or isinstance(after, bool):
        orientation = _boolean_orientation(path)
        if orientation:
            return _classify_boolean_change(
                before,
                after,
                orientation,
            )
        if _is_protection_path(path):
            return (
                "boolean",
                MigrationClassification.INCOMPATIBLE,
                True,
                "boolean protection setting changed without a known safe direction",
            )

    if (
        isinstance(before, (int, float))
        and not isinstance(before, bool)
        or isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        direction = _numeric_direction(path)
        if direction:
            return _classify_numeric_change(before, after, direction)
        if _is_protection_path(path):
            return (
                "numeric",
                MigrationClassification.INCOMPATIBLE,
                True,
                "numeric protection setting changed without a known safe direction",
            )

    if isinstance(before, list) or isinstance(after, list):
        collection_direction = _collection_direction(path)
        if collection_direction:
            return _classify_collection_change(
                before,
                after,
                collection_direction,
                path,
            )
        if _is_protection_path(path):
            return (
                "collection",
                MigrationClassification.INCOMPATIBLE,
                True,
                "protection collection changed without a known safe direction",
            )

    if before_present != after_present and _is_protection_path(path):
        return (
            "structural",
            MigrationClassification.INCOMPATIBLE,
            True,
            "protection setting was added or removed",
        )

    if _is_metadata_path(path):
        return (
            "metadata",
            MigrationClassification.COMPATIBLE,
            False,
            "non-behavioural policy metadata changed",
        )
    if _is_protection_path(path) or key not in _METADATA_KEYS:
        return (
            "structural",
            MigrationClassification.INCOMPATIBLE,
            True,
            "policy behaviour changed without a compatible protection mapping",
        )
    return (
        "metadata",
        MigrationClassification.COMPATIBLE,
        False,
        "non-behavioural policy metadata changed",
    )


def _classify_action_change(
    path: tuple[str, ...],
    before: Any,
    after: Any,
    before_payload: Mapping[str, Any],
    after_payload: Mapping[str, Any],
) -> tuple[str, MigrationClassification, bool, str]:
    before_action = _effective_action(path, before, before_payload)
    after_action = _effective_action(path, after, after_payload)
    if before_action in _ACTION_VALUES and after_action in _ACTION_VALUES:
        before_strength = action_strength(before_action)
        after_strength = action_strength(after_action)
        if after_strength > before_strength:
            return (
                "action",
                MigrationClassification.STRICTER,
                False,
                "redaction action became stronger",
            )
        if after_strength < before_strength:
            return (
                "action",
                MigrationClassification.INCOMPATIBLE,
                True,
                "redaction action became weaker",
            )
        return (
            "action",
            MigrationClassification.COMPATIBLE,
            False,
            "effective redaction action is unchanged",
        )
    return (
        "action",
        MigrationClassification.INCOMPATIBLE,
        True,
        "redaction action is missing or unsupported",
    )


def _classify_boolean_change(
    before: Any,
    after: Any,
    orientation: int,
) -> tuple[str, MigrationClassification, bool, str]:
    if (before is not _MISSING and type(before) is not bool) or (
        after is not _MISSING and type(after) is not bool
    ):
        return (
            "boolean",
            MigrationClassification.INCOMPATIBLE,
            True,
            "boolean protection setting changed type",
        )
    before_bool = False if before is _MISSING else before
    after_bool = False if after is _MISSING else after
    if before_bool == after_bool:
        return (
            "boolean",
            MigrationClassification.COMPATIBLE,
            False,
            "boolean protection setting is unchanged",
        )
    stronger = after_bool if orientation > 0 else not after_bool
    if stronger:
        return (
            "boolean",
            MigrationClassification.STRICTER,
            False,
            "boolean protection setting became stronger",
        )
    return (
        "boolean",
        MigrationClassification.INCOMPATIBLE,
        True,
        "boolean protection setting became weaker",
    )


def _classify_numeric_change(
    before: Any,
    after: Any,
    direction: int,
) -> tuple[str, MigrationClassification, bool, str]:
    if before is _MISSING or after is _MISSING:
        return (
            "numeric",
            MigrationClassification.INCOMPATIBLE,
            True,
            "numeric protection threshold was added or removed",
        )
    if type(before) not in {int, float} or type(after) not in {int, float}:
        return (
            "numeric",
            MigrationClassification.INCOMPATIBLE,
            True,
            "numeric protection threshold changed type",
        )
    if type(before) is not type(after):
        return (
            "numeric",
            MigrationClassification.INCOMPATIBLE,
            True,
            "numeric protection threshold changed type",
        )
    if after == before:
        return (
            "numeric",
            MigrationClassification.COMPATIBLE,
            False,
            "numeric protection threshold is unchanged",
        )
    stronger = after > before if direction > 0 else after < before
    if stronger:
        return (
            "numeric",
            MigrationClassification.STRICTER,
            False,
            "numeric protection threshold became stronger",
        )
    return (
        "numeric",
        MigrationClassification.INCOMPATIBLE,
        True,
        "numeric protection threshold became weaker",
    )


def _classify_collection_change(
    before: Any,
    after: Any,
    direction: int,
    path: tuple[str, ...],
) -> tuple[str, MigrationClassification, bool, str]:
    before_items = (
        [] if before is _MISSING else list(before) if isinstance(before, list) else None
    )
    after_items = (
        [] if after is _MISSING else list(after) if isinstance(after, list) else None
    )
    if before_items is None or after_items is None:
        return (
            "collection",
            MigrationClassification.INCOMPATIBLE,
            True,
            "protection collection changed type",
        )
    before_keys = {_canonical_json(item) for item in before_items}
    after_keys = {_canonical_json(item) for item in after_items}
    added = after_keys - before_keys
    removed = before_keys - after_keys
    if not added and not removed:
        if (
            path
            and _leaf_key(path) in {"cascade", "forced_cascade_tiers"}
            and before_items != after_items
        ):
            return (
                "collection",
                MigrationClassification.INCOMPATIBLE,
                True,
                "protection cascade order changed",
            )
        return (
            "collection",
            MigrationClassification.COMPATIBLE,
            False,
            "protection collection order is compatible",
        )
    stronger = bool(added) if direction > 0 else bool(removed)
    if added and removed:
        return (
            "collection",
            MigrationClassification.INCOMPATIBLE,
            True,
            "protected collection changed in both directions",
        )
    if stronger:
        return (
            "collection",
            MigrationClassification.STRICTER,
            False,
            "protected collection gained coverage",
        )
    return (
        "collection",
        MigrationClassification.INCOMPATIBLE,
        True,
        "protected collection lost coverage",
    )


def _effective_action(
    path: tuple[str, ...],
    value: Any,
    payload: Mapping[str, Any],
) -> str | None:
    if value is not _MISSING and isinstance(value, str):
        return value if value in _ACTION_VALUES else None
    default = payload.get("default_action", "keep")
    if isinstance(default, str) and default in _ACTION_VALUES:
        return (
            default
            if any(segment.lower() in _ACTION_CONTAINER_KEYS for segment in path)
            else "keep"
        )
    return "keep"


def _overall_classification(
    changes: Sequence[PolicyChange],
) -> MigrationClassification:
    if any(
        change.classification is MigrationClassification.INCOMPATIBLE
        for change in changes
    ):
        return MigrationClassification.INCOMPATIBLE
    if any(
        change.classification is MigrationClassification.STRICTER for change in changes
    ):
        return MigrationClassification.STRICTER
    return MigrationClassification.COMPATIBLE


def _boolean_orientation(path: tuple[str, ...]) -> int:
    key = _leaf_key(path)
    if key in _STRONG_TRUE_KEYS:
        return 1
    if key in _WEAK_TRUE_KEYS or key.startswith(("allow_", "retain_", "store_")):
        return -1
    if key == "enabled" and _is_protection_path(path):
        return 1
    return 0


def _numeric_direction(path: tuple[str, ...]) -> int:
    key = _leaf_key(path)
    if "recall_floors" in (segment.lower() for segment in path) or (
        "recall" in key
        and ("floor" in key or key.startswith(("min_", "minimum_", "target_")))
    ):
        return 1
    if "threshold" in key or "confidence" in key:
        return -1
    if key in _STRONGER_HIGHER_KEYS or key.startswith(("min_", "minimum_", "target_")):
        return 1
    if key in _STRONGER_LOWER_KEYS or key.startswith(("max_", "upper_")):
        return -1
    return 0


def _collection_direction(path: tuple[str, ...]) -> int:
    key = _leaf_key(path)
    if key in _PROTECTION_COLLECTION_KEYS:
        return -1 if key in {"excluded_labels", "ignored_labels"} else 1
    if key.startswith(("allow", "exclude", "ignore", "bypass")):
        return -1
    if key.startswith(("protect", "redact", "require", "sensitive")):
        return 1
    return 0


def _is_action_path(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    return _leaf_key(path) in _ACTION_FIELD_KEYS or any(
        segment.lower() in _ACTION_CONTAINER_KEYS for segment in path[:-1]
    )


def _is_metadata_path(path: tuple[str, ...]) -> bool:
    # A detector label such as NAME must not turn its rule configuration into
    # metadata. Unknown changes inside protection rules always fail closed.
    return not _is_protection_path(path) and any(
        segment.lower() in _METADATA_KEYS for segment in path
    )


def _is_protection_path(path: tuple[str, ...]) -> bool:
    return any(segment.lower() in _PROTECTION_CONTEXT_KEYS for segment in path)


def _leaf_key(path: tuple[str, ...]) -> str:
    return path[-1].lower() if path else ""


def _looks_like_action(value: Any) -> bool:
    return isinstance(value, str) and value in _ACTION_VALUES


def _safe_value(path: tuple[str, ...], value: Any, kind: str) -> Any:
    if value is _MISSING:
        return None
    if kind == "action" and isinstance(value, str) and value in _ACTION_VALUES:
        return value
    if kind == "boolean" and isinstance(value, bool):
        return value
    if (
        kind == "numeric"
        and _numeric_direction(path)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return value
    if kind == "collection" and isinstance(value, list):
        return {"count": len(value)}
    if isinstance(value, Mapping):
        return {"key_count": len(value)}
    if isinstance(value, list):
        return {"count": len(value)}
    if value is None:
        return None
    return "<redacted>"


def _safe_path(path: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for segment in path:
        if segment in CANONICAL_LABELS or segment.lower() in _SAFE_PATH_KEYS:
            result.append(segment)
        else:
            result.append(f"<sha256:{_digest(segment)[7:19]}>")
    return tuple(result)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _acknowledgement_token(
    before_digest: str,
    after_digest: str,
    changes: Sequence[PolicyChange],
) -> str:
    paths = "|".join(change.path_key for change in changes if change.weakens_protection)
    payload = f"openmed-policy-migration-v1\0{before_digest}\0{after_digest}\0{paths}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"openmed-ack:{digest}"


def _token_matches(token: str, expected: str) -> bool:
    if (
        type(token) is not str
        or type(expected) is not str
        or _TOKEN_RE.fullmatch(token) is None
        or _TOKEN_RE.fullmatch(expected) is None
    ):
        return False
    return hmac.compare_digest(token, expected)


def _path_sort_key(path: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(part) for part in path)


def _change_sort_key(change: PolicyChange) -> tuple[Any, ...]:
    return (
        change.path,
        change.kind,
        change.classification.value,
        change.before_digest or "",
        change.after_digest or "",
    )
