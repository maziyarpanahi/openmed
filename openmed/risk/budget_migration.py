"""Deterministic, privacy-safe verification of DP budget migrations.

Budget migrations are configuration changes, but treating a new snapshot as a
fresh ledger can silently give a release more privacy budget than it had
before.  This module compares aggregate, one-entry-per-release snapshots and
fails closed when a release disappears, spent budget decreases, limits grow,
or the accounting method or policy fingerprint changes in place.

The verifier accepts the canonical ``entries`` representation and the
``compositions``/``policies`` representation emitted by
``DPGenerationBudgetAccountant.to_dict``.  It never needs a service or source
data.  Reports contain only numeric aggregates, hashes, and validated stable
identifiers; arbitrary input values are never copied into an exception or
report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BUDGET_MIGRATION_SCHEMA_VERSION",
    "BudgetEntry",
    "BudgetLedgerEntry",
    "BudgetLedgerSnapshot",
    "BudgetMigrationEntry",
    "BudgetMigrationError",
    "BudgetMigrationIssue",
    "BudgetMigrationRejected",
    "BudgetMigrationReport",
    "BudgetMigrationVerifier",
    "BudgetSnapshot",
    "check_budget_migration",
    "compare_budget_ledgers",
    "compare_budget_migration",
    "enforce_budget_migration",
    "validate_budget_migration",
    "verify_budget_ledger_migration",
    "verify_budget_migration",
]

BUDGET_MIGRATION_SCHEMA_VERSION = "openmed.dp_budget_migration.v1"
_DEFAULT_SNAPSHOT_SCHEMA_VERSION = "openmed.dp_budget_ledger.v1"
_MISSING = object()
_FLOAT_TOLERANCE = 1e-12
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_PHI_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}[-.)]\d{3}[-.]\d{4}\b"),
)

_RELEASE_KEYS = ("release_id", "release_identifier", "release", "context", "scope")
_COMPOSITION_KEYS = (
    "composition",
    "composition_method",
    "composition_rule",
)
_FINGERPRINT_KEYS = (
    "policy_fingerprint",
    "policy_digest",
    "fingerprint",
    "policy_hash",
)
_SPENT_EPSILON_KEYS = (
    "spent_epsilon",
    "epsilon_spent",
    "consumed_epsilon",
)
_SPENT_DELTA_KEYS = ("spent_delta", "delta_spent", "consumed_delta")
_LIMIT_EPSILON_KEYS = (
    "max_epsilon",
    "budget_epsilon",
    "epsilon_limit",
    "limit_epsilon",
)
_LIMIT_DELTA_KEYS = (
    "max_delta",
    "budget_delta",
    "delta_limit",
    "limit_delta",
)
_POLICY_FIELDS = (
    "scope",
    "name",
    "version",
    "schema_version",
    "max_epsilon",
    "max_delta",
    "composition",
    "composition_method",
    "composition_rule",
    "delta_prime",
)


class BudgetMigrationError(ValueError):
    """Raised when a budget snapshot is malformed or cannot be migrated."""


class BudgetMigrationRejected(BudgetMigrationError):
    """Raised when a valid migration would weaken or reset budget state."""

    def __init__(self, report: "BudgetMigrationReport") -> None:
        self.report = report
        super().__init__("budget migration failed monotonicity and safety checks")


@dataclass(frozen=True, slots=True)
class BudgetEntry:
    """Aggregate state for one stable release identifier.

    ``spent_*`` values are cumulative values at the snapshot boundary.  A
    ``sequence`` is optional for callers that use an append-only release
    sequence; when present it must not be reused by a later release.
    """

    release_id: str
    spent_epsilon: float
    spent_delta: float
    max_epsilon: float
    max_delta: float
    composition: str
    policy_fingerprint: str
    sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "release_id",
            _safe_identifier(self.release_id, field_name="release identifier"),
        )
        object.__setattr__(
            self,
            "spent_epsilon",
            _non_negative_float(self.spent_epsilon, field_name="spent epsilon"),
        )
        object.__setattr__(
            self,
            "spent_delta",
            _delta_float(self.spent_delta, field_name="spent delta"),
        )
        object.__setattr__(
            self,
            "max_epsilon",
            _non_negative_float(self.max_epsilon, field_name="epsilon limit"),
        )
        object.__setattr__(
            self,
            "max_delta",
            _delta_float(self.max_delta, field_name="delta limit"),
        )
        object.__setattr__(
            self,
            "composition",
            _safe_identifier(self.composition, field_name="composition method"),
        )
        object.__setattr__(
            self,
            "policy_fingerprint",
            _safe_identifier(
                self.policy_fingerprint,
                field_name="policy fingerprint",
            ),
        )
        if self.sequence is not None:
            object.__setattr__(
                self,
                "sequence",
                _positive_int(self.sequence, field_name="release sequence"),
            )
        if self.spent_epsilon > self.max_epsilon:
            raise BudgetMigrationError(
                "budget entry spent epsilon exceeds its configured limit"
            )
        if self.spent_delta > self.max_delta:
            raise BudgetMigrationError(
                "budget entry spent delta exceeds its configured limit"
            )

    @property
    def budget_epsilon(self) -> float:
        """Return the epsilon limit using the budget terminology."""

        return self.max_epsilon

    @property
    def budget_delta(self) -> float:
        """Return the delta limit using the budget terminology."""

        return self.max_delta

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        release_id: str | None = None,
        composition: str | None = None,
        policy_fingerprint: str | None = None,
        policy: Mapping[str, Any] | None = None,
    ) -> "BudgetEntry":
        """Build a safe aggregate entry from a JSON-like mapping."""

        return _entry_from_mapping(
            payload,
            default_release_id=release_id,
            default_composition=composition,
            default_policy_fingerprint=policy_fingerprint,
            policy_hint=policy,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical aggregate-only entry representation."""

        payload: dict[str, Any] = {
            "release_id": self.release_id,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "composition": self.composition,
            "policy_fingerprint": self.policy_fingerprint,
        }
        if self.sequence is not None:
            payload["sequence"] = self.sequence
        return payload


BudgetLedgerEntry = BudgetEntry
BudgetMigrationEntry = BudgetEntry


@dataclass(frozen=True, slots=True)
class BudgetLedgerSnapshot:
    """Immutable collection of aggregate budget entries for one migration."""

    entries: tuple[BudgetEntry, ...]
    schema_version: str = _DEFAULT_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise BudgetMigrationError("budget snapshot must contain entries")
        if any(not isinstance(entry, BudgetEntry) for entry in entries):
            raise BudgetMigrationError("budget snapshot entries are incompatible")
        release_ids = [entry.release_id for entry in entries]
        if len(release_ids) != len(set(release_ids)):
            raise BudgetMigrationError("budget snapshot contains duplicate entries")
        sequences = [entry.sequence for entry in entries if entry.sequence is not None]
        if len(sequences) != len(set(sequences)):
            raise BudgetMigrationError("budget snapshot reuses a release sequence")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(entries, key=lambda entry: entry.release_id)),
        )
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(self.schema_version),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BudgetLedgerSnapshot":
        """Build a snapshot from a canonical or accountant-style mapping."""

        return _coerce_snapshot(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate-only snapshot data."""

        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }


BudgetSnapshot = BudgetLedgerSnapshot


@dataclass(frozen=True, slots=True)
class BudgetMigrationIssue:
    """One safe, classified violation found during migration comparison."""

    kind: str
    field: str
    release_id: str | None = None
    before: int | float | str | None = None
    after: int | float | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _safe_identifier(self.kind, field_name="issue kind"),
        )
        object.__setattr__(
            self,
            "field",
            _safe_identifier(self.field, field_name="issue field"),
        )
        if self.release_id is not None:
            object.__setattr__(
                self,
                "release_id",
                _safe_identifier(self.release_id, field_name="release identifier"),
            )
        for name in ("before", "after"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _report_value(value))

    @property
    def reason(self) -> str:
        """Return the stable issue classification."""

        return self.kind

    def to_dict(self) -> dict[str, Any]:
        """Return a raw-value-free issue record."""

        return {
            "kind": self.kind,
            "field": self.field,
            "release_id": self.release_id,
            "before": self.before,
            "after": self.after,
        }


@dataclass(frozen=True, slots=True)
class BudgetMigrationReport:
    """Counts-only result of comparing two budget snapshots."""

    before_digest: str
    after_digest: str
    before_totals: Mapping[str, int | float]
    after_totals: Mapping[str, int | float]
    counts: Mapping[str, int]
    issues: tuple[BudgetMigrationIssue, ...] = ()
    release_identifiers: tuple[str, ...] = ()
    composition_methods: tuple[str, ...] = ()
    policy_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "before_digest", _digest(self.before_digest))
        object.__setattr__(self, "after_digest", _digest(self.after_digest))
        object.__setattr__(self, "before_totals", _totals(self.before_totals))
        object.__setattr__(self, "after_totals", _totals(self.after_totals))
        object.__setattr__(
            self,
            "counts",
            {
                _safe_identifier(key, field_name="count name"): _non_negative_int(
                    value,
                    field_name="count",
                )
                for key, value in sorted(self.counts.items())
            },
        )
        object.__setattr__(
            self,
            "issues",
            tuple(sorted(self.issues, key=_issue_sort_key)),
        )
        object.__setattr__(
            self, "release_identifiers", _safe_identifiers(self.release_identifiers)
        )
        object.__setattr__(
            self, "composition_methods", _safe_identifiers(self.composition_methods)
        )
        object.__setattr__(
            self, "policy_fingerprints", _safe_identifiers(self.policy_fingerprints)
        )

    @property
    def passed(self) -> bool:
        """Return whether the migration preserves the budget safety contract."""

        return not self.issues

    @property
    def valid(self) -> bool:
        """Alias for :attr:`passed`."""

        return self.passed

    @property
    def allowed(self) -> bool:
        """Alias for :attr:`passed`."""

        return self.passed

    @property
    def migration_safe(self) -> bool:
        """Return whether the after snapshot is safe to adopt."""

        return self.passed

    @property
    def release_blocked(self) -> bool:
        """Return whether migration must be blocked."""

        return not self.passed

    @property
    def is_monotonic(self) -> bool:
        """Return whether all migration checks passed."""

        return self.passed

    @property
    def changed_entries(self) -> int:
        """Return the count of existing releases whose state changed."""

        return self.counts.get("changed", 0)

    @property
    def summary(self) -> dict[str, int]:
        """Return deterministic aggregate migration counts."""

        return dict(self.counts)

    @property
    def violations(self) -> tuple[BudgetMigrationIssue, ...]:
        """Return the classified issues that block migration."""

        return self.issues

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic report data without source values."""

        return {
            "schema_version": BUDGET_MIGRATION_SCHEMA_VERSION,
            "passed": self.passed,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "before_totals": dict(self.before_totals),
            "after_totals": dict(self.after_totals),
            "counts": dict(self.counts),
            "summary": dict(self.counts),
            "release_identifiers": list(self.release_identifiers),
            "composition_methods": list(self.composition_methods),
            "policy_fingerprints": list(self.policy_fingerprints),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return canonical JSON for the aggregate report."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Return a compact review summary containing no source payloads."""

        status = "passed" if self.passed else "blocked"
        lines = [
            "# Differential-privacy budget migration",
            "",
            f"Status: **{status}**",
            "",
            "| Metric | Before | After |",
            "|---|---:|---:|",
            f"| Entries | {self.before_totals['entry_count']} | {self.after_totals['entry_count']} |",
            f"| Spent epsilon | {self.before_totals['spent_epsilon']:.12g} | {self.after_totals['spent_epsilon']:.12g} |",
            f"| Spent delta | {self.before_totals['spent_delta']:.12g} | {self.after_totals['spent_delta']:.12g} |",
            f"| Issues | {len(self.issues)} | {len(self.issues)} |",
        ]
        return "\n".join(lines)


class BudgetMigrationVerifier:
    """Stateless facade for applications that prefer an object API."""

    @staticmethod
    def compare(
        before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
        after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    ) -> BudgetMigrationReport:
        """Compare two snapshots without mutating either input."""

        return compare_budget_migration(before, after)

    @staticmethod
    def verify(
        before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
        after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    ) -> BudgetMigrationReport:
        """Alias for :meth:`compare`."""

        return compare_budget_migration(before, after)

    @staticmethod
    def enforce(
        before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
        after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    ) -> BudgetMigrationReport:
        """Compare and raise when the migration is blocked."""

        return enforce_budget_migration(before, after)


def compare_budget_migration(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Compare two budget snapshots and return a deterministic safety report.

    Existing release identifiers must remain present.  For each retained
    release, cumulative spend may only increase, limits may only decrease, and
    the composition method and policy fingerprint must remain unchanged. New
    identifiers are allowed when their entries are valid and their optional
    sequence numbers do not reuse an older sequence.
    """

    before_snapshot = _coerce_snapshot(before)
    after_snapshot = _coerce_snapshot(after)
    before_entries = {entry.release_id: entry for entry in before_snapshot.entries}
    after_entries = {entry.release_id: entry for entry in after_snapshot.entries}
    issues: list[BudgetMigrationIssue] = []

    for release_id in sorted(before_entries):
        previous = before_entries[release_id]
        current = after_entries.get(release_id)
        if current is None:
            issues.append(
                BudgetMigrationIssue(
                    "missing_release",
                    "release_id",
                    release_id=release_id,
                    before=release_id,
                )
            )
            continue
        issues.extend(_compare_entry(previous, current))

    old_sequences = {
        entry.sequence
        for entry in before_snapshot.entries
        if entry.sequence is not None
    }
    max_old_sequence = max(old_sequences, default=0)
    for release_id, entry in sorted(after_entries.items()):
        if release_id in before_entries:
            continue
        if entry.sequence is not None and entry.sequence <= max_old_sequence:
            issues.append(
                BudgetMigrationIssue(
                    "reused_release_sequence",
                    "sequence",
                    release_id=release_id,
                    after=entry.sequence,
                )
            )

    changed_ids = {
        release_id
        for release_id in set(before_entries).intersection(after_entries)
        if before_entries[release_id] != after_entries[release_id]
    }
    counts = {
        "before_entries": len(before_entries),
        "after_entries": len(after_entries),
        "added": len(set(after_entries) - set(before_entries)),
        "removed": len(set(before_entries) - set(after_entries)),
        "unchanged": len(before_entries)
        - len(changed_ids)
        - len(set(before_entries) - set(after_entries)),
        "changed": len(changed_ids),
        "issues": len(issues),
    }
    return BudgetMigrationReport(
        before_digest=_snapshot_digest(before_snapshot),
        after_digest=_snapshot_digest(after_snapshot),
        before_totals=_aggregate_totals(before_snapshot.entries),
        after_totals=_aggregate_totals(after_snapshot.entries),
        counts=counts,
        issues=tuple(issues),
        release_identifiers=tuple(entry.release_id for entry in after_snapshot.entries),
        composition_methods=tuple(
            sorted({entry.composition for entry in after_snapshot.entries})
        ),
        policy_fingerprints=tuple(
            sorted({entry.policy_fingerprint for entry in after_snapshot.entries})
        ),
    )


def verify_budget_migration(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Return the migration report, raising only for malformed snapshots."""

    return compare_budget_migration(before, after)


def check_budget_migration(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Alias for :func:`compare_budget_migration`."""

    return compare_budget_migration(before, after)


def validate_budget_migration(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Alias for :func:`compare_budget_migration`."""

    return compare_budget_migration(before, after)


def compare_budget_ledgers(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Compatibility alias for :func:`compare_budget_migration`."""

    return compare_budget_migration(before, after)


def verify_budget_ledger_migration(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Compatibility alias for :func:`verify_budget_migration`."""

    return compare_budget_migration(before, after)


def enforce_budget_migration(
    before: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
    after: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetMigrationReport:
    """Compare snapshots and raise if any safety check is blocked."""

    report = compare_budget_migration(before, after)
    if not report.passed:
        raise BudgetMigrationRejected(report)
    return report


def _compare_entry(
    before: BudgetEntry,
    after: BudgetEntry,
) -> list[BudgetMigrationIssue]:
    issues: list[BudgetMigrationIssue] = []
    if _decreased(after.spent_epsilon, before.spent_epsilon):
        issues.append(
            BudgetMigrationIssue(
                "spent_budget_decreased",
                "spent_epsilon",
                release_id=before.release_id,
                before=before.spent_epsilon,
                after=after.spent_epsilon,
            )
        )
    if _decreased(after.spent_delta, before.spent_delta):
        issues.append(
            BudgetMigrationIssue(
                "spent_budget_decreased",
                "spent_delta",
                release_id=before.release_id,
                before=before.spent_delta,
                after=after.spent_delta,
            )
        )
    if _increased(after.max_epsilon, before.max_epsilon):
        issues.append(
            BudgetMigrationIssue(
                "budget_limit_increased",
                "max_epsilon",
                release_id=before.release_id,
                before=before.max_epsilon,
                after=after.max_epsilon,
            )
        )
    if _increased(after.max_delta, before.max_delta):
        issues.append(
            BudgetMigrationIssue(
                "budget_limit_increased",
                "max_delta",
                release_id=before.release_id,
                before=before.max_delta,
                after=after.max_delta,
            )
        )
    if after.composition != before.composition:
        issues.append(
            BudgetMigrationIssue(
                "composition_changed",
                "composition",
                release_id=before.release_id,
                before=before.composition,
                after=after.composition,
            )
        )
    if after.policy_fingerprint != before.policy_fingerprint:
        issues.append(
            BudgetMigrationIssue(
                "policy_fingerprint_changed",
                "policy_fingerprint",
                release_id=before.release_id,
                before=before.policy_fingerprint,
                after=after.policy_fingerprint,
            )
        )
    if before.sequence != after.sequence:
        issues.append(
            BudgetMigrationIssue(
                "release_sequence_changed",
                "sequence",
                release_id=before.release_id,
                before=before.sequence,
                after=after.sequence,
            )
        )
    return issues


def _coerce_snapshot(
    value: BudgetLedgerSnapshot | Mapping[str, Any] | Sequence[Any],
) -> BudgetLedgerSnapshot:
    if isinstance(value, BudgetLedgerSnapshot):
        return value
    if isinstance(value, Mapping):
        return _snapshot_from_mapping(value)
    if _is_sequence(value):
        entries = _coerce_entries(value)
        return BudgetLedgerSnapshot(entries=entries)
    raise BudgetMigrationError("budget snapshot must be a mapping or sequence")


def _snapshot_from_mapping(payload: Mapping[str, Any]) -> BudgetLedgerSnapshot:
    schema_version = payload.get(
        "schema_version",
        payload.get("version", _DEFAULT_SNAPSHOT_SCHEMA_VERSION),
    )
    defaults = {
        "release_id": _optional_identifier(payload, _RELEASE_KEYS),
        "composition": _optional_identifier(payload, _COMPOSITION_KEYS),
        "policy_fingerprint": _optional_identifier(payload, _FINGERPRINT_KEYS),
    }
    policy_lookup = payload.get("policies")
    if policy_lookup is not None and not isinstance(policy_lookup, Mapping):
        raise BudgetMigrationError("budget snapshot policies must be a mapping")

    if "compositions" in payload:
        entries = _coerce_accountant_compositions(
            payload["compositions"],
            policy_lookup,
            defaults=defaults,
        )
    else:
        container_key = next(
            (
                key
                for key in ("entries", "ledger", "releases", "budgets", "contexts")
                if key in payload
            ),
            None,
        )
        if container_key is not None:
            entries = _coerce_entries(
                payload[container_key],
                defaults=defaults,
                policy_lookup=policy_lookup,
            )
        elif _looks_like_entry(payload):
            entries = (
                _entry_from_mapping(
                    payload,
                    default_release_id=defaults["release_id"],
                    default_composition=defaults["composition"],
                    default_policy_fingerprint=defaults["policy_fingerprint"],
                    policy_hint=_policy_for(
                        policy_lookup,
                        defaults["release_id"],
                    ),
                ),
            )
        else:
            raise BudgetMigrationError("budget snapshot is missing entries")
    return BudgetLedgerSnapshot(entries=entries, schema_version=schema_version)


def _coerce_accountant_compositions(
    raw: Any,
    policies: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, str | None],
) -> tuple[BudgetEntry, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise BudgetMigrationError("budget snapshot compositions must contain entries")
    entries: list[BudgetEntry] = []
    for scope, composition in raw.items():
        release_id = _safe_identifier(scope, field_name="release identifier")
        if not isinstance(composition, Mapping):
            raise BudgetMigrationError("budget composition entry is incompatible")
        policy = _policy_for(policies, release_id)
        if policy is None:
            raise BudgetMigrationError("budget composition entry is missing its policy")
        entry_payload = dict(composition)
        entry_payload.setdefault("release_id", release_id)
        if defaults["composition"] is not None:
            entry_payload.setdefault("composition", defaults["composition"])
        if defaults["policy_fingerprint"] is not None:
            entry_payload.setdefault(
                "policy_fingerprint",
                defaults["policy_fingerprint"],
            )
        entries.append(
            _entry_from_mapping(
                entry_payload,
                default_release_id=release_id,
                default_composition=defaults["composition"],
                default_policy_fingerprint=defaults["policy_fingerprint"],
                policy_hint=policy,
            )
        )
    return tuple(entries)


def _coerce_entries(
    raw: Any,
    *,
    defaults: Mapping[str, str | None] | None = None,
    policy_lookup: Mapping[str, Any] | None = None,
) -> tuple[BudgetEntry, ...]:
    defaults = defaults or {}
    entries: list[BudgetEntry] = []
    if isinstance(raw, Mapping):
        if _looks_like_entry(raw):
            items: Sequence[tuple[str | None, Any]] = ((None, raw),)
        else:
            items = tuple((str(key), value) for key, value in raw.items())
    elif _is_sequence(raw):
        items = tuple((None, value) for value in raw)
    else:
        raise BudgetMigrationError(
            "budget snapshot entries must be a mapping or sequence"
        )
    if not items:
        raise BudgetMigrationError("budget snapshot must contain entries")
    for key, value in items:
        release_id = key or defaults.get("release_id")
        policy = _policy_for(policy_lookup, release_id)
        if isinstance(value, BudgetEntry):
            entry = value
        else:
            entry = _entry_from_mapping(
                value,
                default_release_id=release_id,
                default_composition=defaults.get("composition"),
                default_policy_fingerprint=defaults.get("policy_fingerprint"),
                policy_hint=policy,
            )
        entries.append(entry)
    return tuple(entries)


def _entry_from_mapping(
    payload: Mapping[str, Any],
    *,
    default_release_id: str | None = None,
    default_composition: str | None = None,
    default_policy_fingerprint: str | None = None,
    policy_hint: Mapping[str, Any] | str | None = None,
) -> BudgetEntry:
    if not isinstance(payload, Mapping):
        raise BudgetMigrationError("budget entry must be a mapping")
    nested_policy = _nested_mapping(payload, ("policy", "policy_config"))
    if nested_policy is None and isinstance(policy_hint, Mapping):
        nested_policy = policy_hint
    hinted_fingerprint = (
        policy_hint if isinstance(policy_hint, str) else None
    ) or _optional_identifier(nested_policy or {}, _FINGERPRINT_KEYS)
    if hinted_fingerprint is None and nested_policy is not None:
        hinted_fingerprint = _policy_fingerprint(nested_policy)

    release_id = _required_identifier(
        _first_present(payload, _RELEASE_KEYS),
        default_release_id,
        field_name="release identifier",
    )
    composition = _required_identifier(
        _first_present(payload, _COMPOSITION_KEYS),
        _first_present(nested_policy or {}, _COMPOSITION_KEYS),
        default_composition,
        field_name="composition method",
    )
    policy_fingerprint = _required_identifier(
        _first_present(payload, _FINGERPRINT_KEYS),
        hinted_fingerprint,
        default_policy_fingerprint,
        field_name="policy fingerprint",
    )

    spent = _nested_mapping(
        payload,
        ("spent", "spend", "spent_budget", "composition_totals"),
    )
    budget = _nested_mapping(payload, ("budget", "limits", "limit"))
    spent_epsilon = _required_value(
        _first_present(payload, _SPENT_EPSILON_KEYS),
        _first_present(spent or {}, _SPENT_EPSILON_KEYS),
        _first_present(spent or {}, ("epsilon",)),
        _first_present(payload, ("epsilon",)),
        field_name="spent epsilon",
    )
    spent_delta = _required_value(
        _first_present(payload, _SPENT_DELTA_KEYS),
        _first_present(spent or {}, _SPENT_DELTA_KEYS),
        _first_present(spent or {}, ("delta",)),
        _first_present(payload, ("delta",)),
        field_name="spent delta",
    )
    max_epsilon = _required_value(
        _first_present(payload, _LIMIT_EPSILON_KEYS),
        _first_present(budget or {}, _LIMIT_EPSILON_KEYS),
        _first_present(budget or {}, ("epsilon",)),
        _first_present(nested_policy or {}, _LIMIT_EPSILON_KEYS),
        field_name="epsilon limit",
    )
    max_delta = _required_value(
        _first_present(payload, _LIMIT_DELTA_KEYS),
        _first_present(budget or {}, _LIMIT_DELTA_KEYS),
        _first_present(budget or {}, ("delta",)),
        _first_present(nested_policy or {}, _LIMIT_DELTA_KEYS),
        field_name="delta limit",
    )
    sequence = _first_present(payload, ("sequence", "release_sequence"))
    if sequence is _MISSING:
        sequence = None
    return BudgetEntry(
        release_id=release_id,
        spent_epsilon=spent_epsilon,
        spent_delta=spent_delta,
        max_epsilon=max_epsilon,
        max_delta=max_delta,
        composition=composition,
        policy_fingerprint=policy_fingerprint,
        sequence=sequence,
    )


def _looks_like_entry(payload: Mapping[str, Any]) -> bool:
    keys = set(payload)
    return bool(
        keys.intersection(
            {
                *_RELEASE_KEYS,
                *_SPENT_EPSILON_KEYS,
                *_SPENT_DELTA_KEYS,
                "epsilon",
                "delta",
                "spent",
                "budget",
                "limits",
                "policy",
                "policy_fingerprint",
            }
        )
    )


def _policy_for(
    policies: Mapping[str, Any] | None,
    release_id: str | None,
) -> Mapping[str, Any] | str | None:
    if policies is None or release_id is None:
        return None
    for key, value in policies.items():
        if str(key) == release_id:
            if isinstance(value, (Mapping, str)):
                return value
            raise BudgetMigrationError("budget policy entry is incompatible")
    return None


def _nested_mapping(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Mapping[str, Any] | None:
    value = _first_present(payload, keys)
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, Mapping):
        raise BudgetMigrationError("budget entry contains an incompatible nested value")
    return value


def _first_present(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    found = [payload[key] for key in keys if key in payload]
    if not found:
        return _MISSING
    first = found[0]
    if any(value != first for value in found[1:]):
        raise BudgetMigrationError("budget entry contains conflicting fields")
    return first


def _optional_identifier(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    value = _first_present(payload, keys)
    if value is _MISSING or value is None:
        return None
    return _safe_identifier(value, field_name="stable identifier")


def _required_identifier(*values: Any, field_name: str) -> str:
    for value in values:
        if value is not _MISSING and value is not None:
            return _safe_identifier(value, field_name=field_name)
    raise BudgetMigrationError(f"budget entry is missing {field_name}")


def _required_value(*values: Any, field_name: str) -> Any:
    for value in values:
        if value is not _MISSING and value is not None:
            return value
    raise BudgetMigrationError(f"budget entry is missing {field_name}")


def _safe_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise BudgetMigrationError(f"{field_name} must be a stable identifier")
    candidate = value.strip()
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise BudgetMigrationError(f"{field_name} must be a stable identifier")
    if any(pattern.search(candidate) for pattern in _PHI_PATTERNS):
        raise BudgetMigrationError(f"{field_name} must not contain sensitive data")
    return candidate


def _safe_identifiers(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _safe_identifier(value, field_name="stable identifier")
                for value in values
            }
        )
    )


def _schema_version(value: Any) -> str:
    if type(value) is int:
        if value < 1:
            raise BudgetMigrationError("budget snapshot schema version is invalid")
        return str(value)
    return _safe_identifier(value, field_name="schema version")


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise BudgetMigrationError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise BudgetMigrationError(f"{field_name} must be a positive integer") from None
    if parsed <= 0:
        raise BudgetMigrationError(f"{field_name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise BudgetMigrationError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise BudgetMigrationError(
            f"{field_name} must be a non-negative integer"
        ) from None
    if parsed < 0:
        raise BudgetMigrationError(f"{field_name} must be a non-negative integer")
    return parsed


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise BudgetMigrationError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise BudgetMigrationError(f"{field_name} must be a finite number") from None
    if not math.isfinite(parsed):
        raise BudgetMigrationError(f"{field_name} must be a finite number")
    return 0.0 if parsed == 0.0 else parsed


def _non_negative_float(value: Any, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name)
    if parsed < 0.0:
        raise BudgetMigrationError(f"{field_name} must be non-negative")
    return parsed


def _delta_float(value: Any, *, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name=field_name)
    if parsed >= 1.0:
        raise BudgetMigrationError(f"{field_name} must be less than one")
    return parsed


def _report_value(value: Any) -> int | float | str:
    if isinstance(value, bool):
        return value
    if type(value) is int:
        return _non_negative_int(value, field_name="report value")
    if isinstance(value, float):
        return _finite_float(value, field_name="report value")
    return _safe_identifier(value, field_name="report identifier")


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BudgetMigrationError("budget migration digest is invalid")
    return value.lower() if value.startswith("sha256:") else f"sha256:{value.lower()}"


def _snapshot_digest(snapshot: BudgetLedgerSnapshot) -> str:
    encoded = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _policy_fingerprint(policy: Mapping[str, Any]) -> str:
    selected: dict[str, Any] = {}
    for field_name in _POLICY_FIELDS:
        if field_name not in policy:
            continue
        value = policy[field_name]
        if field_name in {"max_epsilon", "max_delta", "delta_prime"}:
            selected[field_name] = _finite_float(value, field_name=field_name)
        elif field_name in {"scope", "name", "version", "schema_version"}:
            selected[field_name] = _safe_identifier(
                value, field_name="policy identifier"
            )
        elif field_name in {"composition", "composition_method", "composition_rule"}:
            selected["composition"] = _safe_identifier(
                value,
                field_name="composition method",
            )
    if not selected:
        raise BudgetMigrationError("budget entry is missing policy fingerprint")
    encoded = json.dumps(
        selected,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _aggregate_totals(entries: Sequence[BudgetEntry]) -> dict[str, int | float]:
    return {
        "entry_count": len(entries),
        "spent_epsilon": math.fsum(entry.spent_epsilon for entry in entries),
        "spent_delta": math.fsum(entry.spent_delta for entry in entries),
        "max_epsilon": math.fsum(entry.max_epsilon for entry in entries),
        "max_delta": math.fsum(entry.max_delta for entry in entries),
    }


def _totals(value: Mapping[str, int | float]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if key == "entry_count":
            result[key] = _non_negative_int(item, field_name="entry count")
        else:
            result[key] = _non_negative_float(item, field_name="aggregate budget")
    return result


def _issue_sort_key(issue: BudgetMigrationIssue) -> tuple[str, str, str]:
    return (issue.release_id or "", issue.field, issue.kind)


def _decreased(after: float, before: float) -> bool:
    return after < before and not math.isclose(
        after,
        before,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOLERANCE,
    )


def _increased(after: float, before: float) -> bool:
    return _decreased(before, after)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )
