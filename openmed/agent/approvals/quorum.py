"""Local quorum policy evaluation over protected, consumed authorizations.

This module does not authenticate receipts or authorize dispatch. Applications
must pass live authorizations from their trusted token verifier. Audit receipt
JSON carries no authority. Durable evidence is private host-protected metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .tokens import (
    ApprovalAuthorization,
    ApprovalReceipt,
    ApprovalTokenValidationError,
    _validate_digest,
    _validate_reviewer_role,
    _validate_timestamp,
)

_ACTION_CLASS = re.compile(r"[a-z][a-z0-9-]{0,63}")
_MAX_RECEIPTS = 10_000


class ApprovalQuorumError(ValueError):
    """Fail closed with a fixed code and no input values."""


def _roles(roles: tuple[str, ...]) -> None:
    if type(roles) is not tuple:
        raise ApprovalQuorumError("invalid_roles")
    for role in roles:
        _validate_reviewer_role(role)
    if len(set(roles)) != len(roles):
        raise ApprovalQuorumError("invalid_roles")


@dataclass(frozen=True, slots=True)
class ApprovalQuorumPolicy:
    """Policy for a developer-designated action class.

    Args:
        action_class: Controlled lower-case action-class label, never source text.
        required_count: Positive number of eligible receipts required.
        allowed_reviewer_roles: Canonical policy roles eligible to approve.
        distinct_roles: Count at most one receipt per role when true.
        excluded_requester_roles: Additional requester-category roles prohibited
            from approving. The actual requesting role is always excluded.
    """

    action_class: str
    required_count: int
    allowed_reviewer_roles: tuple[str, ...]
    distinct_roles: bool = True
    excluded_requester_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.action_class) is not str
            or _ACTION_CLASS.fullmatch(self.action_class) is None
        ):
            raise ApprovalQuorumError("invalid_action_class")
        _roles(self.allowed_reviewer_roles)
        _roles(self.excluded_requester_roles)
        if type(self.distinct_roles) is not bool:
            raise ApprovalQuorumError("invalid_distinct_roles")
        if (
            type(self.required_count) is not int
            or not 1 <= self.required_count <= _MAX_RECEIPTS
        ):
            raise ApprovalQuorumError("invalid_required_count")
        available = set(self.allowed_reviewer_roles) - set(
            self.excluded_requester_roles
        )
        if not available or (
            self.distinct_roles and len(available) < self.required_count
        ):
            raise ApprovalQuorumError("impossible_quorum")

    @property
    def digest(self) -> str:
        """Return a domain-separated commitment to the complete policy."""
        payload = json.dumps(
            [
                self.action_class,
                self.required_count,
                sorted(self.allowed_reviewer_roles),
                self.distinct_roles,
                sorted(self.excluded_requester_roles),
            ],
            separators=(",", ":"),
        )
        return (
            "sha256:"
            + hashlib.sha256(
                b"openmed.agent.approval_quorum.policy.v1\0" + payload.encode("ascii")
            ).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class ApprovalQuorumDecision:
    """Value-free advisory decision containing only roles, counts and digests.

    A satisfied decision is not single-use dispatch authority. Protected authorizations carry
    policy roles rather than human identities, so person independence remains
    the authenticating application's responsibility.
    """

    action_digest: str
    policy_digest: str
    requester_role: str
    reviewer_roles: tuple[str, ...]
    receipt_digests: tuple[str, ...]
    required_count: int
    approved_count: int

    def __post_init__(self) -> None:
        _validate_digest(self.action_digest, "action_digest")
        _validate_digest(self.policy_digest, "policy_digest")
        _validate_reviewer_role(self.requester_role)
        _roles(self.reviewer_roles)
        if type(self.receipt_digests) is not tuple:
            raise ApprovalQuorumError("invalid_receipt_digests")
        for digest in self.receipt_digests:
            _validate_digest(digest, "token_digest")
        if (
            type(self.required_count) is not int
            or not 1 <= self.required_count <= _MAX_RECEIPTS
            or type(self.approved_count) is not int
            or not 0 <= self.approved_count <= _MAX_RECEIPTS
            or self.approved_count != len(self.receipt_digests)
            or len(set(self.receipt_digests)) != self.approved_count
            or len(self.reviewer_roles) > self.approved_count
        ):
            raise ApprovalQuorumError("invalid_counts")

    @property
    def satisfied(self) -> bool:
        """Return whether the eligible receipt count meets the policy."""
        return self.approved_count >= self.required_count

    def to_dict(self) -> dict[str, str | int | list[str]]:
        """Return only roles, counts and digests for dispatcher inspection."""
        return {
            "action_digest": self.action_digest,
            "policy_digest": self.policy_digest,
            "requester_role": self.requester_role,
            "reviewer_roles": list(self.reviewer_roles),
            "receipt_digests": list(self.receipt_digests),
            "required_count": self.required_count,
            "approved_count": self.approved_count,
        }


@dataclass(frozen=True, slots=True, repr=False)
class _QuorumEvidence:
    """Protected-store metadata, never a reconstructed authorization object."""

    receipt: ApprovalReceipt
    reviewer_role: str
    consumed_at: int
    expires_at: int

    def __post_init__(self) -> None:
        if type(self.receipt) is not ApprovalReceipt:
            raise ApprovalQuorumError("invalid_receipt")
        ApprovalReceipt.from_dict(self.receipt.to_dict())
        _validate_reviewer_role(self.reviewer_role)
        _validate_timestamp(self.consumed_at, "consumed_at")
        _validate_timestamp(self.expires_at, "expires_at")
        if self.consumed_at >= self.expires_at:
            raise ApprovalQuorumError("invalid_receipt")

    @property
    def action_digest(self) -> str:
        return self.receipt.action_digest

    @property
    def token_digest(self) -> str:
        return self.receipt.token_digest

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": "openmed.agent.quorum-evidence.v1",
                "receipt": self.receipt.to_dict(),
                "reviewer_role": self.reviewer_role,
                "consumed_at": self.consumed_at,
                "expires_at": self.expires_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> _QuorumEvidence:
        row = json.loads(payload)
        if (
            type(row) is not dict
            or set(row)
            != {
                "schema_version",
                "receipt",
                "reviewer_role",
                "consumed_at",
                "expires_at",
            }
            or row["schema_version"] != "openmed.agent.quorum-evidence.v1"
        ):
            raise ApprovalQuorumError("invalid_receipt")
        return cls(
            ApprovalReceipt.from_dict(row["receipt"]),
            row["reviewer_role"],
            row["consumed_at"],
            row["expires_at"],
        )


def _authorizations(
    receipts: Iterable[ApprovalAuthorization],
) -> tuple[_QuorumEvidence, ...]:
    normalized: list[_QuorumEvidence] = []
    failure = None
    try:
        for authorization in receipts:
            if len(normalized) >= _MAX_RECEIPTS:
                failure = "too_many_receipts"
                break
            if type(authorization) is not ApprovalAuthorization:
                failure = "invalid_receipt"
                break
            normalized.append(
                _QuorumEvidence(
                    authorization.receipt,
                    authorization.reviewer_role,
                    authorization.consumed_at,
                    authorization.expires_at,
                )
            )
    except Exception:
        failure = "invalid_receipt"
    if failure is not None:
        # Raise outside the handler so rejected values are absent from context.
        raise ApprovalQuorumError(failure)
    return tuple(normalized)


class ApprovalQuorumEvaluator:
    """Evaluate live authorization sets using policies keyed by action class.

    Args:
        policies: Unique policies; missing classes fail closed.
    """

    def __init__(self, policies: Iterable[ApprovalQuorumPolicy]) -> None:
        self._policies: dict[str, ApprovalQuorumPolicy] = {}
        for policy in policies:
            if type(policy) is not ApprovalQuorumPolicy:
                raise ApprovalQuorumError("invalid_policy")
            if policy.action_class in self._policies:
                raise ApprovalQuorumError("duplicate_action_class")
            self._policies[policy.action_class] = policy

    def policy(self, action_class: str) -> ApprovalQuorumPolicy:
        """Return the designated policy, refusing unknown classes."""
        if type(action_class) is not str or action_class not in self._policies:
            raise ApprovalQuorumError("unknown_action_class")
        return self._policies[action_class]

    def evaluate(
        self,
        *,
        action_class: str,
        action_digest: str,
        requester_role: str,
        receipts: Iterable[ApprovalAuthorization],
        now: int,
        replayed_receipt_digests: tuple[str, ...] = (),
    ) -> ApprovalQuorumDecision:
        """Count unexpired, exact-action receipts under the designated policy.

        Args:
            action_class: Registered controlled class label.
            action_digest: Commitment to the exact reviewed action.
            requester_role: Actual requesting policy role, always excluded.
            receipts: Live protected authorizations from successful token consumption.
            now: Injected Unix timestamp; expiry is exclusive.
            replayed_receipt_digests: Token digests already used elsewhere.

        Returns:
            A deterministic metadata-only advisory quorum decision.
        """
        return self._evaluate_records(
            action_class=action_class,
            action_digest=action_digest,
            requester_role=requester_role,
            receipts=_authorizations(receipts),
            now=now,
            replayed_receipt_digests=replayed_receipt_digests,
        )

    def _evaluate_records(
        self,
        *,
        action_class: str,
        action_digest: str,
        requester_role: str,
        receipts: tuple[_QuorumEvidence, ...],
        now: int,
        replayed_receipt_digests: tuple[str, ...] = (),
    ) -> ApprovalQuorumDecision:
        """Evaluate private records from live verification or the trusted store."""
        policy = self.policy(action_class)
        _validate_digest(action_digest, "action_digest")
        _validate_reviewer_role(requester_role)
        _validate_timestamp(now, "now")
        for digest in replayed_receipt_digests:
            _validate_digest(digest, "token_digest")
        counts = Counter(receipt.token_digest for receipt in receipts)
        # Duplicate token commitments are replay evidence: exclude all copies,
        # including conflicting metadata, rather than choosing a preferred one.
        accepted = []
        roles: set[str] = set()
        for receipt in sorted(
            receipts, key=lambda r: (r.reviewer_role, r.token_digest)
        ):
            if (
                receipt.action_digest != action_digest
                or not receipt.consumed_at <= now < receipt.expires_at
                or receipt.reviewer_role == requester_role
                or receipt.reviewer_role in policy.excluded_requester_roles
                or receipt.reviewer_role not in policy.allowed_reviewer_roles
                or counts[receipt.token_digest] != 1
                or receipt.token_digest in replayed_receipt_digests
                or (policy.distinct_roles and receipt.reviewer_role in roles)
            ):
                continue
            accepted.append(receipt.token_digest)
            roles.add(receipt.reviewer_role)
        return ApprovalQuorumDecision(
            action_digest=action_digest,
            policy_digest=policy.digest,
            requester_role=requester_role,
            reviewer_roles=tuple(sorted(roles)),
            receipt_digests=tuple(sorted(accepted)),
            required_count=policy.required_count,
            approved_count=len(accepted),
        )


class SQLiteApprovalQuorumStore:
    """Persist partial approvals locally with atomic replay and reset handling.

    Args:
        path: Application-owned dedicated local database, protected by the host.

    Only validated receipt metadata and digests are stored. Connections are
    short-lived; independent instances/processes serialize through SQLite.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = path
        try:
            with closing(sqlite3.connect(path)) as db, db:
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS quorum_progress (
                        progress_digest TEXT PRIMARY KEY,
                        action_digest TEXT NOT NULL,
                        policy_digest TEXT NOT NULL,
                        requester_role TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS quorum_receipts (
                        token_digest TEXT PRIMARY KEY,
                        progress_digest TEXT NOT NULL,
                        receipt TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS quorum_seen (
                        token_digest TEXT PRIMARY KEY
                    );
                    CREATE TABLE IF NOT EXISTS quorum_clock (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        last_now INTEGER NOT NULL
                    );
                    """
                )
        except Exception:
            failed = True
        else:
            failed = False
        if failed:
            raise ApprovalQuorumError("store_unavailable")

    def collect(
        self,
        *,
        progress_digest: str,
        evaluator: ApprovalQuorumEvaluator,
        action_class: str,
        action_digest: str,
        requester_role: str,
        receipts: Iterable[ApprovalAuthorization],
        now: int,
    ) -> ApprovalQuorumDecision:
        """Atomically collect new receipts and re-evaluate stored progress.

        Args:
            progress_digest: Stable opaque commitment to one logical action slot.
                Reuse this slot when editing the action; use separate slots for
                independent actions. Never derive it from a lone PHI identifier.
            evaluator: Trusted configured policy evaluator.
            action_class: Registered action class.
            action_digest: Current reviewed action commitment.
            requester_role: Actual requesting policy role.
            receipts: New live authorizations; repeated submissions add nothing.
            now: Injected current Unix time, monotonic across this database.

        Returns:
            Current decision after excluding expired and replayed submissions.

        Raises:
            ApprovalQuorumError: Storage failure or clock rollback, without paths.
        """
        _validate_digest(progress_digest, "progress_digest")
        incoming = _authorizations(receipts)
        initial = evaluator._evaluate_records(
            action_class=action_class,
            action_digest=action_digest,
            requester_role=requester_role,
            receipts=incoming,
            now=now,
        )
        binding = (action_digest, initial.policy_digest, requester_role)
        counts = Counter(receipt.token_digest for receipt in incoming)
        # Retain all individually eligible roles, including multiple receipts
        # for one role, so a newer receipt can outlive an older partial approval.
        eligible = {
            receipt.token_digest
            for receipt in incoming
            if counts[receipt.token_digest] == 1
            and evaluator._evaluate_records(
                action_class=action_class,
                action_digest=action_digest,
                requester_role=requester_role,
                receipts=(receipt,),
                now=now,
            ).approved_count
        }
        try:
            with closing(sqlite3.connect(self._path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                clock = db.execute("SELECT last_now FROM quorum_clock").fetchone()
                if clock is not None and now < clock[0]:
                    raise ApprovalQuorumError("clock_rollback")
                db.execute("INSERT OR REPLACE INTO quorum_clock VALUES (1, ?)", (now,))
                previous = db.execute(
                    "SELECT action_digest, policy_digest, requester_role "
                    "FROM quorum_progress WHERE progress_digest = ?",
                    (progress_digest,),
                ).fetchone()
                if previous != binding:
                    db.execute(
                        "DELETE FROM quorum_receipts WHERE progress_digest = ?",
                        (progress_digest,),
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO quorum_progress VALUES (?, ?, ?, ?)",
                        (progress_digest, *binding),
                    )
                # Keep replay tombstones even after action or policy changes.
                # Claim every submitted digest, including ineligible receipts.
                for receipt in incoming:
                    claimed = db.execute(
                        "INSERT OR IGNORE INTO quorum_seen VALUES (?)",
                        (receipt.token_digest,),
                    ).rowcount
                    if claimed and receipt.token_digest in eligible:
                        db.execute(
                            "INSERT INTO quorum_receipts VALUES (?, ?, ?)",
                            (receipt.token_digest, progress_digest, receipt.to_json()),
                        )
                stored_list: list[_QuorumEvidence] = []
                for row in db.execute(
                    "SELECT receipt FROM quorum_receipts WHERE progress_digest = ?",
                    (progress_digest,),
                ):
                    if len(stored_list) >= _MAX_RECEIPTS:
                        raise ApprovalQuorumError("too_many_receipts")
                    stored_list.append(_QuorumEvidence.from_json(row[0]))
                stored = tuple(stored_list)
                return evaluator._evaluate_records(
                    action_class=action_class,
                    action_digest=action_digest,
                    requester_role=requester_role,
                    receipts=stored,
                    now=now,
                )
        except Exception as exc:
            failure = (
                "clock_rollback"
                if type(exc) is ApprovalQuorumError and exc.args == ("clock_rollback",)
                else "store_unavailable"
            )
        raise ApprovalQuorumError(failure)

    def __repr__(self) -> str:
        """Omit the private local database path from diagnostics."""
        return "SQLiteApprovalQuorumStore(<local>)"
