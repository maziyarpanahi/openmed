"""Offline feedback-budget enforcement for adaptive clinical benchmarks.

The ledger keys policy and usage by digest-only epoch and submission identifiers.
It returns exact scores only while an epoch's detail allowance remains, then
returns deterministic coarse bands. Infrastructure failures can add one
replacement attempt only after a separately recorded operator approval.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping, Sequence

FEEDBACK_BUDGET_POLICY_SCHEMA_VERSION = "openmed.eval.feedback_budget_policy.v1"
FEEDBACK_BUDGET_DECISION_SCHEMA_VERSION = "openmed.eval.feedback_budget_decision.v1"
FEEDBACK_BUDGET_SNAPSHOT_SCHEMA_VERSION = "openmed.eval.feedback_budget_snapshot.v1"
FAILURE_APPROVAL_SCHEMA_VERSION = "openmed.eval.failure_approval.v1"

FEEDBACK_DETAILED = "detailed"
FEEDBACK_COARSE = "coarse"
FEEDBACK_NONE = "none"

REASON_ATTEMPT_RECORDED = "attempt_recorded"
REASON_RERUN_BUDGET_EXHAUSTED = "rerun_budget_exhausted"
REASON_FAILURE_APPROVED = "failure_approved"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


class FeedbackBudgetError(ValueError):
    """Raised when feedback-budget input cannot be handled safely."""


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _require_digest(field: str, value: Any) -> str:
    if not _is_sha256(value):
        raise FeedbackBudgetError(f"{field}: invalid_digest")
    return value


def _normalize_boundaries(boundaries: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise FeedbackBudgetError("coarse_score_boundaries: invalid_sequence")
    normalized = tuple(boundaries)
    if not normalized:
        raise FeedbackBudgetError("coarse_score_boundaries: empty")
    if any(
        type(boundary) not in (int, float)
        or not math.isfinite(boundary)
        or not 0.0 < boundary < 1.0
        for boundary in normalized
    ):
        raise FeedbackBudgetError("coarse_score_boundaries: invalid_boundary")
    as_floats = tuple(float(boundary) for boundary in normalized)
    if any(left >= right for left, right in zip(as_floats, as_floats[1:])):
        raise FeedbackBudgetError("coarse_score_boundaries: not_strictly_increasing")
    return as_floats


@dataclass(frozen=True, slots=True)
class FeedbackBudgetPolicy:
    """Immutable limits for one committed benchmark epoch.

    Args:
        epoch_digest: SHA-256 digest identifying the immutable benchmark epoch.
        detailed_feedback_limit: Maximum scored attempts that may expose an
            exact score for each submission digest.
        rerun_limit: Standard reruns allowed after the first official attempt.
        coarse_score_boundaries: Strictly increasing thresholds in ``(0, 1)``.
            Coarse feedback is the zero-based insertion band for a score.
        schema_version: Closed policy schema identifier.
    """

    epoch_digest: str
    detailed_feedback_limit: int
    rerun_limit: int
    coarse_score_boundaries: tuple[float, ...]
    schema_version: str = FEEDBACK_BUDGET_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEEDBACK_BUDGET_POLICY_SCHEMA_VERSION:
            raise FeedbackBudgetError("schema_version: unsupported_version")
        _require_digest("epoch_digest", self.epoch_digest)
        if (
            type(self.detailed_feedback_limit) is not int
            or self.detailed_feedback_limit < 0
        ):
            raise FeedbackBudgetError("detailed_feedback_limit: invalid")
        if type(self.rerun_limit) is not int or self.rerun_limit < 0:
            raise FeedbackBudgetError("rerun_limit: invalid")
        object.__setattr__(
            self,
            "coarse_score_boundaries",
            _normalize_boundaries(self.coarse_score_boundaries),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible policy."""
        return {
            "coarse_score_boundaries": list(self.coarse_score_boundaries),
            "detailed_feedback_limit": self.detailed_feedback_limit,
            "epoch_digest": self.epoch_digest,
            "rerun_limit": self.rerun_limit,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class FeedbackBudgetDecision:
    """Privacy-safe result of attempting to record one official execution."""

    accepted: bool
    reason_code: str
    official_attempt_number: int | None
    feedback_level: str
    detailed_score: float | None
    coarse_band: int | None
    official_attempts: int
    scored_attempts: int
    detailed_feedback_remaining: int
    attempts_remaining: int
    schema_version: str = FEEDBACK_BUDGET_DECISION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without submission or failure values."""
        return {
            "accepted": self.accepted,
            "attempts_remaining": self.attempts_remaining,
            "coarse_band": self.coarse_band,
            "detailed_feedback_remaining": self.detailed_feedback_remaining,
            "detailed_score": self.detailed_score,
            "feedback_level": self.feedback_level,
            "official_attempt_number": self.official_attempt_number,
            "official_attempts": self.official_attempts,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
            "scored_attempts": self.scored_attempts,
        }


@dataclass(frozen=True, slots=True)
class FailureApproval:
    """Digest-only receipt for one operator-approved replacement attempt."""

    approved: bool
    reason_code: str
    approved_failure_replacements: int
    attempts_remaining: int
    schema_version: str = FAILURE_APPROVAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic receipt containing no caller values."""
        return {
            "approved": self.approved,
            "approved_failure_replacements": self.approved_failure_replacements,
            "attempts_remaining": self.attempts_remaining,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class FeedbackBudgetSnapshot:
    """Content-free usage counters for one epoch and submission digest."""

    official_attempts: int
    scored_attempts: int
    infrastructure_failures: int
    approved_failure_replacements: int
    detailed_feedback_used: int
    detailed_feedback_remaining: int
    attempts_remaining: int
    schema_version: str = FEEDBACK_BUDGET_SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic counters without caller-provided identifiers."""
        return {
            "approved_failure_replacements": self.approved_failure_replacements,
            "attempts_remaining": self.attempts_remaining,
            "detailed_feedback_remaining": self.detailed_feedback_remaining,
            "detailed_feedback_used": self.detailed_feedback_used,
            "infrastructure_failures": self.infrastructure_failures,
            "official_attempts": self.official_attempts,
            "schema_version": self.schema_version,
            "scored_attempts": self.scored_attempts,
        }


@dataclass(slots=True)
class _Usage:
    official_attempts: int = 0
    scored_attempts: int = 0
    detailed_feedback_used: int = 0
    failure_digests: set[str] = field(default_factory=set)
    approved_failure_digests: set[str] = field(default_factory=set)


class FeedbackBudgetLedger:
    """Thread-safe, in-memory enforcement ledger for official attempts.

    The class performs no I/O or network access. Production callers should
    serialize access to one authoritative instance or persist equivalent
    digest-only events transactionally so process restarts cannot reset usage.
    """

    def __init__(self, policies: Sequence[FeedbackBudgetPolicy]) -> None:
        """Create a ledger with exactly one policy per epoch digest."""
        if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes)):
            raise FeedbackBudgetError("policies: invalid_sequence")
        normalized: dict[str, FeedbackBudgetPolicy] = {}
        for policy in policies:
            if not isinstance(policy, FeedbackBudgetPolicy):
                raise FeedbackBudgetError("policies: invalid_policy")
            if policy.epoch_digest in normalized:
                raise FeedbackBudgetError("policies: duplicate_epoch")
            normalized[policy.epoch_digest] = policy
        if not normalized:
            raise FeedbackBudgetError("policies: empty")
        self._policies: Mapping[str, FeedbackBudgetPolicy] = MappingProxyType(
            dict(sorted(normalized.items()))
        )
        self._usage: dict[tuple[str, str], _Usage] = {}
        self._approval_digests: set[str] = set()
        self._lock = Lock()

    def _policy(self, epoch_digest: Any) -> FeedbackBudgetPolicy:
        digest = _require_digest("epoch_digest", epoch_digest)
        policy = self._policies.get(digest)
        if policy is None:
            raise FeedbackBudgetError("epoch_digest: unknown")
        return policy

    @staticmethod
    def _validate_score(score: Any) -> float:
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise FeedbackBudgetError("score: invalid")
        return float(score)

    @staticmethod
    def _attempt_limit(policy: FeedbackBudgetPolicy, usage: _Usage) -> int:
        return 1 + policy.rerun_limit + len(usage.approved_failure_digests)

    @classmethod
    def _snapshot_for(
        cls, policy: FeedbackBudgetPolicy, usage: _Usage
    ) -> FeedbackBudgetSnapshot:
        attempt_limit = cls._attempt_limit(policy, usage)
        failures = len(usage.failure_digests)
        approved = len(usage.approved_failure_digests)
        return FeedbackBudgetSnapshot(
            official_attempts=usage.official_attempts,
            scored_attempts=usage.scored_attempts,
            infrastructure_failures=failures,
            approved_failure_replacements=approved,
            detailed_feedback_used=usage.detailed_feedback_used,
            detailed_feedback_remaining=max(
                0, policy.detailed_feedback_limit - usage.detailed_feedback_used
            ),
            attempts_remaining=max(0, attempt_limit - usage.official_attempts),
        )

    def record_official_attempt(
        self,
        epoch_digest: str,
        submission_digest: str,
        *,
        score: float | None = None,
        infrastructure_failure_digest: str | None = None,
    ) -> FeedbackBudgetDecision:
        """Record a scored attempt or documented infrastructure failure.

        Exactly one of ``score`` and ``infrastructure_failure_digest`` is
        required. A failure consumes an official attempt. It increases the
        allowed attempt count only after :meth:`approve_infrastructure_failure`
        records a unique operator approval.
        """
        policy = self._policy(epoch_digest)
        submission = _require_digest("submission_digest", submission_digest)
        has_score = score is not None
        has_failure = infrastructure_failure_digest is not None
        if has_score == has_failure:
            raise FeedbackBudgetError("attempt_outcome: require_exactly_one")
        normalized_score = self._validate_score(score) if has_score else None
        failure = (
            _require_digest(
                "infrastructure_failure_digest", infrastructure_failure_digest
            )
            if has_failure
            else None
        )
        key = (policy.epoch_digest, submission)

        with self._lock:
            usage = self._usage.setdefault(key, _Usage())
            snapshot = self._snapshot_for(policy, usage)
            if snapshot.attempts_remaining == 0:
                return FeedbackBudgetDecision(
                    accepted=False,
                    reason_code=REASON_RERUN_BUDGET_EXHAUSTED,
                    official_attempt_number=None,
                    feedback_level=FEEDBACK_NONE,
                    detailed_score=None,
                    coarse_band=None,
                    official_attempts=snapshot.official_attempts,
                    scored_attempts=snapshot.scored_attempts,
                    detailed_feedback_remaining=(snapshot.detailed_feedback_remaining),
                    attempts_remaining=0,
                )
            if failure is not None and failure in usage.failure_digests:
                raise FeedbackBudgetError(
                    "infrastructure_failure_digest: already_recorded"
                )

            usage.official_attempts += 1
            feedback_level = FEEDBACK_NONE
            detailed_score: float | None = None
            coarse_band: int | None = None
            if normalized_score is not None:
                usage.scored_attempts += 1
                if usage.detailed_feedback_used < policy.detailed_feedback_limit:
                    usage.detailed_feedback_used += 1
                    feedback_level = FEEDBACK_DETAILED
                    detailed_score = normalized_score
                else:
                    feedback_level = FEEDBACK_COARSE
                    coarse_band = bisect.bisect_right(
                        policy.coarse_score_boundaries, normalized_score
                    )
            else:
                assert failure is not None
                usage.failure_digests.add(failure)

            updated = self._snapshot_for(policy, usage)
            return FeedbackBudgetDecision(
                accepted=True,
                reason_code=REASON_ATTEMPT_RECORDED,
                official_attempt_number=updated.official_attempts,
                feedback_level=feedback_level,
                detailed_score=detailed_score,
                coarse_band=coarse_band,
                official_attempts=updated.official_attempts,
                scored_attempts=updated.scored_attempts,
                detailed_feedback_remaining=updated.detailed_feedback_remaining,
                attempts_remaining=updated.attempts_remaining,
            )

    def approve_infrastructure_failure(
        self,
        epoch_digest: str,
        submission_digest: str,
        failure_digest: str,
        operator_approval_digest: str,
    ) -> FailureApproval:
        """Grant one replacement attempt for a recorded failure.

        ``operator_approval_digest`` must identify an authorization record from
        the caller's access-controlled operator workflow. This method validates
        binding and one-time use; authentication remains the caller's concern.
        """
        policy = self._policy(epoch_digest)
        submission = _require_digest("submission_digest", submission_digest)
        failure = _require_digest("failure_digest", failure_digest)
        approval = _require_digest("operator_approval_digest", operator_approval_digest)
        key = (policy.epoch_digest, submission)

        with self._lock:
            usage = self._usage.get(key)
            if usage is None or failure not in usage.failure_digests:
                raise FeedbackBudgetError("failure_digest: not_recorded")
            if failure in usage.approved_failure_digests:
                raise FeedbackBudgetError("failure_digest: already_approved")
            if approval in self._approval_digests:
                raise FeedbackBudgetError("operator_approval_digest: already_used")
            usage.approved_failure_digests.add(failure)
            self._approval_digests.add(approval)
            snapshot = self._snapshot_for(policy, usage)
            return FailureApproval(
                approved=True,
                reason_code=REASON_FAILURE_APPROVED,
                approved_failure_replacements=(snapshot.approved_failure_replacements),
                attempts_remaining=snapshot.attempts_remaining,
            )

    def snapshot(
        self, epoch_digest: str, submission_digest: str
    ) -> FeedbackBudgetSnapshot:
        """Return content-free counters for one epoch/submission pair."""
        policy = self._policy(epoch_digest)
        submission = _require_digest("submission_digest", submission_digest)
        with self._lock:
            usage = self._usage.get((policy.epoch_digest, submission), _Usage())
            return self._snapshot_for(policy, usage)


__all__ = [
    "FAILURE_APPROVAL_SCHEMA_VERSION",
    "FEEDBACK_BUDGET_DECISION_SCHEMA_VERSION",
    "FEEDBACK_BUDGET_POLICY_SCHEMA_VERSION",
    "FEEDBACK_BUDGET_SNAPSHOT_SCHEMA_VERSION",
    "FEEDBACK_COARSE",
    "FEEDBACK_DETAILED",
    "FEEDBACK_NONE",
    "REASON_ATTEMPT_RECORDED",
    "REASON_FAILURE_APPROVED",
    "REASON_RERUN_BUDGET_EXHAUSTED",
    "FailureApproval",
    "FeedbackBudgetDecision",
    "FeedbackBudgetError",
    "FeedbackBudgetLedger",
    "FeedbackBudgetPolicy",
    "FeedbackBudgetSnapshot",
]
