"""Bounded, digest-keyed retry control for local summary generation.

The controller fails closed at a caller-defined attempt ceiling and when the
same safety-failure class repeats. Its decisions contain only counts, fixed
codes, and an opaque generation-key digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Final

SUMMARY_RETRY_CONTROL_SCHEMA_VERSION: Final[int] = 1
RETRY_CEILING_REFUSAL: Final[str] = "summary_retry_ceiling_exhausted"
REPEATED_FAILURE_REFUSAL: Final[str] = "summary_repeated_failure_class"

_OPAQUE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACTIONS = frozenset({"generate", "retry", "complete", "refuse"})


class SummaryRetryControlError(ValueError):
    """Raised when retry-control input or state is invalid."""


@dataclass(frozen=True, slots=True)
class SummaryRetryPolicy:
    """Policy limits for one retry controller.

    Args:
        max_attempts: Maximum generation attempts for an input/model pair.
        repeated_failure_limit: Occurrence count at which one repeated failure
            class permanently refuses further attempts for that pair.
    """

    max_attempts: int
    repeated_failure_limit: int = 2

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise SummaryRetryControlError("max attempts must be a positive integer")
        if (
            type(self.repeated_failure_limit) is not int
            or self.repeated_failure_limit < 2
        ):
            raise SummaryRetryControlError(
                "repeated failure limit must be an integer of at least two"
            )


@dataclass(frozen=True, slots=True)
class SummaryRetryDecision:
    """Value-free decision for one retry-control transition."""

    generation_key: str
    action: str
    attempts_started: int
    attempts_remaining: int
    refusal_code: str | None = None
    schema_version: int = SUMMARY_RETRY_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_digest(self.generation_key, "generation key")
        if self.action not in _ACTIONS:
            raise SummaryRetryControlError("unsupported retry action")
        if type(self.attempts_started) is not int or self.attempts_started < 0:
            raise SummaryRetryControlError("invalid attempts-started count")
        if type(self.attempts_remaining) is not int or self.attempts_remaining < 0:
            raise SummaryRetryControlError("invalid attempts-remaining count")
        if self.action == "refuse":
            if self.refusal_code not in {
                RETRY_CEILING_REFUSAL,
                REPEATED_FAILURE_REFUSAL,
            }:
                raise SummaryRetryControlError("invalid retry refusal code")
        elif self.refusal_code is not None:
            raise SummaryRetryControlError("non-refusal action has a refusal code")

    @property
    def refused(self) -> bool:
        """Return whether generation must stop for this key."""

        return self.action == "refuse"

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic audit representation."""

        return {
            "action": self.action,
            "attempts_remaining": self.attempts_remaining,
            "attempts_started": self.attempts_started,
            "generation_key": self.generation_key,
            "refusal_code": self.refusal_code,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize this decision with stable ordering and separators."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(slots=True)
class _RetryState:
    attempts_started: int = 0
    active_attempt: bool = False
    failure_counts: dict[str, int] = field(default_factory=dict)
    refusal_code: str | None = None


class SummaryRetryController:
    """Thread-safe retry state keyed by immutable input and model digests.

    Call :meth:`begin_attempt` immediately before each generation. Finish that
    active attempt with exactly one call to :meth:`record_failure` or
    :meth:`record_success`. A refusal is sticky for the input/model pair.
    """

    def __init__(self, policy: SummaryRetryPolicy) -> None:
        if not isinstance(policy, SummaryRetryPolicy):
            raise SummaryRetryControlError("invalid summary retry policy")
        self.policy = policy
        self._states: dict[tuple[str, str], _RetryState] = {}
        self._audit_events: list[SummaryRetryDecision] = []
        self._lock = RLock()

    def begin_attempt(
        self,
        input_digest: str,
        model_digest: str,
    ) -> SummaryRetryDecision:
        """Authorize one generation attempt or return a sticky refusal."""

        key = _validated_key(input_digest, model_digest)
        generation_key = _generation_key(*key)
        with self._lock:
            state = self._states.setdefault(key, _RetryState())
            if state.active_attempt:
                raise SummaryRetryControlError("a generation attempt is already active")
            if state.refusal_code is not None:
                return self._record_decision(_refusal_decision(generation_key, state))
            if state.attempts_started >= self.policy.max_attempts:
                state.refusal_code = RETRY_CEILING_REFUSAL
                return self._record_decision(_refusal_decision(generation_key, state))

            state.attempts_started += 1
            state.active_attempt = True
            return self._record_decision(
                SummaryRetryDecision(
                    generation_key=generation_key,
                    action="generate",
                    attempts_started=state.attempts_started,
                    attempts_remaining=self.policy.max_attempts
                    - state.attempts_started,
                )
            )

    def record_failure(
        self,
        input_digest: str,
        model_digest: str,
        failure_class: str,
    ) -> SummaryRetryDecision:
        """Finish an attempt and decide whether a retry remains permitted.

        The raw failure class is fingerprinted for internal counting and is
        absent from returned decisions and audit events.
        """

        key = _validated_key(input_digest, model_digest)
        failure_fingerprint = _failure_fingerprint(failure_class)
        generation_key = _generation_key(*key)
        with self._lock:
            state = self._states.get(key)
            if state is None or not state.active_attempt:
                raise SummaryRetryControlError("no active generation attempt")
            state.active_attempt = False
            state.failure_counts[failure_fingerprint] = (
                state.failure_counts.get(failure_fingerprint, 0) + 1
            )
            if (
                state.failure_counts[failure_fingerprint]
                >= self.policy.repeated_failure_limit
            ):
                state.refusal_code = REPEATED_FAILURE_REFUSAL
            elif state.attempts_started >= self.policy.max_attempts:
                state.refusal_code = RETRY_CEILING_REFUSAL

            if state.refusal_code is not None:
                decision = _refusal_decision(generation_key, state)
            else:
                decision = SummaryRetryDecision(
                    generation_key=generation_key,
                    action="retry",
                    attempts_started=state.attempts_started,
                    attempts_remaining=self.policy.max_attempts
                    - state.attempts_started,
                )
            return self._record_decision(decision)

    def record_success(
        self,
        input_digest: str,
        model_digest: str,
    ) -> SummaryRetryDecision:
        """Finish a successful attempt and clear retry state for its key."""

        key = _validated_key(input_digest, model_digest)
        generation_key = _generation_key(*key)
        with self._lock:
            state = self._states.get(key)
            if state is None or not state.active_attempt:
                raise SummaryRetryControlError("no active generation attempt")
            decision = SummaryRetryDecision(
                generation_key=generation_key,
                action="complete",
                attempts_started=state.attempts_started,
                attempts_remaining=self.policy.max_attempts - state.attempts_started,
            )
            del self._states[key]
            return self._record_decision(decision)

    def audit_log(self) -> tuple[dict[str, object], ...]:
        """Return value-free decision events in transition order."""

        with self._lock:
            return tuple(event.to_dict() for event in self._audit_events)

    def _record_decision(
        self,
        decision: SummaryRetryDecision,
    ) -> SummaryRetryDecision:
        self._audit_events.append(decision)
        return decision


def _refusal_decision(
    generation_key: str,
    state: _RetryState,
) -> SummaryRetryDecision:
    if state.refusal_code is None:
        raise SummaryRetryControlError("missing retry refusal code")
    return SummaryRetryDecision(
        generation_key=generation_key,
        action="refuse",
        attempts_started=state.attempts_started,
        attempts_remaining=0,
        refusal_code=state.refusal_code,
    )


def _validated_key(input_digest: object, model_digest: object) -> tuple[str, str]:
    _validate_digest(input_digest, "input")
    _validate_digest(model_digest, "model")
    return input_digest, model_digest  # type: ignore[return-value]


def _validate_digest(value: object, kind: str) -> None:
    if type(value) is not str or _OPAQUE_DIGEST_RE.fullmatch(value) is None:
        raise SummaryRetryControlError(f"invalid {kind} digest")


def _generation_key(input_digest: str, model_digest: str) -> str:
    payload = json.dumps(
        ["summary-generation", input_digest, model_digest],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _failure_fingerprint(failure_class: object) -> str:
    if type(failure_class) is not str or not failure_class.strip():
        raise SummaryRetryControlError("invalid summary failure class")
    return "sha256:" + hashlib.sha256(failure_class.strip().encode()).hexdigest()


__all__ = [
    "REPEATED_FAILURE_REFUSAL",
    "RETRY_CEILING_REFUSAL",
    "SUMMARY_RETRY_CONTROL_SCHEMA_VERSION",
    "SummaryRetryControlError",
    "SummaryRetryController",
    "SummaryRetryDecision",
    "SummaryRetryPolicy",
]
