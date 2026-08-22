"""Deterministic, privacy-safe validation for clinical review transitions.

This module models the human-review state machine independently from any
clinical inference.  A transition is accepted only when the configured policy
allows it and the caller supplies an opaque event identifier plus a provenance
fingerprint.  Transition records intentionally contain no reviewer identity,
case content, timestamps, or free-text notes.

The state machine is a local validation primitive.  It makes no network calls,
does not make clinical decisions, and does not imply that a review outcome is
clinically correct.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

REVIEW_TRANSITIONS_SCHEMA_VERSION = "openmed.clinical_review_transitions.v1"
REVIEW_TRANSITION_ADVISORY = (
    "Clinical review-state transitions are deterministic assistive workflow "
    "metadata, not a medical-device decision or a substitute for qualified "
    "clinical judgment."
)


class ReviewState(str, Enum):
    """States supported by the guarded clinical review workflow."""

    QUEUED = "queued"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REOPENED = "reopened"


REVIEW_STATES = tuple(state.value for state in ReviewState)

# The default graph deliberately makes reopening pass through ``reopened`` and
# then ``in_review``.  A previously approved, rejected, or expired result can
# never be approved again without a new review transition.
DEFAULT_REVIEW_TRANSITIONS: Mapping[ReviewState, tuple[ReviewState, ...]] = (
    MappingProxyType(
        {
            ReviewState.QUEUED: (ReviewState.IN_REVIEW, ReviewState.EXPIRED),
            ReviewState.IN_REVIEW: (
                ReviewState.APPROVED,
                ReviewState.REJECTED,
                ReviewState.EXPIRED,
            ),
            ReviewState.APPROVED: (ReviewState.REOPENED,),
            ReviewState.REJECTED: (ReviewState.REOPENED,),
            ReviewState.EXPIRED: (ReviewState.REOPENED,),
            ReviewState.REOPENED: (ReviewState.IN_REVIEW,),
        }
    )
)

_EVENT_ID_RE = re.compile(
    r"^(?:evt_[0-9a-f]{16,128}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
_FINGERPRINT_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")


class ReviewTransitionValidationError(ValueError):
    """Raised when a review transition fails a safe workflow invariant.

    The exception message contains only a stable diagnostic code and known
    review-state values.  It never echoes event identifiers, fingerprints,
    policy inputs, reviewer identities, or case content.
    """

    def __init__(
        self,
        code: str,
        *,
        from_state: ReviewState | None = None,
        to_state: ReviewState | None = None,
    ) -> None:
        safe_code = (
            code
            if isinstance(code, str) and _IDENTIFIER_RE.fullmatch(code)
            else "validation_error"
        )
        self.code = safe_code
        self.from_state = from_state
        self.to_state = to_state
        from_value = (
            from_state.value if isinstance(from_state, ReviewState) else "unknown"
        )
        to_value = to_state.value if isinstance(to_state, ReviewState) else "unknown"
        super().__init__(
            "review transition rejected: "
            f"code={safe_code} from_state={from_value} to_state={to_value}"
        )


# Short aliases make the validation error discoverable without duplicating the
# implementation or changing the safe diagnostic contract.
ReviewTransitionError = ReviewTransitionValidationError
TransitionValidationError = ReviewTransitionValidationError


def _coerce_state(
    value: ReviewState | str,
    *,
    from_state: ReviewState | None = None,
    to_state: ReviewState | None = None,
) -> ReviewState:
    if isinstance(value, ReviewState):
        return value
    if isinstance(value, str):
        try:
            return ReviewState(value.strip().casefold())
        except ValueError:
            pass
    raise ReviewTransitionValidationError(
        "invalid_state",
        from_state=from_state,
        to_state=to_state,
    )


def _state_set(
    values: Iterable[ReviewState | str],
    *,
    from_state: ReviewState | None = None,
    to_state: ReviewState | None = None,
) -> frozenset[ReviewState]:
    try:
        return frozenset(
            _coerce_state(
                value,
                from_state=from_state,
                to_state=to_state,
            )
            for value in values
        )
    except TypeError as exc:
        raise ReviewTransitionValidationError(
            "invalid_state_set",
            from_state=from_state,
            to_state=to_state,
        ) from exc


def _normalise_event_id(
    value: object,
    *,
    from_state: ReviewState | None = None,
    to_state: ReviewState | None = None,
) -> str:
    if not isinstance(value, str) or not _EVENT_ID_RE.fullmatch(value):
        raise ReviewTransitionValidationError(
            "opaque_event_id_required",
            from_state=from_state,
            to_state=to_state,
        )
    return value.lower()


def _normalise_fingerprint(
    value: object,
    *,
    from_state: ReviewState | None = None,
    to_state: ReviewState | None = None,
) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
        raise ReviewTransitionValidationError(
            "provenance_fingerprint_required",
            from_state=from_state,
            to_state=to_state,
        )
    digest = value.removeprefix("sha256:").lower()
    return f"sha256:{digest}"


def _normalise_reason(
    value: str | None,
    *,
    from_state: ReviewState | None = None,
    to_state: ReviewState | None = None,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _REASON_CODE_RE.fullmatch(value):
        raise ReviewTransitionValidationError(
            "safe_reason_code_required",
            from_state=from_state,
            to_state=to_state,
        )
    return value.lower()


def _normalise_policy_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("policy_id must be a safe identifier")
    normalized = value.strip().casefold()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError("policy_id must be a safe identifier")
    return normalized


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("fingerprint input must be JSON-compatible") from exc
    return encoded.encode("utf-8")


def compute_provenance_fingerprint(provenance: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for safe provenance input.

    The input is hashed and is never retained by this module.  Callers should
    provide only non-sensitive provenance such as a schema version, policy
    identifier, or upstream artifact digest; case contents and reviewer
    identities are not valid provenance payloads for a review record.
    """

    digest = hashlib.sha256(_canonical_bytes(provenance)).hexdigest()
    return f"sha256:{digest}"


def make_opaque_event_id(seed: Any) -> str:
    """Derive a deterministic opaque event identifier from caller input.

    Only the derived token is returned.  The seed is not stored in a review
    record, so callers can use a synthetic sequence or an existing safe event
    digest without introducing free text into the audit surface.
    """

    digest = hashlib.sha256(_canonical_bytes(seed)).hexdigest()
    return f"evt_{digest[:32]}"


# Readable aliases for callers that prefer noun-first helper names.
provenance_fingerprint = compute_provenance_fingerprint
opaque_event_id = make_opaque_event_id
make_event_id = make_opaque_event_id


@dataclass(frozen=True)
class ReviewTransitionRequest:
    """PHI-free inputs presented to a transition policy rule."""

    from_state: ReviewState
    to_state: ReviewState
    event_id: str
    provenance_fingerprint: str
    reason_code: str | None = None

    def __post_init__(self) -> None:
        from_state = _coerce_state(self.from_state)
        to_state = _coerce_state(self.to_state, from_state=from_state)
        event_id = _normalise_event_id(
            self.event_id,
            from_state=from_state,
            to_state=to_state,
        )
        fingerprint = _normalise_fingerprint(
            self.provenance_fingerprint,
            from_state=from_state,
            to_state=to_state,
        )
        reason = _normalise_reason(
            self.reason_code,
            from_state=from_state,
            to_state=to_state,
        )
        object.__setattr__(self, "from_state", from_state)
        object.__setattr__(self, "to_state", to_state)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "provenance_fingerprint", fingerprint)
        object.__setattr__(self, "reason_code", reason)

    def to_dict(self) -> dict[str, Any]:
        """Return the safe policy-input representation."""

        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "event_id": self.event_id,
            "provenance_fingerprint": self.provenance_fingerprint,
            "reason_code": self.reason_code,
        }


TransitionRuleCheck: TypeAlias = Callable[[ReviewTransitionRequest], bool | None]


@dataclass(frozen=True)
class ReviewPolicyRule:
    """One injected, named predicate for a transition policy.

    A rule receives only :class:`ReviewTransitionRequest`, which contains no
    reviewer identity or case content.  Returning ``False`` rejects the
    transition using the rule's safe ``code``.  Returning ``True`` or ``None``
    permits evaluation to continue.
    """

    code: str
    check: TransitionRuleCheck

    def __post_init__(self) -> None:
        code = _normalise_policy_id(self.code)
        if not callable(self.check):
            raise ValueError("policy rule check must be callable")
        object.__setattr__(self, "code", code)

    def evaluate(self, request: ReviewTransitionRequest) -> None:
        """Evaluate this rule without exposing predicate details in errors."""

        try:
            result = self.check(request)
        except ReviewTransitionValidationError:
            raise
        except Exception as exc:
            raise ReviewTransitionValidationError(
                "policy_rule_error",
                from_state=request.from_state,
                to_state=request.to_state,
            ) from exc
        if result is False:
            raise ReviewTransitionValidationError(
                self.code,
                from_state=request.from_state,
                to_state=request.to_state,
            )


def _normalise_rules(value: object) -> tuple[ReviewPolicyRule, ...]:
    if value is None:
        return ()
    if isinstance(value, ReviewPolicyRule):
        candidates: Sequence[object] = (value,)
    elif isinstance(value, Mapping):
        candidates: Sequence[object] = tuple(value.items())
    elif callable(value):
        candidates = (value,)
    else:
        try:
            candidates = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("rules must be callable or iterable") from exc

    rules: list[ReviewPolicyRule] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, ReviewPolicyRule):
            rule = candidate
        elif isinstance(candidate, tuple) and len(candidate) == 2:
            code, check = candidate
            rule = ReviewPolicyRule(str(code), check)
        elif callable(candidate):
            code = getattr(candidate, "__name__", "custom_rule")
            if not isinstance(code, str) or not _IDENTIFIER_RE.fullmatch(code):
                code = "custom_rule" if index == 0 else f"custom_rule_{index}"
            rule = ReviewPolicyRule(str(code), candidate)
        else:
            raise ValueError("rules must contain named callables")
        if any(existing.code == rule.code for existing in rules):
            raise ValueError("policy rule codes must be unique")
        rules.append(rule)
    return tuple(rules)


def _normalise_allowed_transitions(
    value: Mapping[ReviewState | str, Iterable[ReviewState | str]] | None,
) -> Mapping[ReviewState, tuple[ReviewState, ...]]:
    source = DEFAULT_REVIEW_TRANSITIONS if value is None else value
    if not isinstance(source, Mapping):
        raise ValueError("allowed_transitions must be a mapping")

    normalized: dict[ReviewState, tuple[ReviewState, ...]] = {}
    try:
        entries = source.items()
    except AttributeError as exc:
        raise ValueError("allowed_transitions must be a mapping") from exc
    for raw_from, raw_targets in entries:
        from_state = _coerce_state(raw_from)
        if isinstance(raw_targets, (str, ReviewState)):
            target_values: Iterable[ReviewState | str] = (raw_targets,)
        else:
            target_values = raw_targets
        try:
            targets = tuple(_coerce_state(target) for target in target_values)
        except TypeError as exc:
            raise ValueError("transition targets must be iterable") from exc
        if len(set(targets)) != len(targets):
            raise ValueError("transition targets must be unique")
        normalized[from_state] = tuple(sorted(targets, key=lambda state: state.value))

    for state in ReviewState:
        normalized.setdefault(state, ())
    return MappingProxyType(
        dict(sorted(normalized.items(), key=lambda item: item[0].value))
    )


@dataclass(frozen=True)
class ReviewTransitionPolicy:
    """Static transition graph plus injected PHI-free policy rules."""

    policy_id: str = "default"
    allowed_transitions: (
        Mapping[ReviewState | str, Iterable[ReviewState | str]] | None
    ) = None
    required_reason_states: Iterable[ReviewState | str] = ()
    rules: object = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _normalise_policy_id(self.policy_id))
        object.__setattr__(
            self,
            "allowed_transitions",
            _normalise_allowed_transitions(self.allowed_transitions),
        )
        object.__setattr__(
            self,
            "required_reason_states",
            _state_set(self.required_reason_states),
        )
        object.__setattr__(self, "rules", _normalise_rules(self.rules))

    @property
    def fingerprint(self) -> str:
        """Return the deterministic fingerprint of this policy contract."""

        payload = {
            "schema_version": REVIEW_TRANSITIONS_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "allowed_transitions": {
                state.value: [target.value for target in targets]
                for state, targets in self.allowed_transitions.items()
            },
            "required_reason_states": sorted(
                state.value for state in self.required_reason_states
            ),
            "rule_codes": [rule.code for rule in self.rules],
        }
        return compute_provenance_fingerprint(payload)

    @property
    def policy_fingerprint(self) -> str:
        """Alias for :attr:`fingerprint`."""

        return self.fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return a safe, JSON-compatible policy description."""

        return {
            "policy_id": self.policy_id,
            "allowed_transitions": {
                state.value: [target.value for target in targets]
                for state, targets in self.allowed_transitions.items()
            },
            "required_reason_states": sorted(
                state.value for state in self.required_reason_states
            ),
            "rule_codes": [rule.code for rule in self.rules],
            "fingerprint": self.fingerprint,
        }

    def validate(self, request: ReviewTransitionRequest) -> None:
        """Raise when ``request`` violates this policy."""

        if request.to_state not in self.allowed_transitions[request.from_state]:
            raise ReviewTransitionValidationError(
                "transition_not_allowed",
                from_state=request.from_state,
                to_state=request.to_state,
            )
        if (
            request.to_state in self.required_reason_states
            and request.reason_code is None
        ):
            raise ReviewTransitionValidationError(
                "reason_code_required",
                from_state=request.from_state,
                to_state=request.to_state,
            )
        for rule in self.rules:
            rule.evaluate(request)


@dataclass(frozen=True)
class ReviewTransition:
    """One validated, PHI-free transition event."""

    sequence: int
    from_state: ReviewState
    to_state: ReviewState
    event_id: str
    provenance_fingerprint: str
    policy_fingerprint: str
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("transition sequence must be a positive integer")
        if self.sequence < 1:
            raise ValueError("transition sequence must be a positive integer")
        from_state = _coerce_state(self.from_state)
        to_state = _coerce_state(self.to_state, from_state=from_state)
        event_id = _normalise_event_id(
            self.event_id,
            from_state=from_state,
            to_state=to_state,
        )
        provenance = _normalise_fingerprint(
            self.provenance_fingerprint,
            from_state=from_state,
            to_state=to_state,
        )
        policy = _normalise_fingerprint(
            self.policy_fingerprint,
            from_state=from_state,
            to_state=to_state,
        )
        reason = _normalise_reason(
            self.reason_code,
            from_state=from_state,
            to_state=to_state,
        )
        object.__setattr__(self, "from_state", from_state)
        object.__setattr__(self, "to_state", to_state)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "provenance_fingerprint", provenance)
        object.__setattr__(self, "policy_fingerprint", policy)
        object.__setattr__(self, "reason_code", reason)

    @property
    def previous_state(self) -> ReviewState:
        """Alias for the state before this transition."""

        return self.from_state

    @property
    def next_state(self) -> ReviewState:
        """Alias for the state after this transition."""

        return self.to_state

    def to_dict(self) -> dict[str, Any]:
        """Return the transition without reviewer or case data."""

        return {
            "sequence": self.sequence,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "event_id": self.event_id,
            "provenance_fingerprint": self.provenance_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewTransition":
        """Rebuild a transition from its safe serialized representation."""

        if not isinstance(payload, Mapping):
            raise ReviewTransitionValidationError("transition_mapping_required")
        try:
            return cls(
                sequence=payload["sequence"],
                from_state=payload["from_state"],
                to_state=payload["to_state"],
                event_id=payload["event_id"],
                provenance_fingerprint=payload["provenance_fingerprint"],
                policy_fingerprint=payload["policy_fingerprint"],
                reason_code=payload.get("reason_code"),
            )
        except KeyError as exc:
            raise ReviewTransitionValidationError("transition_field_required") from exc


ReviewTransitionEvent = ReviewTransition


@dataclass(frozen=True)
class ReviewTransitionReport:
    """A deterministic, privacy-safe review transition report."""

    initial_state: ReviewState
    current_state: ReviewState
    transitions: tuple[ReviewTransition, ...]
    policy_fingerprint: str
    valid: bool = True
    schema_version: str = REVIEW_TRANSITIONS_SCHEMA_VERSION
    advisory: str = REVIEW_TRANSITION_ADVISORY

    def __post_init__(self) -> None:
        initial = _coerce_state(self.initial_state)
        current = _coerce_state(self.current_state)
        if not isinstance(self.transitions, tuple):
            object.__setattr__(self, "transitions", tuple(self.transitions))
        if any(not isinstance(item, ReviewTransition) for item in self.transitions):
            raise ValueError("transitions must contain ReviewTransition records")
        policy = _normalise_fingerprint(self.policy_fingerprint)
        object.__setattr__(self, "initial_state", initial)
        object.__setattr__(self, "current_state", current)
        object.__setattr__(self, "policy_fingerprint", policy)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report with no raw workflow payloads."""

        return {
            "schema_version": self.schema_version,
            "initial_state": self.initial_state.value,
            "current_state": self.current_state.value,
            "valid": self.valid,
            "policy_fingerprint": self.policy_fingerprint,
            "transition_count": len(self.transitions),
            "transitions": [item.to_dict() for item in self.transitions],
            "advisory": self.advisory,
        }


class ReviewStateMachine:
    """Mutable validator for one ordered review-state history.

    The machine mutates only its state and safe transition records.  It never
    accepts a reviewer identity or case payload, and every accepted transition
    is checked against the injected :class:`ReviewTransitionPolicy`.
    """

    def __init__(
        self,
        *,
        initial_state: ReviewState | str = ReviewState.QUEUED,
        policy: ReviewTransitionPolicy | None = None,
    ) -> None:
        self._initial_state = _coerce_state(initial_state)
        self._state = self._initial_state
        self._policy = policy if policy is not None else ReviewTransitionPolicy()
        if not isinstance(self._policy, ReviewTransitionPolicy):
            raise TypeError("policy must be a ReviewTransitionPolicy")
        self._transitions: list[ReviewTransition] = []

    @property
    def initial_state(self) -> ReviewState:
        """Return the machine's starting state."""

        return self._initial_state

    @property
    def current_state(self) -> ReviewState:
        """Return the state after the latest accepted transition."""

        return self._state

    @property
    def state(self) -> ReviewState:
        """Alias for :attr:`current_state`."""

        return self.current_state

    @property
    def policy(self) -> ReviewTransitionPolicy:
        """Return the immutable policy used by this machine."""

        return self._policy

    @property
    def transitions(self) -> tuple[ReviewTransition, ...]:
        """Return accepted transitions in insertion order."""

        return tuple(self._transitions)

    @property
    def history(self) -> tuple[ReviewTransition, ...]:
        """Alias for :attr:`transitions`."""

        return self.transitions

    def validate_transition(
        self,
        to_state: ReviewState | str,
        event_id: str,
        provenance_fingerprint: str,
        *,
        reason_code: str | None = None,
    ) -> ReviewTransitionRequest:
        """Validate the next transition without mutating the machine."""

        next_state = _coerce_state(to_state, from_state=self._state)
        request = ReviewTransitionRequest(
            from_state=self._state,
            to_state=next_state,
            event_id=event_id,
            provenance_fingerprint=provenance_fingerprint,
            reason_code=reason_code,
        )
        if request.event_id in {item.event_id for item in self._transitions}:
            raise ReviewTransitionValidationError(
                "duplicate_event_id",
                from_state=request.from_state,
                to_state=request.to_state,
            )
        self._policy.validate(request)
        return request

    def can_transition(
        self,
        to_state: ReviewState | str,
        event_id: str,
        provenance_fingerprint: str,
        *,
        reason_code: str | None = None,
    ) -> bool:
        """Return whether the next transition satisfies the policy."""

        try:
            self.validate_transition(
                to_state,
                event_id,
                provenance_fingerprint,
                reason_code=reason_code,
            )
        except ReviewTransitionValidationError:
            return False
        return True

    def transition(
        self,
        to_state: ReviewState | str,
        event_id: str,
        provenance_fingerprint: str,
        *,
        reason_code: str | None = None,
    ) -> ReviewTransition:
        """Validate and append one transition, returning its safe record."""

        request = self.validate_transition(
            to_state,
            event_id,
            provenance_fingerprint,
            reason_code=reason_code,
        )
        record = ReviewTransition(
            sequence=len(self._transitions) + 1,
            from_state=request.from_state,
            to_state=request.to_state,
            event_id=request.event_id,
            provenance_fingerprint=request.provenance_fingerprint,
            policy_fingerprint=self._policy.fingerprint,
            reason_code=request.reason_code,
        )
        self._transitions.append(record)
        self._state = record.to_state
        return record

    def report(self) -> ReviewTransitionReport:
        """Return the current deterministic, privacy-safe report."""

        return ReviewTransitionReport(
            initial_state=self._initial_state,
            current_state=self._state,
            transitions=self.transitions,
            policy_fingerprint=self._policy.fingerprint,
        )

    @property
    def report_value(self) -> ReviewTransitionReport:
        """Property form of :meth:`report` for report-oriented callers."""

        return self.report()


def validate_transition(
    from_state: ReviewState | str,
    to_state: ReviewState | str,
    event_id: str,
    provenance_fingerprint: str,
    *,
    policy: ReviewTransitionPolicy | None = None,
    reason_code: str | None = None,
) -> ReviewTransitionRequest:
    """Validate one transition without creating a mutable machine."""

    source = _coerce_state(from_state)
    target = _coerce_state(to_state, from_state=source)
    request = ReviewTransitionRequest(
        from_state=source,
        to_state=target,
        event_id=event_id,
        provenance_fingerprint=provenance_fingerprint,
        reason_code=reason_code,
    )
    resolved_policy = policy if policy is not None else ReviewTransitionPolicy()
    if not isinstance(resolved_policy, ReviewTransitionPolicy):
        raise TypeError("policy must be a ReviewTransitionPolicy")
    resolved_policy.validate(request)
    return request


validate_review_transition = validate_transition


def validate_review_history(
    transitions: Iterable[ReviewTransition | Mapping[str, Any]],
    *,
    initial_state: ReviewState | str = ReviewState.QUEUED,
    policy: ReviewTransitionPolicy | None = None,
) -> ReviewTransitionReport:
    """Validate an ordered safe transition history and return its report.

    Mapping inputs are read only for the fields defined by
    :class:`ReviewTransition`; extra fields are ignored and never copied into
    the returned report.  This allows callers to fail closed when a source
    payload contains accidental reviewer or case fields.
    """

    machine = ReviewStateMachine(initial_state=initial_state, policy=policy)
    expected_sequence = 1
    for raw_transition in transitions:
        transition = (
            raw_transition
            if isinstance(raw_transition, ReviewTransition)
            else ReviewTransition.from_dict(raw_transition)
        )
        if transition.sequence != expected_sequence:
            raise ReviewTransitionValidationError(
                "sequence_mismatch",
                from_state=machine.current_state,
                to_state=transition.to_state,
            )
        if transition.policy_fingerprint != machine.policy.fingerprint:
            raise ReviewTransitionValidationError(
                "policy_fingerprint_mismatch",
                from_state=machine.current_state,
                to_state=transition.to_state,
            )
        if transition.from_state != machine.current_state:
            raise ReviewTransitionValidationError(
                "history_state_mismatch",
                from_state=machine.current_state,
                to_state=transition.to_state,
            )
        machine.transition(
            transition.to_state,
            transition.event_id,
            transition.provenance_fingerprint,
            reason_code=transition.reason_code,
        )
        expected_sequence += 1
    return machine.report()


__all__ = [
    "DEFAULT_REVIEW_TRANSITIONS",
    "REVIEW_STATES",
    "REVIEW_TRANSITIONS_SCHEMA_VERSION",
    "REVIEW_TRANSITION_ADVISORY",
    "ReviewPolicyRule",
    "ReviewState",
    "ReviewStateMachine",
    "ReviewTransition",
    "ReviewTransitionError",
    "ReviewTransitionEvent",
    "ReviewTransitionPolicy",
    "ReviewTransitionReport",
    "ReviewTransitionRequest",
    "ReviewTransitionValidationError",
    "TransitionValidationError",
    "compute_provenance_fingerprint",
    "make_event_id",
    "make_opaque_event_id",
    "opaque_event_id",
    "provenance_fingerprint",
    "validate_review_history",
    "validate_review_transition",
    "validate_transition",
]
