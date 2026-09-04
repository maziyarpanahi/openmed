"""Deterministic expiry enforcement for structured privacy access reviews.

The access review model contains only review metadata: issue and expiry times,
an opaque policy fingerprint, and decision-category identifiers. It deliberately
has no identity, request, record, or free-text fields. The evaluator requires a
caller-supplied clock and returns a stable, aggregate result without making a
network call or consulting ambient state.
"""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

ACCESS_REVIEW_SCHEMA_VERSION: Final = 1
ACCESS_REVIEW_REPORT_TYPE: Final = "structured_privacy_access_review"

ACCESS_REVIEW_PASS: Final = "pass"
ACCESS_REVIEW_BLOCK: Final = "block"
ACCESS_REVIEW_DECISIONS: Final = (ACCESS_REVIEW_PASS, ACCESS_REVIEW_BLOCK)

REASON_NOT_YET_VALID: Final = "not_yet_valid"
REASON_EXPIRED: Final = "expired"
REASON_POLICY_FINGERPRINT_MISMATCH: Final = "policy_fingerprint_mismatch"
REASON_MISSING_DECISION_CATEGORIES: Final = "missing_decision_categories"
ACCESS_REVIEW_REASON_CODES: Final = (
    REASON_NOT_YET_VALID,
    REASON_EXPIRED,
    REASON_POLICY_FINGERPRINT_MISMATCH,
    REASON_MISSING_DECISION_CATEGORIES,
)

_UTC: Final = timezone.utc
_SAFE_CATEGORY = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_SAFE_FINGERPRINT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}\Z")
_REQUIRED_REVIEW_FIELDS: Final = frozenset(
    {"issued_at", "expires_at", "policy_fingerprint"}
)
_OPTIONAL_REVIEW_FIELDS: Final = frozenset({"decision_categories", "decisions"})

__all__ = [
    "ACCESS_REVIEW_BLOCK",
    "ACCESS_REVIEW_DECISIONS",
    "ACCESS_REVIEW_PASS",
    "ACCESS_REVIEW_REASON_CODES",
    "ACCESS_REVIEW_REPORT_TYPE",
    "ACCESS_REVIEW_SCHEMA_VERSION",
    "REASON_EXPIRED",
    "REASON_MISSING_DECISION_CATEGORIES",
    "REASON_NOT_YET_VALID",
    "REASON_POLICY_FINGERPRINT_MISMATCH",
    "AccessReview",
    "AccessReviewDecision",
    "AccessReviewEvaluation",
    "AccessReviewExpiryGate",
    "AccessReviewGate",
    "AccessReviewGateResult",
    "AccessReviewReport",
    "AccessReviewValidationError",
    "check_access_review_expiry",
    "enforce_access_review",
    "evaluate_access_review",
    "validate_access_review",
]


class AccessReviewValidationError(ValueError):
    """Raised when structured access-review metadata is invalid."""


def _safe_category(value: Any, *, field_name: str) -> str:
    """Validate one report-visible category without echoing caller input."""

    if not isinstance(value, str) or _SAFE_CATEGORY.fullmatch(value) is None:
        raise AccessReviewValidationError(
            f"{field_name} must be a safe decision-category identifier"
        )
    return value


def _safe_fingerprint(value: Any, *, field_name: str) -> str:
    """Validate an opaque, non-free-text policy fingerprint."""

    if not isinstance(value, str) or _SAFE_FINGERPRINT.fullmatch(value) is None:
        raise AccessReviewValidationError(
            f"{field_name} must be a non-empty policy fingerprint"
        )
    return value


def _timestamp(value: datetime | str, *, field_name: str) -> datetime:
    """Return a timezone-normalized timestamp without ambient clock access."""

    if isinstance(value, str):
        encoded = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            value = datetime.fromisoformat(encoded)
        except ValueError as exc:
            raise AccessReviewValidationError(
                f"{field_name} must be an ISO-8601 timestamp"
            ) from exc
    if not isinstance(value, datetime):
        raise AccessReviewValidationError(
            f"{field_name} must be a timezone-aware datetime"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise AccessReviewValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(_UTC)


def _format_timestamp(value: datetime) -> str:
    """Serialize a normalized timestamp using one canonical UTC spelling."""

    return value.isoformat().replace("+00:00", "Z")


def _category_values(
    value: Iterable[str] | Mapping[str, Any] | str | None,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Return sorted unique category identifiers and ignore mapping values."""

    if value is None:
        return ()
    if isinstance(value, str):
        candidates: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        # A mapping is accepted for callers that hold category -> decision
        # metadata. Decision values are deliberately never retained or read.
        candidates = value.keys()
    else:
        try:
            candidates = tuple(value)
        except TypeError as exc:
            raise AccessReviewValidationError(
                f"{field_name} must be an iterable of decision categories"
            ) from exc

    return tuple(
        sorted(
            {
                _safe_category(item, field_name=f"{field_name} entry")
                for item in candidates
            }
        )
    )


def _reason_values(value: Iterable[str] | str) -> tuple[str, ...]:
    """Return reason codes in the module's stable order."""

    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise AccessReviewValidationError(
                "reasons must be an iterable of access-review reason codes"
            ) from exc

    unknown = tuple(
        item
        for item in values
        if not isinstance(item, str) or item not in ACCESS_REVIEW_REASON_CODES
    )
    if unknown:
        raise AccessReviewValidationError("reasons contain an unsupported code")
    return tuple(code for code in ACCESS_REVIEW_REASON_CODES if code in set(values))


@dataclass(frozen=True, init=False)
class AccessReview:
    """PHI-safe metadata for one structured privacy access review.

    ``decision_categories`` may be an iterable of category identifiers or a
    mapping whose keys are category identifiers. Mapping values are ignored so
    decision notes or request content cannot enter the review record. The
    ``decisions`` keyword is accepted as a convenience alias for structured
    inputs that use a category-to-decision mapping.
    """

    issued_at: datetime
    expires_at: datetime
    policy_fingerprint: str
    decision_categories: tuple[str, ...]

    def __init__(
        self,
        issued_at: datetime | str,
        expires_at: datetime | str,
        policy_fingerprint: str,
        decision_categories: (Iterable[str] | Mapping[str, Any] | str | None) = (),
        *,
        decisions: Iterable[str] | Mapping[str, Any] | str | None = None,
    ) -> None:
        if decisions is not None and decision_categories not in ((), None):
            raise AccessReviewValidationError(
                "provide either decision_categories or decisions, not both"
            )
        categories = decision_categories if decisions is None else decisions
        normalized_issued_at = _timestamp(issued_at, field_name="issued_at")
        normalized_expires_at = _timestamp(expires_at, field_name="expires_at")
        if normalized_expires_at <= normalized_issued_at:
            raise AccessReviewValidationError("expires_at must be later than issued_at")
        normalized_fingerprint = _safe_fingerprint(
            policy_fingerprint,
            field_name="policy_fingerprint",
        )
        normalized_categories = _category_values(
            categories,
            field_name="decision_categories",
        )

        object.__setattr__(self, "issued_at", normalized_issued_at)
        object.__setattr__(self, "expires_at", normalized_expires_at)
        object.__setattr__(self, "policy_fingerprint", normalized_fingerprint)
        object.__setattr__(self, "decision_categories", normalized_categories)

    @property
    def issue_time(self) -> datetime:
        """Return the normalized review issue timestamp."""

        return self.issued_at

    @property
    def expiry_time(self) -> datetime:
        """Return the normalized review expiry timestamp."""

        return self.expires_at

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the normalized decision categories."""

        return self.decision_categories

    def to_dict(self) -> dict[str, Any]:
        """Return the allow-listed, JSON-compatible review metadata."""

        return {
            "issued_at": _format_timestamp(self.issued_at),
            "expires_at": _format_timestamp(self.expires_at),
            "policy_fingerprint": self.policy_fingerprint,
            "decision_categories": list(self.decision_categories),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize review metadata deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AccessReview:
        """Restore a review from the strict structured metadata shape."""

        if not isinstance(data, Mapping):
            raise AccessReviewValidationError("access review must be a mapping")
        keys = frozenset(data)
        allowed = _REQUIRED_REVIEW_FIELDS | _OPTIONAL_REVIEW_FIELDS
        if not _REQUIRED_REVIEW_FIELDS <= keys or not keys <= allowed:
            raise AccessReviewValidationError("access review fields are invalid")
        if "decision_categories" in data and "decisions" in data:
            raise AccessReviewValidationError(
                "access review contains duplicate decision-category fields"
            )
        categories = data.get("decision_categories", data.get("decisions", ()))
        return cls(
            data["issued_at"],
            data["expires_at"],
            data["policy_fingerprint"],
            categories,
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> AccessReview:
        """Restore review metadata from a JSON object without extra fields."""

        try:
            payload = json.loads(data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AccessReviewValidationError("access review JSON is invalid") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class AccessReviewGateResult:
    """Stable pass/block result containing aggregate review metadata only."""

    decision: str
    reasons: tuple[str, ...]
    checked_at: datetime
    review_issued_at: datetime
    review_expires_at: datetime
    policy_fingerprint_matches: bool
    required_decision_categories: tuple[str, ...]
    present_decision_categories: tuple[str, ...]
    missing_decision_categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.decision not in ACCESS_REVIEW_DECISIONS:
            raise AccessReviewValidationError("decision must be pass or block")
        reasons = _reason_values(self.reasons)
        if self.decision == ACCESS_REVIEW_PASS and reasons:
            raise AccessReviewValidationError("pass decisions cannot contain reasons")
        if self.decision == ACCESS_REVIEW_BLOCK and not reasons:
            raise AccessReviewValidationError("block decisions require a reason")
        if type(self.policy_fingerprint_matches) is not bool:
            raise AccessReviewValidationError(
                "policy_fingerprint_matches must be a boolean"
            )
        required = _category_values(
            self.required_decision_categories,
            field_name="required_decision_categories",
        )
        present = _category_values(
            self.present_decision_categories,
            field_name="present_decision_categories",
        )
        missing = _category_values(
            self.missing_decision_categories,
            field_name="missing_decision_categories",
        )
        expected_missing = tuple(sorted(set(required) - set(present)))
        if missing != expected_missing:
            raise AccessReviewValidationError(
                "missing decision categories do not match the review"
            )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self, "checked_at", _timestamp(self.checked_at, field_name="checked_at")
        )
        object.__setattr__(
            self,
            "review_issued_at",
            _timestamp(self.review_issued_at, field_name="review_issued_at"),
        )
        object.__setattr__(
            self,
            "review_expires_at",
            _timestamp(self.review_expires_at, field_name="review_expires_at"),
        )
        object.__setattr__(self, "required_decision_categories", required)
        object.__setattr__(self, "present_decision_categories", present)
        object.__setattr__(self, "missing_decision_categories", missing)

    @property
    def passed(self) -> bool:
        """Return whether the gate passed."""

        return self.decision == ACCESS_REVIEW_PASS

    @property
    def allowed(self) -> bool:
        """Return the explicit access-allowance interpretation of the result."""

        return self.passed

    @property
    def blocked(self) -> bool:
        """Return whether the gate blocked the review."""

        return not self.passed

    @property
    def status(self) -> str:
        """Return the canonical ``pass`` or ``block`` status."""

        return self.decision

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return stable machine-readable block reasons."""

        return self.reasons

    @property
    def missing_categories(self) -> tuple[str, ...]:
        """Return required categories absent from the review."""

        return self.missing_decision_categories

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without identities or request data."""

        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "checked_at": _format_timestamp(self.checked_at),
            "review_issued_at": _format_timestamp(self.review_issued_at),
            "review_expires_at": _format_timestamp(self.review_expires_at),
            "policy_fingerprint_matches": self.policy_fingerprint_matches,
            "required_decision_categories": list(self.required_decision_categories),
            "present_decision_categories": list(self.present_decision_categories),
            "missing_decision_categories": list(self.missing_decision_categories),
            "passed": self.passed,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the gate result deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )


def _resolve_clock(
    as_of: datetime | str | None,
    now: datetime | str | None,
) -> datetime:
    if as_of is not None and now is not None:
        raise AccessReviewValidationError("provide one supplied clock as as_of or now")
    if as_of is None and now is None:
        raise AccessReviewValidationError("a supplied clock is required")
    return _timestamp(as_of if as_of is not None else now, field_name="as_of")


def _evaluate_access_review(
    review: AccessReview | Mapping[str, Any],
    *,
    expected_policy_fingerprint: str,
    required_decision_categories: (Iterable[str] | Mapping[str, Any] | str | None),
    as_of: datetime | str | None,
    now: datetime | str | None,
) -> AccessReviewGateResult:
    if isinstance(review, Mapping):
        review = AccessReview.from_dict(review)
    if not isinstance(review, AccessReview):
        raise AccessReviewValidationError("review must be an AccessReview")
    checked_at = _resolve_clock(as_of, now)
    expected_fingerprint = _safe_fingerprint(
        expected_policy_fingerprint,
        field_name="expected_policy_fingerprint",
    )
    required = _category_values(
        required_decision_categories,
        field_name="required_decision_categories",
    )
    present = review.decision_categories
    missing = tuple(sorted(set(required) - set(present)))
    fingerprint_matches = hmac.compare_digest(
        review.policy_fingerprint,
        expected_fingerprint,
    )

    reasons: list[str] = []
    if checked_at < review.issued_at:
        reasons.append(REASON_NOT_YET_VALID)
    elif checked_at >= review.expires_at:
        reasons.append(REASON_EXPIRED)
    if not fingerprint_matches:
        reasons.append(REASON_POLICY_FINGERPRINT_MISMATCH)
    if missing:
        reasons.append(REASON_MISSING_DECISION_CATEGORIES)

    decision = ACCESS_REVIEW_PASS if not reasons else ACCESS_REVIEW_BLOCK
    return AccessReviewGateResult(
        decision=decision,
        reasons=tuple(reasons),
        checked_at=checked_at,
        review_issued_at=review.issued_at,
        review_expires_at=review.expires_at,
        policy_fingerprint_matches=fingerprint_matches,
        required_decision_categories=required,
        present_decision_categories=present,
        missing_decision_categories=missing,
    )


def evaluate_access_review(
    review: AccessReview | Mapping[str, Any],
    *,
    expected_policy_fingerprint: str,
    required_decision_categories: (Iterable[str] | Mapping[str, Any] | str | None) = (),
    as_of: datetime | str | None = None,
    now: datetime | str | None = None,
) -> AccessReviewGateResult:
    """Evaluate a review against an explicit clock and policy requirements.

    The expiry boundary is exclusive: a review passes when ``as_of`` equals
    ``issued_at`` and blocks when ``as_of`` equals ``expires_at``. A future
    issue time, expired window, policy mismatch, or missing required category
    contributes a deterministic reason code. No current time is read when the
    caller omits the clock; omission is an input error.
    """

    return _evaluate_access_review(
        review,
        expected_policy_fingerprint=expected_policy_fingerprint,
        required_decision_categories=required_decision_categories,
        as_of=as_of,
        now=now,
    )


@dataclass(frozen=True)
class AccessReviewExpiryGate:
    """Reusable local gate configuration for structured access reviews."""

    expected_policy_fingerprint: str
    required_decision_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_policy_fingerprint",
            _safe_fingerprint(
                self.expected_policy_fingerprint,
                field_name="expected_policy_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "required_decision_categories",
            _category_values(
                self.required_decision_categories,
                field_name="required_decision_categories",
            ),
        )

    def evaluate(
        self,
        review: AccessReview | Mapping[str, Any],
        *,
        as_of: datetime | str | None = None,
        now: datetime | str | None = None,
    ) -> AccessReviewGateResult:
        """Evaluate ``review`` using this gate's fixed policy requirements."""

        return _evaluate_access_review(
            review,
            expected_policy_fingerprint=self.expected_policy_fingerprint,
            required_decision_categories=self.required_decision_categories,
            as_of=as_of,
            now=now,
        )

    check = evaluate


# Descriptive aliases keep the result discoverable for callers using either
# "evaluation" or "decision" terminology without adding another output shape.
AccessReviewDecision = AccessReviewGateResult
AccessReviewEvaluation = AccessReviewGateResult
AccessReviewGate = AccessReviewExpiryGate
AccessReviewReport = AccessReview

check_access_review_expiry = evaluate_access_review
enforce_access_review = evaluate_access_review
validate_access_review = evaluate_access_review
