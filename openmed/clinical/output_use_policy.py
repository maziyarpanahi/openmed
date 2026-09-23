"""Deterministic, privacy-safe policy gates for clinical-output uses.

The gate evaluates output metadata only. Callers declare an output category,
purpose, audience, review state, decision-triggering status, and the policy
fingerprint they used. The gate never accepts an output payload, calls a
network service, or includes caller-supplied values in a denial. Denials contain
stable reason codes so an audit consumer can act without receiving raw
clinical content.

This module is a workflow guard, not a compliance certification or a clinical
decision. An allowed result means that the declared use satisfies this local
policy and review state; it does not validate the clinical content itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from openmed.core.audit import stable_hash

OUTPUT_USE_POLICY_SCHEMA_VERSION: Final = 1
OUTPUT_USE_POLICY_NAME: Final = "openmed-clinical-output-use-v1"
_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class OutputCategory(str, Enum):
    """Supported categories of clinical output metadata."""

    SUMMARY = "summary"
    EXTRACTION = "extraction"
    ANNOTATION = "annotation"
    DECISION_SUPPORT = "decision_support"
    RECOMMENDATION = "recommendation"
    ACTION = "action"


class OutputPurpose(str, Enum):
    """Supported declared purposes for a clinical output."""

    DOCUMENTATION = "documentation"
    REVIEW = "review"
    RESEARCH = "research"
    QUALITY_ASSURANCE = "quality_assurance"
    CARE_COORDINATION = "care_coordination"
    PATIENT_COMMUNICATION = "patient_communication"
    CLINICAL_DECISION = "clinical_decision"


class OutputAudience(str, Enum):
    """Supported audiences for a clinical output."""

    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    QUALITY_TEAM = "quality_team"
    PATIENT = "patient"
    SYSTEM = "system"
    PUBLIC = "public"


class ReviewState(str, Enum):
    """Review states understood by the output-use policy."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class OutputUseReasonCode(str, Enum):
    """Stable, payload-free reason codes emitted by the gate."""

    DECLARATION_INVALID = "declaration_invalid"
    DECLARATION_AMBIGUOUS = "declaration_ambiguous"
    CATEGORY_UNDECLARED = "category_undeclared"
    CATEGORY_INVALID = "category_invalid"
    CATEGORY_UNSUPPORTED = "category_unsupported"
    PURPOSE_UNDECLARED = "purpose_undeclared"
    PURPOSE_INVALID = "purpose_invalid"
    PURPOSE_UNSUPPORTED = "purpose_unsupported"
    AUDIENCE_UNDECLARED = "audience_undeclared"
    AUDIENCE_INVALID = "audience_invalid"
    AUDIENCE_UNSUPPORTED = "audience_unsupported"
    REVIEW_STATE_UNDECLARED = "review_state_undeclared"
    REVIEW_STATE_INVALID = "review_state_invalid"
    REVIEW_STATE_UNSUPPORTED = "review_state_unsupported"
    DECISION_TRIGGERING_UNDECLARED = "decision_triggering_undeclared"
    DECISION_TRIGGERING_INVALID = "decision_triggering_invalid"
    DECISION_TRIGGERING_USE = "decision_triggering_use"
    POLICY_FINGERPRINT_UNDECLARED = "policy_fingerprint_undeclared"
    POLICY_FINGERPRINT_INVALID = "policy_fingerprint_invalid"
    POLICY_FINGERPRINT_MISMATCH = "policy_fingerprint_mismatch"
    REVIEW_REJECTED = "review_rejected"
    REVIEW_STATE_INSUFFICIENT = "review_state_insufficient"
    INCOMPATIBLE_USE = "incompatible_use"
    POLICY_INVALID = "policy_invalid"


_REASON_ORDER: Final = tuple(item.value for item in OutputUseReasonCode)
_REASON_ORDER_INDEX: Final = {code: index for index, code in enumerate(_REASON_ORDER)}


class OutputUsePolicyError(ValueError):
    """Payload-free policy error carrying only stable reason codes."""

    def __init__(self, reason_codes: str | Sequence[str] | Any) -> None:
        if hasattr(reason_codes, "reason_codes"):
            reason_codes = getattr(reason_codes, "reason_codes")
        if isinstance(reason_codes, str):
            codes = (reason_codes,)
        else:
            try:
                codes = tuple(reason_codes)
            except TypeError:
                codes = ()

        normalized = _ordered_reason_codes(codes)
        if not normalized:
            normalized = (OutputUseReasonCode.POLICY_INVALID.value,)
        self.reason_codes = normalized
        super().__init__("output use policy error: " + ", ".join(normalized))


_CATEGORY_ALIASES: Final = {
    "summary": OutputCategory.SUMMARY.value,
    "clinical_summary": OutputCategory.SUMMARY.value,
    "extraction": OutputCategory.EXTRACTION.value,
    "structured_extraction": OutputCategory.EXTRACTION.value,
    "clinical_extraction": OutputCategory.EXTRACTION.value,
    "annotation": OutputCategory.ANNOTATION.value,
    "review_annotation": OutputCategory.ANNOTATION.value,
    "decision_support": OutputCategory.DECISION_SUPPORT.value,
    "clinical_decision_support": OutputCategory.DECISION_SUPPORT.value,
    "recommendation": OutputCategory.RECOMMENDATION.value,
    "clinical_recommendation": OutputCategory.RECOMMENDATION.value,
    "action": OutputCategory.ACTION.value,
    "clinical_action": OutputCategory.ACTION.value,
    "decision": OutputCategory.ACTION.value,
    "clinical_decision": OutputCategory.ACTION.value,
}
_PURPOSE_ALIASES: Final = {
    "documentation": OutputPurpose.DOCUMENTATION.value,
    "clinical_documentation": OutputPurpose.DOCUMENTATION.value,
    "chart_documentation": OutputPurpose.DOCUMENTATION.value,
    "review": OutputPurpose.REVIEW.value,
    "human_review": OutputPurpose.REVIEW.value,
    "research": OutputPurpose.RESEARCH.value,
    "quality": OutputPurpose.QUALITY_ASSURANCE.value,
    "qa": OutputPurpose.QUALITY_ASSURANCE.value,
    "quality_review": OutputPurpose.QUALITY_ASSURANCE.value,
    "quality_assurance": OutputPurpose.QUALITY_ASSURANCE.value,
    "care_coordination": OutputPurpose.CARE_COORDINATION.value,
    "patient_communication": OutputPurpose.PATIENT_COMMUNICATION.value,
    "patient_facing": OutputPurpose.PATIENT_COMMUNICATION.value,
    "clinical_decision": OutputPurpose.CLINICAL_DECISION.value,
    "clinical_decision_support": OutputPurpose.CLINICAL_DECISION.value,
    "treatment_decision": OutputPurpose.CLINICAL_DECISION.value,
    "triage": OutputPurpose.CLINICAL_DECISION.value,
}
_AUDIENCE_ALIASES: Final = {
    "clinician": OutputAudience.CLINICIAN.value,
    "clinical": OutputAudience.CLINICIAN.value,
    "clinical_reviewer": OutputAudience.CLINICIAN.value,
    "care_team": OutputAudience.CLINICIAN.value,
    "researcher": OutputAudience.RESEARCHER.value,
    "research": OutputAudience.RESEARCHER.value,
    "quality_team": OutputAudience.QUALITY_TEAM.value,
    "quality": OutputAudience.QUALITY_TEAM.value,
    "qa": OutputAudience.QUALITY_TEAM.value,
    "patient": OutputAudience.PATIENT.value,
    "system": OutputAudience.SYSTEM.value,
    "machine": OutputAudience.SYSTEM.value,
    "public": OutputAudience.PUBLIC.value,
}
_REVIEW_STATE_ALIASES: Final = {
    "draft": ReviewState.DRAFT.value,
    "unreviewed": ReviewState.DRAFT.value,
    "pending_review": ReviewState.PENDING_REVIEW.value,
    "pending": ReviewState.PENDING_REVIEW.value,
    "in_review": ReviewState.PENDING_REVIEW.value,
    "reviewed": ReviewState.REVIEWED.value,
    "human_reviewed": ReviewState.REVIEWED.value,
    "approved": ReviewState.APPROVED.value,
    "rejected": ReviewState.REJECTED.value,
}
_REVIEW_RANKS: Final = {
    ReviewState.DRAFT.value: 0,
    ReviewState.PENDING_REVIEW.value: 1,
    ReviewState.REVIEWED.value: 2,
    ReviewState.APPROVED.value: 3,
}
_DECISION_TRIGGERING_PURPOSES: Final = frozenset(
    {OutputPurpose.CLINICAL_DECISION.value}
)
_DECISION_TRIGGERING_CATEGORIES: Final = frozenset({OutputCategory.ACTION.value})


def _ordered_reason_codes(codes: Sequence[Any]) -> tuple[str, ...]:
    """Return known reason codes in their stable canonical order."""

    normalized: set[str] = set()
    for code in codes:
        value = code.value if isinstance(code, OutputUseReasonCode) else code
        if isinstance(value, str) and value in _REASON_ORDER_INDEX:
            normalized.add(value)
    return tuple(sorted(normalized, key=_REASON_ORDER_INDEX.__getitem__))


def _token(value: Any) -> str | None:
    """Normalize a caller token without retaining the original value."""

    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold())
    return normalized.strip("_") or None


def _canonical_value(value: Any, aliases: Mapping[str, str]) -> str | None:
    token = _token(value)
    return aliases.get(token) if token is not None else None


def _safe_fingerprint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().casefold()
    return candidate if _FINGERPRINT_RE.fullmatch(candidate) else None


def _invalid_policy() -> OutputUsePolicyError:
    return OutputUsePolicyError(OutputUseReasonCode.POLICY_INVALID.value)


@dataclass(frozen=True, repr=False)
class OutputUseRule:
    """One compatible category, purpose, audience, and review requirement."""

    category: str | OutputCategory
    purpose: str | OutputPurpose
    audience: str | OutputAudience
    minimum_review_state: str | ReviewState = ReviewState.REVIEWED.value
    decision_triggering: bool = False

    def __post_init__(self) -> None:
        category = _canonical_value(self.category, _CATEGORY_ALIASES)
        purpose = _canonical_value(self.purpose, _PURPOSE_ALIASES)
        audience = _canonical_value(self.audience, _AUDIENCE_ALIASES)
        review_state = _canonical_value(
            self.minimum_review_state,
            _REVIEW_STATE_ALIASES,
        )
        if not all((category, purpose, audience, review_state)):
            raise _invalid_policy()
        if review_state == ReviewState.REJECTED.value:
            raise _invalid_policy()
        if type(self.decision_triggering) is not bool or self.decision_triggering:
            raise _invalid_policy()
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "audience", audience)
        object.__setattr__(self, "minimum_review_state", review_state)

    def key(self) -> tuple[str, str, str, str, bool]:
        """Return the deterministic identity of this rule."""

        return (
            self.category,
            self.purpose,
            self.audience,
            self.minimum_review_state,
            self.decision_triggering,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible rule representation."""

        return {
            "category": self.category,
            "purpose": self.purpose,
            "audience": self.audience,
            "minimum_review_state": self.minimum_review_state,
            "decision_triggering": self.decision_triggering,
        }

    def __repr__(self) -> str:
        return (
            "OutputUseRule("
            f"category={self.category!r}, purpose={self.purpose!r}, "
            f"audience={self.audience!r}, "
            f"minimum_review_state={self.minimum_review_state!r})"
        )


@dataclass(frozen=True, repr=False)
class OutputUsePolicy:
    """Immutable local policy defining compatible clinical-output uses."""

    name: str = OUTPUT_USE_POLICY_NAME
    rules: tuple[OutputUseRule, ...] = ()
    schema_version: int = OUTPUT_USE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = _token(self.name)
        if (
            name is None
            or type(self.schema_version) is not int
            or self.schema_version < 1
        ):
            raise _invalid_policy()
        try:
            rules = tuple(self.rules)
        except TypeError as exc:
            raise _invalid_policy() from exc
        if not rules or not all(isinstance(rule, OutputUseRule) for rule in rules):
            raise _invalid_policy()
        unique_rules = {rule.key(): rule for rule in rules}
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "rules",
            tuple(unique_rules[key] for key in sorted(unique_rules)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible policy representation."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @property
    def fingerprint(self) -> str:
        """Return the stable SHA-256 fingerprint of this policy."""

        return stable_hash(
            {"kind": "openmed-clinical-output-use-policy", **self.to_dict()}
        )

    def matching_rules(
        self,
        category: str,
        purpose: str,
        audience: str,
    ) -> tuple[OutputUseRule, ...]:
        """Return rules matching the declaration apart from review state."""

        return tuple(
            rule
            for rule in self.rules
            if (
                rule.category == category
                and rule.purpose == purpose
                and rule.audience == audience
            )
        )

    def __repr__(self) -> str:
        return (
            f"OutputUsePolicy(name={self.name!r}, rules={len(self.rules)}, "
            f"fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True, repr=False)
class OutputUseDeclaration:
    """Metadata declaration evaluated by :func:`evaluate_output_use`.

    The fields intentionally describe use rather than content. The custom
    representation and safe serialization do not echo unknown caller values,
    which keeps accidental logging of a declaration payload-free.
    """

    category: Any = None
    purpose: Any = None
    audience: Any = None
    review_state: Any = None
    decision_triggering: Any = None
    policy_fingerprint: Any = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OutputUseDeclaration":
        """Build a declaration from a mapping using stable field aliases."""

        if not isinstance(payload, Mapping):
            raise OutputUsePolicyError(OutputUseReasonCode.DECLARATION_INVALID.value)
        return cls(
            category=_first_present(payload, "category", "output_category"),
            purpose=_first_present(payload, "purpose", "declared_purpose"),
            audience=_first_present(payload, "audience", "intended_audience"),
            review_state=_first_present(payload, "review_state", "review"),
            decision_triggering=_first_present(
                payload,
                "decision_triggering",
                "decision_trigger",
            ),
            policy_fingerprint=_first_present(
                payload,
                "policy_fingerprint",
                "policy_digest",
            ),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return declaration metadata without unknown caller values."""

        return {
            "category": _canonical_value(self.category, _CATEGORY_ALIASES),
            "purpose": _canonical_value(self.purpose, _PURPOSE_ALIASES),
            "audience": _canonical_value(self.audience, _AUDIENCE_ALIASES),
            "review_state": _canonical_value(
                self.review_state,
                _REVIEW_STATE_ALIASES,
            ),
            "decision_triggering": (
                self.decision_triggering
                if type(self.decision_triggering) is bool
                else None
            ),
            "policy_fingerprint": _safe_fingerprint(self.policy_fingerprint),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for the payload-free :meth:`to_safe_dict` representation."""

        return self.to_safe_dict()

    def __repr__(self) -> str:
        return "OutputUseDeclaration(metadata_only=True)"


@dataclass(frozen=True, repr=False)
class OutputUseDecision:
    """Payload-free result of evaluating one output-use declaration."""

    allowed: bool
    reason_codes: tuple[str, ...]
    policy_fingerprint: str
    schema_version: int = OUTPUT_USE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise _invalid_policy()
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise _invalid_policy()
        reason_codes = _ordered_reason_codes(self.reason_codes)
        if not self.allowed and not reason_codes:
            raise _invalid_policy()
        if not self.allowed:
            object.__setattr__(self, "reason_codes", reason_codes)
        else:
            object.__setattr__(self, "reason_codes", ())
        if not _safe_fingerprint(self.policy_fingerprint):
            raise _invalid_policy()

    @property
    def reason_code(self) -> str | None:
        """Return the first stable reason code, if the use was denied."""

        return self.reason_codes[0] if self.reason_codes else None

    @property
    def is_allowed(self) -> bool:
        """Return whether the declared use passed the policy gate."""

        return self.allowed

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, payload-free decision report."""

        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "policy_fingerprint": self.policy_fingerprint,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Return the same payload-free report for safe audit integration."""

        return self.to_dict()

    def __repr__(self) -> str:
        return (
            f"OutputUseDecision(allowed={self.allowed!r}, "
            f"reason_codes={self.reason_codes!r}, "
            f"policy_fingerprint={self.policy_fingerprint!r})"
        )


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _field_value(
    value: Any,
    aliases: Mapping[str, str],
    *,
    undeclared: OutputUseReasonCode,
    invalid: OutputUseReasonCode,
    unsupported: OutputUseReasonCode,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, undeclared.value
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None, invalid.value
    if _token(value) is None:
        return None, undeclared.value
    canonical = _canonical_value(value, aliases)
    if canonical is None:
        return None, unsupported.value
    return canonical, None


def _declaration_from_input(
    declaration: OutputUseDeclaration | Mapping[str, Any] | None,
    *,
    category: Any,
    purpose: Any,
    audience: Any,
    review_state: Any,
    decision_triggering: Any,
    policy_fingerprint: Any,
) -> tuple[OutputUseDeclaration | None, tuple[str, ...]]:
    keyword_values = (
        category,
        purpose,
        audience,
        review_state,
        decision_triggering,
        policy_fingerprint,
    )
    if declaration is not None and any(value is not None for value in keyword_values):
        return None, (OutputUseReasonCode.DECLARATION_AMBIGUOUS.value,)
    if declaration is None:
        return (
            OutputUseDeclaration(
                category=category,
                purpose=purpose,
                audience=audience,
                review_state=review_state,
                decision_triggering=decision_triggering,
                policy_fingerprint=policy_fingerprint,
            ),
            (),
        )
    if isinstance(declaration, OutputUseDeclaration):
        return declaration, ()
    if isinstance(declaration, Mapping):
        return OutputUseDeclaration.from_mapping(declaration), ()
    return None, (OutputUseReasonCode.DECLARATION_INVALID.value,)


def evaluate_output_use(
    declaration: OutputUseDeclaration | Mapping[str, Any] | None = None,
    *,
    category: Any = None,
    purpose: Any = None,
    audience: Any = None,
    review_state: Any = None,
    decision_triggering: Any = None,
    policy_fingerprint: Any = None,
    policy: OutputUsePolicy | None = None,
) -> OutputUseDecision:
    """Evaluate a declared clinical-output use against a local policy.

    Args:
        declaration: A metadata declaration or mapping. When omitted, the
            keyword fields form the declaration.
        category: Output category, such as ``"summary"`` or ``"extraction"``.
        purpose: Declared purpose, such as ``"documentation"`` or ``"review"``.
        audience: Intended audience, such as ``"clinician"`` or ``"researcher"``.
        review_state: Explicit review state required before release.
        decision_triggering: Must be explicitly ``False`` for a permitted use.
        policy_fingerprint: Fingerprint of the policy used by the caller.
        policy: Immutable local policy to evaluate against.

    Returns:
        A deterministic, payload-free :class:`OutputUseDecision`. Missing,
        unsupported, incompatible, or unsafe declarations are denied.

    Raises:
        OutputUsePolicyError: If ``policy`` is not a valid local policy.
    """

    if policy is None:
        policy = DEFAULT_OUTPUT_USE_POLICY
    if not isinstance(policy, OutputUsePolicy):
        raise _invalid_policy()
    declaration_value, declaration_errors = _declaration_from_input(
        declaration,
        category=category,
        purpose=purpose,
        audience=audience,
        review_state=review_state,
        decision_triggering=decision_triggering,
        policy_fingerprint=policy_fingerprint,
    )
    if declaration_value is None:
        return OutputUseDecision(
            allowed=False,
            reason_codes=declaration_errors,
            policy_fingerprint=policy.fingerprint,
        )

    reason_codes: list[str] = []
    category_value, category_error = _field_value(
        declaration_value.category,
        _CATEGORY_ALIASES,
        undeclared=OutputUseReasonCode.CATEGORY_UNDECLARED,
        invalid=OutputUseReasonCode.CATEGORY_INVALID,
        unsupported=OutputUseReasonCode.CATEGORY_UNSUPPORTED,
    )
    purpose_value, purpose_error = _field_value(
        declaration_value.purpose,
        _PURPOSE_ALIASES,
        undeclared=OutputUseReasonCode.PURPOSE_UNDECLARED,
        invalid=OutputUseReasonCode.PURPOSE_INVALID,
        unsupported=OutputUseReasonCode.PURPOSE_UNSUPPORTED,
    )
    audience_value, audience_error = _field_value(
        declaration_value.audience,
        _AUDIENCE_ALIASES,
        undeclared=OutputUseReasonCode.AUDIENCE_UNDECLARED,
        invalid=OutputUseReasonCode.AUDIENCE_INVALID,
        unsupported=OutputUseReasonCode.AUDIENCE_UNSUPPORTED,
    )
    review_value, review_error = _field_value(
        declaration_value.review_state,
        _REVIEW_STATE_ALIASES,
        undeclared=OutputUseReasonCode.REVIEW_STATE_UNDECLARED,
        invalid=OutputUseReasonCode.REVIEW_STATE_INVALID,
        unsupported=OutputUseReasonCode.REVIEW_STATE_UNSUPPORTED,
    )
    reason_codes.extend(
        error
        for error in (
            category_error,
            purpose_error,
            audience_error,
            review_error,
        )
        if error is not None
    )

    fingerprint_value = _safe_fingerprint(declaration_value.policy_fingerprint)
    if declaration_value.policy_fingerprint is None:
        reason_codes.append(OutputUseReasonCode.POLICY_FINGERPRINT_UNDECLARED.value)
    elif fingerprint_value is None:
        reason_codes.append(OutputUseReasonCode.POLICY_FINGERPRINT_INVALID.value)
    elif fingerprint_value != policy.fingerprint:
        reason_codes.append(OutputUseReasonCode.POLICY_FINGERPRINT_MISMATCH.value)

    trigger_value = declaration_value.decision_triggering
    if trigger_value is None:
        reason_codes.append(OutputUseReasonCode.DECISION_TRIGGERING_UNDECLARED.value)
    elif type(trigger_value) is not bool:
        reason_codes.append(OutputUseReasonCode.DECISION_TRIGGERING_INVALID.value)
    elif trigger_value or (
        category_value in _DECISION_TRIGGERING_CATEGORIES
        or purpose_value in _DECISION_TRIGGERING_PURPOSES
    ):
        reason_codes.append(OutputUseReasonCode.DECISION_TRIGGERING_USE.value)

    if review_value == ReviewState.REJECTED.value:
        reason_codes.append(OutputUseReasonCode.REVIEW_REJECTED.value)

    if all(
        value is not None
        for value in (category_value, purpose_value, audience_value, review_value)
    ):
        matching_rules = policy.matching_rules(
            category_value,
            purpose_value,
            audience_value,
        )
        if not matching_rules:
            reason_codes.append(OutputUseReasonCode.INCOMPATIBLE_USE.value)
        elif review_value != ReviewState.REJECTED.value:
            review_rank = _REVIEW_RANKS[review_value]
            if not any(
                review_rank >= _REVIEW_RANKS[rule.minimum_review_state]
                for rule in matching_rules
            ):
                reason_codes.append(OutputUseReasonCode.REVIEW_STATE_INSUFFICIENT.value)

    ordered_reasons = _ordered_reason_codes(reason_codes)
    return OutputUseDecision(
        allowed=not ordered_reasons,
        reason_codes=ordered_reasons,
        policy_fingerprint=policy.fingerprint,
    )


def enforce_output_use(
    declaration: OutputUseDeclaration | Mapping[str, Any] | None = None,
    *,
    category: Any = None,
    purpose: Any = None,
    audience: Any = None,
    review_state: Any = None,
    decision_triggering: Any = None,
    policy_fingerprint: Any = None,
    policy: OutputUsePolicy | None = None,
) -> OutputUseDecision:
    """Raise a payload-free error unless the declared use is permitted."""

    decision = evaluate_output_use(
        declaration,
        category=category,
        purpose=purpose,
        audience=audience,
        review_state=review_state,
        decision_triggering=decision_triggering,
        policy_fingerprint=policy_fingerprint,
        policy=policy,
    )
    if not decision.allowed:
        raise OutputUsePolicyError(decision.reason_codes)
    return decision


def policy_fingerprint(policy: OutputUsePolicy) -> str:
    """Return the deterministic fingerprint for ``policy``."""

    if not isinstance(policy, OutputUsePolicy):
        raise _invalid_policy()
    return policy.fingerprint


_DEFAULT_OUTPUT_USE_RULES: Final = (
    OutputUseRule("annotation", "quality_assurance", "quality_team", "reviewed"),
    OutputUseRule("annotation", "review", "clinician", "reviewed"),
    OutputUseRule("extraction", "care_coordination", "clinician", "approved"),
    OutputUseRule("extraction", "documentation", "clinician", "reviewed"),
    OutputUseRule("extraction", "quality_assurance", "quality_team", "reviewed"),
    OutputUseRule("extraction", "research", "researcher", "approved"),
    OutputUseRule("recommendation", "review", "clinician", "approved"),
    OutputUseRule("summary", "care_coordination", "clinician", "approved"),
    OutputUseRule("summary", "documentation", "clinician", "reviewed"),
    OutputUseRule("summary", "patient_communication", "patient", "approved"),
    OutputUseRule("summary", "quality_assurance", "quality_team", "reviewed"),
    OutputUseRule("summary", "research", "researcher", "approved"),
    OutputUseRule("decision_support", "review", "clinician", "approved"),
)

DEFAULT_OUTPUT_USE_POLICY: Final = OutputUsePolicy(
    name=OUTPUT_USE_POLICY_NAME,
    rules=_DEFAULT_OUTPUT_USE_RULES,
)
DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT: Final = DEFAULT_OUTPUT_USE_POLICY.fingerprint
DEFAULT_POLICY_FINGERPRINT: Final = DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT


def evaluate_output_use_policy(
    declaration: OutputUseDeclaration | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> OutputUseDecision:
    """Compatibility name for :func:`evaluate_output_use`."""

    return evaluate_output_use(declaration, **kwargs)


def check_output_use_policy(
    declaration: OutputUseDeclaration | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> OutputUseDecision:
    """Return the policy decision for a declared output use."""

    return evaluate_output_use(declaration, **kwargs)


__all__ = [
    "DEFAULT_OUTPUT_USE_POLICY",
    "DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT",
    "DEFAULT_POLICY_FINGERPRINT",
    "OUTPUT_USE_POLICY_NAME",
    "OUTPUT_USE_POLICY_SCHEMA_VERSION",
    "OutputAudience",
    "OutputCategory",
    "OutputPurpose",
    "OutputUseDeclaration",
    "OutputUseDecision",
    "OutputUsePolicy",
    "OutputUsePolicyError",
    "OutputUseReasonCode",
    "OutputUseRule",
    "ReviewState",
    "check_output_use_policy",
    "enforce_output_use",
    "evaluate_output_use",
    "evaluate_output_use_policy",
    "policy_fingerprint",
]
