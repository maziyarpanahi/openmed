"""Deterministic token budgets for local clinical-summary generation.

This module is the metadata-only boundary between policy-approved evidence and
a local summary generator.  It allocates a caller-supplied maximum token
budget across a closed set of clinical evidence classes and reports which
classes could not fit in full.  The generator can use each allocation as its
class-specific output cap before it reads source content.

Only non-sensitive evidence-class identifiers and token counts are accepted.
The module never accepts or stores source text, extracted values, prompts, or
model output.  It uses only the Python standard library, performs no model
loading, and makes no network calls.  The result is an assistive planning
artifact and requires qualified clinical review; it is not a clinical decision
or a guarantee of summary completeness.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION: Final[int] = 1
SUMMARY_LENGTH_BUDGET_POLICY_ID: Final[str] = "clinical_summary_v1"
SUMMARY_LENGTH_BUDGET_DISCLAIMER: Final[str] = (
    "Clinical summary length budgets are deterministic assistive review "
    "artifacts, not clinical decisions or autonomous summaries. Qualified "
    "clinical review is required."
)

MAX_SUMMARY_LENGTH_TOKENS: Final[int] = 10_000_000
MAX_SUMMARY_EVIDENCE_CLASSES: Final[int] = 64
MAX_SUMMARY_EVIDENCE_CLASS_ID_LENGTH: Final[int] = 64
MAX_SUMMARY_REQUESTED_TOKENS: Final[int] = (
    MAX_SUMMARY_LENGTH_TOKENS * MAX_SUMMARY_EVIDENCE_CLASSES
)

_IDENTIFIER_RE = re.compile(
    rf"^[a-z][a-z0-9_]{{0,{MAX_SUMMARY_EVIDENCE_CLASS_ID_LENGTH - 1}}}$"
)
_POLICY_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MISSING = object()


class SummaryLengthBudgetReason(str, Enum):
    """Fixed reason codes for safely rejected budget inputs."""

    INVALID_INPUT = "invalid_input"
    INVALID_MAX_TOKENS = "invalid_max_tokens"
    INVALID_POLICY = "invalid_policy"
    INVALID_POLICY_CLASS = "invalid_policy_class"
    INVALID_EVIDENCE = "invalid_evidence"
    INVALID_EVIDENCE_CLASS = "invalid_evidence_class"
    UNKNOWN_EVIDENCE_CLASS = "unknown_evidence_class"
    UNAPPROVED_EVIDENCE_CLASS = "unapproved_evidence_class"
    INVALID_TOKEN_COUNT = "invalid_token_count"
    UNSUPPORTED_SCHEMA = "unsupported_schema"


class SummaryLengthBudgetError(ValueError):
    """Raised when a summary length budget cannot be built safely.

    The exception message contains only a fixed reason code and never echoes a
    submitted identifier, source value, parser message, or other input.
    """

    def __init__(
        self,
        reason: SummaryLengthBudgetReason,
        *,
        rejected_count: int = 1,
    ) -> None:
        self.reason = _coerce_reason(reason)
        self.rejected_count = _safe_count(
            rejected_count,
            SummaryLengthBudgetReason.INVALID_INPUT,
        )
        super().__init__(f"summary_length_budget_{self.reason.value}")

    @property
    def reason_code(self) -> str:
        """Return the stable machine-readable refusal code."""

        return self.reason.value


def _identifier(value: object) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_EVIDENCE_CLASS)
    return value


def _safe_count(value: object, reason: SummaryLengthBudgetReason) -> int:
    if type(value) is not int or value < 0 or value > MAX_SUMMARY_LENGTH_TOKENS:
        raise SummaryLengthBudgetError(reason)
    return value


def _positive_count(value: object, reason: SummaryLengthBudgetReason) -> int:
    count = _safe_count(value, reason)
    if count == 0:
        raise SummaryLengthBudgetError(reason)
    return count


def _safe_aggregate_count(value: object, reason: SummaryLengthBudgetReason) -> int:
    if type(value) is not int or value < 0 or value > MAX_SUMMARY_REQUESTED_TOKENS:
        raise SummaryLengthBudgetError(reason)
    return value


def _coerce_reason(value: object) -> SummaryLengthBudgetReason:
    if isinstance(value, SummaryLengthBudgetReason):
        return value
    raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)


def _policy_key(item: SummaryEvidenceClassPolicy) -> tuple[int, str]:
    return item.priority, item.evidence_class


@dataclass(frozen=True, slots=True)
class SummaryEvidenceClassPolicy:
    """Policy metadata for one approved summary evidence class.

    ``priority`` is the deterministic tie-break order used when the global
    budget is too small to satisfy all minimums.  ``weight`` controls the
    largest-remainder allocation of remaining tokens.  A class's minimum and
    maximum are both advisory bounds within the global budget; actual demand
    can be lower than either bound.
    """

    evidence_class: str
    priority: int
    weight: int = 1
    minimum_tokens: int = 0
    maximum_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", _identifier(self.evidence_class))
        _safe_count(self.priority, SummaryLengthBudgetReason.INVALID_POLICY_CLASS)
        _safe_count(self.weight, SummaryLengthBudgetReason.INVALID_POLICY_CLASS)
        if self.weight == 0:
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS
            )
        _safe_count(
            self.minimum_tokens,
            SummaryLengthBudgetReason.INVALID_POLICY_CLASS,
        )
        if self.maximum_tokens is not None:
            _safe_count(
                self.maximum_tokens,
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS,
            )
            if self.maximum_tokens < self.minimum_tokens:
                raise SummaryLengthBudgetError(
                    SummaryLengthBudgetReason.INVALID_POLICY_CLASS
                )

    @property
    def name(self) -> str:
        """Return the class identifier under the concise naming convention."""

        return self.evidence_class

    @property
    def min_tokens(self) -> int:
        """Return the minimum-token policy bound."""

        return self.minimum_tokens

    @property
    def max_tokens(self) -> int | None:
        """Return the optional maximum-token policy bound."""

        return self.maximum_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic policy metadata without source content."""

        return {
            "evidence_class": self.evidence_class,
            "priority": self.priority,
            "weight": self.weight,
            "minimum_tokens": self.minimum_tokens,
            "maximum_tokens": self.maximum_tokens,
        }


_DEFAULT_EVIDENCE_CLASS_POLICIES: tuple[SummaryEvidenceClassPolicy, ...] = (
    SummaryEvidenceClassPolicy(
        "safety", priority=0, weight=4, minimum_tokens=16, maximum_tokens=128
    ),
    SummaryEvidenceClassPolicy(
        "active_problems", priority=1, weight=3, minimum_tokens=16, maximum_tokens=192
    ),
    SummaryEvidenceClassPolicy(
        "medications", priority=2, weight=2, minimum_tokens=8, maximum_tokens=128
    ),
    SummaryEvidenceClassPolicy(
        "key_findings", priority=3, weight=2, minimum_tokens=8, maximum_tokens=160
    ),
    SummaryEvidenceClassPolicy(
        "procedures", priority=4, weight=1, minimum_tokens=0, maximum_tokens=96
    ),
    SummaryEvidenceClassPolicy(
        "pending_items", priority=5, weight=2, minimum_tokens=8, maximum_tokens=96
    ),
    SummaryEvidenceClassPolicy(
        "follow_up", priority=6, weight=2, minimum_tokens=8, maximum_tokens=96
    ),
)

SUMMARY_EVIDENCE_CLASS_NAMES: Final[tuple[str, ...]] = tuple(
    item.evidence_class for item in _DEFAULT_EVIDENCE_CLASS_POLICIES
)
DEFAULT_SUMMARY_EVIDENCE_CLASS_POLICIES: Final[
    tuple[SummaryEvidenceClassPolicy, ...]
] = _DEFAULT_EVIDENCE_CLASS_POLICIES


@dataclass(frozen=True, slots=True)
class SummaryLengthBudgetPolicy:
    """Versioned closed-world policy for summary evidence classes."""

    classes: tuple[SummaryEvidenceClassPolicy, ...] = field(
        default_factory=lambda: DEFAULT_SUMMARY_EVIDENCE_CLASS_POLICIES
    )
    policy_id: str = SUMMARY_LENGTH_BUDGET_POLICY_ID
    schema_version: int = SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    disclaimer: str = SUMMARY_LENGTH_BUDGET_DISCLAIMER

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION
        ):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.UNSUPPORTED_SCHEMA)
        if (
            type(self.policy_id) is not str
            or _POLICY_ID_RE.fullmatch(self.policy_id) is None
        ):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        if self.requires_clinician_review is not True:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        if self.autonomous_decision is not False:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        if self.disclaimer != SUMMARY_LENGTH_BUDGET_DISCLAIMER:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)

        try:
            classes = tuple(self.classes)
        except Exception:
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY
            ) from None
        if not classes or len(classes) > MAX_SUMMARY_EVIDENCE_CLASSES:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        if any(type(item) is not SummaryEvidenceClassPolicy for item in classes):
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS
            )
        if len({item.evidence_class for item in classes}) != len(classes):
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS
            )
        ordered = tuple(sorted(classes, key=_policy_key))
        object.__setattr__(self, "classes", ordered)

    @property
    def evidence_classes(self) -> tuple[SummaryEvidenceClassPolicy, ...]:
        """Return the approved classes in deterministic priority order."""

        return self.classes

    def class_policy(self, evidence_class: object) -> SummaryEvidenceClassPolicy:
        """Return the policy for a class or raise a value-free error."""

        identifier = _identifier(evidence_class)
        for item in self.classes:
            if item.evidence_class == identifier:
                return item
        raise SummaryLengthBudgetError(SummaryLengthBudgetReason.UNKNOWN_EVIDENCE_CLASS)

    def to_dict(self) -> dict[str, Any]:
        """Return the closed, deterministic policy representation."""

        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "classes": [item.to_dict() for item in self.classes],
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SummaryLengthBudgetPolicy":
        """Build a policy from safe metadata mappings.

        The mapping is intentionally closed-world: each class must be declared
        explicitly with numeric policy bounds.  Unknown fields are rejected
        without including their values in the exception.
        """

        if not isinstance(value, Mapping):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        allowed = {
            "classes",
            "evidence_classes",
            "policy_id",
            "schema_version",
            "requires_clinician_review",
            "autonomous_decision",
            "disclaimer",
        }
        if any(type(key) is not str or key not in allowed for key in value):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        if "classes" in value and "evidence_classes" in value:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        raw_classes = value.get("classes", value.get("evidence_classes", _MISSING))
        if raw_classes is _MISSING:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        classes = _coerce_policy_classes(raw_classes)
        return cls(
            classes=classes,
            policy_id=value.get("policy_id", SUMMARY_LENGTH_BUDGET_POLICY_ID),
            schema_version=value.get(
                "schema_version", SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION
            ),
            requires_clinician_review=value.get("requires_clinician_review", True),
            autonomous_decision=value.get("autonomous_decision", False),
            disclaimer=value.get("disclaimer", SUMMARY_LENGTH_BUDGET_DISCLAIMER),
        )


DEFAULT_SUMMARY_LENGTH_POLICY: Final[SummaryLengthBudgetPolicy] = (
    SummaryLengthBudgetPolicy()
)
DEFAULT_SUMMARY_BUDGET_POLICY: Final[SummaryLengthBudgetPolicy] = (
    DEFAULT_SUMMARY_LENGTH_POLICY
)


@dataclass(frozen=True, slots=True)
class SummaryEvidenceDemand:
    """Non-sensitive token demand for one policy-approved evidence class."""

    evidence_class: str
    requested_tokens: int
    approved: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", _identifier(self.evidence_class))
        _safe_count(
            self.requested_tokens,
            SummaryLengthBudgetReason.INVALID_TOKEN_COUNT,
        )
        if type(self.approved) is not bool:
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.UNAPPROVED_EVIDENCE_CLASS
            )
        if not self.approved:
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.UNAPPROVED_EVIDENCE_CLASS
            )

    @property
    def name(self) -> str:
        """Return the evidence-class identifier."""

        return self.evidence_class

    @property
    def class_name(self) -> str:
        """Return the evidence-class identifier under an explicit alias."""

        return self.evidence_class

    @property
    def token_count(self) -> int:
        """Return the requested token count."""

        return self.requested_tokens

    @property
    def available_tokens(self) -> int:
        """Return the requested token count under the availability alias."""

        return self.requested_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return value-free demand metadata."""

        return {
            "evidence_class": self.evidence_class,
            "requested_tokens": self.requested_tokens,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SummaryEvidenceDemand":
        """Build a demand from a mapping without accepting source text."""

        if not isinstance(value, Mapping):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_EVIDENCE)
        allowed = {
            "evidence_class",
            "class_name",
            "name",
            "requested_tokens",
            "token_count",
            "available_tokens",
            "approved",
        }
        if any(type(key) is not str or key not in allowed for key in value):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_EVIDENCE)
        class_keys = tuple(
            key for key in ("evidence_class", "class_name", "name") if key in value
        )
        token_keys = tuple(
            key
            for key in ("requested_tokens", "token_count", "available_tokens")
            if key in value
        )
        if len(class_keys) != 1 or len(token_keys) != 1:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_EVIDENCE)
        return cls(
            evidence_class=value[class_keys[0]],
            requested_tokens=value[token_keys[0]],
            approved=value.get("approved", True),
        )


@dataclass(frozen=True, slots=True)
class SummaryClassTokenAllocation:
    """Token allocation and deterministic truncation metadata for one class."""

    evidence_class: str
    requested_tokens: int
    allocated_tokens: int
    deferred_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", _identifier(self.evidence_class))
        for value in (
            self.requested_tokens,
            self.allocated_tokens,
            self.deferred_tokens,
        ):
            _safe_count(value, SummaryLengthBudgetReason.INVALID_INPUT)
        if self.allocated_tokens > self.requested_tokens:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if self.deferred_tokens != self.requested_tokens - self.allocated_tokens:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)

    @property
    def budget_tokens(self) -> int:
        """Return the usable class-specific generation cap."""

        return self.allocated_tokens

    @property
    def token_budget(self) -> int:
        """Return :attr:`allocated_tokens` under the common alias."""

        return self.allocated_tokens

    @property
    def is_deferred(self) -> bool:
        """Return whether any requested evidence from this class was deferred."""

        return self.deferred_tokens > 0

    @property
    def is_truncated(self) -> bool:
        """Return whether the class has both included and deferred tokens."""

        return self.allocated_tokens > 0 and self.deferred_tokens > 0

    @property
    def status(self) -> str:
        """Return a fixed status suitable for reports and audit metadata."""

        if self.requested_tokens == 0:
            return "empty"
        if self.deferred_tokens == 0:
            return "included"
        if self.allocated_tokens == 0:
            return "deferred"
        return "truncated"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, value-free allocation metadata."""

        return {
            "evidence_class": self.evidence_class,
            "requested_tokens": self.requested_tokens,
            "allocated_tokens": self.allocated_tokens,
            "deferred_tokens": self.deferred_tokens,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class SummaryTruncationMetadata:
    """Value-free metadata describing evidence deferred by a budget."""

    deferred_evidence_classes: tuple[str, ...]
    deferred_tokens: int

    def __post_init__(self) -> None:
        classes = tuple(self.deferred_evidence_classes)
        if (
            any(type(evidence_class) is not str for evidence_class in classes)
            or tuple(sorted(classes)) != classes
        ):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if len(set(classes)) != len(classes):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        for evidence_class in classes:
            _identifier(evidence_class)
        _safe_aggregate_count(
            self.deferred_tokens,
            SummaryLengthBudgetReason.INVALID_INPUT,
        )
        if not classes and self.deferred_tokens != 0:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        object.__setattr__(self, "deferred_evidence_classes", classes)

    @property
    def occurred(self) -> bool:
        """Return whether at least one evidence class was deferred."""

        return bool(self.deferred_evidence_classes)

    @property
    def deferred_classes(self) -> tuple[str, ...]:
        """Return deferred class identifiers under a concise alias."""

        return self.deferred_evidence_classes

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic truncation metadata."""

        return {
            "occurred": self.occurred,
            "deferred_evidence_classes": list(self.deferred_evidence_classes),
            "deferred_tokens": self.deferred_tokens,
        }


@dataclass(frozen=True, slots=True)
class SummaryLengthBudget:
    """Versioned, deterministic budget artifact for one summary generation."""

    max_tokens: int
    policy_id: str
    allocations: tuple[SummaryClassTokenAllocation, ...]
    requested_tokens: int
    allocated_tokens: int
    unused_tokens: int
    deferred_evidence_classes: tuple[str, ...]
    schema_version: int = SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    disclaimer: str = SUMMARY_LENGTH_BUDGET_DISCLAIMER

    def __post_init__(self) -> None:
        _positive_count(self.max_tokens, SummaryLengthBudgetReason.INVALID_MAX_TOKENS)
        if (
            type(self.policy_id) is not str
            or _POLICY_ID_RE.fullmatch(self.policy_id) is None
        ):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)
        if type(self.schema_version) is not int or (
            self.schema_version != SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION
        ):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.UNSUPPORTED_SCHEMA)
        if self.requires_clinician_review is not True:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if self.autonomous_decision is not False:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if self.disclaimer != SUMMARY_LENGTH_BUDGET_DISCLAIMER:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)

        allocations = tuple(self.allocations)
        if any(type(item) is not SummaryClassTokenAllocation for item in allocations):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if len({item.evidence_class for item in allocations}) != len(allocations):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if (
            tuple(sorted(allocations, key=lambda item: item.evidence_class))
            != allocations
        ):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        _safe_aggregate_count(
            self.requested_tokens,
            SummaryLengthBudgetReason.INVALID_INPUT,
        )
        _safe_count(self.allocated_tokens, SummaryLengthBudgetReason.INVALID_INPUT)
        _safe_count(self.unused_tokens, SummaryLengthBudgetReason.INVALID_INPUT)
        if self.requested_tokens != sum(item.requested_tokens for item in allocations):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if self.allocated_tokens != sum(item.allocated_tokens for item in allocations):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if self.unused_tokens != self.max_tokens - self.allocated_tokens:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        if self.allocated_tokens > self.max_tokens:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)

        deferred = tuple(
            sorted(
                item.evidence_class for item in allocations if item.deferred_tokens > 0
            )
        )
        if tuple(self.deferred_evidence_classes) != deferred:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "deferred_evidence_classes", deferred)

    @property
    def total_tokens(self) -> int:
        """Return the global generation cap."""

        return self.max_tokens

    @property
    def total_budget(self) -> int:
        """Return the global generation cap under a budget alias."""

        return self.max_tokens

    @property
    def truncated(self) -> bool:
        """Return whether any evidence class has deferred tokens."""

        return bool(self.deferred_evidence_classes)

    @property
    def truncation_occurred(self) -> bool:
        """Return :attr:`truncated` under an explicit metadata alias."""

        return self.truncated

    @property
    def deferred_token_count(self) -> int:
        """Return the aggregate count of deferred requested tokens."""

        return sum(item.deferred_tokens for item in self.allocations)

    @property
    def truncation(self) -> SummaryTruncationMetadata:
        """Return deterministic metadata for deferred evidence classes."""

        return SummaryTruncationMetadata(
            deferred_evidence_classes=self.deferred_evidence_classes,
            deferred_tokens=self.deferred_token_count,
        )

    @property
    def truncation_metadata(self) -> SummaryTruncationMetadata:
        """Return :attr:`truncation` under an explicit metadata alias."""

        return self.truncation

    def allocation_for(
        self, evidence_class: object
    ) -> SummaryClassTokenAllocation | None:
        """Return a class allocation, or ``None`` when demand was not supplied."""

        identifier = _identifier(evidence_class)
        for item in self.allocations:
            if item.evidence_class == identifier:
                return item
        return None

    def budget_for(self, evidence_class: object) -> int:
        """Return the class-specific cap, or zero when demand was not supplied."""

        allocation = self.allocation_for(evidence_class)
        return allocation.allocated_tokens if allocation is not None else 0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report containing only safe metadata."""

        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "max_tokens": self.max_tokens,
            "requested_tokens": self.requested_tokens,
            "allocated_tokens": self.allocated_tokens,
            "unused_tokens": self.unused_tokens,
            "allocations": [item.to_dict() for item in self.allocations],
            "truncation": self.truncation.to_dict(),
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON for local review and regression artifacts."""

        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def build_summary_length_budget(
    max_tokens: object | None = None,
    evidence: object | None = None,
    *,
    total_tokens: object | None = None,
    evidence_classes: object | None = None,
    policy: SummaryLengthBudgetPolicy | Mapping[str, Any] | None = None,
) -> SummaryLengthBudget:
    """Allocate a deterministic token budget across approved evidence classes.

    Args:
        max_tokens: Global maximum number of output tokens. ``total_tokens``
            is an explicit keyword alias; supplying both is rejected.
        evidence: Token demands as a mapping of class identifier to integer,
            an iterable of :class:`SummaryEvidenceDemand` values, or an
            iterable of safe mappings. A demand is an estimate supplied by the
            caller after local evidence review; this function never tokenizes
            source text.
        total_tokens: Alias for ``max_tokens`` for callers that use total-budget
            terminology.
        evidence_classes: Alias for ``evidence``. A sequence of class names
            treats every named class as eligible for the global cap, which is
            useful when planning before class-level demand estimates exist.
        policy: A :class:`SummaryLengthBudgetPolicy` or its metadata mapping.
            The default is the closed ``clinical_summary_v1`` policy.

    Returns:
        A versioned artifact with one allocation for every policy class and
        ``deferred_evidence_classes`` for every class whose requested demand
        did not fit in full. Allocations are class-specific caps for a local
        generator; this function does not generate or truncate text.

    Raises:
        SummaryLengthBudgetError: If the global limit, policy, or demand is
            invalid or references a class outside the approved policy.
    """

    if max_tokens is not None and total_tokens is not None:
        raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
    raw_max_tokens = max_tokens if max_tokens is not None else total_tokens
    if raw_max_tokens is None:
        raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_MAX_TOKENS)
    global_limit = _positive_count(
        raw_max_tokens,
        SummaryLengthBudgetReason.INVALID_MAX_TOKENS,
    )

    if evidence is not None and evidence_classes is not None:
        raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_INPUT)
    raw_evidence = evidence if evidence is not None else evidence_classes
    normalized_policy = _coerce_policy(policy)
    if raw_evidence is None:
        demands = tuple(
            SummaryEvidenceDemand(item.evidence_class, global_limit)
            for item in normalized_policy.classes
        )
    else:
        demands = _coerce_demands(raw_evidence, global_limit)

    demand_by_class: dict[str, int] = {}
    for demand in demands:
        if not isinstance(demand, SummaryEvidenceDemand):
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_EVIDENCE)
        try:
            normalized_policy.class_policy(demand.evidence_class)
        except SummaryLengthBudgetError as error:
            if error.reason is SummaryLengthBudgetReason.UNKNOWN_EVIDENCE_CLASS:
                raise
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_EVIDENCE
            ) from None
        demand_by_class[demand.evidence_class] = (
            demand_by_class.get(demand.evidence_class, 0) + demand.requested_tokens
        )

    allocations_by_class = _allocate(
        global_limit,
        normalized_policy,
        demand_by_class,
    )
    allocations = tuple(
        sorted(allocations_by_class, key=lambda item: item.evidence_class)
    )
    return SummaryLengthBudget(
        max_tokens=global_limit,
        policy_id=normalized_policy.policy_id,
        allocations=allocations,
        requested_tokens=sum(item.requested_tokens for item in allocations),
        allocated_tokens=sum(item.allocated_tokens for item in allocations),
        unused_tokens=global_limit - sum(item.allocated_tokens for item in allocations),
        deferred_evidence_classes=tuple(
            item.evidence_class for item in allocations if item.deferred_tokens > 0
        ),
    )


def allocate_summary_length_budget(
    max_tokens: object | None = None,
    evidence: object | None = None,
    *,
    total_tokens: object | None = None,
    evidence_classes: object | None = None,
    policy: SummaryLengthBudgetPolicy | Mapping[str, Any] | None = None,
) -> SummaryLengthBudget:
    """Alias for :func:`build_summary_length_budget`."""

    return build_summary_length_budget(
        max_tokens,
        evidence,
        total_tokens=total_tokens,
        evidence_classes=evidence_classes,
        policy=policy,
    )


def plan_summary_length_budget(
    max_tokens: object | None = None,
    evidence: object | None = None,
    *,
    total_tokens: object | None = None,
    evidence_classes: object | None = None,
    policy: SummaryLengthBudgetPolicy | Mapping[str, Any] | None = None,
) -> SummaryLengthBudget:
    """Plan a summary budget under the full planning-oriented name."""

    return build_summary_length_budget(
        max_tokens,
        evidence,
        total_tokens=total_tokens,
        evidence_classes=evidence_classes,
        policy=policy,
    )


def _allocate(
    global_limit: int,
    policy: SummaryLengthBudgetPolicy,
    demand_by_class: Mapping[str, int],
) -> tuple[SummaryClassTokenAllocation, ...]:
    capacities: dict[str, int] = {}
    for item in policy.classes:
        requested = demand_by_class.get(item.evidence_class, 0)
        if item.maximum_tokens is None:
            capacities[item.evidence_class] = requested
        else:
            capacities[item.evidence_class] = min(requested, item.maximum_tokens)

    allocated = dict.fromkeys(capacities, 0)
    minimums = {
        item.evidence_class: min(item.minimum_tokens, capacities[item.evidence_class])
        for item in policy.classes
    }
    minimum_total = sum(minimums.values())
    if minimum_total <= global_limit:
        allocated.update(minimums)
        remaining = min(global_limit, sum(capacities.values())) - minimum_total
        _distribute_weighted(remaining, policy.classes, capacities, allocated)
    else:
        # A severely constrained global budget cannot satisfy all minimums. The
        # priority order is the deterministic fail-soft policy, keeping the
        # highest-priority approved classes visible first.
        remaining = global_limit
        for item in policy.classes:
            amount = min(minimums[item.evidence_class], remaining)
            allocated[item.evidence_class] = amount
            remaining -= amount
            if remaining == 0:
                break

    return tuple(
        SummaryClassTokenAllocation(
            evidence_class=item.evidence_class,
            requested_tokens=demand_by_class.get(item.evidence_class, 0),
            allocated_tokens=allocated[item.evidence_class],
            deferred_tokens=(
                demand_by_class.get(item.evidence_class, 0)
                - allocated[item.evidence_class]
            ),
        )
        for item in policy.classes
    )


def _distribute_weighted(
    remaining: int,
    policies: Sequence[SummaryEvidenceClassPolicy],
    capacities: Mapping[str, int],
    allocated: dict[str, int],
) -> None:
    while remaining > 0:
        active = tuple(
            item
            for item in policies
            if allocated[item.evidence_class] < capacities[item.evidence_class]
        )
        if not active:
            return
        total_weight = sum(item.weight for item in active)
        additions: dict[str, int] = {}
        remainders: dict[str, int] = {}
        for item in active:
            room = capacities[item.evidence_class] - allocated[item.evidence_class]
            numerator = remaining * item.weight
            additions[item.evidence_class] = min(room, numerator // total_weight)
            remainders[item.evidence_class] = numerator % total_weight

        used = sum(additions.values())
        for evidence_class, amount in additions.items():
            allocated[evidence_class] += amount
        remaining -= used
        if remaining == 0:
            return

        ranked = sorted(
            active,
            key=lambda item: (
                -remainders[item.evidence_class],
                item.priority,
                item.evidence_class,
            ),
        )
        gave_remainder = False
        for item in ranked:
            if remaining == 0:
                return
            if allocated[item.evidence_class] >= capacities[item.evidence_class]:
                continue
            allocated[item.evidence_class] += 1
            remaining -= 1
            gave_remainder = True
        if not gave_remainder and used == 0:
            return


def _coerce_policy(
    value: SummaryLengthBudgetPolicy | Mapping[str, Any] | None,
) -> SummaryLengthBudgetPolicy:
    if value is None:
        return DEFAULT_SUMMARY_LENGTH_POLICY
    if type(value) is SummaryLengthBudgetPolicy:
        return value
    if isinstance(value, Mapping):
        return SummaryLengthBudgetPolicy.from_mapping(value)
    raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)


def _coerce_policy_classes(value: object) -> tuple[SummaryEvidenceClassPolicy, ...]:
    if isinstance(value, Mapping):
        rows = []
        for name, config in _bounded_items(
            value, SummaryLengthBudgetReason.INVALID_POLICY
        ):
            if not isinstance(config, Mapping):
                raise SummaryLengthBudgetError(
                    SummaryLengthBudgetReason.INVALID_POLICY_CLASS
                )
            row = dict(config)
            row.setdefault("evidence_class", name)
            rows.append(row)
    else:
        rows = _bounded_sequence(value, SummaryLengthBudgetReason.INVALID_POLICY)
    if not rows or len(rows) > MAX_SUMMARY_EVIDENCE_CLASSES:
        raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_POLICY)

    classes: list[SummaryEvidenceClassPolicy] = []
    for row in rows:
        if isinstance(row, SummaryEvidenceClassPolicy):
            classes.append(row)
            continue
        if isinstance(row, str):
            classes.append(SummaryEvidenceClassPolicy(row, priority=len(classes)))
            continue
        if not isinstance(row, Mapping):
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS
            )
        allowed = {
            "evidence_class",
            "class_name",
            "name",
            "priority",
            "weight",
            "minimum_tokens",
            "min_tokens",
            "maximum_tokens",
            "max_tokens",
        }
        if any(type(key) is not str or key not in allowed for key in row):
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS
            )
        class_keys = tuple(
            key for key in ("evidence_class", "class_name", "name") if key in row
        )
        minimum_keys = tuple(
            key for key in ("minimum_tokens", "min_tokens") if key in row
        )
        maximum_keys = tuple(
            key for key in ("maximum_tokens", "max_tokens") if key in row
        )
        if len(class_keys) != 1 or len(minimum_keys) > 1 or len(maximum_keys) > 1:
            raise SummaryLengthBudgetError(
                SummaryLengthBudgetReason.INVALID_POLICY_CLASS
            )
        classes.append(
            SummaryEvidenceClassPolicy(
                evidence_class=row[class_keys[0]],
                priority=row.get("priority", len(classes)),
                weight=row.get("weight", 1),
                minimum_tokens=row.get(minimum_keys[0], 0) if minimum_keys else 0,
                maximum_tokens=row.get(maximum_keys[0]) if maximum_keys else None,
            )
        )
    return tuple(classes)


def _coerce_demands(
    value: object, global_limit: int
) -> tuple[SummaryEvidenceDemand, ...]:
    rows: list[Any]
    if isinstance(value, Mapping):
        nested = tuple(
            key for key in ("evidence", "evidence_classes", "classes") if key in value
        )
        if nested:
            if len(nested) != 1 or len(value) != 1:
                raise SummaryLengthBudgetError(
                    SummaryLengthBudgetReason.INVALID_EVIDENCE
                )
            return _coerce_demands(value[nested[0]], global_limit)
        rows = []
        for name, token_value in _bounded_items(
            value, SummaryLengthBudgetReason.INVALID_EVIDENCE
        ):
            if isinstance(token_value, Mapping):
                row = dict(token_value)
                row.setdefault("evidence_class", name)
                rows.append(row)
            else:
                rows.append((name, token_value))
    elif isinstance(value, (str, bytes, bytearray)):
        rows = [value]
    elif isinstance(value, (tuple, list)) and _looks_like_demand_pair(value):
        rows = [value]
    else:
        rows = _bounded_sequence(value, SummaryLengthBudgetReason.INVALID_EVIDENCE)

    if len(rows) > MAX_SUMMARY_EVIDENCE_CLASSES * 4:
        raise SummaryLengthBudgetError(
            SummaryLengthBudgetReason.INVALID_EVIDENCE,
            rejected_count=len(rows),
        )
    demands: list[SummaryEvidenceDemand] = []
    for row in rows:
        if isinstance(row, SummaryEvidenceDemand):
            demands.append(row)
        elif isinstance(row, SummaryEvidenceClassPolicy):
            demands.append(SummaryEvidenceDemand(row.evidence_class, global_limit))
        elif isinstance(row, str):
            demands.append(SummaryEvidenceDemand(row, global_limit))
        elif isinstance(row, Mapping):
            demands.append(SummaryEvidenceDemand.from_mapping(row))
        elif _looks_like_demand_pair(row):
            demands.append(
                SummaryEvidenceDemand(
                    evidence_class=row[0],
                    requested_tokens=row[1],
                )
            )
        else:
            raise SummaryLengthBudgetError(SummaryLengthBudgetReason.INVALID_EVIDENCE)
    return tuple(demands)


def _looks_like_demand_pair(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
        and isinstance(value[0], str)
        and type(value[1]) is int
    )


def _bounded_sequence(value: object, reason: SummaryLengthBudgetReason) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise SummaryLengthBudgetError(reason)
    try:
        rows = list(itertools.islice(value, MAX_SUMMARY_EVIDENCE_CLASSES * 4 + 1))
    except Exception:
        raise SummaryLengthBudgetError(reason) from None
    if len(rows) > MAX_SUMMARY_EVIDENCE_CLASSES * 4:
        raise SummaryLengthBudgetError(reason, rejected_count=len(rows))
    return rows


def _bounded_items(
    value: Mapping[object, object], reason: SummaryLengthBudgetReason
) -> list[tuple[object, object]]:
    try:
        rows = list(
            itertools.islice(value.items(), MAX_SUMMARY_EVIDENCE_CLASSES * 4 + 1)
        )
    except Exception:
        raise SummaryLengthBudgetError(reason) from None
    if len(rows) > MAX_SUMMARY_EVIDENCE_CLASSES * 4:
        raise SummaryLengthBudgetError(reason, rejected_count=len(rows))
    return rows


# Descriptive aliases keep the API discoverable for callers that use budget or
# allocation terminology while preserving one implementation and one schema.
SummaryEvidencePolicy = SummaryEvidenceClassPolicy
SummaryBudgetPolicy = SummaryLengthBudgetPolicy
EvidenceClassRequest = SummaryEvidenceDemand
SummaryBudgetAllocation = SummaryClassTokenAllocation
EvidenceClassBudget = SummaryClassTokenAllocation
SummaryTokenBudget = SummaryLengthBudget
SummaryEvidenceClass = SummaryEvidenceClassPolicy
SummaryLengthBudgetPlan = SummaryLengthBudget
TruncationMetadata = SummaryTruncationMetadata


__all__ = [
    "DEFAULT_SUMMARY_BUDGET_POLICY",
    "DEFAULT_SUMMARY_EVIDENCE_CLASS_POLICIES",
    "DEFAULT_SUMMARY_LENGTH_POLICY",
    "EvidenceClassBudget",
    "EvidenceClassRequest",
    "MAX_SUMMARY_EVIDENCE_CLASSES",
    "MAX_SUMMARY_EVIDENCE_CLASS_ID_LENGTH",
    "MAX_SUMMARY_LENGTH_TOKENS",
    "MAX_SUMMARY_REQUESTED_TOKENS",
    "SUMMARY_EVIDENCE_CLASS_NAMES",
    "SUMMARY_LENGTH_BUDGET_DISCLAIMER",
    "SUMMARY_LENGTH_BUDGET_POLICY_ID",
    "SUMMARY_LENGTH_BUDGET_SCHEMA_VERSION",
    "SummaryBudgetAllocation",
    "SummaryBudgetPolicy",
    "SummaryClassTokenAllocation",
    "SummaryEvidenceClassPolicy",
    "SummaryEvidenceClass",
    "SummaryEvidenceDemand",
    "SummaryEvidencePolicy",
    "SummaryLengthBudget",
    "SummaryLengthBudgetError",
    "SummaryLengthBudgetPolicy",
    "SummaryLengthBudgetPlan",
    "SummaryLengthBudgetReason",
    "SummaryTruncationMetadata",
    "SummaryTokenBudget",
    "TruncationMetadata",
    "allocate_summary_length_budget",
    "build_summary_length_budget",
    "plan_summary_length_budget",
]
