"""Deterministic, privacy-preserving refusal reasons for clinical surfaces.

Guarded clinical callers should report why a request was refused without
copying the request, note text, or a free-form safety message into logs or
downstream artifacts.  This module accepts only stable category identifiers
and emits fixed, reviewable remediation hints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

REFUSAL_TAXONOMY_SCHEMA_VERSION = 1


class RefusalCategory(str, Enum):
    """Stable identifiers for guarded clinical refusals.

    The identifiers are deliberately short, ASCII, and independent of any
    caller-provided request or document content.  They are part of the
    serialized contract and should not be renamed casually.
    """

    MISSING_EVIDENCE = "missing_evidence"
    POLICY_BLOCK = "policy_block"
    AMBIGUITY = "ambiguity"
    UNSUPPORTED_REQUEST = "unsupported_request"

    @classmethod
    def from_value(cls, value: object) -> "RefusalCategory":
        """Normalize one supported category identifier.

        A small amount of input normalization makes configuration files less
        brittle, while serialization always uses the canonical underscore form.
        Error messages intentionally do not include the rejected value because
        callers must be able to pass untrusted input without leaking it.
        """

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("refusal category must be a supported identifier")

        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError("unsupported refusal category") from None


REFUSAL_CATEGORY_VALUES = tuple(category.value for category in RefusalCategory)

REFUSAL_REMEDIATION_HINTS: Mapping[str, str] = MappingProxyType(
    {
        RefusalCategory.MISSING_EVIDENCE.value: (
            "Provide the minimum required evidence and retry when it is available."
        ),
        RefusalCategory.POLICY_BLOCK.value: (
            "Review the applicable policy and use an approved workflow."
        ),
        RefusalCategory.AMBIGUITY.value: (
            "Clarify the request or supply disambiguating context."
        ),
        RefusalCategory.UNSUPPORTED_REQUEST.value: (
            "Use a supported clinical workflow or consult the capability guidance."
        ),
    }
)


def remediation_hint_for(category: RefusalCategory | str) -> str:
    """Return the fixed remediation hint for ``category``."""

    normalized = RefusalCategory.from_value(category)
    return REFUSAL_REMEDIATION_HINTS[normalized.value]


@dataclass(frozen=True)
class RefusalReason:
    """One counted refusal category without caller-supplied free-form text.

    ``count`` is useful when a producer already has an aggregate.  A single
    refusal normally uses the default count of one.  The remediation hint is a
    property derived from the category, so arbitrary caller text cannot enter
    the serialized representation.
    """

    category: RefusalCategory | str
    count: int = 1

    def __post_init__(self) -> None:
        category = RefusalCategory.from_value(self.category)
        _validate_positive_count(self.count)
        object.__setattr__(self, "category", category)

    @property
    def category_id(self) -> str:
        """Return the canonical serialized category identifier."""

        return self.category.value

    @property
    def remediation_hint(self) -> str:
        """Return the category's fixed, PHI-free remediation hint."""

        return remediation_hint_for(self.category)

    def to_dict(self) -> dict[str, Any]:
        """Return only the category identifier, count, and remediation hint."""

        return {
            "category": self.category_id,
            "count": self.count,
            "remediation_hint": self.remediation_hint,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Return the privacy-safe serialized reason representation."""

        return self.to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefusalReason":
        """Build a reason from a serialized category record.

        Unknown fields are ignored rather than copied.  In particular, a
        producer cannot smuggle a free-form message into a reason by round-trip
        serialization.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("refusal reason must be a mapping")
        return cls(payload.get("category"), payload.get("count", 1))


@dataclass(frozen=True)
class RefusalReport:
    """Aggregate refusal counts with canonical remediation hints.

    The in-memory mapping is normalized to canonical category strings and
    stored in taxonomy order.  ``to_dict`` contains no source text, request
    text, exception message, or caller-provided metadata.
    """

    counts: Mapping[RefusalCategory | str, int] = field(default_factory=dict)

    _CATEGORY_ORDER: ClassVar[tuple[RefusalCategory, ...]] = tuple(RefusalCategory)

    def __post_init__(self) -> None:
        if not isinstance(self.counts, Mapping):
            raise TypeError("refusal counts must be a mapping")

        normalized: dict[str, int] = {}
        for category, count in self.counts.items():
            normalized_category = RefusalCategory.from_value(category).value
            normalized_count = _validate_nonnegative_count(count)
            if normalized_count:
                normalized[normalized_category] = (
                    normalized.get(normalized_category, 0) + normalized_count
                )

        ordered = {
            category.value: normalized[category.value]
            for category in self._CATEGORY_ORDER
            if category.value in normalized
        }
        object.__setattr__(self, "counts", MappingProxyType(ordered))

    @classmethod
    def from_categories(
        cls,
        categories: Iterable[RefusalCategory | str],
    ) -> "RefusalReport":
        """Aggregate a sequence of category identifiers deterministically."""

        return aggregate_refusals(categories)

    @classmethod
    def from_reasons(
        cls,
        reasons: Iterable[RefusalReason | RefusalCategory | str],
    ) -> "RefusalReport":
        """Aggregate counted reasons or individual category identifiers."""

        return aggregate_refusals(reasons)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefusalReport":
        """Rebuild a report while discarding all fields except counts."""

        if not isinstance(payload, Mapping):
            raise TypeError("refusal report must be a mapping")
        counts = payload.get("counts", {})
        return cls(counts)

    @property
    def total_count(self) -> int:
        """Return the aggregate number of refused requests represented."""

        return sum(self.counts.values())

    @property
    def categories(self) -> tuple[str, ...]:
        """Return observed category identifiers in canonical order."""

        return tuple(self.counts)

    @property
    def remediation_hints(self) -> Mapping[str, str]:
        """Return fixed hints for observed categories in canonical order."""

        return MappingProxyType(
            {category: remediation_hint_for(category) for category in self.counts}
        )

    def to_records(self) -> tuple[RefusalReason, ...]:
        """Return observed categories as deterministic counted records."""

        return tuple(
            RefusalReason(category, count) for category, count in self.counts.items()
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize only category counts and fixed remediation hints."""

        counts = dict(self.counts)
        return {
            "counts": counts,
            "remediation_hints": dict(self.remediation_hints),
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Return the privacy-safe serialized report representation."""

        return self.to_dict()


def build_refusal(
    category: RefusalCategory | str,
    *,
    count: int = 1,
) -> RefusalReason:
    """Create one validated, privacy-safe refusal reason."""

    return RefusalReason(category=category, count=count)


def classify_refusal(category: RefusalCategory | str) -> RefusalReason:
    """Classify a refusal from a stable category identifier.

    This function intentionally does not classify arbitrary request or note
    text.  Upstream guarded logic must make that decision and pass only the
    resulting taxonomy identifier.
    """

    return build_refusal(category)


def aggregate_refusals(
    refusals: Iterable[RefusalReason | RefusalCategory | str | Mapping[str, Any]]
    | RefusalReason
    | RefusalCategory
    | str
    | None,
) -> RefusalReport:
    """Aggregate refusal categories without retaining source content.

    Mapping inputs may contain only the same ``category`` and ``count`` fields
    accepted by :meth:`RefusalReason.from_dict`; any other fields are ignored.
    """

    if refusals is None:
        return RefusalReport()
    if isinstance(refusals, (RefusalReason, RefusalCategory, str, Mapping)):
        items: Iterable[Any] = (refusals,)
    else:
        items = refusals

    counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, RefusalReason):
            reason = item
        elif isinstance(item, Mapping):
            reason = RefusalReason.from_dict(item)
        else:
            reason = RefusalReason(item)
        counts[reason.category_id] = counts.get(reason.category_id, 0) + reason.count
    return RefusalReport(counts)


def serialize_refusal(
    category: RefusalCategory | str,
    *,
    count: int = 1,
) -> dict[str, Any]:
    """Serialize one refusal using only taxonomy-safe fields."""

    return build_refusal(category, count=count).to_dict()


def serialize_refusals(
    refusals: Iterable[RefusalReason | RefusalCategory | str | Mapping[str, Any]]
    | RefusalReason
    | RefusalCategory
    | str
    | None,
) -> dict[str, Any]:
    """Serialize an aggregate refusal report without source content."""

    return aggregate_refusals(refusals).to_dict()


def serialize_refusal_report(
    report: RefusalReport | Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize a report while retaining only canonical taxonomy fields."""

    if isinstance(report, RefusalReport):
        return report.to_dict()
    return RefusalReport.from_dict(report).to_dict()


class RefusalTaxonomy:
    """Namespace exposing the canonical refusal taxonomy operations."""

    categories: ClassVar[tuple[str, ...]] = REFUSAL_CATEGORY_VALUES
    remediation_hints: ClassVar[Mapping[str, str]] = REFUSAL_REMEDIATION_HINTS

    @classmethod
    def classify(cls, category: RefusalCategory | str) -> RefusalReason:
        """Build one reason from a canonical category identifier."""

        return classify_refusal(category)

    @classmethod
    def aggregate(
        cls,
        refusals: Iterable[RefusalReason | RefusalCategory | str | Mapping[str, Any]]
        | RefusalReason
        | RefusalCategory
        | str
        | None,
    ) -> RefusalReport:
        """Aggregate category identifiers or counted reasons."""

        return aggregate_refusals(refusals)

    @classmethod
    def serialize(
        cls,
        refusals: Iterable[RefusalReason | RefusalCategory | str | Mapping[str, Any]]
        | RefusalReason
        | RefusalCategory
        | str
        | None,
    ) -> dict[str, Any]:
        """Serialize refusals to the privacy-safe report shape."""

        return serialize_refusals(refusals)


# Descriptive aliases keep the public surface discoverable for clinical callers
# while preserving one implementation and one serialized contract.
ClinicalRefusalCategory = RefusalCategory
ClinicalRefusal = RefusalReason
ClinicalRefusalReport = RefusalReport


def _validate_positive_count(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("refusal count must be a positive integer")


def _validate_nonnegative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("refusal count must be a non-negative integer")
    return value


__all__ = [
    "REFUSAL_CATEGORY_VALUES",
    "REFUSAL_REMEDIATION_HINTS",
    "REFUSAL_TAXONOMY_SCHEMA_VERSION",
    "ClinicalRefusal",
    "ClinicalRefusalCategory",
    "ClinicalRefusalReport",
    "RefusalCategory",
    "RefusalReason",
    "RefusalReport",
    "RefusalTaxonomy",
    "aggregate_refusals",
    "build_refusal",
    "classify_refusal",
    "remediation_hint_for",
    "serialize_refusal",
    "serialize_refusal_report",
    "serialize_refusals",
]
