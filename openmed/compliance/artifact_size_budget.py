"""Deterministic size budgets for counts-only audit-artifact descriptors.

Audit reports may be safe to publish as counts while still becoming too large
to handle operationally.  This module evaluates a small descriptor contract
instead of accepting an artifact path or its contents.  Only byte and count
fields cross the boundary, and results never contain section names or other
free-form values.

The evaluator is local and deterministic: it performs no filesystem, network,
or model access.  Unknown mapping fields are ignored so a caller can pass a
larger descriptor without accidentally copying content into a budget report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Final, Literal

ArtifactBudgetCategory = Literal[
    "total_bytes",
    "section_bytes",
    "record_count",
    "nesting_depth",
]

_CATEGORIES: Final[tuple[ArtifactBudgetCategory, ...]] = (
    "total_bytes",
    "section_bytes",
    "record_count",
    "nesting_depth",
)
_TOTAL_BYTES_KEYS: Final[tuple[str, ...]] = ("total_bytes", "total_size_bytes")
_SECTION_BYTES_KEYS: Final[tuple[str, ...]] = (
    "size_bytes",
    "section_bytes",
    "bytes",
)
_RECORD_COUNT_KEYS: Final[tuple[str, ...]] = (
    "record_count",
    "record_count_total",
    "records",
)
_NESTING_DEPTH_KEYS: Final[tuple[str, ...]] = (
    "nesting_depth",
    "max_nesting_depth",
    "depth",
)
_SECTIONS_KEYS: Final[tuple[str, ...]] = ("sections",)


def _first_value(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Read one of the fixed count-field names without inspecting other data."""

    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _non_negative_int(
    value: Any,
    *,
    field_name: str,
    allow_none: bool = False,
) -> int | None:
    """Validate a count without including the supplied value in an error."""

    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _section_values(value: Any) -> tuple[Any, ...]:
    """Return section mappings without exposing mapping keys in the result."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(key in value for key in (*_SECTION_BYTES_KEYS, *_RECORD_COUNT_KEYS)):
            return (value,)
        return tuple(value.values())
    if not _is_sequence(value):
        raise TypeError("sections must be a sequence of count-only descriptors")
    return tuple(value)


@dataclass(frozen=True)
class ArtifactSectionDescriptor:
    """Count-only measurements for one audit-artifact section.

    ``size_bytes`` is the section's serialized size, ``record_count`` is the
    number of records represented by the section, and ``nesting_depth`` is the
    deepest structured value in that section.  No section identifier or
    content is accepted by this type.
    """

    size_bytes: int = 0
    record_count: int = 0
    nesting_depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "size_bytes",
            _non_negative_int(self.size_bytes, field_name="size_bytes"),
        )
        object.__setattr__(
            self,
            "record_count",
            _non_negative_int(self.record_count, field_name="record_count"),
        )
        object.__setattr__(
            self,
            "nesting_depth",
            _non_negative_int(self.nesting_depth, field_name="nesting_depth"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactSectionDescriptor":
        """Build a section descriptor from allow-listed count fields."""

        if not isinstance(payload, Mapping):
            raise TypeError("each section must be a count-only mapping")
        return cls(
            size_bytes=_first_value(payload, _SECTION_BYTES_KEYS, 0),
            record_count=_first_value(payload, _RECORD_COUNT_KEYS, 0),
            nesting_depth=_first_value(payload, _NESTING_DEPTH_KEYS, 0),
        )

    def to_dict(self) -> dict[str, int]:
        """Return only the section's numeric measurements."""

        return {
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "nesting_depth": self.nesting_depth,
        }


@dataclass(frozen=True)
class ArtifactDescriptor:
    """A synthetic, counts-only description of an audit artifact.

    ``record_count`` and ``max_nesting_depth`` are optional aggregate values.
    When omitted, they are derived from the section descriptors.  If an
    explicit aggregate is smaller than a section-derived value, the larger
    value is used during evaluation so a partial descriptor cannot bypass a
    budget.
    """

    total_bytes: int
    sections: tuple[ArtifactSectionDescriptor, ...] = ()
    record_count: int | None = None
    max_nesting_depth: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "total_bytes",
            _non_negative_int(self.total_bytes, field_name="total_bytes"),
        )
        sections = tuple(self.sections)
        if not all(
            isinstance(section, ArtifactSectionDescriptor) for section in sections
        ):
            raise TypeError("sections must contain count-only descriptors")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(
            self,
            "record_count",
            _non_negative_int(
                self.record_count,
                field_name="record_count",
                allow_none=True,
            ),
        )
        object.__setattr__(
            self,
            "max_nesting_depth",
            _non_negative_int(
                self.max_nesting_depth,
                field_name="max_nesting_depth",
                allow_none=True,
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactDescriptor":
        """Build a descriptor while ignoring non-count fields.

        The total size may be omitted when it can be conservatively derived
        from section sizes.  This method does not call ``to_dict`` on arbitrary
        objects, open paths, or inspect free-form descriptor values.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("artifact descriptor must be a count-only mapping")
        sections = tuple(
            ArtifactSectionDescriptor.from_mapping(section)
            for section in _section_values(_first_value(payload, _SECTIONS_KEYS, ()))
        )
        declared_total = _first_value(payload, _TOTAL_BYTES_KEYS)
        total_bytes = (
            sum(section.size_bytes for section in sections)
            if declared_total is None
            else declared_total
        )
        return cls(
            total_bytes=total_bytes,
            sections=sections,
            record_count=_first_value(payload, _RECORD_COUNT_KEYS),
            max_nesting_depth=_first_value(payload, _NESTING_DEPTH_KEYS),
        )

    @property
    def effective_total_bytes(self) -> int:
        """Return total bytes conservatively bounded by section measurements."""

        section_total = sum(section.size_bytes for section in self.sections)
        return max(self.total_bytes, section_total)

    @property
    def effective_record_count(self) -> int:
        """Return the explicit or section-derived record count."""

        section_total = sum(section.record_count for section in self.sections)
        return max(self.record_count or 0, section_total)

    @property
    def effective_nesting_depth(self) -> int:
        """Return the explicit or section-derived maximum nesting depth."""

        section_depth = max(
            (section.nesting_depth for section in self.sections),
            default=0,
        )
        return max(self.max_nesting_depth or 0, section_depth)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic descriptor containing counts only."""

        return {
            "total_bytes": self.total_bytes,
            "record_count": self.record_count,
            "max_nesting_depth": self.max_nesting_depth,
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(frozen=True)
class ArtifactSizeBudget:
    """Optional upper bounds for a counts-only artifact descriptor.

    ``None`` leaves a dimension unchecked.  All configured limits are
    inclusive: a measurement equal to its limit remains within budget.
    """

    max_total_bytes: int | None = None
    max_section_bytes: int | None = None
    max_record_count: int | None = None
    max_nesting_depth: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_total_bytes",
            "max_section_bytes",
            "max_record_count",
            "max_nesting_depth",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(
                    getattr(self, field_name),
                    field_name=field_name,
                    allow_none=True,
                ),
            )

    @property
    def max_records(self) -> int | None:
        """Compatibility alias for the record-count limit."""

        return self.max_record_count

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactSizeBudget":
        """Build a budget from stable count-limit field aliases."""

        if not isinstance(payload, Mapping):
            raise TypeError("budget must be a count-only mapping")
        return cls(
            max_total_bytes=_first_value(
                payload,
                ("max_total_bytes", "total_bytes"),
            ),
            max_section_bytes=_first_value(
                payload,
                ("max_section_bytes", "max_bytes_per_section", "section_bytes"),
            ),
            max_record_count=_first_value(
                payload,
                ("max_record_count", "max_records", "record_count"),
            ),
            max_nesting_depth=_first_value(
                payload,
                ("max_nesting_depth", "nesting_depth"),
            ),
        )

    def to_dict(self) -> dict[str, int | None]:
        """Return a deterministic JSON-compatible budget payload."""

        return {
            "max_total_bytes": self.max_total_bytes,
            "max_section_bytes": self.max_section_bytes,
            "max_record_count": self.max_record_count,
            "max_nesting_depth": self.max_nesting_depth,
        }


@dataclass(frozen=True)
class ArtifactSizeBudgetViolation:
    """One exceeded count-only artifact budget dimension."""

    category: ArtifactBudgetCategory
    observed: int
    limit: int

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise ValueError("category is not an artifact size budget dimension")
        object.__setattr__(
            self,
            "observed",
            _non_negative_int(self.observed, field_name="observed"),
        )
        object.__setattr__(
            self,
            "limit",
            _non_negative_int(self.limit, field_name="limit"),
        )

    def to_dict(self) -> dict[str, str | int]:
        """Return the category and numeric counts without descriptor data."""

        return {
            "category": self.category,
            "observed": self.observed,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class ArtifactSizeBudgetResult:
    """Deterministic, count-only output from a budget evaluation."""

    within_budget: bool
    observed: Mapping[str, int]
    limits: Mapping[str, int | None]
    violations: tuple[ArtifactSizeBudgetViolation, ...] = ()

    @property
    def exceeded_categories(self) -> tuple[ArtifactBudgetCategory, ...]:
        """Return exceeded dimensions in the fixed, documented order."""

        return tuple(violation.category for violation in self.violations)

    @property
    def exceeded_budget_categories(self) -> tuple[ArtifactBudgetCategory, ...]:
        """Explicit alias for callers that use the gate terminology."""

        return self.exceeded_categories

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report without raw artifact values."""

        return {
            "within_budget": self.within_budget,
            "exceeded_categories": list(self.exceeded_categories),
            "observed": {category: self.observed[category] for category in _CATEGORIES},
            "limits": {category: self.limits[category] for category in _CATEGORIES},
            "violations": [violation.to_dict() for violation in self.violations],
        }


def _resolve_descriptor(
    descriptor: ArtifactDescriptor | Mapping[str, Any],
) -> ArtifactDescriptor:
    if isinstance(descriptor, ArtifactDescriptor):
        return descriptor
    if isinstance(descriptor, Mapping):
        return ArtifactDescriptor.from_mapping(descriptor)
    raise TypeError("descriptor must be a count-only mapping or ArtifactDescriptor")


def _resolve_budget(
    budget: ArtifactSizeBudget | Mapping[str, Any],
) -> ArtifactSizeBudget:
    if isinstance(budget, ArtifactSizeBudget):
        return budget
    if isinstance(budget, Mapping):
        return ArtifactSizeBudget.from_mapping(budget)
    raise TypeError("budget must be a count-only mapping or ArtifactSizeBudget")


def evaluate_artifact_size_budget(
    descriptor: ArtifactDescriptor | Mapping[str, Any],
    budget: ArtifactSizeBudget | Mapping[str, Any],
) -> ArtifactSizeBudgetResult:
    """Evaluate a synthetic artifact descriptor against inclusive limits.

    The function reads only numeric descriptor fields and returns all exceeded
    dimensions in a stable order.  It never opens or serializes artifact
    content, and it does not perform a network call.
    """

    resolved_descriptor = _resolve_descriptor(descriptor)
    resolved_budget = _resolve_budget(budget)
    observed: dict[str, int] = {
        "total_bytes": resolved_descriptor.effective_total_bytes,
        "section_bytes": max(
            (section.size_bytes for section in resolved_descriptor.sections),
            default=0,
        ),
        "record_count": resolved_descriptor.effective_record_count,
        "nesting_depth": resolved_descriptor.effective_nesting_depth,
    }
    limits: dict[str, int | None] = {
        "total_bytes": resolved_budget.max_total_bytes,
        "section_bytes": resolved_budget.max_section_bytes,
        "record_count": resolved_budget.max_record_count,
        "nesting_depth": resolved_budget.max_nesting_depth,
    }
    violations = tuple(
        ArtifactSizeBudgetViolation(
            category=category,
            observed=observed[category],
            limit=limit,
        )
        for category in _CATEGORIES
        if (limit := limits[category]) is not None and observed[category] > limit
    )
    return ArtifactSizeBudgetResult(
        within_budget=not violations,
        observed=observed,
        limits=limits,
        violations=violations,
    )


check_artifact_size_budget = evaluate_artifact_size_budget


__all__ = [
    "ArtifactBudgetCategory",
    "ArtifactDescriptor",
    "ArtifactSectionDescriptor",
    "ArtifactSizeBudget",
    "ArtifactSizeBudgetResult",
    "ArtifactSizeBudgetViolation",
    "check_artifact_size_budget",
    "evaluate_artifact_size_budget",
]
