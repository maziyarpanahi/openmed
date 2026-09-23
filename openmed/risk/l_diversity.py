"""Offline l-diversity and t-closeness checks for tabular records.

The checker measures sensitive-attribute disclosure within the equivalence
classes formed by declared quasi-identifiers. It does not generalize,
suppress, or otherwise transform records. Variational distance is used for
t-closeness: it is the total-variation distance between a class distribution
and the global distribution. Numeric sensitive values are therefore treated
as categorical values; earth-mover variants are intentionally out of scope.

Reports contain row offsets, class hashes, and aggregate metrics only. Raw
quasi-identifier and sensitive values are used in-process by the canonical
tabular measurement layer but are never returned by this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.core.audit import stable_hash

from .kanon import _validated_columns, kanon_report

__all__ = [
    "DiversityClass",
    "LDiversityChecker",
    "LDiversityEngine",
    "LDiversityReport",
    "analyze_l_diversity",
    "check_l_diversity",
    "l_diversity_report",
]

_SUPPORTED_L_METRICS = ("distinct", "entropy")
_SUPPORTED_T_DISTANCES = ("variational",)
_COMPARISON_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DiversityClass:
    """Privacy-safe l-diversity and t-closeness values for one class."""

    class_hash: str
    size: int
    row_indices: tuple[int, ...]
    distinct: int
    entropy: float
    t_closeness: float
    l_violation: bool
    t_violation: bool

    @property
    def meets_l(self) -> bool:
        """Return whether this class meets the configured l threshold."""
        return not self.l_violation

    @property
    def meets_t(self) -> bool:
        """Return whether this class meets the configured t threshold."""
        return not self.t_violation

    @property
    def violates(self) -> bool:
        """Return whether this class violates either configured threshold."""
        return self.l_violation or self.t_violation

    @property
    def l_diversity(self) -> dict[str, int | float]:
        """Return both supported l-diversity measurements."""
        return {"distinct": self.distinct, "entropy": self.entropy}

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable class description."""
        violations = []
        if self.l_violation:
            violations.append("l")
        if self.t_violation:
            violations.append("t")
        return {
            "class_hash": self.class_hash,
            "size": int(self.size),
            "row_indices": list(self.row_indices),
            "distinct": int(self.distinct),
            "entropy": float(self.entropy),
            "t_closeness": float(self.t_closeness),
            "l_diversity": self.l_diversity,
            "l_violation": bool(self.l_violation),
            "t_violation": bool(self.t_violation),
            "violates": bool(self.violates),
            "violations": violations,
        }


@dataclass(frozen=True)
class LDiversityReport(Mapping[str, Any]):
    """Deterministic report for one sensitive attribute and policy."""

    record_count: int
    quasi_identifiers: tuple[str, ...]
    sensitive_attribute: str
    target_l: int
    l_metric: str
    l_threshold: float
    target_t: float
    t_distance: str
    equivalence_classes: tuple[DiversityClass, ...]
    l_violating_classes: tuple[DiversityClass, ...]
    t_violating_classes: tuple[DiversityClass, ...]
    violating_classes: tuple[DiversityClass, ...]
    achieved_distinct: int
    achieved_entropy: float
    achieved_l: int | float
    achieved_t: float
    meets_l: bool
    meets_t: bool
    meets_target: bool

    @property
    def class_count(self) -> int:
        """Return the number of equivalence classes."""
        return len(self.equivalence_classes)

    @property
    def violating_rows(self) -> tuple[int, ...]:
        """Return sorted row offsets belonging to a violating class."""
        return tuple(
            sorted(
                {
                    row_index
                    for equivalence_class in self.violating_classes
                    for row_index in equivalence_class.row_indices
                }
            )
        )

    @property
    def classes(self) -> tuple[DiversityClass, ...]:
        """Alias for :attr:`equivalence_classes`."""
        return self.equivalence_classes

    @property
    def l_violating_class_count(self) -> int:
        """Return the number of classes below the l threshold."""
        return len(self.l_violating_classes)

    @property
    def t_violating_class_count(self) -> int:
        """Return the number of classes above the t threshold."""
        return len(self.t_violating_classes)

    @property
    def violating_class_count(self) -> int:
        """Return the number of classes violating either threshold."""
        return len(self.violating_classes)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable report."""
        classes = [item.to_dict() for item in self.equivalence_classes]
        l_violations = [item.to_dict() for item in self.l_violating_classes]
        t_violations = [item.to_dict() for item in self.t_violating_classes]
        violations = [item.to_dict() for item in self.violating_classes]
        return {
            "record_count": int(self.record_count),
            "quasi_identifiers": list(self.quasi_identifiers),
            "sensitive_attribute": self.sensitive_attribute,
            "target_l": int(self.target_l),
            "l_metric": self.l_metric,
            "l_threshold": float(self.l_threshold),
            "target_t": float(self.target_t),
            "t_distance": self.t_distance,
            "class_count": self.class_count,
            "equivalence_classes": classes,
            "classes": classes,
            "l_violating_classes": l_violations,
            "t_violating_classes": t_violations,
            "violating_classes": violations,
            "l_violating_class_count": self.l_violating_class_count,
            "t_violating_class_count": self.t_violating_class_count,
            "violating_class_count": self.violating_class_count,
            "violating_rows": list(self.violating_rows),
            "achieved_distinct": int(self.achieved_distinct),
            "achieved_entropy": float(self.achieved_entropy),
            "achieved_l": self.achieved_l,
            "achieved_t": float(self.achieved_t),
            "l_diversity": {
                "metric": self.l_metric,
                "achieved": self.achieved_l,
                "threshold": self.l_threshold,
                "violating_classes": self.l_violating_class_count,
                "meets_target": bool(self.meets_l),
            },
            "t_closeness": {
                "distance": self.t_distance,
                "achieved": self.achieved_t,
                "target": self.target_t,
                "violating_classes": self.t_violating_class_count,
                "meets_target": bool(self.meets_t),
            },
            "meets_l": bool(self.meets_l),
            "meets_t": bool(self.meets_t),
            "meets_target": bool(self.meets_target),
        }

    def __getitem__(self, key: str) -> Any:
        """Support read-only mapping access to the serialized report."""
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True, init=False)
class LDiversityEngine:
    """Check l-diversity and t-closeness for one sensitive attribute.

    Args:
        quasi_identifiers: Columns defining equivalence classes. ``None``
            delegates to the canonical risk profiler's auto-detection.
        sensitive_attribute: Column whose values are measured.
        target_l: Minimum distinct-value l threshold. With ``l_metric`` set to
            ``"entropy"``, this is interpreted as the conventional l target
            and compared with ``log2(target_l)`` Shannon bits.
        target_t: Maximum variational distance in the inclusive range [0, 1].
        l_metric: Headline l metric, either ``"distinct"`` or ``"entropy"``.
        t_distance: Currently only ``"variational"`` is supported.
    """

    quasi_identifiers: tuple[str, ...] | None
    sensitive_attribute: str
    target_l: int
    target_t: float
    l_metric: str
    t_distance: str

    def __init__(
        self,
        quasi_identifiers: Sequence[str] | None,
        sensitive_attribute: str | None = None,
        target_l: int = 2,
        target_t: float = 0.2,
        *,
        l_metric: str = "distinct",
        t_distance: str = "variational",
        sensitive_attributes: Sequence[str] | None = None,
    ) -> None:
        normalized_qis = _normalize_quasi_identifiers(quasi_identifiers)
        normalized_sensitive = _resolve_sensitive_attribute(
            sensitive_attribute,
            sensitive_attributes,
        )
        _validate_thresholds(target_l, target_t, l_metric, t_distance)
        if normalized_qis is not None and normalized_sensitive in normalized_qis:
            raise ValueError(
                "sensitive_attribute cannot also be a quasi-identifier: "
                f"{normalized_sensitive!r}"
            )
        object.__setattr__(self, "quasi_identifiers", normalized_qis)
        object.__setattr__(self, "sensitive_attribute", normalized_sensitive)
        object.__setattr__(self, "target_l", target_l)
        object.__setattr__(self, "target_t", float(target_t))
        object.__setattr__(self, "l_metric", l_metric)
        object.__setattr__(self, "t_distance", t_distance)

    def analyze(self, records: Any) -> LDiversityReport:
        """Measure the configured policy over ``records`` in-process."""
        measurement = kanon_report(
            records,
            quasi_identifiers=self.quasi_identifiers,
            sensitive_attributes=(self.sensitive_attribute,),
            l_metric=self.l_metric,
            t_distance=self.t_distance,
        )
        return _report_from_measurement(
            measurement,
            sensitive_attribute=self.sensitive_attribute,
            target_l=self.target_l,
            target_t=self.target_t,
            l_metric=self.l_metric,
            t_distance=self.t_distance,
        )

    def check(self, records: Any) -> LDiversityReport:
        """Alias for :meth:`analyze` using checker-oriented terminology."""
        return self.analyze(records)


LDiversityChecker = LDiversityEngine


def analyze_l_diversity(
    records: Any,
    quasi_identifiers: Sequence[str] | None,
    sensitive_attribute: str | None = None,
    *,
    target_l: int = 2,
    target_t: float = 0.2,
    l_metric: str = "distinct",
    t_distance: str = "variational",
    sensitive_attributes: Sequence[str] | None = None,
) -> LDiversityReport:
    """Return an l-diversity/t-closeness report for one sensitive attribute."""
    return LDiversityEngine(
        quasi_identifiers,
        sensitive_attribute,
        target_l,
        target_t,
        l_metric=l_metric,
        t_distance=t_distance,
        sensitive_attributes=sensitive_attributes,
    ).analyze(records)


def check_l_diversity(
    records: Any,
    quasi_identifiers: Sequence[str] | None,
    sensitive_attribute: str | None = None,
    *,
    target_l: int = 2,
    target_t: float = 0.2,
    l_metric: str = "distinct",
    t_distance: str = "variational",
    sensitive_attributes: Sequence[str] | None = None,
) -> LDiversityReport:
    """Return :func:`analyze_l_diversity` using checker-oriented naming."""
    return analyze_l_diversity(
        records,
        quasi_identifiers,
        sensitive_attribute,
        target_l=target_l,
        target_t=target_t,
        l_metric=l_metric,
        t_distance=t_distance,
        sensitive_attributes=sensitive_attributes,
    )


def l_diversity_report(
    records: Any,
    quasi_identifiers: Sequence[str] | None,
    sensitive_attribute: str | None = None,
    *,
    target_l: int = 2,
    target_t: float = 0.2,
    l_metric: str = "distinct",
    t_distance: str = "variational",
    sensitive_attributes: Sequence[str] | None = None,
) -> LDiversityReport:
    """Return :func:`analyze_l_diversity` using report-oriented naming."""
    return analyze_l_diversity(
        records,
        quasi_identifiers,
        sensitive_attribute,
        target_l=target_l,
        target_t=target_t,
        l_metric=l_metric,
        t_distance=t_distance,
        sensitive_attributes=sensitive_attributes,
    )


def _normalize_quasi_identifiers(
    quasi_identifiers: Sequence[str] | None,
) -> tuple[str, ...] | None:
    normalized = _validated_columns(
        quasi_identifiers,
        name="quasi_identifiers",
        allow_none=True,
    )
    if normalized == ():
        raise ValueError("At least one quasi-identifier must be declared")
    return normalized


def _resolve_sensitive_attribute(
    sensitive_attribute: str | None,
    sensitive_attributes: Sequence[str] | None,
) -> str:
    if sensitive_attribute is not None and sensitive_attributes is not None:
        raise TypeError(
            "provide either sensitive_attribute or sensitive_attributes, not both"
        )
    if sensitive_attribute is None:
        if sensitive_attributes is None:
            raise TypeError("a sensitive_attribute must be declared")
        if isinstance(sensitive_attributes, (str, bytes, bytearray)):
            raise TypeError("sensitive_attributes must be a sequence of column names")
        attributes = tuple(sensitive_attributes)
        if len(attributes) != 1:
            raise ValueError("exactly one sensitive attribute must be declared")
        sensitive_attribute = attributes[0]
    if not isinstance(sensitive_attribute, str) or not sensitive_attribute:
        raise ValueError("sensitive_attribute must be a non-empty column name")
    return sensitive_attribute


def _validate_thresholds(
    target_l: int,
    target_t: float,
    l_metric: str,
    t_distance: str,
) -> None:
    if type(target_l) is not int or target_l < 1:
        raise ValueError("target_l must be an integer >= 1")
    if (
        not isinstance(target_t, (int, float))
        or isinstance(target_t, bool)
        or not math.isfinite(float(target_t))
        or not 0.0 <= float(target_t) <= 1.0
    ):
        raise ValueError("target_t must be between 0.0 and 1.0")
    if l_metric not in _SUPPORTED_L_METRICS:
        raise ValueError(
            f"Unsupported l_metric {l_metric!r}; "
            f"supported: {', '.join(_SUPPORTED_L_METRICS)}."
        )
    if t_distance not in _SUPPORTED_T_DISTANCES:
        raise ValueError(
            f"Unsupported t_distance {t_distance!r}; "
            f"supported: {', '.join(_SUPPORTED_T_DISTANCES)}."
        )


def _report_from_measurement(
    measurement: Mapping[str, Any],
    *,
    sensitive_attribute: str,
    target_l: int,
    target_t: float,
    l_metric: str,
    t_distance: str,
) -> LDiversityReport:
    l_threshold = math.log2(target_l) if l_metric == "entropy" else float(target_l)
    classes: list[DiversityClass] = []
    for raw_class in measurement.get("equivalence_classes", ()):
        if not isinstance(raw_class, Mapping):
            continue
        members = tuple(sorted(int(index) for index in raw_class.get("members", ())))
        l_values = raw_class.get("l_diversity", {})
        if not isinstance(l_values, Mapping):
            l_values = {}
        metrics = l_values.get(sensitive_attribute, {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        distinct = int(metrics.get("distinct", 0))
        entropy = float(metrics.get("entropy", 0.0))
        t_values = raw_class.get("t_closeness", {})
        if not isinstance(t_values, Mapping):
            t_values = {}
        t_closeness = float(t_values.get(sensitive_attribute, 0.0))
        l_value = entropy if l_metric == "entropy" else distinct
        l_violation = l_value + _COMPARISON_TOLERANCE < l_threshold
        t_violation = t_closeness > target_t + _COMPARISON_TOLERANCE
        classes.append(
            DiversityClass(
                class_hash=stable_hash(
                    {
                        "kind": "l-diversity-equivalence-class",
                        "row_indices": members,
                    }
                ),
                size=int(raw_class.get("size", len(members))),
                row_indices=members,
                distinct=distinct,
                entropy=entropy,
                t_closeness=t_closeness,
                l_violation=l_violation,
                t_violation=t_violation,
            )
        )

    classes.sort(key=lambda item: item.row_indices[0] if item.row_indices else -1)
    class_tuple = tuple(classes)
    l_violating = tuple(item for item in class_tuple if item.l_violation)
    t_violating = tuple(item for item in class_tuple if item.t_violation)
    violating = tuple(item for item in class_tuple if item.violates)
    distinct = min((item.distinct for item in class_tuple), default=0)
    entropy = min((item.entropy for item in class_tuple), default=0.0)
    achieved_l: int | float = entropy if l_metric == "entropy" else distinct
    achieved_t = max((item.t_closeness for item in class_tuple), default=0.0)
    has_classes = bool(class_tuple)
    meets_l = has_classes and not l_violating
    meets_t = has_classes and not t_violating
    return LDiversityReport(
        record_count=int(measurement.get("record_count", 0)),
        quasi_identifiers=tuple(
            str(field) for field in measurement.get("quasi_identifiers", ())
        ),
        sensitive_attribute=sensitive_attribute,
        target_l=target_l,
        l_metric=l_metric,
        l_threshold=l_threshold,
        target_t=float(target_t),
        t_distance=t_distance,
        equivalence_classes=class_tuple,
        l_violating_classes=l_violating,
        t_violating_classes=t_violating,
        violating_classes=violating,
        achieved_distinct=distinct,
        achieved_entropy=entropy,
        achieved_l=achieved_l,
        achieved_t=achieved_t,
        meets_l=meets_l,
        meets_t=meets_t,
        meets_target=meets_l and meets_t,
    )
