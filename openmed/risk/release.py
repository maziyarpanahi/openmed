"""Patient-level disclosure-risk assessment and release anonymization.

This module assembles OpenMed's lower-level k-anonymity, l-diversity,
t-closeness, generalization, and suppression primitives into a safe public
workflow. Raw equivalence-class keys and row membership remain in-process.
Public assessment serialization contains aggregate evidence only.

The reports produced here support qualified expert review. They are not an
Expert Determination, a compliance certificate, or a guarantee of zero
re-identification risk.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from statistics import mean, median
from types import MappingProxyType
from typing import Any

from openmed.core.audit import stable_hash

from .kanon import (
    _EMPTY_QI,
    _INTERNAL_QI_TOKEN_PREFIX,
    _MISSING_QI,
    _NULL_QI,
    _build_hierarchy_levels,
    _canonical_decimal_text,
    _coerce_records,
    _field_is_direct_identifier,
    _InternalQIState,
    _transform_record,
    _typed_qi_value,
    _validate_dataframe_temporal_precision,
    build_generalization_hierarchies,
    enforce_kanon,
    kanon_report,
)

__all__ = [
    "AnonymityPolicy",
    "AnonymizationResult",
    "AttributeDisclosureSummary",
    "GeneralizationSummary",
    "ReleaseAssessment",
    "ReleasedOutputValidation",
    "UtilitySummary",
    "anonymize_release",
    "assess_release",
    "release_dataset_digest",
    "release_schema_digest",
    "safe_risk_summary",
    "validate_released_output",
]

_SCHEMA_VERSION = 1
_SUPPORTED_L_METRICS = frozenset({"distinct", "entropy"})
_SUPPORTED_T_DISTANCES = frozenset({"variational"})
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_RISK_CATEGORIES = frozenset(
    {
        "age",
        "date",
        "geography",
        "provider_institution",
        "rare_condition",
        "stable_surrogate",
    }
)
_MISSING = f"{_INTERNAL_QI_TOKEN_PREFIX}state:missing"
_NULL = f"{_INTERNAL_QI_TOKEN_PREFIX}state:null"
_EMPTY = f"{_INTERNAL_QI_TOKEN_PREFIX}state:empty"


@dataclass(frozen=True)
class AnonymityPolicy:
    """Explicit privacy criteria for one intended data release.

    ``target_k`` is deliberately required. OpenMed does not choose a universal
    regulatory threshold on the caller's behalf.
    """

    quasi_identifiers: tuple[str, ...]
    target_k: int
    sensitive_attributes: tuple[str, ...] = ()
    direct_identifiers: tuple[str, ...] = ()
    non_sensitive_attributes: tuple[str, ...] = ()
    excluded_attributes: tuple[str, ...] = ()
    privacy_unit: str | None = None
    target_l: int = 1
    l_metric: str = "distinct"
    target_t: float = 1.0
    t_distance: str = "variational"
    suppression_limit: int | None = None
    suppression_rate: float = 0.0
    max_lattice_nodes: int = 100_000
    max_suppression_subsets: int = 100_000

    def __post_init__(self) -> None:
        qis = _column_tuple(self.quasi_identifiers, name="quasi_identifiers")
        sensitive = _column_tuple(
            self.sensitive_attributes,
            name="sensitive_attributes",
            allow_empty=True,
        )
        direct = _column_tuple(
            self.direct_identifiers,
            name="direct_identifiers",
            allow_empty=True,
        )
        non_sensitive = _column_tuple(
            self.non_sensitive_attributes,
            name="non_sensitive_attributes",
            allow_empty=True,
        )
        excluded = _column_tuple(
            self.excluded_attributes,
            name="excluded_attributes",
            allow_empty=True,
        )
        privacy_unit = _optional_column(self.privacy_unit, name="privacy_unit")
        role_sets = {
            "quasi_identifiers": set(qis),
            "sensitive_attributes": set(sensitive),
            "direct_identifiers": set(direct),
            "non_sensitive_attributes": set(non_sensitive),
            "excluded_attributes": set(excluded),
        }
        role_names = tuple(role_sets)
        for index, left_name in enumerate(role_names):
            for right_name in role_names[index + 1 :]:
                overlap = sorted(role_sets[left_name] & role_sets[right_name])
                if overlap:
                    raise ValueError(
                        f"{left_name} cannot overlap {right_name}: {overlap!r}"
                    )
        if privacy_unit is not None and (
            privacy_unit in role_sets["quasi_identifiers"]
            or privacy_unit in role_sets["sensitive_attributes"]
            or privacy_unit in role_sets["non_sensitive_attributes"]
            or privacy_unit in role_sets["excluded_attributes"]
        ):
            raise ValueError("privacy_unit may only also appear in direct_identifiers")
        if type(self.target_k) is not int or self.target_k < 1:
            raise ValueError("target_k must be an integer >= 1")
        if type(self.target_l) is not int or self.target_l < 1:
            raise ValueError("target_l must be an integer >= 1")
        if self.l_metric not in _SUPPORTED_L_METRICS:
            raise ValueError(
                f"l_metric must be one of {sorted(_SUPPORTED_L_METRICS)!r}"
            )
        if not isinstance(self.target_t, (int, float)) or isinstance(
            self.target_t, bool
        ):
            raise ValueError("target_t must be a number between 0 and 1")
        if (
            not math.isfinite(float(self.target_t))
            or not 0 <= float(self.target_t) <= 1
        ):
            raise ValueError("target_t must be a number between 0 and 1")
        if self.t_distance not in _SUPPORTED_T_DISTANCES:
            raise ValueError(
                f"t_distance must be one of {sorted(_SUPPORTED_T_DISTANCES)!r}"
            )
        if self.target_l > 1 and not sensitive:
            raise ValueError("target_l > 1 requires at least one sensitive attribute")
        if float(self.target_t) < 1.0 and not sensitive:
            raise ValueError("target_t < 1 requires at least one sensitive attribute")
        if self.suppression_limit is not None and (
            type(self.suppression_limit) is not int or self.suppression_limit < 0
        ):
            raise ValueError("suppression_limit must be an integer >= 0")
        if not isinstance(self.suppression_rate, (int, float)) or isinstance(
            self.suppression_rate, bool
        ):
            raise ValueError("suppression_rate must be a number between 0 and 1")
        if (
            not math.isfinite(float(self.suppression_rate))
            or not 0 <= float(self.suppression_rate) <= 1
        ):
            raise ValueError("suppression_rate must be a number between 0 and 1")
        if type(self.max_lattice_nodes) is not int or self.max_lattice_nodes < 1:
            raise ValueError("max_lattice_nodes must be an integer >= 1")
        if (
            type(self.max_suppression_subsets) is not int
            or self.max_suppression_subsets < 1
        ):
            raise ValueError("max_suppression_subsets must be an integer >= 1")

        object.__setattr__(self, "quasi_identifiers", qis)
        object.__setattr__(self, "sensitive_attributes", sensitive)
        object.__setattr__(self, "direct_identifiers", direct)
        object.__setattr__(self, "non_sensitive_attributes", non_sensitive)
        object.__setattr__(self, "excluded_attributes", excluded)
        object.__setattr__(self, "privacy_unit", privacy_unit)
        object.__setattr__(self, "target_t", float(self.target_t))
        object.__setattr__(self, "suppression_rate", float(self.suppression_rate))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe policy description."""
        return {
            "quasi_identifiers": list(self.quasi_identifiers),
            "sensitive_attributes": list(self.sensitive_attributes),
            "direct_identifiers": list(self.direct_identifiers),
            "non_sensitive_attributes": list(self.non_sensitive_attributes),
            "excluded_attributes": list(self.excluded_attributes),
            "privacy_unit": self.privacy_unit,
            "target_k": self.target_k,
            "target_l": self.target_l,
            "l_metric": self.l_metric,
            "target_t": self.target_t,
            "t_distance": self.t_distance,
            "suppression_limit": self.suppression_limit,
            "suppression_rate": self.suppression_rate,
            "max_lattice_nodes": self.max_lattice_nodes,
            "max_suppression_subsets": self.max_suppression_subsets,
        }

    @property
    def digest(self) -> str:
        """Return a stable policy digest."""
        return stable_hash({"kind": "openmed-anonymity-policy", **self.to_dict()})


@dataclass(frozen=True)
class AttributeDisclosureSummary:
    """Aggregate l-diversity and t-closeness evidence for one attribute."""

    attribute: str
    achieved_l: float
    l_threshold: float
    l_metric: str
    violating_l_classes: int
    achieved_t: float
    target_t: float
    t_distance: str
    violating_t_classes: int

    @property
    def meets_l(self) -> bool:
        return self.violating_l_classes == 0

    @property
    def meets_t(self) -> bool:
        return self.violating_t_classes == 0

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate evidence without sensitive values."""
        return {
            "attribute": self.attribute,
            "l_diversity": {
                "metric": self.l_metric,
                "achieved": self.achieved_l,
                "threshold": self.l_threshold,
                "violating_classes": self.violating_l_classes,
                "meets_target": self.meets_l,
            },
            "t_closeness": {
                "distance": self.t_distance,
                "achieved": self.achieved_t,
                "target": self.target_t,
                "violating_classes": self.violating_t_classes,
                "meets_target": self.meets_t,
            },
        }


@dataclass(frozen=True)
class ReleaseAssessment:
    """PHI-safe aggregate disclosure-risk assessment."""

    policy: AnonymityPolicy
    row_count: int
    privacy_unit_count: int
    class_count: int
    achieved_k: int
    class_size_distribution: tuple[tuple[int, int], ...]
    singleton_class_count: int
    singleton_privacy_unit_count: int
    k_violating_class_count: int
    l_violating_class_count: int
    t_violating_class_count: int
    violating_class_count: int
    violating_privacy_unit_count: int
    max_sample_identity_risk: float
    mean_sample_identity_risk: float
    median_sample_identity_risk: float
    p95_sample_identity_risk: float
    attributes: tuple[AttributeDisclosureSummary, ...]
    dataset_digest: str
    policy_digest: str
    warnings: tuple[str, ...] = ()

    @property
    def meets_k(self) -> bool:
        return bool(self.privacy_unit_count) and self.achieved_k >= self.policy.target_k

    @property
    def meets_l(self) -> bool:
        return all(summary.meets_l for summary in self.attributes)

    @property
    def meets_t(self) -> bool:
        return all(summary.meets_t for summary in self.attributes)

    @property
    def meets_policy(self) -> bool:
        return self.meets_k and self.meets_l and self.meets_t

    def to_dict(self) -> dict[str, Any]:
        """Return the allow-listed shareable assessment schema."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact": "deidentification_release_assessment",
            "not_an_expert_determination": True,
            "qualified_expert_review_required": True,
            "row_count": self.row_count,
            "privacy_unit_count": self.privacy_unit_count,
            "privacy_unit_column": self.policy.privacy_unit,
            "quasi_identifiers": list(self.policy.quasi_identifiers),
            "sensitive_attributes": list(self.policy.sensitive_attributes),
            "policy": {
                "target_k": self.policy.target_k,
                "target_l": self.policy.target_l,
                "l_metric": self.policy.l_metric,
                "target_t": self.policy.target_t,
                "t_distance": self.policy.t_distance,
            },
            "k_anonymity": {
                "achieved_k": self.achieved_k,
                "class_count": self.class_count,
                "class_size_distribution": [
                    {"size": size, "class_count": count}
                    for size, count in self.class_size_distribution
                ],
                "singleton_class_count": self.singleton_class_count,
                "singleton_privacy_unit_count": self.singleton_privacy_unit_count,
                "k_violating_class_count": self.k_violating_class_count,
                "l_violating_class_count": self.l_violating_class_count,
                "t_violating_class_count": self.t_violating_class_count,
                "violating_class_count": self.violating_class_count,
                "violating_privacy_unit_count": self.violating_privacy_unit_count,
                "meets_target": self.meets_k,
            },
            "sample_identity_risk": {
                "attacker_model": "prosecutor_exact_match_on_declared_qis",
                "max": self.max_sample_identity_risk,
                "mean": self.mean_sample_identity_risk,
                "median": self.median_sample_identity_risk,
                "p95": self.p95_sample_identity_risk,
                "population_risk_estimated": False,
            },
            "attribute_disclosure": [summary.to_dict() for summary in self.attributes],
            "meets_policy": self.meets_policy,
            "dataset_digest": self.dataset_digest,
            "policy_digest": self.policy_digest,
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the safe report deterministically."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


@dataclass(frozen=True)
class GeneralizationSummary:
    """Aggregate transformation/search evidence."""

    levels: tuple[tuple[str, int, str, float], ...]
    affected_privacy_units: tuple[tuple[str, int], ...]
    affected_qi_cells: tuple[tuple[str, int], ...]
    suppressed_qi_cells: tuple[tuple[str, int], ...]
    search_strategy: str
    search_space_size: int
    nodes_evaluated: int
    max_lattice_nodes: int
    suppression_search_strategy: str
    suppression_subsets_evaluated: int
    suppression_subsets_possible: int | None
    max_suppression_subsets: int
    search_complete: bool
    optimum_proven: bool
    information_loss: float
    generalization_loss: float
    suppression_loss: float
    suppressed_privacy_units: int
    suppressed_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": [
                {
                    "attribute": field,
                    "level": level,
                    "loss": loss,
                    "affected_privacy_unit_count": dict(self.affected_privacy_units)[
                        field
                    ],
                    "affected_cell_count": dict(self.affected_qi_cells)[field],
                    "suppressed_cell_count": dict(self.suppressed_qi_cells)[field],
                }
                for field, level, _name, loss in self.levels
            ],
            "search": {
                "strategy": self.search_strategy,
                "search_space_size": self.search_space_size,
                "nodes_evaluated": self.nodes_evaluated,
                "max_lattice_nodes": self.max_lattice_nodes,
                "suppression_strategy": self.suppression_search_strategy,
                "suppression_subsets_evaluated": (self.suppression_subsets_evaluated),
                "suppression_subsets_possible": self.suppression_subsets_possible,
                "max_suppression_subsets": self.max_suppression_subsets,
                "complete": self.search_complete,
                "optimum_proven": self.optimum_proven,
            },
            "information_loss": self.information_loss,
            "generalization_loss": self.generalization_loss,
            "suppression_loss": self.suppression_loss,
            "suppressed_privacy_units": self.suppressed_privacy_units,
            "suppressed_rows": self.suppressed_rows,
        }


@dataclass(frozen=True)
class UtilitySummary:
    """Aggregate before/after release utility evidence."""

    source_rows: int
    released_rows: int
    source_privacy_units: int
    released_privacy_units: int
    row_suppression_rate: float
    privacy_unit_suppression_rate: float
    quasi_identifier_cells_compared: int
    quasi_identifier_cells_changed: int
    quasi_identifier_cell_change_rate: float
    direct_identifier_cells_removed: int
    missing_qi_cells_before: int
    missing_qi_cells_after: int
    mean_qi_distribution_shift: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rows": self.source_rows,
            "released_rows": self.released_rows,
            "source_privacy_units": self.source_privacy_units,
            "released_privacy_units": self.released_privacy_units,
            "row_suppression_rate": self.row_suppression_rate,
            "privacy_unit_suppression_rate": self.privacy_unit_suppression_rate,
            "quasi_identifier_cells_compared": self.quasi_identifier_cells_compared,
            "quasi_identifier_cells_changed": self.quasi_identifier_cells_changed,
            "quasi_identifier_cell_change_rate": (
                self.quasi_identifier_cell_change_rate
            ),
            "direct_identifier_cells_removed": self.direct_identifier_cells_removed,
            "missing_qi_cells_before": self.missing_qi_cells_before,
            "missing_qi_cells_after": self.missing_qi_cells_after,
            "mean_qi_distribution_shift": self.mean_qi_distribution_shift,
        }


@dataclass(frozen=True)
class AnonymizationResult:
    """Sensitive transformed rows plus a separately safe release summary."""

    policy: AnonymityPolicy
    records: tuple[Mapping[str, Any], ...] = field(repr=False)
    before: ReleaseAssessment
    after: ReleaseAssessment
    generalization: GeneralizationSummary
    utility: UtilitySummary
    source_dataset_digest: str
    released_dataset_digest: str
    released_schema_digest: str
    hierarchy_digest: str
    _privacy_unit_membership: tuple[int, ...] = field(repr=False, compare=False)
    _binding_digest: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, AnonymityPolicy):
            raise TypeError("policy must be an AnonymityPolicy")
        if not isinstance(self.before, ReleaseAssessment) or not isinstance(
            self.after,
            ReleaseAssessment,
        ):
            raise TypeError("before and after must be ReleaseAssessment values")
        if not isinstance(self.generalization, GeneralizationSummary):
            raise TypeError("generalization must be a GeneralizationSummary")
        if not isinstance(self.utility, UtilitySummary):
            raise TypeError("utility must be a UtilitySummary")
        if not isinstance(self.records, tuple) or not all(
            isinstance(row, Mapping) for row in self.records
        ):
            raise TypeError("records must be a tuple of row mappings")
        frozen_records = tuple(MappingProxyType(dict(row)) for row in self.records)
        if _dataset_digest(frozen_records) != self.released_dataset_digest:
            raise ValueError("released_dataset_digest does not match records")
        if _schema_digest(frozen_records) != self.released_schema_digest:
            raise ValueError("released_schema_digest does not match records")
        for name, digest in (
            ("source_dataset_digest", self.source_dataset_digest),
            ("released_dataset_digest", self.released_dataset_digest),
            ("released_schema_digest", self.released_schema_digest),
            ("hierarchy_digest", self.hierarchy_digest),
            ("binding_digest", self._binding_digest),
        ):
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise ValueError(f"{name} must be a canonical SHA-256 digest")
        _validate_result_membership(
            self._privacy_unit_membership,
            row_count=len(frozen_records),
            row_level=self.policy.privacy_unit is None,
        )
        _validate_result_summary_bindings(
            policy=self.policy,
            before=self.before,
            after=self.after,
            generalization=self.generalization,
            utility=self.utility,
            source_dataset_digest=self.source_dataset_digest,
            released_row_count=len(frozen_records),
            released_privacy_unit_count=len(set(self._privacy_unit_membership)),
        )
        remeasured = _remeasure_released_records(
            frozen_records,
            self._privacy_unit_membership,
            self.policy,
        )
        if _comparable_assessment(remeasured) != _comparable_assessment(self.after):
            raise ValueError("after assessment does not match released records")
        expected_binding = _result_binding_digest(
            policy=self.policy,
            before=self.before,
            after=self.after,
            generalization=self.generalization,
            utility=self.utility,
            source_dataset_digest=self.source_dataset_digest,
            released_dataset_digest=self.released_dataset_digest,
            released_schema_digest=self.released_schema_digest,
            hierarchy_digest=self.hierarchy_digest,
            privacy_unit_membership=self._privacy_unit_membership,
        )
        if expected_binding != self._binding_digest:
            raise ValueError("anonymization result binding is inconsistent")
        object.__setattr__(self, "records", frozen_records)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return aggregate evidence without transformed or source rows."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact": "deidentification_anonymization_summary",
            "not_an_expert_determination": True,
            "qualified_expert_review_required": True,
            "policy": self.policy.to_dict(),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "generalization": self.generalization.to_dict(),
            "utility": self.utility.to_dict(),
            "source_dataset_digest": self.source_dataset_digest,
            "released_dataset_digest": self.released_dataset_digest,
            "released_schema_digest": self.released_schema_digest,
            "hierarchy_digest": self.hierarchy_digest,
        }

    def to_safe_json(self, *, indent: int | None = 2) -> str:
        """Serialize only the safe aggregate summary."""
        return json.dumps(
            self.to_safe_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


@dataclass(frozen=True)
class ReleasedOutputValidation:
    """Aggregate validation of a materialized release artifact."""

    row_count: int
    expected_row_count: int
    dataset_digest: str
    expected_digest: str | None
    schema_digest: str
    expected_schema_digest: str
    direct_identifier_columns: tuple[str, ...]
    policy_revalidated_before_identifier_removal: bool
    typed_digest_comparison_available: bool
    policy_value_encoding_preserved: bool

    @property
    def row_count_matches(self) -> bool:
        return self.row_count == self.expected_row_count

    @property
    def digest_matches(self) -> bool | None:
        if self.expected_digest is None:
            return None
        return self.dataset_digest == self.expected_digest

    @property
    def schema_matches(self) -> bool:
        return self.schema_digest == self.expected_schema_digest

    @property
    def passed(self) -> bool:
        return (
            self.row_count_matches
            and not self.direct_identifier_columns
            and self.policy_revalidated_before_identifier_removal
            and self.policy_value_encoding_preserved
            and self.schema_matches
            and self.digest_matches is True
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "expected_row_count": self.expected_row_count,
            "row_count_matches": self.row_count_matches,
            "dataset_digest": self.dataset_digest,
            "expected_digest": self.expected_digest,
            "digest_matches": self.digest_matches,
            "schema_digest": self.schema_digest,
            "expected_schema_digest": self.expected_schema_digest,
            "schema_matches": self.schema_matches,
            "typed_digest_comparison_available": (
                self.typed_digest_comparison_available
            ),
            "policy_value_encoding_preserved": (self.policy_value_encoding_preserved),
            "direct_identifier_columns": list(self.direct_identifier_columns),
            "policy_revalidated_before_identifier_removal": (
                self.policy_revalidated_before_identifier_removal
            ),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class _SubjectProjection:
    token: str
    row_indices: tuple[int, ...]
    record: Mapping[str, Any]
    multi_valued_qis: tuple[str, ...]
    multi_valued_sensitive_attributes: tuple[str, ...]


def assess_release(
    records: Any,
    policy: AnonymityPolicy,
) -> ReleaseAssessment:
    """Assess a release against explicit k/l/t criteria.

    Measurement is performed over the declared privacy unit. When no
    ``privacy_unit`` column is configured, each row is treated as one unit and
    the report says so explicitly.
    """

    rows = _materialize_rows(records)
    _validate_policy_columns(rows, policy)
    projections = _project_privacy_units(rows, policy)
    _validate_multi_valued_sensitive_attributes(projections, policy)
    subject_rows = [dict(projection.record) for projection in projections]
    measurement = kanon_report(
        subject_rows,
        quasi_identifiers=policy.quasi_identifiers,
        sensitive_attributes=policy.sensitive_attributes,
        l_metric=policy.l_metric,
        t_distance=policy.t_distance,
    )
    warnings = _assessment_warnings(rows, projections, policy)
    return _safe_assessment(
        measurement,
        rows=rows,
        policy=policy,
        warnings=warnings,
    )


def anonymize_release(
    records: Any,
    policy: AnonymityPolicy,
    *,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> AnonymizationResult:
    """Generalize/suppress a release and revalidate it over privacy units.

    Suppression is applied to complete privacy units: if one patient-level
    projection is suppressed, every source row for that patient is removed.
    The transformed records are sensitive output and are never included in
    :meth:`AnonymizationResult.to_safe_dict`.
    """

    rows = _materialize_rows(records)
    if not rows:
        raise ValueError("Cannot anonymize an empty release")
    _validate_policy_columns(rows, policy)
    before = assess_release(rows, policy)
    projections = _project_privacy_units(rows, policy)
    _validate_multi_valued_sensitive_attributes(projections, policy)
    subject_rows = [dict(projection.record) for projection in projections]

    enforced = enforce_kanon(
        subject_rows,
        quasi_identifiers=policy.quasi_identifiers,
        sensitive_attributes=policy.sensitive_attributes,
        target_k=policy.target_k,
        target_l=policy.target_l,
        target_t=policy.target_t,
        suppression_limit=policy.suppression_limit,
        suppression_rate=policy.suppression_rate,
        hierarchies=hierarchies,
        remove_direct_identifiers=False,
        l_metric=policy.l_metric,
        max_lattice_nodes=policy.max_lattice_nodes,
        max_suppression_subsets=policy.max_suppression_subsets,
    )
    suppressed_subject_positions = {
        int(item["record_index"]) for item in enforced["suppressed_records"]
    }
    suppressed_row_indices = {
        row_index
        for position in suppressed_subject_positions
        for row_index in projections[position].row_indices
    }

    transformed_with_units = _apply_generalization_to_rows(
        rows,
        policy,
        subject_rows=subject_rows,
        enforced=enforced,
        hierarchies=hierarchies,
        suppressed_row_indices=suppressed_row_indices,
    )
    if not transformed_with_units:
        raise ValueError("Anonymization would produce an empty release")

    after = assess_release(transformed_with_units, policy)
    if not after.meets_policy:
        raise ValueError(
            "The transformed release failed full privacy-unit validation. "
            "Provide explicit hierarchies for multi-valued quasi-identifiers "
            "or tighten the suppression policy."
        )

    released_records = tuple(
        _strip_release_identifiers(
            row,
            privacy_unit=policy.privacy_unit,
            direct_identifiers=policy.direct_identifiers,
            excluded_attributes=policy.excluded_attributes,
        )
        for row in transformed_with_units
    )
    if any(
        policy.privacy_unit is not None and policy.privacy_unit in row
        for row in released_records
    ):
        raise AssertionError("privacy-unit identifiers must not enter release output")

    affected_qi_cells, suppressed_qi_cells = _transformation_cell_counts(
        rows,
        transformed_with_units,
        policy,
        hierarchies=hierarchies,
        suppressed_row_indices=suppressed_row_indices,
    )
    generalization = _generalization_summary(
        enforced,
        affected_privacy_units=_affected_privacy_unit_counts(
            subject_rows,
            policy,
            enforced=enforced,
            hierarchies=hierarchies,
            suppressed_positions=suppressed_subject_positions,
        ),
        affected_qi_cells=affected_qi_cells,
        suppressed_qi_cells=suppressed_qi_cells,
        suppressed_privacy_units=len(suppressed_subject_positions),
        suppressed_rows=len(suppressed_row_indices),
    )
    utility = _utility_summary(
        source_rows=rows,
        released_rows=transformed_with_units,
        policy=policy,
        before=before,
        after=after,
        generalization=generalization,
    )
    source_dataset_digest = _dataset_digest(rows)
    released_dataset_digest = _dataset_digest(released_records)
    released_schema_digest = _schema_digest(released_records)
    hierarchy_digest = _hierarchy_digest(
        subject_rows,
        policy.quasi_identifiers,
        hierarchies,
    )
    privacy_unit_membership = _release_privacy_unit_membership(
        transformed_with_units,
        policy.privacy_unit,
    )
    binding_digest = _result_binding_digest(
        policy=policy,
        before=before,
        after=after,
        generalization=generalization,
        utility=utility,
        source_dataset_digest=source_dataset_digest,
        released_dataset_digest=released_dataset_digest,
        released_schema_digest=released_schema_digest,
        hierarchy_digest=hierarchy_digest,
        privacy_unit_membership=privacy_unit_membership,
    )
    return AnonymizationResult(
        policy=policy,
        records=released_records,
        before=before,
        after=after,
        generalization=generalization,
        utility=utility,
        source_dataset_digest=source_dataset_digest,
        released_dataset_digest=released_dataset_digest,
        released_schema_digest=released_schema_digest,
        hierarchy_digest=hierarchy_digest,
        _privacy_unit_membership=privacy_unit_membership,
        _binding_digest=binding_digest,
    )


def safe_risk_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a detailed ``risk_report`` payload to aggregate safe fields.

    Detailed risk reports contain raw and normalized quasi-identifiers and must
    remain local-sensitive. This helper intentionally copies no unknown field.
    """

    quasi_identifiers = [
        item
        for item in report.get("quasi_identifiers", [])
        if isinstance(item, Mapping)
    ]
    category_counts = Counter(
        (
            str(item.get("category"))
            if isinstance(item.get("category"), str)
            and item.get("category") in _SAFE_RISK_CATEGORIES
            else "unknown"
        )
        for item in quasi_identifiers
    )
    singletons = [
        item
        for item in report.get("singleton_records", [])
        if isinstance(item, Mapping)
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact": "deidentification_risk_summary",
        "detail_level": "aggregate_phi_safe",
        "record_count": _safe_int(report.get("record_count")),
        "leakage_rate": _safe_float(report.get("leakage_rate")),
        "reidentification_rate": _safe_float(report.get("reid_rate")),
        "minimum_k": _safe_int(report.get("k_min")),
        "singleton_record_count": len(singletons),
        "quasi_identifier_count": len(quasi_identifiers),
        "quasi_identifier_categories": dict(sorted(category_counts.items())),
    }


def release_dataset_digest(records: Any) -> str:
    """Return the canonical digest used to bind release evidence to rows."""

    return _dataset_digest(records)


def release_schema_digest(records: Any) -> str:
    """Return an aggregate schema digest for a materialized release."""

    return _schema_digest(records)


def validate_released_output(
    records: Any,
    result: AnonymizationResult,
    *,
    preserve_scalar_types: bool = True,
) -> ReleasedOutputValidation:
    """Validate a reread release without exposing any row values.

    CSV and TSV do not preserve scalar types. Set ``preserve_scalar_types`` to
    false for those formats; the validator compares a canonical string-valued
    digest matching delimited-file semantics, while reporting that a typed
    digest comparison was not available.
    """

    if not isinstance(result, AnonymizationResult):
        raise TypeError("result must be an AnonymizationResult")
    rows = _materialize_rows(records)
    fields = {str(field) for row in rows for field in row}
    forbidden = {
        *result.policy.direct_identifiers,
        *result.policy.excluded_attributes,
        *(
            [result.policy.privacy_unit]
            if result.policy.privacy_unit is not None
            else []
        ),
        *{field for field in fields if _field_is_direct_identifier(field)},
    }
    expected_rows = (
        result.records
        if preserve_scalar_types
        else _stringified_scalar_rows(result.records)
    )
    return ReleasedOutputValidation(
        row_count=len(rows),
        expected_row_count=len(result.records),
        dataset_digest=_dataset_digest(rows),
        expected_digest=(
            result.released_dataset_digest
            if preserve_scalar_types
            else _dataset_digest(expected_rows)
        ),
        schema_digest=_schema_digest(rows),
        expected_schema_digest=_schema_digest(expected_rows),
        direct_identifier_columns=tuple(sorted(fields & forbidden)),
        policy_revalidated_before_identifier_removal=result.after.meets_policy,
        typed_digest_comparison_available=preserve_scalar_types,
        policy_value_encoding_preserved=(
            preserve_scalar_types or _delimited_policy_encoding_is_injective(result)
        ),
    )


def _safe_assessment(
    measurement: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
    warnings: tuple[str, ...],
) -> ReleaseAssessment:
    classes = [
        item
        for item in measurement.get("equivalence_classes", [])
        if isinstance(item, Mapping)
    ]
    class_sizes = [int(item.get("size", 0)) for item in classes]
    risks = [1.0 / size for size in class_sizes if size > 0 for _ in range(size)]
    violating_classes = [
        item for item in classes if not _class_meets_policy(item, policy)
    ]
    k_violating_classes = [
        item for item in classes if int(item.get("size", 0)) < policy.target_k
    ]
    l_violating_classes = [item for item in classes if not _class_meets_l(item, policy)]
    t_violating_classes = [item for item in classes if not _class_meets_t(item, policy)]
    attributes = tuple(
        _attribute_summary(classes, attribute, policy)
        for attribute in policy.sensitive_attributes
    )
    privacy_unit_count = sum(class_sizes)
    return ReleaseAssessment(
        policy=policy,
        row_count=len(rows),
        privacy_unit_count=privacy_unit_count,
        class_count=len(classes),
        achieved_k=int(measurement.get("k", 0)),
        class_size_distribution=tuple(
            (int(size), int(count))
            for size, count in measurement.get("class_size_distribution", [])
        ),
        singleton_class_count=sum(1 for size in class_sizes if size == 1),
        singleton_privacy_unit_count=sum(size for size in class_sizes if size == 1),
        k_violating_class_count=len(k_violating_classes),
        l_violating_class_count=len(l_violating_classes),
        t_violating_class_count=len(t_violating_classes),
        violating_class_count=len(violating_classes),
        violating_privacy_unit_count=sum(
            int(item.get("size", 0)) for item in violating_classes
        ),
        max_sample_identity_risk=max(risks, default=0.0),
        mean_sample_identity_risk=mean(risks) if risks else 0.0,
        median_sample_identity_risk=median(risks) if risks else 0.0,
        p95_sample_identity_risk=_percentile(risks, 0.95),
        attributes=attributes,
        dataset_digest=_dataset_digest(rows),
        policy_digest=policy.digest,
        warnings=warnings,
    )


def _attribute_summary(
    classes: Sequence[Mapping[str, Any]],
    attribute: str,
    policy: AnonymityPolicy,
) -> AttributeDisclosureSummary:
    l_values: list[float] = []
    t_values: list[float] = []
    for equivalence_class in classes:
        l_by_attribute = equivalence_class.get("l_diversity", {})
        t_by_attribute = equivalence_class.get("t_closeness", {})
        l_entry = (
            l_by_attribute.get(attribute, {})
            if isinstance(l_by_attribute, Mapping)
            else {}
        )
        if policy.l_metric == "entropy":
            l_values.append(float(l_entry.get("entropy", 0.0)))
        else:
            l_values.append(float(l_entry.get("distinct", 0)))
        t_values.append(
            float(t_by_attribute.get(attribute, 0.0))
            if isinstance(t_by_attribute, Mapping)
            else 0.0
        )
    l_threshold = (
        math.log2(policy.target_l)
        if policy.l_metric == "entropy"
        else float(policy.target_l)
    )
    return AttributeDisclosureSummary(
        attribute=attribute,
        achieved_l=min(l_values, default=0.0),
        l_threshold=l_threshold,
        l_metric=policy.l_metric,
        violating_l_classes=sum(1 for value in l_values if value + 1e-12 < l_threshold),
        achieved_t=max(t_values, default=0.0),
        target_t=policy.target_t,
        t_distance=policy.t_distance,
        violating_t_classes=sum(
            1 for value in t_values if value > policy.target_t + 1e-12
        ),
    )


def _class_meets_policy(
    equivalence_class: Mapping[str, Any],
    policy: AnonymityPolicy,
) -> bool:
    if int(equivalence_class.get("size", 0)) < policy.target_k:
        return False
    return _class_meets_l(equivalence_class, policy) and _class_meets_t(
        equivalence_class,
        policy,
    )


def _class_meets_l(
    equivalence_class: Mapping[str, Any],
    policy: AnonymityPolicy,
) -> bool:
    l_by_attribute = equivalence_class.get("l_diversity", {})
    for attribute in policy.sensitive_attributes:
        l_entry = (
            l_by_attribute.get(attribute, {})
            if isinstance(l_by_attribute, Mapping)
            else {}
        )
        if policy.l_metric == "entropy":
            achieved_l = float(l_entry.get("entropy", 0.0))
            required_l = math.log2(policy.target_l)
        else:
            achieved_l = float(l_entry.get("distinct", 0))
            required_l = float(policy.target_l)
        if achieved_l + 1e-12 < required_l:
            return False
    return True


def _class_meets_t(
    equivalence_class: Mapping[str, Any],
    policy: AnonymityPolicy,
) -> bool:
    t_by_attribute = equivalence_class.get("t_closeness", {})
    for attribute in policy.sensitive_attributes:
        achieved_t = (
            float(t_by_attribute.get(attribute, 0.0))
            if isinstance(t_by_attribute, Mapping)
            else 0.0
        )
        if achieved_t > policy.target_t + 1e-12:
            return False
    return True


def _project_privacy_units(
    rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
) -> list[_SubjectProjection]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    if policy.privacy_unit is None:
        for index in range(len(rows)):
            grouped[f"row:{index}"].append(index)
    else:
        for index, row in enumerate(rows):
            if policy.privacy_unit not in row:
                raise ValueError(
                    f"privacy_unit column is missing from row offset {index}"
                )
            value = row[policy.privacy_unit]
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(
                    f"privacy_unit is empty at row offset {index}; "
                    "every row must map to one person"
                )
            if isinstance(value, str) and value != value.strip():
                raise ValueError(
                    f"privacy_unit has surrounding whitespace at row offset {index}; "
                    "canonicalize identifiers before patient-level measurement"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"privacy_unit must be scalar at row offset {index}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"privacy_unit must be finite at row offset {index}")
            grouped[_exact_typed_token(value)].append(index)

    projections: list[_SubjectProjection] = []
    for token, row_indices in grouped.items():
        unit_rows = [rows[index] for index in row_indices]
        record, multi_valued_qis = _projected_joint_qi_values(
            unit_rows,
            policy.quasi_identifiers,
        )
        multi_valued_sensitive: list[str] = []
        for field in policy.sensitive_attributes:
            value, is_multi = _projected_value(
                unit_rows,
                field,
                quasi_identifier=False,
            )
            record[field] = value
            if is_multi:
                multi_valued_sensitive.append(field)
        projections.append(
            _SubjectProjection(
                token=token,
                row_indices=tuple(row_indices),
                record=record,
                multi_valued_qis=tuple(sorted(multi_valued_qis)),
                multi_valued_sensitive_attributes=tuple(sorted(multi_valued_sensitive)),
            )
        )
    return projections


def _projected_joint_qi_values(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    joint_rows: list[tuple[tuple[str, ...], tuple[Any, ...]]] = []
    for row in rows:
        values = tuple(_projected_qi_cell(row, field) for field in fields)
        joint_rows.append(
            (
                tuple(
                    _typed_qi_value(field, value)
                    for field, value in zip(fields, values, strict=True)
                ),
                values,
            )
        )
    joint_rows.sort(key=lambda item: item[0])

    projected: dict[str, Any] = {}
    multi_valued: list[str] = []
    for index, field in enumerate(fields):
        ordered_values = tuple(values[index] for _tokens, values in joint_rows)
        if len(ordered_values) == 1:
            projected[field] = ordered_values[0]
        else:
            projected[field] = _InternalQIState(
                "ordered-multiset",
                ordered_values,
            )
        if len({tokens[index] for tokens, _values in joint_rows}) > 1:
            multi_valued.append(field)
    return projected, multi_valued


def _projected_qi_cell(row: Mapping[str, Any], field: str) -> Any:
    if field not in row:
        return _MISSING_QI
    value = row[field]
    if value is None:
        return _NULL_QI
    if isinstance(value, str) and not value:
        return _EMPTY_QI
    return value


def _validate_multi_valued_sensitive_attributes(
    projections: Sequence[_SubjectProjection],
    policy: AnonymityPolicy,
) -> None:
    fields = sorted(
        {
            field
            for projection in projections
            for field in projection.multi_valued_sensitive_attributes
        }
    )
    if fields and (policy.target_l > 1 or policy.target_t < 1.0):
        raise ValueError(
            "Multi-valued sensitive attributes within one privacy unit are not "
            "supported by this l-diversity/t-closeness model; normalize to one "
            f"reviewed value per privacy unit or use a dedicated model: {fields!r}"
        )


def _projected_value(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    quasi_identifier: bool,
) -> tuple[Any, bool]:
    values: list[str] = []
    original_qi_values: dict[str, Any] = {}
    original_sensitive_values: dict[str, Any] = {}
    for row in rows:
        if field not in row:
            values.append(_typed_qi_value(field, _MISSING_QI))
            continue
        value = row[field]
        if value is None:
            values.append(_typed_qi_value(field, _NULL_QI))
        elif isinstance(value, str) and not value:
            values.append(_typed_qi_value(field, _EMPTY_QI))
        elif quasi_identifier:
            token = _typed_qi_value(field, value)
            values.append(token)
            original_qi_values.setdefault(token, value)
        else:
            token = _exact_typed_token(value)
            values.append(token)
            original_sensitive_values.setdefault(token, value)
    unique = sorted(set(values))
    if len(unique) == 1:
        token = unique[0]
        if quasi_identifier and token in original_qi_values:
            return original_qi_values[token], False
        if quasi_identifier:
            payload = json.loads(
                token.removeprefix(_INTERNAL_QI_TOKEN_PREFIX + "typed:")
            )
            state = str(payload.get("type"))
            return {
                "missing": _MISSING_QI,
                "null": _NULL_QI,
                "empty": _EMPTY_QI,
            }[state], False
        return original_sensitive_values[token], False
    if quasi_identifier:
        return _InternalQIState("set", tuple(unique)), True
    return (
        _INTERNAL_QI_TOKEN_PREFIX
        + "sensitive-set:"
        + json.dumps(unique, ensure_ascii=False, separators=(",", ":")),
        True,
    )


def _apply_generalization_to_rows(
    rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
    *,
    subject_rows: Sequence[Mapping[str, Any]],
    enforced: Mapping[str, Any],
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    suppressed_row_indices: set[int],
) -> list[dict[str, Any]]:
    coerced_subjects = _coerce_records(subject_rows, source="deidentified")
    levels = _build_hierarchy_levels(
        coerced_subjects,
        policy.quasi_identifiers,
        hierarchies,
    )
    node_mapping = enforced.get("generalization", {}).get("node", {})
    if not isinstance(node_mapping, Mapping):
        raise ValueError("Anonymization result is missing its generalization node")
    node = tuple(int(node_mapping[field]) for field in policy.quasi_identifiers)

    transformed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in suppressed_row_indices:
            continue
        record = _coerce_records([row], source="deidentified")[0]
        transformed_fields = _transform_record(
            record,
            policy.quasi_identifiers,
            levels,
            node,
            # Keep identifiers only in the in-process validation copy. They
            # are removed from the materialized release after full residual
            # risk remeasurement succeeds.
            remove_direct_identifiers=False,
        )
        output = dict(row)
        for field in policy.quasi_identifiers:
            output[field] = transformed_fields[field]
        if policy.privacy_unit is not None:
            output[policy.privacy_unit] = row[policy.privacy_unit]
        transformed.append(output)
    return transformed


def _strip_release_identifiers(
    row: Mapping[str, Any],
    *,
    privacy_unit: str | None,
    direct_identifiers: Sequence[str],
    excluded_attributes: Sequence[str],
) -> dict[str, Any]:
    return {
        str(field): value
        for field, value in row.items()
        if field != privacy_unit
        and str(field) not in direct_identifiers
        and str(field) not in excluded_attributes
        and not _field_is_direct_identifier(str(field))
    }


def _transformation_cell_counts(
    source_rows: Sequence[Mapping[str, Any]],
    transformed_rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
    *,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    suppressed_row_indices: set[int],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Count QI cells changed and replaced by the suppression marker."""

    retained_source = [
        row
        for index, row in enumerate(source_rows)
        if index not in suppressed_row_indices
    ]
    if len(retained_source) != len(transformed_rows):
        raise AssertionError("transformed rows must align with retained source rows")
    coerced = _coerce_records(retained_source, source="deidentified")
    levels = _build_hierarchy_levels(
        coerced,
        policy.quasi_identifiers,
        hierarchies,
    )
    exact_node = tuple(0 for _field in policy.quasi_identifiers)
    affected = {field: 0 for field in policy.quasi_identifiers}
    suppressed = {field: 0 for field in policy.quasi_identifiers}
    for source, transformed in zip(coerced, transformed_rows):
        exact = _transform_record(
            source,
            policy.quasi_identifiers,
            levels,
            exact_node,
            remove_direct_identifiers=False,
        )
        for field in policy.quasi_identifiers:
            changed = _exact_typed_token(exact[field]) != _exact_typed_token(
                transformed[field]
            )
            affected[field] += int(changed)
            suppressed[field] += int(changed and transformed[field] == "*")
    return (
        tuple((field, affected[field]) for field in policy.quasi_identifiers),
        tuple((field, suppressed[field]) for field in policy.quasi_identifiers),
    )


def _generalization_summary(
    enforced: Mapping[str, Any],
    *,
    affected_privacy_units: tuple[tuple[str, int], ...],
    affected_qi_cells: tuple[tuple[str, int], ...],
    suppressed_qi_cells: tuple[tuple[str, int], ...],
    suppressed_privacy_units: int,
    suppressed_rows: int,
) -> GeneralizationSummary:
    generalization = enforced.get("generalization", {})
    if not isinstance(generalization, Mapping):
        generalization = {}
    level_mapping = generalization.get("levels", {})
    levels = []
    if isinstance(level_mapping, Mapping):
        for field, value in sorted(level_mapping.items()):
            if not isinstance(value, Mapping):
                continue
            levels.append(
                (
                    str(field),
                    int(value.get("level", 0)),
                    str(value.get("name", "unknown")),
                    float(value.get("loss", 0.0)),
                )
            )
    suppression_subsets_possible = generalization.get("suppression_subsets_possible")
    if suppression_subsets_possible is not None:
        suppression_subsets_possible = int(suppression_subsets_possible)
    legacy_optimum = (
        bool(generalization.get("search_complete", False))
        and int(generalization.get("nodes_evaluated", 0))
        == int(generalization.get("search_space_size", 0))
        and suppression_subsets_possible is not None
        and int(generalization.get("suppression_subsets_evaluated", 0))
        == suppression_subsets_possible
        and float(generalization.get("optimality_tolerance", 1.0)) == 0.0
    )
    return GeneralizationSummary(
        levels=tuple(levels),
        affected_privacy_units=affected_privacy_units,
        affected_qi_cells=affected_qi_cells,
        suppressed_qi_cells=suppressed_qi_cells,
        search_strategy=str(generalization.get("search", "unknown")),
        search_space_size=int(generalization.get("search_space_size", 0)),
        nodes_evaluated=int(generalization.get("nodes_evaluated", 0)),
        max_lattice_nodes=int(generalization.get("max_lattice_nodes", 0)),
        suppression_search_strategy=str(
            generalization.get("suppression_search", "unknown")
        ),
        suppression_subsets_evaluated=int(
            generalization.get("suppression_subsets_evaluated", 0)
        ),
        suppression_subsets_possible=suppression_subsets_possible,
        max_suppression_subsets=int(generalization.get("max_suppression_subsets", 0)),
        search_complete=bool(generalization.get("search_complete", False)),
        optimum_proven=bool(
            generalization.get(
                "optimum_proven",
                legacy_optimum,
            )
        ),
        information_loss=float(generalization.get("information_loss", 0.0)),
        generalization_loss=float(generalization.get("generalization_loss", 0.0)),
        suppression_loss=float(generalization.get("suppression_loss", 0.0)),
        suppressed_privacy_units=suppressed_privacy_units,
        suppressed_rows=suppressed_rows,
    )


def _affected_privacy_unit_counts(
    subject_rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
    *,
    enforced: Mapping[str, Any],
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    suppressed_positions: set[int],
) -> tuple[tuple[str, int], ...]:
    coerced = _coerce_records(subject_rows, source="deidentified")
    levels = _build_hierarchy_levels(
        coerced,
        policy.quasi_identifiers,
        hierarchies,
    )
    node_mapping = enforced.get("generalization", {}).get("node", {})
    if not isinstance(node_mapping, Mapping):
        raise ValueError("Anonymization result is missing its generalization node")
    selected_node = tuple(
        int(node_mapping[field]) for field in policy.quasi_identifiers
    )
    exact_node = tuple(0 for _field in policy.quasi_identifiers)
    counts = {field: 0 for field in policy.quasi_identifiers}
    for position, record in enumerate(coerced):
        if position in suppressed_positions:
            continue
        exact = _transform_record(
            record,
            policy.quasi_identifiers,
            levels,
            exact_node,
            remove_direct_identifiers=False,
        )
        selected = _transform_record(
            record,
            policy.quasi_identifiers,
            levels,
            selected_node,
            remove_direct_identifiers=False,
        )
        for index, field in enumerate(policy.quasi_identifiers):
            if selected_node[index] == 0:
                continue
            counts[field] += int(
                _exact_typed_token(exact[field]) != _exact_typed_token(selected[field])
            )
    return tuple((field, counts[field]) for field in policy.quasi_identifiers)


def _utility_summary(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    released_rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
    before: ReleaseAssessment,
    after: ReleaseAssessment,
    generalization: GeneralizationSummary,
) -> UtilitySummary:
    changed = 0
    compared = 0
    missing_before = 0
    missing_after = 0
    source_by_unit = _rows_by_privacy_unit(source_rows, policy.privacy_unit)
    released_by_unit = _rows_by_privacy_unit(released_rows, policy.privacy_unit)
    for token, output_rows in released_by_unit.items():
        input_rows = source_by_unit.get(token, [])
        for input_row, output_row in zip(input_rows, output_rows):
            for field in policy.quasi_identifiers:
                compared += 1
                before_value = _normalized_cell(input_row, field)
                after_value = _normalized_cell(output_row, field)
                missing_before += before_value in {_MISSING, _NULL, _EMPTY}
                missing_after += after_value in {_MISSING, _NULL, _EMPTY}
                changed += before_value != after_value

    direct_identifier_cells_removed = sum(
        1
        for row in source_rows
        for field in row
        if field == policy.privacy_unit
        or str(field) in policy.direct_identifiers
        or _field_is_direct_identifier(str(field))
    )
    shifts = [
        _distribution_shift(source_rows, released_rows, field)
        for field in policy.quasi_identifiers
    ]
    return UtilitySummary(
        source_rows=len(source_rows),
        released_rows=len(released_rows),
        source_privacy_units=before.privacy_unit_count,
        released_privacy_units=after.privacy_unit_count,
        row_suppression_rate=_rate(
            len(source_rows) - len(released_rows),
            len(source_rows),
        ),
        privacy_unit_suppression_rate=_rate(
            before.privacy_unit_count - after.privacy_unit_count,
            before.privacy_unit_count,
        ),
        quasi_identifier_cells_compared=compared,
        quasi_identifier_cells_changed=changed,
        quasi_identifier_cell_change_rate=_rate(changed, compared),
        direct_identifier_cells_removed=direct_identifier_cells_removed,
        missing_qi_cells_before=missing_before,
        missing_qi_cells_after=missing_after,
        mean_qi_distribution_shift=mean(shifts) if shifts else 0.0,
    )


def _rows_by_privacy_unit(
    rows: Sequence[Mapping[str, Any]],
    privacy_unit: str | None,
) -> dict[str, list[Mapping[str, Any]]]:
    result: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        token = (
            f"row:{index}"
            if privacy_unit is None
            else _exact_typed_token(row.get(privacy_unit, _MISSING))
        )
        result[token].append(row)
    return dict(result)


def _release_privacy_unit_membership(
    rows: Sequence[Mapping[str, Any]],
    privacy_unit: str | None,
) -> tuple[int, ...]:
    if privacy_unit is None:
        return tuple(range(len(rows)))
    group_by_token: dict[str, int] = {}
    membership: list[int] = []
    for row in rows:
        token = _exact_typed_token(row[privacy_unit])
        membership.append(group_by_token.setdefault(token, len(group_by_token)))
    return tuple(membership)


def _validate_result_membership(
    membership: tuple[int, ...],
    *,
    row_count: int,
    row_level: bool,
) -> None:
    if not isinstance(membership, tuple) or any(
        type(group) is not int or group < 0 for group in membership
    ):
        raise TypeError(
            "privacy-unit membership must be a tuple of non-negative integers"
        )
    if len(membership) != row_count:
        raise ValueError("privacy-unit membership length does not match records")
    first_seen: dict[int, int] = {}
    canonical = tuple(
        first_seen.setdefault(group, len(first_seen)) for group in membership
    )
    if canonical != membership:
        raise ValueError(
            "privacy-unit membership must use contiguous first-seen group indices"
        )
    if row_level and membership != tuple(range(row_count)):
        raise ValueError("row-level releases require one privacy unit per row")


def _validate_result_summary_bindings(
    *,
    policy: AnonymityPolicy,
    before: ReleaseAssessment,
    after: ReleaseAssessment,
    generalization: GeneralizationSummary,
    utility: UtilitySummary,
    source_dataset_digest: str,
    released_row_count: int,
    released_privacy_unit_count: int,
) -> None:
    if before.policy != policy or after.policy != policy:
        raise ValueError("release assessments must use the result policy")
    for name, assessment in (("before", before), ("after", after)):
        if assessment.policy_digest != policy.digest:
            raise ValueError(f"{name} policy digest does not match the result policy")
        if not _DIGEST_RE.fullmatch(assessment.dataset_digest):
            raise ValueError(f"{name} dataset digest is not canonical")
        _validate_assessment_aggregates(assessment, name=name)
    if before.dataset_digest != source_dataset_digest:
        raise ValueError("source_dataset_digest does not match the before assessment")
    if not after.meets_policy:
        raise ValueError("the after assessment must satisfy the release policy")
    if after.row_count != released_row_count:
        raise ValueError("after row count does not match released records")
    if after.privacy_unit_count != released_privacy_unit_count:
        raise ValueError("after privacy-unit count does not match released membership")
    if before.row_count < after.row_count:
        raise ValueError("released row count cannot exceed source row count")
    if before.privacy_unit_count < after.privacy_unit_count:
        raise ValueError(
            "released privacy-unit count cannot exceed source privacy-unit count"
        )

    suppressed_rows = before.row_count - after.row_count
    suppressed_units = before.privacy_unit_count - after.privacy_unit_count
    if generalization.suppressed_rows != suppressed_rows:
        raise ValueError("suppressed row count is inconsistent")
    if generalization.suppressed_privacy_units != suppressed_units:
        raise ValueError("suppressed privacy-unit count is inconsistent")
    _validate_generalization_summary(
        generalization,
        policy=policy,
        source_privacy_units=before.privacy_unit_count,
        released_rows=after.row_count,
    )
    _validate_utility_summary(
        utility,
        policy=policy,
        before=before,
        after=after,
    )


def _validate_assessment_aggregates(
    assessment: ReleaseAssessment,
    *,
    name: str,
) -> None:
    integer_fields = (
        assessment.row_count,
        assessment.privacy_unit_count,
        assessment.class_count,
        assessment.achieved_k,
        assessment.singleton_class_count,
        assessment.singleton_privacy_unit_count,
        assessment.k_violating_class_count,
        assessment.l_violating_class_count,
        assessment.t_violating_class_count,
        assessment.violating_class_count,
        assessment.violating_privacy_unit_count,
    )
    if any(type(value) is not int or value < 0 for value in integer_fields):
        raise ValueError(f"{name} assessment counts must be non-negative integers")
    if (
        assessment.policy.privacy_unit is None
        and assessment.row_count != assessment.privacy_unit_count
    ):
        raise ValueError(f"{name} row-level privacy-unit count is inconsistent")
    distribution = assessment.class_size_distribution
    if any(
        type(size) is not int or type(count) is not int or size < 1 or count < 1
        for size, count in distribution
    ):
        raise ValueError(f"{name} class-size distribution is invalid")
    if tuple(sorted(dict(distribution).items())) != distribution:
        raise ValueError(f"{name} class-size distribution must be unique and sorted")
    if sum(count for _size, count in distribution) != assessment.class_count:
        raise ValueError(f"{name} class count is inconsistent")
    if (
        sum(size * count for size, count in distribution)
        != assessment.privacy_unit_count
    ):
        raise ValueError(f"{name} privacy-unit count is inconsistent")
    expected_k = min((size for size, _count in distribution), default=0)
    if assessment.achieved_k != expected_k:
        raise ValueError(f"{name} achieved k is inconsistent")
    singleton_classes = dict(distribution).get(1, 0)
    if assessment.singleton_class_count != singleton_classes:
        raise ValueError(f"{name} singleton class count is inconsistent")
    if assessment.singleton_privacy_unit_count != singleton_classes:
        raise ValueError(f"{name} singleton privacy-unit count is inconsistent")
    class_violation_counts = (
        assessment.k_violating_class_count,
        assessment.l_violating_class_count,
        assessment.t_violating_class_count,
    )
    if any(value > assessment.class_count for value in class_violation_counts):
        raise ValueError(f"{name} class violation count exceeds class count")
    if not (
        max(class_violation_counts, default=0)
        <= assessment.violating_class_count
        <= min(sum(class_violation_counts), assessment.class_count)
    ):
        raise ValueError(f"{name} union violation count is inconsistent")
    if assessment.violating_privacy_unit_count > assessment.privacy_unit_count:
        raise ValueError(f"{name} violating privacy-unit count is inconsistent")
    risk_values = (
        assessment.max_sample_identity_risk,
        assessment.mean_sample_identity_risk,
        assessment.median_sample_identity_risk,
        assessment.p95_sample_identity_risk,
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in risk_values
    ):
        raise ValueError(f"{name} identity-risk aggregates are invalid")
    attributes = tuple(summary.attribute for summary in assessment.attributes)
    if attributes != assessment.policy.sensitive_attributes:
        raise ValueError(f"{name} sensitive-attribute summaries are inconsistent")


def _validate_generalization_summary(
    summary: GeneralizationSummary,
    *,
    policy: AnonymityPolicy,
    source_privacy_units: int,
    released_rows: int,
) -> None:
    level_fields = tuple(field for field, _level, _name, _loss in summary.levels)
    if level_fields != policy.quasi_identifiers:
        raise ValueError("generalization levels do not match quasi-identifiers")
    affected_fields = tuple(field for field, _count in summary.affected_privacy_units)
    if affected_fields != policy.quasi_identifiers:
        raise ValueError(
            "generalization affected counts do not match quasi-identifiers"
        )
    affected_cell_fields = tuple(field for field, _count in summary.affected_qi_cells)
    suppressed_cell_fields = tuple(
        field for field, _count in summary.suppressed_qi_cells
    )
    if (
        affected_cell_fields != policy.quasi_identifiers
        or suppressed_cell_fields != policy.quasi_identifiers
    ):
        raise ValueError("generalization cell counts do not match quasi-identifiers")
    released_privacy_units = source_privacy_units - summary.suppressed_privacy_units
    if any(
        type(count) is not int or count < 0 or count > released_privacy_units
        for _field, count in summary.affected_privacy_units
    ):
        raise ValueError("generalization affected counts are invalid")
    affected_cells = dict(summary.affected_qi_cells)
    suppressed_cells = dict(summary.suppressed_qi_cells)
    if any(
        type(affected_cells[field]) is not int
        or type(suppressed_cells[field]) is not int
        or affected_cells[field] < 0
        or affected_cells[field] > released_rows
        or suppressed_cells[field] < 0
        or suppressed_cells[field] > affected_cells[field]
        for field in policy.quasi_identifiers
    ):
        raise ValueError("generalization cell counts are invalid")
    affected_by_field = dict(summary.affected_privacy_units)
    if any(
        level == 0
        and (
            affected_by_field[field] != 0
            or affected_cells[field] != 0
            or suppressed_cells[field] != 0
        )
        for field, level, _name, _loss in summary.levels
    ):
        raise ValueError("exact hierarchy levels cannot report transformations")
    if any(
        type(level) is not int
        or level < 0
        or not isinstance(name, str)
        or not name
        or not math.isfinite(loss)
        or not 0.0 <= loss <= 1.0
        for _field, level, name, loss in summary.levels
    ):
        raise ValueError("generalization levels contain invalid metadata")
    search_counts = (
        summary.search_space_size,
        summary.nodes_evaluated,
        summary.max_lattice_nodes,
        summary.suppression_subsets_evaluated,
        summary.max_suppression_subsets,
    )
    if any(type(value) is not int or value < 1 for value in search_counts):
        raise ValueError("generalization search counts must be positive integers")
    if summary.nodes_evaluated > summary.max_lattice_nodes:
        raise ValueError("lattice search exceeded its configured limit")
    if summary.nodes_evaluated > summary.search_space_size:
        raise ValueError("evaluated lattice nodes exceed the search space")
    if summary.suppression_subsets_evaluated > summary.max_suppression_subsets:
        raise ValueError("suppression search exceeded its configured limit")
    if summary.suppression_subsets_possible is not None and (
        type(summary.suppression_subsets_possible) is not int
        or summary.suppression_subsets_possible < 1
        or summary.suppression_subsets_evaluated > summary.suppression_subsets_possible
    ):
        raise ValueError("evaluated suppression subsets exceed the search space")
    if not isinstance(summary.search_complete, bool):
        raise TypeError("search_complete must be a boolean")
    exhaustive_complete = (
        summary.nodes_evaluated == summary.search_space_size
        and summary.suppression_subsets_possible is not None
        and summary.suppression_subsets_evaluated
        == summary.suppression_subsets_possible
    )
    zero_loss_proof = (
        summary.search_strategy == "zero-loss lower-bound lattice"
        and summary.suppression_search_strategy
        == "zero-loss lower-bound subset pruning"
        and summary.suppression_subsets_possible is None
        and summary.nodes_evaluated >= 1
        and summary.suppression_subsets_evaluated == 1
        and summary.information_loss == 0.0
        and summary.generalization_loss == 0.0
        and summary.suppression_loss == 0.0
        and all(level == 0 for _field, level, _name, _loss in summary.levels)
    )
    if summary.search_complete != exhaustive_complete or summary.optimum_proven != (
        exhaustive_complete or zero_loss_proof
    ):
        raise ValueError("generalization optimality claim is inconsistent")
    losses = (
        summary.information_loss,
        summary.generalization_loss,
        summary.suppression_loss,
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in losses
    ):
        raise ValueError("generalization loss values are invalid")
    expected_suppression_loss = _rate(
        summary.suppressed_privacy_units,
        source_privacy_units,
    )
    if not math.isclose(
        summary.suppression_loss,
        expected_suppression_loss,
        abs_tol=1e-12,
    ):
        raise ValueError("suppression loss is inconsistent")
    if not math.isclose(
        summary.information_loss,
        summary.generalization_loss + summary.suppression_loss,
        abs_tol=1e-12,
    ):
        raise ValueError("information loss is inconsistent")


def _validate_utility_summary(
    summary: UtilitySummary,
    *,
    policy: AnonymityPolicy,
    before: ReleaseAssessment,
    after: ReleaseAssessment,
) -> None:
    expected = (
        ("source_rows", summary.source_rows, before.row_count),
        ("released_rows", summary.released_rows, after.row_count),
        (
            "source_privacy_units",
            summary.source_privacy_units,
            before.privacy_unit_count,
        ),
        (
            "released_privacy_units",
            summary.released_privacy_units,
            after.privacy_unit_count,
        ),
    )
    for name, actual, target in expected:
        if type(actual) is not int or actual != target:
            raise ValueError(f"utility {name} is inconsistent")
    expected_row_rate = _rate(
        before.row_count - after.row_count,
        before.row_count,
    )
    expected_unit_rate = _rate(
        before.privacy_unit_count - after.privacy_unit_count,
        before.privacy_unit_count,
    )
    if not math.isclose(
        summary.row_suppression_rate,
        expected_row_rate,
        abs_tol=1e-12,
    ) or not math.isclose(
        summary.privacy_unit_suppression_rate,
        expected_unit_rate,
        abs_tol=1e-12,
    ):
        raise ValueError("utility suppression rates are inconsistent")
    compared = after.row_count * len(policy.quasi_identifiers)
    if summary.quasi_identifier_cells_compared != compared:
        raise ValueError("utility compared-cell count is inconsistent")
    count_values = (
        summary.quasi_identifier_cells_changed,
        summary.direct_identifier_cells_removed,
        summary.missing_qi_cells_before,
        summary.missing_qi_cells_after,
    )
    if any(type(value) is not int or value < 0 for value in count_values):
        raise ValueError("utility cell counts must be non-negative integers")
    if (
        summary.quasi_identifier_cells_changed > compared
        or summary.missing_qi_cells_before > compared
        or summary.missing_qi_cells_after > compared
    ):
        raise ValueError("utility cell count exceeds compared cells")
    expected_change_rate = _rate(
        summary.quasi_identifier_cells_changed,
        compared,
    )
    if not math.isclose(
        summary.quasi_identifier_cell_change_rate,
        expected_change_rate,
        abs_tol=1e-12,
    ):
        raise ValueError("utility QI-cell change rate is inconsistent")
    rates = (
        summary.row_suppression_rate,
        summary.privacy_unit_suppression_rate,
        summary.quasi_identifier_cell_change_rate,
        summary.mean_qi_distribution_shift,
    )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in rates
    ):
        raise ValueError("utility rates are invalid")


def _remeasure_released_records(
    records: Sequence[Mapping[str, Any]],
    membership: tuple[int, ...],
    policy: AnonymityPolicy,
) -> ReleaseAssessment:
    fields = {field for row in records for field in row}
    privacy_unit = policy.privacy_unit
    validation_policy = AnonymityPolicy(
        quasi_identifiers=policy.quasi_identifiers,
        sensitive_attributes=policy.sensitive_attributes,
        direct_identifiers=(privacy_unit,) if privacy_unit is not None else (),
        non_sensitive_attributes=tuple(
            field for field in policy.non_sensitive_attributes if field in fields
        ),
        privacy_unit=privacy_unit,
        target_k=policy.target_k,
        target_l=policy.target_l,
        l_metric=policy.l_metric,
        target_t=policy.target_t,
        t_distance=policy.t_distance,
        suppression_limit=policy.suppression_limit,
        suppression_rate=policy.suppression_rate,
        max_lattice_nodes=policy.max_lattice_nodes,
        max_suppression_subsets=policy.max_suppression_subsets,
    )
    rows = [dict(row) for row in records]
    if privacy_unit is not None:
        for row, group in zip(rows, membership):
            row[privacy_unit] = group
    return assess_release(rows, validation_policy)


def _comparable_assessment(assessment: ReleaseAssessment) -> dict[str, Any]:
    payload = assessment.to_dict()
    payload.pop("dataset_digest", None)
    payload.pop("policy_digest", None)
    return payload


def _result_binding_digest(
    *,
    policy: AnonymityPolicy,
    before: ReleaseAssessment,
    after: ReleaseAssessment,
    generalization: GeneralizationSummary,
    utility: UtilitySummary,
    source_dataset_digest: str,
    released_dataset_digest: str,
    released_schema_digest: str,
    hierarchy_digest: str,
    privacy_unit_membership: tuple[int, ...],
) -> str:
    return stable_hash(
        {
            "kind": "openmed-anonymization-result-binding",
            "policy": policy.to_dict(),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "generalization": generalization.to_dict(),
            "utility": utility.to_dict(),
            "source_dataset_digest": source_dataset_digest,
            "released_dataset_digest": released_dataset_digest,
            "released_schema_digest": released_schema_digest,
            "hierarchy_digest": hierarchy_digest,
            "privacy_unit_membership": list(privacy_unit_membership),
        }
    )


def _distribution_shift(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    before_counts = Counter(_normalized_cell(row, field) for row in before)
    after_counts = Counter(_normalized_cell(row, field) for row in after)
    before_total = sum(before_counts.values())
    after_total = sum(after_counts.values())
    values = set(before_counts) | set(after_counts)
    if not values or not before_total or not after_total:
        return 0.0
    return 0.5 * sum(
        abs(
            before_counts.get(value, 0) / before_total
            - after_counts.get(value, 0) / after_total
        )
        for value in values
    )


def _assessment_warnings(
    rows: Sequence[Mapping[str, Any]],
    projections: Sequence[_SubjectProjection],
    policy: AnonymityPolicy,
) -> tuple[str, ...]:
    warnings = [
        (
            "This report supports qualified expert review and is not an Expert "
            "Determination or compliance certificate."
        ),
        (
            "Sample equivalence-class risk does not establish population risk "
            "without an appropriate population model."
        ),
    ]
    if policy.privacy_unit is None:
        warnings.append("Each input row is treated as one privacy unit.")
    elif any(len(projection.row_indices) > 1 for projection in projections):
        warnings.append(
            "Repeated rows within keyed privacy units are assessed as part of "
            "deterministic joint ordered-multiset fingerprints; row multiplicity "
            "can distinguish privacy units."
        )
    multi_fields = sorted(
        {field for projection in projections for field in projection.multi_valued_qis}
    )
    if multi_fields:
        warnings.append(
            "Some quasi-identifiers vary within a privacy unit and are assessed "
            "as deterministic joint ordered-multiset fingerprints: "
            f"{', '.join(multi_fields)}."
        )
    multi_sensitive_fields = sorted(
        {
            field
            for projection in projections
            for field in projection.multi_valued_sensitive_attributes
        }
    )
    if multi_sensitive_fields:
        warnings.append(
            "Multi-valued sensitive attributes were not used for a nontrivial "
            "l-diversity or t-closeness claim; configure one reviewed value per "
            f"privacy unit before tightening those criteria: "
            f"{', '.join(multi_sensitive_fields)}."
        )
    incomplete_qis = sorted(
        {
            field
            for row in rows
            for field in policy.quasi_identifiers
            if field not in row
            or row[field] is None
            or (isinstance(row[field], str) and not row[field])
        }
    )
    if incomplete_qis:
        warnings.append(
            "Missing, null, and empty quasi-identifier states are treated as "
            "distinct exact values until a selected hierarchy suppresses them: "
            f"{', '.join(incomplete_qis)}."
        )
    if not rows:
        warnings.append("The input release is empty and cannot satisfy a policy.")
    return tuple(warnings)


def _validate_policy_columns(
    rows: Sequence[Mapping[str, Any]],
    policy: AnonymityPolicy,
) -> None:
    available = {str(field) for row in rows for field in row}
    declared = {
        *policy.quasi_identifiers,
        *policy.sensitive_attributes,
        *policy.direct_identifiers,
        *policy.non_sensitive_attributes,
        *policy.excluded_attributes,
        *([policy.privacy_unit] if policy.privacy_unit is not None else []),
    }
    unknown = sorted(declared - available)
    if unknown:
        raise ValueError(
            f"Declared policy columns are absent from the table: {unknown}"
        )
    unreviewed = sorted(available - declared)
    if unreviewed:
        raise ValueError(
            "Every source column requires an explicit release role; classify "
            "columns as a quasi-identifier, sensitive, direct identifier, "
            f"non-sensitive, or excluded: {unreviewed!r}"
        )
    reviewed_direct = {
        *policy.direct_identifiers,
        *policy.excluded_attributes,
        *([policy.privacy_unit] if policy.privacy_unit is not None else []),
    }
    undeclared_direct = sorted(
        field
        for field in available
        if _field_is_direct_identifier(field) and field not in reviewed_direct
    )
    if undeclared_direct:
        raise ValueError(
            "Columns that appear to be direct identifiers must be explicitly "
            f"removed or excluded: {undeclared_direct!r}"
        )
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if field == policy.privacy_unit:
                continue
            try:
                _canonical_digest_scalar(value)
            except TypeError:
                raise TypeError(
                    f"Structured column {str(field)!r} contains an unsupported "
                    f"value at row offset {row_index}"
                ) from None
            except ValueError:
                raise ValueError(
                    f"Structured column {str(field)!r} must be finite and "
                    f"canonical at row offset {row_index}"
                ) from None
        for field in (*policy.quasi_identifiers, *policy.sensitive_attributes):
            if field not in row:
                if field in policy.sensitive_attributes:
                    raise ValueError(
                        f"Sensitive attribute {field!r} is missing at row "
                        f"offset {row_index}; missing values require an explicit "
                        "preprocessing policy and cannot count toward l-diversity"
                    )
                continue
            value = row[field]
            if field in policy.sensitive_attributes and (
                value is None
                or (
                    isinstance(value, str)
                    and (not value.strip() or value != value.strip())
                )
                or (type(value) is bytes and not value)
            ):
                raise ValueError(
                    f"Sensitive attribute {field!r} is blank or has surrounding "
                    f"whitespace at row offset {row_index}; ambiguous values "
                    "require an explicit preprocessing policy and cannot count "
                    "toward l-diversity"
                )


def _materialize_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []
    _validate_dataframe_temporal_precision(records)
    dataframe_columns = getattr(records, "columns", None)
    if dataframe_columns is not None:
        try:
            columns = list(dataframe_columns)
        except TypeError:
            raise TypeError("DataFrame columns must be an iterable schema") from None
        if any(type(field) is not str for field in columns):
            raise TypeError("DataFrame column names must be strings")
        if len(columns) != len(set(columns)):
            raise ValueError("DataFrame column names must be unique")
        for field in columns:
            _validated_column_name(field, name="DataFrame column")
    to_dicts = getattr(records, "to_dicts", None)
    if callable(to_dicts):
        records = to_dicts()
    else:
        to_dict = getattr(records, "to_dict", None)
        if callable(to_dict) and not isinstance(records, Mapping):
            records = to_dict("records")
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        raise TypeError("records must be a sequence of row mappings")
    if not all(isinstance(row, Mapping) for row in records):
        raise TypeError("records must contain only row mappings")
    materialized = []
    for row_index, row in enumerate(records):
        output: dict[str, Any] = {}
        for column_index, (field, value) in enumerate(row.items()):
            if type(field) is not str:
                raise TypeError(
                    "Structured column names must be strings; unsupported name "
                    f"at row offset {row_index}, column offset {column_index}"
                )
            _validated_column_name(field, name="Structured column")
            output[field] = _normalized_dataframe_scalar(value)
        materialized.append(output)
    return materialized


def _normalized_dataframe_scalar(value: Any) -> Any:
    """Convert common DataFrame scalar wrappers to supported Python scalars."""

    module = type(value).__module__
    type_name = type(value).__name__
    if module.startswith("pandas") and type_name in {"NAType", "NaTType"}:
        return None
    if module.startswith("pandas"):
        to_pydatetime = getattr(value, "to_pydatetime", None)
        if callable(to_pydatetime):
            if getattr(value, "nanosecond", 0):
                raise ValueError(
                    "DataFrame timestamps with sub-microsecond precision are "
                    "unsupported"
                )
            converted = to_pydatetime()
            if type(converted) is datetime:
                return converted
    if module.startswith("numpy"):
        if type_name == "datetime64":
            microseconds = value.astype("datetime64[us]")
            converted = microseconds.item()
            if converted is not None and bool(
                microseconds.astype(value.dtype) != value
            ):
                raise ValueError(
                    "DataFrame timestamps with sub-microsecond precision are "
                    "unsupported"
                )
        elif type_name == "timedelta64":
            raise TypeError("DataFrame time durations are unsupported")
        else:
            item = getattr(value, "item", None)
            converted = item() if callable(item) else value
        if converted is not value:
            return converted
    return value


def _column_tuple(
    value: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of column names")
    columns = []
    for item in value:
        columns.append(_validated_column_name(item, name=name))
    result = tuple(sorted(dict.fromkeys(columns)))
    if not result and not allow_empty:
        raise ValueError(f"{name} must contain at least one column")
    return result


def _optional_column(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _validated_column_name(value, name=name)


def _validated_column_name(value: Any, *, name: str) -> str:
    """Validate a real source-schema label without imposing ASCII syntax."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must contain string column names")
    if (
        not value
        or value != value.strip()
        or len(value) > 256
        or any(
            unicodedata.category(character).startswith("C")
            or unicodedata.category(character) in {"Zl", "Zp"}
            for character in value
        )
    ):
        raise ValueError(
            f"{name} must use non-empty column names without surrounding "
            "whitespace or control characters"
        )
    return value


def _typed_token(value: Any) -> str:
    payload = _canonical_digest_scalar(value)
    if type(value) is float:
        payload = {
            "type": "float",
            "value": "0" if value == 0.0 else repr(value),
        }
    elif type(value) is str:
        payload = {"type": "str", "value": unicodedata.normalize("NFC", value)}
    elif type(value) is Decimal:
        payload = {"type": "decimal", "value": _canonical_decimal_text(value)}
    elif type(value) is datetime and value.tzinfo is not None:
        if value.utcoffset() is None:
            raise ValueError("datetime timezone offsets must be determinate")
        payload = {
            "type": "datetime",
            "value": value.astimezone(timezone.utc).isoformat(),
        }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _exact_typed_token(value: Any) -> str:
    """Return the exact typed representation written for a released cell."""

    payload = _canonical_digest_scalar(value)
    if type(value) is float:
        payload = {"type": "float", "value": repr(value)}
    elif type(value) is Decimal:
        payload = {"type": "decimal", "value": str(value)}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_cell(row: Mapping[str, Any], field: str) -> str:
    if field not in row:
        return _MISSING
    value = row[field]
    if value is None:
        return _NULL
    if isinstance(value, str) and not value:
        return _EMPTY
    if value in {_MISSING, _NULL, _EMPTY}:
        return value
    return _exact_typed_token(value)


def _dataset_digest(rows: Any) -> str:
    materialized = _materialize_rows(rows)
    return stable_hash(
        {
            "kind": "openmed-release-dataset",
            "row_count": len(materialized),
            "rows": [
                {
                    str(field): _canonical_digest_scalar(value)
                    for field, value in row.items()
                }
                for row in materialized
            ],
        }
    )


def _schema_digest(rows: Any) -> str:
    materialized = _materialize_rows(rows)
    fields = sorted({field for row in materialized for field in row})
    schema = []
    for field in fields:
        present_values = [row[field] for row in materialized if field in row]
        schema.append(
            {
                "field": field,
                "present_count": len(present_values),
                "null_count": sum(value is None for value in present_values),
                "types": sorted(
                    {_canonical_scalar_type(value) for value in present_values}
                ),
            }
        )
    return stable_hash(
        {
            "kind": "openmed-release-schema",
            "row_count": len(materialized),
            "fields": schema,
        }
    )


def _canonical_scalar_type(value: Any) -> str:
    return str(_canonical_digest_scalar(value)["type"])


def _canonical_digest_scalar(value: Any) -> dict[str, Any]:
    if isinstance(value, _InternalQIState):
        return {
            "type": value.kind,
            "value": (
                [_canonical_digest_scalar(item) for item in value.values]
                if value.values
                else None
            ),
        }
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("floating-point values must be finite")
        return {"type": "float", "value": repr(value)}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("decimal values must be finite")
        return {"type": "decimal", "value": str(value)}
    if type(value) is datetime:
        if value.tzinfo is not None and value.utcoffset() is None:
            raise ValueError("datetime timezone offsets must be determinate")
        return {"type": "datetime", "value": value.isoformat()}
    if type(value) is date:
        return {"type": "date", "value": value.isoformat()}
    if type(value) is time:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("timezone-aware time values are unsupported")
        return {"type": "time", "value": value.isoformat()}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    raise TypeError("structured values must be supported tabular scalars")


def _hierarchy_digest(
    subject_rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
    supplied: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> str:
    resolved = build_generalization_hierarchies(
        subject_rows,
        quasi_identifiers,
        hierarchies=supplied,
    )
    normalized_supplied: dict[str, list[dict[str, Any]]] = {}
    for field, levels in sorted((supplied or {}).items()):
        normalized_levels = []
        for level in levels:
            normalized: dict[str, Any] = {}
            if "name" in level:
                normalized["name"] = str(level["name"])
            if "loss" in level:
                normalized["loss"] = float(level["loss"])
            if "default" in level:
                normalized["default"] = str(level["default"])
            if "values" in level:
                values = level["values"]
                if isinstance(values, Mapping):
                    normalized["values"] = {
                        str(key): str(value)
                        for key, value in sorted(
                            values.items(),
                            key=lambda item: str(item[0]),
                        )
                    }
            normalized_levels.append(normalized)
        normalized_supplied[str(field)] = normalized_levels
    return stable_hash(
        {
            "kind": "openmed-release-hierarchy",
            "resolved": resolved,
            "supplied": normalized_supplied,
        }
    )


def _stringified_scalar_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    fields = list(dict.fromkeys(str(field) for row in rows for field in row))
    result = []
    for row in rows:
        converted: dict[str, str] = {}
        for field in fields:
            value = row.get(field)
            if value is None:
                converted[field] = ""
            elif isinstance(value, (str, int, float, bool)):
                converted[field] = str(value)
            else:
                raise TypeError(
                    "Delimited release validation supports scalar values only"
                )
        result.append(converted)
    return result


def _delimited_policy_encoding_is_injective(
    result: AnonymizationResult,
) -> bool:
    fields = (
        *result.policy.quasi_identifiers,
        *result.policy.sensitive_attributes,
    )
    for field in fields:
        encoded_to_typed: dict[str, str] = {}
        for row in result.records:
            if field not in row:
                return False
            value = row[field]
            if value is not None and not isinstance(value, (str, int, float, bool)):
                return False
            encoded = "" if value is None else str(value)
            typed = (
                _exact_typed_token(value)
                if field in result.policy.quasi_identifiers
                else _typed_token(value)
            )
            previous = encoded_to_typed.setdefault(encoded, typed)
            if previous != typed:
                return False
    return True


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0
