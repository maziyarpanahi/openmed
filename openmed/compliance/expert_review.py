"""PHI-safe evidence bundles for qualified de-identification review.

This module intentionally accepts only typed aggregate metadata. It has no
input fields for records, equivalence-class keys, samples, record identifiers,
source paths, or transformed data. Serialization is allow-listed field by
field so future inputs are not copied into an expert-review artifact by
accident.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Final, Mapping

from openmed.core.audit import stable_hash

EXPERT_REVIEW_EVIDENCE_SCHEMA_VERSION: Final = 3
_SUPPORTED_EVIDENCE_SCHEMA_VERSIONS: Final = frozenset({2, 3})
EXPERT_REVIEW_EVIDENCE_TITLE: Final = (
    "De-identification Risk Analysis Evidence Bundle — Not an Expert Determination"
)
EXPERT_REVIEW_EVIDENCE_DISCLAIMER: Final = (
    "This evidence bundle is technical decision-support documentation and is "
    "not an Expert Determination, certification, or legal conclusion. A "
    "qualified expert must independently evaluate the data, release context, "
    "recipients, reasonably available auxiliary data, re-identification risk, "
    "and methods, then document and sign any Expert Determination."
)

_REPORT_TYPE: Final = "deidentification_expert_review_evidence"
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")

_PRIVACY_UNITS: Final = frozenset(
    {
        "row",
        "patient",
        "person",
        "encounter",
        "document",
        "event",
        "household",
        "other",
    }
)
_POPULATION_SCOPES: Final = frozenset(
    {
        "source_population",
        "eligible_cohort",
        "sampled_cohort",
        "release_cohort",
        "external_reference_population",
        "other_documented",
    }
)
_RELEASE_MODELS: Final = frozenset(
    {"public", "restricted", "controlled", "internal", "other_documented"}
)
_RECIPIENT_MODELS: Final = frozenset(
    {
        "general_public",
        "named_researchers",
        "covered_entity",
        "authorized_internal",
        "contracted_recipient",
        "other_documented",
    }
)
_AUXILIARY_DATA_MODELS: Final = frozenset(
    {
        "publicly_available",
        "recipient_supplied",
        "reasonably_available",
        "expert_defined",
        "none_assumed",
        "other_documented",
    }
)
_ATTRIBUTE_ROLES: Final = frozenset(
    {
        "direct_identifier",
        "privacy_unit",
        "quasi_identifier",
        "sensitive_attribute",
        "non_sensitive",
        "excluded",
    }
)
_OVERRIDE_REASONS: Final = frozenset(
    {
        "domain_expert_judgment",
        "external_linkability",
        "population_specificity",
        "release_context",
        "recipient_context",
        "sensitive_attribute_dual_role",
        "other_documented_review",
    }
)
_L_VARIANTS: Final = frozenset({"distinct", "entropy"})
_T_VARIANTS: Final = frozenset({"variational"})
_TRANSFORMATION_METHODS: Final = frozenset(
    {
        "generalize",
        "coarsen",
        "bucket",
        "truncate",
        "suppress",
        "remove",
        "mask",
        "other_documented",
    }
)
_UTILITY_UNITS: Final = frozenset(
    {"ratio", "percent", "count", "seconds", "score", "bits"}
)
_SEARCH_STRATEGIES: Final = frozenset(
    {
        "exhaustive_lattice",
        "bounded_lattice",
        "heuristic",
        "measurement_only",
        "external",
    }
)
_TERMINATION_REASONS: Final = frozenset(
    {
        "search_exhausted",
        "optimal_candidate_found",
        "candidate_limit_reached",
        "time_limit_reached",
        "memory_limit_reached",
        "measurement_only",
        "external_result",
        "no_feasible_candidate",
    }
)
_COMPOSITION_STATUSES: Final = frozenset(
    {
        "not_assessed",
        "no_material_increase_observed",
        "increase_observed",
        "inconclusive",
    }
)


class _InvalidEvidenceJson(ValueError):
    pass


def _strict_evidence_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, value in pairs:
        if field in result:
            raise _InvalidEvidenceJson("duplicate object key")
        result[field] = value
    return result


def _reject_evidence_json_constant(_value: str) -> None:
    raise _InvalidEvidenceJson("non-finite JSON number")


def _parse_evidence_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _InvalidEvidenceJson("non-finite JSON number")
    return parsed


@dataclass(frozen=True)
class EvidenceDigests:
    """Content digests that bind the evidence to reviewed inputs and software."""

    source_dataset: str
    dataset: str
    schema: str
    policy: str
    hierarchy: str
    config: str
    software: str

    def __post_init__(self) -> None:
        for value in (
            self.source_dataset,
            self.dataset,
            self.schema,
            self.policy,
            self.hierarchy,
            self.config,
            self.software,
        ):
            _require_digest(value)

    def to_dict(self) -> dict[str, str]:
        """Return the seven allow-listed digests."""

        return {
            "source_dataset": self.source_dataset,
            "dataset": self.dataset,
            "schema": self.schema,
            "policy": self.policy,
            "hierarchy": self.hierarchy,
            "config": self.config,
            "software": self.software,
        }


@dataclass(frozen=True)
class ReleaseAssumptions:
    """Coded release context with a digest binding any detailed notes."""

    privacy_unit: str
    population_scope: str
    release_model: str
    recipient_model: str
    auxiliary_data_model: str
    notes_digest: str

    def __post_init__(self) -> None:
        _require_choice(self.privacy_unit, _PRIVACY_UNITS, "privacy unit")
        _require_choice(self.population_scope, _POPULATION_SCOPES, "population scope")
        _require_choice(self.release_model, _RELEASE_MODELS, "release model")
        _require_choice(self.recipient_model, _RECIPIENT_MODELS, "recipient model")
        _require_choice(
            self.auxiliary_data_model,
            _AUXILIARY_DATA_MODELS,
            "auxiliary-data model",
        )
        _require_digest(self.notes_digest)

    def to_dict(self) -> dict[str, str]:
        """Return coded assumptions without copying free-form review notes."""

        return {
            "privacy_unit": self.privacy_unit,
            "population_scope": self.population_scope,
            "release_model": self.release_model,
            "recipient_model": self.recipient_model,
            "auxiliary_data_model": self.auxiliary_data_model,
            "notes_digest": self.notes_digest,
        }


@dataclass(frozen=True)
class AttributeRoleReview:
    """Reviewed schema role assignments for one aggregate attribute."""

    attribute: str
    roles: tuple[str, ...]
    override_applied: bool = False
    override_reason: str | None = None

    def __post_init__(self) -> None:
        _require_attribute_name(self.attribute, "attribute")
        _require_string_tuple(self.roles, "attribute roles")
        if not self.roles:
            raise ValueError("attribute roles must not be empty")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("attribute roles must be unique")
        for role in self.roles:
            _require_choice(role, _ATTRIBUTE_ROLES, "attribute role")
        if "excluded" in self.roles and len(self.roles) != 1:
            raise ValueError("excluded must be the only role for an attribute")
        if "privacy_unit" in self.roles and set(self.roles) != {
            "direct_identifier",
            "privacy_unit",
        }:
            raise ValueError("privacy_unit must be paired only with direct_identifier")
        if not isinstance(self.override_applied, bool):
            raise TypeError("override_applied must be a boolean")
        if self.override_applied:
            _require_choice(self.override_reason, _OVERRIDE_REASONS, "override reason")
        elif self.override_reason is not None:
            raise ValueError("override_reason requires override_applied=True")

    def to_dict(self) -> dict[str, Any]:
        """Return an allow-listed role review record."""

        return {
            "attribute": self.attribute,
            "roles": sorted(self.roles),
            "override_applied": self.override_applied,
            "override_reason": self.override_reason,
        }


@dataclass(frozen=True)
class PrivacyModelEvidence:
    """Configured and achieved k/l/t values before and after transformation."""

    configured_k: int
    pre_achieved_k: int
    achieved_k: int
    l_variant: str | None = None
    configured_l: int | float | None = None
    pre_achieved_l: int | float | None = None
    achieved_l: int | float | None = None
    t_variant: str | None = None
    configured_t: int | float | None = None
    pre_achieved_t: int | float | None = None
    achieved_t: int | float | None = None

    def __post_init__(self) -> None:
        _require_int(self.configured_k, "configured k", minimum=1)
        _require_int(self.pre_achieved_k, "pre-transform achieved k", minimum=0)
        _require_int(self.achieved_k, "achieved k", minimum=0)
        _validate_optional_model(
            variant=self.l_variant,
            configured=self.configured_l,
            pre_achieved=self.pre_achieved_l,
            achieved=self.achieved_l,
            variants=_L_VARIANTS,
            name="l-diversity",
        )
        _validate_optional_model(
            variant=self.t_variant,
            configured=self.configured_t,
            pre_achieved=self.pre_achieved_t,
            achieved=self.achieved_t,
            variants=_T_VARIANTS,
            name="t-closeness",
        )
        if self.l_variant == "distinct":
            for value in (
                self.configured_l,
                self.pre_achieved_l,
                self.achieved_l,
            ):
                _require_int(value, "distinct l-diversity value", minimum=0)
            if self.configured_l == 0:
                raise ValueError("configured distinct l must be at least 1")
        if self.t_variant == "variational":
            for value in (
                self.configured_t,
                self.pre_achieved_t,
                self.achieved_t,
            ):
                _require_number(
                    value, "variational t-closeness value", minimum=0.0, maximum=1.0
                )

    def to_dict(self) -> dict[str, Any]:
        """Return exact configured and observed privacy-model values."""

        return {
            "k_anonymity": {
                "configured_k": self.configured_k,
                "pre_achieved_k": self.pre_achieved_k,
                "achieved_k": self.achieved_k,
            },
            "l_diversity": (
                None
                if self.l_variant is None
                else {
                    "variant": self.l_variant,
                    "configured_l": self.configured_l,
                    "pre_achieved_l": self.pre_achieved_l,
                    "achieved_l": self.achieved_l,
                }
            ),
            "t_closeness": (
                None
                if self.t_variant is None
                else {
                    "variant": self.t_variant,
                    "configured_t": self.configured_t,
                    "pre_achieved_t": self.pre_achieved_t,
                    "achieved_t": self.achieved_t,
                }
            ),
        }


@dataclass(frozen=True)
class ClassSizeBin:
    """One aggregate equivalence-class-size histogram bin.

    Class membership is counted in the configured privacy unit, which may be
    one row or a keyed entity spanning multiple rows.
    """

    lower_bound: int
    upper_bound: int
    class_count: int
    privacy_unit_count: int

    def __post_init__(self) -> None:
        _require_int(self.lower_bound, "class-size lower bound", minimum=1)
        _require_int(self.upper_bound, "class-size upper bound", minimum=1)
        _require_int(self.class_count, "class count", minimum=1)
        _require_int(self.privacy_unit_count, "privacy-unit count", minimum=1)
        if self.upper_bound < self.lower_bound:
            raise ValueError("class-size bin upper bound must not be smaller")
        minimum_privacy_units = self.lower_bound * self.class_count
        maximum_privacy_units = self.upper_bound * self.class_count
        if not (
            minimum_privacy_units <= self.privacy_unit_count <= maximum_privacy_units
        ):
            raise ValueError("class-size bin privacy-unit count is outside its bounds")

    def to_dict(self) -> dict[str, int]:
        """Return aggregate bin counts without equivalence-class keys."""

        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "class_count": self.class_count,
            "privacy_unit_count": self.privacy_unit_count,
        }


@dataclass(frozen=True)
class AggregateRiskMetrics:
    """Aggregate class sizes and privacy-model violation counts.

    All membership counts use the configured privacy unit rather than source
    rows. This distinction matters when one person, encounter, or other keyed
    unit spans multiple rows.
    """

    privacy_unit_count: int
    equivalence_class_count: int
    smallest_class_size: int
    largest_class_size: int
    mean_class_size: float
    class_size_histogram: tuple[ClassSizeBin, ...]
    k_violating_class_count: int
    l_violating_class_count: int
    t_violating_class_count: int
    any_violating_class_count: int
    violating_privacy_unit_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("privacy-unit count", self.privacy_unit_count),
            ("equivalence-class count", self.equivalence_class_count),
            ("smallest class size", self.smallest_class_size),
            ("largest class size", self.largest_class_size),
            ("k-violating class count", self.k_violating_class_count),
            ("l-violating class count", self.l_violating_class_count),
            ("t-violating class count", self.t_violating_class_count),
            ("any-violating class count", self.any_violating_class_count),
            ("violating privacy-unit count", self.violating_privacy_unit_count),
        ):
            _require_int(value, name, minimum=0)
        _require_number(self.mean_class_size, "mean class size", minimum=0.0)
        if not isinstance(self.class_size_histogram, tuple):
            raise TypeError("class_size_histogram must be a tuple")
        if not all(
            isinstance(item, ClassSizeBin) for item in self.class_size_histogram
        ):
            raise TypeError("class_size_histogram must contain ClassSizeBin values")
        bins = sorted(
            self.class_size_histogram,
            key=lambda item: (item.lower_bound, item.upper_bound),
        )
        for previous, current in zip(bins, bins[1:]):
            if current.lower_bound <= previous.upper_bound:
                raise ValueError("class-size histogram bins must not overlap")
        if self.privacy_unit_count == 0:
            if any(
                (
                    self.equivalence_class_count,
                    self.smallest_class_size,
                    self.largest_class_size,
                    self.mean_class_size,
                    len(self.class_size_histogram),
                    self.k_violating_class_count,
                    self.l_violating_class_count,
                    self.t_violating_class_count,
                    self.any_violating_class_count,
                    self.violating_privacy_unit_count,
                )
            ):
                raise ValueError("empty aggregate metrics must contain only zeros")
            return
        if self.equivalence_class_count == 0:
            raise ValueError("non-empty metrics require equivalence classes")
        if not 1 <= self.smallest_class_size <= self.largest_class_size:
            raise ValueError("class-size extrema are inconsistent")
        if not (
            self.smallest_class_size <= self.mean_class_size <= self.largest_class_size
        ):
            raise ValueError("mean class size is outside the class-size extrema")
        if sum(item.class_count for item in bins) != self.equivalence_class_count:
            raise ValueError("histogram class counts do not match aggregate count")
        if sum(item.privacy_unit_count for item in bins) != self.privacy_unit_count:
            raise ValueError(
                "histogram privacy-unit counts do not match aggregate count"
            )
        if not self.class_size_histogram:
            raise ValueError("non-empty metrics require a class-size histogram")
        for count in (
            self.k_violating_class_count,
            self.l_violating_class_count,
            self.t_violating_class_count,
            self.any_violating_class_count,
        ):
            if count > self.equivalence_class_count:
                raise ValueError("violation count exceeds equivalence-class count")
        if self.violating_privacy_unit_count > self.privacy_unit_count:
            raise ValueError("violating privacy-unit count exceeds privacy-unit count")

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate metrics with a key-free class-size histogram."""

        bins = sorted(
            self.class_size_histogram,
            key=lambda item: (item.lower_bound, item.upper_bound),
        )
        return {
            "privacy_unit_count": self.privacy_unit_count,
            "equivalence_class_count": self.equivalence_class_count,
            "class_sizes": {
                "smallest": self.smallest_class_size,
                "largest": self.largest_class_size,
                "mean": self.mean_class_size,
                "histogram": [item.to_dict() for item in bins],
            },
            "violations": {
                "k_class_count": self.k_violating_class_count,
                "l_class_count": self.l_violating_class_count,
                "t_class_count": self.t_violating_class_count,
                "any_class_count": self.any_violating_class_count,
                "privacy_unit_count": self.violating_privacy_unit_count,
            },
        }


@dataclass(frozen=True)
class TransformationAggregate:
    """Aggregate transformation count for one reviewed schema attribute."""

    attribute: str
    method: str
    affected_privacy_unit_count: int
    hierarchy_level_before: int | None = None
    hierarchy_level_after: int | None = None

    def __post_init__(self) -> None:
        _require_attribute_name(self.attribute, "transformation attribute")
        _require_choice(self.method, _TRANSFORMATION_METHODS, "transformation method")
        _require_int(
            self.affected_privacy_unit_count,
            "affected privacy-unit count",
            minimum=0,
        )
        if (self.hierarchy_level_before is None) != (
            self.hierarchy_level_after is None
        ):
            raise ValueError("hierarchy levels must be provided together")
        if self.hierarchy_level_before is not None:
            _require_int(
                self.hierarchy_level_before,
                "hierarchy level before",
                minimum=0,
            )
            _require_int(
                self.hierarchy_level_after,
                "hierarchy level after",
                minimum=0,
            )

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate transformation metadata."""

        return {
            "attribute": self.attribute,
            "method": self.method,
            "affected_privacy_unit_count": self.affected_privacy_unit_count,
            "hierarchy_level_before": self.hierarchy_level_before,
            "hierarchy_level_after": self.hierarchy_level_after,
        }


@dataclass(frozen=True)
class SuppressionAggregate:
    """Aggregate privacy-unit and cell suppression evidence."""

    privacy_units_suppressed: int
    rows_suppressed: int
    cells_suppressed: int
    suppression_rate: float
    privacy_unit_limit: int

    def __post_init__(self) -> None:
        _require_int(
            self.privacy_units_suppressed,
            "privacy units suppressed",
            minimum=0,
        )
        _require_int(self.rows_suppressed, "rows suppressed", minimum=0)
        _require_int(self.cells_suppressed, "cells suppressed", minimum=0)
        _require_number(
            self.suppression_rate,
            "suppression rate",
            minimum=0.0,
            maximum=1.0,
        )
        _require_int(
            self.privacy_unit_limit,
            "suppression privacy-unit limit",
            minimum=0,
        )
        if self.privacy_units_suppressed > self.privacy_unit_limit:
            raise ValueError("privacy units suppressed exceed the configured limit")
        if self.rows_suppressed < self.privacy_units_suppressed:
            raise ValueError(
                "rows suppressed cannot be fewer than privacy units suppressed"
            )

    def to_dict(self) -> dict[str, int | float]:
        """Return aggregate suppression evidence."""

        return {
            "privacy_units_suppressed": self.privacy_units_suppressed,
            "rows_suppressed": self.rows_suppressed,
            "cells_suppressed": self.cells_suppressed,
            "suppression_rate": self.suppression_rate,
            "privacy_unit_limit": self.privacy_unit_limit,
        }


@dataclass(frozen=True)
class UtilityAggregate:
    """One aggregate utility measurement before and after transformation."""

    metric: str
    before: int | float
    after: int | float
    unit: str
    higher_is_better: bool

    def __post_init__(self) -> None:
        _require_identifier(self.metric, "utility metric")
        _require_number(self.before, "utility value before")
        _require_number(self.after, "utility value after")
        _require_choice(self.unit, _UTILITY_UNITS, "utility unit")
        if not isinstance(self.higher_is_better, bool):
            raise TypeError("higher_is_better must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return utility values and their deterministic aggregate delta."""

        return {
            "metric": self.metric,
            "before": self.before,
            "after": self.after,
            "absolute_delta": self.after - self.before,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True)
class SearchEvidence:
    """Search coverage, optimality proof, limits, and termination metadata."""

    strategy: str
    complete: bool
    evaluated_candidates: int
    total_candidates: int | None
    maximum_quasi_identifiers: int
    candidate_limit: int | None
    suppression_subsets_evaluated: int
    suppression_subsets_total: int | None
    suppression_subset_limit: int | None
    time_limit_seconds: float | None
    termination_reason: str
    optimality_proven: bool | None = None

    def __post_init__(self) -> None:
        _require_choice(self.strategy, _SEARCH_STRATEGIES, "search strategy")
        if not isinstance(self.complete, bool):
            raise TypeError("search complete must be a boolean")
        _require_int(self.evaluated_candidates, "evaluated candidate count", minimum=0)
        _require_int(
            self.maximum_quasi_identifiers,
            "maximum quasi-identifier count",
            minimum=1,
        )
        if self.total_candidates is not None:
            _require_int(self.total_candidates, "total candidate count", minimum=0)
            if self.evaluated_candidates > self.total_candidates:
                raise ValueError("evaluated candidates exceed total candidates")
        if self.candidate_limit is not None:
            _require_int(self.candidate_limit, "candidate limit", minimum=1)
            if self.evaluated_candidates > self.candidate_limit:
                raise ValueError("evaluated candidates exceed candidate limit")
        _require_int(
            self.suppression_subsets_evaluated,
            "evaluated suppression-subset count",
            minimum=0,
        )
        if self.suppression_subsets_total is not None:
            _require_int(
                self.suppression_subsets_total,
                "total suppression-subset count",
                minimum=0,
            )
            if self.suppression_subsets_evaluated > self.suppression_subsets_total:
                raise ValueError(
                    "evaluated suppression subsets exceed total suppression subsets"
                )
        if self.suppression_subset_limit is not None:
            _require_int(
                self.suppression_subset_limit,
                "suppression-subset limit",
                minimum=1,
            )
            if self.suppression_subsets_evaluated > self.suppression_subset_limit:
                raise ValueError(
                    "evaluated suppression subsets exceed suppression-subset limit"
                )
        if self.time_limit_seconds is not None:
            _require_number(self.time_limit_seconds, "time limit", minimum=0.0)
        _require_choice(
            self.termination_reason, _TERMINATION_REASONS, "termination reason"
        )
        if self.optimality_proven is None:
            object.__setattr__(
                self,
                "optimality_proven",
                self.complete or self.termination_reason == "optimal_candidate_found",
            )
        elif not isinstance(self.optimality_proven, bool):
            raise TypeError("search optimality_proven must be a boolean")
        if self.complete:
            if not self.optimality_proven:
                raise ValueError("complete search must prove optimality")
            if self.total_candidates is None:
                raise ValueError("complete search requires total_candidates")
            if self.evaluated_candidates != self.total_candidates:
                raise ValueError("complete search must evaluate every candidate")
            if self.suppression_subsets_total is None:
                raise ValueError("complete search requires suppression_subsets_total")
            if self.suppression_subsets_evaluated != self.suppression_subsets_total:
                raise ValueError(
                    "complete search must evaluate every suppression subset"
                )
            if self.termination_reason not in {
                "search_exhausted",
                "optimal_candidate_found",
                "measurement_only",
                "external_result",
            }:
                raise ValueError("complete search has an incomplete termination reason")
        if self.strategy == "exhaustive_lattice" and not self.complete:
            raise ValueError("exhaustive_lattice search must be complete")
        if self.optimality_proven and not self.complete:
            if (
                self.strategy != "bounded_lattice"
                or self.termination_reason != "optimal_candidate_found"
                or self.evaluated_candidates < 1
                or self.suppression_subsets_evaluated < 1
            ):
                raise ValueError(
                    "pruned optimality proof requires a bounded lattice and an "
                    "evaluated optimal candidate"
                )
        if (
            self.termination_reason == "optimal_candidate_found"
            and not self.optimality_proven
        ):
            raise ValueError(
                "optimal_candidate_found termination requires proven optimality"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return search proof and explicitly configured limits."""

        assert self.optimality_proven is not None
        return {
            "strategy": self.strategy,
            "complete": self.complete,
            "optimality_proven": self.optimality_proven,
            "evaluated_candidates": self.evaluated_candidates,
            "total_candidates": self.total_candidates,
            "maximum_quasi_identifiers": self.maximum_quasi_identifiers,
            "candidate_limit": self.candidate_limit,
            "suppression_subsets_evaluated": self.suppression_subsets_evaluated,
            "suppression_subsets_total": self.suppression_subsets_total,
            "suppression_subset_limit": self.suppression_subset_limit,
            "time_limit_seconds": self.time_limit_seconds,
            "termination_reason": self.termination_reason,
        }


@dataclass(frozen=True)
class CompositionEvidence:
    """Aggregate evidence about repeated and longitudinal data releases."""

    release_count: int
    longitudinal_linkage_assessed: bool
    prior_release_overlap_assessed: bool
    risk_status: str
    evidence_digest: str

    def __post_init__(self) -> None:
        _require_int(self.release_count, "release count", minimum=1)
        if not isinstance(self.longitudinal_linkage_assessed, bool):
            raise TypeError("longitudinal_linkage_assessed must be a boolean")
        if not isinstance(self.prior_release_overlap_assessed, bool):
            raise TypeError("prior_release_overlap_assessed must be a boolean")
        _require_choice(self.risk_status, _COMPOSITION_STATUSES, "composition status")
        _require_digest(self.evidence_digest)
        if self.release_count == 1 and (
            self.longitudinal_linkage_assessed
            or self.prior_release_overlap_assessed
            or self.risk_status != "not_assessed"
        ):
            raise ValueError("single-release composition evidence must be unassessed")

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate composition-review evidence."""

        return {
            "release_count": self.release_count,
            "longitudinal_linkage_assessed": self.longitudinal_linkage_assessed,
            "prior_release_overlap_assessed": self.prior_release_overlap_assessed,
            "risk_status": self.risk_status,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class ExpertReviewEvidenceReport:
    """Deterministic, PHI-safe evidence awaiting qualified expert review."""

    digests: EvidenceDigests
    assumptions: ReleaseAssumptions
    attribute_reviews: tuple[AttributeRoleReview, ...]
    selected_quasi_identifiers: tuple[str, ...]
    sensitive_attributes: tuple[str, ...]
    privacy_models: PrivacyModelEvidence
    pre_metrics: AggregateRiskMetrics
    post_metrics: AggregateRiskMetrics
    transformations: tuple[TransformationAggregate, ...]
    suppression: SuppressionAggregate
    utility: tuple[UtilityAggregate, ...]
    search: SearchEvidence
    composition: CompositionEvidence
    limitations: tuple[str, ...]
    unsupported_modalities: tuple[str, ...]
    integrity_hash: str
    schema_version: int = EXPERT_REVIEW_EVIDENCE_SCHEMA_VERSION
    report_type: str = _REPORT_TYPE
    title: str = EXPERT_REVIEW_EVIDENCE_TITLE
    disclaimer: str = EXPERT_REVIEW_EVIDENCE_DISCLAIMER

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in _SUPPORTED_EVIDENCE_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported expert-review evidence schema version")
        if self.report_type != _REPORT_TYPE:
            raise ValueError("unsupported expert-review evidence report type")
        if self.title != EXPERT_REVIEW_EVIDENCE_TITLE:
            raise ValueError("expert-review evidence title must not be changed")
        if self.disclaimer != EXPERT_REVIEW_EVIDENCE_DISCLAIMER:
            raise ValueError("qualified-expert disclaimer must not be changed")
        for aggregate_input, expected, name in (
            (self.digests, EvidenceDigests, "digests"),
            (self.assumptions, ReleaseAssumptions, "assumptions"),
            (self.privacy_models, PrivacyModelEvidence, "privacy_models"),
            (self.pre_metrics, AggregateRiskMetrics, "pre_metrics"),
            (self.post_metrics, AggregateRiskMetrics, "post_metrics"),
            (self.suppression, SuppressionAggregate, "suppression"),
            (self.search, SearchEvidence, "search"),
            (self.composition, CompositionEvidence, "composition"),
        ):
            if not isinstance(aggregate_input, expected):
                raise TypeError(f"{name} must use its typed aggregate input")
        _require_typed_tuple(
            self.attribute_reviews, AttributeRoleReview, "attribute_reviews"
        )
        _require_typed_tuple(
            self.transformations, TransformationAggregate, "transformations"
        )
        _require_typed_tuple(self.utility, UtilityAggregate, "utility")
        utility_metrics = [item.metric for item in self.utility]
        if len(set(utility_metrics)) != len(utility_metrics):
            raise ValueError("utility metric entries must be unique")
        for values, name in (
            (self.selected_quasi_identifiers, "selected quasi-identifiers"),
            (self.sensitive_attributes, "sensitive attributes"),
        ):
            _require_string_tuple(values, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            for attribute_name in values:
                _require_attribute_name(attribute_name, name)
        for values, name in (
            (self.limitations, "limitations"),
            (self.unsupported_modalities, "unsupported modalities"),
        ):
            _require_string_tuple(values, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            for metadata_identifier in values:
                _require_identifier(metadata_identifier, name)
        if not self.selected_quasi_identifiers:
            raise ValueError("at least one quasi-identifier must be selected")
        _require_digest(self.integrity_hash)
        self._validate_relationships()

    def _validate_relationships(self) -> None:
        reviews = {item.attribute: set(item.roles) for item in self.attribute_reviews}
        if len(reviews) != len(self.attribute_reviews):
            raise ValueError("attribute review entries must be unique")
        privacy_unit_attributes = {
            attribute for attribute, roles in reviews.items() if "privacy_unit" in roles
        }
        if self.assumptions.privacy_unit == "row":
            if privacy_unit_attributes:
                raise ValueError(
                    "row-level privacy assumptions cannot declare a privacy-unit "
                    "attribute"
                )
        elif len(privacy_unit_attributes) != 1:
            raise ValueError(
                "keyed privacy assumptions require exactly one reviewed "
                "privacy-unit attribute"
            )
        for attribute in self.selected_quasi_identifiers:
            if "quasi_identifier" not in reviews.get(attribute, set()):
                raise ValueError("selected quasi-identifiers require a reviewed role")
        for attribute in self.sensitive_attributes:
            if "sensitive_attribute" not in reviews.get(attribute, set()):
                raise ValueError("sensitive attributes require a reviewed role")
        has_l_model = self.privacy_models.l_variant is not None
        has_t_model = self.privacy_models.t_variant is not None
        if self.sensitive_attributes and not (has_l_model and has_t_model):
            raise ValueError(
                "sensitive attributes require both l-diversity and t-closeness evidence"
            )
        if not self.sensitive_attributes and (has_l_model or has_t_model):
            raise ValueError("l/t evidence requires sensitive attributes")
        reviewed_attributes = set(reviews)
        if any(
            item.attribute not in reviewed_attributes for item in self.transformations
        ):
            raise ValueError("transformations require a reviewed attribute")
        if self.pre_metrics.privacy_unit_count != (
            self.post_metrics.privacy_unit_count
            + self.suppression.privacy_units_suppressed
        ):
            raise ValueError("pre/post privacy-unit counts do not match suppression")
        expected_rate = (
            self.suppression.privacy_units_suppressed
            / self.pre_metrics.privacy_unit_count
            if self.pre_metrics.privacy_unit_count
            else 0.0
        )
        if not math.isclose(
            self.suppression.suppression_rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "suppression rate does not match aggregate privacy-unit counts"
            )
        if self.privacy_models.pre_achieved_k != (self.pre_metrics.smallest_class_size):
            raise ValueError("pre-transform achieved k does not match class metrics")
        if self.privacy_models.achieved_k != self.post_metrics.smallest_class_size:
            raise ValueError("achieved k does not match post-transform class metrics")
        if self.search.maximum_quasi_identifiers < len(self.selected_quasi_identifiers):
            raise ValueError("selected quasi-identifiers exceed the search limit")
        if (
            self.schema_version >= 3
            and self.search.optimality_proven
            and not self.search.complete
        ):
            utility_by_metric = {item.metric: item for item in self.utility}
            expected_zero_change_utility = {
                "information_loss": (0.0, 0.0, "score", False),
                "mean_qi_distribution_shift": (0.0, 0.0, "ratio", False),
                "privacy_unit_retention": (1.0, 1.0, "ratio", True),
                "quasi_identifier_cell_retention": (1.0, 1.0, "ratio", True),
                "row_retention": (1.0, 1.0, "ratio", True),
            }
            utility_matches = all(
                (
                    metric in utility_by_metric
                    and utility_by_metric[metric].before == before
                    and utility_by_metric[metric].after == after
                    and utility_by_metric[metric].unit == unit
                    and utility_by_metric[metric].higher_is_better is higher_is_better
                )
                for metric, (
                    before,
                    after,
                    unit,
                    higher_is_better,
                ) in expected_zero_change_utility.items()
            )
            privacy_models_unchanged = (
                self.privacy_models.pre_achieved_k == self.privacy_models.achieved_k
                and self.privacy_models.pre_achieved_l == self.privacy_models.achieved_l
                and self.privacy_models.pre_achieved_t == self.privacy_models.achieved_t
            )
            if (
                not utility_matches
                or self.pre_metrics != self.post_metrics
                or not privacy_models_unchanged
                or self.transformations
                or self.suppression.privacy_units_suppressed != 0
                or self.suppression.rows_suppressed != 0
                or self.suppression.cells_suppressed != 0
                or self.suppression.suppression_rate != 0.0
                or self.search.evaluated_candidates != 1
                or self.search.suppression_subsets_evaluated != 1
                or self.search.suppression_subsets_total is not None
            ):
                raise ValueError(
                    "incomplete search optimality requires an exact zero-loss "
                    "lower-bound proof"
                )

    def _payload(self) -> dict[str, Any]:
        reviews = sorted(
            self.attribute_reviews,
            key=lambda item: item.attribute,
        )
        transformations = sorted(
            self.transformations,
            key=lambda item: (
                item.attribute,
                item.method,
                item.hierarchy_level_before
                if item.hierarchy_level_before is not None
                else -1,
                item.hierarchy_level_after
                if item.hierarchy_level_after is not None
                else -1,
            ),
        )
        utility = sorted(self.utility, key=lambda item: item.metric)
        search_payload = self.search.to_dict()
        if self.schema_version == 2:
            search_payload.pop("optimality_proven")
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "title": self.title,
            "disclaimer": self.disclaimer,
            "digests": self.digests.to_dict(),
            "assumptions": self.assumptions.to_dict(),
            "attribute_review": [item.to_dict() for item in reviews],
            "selected_quasi_identifiers": sorted(self.selected_quasi_identifiers),
            "sensitive_attributes": sorted(self.sensitive_attributes),
            "privacy_models": self.privacy_models.to_dict(),
            "metrics": {
                "pre_transform": self.pre_metrics.to_dict(),
                "post_transform": self.post_metrics.to_dict(),
            },
            "transformations": [item.to_dict() for item in transformations],
            "suppression": self.suppression.to_dict(),
            "utility": [item.to_dict() for item in utility],
            "search": search_payload,
            "composition": self.composition.to_dict(),
            "limitations": sorted(self.limitations),
            "unsupported_modalities": sorted(self.unsupported_modalities),
            "qualified_expert_review": {
                "status": "pending_qualified_expert_review",
                "qualified_expert_name": None,
                "qualifications": None,
                "methodology_review": None,
                "risk_conclusion": None,
                "review_date": None,
                "signature": None,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit PHI-safe evidence schema."""

        return {**self._payload(), "integrity_hash": self.integrity_hash}

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the evidence as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render the evidence as deterministic, PHI-safe Markdown."""

        l_model = self.privacy_models.to_dict()["l_diversity"]
        t_model = self.privacy_models.to_dict()["t_closeness"]
        lines = [
            f"# {self.title}",
            "",
            f"> {self.disclaimer}",
            "",
            "## Evidence identity",
            "",
            f"- Schema version: `{self.schema_version}`",
            f"- Report type: `{self.report_type}`",
            f"- Integrity hash: `{self.integrity_hash}`",
            "",
            "## Bound inputs",
            "",
        ]
        for name, digest in self.digests.to_dict().items():
            lines.append(f"- {name.replace('_', ' ').title()}: `{digest}`")
        lines.extend(
            [
                "",
                "## Release assumptions",
                "",
            ]
        )
        for name, value in self.assumptions.to_dict().items():
            lines.append(f"- {name.replace('_', ' ').title()}: `{value}`")
        lines.extend(
            [
                "",
                "## Reviewed attributes",
                "",
                "| Attribute | Roles | Override | Override reason |",
                "|---|---|---:|---|",
            ]
        )
        for item in sorted(self.attribute_reviews, key=lambda review: review.attribute):
            reason = item.override_reason or "none"
            lines.append(
                f"| {_markdown_code(item.attribute)} | "
                f"{', '.join(f'`{role}`' for role in sorted(item.roles))} | "
                f"`{str(item.override_applied).lower()}` | `{reason}` |"
            )
        lines.extend(
            [
                "",
                "## Selected privacy attributes",
                "",
                "- Quasi-identifiers: "
                + ", ".join(
                    _markdown_code(item)
                    for item in sorted(self.selected_quasi_identifiers)
                ),
                "- Sensitive attributes: "
                + (
                    ", ".join(
                        _markdown_code(item)
                        for item in sorted(self.sensitive_attributes)
                    )
                    or "none"
                ),
                "",
                "## Configured and achieved privacy models",
                "",
                "| Model | Variant | Configured | Pre-transform | Post-transform |",
                "|---|---|---:|---:|---:|",
                "| k-anonymity | exact class size | "
                f"`{self.privacy_models.configured_k}` | "
                f"`{self.privacy_models.pre_achieved_k}` | "
                f"`{self.privacy_models.achieved_k}` |",
                _privacy_model_markdown_row("l-diversity", l_model, "l"),
                _privacy_model_markdown_row("t-closeness", t_model, "t"),
                "",
                "## Aggregate class and violation metrics",
                "",
                "| Stage | Privacy units | Classes | Smallest | Largest | Mean | "
                "Violating classes | Violating privacy units |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                _metrics_markdown_row("Pre-transform", self.pre_metrics),
                _metrics_markdown_row("Post-transform", self.post_metrics),
                "",
                "### Class-size histogram",
                "",
                "| Stage | Lower | Upper | Classes | Privacy units |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for stage, metrics in (
            ("Pre-transform", self.pre_metrics),
            ("Post-transform", self.post_metrics),
        ):
            for histogram_bin in sorted(
                metrics.class_size_histogram,
                key=lambda value: (value.lower_bound, value.upper_bound),
            ):
                lines.append(
                    f"| {stage} | `{histogram_bin.lower_bound}` | "
                    f"`{histogram_bin.upper_bound}` | "
                    f"`{histogram_bin.class_count}` | "
                    f"`{histogram_bin.privacy_unit_count}` |"
                )
        lines.extend(
            [
                "",
                "## Transformations and suppression",
                "",
                "| Attribute | Method | Affected privacy units | Hierarchy before | "
                "Hierarchy after |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for transformation in sorted(
            self.transformations,
            key=lambda value: (value.attribute, value.method),
        ):
            lines.append(
                f"| {_markdown_code(transformation.attribute)} | "
                f"`{transformation.method}` | "
                f"`{transformation.affected_privacy_unit_count}` | "
                f"`{_optional_number(transformation.hierarchy_level_before)}` | "
                f"`{_optional_number(transformation.hierarchy_level_after)}` |"
            )
        lines.extend(
            [
                "",
                "- Privacy units suppressed: "
                f"`{self.suppression.privacy_units_suppressed}`",
                f"- Rows suppressed: `{self.suppression.rows_suppressed}`",
                f"- Cells suppressed: `{self.suppression.cells_suppressed}`",
                f"- Suppression rate: `{self.suppression.suppression_rate}`",
                "- Suppression privacy-unit limit: "
                f"`{self.suppression.privacy_unit_limit}`",
                "",
                "## Aggregate utility",
                "",
                "| Metric | Before | After | Delta | Unit | Higher is better |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for utility_metric in sorted(self.utility, key=lambda value: value.metric):
            lines.append(
                f"| `{utility_metric.metric}` | `{utility_metric.before}` | "
                f"`{utility_metric.after}` | "
                f"`{utility_metric.after - utility_metric.before}` | "
                f"`{utility_metric.unit}` | "
                f"`{str(utility_metric.higher_is_better).lower()}` |"
            )
        lines.extend(
            [
                "",
                "## Search completeness and limits",
                "",
                f"- Strategy: `{self.search.strategy}`",
                f"- Complete: `{str(self.search.complete).lower()}`",
                f"- Optimality proven: `{str(self.search.optimality_proven).lower()}`",
                f"- Evaluated candidates: `{self.search.evaluated_candidates}`",
                "- Total candidates: "
                f"`{_optional_number(self.search.total_candidates)}`",
                "- Maximum quasi-identifiers: "
                f"`{self.search.maximum_quasi_identifiers}`",
                f"- Candidate limit: `{_optional_number(self.search.candidate_limit)}`",
                "- Evaluated suppression subsets: "
                f"`{self.search.suppression_subsets_evaluated}`",
                "- Total suppression subsets: "
                f"`{_optional_number(self.search.suppression_subsets_total)}`",
                "- Suppression-subset limit: "
                f"`{_optional_number(self.search.suppression_subset_limit)}`",
                "- Time limit seconds: "
                f"`{_optional_number(self.search.time_limit_seconds)}`",
                f"- Termination reason: `{self.search.termination_reason}`",
                "",
                "## Composition evidence",
                "",
                f"- Release count: `{self.composition.release_count}`",
                "- Longitudinal linkage assessed: "
                f"`{str(self.composition.longitudinal_linkage_assessed).lower()}`",
                "- Prior-release overlap assessed: "
                f"`{str(self.composition.prior_release_overlap_assessed).lower()}`",
                f"- Risk status: `{self.composition.risk_status}`",
                f"- Evidence digest: `{self.composition.evidence_digest}`",
                "",
                "## Limitations and unsupported modalities",
                "",
                "- Limitations: "
                + (
                    ", ".join(f"`{item}`" for item in sorted(self.limitations))
                    or "none"
                ),
                "- Unsupported modalities: "
                + (
                    ", ".join(
                        f"`{item}`" for item in sorted(self.unsupported_modalities)
                    )
                    or "none"
                ),
                "",
                "## Qualified expert review",
                "",
                "- Status: `pending_qualified_expert_review`",
                "- Qualified expert name: ____________________",
                "- Qualifications: ____________________",
                "- Methodology review: ____________________",
                "- Risk conclusion: ____________________",
                "- Review date: ____________________",
                "- Signature: ____________________",
                "",
            ]
        )
        return "\n".join(lines)

    def integrity_hash_matches(self) -> bool:
        """Return whether the allow-listed payload matches its integrity hash."""

        return self.integrity_hash == stable_hash(self._payload())

    def verify(self) -> bool:
        """Verify schema invariants and reject any payload tampering."""

        try:
            self._validate_relationships()
        except (TypeError, ValueError):
            return False
        return self.integrity_hash_matches()

    def require_valid(self) -> None:
        """Raise when the evidence has been modified or is internally invalid."""

        if not self.verify():
            raise ValueError("expert-review evidence integrity verification failed")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        verify: bool = True,
    ) -> ExpertReviewEvidenceReport:
        """Restore a strictly validated report and reject tampering by default."""

        report = _report_from_mapping(data)
        if verify:
            report.require_valid()
        return report

    @classmethod
    def from_json(
        cls,
        data: str | bytes,
        *,
        verify: bool = True,
    ) -> ExpertReviewEvidenceReport:
        """Restore a report from JSON and reject tampering by default."""

        try:
            payload = json.loads(
                data,
                object_pairs_hook=_strict_evidence_json_object,
                parse_constant=_reject_evidence_json_constant,
                parse_float=_parse_evidence_json_float,
            )
        except (
            _InvalidEvidenceJson,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            raise ValueError("invalid JSON for expert-review evidence") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("expert-review evidence JSON must contain an object")
        return cls.from_dict(payload, verify=verify)


def build_expert_review_evidence(
    *,
    digests: EvidenceDigests,
    assumptions: ReleaseAssumptions,
    attribute_reviews: tuple[AttributeRoleReview, ...],
    selected_quasi_identifiers: tuple[str, ...],
    sensitive_attributes: tuple[str, ...],
    privacy_models: PrivacyModelEvidence,
    pre_metrics: AggregateRiskMetrics,
    post_metrics: AggregateRiskMetrics,
    transformations: tuple[TransformationAggregate, ...],
    suppression: SuppressionAggregate,
    utility: tuple[UtilityAggregate, ...],
    search: SearchEvidence,
    composition: CompositionEvidence,
    limitations: tuple[str, ...] = (),
    unsupported_modalities: tuple[str, ...] = (),
) -> ExpertReviewEvidenceReport:
    """Build a deterministic evidence bundle from typed aggregate inputs.

    There are deliberately no parameters for records, class keys, samples,
    identifiers, paths, or transformed data. Detailed release-context notes
    remain outside the bundle and are bound through ``notes_digest``.
    """

    placeholder_hash = "sha256:" + ("0" * 64)
    report = ExpertReviewEvidenceReport(
        digests=digests,
        assumptions=assumptions,
        attribute_reviews=attribute_reviews,
        selected_quasi_identifiers=selected_quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
        privacy_models=privacy_models,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        transformations=transformations,
        suppression=suppression,
        utility=utility,
        search=search,
        composition=composition,
        limitations=limitations,
        unsupported_modalities=unsupported_modalities,
        integrity_hash=placeholder_hash,
    )
    return replace(report, integrity_hash=stable_hash(report._payload()))


def _report_from_mapping(data: Mapping[str, Any]) -> ExpertReviewEvidenceReport:
    payload = _object(
        data,
        {
            "schema_version",
            "report_type",
            "title",
            "disclaimer",
            "digests",
            "assumptions",
            "attribute_review",
            "selected_quasi_identifiers",
            "sensitive_attributes",
            "privacy_models",
            "metrics",
            "transformations",
            "suppression",
            "utility",
            "search",
            "composition",
            "limitations",
            "unsupported_modalities",
            "qualified_expert_review",
            "integrity_hash",
        },
        "report",
    )
    _validate_review_placeholders(payload["qualified_expert_review"])
    metrics = _object(
        payload["metrics"], {"pre_transform", "post_transform"}, "metrics"
    )
    return ExpertReviewEvidenceReport(
        schema_version=payload["schema_version"],
        report_type=payload["report_type"],
        title=payload["title"],
        disclaimer=payload["disclaimer"],
        digests=_parse_digests(payload["digests"]),
        assumptions=_parse_assumptions(payload["assumptions"]),
        attribute_reviews=tuple(
            _parse_attribute_review(item)
            for item in _array(payload["attribute_review"], "attribute review")
        ),
        selected_quasi_identifiers=_string_array(
            payload["selected_quasi_identifiers"], "selected quasi-identifiers"
        ),
        sensitive_attributes=_string_array(
            payload["sensitive_attributes"], "sensitive attributes"
        ),
        privacy_models=_parse_privacy_models(payload["privacy_models"]),
        pre_metrics=_parse_metrics(metrics["pre_transform"]),
        post_metrics=_parse_metrics(metrics["post_transform"]),
        transformations=tuple(
            _parse_transformation(item)
            for item in _array(payload["transformations"], "transformations")
        ),
        suppression=_parse_suppression(payload["suppression"]),
        utility=tuple(
            _parse_utility(item) for item in _array(payload["utility"], "utility")
        ),
        search=_parse_search(
            payload["search"],
            schema_version=payload["schema_version"],
        ),
        composition=_parse_composition(payload["composition"]),
        limitations=_string_array(payload["limitations"], "limitations"),
        unsupported_modalities=_string_array(
            payload["unsupported_modalities"], "unsupported modalities"
        ),
        integrity_hash=payload["integrity_hash"],
    )


def _parse_digests(value: Any) -> EvidenceDigests:
    item = _object(
        value,
        {
            "source_dataset",
            "dataset",
            "schema",
            "policy",
            "hierarchy",
            "config",
            "software",
        },
        "digests",
    )
    return EvidenceDigests(**item)


def _parse_assumptions(value: Any) -> ReleaseAssumptions:
    item = _object(
        value,
        {
            "privacy_unit",
            "population_scope",
            "release_model",
            "recipient_model",
            "auxiliary_data_model",
            "notes_digest",
        },
        "assumptions",
    )
    return ReleaseAssumptions(**item)


def _parse_attribute_review(value: Any) -> AttributeRoleReview:
    item = _object(
        value,
        {"attribute", "roles", "override_applied", "override_reason"},
        "attribute review",
    )
    return AttributeRoleReview(
        attribute=item["attribute"],
        roles=_string_array(item["roles"], "attribute roles"),
        override_applied=item["override_applied"],
        override_reason=item["override_reason"],
    )


def _parse_privacy_models(value: Any) -> PrivacyModelEvidence:
    item = _object(
        value, {"k_anonymity", "l_diversity", "t_closeness"}, "privacy models"
    )
    k_model = _object(
        item["k_anonymity"],
        {"configured_k", "pre_achieved_k", "achieved_k"},
        "k-anonymity",
    )
    l_model = item["l_diversity"]
    t_model = item["t_closeness"]
    l_values: dict[str, Any] = {
        "l_variant": None,
        "configured_l": None,
        "pre_achieved_l": None,
        "achieved_l": None,
    }
    if l_model is not None:
        parsed_l = _object(
            l_model,
            {"variant", "configured_l", "pre_achieved_l", "achieved_l"},
            "l-diversity",
        )
        l_values = {
            "l_variant": parsed_l["variant"],
            "configured_l": parsed_l["configured_l"],
            "pre_achieved_l": parsed_l["pre_achieved_l"],
            "achieved_l": parsed_l["achieved_l"],
        }
    t_values: dict[str, Any] = {
        "t_variant": None,
        "configured_t": None,
        "pre_achieved_t": None,
        "achieved_t": None,
    }
    if t_model is not None:
        parsed_t = _object(
            t_model,
            {"variant", "configured_t", "pre_achieved_t", "achieved_t"},
            "t-closeness",
        )
        t_values = {
            "t_variant": parsed_t["variant"],
            "configured_t": parsed_t["configured_t"],
            "pre_achieved_t": parsed_t["pre_achieved_t"],
            "achieved_t": parsed_t["achieved_t"],
        }
    return PrivacyModelEvidence(**k_model, **l_values, **t_values)


def _parse_metrics(value: Any) -> AggregateRiskMetrics:
    item = _object(
        value,
        {
            "privacy_unit_count",
            "equivalence_class_count",
            "class_sizes",
            "violations",
        },
        "aggregate metrics",
    )
    class_sizes = _object(
        item["class_sizes"],
        {"smallest", "largest", "mean", "histogram"},
        "class-size metrics",
    )
    violations = _object(
        item["violations"],
        {
            "k_class_count",
            "l_class_count",
            "t_class_count",
            "any_class_count",
            "privacy_unit_count",
        },
        "violation metrics",
    )
    return AggregateRiskMetrics(
        privacy_unit_count=item["privacy_unit_count"],
        equivalence_class_count=item["equivalence_class_count"],
        smallest_class_size=class_sizes["smallest"],
        largest_class_size=class_sizes["largest"],
        mean_class_size=class_sizes["mean"],
        class_size_histogram=tuple(
            _parse_class_size_bin(histogram_item)
            for histogram_item in _array(
                class_sizes["histogram"], "class-size histogram"
            )
        ),
        k_violating_class_count=violations["k_class_count"],
        l_violating_class_count=violations["l_class_count"],
        t_violating_class_count=violations["t_class_count"],
        any_violating_class_count=violations["any_class_count"],
        violating_privacy_unit_count=violations["privacy_unit_count"],
    )


def _parse_class_size_bin(value: Any) -> ClassSizeBin:
    item = _object(
        value,
        {
            "lower_bound",
            "upper_bound",
            "class_count",
            "privacy_unit_count",
        },
        "class-size bin",
    )
    return ClassSizeBin(**item)


def _parse_transformation(value: Any) -> TransformationAggregate:
    item = _object(
        value,
        {
            "attribute",
            "method",
            "affected_privacy_unit_count",
            "hierarchy_level_before",
            "hierarchy_level_after",
        },
        "transformation",
    )
    return TransformationAggregate(**item)


def _parse_suppression(value: Any) -> SuppressionAggregate:
    item = _object(
        value,
        {
            "privacy_units_suppressed",
            "rows_suppressed",
            "cells_suppressed",
            "suppression_rate",
            "privacy_unit_limit",
        },
        "suppression",
    )
    return SuppressionAggregate(**item)


def _parse_utility(value: Any) -> UtilityAggregate:
    item = _object(
        value,
        {
            "metric",
            "before",
            "after",
            "absolute_delta",
            "unit",
            "higher_is_better",
        },
        "utility",
    )
    utility = UtilityAggregate(
        metric=item["metric"],
        before=item["before"],
        after=item["after"],
        unit=item["unit"],
        higher_is_better=item["higher_is_better"],
    )
    if item["absolute_delta"] != utility.after - utility.before:
        raise ValueError("utility absolute_delta does not match aggregate values")
    return utility


def _parse_search(value: Any, *, schema_version: int) -> SearchEvidence:
    if schema_version not in _SUPPORTED_EVIDENCE_SCHEMA_VERSIONS:
        raise ValueError("unsupported expert-review evidence schema version")
    expected_fields = {
        "strategy",
        "complete",
        "evaluated_candidates",
        "total_candidates",
        "maximum_quasi_identifiers",
        "candidate_limit",
        "suppression_subsets_evaluated",
        "suppression_subsets_total",
        "suppression_subset_limit",
        "time_limit_seconds",
        "termination_reason",
    }
    if schema_version >= 3:
        expected_fields.add("optimality_proven")
    item = _object(
        value,
        expected_fields,
        "search",
    )
    if schema_version == 2:
        item["optimality_proven"] = (
            item["complete"] or item["termination_reason"] == "optimal_candidate_found"
        )
    return SearchEvidence(**item)


def _parse_composition(value: Any) -> CompositionEvidence:
    item = _object(
        value,
        {
            "release_count",
            "longitudinal_linkage_assessed",
            "prior_release_overlap_assessed",
            "risk_status",
            "evidence_digest",
        },
        "composition",
    )
    return CompositionEvidence(**item)


def _validate_review_placeholders(value: Any) -> None:
    item = _object(
        value,
        {
            "status",
            "qualified_expert_name",
            "qualifications",
            "methodology_review",
            "risk_conclusion",
            "review_date",
            "signature",
        },
        "qualified expert review",
    )
    if item["status"] != "pending_qualified_expert_review":
        raise ValueError("qualified-expert review status must remain pending")
    if any(
        item[field] is not None
        for field in (
            "qualified_expert_name",
            "qualifications",
            "methodology_review",
            "risk_conclusion",
            "review_date",
            "signature",
        )
    ):
        raise ValueError("qualified-expert review fields must remain placeholders")


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if set(value) != fields:
        raise ValueError(f"{name} contains missing or unsupported fields")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return {field: value[field] for field in fields}


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


def _string_array(value: Any, name: str) -> tuple[str, ...]:
    values = _array(value, name)
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain strings")
    return tuple(values)


def _require_digest(value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("evidence digests must use canonical sha256 values")


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must use a safe metadata identifier")


def _require_attribute_name(value: Any, name: str) -> None:
    """Validate a source-schema label without imposing programming syntax."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(
            unicodedata.category(character).startswith("C")
            or unicodedata.category(character) in {"Zl", "Zp"}
            for character in value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty source column name without surrounding "
            "whitespace or control characters"
        )


def _markdown_code(value: str) -> str:
    """Render one metadata value as a table-safe Markdown code span."""

    text = value.replace("|", r"\|")
    runs = re.findall(r"`+", text)
    fence = "`" * (max((len(run) for run in runs), default=0) + 1)
    if text.startswith("`") or text.endswith("`"):
        text = f" {text} "
    return f"{fence}{text}{fence}"


def _require_choice(value: Any, choices: frozenset[str], name: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"unsupported {name}")


def _require_int(value: Any, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _require_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be no greater than {maximum}")


def _require_string_tuple(value: Any, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain strings")


def _require_typed_tuple(value: Any, expected: type[Any], name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not all(isinstance(item, expected) for item in value):
        raise TypeError(f"{name} contains an unsupported input type")


def _validate_optional_model(
    *,
    variant: Any,
    configured: Any,
    pre_achieved: Any,
    achieved: Any,
    variants: frozenset[str],
    name: str,
) -> None:
    values = (configured, pre_achieved, achieved)
    if variant is None:
        if any(value is not None for value in values):
            raise ValueError(f"{name} values require an explicit variant")
        return
    _require_choice(variant, variants, f"{name} variant")
    if any(value is None for value in values):
        raise ValueError(f"{name} requires configured and achieved values")
    for value in values:
        _require_number(value, f"{name} value", minimum=0.0)


def _privacy_model_markdown_row(
    name: str, model: Mapping[str, Any] | None, suffix: str
) -> str:
    if model is None:
        return f"| {name} | not configured | n/a | n/a | n/a |"
    return (
        f"| {name} | `{model['variant']}` | `{model[f'configured_{suffix}']}` | "
        f"`{model[f'pre_achieved_{suffix}']}` | "
        f"`{model[f'achieved_{suffix}']}` |"
    )


def _metrics_markdown_row(name: str, metrics: AggregateRiskMetrics) -> str:
    return (
        f"| {name} | `{metrics.privacy_unit_count}` | "
        f"`{metrics.equivalence_class_count}` | "
        f"`{metrics.smallest_class_size}` | `{metrics.largest_class_size}` | "
        f"`{metrics.mean_class_size}` | `{metrics.any_violating_class_count}` | "
        f"`{metrics.violating_privacy_unit_count}` |"
    )


def _optional_number(value: int | float | None) -> str:
    return "none" if value is None else str(value)


__all__ = [
    "EXPERT_REVIEW_EVIDENCE_DISCLAIMER",
    "EXPERT_REVIEW_EVIDENCE_SCHEMA_VERSION",
    "EXPERT_REVIEW_EVIDENCE_TITLE",
    "AggregateRiskMetrics",
    "AttributeRoleReview",
    "ClassSizeBin",
    "CompositionEvidence",
    "EvidenceDigests",
    "ExpertReviewEvidenceReport",
    "PrivacyModelEvidence",
    "ReleaseAssumptions",
    "SearchEvidence",
    "SuppressionAggregate",
    "TransformationAggregate",
    "UtilityAggregate",
    "build_expert_review_evidence",
]
