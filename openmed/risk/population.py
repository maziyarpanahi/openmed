"""Exact offline risk assessment against a supplied reference population.

This module measures population-aware disclosure risk without network access or
statistical extrapolation. It makes two explicit model assumptions:

1. the supplied reference population represents the anticipated attack
   population for the declared quasi-identifiers; and
2. every sample analysis unit is contained in that reference population under
   compatible row-level or keyed privacy-unit semantics.

Those assumptions require qualified expert review. A reference-population
assessment is not an Expert Determination or a compliance certificate. When
sample profile frequencies exceed their reference frequencies, or when a
sample profile is absent from the reference, the result fails closed.

For longitudinal data, an analysis-unit profile is the sorted multiset of its
joint row-level quasi-identifier tuples. This preserves correlations between
columns within a row as well as repeated-row multiplicity. Public
serialization contains aggregate counts, rates, fixed assumptions, and
binding digests only. It never contains raw profile keys, cell values, privacy
unit identifiers, or source paths.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any, Final

from openmed.core.audit import stable_hash

from .release import (
    _canonical_digest_scalar,
    _column_tuple,
    _materialize_rows,
    _optional_column,
    _schema_digest,
)

__all__ = ["PopulationRiskAssessment", "assess_population_risk"]

_SCHEMA_VERSION: Final = 1
_MODEL: Final = "exact_reference_population"
_PROFILE_MODEL: Final = "analysis_unit_joint_row_multiset"
_MODEL_ASSUMPTIONS: Final = (
    "reference_population_represents_anticipated_attack_population",
    "sample_units_are_contained_in_reference_population",
)
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SERIALIZED_INT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "quasi_identifier_count",
        "sample_row_count",
        "reference_row_count",
        "sample_unit_count",
        "reference_unit_count",
        "sample_profile_count",
        "reference_profile_count",
        "matched_sample_unit_count",
        "unmatched_sample_unit_count",
        "population_singleton_count",
        "reference_frequency_violation_count",
        "reference_frequency_violation_unit_count",
        "achieved_k_map",
        "target_k_map",
    }
)
_SERIALIZED_FLOAT_FIELDS: Final = frozenset(
    {
        "population_singleton_rate",
        "max_exact_linkage_risk",
        "mean_exact_linkage_risk",
        "max_delta_presence",
        "mean_delta_presence",
        "max_delta_presence_threshold",
    }
)
_SERIALIZED_BOOL_FIELDS: Final = frozenset(
    {
        "not_an_expert_determination",
        "qualified_expert_review_required",
        "reference_model_consistent",
        "meets_k_map",
        "meets_delta_presence",
        "meets_policy",
    }
)
_SERIALIZED_STRING_FIELDS: Final = frozenset(
    {
        "artifact",
        "detail_level",
        "model",
        "profile_model",
        "analysis_unit_model",
        "sample_digest",
        "reference_population_digest",
        "schema_digest",
        "policy_digest",
        "integrity_digest",
    }
)
_SERIALIZED_PAYLOAD_KEYS: Final = frozenset(
    {
        *_SERIALIZED_INT_FIELDS,
        *_SERIALIZED_FLOAT_FIELDS,
        *_SERIALIZED_BOOL_FIELDS,
        *_SERIALIZED_STRING_FIELDS,
        "model_assumptions",
    }
)


@dataclass(frozen=True)
class PopulationRiskAssessment:
    """Aggregate-only exact reference-population risk evidence.

    ``achieved_k_map`` is the smallest reference-population frequency among
    sample profiles and is zero when any sample profile is unmatched.
    Exact-linkage risk is ``1 / F`` for matched profiles with reference
    frequency ``F``; unmatched profiles receive the conservative value ``1``.
    Delta-presence is ``f / F``, where ``f`` is the sample frequency. An
    unmatched profile receives the conservative value ``1``. Ratios greater
    than one remain visible as evidence that the supplied reference population
    is inconsistent with the sample.

    The report does not validate whether the supplied population and
    quasi-identifiers match a real anticipated attacker. A qualified expert
    must review that model assumption.
    """

    schema_version: int
    model: str
    profile_model: str
    model_assumptions: tuple[str, ...]
    analysis_unit_model: str
    quasi_identifier_count: int
    sample_row_count: int
    reference_row_count: int
    sample_unit_count: int
    reference_unit_count: int
    sample_profile_count: int
    reference_profile_count: int
    matched_sample_unit_count: int
    unmatched_sample_unit_count: int
    population_singleton_count: int
    population_singleton_rate: float
    reference_frequency_violation_count: int
    reference_frequency_violation_unit_count: int
    achieved_k_map: int
    target_k_map: int
    max_exact_linkage_risk: float
    mean_exact_linkage_risk: float
    max_delta_presence: float
    mean_delta_presence: float
    max_delta_presence_threshold: float
    sample_digest: str
    reference_population_digest: str
    schema_digest: str
    policy_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported population-risk schema version")
        if self.model != _MODEL:
            raise ValueError("unsupported population-risk model")
        if self.profile_model != _PROFILE_MODEL:
            raise ValueError("unsupported population profile model")
        if self.model_assumptions != _MODEL_ASSUMPTIONS:
            raise ValueError("population-risk model assumptions must remain explicit")
        if self.analysis_unit_model not in {"row", "keyed"}:
            raise ValueError("analysis_unit_model must be 'row' or 'keyed'")

        positive_counts = {
            "quasi_identifier_count": self.quasi_identifier_count,
            "sample_row_count": self.sample_row_count,
            "reference_row_count": self.reference_row_count,
            "sample_unit_count": self.sample_unit_count,
            "reference_unit_count": self.reference_unit_count,
            "sample_profile_count": self.sample_profile_count,
            "reference_profile_count": self.reference_profile_count,
            "target_k_map": self.target_k_map,
        }
        for count_name, count in positive_counts.items():
            if type(count) is not int or count < 1:
                raise ValueError(f"{count_name} must be an integer >= 1")

        bounded_counts = {
            "matched_sample_unit_count": self.matched_sample_unit_count,
            "unmatched_sample_unit_count": self.unmatched_sample_unit_count,
            "population_singleton_count": self.population_singleton_count,
            "reference_frequency_violation_count": (
                self.reference_frequency_violation_count
            ),
            "reference_frequency_violation_unit_count": (
                self.reference_frequency_violation_unit_count
            ),
        }
        for count_name, count in bounded_counts.items():
            if type(count) is not int or not 0 <= count <= self.sample_unit_count:
                raise ValueError(
                    f"{count_name} must be an integer between 0 and sample_unit_count"
                )
        if (
            self.matched_sample_unit_count + self.unmatched_sample_unit_count
            != self.sample_unit_count
        ):
            raise ValueError(
                "matched and unmatched sample-unit counts are inconsistent"
            )
        if self.sample_unit_count > self.sample_row_count:
            raise ValueError("sample_unit_count cannot exceed sample_row_count")
        if self.reference_unit_count > self.reference_row_count:
            raise ValueError("reference_unit_count cannot exceed reference_row_count")
        if self.analysis_unit_model == "row" and (
            self.sample_unit_count != self.sample_row_count
            or self.reference_unit_count != self.reference_row_count
        ):
            raise ValueError(
                "row-level analysis requires one analysis unit per source row"
            )
        if self.sample_profile_count > self.sample_unit_count:
            raise ValueError("sample_profile_count cannot exceed sample_unit_count")
        if self.reference_profile_count > self.reference_unit_count:
            raise ValueError(
                "reference_profile_count cannot exceed reference_unit_count"
            )
        if (
            self.sample_profile_count
            > self.reference_profile_count + self.unmatched_sample_unit_count
        ):
            raise ValueError(
                "sample profiles exceed the matched and unmatched profile bounds"
            )
        minimum_sample_profile_partitions = int(
            self.matched_sample_unit_count > 0
        ) + int(self.unmatched_sample_unit_count > 0)
        if self.sample_profile_count < minimum_sample_profile_partitions:
            raise ValueError(
                "sample profiles cannot combine matched and unmatched partitions"
            )
        if self.population_singleton_count > self.matched_sample_unit_count:
            raise ValueError(
                "population singleton units cannot exceed matched sample units"
            )
        if self.reference_frequency_violation_count > self.sample_profile_count:
            raise ValueError(
                "frequency-violation profile count cannot exceed sample profiles"
            )
        if (
            self.reference_frequency_violation_count
            > self.reference_frequency_violation_unit_count
        ):
            raise ValueError(
                "frequency-violation profile count cannot exceed affected units"
            )
        if (
            self.reference_frequency_violation_unit_count
            < self.unmatched_sample_unit_count
        ):
            raise ValueError(
                "frequency-violation units cannot be fewer than unmatched units"
            )
        if (self.reference_frequency_violation_count == 0) != (
            self.reference_frequency_violation_unit_count == 0
        ):
            raise ValueError(
                "frequency-violation profile and unit counts must both be zero "
                "or both be positive"
            )
        minimum_violation_profiles = int(self.unmatched_sample_unit_count > 0) + int(
            self.reference_frequency_violation_unit_count
            > self.unmatched_sample_unit_count
        )
        if self.reference_frequency_violation_count < minimum_violation_profiles:
            raise ValueError(
                "frequency-violation profiles do not cover unmatched and matched "
                "violation partitions"
            )

        if type(self.achieved_k_map) is not int or self.achieved_k_map < 0:
            raise ValueError("achieved_k_map must be a non-negative integer")
        if self.achieved_k_map > self.reference_unit_count:
            raise ValueError("achieved_k_map cannot exceed reference_unit_count")
        if self.unmatched_sample_unit_count and self.achieved_k_map != 0:
            raise ValueError("unmatched sample profiles require achieved_k_map=0")
        if not self.unmatched_sample_unit_count and self.achieved_k_map < 1:
            raise ValueError("fully matched samples require achieved_k_map >= 1")
        if not self.unmatched_sample_unit_count:
            maximum_reference_profile_frequency = (
                self.reference_unit_count - self.reference_profile_count + 1
            )
            if self.achieved_k_map > maximum_reference_profile_frequency:
                raise ValueError(
                    "achieved_k_map exceeds the reference profile-frequency bound"
                )
            has_population_singletons = self.population_singleton_count > 0
            if (self.achieved_k_map == 1) != has_population_singletons:
                raise ValueError(
                    "achieved_k_map and population singleton counts are inconsistent"
                )
            minimum_reference_units = (
                self.sample_profile_count * self.achieved_k_map
                + self.reference_profile_count
                - self.sample_profile_count
            )
            if self.reference_unit_count < minimum_reference_units:
                raise ValueError(
                    "reference units cannot support every matched sample profile "
                    "at the achieved k-map"
                )
            if (
                self.reference_frequency_violation_count == 0
                and self.sample_unit_count
                > (
                    self.reference_unit_count
                    - self.reference_profile_count
                    + self.sample_profile_count
                )
            ):
                raise ValueError(
                    "model-consistent sample units exceed the reference units "
                    "available to matched profiles"
                )
            if (
                self.reference_profile_count == 1
                and self.achieved_k_map != self.reference_unit_count
            ):
                raise ValueError(
                    "a single reference profile must bind achieved_k_map to "
                    "reference_unit_count"
                )

        rate_fields = {
            "population_singleton_rate": self.population_singleton_rate,
            "max_exact_linkage_risk": self.max_exact_linkage_risk,
            "mean_exact_linkage_risk": self.mean_exact_linkage_risk,
            "max_delta_presence": self.max_delta_presence,
            "mean_delta_presence": self.mean_delta_presence,
            "max_delta_presence_threshold": self.max_delta_presence_threshold,
        }
        for rate_name, rate_value in rate_fields.items():
            if (
                not isinstance(rate_value, (int, float))
                or isinstance(rate_value, bool)
                or not math.isfinite(float(rate_value))
                or float(rate_value) < 0
            ):
                raise ValueError(f"{rate_name} must be a finite non-negative number")
            object.__setattr__(self, rate_name, float(rate_value))
        for name in (
            "population_singleton_rate",
            "max_exact_linkage_risk",
            "mean_exact_linkage_risk",
            "max_delta_presence_threshold",
        ):
            if float(getattr(self, name)) > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.mean_exact_linkage_risk > self.max_exact_linkage_risk:
            raise ValueError("mean exact-linkage risk cannot exceed maximum risk")
        if self.mean_delta_presence > self.max_delta_presence:
            raise ValueError("mean delta-presence cannot exceed maximum")
        if self.max_exact_linkage_risk <= 0 or self.mean_exact_linkage_risk <= 0:
            raise ValueError("exact-linkage risks must be positive")
        if self.max_delta_presence <= 0 or self.mean_delta_presence <= 0:
            raise ValueError("delta-presence values must be positive")
        if self.max_delta_presence > self.sample_unit_count:
            raise ValueError("maximum delta-presence cannot exceed sample_unit_count")
        if (
            self.max_delta_presence + 1e-15 < self.max_exact_linkage_risk
            or self.mean_delta_presence + 1e-15 < self.mean_exact_linkage_risk
        ):
            raise ValueError("delta-presence cannot be lower than exact-linkage risk")
        high_exact_risk_units = (
            self.unmatched_sample_unit_count + self.population_singleton_count
        )
        minimum_mean_exact_risk = (
            high_exact_risk_units
            + (self.sample_unit_count - high_exact_risk_units)
            / self.reference_unit_count
        ) / self.sample_unit_count
        if self.mean_exact_linkage_risk + 1e-15 < minimum_mean_exact_risk:
            raise ValueError(
                "mean exact-linkage risk is below the aggregate count bound"
            )
        if high_exact_risk_units and self.max_delta_presence + 1e-15 < 1.0:
            raise ValueError(
                "unmatched or population-singleton units require "
                "maximum delta-presence >= 1"
            )
        if (
            self.reference_frequency_violation_count == 0
            and self.max_delta_presence > 1.0 + 1e-15
        ):
            raise ValueError("delta-presence above one requires a frequency violation")
        has_matched_frequency_violation = (
            self.reference_frequency_violation_unit_count
            > self.unmatched_sample_unit_count
        )
        if (self.max_delta_presence > 1.0) != has_matched_frequency_violation:
            raise ValueError(
                "maximum delta-presence and matched frequency violations are "
                "inconsistent"
            )
        if self.matched_sample_unit_count == 0 and (
            self.max_delta_presence != 1.0 or self.mean_delta_presence != 1.0
        ):
            raise ValueError(
                "fully unmatched samples require exact delta-presence of one"
            )
        if self.matched_sample_unit_count:
            matched_profile_upper_bound = min(
                self.reference_profile_count,
                self.matched_sample_unit_count,
                self.sample_profile_count - int(self.unmatched_sample_unit_count > 0),
            )
            matched_reference_unit_upper_bound = (
                self.reference_unit_count
                - self.reference_profile_count
                + matched_profile_upper_bound
            )
            if (
                not has_matched_frequency_violation
                and self.population_singleton_count > matched_profile_upper_bound
            ):
                raise ValueError(
                    "population singleton units exceed the nonviolating matched "
                    "profile bound"
                )
            minimum_matched_max_delta = (
                self.matched_sample_unit_count / matched_reference_unit_upper_bound
            )
            minimum_mean_delta = (
                self.unmatched_sample_unit_count
                + (
                    self.matched_sample_unit_count
                    * self.matched_sample_unit_count
                    / matched_reference_unit_upper_bound
                )
            ) / self.sample_unit_count
            minimum_max_delta = max(
                1.0 if self.unmatched_sample_unit_count else 0.0,
                minimum_matched_max_delta,
            )
            if (
                self.max_delta_presence + 1e-15 < minimum_max_delta
                or self.mean_delta_presence + 1e-15 < minimum_mean_delta
            ):
                raise ValueError(
                    "delta-presence is below the aggregate sample/reference "
                    "profile-count bound"
                )
        if not self.unmatched_sample_unit_count and self.sample_profile_count == 1:
            expected_single_profile_exact_risk = 1.0 / self.achieved_k_map
            expected_single_profile_delta = self.sample_unit_count / self.achieved_k_map
            expected_singleton_count = (
                self.sample_unit_count if self.achieved_k_map == 1 else 0
            )
            expected_violation_count = int(self.sample_unit_count > self.achieved_k_map)
            expected_violation_units = (
                self.sample_unit_count if expected_violation_count else 0
            )
            if (
                not math.isclose(
                    self.mean_exact_linkage_risk,
                    expected_single_profile_exact_risk,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    self.max_delta_presence,
                    expected_single_profile_delta,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    self.mean_delta_presence,
                    expected_single_profile_delta,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or self.population_singleton_count != expected_singleton_count
                or self.reference_frequency_violation_count != expected_violation_count
                or self.reference_frequency_violation_unit_count
                != expected_violation_units
            ):
                raise ValueError(
                    "single-profile population-risk aggregates are inconsistent"
                )
        expected_singleton_rate = (
            self.population_singleton_count / self.sample_unit_count
        )
        if not math.isclose(
            self.population_singleton_rate,
            expected_singleton_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("population singleton rate is inconsistent")
        expected_max_exact_risk = (
            1.0 if self.unmatched_sample_unit_count else 1.0 / self.achieved_k_map
        )
        if not math.isclose(
            self.max_exact_linkage_risk,
            expected_max_exact_risk,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("maximum exact-linkage risk is inconsistent")

        for name in (
            "sample_digest",
            "reference_population_digest",
            "schema_digest",
            "policy_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
                raise ValueError(f"{name} must be a canonical SHA-256 digest")

    @property
    def reference_model_consistent(self) -> bool:
        """Whether every sample profile frequency fits the reference model."""

        return (
            self.unmatched_sample_unit_count == 0
            and self.reference_frequency_violation_count == 0
        )

    @property
    def meets_k_map(self) -> bool:
        """Whether the exact k-map threshold is met without model inconsistency."""

        return (
            self.reference_model_consistent and self.achieved_k_map >= self.target_k_map
        )

    @property
    def meets_delta_presence(self) -> bool:
        """Whether the delta-presence threshold is met without inconsistency."""

        return (
            self.reference_model_consistent
            and self.max_delta_presence <= self.max_delta_presence_threshold
        )

    @property
    def meets_policy(self) -> bool:
        """Whether both declared population-risk thresholds are met."""

        return self.meets_k_map and self.meets_delta_presence

    @property
    def matched_unit_count(self) -> int:
        """Compatibility alias for matched sample analysis units."""

        return self.matched_sample_unit_count

    @property
    def unmatched_unit_count(self) -> int:
        """Compatibility alias for unmatched sample analysis units."""

        return self.unmatched_sample_unit_count

    @property
    def digest(self) -> str:
        """Bind the complete canonical aggregate-safe assessment payload."""

        return stable_hash(self._aggregate_payload())

    @property
    def integrity_digest(self) -> str:
        """Compatibility alias for the canonical assessment digest."""

        return self.digest

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate evidence without raw profile data."""

        return {
            **self._aggregate_payload(),
            "integrity_digest": self.integrity_digest,
        }

    def _aggregate_payload(self) -> dict[str, Any]:
        """Return the canonical safe payload before its self-excluding digest."""

        return {
            "schema_version": self.schema_version,
            "artifact": "deidentification_population_risk_assessment",
            "detail_level": "aggregate_phi_safe",
            "not_an_expert_determination": True,
            "qualified_expert_review_required": True,
            "model": self.model,
            "profile_model": self.profile_model,
            "model_assumptions": list(self.model_assumptions),
            "analysis_unit_model": self.analysis_unit_model,
            "quasi_identifier_count": self.quasi_identifier_count,
            "sample_row_count": self.sample_row_count,
            "reference_row_count": self.reference_row_count,
            "sample_unit_count": self.sample_unit_count,
            "reference_unit_count": self.reference_unit_count,
            "sample_profile_count": self.sample_profile_count,
            "reference_profile_count": self.reference_profile_count,
            "matched_sample_unit_count": self.matched_sample_unit_count,
            "unmatched_sample_unit_count": self.unmatched_sample_unit_count,
            "population_singleton_count": self.population_singleton_count,
            "population_singleton_rate": self.population_singleton_rate,
            "reference_frequency_violation_count": (
                self.reference_frequency_violation_count
            ),
            "reference_frequency_violation_unit_count": (
                self.reference_frequency_violation_unit_count
            ),
            "reference_model_consistent": self.reference_model_consistent,
            "achieved_k_map": self.achieved_k_map,
            "target_k_map": self.target_k_map,
            "meets_k_map": self.meets_k_map,
            "max_exact_linkage_risk": self.max_exact_linkage_risk,
            "mean_exact_linkage_risk": self.mean_exact_linkage_risk,
            "max_delta_presence": self.max_delta_presence,
            "mean_delta_presence": self.mean_delta_presence,
            "max_delta_presence_threshold": self.max_delta_presence_threshold,
            "meets_delta_presence": self.meets_delta_presence,
            "meets_policy": self.meets_policy,
            "sample_digest": self.sample_digest,
            "reference_population_digest": self.reference_population_digest,
            "schema_digest": self.schema_digest,
            "policy_digest": self.policy_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PopulationRiskAssessment:
        """Parse and integrity-check a canonical aggregate assessment.

        Unknown, missing, non-canonical, or internally inconsistent fields are
        rejected. The integrity digest detects mutation of a saved artifact;
        an external signature over :attr:`digest` is still required when
        authenticity or provenance matters.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("population-risk payload must be a mapping")
        data = dict(payload)
        if any(type(key) is not str for key in data):
            raise ValueError("population-risk payload keys must be strings")
        expected_keys = _SERIALIZED_PAYLOAD_KEYS
        if set(data) != expected_keys:
            raise ValueError("population-risk payload has missing or unknown fields")

        for field in _SERIALIZED_INT_FIELDS:
            if type(data[field]) is not int:
                raise TypeError(f"population-risk field {field!r} must be an integer")
        for field in _SERIALIZED_FLOAT_FIELDS:
            if type(data[field]) is not float:
                raise TypeError(f"population-risk field {field!r} must be a float")
        for field in _SERIALIZED_BOOL_FIELDS:
            if type(data[field]) is not bool:
                raise TypeError(f"population-risk field {field!r} must be a boolean")
        for field in _SERIALIZED_STRING_FIELDS:
            if type(data[field]) is not str:
                raise TypeError(f"population-risk field {field!r} must be a string")
        assumptions = data["model_assumptions"]
        if type(assumptions) is not list or any(
            type(item) is not str for item in assumptions
        ):
            raise TypeError(
                "population-risk field 'model_assumptions' must be a string list"
            )

        supplied_digest = data["integrity_digest"]
        if not _DIGEST_RE.fullmatch(supplied_digest):
            raise ValueError(
                "population-risk integrity_digest must be a canonical SHA-256 digest"
            )
        aggregate_payload = {
            field: data[field] for field in data if field != "integrity_digest"
        }
        if stable_hash(aggregate_payload) != supplied_digest:
            raise ValueError("population-risk integrity digest mismatch")

        assessment = cls(
            schema_version=data["schema_version"],
            model=data["model"],
            profile_model=data["profile_model"],
            model_assumptions=tuple(assumptions),
            analysis_unit_model=data["analysis_unit_model"],
            quasi_identifier_count=data["quasi_identifier_count"],
            sample_row_count=data["sample_row_count"],
            reference_row_count=data["reference_row_count"],
            sample_unit_count=data["sample_unit_count"],
            reference_unit_count=data["reference_unit_count"],
            sample_profile_count=data["sample_profile_count"],
            reference_profile_count=data["reference_profile_count"],
            matched_sample_unit_count=data["matched_sample_unit_count"],
            unmatched_sample_unit_count=data["unmatched_sample_unit_count"],
            population_singleton_count=data["population_singleton_count"],
            population_singleton_rate=data["population_singleton_rate"],
            reference_frequency_violation_count=data[
                "reference_frequency_violation_count"
            ],
            reference_frequency_violation_unit_count=data[
                "reference_frequency_violation_unit_count"
            ],
            achieved_k_map=data["achieved_k_map"],
            target_k_map=data["target_k_map"],
            max_exact_linkage_risk=data["max_exact_linkage_risk"],
            mean_exact_linkage_risk=data["mean_exact_linkage_risk"],
            max_delta_presence=data["max_delta_presence"],
            mean_delta_presence=data["mean_delta_presence"],
            max_delta_presence_threshold=data["max_delta_presence_threshold"],
            sample_digest=data["sample_digest"],
            reference_population_digest=data["reference_population_digest"],
            schema_digest=data["schema_digest"],
            policy_digest=data["policy_digest"],
        )
        if assessment.to_dict() != data:
            raise ValueError(
                "population-risk payload is not the canonical assessment representation"
            )
        return assessment

    @classmethod
    def from_json(cls, payload: str) -> PopulationRiskAssessment:
        """Parse strict JSON into a verified aggregate assessment."""

        if type(payload) is not str:
            raise TypeError("population-risk JSON payload must be a string")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            raise ValueError("invalid population-risk JSON payload") from None
        if not isinstance(decoded, Mapping):
            raise ValueError("population-risk JSON payload must be an object")
        return cls.from_dict(decoded)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize aggregate evidence deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )


def assess_population_risk(
    sample: Any,
    reference_population: Any,
    quasi_identifiers: Sequence[str],
    sample_privacy_unit: str | None = None,
    population_privacy_unit: str | None = None,
    target_k_map: int | None = None,
    max_delta_presence: float | None = None,
) -> PopulationRiskAssessment:
    """Assess exact sample risk against a caller-supplied reference population.

    This is an offline exact-frequency model, not a statistical population
    estimator. The caller and qualified expert must establish that the
    reference data represents the anticipated attack population, contains the
    sample units, uses compatible privacy-unit semantics, and encodes every
    declared quasi-identifier consistently. No universal threshold is selected
    by OpenMed.

    A keyed analysis unit is represented by the sorted multiset of all its
    joint row-level QI tuples. Consequently, row-wise correlations and
    repeated-event multiplicity are retained. With no privacy-unit columns,
    every row is one analysis unit.

    Args:
        sample: Non-empty structured sample rows.
        reference_population: Non-empty structured reference-population rows.
        quasi_identifiers: Explicit QI columns present in every row.
        sample_privacy_unit: Optional sample column grouping longitudinal rows.
        population_privacy_unit: Optional reference column grouping rows.
        target_k_map: Explicit minimum accepted reference frequency for every
            sample analysis-unit profile. OpenMed selects no default threshold.
        max_delta_presence: Explicit maximum accepted ``sample_frequency /
            reference_frequency`` ratio in ``[0, 1]``. OpenMed selects no
            default threshold.

    Returns:
        A strict aggregate-only assessment containing counts, risks, policy
        verdicts, and canonical binding digests.

    Raises:
        TypeError: If rows, columns, or scalar values use unsupported types.
        ValueError: If schemas, privacy units, thresholds, or model inputs are
            incomplete or incompatible.
    """

    qis = _column_tuple(quasi_identifiers, name="quasi_identifiers")
    sample_unit = _optional_column(
        sample_privacy_unit,
        name="sample_privacy_unit",
    )
    population_unit = _optional_column(
        population_privacy_unit,
        name="population_privacy_unit",
    )
    if (sample_unit is None) != (population_unit is None):
        raise ValueError(
            "sample and reference populations must use compatible row-level "
            "or keyed privacy-unit semantics"
        )
    if sample_unit in qis or population_unit in qis:
        raise ValueError("privacy-unit columns cannot also be quasi-identifiers")
    if type(target_k_map) is not int or target_k_map < 1:
        raise ValueError("target_k_map must be explicitly set to an integer >= 1")
    if (
        not isinstance(max_delta_presence, (int, float))
        or isinstance(max_delta_presence, bool)
        or not math.isfinite(float(max_delta_presence))
        or not 0 <= float(max_delta_presence) <= 1
    ):
        raise ValueError(
            "max_delta_presence must be explicitly set to a finite number "
            "between 0 and 1"
        )

    sample_rows = _materialize_rows(sample)
    reference_rows = _materialize_rows(reference_population)
    if not sample_rows:
        raise ValueError("sample must contain at least one row")
    if not reference_rows:
        raise ValueError("reference_population must contain at least one row")

    sample_types = _validate_rows(
        sample_rows,
        qis,
        privacy_unit=sample_unit,
        dataset_name="sample",
    )
    reference_types = _validate_rows(
        reference_rows,
        qis,
        privacy_unit=population_unit,
        dataset_name="reference_population",
    )
    incompatible_types = {
        field: sorted(sample_types[field] - reference_types[field])
        for field in qis
        if not sample_types[field].issubset(reference_types[field])
    }
    if incompatible_types:
        raise ValueError(
            "reference_population does not model every sample quasi-identifier "
            f"scalar type: {incompatible_types!r}"
        )

    sample_profiles = _analysis_unit_profiles(
        sample_rows,
        qis,
        privacy_unit=sample_unit,
    )
    reference_profiles = _analysis_unit_profiles(
        reference_rows,
        qis,
        privacy_unit=population_unit,
    )
    sample_frequencies = Counter(sample_profiles.values())
    reference_frequencies = Counter(reference_profiles.values())

    matched_units = 0
    unmatched_units = 0
    singleton_units = 0
    frequency_violation_profiles = 0
    frequency_violation_units = 0
    exact_linkage_risks: list[float] = []
    delta_presence_ratios: list[float] = []
    matched_reference_frequencies: list[int] = []

    for profile, sample_frequency in sample_frequencies.items():
        reference_frequency = reference_frequencies.get(profile, 0)
        if reference_frequency:
            matched_units += sample_frequency
            matched_reference_frequencies.append(reference_frequency)
            exact_risk = 1.0 / reference_frequency
            delta_presence = sample_frequency / reference_frequency
        else:
            unmatched_units += sample_frequency
            exact_risk = 1.0
            delta_presence = 1.0
        if reference_frequency == 1:
            singleton_units += sample_frequency
        if sample_frequency > reference_frequency:
            frequency_violation_profiles += 1
            frequency_violation_units += sample_frequency
        exact_linkage_risks.extend([exact_risk] * sample_frequency)
        delta_presence_ratios.extend([delta_presence] * sample_frequency)

    achieved_k_map = min(matched_reference_frequencies) if unmatched_units == 0 else 0
    policy = {
        "kind": "openmed-exact-reference-population-policy",
        "model": _MODEL,
        "profile_model": _PROFILE_MODEL,
        "model_assumptions": list(_MODEL_ASSUMPTIONS),
        "quasi_identifiers": list(qis),
        "sample_privacy_unit": sample_unit,
        "population_privacy_unit": population_unit,
        "target_k_map": target_k_map,
        "max_delta_presence": float(max_delta_presence),
    }
    schema_binding = {
        "kind": "openmed-reference-population-schema-binding",
        "profile_model": _PROFILE_MODEL,
        "analysis_unit_model": "keyed" if sample_unit is not None else "row",
        "quasi_identifiers": list(qis),
        "sample_privacy_unit": sample_unit,
        "population_privacy_unit": population_unit,
        "sample_schema": _schema_digest(sample_rows),
        "reference_population_schema": _schema_digest(reference_rows),
    }
    schema_digest = stable_hash(schema_binding)
    return PopulationRiskAssessment(
        schema_version=_SCHEMA_VERSION,
        model=_MODEL,
        profile_model=_PROFILE_MODEL,
        model_assumptions=_MODEL_ASSUMPTIONS,
        analysis_unit_model="keyed" if sample_unit is not None else "row",
        quasi_identifier_count=len(qis),
        sample_row_count=len(sample_rows),
        reference_row_count=len(reference_rows),
        sample_unit_count=len(sample_profiles),
        reference_unit_count=len(reference_profiles),
        sample_profile_count=len(sample_frequencies),
        reference_profile_count=len(reference_frequencies),
        matched_sample_unit_count=matched_units,
        unmatched_sample_unit_count=unmatched_units,
        population_singleton_count=singleton_units,
        population_singleton_rate=singleton_units / len(sample_profiles),
        reference_frequency_violation_count=frequency_violation_profiles,
        reference_frequency_violation_unit_count=frequency_violation_units,
        achieved_k_map=achieved_k_map,
        target_k_map=target_k_map,
        max_exact_linkage_risk=max(exact_linkage_risks),
        mean_exact_linkage_risk=mean(exact_linkage_risks),
        max_delta_presence=max(delta_presence_ratios),
        mean_delta_presence=mean(delta_presence_ratios),
        max_delta_presence_threshold=float(max_delta_presence),
        sample_digest=_canonical_dataset_digest(sample_rows, kind="sample"),
        reference_population_digest=_canonical_dataset_digest(
            reference_rows,
            kind="reference_population",
        ),
        schema_digest=schema_digest,
        policy_digest=stable_hash(policy),
    )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
    *,
    privacy_unit: str | None,
    dataset_name: str,
) -> dict[str, set[str]]:
    qi_types: dict[str, set[str]] = {field: set() for field in quasi_identifiers}
    for row_index, row in enumerate(rows):
        if not row:
            raise ValueError(
                f"{dataset_name} contains an empty row at offset {row_index}"
            )
        for field, value in row.items():
            try:
                _canonical_digest_scalar(value)
            except TypeError:
                raise TypeError(
                    f"{dataset_name} column {field!r} contains an unsupported "
                    f"value at row offset {row_index}"
                ) from None
            except ValueError:
                raise ValueError(
                    f"{dataset_name} column {field!r} contains a non-canonical "
                    f"value at row offset {row_index}"
                ) from None
        for field in quasi_identifiers:
            if field not in row:
                raise ValueError(
                    f"{dataset_name} quasi-identifier {field!r} is missing at "
                    f"row offset {row_index}"
                )
            qi_types[field].add(_canonical_digest_scalar(row[field])["type"])
        if privacy_unit is not None:
            if privacy_unit not in row:
                raise ValueError(
                    f"{dataset_name} privacy unit {privacy_unit!r} is missing at "
                    f"row offset {row_index}"
                )
            value = row[privacy_unit]
            if value is None or (type(value) in {str, bytes} and not value.strip()):
                raise ValueError(
                    f"{dataset_name} privacy unit {privacy_unit!r} must be "
                    f"non-empty at row offset {row_index}"
                )
            if type(value) in {str, bytes} and value != value.strip():
                raise ValueError(
                    f"{dataset_name} privacy unit {privacy_unit!r} has surrounding "
                    f"whitespace at row offset {row_index}; canonicalize identifiers "
                    "before population-risk measurement"
                )
    return qi_types


def _analysis_unit_profiles(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
    *,
    privacy_unit: str | None,
) -> dict[tuple[str, Any], tuple[tuple[str, ...], ...]]:
    grouped: defaultdict[tuple[str, Any], list[tuple[str, ...]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        unit_key: tuple[str, Any]
        if privacy_unit is None:
            unit_key = ("row", row_index)
        else:
            unit_key = ("keyed", _typed_scalar_token(row[privacy_unit]))
        joint_row_profile = tuple(
            _typed_scalar_token(row[field]) for field in quasi_identifiers
        )
        grouped[unit_key].append(joint_row_profile)
    return {unit: tuple(sorted(joint_rows)) for unit, joint_rows in grouped.items()}


def _typed_scalar_token(value: Any) -> str:
    return json.dumps(
        _canonical_digest_scalar(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_dataset_digest(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> str:
    canonical_rows = []
    for row in rows:
        canonical_rows.append(
            json.dumps(
                [
                    [field, _canonical_digest_scalar(row[field])]
                    for field in sorted(row)
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return stable_hash(
        {
            "kind": f"openmed-population-risk-{kind}",
            "row_count": len(rows),
            "rows": sorted(canonical_rows),
        }
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")
