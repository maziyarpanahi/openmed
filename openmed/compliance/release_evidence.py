"""Bridge safe anonymization results into expert-review evidence bundles."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

from openmed.__about__ import __version__
from openmed.core.audit import stable_hash

from .expert_review import (
    AggregateRiskMetrics,
    AttributeRoleReview,
    ClassSizeBin,
    CompositionEvidence,
    EvidenceDigests,
    ExpertReviewEvidenceReport,
    PrivacyModelEvidence,
    ReleaseAssumptions,
    SearchEvidence,
    SuppressionAggregate,
    TransformationAggregate,
    UtilityAggregate,
    build_expert_review_evidence,
)

if TYPE_CHECKING:
    from openmed.risk.release import (
        AnonymizationResult,
        ReleaseAssessment,
        ReleasedOutputValidation,
    )

__all__ = ["build_release_expert_review_evidence"]

_MANDATORY_RELEASE_LIMITATIONS: Final = (
    "not_compliance_certificate",
    "population_risk_not_estimated",
    "qualified_expert_review_required",
)
_MANDATORY_UNSUPPORTED_MODALITIES: Final = (
    "free_text",
    "images",
    "genomic_data",
)
_SOFTWARE_EVIDENCE_MODULES: Final = (
    "openmed.compliance.expert_review",
    "openmed.compliance.release_evidence",
    "openmed.core.audit",
    "openmed.risk.kanon",
    "openmed.risk.reid",
    "openmed.risk.release",
    "openmed.structured.table_io",
)


def build_release_expert_review_evidence(
    result: AnonymizationResult,
    *,
    validation: ReleasedOutputValidation,
    assumptions: ReleaseAssumptions,
    composition: CompositionEvidence | None = None,
    limitations: tuple[str, ...] = _MANDATORY_RELEASE_LIMITATIONS,
    unsupported_modalities: tuple[str, ...] = _MANDATORY_UNSUPPORTED_MODALITIES,
) -> ExpertReviewEvidenceReport:
    """Build a strict aggregate evidence bundle from an anonymization result.

    The transformed rows are deliberately inaccessible to the lower-level
    evidence builder. Only allow-listed aggregate fields from
    ``AnonymizationResult`` cross this boundary.

    Caller-supplied limitation and unsupported-modality codes extend the
    mandatory baseline caveats; they cannot remove them.
    """

    from openmed.risk.release import AnonymizationResult, ReleasedOutputValidation

    if not isinstance(result, AnonymizationResult):
        raise TypeError("result must be an AnonymizationResult")
    if not isinstance(validation, ReleasedOutputValidation):
        raise TypeError("validation must be a ReleasedOutputValidation")
    if not isinstance(assumptions, ReleaseAssumptions):
        raise TypeError("assumptions must be a ReleaseAssumptions")
    _require_validated_release_binding(result, validation)
    resolved_limitations = _with_mandatory_codes(
        limitations,
        mandatory=_MANDATORY_RELEASE_LIMITATIONS,
        name="limitations",
    )
    resolved_unsupported_modalities = _with_mandatory_codes(
        unsupported_modalities,
        mandatory=_MANDATORY_UNSUPPORTED_MODALITIES,
        name="unsupported_modalities",
    )

    policy = result.policy
    if policy.privacy_unit is None and assumptions.privacy_unit != "row":
        raise ValueError(
            "row-level release policies require assumptions.privacy_unit='row'"
        )
    if policy.privacy_unit is not None and assumptions.privacy_unit == "row":
        raise ValueError(
            "keyed privacy-unit release policies cannot use "
            "assumptions.privacy_unit='row'"
        )
    before = result.before
    after = result.after
    attribute_reviews = _attribute_reviews(result)
    privacy_models = _privacy_models(result)
    pre_metrics = _aggregate_metrics(before)
    post_metrics = _aggregate_metrics(after)
    affected_privacy_units = dict(result.generalization.affected_privacy_units)
    affected_qi_cells = dict(result.generalization.affected_qi_cells)
    suppressed_qi_cells = dict(result.generalization.suppressed_qi_cells)
    transformations = tuple(
        TransformationAggregate(
            attribute=field,
            method=(
                "suppress"
                if affected_qi_cells[field] > 0
                and suppressed_qi_cells[field] == affected_qi_cells[field]
                else "generalize"
            ),
            affected_privacy_unit_count=affected_privacy_units[field],
            hierarchy_level_before=0,
            hierarchy_level_after=level,
        )
        for field, level, _name, _loss in result.generalization.levels
        if level > 0
    )
    suppression_limit = _suppression_limit(result)
    suppression = SuppressionAggregate(
        privacy_units_suppressed=result.generalization.suppressed_privacy_units,
        rows_suppressed=result.generalization.suppressed_rows,
        cells_suppressed=sum(suppressed_qi_cells.values()),
        suppression_rate=result.utility.privacy_unit_suppression_rate,
        privacy_unit_limit=suppression_limit,
    )
    utility = (
        UtilityAggregate(
            metric="row_retention",
            before=1.0,
            after=_retention(
                result.utility.released_rows,
                result.utility.source_rows,
            ),
            unit="ratio",
            higher_is_better=True,
        ),
        UtilityAggregate(
            metric="privacy_unit_retention",
            before=1.0,
            after=_retention(
                result.utility.released_privacy_units,
                result.utility.source_privacy_units,
            ),
            unit="ratio",
            higher_is_better=True,
        ),
        UtilityAggregate(
            metric="quasi_identifier_cell_retention",
            before=1.0,
            after=1.0 - result.utility.quasi_identifier_cell_change_rate,
            unit="ratio",
            higher_is_better=True,
        ),
        UtilityAggregate(
            metric="mean_qi_distribution_shift",
            before=0.0,
            after=result.utility.mean_qi_distribution_shift,
            unit="ratio",
            higher_is_better=False,
        ),
        UtilityAggregate(
            metric="information_loss",
            before=0.0,
            after=result.generalization.information_loss,
            unit="score",
            higher_is_better=False,
        ),
        UtilityAggregate(
            metric="direct_identifier_cells_remaining",
            before=result.utility.direct_identifier_cells_removed,
            after=0,
            unit="count",
            higher_is_better=False,
        ),
    )
    search = SearchEvidence(
        strategy=(
            "exhaustive_lattice"
            if result.generalization.search_complete
            else "bounded_lattice"
        ),
        complete=result.generalization.search_complete,
        optimality_proven=result.generalization.optimum_proven,
        evaluated_candidates=result.generalization.nodes_evaluated,
        total_candidates=result.generalization.search_space_size,
        maximum_quasi_identifiers=max(1, len(policy.quasi_identifiers)),
        candidate_limit=result.generalization.max_lattice_nodes,
        suppression_subsets_evaluated=(
            result.generalization.suppression_subsets_evaluated
        ),
        suppression_subsets_total=(result.generalization.suppression_subsets_possible),
        suppression_subset_limit=result.generalization.max_suppression_subsets,
        time_limit_seconds=None,
        termination_reason=(
            "optimal_candidate_found"
            if result.generalization.optimum_proven
            else "candidate_limit_reached"
        ),
    )
    resolved_composition = composition or CompositionEvidence(
        release_count=1,
        longitudinal_linkage_assessed=False,
        prior_release_overlap_assessed=False,
        risk_status="not_assessed",
        evidence_digest=stable_hash(
            {
                "kind": "openmed-release-composition",
                "status": "not_assessed",
                "release_count": 1,
            }
        ),
    )
    digests = EvidenceDigests(
        source_dataset=result.source_dataset_digest,
        dataset=validation.dataset_digest,
        schema=validation.schema_digest,
        policy=policy.digest,
        hierarchy=result.hierarchy_digest,
        config=stable_hash(
            {
                "kind": "openmed-release-config",
                "policy": policy.to_dict(),
                "search": result.generalization.to_dict()["search"],
            }
        ),
        software=_software_digest(),
    )
    return build_expert_review_evidence(
        digests=digests,
        assumptions=assumptions,
        attribute_reviews=attribute_reviews,
        selected_quasi_identifiers=policy.quasi_identifiers,
        sensitive_attributes=policy.sensitive_attributes,
        privacy_models=privacy_models,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        transformations=transformations,
        suppression=suppression,
        utility=utility,
        search=search,
        composition=resolved_composition,
        limitations=resolved_limitations,
        unsupported_modalities=resolved_unsupported_modalities,
    )


def _with_mandatory_codes(
    supplied: tuple[str, ...],
    *,
    mandatory: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    """Add non-removable baseline caveats while preserving strict extra codes."""

    if not isinstance(supplied, tuple):
        raise TypeError(f"{name} must be a tuple of coded values")
    extras = tuple(value for value in supplied if value not in mandatory)
    return (*mandatory, *extras)


def _attribute_reviews(
    result: AnonymizationResult,
) -> tuple[AttributeRoleReview, ...]:
    policy = result.policy
    attributes = sorted(
        {
            *policy.quasi_identifiers,
            *policy.sensitive_attributes,
            *policy.direct_identifiers,
            *policy.non_sensitive_attributes,
            *policy.excluded_attributes,
            *([policy.privacy_unit] if policy.privacy_unit is not None else []),
        }
    )
    reviews = []
    for attribute in attributes:
        roles = []
        if attribute == policy.privacy_unit:
            roles.append("direct_identifier")
            roles.append("privacy_unit")
        if attribute in policy.direct_identifiers and "direct_identifier" not in roles:
            roles.append("direct_identifier")
        if attribute in policy.quasi_identifiers:
            roles.append("quasi_identifier")
        if attribute in policy.sensitive_attributes:
            roles.append("sensitive_attribute")
        if attribute in policy.non_sensitive_attributes:
            roles.append("non_sensitive")
        if attribute in policy.excluded_attributes:
            roles.append("excluded")
        reviews.append(
            AttributeRoleReview(
                attribute=attribute,
                roles=tuple(roles),
                override_applied=False,
            )
        )
    return tuple(reviews)


def _privacy_models(result: AnonymizationResult) -> PrivacyModelEvidence:
    policy = result.policy
    before_l = _worst_l(result.before)
    after_l = _worst_l(result.after)
    before_t = _worst_t(result.before)
    after_t = _worst_t(result.after)
    configured_l: int | float | None
    if not policy.sensitive_attributes:
        configured_l = None
    elif policy.l_metric == "entropy":
        import math

        configured_l = math.log2(policy.target_l)
    else:
        configured_l = policy.target_l
        before_l = int(before_l) if before_l is not None else None
        after_l = int(after_l) if after_l is not None else None
    return PrivacyModelEvidence(
        configured_k=policy.target_k,
        pre_achieved_k=result.before.achieved_k,
        achieved_k=result.after.achieved_k,
        l_variant=policy.l_metric if policy.sensitive_attributes else None,
        configured_l=configured_l,
        pre_achieved_l=before_l,
        achieved_l=after_l,
        t_variant=policy.t_distance if policy.sensitive_attributes else None,
        configured_t=policy.target_t if policy.sensitive_attributes else None,
        pre_achieved_t=before_t,
        achieved_t=after_t,
    )


def _aggregate_metrics(
    assessment: ReleaseAssessment,
) -> AggregateRiskMetrics:
    bins = tuple(
        ClassSizeBin(
            lower_bound=size,
            upper_bound=size,
            class_count=class_count,
            privacy_unit_count=size * class_count,
        )
        for size, class_count in assessment.class_size_distribution
    )
    sizes = [
        size
        for size, class_count in assessment.class_size_distribution
        for _ in range(class_count)
    ]
    return AggregateRiskMetrics(
        privacy_unit_count=assessment.privacy_unit_count,
        equivalence_class_count=assessment.class_count,
        smallest_class_size=assessment.achieved_k,
        largest_class_size=max(sizes, default=0),
        mean_class_size=mean(sizes) if sizes else 0.0,
        class_size_histogram=bins,
        k_violating_class_count=assessment.k_violating_class_count,
        l_violating_class_count=assessment.l_violating_class_count,
        t_violating_class_count=assessment.t_violating_class_count,
        any_violating_class_count=assessment.violating_class_count,
        violating_privacy_unit_count=assessment.violating_privacy_unit_count,
    )


def _suppression_limit(result: AnonymizationResult) -> int:
    policy = result.policy
    limits = []
    if policy.suppression_limit is not None:
        limits.append(policy.suppression_limit)
    if policy.suppression_rate > 0:
        import math

        limits.append(
            math.floor(result.before.privacy_unit_count * policy.suppression_rate)
        )
    if not limits:
        return result.generalization.suppressed_privacy_units
    return min(limits)


def _worst_l(assessment: ReleaseAssessment) -> int | float | None:
    if not assessment.attributes:
        return None
    return min(item.achieved_l for item in assessment.attributes)


def _worst_t(assessment: ReleaseAssessment) -> float | None:
    if not assessment.attributes:
        return None
    return max(item.achieved_t for item in assessment.attributes)


def _retention(after: int, before: int) -> float:
    return after / before if before else 0.0


def _require_validated_release_binding(
    result: AnonymizationResult,
    validation: ReleasedOutputValidation,
) -> None:
    if not validation.passed:
        raise ValueError(
            "materialized release validation must pass before evidence is built"
        )
    if validation.expected_row_count != len(result.records):
        raise ValueError(
            "materialized release validation row count does not match the result"
        )
    if (
        validation.policy_revalidated_before_identifier_removal
        is not result.after.meets_policy
    ):
        raise ValueError(
            "materialized release validation policy state does not match the result"
        )
    if type(validation.typed_digest_comparison_available) is not bool:
        raise TypeError(
            "validation.typed_digest_comparison_available must be a boolean"
        )

    expected_rows: Sequence[Mapping[str, Any]]
    if validation.typed_digest_comparison_available:
        expected_rows = result.records
    else:
        expected_rows = _stringified_scalar_rows(result.records)

    from openmed.risk.release import (
        release_dataset_digest,
        release_schema_digest,
    )

    expected_dataset_digest = release_dataset_digest(expected_rows)
    expected_schema_digest = release_schema_digest(expected_rows)
    if validation.expected_digest != expected_dataset_digest:
        raise ValueError(
            "materialized release validation dataset binding does not match the result"
        )
    if validation.expected_schema_digest != expected_schema_digest:
        raise ValueError(
            "materialized release validation schema binding does not match the result"
        )
    if not validation.schema_matches:
        raise ValueError("materialized release schema validation must pass")


def _stringified_scalar_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, str], ...]:
    fields = tuple(dict.fromkeys(str(field) for row in rows for field in row))
    converted_rows: list[Mapping[str, str]] = []
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
                    "Delimited release evidence supports scalar values only"
                )
        converted_rows.append(converted)
    return tuple(converted_rows)


def _software_digest() -> str:
    module_digests: dict[str, str] = {}
    for module_name in _SOFTWARE_EVIDENCE_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(
                f"Cannot bind software evidence for module {module_name!r}"
            )
        module_bytes = Path(spec.origin).read_bytes()
        module_digests[module_name] = (
            "sha256:" + hashlib.sha256(module_bytes).hexdigest()
        )
    return stable_hash(
        {
            "kind": "openmed-software-content",
            "package": {
                "name": "openmed",
                "version": __version__,
            },
            "runtime": _runtime_metadata(),
            "modules": module_digests,
        }
    )


def _runtime_metadata() -> dict[str, str]:
    """Return deterministic, non-sensitive runtime provenance."""

    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag or "unknown",
    }
