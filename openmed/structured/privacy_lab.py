"""Evidence-oriented structured privacy and re-identification risk lab.

The lab composes the existing release-policy implementation with an
aggregate-only schema profile, a bounded membership self-test, and a
PHI-safe evidence report. It requires callers to declare the quasi-identifiers,
sensitive attributes, thresholds, and suppression policy; it never silently
chooses a release threshold or presents one score as proof of anonymity.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from openmed.core.audit import stable_hash
from openmed.risk.aggregate_dp import AggregateDPBudgetLedger
from openmed.risk.membership import (
    MembershipSelfTestError,
    MembershipSelfTestResult,
    membership_inference_self_test,
)
from openmed.risk.release import (
    AnonymityPolicy,
    AnonymizationResult,
    ReleaseAssessment,
    anonymize_release,
    assess_release,
    release_dataset_digest,
    release_schema_digest,
)

__all__ = [
    "ColumnPrivacyProfile",
    "PopulationAssumptions",
    "StructuredPrivacyEvidenceReport",
    "StructuredPrivacyLab",
    "StructuredPrivacyLabError",
    "StructuredPrivacyLabResult",
    "StructuredPrivacyPolicy",
    "StructuredPrivacyRiskLab",
    "StructuredTableProfile",
    "build_structured_privacy_evidence",
    "make_synthetic_privacy_fixture",
    "profile_structured_table",
    "profile_table",
    "run_privacy_lab",
    "run_structured_privacy_lab",
    "structured_privacy_fixture",
]


class StructuredPrivacyLabError(ValueError):
    """Raised when a structured privacy lab cannot run safely."""


@dataclass(frozen=True)
class PopulationAssumptions:
    """Coded, user-declared population context bound to a lab run."""

    scope: str = "release_cohort"
    unit: str = "row"
    population_kind: str = "declared"
    details_digest: str | None = None
    declared: bool = True

    def __post_init__(self) -> None:
        for name in ("scope", "unit", "population_kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError(f"{name} must be a non-empty short string")
        if self.details_digest is not None and not _is_digest(self.details_digest):
            raise ValueError("details_digest must be a sha256 digest")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        unit: str,
    ) -> PopulationAssumptions:
        """Build coded assumptions while hashing unshareable free-form detail."""

        if value is None:
            return cls(unit=unit, declared=False)
        if not isinstance(value, Mapping):
            raise TypeError("population_assumptions must be a mapping")
        scope = value.get("scope", value.get("population_scope", "declared"))
        population_kind = value.get("population_kind", value.get("kind", "declared"))
        if not isinstance(scope, str) or not isinstance(population_kind, str):
            raise ValueError("population assumptions use coded string fields")
        return cls(
            scope=scope,
            unit=unit,
            population_kind=population_kind,
            details_digest=stable_hash(
                {
                    "artifact": "openmed-structured-population-assumptions",
                    "values": _canonical_value(value),
                }
            ),
            declared=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return coded context without free-form assumptions."""

        return {
            "scope": self.scope,
            "unit": self.unit,
            "population_kind": self.population_kind,
            "declared": self.declared,
            "details_digest": self.details_digest,
        }


@dataclass(frozen=True)
class StructuredPrivacyPolicy:
    """Explicit policy for one structured privacy lab run."""

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
    suppression_limit: int | None = None
    suppression_rate: float = 0.0
    membership_max_inference_rate: float | None = None
    membership_max_candidates: int = 10_000
    max_lattice_nodes: int = 100_000
    max_suppression_subsets: int = 100_000

    def __post_init__(self) -> None:
        qis = _columns(self.quasi_identifiers, name="quasi_identifiers")
        sensitive = _columns(
            self.sensitive_attributes,
            name="sensitive_attributes",
            allow_empty=True,
        )
        direct = _columns(
            self.direct_identifiers,
            name="direct_identifiers",
            allow_empty=True,
        )
        non_sensitive = _columns(
            self.non_sensitive_attributes,
            name="non_sensitive_attributes",
            allow_empty=True,
        )
        excluded = _columns(
            self.excluded_attributes,
            name="excluded_attributes",
            allow_empty=True,
        )
        _validate_policy_numbers(self)
        # Reuse the release engine's overlap and k/l/t validation so the lab
        # cannot drift from the established patient-level semantics.
        AnonymityPolicy(
            quasi_identifiers=qis,
            target_k=self.target_k,
            sensitive_attributes=sensitive,
            direct_identifiers=direct,
            non_sensitive_attributes=non_sensitive,
            excluded_attributes=excluded,
            privacy_unit=self.privacy_unit,
            target_l=self.target_l,
            l_metric=self.l_metric,
            target_t=self.target_t,
            suppression_limit=self.suppression_limit,
            suppression_rate=self.suppression_rate,
            max_lattice_nodes=self.max_lattice_nodes,
            max_suppression_subsets=self.max_suppression_subsets,
        )
        object.__setattr__(self, "quasi_identifiers", qis)
        object.__setattr__(self, "sensitive_attributes", sensitive)
        object.__setattr__(self, "direct_identifiers", direct)
        object.__setattr__(self, "non_sensitive_attributes", non_sensitive)
        object.__setattr__(self, "excluded_attributes", excluded)
        object.__setattr__(self, "target_t", float(self.target_t))
        object.__setattr__(self, "suppression_rate", float(self.suppression_rate))
        object.__setattr__(
            self,
            "membership_max_inference_rate",
            _optional_rate(self.membership_max_inference_rate),
        )

    def to_anonymity_policy(self) -> AnonymityPolicy:
        """Return the compatible patient-level release policy."""

        return AnonymityPolicy(
            quasi_identifiers=self.quasi_identifiers,
            target_k=self.target_k,
            sensitive_attributes=self.sensitive_attributes,
            direct_identifiers=self.direct_identifiers,
            non_sensitive_attributes=self.non_sensitive_attributes,
            excluded_attributes=self.excluded_attributes,
            privacy_unit=self.privacy_unit,
            target_l=self.target_l,
            l_metric=self.l_metric,
            target_t=self.target_t,
            suppression_limit=self.suppression_limit,
            suppression_rate=self.suppression_rate,
            max_lattice_nodes=self.max_lattice_nodes,
            max_suppression_subsets=self.max_suppression_subsets,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic release and self-test parameters."""

        return {
            **self.to_anonymity_policy().to_dict(),
            "membership_max_inference_rate": self.membership_max_inference_rate,
            "membership_max_candidates": self.membership_max_candidates,
            "suppression_policy": {
                "mode": "whole_privacy_unit",
                "max_units": self.suppression_limit,
                "max_rate": self.suppression_rate,
            },
        }


@dataclass(frozen=True)
class ColumnPrivacyProfile:
    """Aggregate schema statistics for one explicitly classified column."""

    column: str
    role: str
    row_count: int
    missing_count: int
    non_null_count: int
    unique_count: int
    uniqueness_ratio: float
    rare_value_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-safe column statistics."""

        return {
            "column": self.column,
            "role": self.role,
            "row_count": self.row_count,
            "missing_count": self.missing_count,
            "non_null_count": self.non_null_count,
            "unique_count": self.unique_count,
            "uniqueness_ratio": self.uniqueness_ratio,
            "rare_value_count": self.rare_value_count,
        }


@dataclass(frozen=True)
class StructuredTableProfile:
    """Aggregate profile of a structured table and its declared QI set."""

    schema_version: int
    row_count: int
    columns: tuple[ColumnPrivacyProfile, ...]
    quasi_identifiers: tuple[str, ...]
    sensitive_attributes: tuple[str, ...]
    direct_identifiers: tuple[str, ...]
    missing_cell_count: int
    missing_quasi_identifier_row_count: int
    unique_quasi_identifier_combination_count: int
    rare_quasi_identifier_combination_count: int
    rare_combination_threshold: int
    population_assumptions: PopulationAssumptions
    dataset_digest: str
    schema_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe schema and risk profile without cell values."""

        return {
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "columns": [column.to_dict() for column in self.columns],
            "quasi_identifiers": list(self.quasi_identifiers),
            "sensitive_attributes": list(self.sensitive_attributes),
            "direct_identifiers": list(self.direct_identifiers),
            "missingness": {
                "missing_cell_count": self.missing_cell_count,
                "missing_quasi_identifier_row_count": (
                    self.missing_quasi_identifier_row_count
                ),
            },
            "uniqueness": {
                "unique_quasi_identifier_combination_count": (
                    self.unique_quasi_identifier_combination_count
                ),
                "rare_quasi_identifier_combination_count": (
                    self.rare_quasi_identifier_combination_count
                ),
                "rare_combination_threshold": self.rare_combination_threshold,
            },
            "population_assumptions": self.population_assumptions.to_dict(),
            "dataset_digest": self.dataset_digest,
            "schema_digest": self.schema_digest,
        }


@dataclass(frozen=True)
class StructuredPrivacyLabResult:
    """Result with sensitive rows kept local and evidence kept aggregate-only."""

    policy: StructuredPrivacyPolicy
    profile: StructuredTableProfile
    before: ReleaseAssessment
    after: ReleaseAssessment | None
    anonymization: AnonymizationResult | None = field(repr=False, default=None)
    membership_before: MembershipSelfTestResult | None = None
    membership_after: MembershipSelfTestResult | None = None
    transformation_status: str = "not_run"
    evidence: StructuredPrivacyEvidenceReport | None = field(default=None, repr=False)

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        """Return transformed rows for local publication after review."""

        if self.anonymization is None:
            return ()
        return self.anonymization.records

    @property
    def meets_privacy_policy(self) -> bool:
        """Return the conjunction of release and configured attack gates."""

        release_passes = self.after is not None and self.after.meets_policy
        membership_passes = self.policy.membership_max_inference_rate is None or (
            self.membership_after is not None and self.membership_after.meets_policy
        )
        return bool(release_passes and membership_passes)

    @property
    def meets_policy(self) -> bool:
        """Alias for :attr:`meets_privacy_policy`."""

        return self.meets_privacy_policy

    def to_dict(self) -> dict[str, Any]:
        """Return only aggregate evidence and safe transformation status."""

        return {
            "schema_version": 1,
            "artifact": "structured_privacy_lab_result",
            "policy": self.policy.to_dict(),
            "profile": self.profile.to_dict(),
            "before": self.before.to_dict(),
            "after": self.after.to_dict() if self.after is not None else None,
            "transformation": {
                "status": self.transformation_status,
                "available_for_local_review": self.anonymization is not None,
            },
            "membership": {
                "before": (
                    self.membership_before.to_dict()
                    if self.membership_before is not None
                    else {"status": "not_run"}
                ),
                "after": (
                    self.membership_after.to_dict()
                    if self.membership_after is not None
                    else {"status": "not_run"}
                ),
            },
            "risk_delta": _risk_delta(self.before, self.after),
            "utility": (
                self.anonymization.utility.to_dict()
                if self.anonymization is not None
                else {"status": "unavailable"}
            ),
            "meets_policy": self.meets_privacy_policy,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


@dataclass(frozen=True)
class StructuredPrivacyEvidenceReport:
    """SP 800-188-oriented aggregate method/evidence report."""

    result: StructuredPrivacyLabResult = field(repr=False)
    population_assumptions: PopulationAssumptions
    dp_ledger: AggregateDPBudgetLedger | None = field(default=None, repr=False)
    limitations: tuple[str, ...] = (
        "Metrics support qualified review and do not certify anonymity.",
        "k-anonymity, l-diversity, and t-closeness do not model every auxiliary source.",
        "The membership result is a bounded self-test, not a universal attack bound.",
        "Differential privacy in this workflow applies to aggregate queries only.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic evidence with an integrity digest."""

        payload = {
            "schema_version": 1,
            "artifact": "structured_privacy_method_evidence",
            "method": {
                "standard_orientation": "NIST SP 800-188",
                "workflow": "structured_privacy_risk_lab",
                "offline": True,
                "qualified_expert_review_required": True,
            },
            "parameters": self.result.policy.to_dict(),
            "population_assumptions": self.population_assumptions.to_dict(),
            "profile": self.result.profile.to_dict(),
            "source": {
                "dataset_digest": self.result.profile.dataset_digest,
                "schema_digest": self.result.profile.schema_digest,
            },
            "before": self.result.before.to_dict(),
            "after": (
                self.result.after.to_dict() if self.result.after is not None else None
            ),
            "transformation": {
                "status": self.result.transformation_status,
                "generalization": (
                    self.result.anonymization.generalization.to_dict()
                    if self.result.anonymization is not None
                    else None
                ),
            },
            "utility": (
                self.result.anonymization.utility.to_dict()
                if self.result.anonymization is not None
                else {"status": "unavailable"}
            ),
            "risk_delta": _risk_delta(self.result.before, self.result.after),
            "attacks": {
                "membership_inference": (
                    self.result.membership_after.to_dict()
                    if self.result.membership_after is not None
                    else {"status": "not_run"}
                ),
            },
            "differential_privacy": (
                self.dp_ledger.to_dict()
                if self.dp_ledger is not None
                else {"status": "not_configured", "scope": "aggregate_only"}
            ),
            "meets_policy": self.result.meets_privacy_policy,
            "limitations": list(self.limitations),
        }
        return {
            **payload,
            "integrity_digest": stable_hash(
                {"artifact": payload["artifact"], "payload": payload}
            ),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize aggregate evidence deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    def to_markdown(self) -> str:
        """Render a short aggregate-only review handoff."""

        payload = self.to_dict()
        return "\n".join(
            (
                "# Structured privacy method/evidence report",
                "",
                "> This report supports qualified review; it is not a certification "
                "or guarantee of anonymity.",
                "",
                f"- Meets configured policy: `{payload['meets_policy']}`",
                f"- Source dataset digest: `{payload['source']['dataset_digest']}`",
                f"- Source schema digest: `{payload['source']['schema_digest']}`",
                f"- Evidence integrity digest: `{payload['integrity_digest']}`",
                "- Release scope: aggregate evidence only; transformed rows remain "
                "separate.",
                "",
                "See the JSON artifact for the complete coded parameters, profile, "
                "risk measures, utility deltas, attack results, and limitations.",
                "",
            )
        )


class StructuredPrivacyLab:
    """Reusable runner for one explicit :class:`StructuredPrivacyPolicy`."""

    def __init__(self, policy: StructuredPrivacyPolicy) -> None:
        if not isinstance(policy, StructuredPrivacyPolicy):
            raise TypeError("policy must be a StructuredPrivacyPolicy")
        self.policy = policy

    def run(
        self,
        records: Any,
        *,
        population_assumptions: Mapping[str, Any] | None = None,
        membership_candidates: Any | None = None,
        hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        dp_ledger: AggregateDPBudgetLedger | None = None,
    ) -> StructuredPrivacyLabResult:
        """Run the configured lab with local-only optional evidence inputs."""

        return run_structured_privacy_lab(
            records,
            self.policy,
            population_assumptions=population_assumptions,
            membership_candidates=membership_candidates,
            hierarchies=hierarchies,
            dp_ledger=dp_ledger,
        )


StructuredPrivacyRiskLab = StructuredPrivacyLab


def profile_structured_table(
    records: Any,
    *,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    direct_identifiers: Sequence[str] = (),
    population_assumptions: Mapping[str, Any] | None = None,
    privacy_unit: str | None = None,
    rare_combination_threshold: int = 1,
) -> StructuredTableProfile:
    """Profile schema, missingness, uniqueness, rare combinations, and roles."""

    rows = _materialize_rows(records)
    qis = _columns(quasi_identifiers, name="quasi_identifiers")
    sensitive = _columns(
        sensitive_attributes,
        name="sensitive_attributes",
        allow_empty=True,
    )
    direct = _columns(
        direct_identifiers,
        name="direct_identifiers",
        allow_empty=True,
    )
    if type(rare_combination_threshold) is not int or rare_combination_threshold < 1:
        raise ValueError("rare_combination_threshold must be an integer >= 1")
    _validate_role_overlap(
        qis=qis,
        sensitive=sensitive,
        direct=direct,
    )
    fields = _field_order(rows)
    role_by_field = {
        field: (
            "quasi_identifier"
            if field in qis
            else "sensitive"
            if field in sensitive
            else "direct_identifier"
            if field in direct
            else "safe"
        )
        for field in fields
    }
    column_profiles: list[ColumnPrivacyProfile] = []
    missing_cell_count = 0
    for field in fields:
        values = [row.get(field) for row in rows]
        missing = sum(value is None for value in values)
        missing_cell_count += missing
        tokens = [_value_token(value) for value in values if value is not None]
        counts = Counter(tokens)
        non_null = len(tokens)
        column_profiles.append(
            ColumnPrivacyProfile(
                column=field,
                role=role_by_field[field],
                row_count=len(rows),
                missing_count=missing,
                non_null_count=non_null,
                unique_count=len(counts),
                uniqueness_ratio=_rate(len(counts), non_null),
                rare_value_count=sum(
                    count <= rare_combination_threshold for count in counts.values()
                ),
            )
        )
    combination_tokens = [
        tuple(_value_token(row.get(field)) for field in qis) for row in rows
    ]
    combination_counts = Counter(combination_tokens)
    missing_qi_rows = sum(any(row.get(field) is None for field in qis) for row in rows)
    assumptions = PopulationAssumptions.from_mapping(
        population_assumptions,
        unit=privacy_unit or "row",
    )
    return StructuredTableProfile(
        schema_version=1,
        row_count=len(rows),
        columns=tuple(column_profiles),
        quasi_identifiers=qis,
        sensitive_attributes=sensitive,
        direct_identifiers=direct,
        missing_cell_count=missing_cell_count,
        missing_quasi_identifier_row_count=missing_qi_rows,
        unique_quasi_identifier_combination_count=sum(
            count == 1 for count in combination_counts.values()
        ),
        rare_quasi_identifier_combination_count=sum(
            count <= rare_combination_threshold for count in combination_counts.values()
        ),
        rare_combination_threshold=rare_combination_threshold,
        population_assumptions=assumptions,
        dataset_digest=release_dataset_digest(rows),
        schema_digest=release_schema_digest(rows),
    )


def profile_table(
    records: Any,
    *,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    direct_identifiers: Sequence[str] = (),
    population_assumptions: Mapping[str, Any] | None = None,
    privacy_unit: str | None = None,
    rare_combination_threshold: int = 1,
) -> StructuredTableProfile:
    """Alias for :func:`profile_structured_table`."""

    return profile_structured_table(
        records,
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
        direct_identifiers=direct_identifiers,
        population_assumptions=population_assumptions,
        privacy_unit=privacy_unit,
        rare_combination_threshold=rare_combination_threshold,
    )


def run_structured_privacy_lab(
    records: Any,
    policy: StructuredPrivacyPolicy | None = None,
    *,
    quasi_identifiers: Sequence[str] | None = None,
    target_k: int | None = None,
    sensitive_attributes: Sequence[str] = (),
    direct_identifiers: Sequence[str] = (),
    non_sensitive_attributes: Sequence[str] = (),
    excluded_attributes: Sequence[str] = (),
    privacy_unit: str | None = None,
    target_l: int = 1,
    l_metric: str = "distinct",
    target_t: float = 1.0,
    suppression_limit: int | None = None,
    suppression_rate: float = 0.0,
    membership_max_inference_rate: float | None = None,
    membership_max_candidates: int = 10_000,
    max_lattice_nodes: int = 100_000,
    max_suppression_subsets: int = 100_000,
    population_assumptions: Mapping[str, Any] | None = None,
    membership_candidates: Any | None = None,
    candidate_population: Any | None = None,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    dp_ledger: AggregateDPBudgetLedger | None = None,
) -> StructuredPrivacyLabResult:
    """Run the offline structured privacy lab and build safe evidence.

    Pass either an explicit ``policy`` or the keyword policy arguments. The
    keyword form requires ``quasi_identifiers`` and ``target_k``; no role or
    threshold is inferred from the table.
    """

    if candidate_population is not None:
        if membership_candidates is not None:
            raise StructuredPrivacyLabError(
                "provide only one membership candidate population"
            )
        membership_candidates = candidate_population
    if policy is None:
        if quasi_identifiers is None or target_k is None:
            raise StructuredPrivacyLabError(
                "quasi_identifiers and target_k are required for an explicit lab policy"
            )
        policy = StructuredPrivacyPolicy(
            quasi_identifiers=quasi_identifiers,
            target_k=target_k,
            sensitive_attributes=sensitive_attributes,
            direct_identifiers=direct_identifiers,
            non_sensitive_attributes=non_sensitive_attributes,
            excluded_attributes=excluded_attributes,
            privacy_unit=privacy_unit,
            target_l=target_l,
            l_metric=l_metric,
            target_t=target_t,
            suppression_limit=suppression_limit,
            suppression_rate=suppression_rate,
            membership_max_inference_rate=membership_max_inference_rate,
            membership_max_candidates=membership_max_candidates,
            max_lattice_nodes=max_lattice_nodes,
            max_suppression_subsets=max_suppression_subsets,
        )
    elif not isinstance(policy, StructuredPrivacyPolicy):
        raise TypeError("policy must be a StructuredPrivacyPolicy")

    rows = _materialize_rows(records)
    if not rows:
        raise StructuredPrivacyLabError(
            "structured privacy lab requires at least one row"
        )
    profile = profile_structured_table(
        rows,
        quasi_identifiers=policy.quasi_identifiers,
        sensitive_attributes=policy.sensitive_attributes,
        direct_identifiers=policy.direct_identifiers,
        population_assumptions=population_assumptions,
        privacy_unit=policy.privacy_unit,
    )
    release_policy = policy.to_anonymity_policy()
    try:
        before = assess_release(rows, release_policy)
    except (TypeError, ValueError):
        raise StructuredPrivacyLabError(
            "structured privacy lab input failed the declared schema policy"
        ) from None

    membership_before = _run_membership_test(
        rows,
        membership_candidates,
        policy=policy,
    )
    anonymization: AnonymizationResult | None = None
    after: ReleaseAssessment | None = None
    transformation_status = "failed_policy"
    try:
        anonymization = anonymize_release(
            rows,
            release_policy,
            hierarchies=hierarchies,
        )
        after = anonymization.after
        transformation_status = "complete"
    except (TypeError, ValueError):
        # A strict policy may be impossible under the declared suppression
        # budget. The report records a safe failure state rather than echoing an
        # exception that could contain caller-controlled data.
        transformation_status = "failed_policy"

    membership_after = _run_membership_test(
        anonymization.records if anonymization is not None else None,
        membership_candidates,
        policy=policy,
    )
    result = StructuredPrivacyLabResult(
        policy=policy,
        profile=profile,
        before=before,
        after=after,
        anonymization=anonymization,
        membership_before=membership_before,
        membership_after=membership_after,
        transformation_status=transformation_status,
    )
    evidence = build_structured_privacy_evidence(
        result,
        dp_ledger=dp_ledger,
    )
    return replace(result, evidence=evidence)


def run_privacy_lab(
    records: Any,
    policy: StructuredPrivacyPolicy | None = None,
    **kwargs: Any,
) -> StructuredPrivacyLabResult:
    """Short alias for :func:`run_structured_privacy_lab`."""

    return run_structured_privacy_lab(records, policy, **kwargs)


def build_structured_privacy_evidence(
    result: StructuredPrivacyLabResult,
    *,
    dp_ledger: AggregateDPBudgetLedger | None = None,
) -> StructuredPrivacyEvidenceReport:
    """Build a deterministic SP 800-188-oriented evidence handoff."""

    if not isinstance(result, StructuredPrivacyLabResult):
        raise TypeError("result must be a StructuredPrivacyLabResult")
    return StructuredPrivacyEvidenceReport(
        result=result,
        population_assumptions=result.profile.population_assumptions,
        dp_ledger=dp_ledger,
    )


def make_synthetic_privacy_fixture(
    *,
    group_count: int = 4,
    rows_per_group: int = 3,
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic, clearly synthetic structured privacy fixture."""

    if type(group_count) is not int or group_count < 1:
        raise ValueError("group_count must be an integer >= 1")
    if type(rows_per_group) is not int or rows_per_group < 1:
        raise ValueError("rows_per_group must be an integer >= 1")
    rows: list[dict[str, Any]] = []
    for group in range(group_count):
        for member in range(rows_per_group):
            rows.append(
                {
                    "synthetic_record_id": f"synthetic-{group}-{member}",
                    "age": 30 + (group % 3) * 10,
                    "postal_prefix": f"SYN-{group:02d}",
                    "sensitive_outcome": "synthetic-a"
                    if member % 2 == 0
                    else "synthetic-b",
                }
            )
    return tuple(rows)


structured_privacy_fixture = make_synthetic_privacy_fixture


def _run_membership_test(
    released_records: Any,
    candidate_records: Any | None,
    *,
    policy: StructuredPrivacyPolicy,
) -> MembershipSelfTestResult | None:
    if candidate_records is None or released_records is None:
        return None
    try:
        return membership_inference_self_test(
            released_records,
            candidate_records,
            quasi_identifiers=policy.quasi_identifiers,
            max_candidates=policy.membership_max_candidates,
            max_inference_rate=policy.membership_max_inference_rate,
        )
    except MembershipSelfTestError:
        raise StructuredPrivacyLabError(
            "membership self-test failed the declared bounded configuration"
        ) from None


def _materialize_rows(data: Any) -> list[dict[str, Any]]:
    try:
        to_dicts = getattr(data, "to_dicts", None)
        if callable(to_dicts):
            data = to_dicts()
        else:
            to_dict = getattr(data, "to_dict", None)
            if callable(to_dict) and not isinstance(data, Mapping):
                data = to_dict("records")
        if isinstance(data, Mapping):
            rows: Any = [data]
        elif isinstance(data, Sequence) and not isinstance(
            data,
            (str, bytes, bytearray),
        ):
            rows = data
        else:
            raise TypeError
        if not all(isinstance(row, Mapping) for row in rows):
            raise TypeError
        materialized = [dict(row) for row in rows]
    except (AttributeError, TypeError, ValueError):
        raise StructuredPrivacyLabError(
            "structured input must be row mappings"
        ) from None
    for row in materialized:
        if any(not isinstance(field, str) for field in row):
            raise StructuredPrivacyLabError(
                "structured input column names must be strings"
            )
    return materialized


def _field_order(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        raise StructuredPrivacyLabError("structured input must contain a schema")
    return tuple(fields)


def _columns(
    value: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of column names")
    columns: list[str] = []
    for column in value:
        if not isinstance(column, str) or not column:
            raise ValueError(f"{name} must contain non-empty string names")
        if column not in columns:
            columns.append(column)
    if not columns and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return tuple(columns)


def _validate_role_overlap(
    *,
    qis: Sequence[str],
    sensitive: Sequence[str],
    direct: Sequence[str],
) -> None:
    groups = {
        "quasi_identifiers": set(qis),
        "sensitive_attributes": set(sensitive),
        "direct_identifiers": set(direct),
    }
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"privacy roles overlap between {left} and {right}")


def _validate_policy_numbers(policy: StructuredPrivacyPolicy) -> None:
    if type(policy.target_k) is not int or policy.target_k < 1:
        raise ValueError("target_k must be an integer >= 1")
    if type(policy.target_l) is not int or policy.target_l < 1:
        raise ValueError("target_l must be an integer >= 1")
    if (
        not isinstance(policy.target_t, (int, float))
        or isinstance(
            policy.target_t,
            bool,
        )
        or not math.isfinite(float(policy.target_t))
        or not 0.0 <= float(policy.target_t) <= 1.0
    ):
        raise ValueError("target_t must be between 0 and 1")
    if policy.suppression_limit is not None and (
        type(policy.suppression_limit) is not int or policy.suppression_limit < 0
    ):
        raise ValueError("suppression_limit must be an integer >= 0")
    if (
        isinstance(policy.suppression_rate, bool)
        or not isinstance(
            policy.suppression_rate,
            (int, float),
        )
        or not math.isfinite(float(policy.suppression_rate))
        or not 0.0 <= float(policy.suppression_rate) <= 1.0
    ):
        raise ValueError("suppression_rate must be between 0 and 1")
    if (
        type(policy.membership_max_candidates) is not int
        or policy.membership_max_candidates < 1
    ):
        raise ValueError("membership_max_candidates must be an integer >= 1")


def _optional_rate(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("membership_max_inference_rate must be between 0 and 1")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "membership_max_inference_rate must be between 0 and 1"
        ) from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("membership_max_inference_rate must be between 0 and 1")
    return parsed


def _value_token(value: Any) -> str:
    return stable_hash(
        {
            "artifact": "openmed-structured-profile-cell",
            "value": _canonical_value(value),
        }
    )


def _canonical_value(value: Any) -> Any:
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite cell values are unsupported")
        return {"type": "float", "value": repr(value)}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite cell values are unsupported")
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"type": "time", "value": value.isoformat()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "value": {
                str(key): _canonical_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            },
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {"type": "sequence", "value": [_canonical_value(item) for item in value]}
    raise ValueError("unsupported cell values are not allowed")


def _risk_delta(
    before: ReleaseAssessment,
    after: ReleaseAssessment | None,
) -> dict[str, Any]:
    if after is None:
        return {"status": "unavailable"}
    return {
        "status": "measured",
        "achieved_k_before": before.achieved_k,
        "achieved_k_after": after.achieved_k,
        "max_sample_identity_risk_before": before.max_sample_identity_risk,
        "max_sample_identity_risk_after": after.max_sample_identity_risk,
        "max_sample_identity_risk_reduction": (
            before.max_sample_identity_risk - after.max_sample_identity_risk
        ),
        "singleton_class_count_before": before.singleton_class_count,
        "singleton_class_count_after": after.singleton_class_count,
        "singleton_class_reduction": (
            before.singleton_class_count - after.singleton_class_count
        ),
        "utility_loss_is_reported": True,
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _is_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )
