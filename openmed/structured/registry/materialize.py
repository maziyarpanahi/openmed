"""Deterministic registry-case materialization from cohorts and Journey facts."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.clinical.review_sla import (
    ClinicalReviewQueueSummary,
    summarize_review_queue,
)
from openmed.clinical.review_transitions import ClinicalReviewPacket
from openmed.structured.cohort import CohortExecution
from openmed.structured.store import StoreResult, StoreState

from .contracts import (
    REGISTRY_ADVISORY,
    REGISTRY_COMPATIBILITY_POLICY,
    REGISTRY_SCHEMA_VERSION,
    RegistryCase,
    RegistryCaseState,
    RegistryConflictError,
    RegistryContractError,
    RegistryDefinition,
    RegistryDefinitionVersion,
    RegistryFieldEvidence,
    RegistryFieldResult,
    RegistryFieldRule,
    RegistryFieldState,
    RegistryUnsupportedError,
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RegistryFactBinding:
    """One Journey fact plus explicit correction ancestry."""

    fact: ClinicalFact
    corrected_from_fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fact, ClinicalFact):
            raise TypeError("fact must be ClinicalFact")
        corrected = tuple(sorted(set(self.corrected_from_fact_ids)))
        if len(corrected) != len(self.corrected_from_fact_ids):
            raise RegistryConflictError("corrected fact identifiers must be unique")
        for fact_id in corrected:
            if not isinstance(fact_id, str) or _OPAQUE_ID_RE.fullmatch(fact_id) is None:
                raise RegistryContractError("corrected fact identifiers must be opaque")
        if self.fact.fact_id in corrected:
            raise RegistryConflictError("a fact cannot correct itself")
        object.__setattr__(self, "corrected_from_fact_ids", corrected)


@dataclass(frozen=True, slots=True)
class RegistryMaterialization:
    """Deterministic cases, exclusions, and counts-only review queue summary."""

    definition_version_id: str
    definition_digest: str
    cohort_execution_id: str
    cohort_execution_digest: str
    source_snapshot_id: str
    source_snapshot_digest: str
    created_at: str
    cases: tuple[RegistryCase, ...]
    excluded_membership_counts: Mapping[str, int]
    review_summary: ClinicalReviewQueueSummary
    schema_version: str = REGISTRY_SCHEMA_VERSION
    compatibility_policy: str = REGISTRY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise RegistryUnsupportedError("registry schema version is unsupported")
        if self.compatibility_policy != REGISTRY_COMPATIBILITY_POLICY:
            raise RegistryUnsupportedError(
                "registry compatibility policy is unsupported"
            )
        for name in (
            "definition_version_id",
            "cohort_execution_id",
            "source_snapshot_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
                raise RegistryContractError(f"{name} must be opaque")
        for name in (
            "definition_digest",
            "cohort_execution_digest",
            "source_snapshot_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
                raise RegistryContractError(f"{name} must be a digest")
        created_at = _parse_timestamp(self.created_at)
        raw_cases = tuple(self.cases)
        if any(not isinstance(item, RegistryCase) for item in raw_cases):
            raise TypeError("cases must contain RegistryCase")
        cases = tuple(sorted(raw_cases, key=lambda item: item.case_id))
        if len({item.case_id for item in cases}) != len(cases):
            raise RegistryConflictError("registry case identifiers must be unique")
        for case in cases:
            if (
                case.definition_version_id != self.definition_version_id
                or case.definition_digest != self.definition_digest
                or case.cohort_execution_id != self.cohort_execution_id
                or case.cohort_execution_digest != self.cohort_execution_digest
                or case.source_snapshot_id != self.source_snapshot_id
                or case.source_snapshot_digest != self.source_snapshot_digest
                or _parse_timestamp(case.created_at) != created_at
            ):
                raise RegistryConflictError("registry case custody differs")
        raw_counts = self.excluded_membership_counts
        if not isinstance(raw_counts, Mapping):
            raise TypeError("excluded_membership_counts must be a mapping")
        if any(
            not isinstance(key, str) or type(value) is not int
            for key, value in raw_counts.items()
        ):
            raise RegistryContractError(
                "excluded membership counts must use text keys and integer values"
            )
        counts = dict(sorted(raw_counts.items()))
        if any(_CONTROLLED_RE.fullmatch(key) is None for key in counts):
            raise RegistryContractError("excluded membership states must be controlled")
        if any(value < 0 for value in counts.values()):
            raise RegistryContractError(
                "excluded membership counts must be non-negative"
            )
        if not isinstance(self.review_summary, ClinicalReviewQueueSummary):
            raise TypeError("review_summary must be ClinicalReviewQueueSummary")
        packet_ids = {
            packet.packet_id for case in cases for packet in case.review_packets
        }
        if self.review_summary.total != len(packet_ids):
            raise RegistryConflictError(
                "review summary total differs from case packets"
            )
        if _parse_timestamp(self.review_summary.generated_at) != created_at:
            raise RegistryConflictError("review summary timestamp differs")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "excluded_membership_counts", MappingProxyType(counts))

    @property
    def materialization_digest(self) -> str:
        """Return the digest of cases, exclusions, and source custody."""

        return canonical_digest(self.identity_payload)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return all materialization fields covered by the digest."""

        return {
            "cases": {item.case_id: item.case_digest for item in self.cases},
            "cohort_execution_digest": self.cohort_execution_digest,
            "cohort_execution_id": self.cohort_execution_id,
            "compatibility_policy": self.compatibility_policy,
            "created_at": self.created_at,
            "definition_digest": self.definition_digest,
            "definition_version_id": self.definition_version_id,
            "excluded_membership_counts": dict(self.excluded_membership_counts),
            "review_summary": self.review_summary.to_dict(),
            "schema_version": self.schema_version,
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete value-free materialization result."""

        return {
            "advisory": REGISTRY_ADVISORY,
            "artifact_type": "registry_materialization",
            "materialization_digest": self.materialization_digest,
            **self.identity_payload,
        }

    def to_json(self) -> str:
        """Return canonical materialization JSON."""

        return canonical_json(self.to_dict())


def version_registry_definition(
    definition: RegistryDefinition,
) -> RegistryDefinitionVersion:
    """Wrap one validated registry definition in a content-addressed version."""

    return RegistryDefinitionVersion(definition=definition)


def materialize_registry_cases(
    definition_version: RegistryDefinitionVersion,
    execution: CohortExecution,
    facts: Sequence[RegistryFactBinding | ClinicalFact],
    *,
    created_at: str,
    review_packets: Iterable[ClinicalReviewPacket] = (),
) -> StoreResult[RegistryMaterialization]:
    """Build value-free registry cases for eligible saved-cohort memberships."""

    try:
        if not isinstance(definition_version, RegistryDefinitionVersion):
            raise TypeError("definition_version must be RegistryDefinitionVersion")
        if not isinstance(execution, CohortExecution):
            raise TypeError("execution must be CohortExecution")
        definition = definition_version.definition
        if (
            execution.manifest.definition_version_id
            != definition.cohort_definition_version_id
            or execution.manifest.definition_digest
            != definition.cohort_definition_digest
        ):
            return StoreResult.outcome(
                StoreState.CONFLICT, "registry_cohort_definition_conflict"
            )
        normalized_facts = tuple(_fact_binding(item) for item in facts)
        if len({item.fact.fact_id for item in normalized_facts}) != len(
            normalized_facts
        ):
            return StoreResult.outcome(StoreState.CONFLICT, "registry_fact_duplicate")
        packets = tuple(review_packets)
        if any(not isinstance(item, ClinicalReviewPacket) for item in packets):
            raise TypeError("review_packets must contain ClinicalReviewPacket")
        if len({item.packet_id for item in packets}) != len(packets):
            return StoreResult.outcome(
                StoreState.CONFLICT, "registry_review_packet_duplicate"
            )
        eligible = {
            membership.patient_key
            for membership in execution.memberships
            if membership.eligible
        }
        by_subject: dict[str, list[RegistryFactBinding]] = defaultdict(list)
        for item in normalized_facts:
            if item.fact.subject_id in eligible:
                by_subject[item.fact.subject_id].append(item)
        cases = tuple(
            _materialize_case(
                definition_version,
                execution,
                subject_id,
                by_subject.get(subject_id, ()),
                created_at=created_at,
                review_packets=packets,
            )
            for subject_id in sorted(eligible)
        )
        selected_packets = {
            packet.packet_id: packet for case in cases for packet in case.review_packets
        }
        summary = summarize_review_queue(
            selected_packets.values(), now=_parse_timestamp(created_at)
        )
        excluded = Counter(
            membership.state.value
            for membership in execution.memberships
            if not membership.eligible
        )
        materialization = RegistryMaterialization(
            definition_version_id=definition_version.version_id or "",
            definition_digest=definition_version.definition_digest,
            cohort_execution_id=execution.manifest.execution_id or "",
            cohort_execution_digest=execution.execution_digest,
            source_snapshot_id=execution.manifest.source_snapshot.snapshot_id,
            source_snapshot_digest=execution.manifest.source_snapshot.digest,
            created_at=created_at,
            cases=cases,
            excluded_membership_counts=excluded,
            review_summary=summary,
        )
    except RegistryUnsupportedError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "registry_version_unsupported"
        )
    except RegistryConflictError:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_materialization_conflict"
        )
    except (RegistryContractError, TypeError, ValueError):
        return StoreResult.outcome(
            StoreState.FAILURE, "registry_materialization_invalid"
        )
    if not cases:
        return StoreResult.outcome(
            StoreState.UNKNOWN,
            "registry_cohort_empty",
            value=materialization,
        )
    return StoreResult.success(materialization)


def _materialize_case(
    definition_version: RegistryDefinitionVersion,
    execution: CohortExecution,
    subject_id: str,
    facts: Sequence[RegistryFactBinding],
    *,
    created_at: str,
    review_packets: Sequence[ClinicalReviewPacket],
) -> RegistryCase:
    definition = definition_version.definition
    by_type: dict[str, list[RegistryFactBinding]] = defaultdict(list)
    for item in facts:
        by_type[item.fact.fact_type].append(item)
    fields = tuple(
        _field_result(rule, by_type.get(rule.fact_type, ()))
        for rule in definition.fields
    )
    field_fact_ids = {
        fact_id for result in fields for fact_id in result.evidence.fact_ids
    }
    packets = tuple(
        packet for packet in review_packets if set(packet.fact_ids) <= field_fact_ids
    )
    origin_state = (
        RegistryCaseState.REVIEW_REQUIRED
        if any(
            result.state in definition.workflow.review_field_states for result in fields
        )
        else RegistryCaseState.EXPORT_READY
    )
    case_id = derived_opaque_id(
        "registrycase",
        definition_version.version_id,
        execution.manifest.execution_id,
        subject_id,
    )
    return RegistryCase(
        case_id=case_id,
        definition_version_id=definition_version.version_id or "",
        definition_digest=definition_version.definition_digest,
        subject_id=subject_id,
        cohort_execution_id=execution.manifest.execution_id or "",
        cohort_execution_digest=execution.execution_digest,
        source_snapshot_id=execution.manifest.source_snapshot.snapshot_id,
        source_snapshot_digest=execution.manifest.source_snapshot.digest,
        created_at=created_at,
        fields=fields,
        origin_state=origin_state,
        state=origin_state,
        workflow_policy_digest=definition.workflow.digest,
        review_packets=packets,
    )


def _field_result(
    rule: RegistryFieldRule,
    bindings: Sequence[RegistryFactBinding],
) -> RegistryFieldResult:
    ordered = tuple(sorted(bindings, key=lambda item: item.fact.fact_id))
    fact_ids = tuple(item.fact.fact_id for item in ordered)
    evidence_ids = tuple(
        sorted({evidence for item in ordered for evidence in item.fact.evidence_ids})
    )
    value_digests = tuple(canonical_digest(item.fact.value) for item in ordered)
    derivation_digests = tuple(item.fact.derivation_hash for item in ordered)
    corrected_from = tuple(
        sorted(
            {fact_id for item in ordered for fact_id in item.corrected_from_fact_ids}
        )
    )
    evidence = RegistryFieldEvidence(
        fact_ids=fact_ids,
        evidence_ids=evidence_ids,
        value_digests=value_digests,
        derivation_digests=derivation_digests,
        corrected_from_fact_ids=corrected_from,
    )
    statuses = {item.fact.status for item in ordered}
    distinct_values = set(value_digests)
    if not ordered:
        state = (
            RegistryFieldState.MISSING_REQUIRED
            if rule.required
            else RegistryFieldState.NOT_APPLICABLE
        )
        reason = "required_fact_missing" if rule.required else "optional_fact_absent"
    elif (
        statuses.intersection(rule.conflict_statuses)
        or len(distinct_values) > rule.max_values
    ):
        state = RegistryFieldState.CONFLICT
        reason = "fact_conflict"
    elif statuses.intersection(rule.unknown_statuses):
        state = RegistryFieldState.UNKNOWN
        reason = "fact_unknown"
    elif not statuses <= set(rule.allowed_statuses):
        state = RegistryFieldState.UNSUPPORTED
        reason = "fact_status_unsupported"
    elif corrected_from:
        state = RegistryFieldState.CORRECTED
        reason = "fact_corrected"
    else:
        state = RegistryFieldState.PRESENT
        reason = "fact_present"
    return RegistryFieldResult(
        field_id=rule.field_id,
        state=state,
        evidence=evidence,
        reason_code=reason,
    )


def _fact_binding(value: RegistryFactBinding | ClinicalFact) -> RegistryFactBinding:
    if isinstance(value, RegistryFactBinding):
        return value
    if isinstance(value, ClinicalFact):
        return RegistryFactBinding(fact=value)
    raise TypeError("facts must contain RegistryFactBinding or ClinicalFact")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RegistryContractError("created_at must be timezone-aware")
    return parsed


__all__ = [
    "RegistryFactBinding",
    "RegistryMaterialization",
    "materialize_registry_cases",
    "version_registry_definition",
]
