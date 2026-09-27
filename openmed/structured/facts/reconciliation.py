"""Deterministic fact deduplication, conflicts, policy, and corrections.

This module consumes immutable Journey ``ClinicalFact`` records.  It groups
policy-equivalent facts, emits one ``ConflictSet`` for every conflicting
dimension, records every policy decision as an append-only ``ResolutionEvent``,
and creates review packets when policy is insufficient.  It never mutates a
fact or its evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    ConflictSet,
    ResolutionEvent,
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.clinical.review_sla import (
    ClinicalReviewQueueSummary,
    summarize_review_queue,
)
from openmed.clinical.review_transitions import ClinicalReviewPacket
from openmed.structured.store import (
    CanonicalRecord,
    JobMetadata,
    StoreResult,
    StoreState,
    TransactionalJourneyStore,
)

FACT_RECONCILIATION_SCHEMA_VERSION = "1.0.0"
FACT_RECONCILIATION_COMPATIBILITY_POLICY = "same_major"
FACT_RECONCILER_VERSION = "1.0.0"

FACT_CONFLICT_TYPES = frozenset(
    {"value", "unit", "time", "status", "source", "identity", "amendment"}
)
FACT_GROUP_KINDS = frozenset({"unique", "exact_duplicate", "policy_equivalent"})
_PRIORITIES = {"low": 0, "normal": 1, "high": 2, "critical": 3}
_NON_AUTOMATABLE_CONFLICTS = frozenset({"identity"})
_SEMANTIC_ATTRIBUTE_KEYS = frozenset(
    {
        "assertion",
        "certainty",
        "experiencer",
        "mapping",
        "normalized_code",
        "relation_participants",
        "terminology",
    }
)

_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class FactReconciliationError(ValueError):
    """Raised when fact reconciliation inputs violate the public contract."""


@dataclass(frozen=True, slots=True)
class FactReconciliationInput:
    """One immutable fact plus value-free reconciliation metadata."""

    fact: ClinicalFact = field(repr=False)
    reconciliation_id: str
    source: str
    amendment_of: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fact, ClinicalFact):
            raise TypeError("fact must be a ClinicalFact")
        if _OPAQUE_ID_RE.fullmatch(self.reconciliation_id) is None:
            raise FactReconciliationError(
                "reconciliation_id must be an opaque identifier"
            )
        if _CONTROLLED_RE.fullmatch(self.source) is None:
            raise FactReconciliationError("source must be a controlled identifier")
        if self.amendment_of is not None:
            if _OPAQUE_ID_RE.fullmatch(self.amendment_of) is None:
                raise FactReconciliationError(
                    "amendment_of must be an opaque identifier"
                )
            if self.amendment_of == self.fact.fact_id:
                raise FactReconciliationError("a fact cannot amend itself")
            if self.amendment_of not in self.fact.parent_fact_ids:
                raise FactReconciliationError(
                    "an amendment must retain its corrected fact as a parent"
                )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return identifiers and source metadata without clinical values."""

        return {
            "amendment_of": self.amendment_of,
            "fact_id": self.fact.fact_id,
            "reconciliation_id": self.reconciliation_id,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class FactReconciliationPolicy:
    """Versioned deterministic equivalence and resolution policy."""

    policy_id: str = "openmed.fact.reconciliation"
    version: str = "1.0.0"
    status_equivalence: Mapping[str, str] = field(default_factory=dict)
    unit_equivalence: Mapping[str, str] = field(default_factory=dict)
    source_priority: Mapping[str, int] = field(default_factory=dict)
    auto_resolve_conflicts: frozenset[str] = frozenset()
    prefer_valid_amendment: bool = False
    time_tolerance_seconds: float = 0.0
    review_priority: Mapping[str, str] = field(default_factory=dict)
    review_ttl_hours: float = 72.0

    def __post_init__(self) -> None:
        if _CONTROLLED_RE.fullmatch(self.policy_id) is None:
            raise FactReconciliationError("policy_id must be controlled")
        if _VERSION_RE.fullmatch(self.version) is None:
            raise FactReconciliationError("policy version must be semantic")
        statuses = _controlled_mapping(self.status_equivalence, "status equivalence")
        units = _text_mapping(self.unit_equivalence, "unit equivalence")
        priorities: dict[str, int] = {}
        for source, priority in self.source_priority.items():
            if _CONTROLLED_RE.fullmatch(source) is None:
                raise FactReconciliationError("source priority key is invalid")
            if type(priority) is not int:
                raise FactReconciliationError("source priorities must be integers")
            priorities[source] = priority
        auto = frozenset(self.auto_resolve_conflicts)
        if not auto <= FACT_CONFLICT_TYPES:
            raise FactReconciliationError("auto-resolve conflict type is unsupported")
        if isinstance(self.time_tolerance_seconds, bool) or not isinstance(
            self.time_tolerance_seconds, (int, float)
        ):
            raise FactReconciliationError("time tolerance must be numeric")
        tolerance = float(self.time_tolerance_seconds)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise FactReconciliationError(
                "time tolerance must be finite and non-negative"
            )
        if isinstance(self.review_ttl_hours, bool) or not isinstance(
            self.review_ttl_hours, (int, float)
        ):
            raise FactReconciliationError("review TTL must be numeric")
        ttl = float(self.review_ttl_hours)
        if not math.isfinite(ttl) or ttl <= 0:
            raise FactReconciliationError("review TTL must be finite and positive")
        review_priority: dict[str, str] = {}
        for conflict_type, review_priority_value in self.review_priority.items():
            if (
                conflict_type not in FACT_CONFLICT_TYPES
                or review_priority_value not in _PRIORITIES
            ):
                raise FactReconciliationError("review priority mapping is unsupported")
            review_priority[conflict_type] = review_priority_value
        object.__setattr__(self, "status_equivalence", MappingProxyType(statuses))
        object.__setattr__(self, "unit_equivalence", MappingProxyType(units))
        object.__setattr__(self, "source_priority", MappingProxyType(priorities))
        object.__setattr__(self, "auto_resolve_conflicts", auto)
        object.__setattr__(self, "review_priority", MappingProxyType(review_priority))
        object.__setattr__(self, "time_tolerance_seconds", tolerance)
        object.__setattr__(self, "review_ttl_hours", ttl)

    @property
    def digest(self) -> str:
        """Return a stable policy fingerprint."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic public policy controls."""

        return {
            "auto_resolve_conflicts": sorted(self.auto_resolve_conflicts),
            "policy_id": self.policy_id,
            "prefer_valid_amendment": self.prefer_valid_amendment,
            "review_priority": dict(self.review_priority),
            "review_ttl_hours": self.review_ttl_hours,
            "source_priority": dict(self.source_priority),
            "status_equivalence": dict(self.status_equivalence),
            "time_tolerance_seconds": self.time_tolerance_seconds,
            "unit_equivalence": dict(self.unit_equivalence),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class FactEquivalenceGroup:
    """Evidence-preserving group of exact or policy-equivalent facts."""

    group_id: str
    reconciliation_id: str
    kind: str
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    representative_fact_id: str
    policy_digest: str

    def __post_init__(self) -> None:
        if _OPAQUE_ID_RE.fullmatch(self.group_id) is None:
            raise FactReconciliationError("group_id must be opaque")
        if _OPAQUE_ID_RE.fullmatch(self.reconciliation_id) is None:
            raise FactReconciliationError("reconciliation_id must be opaque")
        if self.kind not in FACT_GROUP_KINDS:
            raise FactReconciliationError("unsupported equivalence group kind")
        fact_ids = tuple(sorted(set(self.fact_ids)))
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not fact_ids or self.representative_fact_id not in fact_ids:
            raise FactReconciliationError("group representative must be a member")
        if not evidence_ids:
            raise FactReconciliationError("group must preserve supporting evidence")
        object.__setattr__(self, "fact_ids", fact_ids)
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return source-value-free grouping provenance."""

        return {
            "evidence_ids": list(self.evidence_ids),
            "fact_ids": list(self.fact_ids),
            "group_id": self.group_id,
            "kind": self.kind,
            "policy_digest": self.policy_digest,
            "reconciliation_id": self.reconciliation_id,
            "representative_fact_id": self.representative_fact_id,
        }


@dataclass(frozen=True, slots=True)
class FactReconciliationResult:
    """Replayable plan over immutable facts, conflicts, and review packets."""

    reconciliation_run_id: str
    subject_id: str
    occurred_at: str
    input_fact_ids: tuple[str, ...]
    equivalence_groups: tuple[FactEquivalenceGroup, ...]
    conflicts: tuple[ConflictSet, ...]
    resolutions: tuple[ResolutionEvent, ...]
    canonical_records: tuple[CanonicalRecord, ...]
    review_packets: tuple[ClinicalReviewPacket, ...]
    queue_summary: ClinicalReviewQueueSummary
    policy: FactReconciliationPolicy
    state: str
    schema_version: str = FACT_RECONCILIATION_SCHEMA_VERSION
    compatibility_policy: str = FACT_RECONCILIATION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if _OPAQUE_ID_RE.fullmatch(self.reconciliation_run_id) is None:
            raise FactReconciliationError("reconciliation_run_id must be opaque")
        if _OPAQUE_ID_RE.fullmatch(self.subject_id) is None:
            raise FactReconciliationError("subject_id must be opaque")
        if _TIMESTAMP_RE.fullmatch(self.occurred_at) is None:
            raise FactReconciliationError("occurred_at must be timezone-aware")
        if self.state not in {"resolved", "review_required"}:
            raise FactReconciliationError("unsupported reconciliation state")
        if bool(self.review_packets) != (self.state == "review_required"):
            raise FactReconciliationError("result state must match review packets")
        if self.schema_version != FACT_RECONCILIATION_SCHEMA_VERSION:
            raise FactReconciliationError("unsupported result schema version")
        if self.compatibility_policy != FACT_RECONCILIATION_COMPATIBILITY_POLICY:
            raise FactReconciliationError("unsupported compatibility policy")
        input_ids = tuple(sorted(set(self.input_fact_ids)))
        if not input_ids or len(input_ids) != len(self.input_fact_ids):
            raise FactReconciliationError("input fact identifiers must be unique")
        if any(_OPAQUE_ID_RE.fullmatch(item) is None for item in input_ids):
            raise FactReconciliationError("input fact identifiers must be opaque")
        conflict_ids = [item.conflict_id for item in self.conflicts]
        if len(conflict_ids) != len(set(conflict_ids)):
            raise FactReconciliationError("conflict identifiers must be unique")
        resolution_ids = [item.resolution_id for item in self.resolutions]
        if len(resolution_ids) != len(set(resolution_ids)):
            raise FactReconciliationError("resolution identifiers must be unique")
        if any(
            not set(item.fact_ids) <= set(input_ids) for item in self.equivalence_groups
        ):
            raise FactReconciliationError("equivalence group references unknown fact")
        if any(not set(item.fact_ids) <= set(input_ids) for item in self.conflicts):
            raise FactReconciliationError("conflict references unknown fact")
        if any(item.conflict_id not in set(conflict_ids) for item in self.resolutions):
            raise FactReconciliationError("resolution references unknown conflict")
        if any(
            item.conflict_id not in set(conflict_ids) for item in self.review_packets
        ):
            raise FactReconciliationError("review packet references unknown conflict")
        if any(
            record.fact_id not in set(input_ids) for record in self.canonical_records
        ):
            raise FactReconciliationError("canonical record references unknown fact")
        if self.queue_summary.total != len(self.review_packets):
            raise FactReconciliationError("queue summary does not match review packets")
        object.__setattr__(self, "input_fact_ids", input_ids)

    @property
    def current_fact_ids(self) -> tuple[str, ...]:
        """Return selected canonical facts; unresolved groups remain absent."""

        return tuple(sorted({record.fact_id for record in self.canonical_records}))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic public plan with no raw clinical values."""

        return {
            "canonical_records": [item.to_dict() for item in self.canonical_records],
            "compatibility_policy": self.compatibility_policy,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "current_fact_ids": list(self.current_fact_ids),
            "equivalence_groups": [item.to_dict() for item in self.equivalence_groups],
            "input_fact_ids": list(self.input_fact_ids),
            "occurred_at": self.occurred_at,
            "policy": self.policy.to_dict() | {"digest": self.policy.digest},
            "queue_summary": self.queue_summary.to_dict(),
            "reconciliation_run_id": self.reconciliation_run_id,
            "resolutions": [item.to_dict() for item in self.resolutions],
            "review_packets": [item.to_dict() for item in self.review_packets],
            "schema_version": self.schema_version,
            "state": self.state,
            "subject_id": self.subject_id,
        }

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())


class FactReconciler:
    """Plan deterministic reconciliation without mutating input facts."""

    def __init__(self, policy: FactReconciliationPolicy | None = None) -> None:
        self.policy = policy or FactReconciliationPolicy()

    def reconcile(
        self,
        subject_id: str,
        inputs: Iterable[FactReconciliationInput],
        *,
        occurred_at: str,
        previous_resolutions: Mapping[str, ResolutionEvent] | None = None,
    ) -> StoreResult[FactReconciliationResult]:
        """Build groups, conflicts, policy events, and review packets."""

        if _OPAQUE_ID_RE.fullmatch(subject_id) is None:
            return StoreResult.outcome(StoreState.FAILURE, "subject_id_invalid")
        if _TIMESTAMP_RE.fullmatch(occurred_at) is None:
            return StoreResult.outcome(StoreState.FAILURE, "occurred_at_invalid")
        materialized = tuple(inputs)
        if not materialized:
            return StoreResult.outcome(StoreState.UNKNOWN, "facts_empty")
        if len({item.fact.fact_id for item in materialized}) != len(materialized):
            return StoreResult.outcome(StoreState.CONFLICT, "fact_id_duplicate")
        previous = dict(previous_resolutions or {})
        grouped: dict[str, list[FactReconciliationInput]] = defaultdict(list)
        for item in materialized:
            grouped[item.reconciliation_id].append(item)

        equivalence_groups: list[FactEquivalenceGroup] = []
        conflicts: list[ConflictSet] = []
        resolutions: list[ResolutionEvent] = []
        canonical_records: list[CanonicalRecord] = []
        packets: list[ClinicalReviewPacket] = []

        for reconciliation_id in sorted(grouped):
            members = tuple(
                sorted(grouped[reconciliation_id], key=lambda item: item.fact.fact_id)
            )
            assessment = self._assess_group(subject_id, members)
            if not assessment.conflict_types:
                group = self._equivalence_group(reconciliation_id, members, assessment)
                equivalence_groups.append(group)
                canonical_records.append(
                    self._canonical_record(
                        subject_id=subject_id,
                        reconciliation_id=reconciliation_id,
                        selected_fact_id=group.representative_fact_id,
                        fact_by_id={item.fact.fact_id: item.fact for item in members},
                        occurred_at=occurred_at,
                        reason_code=group.kind,
                    )
                )
                continue

            selected = self._select_candidate(members, assessment.conflict_types)
            group_conflicts = tuple(
                self._conflict(
                    subject_id,
                    reconciliation_id,
                    members,
                    conflict_type,
                )
                for conflict_type in sorted(assessment.conflict_types)
            )
            conflicts.extend(group_conflicts)
            if selected is not None:
                canonical_records.append(
                    self._canonical_record(
                        subject_id=subject_id,
                        reconciliation_id=reconciliation_id,
                        selected_fact_id=selected.fact.fact_id,
                        fact_by_id={item.fact.fact_id: item.fact for item in members},
                        occurred_at=occurred_at,
                        reason_code="policy_resolution",
                    )
                )
            for conflict in group_conflicts:
                resolution = self._resolution(
                    conflict,
                    members,
                    selected=selected,
                    occurred_at=occurred_at,
                    previous=previous.get(conflict.conflict_id),
                )
                resolutions.append(resolution)
                if selected is None:
                    packets.append(
                        self._review_packet(
                            conflict,
                            resolution,
                            occurred_at=occurred_at,
                        )
                    )

        packets_tuple = tuple(sorted(packets, key=lambda item: item.packet_id))
        now = _parse_timestamp(occurred_at)
        summary = summarize_review_queue(packets_tuple, now=now)
        input_ids = tuple(sorted(item.fact.fact_id for item in materialized))
        run_id = derived_opaque_id(
            "reconciliation",
            subject_id,
            input_ids,
            self.policy.digest,
            occurred_at,
        )
        result = FactReconciliationResult(
            reconciliation_run_id=run_id,
            subject_id=subject_id,
            occurred_at=occurred_at,
            input_fact_ids=input_ids,
            equivalence_groups=tuple(
                sorted(equivalence_groups, key=lambda item: item.group_id)
            ),
            conflicts=tuple(sorted(conflicts, key=lambda item: item.conflict_id)),
            resolutions=tuple(sorted(resolutions, key=lambda item: item.resolution_id)),
            canonical_records=tuple(
                sorted(canonical_records, key=lambda item: item.canonical_id)
            ),
            review_packets=packets_tuple,
            queue_summary=summary,
            policy=self.policy,
            state="review_required" if packets_tuple else "resolved",
        )
        if packets_tuple:
            return StoreResult.outcome(
                StoreState.CONFLICT,
                "fact_review_required",
                value=result,
            )
        return StoreResult.success(result)

    def _assess_group(
        self,
        subject_id: str,
        members: Sequence[FactReconciliationInput],
    ) -> "_GroupAssessment":
        if len(members) == 1 and members[0].fact.subject_id == subject_id:
            return _GroupAssessment(kind="unique", conflict_types=frozenset())
        conflict_types: set[str] = set()
        facts = tuple(item.fact for item in members)
        if (
            any(fact.subject_id != subject_id for fact in facts)
            or len({fact.encounter_id for fact in facts}) > 1
        ):
            conflict_types.add("identity")
        if len({_semantic_value_digest(fact) for fact in facts}) > 1:
            conflict_types.add("value")
        if len({self._canonical_unit(fact.unit) for fact in facts}) > 1:
            conflict_types.add("unit")
        if not _times_equivalent(facts, self.policy.time_tolerance_seconds):
            conflict_types.add("time")
        if len({self._canonical_status(fact.status) for fact in facts}) > 1:
            conflict_types.add("status")
        clinical_variants = {
            _policy_clinical_digest(
                fact,
                status=self._canonical_status(fact.status),
                unit=self._canonical_unit(fact.unit),
            )
            for fact in facts
        }
        if len({item.source for item in members}) > 1 and len(clinical_variants) > 1:
            conflict_types.add("source")
        if _amendment_conflict(members):
            conflict_types.add("amendment")
        if conflict_types:
            return _GroupAssessment(
                kind="conflict", conflict_types=frozenset(conflict_types)
            )
        exact = len({_exact_clinical_digest(fact) for fact in facts}) == 1
        return _GroupAssessment(
            kind="exact_duplicate" if exact else "policy_equivalent",
            conflict_types=frozenset(),
        )

    def _equivalence_group(
        self,
        reconciliation_id: str,
        members: Sequence[FactReconciliationInput],
        assessment: "_GroupAssessment",
    ) -> FactEquivalenceGroup:
        representative = min(members, key=self._rank_input)
        fact_ids = tuple(item.fact.fact_id for item in members)
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for item in members
                    for evidence_id in item.fact.evidence_ids
                }
            )
        )
        return FactEquivalenceGroup(
            group_id=derived_opaque_id(
                "factgroup",
                reconciliation_id,
                fact_ids,
                assessment.kind,
                self.policy.digest,
            ),
            reconciliation_id=reconciliation_id,
            kind=assessment.kind,
            fact_ids=fact_ids,
            evidence_ids=evidence_ids,
            representative_fact_id=representative.fact.fact_id,
            policy_digest=self.policy.digest,
        )

    def _select_candidate(
        self,
        members: Sequence[FactReconciliationInput],
        conflict_types: frozenset[str],
    ) -> FactReconciliationInput | None:
        if conflict_types & _NON_AUTOMATABLE_CONFLICTS:
            return None
        amended = _valid_amendment_leaf(members)
        if self.policy.prefer_valid_amendment and amended is not None:
            return amended
        if not conflict_types <= self.policy.auto_resolve_conflicts:
            return None
        ordered = sorted(members, key=self._rank_input)
        if len(ordered) == 1:
            return ordered[0]
        first_priority = self.policy.source_priority.get(ordered[0].source, 0)
        second_priority = self.policy.source_priority.get(ordered[1].source, 0)
        if first_priority <= second_priority:
            return None
        return ordered[0]

    def _rank_input(self, item: FactReconciliationInput) -> tuple[int, float, str]:
        confidence = item.fact.confidence if item.fact.confidence is not None else 0.0
        return (
            -self.policy.source_priority.get(item.source, 0),
            -confidence,
            item.fact.fact_id,
        )

    def _conflict(
        self,
        subject_id: str,
        reconciliation_id: str,
        members: Sequence[FactReconciliationInput],
        conflict_type: str,
    ) -> ConflictSet:
        fact_ids = tuple(sorted(item.fact.fact_id for item in members))
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for item in members
                    for evidence_id in item.fact.evidence_ids
                }
            )
        )
        conflict_id = derived_opaque_id(
            "conflict", subject_id, reconciliation_id, conflict_type, fact_ids
        )
        derivation_hash = canonical_digest(
            {
                "component": "openmed.fact.reconciliation",
                "component_version": FACT_RECONCILER_VERSION,
                "conflict_type": conflict_type,
                "fact_ids": fact_ids,
                "reconciliation_id": reconciliation_id,
            }
        )
        return ConflictSet(
            conflict_id=conflict_id,
            subject_id=subject_id,
            conflict_type=conflict_type,
            fact_ids=fact_ids,
            status="open",
            detected_by="openmed.fact.reconciliation",
            derivation_hash=derivation_hash,
            evidence_ids=evidence_ids,
            attributes={
                "compatibility_policy": FACT_RECONCILIATION_COMPATIBILITY_POLICY,
                "contract_version": FACT_RECONCILIATION_SCHEMA_VERSION,
                "reconciliation_id": reconciliation_id,
            },
        )

    def _resolution(
        self,
        conflict: ConflictSet,
        members: Sequence[FactReconciliationInput],
        *,
        selected: FactReconciliationInput | None,
        occurred_at: str,
        previous: ResolutionEvent | None,
    ) -> ResolutionEvent:
        selected_ids = () if selected is None else (selected.fact.fact_id,)
        rejected_ids = tuple(
            sorted(
                item.fact.fact_id
                for item in members
                if selected is not None and item.fact.fact_id != selected.fact.fact_id
            )
        )
        action = "defer" if selected is None else "select"
        rationale = "policy_insufficient" if selected is None else "policy_priority"
        if (
            previous is not None
            and previous.policy_id == self.policy.policy_id
            and previous.policy_version == self.policy.version
            and previous.action == action
            and previous.selected_fact_ids == selected_ids
            and previous.rejected_fact_ids == rejected_ids
        ):
            return previous
        supersedes = previous.resolution_id if previous is not None else None
        resolution_id = derived_opaque_id(
            "resolution",
            conflict.conflict_id,
            action,
            selected_ids,
            rejected_ids,
            self.policy.digest,
            occurred_at,
            supersedes,
        )
        return ResolutionEvent(
            resolution_id=resolution_id,
            conflict_id=conflict.conflict_id,
            action=action,
            actor_type="policy",
            policy_id=self.policy.policy_id,
            policy_version=self.policy.version,
            occurred_at=occurred_at,
            rationale_code=rationale,
            derivation_hash=canonical_digest(
                {
                    "action": action,
                    "conflict_id": conflict.conflict_id,
                    "policy_digest": self.policy.digest,
                    "rejected_fact_ids": rejected_ids,
                    "selected_fact_ids": selected_ids,
                    "supersedes_resolution_id": supersedes,
                }
            ),
            selected_fact_ids=selected_ids,
            rejected_fact_ids=rejected_ids,
            supersedes_resolution_id=supersedes,
            attributes={
                "compatibility_policy": FACT_RECONCILIATION_COMPATIBILITY_POLICY,
                "contract_version": FACT_RECONCILIATION_SCHEMA_VERSION,
                "input_fact_count": len(members),
                "policy_digest": self.policy.digest,
            },
        )

    def _review_packet(
        self,
        conflict: ConflictSet,
        resolution: ResolutionEvent,
        *,
        occurred_at: str,
    ) -> ClinicalReviewPacket:
        priority = self.policy.review_priority.get(
            conflict.conflict_type,
            "critical" if conflict.conflict_type == "identity" else "high",
        )
        created = _parse_timestamp(occurred_at)
        expires = created + timedelta(hours=self.policy.review_ttl_hours)
        packet_id = derived_opaque_id(
            "reviewpacket",
            conflict.conflict_id,
            self.policy.digest,
            resolution.resolution_id,
        )
        return ClinicalReviewPacket(
            packet_id=packet_id,
            conflict_id=conflict.conflict_id,
            fact_ids=conflict.fact_ids,
            state="queued",
            priority=priority,
            created_at=occurred_at,
            expires_at=expires.isoformat(),
            policy_id=self.policy.policy_id,
            policy_version=self.policy.version,
            provenance_fingerprint=canonical_digest(
                {
                    "conflict_id": conflict.conflict_id,
                    "resolution_id": resolution.resolution_id,
                }
            ),
            extensions={"conflict_type": conflict.conflict_type},
        )

    def _canonical_status(self, status: str) -> str:
        return self.policy.status_equivalence.get(status, status)

    def _canonical_unit(self, unit: str | None) -> str | None:
        if unit is None:
            return None
        normalized = " ".join(unit.casefold().split())
        return self.policy.unit_equivalence.get(normalized, normalized)

    def _canonical_record(
        self,
        *,
        subject_id: str,
        reconciliation_id: str,
        selected_fact_id: str,
        fact_by_id: Mapping[str, ClinicalFact],
        occurred_at: str,
        reason_code: str,
    ) -> CanonicalRecord:
        fact = fact_by_id[selected_fact_id]
        return CanonicalRecord(
            canonical_id=reconciliation_id,
            subject_id=subject_id,
            fact_id=selected_fact_id,
            record_type=fact.fact_type,
            state=fact.status,
            effective_at=occurred_at,
            reason_code=reason_code,
            metadata={
                "compatibility_policy": FACT_RECONCILIATION_COMPATIBILITY_POLICY,
                "contract_version": FACT_RECONCILIATION_SCHEMA_VERSION,
                "policy_digest": self.policy.digest,
                "policy_id": self.policy.policy_id,
                "policy_version": self.policy.version,
            },
        )


def persist_fact_reconciliation(
    result: FactReconciliationResult,
    store: TransactionalJourneyStore,
    *,
    committed_at: str,
) -> StoreResult[FactReconciliationResult]:
    """Atomically append conflicts, resolutions, pointers, and review packets.

    Input facts and evidence must already exist in ``store``. Review packets
    use append-only ``JobMetadata`` versions so later guarded state transitions
    preserve the original queued packet and point-in-time reads.
    """

    try:
        created = False
        revision: int | None = None
        with store.transaction(committed_at=committed_at) as transaction:
            revision = transaction.revision
            for conflict in result.conflicts:
                conflict_write = transaction.put_conflict(conflict)
                if not conflict_write.ok:
                    return StoreResult.outcome(
                        conflict_write.state,
                        conflict_write.code or "conflict_write_failed",
                    )
                created = created or conflict_write.created
            for resolution in result.resolutions:
                resolution_write = transaction.put_resolution(resolution)
                if not resolution_write.ok:
                    return StoreResult.outcome(
                        resolution_write.state,
                        resolution_write.code or "resolution_write_failed",
                    )
                created = created or resolution_write.created
            for record in result.canonical_records:
                canonical_write = transaction.put_canonical(record)
                if not canonical_write.ok:
                    return StoreResult.outcome(
                        canonical_write.state,
                        canonical_write.code or "canonical_write_failed",
                    )
                created = created or canonical_write.created
            for packet in result.review_packets:
                metadata = JobMetadata(
                    job_id=packet.packet_id,
                    state=packet.state,
                    recorded_at=committed_at,
                    metadata={
                        "packet": packet.to_dict(),
                        "packet_digest": canonical_digest(packet.to_dict()),
                    },
                )
                packet_write = transaction.put_job(metadata)
                if not packet_write.ok:
                    return StoreResult.outcome(
                        packet_write.state,
                        packet_write.code or "review_packet_write_failed",
                    )
                created = created or packet_write.created
    except (TypeError, ValueError, RuntimeError):
        return StoreResult.outcome(StoreState.FAILURE, "reconciliation_write_failed")
    if result.state == "review_required":
        return StoreResult.outcome(
            StoreState.CONFLICT,
            "fact_review_required",
            value=result,
            revision=revision if created else None,
        )
    return StoreResult.success(
        result,
        created=created,
        revision=revision if created else None,
    )


def build_human_resolution(
    packet: ClinicalReviewPacket,
    *,
    selected_fact_ids: Iterable[str],
    rejected_fact_ids: Iterable[str],
    occurred_at: str,
    policy_id: str,
    policy_version: str,
    rationale_code: str,
    supersedes_resolution_id: str,
) -> StoreResult[ResolutionEvent]:
    """Build an identity-free human resolution from a completed review packet."""

    if packet.state not in {"approved", "rejected"}:
        return StoreResult.outcome(StoreState.CONFLICT, "review_not_complete")
    selected = tuple(sorted(set(selected_fact_ids)))
    rejected = tuple(sorted(set(rejected_fact_ids)))
    if not selected and not rejected:
        return StoreResult.outcome(StoreState.CONFLICT, "review_decision_empty")
    if set(selected).intersection(rejected):
        return StoreResult.outcome(StoreState.CONFLICT, "review_decision_overlap")
    if not set((*selected, *rejected)) <= set(packet.fact_ids):
        return StoreResult.outcome(StoreState.CONFLICT, "review_fact_mismatch")
    if _OPAQUE_ID_RE.fullmatch(supersedes_resolution_id) is None:
        return StoreResult.outcome(StoreState.FAILURE, "resolution_parent_invalid")
    action = "select" if selected else "reject"
    try:
        event = ResolutionEvent(
            resolution_id=derived_opaque_id(
                "resolution",
                packet.conflict_id,
                action,
                selected,
                rejected,
                policy_id,
                policy_version,
                occurred_at,
                supersedes_resolution_id,
            ),
            conflict_id=packet.conflict_id,
            action=action,
            actor_type="human",
            policy_id=policy_id,
            policy_version=policy_version,
            occurred_at=occurred_at,
            rationale_code=rationale_code,
            derivation_hash=canonical_digest(
                {
                    "action": action,
                    "packet_id": packet.packet_id,
                    "provenance_fingerprint": packet.provenance_fingerprint,
                    "rejected_fact_ids": rejected,
                    "selected_fact_ids": selected,
                    "supersedes_resolution_id": supersedes_resolution_id,
                }
            ),
            selected_fact_ids=selected,
            rejected_fact_ids=rejected,
            supersedes_resolution_id=supersedes_resolution_id,
            attributes={
                "compatibility_policy": FACT_RECONCILIATION_COMPATIBILITY_POLICY,
                "contract_version": FACT_RECONCILIATION_SCHEMA_VERSION,
                "packet_id": packet.packet_id,
                "packet_state": packet.state,
            },
        )
    except (TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "human_resolution_invalid")
    return StoreResult.success(event, created=True)


def load_fact_reconciliation_schema() -> dict[str, Any]:
    """Load the bundled public reconciliation-result JSON Schema."""

    path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "schemas"
        / "json"
        / "fact_reconciliation.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class _GroupAssessment:
    kind: str
    conflict_types: frozenset[str]


def _semantic_value_digest(fact: ClinicalFact) -> str:
    attributes = {
        key: fact.attributes[key]
        for key in sorted(_SEMANTIC_ATTRIBUTE_KEYS)
        if key in fact.attributes
    }
    return canonical_digest({"attributes": attributes, "value": fact.value})


def _exact_clinical_digest(fact: ClinicalFact) -> str:
    return canonical_digest(
        {
            "attributes": fact.attributes,
            "confidence": fact.confidence,
            "effective_time": fact.effective_time,
            "encounter_id": fact.encounter_id,
            "fact_type": fact.fact_type,
            "status": fact.status,
            "subject_id": fact.subject_id,
            "unit": fact.unit,
            "value": fact.value,
        }
    )


def _policy_clinical_digest(
    fact: ClinicalFact,
    *,
    status: str,
    unit: str | None,
) -> str:
    return canonical_digest(
        {
            "effective_time": fact.effective_time,
            "encounter_id": fact.encounter_id,
            "fact_type": fact.fact_type,
            "status": status,
            "subject_id": fact.subject_id,
            "unit": unit,
            "value": fact.value,
            "semantic_attributes": {
                key: fact.attributes[key]
                for key in sorted(_SEMANTIC_ATTRIBUTE_KEYS)
                if key in fact.attributes
            },
        }
    )


def _times_equivalent(facts: Sequence[ClinicalFact], tolerance_seconds: float) -> bool:
    values = tuple(fact.effective_time for fact in facts)
    if len({canonical_digest(value) for value in values}) <= 1:
        return True
    instants: list[datetime] = []
    for value in values:
        candidate = value.get("instant") or value.get("start")
        if not isinstance(candidate, str):
            return False
        try:
            instants.append(_parse_timestamp(candidate))
        except ValueError:
            return False
        end = value.get("end")
        if end not in (None, candidate):
            return False
    return max(instants).timestamp() - min(instants).timestamp() <= tolerance_seconds


def _amendment_conflict(members: Sequence[FactReconciliationInput]) -> bool:
    by_id = {item.fact.fact_id: item for item in members}
    children: dict[str, list[FactReconciliationInput]] = defaultdict(list)
    for item in members:
        if item.amendment_of is None:
            continue
        if item.amendment_of not in by_id:
            return True
        children[item.amendment_of].append(item)
    for amendments in children.values():
        if (
            len(amendments) > 1
            and len({_exact_clinical_digest(item.fact) for item in amendments}) > 1
        ):
            return True
    for item in members:
        seen = {item.fact.fact_id}
        parent = item.amendment_of
        while parent is not None:
            if parent in seen:
                return True
            seen.add(parent)
            parent_item = by_id.get(parent)
            parent = parent_item.amendment_of if parent_item is not None else None
    return False


def _valid_amendment_leaf(
    members: Sequence[FactReconciliationInput],
) -> FactReconciliationInput | None:
    if _amendment_conflict(members):
        return None
    amended_ids = {item.amendment_of for item in members if item.amendment_of}
    leaves = [item for item in members if item.fact.fact_id not in amended_ids]
    if len(leaves) != 1 or leaves[0].amendment_of is None:
        return None
    return leaves[0]


def _controlled_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            _CONTROLLED_RE.fullmatch(key) is None
            or _CONTROLLED_RE.fullmatch(item) is None
        ):
            raise FactReconciliationError(f"{name} values must be controlled")
        result[key] = item
    return result


def _text_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = " ".join(str(key).casefold().split())
        normalized_value = " ".join(str(item).casefold().split())
        if not normalized_key or not normalized_value:
            raise FactReconciliationError(f"{name} values must not be blank")
        result[normalized_key] = normalized_value
    return result


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


__all__ = [
    "FACT_CONFLICT_TYPES",
    "FACT_GROUP_KINDS",
    "FACT_RECONCILIATION_COMPATIBILITY_POLICY",
    "FACT_RECONCILIATION_SCHEMA_VERSION",
    "FACT_RECONCILER_VERSION",
    "FactEquivalenceGroup",
    "FactReconciliationError",
    "FactReconciliationInput",
    "FactReconciliationPolicy",
    "FactReconciliationResult",
    "FactReconciler",
    "build_human_resolution",
    "load_fact_reconciliation_schema",
    "persist_fact_reconciliation",
]
