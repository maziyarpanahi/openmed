"""Point-in-time longitudinal journey materialization over immutable facts."""

from __future__ import annotations

import calendar
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    EvidenceLocator,
    ResolutionEvent,
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.clinical.review_packet_migrations import migrate_review_packet
from openmed.clinical.review_transitions import (
    CLINICAL_REVIEW_STATES,
    ClinicalReviewPacket,
)
from openmed.clinical.timeline_graph import (
    EvidenceLinkedTimelineGraph,
    TimelineGraphEdge,
    TimelineGraphNode,
)
from openmed.structured.store import (
    CanonicalRecordVersion,
    JobMetadata,
    JourneyQueryStore,
    StorePoint,
    StoreResult,
    StoreState,
)

JOURNEY_VIEW_SCHEMA_VERSION = "1.0.0"
JOURNEY_VIEW_COMPATIBILITY_POLICY = "same_major"
JOURNEY_EVENT_TYPES = frozenset(
    {
        "condition",
        "encounter",
        "laboratory",
        "medication",
        "observation",
        "procedure",
        "social_determinant",
    }
)
JOURNEY_STATES = frozenset({"current", "historical", "conflicted"})
JOURNEY_CORRECTION_STATES = frozenset(
    {"none", "amends", "superseded", "amends_and_superseded"}
)

_EVENT_TYPE_ALIASES = {
    "lab": "laboratory",
    "labs": "laboratory",
    "sdoh": "social_determinant",
    "social": "social_determinant",
}
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_CURSOR_RE = re.compile(r"^journeycursor_[A-Za-z0-9_-]{16,128}$")
_PARTIAL_DATE_RE = re.compile(
    r"^(?P<year>[0-9]{4})(?:-(?P<month>0[1-9]|1[0-2])"
    r"(?:-(?P<day>0[1-9]|[12][0-9]|3[01]))?)?$"
)


@dataclass(frozen=True, slots=True)
class JourneySnapshot:
    """Opaque subject-bound pointer to one committed store revision."""

    snapshot_id: str
    subject_id: str
    revision: int
    schema_version: str = JOURNEY_VIEW_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_VIEW_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if _OPAQUE_ID_RE.fullmatch(self.subject_id) is None:
            raise ValueError("snapshot subject_id must be opaque")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("snapshot revision must be positive")
        expected = derived_opaque_id("journeysnapshot", self.subject_id, self.revision)
        if self.snapshot_id != expected:
            raise ValueError("snapshot identifier does not match its revision")
        _require_view_version(self.schema_version, self.compatibility_policy)

    @classmethod
    def at(cls, subject_id: str, revision: int) -> "JourneySnapshot":
        """Build the deterministic snapshot pointer for one revision."""

        return cls(
            snapshot_id=derived_opaque_id("journeysnapshot", subject_id, revision),
            subject_id=subject_id,
            revision=revision,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a public snapshot descriptor."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True, slots=True)
class JourneyQuery:
    """Bounded filters for one point-in-time patient journey query."""

    subject_id: str
    snapshot: JourneySnapshot | None = None
    encounter_ids: tuple[str, ...] = ()
    start_time: str | None = None
    end_time: str | None = None
    event_types: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    review_states: tuple[str, ...] = ()
    journey_states: tuple[str, ...] = ()
    limit: int = 100
    cursor: str | None = None

    def __post_init__(self) -> None:
        if _OPAQUE_ID_RE.fullmatch(self.subject_id) is None:
            raise ValueError("journey subject_id must be opaque")
        if self.snapshot is not None and not isinstance(self.snapshot, JourneySnapshot):
            raise TypeError("journey snapshot must be a JourneySnapshot")
        if self.snapshot is not None and self.snapshot.subject_id != self.subject_id:
            raise ValueError("journey snapshot belongs to another subject")
        encounter_ids = _opaque_ids(self.encounter_ids, "encounter_ids")
        source_ids = _opaque_ids(self.source_ids, "source_ids")
        event_types = tuple(
            sorted({_canonical_event_type(item) for item in self.event_types})
        )
        statuses = _controlled_values(self.statuses, "statuses")
        review_states = tuple(sorted(set(self.review_states)))
        if any(item not in CLINICAL_REVIEW_STATES for item in review_states):
            raise ValueError("journey review state is unsupported")
        journey_states = tuple(sorted(set(self.journey_states)))
        if any(item not in JOURNEY_STATES for item in journey_states):
            raise ValueError("journey state is unsupported")
        start_time = _optional_time_filter(self.start_time, "start_time")
        end_time = _optional_time_filter(self.end_time, "end_time")
        if start_time and end_time:
            if _parse_time_bounds(start_time)[0] > _parse_time_bounds(end_time)[1]:
                raise ValueError("journey start_time must not follow end_time")
        if type(self.limit) is not int or not 1 <= self.limit <= 200:
            raise ValueError("journey limit must be between 1 and 200")
        if self.cursor is not None and _CURSOR_RE.fullmatch(self.cursor) is None:
            raise ValueError("journey cursor must be opaque")
        object.__setattr__(self, "encounter_ids", encounter_ids)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "event_types", event_types)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "review_states", review_states)
        object.__setattr__(self, "journey_states", journey_states)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)

    def fingerprint(self, snapshot: JourneySnapshot) -> str:
        """Return a cursor-binding digest for this query and snapshot."""

        return canonical_digest(self.to_dict(snapshot=snapshot, include_cursor=False))

    def to_dict(
        self,
        *,
        snapshot: JourneySnapshot | None = None,
        include_cursor: bool = True,
    ) -> dict[str, Any]:
        """Return deterministic public filters."""

        active_snapshot = snapshot or self.snapshot
        payload: dict[str, Any] = {
            "encounter_ids": list(self.encounter_ids),
            "end_time": self.end_time,
            "event_types": list(self.event_types),
            "journey_states": list(self.journey_states),
            "limit": self.limit,
            "review_states": list(self.review_states),
            "snapshot": active_snapshot.to_dict() if active_snapshot else None,
            "source_ids": list(self.source_ids),
            "start_time": self.start_time,
            "statuses": list(self.statuses),
            "subject_id": self.subject_id,
        }
        if include_cursor:
            payload["cursor"] = self.cursor
        return payload


@dataclass(frozen=True, slots=True)
class JourneyEvidencePath:
    """Immutable artifact and locator path supporting one fact."""

    locator: EvidenceLocator = field(repr=False)
    artifact: ClinicalArtifact = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.locator, EvidenceLocator):
            raise TypeError("journey evidence locator must be an EvidenceLocator")
        if not isinstance(self.artifact, ClinicalArtifact):
            raise TypeError("journey evidence artifact must be a ClinicalArtifact")
        if self.locator.artifact_id != self.artifact.artifact_id:
            raise ValueError("journey evidence artifact does not match locator")

    def to_dict(self) -> dict[str, Any]:
        """Return source metadata and exact evidence coordinates."""

        return {
            "artifact": self.artifact.to_dict(),
            "locator": self.locator.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class JourneyEvent:
    """One fact-backed event with complete immutable provenance."""

    event_id: str
    position: int
    event_type: str
    journey_state: str
    correction_state: str
    fact: ClinicalFact = field(repr=False)
    evidence_paths: tuple[JourneyEvidencePath, ...] = field(repr=False)
    conflicts: tuple[ConflictSet, ...] = field(default=(), repr=False)
    resolutions: tuple[ResolutionEvent, ...] = field(default=(), repr=False)
    canonical_records: tuple[CanonicalRecordVersion, ...] = field(
        default=(), repr=False
    )
    review_states: tuple[str, ...] = ()
    schema_version: str = JOURNEY_VIEW_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_VIEW_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.fact, ClinicalFact):
            raise TypeError("journey event fact must be a ClinicalFact")
        if _OPAQUE_ID_RE.fullmatch(self.event_id) is None:
            raise ValueError("journey event_id must be opaque")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("journey event position must be non-negative")
        if self.event_type not in JOURNEY_EVENT_TYPES:
            raise ValueError("journey event type is unsupported")
        if self.event_type != _canonical_event_type(self.fact.fact_type):
            raise ValueError("journey event type does not match its fact")
        if self.event_id != derived_opaque_id("journeyevent", self.fact.fact_id):
            raise ValueError("journey event identifier does not match its fact")
        if self.journey_state not in JOURNEY_STATES:
            raise ValueError("journey event state is unsupported")
        if self.correction_state not in JOURNEY_CORRECTION_STATES:
            raise ValueError("journey correction state is unsupported")
        raw_paths = tuple(self.evidence_paths)
        raw_conflicts = tuple(self.conflicts)
        raw_resolutions = tuple(self.resolutions)
        raw_records = tuple(self.canonical_records)
        if any(not isinstance(item, JourneyEvidencePath) for item in raw_paths):
            raise TypeError(
                "journey event evidence_paths must be JourneyEvidencePath values"
            )
        if any(not isinstance(item, ConflictSet) for item in raw_conflicts):
            raise TypeError("journey event conflicts must be ConflictSet values")
        if any(not isinstance(item, ResolutionEvent) for item in raw_resolutions):
            raise TypeError("journey event resolutions must be ResolutionEvent values")
        if any(not isinstance(item, CanonicalRecordVersion) for item in raw_records):
            raise TypeError(
                "journey event canonical_records must be CanonicalRecordVersion values"
            )
        paths = tuple(sorted(raw_paths, key=lambda item: item.locator.locator_id))
        if tuple(item.locator.locator_id for item in paths) != self.fact.evidence_ids:
            raise ValueError("journey event must retain every fact evidence path")
        if any(
            item.subject_id != self.fact.subject_id
            or self.fact.fact_id not in item.fact_ids
            for item in raw_conflicts
        ):
            raise ValueError("journey conflict does not match fact")
        if any(
            item.artifact.subject_id not in {None, self.fact.subject_id}
            for item in paths
        ):
            raise ValueError("journey evidence belongs to another subject")
        conflict_ids = {item.conflict_id for item in raw_conflicts}
        if any(item.conflict_id not in conflict_ids for item in raw_resolutions):
            raise ValueError("journey resolution references an unknown event conflict")
        if any(
            item.record.fact_id != self.fact.fact_id
            or item.record.subject_id != self.fact.subject_id
            for item in raw_records
        ):
            raise ValueError("journey canonical record does not match fact")
        review_states = tuple(sorted(set(self.review_states)))
        if any(item not in CLINICAL_REVIEW_STATES for item in review_states):
            raise ValueError("journey review state is unsupported")
        _require_view_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "evidence_paths", paths)
        object.__setattr__(self, "conflicts", raw_conflicts)
        object.__setattr__(self, "resolutions", raw_resolutions)
        object.__setattr__(self, "canonical_records", raw_records)
        object.__setattr__(self, "review_states", review_states)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Return opaque source identifiers supporting this event."""

        return tuple(sorted({item.artifact.source_id for item in self.evidence_paths}))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return immutable locator identifiers supporting this event."""

        return self.fact.evidence_ids

    def to_dict(self) -> dict[str, Any]:
        """Return the clinical event and its complete derivation path."""

        return {
            "canonical_records": [
                {
                    "record": item.record.to_dict(),
                    "revision": item.revision,
                    "version": item.version,
                }
                for item in self.canonical_records
            ],
            "compatibility_policy": self.compatibility_policy,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "correction_state": self.correction_state,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "evidence_paths": [item.to_dict() for item in self.evidence_paths],
            "fact": self.fact.to_dict(),
            "journey_state": self.journey_state,
            "position": self.position,
            "resolutions": [item.to_dict() for item in self.resolutions],
            "review_states": list(self.review_states),
            "schema_version": self.schema_version,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class JourneyPage:
    """One paginated point-in-time journey and its graph projection."""

    snapshot: JourneySnapshot
    events: tuple[JourneyEvent, ...]
    graph: EvidenceLinkedTimelineGraph
    total_count: int
    next_cursor: str | None
    query_fingerprint: str
    schema_version: str = JOURNEY_VIEW_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_VIEW_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, JourneySnapshot):
            raise TypeError("journey page snapshot must be a JourneySnapshot")
        if not isinstance(self.graph, EvidenceLinkedTimelineGraph):
            raise TypeError("journey page graph must be an EvidenceLinkedTimelineGraph")
        raw_events = tuple(self.events)
        if any(not isinstance(item, JourneyEvent) for item in raw_events):
            raise TypeError("journey page events must be JourneyEvent values")
        if type(self.total_count) is not int or self.total_count < len(self.events):
            raise ValueError("journey total_count is invalid")
        if (
            self.next_cursor is not None
            and _CURSOR_RE.fullmatch(self.next_cursor) is None
        ):
            raise ValueError("journey next_cursor must be opaque")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.query_fingerprint):
            raise ValueError("journey query_fingerprint must be a digest")
        if tuple((item.event_id, item.position) for item in raw_events) != tuple(
            (item.event_id, item.position) for item in self.graph.nodes
        ):
            raise ValueError("journey graph nodes must match page events")
        if tuple(item.position for item in raw_events) != tuple(
            sorted(item.position for item in raw_events)
        ):
            raise ValueError("journey events must preserve deterministic order")
        _require_view_version(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "events", raw_events)

    def to_dict(self) -> dict[str, Any]:
        """Return a public page with event and graph provenance."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "events": [item.to_dict() for item in self.events],
            "graph": self.graph.to_dict(),
            "next_cursor": self.next_cursor,
            "query_fingerprint": self.query_fingerprint,
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.to_dict(),
            "total_count": self.total_count,
        }

    def to_json(self) -> str:
        """Return canonical JSON for the page."""

        return canonical_json(self.to_dict())

    def to_safe_dict(self) -> dict[str, Any]:
        """Return counts and versions suitable for ordinary operational logs."""

        counts: dict[str, int] = defaultdict(int)
        for event in self.events:
            counts[event.journey_state] += 1
        return {
            "compatibility_policy": self.compatibility_policy,
            "event_count": len(self.events),
            "has_next_page": self.next_cursor is not None,
            "schema_version": self.schema_version,
            "state_counts": dict(sorted(counts.items())),
            "total_count": self.total_count,
        }


def query_journey(
    store: JourneyQueryStore,
    query: JourneyQuery,
) -> StoreResult[JourneyPage]:
    """Materialize one evidence-linked journey page at an exact revision."""

    latest_revision = store.latest_revision
    if latest_revision is None:
        return StoreResult.outcome(StoreState.UNKNOWN, "journey_not_found")
    snapshot = query.snapshot or JourneySnapshot.at(query.subject_id, latest_revision)
    if snapshot.revision > latest_revision:
        return StoreResult.outcome(StoreState.UNKNOWN, "journey_snapshot_not_found")
    point = StorePoint(snapshot.revision)

    facts_result = store.list_facts(query.subject_id, as_of=point)
    if not facts_result.ok:
        return _propagate(facts_result, "journey_fact_read_failed")
    facts = facts_result.value or ()
    if not facts:
        return StoreResult.outcome(StoreState.UNKNOWN, "journey_not_found")
    if any(
        _EVENT_TYPE_ALIASES.get(fact.fact_type, fact.fact_type)
        not in JOURNEY_EVENT_TYPES
        for fact in facts
    ):
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "journey_event_type_unsupported"
        )

    conflicts_result = store.list_conflicts(query.subject_id, as_of=point)
    if not conflicts_result.ok:
        return _propagate(conflicts_result, "journey_conflict_read_failed")
    canonicals_result = store.list_canonical_records(query.subject_id, as_of=point)
    if not canonicals_result.ok:
        return _propagate(canonicals_result, "journey_canonical_read_failed")
    jobs_result = store.list_jobs(as_of=point)
    if not jobs_result.ok:
        return _propagate(jobs_result, "journey_review_read_failed")

    conflicts = conflicts_result.value or ()
    resolutions_by_conflict: dict[str, tuple[ResolutionEvent, ...]] = {}
    for conflict in conflicts:
        resolution_result = store.list_resolutions(conflict.conflict_id, as_of=point)
        if not resolution_result.ok:
            return _propagate(resolution_result, "journey_resolution_read_failed")
        resolutions_by_conflict[conflict.conflict_id] = resolution_result.value or ()

    packets_result = _review_packets(
        jobs_result.value or (),
        fact_ids=frozenset(item.fact_id for item in facts),
    )
    if not packets_result.ok:
        return StoreResult.outcome(
            packets_result.state,
            packets_result.code or "journey_review_invalid",
        )
    packets = packets_result.value or ()

    paths_result = _evidence_paths(store, facts, point)
    if not paths_result.ok:
        return StoreResult.outcome(
            paths_result.state,
            paths_result.code or "journey_provenance_incomplete",
        )
    paths_by_fact = paths_result.value or {}

    try:
        events = _materialize_events(
            facts=facts,
            paths_by_fact=paths_by_fact,
            conflicts=conflicts,
            resolutions_by_conflict=resolutions_by_conflict,
            canonicals=canonicals_result.value or (),
            packets=packets,
        )
    except ValueError:
        return StoreResult.outcome(StoreState.FAILURE, "journey_materialization_failed")

    filtered_result = _filter_events(events, query)
    if not filtered_result.ok:
        return StoreResult.outcome(
            filtered_result.state,
            filtered_result.code or "journey_filter_failed",
        )
    filtered = filtered_result.value or ()
    query_fingerprint = query.fingerprint(snapshot)
    start = _cursor_start(filtered, query.cursor, query_fingerprint)
    if start is None:
        return StoreResult.outcome(StoreState.CONFLICT, "journey_cursor_mismatch")
    stop = min(start + query.limit, len(filtered))
    selected = filtered[start:stop]
    page_events = tuple(
        _with_position(event, position=start + offset)
        for offset, event in enumerate(selected)
    )
    next_cursor = None
    if stop < len(filtered) and page_events:
        next_cursor = _event_cursor(
            query_fingerprint,
            page_events[-1].event_id,
            stop - 1,
        )
    try:
        graph = _timeline_graph(page_events)
        page = JourneyPage(
            snapshot=snapshot,
            events=page_events,
            graph=graph,
            total_count=len(filtered),
            next_cursor=next_cursor,
            query_fingerprint=query_fingerprint,
        )
    except ValueError:
        return StoreResult.outcome(StoreState.CONFLICT, "journey_graph_invalid")
    return StoreResult.success(page, revision=snapshot.revision)


def load_journey_view_schema() -> dict[str, Any]:
    """Load the bundled public journey-page JSON Schema."""

    path = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "schemas"
        / "json"
        / "journey_view.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _materialize_events(
    *,
    facts: Sequence[ClinicalFact],
    paths_by_fact: Mapping[str, tuple[JourneyEvidencePath, ...]],
    conflicts: Sequence[ConflictSet],
    resolutions_by_conflict: Mapping[str, tuple[ResolutionEvent, ...]],
    canonicals: Sequence[CanonicalRecordVersion],
    packets: Sequence[ClinicalReviewPacket],
) -> tuple[JourneyEvent, ...]:
    conflicts_by_fact: dict[str, list[ConflictSet]] = defaultdict(list)
    unresolved_facts: set[str] = set()
    rejected_facts: set[str] = set()
    for conflict in conflicts:
        for fact_id in conflict.fact_ids:
            conflicts_by_fact[fact_id].append(conflict)
        leaves = _resolution_leaves(
            resolutions_by_conflict.get(conflict.conflict_id, ())
        )
        if len(leaves) != 1 or leaves[0].action in {"defer", "reopen"}:
            unresolved_facts.update(conflict.fact_ids)
        else:
            rejected_facts.update(leaves[0].rejected_fact_ids)

    canonical_by_fact: dict[str, list[CanonicalRecordVersion]] = defaultdict(list)
    current_fact_ids: set[str] = set()
    for item in canonicals:
        current_fact_ids.add(item.record.fact_id)
        canonical_by_fact[item.record.fact_id].append(item)
    parent_ids = {parent for fact in facts for parent in fact.parent_fact_ids}
    packets_by_conflict: dict[str, list[ClinicalReviewPacket]] = defaultdict(list)
    for packet in packets:
        packets_by_conflict[packet.conflict_id].append(packet)

    materialized: list[JourneyEvent] = []
    for fact in facts:
        event_type = _canonical_event_type(fact.fact_type)
        fact_conflicts = tuple(
            sorted(conflicts_by_fact[fact.fact_id], key=lambda item: item.conflict_id)
        )
        resolutions = tuple(
            sorted(
                (
                    event
                    for conflict in fact_conflicts
                    for event in resolutions_by_conflict.get(conflict.conflict_id, ())
                ),
                key=lambda item: (item.occurred_at, item.resolution_id),
            )
        )
        review_states = tuple(
            sorted(
                {
                    packet.state
                    for conflict in fact_conflicts
                    for packet in packets_by_conflict.get(conflict.conflict_id, ())
                }
            )
        )
        if fact.fact_id in unresolved_facts:
            journey_state = "conflicted"
        elif fact.fact_id in current_fact_ids:
            journey_state = "current"
        elif fact.fact_id in rejected_facts or fact.fact_id in parent_ids:
            journey_state = "historical"
        else:
            journey_state = "current"
        amends = bool(fact.parent_fact_ids)
        superseded = fact.fact_id in parent_ids
        correction_state = (
            "amends_and_superseded"
            if amends and superseded
            else "amends"
            if amends
            else "superseded"
            if superseded
            else "none"
        )
        materialized.append(
            JourneyEvent(
                event_id=derived_opaque_id("journeyevent", fact.fact_id),
                position=0,
                event_type=event_type,
                journey_state=journey_state,
                correction_state=correction_state,
                fact=fact,
                evidence_paths=paths_by_fact[fact.fact_id],
                conflicts=fact_conflicts,
                resolutions=resolutions,
                canonical_records=tuple(
                    sorted(
                        canonical_by_fact[fact.fact_id],
                        key=lambda item: (item.version, item.revision),
                    )
                ),
                review_states=review_states,
            )
        )
    ordered = sorted(materialized, key=_event_sort_key)
    return tuple(
        _with_position(event, position=index) for index, event in enumerate(ordered)
    )


def _evidence_paths(
    store: JourneyQueryStore,
    facts: Sequence[ClinicalFact],
    point: StorePoint,
) -> StoreResult[Mapping[str, tuple[JourneyEvidencePath, ...]]]:
    locators: dict[str, EvidenceLocator] = {}
    artifacts: dict[str, ClinicalArtifact] = {}
    paths: dict[str, tuple[JourneyEvidencePath, ...]] = {}
    for fact in facts:
        fact_paths: list[JourneyEvidencePath] = []
        for evidence_id in fact.evidence_ids:
            locator = locators.get(evidence_id)
            if locator is None:
                locator_result = store.get_evidence(evidence_id, as_of=point)
                if not locator_result.ok or locator_result.value is None:
                    state = (
                        locator_result.state
                        if locator_result.state is StoreState.DENIED
                        else StoreState.PARTIAL
                    )
                    return StoreResult.outcome(state, "journey_provenance_incomplete")
                locator = locator_result.value
                locators[evidence_id] = locator
            artifact = artifacts.get(locator.artifact_id)
            if artifact is None:
                artifact_result = store.get_artifact(locator.artifact_id, as_of=point)
                if not artifact_result.ok or artifact_result.value is None:
                    state = (
                        artifact_result.state
                        if artifact_result.state is StoreState.DENIED
                        else StoreState.PARTIAL
                    )
                    return StoreResult.outcome(state, "journey_provenance_incomplete")
                artifact = artifact_result.value
                artifacts[locator.artifact_id] = artifact
            fact_paths.append(JourneyEvidencePath(locator=locator, artifact=artifact))
        paths[fact.fact_id] = tuple(fact_paths)
    return StoreResult.success(MappingProxyType(paths))


def _review_packets(
    jobs: Sequence[JobMetadata],
    *,
    fact_ids: frozenset[str],
) -> StoreResult[tuple[ClinicalReviewPacket, ...]]:
    packets: list[ClinicalReviewPacket] = []
    for job in jobs:
        payload = job.metadata.get("packet")
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            return StoreResult.outcome(StoreState.FAILURE, "journey_review_invalid")
        packet_fact_ids = payload.get("fact_ids")
        if not isinstance(packet_fact_ids, (tuple, list)):
            return StoreResult.outcome(StoreState.FAILURE, "journey_review_invalid")
        if not fact_ids.intersection(str(item) for item in packet_fact_ids):
            continue
        normalized_payload = json.loads(canonical_json(payload))
        migrated = migrate_review_packet(normalized_payload)
        if not migrated.ok or migrated.value is None:
            return StoreResult.outcome(
                migrated.state,
                migrated.code or "journey_review_invalid",
            )
        packets.append(migrated.value.packet)
    return StoreResult.success(tuple(sorted(packets, key=lambda item: item.packet_id)))


def _filter_events(
    events: Sequence[JourneyEvent], query: JourneyQuery
) -> StoreResult[tuple[JourneyEvent, ...]]:
    start_bound = _parse_time_bounds(query.start_time)[0] if query.start_time else None
    end_bound = _parse_time_bounds(query.end_time)[1] if query.end_time else None
    filtered: list[JourneyEvent] = []
    for event in events:
        if query.encounter_ids and event.fact.encounter_id not in query.encounter_ids:
            continue
        if query.event_types and event.event_type not in query.event_types:
            continue
        if query.statuses and event.fact.status not in query.statuses:
            continue
        if query.source_ids and not set(event.source_ids).intersection(
            query.source_ids
        ):
            continue
        if query.review_states and not set(event.review_states).intersection(
            query.review_states
        ):
            continue
        if query.journey_states and event.journey_state not in query.journey_states:
            continue
        if start_bound is not None or end_bound is not None:
            bounds = _effective_time_bounds(event.fact.effective_time)
            if bounds is None:
                return StoreResult.outcome(StoreState.PARTIAL, "journey_time_unknown")
            event_start, event_end, _ = bounds
            if (
                start_bound is not None
                and event_end is not None
                and event_end < start_bound
            ):
                continue
            if (
                end_bound is not None
                and event_start is not None
                and event_start > end_bound
            ):
                continue
        filtered.append(event)
    return StoreResult.success(tuple(filtered))


def _timeline_graph(events: Sequence[JourneyEvent]) -> EvidenceLinkedTimelineGraph:
    by_fact = {event.fact.fact_id: event for event in events}
    nodes = tuple(
        TimelineGraphNode(
            event_id=event.event_id,
            fact_id=event.fact.fact_id,
            position=event.position,
            event_type=event.event_type,
            journey_state=event.journey_state,
            correction_state=event.correction_state,
            effective_time=event.fact.effective_time,
            evidence_ids=event.fact.evidence_ids,
            derivation_hash=event.fact.derivation_hash,
            assertion_status=_controlled_attribute(event.fact, "assertion"),
            certainty=_controlled_attribute(event.fact, "certainty"),
            experiencer=_controlled_attribute(event.fact, "experiencer"),
        )
        for event in events
    )
    edge_specs: set[tuple[str, str, str, str]] = set()
    for first, second in zip(events, events[1:]):
        if _strictly_before(first.fact.effective_time, second.fact.effective_time):
            edge_specs.add(
                (
                    first.event_id,
                    second.event_id,
                    "chronological_precedes",
                    "effective_time_bounds",
                )
            )
    for event in events:
        for parent_id in event.fact.parent_fact_ids:
            parent = by_fact.get(parent_id)
            if parent is not None:
                edge_specs.add(
                    (
                        parent.event_id,
                        event.event_id,
                        "corrects",
                        "fact_parent_lineage",
                    )
                )
        participants = event.fact.attributes.get("relation_participants", ())
        if isinstance(participants, Sequence) and not isinstance(
            participants, (str, bytes, bytearray)
        ):
            for participant in participants:
                if not isinstance(participant, Mapping):
                    continue
                if participant.get("target_type") != "fact":
                    continue
                target = by_fact.get(str(participant.get("target_id") or ""))
                role = participant.get("role")
                if target is None or not isinstance(role, str):
                    continue
                if _CONTROLLED_RE.fullmatch(role) is None:
                    continue
                edge_specs.add(
                    (
                        event.event_id,
                        target.event_id,
                        f"relation.{role}",
                        "fact_relation_participant",
                    )
                )
    edges = tuple(
        TimelineGraphEdge(
            edge_id=derived_opaque_id("timelineedge", *spec),
            source_event_id=spec[0],
            target_event_id=spec[1],
            relation_type=spec[2],
            ordering_basis=spec[3],
            evidence_ids=tuple(
                sorted(
                    {
                        *next(
                            event.fact.evidence_ids
                            for event in events
                            if event.event_id == spec[0]
                        ),
                        *next(
                            event.fact.evidence_ids
                            for event in events
                            if event.event_id == spec[1]
                        ),
                    }
                )
            ),
            derivation_hash=canonical_digest(
                {
                    "ordering_basis": spec[3],
                    "relation_type": spec[2],
                    "source_event_id": spec[0],
                    "target_event_id": spec[1],
                }
            ),
        )
        for spec in sorted(edge_specs)
    )
    return EvidenceLinkedTimelineGraph(nodes=nodes, edges=edges)


def _event_sort_key(event: JourneyEvent) -> tuple[Any, ...]:
    bounds = _effective_time_bounds(event.fact.effective_time)
    if bounds is None:
        return (1, datetime.max.replace(tzinfo=timezone.utc), "", event.fact.fact_id)
    start, end, precision = bounds
    sort_start = start or datetime.min.replace(tzinfo=timezone.utc)
    sort_end = end or datetime.max.replace(tzinfo=timezone.utc)
    return (0, sort_start, sort_end, precision, event.fact.fact_id)


def _effective_time_bounds(
    value: Mapping[str, Any],
) -> tuple[datetime | None, datetime | None, str] | None:
    if not value:
        return None
    precision = str(value.get("precision") or "unknown")
    instant = value.get("instant")
    if isinstance(instant, str):
        instant_start, instant_end = _parse_time_bounds(instant)
        return instant_start, instant_end, precision
    start_value = value.get("start") or value.get("value")
    end_value = value.get("end")
    if start_value is None and end_value is None:
        return None
    start: datetime | None = (
        _parse_time_bounds(str(start_value))[0] if start_value is not None else None
    )
    end: datetime | None = (
        _parse_time_bounds(str(end_value))[1] if end_value is not None else None
    )
    if end is None and start_value is not None:
        end = _parse_time_bounds(str(start_value))[1]
    return start, end, precision


def _parse_time_bounds(value: str) -> tuple[datetime, datetime]:
    match = _PARTIAL_DATE_RE.fullmatch(value)
    if match is not None:
        year = int(match.group("year"))
        month_text = match.group("month")
        day_text = match.group("day")
        if month_text is None:
            return (
                datetime(year, 1, 1, tzinfo=timezone.utc),
                datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
            )
        month = int(month_text)
        if day_text is None:
            last_day = calendar.monthrange(year, month)[1]
            return (
                datetime(year, month, 1, tzinfo=timezone.utc),
                datetime(
                    year,
                    month,
                    last_day,
                    23,
                    59,
                    59,
                    999999,
                    tzinfo=timezone.utc,
                ),
            )
        day = int(day_text)
        candidate = datetime.combine(
            datetime(year, month, day).date(), time.min, tzinfo=timezone.utc
        )
        return candidate, candidate.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("journey time values must include a timezone")
    normalized = parsed.astimezone(timezone.utc)
    return normalized, normalized


def _strictly_before(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_bounds = _effective_time_bounds(first)
    second_bounds = _effective_time_bounds(second)
    if first_bounds is None or second_bounds is None:
        return False
    first_end = first_bounds[1]
    second_start = second_bounds[0]
    return (
        first_end is not None and second_start is not None and first_end < second_start
    )


def _resolution_leaves(
    events: Sequence[ResolutionEvent],
) -> tuple[ResolutionEvent, ...]:
    superseded = {
        event.supersedes_resolution_id
        for event in events
        if event.supersedes_resolution_id is not None
    }
    return tuple(
        sorted(
            (event for event in events if event.resolution_id not in superseded),
            key=lambda item: (item.occurred_at, item.resolution_id),
        )
    )


def _with_position(event: JourneyEvent, *, position: int) -> JourneyEvent:
    return JourneyEvent(
        event_id=event.event_id,
        position=position,
        event_type=event.event_type,
        journey_state=event.journey_state,
        correction_state=event.correction_state,
        fact=event.fact,
        evidence_paths=event.evidence_paths,
        conflicts=event.conflicts,
        resolutions=event.resolutions,
        canonical_records=event.canonical_records,
        review_states=event.review_states,
    )


def _cursor_start(
    events: Sequence[JourneyEvent], cursor: str | None, fingerprint: str
) -> int | None:
    if cursor is None:
        return 0
    for index, event in enumerate(events):
        if _event_cursor(fingerprint, event.event_id, index) == cursor:
            return index + 1
    return None


def _event_cursor(fingerprint: str, event_id: str, index: int) -> str:
    return derived_opaque_id("journeycursor", fingerprint, event_id, index)


def _canonical_event_type(value: str) -> str:
    candidate = _EVENT_TYPE_ALIASES.get(value, value)
    if candidate not in JOURNEY_EVENT_TYPES:
        raise ValueError("journey event type is unsupported")
    return candidate


def _controlled_values(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if any(_CONTROLLED_RE.fullmatch(item) is None for item in normalized):
        raise ValueError(f"journey {name} must be controlled")
    return normalized


def _opaque_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if any(_OPAQUE_ID_RE.fullmatch(item) is None for item in normalized):
        raise ValueError(f"journey {name} must be opaque")
    return normalized


def _optional_time_filter(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"journey {name} must be an ISO time")
    try:
        _parse_time_bounds(value)
    except (TypeError, ValueError):
        raise ValueError(f"journey {name} must be an ISO time") from None
    return value


def _controlled_attribute(fact: ClinicalFact, name: str) -> str | None:
    value = fact.attributes.get(name)
    if isinstance(value, str) and _CONTROLLED_RE.fullmatch(value):
        return value
    return None


def _propagate(result: StoreResult[Any], fallback: str) -> StoreResult[Any]:
    return StoreResult.outcome(result.state, result.code or fallback)


def _require_view_version(version: str, compatibility_policy: str) -> None:
    if version != JOURNEY_VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported journey view schema version")
    if compatibility_policy != JOURNEY_VIEW_COMPATIBILITY_POLICY:
        raise ValueError("unsupported journey view compatibility policy")


__all__ = [
    "JOURNEY_CORRECTION_STATES",
    "JOURNEY_EVENT_TYPES",
    "JOURNEY_STATES",
    "JOURNEY_VIEW_COMPATIBILITY_POLICY",
    "JOURNEY_VIEW_SCHEMA_VERSION",
    "JourneyEvidencePath",
    "JourneyEvent",
    "JourneyPage",
    "JourneyQuery",
    "JourneySnapshot",
    "load_journey_view_schema",
    "query_journey",
]
