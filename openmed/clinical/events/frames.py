"""N-ary clinical event frame schemas and scoring helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

CLINICAL_EVENT_SCHEMA_VERSION = 1

EventType = Literal["medication_change", "lab_trend"]

ASSISTIVE_EVENT_DISCLAIMER = (
    "Clinical event extraction is deterministic assistive tooling for review "
    "and is not a clinical decision, treatment recommendation, or substitute "
    "for clinician verification."
)


@dataclass(frozen=True)
class RoleSpec:
    """One typed role slot in an n-ary clinical event frame."""

    name: str
    required: bool
    max_count: int | None = 1
    source_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("role name must be non-empty")
        if self.max_count is not None and self.max_count < 1:
            raise ValueError("role max_count must be positive when provided")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "source_labels",
            tuple(str(label) for label in self.source_labels),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible role specification."""

        return {
            "name": self.name,
            "required": self.required,
            "max_count": self.max_count,
            "source_labels": list(self.source_labels),
        }


@dataclass(frozen=True)
class EventFrameSchema:
    """Schema for one clinical event type."""

    event_type: EventType
    roles: tuple[RoleSpec, ...]

    def __post_init__(self) -> None:
        names = [role.name for role in self.roles]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate roles in {self.event_type!r} schema")

    def role(self, name: str) -> RoleSpec:
        """Return the role spec for ``name``."""

        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"unknown {self.event_type!r} role {name!r}")

    def required_role_names(self) -> tuple[str, ...]:
        """Return required role names in schema order."""

        return tuple(role.name for role in self.roles if role.required)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible schema representation."""

        return {
            "event_type": self.event_type,
            "schema_version": CLINICAL_EVENT_SCHEMA_VERSION,
            "roles": [role.to_dict() for role in self.roles],
        }


MEDICATION_CHANGE_SCHEMA = EventFrameSchema(
    event_type="medication_change",
    roles=(
        RoleSpec("action", required=True, source_labels=("trigger",)),
        RoleSpec(
            "drug",
            required=True,
            source_labels=("drug", "medication", "medication_name"),
        ),
        RoleSpec("old_dose", required=False, source_labels=("old_dose", "dose")),
        RoleSpec("new_dose", required=False, source_labels=("new_dose", "dose")),
        RoleSpec(
            "time",
            required=False,
            source_labels=("time", "date", "time_anchor", "time_window"),
        ),
    ),
)

LAB_TREND_SCHEMA = EventFrameSchema(
    event_type="lab_trend",
    roles=(
        RoleSpec("direction", required=True, source_labels=("trigger",)),
        RoleSpec("analyte", required=True, source_labels=("analyte", "lab_name")),
        RoleSpec(
            "magnitude",
            required=False,
            source_labels=("magnitude", "lab_value", "value", "delta"),
        ),
        RoleSpec(
            "time_window",
            required=False,
            source_labels=("time", "date", "time_anchor", "time_window"),
        ),
    ),
)

EVENT_FRAME_SCHEMAS: Mapping[EventType, EventFrameSchema] = {
    "medication_change": MEDICATION_CHANGE_SCHEMA,
    "lab_trend": LAB_TREND_SCHEMA,
}


@dataclass(frozen=True)
class RoleSlot:
    """One filled role slot with source-span provenance."""

    role: str
    value: str
    start: int
    end: int
    label: str
    source_id: str | None = None
    score: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.label:
            raise ValueError("label must be non-empty")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("role slot offsets must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("role slot offsets must satisfy 0 <= start <= end")
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "value", str(self.value))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "provenance", _plain_mapping(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible role-slot representation."""

        return {
            "role": self.role,
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "source_id": self.source_id,
            "score": self.score,
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


@dataclass(frozen=True)
class EventConflict:
    """Explicit conflict surfaced on an event frame."""

    conflict_type: str
    message: str
    roles: tuple[RoleSlot, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conflict_type:
            raise ValueError("conflict_type must be non-empty")
        if not self.message:
            raise ValueError("message must be non-empty")
        object.__setattr__(self, "conflict_type", str(self.conflict_type))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "provenance", _plain_mapping(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible conflict representation."""

        return {
            "type": self.conflict_type,
            "message": self.message,
            "roles": [role.to_dict() for role in self.roles],
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


@dataclass(frozen=True)
class EventFrame:
    """One filled n-ary clinical event frame."""

    frame_id: str
    event_type: EventType
    roles: Mapping[str, Sequence[RoleSlot]]
    conflicts: tuple[EventConflict, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    disclaimer: str = ASSISTIVE_EVENT_DISCLAIMER
    schema_version: int = CLINICAL_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if self.event_type not in EVENT_FRAME_SCHEMAS:
            raise ValueError(f"unsupported event_type {self.event_type!r}")
        normalized_roles: dict[str, tuple[RoleSlot, ...]] = {}
        for role, slots in self.roles.items():
            normalized_roles[str(role)] = tuple(slots)
        object.__setattr__(self, "frame_id", str(self.frame_id))
        object.__setattr__(self, "roles", normalized_roles)
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(self, "provenance", _plain_mapping(self.provenance))

    @property
    def schema(self) -> EventFrameSchema:
        """Return the schema associated with this frame."""

        return EVENT_FRAME_SCHEMAS[self.event_type]

    def role_slots(self, role: str) -> tuple[RoleSlot, ...]:
        """Return all slots filled for ``role``."""

        return tuple(self.roles.get(role, ()))

    def with_conflict(self, conflict: EventConflict) -> "EventFrame":
        """Return a copy of this frame with one additional explicit conflict."""

        return replace(self, conflicts=(*self.conflicts, conflict))

    def required_roles_filled(self) -> bool:
        """Return whether every required schema role has at least one slot."""

        return all(self.role_slots(role) for role in self.schema.required_role_names())

    def cardinality_violations(self) -> tuple[str, ...]:
        """Return role names whose filled slots exceed schema cardinality."""

        violations: list[str] = []
        for spec in self.schema.roles:
            if spec.max_count is None:
                continue
            if len(self.role_slots(spec.name)) > spec.max_count:
                violations.append(spec.name)
        return tuple(violations)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event frame with provenance and disclaimer."""

        return {
            "frame_id": self.frame_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "disclaimer": self.disclaimer,
            "roles": {
                role: [slot.to_dict() for slot in slots]
                for role, slots in self.roles.items()
            },
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


@dataclass(frozen=True)
class EventEvaluationScore:
    """Evaluation summary for event frame extraction."""

    slot_precision: float
    slot_recall: float
    slot_micro_f1: float
    slot_true_positives: int
    slot_false_positives: int
    slot_false_negatives: int
    whole_frame_exact_match: float
    exact_frame_matches: int
    gold_frame_count: int
    cardinality_violations: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible score summary."""

        return {
            "slot_precision": self.slot_precision,
            "slot_recall": self.slot_recall,
            "slot_micro_f1": self.slot_micro_f1,
            "slot_true_positives": self.slot_true_positives,
            "slot_false_positives": self.slot_false_positives,
            "slot_false_negatives": self.slot_false_negatives,
            "whole_frame_exact_match": self.whole_frame_exact_match,
            "exact_frame_matches": self.exact_frame_matches,
            "gold_frame_count": self.gold_frame_count,
            "cardinality_violations": self.cardinality_violations,
        }


FrameLike = EventFrame | Mapping[str, Any]


def score_event_frame_corpus(
    predicted_by_case: Mapping[str, Sequence[FrameLike]],
    gold_by_case: Mapping[str, Sequence[FrameLike]],
) -> EventEvaluationScore:
    """Score event frames with per-slot micro-F1 and whole-frame exact match."""

    predicted_slots: set[tuple[Any, ...]] = set()
    gold_slots: set[tuple[Any, ...]] = set()
    predicted_exact_frames: set[tuple[Any, ...]] = set()
    gold_exact_frames: list[tuple[Any, ...]] = []
    cardinality_violations = 0

    case_ids = set(predicted_by_case) | set(gold_by_case)
    for case_id in case_ids:
        predicted_frames = tuple(predicted_by_case.get(case_id, ()))
        gold_frames = tuple(gold_by_case.get(case_id, ()))

        for frame in predicted_frames:
            predicted_slots.update(_slot_keys(case_id, frame))
            predicted_exact_frames.add(_required_frame_key(case_id, frame))
            cardinality_violations += _cardinality_violation_count(frame)

        for frame in gold_frames:
            gold_slots.update(_slot_keys(case_id, frame))
            gold_exact_frames.append(_required_frame_key(case_id, frame))

    true_positives = len(predicted_slots & gold_slots)
    false_positives = len(predicted_slots - gold_slots)
    false_negatives = len(gold_slots - predicted_slots)
    precision = _safe_rate(true_positives, true_positives + false_positives)
    recall = _safe_rate(true_positives, true_positives + false_negatives)
    f1 = (
        0.0
        if precision + recall == 0.0
        else (2.0 * precision * recall / (precision + recall))
    )

    exact_matches = sum(1 for key in gold_exact_frames if key in predicted_exact_frames)
    exact_rate = _safe_rate(exact_matches, len(gold_exact_frames))
    return EventEvaluationScore(
        slot_precision=precision,
        slot_recall=recall,
        slot_micro_f1=f1,
        slot_true_positives=true_positives,
        slot_false_positives=false_positives,
        slot_false_negatives=false_negatives,
        whole_frame_exact_match=exact_rate,
        exact_frame_matches=exact_matches,
        gold_frame_count=len(gold_exact_frames),
        cardinality_violations=cardinality_violations,
    )


def _slot_keys(case_id: str, frame: FrameLike) -> set[tuple[Any, ...]]:
    event_type = _frame_event_type(frame)
    keys: set[tuple[Any, ...]] = set()
    for role, slots in _frame_roles(frame).items():
        for slot in slots:
            keys.add((case_id, event_type, role, *_slot_identity(slot)))
    return keys


def _required_frame_key(case_id: str, frame: FrameLike) -> tuple[Any, ...]:
    event_type = _frame_event_type(frame)
    schema = EVENT_FRAME_SCHEMAS[event_type]
    roles = _frame_roles(frame)
    required = []
    for role in schema.required_role_names():
        required.append(
            (
                role,
                tuple(sorted(_slot_identity(slot) for slot in roles.get(role, ()))),
            )
        )
    return (case_id, event_type, tuple(required))


def _cardinality_violation_count(frame: FrameLike) -> int:
    if isinstance(frame, EventFrame):
        return len(frame.cardinality_violations())

    event_type = _frame_event_type(frame)
    roles = _frame_roles(frame)
    count = 0
    for spec in EVENT_FRAME_SCHEMAS[event_type].roles:
        if (
            spec.max_count is not None
            and len(roles.get(spec.name, ())) > spec.max_count
        ):
            count += 1
    return count


def _frame_event_type(frame: FrameLike) -> EventType:
    if isinstance(frame, EventFrame):
        return frame.event_type
    raw_type = frame.get("event_type")
    if raw_type not in EVENT_FRAME_SCHEMAS:
        raise ValueError(f"unsupported event_type {raw_type!r}")
    return raw_type


def _frame_roles(
    frame: FrameLike,
) -> dict[str, tuple[RoleSlot | Mapping[str, Any], ...]]:
    if isinstance(frame, EventFrame):
        return {role: tuple(slots) for role, slots in frame.roles.items()}
    raw_roles = frame.get("roles", {})
    if not isinstance(raw_roles, Mapping):
        raise TypeError("frame roles must be a mapping")
    roles: dict[str, tuple[RoleSlot | Mapping[str, Any], ...]] = {}
    for role, raw_slots in raw_roles.items():
        if isinstance(raw_slots, Mapping):
            roles[str(role)] = (raw_slots,)
        else:
            roles[str(role)] = tuple(raw_slots)
    return roles


def _slot_identity(slot: RoleSlot | Mapping[str, Any]) -> tuple[int, int, str]:
    if isinstance(slot, RoleSlot):
        return (slot.start, slot.end, _normalize_value(slot.value))
    start = int(slot["start"])
    end = int(slot["end"])
    value = slot.get("value") or slot.get("text") or ""
    return (start, end, _normalize_value(str(value)))


def _normalize_value(value: str) -> str:
    return " ".join(value.casefold().split())


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _plain_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in mapping.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, RoleSlot):
        return value.to_dict()
    if isinstance(value, EventConflict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_safe(item) for item in value)
    return copy.deepcopy(value)


__all__ = [
    "ASSISTIVE_EVENT_DISCLAIMER",
    "CLINICAL_EVENT_SCHEMA_VERSION",
    "EVENT_FRAME_SCHEMAS",
    "EventConflict",
    "EventEvaluationScore",
    "EventFrame",
    "EventFrameSchema",
    "EventType",
    "LAB_TREND_SCHEMA",
    "MEDICATION_CHANGE_SCHEMA",
    "RoleSlot",
    "RoleSpec",
    "score_event_frame_corpus",
]
