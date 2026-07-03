"""Deterministic n-ary clinical event extraction."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from openmed.clinical.lab_values import lab_value_event_mentions
from openmed.core.decoding import (
    EdgeCardinality,
    SpanEdge,
    SpanGraph,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)

from .frames import (
    ASSISTIVE_EVENT_DISCLAIMER,
    EVENT_FRAME_SCHEMAS,
    EventConflict,
    EventFrame,
    EventType,
    RoleSlot,
)

CLINICAL_EVENT_LEXICON_VERSION = "clinical-events-v1"

MedicationChangeAction = Literal[
    "started",
    "stopped",
    "increased",
    "decreased",
    "held",
    "restarted",
]
LabTrendDirection = Literal["rising", "falling", "stable"]


class ClinicalEventMention(TypedDict, total=False):
    """Already-extracted span candidate used by clinical event role filling."""

    id: str
    label: str
    role: str
    type: str
    start: int
    end: int
    text: str
    value: str
    score: float
    text_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _TriggerCue:
    event_type: EventType
    role: str
    normalized: str
    pattern: re.Pattern[str]
    provenance: str


@dataclass(frozen=True)
class _Trigger:
    event_type: EventType
    role: str
    normalized: str
    start: int
    end: int
    text: str
    provenance: Mapping[str, Any]

    @property
    def node_id(self) -> str:
        return f"trigger:{self.event_type}:{self.start}:{self.end}"


@dataclass(frozen=True)
class _Mention:
    mention_id: str
    label: str
    start: int
    end: int
    value: str
    score: float | None = None
    text_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


_MEDICATION_TRIGGER_CUES: tuple[_TriggerCue, ...] = (
    _TriggerCue(
        "medication_change",
        "action",
        "started",
        re.compile(
            r"\b(?:started|start(?:ing)?|initiated|begin|began|added)\b",
            re.IGNORECASE,
        ),
        "common medication start/initiation cues",
    ),
    _TriggerCue(
        "medication_change",
        "action",
        "stopped",
        re.compile(
            r"\b(?:stopped|stop(?:ping)?|discontinued|discontinue|"
            r"dc(?:'d|ed)?|ceased)\b",
            re.IGNORECASE,
        ),
        "common medication stop/discontinuation cues",
    ),
    _TriggerCue(
        "medication_change",
        "action",
        "increased",
        re.compile(
            r"\b(?:increased|increase|increasing|raised|raise|"
            r"uptitrated|up-titrated|titrated\s+up|escalated)\b",
            re.IGNORECASE,
        ),
        "common medication dose-increase cues",
    ),
    _TriggerCue(
        "medication_change",
        "action",
        "decreased",
        re.compile(
            r"\b(?:decreased|decrease|decreasing|reduced|reduce|"
            r"lowered|lower|downtitrated|down-titrated|titrated\s+down)\b",
            re.IGNORECASE,
        ),
        "common medication dose-decrease cues",
    ),
    _TriggerCue(
        "medication_change",
        "action",
        "held",
        re.compile(r"\b(?:held|hold(?:ing)?|withheld)\b", re.IGNORECASE),
        "common medication hold cues",
    ),
    _TriggerCue(
        "medication_change",
        "action",
        "restarted",
        re.compile(r"\b(?:restarted|restart(?:ing)?|resumed|resume)\b", re.IGNORECASE),
        "common medication restart/resume cues",
    ),
)

_LAB_TREND_TRIGGER_CUES: tuple[_TriggerCue, ...] = (
    _TriggerCue(
        "lab_trend",
        "direction",
        "rising",
        re.compile(
            r"\b(?:rising|rose|uptrending|up-trending|trending\s+up|"
            r"increased|increasing)\b",
            re.IGNORECASE,
        ),
        "common laboratory rising-trend cues",
    ),
    _TriggerCue(
        "lab_trend",
        "direction",
        "falling",
        re.compile(
            r"\b(?:falling|fell|downtrending|down-trending|trending\s+down|"
            r"decreased|decreasing|dropping|dropped)\b",
            re.IGNORECASE,
        ),
        "common laboratory falling-trend cues",
    ),
    _TriggerCue(
        "lab_trend",
        "direction",
        "stable",
        re.compile(r"\b(?:stable|unchanged|flat|plateaued)\b", re.IGNORECASE),
        "common laboratory stable-trend cues",
    ),
)

_TIME_ANCHOR_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|tonight|this\s+morning|this\s+evening|"
    r"overnight|on\s+\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"over(?:\s+the)?\s+(?:past\s+)?\d+\s+(?:hours?|days?|weeks?)|"
    r"past\s+\d+\s+(?:hours?|days?|weeks?)|"
    r"last\s+\d+\s+(?:hours?|days?|weeks?)|"
    r"since\s+[A-Z][a-z]+(?:\s+\d{1,2})?)\b",
    re.IGNORECASE,
)

_LABEL_ALIASES: Mapping[str, str] = {
    "drug": "drug",
    "med": "drug",
    "medication": "drug",
    "medication_name": "drug",
    "medicine": "drug",
    "dose": "dose",
    "dosage": "dose",
    "old_dose": "old_dose",
    "prior_dose": "old_dose",
    "previous_dose": "old_dose",
    "new_dose": "new_dose",
    "target_dose": "new_dose",
    "lab": "analyte",
    "lab_name": "analyte",
    "test": "analyte",
    "analyte": "analyte",
    "lab_value": "lab_value",
    "value": "lab_value",
    "result": "lab_value",
    "magnitude": "magnitude",
    "delta": "magnitude",
    "time": "time_anchor",
    "date": "time_anchor",
    "time_anchor": "time_anchor",
    "time_window": "time_anchor",
}

_CONFLICT_ACTION_PAIRS = {
    frozenset(("increased", "held")),
    frozenset(("increased", "stopped")),
    frozenset(("decreased", "held")),
    frozenset(("decreased", "stopped")),
    frozenset(("started", "held")),
    frozenset(("started", "stopped")),
    frozenset(("restarted", "held")),
    frozenset(("restarted", "stopped")),
    frozenset(("increased", "decreased")),
}


def extract_medication_change_events(
    text: str,
    mentions: Sequence[ClinicalEventMention | Mapping[str, Any]] | None = None,
    *,
    max_distance: int = 160,
    include_detected_time_anchors: bool = True,
) -> list[EventFrame]:
    """Extract medication-change event frames from already-detected spans.

    Args:
        text: Source clinical note text.
        mentions: Candidate spans from upstream extraction. Medication spans
            should use labels such as ``drug`` or ``medication``; dose spans may
            use ``old_dose``, ``new_dose``, or generic ``dose``; time spans may
            use ``time``, ``date``, or ``time_window``.
        max_distance: Maximum character gap for linking a trigger to a role.
        include_detected_time_anchors: Whether to add deterministic time-anchor
            mentions for plain expressions such as ``today`` or ``over 48
            hours``.

    Returns:
        Filled event frames with provenance offsets and an assistive disclaimer.
    """

    source_text = _validate_text(text)
    prepared_mentions = _prepare_mentions(
        source_text,
        mentions,
        include_detected_time_anchors=include_detected_time_anchors,
    )
    frames = [
        frame
        for trigger in _find_triggers(source_text, _MEDICATION_TRIGGER_CUES)
        if (
            frame := _build_event_frame(
                source_text,
                trigger,
                prepared_mentions,
                max_distance=max_distance,
            )
        )
        is not None
    ]
    return _surface_medication_conflicts(frames)


def extract_lab_trend_events(
    text: str,
    mentions: Sequence[ClinicalEventMention | Mapping[str, Any]] | None = None,
    *,
    lab_value_graph: SpanGraph | None = None,
    max_distance: int = 180,
    include_detected_time_anchors: bool = True,
) -> list[EventFrame]:
    """Extract lab-trend event frames from already-detected spans.

    Args:
        text: Source clinical note text.
        mentions: Candidate spans from upstream extraction. Laboratory analyte
            spans should use labels such as ``analyte`` or ``lab_name``; result
            spans may use ``lab_value``, ``value``, ``magnitude``, or ``delta``.
        lab_value_graph: Optional output from
            :func:`openmed.clinical.lab_values.link_lab_value_attributes`.
        max_distance: Maximum character gap for linking a trigger to a role.
        include_detected_time_anchors: Whether to add deterministic time-window
            mentions for plain temporal expressions.

    Returns:
        Filled event frames with provenance offsets and an assistive disclaimer.
    """

    source_text = _validate_text(text)
    raw_mentions = list(mentions or ())
    if lab_value_graph is not None:
        raw_mentions.extend(lab_value_event_mentions(lab_value_graph))
    prepared_mentions = _prepare_mentions(
        source_text,
        raw_mentions,
        include_detected_time_anchors=include_detected_time_anchors,
    )
    return [
        frame
        for trigger in _find_triggers(source_text, _LAB_TREND_TRIGGER_CUES)
        if (
            frame := _build_event_frame(
                source_text,
                trigger,
                prepared_mentions,
                max_distance=max_distance,
            )
        )
        is not None
    ]


def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text


def _find_triggers(text: str, cues: Sequence[_TriggerCue]) -> tuple[_Trigger, ...]:
    triggers: list[_Trigger] = []
    for cue in cues:
        for match in cue.pattern.finditer(text):
            triggers.append(
                _Trigger(
                    event_type=cue.event_type,
                    role=cue.role,
                    normalized=cue.normalized,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    provenance={
                        "lexicon_version": CLINICAL_EVENT_LEXICON_VERSION,
                        "cue": cue.normalized,
                        "cue_source": cue.provenance,
                    },
                )
            )
    return tuple(sorted(triggers, key=lambda item: (item.start, item.end, item.role)))


def _prepare_mentions(
    text: str,
    mentions: Sequence[ClinicalEventMention | Mapping[str, Any]] | None,
    *,
    include_detected_time_anchors: bool,
) -> tuple[_Mention, ...]:
    prepared = [
        _coerce_mention(text, raw_mention, index)
        for index, raw_mention in enumerate(mentions or ())
    ]
    if include_detected_time_anchors:
        prepared.extend(_detected_time_anchor_mentions(text, prepared))
    return tuple(sorted(prepared, key=lambda item: (item.start, item.end, item.label)))


def _coerce_mention(
    text: str,
    raw_mention: ClinicalEventMention | Mapping[str, Any],
    index: int,
) -> _Mention:
    if not isinstance(raw_mention, Mapping):
        raise TypeError("clinical event mentions must be mappings")
    raw_label = (
        raw_mention.get("label") or raw_mention.get("role") or raw_mention.get("type")
    )
    label = _canonical_label(raw_label)
    try:
        start = int(raw_mention["start"])
        end = int(raw_mention["end"])
    except KeyError as exc:
        raise KeyError("clinical event mentions require start and end offsets") from exc
    if start < 0 or end < start or end > len(text):
        raise ValueError("clinical event mention offsets are outside the source text")

    raw_value = raw_mention.get("value") or raw_mention.get("text")
    value = str(raw_value) if raw_value is not None else text[start:end]
    raw_score = raw_mention.get("score")
    score = _finite_float(raw_score) if raw_score is not None else None
    text_hash = raw_mention.get("text_hash")
    metadata = raw_mention.get("metadata")
    source_id = raw_mention.get("id") or f"{label}:{start}:{end}:{index}"
    return _Mention(
        mention_id=str(source_id),
        label=label,
        start=start,
        end=end,
        value=value,
        score=score,
        text_hash=str(text_hash) if text_hash is not None else None,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _detected_time_anchor_mentions(
    text: str,
    existing_mentions: Sequence[_Mention],
) -> list[_Mention]:
    existing_time_spans = [
        (mention.start, mention.end)
        for mention in existing_mentions
        if mention.label == "time_anchor"
    ]
    mentions: list[_Mention] = []
    for index, match in enumerate(_TIME_ANCHOR_RE.finditer(text)):
        if any(
            _spans_overlap(match.start(), match.end(), start, end)
            for start, end in existing_time_spans
        ):
            continue
        mentions.append(
            _Mention(
                mention_id=f"time_anchor:{match.start()}:{match.end()}:{index}",
                label="time_anchor",
                start=match.start(),
                end=match.end(),
                value=match.group(0),
                score=1.0,
                metadata={
                    "source": "clinical_event_time_anchor_lexicon",
                    "lexicon_version": CLINICAL_EVENT_LEXICON_VERSION,
                },
            )
        )
    return mentions


def _build_event_frame(
    text: str,
    trigger: _Trigger,
    mentions: Sequence[_Mention],
    *,
    max_distance: int,
) -> EventFrame | None:
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    trigger_slot = _trigger_slot(trigger)
    graph = _decode_role_graph(text, trigger, mentions, max_distance=max_distance)
    mention_by_id = {mention.mention_id: mention for mention in mentions}
    roles: dict[str, list[RoleSlot]] = {trigger.role: [trigger_slot]}
    for edge in graph.edges:
        mention = mention_by_id[edge.tail]
        roles.setdefault(edge.label, []).append(_role_slot_from_edge(edge, mention))

    frame = EventFrame(
        frame_id=f"{trigger.event_type}:{trigger.start}:{trigger.end}",
        event_type=trigger.event_type,
        roles={role: tuple(slots) for role, slots in roles.items()},
        provenance={
            "disclaimer": ASSISTIVE_EVENT_DISCLAIMER,
            "lexicon_version": CLINICAL_EVENT_LEXICON_VERSION,
            "trigger": trigger_slot.to_dict(),
            "role_graph": graph.explain().to_dict(),
        },
    )
    if not frame.required_roles_filled():
        return None
    return frame


def _decode_role_graph(
    text: str,
    trigger: _Trigger,
    mentions: Sequence[_Mention],
    *,
    max_distance: int,
) -> SpanGraph:
    trigger_label = f"{trigger.event_type}_trigger"
    trigger_node = SpanNode(
        node_id=trigger.node_id,
        start=trigger.start,
        end=trigger.end,
        label=trigger_label,
        score=1.0,
        metadata={
            "role": trigger.role,
            "normalized": trigger.normalized,
            "lexicon_version": CLINICAL_EVENT_LEXICON_VERSION,
        },
    )
    nodes = [
        trigger_node,
        *(
            SpanNode(
                node_id=mention.mention_id,
                start=mention.start,
                end=mention.end,
                label=mention.label,
                score=mention.score,
                text_hash=mention.text_hash,
                metadata=mention.metadata,
            )
            for mention in mentions
        ),
    ]
    schema = EVENT_FRAME_SCHEMAS[trigger.event_type]
    type_compatibility: dict[str, tuple[tuple[str, str], ...]] = {}
    cardinality: dict[str, EdgeCardinality] = {}
    candidate_edges: list[SpanEdge] = []

    for spec in schema.roles:
        if spec.name == trigger.role:
            continue
        accepted_labels = tuple(_canonical_label(label) for label in spec.source_labels)
        type_compatibility[spec.name] = tuple(
            (trigger_label, label) for label in accepted_labels
        )
        if spec.max_count is not None:
            cardinality[spec.name] = EdgeCardinality(
                max_outgoing_per_head=spec.max_count,
            )
        for mention in mentions:
            if mention.label not in accepted_labels:
                continue
            score = _role_candidate_score(
                text,
                trigger,
                mention,
                role=spec.name,
                max_distance=max_distance,
            )
            if score is None:
                continue
            candidate_edges.append(
                SpanEdge(
                    head=trigger.node_id,
                    tail=mention.mention_id,
                    label=spec.name,
                    score=score,
                    metadata={
                        "role": spec.name,
                        "source": "clinical_event_role_linker",
                    },
                )
            )

    return decode_span_graph(
        nodes,
        candidate_edges,
        constraints=SpanGraphConstraints(
            type_compatibility=type_compatibility,
            cardinality=cardinality,
        ),
        min_edge_score=0.01,
    )


def _role_candidate_score(
    text: str,
    trigger: _Trigger,
    mention: _Mention,
    *,
    role: str,
    max_distance: int,
) -> float | None:
    distance = _span_gap(trigger.start, trigger.end, mention.start, mention.end)
    if distance > max_distance:
        return None

    if role in {"old_dose", "new_dose"}:
        dose_score = _dose_role_score(text, trigger, mention, role=role)
        if dose_score is None:
            return None
    else:
        dose_score = 0.0

    proximity = 1.0 if max_distance == 0 else (1.0 - (distance / (max_distance + 1)))
    mention_score = mention.score if mention.score is not None else 1.0
    score = (0.55 * mention_score) + (0.35 * proximity)
    if _same_sentence(text, trigger.start, trigger.end, mention.start, mention.end):
        score += 0.08
    score += dose_score

    if role in {"time", "time_window"}:
        score += 0.04
    return min(score, 1.5)


def _dose_role_score(
    text: str,
    trigger: _Trigger,
    mention: _Mention,
    *,
    role: str,
) -> float | None:
    if mention.label == role:
        return 0.28
    if mention.label != "dose":
        return None

    preceding = text[max(0, mention.start - 24) : mention.start].casefold()
    following_trigger = mention.start >= trigger.end
    before_trigger = mention.end <= trigger.start
    normalized_action = trigger.normalized

    if role == "old_dose":
        if re.search(r"\b(?:from|prior|previous|former|was)\s*$", preceding):
            return 0.20
        if normalized_action in {"increased", "decreased"} and before_trigger:
            return 0.12
        if normalized_action in {"held", "stopped"}:
            return 0.08
        return None

    if re.search(r"\b(?:to|at|now|new)\s*$", preceding):
        return 0.20
    if normalized_action in {"started", "restarted"} and following_trigger:
        return 0.16
    if normalized_action in {"increased", "decreased"} and following_trigger:
        return 0.10
    return None


def _trigger_slot(trigger: _Trigger) -> RoleSlot:
    return RoleSlot(
        role=trigger.role,
        value=trigger.normalized,
        start=trigger.start,
        end=trigger.end,
        label="trigger",
        source_id=trigger.node_id,
        score=1.0,
        provenance={
            **dict(trigger.provenance),
            "surface": trigger.text,
            "source": "clinical_event_trigger_lexicon",
        },
    )


def _role_slot_from_edge(edge: SpanEdge, mention: _Mention) -> RoleSlot:
    return RoleSlot(
        role=edge.label,
        value=mention.value,
        start=mention.start,
        end=mention.end,
        label=mention.label,
        source_id=mention.mention_id,
        score=edge.score,
        provenance={
            "source": "span_relation_graph",
            "edge": edge.to_dict(),
            "mention_score": mention.score,
            "mention_metadata": mention.metadata,
        },
    )


def _surface_medication_conflicts(frames: Sequence[EventFrame]) -> list[EventFrame]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        drug_key = _frame_drug_key(frame)
        if drug_key:
            grouped[drug_key].append(index)

    updated = list(frames)
    for drug_key, indexes in grouped.items():
        if len(indexes) < 2:
            continue
        actions = {
            index: updated[index].role_slots("action")[0]
            for index in indexes
            if updated[index].role_slots("action")
        }
        conflicting_pairs = [
            (left, right)
            for position, left in enumerate(indexes)
            for right in indexes[position + 1 :]
            if _actions_conflict(actions.get(left), actions.get(right))
        ]
        if not conflicting_pairs:
            continue
        conflict_roles = tuple(actions[index] for index in indexes if index in actions)
        for index in indexes:
            conflict = EventConflict(
                conflict_type="contradictory_medication_actions",
                message=(
                    "Medication change actions for the same drug are "
                    "contradictory and require review."
                ),
                roles=conflict_roles,
                provenance={
                    "drug_key": drug_key,
                    "conflicting_frame_ids": [
                        frame_id
                        for pair in conflicting_pairs
                        for frame_id in (
                            updated[pair[0]].frame_id,
                            updated[pair[1]].frame_id,
                        )
                    ],
                },
            )
            updated[index] = updated[index].with_conflict(conflict)
    return updated


def _actions_conflict(left: RoleSlot | None, right: RoleSlot | None) -> bool:
    if left is None or right is None:
        return False
    if left.value == right.value:
        return False
    return frozenset((left.value, right.value)) in _CONFLICT_ACTION_PAIRS


def _frame_drug_key(frame: EventFrame) -> str:
    slots = frame.role_slots("drug")
    if not slots:
        return ""
    return " ".join(slots[0].value.casefold().split())


def _canonical_label(raw_label: object) -> str:
    if not isinstance(raw_label, str):
        raise TypeError("clinical event mention label must be a string")
    normalized = raw_label.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in _LABEL_ALIASES:
        allowed = ", ".join(sorted(_LABEL_ALIASES))
        raise ValueError(
            f"unknown clinical event mention label {raw_label!r}: {allowed}"
        )
    return _LABEL_ALIASES[normalized]


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _span_gap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    if left_end < right_start:
        return right_start - left_end
    if right_end < left_start:
        return left_start - right_end
    return 0


def _same_sentence(
    text: str,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    span_start = min(left_start, right_start)
    span_end = max(left_end, right_end)
    return not re.search(r"[.!?\n]", text[span_start:span_end])


def _spans_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return left_start < right_end and right_start < left_end


__all__ = [
    "CLINICAL_EVENT_LEXICON_VERSION",
    "ClinicalEventMention",
    "LabTrendDirection",
    "MedicationChangeAction",
    "extract_lab_trend_events",
    "extract_medication_change_events",
]
