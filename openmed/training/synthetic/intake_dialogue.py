"""Deterministic synthetic patient-intake conversations with turn-level gold.

Conversational extraction and ambient-scribe systems need redistributable
dialogues that contain what real intake visits contain: several speakers,
caregiver speech, interpreted answers, hesitations, self-repairs, later
corrections, uncertainty and recording-consent events. This module writes such
conversations from repository-authored templates and fictional value pools.
Phone numbers use the reserved 555-01XX range and e-mail addresses use the
reserved ``example.org`` domain. No dataset is read and no network call is
made.

Gold labels are attached per turn so a streaming evaluator can consume one turn
at a time. Every label in turn ``t`` is causal: it only refers to turn ``t`` or
earlier turns. A turn carries PHI and clinical spans with turn-local and stream
offsets, assertion status, speaker and subject roles, revision links that name
the earlier span a repair or correction supersedes, discourse events, and the
recording-consent state that governs whether the turn may be captured.
Exported gold, reports and manifests never repeat a span's surface text; spans
are identified by offsets and SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from openmed.core import labels as L
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash

INTAKE_DIALOGUE_SCHEMA_VERSION: Final = "openmed.training.synthetic_intake_dialogue.v1"
INTAKE_DIALOGUE_MANIFEST_SCHEMA_VERSION: Final = (
    "openmed.training.synthetic_intake_dialogue_manifest.v1"
)
SYNTHETIC_INTAKE_DIALOGUE_SOURCE: Final = "openmed.synthetic.intake_dialogue"
SYNTHETIC_INTAKE_DIALOGUE_LICENSE: Final = "Apache-2.0"

SPEAKER_ROLES: Final[tuple[str, ...]] = (
    "patient",
    "caregiver",
    "clinician",
    "interpreter",
)
INTAKE_SCENARIOS: Final[tuple[str, ...]] = ("patient_only", "caregiver", "interpreter")
ASSERTION_STATUSES: Final[tuple[str, ...]] = ("affirmed", "negated", "uncertain")
REVISION_TYPES: Final[tuple[str, ...]] = ("self_repair", "correction", "completion")
TURN_EVENT_TYPES: Final[tuple[str, ...]] = (
    "hesitation",
    "uncertainty",
    "self_repair",
    "correction",
    "interruption",
    "consent",
)
CONSENT_ACTIONS: Final[tuple[str, ...]] = (
    "request",
    "grant",
    "decline",
    "pause",
    "resume",
)
CONSENT_STATES: Final[tuple[str, ...]] = ("pending", "granted", "declined", "paused")
INTAKE_PHI_LABELS: Final[tuple[str, ...]] = (
    L.PERSON,
    L.DATE_OF_BIRTH,
    L.AGE,
    L.PHONE,
    L.EMAIL,
    L.STREET_ADDRESS,
    L.LOCATION,
    L.ID_NUM,
    L.DATE,
)
INTAKE_CLINICAL_LABELS: Final[tuple[str, ...]] = (L.CONDITION, L.MEDICATION)
STREAM_TURN_SEPARATOR: Final = "\n"

_SPAN_KINDS: Final = ("phi", "clinical")
_CONSENT_TRANSITIONS: Final[Mapping[str, tuple[frozenset[str], str]]] = {
    "grant": (frozenset({"pending", "granted"}), "granted"),
    "decline": (frozenset({"pending"}), "declined"),
    "pause": (frozenset({"granted"}), "paused"),
    "resume": (frozenset({"paused"}), "granted"),
}
_EVENT_SPAN_KEYS: Final = ("span_id", "supersedes", "reparandum", "repair")
_IDENTIFIER_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_TURN_CHARS: Final = 2_000
_MAX_TURNS: Final = 200
_GOLD_DETECTOR: Final = "synthetic_intake_gold"
# AGE values are computed against a fixed date so output never depends on the
# wall clock.
_REFERENCE_DATE: Final = date(2026, 3, 2)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class IntakeSpan:
    """One gold span inside a generated intake turn.

    Offsets are turn-local and half-open. The span's surface text is not
    stored; ``text_sha256`` binds the span to the exact generated characters.

    Attributes:
        span_id: Conversation-unique identifier such as ``"t04.s01"``.
        start: Turn-local start offset.
        end: Turn-local end offset (exclusive).
        label: Canonical OpenMed label from :mod:`openmed.core.labels`.
        kind: ``"phi"`` for redaction targets or ``"clinical"`` for intake facts.
        subject_role: Participant the fact is about, which can differ from the
            speaker (a caregiver giving the patient's birth date, or an
            interpreter speaking for the patient in the first person).
        attribute: Intake attribute such as ``"phone"`` or ``"allergy"``.
        assertion: ``"affirmed"``, ``"negated"`` or ``"uncertain"`` as stated
            in this turn.
        revision: How this span revises an earlier span, if it does.
        supersedes: Identifier of the earlier span this span replaces.
        partial: Whether the speaker was cut off before finishing the value.
        needs_adjudication: Whether this mention disagrees with an earlier
            mention by another speaker.
        conflicts_with: Earlier span identifiers this mention disagrees with.
        text_sha256: SHA-256 hex digest of the span's surface text.
    """

    span_id: str
    start: int
    end: int
    label: str
    kind: str
    subject_role: str
    attribute: str
    assertion: str
    text_sha256: str
    revision: str | None = None
    supersedes: str | None = None
    partial: bool = False
    needs_adjudication: bool = False
    conflicts_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id:
            raise ValueError("intake span requires a non-empty span_id")
        if not (_is_int(self.start) and _is_int(self.end)) or not (
            0 <= self.start < self.end
        ):
            raise ValueError("intake span offsets must be ordered integers")
        if self.kind not in _SPAN_KINDS:
            raise ValueError("intake span kind must be 'phi' or 'clinical'")
        allowed = INTAKE_PHI_LABELS if self.kind == "phi" else INTAKE_CLINICAL_LABELS
        if self.label not in allowed:
            raise ValueError("intake span label is not supported for its kind")
        if self.subject_role not in SPEAKER_ROLES:
            raise ValueError("intake span subject_role is not a speaker role")
        if not isinstance(self.attribute, str) or not _IDENTIFIER_RE.match(
            self.attribute
        ):
            raise ValueError("intake span attribute must be a lowercase identifier")
        if self.assertion not in ASSERTION_STATUSES:
            raise ValueError("intake span assertion status is not supported")
        if not isinstance(self.text_sha256, str) or not _SHA256_RE.match(
            self.text_sha256
        ):
            raise ValueError("intake span text_sha256 must be a SHA-256 hex digest")
        if self.revision is not None and self.revision not in REVISION_TYPES:
            raise ValueError("intake span revision type is not supported")
        if (self.revision is None) != (self.supersedes is None):
            raise ValueError("intake span revision and supersedes must be paired")
        if self.supersedes is not None and (
            not isinstance(self.supersedes, str) or not self.supersedes
        ):
            raise ValueError("intake span supersedes must name a span")
        if not isinstance(self.partial, bool):
            raise ValueError("intake span partial must be a boolean")
        conflicts = self.conflicts_with
        if isinstance(conflicts, str) or not isinstance(conflicts, Sequence):
            raise ValueError("intake span conflicts_with must be a sequence")
        conflicts = tuple(conflicts)
        if any(not isinstance(item, str) or not item for item in conflicts):
            raise ValueError("intake span conflicts_with must name spans")
        if self.needs_adjudication is not bool(conflicts):
            raise ValueError("intake span adjudication flag must match its conflicts")
        object.__setattr__(self, "conflicts_with", conflicts)

    @property
    def slot(self) -> str:
        """Return the fact slot, ``"<subject_role>.<attribute>"``."""

        return f"{self.subject_role}.{self.attribute}"

    def to_dict(self, *, stream_offset: int = 0) -> dict[str, Any]:
        """Return a JSON-compatible gold span without its surface text.

        Args:
            stream_offset: Offset of the owning turn in the conversation
                stream, used to add ``stream_start`` and ``stream_end``.

        Returns:
            Span offsets, label, roles, assertion, revision links and digest.
        """

        return {
            "span_id": self.span_id,
            "start": self.start,
            "end": self.end,
            "stream_start": stream_offset + self.start,
            "stream_end": stream_offset + self.end,
            "label": self.label,
            "kind": self.kind,
            "subject_role": self.subject_role,
            "attribute": self.attribute,
            "slot": self.slot,
            "assertion": self.assertion,
            "revision": self.revision,
            "supersedes": self.supersedes,
            "partial": self.partial,
            "needs_adjudication": self.needs_adjudication,
            "conflicts_with": list(self.conflicts_with),
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True)
class IntakeEvent:
    """A discourse or consent event anchored to its cue inside one turn.

    Attributes:
        event_id: Conversation-unique identifier such as ``"t04.e00"``.
        type: One of :data:`TURN_EVENT_TYPES`.
        start: Turn-local start offset of the cue.
        end: Turn-local end offset of the cue (exclusive).
        attributes: Event details. Consent events carry ``action``, ``scope``
            and ``subject_role``; repair and correction events name the
            superseded and replacing span identifiers; uncertainty events name
            the hedged span; interruption events record the position and, for
            the interrupting turn, the interrupted turn index.
    """

    event_id: str
    type: str
    start: int
    end: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("intake event requires a non-empty event_id")
        if self.type not in TURN_EVENT_TYPES:
            raise ValueError("intake event type is not supported")
        if not (_is_int(self.start) and _is_int(self.end)) or not (
            0 <= self.start < self.end
        ):
            raise ValueError("intake event offsets must be ordered integers")
        if not isinstance(self.attributes, Mapping):
            raise ValueError("intake event attributes must be a mapping")
        attributes = dict(self.attributes)
        if self.type == "consent" and (
            attributes.get("action") not in CONSENT_ACTIONS
            or attributes.get("subject_role") not in SPEAKER_ROLES
            or not isinstance(attributes.get("scope"), str)
        ):
            raise ValueError("consent event requires action, scope and subject_role")
        object.__setattr__(self, "attributes", attributes)

    def to_dict(self, *, stream_offset: int = 0) -> dict[str, Any]:
        """Return a JSON-compatible event with turn-local and stream offsets."""

        return {
            "event_id": self.event_id,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "stream_start": stream_offset + self.start,
            "stream_end": stream_offset + self.end,
            "attributes": dict(sorted(self.attributes.items())),
        }


@dataclass(frozen=True)
class IntakeTurn:
    """One generated utterance with its causal turn-level gold labels.

    Attributes:
        index: Zero-based turn position in the conversation.
        speaker_role: One of :data:`SPEAKER_ROLES`.
        act: Scripted dialogue act that produced the turn, for provenance.
        text: Generated synthetic utterance.
        spans: Gold PHI and clinical spans ordered by offset.
        events: Discourse and consent events ordered by offset.
        consent_state: Recording-consent state in effect when the turn starts.
        stream_offset: Offset of ``text`` in the conversation stream.
    """

    index: int
    speaker_role: str
    act: str
    text: str
    spans: tuple[IntakeSpan, ...]
    events: tuple[IntakeEvent, ...]
    consent_state: str
    stream_offset: int

    def __post_init__(self) -> None:
        if not _is_int(self.index) or not 0 <= self.index < _MAX_TURNS:
            raise ValueError("intake turn index is out of range")
        if self.speaker_role not in SPEAKER_ROLES:
            raise ValueError("intake turn speaker_role is not supported")
        if not isinstance(self.act, str) or not _IDENTIFIER_RE.match(self.act):
            raise ValueError("intake turn act must be a lowercase identifier")
        if not isinstance(self.text, str) or not 1 <= len(self.text) <= (
            _MAX_TURN_CHARS
        ):
            raise ValueError("intake turn text must be a bounded non-empty string")
        if STREAM_TURN_SEPARATOR in self.text:
            raise ValueError("intake turn text must not contain the stream separator")
        if self.consent_state not in CONSENT_STATES:
            raise ValueError("intake turn consent_state is not supported")
        if not _is_int(self.stream_offset) or self.stream_offset < 0:
            raise ValueError("intake turn stream_offset must be non-negative")
        for name, item_type in (("spans", IntakeSpan), ("events", IntakeEvent)):
            values = getattr(self, name)
            if isinstance(values, str) or not isinstance(values, Sequence):
                raise ValueError(f"intake turn {name} must be a sequence")
            values = tuple(values)
            if any(not isinstance(value, item_type) for value in values):
                raise ValueError(
                    f"intake turn {name} must contain only {item_type.__name__}"
                )
            if any(value.end > len(self.text) for value in values):
                raise ValueError(f"intake turn {name} must lie inside the turn text")
            object.__setattr__(self, name, values)
        previous_end = 0
        for span in self.spans:
            if span.start < previous_end:
                raise ValueError("intake turn spans must be ordered and disjoint")
            if _sha256(self.text[span.start : span.end]) != span.text_sha256:
                raise ValueError("intake span digest does not match the turn text")
            previous_end = span.end
        if list(self.events) != sorted(
            self.events, key=lambda event: (event.start, event.end)
        ):
            raise ValueError("intake turn events must be ordered by offset")

    @property
    def capture_permitted(self) -> bool:
        """Return whether recording consent covers this turn.

        Turns spoken before consent is given, after it is declined, or while
        recording is paused are not covered, but their PHI spans remain gold
        redaction targets.
        """

        return self.consent_state == "granted"

    @property
    def phi_spans(self) -> tuple[IntakeSpan, ...]:
        """Return the PHI redaction-target spans in offset order."""

        return tuple(span for span in self.spans if span.kind == "phi")

    @property
    def clinical_spans(self) -> tuple[IntakeSpan, ...]:
        """Return the clinical intake-fact spans in offset order."""

        return tuple(span for span in self.spans if span.kind == "clinical")

    def fact_updates(self) -> dict[str, str]:
        """Return ``slot -> span_id`` for the latest mention of each slot here."""

        return {span.slot: span.span_id for span in self.spans}

    def to_dict(self) -> dict[str, Any]:
        """Return the turn input fields and its causal gold labels.

        ``content`` follows the one-turn-per-line chat-log convention used by
        :mod:`openmed.multimodal.chatlog_jsonl`; every label lives under
        ``gold`` and none of it repeats span surface text.
        """

        offset = self.stream_offset
        return {
            "turn_index": self.index,
            "speaker_role": self.speaker_role,
            "content": self.text,
            "stream_offset": offset,
            "gold": {
                "act": self.act,
                "consent_state": self.consent_state,
                "capture_permitted": self.capture_permitted,
                "phi_spans": [
                    span.to_dict(stream_offset=offset) for span in self.phi_spans
                ],
                "clinical_spans": [
                    span.to_dict(stream_offset=offset) for span in self.clinical_spans
                ],
                "events": [
                    event.to_dict(stream_offset=offset) for event in self.events
                ],
                "fact_updates": dict(sorted(self.fact_updates().items())),
            },
        }


@dataclass(frozen=True)
class IntakeConversation:
    """A synthetic intake conversation with causal, turn-level gold labels.

    Attributes:
        conversation_id: Stable identifier derived from seed and index.
        scenario: One of :data:`INTAKE_SCENARIOS`.
        seed: Generation seed.
        index: Position of the conversation in its generated batch.
        turns: Ordered turns; turn ``i`` has ``index == i``.
    """

    conversation_id: str
    scenario: str
    seed: int
    index: int
    turns: tuple[IntakeTurn, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise ValueError("intake conversation requires a conversation_id")
        if self.scenario not in INTAKE_SCENARIOS:
            raise ValueError("intake conversation scenario is not supported")
        for name in ("seed", "index"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"intake conversation {name} must be non-negative")
        if isinstance(self.turns, str) or not isinstance(self.turns, Sequence):
            raise ValueError("intake conversation turns must be a sequence")
        turns = tuple(self.turns)
        if not 1 <= len(turns) <= _MAX_TURNS or any(
            not isinstance(turn, IntakeTurn) or turn.index != position
            for position, turn in enumerate(turns)
        ):
            raise ValueError("intake conversation requires ordered intake turns")
        _validate_conversation_links(turns)
        object.__setattr__(self, "turns", turns)

    @property
    def participants(self) -> tuple[str, ...]:
        """Return the speaker roles present, in :data:`SPEAKER_ROLES` order."""

        present = {turn.speaker_role for turn in self.turns}
        return tuple(role for role in SPEAKER_ROLES if role in present)

    @property
    def stream_text(self) -> str:
        """Return the concatenated stream that ``stream_*`` offsets index."""

        return "".join(self.iter_stream_chunks())

    def iter_stream_chunks(self) -> Iterator[str]:
        """Yield each turn followed by :data:`STREAM_TURN_SEPARATOR`.

        Feeding these chunks to an incremental de-identifier in order yields
        global offsets that match each gold span's ``stream_start`` and
        ``stream_end``.
        """

        for turn in self.turns:
            yield turn.text + STREAM_TURN_SEPARATOR

    def fact_timeline(self) -> tuple[dict[str, str], ...]:
        """Return the ``slot -> span_id`` fact state after each turn.

        The latest mention of a slot is its current value; explicit revision
        links say why a mention replaced an earlier one. Replaying each turn's
        ``gold.fact_updates`` reproduces this timeline incrementally.
        """

        state: dict[str, str] = {}
        timeline: list[dict[str, str]] = []
        for turn in self.turns:
            state.update(turn.fact_updates())
            timeline.append(dict(sorted(state.items())))
        return tuple(timeline)

    def final_facts(self) -> dict[str, dict[str, Any]]:
        """Return the end-of-conversation value reference for every slot."""

        located = _span_locations(self.turns)
        final: dict[str, dict[str, Any]] = {}
        for slot, span_id in self.fact_timeline()[-1].items():
            turn, span = located[span_id]
            final[slot] = {
                "span_id": span_id,
                "turn_index": turn.index,
                "label": span.label,
                "assertion": span.assertion,
                "partial": span.partial,
                "needs_adjudication": span.needs_adjudication,
            }
        return final

    def supersessions(self) -> tuple[dict[str, Any], ...]:
        """Return every explicit repair, correction and completion link."""

        located = _span_locations(self.turns)
        return tuple(
            {
                "span_id": span.supersedes,
                "turn_index": located[span.supersedes][0].index,
                "superseded_by": span.span_id,
                "superseded_in_turn": turn.index,
                "revision": span.revision,
                "slot": span.slot,
            }
            for turn in self.turns
            for span in turn.spans
            if span.supersedes is not None
        )

    def adjudication_items(self) -> tuple[dict[str, Any], ...]:
        """Return mentions where speakers disagree and a reviewer must decide."""

        return tuple(
            {
                "span_id": span.span_id,
                "turn_index": turn.index,
                "conflicts_with": list(span.conflicts_with),
                "slot": span.slot,
            }
            for turn in self.turns
            for span in turn.spans
            if span.needs_adjudication
        )

    def provenance(self) -> dict[str, Any]:
        """Return synthetic lineage metadata shared by every exported record."""

        return {
            "synthetic": True,
            "contains_real_phi": False,
            "restricted_data": False,
            "source": SYNTHETIC_INTAKE_DIALOGUE_SOURCE,
            "license": SYNTHETIC_INTAKE_DIALOGUE_LICENSE,
            "seed": self.seed,
            "conversation_index": self.index,
            "scenario": self.scenario,
        }

    def iter_stream_records(self) -> Iterator[dict[str, Any]]:
        """Yield one self-contained JSON-compatible record per turn, in order.

        Each record carries the turn input (``content``), its causal gold
        labels and lineage, so a streaming evaluator can score turn ``t``
        before it reads turn ``t + 1``.
        """

        provenance = self.provenance()
        for turn in self.turns:
            yield {
                "schema_version": INTAKE_DIALOGUE_SCHEMA_VERSION,
                "record_type": "intake_turn",
                "conversation_id": self.conversation_id,
                **turn.to_dict(),
                "provenance": dict(provenance),
            }

    def to_dict(self) -> dict[str, Any]:
        """Return the whole conversation with its end-of-conversation views."""

        return {
            "schema_version": INTAKE_DIALOGUE_SCHEMA_VERSION,
            "record_type": "intake_conversation",
            "conversation_id": self.conversation_id,
            "scenario": self.scenario,
            "participants": list(self.participants),
            "turns": [turn.to_dict() for turn in self.turns],
            "final_facts": self.final_facts(),
            "supersessions": list(self.supersessions()),
            "adjudication": list(self.adjudication_items()),
            "provenance": self.provenance(),
        }

    def gold_openmed_spans(self, *, secret: str | bytes) -> tuple[OpenMedSpan, ...]:
        """Return PHI gold spans as stream-offset :class:`OpenMedSpan` records.

        The offsets index :attr:`stream_text`, which is the coordinate system
        a streaming de-identifier reports after consuming
        :meth:`iter_stream_chunks`. Surface text is represented only by the
        keyed ``text_hash``. Superseded and partial values are included because
        every spoken identifier remains a redaction target.

        Args:
            secret: Non-empty HMAC key used for ``text_hash``.

        Returns:
            One span per PHI gold span, in stream order.
        """

        stream = self.stream_text
        spans: list[OpenMedSpan] = []
        for turn in self.turns:
            for span in turn.phi_spans:
                start = turn.stream_offset + span.start
                end = turn.stream_offset + span.end
                spans.append(
                    OpenMedSpan(
                        doc_id=self.conversation_id,
                        start=start,
                        end=end,
                        text_hash=hmac_text_hash(stream[start:end], secret),
                        entity_type=span.label,
                        canonical_label=span.label,
                        detector=_GOLD_DETECTOR,
                        metadata={
                            "span_id": span.span_id,
                            "turn_index": turn.index,
                            "speaker_role": turn.speaker_role,
                            "slot": span.slot,
                            "assertion": span.assertion,
                            "capture_permitted": turn.capture_permitted,
                            "synthetic": True,
                        },
                    )
                )
        return tuple(spans)

    def report(self) -> dict[str, Any]:
        """Return counts and a content digest, never generated values."""

        counts = _conversation_counts((self,))
        return {
            "conversation_id": self.conversation_id,
            "scenario": self.scenario,
            "synthetic": True,
            "contains_real_phi": False,
            **counts,
            "content_sha256": _sha256(_canonical_json(self.to_dict())),
        }


def generate_intake_conversations(
    count: int,
    *,
    seed: int = 0,
    scenarios: Sequence[str] | None = None,
) -> tuple[IntakeConversation, ...]:
    """Generate deterministic synthetic patient-intake conversations offline.

    Conversation ``i`` uses scenario ``scenarios[i % len(scenarios)]``. Every
    conversation contains a recording-consent request and response, at least
    one hesitation, an explicit uncertainty cue, a within-turn self-repair and
    a later turn that corrects an earlier fact. Interruptions, consent
    declines, private recording pauses and misheard read-backs rotate across
    consecutive conversations. The same arguments always produce identical
    text, offsets and labels; nothing is read from disk or the network.

    Args:
        count: Number of conversations to generate.
        seed: Non-negative generation seed.
        scenarios: Scenario rotation drawn from :data:`INTAKE_SCENARIOS`.
            Defaults to all scenarios in their declared order.

    Returns:
        Conversations with causal turn-level PHI, assertion, revision,
        consent and provenance labels.

    Raises:
        ValueError: If an argument is malformed or names an unknown scenario.
    """

    if not _is_int(count) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    plan = _validated_scenarios(scenarios)
    return tuple(
        _generate_conversation(seed=seed, index=index, scenario=plan[index % len(plan)])
        for index in range(count)
    )


def build_intake_dialogue_manifest(
    conversations: Sequence[IntakeConversation],
) -> dict[str, Any]:
    """Summarize a generated batch for lineage without any generated values.

    Args:
        conversations: Conversations produced by
            :func:`generate_intake_conversations`.

    Returns:
        Schema versions, source, license, seeds, aggregate counts and a
        dataset digest over the canonical conversation records.

    Raises:
        ValueError: If ``conversations`` is not a sequence of conversations.
    """

    if isinstance(conversations, str) or not isinstance(conversations, Sequence):
        raise ValueError("conversations must be a sequence of intake conversations")
    batch = tuple(conversations)
    if any(not isinstance(item, IntakeConversation) for item in batch):
        raise ValueError("conversations must be a sequence of intake conversations")
    digest = hashlib.sha256()
    for conversation in batch:
        digest.update(_canonical_json(conversation.to_dict()).encode("utf-8"))
        digest.update(b"\n")
    return {
        "schema_version": INTAKE_DIALOGUE_MANIFEST_SCHEMA_VERSION,
        "record_schema_version": INTAKE_DIALOGUE_SCHEMA_VERSION,
        "dataset_source": SYNTHETIC_INTAKE_DIALOGUE_SOURCE,
        "license": SYNTHETIC_INTAKE_DIALOGUE_LICENSE,
        "synthetic": True,
        "contains_real_phi": False,
        "restricted_data": False,
        "network_accessed": False,
        "seeds": sorted({conversation.seed for conversation in batch}),
        "conversation_count": len(batch),
        "scenario_counts": dict(
            sorted(Counter(item.scenario for item in batch).items())
        ),
        **_conversation_counts(batch),
        "dataset_sha256": digest.hexdigest(),
    }


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _conversation_counts(
    conversations: Sequence[IntakeConversation],
) -> dict[str, Any]:
    turns = [turn for conversation in conversations for turn in conversation.turns]
    spans = [span for turn in turns for span in turn.spans]
    events = [event for turn in turns for event in turn.events]

    def counted(values: Any) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    return {
        "turn_count": len(turns),
        "role_counts": counted(turn.speaker_role for turn in turns),
        "phi_label_counts": counted(s.label for s in spans if s.kind == "phi"),
        "clinical_label_counts": counted(
            s.label for s in spans if s.kind == "clinical"
        ),
        "assertion_counts": counted(span.assertion for span in spans),
        "revision_counts": counted(s.revision for s in spans if s.revision),
        "event_counts": counted(event.type for event in events),
        "consent_action_counts": counted(
            event.attributes["action"] for event in events if event.type == "consent"
        ),
        "capture_suppressed_turn_count": sum(
            not turn.capture_permitted for turn in turns
        ),
        "partial_span_count": sum(span.partial for span in spans),
        "adjudication_span_count": sum(span.needs_adjudication for span in spans),
    }


def _span_locations(
    turns: Sequence[IntakeTurn],
) -> dict[str, tuple[IntakeTurn, IntakeSpan]]:
    return {span.span_id: (turn, span) for turn in turns for span in turn.spans}


def _validate_conversation_links(turns: Sequence[IntakeTurn]) -> None:
    """Check offsets, identifiers, causal links and consent transitions."""

    expected_offset = 0
    seen_spans: dict[str, IntakeSpan] = {}
    seen_events: set[str] = set()
    state = "pending"
    for turn in turns:
        if turn.stream_offset != expected_offset:
            raise ValueError("intake turn stream offsets are not contiguous")
        expected_offset += len(turn.text) + len(STREAM_TURN_SEPARATOR)
        if turn.consent_state != state:
            raise ValueError("intake turn consent_state does not follow its events")
        for span in turn.spans:
            if span.span_id in seen_spans:
                raise ValueError("intake span identifiers must be unique")
            references = list(span.conflicts_with)
            if span.supersedes is not None:
                references.append(span.supersedes)
            for reference in references:
                earlier = seen_spans.get(reference)
                if earlier is None or earlier.slot != span.slot:
                    raise ValueError(
                        "intake span links must point to an earlier span of "
                        "the same slot"
                    )
            seen_spans[span.span_id] = span
        for event in turn.events:
            if event.event_id in seen_events:
                raise ValueError("intake event identifiers must be unique")
            seen_events.add(event.event_id)
            # Event links may name spans of this or an earlier turn only.
            if any(
                not isinstance(event.attributes[key], str)
                or event.attributes[key] not in seen_spans
                for key in _EVENT_SPAN_KEYS
                if key in event.attributes
            ):
                raise ValueError("intake event links must name known spans")
            interrupted = event.attributes.get("interrupts_turn")
            if interrupted is not None and (
                not _is_int(interrupted) or not 0 <= interrupted < turn.index
            ):
                raise ValueError("intake interruption must name an earlier turn")
            if event.type != "consent":
                continue
            transition = _CONSENT_TRANSITIONS.get(event.attributes["action"])
            if transition is None:
                continue
            allowed, next_state = transition
            if state not in allowed:
                raise ValueError("intake consent event is not a valid transition")
            state = next_state


def _validated_scenarios(scenarios: Sequence[str] | None) -> tuple[str, ...]:
    if scenarios is None:
        return INTAKE_SCENARIOS
    if isinstance(scenarios, str) or not isinstance(scenarios, Sequence):
        raise ValueError("scenarios must be a sequence of scenario names")
    plan = tuple(scenarios)
    if not plan or any(
        not isinstance(name, str) or name not in INTAKE_SCENARIOS for name in plan
    ):
        raise ValueError(f"scenarios must be drawn from {list(INTAKE_SCENARIOS)!r}")
    return plan


# ---------------------------------------------------------------------------
# Repository-authored fictional value pools and template helpers.
# ---------------------------------------------------------------------------

_GIVEN_NAMES: Final = tuple(
    "Mara Tomas Ines Kofi Aiyana Rafael Noor Elias Priya Jonah Sigrid Mateo "
    "Yuki Amara Lucan Odile Dalia Teodor Wren Ilse".split()
)
_FAMILY_NAMES: Final = tuple(
    "Varga Okafor Lindqvist Ferreira Haddad Novak Castellanos Abernathy Takeda "
    "Moreau Kowalczyk Adeyemi Halloran Sandoval Iversen Petrakis".split()
)
_INTERPRETER_LANGUAGES: Final = (
    "Spanish",
    "Portuguese",
    "Vietnamese",
    "Arabic",
    "Somali",
    "Haitian Creole",
    "Tagalog",
    "Polish",
)
_AREA_CODES: Final = ("206", "312", "415", "503", "617", "720", "802", "919")
_STREET_NAMES: Final = (
    "Alder",
    "Juniper",
    "Harbor View",
    "Millbrook",
    "Cedar Hollow",
    "Larkspur",
    "Quarry",
    "Wren Hill",
)
_STREET_SUFFIXES: Final = ("Lane", "Street", "Avenue", "Court", "Road", "Drive")
_CITIES: Final = (
    "Riverton",
    "Fairhaven",
    "Brookside",
    "Linden Falls",
    "Westmere",
    "Oak Harbor",
)
_SYMPTOMS: Final = (
    "a dry cough",
    "headaches",
    "dizzy spells",
    "chest tightness",
    "shortness of breath",
    "nausea",
    "lower back pain",
    "a low fever",
)
_ALLERGENS: Final = ("penicillin", "sulfa drugs", "codeine", "amoxicillin", "ibuprofen")
_MEDICATIONS: Final = tuple(
    "metformin lisinopril atorvastatin levothyroxine amlodipine omeprazole".split()
)
_RELATIONSHIPS: Final = ("daughter", "son", "partner", "neighbor")
_MONTHS: Final = tuple(
    "January February March April May June July August September October "
    "November December".split()
)
_DIGIT_WORDS: Final = tuple("zero one two three four five six seven eight nine".split())
_SMALL_NUMBER_WORDS: Final = _DIGIT_WORDS + tuple(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
    "nineteen".split()
)
_TENS_WORDS: Final = ("", "") + tuple(
    "twenty thirty forty fifty sixty seventy eighty ninety".split()
)
_ORDINAL_WORDS: Final = ("",) + tuple(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh "
    "twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
    "nineteenth twentieth".split()
)
_INITIAL_HESITATIONS: Final = ("um", "uh", "hmm", "well, um")
_MEDIAL_HESITATIONS: Final = ("um", "uh", "let me think")


@dataclass(frozen=True)
class _Pronouns:
    subject: str
    object: str
    possessive: str
    be_contracted: str
    have: str
    lives: str


_PRONOUN_SETS: Final = (
    _Pronouns("she", "her", "her", "she's", "has", "lives"),
    _Pronouns("he", "him", "his", "he's", "has", "lives"),
    _Pronouns("they", "them", "their", "they're", "have", "live"),
)


@dataclass(frozen=True)
class _Cast:
    """Fictional participants and values for one conversation."""

    patient_first: str
    patient_last: str
    pronouns: _Pronouns
    dob: date
    wrong_dob: date
    age: int
    phone: str
    new_phone: str
    work_phone: str
    phone_style: str
    dob_style: str
    address_number: int
    street: str
    unit: str | None
    city: str
    email: str
    mrn: str
    symptom: str
    second_symptom: str
    onset: date
    allergen: str
    allergy_assertion: str
    medication: str
    new_medication: str
    clinician_first: str
    clinician_last: str
    clinician_is_doctor: bool
    caregiver_first: str
    caregiver_last: str
    relationship: str
    caregiver_phone: str
    caregiver_wrong_phone: str
    caregiver_email: str
    interpreter_first: str
    interpreter_id: str
    interpreter_language: str


def _two_digit_words(value: int) -> str:
    if value < 20:
        return _SMALL_NUMBER_WORDS[value]
    tens, ones = divmod(value, 10)
    return _TENS_WORDS[tens] + (f"-{_DIGIT_WORDS[ones]}" if ones else "")


def _ordinal_word(day: int) -> str:
    if day <= 20:
        return _ORDINAL_WORDS[day]
    return f"twenty-{_ORDINAL_WORDS[day - 20]}"


def _ordinal_suffix(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _format_date(value: date, style: str) -> str:
    month = _MONTHS[value.month - 1]
    if style == "numeric":
        return f"{value.month:02d}/{value.day:02d}/{value.year}"
    if style == "spoken":
        century, rest = divmod(value.year, 100)
        year = f"{_two_digit_words(century)} {_two_digit_words(rest)}"
        return f"{month} {_ordinal_word(value.day)}, {year}"
    return f"{month} {value.day}{_ordinal_suffix(value.day)}, {value.year}"


def _format_month_day(value: date) -> str:
    return f"{_MONTHS[value.month - 1]} {value.day}{_ordinal_suffix(value.day)}"


def _format_phone(digits: str, style: str) -> str:
    area, exchange, line = digits[:3], digits[3:6], digits[6:]
    if style == "spoken":
        return ", ".join(
            " ".join(_DIGIT_WORDS[int(digit)] for digit in group)
            for group in (area, exchange, line)
        )
    if style == "dashed":
        return f"{area}-{exchange}-{line}"
    return f"({area}) {exchange}-{line}"


def _shifted_day(value: date) -> date:
    """Return a plausible mistaken date: the fourth heard as the fourteenth."""

    day = value.day + 10 if value.day <= 18 else value.day - 10
    return value.replace(day=day)


def _swapped_phone(digits: str) -> str:
    """Return a plausible transposition of the last two digits."""

    if digits[-1] != digits[-2]:
        return digits[:-2] + digits[-1] + digits[-2]
    return digits[:-1] + str((int(digits[-1]) + 1) % 10)


def _age_on_reference(dob: date) -> int:
    before_birthday = (_REFERENCE_DATE.month, _REFERENCE_DATE.day) < (
        dob.month,
        dob.day,
    )
    return _REFERENCE_DATE.year - dob.year - int(before_birthday)


def _derived_seed(seed: int, index: int) -> int:
    payload = f"{SYNTHETIC_INTAKE_DIALOGUE_SOURCE}:{seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_cast(rng: random.Random, scenario: str) -> _Cast:
    given = rng.sample(_GIVEN_NAMES, 4)
    family = rng.sample(_FAMILY_NAMES, 3)
    first_year, last_year = (1938, 1952) if scenario == "caregiver" else (1946, 1999)
    dob = date(
        rng.randint(first_year, last_year), rng.randint(1, 12), rng.randint(1, 28)
    )
    area = rng.choice(_AREA_CODES)
    lines = rng.sample(range(100), 4)
    phones = [f"{area}55501{line:02d}" for line in lines]
    symptom, second_symptom = rng.sample(_SYMPTOMS, 2)
    medication, new_medication = rng.sample(_MEDICATIONS, 2)
    patient_last = family[0]
    caregiver_last = patient_last if rng.random() < 0.6 else family[2]
    caregiver_first = given[2]
    return _Cast(
        patient_first=given[0],
        patient_last=patient_last,
        pronouns=rng.choice(_PRONOUN_SETS),
        dob=dob,
        wrong_dob=_shifted_day(dob),
        age=_age_on_reference(dob),
        phone=phones[0],
        new_phone=phones[1],
        work_phone=phones[2],
        phone_style=rng.choice(("parenthesized", "dashed", "spoken")),
        dob_style=rng.choice(("long", "numeric", "spoken")),
        address_number=rng.randint(12, 9800),
        street=f"{rng.choice(_STREET_NAMES)} {rng.choice(_STREET_SUFFIXES)}",
        unit=(
            f"{rng.randint(1, 9)}{rng.choice('ABCD')}" if rng.random() < 0.4 else None
        ),
        city=rng.choice(_CITIES),
        email=(
            f"{given[0].lower()}.{patient_last.lower()}{rng.randint(10, 99)}"
            "@example.org"
        ),
        mrn=str(rng.randrange(10_000_000, 100_000_000)),
        symptom=symptom,
        second_symptom=second_symptom,
        onset=date(_REFERENCE_DATE.year, rng.randint(1, 2), rng.randint(1, 28)),
        allergen=rng.choice(_ALLERGENS),
        allergy_assertion=rng.choice(ASSERTION_STATUSES),
        medication=medication,
        new_medication=new_medication,
        clinician_first=given[1],
        clinician_last=family[1],
        clinician_is_doctor=rng.random() < 0.6,
        caregiver_first=caregiver_first,
        caregiver_last=caregiver_last,
        relationship=rng.choice(_RELATIONSHIPS),
        caregiver_phone=phones[3],
        caregiver_wrong_phone=_swapped_phone(phones[3]),
        caregiver_email=(
            f"{caregiver_first.lower()}.{caregiver_last.lower()}"
            f"{rng.randint(10, 99)}@example.org"
        ),
        interpreter_first=given[3],
        interpreter_id=str(rng.randrange(10_000, 100_000)),
        interpreter_language=rng.choice(_INTERPRETER_LANGUAGES),
    )


class _TurnDraft:
    """Accumulate one turn's text while recording exact gold offsets."""

    def __init__(self, index: int, speaker_role: str, act: str) -> None:
        self.index = index
        self.speaker_role = speaker_role
        self.act = act
        self.text = ""
        self.spans: list[IntakeSpan] = []
        self._events: list[tuple[str, int, int, dict[str, Any]]] = []

    def say(self, text: str) -> tuple[int, int]:
        if not self.text and text:
            text = text[0].upper() + text[1:]
        start = len(self.text)
        self.text += text
        return start, len(self.text)

    def value(
        self,
        text: str,
        label: str,
        subject_role: str,
        attribute: str,
        *,
        assertion: str = "affirmed",
        revision: str | None = None,
        supersedes: str | None = None,
        partial: bool = False,
        conflicts_with: tuple[str, ...] = (),
    ) -> str:
        start, end = self.say(text)
        span_id = f"t{self.index:02d}.s{len(self.spans):02d}"
        self.spans.append(
            IntakeSpan(
                span_id=span_id,
                start=start,
                end=end,
                label=label,
                kind="clinical" if label in INTAKE_CLINICAL_LABELS else "phi",
                subject_role=subject_role,
                attribute=attribute,
                assertion=assertion,
                text_sha256=_sha256(self.text[start:end]),
                revision=revision,
                supersedes=supersedes,
                partial=partial,
                needs_adjudication=bool(conflicts_with),
                conflicts_with=conflicts_with,
            )
        )
        return span_id

    def event(self, event_type: str, start: int, end: int, **attributes: Any) -> None:
        self._events.append((event_type, start, end, attributes))

    def cue(self, text: str, event_type: str, **attributes: Any) -> None:
        start, end = self.say(text)
        self.event(event_type, start, end, **attributes)

    def hesitate(self, rng: random.Random) -> None:
        pool = _MEDIAL_HESITATIONS if self.text else _INITIAL_HESITATIONS
        self.cue(rng.choice(pool), "hesitation")
        self.say(", ")

    def build(self, *, consent_state: str, stream_offset: int) -> IntakeTurn:
        ordered = sorted(self._events, key=lambda item: (item[1], item[2]))
        events = tuple(
            IntakeEvent(
                event_id=f"t{self.index:02d}.e{position:02d}",
                type=event_type,
                start=start,
                end=end,
                attributes=attributes,
            )
            for position, (event_type, start, end, attributes) in enumerate(ordered)
        )
        return IntakeTurn(
            index=self.index,
            speaker_role=self.speaker_role,
            act=self.act,
            text=self.text,
            spans=tuple(self.spans),
            events=events,
            consent_state=consent_state,
            stream_offset=stream_offset,
        )


class _IntakeScript:
    """Compose scripted dialogue acts into one causal, labelled conversation."""

    def __init__(
        self, rng: random.Random, cast: _Cast, scenario: str, index: int
    ) -> None:
        self.rng = rng
        self.cast = cast
        self.scenario = scenario
        self.index = index
        # The interpreter speaks for the patient in the first person.
        self.voice = "interpreter" if scenario == "interpreter" else "patient"
        self.turns: list[IntakeTurn] = []
        self.consent_state = "pending"
        self._offset = 0
        self._latest: dict[str, str] = {}

    def turn(self, speaker_role: str, act: str) -> _TurnDraft:
        return _TurnDraft(len(self.turns), speaker_role, act)

    def commit(self, draft: _TurnDraft) -> IntakeTurn:
        turn = draft.build(consent_state=self.consent_state, stream_offset=self._offset)
        self.turns.append(turn)
        self._offset += len(turn.text) + len(STREAM_TURN_SEPARATOR)
        for event in turn.events:
            if event.type == "consent":
                transition = _CONSENT_TRANSITIONS.get(event.attributes["action"])
                if transition is not None:
                    self.consent_state = transition[1]
        self._latest.update(turn.fact_updates())
        return turn

    def ask(self, text: str, act: str) -> None:
        draft = self.turn("clinician", act)
        draft.say(text)
        self.commit(draft)

    def run(self) -> tuple[IntakeTurn, ...]:
        decline = self.index % 5 == 3
        self.opening()
        self.consent(decline=decline)
        if self.scenario == "caregiver":
            self.caregiver_introduction()
        self.identity()
        self.readback(misheard=self.index % 2 == 1)
        self.phone()
        self.address(interrupted=self.index % 2 == 0)
        self.email()
        self.complaint()
        self.allergies()
        self.medications()
        if not decline and self.index % 2 == 1:
            self.private_pause()
        if self.scenario != "caregiver":
            self.late_phone_correction()
        self.closing()
        return tuple(self.turns)

    def opening(self) -> None:
        cast = self.cast
        draft = self.turn("clinician", "greeting")
        draft.say(self.rng.choice(("Good morning. ", "Hi, thanks for coming in. ")))
        if cast.clinician_is_doctor:
            draft.say("I'm Dr. ")
            draft.value(
                f"{cast.clinician_first} {cast.clinician_last}",
                L.PERSON,
                "clinician",
                "name",
            )
            draft.say(", and I'll be doing your intake today.")
        else:
            draft.say("My name is ")
            draft.value(cast.clinician_first, L.PERSON, "clinician", "name")
            draft.say(", and I'm the nurse doing your intake today.")
        self.commit(draft)
        if self.scenario == "interpreter":
            draft = self.turn("interpreter", "interpreter_introduction")
            draft.say("Hello, this is ")
            draft.value(cast.interpreter_first, L.PERSON, "interpreter", "name")
            draft.say(f", your {cast.interpreter_language} interpreter, ID number ")
            draft.value(cast.interpreter_id, L.ID_NUM, "interpreter", "badge_number")
            draft.say(". I will interpret everything in the first person.")
            self.commit(draft)

    def consent(self, *, decline: bool) -> None:
        draft = self.turn("clinician", "consent_request")
        draft.say("Before we start, ")
        draft.cue(
            "is it okay if I use a recording assistant to help with my notes?",
            "consent",
            action="request",
            scope="recording",
            subject_role="patient",
        )
        draft.say(" You can ask me to pause it at any time.")
        self.commit(draft)

        draft = self.turn(self.voice, "consent_response")
        if decline:
            draft.cue(
                "I'd rather you didn't record",
                "consent",
                action="decline",
                scope="recording",
                subject_role="patient",
            )
            draft.say(", if that's alright.")
        else:
            if self.rng.random() < 0.5:
                draft.hesitate(self.rng)
            draft.cue(
                self.rng.choice(("sure, that's fine", "yes, you can record")),
                "consent",
                action="grant",
                scope="recording",
                subject_role="patient",
            )
            draft.say(".")
        self.commit(draft)

        if decline:
            self.ask("No problem at all. I'll type my notes instead.", "consent_ack")
        elif self.scenario == "caregiver":
            draft = self.turn("caregiver", "consent_response")
            draft.cue(
                "that's fine with me too",
                "consent",
                action="grant",
                scope="recording",
                subject_role="caregiver",
            )
            draft.say(".")
            self.commit(draft)

    def caregiver_introduction(self) -> None:
        cast = self.cast
        draft = self.turn("caregiver", "caregiver_introduction")
        draft.say("And I'm ")
        draft.value(
            f"{cast.caregiver_first} {cast.caregiver_last}",
            L.PERSON,
            "caregiver",
            "name",
        )
        draft.say(
            f", {cast.pronouns.possessive} {cast.relationship}. I usually come "
            f"with {cast.pronouns.object} to appointments."
        )
        self.commit(draft)

    def identity(self) -> None:
        cast = self.cast
        full_name = f"{cast.patient_first} {cast.patient_last}"
        if self.scenario == "caregiver":
            self.ask(
                "Could one of you give me the full name and date of birth?",
                "identity_prompt",
            )
            pronouns = cast.pronouns
            draft = self.turn("caregiver", "identity_answer")
            draft.hesitate(self.rng)
            draft.say(f"{pronouns.possessive} name is ")
            draft.value(full_name, L.PERSON, "patient", "name")
            draft.say(f". {pronouns.be_contracted.capitalize()} ")
            draft.value(str(cast.age), L.AGE, "patient", "age")
            draft.say(f" now, and {pronouns.possessive} birthday is, ")
            cue_start, cue_end = draft.say("I think")
            draft.say(", ")
            guess = draft.value(
                _format_date(cast.wrong_dob, "long"),
                L.DATE_OF_BIRTH,
                "patient",
                "date_of_birth",
                assertion="uncertain",
            )
            draft.event("uncertainty", cue_start, cue_end, span_id=guess)
            draft.say(".")
            self.commit(draft)

            draft = self.turn("patient", "identity_correction")
            cue_start, cue_end = draft.say("no")
            draft.say(", ")
            draft.value(cast.caregiver_first, L.PERSON, "caregiver", "given_name")
            draft.say(", it's ")
            fixed = draft.value(
                _format_date(cast.dob, cast.dob_style),
                L.DATE_OF_BIRTH,
                "patient",
                "date_of_birth",
                revision="correction",
                supersedes=guess,
            )
            draft.event(
                "correction", cue_start, cue_end, supersedes=guess, repair=fixed
            )
            draft.say(".")
            self.commit(draft)
            return

        self.ask("Can you tell me your full name and date of birth?", "identity_prompt")
        if self.scenario == "interpreter":
            draft = self.turn("patient", "identity_answer")
            draft.value(full_name, L.PERSON, "patient", "name")
            draft.say(".")
            self.commit(draft)
            draft = self.turn("interpreter", "identity_answer")
            draft.say("My name is ")
            draft.value(full_name, L.PERSON, "patient", "name")
            draft.say(", and my date of birth is ")
            editing_term = "interpreter correction"
        else:
            draft = self.turn("patient", "identity_answer")
            if self.rng.random() < 0.5:
                draft.hesitate(self.rng)
            draft.value(full_name, L.PERSON, "patient", "name")
            draft.say(". My date of birth is ")
            editing_term = "no, sorry"
        wrong = draft.value(
            _format_date(cast.wrong_dob, cast.dob_style),
            L.DATE_OF_BIRTH,
            "patient",
            "date_of_birth",
        )
        draft.say(", ")
        cue_start, cue_end = draft.say(editing_term)
        draft.say(", ")
        right = draft.value(
            _format_date(cast.dob, cast.dob_style),
            L.DATE_OF_BIRTH,
            "patient",
            "date_of_birth",
            revision="self_repair",
            supersedes=wrong,
        )
        draft.event("self_repair", cue_start, cue_end, reparandum=wrong, repair=right)
        draft.say(".")
        self.commit(draft)

    def readback(self, *, misheard: bool) -> None:
        cast = self.cast
        draft = self.turn("clinician", "readback")
        draft.say("Thank you. I have your record number as ")
        draft.value(cast.mrn, L.ID_NUM, "patient", "record_number")
        draft.say(" and your date of birth as ")
        # The clinician asserts what the chart says; a misheard value is
        # corrected by the next turn rather than flagged in advance.
        shown = cast.wrong_dob if misheard else cast.dob
        readback = draft.value(
            _format_date(shown, "long"), L.DATE_OF_BIRTH, "patient", "date_of_birth"
        )
        draft.say(", is that right?")
        self.commit(draft)

        draft = self.turn(self.voice, "readback_response")
        if misheard:
            cue_start, cue_end = draft.say("no")
            draft.say(", it's ")
            fixed = draft.value(
                _format_date(cast.dob, "long"),
                L.DATE_OF_BIRTH,
                "patient",
                "date_of_birth",
                revision="correction",
                supersedes=readback,
            )
            draft.event(
                "correction", cue_start, cue_end, supersedes=readback, repair=fixed
            )
            draft.say(".")
        else:
            draft.say("Yes, that's right.")
        self.commit(draft)

    def phone(self) -> None:
        cast = self.cast
        self.ask("What's the best phone number to reach you?", "phone_prompt")
        if self.scenario == "caregiver":
            draft = self.turn("caregiver", "phone_answer")
            draft.say(f"{cast.pronouns.possessive} cell is ")
            draft.value(
                _format_phone(cast.phone, cast.phone_style), L.PHONE, "patient", "phone"
            )
            draft.say(". You can also reach me at ")
            wrong = draft.value(
                _format_phone(cast.caregiver_wrong_phone, "dashed"),
                L.PHONE,
                "caregiver",
                "phone",
            )
            draft.say(", ")
            cue_start, cue_end = draft.say("sorry, I mean")
            draft.say(" ")
            right = draft.value(
                _format_phone(cast.caregiver_phone, "dashed"),
                L.PHONE,
                "caregiver",
                "phone",
                revision="self_repair",
                supersedes=wrong,
            )
            draft.event(
                "self_repair", cue_start, cue_end, reparandum=wrong, repair=right
            )
            draft.say(f". I'm {cast.pronouns.possessive} emergency contact.")
            self.commit(draft)
            return
        draft = self.turn(self.voice, "phone_answer")
        draft.hesitate(self.rng)
        draft.say("my number is " if self.voice == "interpreter" else "it's ")
        draft.value(
            _format_phone(cast.phone, cast.phone_style), L.PHONE, "patient", "phone"
        )
        draft.say(".")
        self.commit(draft)

    def address(self, *, interrupted: bool) -> None:
        cast = self.cast
        self.ask("And what's your home address?", "address_prompt")
        speaker = "caregiver" if self.scenario == "caregiver" else self.voice
        if speaker == "caregiver":
            lead = f"{cast.pronouns.subject} {cast.pronouns.lives} at "
        else:
            lead = "I live at "
        full = f"{cast.address_number} {cast.street}"
        if cast.unit is not None:
            full += f", Apartment {cast.unit}"
        supersedes = None
        if interrupted:
            draft = self.turn(speaker, "address_answer")
            draft.say(lead)
            supersedes = draft.value(
                f"{cast.address_number} {cast.street.split()[0]}",
                L.STREET_ADDRESS,
                "patient",
                "address",
                partial=True,
            )
            cut_start, cut_end = draft.say("--")
            draft.event("interruption", cut_start, cut_end, position="interrupted")
            interrupted_turn = self.commit(draft)

            draft = self.turn("clinician", "interruption")
            draft.cue(
                "sorry, one second",
                "interruption",
                position="interrupting",
                interrupts_turn=interrupted_turn.index,
            )
            draft.say(" -- I missed that. Go ahead.")
            self.commit(draft)
            draft = self.turn(speaker, "address_answer")
        else:
            draft = self.turn(speaker, "address_answer")
            draft.say(lead)
        draft.value(
            full,
            L.STREET_ADDRESS,
            "patient",
            "address",
            revision="completion" if supersedes else None,
            supersedes=supersedes,
        )
        draft.say(", in ")
        draft.value(cast.city, L.LOCATION, "patient", "city")
        draft.say(".")
        self.commit(draft)

    def email(self) -> None:
        cast = self.cast
        self.ask("Is there an email address for the patient portal?", "email_prompt")
        if self.scenario == "caregiver":
            draft = self.turn("caregiver", "email_answer")
            draft.say("You can use mine: ")
            draft.value(cast.caregiver_email, L.EMAIL, "caregiver", "email")
        else:
            draft = self.turn(self.voice, "email_answer")
            draft.say("Yes, it's ")
            draft.value(cast.email, L.EMAIL, "patient", "email")
        draft.say(".")
        self.commit(draft)

    def complaint(self) -> None:
        cast = self.cast
        self.ask("What brings you in today?", "complaint_prompt")
        draft = self.turn(self.voice, "complaint_answer")
        draft.say("I've had ")
        draft.value(cast.symptom, L.CONDITION, "patient", "chief_complaint")
        if self.scenario == "caregiver":
            draft.say(" since ")
            draft.value(_format_month_day(cast.onset), L.DATE, "patient", "onset")
            draft.say(".")
            self.commit(draft)

            pronouns = cast.pronouns
            draft = self.turn("caregiver", "complaint_addition")
            draft.say(f"{pronouns.subject} {pronouns.have} also had ")
            reported = draft.value(
                cast.second_symptom, L.CONDITION, "patient", "additional_symptom"
            )
            draft.say(" at night.")
            self.commit(draft)

            draft = self.turn("patient", "complaint_disagreement")
            draft.say("No, I haven't had ")
            draft.value(
                cast.second_symptom,
                L.CONDITION,
                "patient",
                "additional_symptom",
                assertion="negated",
                conflicts_with=(reported,),
            )
            draft.say(".")
            self.commit(draft)
            return
        draft.say(" since, ")
        cue_start, cue_end = draft.say("I think")
        draft.say(", ")
        onset = draft.value(
            _format_month_day(cast.onset),
            L.DATE,
            "patient",
            "onset",
            assertion="uncertain",
        )
        draft.event("uncertainty", cue_start, cue_end, span_id=onset)
        draft.say(".")
        self.commit(draft)

    def allergies(self) -> None:
        cast = self.cast
        self.ask("Do you have any allergies to medications?", "allergy_prompt")
        draft = self.turn(self.voice, "allergy_answer")
        if cast.allergy_assertion == "uncertain":
            cue_start, cue_end = draft.say("maybe")
            draft.say(" ")
            span_id = draft.value(
                cast.allergen,
                L.MEDICATION,
                "patient",
                "allergy",
                assertion="uncertain",
            )
            draft.event("uncertainty", cue_start, cue_end, span_id=span_id)
            draft.say("? It was a long time ago, I'm not sure.")
        elif cast.allergy_assertion == "negated":
            draft.say("No. I've taken ")
            draft.value(
                cast.allergen,
                L.MEDICATION,
                "patient",
                "allergy",
                assertion="negated",
            )
            draft.say(" before and I was fine.")
        else:
            draft.say("Yes, ")
            draft.value(cast.allergen, L.MEDICATION, "patient", "allergy")
            draft.say(". I break out in a rash.")
        self.commit(draft)

    def medications(self) -> None:
        cast = self.cast
        self.ask("What medications are you taking right now?", "medication_prompt")
        draft = self.turn(self.voice, "medication_answer")
        draft.say("Just ")
        current = draft.value(
            cast.medication, L.MEDICATION, "patient", "current_medication"
        )
        draft.say(", once a day.")
        self.commit(draft)
        if self.scenario != "caregiver":
            return
        draft = self.turn("caregiver", "medication_correction")
        cue_start, cue_end = draft.say("actually")
        draft.say(f", they switched {cast.pronouns.object} to ")
        replacement = draft.value(
            cast.new_medication,
            L.MEDICATION,
            "patient",
            "current_medication",
            revision="correction",
            supersedes=current,
        )
        draft.event(
            "correction", cue_start, cue_end, supersedes=current, repair=replacement
        )
        draft.say(" last month.")
        self.commit(draft)

    def private_pause(self) -> None:
        cast = self.cast
        consent = {"scope": "recording", "subject_role": "patient"}
        draft = self.turn(self.voice, "consent_pause")
        draft.cue(
            "could you pause the recording for a minute?",
            "consent",
            action="pause",
            **consent,
        )
        draft.say(" This next part is private.")
        self.commit(draft)
        self.ask("Of course, it's paused.", "consent_ack")
        draft = self.turn(self.voice, "private_disclosure")
        draft.say("My work number is ")
        draft.value(
            _format_phone(cast.work_phone, "dashed"), L.PHONE, "patient", "work_phone"
        )
        draft.say(", but please don't call me there.")
        self.commit(draft)
        draft = self.turn("clinician", "consent_request")
        draft.say("Understood. ")
        draft.cue(
            "Is it okay to turn the recording back on?",
            "consent",
            action="request",
            **consent,
        )
        self.commit(draft)
        draft = self.turn(self.voice, "consent_resume")
        draft.cue("yes, go ahead", "consent", action="resume", **consent)
        draft.say(".")
        self.commit(draft)

    def late_phone_correction(self) -> None:
        cast = self.cast
        draft = self.turn(self.voice, "late_correction")
        draft.say("Oh, wait. ")
        cue_start, cue_end = draft.say("The number I gave you earlier is my old one")
        draft.say(". The new one is ")
        previous = self._latest["patient.phone"]
        replacement = draft.value(
            _format_phone(cast.new_phone, cast.phone_style),
            L.PHONE,
            "patient",
            "phone",
            revision="correction",
            supersedes=previous,
        )
        draft.event(
            "correction", cue_start, cue_end, supersedes=previous, repair=replacement
        )
        draft.say(".")
        self.commit(draft)

    def closing(self) -> None:
        cast = self.cast
        draft = self.turn("clinician", "closing")
        draft.say("Thank you, ")
        draft.value(cast.patient_first, L.PERSON, "patient", "given_name")
        if cast.clinician_is_doctor:
            draft.say(". The nurse will take your vitals next.")
        else:
            draft.say(". The doctor will be in shortly.")
        self.commit(draft)


def _generate_conversation(
    *, seed: int, index: int, scenario: str
) -> IntakeConversation:
    rng = random.Random(_derived_seed(seed, index))
    cast = _sample_cast(rng, scenario)
    turns = _IntakeScript(rng, cast, scenario, index).run()
    return IntakeConversation(
        conversation_id=f"synthetic-intake-{seed}-{index:04d}",
        scenario=scenario,
        seed=seed,
        index=index,
        turns=turns,
    )


__all__ = [
    "ASSERTION_STATUSES",
    "CONSENT_ACTIONS",
    "CONSENT_STATES",
    "INTAKE_CLINICAL_LABELS",
    "INTAKE_DIALOGUE_MANIFEST_SCHEMA_VERSION",
    "INTAKE_DIALOGUE_SCHEMA_VERSION",
    "INTAKE_PHI_LABELS",
    "INTAKE_SCENARIOS",
    "REVISION_TYPES",
    "SPEAKER_ROLES",
    "STREAM_TURN_SEPARATOR",
    "SYNTHETIC_INTAKE_DIALOGUE_LICENSE",
    "SYNTHETIC_INTAKE_DIALOGUE_SOURCE",
    "TURN_EVENT_TYPES",
    "IntakeConversation",
    "IntakeEvent",
    "IntakeSpan",
    "IntakeTurn",
    "build_intake_dialogue_manifest",
    "generate_intake_conversations",
]
