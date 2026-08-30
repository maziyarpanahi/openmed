"""Document-local medication mention reconciliation.

This module consumes medication mentions that have already been detected and
optionally grounded.  It deliberately does not parse sigs, call a terminology
service, or infer medication-to-attribute relations.  Reconciliation is a
deterministic organization aid for downstream review: source offsets and
normalized values are retained, while source mention text is never emitted by
the reconciled records.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Literal

SpanOffset = tuple[int, int]
MedicationStatus = Literal["started", "continued", "held", "changed", "stopped"]

MEDICATION_STATUS_VALUES: tuple[MedicationStatus, ...] = (
    "started",
    "continued",
    "held",
    "changed",
    "stopped",
)

MEDICATION_RECONCILIATION_ADVISORY = (
    "Medication reconciliation is deterministic document-local organization "
    "for review and downstream exchange, not a prescription or clinical "
    "decision. Unresolved dose and route conflicts require review."
)

_DEFAULT_DOCUMENT_ID = "document"
_MISSING = object()
_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w/+.-]+", re.UNICODE)
_STRENGTH_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|μg|ug|ng|kg|ml|mL|iu|units?)\b"
    r"(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|g|mcg|μg|ug|ng|kg|ml|mL|iu|units?))?",
    re.IGNORECASE,
)
_FORM_RE = re.compile(
    r"\b(?:tablets?|tabs?|capsules?|caps?|pills?|solutions?|syrups?|"
    r"suspensions?|injections?|patches?|creams?|ointments?|drops?|"
    r"oral|intravenous|iv|intramuscular|im|subcutaneous|subcut|sc|"
    r"po|by\s+mouth)\b",
    re.IGNORECASE,
)
_ISO_MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")
_ISO_YEAR_RE = re.compile(r"^\d{4}$")

_STATUS_ALIASES: dict[str, MedicationStatus] = {
    "start": "started",
    "started": "started",
    "starting": "started",
    "initiate": "started",
    "initiated": "started",
    "begin": "started",
    "began": "started",
    "add": "started",
    "added": "started",
    "continue": "continued",
    "continued": "continued",
    "continuing": "continued",
    "resume": "continued",
    "resumed": "continued",
    "restart": "continued",
    "restarted": "continued",
    "active": "continued",
    "ongoing": "continued",
    "hold": "held",
    "held": "held",
    "holding": "held",
    "withhold": "held",
    "withheld": "held",
    "on hold": "held",
    "on-hold": "held",
    "change": "changed",
    "changed": "changed",
    "increase": "changed",
    "increased": "changed",
    "decrease": "changed",
    "decreased": "changed",
    "reduced": "changed",
    "titrated": "changed",
    "stop": "stopped",
    "stopped": "stopped",
    "stopping": "stopped",
    "discontinue": "stopped",
    "discontinued": "stopped",
    "discontinuation": "stopped",
    "ceased": "stopped",
    "inactive": "stopped",
    "completed": "stopped",
}

_ROUTE_ALIASES = {
    "po": "oral",
    "p.o.": "oral",
    "by mouth": "oral",
    "oral": "oral",
    "iv": "intravenous",
    "i.v.": "intravenous",
    "intravenous": "intravenous",
    "im": "intramuscular",
    "i.m.": "intramuscular",
    "intramuscular": "intramuscular",
    "sc": "subcutaneous",
    "s.c.": "subcutaneous",
    "subcut": "subcutaneous",
    "subcutaneous": "subcutaneous",
    "sl": "sublingual",
    "s.l.": "sublingual",
    "sublingual": "sublingual",
    "inhaled": "inhaled",
    "inhalation": "inhaled",
    "topical": "topical",
    "transdermal": "transdermal",
    "rectal": "rectal",
    "vaginal": "vaginal",
    "intranasal": "intranasal",
    "nasal": "intranasal",
}

_SECTION_PRECEDENCE = {
    "assessment": 50,
    "plan": 50,
    "medication_list": 45,
    "medications": 45,
    "current_medications": 45,
    "history_of_present_illness": 40,
    "hpi": 40,
    "past_medical_history": 30,
    "history": 30,
    "social_history": 20,
    "family_history": 10,
}


@dataclass(frozen=True)
class MedicationMention:
    """Input medication mention with optional grounding and state attributes.

    ``text`` is accepted for adapters and locating offsets, but is not copied
    into reconciled output.  Callers may provide ``ingredient`` or
    ``normalized_ingredient`` when a local grounding layer already resolved
    the active ingredient.  ``timestamp`` and ``normalized_timestamp`` are
    aliases for ``effective_time``.
    """

    text: str | None = None
    ingredient: str | None = None
    normalized_ingredient: str | None = None
    name: str | None = None
    system: str | None = None
    code: str | None = None
    dose: str | None = None
    route: str | None = None
    status: str | None = None
    action: str | None = None
    effective_time: str | date | datetime | Mapping[str, Any] | None = None
    timestamp: str | date | datetime | Mapping[str, Any] | None = None
    normalized_timestamp: str | date | datetime | Mapping[str, Any] | None = None
    offset: SpanOffset | None = None
    start: int | None = None
    end: int | None = None
    document_id: str | None = None
    coref_entity_id: str | None = None
    entity_id: str | None = None
    cluster_id: str | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate only adapter-facing fields; normalization happens on use."""

        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("medication mention text must be a string when provided")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("medication mention metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MedicationHistoryEntry:
    """Privacy-safe normalized state evidence for one source mention."""

    status: MedicationStatus
    dose: str | None
    route: str | None
    effective_time: str | None
    source_offset: SpanOffset | None
    source_index: int

    @property
    def offset(self) -> SpanOffset | None:
        """Return the source offset under the short adapter-facing name."""

        return self.source_offset

    @property
    def timestamp(self) -> str | None:
        """Return the normalized effective time."""

        return self.effective_time

    @property
    def action(self) -> MedicationStatus:
        """Return the normalized transition action."""

        return self.status

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without source mention text."""

        return {
            "status": self.status,
            "dose": self.dose,
            "route": self.route,
            "effective_time": self.effective_time,
            "source_offset": list(self.source_offset)
            if self.source_offset is not None
            else None,
            "source_index": self.source_index,
        }


@dataclass(frozen=True)
class MedicationConflict:
    """Unresolved state conflict with normalized values and source offsets."""

    field: str
    values: tuple[str, ...]
    source_offsets: tuple[SpanOffset, ...]
    reason: str

    @property
    def attribute(self) -> str:
        """Return ``field`` under the relation-style attribute name."""

        return self.field

    @property
    def offsets(self) -> tuple[SpanOffset, ...]:
        """Return all contributing offsets."""

        return self.source_offsets

    @property
    def unresolved(self) -> bool:
        """Return whether this conflict needs human review."""

        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe conflict record without source mention text."""

        return {
            "field": self.field,
            "values": list(self.values),
            "source_offsets": [list(offset) for offset in self.source_offsets],
            "reason": self.reason,
            "unresolved": True,
        }


@dataclass(frozen=True)
class MedicationState:
    """Current normalized state exposed as a small value object."""

    status: MedicationStatus
    dose: str | None
    route: str | None
    effective_time: str | None

    def __getitem__(self, key: str) -> str | None:
        """Support mapping-style access for pipeline adapters."""

        if key not in {"status", "dose", "route", "effective_time"}:
            raise KeyError(key)
        return getattr(self, key)

    def to_dict(self) -> dict[str, str | None]:
        """Return the normalized current state."""

        return {
            "status": self.status,
            "dose": self.dose,
            "route": self.route,
            "effective_time": self.effective_time,
        }


@dataclass(frozen=True)
class ReconciledMedication:
    """One document-local medication with current state and history.

    No source surface text is stored.  ``ingredient`` is a normalized identity
    value; ``source_offsets`` and each history entry retain only offsets for
    traceability back to a caller-controlled document.
    """

    ingredient: str
    medication_key: str
    system: str | None
    code: str | None
    current_status: MedicationStatus
    current_dose: str | None
    current_route: str | None
    history: tuple[MedicationHistoryEntry, ...]
    source_offsets: tuple[SpanOffset, ...]
    conflicts: tuple[MedicationConflict, ...] = ()
    document_id: str = _DEFAULT_DOCUMENT_ID
    coref_entity_id: str | None = None
    advisory: str = MEDICATION_RECONCILIATION_ADVISORY

    def __post_init__(self) -> None:
        """Freeze sequence fields and validate the privacy-safe record."""

        if not self.ingredient:
            raise ValueError("reconciled medication ingredient must be non-empty")
        if not self.medication_key:
            raise ValueError("reconciled medication key must be non-empty")
        if self.current_status not in MEDICATION_STATUS_VALUES:
            raise ValueError("reconciled medication status is not normalized")
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "source_offsets", tuple(self.source_offsets))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    @property
    def normalized_ingredient(self) -> str:
        """Return the normalized medication identity."""

        return self.ingredient

    @property
    def status(self) -> MedicationStatus:
        """Return the current normalized status."""

        return self.current_status

    @property
    def dose(self) -> str | None:
        """Return the current normalized dose."""

        return self.current_dose

    @property
    def route(self) -> str | None:
        """Return the current normalized route."""

        return self.current_route

    @property
    def drug(self) -> str:
        """Return the normalized drug identity under a common alias."""

        return self.ingredient

    @property
    def mention_count(self) -> int:
        """Return the number of contributing source mentions."""

        return len(self.history)

    @property
    def source_span_offsets(self) -> tuple[SpanOffset, ...]:
        """Return source offsets under the explicit provenance name."""

        return self.source_offsets

    @property
    def has_conflicts(self) -> bool:
        """Return whether any unresolved attribute conflict was surfaced."""

        return bool(self.conflicts)

    @property
    def unresolved_conflicts(self) -> tuple[MedicationConflict, ...]:
        """Return conflicts requiring review."""

        return self.conflicts

    @property
    def current_state(self) -> MedicationState:
        """Return the normalized current state value object."""

        effective_time = next(
            (
                entry.effective_time
                for entry in reversed(self.history)
                if entry.effective_time is not None
            ),
            None,
        )
        return MedicationState(
            status=self.current_status,
            dose=self.current_dose,
            route=self.current_route,
            effective_time=effective_time,
        )

    @property
    def state(self) -> MedicationState:
        """Return ``current_state`` under the concise adapter-facing name."""

        return self.current_state

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without raw source mention text."""

        return {
            "medication_key": self.medication_key,
            "ingredient": self.ingredient,
            "normalized_ingredient": self.ingredient,
            "system": self.system,
            "code": self.code,
            "current_status": self.current_status,
            "current_dose": self.current_dose,
            "current_route": self.current_route,
            "current_state": self.current_state.to_dict(),
            "history": [entry.to_dict() for entry in self.history],
            "source_offsets": [list(offset) for offset in self.source_offsets],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "document_id": self.document_id,
            "coref_entity_id": self.coref_entity_id,
            "advisory": self.advisory,
        }


@dataclass(frozen=True)
class _NormalizedTimestamp:
    value: str
    sort_value: datetime


@dataclass(frozen=True)
class _PreparedMention:
    source_index: int
    document_id: str
    ingredient: str
    system: str | None
    code: str | None
    dose: str | None
    route: str | None
    status: MedicationStatus
    effective_time: str | None
    effective_sort: datetime | None
    source_offset: SpanOffset | None
    coref_entity_id: str | None
    section: str | None
    identity_keys: tuple[str, ...]


def normalize_medication_status(value: object | None) -> MedicationStatus:
    """Normalize common start/continue/hold/change/stop status cues."""

    if value is None or not str(value).strip():
        return "continued"
    if isinstance(value, Mapping):
        value = _first_mapping_value(value, ("status", "action", "value"))
    normalized = _normalize_scalar(value)
    status = _STATUS_ALIASES.get(normalized)
    if status is None:
        raise ValueError(
            f"unsupported medication status {value!r}; expected one of "
            f"{', '.join(MEDICATION_STATUS_VALUES)}"
        )
    return status


def normalize_medication_dose(value: object | None) -> str | None:
    """Normalize a supplied dose value without parsing sig frequency."""

    return _normalize_scalar_or_none(value)


def normalize_medication_route(value: object | None) -> str | None:
    """Normalize common route abbreviations to stable lower-case values."""

    normalized = _normalize_scalar_or_none(value)
    if normalized is None:
        return None
    return _ROUTE_ALIASES.get(normalized, normalized)


def normalize_medication_timestamp(value: object | None) -> str | None:
    """Normalize an absolute effective date/time to an ISO value.

    Relative expressions are intentionally rejected here.  They must be
    normalized by the caller's document-timeline layer with an explicit
    reference time before reconciliation.
    """

    normalized = _normalize_timestamp(value)
    return normalized.value if normalized is not None else None


def reconcile_medications(
    mentions: Iterable[MedicationMention | Mapping[str, Any] | Any],
    *,
    document_id: str | None = None,
    document_text: str | None = None,
    coreference_chains: Sequence[Any] = (),
    coref_chains: Sequence[Any] | None = None,
    ingredient_grounder: Callable[[str], Any] | None = None,
    grounder: Callable[[str], Any] | None = None,
) -> list[ReconciledMedication]:
    """Reconcile document-local medication mentions into current states.

    Identity precedence is coreference entity, explicit ingredient, coded
    grounding, then normalized mention surface.  A caller-owned local
    ``ingredient_grounder``/``grounder`` may supply identity for otherwise
    ungrounded text; no default grounder or network call is used.

    Effective timestamps are normalized before ordering.  The latest
    timestamped evidence determines status and attribute precedence.  When
    conflicting dose or route values share the highest timestamp and section
    precedence, the current value is ``None`` and a conflict records the
    normalized values and their source offsets.  Untimestamped disagreements
    are likewise left unresolved unless section precedence gives one value a
    unique, higher-authority source.

    Args:
        mentions: Medication mention mappings, :class:`MedicationMention`
            values, ``GroundedSpan``-like objects, or objects exposing the
            same fields.  Mappings accept ``offset`` or ``start``/``end``;
            identity can come from ``ingredient``, ``system`` + ``code``,
            grounding candidates, or text.
        document_id: Optional document id applied to mentions that omit one.
            Reconciliation rejects mixed-document input because cross-document
            reconciliation is out of scope.
        document_text: Optional caller-controlled source text used only to
            locate a missing offset.  It is never copied to output.
        coreference_chains: Optional span-native coreference chains.  A chain
            member offset receives the chain id as a preferred identity key.
        coref_chains: Alias for ``coreference_chains``.
        ingredient_grounder: Optional local callback used only when a mention
            has no coded or ingredient identity.
        grounder: Alias for ``ingredient_grounder``.

    Returns:
        One privacy-safe :class:`ReconciledMedication` per document-local
        medication identity, in first-group encounter order.

    Raises:
        TypeError: If a mention, offset, timestamp, or callback has an invalid
            shape.
        ValueError: If input spans are invalid, documents are mixed, or a
            status/timestamp cannot be normalized safely.
    """

    if coref_chains is not None:
        if coreference_chains:
            raise ValueError("provide only one of coreference_chains and coref_chains")
        coreference_chains = coref_chains
    if coreference_chains is None:
        coreference_chains = ()
    if ingredient_grounder is not None and grounder is not None:
        raise ValueError("provide only one of ingredient_grounder and grounder")
    local_grounder = ingredient_grounder or grounder
    if local_grounder is not None and not callable(local_grounder):
        raise TypeError("ingredient_grounder must be callable")
    if document_text is not None and not isinstance(document_text, str):
        raise TypeError("document_text must be a string when provided")

    chain_index, chain_document_ids = _coreference_index(coreference_chains)
    default_document_id = _resolve_document_id(document_id, chain_document_ids)
    if mentions is None:
        return []
    prepared = tuple(
        _prepare_mention(
            raw,
            index,
            default_document_id=default_document_id,
            document_text=document_text,
            coreference_index=chain_index,
            ingredient_grounder=local_grounder,
        )
        for index, raw in enumerate(mentions)
    )
    document_ids = {mention.document_id for mention in prepared}
    if len(document_ids) > 1:
        raise ValueError("medication reconciliation is limited to one document")
    if not prepared:
        return []

    groups = _group_mentions(prepared)
    return [_reconcile_group(group) for group in groups]


def _prepare_mention(
    raw: MedicationMention | Mapping[str, Any] | Any,
    index: int,
    *,
    default_document_id: str,
    document_text: str | None,
    coreference_index: Mapping[tuple[str, SpanOffset], str],
    ingredient_grounder: Callable[[str], Any] | None,
) -> _PreparedMention:
    text = _first_value(raw, ("text", "surface", "mention", "label", "term", "name"))
    if text is not None and not isinstance(text, str):
        raise TypeError("medication mention text must be a string when provided")
    offset = _coerce_offset(raw)
    if offset is None and document_text is not None and text:
        offset = _locate_text(document_text, text, _first_value(raw, ("occurrence",)))

    raw_document_id = _first_value(raw, ("document_id", "doc_id", "source_doc_id"))
    current_document_id = _clean_identifier(raw_document_id) or default_document_id
    if offset is not None:
        chain_id = coreference_index.get((current_document_id, offset))
    else:
        chain_id = None

    grounder_result: Any = None
    if ingredient_grounder is not None and text:
        has_explicit_identity = any(
            _first_value(raw, keys) is not None
            for keys in (
                ("ingredient", "normalized_ingredient", "active_ingredient"),
                ("code", "concept_code"),
            )
        )
        has_grounding_identity = any(_grounding_identity(raw, None))
        if not has_explicit_identity and not has_grounding_identity:
            grounder_result = ingredient_grounder(text)

    candidate_system, candidate_code, candidate_display = _grounding_identity(
        raw, grounder_result
    )
    explicit_ingredient = _first_value(
        raw,
        (
            "normalized_ingredient",
            "ingredient",
            "active_ingredient",
            "canonical_ingredient",
            "canonical_name",
            "drug",
            "medication",
            "medication_name",
        ),
    )
    if explicit_ingredient is None and isinstance(grounder_result, str):
        explicit_ingredient = grounder_result
    if explicit_ingredient is None:
        explicit_ingredient = _first_value(
            grounder_result,
            (
                "normalized_ingredient",
                "ingredient",
                "active_ingredient",
                "canonical_ingredient",
                "canonical_name",
            ),
        )
    ingredient_source = explicit_ingredient or candidate_display or text
    if ingredient_source is None:
        raise ValueError("medication mention requires text, ingredient, or grounding")
    ingredient = _normalize_ingredient(ingredient_source)
    if not ingredient:
        raise ValueError("medication mention ingredient must not be empty")

    raw_system = _first_value(raw, ("system", "coding_system", "code_system"))
    raw_code = _first_value(raw, ("code", "concept_code", "coding_code"))
    system = _normalize_system(raw_system or candidate_system)
    code = _normalize_code(raw_code or candidate_code)

    explicit_coref_id = _first_value(
        raw, ("coref_entity_id", "coref_id", "entity_id", "cluster_id")
    )
    coref_entity_id = _clean_identifier(explicit_coref_id) or chain_id

    status_value = _first_value(
        raw,
        ("status", "medication_status", "action", "event", "state"),
    )
    status = normalize_medication_status(status_value)
    dose = normalize_medication_dose(
        _first_value(raw, ("dose", "dosage", "strength", "dose_value"))
    )
    route = normalize_medication_route(
        _first_value(raw, ("route", "administration_route"))
    )
    time_value = _first_value(
        raw,
        (
            "normalized_timestamp",
            "effective_time",
            "effective_at",
            "timestamp",
            "effective_date",
            "date",
        ),
    )
    normalized_time = _normalize_timestamp(time_value)
    section = _normalize_section(_first_value(raw, ("section", "section_label")))
    keys = _identity_keys(
        document_id=current_document_id,
        ingredient=ingredient,
        system=system,
        code=code,
        coref_entity_id=coref_entity_id,
    )
    return _PreparedMention(
        source_index=index,
        document_id=current_document_id,
        ingredient=ingredient,
        system=system,
        code=code,
        dose=dose,
        route=route,
        status=status,
        effective_time=normalized_time.value if normalized_time else None,
        effective_sort=normalized_time.sort_value if normalized_time else None,
        source_offset=offset,
        coref_entity_id=coref_entity_id,
        section=section,
        identity_keys=keys,
    )


def _group_mentions(
    mentions: Sequence[_PreparedMention],
) -> list[list[_PreparedMention]]:
    groups: list[list[_PreparedMention]] = []
    key_to_group: dict[str, int] = {}
    for mention in mentions:
        matching_groups = sorted(
            {key_to_group[key] for key in mention.identity_keys if key in key_to_group}
        )
        if not matching_groups:
            groups.append([mention])
            target = len(groups) - 1
        else:
            target = matching_groups[0]
            groups[target].append(mention)
            for source_group in reversed(matching_groups[1:]):
                groups[target].extend(groups[source_group])
                groups[source_group] = []
                for key, group_index in tuple(key_to_group.items()):
                    if group_index == source_group:
                        key_to_group[key] = target
        for key in mention.identity_keys:
            key_to_group[key] = target

    return [
        sorted(group, key=lambda item: item.source_index) for group in groups if group
    ]


def _reconcile_group(group: Sequence[_PreparedMention]) -> ReconciledMedication:
    history_mentions = _history_order(group)
    history = tuple(
        MedicationHistoryEntry(
            status=mention.status,
            dose=mention.dose,
            route=mention.route,
            effective_time=mention.effective_time,
            source_offset=mention.source_offset,
            source_index=mention.source_index,
        )
        for mention in history_mentions
    )
    current_mention = _latest_status_mention(group)
    current_dose, dose_conflict = _resolve_attribute(group, "dose")
    current_route, route_conflict = _resolve_attribute(group, "route")
    conflicts = tuple(
        conflict for conflict in (dose_conflict, route_conflict) if conflict is not None
    )
    identity = _identity_summary(group)
    offsets = tuple(
        mention.source_offset
        for mention in sorted(group, key=_source_order_key)
        if mention.source_offset is not None
    )
    return ReconciledMedication(
        ingredient=identity[0],
        medication_key=identity[1],
        system=identity[2],
        code=identity[3],
        current_status=current_mention.status,
        current_dose=current_dose,
        current_route=current_route,
        history=history,
        source_offsets=offsets,
        conflicts=conflicts,
        document_id=group[0].document_id,
        coref_entity_id=identity[4],
    )


def _resolve_attribute(
    group: Sequence[_PreparedMention], field_name: Literal["dose", "route"]
) -> tuple[str | None, MedicationConflict | None]:
    valued = [mention for mention in group if getattr(mention, field_name) is not None]
    if not valued:
        return None, None
    values = _ordered_unique(str(getattr(item, field_name)) for item in valued)
    if len(values) == 1:
        return values[0], None

    timestamped = [item for item in valued if item.effective_sort is not None]
    if timestamped:
        latest_time = max(item.effective_sort for item in timestamped)
        finalists = [item for item in timestamped if item.effective_sort == latest_time]
        resolution_reason = "latest normalized effective timestamp"
    else:
        highest_section = max(_section_priority(item.section) for item in valued)
        finalists = [
            item
            for item in valued
            if _section_priority(item.section) == highest_section
        ]
        resolution_reason = "highest section precedence without timestamps"

    finalist_values = _ordered_unique(
        str(getattr(item, field_name)) for item in finalists
    )
    if len(finalist_values) == 1:
        return finalist_values[0], None

    conflict_offsets = tuple(
        mention.source_offset
        for mention in sorted(valued, key=_source_order_key)
        if mention.source_offset is not None
    )
    return None, MedicationConflict(
        field=field_name,
        values=tuple(sorted(values)),
        source_offsets=conflict_offsets,
        reason=(
            f"conflicting normalized {field_name} values remain unresolved at "
            f"the {resolution_reason}"
        ),
    )


def _history_order(group: Sequence[_PreparedMention]) -> list[_PreparedMention]:
    if any(item.effective_sort is not None for item in group):
        return sorted(
            group,
            key=lambda item: (
                0 if item.effective_sort is not None else 1,
                item.effective_sort or datetime.max,
                _source_order_key(item),
            ),
        )
    return sorted(group, key=_source_order_key)


def _latest_status_mention(group: Sequence[_PreparedMention]) -> _PreparedMention:
    timestamped = [item for item in group if item.effective_sort is not None]
    if timestamped:
        return max(
            timestamped,
            key=lambda item: (item.effective_sort, _source_order_key(item)),
        )
    return max(group, key=_source_order_key)


def _identity_summary(
    group: Sequence[_PreparedMention],
) -> tuple[str, str, str | None, str | None, str | None]:
    ingredient = _ordered_unique(item.ingredient for item in group)[0]
    coded = _ordered_unique(
        (item.system, item.code)
        for item in group
        if item.system is not None and item.code is not None
    )
    system: str | None = None
    code: str | None = None
    if coded:
        system, code = coded[0]
    coref_ids = _ordered_unique(
        item.coref_entity_id for item in group if item.coref_entity_id is not None
    )
    coref_entity_id = coref_ids[0] if coref_ids else None
    medication_key = (
        f"{system}:{code}" if system is not None and code is not None else ingredient
    )
    return ingredient, medication_key, system, code, coref_entity_id


def _identity_keys(
    *,
    document_id: str,
    ingredient: str,
    system: str | None,
    code: str | None,
    coref_entity_id: str | None,
) -> tuple[str, ...]:
    keys: list[str] = []
    if coref_entity_id:
        keys.append(f"coref:{document_id}:{coref_entity_id}")
    if ingredient:
        keys.append(f"ingredient:{ingredient}")
    if system is not None and code is not None:
        keys.append(f"code:{system}:{code}")
    if not keys:
        raise ValueError("medication mention has no usable identity")
    return tuple(keys)


def _coreference_index(
    chains: Sequence[Any],
) -> tuple[dict[tuple[str, SpanOffset], str], set[str]]:
    index: dict[tuple[str, SpanOffset], str] = {}
    documents: set[str] = set()
    for chain in chains:
        chain_id = _first_value(chain, ("chain_id", "cluster_id", "entity_id"))
        if chain_id is None:
            raise TypeError("coreference chain must expose chain_id")
        normalized_chain_id = _clean_identifier(chain_id)
        if normalized_chain_id is None:
            raise ValueError("coreference chain id must be non-empty")
        members = _first_value(chain, ("members", "member_spans", "spans"))
        if members is None:
            raise TypeError("coreference chain must expose members")
        member_documents: set[str] = set()
        for member in members:
            member_document = _clean_identifier(
                _first_value(member, ("document_id", "doc_id", "source_doc_id"))
            )
            if member_document is None:
                member_document = _DEFAULT_DOCUMENT_ID
            offset = _coerce_offset(member)
            if offset is None:
                raise ValueError("coreference chain members require source offsets")
            member_documents.add(member_document)
            key = (member_document, offset)
            previous = index.get(key)
            if previous is not None and previous != normalized_chain_id:
                raise ValueError(
                    "one medication source offset cannot belong to two chains"
                )
            index[key] = normalized_chain_id
        if len(member_documents) != 1:
            raise ValueError("coreference chains must be document-local")
        documents.update(member_documents)
    return index, documents


def _resolve_document_id(document_id: str | None, chain_documents: set[str]) -> str:
    if document_id is not None:
        cleaned = _clean_identifier(document_id)
        if cleaned is None:
            raise ValueError("document_id must be non-empty")
        return cleaned
    if len(chain_documents) == 1:
        return next(iter(chain_documents))
    return _DEFAULT_DOCUMENT_ID


def _grounding_identity(
    raw: Any, grounder_result: Any
) -> tuple[str | None, str | None, str | None]:
    candidates = [*_candidate_records(raw), *_candidate_records(grounder_result)]
    if not candidates:
        return None, None, None
    candidates.sort(
        key=lambda item: (
            0 if item[0] and item[0].casefold() == "rxnorm" else 1,
            -item[3],
            item[0] or "",
            item[1] or "",
        )
    )
    system, code, display, _score = candidates[0]
    return system, code, display


def _candidate_records(
    raw: Any, depth: int = 0
) -> list[tuple[str | None, str | None, str | None, float]]:
    if raw is None or depth > 3:
        return []
    records: list[tuple[str | None, str | None, str | None, float]] = []
    system = _first_value(raw, ("system", "coding_system", "code_system"))
    code = _first_value(raw, ("code", "concept_code", "coding_code"))
    display = _first_value(raw, ("display", "concept_display", "canonical_name"))
    if system is not None or code is not None:
        score_value = _first_value(raw, ("score", "confidence"))
        try:
            score = float(score_value) if score_value is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        records.append(
            (
                _clean_optional_scalar(system),
                _clean_optional_scalar(code),
                _clean_optional_scalar(display),
                score,
            )
        )
    codes = _first_value(raw, ("codes",))
    if isinstance(codes, Mapping):
        records.extend(
            (
                _clean_optional_scalar(system_name),
                _clean_optional_scalar(code_value),
                None,
                0.0,
            )
            for system_name, code_value in codes.items()
        )
    for nested_key in ("candidates", "grounding", "grounded", "concepts"):
        nested = _first_value(raw, (nested_key,))
        if nested is None:
            continue
        if isinstance(nested, Mapping) or isinstance(nested, str):
            records.extend(_candidate_records(nested, depth + 1))
        else:
            try:
                for item in nested:
                    records.extend(_candidate_records(item, depth + 1))
            except TypeError:
                records.extend(_candidate_records(nested, depth + 1))
    return records


def _first_value(raw: Any, keys: Sequence[str]) -> Any:
    if raw is None:
        return None
    metadata = None
    if isinstance(raw, Mapping):
        sources: tuple[Any, ...] = (raw,)
        metadata = raw.get("metadata")
    else:
        sources = (raw,)
        metadata = getattr(raw, "metadata", None)
    for source in sources:
        for key in keys:
            value = (
                source.get(key, _MISSING)
                if isinstance(source, Mapping)
                else getattr(source, key, _MISSING)
            )
            if value is not _MISSING and value is not None:
                return value
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key, _MISSING)
            if value is not _MISSING and value is not None:
                return value
    return None


def _first_mapping_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _coerce_offset(raw: Any) -> SpanOffset | None:
    value = _first_value(raw, ("offset", "span", "source_offset"))
    if value is None:
        start = _first_value(raw, ("start", "start_char", "offset_start"))
        end = _first_value(raw, ("end", "end_char", "offset_end"))
        if start is None and end is None:
            return None
        value = (start, end)
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise TypeError("medication source offset must be a (start, end) integer pair")
    start, end = value
    if start < 0 or end < start:
        raise ValueError("medication source offset must satisfy 0 <= start <= end")
    return start, end


def _locate_text(text: str, surface: str, occurrence: Any) -> SpanOffset | None:
    if occurrence is None:
        start = text.find(surface)
    else:
        if type(occurrence) is not int or occurrence < 0:
            raise ValueError("medication occurrence must be a non-negative integer")
        start = -1
        cursor = 0
        for _ in range(occurrence + 1):
            start = text.find(surface, cursor)
            if start < 0:
                break
            cursor = start + len(surface)
    if start < 0:
        return None
    return start, start + len(surface)


def _normalize_ingredient(value: object) -> str:
    normalized = _normalize_scalar(value)
    normalized = _STRENGTH_RE.sub(" ", normalized)
    normalized = _FORM_RE.sub(" ", normalized)
    normalized = _NON_WORD_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip(" -_/.")
    return normalized


def _normalize_scalar(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-")
    text = text.replace("\u2212", "-").replace("μ", "µ")
    return _WHITESPACE_RE.sub(" ", text.casefold()).strip()


def _normalize_scalar_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_scalar(value)
    return normalized or None


def _clean_optional_scalar(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_scalar(value)
    return normalized or None


def _clean_identifier(value: object | None) -> str | None:
    normalized = _normalize_scalar_or_none(value)
    return normalized


def _normalize_code(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = _WHITESPACE_RE.sub(" ", str(value).strip())
    return cleaned or None


def _normalize_system(value: object | None) -> str | None:
    normalized = _normalize_scalar_or_none(value)
    if normalized is None:
        return None
    if "rxnorm" in normalized or normalized in {"rx-norm", "rx norm"}:
        return "RXNORM"
    return normalized.upper()


def _normalize_section(value: object | None) -> str | None:
    normalized = _normalize_scalar_or_none(value)
    if normalized is None:
        return None
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized


def _normalize_timestamp(value: object | None) -> _NormalizedTimestamp | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        nested = _first_mapping_value(value, ("value", "normalized_value", "iso_value"))
        return _normalize_timestamp(nested)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, date, datetime)):
        nested = getattr(value, "value")
        return _normalize_timestamp(nested)
    if isinstance(value, datetime):
        return _timestamp_from_datetime(value)
    if isinstance(value, date):
        return _NormalizedTimestamp(value.isoformat(), datetime.combine(value, time()))
    if not isinstance(value, str):
        raise TypeError(
            "medication effective time must be an ISO string, date, or datetime"
        )
    text = _normalize_scalar(value)
    if not text:
        return None
    if _ISO_YEAR_RE.fullmatch(text):
        parsed = datetime(int(text), 1, 1)
        return _NormalizedTimestamp(text, parsed)
    if month_match := _ISO_MONTH_RE.fullmatch(text):
        parsed = datetime(
            int(month_match.group("year")), int(month_match.group("month")), 1
        )
        return _NormalizedTimestamp(text, parsed)
    candidate = text[:-1] + "+00:00" if text.endswith("z") else text
    parsed_datetime: datetime | None = None
    try:
        parsed_datetime = datetime.fromisoformat(candidate)
    except ValueError:
        for pattern in ("%Y/%m/%d", "%m/%d/%Y", "%d %B %Y", "%d %b %Y"):
            try:
                parsed_datetime = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed_datetime is None:
        raise ValueError(
            "medication effective time must be an absolute normalized ISO date/time"
        )
    return _timestamp_from_datetime(parsed_datetime, original=text)


def _timestamp_from_datetime(
    value: datetime, *, original: str | None = None
) -> _NormalizedTimestamp:
    if value.tzinfo is not None:
        normalized_value = value.astimezone(timezone.utc).replace(tzinfo=None)
        serialized = normalized_value.isoformat(timespec="seconds") + "Z"
    else:
        normalized_value = value.replace(microsecond=0)
        serialized = normalized_value.isoformat(timespec="seconds")
    if original is not None and re.fullmatch(r"\d{4}/\d{2}/\d{2}", original):
        serialized = normalized_value.date().isoformat()
    elif original is not None and re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", original):
        serialized = normalized_value.date().isoformat()
    elif (
        value.time() == time()
        and original is not None
        and "T" not in original
        and " " not in original
    ):
        serialized = normalized_value.date().isoformat()
    return _NormalizedTimestamp(serialized, normalized_value)


def _section_priority(section: str | None) -> int:
    return _SECTION_PRECEDENCE.get(section or "", 25)


def _source_order_key(mention: _PreparedMention) -> tuple[int, int, int]:
    if mention.source_offset is not None:
        return mention.source_offset[0], mention.source_offset[1], mention.source_index
    return (10**12, 10**12, mention.source_index)


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


__all__ = [
    "MEDICATION_RECONCILIATION_ADVISORY",
    "MEDICATION_STATUS_VALUES",
    "MedicationConflict",
    "MedicationHistoryEntry",
    "MedicationMention",
    "MedicationState",
    "MedicationStatus",
    "ReconciledMedication",
    "SpanOffset",
    "normalize_medication_dose",
    "normalize_medication_route",
    "normalize_medication_status",
    "normalize_medication_timestamp",
    "reconcile_medication_mentions",
    "reconcile_medications",
]


reconcile_medication_mentions = reconcile_medications
