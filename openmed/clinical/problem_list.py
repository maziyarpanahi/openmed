"""Problem-list deduplication and clinical-status reconciliation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from .context import (
    AFFIRMED,
    CERTAIN,
    CERTAINTY_VALUES,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATION_VALUES,
    PATIENT_EXPERIENCER,
    RECENT,
    TEMPORALITY_VALUES,
    Certainty,
    ClinicalAssertion,
    Negation,
)

SpanOffset = tuple[int, int]

ProblemClinicalStatus = Literal["active", "inactive", "unconfirmed", "refuted"]

ACTIVE: ProblemClinicalStatus = "active"
INACTIVE: ProblemClinicalStatus = "inactive"
UNCONFIRMED: ProblemClinicalStatus = "unconfirmed"
REFUTED: ProblemClinicalStatus = "refuted"

PROBLEM_CLINICAL_STATUS_VALUES = (ACTIVE, INACTIVE, UNCONFIRMED, REFUTED)

PROBLEM_LIST_RECONCILIATION_ADVISORY = (
    "Problem-list status reconciliation is a heuristic aggregation for review "
    "and downstream organization, not an automated clinical decision."
)

_WHITESPACE_RE = re.compile(r"\s+")

_IdentityKey = tuple[Literal["code", "coref", "text"], str, str]


@dataclass(frozen=True)
class ProblemMention:
    """Candidate problem mention already identified by an upstream extractor.

    The helper consumes existing mention text and optional coded identity. It
    does not ground concepts, assign codes, or emit FHIR Condition resources.
    When ``coref_entity_id`` is supplied by a document-level coreference layer,
    it becomes the preferred deduplication key for that mention.
    """

    text: str
    system: str | None = None
    code: str | None = None
    offset: SpanOffset | None = None
    negation: Negation = AFFIRMED
    temporality: str = RECENT
    coref_entity_id: str | None = None
    certainty: Certainty = CERTAIN
    experiencer: str | None = None


@dataclass(frozen=True)
class ReconciledProblem:
    """Deduplicated problem with advisory status and provenance offsets."""

    text: str
    normalized_text: str
    system: str | None
    code: str | None
    clinical_status: ProblemClinicalStatus
    mention_count: int
    source_offsets: tuple[SpanOffset, ...]
    coref_entity_id: str | None = None


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("problem mention text must be a string")
    cleaned = _WHITESPACE_RE.sub(" ", value.strip())
    if not cleaned:
        raise ValueError("problem mention text must not be empty")
    return cleaned


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("coded identity values must be strings when provided")
    cleaned = _WHITESPACE_RE.sub(" ", value.strip())
    return cleaned or None


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip()).casefold()


def _normalize_code_part(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip()).casefold()


def _validate_offset(offset: object) -> SpanOffset | None:
    if offset is None:
        return None
    if (
        not isinstance(offset, tuple | list)
        or len(offset) != 2
        or not all(isinstance(part, int) for part in offset)
    ):
        raise TypeError("problem mention offset must be a (start, end) integer pair")
    start, end = offset
    if start < 0 or end < start:
        raise ValueError("problem mention offset must satisfy 0 <= start <= end")
    return (start, end)


def _mapping_text(mapping: Mapping[str, object]) -> object:
    for key in ("text", "label", "name", "term", "surface"):
        value = mapping.get(key)
        if value is not None:
            return value
    raise KeyError("problem mention mapping must include a text-like field")


def _mapping_offset(mapping: Mapping[str, object]) -> SpanOffset | None:
    if "offset" in mapping:
        return _validate_offset(mapping["offset"])

    start = mapping.get("start")
    end = mapping.get("end")
    if start is None and end is None:
        return None
    return _validate_offset((start, end))


def _mapping_coref_entity_id(mapping: Mapping[str, object]) -> str | None:
    for key in ("coref_entity_id", "entity_id", "cluster_id"):
        value = mapping.get(key)
        if value is not None:
            return _clean_optional_text(value)
    return None


def _normalize_negation(value: object) -> Negation:
    if value is None:
        return AFFIRMED
    if not isinstance(value, str):
        raise TypeError("problem mention negation must be a string")
    normalized = value.strip().casefold()
    if normalized not in NEGATION_VALUES:
        raise ValueError(
            f"problem mention negation must be one of {', '.join(NEGATION_VALUES)}"
        )
    return cast(Negation, normalized)


def _normalize_temporality(value: object) -> str:
    if value is None:
        return RECENT
    if not isinstance(value, str):
        raise TypeError("problem mention temporality must be a string")
    normalized = value.strip().casefold()
    if normalized not in TEMPORALITY_VALUES:
        raise ValueError(
            "problem mention temporality must be one of "
            f"{', '.join(TEMPORALITY_VALUES)}"
        )
    return normalized


def _normalize_certainty(value: object) -> Certainty:
    if value is None:
        return CERTAIN
    if not isinstance(value, str):
        raise TypeError("problem mention certainty must be a string")
    normalized = value.strip().casefold()
    if normalized not in CERTAINTY_VALUES:
        raise ValueError(
            f"problem mention certainty must be one of {', '.join(CERTAINTY_VALUES)}"
        )
    return cast(Certainty, normalized)


def _normalize_experiencer(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("problem mention experiencer must be a string")
    normalized = _WHITESPACE_RE.sub(" ", value.strip()).casefold()
    return normalized or None


def _coerce_mention(
    mention: ProblemMention | Mapping[str, object],
) -> ProblemMention:
    if isinstance(mention, ProblemMention):
        return ProblemMention(
            text=_clean_text(mention.text),
            system=_clean_optional_text(mention.system),
            code=_clean_optional_text(mention.code),
            offset=_validate_offset(mention.offset),
            negation=_normalize_negation(mention.negation),
            temporality=_normalize_temporality(mention.temporality),
            coref_entity_id=_clean_optional_text(mention.coref_entity_id),
            certainty=_normalize_certainty(mention.certainty),
            experiencer=_normalize_experiencer(mention.experiencer),
        )

    if isinstance(mention, Mapping):
        return ProblemMention(
            text=_clean_text(_mapping_text(mention)),
            system=_clean_optional_text(mention.get("system")),
            code=_clean_optional_text(mention.get("code")),
            offset=_mapping_offset(mention),
            negation=_normalize_negation(mention.get("negation")),
            temporality=_normalize_temporality(mention.get("temporality")),
            coref_entity_id=_mapping_coref_entity_id(mention),
            certainty=_normalize_certainty(mention.get("certainty")),
            experiencer=_normalize_experiencer(mention.get("experiencer")),
        )

    raise TypeError("problem mentions must be ProblemMention instances or mappings")


def _identity_key(mention: ProblemMention) -> _IdentityKey:
    if mention.coref_entity_id:
        return ("coref", mention.coref_entity_id, "")
    if mention.system and mention.code:
        return (
            "code",
            _normalize_code_part(mention.system),
            _normalize_code_part(mention.code),
        )
    return ("text", _normalize_text(mention.text), "")


def clinical_status_from_assertion(
    assertion: ClinicalAssertion | Mapping[str, object],
) -> ProblemClinicalStatus:
    """Map a reconciled clinical assertion to advisory problem-list status.

    Negation refutes patient assertions. Uncertainty, hypothetical temporality,
    and non-patient experiencers produce ``"unconfirmed"`` because they are not
    patient conditions asserted as present. Recent affirmed assertions map to
    ``"active"`` and historical affirmed assertions map to ``"inactive"``.
    """

    if isinstance(assertion, ClinicalAssertion):
        normalized = ClinicalAssertion(
            temporality=_normalize_temporality(assertion.temporality),
            certainty=_normalize_certainty(assertion.certainty),
            negation=_normalize_negation(assertion.negation),
            experiencer=_normalize_experiencer(assertion.experiencer),
        )
    elif isinstance(assertion, Mapping):
        normalized = ClinicalAssertion(
            temporality=_normalize_temporality(assertion.get("temporality")),
            certainty=_normalize_certainty(assertion.get("certainty")),
            negation=_normalize_negation(assertion.get("negation")),
            experiencer=_normalize_experiencer(assertion.get("experiencer")),
        )
    else:
        raise TypeError("assertion must be a ClinicalAssertion or mapping")

    if (
        normalized.experiencer is not None
        and normalized.experiencer != PATIENT_EXPERIENCER
    ):
        return UNCONFIRMED
    if normalized.negation != AFFIRMED:
        return REFUTED
    if normalized.certainty != CERTAIN:
        return UNCONFIRMED
    if normalized.temporality == RECENT:
        return ACTIVE
    if normalized.temporality == HISTORICAL:
        return INACTIVE
    if normalized.temporality == HYPOTHETICAL:
        return UNCONFIRMED
    return UNCONFIRMED


def _reconcile_status(mentions: Iterable[ProblemMention]) -> ProblemClinicalStatus:
    statuses = [
        clinical_status_from_assertion(
            ClinicalAssertion(
                temporality=mention.temporality,
                certainty=mention.certainty,
                negation=mention.negation,
                experiencer=mention.experiencer,
            )
        )
        for mention in mentions
    ]
    if ACTIVE in statuses:
        return ACTIVE
    if INACTIVE in statuses:
        return INACTIVE
    if UNCONFIRMED in statuses:
        return UNCONFIRMED
    return REFUTED


def deduplicate_problem_list(
    mentions: Iterable[ProblemMention | Mapping[str, object]],
) -> list[ReconciledProblem]:
    """Collapse candidate problem mentions into reconciled problem entries.

    Group identity is deterministic and intentionally conservative: mentions
    with a ``coref_entity_id`` are grouped by that document-level entity,
    mentions with both ``system`` and ``code`` are grouped by coded identity,
    and mentions without either signal fall back to normalized text with
    whitespace collapsed and casing folded. This does not infer that an uncoded
    text mention is the same concept as a coded mention.

    Clinical status reconciliation is an advisory heuristic with explicit
    precedence. Mentions are first mapped through
    ``clinical_status_from_assertion`` so reconciled negation, temporality,
    certainty, and experiencer axes drive status. ``"active"`` wins over
    ``"inactive"``, then ``"unconfirmed"``, then ``"refuted"``.

    Groups are emitted in first-seen input order, and source offsets are
    preserved in contributing mention order.

    Args:
        mentions: Candidate problem mentions as ``ProblemMention`` instances
            or mappings with text-like fields and optional ``system``, ``code``,
            ``offset`` or ``start``/``end``, ``negation``, and ``temporality``.
            ``coref_entity_id``/``entity_id``/``cluster_id`` may be supplied to
            deduplicate by document-level coreference.
            ``certainty`` and ``experiencer`` may also be supplied when the
            mention already carries reconciled assertion axes.

    Returns:
        A deterministic list of deduplicated problem entries with reconciled
        status, source offsets, and merged mention counts.
    """

    groups: dict[_IdentityKey, list[ProblemMention]] = {}
    order: list[_IdentityKey] = []

    for raw_mention in mentions:
        mention = _coerce_mention(raw_mention)
        key = _identity_key(mention)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(mention)

    reconciled: list[ReconciledProblem] = []
    for key in order:
        group = groups[key]
        first = group[0]
        has_code_identity = key[0] == "code"
        has_coref_identity = key[0] == "coref"
        reconciled.append(
            ReconciledProblem(
                text=first.text,
                normalized_text=_normalize_text(first.text),
                system=first.system if has_code_identity else None,
                code=first.code if has_code_identity else None,
                clinical_status=_reconcile_status(group),
                mention_count=len(group),
                source_offsets=tuple(
                    mention.offset for mention in group if mention.offset is not None
                ),
                coref_entity_id=first.coref_entity_id if has_coref_identity else None,
            )
        )

    return reconciled


__all__ = [
    "ACTIVE",
    "INACTIVE",
    "PROBLEM_CLINICAL_STATUS_VALUES",
    "PROBLEM_LIST_RECONCILIATION_ADVISORY",
    "ProblemClinicalStatus",
    "ProblemMention",
    "REFUTED",
    "ReconciledProblem",
    "SpanOffset",
    "UNCONFIRMED",
    "clinical_status_from_assertion",
    "deduplicate_problem_list",
]
