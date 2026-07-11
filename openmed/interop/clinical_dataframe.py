"""Clinical extraction rows shared by dataframe and SQL adapters.

The helpers in this module deliberately keep the default path dependency-light
and offline. Callers can inject richer extractors or request vocabulary linkers,
but the built-in extractor uses only local deterministic patterns and a tiny
free-vocabulary code map for common notebook fixtures.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.exporters.flat_table import FLAT_TABLE_COLUMNS, flatten_entities

ClinicalExtractor = Callable[..., Any]

_RXNORM_URI = "http://www.nlm.nih.gov/research/umls/rxnorm"
_ICD10CM_URI = "http://hl7.org/fhir/sid/icd-10-cm"

_PHI_COLUMN_HINTS = (
    "phi",
    "raw",
    "unredacted",
    "identifier",
    "identifiers",
    "patient_name",
    "mrn",
    "ssn",
    "dob",
    "birth",
    "email",
    "phone",
)


@dataclass(frozen=True)
class _ConceptPattern:
    label: str
    normalized_text: str
    pattern: re.Pattern[str]
    system: str
    code: str
    display: str


_DEFAULT_CONCEPTS: tuple[_ConceptPattern, ...] = (
    _ConceptPattern(
        "medication",
        "metformin",
        re.compile(r"\bmetformin\b", re.IGNORECASE),
        _RXNORM_URI,
        "6809",
        "Metformin",
    ),
    _ConceptPattern(
        "medication",
        "aspirin",
        re.compile(r"\baspirin\b", re.IGNORECASE),
        _RXNORM_URI,
        "1191",
        "Aspirin",
    ),
    _ConceptPattern(
        "condition",
        "type 2 diabetes",
        re.compile(r"\b(?:type\s+2\s+)?diabetes\b", re.IGNORECASE),
        _ICD10CM_URI,
        "E11.9",
        "Type 2 diabetes mellitus without complications",
    ),
    _ConceptPattern(
        "condition",
        "asthma",
        re.compile(r"\basthma\b", re.IGNORECASE),
        _ICD10CM_URI,
        "J45.909",
        "Unspecified asthma, uncomplicated",
    ),
    _ConceptPattern(
        "condition",
        "pneumonia",
        re.compile(r"\bpneumonia\b", re.IGNORECASE),
        _ICD10CM_URI,
        "J18.9",
        "Pneumonia, unspecified organism",
    ),
    _ConceptPattern(
        "finding",
        "fever",
        re.compile(r"\bfever\b", re.IGNORECASE),
        _ICD10CM_URI,
        "R50.9",
        "Fever, unspecified",
    ),
    _ConceptPattern(
        "finding",
        "cough",
        re.compile(r"\bcough\b", re.IGNORECASE),
        _ICD10CM_URI,
        "R05.9",
        "Cough, unspecified",
    ),
    _ConceptPattern(
        "finding",
        "chest pain",
        re.compile(r"\bchest\s+pain\b", re.IGNORECASE),
        _ICD10CM_URI,
        "R07.9",
        "Chest pain, unspecified",
    ),
)


def extract_records(
    records: Iterable[Mapping[str, Any]],
    text_column: str,
    *,
    extractor: ClinicalExtractor | None = None,
    extractor_kwargs: Mapping[str, Any] | None = None,
    systems: Sequence[str] | None = None,
    top_k: int = 1,
    warn_on_phi: bool = True,
) -> list[dict[str, Any]]:
    """Extract grounded clinical rows from mapping records.

    Args:
        records: Table rows as mappings.
        text_column: Column containing already-de-identified clinical text.
        extractor: Optional callable receiving ``text`` and returning entities.
        extractor_kwargs: Extra keyword arguments passed to ``extractor``.
        systems: Optional grounding systems to run through registered linkers.
        top_k: Maximum candidates per linker when ``systems`` is provided.
        warn_on_phi: Emit a privacy warning for likely raw-PHI column names.

    Returns:
        Row dictionaries ordered exactly as
        :data:`openmed.clinical.exporters.flat_table.FLAT_TABLE_COLUMNS`.
    """

    if warn_on_phi:
        warn_if_likely_phi_column(text_column)

    rows: list[dict[str, Any]] = []
    for record in records:
        value = record.get(text_column)
        if not isinstance(value, str) or not value.strip():
            continue
        rows.extend(
            extract_text_rows(
                value,
                extractor=extractor,
                extractor_kwargs=extractor_kwargs,
                systems=systems,
                top_k=top_k,
            )
        )
    return rows


def extract_text_rows(
    text: str,
    *,
    extractor: ClinicalExtractor | None = None,
    extractor_kwargs: Mapping[str, Any] | None = None,
    systems: Sequence[str] | None = None,
    top_k: int = 1,
) -> list[dict[str, Any]]:
    """Return flat-table clinical rows for one already-de-identified note."""

    raw_entities = _entities_from_result(
        (extractor or default_clinical_extractor)(text, **dict(extractor_kwargs or {}))
    )
    entities = [
        _with_grounding(entity, systems=systems, top_k=top_k) for entity in raw_entities
    ]
    return flatten_entities(entities)


def default_clinical_extractor(text: str) -> list[dict[str, Any]]:
    """Extract a small local set of grounded clinical entities.

    This is intentionally conservative: it gives notebooks a runnable offline
    first cell and keeps richer model-backed extraction injectable by callers.
    """

    entities: list[dict[str, Any]] = []
    occupied: set[tuple[int, int]] = set()

    for concept in _DEFAULT_CONCEPTS:
        for match in concept.pattern.finditer(text):
            span = match.span()
            if span in occupied:
                continue
            occupied.add(span)
            entities.append(
                {
                    "label": concept.label,
                    "normalized_text": concept.normalized_text,
                    "start": match.start(),
                    "end": match.end(),
                    "coding": {
                        "system": concept.system,
                        "code": concept.code,
                        "display": concept.display,
                    },
                    "context": {
                        "negation": _negation_for_span(text, match.start()),
                        "certainty": "certain",
                    },
                }
            )

    return sorted(entities, key=lambda entity: (entity["start"], entity["end"]))


def warn_if_likely_phi_column(column: str) -> None:
    """Warn when a column name suggests raw identifiers may be present."""

    normalized = str(column).strip().lower().replace("-", "_")
    if any(hint in normalized for hint in _PHI_COLUMN_HINTS):
        warnings.warn(
            "OpenMed dataframe clinical extraction expects already-de-identified "
            f"text; column {column!r} looks like it may contain raw identifiers.",
            UserWarning,
            stacklevel=3,
        )


def _entities_from_result(result: Any) -> tuple[Any, ...]:
    if result is None:
        return ()
    if isinstance(result, Mapping):
        nested = result.get("entities")
        if nested is not None:
            return _entities_from_result(nested)
    nested = getattr(result, "entities", None)
    if nested is not None:
        return _entities_from_result(nested)
    if isinstance(result, (str, bytes)):
        return ()
    try:
        return tuple(result)
    except TypeError:
        return (result,)


def _with_grounding(
    entity: Any,
    *,
    systems: Sequence[str] | None,
    top_k: int,
) -> Any:
    if not systems:
        return entity

    text = _entity_text(entity)
    if not text:
        return entity

    candidates: list[Any] = []
    canonical_label = _entity_label(entity)
    for system in systems:
        candidates.extend(
            _link_candidates(
                system,
                text,
                canonical_label=canonical_label,
                top_k=top_k,
            )
        )

    if not candidates:
        return entity

    mapping = _entity_mapping(entity)
    mapping["candidates"] = tuple(candidates)
    return mapping


def _link_candidates(
    system: str,
    text: str,
    *,
    canonical_label: str,
    top_k: int,
) -> tuple[Any, ...]:
    from openmed.clinical.grounding import get_index, get_linker

    linker_factory = get_linker(system)
    linker = linker_factory(get_index(system))
    return tuple(
        linker.link(
            text,
            canonical_label=canonical_label or None,
            top_k=top_k,
        )
    )


def _entity_mapping(entity: Any) -> dict[str, Any]:
    if isinstance(entity, Mapping):
        return dict(entity)
    mapping: dict[str, Any] = {}
    for name in (
        "label",
        "entity_label",
        "entity_type",
        "text",
        "normalized_text",
        "start",
        "end",
        "metadata",
        "context",
        "coding",
        "codings",
    ):
        if hasattr(entity, name):
            mapping[name] = getattr(entity, name)
    return mapping


def _entity_text(entity: Any) -> str:
    mapping = _entity_mapping(entity)
    for name in ("normalized_text", "text", "entity_text", "word", "surface"):
        value = mapping.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _entity_label(entity: Any) -> str:
    mapping = _entity_mapping(entity)
    for name in ("entity_label", "label", "entity_type", "entity_group", "entity"):
        value = mapping.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _negation_for_span(text: str, start: int) -> str:
    window = text[max(0, start - 24) : start].lower()
    if re.search(r"\b(no|denies|without)\s+$", window):
        return "negated"
    return "affirmed"


__all__ = [
    "ClinicalExtractor",
    "FLAT_TABLE_COLUMNS",
    "default_clinical_extractor",
    "extract_records",
    "extract_text_rows",
    "warn_if_likely_phi_column",
]
