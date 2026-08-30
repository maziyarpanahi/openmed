"""Shared helpers for table-specific OMOP exporters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, TypeAlias

from openmed.clinical.grounding.assertion_grounding import (
    GROUNDING_HYPOTHETICAL,
    GROUNDING_NON_PATIENT,
    GROUNDING_REFUTED,
    assertion_grounding_status,
)
from openmed.clinical.grounding.types import Candidate, GroundedSpan
from openmed.interop.omop import deterministic_omop_id

__all__ = [
    "ConceptResolver",
    "DOMAIN_BY_LABEL",
    "DOMAIN_BY_SYSTEM",
    "ResolvedConcept",
    "assertion_status",
    "context_value",
    "date_value",
    "domain_for_span",
    "foreign_key",
    "iter_spans",
    "resolve_concept",
    "source_value",
    "span_candidate",
    "span_is_exportable",
    "table_row_id",
]

DOMAIN_BY_LABEL = {
    "CONDITION": "Condition",
    "MEDICATION": "Drug",
    "LAB_TEST": "Measurement",
    "PROCEDURE": "Procedure",
}
DOMAIN_BY_SYSTEM = {
    "HPO": "Condition",
    "ICD10CM": "Condition",
    "ICD11": "Condition",
    "MESH": "Condition",
    "SNOMED": "Condition",
    "UMLS": "Condition",
    "RXNORM": "Drug",
    "LOINC": "Measurement",
}

ConceptResolver: TypeAlias = Mapping[Any, Any] | Callable[..., Any]


@dataclass(frozen=True)
class ResolvedConcept:
    """Source and standard concept identifiers returned by a resolver."""

    standard_concept_id: int
    source_concept_id: int
    source_code: str
    source_vocabulary_id: str


def iter_spans(
    grounded: GroundedSpan | Iterable[GroundedSpan],
) -> tuple[GroundedSpan, ...]:
    """Validate and normalize one span or an iterable of spans."""

    spans = (grounded,) if isinstance(grounded, GroundedSpan) else tuple(grounded)
    if any(not isinstance(span, GroundedSpan) for span in spans):
        raise TypeError("OMOP exporters expect GroundedSpan objects")
    return spans


def span_candidate(span: GroundedSpan) -> Candidate | None:
    """Return the highest-ranked candidate selected for ``span``."""

    return span.candidates[0] if span.candidates else None


def domain_for_span(span: GroundedSpan) -> str:
    """Infer the OMOP domain from a canonical label or coding system."""

    label = (span.canonical_label or "").strip().upper()
    if label in DOMAIN_BY_LABEL:
        return DOMAIN_BY_LABEL[label]
    candidate = span_candidate(span)
    if candidate is not None:
        domain = DOMAIN_BY_SYSTEM.get(candidate.system.strip().upper())
        if domain is not None:
            return domain
    raise ValueError("cannot infer an OMOP domain; provide a supported canonical_label")


def assertion_status(span: GroundedSpan) -> str | None:
    """Return the derived assertion status, or ``None`` when unset."""

    if span.assertion is None:
        return None
    return assertion_grounding_status(span.assertion).status


def span_is_exportable(
    span: GroundedSpan,
    *,
    include_refuted: bool = False,
) -> bool:
    """Apply the conservative OMOP assertion policy to one span.

    Refuted conditions are excluded by default. Hypothetical and non-patient
    findings are always withheld because emitting either as a patient row would
    claim a clinical fact that the assertion layer did not establish.
    """

    status = assertion_status(span)
    if status is None:
        return True
    if status == GROUNDING_REFUTED:
        return include_refuted
    return status not in {GROUNDING_HYPOTHETICAL, GROUNDING_NON_PATIENT}


def _metadata_mappings(span: GroundedSpan) -> tuple[Mapping[str, Any], ...]:
    metadata = span.metadata
    mappings: list[Mapping[str, Any]] = [metadata]
    for key in ("omop", "omop_context", "export"):
        nested = metadata.get(key)
        if isinstance(nested, Mapping):
            mappings.append(nested)
    label = (span.canonical_label or "").strip().casefold()
    if label:
        nested = metadata.get(label)
        if isinstance(nested, Mapping):
            mappings.append(nested)
    return tuple(mappings)


def context_value(span: GroundedSpan, *names: str) -> Any:
    """Return the first non-null value from span export metadata."""

    for mapping in _metadata_mappings(span):
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
    return None


def date_value(value: Any) -> str | None:
    """Normalize a date-like value to the string form accepted by CDM rows."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def foreign_key(value: Any, *, namespace: str) -> int | None:
    """Coerce a caller-provided CDM foreign key without inventing text IDs."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{namespace} must be an integer, string, or None")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return deterministic_omop_id(namespace, stripped)
    raise TypeError(f"{namespace} must be an integer, string, or None")


def concept_id(value: Any, *, name: str, default: int | None = None) -> int | None:
    """Validate an optional OMOP concept identifier."""

    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or None")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer or None") from exc
        if parsed < 0:
            raise ValueError(f"{name} must be non-negative")
        return parsed
    raise TypeError(f"{name} must be an integer or None")


def source_value(
    span: GroundedSpan,
    *,
    explicit: Any = None,
    fallback_code: str = "",
) -> str:
    """Return the source text/code retained in an OMOP ``*_source_value``."""

    value = explicit
    if value is None:
        value = context_value(span, "source_value", "source_text", "lexical_variant")
    if value is None:
        value = span.text or fallback_code
    return str(value)


def table_row_id(
    table: str,
    span: GroundedSpan,
    *,
    index: int,
    document_id: str,
    person_id: int | None,
    visit_occurrence_id: int | None,
    concept_id: int,
) -> int:
    """Return a deterministic primary key for a table row."""

    return deterministic_omop_id(
        table,
        document_id,
        person_id,
        visit_occurrence_id,
        span.start,
        span.end,
        span.text,
        concept_id,
        index,
    )


def _resolver_input(span: GroundedSpan, candidate: Candidate | None) -> dict[str, Any]:
    domain = domain_for_span(span)
    return {
        "text": span.text,
        "normalized_text": span.text,
        "start": span.start,
        "end": span.end,
        "label": span.canonical_label,
        "entity_label": domain,
        "domain": domain,
        "code": candidate.code if candidate else "",
        "concept_code": candidate.code if candidate else "",
        "system": candidate.system if candidate else "",
        "vocabulary_id": candidate.system if candidate else "",
        "source_value": span.text,
        "metadata": dict(span.metadata),
    }


def resolve_concept(
    span: GroundedSpan,
    resolver: ConceptResolver | Any | None,
) -> ResolvedConcept:
    """Resolve a grounded candidate through an injected Athena-compatible API."""

    candidate = span_candidate(span)
    source_code = candidate.code if candidate else ""
    source_vocabulary = candidate.system if candidate else ""
    if resolver is None or candidate is None:
        return ResolvedConcept(0, 0, source_code, source_vocabulary)

    raw = _resolve_raw(span, candidate, resolver)
    return _coerce_resolved(raw, candidate)


def _resolve_raw(
    span: GroundedSpan,
    candidate: Candidate,
    resolver: ConceptResolver | Any,
) -> Any:
    if isinstance(resolver, Mapping):
        keys: tuple[Any, ...] = (
            (candidate.system, candidate.code),
            (candidate.system.upper(), candidate.code),
            (candidate.system.casefold(), candidate.code),
            candidate.code,
        )
        for key in keys:
            if key in resolver:
                return resolver[key]
        return None

    resolver_input = _resolver_input(span, candidate)
    route_span = getattr(resolver, "route_span", None)
    if callable(route_span):
        result = _try_calls(
            route_span,
            ((resolver_input,), (span,)),
        )
        if result is not _MISSING:
            return result

    resolve = getattr(resolver, "resolve", None)
    if callable(resolve):
        result = _try_calls(
            resolve,
            (
                (resolver_input,),
                (resolver_input, {"domain": domain_for_span(span).casefold()}),
                (span, candidate),
                (candidate.system, candidate.code),
                (candidate.code,),
            ),
        )
        if result is not _MISSING:
            return result

    for method_name in ("resolve_code", "lookup"):
        method = getattr(resolver, method_name, None)
        if callable(method):
            result = _try_calls(
                method,
                (
                    (candidate.system, candidate.code),
                    (candidate.code,),
                    (resolver_input,),
                ),
            )
            if result is not _MISSING:
                return result

    if callable(resolver):
        result = _try_calls(
            resolver,
            (
                (span, candidate),
                (span,),
                (resolver_input,),
                (resolver_input, {"domain": domain_for_span(span).casefold()}),
                (candidate.system, candidate.code),
                (candidate.code,),
            ),
        )
        if result is not _MISSING:
            return result
    return None


_MISSING = object()


def _try_calls(method: Callable[..., Any], calls: Sequence[tuple[Any, ...]]) -> Any:
    for args in calls:
        try:
            if len(args) == 2 and isinstance(args[1], Mapping):
                return method(args[0], **args[1])
            return method(*args)
        except (KeyError, TypeError):
            continue
    return _MISSING


def _coerce_resolved(raw: Any, candidate: Candidate) -> ResolvedConcept:
    if raw is None:
        return ResolvedConcept(0, 0, candidate.code, candidate.system)
    if isinstance(raw, bool):
        raise TypeError("concept resolver must return a non-negative integer or record")
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError("concept resolver IDs must be non-negative")
        return ResolvedConcept(raw, 0, candidate.code, candidate.system)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) == 2:
            standard = concept_id(raw[0], name="standard_concept_id", default=0) or 0
            source = concept_id(raw[1], name="source_concept_id", default=0) or 0
            return ResolvedConcept(standard, source, candidate.code, candidate.system)
        raise ValueError(
            "concept resolver sequences must contain standard and source IDs"
        )

    standard = _record_value(
        raw,
        "target_concept_id",
        "standard_concept_id",
        "concept_id",
        "omop_concept_id",
    )
    source = _record_value(raw, "source_concept_id", "source_id")
    standard_id = concept_id(standard, name="standard_concept_id", default=0) or 0
    source_id = concept_id(source, name="source_concept_id", default=0) or 0
    source_code = str(
        _record_value(raw, "source_code", "concept_code", "code") or candidate.code
    )
    source_vocabulary = str(
        _record_value(raw, "source_vocabulary_id", "vocabulary_id", "system")
        or candidate.system
    )
    return ResolvedConcept(standard_id, source_id, source_code, source_vocabulary)


def _record_value(record: Any, *names: str) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return None
