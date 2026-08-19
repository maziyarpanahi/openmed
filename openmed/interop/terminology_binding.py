"""Bind coded clinical spans to FHIR ``CodeableConcept`` values.

The binder is the lookup step between a grounded source span and a caller-
supplied OHDSI Athena index. It resolves only exact vocabulary/code pairs,
uses the shared FHIR system map, and falls back to source text when the
supplied index cannot prove a code. Athena indexes remain caller-owned; no
terminology content is downloaded, copied, or bundled here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openmed.clinical.exporters.codeable_concept import SYSTEM_URI
from openmed.clinical.exporters.codeable_concept_simple import codeable_concept

from .athena import AthenaVocabularyIndex

__all__ = ["RestrictedVocabularyBindingError", "bind_codeable_concept"]

_META_KEY = "_meta"
_BINDING_MISSES_KEY = "binding_misses"
_MISSING = object()

_CODE_FIELDS = (
    "code",
    "concept_code",
    "source_code",
    "code_value",
    "coding_code",
    "concept_id",
    "cui",
)
_VOCABULARY_FIELDS = (
    "vocabulary_id",
    "source_vocabulary_id",
    "system",
    "code_system",
    "coding_system",
    "vocabulary",
)
_TEXT_FIELDS = (
    "text",
    "surface",
    "entity_text",
    "normalized_text",
    "source_value",
    "word",
)
_RESTRICTED_VOCABULARY_IDS = frozenset({"cpt", "cpt4"})


class RestrictedVocabularyBindingError(ValueError):
    """Raised when a restricted Athena vocabulary lacks explicit opt-in."""


@dataclass(frozen=True)
class _SourceCode:
    """One source code extracted from a grounded span."""

    code: str
    vocabulary_id: str | None


def bind_codeable_concept(
    span: Any,
    athena_index: AthenaVocabularyIndex | Mapping[str, Any],
    *,
    vocabularies: Iterable[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a coded span to a FHIR R4 ``CodeableConcept``.

    ``span`` may be a :class:`GroundedSpan`, its JSON-shaped mapping, or a
    simple mapping/object with ``code`` and ``vocabulary_id`` fields. Grounded
    candidates and ``codes`` mappings are also accepted. Every source code is
    looked up exactly in ``athena_index``; a missing or unsupported code makes
    the whole result text-only so no partial or fabricated coding is emitted.

    Args:
        span: Coded source span carrying text and one or more source codes.
        athena_index: An index returned by :func:`load_athena_vocab`, or an
            equivalent caller-owned mapping.
        vocabularies: Optional allow-list of vocabulary IDs to search. A
            mapping additionally lets a caller provide a canonical system URI
            or shared vocabulary ID for a user-supplied vocabulary, for
            example ``{"CPT4": "http://www.ama-assn.org/go/cpt"}`` or
            ``{"RXTEST": "RXNORM"}``. Supplying a restricted vocabulary here
            is the explicit opt-in required to use it.

    Returns:
        A FHIR R4 ``CodeableConcept`` mapping with ``text`` and, when every
        source code resolves, one or more ``coding`` entries. Misses are kept
        in ``athena_index["_meta"]["binding_misses"]`` as PHI-free records.

    Raises:
        RestrictedVocabularyBindingError: If a restricted vocabulary is used
            without being explicitly included in ``vocabularies``.
        ValueError: If the index provenance says it is bundled or not
            user-supplied.
    """
    if not isinstance(athena_index, Mapping):
        raise TypeError("athena_index must be a mapping")

    _validate_index_provenance(athena_index)
    selected, explicit, system_overrides = _select_vocabularies(
        athena_index, vocabularies
    )
    restricted = _restricted_vocabulary_ids(athena_index, selected)
    text = _span_text(span)
    source_codes = _source_codes(span)
    if not source_codes:
        return {"text": text}

    codings: list[dict[str, Any]] = []
    misses: list[dict[str, str]] = []
    for source_code in source_codes:
        record, actual_vocabulary_id, miss_reason = _lookup_record(
            source_code,
            athena_index,
            selected,
        )
        vocabulary_id = actual_vocabulary_id or source_code.vocabulary_id or ""
        _ensure_vocabulary_allowed(vocabulary_id, restricted, explicit)

        if record is None:
            misses.append(
                {
                    "vocabulary_id": vocabulary_id,
                    "code": source_code.code,
                    "reason": miss_reason,
                }
            )
            continue

        record_vocabulary_id = _first_text((record,), _VOCABULARY_FIELDS)
        vocabulary_id = record_vocabulary_id or vocabulary_id
        _ensure_vocabulary_allowed(vocabulary_id, restricted, explicit)
        system = _canonical_system_uri(
            vocabulary_id,
            record,
            system_overrides,
        )
        if system is None:
            misses.append(
                {
                    "vocabulary_id": vocabulary_id,
                    "code": source_code.code,
                    "reason": "unsupported_vocabulary",
                }
            )
            continue

        coding: dict[str, Any] = {
            "system": system,
            "code": source_code.code,
        }
        display = _first_text((record,), ("concept_name", "display"))
        if display:
            coding["display"] = display
        codings.append(coding)

    if misses:
        _record_binding_misses(athena_index, misses)
        return {"text": text}

    return codeable_concept(codings, text=text)


def _lookup_record(
    source_code: _SourceCode,
    athena_index: Mapping[str, Any],
    selected: tuple[str, ...],
) -> tuple[Mapping[str, Any] | None, str | None, str]:
    code = source_code.code
    requested_vocabulary = source_code.vocabulary_id
    if requested_vocabulary:
        actual_vocabulary = _matching_vocabulary(
            requested_vocabulary,
            selected,
        )
        if actual_vocabulary is None:
            return None, requested_vocabulary, "vocabulary_not_selected"
        record = _record_for_code(athena_index.get(actual_vocabulary), code)
        if record is None:
            return None, actual_vocabulary, "not_found"
        return record, actual_vocabulary, ""

    matches: list[tuple[str, Mapping[str, Any]]] = []
    for vocabulary_id in selected:
        record = _record_for_code(athena_index.get(vocabulary_id), code)
        if record is not None:
            matches.append((vocabulary_id, record))
    if not matches:
        return None, None, "not_found"
    if len(matches) > 1:
        return None, None, "ambiguous_vocabulary"
    vocabulary_id, record = matches[0]
    return record, vocabulary_id, ""


def _record_for_code(value: Any, code: str) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    record = value.get(code)
    if isinstance(record, Mapping):
        return record
    normalized_code = code.casefold()
    for indexed_code, candidate in value.items():
        if str(indexed_code).strip().casefold() == normalized_code:
            if isinstance(candidate, Mapping):
                return candidate
            return None
    return None


def _select_vocabularies(
    athena_index: Mapping[str, Any],
    vocabularies: Iterable[str] | Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], frozenset[str], dict[str, str]]:
    available = tuple(
        sorted(
            (
                str(vocabulary_id)
                for vocabulary_id, records in athena_index.items()
                if vocabulary_id != _META_KEY and isinstance(records, Mapping)
            ),
            key=lambda value: (_normalize_identifier(value), value),
        )
    )
    available_by_normalized: dict[str, str] = {}
    for vocabulary_id in available:
        for alias in _vocabulary_aliases(vocabulary_id):
            available_by_normalized.setdefault(alias, vocabulary_id)

    if vocabularies is None:
        return available, frozenset(), {}

    if isinstance(vocabularies, Mapping):
        requested = tuple(vocabularies)
        values = vocabularies
    elif isinstance(vocabularies, (str, bytes)):
        requested = (str(vocabularies),)
        values = {}
    else:
        requested = tuple(str(value) for value in vocabularies)
        values = {}

    selected = tuple(
        available_by_normalized[normalized]
        for requested_id in requested
        if (normalized := _normalize_identifier(requested_id))
        and normalized in available_by_normalized
    )
    selected = tuple(dict.fromkeys(selected))
    explicit = frozenset(
        alias
        for requested_id in requested
        for alias in _vocabulary_aliases(requested_id)
    )
    overrides: dict[str, str] = {}
    for requested_id, value in values.items():
        uri = _system_uri_value(value)
        if uri is not None:
            for alias in _vocabulary_aliases(requested_id):
                overrides[alias] = uri
    return selected, explicit, overrides


def _span_text(span: Any) -> str:
    return _first_text(_span_sources(span), _TEXT_FIELDS)


def _source_codes(span: Any) -> tuple[_SourceCode, ...]:
    sources = _span_sources(span)
    extracted: list[_SourceCode] = []

    for source in sources:
        candidates = _value(source, "candidates")
        if candidates is not _MISSING and candidates:
            for candidate in _as_sequence(candidates):
                _append_source_code(extracted, candidate)

    if extracted:
        return tuple(extracted)

    for source in sources:
        codes = _value(source, "codes")
        if isinstance(codes, Mapping):
            for vocabulary_id, code in codes.items():
                if code is not None and str(code).strip():
                    extracted.append(
                        _SourceCode(
                            code=str(code).strip(),
                            vocabulary_id=str(vocabulary_id).strip() or None,
                        )
                    )

        for field in ("coding", "codings", "codeable_concept"):
            value = _value(source, field)
            if value is _MISSING or value is None:
                continue
            nested = value.get("coding") if isinstance(value, Mapping) else value
            for coding in _as_sequence(nested):
                _append_source_code(extracted, coding)

    if extracted:
        return tuple(_dedupe_source_codes(extracted))

    for source in sources:
        _append_source_code(extracted, source)
    return tuple(_dedupe_source_codes(extracted))


def _append_source_code(values: list[_SourceCode], source: Any) -> None:
    code = _first_text((source,), _CODE_FIELDS)
    if not code:
        return
    vocabulary_id = _first_text((source,), _VOCABULARY_FIELDS) or None
    values.append(_SourceCode(code=code, vocabulary_id=vocabulary_id))


def _dedupe_source_codes(values: Iterable[_SourceCode]) -> list[_SourceCode]:
    result: list[_SourceCode] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (
            _normalize_identifier(value.vocabulary_id or ""),
            value.code.casefold(),
        )
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _span_sources(span: Any) -> tuple[Any, ...]:
    sources = [span]
    for source in tuple(sources):
        for field in ("metadata", "meta"):
            metadata = _value(source, field)
            if metadata is not _MISSING and isinstance(metadata, Mapping):
                sources.append(metadata)
    return tuple(sources)


def _canonical_system_uri(
    vocabulary_id: str,
    record: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> str | None:
    normalized = _normalize_identifier(vocabulary_id)
    if normalized in overrides:
        return overrides[normalized]

    record_uri = _first_text((record,), ("system_uri", "code_system_uri"))
    if record_uri.startswith(("http://", "https://")):
        return record_uri

    for vocabulary, uri in SYSTEM_URI.items():
        if normalized in {
            _normalize_identifier(vocabulary),
            _normalize_identifier(uri),
        }:
            return uri
    return None


def _matching_vocabulary(
    vocabulary_id: str,
    selected: tuple[str, ...],
) -> str | None:
    aliases = _vocabulary_aliases(vocabulary_id)
    for selected_id in selected:
        if aliases.intersection(_vocabulary_aliases(selected_id)):
            return selected_id
    return None


def _vocabulary_aliases(value: Any) -> frozenset[str]:
    normalized = _normalize_identifier(value)
    if not normalized:
        return frozenset()
    aliases = {normalized}
    for vocabulary, uri in SYSTEM_URI.items():
        if normalized in {
            _normalize_identifier(vocabulary),
            _normalize_identifier(uri),
        }:
            aliases.update(
                {
                    _normalize_identifier(vocabulary),
                    _normalize_identifier(uri),
                }
            )
    return frozenset(aliases)


def _restricted_vocabulary_ids(
    athena_index: Mapping[str, Any],
    selected: tuple[str, ...],
) -> frozenset[str]:
    restricted = {
        _normalize_identifier(vocabulary_id)
        for vocabulary_id in selected
        if _normalize_identifier(vocabulary_id) in _RESTRICTED_VOCABULARY_IDS
    }
    meta = athena_index.get(_META_KEY)
    if not isinstance(meta, Mapping):
        return frozenset(restricted)

    for source in (meta, meta.get("provenance")):
        if not isinstance(source, Mapping):
            continue
        restricted.update(_normalized_values(source.get("restricted_vocabularies")))
        if source.get("restricted") is True:
            restricted.update(_normalize_identifier(vocab_id) for vocab_id in selected)
        license_metadata = source.get("license")
        if (
            isinstance(license_metadata, Mapping)
            and license_metadata.get("restricted") is True
        ):
            restricted.update(_normalize_identifier(vocab_id) for vocab_id in selected)
    return frozenset(restricted)


def _ensure_vocabulary_allowed(
    vocabulary_id: str,
    restricted: frozenset[str],
    explicit: frozenset[str],
) -> None:
    normalized = _normalize_identifier(vocabulary_id)
    if (
        normalized in _RESTRICTED_VOCABULARY_IDS or normalized in restricted
    ) and normalized not in explicit:
        raise RestrictedVocabularyBindingError(
            f"restricted Athena vocabulary {vocabulary_id!r} requires explicit "
            "inclusion in vocabularies"
        )


def _validate_index_provenance(athena_index: Mapping[str, Any]) -> None:
    meta = athena_index.get(_META_KEY)
    if not isinstance(meta, Mapping):
        return
    provenance = meta.get("provenance")
    if not isinstance(provenance, Mapping):
        return
    if provenance.get("bundled") is True:
        raise ValueError("Athena vocabulary content must remain user-supplied")
    if provenance.get("user_supplied") is False:
        raise ValueError("Athena vocabulary index must be user-supplied")


def _record_binding_misses(
    athena_index: Mapping[str, Any],
    misses: Iterable[Mapping[str, str]],
) -> None:
    if not isinstance(athena_index, dict):
        return
    meta = athena_index.get(_META_KEY)
    if meta is None:
        meta = {}
        athena_index[_META_KEY] = meta
    if not isinstance(meta, dict):
        return
    recorded = meta.setdefault(_BINDING_MISSES_KEY, [])
    if not isinstance(recorded, list):
        recorded = list(recorded) if isinstance(recorded, Iterable) else []
        meta[_BINDING_MISSES_KEY] = recorded
    for miss in misses:
        item = dict(miss)
        if item not in recorded:
            recorded.append(item)


def _normalized_values(value: Any) -> set[str]:
    if value is None or isinstance(value, (str, bytes)):
        values = (value,) if value is not None else ()
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
    return {
        normalized for item in values if (normalized := _normalize_identifier(item))
    }


def _system_uri_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("system_uri") or value.get("uri")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    for vocabulary, uri in SYSTEM_URI.items():
        if _normalize_identifier(value) in {
            _normalize_identifier(vocabulary),
            _normalize_identifier(uri),
        }:
            return uri
    if not value.startswith(("http://", "https://")):
        raise ValueError("vocabulary system URI must be an absolute HTTP(S) URI")
    return value


def _first_text(sources: Iterable[Any], fields: Iterable[str]) -> str:
    for source in sources:
        for field in fields:
            value = _value(source, field)
            if value is not _MISSING and value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _value(source: Any, field: str) -> Any:
    if isinstance(source, Mapping) and field in source:
        return source[field]
    if hasattr(source, field):
        return getattr(source, field)
    return _MISSING


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None or value is _MISSING:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _normalize_identifier(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in str(value or "").strip()
        if character.isalnum()
    )
