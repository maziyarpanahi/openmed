"""Resolve Usagi source codes to caller-supplied OMOP concepts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from .athena import AthenaVocabularyIndex, UsagiMapping

__all__ = ["ConceptResolution", "UsagiConceptResolver"]

_META_KEYS = frozenset({"_meta", "meta", "metadata"})
_MAPS_TO_KEYS = frozenset(
    {"mapsto", "mapstoconceptid", "targetconceptid", "mappedconceptid"}
)
_RELATIONSHIP_KEYS = frozenset(
    {"relationships", "conceptrelationships", "relations", "mapsto"}
)
_RELATIONSHIP_NAME_FIELDS = (
    "relationship_id",
    "relationshipId",
    "relationship",
    "relationship_name",
    "relationshipName",
)
_SOURCE_ID_FIELDS = (
    "source_concept_id",
    "sourceConceptId",
    "concept_id_1",
    "conceptId1",
)
_TARGET_ID_FIELDS = (
    "target_concept_id",
    "targetConceptId",
    "concept_id_2",
    "conceptId2",
    "standard_concept_id",
    "standardConceptId",
    "mapped_concept_id",
    "mappedConceptId",
    "concept_id",
    "conceptId",
)
_CODE_FIELDS = ("concept_code", "conceptCode", "code")
_VOCABULARY_FIELDS = ("vocabulary_id", "vocabularyId", "system")
_MISSING = object()


@dataclass(frozen=True)
class ConceptResolution:
    """Resolved standard concept and source/metadata needed by OMOP export."""

    concept_id: int
    source_concept_id: int
    standard_concept: str | None
    mapped: bool
    vocabulary_id: str | None = None
    domain_id: str | None = None

    def __post_init__(self) -> None:
        """Keep the result consistent with OMOP's zero-as-unmapped contract."""

        if type(self.concept_id) is not int or self.concept_id < 0:
            raise ValueError("concept_id must be a non-negative integer")
        if type(self.source_concept_id) is not int or self.source_concept_id < 0:
            raise ValueError("source_concept_id must be a non-negative integer")
        if type(self.mapped) is not bool:
            raise TypeError("mapped must be a boolean")
        if self.mapped != (self.concept_id > 0):
            raise ValueError("mapped must agree with concept_id")

    @property
    def is_mapped(self) -> bool:
        """Return whether a positive standard concept was resolved."""

        return self.mapped

    @property
    def target_vocabulary_id(self) -> str | None:
        """Return the target vocabulary ID under the OMOP field name."""

        return self.vocabulary_id

    @property
    def target_domain_id(self) -> str | None:
        """Return the target domain ID under the OMOP field name."""

        return self.domain_id

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolution record."""

        return asdict(self)


@dataclass(frozen=True)
class _Record:
    vocabulary_id: str
    concept_code: str
    values: Mapping[str, Any]
    concept_id: int


class UsagiConceptResolver:
    """Resolve source codes with Usagi precedence and optional Athena metadata.

    ``usagi_mapping`` is normally the result of
    :func:`openmed.interop.athena.load_usagi_mapping`; keys may be bare source
    codes or ``"VOCABULARY:CODE"``. ``athena_index`` may be the result of
    :func:`openmed.interop.athena.load_athena_vocab` and may additionally
    contain synthetic ``Maps to`` relationship records.

    The instance is also callable with the ``(GroundedSpan, Candidate)``
    contract accepted by the clinical OMOP exporter. That form returns a
    positive concept ID or ``None``; :meth:`resolve` returns full metadata.
    """

    def __init__(
        self,
        usagi_mapping: UsagiMapping | Mapping[Any, Any],
        athena_index: AthenaVocabularyIndex | Mapping[Any, Any] | None = None,
    ) -> None:
        if not isinstance(usagi_mapping, Mapping):
            raise TypeError("usagi_mapping must be a mapping")
        if athena_index is not None and not isinstance(athena_index, Mapping):
            raise TypeError("athena_index must be a mapping or None")

        self._usagi = {
            _usagi_key(key): concept_id
            for key, value in usagi_mapping.items()
            if not _is_meta_key(key)
            and (concept_id := _positive_int(value)) is not None
        }
        self._by_key: dict[tuple[str, str], _Record] = {}
        self._by_code: dict[str, list[_Record]] = {}
        self._by_id: dict[int, _Record] = {}
        self._maps_to_by_id: dict[int, int] = {}
        self._maps_to_by_key: dict[tuple[str, str], int] = {}
        self._index_athena(athena_index)

        self._usagi_provenance = _provenance_for_usagi(usagi_mapping, len(self._usagi))
        self._athena_provenance = _provenance_for_athena(
            athena_index,
            len(self._by_key),
            {record.vocabulary_id for record in self._by_key.values()},
        )
        self._mapping_hash = _mapping_hash(
            self._usagi,
            self._by_key,
            self._maps_to_by_key,
            self._maps_to_by_id,
        )

    @property
    def mapping_hash(self) -> str:
        """Return the stable SHA-256 hash of the mapping set."""

        return self._mapping_hash

    @property
    def mapping_set_hash(self) -> str:
        """Return :attr:`mapping_hash` under an explicit versioning name."""

        return self._mapping_hash

    @property
    def provenance_hash(self) -> str:
        """Return the hash suitable for pinning this vocabulary set."""

        return self._mapping_hash

    @property
    def vocabulary_version(self) -> str:
        """Return the mapping hash as an OMOP vocabulary version pin."""

        return self._mapping_hash

    @property
    def provenance(self) -> dict[str, Any]:
        """Return loaded Usagi/Athena provenance and the mapping hash."""

        return {
            "usagi": deepcopy(self._usagi_provenance),
            "athena": deepcopy(self._athena_provenance),
            "mapping_hash": self._mapping_hash,
        }

    def provenance_record(self) -> dict[str, Any]:
        """Return :attr:`provenance` for an export manifest."""

        return self.provenance

    def resolve(self, system: str | None, source_code: str) -> ConceptResolution:
        """Resolve ``(system, source_code)`` to a standard concept.

        Usagi is authoritative when it contains the source. Otherwise an
        Athena ``Maps to`` target or already-standard source concept is used.
        A miss returns concept ID ``0`` and never fabricates a target.
        """

        system_value = _clean(system)
        code_value = _clean(source_code)
        source = self._record_for(system_value, code_value)
        source_id = source.concept_id if source is not None else 0

        target_id = self._usagi.get(_usagi_key((system_value, code_value)))
        if target_id is None:
            target_id = self._usagi.get(_usagi_key(code_value))
        if target_id is None and source is not None:
            target_id = self._maps_to(source)
        if target_id is None and source is not None and _is_standard(source.values):
            target_id = source.concept_id

        if target_id is None:
            return ConceptResolution(0, source_id, None, False)
        target = self._by_id.get(target_id)
        return _resolution(target_id, source_id, target)

    def __call__(
        self,
        span_or_system: Any,
        candidate_or_code: Any = None,
    ) -> int | None | ConceptResolution:
        """Adapt :meth:`resolve` to the clinical OMOP exporter callable shape."""

        if isinstance(span_or_system, str) and isinstance(candidate_or_code, str):
            return self.resolve(span_or_system, candidate_or_code)

        candidate = candidate_or_code
        system = _value(candidate, "system")
        code = _value(candidate, "code")
        if not isinstance(system, str) or not isinstance(code, str):
            candidates = _value(span_or_system, "candidates")
            if isinstance(candidates, Iterable) and not isinstance(
                candidates, (str, bytes, Mapping)
            ):
                first = next(iter(candidates), None)
                system = _value(first, "system")
                code = _value(first, "code")
        if not isinstance(system, str) or not isinstance(code, str):
            return None
        resolution = self.resolve(system, code)
        return resolution.concept_id if resolution.mapped else None

    def _index_athena(self, index: Mapping[Any, Any] | None) -> None:
        if index is None:
            return
        for top_key, value in index.items():
            if _is_meta_key(top_key):
                continue
            if _name(top_key) in _RELATIONSHIP_KEYS:
                self._index_relationships(value)
            elif _looks_like_record(value):
                self._add_record(top_key, value)
            elif isinstance(value, Mapping):
                for code, record in value.items():
                    if _is_meta_key(code):
                        continue
                    if isinstance(record, Mapping):
                        self._add_record((top_key, code), record)
                    elif (target_id := _positive_int(record)) is not None:
                        self._maps_to_by_key[(_norm(top_key), _norm(code))] = target_id
            elif (target_id := _positive_int(value)) is not None:
                parsed = _parse_key(top_key)
                if parsed is not None:
                    self._maps_to_by_key[(_norm(parsed[0]), _norm(parsed[1]))] = (
                        target_id
                    )

        for records in self._by_code.values():
            records.sort(
                key=lambda record: (record.vocabulary_id.casefold(), record.concept_id)
            )

    def _add_record(self, key: Any, values: Mapping[str, Any]) -> None:
        parsed = _parse_key(key)
        vocabulary_id = _first_text(values, _VOCABULARY_FIELDS)
        concept_code = _first_text(values, _CODE_FIELDS)
        if parsed is not None:
            vocabulary_id, concept_code = parsed
        vocabulary_id = vocabulary_id or _clean(key)
        concept_code = concept_code or _clean(key)
        concept_id = _first_positive_int(values, ("concept_id", "conceptId"))
        if concept_id is None:
            return
        record = _Record(vocabulary_id, concept_code, dict(values), concept_id)
        normalized_key = (_norm(vocabulary_id), _norm(concept_code))
        self._by_key[normalized_key] = record
        self._by_code.setdefault(normalized_key[1], []).append(record)
        existing = self._by_id.get(concept_id)
        if existing is None or _record_key(record) < _record_key(existing):
            self._by_id[concept_id] = record
        if (target_id := _direct_maps_to(values)) is not None:
            self._maps_to_by_id[concept_id] = target_id
            self._maps_to_by_key[normalized_key] = target_id

    def _index_relationships(self, relationships: Any) -> None:
        if isinstance(relationships, Mapping):
            if _looks_like_relationship(relationships):
                self._add_relationship(relationships)
                return
            for source, value in relationships.items():
                if isinstance(value, Mapping):
                    self._add_relationship(value, source_key=source)
                elif (target_id := _positive_int(value)) is not None:
                    if (source_id := _positive_int(source)) is not None:
                        self._maps_to_by_id[source_id] = target_id
                    elif (parsed := _parse_key(source)) is not None:
                        self._maps_to_by_key[(_norm(parsed[0]), _norm(parsed[1]))] = (
                            target_id
                        )
        elif isinstance(relationships, Iterable) and not isinstance(
            relationships, (str, bytes)
        ):
            for relationship in relationships:
                if isinstance(relationship, Mapping):
                    self._add_relationship(relationship)

    def _add_relationship(
        self,
        relationship: Mapping[str, Any],
        *,
        source_key: Any = None,
    ) -> None:
        name = _first_value(relationship, _RELATIONSHIP_NAME_FIELDS)
        if name is not _MISSING and _name(name) != "mapsto":
            return
        target_id = _first_positive_int(relationship, _TARGET_ID_FIELDS)
        if target_id is None:
            return
        source_id = _first_positive_int(relationship, _SOURCE_ID_FIELDS)
        if source_id is not None:
            self._maps_to_by_id[source_id] = target_id
        parsed = _parse_key(source_key)
        if parsed is None:
            source = _first_value(relationship, ("source", "source_key"))
            parsed = _parse_key(source)
        if parsed is None:
            code = _first_text(relationship, ("source_code", "sourceCode"))
            vocabulary = _first_text(
                relationship, ("source_vocabulary_id", "sourceVocabularyId")
            )
            if code:
                parsed = (vocabulary, code)
        if parsed is not None:
            self._maps_to_by_key[(_norm(parsed[0]), _norm(parsed[1]))] = target_id

    def _record_for(self, system: str, code: str) -> _Record | None:
        if not code:
            return None
        if system:
            return self._by_key.get((_norm(system), _norm(code)))
        records = self._by_code.get(_norm(code), ())
        return records[0] if records else None

    def _maps_to(self, source: _Record) -> int | None:
        return self._maps_to_by_id.get(source.concept_id) or self._maps_to_by_key.get(
            (_norm(source.vocabulary_id), _norm(source.concept_code))
        )


def _resolution(
    target_id: int,
    source_id: int,
    target: _Record | None,
) -> ConceptResolution:
    if target is None:
        return ConceptResolution(target_id, source_id, "S", True)
    return ConceptResolution(
        target_id,
        source_id,
        _standard_concept(target.values) or "S",
        True,
        vocabulary_id=target.vocabulary_id or None,
        domain_id=_first_text(target.values, ("domain_id", "domainId")) or None,
    )


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _norm(value: Any) -> str:
    return _clean(value).casefold()


def _name(value: Any) -> str:
    return "".join(character for character in _norm(value) if character.isalnum())


def _is_meta_key(value: Any) -> bool:
    return _norm(value) in _META_KEYS


def _usagi_key(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{_norm(value[0])}:{_norm(value[1])}"
    return _norm(value)


def _parse_key(value: Any) -> tuple[str, str] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        system, code = value
    else:
        text = _clean(value)
        if ":" not in text:
            return None
        system, code = text.split(":", 1)
    system, code = _clean(system), _clean(code)
    return (system, code) if system and code else None


def _value(item: Any, field: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(field, _MISSING)
    return getattr(item, field, _MISSING)


def _first_value(item: Mapping[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = _value(item, field)
        if value is not _MISSING and value is not None and _clean(value):
            return value
    return _MISSING


def _first_text(item: Mapping[str, Any], fields: Iterable[str]) -> str:
    value = _first_value(item, fields)
    return _clean(value) if value is not _MISSING else ""


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _first_positive_int(item: Mapping[str, Any], fields: Iterable[str]) -> int | None:
    for field in fields:
        value = _value(item, field)
        if value is not _MISSING and (result := _positive_int(value)) is not None:
            return result
    return None


def _standard_concept(record: Mapping[str, Any]) -> str | None:
    return _first_text(record, ("standard_concept", "standardConcept")) or None


def _is_standard(record: Mapping[str, Any]) -> bool:
    standard = _standard_concept(record)
    return standard is not None and standard.casefold() == "s"


def _target_id(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _first_positive_int(value, _TARGET_ID_FIELDS)
    return _positive_int(value)


def _direct_maps_to(record: Mapping[str, Any]) -> int | None:
    for key, value in record.items():
        if _name(key) in _MAPS_TO_KEYS and (target_id := _target_id(value)) is not None:
            return target_id
        if _name(key) not in _RELATIONSHIP_KEYS:
            continue
        if isinstance(value, Mapping):
            if _looks_like_relationship(value):
                target_id = _relationship_target(value)
                if target_id is not None:
                    return target_id
            for relationship_name, target in value.items():
                if (
                    _name(relationship_name) == "mapsto"
                    and (target_id := _target_id(target)) is not None
                ):
                    return target_id
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for relationship in value:
                if (
                    isinstance(relationship, Mapping)
                    and (target_id := _relationship_target(relationship)) is not None
                ):
                    return target_id
    return None


def _relationship_target(record: Mapping[str, Any]) -> int | None:
    name = _first_value(record, _RELATIONSHIP_NAME_FIELDS)
    if name is not _MISSING and _name(name) != "mapsto":
        return None
    return _first_positive_int(record, _TARGET_ID_FIELDS)


def _looks_like_record(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        field in value
        for field in ("concept_id", "conceptId", "concept_code", "conceptCode")
    )


def _looks_like_relationship(value: Mapping[str, Any]) -> bool:
    return _first_value(value, _RELATIONSHIP_NAME_FIELDS) is not _MISSING


def _record_key(record: _Record) -> tuple[str, str, int]:
    return (
        record.vocabulary_id.casefold(),
        record.concept_code.casefold(),
        record.concept_id,
    )


def _metadata(value: Mapping[Any, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    for key in _META_KEYS:
        metadata = value.get(key, _MISSING)
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return {}


def _provenance_for_usagi(
    mapping: Mapping[Any, Any], entry_count: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "loaded": True,
        "user_supplied": True,
        "bundled": False,
        "entry_count": entry_count,
    }
    metadata = _metadata(mapping)
    if metadata:
        result["metadata"] = metadata
        if isinstance(metadata.get("provenance"), Mapping):
            result["provenance"] = dict(metadata["provenance"])
    return result


def _provenance_for_athena(
    index: Mapping[Any, Any] | None,
    concept_count: int,
    vocabulary_ids: set[str],
) -> dict[str, Any]:
    if index is None:
        return {
            "loaded": False,
            "user_supplied": False,
            "bundled": False,
            "concept_count": 0,
            "vocabulary_ids": [],
        }
    result: dict[str, Any] = {
        "loaded": True,
        "user_supplied": True,
        "bundled": False,
        "concept_count": concept_count,
        "vocabulary_ids": sorted(vocabulary_ids),
    }
    metadata = _metadata(index)
    if metadata:
        result["metadata"] = metadata
        for key in ("source", "license", "provenance"):
            if key in metadata:
                result[key] = metadata[key]
    return result


def _mapping_hash(
    usagi: Mapping[str, int],
    records: Mapping[tuple[str, str], _Record],
    relationships: Mapping[tuple[str, str], int],
    relationships_by_id: Mapping[int, int],
) -> str:
    payload = {
        "usagi": sorted(usagi.items()),
        "athena": [
            {
                "vocabulary_id": record.vocabulary_id,
                "concept_code": record.concept_code,
                "record": _canonical(record.values),
            }
            for record in sorted(records.values(), key=_record_key)
        ],
        "relationships": sorted(relationships.items()),
        "relationships_by_id": sorted(relationships_by_id.items()),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_canonical(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
