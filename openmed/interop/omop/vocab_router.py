"""Vocabulary-aware domain routing and source-to-concept provenance.

This module resolves grounded source terms to OMOP standard concepts and their
target CDM domain tables using caller-supplied OHDSI Athena vocabulary indexes
and Usagi mappings (see :mod:`openmed.interop.athena`). It records the full
source-to-concept provenance for every routed term, including the vocabulary
version and the standard-concept metadata, and never fabricates a ``concept_id``
when no mapping exists: unmapped terms stay source-only with ``concept_id`` 0.

The router is deliberately decoupled from any writer. It emits plain provenance
records that mirror the OMOP ``SOURCE_TO_CONCEPT_MAP`` shape so the CDM loader
(:mod:`openmed.interop.omop.cdm_loader`) or any downstream exporter can persist
them without duplicating the routing logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ..athena import (
    AthenaVocabularyIndex,
    UsagiMapping,
    load_athena_vocab,
    load_usagi_mapping,
)
from .cdm_loader import (
    _DOMAIN_ALIASES,
    _DOMAIN_TABLES,
    UNMAPPED_CONCEPT_ID,
    UNMAPPED_VOCABULARY_ID,
    OmopDomain,
)

MappingStatus = Literal["mapped", "source_only"]

_ATHENA_META_KEY = "_meta"
_STANDARD_FLAG = "S"

# Reuse the loader's field probes for span-shaped inputs so callers can pass the
# same grounded records to the router and the loader.
_CODE_FIELDS = ("concept_code", "code", "source_code", "coding_code")
_VOCABULARY_FIELDS = ("source_vocabulary_id", "vocabulary_id", "system")
_DOMAIN_FIELDS = ("domain_id", "omop_domain", "domain", "entity_label", "label")
_DESCRIPTION_FIELDS = (
    "source_code_description",
    "lexical_variant",
    "normalized_text",
    "text",
    "entity_text",
    "concept_name",
)

_MISSING = object()


@dataclass(frozen=True)
class SourceToConceptMapping:
    """PHI-free source-to-concept routing decision with full provenance.

    A mapping records how one source term routed to an OMOP standard concept and
    a CDM domain table. ``target_concept_id`` is ``0`` when no standard concept
    could be resolved, in which case only source metadata is populated.
    """

    source_code: str
    source_vocabulary_id: str
    source_concept_id: int
    source_code_description: str | None
    target_concept_id: int
    target_vocabulary_id: str
    target_domain_id: str
    domain: OmopDomain | None
    cdm_table: str | None
    standard_concept: str | None
    vocabulary_version: str | None
    mapping_status: MappingStatus

    @property
    def is_mapped(self) -> bool:
        """Return ``True`` when a standard concept was resolved."""

        return self.target_concept_id != UNMAPPED_CONCEPT_ID

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable routing record."""

        return asdict(self)

    def to_source_to_concept_map_row(self) -> dict[str, Any]:
        """Return an OMOP ``SOURCE_TO_CONCEPT_MAP`` provenance row.

        The row omits the loader-owned surrogate key and note linkage columns so
        it can be merged into a full CDM ``source_to_concept_map`` record without
        conflicting with the loader's identifiers.
        """

        return {
            "source_code": self.source_code,
            "source_concept_id": self.source_concept_id,
            "source_vocabulary_id": self.source_vocabulary_id,
            "source_code_description": self.source_code_description,
            "target_concept_id": self.target_concept_id,
            "target_vocabulary_id": self.target_vocabulary_id,
            "valid_start_date": None,
            "valid_end_date": None,
            "invalid_reason": None if self.is_mapped else "UNMAPPED",
            "vocabulary_version": self.vocabulary_version,
        }


class VocabularyRouter:
    """Route source terms to OMOP domains using Athena and Usagi artifacts.

    Args:
        vocabulary: An Athena vocabulary index produced by
            :func:`openmed.interop.athena.load_athena_vocab`.
        usagi: Optional Usagi source-to-standard mapping produced by
            :func:`openmed.interop.athena.load_usagi_mapping`.
        vocabulary_version: Optional user-supplied vocabulary release version.
            A single string applies to every vocabulary; a mapping resolves the
            version per ``vocabulary_id`` (an empty-string key acts as default).
    """

    def __init__(
        self,
        vocabulary: AthenaVocabularyIndex | Mapping[str, Any],
        usagi: UsagiMapping | Mapping[str, int] | None = None,
        *,
        vocabulary_version: str | Mapping[str, str] | None = None,
    ) -> None:
        self._by_vocab_code: dict[str, dict[str, Mapping[str, Any]]] = {}
        self._by_concept_id: dict[int, Mapping[str, Any]] = {}
        for vocab_id, concepts in vocabulary.items():
            if vocab_id == _ATHENA_META_KEY or not isinstance(concepts, Mapping):
                continue
            code_index: dict[str, Mapping[str, Any]] = {}
            for concept_code, record in concepts.items():
                if not isinstance(record, Mapping):
                    continue
                code_index[str(concept_code)] = record
                concept_id = _optional_int(record.get("concept_id"))
                if concept_id is not None and concept_id != UNMAPPED_CONCEPT_ID:
                    self._by_concept_id.setdefault(concept_id, record)
            self._by_vocab_code[str(vocab_id)] = code_index

        self._usagi: dict[str, int] = {
            str(key): int(value) for key, value in dict(usagi or {}).items()
        }
        self._vocabulary_version = vocabulary_version

    @classmethod
    def from_athena(
        cls,
        vocabulary_path: str | Path,
        usagi_path: str | Path | None = None,
        *,
        vocabulary_version: str | Mapping[str, str] | None = None,
        include_synonyms: bool = True,
        approved_only: bool = True,
    ) -> VocabularyRouter:
        """Build a router by loading Athena and Usagi export files."""

        vocabulary = load_athena_vocab(
            vocabulary_path, include_synonyms=include_synonyms
        )
        usagi = (
            load_usagi_mapping(usagi_path, approved_only=approved_only)
            if usagi_path is not None
            else None
        )
        return cls(vocabulary, usagi, vocabulary_version=vocabulary_version)

    def route(
        self,
        source_code: str,
        *,
        source_vocabulary_id: str | None = None,
        domain_hint: str | None = None,
        source_code_description: str | None = None,
    ) -> SourceToConceptMapping:
        """Route one source term to a standard concept and CDM domain table."""

        code = str(source_code).strip()
        source_vocab = (source_vocabulary_id or "").strip()

        source_record = self._source_record(source_vocab, code)
        source_concept_id = UNMAPPED_CONCEPT_ID
        if source_record is not None:
            source_concept_id = (
                _optional_int(source_record.get("concept_id")) or UNMAPPED_CONCEPT_ID
            )
            if not source_vocab:
                source_vocab = str(source_record.get("vocabulary_id") or "")

        target_record: Mapping[str, Any] | None = None
        target_concept_id = UNMAPPED_CONCEPT_ID
        if source_record is not None and _is_standard(source_record):
            target_record = source_record
            target_concept_id = source_concept_id
        else:
            mapped_id = self._usagi_target(source_vocab, code)
            if mapped_id:
                target_concept_id = mapped_id
                target_record = self._by_concept_id.get(mapped_id)

        domain_id = ""
        if target_record is not None:
            domain_id = str(target_record.get("domain_id") or "")
        if not domain_id and source_record is not None:
            domain_id = str(source_record.get("domain_id") or "")
        if not domain_id and domain_hint:
            domain_id = str(domain_hint).strip()

        domain = route_domain(domain_id)
        cdm_table = _DOMAIN_TABLES[domain][0] if domain is not None else None

        mapped = target_concept_id != UNMAPPED_CONCEPT_ID
        if mapped and target_record is not None:
            target_vocab = str(target_record.get("vocabulary_id") or "")
            standard_concept = target_record.get("standard_concept")
        elif mapped:
            # Usagi resolved a real concept_id, but the target concept is
            # absent from the supplied Athena index — its vocabulary is
            # unknown here, so don't mislabel it as the source vocabulary.
            target_vocab = ""
            standard_concept = _STANDARD_FLAG
        else:
            target_vocab = UNMAPPED_VOCABULARY_ID
            standard_concept = None

        description = (source_code_description or "").strip() or None

        return SourceToConceptMapping(
            source_code=code,
            source_vocabulary_id=source_vocab,
            source_concept_id=source_concept_id,
            source_code_description=description,
            target_concept_id=target_concept_id,
            target_vocabulary_id=target_vocab,
            target_domain_id=domain_id,
            domain=domain,
            cdm_table=cdm_table,
            standard_concept=standard_concept,
            vocabulary_version=self._resolve_version(target_vocab, source_vocab),
            mapping_status="mapped" if mapped else "source_only",
        )

    def route_span(self, span: Any) -> SourceToConceptMapping:
        """Route a grounded span mapping/object using its coded fields."""

        return self.route(
            _first_text((span,), _CODE_FIELDS),
            source_vocabulary_id=_first_text((span,), _VOCABULARY_FIELDS) or None,
            domain_hint=_first_text((span,), _DOMAIN_FIELDS) or None,
            source_code_description=_first_text((span,), _DESCRIPTION_FIELDS) or None,
        )

    def route_all(self, spans: Iterable[Any]) -> tuple[SourceToConceptMapping, ...]:
        """Route an iterable of grounded spans to source-to-concept mappings."""

        return tuple(self.route_span(span) for span in spans)

    def concept_record(self, concept_id: int) -> Mapping[str, Any] | None:
        """Return a copy of a caller-supplied Athena concept record by ID."""

        record = self._by_concept_id.get(int(concept_id))
        return dict(record) if record is not None else None

    def _source_record(self, source_vocab: str, code: str) -> Mapping[str, Any] | None:
        if not code:
            return None
        if source_vocab:
            return self._by_vocab_code.get(source_vocab, {}).get(code)
        for code_index in self._by_vocab_code.values():
            record = code_index.get(code)
            if record is not None:
                return record
        return None

    def _usagi_target(self, source_vocab: str, code: str) -> int:
        if not code:
            return UNMAPPED_CONCEPT_ID
        if source_vocab:
            keyed = self._usagi.get(f"{source_vocab}:{code}")
            if keyed:
                return keyed
        return self._usagi.get(code, UNMAPPED_CONCEPT_ID)

    def _resolve_version(self, *vocabulary_ids: str) -> str | None:
        version = self._vocabulary_version
        if version is None:
            return None
        if isinstance(version, str):
            return version
        for vocab_id in vocabulary_ids:
            if vocab_id and vocab_id in version:
                return version[vocab_id]
        return version.get("", None)


def route_domain(domain_id: str | None) -> OmopDomain | None:
    """Resolve an OMOP ``domain_id`` label to a CDM domain, or ``None``."""

    if not domain_id:
        return None
    normalized = str(domain_id).strip().replace("-", "_").replace(" ", "_").casefold()
    return _DOMAIN_ALIASES.get(normalized)


def domain_cdm_table(domain: OmopDomain) -> str:
    """Return the CDM domain table name for a routed OMOP domain."""

    return _DOMAIN_TABLES[domain][0]


def _is_standard(record: Mapping[str, Any]) -> bool:
    flag = record.get("standard_concept")
    return isinstance(flag, str) and flag.strip().upper() == _STANDARD_FLAG


def _first_text(sources: Iterable[Any], names: Iterable[str]) -> str:
    for source in sources:
        for name in names:
            value = _value(source, name)
            if value is not _MISSING and value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping) and name in item:
        return item[name]
    if hasattr(item, name):
        return getattr(item, name)
    return _MISSING


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MappingStatus",
    "SourceToConceptMapping",
    "VocabularyRouter",
    "domain_cdm_table",
    "route_domain",
]
