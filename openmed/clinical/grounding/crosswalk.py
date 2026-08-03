"""Offline ICD-10-CM/SNOMED CT crosswalks over caller-supplied UMLS data.

UMLS and SNOMED CT are licensed terminologies.  This module deliberately has
no default source, download client, or bundled terminology content.  A caller
must point :class:`UMLSCrosswalk` at local ``MRCONSO`` and ``MRMAP`` files from
their licensed environment.  The files are read only when a lookup is first
requested and all lookup work remains local to the process.

The parser accepts the normal pipe-delimited UMLS RRF layout as well as small
headered CSV, TSV, or JSONL projections.  Headered projections are useful for
an out-of-process terminology service to expose only the rows and fields that
OpenMed needs without handing the full UMLS release to this package.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vocab import RestrictedVocabularyError

__all__ = [
    "Crosswalk",
    "CrosswalkCandidate",
    "CrosswalkConfigurationError",
    "CrosswalkDataError",
    "CrosswalkProvenance",
    "ICDSNOMEDCrosswalk",
    "UMLSCodeCrosswalk",
    "UMLSMappingSource",
    "UMLSCrosswalk",
    "crosswalk",
]

ICD10CM = "ICD10CM"
SNOMEDCT = "SNOMEDCT"
_SUPPORTED_SYSTEMS = frozenset({ICD10CM, SNOMEDCT})
_SUPPORTED_PAIRS = frozenset(
    {
        (ICD10CM, SNOMEDCT),
        (SNOMEDCT, ICD10CM),
    }
)
_CUI_RE = re.compile(r"^C[0-9A-Z]+$", re.IGNORECASE)
_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")


class CrosswalkConfigurationError(RestrictedVocabularyError):
    """Raised when a caller has not configured a usable local UMLS source."""


class CrosswalkDataError(CrosswalkConfigurationError):
    """Raised when a configured UMLS projection cannot be interpreted."""


@dataclass(frozen=True)
class UMLSMappingSource:
    """Resolved local ``MRCONSO`` and ``MRMAP`` paths.

    ``path`` may be a directory containing files named ``MRCONSO.RRF`` and
    ``MRMAP.RRF`` (case-insensitive).  It may also be the MRCONSO file itself
    when ``mrmap_path`` is supplied.  No URL, archive, or network source is
    accepted.
    """

    mrconso_path: Path
    mrmap_path: Path

    def __init__(
        self,
        path: str | Path | None = None,
        mrmap_path: str | Path | None = None,
        *,
        mrconso_path: str | Path | None = None,
    ) -> None:
        if path is not None and mrconso_path is not None:
            raise CrosswalkConfigurationError(
                "provide either path or mrconso_path, not both"
            )
        if path is not None:
            candidate = _local_path(path, "UMLS source")
            if candidate.is_dir():
                conso = _find_release_file(candidate, "MRCONSO")
                maps = (
                    _local_path(mrmap_path, "MRMAP source")
                    if mrmap_path is not None
                    else _find_release_file(candidate, "MRMAP")
                )
            else:
                conso = candidate
                if mrmap_path is None:
                    raise CrosswalkConfigurationError(
                        "a file source must be paired with a user-supplied MRMAP path"
                    )
                maps = _local_path(mrmap_path, "MRMAP source")
        else:
            if mrconso_path is None or mrmap_path is None:
                raise CrosswalkConfigurationError(
                    "ICD-10-CM/SNOMED CT crosswalk requires a user-configured "
                    "UMLS MRCONSO and MRMAP source; no UMLS data is bundled "
                    "or downloaded"
                )
            conso = _local_path(mrconso_path, "MRCONSO source")
            maps = _local_path(mrmap_path, "MRMAP source")

        if not conso.is_file():
            raise CrosswalkConfigurationError(
                f"user-supplied UMLS MRCONSO source is not a file: {conso}"
            )
        if not maps.is_file():
            raise CrosswalkConfigurationError(
                f"user-supplied UMLS MRMAP source is not a file: {maps}"
            )
        object.__setattr__(self, "mrconso_path", conso)
        object.__setattr__(self, "mrmap_path", maps)

    @property
    def root(self) -> Path:
        """Return the directory containing the supplied source files."""

        if self.mrconso_path.parent == self.mrmap_path.parent:
            return self.mrconso_path.parent
        return self.mrconso_path.parent


@dataclass(frozen=True)
class CrosswalkProvenance(Mapping[str, object]):
    """Map provenance carried with one crosswalk candidate.

    Lower numeric ``map_priority`` values sort before higher values.  A
    ``None`` priority means that the source did not provide one; deterministic
    code/rule tie-breakers are used in that case.
    """

    source_system: str
    target_system: str
    source_code: str
    map_rule: str
    map_priority: int | None = None
    map_advice: str = ""
    source_cui: str = ""
    target_cui: str = ""
    mapping_source_system: str = ""
    mapping_target_system: str = ""
    data_source: str = "user-supplied-local"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable provenance fields."""

        return {
            "source_system": self.source_system,
            "target_system": self.target_system,
            "source_code": self.source_code,
            "map_rule": self.map_rule,
            "map_priority": self.map_priority,
            "map_advice": self.map_advice,
            "source_cui": self.source_cui,
            "target_cui": self.target_cui,
            "mapping_source_system": self.mapping_source_system,
            "mapping_target_system": self.mapping_target_system,
            "data_source": self.data_source,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class CrosswalkCandidate(Mapping[str, object]):
    """One target code returned by :meth:`UMLSCrosswalk.crosswalk`."""

    target_code: str
    provenance: CrosswalkProvenance

    @property
    def code(self) -> str:
        """Alias for ``target_code`` used by other grounding candidates."""

        return self.target_code

    @property
    def source_code(self) -> str:
        """Return the queried source code recorded in provenance."""

        return self.provenance.source_code

    @property
    def source_system(self) -> str:
        """Return the canonical queried source system."""

        return self.provenance.source_system

    @property
    def target_system(self) -> str:
        """Return the canonical target system."""

        return self.provenance.target_system

    @property
    def map_rule(self) -> str:
        """Return the UMLS map rule."""

        return self.provenance.map_rule

    @property
    def map_priority(self) -> int | None:
        """Return the UMLS map priority, if supplied."""

        return self.provenance.map_priority

    @property
    def map_advice(self) -> str:
        """Return the UMLS map advice, if supplied."""

        return self.provenance.map_advice

    def to_dict(self) -> dict[str, object]:
        """Return the candidate and nested provenance as a JSON object."""

        return {
            "target_code": self.target_code,
            "source_code": self.source_code,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "map_rule": self.map_rule,
            "map_priority": self.map_priority,
            "provenance": self.provenance.to_dict(),
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class _Concept:
    cui: str
    system: str
    code: str
    row_number: int


@dataclass(frozen=True)
class _RawMapping:
    source_system: str | None
    source_code: str | None
    target_system: str | None
    target_code: str | None
    source_cui: str = ""
    target_cui: str = ""
    map_rule: str = "unspecified"
    map_advice: str = ""
    map_priority: int | None = None
    row_number: int = 0


@dataclass(frozen=True)
class _ResolvedMapping:
    source_system: str
    source_code: str
    target_system: str
    target_code: str
    source_cui: str
    target_cui: str
    map_rule: str
    map_advice: str
    map_priority: int | None
    row_number: int


class UMLSCrosswalk:
    """Crosswalk ICD-10-CM and SNOMED CT codes from local UMLS mappings.

    Args:
        source: Local directory containing ``MRCONSO`` and ``MRMAP``, or a
            local MRCONSO file when ``mrmap_path`` is also given.
        mrmap_path: Optional explicit local MRMAP path.
        mrconso_path: Optional explicit local MRCONSO path.  Use this with
            ``mrmap_path`` when the two files are in different directories.

    The source is intentionally mandatory.  The class never downloads, calls
    a terminology service, or reads an environment variable for a fallback.
    Callers remain responsible for the UMLS and SNOMED CT licenses governing
    the supplied files.
    """

    def __init__(
        self,
        source: str | Path | UMLSMappingSource | None = None,
        mrmap_path: str | Path | None = None,
        *,
        mrconso_path: str | Path | None = None,
    ) -> None:
        if isinstance(source, UMLSMappingSource):
            if mrmap_path is not None or mrconso_path is not None:
                raise CrosswalkConfigurationError(
                    "a resolved UMLSMappingSource cannot be combined with file paths"
                )
            resolved = source
        else:
            resolved = UMLSMappingSource(
                source,
                mrmap_path,
                mrconso_path=mrconso_path,
            )
        self.source = resolved
        self._index: dict[tuple[str, str, str], tuple[_ResolvedMapping, ...]] | None = (
            None
        )

    def crosswalk(
        self,
        code: str,
        source_system: str,
        target_system: str,
    ) -> tuple[CrosswalkCandidate, ...]:
        """Return deterministic target candidates for one source code.

        Both ``ICD-10-CM`` to ``SNOMED CT`` and the reverse direction are
        supported.  Candidates are ordered by ascending map priority, then by
        target code, map rule, and source row as deterministic tie-breakers.
        An unknown code returns an empty tuple.
        """

        source_key = _normalize_supported_system(source_system)
        target_key = _normalize_supported_system(target_system)
        if (source_key, target_key) not in _SUPPORTED_PAIRS:
            raise ValueError(
                "crosswalk supports only ICD-10-CM <-> SNOMED CT; received "
                f"{source_system!r} -> {target_system!r}"
            )
        query = _normalize_code(source_key, code)
        if not query:
            raise ValueError("crosswalk code must be non-empty text")

        mappings = self._load_index().get((source_key, query, target_key), ())
        return tuple(
            _candidate_from_mapping(mapping, source_key, target_key)
            for mapping in mappings
        )

    def lookup(
        self,
        code: str,
        source_system: str,
        target_system: str,
    ) -> tuple[CrosswalkCandidate, ...]:
        """Alias for :meth:`crosswalk`."""

        return self.crosswalk(code, source_system, target_system)

    def __call__(
        self,
        code: str,
        source_system: str,
        target_system: str,
    ) -> tuple[CrosswalkCandidate, ...]:
        """Call the engine as a shorthand for :meth:`crosswalk`."""

        return self.crosswalk(code, source_system, target_system)

    def _load_index(self) -> dict[tuple[str, str, str], tuple[_ResolvedMapping, ...]]:
        if self._index is not None:
            return self._index

        concepts = _read_conso(self.source.mrconso_path)
        mappings = _resolve_mappings(
            _read_map(self.source.mrmap_path),
            concepts,
        )
        index: dict[tuple[str, str, str], list[_ResolvedMapping]] = defaultdict(list)
        for mapping in mappings:
            index[
                (
                    mapping.source_system,
                    _normalize_code(mapping.source_system, mapping.source_code),
                    mapping.target_system,
                )
            ].append(mapping)
            index[
                (
                    mapping.target_system,
                    _normalize_code(mapping.target_system, mapping.target_code),
                    mapping.source_system,
                )
            ].append(mapping)

        self._index = {
            key: _ordered_unique(
                values,
                reverse=bool(values and key[0] == values[0].target_system),
            )
            for key, values in index.items()
        }
        return self._index


def crosswalk(
    code: str,
    source_system: str,
    target_system: str,
    *,
    source: str | Path | UMLSMappingSource | UMLSCrosswalk | None = None,
    mrmap_path: str | Path | None = None,
    mrconso_path: str | Path | None = None,
) -> tuple[CrosswalkCandidate, ...]:
    """Crosswalk one code using an explicitly supplied local UMLS source.

    ``source`` may be an existing :class:`UMLSCrosswalk`, a resolved
    :class:`UMLSMappingSource`, or a local source directory/file.  Omitting it
    is an error; no global, cached, downloaded, or bundled UMLS source exists.
    """

    if source is None and mrmap_path is None and mrconso_path is None:
        raise CrosswalkConfigurationError(
            "crosswalk requires a user-configured local UMLS MRCONSO/MRMAP "
            "source; UMLS data is licensed and is never bundled or downloaded"
        )
    engine = (
        source
        if isinstance(source, UMLSCrosswalk)
        else UMLSCrosswalk(
            source,
            mrmap_path,
            mrconso_path=mrconso_path,
        )
    )
    return engine.crosswalk(code, source_system, target_system)


def _candidate_from_mapping(
    mapping: _ResolvedMapping,
    source_system: str,
    target_system: str,
) -> CrosswalkCandidate:
    reverse = source_system == mapping.target_system
    source_code = mapping.target_code if reverse else mapping.source_code
    source_cui = mapping.target_cui if reverse else mapping.source_cui
    target_cui = mapping.source_cui if reverse else mapping.target_cui
    provenance = CrosswalkProvenance(
        source_system=source_system,
        target_system=target_system,
        source_code=source_code,
        map_rule=mapping.map_rule,
        map_priority=mapping.map_priority,
        map_advice=mapping.map_advice,
        source_cui=source_cui,
        target_cui=target_cui,
        mapping_source_system=mapping.source_system,
        mapping_target_system=mapping.target_system,
    )
    return CrosswalkCandidate(
        target_code=mapping.source_code if reverse else mapping.target_code,
        provenance=provenance,
    )


def _ordered_unique(
    mappings: list[_ResolvedMapping],
    *,
    reverse: bool = False,
) -> tuple[_ResolvedMapping, ...]:
    ordered = sorted(
        mappings,
        key=lambda item: (
            item.map_priority is None,
            item.map_priority if item.map_priority is not None else 0,
            item.source_code if reverse else item.target_code,
            item.map_rule,
            item.map_advice,
            item.row_number,
        ),
    )
    result: list[_ResolvedMapping] = []
    seen: set[str] = set()
    for mapping in ordered:
        target_system = mapping.source_system if reverse else mapping.target_system
        target_code = mapping.source_code if reverse else mapping.target_code
        key = _normalize_code(target_system, target_code)
        if key in seen:
            continue
        seen.add(key)
        result.append(mapping)
    return tuple(result)


def _resolve_mappings(
    raw_mappings: Iterator[_RawMapping],
    concepts: tuple[_Concept, ...],
) -> tuple[_ResolvedMapping, ...]:
    by_code: dict[tuple[str, str], list[_Concept]] = defaultdict(list)
    by_cui: dict[str, list[_Concept]] = defaultdict(list)
    for concept in concepts:
        by_code[(concept.system, _normalize_code(concept.system, concept.code))].append(
            concept
        )
        if concept.cui:
            by_cui[concept.cui].append(concept)

    resolved: list[_ResolvedMapping] = []
    for raw in raw_mappings:
        source_system = _canonical_system(raw.source_system)
        target_system = _canonical_system(raw.target_system)
        source_code = _clean(raw.source_code)
        target_code = _clean(raw.target_code)

        parsed_source_system, parsed_source_code = _split_qualified_code(source_code)
        if parsed_source_system is not None:
            source_system = source_system or parsed_source_system
            source_code = parsed_source_code
        parsed_target_system, parsed_target_code = _split_qualified_code(target_code)
        if parsed_target_system is not None:
            target_system = target_system or parsed_target_system
            target_code = parsed_target_code
        target_cui = raw.target_cui
        if not target_cui and _looks_like_cui(target_code):
            target_cui = target_code
            target_code = ""

        source_candidates = _concept_candidates(
            source_system,
            source_code,
            raw.source_cui,
            by_code,
            by_cui,
        )
        if source_system is None and source_candidates:
            source_system = source_candidates[0].system
        if not source_code and source_candidates:
            source_code = source_candidates[0].code
        if not raw.source_cui and source_candidates:
            raw_source_cui = source_candidates[0].cui
        else:
            raw_source_cui = raw.source_cui

        target_candidates = _concept_candidates(
            target_system,
            target_code,
            target_cui,
            by_code,
            by_cui,
        )
        if target_system is None and target_candidates:
            target_system = target_candidates[0].system
        if not target_code and target_candidates:
            target_code = target_candidates[0].code
        if not target_cui and target_candidates:
            raw_target_cui = target_candidates[0].cui
        else:
            raw_target_cui = target_cui

        if not source_system or not target_system or not source_code or not target_code:
            continue
        if (source_system, target_system) not in _SUPPORTED_PAIRS:
            continue
        resolved.append(
            _ResolvedMapping(
                source_system=source_system,
                source_code=source_code,
                target_system=target_system,
                target_code=target_code,
                source_cui=raw_source_cui,
                target_cui=raw_target_cui,
                map_rule=raw.map_rule or "unspecified",
                map_advice=raw.map_advice,
                map_priority=raw.map_priority,
                row_number=raw.row_number,
            )
        )
    return tuple(resolved)


def _concept_candidates(
    system: str | None,
    code: str,
    cui: str,
    by_code: Mapping[tuple[str, str], list[_Concept]],
    by_cui: Mapping[str, list[_Concept]],
) -> list[_Concept]:
    if cui:
        candidates = list(by_cui.get(cui, ()))
        if system is not None:
            candidates = [item for item in candidates if item.system == system]
        if candidates:
            return candidates
    if not code:
        return []
    parsed_system, parsed_code = _split_qualified_code(code)
    if parsed_system is not None:
        system = system or parsed_system
        code = parsed_code
    if system is not None:
        return list(by_code.get((system, _normalize_code(system, code)), ()))
    candidates = [
        concept
        for (candidate_system, candidate_code), values in by_code.items()
        if candidate_code == _normalize_code(candidate_system, code)
        for concept in values
    ]
    return candidates


def _read_conso(path: Path) -> tuple[_Concept, ...]:
    concepts: list[_Concept] = []
    for row_number, row in _iter_table(path):
        values = _row_values(row)
        if isinstance(row, Mapping):
            cui = _value(row, "cui", "concept_unique_identifier")
            raw_system = _value(
                row,
                "sab",
                "source_abbreviation",
                "source_system",
                "vocabulary",
                "source_vocabulary",
            )
            code = _value(row, "code", "scui", "sdui", "concept_code")
        else:
            cui, raw_system, code = _conso_positions(values)
        system = _canonical_system(raw_system)
        if not cui or not system or not code:
            continue
        concepts.append(
            _Concept(
                cui=cui,
                system=system,
                code=code,
                row_number=row_number,
            )
        )
    return tuple(concepts)


def _read_map(path: Path) -> Iterator[_RawMapping]:
    for row_number, row in _iter_table(path):
        values = _row_values(row)
        if isinstance(row, Mapping):
            yield _mapping_from_header(row, row_number)
        else:
            yield _mapping_from_positions(path, values, row_number)


def _mapping_from_header(row: Mapping[str, str], row_number: int) -> _RawMapping:
    source_system = _value(
        row,
        "source_system",
        "source_sab",
        "from_system",
        "from_sab",
        "source_vocabulary",
        "source_vocab",
        "sab",
    )
    target_system = _value(
        row,
        "target_system",
        "target_sab",
        "to_system",
        "to_sab",
        "target_vocabulary",
        "target_vocab",
        "target_source",
    )
    source_code = _value(
        row,
        "source_code",
        "from_code",
        "source_concept_code",
        "from",
        "code",
        "scui",
    )
    target_code = _value(
        row,
        "target_code",
        "to_code",
        "target_concept_code",
        "mapped_code",
        "map_target",
        "to",
        "target",
        "sctid",
        "atv",
    )
    return _RawMapping(
        source_system=source_system or None,
        source_code=source_code or None,
        target_system=target_system or None,
        target_code=target_code or None,
        source_cui=_value(row, "source_cui", "from_cui", "cui"),
        target_cui=_value(row, "target_cui", "to_cui", "mapped_cui"),
        map_rule=_value(row, "map_rule", "mapping_rule", "maprule", "rule")
        or "unspecified",
        map_advice=_value(row, "map_advice", "mapping_advice", "mapadvice", "advice"),
        map_priority=_priority(
            _value(
                row,
                "map_priority",
                "mappriority",
                "priority",
                "map_group",
                "mapgroup",
                "rank",
                "order",
                "sequence",
            )
        ),
        row_number=row_number,
    )


def _mapping_from_positions(
    path: Path,
    values: tuple[str, ...],
    row_number: int,
) -> _RawMapping:
    if not values:
        return _RawMapping(None, None, None, None, row_number=row_number)

    if len(values) >= 16 and path.name.casefold().startswith("mrmap"):
        target_code = _first_nonempty(values[13], values[12], values[6])
        return _RawMapping(
            source_system=values[7] or None,
            source_code=_first_nonempty(values[4], values[8], values[10]) or None,
            target_system=_system_in_values(values[5], values[6]),
            target_code=target_code or None,
            source_cui=values[0],
            map_rule=values[14] or "unspecified",
            map_advice=values[15],
            row_number=row_number,
        )

    if len(values) >= 8 and _looks_like_cui(values[0]) and _looks_like_cui(values[1]):
        return _RawMapping(
            source_system=values[2] or None,
            source_code=values[3] or None,
            target_system=values[4] or None,
            target_code=values[5] or None,
            source_cui=values[0],
            target_cui=values[1],
            map_rule=values[6] or "unspecified",
            map_priority=_priority(values[7]),
            row_number=row_number,
        )

    if (
        len(values) >= 6
        and _canonical_system(values[0])
        and _canonical_system(values[2])
    ):
        return _RawMapping(
            source_system=values[0],
            source_code=values[1] or None,
            target_system=values[2],
            target_code=values[3] or None,
            map_rule=values[4] or "unspecified",
            map_priority=_priority(values[5]),
            row_number=row_number,
        )

    if len(values) >= 4 and _looks_like_cui(values[0]) and _looks_like_cui(values[1]):
        return _RawMapping(
            source_system=None,
            source_code=None,
            target_system=None,
            target_code=None,
            source_cui=values[0],
            target_cui=values[1],
            map_rule=values[2] or "unspecified",
            map_priority=_priority(values[3]),
            row_number=row_number,
        )

    return _RawMapping(
        source_system=values[0] or None,
        source_code=values[1] or None,
        target_system=values[2] if len(values) > 2 and values[2] else None,
        target_code=values[3] if len(values) > 3 and values[3] else None,
        map_rule=values[4] if len(values) > 4 and values[4] else "unspecified",
        map_priority=_priority(values[5]) if len(values) > 5 else None,
        row_number=row_number,
    )


def _conso_positions(values: tuple[str, ...]) -> tuple[str, str, str]:
    if len(values) >= 16:
        return values[0], values[11], values[13]
    if len(values) >= 5 and _looks_like_language(values[1]):
        return values[0], values[2], values[3]
    if len(values) >= 4:
        return values[0], values[1], values[2]
    return "", "", ""


def _iter_table(
    path: Path,
) -> Iterator[tuple[int, Mapping[str, str] | tuple[str, ...]]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CrosswalkDataError(
                        f"{path}:{row_number} is not valid JSONL"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise CrosswalkDataError(
                        f"{path}:{row_number} must contain a JSON object"
                    )
                yield (
                    row_number,
                    {
                        _column_name(str(key)): _clean(value)
                        for key, value in row.items()
                    },
                )
        return

    with path.open(encoding="utf-8", newline="") as handle:
        first_line = _next_data_line(handle)
        if first_line is None:
            return
        delimiter = _delimiter(first_line)
        first_values = tuple(next(csv.reader([first_line], delimiter=delimiter)))
        if _looks_like_header(first_values):
            headers = tuple(_column_name(value) for value in first_values)
            reader = csv.reader(handle, delimiter=delimiter)
            for row_number, values in enumerate(reader, start=2):
                if not values or not any(value.strip() for value in values):
                    continue
                yield (
                    row_number,
                    {
                        header: _clean(values[index]) if index < len(values) else ""
                        for index, header in enumerate(headers)
                        if header
                    },
                )
            return

        yield 1, tuple(_clean(value) for value in first_values)
        reader = csv.reader(handle, delimiter=delimiter)
        for row_number, values in enumerate(reader, start=2):
            if not values or not any(value.strip() for value in values):
                continue
            yield row_number, tuple(_clean(value) for value in values)


def _next_data_line(handle: Any) -> str | None:
    for line in handle:
        if line.strip() and not line.lstrip().startswith("#"):
            return line
    return None


def _delimiter(line: str) -> str:
    if "|" in line:
        return "|"
    if "\t" in line:
        return "\t"
    return ","


def _looks_like_header(values: tuple[str, ...]) -> bool:
    normalized = {_column_name(value) for value in values}
    return bool(
        normalized
        & {
            "cui",
            "sab",
            "code",
            "source_system",
            "source_code",
            "target_system",
            "target_code",
            "source_sab",
            "target_sab",
            "from_code",
            "to_code",
            "map_rule",
            "map_advice",
            "map_priority",
            "priority",
            "maprule",
        }
    ) and not _looks_like_cui(values[0] if values else "")


def _row_values(row: Mapping[str, str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(row, Mapping):
        return tuple(row.values())
    return row


def _value(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        key = _column_name(name)
        value = row.get(key)
        if value is not None and str(value).strip():
            return _clean(value)
    return ""


def _column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _first_nonempty(*values: str) -> str:
    return next((value for value in values if value), "")


def _local_path(value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise CrosswalkConfigurationError(f"{label} must be a local filesystem path")
    path = Path(value).expanduser()
    if not path.exists():
        raise CrosswalkConfigurationError(f"{label} does not exist: {path}")
    return path


def _find_release_file(root: Path, stem: str) -> Path:
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.stem.casefold() == stem.casefold()
    )
    if not candidates:
        raise CrosswalkConfigurationError(
            f"user-supplied UMLS source is missing {stem}.RRF"
        )
    return candidates[0]


def _normalize_supported_system(value: str) -> str:
    normalized = _canonical_system(value)
    if normalized is None:
        raise ValueError(
            f"crosswalk system must be ICD-10-CM or SNOMED CT; received {value!r}"
        )
    return normalized


def _canonical_system(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if key in {"icd10", "icd10cm", "icd10cmus"} or key.endswith("icd10cm"):
        return ICD10CM
    if key in {
        "snomed",
        "snomedct",
        "snomedctus",
        "snomedctint",
        "sct",
    } or key.endswith(("snomedct", "sct")):
        return SNOMEDCT
    return None


def _split_qualified_code(value: str) -> tuple[str | None, str]:
    if ":" not in value:
        return None, value
    possible_system, possible_code = value.split(":", 1)
    system = _canonical_system(possible_system)
    return (
        (system, possible_code) if system and possible_code.strip() else (None, value)
    )


def _normalize_code(system: str, value: object) -> str:
    text = _clean(value).casefold().replace(" ", "")
    if system == ICD10CM:
        return text.replace(".", "")
    return text


def _looks_like_cui(value: object) -> bool:
    return bool(_CUI_RE.fullmatch(_clean(value)))


def _looks_like_language(value: object) -> bool:
    return _clean(value).casefold() in {"eng", "spa", "fra", "deu", "por", "ita"}


def _system_in_values(*values: str) -> str | None:
    for value in values:
        if _canonical_system(value):
            return value
    return None


def _priority(value: object) -> int | None:
    text = _clean(value)
    if not text:
        return None
    if not _INTEGER_RE.fullmatch(text):
        try:
            numeric = float(text)
        except ValueError:
            return None
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        return int(numeric)
    return int(text)


# Short names make the common constructor forms discoverable without creating
# separate implementations or allowing an unconfigured global source.
Crosswalk = UMLSCrosswalk
ICDSNOMEDCrosswalk = UMLSCrosswalk
UMLSCodeCrosswalk = UMLSCrosswalk
