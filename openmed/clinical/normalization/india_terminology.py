"""License-aware grounding from caller-supplied Indian terminology files.

The loader reads local AYUSH morbidity and Indian drug dictionaries directly
from paths chosen by the caller. No vocabulary payload, cache, or network
fallback is bundled. Restricted dictionaries are additionally required to live
outside the OpenMed repository and are never included in audit records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openmed.core.labels import CONDITION, MEDICATION
from openmed.core.terminology_licenses import (
    TerminologyLicense,
    validate_terminology_source_path,
)

AYUSH = "ayush"
INDIAN_DRUG = "indian_drug"
INDIA_TERMINOLOGY_KINDS = (AYUSH, INDIAN_DRUG)
IndiaTerminologyKind = Literal["ayush", "indian_drug"]

DEFAULT_AYUSH_SYSTEM_URI = "https://openmed.ai/fhir/CodeSystem/user-supplied-ayush"
DEFAULT_INDIAN_DRUG_SYSTEM_URI = (
    "https://openmed.ai/fhir/CodeSystem/user-supplied-indian-drug"
)

_KIND_TO_LABEL = {AYUSH: CONDITION, INDIAN_DRUG: MEDICATION}
_KIND_TO_SYSTEM = {
    AYUSH: DEFAULT_AYUSH_SYSTEM_URI,
    INDIAN_DRUG: DEFAULT_INDIAN_DRUG_SYSTEM_URI,
}
_SUPPORTED_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".tsv"})


class IndiaTerminologyError(ValueError):
    """Base error for invalid caller-supplied terminology metadata or rows."""


class IndiaTerminologyFormatError(IndiaTerminologyError):
    """Raised when a present dictionary has an invalid or unsupported shape."""


@dataclass(frozen=True)
class IndiaTerminologyDictionary:
    """Declaration for one caller-supplied local dictionary.

    A ``None`` or missing ``path`` produces an explicit skip notice instead of
    an exception. The dictionary kind fixes the canonical label: AYUSH rows are
    always ``CONDITION`` and Indian drug rows are always ``MEDICATION``.
    """

    source_name: str
    kind: IndiaTerminologyKind
    license: TerminologyLicense
    path: str | Path | None = None
    system_uri: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        source_name = self.source_name.strip()
        if not source_name:
            raise IndiaTerminologyError("dictionary source_name must not be blank")
        if self.kind not in INDIA_TERMINOLOGY_KINDS:
            raise IndiaTerminologyError(
                f"dictionary kind must be one of {INDIA_TERMINOLOGY_KINDS!r}"
            )
        system_uri = (self.system_uri or _KIND_TO_SYSTEM[self.kind]).strip()
        if not system_uri.startswith(("http://", "https://", "urn:")):
            raise IndiaTerminologyError("dictionary system_uri must be an absolute URI")
        version = self.version.strip() if self.version is not None else None
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "system_uri", system_uri)
        object.__setattr__(self, "version", version or None)


@dataclass(frozen=True)
class TerminologySkipNotice:
    """PHI-free reason that a declared dictionary was not loaded."""

    source_name: str
    kind: IndiaTerminologyKind
    reason: Literal["dictionary_absent", "dictionary_empty"]


@dataclass(frozen=True, repr=False)
class GroundedIndiaTerm:
    """One exact terminology match with canonical label and provenance."""

    text: str
    start: int
    end: int
    canonical_label: str
    system_uri: str
    code: str
    display: str
    source_name: str
    license_id: str
    restricted: bool
    version: str | None = None

    def __repr__(self) -> str:
        """Return metadata-only representation safe for diagnostic logs."""

        return (
            "GroundedIndiaTerm("
            f"label={self.canonical_label!r}, start={self.start}, end={self.end}, "
            f"source_name={self.source_name!r}, restricted={self.restricted})"
        )

    def to_codeable_concept(self) -> dict[str, Any]:
        """Return a provenance-stamped FHIR R4 ``CodeableConcept``."""

        from openmed.clinical.exporters.code_provenance import (
            UserSuppliedTerminologyProvenance,
            stamp_user_supplied_terminology_provenance,
        )
        from openmed.clinical.exporters.codeable_concept_simple import (
            codeable_concept,
            coding,
        )

        result = coding(self.system_uri, self.code, self.display)
        if self.version is not None:
            result["version"] = self.version
        result = stamp_user_supplied_terminology_provenance(
            result,
            UserSuppliedTerminologyProvenance(
                source_name=self.source_name,
                license_id=self.license_id,
                restricted=self.restricted,
            ),
        )
        return codeable_concept([result], text=self.text)

    def to_audit_dict(self) -> dict[str, Any]:
        """Return an audit-safe record with no raw term, code, or local path."""

        return {
            "term_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "code_sha256": hashlib.sha256(self.code.encode("utf-8")).hexdigest(),
            "start": self.start,
            "end": self.end,
            "canonical_label": self.canonical_label,
            "source_name": self.source_name,
            "license_id": self.license_id,
            "restricted": self.restricted,
        }


@dataclass(frozen=True, repr=False)
class _LoadedTerm:
    code: str
    display: str
    aliases: tuple[str, ...]
    canonical_label: str
    system_uri: str
    source_name: str
    license_id: str
    restricted: bool
    version: str | None


class UserSuppliedIndiaTerminology:
    """In-memory exact-match index built without retaining any source path."""

    def __init__(self, terms: Iterable[_LoadedTerm] = ()) -> None:
        self._terms = tuple(terms)
        index: dict[str, list[_LoadedTerm]] = defaultdict(list)
        patterns: list[tuple[re.Pattern[str], _LoadedTerm]] = []
        for term in self._terms:
            for surface in _unique_text((term.display, *term.aliases)):
                normalized = normalize_india_terminology_surface(surface)
                if normalized:
                    index[normalized].append(term)
                patterns.append((_surface_pattern(surface), term))
        self._index = {key: tuple(value) for key, value in index.items()}
        self._patterns = tuple(
            sorted(patterns, key=lambda item: len(item[0].pattern), reverse=True)
        )

    def __repr__(self) -> str:
        """Return a payload-free representation."""

        restricted_sources = len(
            {term.source_name for term in self._terms if term.restricted}
        )
        return (
            "UserSuppliedIndiaTerminology("
            f"term_count={len(self._terms)}, restricted_sources={restricted_sources})"
        )

    @property
    def term_count(self) -> int:
        """Return the number of loaded concept rows."""

        return len(self._terms)

    def ground_surface(
        self,
        surface: str,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[GroundedIndiaTerm, ...]:
        """Ground one complete surface by deterministic exact matching."""

        if not isinstance(surface, str):
            raise TypeError("surface must be a string")
        resolved_end = start + len(surface) if end is None else end
        if start < 0 or resolved_end < start:
            raise ValueError("term offsets must satisfy 0 <= start <= end")
        return tuple(
            _grounded(term, surface, start, resolved_end)
            for term in self._index.get(
                normalize_india_terminology_surface(surface), ()
            )
        )

    def ground_text(self, text: str) -> tuple[GroundedIndiaTerm, ...]:
        """Find registered terminology surfaces without creating new text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        grounded: list[GroundedIndiaTerm] = []
        seen: set[tuple[int, int, str, str]] = set()
        occupied: set[tuple[int, int, str, str]] = set()
        for pattern, term in self._patterns:
            term_key = (term.source_name, term.code)
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), *term_key)
                overlap_key = (*term_key,)
                if key in seen or any(
                    existing_source == overlap_key[0]
                    and existing_code == overlap_key[1]
                    and match.start() < existing_end
                    and match.end() > existing_start
                    for existing_start, existing_end, existing_source, existing_code in occupied
                ):
                    continue
                seen.add(key)
                occupied.add(key)
                grounded.append(
                    _grounded(term, match.group(0), match.start(), match.end())
                )
        return tuple(
            sorted(grounded, key=lambda item: (item.start, item.end, item.code))
        )

    def audit_summary(self) -> dict[str, Any]:
        """Return source/license counts without paths or dictionary payloads."""

        sources = {
            (term.source_name, term.license_id, term.restricted) for term in self._terms
        }
        return {
            "term_count": len(self._terms),
            "sources": [
                {
                    "source_name": source_name,
                    "license_id": license_id,
                    "restricted": restricted,
                }
                for source_name, license_id, restricted in sorted(sources)
            ],
        }


@dataclass(frozen=True)
class IndiaTerminologyLoadResult:
    """Loaded terminology plus explicit notices for absent dictionaries."""

    terminology: UserSuppliedIndiaTerminology
    skipped: tuple[TerminologySkipNotice, ...] = ()


class IndiaTerminologyLoader:
    """Load user-supplied AYUSH and Indian drug dictionaries without copying."""

    def __init__(self, *, repository_root: str | Path | None = None) -> None:
        self.repository_root = repository_root

    def load(
        self,
        dictionaries: Sequence[IndiaTerminologyDictionary],
    ) -> IndiaTerminologyLoadResult:
        """Load present dictionaries and report absent ones without raising."""

        loaded: list[_LoadedTerm] = []
        skipped: list[TerminologySkipNotice] = []
        for dictionary in dictionaries:
            if (
                dictionary.path is None
                or not Path(dictionary.path).expanduser().exists()
            ):
                skipped.append(
                    TerminologySkipNotice(
                        source_name=dictionary.source_name,
                        kind=dictionary.kind,
                        reason="dictionary_absent",
                    )
                )
                continue
            source_path = validate_terminology_source_path(
                dictionary.path,
                dictionary.license,
                repository_root=self.repository_root,
            )
            rows = _read_dictionary(source_path, dictionary.source_name)
            terms = [_term_from_row(indexed_row, dictionary) for indexed_row in rows]
            if not terms:
                skipped.append(
                    TerminologySkipNotice(
                        source_name=dictionary.source_name,
                        kind=dictionary.kind,
                        reason="dictionary_empty",
                    )
                )
                continue
            loaded.extend(terms)
        return IndiaTerminologyLoadResult(
            terminology=UserSuppliedIndiaTerminology(loaded),
            skipped=tuple(skipped),
        )


def normalize_india_terminology_surface(value: str) -> str:
    """Normalize a terminology surface while preserving non-Latin scripts."""

    if not isinstance(value, str):
        raise TypeError("terminology surface must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )


def _read_dictionary(
    path: Path,
    source_name: str,
) -> list[tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.casefold()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise IndiaTerminologyFormatError(
            f"{source_name!r} must use one of {sorted(_SUPPORTED_SUFFIXES)!r}"
        )
    try:
        if suffix in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(
                    handle,
                    delimiter="\t" if suffix == ".tsv" else ",",
                )
                return [(index, row) for index, row in enumerate(reader, start=2)]
        if suffix == ".jsonl":
            rows: list[tuple[int, Mapping[str, Any]]] = []
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise TypeError
                    rows.append((index, row))
            return rows
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("terms")
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise TypeError
        return [
            (index, row)
            for index, row in enumerate(payload, start=1)
            if isinstance(row, Mapping)
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise IndiaTerminologyFormatError(
            f"{source_name!r} could not be read as a terminology dictionary"
        ) from exc


def _term_from_row(
    indexed_row: tuple[int, Mapping[str, Any]],
    dictionary: IndiaTerminologyDictionary,
) -> _LoadedTerm:
    row_number, row = indexed_row
    code = _first_text(row, ("code", "concept_id", "id"))
    display = _first_text(row, ("display", "preferred_term", "term", "name"))
    if not code or not display:
        raise IndiaTerminologyFormatError(
            f"{dictionary.source_name!r} row {row_number} must "
            "contain non-empty code and display fields"
        )
    aliases = _aliases(row)
    return _LoadedTerm(
        code=code,
        display=display,
        aliases=aliases,
        canonical_label=_KIND_TO_LABEL[dictionary.kind],
        system_uri=str(dictionary.system_uri),
        source_name=dictionary.source_name,
        license_id=dictionary.license.license_id,
        restricted=dictionary.license.restricted,
        version=dictionary.version,
    )


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return ""


def _aliases(row: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("aliases", "synonyms", "surfaces", "surface"):
        value = row.get(key)
        if isinstance(value, str):
            values.extend(part.strip() for part in value.split("|") if part.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(str(part).strip() for part in value if str(part).strip())
    return _unique_text(values)


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_india_terminology_surface(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return tuple(result)


def _surface_pattern(surface: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in surface.split()]
    pattern = r"\s+".join(pieces)
    return re.compile(rf"(?<!\w){pattern}(?!\w)", re.IGNORECASE | re.UNICODE)


def _grounded(
    term: _LoadedTerm,
    text: str,
    start: int,
    end: int,
) -> GroundedIndiaTerm:
    return GroundedIndiaTerm(
        text=text,
        start=start,
        end=end,
        canonical_label=term.canonical_label,
        system_uri=term.system_uri,
        code=term.code,
        display=term.display,
        source_name=term.source_name,
        license_id=term.license_id,
        restricted=term.restricted,
        version=term.version,
    )


__all__ = [
    "AYUSH",
    "DEFAULT_AYUSH_SYSTEM_URI",
    "DEFAULT_INDIAN_DRUG_SYSTEM_URI",
    "INDIA_TERMINOLOGY_KINDS",
    "INDIAN_DRUG",
    "GroundedIndiaTerm",
    "IndiaTerminologyDictionary",
    "IndiaTerminologyError",
    "IndiaTerminologyFormatError",
    "IndiaTerminologyKind",
    "IndiaTerminologyLoadResult",
    "IndiaTerminologyLoader",
    "TerminologySkipNotice",
    "UserSuppliedIndiaTerminology",
    "normalize_india_terminology_surface",
]
