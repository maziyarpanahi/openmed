"""Resolve OMOP source codes from a caller-supplied Athena release.

The OHDSI Athena vocabulary bundle is deliberately not a package resource.
Callers download the release under their own credentials and pass its local
directory to :class:`AthenaResolver`.  This module reads only the three local
Athena tables needed for source-to-standard resolution and never performs a
network request.  CPT4 rows are omitted unless the caller explicitly opts in
to a user-provided bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.offline import is_local_only

__all__ = [
    "ATHENA_REQUIRED_FILES",
    "CPT4_VOCABULARY_ID",
    "AthenaBundleError",
    "AthenaConcept",
    "AthenaResolver",
]

ATHENA_REQUIRED_FILES = (
    "CONCEPT.csv",
    "CONCEPT_RELATIONSHIP.csv",
    "VOCABULARY.csv",
)
CPT4_VOCABULARY_ID = "CPT4"
_MAPS_TO = "maps to"
_HASH_SCHEMA_VERSION = 1

_CONCEPT_COLUMNS = (
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
)
_CONCEPT_REQUIRED_COLUMNS = (
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "standard_concept",
    "concept_code",
)
_RELATIONSHIP_COLUMNS = (
    "concept_id_1",
    "concept_id_2",
    "relationship_id",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
)
_RELATIONSHIP_REQUIRED_COLUMNS = (
    "concept_id_1",
    "concept_id_2",
    "relationship_id",
)
_VOCABULARY_COLUMNS = (
    "vocabulary_id",
    "vocabulary_name",
    "vocabulary_reference",
    "vocabulary_version",
    "vocabulary_concept_id",
)
_VOCABULARY_REQUIRED_COLUMNS = ("vocabulary_id", "vocabulary_version")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class AthenaBundleError(ValueError):
    """Raised when a local Athena bundle is malformed."""


@dataclass(frozen=True, slots=True)
class AthenaConcept:
    """Metadata for one concept loaded from Athena.

    The fields mirror the corresponding OMOP ``CONCEPT`` columns.  The
    ``standard_concept`` flag is ``"S"`` for a standard concept and ``None``
    for the usual non-standard source concept.
    """

    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str | None
    concept_code: str
    valid_start_date: str
    valid_end_date: str
    invalid_reason: str | None

    @property
    def is_standard(self) -> bool:
        """Return whether this concept is marked as an OMOP standard concept."""

        return (self.standard_concept or "").casefold() == "s"

    def to_dict(self) -> dict[str, Any]:
        """Return the concept metadata as a JSON-serializable mapping."""

        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "domain_id": self.domain_id,
            "vocabulary_id": self.vocabulary_id,
            "concept_class_id": self.concept_class_id,
            "standard_concept": self.standard_concept,
            "concept_code": self.concept_code,
            "valid_start_date": self.valid_start_date,
            "valid_end_date": self.valid_end_date,
            "invalid_reason": self.invalid_reason,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow metadata to be read with either attributes or column names."""

        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Return a metadata field, matching the small mapping-style API."""

        try:
            return self[key]
        except KeyError:
            return default


class AthenaResolver:
    """Resolve source codes to OMOP standard concepts from local Athena files.

    Args:
        path: Directory containing ``CONCEPT.csv``,
            ``CONCEPT_RELATIONSHIP.csv``, and ``VOCABULARY.csv``.  A path to
            any one of those files is also accepted and resolves to its parent
            directory.
        include_cpt4: Include CPT4 rows when true.  The default is false
            because CPT4 is restricted; enabling it is valid only for a
            caller-provided bundle whose user has the applicable rights.

    The resolver is intentionally local-only regardless of environment.  When
    ``OPENMED_OFFLINE`` is set, the flag is recorded in
    :attr:`offline_requested`, and the same local code path is used without
    attempting any network access.
    """

    def __init__(self, path: str | Path, *, include_cpt4: bool = False) -> None:
        self.bundle_path = _resolve_bundle_path(path)
        self.include_cpt4 = bool(include_cpt4)
        self._offline_requested = is_local_only()
        self._concepts: dict[int, AthenaConcept] = {}
        self._source_concepts: dict[tuple[str, str], list[int]] = {}
        self._maps_to: dict[int, set[int]] = {}
        self._vocabulary_ids: set[str] = set()
        self._vocabulary_versions: dict[str, set[str]] = {}
        self._concept_hash_rows: list[tuple[str, ...]] = []
        self._relationship_hash_rows: list[tuple[str, ...]] = []
        self._vocabulary_hash_rows: list[tuple[str, ...]] = []

        files = {
            name: _find_bundle_file(self.bundle_path, name)
            for name in ATHENA_REQUIRED_FILES
        }
        for name, file_path in files.items():
            if file_path is None:
                raise FileNotFoundError(
                    f"Athena bundle is missing required file {name} in "
                    f"{self.bundle_path}"
                )

        self._load_vocabulary(files["VOCABULARY.csv"])
        self._load_concepts(files["CONCEPT.csv"])
        self._load_relationships(files["CONCEPT_RELATIONSHIP.csv"])
        self._reproducibility_hash = self._build_reproducibility_hash()

    @property
    def offline_requested(self) -> bool:
        """Return whether ``OPENMED_OFFLINE`` or local-only mode was enabled."""

        return self._offline_requested

    @property
    def local_only(self) -> bool:
        """Return true because Athena resolution never accesses a network."""

        return True

    @property
    def vocabulary_ids(self) -> tuple[str, ...]:
        """Return loaded vocabulary IDs in deterministic order."""

        return tuple(sorted(self._vocabulary_ids, key=lambda value: value.casefold()))

    @property
    def vocabulary_versions(self) -> Mapping[str, str]:
        """Return the loaded Athena version for each vocabulary ID."""

        return {
            vocabulary_id: _version_value(versions)
            for vocabulary_id, versions in sorted(
                self._vocabulary_versions.items(),
                key=lambda item: item[0].casefold(),
            )
            if versions
        }

    @property
    def vocabulary_version(self) -> str:
        """Return a pin for the loaded release.

        A single version is returned unchanged.  A bundle containing multiple
        distinct versions returns a deterministic ``VOCABULARY=VERSION`` list;
        :attr:`vocabulary_versions` remains available for structured use.
        """

        versions = self.vocabulary_versions
        unique_versions = sorted(set(versions.values()))
        if len(unique_versions) == 1:
            return unique_versions[0]
        return ";".join(
            f"{vocabulary_id}={version}" for vocabulary_id, version in versions.items()
        )

    @property
    def version(self) -> str:
        """Alias for :attr:`vocabulary_version`."""

        return self.vocabulary_version

    @property
    def vocab_version(self) -> str:
        """Alias for :attr:`vocabulary_version` used by grounding records."""

        return self.vocabulary_version

    @property
    def reproducibility_hash(self) -> str:
        """Return the stable hash of the loaded, filtered Athena set."""

        return self._reproducibility_hash

    @property
    def content_hash(self) -> str:
        """Alias for :attr:`reproducibility_hash`."""

        return self.reproducibility_hash

    @property
    def concepts(self) -> Mapping[int, AthenaConcept]:
        """Return loaded concept metadata keyed by OMOP concept ID."""

        return dict(self._concepts)

    @property
    def concept_count(self) -> int:
        """Return the number of loaded concept rows keyed by concept ID."""

        return len(self._concepts)

    @property
    def relationship_count(self) -> int:
        """Return the number of active ``Maps to`` relationships loaded."""

        return len(self._relationship_hash_rows)

    def resolve(self, system: str, code: str) -> int:
        """Return a standard concept ID for ``(system, code)`` or ``0``.

        Non-standard source concepts are followed through active ``Maps to``
        relationships.  A source concept already marked standard resolves to
        itself, which supports callers that already hold an OMOP standard code.
        No identifier is invented when the source is absent or unmapped.
        """

        concept = self.resolve_concept(system, code)
        return concept.concept_id if concept is not None else 0

    def source_code(self, system: str, code: str) -> int:
        """Resolve a source code using the issue's ``source_code`` shape."""

        return self.resolve(system, code)

    def resolve_source_code(self, system: str, code: str) -> int:
        """Alias for :meth:`resolve`."""

        return self.resolve(system, code)

    def standard_concept_id(self, system: str, code: str) -> int:
        """Return the resolved standard concept ID or ``0``."""

        return self.resolve(system, code)

    def resolve_concept(self, system: str, code: str) -> AthenaConcept | None:
        """Return metadata for the resolved standard concept, if any."""

        source = self.source_concept(system, code)
        if source is None:
            return None

        if source.is_standard:
            return source

        target_ids = self._maps_to.get(source.concept_id, ())
        targets = [
            self._concepts[target_id]
            for target_id in target_ids
            if target_id in self._concepts
            and self._is_active(self._concepts[target_id])
            and self._concepts[target_id].is_standard
        ]
        if not targets:
            return None
        return min(targets, key=lambda concept: concept.concept_id)

    def lookup(self, system: str, code: str) -> AthenaConcept | None:
        """Alias for :meth:`resolve_concept`."""

        return self.resolve_concept(system, code)

    def source_concept(self, system: str, code: str) -> AthenaConcept | None:
        """Return the active source concept before following ``Maps to``."""

        key = (_vocabulary_key(system), _clean(code))
        concept_ids = self._source_concepts.get(key, ())
        candidates = [
            self._concepts[concept_id]
            for concept_id in concept_ids
            if concept_id in self._concepts
            and self._is_active(self._concepts[concept_id])
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda concept: (
                0 if concept.is_standard else 1,
                _date_sort_key(concept.valid_start_date),
                concept.concept_id,
            ),
        )

    def get_concept(self, concept_id: int) -> AthenaConcept | None:
        """Return metadata for a loaded OMOP concept ID."""

        try:
            normalized_id = int(concept_id)
        except (TypeError, ValueError):
            return None
        return self._concepts.get(normalized_id)

    def concept_metadata(self, concept_id: int) -> dict[str, Any] | None:
        """Return a JSON-ready metadata mapping for ``concept_id``."""

        concept = self.get_concept(concept_id)
        return concept.to_dict() if concept is not None else None

    def provenance(self) -> dict[str, Any]:
        """Return PHI-free version and reproducibility metadata."""

        return {
            "vocabulary_version": self.vocabulary_version,
            "vocabulary_versions": dict(self.vocabulary_versions),
            "reproducibility_hash": self.reproducibility_hash,
            "vocabulary_ids": list(self.vocabulary_ids),
            "include_cpt4": self.include_cpt4,
            "bundled": False,
            "user_supplied": True,
            "offline": True,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return the same provenance metadata as a read-only-style mapping."""

        return self.provenance()

    def __getitem__(self, key: tuple[str, str]) -> int:
        """Provide mapping-style access for ``resolver[(system, code)]``."""

        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("AthenaResolver keys must be (system, code) tuples")
        return self.resolve(key[0], key[1])

    def __call__(self, value: Any, candidate: Any = None) -> int | None:
        """Support direct ``(system, code)`` and OMOP exporter callbacks."""

        if isinstance(value, str):
            if not isinstance(candidate, str):
                raise TypeError("direct AthenaResolver calls require system and code")
            return self.resolve(value, candidate)
        if candidate is None:
            return None
        system = getattr(candidate, "system", None)
        code = getattr(candidate, "code", None)
        if system is None or code is None:
            return None
        resolved = self.resolve(str(system), str(code))
        return resolved or None

    def _load_vocabulary(self, path: Path | None) -> None:
        assert path is not None
        for row in _iter_rows(
            path,
            _VOCABULARY_COLUMNS,
            required=_VOCABULARY_REQUIRED_COLUMNS,
        ):
            vocabulary_id = row["vocabulary_id"]
            if not vocabulary_id or self._excluded(vocabulary_id):
                continue
            version = row["vocabulary_version"]
            self._vocabulary_ids.add(vocabulary_id)
            if version:
                self._vocabulary_versions.setdefault(vocabulary_id, set()).add(version)
            self._vocabulary_hash_rows.append(
                tuple(row[column] for column in _VOCABULARY_COLUMNS)
            )

    def _load_concepts(self, path: Path | None) -> None:
        assert path is not None
        for row in _iter_rows(
            path,
            _CONCEPT_COLUMNS,
            required=_CONCEPT_REQUIRED_COLUMNS,
        ):
            vocabulary_id = row["vocabulary_id"]
            if not vocabulary_id or self._excluded(vocabulary_id):
                continue
            concept_code = row["concept_code"]
            if not concept_code:
                continue
            concept_id = _parse_int(row["concept_id"], path, "concept_id")
            standard_concept = row["standard_concept"] or None
            invalid_reason = row["invalid_reason"] or None
            concept = AthenaConcept(
                concept_id=concept_id,
                concept_name=row["concept_name"],
                domain_id=row["domain_id"],
                vocabulary_id=vocabulary_id,
                concept_class_id=row["concept_class_id"],
                standard_concept=standard_concept,
                concept_code=concept_code,
                valid_start_date=row["valid_start_date"],
                valid_end_date=row["valid_end_date"],
                invalid_reason=invalid_reason,
            )
            self._concepts.setdefault(concept_id, concept)
            self._source_concepts.setdefault(
                (_vocabulary_key(vocabulary_id), concept_code), []
            ).append(concept_id)
            self._vocabulary_ids.add(vocabulary_id)
            self._concept_hash_rows.append(
                tuple(row[column] for column in _CONCEPT_COLUMNS)
            )

    def _load_relationships(self, path: Path | None) -> None:
        assert path is not None
        for row in _iter_rows(
            path,
            _RELATIONSHIP_COLUMNS,
            required=_RELATIONSHIP_REQUIRED_COLUMNS,
        ):
            if row["relationship_id"].casefold() != _MAPS_TO:
                continue
            if row["invalid_reason"]:
                continue
            concept_id_1 = _parse_int(row["concept_id_1"], path, "concept_id_1")
            concept_id_2 = _parse_int(row["concept_id_2"], path, "concept_id_2")
            if concept_id_1 not in self._concepts or concept_id_2 not in self._concepts:
                continue
            self._maps_to.setdefault(concept_id_1, set()).add(concept_id_2)
            self._relationship_hash_rows.append(
                (
                    str(concept_id_1),
                    str(concept_id_2),
                    _clean(row["relationship_id"]),
                    row["valid_start_date"],
                    row["valid_end_date"],
                    row["invalid_reason"],
                )
            )

    def _excluded(self, vocabulary_id: str) -> bool:
        return (
            _vocabulary_key(vocabulary_id) == _vocabulary_key(CPT4_VOCABULARY_ID)
            and not self.include_cpt4
        )

    @staticmethod
    def _is_active(concept: AthenaConcept) -> bool:
        return not concept.invalid_reason

    def _build_reproducibility_hash(self) -> str:
        payload = {
            "schema_version": _HASH_SCHEMA_VERSION,
            "include_cpt4": self.include_cpt4,
            "concepts": sorted(self._concept_hash_rows),
            "relationships": sorted(self._relationship_hash_rows),
            "vocabularies": sorted(self._vocabulary_hash_rows),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resolve_bundle_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Athena bundle path does not exist: {resolved}")
    if resolved.is_file():
        resolved = resolved.parent
    if not resolved.is_dir():
        raise FileNotFoundError(f"Athena bundle path is not a directory: {resolved}")
    return resolved


def _find_bundle_file(root: Path, name: str) -> Path | None:
    exact = root / name
    if exact.is_file():
        return exact
    wanted = name.casefold()
    for candidate in root.iterdir():
        if candidate.is_file() and candidate.name.casefold() == wanted:
            return candidate
    return None


def _iter_rows(
    path: Path,
    columns: tuple[str, ...],
    *,
    required: tuple[str, ...],
) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        if not header:
            raise AthenaBundleError(f"{path.name} is empty")
        delimiter = "\t" if "\t" in header else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        header_by_key = {
            _normalise_header(field): field
            for field in (reader.fieldnames or [])
            if field is not None
        }
        missing = [
            column
            for column in required
            if _normalise_header(column) not in header_by_key
        ]
        if missing:
            raise AthenaBundleError(
                f"{path.name} is missing required columns: {', '.join(missing)}"
            )
        for raw_row in reader:
            if not any(value not in (None, "") for value in raw_row.values()):
                continue
            yield {
                column: _clean(
                    raw_row.get(header_by_key.get(_normalise_header(column)))
                )
                for column in columns
            }


def _normalise_header(value: str) -> str:
    return value.lstrip("\ufeff").strip().casefold()


def _vocabulary_key(value: object) -> str:
    return _NON_ALNUM_RE.sub("", _clean(value).casefold())


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _parse_int(value: str, path: Path, column: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AthenaBundleError(
            f"{path.name} has invalid integer in {column}: {value!r}"
        ) from exc
    if parsed <= 0:
        raise AthenaBundleError(
            f"{path.name} has non-positive integer in {column}: {value!r}"
        )
    return parsed


def _date_sort_key(value: str) -> tuple[int, str]:
    return (0 if value else 1, value)


def _version_value(versions: set[str]) -> str:
    return ",".join(sorted(versions))
