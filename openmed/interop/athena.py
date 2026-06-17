"""Load user-supplied OHDSI Athena vocabularies and Usagi mappings.

The loader accepts Athena export files supplied by the caller. OpenMed does
not ship Athena, OMOP, UMLS, SNOMED, CPT4, or other restricted vocabulary
content. Returned indexes include license and provenance metadata so downstream
exporters can preserve the source and redistribution status of the data they
were built from.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ConceptRecord = dict[str, Any]
ConceptIndex = dict[str, dict[str, ConceptRecord]]
AthenaVocabularyIndex = dict[str, dict[str, ConceptRecord] | dict[str, Any]]
UsagiMapping = dict[str, int]

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
_SYNONYM_COLUMNS = ("concept_id", "concept_synonym_name")

_SOURCE_CODE_FIELDS = ("sourceCode", "source_code")
_SOURCE_VOCAB_FIELDS = ("sourceVocabularyId", "source_vocabulary_id")
_STATUS_FIELDS = ("mappingStatus", "mapping_status")
_EQUIVALENCE_FIELDS = ("equivalence",)
_CONCEPT_ID_FIELDS = (
    "conceptId",
    "concept_id",
    "targetConceptId",
    "target_concept_id",
    "standardConceptId",
    "standard_concept_id",
)

_ATHENA_LICENSE_NOTICE = (
    "OHDSI Athena vocabulary content is user-supplied. OHDSI content is "
    "generally CC BY-SA 4.0, but restricted vocabularies require the user's "
    "own rights and must not be redistributed through OpenMed."
)


def load_athena_vocab(
    path: str | Path,
    *,
    include_synonyms: bool = True,
    vocabulary_ids: Iterable[str] | None = None,
) -> AthenaVocabularyIndex:
    """Parse an Athena export into a vocabulary/code concept index.

    Parameters
    ----------
    path:
        Directory containing ``CONCEPT.csv`` and optionally
        ``CONCEPT_SYNONYM.csv``. Passing the ``CONCEPT.csv`` file itself is
        also supported.
    include_synonyms:
        When true, attach synonyms from ``CONCEPT_SYNONYM.csv`` if the file is
        present.
    vocabulary_ids:
        Optional allow-list of vocabulary IDs to load. ``None`` loads every
        vocabulary in the export; an empty iterable loads none.

    Returns
    -------
    dict
        ``{vocabulary_id: {concept_code: concept_record}, "_meta": ...}``.
        Each concept record includes ``aliases`` and ``synonyms`` lists. The
        ``aliases`` list contains the primary concept name plus synonyms.
    """

    export_dir = _resolve_athena_dir(path)
    concept_csv = export_dir / "CONCEPT.csv"
    synonym_csv = export_dir / "CONCEPT_SYNONYM.csv"

    if not concept_csv.exists():
        raise FileNotFoundError(f"CONCEPT.csv not found in {export_dir}")

    allowed_vocabs = (
        {str(vocab).strip() for vocab in vocabulary_ids}
        if vocabulary_ids is not None
        else None
    )

    index: ConceptIndex = {}
    concept_id_to_records: dict[int, list[ConceptRecord]] = {}
    concept_count = 0

    with concept_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(reader.fieldnames, _CONCEPT_COLUMNS, concept_csv)

        for line_number, row in enumerate(reader, start=2):
            vocabulary_id = _clean(row.get("vocabulary_id"))
            if allowed_vocabs is not None and vocabulary_id not in allowed_vocabs:
                continue

            concept_code = _clean(row.get("concept_code"))
            if not concept_code:
                continue

            concept_id = _parse_required_int(
                row.get("concept_id"), "concept_id", concept_csv, line_number
            )
            concept_name = _clean(row.get("concept_name"))
            record: ConceptRecord = {
                "concept_id": concept_id,
                "concept_name": concept_name,
                "domain_id": _clean(row.get("domain_id")),
                "vocabulary_id": vocabulary_id,
                "concept_class_id": _clean(row.get("concept_class_id")),
                "standard_concept": _clean(row.get("standard_concept")) or None,
                "concept_code": concept_code,
                "valid_start_date": _clean(row.get("valid_start_date")) or None,
                "valid_end_date": _clean(row.get("valid_end_date")) or None,
                "invalid_reason": _clean(row.get("invalid_reason")) or None,
                "synonyms": [],
                "aliases": _dedupe([concept_name]),
            }

            index.setdefault(vocabulary_id, {})[concept_code] = record
            concept_id_to_records.setdefault(concept_id, []).append(record)
            concept_count += 1

    synonym_count = 0
    if include_synonyms and synonym_csv.exists():
        with synonym_csv.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _require_columns(reader.fieldnames, _SYNONYM_COLUMNS, synonym_csv)

            for line_number, row in enumerate(reader, start=2):
                concept_id = _parse_required_int(
                    row.get("concept_id"), "concept_id", synonym_csv, line_number
                )
                synonym = _clean(row.get("concept_synonym_name"))
                if not synonym:
                    continue

                records = concept_id_to_records.get(concept_id, ())
                if not records:
                    continue

                for record in records:
                    _append_unique(record["synonyms"], synonym)
                    _append_unique(record["aliases"], synonym)
                synonym_count += 1

    result: AthenaVocabularyIndex = dict(index)
    result["_meta"] = {
        "source": str(export_dir),
        "vocabulary_ids": sorted(index),
        "concept_count": concept_count,
        "synonym_count": synonym_count,
        "license": _ATHENA_LICENSE_NOTICE,
        "provenance": {
            "user_supplied": True,
            "bundled": False,
            "concept_file": str(concept_csv),
            "synonym_file": (
                str(synonym_csv) if include_synonyms and synonym_csv.exists() else None
            ),
        },
    }
    return result


def load_usagi_mapping(
    path: str | Path,
    *,
    approved_only: bool = True,
    min_equivalence: str | None = None,
) -> UsagiMapping:
    """Parse a Usagi CSV export into source-code to standard-concept mappings.

    Parameters
    ----------
    path:
        Usagi export CSV path.
    approved_only:
        When true, skip rows whose ``mappingStatus`` is present and is not
        ``APPROVED``.
    min_equivalence:
        Optional exact equivalence filter such as ``"EQUIVALENT"``.

    Returns
    -------
    dict[str, int]
        Mapping from ``sourceCode`` to target concept ID. If the export includes
        ``sourceVocabularyId``, keys use ``"{sourceVocabularyId}:{sourceCode}"``.
    """

    mapping_csv = Path(path).expanduser()
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Usagi mapping file not found: {mapping_csv}")

    mapping: UsagiMapping = {}
    with mapping_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_any_column(reader.fieldnames, _SOURCE_CODE_FIELDS, mapping_csv)
        _require_any_column(reader.fieldnames, _CONCEPT_ID_FIELDS, mapping_csv)

        for row in reader:
            if approved_only:
                status = _row_value(row, _STATUS_FIELDS).upper()
                if status and status != "APPROVED":
                    continue

            if min_equivalence is not None:
                equivalence = _row_value(row, _EQUIVALENCE_FIELDS).upper()
                if equivalence != min_equivalence.upper():
                    continue

            concept_id = _optional_int(_row_value(row, _CONCEPT_ID_FIELDS))
            if concept_id is None or concept_id == 0:
                continue

            source_code = _row_value(row, _SOURCE_CODE_FIELDS)
            if not source_code:
                continue

            source_vocabulary = _row_value(row, _SOURCE_VOCAB_FIELDS)
            key = (
                f"{source_vocabulary}:{source_code}"
                if source_vocabulary
                else source_code
            )
            mapping[key] = concept_id

    return mapping


def _resolve_athena_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Athena export path does not exist: {resolved}")
    return resolved.parent if resolved.is_file() else resolved


def _require_columns(
    fieldnames: list[str] | None, required: Iterable[str], path: Path
) -> None:
    present = set(fieldnames or ())
    missing = [column for column in required if column not in present]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")


def _require_any_column(
    fieldnames: list[str] | None, candidates: Iterable[str], path: Path
) -> None:
    present = {name.lower() for name in fieldnames or ()}
    if not any(candidate.lower() in present for candidate in candidates):
        expected = ", ".join(candidates)
        raise ValueError(f"{path.name} is missing one of: {expected}")


def _row_value(row: dict[str, str | None], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in row:
            return _clean(row.get(candidate))

    lower_to_key = {key.lower(): key for key in row if key is not None}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return _clean(row.get(key))

    return ""


def _parse_required_int(
    value: str | None, column: str, path: Path, line_number: int
) -> int:
    cleaned = _clean(value)
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"{path.name}:{line_number} has invalid integer in {column}: {cleaned!r}"
        ) from exc


def _optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _clean(value: str | None) -> str:
    return value.strip() if value is not None else ""


def _dedupe(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        _append_unique(deduped, value)
    return deduped


def _append_unique(values: Any, value: str) -> None:
    if not isinstance(values, list):
        raise TypeError("expected list for alias collection")
    if value and value not in values:
        values.append(value)


__all__ = [
    "AthenaVocabularyIndex",
    "ConceptIndex",
    "ConceptRecord",
    "UsagiMapping",
    "load_athena_vocab",
    "load_usagi_mapping",
]
