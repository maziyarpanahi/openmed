"""Compile and execute phenotype definitions over local OMOP-style stores."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from .dsl import Criterion, Expression, PhenotypeDefinition

COHORT_RESULT_SCHEMA_VERSION: Final = "openmed.cohort.result.v1"
COHORT_PROVENANCE_SCHEMA_VERSION: Final = "openmed.cohort.provenance.v1"
COHORT_ADVISORY: Final = (
    "Cohort matches are analytical candidates for review and must not "
    "automatically trigger clinical decisions."
)

_DOMAIN_TABLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "condition_occurrence",
        "condition_occurrence_id",
        "condition_concept_id",
        "condition_start_date",
    ),
    (
        "drug_exposure",
        "drug_exposure_id",
        "drug_concept_id",
        "drug_exposure_start_date",
    ),
    (
        "measurement",
        "measurement_id",
        "measurement_concept_id",
        "measurement_date",
    ),
    (
        "procedure_occurrence",
        "procedure_occurrence_id",
        "procedure_concept_id",
        "procedure_date",
    ),
    (
        "observation",
        "observation_id",
        "observation_concept_id",
        "observation_date",
    ),
)
_REQUIRED_TABLES = frozenset(
    {"concept", "person", "note_nlp", *(item[0] for item in _DOMAIN_TABLES)}
)


@dataclass(frozen=True)
class ConceptHierarchy:
    """Caller-supplied ancestor-to-descendant concept relationships."""

    descendants: Mapping[int, tuple[int, ...]]
    source_sha256: str | None = None

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any] | Sequence[Any]],
        *,
        source_sha256: str | None = None,
    ) -> "ConceptHierarchy":
        """Build transitive descendant indexes from ancestor/descendant rows."""

        direct: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            if isinstance(row, Mapping):
                ancestor_value = row.get("ancestor_concept_id")
                descendant_value = row.get("descendant_concept_id")
            else:
                try:
                    ancestor_value, descendant_value = row[0], row[1]
                except (IndexError, TypeError) as exc:
                    raise ValueError(
                        "hierarchy rows require ancestor and descendant concept ids"
                    ) from exc
            try:
                ancestor = int(ancestor_value)
                descendant = int(descendant_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("hierarchy concept ids must be integers") from exc
            if ancestor <= 0 or descendant <= 0:
                raise ValueError("hierarchy concept ids must be positive")
            direct[ancestor].add(descendant)

        closure: dict[int, tuple[int, ...]] = {}
        all_ancestors = set(direct)
        for ancestor in sorted(all_ancestors):
            seen = {ancestor}
            pending = list(direct.get(ancestor, ()))
            while pending:
                descendant = pending.pop()
                if descendant in seen:
                    continue
                seen.add(descendant)
                pending.extend(direct.get(descendant, ()))
            closure[ancestor] = tuple(sorted(seen))
        return cls(descendants=closure, source_sha256=source_sha256)

    def expand(self, concept_ids: Iterable[int]) -> tuple[int, ...]:
        """Return roots plus every supplied descendant reachable from them."""

        expanded: set[int] = set()
        for concept_id in concept_ids:
            normalized = int(concept_id)
            expanded.add(normalized)
            expanded.update(self.descendants.get(normalized, ()))
        return tuple(sorted(expanded))


def load_athena_hierarchy(path: str | Path) -> ConceptHierarchy:
    """Load a caller-supplied Athena concept hierarchy export.

    ``CONCEPT_ANCESTOR.csv`` is preferred.  A ``CONCEPT_RELATIONSHIP.csv``
    containing ``Is a`` or ``Subsumes`` edges is also accepted and closed
    transitively.  OpenMed does not bundle either vocabulary artifact.
    """

    resolved = Path(path).expanduser()
    if resolved.is_dir():
        ancestor_path = resolved / "CONCEPT_ANCESTOR.csv"
        relationship_path = resolved / "CONCEPT_RELATIONSHIP.csv"
        if ancestor_path.exists():
            resolved = ancestor_path
        elif relationship_path.exists():
            resolved = relationship_path
        else:
            raise FileNotFoundError(
                "Athena directory must contain CONCEPT_ANCESTOR.csv or "
                "CONCEPT_RELATIONSHIP.csv"
            )
    if not resolved.is_file():
        raise FileNotFoundError("Athena hierarchy file does not exist")

    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    with resolved.open(encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = set(reader.fieldnames or ())
        if {"ancestor_concept_id", "descendant_concept_id"} <= fields:
            rows = [
                {
                    "ancestor_concept_id": row["ancestor_concept_id"],
                    "descendant_concept_id": row["descendant_concept_id"],
                }
                for row in reader
            ]
            return ConceptHierarchy.from_rows(rows, source_sha256=digest)
        required = {"concept_id_1", "concept_id_2", "relationship_id"}
        if not required <= fields:
            missing = sorted(required.difference(fields))
            raise ValueError(
                "Athena hierarchy is missing required columns: " + ", ".join(missing)
            )
        relationships = []
        for row in reader:
            if str(row.get("invalid_reason") or "").strip():
                continue
            relationship = str(row.get("relationship_id") or "").strip().casefold()
            if relationship == "is a":
                relationships.append(
                    {
                        "ancestor_concept_id": row["concept_id_2"],
                        "descendant_concept_id": row["concept_id_1"],
                    }
                )
            elif relationship == "subsumes":
                relationships.append(
                    {
                        "ancestor_concept_id": row["concept_id_1"],
                        "descendant_concept_id": row["concept_id_2"],
                    }
                )
    return ConceptHierarchy.from_rows(relationships, source_sha256=digest)


@dataclass(frozen=True)
class EvidencePointer:
    """PHI-free pointer from a matched criterion to a grounded event."""

    criterion_id: str
    concept_set_id: str
    concept_id: int
    vocabulary: str
    domain_table: str
    event_id: int
    note_id: int
    note_nlp_id: int
    source_note_hash: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        """Return offsets, hashes, and identifiers only; never source text."""

        return {
            "criterion_id": self.criterion_id,
            "concept_set_id": self.concept_set_id,
            "concept_id": self.concept_id,
            "vocabulary": self.vocabulary,
            "domain_table": self.domain_table,
            "event_id": self.event_id,
            "note_id": self.note_id,
            "note_nlp_id": self.note_nlp_id,
            "source_note_hash": self.source_note_hash,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class MatchedConceptProvenance:
    """Aggregate match counts for one expanded concept-set member."""

    concept_id: int
    patient_count: int
    occurrence_count: int

    def to_dict(self) -> dict[str, int]:
        """Return aggregate member counts."""

        return {
            "concept_id": self.concept_id,
            "patient_count": self.patient_count,
            "occurrence_count": self.occurrence_count,
        }


@dataclass(frozen=True)
class ConceptSetProvenance:
    """Expanded membership and aggregate matches for one concept set."""

    concept_set_id: str
    vocabulary: str
    requested_concept_ids: tuple[int, ...]
    expanded_concept_ids: tuple[int, ...]
    matched_members: tuple[MatchedConceptProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-free concept-set provenance."""

        return {
            "concept_set_id": self.concept_set_id,
            "vocabulary": self.vocabulary,
            "requested_concept_ids": list(self.requested_concept_ids),
            "expanded_concept_ids": list(self.expanded_concept_ids),
            "matched_members": [item.to_dict() for item in self.matched_members],
        }


@dataclass(frozen=True)
class PhenotypeProvenance:
    """Count-only phenotype resolution provenance."""

    definition_sha256: str
    hierarchy_sha256: str | None
    matched_patient_count: int
    evidence_pointer_count: int
    concept_sets: tuple[ConceptSetProvenance, ...]
    schema_version: str = COHORT_PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, PHI-free provenance report."""

        return {
            "schema_version": self.schema_version,
            "definition_sha256": self.definition_sha256,
            "hierarchy_sha256": self.hierarchy_sha256,
            "matched_patient_count": self.matched_patient_count,
            "evidence_pointer_count": self.evidence_pointer_count,
            "concept_sets": [item.to_dict() for item in self.concept_sets],
        }


@dataclass(frozen=True)
class CohortResult:
    """Resolved internal patient IDs, evidence pointers, and provenance."""

    patient_ids: tuple[int, ...]
    evidence: Mapping[int, tuple[EvidencePointer, ...]]
    provenance: PhenotypeProvenance
    schema_version: str = COHORT_RESULT_SCHEMA_VERSION

    @property
    def patient_id_set(self) -> frozenset[int]:
        """Return matched internal patient IDs as a set."""

        return frozenset(self.patient_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible response."""

        return {
            "schema_version": self.schema_version,
            "advisory": COHORT_ADVISORY,
            "patient_ids": list(self.patient_ids),
            "evidence": [
                {
                    "patient_id": patient_id,
                    "matches": [item.to_dict() for item in self.evidence[patient_id]],
                }
                for patient_id in self.patient_ids
            ],
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class CompiledPhenotypeQuery:
    """Parameterized DuckDB queries and expanded concept membership."""

    patient_sql: str
    evidence_sql: str
    parameters: tuple[Any, ...]
    expanded_concept_sets: Mapping[str, tuple[int, ...]]


def _events_cte() -> str:
    selects = []
    for table, event_id, concept_id, event_date in _DOMAIN_TABLES:
        selects.append(
            f"""
SELECT
    CAST(event.person_id AS BIGINT) AS person_id,
    CAST(event.{concept_id} AS BIGINT) AS concept_id,
    concept.vocabulary_id AS vocabulary,
    event.{event_date} AS event_date,
    '{table}' AS domain_table,
    CAST(event.{event_id} AS BIGINT) AS event_id,
    CAST(event.note_id AS BIGINT) AS note_id,
    CAST(event.note_nlp_id AS BIGINT) AS note_nlp_id,
    event.source_note_hash AS source_note_hash,
    CAST(note_nlp."offset" AS BIGINT) AS start_offset,
    CAST(note_nlp.offset_end AS BIGINT) AS end_offset,
    upper(trim(COALESCE(note_nlp.term_exists, 'Y')))
        NOT IN ('N', 'NO', 'FALSE', '0') AS term_present,
    lower(COALESCE(
        json_extract_string(TRY_CAST(note_nlp.term_modifiers AS JSON), '$.negation'),
        CASE
            WHEN upper(trim(COALESCE(note_nlp.term_exists, 'Y')))
                IN ('N', 'NO', 'FALSE', '0') THEN 'negated'
            ELSE 'affirmed'
        END
    )) AS negation,
    lower(COALESCE(
        note_nlp.term_temporal,
        json_extract_string(
            TRY_CAST(note_nlp.term_modifiers AS JSON), '$.temporality'
        ),
        'recent'
    )) AS temporality,
    lower(COALESCE(
        json_extract_string(TRY_CAST(note_nlp.term_modifiers AS JSON), '$.certainty'),
        'certain'
    )) AS certainty,
    lower(COALESCE(
        json_extract_string(
            TRY_CAST(note_nlp.term_modifiers AS JSON), '$.experiencer'
        ),
        'patient'
    )) AS experiencer
FROM {table} AS event
JOIN note_nlp ON note_nlp.note_nlp_id = event.note_nlp_id
JOIN concept ON concept.concept_id = event.{concept_id}
""".strip()
        )
    return "\nUNION ALL\n".join(selects)


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _criterion_raw_sql(
    criterion: Criterion,
    definition: PhenotypeDefinition,
    expanded: Mapping[str, tuple[int, ...]],
) -> tuple[str, list[Any]]:
    concept_set = definition.concept_set(criterion.concept_set)
    concept_ids = expanded[concept_set.id]
    filters = [
        f"event.concept_id IN ({_placeholders(len(concept_ids))})",
        "upper(event.vocabulary) = upper(?)",
    ]
    parameters: list[Any] = [*concept_ids, concept_set.vocabulary]
    negation_values = set(criterion.assertion.negation)
    if "negated" not in negation_values:
        filters.append("event.term_present = TRUE")
    elif "affirmed" not in negation_values:
        filters.append("event.term_present = FALSE")
    for column, values in (
        ("negation", criterion.assertion.negation),
        ("temporality", criterion.assertion.temporality),
        ("certainty", criterion.assertion.certainty),
        ("experiencer", criterion.assertion.experiencer),
    ):
        if values:
            filters.append(f"event.{column} IN ({_placeholders(len(values))})")
            parameters.extend(values)
    temporal = criterion.temporal
    if temporal is not None and temporal.start_date is not None:
        filters.append("TRY_CAST(event.event_date AS DATE) >= CAST(? AS DATE)")
        parameters.append(temporal.start_date)
    if temporal is not None and temporal.end_date is not None:
        filters.append("TRY_CAST(event.event_date AS DATE) <= CAST(? AS DATE)")
        parameters.append(temporal.end_date)
    return "\n        AND ".join(filters), parameters


def _temporal_order(criteria: Sequence[Criterion]) -> tuple[Criterion, ...]:
    by_id = {criterion.id: criterion for criterion in criteria}
    ordered: list[Criterion] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(criterion: Criterion) -> None:
        if criterion.id in visited:
            return
        if criterion.id in visiting:
            raise ValueError("temporal criterion anchors must be acyclic")
        visiting.add(criterion.id)
        temporal = criterion.temporal
        if temporal is not None and temporal.anchor_criterion is not None:
            visit(by_id[temporal.anchor_criterion])
        visiting.remove(criterion.id)
        visited.add(criterion.id)
        ordered.append(criterion)

    for item in criteria:
        visit(item)
    return tuple(ordered)


def _expression_ctes(
    expression: Expression,
    criterion_names: Mapping[str, str],
) -> tuple[list[str], str]:
    ctes: list[str] = []
    counter = 0

    def compile_node(node: Expression) -> str:
        nonlocal counter
        if node.criterion is not None:
            return criterion_names[node.criterion.id]
        child_names = [compile_node(child) for child in node.children]
        name = f"logical_{counter}"
        counter += 1
        if node.operator == "and":
            body = "\nINTERSECT\n".join(
                f"SELECT person_id FROM {child}" for child in child_names
            )
        elif node.operator == "or":
            body = "\nUNION\n".join(
                f"SELECT person_id FROM {child}" for child in child_names
            )
        else:
            body = (
                "SELECT person_id FROM patient_universe\nEXCEPT\n"
                f"SELECT person_id FROM {child_names[0]}"
            )
        ctes.append(f"{name} AS (\n{body}\n)")
        return name

    return ctes, compile_node(expression)


def compile_phenotype(
    definition: PhenotypeDefinition,
    *,
    hierarchy: ConceptHierarchy | None = None,
) -> CompiledPhenotypeQuery:
    """Compile a phenotype to parameterized DuckDB patient/evidence queries."""

    expanded: dict[str, tuple[int, ...]] = {}
    for concept_set in definition.concept_sets:
        if concept_set.include_descendants:
            if hierarchy is None:
                raise ValueError(
                    f"concept set {concept_set.id} requires an Athena hierarchy"
                )
            expanded[concept_set.id] = hierarchy.expand(concept_set.concept_ids)
        else:
            expanded[concept_set.id] = concept_set.concept_ids

    criteria = definition.criteria()
    criterion_indexes = {
        criterion.id: index for index, criterion in enumerate(criteria)
    }
    parameters: list[Any] = []
    ctes = [
        "patient_universe AS (\n"
        "SELECT DISTINCT CAST(person_id AS BIGINT) AS person_id FROM person\n)",
        f"events AS (\n{_events_cte()}\n)",
    ]

    for criterion in criteria:
        index = criterion_indexes[criterion.id]
        filters, raw_parameters = _criterion_raw_sql(
            criterion,
            definition,
            expanded,
        )
        ctes.append(
            f"criterion_{index}_raw AS (\n"
            "    SELECT event.* FROM events AS event\n"
            f"    WHERE {filters}\n"
            ")"
        )
        parameters.extend(raw_parameters)

    for criterion in _temporal_order(criteria):
        index = criterion_indexes[criterion.id]
        temporal = criterion.temporal
        if temporal is None or temporal.anchor_criterion is None:
            body = f"SELECT * FROM criterion_{index}_raw"
        else:
            anchor_index = criterion_indexes[temporal.anchor_criterion]
            days_before = temporal.days_before or 0
            days_after = temporal.days_after or 0
            body = f"""
SELECT DISTINCT event.*
FROM criterion_{index}_raw AS event
JOIN criterion_{anchor_index}_events AS anchor
    ON anchor.person_id = event.person_id
WHERE TRY_CAST(event.event_date AS DATE) IS NOT NULL
    AND TRY_CAST(anchor.event_date AS DATE) IS NOT NULL
    AND date_diff(
        'day',
        TRY_CAST(anchor.event_date AS DATE),
        TRY_CAST(event.event_date AS DATE)
    ) BETWEEN ? AND ?
""".strip()
            parameters.extend((-days_before, days_after))
        ctes.append(f"criterion_{index}_events AS (\n{body}\n)")

    criterion_match_names: dict[str, str] = {}
    for criterion in criteria:
        index = criterion_indexes[criterion.id]
        match_name = f"criterion_{index}_match"
        criterion_match_names[criterion.id] = match_name
        having = "COUNT(*) >= ?"
        parameters.append(criterion.occurrence.minimum)
        if criterion.occurrence.maximum is not None:
            having += " AND COUNT(*) <= ?"
            parameters.append(criterion.occurrence.maximum)
        ctes.append(
            f"{match_name} AS (\n"
            f"    SELECT person_id FROM criterion_{index}_events\n"
            "    GROUP BY person_id\n"
            f"    HAVING {having}\n"
            ")"
        )

    logical_ctes, root_name = _expression_ctes(
        definition.expression,
        criterion_match_names,
    )
    ctes.extend(logical_ctes)
    ctes.append(f"resolved_patients AS (\n    SELECT person_id FROM {root_name}\n)")
    prefix = "WITH\n" + ",\n".join(ctes)
    patient_sql = (
        prefix + "\nSELECT person_id FROM resolved_patients ORDER BY person_id"
    )

    evidence_selects = []
    for criterion in criteria:
        index = criterion_indexes[criterion.id]
        evidence_selects.append(
            f"""
SELECT
    resolved.person_id,
    '{criterion.id}' AS criterion_id,
    '{criterion.concept_set}' AS concept_set_id,
    event.concept_id,
    event.vocabulary,
    event.domain_table,
    event.event_id,
    event.note_id,
    event.note_nlp_id,
    event.source_note_hash,
    event.start_offset,
    event.end_offset
FROM resolved_patients AS resolved
JOIN criterion_{index}_match AS matched
    ON matched.person_id = resolved.person_id
JOIN criterion_{index}_events AS event
    ON event.person_id = resolved.person_id
""".strip()
        )
    evidence_sql = (
        prefix
        + "\n"
        + "\nUNION ALL\n".join(evidence_selects)
        + "\nORDER BY person_id, criterion_id, domain_table, event_id"
    )
    return CompiledPhenotypeQuery(
        patient_sql=patient_sql,
        evidence_sql=evidence_sql,
        parameters=tuple(parameters),
        expanded_concept_sets=expanded,
    )


def _safe_hash(value: Any) -> str:
    normalized = str(value or "")
    if len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    ):
        return normalized
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _provenance(
    definition: PhenotypeDefinition,
    hierarchy: ConceptHierarchy | None,
    compiled: CompiledPhenotypeQuery,
    patient_ids: tuple[int, ...],
    evidence: Mapping[int, tuple[EvidencePointer, ...]],
) -> PhenotypeProvenance:
    pointers = [item for patient in patient_ids for item in evidence[patient]]
    concept_sets = []
    for concept_set in definition.concept_sets:
        member_rows: dict[int, list[tuple[int, EvidencePointer]]] = defaultdict(list)
        for patient_id in patient_ids:
            for pointer in evidence[patient_id]:
                if pointer.concept_set_id == concept_set.id:
                    member_rows[pointer.concept_id].append((patient_id, pointer))
        matched_members = tuple(
            MatchedConceptProvenance(
                concept_id=concept_id,
                patient_count=len({row[0] for row in rows}),
                occurrence_count=len(
                    {
                        (
                            row[0],
                            row[1].domain_table,
                            row[1].event_id,
                            row[1].note_nlp_id,
                        )
                        for row in rows
                    }
                ),
            )
            for concept_id, rows in sorted(member_rows.items())
        )
        concept_sets.append(
            ConceptSetProvenance(
                concept_set_id=concept_set.id,
                vocabulary=concept_set.vocabulary,
                requested_concept_ids=concept_set.concept_ids,
                expanded_concept_ids=compiled.expanded_concept_sets[concept_set.id],
                matched_members=matched_members,
            )
        )
    return PhenotypeProvenance(
        definition_sha256=definition.sha256,
        hierarchy_sha256=hierarchy.source_sha256 if hierarchy is not None else None,
        matched_patient_count=len(patient_ids),
        evidence_pointer_count=len(pointers),
        concept_sets=tuple(concept_sets),
    )


class CohortResolver:
    """Resolve phenotypes through one existing DuckDB connection."""

    def __init__(
        self,
        connection: Any,
        *,
        hierarchy: ConceptHierarchy | None = None,
    ) -> None:
        self.connection = connection
        self.hierarchy = hierarchy

    def resolve(self, definition: PhenotypeDefinition) -> CohortResult:
        """Execute one phenotype and return privacy-minimized matches."""

        self._validate_tables()
        compiled = compile_phenotype(definition, hierarchy=self.hierarchy)
        patient_rows = self.connection.execute(
            compiled.patient_sql,
            compiled.parameters,
        ).fetchall()
        patient_ids = tuple(sorted(int(row[0]) for row in patient_rows))

        evidence_by_patient: dict[int, list[EvidencePointer]] = {
            patient_id: [] for patient_id in patient_ids
        }
        if patient_ids:
            evidence_rows = self.connection.execute(
                compiled.evidence_sql,
                compiled.parameters,
            ).fetchall()
            for row in evidence_rows:
                patient_id = int(row[0])
                evidence_by_patient[patient_id].append(
                    EvidencePointer(
                        criterion_id=str(row[1]),
                        concept_set_id=str(row[2]),
                        concept_id=int(row[3]),
                        vocabulary=str(row[4] or ""),
                        domain_table=str(row[5]),
                        event_id=int(row[6]),
                        note_id=int(row[7]),
                        note_nlp_id=int(row[8]),
                        source_note_hash=_safe_hash(row[9]),
                        start=int(row[10]),
                        end=int(row[11]),
                    )
                )
        evidence = {
            patient_id: tuple(items)
            for patient_id, items in sorted(evidence_by_patient.items())
        }
        provenance = _provenance(
            definition,
            self.hierarchy,
            compiled,
            patient_ids,
            evidence,
        )
        return CohortResult(
            patient_ids=patient_ids,
            evidence=evidence,
            provenance=provenance,
        )

    def _validate_tables(self) -> None:
        rows = self.connection.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
        available = {str(row[0]) for row in rows}
        missing = sorted(_REQUIRED_TABLES.difference(available))
        if missing:
            raise ValueError(
                "cohort source is missing required tables: " + ", ".join(missing)
            )


def _load_duckdb() -> Any:
    try:
        return import_module("duckdb")
    except ImportError as exc:
        raise ImportError(
            "Cohort resolution requires the optional dependency; install "
            "openmed[duckdb]"
        ) from exc


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parquet_connection(directory: str | Path) -> Any:
    source = Path(directory).expanduser()
    if not source.is_dir():
        raise FileNotFoundError("cohort Parquet source directory does not exist")
    missing = sorted(
        table
        for table in _REQUIRED_TABLES
        if not (source / f"{table}.parquet").is_file()
    )
    if missing:
        raise ValueError(
            "cohort Parquet source is missing tables: " + ", ".join(missing)
        )
    connection = _load_duckdb().connect(":memory:")
    for table in sorted(_REQUIRED_TABLES):
        parquet_path = str(source / f"{table}.parquet")
        connection.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet({_sql_literal(parquet_path)})"
        )
    return connection


def resolve_phenotype(
    definition: PhenotypeDefinition,
    connection: Any | None = None,
    *,
    duckdb_path: str | Path | None = None,
    parquet_directory: str | Path | None = None,
    hierarchy: ConceptHierarchy | None = None,
) -> CohortResult:
    """Resolve a phenotype against a connection, DuckDB file, or Parquet set.

    Exactly one source must be supplied.  Connections owned by the caller are
    left open; connections created for file-backed sources are always closed.
    """

    sources = sum(
        value is not None for value in (connection, duckdb_path, parquet_directory)
    )
    if sources != 1:
        raise ValueError(
            "provide exactly one cohort source: connection, duckdb_path, or "
            "parquet_directory"
        )

    owned_connection = False
    if duckdb_path is not None:
        path = Path(duckdb_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("cohort DuckDB source file does not exist")
        connection = _load_duckdb().connect(str(path), read_only=True)
        owned_connection = True
    elif parquet_directory is not None:
        connection = _parquet_connection(parquet_directory)
        owned_connection = True

    if connection is None:  # pragma: no cover - source count invariant
        raise ValueError("cohort source connection is required")
    try:
        return CohortResolver(connection, hierarchy=hierarchy).resolve(definition)
    finally:
        if owned_connection:
            connection.close()


__all__ = [
    "COHORT_ADVISORY",
    "COHORT_PROVENANCE_SCHEMA_VERSION",
    "COHORT_RESULT_SCHEMA_VERSION",
    "CohortResolver",
    "CohortResult",
    "CompiledPhenotypeQuery",
    "ConceptHierarchy",
    "ConceptSetProvenance",
    "EvidencePointer",
    "MatchedConceptProvenance",
    "PhenotypeProvenance",
    "compile_phenotype",
    "load_athena_hierarchy",
    "resolve_phenotype",
]
