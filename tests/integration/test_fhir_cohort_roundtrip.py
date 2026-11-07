"""End-to-end cohort membership reproducibility after envelope round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.interop.omop import load_grounded_jsonl, write_omop_duckdb
from openmed.structured.cohort import (
    CohortSourceSnapshot,
    PhenotypeDefinition,
    export_cohort_definition,
    import_cohort_definition,
    load_athena_hierarchy,
    resolve_phenotype,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cohort"


def test_cohort_membership_is_identical_after_exchange_round_trip() -> None:
    definition = PhenotypeDefinition.load(
        FIXTURES / "phenotypes" / "diabetes_on_metformin.json"
    )
    envelope = export_cohort_definition(
        definition,
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_cohortintegration1",
            digest="sha256:" + "a" * 64,
            schema_version="omop-5.4-synthetic",
            license_tags=("synthetic",),
        ),
    )
    assert envelope.value is not None
    restored = import_cohort_definition(envelope.value.to_json_bytes())
    assert restored.value is not None
    hierarchy = load_athena_hierarchy(FIXTURES / "athena")
    connection = write_omop_duckdb(
        load_grounded_jsonl(FIXTURES / "synthetic_grounded.jsonl")
    )
    try:
        first = resolve_phenotype(definition, connection, hierarchy=hierarchy)
        second = resolve_phenotype(
            restored.value.definition,
            connection,
            hierarchy=hierarchy,
        )
    finally:
        connection.close()

    assert first.patient_ids == second.patient_ids
    assert first.to_dict() == second.to_dict()
