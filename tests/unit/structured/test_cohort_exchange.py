"""Stable cohort exchange envelope tests."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema.validators import validator_for

from openmed.structured.cohort import (
    CohortSourceSnapshot,
    PhenotypeDefinition,
    export_cohort_definition,
    import_cohort_definition,
    load_cohort_exchange_schema,
)
from openmed.structured.store import StoreState

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort"


def _snapshot(*, bundled: bool = False) -> CohortSourceSnapshot:
    return CohortSourceSnapshot(
        snapshot_id="snapshot_cohortsynthetic1",
        digest="sha256:" + "a" * 64,
        schema_version="omop-5.4-synthetic",
        license_tags=("synthetic",),
        bundled_vocabulary=bundled,
    )


def test_definition_envelope_round_trip_is_byte_stable_and_schema_valid() -> None:
    definition = PhenotypeDefinition.load(
        FIXTURES / "phenotypes" / "diabetes_on_metformin.json"
    )
    exported = export_cohort_definition(definition, source_snapshot=_snapshot())
    assert exported.state is StoreState.SUCCESS
    assert exported.value is not None

    restored = import_cohort_definition(exported.value.to_json_bytes())

    assert restored.state is StoreState.SUCCESS
    assert restored.value is not None
    assert restored.value.to_json_bytes() == exported.value.to_json_bytes()
    assert restored.value.definition.to_json_bytes() == definition.to_json_bytes()
    assert {
        item.criterion_id: item.concept_set_id
        for item in restored.value.evidence_mapping
    } == {item.id: item.concept_set for item in definition.criteria()}
    schema = load_cohort_exchange_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(exported.value.to_dict()))


def test_tampered_definition_digest_and_snapshot_are_conflicts() -> None:
    definition = PhenotypeDefinition.load(
        FIXTURES / "phenotypes" / "diabetes_on_metformin.json"
    )
    exported = export_cohort_definition(definition, source_snapshot=_snapshot())
    assert exported.value is not None
    payload = json.loads(exported.value.to_json())
    payload["definition_digest"] = "sha256:" + "b" * 64

    assert import_cohort_definition(payload).state is StoreState.CONFLICT
    different_snapshot = CohortSourceSnapshot(
        snapshot_id="snapshot_cohortsynthetic2",
        digest="sha256:" + "c" * 64,
        schema_version="omop-5.4-synthetic",
        license_tags=("synthetic",),
    )
    assert (
        import_cohort_definition(
            exported.value.to_dict(), expected_snapshot=different_snapshot
        ).state
        is StoreState.CONFLICT
    )


def test_bundled_vocabulary_is_rejected_by_license_boundary() -> None:
    try:
        _snapshot(bundled=True)
    except Exception as error:
        assert "cannot bundle vocabulary" in str(error)
    else:  # pragma: no cover - fail if the safety invariant is weakened
        raise AssertionError("bundled vocabulary unexpectedly accepted")
