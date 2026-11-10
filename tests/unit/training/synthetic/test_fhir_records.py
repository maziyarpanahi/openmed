"""Synthetic FHIR core resources, referential closure and negative controls."""

from copy import deepcopy
from dataclasses import replace

import pytest

from openmed.clinical.journey_contracts import ClinicalFact, derived_opaque_id
from openmed.training.synthetic.fhir_records import (
    generate_fhir_records,
    validate_synthetic_fhir,
)


def facts():
    subject = derived_opaque_id("subject", "synthetic")
    encounter = derived_opaque_id("encounter", "synthetic")
    rows = []
    for i, kind in enumerate(
        ("encounter", "condition", "medication", "laboratory", "procedure", "outcome")
    ):
        rows.append(
            ClinicalFact(
                derived_opaque_id("fact", kind),
                subject,
                kind,
                {"synthetic_code": "synthetic_" + kind},
                "synthetic",
                (derived_opaque_id("evidence", kind),),
                "a" * 64,
                encounter_id=encounter,
                effective_time={"date": "2000-01-01"},
                attributes={"synthetic": True, "observed_at": "2000-01-02T00:00:00Z"},
            )
        )
    return tuple(rows)


def test_deterministic_core_resources_and_all_references_resolve():
    bundle = generate_fhir_records(facts(), seed=42)
    assert bundle == generate_fhir_records(tuple(reversed(facts())), seed=42)
    assert validate_synthetic_fhir(bundle) == ()
    kinds = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert set(kinds) == {
        "Patient",
        "Encounter",
        "Condition",
        "MedicationStatement",
        "Observation",
        "Procedure",
        "Provenance",
    }
    assert kinds.count("Provenance") == 6
    assert all("request" not in e for e in bundle["entry"])


def test_dangling_reference_and_missing_cardinality_are_independently_caught():
    bundle = generate_fhir_records(facts())
    bad = deepcopy(bundle)
    row = next(
        e["resource"]
        for e in bad["entry"]
        if e["resource"]["resourceType"] == "Condition"
    )
    row["subject"]["reference"] = "urn:uuid:missing"
    assert "dangling_reference" in validate_synthetic_fhir(bad)
    del row["subject"]
    assert "base_cardinality_or_type" in validate_synthetic_fhir(bad)


def test_synthetic_markers_chronology_and_no_write_requests_are_checked():
    bundle = generate_fhir_records(facts())
    bad = deepcopy(bundle)
    bad["entry"][0]["resource"]["meta"] = {}
    assert "synthetic_tag_missing" in validate_synthetic_fhir(bad)
    bad = deepcopy(bundle)
    obs = next(
        e["resource"]
        for e in bad["entry"]
        if e["resource"]["resourceType"] == "Observation"
    )
    obs["effectiveDateTime"] = "1999-01-01"
    assert "clinical_time_before_encounter" in validate_synthetic_fhir(bad)
    bad["entry"][0]["request"] = {"method": "POST", "url": "Patient"}
    assert "unexpected_write_request" in validate_synthetic_fhir(bad)


def test_non_synthetic_missing_encounters_and_future_dates_reject():
    rows = facts()
    with pytest.raises(ValueError, match="synthetic"):
        generate_fhir_records((replace(rows[0], status="current"),) + rows[1:])
    with pytest.raises(ValueError, match="encounter"):
        generate_fhir_records(rows[1:])
    with pytest.raises(ValueError, match="chronology"):
        generate_fhir_records(
            (replace(rows[0], effective_time={"date": "2001-01-01"}),) + rows[1:]
        )


def test_duplicate_fact_id_and_oversize_input_reject():
    rows = facts()
    with pytest.raises(ValueError, match="duplicate"):
        generate_fhir_records(rows + (rows[0],))
    with pytest.raises(ValueError, match="collection"):
        generate_fhir_records(iter([rows[0]] * 1001))


def test_malformed_metadata_and_deep_reference_structures_fail_controlled_checks():
    bundle = generate_fhir_records(facts())
    bad = deepcopy(bundle)
    bad["entry"][0]["resource"]["meta"] = None
    assert "synthetic_tag_missing" in validate_synthetic_fhir(bad)
    bad = deepcopy(bundle)
    nested = {}
    current = nested
    for _ in range(40):
        current["next"] = {}
        current = current["next"]
    bad["entry"][0]["resource"]["extra"] = nested
    assert "reference_structure_limit" in validate_synthetic_fhir(bad)


def test_same_day_causal_cycle_does_not_pass_fixture_generation():
    rows = list(facts())
    rows[1] = replace(rows[1], parent_fact_ids=(rows[2].fact_id,))
    rows[2] = replace(rows[2], parent_fact_ids=(rows[1].fact_id,))
    with pytest.raises(ValueError, match="causal cycle"):
        generate_fhir_records(tuple(rows))


def test_all_and_partial_provenance_omission_are_rejected():
    original = generate_fhir_records(facts())
    for remove_all in (True, False):
        bad = deepcopy(original)
        omitted = False
        retained = []
        for entry in bad["entry"]:
            if entry["resource"]["resourceType"] == "Provenance" and (
                remove_all or not omitted
            ):
                omitted = True
                continue
            retained.append(entry)
        bad["entry"] = retained
        assert "provenance_coverage_missing" in validate_synthetic_fhir(bad)


def test_existing_wrong_reference_types_are_rejected():
    bad = generate_fhir_records(facts())
    encounter = next(
        e for e in bad["entry"] if e["resource"]["resourceType"] == "Encounter"
    )
    condition = next(
        e["resource"]
        for e in bad["entry"]
        if e["resource"]["resourceType"] == "Condition"
    )
    condition["subject"]["reference"] = encounter["fullUrl"]
    assert "subject_reference_type" in validate_synthetic_fhir(bad)
    condition["encounter"]["reference"] = next(
        e["fullUrl"] for e in bad["entry"] if e["resource"]["resourceType"] == "Patient"
    )
    assert "encounter_reference_type" in validate_synthetic_fhir(bad)


@pytest.mark.parametrize(
    "encounter_start,should_fail",
    [("2000-01-01T00:00:00-12:00", True), ("2000-01-01T03:00:00+12:00", False)],
)
def test_temporal_fixture_order_uses_timezone_instants(encounter_start, should_fail):
    bad = generate_fhir_records(facts())
    encounter = next(
        e["resource"]
        for e in bad["entry"]
        if e["resource"]["resourceType"] == "Encounter"
    )
    condition = next(
        e["resource"]
        for e in bad["entry"]
        if e["resource"]["resourceType"] == "Condition"
    )
    encounter["period"]["start"] = encounter_start
    condition["onsetDateTime"] = "2000-01-01T01:00:00Z"
    assert (
        "clinical_time_before_encounter" in validate_synthetic_fhir(bad)
    ) is should_fail


@pytest.mark.parametrize(
    "recorded,category",
    [
        ("not_a_datetime", "provenance_time_invalid"),
        ("2000-01-02T00:00:00", "provenance_time_invalid"),
        ("1999-12-31T23:00:00Z", "provenance_before_target"),
    ],
)
def test_provenance_recorded_time_is_valid_aware_and_not_before_target(
    recorded, category
):
    bad = generate_fhir_records(facts())
    provenance = next(
        e["resource"]
        for e in bad["entry"]
        if e["resource"]["resourceType"] == "Provenance"
    )
    provenance["recorded"] = recorded
    assert category in validate_synthetic_fhir(bad)
