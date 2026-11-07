"""Synthetic OMOP relational coherence and independently corrupted cohort truth."""

from copy import deepcopy
from dataclasses import replace

import pytest

from openmed.clinical.journey_contracts import ClinicalFact, derived_opaque_id
from openmed.training.synthetic.omop_cohorts import (
    generate_omop_cohort,
    validate_synthetic_omop,
)


def facts():
    rows = []
    for person in ("one", "two"):
        subject = derived_opaque_id("subject", person)
        encounter = derived_opaque_id("encounter", person)
        for kind in ("encounter", "condition", "medication", "laboratory"):
            code = (
                "synthetic_condition_one"
                if kind == "condition" and person == "one"
                else "synthetic_" + kind
            )
            rows.append(
                ClinicalFact(
                    derived_opaque_id("fact", person, kind),
                    subject,
                    kind,
                    {"synthetic_code": code},
                    "synthetic",
                    (derived_opaque_id("evidence", person, kind),),
                    "a" * 64,
                    encounter_id=encounter,
                    effective_time={"date": "2000-01-01"},
                    attributes={"synthetic": True},
                )
            )
    return tuple(rows)


def test_deterministic_tables_and_gold_membership_from_declared_conditions():
    export = generate_omop_cohort(
        facts(), seed=42, cohort_condition_codes=("synthetic_condition_one",)
    )
    assert export == generate_omop_cohort(
        tuple(reversed(facts())),
        seed=42,
        cohort_condition_codes=("synthetic_condition_one",),
    )
    assert validate_synthetic_omop(export) == ()
    assert len(export["tables"]["person"]) == 2
    assert len(export["tables"]["visit_occurrence"]) == 2
    assert len(export["cohort"]["member_person_ids"]) == 1
    assert all(
        r["vocabulary_id"] == "OPENMED_SYNTHETIC" for r in export["tables"]["concept"]
    )
    assert all(r["value_as_number"] is None for r in export["tables"]["measurement"])


@pytest.mark.parametrize(
    "table", ["condition_occurrence", "drug_exposure", "measurement"]
)
def test_orphan_people_and_cross_person_visits_fail_existing_runtime_checker(table):
    export = generate_omop_cohort(facts())
    bad = deepcopy(export)
    bad["tables"][table][0]["person_id"] = 999999
    assert "relational_or_concept_constraint" in validate_synthetic_omop(bad)
    bad = deepcopy(export)
    row = bad["tables"][table][0]
    other = next(
        v
        for v in bad["tables"]["visit_occurrence"]
        if v["person_id"] != row["person_id"]
    )
    row["visit_occurrence_id"] = other["visit_occurrence_id"]
    assert "relational_or_concept_constraint" in validate_synthetic_omop(bad)


def test_reversed_time_cohort_truth_and_actual_synthetic_note_hash_are_checked():
    export = generate_omop_cohort(facts())
    bad = deepcopy(export)
    bad["tables"]["visit_occurrence"][0]["visit_end_date"] = "1999-01-01"
    assert "visit_time_reversed" in validate_synthetic_omop(bad)
    bad = deepcopy(export)
    bad["cohort"]["member_person_ids"] = []
    assert "cohort_truth_mismatch" in validate_synthetic_omop(bad)
    bad = deepcopy(export)
    bad["tables"]["note"][0]["note_text"] = "changed synthetic note"
    assert "synthetic_note_hash_mismatch" in validate_synthetic_omop(bad)


def test_real_or_missing_visit_graph_inputs_reject():
    rows = facts()
    with pytest.raises(ValueError, match="synthetic"):
        generate_omop_cohort((replace(rows[0], status="current"),) + rows[1:])
    with pytest.raises(ValueError, match="visit"):
        generate_omop_cohort(tuple(f for f in rows if f.fact_type != "encounter"))
    with pytest.raises(ValueError, match="criteria"):
        generate_omop_cohort(rows, cohort_condition_codes=("private raw text",))


def test_same_day_causal_cycles_and_malformed_table_collections_reject():
    rows = list(facts())
    rows[1] = replace(rows[1], parent_fact_ids=(rows[2].fact_id,))
    rows[2] = replace(rows[2], parent_fact_ids=(rows[1].fact_id,))
    with pytest.raises(ValueError, match="causal cycle"):
        generate_omop_cohort(tuple(rows))
    with pytest.raises(ValueError, match="row collection"):
        validate_synthetic_omop({"tables": {"person": "invalid"}})


def test_demographic_fixture_defaults_stay_coherent_for_early_visit_dates():
    rows = tuple(replace(f, effective_time={"date": "1900-01-01"}) for f in facts())
    export = generate_omop_cohort(rows)
    assert all(r["year_of_birth"] == 1870 for r in export["tables"]["person"])
    bad = deepcopy(export)
    bad["tables"]["person"][0]["year_of_birth"] = 2000
    assert "invalid_birth_visit_chronology" in validate_synthetic_omop(bad)


def test_invalid_dates_produce_categories_without_echoing_input_values():
    export = generate_omop_cohort(facts())
    bad = deepcopy(export)
    bad["tables"]["visit_occurrence"][0]["visit_start_date"] = "private raw text"
    result = validate_synthetic_omop(bad)
    assert "invalid_visit_date" in result
    assert "private raw text" not in repr(result)


def test_malformed_relational_rows_return_controlled_failure():
    original = generate_omop_cohort(facts())
    for table, field, value in [
        ("person", "person_id", None),
        ("measurement", "visit_occurrence_id", []),
        ("person", "person_id", []),
    ]:
        bad = deepcopy(original)
        if value is None:
            bad["tables"][table][0].pop(field)
        else:
            bad["tables"][table][0][field] = value
        assert "relational_or_concept_constraint" in validate_synthetic_omop(bad)


def test_malformed_cohort_declarations_return_controlled_failure():
    original = generate_omop_cohort(facts())
    for field, value in [
        ("required_condition_codes", [[]]),
        ("member_person_ids", [{}]),
        ("member_person_ids", [True]),
        ("required_condition_codes", ["raw submitted prose"]),
    ]:
        bad = deepcopy(original)
        bad["cohort"][field] = value
        assert "invalid_cohort_structure" in validate_synthetic_omop(bad)


def test_noncanonical_effective_dates_reject_before_generation():
    altered = tuple(
        replace(fact, effective_time={"date": "20200101"}) for fact in facts()
    )
    with pytest.raises(ValueError, match="effective date"):
        generate_omop_cohort(altered)
