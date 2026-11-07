"""Coherent synthetic OMOP-shaped cohorts from existing Journey clinical facts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date
from itertools import islice

from openmed.clinical.journey_contracts import ClinicalFact, sha256_digest
from openmed.interop.omop_cohort_check import validate_omop_cohort_export

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_DOMAINS = {
    "condition": ("condition_occurrence", "condition", "Condition"),
    "medication": ("drug_exposure", "drug", "Drug"),
    "laboratory": ("measurement", "measurement", "Measurement"),
    "procedure": ("procedure_occurrence", "procedure", "Procedure"),
    "outcome": ("observation", "observation", "Observation"),
}


def _identity(namespace: str, value: str, seed: int) -> int:
    # Keep IDs inside positive signed BIGINT without aliasing textual identities.
    return (
        int(hashlib.sha256(f"{namespace}:{value}:{seed}".encode()).hexdigest()[:15], 16)
        + 1
    )


def generate_omop_cohort(
    facts: tuple[ClinicalFact, ...],
    *,
    seed: int = 0,
    cohort_condition_codes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Create relational synthetic person/visit/condition/drug/measurement fixtures.

    Local concepts are explicit OPENMED_SYNTHETIC placeholders, not Athena,
    SNOMED, CPT or standard clinical vocabulary claims. No database writes occur.
    """
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("invalid synthetic OMOP seed")
    facts = tuple(islice(facts, 1001))
    if (
        not facts
        or len(facts) > 1000
        or any(not isinstance(f, ClinicalFact) for f in facts)
    ):
        raise ValueError("invalid synthetic OMOP fact collection")
    criteria = tuple(islice(cohort_condition_codes, 129))
    if (
        len(criteria) > 128
        or any(not isinstance(c, str) or not _CODE.fullmatch(c) for c in criteria)
        or len(set(criteria)) != len(criteria)
    ):
        raise ValueError("invalid synthetic OMOP cohort criteria")
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("duplicate synthetic OMOP fact identity")
    facts = tuple(sorted(facts, key=lambda f: (f.subject_id, f.fact_id)))
    encounters = {}
    for fact in facts:
        if (
            fact.attributes.get("synthetic") is not True
            or fact.status != "synthetic"
            or fact.fact_type not in _DOMAINS.keys() | {"encounter"}
        ):
            raise ValueError("OMOP fixture requires declared synthetic facts")
        if not isinstance(fact.value, Mapping) or set(fact.value) != {"synthetic_code"}:
            raise ValueError("invalid synthetic OMOP fact value")
        code = fact.value["synthetic_code"]
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise ValueError("invalid synthetic OMOP concept placeholder")
        try:
            effective = fact.effective_time["date"]
            if (
                not isinstance(effective, str)
                or date.fromisoformat(effective).isoformat() != effective
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ValueError("invalid synthetic OMOP effective date") from None
        if fact.fact_type == "encounter":
            if fact.encounter_id is None or fact.encounter_id in encounters:
                raise ValueError("invalid synthetic OMOP encounter inventory")
            encounters[fact.encounter_id] = fact
    indexed = {f.fact_id: f for f in facts}
    for fact in facts:
        encounter = encounters.get(fact.encounter_id)
        if (
            encounter is None
            or encounter.subject_id != fact.subject_id
            or encounter.effective_time["date"] > fact.effective_time["date"]
        ):
            raise ValueError("invalid synthetic OMOP visit relationship")
        if any(
            p not in indexed
            or indexed[p].subject_id != fact.subject_id
            or indexed[p].effective_time["date"] > fact.effective_time["date"]
            for p in fact.parent_fact_ids
        ):
            raise ValueError("invalid synthetic OMOP parent relationship")
    completed = set()
    remaining = set(indexed)
    while remaining:
        ready = {
            code
            for code in remaining
            if set(indexed[code].parent_fact_ids) <= completed
        }
        if not ready:
            raise ValueError("synthetic graph contains a causal cycle")
        completed |= ready
        remaining -= ready
    subjects = sorted({f.subject_id for f in facts})
    persons = {subject: _identity("person", subject, seed) for subject in subjects}
    visits = {
        encounter: _identity("visit", encounter, seed) for encounter in encounters
    }
    tables = {
        name: []
        for name in (
            "concept",
            "person",
            "visit_occurrence",
            "note",
            "condition_occurrence",
            "drug_exposure",
            "measurement",
            "procedure_occurrence",
            "observation",
        )
    }
    tables["concept"].append(
        {
            "concept_id": 0,
            "concept_name": "No matching synthetic concept",
            "domain_id": "Metadata",
            "vocabulary_id": "OPENMED_SYNTHETIC",
            "concept_class_id": "Synthetic",
            "standard_concept": "",
            "concept_code": "0",
            "synthetic": True,
        }
    )
    for subject in subjects:
        first_visit_year = min(
            int(fact.effective_time["date"][:4])
            for fact in facts
            if fact.subject_id == subject
        )
        synthetic_birth_year = max(1, first_visit_year - 30)
        tables["person"].append(
            {
                "person_id": persons[subject],
                "gender_concept_id": 0,
                "year_of_birth": synthetic_birth_year,
                "race_concept_id": 0,
                "ethnicity_concept_id": 0,
                "person_source_value": "synthetic_person",
                "synthetic": True,
            }
        )
    for encounter, fact in sorted(encounters.items()):
        related = [
            f.effective_time["date"] for f in facts if f.encounter_id == encounter
        ]
        tables["visit_occurrence"].append(
            {
                "visit_occurrence_id": visits[encounter],
                "person_id": persons[fact.subject_id],
                "visit_concept_id": 0,
                "visit_source_concept_id": 0,
                "visit_type_concept_id": 0,
                "visit_start_date": fact.effective_time["date"],
                "visit_end_date": max(related),
                "synthetic": True,
            }
        )
    concepts = {}
    for fact in facts:
        if fact.fact_type == "encounter":
            continue
        table, prefix, domain = _DOMAINS[fact.fact_type]
        code = fact.value["synthetic_code"]
        concept_key = (domain, code)
        if concept_key not in concepts:
            concept_id = _identity("concept", domain + ":" + code, seed)
            concepts[concept_key] = concept_id
            tables["concept"].append(
                {
                    "concept_id": concept_id,
                    "concept_name": code,
                    "domain_id": domain,
                    "vocabulary_id": "OPENMED_SYNTHETIC",
                    "concept_class_id": "Synthetic",
                    "standard_concept": "",
                    "concept_code": code,
                    "synthetic": True,
                }
            )
        note_id = _identity("note", fact.fact_id, seed)
        text = f"SYNTHETIC {fact.fact_type}: {code}"
        note_hash = sha256_digest(text).removeprefix("sha256:")
        tables["note"].append(
            {
                "note_id": note_id,
                "person_id": persons[fact.subject_id],
                "visit_occurrence_id": visits[fact.encounter_id],
                "note_date": fact.effective_time["date"],
                "note_text": text,
                "source_note_hash": note_hash,
                "synthetic": True,
            }
        )
        row = {
            prefix + "_id": _identity(table, fact.fact_id, seed),
            "person_id": persons[fact.subject_id],
            "visit_occurrence_id": visits[fact.encounter_id],
            prefix + "_concept_id": concepts[concept_key],
            prefix + "_source_concept_id": concepts[concept_key],
            prefix + "_source_value": code,
            "note_id": note_id,
            "source_note_hash": note_hash,
            "synthetic": True,
        }
        effective = fact.effective_time["date"]
        if table == "condition_occurrence":
            row.update(
                condition_occurrence_id=row.pop("condition_id"),
                condition_start_date=effective,
                condition_end_date=effective,
                condition_type_concept_id=0,
            )
        elif table == "drug_exposure":
            row.update(
                drug_exposure_id=row.pop("drug_id"),
                drug_exposure_start_date=effective,
                drug_exposure_end_date=effective,
                drug_type_concept_id=0,
            )
        elif table == "procedure_occurrence":
            row.update(
                procedure_occurrence_id=row.pop("procedure_id"),
                procedure_date=effective,
                procedure_type_concept_id=0,
            )
        else:
            row[prefix + "_date"] = effective
            row[prefix + "_type_concept_id"] = 0
            # Missing numeric truth stays missing, rather than fabricating measurements.
            row["value_as_number"] = None
        tables[table].append(row)
    for table, rows in tables.items():
        key = {
            "concept": "concept_id",
            "person": "person_id",
            "visit_occurrence": "visit_occurrence_id",
            "note": "note_id",
        }.get(table, table + "_id")
        rows.sort(key=lambda row: row[key])
    condition_sets = {person: set() for person in persons.values()}
    for row in tables["condition_occurrence"]:
        condition_sets[row["person_id"]].add(row["condition_source_value"])
    cohort = {
        "required_condition_codes": sorted(criteria),
        "member_person_ids": sorted(
            person for person, codes in condition_sets.items() if set(criteria) <= codes
        ),
    }
    result = {
        "schema_version": "1.0.0",
        "synthetic": True,
        "tables": tables,
        "cohort": cohort,
    }
    if validate_synthetic_omop(result):
        raise ValueError("synthetic OMOP generation failed fixture constraints")
    return result


def validate_synthetic_omop(export: Mapping[str, object]) -> tuple[str, ...]:
    """Independently check runtime relational rules, visit time and cohort membership."""
    if not isinstance(export, Mapping) or not isinstance(export.get("tables"), Mapping):
        raise ValueError("invalid synthetic OMOP export")
    tables = export["tables"]
    if any(
        not isinstance(rows, (tuple, list))
        or any(not isinstance(row, Mapping) for row in rows)
        for rows in tables.values()
    ):
        raise ValueError("invalid synthetic OMOP row collection")
    if sum(len(rows) for rows in tables.values()) > 10000:
        raise ValueError("synthetic OMOP table limit exceeded")
    errors = set()
    if export.get("synthetic") is not True or export.get("schema_version") != "1.0.0":
        errors.add("synthetic_provenance_missing")
    if any(
        row.get("synthetic") is not True for rows in tables.values() for row in rows
    ):
        errors.add("synthetic_row_marker_missing")
    try:
        relational_valid = validate_omop_cohort_export(tables).is_valid
    except (TypeError, KeyError, ValueError, AttributeError):
        relational_valid = False
    if not relational_valid:
        errors.add("relational_or_concept_constraint")
        return tuple(sorted(errors))
    cohort = export.get("cohort")
    if not isinstance(cohort, Mapping) or set(cohort) != {
        "required_condition_codes",
        "member_person_ids",
    }:
        return tuple(sorted(errors | {"invalid_cohort_structure"}))
    criteria, members = cohort["required_condition_codes"], cohort["member_person_ids"]
    if (
        not isinstance(criteria, list)
        or len(criteria) > 128
        or any(
            not isinstance(code, str) or not _CODE.fullmatch(code) for code in criteria
        )
        or len(set(criteria)) != len(criteria)
    ):
        return tuple(sorted(errors | {"invalid_cohort_structure"}))
    if (
        not isinstance(members, list)
        or len(members) > 10000
        or any(type(person) is not int or not 1 <= person < 2**63 for person in members)
        or len(set(members)) != len(members)
    ):
        return tuple(sorted(errors | {"invalid_cohort_structure"}))
    visits = {r["visit_occurrence_id"]: r for r in tables.get("visit_occurrence", [])}
    persons = {r["person_id"] for r in tables.get("person", [])}

    def valid_date(value):
        if not isinstance(value, str) or len(value) != 10:
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True

    birth_years = {
        r["person_id"]: r.get("year_of_birth") for r in tables.get("person", [])
    }
    for visit in visits.values():
        start, end = visit.get("visit_start_date"), visit.get("visit_end_date")
        if not valid_date(start) or not valid_date(end):
            errors.add("invalid_visit_date")
            continue
        birth = birth_years.get(visit.get("person_id"))
        if type(birth) is not int or not 1 <= birth <= int(start[:4]):
            errors.add("invalid_birth_visit_chronology")
        if start > end:
            errors.add("visit_time_reversed")
    dates = {
        "condition_occurrence": "condition_start_date",
        "drug_exposure": "drug_exposure_start_date",
        "measurement": "measurement_date",
        "procedure_occurrence": "procedure_date",
        "observation": "observation_date",
    }
    for table, column in dates.items():
        for row in tables.get(table, []):
            value = row.get(column)
            if not valid_date(value):
                errors.add("invalid_event_date")
                continue
            visit = visits.get(row.get("visit_occurrence_id"))
            if (
                visit is not None
                and valid_date(visit.get("visit_start_date"))
                and valid_date(visit.get("visit_end_date"))
                and not visit["visit_start_date"] <= value <= visit["visit_end_date"]
            ):
                errors.add("event_outside_visit")
            if table in {"condition_occurrence", "drug_exposure"}:
                end = (
                    "condition_end_date"
                    if table == "condition_occurrence"
                    else "drug_exposure_end_date"
                )
                if not valid_date(row.get(end)):
                    errors.add("invalid_event_date")
                elif value > row[end]:
                    errors.add("event_time_reversed")
    conditions = {person: set() for person in persons}
    for row in tables.get("condition_occurrence", []):
        if row.get("person_id") in conditions:
            conditions[row["person_id"]].add(row["condition_source_value"])
    cohort = export.get("cohort", {})
    if (
        not isinstance(cohort, Mapping)
        or not isinstance(cohort.get("required_condition_codes"), list)
        or not isinstance(cohort.get("member_person_ids"), list)
    ):
        errors.add("invalid_cohort_structure")
        cohort = {"required_condition_codes": [], "member_person_ids": []}
    expected = sorted(
        person
        for person, codes in conditions.items()
        if set(cohort.get("required_condition_codes", [])) <= codes
    )
    if expected != cohort.get("member_person_ids"):
        errors.add("cohort_truth_mismatch")
    for note in tables.get("note", []):
        if not isinstance(note.get("note_text"), str) or note.get(
            "source_note_hash"
        ) != sha256_digest(note["note_text"]).removeprefix("sha256:"):
            errors.add("synthetic_note_hash_mismatch")
    return tuple(sorted(errors))
