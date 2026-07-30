"""Tests for the referential-integrity-preserving linked-table surrogate map."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from openmed.core.surrogate_vault import SurrogateVault
from openmed.structured.relational import (
    ColumnRef,
    DanglingForeignKeyError,
    DateColumn,
    ForeignKey,
    KeySpace,
    RelationalSchema,
    RelationalSchemaError,
    _namespaced_source,
    _shift_date,
    deidentify_linked_tables,
)


def _surrogate(vault, key_space: str, value: str) -> str:
    """Look a surrogate back up the same way the transform keyed it."""

    return vault.get(_namespaced_source(key_space, value), label=key_space)


VAULT_SECRET = "unit-test-vault-secret-0123456789"
DATE_SECRET = b"unit-test-date-shift-secret-9876543210"


# ---------------------------------------------------------------------------
# Synthetic 3-table linked fixture (patient joins across all three tables).
# Values are generated algorithmically; no real identifiers are used.
# ---------------------------------------------------------------------------


def build_fixture(n_patients: int = 5):
    """Return demographics/encounters/labs linked by patient and encounter keys."""

    demographics: list[dict] = []
    encounters: list[dict] = []
    labs: list[dict] = []
    epoch = date(2020, 1, 1)
    for patient in range(n_patients):
        pid = f"P{patient:04d}"
        demographics.append(
            {
                "patient_id": pid,
                "sex": "MF"[patient % 2],
                "birth_date": date(1950 + patient, 3, 14).isoformat(),
            }
        )
        for enc in range(1 + patient % 2):  # 1 or 2 encounters per patient
            eid = f"E{patient:04d}{enc}"
            admission = epoch + timedelta(days=patient * 11 + enc * 37)
            length_of_stay = 2 + (patient + enc) % 6
            discharge = admission + timedelta(days=length_of_stay)
            encounters.append(
                {
                    "encounter_id": eid,
                    "patient_id": pid,
                    "admission_date": admission.isoformat(),
                    "discharge_date": discharge.isoformat(),
                }
            )
            for lab in range(2):
                labs.append(
                    {
                        "lab_id": f"L{patient:04d}{enc}{lab}",
                        "encounter_id": eid,
                        "patient_id": pid,
                        "collected_date": (admission + timedelta(days=lab)).isoformat(),
                        "analyte": ["HGB", "WBC"][lab],
                        "value": 10 + patient + lab,
                    }
                )
    return {"demographics": demographics, "encounters": encounters, "labs": labs}


def build_schema() -> RelationalSchema:
    """Return the schema wiring the three fixture tables together."""

    return RelationalSchema(
        key_spaces=(
            KeySpace(
                "PATIENT_ID",
                (
                    ColumnRef("demographics", "patient_id"),
                    ColumnRef("encounters", "patient_id"),
                    ColumnRef("labs", "patient_id"),
                ),
            ),
            KeySpace(
                "ENCOUNTER_ID",
                (
                    ColumnRef("encounters", "encounter_id"),
                    ColumnRef("labs", "encounter_id"),
                ),
            ),
            KeySpace("LAB_ID", (ColumnRef("labs", "lab_id"),)),
        ),
        subject_key_space="PATIENT_ID",
        foreign_keys=(
            ForeignKey("encounters", "patient_id", "demographics", "patient_id"),
            ForeignKey("labs", "patient_id", "demographics", "patient_id"),
            ForeignKey("labs", "encounter_id", "encounters", "encounter_id"),
        ),
        date_columns=(
            DateColumn("demographics", "birth_date", "patient_id"),
            DateColumn("encounters", "admission_date", "patient_id"),
            DateColumn("encounters", "discharge_date", "patient_id"),
            DateColumn("labs", "collected_date", "patient_id"),
        ),
    )


def _vault() -> SurrogateVault:
    return SurrogateVault.in_memory(VAULT_SECRET)


def _deidentify(tables=None, vault=None):
    tables = build_fixture() if tables is None else tables
    vault = _vault() if vault is None else vault
    result = deidentify_linked_tables(
        tables,
        build_schema(),
        vault=vault,
        date_shift_secret=DATE_SECRET,
    )
    return tables, vault, result


def _natural_join(left, right, left_key, right_key):
    index: dict = {}
    for row in right:
        index.setdefault(row[right_key], []).append(row)
    return [(l, r) for l in left for r in index.get(l[left_key], [])]


# ---------------------------------------------------------------------------
# Referential integrity — joins survive de-identification
# ---------------------------------------------------------------------------


def test_encounter_foreign_key_join_is_reproduced_on_surrogates():
    tables, vault, result = _deidentify()

    original = _natural_join(
        tables["labs"], tables["encounters"], "encounter_id", "encounter_id"
    )
    deidentified = _natural_join(
        result.tables["labs"],
        result.tables["encounters"],
        "encounter_id",
        "encounter_id",
    )

    # The join must pair the same rows, and each original pair must map onto a
    # de-identified pair through the vault surrogates.
    assert len(deidentified) == len(original) == len(tables["labs"])
    expected = sorted(
        (
            _surrogate(vault, "LAB_ID", lab["lab_id"]),
            _surrogate(vault, "ENCOUNTER_ID", enc["encounter_id"]),
        )
        for lab, enc in original
    )
    actual = sorted((lab["lab_id"], enc["encounter_id"]) for lab, enc in deidentified)
    assert actual == expected


def test_patient_foreign_key_join_is_reproduced_on_surrogates():
    tables, vault, result = _deidentify()

    original = _natural_join(
        tables["encounters"], tables["demographics"], "patient_id", "patient_id"
    )
    deidentified = _natural_join(
        result.tables["encounters"],
        result.tables["demographics"],
        "patient_id",
        "patient_id",
    )
    assert len(deidentified) == len(original) == len(tables["encounters"])
    expected = sorted(
        (
            _surrogate(vault, "ENCOUNTER_ID", enc["encounter_id"]),
            _surrogate(vault, "PATIENT_ID", dem["patient_id"]),
        )
        for enc, dem in original
    )
    actual = sorted(
        (enc["encounter_id"], dem["patient_id"]) for enc, dem in deidentified
    )
    assert actual == expected


def test_manifest_reports_zero_orphaned_keys():
    _, _, result = _deidentify()
    assert result.manifest.orphaned_foreign_keys == 0


# ---------------------------------------------------------------------------
# Surrogate consistency — same source key, same surrogate, sourced from vault
# ---------------------------------------------------------------------------


def test_same_source_key_yields_one_surrogate_across_all_tables():
    tables, vault, result = _deidentify()

    # Pick a patient that appears in all three tables.
    pid = "P0001"
    dem_surrogate = next(
        row["patient_id"]
        for original, row in zip(tables["demographics"], result.tables["demographics"])
        if original["patient_id"] == pid
    )
    enc_surrogates = {
        row["patient_id"]
        for original, row in zip(tables["encounters"], result.tables["encounters"])
        if original["patient_id"] == pid
    }
    lab_surrogates = {
        row["patient_id"]
        for original, row in zip(tables["labs"], result.tables["labs"])
        if original["patient_id"] == pid
    }
    assert enc_surrogates == {dem_surrogate}
    assert lab_surrogates == {dem_surrogate}


def test_surrogates_are_sourced_from_the_vault():
    tables, vault, result = _deidentify()
    for original, row in zip(tables["demographics"], result.tables["demographics"]):
        assert row["patient_id"] == _surrogate(
            vault, "PATIENT_ID", original["patient_id"]
        )
        assert row["patient_id"] != original["patient_id"]


def test_transformation_is_deterministic_across_runs():
    tables = build_fixture()
    _, _, first = _deidentify(tables=tables, vault=_vault())
    _, _, second = _deidentify(tables=tables, vault=_vault())
    assert first.tables == second.tables


# ---------------------------------------------------------------------------
# Date coordination — intervals preserved, absolute dates shifted, one offset
# ---------------------------------------------------------------------------


def test_length_of_stay_interval_is_preserved_and_dates_shift():
    tables, _, result = _deidentify()
    for original, row in zip(tables["encounters"], result.tables["encounters"]):
        orig_los = date.fromisoformat(original["discharge_date"]) - date.fromisoformat(
            original["admission_date"]
        )
        new_los = date.fromisoformat(row["discharge_date"]) - date.fromisoformat(
            row["admission_date"]
        )
        assert new_los == orig_los
        assert row["admission_date"] != original["admission_date"]
        assert row["discharge_date"] != original["discharge_date"]


def test_cross_table_interval_for_a_subject_is_preserved():
    tables, _, result = _deidentify()
    enc_by_id = {row["encounter_id"]: row for row in tables["encounters"]}
    new_enc_by_index = list(zip(tables["encounters"], result.tables["encounters"]))
    new_enc_lookup = {
        original["encounter_id"]: new for original, new in new_enc_by_index
    }
    for original_lab, new_lab in zip(tables["labs"], result.tables["labs"]):
        enc = enc_by_id[original_lab["encounter_id"]]
        new_enc = new_enc_lookup[original_lab["encounter_id"]]
        orig_gap = date.fromisoformat(
            original_lab["collected_date"]
        ) - date.fromisoformat(enc["admission_date"])
        new_gap = date.fromisoformat(new_lab["collected_date"]) - date.fromisoformat(
            new_enc["admission_date"]
        )
        assert new_gap == orig_gap


def test_one_offset_per_subject_applies_across_tables():
    tables, _, result = _deidentify()
    # A patient's demographics offset must equal that patient's encounter offset.
    pid = "P0002"
    dem_original = next(r for r in tables["demographics"] if r["patient_id"] == pid)
    dem_new = next(
        new
        for original, new in zip(tables["demographics"], result.tables["demographics"])
        if original["patient_id"] == pid
    )
    dem_offset = date.fromisoformat(dem_new["birth_date"]) - date.fromisoformat(
        dem_original["birth_date"]
    )
    for original, new in zip(tables["encounters"], result.tables["encounters"]):
        if original["patient_id"] != pid:
            continue
        enc_offset = date.fromisoformat(new["admission_date"]) - date.fromisoformat(
            original["admission_date"]
        )
        assert enc_offset == dem_offset
        assert dem_offset != timedelta(0)


def test_datetime_and_date_objects_are_shifted_in_place():
    tables = {
        "demographics": [{"patient_id": "P1", "birth_date": date(1980, 6, 1)}],
        "visits": [
            {
                "visit_id": "V1",
                "patient_id": "P1",
                "seen_at": datetime(2021, 5, 4, 9, 30),
            }
        ],
    }
    schema = RelationalSchema(
        key_spaces=(
            KeySpace(
                "PATIENT_ID",
                (
                    ColumnRef("demographics", "patient_id"),
                    ColumnRef("visits", "patient_id"),
                ),
            ),
            KeySpace("VISIT_ID", (ColumnRef("visits", "visit_id"),)),
        ),
        subject_key_space="PATIENT_ID",
        foreign_keys=(
            ForeignKey("visits", "patient_id", "demographics", "patient_id"),
        ),
        date_columns=(
            DateColumn("demographics", "birth_date", "patient_id"),
            DateColumn("visits", "seen_at", "patient_id"),
        ),
    )
    result = deidentify_linked_tables(
        tables, schema, vault=_vault(), date_shift_secret=DATE_SECRET
    )
    new_birth = result.tables["demographics"][0]["birth_date"]
    new_seen = result.tables["visits"][0]["seen_at"]
    assert isinstance(new_birth, date) and not isinstance(new_birth, datetime)
    assert isinstance(new_seen, datetime)
    assert new_seen.time() == datetime(2021, 5, 4, 9, 30).time()
    assert new_birth != date(1980, 6, 1)


# ---------------------------------------------------------------------------
# Manifest privacy — no raw identifiers, no raw-to-surrogate map
# ---------------------------------------------------------------------------


def test_manifest_never_contains_raw_identifiers():
    tables, _, result = _deidentify()
    serialized = json.dumps(result.manifest.to_dict())
    for row in tables["demographics"]:
        assert row["patient_id"] not in serialized
    for row in tables["encounters"]:
        assert row["encounter_id"] not in serialized
    for row in tables["labs"]:
        assert row["lab_id"] not in serialized


def test_manifest_entries_are_privacy_safe_proofs():
    _, _, result = _deidentify()
    manifest = result.manifest
    assert manifest.entries, "manifest should reference the pseudonymized keys"
    for source_proof, surrogate_proof in manifest.entries.items():
        assert source_proof.startswith("hmac-sha256:")
        assert surrogate_proof.startswith("hmac-sha256:")


def test_manifest_offsets_are_keyed_by_surrogate_not_raw():
    tables, _, result = _deidentify()
    raw_patient_ids = {row["patient_id"] for row in tables["demographics"]}
    assert set(result.manifest.subject_offsets).isdisjoint(raw_patient_ids)
    assert len(result.manifest.subject_offsets) == len(raw_patient_ids)


# ---------------------------------------------------------------------------
# Key-space injectivity — same raw value in distinct spaces must diverge
# ---------------------------------------------------------------------------


def _two_space_tables(order):
    """Return two tables that share the raw value ``X1`` under distinct keys."""

    people = [{"person_id": "X1", "note": "p"}]
    orders = [{"order_id": "X1", "person_id": "X1"}]
    tables = {"people": people, "orders": orders}
    return {name: tables[name] for name in order}


def _two_space_schema() -> RelationalSchema:
    return RelationalSchema(
        key_spaces=(
            KeySpace(
                "PERSON_ID",
                (
                    ColumnRef("people", "person_id"),
                    ColumnRef("orders", "person_id"),
                ),
            ),
            KeySpace("ORDER_ID", (ColumnRef("orders", "order_id"),)),
        ),
        subject_key_space="PERSON_ID",
        foreign_keys=(ForeignKey("orders", "person_id", "people", "person_id"),),
    )


def test_same_raw_value_in_distinct_key_spaces_yields_distinct_surrogates():
    result = deidentify_linked_tables(
        _two_space_tables(("people", "orders")),
        _two_space_schema(),
        vault=_vault(),
        date_shift_secret=DATE_SECRET,
    )
    person_surrogate = result.tables["people"][0]["person_id"]
    order_surrogate = result.tables["orders"][0]["order_id"]
    # The shared FK column still matches its parent (same key space).
    assert result.tables["orders"][0]["person_id"] == person_surrogate
    # But the identically-valued key in a different key space must diverge.
    assert order_surrogate != person_surrogate


def test_surrogates_are_independent_of_table_processing_order():
    forward = deidentify_linked_tables(
        _two_space_tables(("people", "orders")),
        _two_space_schema(),
        vault=_vault(),
        date_shift_secret=DATE_SECRET,
    )
    reverse = deidentify_linked_tables(
        _two_space_tables(("orders", "people")),
        _two_space_schema(),
        vault=_vault(),
        date_shift_secret=DATE_SECRET,
    )
    assert forward.tables["people"] == reverse.tables["people"]
    assert forward.tables["orders"] == reverse.tables["orders"]


# ---------------------------------------------------------------------------
# Robustness — out-of-range date shift is a descriptive error
# ---------------------------------------------------------------------------


def test_out_of_range_date_shift_raises_schema_error():
    column = DateColumn("demographics", "event_date", "patient_id")
    with pytest.raises(RelationalSchemaError, match="representable date range"):
        _shift_date(date(9999, 12, 31), 5, column)
    with pytest.raises(RelationalSchemaError, match="representable date range"):
        _shift_date(date(1, 1, 1), -5, column)


# ---------------------------------------------------------------------------
# Dangling foreign keys — refuse to emit
# ---------------------------------------------------------------------------


def test_dangling_foreign_key_is_refused():
    tables = build_fixture()
    tables["labs"].append(
        {
            "lab_id": "L99999",
            "encounter_id": "E-does-not-exist",
            "patient_id": "P0000",
            "collected_date": "2020-02-02",
            "analyte": "HGB",
            "value": 42,
        }
    )
    with pytest.warns(UserWarning, match="dangling"):
        with pytest.raises(DanglingForeignKeyError):
            deidentify_linked_tables(
                tables,
                build_schema(),
                vault=_vault(),
                date_shift_secret=DATE_SECRET,
            )


def test_null_foreign_keys_are_not_treated_as_dangling():
    tables = build_fixture()
    tables["labs"].append(
        {
            "lab_id": "L88888",
            "encounter_id": None,
            "patient_id": "P0000",
            "collected_date": "2020-02-02",
            "analyte": "WBC",
            "value": 7,
        }
    )
    result = deidentify_linked_tables(
        tables, build_schema(), vault=_vault(), date_shift_secret=DATE_SECRET
    )
    assert result.manifest.orphaned_foreign_keys == 0


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_foreign_key_spanning_distinct_key_spaces_is_rejected():
    with pytest.raises(RelationalSchemaError, match="distinct key spaces"):
        RelationalSchema(
            key_spaces=(
                KeySpace("PATIENT_ID", (ColumnRef("a", "patient_id"),)),
                KeySpace("OTHER_ID", (ColumnRef("b", "ref_id"),)),
            ),
            subject_key_space="PATIENT_ID",
            foreign_keys=(ForeignKey("b", "ref_id", "a", "patient_id"),),
        )


def test_column_in_two_key_spaces_is_rejected():
    with pytest.raises(RelationalSchemaError, match="enrolled in key spaces"):
        RelationalSchema(
            key_spaces=(
                KeySpace("A", (ColumnRef("t", "id"),)),
                KeySpace("B", (ColumnRef("t", "id"),)),
            ),
            subject_key_space="A",
        )


def test_missing_column_is_rejected():
    tables = {"demographics": [{"patient_id": "P1"}]}
    schema = RelationalSchema(
        key_spaces=(
            KeySpace("PATIENT_ID", (ColumnRef("demographics", "patient_id"),)),
        ),
        subject_key_space="PATIENT_ID",
        date_columns=(DateColumn("demographics", "birth_date", "patient_id"),),
    )
    with pytest.raises(RelationalSchemaError, match="missing column"):
        deidentify_linked_tables(
            tables, schema, vault=_vault(), date_shift_secret=DATE_SECRET
        )


def test_unknown_subject_key_space_is_rejected():
    with pytest.raises(RelationalSchemaError, match="subject_key_space"):
        RelationalSchema(
            key_spaces=(KeySpace("A", (ColumnRef("t", "id"),)),),
            subject_key_space="MISSING",
        )
