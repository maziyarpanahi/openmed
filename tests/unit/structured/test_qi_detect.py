"""Tests for automatic tabular quasi-identifier detection."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from openmed.risk import kanon_report, risk_report
from openmed.structured import (
    ROLE_DIRECT_ID,
    ROLE_FREE_TEXT,
    ROLE_INTERNAL_LINKAGE,
    ROLE_QUASI_ID,
    ROLE_SAFE,
    ROLE_SENSITIVE,
    scan_table,
    write_table,
)
from openmed.structured.qi_detect import _risk_key_bytes

QI_COLUMNS = {"age", "zip_code", "admission_date", "diagnosis"}
FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "structured"
    / "quasi_identifier_rare_singleton.csv"
)


def _read_golden_rows() -> list[dict[str, str]]:
    with FIXTURE_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


ROWS = _read_golden_rows()


def test_scan_table_surfaces_planted_singleton_qi_set(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "released_style.csv", ROWS)

    manifest = scan_table(csv_path)

    assert manifest["columns"]["record_id"]["role"] == ROLE_DIRECT_ID
    assert manifest["columns"]["age"]["role"] == ROLE_QUASI_ID
    assert manifest["columns"]["zip_code"]["role"] == ROLE_QUASI_ID
    assert manifest["columns"]["admission_date"]["role"] == ROLE_QUASI_ID
    assert manifest["quasi_identifier_sets"]
    assert any(
        QI_COLUMNS <= set(qi_set["columns"])
        and qi_set["singleton_count"] >= 1
        and qi_set["min_equivalence_class_size"] == 1
        for qi_set in manifest["quasi_identifier_sets"]
    )


def test_scan_table_strips_utf8_byte_order_mark(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "bom.csv", ROWS)
    csv_path.write_text(
        "\ufeff" + csv_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manifest = scan_table(csv_path)

    assert "record_id" in manifest["columns"]
    assert "\ufeffrecord_id" not in manifest["columns"]
    assert "\ufeff" not in json.dumps(manifest, ensure_ascii=False)


def test_detected_qi_class_sizes_match_explicit_kanon_report(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "keys.csv", ROWS)
    manifest = scan_table(csv_path)
    qi_set = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if QI_COLUMNS <= set(item["columns"])
    )
    report = kanon_report(
        ROWS,
        quasi_identifiers=qi_set["columns"],
    )

    assert qi_set["equivalence_class_count"] == report["class_count"]
    assert qi_set["min_equivalence_class_size"] == report["k"]
    assert qi_set["singleton_count"] == sum(
        item["size"] for item in report["equivalence_classes"] if item["size"] == 1
    )


def test_detected_qi_keys_are_byte_identical_to_risk_report(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "risk-keys.csv", ROWS)
    manifest = scan_table(csv_path)
    qi_set = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if QI_COLUMNS <= set(item["columns"])
    )
    report = risk_report(
        ROWS,
        quasi_identifier_fields=qi_set["columns"],
    )
    target = next(row for row in ROWS if row["record_id"] == "target")
    singleton = next(
        item for item in report["singleton_records"] if item["record_id"] == "target"
    )

    assert _risk_key_bytes(target, qi_set["columns"]) == _canonical_json_bytes(
        singleton["quasi_identifier_key"]
    )


def test_released_style_fixture_recall_and_phi_safe_manifest(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "manifest.csv", ROWS)
    manifest = scan_table(csv_path)
    best_recall = max(
        len(QI_COLUMNS & set(item["columns"])) / len(QI_COLUMNS)
        for item in manifest["quasi_identifier_sets"]
    )

    assert best_recall >= 0.9

    serialized = json.dumps(manifest, sort_keys=True)
    for raw_value in (
        "target",
        "02139",
        "30301",
        "2025-04-09",
        "rare-alpha-syndrome",
        "routine-follow-up",
    ):
        assert raw_value not in serialized
    assert "path" not in manifest
    assert "key_fingerprints" not in serialized
    assert "singleton_key_fingerprints" not in serialized


def test_manifest_omits_source_name_and_low_entropy_hashes(tmp_path: Path) -> None:
    source_canary = "patient-alice-1970-01-01"
    csv_path = _write_csv(tmp_path / f"{source_canary}.csv", ROWS)
    manifest = scan_table(csv_path)
    raw_key = _canonical_json_bytes(
        [[field, ROWS[0][field]] for field in sorted(QI_COLUMNS)]
    )
    unsalted_digest = hashlib.sha256(raw_key).hexdigest()

    serialized = json.dumps(manifest, sort_keys=True)

    assert source_canary not in serialized
    assert unsalted_digest not in serialized
    assert ROWS[0]["record_id"] not in serialized


def test_rare_diagnosis_has_overlapping_sensitive_and_qi_roles(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "dual-role.csv", ROWS)

    manifest = scan_table(csv_path)
    diagnosis = manifest["columns"]["diagnosis"]

    assert diagnosis["role"] == ROLE_QUASI_ID
    assert diagnosis["roles"] == [ROLE_QUASI_ID, ROLE_SENSITIVE]
    assert diagnosis["canonical_label"] == "CONDITION"
    assert manifest["column_role_sets"]["diagnosis"] == [
        ROLE_QUASI_ID,
        ROLE_SENSITIVE,
    ]


def test_free_text_is_sensitive_without_becoming_a_qi(tmp_path: Path) -> None:
    rows = [
        {"patient_id": "a", "clinical_note": "canary one"},
        {"patient_id": "b", "clinical_note": "canary two"},
    ]
    csv_path = _write_csv(tmp_path / "notes.csv", rows)

    manifest = scan_table(csv_path)

    assert manifest["columns"]["clinical_note"]["roles"] == [
        ROLE_SENSITIVE,
        ROLE_FREE_TEXT,
    ]
    assert "clinical_note" not in {
        column
        for candidate in manifest["quasi_identifier_sets"]
        for column in candidate["columns"]
    }


def test_privacy_unit_uses_subjects_not_repeated_encounters(
    tmp_path: Path,
) -> None:
    rows = [
        {"patient_id": "p-1", "age": "70", "zip_code": "02139"},
        {"patient_id": "p-1", "age": "70", "zip_code": "02139"},
        {"patient_id": "p-2", "age": "70", "zip_code": "02139"},
    ]
    csv_path = _write_csv(tmp_path / "encounters.csv", rows)

    manifest = scan_table(
        csv_path,
        full_scan=True,
        privacy_unit="patient_id",
        quasi_identifier_columns=("age", "zip_code"),
    )
    pair = next(
        candidate
        for candidate in manifest["quasi_identifier_sets"]
        if set(candidate["columns"]) == {"age", "zip_code"}
    )

    assert manifest["analysis_unit"] == {
        "kind": "subject",
        "method": "longitudinal-subject-profiles",
        "column": "patient_id",
        "record_count": 3,
        "unit_count": 2,
        "repeated_unit_count": 1,
        "records_in_repeated_units": 2,
        "missing_unit_count": 0,
        "max_records_per_unit": 2,
    }
    assert manifest["columns"]["patient_id"]["roles"] == [
        ROLE_DIRECT_ID,
        ROLE_INTERNAL_LINKAGE,
    ]
    assert pair["sampled_rows"] == 3
    assert pair["analysis_unit_count"] == 2
    assert pair["equivalence_class_count"] == 2
    assert pair["singleton_count"] == 2
    assert pair["min_equivalence_class_size"] == 1


def test_broad_candidate_scan_finds_combination_only_qis(tmp_path: Path) -> None:
    rows = [
        {
            "factor_a": f"a-{index // 10}",
            "factor_b": f"b-{index % 10}",
        }
        for index in range(100)
    ]
    csv_path = _write_csv(tmp_path / "factor-design.csv", rows)

    default = scan_table(csv_path, full_scan=True, max_set_size=2)
    broad = scan_table(
        csv_path,
        full_scan=True,
        max_set_size=2,
        max_candidate_columns=2,
        include_safe_candidates=True,
    )

    assert default["search"]["eligible_column_count"] == 0
    assert broad["search"]["candidate_scope"] == "all_reviewed_scalar_columns"
    pair = next(
        candidate
        for candidate in broad["quasi_identifier_sets"]
        if candidate["columns"] == ["factor_a", "factor_b"]
    )
    assert pair["analysis_unit_count"] == 100
    assert pair["singleton_count"] == 100
    assert pair["min_equivalence_class_size"] == 1


def test_explicit_overrides_win_over_heuristics(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "overrides.csv", ROWS)

    manifest = scan_table(
        csv_path,
        role_overrides={"age": ROLE_SAFE},
        quasi_identifier_columns=("age",),
        sensitive_columns=("department",),
        privacy_unit="record_id",
    )

    assert manifest["columns"]["age"]["roles"] == [ROLE_SAFE]
    assert manifest["columns"]["department"]["roles"] == [ROLE_SENSITIVE]
    assert ROLE_INTERNAL_LINKAGE in manifest["columns"]["record_id"]["roles"]
    assert all(
        "age" not in candidate["columns"]
        for candidate in manifest["quasi_identifier_sets"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role_overrides": {"not_a_column": ROLE_SAFE}},
        {"quasi_identifier_columns": ("not_a_column",)},
        {"sensitive_columns": ("not_a_column",)},
        {"privacy_unit": "not_a_column"},
    ],
)
def test_unknown_override_columns_are_rejected(
    tmp_path: Path,
    kwargs,
) -> None:
    csv_path = _write_csv(tmp_path / "unknown.csv", ROWS)

    with pytest.raises(ValueError, match="Unknown override columns: not_a_column"):
        scan_table(csv_path, **kwargs)


def test_unrecognized_explicit_qi_field_is_measured(tmp_path: Path) -> None:
    rows = [
        {"custom_release_attribute": "group-a"},
        {"custom_release_attribute": "group-a"},
        {"custom_release_attribute": "group-b"},
    ]
    csv_path = _write_csv(tmp_path / "custom.csv", rows)

    manifest = scan_table(
        csv_path,
        full_scan=True,
        quasi_identifier_columns=("custom_release_attribute",),
    )
    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["custom_release_attribute"]
    )

    assert candidate["equivalence_class_count"] == 2
    assert candidate["min_equivalence_class_size"] == 1
    assert candidate["singleton_count"] == 1


def test_explicit_qi_distinguishes_missing_null_and_empty(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "typed.jsonl"
    rows = [
        {"row": 1},
        {"row": 2, "custom_qi": None},
        {"row": 3, "custom_qi": ""},
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    manifest = scan_table(
        jsonl_path,
        full_scan=True,
        quasi_identifier_columns=("custom_qi",),
    )
    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["custom_qi"]
    )

    assert candidate["analysis_unit_count"] == 3
    assert candidate["equivalence_class_count"] == 3
    assert candidate["singleton_count"] == 3


def test_explicit_qi_preserves_published_unicode_representation(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "unicode-equivalence.csv",
        [
            {"custom_qi": "café"},
            {"custom_qi": "cafe\u0301"},
        ],
    )

    manifest = scan_table(
        path,
        full_scan=True,
        quasi_identifier_columns=("custom_qi",),
        max_set_size=1,
    )
    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["custom_qi"]
    )

    assert manifest["columns"]["custom_qi"]["profile"]["cardinality"] == 2
    assert candidate["equivalence_class_count"] == 2
    assert candidate["min_equivalence_class_size"] == 1


def test_explicit_qi_discovery_preserves_published_case(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "case-sensitive-qis.csv",
        [{"city": "Paris"}, {"city": "paris"}],
    )

    manifest = scan_table(
        path,
        full_scan=True,
        quasi_identifier_columns=("city",),
        max_set_size=1,
    )
    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["city"]
    )

    assert manifest["columns"]["city"]["profile"]["cardinality"] == 2
    assert candidate["equivalence_class_count"] == 2
    assert candidate["min_equivalence_class_size"] == 1


def test_full_subject_scan_rejects_missing_privacy_units(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "missing-subject.jsonl"
    rows = [
        {"patient_id": "p-1", "age": 70},
        {"patient_id": None, "age": 70},
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    advisory = scan_table(
        jsonl_path,
        privacy_unit="patient_id",
        quasi_identifier_columns=("age",),
    )

    assert advisory["analysis_unit"]["missing_unit_count"] == 1
    assert advisory["discovery"]["advisory"] is True

    with pytest.raises(ValueError, match="complete subject-level measurement"):
        scan_table(
            jsonl_path,
            full_scan=True,
            privacy_unit="patient_id",
            quasi_identifier_columns=("age",),
        )


def test_full_subject_scan_rejects_privacy_unit_whitespace_variants(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "whitespace-subject.csv",
        [
            {"patient_id": " patient-a ", "age": 30},
            {"patient_id": "patient-a", "age": 40},
        ],
    )

    advisory = scan_table(
        path,
        privacy_unit="patient_id",
        quasi_identifier_columns=("age",),
        max_set_size=1,
    )

    assert advisory["analysis_unit"]["missing_unit_count"] == 1
    assert advisory["analysis_unit"]["unit_count"] == 2
    with pytest.raises(ValueError, match="complete subject-level measurement"):
        scan_table(
            path,
            full_scan=True,
            privacy_unit="patient_id",
            quasi_identifier_columns=("age",),
            max_set_size=1,
        )


def test_subject_scan_preserves_exact_unicode_privacy_unit_ids(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "unicode-subject.csv",
        [
            {"patient_id": "café", "age": 30},
            {"patient_id": "cafe\u0301", "age": 30},
        ],
    )

    manifest = scan_table(
        path,
        full_scan=True,
        privacy_unit="patient_id",
        quasi_identifier_columns=("age",),
        max_set_size=1,
    )

    assert manifest["analysis_unit"]["unit_count"] == 2
    assert manifest["analysis_unit"]["records_in_repeated_units"] == 0


def test_subject_scan_unicode_aliases_cannot_create_false_longitudinal_k(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "unicode-longitudinal-subject.csv",
        [
            {"patient_id": "é", "event": "A"},
            {"patient_id": "e\u0301", "event": "B"},
            {"patient_id": "third", "event": "A"},
            {"patient_id": "third", "event": "B"},
        ],
    )

    manifest = scan_table(
        path,
        full_scan=True,
        privacy_unit="patient_id",
        quasi_identifier_columns=("event",),
        max_set_size=1,
    )
    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["event"]
    )

    assert manifest["analysis_unit"]["unit_count"] == 3
    assert candidate["analysis_unit_count"] == 3
    assert candidate["equivalence_class_count"] == 3
    assert candidate["min_equivalence_class_size"] == 1


def test_no_qi_is_review_required_not_evidence_of_safety(tmp_path: Path) -> None:
    rows = [
        {"department": "cardiology", "status": "active"},
        {"department": "cardiology", "status": "active"},
    ]
    csv_path = _write_csv(tmp_path / "no-qi.csv", rows)

    manifest = scan_table(
        csv_path,
        full_scan=True,
        role_overrides={"department": ROLE_SAFE, "status": ROLE_SAFE},
    )

    assert manifest["quasi_identifier_sets"] == []
    assert manifest["confidence"] == 0.0
    assert manifest["discovery"] == {
        "status": "insufficient-discovery",
        "advisory": False,
        "review_required": True,
        "final_measurement_ready": False,
        "reasons": ["no_candidate_qi_set_detected"],
        "no_candidate_is_not_evidence_of_safety": True,
    }


def test_sample_is_advisory_and_full_scan_is_complete(tmp_path: Path) -> None:
    rows = [
        {"age": str(60 + index), "zip_code": f"021{index:02d}"} for index in range(5)
    ]
    csv_path = _write_csv(tmp_path / "scan-modes.csv", rows)

    sampled = scan_table(
        csv_path,
        max_rows=2,
        quasi_identifier_columns=("age", "zip_code"),
    )
    complete = scan_table(
        csv_path,
        full_scan=True,
        max_rows=2,
        quasi_identifier_columns=("age", "zip_code"),
    )

    assert sampled["sample"] == {
        "sampled_rows": 2,
        "max_rows": 2,
        "source_rows": None,
        "bounded": True,
        "complete": False,
        "mode": "sample",
        "advisory": True,
    }
    assert sampled["discovery"]["status"] == "advisory-candidates"
    assert sampled["discovery"]["final_measurement_ready"] is False
    assert complete["sample"] == {
        "sampled_rows": 5,
        "max_rows": None,
        "source_rows": 5,
        "bounded": False,
        "complete": True,
        "mode": "full-scan",
        "advisory": False,
    }
    assert complete["discovery"]["status"] == "candidates-found"
    assert complete["discovery"]["final_measurement_ready"] is True


def test_search_budget_reports_incomplete_discovery(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "budget.csv", ROWS)

    manifest = scan_table(
        csv_path,
        search_budget=2,
        max_set_size=4,
    )

    assert manifest["search"]["combinations_evaluated"] == 2
    assert manifest["search"]["combinations_possible"] > 2
    assert manifest["search"]["budget_exhausted"] is True
    assert manifest["search"]["complete"] is False
    assert manifest["discovery"]["advisory"] is True
    assert "candidate_search_incomplete" in manifest["discovery"]["reasons"]


def test_max_set_size_truncation_never_claims_complete_discovery(
    tmp_path: Path,
) -> None:
    rows = [
        {"qi_a": "a1", "qi_b": "b1", "qi_c": "c1", "qi_d": "d1"},
        {"qi_a": "a2", "qi_b": "b2", "qi_c": "c2", "qi_d": "d2"},
    ]
    path = _write_csv(tmp_path / "bounded-sets.csv", rows)

    manifest = scan_table(
        path,
        full_scan=True,
        quasi_identifier_columns=("qi_a", "qi_b", "qi_c", "qi_d"),
        max_set_size=2,
        max_candidate_columns=4,
        search_budget=100,
    )

    assert manifest["search"]["candidate_column_count"] == 4
    assert manifest["search"]["effective_max_set_size"] == 2
    assert manifest["search"]["set_size_truncated"] is True
    assert manifest["search"]["combinations_possible"] == 10
    assert manifest["search"]["combinations_possible_all_set_sizes"] == 15
    assert manifest["search"]["complete"] is False
    assert manifest["discovery"]["final_measurement_ready"] is False
    assert "candidate_set_size_truncated" in manifest["discovery"]["reasons"]


def test_jsonl_scan_uses_same_manifest_contract(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in ROWS) + "\n",
        encoding="utf-8",
    )

    manifest = scan_table(jsonl_path)

    assert manifest["format"] == "jsonl"
    assert manifest["sample"]["sampled_rows"] == len(ROWS)
    assert any(
        QI_COLUMNS <= set(item["columns"]) for item in manifest["quasi_identifier_sets"]
    )


@pytest.mark.parametrize(
    ("suffix", "delimiter", "payload", "message"),
    [
        (".csv", ",", "age,age\n30,40\n", "duplicate column names"),
        (".tsv", "\t", "age\t \n30\t40\n", "empty column name"),
        (
            ".csv",
            ",",
            "age,disease\n30,example,sensitive-overflow-canary\n",
            "more cells",
        ),
        (".tsv", "\t", "age\tdisease\n30\n", "fewer cells"),
    ],
)
def test_discovery_rejects_malformed_delimited_shapes_without_values(
    tmp_path: Path,
    suffix: str,
    delimiter: str,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / f"malformed{suffix}"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as raised:
        scan_table(path, full_scan=True)

    assert "sensitive-overflow-canary" not in str(raised.value)
    assert delimiter not in str(raised.value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"age":30,"age":40}\n', "duplicate object keys"),
        ('{"profile":{"age":30,"age":40}}\n', "duplicate object keys"),
        ('{"age":NaN}\n', "row 1 is invalid"),
        ('{"age":1e9999}\n', "row 1 is invalid"),
        ('{"age":["sensitive-nested-canary"]}\n', "unsupported scalar"),
        ('{"age":{"value":"sensitive-nested-canary"}}\n', "unsupported scalar"),
    ],
)
def test_discovery_rejects_ambiguous_or_nested_json_without_values(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message) as raised:
        scan_table(path, full_scan=True)

    assert "sensitive-nested-canary" not in str(raised.value)


def test_discovery_profiles_typed_values_without_text_collapse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "typed.jsonl"
    path.write_text(
        '{"custom_qi":1}\n{"custom_qi":"1"}\n',
        encoding="utf-8",
    )

    manifest = scan_table(
        path,
        full_scan=True,
        quasi_identifier_columns=("custom_qi",),
    )

    assert manifest["columns"]["custom_qi"]["profile"]["cardinality"] == 2
    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["custom_qi"]
    )
    assert candidate["equivalence_class_count"] == 2


def test_discovery_preserves_binary_and_clinical_scalar_qis(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "typed.parquet"
    rows = [
        {
            "payload": b"\x00",
            "service_date": date(2025, 1, 1),
            "recorded_at": datetime(2025, 1, 1, 8, tzinfo=timezone.utc),
            "collection_time": time(8),
            "measurement": Decimal("1.20"),
        },
        {
            "payload": b"\x01",
            "service_date": date(2025, 1, 2),
            "recorded_at": datetime(2025, 1, 2, 8, tzinfo=timezone.utc),
            "collection_time": time(9),
            "measurement": Decimal("2.30"),
        },
    ]
    write_table(path, rows)

    manifest = scan_table(
        path,
        full_scan=True,
        quasi_identifier_columns=tuple(rows[0]),
    )

    for field in rows[0]:
        assert manifest["columns"][field]["profile"]["non_null_count"] == 2
        assert manifest["columns"][field]["profile"]["cardinality"] == 2
    payload_candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["payload"]
    )
    assert payload_candidate["equivalence_class_count"] == 2


def test_discovery_preserves_high_precision_decimal_qis(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "high-precision-decimal.parquet"
    write_table(
        path,
        [
            {"amount": Decimal("1.0000000000000000000000000000000000001")},
            {"amount": Decimal("1.0000000000000000000000000000000000002")},
        ],
    )

    manifest = scan_table(
        path,
        full_scan=True,
        quasi_identifier_columns=("amount",),
    )

    candidate = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if item["columns"] == ["amount"]
    )
    assert manifest["columns"]["amount"]["profile"]["cardinality"] == 2
    assert candidate["equivalence_class_count"] == 2
    assert candidate["min_equivalence_class_size"] == 1


@pytest.mark.parametrize("temporal_kind", ["timestamp_ns", "time64_ns"])
def test_full_scan_rejects_submicrosecond_parquet_temporal_precision(
    temporal_kind: str,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data_type = (
        pa.timestamp("ns") if temporal_kind == "timestamp_ns" else pa.time64("ns")
    )
    path = tmp_path / f"{temporal_kind}.parquet"
    pq.write_table(
        pa.table({"event_time": pa.array([1, 2], type=data_type)}),
        path,
    )

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        scan_table(
            path,
            full_scan=True,
            quasi_identifier_columns=("event_time",),
        )


def test_discovery_rejects_duplicate_parquet_schema_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeArrowException(Exception):
        pass

    class FakeSchema:
        names = ["age", "age"]

    class FakeMetadata:
        num_rows = 0

    class FakeParquetFile:
        schema_arrow = FakeSchema()
        metadata = FakeMetadata()

        def __init__(self, _path):
            pass

        def iter_batches(self, *, batch_size):
            del batch_size
            return iter(())

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.ArrowException = FakeArrowException
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.ParquetFile = FakeParquetFile
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    with pytest.raises(ValueError, match="duplicate column names"):
        scan_table(tmp_path / "duplicate.parquet", full_scan=True)


def test_tsv_scan_remains_bounded(tmp_path: Path) -> None:
    tsv_path = tmp_path / "records.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(ROWS[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(ROWS)

    manifest = scan_table(tsv_path, max_rows=3)

    assert manifest["format"] == "tsv"
    assert manifest["sample"]["sampled_rows"] == 3
    assert manifest["sample"]["bounded"] is True
    assert manifest["sample"]["advisory"] is True


def test_ndjson_scan_remains_bounded(tmp_path: Path) -> None:
    ndjson_path = tmp_path / "records.ndjson"
    ndjson_path.write_text(
        "\n".join(json.dumps(row) for row in ROWS) + "\n",
        encoding="utf-8",
    )

    manifest = scan_table(ndjson_path, max_rows=3)

    assert manifest["format"] == "jsonl"
    assert manifest["sample"]["sampled_rows"] == 3
    assert manifest["sample"]["bounded"] is True
    assert manifest["sample"]["advisory"] is True


def test_parquet_sampling_stops_at_fixed_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[int] = []

    class FakeSchema:
        names = ["age", "zip_code", "admission_date", "diagnosis"]

    class FakeMetadata:
        num_rows = 5_000_000

    class FakeBatch:
        def __init__(self, rows):
            self._rows = rows

        def to_pylist(self):
            return list(self._rows)

    class FakeParquetFile:
        schema_arrow = FakeSchema()
        metadata = FakeMetadata()

        def __init__(self, path):
            self.path = path

        def iter_batches(self, *, batch_size):
            calls.append(batch_size)
            yield FakeBatch(ROWS[:2])
            calls.append(batch_size)
            yield FakeBatch(ROWS[2:4])
            raise AssertionError("scan_table read beyond the sampling budget")

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.ParquetFile = FakeParquetFile
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    manifest = scan_table(tmp_path / "huge.parquet", max_rows=3, batch_size=2)

    assert calls == [2, 2]
    assert manifest["format"] == "parquet"
    assert manifest["sample"] == {
        "sampled_rows": 3,
        "max_rows": 3,
        "source_rows": 5_000_000,
        "bounded": True,
        "complete": False,
        "mode": "sample",
        "advisory": True,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
