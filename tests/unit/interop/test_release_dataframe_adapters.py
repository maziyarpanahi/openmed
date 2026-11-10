"""Pandas and Polars parity tests for the safe release workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from openmed.interop import get_adapter
from openmed.risk import (
    AnonymityPolicy,
    AnonymizationResult,
    ReleaseAssessment,
    assess_release,
)

_PATIENT_CANARY = "patient-meridian-canary"
_DIAGNOSIS_CANARY = "diagnosis-meridian-canary"
_LOCATION_CANARY = "90876"


def _patient_rows() -> list[dict[str, object]]:
    return [
        {
            "patient_id": _PATIENT_CANARY + "-a",
            "patient_name": "Synthetic Alpha",
            "age": 30,
            "zip": _LOCATION_CANARY,
            "disease": _DIAGNOSIS_CANARY + "-a",
        },
        {
            "patient_id": _PATIENT_CANARY + "-a",
            "patient_name": "Synthetic Alpha",
            "age": 30,
            "zip": _LOCATION_CANARY,
            "disease": _DIAGNOSIS_CANARY + "-b",
        },
        {
            "patient_id": _PATIENT_CANARY + "-b",
            "patient_name": "Synthetic Beta",
            "age": 30,
            "zip": _LOCATION_CANARY,
            "disease": _DIAGNOSIS_CANARY + "-a",
        },
        {
            "patient_id": _PATIENT_CANARY + "-b",
            "patient_name": "Synthetic Beta",
            "age": 30,
            "zip": _LOCATION_CANARY,
            "disease": _DIAGNOSIS_CANARY + "-b",
        },
    ]


def _policy() -> AnonymityPolicy:
    return AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("disease",),
        direct_identifiers=("patient_name",),
        privacy_unit="patient_id",
        target_k=2,
        target_l=1,
        target_t=1.0,
    )


def test_assessment_parity_uses_patient_units_and_safe_json() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("pandas")
    polars_adapter = get_adapter("polars")
    rows = _patient_rows()

    pandas_report = pd.DataFrame(rows).openmed.assess_release(_policy())
    polars_report = polars_adapter.assess_release(pl.DataFrame(rows), _policy())

    assert isinstance(pandas_report, ReleaseAssessment)
    assert isinstance(polars_report, ReleaseAssessment)
    assert pandas_report.to_dict() == polars_report.to_dict()
    assert pandas_report.row_count == 4
    assert pandas_report.privacy_unit_count == 2
    assert pandas_report.achieved_k == 2
    assert pandas_report.meets_policy is True
    safe_json = pandas_report.to_json()
    for canary in (
        _PATIENT_CANARY,
        _DIAGNOSIS_CANARY,
        _LOCATION_CANARY,
        "Synthetic Alpha",
    ):
        assert canary not in safe_json
    assert "equivalence_classes" not in safe_json
    assert "members" not in safe_json


def test_anonymization_parity_keeps_records_out_of_safe_evidence() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("pandas")
    polars_adapter = get_adapter("polars")
    rows = _patient_rows()

    pandas_result = pd.DataFrame(rows).openmed.anonymize_release(_policy())
    polars_result = polars_adapter.anonymize_release(
        pl.DataFrame(rows),
        _policy(),
    )

    assert isinstance(pandas_result, AnonymizationResult)
    assert isinstance(polars_result, AnonymizationResult)
    assert pandas_result.to_safe_dict() == polars_result.to_safe_dict()
    assert pandas_result.records == polars_result.records
    assert len(pandas_result.records) == 4
    assert all("patient_id" not in row for row in pandas_result.records)
    assert all("patient_name" not in row for row in pandas_result.records)
    assert any(
        _DIAGNOSIS_CANARY in str(row["disease"]) for row in pandas_result.records
    )
    safe_json = pandas_result.to_safe_json()
    assert '"records"' not in safe_json
    for canary in (
        _PATIENT_CANARY,
        _DIAGNOSIS_CANARY,
        _LOCATION_CANARY,
        "Synthetic Alpha",
    ):
        assert canary not in safe_json


def test_pandas_release_adapter_normalizes_timestamps_and_missing_values() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    rows = _patient_rows()
    frame = pd.DataFrame(rows)
    frame["visit_date"] = pd.to_datetime(
        [
            "2025-01-01T08:00:00Z",
            "2025-02-01T08:00:00Z",
            "2025-01-01T08:00:00Z",
            "2025-02-01T08:00:00Z",
        ],
        utc=True,
    )
    frame["recorded_at"] = [
        pd.Timestamp("2025-03-01T08:00:00Z"),
        pd.NaT,
        pd.Timestamp("2025-03-02T08:00:00Z"),
        pd.Timestamp("2025-03-03T08:00:00Z"),
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip", "visit_date"),
        sensitive_attributes=("disease",),
        direct_identifiers=("patient_name",),
        non_sensitive_attributes=("recorded_at",),
        privacy_unit="patient_id",
        target_k=2,
        target_l=1,
        target_t=1.0,
    )

    assessment = frame.openmed.assess_release(policy)
    result = frame.openmed.anonymize_release(policy)

    assert assessment.achieved_k == 2
    assert assessment.meets_policy is True
    assert any(row["recorded_at"] is None for row in result.records)
    timestamps = [
        row["recorded_at"] for row in result.records if row["recorded_at"] is not None
    ]
    assert timestamps
    assert all(type(value) is datetime for value in timestamps)
    assert all(value.tzinfo is timezone.utc for value in timestamps)


def test_pandas_release_scalar_normalizes_numpy_values_without_text_coercion() -> None:
    np = pytest.importorskip("numpy", exc_type=ImportError)
    pytest.importorskip("pandas", exc_type=ImportError)
    from openmed.interop.pandas_accessor import _release_scalar

    integer = _release_scalar(np.int64(7))
    floating = _release_scalar(np.float32(1.5))
    timestamp = _release_scalar(np.datetime64("2025-01-01T08:30:00.123456"))
    missing_timestamp = _release_scalar(np.datetime64("NaT"))

    assert type(integer) is int
    assert type(floating) is float
    assert floating == pytest.approx(1.5)
    assert type(timestamp) is datetime
    assert timestamp == datetime(2025, 1, 1, 8, 30, 0, 123456)
    assert missing_timestamp is None


def test_pandas_release_rejects_submicrosecond_timestamp_precision() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame(_patient_rows())
    frame["recorded_at"] = [
        pd.Timestamp("2025-01-01T00:00:00.000000001Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("disease",),
        direct_identifiers=("patient_name",),
        non_sensitive_attributes=("recorded_at",),
        privacy_unit="patient_id",
        target_k=2,
        target_l=1,
        target_t=1.0,
    )

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        frame.openmed.assess_release(policy)


@pytest.mark.parametrize("operation", ["assess_release", "anonymize_release"])
def test_pandas_release_rejects_duplicate_columns_before_record_conversion(
    operation: str,
) -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame(
        [[30, 40, "flu"], [31, 40, "cold"]],
        columns=["age", "age", "disease"],
    )
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        sensitive_attributes=("disease",),
        target_k=2,
    )

    with pytest.raises(ValueError, match="column names must be unique"):
        getattr(frame.openmed, operation)(policy)


def test_polars_namespace_matches_top_level_release_helpers() -> None:
    pl = pytest.importorskip("polars", exc_type=ImportError)
    adapter = get_adapter("polars")
    frame = pl.DataFrame(_patient_rows())
    if not hasattr(frame, "openmed"):
        pytest.skip("installed polars does not support DataFrame namespaces")

    namespace_report = frame.openmed.assess_release(_policy())
    helper_report = adapter.assess_release(frame, _policy())
    namespace_result = frame.openmed.anonymize_release(_policy())
    helper_result = adapter.anonymize_release(frame, _policy())

    assert namespace_report.to_dict() == helper_report.to_dict()
    assert namespace_result.to_safe_dict() == helper_result.to_safe_dict()
    assert namespace_result.records == helper_result.records


@pytest.mark.parametrize("temporal_kind", ["datetime_ns", "time_ns"])
@pytest.mark.parametrize("surface", ["direct", "namespace"])
def test_polars_release_rejects_submicrosecond_temporal_precision(
    temporal_kind: str,
    surface: str,
) -> None:
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("polars")
    dtype = pl.Datetime("ns", "UTC") if temporal_kind == "datetime_ns" else pl.Time
    frame = pl.DataFrame({"event_time": pl.Series("event_time", [1, 2], dtype=dtype)})
    policy = AnonymityPolicy(
        quasi_identifiers=("event_time",),
        target_k=2,
    )

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        if surface == "direct":
            assess_release(frame, policy)
        else:
            frame.openmed.assess_release(policy)


def test_legacy_risk_report_remains_local_sensitive() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame(_patient_rows())

    detailed = frame.openmed.risk_report(qi_columns=["age", "zip"])
    serialized_detail = json.dumps(detailed)
    safe_assessment = frame.openmed.assess_release(_policy()).to_json()

    assert _LOCATION_CANARY in serialized_detail
    assert _LOCATION_CANARY not in safe_assessment
