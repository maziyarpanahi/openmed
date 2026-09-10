"""Focused tests for the aggregate-only core telemetry exporter."""

from __future__ import annotations

import socket
from dataclasses import dataclass

import pytest

from openmed.core.no_phi_telemetry import (
    MAX_COUNTER_VALUE,
    MAX_ENTITY_COUNT,
    MAX_LATENCY_SECONDS,
    MAX_RESULT_STAGE_DURATIONS,
    CounterName,
    CounterSample,
    LatencySample,
    NoPHITelemetryExporter,
    TelemetrySchemaError,
    TelemetrySnapshot,
    UnapprovedTelemetryKeyError,
    sanitize_exception_category,
)


def _sample(payload: dict, name: str) -> dict:
    return next(item for item in payload["counters"] if item["name"] == name)


def test_pipeline_records_typed_counters_and_bounded_dimensions() -> None:
    exporter = NoPHITelemetryExporter()

    exporter.record_pipeline(
        stage="synthetic-unbounded-stage",
        status="synthetic-unbounded-status",
        method="synthetic-model-id",
        latency_ms=125.0,
        entity_count=3,
    )

    payload = exporter.export()
    run = _sample(payload, CounterName.PIPELINE_RUNS.value)
    entities = _sample(payload, CounterName.PIPELINE_ENTITIES.value)
    assert run["value"] == 1
    assert run["dimensions"] == {
        "exception_category": "unknown",
        "method": "other",
        "stage": "other",
        "status": "other",
    }
    assert entities["value"] == 3

    latency = payload["latencies"][0]
    assert latency["count"] == 1
    assert latency["sum_seconds"] == pytest.approx(0.125)
    assert latency["buckets"]["+Inf"] == 1


def test_record_rejects_unapproved_keys_without_echoing_them() -> None:
    secret_key = "synthetic_patient_identifier"
    exporter = NoPHITelemetryExporter()

    with pytest.raises(UnapprovedTelemetryKeyError) as exc_info:
        exporter.record(
            {
                "counter": CounterName.PIPELINE_RUNS.value,
                secret_key: "synthetic-secret-value",
            }
        )

    assert secret_key not in str(exc_info.value)
    assert exporter.export() == {
        "schema_version": 1,
        "counters": [],
        "latencies": [],
    }


def test_record_rejects_unapproved_dimension_keys_without_echoing_them() -> None:
    secret_key = "synthetic_entity_text"
    exporter = NoPHITelemetryExporter()

    with pytest.raises(UnapprovedTelemetryKeyError) as exc_info:
        exporter.increment(
            CounterName.PIPELINE_RUNS,
            dimensions={secret_key: "synthetic-secret-value"},
        )

    assert secret_key not in str(exc_info.value)
    assert exporter.export()["counters"] == []


def test_exception_categories_use_type_only_and_never_export_messages() -> None:
    secret_message = "synthetic-secret-value must never leave this process"
    exporter = NoPHITelemetryExporter()

    exporter.record_pipeline(
        status="error",
        exception=ValueError(secret_message),
        latency_ms=10,
    )

    exported_json = exporter.export_json()
    prometheus = exporter.render_prometheus()
    assert secret_message not in exported_json
    assert secret_message not in prometheus
    assert 'exception_category="validation"' in prometheus
    assert prometheus.count(f"# TYPE {CounterName.PIPELINE_RUNS.value} counter") == 1
    assert sanitize_exception_category("synthetic-secret-category") == "unknown"
    assert sanitize_exception_category(ValueError) == "validation"


def test_mapping_events_accept_only_approved_measurements() -> None:
    exporter = NoPHITelemetryExporter()

    exporter.record(
        {
            "name": CounterName.PIPELINE_RUNS.value,
            "value": 2,
            "stage": "emit",
            "status": "success",
            "method": "mask",
            "latency_ms": 5,
        }
    )

    payload = exporter.export()
    assert _sample(payload, CounterName.PIPELINE_RUNS.value)["value"] == 2
    assert payload["latencies"][0]["count"] == 1

    with pytest.raises(TelemetrySchemaError):
        exporter.record(
            {
                "counter": CounterName.PIPELINE_RUNS.value,
                "name": CounterName.PIPELINE_RUNS.value,
            }
        )


def test_result_recording_reads_only_aggregate_fields() -> None:
    secret_value = "synthetic-secret-value"

    @dataclass
    class SyntheticResult:
        original_text: str
        redacted_text: str
        spans: tuple[object, ...]
        stage_durations_ms: dict[str, float]

    result = SyntheticResult(
        original_text=secret_value,
        redacted_text="[REDACTED]",
        spans=(object(), object()),
        stage_durations_ms={
            "emit": 2.0,
            "synthetic-secret-stage": 3.0,
        },
    )
    exporter = NoPHITelemetryExporter()

    exporter.record_pipeline_result(result, method="mask")

    payload = exporter.export_json()
    assert secret_value not in payload
    assert "synthetic-secret-stage" not in payload
    assert _sample(exporter.export(), CounterName.PIPELINE_ENTITIES.value)["value"] == 2


def test_export_is_deterministic_and_does_not_open_network_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = NoPHITelemetryExporter()
    second = NoPHITelemetryExporter()
    events = (
        {"counter": CounterName.PIPELINE_FAILURES.value, "status": "error"},
        {"latency_seconds": 0.25, "stage": "emit"},
    )
    for event in events:
        first.record(event)
    for event in reversed(events):
        second.record(event)

    def deny_socket(*_: object, **__: object) -> None:
        raise AssertionError("telemetry export attempted network access")

    monkeypatch.setattr(socket, "socket", deny_socket)
    assert first.export_json() == second.export_json()


@pytest.mark.parametrize(
    "operation",
    [
        lambda exporter: exporter.increment(CounterName.PIPELINE_RUNS, amount=0),
        lambda exporter: exporter.observe_latency_seconds(-1),
        lambda exporter: exporter.record_pipeline(latency_ms=1, latency_seconds=0.001),
    ],
)
def test_invalid_typed_inputs_fail_without_exporting_values(operation) -> None:
    exporter = NoPHITelemetryExporter()

    with pytest.raises(TelemetrySchemaError):
        operation(exporter)

    assert exporter.export()["counters"] == []
    assert exporter.export()["latencies"] == []


def test_pipeline_result_converts_stage_milliseconds_to_seconds() -> None:
    @dataclass
    class SyntheticResult:
        spans: tuple[object, ...]
        stage_durations_ms: dict[str, float]

    exporter = NoPHITelemetryExporter()
    exporter.record_pipeline_result(
        SyntheticResult(spans=(), stage_durations_ms={"emit": 250.0})
    )

    latencies = {
        item["dimensions"]["stage"]: item for item in exporter.export()["latencies"]
    }
    assert latencies["emit"]["sum_seconds"] == pytest.approx(0.25)
    assert latencies["pipeline"]["sum_seconds"] == pytest.approx(0.25)


def test_mapping_event_validation_is_atomic() -> None:
    exporter = NoPHITelemetryExporter()

    with pytest.raises(TelemetrySchemaError):
        exporter.record(
            {
                "counter": CounterName.PIPELINE_RUNS.value,
                "entity_count": "SYNTHETIC-SECRET",
            }
        )

    assert exporter.export() == {
        "schema_version": 1,
        "counters": [],
        "latencies": [],
    }


def test_hostile_mapping_and_result_failures_are_value_free() -> None:
    class HostileMapping(dict[str, object]):
        def items(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("SYNTHETIC-SECRET")

    exporter = NoPHITelemetryExporter()
    with pytest.raises(TelemetrySchemaError, match="could not be read") as event_error:
        exporter.record(HostileMapping())
    assert "SYNTHETIC-SECRET" not in str(event_error.value)

    with pytest.raises(TelemetrySchemaError, match="could not be read") as dims_error:
        exporter.increment(
            CounterName.PIPELINE_RUNS,
            dimensions=HostileMapping(),
        )
    assert "SYNTHETIC-SECRET" not in str(dims_error.value)

    class HostileResult:
        @property
        def stage_durations_ms(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("SYNTHETIC-SECRET")

    with pytest.raises(TelemetrySchemaError, match="could not be read") as result_error:
        exporter.record_pipeline_result(HostileResult())
    assert "SYNTHETIC-SECRET" not in str(result_error.value)
    assert exporter.export()["counters"] == []


def test_pipeline_result_stage_and_entity_inputs_are_bounded() -> None:
    @dataclass
    class SyntheticResult:
        spans: object
        stage_durations_ms: dict[str, float]

    exporter = NoPHITelemetryExporter()
    too_many_stages = {
        f"stage_{index}": 1.0 for index in range(MAX_RESULT_STAGE_DURATIONS + 1)
    }
    with pytest.raises(UnapprovedTelemetryKeyError):
        exporter.record_pipeline_result(
            SyntheticResult(spans=(), stage_durations_ms=too_many_stages)
        )

    class TooManySpans:
        def __len__(self) -> int:
            return MAX_ENTITY_COUNT + 1

    with pytest.raises(TelemetrySchemaError, match="entity count"):
        exporter.record_pipeline_result(
            SyntheticResult(spans=TooManySpans(), stage_durations_ms={})
        )
    with pytest.raises(TelemetrySchemaError, match="spans must be a sequence"):
        exporter.record_pipeline_result(
            SyntheticResult(
                spans="SYNTHETIC-SECRET",
                stage_durations_ms={},
            )
        )
    assert exporter.export()["counters"] == []


def test_counter_and_latency_totals_are_bounded_without_partial_updates() -> None:
    exporter = NoPHITelemetryExporter()
    exporter.increment(CounterName.PIPELINE_RUNS, amount=MAX_COUNTER_VALUE)

    with pytest.raises(TelemetrySchemaError, match="safe limit"):
        exporter.increment(CounterName.PIPELINE_RUNS)
    assert (
        _sample(exporter.export(), CounterName.PIPELINE_RUNS.value)["value"]
        == MAX_COUNTER_VALUE
    )

    with pytest.raises(TelemetrySchemaError, match="bounded"):
        exporter.observe_latency_seconds(MAX_LATENCY_SECONDS + 1)
    assert exporter.export()["latencies"] == []


def test_public_snapshot_types_reject_arbitrary_names_labels_and_outcomes() -> None:
    with pytest.raises(
        TelemetrySchemaError, match="name is not approved"
    ) as name_error:
        CounterSample("SYNTHETIC-SECRET", 1)  # type: ignore[arg-type]
    assert "SYNTHETIC-SECRET" not in str(name_error.value)

    with pytest.raises(TelemetrySchemaError, match="dimensions are invalid"):
        CounterSample(
            CounterName.PIPELINE_RUNS,
            1,
            (("SYNTHETIC-SECRET", "SYNTHETIC-SECRET"),),
        )
    with pytest.raises(TelemetrySchemaError, match="name is not approved"):
        LatencySample(
            name="SYNTHETIC-SECRET",
            count=1,
            sum_seconds=0.1,
            buckets=(("1.0", 1), ("+Inf", 1)),
        )
    with pytest.raises(TelemetrySchemaError, match="schema is unsupported"):
        TelemetrySnapshot(schema_version=2, counters=(), latencies=())


def test_latency_sum_is_deterministic_across_observation_order() -> None:
    first = NoPHITelemetryExporter()
    second = NoPHITelemetryExporter()
    for value in (0.1, 0.2, 0.3):
        first.observe_latency_seconds(value, dimensions={"stage": "emit"})
    for value in (0.3, 0.2, 0.1):
        second.observe_latency_seconds(value, dimensions={"stage": "emit"})

    assert first.export_json() == second.export_json()


def test_duplicate_exception_sources_are_rejected_before_mutation() -> None:
    exporter = NoPHITelemetryExporter()

    with pytest.raises(TelemetrySchemaError, match="multiple sources"):
        exporter.record_pipeline(
            exception=ValueError("SYNTHETIC-SECRET"),
            exception_category="validation",
        )
    with pytest.raises(TelemetrySchemaError, match="duplicate fields"):
        exporter.record(
            {
                "counter": CounterName.PIPELINE_RUNS.value,
                "exception": ValueError("SYNTHETIC-SECRET"),
                "exception_category": "validation",
            }
        )
    assert exporter.export()["counters"] == []
