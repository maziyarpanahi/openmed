"""Regression tests for opt-in no-PHI core pipeline observability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openmed.core import telemetry as telemetry_module
from openmed.core.pipeline import STAGE_NAMES, Pipeline
from openmed.core.telemetry import (
    DURATION_METRIC_NAME,
    ENTITY_COUNT_METRIC_NAME,
    SPAN_COUNT_METRIC_NAME,
    TELEMETRY_ENABLED_ENV_VAR,
    PipelineTelemetry,
    parse_telemetry_enabled,
    safe_stage_attributes,
    telemetry_enabled_from_env,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

SYNTHETIC_PHI = (
    "Patient Juniper Solstice, MRN JS-1188, DOB 02/03/1979, "
    "phone 425-555-0199, email juniper@example.test."
)
SYNTHETIC_NAME = "Juniper Solstice"
PHI_SUBSTRINGS = (
    SYNTHETIC_NAME,
    "Juniper",
    "Solstice",
    "JS-1188",
    "02/03/1979",
    "425-555-0199",
    "juniper@example.test",
)


def _name_detector(text: str, **_: Any) -> PredictionResult:
    start = text.index(SYNTHETIC_NAME)
    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=SYNTHETIC_NAME,
                label="NAME",
                start=start,
                end=start + len(SYNTHETIC_NAME),
                confidence=0.99,
            )
        ],
        model_name="synthetic-pii-detector",
        timestamp="2026-08-19T00:00:00Z",
    )


@pytest.fixture(autouse=True)
def clear_otel_import_cache() -> Any:
    telemetry_module._load_otel.cache_clear()
    yield
    telemetry_module._load_otel.cache_clear()


@pytest.fixture
def otel_backend() -> Any:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    telemetry = PipelineTelemetry(
        enabled=True,
        tracer=tracer_provider.get_tracer("openmed.pipeline.test"),
        meter=meter_provider.get_meter("openmed.pipeline.test"),
    )
    yield telemetry, span_exporter, metric_reader
    tracer_provider.shutdown()
    meter_provider.shutdown()


def _metrics_by_name(metric_reader: InMemoryMetricReader) -> dict[str, Any]:
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    return {
        metric.name: metric
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }


def _render_observability(
    spans: list[Any],
    metrics: dict[str, Any],
) -> str:
    rendered: list[str] = []
    for span in spans:
        rendered.append(span.name)
        rendered.extend(f"{key}={value}" for key, value in span.attributes.items())
        for event in span.events:
            rendered.append(event.name)
            rendered.extend(f"{key}={value}" for key, value in event.attributes.items())
    for metric in metrics.values():
        rendered.append(metric.name)
        for point in metric.data.data_points:
            rendered.extend(f"{key}={value}" for key, value in point.attributes.items())
    return "\n".join(rendered)


def test_telemetry_is_off_by_default_and_does_not_import_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def unexpected_import(name: str) -> Any:
        calls.append(name)
        raise AssertionError("disabled telemetry must not import OpenTelemetry")

    monkeypatch.setattr(telemetry_module, "import_module", unexpected_import)

    telemetry = PipelineTelemetry()
    result = Pipeline(
        model_detector=_name_detector,
        telemetry=telemetry,
    ).run(SYNTHETIC_PHI)

    assert telemetry.enabled is False
    assert calls == []
    assert set(result.stage_durations_ms) == set(STAGE_NAMES)


def test_explicit_opt_in_without_otel_degrades_to_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_otel(name: str) -> Any:
        if name.startswith("opentelemetry"):
            raise ModuleNotFoundError(name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(telemetry_module, "import_module", missing_otel)

    telemetry = PipelineTelemetry(enabled=True)
    assert telemetry.enabled is False

    result = Pipeline(
        model_detector=_name_detector,
        telemetry=telemetry,
    ).run(SYNTHETIC_PHI)
    assert set(result.stage_durations_ms) == set(STAGE_NAMES)


def test_env_and_constructor_flags_are_explicit_opt_ins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TELEMETRY_ENABLED_ENV_VAR, raising=False)
    assert telemetry_enabled_from_env() is False
    assert Pipeline(model_detector=_name_detector).telemetry.enabled is False

    monkeypatch.setenv(TELEMETRY_ENABLED_ENV_VAR, "true")
    assert telemetry_enabled_from_env() is True
    assert Pipeline(model_detector=_name_detector).telemetry.enabled is True
    assert (
        Pipeline(
            model_detector=_name_detector,
            telemetry_enabled=False,
        ).telemetry.enabled
        is False
    )

    assert parse_telemetry_enabled("1") is True
    assert parse_telemetry_enabled("off") is False
    with pytest.raises(ValueError):
        parse_telemetry_enabled("sometimes")
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        PipelineTelemetry(enabled="true")  # type: ignore[arg-type]


def test_pipeline_emits_one_span_and_metric_point_per_stage(
    otel_backend: Any,
) -> None:
    telemetry, span_exporter, metric_reader = otel_backend
    result = Pipeline(
        model_detector=_name_detector,
        telemetry=telemetry,
    ).run(SYNTHETIC_PHI)

    spans = list(span_exporter.get_finished_spans())
    assert len(spans) == len(STAGE_NAMES)
    assert {span.name for span in spans} == {
        f"openmed.pipeline.{stage}" for stage in STAGE_NAMES
    }
    for index, stage_name in enumerate(STAGE_NAMES, start=1):
        span = next(
            span for span in spans if span.name == f"openmed.pipeline.{stage_name}"
        )
        assert span.attributes["openmed.stage"] == stage_name
        assert span.attributes["openmed.stage.index"] == index
        assert span.attributes["openmed.stage.duration_ms"] == pytest.approx(
            result.stage_duration_ms(stage_name)
        )

    metrics = _metrics_by_name(metric_reader)
    assert set(metrics) == {
        DURATION_METRIC_NAME,
        SPAN_COUNT_METRIC_NAME,
        ENTITY_COUNT_METRIC_NAME,
    }
    for metric in metrics.values():
        points = list(metric.data.data_points)
        assert len(points) == len(STAGE_NAMES)
        assert {point.attributes["openmed.stage"] for point in points} == set(
            STAGE_NAMES
        )
        for point in points:
            assert set(point.attributes) == {
                "openmed.stage",
                "openmed.stage.index",
            }


def test_synthetic_phi_never_reaches_spans_or_metric_labels(
    otel_backend: Any,
) -> None:
    telemetry, span_exporter, metric_reader = otel_backend
    result = Pipeline(
        model_detector=_name_detector,
        telemetry=telemetry,
    ).run(SYNTHETIC_PHI)

    assert SYNTHETIC_NAME not in result.redacted_text
    spans = list(span_exporter.get_finished_spans())
    metrics = _metrics_by_name(metric_reader)
    rendered = _render_observability(spans, metrics)

    assert [value for value in PHI_SUBSTRINGS if value in rendered] == []
    assert all(span.events == () for span in spans)


def test_phi_bearing_exception_is_not_recorded(
    otel_backend: Any,
) -> None:
    telemetry, span_exporter, metric_reader = otel_backend

    def failing_detector(text: str, **_: Any) -> PredictionResult:
        raise RuntimeError(f"failed to process {text}")

    with pytest.raises(RuntimeError, match="failed to process"):
        Pipeline(
            model_detector=failing_detector,
            telemetry=telemetry,
        ).run(SYNTHETIC_PHI)

    spans = list(span_exporter.get_finished_spans())
    failed = next(
        span for span in spans if span.name == "openmed.pipeline.fast_pii_model"
    )
    assert failed.attributes["openmed.stage.failed"] is True
    assert failed.events == ()

    rendered = _render_observability(spans, _metrics_by_name(metric_reader))
    assert [value for value in PHI_SUBSTRINGS if value in rendered] == []


def test_attribute_allowlist_rejects_raw_or_noncanonical_strings() -> None:
    attributes = safe_stage_attributes(
        {
            "openmed.stage": SYNTHETIC_NAME,
            "openmed.stage.labels": ["PERSON", SYNTHETIC_NAME, "JS-1188"],
            "openmed.stage.span_count": 2,
            "openmed.stage.duration_ms": 1.5,
            "openmed.input.text": SYNTHETIC_PHI,
        }
    )

    assert attributes == {
        "openmed.stage.labels": ("PERSON",),
        "openmed.stage.span_count": 2,
        "openmed.stage.duration_ms": 1.5,
    }


def test_otel_extra_contains_no_network_exporter() -> None:
    project_root = Path(__file__).resolve().parents[3]
    with (project_root / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["optional-dependencies"]

    assert dependencies["otel"] == [
        "opentelemetry-api>=1.26,<2",
        "opentelemetry-sdk>=1.26,<2",
    ]
    assert all("exporter" not in dependency for dependency in dependencies["otel"])
