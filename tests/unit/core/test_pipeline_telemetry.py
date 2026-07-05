"""Tests for the optional, no-PHI core pipeline telemetry layer.

These cover the OM-821 invariants:

- Telemetry is OFF by default and only records when explicitly opted in.
- OpenTelemetry is an optional, lazily-imported dependency; a no-op backend is
  used when it is absent so importing/using the module never raises.
- Every pipeline stage is traced with no-PHI attributes only.
- Feeding synthetic PHI through the pipeline never leaks raw text or detected
  identifiers into any emitted span attribute.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openmed.core import telemetry as telemetry_module
from openmed.core.pipeline import STAGE_NAMES, Pipeline
from openmed.core.telemetry import (
    TELEMETRY_ENABLED_ENV_VAR,
    PipelineTelemetry,
    otel_available,
    parse_telemetry_enabled,
    safe_stage_attributes,
    telemetry_enabled_from_env,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

# Synthetic PHI only — never a real person. These substrings must never appear
# in any span attribute or metric label emitted by the pipeline.
PHI_TEXT = (
    "Patient Juniper Solstice, MRN 1234567890123, "
    "DOB 02/03/1979, phone 425-555-0199, email jsolstice@example.com."
)
PHI_SUBSTRINGS = (
    "Juniper Solstice",
    "Juniper",
    "Solstice",
    "1234567890123",
    "02/03/1979",
    "425-555-0199",
    "jsolstice@example.com",
)


def _pii_detector(text: str, **_: Any) -> PredictionResult:
    """Return a fixture PII result that flags the synthetic name span."""
    entities = []
    index = text.find("Juniper Solstice")
    if index >= 0:
        entities.append(
            EntityPrediction(
                text="Juniper Solstice",
                label="NAME",
                start=index,
                end=index + len("Juniper Solstice"),
                confidence=0.99,
            )
        )
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="fixture-pii-model",
        timestamp=datetime.now().isoformat(),
    )


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def enabled_telemetry(span_exporter: InMemorySpanExporter) -> PipelineTelemetry:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    tracer = provider.get_tracer("openmed.pipeline.test")
    return PipelineTelemetry(enabled=True, tracer=tracer)


# ---------------------------------------------------------------------------
# Opt-in / default-off behavior
# ---------------------------------------------------------------------------


def test_telemetry_is_disabled_by_default() -> None:
    telemetry = PipelineTelemetry()
    assert telemetry.enabled is False


def test_pipeline_defaults_to_disabled_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TELEMETRY_ENABLED_ENV_VAR, raising=False)
    pipeline = Pipeline(model_detector=_pii_detector)
    assert pipeline.telemetry.enabled is False


def test_env_flag_opts_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENV_VAR, "true")
    assert telemetry_enabled_from_env() is True
    monkeypatch.setenv(TELEMETRY_ENABLED_ENV_VAR, "false")
    assert telemetry_enabled_from_env() is False
    monkeypatch.delenv(TELEMETRY_ENABLED_ENV_VAR, raising=False)
    assert telemetry_enabled_from_env() is False


def test_parse_telemetry_enabled_rejects_garbage() -> None:
    assert parse_telemetry_enabled(None) is False
    assert parse_telemetry_enabled("") is False
    assert parse_telemetry_enabled("1") is True
    assert parse_telemetry_enabled("off") is False
    with pytest.raises(ValueError):
        parse_telemetry_enabled("maybe")


def test_disabled_telemetry_records_nothing_to_a_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    telemetry = PipelineTelemetry(enabled=False)
    telemetry.start_pipeline()
    with telemetry.stage_span(1, STAGE_NAMES[0]) as stage:
        stage.set_span_count(5)
        stage.set_labels(["NAME"])
    # No spans exported, but stage timing metrics are still collected locally.
    assert span_exporter.get_finished_spans() == ()
    summary = telemetry.summary()
    assert summary["enabled"] is False
    assert summary["stage_count"] == 1


# ---------------------------------------------------------------------------
# Optional / lazy OpenTelemetry dependency
# ---------------------------------------------------------------------------


def test_otel_available_reports_installed_backend() -> None:
    # The dev/test environment installs OTel, so this should be True here.
    assert otel_available() is True


def test_opt_in_without_otel_degrades_to_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the optional dependency being absent.
    monkeypatch.setattr(telemetry_module, "_OTEL_IMPORT_ERROR", ImportError("no otel"))
    monkeypatch.setattr(telemetry_module, "_otel_trace", None)

    telemetry = PipelineTelemetry(enabled=True)
    # Requesting telemetry without a backend must not raise and must fall back
    # to the no-op path (enabled becomes False because no span can be created).
    assert telemetry.enabled is False
    telemetry.start_pipeline()
    with telemetry.stage_span(1, STAGE_NAMES[0]) as stage:
        stage.set_span_count(3)
    assert telemetry.summary()["stage_count"] == 1


def test_pipeline_runs_without_otel_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_module, "_OTEL_IMPORT_ERROR", ImportError("no otel"))
    monkeypatch.setattr(telemetry_module, "_otel_trace", None)

    telemetry = PipelineTelemetry(enabled=True)
    pipeline = Pipeline(model_detector=_pii_detector, telemetry=telemetry)
    result = pipeline.run(PHI_TEXT, method="mask")

    assert "Juniper Solstice" not in result.redacted_text
    # Local stage metrics still collected even though no spans are exported.
    assert len(telemetry.stage_metrics) == len(STAGE_NAMES)


# ---------------------------------------------------------------------------
# Stage tracing
# ---------------------------------------------------------------------------


def test_pipeline_traces_every_stage(
    span_exporter: InMemorySpanExporter,
    enabled_telemetry: PipelineTelemetry,
) -> None:
    pipeline = Pipeline(model_detector=_pii_detector, telemetry=enabled_telemetry)
    pipeline.run(PHI_TEXT, method="mask")

    spans = list(span_exporter.get_finished_spans())
    traced_stages = {span.name.removeprefix("openmed.pipeline.") for span in spans}
    assert set(STAGE_NAMES) <= traced_stages
    assert len(spans) == len(STAGE_NAMES)

    for span in spans:
        assert "openmed.stage.duration_ms" in span.attributes
        assert span.attributes["openmed.stage"] in STAGE_NAMES


def test_stage_spans_carry_only_aggregate_signals(
    span_exporter: InMemorySpanExporter,
    enabled_telemetry: PipelineTelemetry,
) -> None:
    pipeline = Pipeline(model_detector=_pii_detector, telemetry=enabled_telemetry)
    pipeline.run(PHI_TEXT, method="mask")

    spans = list(span_exporter.get_finished_spans())
    emit_span = next(
        span for span in spans if span.attributes["openmed.stage"] == "emit"
    )
    # Emit stage reports offsets/counts/hashes/durations — never text.
    assert emit_span.attributes["openmed.stage.span_count"] >= 1
    assert emit_span.attributes["openmed.stage.redacted_length"] > 0
    assert emit_span.attributes["openmed.pipeline.language"] == "en"
    assert emit_span.attributes["openmed.pipeline.doc_id_hash"].startswith(
        "hmac-sha256:"
    )


def test_labels_are_category_names_not_raw_text(
    span_exporter: InMemorySpanExporter,
    enabled_telemetry: PipelineTelemetry,
) -> None:
    pipeline = Pipeline(model_detector=_pii_detector, telemetry=enabled_telemetry)
    pipeline.run(PHI_TEXT, method="mask")

    spans = list(span_exporter.get_finished_spans())
    labelled = [span for span in spans if "openmed.stage.labels" in span.attributes]
    assert labelled, "at least one stage should report a label set"
    for span in labelled:
        for label in span.attributes["openmed.stage.labels"]:
            # Labels are canonical categories; none of them is the raw surface.
            assert label not in PHI_SUBSTRINGS


def test_telemetry_summary_shape(
    enabled_telemetry: PipelineTelemetry,
) -> None:
    pipeline = Pipeline(model_detector=_pii_detector, telemetry=enabled_telemetry)
    pipeline.run(PHI_TEXT, method="mask")

    summary = enabled_telemetry.summary()
    assert summary["enabled"] is True
    assert summary["stage_count"] == len(STAGE_NAMES)
    assert summary["total_duration_ms"] >= 0.0
    assert [stage["name"] for stage in summary["stages"]] == list(STAGE_NAMES)


# ---------------------------------------------------------------------------
# The key invariant: NO raw PHI in any emitted attribute
# ---------------------------------------------------------------------------


def test_no_raw_phi_in_any_span_attribute(
    span_exporter: InMemorySpanExporter,
    enabled_telemetry: PipelineTelemetry,
) -> None:
    pipeline = Pipeline(model_detector=_pii_detector, telemetry=enabled_telemetry)
    result = pipeline.run(PHI_TEXT, method="mask")

    # Sanity: the pipeline actually detected and redacted the synthetic name,
    # so this test would catch a real leak rather than passing vacuously.
    assert "Juniper Solstice" not in result.redacted_text

    spans = list(span_exporter.get_finished_spans())
    assert spans, "telemetry should have exported spans"

    rendered = []
    for span in spans:
        rendered.append(span.name)
        for key, value in span.attributes.items():
            rendered.append(str(key))
            rendered.append(str(value))
        for event in span.events:
            rendered.append(event.name)
            for key, value in event.attributes.items():
                rendered.append(str(key))
                rendered.append(str(value))
    haystack = "\n".join(rendered)

    leaked = [substring for substring in PHI_SUBSTRINGS if substring in haystack]
    assert leaked == [], f"raw PHI leaked into spans: {leaked}"


def test_safe_stage_attributes_drops_unapproved_and_text_keys() -> None:
    attributes = safe_stage_attributes(
        {
            "openmed.stage": "emit",
            "openmed.stage.span_count": 3,
            "openmed.stage.labels": ["NAME", "PHONE"],
            # Unapproved keys carrying raw text must be dropped entirely.
            "openmed.stage.text": "Juniper Solstice",
            "openmed.input.raw": "MRN 1234567890123",
            "openmed.stage.surface": "425-555-0199",
        }
    )
    assert attributes == {
        "openmed.stage": "emit",
        "openmed.stage.span_count": 3,
        "openmed.stage.labels": ("NAME", "PHONE"),
    }


def test_safe_stage_attributes_scrubs_unsafe_label_values() -> None:
    attributes = safe_stage_attributes(
        {"openmed.stage.labels": ["NAME", "Juniper Solstice, MRN 123"]}
    )
    # The free-text value with commas fails the conservative label pattern and
    # is dropped; the clean category name survives.
    assert attributes["openmed.stage.labels"] == ("NAME",)


def test_module_reimport_is_safe() -> None:
    # Re-importing must not raise even if OTel state was monkeypatched elsewhere.
    reloaded = importlib.reload(telemetry_module)
    assert hasattr(reloaded, "PipelineTelemetry")
