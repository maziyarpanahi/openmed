"""Synthetic agent event integration with caller-owned in-memory OTel sinks."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openmed.agent.action_phases import ActionPhase
from openmed.agent.event_attributes import (
    ALLOWED_ATTRIBUTES,
    AttributeKind,
    EventAttributeError,
    EventAttributes,
)
from openmed.agent.telemetry import (
    DURATION_METRIC_NAME,
    EVENT_COUNTER_NAME,
    AgentTelemetry,
)

pytestmark = pytest.mark.integration

RUN_ID = "run_0123456789abcdef0123456789abcdef"
ACTION_ID = "act_0123456789abcdef0123456789abcdef"
DIGEST = "sha256:" + "0" * 64
CANARIES = (
    "Juniper Solstice",
    "JS-1188",
    "02/03/1979",
    "425-555-0199",
    "juniper@example.test",
    "山田太郎",
    "Bearer synthetic-secret",
    "/private/synthetic/clinical-note.txt",
)
VALUES = {
    "run_id": RUN_ID,
    "action_id": ACTION_ID,
    "parent_action_id": ACTION_ID,
    "capability_id": "capability:openmed.agent/summarize",
    "policy_id": "policy:openmed.agent/phi-guard@1.0.0",
    "purpose_id": "purpose:openmed.agent/care-coordination",
    "tool_id": "tool:openmed.agent/summarize@1.0.0",
    "workflow_id": "workflow:openmed.agent/discharge",
    "execution_stage": "tool_call",
    "outcome_class": "success",
    "outcome_reason": "completed",
    "input_digest": DIGEST,
    "output_digest": DIGEST,
    "artifact_digest": DIGEST,
    "sequence_number": 3,
    "attempt_number": 1,
    "retry_count": 0,
    "tool_call_count": 2,
    "artifact_count": 1,
    "duration_ms": 12.5,
    "retryable": False,
    "redacted": True,
}


@pytest.fixture
def backend() -> Any:
    exporter = InMemorySpanExporter()
    traces = TracerProvider()
    traces.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    yield (
        traces.get_tracer("openmed.agent"),
        meters.get_meter("openmed.agent"),
        exporter,
        reader,
    )
    traces.shutdown()
    meters.shutdown()


def metrics(reader: InMemoryMetricReader) -> dict[str, Any]:
    data = reader.get_metrics_data()
    assert data is not None
    return {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }


@pytest.mark.parametrize("mode", ["omit", "hash"])
def test_canaries_and_all_identifier_kinds_never_reach_sinks(
    backend: Any, mode: str
) -> None:
    tracer, meter, exporter, reader = backend
    telemetry = AgentTelemetry(
        enabled=True, tracer=tracer, meter=meter, run_id_mode=mode
    )
    assert set(VALUES) == set(ALLOWED_ATTRIBUTES)
    for canary in CANARIES:
        for key in ("arguments", "output", "patient_name", "input_digest", "run_id"):
            with pytest.raises(EventAttributeError) as error:
                EventAttributes.from_mapping({key: canary})
            assert canary not in str(error.value)
        with pytest.raises(RuntimeError):
            with telemetry.event_span("running", EventAttributes.from_mapping(VALUES)):
                raise RuntimeError(canary)

    spans = exporter.get_finished_spans()
    assert len(spans) == len(CANARIES)
    emitted_metrics = metrics(reader)
    rendered = repr([(s.name, dict(s.attributes), s.events, s.status) for s in spans])
    rendered += repr(emitted_metrics)
    assert all(canary not in rendered for canary in CANARIES)
    assert RUN_ID not in rendered
    assert ACTION_ID not in rendered
    assert all(
        value not in rendered for key, value in VALUES.items() if key.endswith("_id")
    )
    allowed_kinds = {
        AttributeKind.EXECUTION_STAGE,
        AttributeKind.OUTCOME_CLASS,
        AttributeKind.OUTCOME_REASON,
        AttributeKind.DIGEST,
        AttributeKind.COUNT,
        AttributeKind.DURATION_MS,
        AttributeKind.FLAG,
    }
    for span in spans:
        assert span.name == "openmed.agent.running"
        assert not span.events
        assert span.status.description is None
        for key, value in span.attributes.items():
            field = key.removeprefix("openmed.agent.")
            assert field in ALLOWED_ATTRIBUTES
            if field == "run_id":
                assert mode == "hash"
                assert (
                    value
                    == "sha256:"
                    + hashlib.sha256(
                        b"openmed.agent.telemetry.run_id.v1\0" + RUN_ID.encode("ascii")
                    ).hexdigest()
                )
            else:
                assert ALLOWED_ATTRIBUTES[field] in allowed_kinds
                assert value == VALUES[field]
        if mode == "omit":
            assert "openmed.agent.run_id" not in span.attributes
    for metric in emitted_metrics.values():
        for point in metric.data.data_points:
            assert set(point.attributes) == {
                "openmed.agent.phase",
                "openmed.agent.execution_stage",
                "openmed.agent.outcome_class",
                "openmed.agent.outcome_reason",
                "openmed.agent.redacted",
                "openmed.agent.retryable",
            }
    counter = emitted_metrics[EVENT_COUNTER_NAME].data.data_points[0]
    duration = emitted_metrics[DURATION_METRIC_NAME].data.data_points[0]
    assert counter.value == len(CANARIES)
    assert duration.count == len(CANARIES)
    assert duration.sum == len(CANARIES) * 12.5


def test_synthetic_run_has_deterministic_structure_and_counts(backend: Any) -> None:
    tracer, meter, exporter, reader = backend
    telemetry = AgentTelemetry(enabled=True, tracer=tracer, meter=meter)
    phases = [
        (ActionPhase.WAITING_REVIEW, "review_required", "human_gate"),
        (ActionPhase.ABORTED, "policy_denied", "phi_policy"),
        (ActionPhase.COMPLETED, "success", "completed"),
        (ActionPhase.ABORTED, "failed", "timeout"),
    ]
    snapshots = []
    for _ in range(2):
        with telemetry.event_span(
            "running", EventAttributes.from_mapping({"duration_ms": 10})
        ):
            for sequence, (phase, outcome, reason) in enumerate(phases):
                with telemetry.event_span(
                    phase,
                    EventAttributes.from_mapping(
                        {
                            "sequence_number": sequence,
                            "outcome_class": outcome,
                            "outcome_reason": reason,
                            "duration_ms": 2,
                        }
                    ),
                ):
                    pass
        spans = exporter.get_finished_spans()
        parent = spans[-1]
        assert parent.name == "openmed.agent.running"
        assert parent.parent is None
        assert [s.name for s in spans] == [
            "openmed.agent.waiting-review",
            "openmed.agent.aborted",
            "openmed.agent.completed",
            "openmed.agent.aborted",
            "openmed.agent.running",
        ]
        assert all(s.parent.span_id == parent.context.span_id for s in spans[:-1])
        assert len({s.context.trace_id for s in spans}) == 1
        snapshots.append(
            [(s.name, dict(s.attributes), s.parent is None) for s in spans]
        )
        exporter.clear()
    # Random trace IDs and wall-clock timestamps are provider-owned, not golden data.
    assert snapshots[0] == snapshots[1]
    emitted = metrics(reader)
    assert set(emitted) == {EVENT_COUNTER_NAME, DURATION_METRIC_NAME}
    points = emitted[EVENT_COUNTER_NAME].data.data_points
    assert sum(p.value for p in points) == 10
    assert len(points) == 5
    assert all(p.value == 2 for p in points)
    assert sum(p.sum for p in emitted[DURATION_METRIC_NAME].data.data_points) == 36


def test_hashes_correlate_runs_without_colliding_or_metric_cardinality(
    backend: Any,
) -> None:
    tracer, meter, exporter, reader = backend
    telemetry = AgentTelemetry(
        enabled=True, tracer=tracer, meter=meter, run_id_mode="hash"
    )
    for run_id in [RUN_ID, RUN_ID, "run_" + "f" * 32]:
        with telemetry.event_span(
            "completed",
            EventAttributes.from_mapping({"run_id": run_id, "duration_ms": 1}),
        ):
            pass
    digests = [
        s.attributes["openmed.agent.run_id"] for s in exporter.get_finished_spans()
    ]
    assert digests[0] == digests[1]
    assert digests[0] != digests[2]
    points = metrics(reader)[EVENT_COUNTER_NAME].data.data_points
    assert len(points) == 1
    assert points[0].attributes == {"openmed.agent.phase": "completed"}
    assert points[0].value == 3
