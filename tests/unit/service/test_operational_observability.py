"""Value-free metrics, traces, and aggregate operational alerts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.service.metrics import (
    OPERATIONAL_DURATION_NAME,
    OPERATIONAL_TOTAL_NAME,
    PrometheusMetricsRegistry,
)
from openmed.service.operational import (
    OPERATIONS_BY_CATEGORY,
    OperationalAlert,
    OperationalCategory,
    OperationalEvent,
    OperationalState,
)
from openmed.service.tracing import operational_trace_attributes, safe_trace_attributes

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_ROOT = ROOT / "openmed" / "core" / "schemas" / "json"


def test_all_operational_families_emit_only_closed_vocabulary_labels() -> None:
    registry = PrometheusMetricsRegistry()
    for category, operations in OPERATIONS_BY_CATEGORY.items():
        registry.record_operational_event(
            OperationalEvent(
                category=category,
                operation=sorted(operations)[0],
                state=OperationalState.SUCCESS,
                duration_seconds=0.25,
            )
        )

    rendered = registry.render()

    for category in OperationalCategory:
        assert f'category="{category.value}"' in rendered
    assert f"# TYPE {OPERATIONAL_TOTAL_NAME} counter" in rendered
    assert f"# TYPE {OPERATIONAL_DURATION_NAME} summary" in rendered
    for line in rendered.splitlines():
        if line.startswith(OPERATIONAL_TOTAL_NAME):
            assert set(_label_names(line)) == {"category", "operation", "state"}


def test_operational_trace_and_alerts_are_aggregate_and_value_free() -> None:
    event = OperationalEvent(
        category=OperationalCategory.EXPORT,
        operation="authorize",
        state=OperationalState.DENIED,
        count=3,
    )
    alert = OperationalAlert(
        alert_id="alert_deadbeef00112233",
        category=event.category,
        operation=event.operation,
        state=event.state,
        observed_count=3,
        threshold_count=2,
        window_seconds=60,
    )

    assert operational_trace_attributes(event) == {
        "openmed.operation.category": "export",
        "openmed.operation.count": 3,
        "openmed.operation.name": "authorize",
        "openmed.operation.state": "denied",
    }
    assert alert.to_dict()["observed_count"] == 3
    assert alert.to_dict()["alert_id"] == "alert_deadbeef00112233"
    event_schema = json.loads(
        (SCHEMA_ROOT / "operational_event.schema.json").read_text(encoding="utf-8")
    )
    alert_schema = json.loads(
        (SCHEMA_ROOT / "operational_alert.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(event_schema).validate(event.to_dict())
    Draft202012Validator(alert_schema).validate(alert.to_dict())
    rendered = repr(event.to_dict()) + repr(alert.to_dict())
    for forbidden in (
        "source_text",
        "identifier",
        "vault",
        "fact_id",
        "reviewer",
    ):
        assert forbidden not in rendered


def test_arbitrary_operation_values_cannot_reach_telemetry() -> None:
    canary = "synthetic-sensitive-operation-4452"

    with pytest.raises(ValueError) as error:
        OperationalEvent(
            category=OperationalCategory.QUERY,
            operation=canary,
            state=OperationalState.FAILURE,
        )

    assert canary not in str(error.value)
    with pytest.raises(ValueError) as category_error:
        OperationalEvent(
            category=canary,  # type: ignore[arg-type]
            operation="list",
            state=OperationalState.FAILURE,
        )
    assert canary not in str(category_error.value)
    assert (
        safe_trace_attributes(
            {
                "openmed.operation.category": canary,
                "openmed.operation.name": canary,
                "openmed.operation.state": canary,
                "openmed.operation.count": canary,
            }
        )
        == {}
    )


def _label_names(line: str) -> tuple[str, ...]:
    labels = line.split("{", 1)[1].split("}", 1)[0]
    return tuple(item.split("=", 1)[0] for item in labels.split(","))
