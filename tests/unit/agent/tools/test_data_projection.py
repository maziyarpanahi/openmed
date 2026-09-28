"""Offline tests for minimum-data projection planning."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent.tools import (
    DATA_PROJECTION_PLAN_SCHEMA_VERSION,
    DataProjectionDeniedError,
    DataProjectionPlan,
    DataProjectionValidationError,
    ProjectionDecision,
    ProjectionReasonCode,
    plan_data_projection,
)

PURPOSE = "purpose:org.example/care-summary@1.0.0"
OTHER_PURPOSE = "purpose:org.example/safety-review@1.0.0"
CLINICAL = "data:org.example/clinical-text@1.0.0"
FORMAT = "data:org.example/document-format@1.0.0"
EXTRA = "data:org.example/non-sensitive@1.0.0"


def _field(
    data_class: str,
    *,
    field_type: str = "string",
    purpose: str = PURPOSE,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": field_type,
        "x-openmed-purpose": purpose,
        "x-openmed-minimum-data": "required",
        "x-openmed-data-class": data_class,
        **extra,
    }


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "x-openmed-purpose": PURPOSE,
        "properties": {
            "documents": _field(
                CLINICAL,
                field_type="array",
                items={
                    "type": "object",
                    "properties": {
                        "text": _field(CLINICAL),
                        "format": _field(FORMAT),
                    },
                    "required": ["text", "format"],
                    "additionalProperties": False,
                },
            ),
        },
        "required": ["documents"],
        "additionalProperties": False,
    }


def test_projection_is_deterministic_and_canonical() -> None:
    first = plan_data_projection(
        _schema(),
        workflow_purpose=PURPOSE,
        granted_data_classes=(FORMAT, CLINICAL, EXTRA),
    )
    schema = _schema()
    item_properties = schema["properties"]["documents"]["items"]["properties"]
    schema["properties"]["documents"]["items"]["properties"] = {
        "format": item_properties["format"],
        "text": item_properties["text"],
    }
    second = plan_data_projection(
        schema,
        workflow_purpose=PURPOSE,
        granted_data_classes=(EXTRA, CLINICAL, FORMAT),
    )

    assert first == second
    assert first.schema_version == DATA_PROJECTION_PLAN_SCHEMA_VERSION
    assert first.field_paths == (
        "/documents",
        "/documents/*/format",
        "/documents/*/text",
    )
    assert first.data_classes == (CLINICAL, FORMAT)
    assert first.rationale.approved
    assert all(
        entry.decision is ProjectionDecision.INCLUDE
        and entry.reason_code is ProjectionReasonCode.REQUIRED_FOR_PURPOSE
        for entry in first.rationale.entries
    )
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_unused_granted_classes_do_not_expand_the_projection() -> None:
    plan = plan_data_projection(
        _schema(),
        workflow_purpose=PURPOSE,
        granted_data_classes=(CLINICAL, FORMAT, EXTRA),
    )

    assert EXTRA not in plan.data_classes
    assert EXTRA not in plan.to_json()


def test_requirement_outside_grant_is_rejected_with_value_free_rationale() -> None:
    with pytest.raises(DataProjectionDeniedError) as caught:
        plan_data_projection(
            _schema(),
            workflow_purpose=PURPOSE,
            granted_data_classes=(CLINICAL,),
        )

    error = caught.value
    assert error.code == "projection_denied"
    assert error.field_name == "projection"
    assert not error.rationale.approved
    assert [entry.to_dict() for entry in error.rationale.entries] == [
        {
            "data_class": CLINICAL,
            "decision": "include",
            "field_path": "/documents",
            "reason_code": "required_for_purpose",
            "schema_path": "#/properties/documents",
        },
        {
            "data_class": FORMAT,
            "decision": "deny",
            "field_path": "/documents/*/format",
            "reason_code": "data_class_not_granted",
            "schema_path": "#/properties/documents/items/properties/format",
        },
        {
            "data_class": CLINICAL,
            "decision": "include",
            "field_path": "/documents/*/text",
            "reason_code": "required_for_purpose",
            "schema_path": "#/properties/documents/items/properties/text",
        },
    ]
    assert str(error) == "projection: projection_denied"


def test_mismatched_workflow_purpose_is_rejected_without_echoing_purposes() -> None:
    with pytest.raises(DataProjectionDeniedError) as caught:
        plan_data_projection(
            _schema(),
            workflow_purpose=OTHER_PURPOSE,
            granted_data_classes=(CLINICAL, FORMAT),
        )

    assert caught.value.rationale.to_dict() == {
        "approved": False,
        "entries": [
            {
                "data_class": None,
                "decision": "deny",
                "field_path": None,
                "reason_code": "purpose_mismatch",
                "schema_path": "#/x-openmed-purpose",
            }
        ],
        "schema_version": "openmed.agent.data_projection_rationale.v1",
    }
    combined = str(caught.value) + repr(caught.value.rationale)
    assert PURPOSE not in combined
    assert OTHER_PURPOSE not in combined


@pytest.mark.parametrize(
    "mutate",
    [
        lambda schema: schema.pop("x-openmed-purpose"),
        lambda schema: schema.update({"additionalProperties": True}),
        lambda schema: schema["properties"]["documents"].update(
            {"x-openmed-minimum-data": "optional"}
        ),
        lambda schema: schema.update({"required": []}),
    ],
)
def test_unreviewed_or_overbroad_tool_contracts_fail_closed(mutate: Any) -> None:
    schema = _schema()
    mutate(schema)

    with pytest.raises(DataProjectionValidationError) as caught:
        plan_data_projection(
            schema,
            workflow_purpose=PURPOSE,
            granted_data_classes=(CLINICAL, FORMAT),
        )

    assert caught.value.code == "invalid_tool_schema"
    assert caught.value.field_name == "schema"


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("schema", "clinical-text"),
        ("grant", "clinical-text"),
    ],
)
def test_invalid_data_class_metadata_fails_without_echoing_it(
    target: str, value: str
) -> None:
    schema = _schema()
    grants = (CLINICAL, FORMAT)
    if target == "schema":
        schema["properties"]["documents"]["x-openmed-data-class"] = value
    else:
        grants = (value,)

    with pytest.raises(DataProjectionValidationError) as caught:
        plan_data_projection(
            schema,
            workflow_purpose=PURPOSE,
            granted_data_classes=grants,
        )

    assert value not in str(caught.value)


def test_missing_data_class_annotation_fails_closed() -> None:
    schema = _schema()
    del schema["properties"]["documents"]["x-openmed-data-class"]

    with pytest.raises(DataProjectionValidationError) as caught:
        plan_data_projection(
            schema,
            workflow_purpose=PURPOSE,
            granted_data_classes=(CLINICAL, FORMAT),
        )

    assert caught.value.code == "invalid_governance_identifier"
    assert caught.value.field_name == "schema"


def test_duplicate_grants_fail_closed() -> None:
    with pytest.raises(DataProjectionValidationError) as caught:
        plan_data_projection(
            _schema(),
            workflow_purpose=PURPOSE,
            granted_data_classes=(CLINICAL, CLINICAL),
        )

    assert caught.value.code == "duplicate_grant"


def test_unsafe_field_name_cannot_enter_rationale() -> None:
    schema = _schema()
    schema["properties"] = {"record/clinical-value": _field(CLINICAL)}
    schema["required"] = ["record/clinical-value"]

    with pytest.raises(DataProjectionValidationError) as caught:
        plan_data_projection(
            schema,
            workflow_purpose=PURPOSE,
            granted_data_classes=(CLINICAL,),
        )

    assert caught.value.code == "unsafe_field_name"
    assert "record/clinical-value" not in str(caught.value)


def test_schema_values_are_never_retained_or_rendered() -> None:
    sensitive = "synthetic-clinical-value-never-materialized"
    schema = _schema()
    field = schema["properties"]["documents"]["items"]["properties"]["text"]
    field.update({"description": sensitive, "default": sensitive, "example": sensitive})
    plan = plan_data_projection(
        schema,
        workflow_purpose=PURPOSE,
        granted_data_classes=(CLINICAL, FORMAT),
    )

    combined = " ".join(
        (
            repr(plan),
            repr(plan.rationale),
            *(repr(entry) for entry in plan.rationale.entries),
            plan.to_json(),
        )
    )
    assert sensitive not in combined


def test_empty_schema_produces_an_empty_approved_projection() -> None:
    schema = {
        "type": "object",
        "x-openmed-purpose": PURPOSE,
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    plan = plan_data_projection(
        schema,
        workflow_purpose=PURPOSE,
        granted_data_classes=(),
    )

    assert plan == DataProjectionPlan((), (), plan.rationale)
    assert plan.rationale.approved


def test_validation_errors_do_not_echo_rejected_purpose_or_input_types() -> None:
    rejected = "raw-purpose-value"
    with pytest.raises(DataProjectionValidationError) as purpose_error:
        plan_data_projection(
            _schema(),
            workflow_purpose=rejected,
            granted_data_classes=(CLINICAL, FORMAT),
        )
    with pytest.raises(DataProjectionValidationError) as grant_error:
        plan_data_projection(
            _schema(),
            workflow_purpose=PURPOSE,
            granted_data_classes=[CLINICAL, FORMAT],  # type: ignore[arg-type]
        )

    assert rejected not in str(purpose_error.value)
    assert grant_error.value.code == "invalid_grant"
