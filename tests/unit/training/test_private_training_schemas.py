"""Offline contracts for the private-training JSON Schema catalog."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from openmed.training.federated_metrics import (
    FEDERATED_METRIC_SCHEMA_VERSION,
    FederatedMetricKind,
    FederatedParticipantCountBand,
    FederatedPrivacyMechanism,
    FederatedUncertaintyMethod,
    build_federated_metric_envelope,
)
from openmed.training.federated_round import (
    FEDERATED_ROUND_SCHEMA_VERSION,
    FederatedRoundLifecycle,
    FederatedRoundState,
)
from openmed.training.federated_schedule import (
    FEDERATED_SCHEDULE_SCHEMA_VERSION,
    FederatedRoundSchedule,
)
from openmed.training.federated_status import (
    FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
    FederatedCompletionBand,
    FederatedQuorumStatus,
    FederatedRoundReasonCode,
    build_federated_round_status,
)
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)
from openmed.training.private_training_schemas import (
    PRIVATE_TRAINING_SCHEMA_BUILDERS,
    PRIVATE_TRAINING_SCHEMA_DIALECT,
    PrivateTrainingSchemaError,
    build_private_training_schemas,
    build_schema,
    render_private_training_schemas,
)

MODEL = "sha256:" + "a" * 64
UPDATE = "sha256:" + "b" * 64
_BASE = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
_DAY = timedelta(days=1)


def _lifecycle_payload() -> dict[str, Any]:
    return FederatedRoundLifecycle(state=FederatedRoundState.COLLECTING).to_dict()


def _schedule_payload() -> dict[str, Any]:
    return FederatedRoundSchedule(
        enrollment_starts_at=_BASE,
        update_submission_starts_at=_BASE + _DAY,
        aggregation_starts_at=_BASE + 2 * _DAY,
        evaluation_starts_at=_BASE + 3 * _DAY,
        finishes_at=_BASE + 4 * _DAY,
    ).to_dict()


def _status_payload() -> dict[str, Any]:
    return build_federated_round_status(
        state=FederatedRoundState.COLLECTING,
        participant_count=10,
        completed_participant_count=6,
        required_quorum=5,
        aggregate_digest_refs=(MODEL,),
    ).to_dict()


def _update_payload() -> dict[str, Any]:
    policy = FederatedUpdatePolicy(
        model_digest=MODEL,
        parameters=(
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
            FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
        ),
        max_total_elements=14,
    )
    payload = {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": MODEL,
        "adapter_format": "dense",
        "parameters": [
            {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"},
            {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
        ],
        "total_elements": 14,
        "update_digest": UPDATE,
        "clipped": True,
    }
    return FederatedUpdateMetadata.from_dict(payload, policy=policy).to_dict()


def _metric_payload() -> dict[str, Any]:
    return build_federated_metric_envelope(
        metric_id="round.loss",
        metric_kind=FederatedMetricKind.BOUNDED_MEAN,
        aggregate_value=0.5,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=FederatedPrivacyMechanism.LAPLACE,
        privacy_mechanism_version="v1",
        participant_count=100,
    ).to_dict()


_VALID_PAYLOADS: dict[str, Callable[[], dict[str, Any]]] = {
    "federated_round_lifecycle": _lifecycle_payload,
    "federated_round_schedule": _schedule_payload,
    "federated_round_status": _status_payload,
    "federated_update_metadata": _update_payload,
    "federated_aggregate_metric": _metric_payload,
}


def _validator(name: str) -> Draft202012Validator:
    def reject_remote_resolution(uri: str) -> NoReturn:
        raise AssertionError(f"unexpected remote schema resolution: {uri}")

    return Draft202012Validator(
        build_schema(name),
        registry=Registry(retrieve=reject_remote_resolution),
    )


@pytest.mark.parametrize("name", sorted(PRIVATE_TRAINING_SCHEMA_BUILDERS))
def test_catalog_schemas_are_valid_draft_2020_12_and_accept_payloads(
    name: str,
) -> None:
    schema = build_schema(name)

    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == PRIVATE_TRAINING_SCHEMA_DIALECT
    _validator(name).validate(_VALID_PAYLOADS[name]())


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("federated_round_lifecycle", lambda p: p.update({"extra": 1})),
        ("federated_round_lifecycle", lambda p: p.update({"state": "finished"})),
        (
            "federated_round_lifecycle",
            lambda p: p.update(
                {"schema_version": "openmed.training.federated_round.v2"}
            ),
        ),
        ("federated_round_schedule", lambda p: p.update({"client_ids": ["c1"]})),
        (
            "federated_round_schedule",
            lambda p: p["boundaries"].update({"finishes_at": "2026-09-01T08:00:00"}),
        ),
        (
            "federated_round_schedule",
            lambda p: p["maximum_duration_seconds"].update({"enrollment": 0}),
        ),
        ("federated_round_status", lambda p: p.update({"site_ids": ["site-a"]})),
        (
            "federated_round_status",
            lambda p: p.update({"participant_count": 1}),
        ),
        ("federated_round_status", lambda p: p.update({"quorum_status": "unknown"})),
        (
            "federated_update_metadata",
            lambda p: p.update({"tensor_shape": [2, 3]}),
        ),
        (
            "federated_update_metadata",
            lambda p: p.update({"adapter_format": "sparse"}),
        ),
        (
            "federated_update_metadata",
            lambda p: p["parameters"][0].update({"dtype": "float128"}),
        ),
        (
            "federated_update_metadata",
            lambda p: p.update({"model_digest": "sha256:synthetic-patient-content"}),
        ),
        ("federated_aggregate_metric", lambda p: p.update({"gradients": [1.0]})),
        (
            "federated_aggregate_metric",
            lambda p: p.update({"metric_kind": "arbitrary"}),
        ),
        (
            "federated_aggregate_metric",
            lambda p: p.update({"privacy_mechanism_version": "latest"}),
        ),
    ],
)
def test_malformed_or_content_bearing_payloads_fail(
    name: str,
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    payload = _VALID_PAYLOADS[name]()
    mutate(payload)

    with pytest.raises(ValidationError):
        _validator(name).validate(payload)


def test_catalog_exposes_every_schema_name_exactly_once() -> None:
    catalog = build_private_training_schemas()

    assert set(catalog) == {
        "federated_round_lifecycle",
        "federated_round_schedule",
        "federated_round_status",
        "federated_update_metadata",
        "federated_aggregate_metric",
    }
    for name in catalog:
        assert catalog[name]["$schema"] == PRIVATE_TRAINING_SCHEMA_DIALECT
    with pytest.raises(PrivateTrainingSchemaError):
        build_schema("not_a_catalog_name")
    with pytest.raises(PrivateTrainingSchemaError):
        build_schema(3)


def test_schemas_are_bounded_and_closed() -> None:
    for name in PRIVATE_TRAINING_SCHEMA_BUILDERS:
        schema = build_schema(name)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

    schedule = build_schema("federated_round_schedule")
    assert (
        schedule["properties"]["maximum_duration_seconds"]["properties"]["enrollment"][
            "anyOf"
        ][0]["maximum"]
        == 365 * 24 * 60 * 60
    )
    update = build_schema("federated_update_metadata")
    assert update["properties"]["total_elements"]["minimum"] == 1
    metric = build_schema("federated_aggregate_metric")
    assert metric["properties"]["minimum_group_size"]["minimum"] == 2


def test_schema_fields_enums_and_versions_track_python_sources() -> None:
    lifecycle = build_schema("federated_round_lifecycle")
    schedule = build_schema("federated_round_schedule")
    status = build_schema("federated_round_status")
    update = build_schema("federated_update_metadata")
    metric = build_schema("federated_aggregate_metric")

    assert lifecycle["properties"]["schema_version"]["const"] == (
        FEDERATED_ROUND_SCHEMA_VERSION
    )
    assert lifecycle["properties"]["state"]["enum"] == [
        item.value for item in FederatedRoundState
    ]
    assert schedule["properties"]["schema_version"]["const"] == (
        FEDERATED_SCHEDULE_SCHEMA_VERSION
    )
    assert status["properties"]["schema_version"]["const"] == (
        FEDERATED_ROUND_STATUS_SCHEMA_VERSION
    )
    assert status["properties"]["quorum_status"]["enum"] == [
        item.value for item in FederatedQuorumStatus
    ]
    assert status["properties"]["completion_band"]["enum"] == [
        item.value for item in FederatedCompletionBand
    ]
    assert status["properties"]["reason_code"]["anyOf"][0]["enum"] == [
        item.value for item in FederatedRoundReasonCode
    ]
    assert update["properties"]["schema_version"]["const"] == (
        FEDERATED_UPDATE_METADATA_SCHEMA_VERSION
    )
    assert update["properties"]["parameters"]["items"]["properties"]["dtype"][
        "enum"
    ] == ["bfloat16", "float16", "float32", "float64"]
    assert metric["properties"]["schema_version"]["const"] == (
        FEDERATED_METRIC_SCHEMA_VERSION
    )
    assert metric["properties"]["metric_kind"]["enum"] == [
        item.value for item in FederatedMetricKind
    ]
    assert metric["properties"]["privacy_mechanism"]["enum"] == [
        item.value for item in FederatedPrivacyMechanism
    ]
    assert metric["properties"]["participant_count_band"]["enum"] == [
        item.value for item in FederatedParticipantCountBand
    ]
    assert metric["properties"]["uncertainty_method"]["enum"] == [
        item.value for item in FederatedUncertaintyMethod
    ]


def test_schema_rendering_is_byte_stable_and_returns_fresh_mappings() -> None:
    first = build_private_training_schemas()
    first["federated_round_lifecycle"]["properties"].clear()

    encoded = render_private_training_schemas()

    assert build_private_training_schemas()["federated_round_lifecycle"]["properties"]
    assert encoded == render_private_training_schemas()
    assert hashlib.sha256(encoded.encode("ascii")).hexdigest() == (
        "10c237aaad7fd4cfb945abff0a8bb13ba9c07423b68ddb21fce8ee14e0b2ed2f"
    )
