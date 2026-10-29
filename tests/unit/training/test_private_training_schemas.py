"""Offline contracts for the private-training JSON Schema catalog."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from openmed.training.federated_contribution_floors import (
    FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION,
    FEDERATED_CONTRIBUTION_SCHEMA_VERSION,
    FederatedContributionBand,
    FederatedContributionFloorPolicy,
    seal_federated_contribution,
)
from openmed.training.federated_metrics import (
    FEDERATED_METRIC_SCHEMA_VERSION,
    FederatedMetricEnvelope,
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
    _MAX_DIMENSION,
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


def _contribution_policy_payload() -> dict[str, Any]:
    return FederatedContributionFloorPolicy().to_dict()


def _contribution_payload() -> dict[str, Any]:
    return seal_federated_contribution(
        FederatedContributionFloorPolicy(),
        update_digest=UPDATE,
        local_examples=300,
        local_steps=10,
        effective_batch_size=40,
    ).to_dict()


_VALID_PAYLOADS: dict[str, Callable[[], dict[str, Any]]] = {
    "federated_round_lifecycle": _lifecycle_payload,
    "federated_round_schedule": _schedule_payload,
    "federated_round_status": _status_payload,
    "federated_update_metadata": _update_payload,
    "federated_aggregate_metric": _metric_payload,
    "federated_contribution_floor_policy": _contribution_policy_payload,
    "federated_contribution_declaration": _contribution_payload,
}


def _validator(name: str) -> Draft202012Validator:
    def reject_remote_resolution(uri: str) -> NoReturn:
        raise AssertionError(f"unexpected remote schema resolution: {uri}")

    return Draft202012Validator(
        build_schema(name),
        registry=Registry(retrieve=reject_remote_resolution),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
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
        (
            "federated_contribution_floor_policy",
            lambda p: p.update({"min_local_steps": 1}),
        ),
        (
            "federated_contribution_floor_policy",
            lambda p: p.update({"min_local_examples": 300}),
        ),
        (
            "federated_contribution_floor_policy",
            lambda p: p.update({"min_effective_batch_size": 1 << 17}),
        ),
        (
            "federated_contribution_floor_policy",
            lambda p: p.update({"allow_floors_below_default": True}),
        ),
        (
            "federated_contribution_floor_policy",
            lambda p: p.update({"privacy_mechanism": "none"}),
        ),
        (
            "federated_contribution_declaration",
            lambda p: p.update({"local_examples": 300}),
        ),
        (
            "federated_contribution_declaration",
            lambda p: p.update({"local_step_band": "10"}),
        ),
        (
            "federated_contribution_declaration",
            lambda p: p.update({"contract_digest": "sha256:" + "A" * 64}),
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
        "federated_contribution_floor_policy",
        "federated_contribution_declaration",
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
    floors = build_schema("federated_contribution_floor_policy")
    contribution = build_schema("federated_contribution_declaration")
    assert floors["properties"]["schema_version"]["const"] == (
        FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION
    )
    assert floors["properties"]["privacy_mechanism"]["enum"] == [
        item.value for item in FederatedPrivacyMechanism
    ]
    assert floors["properties"]["min_local_steps"]["enum"] == [
        item.lower_bound for item in FederatedContributionBand
    ]
    assert contribution["properties"]["schema_version"]["const"] == (
        FEDERATED_CONTRIBUTION_SCHEMA_VERSION
    )
    for field in (
        "local_example_band",
        "local_step_band",
        "effective_batch_size_band",
    ):
        assert contribution["properties"][field]["enum"] == [
            item.value for item in FederatedContributionBand
        ]


def test_schema_rendering_is_byte_stable_and_returns_fresh_mappings() -> None:
    first = build_private_training_schemas()
    first["federated_round_lifecycle"]["properties"].clear()

    encoded = render_private_training_schemas()

    assert build_private_training_schemas()["federated_round_lifecycle"]["properties"]
    assert encoded == render_private_training_schemas()
    assert hashlib.sha256(encoded.encode("ascii")).hexdigest() == (
        "dc028cd63442814bb73182242daeedef27cbe4a24e463bb9b85ee8a777f43f19"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"participant_count_band": "suppressed"},
        {"participant_count_band": "suppressed", "aggregate_value": None},
        {"uncertainty_method": "suppressed"},
        {"aggregate_value": None},
        {"confidence_level": 0.5},
        {"uncertainty_lower_bound": 0.1},
        {"uncertainty_upper_bound": 0.9},
        {"uncertainty_method": "confidence_interval"},
        {"metric_kind": "count", "aggregate_value": 0.5},
        {"metric_kind": "count", "clipping_lower_bound": -1},
        {"metric_kind": "count", "clipping_upper_bound": 1.5},
        {"metric_kind": "rate", "clipping_lower_bound": -0.1},
        {"metric_kind": "rate", "clipping_upper_bound": 1.1},
        {"metric_id": "round.loss\n"},
        {"privacy_mechanism_version": "v1\n"},
    ],
)
def test_metric_schema_rejects_invalid_runtime_contract_combinations(changes):
    payload = {**_metric_payload(), **changes}

    with pytest.raises(ValueError):
        FederatedMetricEnvelope.from_dict(payload)
    with pytest.raises(ValidationError):
        _validator("federated_aggregate_metric").validate(payload)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1])
def test_interval_confidence_excludes_both_endpoints(confidence):
    payload = {
        **_metric_payload(),
        "uncertainty_method": "confidence_interval",
        "uncertainty_lower_bound": 0.25,
        "uncertainty_upper_bound": 0.75,
        "confidence_level": confidence,
    }

    with pytest.raises(ValidationError):
        _validator("federated_aggregate_metric").validate(payload)


@pytest.mark.parametrize("kind", list(FederatedMetricKind))
@pytest.mark.parametrize("participants", [0, 100])
@pytest.mark.parametrize("interval", [False, True])
def test_schema_accepts_runtime_releases_and_suppressed_metrics(
    kind, participants, interval
):
    envelope = build_federated_metric_envelope(
        metric_id="round.metric",
        metric_kind=kind,
        aggregate_value=1 if kind is FederatedMetricKind.COUNT else 0.5,
        clipping_lower_bound=0,
        clipping_upper_bound=2 if kind is FederatedMetricKind.COUNT else 1,
        privacy_mechanism=FederatedPrivacyMechanism.LAPLACE,
        privacy_mechanism_version="v1",
        participant_count=participants,
        uncertainty_method=(
            FederatedUncertaintyMethod.CONFIDENCE_INTERVAL
            if interval
            else FederatedUncertaintyMethod.NONE
        ),
        uncertainty_lower_bound=0 if interval else None,
        uncertainty_upper_bound=1 if interval else None,
        confidence_level=0.95 if interval else None,
    )
    _validator("federated_aggregate_metric").validate(envelope.to_dict())


@pytest.mark.parametrize(
    "changes",
    [
        {"reason_code": "operator_cancelled"},
        {"state": "held", "reason_code": None},
        {"state": "aborted", "reason_code": None},
        {"state": "held", "reason_code": "operator_cancelled"},
        {"participant_count": None, "completed_participant_count": 6},
    ],
)
def test_status_schema_rejects_invalid_reason_and_suppression_combinations(changes):
    with pytest.raises(ValidationError):
        _validator("federated_round_status").validate({**_status_payload(), **changes})


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (FederatedRoundState.HELD, FederatedRoundReasonCode.PRIVACY_REVIEW_REQUIRED),
        (FederatedRoundState.ABORTED, FederatedRoundReasonCode.OPERATOR_CANCELLED),
        (FederatedRoundState.COLLECTING, None),
    ],
)
def test_status_schema_accepts_runtime_reason_codes_and_suppression(state, reason):
    payload = build_federated_round_status(
        state=state,
        participant_count=1,
        completed_participant_count=0,
        required_quorum=5,
        reason_code=reason,
    ).to_dict()
    _validator("federated_round_status").validate(payload)


def test_schema_rejects_out_of_range_dimensions_and_newline_digest():
    payload = _update_payload()
    payload["parameters"][0]["shape"] = [_MAX_DIMENSION + 1]
    with pytest.raises(ValidationError):
        _validator("federated_update_metadata").validate(payload)
    payload = _update_payload()
    payload["model_digest"] += "\n"
    with pytest.raises(ValidationError):
        _validator("federated_update_metadata").validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"min_local_examples": 64, "allow_floors_below_default": True},
        {"min_local_steps": 1, "allow_floors_below_default": True},
        {"min_effective_batch_size": 8, "allow_floors_below_default": True},
        {"privacy_mechanism": "gaussian", "min_local_examples": 64},
        {"privacy_mechanism": "laplace", "min_local_steps": 1},
        {
            "privacy_mechanism": "gaussian",
            "min_effective_batch_size": 4,
            "allow_floors_below_default": True,
        },
    ],
)
def test_floor_policy_schema_accepts_runtime_overrides(changes):
    payload = {**_contribution_policy_payload(), **changes}

    FederatedContributionFloorPolicy.from_dict(payload)
    _validator("federated_contribution_floor_policy").validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"min_local_examples": 64},
        {"min_local_steps": 4},
        {"min_effective_batch_size": 16},
        {"privacy_mechanism": "gaussian", "min_local_examples": 32},
        {"privacy_mechanism": "gaussian", "allow_floors_below_default": True},
        {"min_local_steps": 0, "allow_floors_below_default": True},
        {"min_local_steps": 3, "allow_floors_below_default": True},
    ],
)
def test_floor_policy_schema_rejects_unauthorized_lowering(changes):
    payload = {**_contribution_policy_payload(), **changes}

    with pytest.raises(ValueError):
        FederatedContributionFloorPolicy.from_dict(payload)
    with pytest.raises(ValidationError):
        _validator("federated_contribution_floor_policy").validate(payload)


def test_schema_ids_are_unique_and_fragment_definitions_resolve_offline():
    catalog = build_private_training_schemas()
    identifiers = [schema["$id"] for schema in catalog.values()]
    assert len(identifiers) == len(set(identifiers))
    for schema in catalog.values():
        assert schema["$id"].startswith("urn:openmed:training:")
        assert schema["properties"]["schema_version"]["const"] in schema["$id"]
    metric = catalog["federated_aggregate_metric"]
    assert metric["$defs"]
    assert metric["properties"]["aggregate_value"]["$ref"] == "#/$defs/nullableNumber"
    _validator("federated_aggregate_metric").validate(_metric_payload())
    assert json.loads(render_private_training_schemas()) == catalog
