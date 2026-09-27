"""Shared forbidden-field catalog run against every private-learning validator."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from openmed.training.federated_metrics import (
    FederatedMetricEnvelope,
    FederatedMetricError,
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundState,
    FederatedRoundStateError,
)
from openmed.training.federated_schedule import (
    FederatedRoundSchedule,
    FederatedScheduleError,
)
from openmed.training.federated_status import build_federated_round_status
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)
from tests.fixtures.private_learning_forbidden import (
    FORBIDDEN_FIELD_CASES,
    ForbiddenFieldCase,
)

_MODEL = "sha256:" + "a" * 64
_UPDATE = "sha256:" + "b" * 64
_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
_POLICY = FederatedUpdatePolicy(
    model_digest=_MODEL,
    parameters=(
        FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
    ),
    max_total_elements=6,
)


def _update_payload() -> dict[str, Any]:
    return {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": _MODEL,
        "adapter_format": "dense",
        "parameters": [
            {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"}
        ],
        "total_elements": 6,
        "update_digest": _UPDATE,
        "clipped": True,
    }


def _metric_payload() -> dict[str, Any]:
    return build_federated_metric_envelope(
        metric_id="safe_completion_rate",
        metric_kind=FederatedMetricKind.RATE,
        aggregate_value=0.82,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    ).to_dict()


def _schedule_payload() -> dict[str, Any]:
    return FederatedRoundSchedule(
        enrollment_starts_at=_START,
        update_submission_starts_at=_START + timedelta(hours=1),
        aggregation_starts_at=_START + timedelta(hours=2),
        evaluation_starts_at=_START + timedelta(hours=3),
        finishes_at=_START + timedelta(hours=4),
    ).to_dict()


def _round_payload() -> dict[str, Any]:
    return FederatedRoundLifecycle(state=FederatedRoundState.PLANNED).to_dict()


def _as_json(parse: Callable[[str], object]) -> Callable[[Any], object]:
    return lambda payload: parse(json.dumps(payload))


@dataclass(frozen=True)
class Surface:
    """A validator entry point and the stable rejection it reports."""

    build: Callable[[], dict[str, Any]]
    parse: Callable[[Any], object]
    error: type[Exception]
    rejection: str
    level: tuple[str | int, ...] = ()


SURFACES: dict[str, Surface] = {
    "update-metadata-dict": Surface(
        _update_payload,
        lambda p: FederatedUpdateMetadata.from_dict(p, policy=_POLICY),
        FederatedUpdateMetadataError,
        "invalid metadata fields",
    ),
    "update-metadata-json": Surface(
        _update_payload,
        _as_json(lambda p: FederatedUpdateMetadata.from_json(p, policy=_POLICY)),
        FederatedUpdateMetadataError,
        "invalid metadata fields",
    ),
    "update-metadata-parameter": Surface(
        _update_payload,
        lambda p: FederatedUpdateMetadata.from_dict(p, policy=_POLICY),
        FederatedUpdateMetadataError,
        "invalid metadata fields",
        ("parameters", 0),
    ),
    "metric-envelope": Surface(
        _metric_payload,
        FederatedMetricEnvelope.from_dict,
        FederatedMetricError,
        "invalid federated metric envelope fields",
    ),
    "schedule-dict": Surface(
        _schedule_payload,
        FederatedRoundSchedule.from_dict,
        FederatedScheduleError,
        "invalid federated schedule payload",
    ),
    "schedule-json": Surface(
        _schedule_payload,
        _as_json(FederatedRoundSchedule.from_json),
        FederatedScheduleError,
        "invalid federated schedule payload",
    ),
    "schedule-boundaries": Surface(
        _schedule_payload,
        FederatedRoundSchedule.from_dict,
        FederatedScheduleError,
        "invalid federated schedule boundaries",
        ("boundaries",),
    ),
    "schedule-durations": Surface(
        _schedule_payload,
        FederatedRoundSchedule.from_dict,
        FederatedScheduleError,
        "invalid federated schedule durations",
        ("maximum_duration_seconds",),
    ),
    "round-lifecycle-dict": Surface(
        _round_payload,
        FederatedRoundLifecycle.from_dict,
        FederatedRoundStateError,
        "invalid federated round lifecycle payload",
    ),
    "round-lifecycle-json": Surface(
        _round_payload,
        _as_json(FederatedRoundLifecycle.from_json),
        FederatedRoundStateError,
        "invalid federated round lifecycle payload",
    ),
}

_CASE_IDS = [case.reason_code for case in FORBIDDEN_FIELD_CASES]


def _target(payload: Any, level: tuple[str | int, ...]) -> dict[str, Any]:
    for key in level:
        payload = payload[key]
    return payload


def test_catalog_has_stable_unique_reason_codes_and_fields() -> None:
    assert all(re.fullmatch(r"forbidden_[a-z_]+", code) for code in _CASE_IDS)
    assert len(set(_CASE_IDS)) == len(FORBIDDEN_FIELD_CASES)
    fields = [case.field for case in FORBIDDEN_FIELD_CASES]
    assert len(set(fields)) == len(fields)
    assert {
        "site_id",
        "client_id",
        "path",
        "endpoint",
        "examples",
        "tensors",
        "gradients",
        "message",
        "local_metrics",
        "patient_count",
    } == set(fields)


def test_catalog_values_are_placeholders_only() -> None:
    for case in FORBIDDEN_FIELD_CASES:
        rendered = json.dumps(case.value)
        if case.marker is not None:
            assert "placeholder" in rendered
            assert case.marker in rendered
    endpoints = [c for c in FORBIDDEN_FIELD_CASES if c.field == "endpoint"]
    assert all(".invalid" in c.value for c in endpoints)


@pytest.mark.parametrize("surface", SURFACES.values(), ids=SURFACES.keys())
def test_valid_payloads_parse_without_forbidden_fields(surface: Surface) -> None:
    assert surface.parse(surface.build()) is not None


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("surface", SURFACES.values(), ids=SURFACES.keys())
def test_each_forbidden_field_is_rejected_on_its_own(
    surface: Surface, case: ForbiddenFieldCase
) -> None:
    payload = surface.build()
    original = copy.deepcopy(payload)
    _target(payload, surface.level)[case.field] = copy.deepcopy(case.value)

    with pytest.raises(surface.error) as error:
        surface.parse(payload)

    assert str(error.value) == surface.rejection
    assert case.field not in str(error.value)
    if case.marker is not None:
        assert case.marker not in str(error.value)

    del _target(payload, surface.level)[case.field]
    assert payload == original
    assert surface.parse(payload) is not None


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=_CASE_IDS)
def test_round_status_builder_has_no_surface_for_forbidden_fields(
    case: ForbiddenFieldCase,
) -> None:
    arguments: dict[str, Any] = {
        "state": FederatedRoundState.PLANNED,
        "participant_count": 0,
        "completed_participant_count": 0,
        "required_quorum": 2,
    }
    assert build_federated_round_status(**arguments) is not None

    with pytest.raises(TypeError) as error:
        build_federated_round_status(**arguments, **{case.field: case.value})

    if case.marker is not None:
        assert case.marker not in str(error.value)
