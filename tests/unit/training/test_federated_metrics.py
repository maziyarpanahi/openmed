from __future__ import annotations

import inspect
import json
import traceback
from dataclasses import FrozenInstanceError

import pytest

from openmed.training import (
    FEDERATED_METRIC_SCHEMA_VERSION,
    FederatedMetricEnvelope,
    FederatedMetricError,
    FederatedMetricKind,
    FederatedParticipantCountBand,
    FederatedPrivacyMechanism,
    FederatedUncertaintyMethod,
    build_federated_metric_envelope,
)

K = FederatedMetricKind
P = FederatedPrivacyMechanism
U = FederatedUncertaintyMethod


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "metric_id": "documents_processed",
            "metric_kind": K.COUNT,
            "aggregate_value": 18,
            "clipping_lower_bound": 0,
            "clipping_upper_bound": 100,
            "privacy_mechanism": P.THRESHOLD_ONLY,
            "privacy_mechanism_version": "v1",
            "participant_count": 12,
        },
        {
            "metric_id": "safe_completion_rate",
            "metric_kind": K.RATE,
            "aggregate_value": 0.82,
            "clipping_lower_bound": 0.0,
            "clipping_upper_bound": 1.0,
            "privacy_mechanism": P.LAPLACE,
            "privacy_mechanism_version": "v2",
            "participant_count": 25,
            "uncertainty_method": U.CONFIDENCE_INTERVAL,
            "uncertainty_lower_bound": 0.75,
            "uncertainty_upper_bound": 0.88,
            "confidence_level": 0.95,
        },
        {
            "metric_id": "bounded_utility_mean",
            "metric_kind": K.BOUNDED_MEAN,
            "aggregate_value": 7.25,
            "clipping_lower_bound": 0.0,
            "clipping_upper_bound": 10.0,
            "privacy_mechanism": P.GAUSSIAN,
            "privacy_mechanism_version": "v1",
            "participant_count": 8,
            "uncertainty_method": U.CONFIDENCE_INTERVAL,
            "uncertainty_lower_bound": 6.8,
            "uncertainty_upper_bound": 7.7,
            "confidence_level": 0.9,
        },
    ],
)
def test_metric_examples_round_trip_with_deterministic_json(
    arguments: dict[str, object],
) -> None:
    envelope = build_federated_metric_envelope(**arguments)
    payload = envelope.to_dict()

    assert payload["schema_version"] == FEDERATED_METRIC_SCHEMA_VERSION
    assert envelope.to_json() == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert FederatedMetricEnvelope.from_dict(json.loads(envelope.to_json())) == envelope


def test_small_group_values_are_suppressed_and_not_retained() -> None:
    envelope = build_federated_metric_envelope(
        metric_id="bounded_utility_mean",
        metric_kind=K.BOUNDED_MEAN,
        aggregate_value=7.25,
        clipping_lower_bound=0.0,
        clipping_upper_bound=10.0,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=3,
        minimum_group_size=5,
        uncertainty_method=U.CONFIDENCE_INTERVAL,
        uncertainty_lower_bound=6.8,
        uncertainty_upper_bound=7.7,
        confidence_level=0.9,
    )

    assert envelope.participant_count_band is FederatedParticipantCountBand.SUPPRESSED
    assert envelope.aggregate_value is None
    assert envelope.uncertainty_method is U.SUPPRESSED
    assert envelope.uncertainty_lower_bound is None
    assert envelope.uncertainty_upper_bound is None
    assert envelope.confidence_level is None
    assert "7.25" not in repr(envelope)
    assert "participant_count" not in envelope.to_dict()


def test_small_group_still_rejects_non_finite_input() -> None:
    with pytest.raises(FederatedMetricError):
        build_federated_metric_envelope(
            metric_id="bounded_utility_mean",
            metric_kind=K.BOUNDED_MEAN,
            aggregate_value=float("nan"),
            clipping_lower_bound=0.0,
            clipping_upper_bound=10.0,
            privacy_mechanism=P.THRESHOLD_ONLY,
            privacy_mechanism_version="v1",
            participant_count=3,
            minimum_group_size=5,
        )


@pytest.mark.parametrize(
    ("participant_count", "expected"),
    [
        (4, FederatedParticipantCountBand.SUPPRESSED),
        (5, FederatedParticipantCountBand.MINIMUM_TO_UNDER_DOUBLE),
        (9, FederatedParticipantCountBand.MINIMUM_TO_UNDER_DOUBLE),
        (10, FederatedParticipantCountBand.DOUBLE_TO_UNDER_FOURFOLD),
        (19, FederatedParticipantCountBand.DOUBLE_TO_UNDER_FOURFOLD),
        (20, FederatedParticipantCountBand.FOURFOLD_OR_MORE),
    ],
)
def test_participant_count_bands_follow_the_minimum_group_rule(
    participant_count: int,
    expected: FederatedParticipantCountBand,
) -> None:
    envelope = build_federated_metric_envelope(
        metric_id="documents_processed",
        metric_kind=K.COUNT,
        aggregate_value=18,
        clipping_lower_bound=0,
        clipping_upper_bound=100,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=participant_count,
        minimum_group_size=5,
    )

    assert envelope.participant_count_band is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"aggregate_value": 101},
        {"aggregate_value": 1.5},
        {"aggregate_value": 18.0},
        {"aggregate_value": float("nan")},
        {"aggregate_value": float("inf")},
        {"clipping_lower_bound": -1},
        {"clipping_lower_bound": 0.0},
        {"clipping_lower_bound": 100},
        {"clipping_upper_bound": 100.0},
        {"clipping_upper_bound": float("inf")},
        {"participant_count": -1},
        {"participant_count": True},
        {"minimum_group_size": 1},
        {"minimum_group_size": True},
        {"metric_kind": "count"},
        {"privacy_mechanism": "custom"},
        {"privacy_mechanism_version": "latest"},
    ],
)
def test_count_clipping_finite_number_and_group_invariants_fail_closed(
    overrides: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "metric_id": "documents_processed",
        "metric_kind": K.COUNT,
        "aggregate_value": 18,
        "clipping_lower_bound": 0,
        "clipping_upper_bound": 100,
        "privacy_mechanism": P.THRESHOLD_ONLY,
        "privacy_mechanism_version": "v1",
        "participant_count": 12,
    }
    arguments.update(overrides)

    with pytest.raises(FederatedMetricError):
        build_federated_metric_envelope(**arguments)


@pytest.mark.parametrize(
    "overrides",
    [
        {"aggregate_value": 1.01},
        {"clipping_lower_bound": -0.01},
        {"clipping_upper_bound": 1.01},
        {"uncertainty_lower_bound": 0.9},
        {"uncertainty_upper_bound": 0.7},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"confidence_level": float("nan")},
    ],
)
def test_rate_and_uncertainty_invariants_fail_closed(
    overrides: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "metric_id": "safe_completion_rate",
        "metric_kind": K.RATE,
        "aggregate_value": 0.82,
        "clipping_lower_bound": 0.0,
        "clipping_upper_bound": 1.0,
        "privacy_mechanism": P.LAPLACE,
        "privacy_mechanism_version": "v1",
        "participant_count": 12,
        "uncertainty_method": U.CONFIDENCE_INTERVAL,
        "uncertainty_lower_bound": 0.75,
        "uncertainty_upper_bound": 0.88,
        "confidence_level": 0.95,
    }
    arguments.update(overrides)

    with pytest.raises(FederatedMetricError):
        build_federated_metric_envelope(**arguments)


def test_uncertainty_values_require_a_confidence_interval() -> None:
    with pytest.raises(FederatedMetricError):
        build_federated_metric_envelope(
            metric_id="safe_completion_rate",
            metric_kind=K.RATE,
            aggregate_value=0.82,
            clipping_lower_bound=0.0,
            clipping_upper_bound=1.0,
            privacy_mechanism=P.LAPLACE,
            privacy_mechanism_version="v1",
            participant_count=12,
            uncertainty_lower_bound=0.75,
        )


def test_large_integer_counts_preserve_exact_bounds_without_float_conversion() -> None:
    lower = 1 << 60
    envelope = build_federated_metric_envelope(
        metric_id="large_document_count",
        metric_kind=K.COUNT,
        aggregate_value=lower + 1,
        clipping_lower_bound=lower,
        clipping_upper_bound=lower + 2,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    )

    assert envelope.aggregate_value == lower + 1
    assert envelope.clipping_lower_bound == lower
    assert envelope.clipping_upper_bound == lower + 2


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "site_name",
        "client_id",
        "patient_count",
        "local_losses",
        "gradients",
        "examples",
        "endpoint",
        "per_client_metrics",
    ],
)
def test_unknown_and_client_level_fields_cannot_enter_the_schema(
    forbidden_field: str,
) -> None:
    envelope = build_federated_metric_envelope(
        metric_id="safe_completion_rate",
        metric_kind=K.RATE,
        aggregate_value=0.82,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    )
    payload = envelope.to_dict()
    payload[forbidden_field] = "North Hospital"

    with pytest.raises(FederatedMetricError) as error:
        FederatedMetricEnvelope.from_dict(payload)
    assert "North Hospital" not in str(error.value)
    assert (
        forbidden_field
        not in inspect.signature(build_federated_metric_envelope).parameters
    )
    assert forbidden_field not in inspect.signature(FederatedMetricEnvelope).parameters


def test_invalid_enum_errors_do_not_chain_submitted_values() -> None:
    marker = "SYNTHETIC_PRIVATE_SENTINEL_3011"
    envelope = build_federated_metric_envelope(
        metric_id="safe_completion_rate",
        metric_kind=K.RATE,
        aggregate_value=0.82,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    )
    payload = envelope.to_dict()
    payload["metric_kind"] = marker

    with pytest.raises(FederatedMetricError) as error:
        FederatedMetricEnvelope.from_dict(payload)

    rendered = "".join(traceback.format_exception(error.value))
    assert marker not in rendered
    assert error.value.__cause__ is None


def test_direct_construction_and_unknown_versions_are_rejected() -> None:
    with pytest.raises(FederatedMetricError):
        FederatedMetricEnvelope(
            metric_id="documents_processed",
            metric_kind=K.COUNT,
            aggregate_value=18,
            clipping_lower_bound=0,
            clipping_upper_bound=100,
            privacy_mechanism=P.THRESHOLD_ONLY,
            privacy_mechanism_version="v1",
            minimum_group_size=5,
            participant_count_band=(
                FederatedParticipantCountBand.DOUBLE_TO_UNDER_FOURFOLD
            ),
            uncertainty_method=U.NONE,
        )

    payload = build_federated_metric_envelope(
        metric_id="documents_processed",
        metric_kind=K.COUNT,
        aggregate_value=18,
        clipping_lower_bound=0,
        clipping_upper_bound=100,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    ).to_dict()
    payload["schema_version"] = "openmed.training.federated_metric.v2"
    with pytest.raises(FederatedMetricError):
        FederatedMetricEnvelope.from_dict(payload)


def test_metric_envelopes_are_immutable() -> None:
    envelope = build_federated_metric_envelope(
        metric_id="documents_processed",
        metric_kind=K.COUNT,
        aggregate_value=18,
        clipping_lower_bound=0,
        clipping_upper_bound=100,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    )

    with pytest.raises(FrozenInstanceError):
        envelope.metric_id = "changed"
