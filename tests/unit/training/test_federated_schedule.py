from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone

import pytest

from openmed.training.federated_schedule import (
    FEDERATED_SCHEDULE_PHASES,
    FEDERATED_SCHEDULE_SCHEMA_VERSION,
    MAX_FEDERATED_PHASE_DURATION_SECONDS,
    FederatedRoundSchedule,
    FederatedScheduleError,
    FederatedSchedulePhase,
)

P = FederatedSchedulePhase
START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _schedule(**overrides: object) -> FederatedRoundSchedule:
    values: dict[str, object] = {
        "enrollment_starts_at": START,
        "update_submission_starts_at": START + timedelta(hours=1),
        "aggregation_starts_at": START + timedelta(hours=2),
        "evaluation_starts_at": START + timedelta(hours=3),
        "finishes_at": START + timedelta(hours=4),
    }
    values.update(overrides)
    return FederatedRoundSchedule(**values)


def test_schedule_serialization_is_deterministic_and_round_trips() -> None:
    schedule = _schedule(
        max_enrollment_duration_seconds=3600,
        max_update_submission_duration_seconds=3600,
        max_aggregation_duration_seconds=3600,
        max_evaluation_duration_seconds=3600,
    )
    expected = {
        "boundaries": {
            "aggregation_starts_at": "2026-09-01T02:00:00Z",
            "enrollment_starts_at": "2026-09-01T00:00:00Z",
            "evaluation_starts_at": "2026-09-01T03:00:00Z",
            "finishes_at": "2026-09-01T04:00:00Z",
            "update_submission_starts_at": "2026-09-01T01:00:00Z",
        },
        "maximum_duration_seconds": {
            "aggregation": 3600,
            "enrollment": 3600,
            "evaluation": 3600,
            "update_submission": 3600,
        },
        "schema_version": FEDERATED_SCHEDULE_SCHEMA_VERSION,
    }

    assert schedule.to_dict() == expected
    assert schedule.to_json() == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    assert FederatedRoundSchedule.from_dict(expected) == schedule
    assert FederatedRoundSchedule.from_json(schedule.to_json()) == schedule


def test_active_and_next_phase_resolution_uses_half_open_boundaries(subtests) -> None:
    for timestamp, active, next_phase in [
        (START - timedelta(microseconds=1), None, P.ENROLLMENT),
        (START, P.ENROLLMENT, P.UPDATE_SUBMISSION),
        (START + timedelta(minutes=30), P.ENROLLMENT, P.UPDATE_SUBMISSION),
        (START + timedelta(hours=1), P.UPDATE_SUBMISSION, P.AGGREGATION),
        (START + timedelta(hours=1, minutes=30), P.UPDATE_SUBMISSION, P.AGGREGATION),
        (START + timedelta(hours=2), P.AGGREGATION, P.EVALUATION),
        (START + timedelta(hours=2, minutes=30), P.AGGREGATION, P.EVALUATION),
        (START + timedelta(hours=3), P.EVALUATION, None),
        (START + timedelta(hours=3, minutes=30), P.EVALUATION, None),
        (START + timedelta(hours=4), None, None),
        (START + timedelta(days=1), None, None),
    ]:
        with subtests.test(timestamp=timestamp, active=active, next_phase=next_phase):
            schedule = _schedule()

            assert schedule.active_phase_at(timestamp) is active
            assert schedule.next_phase_at(timestamp) is next_phase


def test_microseconds_are_preserved_in_canonical_serialization() -> None:
    schedule = _schedule(
        finishes_at=START + timedelta(hours=4, microseconds=1),
    )

    assert schedule.to_dict()["boundaries"]["finishes_at"] == (
        "2026-09-01T04:00:00.000001Z"
    )
    assert FederatedRoundSchedule.from_json(schedule.to_json()) == schedule


def test_naive_boundaries_fail_with_field_only_errors(subtests) -> None:
    for field in [
        "enrollment_starts_at",
        "update_submission_starts_at",
        "aggregation_starts_at",
        "evaluation_starts_at",
        "finishes_at",
    ]:
        with subtests.test(field=field):
            value = getattr(_schedule(), field).replace(tzinfo=None)

            with pytest.raises(FederatedScheduleError) as error:
                _schedule(**{field: value})
            assert str(error.value) == f"{field} must be timezone-aware"


def test_non_utc_boundaries_fail_with_field_only_errors(subtests) -> None:
    for field in [
        "enrollment_starts_at",
        "update_submission_starts_at",
        "aggregation_starts_at",
        "evaluation_starts_at",
        "finishes_at",
    ]:
        with subtests.test(field=field):
            value = getattr(_schedule(), field).astimezone(timezone(timedelta(hours=1)))

            with pytest.raises(FederatedScheduleError) as error:
                _schedule(**{field: value})
            assert str(error.value) == f"{field} must use UTC"


def test_boolean_boundaries_are_rejected(subtests) -> None:
    for field in [
        "enrollment_starts_at",
        "update_submission_starts_at",
        "aggregation_starts_at",
        "evaluation_starts_at",
        "finishes_at",
    ]:
        with subtests.test(field=field):
            with pytest.raises(FederatedScheduleError) as error:
                _schedule(**{field: True})
            assert str(error.value) == f"{field} must be a datetime"


def test_equal_and_reversed_boundaries_are_rejected(subtests) -> None:
    for field, previous_field in [
        ("update_submission_starts_at", "enrollment_starts_at"),
        ("aggregation_starts_at", "update_submission_starts_at"),
        ("evaluation_starts_at", "aggregation_starts_at"),
        ("finishes_at", "evaluation_starts_at"),
    ]:
        for offset in [timedelta(0), timedelta(microseconds=-1)]:
            with subtests.test(
                field=field, previous_field=previous_field, offset=offset
            ):
                schedule = _schedule()
                invalid = getattr(schedule, previous_field) + offset

                with pytest.raises(FederatedScheduleError) as error:
                    _schedule(**{field: invalid})
                assert (
                    str(error.value) == f"{field} must be later than {previous_field}"
                )


def test_invalid_maximum_durations_fail_with_field_only_errors(subtests) -> None:
    for field in [
        "max_enrollment_duration_seconds",
        "max_update_submission_duration_seconds",
        "max_aggregation_duration_seconds",
        "max_evaluation_duration_seconds",
    ]:
        for value in [True, False, 0, -1, 1.5, "3600"]:
            with subtests.test(field=field, value=value):
                with pytest.raises(FederatedScheduleError) as error:
                    _schedule(**{field: value})
                assert (
                    str(error.value) == f"{field} must be a supported positive integer"
                )


def test_exact_maximum_duration_is_allowed_and_one_second_over_is_rejected(
    subtests,
) -> None:
    for maximum_field, end_field in [
        (
            "max_enrollment_duration_seconds",
            "update_submission_starts_at",
        ),
        (
            "max_update_submission_duration_seconds",
            "aggregation_starts_at",
        ),
        (
            "max_aggregation_duration_seconds",
            "evaluation_starts_at",
        ),
        (
            "max_evaluation_duration_seconds",
            "finishes_at",
        ),
    ]:
        with subtests.test(maximum_field=maximum_field, end_field=end_field):
            assert _schedule(**{maximum_field: 3600})

            with pytest.raises(FederatedScheduleError) as error:
                _schedule(**{maximum_field: 3599})
            assert str(error.value) == f"{end_field} exceeds {maximum_field}"


def test_unsupported_configured_and_actual_durations_are_rejected() -> None:
    with pytest.raises(FederatedScheduleError):
        _schedule(
            max_enrollment_duration_seconds=(MAX_FEDERATED_PHASE_DURATION_SECONDS + 1)
        )

    too_late = START + timedelta(seconds=MAX_FEDERATED_PHASE_DURATION_SECONDS + 1)
    with pytest.raises(
        FederatedScheduleError,
        match="update_submission_starts_at exceeds the supported phase duration",
    ):
        _schedule(
            update_submission_starts_at=too_late,
            aggregation_starts_at=too_late + timedelta(hours=1),
            evaluation_starts_at=too_late + timedelta(hours=2),
            finishes_at=too_late + timedelta(hours=3),
        )


def test_phase_resolution_rejects_non_utc_or_non_datetime_inputs(subtests) -> None:
    for timestamp in [
        START.replace(tzinfo=None),
        START.astimezone(timezone(timedelta(hours=-4))),
        True,
    ]:
        with subtests.test(timestamp=timestamp):
            schedule = _schedule()

            with pytest.raises(FederatedScheduleError):
                schedule.active_phase_at(timestamp)
            with pytest.raises(FederatedScheduleError):
                schedule.next_phase_at(timestamp)


def test_deserialization_rejects_incomplete_and_unknown_fields(subtests) -> None:
    for payload in [
        {},
        {"schema_version": FEDERATED_SCHEDULE_SCHEMA_VERSION},
        {
            "schema_version": FEDERATED_SCHEDULE_SCHEMA_VERSION,
            "boundaries": {},
            "maximum_duration_seconds": {},
        },
        {
            **_schedule().to_dict(),
            "site_name": "north-hospital",
        },
    ]:
        with subtests.test(payload=payload):
            with pytest.raises(FederatedScheduleError):
                FederatedRoundSchedule.from_dict(payload)


def test_noncanonical_timestamps_and_duplicate_json_fields_are_rejected() -> None:
    payload = _schedule().to_dict()
    payload["boundaries"]["enrollment_starts_at"] = "2026-09-01T00:00:00+00:00"
    with pytest.raises(FederatedScheduleError):
        FederatedRoundSchedule.from_dict(payload)

    duplicate = (
        _schedule()
        .to_json()
        .replace(
            '"schema_version": "openmed.training.federated_schedule.v1"',
            '"schema_version": "openmed.training.federated_schedule.v1",\n'
            '  "schema_version": "openmed.training.federated_schedule.v1"',
        )
    )
    with pytest.raises(FederatedScheduleError, match="invalid federated schedule JSON"):
        FederatedRoundSchedule.from_json(duplicate)


def test_errors_and_schema_never_retain_caller_metadata() -> None:
    sentinel = "Patient Jane Roe /srv/site-a local_loss=0.42"
    payload = _schedule().to_dict()
    payload["boundaries"]["enrollment_starts_at"] = sentinel

    with pytest.raises(FederatedScheduleError) as error:
        FederatedRoundSchedule.from_dict(payload)
    assert sentinel not in str(error.value)

    forbidden = {
        "client_id",
        "site_name",
        "patient_count",
        "local_metric",
        "network_endpoint",
        "url",
    }
    assert forbidden.isdisjoint(inspect.signature(FederatedRoundSchedule).parameters)
    assert forbidden.isdisjoint(field.name for field in fields(FederatedRoundSchedule))
    serialized = _schedule().to_json()
    assert all(name not in serialized for name in forbidden)


def test_schedule_is_immutable_and_phase_order_is_stable() -> None:
    schedule = _schedule()

    assert FEDERATED_SCHEDULE_PHASES == (
        P.ENROLLMENT,
        P.UPDATE_SUBMISSION,
        P.AGGREGATION,
        P.EVALUATION,
    )
    with pytest.raises(FrozenInstanceError):
        schedule.finishes_at = START  # type: ignore[misc]


def test_schedule_api_is_available_through_lazy_training_exports() -> None:
    import openmed.training as training

    assert training.FederatedRoundSchedule is FederatedRoundSchedule
    assert training.FederatedSchedulePhase is FederatedSchedulePhase
    assert training.__all__.count("FederatedRoundSchedule") == 1
