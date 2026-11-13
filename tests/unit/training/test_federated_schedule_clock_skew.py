"""Clock-skew tolerance and admission contracts for federated round schedules."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import pytest

import openmed.training.federated_schedule as fs
from openmed.training.federated_schedule import (
    FEDERATED_SCHEDULE_SCHEMA_VERSION,
    FederatedRoundSchedule,
    FederatedScheduleError,
    FederatedSchedulePhase,
)

P = FederatedSchedulePhase
V1 = "openmed.training.federated_schedule.v1"
V2 = "openmed.training.federated_schedule.v2"
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
SKEW = 120
TICK = timedelta(microseconds=1)
TOLERANCE = timedelta(seconds=SKEW)

BEFORE = "local_clock_before_window"
AFTER = "local_clock_after_window"
EARLY = "received_before_window"
LATE = "received_after_window"

# (earlier phase, later phase, transition offset from START)
TRANSITIONS = [
    (None, P.ENROLLMENT, timedelta(0)),
    (P.ENROLLMENT, P.UPDATE_SUBMISSION, timedelta(hours=1)),
    (P.UPDATE_SUBMISSION, P.AGGREGATION, timedelta(hours=2)),
    (P.AGGREGATION, P.EVALUATION, timedelta(hours=3)),
    (P.EVALUATION, None, timedelta(hours=4)),
]

# (offset from transition, earlier client, later client, earlier coordinator,
#  later coordinator); None means the decision admits the phase.
BOUNDARY_DECISIONS = [
    (-TOLERANCE - TICK, None, BEFORE, None, EARLY),
    (-TOLERANCE, AFTER, BEFORE, None, EARLY),
    (-TICK, AFTER, BEFORE, None, EARLY),
    (timedelta(0), AFTER, BEFORE, LATE, None),
    (TOLERANCE - TICK, AFTER, BEFORE, LATE, None),
    (TOLERANCE, AFTER, None, LATE, None),
]


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


def _skewed(skew: object = SKEW, **overrides: object) -> FederatedRoundSchedule:
    return _schedule(schema_version=V2, max_clock_skew_seconds=skew, **overrides)


def _code(decision: object) -> str | None:
    reason = decision.reason_code  # type: ignore[attr-defined]
    return None if reason is None else str(reason)


def test_v2_schema_constants_are_declared_and_bounded() -> None:
    assert FEDERATED_SCHEDULE_SCHEMA_VERSION == V1
    assert fs.FEDERATED_SCHEDULE_V2_SCHEMA_VERSION == V2
    assert fs.FEDERATED_SCHEDULE_SCHEMA_VERSIONS == (V1, V2)
    assert fs.MAX_FEDERATED_CLOCK_SKEW_SECONDS == 15 * 60
    assert [item.value for item in fs.FederatedAdmissionReasonCode] == [
        BEFORE,
        AFTER,
        EARLY,
        LATE,
    ]
    assert [item.value for item in fs.FederatedClockBasis] == [
        "site_advisory",
        "coordinator_receipt",
    ]


@pytest.mark.parametrize(
    ("offset", "earlier_client", "later_client", "earlier_coord", "later_coord"),
    BOUNDARY_DECISIONS,
)
@pytest.mark.parametrize(
    ("earlier", "later", "transition"),
    TRANSITIONS,
    ids=["enrollment", "submission", "aggregation", "evaluation", "finish"],
)
def test_decisions_at_plus_and_minus_skew_around_each_transition(
    earlier: FederatedSchedulePhase | None,
    later: FederatedSchedulePhase | None,
    transition: timedelta,
    offset: timedelta,
    earlier_client: str | None,
    later_client: str | None,
    earlier_coord: str | None,
    later_coord: str | None,
) -> None:
    schedule = _skewed()
    moment = START + transition + offset

    for phase, client, coordinator in (
        (earlier, earlier_client, earlier_coord),
        (later, later_client, later_coord),
    ):
        if phase is None:
            continue
        local = schedule.client_admission_at(phase, moment)
        received = schedule.coordinator_admission_at(
            phase, moment, coordinator_sequence=1
        )

        assert local.phase is phase
        assert _code(local) == client
        assert local.admitted is (client is None)
        assert received.phase is phase
        assert _code(received) == coordinator
        assert received.admitted is (coordinator is None)
        assert schedule.client_admission_at(phase, moment) == local
        assert (
            schedule.coordinator_admission_at(phase, moment, coordinator_sequence=1)
            == received
        )


@pytest.mark.parametrize("phase", list(FederatedSchedulePhase))
@pytest.mark.parametrize("skew", [0, 1, SKEW, 900])
def test_coordinator_admission_uses_receipt_time_only(
    phase: FederatedSchedulePhase, skew: int
) -> None:
    exact = _schedule()
    skewed = _skewed(skew)
    start, end = {
        P.ENROLLMENT: (START, START + timedelta(hours=1)),
        P.UPDATE_SUBMISSION: (START + timedelta(hours=1), START + timedelta(hours=2)),
        P.AGGREGATION: (START + timedelta(hours=2), START + timedelta(hours=3)),
        P.EVALUATION: (START + timedelta(hours=3), START + timedelta(hours=4)),
    }[phase]

    for moment in (start - TICK, start, end - TICK, end):
        decision = skewed.coordinator_admission_at(
            phase, moment, coordinator_sequence=5
        )
        assert decision == exact.coordinator_admission_at(
            phase, moment, coordinator_sequence=5
        )
        assert decision.admitted is (start <= moment < end)
        assert decision.admitted is (skewed.active_phase_at(moment) is phase)


def test_client_outside_tolerated_skew_is_refused_before_training() -> None:
    schedule = _skewed()
    submission_opens = START + timedelta(hours=1)
    trained: list[FederatedSchedulePhase] = []

    def start_training(clock: Callable[[], datetime]) -> object:
        decision = schedule.client_admission_at(P.UPDATE_SUBMISSION, clock())
        if decision.admitted:
            trained.append(decision.phase)
        return decision

    early = start_training(lambda: submission_opens + TOLERANCE - TICK)
    late = start_training(lambda: START + timedelta(hours=2) - TOLERANCE)

    assert trained == []
    assert _code(early) == BEFORE
    assert _code(late) == AFTER
    assert str(early.clock_basis) == "site_advisory"

    admitted = start_training(lambda: submission_opens + TOLERANCE)
    assert admitted.admitted is True
    assert trained == [P.UPDATE_SUBMISSION]


def test_zero_skew_client_window_matches_the_phase_window() -> None:
    schedule = _skewed(0)
    opens = START + timedelta(hours=1)
    closes = START + timedelta(hours=2)

    def code_at(moment: datetime) -> str | None:
        return _code(schedule.client_admission_at(P.UPDATE_SUBMISSION, moment))

    assert code_at(opens - TICK) == BEFORE
    assert code_at(opens) is None
    assert code_at(closes - TICK) is None
    assert code_at(closes) == AFTER


def test_client_admission_requires_a_declared_skew_bound() -> None:
    with pytest.raises(FederatedScheduleError) as error:
        _schedule().client_admission_at(P.ENROLLMENT, START)
    assert str(error.value) == "client admission requires max_clock_skew_seconds"


@pytest.mark.parametrize(
    "skew",
    [True, False, -1, 901, 1.5, 60.0, math.nan, math.inf, -math.inf, "60", None],
)
def test_negative_excessive_and_non_integer_skew_values_are_rejected(
    skew: object,
) -> None:
    with pytest.raises(FederatedScheduleError) as error:
        _skewed(skew)
    assert str(error.value) == (
        "max_clock_skew_seconds must be a supported non-negative integer"
    )


@pytest.mark.parametrize("skew", [0, 1, 900])
def test_supported_skew_values_round_trip(skew: int) -> None:
    schedule = _skewed(skew)
    payload = schedule.to_dict()

    assert payload["schema_version"] == V2
    assert payload["max_clock_skew_seconds"] == skew
    assert set(payload) == {
        "boundaries",
        "max_clock_skew_seconds",
        "maximum_duration_seconds",
        "schema_version",
    }
    assert FederatedRoundSchedule.from_dict(payload) == schedule
    assert FederatedRoundSchedule.from_json(schedule.to_json()) == schedule
    assert schedule.to_json() == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_v1_schedule_cannot_declare_a_skew_bound() -> None:
    with pytest.raises(FederatedScheduleError) as error:
        _schedule(max_clock_skew_seconds=SKEW)
    assert str(error.value) == (
        "max_clock_skew_seconds is not supported by this schedule schema"
    )


@pytest.mark.parametrize(
    ("end_field", "start_offset"),
    [
        ("update_submission_starts_at", timedelta(0)),
        ("aggregation_starts_at", timedelta(hours=1)),
        ("evaluation_starts_at", timedelta(hours=2)),
        ("finishes_at", timedelta(hours=3)),
    ],
)
def test_skew_must_leave_a_client_window_in_every_phase(
    end_field: str, start_offset: timedelta
) -> None:
    boundaries = {
        "update_submission_starts_at": START + timedelta(hours=1),
        "aggregation_starts_at": START + timedelta(hours=2),
        "evaluation_starts_at": START + timedelta(hours=3),
        "finishes_at": START + timedelta(hours=4),
    }
    names = list(boundaries)
    shortened = START + start_offset + timedelta(seconds=2 * SKEW)
    shift = boundaries[end_field] - shortened
    for name in names[names.index(end_field) :]:
        boundaries[name] -= shift

    with pytest.raises(FederatedScheduleError) as error:
        _skewed(**boundaries)
    assert str(error.value) == (
        f"{end_field} leaves no client window within max_clock_skew_seconds"
    )

    for name in names[names.index(end_field) :]:
        boundaries[name] += TICK
    assert _skewed(**boundaries).max_clock_skew_seconds == SKEW


def test_v1_schedules_serialize_and_parse_unchanged() -> None:
    schedule = _schedule(max_enrollment_duration_seconds=3600)
    golden = (
        "{\n"
        '  "boundaries": {\n'
        '    "aggregation_starts_at": "2026-09-01T02:00:00Z",\n'
        '    "enrollment_starts_at": "2026-09-01T00:00:00Z",\n'
        '    "evaluation_starts_at": "2026-09-01T03:00:00Z",\n'
        '    "finishes_at": "2026-09-01T04:00:00Z",\n'
        '    "update_submission_starts_at": "2026-09-01T01:00:00Z"\n'
        "  },\n"
        '  "maximum_duration_seconds": {\n'
        '    "aggregation": null,\n'
        '    "enrollment": 3600,\n'
        '    "evaluation": null,\n'
        '    "update_submission": null\n'
        "  },\n"
        '  "schema_version": "openmed.training.federated_schedule.v1"\n'
        "}\n"
    )

    assert schedule.to_json() == golden
    assert schedule.schema_version == V1
    assert schedule.max_clock_skew_seconds is None
    restored = FederatedRoundSchedule.from_json(golden)
    assert restored == schedule
    assert restored.max_clock_skew_seconds is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p.pop("max_clock_skew_seconds"),
            "invalid federated schedule payload",
        ),
        (
            lambda p: p.update({"schema_version": V1}),
            "invalid federated schedule payload",
        ),
        (
            lambda p: p.update(
                {"schema_version": "openmed.training.federated_schedule.v3"}
            ),
            "unsupported federated schedule schema",
        ),
        (
            lambda p: p.update({"max_clock_skew_seconds": -1}),
            "max_clock_skew_seconds must be a supported non-negative integer",
        ),
        (
            lambda p: p.update({"max_clock_skew_seconds": 901}),
            "max_clock_skew_seconds must be a supported non-negative integer",
        ),
        (
            lambda p: p.update({"max_clock_skew_seconds": True}),
            "max_clock_skew_seconds must be a supported non-negative integer",
        ),
        (
            lambda p: p.update({"clock_offset_seconds": 3}),
            "invalid federated schedule payload",
        ),
    ],
)
def test_v2_deserialization_is_strict(mutate: object, message: str) -> None:
    payload = _skewed().to_dict()
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(FederatedScheduleError) as error:
        FederatedRoundSchedule.from_dict(payload)
    assert str(error.value) == message


def test_v1_payload_with_a_skew_field_is_rejected() -> None:
    payload = _schedule().to_dict()
    payload["max_clock_skew_seconds"] = SKEW

    with pytest.raises(FederatedScheduleError) as error:
        FederatedRoundSchedule.from_dict(payload)
    assert str(error.value) == "invalid federated schedule payload"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e2", "60.0"])
def test_non_finite_and_float_skew_json_is_rejected(literal: str) -> None:
    encoded = (
        _skewed()
        .to_json()
        .replace(
            f'"max_clock_skew_seconds": {SKEW}', f'"max_clock_skew_seconds": {literal}'
        )
    )
    assert literal in encoded

    with pytest.raises(FederatedScheduleError) as error:
        FederatedRoundSchedule.from_json(encoded)
    assert str(error.value) == (
        "max_clock_skew_seconds must be a supported non-negative integer"
    )


@pytest.mark.parametrize("method", ["client_admission_at", "coordinator_admission_at"])
def test_admission_rejects_unknown_phases_and_non_utc_clocks(method: str) -> None:
    schedule = _skewed()
    check = getattr(schedule, method)
    field = "local_time" if method == "client_admission_at" else "received_at"
    extra = {} if method == "client_admission_at" else {"coordinator_sequence": 1}

    with pytest.raises(FederatedScheduleError) as error:
        check("enrollment", START, **extra)
    assert str(error.value) == "phase must be a federated schedule phase"

    naive = START.replace(tzinfo=None)
    with pytest.raises(FederatedScheduleError) as error:
        check(P.ENROLLMENT, naive, **extra)
    assert str(error.value) == f"{field} must be timezone-aware"

    shifted = START.astimezone(timezone(timedelta(hours=-4)))
    with pytest.raises(FederatedScheduleError) as error:
        check(P.ENROLLMENT, shifted, **extra)
    assert str(error.value) == f"{field} must use UTC"
    assert "2026" not in str(error.value)


@pytest.mark.parametrize("sequence", [True, False, 0, -1, 1.5, "1", None])
def test_coordinator_sequence_must_be_a_positive_integer(sequence: object) -> None:
    with pytest.raises(FederatedScheduleError) as error:
        _skewed().coordinator_admission_at(
            P.ENROLLMENT, START, coordinator_sequence=sequence
        )
    assert str(error.value) == "coordinator_sequence must be a positive integer"


def test_shared_decision_records_have_no_site_timestamps_or_offsets() -> None:
    schedule = _skewed()
    moment = START + timedelta(minutes=30, seconds=17, microseconds=4321)
    local = schedule.client_admission_at(P.ENROLLMENT, moment)
    received = schedule.coordinator_admission_at(
        P.ENROLLMENT, moment, coordinator_sequence=9
    )

    assert local.to_dict() == {
        "admitted": True,
        "clock_basis": "site_advisory",
        "coordinator_sequence": None,
        "phase": "enrollment",
        "reason_code": None,
    }
    assert received.to_dict() == {
        "admitted": True,
        "clock_basis": "coordinator_receipt",
        "coordinator_sequence": 9,
        "phase": "enrollment",
        "reason_code": None,
    }
    record_fields = {item.name for item in fields(fs.FederatedPhaseAdmission)}
    assert record_fields == {
        "phase",
        "clock_basis",
        "reason_code",
        "coordinator_sequence",
    }
    for decision in (local, received):
        rendered = json.dumps(decision.to_dict()) + repr(decision)
        assert decision.to_json() == (
            json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        for forbidden in ("2026", "00:30", "4321", "offset", "skew", "_at"):
            assert forbidden not in rendered
        assert all(
            not isinstance(getattr(decision, item), datetime) for item in record_fields
        )


def test_admission_evidence_is_ordered_by_coordinator_sequence() -> None:
    schedule = _skewed()
    phase = P.UPDATE_SUBMISSION
    opens = START + timedelta(hours=1)
    # Receipt order and site-reported order both disagree with the sequence.
    arrivals = [
        schedule.coordinator_admission_at(phase, opens, coordinator_sequence=3),
        schedule.coordinator_admission_at(
            phase, opens + timedelta(minutes=2), coordinator_sequence=1
        ),
        schedule.coordinator_admission_at(
            phase, opens + timedelta(hours=1), coordinator_sequence=2
        ),
    ]

    ordered = fs.order_admission_evidence(arrivals)

    assert [item.coordinator_sequence for item in ordered] == [1, 2, 3]
    assert [_code(item) for item in ordered] == [None, LATE, None]
    assert fs.order_admission_evidence(reversed(arrivals)) == ordered
    assert type(ordered) is tuple


def test_admission_evidence_rejects_duplicates_and_site_clock_records() -> None:
    schedule = _skewed()
    first = schedule.coordinator_admission_at(
        P.ENROLLMENT, START, coordinator_sequence=4
    )
    duplicate = schedule.coordinator_admission_at(
        P.ENROLLMENT, START + timedelta(minutes=5), coordinator_sequence=4
    )
    local = schedule.client_admission_at(P.ENROLLMENT, START + TOLERANCE)

    with pytest.raises(FederatedScheduleError) as error:
        fs.order_admission_evidence([first, duplicate])
    assert str(error.value) == "duplicate coordinator sequence"

    with pytest.raises(FederatedScheduleError) as error:
        fs.order_admission_evidence([first, local])
    assert str(error.value) == "admission evidence requires a coordinator sequence"

    for invalid in ("evidence", [first.to_dict()], None):
        with pytest.raises(FederatedScheduleError) as error:
            fs.order_admission_evidence(invalid)
        assert str(error.value) == "invalid admission evidence"


@pytest.mark.parametrize(
    "values",
    [
        {"phase": "enrollment", "clock_basis": "site_advisory"},
        {"phase": P.ENROLLMENT, "clock_basis": "site_clock"},
        {
            "phase": P.ENROLLMENT,
            "clock_basis": "site_advisory",
            "coordinator_sequence": 1,
        },
        {"phase": P.ENROLLMENT, "clock_basis": "coordinator_receipt"},
        {
            "phase": P.ENROLLMENT,
            "clock_basis": "coordinator_receipt",
            "coordinator_sequence": True,
        },
        {
            "phase": P.ENROLLMENT,
            "clock_basis": "coordinator_receipt",
            "coordinator_sequence": 1,
            "reason_code": BEFORE,
        },
        {
            "phase": P.ENROLLMENT,
            "clock_basis": "site_advisory",
            "reason_code": LATE,
        },
        {
            "phase": P.ENROLLMENT,
            "clock_basis": "site_advisory",
            "reason_code": "clock_offset",
        },
    ],
)
def test_decision_records_reject_inconsistent_fields(
    values: dict[str, object],
) -> None:
    resolved = dict(values)
    for key, enum in (
        ("clock_basis", fs.FederatedClockBasis),
        ("reason_code", fs.FederatedAdmissionReasonCode),
    ):
        if resolved.get(key) in {item.value for item in enum}:
            resolved[key] = enum(resolved[key])

    with pytest.raises(FederatedScheduleError):
        fs.FederatedPhaseAdmission(**resolved)


def test_admission_api_is_available_through_lazy_training_exports() -> None:
    import openmed.training as training

    for name in (
        "FEDERATED_SCHEDULE_SCHEMA_VERSIONS",
        "FEDERATED_SCHEDULE_V2_SCHEMA_VERSION",
        "MAX_FEDERATED_CLOCK_SKEW_SECONDS",
        "FederatedAdmissionReasonCode",
        "FederatedClockBasis",
        "FederatedPhaseAdmission",
        "order_admission_evidence",
    ):
        assert getattr(training, name) is getattr(fs, name)
        assert training.__all__.count(name) == 1
        assert name in fs.__all__
