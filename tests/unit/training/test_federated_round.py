from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.training.federated_round import (
    FEDERATED_ROUND_SCHEMA_VERSION,
    FEDERATED_ROUND_STATES,
    FEDERATED_ROUND_TERMINAL_STATES,
    FEDERATED_ROUND_TRANSITIONS,
    FederatedRoundLifecycle,
    FederatedRoundState,
    FederatedRoundStateError,
    FederatedRoundTransitionError,
    allowed_round_transitions,
    can_transition_round,
    validate_round_transition,
)

S = FederatedRoundState

EXPECTED_TRANSITIONS = {
    S.PLANNED: {S.PREFLIGHT, S.ABORTED},
    S.PREFLIGHT: {S.COLLECTING, S.ABORTED},
    S.COLLECTING: {S.AGGREGATING, S.ABORTED},
    S.AGGREGATING: {S.EVALUATING, S.ABORTED},
    S.EVALUATING: {S.HELD, S.PROMOTED, S.ABORTED},
    S.HELD: {S.EVALUATING, S.ABORTED},
    S.PROMOTED: set(),
    S.ABORTED: set(),
}


@pytest.mark.parametrize("current", FEDERATED_ROUND_STATES)
def test_transition_table_covers_every_state(current: FederatedRoundState) -> None:
    expected = EXPECTED_TRANSITIONS[current]

    assert set(FEDERATED_ROUND_TRANSITIONS[current]) == expected
    assert set(allowed_round_transitions(current)) == expected
    for target in FEDERATED_ROUND_STATES:
        assert can_transition_round(current, target) is (target in expected)
        if target in expected:
            validate_round_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (S.PLANNED, S.COLLECTING),
        (S.COLLECTING, S.PREFLIGHT),
        (S.EVALUATING, S.PREFLIGHT),
        (S.PROMOTED, S.ABORTED),
        (S.ABORTED, S.PLANNED),
    ],
)
def test_skipped_backward_and_post_terminal_transitions_are_rejected(
    current: FederatedRoundState,
    target: FederatedRoundState,
) -> None:
    with pytest.raises(FederatedRoundTransitionError):
        validate_round_transition(current, target)


def test_held_round_must_return_through_evaluation_before_promotion() -> None:
    held = FederatedRoundLifecycle(S.EVALUATING).transition_to(S.HELD)

    with pytest.raises(FederatedRoundTransitionError):
        held.transition_to(S.PROMOTED)

    promoted = held.transition_to(S.EVALUATING).transition_to(S.PROMOTED)
    assert promoted.state is S.PROMOTED
    assert promoted.is_terminal is True


@pytest.mark.parametrize("terminal", FEDERATED_ROUND_TERMINAL_STATES)
def test_promoted_and_aborted_rounds_are_terminal(
    terminal: FederatedRoundState,
) -> None:
    lifecycle = FederatedRoundLifecycle(terminal)

    assert lifecycle.is_terminal is True
    assert lifecycle.allowed_transitions() == ()
    for target in FEDERATED_ROUND_STATES:
        with pytest.raises(FederatedRoundTransitionError):
            lifecycle.transition_to(target)


def test_lifecycle_is_immutable_and_transitions_return_new_values() -> None:
    planned = FederatedRoundLifecycle()
    preflight = planned.transition_to(S.PREFLIGHT)

    assert planned.state is S.PLANNED
    assert preflight.state is S.PREFLIGHT
    with pytest.raises(FrozenInstanceError):
        planned.state = S.ABORTED  # type: ignore[misc]


def test_serialization_is_versioned_deterministic_and_round_trips() -> None:
    lifecycle = FederatedRoundLifecycle(S.EVALUATING)

    assert lifecycle.to_dict() == {
        "schema_version": FEDERATED_ROUND_SCHEMA_VERSION,
        "state": "evaluating",
    }
    assert lifecycle.to_json() == (
        "{\n"
        f'  "schema_version": "{FEDERATED_ROUND_SCHEMA_VERSION}",\n'
        '  "state": "evaluating"\n'
        "}\n"
    )
    assert FederatedRoundLifecycle.from_dict(lifecycle.to_dict()) == lifecycle
    assert FederatedRoundLifecycle.from_json(lifecycle.to_json()) == lifecycle
    assert json.loads(lifecycle.to_json()) == lifecycle.to_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": FEDERATED_ROUND_SCHEMA_VERSION},
        {
            "schema_version": FEDERATED_ROUND_SCHEMA_VERSION,
            "state": "planned",
            "participant_id": "site-7",
        },
        {"schema_version": "v2", "state": "planned"},
        {"schema_version": FEDERATED_ROUND_SCHEMA_VERSION, "state": "unknown"},
        {"schema_version": FEDERATED_ROUND_SCHEMA_VERSION, "state": 1},
    ],
)
def test_deserialization_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(FederatedRoundStateError):
        FederatedRoundLifecycle.from_dict(payload)


def test_errors_and_serialization_do_not_expose_caller_metadata() -> None:
    sentinel = "Patient Jane Roe /srv/charts/123 local_recall=0.42"

    with pytest.raises(FederatedRoundStateError) as error:
        FederatedRoundLifecycle.from_dict(
            {
                "schema_version": FEDERATED_ROUND_SCHEMA_VERSION,
                "state": sentinel,
            }
        )
    assert sentinel not in str(error.value)

    with pytest.raises(FederatedRoundStateError) as error:
        FederatedRoundLifecycle.from_json(sentinel)
    assert sentinel not in str(error.value)

    serialized = FederatedRoundLifecycle().to_json()
    assert set(json.loads(serialized)) == {"schema_version", "state"}
    assert sentinel not in serialized


def test_duplicate_json_fields_are_rejected_instead_of_last_value_winning() -> None:
    payload = (
        '{"schema_version":"openmed.training.federated_round.v1",'
        '"state":"planned","state":"promoted"}'
    )

    with pytest.raises(
        FederatedRoundStateError,
        match="invalid federated round lifecycle JSON",
    ):
        FederatedRoundLifecycle.from_json(payload)


class SchemaVersionSubclass(str):
    """A string subtype that must not be retained in immutable state."""


def test_schema_version_subclasses_are_rejected_by_all_constructors() -> None:
    schema = SchemaVersionSubclass(FEDERATED_ROUND_SCHEMA_VERSION)

    with pytest.raises(FederatedRoundStateError):
        FederatedRoundLifecycle(schema_version=schema)
    with pytest.raises(FederatedRoundStateError):
        FederatedRoundLifecycle.from_dict(
            {"schema_version": schema, "state": "planned"}
        )


def test_plain_strings_cannot_bypass_typed_transition_validation() -> None:
    with pytest.raises(FederatedRoundStateError):
        can_transition_round("planned", S.PREFLIGHT)  # type: ignore[arg-type]
    with pytest.raises(FederatedRoundStateError):
        FederatedRoundLifecycle("planned")  # type: ignore[arg-type]


def test_lifecycle_is_available_through_lazy_training_exports() -> None:
    import openmed.training as training

    assert training.FederatedRoundLifecycle is FederatedRoundLifecycle
    assert training.FederatedRoundState is FederatedRoundState
    assert training.validate_round_transition is validate_round_transition
    assert training.__all__.count("FederatedRoundLifecycle") == 1
