from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.training.federated_round import FederatedRoundState
from openmed.training.federated_status import (
    FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
    FederatedCompletionBand,
    FederatedQuorumStatus,
    FederatedRoundReasonCode,
    FederatedRoundStatus,
    FederatedRoundStatusError,
    build_federated_round_status,
)

S = FederatedRoundState
R = FederatedRoundReasonCode
DIGEST_1 = "sha256:" + "1" * 64
DIGEST_2 = "sha256:" + "2" * 64


GOLDEN_SCENARIOS = {
    "empty": {
        "arguments": {
            "state": S.PLANNED,
            "participant_count": 0,
            "completed_participant_count": 0,
            "required_quorum": 3,
        },
        "payload": {
            "aggregate_digest_refs": [],
            "completed_participant_count": None,
            "completion_band": "not_started",
            "minimum_group_size": 5,
            "participant_count": None,
            "quorum_status": "not_met",
            "reason_code": None,
            "schema_version": FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
            "state": "planned",
        },
        "markdown": """# Federated round status

| Field | Value |
| --- | --- |
| Schema | `openmed.training.federated_status.v1` |
| State | `planned` |
| Quorum | `not_met` |
| Participants | `suppressed (<5)` |
| Completed participants | `suppressed (<5)` |
| Completion | `not_started` |
| Reason | `none` |

## Aggregate digests

None.
""",
    },
    "active": {
        "arguments": {
            "state": S.COLLECTING,
            "participant_count": 8,
            "completed_participant_count": 3,
            "required_quorum": 5,
            "aggregate_digest_refs": [DIGEST_2, DIGEST_1],
        },
        "payload": {
            "aggregate_digest_refs": [DIGEST_1, DIGEST_2],
            "completed_participant_count": None,
            "completion_band": "under_half",
            "minimum_group_size": 5,
            "participant_count": 8,
            "quorum_status": "met",
            "reason_code": None,
            "schema_version": FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
            "state": "collecting",
        },
        "markdown": f"""# Federated round status

| Field | Value |
| --- | --- |
| Schema | `openmed.training.federated_status.v1` |
| State | `collecting` |
| Quorum | `met` |
| Participants | `8` |
| Completed participants | `suppressed (<5)` |
| Completion | `under_half` |
| Reason | `none` |

## Aggregate digests

- `{DIGEST_1}`
- `{DIGEST_2}`
""",
    },
    "held": {
        "arguments": {
            "state": S.HELD,
            "participant_count": 8,
            "completed_participant_count": 8,
            "required_quorum": 5,
            "aggregate_digest_refs": [DIGEST_1],
            "reason_code": R.QUALITY_REVIEW_REQUIRED,
        },
        "payload": {
            "aggregate_digest_refs": [DIGEST_1],
            "completed_participant_count": 8,
            "completion_band": "complete",
            "minimum_group_size": 5,
            "participant_count": 8,
            "quorum_status": "met",
            "reason_code": "quality_review_required",
            "schema_version": FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
            "state": "held",
        },
        "markdown": f"""# Federated round status

| Field | Value |
| --- | --- |
| Schema | `openmed.training.federated_status.v1` |
| State | `held` |
| Quorum | `met` |
| Participants | `8` |
| Completed participants | `8` |
| Completion | `complete` |
| Reason | `quality_review_required` |

## Aggregate digests

- `{DIGEST_1}`
""",
    },
    "promoted": {
        "arguments": {
            "state": S.PROMOTED,
            "participant_count": 8,
            "completed_participant_count": 8,
            "required_quorum": 5,
            "aggregate_digest_refs": [DIGEST_2],
        },
        "payload": {
            "aggregate_digest_refs": [DIGEST_2],
            "completed_participant_count": 8,
            "completion_band": "complete",
            "minimum_group_size": 5,
            "participant_count": 8,
            "quorum_status": "met",
            "reason_code": None,
            "schema_version": FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
            "state": "promoted",
        },
        "markdown": f"""# Federated round status

| Field | Value |
| --- | --- |
| Schema | `openmed.training.federated_status.v1` |
| State | `promoted` |
| Quorum | `met` |
| Participants | `8` |
| Completed participants | `8` |
| Completion | `complete` |
| Reason | `none` |

## Aggregate digests

- `{DIGEST_2}`
""",
    },
    "aborted": {
        "arguments": {
            "state": S.ABORTED,
            "participant_count": 3,
            "completed_participant_count": 2,
            "required_quorum": 5,
            "reason_code": R.QUORUM_NOT_MET,
        },
        "payload": {
            "aggregate_digest_refs": [],
            "completed_participant_count": None,
            "completion_band": "half_or_more",
            "minimum_group_size": 5,
            "participant_count": None,
            "quorum_status": "not_met",
            "reason_code": "quorum_not_met",
            "schema_version": FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
            "state": "aborted",
        },
        "markdown": """# Federated round status

| Field | Value |
| --- | --- |
| Schema | `openmed.training.federated_status.v1` |
| State | `aborted` |
| Quorum | `not_met` |
| Participants | `suppressed (<5)` |
| Completed participants | `suppressed (<5)` |
| Completion | `half_or_more` |
| Reason | `quorum_not_met` |

## Aggregate digests

None.
""",
    },
}


@pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
def test_round_summaries_have_stable_json_and_markdown_golden_outputs(
    scenario: str,
) -> None:
    golden = GOLDEN_SCENARIOS[scenario]
    summary = build_federated_round_status(**golden["arguments"])

    assert summary.to_dict() == golden["payload"]
    assert summary.to_json() == (
        json.dumps(golden["payload"], indent=2, sort_keys=True) + "\n"
    )
    assert summary.to_markdown() == golden["markdown"]


def test_sub_threshold_count_is_suppressed_while_quorum_remains_correct() -> None:
    summary = build_federated_round_status(
        state=S.PREFLIGHT,
        participant_count=3,
        completed_participant_count=0,
        required_quorum=2,
        minimum_group_size=5,
    )

    assert summary.quorum_status is FederatedQuorumStatus.MET
    assert summary.participant_count is None
    assert summary.to_dict()["participant_count"] is None
    assert summary.to_dict()["minimum_group_size"] == 5
    assert "`met`" in summary.to_markdown()
    assert "`3`" not in summary.to_markdown()


def test_builder_has_no_surface_for_participant_or_local_training_metadata() -> None:
    forbidden = {
        "client_id",
        "client_ids",
        "site_name",
        "site_names",
        "patient_count",
        "local_loss",
        "gradients",
        "per_client_metrics",
    }

    assert forbidden.isdisjoint(
        inspect.signature(build_federated_round_status).parameters
    )
    assert forbidden.isdisjoint(inspect.signature(FederatedRoundStatus).parameters)
    with pytest.raises(TypeError) as error:
        build_federated_round_status(
            state=S.PLANNED,
            participant_count=0,
            completed_participant_count=0,
            required_quorum=2,
            **{"site_names": ["North Hospital"]},
        )
    assert "North Hospital" not in str(error.value)


def test_suppressed_values_are_not_retained_by_the_returned_object() -> None:
    summary = build_federated_round_status(
        state=S.COLLECTING,
        participant_count=4,
        completed_participant_count=1,
        required_quorum=2,
        minimum_group_size=5,
    )

    assert summary.participant_count is None
    assert summary.completed_participant_count is None
    assert "participant_count=4" not in repr(summary)
    assert "completed_participant_count=1" not in repr(summary)


@pytest.mark.parametrize(
    ("participants", "completed", "expected"),
    [
        (8, 0, FederatedCompletionBand.NOT_STARTED),
        (8, 1, FederatedCompletionBand.UNDER_HALF),
        (8, 3, FederatedCompletionBand.UNDER_HALF),
        (8, 4, FederatedCompletionBand.HALF_OR_MORE),
        (8, 7, FederatedCompletionBand.HALF_OR_MORE),
        (8, 8, FederatedCompletionBand.COMPLETE),
    ],
)
def test_completion_band_boundaries_are_stable(
    participants: int,
    completed: int,
    expected: FederatedCompletionBand,
) -> None:
    summary = build_federated_round_status(
        state=S.COLLECTING,
        participant_count=participants,
        completed_participant_count=completed,
        required_quorum=2,
    )

    assert summary.completion_band is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"participant_count": -1},
        {"participant_count": True},
        {"completed_participant_count": -1},
        {"completed_participant_count": 4, "participant_count": 3},
        {"required_quorum": 0},
        {"required_quorum": True},
        {"minimum_group_size": 1},
        {"minimum_group_size": True},
        {"state": "planned"},
    ],
)
def test_invalid_aggregate_inputs_fail_closed(overrides: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "state": S.PLANNED,
        "participant_count": 3,
        "completed_participant_count": 1,
        "required_quorum": 2,
    }
    arguments.update(overrides)

    with pytest.raises(FederatedRoundStatusError):
        build_federated_round_status(**arguments)


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (S.HELD, None),
        (S.HELD, R.ROUND_ERROR),
        (S.ABORTED, None),
        (S.ABORTED, R.QUALITY_REVIEW_REQUIRED),
        (S.COLLECTING, R.QUALITY_REVIEW_REQUIRED),
    ],
)
def test_reason_codes_are_required_and_state_specific(
    state: FederatedRoundState,
    reason: FederatedRoundReasonCode | None,
) -> None:
    with pytest.raises(FederatedRoundStatusError):
        build_federated_round_status(
            state=state,
            participant_count=8,
            completed_participant_count=4,
            required_quorum=5,
            reason_code=reason,
        )


def test_digest_references_are_validated_sorted_and_unique() -> None:
    summary = build_federated_round_status(
        state=S.EVALUATING,
        participant_count=8,
        completed_participant_count=8,
        required_quorum=5,
        aggregate_digest_refs=[DIGEST_2, DIGEST_1],
    )
    assert summary.aggregate_digest_refs == (DIGEST_1, DIGEST_2)

    for invalid in (["not-a-digest"], [DIGEST_1, DIGEST_1], "secret"):
        with pytest.raises(FederatedRoundStatusError):
            build_federated_round_status(
                state=S.EVALUATING,
                participant_count=8,
                completed_participant_count=8,
                required_quorum=5,
                aggregate_digest_refs=invalid,
            )


def test_errors_do_not_echo_rejected_digest_values() -> None:
    sentinel = "Patient Jane Roe /srv/site-a local_loss=0.42"

    with pytest.raises(FederatedRoundStatusError) as error:
        build_federated_round_status(
            state=S.COLLECTING,
            participant_count=8,
            completed_participant_count=3,
            required_quorum=5,
            aggregate_digest_refs=[sentinel],
        )
    assert sentinel not in str(error.value)


def test_status_is_immutable() -> None:
    summary = build_federated_round_status(
        state=S.PLANNED,
        participant_count=0,
        completed_participant_count=0,
        required_quorum=2,
    )

    with pytest.raises(FrozenInstanceError):
        summary.state = S.ABORTED  # type: ignore[misc]


def test_direct_status_construction_cannot_forge_derived_fields() -> None:
    with pytest.raises(
        FederatedRoundStatusError,
        match="must be built from aggregate inputs",
    ):
        FederatedRoundStatus(
            state=S.COLLECTING,
            quorum_status=FederatedQuorumStatus.NOT_MET,
            participant_count=8,
            completed_participant_count=8,
            minimum_group_size=5,
            completion_band=FederatedCompletionBand.NOT_STARTED,
        )


def test_status_api_is_available_through_lazy_training_exports() -> None:
    import openmed.training as training

    assert training.build_federated_round_status is build_federated_round_status
    assert training.FederatedRoundReasonCode is FederatedRoundReasonCode
    assert training.__all__.count("FederatedRoundStatus") == 1
