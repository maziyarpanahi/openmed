"""Focused contracts for the synthetic metadata-only federated walkthrough."""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest

from examples import federated_round_walkthrough as example

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/training"


def test_validation_rejects_only_the_unknown_parameter_and_groups_accepted_updates(
    monkeypatch,
):
    validate = example.FederatedUpdateMetadata.from_dict
    fingerprint = example.fingerprint_update_schema
    submissions = []
    fingerprinted = []

    def record_validation(cls, payload, *, policy):
        submissions.append((deepcopy(payload), policy))
        return validate(payload, policy=policy)

    def record_fingerprint(metadata):
        fingerprinted.append(metadata)
        return fingerprint(metadata)

    monkeypatch.setattr(
        example.FederatedUpdateMetadata,
        "from_dict",
        classmethod(record_validation),
    )
    monkeypatch.setattr(example, "fingerprint_update_schema", record_fingerprint)
    report, status = example.run_example()

    assert len(submissions) == 6
    policy = submissions[0][1]
    assert all(submitted_policy is policy for _, submitted_policy in submissions)
    assert [parameter.name for parameter in policy.parameters] == [
        "adapter.lora_A.weight",
        "adapter.lora_B.weight",
    ]
    assert policy.require_clipped is True
    assert policy.max_total_elements == 14
    rejected = submissions[-1][0]
    with pytest.raises(
        example.FederatedUpdateMetadataError,
        match="^parameters do not match policy$",
    ):
        validate(rejected, policy=policy)

    corrected = deepcopy(rejected)
    corrected["parameters"][1]["name"] = "adapter.lora_B.weight"
    assert corrected == {
        **submissions[0][0],
        "update_digest": rejected["update_digest"],
    }
    assert validate(corrected, policy=policy).total_elements == 14

    assert len(fingerprinted) == 5
    assert {metadata.update_digest for metadata in fingerprinted} == {
        payload["update_digest"] for payload, _ in submissions[:-1]
    }
    assert rejected["update_digest"] not in {
        metadata.update_digest for metadata in fingerprinted
    }
    assert len({fingerprint(metadata) for metadata in fingerprinted}) == 1
    assert submissions[0][0]["parameters"] == list(
        reversed(submissions[1][0]["parameters"])
    )
    assert report["updates"] == {
        "accepted_count": 5,
        "rejected_count": 1,
        "rejection_code": "unknown_parameter",
    }
    assert len(report["schema_groups"]) == 1
    assert report["schema_groups"][0]["update_count"] == 5
    assert status.participant_count == status.completed_participant_count == 5


def test_incoming_parameters_cannot_authorize_themselves(monkeypatch):
    build_payload = example._update_payload

    def unknown_payload(*args, **kwargs):
        payload = build_payload(*args, **kwargs)
        payload["parameters"][0]["name"] = "adapter.unknown.weight"
        return payload

    monkeypatch.setattr(example, "_update_payload", unknown_payload)
    with pytest.raises(
        example.FederatedUpdateMetadataError,
        match="^parameters do not match policy$",
    ):
        example.run_example()


def test_bypassing_rejection_causes_explicit_failure(monkeypatch):
    validate = example.FederatedUpdateMetadata.from_dict
    accepted = None

    def bypass_rejection(cls, payload, *, policy):
        nonlocal accepted
        if any(
            parameter["name"] == "adapter.unknown.weight"
            for parameter in payload["parameters"]
        ):
            return accepted
        accepted = validate(payload, policy=policy)
        return accepted

    monkeypatch.setattr(
        example.FederatedUpdateMetadata, "from_dict", classmethod(bypass_rejection)
    )
    with pytest.raises(AssertionError, match="^unknown_parameter_not_rejected$"):
        example.run_example()


def test_schedule_lifecycle_and_status_are_consistent():
    report, status = example.run_example()
    schedule = example.FederatedRoundSchedule.from_dict(report["schedule"])
    for boundary, timestamp in report["schedule"]["boundaries"].items():
        phase = schedule.active_phase_at(
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        )
        assert report["schedule_phases_at_boundaries"][boundary] == (
            phase.value if phase is not None else None
        )
    assert report["schedule_phases_at_boundaries"]["finishes_at"] is None

    assert report["lifecycle_states"] == [
        "planned",
        "preflight",
        "collecting",
        "aggregating",
        "evaluating",
        "held",
    ]
    lifecycle = example.FederatedRoundLifecycle()
    for state in report["lifecycle_states"][1:]:
        lifecycle = lifecycle.transition_to(example.FederatedRoundState(state))
    assert lifecycle.state is status.state is example.FederatedRoundState.HELD
    assert (
        status.reason_code is example.FederatedRoundReasonCode.QUALITY_REVIEW_REQUIRED
    )
    assert status.quorum_status.value == "met"
    assert status.completion_band.value == "complete"
    assert report["status"] == status.to_dict()


def test_metric_thresholds_discard_values_and_uncertainty_before_serialization(
    monkeypatch,
    capsys,
):
    build_metric = example.build_federated_metric_envelope
    inputs = []
    envelopes = []

    def record_metric(**kwargs):
        inputs.append(kwargs)
        envelope = build_metric(**kwargs)
        envelopes.append(envelope)
        return envelope

    monkeypatch.setattr(example, "build_federated_metric_envelope", record_metric)
    report, _ = example.run_example()
    suppressed, released = envelopes
    assert inputs[0]["participant_count"] == 4
    assert inputs[1]["participant_count"] == 5
    for field in (
        "aggregate_value",
        "uncertainty_lower_bound",
        "uncertainty_upper_bound",
        "confidence_level",
    ):
        assert inputs[0][field] is not None
        assert getattr(suppressed, field) is None
        assert report["metrics"][0][field] is None
    assert suppressed.participant_count_band.value == "suppressed"
    assert suppressed.uncertainty_method.value == "suppressed"
    assert released.aggregate_value == 0.8
    assert released.participant_count_band.value == "minimum_to_under_double"
    assert "participant_count" not in suppressed.to_dict()
    assert "participant_count" not in released.to_dict()

    for output_format in ("json", "markdown"):
        assert example.main(["--format", output_format]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        for value in (0.73, 0.61, 0.89, 0.95):
            assert str(value) not in captured.out
        for forbidden in (
            "adapter.unknown.weight",
            "parameters",
            "update_digest",
            "model_digest",
            "client_id",
            "site_name",
            "tensor",
            "gradient",
            "payload",
        ):
            assert forbidden not in captured.out


def test_serialization_matches_reviewed_goldens_and_default_is_json(capsys):
    report, status = example.run_example()
    expected_json = (FIXTURES / "federated_round_walkthrough.json").read_bytes()
    expected_markdown = (FIXTURES / "federated_round_walkthrough.md").read_bytes()
    assert (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8") == expected_json
    assert status.to_markdown().encode("utf-8") == expected_markdown
    assert example.main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.encode("utf-8") == expected_json
