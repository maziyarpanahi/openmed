"""Offline tests for the anonymous federated update batch preflight."""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import replace

import pytest

from openmed.training import federated_update_batch as batch_module
from openmed.training.federated_schema_fingerprint import fingerprint_update_schema
from openmed.training.federated_update_batch import (
    DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE,
    DEFAULT_MAX_BATCH_UPDATES,
    FEDERATED_UPDATE_BATCH_REASON_CODES,
    FEDERATED_UPDATE_BATCH_SCHEMA_VERSION,
    MAX_BATCH_UPDATES,
    FederatedUpdateBatchError,
    FederatedUpdateBatchFinding,
    FederatedUpdateBatchPolicy,
    FederatedUpdateBatchReasonCode,
    FederatedUpdateBatchReport,
    FederatedUpdateBatchStatus,
    FederatedUpdateGroup,
    FederatedUpdateOutcome,
    check_federated_update_batch,
)
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

MODEL = "sha256:" + "a" * 64
OTHER_MODEL = "sha256:" + "b" * 64
SENTINEL = "SYNTHETIC_PRIVATE_SENTINEL_3056"
_GOLDEN_REPORT_SHA256 = (
    "98d87ba306d97a0ef963b5ab3da340594c5cd20848129e31b21204dbd5db5c0e"
)

_PARAMETERS = (
    FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
    FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
)
_UPDATE_POLICY = FederatedUpdatePolicy(
    model_digest=MODEL, parameters=_PARAMETERS, max_total_elements=14
)
_OTHER_POLICY = FederatedUpdatePolicy(
    model_digest=OTHER_MODEL, parameters=_PARAMETERS, max_total_elements=14
)


def _digest(index: int) -> str:
    return "sha256:" + f"{index:064x}"


def _payload(index: int, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": MODEL,
        "adapter_format": "dense",
        "parameters": [
            {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"},
            {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
        ],
        "total_elements": 14,
        "update_digest": _digest(index),
        "clipped": True,
    }
    payload.update(overrides)
    return payload


def _batch_policy(**overrides: object) -> FederatedUpdateBatchPolicy:
    values: dict[str, object] = {
        "update_policy": _UPDATE_POLICY,
        "minimum_group_size": 2,
    }
    values.update(overrides)
    return FederatedUpdateBatchPolicy(**values)  # type: ignore[arg-type]


def _reasons(report: FederatedUpdateBatchReport) -> list[str]:
    return [str(finding.reason) for finding in report.findings]


def _cannot_reach_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("network access attempted")


# --- constants --------------------------------------------------------------


def test_schema_version_constant() -> None:
    assert (
        FEDERATED_UPDATE_BATCH_SCHEMA_VERSION
        == "openmed.training.federated_update_batch.v1"
    )


def test_reason_codes_are_a_sorted_closed_set() -> None:
    values = [reason.value for reason in FEDERATED_UPDATE_BATCH_REASON_CODES]
    assert values == sorted(values)
    assert len(set(values)) == len(values) == 8
    assert all(
        type(reason) is FederatedUpdateBatchReasonCode
        for reason in FEDERATED_UPDATE_BATCH_REASON_CODES
    )
    assert {str(reason) for reason in FEDERATED_UPDATE_BATCH_REASON_CODES} == set(
        values
    )


def test_batch_caps_and_shared_floor() -> None:
    assert DEFAULT_MAX_BATCH_UPDATES == 64
    assert MAX_BATCH_UPDATES == 512
    assert DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE == 5
    assert (
        DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE
        == batch_module.DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE
    )


def test_statuses_are_string_enums() -> None:
    assert {str(status) for status in FederatedUpdateBatchStatus} == {
        "accepted",
        "rejected",
        "skipped",
    }


# --- accepted batches -------------------------------------------------------


def test_homogeneous_batch_accepted() -> None:
    report = check_federated_update_batch(
        [_payload(index) for index in range(5)],
        policy=_batch_policy(minimum_group_size=5),
    )
    assert report.received == 5
    assert report.accepted == 5
    assert report.rejected == 0
    assert report.suppressed_groups == 0
    assert report.limit_exceeded is False
    assert report.ok is True
    assert _reasons(report) == ["batch_digest_withheld", "update_accepted"]
    assert [finding.count for finding in report.findings] == [5, 5]
    assert [str(finding.status) for finding in report.findings] == [
        "skipped",
        "accepted",
    ]
    assert len(report.groups) == 1
    assert report.groups[0].accepted == 5
    assert report.groups[0].suppressed is False
    assert all(
        outcome.status is FederatedUpdateBatchStatus.ACCEPTED
        for outcome in report.outcomes
    )
    assert [outcome.index for outcome in report.outcomes] == list(range(5))


def test_group_digests_are_withheld_by_default() -> None:
    report = check_federated_update_batch(
        [_payload(index) for index in range(3)], policy=_batch_policy()
    )
    assert all(outcome.update_digest is None for outcome in report.outcomes)
    assert report.groups[0].update_digests == ()
    assert "batch_digest_withheld" in _reasons(report)


def test_digests_disclosed_when_permitted() -> None:
    report = check_federated_update_batch(
        [_payload(index) for index in range(3)],
        policy=_batch_policy(disclose_digests=True),
    )
    assert [outcome.update_digest for outcome in report.outcomes] == [
        _digest(0),
        _digest(1),
        _digest(2),
    ]
    assert report.groups[0].update_digests == (_digest(0), _digest(1), _digest(2))
    assert _reasons(report) == ["update_accepted"]


def test_group_below_floor_is_suppressed() -> None:
    report = check_federated_update_batch(
        [_payload(index) for index in range(3)],
        policy=_batch_policy(minimum_group_size=4, disclose_digests=True),
    )
    assert report.groups[0].suppressed is True
    assert report.groups[0].accepted == 3
    assert report.suppressed_groups == 1
    assert [finding.count for finding in report.findings] == [3, 1, 3]


def test_suppressed_group_withholds_disclosed_digests() -> None:
    report = check_federated_update_batch(
        [_payload(index) for index in range(3)],
        policy=_batch_policy(minimum_group_size=5, disclose_digests=True),
    )
    assert report.groups[0].suppressed is True
    assert report.groups[0].update_digests == ()
    assert all(outcome.update_digest is None for outcome in report.outcomes)
    assert all(_digest(index) not in report.to_json() for index in range(3))
    assert _reasons(report) == [
        "batch_digest_withheld",
        "batch_group_suppressed",
        "update_accepted",
    ]


def test_fingerprint_matches_policy_fingerprint() -> None:
    metadata = FederatedUpdateMetadata.from_dict(_payload(0), policy=_UPDATE_POLICY)
    expected = fingerprint_update_schema(metadata)
    report = check_federated_update_batch(
        [_payload(0), _payload(1)],
        policy=_batch_policy(expected_fingerprint=expected),
    )
    assert report.accepted == 2
    assert all(outcome.fingerprint == expected for outcome in report.outcomes)


# --- rejected and skipped envelopes ----------------------------------------


def test_mixed_batch_reports_every_envelope() -> None:
    report = check_federated_update_batch(
        [
            _payload(0),
            _payload(1),
            _payload(0),
            {"schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION},
            "{not json",
            42,
        ],
        policy=_batch_policy(),
    )
    assert report.received == 6
    assert report.accepted == 2
    assert report.rejected == 4
    assert [str(outcome.reason) for outcome in report.outcomes] == [
        "update_accepted",
        "update_accepted",
        "update_duplicate",
        "update_schema_invalid",
        "update_schema_invalid",
        "update_schema_invalid",
    ]
    assert [(str(finding.reason), finding.count) for finding in report.findings] == [
        ("batch_digest_withheld", 2),
        ("update_accepted", 2),
        ("update_duplicate", 1),
        ("update_schema_invalid", 3),
    ]
    assert report.ok is False


def test_invalid_envelope_does_not_stop_the_batch() -> None:
    report = check_federated_update_batch(
        [_payload(0), "not-json", _payload(1)], policy=_batch_policy()
    )
    assert report.accepted == 2
    assert str(report.outcomes[1].reason) == "update_schema_invalid"
    assert str(report.outcomes[2].reason) == "update_accepted"


def test_expected_fingerprint_mismatch_is_rejected() -> None:
    other = fingerprint_update_schema(
        FederatedUpdateMetadata.from_dict(
            _payload(0, model_digest=OTHER_MODEL), policy=_OTHER_POLICY
        )
    )
    report = check_federated_update_batch(
        [_payload(0), _payload(1)],
        policy=_batch_policy(expected_fingerprint=other),
    )
    assert report.accepted == 0
    assert report.rejected == 2
    assert _reasons(report) == ["update_fingerprint_mismatch"]
    assert all(outcome.fingerprint is None for outcome in report.outcomes)


def test_empty_batch_is_reported() -> None:
    report = check_federated_update_batch([], policy=_batch_policy())
    assert report.received == 0
    assert report.ok is True
    assert _reasons(report) == ["batch_empty"]
    assert report.findings[0].count == 1
    assert report.outcomes == ()
    assert report.groups == ()


def test_over_limit_batch_short_circuits() -> None:
    report = check_federated_update_batch(
        [_payload(index) for index in range(3)], policy=_batch_policy(max_updates=2)
    )
    assert report.limit_exceeded is True
    assert report.ok is False
    assert report.received == 3
    assert report.accepted == 0
    assert report.rejected == 0
    assert report.outcomes == ()
    assert report.groups == ()
    assert report.findings[0].reason is (
        FederatedUpdateBatchReasonCode.BATCH_LIMIT_EXCEEDED
    )
    assert report.findings[0].status is FederatedUpdateBatchStatus.SKIPPED


def test_tuple_payloads_are_accepted() -> None:
    report = check_federated_update_batch(
        (_payload(0), _payload(1)), policy=_batch_policy()
    )
    assert report.accepted == 2


# --- determinism and serialization -----------------------------------------


def test_report_json_is_byte_stable() -> None:
    first = check_federated_update_batch(
        [_payload(index) for index in range(4)], policy=_batch_policy()
    )
    second = check_federated_update_batch(
        [_payload(index) for index in range(4)], policy=_batch_policy()
    )
    assert first.to_json() == second.to_json()
    assert first.to_json().endswith("}\n")
    assert json.loads(first.to_json())["schema_version"] == (
        FEDERATED_UPDATE_BATCH_SCHEMA_VERSION
    )


def test_golden_report_digest() -> None:
    report = check_federated_update_batch(
        [_payload(0), _payload(1)], policy=_batch_policy()
    )
    digest = hashlib.sha256(report.to_json().encode("utf-8")).hexdigest()
    assert digest == _GOLDEN_REPORT_SHA256


def test_report_key_order() -> None:
    report = check_federated_update_batch([_payload(0)], policy=_batch_policy())
    assert list(report.to_dict()) == [
        "schema_version",
        "received",
        "accepted",
        "rejected",
        "suppressed_groups",
        "limit_exceeded",
        "findings",
        "groups",
        "outcomes",
    ]
    assert list(report.findings[0].to_dict()) == ["count", "reason", "status"]
    assert list(report.groups[0].to_dict()) == [
        "accepted",
        "fingerprint",
        "suppressed",
        "update_digests",
    ]
    assert list(report.outcomes[0].to_dict()) == [
        "fingerprint",
        "index",
        "reason",
        "status",
        "update_digest",
    ]


# --- privacy ----------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    FORBIDDEN_FIELD_CASES,
    ids=[case.reason_code for case in FORBIDDEN_FIELD_CASES],
)
def test_forbidden_field_is_rejected_without_echo(case) -> None:
    injected = _payload(0, **{case.field: case.value})
    report = check_federated_update_batch(
        [injected, _payload(1)], policy=_batch_policy()
    )
    text = report.to_json()
    assert report.accepted == 1
    assert str(report.outcomes[0].reason) == "update_schema_invalid"
    assert case.field not in text
    if case.marker is not None:
        assert case.marker not in text


def test_nested_forbidden_field_is_rejected_without_echo() -> None:
    payload = _payload(0)
    payload["parameters"] = [
        {
            "name": "adapter.lora_A.weight",
            "shape": [2, 3],
            "dtype": "float32",
            "path": SENTINEL,
        },
        {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
    ]
    report = check_federated_update_batch([payload], policy=_batch_policy())
    assert str(report.outcomes[0].reason) == "update_schema_invalid"
    assert SENTINEL not in report.to_json()


def test_report_retains_no_submitted_values(caplog, capsys) -> None:
    report = check_federated_update_batch(
        [_payload(0, path=SENTINEL)], policy=_batch_policy()
    )
    text = report.to_json()
    assert SENTINEL not in text
    assert "adapter.lora_A.weight" not in text
    assert MODEL not in text
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


def test_batch_check_is_offline(monkeypatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _cannot_reach_network)
    monkeypatch.setattr(socket, "create_connection", _cannot_reach_network)
    report = check_federated_update_batch(
        [_payload(0), _payload(1)], policy=_batch_policy()
    )
    assert report.accepted == 2


# --- argument validation ----------------------------------------------------


@pytest.mark.parametrize("payloads", [{}, "batch", 3, None, frozenset({1}), 1.5])
def test_checker_rejects_non_sequence_payloads(payloads) -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="payloads must be a list or tuple"
    ):
        check_federated_update_batch(payloads, policy=_batch_policy())


@pytest.mark.parametrize("policy", [None, object(), _UPDATE_POLICY, "batch"])
def test_checker_rejects_non_batch_policy(policy) -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="batch policy must be a FederatedUpdateBatchPolicy",
    ):
        check_federated_update_batch([], policy=policy)


@pytest.mark.parametrize("bad", ["sha256:zz", "abc", 7, True, "", None])
def test_policy_rejects_invalid_expected_fingerprint(bad) -> None:
    if bad is None:
        assert _batch_policy(expected_fingerprint=None).expected_fingerprint is None
        return
    with pytest.raises(
        FederatedUpdateBatchError,
        match="expected_fingerprint must be a sha256 digest reference",
    ):
        _batch_policy(expected_fingerprint=bad)


@pytest.mark.parametrize("bad", [0, -1, 513, True, 1.5, "3", None])
def test_policy_rejects_invalid_max_updates(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="max_updates must be an integer between 1 and 512",
    ):
        _batch_policy(max_updates=bad)


@pytest.mark.parametrize("bad", [0, 1, 513, True, 2.0, "5", None])
def test_policy_rejects_invalid_minimum_group_size(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="minimum_group_size must be an integer between 2 and 512",
    ):
        _batch_policy(minimum_group_size=bad)


@pytest.mark.parametrize("bad", ["yes", 1, 0, None, []])
def test_policy_rejects_invalid_disclose_digests(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="disclose_digests must be a boolean"
    ):
        _batch_policy(disclose_digests=bad)


def test_policy_accepts_boundary_values() -> None:
    policy = _batch_policy(
        max_updates=MAX_BATCH_UPDATES, minimum_group_size=MAX_BATCH_UPDATES
    )
    assert policy.max_updates == MAX_BATCH_UPDATES
    assert policy.minimum_group_size == MAX_BATCH_UPDATES
    assert _batch_policy(max_updates=1, minimum_group_size=2).max_updates == 1


def test_policy_rejects_non_update_policy() -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="batch policy must be a FederatedUpdateBatchPolicy",
    ):
        FederatedUpdateBatchPolicy(update_policy=object())  # type: ignore[arg-type]


# --- record invariants ------------------------------------------------------


def _fingerprint() -> str:
    return fingerprint_update_schema(
        FederatedUpdateMetadata.from_dict(_payload(0), policy=_UPDATE_POLICY)
    )


def _accepted_outcome(index: int = 0) -> FederatedUpdateOutcome:
    return FederatedUpdateOutcome(
        index,
        FederatedUpdateBatchStatus.ACCEPTED,
        FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
        fingerprint=_fingerprint(),
    )


def test_outcome_rejects_reason_status_mismatch() -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="outcome reason does not match outcome status"
    ):
        FederatedUpdateOutcome(
            0,
            FederatedUpdateBatchStatus.ACCEPTED,
            FederatedUpdateBatchReasonCode.UPDATE_DUPLICATE,
        )


def test_outcome_rejects_digest_on_rejected_verdict() -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="rejected outcomes must not carry digests"
    ):
        FederatedUpdateOutcome(
            0,
            FederatedUpdateBatchStatus.REJECTED,
            FederatedUpdateBatchReasonCode.UPDATE_SCHEMA_INVALID,
            fingerprint=_fingerprint(),
        )


@pytest.mark.parametrize("bad", ["sha256:zz", "abc", 7, True, ""])
def test_outcome_rejects_invalid_digests(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="outcome digests must be sha256 digest references",
    ):
        FederatedUpdateOutcome(
            0,
            FederatedUpdateBatchStatus.ACCEPTED,
            FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
            fingerprint=bad,
        )


@pytest.mark.parametrize("bad", [-1, True, 1.5, "0", None])
def test_outcome_rejects_invalid_index(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="outcome index must be a non-negative integer",
    ):
        FederatedUpdateOutcome(
            bad,
            FederatedUpdateBatchStatus.ACCEPTED,
            FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
        )


def test_group_rejects_suppressed_digests() -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="suppressed groups must not carry digests"
    ):
        FederatedUpdateGroup(_fingerprint(), 2, True, (_digest(0), _digest(1)))


def test_group_rejects_digest_count_mismatch() -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="group digest count does not match the accepted count",
    ):
        FederatedUpdateGroup(_fingerprint(), 3, False, (_digest(0),))


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "2", None])
def test_group_rejects_invalid_counts(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError,
        match="group accepted count must be a positive integer",
    ):
        FederatedUpdateGroup(_fingerprint(), bad, False)


def test_finding_rejects_status_mismatch() -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="finding reason does not match finding status"
    ):
        FederatedUpdateBatchFinding(
            FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
            FederatedUpdateBatchStatus.REJECTED,
            1,
        )


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "1", None])
def test_finding_rejects_invalid_counts(bad) -> None:
    with pytest.raises(
        FederatedUpdateBatchError, match="finding count must be a positive integer"
    ):
        FederatedUpdateBatchFinding(
            FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
            FederatedUpdateBatchStatus.ACCEPTED,
            bad,
        )


def test_report_rejects_unreconciled_counts() -> None:
    finding = FederatedUpdateBatchFinding(
        FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
        FederatedUpdateBatchStatus.ACCEPTED,
        1,
    )
    with pytest.raises(
        FederatedUpdateBatchError, match="batch report counts do not reconcile"
    ):
        FederatedUpdateBatchReport(3, 1, 1, 0, False, (finding,), (), ())


def test_report_rejects_unordered_findings() -> None:
    findings = (
        FederatedUpdateBatchFinding(
            FederatedUpdateBatchReasonCode.UPDATE_SCHEMA_INVALID,
            FederatedUpdateBatchStatus.REJECTED,
            1,
        ),
        FederatedUpdateBatchFinding(
            FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
            FederatedUpdateBatchStatus.ACCEPTED,
            1,
        ),
    )
    with pytest.raises(
        FederatedUpdateBatchError,
        match="batch report findings must be ordered reason codes",
    ):
        FederatedUpdateBatchReport(2, 1, 1, 0, False, findings, (), ())


def test_report_rejects_unordered_groups() -> None:
    finding = FederatedUpdateBatchFinding(
        FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
        FederatedUpdateBatchStatus.ACCEPTED,
        2,
    )
    groups = (
        FederatedUpdateGroup(_digest(2), 1, False),
        FederatedUpdateGroup(_digest(1), 1, False),
    )
    with pytest.raises(
        FederatedUpdateBatchError,
        match="batch report groups must be ordered fingerprints",
    ):
        FederatedUpdateBatchReport(2, 2, 0, 0, False, (finding,), groups, ())


def test_report_rejects_suppression_mismatch() -> None:
    finding = FederatedUpdateBatchFinding(
        FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
        FederatedUpdateBatchStatus.ACCEPTED,
        1,
    )
    with pytest.raises(
        FederatedUpdateBatchError,
        match="batch report suppression count does not reconcile",
    ):
        FederatedUpdateBatchReport(
            1,
            1,
            0,
            1,
            False,
            (finding,),
            (FederatedUpdateGroup(_fingerprint(), 1, False),),
            (_accepted_outcome(),),
        )


def test_report_rejects_outcome_mismatch() -> None:
    finding = FederatedUpdateBatchFinding(
        FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
        FederatedUpdateBatchStatus.ACCEPTED,
        1,
    )
    with pytest.raises(
        FederatedUpdateBatchError,
        match="batch report outcomes do not reconcile with the received count",
    ):
        FederatedUpdateBatchReport(
            1,
            1,
            0,
            0,
            False,
            (finding,),
            (FederatedUpdateGroup(_fingerprint(), 1, False),),
            (),
        )


def test_report_rejects_unsupported_schema() -> None:
    report = check_federated_update_batch([_payload(0)], policy=_batch_policy())
    with pytest.raises(
        FederatedUpdateBatchError, match="unsupported batch report schema"
    ):
        replace(report, schema_version="openmed.training.federated_update_batch.v2")


def test_report_ok_requires_bounded_and_rejection_free() -> None:
    accepted = check_federated_update_batch([_payload(0)], policy=_batch_policy())
    rejected = check_federated_update_batch([""], policy=_batch_policy())
    over_limit = check_federated_update_batch(
        [_payload(0), _payload(1)], policy=_batch_policy(max_updates=1)
    )
    assert accepted.ok is True
    assert rejected.ok is False
    assert over_limit.ok is False


# --- exports ----------------------------------------------------------------


def test_module_all_is_complete() -> None:
    assert frozenset(batch_module.__all__) == frozenset(
        {
            "DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE",
            "DEFAULT_MAX_BATCH_UPDATES",
            "FEDERATED_UPDATE_BATCH_REASON_CODES",
            "FEDERATED_UPDATE_BATCH_SCHEMA_VERSION",
            "MAX_BATCH_UPDATES",
            "FederatedUpdateBatchError",
            "FederatedUpdateBatchFinding",
            "FederatedUpdateBatchPolicy",
            "FederatedUpdateBatchReasonCode",
            "FederatedUpdateBatchReport",
            "FederatedUpdateBatchStatus",
            "FederatedUpdateGroup",
            "FederatedUpdateOutcome",
            "check_federated_update_batch",
        }
    )
    for name in batch_module.__all__:
        assert hasattr(batch_module, name)


def test_lazy_training_exports_resolve() -> None:
    import openmed.training as training

    for name in batch_module.__all__:
        assert name in training.__all__
        assert getattr(training, name) is getattr(batch_module, name)
