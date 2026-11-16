"""Tests for the value-free OS-enforced no-egress attestation."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.core.attestation import (
    NO_EGRESS_EXIT_CODES,
    NO_EGRESS_NEGATIVE_CONTROLS,
    NO_EGRESS_POSITIVE_CONTROLS,
    NO_EGRESS_WORKFLOW_CASES,
    NoEgressAttestation,
    NoEgressAttestationError,
    build_no_egress_attestation,
    load_no_egress_attestation_schema,
    validate_no_egress_attestation,
)
from openmed.core.audit import stable_hash

_DIGEST = "sha256:" + "a" * 64
_FIXTURES = {
    "service_workflow_fixture": "sha256:" + "1" * 64,
    "no_egress_gate_runner": "sha256:" + "2" * 64,
}
_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_ISOLATED_INTERFACES = {"total": 10, "loopback_up": 1, "non_loopback_up": 0}
_NO_ROUTES = {"ipv4_non_loopback": 0, "ipv6_non_loopback": 0}
_CONTROLLED_STRING = re.compile(
    r"^(?:[a-z][a-z0-9_.]{0,63}|sha256:[0-9a-f]{64}|[0-9a-f]{40}|"
    r"[0-9][0-9A-Za-z.+-]{0,63}|[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z)$"
)


def _negatives(**overrides: tuple[str, str]) -> list[dict[str, str]]:
    rows = []
    for case_id in NO_EGRESS_NEGATIVE_CONTROLS:
        status, reason = overrides.get(case_id, ("blocked", "network_unreachable"))
        rows.append({"id": case_id, "status": status, "reason_code": reason})
    return rows


def _positives(**overrides: tuple[str, str]) -> list[dict[str, str]]:
    rows = []
    for case_id in NO_EGRESS_POSITIVE_CONTROLS:
        status, reason = overrides.get(case_id, ("reachable", "loopback_reachable"))
        rows.append({"id": case_id, "status": status, "reason_code": reason})
    return rows


def _workflow(**overrides: tuple[str, str]) -> list[dict[str, Any]]:
    rows = []
    for case_id in NO_EGRESS_WORKFLOW_CASES:
        status, reason = overrides.get(case_id, ("passed", "ok"))
        rows.append(
            {
                "id": case_id,
                "status": status,
                "reason_code": reason,
                "result_digest": _DIGEST if status == "passed" else None,
            }
        )
    return rows


def _build(**overrides: Any) -> NoEgressAttestation:
    arguments: dict[str, Any] = {
        "host_kind": "linux",
        "requested_boundary": "netns",
        "boundary_kind": "netns",
        "kernel": "linux",
        "interfaces": _ISOLATED_INTERFACES,
        "routes": _NO_ROUTES,
        "negative_controls": _negatives(),
        "positive_controls": _positives(),
        "workflow": _workflow(),
        "fixtures": _FIXTURES,
        "commit": _COMMIT,
        "openmed_version": "4.0.0",
        "generated_at": "2026-11-07T12:00:00Z",
    }
    arguments.update(overrides)
    return build_no_egress_attestation(**arguments)


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _string_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _string_leaves(item)]
    return []


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    repro = copy.deepcopy(payload)
    for key in ("generated_at", "repro_hash", "integrity_hash"):
        repro.pop(key, None)
    payload["repro_hash"] = stable_hash(repro)
    payload.pop("integrity_hash", None)
    payload["integrity_hash"] = stable_hash(payload)
    return payload


def test_bundled_schema_is_a_valid_draft_2020_12_schema() -> None:
    schema = load_no_egress_attestation_schema()

    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["attestation_type"] == {"const": "openmed.no_egress"}


def test_passing_run_is_schema_valid_and_exits_zero() -> None:
    attestation = _build()
    payload = attestation.to_dict()

    Draft202012Validator(load_no_egress_attestation_schema()).validate(payload)
    assert attestation.passed
    assert attestation.status == "passed"
    assert attestation.reason_code == "all_controls_blocked"
    assert attestation.exit_code == 0
    assert payload["boundary"] == {
        "requested": "netns",
        "kind": "netns",
        "mechanism": "linux_user_network_namespace",
        "kernel": "linux",
        "isolated": True,
        "interfaces": _ISOLATED_INTERFACES,
        "routes": _NO_ROUTES,
    }
    assert payload["counts"]["negative_controls"] == {
        "total": len(NO_EGRESS_NEGATIVE_CONTROLS),
        "blocked": len(NO_EGRESS_NEGATIVE_CONTROLS),
        "not_blocked": 0,
        "inconclusive": 0,
        "not_run": 0,
    }
    assert {row["family"] for row in payload["negative_controls"]} == {
        "ipv4",
        "ipv6",
        "any",
    }
    assert {row["transport"] for row in payload["negative_controls"]} == {
        "tcp",
        "udp",
        "dns",
    }
    assert {row["executor"] for row in payload["negative_controls"]} >= {
        "subprocess",
        "native_libc",
    }
    assert validate_no_egress_attestation(payload).passed


def test_attestation_is_value_free() -> None:
    payload = _build().to_dict()

    leaves = _string_leaves(payload)
    assert leaves
    assert all(_CONTROLLED_STRING.match(leaf) for leaf in leaves), [
        leaf for leaf in leaves if not _CONTROLLED_STRING.match(leaf)
    ]
    serialized = json.dumps(payload)
    for forbidden in ("192.0.2", "2001:db8", ".invalid", "127.0.0.1", "/tmp", "http"):
        assert forbidden not in serialized


def test_attestation_is_deterministic_apart_from_declared_volatile_fields() -> None:
    first = _build(generated_at="2026-11-07T12:00:00Z").to_dict()
    second = _build(generated_at="2026-11-08T08:30:00Z").to_dict()

    assert first["volatile_fields"] == ["generated_at", "integrity_hash"]
    assert first["repro_hash"] == second["repro_hash"]
    assert first["integrity_hash"] != second["integrity_hash"]
    for key in first["volatile_fields"]:
        first.pop(key)
        second.pop(key)
    assert first == second


def test_input_order_does_not_change_the_evidence() -> None:
    forward = _build().to_dict()
    reversed_ = _build(
        negative_controls=list(reversed(_negatives())),
        positive_controls=list(reversed(_positives())),
        workflow=list(reversed(_workflow())),
    ).to_dict()

    assert forward == reversed_


@pytest.mark.parametrize("host_kind", ["darwin", "windows", "other", "linux"])
def test_unsupported_host_is_a_controlled_skip_and_never_a_pass(host_kind) -> None:
    reason = "boundary_unavailable" if host_kind == "linux" else "unsupported_host"
    attestation = build_no_egress_attestation(
        host_kind=host_kind,
        requested_boundary="auto",
        boundary_kind="none",
        skip_reason=reason,
        fixtures=_FIXTURES,
        generated_at="2026-11-07T12:00:00Z",
    )

    assert attestation.status == "skipped"
    assert attestation.reason_code == reason
    assert not attestation.passed
    assert attestation.exit_code == NO_EGRESS_EXIT_CODES["skipped"] == 3
    assert attestation.exit_code != 0
    assert attestation.to_dict()["boundary"]["mechanism"] == "none"
    assert validate_no_egress_attestation(attestation.to_dict()).exit_code == 3


def test_no_boundary_without_a_skip_reason_is_refused() -> None:
    with pytest.raises(NoEgressAttestationError) as exc_info:
        _build(boundary_kind="none")

    assert exc_info.value.code == "boundary_required"


def test_skip_cannot_carry_evidence_or_a_boundary() -> None:
    with pytest.raises(NoEgressAttestationError, match="skip_with_evidence"):
        _build(boundary_kind="none", skip_reason="unsupported_host")
    with pytest.raises(NoEgressAttestationError, match="skip_requires_no_boundary"):
        _build(
            skip_reason="unsupported_host",
            negative_controls=(),
            positive_controls=(),
            workflow=(),
        )


def test_hand_written_pass_without_a_boundary_does_not_validate() -> None:
    payload = _build().to_dict()
    payload["boundary"].update(kind="none", mechanism="none")

    with pytest.raises(NoEgressAttestationError, match="schema_invalid"):
        validate_no_egress_attestation(_rehash(payload))


def test_skipped_status_cannot_be_relabelled_as_passed() -> None:
    payload = build_no_egress_attestation(
        host_kind="darwin",
        requested_boundary="auto",
        boundary_kind="none",
        skip_reason="unsupported_host",
        fixtures=_FIXTURES,
        generated_at="2026-11-07T12:00:00Z",
    ).to_dict()
    payload.update(status="passed", reason_code="all_controls_blocked")

    with pytest.raises(NoEgressAttestationError, match="schema_invalid"):
        validate_no_egress_attestation(_rehash(payload))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {
                "negative_controls": _negatives(
                    ipv6_udp_socket=("not_blocked", "datagram_sent")
                )
            },
            "egress_not_blocked",
        ),
        (
            {
                "negative_controls": _negatives(
                    native_ipv4_tcp=("not_blocked", "connection_established")
                )
            },
            "egress_not_blocked",
        ),
        (
            {
                "negative_controls": _negatives(
                    subprocess_dns=("not_blocked", "resolver_answered")
                )
            },
            "egress_not_blocked",
        ),
        (
            {
                "negative_controls": _negatives(
                    ipv4_tcp_socket=("inconclusive", "timeout")
                )
            },
            "negative_control_incomplete",
        ),
        (
            {
                "negative_controls": _negatives(
                    native_ipv6_tcp=("not_run", "executor_unavailable")
                )
            },
            "negative_control_incomplete",
        ),
        ({"negative_controls": _negatives()[:-1]}, "negative_control_incomplete"),
        (
            {"interfaces": {"total": 2, "loopback_up": 1, "non_loopback_up": 1}},
            "boundary_not_isolated",
        ),
        (
            {"routes": {"ipv4_non_loopback": 1, "ipv6_non_loopback": 0}},
            "boundary_not_isolated",
        ),
        (
            {"interfaces": {"total": 1, "loopback_up": 0, "non_loopback_up": 0}},
            "boundary_not_isolated",
        ),
        ({"kernel": "unknown"}, "boundary_not_isolated"),
        (
            {
                "positive_controls": _positives(
                    loopback_native_ipv4_tcp=("unreachable", "network_unreachable")
                )
            },
            "positive_control_failed",
        ),
        ({"positive_controls": _positives()[1:]}, "positive_control_failed"),
        (
            {"workflow": _workflow(grpc_stream_cancel=("failed", "not_cancelled"))},
            "workflow_failed",
        ),
        (
            {"workflow": _workflow(service_shutdown=("not_run", "dependency_failed"))},
            "workflow_failed",
        ),
        ({"workflow": _workflow()[:-1]}, "workflow_failed"),
    ],
)
def test_status_is_derived_from_evidence(overrides, reason) -> None:
    attestation = _build(**overrides)

    assert attestation.status == "failed"
    assert attestation.reason_code == reason
    assert attestation.exit_code == 1
    assert validate_no_egress_attestation(attestation.to_dict()).reason_code == reason


def test_egress_outranks_every_other_failure() -> None:
    attestation = _build(
        negative_controls=_negatives(ipv4_tcp_socket=("not_blocked", "peer_refused")),
        interfaces={"total": 2, "loopback_up": 1, "non_loopback_up": 1},
        workflow=_workflow(rest_analyze=("failed", "unexpected_result")),
    )

    assert attestation.reason_code == "egress_not_blocked"


@pytest.mark.parametrize(
    "failure", ["runner_failed", "runner_timeout", "runner_result_invalid"]
)
def test_runner_failures_fail_closed(failure) -> None:
    attestation = _build(
        runner_failure=failure,
        negative_controls=(),
        positive_controls=(),
        workflow=(),
    )

    assert attestation.status == "failed"
    assert attestation.reason_code == failure
    assert attestation.exit_code == 1


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"host_kind": "plan9"}, "host_kind_invalid"),
        ({"requested_boundary": "chroot"}, "requested_boundary_invalid"),
        ({"boundary_kind": "vm"}, "boundary_kind_invalid"),
        ({"kernel": "darwin"}, "kernel_invalid"),
        ({"skip_reason": "operator_said_so"}, "skip_reason_invalid"),
        ({"runner_failure": "flaky"}, "runner_failure_invalid"),
        (
            {
                "negative_controls": [
                    {
                        "id": "ipv4_tcp_to_example_org",
                        "status": "blocked",
                        "reason_code": "network_unreachable",
                    }
                ]
            },
            "case_unknown",
        ),
        (
            {"negative_controls": _negatives() + _negatives()[:1]},
            "duplicate_case",
        ),
        (
            {"negative_controls": _negatives(ipv4_tcp_socket=("blocked", "timeout"))},
            "blocked_reason_invalid",
        ),
        (
            {
                "negative_controls": _negatives(
                    ipv4_tcp_socket=("blocked", "raw error text")
                )
            },
            "case_reason_invalid",
        ),
        (
            {"workflow": _workflow(rest_job=("passed", "unexpected_result"))},
            "case_reason_invalid",
        ),
        (
            {
                "workflow": [
                    {
                        "id": "rest_analyze",
                        "status": "passed",
                        "reason_code": "ok",
                        "result_digest": "md5:abc",
                    }
                ]
            },
            "case_digest_invalid",
        ),
        ({"interfaces": {"total": -1}}, "count_invalid"),
        ({"interfaces": {"total": True}}, "count_invalid"),
        ({"fixtures": {}}, "fixtures_required"),
        ({"fixtures": {"/etc/passwd": _DIGEST}}, "fixture_unknown"),
        ({"fixtures": {"service_workflow_fixture": "abc"}}, "fixture_digest_invalid"),
    ],
)
def test_inputs_outside_the_vocabulary_are_rejected(overrides, code) -> None:
    with pytest.raises(NoEgressAttestationError) as exc_info:
        _build(**overrides)

    assert exc_info.value.code == code
    assert str(exc_info.value) == code


def test_unknown_commit_and_version_are_recorded_as_unknown() -> None:
    payload = _build(commit="main; rm -rf /", openmed_version="dev build").to_dict()

    assert payload["commit"] == "unknown"
    assert payload["openmed_version"] == "unknown"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda payload: payload.update(status="failed"), "schema_invalid"),
        (
            lambda payload: payload["negative_controls"][0].update(
                status="not_blocked", reason_code="connection_established"
            ),
            "counts_mismatch",
        ),
        (lambda payload: payload["boundary"].update(isolated=False), "schema_invalid"),
        (
            lambda payload: payload["negative_controls"][0].update(
                executor="subprocess"
            ),
            "case_vocabulary_mismatch",
        ),
        (
            lambda payload: payload["workflow"][0].update(surface="grpc"),
            "case_vocabulary_mismatch",
        ),
        (lambda payload: payload.update(commit="f" * 40), "repro_hash_mismatch"),
        (
            lambda payload: payload.update(generated_at="2027-01-01T00:00:00Z"),
            "integrity_hash_mismatch",
        ),
        (lambda payload: payload.update(destination="example.org"), "schema_invalid"),
    ],
)
def test_tampering_is_detected(mutate, code) -> None:
    payload = _build().to_dict()
    mutate(payload)

    with pytest.raises(NoEgressAttestationError) as exc_info:
        validate_no_egress_attestation(payload)

    assert exc_info.value.code == code


def test_counts_tampered_back_into_shape_still_fail_on_status() -> None:
    payload = _build(
        negative_controls=_negatives(ipv4_udp_socket=("not_blocked", "datagram_sent"))
    ).to_dict()
    payload["negative_controls"][2].update(
        status="blocked", reason_code="network_unreachable"
    )
    payload["counts"]["negative_controls"].update(blocked=12, not_blocked=0)

    with pytest.raises(NoEgressAttestationError, match="status_mismatch"):
        validate_no_egress_attestation(_rehash(payload))


def test_json_round_trip_and_markdown_are_value_free() -> None:
    attestation = _build()
    restored = NoEgressAttestation.from_json(attestation.to_json())
    markdown = attestation.to_markdown()

    assert restored.to_dict() == attestation.to_dict()
    assert restored.repro_hash_matches()
    assert restored.integrity_hash_matches()
    assert "| `ipv6_tcp_socket` | `tcp` | `ipv6` | `python_socket` |" in markdown
    assert "`all_controls_blocked`" in markdown
    assert "192.0.2" not in markdown


def test_validate_rejects_non_objects() -> None:
    with pytest.raises(NoEgressAttestationError, match="attestation_not_object"):
        validate_no_egress_attestation(["not", "an", "object"])  # type: ignore[arg-type]
