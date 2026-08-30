"""Offline regression tests for the MCP upstream endpoint policy."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import pytest

from openmed.mcp.upstream_endpoint_policy import (
    ResolvedAddressClass,
    UpstreamEndpointPolicy,
    UpstreamEndpointPolicyError,
)

PUBLIC_ADDRESS = "93.184.216.34"
SYNTHETIC_TOKEN = "synthetic-upstream-token"
SYNTHETIC_TERM = "synthetic clinical term"


def _resolver(
    answers: dict[str, Iterable[Any]],
) -> tuple[Any, list[tuple[str, int]]]:
    calls: list[tuple[str, int]] = []

    def resolve(host: str, port: int) -> Iterable[Any]:
        calls.append((host, port))
        return answers[host]

    return resolve, calls


def _policy(
    answers: dict[str, Iterable[Any]],
    *origins: str,
    allow_loopback: bool = False,
) -> tuple[UpstreamEndpointPolicy, list[tuple[str, int]]]:
    resolver, calls = _resolver(answers)
    return (
        UpstreamEndpointPolicy(
            allowed_origins=origins,
            resolver=resolver,
            allow_loopback=allow_loopback,
        ),
        calls,
    )


@pytest.mark.parametrize(
    ("answer", "reason_code"),
    [
        ("169.254.169.254", "cloud_metadata_address"),
        ("10.24.8.9", "private_address"),
        ("169.254.8.9", "link_local_address"),
        ("224.0.0.1", "multicast_address"),
        ("0.0.0.0", "unspecified_address"),
    ],
)
def test_prohibited_resolved_addresses_are_rejected_before_requester(
    answer: str,
    reason_code: str,
) -> None:
    host = f"{reason_code}.example.test"
    policy, calls = _policy({host: [answer]}, f"https://{host}")
    requested: list[str] = []
    endpoint = (
        f"https://{host}/search?access_token={SYNTHETIC_TOKEN}"
        f"&term={SYNTHETIC_TERM.replace(' ', '%20')}"
    )

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.call(endpoint, requested.append)

    assert raised.value.reason_code == reason_code
    assert requested == []
    assert calls == [(host, 443)]


def test_mixed_public_and_private_dns_answers_fail_deterministically() -> None:
    host = "mixed.example.test"
    policy, calls = _policy(
        {host: [PUBLIC_ADDRESS, "10.24.8.9"]},
        f"https://{host}",
    )

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate(f"https://{host}/fhir")

    assert raised.value.reason_code == "mixed_public_prohibited_addresses"
    assert calls == [(host, 443)]


def test_allowlisted_https_endpoint_is_authorized_and_resolution_is_recorded() -> None:
    host = "approved.example.test"
    policy, calls = _policy({host: [PUBLIC_ADDRESS]}, f"https://{host}")
    endpoint = f"https://{host}/fhir?term={SYNTHETIC_TERM.replace(' ', '%20')}"

    approval = policy.validate(endpoint)

    assert approval.url == endpoint
    assert approval.origin == f"https://{host}"
    assert approval.addresses == (PUBLIC_ADDRESS,)
    assert approval.address_classes == (ResolvedAddressClass.PUBLIC,)
    assert approval.to_safe_dict() == {
        "origin": f"https://{host}",
        "port": 443,
        "resolved_address_count": 1,
        "loopback": False,
    }
    assert calls == [(host, 443)]


def test_unapproved_https_origin_is_rejected_without_dns_lookup() -> None:
    host = "unapproved.example.test"
    policy, calls = _policy({host: [PUBLIC_ADDRESS]})

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate(f"https://{host}/fhir")

    assert raised.value.reason_code == "origin_not_allowed"
    assert calls == []


def test_remote_http_and_user_info_are_rejected_without_leaking_input() -> None:
    host = "approved.example.test"
    policy, _ = _policy({host: [PUBLIC_ADDRESS]}, f"https://{host}")
    cases = (
        f"http://{host}/?access_token={SYNTHETIC_TOKEN}",
        f"https://user:{SYNTHETIC_TOKEN}@{host}/?term={SYNTHETIC_TERM}",
    )

    for endpoint in cases:
        with pytest.raises(UpstreamEndpointPolicyError) as raised:
            policy.validate(endpoint)
        serialized = json.dumps(raised.value.to_dict())
        assert SYNTHETIC_TOKEN not in str(raised.value)
        assert SYNTHETIC_TERM not in serialized
        assert endpoint not in serialized


def test_localhost_requires_explicit_loopback_policy() -> None:
    host = "localhost"
    answers = {host: ["127.0.0.1"]}
    strict_policy, strict_calls = _policy(answers)
    endpoint = f"http://{host}:8081/mcp?token={SYNTHETIC_TOKEN}"

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        strict_policy.validate(endpoint)
    assert raised.value.reason_code == "loopback_not_allowed"
    assert strict_calls == []

    development_policy, development_calls = _policy(
        answers,
        allow_loopback=True,
    )
    approval = development_policy.validate(endpoint)
    assert approval.is_loopback is True
    assert approval.address_classes == (ResolvedAddressClass.LOOPBACK,)
    assert development_calls == [(host, 8081)]


def test_loopback_policy_rejects_dns_rebinding_and_mixed_answers() -> None:
    host = "localhost"
    policy, calls = _policy(
        {host: ["127.0.0.1", PUBLIC_ADDRESS]},
        allow_loopback=True,
    )

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate(f"http://{host}:8081/mcp")

    assert raised.value.reason_code == "loopback_only_violation"
    assert calls == [(host, 8081)]


def test_allowlisted_private_resolution_is_still_rejected() -> None:
    host = "approved-but-private.example.test"
    policy, _ = _policy({host: ["192.168.40.10"]}, f"https://{host}")

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate(f"https://{host}/fhir")

    assert raised.value.reason_code == "private_address"


def test_redirect_targets_are_revalidated_and_cannot_change_policy_mode() -> None:
    remote_host = "approved.example.test"
    redirect_host = "redirect.example.test"
    answers = {
        remote_host: [PUBLIC_ADDRESS],
        redirect_host: ["10.24.8.9"],
        "localhost": ["127.0.0.1"],
    }
    policy, calls = _policy(
        answers,
        f"https://{remote_host}",
        f"https://{redirect_host}",
        allow_loopback=True,
    )
    initial = f"https://{remote_host}/mcp?term={SYNTHETIC_TERM.replace(' ', '%20')}"

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate_redirect(
            f"https://{redirect_host}/fhir?access_token={SYNTHETIC_TOKEN}",
            base_url=initial,
        )
    assert raised.value.reason_code == "private_address"
    assert calls == [(remote_host, 443), (redirect_host, 443)]

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate_redirect("http://localhost:8081/mcp", base_url=initial)
    assert raised.value.reason_code == "redirect_mode_change"


def test_relative_redirect_chain_is_checked_at_each_hop() -> None:
    host = "approved.example.test"
    policy, calls = _policy({host: [PUBLIC_ADDRESS]}, f"https://{host}")

    approvals = policy.validate_redirect_chain(
        f"https://{host}/mcp",
        ("/mcp/next", "?cursor=synthetic-cursor"),
    )

    assert len(approvals) == 3
    assert approvals[-1].url == f"https://{host}/mcp/next?cursor=synthetic-cursor"
    assert calls == [(host, 443)] * 5


def test_policy_errors_and_logs_are_redacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    host = "private.example.test"
    policy, _ = _policy({host: ["10.0.0.8"]}, f"https://{host}")
    endpoint = (
        f"https://{host}/lookup?access_token={SYNTHETIC_TOKEN}"
        f"&query={SYNTHETIC_TERM.replace(' ', '%20')}"
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(UpstreamEndpointPolicyError) as raised:
            policy.validate(endpoint)

    captured = "\n".join(record.getMessage() for record in caplog.records)
    serialized = json.dumps(raised.value.to_dict())
    assert SYNTHETIC_TOKEN not in captured
    assert SYNTHETIC_TERM not in captured
    assert SYNTHETIC_TOKEN not in serialized
    assert SYNTHETIC_TERM not in serialized
    assert endpoint not in serialized


def test_resolver_exception_is_redacted() -> None:
    host = "resolver-failure.example.test"

    def failing_resolver(_: str, __: int) -> Iterable[str]:
        raise RuntimeError(f"{SYNTHETIC_TOKEN} {SYNTHETIC_TERM}")

    policy = UpstreamEndpointPolicy(
        allowed_origins={f"https://{host}"},
        resolver=failing_resolver,
    )

    with pytest.raises(UpstreamEndpointPolicyError) as raised:
        policy.validate(f"https://{host}/fhir?access_token={SYNTHETIC_TOKEN}")

    assert raised.value.reason_code == "dns_resolution_failed"
    assert SYNTHETIC_TOKEN not in repr(raised.value)
    assert SYNTHETIC_TERM not in repr(raised.value)
    assert raised.value.__context__ is None
