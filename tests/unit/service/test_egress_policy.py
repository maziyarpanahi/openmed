"""Unit and property tests for the fail-closed service egress policy (#3581).

Every destination, secret, and payload below is synthetic. Resolution is
injected, so no test performs DNS or network I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.offline import OFFLINE_ENV_VAR
from openmed.service.egress import (
    ADAPTER_COVERAGE,
    ADAPTER_PURPOSES,
    DENY_REASON_CODES,
    EGRESS_MODE_ENV_VAR,
    EGRESS_REASON_CODES,
    EGRESS_RULES_ENV_VAR,
    EGRESS_RULES_FILE_ENV_VAR,
    PURPOSE_DATA_CLASSES,
    EgressAdapter,
    EgressConfigurationError,
    EgressDenied,
    EgressMode,
    EgressOperation,
    EgressPolicy,
    EgressPurpose,
    EgressRequest,
    current_egress_policy,
    default_mode_for_profile,
    egress_http_client,
    egress_scope,
    parse_destination,
    pinned_grpc_target,
)

SENTINEL_HOST = "sentinel-callback.example.org"
SENTINEL_PATH = "/patient/jane-roe-mrn-4417"
SENTINEL_QUERY = "token=synthetic-secret-91f2"
PUBLIC_IP = "93.184.216.34"


def _resolver(*answers: Any):
    """Return a resolver yielding each answer in turn (last one repeats)."""

    calls: list[tuple[str, int]] = []
    queue = list(answers)

    def resolve(host: str, port: int) -> Any:
        calls.append((host, port))
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


def _policy(
    *rules: dict[str, Any],
    mode: str = "enforce",
    resolver: Any = None,
    receipts: list[Any] | None = None,
    **kwargs: Any,
) -> EgressPolicy:
    return EgressPolicy.from_rules(
        rules,
        mode=mode,
        resolver=resolver or _resolver([PUBLIC_IP]),
        receipt_sink=receipts.append if receipts is not None else None,
        **kwargs,
    )


def _webhook_rule(destination: str = f"https://{SENTINEL_HOST}", **extra: Any):
    return {"purposes": ["webhook"], "destination": destination, **extra}


def _request(destination: str, **overrides: Any) -> EgressRequest:
    values = {
        "adapter": "webhooks",
        "purpose": "webhook",
        "destination": destination,
        "data_class": "metadata",
        "credential": "signature",
    }
    values.update(overrides)
    return EgressRequest(**values)


@pytest.fixture(autouse=True)
def _clean_egress_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        EGRESS_MODE_ENV_VAR,
        EGRESS_RULES_ENV_VAR,
        EGRESS_RULES_FILE_ENV_VAR,
        "OPENMED_PROFILE",
        OFFLINE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_prod_and_unknown_profiles_default_to_enforce() -> None:
    assert default_mode_for_profile(None) is EgressMode.ENFORCE
    assert default_mode_for_profile("prod") is EgressMode.ENFORCE
    assert default_mode_for_profile("low_resource") is EgressMode.ENFORCE
    assert default_mode_for_profile("dev") is EgressMode.PERMISSIVE
    assert default_mode_for_profile("test") is EgressMode.PERMISSIVE
    assert EgressPolicy.from_env({}).mode is EgressMode.ENFORCE
    assert EgressPolicy.from_env({}).rules == ()


def test_environment_mode_overrides_profile_default() -> None:
    policy = EgressPolicy.from_env(
        {EGRESS_MODE_ENV_VAR: "deny", "OPENMED_PROFILE": "dev"}
    )
    assert policy.mode is EgressMode.DENY


def test_rules_load_from_env_and_file(tmp_path: Path) -> None:
    rules_file = tmp_path / "egress.json"
    rules_file.write_text(
        json.dumps({"rules": [_webhook_rule("https://hooks-b.example.org")]}),
        encoding="utf-8",
    )
    policy = EgressPolicy.from_env(
        {
            EGRESS_RULES_ENV_VAR: json.dumps([_webhook_rule()]),
            EGRESS_RULES_FILE_ENV_VAR: str(rules_file),
        }
    )
    assert len(policy.rules) == 2
    assert {rule.destination.port for rule in policy.rules} == {443}


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("not-json", "invalid_rules"),
        (json.dumps({"rule": []}), "invalid_rules"),
        (json.dumps([{"purposes": ["webhook"]}]), "invalid_rule_destination"),
        (json.dumps([_webhook_rule(extra_key=True)]), "invalid_rule"),
        (
            json.dumps(
                [{"purposes": ["telemetry_upload"], "destination": "https://a.b"}]
            ),
            "invalid_rule_purpose",
        ),
        (
            json.dumps([_webhook_rule(f"https://{SENTINEL_HOST}{SENTINEL_PATH}")]),
            "invalid_rule_destination",
        ),
        (
            json.dumps([_webhook_rule(f"https://user:pw@{SENTINEL_HOST}")]),
            "invalid_rule_destination",
        ),
        (json.dumps([_webhook_rule(cidrs=["0.0.0.0/0"])]), "invalid_rule_network"),
        (json.dumps([_webhook_rule(cidrs=["not-a-network"])]), "invalid_rule_network"),
        (json.dumps([_webhook_rule(max_request_bytes=-1)]), "invalid_rule_limit"),
        (
            json.dumps([{"purposes": ["webhook"], "destination": "grpc://a.b:1"}]),
            "invalid_rule_destination",
        ),
    ],
)
def test_invalid_rules_fail_closed_without_echoing_values(
    raw: str, reason: str
) -> None:
    with pytest.raises(EgressConfigurationError) as exc_info:
        EgressPolicy.from_env({EGRESS_RULES_ENV_VAR: raw})
    assert exc_info.value.reason_code == reason
    message = str(exc_info.value)
    for sentinel in (SENTINEL_HOST, SENTINEL_PATH, "jane-roe", "user:pw"):
        assert sentinel not in message


def test_invalid_mode_and_unreadable_rules_file_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(EgressConfigurationError) as mode_error:
        EgressPolicy.from_env({EGRESS_MODE_ENV_VAR: "allow-everything"})
    assert mode_error.value.reason_code == "invalid_mode"
    missing = tmp_path / "missing-sentinel-rules.json"
    with pytest.raises(EgressConfigurationError) as file_error:
        EgressPolicy.from_env({EGRESS_RULES_FILE_ENV_VAR: str(missing)})
    assert file_error.value.reason_code == "rules_file_unreadable"
    assert "missing-sentinel-rules" not in str(file_error.value)


def test_current_policy_tracks_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert current_egress_policy().mode is EgressMode.ENFORCE
    monkeypatch.setenv(EGRESS_MODE_ENV_VAR, "permissive")
    assert current_egress_policy().mode is EgressMode.PERMISSIVE
    monkeypatch.setenv(EGRESS_RULES_ENV_VAR, json.dumps([_webhook_rule()]))
    assert len(current_egress_policy().rules) == 1


def test_safe_posture_summary_contains_no_destinations() -> None:
    policy = _policy(
        _webhook_rule(allow_credentials=True),
        {
            "purposes": ["openhim"],
            "destination": "http://openhim-core.internal:8080",
            "cidrs": ["10.20.0.0/16"],
        },
    )
    summary = policy.to_safe_dict()
    encoded = json.dumps(summary)
    assert summary["mode"] == "enforce"
    assert summary["rule_count"] == 2
    assert summary["rules_by_purpose"]["webhook"] == 1
    assert summary["credential_rules"] == 1
    assert summary["private_network_rules"] == 1
    assert SENTINEL_HOST not in encoded
    assert "openhim-core" not in encoded
    assert "10.20" not in encoded


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def test_enforce_mode_denies_everything_without_a_rule() -> None:
    decision = _policy().authorize(_request(f"https://{SENTINEL_HOST}/hook"))
    assert not decision.allowed
    assert decision.reason_code == "no_matching_rule"


def test_matching_rule_allows_its_purpose_and_origin_only() -> None:
    policy = _policy(_webhook_rule())
    allowed = policy.authorize(_request(f"https://{SENTINEL_HOST}/hook?x=1"))
    assert allowed.allowed
    assert allowed.reason_code == "allowed_by_rule"
    assert allowed.destination_class == "unresolved"

    other_port = policy.authorize(_request(f"https://{SENTINEL_HOST}:8443/hook"))
    other_scheme = policy.authorize(_request(f"http://{SENTINEL_HOST}/hook"))
    other_host = policy.authorize(_request("https://other.example.org/hook"))
    assert {
        other_port.reason_code,
        other_scheme.reason_code,
        other_host.reason_code,
    } == {"no_matching_rule"}


def test_cross_purpose_use_of_a_rule_is_denied() -> None:
    policy = _policy(_webhook_rule())
    decision = policy.authorize(
        _request(
            f"https://{SENTINEL_HOST}/v1",
            adapter="privacy_gateway",
            purpose="privacy_gateway",
            data_class="deidentified_text",
            credential="none",
        )
    )
    assert decision.reason_code == "no_matching_rule"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"adapter": "telemetry_beacon"}, "unknown_adapter"),
        ({"purpose": "analytics"}, "unknown_purpose"),
        ({"data_class": "raw_phi"}, "unknown_data_class"),
        ({"credential": "cookie"}, "unknown_credential"),
        (
            {"purpose": "remote_inference", "data_class": "tensors"},
            "purpose_not_permitted",
        ),
        ({"data_class": "deidentified_text"}, "data_class_not_permitted"),
    ],
)
def test_unknown_or_mismatched_classifications_fail_closed(
    overrides: dict[str, str], reason: str
) -> None:
    policy = _policy(_webhook_rule())
    decision = policy.authorize(_request(f"https://{SENTINEL_HOST}/hook", **overrides))
    assert not decision.allowed
    assert decision.reason_code == reason


@pytest.mark.parametrize(
    ("destination", "reason"),
    [
        ("ftp://files.example.org/x", "unsupported_scheme"),
        ("grpc://triton.example.org:8001", "unsupported_scheme"),
        (f"https://user:secret@{SENTINEL_HOST}/hook", "userinfo_not_allowed"),
        ("https:///no-host", "invalid_destination"),
        ("https://exa mple.org/", "invalid_destination"),
        ("https://*.example.org/", "invalid_destination"),
        ("https://example.org:0/", "invalid_destination"),
        ("not a url", "invalid_destination"),
        ("example.org/no-scheme", "unsupported_scheme"),
        ("", "invalid_destination"),
    ],
)
def test_malformed_destinations_are_denied(destination: str, reason: str) -> None:
    decision = _policy(_webhook_rule(), mode="permissive").authorize(
        _request(destination)
    )
    assert not decision.allowed
    assert decision.reason_code == reason


def test_permissive_mode_allows_public_https_but_not_insecure_http() -> None:
    policy = _policy(mode="permissive")
    allowed = policy.authorize(_request("https://hooks.example.org/x"))
    insecure = policy.authorize(_request("http://hooks.example.org/x"))
    assert allowed.allowed and allowed.reason_code == "allowed_permissive"
    assert insecure.reason_code == "insecure_transport"


def test_explicit_http_rule_allows_plain_http() -> None:
    policy = _policy(_webhook_rule("http://hooks.example.org:8080"))
    assert policy.authorize(_request("http://hooks.example.org:8080/x")).allowed


def test_deny_mode_overrides_rules() -> None:
    decision = _policy(_webhook_rule(), mode="deny").authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    assert decision.reason_code == "egress_disabled"


def test_offline_environment_overrides_allow_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(_webhook_rule(), mode="permissive")
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    decision = policy.authorize(_request(f"https://{SENTINEL_HOST}/hook"))
    assert decision.reason_code == "offline_mode"


def test_local_only_policy_overrides_allow_rules() -> None:
    policy = _policy(_webhook_rule(), mode="permissive", local_only=True)
    decision = policy.authorize(_request(f"https://{SENTINEL_HOST}/hook"))
    assert decision.reason_code == "offline_mode"
    assert policy.to_safe_dict()["offline_override"] is True


def test_offline_toggle_after_authorization_blocks_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _policy(_webhook_rule()).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    monkeypatch.setenv(OFFLINE_ENV_VAR, "true")
    with pytest.raises(EgressDenied) as exc_info:
        decision.resolve()
    assert exc_info.value.reason_code == "offline_mode"


def test_credentials_require_an_explicit_rule_flag() -> None:
    bearer = {
        "adapter": "smart_backend",
        "purpose": "fhir_bulk",
        "data_class": "none",
        "credential": "bearer",
    }
    rule = {"purposes": ["fhir_bulk"], "destination": "https://fhir.example.org"}
    denied = _policy(rule).authorize(
        _request("https://fhir.example.org/$export", **bearer)
    )
    allowed = _policy({**rule, "allow_credentials": True}).authorize(
        _request("https://fhir.example.org/$export", **bearer)
    )
    assert denied.reason_code == "credentials_not_permitted"
    assert allowed.allowed


def test_request_signatures_are_not_treated_as_forwarded_credentials() -> None:
    assert (
        _policy(_webhook_rule())
        .authorize(_request(f"https://{SENTINEL_HOST}/hook", credential="signature"))
        .allowed
    )


def test_redirect_hops_need_an_explicit_rule_flag() -> None:
    destination = f"https://{SENTINEL_HOST}/next"
    assert (
        _policy(_webhook_rule()).authorize(_request(destination, redirect=True))
    ).reason_code == "redirect_not_permitted"
    assert (
        _policy(_webhook_rule(allow_redirects=True))
        .authorize(_request(destination, redirect=True))
        .allowed
    )
    assert (
        _policy(mode="permissive").authorize(_request(destination, redirect=True))
    ).reason_code == "redirect_not_permitted"


def test_byte_and_deadline_budgets_are_enforced() -> None:
    policy = _policy(_webhook_rule(max_request_bytes=64, max_timeout_seconds=5))
    destination = f"https://{SENTINEL_HOST}/hook"
    assert policy.authorize(_request(destination, body_bytes=65)).reason_code == (
        "budget_exceeded"
    )
    assert policy.authorize(_request(destination, timeout_seconds=6)).reason_code == (
        "deadline_exceeded"
    )
    assert policy.authorize(
        _request(destination, timeout_seconds=float("inf"))
    ).reason_code == ("deadline_exceeded")
    decision = policy.authorize(_request(destination, timeout_seconds=5))
    assert decision.allowed
    assert decision.require_body(64).body_bytes == 64
    with pytest.raises(EgressDenied) as exc_info:
        decision.require_body(65)
    assert exc_info.value.reason_code == "budget_exceeded"
    assert exc_info.value.receipt is not None
    assert exc_info.value.receipt.stage == "budget"


def test_default_budget_applies_without_rule_limits() -> None:
    policy = _policy(_webhook_rule(), max_request_bytes=10)
    decision = policy.authorize(_request(f"https://{SENTINEL_HOST}/hook"))
    with pytest.raises(EgressDenied):
        decision.require_body(11)


# ---------------------------------------------------------------------------
# Address and resolution checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "reason"),
    [
        ("127.0.0.1", "loopback_address"),
        ("[::1]", "loopback_address"),
        ("10.1.2.3", "private_address"),
        ("192.168.4.5", "private_address"),
        ("[fd12:3456::1]", "private_address"),
        ("169.254.10.10", "link_local_address"),
        ("169.254.169.254", "cloud_metadata_address"),
        ("[fd00:ec2::254]", "cloud_metadata_address"),
        ("100.100.100.200", "cloud_metadata_address"),
        ("224.0.0.1", "multicast_address"),
        ("0.0.0.0", "unspecified_address"),
        ("[::ffff:127.0.0.1]", "loopback_address"),
        ("[::ffff:169.254.169.254]", "cloud_metadata_address"),
    ],
)
def test_literal_non_public_addresses_are_denied_before_any_io(
    address: str, reason: str
) -> None:
    resolver = _resolver(AssertionError("literal addresses must not be resolved"))
    decision = _policy(mode="permissive", resolver=resolver).authorize(
        _request(f"https://{address}/latest/meta-data")
    )
    assert not decision.allowed
    assert decision.reason_code == reason
    assert resolver.calls == []


def test_literal_ip_rule_allows_exactly_that_address() -> None:
    policy = _policy(_webhook_rule("http://127.0.0.1:8099"))
    decision = policy.authorize(_request("http://127.0.0.1:8099/hook"))
    assert decision.allowed
    assert decision.destination_class == "loopback"
    assert decision.resolve() == ("127.0.0.1",)


def test_cloud_metadata_requires_a_single_address_network() -> None:
    destination = "https://metadata.internal"
    broad = _policy(
        _webhook_rule(destination, cidrs=["169.254.0.0/16"]),
        resolver=_resolver(["169.254.169.254"]),
    )
    exact = _policy(
        _webhook_rule(destination, cidrs=["169.254.169.254/32"]),
        resolver=_resolver(["169.254.169.254"]),
    )
    with pytest.raises(EgressDenied) as exc_info:
        broad.authorize(_request(destination + "/x")).resolve()
    assert exc_info.value.reason_code == "cloud_metadata_address"
    assert exact.authorize(_request(destination + "/x")).resolve() == (
        "169.254.169.254",
    )


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (["10.0.0.7"], "private_address"),
        ([PUBLIC_IP, "127.0.0.1"], "loopback_address"),
        (["fe80::1"], "link_local_address"),
        ([], "resolution_failed"),
        (["not-an-address"], "resolution_failed"),
        (["fe80::1%en0"], "resolution_failed"),
        (OSError("synthetic resolver failure"), "resolution_failed"),
    ],
)
def test_hostname_resolution_must_yield_only_permitted_addresses(
    answer: Any, reason: str
) -> None:
    decision = _policy(_webhook_rule(), resolver=_resolver(answer)).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    assert decision.allowed
    with pytest.raises(EgressDenied) as exc_info:
        decision.resolve()
    assert exc_info.value.reason_code == reason


def test_rule_networks_allow_private_resolution_and_bound_public_answers() -> None:
    rule = _webhook_rule("http://callbacks.internal:9000", cidrs=["10.20.0.0/16"])
    private_ok = _policy(rule, resolver=_resolver(["10.20.3.4"]))
    public_outside = _policy(rule, resolver=_resolver([PUBLIC_IP]))
    private_outside = _policy(rule, resolver=_resolver(["10.30.0.1"]))
    request = _request("http://callbacks.internal:9000/hook")
    assert private_ok.authorize(request).resolve() == ("10.20.3.4",)
    with pytest.raises(EgressDenied) as public_error:
        public_outside.authorize(request).resolve()
    with pytest.raises(EgressDenied) as private_error:
        private_outside.authorize(request).resolve()
    assert public_error.value.reason_code == "address_outside_rule_networks"
    assert private_error.value.reason_code == "private_address"


def test_resolution_change_between_connections_is_rechecked() -> None:
    resolver = _resolver([PUBLIC_IP], ["169.254.169.254"])
    decision = _policy(_webhook_rule(), resolver=resolver).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    assert decision.resolve() == (PUBLIC_IP,)
    with pytest.raises(EgressDenied) as exc_info:
        decision.resolve()
    assert exc_info.value.reason_code == "cloud_metadata_address"
    assert len(resolver.calls) == 2


def test_connect_target_must_match_the_decision() -> None:
    decision = _policy(_webhook_rule()).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    with pytest.raises(EgressDenied) as host_error:
        decision.resolve(host="proxy.example.org", port=443)
    with pytest.raises(EgressDenied) as port_error:
        decision.resolve(host=SENTINEL_HOST, port=8443)
    assert host_error.value.reason_code == "destination_mismatch"
    assert port_error.value.reason_code == "destination_mismatch"


def test_denied_decisions_cannot_be_resolved_or_scoped() -> None:
    denied = _policy().authorize(_request(f"https://{SENTINEL_HOST}/hook"))
    with pytest.raises(EgressDenied) as resolve_error:
        denied.resolve()
    with pytest.raises(EgressDenied) as scope_error:
        with egress_scope(denied):
            pass
    assert resolve_error.value.reason_code == "no_decision"
    assert scope_error.value.reason_code == "no_decision"


def test_pinned_grpc_target_uses_checked_addresses_and_keeps_tls_name() -> None:
    policy = EgressPolicy.from_rules(
        [
            {
                "purposes": ["remote_inference"],
                "destination": "grpcs://triton.internal:8001",
                "cidrs": ["10.0.0.0/8", "fd00::/8"],
            }
        ],
        resolver=_resolver(["10.0.0.9", "10.0.0.5", "fd00::5"]),
    )
    decision = policy.authorize(
        EgressRequest(
            adapter="remote_inference",
            purpose="remote_inference",
            destination="grpcs://triton.internal:8001",
            data_class="tensors",
        )
    )
    target, options = pinned_grpc_target(decision)
    assert target == "ipv4:10.0.0.5:8001,10.0.0.9:8001"
    assert dict(options) == {
        "grpc.ssl_target_name_override": "triton.internal",
        "grpc.default_authority": "triton.internal:8001",
    }


# ---------------------------------------------------------------------------
# Logical operations (retries, binding, credential authority)
# ---------------------------------------------------------------------------


def test_operation_retries_share_one_context_and_count_attempts() -> None:
    receipts: list[Any] = []
    operation = EgressOperation(
        EgressAdapter.WEBHOOKS,
        policy=_policy(_webhook_rule(), receipts=receipts),
        bound_destinations=(f"https://{SENTINEL_HOST}/hook",),
    )
    for _ in range(3):
        operation.authorize(
            f"https://{SENTINEL_HOST}/hook",
            purpose=EgressPurpose.WEBHOOK,
            data_class="metadata",
            credential="signature",
        )
    assert [receipt.attempt for receipt in receipts] == [1, 2, 3]
    assert {receipt.operation_id for receipt in receipts} == {operation.operation_id}


def test_operation_cannot_expand_to_unbound_destinations() -> None:
    operation = EgressOperation(
        EgressAdapter.WEBHOOKS,
        policy=_policy(_webhook_rule(), _webhook_rule("https://second.example.org")),
        bound_destinations=(f"https://{SENTINEL_HOST}/hook",),
    )
    with pytest.raises(EgressDenied) as exc_info:
        operation.authorize(
            "https://second.example.org/hook",
            purpose="webhook",
            data_class="metadata",
        )
    assert exc_info.value.reason_code == "destination_not_bound"


def test_operation_cannot_swap_credential_classes() -> None:
    rule = {
        "purposes": ["openhim"],
        "destination": "https://openhim.example.org",
        "allow_credentials": True,
    }
    operation = EgressOperation(EgressAdapter.OPENHIM, policy=_policy(rule))
    url = "https://openhim.example.org/mediators"
    operation.authorize(url, purpose="openhim", data_class="none")
    operation.authorize(
        url, purpose="openhim", data_class="metadata", credential="basic"
    )
    operation.authorize(url, purpose="openhim", data_class="none")
    with pytest.raises(EgressDenied) as exc_info:
        operation.authorize(
            url, purpose="openhim", data_class="metadata", credential="header"
        )
    assert exc_info.value.reason_code == "credential_escalation"


def test_operation_keeps_the_policy_it_started_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EGRESS_RULES_ENV_VAR, json.dumps([_webhook_rule()]))
    operation = EgressOperation(EgressAdapter.WEBHOOKS)
    monkeypatch.delenv(EGRESS_RULES_ENV_VAR)
    assert operation.authorize(
        f"https://{SENTINEL_HOST}/hook", purpose="webhook", data_class="metadata"
    ).allowed


# ---------------------------------------------------------------------------
# Transport enforcement
# ---------------------------------------------------------------------------


def test_guarded_client_refuses_requests_without_a_decision() -> None:
    with egress_http_client(timeout=1.0) as client:
        with pytest.raises(EgressDenied) as exc_info:
            client.get(f"https://{SENTINEL_HOST}/hook")
    assert exc_info.value.reason_code == "no_decision"


def test_guarded_client_refuses_requests_outside_the_active_decision() -> None:
    decision = _policy(_webhook_rule()).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    with egress_http_client(timeout=1.0) as client, egress_scope(decision):
        with pytest.raises(EgressDenied) as exc_info:
            client.get("https://other.example.org/hook")
    assert exc_info.value.reason_code == "destination_mismatch"


def test_guarded_client_connects_to_the_pinned_address_only() -> None:
    import httpcore

    connected: list[tuple[str, int]] = []

    class RecordingBackend(httpcore.NetworkBackend):
        def connect_tcp(self, host, port, timeout=None, **kwargs):
            connected.append((host, port))
            raise httpcore.ConnectError("synthetic refusal")

        def connect_unix_socket(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("unix sockets are not used")

        def sleep(self, seconds):  # pragma: no cover
            return None

    resolver = _resolver([PUBLIC_IP])
    decision = _policy(_webhook_rule(), resolver=resolver).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    client = egress_http_client(timeout=1.0)
    pool = client._transport._inner._pool  # noqa: SLF001 - verify the pin seam.
    pool._network_backend._inner = RecordingBackend()  # noqa: SLF001
    with client, egress_scope(decision):
        with pytest.raises(Exception):
            client.get(f"https://{SENTINEL_HOST}/hook")
    assert connected == [(PUBLIC_IP, 443)]
    assert resolver.calls == [(SENTINEL_HOST, 443)]


def test_guarded_client_refuses_rebound_addresses_before_connecting() -> None:
    import httpcore

    connected: list[str] = []

    class RecordingBackend(httpcore.NetworkBackend):
        def connect_tcp(self, host, port, timeout=None, **kwargs):  # pragma: no cover
            connected.append(host)
            raise httpcore.ConnectError("synthetic refusal")

        def connect_unix_socket(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("unix sockets are not used")

        def sleep(self, seconds):  # pragma: no cover
            return None

    decision = _policy(
        _webhook_rule(), resolver=_resolver(["169.254.169.254"])
    ).authorize(_request(f"https://{SENTINEL_HOST}/hook"))
    client = egress_http_client(timeout=1.0)
    pool = client._transport._inner._pool  # noqa: SLF001 - verify the pin seam.
    pool._network_backend._inner = RecordingBackend()  # noqa: SLF001
    with client, egress_scope(decision):
        with pytest.raises(EgressDenied) as exc_info:
            client.get(f"https://{SENTINEL_HOST}/hook")
    assert exc_info.value.reason_code == "cloud_metadata_address"
    assert connected == []


# ---------------------------------------------------------------------------
# Receipts and sentinels
# ---------------------------------------------------------------------------


def test_receipts_and_logs_are_value_free(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="openmed.service.egress")
    destination = f"https://{SENTINEL_HOST}{SENTINEL_PATH}?{SENTINEL_QUERY}"
    policy = _policy(_webhook_rule(), resolver=_resolver(["10.9.9.9"]))
    allowed = policy.authorize(_request(destination, body_bytes=12))
    with pytest.raises(EgressDenied) as exc_info:
        allowed.resolve()
    denied = policy.authorize(_request(destination.replace("https", "http")))
    rendered = " ".join(
        [
            caplog.text,
            repr(allowed),
            repr(denied),
            str(exc_info.value),
            json.dumps(allowed.receipt().to_dict()),
            json.dumps(exc_info.value.receipt.to_dict()),
            repr(parse_destination(destination)),
        ]
    )
    for sentinel in (
        SENTINEL_HOST,
        "sentinel-callback",
        SENTINEL_PATH,
        "jane-roe",
        SENTINEL_QUERY,
        "synthetic-secret",
        "10.9.9.9",
    ):
        assert sentinel not in rendered
    receipt = allowed.receipt().to_dict()
    assert receipt["destination_digest"] == parse_destination(destination).digest
    assert len(receipt["destination_digest"]) == 64
    assert set(receipt) == {
        "policy_version",
        "policy_digest",
        "mode",
        "stage",
        "decision",
        "reason_code",
        "adapter",
        "purpose",
        "data_class",
        "credential",
        "destination_digest",
        "destination_class",
        "rule_id",
        "operation_id",
        "attempt",
        "request_bytes",
        "duration_ms",
    }


def test_failing_receipt_sink_does_not_change_decisions() -> None:
    def sink(_: Any) -> None:
        raise RuntimeError("sink down")

    policy = EgressPolicy.from_rules(
        [_webhook_rule()], resolver=_resolver([PUBLIC_IP]), receipt_sink=sink
    )
    assert policy.authorize(_request(f"https://{SENTINEL_HOST}/hook")).allowed
    assert not policy.authorize(_request("https://elsewhere.example.org/x")).allowed


def test_reason_code_and_coverage_inventories_are_consistent() -> None:
    assert "no_decision" in DENY_REASON_CODES
    assert set(EgressAdapter) == {row.adapter for row in ADAPTER_COVERAGE}
    for row in ADAPTER_COVERAGE:
        assert row.purposes <= ADAPTER_PURPOSES[row.adapter]
    assert set(PURPOSE_DATA_CLASSES) == set(EgressPurpose)
    covered = set().union(*ADAPTER_PURPOSES.values())
    assert covered == set(EgressPurpose)


_DESTINATION_TEXT = st.one_of(
    st.text(max_size=80),
    st.builds(
        lambda scheme, host, port, path: f"{scheme}://{host}:{port}/{path}",
        st.sampled_from(["http", "https", "grpc", "grpcs", "ftp", "file"]),
        st.from_regex(r"[a-z0-9.\-]{0,30}", fullmatch=True),
        st.integers(min_value=-5, max_value=70000),
        st.text(max_size=20),
    ),
    st.ip_addresses().map(
        lambda address: (
            f"https://[{address}]/x" if address.version == 6 else f"https://{address}/x"
        )
    ),
)


@settings(max_examples=200, deadline=None)
@given(destination=_DESTINATION_TEXT)
def test_authorize_never_raises_and_only_emits_controlled_codes(
    destination: str,
) -> None:
    receipts: list[Any] = []
    policy = _policy(_webhook_rule(), mode="permissive", receipts=receipts)
    decision = policy.authorize(_request(destination))
    assert decision.reason_code in EGRESS_REASON_CODES
    assert all(receipt.reason_code in EGRESS_REASON_CODES for receipt in receipts)
    if decision.allowed:
        assert decision.destination is not None
        assert decision.destination.secure


@settings(max_examples=200, deadline=None)
@given(address=st.ip_addresses())
def test_without_rule_networks_only_public_addresses_resolve(address: Any) -> None:
    from openmed.mcp.upstream_endpoint_policy import (
        ResolvedAddressClass,
        classify_address,
    )

    decision = _policy(_webhook_rule(), resolver=_resolver([str(address)])).authorize(
        _request(f"https://{SENTINEL_HOST}/hook")
    )
    public = classify_address(address) is ResolvedAddressClass.PUBLIC
    if public:
        assert decision.resolve() == (str(address),)
    else:
        with pytest.raises(EgressDenied) as exc_info:
            decision.resolve()
        assert exc_info.value.reason_code in DENY_REASON_CODES
