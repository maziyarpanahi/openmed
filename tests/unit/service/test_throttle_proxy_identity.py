"""Proxy-aware throttle identity and bounded limiter state tests."""

from __future__ import annotations

import asyncio
import ipaddress
import json
from datetime import datetime
from typing import Any, Optional

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

import openmed
from openmed.processing.outputs import PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key, parse_service_auth_config
from openmed.service.runtime import TrustedProxyConfig
from openmed.service.throttle import (
    ConcurrencyGate,
    TokenBucketRateLimiter,
    client_identity,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
PROXY_NETWORK = "198.51.100.0/24"
PROXY_PEER = "198.51.100.7"
SECOND_PROXY = "198.51.100.20"
UNTRUSTED_PEER = "203.0.113.50"
CLIENT_ADDRESS = "203.0.113.9"
SPOOFED_ADDRESS = "192.0.2.66"

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_METRICS_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_TRUSTED_PROXIES",
    "OPENMED_SERVICE_TRUSTED_PROXY_HOPS",
    "OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY",
    "OPENMED_SERVICE_MTLS_ENABLED",
)


class FakeLoader:
    """Minimal model loader double for throttled service tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


class PeerScopeMiddleware:
    """Override the HTTP peer address with a synthetic proxy address."""

    def __init__(self, app: ASGIApp, client: tuple[str, int]) -> None:
        self.app = app
        self.client = client

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = dict(scope)
            scope["client"] = self.client
        await self.app(scope, receive, send)


class ManualClock:
    """Injected monotonic clock advanced explicitly by tests."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _proxies(*networks: str, hops: int = 1) -> TrustedProxyConfig:
    return TrustedProxyConfig(
        networks=tuple(ipaddress.ip_network(item) for item in networks),
        hops=hops,
    )


def _request(
    peer: Optional[str],
    *forwarded_for: str,
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/analyze",
        "raw_path": b"/analyze",
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 8080),
        "client": (peer, 40000) if peer is not None else None,
        "headers": [
            (b"x-forwarded-for", value.encode("latin-1")) for value in forwarded_for
        ],
    }
    return Request(scope)


def _prediction_result(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="disease_detection_superclinical",
        timestamp=datetime.now().isoformat(),
        processing_time=0.01,
    )


def _consume(limiter: TokenBucketRateLimiter, key: str) -> bool:
    return asyncio.run(limiter.consume(key)).allowed


# Client identity resolution ------------------------------------------------


def test_rotating_leftmost_xff_from_trusted_proxy_creates_one_bucket() -> None:
    proxies = _proxies(PROXY_NETWORK)
    clock = ManualClock()
    limiter = TokenBucketRateLimiter(
        requests_per_second=0.001,
        burst_size=1,
        clock=clock,
    )

    async def scenario() -> int:
        admitted = 0
        for index in range(250):
            request = _request(
                PROXY_PEER,
                f"192.0.2.{index % 250}, {CLIENT_ADDRESS}",
            )
            key = client_identity(request, "x-forwarded-for", proxies)
            decision = await limiter.consume(key)
            admitted += int(decision.allowed)
        return admitted

    assert asyncio.run(scenario()) == 1
    assert limiter.tracked_key_count == 1


def test_rotating_xff_without_trusted_proxies_keys_by_peer() -> None:
    limiter = TokenBucketRateLimiter(
        requests_per_second=0.001,
        burst_size=1,
        clock=ManualClock(),
    )

    async def scenario() -> int:
        admitted = 0
        for index in range(250):
            request = _request(PROXY_PEER, f"192.0.2.{index % 250}")
            decision = await limiter.consume(
                client_identity(request, "x-forwarded-for")
            )
            admitted += int(decision.allowed)
        return admitted

    assert asyncio.run(scenario()) == 1
    assert limiter.tracked_key_count == 1


def test_forwarding_headers_are_ignored_from_untrusted_peers() -> None:
    proxies = _proxies(PROXY_NETWORK, hops=3)
    for key_by in ("x-forwarded-for", "peer"):
        request = _request(UNTRUSTED_PEER, f"{SPOOFED_ADDRESS}, {CLIENT_ADDRESS}")
        assert client_identity(request, key_by, proxies) == UNTRUSTED_PEER

    no_proxy_request = _request(PROXY_PEER, CLIENT_ADDRESS)
    assert client_identity(no_proxy_request, "x-forwarded-for") == PROXY_PEER
    assert (
        client_identity(no_proxy_request, "x-forwarded-for", TrustedProxyConfig())
        == PROXY_PEER
    )


def test_peer_mode_never_reads_forwarding_headers() -> None:
    request = _request(PROXY_PEER, CLIENT_ADDRESS)

    assert client_identity(request, "peer", _proxies(PROXY_NETWORK)) == PROXY_PEER
    assert client_identity(request, "global", _proxies(PROXY_NETWORK)) == "global"


def test_rightmost_untrusted_address_is_selected_across_trusted_hops() -> None:
    proxies = _proxies(PROXY_NETWORK, hops=3)
    request = _request(
        PROXY_PEER,
        f"{SPOOFED_ADDRESS}, {CLIENT_ADDRESS}, {SECOND_PROXY}",
    )

    assert client_identity(request, "x-forwarded-for", proxies) == CLIENT_ADDRESS


def test_multiple_forwarded_for_header_lines_are_read_right_to_left() -> None:
    proxies = _proxies(PROXY_NETWORK, hops=2)
    request = _request(
        PROXY_PEER,
        f"{SPOOFED_ADDRESS}, 192.0.2.67",
        f"{CLIENT_ADDRESS}, {SECOND_PROXY}",
    )

    assert client_identity(request, "x-forwarded-for", proxies) == CLIENT_ADDRESS


@pytest.mark.parametrize(
    ("hops", "expected"),
    [
        (1, SECOND_PROXY),
        (2, CLIENT_ADDRESS),
        (3, CLIENT_ADDRESS),
    ],
)
def test_hop_count_bounds_how_many_trusted_proxies_are_skipped(
    hops: int,
    expected: str,
) -> None:
    proxies = _proxies(PROXY_NETWORK, hops=hops)
    request = _request(
        PROXY_PEER,
        f"{SPOOFED_ADDRESS}, {CLIENT_ADDRESS}, {SECOND_PROXY}",
    )

    assert client_identity(request, "x-forwarded-for", proxies) == expected


def test_trusted_chain_stops_at_the_last_examined_trusted_hop() -> None:
    proxies = _proxies(PROXY_NETWORK, hops=5)
    request = _request(PROXY_PEER, f"198.51.100.30, {SECOND_PROXY}")

    assert client_identity(request, "x-forwarded-for", proxies) == "198.51.100.30"


@pytest.mark.parametrize(
    "forwarded_for",
    ["not-an-address", "", " , ", "203.0.113.9:port", "[2001:db8::1"],
)
def test_malformed_forwarded_entries_fall_back_to_the_trusted_hop(
    forwarded_for: str,
) -> None:
    proxies = _proxies(PROXY_NETWORK, hops=2)
    request = _request(PROXY_PEER, forwarded_for)

    assert client_identity(request, "x-forwarded-for", proxies) == PROXY_PEER


@pytest.mark.parametrize(
    ("forwarded_for", "expected"),
    [
        ("[2001:DB8::1]:443", "2001:db8::1"),
        ("2001:0db8:0000::0001", "2001:db8::1"),
        ("203.0.113.9:8443", CLIENT_ADDRESS),
        ("::ffff:203.0.113.9", CLIENT_ADDRESS),
    ],
)
def test_forwarded_addresses_are_canonicalized(
    forwarded_for: str,
    expected: str,
) -> None:
    proxies = _proxies(PROXY_NETWORK)
    request = _request(PROXY_PEER, forwarded_for)

    assert client_identity(request, "x-forwarded-for", proxies) == expected


def test_ipv6_proxy_networks_are_trusted() -> None:
    proxies = _proxies("2001:db8:ffff::/48", hops=2)
    request = _request("2001:db8:ffff::10", f"{CLIENT_ADDRESS}, 2001:db8:ffff::20")

    assert client_identity(request, "x-forwarded-for", proxies) == CLIENT_ADDRESS


# Token-bucket key bounds ---------------------------------------------------


def test_limiter_memory_stays_bounded_under_50000_distinct_identities() -> None:
    clock = ManualClock()
    limiter = TokenBucketRateLimiter(
        requests_per_second=10.0,
        burst_size=2,
        max_keys=1_000,
        clock=clock,
    )

    async def scenario() -> int:
        peak = 0
        for index in range(50_000):
            clock.advance(0.0001)
            address = ipaddress.ip_address("2001:db8::") + index
            await limiter.consume(str(address))
            peak = max(peak, limiter.tracked_key_count)
        return peak

    assert asyncio.run(scenario()) <= 1_000
    assert limiter.tracked_key_count <= 1_000


def test_limiter_bound_holds_without_any_idle_key_to_evict() -> None:
    limiter = TokenBucketRateLimiter(
        requests_per_second=1.0,
        burst_size=1,
        max_keys=64,
        clock=ManualClock(),
    )

    async def scenario() -> None:
        for index in range(50_000):
            await limiter.consume(f"identity-{index}")

    asyncio.run(scenario())

    assert limiter.tracked_key_count == 64


def test_eviction_pressure_never_resets_a_depleted_key() -> None:
    clock = ManualClock()
    limiter = TokenBucketRateLimiter(
        requests_per_second=1.0,
        burst_size=2,
        max_keys=4,
        clock=clock,
    )

    assert _consume(limiter, "victim") is True
    assert _consume(limiter, "victim") is True
    assert _consume(limiter, "victim") is False

    for index in range(10_000):
        clock.advance(0.00001)
        _consume(limiter, f"pressure-{index}")
        assert limiter.tracked_key_count <= 4

    assert _consume(limiter, "victim") is False
    clock.advance(0.5)
    assert _consume(limiter, "victim") is False
    clock.advance(1.0)
    assert _consume(limiter, "victim") is True
    assert _consume(limiter, "victim") is False


def test_idle_buckets_are_evicted_to_admit_new_keys_individually() -> None:
    clock = ManualClock()
    limiter = TokenBucketRateLimiter(
        requests_per_second=1.0,
        burst_size=1,
        max_keys=2,
        clock=clock,
    )

    assert _consume(limiter, "first") is True
    assert _consume(limiter, "second") is True
    clock.advance(1.0)

    assert _consume(limiter, "third") is True
    assert _consume(limiter, "fourth") is True
    assert limiter.tracked_key_count == 2
    assert _consume(limiter, "third") is False
    assert _consume(limiter, "fourth") is False


def test_overflow_bucket_is_shared_when_the_key_bound_is_reached() -> None:
    clock = ManualClock()
    limiter = TokenBucketRateLimiter(
        requests_per_second=0.001,
        burst_size=1,
        max_keys=2,
        clock=clock,
    )

    assert _consume(limiter, "first") is True
    assert _consume(limiter, "second") is True
    assert _consume(limiter, "third") is True
    assert _consume(limiter, "fourth") is False
    assert _consume(limiter, "fifth") is False
    assert limiter.tracked_key_count == 2


def test_limiter_rejects_non_positive_key_bounds() -> None:
    with pytest.raises(ValueError, match="max_keys"):
        TokenBucketRateLimiter(requests_per_second=1, burst_size=1, max_keys=0)
    with pytest.raises(ValueError, match="max_keys"):
        ConcurrencyGate(max_concurrency=1, wait_seconds=0, max_keys=0)


# Concurrency gate state ----------------------------------------------------


def test_concurrency_gate_evicts_semaphores_only_when_idle() -> None:
    gate = ConcurrencyGate(max_concurrency=1, wait_seconds=0.0)

    async def scenario() -> None:
        permit = await gate.acquire("held")
        assert permit is not None
        assert gate.tracked_key_count == 1

        assert await gate.acquire("held") is None
        assert gate.tracked_key_count == 1

        permit.release()
        assert gate.tracked_key_count == 0
        permit.release()
        assert gate.tracked_key_count == 0

        again = await gate.acquire("held")
        assert again is not None
        again.release()

        for index in range(50_000):
            transient = await gate.acquire(f"identity-{index}")
            assert transient is not None
            transient.release()
        assert gate.tracked_key_count == 0

    asyncio.run(scenario())


def test_concurrency_gate_keeps_slots_with_waiters() -> None:
    gate = ConcurrencyGate(max_concurrency=1, wait_seconds=1.0)

    async def scenario() -> None:
        permit = await gate.acquire("held")
        assert permit is not None
        waiter = asyncio.ensure_future(gate.acquire("held"))
        await asyncio.sleep(0)
        assert gate.tracked_key_count == 1

        permit.release()
        second = await waiter
        assert second is not None
        assert gate.tracked_key_count == 1
        assert await gate.acquire("held") is None

        second.release()
        assert gate.tracked_key_count == 0

    asyncio.run(scenario())


def test_concurrency_gate_bound_uses_overflow_without_resetting_held_keys() -> None:
    gate = ConcurrencyGate(max_concurrency=1, wait_seconds=0.0, max_keys=3)

    async def scenario() -> None:
        held = [await gate.acquire(f"held-{index}") for index in range(3)]
        assert all(permit is not None for permit in held)
        assert gate.tracked_key_count == 3

        overflow = await gate.acquire("new-a")
        assert overflow is not None
        assert await gate.acquire("new-b") is None
        assert gate.tracked_key_count == 3

        for index in range(3):
            assert await gate.acquire(f"held-{index}") is None

        overflow.release()
        for permit in held:
            assert permit is not None
            permit.release()
        assert gate.tracked_key_count == 0

    asyncio.run(scenario())


# Configuration -------------------------------------------------------------


def test_trusted_proxy_settings_parse_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENMED_SERVICE_TRUSTED_PROXIES",
        f"{PROXY_NETWORK}, 2001:db8:ffff::/48",
    )
    monkeypatch.setenv("OPENMED_SERVICE_TRUSTED_PROXY_HOPS", "2")
    monkeypatch.setenv("OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS", "512")

    config = service_runtime.parse_service_throttle_config()

    assert config.proxies.networks == (
        ipaddress.ip_network(PROXY_NETWORK),
        ipaddress.ip_network("2001:db8:ffff::/48"),
    )
    assert config.proxies.hops == 2
    assert config.max_tracked_keys == 512


def test_trusted_proxy_defaults_are_secure() -> None:
    config = service_runtime.parse_service_throttle_config()

    assert config.proxies.networks == ()
    assert config.proxies.hops == 1
    assert config.max_tracked_keys == 10_000


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("OPENMED_SERVICE_TRUSTED_PROXIES", "203.0.113.999/24"),
        ("OPENMED_SERVICE_TRUSTED_PROXY_HOPS", "-3"),
        ("OPENMED_SERVICE_TRUSTED_PROXY_HOPS", "many"),
        ("OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS", "-3"),
    ],
)
def test_invalid_proxy_settings_fail_closed_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    value: str,
) -> None:
    monkeypatch.setenv(env_var, value)

    with pytest.raises(ValueError) as exc_info:
        service_runtime.parse_service_throttle_config()

    assert env_var in str(exc_info.value)
    assert value not in str(exc_info.value)


def test_auth_failure_limiter_shares_the_throttle_identity_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_TRUSTED_PROXIES", PROXY_NETWORK)
    monkeypatch.setenv("OPENMED_SERVICE_TRUSTED_PROXY_HOPS", "2")
    monkeypatch.setenv("OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS", "128")

    auth_config = parse_service_auth_config()
    throttle_config = service_runtime.parse_service_throttle_config()

    assert auth_config.proxies == throttle_config.proxies
    assert auth_config.failure_rate_limit_key_by == "x-forwarded-for"
    assert auth_config.failure_rate_limit_max_tracked_keys == 128


# Service-level reproductions -----------------------------------------------


def _enable_proxy_throttle(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "0.001")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("OPENMED_SERVICE_THROTTLE_KEY", "x-forwarded-for")
    monkeypatch.setenv("OPENMED_SERVICE_TRUSTED_PROXIES", PROXY_NETWORK)
    calls: list[str] = []

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        calls.append(text)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    return calls


def test_service_rotated_leftmost_values_admit_at_most_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _enable_proxy_throttle(monkeypatch)
    app = PeerScopeMiddleware(create_app(), (PROXY_PEER, 54321))

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        statuses = [
            client.post(
                "/analyze",
                json={"text": "synthetic"},
                headers={"X-Forwarded-For": f"192.0.2.{index}, {CLIENT_ADDRESS}"},
            ).status_code
            for index in range(250)
        ]
        other_client = client.post(
            "/analyze",
            json={"text": "synthetic"},
            headers={"X-Forwarded-For": "203.0.113.10"},
        )

    assert statuses.count(200) == 1
    assert statuses.count(429) == 249
    assert other_client.status_code == 200
    assert len(calls) == 2


def _enable_auth(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "synthetic",
                    "key_hash": hash_api_key("synthetic-secret"),
                    "scopes": ["analyze:write"],
                }
            ]
        ),
    )
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS", "0.001")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST", "1")
    return {"X-API-Key": "synthetic-secret"}


def test_service_auth_failures_use_the_proxy_resolved_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_auth(monkeypatch)
    monkeypatch.setenv("OPENMED_SERVICE_TRUSTED_PROXIES", PROXY_NETWORK)
    app = PeerScopeMiddleware(create_app(), (PROXY_PEER, 54321))

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        attacker = [
            client.get(
                "/models/loaded",
                headers={"X-Forwarded-For": f"192.0.2.{index}, {CLIENT_ADDRESS}"},
            ).status_code
            for index in range(20)
        ]
        other_client = client.get(
            "/models/loaded",
            headers={"X-Forwarded-For": "203.0.113.10"},
        )

    assert attacker.count(401) == 1
    assert attacker.count(429) == 19
    assert other_client.status_code == 401


def test_rejections_and_metrics_expose_no_addresses_or_header_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_sentinel = "198.51.100.77"
    client_sentinel = "203.0.113.201"
    header_sentinel = "192.0.2.123"
    forwarded_for = f"{header_sentinel}, {client_sentinel}"
    _enable_proxy_throttle(monkeypatch)
    api_key = _enable_auth(monkeypatch)
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    app = PeerScopeMiddleware(create_app(), (peer_sentinel, 54321))

    # Negative control: the resolver sees the sentinel, so its absence below is
    # meaningful rather than an artifact of a different identity.
    sentinel_request = _request(peer_sentinel, forwarded_for)
    assert (
        client_identity(sentinel_request, "x-forwarded-for", _proxies(PROXY_NETWORK))
        == client_sentinel
    )

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        headers = {"X-Forwarded-For": forwarded_for}
        authorized = {**headers, **api_key}
        first = client.post("/analyze", json={"text": "synthetic"}, headers=authorized)
        limited = client.post(
            "/analyze", json={"text": "synthetic"}, headers=authorized
        )
        failed = client.get("/models/loaded", headers=headers)
        auth_limited = client.get("/models/loaded", headers=headers)
        metrics = client.get("/metrics")

    assert first.status_code == 200
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    assert failed.status_code == 401
    assert auth_limited.status_code == 429
    assert auth_limited.json()["error"]["code"] == "auth_rate_limited"
    assert metrics.status_code == 200
    rendered = [
        limited.text,
        json.dumps(dict(limited.headers)),
        auth_limited.text,
        json.dumps(dict(auth_limited.headers)),
        metrics.text,
    ]
    for sentinel in (peer_sentinel, client_sentinel, header_sentinel):
        assert all(sentinel not in item for item in rendered)
