"""Replica-local state inventory, readiness gate, and route refusal tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.service import runtime as service_runtime
from openmed.service.api_compatibility import discover_service_error_categories
from openmed.service.app import create_app, replica_state_report
from openmed.service.auth import hash_api_key
from openmed.service.replica_state import (
    REPLICA_LOCAL_COMPONENTS,
    REPLICA_STATE_REPORT_SCHEMA,
    ComponentState,
    ReplicaComponentUnavailableError,
    ReplicaMode,
    ReplicaSafety,
    ReplicaStatePolicy,
    StateScope,
    parse_replica_mode,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
SENTINEL = "SYNTHETIC-REPLICA-SENTINEL-7f3a"
JOB_CATEGORIES = ["async_jobs", "fhir_bulk_jobs", "smart_backend_jobs"]
ALL_JOBS_DISABLED = ",".join(JOB_CATEGORIES)

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_JOBS_STORE_PATH",
    "OPENMED_SERVICE_REPLICA_MODE",
    "OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE",
    "OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE",
)


class FakeLoader:
    """Minimal model loader double for service startup tests."""

    def __init__(self, config: Any):
        self.config = config

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        return object()

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(
        "OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "service-jobs.json")
    )
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _client(app) -> TestClient:
    return TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)


def _error(response, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    error = response.json()["error"]
    assert error["code"] == code
    return error


# --- Policy parsing ------------------------------------------------------------


def test_unset_policy_defaults_to_single_with_nothing_accepted_or_disabled():
    policy = ReplicaStatePolicy.from_env({})

    assert policy == ReplicaStatePolicy()
    assert policy.mode is ReplicaMode.SINGLE
    assert policy.accepted == frozenset()
    assert policy.disabled == frozenset()
    assert parse_replica_mode("  ") is ReplicaMode.SINGLE


def test_policy_normalizes_mode_and_category_lists():
    policy = ReplicaStatePolicy.from_env(
        {
            "OPENMED_SERVICE_REPLICA_MODE": " Multi ",
            "OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE": "Rate_Limiter, ,",
            "OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE": "async_jobs,FHIR_BULK_JOBS",
        }
    )

    assert policy.mode is ReplicaMode.MULTI
    assert policy.accepted == {"rate_limiter"}
    assert policy.disabled == {"async_jobs", "fhir_bulk_jobs"}


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"OPENMED_SERVICE_REPLICA_MODE": SENTINEL}, "'single' or 'multi'"),
        (
            {"OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE": f"rate_limiter,{SENTINEL}"},
            "may list only",
        ),
        (
            {"OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE": SENTINEL},
            "may list only",
        ),
        (
            {"OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE": "rate_limiter"},
            "may list only async_jobs, fhir_bulk_jobs, smart_backend_jobs",
        ),
        (
            {
                "OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE": "async_jobs",
                "OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE": "async_jobs",
            },
            "both accepted and disabled",
        ),
    ],
)
def test_invalid_policy_fails_closed_without_echoing_values(env, fragment):
    with pytest.raises(ValueError, match=fragment) as raised:
        ReplicaStatePolicy.from_env(env)

    assert SENTINEL not in str(raised.value)
    assert raised.value.__cause__ is None


def test_invalid_policy_stops_app_creation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "replicated")

    with pytest.raises(ValueError, match="OPENMED_SERVICE_REPLICA_MODE"):
        create_app()


# --- Inventory and report -------------------------------------------------------


def test_inventory_records_scope_and_safety_for_each_stateful_component():
    inventory = {
        component.category: (component.scope, component.safety, component.disableable)
        for component in REPLICA_LOCAL_COMPONENTS
    }

    assert inventory == {
        "async_jobs": (StateScope.NODE, ReplicaSafety.UNSAFE, True),
        "fhir_bulk_jobs": (StateScope.PROCESS, ReplicaSafety.UNSAFE, True),
        "smart_backend_jobs": (StateScope.PROCESS, ReplicaSafety.UNSAFE, True),
        "rate_limiter": (StateScope.PROCESS, ReplicaSafety.PER_REPLICA, False),
        "concurrency_limiter": (StateScope.PROCESS, ReplicaSafety.PER_REPLICA, False),
        "auth_failure_limiter": (
            StateScope.PROCESS,
            ReplicaSafety.PER_REPLICA,
            False,
        ),
        "request_coalescer": (StateScope.PROCESS, ReplicaSafety.SAFE, False),
    }


def test_single_mode_report_is_ready_for_every_enabled_component():
    report = ReplicaStatePolicy().evaluate({})

    assert report.ready
    assert report.reason_code is None
    assert {status.state for status in report.components} == {ComponentState.SAFE}


def test_multi_mode_blocks_unsafe_and_per_replica_components_but_not_safe_ones():
    report = ReplicaStatePolicy(mode=ReplicaMode.MULTI).evaluate(
        {
            "rate_limiter": True,
            "concurrency_limiter": False,
            "auth_failure_limiter": True,
            "request_coalescer": True,
        }
    )

    states = {status.category: status.state for status in report.components}
    assert not report.ready
    assert report.reason_code == "replica_unsafe_state"
    assert report.unsafe_categories == (
        "async_jobs",
        "auth_failure_limiter",
        "fhir_bulk_jobs",
        "rate_limiter",
        "smart_backend_jobs",
    )
    assert states["concurrency_limiter"] is ComponentState.DISABLED
    assert states["request_coalescer"] is ComponentState.SAFE


def test_missing_enablement_is_treated_as_enabled_so_callers_fail_closed():
    report = ReplicaStatePolicy(
        mode=ReplicaMode.MULTI, disabled=frozenset(JOB_CATEGORIES)
    ).evaluate({})

    assert report.unsafe_categories == (
        "auth_failure_limiter",
        "concurrency_limiter",
        "rate_limiter",
    )


def test_accepted_and_disabled_components_let_multi_mode_become_ready():
    policy = ReplicaStatePolicy(
        mode=ReplicaMode.MULTI,
        accepted=frozenset({"rate_limiter", "concurrency_limiter"}),
        disabled=frozenset(JOB_CATEGORIES),
    )
    report = policy.evaluate(
        {
            "rate_limiter": True,
            "concurrency_limiter": True,
            "auth_failure_limiter": False,
            "request_coalescer": True,
        }
    )

    states = {status.category: status.state for status in report.components}
    assert report.ready
    assert states == {
        "async_jobs": ComponentState.DISABLED,
        "fhir_bulk_jobs": ComponentState.DISABLED,
        "smart_backend_jobs": ComponentState.DISABLED,
        "rate_limiter": ComponentState.ACCEPTED,
        "concurrency_limiter": ComponentState.ACCEPTED,
        "auth_failure_limiter": ComponentState.DISABLED,
        "request_coalescer": ComponentState.SAFE,
    }


def test_report_serializes_controlled_category_strings_only():
    report = ReplicaStatePolicy(mode=ReplicaMode.MULTI).evaluate({})
    payload = report.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["schema"] == REPLICA_STATE_REPORT_SCHEMA
    assert payload["replica_mode"] == "multi"
    assert payload["ready"] is False
    assert payload["reason_code"] == "replica_unsafe_state"
    assert [item["category"] for item in payload["components"]] == [
        component.category for component in REPLICA_LOCAL_COMPONENTS
    ]
    for item in payload["components"]:
        assert set(item) == {"category", "scope", "safety", "state"}
        assert item["scope"] in {scope.value for scope in StateScope}
        assert item["safety"] in {safety.value for safety in ReplicaSafety}
        assert item["state"] in {state.value for state in ComponentState}
    assert "/" not in encoded
    assert report.readiness_details() == {
        "replica_mode": "multi",
        "components": list(report.unsafe_categories),
    }


def test_route_refusal_codes_follow_policy():
    disabled = ReplicaStatePolicy(disabled=frozenset({"async_jobs"}))
    unsafe = ReplicaStatePolicy(mode=ReplicaMode.MULTI)
    accepted = ReplicaStatePolicy(
        mode=ReplicaMode.MULTI, accepted=frozenset({"async_jobs"})
    )

    assert disabled.refusal_code("async_jobs") == "component_disabled"
    assert disabled.refusal_code("fhir_bulk_jobs") is None
    assert unsafe.refusal_code("smart_backend_jobs") == "replica_unsafe_state"
    assert accepted.refusal_code("async_jobs") is None
    with pytest.raises(ReplicaComponentUnavailableError) as raised:
        unsafe.require_available("fhir_bulk_jobs")
    assert raised.value.category == "fhir_bulk_jobs"
    assert raised.value.code == "replica_unsafe_state"


def test_new_error_codes_are_part_of_the_stable_service_contract():
    categories = discover_service_error_categories()

    assert "replica_unsafe_state" in categories
    assert "component_disabled" in categories


# --- Readiness ------------------------------------------------------------------


def test_single_replica_default_stays_ready_with_jobs_available():
    app = create_app()

    with _client(app) as client:
        ready = client.get("/readyz")
        job = client.get(f"/jobs/{SENTINEL}")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "service": "openmed-rest"}
    assert job.status_code == 404
    assert replica_state_report(app).mode is ReplicaMode.SINGLE


def test_multi_replica_with_local_job_store_keeps_readiness_false(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv(
        "OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE",
        "fhir_bulk_jobs,smart_backend_jobs",
    )
    app = create_app()

    with _client(app) as client:
        ready = client.get("/readyz")
        live = client.get("/livez")

    error = _error(ready, 503, "replica_unsafe_state")
    assert error["details"] == {"replica_mode": "multi", "components": ["async_jobs"]}
    assert live.status_code == 200


def test_multi_replica_defaults_name_every_unsafe_job_component(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    app = create_app()

    with _client(app) as client:
        error = _error(client.get("/readyz"), 503, "replica_unsafe_state")

    assert error["details"] == {"replica_mode": "multi", "components": JOB_CATEGORIES}


def test_preload_readiness_still_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    client = _client(create_app())
    try:
        response = client.get("/readyz")
    finally:
        client.close()

    _error(response, 503, "not_ready")


def test_unaccepted_limiters_keep_multi_replica_readiness_false(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE", ALL_JOBS_DISABLED)
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "5")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "5")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("OPENMED_SERVICE_COALESCING_ENABLED", "true")
    app = create_app()

    with _client(app) as client:
        error = _error(client.get("/readyz"), 503, "replica_unsafe_state")

    assert error["details"]["components"] == ["concurrency_limiter", "rate_limiter"]


def test_explicitly_accepted_limiters_stay_ready_in_multi_replica_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE", ALL_JOBS_DISABLED)
    monkeypatch.setenv(
        "OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE",
        "rate_limiter,concurrency_limiter",
    )
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "5")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "5")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("OPENMED_SERVICE_COALESCING_ENABLED", "true")
    app = create_app()

    with _client(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    report = replica_state_report(app)
    states = {status.category: status.state for status in report.components}
    assert states["rate_limiter"] is ComponentState.ACCEPTED
    assert states["concurrency_limiter"] is ComponentState.ACCEPTED
    assert states["request_coalescer"] is ComponentState.SAFE


def test_auth_failure_limiter_needs_acceptance_in_multi_replica_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE", ALL_JOBS_DISABLED)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "synthetic-key",
                    "key_hash": hash_api_key("synthetic-secret"),
                    "principal": "synthetic-service",
                    "scopes": ["jobs:read"],
                }
            ]
        ),
    )

    with _client(create_app()) as client:
        blocked = client.get("/readyz")
    monkeypatch.setenv(
        "OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE", "auth_failure_limiter"
    )
    with _client(create_app()) as client:
        accepted = client.get("/readyz")

    error = _error(blocked, 503, "replica_unsafe_state")
    assert error["details"]["components"] == ["auth_failure_limiter"]
    assert accepted.status_code == 200


def test_report_before_lifespan_reads_limiter_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE", ALL_JOBS_DISABLED)
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "3")

    report = replica_state_report(create_app())

    assert report.unsafe_categories == ("concurrency_limiter",)


# --- Route refusal --------------------------------------------------------------

_GATED_ROUTES = [
    ("post", "/jobs", "async_jobs"),
    ("get", f"/jobs/{SENTINEL}", "async_jobs"),
    ("post", "/fhir/bulk/exports", "fhir_bulk_jobs"),
    ("post", "/fhir/bulk/imports", "fhir_bulk_jobs"),
    ("get", f"/fhir/bulk/exports/{SENTINEL}", "fhir_bulk_jobs"),
    ("get", f"/fhir/bulk/imports/{SENTINEL}/manifest", "fhir_bulk_jobs"),
    ("get", f"/fhir/bulk/exports/{SENTINEL}/report", "fhir_bulk_jobs"),
    ("delete", f"/fhir/bulk/imports/{SENTINEL}", "fhir_bulk_jobs"),
    ("post", "/fhir/smart-backend/ingestions", "smart_backend_jobs"),
    ("get", f"/fhir/smart-backend/ingestions/{SENTINEL}", "smart_backend_jobs"),
    (
        "get",
        f"/fhir/smart-backend/ingestions/{SENTINEL}/summary",
        "smart_backend_jobs",
    ),
    ("delete", f"/fhir/smart-backend/ingestions/{SENTINEL}", "smart_backend_jobs"),
]


def _call(client: TestClient, method: str, path: str):
    if method == "post":
        # Deliberately invalid bodies prove the gate runs before validation.
        return client.post(path, json={"unexpected": SENTINEL})
    return client.request(method.upper(), path)


def test_unaccepted_job_routes_refuse_with_controlled_503_in_multi_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    app = create_app()

    with _client(app) as client:
        responses = [
            (category, _call(client, method, path))
            for method, path, category in _GATED_ROUTES
        ]

    for category, response in responses:
        error = _error(response, 503, "replica_unsafe_state")
        assert error["details"] == {"component": category, "replica_mode": "multi"}
        assert SENTINEL not in response.text
    assert app.state.job_queue is None


def test_disabled_job_routes_refuse_in_single_mode_and_stay_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE", ALL_JOBS_DISABLED)
    app = create_app()

    with _client(app) as client:
        ready = client.get("/readyz")
        responses = [
            (category, _call(client, method, path))
            for method, path, category in _GATED_ROUTES
        ]

    assert ready.status_code == 200
    for category, response in responses:
        error = _error(response, 503, "component_disabled")
        assert error["details"] == {"component": category}
        assert SENTINEL not in response.text


def test_accepted_job_component_serves_its_routes_in_multi_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE", ALL_JOBS_DISABLED)
    app = create_app()

    with _client(app) as client:
        ready = client.get("/readyz")
        job = client.get(f"/jobs/{SENTINEL}")
        bulk = client.get(f"/fhir/bulk/exports/{SENTINEL}")
        smart = client.get(f"/fhir/smart-backend/ingestions/{SENTINEL}")

    assert ready.status_code == 200
    assert [job.status_code, bulk.status_code, smart.status_code] == [404, 404, 404]


def test_blocked_job_store_never_loads_local_job_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    damaged = tmp_path / "damaged-jobs.json"
    damaged.write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(damaged))
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE", "async_jobs")
    app = create_app()

    with _client(app) as client:
        ready = client.get("/readyz")

    assert ready.status_code == 200
    assert app.state.job_queue is None
    assert damaged.read_text(encoding="utf-8") == "{broken"

    monkeypatch.delenv("OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE")
    with pytest.raises(ValueError, match="invalid JSON"):
        with _client(create_app()):
            pass


def test_authentication_runs_before_the_replica_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_REPLICA_MODE", "multi")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS", "[]")
    app = create_app()

    with _client(app) as client:
        response = client.get(f"/jobs/{SENTINEL}")

    assert response.status_code == 401
    assert "replica" not in response.text
