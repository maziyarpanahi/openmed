"""Validated security-policy generations for the REST service (#3704)."""

from __future__ import annotations

import copy
import json
import re
import threading
import time

import pytest

from openmed.service.policy_generations import (
    EGRESS_PURPOSES,
    POLICY_GENERATION_SCHEMA_VERSION,
    CheckpointAction,
    PolicyFinding,
    PolicyGenerationError,
    PolicyGenerationManager,
    PolicyGenerationMiddleware,
    policy_document_from_environment,
    production_policy_findings,
    validate_policy_document,
)

SENTINEL = "SENTINEL-POLICY-7f3a"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class FakeClock:
    """Deterministic monotonic test clock."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _document(**sections: dict) -> dict:
    document = {
        "schema_version": POLICY_GENERATION_SCHEMA_VERSION,
        "profile": "prod",
        "limits": {
            "max_request_bytes": 1_048_576,
            "max_text_length": 100_000,
            "rate_limit_rps": 10.0,
            "rate_limit_burst": 20,
            "max_concurrency": 8,
            "concurrency_wait_seconds": 1.0,
            "throttle_key": "global",
        },
        "authorization": {
            "enabled": True,
            "deny_by_default": True,
            "route_scopes": {},
            "failure_rate_limit_rps": 5.0,
            "failure_rate_limit_burst": 10,
            "failure_rate_limit_key": "peer",
        },
        "egress": {
            "offline": False,
            "require_tls": True,
            "allowed_purposes": ["privacy_gateway", "webhook_callback"],
        },
    }
    for name, values in sections.items():
        if name in {"schema_version", "profile"}:
            document[name] = values
        else:
            document[name] = {**document[name], **values}
    return document


def _manager(document: dict | None = None, **kwargs) -> PolicyGenerationManager:
    return PolicyGenerationManager(
        document or _document(), clock=kwargs.pop("clock", FakeClock()), **kwargs
    )


def _codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def _serialized(*objects) -> str:
    return "\n".join(
        json.dumps(item, sort_keys=True, default=str)
        if isinstance(item, (dict, list, tuple))
        else repr(item) + str(item)
        for item in objects
    )


# --- bootstrap ---------------------------------------------------------------


def test_bootstrap_activates_first_generation_with_value_free_summary() -> None:
    manager = _manager()
    active = manager.active

    assert active.number == 1
    assert active.mode == "bootstrap"
    assert active.activated_at == 1_001.0
    assert _DIGEST.fullmatch(active.digest)
    assert active.settings.limits.max_request_bytes == 1_048_576
    readiness = manager.readiness()
    assert readiness["ready"] is True
    assert readiness["generation"] == 1
    assert readiness["digest"] == active.digest
    assert readiness["last_reload"] is None
    assert set(active.to_dict()) == {"activated_at", "digest", "generation", "mode"}


def test_invalid_bootstrap_raises_fixed_code_with_findings() -> None:
    document = _document(authorization={"enabled": False})

    with pytest.raises(PolicyGenerationError) as excinfo:
        _manager(document)

    assert str(excinfo.value) == "initial_generation_invalid"
    assert excinfo.value.code == "initial_generation_invalid"
    assert "auth_disabled" in {item.code for item in excinfo.value.findings}


# --- activation and rejection ------------------------------------------------


def test_valid_reload_activates_new_generation_atomically() -> None:
    manager = _manager()
    result = manager.reload(_document(limits={"max_request_bytes": 2048}))

    assert result.outcome == "activated"
    assert result.findings == ()
    assert result.active_generation == 2
    assert manager.active.number == 2
    assert manager.active.settings.limits.max_request_bytes == 2048
    assert manager.active.activated_at == 1_002.0
    assert [item["generation"] for item in manager.history()] == [1, 2]
    assert manager.readiness()["last_reload"] == {
        "findings": [],
        "outcome": "activated",
    }


def test_identical_effective_policy_is_unchanged() -> None:
    manager = _manager()
    default_override = {"POST /analyze": ["analyze:write"]}

    result = manager.reload(_document(authorization={"route_scopes": default_override}))

    assert result.outcome == "unchanged"
    assert manager.active.number == 1
    assert result.candidate_digest == manager.active.digest


def _drop(section: str, key: str):
    def mutate(document: dict) -> None:
        del document[section][key]

    return mutate


def _set(section: str, key: str, value):
    def mutate(document: dict) -> None:
        document[section][key] = value

    return mutate


def _top(key: str, value):
    def mutate(document: dict) -> None:
        if value is _DELETE:
            del document[key]
        else:
            document[key] = value

    return mutate


_DELETE = object()


@pytest.mark.parametrize(
    ("mutate", "code", "field"),
    [
        (_drop("limits", "max_concurrency"), "field_missing", "limits.max_concurrency"),
        (_drop("egress", "require_tls"), "field_missing", "egress.require_tls"),
        (_top("egress", _DELETE), "field_missing", "egress"),
        (_set("limits", SENTINEL, 1), "field_unsupported", "limits"),
        (_top(SENTINEL, 1), "field_unsupported", None),
        (_set("limits", "max_request_bytes", True), "field_invalid", None),
        (_set("limits", "max_request_bytes", 0), "field_invalid", None),
        (_set("limits", "max_request_bytes", 1 << 40), "field_invalid", None),
        (_set("limits", "max_text_length", -1), "field_invalid", None),
        (_set("limits", "rate_limit_rps", -0.5), "field_invalid", None),
        (_set("limits", "rate_limit_rps", float("inf")), "field_invalid", None),
        (_set("limits", "rate_limit_burst", 1.5), "field_invalid", None),
        (_set("limits", "throttle_key", SENTINEL), "field_invalid", None),
        (_set("authorization", "enabled", "yes"), "field_invalid", None),
        (
            _set("authorization", "route_scopes", {f"POST /{SENTINEL}": ["x"]}),
            "route_scopes_invalid",
            "authorization.route_scopes",
        ),
        (
            _set("authorization", "route_scopes", {"POST /analyze": [SENTINEL, 3]}),
            "route_scopes_invalid",
            "authorization.route_scopes",
        ),
        (
            _set("authorization", "failure_rate_limit_key", SENTINEL),
            "field_invalid",
            "authorization.failure_rate_limit_key",
        ),
        (
            _set("egress", "allowed_purposes", [SENTINEL]),
            "egress_purpose_unsupported",
            "egress.allowed_purposes",
        ),
        (_top("schema_version", "9.0.0"), "schema_version_unsupported", None),
        (_top("profile", "dev"), "profile_change_unsupported", "profile"),
        (_top("profile", f"bad {SENTINEL}"), "profile_invalid", "profile"),
        (_set("authorization", "enabled", False), "auth_disabled", None),
        (
            _set("authorization", "deny_by_default", False),
            "deny_by_default_disabled",
            None,
        ),
        (_set("egress", "require_tls", False), "egress_tls_not_required", None),
        (
            _set("authorization", "route_scopes", {"POST /analyze": []}),
            "route_scope_missing",
            "authorization.route_scopes",
        ),
        (
            _set("authorization", "failure_rate_limit_burst", 0),
            "auth_failure_limit_disabled",
            None,
        ),
    ],
)
def test_invalid_or_incomplete_candidates_keep_prior_generation(
    mutate, code, field
) -> None:
    manager = _manager()
    prior = manager.active
    document = copy.deepcopy(_document())
    mutate(document)

    result = manager.reload(document)

    assert result.outcome == "rejected"
    assert code in _codes(result)
    if field is not None:
        assert PolicyFinding(code, field) in result.findings
    assert manager.active is prior
    readiness = manager.readiness()
    assert readiness["ready"] is True
    assert readiness["generation"] == 1
    assert readiness["last_reload"]["outcome"] == "rejected"
    assert code in readiness["last_reload"]["findings"]
    assert SENTINEL not in _serialized(result, result.to_dict(), readiness)


def test_production_requires_some_admission_budget() -> None:
    manager = _manager()
    document = _document(
        limits={"max_concurrency": 0, "rate_limit_rps": 0.0, "rate_limit_burst": 0}
    )

    result = manager.reload(document)

    assert result.outcome == "rejected"
    assert "admission_unbounded" in _codes(result)


def test_non_mapping_document_is_rejected_without_echo() -> None:
    manager = _manager()
    result = manager.reload([SENTINEL])  # type: ignore[arg-type]

    assert _codes(result) == {"document_invalid"}
    assert result.candidate_digest is None
    assert SENTINEL not in _serialized(result, result.to_dict())


def test_development_profile_allows_explicit_development_posture() -> None:
    document = _document(
        profile="dev",
        authorization={"enabled": False},
        egress={"require_tls": False},
    )
    manager = _manager(document)

    assert manager.active.settings.profile == "dev"
    assert production_policy_findings(manager.active.settings) == ()


def test_validate_policy_document_reports_every_finding_once() -> None:
    document = _document(
        authorization={"enabled": False, "deny_by_default": False},
        egress={"require_tls": False},
    )
    validation = validate_policy_document(document)

    assert validation.settings is None
    codes = [item.code for item in validation.findings]
    assert len(codes) == len(set(codes))
    assert {"auth_disabled", "deny_by_default_disabled"} <= set(codes)
    assert list(validation.findings) == sorted(validation.findings)


# --- injected self-check -----------------------------------------------------


def test_injected_self_check_findings_block_activation() -> None:
    calls = []

    def gate(settings):
        calls.append(settings.digest())
        if settings.limits.max_request_bytes > 4096:
            return ["custom_gate_failed"]
        return []

    manager = _manager(
        _document(limits={"max_request_bytes": 4096}), self_checks=[gate]
    )
    result = manager.reload(_document(limits={"max_request_bytes": 8192}))

    assert result.outcome == "rejected"
    assert _codes(result) == {"custom_gate_failed"}
    assert manager.active.number == 1
    assert len(calls) == 2


def test_self_check_errors_and_free_text_findings_are_controlled() -> None:
    def broken(settings):
        if settings.limits.max_concurrency == 2:
            raise RuntimeError(SENTINEL)
        return ()

    def chatty(settings):
        if settings.limits.max_concurrency == 2:
            return [f"Leaked {SENTINEL} value"]
        return ()

    manager = _manager(self_checks=(broken, chatty))
    result = manager.reload(_document(limits={"max_concurrency": 2}))

    assert _codes(result) == {"self_check_failed", "self_check_finding_invalid"}
    assert manager.active.number == 1
    assert SENTINEL not in _serialized(result, result.to_dict(), manager.readiness())


def test_concurrency_cap_alone_is_an_admission_budget() -> None:
    manager = _manager()
    result = manager.reload(_document(limits={"max_concurrency": 0}))

    assert result.outcome == "activated"


# --- pinning and continuation semantics -------------------------------------


def test_in_flight_pins_keep_their_generation_through_reload() -> None:
    manager = _manager()
    with manager.pin() as old_pin:
        manager.reload(_document(limits={"max_request_bytes": 2048}))
        with manager.pin() as new_pin:
            assert old_pin.generation.number == 1
            assert old_pin.settings.limits.max_request_bytes == 1_048_576
            assert new_pin.generation.number == 2
            assert manager.readiness()["inflight"] == {"1": 1, "2": 1}
            # A standard reload never rebinds in-flight work at checkpoints.
            decision = old_pin.checkpoint()
            assert decision.action is CheckpointAction.CONTINUE
            assert decision.generation == 1
    assert manager.readiness()["inflight"] == {}
    assert old_pin.released


def test_release_is_idempotent() -> None:
    manager = _manager()
    pin = manager.pin()
    pin.release()
    pin.release()
    assert manager.readiness()["inflight"] == {}


def test_stale_candidate_is_rejected() -> None:
    manager = _manager()
    candidate = manager.propose(_document(limits={"max_request_bytes": 4096}))
    assert candidate.valid
    manager.reload(_document(limits={"max_request_bytes": 2048}))

    result = manager.activate(candidate)

    assert result.outcome == "rejected"
    assert _codes(result) == {"candidate_stale"}
    assert manager.active.settings.limits.max_request_bytes == 2048


def test_invalid_candidate_cannot_be_activated() -> None:
    manager = _manager()
    candidate = manager.propose(_document(limits={"max_request_bytes": -1}))

    assert not candidate.valid
    assert manager.activate(candidate).outcome == "rejected"
    assert manager.active.number == 1


def test_emergency_change_must_be_restrictive() -> None:
    manager = _manager()
    looser = _document(limits={"max_request_bytes": 4_194_304})

    result = manager.reload(looser, emergency=True)

    assert result.outcome == "rejected"
    assert _codes(result) == {"emergency_not_restrictive"}
    assert manager.active.number == 1


@pytest.mark.parametrize(
    "tighter",
    [
        {"limits": {"max_request_bytes": 1024}},
        {"limits": {"max_concurrency": 1}},
        {"limits": {"rate_limit_rps": 1.0, "rate_limit_burst": 1}},
        {"egress": {"allowed_purposes": []}},
        {"egress": {"offline": True}},
        {"authorization": {"route_scopes": {"POST /analyze": ["analyze:write", "x"]}}},
    ],
)
def test_emergency_restriction_rebinds_in_flight_work_at_checkpoint(tighter) -> None:
    manager = _manager()
    pin = manager.pin()

    result = manager.reload(_document(**tighter), emergency=True)

    assert result.outcome == "activated"
    assert result.mode == "emergency"
    assert pin.generation.number == 1  # untouched until a declared checkpoint
    decision = pin.checkpoint()
    assert decision.action is CheckpointAction.REBOUND
    assert decision.generation == 2
    assert pin.generation is manager.active
    assert manager.readiness()["inflight"] == {"2": 1}
    assert pin.checkpoint().action is CheckpointAction.CONTINUE
    pin.release()
    # Work admitted after the emergency already runs under it.
    with manager.pin() as later:
        assert later.checkpoint().action is CheckpointAction.CONTINUE


def test_checkpoint_keeps_stricter_pin_when_emergency_was_lifted() -> None:
    manager = _manager()
    pin = manager.pin()
    manager.reload(_document(limits={"max_request_bytes": 1024}), emergency=True)
    manager.reload(_document(limits={"max_request_bytes": 4_194_304}))

    decision = pin.checkpoint()

    assert decision.action is CheckpointAction.CONTINUE
    assert decision.generation == 1
    assert pin.generation.number == 1


def test_checkpoint_halts_when_policies_are_incomparable() -> None:
    manager = _manager()
    pin = manager.pin()
    manager.reload(_document(limits={"max_request_bytes": 4_194_304}))
    restricted = _document(
        limits={"max_request_bytes": 4_194_304},
        egress={"allowed_purposes": ["privacy_gateway"]},
    )
    assert manager.reload(restricted, emergency=True).outcome == "activated"

    decision = pin.checkpoint()

    assert decision.action is CheckpointAction.HALT
    assert decision.code == "policy_continuation_denied"
    assert pin.halted
    assert pin.generation.number == 1
    assert pin.checkpoint().action is CheckpointAction.HALT
    pin.release()


def test_restrictiveness_relation_is_reflexive_and_directional() -> None:
    base = validate_policy_document(_document()).settings
    tight = validate_policy_document(
        _document(limits={"max_request_bytes": 1024})
    ).settings
    assert base is not None and tight is not None
    assert base.is_no_more_permissive_than(base)
    assert tight.is_no_more_permissive_than(base)
    assert not base.is_no_more_permissive_than(tight)


# --- egress decisions --------------------------------------------------------


def test_egress_rules_are_fail_closed() -> None:
    settings = validate_policy_document(_document()).settings
    assert settings is not None
    egress = settings.egress

    assert egress.allows("privacy_gateway", tls=True)
    assert not egress.allows("privacy_gateway", tls=False)
    assert not egress.allows("remote_inference", tls=True)
    assert not egress.allows(SENTINEL, tls=True)
    offline = validate_policy_document(_document(egress={"offline": True})).settings
    assert offline is not None
    assert not offline.egress.allows("privacy_gateway", tls=True)
    assert "privacy_gateway" in EGRESS_PURPOSES


# --- concurrency -------------------------------------------------------------


def test_concurrent_requests_never_observe_mixed_generations() -> None:
    doc_a = _document(
        limits={"max_request_bytes": 1024, "max_concurrency": 2},
        egress={"allowed_purposes": ["privacy_gateway"]},
    )
    doc_b = _document(
        limits={"max_request_bytes": 2048, "max_concurrency": 4},
        egress={"allowed_purposes": ["webhook_callback"]},
    )
    expected = {
        (1024, 2, frozenset({"privacy_gateway"})),
        (2048, 4, frozenset({"webhook_callback"})),
    }
    manager = _manager(doc_a)
    stop = threading.Event()
    errors: list[str] = []
    observed: set[tuple] = set()

    def reader() -> None:
        while not stop.is_set():
            with manager.pin() as pin:
                first = pin.settings
                for _ in range(5):
                    # Each field is read through the pin separately, with a
                    # forced thread switch in between, as a request would.
                    size = pin.settings.limits.max_request_bytes
                    time.sleep(0)
                    cap = pin.settings.limits.max_concurrency
                    time.sleep(0)
                    purposes = pin.settings.egress.allowed_purposes
                    snapshot = (size, cap, purposes)
                    if snapshot not in expected:
                        errors.append("mixed")
                    if pin.settings is not first:
                        errors.append("pin_changed")
                    observed.add(snapshot)
                if pin.generation.digest != first.digest():
                    errors.append("digest")

    def writer() -> None:
        for index in range(300):
            manager.reload(doc_b if index % 2 == 0 else doc_a)
            time.sleep(0)
        stop.set()

    readers = [threading.Thread(target=reader) for _ in range(8)]
    for thread in readers:
        thread.start()
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    writer_thread.join(timeout=60)
    stop.set()
    for thread in readers:
        thread.join(timeout=60)

    assert errors == []
    assert observed <= expected
    assert manager.active.number == 301
    assert manager.readiness()["inflight"] == {}


def test_concurrent_writers_produce_unique_ordered_generations() -> None:
    manager = _manager()
    numbers: list[int] = []
    lock = threading.Lock()

    def writer(offset: int) -> None:
        for index in range(50):
            size = 1024 * (offset * 100 + index + 1)
            result = manager.reload(_document(limits={"max_request_bytes": size}))
            with lock:
                numbers.append(result.active_generation)

    threads = [threading.Thread(target=writer, args=(item,)) for item in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert sorted(numbers) == list(range(2, 202))
    assert manager.active.number == 201


# --- environment bridge ------------------------------------------------------


def _clear_service_env(monkeypatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("OPENMED_"):
            monkeypatch.delenv(name, raising=False)


def test_environment_bridge_uses_existing_service_parsers(monkeypatch) -> None:
    _clear_service_env(monkeypatch)
    environ = {
        "OPENMED_PROFILE": "prod",
        "OPENMED_SERVICE_MAX_REQUEST_BYTES": "4096",
        "OPENMED_SERVICE_MAX_TEXT_LENGTH": "5000",
        "OPENMED_SERVICE_RATE_LIMIT_RPS": "3",
        "OPENMED_SERVICE_RATE_LIMIT_BURST": "6",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "2",
        "OPENMED_SERVICE_THROTTLE_KEY": "xff",
        "OPENMED_SERVICE_AUTH_ENABLED": "true",
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES": json.dumps(
            {"POST /analyze": "analyze:write extra:scope"}
        ),
        "OPENMED_SERVICE_AUTH_API_KEYS": json.dumps([{"key_hash": SENTINEL}]),
        "OPENMED_SERVICE_AUTH_JWKS": SENTINEL,
        "OPENMED_SERVICE_AUTH_JWT_ISSUER": SENTINEL,
        "OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT": f"https://{SENTINEL}.invalid",
        "OPENMED_SERVICE_OTLP_HEADERS": SENTINEL,
        "OPENMED_SERVICE_MTLS_CA_BUNDLE": f"/tmp/{SENTINEL}.pem",
    }

    document = policy_document_from_environment(environ)

    assert SENTINEL not in json.dumps(document)
    assert document["limits"]["max_request_bytes"] == 4096
    assert document["limits"]["max_text_length"] == 5000
    assert document["limits"]["rate_limit_rps"] == 3.0
    assert document["limits"]["max_concurrency"] == 2
    assert document["limits"]["throttle_key"] == "x-forwarded-for"
    assert document["authorization"]["enabled"] is True
    assert document["authorization"]["route_scopes"]["POST /analyze"] == [
        "analyze:write",
        "extra:scope",
    ]
    assert document["egress"]["allowed_purposes"] == ["privacy_gateway"]
    manager = PolicyGenerationManager(document, clock=FakeClock())
    assert manager.active.settings.limits.max_request_bytes == 4096
    assert SENTINEL not in _serialized(
        manager, manager.active, manager.readiness(), manager.history()
    )


def test_environment_bridge_honors_offline_flag() -> None:
    document = policy_document_from_environment(
        {"OPENMED_OFFLINE": "1", "OPENMED_PROFILE": "dev"}
    )
    assert document["egress"]["offline"] is True
    assert document["egress"]["allowed_purposes"] == []


def test_invalid_environment_raises_fixed_code() -> None:
    with pytest.raises(PolicyGenerationError) as excinfo:
        policy_document_from_environment(
            {"OPENMED_SERVICE_RATE_LIMIT_RPS": f"{SENTINEL}"}
        )

    assert str(excinfo.value) == "environment_invalid"
    assert excinfo.value.__cause__ is None
    assert SENTINEL not in repr(excinfo.value)


# --- diagnostics -------------------------------------------------------------


def test_diagnostics_never_dump_policy_text() -> None:
    document = _document(
        authorization={
            "route_scopes": {"POST /analyze": ["analyze:write", f"scope-{SENTINEL}"]}
        }
    )
    manager = _manager(document)
    candidate = manager.propose(document)
    result = manager.reload(document)
    pin = manager.pin()

    text = _serialized(
        manager,
        manager.active,
        manager.active.settings,
        manager.active.settings.authorization,
        manager.active.to_dict(),
        manager.readiness(),
        manager.history(),
        candidate,
        result,
        result.to_dict(),
        pin,
        pin.checkpoint(),
    )
    pin.release()

    assert SENTINEL not in text
    assert "analyze:write" not in text


def test_policy_finding_rejects_uncontrolled_values() -> None:
    with pytest.raises(PolicyGenerationError):
        PolicyFinding(f"bad {SENTINEL}")
    assert PolicyFinding("field_missing", f"limits.{SENTINEL}").field is None


# --- ASGI integration --------------------------------------------------------


def test_middleware_pins_each_request_for_its_full_lifetime() -> None:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    manager = _manager()

    async def handler(request: Request) -> JSONResponse:
        pin = request.scope["state"][PolicyGenerationMiddleware.STATE_KEY]
        before = pin.generation.number
        if request.query_params.get("reload") == "1":
            manager.reload(_document(limits={"max_request_bytes": 2048}))
        return JSONResponse(
            {
                "before": before,
                "after": pin.generation.number,
                "bytes": pin.settings.limits.max_request_bytes,
                "inflight": manager.readiness()["inflight"],
            }
        )

    app = Starlette(routes=[Route("/work", handler)])
    app.add_middleware(PolicyGenerationMiddleware, manager=manager)

    with TestClient(app) as client:
        first = client.get("/work", params={"reload": "1"}).json()
        second = client.get("/work").json()

    assert first == {
        "before": 1,
        "after": 1,
        "bytes": 1_048_576,
        "inflight": {"1": 1},
    }
    assert second["before"] == 2
    assert second["bytes"] == 2048
    assert manager.readiness()["inflight"] == {}


def test_activate_rejects_foreign_objects() -> None:
    manager = _manager()
    result = manager.activate(object())  # type: ignore[arg-type]

    assert _codes(result) == {"candidate_invalid"}
    assert manager.active.number == 1
