"""Offline refresh grants, safety refusals and secret-free diagnostics."""

import base64
import json

import pytest

from openmed.interop.fhir.smart_refresh import (
    SmartCredential,
    SmartCredentialRefresher,
)
from openmed.service.smart_backend import SMARTBackendConfig
from tests.fixtures.smart_refresh import (
    HANDLE,
    REQUESTED,
    Clock,
    Custody,
    old_credential,
    response,
)


def setup_refresher(payload=None, **kwargs):
    clock = Clock()
    custody = Custody(old_credential())
    forms = []
    sent = []

    def transport(form):
        forms.append(dict(form))
        return response() if payload is None else payload

    refresher = SmartCredentialRefresher(
        custody,
        transport,
        requested_scopes=REQUESTED,
        clock=clock,
        sender=sent.append,
        **kwargs,
    )
    return refresher, custody, clock, forms, sent


def test_narrowing_blocks_dropped_operation_and_preserves_allowed_dispatch():
    refresher, custody, _, forms, sent = setup_refresher(
        response(scope="system/SyntheticObservation.c")
    )
    report = refresher.dispatch(HANDLE, required_scopes="system/SyntheticObservation.u")
    assert report.to_dict() == {
        "code": "insufficient_scope",
        "findings": ["scope_narrowed"],
        "dropped_scope_count": 2,
    }
    assert not report.usable
    assert sent == []
    assert custody.slot.read().scopes == frozenset({"system/SyntheticObservation.c"})
    assert (
        refresher.dispatch(HANDLE, required_scopes="system/SyntheticObservation.c").code
        == "dispatched"
    )
    assert sent == ["Bearer synthetic-access-1"]
    assert len(forms) == 1


def test_rotation_replaces_entire_record_and_never_reuses_old_refresh():
    refresher, custody, clock, forms, _ = setup_refresher()
    assert refresher.ensure(HANDLE).code == "refreshed"
    assert custody.slot.read() == SmartCredential(
        "synthetic-access-1", "synthetic-refresh-1", 1300, frozenset(REQUESTED.split())
    )
    clock.now = 1240
    assert refresher.ensure(HANDLE).code == "refreshed"
    assert [f["refresh_token"] for f in forms] == [
        "synthetic-refresh-0",
        "synthetic-refresh-1",
    ]
    assert all(f["grant_type"] == "refresh_token" for f in forms)


def test_missing_rotation_retains_current_secret_and_missing_scope_uses_request():
    payload = response()
    del payload["refresh_token"]
    del payload["scope"]
    refresher, custody, _, _, _ = setup_refresher(payload)
    assert refresher.ensure(HANDLE).to_dict() == {
        "code": "refreshed",
        "findings": [],
        "dropped_scope_count": 0,
    }
    assert custody.slot.read().refresh_token == "synthetic-refresh-0"
    assert custody.slot.read().scopes == frozenset(REQUESTED.split())


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            {"error": "invalid_grant", "error_description": "synthetic-secret"},
            "invalid_grant",
        ),
        ({"error": "synthetic-secret"}, "grant_failed"),
        ([], "malformed_response"),
        ({}, "malformed_response"),
        (response(access_token=None), "malformed_response"),
        (response(access_token="synthetic\nsecret"), "malformed_response"),
        (response(access_token="x" * 8193), "malformed_response"),
        (response(token_type="Basic"), "unsupported_token_type"),
        (response(token_type=None), "unsupported_token_type"),
        (response(expires_in=0), "invalid_lifetime"),
        (response(expires_in=-1), "invalid_lifetime"),
        (response(expires_in=86401), "invalid_lifetime"),
        (response(expires_in=60), "invalid_lifetime"),
        (response(expires_in="300"), "invalid_lifetime"),
        (response(expires_in=True), "invalid_lifetime"),
        (response(expires_in=300.0), "invalid_lifetime"),
        (response(expires_in=float("nan")), "invalid_lifetime"),
        (response(expires_in=float("inf")), "invalid_lifetime"),
        (response(scope=None), "malformed_response"),
        (response(scope="synthetic\nsecret"), "malformed_response"),
        (response(scope='synthetic"secret'), "malformed_response"),
        (response(scope="system/SyntheticObservation.d"), "scope_escalation"),
        (response(scope="system/*.cu"), "scope_escalation"),
        (response(refresh_token=""), "malformed_response"),
        (response(refresh_token="synthetic\nsecret"), "malformed_response"),
    ],
)
def test_invalid_response_erases_credentials_and_never_retries(payload, code):
    refresher, custody, clock, forms, sent = setup_refresher(payload)
    assert refresher.ensure(HANDLE).code == code
    assert custody.slot.read() is None
    assert custody.slot.revoked
    for _ in range(5):
        clock.now += 10000
        assert refresher.ensure(HANDLE).code == "revoked"
        assert refresher.acquire(HANDLE).code == "revoked"
        assert (
            refresher.dispatch(
                HANDLE, required_scopes="system/SyntheticObservation.c"
            ).code
            == "revoked"
        )
    assert len(forms) == 1
    assert sent == []


def test_new_instance_cannot_retry_revoked_handle():
    refresher, custody, clock, forms, _ = setup_refresher({"error": "invalid_grant"})
    assert refresher.ensure(HANDLE).code == "invalid_grant"
    other = SmartCredentialRefresher(
        custody,
        lambda form: forms.append(form),
        requested_scopes=REQUESTED,
        clock=clock,
    )
    assert other.ensure(HANDLE).code == "revoked"
    assert len(forms) == 1


@pytest.mark.parametrize("margin", [0, 10, 60])
def test_margin_boundary_and_no_early_network(margin):
    refresher, custody, clock, forms, _ = setup_refresher(refresh_margin=margin)
    custody.slot.replace(old_credential(expires_at=1100))
    clock.now = 1100 - margin - 0.01
    assert refresher.ensure(HANDLE).code == "current"
    assert not forms
    clock.now = 1100 - margin
    assert refresher.ensure(HANDLE).code == "refreshed"
    assert len(forms) == 1


def test_slow_response_cannot_extend_expiry_or_trigger_success_loop():
    refresher, custody, clock, forms, _ = setup_refresher()

    def slow(form):
        forms.append(form)
        clock.now += 250
        return response(expires_in=300)

    refresher._transport = slow
    assert refresher.ensure(HANDLE).code == "invalid_lifetime"
    assert custody.slot.revoked


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_clock_revokes_without_transport(value):
    refresher, custody, clock, forms, _ = setup_refresher()
    clock.now = value
    assert refresher.ensure(HANDLE).code == "clock_unavailable"
    assert custody.slot.revoked
    assert not forms


def test_backward_clock_during_transport_revokes():
    refresher, custody, clock, _, _ = setup_refresher()

    def transport(form):
        clock.now -= 1
        return response()

    refresher._transport = transport
    assert refresher.ensure(HANDLE).code == "clock_unavailable"
    assert custody.slot.revoked


def test_empty_granted_scope_blocks_all_dispatch():
    refresher, _, _, _, sent = setup_refresher(response(scope=""))
    assert (
        refresher.dispatch(HANDLE, required_scopes="system/SyntheticObservation.c").code
        == "insufficient_scope"
    )
    assert not sent


def test_current_credentials_still_check_scopes():
    refresher, custody, _, forms, sent = setup_refresher()
    custody.slot.replace(old_credential(expires_at=1500))
    assert (
        refresher.dispatch(HANDLE, required_scopes="system/SyntheticObservation.d").code
        == "insufficient_scope"
    )
    assert not forms and not sent


def test_unknown_handle_is_not_implicitly_acquired():
    refresher, custody, _, forms, _ = setup_refresher()
    custody.slot.credential = None
    assert refresher.ensure(HANDLE).code == "unavailable"
    assert not forms


def backend_config():
    return SMARTBackendConfig(
        fhir_base_url="https://synthetic-fhir.test",
        token_url="https://synthetic-endpoint.test/token",
        client_id="synthetic-client-secret",
        private_key_pem="synthetic-key-secret",
        output_dir="synthetic-unused-output",
        scope=REQUESTED,
    )


def test_backend_acquisition_and_reacquisition_reuses_existing_builder(monkeypatch):
    from openmed.service import smart_backend

    monkeypatch.setattr(
        smart_backend, "_sign_rs384", lambda data, key: b"synthetic-signature"
    )
    clock = Clock()
    custody = Custody()
    forms = []

    def transport(form):
        forms.append(dict(form))
        payload = response(token_type="bEaReR")
        del payload["refresh_token"]
        return payload

    refresher = SmartCredentialRefresher(
        custody,
        transport,
        requested_scopes=REQUESTED,
        clock=clock,
        backend_config=backend_config(),
    )
    assert refresher.acquire(HANDLE).code == "acquired"
    assert refresher.acquire(HANDLE).code == "already_acquired"
    clock.now = 1240
    assert refresher.ensure(HANDLE).code == "refreshed"
    assert len(forms) == 2
    for form, now in zip(forms, [1000, 1240], strict=True):
        assert form["grant_type"] == "client_credentials"
        assert form["client_assertion_type"].endswith("jwt-bearer")
        claims = json.loads(
            base64.urlsafe_b64decode(form["client_assertion"].split(".")[1] + "==")
        )
        assert claims["iat"] == now
        assert claims["exp"] == now + 300
        assert claims["sub"] == "synthetic-client-secret"


def test_no_refresh_secret_or_backend_config_revokes():
    refresher, custody, _, forms, _ = setup_refresher()
    custody.slot.replace(old_credential(refresh=None))
    assert refresher.ensure(HANDLE).code == "refresh_unavailable"
    assert custody.slot.revoked
    assert not forms


@pytest.mark.parametrize(
    "failure_at",
    ["transport", "signer", "replace", "read", "revoke", "clock", "sender"],
)
def test_no_secrets_in_results_logs_or_exceptions(failure_at, caplog, capsys):
    secrets = [
        "synthetic-access-0",
        "synthetic-refresh-0",
        "synthetic-client-secret",
        "https://synthetic-endpoint.test/token",
        "synthetic-assertion-secret",
    ]
    leaked = " ".join(secrets)

    def failing(*args):
        raise RuntimeError(leaked)

    refresher, custody, _, _, _ = setup_refresher(
        backend_config=backend_config(),
        assertion_builder=lambda config: "synthetic-assertion-secret",
    )
    if failure_at == "transport":
        refresher._transport = failing
    elif failure_at == "signer":
        refresher._assertion_builder = failing
    elif failure_at in {"read", "replace"}:
        setattr(custody.slot, failure_at, failing)
    elif failure_at == "revoke":
        refresher._transport = failing
        custody.slot.revoke = failing
    elif failure_at == "clock":
        refresher._clock = failing
    else:
        refresher._sender = failing
    report = refresher.dispatch(HANDLE, required_scopes="system/SyntheticObservation.c")
    observed = (
        json.dumps(report.to_dict()) + repr(report) + repr(custody.slot.credential)
    )
    captured = capsys.readouterr()
    observed += caplog.text + captured.out + captured.err
    assert not report.usable
    for secret in secrets:
        assert secret not in observed
    assert report.code in {"refresh_failed", "custody_unavailable", "dispatch_failed"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refresh_margin": -1},
        {"refresh_margin": float("nan")},
        {"refresh_margin": 86400},
        {"max_lifetime": True},
        {"max_lifetime": 86401},
    ],
)
def test_configuration_errors_are_fixed(kwargs):
    with pytest.raises(ValueError, match="^invalid_refresh_configuration$"):
        setup_refresher(**kwargs)


@pytest.mark.parametrize("scopes", ["", "synthetic\nsecret", 'synthetic"secret'])
def test_invalid_dispatch_requirement_does_not_call_endpoint(scopes):
    refresher, _, _, forms, sent = setup_refresher()
    assert (
        refresher.dispatch(HANDLE, required_scopes=scopes).code
        == "invalid_required_scopes"
    )
    assert not forms and not sent


def test_sender_must_be_bound_before_dispatch():
    refresher = SmartCredentialRefresher(
        Custody(old_credential()),
        lambda form: response(),
        requested_scopes=REQUESTED,
        clock=Clock(),
    )
    assert (
        refresher.dispatch(HANDLE, required_scopes="system/SyntheticObservation.c").code
        == "dispatch_unavailable"
    )
