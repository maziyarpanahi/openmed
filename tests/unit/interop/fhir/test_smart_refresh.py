"""Offline token rotation, narrowing, custody faults and privacy controls."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from openmed.interop.fhir.smart_refresh import (
    SMARTCredential,
    SMARTCredentialRefresher,
    SMARTRefreshConfig,
    SMARTRefreshLease,
    SMARTRefreshReport,
    SMARTTokenRequest,
    SMARTTokenResponse,
    SMARTTokenValidationError,
    validate_smart_token_response,
)
from openmed.interop.smart_scope_audit import (
    SmartGrantedScopeAudit,
    audit_granted_smart_scopes,
    smart_scopes_cover,
)

HANDLE = "cred_" + "1" * 32
SCOPES = ("system/Observation.cruds",)
SECRET = "refresh-secret-canary"


class Custody:
    """Synthetic atomic store; reservations never auto-restore old tokens."""

    def __init__(
        self, config: SMARTRefreshConfig, credential: SMARTCredential | None = None
    ):
        self.config = config
        self.current = credential or SMARTCredential("old-access", SECRET, 110, SCOPES)
        self.busy = False
        self.revoked = False
        self.generation = 0
        self.lease = None
        self.replacements = []

    def reserve(self, handle):
        if handle != HANDLE or self.busy or self.revoked:
            return None
        self.busy = True
        self.lease = SMARTRefreshLease(
            handle,
            "lease_" + f"{self.generation:032x}",
            self.generation,
            self.config.binding_digest,
            self.current,
        )
        return self.lease

    def replace(self, lease, credential):
        if (
            self.revoked
            or not self.busy
            or lease is not self.lease
            or lease.generation != self.generation
        ):
            return False
        self.current = credential
        self.replacements.append(credential)
        self.generation += 1
        self.busy = False
        return True

    def release(self, lease):
        if self.revoked or not self.busy or lease is not self.lease:
            return False
        self.busy = False
        self.generation += 1
        return True

    def revoke(self, handle):
        assert handle == HANDLE
        self.revoked = True
        self.busy = True
        self.current = None
        return True


def config(**kwargs):
    return SMARTRefreshConfig(
        "https://auth.example.test/private-token",
        "private-client-canary",
        "refresh_token",
        SCOPES,
        **kwargs,
    )


def response(**kwargs):
    payload = {
        "access_token": "new-access",
        "token_type": "Bearer",
        "expires_in": 300,
        "scope": " ".join(SCOPES),
    }
    payload.update(kwargs)
    return SMARTTokenResponse(200, json.dumps(payload).encode())


def refresher(store, transport, clock=lambda: 100, **kwargs):
    return SMARTCredentialRefresher(
        store.config, custody=store, transport=transport, clock=clock, **kwargs
    )


@pytest.mark.parametrize(
    ("required", "granted", "covered"),
    [
        (SCOPES, ("system/Observation.rs", "system/Observation.cud"), True),
        (("system/Observation.read",), ("system/Observation.rs",), True),
        (("system/Observation.write",), ("system/Observation.cud",), True),
        (("system/Observation.cruds",), ("system/*.*",), True),
        (("system/*.read",), ("system/Observation.rs",), False),
        (("system/Observation.rs",), ("patient/Observation.rs",), False),
        (("system/Observation.rs?category=private",), ("system/Observation.rs",), True),
        (
            ("system/Observation.rs",),
            ("system/Observation.rs?category=private",),
            False,
        ),
        (
            ("system/Observation.rs?category=a",),
            ("system/Observation.rs?category=b",),
            False,
        ),
        (
            ("system/Observation.rs?category=a",),
            ("system/Observation.r?category=a", "system/Observation.s?category=a"),
            True,
        ),
        (("offline_access",), ("online_access",), False),
        ((), (), True),
    ],
)
def test_permission_unions(required, granted, covered):
    assert smart_scopes_cover(required, granted) is covered


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "system/Observation.rr",
        "system/Observation.BAD",
        "system/observation.r",
        "system/Observation.r\nprivate",
        "system/Observation.r private",
        'private"canary',
        "private\\canary",
        "system/Observation.r?name=é",
        "x" * 513,
    ],
)
def test_scope_errors_do_not_echo_private_values(bad):
    with pytest.raises(ValueError, match="Invalid SMART") as exc:
        smart_scopes_cover((bad,), SCOPES)
    assert bad not in str(exc.value) if bad else True


def test_scope_bounds_and_value_free_findings():
    assert smart_scopes_cover(
        (f"scope{i}" for i in range(128)), (f"scope{i}" for i in range(128))
    )
    with pytest.raises(ValueError):
        smart_scopes_cover((f"scope{i}" for i in range(129)), ())
    audit = audit_granted_smart_scopes(
        requested_scopes=SCOPES,
        granted_scopes=("system/Observation.rs?name=private-canary",),
    )
    assert audit.to_dict() == {
        "requested_count": 1,
        "granted_count": 1,
        "narrowed": True,
        "expanded": False,
    }
    assert "private-canary" not in repr(audit)
    with pytest.raises(ValueError):
        SmartGrantedScopeAudit(True, 1, False, False)


def test_narrowed_rotation_changes_actual_dispatch_permissions():
    store = Custody(config())
    forms = []

    def exchange(request):
        forms.append(dict(request.form))
        assert store.busy
        assert store.reserve(HANDLE) is None
        return response(scope="system/Observation.rs", refresh_token="rotated-secret")

    worker = refresher(store, exchange)
    result = worker.ensure_ready(HANDLE, required_scopes=("system/Observation.u",))
    assert result.code == "insufficient_scope" and result.narrowed and result.refreshed
    assert result.refresh_token_rotated
    assert store.current.refresh_token == "rotated-secret"
    assert forms[0]["refresh_token"] == SECRET
    assert (
        worker.ensure_ready(HANDLE, required_scopes=("system/Observation.r",)).code
        == "ready"
    )
    assert len(forms) == 1
    store.current = replace(store.current, expires_at=110)
    assert (
        worker.ensure_ready(HANDLE, required_scopes=("system/Observation.u",)).code
        == "insufficient_scope"
    )
    assert forms[1]["refresh_token"] == "rotated-secret"
    assert forms[1]["scope"] == "system/Observation.rs"
    assert all(form["refresh_token"] != SECRET for form in forms[1:])


def test_due_margin_boundary_and_optional_unchanged_refresh_response():
    store = Custody(config())
    store.current = replace(store.current, expires_at=161)
    calls = []

    def exchange(request):
        calls.append(request)
        return SMARTTokenResponse(
            200, b'{"access_token":"new-access","token_type":"bEaReR","expires_in":300}'
        )

    worker = refresher(store, exchange)
    assert worker.ensure_ready(HANDLE).code == "ready"
    assert not calls
    store.current = replace(store.current, expires_at=160)
    assert worker.ensure_ready(HANDLE).refreshed
    assert store.current.refresh_token == SECRET
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"token_type": "MAC"},
        {"token_type": None},
        {"access_token": "private bad token"},
        {"access_token": "private\ncanary"},
        {"expires_in": True},
        {"expires_in": "300"},
        {"expires_in": 0},
        {"expires_in": -1},
        {"expires_in": 301},
        {"expires_in": 10**100},
        {"expires_in": 1.5},
        {"scope": None},
        {"scope": ""},
        {"scope": "system/Observation.r "},
        {"scope": "system/Observation.rr"},
        {"scope": "system/*.*"},
        {"refresh_token": ""},
        {"refresh_token": None},
        {"refresh_token": "private\ncanary"},
        {
            "error": "invalid_grant",
            "error_description": "private token/client/endpoint canary",
        },
    ],
)
def test_invalid_tokens_revoke_and_never_retry(changes, caplog):
    store = Custody(config())
    calls = []
    worker = refresher(
        store, lambda request: calls.append(request) or response(**changes)
    )
    result = worker.ensure_ready(HANDLE)
    assert result.code in {"invalid_response", "invalid_grant", "scope_expansion"}
    assert result.revocation_confirmed and store.current is None
    assert worker.ensure_ready(HANDLE).code == "credential_unavailable"
    assert len(calls) == 1
    assert "private" not in json.dumps(result.to_dict()) + caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json-private",
        b"[]",
        b"null",
        b"\xff",
        b'{"token_type":"Bearer","token_type":"MAC"}',
        b'{"error":"invalid_grant","error":"other"}',
        b'{"x":NaN}',
        json.dumps({"x": [[[[[[[[[[]]]]]]]]]]}).encode(),
        json.dumps({"x": [None] * 513}).encode(),
        b"{" + b"private" * 10000,
    ],
)
def test_malformed_and_bounded_responses(raw):
    store = Custody(config())

    def exchange(_):
        return SMARTTokenResponse(200, raw)

    result = refresher(store, exchange).ensure_ready(HANDLE)
    assert result.code == "invalid_response" and result.revocation_confirmed
    assert store.current is None


@pytest.mark.parametrize("status", [201, 302, 400, 401, 429, 500])
def test_no_redirect_or_retry_on_http_failure(status):
    store = Custody(config())
    result = refresher(
        store, lambda _: SMARTTokenResponse(status, response().body)
    ).ensure_ready(HANDLE)
    assert result.code == "invalid_response" and result.revocation_confirmed


@pytest.mark.parametrize("operation", ["reserve", "replace", "release", "revoke"])
@pytest.mark.parametrize("fault", ["raise", "truthy", "false"])
def test_custody_errors_do_not_release_unknown_credential(operation, fault):
    store = Custody(config())
    if operation == "release":
        store.current = replace(store.current, expires_at=500)
    original = getattr(store, operation)

    def fail(*args):
        if operation in {"replace", "release"}:
            original(*args)  # Simulate a lost acknowledgement after real commit.
        if fault == "raise":
            raise RuntimeError("private custody secret/client/endpoint")
        return 1 if fault == "truthy" else False

    setattr(store, operation, fail)
    transport = (
        (lambda _: response(token_type="MAC"))
        if operation == "revoke"
        else (lambda _: response())
    )
    result = refresher(store, transport).ensure_ready(HANDLE)
    assert result.code == "custody_unavailable"
    if operation != "revoke":
        assert result.revocation_confirmed and store.revoked and store.current is None
    else:
        assert not result.revocation_confirmed and store.busy
        assert store.reserve(HANDLE) is None


@pytest.mark.parametrize(
    "clock",
    [
        lambda: True,
        lambda: float("nan"),
        lambda: float("inf"),
        lambda: -1,
        lambda: "private",
    ],
)
def test_clock_failures_revoke(clock):
    store = Custody(config())
    result = refresher(
        store, lambda _: pytest.fail("network before valid clock"), clock
    ).ensure_ready(HANDLE)
    assert result.code == "invalid_clock" and result.revocation_confirmed


@pytest.mark.parametrize("ending", [99, 340, 400])
def test_backwards_clock_or_response_latency_consumes_lifetime(ending):
    times = iter((100, ending))
    store = Custody(config())
    result = refresher(store, lambda _: response(), lambda: next(times)).ensure_ready(
        HANDLE
    )
    assert result.code == ("invalid_clock" if ending == 99 else "invalid_response")
    assert result.revocation_confirmed


@pytest.mark.parametrize("operation", ["release", "replace"])
def test_custody_latency_cannot_return_expired_credential(operation):
    store = Custody(config())
    now = [100]
    if operation == "release":
        store.current = replace(store.current, expires_at=500)
    original = getattr(store, operation)

    def delayed(*args):
        result = original(*args)
        now[0] = 500
        return result

    setattr(store, operation, delayed)
    result = refresher(store, lambda _: response(), lambda: now[0]).ensure_ready(HANDLE)
    assert result.code == "invalid_response" and result.revocation_confirmed
    assert store.current is None


def test_refresh_without_a_refresh_secret_revokes_without_transport():
    store = Custody(config())
    store.current = replace(store.current, refresh_token=None)
    result = refresher(
        store, lambda _: pytest.fail("Missing refresh secret")
    ).ensure_ready(HANDLE)
    assert result.code == "invalid_grant" and result.revocation_confirmed


@pytest.mark.parametrize("handle", ["private-handle", None, 1])
def test_invalid_handle_never_reaches_custody_or_transport(handle):
    store = Custody(config())
    store.reserve = lambda _: pytest.fail("Invalid handle reached custody")
    result = refresher(
        store, lambda _: pytest.fail("Invalid handle reached transport")
    ).ensure_ready(handle)
    assert result.code == "invalid_request" and "private" not in str(result.to_dict())


def test_foreign_permission_in_custody_never_reaches_transport():
    store = Custody(config())
    store.current = replace(store.current, granted_scopes=("system/*.*",))
    result = refresher(
        store, lambda _: pytest.fail("Foreign authority reached transport")
    ).ensure_ready(HANDLE)
    assert result.code == "binding_mismatch" and result.revocation_confirmed


def test_binding_mismatch_revokes_requested_handle_not_substituted_handle():
    store = Custody(config())
    original = store.reserve

    def wrong(handle):
        return replace(original(handle), handle="cred_" + "2" * 32)

    store.reserve = wrong
    result = refresher(
        store, lambda _: pytest.fail("foreign binding must not contact transport")
    ).ensure_ready(HANDLE)
    assert result.code == "binding_mismatch" and store.revoked


def test_private_objects_and_error_results_do_not_echo_secrets(caplog):
    store = Custody(config())
    values = [
        store.config,
        store.current,
        SMARTTokenRequest(store.config.token_endpoint, {"refresh_token": SECRET}),
        response(),
    ]

    def fail(_):
        raise RuntimeError(
            f"{SECRET} {store.config.token_endpoint} {store.config.client_id}"
        )

    result = refresher(store, fail).ensure_ready(HANDLE)
    public = repr(values) + repr(result) + json.dumps(result.to_dict()) + caplog.text
    for secret in (
        SECRET,
        "old-access",
        "new-access",
        store.config.token_endpoint,
        store.config.client_id,
    ):
        assert secret not in public
    assert result.code == "transport_unavailable" and result.revocation_confirmed


@pytest.mark.parametrize(
    "changes",
    [
        {"token_endpoint": "http://private.example/token"},
        {"token_endpoint": "https://user:pass@private.example/token"},
        {"token_endpoint": "https://private.example/token#secret"},
        {"token_endpoint": "https://private.example:bad/token"},
        {"client_id": "private\nclient"},
        {"grant_type": "password"},
        {"requested_scopes": ()},
        {"refresh_margin_seconds": 300},
        {"max_lifetime_seconds": 86401},
        {"max_lifetime_seconds": True},
    ],
)
def test_config_errors_are_value_free(changes):
    kwargs = {
        "token_endpoint": "https://auth.example/token",
        "client_id": "private-client",
        "grant_type": "refresh_token",
        "requested_scopes": SCOPES,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match="Invalid SMART refresh configuration"):
        SMARTRefreshConfig(**kwargs)


def test_validator_backend_requires_scope_and_preserves_split_permission_union():
    missing = SMARTTokenResponse(
        200, b'{"access_token":"a","token_type":"Bearer","expires_in":300}'
    )
    with pytest.raises(SMARTTokenValidationError, match="invalid_response"):
        validate_smart_token_response(missing, requested_scopes=SCOPES, issued_at=100)
    credential, audit = validate_smart_token_response(
        response(scope="system/Observation.rs system/Observation.cud"),
        requested_scopes=SCOPES,
        issued_at=100,
    )
    assert not audit.narrowed and not audit.expanded and credential.expires_at == 400


def test_report_rejects_free_text_codes():
    with pytest.raises(ValueError, match="Invalid SMART refresh report"):
        SMARTRefreshReport("private-token-value")


def test_opaque_refresh_and_client_values_preserve_ascii_spaces():
    cfg = replace(config(), client_id=" private client ")
    store = Custody(cfg)
    store.current = replace(store.current, refresh_token=" private refresh token ")
    seen = []

    def exchange(request):
        seen.append(dict(request.form))
        return response(refresh_token=" rotated refresh token ")

    result = refresher(store, exchange).ensure_ready(HANDLE)
    assert result.code == "ready" and result.refresh_token_rotated
    assert seen[0]["client_id"] == " private client "
    assert seen[0]["refresh_token"] == " private refresh token "
    assert store.current.refresh_token == " rotated refresh token "
    assert "private" not in json.dumps(result.to_dict())


def test_backend_grant_reuses_existing_assertion_builder(monkeypatch, tmp_path):
    from openmed.service import smart_backend

    cfg = replace(config(), grant_type="client_credentials")
    store = Custody(cfg)
    store.current = None
    backend = smart_backend.SMARTBackendConfig(
        "https://fhir.example.test",
        cfg.token_endpoint,
        cfg.client_id,
        "private-key-canary",
        tmp_path,
    )
    seen = []

    def sign(data, private_key):
        header, payload = data.decode().split(".")
        import base64

        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        assert claims["iat"] == 100 and claims["exp"] == 400
        assert claims["aud"] == cfg.token_endpoint and claims["iss"] == cfg.client_id
        assert private_key == "private-key-canary"
        seen.append(claims["jti"])
        return b"synthetic-signature"

    monkeypatch.setattr(smart_backend, "_sign_rs384", sign)

    def exchange(request):
        assert request.form["grant_type"] == "client_credentials"
        assert "refresh_token" not in request.form
        assert request.form["client_assertion_type"].endswith(":jwt-bearer")
        assert len(request.form["client_assertion"].split(".")) == 3
        return response()

    worker = refresher(store, exchange, backend_config=backend)
    assert worker.ensure_ready(HANDLE).refreshed
    store.current = replace(store.current, expires_at=110)
    assert worker.ensure_ready(HANDLE).refreshed
    assert len(set(seen)) == 2
    with pytest.raises(ValueError, match="Invalid SMART refresh dependencies"):
        refresher(
            store, exchange, backend_config=replace(backend, client_id="other-client")
        )
    assert cfg.token_endpoint not in repr(backend)
