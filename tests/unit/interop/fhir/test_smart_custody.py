"""Offline tests for in-process SMART credential custody."""

from __future__ import annotations

import socket
from datetime import datetime, timedelta, timezone

import pytest

from openmed.interop.fhir.smart_custody import (
    ScopeEvidence,
    SmartCustodyError,
    SmartTokenCustody,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_AUDIENCE = "https://fhir.example.test/r4"
_ACCESS = "synthetic-access-secret"
_REFRESH = "synthetic-refresh-secret"


def _custody(sender, *, now: datetime = _NOW) -> SmartTokenCustody:
    return SmartTokenCustody(sender, clock=lambda: now)


def _store(custody: SmartTokenCustody, **overrides) -> str:
    values = {
        "access_token": _ACCESS,
        "refresh_token": _REFRESH,
        "audience": _AUDIENCE,
        "expires_at": _NOW + timedelta(minutes=5),
        "scopes": ("patient/Observation.sr", "launch/patient"),
    }
    values.update(overrides)
    return custody.store(**values)


def test_dispatch_passes_bearer_only_to_bound_sender_and_returns_scope_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "openmed.interop.fhir.smart_custody.secrets.token_urlsafe",
        lambda length: "opaque-handle",
    )
    custody = _custody(lambda audience, header: calls.append((audience, header)))
    handle = _store(custody)
    assert handle == "smart_opaque-handle"
    assert _ACCESS not in handle and _REFRESH not in handle

    evidence = custody.dispatch(
        handle, audience=_AUDIENCE, required_scopes=("patient/Observation.r",)
    )
    assert calls == [(_AUDIENCE, f"Bearer {_ACCESS}")]
    assert evidence.to_dict() == {"scopes": ["patient/Observation.r"]}
    assert _ACCESS not in repr(evidence)
    assert _REFRESH not in repr(custody)
    assert _AUDIENCE not in repr(evidence)


@pytest.mark.parametrize(
    "audience,scopes,reason",
    [
        (
            "https://other.example.test/r4",
            ("patient/Observation.r",),
            "audience_mismatch",
        ),
        (_AUDIENCE, ("patient/Patient.r",), "insufficient_scope"),
        (_AUDIENCE, ("user/Observation.r",), "insufficient_scope"),
        (_AUDIENCE, ("patient/Observation.u",), "insufficient_scope"),
        (_AUDIENCE, ("launch/encounter",), "insufficient_scope"),
    ],
)
def test_rejections_do_not_send_or_echo_secrets(audience, scopes, reason) -> None:
    calls: list[tuple[str, str]] = []
    custody = _custody(lambda target, header: calls.append((target, header)))
    handle = _store(custody)
    with pytest.raises(SmartCustodyError) as error:
        custody.dispatch(handle, audience=audience, required_scopes=scopes)
    assert error.value.reason_code == reason
    assert _ACCESS not in repr(error.value)
    assert _REFRESH not in str(error.value)
    assert calls == []


def test_expiry_unknown_handle_and_revocation_fail_closed() -> None:
    calls: list[tuple[str, str]] = []
    now = [_NOW]
    custody = SmartTokenCustody(
        lambda target, header: calls.append((target, header)), clock=lambda: now[0]
    )
    handle = _store(custody)
    now[0] += timedelta(minutes=5)
    with pytest.raises(SmartCustodyError, match="expired"):
        custody.dispatch(
            handle, audience=_AUDIENCE, required_scopes=("launch/patient",)
        )
    custody.revoke(handle)
    with pytest.raises(SmartCustodyError, match="unknown_handle"):
        custody.dispatch(
            handle, audience=_AUDIENCE, required_scopes=("launch/patient",)
        )
    assert calls == []


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"access_token": "bad\r\nsecret"}, "invalid_credential"),
        ({"refresh_token": "bad secret"}, "invalid_credential"),
        ({"audience": "https://user:secret@fhir.example.test/r4"}, "invalid_audience"),
        ({"audience": "https://fhir.example.test/r4?secret=x"}, "invalid_audience"),
        ({"expires_at": _NOW}, "expired"),
        ({"expires_at": datetime(2026, 1, 2)}, "invalid_expiry"),
        ({"scopes": ("Bearer synthetic-secret",)}, "invalid_scopes"),
    ],
)
def test_invalid_registration_has_value_free_errors(overrides, reason) -> None:
    custody = _custody(lambda _target, _header: None)
    with pytest.raises(SmartCustodyError) as error:
        _store(custody, **overrides)
    assert error.value.reason_code == reason
    for secret in (_ACCESS, _REFRESH, "synthetic-secret", "user:secret"):
        assert secret not in repr(error.value)


def test_sender_exception_is_replaced_by_value_free_error() -> None:
    def fail(_target: str, _header: str) -> None:
        raise RuntimeError("synthetic-access-secret")

    custody = _custody(fail)
    handle = _store(custody)
    with pytest.raises(SmartCustodyError) as error:
        custody.dispatch(
            handle, audience=_AUDIENCE, required_scopes=("launch/patient",)
        )
    assert error.value.reason_code == "dispatch_failed"
    assert _ACCESS not in str(error.value)
    assert error.value.__context__ is None


def test_untrusted_scope_iterable_exception_is_sanitized() -> None:
    def unsafe_scopes():
        raise RuntimeError("synthetic-access-secret")
        yield "patient/Observation.r"

    custody = _custody(lambda _target, _header: None)
    with pytest.raises(SmartCustodyError) as error:
        _store(custody, scopes=unsafe_scopes())
    assert error.value.reason_code == "invalid_scopes"
    assert error.value.__context__ is None


def test_scope_evidence_rejects_unvalidated_values() -> None:
    with pytest.raises(SmartCustodyError, match="invalid_scopes"):
        ScopeEvidence(("Bearer synthetic-access-secret",))
    assert _ACCESS not in repr(SmartCustodyError(_ACCESS))


def test_clock_exception_is_sanitized() -> None:
    def fail_clock() -> datetime:
        raise RuntimeError("synthetic-access-secret")

    custody = SmartTokenCustody(lambda _target, _header: None, clock=fail_clock)
    with pytest.raises(SmartCustodyError) as error:
        _store(custody)
    assert error.value.reason_code == "invalid_clock"
    assert error.value.__context__ is None


def test_custody_performs_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*_args, **_kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(socket, "socket", fail_socket)
    custody = _custody(lambda _target, _header: None)
    handle = _store(custody)
    custody.dispatch(handle, audience=_AUDIENCE, required_scopes=("launch/patient",))
