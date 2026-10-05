"""Opt-in SMART credential refresh at a trusted, serialized custody boundary.

No HTTP client or storage backend is installed here. Transports, custody adapters
and dispatch callbacks are trusted secret-bearing code; public reports are not.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeGuard

from openmed.interop.smart_scope_audit import parse_smart_scope

if TYPE_CHECKING:
    from openmed.service.smart_backend import SMARTBackendConfig

_SCOPE_TOKEN = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]+")
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+=*")
_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


@dataclass(frozen=True, repr=False)
class SmartCredential:
    """Secret-bearing custody record, never a public result or audit artifact."""

    access_token: str
    refresh_token: str | None
    expires_at: float
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        try:
            if (
                not isinstance(self.access_token, str)
                or len(self.access_token) > 8192
                or not _BEARER_TOKEN.fullmatch(self.access_token)
                or not _finite(self.expires_at)
                or self.expires_at <= 0
                or not isinstance(self.scopes, frozenset)
                or _scopes(" ".join(sorted(self.scopes))) != self.scopes
                or not _valid_refresh(self.refresh_token)
            ):
                raise ValueError
        except Exception:
            raise ValueError("invalid_credential") from None


class CredentialSlot(Protocol):
    """Exclusive custody view; revocation is terminal and erases both secrets.

    Replacement is atomic, including scopes, expiry and the refresh token. The
    adapter must not roll back revocation when the transaction exits. Dispatch,
    refresh and external revocation must all use the same exclusion mechanism.
    """

    @property
    def revoked(self) -> bool:
        """Whether this handle has been permanently revoked."""
        ...

    def read(self) -> SmartCredential | None:
        """Read the protected record, or None for an empty/revoked handle."""
        ...

    def replace(self, credential: SmartCredential) -> None:
        """Atomically replace the record; refuse a revoked handle."""
        ...

    def revoke(self) -> None:
        """Erase the record and permanently disable the handle."""
        ...


class CredentialCustody(Protocol):
    """Minimal adapter contract for existing or application-owned custody."""

    def transaction(self, handle: str) -> AbstractContextManager[CredentialSlot]:
        """Hold exclusive access through token transport and dispatch completion."""
        ...


class TokenEndpointTransport(Protocol):
    """Caller-configured token endpoint with bounded I/O and no secret logging."""

    def __call__(self, form: Mapping[str, str]) -> Mapping[str, object]:
        """Submit a grant; return its decoded success or OAuth error object."""
        ...


@dataclass(frozen=True)
class RefreshReport:
    """Value-free outcome; findings and counts never contain scope values."""

    code: str
    findings: tuple[str, ...] = ()
    dropped_scope_count: int = 0

    @property
    def usable(self) -> bool:
        """Whether the protected credential is valid at this check."""
        return self.code in {"current", "refreshed", "acquired", "dispatched"}

    def to_dict(self) -> dict[str, object]:
        """Return controlled codes and counts only."""
        return {
            "code": self.code,
            "findings": list(self.findings),
            "dropped_scope_count": self.dropped_scope_count,
        }


class _Rejected(Exception):
    pass


def _scopes(value: object) -> frozenset[str]:
    if not isinstance(value, str) or len(value) > 8192:
        raise _Rejected("malformed_response")
    parts = value.split(" ") if value else []
    if len(parts) > 128 or any(not _SCOPE_TOKEN.fullmatch(p) for p in parts):
        raise _Rejected("malformed_response")
    return frozenset(parts)


def _atoms(scopes: frozenset[str]) -> frozenset[tuple[str, ...]]:
    atoms: set[tuple[str, ...]] = set()
    for scope in scopes:
        try:
            atoms.update(parse_smart_scope(scope).atoms())
        except ValueError:
            # Other OAuth scopes are exact matches. No wildcard, SMART v1 or
            # granular-scope implication is invented at this boundary.
            atoms.add((scope,))
    return frozenset(atoms)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_refresh(value: object) -> TypeGuard[str | None]:
    return value is None or (
        isinstance(value, str)
        and 0 < len(value) <= 8192
        and all(33 <= ord(c) <= 126 for c in value)
    )


class SmartCredentialRefresher:
    """Refresh and dispatch through injected custody, transport and clock.

    Args:
        custody: Adapter implementing exclusive, atomic custody transactions.
        transport: Explicitly configured secret-bearing token endpoint callable.
        requested_scopes: Original OAuth scope request, pinned for this instance.
        clock: Trusted epoch-seconds clock shared with credential expiry.
        refresh_margin: Refresh at or before expiry minus this many seconds.
        max_lifetime: Maximum accepted expires_in (default one day).
        backend_config: Existing SMART backend configuration for JWT grants.
        assertion_builder: Optional signer override for tests or a trusted signer.
        sender: Trusted callback bound by the host, never by a handle holder.

    A failed grant permanently revokes its handle, so neither this instance nor
    another adapter client can retry the superseded refresh secret. Reauthorization
    requires a fresh handle. This Python server-side slice does not add app launch
    or an on-device OAuth client.
    """

    def __init__(
        self,
        custody: CredentialCustody,
        transport: TokenEndpointTransport,
        *,
        requested_scopes: str,
        clock: Callable[[], float] = time.time,
        refresh_margin: float = 60,
        max_lifetime: int = 86400,
        backend_config: SMARTBackendConfig | None = None,
        assertion_builder: Callable[[SMARTBackendConfig], str] | None = None,
        sender: Callable[[str], None] | None = None,
    ) -> None:
        try:
            scopes = _scopes(requested_scopes)
            if not scopes:
                raise ValueError
            if not _finite(refresh_margin) or refresh_margin < 0:
                raise ValueError
            if type(max_lifetime) is not int or not 1 <= max_lifetime <= 86400:
                raise ValueError
            if refresh_margin >= max_lifetime:
                raise ValueError
            if assertion_builder is not None and backend_config is None:
                raise ValueError
        except Exception:
            raise ValueError("invalid_refresh_configuration") from None
        self._custody = custody
        self._transport = transport
        self._requested = scopes
        self._clock = clock
        self._margin = refresh_margin
        self._max_lifetime = max_lifetime
        self._backend_config = backend_config
        self._assertion_builder = assertion_builder
        self._sender = sender
        self._failed: set[str] = set()

    def acquire(self, handle: str) -> RefreshReport:
        """Acquire a backend credential into a caller-created empty handle."""
        return self._run(handle, acquire=True)

    def ensure(self, handle: str) -> RefreshReport:
        """Refresh a near-expiry credential, or return a value-free refusal."""
        return self._run(handle)

    def dispatch(
        self,
        handle: str,
        *,
        required_scopes: str,
    ) -> RefreshReport:
        """Refresh, authorize and call a trusted sender with a bearer header.

        The constructor-bound sender's return value is discarded. Exceptions
        become a fixed code; no automatic effect retry is attempted. Audience
        binding and other write guards remain the responsibility of the existing
        custody adapter and the application, which must bind this sender to the
        reviewed target.
        """
        try:
            required = _scopes(required_scopes)
            if not required:
                raise ValueError
        except Exception:
            return RefreshReport("invalid_required_scopes")
        if self._sender is None:
            return RefreshReport("dispatch_unavailable")
        return self._run(handle, required=required, sender=self._sender)

    def _run(
        self,
        handle: str,
        *,
        acquire: bool = False,
        required: frozenset[str] = frozenset(),
        sender: Callable[[str], None] | None = None,
    ) -> RefreshReport:
        if handle in self._failed:
            return RefreshReport("revoked")
        try:
            with self._custody.transaction(handle) as slot:
                try:
                    if slot.revoked:
                        return RefreshReport("revoked")
                    credential = slot.read()
                    if credential is None and not acquire:
                        return RefreshReport("unavailable")
                    if credential is not None and acquire:
                        return RefreshReport("already_acquired")
                    now = self._now()
                    if credential is not None and not _atoms(
                        credential.scopes
                    ).issubset(_atoms(self._requested)):
                        raise _Rejected("scope_escalation")
                    if (
                        credential is None
                        or credential.expires_at - now <= self._margin
                    ):
                        credential = self._grant(credential, now)
                        slot.replace(credential)
                        code = "acquired" if acquire else "refreshed"
                    else:
                        code = "current"
                    checked = self._now()
                    if checked < now:
                        raise _Rejected("clock_unavailable")
                    if credential.expires_at <= checked:
                        raise _Rejected("expired")
                    dropped = len(_atoms(self._requested) - _atoms(credential.scopes))
                    findings = ("scope_narrowed",) if dropped else ()
                    if required - credential.scopes and not _atoms(required).issubset(
                        _atoms(credential.scopes)
                    ):
                        return RefreshReport("insufficient_scope", findings, dropped)
                    if sender is not None:
                        try:
                            sender(f"Bearer {credential.access_token}")
                        except Exception:
                            return RefreshReport("dispatch_failed", findings, dropped)
                        code = "dispatched"
                    return RefreshReport(code, findings, dropped)
                except Exception as exc:
                    # Permanent local suppression also protects against a broken
                    # adapter. A conforming adapter must persist revocation.
                    self._failed.add(handle)
                    slot.revoke()
                    code = str(exc) if isinstance(exc, _Rejected) else "refresh_failed"
                    return RefreshReport(code)
        except Exception:
            self._failed.add(handle)
            return RefreshReport("custody_unavailable")

    def _now(self) -> float:
        now = self._clock()
        if not _finite(now) or now < 0:
            raise _Rejected("clock_unavailable")
        return now

    def _grant(self, old: SmartCredential | None, started: float) -> SmartCredential:
        form: dict[str, str]
        if old is not None and old.refresh_token:
            form = {
                "grant_type": "refresh_token",
                "refresh_token": old.refresh_token,
                "scope": " ".join(sorted(old.scopes)),
            }
        else:
            if self._backend_config is None:
                raise _Rejected("refresh_unavailable")
            form = {
                "grant_type": "client_credentials",
                "scope": " ".join(sorted(self._requested)),
            }
        if self._backend_config is not None:
            builder = self._assertion_builder
            if builder is None:
                from openmed.service.smart_backend import build_client_assertion

                assertion = build_client_assertion(
                    self._backend_config, clock=self._clock
                )
            else:
                assertion = builder(self._backend_config)
            if not isinstance(assertion, str) or not assertion:
                raise _Rejected("assertion_unavailable")
            form.update(
                client_assertion_type=_ASSERTION_TYPE, client_assertion=assertion
            )
        response = self._transport(form)
        if not isinstance(response, Mapping):
            raise _Rejected("malformed_response")
        if "error" in response:
            code = (
                "invalid_grant"
                if response["error"] == "invalid_grant"
                else "grant_failed"
            )
            raise _Rejected(code)
        token = response.get("access_token")
        if (
            not isinstance(token, str)
            or len(token) > 8192
            or not _BEARER_TOKEN.fullmatch(token)
        ):
            raise _Rejected("malformed_response")
        kind = response.get("token_type")
        if not isinstance(kind, str) or kind.lower() != "bearer":
            raise _Rejected("unsupported_token_type")
        lifetime = response.get("expires_in")
        if type(lifetime) is not int or not 1 <= lifetime <= self._max_lifetime:
            raise _Rejected("invalid_lifetime")
        # Require a useful interval beyond the margin to prevent rapid successful
        # refresh loops too. Anchor expiry at request start, not response arrival.
        expires_at = started + lifetime
        if expires_at - self._now() <= self._margin:
            raise _Rejected("invalid_lifetime")
        requested = _scopes(form["scope"])
        granted = _scopes(response["scope"]) if "scope" in response else requested
        if not _atoms(granted).issubset(_atoms(requested)):
            raise _Rejected("scope_escalation")
        refresh = response.get("refresh_token", old.refresh_token if old else None)
        if not _valid_refresh(refresh) or (
            "refresh_token" in response and refresh is None
        ):
            raise _Rejected("malformed_response")
        return SmartCredential(token, refresh, expires_at, granted)
