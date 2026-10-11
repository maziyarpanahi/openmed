"""Explicit SMART token refresh with atomic, caller-supplied credential custody.

This module has no HTTP client, persistence, scheduler or clinical effect path.
Secret-bearing objects are confined to the injected transport/custody boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from openmed.interop.smart_scope_audit import (
    SmartGrantedScopeAudit,
    _granted_scope_tokens,
    audit_granted_smart_scopes,
    smart_scopes_cover,
)

if TYPE_CHECKING:
    from openmed.service.smart_backend import SMARTBackendConfig

__all__ = [
    "SMARTCredential",
    "SMARTCredentialCustody",
    "SMARTRefreshConfig",
    "SMARTRefreshLease",
    "SMARTRefreshReport",
    "SMARTCredentialRefresher",
    "SMARTTokenRequest",
    "SMARTTokenResponse",
    "SMARTTokenValidationError",
    "validate_smart_token_response",
]

_HANDLE = re.compile(r"cred_[0-9a-f]{32}")
_LEASE = re.compile(r"lease_[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ACCESS_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]{1,8192}=*")
_SECRET = re.compile(r"[\x21-\x7e]{1,8192}")
_OAUTH_VALUE = re.compile(r"[\x20-\x7e]{1,8192}")
_MAX_EPOCH = 253402300799
_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_CODES = frozenset(
    {
        "ready",
        "insufficient_scope",
        "invalid_grant",
        "invalid_response",
        "scope_expansion",
        "transport_unavailable",
        "credential_unavailable",
        "custody_unavailable",
        "invalid_clock",
        "binding_mismatch",
        "invalid_request",
    }
)


class SMARTTokenValidationError(ValueError):
    """A controlled token-response failure with no endpoint or secret values."""

    def __init__(self, code: str = "invalid_response") -> None:
        self.code = (
            code
            if type(code) is str
            and code in {"invalid_grant", "invalid_response", "scope_expansion"}
            else "invalid_response"
        )
        super().__init__(self.code)


def _validation_code(error: SMARTTokenValidationError) -> str:
    """Read only an exact local exception's closed, stored diagnostic code."""
    if type(error) is SMARTTokenValidationError:
        code = vars(error).get("code")
        if type(code) is str and code in {
            "invalid_grant",
            "invalid_response",
            "scope_expansion",
        }:
            return code
    return "invalid_response"


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


class _ClockError(ValueError):
    pass


def _clock_now(clock: Callable[[], float]) -> int:
    try:
        value = clock()
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= _MAX_EPOCH
        ):
            raise ValueError
        return math.floor(value)
    except Exception:
        pass
    raise _ClockError("invalid_clock")


@dataclass(frozen=True, repr=False)
class SMARTRefreshConfig:
    """Private token-endpoint binding and bounded refresh policy.

    Args:
        token_endpoint: Caller-configured HTTPS token endpoint; no discovery.
        client_id: Client bound to both custody and the token request.
        grant_type: Either refresh_token or client_credentials.
        requested_scopes: Original permission ceiling, at most 128 scope tokens.
        refresh_margin_seconds: Refresh when remaining lifetime is at this margin.
        max_lifetime_seconds: Explicit token lifetime cap, from 1 to 86400 seconds.
            The default follows SMART Backend Services' 300-second recommendation.
    """

    token_endpoint: str
    client_id: str
    grant_type: str
    requested_scopes: tuple[str, ...]
    refresh_margin_seconds: int = 60
    max_lifetime_seconds: int = 300

    def __post_init__(self) -> None:
        try:
            endpoint = urlsplit(self.token_endpoint)
            valid_endpoint = (
                type(self.token_endpoint) is str
                and len(self.token_endpoint) <= 2048
                and all(0x21 <= ord(c) <= 0x7E for c in self.token_endpoint)
                and endpoint.scheme == "https"
                and bool(endpoint.hostname)
                and endpoint.username is None
                and endpoint.password is None
                and not endpoint.fragment
                and endpoint.port != 0
            )
            scopes = _granted_scope_tokens(self.requested_scopes)
            valid = (
                valid_endpoint
                and type(self.client_id) is str
                and _OAUTH_VALUE.fullmatch(self.client_id) is not None
                and self.grant_type in {"refresh_token", "client_credentials"}
                and bool(scopes)
                and _integer(self.max_lifetime_seconds, 1, 86400)
                and _integer(
                    self.refresh_margin_seconds, 0, self.max_lifetime_seconds - 1
                )
            )
        except Exception:
            valid = False
            scopes = ()
        if not valid:
            raise ValueError("Invalid SMART refresh configuration.")
        object.__setattr__(self, "requested_scopes", scopes)

    @property
    def binding_digest(self) -> str:
        """Return an opaque digest binding custody to this client/grant/endpoint."""
        material = json.dumps(
            [
                self.token_endpoint,
                self.client_id,
                self.grant_type,
                self.requested_scopes,
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("ascii")).hexdigest()


@dataclass(frozen=True, repr=False)
class SMARTCredential:
    """Private credential exchanged only with the trusted custody adapter.

    Args:
        access_token: Bearer secret; never serialize or log this object.
        refresh_token: Optional refresh secret, retained only inside custody.
        expires_at: Conservative absolute expiry in UTC epoch seconds.
        granted_scopes: Validated actual permission tokens, including private filters.
    """

    access_token: str
    refresh_token: str | None
    expires_at: int
    granted_scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            valid = (
                type(self.access_token) is str
                and len(self.access_token) <= 8192
                and _ACCESS_TOKEN.fullmatch(self.access_token) is not None
                and (
                    self.refresh_token is None
                    or (
                        type(self.refresh_token) is str
                        and _OAUTH_VALUE.fullmatch(self.refresh_token) is not None
                    )
                )
                and _integer(self.expires_at, 1, _MAX_EPOCH)
            )
            scopes = _granted_scope_tokens(self.granted_scopes)
        except Exception:
            valid = False
            scopes = ()
        if not valid:
            raise ValueError("Invalid SMART credential.")
        object.__setattr__(self, "granted_scopes", scopes)


@dataclass(frozen=True, repr=False)
class SMARTRefreshLease:
    """Exclusive, quarantined custody snapshot with an atomic replacement token.

    Args:
        handle: Opaque credential handle.
        lease_id: Unique opaque reservation ID, never reused after completion.
        generation: Bounded compare-and-swap version.
        binding_digest: Client/grant/endpoint binding from SMARTRefreshConfig.
        credential: Private current credential, or None for initial backend acquisition.
    """

    handle: str
    lease_id: str
    generation: int
    binding_digest: str
    credential: SMARTCredential | None

    def __post_init__(self) -> None:
        if (
            type(self.handle) is not str
            or _HANDLE.fullmatch(self.handle) is None
            or type(self.lease_id) is not str
            or _LEASE.fullmatch(self.lease_id) is None
            or not _integer(self.generation, 0, 2**63 - 1)
            or type(self.binding_digest) is not str
            or _DIGEST.fullmatch(self.binding_digest) is None
            or (
                self.credential is not None
                and type(self.credential) is not SMARTCredential
            )
        ):
            raise ValueError("Invalid SMART custody lease.")


class SMARTCredentialCustody(Protocol):
    """Atomic storage boundary; the caller implements and enforces this contract.

    reserve quarantines the handle for all borrowers and concurrent refreshers.
    replace commits all credential fields atomically for the exact lease/version.
    release restores an unchanged healthy snapshot without reusing its lease.
    revoke permanently invalidates the handle, all leases and subsequent commits.
    A failed/uncertain operation MUST retain quarantine; no lease timeout may
    silently restore the old secret. Methods must not log secret-bearing objects.
    """

    def reserve(self, handle: str) -> SMARTRefreshLease | None:
        """Quarantine and exclusively reserve the requested handle."""
        ...

    def replace(self, lease: SMARTRefreshLease, credential: SMARTCredential) -> bool:
        """Atomically replace all secrets/metadata for this exact reservation."""
        ...

    def release(self, lease: SMARTRefreshLease) -> bool:
        """End a reservation without changing a healthy credential."""
        ...

    def revoke(self, handle: str) -> bool:
        """Tombstone the handle, clear secrets and invalidate every lease."""
        ...


@dataclass(frozen=True, repr=False)
class SMARTTokenRequest:
    """Private POST endpoint/form passed only to an explicitly injected transport."""

    endpoint: str
    form: Mapping[str, str]

    def __post_init__(self) -> None:
        try:
            form = dict(self.form)
        except Exception:
            form = None
        if form is None:
            raise ValueError("Invalid SMART token request.")
        object.__setattr__(self, "form", MappingProxyType(form))


@dataclass(frozen=True, repr=False)
class SMARTTokenResponse:
    """Private bounded HTTP status/body returned by the injected transport."""

    status_code: int
    body: bytes

    def __post_init__(self) -> None:
        if (
            not _integer(self.status_code, 100, 599)
            or type(self.body) is not bytes
            or len(self.body) > 65536
        ):
            raise SMARTTokenValidationError()


@dataclass(frozen=True)
class SMARTRefreshReport:
    """Value-free outcome; readiness is not clinical-action authorization."""

    code: str
    refreshed: bool = False
    narrowed: bool = False
    requested_count: int = 0
    granted_count: int = 0
    refresh_token_rotated: bool = False
    revocation_confirmed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.code) is not str
            or self.code not in _CODES
            or any(
                type(v) is not bool
                for v in (
                    self.refreshed,
                    self.narrowed,
                    self.refresh_token_rotated,
                    self.revocation_confirmed,
                )
            )
            or any(
                not _integer(n, 0, 128)
                for n in (self.requested_count, self.granted_count)
            )
        ):
            raise ValueError("Invalid SMART refresh report.")

    def to_dict(self) -> dict[str, Any]:
        """Return controlled codes, counts and flags without secret values."""
        return {
            "schema_version": "openmed.smart_refresh.v1",
            "code": self.code,
            "refreshed": self.refreshed,
            "findings": ["scope_narrowed"] if self.narrowed else [],
            "requested_count": self.requested_count,
            "granted_count": self.granted_count,
            "refresh_token_rotated": self.refresh_token_rotated,
            "revocation_confirmed": self.revocation_confirmed,
        }


def _payload(response: SMARTTokenResponse) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SMARTTokenValidationError()
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise SMARTTokenValidationError()

    try:
        if type(response) is not SMARTTokenResponse:
            raise SMARTTokenValidationError()
        value = json.loads(
            response.body.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
        nodes = [(value, 0)]
        visited = 0
        while nodes:
            node, depth = nodes.pop()
            visited += 1
            if depth > 8 or visited > 512:
                raise SMARTTokenValidationError()
            if isinstance(node, dict):
                nodes.extend((v, depth + 1) for v in node.values())
            elif isinstance(node, list):
                nodes.extend((v, depth + 1) for v in node)
        if type(value) is not dict:
            raise SMARTTokenValidationError()
        return value
    except Exception:
        pass
    raise SMARTTokenValidationError()


def validate_smart_token_response(
    response: SMARTTokenResponse,
    *,
    requested_scopes: Iterable[str],
    issued_at: int,
    max_lifetime_seconds: int = 300,
    previous_refresh_token: str | None = None,
    allow_omitted_scope: bool = False,
) -> tuple[SMARTCredential, SmartGrantedScopeAudit]:
    """Validate a bounded token response and return private custody material.

    Args:
        response: Raw response from an explicit token-endpoint transport.
        requested_scopes: Exact scopes sent in this exchange.
        issued_at: Trusted clock at request start; network latency consumes lifetime.
        max_lifetime_seconds: Explicit bounded policy cap, at most 86400 seconds.
        previous_refresh_token: Retain this secret if no rotated token is returned.
        allow_omitted_scope: RFC6749 refresh responses may omit unchanged scope.
            SMART Backend Services requires an explicit scope response.

    Returns:
        The private credential and a value-free permission comparison.

    Raises:
        SMARTTokenValidationError: A fixed failure code, without response values.
    """
    try:
        requested = _granted_scope_tokens(requested_scopes)
        if (
            not requested
            or not _integer(issued_at, 0, _MAX_EPOCH)
            or not _integer(max_lifetime_seconds, 1, 86400)
            or type(allow_omitted_scope) is not bool
        ):
            raise SMARTTokenValidationError()
        payload = _payload(response)
        if "error" in payload:
            raise SMARTTokenValidationError(
                "invalid_grant"
                if payload["error"] == "invalid_grant"
                else "invalid_response"
            )
        if response.status_code != 200:
            raise SMARTTokenValidationError()
        if (
            type(payload.get("token_type")) is not str
            or payload["token_type"].lower() != "bearer"
        ):
            raise SMARTTokenValidationError()
        lifetime = payload.get("expires_in")
        if (
            not isinstance(lifetime, int)
            or not _integer(lifetime, 1, max_lifetime_seconds)
            or issued_at + lifetime > _MAX_EPOCH
        ):
            raise SMARTTokenValidationError()
        if "scope" not in payload and allow_omitted_scope:
            granted = requested
        else:
            scope = payload.get("scope")
            if (
                type(scope) is not str
                or not scope
                or len(scope) > 65536
                or scope.strip() != scope
                or "  " in scope
            ):
                raise SMARTTokenValidationError()
            granted = _granted_scope_tokens(scope.split(" ") if scope else ())
        audit = audit_granted_smart_scopes(
            requested_scopes=requested, granted_scopes=granted
        )
        if audit.expanded:
            raise SMARTTokenValidationError("scope_expansion")
        token = payload.get("access_token")
        if not isinstance(token, str):
            raise SMARTTokenValidationError()
        credential = SMARTCredential(
            token,
            payload.get("refresh_token", previous_refresh_token),
            issued_at + lifetime,
            granted,
        )
        if "refresh_token" in payload and payload["refresh_token"] is None:
            raise SMARTTokenValidationError()
        return credential, audit
    except SMARTTokenValidationError as error:
        failure_code = _validation_code(error)
    except Exception:
        failure_code = "invalid_response"
    raise SMARTTokenValidationError(failure_code)


class SMARTCredentialRefresher:
    """Perform one explicit refresh, atomically rotate secrets, and report scope.

    Args:
        config: Private endpoint/client binding and lifetime policy.
        custody: Caller adapter implementing atomic reservation and revocation.
        transport: Explicit HTTPS POST transport; must bound reads, forbid redirects,
            enforce TLS verification/timeouts, and never log endpoint or form values.
        clock: Caller-configured UTC epoch clock.
        backend_config: Existing SMARTBackendConfig for private-key JWT grants.
        client_assertion_builder: Optional explicit builder; otherwise reuse the
            existing RS384 builder with the injected clock. Required backend config
            must exactly match the refresh endpoint/client identity.
    """

    def __init__(
        self,
        config: SMARTRefreshConfig,
        *,
        custody: SMARTCredentialCustody,
        transport: Callable[[SMARTTokenRequest], SMARTTokenResponse],
        clock: Callable[[], float],
        backend_config: SMARTBackendConfig | None = None,
        client_assertion_builder: Callable[[SMARTBackendConfig], str] | None = None,
    ) -> None:
        try:
            valid = (
                type(config) is SMARTRefreshConfig
                and callable(transport)
                and callable(clock)
                and all(
                    callable(getattr(custody, name, None))
                    for name in ("reserve", "replace", "release", "revoke")
                )
            )
            if backend_config is not None:
                valid = (
                    valid
                    and backend_config.token_url == config.token_endpoint
                    and backend_config.client_id == config.client_id
                )
            if config.grant_type == "client_credentials" and backend_config is None:
                valid = False
            if client_assertion_builder is not None and (
                backend_config is None or not callable(client_assertion_builder)
            ):
                valid = False
        except Exception:
            valid = False
        if not valid:
            raise ValueError("Invalid SMART refresh dependencies.")
        self._config = config
        self._custody = custody
        self._transport = transport
        self._clock = clock
        self._backend = backend_config
        self._builder = client_assertion_builder

    def _revoke(self, handle: str, code: str) -> SMARTRefreshReport:
        try:
            confirmed = self._custody.revoke(handle) is True
        except Exception:
            confirmed = False
        return SMARTRefreshReport(
            code if confirmed else "custody_unavailable", revocation_confirmed=confirmed
        )

    def _report(
        self,
        credential: SMARTCredential,
        required: tuple[str, ...],
        *,
        refreshed: bool = False,
        rotated: bool = False,
    ) -> SMARTRefreshReport:
        audit = audit_granted_smart_scopes(
            requested_scopes=self._config.requested_scopes,
            granted_scopes=credential.granted_scopes,
        )
        return SMARTRefreshReport(
            "ready"
            if smart_scopes_cover(required, credential.granted_scopes)
            else "insufficient_scope",
            refreshed=refreshed,
            narrowed=audit.narrowed,
            requested_count=audit.requested_count,
            granted_count=audit.granted_count,
            refresh_token_rotated=rotated,
        )

    def _form(
        self, lease: SMARTRefreshLease, scopes: tuple[str, ...]
    ) -> Mapping[str, str]:
        form = {"grant_type": self._config.grant_type, "scope": " ".join(scopes)}
        if self._config.grant_type == "refresh_token":
            if lease.credential is None or lease.credential.refresh_token is None:
                raise SMARTTokenValidationError("invalid_grant")
            form["refresh_token"] = lease.credential.refresh_token
        if self._backend is not None:
            if self._builder is None:
                from openmed.service.smart_backend import build_client_assertion

                assertion = build_client_assertion(self._backend, clock=self._clock)
            else:
                assertion = self._builder(self._backend)
            if type(assertion) is not str or _SECRET.fullmatch(assertion) is None:
                raise SMARTTokenValidationError()
            form.update(
                client_assertion_type=_ASSERTION_TYPE, client_assertion=assertion
            )
        else:
            form["client_id"] = self._config.client_id
        return form

    def ensure_ready(
        self, handle: str, *, required_scopes: Iterable[str] = ()
    ) -> SMARTRefreshReport:
        """Refresh once when due and check the actual granted permission union.

        Any uncertain refresh/commit revokes the handle. A lost revocation ack
        reports custody_unavailable and relies on custody retaining quarantine.
        Callers must separately bind action approval and recheck custody at the
        eventual dispatch boundary; this report carries no bearer credential.
        """
        try:
            required = _granted_scope_tokens(required_scopes)
            if type(handle) is not str or _HANDLE.fullmatch(handle) is None:
                return SMARTRefreshReport("invalid_request")
        except Exception:
            return SMARTRefreshReport("invalid_request")
        try:
            lease = self._custody.reserve(handle)
        except Exception:
            return self._revoke(handle, "custody_unavailable")
        if lease is None:
            return SMARTRefreshReport("credential_unavailable")
        if type(lease) is not SMARTRefreshLease:
            return self._revoke(handle, "custody_unavailable")
        if (
            lease.handle != handle
            or lease.binding_digest != self._config.binding_digest
        ):
            return self._revoke(handle, "binding_mismatch")
        current = lease.credential
        try:
            started = _clock_now(self._clock)
        except _ClockError:
            return self._revoke(handle, "invalid_clock")
        if current is not None:
            if not smart_scopes_cover(
                current.granted_scopes, self._config.requested_scopes
            ):
                return self._revoke(handle, "binding_mismatch")
            if current.expires_at > started + self._config.refresh_margin_seconds:
                report = self._report(current, required)
                try:
                    if self._custody.release(lease) is True:
                        after_release = _clock_now(self._clock)
                        if after_release < started:
                            return self._revoke(handle, "invalid_clock")
                        if current.expires_at <= after_release:
                            return self._revoke(handle, "invalid_response")
                        return report
                except _ClockError:
                    return self._revoke(handle, "invalid_clock")
                except Exception:
                    pass
                return self._revoke(handle, "custody_unavailable")
        # Refresh scope cannot grow beyond the previous credential's grant.
        scopes = (
            current.granted_scopes
            if current is not None and self._config.grant_type == "refresh_token"
            else self._config.requested_scopes
        )
        try:
            response = self._transport(
                SMARTTokenRequest(
                    self._config.token_endpoint, self._form(lease, scopes)
                )
            )
        except SMARTTokenValidationError as exc:
            return self._revoke(handle, _validation_code(exc))
        except Exception:
            return self._revoke(handle, "transport_unavailable")
        try:
            credential, _ = validate_smart_token_response(
                response,
                requested_scopes=scopes,
                issued_at=started,
                max_lifetime_seconds=self._config.max_lifetime_seconds,
                previous_refresh_token=current.refresh_token
                if current is not None
                else None,
                allow_omitted_scope=self._config.grant_type == "refresh_token",
            )
            completed = _clock_now(self._clock)
            if completed < started:
                return self._revoke(handle, "invalid_clock")
            if credential.expires_at <= completed + self._config.refresh_margin_seconds:
                return self._revoke(handle, "invalid_response")
        except SMARTTokenValidationError as exc:
            return self._revoke(handle, _validation_code(exc))
        except _ClockError:
            return self._revoke(handle, "invalid_clock")
        rotated = credential.refresh_token is not None and (
            current is None or credential.refresh_token != current.refresh_token
        )
        try:
            if self._custody.replace(lease, credential) is True:
                after_commit = _clock_now(self._clock)
                if after_commit < completed:
                    return self._revoke(handle, "invalid_clock")
                if (
                    credential.expires_at
                    <= after_commit + self._config.refresh_margin_seconds
                ):
                    return self._revoke(handle, "invalid_response")
                return self._report(
                    credential, required, refreshed=True, rotated=rotated
                )
        except _ClockError:
            return self._revoke(handle, "invalid_clock")
        except Exception:
            pass
        return self._revoke(handle, "custody_unavailable")
