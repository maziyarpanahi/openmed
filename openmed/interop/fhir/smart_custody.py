"""In-memory SMART credential custody for trusted FHIR request dispatchers.

Only an opaque handle crosses into an agent context. A trusted sender, bound
when the custody object is created, receives the bearer header after local
audience, expiry, and scope checks. No token is returned by the public API.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

from openmed.interop.smart_scope_audit import audit_smart_scopes, parse_smart_scope

_BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+={0,}\Z")
_REASONS = frozenset(
    {
        "audience_mismatch",
        "dispatch_failed",
        "expired",
        "handle_collision",
        "insufficient_scope",
        "invalid_audience",
        "invalid_clock",
        "invalid_configuration",
        "invalid_credential",
        "invalid_expiry",
        "invalid_scopes",
        "unknown_handle",
    }
)


class SmartCustodyError(ValueError):
    """A fixed, value-free reason for rejecting a custody operation."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in _REASONS:
            reason_code = "invalid_configuration"
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class ScopeEvidence:
    """Normalized scope names suitable for a value-free action ledger entry."""

    scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", _scopes(self.scopes))

    def to_dict(self) -> dict[str, list[str]]:
        """Export scope evidence without a handle, audience, or credential."""

        return {"scopes": list(self.scopes)}


@dataclass(frozen=True, slots=True, repr=False)
class _Entry:
    access_token: str = field(repr=False)
    refresh_token: str | None = field(repr=False)
    audience: str = field(repr=False)
    expires_at: datetime = field(repr=False)
    scopes: tuple[str, ...] = field(repr=False)


def _audience(value: str) -> str:
    if type(value) is not str:
        raise SmartCustodyError("invalid_audience")
    try:
        parsed = urlsplit(value)
        parsed.port  # Reject malformed ports without including their value in an error.
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(ord(char) < 33 or ord(char) == 127 for char in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise SmartCustodyError("invalid_audience") from None
    return value


def _scopes(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise SmartCustodyError("invalid_scopes")
    try:
        normalized = tuple(sorted({parse_smart_scope(value).name for value in values}))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        normalized = ()
    if not normalized:
        raise SmartCustodyError("invalid_scopes")
    return normalized


def _token(value: str | None, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or _BEARER_TOKEN.fullmatch(value) is None:
        raise SmartCustodyError("invalid_credential")
    return value


class SmartTokenCustody:
    """Hold SMART credentials and dispatch them only through a trusted sender.

    Args:
        sender: Trusted callback receiving the audience and Authorization
            header. It must not expose either argument to agent traces.
        clock: Injectable, timezone-aware clock for deterministic checks.
    """

    def __init__(
        self,
        sender: Callable[[str, str], None],
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not callable(sender) or not callable(clock):
            raise SmartCustodyError("invalid_configuration")
        self._sender = sender
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    def _now(self) -> datetime:
        try:
            now = self._clock()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            now = None
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise SmartCustodyError("invalid_clock")
        return now

    def store(
        self,
        *,
        access_token: str,
        audience: str,
        expires_at: datetime,
        scopes: Iterable[str],
        refresh_token: str | None = None,
    ) -> str:
        """Return a random opaque handle for a scoped, unexpired credential."""

        token = _token(access_token)
        refresh = _token(refresh_token, optional=True)
        target = _audience(audience)
        normalized = _scopes(scopes)
        if (
            type(expires_at) is not datetime
            or expires_at.tzinfo is None
            or expires_at.utcoffset() is None
        ):
            raise SmartCustodyError("invalid_expiry")
        if expires_at <= self._now():
            raise SmartCustodyError("expired")
        for _ in range(3):
            handle = "smart_" + secrets.token_urlsafe(32)
            if handle not in self._entries:
                self._entries[handle] = _Entry(
                    token, refresh, target, expires_at, normalized
                )
                return handle
        raise SmartCustodyError("handle_collision")

    def dispatch(
        self, handle: str, *, audience: str, required_scopes: Iterable[str]
    ) -> ScopeEvidence:
        """Send a bearer header after checks; return only scope evidence.

        The trusted sender is bound at construction and cannot be selected by
        the handle holder. The caller must separately enforce grant, approval,
        and FHIR write controls before invoking this method.
        """

        if type(handle) is not str:
            raise SmartCustodyError("unknown_handle")
        entry = self._entries.get(handle)
        if entry is None:
            raise SmartCustodyError("unknown_handle")
        target = _audience(audience)
        if target != entry.audience:
            raise SmartCustodyError("audience_mismatch")
        if self._now() >= entry.expires_at:
            raise SmartCustodyError("expired")
        required = _scopes(required_scopes)
        if audit_smart_scopes(
            required_scopes=required, requested_scopes=entry.scopes
        ).missing_scopes:
            raise SmartCustodyError("insufficient_scope")
        failed = False
        try:
            self._sender(entry.audience, f"Bearer {entry.access_token}")
        except Exception:
            failed = True
        if failed:
            raise SmartCustodyError("dispatch_failed")
        return ScopeEvidence(required)

    def revoke(self, handle: str) -> None:
        """Remove a handle and its credentials from in-process custody."""

        if type(handle) is not str or self._entries.pop(handle, None) is None:
            raise SmartCustodyError("unknown_handle")


__all__ = ["ScopeEvidence", "SmartCustodyError", "SmartTokenCustody"]
