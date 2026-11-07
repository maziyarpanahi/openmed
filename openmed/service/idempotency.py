"""Idempotent starts for the FHIR Bulk Data and SMART ingestion routes.

A client that loses the response to a start request can retry it with the same
``Idempotency-Key`` header and receive the job the first attempt created, instead
of starting a second export against the source FHIR server and writing duplicate
output. Keys are scoped to the submitting principal and the start route, and
each remembered key is bound to a keyed digest of the request's non-credential
fields, so reusing a key for a different request is refused before any side
effect.

The store never holds the raw key, credentials, paths or URLs. It keeps a keyed
HMAC digest of the scope, a keyed HMAC digest of the request fields, the job id
and the PHI-free initial status payload, for a bounded time and up to a bounded
number of entries per manager.
"""

from __future__ import annotations

import copy
import hmac
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Optional, Union

from .keyed_digest import ServiceDigestKey, encode_digest_fields

IDEMPOTENCY_KEY_HEADER: Final = "Idempotency-Key"
IDEMPOTENCY_REPLAYED_HEADER: Final = "Idempotency-Replayed"
IDEMPOTENCY_KEY_MAX_LENGTH: Final = 255
IDEMPOTENCY_TTL_ENV_VAR: Final = "OPENMED_SERVICE_IDEMPOTENCY_TTL_SECONDS"
IDEMPOTENCY_MAX_KEYS_ENV_VAR: Final = "OPENMED_SERVICE_IDEMPOTENCY_MAX_KEYS"
DEFAULT_IDEMPOTENCY_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_IDEMPOTENCY_MAX_KEYS: Final = 1024
MAX_IDEMPOTENCY_TTL_SECONDS: Final = 7 * 24 * 60 * 60
MAX_IDEMPOTENCY_MAX_KEYS: Final = 100_000
IDEMPOTENCY_SCOPE_DIGEST_DOMAIN: Final = "start-idempotency-scope"
IDEMPOTENCY_REQUEST_DIGEST_DOMAIN: Final = "start-idempotency-request"
CREDENTIAL_REQUEST_FIELDS: Final = frozenset({"private_key_pem"})

# A key that is still being processed blocks retries only briefly, so a start
# that never completes cannot pin its key for the whole retention window.
PENDING_LEASE_SECONDS: Final = 60.0
IN_PROGRESS_RETRY_AFTER_SECONDS: Final = 1

_KEY_RE: Final = re.compile(r"[A-Za-z0-9._:~+/=-]{1,255}\Z")
_ERROR_STATUS: Final = {
    "idempotency_key_invalid": 400,
    "idempotency_key_conflict": 409,
    "idempotency_key_in_progress": 409,
    "idempotency_capacity_exhausted": 503,
}
_ERROR_MESSAGES: Final = {
    "idempotency_key_invalid": (
        "The Idempotency-Key header must be a single value of 1-255 letters, "
        "digits or ._:~+/=- characters."
    ),
    "idempotency_key_conflict": (
        "The Idempotency-Key was already used for a different request. Use a new "
        "key for a new request."
    ),
    "idempotency_key_in_progress": (
        "A request with this Idempotency-Key is still being processed. Retry later."
    ),
    "idempotency_capacity_exhausted": (
        "The service cannot remember another Idempotency-Key right now. Retry later."
    ),
}


class IdempotencyError(Exception):
    """Value-free refusal of a keyed start request.

    Attributes:
        code: Stable reason code safe to return, log or count.
        status_code: HTTP status the service answers with.
        retry_after: Seconds a client should wait before retrying, if any.
    """

    def __init__(self, code: str) -> None:
        if code not in _ERROR_STATUS:
            code = "idempotency_key_invalid"
        self.code = code
        self.status_code = _ERROR_STATUS[code]
        self.retry_after: Optional[int] = (
            IN_PROGRESS_RETRY_AFTER_SECONDS
            if code in {"idempotency_key_in_progress", "idempotency_capacity_exhausted"}
            else None
        )
        super().__init__(_ERROR_MESSAGES[code])


def parse_idempotency_key(values: Sequence[str]) -> Optional[str]:
    """Validate the raw ``Idempotency-Key`` header values of one request.

    The key may be sent bare or as a quoted structured-field string, as in
    ``Idempotency-Key: "8e03978e-40d5-43e8-bc93-6894a57f9324"``.

    Args:
        values: Every value received for the header, in order.

    Returns:
        The key, or ``None`` when the header is absent.

    Raises:
        IdempotencyError: ``idempotency_key_invalid`` for repeated, empty,
            over-long or malformed values. The value is never echoed.
    """
    if not values:
        return None
    if len(values) != 1 or type(values[0]) is not str:
        raise IdempotencyError("idempotency_key_invalid")
    value = values[0].strip(" \t")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if not _KEY_RE.fullmatch(value):
        raise IdempotencyError("idempotency_key_invalid")
    return value


def request_digest_material(
    fields: Mapping[str, Any],
    *,
    exclude: frozenset[str] = CREDENTIAL_REQUEST_FIELDS,
) -> bytes:
    """Encode the non-credential fields of a validated start request.

    Args:
        fields: Field names and validated JSON-compatible values.
        exclude: Credential field names left out of the digest, so retrying
            with rotated credentials still matches the original request.

    Returns:
        Canonical ASCII JSON object bytes with sorted keys.

    Raises:
        TypeError: If a field value is not JSON-compatible.
    """
    selected = {name: value for name, value in fields.items() if name not in exclude}
    return json.dumps(
        selected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


@dataclass(frozen=True)
class IdempotentStartRequest:
    """Digest-only identity of one keyed start request.

    Attributes:
        scope_digest: Keyed digest of the route, owner and idempotency key.
        request_digest: Keyed digest of the non-credential request fields.
    """

    scope_digest: str
    request_digest: str = field(repr=False)

    @classmethod
    def derive(
        cls,
        key: ServiceDigestKey,
        *,
        route: str,
        owner_digest: Optional[Mapping[str, str]],
        idempotency_key: str,
        request_fields: Mapping[str, Any],
    ) -> "IdempotentStartRequest":
        """Derive the stored identity of a keyed start.

        Args:
            key: Deployment digest key shared with job ownership.
            route: Fixed label of the start route, such as ``fhir-bulk-export``.
            owner_digest: Keyed owner digest of the submitting principal, or
                ``None`` when authentication is disabled.
            idempotency_key: Validated client key; see
                :func:`parse_idempotency_key`.
            request_fields: Validated request fields; credentials are excluded.

        Returns:
            The scope and request digests used by :class:`StartIdempotencyStore`.

        Raises:
            ValueError: If *owner_digest* is present but has no digest value.
        """
        owner = None if owner_digest is None else owner_digest.get("value")
        if owner_digest is not None and type(owner) is not str:
            raise ValueError("owner digest is malformed")
        scope = key.digest(
            IDEMPOTENCY_SCOPE_DIGEST_DOMAIN,
            encode_digest_fields(route, owner, idempotency_key),
        )
        request = key.digest(
            IDEMPOTENCY_REQUEST_DIGEST_DOMAIN,
            request_digest_material(request_fields),
        )
        return cls(scope_digest=scope.value, request_digest=request.value)


@dataclass(frozen=True)
class IdempotentReplay:
    """The remembered first response for a repeated keyed start.

    Attributes:
        job_id: Job created by the first request.
        response: Copy of the PHI-free status payload first returned.
    """

    job_id: str
    response: dict[str, Any]


@dataclass(frozen=True, eq=False)
class IdempotencyReservation:
    """Exclusive claim on a key while its first request is processed.

    Attributes:
        request: Identity of the reserved start.
    """

    request: IdempotentStartRequest


@dataclass
class _Entry:
    request_digest: str = field(repr=False)
    expires_at: float
    reservation: Optional[IdempotencyReservation]
    job_id: Optional[str] = None
    response: Optional[dict[str, Any]] = field(default=None, repr=False)


StartOutcome = Union[IdempotentReplay, IdempotencyReservation]


class StartIdempotencyStore:
    """Bounded, expiring in-memory map from keyed starts to created jobs.

    Entries expire ``ttl_seconds`` after their job was created. When the store
    is full, the oldest completed entry is evicted first. All methods are safe
    to call from several threads.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_IDEMPOTENCY_TTL_SECONDS,
        max_entries: int = DEFAULT_IDEMPOTENCY_MAX_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty store.

        Args:
            ttl_seconds: Retention of a completed entry, in seconds.
            max_entries: Maximum number of remembered keys.
            clock: Monotonic seconds source; injectable for tests.

        Raises:
            ValueError: If a bound is not positive and finite.
        """
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a positive finite number")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "StartIdempotencyStore":
        """Build a store from ``OPENMED_SERVICE_IDEMPOTENCY_*`` settings.

        Args:
            environ: Environment mapping; defaults to the process environment.

        Returns:
            A store with the configured retention and capacity.

        Raises:
            ValueError: If a setting is not an integer within its bounds. The
                message names the variable, never its value.
        """
        env = os.environ if environ is None else environ
        return cls(
            ttl_seconds=_bounded_int_env(
                env,
                IDEMPOTENCY_TTL_ENV_VAR,
                DEFAULT_IDEMPOTENCY_TTL_SECONDS,
                MAX_IDEMPOTENCY_TTL_SECONDS,
            ),
            max_entries=_bounded_int_env(
                env,
                IDEMPOTENCY_MAX_KEYS_ENV_VAR,
                DEFAULT_IDEMPOTENCY_MAX_KEYS,
                MAX_IDEMPOTENCY_MAX_KEYS,
            ),
        )

    @property
    def ttl_seconds(self) -> float:
        """Retention of a completed entry, in seconds."""
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        """Maximum number of remembered keys."""
        return self._max_entries

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._entries)

    def reserve(
        self,
        request: IdempotentStartRequest,
        *,
        job_exists: Callable[[str], bool],
    ) -> StartOutcome:
        """Return the remembered job for *request*, or claim its key.

        Args:
            request: Identity of the keyed start.
            job_exists: Whether the manager still holds a job id; entries for
                removed jobs are forgotten instead of replayed.

        Returns:
            An :class:`IdempotentReplay` for a repeated request, otherwise an
            :class:`IdempotencyReservation` the caller must pass to
            :meth:`complete` or :meth:`release`.

        Raises:
            IdempotencyError: ``idempotency_key_conflict`` when the key was
                used for a different request, ``idempotency_key_in_progress``
                while the first request is still processed, and
                ``idempotency_capacity_exhausted`` when no entry can be evicted.
        """
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            entry = self._entries.get(request.scope_digest)
            if (
                entry is not None
                and entry.job_id is not None
                and not job_exists(entry.job_id)
            ):
                del self._entries[request.scope_digest]
                entry = None
            if entry is not None:
                if not hmac.compare_digest(
                    entry.request_digest, request.request_digest
                ):
                    raise IdempotencyError("idempotency_key_conflict")
                if entry.job_id is None or entry.response is None:
                    raise IdempotencyError("idempotency_key_in_progress")
                return IdempotentReplay(
                    job_id=entry.job_id,
                    response=copy.deepcopy(entry.response),
                )
            self._make_room()
            reservation = IdempotencyReservation(request=request)
            self._entries[request.scope_digest] = _Entry(
                request_digest=request.request_digest,
                expires_at=now + min(self._ttl_seconds, PENDING_LEASE_SECONDS),
                reservation=reservation,
            )
            return reservation

    def complete(
        self,
        reservation: IdempotencyReservation,
        *,
        job_id: str,
        response: Mapping[str, Any],
    ) -> None:
        """Remember the job created for a reserved key.

        Args:
            reservation: Claim returned by :meth:`reserve`.
            job_id: Job created for the request.
            response: PHI-free status payload returned to the client.
        """
        with self._lock:
            now = self._clock()
            scope = reservation.request.scope_digest
            entry = self._entries.get(scope)
            if entry is None:
                self._purge_expired(now)
                try:
                    self._make_room()
                except IdempotencyError:
                    return
            elif entry.reservation is not reservation:
                return
            self._entries[scope] = _Entry(
                request_digest=reservation.request.request_digest,
                expires_at=now + self._ttl_seconds,
                reservation=None,
                job_id=job_id,
                response=copy.deepcopy(dict(response)),
            )

    def release(self, reservation: IdempotencyReservation) -> None:
        """Drop a reservation whose request did not create a job.

        Completed entries are kept, so calling this after :meth:`complete` is
        a no-op.

        Args:
            reservation: Claim returned by :meth:`reserve`.
        """
        with self._lock:
            scope = reservation.request.scope_digest
            entry = self._entries.get(scope)
            if entry is not None and entry.reservation is reservation:
                del self._entries[scope]

    def _purge_expired(self, now: float) -> None:
        expired = [
            scope for scope, entry in self._entries.items() if entry.expires_at <= now
        ]
        for scope in expired:
            del self._entries[scope]

    def _make_room(self) -> None:
        while len(self._entries) >= self._max_entries:
            oldest = next(
                (
                    scope
                    for scope, entry in self._entries.items()
                    if entry.reservation is None
                ),
                None,
            )
            if oldest is None:
                raise IdempotencyError("idempotency_capacity_exhausted")
            del self._entries[oldest]


def _bounded_int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    maximum: int,
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip(), 10)
    except ValueError:
        value = 0
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


__all__ = [
    "CREDENTIAL_REQUEST_FIELDS",
    "DEFAULT_IDEMPOTENCY_MAX_KEYS",
    "DEFAULT_IDEMPOTENCY_TTL_SECONDS",
    "IDEMPOTENCY_KEY_HEADER",
    "IDEMPOTENCY_KEY_MAX_LENGTH",
    "IDEMPOTENCY_MAX_KEYS_ENV_VAR",
    "IDEMPOTENCY_REPLAYED_HEADER",
    "IDEMPOTENCY_REQUEST_DIGEST_DOMAIN",
    "IDEMPOTENCY_SCOPE_DIGEST_DOMAIN",
    "IDEMPOTENCY_TTL_ENV_VAR",
    "IdempotencyError",
    "IdempotencyReservation",
    "IdempotentReplay",
    "IdempotentStartRequest",
    "StartIdempotencyStore",
    "StartOutcome",
    "parse_idempotency_key",
    "request_digest_material",
]
