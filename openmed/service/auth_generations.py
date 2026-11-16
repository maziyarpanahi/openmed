"""Atomic API-key and JWT verifier generations for the REST service.

A verifier generation is one complete, caller-supplied set of API-key hashes,
JWT verification keys and JWT issuer/audience/lifetime bounds. Generations are
validated before they can be staged or activated, published as one immutable
snapshot, and retired through a bounded overlap window or immediate revocation.
Every authentication decision is evaluated against exactly one generation from
one snapshot, so a request never observes a partially reloaded key set.

The module reuses :func:`openmed.service.auth.authenticate_api_key` and
:func:`openmed.service.auth.authenticate_jwt` for the actual checks. Receipts,
metrics, status objects and errors carry only fixed reason codes, counts,
caller-chosen generation labels, timestamps and digests; they never contain
raw API keys, key hashes, JWK material, tokens or token claims.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any, Optional, Union

from fastapi import Request

from openmed.core.audit import stable_hash

from .auth import (
    APIKeyCredential,
    AuthError,
    AuthPrincipal,
    ErrorResponseFactory,
    ServiceAuth,
    ServiceAuthConfig,
    authenticate_api_key,
    authenticate_jwt,
    invalid_credentials,
    missing_credentials,
    parse_api_key_credentials,
    parse_audiences,
    parse_jwks,
)

AUTH_GENERATION_SCOPE_KEY = "openmed.auth_generation"
DEFAULT_MAX_OVERLAP_SECONDS = 3600.0
DEFAULT_MAX_STAGED_GENERATIONS = 4
DEFAULT_MIN_RSA_MODULUS_BITS = 2048
DEFAULT_JWT_MAX_LIFETIME_SECONDS = 3600.0
MIN_RSA_MODULUS_BITS_FLOOR = 1024
MAX_RSA_MODULUS_BITS = 16384
MAX_JWT_LEEWAY_SECONDS = 300.0
MIN_HS256_SECRET_BYTES = 32

GENERATION_REASON_CODES = frozenset(
    {
        "ok",
        "document_malformed",
        "document_field_unknown",
        "generation_id_invalid",
        "generation_serial_invalid",
        "generation_serial_stale",
        "generation_id_reused",
        "generation_empty",
        "generation_unknown",
        "api_keys_malformed",
        "api_key_hash_invalid",
        "api_key_duplicate",
        "api_key_conflict",
        "jwks_malformed",
        "jwk_unsupported",
        "jwk_kid_missing",
        "jwk_kid_duplicate",
        "jwk_kid_conflict",
        "jwk_private_material",
        "jwk_key_invalid",
        "jwk_key_too_weak",
        "jwt_binding_incomplete",
        "jwt_leeway_invalid",
        "jwt_lifetime_invalid",
        "overlap_out_of_bounds",
        "staged_capacity_exceeded",
        "clock_invalid",
    }
)

_GENERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z", re.ASCII)
_API_KEY_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+\Z", re.ASCII)
_MAX_KID_LENGTH = 256
_MAX_SERIAL = 2**63 - 1
_USED_ID_MEMORY = 256
_PRIVATE_JWK_MEMBERS = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth"})
_JWK_ALGORITHMS = {"oct": "HS256", "RSA": "RS256"}
_DOCUMENT_FIELDS = frozenset({"generation_id", "serial", "api_keys", "jwks", "jwt"})
_JWT_FIELDS = frozenset(
    {"issuer", "audience", "audiences", "leeway_seconds", "max_lifetime_seconds"}
)

Clock = Callable[[], float]
GenerationInput = Union["VerifierGenerationSpec", Mapping[str, Any]]


class GenerationValidationError(ValueError):
    """Verifier generation rejected with one fixed, value-free reason code."""

    def __init__(self, reason: str) -> None:
        """Store the reason code as both the attribute and the message.

        Args:
            reason: One member of ``GENERATION_REASON_CODES``.
        """
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, repr=False)
class VerifierGenerationSpec:
    """Caller-supplied, not yet validated verifier generation.

    Attributes:
        generation_id: Opaque operator label, 1-64 ASCII letters, digits,
            ``.``, ``_``, ``:`` or ``-``.
        serial: Positive integer that must increase on every activation.
        api_keys: Hashed API-key credentials accepted by this generation.
        jwks: Public RSA or symmetric HS256 JWK objects, each with a ``kid``.
        jwt_issuer: Required ``iss`` claim when ``jwks`` is non-empty.
        jwt_audiences: Accepted ``aud`` values; required when ``jwks`` is set.
        jwt_leeway_seconds: Clock-skew leeway between 0 and 300 seconds.
        jwt_max_lifetime_seconds: Maximum ``exp - iat`` span in seconds.
    """

    generation_id: str
    serial: int
    api_keys: Sequence[APIKeyCredential] = ()
    jwks: Sequence[Mapping[str, Any]] = ()
    jwt_issuer: Optional[str] = None
    jwt_audiences: Sequence[str] = ()
    jwt_leeway_seconds: float = 0.0
    jwt_max_lifetime_seconds: Optional[float] = DEFAULT_JWT_MAX_LIFETIME_SECONDS

    def __repr__(self) -> str:
        """Return a representation without credentials or key material."""
        return (
            f"VerifierGenerationSpec(serial={_safe_serial(self.serial)!r}, "
            f"api_key_count={_safe_len(self.api_keys)}, "
            f"jwk_count={_safe_len(self.jwks)})"
        )


@dataclass(frozen=True, repr=False)
class VerifierGeneration:
    """Validated, deeply immutable verifier generation.

    Instances are produced by :func:`validate_verifier_generation`; callers do
    not construct them directly.

    Attributes:
        generation_id: Opaque operator label.
        serial: Activation serial.
        api_keys: Frozen API-key credentials.
        config: Verification-only service auth config used for JWT checks.
    """

    generation_id: str
    serial: int
    api_keys: tuple[APIKeyCredential, ...]
    config: ServiceAuthConfig
    _jwk_material: Mapping[str, Mapping[str, Any]] = field(compare=False)

    @property
    def api_key_count(self) -> int:
        """Return the number of API-key credentials."""
        return len(self.api_keys)

    @property
    def jwk_count(self) -> int:
        """Return the number of JWT verification keys."""
        return len(self.config.jwks)

    def __repr__(self) -> str:
        """Return a representation without credentials or key material."""
        return (
            f"VerifierGeneration(generation_id={self.generation_id!r}, "
            f"serial={self.serial}, api_key_count={self.api_key_count}, "
            f"jwk_count={self.jwk_count})"
        )


@dataclass(frozen=True)
class GenerationPin:
    """Generation that produced one authentication decision.

    Attributes:
        generation_id: Label of the generation that verified the credential.
        serial: Serial of that generation.
        epoch: Registry snapshot epoch the decision was evaluated against.
        role: ``"active"`` or ``"overlap"`` for a retiring generation.
    """

    generation_id: str
    serial: int
    epoch: int
    role: str


@dataclass(frozen=True)
class GenerationDecision:
    """Successful authentication pinned to one verifier generation.

    Attributes:
        principal: Authenticated caller from the reused ``auth`` verifiers.
        pin: Generation, serial, epoch and role that admitted the caller.
    """

    principal: AuthPrincipal = field(repr=False)
    pin: GenerationPin

    @property
    def generation_id(self) -> str:
        """Return the pinned generation label."""
        return self.pin.generation_id

    @property
    def serial(self) -> int:
        """Return the pinned generation serial."""
        return self.pin.serial

    @property
    def epoch(self) -> int:
        """Return the snapshot epoch used for the decision."""
        return self.pin.epoch

    @property
    def role(self) -> str:
        """Return ``"active"`` or ``"overlap"``."""
        return self.pin.role


@dataclass(frozen=True)
class GenerationReceipt:
    """Value-free record of one lifecycle operation.

    Attributes:
        action: ``"stage"``, ``"activate"``, ``"reload"`` or ``"revoke"``.
        accepted: Whether the operation changed the published snapshot.
        reason: ``"ok"`` or a member of ``GENERATION_REASON_CODES``.
        epoch: Snapshot epoch after the operation.
        at: Injected-clock time of the operation, or ``None`` if invalid.
        generation_id: Label of a validated or known generation, else ``None``.
        serial: Serial of a validated or known generation, else ``None``.
        retire_at: End of the previous generation's overlap window, if any.
        api_key_count: API-key credentials in the generation.
        jwk_count: JWT verification keys in the generation.
        digest: Canonical SHA-256 digest over every other field.
    """

    action: str
    accepted: bool
    reason: str
    epoch: int
    at: Optional[float]
    generation_id: Optional[str] = None
    serial: Optional[int] = None
    retire_at: Optional[float] = None
    api_key_count: int = 0
    jwk_count: int = 0
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class GenerationStatus:
    """Value-free view of the published verifier generations.

    Attributes:
        epoch: Current snapshot epoch.
        active_generation_id: Label of the active generation, if any.
        active_serial: Serial of the active generation, if any.
        overlap: ``(generation_id, retire_at)`` for an unexpired retiring
            generation.
        staged: Labels of staged, not yet active generations.
    """

    epoch: int
    active_generation_id: Optional[str]
    active_serial: Optional[int]
    overlap: tuple[tuple[str, float], ...]
    staged: tuple[str, ...]


@dataclass(frozen=True)
class _Overlap:
    generation: VerifierGeneration
    retire_at: float


@dataclass(frozen=True)
class _Snapshot:
    epoch: int
    active: Optional[VerifierGeneration]
    overlap: Optional[_Overlap]
    staged: tuple[VerifierGeneration, ...]


def parse_verifier_generation(payload: Any) -> VerifierGenerationSpec:
    """Parse a verifier generation document with the service auth parsers.

    The document shape is ``{"generation_id", "serial", "api_keys", "jwks",
    "jwt": {"issuer", "audience", "leeway_seconds", "max_lifetime_seconds"}}``.
    ``api_keys`` and ``jwks`` use the same JSON shapes as the
    ``OPENMED_SERVICE_AUTH_API_KEYS`` and ``OPENMED_SERVICE_AUTH_JWKS``
    settings and are parsed by the same functions.

    Args:
        payload: Decoded JSON mapping.

    Returns:
        An unvalidated :class:`VerifierGenerationSpec`.

    Raises:
        GenerationValidationError: If the document shape is malformed.
    """
    if not isinstance(payload, Mapping):
        raise GenerationValidationError("document_malformed")
    if not all(type(key) is str and key in _DOCUMENT_FIELDS for key in payload):
        raise GenerationValidationError("document_field_unknown")
    jwt = payload.get("jwt", {})
    if not isinstance(jwt, Mapping):
        raise GenerationValidationError("document_malformed")
    if not all(type(key) is str and key in _JWT_FIELDS for key in jwt):
        raise GenerationValidationError("document_field_unknown")

    api_keys = _parse_with(parse_api_key_credentials, payload.get("api_keys"))
    if api_keys is None:
        raise GenerationValidationError("api_keys_malformed")
    jwks = _parse_with(lambda raw: parse_jwks(raw, None), payload.get("jwks"))
    if jwks is None:
        raise GenerationValidationError("jwks_malformed")
    audiences = _parse_with(
        parse_audiences, jwt.get("audience", jwt.get("audiences")), encode=False
    )
    issuer = jwt.get("issuer")
    if audiences is None or (issuer is not None and type(issuer) is not str):
        raise GenerationValidationError("jwt_binding_incomplete")
    return VerifierGenerationSpec(
        generation_id=payload.get("generation_id"),  # type: ignore[arg-type]
        serial=payload.get("serial"),  # type: ignore[arg-type]
        api_keys=api_keys,
        jwks=jwks,
        jwt_issuer=issuer,
        jwt_audiences=audiences,
        jwt_leeway_seconds=jwt.get("leeway_seconds", 0.0),
        jwt_max_lifetime_seconds=jwt.get(
            "max_lifetime_seconds", DEFAULT_JWT_MAX_LIFETIME_SECONDS
        ),
    )


def validate_verifier_generation(
    spec: GenerationInput,
    *,
    min_rsa_modulus_bits: int = DEFAULT_MIN_RSA_MODULUS_BITS,
) -> VerifierGeneration:
    """Validate one complete verifier generation and freeze its contents.

    Args:
        spec: A :class:`VerifierGenerationSpec` or a generation document.
        min_rsa_modulus_bits: Smallest accepted RSA modulus size.

    Returns:
        A deeply immutable :class:`VerifierGeneration`.

    Raises:
        GenerationValidationError: If the generation is incomplete, malformed
            or internally conflicting.
    """
    if isinstance(spec, Mapping):
        spec = parse_verifier_generation(spec)
    if not isinstance(spec, VerifierGenerationSpec):
        raise GenerationValidationError("document_malformed")
    generation_id = spec.generation_id
    if type(generation_id) is not str or not _GENERATION_ID.match(generation_id):
        raise GenerationValidationError("generation_id_invalid")
    if _safe_serial(spec.serial) is None:
        raise GenerationValidationError("generation_serial_invalid")
    if not _is_sequence(spec.api_keys):
        raise GenerationValidationError("api_keys_malformed")
    if not _is_sequence(spec.jwks):
        raise GenerationValidationError("jwks_malformed")
    if not spec.api_keys and not spec.jwks:
        raise GenerationValidationError("generation_empty")

    api_keys = _validate_api_keys(spec.api_keys)
    jwks = _validate_jwks(spec.jwks, min_rsa_modulus_bits=min_rsa_modulus_bits)
    leeway = _finite_number(spec.jwt_leeway_seconds)
    if leeway is None or not 0 <= leeway <= MAX_JWT_LEEWAY_SECONDS:
        raise GenerationValidationError("jwt_leeway_invalid")
    lifetime = _finite_number(spec.jwt_max_lifetime_seconds)
    if lifetime is None or lifetime <= 0:
        raise GenerationValidationError("jwt_lifetime_invalid")
    issuer, audiences = _validate_jwt_binding(spec, required=bool(jwks))

    config = ServiceAuthConfig(
        enabled=True,
        api_keys=api_keys,
        jwks=tuple(jwks.values()),
        jwt_issuer=issuer,
        jwt_audiences=audiences,
        jwt_leeway_seconds=leeway,
        jwt_max_lifetime_seconds=lifetime,
    )
    return VerifierGeneration(
        generation_id=generation_id,
        serial=spec.serial,
        api_keys=api_keys,
        config=config,
        _jwk_material=MappingProxyType(jwks),
    )


class VerifierView:
    """Authentication view pinned to one immutable registry snapshot."""

    __slots__ = ("_registry", "_snapshot")

    def __init__(
        self, registry: VerifierGenerationRegistry, snapshot: _Snapshot
    ) -> None:
        """Bind the view to one snapshot of ``registry``.

        Args:
            registry: Registry that supplies the clock and metrics.
            snapshot: Immutable snapshot every decision in this view uses.
        """
        self._registry = registry
        self._snapshot = snapshot

    @property
    def epoch(self) -> int:
        """Return the pinned snapshot epoch."""
        return self._snapshot.epoch

    @property
    def active_generation_id(self) -> Optional[str]:
        """Return the active generation label in the pinned snapshot."""
        active = self._snapshot.active
        return None if active is None else active.generation_id

    def authenticate_api_key(self, api_key: str) -> GenerationDecision:
        """Authenticate a raw API key against the pinned generations.

        Args:
            api_key: Raw API key presented by the caller.

        Returns:
            The decision pinned to the generation that accepted the key.

        Raises:
            AuthError: If no pinned generation accepts the key.
        """
        return self._decide(api_key, bearer=False)

    def authenticate_bearer(self, token: str) -> GenerationDecision:
        """Authenticate a JWT bearer token against the pinned generations.

        Args:
            token: Compact JWS bearer token.

        Returns:
            The decision pinned to the generation that verified the token.

        Raises:
            AuthError: If no pinned generation verifies the token.
        """
        return self._decide(token, bearer=True)

    def _decide(self, credential: str, *, bearer: bool) -> GenerationDecision:
        registry = self._registry
        now = registry._now()
        if now is None:
            registry._count("decision_clock_invalid")
            raise invalid_credentials()
        candidates = _candidates(self._snapshot, now)
        if not candidates:
            registry._count("decision_no_active_generation")
            raise invalid_credentials()
        principal = None
        if type(credential) is str and credential:
            for generation, role in candidates:
                principal = _verify(generation, credential, bearer=bearer, now=now)
                if principal is not None:
                    registry._count(f"decision_{role}")
                    pin = GenerationPin(
                        generation_id=generation.generation_id,
                        serial=generation.serial,
                        epoch=self._snapshot.epoch,
                        role=role,
                    )
                    return GenerationDecision(principal=principal, pin=pin)
        registry._count("decision_rejected")
        raise invalid_credentials()

    def __repr__(self) -> str:
        """Return a representation without credentials or key material."""
        return (
            f"VerifierView(epoch={self.epoch}, "
            f"active_generation_id={self.active_generation_id!r})"
        )


class VerifierGenerationRegistry:
    """Caller-controlled lifecycle for API-key and JWT verifier generations.

    Lifecycle operations are serialized by a lock and publish a new immutable
    snapshot with one reference assignment. Readers take one snapshot per
    decision, so they see either the old or the new complete generation set.
    At most one previous generation stays accepted during a bounded overlap.
    Rejected operations leave the published snapshot unchanged.

    Args:
        clock: Injected wall clock returning POSIX seconds.
        max_overlap_seconds: Upper bound for any activation overlap window.
        max_staged: Maximum number of staged, inactive generations.
        min_rsa_modulus_bits: Smallest accepted RSA modulus size.

    Raises:
        ValueError: If a registry bound is invalid.
    """

    def __init__(
        self,
        *,
        clock: Clock = time.time,
        max_overlap_seconds: float = DEFAULT_MAX_OVERLAP_SECONDS,
        max_staged: int = DEFAULT_MAX_STAGED_GENERATIONS,
        min_rsa_modulus_bits: int = DEFAULT_MIN_RSA_MODULUS_BITS,
    ) -> None:
        """Create an empty registry; see the class docstring for arguments."""
        overlap = _finite_number(max_overlap_seconds)
        if overlap is None or overlap < 0:
            raise ValueError("max_overlap_seconds_invalid")
        if type(max_staged) is not int or max_staged < 1:
            raise ValueError("max_staged_invalid")
        if (
            type(min_rsa_modulus_bits) is not int
            or not MIN_RSA_MODULUS_BITS_FLOOR
            <= min_rsa_modulus_bits
            <= MAX_RSA_MODULUS_BITS
        ):
            raise ValueError("min_rsa_modulus_bits_invalid")
        self._clock = clock
        self._max_overlap_seconds = overlap
        self._max_staged = max_staged
        self._min_rsa_modulus_bits = min_rsa_modulus_bits
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics: Counter[str] = Counter()
        self._snapshot = _Snapshot(epoch=0, active=None, overlap=None, staged=())
        self._high_water_serial = 0
        self._used_ids: deque[str] = deque(maxlen=_USED_ID_MEMORY)

    def view(self) -> VerifierView:
        """Return an authentication view pinned to the current snapshot."""
        return VerifierView(self, self._snapshot)

    def authenticate_api_key(self, api_key: str) -> GenerationDecision:
        """Authenticate a raw API key against the current snapshot.

        Args:
            api_key: Raw API key presented by the caller.

        Returns:
            The decision pinned to the generation that accepted the key.

        Raises:
            AuthError: If no current generation accepts the key.
        """
        return self.view().authenticate_api_key(api_key)

    def authenticate_bearer(self, token: str) -> GenerationDecision:
        """Authenticate a JWT bearer token against the current snapshot.

        Args:
            token: Compact JWS bearer token.

        Returns:
            The decision pinned to the generation that verified the token.

        Raises:
            AuthError: If no current generation verifies the token.
        """
        return self.view().authenticate_bearer(token)

    def stage(self, spec: GenerationInput) -> GenerationReceipt:
        """Validate and stage a generation without activating it.

        Args:
            spec: A :class:`VerifierGenerationSpec` or a generation document.

        Returns:
            A receipt; ``accepted`` is false when the generation was rejected.
        """
        with self._lock:
            now = self._now()
            if now is None:
                return self._reject("stage", "clock_invalid", None)
            generation, reason = self._validated(spec)
            if generation is None:
                return self._reject("stage", reason, now)
            snapshot = self._pruned(now)
            reason = self._admission_reason(generation, snapshot)
            if reason is None and len(snapshot.staged) >= self._max_staged:
                reason = "staged_capacity_exceeded"
            if reason is not None:
                return self._reject("stage", reason, now, generation)
            self._used_ids.append(generation.generation_id)
            self._publish(replace(snapshot, staged=(*snapshot.staged, generation)))
            return self._accept("stage", now, generation)

    def activate(
        self, generation_id: str, *, overlap_seconds: float = 0.0
    ) -> GenerationReceipt:
        """Atomically activate a staged generation.

        Args:
            generation_id: Label of a staged generation.
            overlap_seconds: How long the previously active generation keeps
                being accepted; ``0`` cuts over immediately.

        Returns:
            A receipt; ``accepted`` is false when activation was rejected.
        """
        with self._lock:
            now = self._now()
            if now is None:
                return self._reject("activate", "clock_invalid", None)
            snapshot = self._pruned(now)
            generation = next(
                (
                    staged
                    for staged in snapshot.staged
                    if staged.generation_id == generation_id
                ),
                None,
            )
            if generation is None:
                return self._reject("activate", "generation_unknown", now)
            return self._activate(
                "activate", generation, snapshot, overlap_seconds, now
            )

    def reload(
        self, spec: GenerationInput, *, overlap_seconds: float = 0.0
    ) -> GenerationReceipt:
        """Validate and activate a complete generation in one atomic step.

        Args:
            spec: A :class:`VerifierGenerationSpec` or a generation document.
            overlap_seconds: How long the previously active generation keeps
                being accepted; ``0`` cuts over immediately.

        Returns:
            A receipt; on rejection the last valid generation stays active.
        """
        with self._lock:
            now = self._now()
            if now is None:
                return self._reject("reload", "clock_invalid", None)
            generation, reason = self._validated(spec)
            if generation is None:
                return self._reject("reload", reason, now)
            snapshot = self._pruned(now)
            reason = self._admission_reason(generation, snapshot)
            if reason is not None:
                return self._reject("reload", reason, now, generation)
            return self._activate("reload", generation, snapshot, overlap_seconds, now)

    def revoke(self, generation_id: str) -> GenerationReceipt:
        """Immediately stop accepting a staged, active or overlapping generation.

        Revoking the active generation fails closed: every request is rejected
        until another generation is activated. Revocation does not depend on a
        valid clock reading.

        Args:
            generation_id: Label of a known generation.

        Returns:
            A receipt; ``accepted`` is false for an unknown generation.
        """
        with self._lock:
            now = self._now()
            snapshot = self._snapshot
            target = None
            for generation in (
                snapshot.active,
                None if snapshot.overlap is None else snapshot.overlap.generation,
                *snapshot.staged,
            ):
                if generation is not None and generation.generation_id == generation_id:
                    target = generation
                    break
            if target is None:
                return self._reject("revoke", "generation_unknown", now)
            overlap = snapshot.overlap
            self._publish(
                replace(
                    snapshot,
                    active=None if snapshot.active is target else snapshot.active,
                    overlap=(
                        None
                        if overlap is not None and overlap.generation is target
                        else overlap
                    ),
                    staged=tuple(
                        item for item in snapshot.staged if item is not target
                    ),
                )
            )
            return self._accept("revoke", now, target)

    def status(self) -> GenerationStatus:
        """Return the published generations without key material."""
        snapshot = self._snapshot
        now = self._now()
        overlap: tuple[tuple[str, float], ...] = ()
        current = snapshot.overlap
        if current is not None and now is not None and now < current.retire_at:
            overlap = ((current.generation.generation_id, current.retire_at),)
        active = snapshot.active
        return GenerationStatus(
            epoch=snapshot.epoch,
            active_generation_id=None if active is None else active.generation_id,
            active_serial=None if active is None else active.serial,
            overlap=overlap,
            staged=tuple(item.generation_id for item in snapshot.staged),
        )

    def metrics(self) -> dict[str, int]:
        """Return lifecycle and decision counters keyed by fixed codes."""
        with self._metrics_lock:
            return dict(sorted(self._metrics.items()))

    def _now(self) -> Optional[float]:
        try:
            value = self._clock()
        except Exception:  # noqa: BLE001 - any clock failure fails closed
            return None
        return _finite_number(value)

    def _count(self, name: str) -> None:
        with self._metrics_lock:
            self._metrics[name] += 1

    def _validated(
        self, spec: GenerationInput
    ) -> tuple[Optional[VerifierGeneration], str]:
        reason = "document_malformed"
        try:
            return (
                validate_verifier_generation(
                    spec, min_rsa_modulus_bits=self._min_rsa_modulus_bits
                ),
                "ok",
            )
        except GenerationValidationError as exc:
            reason = exc.reason
        return None, reason

    def _admission_reason(
        self, generation: VerifierGeneration, snapshot: _Snapshot
    ) -> Optional[str]:
        if generation.generation_id in self._used_ids or generation.generation_id in {
            item.generation_id for item in _known(snapshot)
        }:
            return "generation_id_reused"
        if generation.serial <= self._high_water_serial:
            return "generation_serial_stale"
        return None

    def _activate(
        self,
        action: str,
        generation: VerifierGeneration,
        snapshot: _Snapshot,
        overlap_seconds: float,
        now: float,
    ) -> GenerationReceipt:
        overlap = _finite_number(overlap_seconds)
        if overlap is None or not 0 <= overlap <= self._max_overlap_seconds:
            return self._reject(action, "overlap_out_of_bounds", now, generation)
        if generation.serial <= self._high_water_serial:
            return self._reject(action, "generation_serial_stale", now, generation)
        previous = snapshot.active
        retiring = None
        if previous is not None and overlap > 0:
            reason = _conflict_reason(generation, previous)
            if reason is not None:
                return self._reject(action, reason, now, generation)
            retiring = _Overlap(generation=previous, retire_at=now + overlap)
        if generation.generation_id not in self._used_ids:
            self._used_ids.append(generation.generation_id)
        self._high_water_serial = generation.serial
        self._publish(
            _Snapshot(
                epoch=snapshot.epoch,
                active=generation,
                overlap=retiring,
                staged=tuple(
                    item for item in snapshot.staged if item is not generation
                ),
            )
        )
        retire_at = None if retiring is None else retiring.retire_at
        return self._accept(action, now, generation, retire_at=retire_at)

    def _pruned(self, now: float) -> _Snapshot:
        snapshot = self._snapshot
        if snapshot.overlap is not None and now >= snapshot.overlap.retire_at:
            return replace(snapshot, overlap=None)
        return snapshot

    def _publish(self, snapshot: _Snapshot) -> None:
        # One reference assignment: readers see the old or the new snapshot.
        self._snapshot = replace(snapshot, epoch=self._snapshot.epoch + 1)

    def _accept(
        self,
        action: str,
        now: Optional[float],
        generation: VerifierGeneration,
        *,
        retire_at: Optional[float] = None,
    ) -> GenerationReceipt:
        self._count(f"{action}_accepted")
        return _receipt(
            action, True, "ok", self._snapshot.epoch, now, generation, retire_at
        )

    def _reject(
        self,
        action: str,
        reason: str,
        now: Optional[float],
        generation: Optional[VerifierGeneration] = None,
    ) -> GenerationReceipt:
        self._count(f"{action}_rejected:{reason}")
        return _receipt(
            action, False, reason, self._snapshot.epoch, now, generation, None
        )


class GenerationalServiceAuth(ServiceAuth):
    """Service auth middleware that verifies through a generation registry.

    Route scopes, exemptions and failure rate limits come from ``config``;
    API keys and JWKS come only from ``registry``. Each authenticated request
    carries its :class:`GenerationPin` in ``request.state.auth_generation``
    and ``request.scope[AUTH_GENERATION_SCOPE_KEY]``.

    Args:
        config: Base service auth settings; its static keys are ignored.
        registry: Verifier generation registry.
        error_response: Error envelope factory used by the service app.
    """

    def __init__(
        self,
        config: ServiceAuthConfig,
        *,
        registry: VerifierGenerationRegistry,
        error_response: ErrorResponseFactory,
    ) -> None:
        """Create the middleware; see the class docstring for arguments."""
        super().__init__(
            replace(config, enabled=True, api_keys=(), jwks=()),
            error_response=error_response,
        )
        self.registry = registry

    @classmethod
    def replacing(
        cls, existing: ServiceAuth, *, registry: VerifierGenerationRegistry
    ) -> GenerationalServiceAuth:
        """Build a replacement for an app's existing ``ServiceAuth`` instance.

        Assign the result to ``app.state.auth``; the service middleware reads
        that attribute on every request.

        Args:
            existing: Current ``app.state.auth`` instance.
            registry: Verifier generation registry.

        Returns:
            Middleware with the same route policy and error envelope.
        """
        return cls(
            existing.config,
            registry=registry,
            error_response=existing._error_response,
        )

    def _authenticate_request(self, request: Request) -> AuthPrincipal:
        view = self.registry.view()
        api_key = request.headers.get("x-api-key")
        if api_key:
            return self._pin(request, view.authenticate_api_key(api_key))

        authorization = request.headers.get("authorization")
        if not authorization:
            mtls_principal = request.scope.get("openmed.auth")
            if isinstance(mtls_principal, AuthPrincipal):
                return mtls_principal
            raise missing_credentials()

        scheme, _, value = authorization.strip().partition(" ")
        if not value:
            raise missing_credentials()
        scheme = scheme.lower()
        if scheme == "bearer":
            return self._pin(request, view.authenticate_bearer(value.strip()))
        if scheme in {"apikey", "api-key"}:
            return self._pin(request, view.authenticate_api_key(value.strip()))
        raise invalid_credentials()

    @staticmethod
    def _pin(request: Request, decision: GenerationDecision) -> AuthPrincipal:
        request.state.auth_generation = decision.pin
        request.scope[AUTH_GENERATION_SCOPE_KEY] = decision.pin
        return decision.principal


def _candidates(
    snapshot: _Snapshot, now: float
) -> list[tuple[VerifierGeneration, str]]:
    candidates = []
    if snapshot.active is not None:
        candidates.append((snapshot.active, "active"))
    overlap = snapshot.overlap
    if overlap is not None and now < overlap.retire_at:
        candidates.append((overlap.generation, "overlap"))
    return candidates


def _verify(
    generation: VerifierGeneration, credential: str, *, bearer: bool, now: float
) -> Optional[AuthPrincipal]:
    try:
        if bearer:
            return authenticate_jwt(credential, generation.config, now=now)
        return authenticate_api_key(credential, generation.api_keys)
    except (AuthError, ValueError, TypeError, OverflowError):
        return None


def _known(snapshot: _Snapshot) -> tuple[VerifierGeneration, ...]:
    items = [*snapshot.staged]
    if snapshot.active is not None:
        items.append(snapshot.active)
    if snapshot.overlap is not None:
        items.append(snapshot.overlap.generation)
    return tuple(items)


def _conflict_reason(
    generation: VerifierGeneration, previous: VerifierGeneration
) -> Optional[str]:
    previous_keys = {item.key_hash: item for item in previous.api_keys}
    for credential in generation.api_keys:
        earlier = previous_keys.get(credential.key_hash)
        if earlier is not None and earlier != credential:
            return "api_key_conflict"
    for kid, material in generation._jwk_material.items():
        earlier_material = previous._jwk_material.get(kid)
        if earlier_material is not None and dict(earlier_material) != dict(material):
            return "jwk_kid_conflict"
    return None


def _receipt(
    action: str,
    accepted: bool,
    reason: str,
    epoch: int,
    now: Optional[float],
    generation: Optional[VerifierGeneration],
    retire_at: Optional[float],
) -> GenerationReceipt:
    receipt = GenerationReceipt(
        action=action,
        accepted=accepted,
        reason=reason,
        epoch=epoch,
        at=now,
        generation_id=None if generation is None else generation.generation_id,
        serial=None if generation is None else generation.serial,
        retire_at=retire_at,
        api_key_count=0 if generation is None else generation.api_key_count,
        jwk_count=0 if generation is None else generation.jwk_count,
    )
    fields = receipt.to_dict()
    del fields["digest"]
    return replace(receipt, digest=stable_hash(fields))


def _validate_api_keys(
    credentials: Sequence[Any],
) -> tuple[APIKeyCredential, ...]:
    frozen: list[APIKeyCredential] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for credential in credentials:
        if not isinstance(credential, APIKeyCredential):
            raise GenerationValidationError("api_keys_malformed")
        key_id, principal, scopes = (
            credential.key_id,
            credential.principal,
            credential.scopes,
        )
        if (
            type(key_id) is not str
            or not key_id
            or type(principal) is not str
            or not principal
            or not _is_sequence(scopes)
            or not all(type(scope) is str and scope for scope in scopes)
        ):
            raise GenerationValidationError("api_keys_malformed")
        key_hash = credential.key_hash
        if type(key_hash) is not str or not _API_KEY_HASH.match(key_hash):
            raise GenerationValidationError("api_key_hash_invalid")
        if key_id in seen_ids or key_hash in seen_hashes:
            raise GenerationValidationError("api_key_duplicate")
        seen_ids.add(key_id)
        seen_hashes.add(key_hash)
        frozen.append(
            APIKeyCredential(
                key_id=key_id,
                key_hash=key_hash,
                principal=principal,
                scopes=tuple(scopes),
            )
        )
    return tuple(frozen)


def _validate_jwks(
    keys: Sequence[Any], *, min_rsa_modulus_bits: int
) -> dict[str, Mapping[str, Any]]:
    frozen: dict[str, Mapping[str, Any]] = {}
    for key in keys:
        copied = _json_copy(key)
        if copied is None:
            raise GenerationValidationError("jwks_malformed")
        kty = copied.get("kty")
        expected_alg = _JWK_ALGORITHMS.get(kty) if type(kty) is str else None
        if (
            expected_alg is None
            or copied.get("alg", expected_alg) != expected_alg
            or copied.get("use", "sig") != "sig"
        ):
            raise GenerationValidationError("jwk_unsupported")
        kid = copied.get("kid")
        if type(kid) is not str or not kid.strip() or len(kid) > _MAX_KID_LENGTH:
            raise GenerationValidationError("jwk_kid_missing")
        if kid in frozen:
            raise GenerationValidationError("jwk_kid_duplicate")
        if _PRIVATE_JWK_MEMBERS.intersection(copied):
            raise GenerationValidationError("jwk_private_material")
        if kty == "oct":
            secret = _b64url_bytes(copied.get("k"))
            if not secret:
                raise GenerationValidationError("jwk_key_invalid")
            if len(secret) < MIN_HS256_SECRET_BYTES:
                raise GenerationValidationError("jwk_key_too_weak")
        else:
            modulus = _b64url_int(copied.get("n"))
            exponent = _b64url_int(copied.get("e"))
            if modulus is None or exponent is None or exponent < 3 or exponent % 2 == 0:
                raise GenerationValidationError("jwk_key_invalid")
            if modulus.bit_length() < min_rsa_modulus_bits:
                raise GenerationValidationError("jwk_key_too_weak")
        frozen[kid] = MappingProxyType(copied)
    return frozen


def _validate_jwt_binding(
    spec: VerifierGenerationSpec, *, required: bool
) -> tuple[Optional[str], tuple[str, ...]]:
    issuer = spec.jwt_issuer
    audiences: Any = spec.jwt_audiences
    if type(audiences) is str:
        audiences = (audiences,)
    if (issuer is not None and (type(issuer) is not str or not issuer.strip())) or (
        not _is_sequence(audiences)
        or not all(type(item) is str and item.strip() for item in audiences)
    ):
        raise GenerationValidationError("jwt_binding_incomplete")
    if required and (issuer is None or not audiences):
        raise GenerationValidationError("jwt_binding_incomplete")
    return issuer, tuple(audiences)


def _parse_with(
    parser: Callable[[Any], Any], raw: Any, *, encode: bool = True
) -> Optional[tuple[Any, ...]]:
    if raw is None:
        return ()
    parsed = None
    try:
        parsed = tuple(parser(json.dumps(raw, allow_nan=False) if encode else raw))
    except (TypeError, ValueError, RecursionError):
        pass
    return parsed


def _json_copy(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    copied = None
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError, RecursionError):
        pass
    return copied if isinstance(copied, dict) else None


def _b64url_bytes(value: Any) -> Optional[bytes]:
    if type(value) is not str or not _BASE64URL.match(value) or len(value) % 4 == 1:
        return None
    decoded = None
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        pass
    return decoded


def _b64url_int(value: Any) -> Optional[int]:
    decoded = _b64url_bytes(value)
    if not decoded:
        return None
    number = int.from_bytes(decoded, "big")
    return number if number > 0 else None


def _finite_number(value: Any) -> Optional[float]:
    if type(value) not in {int, float}:
        return None
    number = None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        pass
    return number if number is not None and math.isfinite(number) else None


def _safe_serial(value: Any) -> Optional[int]:
    if type(value) is int and 1 <= value <= _MAX_SERIAL:
        return value
    return None


def _safe_len(value: Any) -> int:
    return len(value) if _is_sequence(value) else 0


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
