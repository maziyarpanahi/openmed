"""Injected, content-free runtime revocation checks for local agent authority.

Providers and generation stores are trusted application boundaries. Neither
checkpoint data nor a valid historical signature establishes current authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn, Protocol


class AuthorityKind(str, Enum):
    """Existing authority contracts whose current status must be checked."""

    GRANT = "grant"
    PURPOSE_TICKET = "purpose_ticket"
    DELEGATION = "delegation"


class AuthorityBoundary(str, Enum):
    """Runtime boundaries that require a fresh status lookup."""

    PREVIEW = "preview"
    SENSITIVE_READ = "sensitive_read"
    EFFECT_DISPATCH = "effect_dispatch"
    RESUME = "resume"


class AuthorityReason(str, Enum):
    """Controlled reasons safe to retain without authority or source content."""

    ACTIVE = "authority_active"
    REVOKED = "authority_revoked"
    EXPIRED = "authority_expired"
    GENERATION_CHANGED = "authority_generation_changed"
    UNAVAILABLE = "authority_status_unavailable"
    STALE = "authority_status_stale"
    ROLLBACK = "authority_status_rollback"
    STORE_UNAVAILABLE = "authority_generation_store_unavailable"
    CONTRACT_MISMATCH = "authority_contract_mismatch"
    APPROVAL_MISMATCH = "authority_approval_mismatch"


def _integer(value: object) -> None:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("invalid_authority_integer")


def _digest(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("invalid_authority_digest")


def _hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthorityContract:
    """Digest and exclusive expiry of an existing authority artifact."""

    kind: AuthorityKind
    digest: str
    expires_at: int

    def __post_init__(self) -> None:
        if type(self.kind) is not AuthorityKind:
            raise ValueError("invalid_authority_kind")
        _digest(self.digest)
        _integer(self.expires_at)


@dataclass(frozen=True, slots=True)
class AuthorityStatus:
    """Fresh provider response for exactly one contract digest.

    Revocation increments generation and permanently tombstones this digest.
    A new grant requires a new artifact. ``observed_at`` is provider observation
    time, not the time a cached response was returned.
    """

    kind: AuthorityKind
    digest: str
    generation: int
    revoked: bool
    observed_at: int

    def __post_init__(self) -> None:
        if type(self.kind) is not AuthorityKind or type(self.revoked) is not bool:
            raise ValueError("invalid_authority_status")
        _digest(self.digest)
        _integer(self.generation)
        _integer(self.observed_at)


class AuthorityStatusProvider(Protocol):
    """Resolve current status locally; unknown authority must return ``None``."""

    def get_status(self, kind: AuthorityKind, digest: str) -> AuthorityStatus | None:
        """Return authoritative status, never infer active from a missing record."""


class AuthorityGenerationStore(Protocol):
    """Atomic rollback protection retained independently of run recovery data."""

    def observe(self, status: AuthorityStatus) -> bool:
        """Retain high water and tombstones; reject rollback or resurrection.

        Return exactly ``bool``. An equal generation cannot change revoked
        state. A previously revoked digest can never become active again.
        """


class InMemoryAuthorityGenerationStore:
    """Thread-safe process-local store; restart requires a durable injected store."""

    def __init__(self) -> None:
        self._latest: dict[tuple[AuthorityKind, str], tuple[int, bool]] = {}
        self._lock = threading.Lock()

    def observe(self, status: AuthorityStatus) -> bool:
        """Atomically advance generations without forgetting a tombstone."""
        if type(status) is not AuthorityStatus:
            raise ValueError("invalid_authority_status")
        key = (status.kind, status.digest)
        with self._lock:
            previous = self._latest.get(key)
            if previous is not None:
                generation, revoked = previous
                if (
                    status.generation < generation
                    or (status.generation == generation and status.revoked != revoked)
                    or (revoked and not status.revoked)
                ):
                    return False
            self._latest[key] = (status.generation, status.revoked)
            return True


@dataclass(frozen=True, slots=True)
class AuthorityVersion:
    """Preview generation pinned to one exact authority contract."""

    contract: AuthorityContract
    generation: int

    def __post_init__(self) -> None:
        if type(self.contract) is not AuthorityContract:
            raise ValueError("invalid_authority_contract")
        _integer(self.generation)


@dataclass(frozen=True, slots=True)
class AuthorityBinding:
    """Content-free preview state retained with trusted run/approval state.

    This is a checkpoint contract, not a signed bearer credential. Applications
    must protect its integrity and association with the run and reviewed action.
    """

    versions: tuple[AuthorityVersion, ...]

    def __post_init__(self) -> None:
        if (
            type(self.versions) is not tuple
            or not self.versions
            or any(type(v) is not AuthorityVersion for v in self.versions)
        ):
            raise ValueError("invalid_authority_binding")
        keys = [(v.contract.kind.value, v.contract.digest) for v in self.versions]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate_authority_contract")
        object.__setattr__(
            self,
            "versions",
            tuple(
                sorted(
                    self.versions,
                    key=lambda v: (v.contract.kind.value, v.contract.digest),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize only contract digests, expiries and pinned generations."""
        return {
            "schema_version": "openmed.agent.authority_binding.v1",
            "versions": [
                {
                    "kind": v.contract.kind.value,
                    "digest": v.contract.digest,
                    "expires_at": v.contract.expires_at,
                    "generation": v.generation,
                }
                for v in self.versions
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuthorityBinding:
        """Restore exact state without replacing an older generation with current."""
        try:
            if (
                set(payload) != {"schema_version", "versions"}
                or payload["schema_version"] != "openmed.agent.authority_binding.v1"
                or type(payload["versions"]) is not list
            ):
                raise ValueError
            versions = []
            for item in payload["versions"]:
                if type(item) is not dict or set(item) != {
                    "kind",
                    "digest",
                    "expires_at",
                    "generation",
                }:
                    raise ValueError
                versions.append(
                    AuthorityVersion(
                        AuthorityContract(
                            AuthorityKind(item["kind"]),
                            item["digest"],
                            item["expires_at"],
                        ),
                        item["generation"],
                    )
                )
            return cls(tuple(versions))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("invalid_authority_binding") from None

    def digest(self) -> str:
        """Return a content-free commitment to the complete pinned state."""
        return _hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class AuthorityReceipt:
    """Codes, count and digest only; no artifacts or provider exception text."""

    boundary: AuthorityBoundary
    reason: AuthorityReason
    authority_count: int
    binding_digest: str

    def __post_init__(self) -> None:
        if type(self.boundary) is not AuthorityBoundary:
            raise ValueError("invalid_authority_boundary")
        if type(self.reason) is not AuthorityReason:
            raise ValueError("invalid_authority_reason")
        _integer(self.authority_count)
        _digest(self.binding_digest)

    def to_dict(self) -> dict[str, str | int]:
        """Return controlled diagnostic metadata suitable for audit storage."""
        return {
            "schema_version": "openmed.agent.authority_receipt.v1",
            "boundary": self.boundary.value,
            "reason_code": self.reason.value,
            "authority_count": self.authority_count,
            "binding_digest": self.binding_digest,
        }


class AuthorityDeniedError(ValueError):
    """Fail-closed runtime denial carrying only a content-free receipt."""

    def __init__(self, receipt: AuthorityReceipt) -> None:
        self.receipt = receipt
        self.code = receipt.reason.value
        super().__init__(self.code)


class AuthorityRuntime:
    """Revalidate each authority against fresh status and monotonic high water.

    Args:
        provider: Trusted, injected local status provider.
        generations: Atomic high-water store, independent of restored runs.
        clock: Trusted integer Unix clock; recovery cannot override it.
        max_status_age: Maximum acceptable observation age, default zero.
    """

    def __init__(
        self,
        provider: AuthorityStatusProvider,
        generations: AuthorityGenerationStore,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
        max_status_age: int = 0,
    ) -> None:
        if not callable(getattr(provider, "get_status", None)):
            raise ValueError("invalid_authority_provider")
        if not callable(getattr(generations, "observe", None)) or not callable(clock):
            raise ValueError("invalid_authority_runtime")
        _integer(max_status_age)
        self._provider = provider
        self._generations = generations
        self._clock = clock
        self._max_status_age = max_status_age

    def bind(self, contracts: tuple[AuthorityContract, ...]) -> AuthorityBinding:
        """Capture active generations at preview, after static verification."""
        initial = AuthorityBinding(tuple(AuthorityVersion(c, 0) for c in contracts))
        versions = []
        statuses = []
        now = self._now(initial, AuthorityBoundary.PREVIEW)
        for version in initial.versions:
            status = self._status(
                version.contract, initial, AuthorityBoundary.PREVIEW, now
            )
            if status.revoked:
                self.deny(initial, AuthorityBoundary.PREVIEW, AuthorityReason.REVOKED)
            if now >= version.contract.expires_at:
                self.deny(initial, AuthorityBoundary.PREVIEW, AuthorityReason.EXPIRED)
            versions.append(AuthorityVersion(version.contract, status.generation))
            statuses.append(status)
        self._check_final_time(initial, AuthorityBoundary.PREVIEW, statuses)
        return AuthorityBinding(tuple(versions))

    def check(
        self, binding: AuthorityBinding, boundary: AuthorityBoundary
    ) -> AuthorityReceipt:
        """Perform new lookups; never renew or overwrite preview generations."""
        if (
            type(binding) is not AuthorityBinding
            or type(boundary) is not AuthorityBoundary
        ):
            raise ValueError("invalid_authority_check")
        now = self._now(binding, boundary)
        statuses = []
        for version in binding.versions:
            status = self._status(version.contract, binding, boundary, now)
            if status.revoked:
                self.deny(binding, boundary, AuthorityReason.REVOKED)
            if status.generation < version.generation:
                self.deny(binding, boundary, AuthorityReason.ROLLBACK)
            if status.generation != version.generation:
                self.deny(binding, boundary, AuthorityReason.GENERATION_CHANGED)
            if now >= version.contract.expires_at:
                self.deny(binding, boundary, AuthorityReason.EXPIRED)
            statuses.append(status)
        # Provider/store calls may advance time. Recheck the entire cohort at
        # admission so an earlier response cannot silently age out mid-check.
        self._check_final_time(binding, boundary, statuses)
        return AuthorityReceipt(
            boundary, AuthorityReason.ACTIVE, len(binding.versions), binding.digest()
        )

    def deny(
        self,
        binding: AuthorityBinding,
        boundary: AuthorityBoundary,
        reason: AuthorityReason,
    ) -> NoReturn:
        """Raise a controlled denial, including expiry from a static verifier."""
        raise AuthorityDeniedError(
            AuthorityReceipt(boundary, reason, len(binding.versions), binding.digest())
        ) from None

    def require_contracts(
        self,
        binding: AuthorityBinding,
        contracts: tuple[AuthorityContract, ...],
        boundary: AuthorityBoundary,
    ) -> None:
        """Reject omitted parents or a binding substituted from another artifact."""
        bound = {v.contract for v in binding.versions}
        if not contracts or not set(contracts).issubset(bound):
            self.deny(binding, boundary, AuthorityReason.CONTRACT_MISMATCH)

    def _now(self, binding: AuthorityBinding, boundary: AuthorityBoundary) -> int:
        try:
            now = self._clock()
            _integer(now)
            return now
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            self.deny(binding, boundary, AuthorityReason.UNAVAILABLE)
        raise AssertionError("unreachable")

    def _status(
        self,
        contract: AuthorityContract,
        binding: AuthorityBinding,
        boundary: AuthorityBoundary,
        now: int,
    ) -> AuthorityStatus:
        try:
            status = self._provider.get_status(contract.kind, contract.digest)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            self.deny(binding, boundary, AuthorityReason.UNAVAILABLE)
        if (
            type(status) is not AuthorityStatus
            or status.kind != contract.kind
            or status.digest != contract.digest
        ):
            self.deny(binding, boundary, AuthorityReason.UNAVAILABLE)
        now = self._now(binding, boundary)
        if not 0 <= now - status.observed_at <= self._max_status_age:
            self.deny(binding, boundary, AuthorityReason.STALE)
        try:
            accepted = self._generations.observe(status)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            self.deny(binding, boundary, AuthorityReason.STORE_UNAVAILABLE)
        if type(accepted) is not bool:
            self.deny(binding, boundary, AuthorityReason.STORE_UNAVAILABLE)
        if not accepted:
            self.deny(binding, boundary, AuthorityReason.ROLLBACK)
        return status

    def _check_final_time(
        self,
        binding: AuthorityBinding,
        boundary: AuthorityBoundary,
        statuses: list[AuthorityStatus],
    ) -> None:
        now = self._now(binding, boundary)
        for version, status in zip(binding.versions, statuses, strict=True):
            if not 0 <= now - status.observed_at <= self._max_status_age:
                self.deny(binding, boundary, AuthorityReason.STALE)
            if now >= version.contract.expires_at:
                self.deny(binding, boundary, AuthorityReason.EXPIRED)
