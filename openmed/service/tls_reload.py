"""Versioned TLS and mTLS credential generations with validated hot reload.

:class:`TLSCredentialProvider` owns the credential *generation* that new
connections use. A reload reads a complete candidate from the configured
files, validates it without touching the active generation, and only then
swaps it in with a single reference assignment, so a new connection never
observes a partial credential set. Work that was admitted under the previous
generation keeps it. When the client CA changes, a bounded trust overlap keeps
accepting clients issued by the previous CA until the overlap deadline, after
which that trust generation is retired.

Any validation or activation failure keeps the previous complete generation
active; a failed reload can never weaken client-certificate requirements or
fall back to plaintext. Results, reports, readiness, errors and log records
carry only opaque generation IDs, certificate SHA-256 fingerprints, controlled
states, counts, times and reason codes. They never include file paths, PEM
text, private-key bytes, certificate subjects or SANs.

Reload triggers are local only: call :meth:`TLSCredentialProvider.reload`,
install :func:`install_reload_signal_handler` (``SIGHUP`` by default), or run a
:class:`TLSCredentialWatcher` that polls file metadata.

The module depends on ``cryptography`` only. The gRPC binding lives in
:mod:`openmed.service.grpc_security` and the REST client-trust binding in
:mod:`openmed.service.mtls`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Protocol, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.x509.verification import Store

StrPath = Union[str, "os.PathLike[str]"]
ActivationHook = Callable[["TLSActivationMaterial"], Any]
Clock = Callable[[], datetime]

TLS_REASON_ACTIVATED = "tls_activated"
TLS_REASON_UNCHANGED = "tls_unchanged"
TLS_REASON_SERVING = "tls_serving"
TLS_REASON_INCOMPLETE = "tls_incomplete"
TLS_REASON_CLIENT_AUTH_AMBIGUOUS = "tls_client_auth_ambiguous"
TLS_REASON_UNREADABLE = "tls_unreadable"
TLS_REASON_MALFORMED = "tls_malformed"
TLS_REASON_KEY_MISMATCH = "tls_key_mismatch"
TLS_REASON_CERTIFICATE_EXPIRED = "tls_certificate_expired"
TLS_REASON_CERTIFICATE_NOT_YET_VALID = "tls_certificate_not_yet_valid"
TLS_REASON_CERTIFICATE_EXPIRING = "tls_certificate_expiring"
TLS_REASON_CHAIN_UNTRUSTED = "tls_chain_untrusted"
TLS_REASON_CLIENT_CA_INVALID = "tls_client_ca_invalid"
TLS_REASON_POLICY_INCOMPATIBLE = "tls_policy_incompatible"
TLS_REASON_CANDIDATE_CHANGED = "tls_candidate_changed"
TLS_REASON_RELOAD_IN_PROGRESS = "tls_reload_in_progress"
TLS_REASON_ACTIVATION_FAILED = "tls_activation_failed"
TLS_REASON_CLIENT_TRUST_UNENFORCEABLE = "tls_client_trust_unenforceable"
TLS_REASON_FAILED = "tls_failed"

TLS_REASON_CODES = frozenset(
    {
        TLS_REASON_ACTIVATED,
        TLS_REASON_UNCHANGED,
        TLS_REASON_SERVING,
        TLS_REASON_INCOMPLETE,
        TLS_REASON_CLIENT_AUTH_AMBIGUOUS,
        TLS_REASON_UNREADABLE,
        TLS_REASON_MALFORMED,
        TLS_REASON_KEY_MISMATCH,
        TLS_REASON_CERTIFICATE_EXPIRED,
        TLS_REASON_CERTIFICATE_NOT_YET_VALID,
        TLS_REASON_CERTIFICATE_EXPIRING,
        TLS_REASON_CHAIN_UNTRUSTED,
        TLS_REASON_CLIENT_CA_INVALID,
        TLS_REASON_POLICY_INCOMPATIBLE,
        TLS_REASON_CANDIDATE_CHANGED,
        TLS_REASON_RELOAD_IN_PROGRESS,
        TLS_REASON_ACTIVATION_FAILED,
        TLS_REASON_CLIENT_TRUST_UNENFORCEABLE,
        TLS_REASON_FAILED,
    }
)

TLS_CATEGORY_CERTIFICATE_CHAIN = "certificate_chain"
TLS_CATEGORY_PRIVATE_KEY = "private_key"
TLS_CATEGORY_CLIENT_CA = "client_ca"
TLS_CATEGORY_SERVER_CA = "server_ca"
TLS_CATEGORY_CLIENT_AUTH = "client_auth"
TLS_CATEGORY_RELOAD = "reload"

TLS_CATEGORIES = frozenset(
    {
        TLS_CATEGORY_CERTIFICATE_CHAIN,
        TLS_CATEGORY_PRIVATE_KEY,
        TLS_CATEGORY_CLIENT_CA,
        TLS_CATEGORY_SERVER_CA,
        TLS_CATEGORY_CLIENT_AUTH,
        TLS_CATEGORY_RELOAD,
    }
)

TLS_MODE_TLS = "tls"
TLS_MODE_MTLS = "mtls"
TLS_MODE_CLIENT_TRUST = "client_trust"

RELOAD_OUTCOME_ACTIVATED = "activated"
RELOAD_OUTCOME_UNCHANGED = "unchanged"
RELOAD_OUTCOME_REJECTED = "rejected"

VALIDATION_VALID = "valid"
VALIDATION_INVALID = "invalid"
VALIDATION_NOT_RUN = "not_run"

ACTIVATION_ACTIVATED = "activated"
ACTIVATION_UNCHANGED = "unchanged"
ACTIVATION_NOT_ATTEMPTED = "not_attempted"
ACTIVATION_FAILED = "failed"

DEFAULT_TRUST_OVERLAP_SECONDS = 300.0
MAX_TRUST_OVERLAP_SECONDS = 86_400.0
DEFAULT_MIN_REMAINING_VALIDITY_SECONDS = 300.0
MAX_MIN_REMAINING_VALIDITY_SECONDS = 30 * 86_400.0
MAX_WATCH_INTERVAL_SECONDS = 3_600.0
MAX_CREDENTIAL_FILE_BYTES = 1024 * 1024

_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_HOOK_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")
_GENERATION_PREFIX = "tlsgen-"

logger = logging.getLogger(__name__)


class TLSCredentialError(ValueError):
    """Value-free refusal of TLS credential material.

    Attributes:
        reason: Stable reason code from :data:`TLS_REASON_CODES`.
        category: Controlled credential category, when relevant.
        fingerprint_sha256: Lowercase SHA-256 fingerprint of the offending
            certificate, when one was parsed. Never key material.
    """

    def __init__(
        self,
        reason: str,
        *,
        category: Optional[str] = None,
        fingerprint_sha256: Optional[str] = None,
    ) -> None:
        self.reason = reason if reason in TLS_REASON_CODES else TLS_REASON_FAILED
        self.category = category if category in TLS_CATEGORIES else None
        self.fingerprint_sha256 = _safe_fingerprint(fingerprint_sha256)
        detail = self.reason
        if self.category is not None:
            detail += f" [{self.category}]"
        if self.fingerprint_sha256 is not None:
            detail += f" (certificate sha256:{self.fingerprint_sha256})"
        super().__init__(f"TLS credentials refused: {detail}")


@dataclass(frozen=True, kw_only=True)
class TLSCredentialSource:
    """File locations that define every candidate credential generation.

    Supply ``certificate_chain_file`` and ``private_key_file`` for a server
    listener, optionally with ``client_ca_file`` to require mutual TLS. Supply
    only ``client_ca_file`` for a client-trust generation that verifies client
    certificates at the application layer, as REST mTLS does. Paths are never
    shown in ``repr`` output, results, reports or errors.

    Attributes:
        certificate_chain_file: PEM server certificate, leaf first, optionally
            followed by intermediates.
        private_key_file: Unencrypted PEM private key matching the leaf.
        client_ca_file: PEM bundle of CA certificates trusted to issue client
            certificates.
        server_ca_file: Optional PEM bundle that must issue the served chain.
            When set, a candidate chain that does not lead to one of these CAs
            is refused with ``tls_chain_untrusted``.
        require_client_cert: Whether clients must present a certificate.
            ``None`` derives it from whether a client CA is configured;
            ``True`` without a CA and ``False`` with one are both refused.
    """

    certificate_chain_file: Optional[StrPath] = field(default=None, repr=False)
    private_key_file: Optional[StrPath] = field(default=None, repr=False)
    client_ca_file: Optional[StrPath] = field(default=None, repr=False)
    server_ca_file: Optional[StrPath] = field(default=None, repr=False)
    require_client_cert: Optional[bool] = None


@dataclass(frozen=True, eq=False)
class TLSCredentialGeneration:
    """One complete, validated credential generation and its safe provenance.

    The generation ID is derived from public certificate fingerprints, so the
    same configured material always yields the same ID, including after a
    restart. Private-key bytes are never stored here.

    Attributes:
        generation_id: Opaque, content-derived generation identifier.
        sequence: Activation order within this provider, starting at 1.
        mode: ``"tls"``, ``"mtls"`` or ``"client_trust"``.
        loaded_at: When the material was read and validated.
        require_client_cert: Whether client certificates are required.
        certificate_sha256: Served leaf fingerprint, when serving TLS.
        certificate_not_valid_before: Served leaf ``notBefore``.
        certificate_not_valid_after: Served leaf ``notAfter``.
        client_ca_sha256: Fingerprints of the trusted client CAs.
        client_ca_not_valid_after: Latest ``notAfter`` among the client CAs.
        client_ca_certificates: Parsed public client CA certificates.
        client_ca_pem: Public PEM bytes of the client CA bundle.
        client_trust_store: Verification store built from the client CAs.
    """

    generation_id: str
    sequence: int
    mode: str
    loaded_at: datetime
    require_client_cert: bool
    certificate_sha256: Optional[str] = None
    certificate_not_valid_before: Optional[datetime] = None
    certificate_not_valid_after: Optional[datetime] = None
    client_ca_sha256: tuple[str, ...] = ()
    client_ca_not_valid_after: Optional[datetime] = None
    client_ca_certificates: tuple[x509.Certificate, ...] = field(default=(), repr=False)
    client_ca_pem: bytes = field(default=b"", repr=False)
    client_trust_store: Optional[Store] = field(default=None, repr=False)

    def report(self) -> dict[str, Any]:
        """Return a value-free summary of the generation.

        Returns:
            Generation ID, sequence, mode, fingerprints and validity times.
        """
        return {
            "generation_id": self.generation_id,
            "sequence": self.sequence,
            "mode": self.mode,
            "loaded_at": self.loaded_at.isoformat(),
            "require_client_cert": self.require_client_cert,
            "certificate_sha256": self.certificate_sha256,
            "certificate_not_valid_before": _isoformat(
                self.certificate_not_valid_before
            ),
            "certificate_not_valid_after": _isoformat(self.certificate_not_valid_after),
            "client_ca_sha256": list(self.client_ca_sha256),
            "client_ca_not_valid_after": _isoformat(self.client_ca_not_valid_after),
        }


class TLSActivationMaterial:
    """Validated candidate bytes handed to activation hooks, then cleared.

    Hooks run before the candidate is activated and build transport-specific
    objects, such as grpcio certificate configurations. The material is
    cleared as soon as every hook returns. It refuses to be pickled and its
    ``repr`` never shows bytes.

    Attributes:
        generation: The candidate generation being activated.
        certificate_chain_pem: Served certificate chain, or ``None``.
        private_key_pem: Served private key, or ``None``.
        client_ca_pem: Client CA bundle of the candidate, or ``None``.
        overlap_client_ca_pem: Client CA bundle of the outgoing generation
            when a bounded trust overlap applies, otherwise ``None``.
    """

    __slots__ = (
        "generation",
        "certificate_chain_pem",
        "private_key_pem",
        "client_ca_pem",
        "overlap_client_ca_pem",
    )

    def __init__(
        self,
        generation: TLSCredentialGeneration,
        *,
        certificate_chain_pem: Optional[bytes],
        private_key_pem: Optional[bytes],
        client_ca_pem: Optional[bytes],
        overlap_client_ca_pem: Optional[bytes] = None,
    ) -> None:
        self.generation = generation
        self.certificate_chain_pem = certificate_chain_pem
        self.private_key_pem = private_key_pem
        self.client_ca_pem = client_ca_pem
        self.overlap_client_ca_pem = overlap_client_ca_pem

    def clear(self) -> None:
        """Drop references to every credential byte string."""
        self.certificate_chain_pem = None
        self.private_key_pem = None
        self.client_ca_pem = None
        self.overlap_client_ca_pem = None

    def __repr__(self) -> str:
        return f"TLSActivationMaterial(generation_id={self.generation.generation_id})"

    def __reduce__(self) -> Any:
        raise TypeError("TLS activation material cannot be serialized")


@dataclass(frozen=True, eq=False)
class TLSCredentialSnapshot:
    """The credential state that a new connection or request should use.

    Attributes:
        active: Generation that new connections are served with.
        overlap: Outgoing generation whose client CAs are still trusted for
            new connections until ``overlap_deadline``, or ``None``.
        overlap_deadline: End of the bounded trust overlap, or ``None``.
        artifacts: Objects built by activation hooks for ``active``.
    """

    active: TLSCredentialGeneration
    overlap: Optional[TLSCredentialGeneration]
    overlap_deadline: Optional[datetime]
    artifacts: Mapping[str, Any] = field(repr=False)

    @property
    def trusted_generations(self) -> tuple[TLSCredentialGeneration, ...]:
        """Return the active generation, then the overlap generation if any."""
        if self.overlap is None:
            return (self.active,)
        return (self.active, self.overlap)


@dataclass(frozen=True)
class TLSReloadResult:
    """Typed, value-free outcome of one reload attempt.

    Attributes:
        outcome: ``"activated"``, ``"unchanged"`` or ``"rejected"``.
        reason: Reason code from :data:`TLS_REASON_CODES`.
        generation_id: Generation active after the attempt.
        previous_generation_id: Generation active when the attempt began.
        candidate_generation_id: Validated candidate ID, when validation ran
            far enough to derive one.
        validation_state: ``"valid"``, ``"invalid"`` or ``"not_run"``.
        activation_state: ``"activated"``, ``"unchanged"``,
            ``"not_attempted"`` or ``"failed"``.
        overlap_deadline: End of the trust overlap now in force, if any.
        sequence: Sequence number of the generation active after the attempt.
        completed_at: When the attempt finished.
        category: Controlled credential category for a rejection.
        fingerprint_sha256: Fingerprint of an offending certificate.
    """

    outcome: str
    reason: str
    generation_id: str
    previous_generation_id: str
    candidate_generation_id: Optional[str]
    validation_state: str
    activation_state: str
    overlap_deadline: Optional[datetime]
    sequence: int
    completed_at: datetime
    category: Optional[str] = None
    fingerprint_sha256: Optional[str] = None

    @property
    def activated(self) -> bool:
        """Return whether a new generation became active."""
        return self.outcome == RELOAD_OUTCOME_ACTIVATED

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a value-free mapping.

        Returns:
            Outcome, reason, generation IDs, states, deadline and times.
        """
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "generation_id": self.generation_id,
            "previous_generation_id": self.previous_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "validation_state": self.validation_state,
            "activation_state": self.activation_state,
            "overlap_deadline": _isoformat(self.overlap_deadline),
            "sequence": self.sequence,
            "completed_at": self.completed_at.isoformat(),
            "category": self.category,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


@dataclass(frozen=True)
class TLSReadiness:
    """Whether a complete, allowed credential generation is serving.

    Attributes:
        serving: ``True`` when the active generation is inside its validity
            window and required client trust is enforceable.
        reason: ``tls_serving`` or the reason serving stopped.
        generation_id: Active generation ID.
        mode: Active generation mode.
        client_trust_enforceable: Whether at least one trusted client CA is
            currently valid, or ``True`` when client certificates are not
            required.
        overlap_active: Whether a trust overlap is in force.
        overlap_deadline: End of the trust overlap, if any.
        last_reload_outcome: Outcome of the latest reload attempt, if any.
        last_reload_reason: Reason of the latest reload attempt, if any.
    """

    serving: bool
    reason: str
    generation_id: str
    mode: str
    client_trust_enforceable: bool
    overlap_active: bool
    overlap_deadline: Optional[datetime]
    last_reload_outcome: Optional[str]
    last_reload_reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Return readiness as a value-free mapping.

        Returns:
            Serving flag, reason, generation ID and overlap state.
        """
        return {
            "serving": self.serving,
            "reason": self.reason,
            "generation_id": self.generation_id,
            "mode": self.mode,
            "client_trust_enforceable": self.client_trust_enforceable,
            "overlap_active": self.overlap_active,
            "overlap_deadline": _isoformat(self.overlap_deadline),
            "last_reload_outcome": self.last_reload_outcome,
            "last_reload_reason": self.last_reload_reason,
        }


class SupportsReload(Protocol):
    """Anything that can reload TLS credentials on a local trigger."""

    def reload(self) -> TLSReloadResult:
        """Reload credentials and return the typed outcome."""
        ...


@dataclass(frozen=True, eq=False)
class _ProviderState:
    active: TLSCredentialGeneration
    previous: Optional[TLSCredentialGeneration]
    overlap_deadline: Optional[datetime]
    artifacts: Mapping[str, Any]


@dataclass
class _Candidate:
    generation: TLSCredentialGeneration
    material: TLSActivationMaterial


class TLSCredentialProvider:
    """Own the active credential generation and reload it atomically.

    The constructor loads and validates the configured material and fails
    closed with :class:`TLSCredentialError` if it is incomplete or invalid; a
    restart therefore activates the configured generation only when it is
    complete. :meth:`reload` validates a new candidate without touching the
    active generation and activates it with one reference assignment.
    Concurrent reloads are rejected deterministically with
    ``tls_reload_in_progress`` rather than queued.

    At most two generations are retained: the active one and, during a trust
    overlap, the outgoing one. Older generations, and every activation hook
    artifact that held private-key material, are released on activation.

    Args:
        source: Files that define each candidate generation.
        trust_overlap_seconds: How long clients issued by the outgoing client
            CA stay trusted for new connections after a client CA change.
            ``0`` disables the overlap. Bounded by
            :data:`MAX_TRUST_OVERLAP_SECONDS`.
        min_remaining_validity_seconds: A candidate whose served certificate,
            or whose latest client CA, expires sooner than this is refused
            with ``tls_certificate_expiring``.
        activation_hooks: Named callables that build transport artifacts
            from :class:`TLSActivationMaterial` before activation. A hook that
            raises fails the activation and keeps the previous generation.
        clock: Returns the current time; defaults to UTC wall time.

    Raises:
        TLSCredentialError: If the configuration or initial material is
            refused.
    """

    def __init__(
        self,
        source: TLSCredentialSource,
        *,
        trust_overlap_seconds: float = DEFAULT_TRUST_OVERLAP_SECONDS,
        min_remaining_validity_seconds: float = DEFAULT_MIN_REMAINING_VALIDITY_SECONDS,
        activation_hooks: Optional[Mapping[str, ActivationHook]] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        if not isinstance(source, TLSCredentialSource):
            raise TLSCredentialError(
                TLS_REASON_INCOMPLETE, category=TLS_CATEGORY_CERTIFICATE_CHAIN
            )
        self._source = source
        self._overlap = timedelta(
            seconds=_bounded_seconds(
                trust_overlap_seconds, maximum=MAX_TRUST_OVERLAP_SECONDS
            )
        )
        self._min_remaining = timedelta(
            seconds=_bounded_seconds(
                min_remaining_validity_seconds,
                maximum=MAX_MIN_REMAINING_VALIDITY_SECONDS,
            )
        )
        self._hooks = _validated_hooks(activation_hooks)
        self._clock: Clock = clock if clock is not None else _utc_now
        self._reload_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._last_result: Optional[TLSReloadResult] = None
        self._counts: dict[str, int] = {
            "reload_attempts_total": 0,
            "reload_activated_total": 0,
            "reload_unchanged_total": 0,
            "reload_rejected_total": 0,
            "generations_retired_total": 0,
        }
        self._rejections: dict[str, int] = {}

        now = self._now()
        candidate = self._load_candidate(now, sequence=1)
        try:
            artifacts = self._build_artifacts(candidate.material)
        finally:
            candidate.material.clear()
        self._mode = candidate.generation.mode
        self._require_client_cert = candidate.generation.require_client_cert
        self._state = _ProviderState(
            active=candidate.generation,
            previous=None,
            overlap_deadline=None,
            artifacts=artifacts,
        )
        logger.info(
            "tls_credentials_loaded",
            extra={
                "tls_generation_id": candidate.generation.generation_id,
                "tls_mode": self._mode,
            },
        )

    @property
    def mode(self) -> str:
        """Return the fixed mode: ``"tls"``, ``"mtls"`` or ``"client_trust"``."""
        return self._mode

    @property
    def trust_overlap_seconds(self) -> float:
        """Return the configured trust overlap window in seconds."""
        return self._overlap.total_seconds()

    @property
    def active(self) -> TLSCredentialGeneration:
        """Return the generation that new connections are served with."""
        return self.snapshot().active

    @property
    def last_reload(self) -> Optional[TLSReloadResult]:
        """Return the outcome of the latest reload attempt, if any."""
        return self._last_result

    def snapshot(self, now: Optional[datetime] = None) -> TLSCredentialSnapshot:
        """Return the state new connections should use at ``now``.

        Reading the snapshot retires an overlap whose deadline has passed.

        Args:
            now: Reference time; defaults to the provider clock.

        Returns:
            The active generation, any overlap generation and hook artifacts.
        """
        state = self._current_state(self._now() if now is None else _aware(now))
        return TLSCredentialSnapshot(
            active=state.active,
            overlap=state.previous,
            overlap_deadline=state.overlap_deadline,
            artifacts=state.artifacts,
        )

    def reload(self, source: Optional[TLSCredentialSource] = None) -> TLSReloadResult:
        """Validate the configured material and activate it if it changed.

        Args:
            source: Optional replacement file locations, for deployments that
                stage each generation in a new directory. It becomes the
                configured source only when its material validates; a refused
                candidate leaves both the generation and the source unchanged.
                The mode and client-certificate requirement cannot change.

        Returns:
            A typed, value-free result. A rejected or failed attempt leaves the
            previous generation fully active.
        """
        if source is not None and not isinstance(source, TLSCredentialSource):
            state = self._state
            return self._finish(
                RELOAD_OUTCOME_REJECTED,
                TLS_REASON_INCOMPLETE,
                state=state,
                previous_id=state.active.generation_id,
                validation_state=VALIDATION_INVALID,
                activation_state=ACTIVATION_NOT_ATTEMPTED,
                category=TLS_CATEGORY_RELOAD,
            )
        if not self._reload_lock.acquire(blocking=False):
            state = self._state
            return self._finish(
                RELOAD_OUTCOME_REJECTED,
                TLS_REASON_RELOAD_IN_PROGRESS,
                state=state,
                previous_id=state.active.generation_id,
                validation_state=VALIDATION_NOT_RUN,
                activation_state=ACTIVATION_NOT_ATTEMPTED,
                category=TLS_CATEGORY_RELOAD,
            )
        try:
            return self._reload_locked(source or self._source)
        finally:
            self._reload_lock.release()

    def readiness(self, now: Optional[datetime] = None) -> TLSReadiness:
        """Report whether a complete, allowed generation is serving.

        Args:
            now: Reference time; defaults to the provider clock.

        Returns:
            Value-free readiness for health endpoints and self-checks.
        """
        current = self._now() if now is None else _aware(now)
        snapshot = self.snapshot(current)
        active = snapshot.active
        reason = TLS_REASON_SERVING
        not_before = active.certificate_not_valid_before
        not_after = active.certificate_not_valid_after
        if not_before is not None and current < not_before:
            reason = TLS_REASON_CERTIFICATE_NOT_YET_VALID
        elif not_after is not None and current > not_after:
            reason = TLS_REASON_CERTIFICATE_EXPIRED
        enforceable = True
        if active.require_client_cert:
            enforceable = any(
                _within_validity(certificate, current)
                for generation in snapshot.trusted_generations
                for certificate in generation.client_ca_certificates
            )
            if not enforceable and reason == TLS_REASON_SERVING:
                reason = TLS_REASON_CLIENT_TRUST_UNENFORCEABLE
        last = self._last_result
        return TLSReadiness(
            serving=reason == TLS_REASON_SERVING,
            reason=reason,
            generation_id=active.generation_id,
            mode=active.mode,
            client_trust_enforceable=enforceable,
            overlap_active=snapshot.overlap is not None,
            overlap_deadline=snapshot.overlap_deadline,
            last_reload_outcome=last.outcome if last is not None else None,
            last_reload_reason=last.reason if last is not None else None,
        )

    def metrics(self) -> dict[str, Any]:
        """Return value-free reload counters and generation gauges.

        Returns:
            Attempt, activation, rejection and retirement counts, rejection
            counts per reason code, and the retained generation count.
        """
        state = self._state
        with self._metrics_lock:
            counts: dict[str, Any] = dict(self._counts)
            counts["reload_rejected_by_reason"] = dict(sorted(self._rejections.items()))
        counts["active_generation_sequence"] = state.active.sequence
        counts["overlap_active"] = state.previous is not None
        counts["retained_generations"] = 1 + (state.previous is not None)
        return counts

    def report(self) -> dict[str, Any]:
        """Return a value-free summary for logs, self-checks and evidence.

        Returns:
            Mode, active and overlap generations, readiness, the latest reload
            outcome and counters. Never paths, PEM text, subjects or SANs.
        """
        snapshot = self.snapshot()
        last = self._last_result
        return {
            "mode": self._mode,
            "trust_overlap_seconds": self.trust_overlap_seconds,
            "active": snapshot.active.report(),
            "overlap": (
                snapshot.overlap.report() if snapshot.overlap is not None else None
            ),
            "overlap_deadline": _isoformat(snapshot.overlap_deadline),
            "readiness": self.readiness().to_dict(),
            "last_reload": last.to_dict() if last is not None else None,
            "metrics": self.metrics(),
        }

    def watched_files(self) -> tuple[Path, ...]:
        """Return the configured credential files, for metadata polling only.

        Returns:
            The configured paths. Callers must not log or report them.
        """
        paths = (
            self._source.certificate_chain_file,
            self._source.private_key_file,
            self._source.client_ca_file,
            self._source.server_ca_file,
        )
        return tuple(
            Path(os.fspath(value))
            for value in paths
            if value is not None and str(os.fspath(value)).strip()
        )

    def _reload_locked(self, source: TLSCredentialSource) -> TLSReloadResult:
        now = self._now()
        state = self._current_state(now)
        active = state.active
        previous_id = active.generation_id
        try:
            candidate = self._load_candidate(
                now, sequence=active.sequence + 1, source=source
            )
        except TLSCredentialError as exc:
            return self._finish(
                RELOAD_OUTCOME_REJECTED,
                exc.reason,
                state=state,
                previous_id=previous_id,
                validation_state=VALIDATION_INVALID,
                activation_state=ACTIVATION_NOT_ATTEMPTED,
                category=exc.category,
                fingerprint=exc.fingerprint_sha256,
            )

        generation = candidate.generation
        try:
            if generation.generation_id == active.generation_id:
                self._source = source
                return self._finish(
                    RELOAD_OUTCOME_UNCHANGED,
                    TLS_REASON_UNCHANGED,
                    state=state,
                    previous_id=previous_id,
                    validation_state=VALIDATION_VALID,
                    activation_state=ACTIVATION_UNCHANGED,
                    candidate_id=generation.generation_id,
                )
            if (
                generation.mode != self._mode
                or generation.require_client_cert != self._require_client_cert
            ):
                return self._finish(
                    RELOAD_OUTCOME_REJECTED,
                    TLS_REASON_POLICY_INCOMPATIBLE,
                    state=state,
                    previous_id=previous_id,
                    validation_state=VALIDATION_INVALID,
                    activation_state=ACTIVATION_NOT_ATTEMPTED,
                    candidate_id=generation.generation_id,
                    category=TLS_CATEGORY_CLIENT_AUTH,
                )

            overlap = self._overlap_generation(active, generation)
            if overlap is not None:
                candidate.material.overlap_client_ca_pem = overlap.client_ca_pem
            try:
                artifacts = self._build_artifacts(candidate.material)
            except TLSCredentialError as exc:
                return self._finish(
                    RELOAD_OUTCOME_REJECTED,
                    exc.reason,
                    state=state,
                    previous_id=previous_id,
                    validation_state=VALIDATION_VALID,
                    activation_state=ACTIVATION_FAILED,
                    candidate_id=generation.generation_id,
                    category=exc.category,
                )
        finally:
            candidate.material.clear()

        deadline = now + self._overlap if overlap is not None else None
        new_state = _ProviderState(
            active=generation,
            previous=overlap,
            overlap_deadline=deadline,
            artifacts=artifacts,
        )
        with self._state_lock:
            retired_state = self._state
            self._state = new_state
            self._source = source
        retired = 1 if overlap is None else 0
        if retired_state.previous is not None:
            retired += 1
        with self._metrics_lock:
            self._counts["generations_retired_total"] += retired
        return self._finish(
            RELOAD_OUTCOME_ACTIVATED,
            TLS_REASON_ACTIVATED,
            state=new_state,
            previous_id=previous_id,
            validation_state=VALIDATION_VALID,
            activation_state=ACTIVATION_ACTIVATED,
            candidate_id=generation.generation_id,
        )

    def _overlap_generation(
        self,
        active: TLSCredentialGeneration,
        candidate: TLSCredentialGeneration,
    ) -> Optional[TLSCredentialGeneration]:
        if self._overlap <= timedelta(0) or not active.client_ca_sha256:
            return None
        if set(active.client_ca_sha256) == set(candidate.client_ca_sha256):
            return None
        return active

    def _current_state(self, now: datetime) -> _ProviderState:
        state = self._state
        deadline = state.overlap_deadline
        if state.previous is None or deadline is None or now < deadline:
            return state
        with self._state_lock:
            state = self._state
            deadline = state.overlap_deadline
            if state.previous is not None and deadline is not None and now >= deadline:
                state = replace(state, previous=None, overlap_deadline=None)
                self._state = state
                with self._metrics_lock:
                    self._counts["generations_retired_total"] += 1
                logger.info(
                    "tls_trust_overlap_retired",
                    extra={"tls_generation_id": state.active.generation_id},
                )
        return state

    def _build_artifacts(self, material: TLSActivationMaterial) -> Mapping[str, Any]:
        artifacts: dict[str, Any] = {}
        for name, hook in self._hooks.items():
            try:
                artifacts[name] = hook(material)
            except Exception:
                raise TLSCredentialError(
                    TLS_REASON_ACTIVATION_FAILED, category=TLS_CATEGORY_RELOAD
                ) from None
        return MappingProxyType(artifacts)

    def _finish(
        self,
        outcome: str,
        reason: str,
        *,
        state: _ProviderState,
        previous_id: str,
        validation_state: str,
        activation_state: str,
        candidate_id: Optional[str] = None,
        category: Optional[str] = None,
        fingerprint: Optional[str] = None,
    ) -> TLSReloadResult:
        result = TLSReloadResult(
            outcome=outcome,
            reason=reason,
            generation_id=state.active.generation_id,
            previous_generation_id=previous_id,
            candidate_generation_id=candidate_id,
            validation_state=validation_state,
            activation_state=activation_state,
            overlap_deadline=state.overlap_deadline,
            sequence=state.active.sequence,
            completed_at=self._now(),
            category=category if category in TLS_CATEGORIES else None,
            fingerprint_sha256=_safe_fingerprint(fingerprint),
        )
        with self._metrics_lock:
            self._counts["reload_attempts_total"] += 1
            self._counts[f"reload_{outcome}_total"] += 1
            if outcome == RELOAD_OUTCOME_REJECTED:
                self._rejections[result.reason] = (
                    self._rejections.get(result.reason, 0) + 1
                )
            self._last_result = result
        extra = {
            "tls_reload_outcome": result.outcome,
            "tls_reload_reason": result.reason,
            "tls_generation_id": result.generation_id,
            "tls_previous_generation_id": result.previous_generation_id,
            "tls_category": result.category,
        }
        if outcome == RELOAD_OUTCOME_REJECTED:
            logger.warning("tls_credential_reload", extra=extra)
        else:
            logger.info("tls_credential_reload", extra=extra)
        return result

    def _now(self) -> datetime:
        return _aware(self._clock())

    def _load_candidate(
        self,
        now: datetime,
        *,
        sequence: int,
        source: Optional[TLSCredentialSource] = None,
    ) -> _Candidate:
        if source is None:
            source = self._source
        certificate_path = _configured_path(
            source.certificate_chain_file, TLS_CATEGORY_CERTIFICATE_CHAIN
        )
        key_path = _configured_path(source.private_key_file, TLS_CATEGORY_PRIVATE_KEY)
        client_ca_path = _configured_path(source.client_ca_file, TLS_CATEGORY_CLIENT_CA)
        server_ca_path = _configured_path(source.server_ca_file, TLS_CATEGORY_SERVER_CA)
        mode, require_client_cert = _resolve_mode(
            has_certificate=certificate_path is not None,
            has_key=key_path is not None,
            has_client_ca=client_ca_path is not None,
            has_server_ca=server_ca_path is not None,
            requested=source.require_client_cert,
        )

        files: list[tuple[Path, str, bytes]] = []

        def read(path: Path, category: str) -> bytes:
            data = _read_bounded(path, category)
            files.append((path, category, data))
            return data

        certificate_pem: Optional[bytes] = None
        key_pem: Optional[bytes] = None
        chain_sha256: tuple[str, ...] = ()
        server_ca_sha256: tuple[str, ...] = ()
        leaf: Optional[x509.Certificate] = None
        if certificate_path is not None and key_path is not None:
            certificate_pem = read(certificate_path, TLS_CATEGORY_CERTIFICATE_CHAIN)
            chain = _load_certificates(certificate_pem, TLS_CATEGORY_CERTIFICATE_CHAIN)
            leaf = chain[0]
            chain_sha256 = tuple(certificate_sha256(item) for item in chain)
            _check_validity(
                leaf,
                now,
                category=TLS_CATEGORY_CERTIFICATE_CHAIN,
                min_remaining=self._min_remaining,
            )
            for intermediate in chain[1:]:
                _check_validity(
                    intermediate, now, category=TLS_CATEGORY_CERTIFICATE_CHAIN
                )
            key_pem = read(key_path, TLS_CATEGORY_PRIVATE_KEY)
            _check_key_matches(key_pem, leaf, chain_sha256[0])
            _check_chain_links(chain, chain_sha256[0])
            if server_ca_path is not None:
                server_ca_pem = read(server_ca_path, TLS_CATEGORY_SERVER_CA)
                server_cas = _load_certificates(server_ca_pem, TLS_CATEGORY_SERVER_CA)
                for authority in server_cas:
                    _check_validity(authority, now, category=TLS_CATEGORY_SERVER_CA)
                server_ca_sha256 = tuple(
                    certificate_sha256(item) for item in server_cas
                )
                _check_chain_anchor(chain, server_cas, server_ca_sha256)

        client_ca_pem: Optional[bytes] = None
        client_cas: tuple[x509.Certificate, ...] = ()
        client_ca_sha256: tuple[str, ...] = ()
        store: Optional[Store] = None
        if client_ca_path is not None:
            client_ca_pem = read(client_ca_path, TLS_CATEGORY_CLIENT_CA)
            client_cas = tuple(
                _load_certificates(client_ca_pem, TLS_CATEGORY_CLIENT_CA)
            )
            client_ca_sha256 = tuple(certificate_sha256(item) for item in client_cas)
            for authority, fingerprint in zip(client_cas, client_ca_sha256):
                if not _is_ca_certificate(authority):
                    raise TLSCredentialError(
                        TLS_REASON_CLIENT_CA_INVALID,
                        category=TLS_CATEGORY_CLIENT_CA,
                        fingerprint_sha256=fingerprint,
                    )
                _check_validity(authority, now, category=TLS_CATEGORY_CLIENT_CA)
            latest = max(client_cas, key=lambda item: item.not_valid_after_utc)
            _check_remaining(
                latest,
                now,
                category=TLS_CATEGORY_CLIENT_CA,
                min_remaining=self._min_remaining,
            )
            try:
                store = Store(list(client_cas))
            except (TypeError, ValueError):
                raise TLSCredentialError(
                    TLS_REASON_MALFORMED, category=TLS_CATEGORY_CLIENT_CA
                ) from None

        for path, category, data in files:
            if _digest(_read_bounded(path, category)) != _digest(data):
                raise TLSCredentialError(
                    TLS_REASON_CANDIDATE_CHANGED, category=category
                )

        generation = TLSCredentialGeneration(
            generation_id=_generation_id(
                mode,
                require_client_cert,
                chain_sha256,
                client_ca_sha256,
                server_ca_sha256,
            ),
            sequence=sequence,
            mode=mode,
            loaded_at=now,
            require_client_cert=require_client_cert,
            certificate_sha256=chain_sha256[0] if chain_sha256 else None,
            certificate_not_valid_before=(
                leaf.not_valid_before_utc if leaf is not None else None
            ),
            certificate_not_valid_after=(
                leaf.not_valid_after_utc if leaf is not None else None
            ),
            client_ca_sha256=client_ca_sha256,
            client_ca_not_valid_after=(
                max(item.not_valid_after_utc for item in client_cas)
                if client_cas
                else None
            ),
            client_ca_certificates=client_cas,
            client_ca_pem=client_ca_pem or b"",
            client_trust_store=store,
        )
        return _Candidate(
            generation=generation,
            material=TLSActivationMaterial(
                generation,
                certificate_chain_pem=certificate_pem,
                private_key_pem=key_pem,
                client_ca_pem=client_ca_pem,
            ),
        )


class TLSCredentialWatcher:
    """Poll credential file metadata and reload when it changes.

    The watcher compares only ``stat`` metadata (inode, device, size and
    modification time), so it never reads credential bytes itself; every
    detected change goes through :meth:`TLSCredentialProvider.reload`. A
    rejected candidate is not retried until the files change again. The first
    poll always reloads, so a change made between provider construction and
    watcher start is picked up.

    Args:
        provider: The provider (or binding) whose files are watched. It must
            expose ``reload()`` and ``watched_files()``.
        interval_seconds: Polling interval, greater than zero and at most
            :data:`MAX_WATCH_INTERVAL_SECONDS`.
    """

    def __init__(self, provider: Any, *, interval_seconds: float) -> None:
        if not callable(getattr(provider, "reload", None)) or not callable(
            getattr(provider, "watched_files", None)
        ):
            raise TypeError("provider must expose reload() and watched_files()")
        interval = _bounded_seconds(
            interval_seconds, maximum=MAX_WATCH_INTERVAL_SECONDS
        )
        if interval <= 0:
            raise ValueError("watch interval must be greater than zero")
        self._provider = provider
        self._interval = interval
        self._signature: Optional[tuple[Any, ...]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def interval_seconds(self) -> float:
        """Return the polling interval in seconds."""
        return self._interval

    @property
    def running(self) -> bool:
        """Return whether the polling thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def poll_once(self) -> Optional[TLSReloadResult]:
        """Reload once if the watched file metadata changed since the last poll.

        Returns:
            The reload result, or ``None`` when nothing changed.
        """
        signature = _file_signature(self._provider.watched_files())
        if signature == self._signature:
            return None
        result = self._provider.reload()
        if result.reason != TLS_REASON_RELOAD_IN_PROGRESS:
            self._signature = signature
        return result

    def start(self) -> None:
        """Start the daemon polling thread; calling it twice is harmless."""
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="openmed-tls-reload", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the polling thread and wait for it to exit.

        Args:
            timeout: Maximum seconds to wait for the thread to finish.
        """
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def __enter__(self) -> TLSCredentialWatcher:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.warning(
                    "tls_credential_watch_failed",
                    extra={"tls_reload_reason": TLS_REASON_FAILED},
                )
            self._stop.wait(self._interval)


def install_reload_signal_handler(
    *targets: SupportsReload,
    signum: Optional[int] = None,
) -> Callable[[], None]:
    """Reload credentials when this process receives a local signal.

    The handler only starts a short-lived thread that calls ``reload()`` on
    each target, so it never runs credential parsing inside signal context
    and cannot deadlock with a reload already in progress. Overlapping
    signals are rejected by the providers with ``tls_reload_in_progress``.

    Args:
        *targets: Providers or bindings to reload, in order.
        signum: Signal number; defaults to ``SIGHUP``.

    Returns:
        A callable that restores the previously installed handler.

    Raises:
        ValueError: If no target is given or the signal is unavailable on
            this platform (for example ``SIGHUP`` on Windows), or when called
            from a thread other than the main thread.
    """
    if not targets or not all(callable(getattr(t, "reload", None)) for t in targets):
        raise ValueError("at least one reloadable target is required")
    if signum is None:
        signum = getattr(signal, "SIGHUP", None)
        if signum is None:
            raise ValueError("SIGHUP is not available on this platform")
    reload_targets = tuple(targets)

    def reload_all() -> None:
        for target in reload_targets:
            try:
                target.reload()
            except Exception:
                logger.warning(
                    "tls_credential_signal_reload_failed",
                    extra={"tls_reload_reason": TLS_REASON_FAILED},
                )

    def handler(_signum: int, _frame: Any) -> None:
        threading.Thread(
            target=reload_all, name="openmed-tls-reload-signal", daemon=True
        ).start()

    previous = signal.getsignal(signum)
    signal.signal(signum, handler)

    def restore() -> None:
        signal.signal(signum, previous)

    return restore


def certificate_sha256(certificate: x509.Certificate) -> str:
    """Return the lowercase SHA-256 fingerprint of a certificate's DER bytes.

    Args:
        certificate: The certificate to fingerprint.

    Returns:
        A 64-character lowercase hex digest.
    """
    der = certificate.public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()


def _resolve_mode(
    *,
    has_certificate: bool,
    has_key: bool,
    has_client_ca: bool,
    has_server_ca: bool,
    requested: Optional[bool],
) -> tuple[str, bool]:
    if requested is not None and type(requested) is not bool:
        raise TLSCredentialError(
            TLS_REASON_CLIENT_AUTH_AMBIGUOUS, category=TLS_CATEGORY_CLIENT_AUTH
        )
    if not has_certificate and not has_key:
        if not has_client_ca or has_server_ca:
            raise TLSCredentialError(
                TLS_REASON_INCOMPLETE, category=TLS_CATEGORY_CERTIFICATE_CHAIN
            )
        if requested is False:
            raise TLSCredentialError(
                TLS_REASON_CLIENT_AUTH_AMBIGUOUS, category=TLS_CATEGORY_CLIENT_AUTH
            )
        return TLS_MODE_CLIENT_TRUST, True
    if not has_certificate:
        raise TLSCredentialError(
            TLS_REASON_INCOMPLETE, category=TLS_CATEGORY_CERTIFICATE_CHAIN
        )
    if not has_key:
        raise TLSCredentialError(
            TLS_REASON_INCOMPLETE, category=TLS_CATEGORY_PRIVATE_KEY
        )
    if requested is None:
        require = has_client_ca
    elif requested and not has_client_ca:
        raise TLSCredentialError(TLS_REASON_INCOMPLETE, category=TLS_CATEGORY_CLIENT_CA)
    elif not requested and has_client_ca:
        raise TLSCredentialError(
            TLS_REASON_CLIENT_AUTH_AMBIGUOUS, category=TLS_CATEGORY_CLIENT_AUTH
        )
    else:
        require = requested
    return (TLS_MODE_MTLS if require else TLS_MODE_TLS), require


def _configured_path(value: Optional[StrPath], category: str) -> Optional[Path]:
    if value is None:
        return None
    try:
        text = os.fspath(value)
    except TypeError:
        raise TLSCredentialError(TLS_REASON_INCOMPLETE, category=category) from None
    if not isinstance(text, str):
        raise TLSCredentialError(TLS_REASON_INCOMPLETE, category=category)
    if not text.strip():
        return None
    return Path(text)


def _read_bounded(path: Path, category: str) -> bytes:
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_CREDENTIAL_FILE_BYTES + 1)
    except (OSError, ValueError):
        raise TLSCredentialError(TLS_REASON_UNREADABLE, category=category) from None
    if not data.strip() or len(data) > MAX_CREDENTIAL_FILE_BYTES:
        raise TLSCredentialError(TLS_REASON_MALFORMED, category=category)
    return data


def _load_certificates(data: bytes, category: str) -> list[x509.Certificate]:
    try:
        certificates = x509.load_pem_x509_certificates(data)
    except ValueError:
        raise TLSCredentialError(TLS_REASON_MALFORMED, category=category) from None
    if not certificates:
        raise TLSCredentialError(TLS_REASON_MALFORMED, category=category)
    return certificates


def _check_validity(
    certificate: x509.Certificate,
    now: datetime,
    *,
    category: str,
    min_remaining: Optional[timedelta] = None,
) -> None:
    if now < certificate.not_valid_before_utc:
        raise TLSCredentialError(
            TLS_REASON_CERTIFICATE_NOT_YET_VALID,
            category=category,
            fingerprint_sha256=certificate_sha256(certificate),
        )
    if now > certificate.not_valid_after_utc:
        raise TLSCredentialError(
            TLS_REASON_CERTIFICATE_EXPIRED,
            category=category,
            fingerprint_sha256=certificate_sha256(certificate),
        )
    if min_remaining is not None:
        _check_remaining(
            certificate, now, category=category, min_remaining=min_remaining
        )


def _check_remaining(
    certificate: x509.Certificate,
    now: datetime,
    *,
    category: str,
    min_remaining: timedelta,
) -> None:
    if certificate.not_valid_after_utc - now < min_remaining:
        raise TLSCredentialError(
            TLS_REASON_CERTIFICATE_EXPIRING,
            category=category,
            fingerprint_sha256=certificate_sha256(certificate),
        )


def _check_key_matches(
    key_pem: bytes, leaf: x509.Certificate, leaf_sha256: str
) -> None:
    try:
        private_key = serialization.load_pem_private_key(key_pem, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm):
        raise TLSCredentialError(
            TLS_REASON_MALFORMED, category=TLS_CATEGORY_PRIVATE_KEY
        ) from None
    if _public_key_der(private_key.public_key()) != _public_key_der(leaf.public_key()):
        raise TLSCredentialError(
            TLS_REASON_KEY_MISMATCH,
            category=TLS_CATEGORY_PRIVATE_KEY,
            fingerprint_sha256=leaf_sha256,
        )


def _check_chain_links(chain: list[x509.Certificate], leaf_sha256: str) -> None:
    for child, issuer in zip(chain, chain[1:]):
        if not _directly_issued_by(child, issuer):
            raise TLSCredentialError(
                TLS_REASON_CHAIN_UNTRUSTED,
                category=TLS_CATEGORY_CERTIFICATE_CHAIN,
                fingerprint_sha256=leaf_sha256,
            )


def _check_chain_anchor(
    chain: list[x509.Certificate],
    authorities: list[x509.Certificate],
    authority_sha256: tuple[str, ...],
) -> None:
    top = chain[-1]
    if certificate_sha256(top) in authority_sha256:
        return
    if any(_directly_issued_by(top, authority) for authority in authorities):
        return
    raise TLSCredentialError(
        TLS_REASON_CHAIN_UNTRUSTED,
        category=TLS_CATEGORY_SERVER_CA,
        fingerprint_sha256=certificate_sha256(chain[0]),
    )


def _directly_issued_by(child: x509.Certificate, issuer: x509.Certificate) -> bool:
    try:
        child.verify_directly_issued_by(issuer)
    except (ValueError, TypeError, InvalidSignature, UnsupportedAlgorithm):
        return False
    return True


def _is_ca_certificate(certificate: x509.Certificate) -> bool:
    try:
        constraints = certificate.extensions.get_extension_for_class(
            x509.BasicConstraints
        ).value
    except (x509.ExtensionNotFound, ValueError):
        return False
    return bool(constraints.ca)


def _within_validity(certificate: x509.Certificate, now: datetime) -> bool:
    return certificate.not_valid_before_utc <= now <= certificate.not_valid_after_utc


def _public_key_der(public_key: Any) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _generation_id(
    mode: str,
    require_client_cert: bool,
    chain_sha256: tuple[str, ...],
    client_ca_sha256: tuple[str, ...],
    server_ca_sha256: tuple[str, ...],
) -> str:
    parts = (
        "v1",
        mode,
        "1" if require_client_cert else "0",
        "chain",
        *chain_sha256,
        "client_ca",
        *sorted(client_ca_sha256),
        "server_ca",
        *sorted(server_ca_sha256),
    )
    digest = hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()
    return _GENERATION_PREFIX + digest[:16]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_signature(paths: tuple[Path, ...]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for path in paths:
        try:
            status = os.stat(path)
        except OSError:
            signature.append(None)
            continue
        signature.append(
            (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        )
    return tuple(signature)


def _validated_hooks(
    hooks: Optional[Mapping[str, ActivationHook]],
) -> dict[str, ActivationHook]:
    if hooks is None:
        return {}
    if not isinstance(hooks, Mapping):
        raise TypeError("activation_hooks must be a mapping")
    validated: dict[str, ActivationHook] = {}
    for name, hook in hooks.items():
        if not isinstance(name, str) or not _HOOK_NAME_PATTERN.fullmatch(name):
            raise ValueError("activation hook names must be short lowercase words")
        if not callable(hook):
            raise TypeError("activation hooks must be callable")
        validated[name] = hook
    return validated


def _bounded_seconds(value: Any, *, maximum: float) -> float:
    if type(value) not in (int, float):
        raise ValueError("seconds must be a number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0 or seconds > maximum:
        raise ValueError("seconds are outside the supported range")
    return seconds


def _safe_fingerprint(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str) and _FINGERPRINT_PATTERN.fullmatch(value):
        return value
    return None


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


__all__ = [
    "ACTIVATION_ACTIVATED",
    "ACTIVATION_FAILED",
    "ACTIVATION_NOT_ATTEMPTED",
    "ACTIVATION_UNCHANGED",
    "ActivationHook",
    "DEFAULT_MIN_REMAINING_VALIDITY_SECONDS",
    "DEFAULT_TRUST_OVERLAP_SECONDS",
    "MAX_CREDENTIAL_FILE_BYTES",
    "MAX_MIN_REMAINING_VALIDITY_SECONDS",
    "MAX_TRUST_OVERLAP_SECONDS",
    "MAX_WATCH_INTERVAL_SECONDS",
    "RELOAD_OUTCOME_ACTIVATED",
    "RELOAD_OUTCOME_REJECTED",
    "RELOAD_OUTCOME_UNCHANGED",
    "SupportsReload",
    "TLSActivationMaterial",
    "TLSCredentialError",
    "TLSCredentialGeneration",
    "TLSCredentialProvider",
    "TLSCredentialSnapshot",
    "TLSCredentialSource",
    "TLSCredentialWatcher",
    "TLSReadiness",
    "TLSReloadResult",
    "TLS_CATEGORIES",
    "TLS_CATEGORY_CERTIFICATE_CHAIN",
    "TLS_CATEGORY_CLIENT_AUTH",
    "TLS_CATEGORY_CLIENT_CA",
    "TLS_CATEGORY_PRIVATE_KEY",
    "TLS_CATEGORY_RELOAD",
    "TLS_CATEGORY_SERVER_CA",
    "TLS_MODE_CLIENT_TRUST",
    "TLS_MODE_MTLS",
    "TLS_MODE_TLS",
    "TLS_REASON_ACTIVATED",
    "TLS_REASON_ACTIVATION_FAILED",
    "TLS_REASON_CANDIDATE_CHANGED",
    "TLS_REASON_CERTIFICATE_EXPIRED",
    "TLS_REASON_CERTIFICATE_EXPIRING",
    "TLS_REASON_CERTIFICATE_NOT_YET_VALID",
    "TLS_REASON_CHAIN_UNTRUSTED",
    "TLS_REASON_CLIENT_AUTH_AMBIGUOUS",
    "TLS_REASON_CLIENT_CA_INVALID",
    "TLS_REASON_CLIENT_TRUST_UNENFORCEABLE",
    "TLS_REASON_CODES",
    "TLS_REASON_FAILED",
    "TLS_REASON_INCOMPLETE",
    "TLS_REASON_KEY_MISMATCH",
    "TLS_REASON_MALFORMED",
    "TLS_REASON_POLICY_INCOMPATIBLE",
    "TLS_REASON_RELOAD_IN_PROGRESS",
    "TLS_REASON_SERVING",
    "TLS_REASON_UNCHANGED",
    "TLS_REASON_UNREADABLE",
    "VALIDATION_INVALID",
    "VALIDATION_NOT_RUN",
    "VALIDATION_VALID",
    "certificate_sha256",
    "install_reload_signal_handler",
]
