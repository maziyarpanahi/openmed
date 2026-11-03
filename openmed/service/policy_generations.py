"""Validated generations for the non-secret service security-policy subset.

A *policy generation* is one immutable, fully validated snapshot of the REST
service settings that bound admission, authorization policy, and outbound
egress. Operators submit a complete candidate document; the manager validates
the whole candidate with the same parsers the service uses at startup plus the
production posture rules below, and then either activates it atomically as the
next generation or rejects it and keeps the prior generation.

In-flight work pins one generation when it is admitted and reads every limit,
scope, and egress rule from that pin, so a request never combines values from
two generations. Standard reloads affect new work only. Emergency restrictive
reloads must be no more permissive than the active generation in every
dimension; pinned work observes them at its next declared checkpoint, where it
either continues, rebinds to the stricter generation as a whole, or halts.

Diagnostics contain generation numbers, digests, timestamps, counts, and fixed
reason codes only. They never include environment values, endpoints,
credentials, scope strings, route templates, or other policy text. TLS
material and API-key or JWT verifier rotation are out of scope.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from openmed.core.config import PROFILE_ENV_VAR
from openmed.core.offline import OFFLINE_ENV_VAR, env_flag_enabled
from openmed.guard.operational_limits import OperationalLimits
from openmed.service.auth import (
    DEFAULT_AUTH_FAILURE_RATE_LIMIT_BURST,
    DEFAULT_AUTH_FAILURE_RATE_LIMIT_RPS,
    DEFAULT_ROUTE_SCOPES,
    SERVICE_AUTH_DENY_BY_DEFAULT_ENV_VAR,
    SERVICE_AUTH_ENABLED_ENV_VAR,
    SERVICE_AUTH_FAILURE_BURST_ENV_VAR,
    SERVICE_AUTH_FAILURE_KEY_ENV_VAR,
    SERVICE_AUTH_FAILURE_RPS_ENV_VAR,
    SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR,
    parse_bool,
    parse_failure_rate_limit_key,
    parse_non_negative_float,
    parse_non_negative_int,
    parse_route_scopes,
)
from openmed.service.limits import (
    SERVICE_MAX_TEXT_LENGTH_ENV_VAR,
    parse_max_text_length,
)
from openmed.service.privacy_gateway import SERVICE_PRIVACY_GATEWAY_ENDPOINT_ENV_VAR
from openmed.service.request_limits import MAX_REQUEST_BYTES_ENV_VAR
from openmed.service.runtime import (
    SERVICE_CONCURRENCY_WAIT_ENV_VAR,
    SERVICE_MAX_CONCURRENCY_ENV_VAR,
    SERVICE_RATE_LIMIT_BURST_ENV_VAR,
    SERVICE_RATE_LIMIT_RPS_ENV_VAR,
    SERVICE_THROTTLE_KEY_ENV_VAR,
    parse_service_concurrency_wait_seconds,
    parse_service_max_concurrency,
    parse_service_rate_limit_burst,
    parse_service_rate_limit_rps,
    parse_service_throttle_key,
)
from openmed.service.tracing import OTLP_ENDPOINT_ENV_VAR

POLICY_GENERATION_SCHEMA_VERSION: Final = "1.0.0"
"""Schema version of candidate documents and value-free reports."""

EGRESS_PURPOSES: Final = frozenset(
    {
        "fhir_bulk",
        "openhim",
        "privacy_gateway",
        "remote_inference",
        "smart_backend",
        "trace_export",
        "webhook_callback",
    }
)
"""Closed vocabulary of outbound purposes an egress rule may allow."""

PolicySelfCheck = Callable[["SecurityPolicySettings"], Iterable[str]]
"""Injected whole-candidate check returning controlled finding codes."""

_CODE_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_PROFILE_PATTERN: Final = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "limits": (
            "max_request_bytes",
            "max_text_length",
            "rate_limit_rps",
            "rate_limit_burst",
            "max_concurrency",
            "concurrency_wait_seconds",
            "throttle_key",
        ),
        "authorization": (
            "enabled",
            "deny_by_default",
            "route_scopes",
            "failure_rate_limit_rps",
            "failure_rate_limit_burst",
            "failure_rate_limit_key",
        ),
        "egress": ("offline", "require_tls", "allowed_purposes"),
    }
)
_TOP_LEVEL: Final = frozenset({"schema_version", "profile", *_SECTIONS})
_FIELDS: Final = frozenset(
    {
        *_TOP_LEVEL,
        *(f"{section}.{key}" for section, keys in _SECTIONS.items() for key in keys),
    }
)
_ERROR_CODES: Final = frozenset(
    {
        "clock_invalid",
        "environment_invalid",
        "finding_code_invalid",
        "initial_generation_invalid",
        "manager_config_invalid",
    }
)
_HISTORY_LIMIT: Final = 32


class PolicyGenerationError(ValueError):
    """Fixed-code refusal that never echoes candidate or environment values.

    Attributes:
        code: Stable reason code; also the exception message.
        findings: Controlled findings when a candidate caused the refusal.
    """

    def __init__(self, code: str, findings: Sequence["PolicyFinding"] = ()) -> None:
        self.code = code if code in _ERROR_CODES else "manager_config_invalid"
        self.findings = tuple(findings)
        super().__init__(self.code)


@dataclass(frozen=True)
class PolicyFinding:
    """One controlled validation finding.

    Attributes:
        code: Stable snake_case reason code.
        field: Dotted schema field name from the fixed policy vocabulary, or
            ``None``. Names outside the vocabulary are dropped, never echoed.
    """

    code: str
    field: str | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not str or not _CODE_PATTERN.fullmatch(self.code):
            raise PolicyGenerationError("finding_code_invalid")
        if self.field is not None and self.field not in _FIELDS:
            object.__setattr__(self, "field", None)

    def __lt__(self, other: "PolicyFinding") -> bool:
        if not isinstance(other, PolicyFinding):
            return NotImplemented
        return (self.code, self.field or "") < (other.code, other.field or "")

    def to_dict(self) -> dict[str, str | None]:
        """Return the finding as a value-free mapping."""
        return {"code": self.code, "field": self.field}


@dataclass(frozen=True)
class LimitPolicy:
    """Request admission limits shared by every REST route.

    A zero rate, burst, or concurrency value disables that gate, matching the
    existing ``OPENMED_SERVICE_*`` environment semantics.
    """

    max_request_bytes: int
    max_text_length: int
    rate_limit_rps: float
    rate_limit_burst: int
    max_concurrency: int
    concurrency_wait_seconds: float
    throttle_key: str

    @property
    def rate_limit_enabled(self) -> bool:
        """Return whether token-bucket rate limiting is active."""
        return self.rate_limit_rps > 0 and self.rate_limit_burst > 0

    def allows_request_bytes(self, size: int) -> bool:
        """Return whether a request body of ``size`` bytes is admissible."""
        return type(size) is int and 0 <= size <= self.max_request_bytes

    def allows_text_length(self, length: int) -> bool:
        """Return whether text of ``length`` characters is admissible."""
        return type(length) is int and 0 <= length <= self.max_text_length


@dataclass(frozen=True)
class AuthorizationPolicy:
    """Non-secret authorization policy; verifier material is excluded."""

    enabled: bool
    deny_by_default: bool
    route_scopes: Mapping[tuple[str, str], tuple[str, ...]] = field(repr=False)
    failure_rate_limit_rps: float
    failure_rate_limit_burst: int
    failure_rate_limit_key: str

    @property
    def failure_rate_limit_enabled(self) -> bool:
        """Return whether failed authentication attempts are rate limited."""
        return self.failure_rate_limit_rps > 0 and self.failure_rate_limit_burst > 0

    def required_scopes(self, method: str, path: str) -> tuple[str, ...] | None:
        """Return the scopes for a declared route, or ``None`` if undeclared."""
        return self.route_scopes.get((method, path))

    def __repr__(self) -> str:
        return (
            f"AuthorizationPolicy(enabled={self.enabled}, "
            f"deny_by_default={self.deny_by_default}, "
            f"routes={len(self.route_scopes)})"
        )


@dataclass(frozen=True)
class EgressPolicy:
    """Outbound rules: an allow-list of purposes plus transport requirements."""

    offline: bool
    require_tls: bool
    allowed_purposes: frozenset[str]

    def allows(self, purpose: str, *, tls: bool) -> bool:
        """Return whether an outbound operation may proceed.

        Args:
            purpose: Declared purpose from :data:`EGRESS_PURPOSES`.
            tls: Whether the transport is TLS-protected.

        Returns:
            ``False`` when offline, when the purpose is undeclared or unknown,
            or when TLS is required but absent; ``True`` otherwise.
        """
        if self.offline or type(purpose) is not str:
            return False
        if purpose not in self.allowed_purposes:
            return False
        return bool(tls) or not self.require_tls


@dataclass(frozen=True)
class SecurityPolicySettings:
    """One complete, validated security-policy snapshot.

    Instances are only produced by :func:`validate_policy_document`. The
    representation shows the digest only, never policy values.
    """

    profile: str
    limits: LimitPolicy
    authorization: AuthorizationPolicy
    egress: EgressPolicy
    _digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        payload = json.dumps(
            self.to_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        object.__setattr__(
            self, "_digest", "sha256:" + hashlib.sha256(payload).hexdigest()
        )

    def digest(self) -> str:
        """Return the canonical SHA-256 digest of the effective policy."""
        return self._digest

    def to_document(self) -> dict[str, Any]:
        """Return the effective policy as a complete candidate document.

        The result contains operator policy text (route scopes) and is meant
        for round-tripping configuration, not for logs or diagnostics.
        """
        limits = self.limits
        auth = self.authorization
        return {
            "schema_version": POLICY_GENERATION_SCHEMA_VERSION,
            "profile": self.profile,
            "limits": {
                "max_request_bytes": limits.max_request_bytes,
                "max_text_length": limits.max_text_length,
                "rate_limit_rps": limits.rate_limit_rps,
                "rate_limit_burst": limits.rate_limit_burst,
                "max_concurrency": limits.max_concurrency,
                "concurrency_wait_seconds": limits.concurrency_wait_seconds,
                "throttle_key": limits.throttle_key,
            },
            "authorization": {
                "enabled": auth.enabled,
                "deny_by_default": auth.deny_by_default,
                "route_scopes": _route_scopes_document(auth.route_scopes),
                "failure_rate_limit_rps": auth.failure_rate_limit_rps,
                "failure_rate_limit_burst": auth.failure_rate_limit_burst,
                "failure_rate_limit_key": auth.failure_rate_limit_key,
            },
            "egress": {
                "offline": self.egress.offline,
                "require_tls": self.egress.require_tls,
                "allowed_purposes": sorted(self.egress.allowed_purposes),
            },
        }

    def is_no_more_permissive_than(self, other: "SecurityPolicySettings") -> bool:
        """Return whether every dimension is at least as strict as ``other``.

        Lower byte, length, rate, burst, concurrency, and wait limits are
        stricter, where a disabled gate counts as unlimited. Enabled
        authorization, deny-by-default, offline mode, and required TLS are
        stricter. Route scopes must be supersets and egress purposes subsets.
        Profiles and throttle keying must be identical.
        """
        if self.profile != other.profile:
            return False
        return (
            _limits_le(self.limits, other.limits)
            and _authorization_le(self.authorization, other.authorization)
            and _egress_le(self.egress, other.egress)
        )

    def __repr__(self) -> str:
        return f"SecurityPolicySettings(digest={self._digest!r})"


@dataclass(frozen=True)
class PolicyValidation:
    """Outcome of validating one candidate document.

    Attributes:
        settings: Validated settings, or ``None`` when any finding exists.
        findings: Sorted, de-duplicated controlled findings.
    """

    settings: SecurityPolicySettings | None
    findings: tuple[PolicyFinding, ...]

    @property
    def valid(self) -> bool:
        """Return whether the candidate passed every check."""
        return self.settings is not None and not self.findings

    def __repr__(self) -> str:
        digest = None if self.settings is None else self.settings.digest()
        return f"PolicyValidation(digest={digest!r}, findings={len(self.findings)})"


@dataclass(frozen=True)
class PolicyGeneration:
    """One activated policy generation.

    Attributes:
        number: Strictly increasing generation number starting at 1.
        digest: Digest of the effective settings.
        activated_at: Injected-clock timestamp of activation.
        mode: ``bootstrap``, ``standard``, or ``emergency``.
        settings: The immutable policy values.
    """

    number: int
    digest: str
    activated_at: float
    mode: str
    settings: SecurityPolicySettings = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free summary of the generation."""
        return {
            "activated_at": self.activated_at,
            "digest": self.digest,
            "generation": self.number,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class PolicyCandidate:
    """A validated-or-rejected candidate bound to the generation it targets.

    Attributes:
        base_generation: Generation number that was active when proposed.
        emergency: Whether the candidate uses the restrictive fast path.
        digest: Candidate settings digest, or ``None`` if it did not parse.
        findings: Controlled findings; empty when the candidate is valid.
        settings: Validated settings, or ``None``.
    """

    base_generation: int
    emergency: bool
    digest: str | None
    findings: tuple[PolicyFinding, ...]
    settings: SecurityPolicySettings | None = field(repr=False, compare=False)

    @property
    def valid(self) -> bool:
        """Return whether the candidate may be activated."""
        return self.settings is not None and not self.findings


@dataclass(frozen=True)
class ReloadResult:
    """Value-free outcome of an activation attempt.

    Attributes:
        outcome: ``activated``, ``unchanged``, or ``rejected``.
        mode: ``standard`` or ``emergency``.
        active_generation: Generation number active after the attempt.
        active_digest: Digest of the generation active after the attempt.
        candidate_digest: Candidate digest, or ``None`` if it did not parse.
        findings: Controlled findings explaining a rejection.
    """

    outcome: str
    mode: str
    active_generation: int
    active_digest: str
    candidate_digest: str | None
    findings: tuple[PolicyFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-ready value-free mapping."""
        return {
            "active_digest": self.active_digest,
            "active_generation": self.active_generation,
            "candidate_digest": self.candidate_digest,
            "findings": [item.to_dict() for item in self.findings],
            "mode": self.mode,
            "outcome": self.outcome,
            "schema_version": POLICY_GENERATION_SCHEMA_VERSION,
        }


class CheckpointAction(str, enum.Enum):
    """What pinned work must do at a declared continuation checkpoint."""

    CONTINUE = "continue"
    REBOUND = "rebound"
    HALT = "halt"


@dataclass(frozen=True)
class CheckpointDecision:
    """Value-free continuation decision for one checkpoint.

    Attributes:
        action: Continue, continue under a rebound generation, or halt.
        generation: Generation number the work is bound to afterwards.
        code: Stable reason code for the decision.
    """

    action: CheckpointAction
    generation: int
    code: str


class PolicyPin:
    """A unit of in-flight work bound to exactly one policy generation.

    Read every limit, scope, and egress rule through :attr:`settings`. The
    binding changes only inside :meth:`checkpoint`, and then as a whole.
    """

    def __init__(
        self, manager: "PolicyGenerationManager", generation: PolicyGeneration
    ) -> None:
        self._manager = manager
        self._generation = generation
        self._released = False
        self._halted = False

    @property
    def generation(self) -> PolicyGeneration:
        """Return the generation this work is currently bound to."""
        return self._generation

    @property
    def settings(self) -> SecurityPolicySettings:
        """Return the policy values of the bound generation."""
        return self._generation.settings

    @property
    def halted(self) -> bool:
        """Return whether a checkpoint required this work to stop."""
        return self._halted

    @property
    def released(self) -> bool:
        """Return whether the pin has been released."""
        return self._released

    def checkpoint(self) -> CheckpointDecision:
        """Apply emergency continuation semantics at a declared point.

        Returns:
            ``CONTINUE`` when no emergency generation was activated since the
            work was bound, or when the bound generation is already at least
            as strict as the active one. ``REBOUND`` when the active
            generation is no more permissive than the bound one; the work then
            reads the active generation as a whole. ``HALT`` when neither is
            uniformly stricter; the work must stop or checkpoint durably.
        """
        return self._manager._checkpoint(self)

    def release(self) -> None:
        """Release the pin; repeated calls are no-ops."""
        self._manager._release(self)

    def __enter__(self) -> "PolicyPin":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"PolicyPin(generation={self._generation.number}, "
            f"halted={self._halted}, released={self._released})"
        )


class PolicyGenerationManager:
    """Hold the active policy generation and apply validated reloads.

    Args:
        document: Complete initial candidate document.
        clock: Injected wall clock returning seconds; defaults to
            :func:`time.time`.
        self_checks: Additional whole-candidate checks, for example a
            deployment self-check. Each returns controlled snake_case finding
            codes; any finding blocks activation.
        history_limit: Number of value-free generation summaries retained.

    Raises:
        PolicyGenerationError: ``initial_generation_invalid`` when the initial
            document fails validation; the error carries the findings.
    """

    def __init__(
        self,
        document: Mapping[str, Any],
        *,
        clock: Callable[[], float] = time.time,
        self_checks: Sequence[PolicySelfCheck] = (),
        history_limit: int = _HISTORY_LIMIT,
    ) -> None:
        if not callable(clock) or type(history_limit) is not int or history_limit < 1:
            raise PolicyGenerationError("manager_config_invalid")
        checks = tuple(self_checks)
        if not all(callable(check) for check in checks):
            raise PolicyGenerationError("manager_config_invalid")
        self._clock = clock
        self._self_checks = checks
        self._lock = threading.Lock()
        self._inflight: dict[int, int] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._last_reload: dict[str, Any] | None = None
        self._last_emergency = 0
        validation = validate_policy_document(document, self_checks=checks)
        if validation.settings is None:
            raise PolicyGenerationError(
                "initial_generation_invalid", validation.findings
            )
        self._active = self._new_generation(1, validation.settings, "bootstrap")
        self._history.append(self._active.to_dict())

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "PolicyGenerationManager":
        """Bootstrap from the environment variables the service reads today.

        Args:
            environ: Mapping to read; defaults to :data:`os.environ`.
            **kwargs: Forwarded to the constructor.

        Returns:
            A manager whose first generation reflects the current settings.
        """
        return cls(policy_document_from_environment(environ), **kwargs)

    @property
    def active(self) -> PolicyGeneration:
        """Return the currently active generation."""
        return self._active

    def pin(self) -> PolicyPin:
        """Bind new work to the active generation and return its pin."""
        with self._lock:
            generation = self._active
            self._inflight[generation.number] = (
                self._inflight.get(generation.number, 0) + 1
            )
            return PolicyPin(self, generation)

    def propose(
        self, document: Mapping[str, Any], *, emergency: bool = False
    ) -> PolicyCandidate:
        """Validate a complete candidate against the active generation.

        Args:
            document: Complete candidate document.
            emergency: Require the candidate to be no more permissive than
                the active generation in every dimension.

        Returns:
            A candidate bound to the active generation number.
        """
        with self._lock:
            return self._propose_locked(document, emergency=bool(emergency))

    def activate(self, candidate: PolicyCandidate) -> ReloadResult:
        """Atomically activate a valid candidate or reject it.

        Args:
            candidate: A candidate returned by :meth:`propose`.

        Returns:
            The value-free outcome. A candidate proposed against an older
            generation is rejected with ``candidate_stale``.
        """
        with self._lock:
            return self._activate_locked(candidate)

    def reload(
        self, document: Mapping[str, Any], *, emergency: bool = False
    ) -> ReloadResult:
        """Validate and activate a candidate in one atomic step.

        Args:
            document: Complete candidate document.
            emergency: Use the restrictive fast path; see :meth:`propose`.

        Returns:
            The value-free outcome. Rejections keep the prior generation.
        """
        with self._lock:
            candidate = self._propose_locked(document, emergency=bool(emergency))
            return self._activate_locked(candidate)

    def readiness(self) -> dict[str, Any]:
        """Return a value-free readiness and drain report.

        The active generation always passed validation, so a rejected reload
        never turns readiness false.
        """
        with self._lock:
            active = self._active
            return {
                "digest": active.digest,
                "generation": active.number,
                "inflight": {
                    str(number): count
                    for number, count in sorted(self._inflight.items())
                    if count
                },
                "last_reload": None
                if self._last_reload is None
                else dict(self._last_reload),
                "mode": active.mode,
                "ready": True,
                "schema_version": POLICY_GENERATION_SCHEMA_VERSION,
            }

    def history(self) -> tuple[dict[str, Any], ...]:
        """Return value-free summaries of recently activated generations."""
        with self._lock:
            return tuple(dict(item) for item in self._history)

    def __repr__(self) -> str:
        active = self._active
        return (
            f"PolicyGenerationManager(generation={active.number}, "
            f"digest={active.digest!r})"
        )

    def _new_generation(
        self, number: int, settings: SecurityPolicySettings, mode: str
    ) -> PolicyGeneration:
        try:
            activated_at = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            raise PolicyGenerationError("clock_invalid") from None
        if not math.isfinite(activated_at):
            raise PolicyGenerationError("clock_invalid")
        return PolicyGeneration(
            number=number,
            digest=settings.digest(),
            activated_at=activated_at,
            mode=mode,
            settings=settings,
        )

    def _propose_locked(
        self, document: Mapping[str, Any], *, emergency: bool
    ) -> PolicyCandidate:
        active = self._active
        validation = validate_policy_document(document, self_checks=self._self_checks)
        findings = list(validation.findings)
        settings = validation.settings
        if settings is not None:
            if settings.profile != active.settings.profile:
                findings.append(PolicyFinding("profile_change_unsupported", "profile"))
            elif emergency and not settings.is_no_more_permissive_than(active.settings):
                findings.append(PolicyFinding("emergency_not_restrictive"))
        findings_tuple = _normalized(findings)
        return PolicyCandidate(
            base_generation=active.number,
            emergency=emergency,
            digest=None if settings is None else settings.digest(),
            findings=findings_tuple,
            settings=None if findings_tuple else settings,
        )

    def _activate_locked(self, candidate: PolicyCandidate) -> ReloadResult:
        active = self._active
        if not isinstance(candidate, PolicyCandidate):
            return self._record(
                ReloadResult(
                    outcome="rejected",
                    mode="standard",
                    active_generation=active.number,
                    active_digest=active.digest,
                    candidate_digest=None,
                    findings=(PolicyFinding("candidate_invalid"),),
                )
            )
        mode = "emergency" if candidate.emergency else "standard"
        settings = candidate.settings
        findings = candidate.findings
        if settings is None or findings:
            findings = findings or (PolicyFinding("candidate_invalid"),)
        elif candidate.base_generation != active.number:
            findings = (PolicyFinding("candidate_stale"),)
        if settings is None or findings:
            return self._record(
                ReloadResult(
                    outcome="rejected",
                    mode=mode,
                    active_generation=active.number,
                    active_digest=active.digest,
                    candidate_digest=candidate.digest,
                    findings=findings,
                )
            )
        if settings.digest() == active.digest:
            return self._record(
                ReloadResult(
                    outcome="unchanged",
                    mode=mode,
                    active_generation=active.number,
                    active_digest=active.digest,
                    candidate_digest=candidate.digest,
                )
            )
        generation = self._new_generation(active.number + 1, settings, mode)
        self._active = generation
        if candidate.emergency:
            self._last_emergency = generation.number
        self._history.append(generation.to_dict())
        return self._record(
            ReloadResult(
                outcome="activated",
                mode=mode,
                active_generation=generation.number,
                active_digest=generation.digest,
                candidate_digest=candidate.digest,
            )
        )

    def _record(self, result: ReloadResult) -> ReloadResult:
        self._last_reload = {
            "findings": [item.code for item in result.findings],
            "outcome": result.outcome,
        }
        return result

    def _checkpoint(self, pin: PolicyPin) -> CheckpointDecision:
        with self._lock:
            pinned = pin._generation
            if pin._released:
                return CheckpointDecision(
                    CheckpointAction.HALT, pinned.number, "pin_released"
                )
            if pin._halted:
                return CheckpointDecision(
                    CheckpointAction.HALT, pinned.number, "policy_continuation_denied"
                )
            active = self._active
            if self._last_emergency <= pinned.number or active is pinned:
                return CheckpointDecision(
                    CheckpointAction.CONTINUE, pinned.number, "policy_pinned"
                )
            if active.settings.is_no_more_permissive_than(pinned.settings):
                self._move_inflight(pinned.number, active.number)
                pin._generation = active
                return CheckpointDecision(
                    CheckpointAction.REBOUND, active.number, "policy_rebound"
                )
            if pinned.settings.is_no_more_permissive_than(active.settings):
                return CheckpointDecision(
                    CheckpointAction.CONTINUE, pinned.number, "policy_pinned"
                )
            pin._halted = True
            return CheckpointDecision(
                CheckpointAction.HALT, pinned.number, "policy_continuation_denied"
            )

    def _release(self, pin: PolicyPin) -> None:
        with self._lock:
            if pin._released:
                return
            pin._released = True
            self._decrement(pin._generation.number)

    def _move_inflight(self, source: int, target: int) -> None:
        self._decrement(source)
        self._inflight[target] = self._inflight.get(target, 0) + 1

    def _decrement(self, number: int) -> None:
        remaining = self._inflight.get(number, 0) - 1
        if remaining > 0:
            self._inflight[number] = remaining
        else:
            self._inflight.pop(number, None)


class PolicyGenerationMiddleware:
    """Pure ASGI middleware that pins each HTTP or WebSocket connection.

    The pin is stored in ``scope["state"]`` under :attr:`STATE_KEY` and is
    released only after the downstream application returns, which for
    streaming responses is after the final body chunk was sent.

    Args:
        app: Downstream ASGI application.
        manager: Generation manager providing the pins.
    """

    STATE_KEY: Final = "openmed_policy_pin"

    def __init__(self, app: Any, *, manager: PolicyGenerationManager) -> None:
        self.app = app
        self.manager = manager

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Pin the connection for its full lifetime and call the app."""
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        pin = self.manager.pin()
        scoped = dict(scope)
        scoped["state"] = {**dict(scope.get("state") or {}), self.STATE_KEY: pin}
        try:
            await self.app(scoped, receive, send)
        finally:
            pin.release()


def validate_policy_document(
    document: Mapping[str, Any],
    *,
    self_checks: Sequence[PolicySelfCheck] = (),
) -> PolicyValidation:
    """Validate a complete candidate document as one unit.

    Every field is parsed with the existing service startup parsers, then the
    production posture rules in :func:`production_policy_findings` and every
    injected self-check run against the assembled settings.

    Args:
        document: Candidate with ``schema_version``, ``profile``, ``limits``,
            ``authorization``, and ``egress`` sections. Every field must be
            present; partial documents are incomplete, never merged.
        self_checks: Additional checks returning controlled finding codes.

    Returns:
        The validation outcome; ``settings`` is ``None`` on any finding.
    """
    findings: list[PolicyFinding] = []
    if not isinstance(document, Mapping):
        return PolicyValidation(None, (PolicyFinding("document_invalid"),))
    if any(type(key) is not str or key not in _TOP_LEVEL for key in document):
        findings.append(PolicyFinding("field_unsupported"))
    for key in sorted(_TOP_LEVEL):
        if key not in document:
            findings.append(PolicyFinding("field_missing", key))
    if (
        "schema_version" in document
        and document["schema_version"] != POLICY_GENERATION_SCHEMA_VERSION
    ):
        findings.append(PolicyFinding("schema_version_unsupported", "schema_version"))
    profile = document.get("profile")
    if "profile" in document and (
        type(profile) is not str or not _PROFILE_PATTERN.fullmatch(profile)
    ):
        findings.append(PolicyFinding("profile_invalid", "profile"))

    sections: dict[str, Mapping[str, Any]] = {}
    for section, keys in _SECTIONS.items():
        if section not in document:
            continue
        raw = document[section]
        if not isinstance(raw, Mapping):
            findings.append(PolicyFinding("field_invalid", section))
            continue
        if any(type(key) is not str or key not in keys for key in raw):
            findings.append(PolicyFinding("field_unsupported", section))
        for key in keys:
            if key not in raw:
                findings.append(PolicyFinding("field_missing", f"{section}.{key}"))
        sections[section] = raw

    limits = _parse_limits(sections.get("limits"), findings)
    authorization = _parse_authorization(sections.get("authorization"), findings)
    egress = _parse_egress(sections.get("egress"), findings)
    if findings or limits is None or authorization is None or egress is None:
        return PolicyValidation(None, _normalized(findings))

    settings = SecurityPolicySettings(
        profile=str(profile),
        limits=limits,
        authorization=authorization,
        egress=egress,
    )
    findings.extend(production_policy_findings(settings))
    findings.extend(_run_self_checks(settings, self_checks))
    if findings:
        return PolicyValidation(None, _normalized(findings))
    return PolicyValidation(settings, ())


def production_policy_findings(
    settings: SecurityPolicySettings,
) -> tuple[PolicyFinding, ...]:
    """Return production posture findings for already-parsed settings.

    Only the ``prod`` profile is checked. Production requires authentication,
    deny-by-default authorization, a failed-authentication rate limit, at
    least one admission budget (concurrency or rate limit), TLS for every
    allowed egress purpose, and non-empty scopes on every route that declares
    scopes by default.

    Args:
        settings: Validated settings.

    Returns:
        Sorted controlled findings; empty outside the ``prod`` profile.
    """
    if settings.profile != "prod":
        return ()
    findings: list[PolicyFinding] = []
    auth = settings.authorization
    if not auth.enabled:
        findings.append(PolicyFinding("auth_disabled", "authorization.enabled"))
    if not auth.deny_by_default:
        findings.append(
            PolicyFinding("deny_by_default_disabled", "authorization.deny_by_default")
        )
    if not auth.failure_rate_limit_enabled:
        findings.append(PolicyFinding("auth_failure_limit_disabled", "authorization"))
    if any(
        default and not auth.route_scopes.get(route)
        for route, default in DEFAULT_ROUTE_SCOPES.items()
    ):
        findings.append(
            PolicyFinding("route_scope_missing", "authorization.route_scopes")
        )
    limits = settings.limits
    if limits.max_concurrency == 0 and not limits.rate_limit_enabled:
        findings.append(PolicyFinding("admission_unbounded", "limits"))
    if not settings.egress.require_tls:
        findings.append(PolicyFinding("egress_tls_not_required", "egress.require_tls"))
    return _normalized(findings)


def policy_document_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a complete candidate from the service environment variables.

    Only the supported non-secret subset is read, through the same parsers the
    service uses at startup. Credentials, verifier keys, endpoints, headers,
    and TLS material are never copied; a configured privacy-gateway or OTLP
    endpoint only contributes its egress purpose.

    Args:
        environ: Mapping to read; defaults to :data:`os.environ`.

    Returns:
        A complete candidate document.

    Raises:
        PolicyGenerationError: ``environment_invalid`` when a supported
            variable fails its existing parser; the value is not echoed.
    """
    env = os.environ if environ is None else environ
    get = env.get
    try:
        raw_bytes = get(MAX_REQUEST_BYTES_ENV_VAR)
        max_request_bytes = (
            OperationalLimits().max_request_bytes
            if raw_bytes is None or not raw_bytes.strip()
            else OperationalLimits(max_request_bytes=int(raw_bytes)).max_request_bytes
        )
        limits = {
            "max_request_bytes": max_request_bytes,
            "max_text_length": parse_max_text_length(
                get(SERVICE_MAX_TEXT_LENGTH_ENV_VAR)
            ),
            "rate_limit_rps": parse_service_rate_limit_rps(
                get(SERVICE_RATE_LIMIT_RPS_ENV_VAR)
            ),
            "rate_limit_burst": parse_service_rate_limit_burst(
                get(SERVICE_RATE_LIMIT_BURST_ENV_VAR)
            ),
            "max_concurrency": parse_service_max_concurrency(
                get(SERVICE_MAX_CONCURRENCY_ENV_VAR)
            ),
            "concurrency_wait_seconds": parse_service_concurrency_wait_seconds(
                get(SERVICE_CONCURRENCY_WAIT_ENV_VAR)
            ),
            "throttle_key": parse_service_throttle_key(
                get(SERVICE_THROTTLE_KEY_ENV_VAR)
            ),
        }
        authorization = {
            "enabled": parse_bool(
                get(SERVICE_AUTH_ENABLED_ENV_VAR),
                env_var=SERVICE_AUTH_ENABLED_ENV_VAR,
                default=False,
            ),
            "deny_by_default": parse_bool(
                get(SERVICE_AUTH_DENY_BY_DEFAULT_ENV_VAR),
                env_var=SERVICE_AUTH_DENY_BY_DEFAULT_ENV_VAR,
                default=True,
            ),
            "route_scopes": _route_scopes_document(
                parse_route_scopes(get(SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR))
            ),
            "failure_rate_limit_rps": parse_non_negative_float(
                get(SERVICE_AUTH_FAILURE_RPS_ENV_VAR),
                env_var=SERVICE_AUTH_FAILURE_RPS_ENV_VAR,
                default=DEFAULT_AUTH_FAILURE_RATE_LIMIT_RPS,
            ),
            "failure_rate_limit_burst": parse_non_negative_int(
                get(SERVICE_AUTH_FAILURE_BURST_ENV_VAR),
                env_var=SERVICE_AUTH_FAILURE_BURST_ENV_VAR,
                default=DEFAULT_AUTH_FAILURE_RATE_LIMIT_BURST,
            ),
            "failure_rate_limit_key": parse_failure_rate_limit_key(
                get(SERVICE_AUTH_FAILURE_KEY_ENV_VAR)
            ),
        }
    except (TypeError, ValueError, OverflowError, AttributeError):
        raise PolicyGenerationError("environment_invalid") from None
    offline = env_flag_enabled(get(OFFLINE_ENV_VAR))
    purposes: list[str] = []
    if not offline:
        if (get(SERVICE_PRIVACY_GATEWAY_ENDPOINT_ENV_VAR) or "").strip():
            purposes.append("privacy_gateway")
        if (get(OTLP_ENDPOINT_ENV_VAR) or "").strip():
            purposes.append("trace_export")
    profile = (get(PROFILE_ENV_VAR) or "").strip() or "prod"
    return {
        "schema_version": POLICY_GENERATION_SCHEMA_VERSION,
        "profile": profile,
        "limits": limits,
        "authorization": authorization,
        "egress": {
            "offline": offline,
            "require_tls": True,
            "allowed_purposes": sorted(purposes),
        },
    }


def _normalized(findings: Iterable[PolicyFinding]) -> tuple[PolicyFinding, ...]:
    return tuple(sorted(set(findings)))


def _route_scopes_document(
    route_scopes: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[str, list[str]]:
    return {
        f"{method} {path}": list(scopes)
        for (method, path), scopes in sorted(route_scopes.items())
    }


def _strict_int(value: Any) -> int:
    if type(value) is not int:
        raise TypeError
    return value


def _strict_number(value: Any) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise TypeError
    return float(value)


def _strict_bool(value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError
    return value


def _strict_str(value: Any) -> str:
    if type(value) is not str:
        raise TypeError
    return value


def _parse_field(
    section: Mapping[str, Any],
    name: str,
    key: str,
    parser: Callable[[Any], Any],
    findings: list[PolicyFinding],
) -> Any:
    if key not in section:
        return None
    try:
        return parser(section[key])
    except (TypeError, ValueError, OverflowError):
        findings.append(PolicyFinding("field_invalid", f"{name}.{key}"))
        return None


def _max_request_bytes(value: Any) -> int:
    return OperationalLimits(max_request_bytes=_strict_int(value)).max_request_bytes


def _max_text_length(value: Any) -> int:
    parsed = _strict_int(value)
    # The startup parser silently substitutes its default for non-positive
    # values; a reload candidate must instead be rejected.
    if parsed <= 0 or parse_max_text_length(str(parsed)) != parsed:
        raise ValueError
    return parsed


def _parse_limits(
    section: Mapping[str, Any] | None, findings: list[PolicyFinding]
) -> LimitPolicy | None:
    if section is None:
        return None
    name = "limits"
    values = {
        "max_request_bytes": _parse_field(
            section, name, "max_request_bytes", _max_request_bytes, findings
        ),
        "max_text_length": _parse_field(
            section, name, "max_text_length", _max_text_length, findings
        ),
        "rate_limit_rps": _parse_field(
            section,
            name,
            "rate_limit_rps",
            lambda v: parse_service_rate_limit_rps(repr(_strict_number(v))),
            findings,
        ),
        "rate_limit_burst": _parse_field(
            section,
            name,
            "rate_limit_burst",
            lambda v: parse_service_rate_limit_burst(str(_strict_int(v))),
            findings,
        ),
        "max_concurrency": _parse_field(
            section,
            name,
            "max_concurrency",
            lambda v: parse_service_max_concurrency(str(_strict_int(v))),
            findings,
        ),
        "concurrency_wait_seconds": _parse_field(
            section,
            name,
            "concurrency_wait_seconds",
            lambda v: parse_service_concurrency_wait_seconds(repr(_strict_number(v))),
            findings,
        ),
        "throttle_key": _parse_field(
            section,
            name,
            "throttle_key",
            lambda v: parse_service_throttle_key(_strict_str(v)),
            findings,
        ),
    }
    if any(value is None for value in values.values()):
        return None
    return LimitPolicy(**values)


def _route_scopes(value: Any) -> Mapping[tuple[str, str], tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError
    overrides: dict[str, list[str]] = {}
    for route, scopes in value.items():
        if type(route) is not str:
            raise TypeError
        if type(scopes) is str:
            overrides[route] = scopes.replace(",", " ").split()
            continue
        if not isinstance(scopes, (list, tuple)) or not all(
            type(scope) is str for scope in scopes
        ):
            raise TypeError
        overrides[route] = list(scopes)
    parsed = parse_route_scopes(json.dumps(overrides)) if overrides else None
    mapping = dict(DEFAULT_ROUTE_SCOPES) if parsed is None else dict(parsed)
    return MappingProxyType(mapping)


def _parse_authorization(
    section: Mapping[str, Any] | None, findings: list[PolicyFinding]
) -> AuthorizationPolicy | None:
    if section is None:
        return None
    name = "authorization"
    route_scopes = None
    if "route_scopes" in section:
        try:
            route_scopes = _route_scopes(section["route_scopes"])
        except (TypeError, ValueError, OverflowError):
            findings.append(
                PolicyFinding("route_scopes_invalid", "authorization.route_scopes")
            )
    values = {
        "enabled": _parse_field(section, name, "enabled", _strict_bool, findings),
        "deny_by_default": _parse_field(
            section, name, "deny_by_default", _strict_bool, findings
        ),
        "failure_rate_limit_rps": _parse_field(
            section,
            name,
            "failure_rate_limit_rps",
            lambda v: parse_non_negative_float(
                repr(_strict_number(v)),
                env_var=SERVICE_AUTH_FAILURE_RPS_ENV_VAR,
                default=DEFAULT_AUTH_FAILURE_RATE_LIMIT_RPS,
            ),
            findings,
        ),
        "failure_rate_limit_burst": _parse_field(
            section,
            name,
            "failure_rate_limit_burst",
            lambda v: parse_non_negative_int(
                str(_strict_int(v)),
                env_var=SERVICE_AUTH_FAILURE_BURST_ENV_VAR,
                default=DEFAULT_AUTH_FAILURE_RATE_LIMIT_BURST,
            ),
            findings,
        ),
        "failure_rate_limit_key": _parse_field(
            section,
            name,
            "failure_rate_limit_key",
            lambda v: parse_failure_rate_limit_key(_strict_str(v)),
            findings,
        ),
    }
    if route_scopes is None or any(value is None for value in values.values()):
        return None
    return AuthorizationPolicy(route_scopes=route_scopes, **values)


def _parse_egress(
    section: Mapping[str, Any] | None, findings: list[PolicyFinding]
) -> EgressPolicy | None:
    if section is None:
        return None
    name = "egress"
    offline = _parse_field(section, name, "offline", _strict_bool, findings)
    require_tls = _parse_field(section, name, "require_tls", _strict_bool, findings)
    purposes = None
    if "allowed_purposes" in section:
        raw = section["allowed_purposes"]
        if not isinstance(raw, (list, tuple)) or not all(
            type(item) is str for item in raw
        ):
            findings.append(PolicyFinding("field_invalid", "egress.allowed_purposes"))
        elif not set(raw) <= EGRESS_PURPOSES:
            findings.append(
                PolicyFinding("egress_purpose_unsupported", "egress.allowed_purposes")
            )
        else:
            purposes = frozenset(raw)
    if offline is None or require_tls is None or purposes is None:
        return None
    return EgressPolicy(
        offline=offline, require_tls=require_tls, allowed_purposes=purposes
    )


def _run_self_checks(
    settings: SecurityPolicySettings, checks: Sequence[PolicySelfCheck]
) -> list[PolicyFinding]:
    findings: list[PolicyFinding] = []
    for check in checks:
        try:
            codes = list(check(settings))
        except Exception:  # Injected checks fail closed without detail.
            findings.append(PolicyFinding("self_check_failed"))
            continue
        for code in codes:
            if type(code) is str and _CODE_PATTERN.fullmatch(code):
                findings.append(PolicyFinding(code))
            else:
                findings.append(PolicyFinding("self_check_finding_invalid"))
    return findings


def _gate_limit(rate: float, burst: int) -> tuple[float, float]:
    if rate > 0 and burst > 0:
        return rate, float(burst)
    return math.inf, math.inf


def _limits_le(left: LimitPolicy, right: LimitPolicy) -> bool:
    left_rate, left_burst = _gate_limit(left.rate_limit_rps, left.rate_limit_burst)
    right_rate, right_burst = _gate_limit(right.rate_limit_rps, right.rate_limit_burst)
    left_cap = left.max_concurrency or math.inf
    right_cap = right.max_concurrency or math.inf
    return (
        left.max_request_bytes <= right.max_request_bytes
        and left.max_text_length <= right.max_text_length
        and left_rate <= right_rate
        and left_burst <= right_burst
        and left_cap <= right_cap
        and left.concurrency_wait_seconds <= right.concurrency_wait_seconds
        and left.throttle_key == right.throttle_key
    )


def _authorization_le(left: AuthorizationPolicy, right: AuthorizationPolicy) -> bool:
    if right.enabled and not left.enabled:
        return False
    if right.deny_by_default and not left.deny_by_default:
        return False
    if left.failure_rate_limit_key != right.failure_rate_limit_key:
        return False
    left_rate, left_burst = _gate_limit(
        left.failure_rate_limit_rps, left.failure_rate_limit_burst
    )
    right_rate, right_burst = _gate_limit(
        right.failure_rate_limit_rps, right.failure_rate_limit_burst
    )
    if left_rate > right_rate or left_burst > right_burst:
        return False
    for route, scopes in right.route_scopes.items():
        granted = left.route_scopes.get(route)
        if granted is None or not set(scopes) <= set(granted):
            return False
    return True


def _egress_le(left: EgressPolicy, right: EgressPolicy) -> bool:
    if right.offline and not left.offline:
        return False
    if right.require_tls and not left.require_tls:
        return False
    if left.offline:
        return True
    return left.allowed_purposes <= right.allowed_purposes


__all__ = [
    "AuthorizationPolicy",
    "CheckpointAction",
    "CheckpointDecision",
    "EGRESS_PURPOSES",
    "EgressPolicy",
    "LimitPolicy",
    "POLICY_GENERATION_SCHEMA_VERSION",
    "PolicyCandidate",
    "PolicyFinding",
    "PolicyGeneration",
    "PolicyGenerationError",
    "PolicyGenerationManager",
    "PolicyGenerationMiddleware",
    "PolicySelfCheck",
    "PolicyValidation",
    "ReloadResult",
    "SecurityPolicySettings",
    "policy_document_from_environment",
    "production_policy_findings",
    "validate_policy_document",
]
