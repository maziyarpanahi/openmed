"""Typed fixed-option decisions with calibrated abstention.

The contract is local-first and backend-neutral. It scores caller-declared
options, preserves their order, and never turns ambiguity or a backend failure
into an apparent success. Results are assistive review artifacts; they do not
authorize diagnosis, treatment, enrollment, outreach, ordering, or any other
patient-care action.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

DECISION_SCHEMA_VERSION: Final = "1.0.0"
DECISION_COMPATIBILITY_POLICY: Final = "same_major"
DEFAULT_CALIBRATION_ID: Final = "openmed.synthetic.fixed_option.v1"
DEFAULT_CALIBRATION_VERSION: Final = "1.0.0"
MAX_INPUT_CHARS: Final = 8192
MAX_INPUT_BYTES: Final = 32768
MAX_OPTIONS: Final = 64
MAX_OPTION_CHARS: Final = 256
MAX_OPTION_BYTES: Final = 1024
MAX_TOTAL_OPTION_BYTES: Final = 16384
MAX_BATCH_SIZE: Final = 32
MAX_BATCH_INPUT_CHARS: Final = 131072
MIN_TIMEOUT_MS: Final = 1
MAX_TIMEOUT_MS: Final = 30000
DECISION_ADVISORY: Final = (
    "Decision scores are assistive review evidence, not a diagnosis, treatment "
    "recommendation, enrollment decision, or autonomous clinical action."
)
PERMISSIVE_LICENSES: Final = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "MIT",
        "MPL-2.0",
    }
)
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
_SCHEMA_FILES: Final = MappingProxyType(
    {
        "decision_request": "decision_request.schema.json",
        "decision_result": "decision_result.schema.json",
    }
)


class DecisionMode(str, Enum):
    """Supported fixed-output decision shapes."""

    FIXED_CHOICE = "fixed_choice"
    BOOLEAN_CHOICE = "boolean_choice"
    ORDERED_PREFERENCE = "ordered_preference"
    SCALAR_SCORE = "scalar_score"
    MULTI_LABEL = "multi_label"


class DecisionState(str, Enum):
    """Lossless public decision states."""

    SUCCESS = "success"
    ABSTAINED = "abstained"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    FAILURE = "failure"


class DecisionBackendKind(str, Enum):
    """Supported local backend families."""

    DETERMINISTIC = "deterministic"
    ENCODER = "encoder"
    CROSS_ENCODER = "cross_encoder"
    SPECIALIST = "specialist"


class DecisionError(ValueError):
    """Base error for invalid decision contracts."""


class DecisionUnsupportedError(DecisionError):
    """Raised when a backend cannot score the requested mode."""


class DecisionConflictError(DecisionError):
    """Raised when backend output conflicts with the declared contract."""


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DecisionError(f"{name} must be finite and in [0, 1]")
    return result


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise DecisionError(f"{name} must be a controlled identifier")
    return value


def _semantic_version(value: Any, name: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise DecisionError(f"{name} must be semantic")
    return value


def _bounded_text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DecisionError(f"{name} must be bounded non-empty text")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise DecisionError(f"{name} contains control characters")
    return value


@dataclass(frozen=True, slots=True)
class DecisionBackendIdentity:
    """Inspectable identity for a local decision backend."""

    backend_id: str
    kind: DecisionBackendKind
    runtime: str
    model_id: str
    revision: str
    license_id: str
    local: bool = True
    deterministic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DecisionBackendKind(self.kind))
        for name in ("backend_id", "runtime", "model_id", "revision"):
            _bounded_text(getattr(self, name), name, maximum=256)
        if self.license_id not in PERMISSIVE_LICENSES:
            raise DecisionError("decision backend license must be permissive")
        if self.local is not True:
            raise DecisionError("decision backends must be local by default")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "kind": self.kind.value,
            "runtime": self.runtime,
            "model_id": self.model_id,
            "revision": self.revision,
            "license_id": self.license_id,
            "local": self.local,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class DecisionCalibrationProfile:
    """Versioned thresholds pinned to frozen calibration slices."""

    calibration_id: str = DEFAULT_CALIBRATION_ID
    version: str = DEFAULT_CALIBRATION_VERSION
    method: str = "deterministic_margin"
    minimum_confidence: float = 0.70
    minimum_margin: float = 0.15
    multi_label_threshold: float = 0.70
    in_domain_digest: str = (
        "sha256:a6a2e0596c95b0568fe8871c99774fb1edc5b1db6de692e86172df952642dd6c"
    )
    out_of_domain_digest: str = (
        "sha256:d20c55e3877710223c6c37f87f6a8aaf9dea1b326087b37c60a81522ac0c9289"
    )

    def __post_init__(self) -> None:
        _controlled(self.calibration_id, "calibration_id")
        _semantic_version(self.version, "calibration version")
        _controlled(self.method, "calibration method")
        for name in (
            "minimum_confidence",
            "minimum_margin",
            "multi_label_threshold",
        ):
            _probability(getattr(self, name), name)
        if self.minimum_margin > self.minimum_confidence:
            raise DecisionError("minimum margin cannot exceed confidence threshold")
        for name in ("in_domain_digest", "out_of_domain_digest"):
            if _DIGEST_RE.fullmatch(getattr(self, name)) is None:
                raise DecisionError(f"{name} must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "version": self.version,
            "method": self.method,
            "minimum_confidence": self.minimum_confidence,
            "minimum_margin": self.minimum_margin,
            "multi_label_threshold": self.multi_label_threshold,
            "in_domain_digest": self.in_domain_digest,
            "out_of_domain_digest": self.out_of_domain_digest,
        }


DEFAULT_CALIBRATION_PROFILE: Final = DecisionCalibrationProfile()


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One bounded request shared by Python, REST, and MCP."""

    mode: DecisionMode
    input_text: str = field(repr=False)
    options: tuple[str, ...] = field(default_factory=tuple, repr=False)
    namespace: str = "default"
    purpose: str = "care_review"
    calibration_id: str = DEFAULT_CALIBRATION_ID
    timeout_ms: int = 5000
    schema_version: str = DECISION_SCHEMA_VERSION
    compatibility_policy: str = DECISION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        mode = DecisionMode(self.mode)
        text = _bounded_input(self.input_text)
        options = _validated_options(mode, self.options)
        _controlled(self.namespace, "namespace")
        _controlled(self.purpose, "purpose")
        _controlled(self.calibration_id, "calibration_id")
        _semantic_version(self.schema_version, "decision schema version")
        if self.schema_version.split(".", 1)[0] != "1":
            raise DecisionUnsupportedError(
                "decision request schema major is unsupported"
            )
        if self.compatibility_policy != DECISION_COMPATIBILITY_POLICY:
            raise DecisionUnsupportedError(
                "decision compatibility policy is unsupported"
            )
        if type(self.timeout_ms) is not int or not (
            MIN_TIMEOUT_MS <= self.timeout_ms <= MAX_TIMEOUT_MS
        ):
            raise DecisionError(
                f"timeout_ms must be between {MIN_TIMEOUT_MS} and {MAX_TIMEOUT_MS}"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "input_text", text)
        object.__setattr__(self, "options", options)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionRequest":
        """Parse a strict request mapping without retaining caller ownership."""

        expected = {
            "mode",
            "input_text",
            "options",
            "namespace",
            "purpose",
            "calibration_id",
            "timeout_ms",
            "schema_version",
            "compatibility_policy",
        }
        unknown = set(payload) - expected
        if unknown:
            raise DecisionError("decision request contains unsupported fields")
        try:
            return cls(
                mode=DecisionMode(payload["mode"]),
                input_text=payload["input_text"],
                options=tuple(payload.get("options", ())),
                namespace=payload.get("namespace", "default"),
                purpose=payload.get("purpose", "care_review"),
                calibration_id=payload.get("calibration_id", DEFAULT_CALIBRATION_ID),
                timeout_ms=payload.get("timeout_ms", 5000),
                schema_version=payload.get("schema_version", DECISION_SCHEMA_VERSION),
                compatibility_policy=payload.get(
                    "compatibility_policy", DECISION_COMPATIBILITY_POLICY
                ),
            )
        except KeyError:
            raise DecisionError(
                "decision request is missing a required field"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical transport request, including transient input."""

        return {
            "mode": self.mode.value,
            "input_text": self.input_text,
            "options": list(self.options),
            "namespace": self.namespace,
            "purpose": self.purpose,
            "calibration_id": self.calibration_id,
            "timeout_ms": self.timeout_ms,
            "schema_version": self.schema_version,
            "compatibility_policy": self.compatibility_policy,
        }

    def safe_metadata(self) -> dict[str, Any]:
        """Return log-safe bounds metadata without text or option values."""

        return {
            "mode": self.mode.value,
            "input_chars": len(self.input_text),
            "input_bytes": len(self.input_text.encode("utf-8")),
            "option_count": len(self.options),
            "namespace": self.namespace,
            "purpose": self.purpose,
            "calibration_id": self.calibration_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class DecisionBackendOutput:
    """Raw scores returned by a local backend before calibration policy."""

    option_scores: tuple[float, ...] = ()
    scalar_score: float | None = None
    confidence: float | None = None
    calibration_id: str = DEFAULT_CALIBRATION_ID
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scores = tuple(
            _probability(value, "option score") for value in self.option_scores
        )
        scalar = (
            None
            if self.scalar_score is None
            else _probability(self.scalar_score, "scalar score")
        )
        confidence = (
            None
            if self.confidence is None
            else _probability(self.confidence, "backend confidence")
        )
        _controlled(self.calibration_id, "calibration_id")
        warnings = tuple(
            dict.fromkeys(_controlled(item, "warning") for item in self.warnings)
        )
        object.__setattr__(self, "option_scores", scores)
        object.__setattr__(self, "scalar_score", scalar)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "warnings", warnings)


@runtime_checkable
class DecisionBackend(Protocol):
    """Protocol implemented by deterministic and learned local backends."""

    identity: DecisionBackendIdentity

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        """Return ordered option scores or one scalar score."""


class DeterministicDecisionBackend:
    """Conservative dependency-free fallback for explicit lexical choices."""

    identity = DecisionBackendIdentity(
        backend_id="openmed.deterministic.fixed_option.v1",
        kind=DecisionBackendKind.DETERMINISTIC,
        runtime="python",
        model_id="openmed/deterministic-fixed-option",
        revision="1.0.0",
        license_id="Apache-2.0",
        deterministic=True,
    )

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        if request.mode is DecisionMode.SCALAR_SCORE:
            value = _deterministic_scalar(request.input_text)
            if value is None:
                raise DecisionUnsupportedError(
                    "deterministic scalar scoring requires an explicit value in [0, 1]"
                )
            return DecisionBackendOutput(
                scalar_score=value,
                confidence=1.0,
                calibration_id=request.calibration_id,
            )
        return DecisionBackendOutput(
            option_scores=tuple(
                _lexical_option_score(request.input_text, option)
                for option in request.options
            ),
            calibration_id=request.calibration_id,
        )


Scorer = Callable[[str, tuple[str, ...], DecisionMode], Any]


class CallableDecisionBackend:
    """Adapter for local encoder, cross-encoder, or specialist scorers."""

    def __init__(self, *, identity: DecisionBackendIdentity, scorer: Scorer) -> None:
        if identity.kind is DecisionBackendKind.DETERMINISTIC:
            raise DecisionError("callable learned backends require a learned kind")
        if not callable(scorer):
            raise TypeError("decision scorer must be callable")
        self.identity = identity
        self._scorer = scorer

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        return _normalize_backend_output(
            self._scorer(request.input_text, request.options, request.mode),
            calibration_id=request.calibration_id,
        )


class EncoderDecisionBackend(CallableDecisionBackend):
    """Local permissively licensed encoder scorer adapter."""

    def __init__(
        self,
        *,
        backend_id: str,
        model_id: str,
        revision: str,
        license_id: str,
        scorer: Scorer,
        runtime: str = "torch",
    ) -> None:
        super().__init__(
            identity=DecisionBackendIdentity(
                backend_id=backend_id,
                kind=DecisionBackendKind.ENCODER,
                runtime=runtime,
                model_id=model_id,
                revision=revision,
                license_id=license_id,
            ),
            scorer=scorer,
        )


class CrossEncoderDecisionBackend(CallableDecisionBackend):
    """Local permissively licensed cross-encoder scorer adapter."""

    def __init__(
        self,
        *,
        backend_id: str,
        model_id: str,
        revision: str,
        license_id: str,
        scorer: Scorer,
        runtime: str = "torch",
    ) -> None:
        super().__init__(
            identity=DecisionBackendIdentity(
                backend_id=backend_id,
                kind=DecisionBackendKind.CROSS_ENCODER,
                runtime=runtime,
                model_id=model_id,
                revision=revision,
                license_id=license_id,
            ),
            scorer=scorer,
        )


class SpecialistDecisionBackend(CallableDecisionBackend):
    """Adapter for a small locally fine-tuned fixed-option specialist."""

    def __init__(
        self,
        *,
        backend_id: str,
        model_id: str,
        revision: str,
        license_id: str,
        scorer: Scorer,
        runtime: str = "torch",
    ) -> None:
        super().__init__(
            identity=DecisionBackendIdentity(
                backend_id=backend_id,
                kind=DecisionBackendKind.SPECIALIST,
                runtime=runtime,
                model_id=model_id,
                revision=revision,
                license_id=license_id,
            ),
            scorer=scorer,
        )


DETERMINISTIC_DECISION_BACKEND: Final = DeterministicDecisionBackend()


@dataclass(frozen=True, slots=True)
class DecisionAccessPolicy:
    """Purpose, namespace, and backend-family allowlist."""

    allowed_namespaces: frozenset[str] = frozenset({"default"})
    allowed_purposes: frozenset[str] = frozenset(
        {"analytics", "care_review", "quality"}
    )
    allowed_backend_kinds: frozenset[DecisionBackendKind] = frozenset(
        DecisionBackendKind
    )
    policy_version: str = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        namespaces = frozenset(
            _controlled(item, "allowed namespace") for item in self.allowed_namespaces
        )
        purposes = frozenset(
            _controlled(item, "allowed purpose") for item in self.allowed_purposes
        )
        kinds = frozenset(
            DecisionBackendKind(item) for item in self.allowed_backend_kinds
        )
        if not namespaces or not purposes or not kinds:
            raise DecisionError("decision access-policy allowlists cannot be empty")
        _semantic_version(self.policy_version, "policy version")
        object.__setattr__(self, "allowed_namespaces", namespaces)
        object.__setattr__(self, "allowed_purposes", purposes)
        object.__setattr__(self, "allowed_backend_kinds", kinds)

    def authorize(
        self,
        request: DecisionRequest,
        backend: DecisionBackendIdentity,
    ) -> tuple[bool, str | None]:
        if request.namespace not in self.allowed_namespaces:
            return False, "namespace_denied"
        if request.purpose not in self.allowed_purposes:
            return False, "purpose_denied"
        if backend.kind not in self.allowed_backend_kinds:
            return False, "backend_denied"
        return True, None

    def decision(
        self,
        request: DecisionRequest,
        backend: DecisionBackendIdentity,
    ) -> dict[str, Any]:
        allowed, code = self.authorize(request, backend)
        return {
            "allowed": allowed,
            "code": code,
            "namespace": request.namespace,
            "purpose": request.purpose,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Canonical result shared by Python, REST, MCP, and persistence."""

    mode: DecisionMode
    state: DecisionState
    option_scores: tuple[Mapping[str, Any], ...]
    choice: str | None
    choices: tuple[str, ...]
    ranking: tuple[str, ...]
    scalar_score: float | None
    confidence: float | None
    margin: float | None
    calibration: Mapping[str, Any]
    backend: Mapping[str, Any]
    access: Mapping[str, Any]
    warnings: tuple[str, ...]
    review: Mapping[str, Any]
    code: str | None = None
    advisory: str = DECISION_ADVISORY
    autonomous_action: bool = False
    schema_version: str = DECISION_SCHEMA_VERSION
    compatibility_policy: str = DECISION_COMPATIBILITY_POLICY
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DecisionMode(self.mode)
        state = DecisionState(self.state)
        if state is DecisionState.SUCCESS and self.code is not None:
            raise DecisionError("successful decision results cannot carry a code")
        if state is not DecisionState.SUCCESS and self.code is None:
            raise DecisionError("non-success decision results require a code")
        if self.autonomous_action:
            raise DecisionError("decision results cannot authorize autonomous action")
        if self.advisory != DECISION_ADVISORY:
            raise DecisionError("decision advisory cannot be replaced")
        _semantic_version(self.schema_version, "decision schema version")
        if self.compatibility_policy != DECISION_COMPATIBILITY_POLICY:
            raise DecisionUnsupportedError(
                "decision compatibility policy is unsupported"
            )
        if self.code is not None:
            _controlled(self.code, "decision code")
        option_scores = tuple(_frozen_mapping(item) for item in self.option_scores)
        for position, item in enumerate(option_scores):
            if item.get("index") != position:
                raise DecisionError("decision option scores must preserve input order")
            _bounded_text(item.get("option"), "scored option", maximum=MAX_OPTION_CHARS)
            _probability(item.get("score"), "option score")
        choices = tuple(str(item) for item in self.choices)
        ranking = tuple(str(item) for item in self.ranking)
        for value, name in (
            (self.scalar_score, "scalar score"),
            (self.confidence, "confidence"),
            (self.margin, "margin"),
        ):
            if value is not None:
                _probability(value, name)
        if self.choice is not None:
            _bounded_text(self.choice, "choice", maximum=MAX_OPTION_CHARS)
            if self.choice not in choices or self.choice not in ranking:
                raise DecisionError(
                    "decision choice must appear in choices and ranking"
                )
        if state is not DecisionState.SUCCESS and (
            self.choice is not None or choices or self.scalar_score is not None
        ):
            raise DecisionError("non-success decision results cannot carry selections")
        if mode is DecisionMode.SCALAR_SCORE:
            if state is DecisionState.SUCCESS and self.scalar_score is None:
                raise DecisionError("successful scalar decisions require a score")
            if option_scores or ranking:
                raise DecisionError("scalar decisions cannot carry option scores")
        elif state is DecisionState.SUCCESS and (
            self.choice is None or not option_scores or not ranking
        ):
            raise DecisionError(
                "successful option decisions require a choice and scores"
            )
        warnings = tuple(
            dict.fromkeys(_controlled(item, "warning") for item in self.warnings)
        )
        review = _frozen_mapping(self.review)
        if review.get("required") is not True or not review.get("reasons"):
            raise DecisionError("decision results always require review reasons")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "option_scores", option_scores)
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "ranking", ranking)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "calibration", _frozen_mapping(self.calibration))
        object.__setattr__(self, "backend", _frozen_mapping(self.backend))
        object.__setattr__(self, "access", _frozen_mapping(self.access))
        object.__setattr__(self, "review", review)
        object.__setattr__(self, "extensions", _frozen_mapping(self.extensions))

    @property
    def abstained(self) -> bool:
        return self.state is DecisionState.ABSTAINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "state": self.state.value,
            "code": self.code,
            "option_scores": [_plain(item) for item in self.option_scores],
            "choice": self.choice,
            "choices": list(self.choices),
            "ranking": list(self.ranking),
            "scalar_score": self.scalar_score,
            "confidence": self.confidence,
            "margin": self.margin,
            "calibration": _plain(self.calibration),
            "backend": _plain(self.backend),
            "access": _plain(self.access),
            "warnings": list(self.warnings),
            "review": _plain(self.review),
            "advisory": self.advisory,
            "autonomous_action": self.autonomous_action,
            "schema_version": self.schema_version,
            "compatibility_policy": self.compatibility_policy,
            "extensions": _plain(self.extensions),
        }


def decide(
    request: DecisionRequest | Mapping[str, Any],
    *,
    backend: DecisionBackend | None = None,
    policy: DecisionAccessPolicy | None = None,
    calibration_profiles: Mapping[str, DecisionCalibrationProfile] | None = None,
) -> DecisionResult:
    """Score one bounded request and apply calibrated abstention policy."""

    normalized = (
        request
        if isinstance(request, DecisionRequest)
        else DecisionRequest.from_dict(request)
    )
    selected_backend = backend or DETERMINISTIC_DECISION_BACKEND
    if not isinstance(selected_backend, DecisionBackend):
        raise TypeError("backend must implement the DecisionBackend protocol")
    active_policy = policy or DecisionAccessPolicy()
    access = active_policy.decision(normalized, selected_backend.identity)
    try:
        profile = _calibration_profile(normalized, calibration_profiles)
    except DecisionUnsupportedError:
        return _terminal_result(
            normalized,
            DecisionState.UNSUPPORTED,
            "calibration_unavailable",
            selected_backend.identity,
            DEFAULT_CALIBRATION_PROFILE,
            access,
        )
    if not access["allowed"]:
        return _terminal_result(
            normalized,
            DecisionState.DENIED,
            str(access["code"]),
            selected_backend.identity,
            profile,
            access,
        )

    started = time.monotonic()
    try:
        raw = selected_backend.score(normalized)
    except DecisionUnsupportedError:
        return _terminal_result(
            normalized,
            DecisionState.UNSUPPORTED,
            "backend_unsupported",
            selected_backend.identity,
            profile,
            access,
        )
    except DecisionConflictError:
        return _terminal_result(
            normalized,
            DecisionState.CONFLICT,
            "backend_conflict",
            selected_backend.identity,
            profile,
            access,
        )
    except TimeoutError:
        return _terminal_result(
            normalized,
            DecisionState.FAILURE,
            "backend_timeout",
            selected_backend.identity,
            profile,
            access,
        )
    except Exception:
        return _terminal_result(
            normalized,
            DecisionState.FAILURE,
            "backend_failure",
            selected_backend.identity,
            profile,
            access,
        )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if elapsed_ms > normalized.timeout_ms:
        return _terminal_result(
            normalized,
            DecisionState.FAILURE,
            "backend_timeout",
            selected_backend.identity,
            profile,
            access,
        )
    if not isinstance(raw, DecisionBackendOutput):
        return _terminal_result(
            normalized,
            DecisionState.FAILURE,
            "backend_contract_invalid",
            selected_backend.identity,
            profile,
            access,
        )
    if raw.calibration_id != profile.calibration_id:
        return _terminal_result(
            normalized,
            DecisionState.CONFLICT,
            "calibration_mismatch",
            selected_backend.identity,
            profile,
            access,
        )
    try:
        return _decision_from_scores(
            normalized,
            raw,
            selected_backend.identity,
            profile,
            access,
        )
    except (DecisionError, TypeError, ValueError):
        return _terminal_result(
            normalized,
            DecisionState.FAILURE,
            "backend_contract_invalid",
            selected_backend.identity,
            profile,
            access,
        )


def decide_batch(
    requests: Iterable[DecisionRequest | Mapping[str, Any]],
    *,
    backend: DecisionBackend | None = None,
    policy: DecisionAccessPolicy | None = None,
    calibration_profiles: Mapping[str, DecisionCalibrationProfile] | None = None,
) -> tuple[DecisionResult, ...]:
    """Evaluate a bounded batch without changing per-item semantics."""

    items = tuple(
        item if isinstance(item, DecisionRequest) else DecisionRequest.from_dict(item)
        for item in requests
    )
    if not 1 <= len(items) <= MAX_BATCH_SIZE:
        raise DecisionError(f"decision batch must contain 1 to {MAX_BATCH_SIZE} items")
    if sum(len(item.input_text) for item in items) > MAX_BATCH_INPUT_CHARS:
        raise DecisionError(
            "decision batch input exceeds the aggregate character limit"
        )
    return tuple(
        decide(
            item,
            backend=backend,
            policy=policy,
            calibration_profiles=calibration_profiles,
        )
        for item in items
    )


@dataclass(frozen=True, slots=True)
class DecisionCalibrationExample:
    """One synthetic frozen calibration example."""

    slice_name: str
    request: DecisionRequest
    expected_choices: tuple[str, ...]

    def __post_init__(self) -> None:
        _controlled(self.slice_name, "slice name")
        if not self.expected_choices:
            raise DecisionError("calibration examples require expected choices")
        if not set(self.expected_choices).issubset(self.request.options):
            raise DecisionError("calibration expected choices must be request options")


def evaluate_decision_calibration(
    examples: Sequence[DecisionCalibrationExample],
    *,
    backend: DecisionBackend | None = None,
    profile: DecisionCalibrationProfile = DEFAULT_CALIBRATION_PROFILE,
) -> dict[str, Any]:
    """Evaluate coverage, accuracy, and ECE on named frozen slices."""

    if not examples:
        raise DecisionError("calibration evaluation requires examples")
    grouped: dict[str, list[tuple[DecisionResult, DecisionCalibrationExample]]] = {}
    for example in examples:
        if example.request.calibration_id != profile.calibration_id:
            raise DecisionConflictError("calibration example uses another profile")
        result = decide(
            example.request,
            backend=backend,
            calibration_profiles={profile.calibration_id: profile},
        )
        grouped.setdefault(example.slice_name, []).append((result, example))

    slices = []
    for slice_name, records in sorted(grouped.items()):
        covered = [item for item in records if item[0].state is DecisionState.SUCCESS]
        correct = [
            item
            for item in covered
            if set(item[0].choices or ((item[0].choice,) if item[0].choice else ()))
            == set(item[1].expected_choices)
        ]
        accuracy = len(correct) / len(covered) if covered else 0.0
        average_confidence = (
            sum(float(item[0].confidence or 0.0) for item in covered) / len(covered)
            if covered
            else 0.0
        )
        slices.append(
            {
                "slice": slice_name,
                "total": len(records),
                "covered": len(covered),
                "coverage": len(covered) / len(records),
                "accuracy": accuracy,
                "expected_calibration_error": abs(average_confidence - accuracy),
                "abstained": len(records) - len(covered),
            }
        )
    fixture_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [
                    {
                        "slice": item.slice_name,
                        "request": item.request.to_dict(),
                        "expected_choices": list(item.expected_choices),
                    }
                    for item in examples
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "calibration": profile.to_dict(),
        "fixture_digest": fixture_digest,
        "slices": slices,
        "contains_source_text": False,
    }


def load_decision_schema(name: str) -> dict[str, Any]:
    """Load one committed decision JSON Schema."""

    try:
        filename = _SCHEMA_FILES[name]
    except KeyError:
        raise KeyError(f"unknown decision schema {name!r}") from None
    resource = resources.files(_SCHEMA_PACKAGE).joinpath(filename)
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decision_request_schema() -> dict[str, Any]:
    """Return the canonical request JSON Schema."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openmed.dev/schemas/decision-request/1.0.0",
        "title": "OpenMed Fixed-Option Decision Request",
        "type": "object",
        "additionalProperties": False,
        "required": ["mode", "input_text"],
        "allOf": [
            {
                "if": {"properties": {"mode": {"const": "scalar_score"}}},
                "then": {"properties": {"options": {"maxItems": 0}}},
            },
            {
                "if": {"properties": {"mode": {"const": "boolean_choice"}}},
                "then": {"properties": {"options": {"minItems": 2, "maxItems": 2}}},
            },
            {
                "if": {
                    "properties": {
                        "mode": {
                            "enum": [
                                "fixed_choice",
                                "ordered_preference",
                                "multi_label",
                            ]
                        }
                    }
                },
                "then": {
                    "properties": {"options": {"minItems": 2, "maxItems": MAX_OPTIONS}}
                },
            },
        ],
        "properties": {
            "mode": {
                "type": "string",
                "enum": [item.value for item in DecisionMode],
            },
            "input_text": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_INPUT_CHARS,
            },
            "options": {
                "type": "array",
                "maxItems": MAX_OPTIONS,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_OPTION_CHARS,
                },
                "default": [],
            },
            "namespace": {
                "type": "string",
                "pattern": _CONTROLLED_RE.pattern,
                "default": "default",
            },
            "purpose": {
                "type": "string",
                "pattern": _CONTROLLED_RE.pattern,
                "default": "care_review",
            },
            "calibration_id": {
                "type": "string",
                "pattern": _CONTROLLED_RE.pattern,
                "default": DEFAULT_CALIBRATION_ID,
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": MIN_TIMEOUT_MS,
                "maximum": MAX_TIMEOUT_MS,
                "default": 5000,
            },
            "schema_version": {
                "type": "string",
                "const": DECISION_SCHEMA_VERSION,
                "default": DECISION_SCHEMA_VERSION,
            },
            "compatibility_policy": {
                "type": "string",
                "const": DECISION_COMPATIBILITY_POLICY,
                "default": DECISION_COMPATIBILITY_POLICY,
            },
        },
    }


def decision_result_schema() -> dict[str, Any]:
    """Return the canonical result JSON Schema."""

    nullable_number = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
    controlled = {"type": "string", "pattern": _CONTROLLED_RE.pattern}
    nullable_controlled = {"anyOf": [controlled, {"type": "null"}]}
    calibration = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "calibration_id",
            "version",
            "method",
            "minimum_confidence",
            "minimum_margin",
            "multi_label_threshold",
            "in_domain_digest",
            "out_of_domain_digest",
        ],
        "properties": {
            "calibration_id": controlled,
            "version": {"type": "string", "pattern": _VERSION_RE.pattern},
            "method": controlled,
            "minimum_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "minimum_margin": {"type": "number", "minimum": 0, "maximum": 1},
            "multi_label_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "in_domain_digest": {"type": "string", "pattern": _DIGEST_RE.pattern},
            "out_of_domain_digest": {
                "type": "string",
                "pattern": _DIGEST_RE.pattern,
            },
        },
    }
    backend = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "backend_id",
            "kind",
            "runtime",
            "model_id",
            "revision",
            "license_id",
            "local",
            "deterministic",
        ],
        "properties": {
            "backend_id": {"type": "string", "maxLength": 256},
            "kind": {
                "type": "string",
                "enum": [item.value for item in DecisionBackendKind],
            },
            "runtime": {"type": "string", "maxLength": 256},
            "model_id": {"type": "string", "maxLength": 256},
            "revision": {"type": "string", "maxLength": 256},
            "license_id": {
                "type": "string",
                "enum": sorted(PERMISSIVE_LICENSES),
            },
            "local": {"type": "boolean", "const": True},
            "deterministic": {"type": "boolean"},
        },
    }
    access = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "allowed",
            "code",
            "namespace",
            "purpose",
            "policy_version",
        ],
        "properties": {
            "allowed": {"type": "boolean"},
            "code": nullable_controlled,
            "namespace": controlled,
            "purpose": controlled,
            "policy_version": {"type": "string", "pattern": _VERSION_RE.pattern},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openmed.dev/schemas/decision-result/1.0.0",
        "title": "OpenMed Fixed-Option Decision Result",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "mode",
            "state",
            "code",
            "option_scores",
            "choice",
            "choices",
            "ranking",
            "scalar_score",
            "confidence",
            "margin",
            "calibration",
            "backend",
            "access",
            "warnings",
            "review",
            "advisory",
            "autonomous_action",
            "schema_version",
            "compatibility_policy",
            "extensions",
        ],
        "properties": {
            "mode": {
                "type": "string",
                "enum": [item.value for item in DecisionMode],
            },
            "state": {
                "type": "string",
                "enum": [item.value for item in DecisionState],
            },
            "code": nullable_controlled,
            "option_scores": {
                "type": "array",
                "maxItems": MAX_OPTIONS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "option", "score"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 0},
                        "option": {"type": "string", "maxLength": MAX_OPTION_CHARS},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "choice": {"type": ["string", "null"], "maxLength": MAX_OPTION_CHARS},
            "choices": {
                "type": "array",
                "maxItems": MAX_OPTIONS,
                "items": {"type": "string", "maxLength": MAX_OPTION_CHARS},
            },
            "ranking": {
                "type": "array",
                "maxItems": MAX_OPTIONS,
                "items": {"type": "string", "maxLength": MAX_OPTION_CHARS},
            },
            "scalar_score": nullable_number,
            "confidence": nullable_number,
            "margin": nullable_number,
            "calibration": calibration,
            "backend": backend,
            "access": access,
            "warnings": {
                "type": "array",
                "maxItems": 32,
                "uniqueItems": True,
                "items": controlled,
            },
            "review": {
                "type": "object",
                "additionalProperties": False,
                "required": ["required", "reasons"],
                "properties": {
                    "required": {"type": "boolean", "const": True},
                    "reasons": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 32,
                        "uniqueItems": True,
                        "items": controlled,
                    },
                },
            },
            "advisory": {"type": "string", "const": DECISION_ADVISORY},
            "autonomous_action": {"type": "boolean", "const": False},
            "schema_version": {"type": "string", "const": DECISION_SCHEMA_VERSION},
            "compatibility_policy": {
                "type": "string",
                "const": DECISION_COMPATIBILITY_POLICY,
            },
            "extensions": {"type": "object"},
        },
    }


def migrate_decision_result(
    payload: Mapping[str, Any],
    *,
    target_version: str = DECISION_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Migrate same-major results while retaining unknown fields in extensions."""

    source_version = _semantic_version(
        payload.get("schema_version"), "decision schema version"
    )
    target = _semantic_version(target_version, "target decision schema version")
    if source_version.split(".", 1)[0] != target.split(".", 1)[0]:
        raise DecisionUnsupportedError("decision result schema major is unsupported")
    schema = decision_result_schema()
    known = set(schema["properties"])
    migrated = {key: deepcopy(value) for key, value in payload.items() if key in known}
    extensions = dict(migrated.get("extensions") or {})
    extensions.update(
        {key: deepcopy(value) for key, value in payload.items() if key not in known}
    )
    migrated["extensions"] = extensions
    migrated["schema_version"] = target
    migrated.setdefault("compatibility_policy", DECISION_COMPATIBILITY_POLICY)
    return migrated


def _decision_from_scores(
    request: DecisionRequest,
    raw: DecisionBackendOutput,
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    if request.mode is DecisionMode.SCALAR_SCORE:
        if raw.scalar_score is None or raw.option_scores:
            raise DecisionConflictError("scalar backend output has invalid shape")
        confidence = raw.confidence if raw.confidence is not None else 0.0
        state = (
            DecisionState.SUCCESS
            if confidence >= profile.minimum_confidence
            else DecisionState.ABSTAINED
        )
        code = None if state is DecisionState.SUCCESS else "low_confidence"
        return _result(
            request,
            state=state,
            code=code,
            option_scores=(),
            choice=None,
            choices=(),
            ranking=(),
            scalar_score=raw.scalar_score if state is DecisionState.SUCCESS else None,
            confidence=confidence,
            margin=None,
            identity=identity,
            profile=profile,
            access=access,
            backend_warnings=raw.warnings,
        )

    if raw.scalar_score is not None or len(raw.option_scores) != len(request.options):
        raise DecisionConflictError("option backend output has invalid shape")
    scored = tuple(
        {"index": index, "option": option, "score": score}
        for index, (option, score) in enumerate(
            zip(request.options, raw.option_scores, strict=True)
        )
    )
    ranked_indices = tuple(
        sorted(
            range(len(request.options)),
            key=lambda index: (-raw.option_scores[index], index),
        )
    )
    confidence = raw.option_scores[ranked_indices[0]]
    margin = confidence - raw.option_scores[ranked_indices[1]]
    ranking = tuple(request.options[index] for index in ranked_indices)

    if request.mode is DecisionMode.MULTI_LABEL:
        selected = tuple(
            request.options[index]
            for index, score in enumerate(raw.option_scores)
            if score >= profile.multi_label_threshold
        )
        state = DecisionState.SUCCESS if selected else DecisionState.ABSTAINED
        code = None if selected else "no_label_above_threshold"
        choice = selected[0] if selected else None
        choices = selected
    else:
        clears_confidence = confidence >= profile.minimum_confidence
        clears_margin = margin >= profile.minimum_margin
        state = (
            DecisionState.SUCCESS
            if clears_confidence and clears_margin
            else DecisionState.ABSTAINED
        )
        code = None
        if not clears_confidence:
            code = "low_confidence"
        elif not clears_margin:
            code = "ambiguous_scores"
        choice = ranking[0] if state is DecisionState.SUCCESS else None
        choices = (choice,) if choice is not None else ()

    return _result(
        request,
        state=state,
        code=code,
        option_scores=scored,
        choice=choice,
        choices=choices,
        ranking=ranking,
        scalar_score=None,
        confidence=confidence,
        margin=margin,
        identity=identity,
        profile=profile,
        access=access,
        backend_warnings=raw.warnings,
    )


def _result(
    request: DecisionRequest,
    *,
    state: DecisionState,
    code: str | None,
    option_scores: Sequence[Mapping[str, Any]],
    choice: str | None,
    choices: Sequence[str],
    ranking: Sequence[str],
    scalar_score: float | None,
    confidence: float | None,
    margin: float | None,
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
    backend_warnings: Sequence[str] = (),
) -> DecisionResult:
    review_reasons = ["clinical_use_requires_review"]
    if code is not None:
        review_reasons.append(code)
    warnings = list(backend_warnings)
    warnings.append("human_review_required")
    if code is not None:
        warnings.append(code)
    return DecisionResult(
        mode=request.mode,
        state=state,
        code=code,
        option_scores=tuple(option_scores),
        choice=choice,
        choices=tuple(choices),
        ranking=tuple(ranking),
        scalar_score=scalar_score,
        confidence=confidence,
        margin=margin,
        calibration=profile.to_dict(),
        backend=identity.to_dict(),
        access=dict(access),
        warnings=tuple(warnings),
        review={"required": True, "reasons": review_reasons},
    )


def _terminal_result(
    request: DecisionRequest,
    state: DecisionState,
    code: str,
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    return _result(
        request,
        state=state,
        code=code,
        option_scores=(),
        choice=None,
        choices=(),
        ranking=(),
        scalar_score=None,
        confidence=None,
        margin=None,
        identity=identity,
        profile=profile,
        access=access,
    )


def _calibration_profile(
    request: DecisionRequest,
    profiles: Mapping[str, DecisionCalibrationProfile] | None,
) -> DecisionCalibrationProfile:
    available = profiles or {DEFAULT_CALIBRATION_ID: DEFAULT_CALIBRATION_PROFILE}
    try:
        profile = available[request.calibration_id]
    except KeyError:
        raise DecisionUnsupportedError(
            "decision calibration profile is unavailable"
        ) from None
    if not isinstance(profile, DecisionCalibrationProfile):
        raise TypeError("calibration profiles must contain DecisionCalibrationProfile")
    return profile


def _normalize_backend_output(
    value: Any,
    *,
    calibration_id: str,
) -> DecisionBackendOutput:
    if isinstance(value, DecisionBackendOutput):
        return value
    if isinstance(value, Mapping):
        return DecisionBackendOutput(
            option_scores=tuple(value.get("option_scores", ())),
            scalar_score=value.get("scalar_score"),
            confidence=value.get("confidence"),
            calibration_id=value.get("calibration_id", calibration_id),
            warnings=tuple(value.get("warnings", ())),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return DecisionBackendOutput(
            option_scores=tuple(value),
            calibration_id=calibration_id,
        )
    raise TypeError("decision scorer must return scores or a mapping")


def _validated_options(mode: DecisionMode, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("decision options must be a sequence of strings")
    options = tuple(_bounded_option(item) for item in values)
    if mode is DecisionMode.SCALAR_SCORE:
        if options:
            raise DecisionError("scalar_score requests cannot include options")
        return options
    if mode is DecisionMode.BOOLEAN_CHOICE and len(options) != 2:
        raise DecisionError("boolean_choice requests require exactly two options")
    if mode is not DecisionMode.BOOLEAN_CHOICE and not 2 <= len(options) <= MAX_OPTIONS:
        raise DecisionError(f"decision requests require 2 to {MAX_OPTIONS} options")
    normalized = [_normalize_option(item) for item in options]
    if len(normalized) != len(set(normalized)):
        raise DecisionError("decision options must be unique after normalization")
    if sum(len(item.encode("utf-8")) for item in options) > MAX_TOTAL_OPTION_BYTES:
        raise DecisionError("decision options exceed the aggregate byte limit")
    return options


def _bounded_input(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("input_text must be a string")
    if not value.strip():
        raise DecisionError("input_text must not be blank")
    if len(value) > MAX_INPUT_CHARS or len(value.encode("utf-8")) > MAX_INPUT_BYTES:
        raise DecisionError("input_text exceeds the decision input limit")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise DecisionError("input_text contains invalid characters")
    return value


def _bounded_option(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("decision options must be strings")
    normalized = value.strip()
    if not normalized:
        raise DecisionError("decision options must not be blank")
    if (
        len(normalized) > MAX_OPTION_CHARS
        or len(normalized.encode("utf-8")) > MAX_OPTION_BYTES
    ):
        raise DecisionError("a decision option exceeds its size limit")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise DecisionError("decision options cannot contain control characters")
    return normalized


def _normalize_option(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _lexical_option_score(input_text: str, option: str) -> float:
    source = _normalize_option(input_text)
    candidate = _normalize_option(option)
    source_tokens = set(_TOKEN_RE.findall(source))
    option_tokens = set(_TOKEN_RE.findall(candidate))
    if not option_tokens:
        return 0.0
    coverage = len(source_tokens & option_tokens) / len(option_tokens)
    substring_bonus = 0.15 if candidate in source else 0.0
    return min(1.0, 0.05 + 0.80 * coverage + substring_bonus)


def _deterministic_scalar(input_text: str) -> float | None:
    matches = re.findall(
        r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d|\.\d)",
        input_text,
    )
    if len(matches) != 1:
        return None
    value = float(matches[0])
    return value if 0.0 <= value <= 1.0 else None


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("decision metadata must be a mapping")
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _frozen_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("decision metadata must be JSON-compatible")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "DECISION_ADVISORY",
    "DECISION_COMPATIBILITY_POLICY",
    "DECISION_SCHEMA_VERSION",
    "DEFAULT_CALIBRATION_ID",
    "DEFAULT_CALIBRATION_PROFILE",
    "DETERMINISTIC_DECISION_BACKEND",
    "MAX_BATCH_SIZE",
    "MAX_INPUT_CHARS",
    "MAX_OPTIONS",
    "CallableDecisionBackend",
    "CrossEncoderDecisionBackend",
    "DecisionAccessPolicy",
    "DecisionBackend",
    "DecisionBackendIdentity",
    "DecisionBackendKind",
    "DecisionBackendOutput",
    "DecisionCalibrationExample",
    "DecisionCalibrationProfile",
    "DecisionConflictError",
    "DecisionError",
    "DecisionMode",
    "DecisionRequest",
    "DecisionResult",
    "DecisionState",
    "DecisionUnsupportedError",
    "DeterministicDecisionBackend",
    "EncoderDecisionBackend",
    "SpecialistDecisionBackend",
    "decide",
    "decide_batch",
    "decision_request_schema",
    "decision_result_schema",
    "evaluate_decision_calibration",
    "load_decision_schema",
    "migrate_decision_result",
]
