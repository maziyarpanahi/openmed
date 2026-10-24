"""Typed fixed-option decisions with calibrated abstention.

The contract is local-first and backend-neutral. It scores caller-declared
options, preserves their order, and never turns ambiguity or a backend failure
into an apparent success. Results are assistive review artifacts; they do not
authorize diagnosis, treatment, enrollment, outreach, ordering, or any other
patient-care action.

Schema 1.1.0 adds optional fields only: per-question instructions, option
descriptions, a named question set over one input, ordered score levels with an
expected-level output, JSON inputs, and a configurable input bound. Requests
that use none of them resolve to schema 1.0.0 and produce results that are
identical to the 1.0.0 contract.
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

DECISION_SCHEMA_VERSION: Final = "1.1.0"
DECISION_BASE_SCHEMA_VERSION: Final = "1.0.0"
SUPPORTED_DECISION_SCHEMA_VERSIONS: Final = (
    DECISION_BASE_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
)
DECISION_POLICY_VERSION: Final = "1.0.0"
DECISION_COMPATIBILITY_POLICY: Final = "same_major"
DEFAULT_CALIBRATION_ID: Final = "openmed.synthetic.fixed_option.v1"
DEFAULT_CALIBRATION_VERSION: Final = "1.0.0"
DEFAULT_QUESTION_ID: Final = "decision"
MAX_INPUT_CHARS: Final = 8192
MAX_INPUT_BYTES: Final = 32768
MAX_CONFIGURABLE_INPUT_CHARS: Final = 131072
INPUT_BYTES_PER_CHAR: Final = 4
MAX_OPTIONS: Final = 64
MAX_OPTION_CHARS: Final = 256
MAX_OPTION_BYTES: Final = 1024
MAX_TOTAL_OPTION_BYTES: Final = 16384
MAX_QUESTIONS: Final = 32
MAX_INSTRUCTIONS_CHARS: Final = 2048
MAX_DESCRIPTION_CHARS: Final = 1024
MAX_TOTAL_QUESTION_BYTES: Final = 65536
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
_QUESTION_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$")
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
    QUESTION_SET = "question_set"


QUESTION_MODES: Final = tuple(
    mode for mode in DecisionMode if mode is not DecisionMode.QUESTION_SET
)


class DecisionInputFormat(str, Enum):
    """How the shared decision input should be read."""

    TEXT = "text"
    JSON = "json"


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


def _bounded_prose(value: Any, name: str, *, maximum: int) -> str | None:
    """Validate optional caller prose such as instructions or descriptions."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null")
    if not value.strip() or len(value) > maximum:
        raise DecisionError(f"{name} must be bounded non-blank text or null")
    if len(value.encode("utf-8")) > maximum * INPUT_BYTES_PER_CHAR:
        raise DecisionError(f"{name} exceeds its byte limit")
    if any(
        (ord(char) < 32 and char not in "\n\r\t")
        or ord(char) == 127
        or 0xD800 <= ord(char) <= 0xDFFF
        for char in value
    ):
        raise DecisionError(f"{name} contains control characters")
    return value


def _question_id(value: Any, name: str = "question_id") -> str:
    if not isinstance(value, str) or _QUESTION_ID_RE.fullmatch(value) is None:
        raise DecisionError(f"{name} must be a bounded question identifier")
    return value


def _version_key(value: str) -> tuple[int, int, int]:
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def _contract_version(declared: str) -> str:
    """Return the newest supported schema version not newer than *declared*."""

    key = _version_key(declared)
    supported = [
        version
        for version in SUPPORTED_DECISION_SCHEMA_VERSIONS
        if _version_key(version) <= key
    ]
    return supported[-1] if supported else DECISION_BASE_SCHEMA_VERSION


def _is_base_version(version: str) -> bool:
    return _version_key(version)[:2] <= _version_key(DECISION_BASE_SCHEMA_VERSION)[:2]


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


_QUESTION_FIELDS: Final = frozenset(
    {
        "question_id",
        "mode",
        "options",
        "option_descriptions",
        "instructions",
        "levels",
    }
)
_REQUEST_BASE_FIELDS: Final = frozenset(
    {
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
)
_REQUEST_EXTENSION_FIELDS: Final = (
    "question_id",
    "instructions",
    "option_descriptions",
    "levels",
    "questions",
    "input_format",
    "max_input_chars",
)
_RESULT_EXTENSION_DEFAULTS: Final = MappingProxyType(
    {"questions": [], "level_scores": [], "expected_level": None}
)


@dataclass(frozen=True, slots=True)
class DecisionQuestion:
    """One named question scored against a request's shared input.

    Attributes:
        question_id: Bounded identifier, unique within a question set. It is the
            only question value that appears in safe metadata and audit output.
        mode: Decision shape for this question. Question sets cannot nest.
        options: Caller-declared options, scored in this order.
        option_descriptions: Optional descriptions aligned with ``options``;
            ``None`` entries mark options without a description.
        instructions: Optional question wording passed to the backend.
        levels: Ordered ``scalar_score`` levels, lowest first.
    """

    question_id: str
    mode: DecisionMode
    options: tuple[str, ...] = field(default_factory=tuple, repr=False)
    option_descriptions: tuple[str | None, ...] | None = field(default=None, repr=False)
    instructions: str | None = field(default=None, repr=False)
    levels: tuple[str, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        _question_id(self.question_id)
        mode = DecisionMode(self.mode)
        if mode is DecisionMode.QUESTION_SET:
            raise DecisionError("question sets cannot be nested")
        options, descriptions, instructions, levels = _validated_question_parts(
            mode,
            self.options,
            self.option_descriptions,
            self.instructions,
            self.levels,
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "option_descriptions", descriptions)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionQuestion":
        """Parse one strict question mapping.

        Args:
            payload: Mapping with ``question_id``, ``mode`` and optional
                ``options``, ``option_descriptions``, ``instructions`` and
                ``levels``.

        Returns:
            A validated question.

        Raises:
            DecisionError: If the mapping has unknown or missing fields or
                violates a bound.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("decision questions must be mappings")
        if set(payload) - _QUESTION_FIELDS:
            raise DecisionError("decision question contains unsupported fields")
        try:
            descriptions = payload.get("option_descriptions")
            return cls(
                question_id=payload["question_id"],
                mode=DecisionMode(payload["mode"]),
                options=_sequence(payload.get("options") or ()),
                option_descriptions=(
                    None if descriptions is None else _sequence(descriptions)
                ),
                instructions=payload.get("instructions"),
                levels=_sequence(payload.get("levels") or ()),
            )
        except KeyError:
            raise DecisionError(
                "decision question is missing a required field"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical question mapping, omitting unset fields."""

        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "mode": self.mode.value,
            "options": list(self.options),
        }
        if self.option_descriptions is not None:
            payload["option_descriptions"] = list(self.option_descriptions)
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.levels:
            payload["levels"] = list(self.levels)
        return payload

    def safe_metadata(self) -> dict[str, Any]:
        """Return identifiers, counts and a definition digest only.

        The digest covers options, descriptions, instructions and levels so an
        audit trail can prove which definition was scored without storing it.
        """

        return {
            "question_id": self.question_id,
            "mode": self.mode.value,
            "option_count": len(self.options),
            "level_count": len(self.levels),
            "has_instructions": self.instructions is not None,
            "definition_digest": _digest(self.to_dict()),
        }


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One bounded request shared by Python, REST, and MCP.

    The first nine fields are the 1.0.0 contract. The remaining fields are
    optional 1.1.0 extensions; a request that leaves them unset and does not
    declare ``schema_version`` resolves to 1.0.0 and behaves exactly as before.

    Attributes:
        mode: Decision shape, or ``question_set`` for several named questions.
        input_text: Transient input shared by every question. It never appears
            in results, safe metadata or audit output.
        options: Options for a single-question request.
        namespace: Controlled access-policy namespace.
        purpose: Controlled access-policy purpose.
        calibration_id: Calibration profile applied to every question.
        timeout_ms: Backend budget for the whole request.
        schema_version: Declared contract version. ``None`` resolves to 1.1.0
            when an extension field is set and to 1.0.0 otherwise.
        compatibility_policy: Always ``same_major``.
        question_id: Optional identifier for a single-question request.
        instructions: Optional question wording for a single-question request.
        option_descriptions: Optional descriptions aligned with ``options``.
        levels: Ordered ``scalar_score`` levels, lowest first.
        questions: Named questions for ``question_set`` requests.
        input_format: ``text`` or ``json`` (a JSON object or array as text).
        max_input_chars: Optional input bound up to
            ``MAX_CONFIGURABLE_INPUT_CHARS``; the default stays 8,192.
    """

    mode: DecisionMode
    input_text: str = field(repr=False)
    options: tuple[str, ...] = field(default_factory=tuple, repr=False)
    namespace: str = "default"
    purpose: str = "care_review"
    calibration_id: str = DEFAULT_CALIBRATION_ID
    timeout_ms: int = 5000
    schema_version: str | None = None
    compatibility_policy: str = DECISION_COMPATIBILITY_POLICY
    question_id: str | None = None
    instructions: str | None = field(default=None, repr=False)
    option_descriptions: tuple[str | None, ...] | None = field(default=None, repr=False)
    levels: tuple[str, ...] = field(default_factory=tuple, repr=False)
    questions: tuple[DecisionQuestion, ...] = field(default_factory=tuple, repr=False)
    input_format: DecisionInputFormat = DecisionInputFormat.TEXT
    max_input_chars: int | None = None

    def __post_init__(self) -> None:
        mode = DecisionMode(self.mode)
        input_format = DecisionInputFormat(self.input_format)
        max_input_chars = _input_bound(self.max_input_chars)
        text = _bounded_input(
            self.input_text,
            maximum=max_input_chars or MAX_INPUT_CHARS,
            input_format=input_format,
        )
        question_id: str | None = None
        questions: tuple[DecisionQuestion, ...] = ()
        if mode is DecisionMode.QUESTION_SET:
            if (
                self.options
                or self.option_descriptions is not None
                or self.instructions is not None
                or self.levels
                or self.question_id is not None
            ):
                raise DecisionError(
                    "question_set requests declare options inside questions"
                )
            questions = _validated_questions(self.questions)
            options: tuple[str, ...] = ()
            descriptions: tuple[str | None, ...] | None = None
            instructions: str | None = None
            levels: tuple[str, ...] = ()
        else:
            if self.questions:
                raise DecisionError("questions require question_set mode")
            if self.question_id is not None:
                question_id = _question_id(self.question_id)
            options, descriptions, instructions, levels = _validated_question_parts(
                mode,
                self.options,
                self.option_descriptions,
                self.instructions,
                self.levels,
            )
            if (
                _question_text_bytes(options, descriptions, instructions, levels)
                > MAX_TOTAL_QUESTION_BYTES
            ):
                raise DecisionError("questions exceed the aggregate byte limit")
        _controlled(self.namespace, "namespace")
        _controlled(self.purpose, "purpose")
        _controlled(self.calibration_id, "calibration_id")
        uses_extensions = (
            mode is DecisionMode.QUESTION_SET
            or question_id is not None
            or descriptions is not None
            or instructions is not None
            or bool(levels)
            or input_format is not DecisionInputFormat.TEXT
            or max_input_chars is not None
        )
        schema_version = self.schema_version
        if schema_version is None:
            schema_version = (
                DECISION_SCHEMA_VERSION
                if uses_extensions
                else DECISION_BASE_SCHEMA_VERSION
            )
        _semantic_version(schema_version, "decision schema version")
        if schema_version.split(".", 1)[0] != "1":
            raise DecisionUnsupportedError(
                "decision request schema major is unsupported"
            )
        if uses_extensions and _is_base_version(_contract_version(schema_version)):
            raise DecisionUnsupportedError(
                f"decision request fields require schema {DECISION_SCHEMA_VERSION}"
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
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "question_id", question_id)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "option_descriptions", descriptions)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "questions", questions)
        object.__setattr__(self, "input_format", input_format)
        object.__setattr__(self, "max_input_chars", max_input_chars)

    @property
    def contract_version(self) -> str:
        """Return the supported schema version this request is served under."""

        return _contract_version(str(self.schema_version))

    @property
    def input_char_limit(self) -> int:
        """Return the effective input character bound."""

        return self.max_input_chars or MAX_INPUT_CHARS

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionRequest":
        """Parse a strict request mapping without retaining caller ownership.

        ``None`` values for the optional 1.1.0 fields and ``schema_version``
        are treated as absent so validated transport models can be passed
        through unchanged.
        """

        unknown = set(payload) - _REQUEST_BASE_FIELDS - set(_REQUEST_EXTENSION_FIELDS)
        if unknown:
            raise DecisionError("decision request contains unsupported fields")
        descriptions = payload.get("option_descriptions")
        input_format = payload.get("input_format")
        try:
            return cls(
                mode=DecisionMode(payload["mode"]),
                input_text=payload["input_text"],
                options=tuple(payload.get("options", ())),
                namespace=payload.get("namespace", "default"),
                purpose=payload.get("purpose", "care_review"),
                calibration_id=payload.get("calibration_id", DEFAULT_CALIBRATION_ID),
                timeout_ms=payload.get("timeout_ms", 5000),
                schema_version=payload.get("schema_version"),
                compatibility_policy=payload.get(
                    "compatibility_policy", DECISION_COMPATIBILITY_POLICY
                ),
                question_id=payload.get("question_id"),
                instructions=payload.get("instructions"),
                option_descriptions=(
                    None if descriptions is None else _sequence(descriptions)
                ),
                levels=_sequence(payload.get("levels") or ()),
                questions=_sequence(payload.get("questions") or ()),
                input_format=(
                    DecisionInputFormat.TEXT
                    if input_format is None
                    else DecisionInputFormat(input_format)
                ),
                max_input_chars=payload.get("max_input_chars"),
            )
        except KeyError:
            raise DecisionError(
                "decision request is missing a required field"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical transport request, including transient input.

        Unset 1.1.0 fields are omitted, so a 1.0.0 request renders exactly as
        it did under the 1.0.0 contract.
        """

        payload: dict[str, Any] = {
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
        if self.question_id is not None:
            payload["question_id"] = self.question_id
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.option_descriptions is not None:
            payload["option_descriptions"] = list(self.option_descriptions)
        if self.levels:
            payload["levels"] = list(self.levels)
        if self.questions:
            payload["questions"] = [item.to_dict() for item in self.questions]
        if self.input_format is not DecisionInputFormat.TEXT:
            payload["input_format"] = self.input_format.value
        if self.max_input_chars is not None:
            payload["max_input_chars"] = self.max_input_chars
        return payload

    def as_questions(self) -> tuple[DecisionQuestion, ...]:
        """Return the request as named questions.

        A single-question request becomes one question named by
        ``question_id`` or ``DEFAULT_QUESTION_ID``.
        """

        if self.mode is DecisionMode.QUESTION_SET:
            return self.questions
        return (
            DecisionQuestion(
                question_id=self.question_id or DEFAULT_QUESTION_ID,
                mode=self.mode,
                options=self.options,
                option_descriptions=self.option_descriptions,
                instructions=self.instructions,
                levels=self.levels,
            ),
        )

    def for_question(self, question: DecisionQuestion) -> "DecisionRequest":
        """Return the single-question request a backend scores for *question*.

        The shared input, policy fields, calibration and timeout are carried
        over unchanged.
        """

        if not isinstance(question, DecisionQuestion):
            raise TypeError("question must be a DecisionQuestion")
        schema_version = (
            DECISION_SCHEMA_VERSION
            if _is_base_version(self.contract_version)
            else str(self.schema_version)
        )
        return DecisionRequest(
            mode=question.mode,
            input_text=self.input_text,
            options=question.options,
            namespace=self.namespace,
            purpose=self.purpose,
            calibration_id=self.calibration_id,
            timeout_ms=self.timeout_ms,
            schema_version=schema_version,
            compatibility_policy=self.compatibility_policy,
            question_id=question.question_id,
            instructions=question.instructions,
            option_descriptions=question.option_descriptions,
            levels=question.levels,
            input_format=self.input_format,
            max_input_chars=self.max_input_chars,
        )

    def safe_metadata(self) -> dict[str, Any]:
        """Return log-safe bounds metadata without text or option values.

        1.1.0 requests add the input format and bound plus one entry per
        question with its identifier, mode, counts and definition digest. The
        input, instructions, options and descriptions never appear.
        """

        metadata: dict[str, Any] = {
            "mode": self.mode.value,
            "input_chars": len(self.input_text),
            "input_bytes": len(self.input_text.encode("utf-8")),
            "option_count": len(self.options),
            "namespace": self.namespace,
            "purpose": self.purpose,
            "calibration_id": self.calibration_id,
            "schema_version": self.schema_version,
        }
        if not _is_base_version(self.contract_version):
            questions = self.as_questions()
            metadata.update(
                {
                    "input_format": self.input_format.value,
                    "input_char_limit": self.input_char_limit,
                    "question_count": len(questions),
                    "questions": [item.safe_metadata() for item in questions],
                }
            )
        return metadata


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


@runtime_checkable
class DecisionQuestionSetBackend(DecisionBackend, Protocol):
    """Optional protocol for backends that score a question set in one call.

    Backends without ``score_questions`` receive one single-question request
    per question through ``score``. Either way, calibration and abstention are
    applied per question by :func:`decide`.
    """

    def score_questions(
        self,
        request: DecisionRequest,
    ) -> Mapping[str, DecisionBackendOutput]:
        """Return one backend output per question identifier."""


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
        if request.mode is DecisionMode.QUESTION_SET:
            raise DecisionUnsupportedError("question sets are scored per question")
        if request.mode is DecisionMode.SCALAR_SCORE and request.levels:
            return DecisionBackendOutput(
                option_scores=tuple(
                    _lexical_option_score(request.input_text, level)
                    for level in request.levels
                ),
                calibration_id=request.calibration_id,
            )
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
        # Levelled scalar questions are scored like options over their levels.
        return _normalize_backend_output(
            self._scorer(
                request.input_text,
                request.options or request.levels,
                request.mode,
            ),
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
    policy_version: str = DECISION_POLICY_VERSION

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
class _Answer:
    """Validated selection fields shared by results and question results."""

    option_scores: tuple[Mapping[str, Any], ...]
    choices: tuple[str, ...]
    ranking: tuple[str, ...]
    level_scores: tuple[Mapping[str, Any], ...]


def _validated_answer(
    *,
    mode: DecisionMode,
    state: DecisionState,
    code: str | None,
    option_scores: Sequence[Mapping[str, Any]],
    choice: str | None,
    choices: Sequence[str],
    ranking: Sequence[str],
    scalar_score: float | None,
    confidence: float | None,
    margin: float | None,
    level_scores: Sequence[Mapping[str, Any]],
    expected_level: float | None,
) -> _Answer:
    if state is DecisionState.SUCCESS and code is not None:
        raise DecisionError("successful decision results cannot carry a code")
    if state is not DecisionState.SUCCESS and code is None:
        raise DecisionError("non-success decision results require a code")
    if code is not None:
        _controlled(code, "decision code")
    scored = tuple(_frozen_mapping(item) for item in option_scores)
    for position, item in enumerate(scored):
        if item.get("index") != position:
            raise DecisionError("decision option scores must preserve input order")
        _bounded_text(item.get("option"), "scored option", maximum=MAX_OPTION_CHARS)
        _probability(item.get("score"), "option score")
    levelled = tuple(_frozen_mapping(item) for item in level_scores)
    for position, item in enumerate(levelled):
        if item.get("index") != position:
            raise DecisionError("decision level scores must preserve level order")
        _bounded_text(item.get("level"), "scored level", maximum=MAX_OPTION_CHARS)
        _probability(item.get("score"), "level score")
    selected = tuple(str(item) for item in choices)
    ordered = tuple(str(item) for item in ranking)
    for value, name in (
        (scalar_score, "scalar score"),
        (confidence, "confidence"),
        (margin, "margin"),
    ):
        if value is not None:
            _probability(value, name)
    if expected_level is not None:
        if isinstance(expected_level, bool) or not isinstance(
            expected_level, int | float
        ):
            raise TypeError("expected level must be a number")
        highest = len(levelled) - 1 if levelled else MAX_OPTIONS - 1
        if not math.isfinite(expected_level) or not (
            0.0 <= float(expected_level) <= highest
        ):
            raise DecisionError("expected level must lie within the declared levels")
    if choice is not None:
        _bounded_text(choice, "choice", maximum=MAX_OPTION_CHARS)
        if choice not in selected or choice not in ordered:
            raise DecisionError("decision choice must appear in choices and ranking")
    if state is not DecisionState.SUCCESS and (
        choice is not None
        or selected
        or scalar_score is not None
        or expected_level is not None
    ):
        raise DecisionError("non-success decision results cannot carry selections")
    if mode is DecisionMode.SCALAR_SCORE:
        if state is DecisionState.SUCCESS and scalar_score is None:
            raise DecisionError("successful scalar decisions require a score")
        if scored or ordered:
            raise DecisionError("scalar decisions cannot carry option scores")
    else:
        if levelled or expected_level is not None:
            raise DecisionError("only scalar decisions can carry score levels")
        if state is DecisionState.SUCCESS and (
            choice is None or not scored or not ordered
        ):
            raise DecisionError(
                "successful option decisions require a choice and scores"
            )
    return _Answer(
        option_scores=scored,
        choices=selected,
        ranking=ordered,
        level_scores=levelled,
    )


def _validated_review(
    warnings: Sequence[str],
    review: Mapping[str, Any],
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    unique = tuple(dict.fromkeys(_controlled(item, "warning") for item in warnings))
    frozen = _frozen_mapping(review)
    if frozen.get("required") is not True or not frozen.get("reasons"):
        raise DecisionError("decision results always require review reasons")
    return unique, frozen


@dataclass(frozen=True, slots=True)
class DecisionQuestionResult:
    """Calibrated outcome for one named question of a question set.

    Every question carries its own state and code, so one abstaining or
    failing question never hides the others.
    """

    question_id: str
    mode: DecisionMode
    state: DecisionState
    code: str | None
    option_scores: tuple[Mapping[str, Any], ...]
    choice: str | None
    choices: tuple[str, ...]
    ranking: tuple[str, ...]
    scalar_score: float | None
    confidence: float | None
    margin: float | None
    level_scores: tuple[Mapping[str, Any], ...]
    expected_level: float | None
    warnings: tuple[str, ...]
    review: Mapping[str, Any]

    def __post_init__(self) -> None:
        _question_id(self.question_id)
        mode = DecisionMode(self.mode)
        if mode is DecisionMode.QUESTION_SET:
            raise DecisionError("question results cannot nest question sets")
        state = DecisionState(self.state)
        answer = _validated_answer(
            mode=mode,
            state=state,
            code=self.code,
            option_scores=self.option_scores,
            choice=self.choice,
            choices=self.choices,
            ranking=self.ranking,
            scalar_score=self.scalar_score,
            confidence=self.confidence,
            margin=self.margin,
            level_scores=self.level_scores,
            expected_level=self.expected_level,
        )
        warnings, review = _validated_review(self.warnings, self.review)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "option_scores", answer.option_scores)
        object.__setattr__(self, "choices", answer.choices)
        object.__setattr__(self, "ranking", answer.ranking)
        object.__setattr__(self, "level_scores", answer.level_scores)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "review", review)

    @classmethod
    def from_result(
        cls,
        question_id: str,
        result: "DecisionResult",
    ) -> "DecisionQuestionResult":
        """Project a single-question result onto a named question."""

        if result.mode is DecisionMode.QUESTION_SET:
            raise DecisionError("question results cannot nest question sets")
        return cls(
            question_id=question_id,
            mode=result.mode,
            state=result.state,
            code=result.code,
            option_scores=result.option_scores,
            choice=result.choice,
            choices=result.choices,
            ranking=result.ranking,
            scalar_score=result.scalar_score,
            confidence=result.confidence,
            margin=result.margin,
            level_scores=result.level_scores,
            expected_level=result.expected_level,
            warnings=result.warnings,
            review=result.review,
        )

    @property
    def abstained(self) -> bool:
        return self.state is DecisionState.ABSTAINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
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
            "level_scores": [_plain(item) for item in self.level_scores],
            "expected_level": self.expected_level,
            "warnings": list(self.warnings),
            "review": _plain(self.review),
        }

    def safe_metadata(self) -> dict[str, Any]:
        """Return identifiers, states and counts without option values."""

        return {
            "question_id": self.question_id,
            "mode": self.mode.value,
            "state": self.state.value,
            "code": self.code,
            "option_count": len(self.option_scores),
            "level_count": len(self.level_scores),
        }


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Canonical result shared by Python, REST, MCP, and persistence.

    ``schema_version`` follows the request: 1.0.0 results render exactly the
    1.0.0 fields, and 1.1.0 results add ``questions``, ``level_scores`` and
    ``expected_level``.
    """

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
    schema_version: str = DECISION_BASE_SCHEMA_VERSION
    compatibility_policy: str = DECISION_COMPATIBILITY_POLICY
    extensions: Mapping[str, Any] = field(default_factory=dict)
    questions: tuple[DecisionQuestionResult, ...] = ()
    level_scores: tuple[Mapping[str, Any], ...] = ()
    expected_level: float | None = None

    def __post_init__(self) -> None:
        mode = DecisionMode(self.mode)
        state = DecisionState(self.state)
        if self.autonomous_action:
            raise DecisionError("decision results cannot authorize autonomous action")
        if self.advisory != DECISION_ADVISORY:
            raise DecisionError("decision advisory cannot be replaced")
        _semantic_version(self.schema_version, "decision schema version")
        if self.compatibility_policy != DECISION_COMPATIBILITY_POLICY:
            raise DecisionUnsupportedError(
                "decision compatibility policy is unsupported"
            )
        questions = tuple(self.questions)
        if mode is DecisionMode.QUESTION_SET:
            if state is DecisionState.SUCCESS and self.code is not None:
                raise DecisionError("successful decision results cannot carry a code")
            if state is not DecisionState.SUCCESS and self.code is None:
                raise DecisionError("non-success decision results require a code")
            if self.code is not None:
                _controlled(self.code, "decision code")
            if (
                self.option_scores
                or self.choice is not None
                or self.choices
                or self.ranking
                or self.scalar_score is not None
                or self.confidence is not None
                or self.margin is not None
                or self.level_scores
                or self.expected_level is not None
            ):
                raise DecisionError(
                    "question-set results report selections per question"
                )
            if not 1 <= len(questions) <= MAX_QUESTIONS:
                raise DecisionError(
                    f"question-set results require 1 to {MAX_QUESTIONS} questions"
                )
            if not all(isinstance(item, DecisionQuestionResult) for item in questions):
                raise TypeError("questions must contain DecisionQuestionResult")
            identifiers = [item.question_id for item in questions]
            if len(identifiers) != len(set(identifiers)):
                raise DecisionError("question-set result identifiers must be unique")
            answer = _Answer((), (), (), ())
        else:
            if questions:
                raise DecisionError("only question-set results carry questions")
            answer = _validated_answer(
                mode=mode,
                state=state,
                code=self.code,
                option_scores=self.option_scores,
                choice=self.choice,
                choices=self.choices,
                ranking=self.ranking,
                scalar_score=self.scalar_score,
                confidence=self.confidence,
                margin=self.margin,
                level_scores=self.level_scores,
                expected_level=self.expected_level,
            )
        if _is_base_version(self.schema_version) and (
            mode is DecisionMode.QUESTION_SET
            or answer.level_scores
            or self.expected_level is not None
        ):
            raise DecisionError(
                f"question sets and score levels require schema {DECISION_SCHEMA_VERSION}"
            )
        warnings, review = _validated_review(self.warnings, self.review)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "option_scores", answer.option_scores)
        object.__setattr__(self, "choices", answer.choices)
        object.__setattr__(self, "ranking", answer.ranking)
        object.__setattr__(self, "level_scores", answer.level_scores)
        object.__setattr__(self, "questions", questions)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "calibration", _frozen_mapping(self.calibration))
        object.__setattr__(self, "backend", _frozen_mapping(self.backend))
        object.__setattr__(self, "access", _frozen_mapping(self.access))
        object.__setattr__(self, "review", review)
        object.__setattr__(self, "extensions", _frozen_mapping(self.extensions))

    @property
    def abstained(self) -> bool:
        return self.state is DecisionState.ABSTAINED

    def question_results(
        self,
        question_id: str = DEFAULT_QUESTION_ID,
    ) -> tuple[DecisionQuestionResult, ...]:
        """Return per-question results.

        Args:
            question_id: Name given to the only question of a single-question
                result. Ignored for question-set results.

        Returns:
            One result per question, in request order.
        """

        if self.mode is DecisionMode.QUESTION_SET:
            return self.questions
        return (DecisionQuestionResult.from_result(question_id, self),)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        }
        if not _is_base_version(self.schema_version):
            payload["level_scores"] = [_plain(item) for item in self.level_scores]
            payload["expected_level"] = self.expected_level
            payload["questions"] = [item.to_dict() for item in self.questions]
        payload.update(
            {
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
        )
        return payload

    def safe_metadata(self) -> dict[str, Any]:
        """Return log-safe states, counts and provenance without option values."""

        return {
            "mode": self.mode.value,
            "state": self.state.value,
            "code": self.code,
            "option_count": len(self.option_scores),
            "level_count": len(self.level_scores),
            "backend_id": self.backend.get("backend_id"),
            "calibration_id": self.calibration.get("calibration_id"),
            "schema_version": self.schema_version,
            "questions": [item.safe_metadata() for item in self.questions],
        }


def decide(
    request: DecisionRequest | Mapping[str, Any],
    *,
    backend: DecisionBackend | None = None,
    policy: DecisionAccessPolicy | None = None,
    calibration_profiles: Mapping[str, DecisionCalibrationProfile] | None = None,
) -> DecisionResult:
    """Score one bounded request and apply calibrated abstention policy.

    Question-set requests share one access decision and calibration profile.
    Each question is then scored, calibrated and abstains on its own, and the
    result reports every question with its own state.

    Args:
        request: A request or a request mapping.
        backend: Local backend; the deterministic backend by default. A backend
            with ``score_questions`` scores a question set in one call;
            otherwise each question is scored through ``score``.
        policy: Namespace, purpose and backend-family allowlist.
        calibration_profiles: Profiles keyed by calibration identifier.

    Returns:
        A review-only decision result. Failures, denials and abstentions are
        typed states, never fabricated selections.
    """

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
    if normalized.mode is DecisionMode.QUESTION_SET:
        return _decide_question_set(normalized, selected_backend, profile, access)
    return _score_request(normalized, selected_backend, profile, access)


def _score_request(
    request: DecisionRequest,
    backend: DecisionBackend,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
    *,
    started: float | None = None,
) -> DecisionResult:
    identity = backend.identity
    if started is None:
        started = time.monotonic()
    try:
        raw = backend.score(request)
    except Exception as error:
        state, code = _backend_error_state(error)
        return _terminal_result(request, state, code, identity, profile, access)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if elapsed_ms > request.timeout_ms:
        return _terminal_result(
            request,
            DecisionState.FAILURE,
            "backend_timeout",
            identity,
            profile,
            access,
        )
    return _result_from_output(request, raw, identity, profile, access)


def _decide_question_set(
    request: DecisionRequest,
    backend: DecisionBackend,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    identity = backend.identity
    questions = request.questions
    sub_requests = tuple(request.for_question(question) for question in questions)
    started = time.monotonic()
    outcomes: list[DecisionResult] = []
    if isinstance(backend, DecisionQuestionSetBackend):
        try:
            outputs = backend.score_questions(request)
        except Exception as error:
            state, code = _backend_error_state(error)
            return _terminal_result(request, state, code, identity, profile, access)
        if (time.monotonic() - started) * 1000.0 > request.timeout_ms:
            return _terminal_result(
                request,
                DecisionState.FAILURE,
                "backend_timeout",
                identity,
                profile,
                access,
            )
        if not isinstance(outputs, Mapping) or set(outputs) != {
            question.question_id for question in questions
        }:
            return _terminal_result(
                request,
                DecisionState.FAILURE,
                "backend_contract_invalid",
                identity,
                profile,
                access,
            )
        for question, sub_request in zip(questions, sub_requests, strict=True):
            outcomes.append(
                _result_from_output(
                    sub_request,
                    outputs[question.question_id],
                    identity,
                    profile,
                    access,
                )
            )
    else:
        for sub_request in sub_requests:
            if (time.monotonic() - started) * 1000.0 > request.timeout_ms:
                outcomes.append(
                    _terminal_result(
                        sub_request,
                        DecisionState.FAILURE,
                        "backend_timeout",
                        identity,
                        profile,
                        access,
                    )
                )
                continue
            outcomes.append(
                _score_request(
                    sub_request,
                    backend,
                    profile,
                    access,
                    started=started,
                )
            )
    return _question_set_result(request, outcomes, identity, profile, access)


def _backend_error_state(error: Exception) -> tuple[DecisionState, str]:
    if isinstance(error, DecisionUnsupportedError):
        return DecisionState.UNSUPPORTED, "backend_unsupported"
    if isinstance(error, DecisionConflictError):
        return DecisionState.CONFLICT, "backend_conflict"
    if isinstance(error, TimeoutError):
        return DecisionState.FAILURE, "backend_timeout"
    return DecisionState.FAILURE, "backend_failure"


def _result_from_output(
    request: DecisionRequest,
    raw: Any,
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    if not isinstance(raw, DecisionBackendOutput):
        return _terminal_result(
            request,
            DecisionState.FAILURE,
            "backend_contract_invalid",
            identity,
            profile,
            access,
        )
    if raw.calibration_id != profile.calibration_id:
        return _terminal_result(
            request,
            DecisionState.CONFLICT,
            "calibration_mismatch",
            identity,
            profile,
            access,
        )
    try:
        return _decision_from_scores(request, raw, identity, profile, access)
    except (DecisionError, TypeError, ValueError):
        return _terminal_result(
            request,
            DecisionState.FAILURE,
            "backend_contract_invalid",
            identity,
            profile,
            access,
        )


def _question_set_result(
    request: DecisionRequest,
    outcomes: Sequence[DecisionResult],
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    question_results = tuple(
        DecisionQuestionResult.from_result(question.question_id, outcome)
        for question, outcome in zip(request.questions, outcomes, strict=True)
    )
    states = {item.state for item in question_results}
    outcomes_by_code = {(item.state, item.code) for item in question_results}
    code: str | None
    if states == {DecisionState.SUCCESS}:
        state, code = DecisionState.SUCCESS, None
    elif len(outcomes_by_code) == 1:
        state, code = next(iter(outcomes_by_code))
    elif len(states) == 1:
        state = next(iter(states))
        code = f"question_set_{state.value}"
    else:
        state, code = DecisionState.PARTIAL, "question_set_partial"
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
        questions=question_results,
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


def decision_audit_record(
    request: DecisionRequest,
    result: DecisionResult,
) -> dict[str, Any]:
    """Return a text-free audit record for one decision.

    Args:
        request: The request that was decided.
        result: Its result.

    Returns:
        Question identifiers, modes, states, codes, counts, provenance and
        question-definition digests. The input, instructions, options and
        option descriptions are never included.
    """

    if not isinstance(request, DecisionRequest):
        raise TypeError("request must be a DecisionRequest")
    if not isinstance(result, DecisionResult):
        raise TypeError("result must be a DecisionResult")
    return {
        "schema_version": result.schema_version,
        "request": request.safe_metadata(),
        "result": result.safe_metadata(),
        "contains_source_text": False,
    }


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


def _option_items_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": MAX_OPTION_CHARS}


def _options_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": MAX_OPTIONS,
        "items": _option_items_schema(),
        "default": [],
    }


def _option_mode_rules() -> list[dict[str, Any]]:
    """Return the 1.0.0 option-count rules plus the score-level rule."""

    return [
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
        {
            "if": {"properties": {"mode": {"const": "scalar_score"}}},
            "else": {"not": {"required": ["levels"]}},
        },
    ]


def _question_definition_properties() -> dict[str, Any]:
    return {
        "options": _options_schema(),
        "option_descriptions": {
            "type": "array",
            "maxItems": MAX_OPTIONS,
            "description": (
                "Optional descriptions aligned with options; null marks an "
                "option without a description."
            ),
            "items": {
                "anyOf": [
                    {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_DESCRIPTION_CHARS,
                    },
                    {"type": "null"},
                ]
            },
        },
        "instructions": {
            "anyOf": [
                {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_INSTRUCTIONS_CHARS,
                },
                {"type": "null"},
            ],
            "description": "Optional question wording passed to the backend.",
        },
        "levels": {
            "type": "array",
            "minItems": 2,
            "maxItems": MAX_OPTIONS,
            "items": _option_items_schema(),
            "description": "Ordered scalar_score levels, lowest first.",
        },
    }


def decision_request_schema() -> dict[str, Any]:
    """Return the canonical request JSON Schema.

    The schema accepts every 1.0.0 request unchanged. A request that declares
    ``schema_version`` 1.0.0 cannot use the 1.1.0 fields, and the input stays
    bounded to 8,192 characters unless ``max_input_chars`` raises the bound.
    """

    question_id = {"type": "string", "pattern": _QUESTION_ID_RE.pattern}
    question = {
        "type": "object",
        "additionalProperties": False,
        "required": ["question_id", "mode"],
        "allOf": _option_mode_rules(),
        "properties": {
            "question_id": question_id,
            "mode": {
                "type": "string",
                "enum": [item.value for item in QUESTION_MODES],
            },
            **_question_definition_properties(),
        },
    }
    single_question_fields = (
        "question_id",
        "instructions",
        "option_descriptions",
        "levels",
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://openmed.dev/schemas/decision-request/{DECISION_SCHEMA_VERSION}",
        "title": "OpenMed Fixed-Option Decision Request",
        "type": "object",
        "additionalProperties": False,
        "required": ["mode", "input_text"],
        "allOf": [
            *_option_mode_rules(),
            {
                "if": {"properties": {"mode": {"const": "question_set"}}},
                "then": {
                    "required": ["questions"],
                    "properties": {"options": {"maxItems": 0}},
                    "not": {
                        "anyOf": [
                            {"required": [name]} for name in single_question_fields
                        ]
                    },
                },
                "else": {"not": {"required": ["questions"]}},
            },
            {
                "if": {
                    "required": ["schema_version"],
                    "properties": {
                        "schema_version": {"const": DECISION_BASE_SCHEMA_VERSION}
                    },
                },
                "then": {
                    "properties": {"mode": {"not": {"const": "question_set"}}},
                    "not": {
                        "anyOf": [
                            {"required": [name]} for name in _REQUEST_EXTENSION_FIELDS
                        ]
                    },
                },
            },
            {
                "if": {"not": {"required": ["max_input_chars"]}},
                "then": {"properties": {"input_text": {"maxLength": MAX_INPUT_CHARS}}},
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
                "maxLength": MAX_CONFIGURABLE_INPUT_CHARS,
            },
            "options": _options_schema(),
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
                "enum": list(SUPPORTED_DECISION_SCHEMA_VERSIONS),
                "description": (
                    f"Omitted, it resolves to {DECISION_SCHEMA_VERSION} when a "
                    f"{DECISION_SCHEMA_VERSION} field is present and to "
                    f"{DECISION_BASE_SCHEMA_VERSION} otherwise."
                ),
            },
            "compatibility_policy": {
                "type": "string",
                "const": DECISION_COMPATIBILITY_POLICY,
                "default": DECISION_COMPATIBILITY_POLICY,
            },
            "question_id": question_id,
            **{
                name: schema
                for name, schema in _question_definition_properties().items()
                if name != "options"
            },
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS,
                "items": question,
                "description": "Named questions scored against the one input.",
            },
            "input_format": {
                "type": "string",
                "enum": [item.value for item in DecisionInputFormat],
                "default": DecisionInputFormat.TEXT.value,
            },
            "max_input_chars": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CONFIGURABLE_INPUT_CHARS,
                "description": (
                    f"Input character bound; {MAX_INPUT_CHARS} when omitted."
                ),
            },
        },
    }


def decision_result_schema() -> dict[str, Any]:
    """Return the canonical result JSON Schema.

    1.0.0 results keep exactly the 1.0.0 fields. 1.1.0 results always carry
    ``questions``, ``level_scores`` and ``expected_level``.
    """

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
    answer_properties = {
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
    }
    level_properties = {
        "level_scores": {
            "type": "array",
            "maxItems": MAX_OPTIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "level", "score"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "level": {"type": "string", "maxLength": MAX_OPTION_CHARS},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "expected_level": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": MAX_OPTIONS - 1,
        },
    }
    level_rule = {
        "if": {"properties": {"mode": {"const": "scalar_score"}}},
        "else": {
            "properties": {
                "level_scores": {"maxItems": 0},
                "expected_level": {"type": "null"},
            }
        },
    }
    question_result = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "question_id",
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
            "level_scores",
            "expected_level",
            "warnings",
            "review",
        ],
        "allOf": [level_rule],
        "properties": {
            "question_id": {"type": "string", "pattern": _QUESTION_ID_RE.pattern},
            "mode": {
                "type": "string",
                "enum": [item.value for item in QUESTION_MODES],
            },
            **answer_properties,
            **level_properties,
        },
    }
    extension_fields = list(_RESULT_EXTENSION_DEFAULTS)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://openmed.dev/schemas/decision-result/{DECISION_SCHEMA_VERSION}",
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
        "allOf": [
            {
                "if": {
                    "properties": {
                        "schema_version": {"const": DECISION_BASE_SCHEMA_VERSION}
                    }
                },
                "then": {
                    "properties": {"mode": {"not": {"const": "question_set"}}},
                    "not": {
                        "anyOf": [{"required": [name]} for name in extension_fields]
                    },
                },
                "else": {"required": extension_fields},
            },
            {
                "if": {"properties": {"mode": {"const": "question_set"}}},
                "then": {"properties": {"questions": {"minItems": 1}}},
                "else": {"properties": {"questions": {"maxItems": 0}}},
            },
            level_rule,
        ],
        "properties": {
            "mode": {
                "type": "string",
                "enum": [item.value for item in DecisionMode],
            },
            **answer_properties,
            "calibration": calibration,
            "backend": backend,
            "access": access,
            "advisory": {"type": "string", "const": DECISION_ADVISORY},
            "autonomous_action": {"type": "boolean", "const": False},
            "schema_version": {
                "type": "string",
                "enum": list(SUPPORTED_DECISION_SCHEMA_VERSIONS),
            },
            "compatibility_policy": {
                "type": "string",
                "const": DECISION_COMPATIBILITY_POLICY,
            },
            "extensions": {"type": "object"},
            **level_properties,
            "questions": {
                "type": "array",
                "maxItems": MAX_QUESTIONS,
                "items": question_result,
            },
        },
    }


def migrate_decision_result(
    payload: Mapping[str, Any],
    *,
    target_version: str = DECISION_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Migrate same-major results while retaining unknown fields in extensions.

    Downgrading to 1.0.0 moves non-empty ``questions``, ``level_scores`` and
    ``expected_level`` values into ``extensions`` and drops their empty
    defaults. Upgrading to 1.1.0 lifts them back out of ``extensions`` or adds
    the empty defaults, so a downgrade followed by an upgrade is lossless.

    Args:
        payload: A serialized decision result.
        target_version: Same-major schema version to migrate to.

    Returns:
        A new result mapping at ``target_version``.

    Raises:
        DecisionUnsupportedError: If the major versions differ, or a
            question-set result is downgraded below 1.1.0.
    """

    source_version = _semantic_version(
        payload.get("schema_version"), "decision schema version"
    )
    target = _semantic_version(target_version, "target decision schema version")
    if source_version.split(".", 1)[0] != target.split(".", 1)[0]:
        raise DecisionUnsupportedError("decision result schema major is unsupported")
    downgrade = _is_base_version(target)
    if downgrade and payload.get("mode") == DecisionMode.QUESTION_SET.value:
        raise DecisionUnsupportedError(
            f"question-set results require schema {DECISION_SCHEMA_VERSION}"
        )
    known = set(decision_result_schema()["properties"])
    if downgrade:
        known -= set(_RESULT_EXTENSION_DEFAULTS)
    migrated = {key: deepcopy(value) for key, value in payload.items() if key in known}
    extensions = dict(migrated.get("extensions") or {})
    for key, value in payload.items():
        if key in known:
            continue
        if (
            key in _RESULT_EXTENSION_DEFAULTS
            and value == _RESULT_EXTENSION_DEFAULTS[key]
        ):
            continue
        extensions[key] = deepcopy(value)
    if not downgrade:
        for key, default in _RESULT_EXTENSION_DEFAULTS.items():
            if key not in migrated:
                migrated[key] = (
                    extensions.pop(key) if key in extensions else deepcopy(default)
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
    if request.mode is DecisionMode.SCALAR_SCORE and request.levels:
        return _levelled_decision(request, raw, identity, profile, access)
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


def _levelled_decision(
    request: DecisionRequest,
    raw: DecisionBackendOutput,
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    """Score ordered levels as an expected level and a normalized scalar.

    A level distribution yields the expected level index, a scalar in
    ``[0, 1]`` (expected level over the highest index) and a confidence equal
    to the most likely level's normalized probability. A backend may instead
    return a scalar and confidence directly.
    """

    levels = request.levels
    highest = len(levels) - 1
    if raw.option_scores:
        if raw.scalar_score is not None or len(raw.option_scores) != len(levels):
            raise DecisionConflictError("level backend output has invalid shape")
        total = math.fsum(raw.option_scores)
        if total <= 0.0:
            raise DecisionConflictError("level backend output has no probability")
        distribution = [score / total for score in raw.option_scores]
        expected = math.fsum(
            index * probability for index, probability in enumerate(distribution)
        )
        confidence = max(distribution)
        level_scores = tuple(
            {"index": index, "level": level, "score": score}
            for index, (level, score) in enumerate(
                zip(levels, raw.option_scores, strict=True)
            )
        )
    elif raw.scalar_score is not None:
        expected = raw.scalar_score * highest
        confidence = raw.confidence if raw.confidence is not None else 0.0
        level_scores = ()
    else:
        raise DecisionConflictError("level backend output has invalid shape")
    expected = min(float(highest), max(0.0, expected))
    confidence = min(1.0, max(0.0, confidence))
    scalar = min(1.0, max(0.0, expected / highest))
    state = (
        DecisionState.SUCCESS
        if confidence >= profile.minimum_confidence
        else DecisionState.ABSTAINED
    )
    succeeded = state is DecisionState.SUCCESS
    return _result(
        request,
        state=state,
        code=None if succeeded else "low_confidence",
        option_scores=(),
        choice=None,
        choices=(),
        ranking=(),
        scalar_score=scalar if succeeded else None,
        confidence=confidence,
        margin=None,
        identity=identity,
        profile=profile,
        access=access,
        backend_warnings=raw.warnings,
        level_scores=level_scores,
        expected_level=expected if succeeded else None,
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
    level_scores: Sequence[Mapping[str, Any]] = (),
    expected_level: float | None = None,
    questions: Sequence[DecisionQuestionResult] = (),
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
        schema_version=request.contract_version,
        questions=tuple(questions),
        level_scores=tuple(level_scores),
        expected_level=expected_level,
    )


def _terminal_result(
    request: DecisionRequest,
    state: DecisionState,
    code: str,
    identity: DecisionBackendIdentity,
    profile: DecisionCalibrationProfile,
    access: Mapping[str, Any],
) -> DecisionResult:
    if request.mode is DecisionMode.QUESTION_SET:
        outcomes = tuple(
            _terminal_result(
                request.for_question(question),
                state,
                code,
                identity,
                profile,
                access,
            )
            for question in request.questions
        )
        return _question_set_result(request, outcomes, identity, profile, access)
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


def _validated_levels(mode: DecisionMode, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("score levels must be a sequence of strings")
    if not values:
        return ()
    if mode is not DecisionMode.SCALAR_SCORE:
        raise DecisionError("score levels require scalar_score mode")
    levels = []
    for item in values:
        try:
            levels.append(_bounded_option(item))
        except TypeError:
            raise TypeError("score levels must be strings") from None
        except DecisionError:
            raise DecisionError("score levels must be bounded text") from None
    if not 2 <= len(levels) <= MAX_OPTIONS:
        raise DecisionError(f"score levels require 2 to {MAX_OPTIONS} entries")
    normalized = [_normalize_option(item) for item in levels]
    if len(normalized) != len(set(normalized)):
        raise DecisionError("score levels must be unique after normalization")
    if sum(len(item.encode("utf-8")) for item in levels) > MAX_TOTAL_OPTION_BYTES:
        raise DecisionError("score levels exceed the aggregate byte limit")
    return tuple(levels)


def _validated_question_parts(
    mode: DecisionMode,
    options: Any,
    option_descriptions: Any,
    instructions: Any,
    levels: Any,
) -> tuple[tuple[str, ...], tuple[str | None, ...] | None, str | None, tuple[str, ...]]:
    validated_options = _validated_options(mode, options)
    descriptions: tuple[str | None, ...] | None = None
    if option_descriptions is not None:
        if isinstance(option_descriptions, (str, bytes)) or not isinstance(
            option_descriptions, Sequence
        ):
            raise TypeError("option_descriptions must be a sequence")
        if not validated_options:
            raise DecisionError("option_descriptions require options")
        descriptions = tuple(
            _bounded_prose(item, "option description", maximum=MAX_DESCRIPTION_CHARS)
            for item in option_descriptions
        )
        if len(descriptions) != len(validated_options):
            raise DecisionError("option_descriptions must align with options")
    text = _bounded_prose(instructions, "instructions", maximum=MAX_INSTRUCTIONS_CHARS)
    validated_levels = _validated_levels(mode, levels)
    return validated_options, descriptions, text, validated_levels


def _question_text_bytes(
    options: Sequence[str],
    descriptions: Sequence[str | None] | None,
    instructions: str | None,
    levels: Sequence[str],
) -> int:
    parts = [
        *options,
        *(item for item in descriptions or () if item is not None),
        *levels,
    ]
    if instructions is not None:
        parts.append(instructions)
    return sum(len(item.encode("utf-8")) for item in parts)


def _validated_questions(values: Any) -> tuple[DecisionQuestion, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise TypeError("questions must be a sequence of questions")
    questions = tuple(
        item if isinstance(item, DecisionQuestion) else DecisionQuestion.from_dict(item)
        for item in values
    )
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise DecisionError(
            f"question_set requests require 1 to {MAX_QUESTIONS} questions"
        )
    identifiers = [item.question_id for item in questions]
    if len(identifiers) != len(set(identifiers)):
        raise DecisionError("question identifiers must be unique")
    total_bytes = sum(
        _question_text_bytes(
            item.options,
            item.option_descriptions,
            item.instructions,
            item.levels,
        )
        for item in questions
    )
    if total_bytes > MAX_TOTAL_QUESTION_BYTES:
        raise DecisionError("questions exceed the aggregate byte limit")
    return questions


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError("decision request lists must be sequences")
    return tuple(value)


def _input_bound(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= MAX_CONFIGURABLE_INPUT_CHARS:
        raise DecisionError(
            f"max_input_chars must be between 1 and {MAX_CONFIGURABLE_INPUT_CHARS}"
        )
    return value


def _reject_json_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON numbers are not supported")


def _bounded_input(
    value: Any,
    *,
    maximum: int = MAX_INPUT_CHARS,
    input_format: DecisionInputFormat = DecisionInputFormat.TEXT,
) -> str:
    if not isinstance(value, str):
        raise TypeError("input_text must be a string")
    if not value.strip():
        raise DecisionError("input_text must not be blank")
    if (
        len(value) > maximum
        or len(value.encode("utf-8")) > maximum * INPUT_BYTES_PER_CHAR
    ):
        raise DecisionError("input_text exceeds the decision input limit")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise DecisionError("input_text contains invalid characters")
    if input_format is DecisionInputFormat.JSON:
        try:
            parsed = json.loads(value, parse_constant=_reject_json_constant)
        except (ValueError, RecursionError):
            raise DecisionError("input_text must be a JSON object or array") from None
        if not isinstance(parsed, (dict, list)):
            raise DecisionError("input_text must be a JSON object or array")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    "DECISION_BASE_SCHEMA_VERSION",
    "DECISION_COMPATIBILITY_POLICY",
    "DECISION_POLICY_VERSION",
    "DECISION_SCHEMA_VERSION",
    "DEFAULT_CALIBRATION_ID",
    "DEFAULT_CALIBRATION_PROFILE",
    "DEFAULT_QUESTION_ID",
    "DETERMINISTIC_DECISION_BACKEND",
    "MAX_BATCH_SIZE",
    "MAX_CONFIGURABLE_INPUT_CHARS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_INPUT_CHARS",
    "MAX_INSTRUCTIONS_CHARS",
    "MAX_OPTIONS",
    "MAX_QUESTIONS",
    "QUESTION_MODES",
    "SUPPORTED_DECISION_SCHEMA_VERSIONS",
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
    "DecisionInputFormat",
    "DecisionMode",
    "DecisionQuestion",
    "DecisionQuestionResult",
    "DecisionQuestionSetBackend",
    "DecisionRequest",
    "DecisionResult",
    "DecisionState",
    "DecisionUnsupportedError",
    "DeterministicDecisionBackend",
    "EncoderDecisionBackend",
    "SpecialistDecisionBackend",
    "decide",
    "decide_batch",
    "decision_audit_record",
    "decision_request_schema",
    "decision_result_schema",
    "evaluate_decision_calibration",
    "load_decision_schema",
    "migrate_decision_result",
]
