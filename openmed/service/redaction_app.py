"""Local-only FastAPI service for text and file redaction.

The default redactor is a small deterministic, standard-library baseline.  It
does not load a model or make a network request.  Deployments that already
have a model on disk can inject :func:`local_model_redactor` explicitly.

Only aggregate counts and artifact status are retained for the review page.
Source text, redacted text, entity surfaces, and caller-supplied file contents
are deliberately excluded from service state, logs, and error responses.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import os
import re
import tempfile
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.policy import canonical_policy_name
from openmed.core.safety_sweep import safety_sweep

from .security_headers import DEFAULT_TRUSTED_HOSTS, ErrorEnvelopeTrustedHostMiddleware

try:
    from pydantic import BaseModel, ConfigDict, Field, StrictInt

    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - retained for older service installs
    from pydantic import BaseModel, Field, StrictInt

    ConfigDict = None  # type: ignore[assignment,misc]
    _PYDANTIC_V2 = False


DEFAULT_POLICY = "strict_no_leak"
RedactionMethod = Literal[
    "mask", "remove", "replace", "hash", "shift_dates", "format_preserve"
]

DEFAULT_METHOD: RedactionMethod = "mask"
MAX_TEXT_CHARS = 4_000_000
MAX_PATH_CHARS = 4_096
MAX_REQUEST_BODY_BYTES = MAX_TEXT_CHARS * 12 + 65_536
MIN_SEED = -(2**63)
MAX_SEED = 2**63 - 1
SUPPORTED_METHODS: frozenset[RedactionMethod] = frozenset(
    {"mask", "remove", "replace", "hash", "shift_dates", "format_preserve"}
)
_SAFE_POLICY = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SAFE_REQUEST_FIELDS = frozenset(
    {"text", "input_path", "output_path", "policy", "method", "seed"}
)
_SAFE_COUNTER_LABELS = frozenset(CANONICAL_LABELS) | frozenset(
    {"ID", "NAME", "UNKNOWN"}
)


class RedactionServiceError(Exception):
    """Typed, content-free error raised by the local redaction service."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _BoundedRequestBodyMiddleware:
    """Reject oversized HTTP request bodies before JSON parsing allocates them."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await _error_response(
                    400,
                    "bad_request",
                    "Request metadata is invalid",
                )(scope, receive, send)
                return
            if declared_length < 0:
                await _error_response(
                    400,
                    "bad_request",
                    "Request metadata is invalid",
                )(scope, receive, send)
                return
            if declared_length > self.max_bytes:
                await _error_response(
                    413,
                    "budget_exceeded",
                    "Request body exceeds the configured limit",
                )(scope, receive, send)
                return

        messages: list[Message] = []
        received_bytes = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            received_bytes += len(body)
            if received_bytes > self.max_bytes:
                await _error_response(
                    413,
                    "budget_exceeded",
                    "Request body exceeds the configured limit",
                )(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay, send)


class _RequestModel(BaseModel):
    """Reject unknown request fields so input paths stay explicit."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover

        class Config:
            extra = "forbid"


class TextRedactionRequest(_RequestModel):
    """Request body for the text redaction endpoint."""

    text: str = Field(..., max_length=MAX_TEXT_CHARS)
    output_path: str | None = Field(default=None, max_length=MAX_PATH_CHARS)
    policy: str = Field(default=DEFAULT_POLICY, max_length=64)
    method: str = Field(default=DEFAULT_METHOD, max_length=32)
    seed: StrictInt = 0


class FileRedactionRequest(_RequestModel):
    """Request body for the explicit input/output file endpoint."""

    input_path: str = Field(..., max_length=MAX_PATH_CHARS)
    output_path: str = Field(..., max_length=MAX_PATH_CHARS)
    policy: str = Field(default=DEFAULT_POLICY, max_length=64)
    method: str = Field(default=DEFAULT_METHOD, max_length=32)
    seed: StrictInt = 0


class RedactionRequest(_RequestModel):
    """Compatibility body for callers that use one redaction route."""

    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    input_path: str | None = Field(default=None, max_length=MAX_PATH_CHARS)
    output_path: str | None = Field(default=None, max_length=MAX_PATH_CHARS)
    policy: str = Field(default=DEFAULT_POLICY, max_length=64)
    method: str = Field(default=DEFAULT_METHOD, max_length=32)
    seed: StrictInt = 0


@dataclass(frozen=True, repr=False)
class RedactionResult:
    """Content-free result metadata plus the redacted artifact text."""

    redacted_text: str
    entity_counts: Mapping[str, int] = field(default_factory=dict)
    input_characters: int = 0

    def __post_init__(self) -> None:
        redacted_text = _plain_text(self.redacted_text)
        if redacted_text is None:
            raise RedactionServiceError("redaction_failed")
        if type(self.input_characters) is not int or self.input_characters < 0:
            raise RedactionServiceError("redaction_failed")
        object.__setattr__(self, "redacted_text", redacted_text)
        object.__setattr__(self, "entity_counts", _normalize_counts(self.entity_counts))

    @property
    def total_entities(self) -> int:
        """Return the aggregate number of redacted entities."""

        return sum(self.entity_counts.values())

    def __repr__(self) -> str:
        return (
            "RedactionResult("
            f"total_entities={self.total_entities}, "
            f"input_characters={self.input_characters}, "
            f"output_characters={len(self.redacted_text)})"
        )


@dataclass(frozen=True)
class _ReviewState:
    """The only operation state retained by the app."""

    status: str = "idle"
    policy: str = DEFAULT_POLICY
    method: str = DEFAULT_METHOD
    kind: str = "none"
    artifact_status: str = "not_started"
    entity_counts: Mapping[str, int] = field(default_factory=dict)
    input_characters: int = 0
    output_characters: int = 0

    def to_dict(self) -> dict[str, Any]:
        counts = _normalize_counts(self.entity_counts)
        return {
            "status": self.status,
            "policy": self.policy,
            "method": self.method,
            "kind": self.kind,
            "artifact": {"status": self.artifact_status},
            "counts": {
                "total_entities": sum(counts.values()),
                "by_label": counts,
            },
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
        }


@dataclass(frozen=True, repr=False)
class _ArtifactResult:
    """Operation result returned by the service layer."""

    result: RedactionResult
    artifact_status: str
    artifact_sha256: str | None
    policy: str
    method: RedactionMethod
    kind: Literal["text", "file"]


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    label: str
    priority: int


_EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9-])"
)
_SSN_PATTERN = re.compile(r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-])\d{3}[ .-]\d{4}(?!\w)"
    r"|(?<!\w)\d{3}[ -]\d{4}(?!\w)"
)
_IP_PATTERN = re.compile(r"(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)")
_DATE_PATTERN = re.compile(
    r"(?<!\w)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2})(?!\w)"
)
_NAME_CONTEXT_PATTERN = re.compile(
    r"\b(?i:patient|name|mr|mrs|ms|dr)\b[:#]?\s+"
    r"(?P<value>[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})"
)
_ID_CONTEXT_PATTERN = re.compile(
    r"\b(?:mrn|patient\s+id|record\s+id)\b[:#]?\s+"
    r"(?P<value>(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{2,})",
    re.IGNORECASE,
)


def _plain_text(value: object) -> str | None:
    """Copy a string into a base ``str`` without invoking subclass hooks."""

    if not isinstance(value, str):
        return None
    try:
        return str.encode(value, "utf-8").decode("utf-8")
    except Exception:
        return None


def _normalize_label(value: Any) -> str:
    """Convert a detector label to a bounded, content-free counter key."""

    plain = _plain_text(value)
    if plain is None:
        return "UNKNOWN"
    normalized = re.sub(r"[^A-Z0-9_]+", "_", plain.strip().upper())
    normalized = normalized.strip("_")[:64]
    if normalized in _SAFE_COUNTER_LABELS:
        return normalized
    canonical = normalize_label(normalized)
    if canonical != "OTHER" and canonical in _SAFE_COUNTER_LABELS:
        return canonical
    return "UNKNOWN"


def _normalize_counts(value: Any) -> dict[str, int]:
    """Normalize count mappings without retaining detector payloads."""

    if not isinstance(value, Mapping):
        return {}
    counts: Counter[str] = Counter()
    try:
        entries = tuple(value.items())
    except Exception:
        return {}
    for entry in entries:
        try:
            label, count = entry
        except Exception:
            continue
        if type(count) is not int:
            continue
        if count <= 0:
            continue
        counts[_normalize_label(label)] += count
    return dict(sorted(counts.items()))


def _entity_label(entity: Any) -> Any:
    if isinstance(entity, Mapping):
        return entity.get("canonical_label") or entity.get("label")
    return getattr(entity, "canonical_label", None) or getattr(entity, "label", None)


def _counts_from_entities(entities: Any) -> dict[str, int]:
    if isinstance(entities, (str, bytes, bytearray)) or not isinstance(
        entities, Sequence
    ):
        return {}
    counts: Counter[str] = Counter()
    try:
        entries = tuple(entities)
    except Exception:
        return {}
    for entity in entries:
        try:
            label = _entity_label(entity)
        except Exception:
            continue
        if label is not None:
            counts[_normalize_label(label)] += 1
    return dict(sorted(counts.items()))


def _coerce_redaction_result(value: Any, input_text: str) -> RedactionResult:
    """Keep only redacted text and aggregate labels from an injected result."""

    if isinstance(value, RedactionResult):
        return RedactionResult(
            redacted_text=value.redacted_text,
            entity_counts=value.entity_counts,
            input_characters=len(input_text),
        )

    output: Any = None
    counts: Mapping[str, int] | None = None
    entities: Any = None
    if isinstance(value, Mapping):
        output = value.get("redacted_text", value.get("deidentified_text"))
        candidate_counts = value.get("entity_counts", value.get("counts"))
        if isinstance(candidate_counts, Mapping):
            counts = candidate_counts
        entities = value.get("pii_entities", value.get("entities"))
    else:
        output = getattr(value, "redacted_text", None)
        if output is None:
            output = getattr(value, "deidentified_text", None)
        candidate_counts = getattr(value, "entity_counts", None)
        if isinstance(candidate_counts, Mapping):
            counts = candidate_counts
        entities = getattr(value, "pii_entities", None)
        if entities is None:
            entities = getattr(value, "entities", None)

    if isinstance(value, str):
        output = value
    if not isinstance(output, str):
        raise RedactionServiceError("redaction_failed")

    normalized_counts = _normalize_counts(counts)
    if not normalized_counts:
        normalized_counts = _counts_from_entities(entities)
    return RedactionResult(
        redacted_text=output,
        entity_counts=normalized_counts,
        input_characters=len(input_text),
    )


def _find_local_spans(text: str) -> list[_Span]:
    patterns = (
        ("EMAIL", _EMAIL_PATTERN, 10),
        ("SSN", _SSN_PATTERN, 20),
        ("PHONE", _PHONE_PATTERN, 30),
        ("IP_ADDRESS", _IP_PATTERN, 40),
        ("DATE", _DATE_PATTERN, 50),
    )
    candidates: list[_Span] = []
    try:
        sweep_entities = safety_sweep(text, [])
    except Exception:
        raise RedactionServiceError("redaction_failed") from None
    for entity in sweep_entities:
        try:
            start = entity.start
            end = entity.end
            label = _normalize_label(_entity_label(entity))
        except Exception:
            continue
        if type(start) is int and type(end) is int and 0 <= start < end <= len(text):
            candidates.append(_Span(start, end, label, 0))
    for label, pattern, priority in patterns:
        candidates.extend(
            _Span(match.start(), match.end(), label, priority)
            for match in pattern.finditer(text)
        )
    for pattern, label, priority in (
        (_NAME_CONTEXT_PATTERN, "NAME", 5),
        (_ID_CONTEXT_PATTERN, "ID", 15),
    ):
        candidates.extend(
            _Span(match.start("value"), match.end("value"), label, priority)
            for match in pattern.finditer(text)
        )

    selected: list[_Span] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.start, -(item.end - item.start), item.priority),
    ):
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


def _format_preserving_placeholder(source: str) -> str:
    placeholder: list[str] = []
    for character in source:
        category = unicodedata.category(character)
        if character.isdigit() or category.startswith("N"):
            placeholder.append("0")
        elif character.isupper():
            placeholder.append("X")
        elif character.islower():
            placeholder.append("x")
        elif character.isalpha() or category.startswith(("L", "M")):
            placeholder.append("x")
        else:
            placeholder.append(character)
    return "".join(placeholder)


def _shift_date(source: str, seed: int) -> str | None:
    formats = (
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%Y-%m-%d",
    )
    digest = hashlib.sha256(f"openmed-local-date-shift:{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 730
    offset = bucket - 365 if bucket < 365 else bucket - 364
    for date_format in formats:
        try:
            parsed = datetime.strptime(source, date_format)
        except ValueError:
            continue
        return (parsed + timedelta(days=offset)).strftime(date_format)
    return None


def _replacement_for(label: str, source: str, method: str, seed: int) -> str:
    if method == "remove":
        return ""
    if method == "replace":
        return "[REDACTED]"
    if method == "hash":
        material = f"{seed}:{label}:{source}".encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()[:12]
        return f"[{label}_{digest}]"
    if method == "format_preserve":
        return _format_preserving_placeholder(source)
    if method == "shift_dates" and label in {"DATE", "DATE_OF_BIRTH"}:
        return _shift_date(source, seed) or f"[{label}]"
    return f"[{label}]"


def local_redactor(
    text: str,
    *,
    policy: str = DEFAULT_POLICY,
    method: str = DEFAULT_METHOD,
    seed: int = 0,
) -> RedactionResult:
    """Redact common direct identifiers with deterministic local patterns.

    This baseline intentionally favors stable masking over probabilistic name
    inference.  A locally available model can be supplied through
    :func:`local_model_redactor` when broader detection is required.
    """

    normalized_text = _plain_text(text)
    if normalized_text is None:
        raise RedactionServiceError("invalid_input")
    _validate_policy(policy)
    normalized_method = _validate_method(method)
    seed = _validate_seed(seed)
    text = normalized_text
    spans = _find_local_spans(text)
    pieces: list[str] = []
    counts: Counter[str] = Counter()
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        source = text[span.start : span.end]
        label = _normalize_label(span.label)
        pieces.append(_replacement_for(label, source, normalized_method, seed))
        counts[label] += 1
        cursor = span.end
    pieces.append(text[cursor:])
    return RedactionResult(
        redacted_text="".join(pieces),
        entity_counts=dict(sorted(counts.items())),
        input_characters=len(text),
    )


def local_model_redactor(model_path: str | Path) -> Callable[..., Any]:
    """Build a redactor backed by a model that is already present on disk.

    The path is validated before the app starts.  The service never downloads
    a model; callers remain responsible for provisioning local model files.
    """

    try:
        resolved_path = Path(model_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ValueError("The local model path is unavailable") from None
    if not resolved_path.exists():
        raise ValueError("The local model path is unavailable")

    def _redact(
        text: str,
        *,
        policy: str = DEFAULT_POLICY,
        method: str = DEFAULT_METHOD,
        seed: int = 0,
    ) -> Any:
        from openmed import deidentify

        return deidentify(
            text,
            method=cast(RedactionMethod, method),
            model_name=str(resolved_path),
            confidence_threshold=0.7,
            policy=policy,
            use_safety_sweep=True,
            consistent=True,
            seed=seed,
        )

    return _redact


def _validate_policy(value: str) -> str:
    plain = _plain_text(value)
    normalized = plain.strip().lower() if plain is not None else ""
    if not _SAFE_POLICY.fullmatch(normalized):
        raise RedactionServiceError("invalid_policy")
    try:
        return canonical_policy_name(normalized)
    except (TypeError, ValueError):
        raise RedactionServiceError("invalid_policy") from None


def _validate_method(value: str) -> RedactionMethod:
    plain = _plain_text(value)
    normalized = plain.strip().lower() if plain is not None else ""
    if normalized not in SUPPORTED_METHODS:
        raise RedactionServiceError("invalid_method")
    return cast(RedactionMethod, normalized)


def _validate_seed(value: int) -> int:
    if type(value) is not int or not MIN_SEED <= value <= MAX_SEED:
        raise RedactionServiceError("invalid_seed")
    return value


def _resolve_path(value: str) -> Path:
    plain = _plain_text(value)
    if plain is None or not plain.strip() or "\x00" in plain:
        raise RedactionServiceError("invalid_path")
    try:
        return Path(plain).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise RedactionServiceError("invalid_path") from None


def _read_utf8_bounded(path: Path, max_characters: int) -> str:
    """Read at most one UTF-8 character budget without loading an unbounded file."""

    max_bytes = max_characters * 4
    try:
        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError:
        raise RedactionServiceError("input_unavailable") from None
    if len(payload) > max_bytes:
        raise RedactionServiceError("input_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        raise RedactionServiceError("input_unavailable") from None
    if len(text) > max_characters:
        raise RedactionServiceError("input_too_large")
    return text


def _write_artifact(path: Path, text: str) -> str:
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".openmed-redaction-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        return digest
    except (OSError, UnicodeError):
        raise RedactionServiceError("output_unavailable") from None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


class RedactionService:
    """State-light orchestration for text and explicit file redaction."""

    def __init__(
        self,
        redactor: Callable[..., Any] = local_redactor,
        *,
        max_input_characters: int = MAX_TEXT_CHARS,
    ) -> None:
        if type(max_input_characters) is not int or max_input_characters < 1:
            raise ValueError("max_input_characters must be positive")
        if not callable(redactor):
            raise ValueError("redactor must be callable")
        self._redactor = redactor
        self._max_input_characters = max_input_characters
        self._lock = threading.Lock()
        self._state = _ReviewState()
        self._operation_generation = 0

    def _set_processing(self, policy: str, method: str, kind: str) -> int:
        with self._lock:
            self._operation_generation += 1
            self._state = _ReviewState(
                status="processing",
                policy=policy,
                method=method,
                kind=kind,
                artifact_status="pending",
            )
            return self._operation_generation

    def _set_failed(
        self,
        operation_generation: int,
        policy: str,
        method: str,
        kind: str,
    ) -> None:
        with self._lock:
            if operation_generation != self._operation_generation:
                return
            self._state = _ReviewState(
                status="failed",
                policy=policy,
                method=method,
                kind=kind,
                artifact_status="not_written",
            )

    def _set_completed(
        self,
        operation_generation: int,
        policy: str,
        method: str,
        kind: str,
        result: RedactionResult,
        artifact_status: str,
    ) -> None:
        with self._lock:
            if operation_generation != self._operation_generation:
                return
            self._state = _ReviewState(
                status="completed",
                policy=policy,
                method=method,
                kind=kind,
                artifact_status=artifact_status,
                entity_counts=result.entity_counts,
                input_characters=result.input_characters,
                output_characters=len(result.redacted_text),
            )

    def _validate_text(self, text: str) -> str:
        normalized = _plain_text(text)
        if normalized is None or not normalized.strip() or "\x00" in normalized:
            raise RedactionServiceError("invalid_input")
        if len(normalized) > self._max_input_characters:
            raise RedactionServiceError("input_too_large")
        return normalized

    def _run_redactor(
        self,
        text: str,
        *,
        policy: str,
        method: str,
        seed: int,
    ) -> RedactionResult:
        try:
            signature = inspect.signature(self._redactor)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            kwargs = {
                name: value
                for name, value in {
                    "policy": policy,
                    "method": method,
                    "seed": seed,
                }.items()
                if accepts_kwargs or name in signature.parameters
            }
            candidate = self._redactor(text, **kwargs)
        except RedactionServiceError:
            raise
        except Exception:
            raise RedactionServiceError("redaction_failed") from None
        try:
            return _coerce_redaction_result(candidate, text)
        except RedactionServiceError:
            raise
        except Exception:
            raise RedactionServiceError("redaction_failed") from None

    def process_text(
        self,
        text: str,
        *,
        policy: str = DEFAULT_POLICY,
        method: str = DEFAULT_METHOD,
        seed: int = 0,
        output_path: str | None = None,
    ) -> _ArtifactResult:
        normalized_policy = _validate_policy(policy)
        normalized_method = _validate_method(method)
        seed = _validate_seed(seed)
        operation_generation = self._set_processing(
            normalized_policy,
            normalized_method,
            "text",
        )
        try:
            text = self._validate_text(text)
            result = self._run_redactor(
                text,
                policy=normalized_policy,
                method=normalized_method,
                seed=seed,
            )
            if output_path is None:
                artifact_status = "returned"
                digest = None
            else:
                destination = _resolve_path(output_path)
                digest = _write_artifact(destination, result.redacted_text)
                artifact_status = "written"
        except RedactionServiceError:
            self._set_failed(
                operation_generation,
                normalized_policy,
                normalized_method,
                "text",
            )
            raise
        except Exception:
            self._set_failed(
                operation_generation,
                normalized_policy,
                normalized_method,
                "text",
            )
            raise RedactionServiceError("redaction_failed") from None
        self._set_completed(
            operation_generation,
            normalized_policy,
            normalized_method,
            "text",
            result,
            artifact_status,
        )
        return _ArtifactResult(
            result,
            artifact_status,
            digest,
            normalized_policy,
            normalized_method,
            "text",
        )

    def process_file(
        self,
        input_path: str,
        output_path: str,
        *,
        policy: str = DEFAULT_POLICY,
        method: str = DEFAULT_METHOD,
        seed: int = 0,
    ) -> _ArtifactResult:
        normalized_policy = _validate_policy(policy)
        normalized_method = _validate_method(method)
        seed = _validate_seed(seed)
        operation_generation = self._set_processing(
            normalized_policy,
            normalized_method,
            "file",
        )
        try:
            source = _resolve_path(input_path)
            destination = _resolve_path(output_path)
            if source == destination:
                raise RedactionServiceError("input_output_same")
            if not source.is_file():
                raise RedactionServiceError("input_unavailable")
            text = _read_utf8_bounded(source, self._max_input_characters)
            text = self._validate_text(text)
            result = self._run_redactor(
                text,
                policy=normalized_policy,
                method=normalized_method,
                seed=seed,
            )
            digest = _write_artifact(destination, result.redacted_text)
        except RedactionServiceError:
            self._set_failed(
                operation_generation,
                normalized_policy,
                normalized_method,
                "file",
            )
            raise
        except Exception:
            self._set_failed(
                operation_generation,
                normalized_policy,
                normalized_method,
                "file",
            )
            raise RedactionServiceError("redaction_failed") from None
        self._set_completed(
            operation_generation,
            normalized_policy,
            normalized_method,
            "file",
            result,
            "written",
        )
        return _ArtifactResult(
            result,
            "written",
            digest,
            normalized_policy,
            normalized_method,
            "file",
        )

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate state safe for JSON and HTML rendering."""

        with self._lock:
            return self._state.to_dict()


_ERROR_DETAILS: dict[str, tuple[int, str, str]] = {
    "invalid_input": (400, "bad_request", "Input text is invalid"),
    "input_too_large": (
        413,
        "budget_exceeded",
        "Input exceeds the configured limit",
    ),
    "invalid_path": (400, "bad_request", "A file path is invalid"),
    "input_output_same": (400, "bad_request", "Input and output must differ"),
    "input_unavailable": (400, "bad_request", "Input artifact is unavailable"),
    "output_unavailable": (
        500,
        "internal_error",
        "Output artifact could not be written",
    ),
    "invalid_policy": (422, "validation_error", "Policy is invalid"),
    "invalid_method": (422, "validation_error", "Redaction method is invalid"),
    "invalid_seed": (422, "validation_error", "Seed is invalid"),
    "redaction_failed": (500, "internal_error", "Redaction failed"),
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Return an error envelope that never contains caller content."""

    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": None}},
    )


def _service_error_response(error: RedactionServiceError) -> JSONResponse:
    status_code, code, message = _ERROR_DETAILS.get(
        error.code,
        _ERROR_DETAILS["redaction_failed"],
    )
    return _error_response(status_code, code, message)


def _validation_response(exc: RequestValidationError) -> JSONResponse:
    details = []
    for error in exc.errors():
        location = error.get("loc", ("body",))
        if isinstance(location, (list, tuple)):
            parts = [part for part in location if part != "body"]
            if all(
                (isinstance(part, str) and part in _SAFE_REQUEST_FIELDS)
                or (type(part) is int and part >= 0)
                for part in parts
            ):
                safe_parts = [str(part) for part in parts]
                field_name = ".".join(safe_parts)
            else:
                field_name = "body"
        else:
            field_name = "body"
        details.append({"field": field_name or "body", "message": "invalid value"})
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": details,
            }
        },
    )


def _operation_payload(
    operation: _ArtifactResult,
    *,
    include_text: bool,
) -> dict[str, Any]:
    result = operation.result
    payload: dict[str, Any] = {
        "status": "completed",
        "summary": {
            "policy": operation.policy,
            "method": operation.method,
            "kind": operation.kind,
            "input_characters": result.input_characters,
            "output_characters": len(result.redacted_text),
            "counts": {
                "total_entities": result.total_entities,
                "by_label": dict(result.entity_counts),
            },
        },
        "artifact": {
            "status": operation.artifact_status,
            "kind": operation.kind,
            "sha256": operation.artifact_sha256,
        },
    }
    if include_text:
        payload["redacted_text"] = result.redacted_text
    return payload


def _render_review_page(state: Mapping[str, Any]) -> str:
    """Render the aggregate-only accessible review page."""

    counts = state.get("counts", {})
    by_label = counts.get("by_label", {}) if isinstance(counts, Mapping) else {}
    rows = "".join(
        f'<tr><th scope="row">{html.escape(str(label))}</th><td>{int(count)}</td></tr>'
        for label, count in sorted(by_label.items())
    )
    if not rows:
        rows = '<tr><th scope="row">None</th><td>0</td></tr>'
    status = html.escape(str(state.get("status", "idle")))
    policy = html.escape(str(state.get("policy", DEFAULT_POLICY)))
    method = html.escape(str(state.get("method", DEFAULT_METHOD)))
    kind = html.escape(str(state.get("kind", "none")))
    artifact_status = html.escape(
        str((state.get("artifact") or {}).get("status", "not_started"))
    )
    total = int(counts.get("total_entities", 0)) if isinstance(counts, Mapping) else 0
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenMed redaction review</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 48rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid currentColor; padding: .5rem; text-align: left; }}
    .status {{ display: grid; gap: .35rem; grid-template-columns: 10rem 1fr; }}
  </style>
</head>
<body>
  <main>
    <h1>OpenMed redaction review</h1>
    <p>This page exposes aggregate counts and artifact status only.</p>
    <section aria-labelledby="operation-heading">
      <h2 id="operation-heading">Latest operation</h2>
      <dl class="status">
        <dt>Status</dt><dd>{status}</dd>
        <dt>Policy</dt><dd>{policy}</dd>
        <dt>Method</dt><dd>{method}</dd>
        <dt>Input kind</dt><dd>{kind}</dd>
        <dt>Artifact status</dt><dd>{artifact_status}</dd>
        <dt>Total entities</dt><dd>{total}</dd>
      </dl>
    </section>
    <section aria-labelledby="counts-heading">
      <h2 id="counts-heading">Entity counts</h2>
      <table>
        <caption>Detected entity counts by label</caption>
        <thead><tr><th scope="col">Label</th><th scope="col">Count</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def create_app(
    *,
    redactor: Callable[..., Any] | None = None,
    local_model_path: str | Path | None = None,
    max_input_characters: int = MAX_TEXT_CHARS,
    max_request_body_bytes: int = MAX_REQUEST_BODY_BYTES,
) -> FastAPI:
    """Create the local redaction app without starting any network client.

    Args:
        redactor: Optional callable accepting text and optional ``policy``,
            ``method``, and ``seed`` keyword arguments.  It may return a
            :class:`RedactionResult`, a core de-identification result, a
            mapping with ``redacted_text``, or a string.
        local_model_path: Optional path to an already-provisioned local model.
            This is mutually exclusive with ``redactor``.
        max_input_characters: Maximum text size accepted by both endpoints.
        max_request_body_bytes: Maximum encoded request body size accepted
            before JSON parsing.
    """

    if redactor is not None and local_model_path is not None:
        raise ValueError("redactor and local_model_path are mutually exclusive")
    if type(max_request_body_bytes) is not int or max_request_body_bytes < 1:
        raise ValueError("max_request_body_bytes must be positive")
    selected_redactor = redactor
    if local_model_path is not None:
        selected_redactor = local_model_redactor(local_model_path)
    if selected_redactor is None:
        selected_redactor = local_redactor
    service = RedactionService(
        selected_redactor,
        max_input_characters=max_input_characters,
    )
    app = FastAPI(
        title="OpenMed self-hosted redaction service",
        description="Local-only text and explicit file redaction.",
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        _BoundedRequestBodyMiddleware,
        max_bytes=max_request_body_bytes,
    )
    app.add_middleware(
        ErrorEnvelopeTrustedHostMiddleware,
        allowed_hosts=DEFAULT_TRUSTED_HOSTS,
        www_redirect=False,
    )
    app.state.redaction_service = service

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _validation_response(exc)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
        return _error_response(500, "internal_error", "Internal server error")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/review", response_class=HTMLResponse)
    def review() -> HTMLResponse:
        return HTMLResponse(_render_review_page(service.snapshot()))

    @app.get("/status")
    def status() -> dict[str, Any]:
        return service.snapshot()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "openmed-redaction", "mode": "local"}

    @app.post("/redact/text")
    def redact_text(payload: TextRedactionRequest) -> Any:
        try:
            operation = service.process_text(
                payload.text,
                policy=payload.policy,
                method=payload.method,
                seed=payload.seed,
                output_path=payload.output_path,
            )
        except RedactionServiceError as exc:
            return _service_error_response(exc)
        return _operation_payload(
            operation,
            include_text=True,
        )

    @app.post("/redact/file")
    def redact_file(payload: FileRedactionRequest) -> Any:
        try:
            operation = service.process_file(
                payload.input_path,
                payload.output_path,
                policy=payload.policy,
                method=payload.method,
                seed=payload.seed,
            )
        except RedactionServiceError as exc:
            return _service_error_response(exc)
        return _operation_payload(
            operation,
            include_text=False,
        )

    @app.post("/redact")
    def redact(payload: RedactionRequest) -> Any:
        if payload.text is not None and payload.input_path is not None:
            return _error_response(
                400,
                "bad_request",
                "Provide text or input_path, not both",
            )
        if payload.text is not None:
            try:
                operation = service.process_text(
                    payload.text,
                    policy=payload.policy,
                    method=payload.method,
                    seed=payload.seed,
                    output_path=payload.output_path,
                )
            except RedactionServiceError as exc:
                return _service_error_response(exc)
            return _operation_payload(
                operation,
                include_text=True,
            )
        if payload.input_path is None or payload.output_path is None:
            return _error_response(
                400,
                "bad_request",
                "File redaction requires input_path and output_path",
            )
        try:
            operation = service.process_file(
                payload.input_path,
                payload.output_path,
                policy=payload.policy,
                method=payload.method,
                seed=payload.seed,
            )
        except RedactionServiceError as exc:
            return _service_error_response(exc)
        return _operation_payload(
            operation,
            include_text=False,
        )

    return app


app = create_app()


__all__ = [
    "DEFAULT_METHOD",
    "DEFAULT_POLICY",
    "FileRedactionRequest",
    "RedactionResult",
    "RedactionService",
    "RedactionServiceError",
    "RedactionRequest",
    "TextRedactionRequest",
    "app",
    "create_app",
    "local_model_redactor",
    "local_redactor",
]
