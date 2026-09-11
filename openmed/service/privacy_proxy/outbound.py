"""Fail-closed privacy filtering for outbound message requests.

This module owns the boundary immediately before a caller's inference
transport.  It accepts JSON message-style request bodies, applies an injected
local text redactor to supported text content, and keeps the resulting
replacement map in request-scoped memory for a later response-restoration
stage.

The module deliberately has no HTTP client and no model-loading default.  A
caller must provide the local redactor explicitly, which keeps filtering
deterministic and makes network access an explicit concern of the caller.
Unsupported media types, request shapes, message content parts, and redactor
outputs are rejected before a transformed body is returned.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONTENT_TYPE = "application/json"
SUPPORTED_CONTENT_TYPES = frozenset({DEFAULT_CONTENT_TYPE})
DEFAULT_MAX_STATES = 1024

_MISSING = object()
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_TEXT_CONTENT_TYPES = frozenset({"text", "input_text", "output_text"})

TextRedactor = Callable[[str], Any]
RequestBody = Mapping[str, Any] | str | bytes


class OutboundPrivacyError(ValueError):
    """Base class for fail-closed outbound privacy errors."""

    error_code = "outbound_privacy_error"
    reason_code = "outbound_privacy_error"

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        if reason_code is not None:
            self.reason_code = reason_code


class RedactorRequiredError(OutboundPrivacyError):
    """Raised when no local redactor was configured."""

    error_code = "outbound_redactor_required"
    reason_code = "redactor_required"


class UnsupportedContentTypeError(OutboundPrivacyError):
    """Raised when the request media type is not supported safely."""

    error_code = "unsupported_content_type"
    reason_code = "unsupported_content_type"


class UnsupportedRequestBodyError(OutboundPrivacyError):
    """Raised when a request body cannot be handled as JSON messages."""

    error_code = "unsupported_request_body"
    reason_code = "unsupported_request_body"


class RedactionError(OutboundPrivacyError):
    """Raised when local redaction fails or returns an unsafe result."""

    error_code = "outbound_redaction_failed"
    reason_code = "redaction_failed"


class RedactionOutputError(RedactionError):
    """Raised when a redactor leaves a mapped value in outbound content."""

    error_code = "unsafe_redaction_output"
    reason_code = "redaction_incomplete"


class ReplacementStateError(OutboundPrivacyError):
    """Raised when request-scoped replacement state cannot be maintained."""

    error_code = "replacement_state_error"
    reason_code = "replacement_state_error"


class RequestStateNotFoundError(ReplacementStateError):
    """Raised when a later stage asks for an unknown request state."""

    error_code = "replacement_state_not_found"
    reason_code = "replacement_state_not_found"


class ReplacementStateLimitError(ReplacementStateError):
    """Raised when the bounded in-memory state store is full."""

    error_code = "replacement_state_limit"
    reason_code = "replacement_state_limit"


class ReplacementMap(Mapping[str, str]):
    """Read-only replacement mapping with a PHI-safe representation.

    The mapping remains usable by a later local restoration stage, but its
    string and representation forms expose only the number of entries.  This
    prevents accidental logging of original sensitive values.
    """

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self._data = dict(values or {})

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        entry = "entry" if len(self) == 1 else "entries"
        return f"{self.__class__.__name__}(<{len(self)} {entry}>)"

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class RedactionResult:
    """Normalized result contract for an injected local text redactor.

    ``replacements`` maps the redacted surface to its original value.  The
    original values are intentionally hidden from ``repr`` and ``str`` while
    remaining available through the mapping for local restoration.
    """

    text: str
    replacements: Mapping[str, str] = field(default_factory=dict)

    @property
    def redacted_text(self) -> str:
        """Return the redacted text under the common de-identification name."""
        return self.text

    @property
    def mapping(self) -> Mapping[str, str]:
        """Return the replacement mapping under the common gateway name."""
        return self.replacements

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(text_length={len(self.text)}, "
            f"replacement_count={len(self.replacements)})"
        )

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class RequestReplacementState:
    """Request-scoped replacement state retained for local restoration."""

    request_id: str
    replacements: ReplacementMap
    message_count: int = 0
    redacted_field_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.replacements, ReplacementMap):
            object.__setattr__(self, "replacements", ReplacementMap(self.replacements))

    @property
    def mapping(self) -> ReplacementMap:
        """Return the read-only mapping consumed by response restoration."""
        return self.replacements

    @property
    def replacement_map(self) -> ReplacementMap:
        """Compatibility alias for callers that use the explicit map name."""
        return self.replacements

    def to_metadata(self) -> dict[str, Any]:
        """Return PHI-free state metadata for logs or reports."""
        return {
            "request_id": self.request_id,
            "replacement_count": len(self.replacements),
            "message_count": self.message_count,
            "redacted_field_count": self.redacted_field_count,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(request_id={self.request_id!r}, "
            f"replacement_count={len(self.replacements)}, "
            f"message_count={self.message_count}, "
            f"redacted_field_count={self.redacted_field_count})"
        )

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class PreparedOutboundRequest:
    """A transformed request body and its local restoration state."""

    request_id: str
    body: Any
    state: RequestReplacementState
    content_type: str = DEFAULT_CONTENT_TYPE

    @property
    def request_body(self) -> Any:
        """Return the transformed body under an explicit request name."""
        return self.body

    @property
    def replacement_state(self) -> RequestReplacementState:
        """Return request-scoped state for a later local restoration stage."""
        return self.state

    @property
    def replacements(self) -> ReplacementMap:
        """Return the read-only replacement mapping for this request."""
        return self.state.replacements

    def to_metadata(self) -> dict[str, Any]:
        """Return PHI-free request metadata without serializing the body."""
        return {
            "request_id": self.request_id,
            "content_type": self.content_type,
            **self.state.to_metadata(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(request_id={self.request_id!r}, "
            f"content_type={self.content_type!r}, "
            f"state={self.state!r})"
        )

    __str__ = __repr__


class RequestStateStore:
    """Thread-safe bounded in-memory store for request replacement state."""

    def __init__(self, *, max_entries: int = DEFAULT_MAX_STATES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise ValueError("max_entries must be a positive integer")
        if max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._states: OrderedDict[str, RequestReplacementState] = OrderedDict()
        self._lock = threading.RLock()

    def save(self, state: RequestReplacementState) -> None:
        """Save state, refusing new entries when the bounded store is full."""
        with self._lock:
            if (
                state.request_id not in self._states
                and len(self._states) >= self.max_entries
            ):
                raise ReplacementStateLimitError(
                    "Outbound replacement state capacity has been reached"
                )
            self._states[state.request_id] = state

    def get(self, request_id: str) -> RequestReplacementState:
        """Return state for ``request_id`` without exposing mutable storage."""
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                raise RequestStateNotFoundError(
                    "No outbound replacement state exists for this request"
                )
            return state

    def pop(self, request_id: str) -> RequestReplacementState:
        """Remove and return state for ``request_id``."""
        with self._lock:
            try:
                return self._states.pop(request_id)
            except KeyError:
                raise RequestStateNotFoundError(
                    "No outbound replacement state exists for this request"
                ) from None

    def discard(self, request_id: str) -> None:
        """Remove state if present, without raising for an already-cleaned request."""
        with self._lock:
            self._states.pop(request_id, None)

    def contains(self, request_id: str) -> bool:
        """Return whether state is currently retained for ``request_id``."""
        with self._lock:
            return request_id in self._states

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(entries={len(self)}, "
            f"max_entries={self.max_entries})"
        )


class OutboundRequestPrivacyFilter:
    """Prepare JSON message requests for safe outbound transport dispatch.

    The filter never calls a transport.  Callers invoke :meth:`transform`,
    pass the returned ``body`` to their configured transport, and retain the
    returned request state until the local response-restoration stage
    completes.
    """

    def __init__(
        self,
        redactor: TextRedactor | Any | None = None,
        *,
        redact: TextRedactor | Any | None = None,
        state_store: RequestStateStore | None = None,
        max_states: int = DEFAULT_MAX_STATES,
    ) -> None:
        if redactor is not None and redact is not None:
            raise ValueError("configure either redactor or redact, not both")
        candidate = redactor if redactor is not None else redact
        self._redactor = _coerce_redactor(candidate)
        self.state_store = (
            state_store
            if state_store is not None
            else RequestStateStore(max_entries=max_states)
        )

    def transform(
        self,
        body: RequestBody,
        *,
        request_id: str | None = None,
        content_type: str = DEFAULT_CONTENT_TYPE,
    ) -> PreparedOutboundRequest:
        """Transform supported message content and retain its replacement state.

        Args:
            body: A JSON object, JSON string, or UTF-8 JSON bytes containing a
                top-level ``messages`` list.
            request_id: Optional caller-owned correlation identifier. A UUID is
                generated when omitted; generated identifiers contain no body
                content.
            content_type: HTTP media type. Only JSON is accepted, including
                parameters such as ``; charset=utf-8``.

        Raises:
            OutboundPrivacyError: If the media type, body shape, message
                content, or redactor output cannot be handled safely.
        """
        normalized_content_type = _normalize_content_type(content_type)
        active_request_id = _coerce_request_id(request_id)
        payload, encoding = _load_json_body(body)
        replacements: dict[str, str] = {}
        message_count, redacted_field_count = self._transform_messages(
            payload,
            replacements,
        )
        _assert_no_sensitive_values(payload, replacements)
        encoded_payload = _encode_json_payload(payload)
        transformed_body: Any
        if encoding == "mapping":
            transformed_body = payload
        elif encoding == "bytes":
            transformed_body = encoded_payload.encode("utf-8")
        else:
            transformed_body = encoded_payload

        state = RequestReplacementState(
            request_id=active_request_id,
            replacements=ReplacementMap(replacements),
            message_count=message_count,
            redacted_field_count=redacted_field_count,
        )
        self.state_store.save(state)
        return PreparedOutboundRequest(
            request_id=active_request_id,
            body=transformed_body,
            state=state,
            content_type=normalized_content_type,
        )

    def prepare(self, body: RequestBody, **kwargs: Any) -> PreparedOutboundRequest:
        """Alias for :meth:`transform` used by transport adapters."""
        return self.transform(body, **kwargs)

    def filter_request(
        self, body: RequestBody, **kwargs: Any
    ) -> PreparedOutboundRequest:
        """Alias for :meth:`transform` used by request middleware."""
        return self.transform(body, **kwargs)

    def transform_messages(
        self,
        messages: list[Mapping[str, Any]],
        **kwargs: Any,
    ) -> PreparedOutboundRequest:
        """Transform a message list using the standard request envelope."""
        return self.transform({"messages": messages}, **kwargs)

    def get_state(self, request_id: str) -> RequestReplacementState:
        """Return request-scoped state for a later local restoration stage."""
        return self.state_store.get(_coerce_request_id(request_id))

    def replacement_state(self, request_id: str) -> RequestReplacementState:
        """Alias for :meth:`get_state`."""
        return self.get_state(request_id)

    def discard_state(self, request_id: str) -> None:
        """Discard state after the local restoration stage is complete."""
        self.state_store.discard(_coerce_request_id(request_id))

    def consume_state(self, request_id: str) -> RequestReplacementState:
        """Remove and return state for a one-time restoration handoff."""
        return self.state_store.pop(_coerce_request_id(request_id))

    def __call__(self, body: RequestBody, **kwargs: Any) -> PreparedOutboundRequest:
        """Make the filter usable as a request-preparation callable."""
        return self.transform(body, **kwargs)

    def _transform_messages(
        self,
        payload: MutableMapping[str, Any],
        replacements: dict[str, str],
    ) -> tuple[int, int]:
        messages = payload.get("messages", _MISSING)
        if not isinstance(messages, list):
            raise UnsupportedRequestBodyError(
                "JSON request must contain a messages list"
            )

        redacted_field_count = 0
        for index, original_message in enumerate(messages):
            if not isinstance(original_message, Mapping):
                raise UnsupportedRequestBodyError("Every message must be a JSON object")
            if not isinstance(original_message, MutableMapping):
                message = dict(original_message)
                messages[index] = message
            else:
                message = original_message

            if "content" not in message or message["content"] is None:
                continue
            content = message["content"]
            if isinstance(content, str):
                transformed, changed = self._redact_text(content, replacements)
                message["content"] = transformed
                redacted_field_count += int(changed)
                continue
            if isinstance(content, list):
                redacted_field_count += self._transform_content_parts(
                    content,
                    replacements,
                )
                continue
            raise UnsupportedRequestBodyError(
                "Message content must be text or a list of text parts"
            )

        return len(messages), redacted_field_count

    def _transform_content_parts(
        self,
        content: list[Any],
        replacements: dict[str, str],
    ) -> int:
        redacted_field_count = 0
        for index, original_part in enumerate(content):
            if not isinstance(original_part, Mapping):
                raise UnsupportedRequestBodyError(
                    "Every message content part must be a JSON object"
                )
            if not isinstance(original_part, MutableMapping):
                part = dict(original_part)
                content[index] = part
            else:
                part = original_part
            part_type = part.get("type", _MISSING)
            text = part.get("text", _MISSING)
            if part_type not in _TEXT_CONTENT_TYPES or not isinstance(text, str):
                raise UnsupportedRequestBodyError(
                    "Only typed text message content parts are supported"
                )
            transformed, changed = self._redact_text(text, replacements)
            part["text"] = transformed
            redacted_field_count += int(changed)
        return redacted_field_count

    def _redact_text(
        self,
        text: str,
        replacements: dict[str, str],
    ) -> tuple[str, bool]:
        try:
            result = self._redactor(text)
        except Exception:
            raise RedactionError(
                "Local redaction failed before outbound dispatch"
            ) from None
        redacted_text, redaction_mapping = _coerce_redaction_result(result)
        for key, original in redaction_mapping.items():
            existing = replacements.get(key)
            if existing is not None and existing != original:
                raise RedactionError(
                    "A replacement token mapped to conflicting request values",
                    reason_code="replacement_collision",
                )
            replacements[key] = original
        if any(
            original and original in redacted_text
            for original in redaction_mapping.values()
        ):
            raise RedactionOutputError(
                "Local redaction left sensitive content in a message"
            )
        return redacted_text, redacted_text != text


def transform_request(
    body: RequestBody,
    redactor: TextRedactor | Any | None = None,
    *,
    redact: TextRedactor | Any | None = None,
    request_id: str | None = None,
    content_type: str = DEFAULT_CONTENT_TYPE,
    state_store: RequestStateStore | None = None,
    max_states: int = DEFAULT_MAX_STATES,
) -> PreparedOutboundRequest:
    """Transform one message request with a fresh or supplied state store."""
    privacy_filter = OutboundRequestPrivacyFilter(
        redactor,
        redact=redact,
        state_store=state_store,
        max_states=max_states,
    )
    return privacy_filter.transform(
        body,
        request_id=request_id,
        content_type=content_type,
    )


def filter_request(*args: Any, **kwargs: Any) -> PreparedOutboundRequest:
    """Function alias for :func:`transform_request`."""
    return transform_request(*args, **kwargs)


def _coerce_redactor(candidate: TextRedactor | Any | None) -> TextRedactor:
    if candidate is None:
        raise RedactorRequiredError(
            "A local redactor is required before outbound dispatch"
        )
    if callable(candidate):
        return candidate
    for attribute in ("redact", "redact_text", "deidentify"):
        method = getattr(candidate, attribute, None)
        if callable(method):
            return method
    raise RedactorRequiredError("The configured local redactor is not callable")


def _normalize_content_type(content_type: str) -> str:
    if not isinstance(content_type, str):
        raise UnsupportedContentTypeError("Outbound request content type must be JSON")
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized not in SUPPORTED_CONTENT_TYPES:
        raise UnsupportedContentTypeError(
            "Outbound request content type is not supported safely"
        )
    return normalized


def _coerce_request_id(request_id: str | None) -> str:
    if request_id is None:
        return uuid.uuid4().hex
    if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise OutboundPrivacyError(
            "request_id contains unsupported characters",
            reason_code="invalid_request_id",
        )
    return request_id


def _load_json_body(body: RequestBody) -> tuple[dict[str, Any], str]:
    encoding = "mapping"
    candidate: Any = body
    if isinstance(body, bytes):
        encoding = "bytes"
        try:
            candidate = body.decode("utf-8")
        except UnicodeDecodeError:
            raise UnsupportedRequestBodyError(
                "Outbound JSON body must be valid UTF-8"
            ) from None
    elif isinstance(body, str):
        encoding = "text"
    elif isinstance(body, Mapping):
        try:
            candidate = json.loads(json.dumps(dict(body), allow_nan=False))
        except Exception:
            raise UnsupportedRequestBodyError(
                "Outbound request body could not be copied safely"
            ) from None
    else:
        raise UnsupportedRequestBodyError(
            "Outbound request body must be a JSON object, string, or bytes"
        )

    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except (TypeError, ValueError):
            raise UnsupportedRequestBodyError(
                "Outbound request body is not valid JSON"
            ) from None
    if not isinstance(candidate, dict):
        raise UnsupportedRequestBodyError("Outbound JSON request must be an object")
    try:
        json.dumps(candidate, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise UnsupportedRequestBodyError(
            "Outbound request body contains unsupported JSON values"
        ) from None
    return candidate, encoding


def _encode_json_payload(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise UnsupportedRequestBodyError(
            "Transformed outbound request is not valid JSON"
        ) from None


def _coerce_redaction_result(result: Any) -> tuple[str, dict[str, str]]:
    try:
        if isinstance(result, RedactionResult):
            redacted_text = result.text
            mapping = result.replacements
        elif isinstance(result, str):
            redacted_text = result
            mapping = {}
        elif isinstance(result, tuple) and len(result) == 2:
            redacted_text, mapping = result
        elif isinstance(result, Mapping):
            redacted_text = _first_present(
                result,
                "redacted_text",
                "deidentified_text",
                "text",
            )
            mapping = _first_present(result, "replacements", "mapping", default=None)
        else:
            redacted_text = _first_present(
                result,
                "redacted_text",
                "deidentified_text",
                "text",
            )
            mapping = _first_present(result, "replacements", "mapping", default=None)
    except Exception:
        raise RedactionError("Local redactor returned an invalid result") from None

    if not isinstance(redacted_text, str):
        raise RedactionError("Local redactor must return redacted text")
    return redacted_text, _coerce_replacement_mapping(mapping)


def _first_present(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        candidate = (
            value.get(name, _MISSING)
            if isinstance(value, Mapping)
            else getattr(value, name, _MISSING)
        )
        if candidate is not _MISSING:
            return candidate
    if default is not _MISSING:
        return default
    raise KeyError("redacted text field missing")


def _coerce_replacement_mapping(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RedactionError("Local redactor replacements must be a mapping")
    try:
        items = list(value.items())
    except Exception:
        raise RedactionError("Local redactor replacements could not be read") from None
    normalized: dict[str, str] = {}
    for key, original in items:
        if not isinstance(key, str) or not key:
            raise RedactionError(
                "Local redactor replacement keys must be non-empty strings"
            )
        if not isinstance(original, str):
            raise RedactionError("Local redactor replacement values must be strings")
        normalized[key] = original
    return normalized


def _assert_no_sensitive_values(
    payload: Any,
    replacements: Mapping[str, str],
) -> None:
    for original in replacements.values():
        if original and _contains_string(payload, original):
            raise RedactionOutputError(
                "Local redaction left sensitive content in the outbound body"
            )


def _contains_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(
            (isinstance(key, str) and needle in key) or _contains_string(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_string(item, needle) for item in value)
    return False


# Public aliases keep the boundary discoverable for middleware integrations
# that use either the long feature name or the shorter proxy terminology.
OutboundPrivacyFilter = OutboundRequestPrivacyFilter
OutboundRequestFilter = OutboundRequestPrivacyFilter
FilteredRequest = PreparedOutboundRequest
OutboundRequest = PreparedOutboundRequest
OutboundRequestState = RequestReplacementState
InMemoryReplacementStore = RequestStateStore
PrivacyProxyError = OutboundPrivacyError


__all__ = [
    "DEFAULT_CONTENT_TYPE",
    "DEFAULT_MAX_STATES",
    "FilteredRequest",
    "InMemoryReplacementStore",
    "OutboundPrivacyError",
    "OutboundPrivacyFilter",
    "OutboundRequest",
    "OutboundRequestFilter",
    "OutboundRequestPrivacyFilter",
    "OutboundRequestState",
    "PreparedOutboundRequest",
    "PrivacyProxyError",
    "RedactionError",
    "RedactionOutputError",
    "RedactionResult",
    "RedactorRequiredError",
    "ReplacementMap",
    "ReplacementStateError",
    "ReplacementStateLimitError",
    "RequestReplacementState",
    "RequestStateNotFoundError",
    "RequestStateStore",
    "SUPPORTED_CONTENT_TYPES",
    "UnsupportedContentTypeError",
    "UnsupportedRequestBodyError",
    "TextRedactor",
    "filter_request",
    "transform_request",
]
