"""Restore privacy placeholders in local inference responses.

The inbound boundary accepts a request-scoped placeholder mapping that was
created locally.  It never contacts a transport and never writes the mapping
to logs, reports, or disk.  Text and JSON-like response content are restored
with the same validation rules so a malformed response cannot silently cross
the local privacy boundary.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional

DEFAULT_MAX_PLACEHOLDERS = 256
DEFAULT_MAX_MAPPING_BYTES = 1_048_576
DEFAULT_MAX_RESPONSE_BYTES = 4_194_304
DEFAULT_MAX_RESTORED_BYTES = 4_194_304
DEFAULT_MAX_PLACEHOLDER_OCCURRENCES = 1_024
DEFAULT_MAX_RESPONSE_NODES = 8_192
DEFAULT_MAX_RESPONSE_DEPTH = 32
DEFAULT_MAX_ACTIVE_REQUESTS = 128
DEFAULT_MAX_TOTAL_MAPPING_BYTES = 16 * 1_048_576

_DEFAULT_REQUEST_ID = "unscoped"
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_PLACEHOLDER_PATTERN = re.compile(r"<<OPENMED_PHI_[A-Z0-9_]+_[0-9A-F]{8}_[0-9]{6,}>>")
_PLACEHOLDER_CANDIDATE_PATTERN = re.compile(r"<<OPENMED_PHI_[^<>\s]*>>")
_PLACEHOLDER_FRAGMENT_PATTERN = re.compile(r"OPENMED[_-]PHI", re.IGNORECASE)


class InboundRestorationError(ValueError):
    """Base class for fail-closed inbound restoration errors."""

    error_code = "inbound_restoration_error"
    reason_code = "inbound_restoration_error"
    default_message = "Inbound placeholder restoration was rejected"

    def __init__(self, message: Optional[str] = None) -> None:
        # Callers should not put response content or mapping values in errors.
        # The module's own messages are deliberately constant and PHI-free.
        super().__init__(message or self.default_message)


class MalformedPlaceholderError(InboundRestorationError):
    """Raised when a response contains a token with an invalid shape."""

    error_code = "inbound_malformed_placeholder"
    reason_code = "malformed_placeholder"
    default_message = "Inbound response contained a malformed placeholder"


class UnknownPlaceholderError(InboundRestorationError):
    """Raised when a response token is absent from the request mapping."""

    error_code = "inbound_unknown_placeholder"
    reason_code = "unknown_placeholder"
    default_message = "Inbound response contained an unknown placeholder"


class DuplicatePlaceholderError(InboundRestorationError):
    """Raised when a mapped placeholder occurs more than once by policy."""

    error_code = "inbound_duplicate_placeholder"
    reason_code = "duplicate_placeholder"
    default_message = "Inbound response contained a duplicated placeholder"


class DuplicateMappingError(InboundRestorationError):
    """Raised when one request mapping defines a placeholder more than once."""

    error_code = "inbound_duplicate_mapping"
    reason_code = "duplicate_mapping_placeholder"
    default_message = "Inbound mapping contained a duplicated placeholder"


class RestorationLimitError(InboundRestorationError):
    """Raised when a request or response exceeds its configured budget."""

    error_code = "inbound_restoration_limit"
    reason_code = "restoration_limit_exceeded"
    default_message = "Inbound placeholder restoration exceeded its limit"


class RequestScopeError(InboundRestorationError):
    """Raised when request-scoped restoration state is invalid or missing."""

    error_code = "inbound_request_scope_error"
    reason_code = "invalid_request_scope"
    default_message = "Inbound restoration state is not valid for this request"


class UnknownRequestStateError(RequestScopeError):
    """Raised when a request has no active restoration state."""

    reason_code = "missing_request_state"
    default_message = "No inbound restoration state exists for this request"


class UnsupportedResponseError(InboundRestorationError):
    """Raised when response content is not JSON-like or text."""

    error_code = "inbound_unsupported_response"
    reason_code = "unsupported_response_content"
    default_message = "Inbound response content is unsupported"


class DuplicateResponseKeyError(InboundRestorationError):
    """Raised when restoration would collapse two structured response keys."""

    error_code = "inbound_duplicate_response_key"
    reason_code = "duplicate_response_key"
    default_message = "Inbound restoration would create a duplicate response key"


@dataclass(frozen=True)
class InboundRestorationPolicy:
    """Limits and validation choices for one inbound response.

    The defaults are intentionally strict.  A repeated placeholder can be
    allowed for a response format that explicitly permits repeated references,
    but unknown and malformed tokens remain rejected unless the caller opts
    into a less restrictive policy.
    """

    max_placeholders: int = DEFAULT_MAX_PLACEHOLDERS
    max_mapping_bytes: int = DEFAULT_MAX_MAPPING_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_restored_bytes: int = DEFAULT_MAX_RESTORED_BYTES
    max_placeholder_occurrences: int = DEFAULT_MAX_PLACEHOLDER_OCCURRENCES
    max_response_nodes: int = DEFAULT_MAX_RESPONSE_NODES
    max_response_depth: int = DEFAULT_MAX_RESPONSE_DEPTH
    reject_unknown_placeholders: bool = True
    reject_duplicate_placeholders: bool = True
    reject_malformed_placeholders: bool = True
    # Short aliases make policy construction convenient without changing the
    # explicit, descriptive fields above.
    reject_unknown: Optional[bool] = field(default=None, repr=False, compare=False)
    reject_duplicates: Optional[bool] = field(default=None, repr=False, compare=False)
    reject_malformed: Optional[bool] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "max_placeholders",
            "max_mapping_bytes",
            "max_response_bytes",
            "max_restored_bytes",
            "max_placeholder_occurrences",
            "max_response_nodes",
            "max_response_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        aliases = (
            ("reject_unknown", "reject_unknown_placeholders"),
            ("reject_duplicates", "reject_duplicate_placeholders"),
            ("reject_malformed", "reject_malformed_placeholders"),
        )
        for alias, canonical in aliases:
            alias_value = getattr(self, alias)
            if alias_value is not None:
                if not isinstance(alias_value, bool):
                    raise ValueError(f"{alias} must be a boolean")
                object.__setattr__(self, canonical, alias_value)
            elif not isinstance(getattr(self, canonical), bool):
                raise ValueError(f"{canonical} must be a boolean")

    def to_safe_dict(self) -> dict[str, Any]:
        """Return policy metadata that contains no mapping or response values."""

        return {
            "max_placeholders": self.max_placeholders,
            "max_mapping_bytes": self.max_mapping_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_restored_bytes": self.max_restored_bytes,
            "max_placeholder_occurrences": self.max_placeholder_occurrences,
            "max_response_nodes": self.max_response_nodes,
            "max_response_depth": self.max_response_depth,
            "reject_unknown_placeholders": self.reject_unknown_placeholders,
            "reject_duplicate_placeholders": self.reject_duplicate_placeholders,
            "reject_malformed_placeholders": self.reject_malformed_placeholders,
        }


@dataclass(frozen=True, init=False, repr=False)
class InboundRestorationState:
    """Immutable, request-scoped placeholder state held only in memory."""

    request_id: str = field(init=False)
    mapping: Mapping[str, str] = field(init=False, repr=False)
    mapping_bytes: int = field(init=False)
    placeholder_count: int = field(init=False)

    def __init__(
        self,
        request_id: str,
        mapping: Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        policy: Optional[InboundRestorationPolicy] = None,
    ) -> None:
        active_policy = policy or InboundRestorationPolicy()
        normalized_request_id = _validate_request_id(request_id)
        clean_mapping, mapping_bytes = _normalize_mapping(mapping, active_policy)
        object.__setattr__(self, "request_id", normalized_request_id)
        object.__setattr__(self, "mapping", MappingProxyType(clean_mapping))
        object.__setattr__(self, "mapping_bytes", mapping_bytes)
        object.__setattr__(self, "placeholder_count", len(clean_mapping))

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        request_id: str,
        policy: Optional[InboundRestorationPolicy] = None,
    ) -> "InboundRestorationState":
        """Build validated state for one request without persisting it."""

        return cls(request_id, mapping, policy=policy)

    @property
    def placeholder_map(self) -> Mapping[str, str]:
        """Return the read-only mapping used by the local restoration boundary."""

        return self.mapping

    def to_safe_dict(self) -> dict[str, Any]:
        """Return state metadata without exposing placeholder values."""

        return {
            "request_id": self.request_id,
            "placeholder_count": self.placeholder_count,
            "mapping_bytes": self.mapping_bytes,
        }

    def __repr__(self) -> str:
        return (
            "InboundRestorationState("
            f"request_id={self.request_id!r}, "
            f"placeholder_count={self.placeholder_count}, "
            f"mapping_bytes={self.mapping_bytes})"
        )


@dataclass
class _RestoreTracker:
    state: InboundRestorationState
    policy: InboundRestorationPolicy
    seen_placeholders: set[str] = field(default_factory=set)
    placeholder_occurrences: int = 0
    response_nodes: int = 0
    response_bytes: int = 0
    restored_bytes: int = 0

    def visit_node(self) -> None:
        self.response_nodes += 1
        if self.response_nodes > self.policy.max_response_nodes:
            raise RestorationLimitError()

    def record_response_bytes(self, value: str) -> None:
        size = _utf8_size(value)
        self.response_bytes += size
        if self.response_bytes > self.policy.max_response_bytes:
            raise RestorationLimitError()

    def record_restored_bytes(self, value: str) -> None:
        size = _utf8_size(value)
        self.restored_bytes += size
        if self.restored_bytes > self.policy.max_restored_bytes:
            raise RestorationLimitError()

    def record_placeholder(self, token: str) -> None:
        self.placeholder_occurrences += 1
        if self.placeholder_occurrences > self.policy.max_placeholder_occurrences:
            raise RestorationLimitError()
        if token in self.seen_placeholders:
            if self.policy.reject_duplicate_placeholders:
                raise DuplicatePlaceholderError()
            return
        self.seen_placeholders.add(token)


class InboundPlaceholderRestorer:
    """Restore text or structured response content for one request.

    A restorer owns one mapping for its lifetime.  Use it as a context manager
    or call :meth:`close` after the response has been handled to release the
    mapping as soon as the request ends.
    """

    def __init__(
        self,
        state: InboundRestorationState | Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        policy: Optional[InboundRestorationPolicy] = None,
        request_id: Optional[str] = None,
    ) -> None:
        active_policy = policy or InboundRestorationPolicy()
        if isinstance(state, InboundRestorationState):
            if request_id is not None and request_id != state.request_id:
                raise RequestScopeError()
            _validate_state_budget(state, active_policy)
            active_state = state
        else:
            active_state = InboundRestorationState(
                request_id or _DEFAULT_REQUEST_ID,
                state,
                policy=active_policy,
            )
        self._state: Optional[InboundRestorationState] = active_state
        self.policy = active_policy
        self._request_id = active_state.request_id

    @property
    def request_id(self) -> str:
        """Return the safe request identifier associated with this restorer."""

        return self._request_id

    @property
    def state(self) -> InboundRestorationState:
        """Return active state, or fail after the request has been closed."""

        state = self._state
        if state is None:
            raise RequestScopeError()
        return state

    def restore(self, response: Any) -> Any:
        """Restore a string or JSON-like response tree deterministically."""

        tracker = _RestoreTracker(self.state, self.policy)
        return _restore_value(response, depth=0, tracker=tracker)

    def restore_text(self, response_text: str) -> str:
        """Restore a text response after validating every placeholder."""

        tracker = _RestoreTracker(self.state, self.policy)
        tracker.visit_node()
        return _restore_text(response_text, tracker)

    def restore_structured(self, response: Any) -> Any:
        """Restore JSON-like mappings, lists, tuples, and nested text values."""

        return self.restore(response)

    def close(self) -> None:
        """Release the mapping so it cannot be reused after request completion."""

        self._state = None

    def __enter__(self) -> "InboundPlaceholderRestorer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = self._state
        count = state.placeholder_count if state is not None else 0
        return (
            "InboundPlaceholderRestorer("
            f"request_id={self._request_id!r}, placeholder_count={count})"
        )


class InboundRestorationStore:
    """Thread-safe bounded store for active request restoration state.

    The store never evicts an active request to make room for another one.
    Saturation fails closed, preserving request-to-mapping isolation.
    """

    def __init__(
        self,
        *,
        policy: Optional[InboundRestorationPolicy] = None,
        max_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
        max_total_mapping_bytes: int = DEFAULT_MAX_TOTAL_MAPPING_BYTES,
    ) -> None:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int):
            raise ValueError("max_requests must be a positive integer")
        if max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        if (
            isinstance(max_total_mapping_bytes, bool)
            or not isinstance(max_total_mapping_bytes, int)
            or max_total_mapping_bytes < 1
        ):
            raise ValueError("max_total_mapping_bytes must be a positive integer")
        self.policy = policy or InboundRestorationPolicy()
        self.max_requests = max_requests
        self.max_total_mapping_bytes = max_total_mapping_bytes
        self._states: dict[str, InboundRestorationState] = {}
        self._total_mapping_bytes = 0
        self._lock = threading.RLock()

    def put(
        self,
        request_id: str,
        mapping: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> InboundRestorationState:
        """Validate and retain one request mapping until it is popped."""

        state = InboundRestorationState(request_id, mapping, policy=self.policy)
        with self._lock:
            if state.request_id in self._states:
                raise RequestScopeError()
            if len(self._states) >= self.max_requests:
                raise RestorationLimitError()
            if (
                self._total_mapping_bytes + state.mapping_bytes
                > self.max_total_mapping_bytes
            ):
                raise RestorationLimitError()
            self._states[state.request_id] = state
            self._total_mapping_bytes += state.mapping_bytes
        return state

    def get(self, request_id: str) -> InboundRestorationState:
        """Return active state without removing it."""

        normalized_request_id = _validate_request_id(request_id)
        with self._lock:
            state = self._states.get(normalized_request_id)
            if state is None:
                raise UnknownRequestStateError()
            return state

    def pop(self, request_id: str) -> Optional[InboundRestorationState]:
        """Remove and return one request state, if present."""

        normalized_request_id = _validate_request_id(request_id)
        with self._lock:
            state = self._states.pop(normalized_request_id, None)
            if state is not None:
                self._total_mapping_bytes -= state.mapping_bytes
            return state

    def remove(self, request_id: str) -> bool:
        """Remove one request state and report whether it was active."""

        return self.pop(request_id) is not None

    def restore(
        self,
        request_id: str,
        response: Any,
        *,
        policy: Optional[InboundRestorationPolicy] = None,
        consume: bool = True,
    ) -> Any:
        """Restore a response and consume state by default."""

        state = self.pop(request_id) if consume else self.get(request_id)
        if state is None:
            raise UnknownRequestStateError()
        return InboundPlaceholderRestorer(state, policy=policy).restore(response)

    def clear(self) -> None:
        """Release every active request mapping."""

        with self._lock:
            self._states.clear()
            self._total_mapping_bytes = 0

    @property
    def active_requests(self) -> int:
        """Return the number of active request mappings."""

        with self._lock:
            return len(self._states)

    @property
    def total_mapping_bytes(self) -> int:
        """Return the safe aggregate size of active mapping strings."""

        with self._lock:
            return self._total_mapping_bytes

    def __len__(self) -> int:
        return self.active_requests


def restore_inbound_response(
    response: Any,
    state: InboundRestorationState | Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    policy: Optional[InboundRestorationPolicy] = None,
    request_id: Optional[str] = None,
) -> Any:
    """Restore text or structured response content in one request scope."""

    return InboundPlaceholderRestorer(
        state,
        policy=policy,
        request_id=request_id,
    ).restore(response)


def restore_text(
    response_text: str,
    state: InboundRestorationState | Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    policy: Optional[InboundRestorationPolicy] = None,
    request_id: Optional[str] = None,
) -> str:
    """Restore a text response in a single request-scoped operation."""

    return InboundPlaceholderRestorer(
        state,
        policy=policy,
        request_id=request_id,
    ).restore_text(response_text)


def restore_structured_response(
    response: Any,
    state: InboundRestorationState | Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    policy: Optional[InboundRestorationPolicy] = None,
    request_id: Optional[str] = None,
) -> Any:
    """Restore every string value in a JSON-like response tree."""

    return restore_inbound_response(
        response,
        state,
        policy=policy,
        request_id=request_id,
    )


def _restore_value(value: Any, *, depth: int, tracker: _RestoreTracker) -> Any:
    if depth > tracker.policy.max_response_depth:
        raise RestorationLimitError()
    tracker.visit_node()

    if isinstance(value, str):
        return _restore_text(value, tracker)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        restored: dict[Any, Any] = {}
        try:
            items = value.items()
            for key, item in items:
                if isinstance(key, str):
                    restored_key = _restore_text(key, tracker)
                elif key is None or isinstance(key, (bool, int, float)):
                    restored_key = key
                else:
                    raise UnsupportedResponseError()
                try:
                    duplicate_key = restored_key in restored
                except TypeError:
                    raise UnsupportedResponseError() from None
                if duplicate_key:
                    raise DuplicateResponseKeyError()
                restored[restored_key] = _restore_value(
                    item,
                    depth=depth + 1,
                    tracker=tracker,
                )
        except InboundRestorationError:
            raise
        except Exception:
            raise UnsupportedResponseError() from None
        return restored
    if isinstance(value, list):
        return [
            _restore_value(item, depth=depth + 1, tracker=tracker) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _restore_value(item, depth=depth + 1, tracker=tracker) for item in value
        )
    raise UnsupportedResponseError()


def _restore_text(text: str, tracker: _RestoreTracker) -> str:
    if not isinstance(text, str):
        raise UnsupportedResponseError()
    tracker.record_response_bytes(text)
    matches = []
    for match in _PLACEHOLDER_CANDIDATE_PATTERN.finditer(text):
        token = match.group(0)
        tracker.record_placeholder(token)
        is_valid = _PLACEHOLDER_PATTERN.fullmatch(token) is not None
        if not is_valid:
            if tracker.policy.reject_malformed_placeholders:
                raise MalformedPlaceholderError()
        elif token not in tracker.state.mapping:
            if tracker.policy.reject_unknown_placeholders:
                raise UnknownPlaceholderError()
        matches.append(match)

    if tracker.policy.reject_malformed_placeholders:
        residual = _PLACEHOLDER_CANDIDATE_PATTERN.sub("", text)
        if _PLACEHOLDER_FRAGMENT_PATTERN.search(residual):
            raise MalformedPlaceholderError()

    if not matches:
        tracker.record_restored_bytes(text)
        return text

    pieces: list[str] = []
    cursor = 0
    for match in matches:
        literal = text[cursor : match.start()]
        token = match.group(0)
        replacement = tracker.state.mapping.get(token, token)
        pieces.append(literal)
        pieces.append(replacement)
        tracker.record_restored_bytes(literal)
        tracker.record_restored_bytes(replacement)
        cursor = match.end()
    tail = text[cursor:]
    pieces.append(tail)
    tracker.record_restored_bytes(tail)
    return "".join(pieces)


def _normalize_mapping(
    mapping: Mapping[str, str] | Iterable[tuple[str, str]],
    policy: InboundRestorationPolicy,
) -> tuple[dict[str, str], int]:
    if isinstance(mapping, Mapping):
        try:
            items = iter(mapping.items())
        except Exception:
            raise InboundRestorationError("Inbound mapping could not be read") from None
    else:
        if isinstance(mapping, (str, bytes, bytearray)):
            raise InboundRestorationError("Inbound mapping must contain pairs")
        try:
            items = iter(mapping)
        except Exception:
            raise InboundRestorationError(
                "Inbound mapping must contain pairs"
            ) from None

    clean: dict[str, str] = {}
    total_bytes = 0
    try:
        for raw_item in items:
            if isinstance(mapping, Mapping):
                key, value = raw_item
            else:
                key, value = _coerce_mapping_pair(raw_item)
            if not isinstance(key, str) or _PLACEHOLDER_PATTERN.fullmatch(key) is None:
                raise MalformedPlaceholderError(
                    "Inbound mapping contained a malformed placeholder"
                )
            if key in clean:
                raise DuplicateMappingError()
            if not isinstance(value, str):
                raise InboundRestorationError("Inbound mapping values must be strings")
            if len(clean) >= policy.max_placeholders:
                raise RestorationLimitError()
            total_bytes += _utf8_size(key) + _utf8_size(value)
            if total_bytes > policy.max_mapping_bytes:
                raise RestorationLimitError()
            clean[key] = value
    except InboundRestorationError:
        raise
    except Exception:
        raise InboundRestorationError(
            "Inbound mapping could not be validated"
        ) from None
    return clean, total_bytes


def _coerce_mapping_pair(raw_item: Any) -> tuple[Any, Any]:
    if isinstance(raw_item, (str, bytes, bytearray)):
        raise InboundRestorationError("Inbound mapping must contain pairs")
    try:
        pair = iter(raw_item)
        key = next(pair)
        value = next(pair)
    except StopIteration:
        # A pair must contain exactly two values.  The exception text is never
        # surfaced, so malformed user data cannot enter an error message.
        raise InboundRestorationError("Inbound mapping must contain pairs") from None
    except Exception:
        raise InboundRestorationError("Inbound mapping must contain pairs") from None
    try:
        next(pair)
    except StopIteration:
        return key, value
    except Exception:
        raise InboundRestorationError("Inbound mapping must contain pairs") from None
    raise InboundRestorationError("Inbound mapping must contain pairs")


def _validate_request_id(request_id: str) -> str:
    if (
        not isinstance(request_id, str)
        or _REQUEST_ID_PATTERN.fullmatch(request_id) is None
    ):
        raise RequestScopeError("Inbound request id is invalid")
    return request_id


def _validate_state_budget(
    state: InboundRestorationState,
    policy: InboundRestorationPolicy,
) -> None:
    if state.placeholder_count > policy.max_placeholders:
        raise RestorationLimitError()
    if state.mapping_bytes > policy.max_mapping_bytes:
        raise RestorationLimitError()


def _utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise InboundRestorationError("Inbound content is not valid UTF-8") from None


# Compatibility names for callers that use the shorter boundary vocabulary.
InboundPlaceholderPolicy = InboundRestorationPolicy
RestorationState = InboundRestorationState
InboundRestorer = InboundPlaceholderRestorer
RequestScopedRestorationStore = InboundRestorationStore
PlaceholderRestorationError = InboundRestorationError
restore_response = restore_inbound_response


__all__ = [
    "DEFAULT_MAX_ACTIVE_REQUESTS",
    "DEFAULT_MAX_MAPPING_BYTES",
    "DEFAULT_MAX_PLACEHOLDER_OCCURRENCES",
    "DEFAULT_MAX_PLACEHOLDERS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_MAX_RESPONSE_DEPTH",
    "DEFAULT_MAX_RESPONSE_NODES",
    "DEFAULT_MAX_RESTORED_BYTES",
    "DEFAULT_MAX_TOTAL_MAPPING_BYTES",
    "DuplicateMappingError",
    "DuplicatePlaceholderError",
    "DuplicateResponseKeyError",
    "InboundPlaceholderPolicy",
    "InboundPlaceholderRestorer",
    "InboundRestorationError",
    "InboundRestorationPolicy",
    "InboundRestorationState",
    "InboundRestorationStore",
    "InboundRestorer",
    "MalformedPlaceholderError",
    "PlaceholderRestorationError",
    "RequestScopeError",
    "RequestScopedRestorationStore",
    "RestorationLimitError",
    "RestorationState",
    "UnknownPlaceholderError",
    "UnknownRequestStateError",
    "UnsupportedResponseError",
    "restore_inbound_response",
    "restore_response",
    "restore_structured_response",
    "restore_text",
]
