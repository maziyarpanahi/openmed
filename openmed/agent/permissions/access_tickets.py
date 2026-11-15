"""Purpose-bound, single-run data-access tickets for local agents.

Tickets carry only governance metadata and keyed record-selector digests. Raw
record identifiers, clinical values, credentials, and tool arguments must not
be placed in a ticket. Authorization is deterministic, local, and fail closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeVar, cast

from openmed.agent.correlation import RunId
from openmed.agent.identifiers import GovernanceIdError, PurposeId, ToolId

if TYPE_CHECKING:
    from .access_events import AccessEventSink

ACCESS_TICKET_DENIAL_SCHEMA_VERSION: Final = "openmed.agent.access_denial.v1"
ACCESS_TICKET_SELECTOR_DIGEST_ALGORITHM: Final = "hmac-sha256"

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_DATA_CLASS_RE = re.compile(rf"data:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_ACTION_RE = re.compile(rf"action:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_SELECTOR_KIND_RE = re.compile(rf"selector:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_SELECTOR_DIGEST_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_DENIAL_FIELDS = frozenset(
    {
        "dispatch",
        "expires_at",
        "purpose",
        "record_selectors",
        "request",
        "run_id",
        "ticket",
        "tool_action",
        "projection",
    }
)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class AccessTicketDenialEvidence:
    """Value-free evidence describing why access was denied."""

    reason_code: str
    field_name: str
    schema_version: str = ACCESS_TICKET_DENIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.reason_code) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason_code) is None
        ):
            raise ValueError("invalid_reason_code")
        if self.field_name not in _DENIAL_FIELDS:
            raise ValueError("invalid_field_name")
        if self.schema_version != ACCESS_TICKET_DENIAL_SCHEMA_VERSION:
            raise ValueError("invalid_schema_version")

    def to_dict(self) -> dict[str, str]:
        """Return deterministic evidence containing no request values."""

        return {
            "field_name": self.field_name,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize the evidence as canonical JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class AccessTicketError(ValueError):
    """Base class for value-free access-ticket failures."""

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        self.evidence = AccessTicketDenialEvidence(code, field_name)
        super().__init__(f"{field_name}: {code}")


class AccessTicketValidationError(AccessTicketError):
    """Raised when a ticket, request, selector, or callback is malformed."""


class AccessTicketDeniedError(AccessTicketError):
    """Raised when a valid request is not authorized by a ticket."""


class AccessTicketRequiredError(AccessTicketDeniedError):
    """Raised when data access is attempted without a ticket."""


class AccessTicketExpiredError(AccessTicketDeniedError):
    """Raised when a ticket has reached its exclusive expiry time."""


class AccessTicketRunMismatchError(AccessTicketDeniedError):
    """Raised when a ticket is presented by a different agent run."""


class AccessTicketPurposeMismatchError(AccessTicketDeniedError):
    """Raised when the request does not retain the ticket's purpose."""


class AccessTicketProjectionError(AccessTicketDeniedError):
    """Raised when a requested data class is outside the ticket."""


class AccessTicketSelectorError(AccessTicketDeniedError):
    """Raised when a requested record selector is outside the ticket."""


class AccessTicketToolActionError(AccessTicketDeniedError):
    """Raised when a requested tool action is outside the ticket."""


@dataclass(frozen=True, slots=True, repr=False)
class RecordSelector:
    """An exact record selector represented by a keyed, opaque digest."""

    kind: str
    digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.kind, "record_selectors", _SELECTOR_KIND_RE)
        if (
            type(self.digest) is not str
            or _SELECTOR_DIGEST_RE.fullmatch(self.digest) is None
        ):
            raise AccessTicketValidationError(
                "invalid_selector_digest", "record_selectors"
            )

    @classmethod
    def from_value(
        cls,
        *,
        kind: str,
        value: str | bytes,
        key: bytes,
    ) -> "RecordSelector":
        """Create a selector without retaining its raw local record value."""

        _validate_identifier(kind, "record_selectors", _SELECTOR_KIND_RE)
        if type(key) is not bytes or len(key) < 32:
            raise AccessTicketValidationError(
                "invalid_selector_key", "record_selectors"
            )
        if type(value) is str:
            try:
                encoded = value.encode("utf-8")
            except UnicodeError:
                raise AccessTicketValidationError(
                    "invalid_selector_value", "record_selectors"
                ) from None
        elif type(value) is bytes:
            encoded = value
        else:
            raise AccessTicketValidationError(
                "invalid_selector_value", "record_selectors"
            )
        digest = hmac.new(key, encoded, hashlib.sha256).hexdigest()
        return cls(kind=kind, digest=f"hmac-sha256:{digest}")

    def __repr__(self) -> str:
        """Return a representation that does not expose selector metadata."""

        return "RecordSelector(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ToolAction:
    """One exact tool and action pair permitted by an access ticket."""

    tool: str
    action: str

    def __post_init__(self) -> None:
        try:
            ToolId.parse(self.tool)
        except GovernanceIdError as exc:
            raise AccessTicketValidationError(
                "invalid_governance_identifier", "tool_action"
            ) from exc
        _validate_identifier(self.action, "tool_action", _ACTION_RE)

    def __repr__(self) -> str:
        """Return a value-free representation."""

        return "ToolAction(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class AccessTicket:
    """Immutable data authority bound to exactly one local agent run."""

    run_id: RunId
    purpose: str
    permitted_data_classes: tuple[str, ...]
    record_selectors: tuple[RecordSelector, ...]
    permitted_tool_actions: tuple[ToolAction, ...]
    expires_at: int

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_purpose(self.purpose)
        data_classes = _validate_string_tuple(
            self.permitted_data_classes,
            field_name="projection",
            pattern=_DATA_CLASS_RE,
        )
        selectors = _validate_typed_tuple(
            self.record_selectors,
            expected_type=RecordSelector,
            field_name="record_selectors",
        )
        tool_actions = _validate_typed_tuple(
            self.permitted_tool_actions,
            expected_type=ToolAction,
            field_name="tool_action",
        )
        object.__setattr__(self, "permitted_data_classes", data_classes)
        object.__setattr__(self, "record_selectors", selectors)
        object.__setattr__(self, "permitted_tool_actions", tool_actions)
        _validate_timestamp(self.expires_at, "expires_at")

    def __repr__(self) -> str:
        """Return a representation that does not expose ticket contents."""

        return "AccessTicket(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class AccessTicketRequest:
    """One proposed purpose-bound data access within an agent run."""

    run_id: RunId
    purpose: str
    projection: tuple[str, ...]
    record_selectors: tuple[RecordSelector, ...]
    tool_action: ToolAction

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_purpose(self.purpose)
        projection = _validate_string_tuple(
            self.projection,
            field_name="projection",
            pattern=_DATA_CLASS_RE,
        )
        selectors = _validate_typed_tuple(
            self.record_selectors,
            expected_type=RecordSelector,
            field_name="record_selectors",
        )
        if type(self.tool_action) is not ToolAction:
            raise AccessTicketValidationError("invalid_tool_action", "tool_action")
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "record_selectors", selectors)

    def __repr__(self) -> str:
        """Return a representation that does not expose request contents."""

        return "AccessTicketRequest(<redacted>)"


class AccessTicketVerifier:
    """Verify run binding, purpose, expiry, and exact ticket scope locally."""

    def __init__(
        self,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
        sink: AccessEventSink | None = None,
    ) -> None:
        from .access_events import MemoryAccessEventSink

        if not callable(clock):
            raise AccessTicketValidationError("invalid_clock", "ticket")
        self.clock = clock
        self.sink = MemoryAccessEventSink() if sink is None else sink

    def verify(
        self,
        ticket: AccessTicket | None,
        request: AccessTicketRequest,
        *,
        now: int | None = None,
    ) -> AccessTicket:
        """Record exactly one decision before returning authority or denying it.

        Args:
            ticket: Presented authority, or None to deny a missing ticket.
            request: Validated request whose narrow metadata is recorded.
            now: Optional verification epoch seconds for deterministic replay.

        Returns:
            The ticket when it authorizes the complete request.

        Raises:
            AccessTicketError: When authorization fails.
            AccessEventError: When the injected sink fails; dispatch is blocked.
        """
        current_time = None
        try:
            if type(request) is not AccessTicketRequest:
                raise AccessTicketValidationError("invalid_request", "request")
            if now is None:
                try:
                    candidate_time = self.clock()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    candidate_time = None
                if candidate_time is None:
                    raise AccessTicketValidationError("clock_unavailable", "expires_at")
            else:
                candidate_time = now
            current_time = _validate_timestamp(candidate_time, "expires_at")
            if ticket is None:
                raise AccessTicketRequiredError("missing_ticket", "ticket")
            if type(ticket) is not AccessTicket:
                raise AccessTicketValidationError("invalid_ticket", "ticket")
            if current_time >= ticket.expires_at:
                raise AccessTicketExpiredError("expired", "expires_at")
            if request.run_id != ticket.run_id:
                raise AccessTicketRunMismatchError("run_mismatch", "run_id")
            if request.purpose != ticket.purpose:
                raise AccessTicketPurposeMismatchError("purpose_mismatch", "purpose")
            if not set(request.projection).issubset(ticket.permitted_data_classes):
                raise AccessTicketProjectionError("projection_denied", "projection")
            if not set(request.record_selectors).issubset(ticket.record_selectors):
                raise AccessTicketSelectorError(
                    "record_selector_denied", "record_selectors"
                )
            if request.tool_action not in ticket.permitted_tool_actions:
                raise AccessTicketToolActionError("tool_action_denied", "tool_action")
        except AccessTicketError as error:
            # Leave the handler before calling the sink: a failed sink must not
            # retain the denial exception or any provider exception as context.
            denial = error
        else:
            self._emit(request, "allow", None, current_time)
            return ticket
        self._emit(request, "deny", denial.code, current_time)
        raise denial

    def _emit(
        self,
        request: AccessTicketRequest,
        outcome: str,
        reason_code: str | None,
        timestamp: int | None,
    ) -> None:
        from .access_events import AccessEvent, AccessEventError

        event = AccessEvent.for_request(
            request,
            outcome=outcome,
            reason_code=reason_code,
            timestamp=timestamp,
        )
        try:
            self.sink.emit(event)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass
        else:
            return
        raise AccessEventError("sink_unavailable")


def dispatch_with_access_ticket(
    ticket: AccessTicket | None,
    request: AccessTicketRequest,
    verifier: AccessTicketVerifier,
    dispatch: Callable[[], _T],
    *,
    now: int | None = None,
) -> _T:
    """Verify complete data authority before invoking a local callback."""

    if type(verifier) is not AccessTicketVerifier:
        raise AccessTicketValidationError("invalid_verifier", "ticket")
    if not callable(dispatch):
        raise AccessTicketValidationError("invalid_dispatch", "dispatch")
    verifier.verify(ticket, request, now=now)
    return dispatch()


def _validate_run_id(value: object) -> RunId:
    if type(value) is not RunId:
        raise AccessTicketValidationError("invalid_run_id", "run_id")
    return value


def _validate_purpose(value: object) -> str:
    try:
        PurposeId.parse(value)
    except GovernanceIdError as exc:
        raise AccessTicketValidationError(
            "invalid_governance_identifier", "purpose"
        ) from exc
    return cast(str, value)


def _validate_identifier(
    value: object,
    field_name: str,
    pattern: re.Pattern[str],
) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise AccessTicketValidationError("invalid_governance_identifier", field_name)
    return value


def _validate_string_tuple(
    values: object,
    *,
    field_name: str,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise AccessTicketValidationError("invalid_scope", field_name)
    for value in values:
        _validate_identifier(value, field_name, pattern)
    canonical = tuple(sorted(values))
    if len(set(canonical)) != len(canonical):
        raise AccessTicketValidationError("duplicate_scope", field_name)
    return canonical


def _validate_typed_tuple(
    values: object,
    *,
    expected_type: type[RecordSelector] | type[ToolAction],
    field_name: str,
) -> tuple[RecordSelector, ...] | tuple[ToolAction, ...]:
    if type(values) is not tuple or not values:
        raise AccessTicketValidationError("invalid_scope", field_name)
    if not all(type(value) is expected_type for value in values):
        raise AccessTicketValidationError("invalid_scope", field_name)
    if len(set(values)) != len(values):
        raise AccessTicketValidationError("duplicate_scope", field_name)
    typed_values = cast(tuple[RecordSelector | ToolAction, ...], values)
    return cast(
        tuple[RecordSelector, ...] | tuple[ToolAction, ...],
        tuple(sorted(typed_values, key=_typed_scope_sort_key)),
    )


def _typed_scope_sort_key(value: RecordSelector | ToolAction) -> tuple[str, str]:
    if type(value) is RecordSelector:
        return (value.kind, value.digest)
    action = cast(ToolAction, value)
    return (action.tool, action.action)


def _validate_timestamp(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise AccessTicketValidationError("invalid_timestamp", field_name)
    return value


__all__ = [
    "ACCESS_TICKET_DENIAL_SCHEMA_VERSION",
    "ACCESS_TICKET_SELECTOR_DIGEST_ALGORITHM",
    "AccessTicket",
    "AccessTicketDenialEvidence",
    "AccessTicketDeniedError",
    "AccessTicketError",
    "AccessTicketExpiredError",
    "AccessTicketProjectionError",
    "AccessTicketPurposeMismatchError",
    "AccessTicketRequest",
    "AccessTicketRequiredError",
    "AccessTicketRunMismatchError",
    "AccessTicketSelectorError",
    "AccessTicketToolActionError",
    "AccessTicketValidationError",
    "AccessTicketVerifier",
    "RecordSelector",
    "ToolAction",
    "dispatch_with_access_ticket",
]
