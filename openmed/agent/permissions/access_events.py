"""Value-free access decisions and a local, append-only digest-chain sink."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from openmed.agent.correlation import RunId

from .access_tickets import (
    _DATA_CLASS_RE,
    AccessTicketRequest,
    AccessTicketValidationError,
    ToolAction,
    _validate_purpose,
    _validate_string_tuple,
    _validate_timestamp,
)

ACCESS_EVENT_SCHEMA_VERSION = "openmed.agent.access_event.v1"
_GENESIS = "sha256:" + "0" * 64
_REASONS = frozenset(
    {
        "missing_ticket",
        "invalid_ticket",
        "invalid_request",
        "clock_unavailable",
        "invalid_timestamp",
        "expired",
        "run_mismatch",
        "purpose_mismatch",
        "projection_denied",
        "record_selector_denied",
        "tool_action_denied",
    }
)


class AccessEventError(ValueError):
    """Expose a controlled code without payloads, paths, or provider messages."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class AccessEvent:
    """One verification decision, containing no selectors or record values.

    Args:
        run_id: Opaque requesting run, or None for an invalid request.
        purpose: Developer-authored purpose, or None for an invalid request.
        data_classes: Requested projection, never the broader ticket scope.
        selector_count: Number of requested selectors, without their metadata.
        tool_action: Requested governance tool/action pair.
        outcome: Controlled allow or deny decision.
        reason_code: Controlled denial code, or None for allow.
        timestamp: Verification time in epoch seconds; None if unavailable.
    """

    run_id: RunId | None
    purpose: str | None
    data_classes: tuple[str, ...]
    selector_count: int
    tool_action: ToolAction | None
    outcome: str
    reason_code: str | None
    timestamp: int | None

    def __post_init__(self) -> None:
        if type(self.outcome) is not str or self.outcome not in {"allow", "deny"}:
            raise AccessEventError("invalid_event")
        if self.outcome == "allow":
            if self.reason_code is not None or self.timestamp is None:
                raise AccessEventError("invalid_event")
        elif type(self.reason_code) is not str or self.reason_code not in _REASONS:
            raise AccessEventError("invalid_event")
        if self.timestamp is not None:
            _validate_timestamp(self.timestamp, "expires_at")
        if type(self.selector_count) is not int or self.selector_count < 0:
            raise AccessEventError("invalid_event")
        if self.run_id is None:
            if (
                self.outcome != "deny"
                or self.reason_code != "invalid_request"
                or self.purpose is not None
                or self.data_classes != ()
                or self.selector_count != 0
                or self.tool_action is not None
            ):
                raise AccessEventError("invalid_event")
        else:
            if (
                type(self.run_id) is not RunId
                or type(self.tool_action) is not ToolAction
            ):
                raise AccessEventError("invalid_event")
            _validate_purpose(self.purpose)
            normalized = _validate_string_tuple(
                self.data_classes, field_name="projection", pattern=_DATA_CLASS_RE
            )
            object.__setattr__(self, "data_classes", normalized)

    @classmethod
    def for_request(
        cls,
        request: AccessTicketRequest | object,
        *,
        outcome: str,
        reason_code: str | None,
        timestamp: int | None,
    ) -> AccessEvent:
        """Copy only approved metadata from a validated request."""
        valid = type(request) is AccessTicketRequest
        metadata = cast(AccessTicketRequest, request)
        return cls(
            run_id=metadata.run_id if valid else None,
            purpose=metadata.purpose if valid else None,
            data_classes=metadata.projection if valid else (),
            selector_count=len(metadata.record_selectors) if valid else 0,
            tool_action=metadata.tool_action if valid else None,
            outcome=outcome,
            reason_code=reason_code,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export only the closed event schema, without selector digests."""
        return {
            "schema_version": ACCESS_EVENT_SCHEMA_VERSION,
            "run_id": None if self.run_id is None else self.run_id.value,
            "purpose": self.purpose,
            "data_classes": list(self.data_classes),
            "selector_count": self.selector_count,
            "tool": None if self.tool_action is None else self.tool_action.tool,
            "action": None if self.tool_action is None else self.tool_action.action,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        """Keep diagnostic representations value-free."""
        return "AccessEvent(<metadata>)"


class AccessEventSink(Protocol):
    """An injected sink must accept one decision or raise before dispatch."""

    def emit(self, event: AccessEvent) -> None:
        """Accept exactly one event; do not retry or silently drop failures."""
        ...


class MemoryAccessEventSink:
    """Process-local default sink; hosts own retention and access controls."""

    def __init__(self) -> None:
        self.events: list[AccessEvent] = []

    def emit(self, event: AccessEvent) -> None:
        """Retain one immutable access decision."""
        if type(event) is not AccessEvent:
            raise AccessEventError("invalid_event")
        self.events.append(event)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _event_from_dict(value: Any) -> AccessEvent:
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema_version",
            "run_id",
            "purpose",
            "data_classes",
            "selector_count",
            "tool",
            "action",
            "outcome",
            "reason_code",
            "timestamp",
        }
        or value["schema_version"] != ACCESS_EVENT_SCHEMA_VERSION
    ):
        raise AccessEventError("invalid_chain")
    if type(value["data_classes"]) is not list:
        raise AccessEventError("invalid_chain")
    return AccessEvent(
        run_id=None if value["run_id"] is None else RunId(value["run_id"]),
        purpose=value["purpose"],
        data_classes=tuple(value["data_classes"]),
        selector_count=value["selector_count"],
        tool_action=(
            None
            if value["tool"] is None and value["action"] is None
            else ToolAction(value["tool"], value["action"])
        ),
        outcome=value["outcome"],
        reason_code=value["reason_code"],
        timestamp=value["timestamp"],
    )


class LocalAccessEventSink:
    """Append fsynced JSON lines under a POSIX advisory file lock.

    Args:
        path: Host-owned local log path. New files use mode 0600. Symbolic links
            and non-regular files are refused. Parent directories must exist.

    All cooperating writers lock the same file. The host must prevent replacement
    or rotation while writers are active and protect directories and existing
    file permissions. The chain detects corruption, not a malicious owner who
    can rewrite every digest. Retain an independent head digest for that threat.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __repr__(self) -> str:
        """Omit the private log path from diagnostics."""
        return "LocalAccessEventSink(<local>)"

    def _operate(self, event: AccessEvent | None = None) -> tuple[AccessEvent, ...]:
        # Import locally: the durable sink requires POSIX, the verifier does not.
        try:
            import fcntl

            fd = os.open(
                self._path,
                os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            with os.fdopen(fd, "r+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise AccessEventError("sink_unavailable")
                events: list[AccessEvent] = []
                previous = _GENESIS
                for line in stream:
                    record = json.loads(line)
                    if type(record) is not dict or set(record) != {
                        "event",
                        "sequence",
                        "previous_digest",
                        "digest",
                    }:
                        raise AccessEventError("invalid_chain")
                    body = {
                        key: record[key]
                        for key in ("event", "sequence", "previous_digest")
                    }
                    decoded = _event_from_dict(record["event"])
                    if (
                        type(record["sequence"]) is not int
                        or record["sequence"] != len(events) + 1
                        or record["previous_digest"] != previous
                        or record["digest"] != _digest(body)
                        or line != _canonical(record) + b"\n"
                    ):
                        raise AccessEventError("invalid_chain")
                    previous = record["digest"]
                    events.append(decoded)
                if event is not None:
                    body = {
                        "event": event.to_dict(),
                        "sequence": len(events) + 1,
                        "previous_digest": previous,
                    }
                    stream.write(_canonical({**body, "digest": _digest(body)}) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    events.append(event)
                return tuple(events)
        except AccessEventError:
            raise
        except (ValueError, TypeError, KeyError, AccessTicketValidationError):
            failure = "invalid_chain"
        except (OSError, ImportError):
            failure = "sink_unavailable"
        # Raise outside the handler so private paths and parser payloads are not
        # retained in exception context.
        raise AccessEventError(failure)

    def emit(self, event: AccessEvent) -> None:
        """Verify the existing chain and append one durable decision."""
        if type(event) is not AccessEvent:
            raise AccessEventError("invalid_event")
        self._operate(event)

    def verify_chain(self, *, expected_head: str | None = None) -> str:
        """Return the verified head; optionally check an independently saved head."""
        events = self._operate()
        previous = _GENESIS
        for sequence, event in enumerate(events, 1):
            previous = _digest(
                {
                    "event": event.to_dict(),
                    "sequence": sequence,
                    "previous_digest": previous,
                }
            )
        if expected_head is not None and previous != expected_head:
            raise AccessEventError("head_mismatch")
        return previous

    def export_counts(self) -> dict[str, Any]:
        """Export verified aggregate counts, omitting runs and record selectors.

        Data-class counts count decisions containing the class; selector_count
        sums requested selectors, including denied requests. Neither is a count
        of records actually returned by a tool.
        """
        events = self._operate()
        return {
            "event_count": len(events),
            "selector_count": sum(event.selector_count for event in events),
            "outcomes": dict(
                sorted(Counter(event.outcome for event in events).items())
            ),
            "denial_reasons": dict(
                sorted(
                    Counter(
                        event.reason_code
                        for event in events
                        if event.reason_code is not None
                    ).items()
                )
            ),
            "purposes": dict(
                sorted(
                    Counter(
                        event.purpose for event in events if event.purpose is not None
                    ).items()
                )
            ),
            "data_classes": dict(
                sorted(
                    Counter(
                        value for event in events for value in event.data_classes
                    ).items()
                )
            ),
            "tool_actions": [
                {"tool": tool, "action": action, "count": count}
                for (tool, action), count in sorted(
                    Counter(
                        (event.tool_action.tool, event.tool_action.action)
                        for event in events
                        if event.tool_action is not None
                    ).items()
                )
            ],
        }
