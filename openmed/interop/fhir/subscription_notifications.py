"""Bounded, value-free FHIR Subscription notification parsing.

The caller supplies the negotiated FHIR release. This module neither infers
authority nor reads resources, persists checkpoints or dispatches workflows.
Resource content and reference strings remain confined to the parse call.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .versions import FHIRVersion

__all__ = [
    "SubscriptionContentMode",
    "SubscriptionEvent",
    "SubscriptionFindingCode",
    "SubscriptionNotification",
    "SubscriptionNotificationError",
    "SubscriptionNotificationFinding",
    "SubscriptionNotificationType",
    "SubscriptionNotificationVersion",
    "parse_subscription_notification",
]

MAX_NOTIFICATION_BYTES = 1_048_576
MAX_NOTIFICATION_NODES = 32_768
MAX_NOTIFICATION_DEPTH = 32
MAX_NOTIFICATION_ENTRIES = 1_024
MAX_NOTIFICATION_EVENTS = 256
MAX_EVENT_REFERENCES = 64
MAX_EVENT_NUMBER = 2**63 - 1
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COUNTER = re.compile(r"(?:0|[1-9][0-9]{0,18})\Z")
_RESOURCE_TYPE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_FHIR_ID = re.compile(r"[A-Za-z0-9.-]{1,64}\Z")
_ERROR_CODES = frozenset(
    {
        "unsupported_version",
        "invalid_options",
        "invalid_json",
        "duplicate_json_key",
        "payload_too_large",
        "payload_too_deep",
        "payload_too_complex",
        "invalid_bundle",
        "invalid_status",
        "mixed_version",
        "unsupported_modifier",
        "invalid_counter",
        "invalid_reference",
        "invalid_event",
        "invalid_parameter",
        "duplicate_parameter",
        "unsupported_parameter",
        "ambiguous_reference",
        "invalid_metadata",
    }
)


class SubscriptionNotificationVersion(str, Enum):
    """Notification formats; R4B support does not extend the exchange converter."""

    R4 = "R4"
    R4B = "R4B"
    R5 = "R5"


class SubscriptionNotificationType(str, Enum):
    """The five published topic-based notification types."""

    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    EVENT = "event-notification"
    QUERY_STATUS = "query-status"
    QUERY_EVENT = "query-event"


class SubscriptionContentMode(str, Enum):
    """Observed content coverage of one event, without returning that content."""

    EMPTY = "empty"
    ID_ONLY = "id-only"
    FULL_RESOURCE = "full-resource"


class SubscriptionFindingCode(str, Enum):
    """Closed reconciliation reasons, independent of raw server messages."""

    DUPLICATE = "duplicate_event_number"
    OUT_OF_ORDER = "out_of_order_event_number"
    GAP = "missing_event_numbers"
    COUNTER_BEHIND = "counter_behind_event"
    COUNTER_MISSING = "event_counter_missing"
    HISTORY_UNKNOWN = "history_not_examined"
    COUNTER_REGRESSED = "counter_regressed"
    SERVER_ERROR = "subscription_error_reported"
    INACTIVE = "subscription_not_active"


class SubscriptionNotificationError(ValueError):
    """A closed refusal code and numeric position; never retain rejected values."""

    def __init__(self, code: str, position: int | None = None) -> None:
        if type(code) is not str or code not in _ERROR_CODES:
            code = "invalid_metadata"
        if position is not None and (type(position) is not int or position < 0):
            position = None
        self.code = code
        self.position = position
        super().__init__(code if position is None else f"{code}:{position}")


def _require_count(value: Any, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if type(value) is not int or not 0 <= value <= MAX_EVENT_NUMBER:
        raise SubscriptionNotificationError("invalid_metadata")


def _require_digest(value: Any, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise SubscriptionNotificationError("invalid_metadata")


@dataclass(frozen=True, slots=True)
class SubscriptionNotificationFinding:
    """A controlled reason, position, count and optional missing-number range."""

    code: SubscriptionFindingCode
    position: int | None = None
    count: int = 1
    first_missing: int | None = None
    last_missing: int | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not SubscriptionFindingCode:
            raise SubscriptionNotificationError("invalid_metadata")
        for value in (self.position, self.first_missing, self.last_missing):
            _require_count(value, optional=True)
        _require_count(self.count)
        if (self.first_missing is None) != (self.last_missing is None):
            raise SubscriptionNotificationError("invalid_metadata")
        if self.first_missing is not None and self.last_missing < self.first_missing:
            raise SubscriptionNotificationError("invalid_metadata")

    def to_dict(self) -> dict[str, Any]:
        """Return content-free reconciliation metadata."""
        return {
            "code": self.code.value,
            "position": self.position,
            "count": self.count,
            "first_missing": self.first_missing,
            "last_missing": self.last_missing,
        }


@dataclass(frozen=True, slots=True)
class SubscriptionEvent:
    """One event's number, digest-only references and observed content coverage."""

    position: int
    event_number: int
    focus_reference_digest: str | None
    additional_reference_digests: tuple[str, ...]
    content_mode: SubscriptionContentMode

    def __post_init__(self) -> None:
        _require_count(self.position)
        _require_count(self.event_number)
        if (
            self.event_number == 0
            or type(self.content_mode) is not SubscriptionContentMode
        ):
            raise SubscriptionNotificationError("invalid_metadata")
        _require_digest(self.focus_reference_digest, optional=True)
        if (
            type(self.additional_reference_digests) is not tuple
            or len(self.additional_reference_digests) > MAX_EVENT_REFERENCES
        ):
            raise SubscriptionNotificationError("invalid_metadata")
        for digest in self.additional_reference_digests:
            _require_digest(digest)
        if self.content_mode is SubscriptionContentMode.EMPTY:
            if (
                self.focus_reference_digest is not None
                or self.additional_reference_digests
            ):
                raise SubscriptionNotificationError("invalid_metadata")
        elif self.focus_reference_digest is None:
            raise SubscriptionNotificationError("invalid_metadata")

    @property
    def requires_authorized_read(self) -> bool:
        """Whether event content is absent or incomplete in the supplied Bundle."""
        return self.content_mode is not SubscriptionContentMode.FULL_RESOURCE

    def to_dict(self) -> dict[str, Any]:
        """Return only number, position, digests and closed classifications."""
        return {
            "position": self.position,
            "event_number": self.event_number,
            "focus_reference_digest": self.focus_reference_digest,
            "additional_reference_digests": list(self.additional_reference_digests),
            "content_mode": self.content_mode.value,
            "requires_authorized_read": self.requires_authorized_read,
        }


@dataclass(frozen=True, slots=True)
class SubscriptionNotification:
    """Immutable intake metadata, suitable for a separately governed checkpoint."""

    version: SubscriptionNotificationVersion
    notification_type: SubscriptionNotificationType
    bundle_digest: str
    subscription_reference_digest: str
    events_since_start: int | None
    events: tuple[SubscriptionEvent, ...]
    findings: tuple[SubscriptionNotificationFinding, ...]
    resource_count: int
    error_count: int

    def __post_init__(self) -> None:
        if (
            type(self.version) is not SubscriptionNotificationVersion
            or type(self.notification_type) is not SubscriptionNotificationType
        ):
            raise SubscriptionNotificationError("invalid_metadata")
        _require_digest(self.bundle_digest)
        _require_digest(self.subscription_reference_digest)
        _require_count(self.events_since_start, optional=True)
        _require_count(self.resource_count)
        _require_count(self.error_count)
        for items, cls in (
            (self.events, SubscriptionEvent),
            (self.findings, SubscriptionNotificationFinding),
        ):
            if type(items) is not tuple or any(type(item) is not cls for item in items):
                raise SubscriptionNotificationError("invalid_metadata")
        if len(self.events) > MAX_NOTIFICATION_EVENTS:
            raise SubscriptionNotificationError("invalid_metadata")
        if len(self.findings) > 3 * MAX_NOTIFICATION_EVENTS + 8:
            raise SubscriptionNotificationError("invalid_metadata")
        if (
            self.resource_count >= MAX_NOTIFICATION_ENTRIES
            or self.error_count > MAX_NOTIFICATION_EVENTS
        ):
            raise SubscriptionNotificationError("invalid_metadata")
        if any(event.position != index for index, event in enumerate(self.events)):
            raise SubscriptionNotificationError("invalid_metadata")

    @property
    def requires_reconciliation(self) -> bool:
        """Whether sequence or server-state evidence needs separate review."""
        return bool(self.findings)

    @property
    def workflow_events(self) -> tuple[SubscriptionEvent, ...]:
        """New event metadata only; returning it does not authorize a workflow."""
        if (
            self.notification_type is not SubscriptionNotificationType.EVENT
            or self.findings
        ):
            return ()
        return self.events

    def to_dict(self) -> dict[str, Any]:
        """Serialize the closed, value-free intake report."""
        return {
            "schema": "openmed.interop.fhir.subscription_notification.v1",
            "version": self.version.value,
            "notification_type": self.notification_type.value,
            "bundle_digest": self.bundle_digest,
            "subscription_reference_digest": self.subscription_reference_digest,
            "events_since_start": self.events_since_start,
            "events": [e.to_dict() for e in self.events],
            "findings": [f.to_dict() for f in self.findings],
            "resource_count": self.resource_count,
            "error_count": self.error_count,
            "requires_reconciliation": self.requires_reconciliation,
            "workflow_event_count": len(self.workflow_events),
        }

    def to_json(self) -> str:
        """Return deterministic JSON containing no source content or identifiers."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _digest(scope: str, text: str) -> str:
    return "sha256:" + hashlib.sha256((scope + "\0" + text).encode("utf-8")).hexdigest()


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SubscriptionNotificationError("duplicate_json_key")
        result[key] = value
    return result


def _load_bundle(payload: Any) -> tuple[dict[str, Any], str]:
    error = False
    if type(payload) in (str, bytes):
        if len(payload) > MAX_NOTIFICATION_BYTES:
            raise SubscriptionNotificationError("payload_too_large")
        try:
            text = payload.decode("utf-8") if type(payload) is bytes else payload
            if len(text.encode("utf-8")) > MAX_NOTIFICATION_BYTES:
                raise SubscriptionNotificationError("payload_too_large")
            payload = json.loads(text, object_pairs_hook=_json_pairs)
        except SubscriptionNotificationError:
            raise
        except (ValueError, UnicodeError, RecursionError):
            error = True
        if error:
            raise SubscriptionNotificationError("invalid_json")
    if type(payload) is not dict:
        raise SubscriptionNotificationError("invalid_bundle")
    pending = [(payload, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if depth > MAX_NOTIFICATION_DEPTH:
            raise SubscriptionNotificationError("payload_too_deep")
        if nodes > MAX_NOTIFICATION_NODES:
            raise SubscriptionNotificationError("payload_too_complex")
        if type(value) is dict:
            if len(value) > MAX_NOTIFICATION_NODES or any(
                type(k) is not str
                or len(k) > 128
                or any(0xD800 <= ord(char) <= 0xDFFF for char in k)
                for k in value
            ):
                raise SubscriptionNotificationError("invalid_json")
            if nodes + len(pending) + len(value) > MAX_NOTIFICATION_NODES:
                raise SubscriptionNotificationError("payload_too_complex")
            pending.extend((item, depth + 1) for item in value.values())
        elif type(value) is list:
            if nodes + len(pending) + len(value) > MAX_NOTIFICATION_NODES:
                raise SubscriptionNotificationError("payload_too_complex")
            pending.extend((item, depth + 1) for item in value)
        elif type(value) is str:
            if len(value) > MAX_NOTIFICATION_BYTES:
                raise SubscriptionNotificationError("payload_too_large")
            if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
                raise SubscriptionNotificationError("invalid_json")
        elif type(value) is float:
            if not math.isfinite(value):
                raise SubscriptionNotificationError("invalid_json")
        elif value is not None and type(value) not in (bool, int):
            raise SubscriptionNotificationError("invalid_json")
        elif type(value) is int and value.bit_length() > 256:
            raise SubscriptionNotificationError("invalid_json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_NOTIFICATION_BYTES:
        raise SubscriptionNotificationError("payload_too_large")
    # Snapshot the caller's builtin JSON containers before interpreting them.
    return json.loads(encoded), _digest("bundle", encoded.decode("utf-8"))


def _counter(value: Any, *, event: bool = False) -> int:
    if type(value) is not str or _COUNTER.fullmatch(value) is None:
        raise SubscriptionNotificationError("invalid_counter")
    number = int(value)
    if number > MAX_EVENT_NUMBER or (event and number == 0):
        raise SubscriptionNotificationError("invalid_counter")
    return number


def _reference(value: Any) -> str:
    if type(value) is not dict or "reference" not in value:
        raise SubscriptionNotificationError("invalid_reference")
    reference = value["reference"]
    if (
        type(reference) is not str
        or not reference
        or len(reference) > 2_048
        or any(char.isspace() or ord(char) < 32 for char in reference)
    ):
        raise SubscriptionNotificationError("invalid_reference")
    _reject_extensions(value)
    return reference


def _reject_extensions(value: dict[str, Any], position: int | None = None) -> None:
    """Refuse unhandled extensions even when their malformed value is falsey."""
    if any(
        key in {"extension", "modifierExtension", "implicitRules"}
        or key.startswith("_")
        for key in value
    ):
        raise SubscriptionNotificationError("unsupported_modifier", position)


def _array(value: Any, limit: int, code: str) -> list[Any]:
    if type(value) is not list or len(value) > limit:
        raise SubscriptionNotificationError(code)
    return value


def _parameters(items: Any, *, event: bool = False) -> dict[str, Any]:
    specs = (
        {
            "event-number": "valueString",
            "timestamp": "valueInstant",
            "focus": "valueReference",
            "additional-context": "valueReference",
        }
        if event
        else {
            "subscription": "valueReference",
            "status": "valueCode",
            "type": "valueCode",
            "topic": "valueCanonical",
            "events-since-subscription-start": "valueString",
            "notification-event": "part",
            "error": "valueCodeableConcept",
        }
    )
    repeatable = {"additional-context"} if event else {"notification-event", "error"}
    result: dict[str, Any] = {}
    for index, item in enumerate(
        _array(items, MAX_NOTIFICATION_EVENTS + 8, "invalid_parameter")
    ):
        if type(item) is not dict:
            raise SubscriptionNotificationError("invalid_parameter", index)
        _reject_extensions(item, index)
        name = item.get("name")
        if type(name) is not str or name not in specs:
            raise SubscriptionNotificationError("unsupported_parameter", index)
        key = specs[name]
        if key not in item or set(item) - {"name", key, "id", "extension"}:
            raise SubscriptionNotificationError("invalid_parameter", index)
        if name in repeatable:
            result.setdefault(name, []).append(item[key])
        elif name in result:
            raise SubscriptionNotificationError("duplicate_parameter", index)
        else:
            result[name] = item[key]
    return result


def _status(
    resource: dict[str, Any], version: SubscriptionNotificationVersion
) -> dict[str, Any]:
    expected = (
        "Parameters"
        if version is SubscriptionNotificationVersion.R4
        else "SubscriptionStatus"
    )
    if resource.get("resourceType") != expected:
        raise SubscriptionNotificationError("mixed_version")
    _reject_extensions(resource)
    if version is not SubscriptionNotificationVersion.R4:
        return resource
    params = _parameters(resource.get("parameter"))
    if "status" not in params:
        raise SubscriptionNotificationError("invalid_status")
    events = []
    for parts in params.get("notification-event", []):
        item = _parameters(parts, event=True)
        events.append(
            {
                target: item[key]
                for key, target in (
                    ("event-number", "eventNumber"),
                    ("timestamp", "timestamp"),
                    ("focus", "focus"),
                    ("additional-context", "additionalContext"),
                )
                if key in item
            }
        )
    return {
        target: params[key]
        for key, target in (
            ("subscription", "subscription"),
            ("status", "status"),
            ("type", "type"),
            ("events-since-subscription-start", "eventsSinceSubscriptionStart"),
            ("error", "error"),
        )
        if key in params
    } | {"notificationEvent": events}


def _check_profile_version(
    resource: dict[str, Any], release: SubscriptionNotificationVersion
) -> None:
    """Reject conflicting recognized core/notification profile declarations."""
    meta = resource.get("meta", {})
    if type(meta) is not dict:
        raise SubscriptionNotificationError("invalid_bundle")
    for profile in _array(
        meta.get("profile", []), MAX_EVENT_REFERENCES, "invalid_bundle"
    ):
        if type(profile) is not str or len(profile) > 2_048:
            raise SubscriptionNotificationError("invalid_bundle")
        canonical, _, declared_version = profile.partition("|")
        if canonical.startswith(
            (
                "http://hl7.org/fhir/StructureDefinition/",
                "https://hl7.org/fhir/StructureDefinition/",
            )
        ):
            known = {"4.0.1": "R4", "4.3.0": "R4B", "5.0.0": "R5"}
            if declared_version in known and known[declared_version] != release.value:
                raise SubscriptionNotificationError("mixed_version")
        if canonical.startswith(
            (
                "http://hl7.org/fhir/uv/subscriptions-backport/StructureDefinition/backport-subscription-",
                "https://hl7.org/fhir/uv/subscriptions-backport/StructureDefinition/backport-subscription-",
            )
        ):
            if (
                canonical.endswith("-r4")
                and release is not SubscriptionNotificationVersion.R4
            ):
                raise SubscriptionNotificationError("mixed_version")
            if (
                canonical.endswith("/backport-subscription-notification")
                and release is not SubscriptionNotificationVersion.R4B
            ):
                raise SubscriptionNotificationError("mixed_version")
            if (
                canonical.endswith("-r4b")
                and release is not SubscriptionNotificationVersion.R4B
            ):
                raise SubscriptionNotificationError("mixed_version")


def parse_subscription_notification(
    payload: dict[str, Any] | str | bytes,
    *,
    version: SubscriptionNotificationVersion | FHIRVersion | str,
    previous_event_number: int | None = None,
    global_event_numbers: bool = True,
) -> SubscriptionNotification:
    """Parse an in-memory notification without performing an effect.

    Args:
        payload: Builtin JSON containers or bounded UTF-8 JSON.
        version: Explicit R4 backport, R4B or R5 notification format.
        previous_event_number: Last processed global number from a separately
            governed checkpoint. None leaves the preceding history unknown.
        global_event_numbers: False for R5 guaranteed-delivery, bundle-relative
            numbering; global counter comparisons are then disabled.

    Returns:
        Digest-only event metadata and closed reconciliation findings.

    Raises:
        SubscriptionNotificationError: A stable, value-free refusal code for
            an unsupported format, invalid structure or exceeded bound.
    """
    versions = {
        "R4": SubscriptionNotificationVersion.R4,
        "4.0.1": SubscriptionNotificationVersion.R4,
        "R4B": SubscriptionNotificationVersion.R4B,
        "4.3.0": SubscriptionNotificationVersion.R4B,
        "R5": SubscriptionNotificationVersion.R5,
        "5.0.0": SubscriptionNotificationVersion.R5,
    }
    raw_version = (
        version.value
        if type(version) in (SubscriptionNotificationVersion, FHIRVersion)
        else version
    )
    if type(raw_version) is not str or raw_version not in versions:
        raise SubscriptionNotificationError("unsupported_version")
    release = versions[raw_version]
    if type(global_event_numbers) is not bool or (
        not global_event_numbers and release is not SubscriptionNotificationVersion.R5
    ):
        raise SubscriptionNotificationError("invalid_options")
    _require_count(previous_event_number, optional=True)
    if not global_event_numbers and previous_event_number is not None:
        raise SubscriptionNotificationError("invalid_options")
    bundle, bundle_digest = _load_bundle(payload)
    expected_type = (
        "subscription-notification"
        if release is SubscriptionNotificationVersion.R5
        else "history"
    )
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != expected_type:
        raise SubscriptionNotificationError("invalid_bundle")
    _reject_extensions(bundle)
    _check_profile_version(bundle, release)
    entries = _array(bundle.get("entry"), MAX_NOTIFICATION_ENTRIES, "invalid_bundle")
    if not entries or any(
        type(e) is not dict or type(e.get("resource")) is not dict for e in entries
    ):
        raise SubscriptionNotificationError("invalid_bundle")
    for index, entry in enumerate(entries):
        _reject_extensions(entry, index)
        _check_profile_version(entry["resource"], release)
    status = _status(entries[0]["resource"], release)
    raw_type = status.get("type")
    if type(raw_type) is not str or raw_type not in {
        t.value for t in SubscriptionNotificationType
    }:
        raise SubscriptionNotificationError("invalid_status")
    notification_type = SubscriptionNotificationType(raw_type)
    subscription = _reference(status.get("subscription"))
    state = status.get("status")
    if "status" in status and (
        type(state) is not str
        or state not in {"requested", "active", "error", "off", "entered-in-error"}
    ):
        raise SubscriptionNotificationError("invalid_status")
    if notification_type is SubscriptionNotificationType.QUERY_STATUS and state is None:
        raise SubscriptionNotificationError("invalid_status")
    since = (
        _counter(status["eventsSinceSubscriptionStart"])
        if "eventsSinceSubscriptionStart" in status
        else None
    )
    raw_events = _array(
        status.get("notificationEvent", []), MAX_NOTIFICATION_EVENTS, "invalid_event"
    )
    if (
        notification_type
        in (
            SubscriptionNotificationType.EVENT,
            SubscriptionNotificationType.QUERY_EVENT,
        )
        and not raw_events
    ):
        raise SubscriptionNotificationError("invalid_event")
    if notification_type is SubscriptionNotificationType.QUERY_STATUS and raw_events:
        raise SubscriptionNotificationError("invalid_event")
    if (
        notification_type is SubscriptionNotificationType.QUERY_STATUS
        and len(entries) != 1
    ):
        raise SubscriptionNotificationError("invalid_bundle")
    errors = _array(status.get("error", []), MAX_NOTIFICATION_EVENTS, "invalid_status")
    if any(type(error) is not dict for error in errors):
        raise SubscriptionNotificationError("invalid_status")
    available: set[str] = set()
    for index, entry in enumerate(entries[1:], 1):
        resource = entry["resource"]
        rt = resource.get("resourceType")
        if type(rt) is not str or _RESOURCE_TYPE.fullmatch(rt) is None:
            raise SubscriptionNotificationError("invalid_bundle", index)
        parameters = resource.get("parameter", [])
        if rt == "Parameters" and type(parameters) is not list:
            raise SubscriptionNotificationError("invalid_bundle", index)
        if rt == "SubscriptionStatus" or (
            rt == "Parameters"
            and any(
                p.get("name") == "subscription" for p in parameters if type(p) is dict
            )
        ):
            raise SubscriptionNotificationError("mixed_version", index)
        references = []
        if "fullUrl" in entry:
            references.append(_reference({"reference": entry["fullUrl"]}))
        if "id" in resource:
            if (
                type(resource["id"]) is not str
                or _FHIR_ID.fullmatch(resource["id"]) is None
            ):
                raise SubscriptionNotificationError("invalid_reference", index)
            references.append(rt + "/" + resource["id"])
        for reference in dict.fromkeys(references):
            if not reference or reference in available:
                raise SubscriptionNotificationError("ambiguous_reference", index)
            available.add(reference)
    events: list[SubscriptionEvent] = []
    for index, event in enumerate(raw_events):
        if type(event) is not dict:
            raise SubscriptionNotificationError("invalid_event", index)
        _reject_extensions(event, index)
        number = _counter(event.get("eventNumber"), event=True)
        focus = _reference(event["focus"]) if "focus" in event else None
        context = [
            _reference(ref)
            for ref in _array(
                event.get("additionalContext", []),
                MAX_EVENT_REFERENCES,
                "invalid_reference",
            )
        ]
        if focus is None and context:
            raise SubscriptionNotificationError("invalid_event", index)
        references = ([] if focus is None else [focus]) + context
        mode = (
            SubscriptionContentMode.EMPTY
            if not references
            else (
                SubscriptionContentMode.FULL_RESOURCE
                if all(ref in available for ref in references)
                else SubscriptionContentMode.ID_ONLY
            )
        )
        events.append(
            SubscriptionEvent(
                index,
                number,
                _digest("reference", focus) if focus else None,
                tuple(_digest("reference", ref) for ref in context),
                mode,
            )
        )
    findings: list[SubscriptionNotificationFinding] = []

    def finding(code: SubscriptionFindingCode, position: int | None = None) -> None:
        findings.append(SubscriptionNotificationFinding(code, position))

    if errors:
        findings.append(
            SubscriptionNotificationFinding(
                SubscriptionFindingCode.SERVER_ERROR, count=len(errors)
            )
        )
    if state is not None and state != "active":
        finding(SubscriptionFindingCode.INACTIVE)
    seen: set[int] = set()
    for index, event in enumerate(events):
        if event.event_number in seen or (
            global_event_numbers
            and previous_event_number is not None
            and event.event_number <= previous_event_number
        ):
            finding(SubscriptionFindingCode.DUPLICATE, index)
        if index and event.event_number < events[index - 1].event_number:
            finding(SubscriptionFindingCode.OUT_OF_ORDER, index)
        seen.add(event.event_number)
    if not global_event_numbers:
        cursor = 0
        for number in sorted(seen):
            if number > cursor + 1:
                findings.append(
                    SubscriptionNotificationFinding(
                        SubscriptionFindingCode.GAP,
                        count=number - cursor - 1,
                        first_missing=cursor + 1,
                        last_missing=number - 1,
                    )
                )
            cursor = number
    if global_event_numbers:
        if (
            since is not None
            and previous_event_number is not None
            and since < previous_event_number
        ):
            finding(SubscriptionFindingCode.COUNTER_REGRESSED)
        if seen and since is not None and max(seen) > since:
            finding(SubscriptionFindingCode.COUNTER_BEHIND)
        if notification_type is not SubscriptionNotificationType.QUERY_EVENT:
            if since is None:
                if notification_type is SubscriptionNotificationType.EVENT:
                    finding(SubscriptionFindingCode.COUNTER_MISSING)
            ordered = sorted(
                number
                for number in seen
                if previous_event_number is None or number > previous_event_number
            )
            cursor = previous_event_number
            if cursor is None and ordered:
                cursor = ordered[0] - 1
                if cursor > 0:
                    finding(SubscriptionFindingCode.HISTORY_UNKNOWN)
            elif cursor is None and since is not None and since > 0:
                finding(SubscriptionFindingCode.HISTORY_UNKNOWN)
            for number in ordered:
                if cursor is not None and number > cursor + 1:
                    findings.append(
                        SubscriptionNotificationFinding(
                            SubscriptionFindingCode.GAP,
                            count=number - cursor - 1,
                            first_missing=cursor + 1,
                            last_missing=number - 1,
                        )
                    )
                cursor = number
            if since is not None and cursor is not None and since > cursor:
                findings.append(
                    SubscriptionNotificationFinding(
                        SubscriptionFindingCode.GAP,
                        count=since - cursor,
                        first_missing=cursor + 1,
                        last_missing=since,
                    )
                )
    return SubscriptionNotification(
        release,
        notification_type,
        bundle_digest,
        _digest("reference", subscription),
        since,
        tuple(events),
        tuple(findings),
        len(entries) - 1,
        len(errors),
    )
