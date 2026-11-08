"""Bounded revision-aware PHI holdback for locally screened speech segments."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from ..transcript.revision_ledger import RevisionKind, RevisionLedger, TranscriptToken

WITHHOLDING_SCHEMA_VERSION = "openmed.speech.withholding.v1"
_REF = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REDACTION = "[REDACTED]"


class WithholdingError(ValueError):
    """A fixed-message caller-contract failure, without transcript values."""


class BufferState(str, Enum):
    """Closed and abstained buffers never accept or release further text."""

    OPEN = "open"
    ENDING = "ending"
    CLOSED = "closed"
    ABSTAINED = "abstained"


class HoldReason(str, Enum):
    """Controlled reasons for withholding or discarding protected speech."""

    PROVISIONAL = "provisional"
    TRAILING_CONTEXT = "trailing_context"
    UNRESOLVED_PHI = "unresolved_phi"
    CRITICAL_IDENTIFIER = "critical_identifier"
    SCREEN_FAILED = "screen_failed"
    BUFFER_LIMIT = "buffer_limit"
    LATENCY_LIMIT = "latency_limit"
    CLOCK_INVALID = "clock_invalid"
    OPERATION_LIMIT = "operation_limit"
    CLOSED = "closed"


class PhiResolution(str, Enum):
    """RESOLVED means every detected identifier has a complete redaction span."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    CRITICAL_UNRESOLVED = "critical_unresolved"


def _require(condition: bool) -> None:
    if not condition:
        raise WithholdingError("invalid speech withholding contract")


def _integer(value: object, maximum: int, minimum: int = 0) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _ref(value: object) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


def _text(value: object, maximum: int) -> None:
    _require(type(value) is str and len(value) <= maximum)
    valid = True
    try:
        cast(str, value).encode("utf-8")
    except UnicodeError:
        valid = False
    _require(valid)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class WithholdingPolicy:
    """Explicit ceilings and the amount of trailing context retained.

    Args:
        max_segments: Maximum simultaneously buffered segments, at most 64.
        max_chars: Maximum protected characters held, at most 65,536.
        max_latency_ms: Maximum age since first insertion, including revisions
            and screening time. Expiry discards pending text without release.
        trailing_segments: Segments kept as context until further input or end.
        max_operations: Bound edits, evaluations and total segment identities.
    """

    max_segments: int = 16
    max_chars: int = 32_768
    max_latency_ms: int = 5000
    trailing_segments: int = 1
    max_operations: int = 4096

    def __post_init__(self) -> None:
        _integer(self.max_segments, 64, 1)
        _integer(self.max_chars, 65_536, 1)
        _integer(self.max_latency_ms, 60_000, 1)
        _integer(self.trailing_segments, self.max_segments - 1)
        _integer(self.max_operations, 16_384, 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class PhiScreenDeclaration:
    """Pinned local screening policy/model identity, without free-form names."""

    artifact_digest: str
    requires_network: bool = False

    def __post_init__(self) -> None:
        _require(
            type(self.artifact_digest) is str
            and _DIGEST.fullmatch(self.artifact_digest) is not None
            and self.requires_network is False
        )


@dataclass(frozen=True, slots=True)
class ProtectedSpeechSegment:
    """One current revision for the trusted local screener; never serialize text."""

    segment_ref: str
    revision: int
    finalized: bool
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class PhiScreenRequest:
    """Full bounded pending context and its exact revision/content fingerprint."""

    request_digest: str
    segments: tuple[ProtectedSpeechSegment, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PhiRedactionSpan:
    """Complete half-open Unicode code-point interval of a detected identifier."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _integer(self.start, 65_536)
        _integer(self.end, 65_536)
        _require(self.start < self.end)


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentPhiDecision:
    """Decision for one exact segment revision under the full request context."""

    segment_ref: str
    revision: int
    resolution: PhiResolution
    redactions: tuple[PhiRedactionSpan, ...] = ()

    def __post_init__(self) -> None:
        _ref(self.segment_ref)
        _integer(self.revision, 4096)
        _require(type(self.resolution) is PhiResolution)
        _require(type(self.redactions) is tuple and len(self.redactions) <= 4096)
        _require(all(type(span) is PhiRedactionSpan for span in self.redactions))


@dataclass(frozen=True, slots=True, kw_only=True)
class PhiScreenResult:
    """A response bound to the exact context; omission never implies safety."""

    request_digest: str
    decisions: tuple[SegmentPhiDecision, ...]


class LocalPhiScreener(Protocol):
    """Trusted local detector resolving identifiers across all pending context.

    A RESOLVED decision asserts complete coverage, including direct identifiers
    and cross-segment fragments. Return UNRESOLVED or CRITICAL_UNRESOLVED when
    context is insufficient; confidence alone must not authorize release.
    The runtime enforces bounds and bindings, not detector recall. Qualify the
    detector separately and disable logs, telemetry and remote fallback.
    """

    @property
    def declaration(self) -> PhiScreenDeclaration:
        """Return the local policy/model fingerprint without reading speech."""
        ...

    def screen(self, request: PhiScreenRequest) -> PhiScreenResult:
        """Inspect protected context in memory and return bounded decisions."""
        ...


@dataclass(frozen=True, slots=True)
class RedactionMapping:
    """Map released character intervals back to original finalized lineage."""

    source_start: int
    source_end: int
    output_start: int
    output_end: int
    redacted: bool


@dataclass(frozen=True, slots=True)
class ReleasedSpeechSegment:
    """Final screened text and source-coordinate mappings; text remains protected."""

    segment_ref: str
    revision: int
    request_digest: str
    output_digest: str
    mappings: tuple[RedactionMapping, ...]
    lineage_json: str = field(repr=False)
    text: str = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return offsets and provenance without the released text."""
        return {
            "segment_ref": self.segment_ref,
            "revision": self.revision,
            "request_digest": self.request_digest,
            "output_digest": self.output_digest,
            "mappings": [asdict(mapping) for mapping in self.mappings],
        }


@dataclass(slots=True)
class _Pending:
    ledger: RevisionLedger
    text: str = field(repr=False)
    inserted_ms: int
    redactions: tuple[PhiRedactionSpan, ...] = ()


class WithholdingBuffer:
    """Own bounded trailing text and its existing revision ledger until release.

    Args:
        session_ref: Opaque session identity, never a patient-derived identifier.
        screener: Injected local PHI detector with a pinned declaration.
        policy: Retention, trailing-context and operation ceilings.
        clock: Monotonic seconds. Inject a fake clock for deterministic replay.

    Serialize calls. There is no background scheduler: call ``tick`` while input
    is idle and always ``close`` on exit. Every mutation and release checks age,
    including after screening. Text is never released after its latency budget.
    Caller-held released outputs are not retained or erased by this buffer.
    """

    def __init__(
        self,
        session_ref: str,
        screener: LocalPhiScreener,
        *,
        policy: WithholdingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _ref(session_ref)
        self._policy = policy if policy is not None else WithholdingPolicy()
        _require(type(self._policy) is WithholdingPolicy and callable(clock))
        declaration = None
        try:
            declaration = screener.declaration
            if type(declaration) is PhiScreenDeclaration:
                declaration.__post_init__()
        except Exception:
            declaration = None
        _require(type(declaration) is PhiScreenDeclaration)
        assert declaration is not None
        self._declaration = declaration
        self._session_ref, self._screener, self._clock = session_ref, screener, clock
        self._state = BufferState.OPEN
        self._reason: HoldReason | None = None
        self._pending: dict[str, _Pending] = {}
        self._seen: set[str] = set()
        self._operations = self._released = self._discarded = 0
        self._busy = False
        self._origin: float | None = None
        self._last_clock: float | None = None
        self._now = 0
        self._observe_time()

    def __enter__(self) -> WithholdingBuffer:
        _require(self._state is BufferState.OPEN)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def state(self) -> BufferState:
        """Current lifecycle state; failures discard all pending protected text."""
        return self._state

    @property
    def screen_artifact_digest(self) -> str:
        """Return the pinned local screen identity for external runtime admission."""
        return self._declaration.artifact_digest

    @property
    def reason(self) -> HoldReason | None:
        """Controlled current hold or terminal failure reason."""
        return self._reason

    def _fail(self, reason: HoldReason) -> None:
        self._discarded += len(self._pending)
        self._pending.clear()
        self._state, self._reason = BufferState.ABSTAINED, reason

    def _observe_time(self) -> bool:
        current = None
        try:
            current = self._clock()
        except Exception:
            pass
        if (
            current is None
            or type(current) not in (int, float)
            or (type(current) is int and current.bit_length() > 64)
            or not math.isfinite(current)
            or (self._last_clock is not None and current < self._last_clock)
        ):
            self._fail(HoldReason.CLOCK_INVALID)
            return False
        if self._origin is None:
            self._origin = float(current)
        self._last_clock = float(current)
        elapsed = (current - self._origin) * 1000
        if not math.isfinite(elapsed) or not self._now <= elapsed <= 86_400_000:
            self._fail(HoldReason.CLOCK_INVALID)
            return False
        self._now = math.floor(elapsed)
        if any(
            self._now - pending.inserted_ms > self._policy.max_latency_ms
            for pending in self._pending.values()
        ):
            self._fail(HoldReason.LATENCY_LIMIT)
            return False
        return True

    def _admit(self) -> bool:
        _require(not self._busy)
        _require(self._state in (BufferState.OPEN, BufferState.ENDING))
        if not self._observe_time():
            return False
        if self._operations >= self._policy.max_operations:
            self._fail(HoldReason.OPERATION_LIMIT)
            return False
        self._operations += 1
        return True

    def _capacity(self, length: int, *, replacing: int = 0, added: int = 0) -> bool:
        chars = sum(len(item.text) for item in self._pending.values())
        if (
            len(self._pending) + added > self._policy.max_segments
            or chars - replacing + length > self._policy.max_chars
        ):
            self._fail(HoldReason.BUFFER_LIMIT)
            return False
        return True

    def append(
        self, segment_ref: str, text: str, tokens: tuple[TranscriptToken, ...] = ()
    ) -> bool:
        """Insert a new provisional segment; overflow fails closed for the buffer."""
        _ref(segment_ref)
        _text(text, 65_536)
        _require(self._state is BufferState.OPEN and segment_ref not in self._seen)
        if not self._admit():
            return False
        ledger = RevisionLedger(self._session_ref, segment_ref, len(text), tokens)
        if not self._capacity(len(text), added=1):
            return False
        self._pending[segment_ref] = _Pending(ledger, text, self._now)
        self._seen.add(segment_ref)
        self._reason = HoldReason.PROVISIONAL
        return True

    def edit(
        self,
        segment_ref: str,
        kind: RevisionKind,
        start: int,
        end: int,
        text: str = "",
        tokens: tuple[TranscriptToken, ...] = (),
        *,
        expected_revision: int,
    ) -> bool:
        """Apply the same atomic insert/delete/replace to text and existing lineage.

        Replacement token offsets are fragment-local. Fresh token references are
        required by the ledger. A revision never resets the retention clock.
        """
        _ref(segment_ref)
        _require(segment_ref in self._pending)
        _require(
            type(kind) is RevisionKind
            and kind in (RevisionKind.INSERT, RevisionKind.DELETE, RevisionKind.REPLACE)
        )
        _text(text, 65_536)
        pending = self._pending[segment_ref]
        _require(
            type(expected_revision) is int
            and expected_revision == pending.ledger.revision
            and not pending.ledger.to_dict()["finalized"]
        )
        _integer(start, len(pending.text))
        _integer(end, len(pending.text))
        _require(start <= end)
        _require(kind is not RevisionKind.INSERT or start == end)
        _require(kind is not RevisionKind.DELETE or (text == "" and tokens == ()))
        if not self._admit():
            return False
        updated = pending.text[:start] + text + pending.text[end:]
        if not self._capacity(len(updated), replacing=len(pending.text)):
            return False
        if kind is RevisionKind.INSERT:
            pending.ledger.insert(
                start, len(text), tokens, expected_revision=expected_revision
            )
        elif kind is RevisionKind.DELETE:
            pending.ledger.delete(start, end, expected_revision=expected_revision)
        else:
            pending.ledger.replace(
                start, end, len(text), tokens, expected_revision=expected_revision
            )
        pending.text = updated
        pending.redactions = ()
        self._reason = HoldReason.PROVISIONAL
        return True

    def finalize(self, segment_ref: str, *, expected_revision: int) -> bool:
        """Seal text and timing through the existing ledger; this does not release it."""
        _ref(segment_ref)
        _require(segment_ref in self._pending)
        ledger = self._pending[segment_ref].ledger
        _require(
            type(expected_revision) is int
            and expected_revision == ledger.revision
            and not ledger.to_dict()["finalized"]
        )
        if not self._admit():
            return False
        ledger.finalize(expected_revision=expected_revision)
        return True

    def lineage(self, segment_ref: str) -> dict[str, Any]:
        """Copy current value-free lineage for citations or revision consumers."""
        _ref(segment_ref)
        _require(segment_ref in self._pending)
        return self._pending[segment_ref].ledger.to_dict()

    def tick(self) -> BufferState:
        """Enforce the declared deadline while capture is idle; never release text."""
        if self._state in (BufferState.OPEN, BufferState.ENDING):
            self._admit()
        return self._state

    def finish(self) -> tuple[ReleasedSpeechSegment, ...]:
        """Mark end of input and screen finalized text without trailing-context hold.

        Provisional or unresolved segments remain held until explicit revision,
        finalization and re-evaluation, deadline expiry, or close. No implicit
        finalization or safe decision is manufactured at end of stream.
        """
        _require(
            not self._busy and self._state in (BufferState.OPEN, BufferState.ENDING)
        )
        self._state = BufferState.ENDING
        return self.release_ready()

    def _request(self) -> PhiScreenRequest:
        segments = tuple(
            ProtectedSpeechSegment(
                ref, item.ledger.revision, item.ledger.to_dict()["finalized"], item.text
            )
            for ref, item in self._pending.items()
        )
        fingerprint = _digest(
            {
                "session": self._session_ref,
                "policy": asdict(self._policy),
                "detector": self._declaration.artifact_digest,
                "ending": self._state is BufferState.ENDING,
                "segments": [
                    [
                        item.segment_ref,
                        item.revision,
                        item.finalized,
                        hashlib.sha256(item.text.encode()).hexdigest(),
                    ]
                    for item in segments
                ],
            }
        )
        return PhiScreenRequest(fingerprint, segments)

    def _validate(self, request: PhiScreenRequest, response: PhiScreenResult) -> None:
        _require(
            type(response) is PhiScreenResult
            and response.request_digest == request.request_digest
        )
        _require(
            type(response.decisions) is tuple
            and len(response.decisions) == len(request.segments)
        )
        for segment, decision in zip(request.segments, response.decisions):
            _require(type(decision) is SegmentPhiDecision)
            decision.__post_init__()
            _require(
                (decision.segment_ref, decision.revision)
                == (segment.segment_ref, segment.revision)
            )
            _require(all(span.end <= len(segment.text) for span in decision.redactions))

    def release_ready(self) -> tuple[ReleasedSpeechSegment, ...]:
        """Screen current context and release only the finalized, resolved prefix.

        No previous decision is reused after a revision or context change. Invalid,
        stale, partial or failed detector responses hold all text until retry or
        expiry. Expiry during detector execution discards its output too.
        """
        if not self._admit():
            return ()
        if not self._pending:
            if self._state is BufferState.ENDING:
                self._state = BufferState.CLOSED
            return ()
        request = self._request()
        eligible = len(request.segments) - (
            self._policy.trailing_segments if self._state is BufferState.OPEN else 0
        )
        if eligible <= 0:
            self._reason = HoldReason.TRAILING_CONTEXT
            return ()
        if any(not segment.finalized for segment in request.segments):
            self._reason = HoldReason.PROVISIONAL
            return ()
        valid = False
        known_redactions: dict[str, tuple[PhiRedactionSpan, ...]] = {}
        self._busy = True
        try:
            declaration = self._screener.declaration
            _require(
                type(declaration) is PhiScreenDeclaration
                and declaration == self._declaration
            )
            response = self._screener.screen(request)
            self._validate(request, response)
            known_redactions = {
                segment.segment_ref: _merge_spans(
                    self._pending[segment.segment_ref].redactions + decision.redactions
                )
                for segment, decision in zip(request.segments, response.decisions)
            }
            valid = True
        except Exception:
            pass
        finally:
            self._busy = False
        if not self._observe_time():
            return ()
        if not valid:
            self._reason = HoldReason.SCREEN_FAILED
            return ()
        # Finalized context cannot change under an already released decision.
        # Keep every known redaction when earlier context leaves the buffer;
        # subsequent detector omissions never make known PHI releasable again.
        for segment, decision in zip(request.segments, response.decisions):
            pending = self._pending[segment.segment_ref]
            pending.redactions = known_redactions[segment.segment_ref]
        unresolved = [
            decision
            for decision in response.decisions
            if decision.resolution is not PhiResolution.RESOLVED
        ]
        if unresolved:
            self._reason = (
                HoldReason.CRITICAL_IDENTIFIER
                if any(
                    item.resolution is PhiResolution.CRITICAL_UNRESOLVED
                    for item in unresolved
                )
                else HoldReason.UNRESOLVED_PHI
            )
            return ()
        released = []
        self._reason = None
        for segment, decision in zip(request.segments[:eligible], response.decisions):
            if not segment.finalized:
                self._reason = HoldReason.PROVISIONAL
                break
            if decision.resolution is not PhiResolution.RESOLVED:
                self._reason = (
                    HoldReason.CRITICAL_IDENTIFIER
                    if decision.resolution is PhiResolution.CRITICAL_UNRESOLVED
                    else HoldReason.UNRESOLVED_PHI
                )
                break
            pending = self._pending[segment.segment_ref]
            text, mappings = _redact(segment.text, pending.redactions)
            released.append(
                ReleasedSpeechSegment(
                    segment.segment_ref,
                    segment.revision,
                    request.request_digest,
                    hashlib.sha256(text.encode()).hexdigest(),
                    mappings,
                    pending.ledger.to_json(),
                    text,
                )
            )
        # Rendering and lineage export also consume the declared latency budget.
        # Commit the whole prefix only after their work passes the final check.
        if not self._observe_time():
            return ()
        for item in released:
            self._pending.pop(item.segment_ref)
        self._released += len(released)
        if not self._pending and self._state is BufferState.ENDING:
            self._state = BufferState.CLOSED
        elif self._pending and self._reason is None:
            self._reason = HoldReason.TRAILING_CONTEXT
        return tuple(released)

    def close(self) -> None:
        """Drop all pending protected text without release; leave borrowed detector open."""
        _require(not self._busy)
        self._discarded += len(self._pending)
        self._pending.clear()
        if self._state is not BufferState.ABSTAINED:
            self._state, self._reason = BufferState.CLOSED, HoldReason.CLOSED

    def to_json(self) -> str:
        """Return bounded codes, counters and opaque revision references only."""
        return json.dumps(
            {
                "schema_version": WITHHOLDING_SCHEMA_VERSION,
                "state": self._state.value,
                "reason": self._reason.value if self._reason else None,
                "policy": asdict(self._policy),
                "screen_artifact_digest": self._declaration.artifact_digest,
                "operations": self._operations,
                "released": self._released,
                "discarded": self._discarded,
                "pending": [
                    {
                        "segment_ref": ref,
                        "revision": item.ledger.revision,
                        "characters": len(item.text),
                        "age_ms": self._now - item.inserted_ms,
                    }
                    for ref, item in self._pending.items()
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _merge_spans(spans: tuple[PhiRedactionSpan, ...]) -> tuple[PhiRedactionSpan, ...]:
    merged: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if merged and span.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span.end))
        else:
            merged.append((span.start, span.end))
    _require(len(merged) <= 4096)
    return tuple(PhiRedactionSpan(start, end) for start, end in merged)


def _redact(
    text: str, spans: tuple[PhiRedactionSpan, ...]
) -> tuple[str, tuple[RedactionMapping, ...]]:
    parts: list[str] = []
    mappings: list[RedactionMapping] = []
    source_at = output_at = 0
    for span in spans:
        start, end = span.start, span.end
        if start > source_at:
            part = text[source_at:start]
            parts.append(part)
            mappings.append(
                RedactionMapping(
                    source_at, start, output_at, output_at + len(part), False
                )
            )
            output_at += len(part)
        parts.append(_REDACTION)
        mappings.append(
            RedactionMapping(start, end, output_at, output_at + len(_REDACTION), True)
        )
        output_at += len(_REDACTION)
        source_at = end
    if source_at < len(text):
        parts.append(text[source_at:])
        mappings.append(
            RedactionMapping(
                source_at,
                len(text),
                output_at,
                output_at + len(text) - source_at,
                False,
            )
        )
    return "".join(parts), tuple(mappings)
