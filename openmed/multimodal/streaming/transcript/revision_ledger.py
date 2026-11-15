"""Bounded, deterministic lineage for evolving transcript spans, without text."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

REVISION_SCHEMA_VERSION = "openmed.transcript.revisions.v1"
MAX_TEXT_LENGTH = 65_536
MAX_TOKENS = 4096
MAX_REVISIONS = 4096
MAX_SPANS = 4096
MAX_TOKEN_LINKS = 65_536
MAX_TIME_MS = 86_400_000
_REF = re.compile(r"[0-9a-f]{32}\Z")


class RevisionLedgerError(ValueError):
    """Value-free malformed, stale or sealed-ledger error."""


class RevisionKind(str, Enum):
    """Supported atomic operations on one stable segment identity."""

    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    TIMESTAMP_SHIFT = "timestamp_shift"
    FINALIZE = "finalize"


class SpanInvalidation(str, Enum):
    """Why a formerly registered citation can no longer be remapped safely."""

    TEXT_CHANGED = "cited_text_changed"
    TOKEN_CHANGED = "cited_token_changed"


def _require(condition: bool) -> None:
    if not condition:
        raise RevisionLedgerError("invalid transcript lineage")


def _ref(value: object) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


def _integer(value: object, maximum: int) -> None:
    _require(type(value) is int and 0 <= value <= maximum)


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptToken:
    """Opaque token identity with character and stream-relative time coordinates.

    Args:
        token_ref: Opaque, never-reused token reference within the ledger.
        start: Inclusive Unicode code-point offset, not a UTF-8 byte offset.
        end: Exclusive code-point offset.
        start_ms: Inclusive stream-relative token time in milliseconds.
        end_ms: Exclusive stream-relative token time; zero-duration tokens allowed.
    """

    token_ref: str
    start: int
    end: int
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _ref(self.token_ref)
        for value in (self.start, self.end):
            _integer(value, MAX_TEXT_LENGTH)
        for value in (self.start_ms, self.end_ms):
            _integer(value, MAX_TIME_MS)
        _require(self.start < self.end and self.start_ms <= self.end_ms)


def _tokens(tokens: tuple[TranscriptToken, ...], length: int) -> None:
    _require(type(tokens) is tuple and len(tokens) <= MAX_TOKENS)
    _require(all(type(token) is TranscriptToken for token in tokens))
    _require(len({token.token_ref for token in tokens}) == len(tokens))
    _require(all(token.end <= length for token in tokens))
    for left, right in zip(tokens, tokens[1:]):
        _require(left.end <= right.start)
        _require(left.start_ms <= right.start_ms and left.end_ms <= right.end_ms)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpanReference:
    """Registered citation origin; use RevisionLedger.capture_span to create it.

    Coordinates and token references bind the original citation to one segment
    revision. A ledger accepts only the exact registered reference, not an edited
    copy or a reference from a different session.
    """

    span_ref: str
    session_ref: str
    segment_ref: str
    revision: int
    start: int
    end: int
    token_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (self.span_ref, self.session_ref, self.segment_ref):
            _ref(value)
        _integer(self.revision, MAX_REVISIONS)
        _integer(self.start, MAX_TEXT_LENGTH)
        _integer(self.end, MAX_TEXT_LENGTH)
        _require(self.start < self.end)
        _require(
            type(self.token_refs) is tuple and 0 < len(self.token_refs) <= MAX_TOKENS
        )
        for value in self.token_refs:
            _ref(value)
        _require(len(set(self.token_refs)) == len(self.token_refs))


@dataclass(frozen=True, slots=True)
class MappedSpan:
    """Current citation result; invalidated coordinates and timing are withheld."""

    reference: SpanReference
    revision: int
    start: int | None
    end: int | None
    start_ms: int | None
    end_ms: int | None
    invalidation: SpanInvalidation | None
    finalized: bool

    @property
    def valid(self) -> bool:
        """Whether the original citation still has unambiguous current lineage."""
        return self.invalidation is None


class RevisionLedger:
    """Atomic insert/delete/replace, timestamp shifts and finalization for a segment.

    Args:
        session_ref: Opaque session reference, independent of patient identifiers.
        segment_ref: Stable opaque utterance identity, never the transcript text.
        length: Initial text length in Unicode code points.
        tokens: Ordered initial token identities and source coordinates.

    The caller applies the same operations to its protected transcript. This
    ledger never receives text and cannot verify that assertion. Use minimal
    edits to preserve unchanged token identities; replaced tokens always get new
    references. Serialize access to an instance and use expected_revision to
    reject stale updates. Finalization seals both text and timing changes.
    """

    def __init__(
        self,
        session_ref: str,
        segment_ref: str,
        length: int,
        tokens: tuple[TranscriptToken, ...] = (),
    ) -> None:
        _ref(session_ref)
        _ref(segment_ref)
        _integer(length, MAX_TEXT_LENGTH)
        _tokens(tokens, length)
        self._session_ref, self._segment_ref = session_ref, segment_ref
        self._length, self._tokens = length, tokens
        self._revision = 0
        self._finalized = False
        self._seen = {token.token_ref for token in tokens}
        self._references: dict[str, SpanReference] = {}
        self._mapped: dict[
            str, tuple[int | None, int | None, SpanInvalidation | None]
        ] = {}
        self._operations: list[dict[str, Any]] = []
        self._span_links = 0
        self._history_links = 0

    @property
    def revision(self) -> int:
        """Current revision; all mutating operations advance it exactly once."""
        return self._revision

    @property
    def tokens(self) -> tuple[TranscriptToken, ...]:
        """Current immutable token coordinates, without transcript values."""
        return self._tokens

    def _admit(self, expected_revision: int) -> None:
        _integer(expected_revision, MAX_REVISIONS)
        _require(not self._finalized and expected_revision == self._revision)
        _require(self._revision < MAX_REVISIONS)

    def capture_span(self, span_ref: str, start: int, end: int) -> SpanReference:
        """Register a citation against current half-open character offsets.

        At least one token must intersect it. Gaps between tokens carry no claim
        of recognized content. Token timing covers whole intersecting tokens;
        no sub-token acoustic alignment is inferred.
        """
        _ref(span_ref)
        _integer(start, self._length)
        _integer(end, self._length)
        _require(start < end and span_ref not in self._references)
        _require(len(self._references) < MAX_SPANS)
        refs = tuple(
            t.token_ref for t in self._tokens if t.start < end and start < t.end
        )
        _require(bool(refs))
        _require(self._span_links + len(refs) <= MAX_TOKEN_LINKS)
        reference = SpanReference(
            span_ref=span_ref,
            session_ref=self._session_ref,
            segment_ref=self._segment_ref,
            revision=self._revision,
            start=start,
            end=end,
            token_refs=refs,
        )
        self._references[span_ref] = reference
        self._mapped[span_ref] = (start, end, None)
        self._span_links += len(refs)
        return reference

    def remap_span(self, reference: SpanReference) -> MappedSpan:
        """Resolve an exact registered citation or withhold ambiguous coordinates."""
        _require(type(reference) is SpanReference)
        _require(self._references.get(reference.span_ref) == reference)
        start, end, invalidation = self._mapped[reference.span_ref]
        references = set(reference.token_refs)
        selected = [t for t in self._tokens if t.token_ref in references]
        first = min(t.start_ms for t in selected) if invalidation is None else None
        last = max(t.end_ms for t in selected) if invalidation is None else None
        return MappedSpan(
            reference,
            self._revision,
            start,
            end,
            first,
            last,
            invalidation,
            self._finalized,
        )

    def insert(
        self,
        at: int,
        length: int,
        tokens: tuple[TranscriptToken, ...],
        *,
        expected_revision: int,
    ) -> None:
        """Insert a positive-length fragment; token character offsets are fragment-local."""
        self._edit(RevisionKind.INSERT, at, at, length, tokens, expected_revision)

    def delete(self, start: int, end: int, *, expected_revision: int) -> None:
        """Delete a nonempty half-open interval and retire all intersecting tokens."""
        self._edit(RevisionKind.DELETE, start, end, 0, (), expected_revision)

    def replace(
        self,
        start: int,
        end: int,
        length: int,
        tokens: tuple[TranscriptToken, ...],
        *,
        expected_revision: int,
    ) -> None:
        """Replace a nonempty interval with a positive-length fragment and fresh tokens."""
        self._edit(RevisionKind.REPLACE, start, end, length, tokens, expected_revision)

    def _edit(
        self,
        kind: RevisionKind,
        start: int,
        end: int,
        length: int,
        tokens: tuple[TranscriptToken, ...],
        expected_revision: int,
    ) -> None:
        self._admit(expected_revision)
        _integer(start, self._length)
        _integer(end, self._length)
        _integer(length, MAX_TEXT_LENGTH)
        _require(start <= end)
        _require(length > 0 if kind is RevisionKind.INSERT else start < end)
        _require(kind is not RevisionKind.REPLACE or length > 0)
        updated_length = self._length + length - (end - start)
        _integer(updated_length, MAX_TEXT_LENGTH)
        _tokens(tokens, length)
        new_refs = {token.token_ref for token in tokens}
        _require(not new_refs.intersection(self._seen))
        _require(len(self._seen) + len(new_refs) <= MAX_TOKENS)
        delta = length - (end - start)
        retained, retired = [], set()
        for token in self._tokens:
            intersects = (
                (token.start < end and start < token.end)
                if start != end
                else (token.start < start < token.end)
            )
            if intersects:
                retired.add(token.token_ref)
            else:
                shift = delta if token.start >= end else 0
                retained.append(
                    replace(token, start=token.start + shift, end=token.end + shift)
                )
        retained.extend(
            replace(token, start=token.start + start, end=token.end + start)
            for token in tokens
        )
        updated_tokens = tuple(
            sorted(retained, key=lambda t: (t.start, t.end, t.token_ref))
        )
        _tokens(updated_tokens, updated_length)
        links = len(retired) + len(new_refs)
        _require(self._history_links + links <= MAX_TOKEN_LINKS)
        mapped = self._mapped.copy()
        for key, (left, right, invalidation) in self._mapped.items():
            if invalidation is not None:
                continue
            assert left is not None and right is not None
            changed = (
                (left < start < right)
                if start == end
                else (left < end and start < right)
            )
            if changed:
                mapped[key] = (None, None, SpanInvalidation.TEXT_CHANGED)
            elif retired.intersection(self._references[key].token_refs):
                mapped[key] = (None, None, SpanInvalidation.TOKEN_CHANGED)
            else:
                shift = delta if end <= left else 0
                mapped[key] = (left + shift, right + shift, None)
        operation = {
            "revision": self._revision + 1,
            "kind": kind.value,
            "start": start,
            "end": end,
            "inserted_length": length,
            "retired_tokens": tuple(sorted(retired)),
            "new_tokens": tuple(sorted(new_refs)),
        }
        self._tokens, self._length, self._mapped = (
            updated_tokens,
            updated_length,
            mapped,
        )
        self._seen.update(new_refs)
        self._operations.append(operation)
        self._history_links += links
        self._revision += 1

    def shift_timestamps(
        self, token_refs: tuple[str, ...], delta_ms: int, *, expected_revision: int
    ) -> None:
        """Shift selected token clocks atomically, preserving character positions and IDs."""
        self._admit(expected_revision)
        _require(type(token_refs) is tuple and 0 < len(token_refs) <= MAX_TOKENS)
        for token_ref in token_refs:
            _ref(token_ref)
        selected = set(token_refs)
        _require(len(selected) == len(token_refs))
        _require(selected.issubset({t.token_ref for t in self._tokens}))
        _require(self._history_links + len(selected) <= MAX_TOKEN_LINKS)
        _require(
            type(delta_ms) is int
            and -MAX_TIME_MS <= delta_ms <= MAX_TIME_MS
            and delta_ms != 0
        )
        tokens = tuple(
            replace(t, start_ms=t.start_ms + delta_ms, end_ms=t.end_ms + delta_ms)
            if t.token_ref in selected
            else t
            for t in self._tokens
        )
        _tokens(tokens, self._length)
        self._tokens = tokens
        self._history_links += len(selected)
        self._revision += 1
        self._operations.append(
            {
                "revision": self._revision,
                "kind": RevisionKind.TIMESTAMP_SHIFT.value,
                "token_refs": tuple(sorted(selected)),
                "delta_ms": delta_ms,
            }
        )

    def finalize(self, *, expected_revision: int) -> None:
        """Seal the segment, retaining its identity and all valid citation coordinates."""
        self._admit(expected_revision)
        self._revision += 1
        self._finalized = True
        self._operations.append(
            {"revision": self._revision, "kind": RevisionKind.FINALIZE.value}
        )

    def to_dict(self) -> dict[str, Any]:
        """Export opaque lineage, offsets and relative timing; never transcript text."""
        return {
            "schema_version": REVISION_SCHEMA_VERSION,
            "session_ref": self._session_ref,
            "segment_ref": self._segment_ref,
            "revision": self._revision,
            "length": self._length,
            "finalized": self._finalized,
            "tokens": [asdict(token) for token in self._tokens],
            "operations": [operation.copy() for operation in self._operations],
            "spans": [
                asdict(self.remap_span(self._references[key]))
                for key in sorted(self._references)
            ],
        }

    def to_json(self) -> str:
        """Return deterministic content-free ASCII JSON without file or network IO."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
