"""Declare and drive interchangeable offline streaming-ASR providers.

Ambient dictation has to swap local speech engines without changing the
surrounding application. This module fixes that surface: a provider declares its
provenance, languages and limits, and a session accepts typed audio chunks while
tracking partial hypotheses, finalized segments, token times and language
confidence through an explicit bounded buffer.

Nothing here opens a socket, imports a model, or reads audio. A declaration that
requires network access is rejected outright, buffering is bounded by declared
limits, cancellation and finalization are terminal, and every diagnostic carries
counts instead of payload. Raw chunk payloads and hypothesis text never appear in
``repr`` output, exception messages, or session reports.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

SPEECH_PROVIDER_CONTRACT_VERSION: Final[str] = (
    "openmed.multimodal.streaming.speech.providers.v1"
)

MAX_CHUNK_BYTE_COUNT: Final[int] = 8 * 1024 * 1024
MAX_CHUNK_DURATION_MS: Final[int] = 60_000
MAX_CHUNK_SEQUENCE: Final[int] = (1 << 31) - 1
MAX_SAMPLE_RATE_HZ: Final[int] = 768_000
MAX_CHANNEL_COUNT: Final[int] = 64

MAX_BUFFERED_CHUNKS: Final[int] = 4096
MAX_BUFFERED_BYTE_COUNT: Final[int] = 512 * 1024 * 1024
MAX_BUFFERED_DURATION_MS: Final[int] = 3_600_000
MAX_STREAM_DURATION_MS: Final[int] = 86_400_000
MAX_PARTIAL_HYPOTHESES: Final[int] = 4096
MAX_SEGMENTS: Final[int] = 4096
MAX_TOKEN_TIMES: Final[int] = 4096
MAX_HYPOTHESIS_CHARS: Final[int] = 8192
MAX_PROVIDER_LANGUAGES: Final[int] = 64
MAX_PROVIDER_REVISION_CHARS: Final[int] = 64
MAX_ENTRYPOINT_CHARS: Final[int] = 128

SPEECH_PROVIDER_REASON_CODES: Final[tuple[str, ...]] = (
    "provider_requires_network",
    "buffer_limit_exceeded",
    "stream_duration_exceeded",
    "chunk_sequence_out_of_order",
    "hypothesis_revision_out_of_order",
    "partial_hypothesis_limit_exceeded",
    "segment_index_out_of_order",
    "segment_times_out_of_order",
    "segment_limit_exceeded",
    "session_closed",
    "session_cancelled",
)

_PROVIDER_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$"
)
_PROVIDER_REVISION_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.+-]{0,62}[A-Za-z0-9])?$"
)
_MODEL_FINGERPRINT_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_LANGUAGE_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_ENTRYPOINT_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+$"
)

_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "provider_id",
    "provider_revision",
    "state",
    "buffered_chunks",
    "buffered_byte_count",
    "buffered_duration_ms",
    "stream_duration_ms",
    "partial_hypotheses",
    "finalized_segments",
    "reason_code",
    "requires_network",
)


class StreamState(str, Enum):
    """Closed lifecycle vocabulary for one streaming session.

    Values:
        OPEN: The session accepts chunks, hypotheses and segments.
        FINALIZED: The session completed and accepts no further input.
        CANCELLED: The session was cancelled and accepts no further input.
    """

    OPEN = "open"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


class SpeechProviderError(ValueError):
    """Value-free failure raised for an unusable provider or session step.

    The message is the category itself, so no rejected payload, hypothesis text,
    or identifier ever reaches a log line.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One typed, declared slice of an audio stream.

    The payload is required input but stays out of ``repr`` and out of
    :meth:`to_dict`; only its declared size travels with diagnostics.

    Attributes:
        sequence: Zero-based position in the stream, increasing by one.
        payload: Encoded audio bytes owned by the caller.
        sample_rate_hz: Sample rate of the payload.
        channel_count: Interleaved channel count.
        duration_ms: Duration covered by the payload in milliseconds.
    """

    sequence: int
    payload: bytes = field(repr=False)
    sample_rate_hz: int
    channel_count: int
    duration_ms: int

    def __post_init__(self) -> None:
        _require_bounded_int(self.sequence, "chunk_sequence", 0, MAX_CHUNK_SEQUENCE)
        if type(self.payload) is not bytes:
            raise SpeechProviderError("audio_chunk_payload_invalid")
        if not 1 <= len(self.payload) <= MAX_CHUNK_BYTE_COUNT:
            raise SpeechProviderError("audio_chunk_payload_out_of_range")
        _require_bounded_int(
            self.sample_rate_hz, "chunk_sample_rate_hz", 1, MAX_SAMPLE_RATE_HZ
        )
        _require_bounded_int(
            self.channel_count, "chunk_channel_count", 1, MAX_CHANNEL_COUNT
        )
        _require_bounded_int(
            self.duration_ms, "chunk_duration_ms", 1, MAX_CHUNK_DURATION_MS
        )

    @property
    def byte_count(self) -> int:
        """Return the declared payload size in bytes."""

        return len(self.payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a content-free manifest for this chunk."""

        return {
            "sequence": self.sequence,
            "byte_count": self.byte_count,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_count": self.channel_count,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class TokenTime:
    """Declared timing for a single recognized token.

    Attributes:
        token_index: Zero-based position of the token in its hypothesis.
        start_ms: Inclusive start offset in milliseconds.
        end_ms: Inclusive end offset in milliseconds.
        confidence_ppm: Confidence in parts per million.
        token: Token text, excluded from ``repr`` and from diagnostics.
    """

    token_index: int
    start_ms: int
    end_ms: int
    confidence_ppm: int
    token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        _require_bounded_int(self.token_index, "token_index", 0, MAX_TOKEN_TIMES)
        _require_bounded_int(self.start_ms, "token_start_ms", 0, MAX_STREAM_DURATION_MS)
        _require_bounded_int(self.end_ms, "token_end_ms", 0, MAX_STREAM_DURATION_MS)
        if self.end_ms < self.start_ms:
            raise SpeechProviderError("token_time_range_invalid")
        _require_bounded_int(self.confidence_ppm, "token_confidence_ppm", 0, 1_000_000)
        if type(self.token) is not str or len(self.token) > MAX_HYPOTHESIS_CHARS:
            raise SpeechProviderError("token_text_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return declared timing without echoing token text."""

        return {
            "token_index": self.token_index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence_ppm": self.confidence_ppm,
            "token_length": len(self.token),
        }


@dataclass(frozen=True, slots=True)
class LanguageConfidence:
    """Declared language guess with an integer confidence.

    Attributes:
        language: Lowercase bounded language tag.
        confidence_ppm: Confidence in parts per million.
    """

    language: str
    confidence_ppm: int

    def __post_init__(self) -> None:
        _require_language(self.language)
        _require_bounded_int(
            self.confidence_ppm, "language_confidence_ppm", 0, 1_000_000
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the declared language guess."""

        return {
            "language": self.language,
            "confidence_ppm": self.confidence_ppm,
        }


@dataclass(frozen=True, slots=True)
class ProviderProvenance:
    """Immutable identity of one local speech engine.

    Attributes:
        provider_id: Lowercase bounded identifier.
        provider_revision: Bounded revision label of the adapter build.
        model_fingerprint: ``sha256:`` fingerprint of the weights.
        requires_network: Whether the engine needs network access to run.
    """

    provider_id: str
    provider_revision: str
    model_fingerprint: str
    requires_network: bool = False

    def __post_init__(self) -> None:
        _require_provider_id(self.provider_id)
        if (
            type(self.provider_revision) is not str
            or len(self.provider_revision) > MAX_PROVIDER_REVISION_CHARS
            or _PROVIDER_REVISION_RE.fullmatch(self.provider_revision) is None
        ):
            raise SpeechProviderError("provider_revision_invalid")
        if (
            type(self.model_fingerprint) is not str
            or _MODEL_FINGERPRINT_RE.fullmatch(self.model_fingerprint) is None
        ):
            raise SpeechProviderError("provider_model_fingerprint_invalid")
        if type(self.requires_network) is not bool:
            raise SpeechProviderError("provider_network_flag_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return the declared provenance in fixed order."""

        return {
            "provider_id": self.provider_id,
            "provider_revision": self.provider_revision,
            "model_fingerprint": self.model_fingerprint,
            "requires_network": self.requires_network,
        }


@dataclass(frozen=True, slots=True)
class SpeechStreamLimits:
    """Explicit bounds a session enforces while buffering and emitting.

    Attributes:
        max_buffered_chunks: Chunks held before a drain is mandatory.
        max_buffered_byte_count: Payload bytes held before a drain is mandatory.
        max_buffered_duration_ms: Audio milliseconds held before a drain.
        max_stream_duration_ms: Audio milliseconds one stream may cover.
        max_partial_hypotheses: Partial hypotheses one stream may emit.
        max_segments: Finalized segments one stream may emit.
    """

    max_buffered_chunks: int = MAX_BUFFERED_CHUNKS
    max_buffered_byte_count: int = MAX_BUFFERED_BYTE_COUNT
    max_buffered_duration_ms: int = MAX_BUFFERED_DURATION_MS
    max_stream_duration_ms: int = MAX_STREAM_DURATION_MS
    max_partial_hypotheses: int = MAX_PARTIAL_HYPOTHESES
    max_segments: int = MAX_SEGMENTS

    def __post_init__(self) -> None:
        for name, value, ceiling in (
            (
                "limit_max_buffered_chunks",
                self.max_buffered_chunks,
                MAX_BUFFERED_CHUNKS,
            ),
            (
                "limit_max_buffered_byte_count",
                self.max_buffered_byte_count,
                MAX_BUFFERED_BYTE_COUNT,
            ),
            (
                "limit_max_buffered_duration_ms",
                self.max_buffered_duration_ms,
                MAX_BUFFERED_DURATION_MS,
            ),
            (
                "limit_max_stream_duration_ms",
                self.max_stream_duration_ms,
                MAX_STREAM_DURATION_MS,
            ),
            (
                "limit_max_partial_hypotheses",
                self.max_partial_hypotheses,
                MAX_PARTIAL_HYPOTHESES,
            ),
            ("limit_max_segments", self.max_segments, MAX_SEGMENTS),
        ):
            _require_bounded_int(value, name, 1, ceiling)

    def to_dict(self) -> dict[str, Any]:
        """Return the declared limits in fixed order."""

        return {
            "max_buffered_chunks": self.max_buffered_chunks,
            "max_buffered_byte_count": self.max_buffered_byte_count,
            "max_buffered_duration_ms": self.max_buffered_duration_ms,
            "max_stream_duration_ms": self.max_stream_duration_ms,
            "max_partial_hypotheses": self.max_partial_hypotheses,
            "max_segments": self.max_segments,
        }


@dataclass(frozen=True, slots=True)
class SpeechProviderDeclaration:
    """What one local speech engine promises before any audio arrives.

    A declaration that requires network access is rejected here, so a
    network-only engine can never register or open a session.

    Attributes:
        provenance: Identity and offline flag of the engine.
        languages: Accepted lowercase language tags, sorted and unique.
        sample_rates_hz: Accepted sample rates, sorted and unique.
        limits: Buffering and emission bounds for one session.
    """

    provenance: ProviderProvenance
    languages: tuple[str, ...]
    sample_rates_hz: tuple[int, ...]
    limits: SpeechStreamLimits = field(default_factory=SpeechStreamLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ProviderProvenance):
            raise SpeechProviderError("provider_provenance_type_invalid")
        if self.provenance.requires_network:
            raise SpeechProviderError("provider_requires_network")
        _require_sorted_languages(self.languages)
        _require_sorted_ints(
            self.sample_rates_hz, "provider_sample_rates", 1, MAX_SAMPLE_RATE_HZ
        )
        if not isinstance(self.limits, SpeechStreamLimits):
            raise SpeechProviderError("provider_limits_type_invalid")

    @property
    def provider_id(self) -> str:
        """Return the declared provider identifier."""

        return self.provenance.provider_id

    def to_dict(self) -> dict[str, Any]:
        """Return the declaration as a deterministic dictionary."""

        return {
            "provenance": self.provenance.to_dict(),
            "languages": list(self.languages),
            "sample_rates_hz": list(self.sample_rates_hz),
            "limits": self.limits.to_dict(),
        }

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class PartialHypothesis:
    """One revision of a streaming transcription guess.

    Attributes:
        revision: Monotonically increasing revision of the same utterance.
        text: Hypothesis text, excluded from ``repr`` and diagnostics.
        is_final: Whether the engine considers this revision stable.
        token_times: Declared token timings, ordered by token index.
        language: Optional language guess for the hypothesis.
    """

    revision: int
    text: str = field(repr=False)
    is_final: bool = False
    token_times: tuple[TokenTime, ...] = ()
    language: LanguageConfidence | None = None

    def __post_init__(self) -> None:
        _require_bounded_int(
            self.revision, "hypothesis_revision", 0, MAX_CHUNK_SEQUENCE
        )
        if type(self.text) is not str or len(self.text) > MAX_HYPOTHESIS_CHARS:
            raise SpeechProviderError("hypothesis_text_invalid")
        if type(self.is_final) is not bool:
            raise SpeechProviderError("hypothesis_final_flag_invalid")
        _require_token_times(self.token_times)
        if self.language is not None and not isinstance(
            self.language, LanguageConfidence
        ):
            raise SpeechProviderError("hypothesis_language_type_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a content-free summary of this hypothesis revision."""

        return {
            "revision": self.revision,
            "text_length": len(self.text),
            "is_final": self.is_final,
            "token_count": len(self.token_times),
            "language": None if self.language is None else self.language.language,
        }


@dataclass(frozen=True, slots=True)
class FinalizedSegment:
    """One closed utterance emitted by a session.

    Attributes:
        segment_index: Zero-based position of the segment in the stream.
        text: Segment text, excluded from ``repr`` and diagnostics.
        start_ms: Inclusive start offset in milliseconds.
        end_ms: Inclusive end offset in milliseconds.
        token_times: Declared token timings, ordered by token index.
        language: Optional language guess for the segment.
    """

    segment_index: int
    text: str = field(repr=False)
    start_ms: int = 0
    end_ms: int = 0
    token_times: tuple[TokenTime, ...] = ()
    language: LanguageConfidence | None = None

    def __post_init__(self) -> None:
        _require_bounded_int(self.segment_index, "segment_index", 0, MAX_CHUNK_SEQUENCE)
        if type(self.text) is not str or len(self.text) > MAX_HYPOTHESIS_CHARS:
            raise SpeechProviderError("segment_text_invalid")
        _require_bounded_int(
            self.start_ms, "segment_start_ms", 0, MAX_STREAM_DURATION_MS
        )
        _require_bounded_int(self.end_ms, "segment_end_ms", 0, MAX_STREAM_DURATION_MS)
        if self.end_ms < self.start_ms:
            raise SpeechProviderError("segment_time_range_invalid")
        _require_token_times(self.token_times)
        if self.language is not None and not isinstance(
            self.language, LanguageConfidence
        ):
            raise SpeechProviderError("segment_language_type_invalid")

    @property
    def duration_ms(self) -> int:
        """Return the declared segment duration in milliseconds."""

        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        """Return a content-free summary of this segment."""

        return {
            "segment_index": self.segment_index,
            "text_length": len(self.text),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "token_count": len(self.token_times),
            "language": None if self.language is None else self.language.language,
        }


@dataclass(frozen=True, slots=True)
class RegisteredSpeechProvider:
    """A local engine paired with the entry point that builds it.

    Attributes:
        declaration: The engine declaration, which forbids network use.
        entrypoint: Dotted import path of the offline adapter factory.
    """

    declaration: SpeechProviderDeclaration
    entrypoint: str

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, SpeechProviderDeclaration):
            raise SpeechProviderError("registered_declaration_type_invalid")
        if (
            type(self.entrypoint) is not str
            or len(self.entrypoint) > MAX_ENTRYPOINT_CHARS
            or _ENTRYPOINT_RE.fullmatch(self.entrypoint) is None
        ):
            raise SpeechProviderError("registered_entrypoint_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return the registration without importing the adapter."""

        return {
            "provider_id": self.declaration.provider_id,
            "entrypoint": self.entrypoint,
            "declaration": self.declaration.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StreamingAsrSessionReport:
    """Deterministic, content-free state of one streaming session."""

    provider_id: str
    provider_revision: str
    state: str
    buffered_chunks: int
    buffered_byte_count: int
    buffered_duration_ms: int
    stream_duration_ms: int
    partial_hypotheses: int
    finalized_segments: int
    reason_code: str | None = None
    requires_network: bool = False
    schema_version: str = SPEECH_PROVIDER_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_revision": self.provider_revision,
            "state": self.state,
            "buffered_chunks": self.buffered_chunks,
            "buffered_byte_count": self.buffered_byte_count,
            "buffered_duration_ms": self.buffered_duration_ms,
            "stream_duration_ms": self.stream_duration_ms,
            "partial_hypotheses": self.partial_hypotheses,
            "finalized_segments": self.finalized_segments,
            "reason_code": self.reason_code,
            "requires_network": self.requires_network,
        }
        return {name: values[name] for name in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class StreamingAsrSession:
    """Drive one declared provider through a bounded, cancellable stream.

    The session owns no model and performs no I/O. It enforces the declared
    buffering limits, requires contiguous chunk sequences, keeps hypothesis
    revisions and segment indexes monotonic, and makes finalization and
    cancellation terminal.
    """

    def __init__(
        self,
        declaration: SpeechProviderDeclaration,
        limits: SpeechStreamLimits | None = None,
    ) -> None:
        if not isinstance(declaration, SpeechProviderDeclaration):
            raise SpeechProviderError("session_declaration_type_invalid")
        if limits is not None and not isinstance(limits, SpeechStreamLimits):
            raise SpeechProviderError("session_limits_type_invalid")
        self._declaration = declaration
        self._limits = declaration.limits if limits is None else limits
        self._state = StreamState.OPEN
        self._reason_code: str | None = None
        self._next_sequence = 0
        self._buffered_chunks = 0
        self._buffered_byte_count = 0
        self._buffered_duration_ms = 0
        self._stream_duration_ms = 0
        self._partial_hypotheses = 0
        self._finalized_segments = 0
        self._last_revision = -1
        self._last_segment_index = -1
        self._last_segment_end_ms = 0

    @property
    def declaration(self) -> SpeechProviderDeclaration:
        """Return the declaration this session runs against."""

        return self._declaration

    @property
    def limits(self) -> SpeechStreamLimits:
        """Return the effective buffering and emission limits."""

        return self._limits

    @property
    def state(self) -> StreamState:
        """Return the current lifecycle state."""

        return self._state

    @property
    def buffered_chunks(self) -> int:
        """Return the number of chunks held in the buffer."""

        return self._buffered_chunks

    @property
    def buffered_byte_count(self) -> int:
        """Return the number of payload bytes held in the buffer."""

        return self._buffered_byte_count

    @property
    def buffered_duration_ms(self) -> int:
        """Return the audio milliseconds held in the buffer."""

        return self._buffered_duration_ms

    @property
    def stream_duration_ms(self) -> int:
        """Return the audio milliseconds accepted so far."""

        return self._stream_duration_ms

    @property
    def partial_hypotheses(self) -> int:
        """Return the number of accepted partial hypotheses."""

        return self._partial_hypotheses

    @property
    def finalized_segments(self) -> int:
        """Return the number of accepted finalized segments."""

        return self._finalized_segments

    @property
    def reason_code(self) -> str | None:
        """Return the terminal reason code, if the session has one."""

        return self._reason_code

    def push_chunk(self, chunk: AudioChunk) -> int:
        """Buffer one audio chunk after enforcing sequence and limits.

        Args:
            chunk: The next chunk of the stream.

        Returns:
            The number of chunks currently buffered.

        Raises:
            SpeechProviderError: If the session is closed, the chunk is not
                typed, its sequence is not the expected one, or the buffered or
                stream limits would be exceeded.
        """

        self._require_open()
        if not isinstance(chunk, AudioChunk):
            raise SpeechProviderError("audio_chunk_type_invalid")
        if chunk.sequence != self._next_sequence:
            raise SpeechProviderError("chunk_sequence_out_of_order")
        if self._buffered_chunks + 1 > self._limits.max_buffered_chunks:
            raise SpeechProviderError("buffer_limit_exceeded")
        if (
            self._buffered_byte_count + chunk.byte_count
            > self._limits.max_buffered_byte_count
        ):
            raise SpeechProviderError("buffer_limit_exceeded")
        if (
            self._buffered_duration_ms + chunk.duration_ms
            > self._limits.max_buffered_duration_ms
        ):
            raise SpeechProviderError("buffer_limit_exceeded")
        if (
            self._stream_duration_ms + chunk.duration_ms
            > self._limits.max_stream_duration_ms
        ):
            raise SpeechProviderError("stream_duration_exceeded")
        self._next_sequence += 1
        self._buffered_chunks += 1
        self._buffered_byte_count += chunk.byte_count
        self._buffered_duration_ms += chunk.duration_ms
        self._stream_duration_ms += chunk.duration_ms
        return self._buffered_chunks

    def drain(self) -> int:
        """Release every buffered chunk and return how many were released.

        Draining is allowed in any state: it only drops buffered counters, so a
        long stream can keep its memory bounded while remaining open.
        """

        released = self._buffered_chunks
        self._buffered_chunks = 0
        self._buffered_byte_count = 0
        self._buffered_duration_ms = 0
        return released

    def submit_partial(self, hypothesis: PartialHypothesis) -> int:
        """Accept one partial hypothesis revision.

        Args:
            hypothesis: The next revision for the current utterance.

        Returns:
            The number of partial hypotheses accepted so far.

        Raises:
            SpeechProviderError: If the session is closed, the hypothesis is not
                typed, its revision is not newer, or the emission limit is hit.
        """

        self._require_open()
        if not isinstance(hypothesis, PartialHypothesis):
            raise SpeechProviderError("partial_hypothesis_type_invalid")
        if hypothesis.revision <= self._last_revision:
            raise SpeechProviderError("hypothesis_revision_out_of_order")
        if self._partial_hypotheses + 1 > self._limits.max_partial_hypotheses:
            raise SpeechProviderError("partial_hypothesis_limit_exceeded")
        self._last_revision = hypothesis.revision
        self._partial_hypotheses += 1
        return self._partial_hypotheses

    def submit_segment(self, segment: FinalizedSegment) -> int:
        """Accept one finalized segment in stream order.

        Args:
            segment: The next closed segment.

        Returns:
            The number of finalized segments accepted so far.

        Raises:
            SpeechProviderError: If the session is closed, the segment is not
                typed, its index or start time goes backwards, or the emission
                limit is hit.
        """

        self._require_open()
        if not isinstance(segment, FinalizedSegment):
            raise SpeechProviderError("finalized_segment_type_invalid")
        if segment.segment_index <= self._last_segment_index:
            raise SpeechProviderError("segment_index_out_of_order")
        if segment.start_ms < self._last_segment_end_ms:
            raise SpeechProviderError("segment_times_out_of_order")
        if self._finalized_segments + 1 > self._limits.max_segments:
            raise SpeechProviderError("segment_limit_exceeded")
        self._last_segment_index = segment.segment_index
        self._last_segment_end_ms = segment.end_ms
        self._finalized_segments += 1
        return self._finalized_segments

    def finalize(
        self, segment: FinalizedSegment | None = None
    ) -> StreamingAsrSessionReport:
        """Accept an optional last segment and close the session.

        Args:
            segment: Optional last segment to record before closing.

        Returns:
            The final, content-free session report.

        Raises:
            SpeechProviderError: If the session already ended, or the segment is
                rejected by :meth:`submit_segment`.
        """

        self._require_open()
        if segment is not None:
            self.submit_segment(segment)
        self._state = StreamState.FINALIZED
        return self.report()

    def cancel(self) -> StreamingAsrSessionReport:
        """Cancel the session and close it for further input.

        Returns:
            The final report, whose reason code is ``session_cancelled``.

        Raises:
            SpeechProviderError: If the session already ended.
        """

        self._require_open()
        self._state = StreamState.CANCELLED
        self._reason_code = "session_cancelled"
        return self.report()

    def report(self) -> StreamingAsrSessionReport:
        """Return a content-free report of the current session state."""

        return StreamingAsrSessionReport(
            provider_id=self._declaration.provider_id,
            provider_revision=self._declaration.provenance.provider_revision,
            state=self._state.value,
            buffered_chunks=self._buffered_chunks,
            buffered_byte_count=self._buffered_byte_count,
            buffered_duration_ms=self._buffered_duration_ms,
            stream_duration_ms=self._stream_duration_ms,
            partial_hypotheses=self._partial_hypotheses,
            finalized_segments=self._finalized_segments,
            reason_code=self._reason_code,
            requires_network=self._declaration.provenance.requires_network,
        )

    def _require_open(self) -> None:
        if self._state is StreamState.FINALIZED:
            raise SpeechProviderError("session_closed")
        if self._state is StreamState.CANCELLED:
            raise SpeechProviderError("session_cancelled")


def register_speech_providers(
    providers: Iterable[RegisteredSpeechProvider],
) -> tuple[RegisteredSpeechProvider, ...]:
    """Validate and order local provider registrations without importing them.

    Args:
        providers: Registrations to validate.

    Returns:
        The registrations sorted by provider identifier.

    Raises:
        SpeechProviderError: If the argument is not a sequence of registrations
            or two registrations share a provider identifier.
    """

    if isinstance(providers, (str, bytes, bytearray, dict)) or not isinstance(
        providers, Iterable
    ):
        raise SpeechProviderError("provider_registration_type_invalid")
    ordered: list[RegisteredSpeechProvider] = []
    seen: set[str] = set()
    for provider in providers:
        if not isinstance(provider, RegisteredSpeechProvider):
            raise SpeechProviderError("provider_registration_entry_invalid")
        identifier = provider.declaration.provider_id
        if identifier in seen:
            raise SpeechProviderError("duplicate_provider_id")
        seen.add(identifier)
        ordered.append(provider)
    return tuple(sorted(ordered, key=lambda item: item.declaration.provider_id))


def _require_bounded_int(value: Any, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int:
        raise SpeechProviderError(f"{name}_invalid")
    if not minimum <= value <= maximum:
        raise SpeechProviderError(f"{name}_out_of_range")


def _require_provider_id(value: Any) -> None:
    if type(value) is not str or _PROVIDER_ID_RE.fullmatch(value) is None:
        raise SpeechProviderError("provider_id_invalid")


def _require_language(value: Any) -> None:
    if type(value) is not str or _LANGUAGE_RE.fullmatch(value) is None:
        raise SpeechProviderError("provider_language_invalid")


def _require_sorted_languages(values: Any) -> None:
    if type(values) is not tuple:
        raise SpeechProviderError("provider_languages_invalid")
    if not values:
        raise SpeechProviderError("provider_languages_empty")
    if len(values) > MAX_PROVIDER_LANGUAGES:
        raise SpeechProviderError("provider_languages_out_of_range")
    for value in values:
        _require_language(value)
    if tuple(sorted(set(values))) != values:
        raise SpeechProviderError("provider_languages_unsorted")


def _require_sorted_ints(values: Any, name: str, minimum: int, maximum: int) -> None:
    if type(values) is not tuple:
        raise SpeechProviderError(f"{name}_invalid")
    if not values:
        raise SpeechProviderError(f"{name}_empty")
    for value in values:
        _require_bounded_int(value, name, minimum, maximum)
    if tuple(sorted(set(values))) != values:
        raise SpeechProviderError(f"{name}_unsorted")


def _require_token_times(values: Any) -> None:
    if type(values) is not tuple:
        raise SpeechProviderError("token_times_invalid")
    if len(values) > MAX_TOKEN_TIMES:
        raise SpeechProviderError("token_times_out_of_range")
    previous = -1
    for value in values:
        if not isinstance(value, TokenTime):
            raise SpeechProviderError("token_times_invalid")
        if value.token_index <= previous:
            raise SpeechProviderError("token_times_unsorted")
        previous = value.token_index


__all__ = [
    "MAX_BUFFERED_BYTE_COUNT",
    "MAX_BUFFERED_CHUNKS",
    "MAX_BUFFERED_DURATION_MS",
    "MAX_CHANNEL_COUNT",
    "MAX_CHUNK_BYTE_COUNT",
    "MAX_CHUNK_DURATION_MS",
    "MAX_CHUNK_SEQUENCE",
    "MAX_ENTRYPOINT_CHARS",
    "MAX_HYPOTHESIS_CHARS",
    "MAX_PARTIAL_HYPOTHESES",
    "MAX_PROVIDER_LANGUAGES",
    "MAX_PROVIDER_REVISION_CHARS",
    "MAX_SAMPLE_RATE_HZ",
    "MAX_SEGMENTS",
    "MAX_STREAM_DURATION_MS",
    "MAX_TOKEN_TIMES",
    "SPEECH_PROVIDER_CONTRACT_VERSION",
    "SPEECH_PROVIDER_REASON_CODES",
    "AudioChunk",
    "FinalizedSegment",
    "LanguageConfidence",
    "PartialHypothesis",
    "ProviderProvenance",
    "RegisteredSpeechProvider",
    "SpeechProviderDeclaration",
    "SpeechProviderError",
    "SpeechStreamLimits",
    "StreamState",
    "StreamingAsrSession",
    "StreamingAsrSessionReport",
    "TokenTime",
    "register_speech_providers",
]
