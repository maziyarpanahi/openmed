"""Owned, cancellable local runtime composition for images, PDFs and WAV streams."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, BinaryIO, Protocol, cast

from .abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from .asset_limits import DESKTOP_V1, LimitProfile
from .asset_manifest import AssetManifest
from .base import ExtractedDocument, SourceSpan
from .image_header import ImageHeaderError, read_image_header
from .pdf_geometry import PdfGeometryStatus, read_pdf_geometry
from .preflight import PreflightStatus, preflight_asset
from .provider_result import ProviderResultEnvelope, ProviderResultOutcome
from .streaming.speech.providers import (
    AudioChunk,
    FinalizedSegment,
    PartialHypothesis,
    SpeechProviderDeclaration,
    SpeechProviderError,
    StreamingAsrSession,
    StreamState,
)
from .vision.providers.base import (
    EdgeVLMProvider,
    ImageInput,
    LocalProviderMetadata,
    Prompt,
    VLMRequest,
    VLMResult,
    run_local_vlm,
)
from .wav_metadata import WavMetadataError, read_wav_metadata
from .webp_dimensions import WebpDimensionsError, read_webp_dimensions

SESSION_SCHEMA_VERSION = "openmed.multimodal.session.v1"
SpeechEmission = PartialHypothesis | FinalizedSegment
ProtectedOutput = ExtractedDocument | VLMResult | tuple[SpeechEmission, ...]


class RuntimeSessionError(ValueError):
    """Value-free caller-contract or lifecycle error."""


class SessionState(str, Enum):
    """Cancellation remains pending until active work and owned cleanup finish."""

    OPEN = "open"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class SessionOperation(str, Enum):
    """Supported runtime operations; decoding stays with injected local adapters."""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    FINISH_AUDIO = "finish_audio"


class SessionOutcome(str, Enum):
    """Terminal result of one admitted operation."""

    SUCCESS = "success"
    ABSTAINED = "abstained"
    CANCELLED = "cancelled"


def _require(condition: bool) -> None:
    if not condition:
        raise RuntimeSessionError("invalid runtime session contract")


def _bounded_metadata(
    value: Any, budget: list[int], max_items: int, depth: int = 0
) -> Any:
    """Copy a closed, bounded metadata tree into the protected document result."""
    budget[0] -= 1
    _require(budget[0] >= 0 and depth <= 8)
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        budget[1] -= len(value)
        _require(budget[1] >= 0)
        return value
    if type(value) is int:
        _require(value.bit_length() <= 64)
        return value
    if type(value) is float:
        _require(math.isfinite(value))
        return value
    _require(type(value) in (dict, list, tuple) and len(value) <= max_items)
    if type(value) is dict:
        _require(all(type(key) is str for key in value))
        return {
            _bounded_metadata(key, budget, max_items, depth + 1): _bounded_metadata(
                item, budget, max_items, depth + 1
            )
            for key, item in value.items()
        }
    return [_bounded_metadata(item, budget, max_items, depth + 1) for item in value]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionLimits:
    """Bounded input snapshots, output payloads and operation history.

    Args:
        max_asset_bytes: Maximum encoded bytes held for one asset, at most 32 MiB.
        max_total_input_bytes: Cumulative declared input across admitted attempts.
        max_operations: Maximum number of admitted operations, including failures.
        max_output_chars: Maximum protected text per operation, including document
            metadata keys and string values.
        max_output_items: Maximum source spans or speech emissions per operation.
        pdf_dpi: Declared maximum raster resolution for the document adapter.
        preflight: Existing resource-limit profile, applied before execution.
    """

    max_asset_bytes: int = 32 * 1024 * 1024
    max_total_input_bytes: int = 1024 * 1024 * 1024
    max_operations: int = 1024
    max_output_chars: int = 65_536
    max_output_items: int = 256
    pdf_dpi: int = 150
    preflight: LimitProfile = DESKTOP_V1

    def __post_init__(self) -> None:
        for value, ceiling in (
            (self.max_asset_bytes, 32 * 1024 * 1024),
            (self.max_total_input_bytes, 1024 * 1024 * 1024),
            (self.max_operations, 4096),
            (self.max_output_chars, 65_536),
            (self.max_output_items, 4096),
            (self.pdf_dpi, 600),
        ):
            _require(type(value) is int and 1 <= value <= ceiling)
        _require(type(self.preflight) is LimitProfile)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssetHandle:
    """One explicit local source lease, transferred for the duration of a call.

    Args:
        manifest: Declared immutable metadata and source digest.
        source: Readable local binary handle, excluded from representations.
        owned: Close the handle on every return or error when True. Borrowed
            seekable handles are restored; borrowed nonseekable ones are consumed.
    """

    manifest: AssetManifest = field(repr=False)
    source: BinaryIO = field(repr=False)
    owned: bool = False

    def __post_init__(self) -> None:
        _require(type(self.manifest) is AssetManifest)
        _require(
            type(self.owned) is bool and callable(getattr(self.source, "read", None))
        )
        _require(not self.owned or callable(getattr(self.source, "close", None)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderLease:
    """Injected local provider with explicit lifetime and cancellation hooks.

    Args:
        provider: Image, document or speech adapter, hidden from representations.
        owned: Whether the runtime owns the provider resource until session end.
        release: Required for owned providers; invoked exactly once at cleanup.
        cancel: Optional cooperative, nonblocking cancellation hook for this lease.
            It is also usable on a borrowed provider when explicitly supplied.

    Reuse the same lease object to share a resource across operation types.
    Providers and hooks are trusted local code; declarations are not a sandbox.
    """

    provider: Any = field(repr=False)
    owned: bool = False
    release: Callable[[], None] | None = field(default=None, repr=False)
    cancel: Callable[[], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require(type(self.owned) is bool)
        _require(
            (self.owned and callable(self.release))
            or (not self.owned and self.release is None)
        )
        _require(self.cancel is None or callable(self.cancel))


class DocumentProvider(Protocol):
    """Local extraction adapter; obey the admitted raster and output ceilings."""

    @property
    def metadata(self) -> LocalProviderMetadata:
        """Pinned offline declaration, available without decoding or inference."""
        ...

    def extract(
        self, data: bytes, *, limits: SessionLimits, cancellation: threading.Event
    ) -> ExtractedDocument:
        """Extract a PDF snapshot with bounded allocation and protected output.

        Args:
            data: Immutable preflighted PDF bytes; never log or retain them.
            limits: Enforce the page, pixel, DPI and output ceilings during actual
                decoding. Header geometry is admission evidence, not a decoder.
            cancellation: Cooperative stop signal; release temporary resources
                on every return or exception.
        """
        ...


class SpeechExecutionProvider(Protocol):
    """Execution seam composed with the existing streaming-ASR state contract."""

    @property
    def declaration(self) -> SpeechProviderDeclaration:
        """Return the existing offline speech declaration without input I/O."""
        ...

    def push(
        self, chunk: AudioChunk, *, cancellation: threading.Event
    ) -> tuple[SpeechEmission, ...]:
        """Process one WAV chunk; retain only declaration-bounded stream state.

        The chunk payload must be released before returning. Reusable decoder
        state is owned by this stream until ``finish`` or ``abort``.
        """
        ...

    def finish(self, *, cancellation: threading.Event) -> tuple[SpeechEmission, ...]:
        """Flush the stream, release adapter buffers and return final emissions."""
        ...

    def abort(self) -> None:
        """Release stream buffers after failure or cancellation without output.

        Do not close a shared provider resource. This hook must be idempotent
        and safe alongside a cooperative lease cancellation hook.
        """
        ...


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One ordered metadata event; no source identifiers or protected values."""

    sequence: int
    operation: SessionOperation
    outcome: SessionOutcome
    input_digest: str | None
    input_bytes: int
    pdf_geometry_resolved: bool
    abstention: AbstentionRecord | None
    result: ProviderResultEnvelope | None
    cleanup_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return controlled lifecycle, provenance and aggregate evidence only."""
        return {
            "sequence": self.sequence,
            "operation": self.operation.value,
            "outcome": self.outcome.value,
            "input_digest": self.input_digest,
            "input_bytes": self.input_bytes,
            "pdf_geometry_resolved": self.pdf_geometry_resolved,
            "abstention": self.abstention.to_dict() if self.abstention else None,
            "result": self.result.to_dict() if self.result else None,
            "cleanup_failed": self.cleanup_failed,
        }


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Metadata plus a protected, explicitly reviewed application result."""

    event: SessionEvent
    _output: ProtectedOutput | None = field(default=None, repr=False)

    def reviewed_output(self, *, reviewer_confirmed: bool) -> ProtectedOutput:
        """Return protected output only after explicit confirmation of local review.

        This is not PHI de-identification, disclosure authorization or a clinical
        validation gate. Feed speech outputs into the applicable privacy gates.
        """
        _require(
            reviewer_confirmed is True and self.event.outcome is SessionOutcome.SUCCESS
        )
        _require(self._output is not None)
        return cast(ProtectedOutput, self._output)

    def to_json(self) -> str:
        """Serialize only the content-free event, never the protected output."""
        return json.dumps(self.event.to_dict(), sort_keys=True, separators=(",", ":"))


class _Abstain(Exception):
    def __init__(self, stage: AbstentionStage, reason: AbstentionReason) -> None:
        self.record = AbstentionRecord(stage=stage, reason=reason)


class _Cancelled(Exception):
    pass


class MultimodalSession:
    """Compose local preflight and providers with one bounded in-flight operation.

    Args:
        image: Optional lease wrapping an existing EdgeVLMProvider.
        document: Optional lease wrapping a DocumentProvider.
        speech: Optional lease wrapping a SpeechExecutionProvider.
        limits: Input, output, event and preflight ceilings.
        clock: Trusted monotonic clock; inject a deterministic clock in tests.

    Operations are synchronous and reject concurrent calls instead of queuing.
    Another thread may cancel. Pending work may unwind cooperatively, but can
    never publish success after cancellation wins the result-commit lock.
    Terminal state is reached only after active work and cleanup finish.
    """

    def __init__(
        self,
        *,
        image: ProviderLease | None = None,
        document: ProviderLease | None = None,
        speech: ProviderLease | None = None,
        limits: SessionLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits if limits is not None else SessionLimits()
        _require(type(self._limits) is SessionLimits and callable(clock))
        self._providers: dict[SessionOperation, ProviderLease] = {}
        identities: dict[int, ProviderLease] = {}
        for operation, lease in (
            (SessionOperation.IMAGE, image),
            (SessionOperation.DOCUMENT, document),
            (SessionOperation.AUDIO, speech),
        ):
            if lease is not None:
                _require(type(lease) is ProviderLease)
                previous = identities.get(id(lease.provider))
                _require(previous is None or previous is lease)
                identities[id(lease.provider)] = lease
                self._providers[operation] = lease
        self._clock = clock
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._state = SessionState.OPEN
        self._target = SessionState.CANCELLED
        self._active = False
        self._closing = False
        self._cancel_hooks_active = False
        self._events: list[SessionEvent] = []
        self._input_bytes = 0
        self._cleanup_failures = 0
        self._speech: StreamingAsrSession | None = None
        self._speech_started = False
        self._speech_terminated = False
        self._audio_digest = hashlib.sha256()

    def __enter__(self) -> MultimodalSession:
        """Enter an open session; context exit always releases owned leases."""
        _require(self.state is SessionState.OPEN)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def state(self) -> SessionState:
        """Current lifecycle state, including honest pending cancellation."""
        with self._lock:
            return self._state

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """Immutable ordered events; protected inputs and outputs are not retained."""
        with self._lock:
            return tuple(self._events)

    def to_json(self) -> str:
        """Serialize bounded state and aggregate evidence, never provider objects."""
        with self._lock:
            return json.dumps(
                {
                    "schema_version": SESSION_SCHEMA_VERSION,
                    "state": self._state.value,
                    "input_bytes": self._input_bytes,
                    "cleanup_failures": self._cleanup_failures,
                    "events": [event.to_dict() for event in self._events],
                    "requires_network": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )

    def infer_image(self, asset: AssetHandle, prompt: Prompt) -> SessionResult:
        """Preflight an image snapshot, then execute the existing local VLM contract."""

        def execute(data: bytes) -> tuple[ProtectedOutput, str]:
            _require(type(prompt) is Prompt)
            manifest = asset.manifest
            provider = cast(EdgeVLMProvider, self._provider(SessionOperation.IMAGE))
            request = VLMRequest(
                image=ImageInput(
                    data=data,
                    media_type=manifest.media_type,
                    width=cast(int, manifest.width),
                    height=cast(int, manifest.height),
                ),
                prompt=prompt,
            )
            result = run_local_vlm(provider, request)
            if result.abstention is not None:
                raise _Abstain(result.abstention.stage, result.abstention.reason)
            return result, result.runtime.provider.model_digest

        return self._run(SessionOperation.IMAGE, asset, execute)

    def extract_document(self, asset: AssetHandle) -> SessionResult:
        """Resolve bounded PDF geometry and delegate extraction to a local adapter."""

        def execute(data: bytes) -> tuple[ProtectedOutput, str]:
            provider = cast(DocumentProvider, self._provider(SessionOperation.DOCUMENT))
            metadata = provider.metadata
            _require(type(metadata) is LocalProviderMetadata)
            metadata.__post_init__()
            result = provider.extract(
                data, limits=self._limits, cancellation=self._cancelled
            )
            _require(type(result) is ExtractedDocument and type(result.text) is str)
            _require(len(result.text) <= self._limits.max_output_chars)
            result.text.encode("utf-8")
            _require(
                type(result.spans) is tuple
                and len(result.spans) <= self._limits.max_output_items
            )
            budget = [4096, self._limits.max_output_chars - len(result.text)]
            _require(type(result.metadata) is dict)
            document_metadata = _bounded_metadata(
                result.metadata, budget, self._limits.max_output_items
            )
            spans = []
            for span in result.spans:
                _require(
                    type(span) is SourceSpan
                    and type(span.start) is int
                    and type(span.end) is int
                )
                _require(0 <= span.start <= span.end <= len(result.text))
                _require(
                    type(span.page) is int
                    and 0 <= span.page < cast(int, asset.manifest.pages)
                )
                _require(type(span.metadata) is dict)
                if span.bbox is not None:
                    _require(type(span.bbox) is tuple and len(span.bbox) == 4)
                    _require(
                        all(
                            type(value) in (int, float)
                            and math.isfinite(value)
                            and abs(value) <= 1_000_000_000
                            for value in span.bbox
                        )
                    )
                    _require(
                        span.bbox[0] <= span.bbox[2] and span.bbox[1] <= span.bbox[3]
                    )
                spans.append(
                    SourceSpan(
                        span.start,
                        span.end,
                        span.page,
                        span.bbox,
                        _bounded_metadata(
                            span.metadata, budget, self._limits.max_output_items
                        ),
                    )
                )
            return ExtractedDocument(
                result.text, tuple(spans), document_metadata
            ), metadata.model_digest

        return self._run(SessionOperation.DOCUMENT, asset, execute)

    def push_audio(self, asset: AssetHandle, *, sequence: int) -> SessionResult:
        """Admit one WAV chunk and validate ordered emissions with the ASR contract."""

        def execute(data: bytes) -> tuple[ProtectedOutput, str]:
            provider = cast(
                SpeechExecutionProvider, self._provider(SessionOperation.AUDIO)
            )
            if self._speech is None:
                _require(callable(getattr(provider, "abort", None)))
                declaration = provider.declaration
                _require(type(declaration) is SpeechProviderDeclaration)
                declaration.__post_init__()
                declaration.provenance.__post_init__()
                self._speech = StreamingAsrSession(declaration)
            metadata = read_wav_metadata(data)
            chunk = AudioChunk(
                sequence=sequence,
                payload=data,
                sample_rate_hz=metadata.sample_rate_hz,
                channel_count=metadata.channels,
                duration_ms=math.ceil(metadata.duration_seconds * 1000),
            )
            self._speech.push_chunk(chunk)
            self._audio_digest.update(bytes.fromhex(asset.manifest.sha256))
            try:
                self._speech_started = True
                emissions = provider.push(chunk, cancellation=self._cancelled)
                self._accept_speech(emissions)
            except Exception:
                self._abort_speech()
                raise
            finally:
                self._speech.drain()
            return (
                emissions,
                self._speech.declaration.provenance.model_fingerprint.removeprefix(
                    "sha256:"
                ),
            )

        return self._run(SessionOperation.AUDIO, asset, execute)

    def finish_audio(self) -> SessionResult:
        """Flush an active speech stream without silently reopening finalized audio."""

        def execute(_: bytes) -> tuple[ProtectedOutput, str]:
            _require(
                self._speech is not None and self._speech.state is StreamState.OPEN
            )
            assert self._speech is not None
            provider = cast(
                SpeechExecutionProvider, self._provider(SessionOperation.AUDIO)
            )
            try:
                emissions = provider.finish(cancellation=self._cancelled)
                self._accept_speech(emissions)
                self._speech.finalize()
                self._speech_terminated = True
            except Exception:
                self._abort_speech()
                raise
            finally:
                self._speech.drain()
            return (
                emissions,
                self._speech.declaration.provenance.model_fingerprint.removeprefix(
                    "sha256:"
                ),
            )

        return self._run(SessionOperation.FINISH_AUDIO, None, execute)

    def _provider(self, operation: SessionOperation) -> Any:
        self._check_cancel()
        lease = self._providers.get(operation)
        if lease is None:
            raise _Abstain(
                AbstentionStage.PREFLIGHT, AbstentionReason.PROVIDER_UNAVAILABLE
            )
        return lease.provider

    def _check_cancel(self) -> None:
        if self._cancelled.is_set():
            raise _Cancelled

    def _accept_speech(self, emissions: tuple[SpeechEmission, ...]) -> None:
        assert self._speech is not None
        _require(
            type(emissions) is tuple and len(emissions) <= self._limits.max_output_items
        )
        _require(
            all(
                type(item) in (PartialHypothesis, FinalizedSegment)
                for item in emissions
            )
        )
        _require(
            sum(len(item.text) for item in emissions) <= self._limits.max_output_chars
        )
        _require(
            sum(len(item.token_times) for item in emissions)
            <= self._limits.max_output_items
        )
        for item in emissions:
            item.__post_init__()
            if type(item) is PartialHypothesis:
                self._speech.submit_partial(item)
            else:
                self._speech.submit_segment(cast(FinalizedSegment, item))

    def _abort_speech(self) -> None:
        if self._speech is not None:
            if self._speech.state is StreamState.OPEN:
                self._speech.cancel()
            self._speech.drain()
        if self._speech_started and not self._speech_terminated:
            self._speech_terminated = True
            lease = self._providers.get(SessionOperation.AUDIO)
            if lease is not None:
                try:
                    cast(SpeechExecutionProvider, lease.provider).abort()
                except Exception:
                    with self._lock:
                        self._cleanup_failures += 1

    def _snapshot(self, asset: AssetHandle, operation: SessionOperation) -> bytes:
        manifest = asset.manifest
        allowed = {
            SessionOperation.IMAGE: ("image/png", "image/jpeg", "image/webp"),
            SessionOperation.DOCUMENT: ("application/pdf",),
            SessionOperation.AUDIO: ("audio/wav",),
        }
        if manifest.media_type not in allowed[operation]:
            raise _Abstain(
                AbstentionStage.PREFLIGHT, AbstentionReason.UNSUPPORTED_MEDIA
            )
        if manifest.byte_size > min(
            self._limits.max_asset_bytes, self._limits.preflight.max_byte_size
        ):
            raise _Abstain(AbstentionStage.PREFLIGHT, AbstentionReason.RESOURCE_LIMIT)
        if self._input_bytes + manifest.byte_size > self._limits.max_total_input_bytes:
            raise _Abstain(AbstentionStage.PREFLIGHT, AbstentionReason.RESOURCE_LIMIT)
        with self._lock:
            self._input_bytes += manifest.byte_size
        source = asset.source
        position = None
        if not asset.owned:
            try:
                position = source.tell()
            except (AttributeError, OSError):
                pass
        buffer = bytearray()
        try:
            while len(buffer) <= manifest.byte_size:
                self._check_cancel()
                requested = min(65_536, manifest.byte_size + 1 - len(buffer))
                chunk = source.read(requested)
                _require(type(chunk) is bytes and len(chunk) <= requested)
                if not chunk:
                    break
                buffer.extend(chunk)
        finally:
            if position is not None:
                source.seek(position)
        return bytes(buffer)

    def _preflight(
        self, asset: AssetHandle, data: bytes, operation: SessionOperation
    ) -> bool:
        manifest = asset.manifest
        report = preflight_asset(manifest, data, limit_profile=self._limits.preflight)
        resolved = False
        if operation is SessionOperation.DOCUMENT:
            expected = {
                ("limits", "insufficient_metadata", "pixels"),
                ("limits", "insufficient_metadata", "total_pixels"),
            }
            findings = {(f.check, f.reason_code, f.field_name) for f in report.findings}
            if findings != expected:
                raise _Abstain(
                    AbstentionStage.PREFLIGHT,
                    report.abstention.reason
                    if report.abstention
                    else AbstentionReason.UNSUPPORTED_MEDIA,
                )
            geometry = read_pdf_geometry(
                data,
                max_bytes=self._limits.max_asset_bytes,
                max_pages=self._limits.preflight.max_pages,
                max_objects=25_000,
                max_decompressed_bytes=self._limits.max_asset_bytes,
            )
            if (
                geometry.status is not PdfGeometryStatus.READABLE
                or geometry.page_count != manifest.pages
            ):
                raise _Abstain(AbstentionStage.DECODE, AbstentionReason.MALFORMED_MEDIA)
            pixels = [
                math.ceil(cast(float, page.width) * self._limits.pdf_dpi / 72)
                * math.ceil(cast(float, page.height) * self._limits.pdf_dpi / 72)
                for page in geometry.pages
            ]
            if (
                not pixels
                or max(pixels) > self._limits.preflight.max_pixels
                or sum(pixels) > self._limits.preflight.max_total_pixels
            ):
                raise _Abstain(
                    AbstentionStage.PREFLIGHT, AbstentionReason.RESOURCE_LIMIT
                )
            resolved = True
        elif report.status is not PreflightStatus.ACCEPT:
            assert report.abstention is not None
            raise _Abstain(report.abstention.stage, report.abstention.reason)
        if operation is SessionOperation.IMAGE:
            header = (
                read_webp_dimensions(data, max_pixels=self._limits.preflight.max_pixels)
                if manifest.media_type == "image/webp"
                else read_image_header(
                    data, max_pixels=self._limits.preflight.max_pixels
                )
            )
            if (header.width, header.height) != (manifest.width, manifest.height):
                raise _Abstain(AbstentionStage.DECODE, AbstentionReason.MALFORMED_MEDIA)
        if operation is SessionOperation.AUDIO:
            metadata = read_wav_metadata(data)
            if (
                struct.unpack_from("<I", data, 4)[0] + 8 != len(data)
                or manifest.duration_seconds is None
                or abs(metadata.duration_seconds - manifest.duration_seconds)
                > 1 / metadata.sample_rate_hz
            ):
                raise _Abstain(AbstentionStage.DECODE, AbstentionReason.MALFORMED_MEDIA)
        return resolved

    def _release_asset(self, asset: AssetHandle | None) -> bool:
        if asset is not None and asset.owned:
            try:
                asset.source.close()
            except Exception:
                with self._lock:
                    self._cleanup_failures += 1
                return False
        return True

    def _run(
        self,
        operation: SessionOperation,
        asset: AssetHandle | None,
        execute: Callable[[bytes], tuple[ProtectedOutput, str]],
    ) -> SessionResult:
        _require(asset is None or type(asset) is AssetHandle)
        with self._lock:
            admitted = (
                self._state is SessionState.OPEN
                and not self._active
                and len(self._events) < self._limits.max_operations
            )
            if admitted:
                self._active = True
                sequence = len(self._events)
        if not admitted:
            self._release_asset(asset)
            raise RuntimeSessionError("session is closed, busy or exhausted")
        data = b""
        output = None
        envelope = None
        abstention = None
        resolved = False
        input_digest = (
            asset.manifest.sha256 if asset else self._audio_digest.hexdigest()
        )
        try:
            self._check_cancel()
            if asset is not None:
                data = self._snapshot(asset, operation)
                resolved = self._preflight(asset, data, operation)
            self._check_cancel()
            started = self._clock()
            output, model_digest = execute(data)
            duration = (self._clock() - started) * 1000
            if isinstance(output, (ExtractedDocument, VLMResult)):
                _require(len(output.text) <= self._limits.max_output_chars)
                if isinstance(output, ExtractedDocument):
                    payload = {
                        "text": output.text,
                        "metadata": output.metadata,
                        "spans": [
                            [s.start, s.end, s.page, s.bbox, s.metadata]
                            for s in output.spans
                        ],
                    }
                    digest = hashlib.sha256(
                        json.dumps(
                            payload, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                else:
                    digest = hashlib.sha256(output.text.encode()).hexdigest()
                count = 1
            else:
                digest = hashlib.sha256(
                    json.dumps(
                        [
                            [
                                type(item).__name__,
                                item.to_dict(),
                                hashlib.sha256(item.text.encode()).hexdigest(),
                            ]
                            for item in output
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                count = len(output)
            envelope = ProviderResultEnvelope(
                provider_id="runtime." + operation.value,
                model_id="sha256." + model_digest,
                input_digest=input_digest,
                outcome=ProviderResultOutcome.SUCCESS,
                output_digest=digest,
                duration_ms=duration,
                count_metadata={"input_bytes": len(data), "output_items": count},
            )
        except _Cancelled:
            pass
        except _Abstain as stopped:
            abstention = stopped.record
        except (ImageHeaderError, WebpDimensionsError, WavMetadataError):
            abstention = AbstentionRecord(
                stage=AbstentionStage.DECODE, reason=AbstentionReason.MALFORMED_MEDIA
            )
        except SpeechProviderError as stopped:
            reason = AbstentionReason.PROVIDER_UNAVAILABLE
            if (
                "limit" in stopped.category
                or "duration_exceeded" in stopped.category
                or "payload_out_of_range" in stopped.category
            ):
                reason = AbstentionReason.RESOURCE_LIMIT
            elif (
                "order" in stopped.category
                or "times" in stopped.category
                or "session_closed" in stopped.category
                or "session_cancelled" in stopped.category
            ):
                reason = AbstentionReason.TEMPORAL_INSTABILITY
            abstention = AbstentionRecord(
                stage=AbstentionStage.INFERENCE, reason=reason
            )
        except Exception:
            abstention = AbstentionRecord(
                stage=AbstentionStage.INFERENCE,
                reason=AbstentionReason.PROVIDER_UNAVAILABLE,
            )
        finally:
            input_bytes = len(data)
            data = b""
            cleanup_ok = self._release_asset(asset)
            if operation in (SessionOperation.AUDIO, SessionOperation.FINISH_AUDIO):
                if (
                    self._cancelled.is_set()
                    or not cleanup_ok
                    or (abstention is not None or envelope is None)
                ):
                    # A provider may have retained stream state even when its
                    # output or the final result envelope fails validation.
                    # A rejected preflight/sequence alone keeps a healthy stream.
                    if (
                        envelope is not None
                        or output is not None
                        or self._cancelled.is_set()
                    ):
                        self._abort_speech()
            with self._lock:
                if self._cancelled.is_set():
                    outcome = SessionOutcome.CANCELLED
                    abstention = None
                elif not cleanup_ok or abstention is not None or envelope is None:
                    outcome = SessionOutcome.ABSTAINED
                    if abstention is None:
                        abstention = AbstentionRecord(
                            stage=AbstentionStage.INFERENCE,
                            reason=AbstentionReason.PROVIDER_UNAVAILABLE,
                        )
                else:
                    outcome = SessionOutcome.SUCCESS
                if outcome is not SessionOutcome.SUCCESS:
                    envelope, output = None, None
                event = SessionEvent(
                    sequence,
                    operation,
                    outcome,
                    input_digest,
                    input_bytes,
                    resolved,
                    abstention,
                    envelope,
                    not cleanup_ok,
                )
                self._events.append(event)
                self._active = False
            self._complete_termination()
        return SessionResult(event, output)

    def cancel(self) -> SessionState:
        """Request cooperative cancellation and suppress every later successful commit."""
        return self._terminate(SessionState.CANCELLED)

    def close(self) -> SessionState:
        """Close owned leases; active calls are cancelled and remain pending until cleanup."""
        return self._terminate(SessionState.CLOSED)

    def _terminate(self, target: SessionState) -> SessionState:
        hooks: tuple[ProviderLease, ...] = ()
        owns_hook_round = False
        with self._lock:
            if self._state is SessionState.OPEN:
                self._state, self._target = SessionState.CANCELLING, target
                self._cancelled.set()
                if self._active:
                    hooks = tuple(
                        {
                            id(lease): lease for lease in self._providers.values()
                        }.values()
                    )
                    self._cancel_hooks_active = True
                    owns_hook_round = True
        try:
            for lease in hooks:
                if lease.cancel is not None:
                    try:
                        lease.cancel()
                    except Exception:
                        with self._lock:
                            self._cleanup_failures += 1
        finally:
            if owns_hook_round:
                with self._lock:
                    self._cancel_hooks_active = False
            self._complete_termination()
        return self.state

    def _complete_termination(self) -> None:
        with self._lock:
            if (
                self._state is not SessionState.CANCELLING
                or self._active
                or self._closing
                or self._cancel_hooks_active
            ):
                return
            self._closing = True
            leases = tuple(
                {id(lease): lease for lease in self._providers.values()}.values()
            )
        try:
            self._abort_speech()
            self._speech = None
            self._providers.clear()
            for lease in leases:
                if lease.owned:
                    try:
                        assert lease.release is not None
                        lease.release()
                    except Exception:
                        with self._lock:
                            self._cleanup_failures += 1
        finally:
            with self._lock:
                self._closing = False
                self._state = self._target
