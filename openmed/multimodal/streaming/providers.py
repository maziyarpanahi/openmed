"""A bounded offline Whisper stream with private revisions and screened release.

The caller supplies local PHI screening, voice-activity evidence and fresh host
resource observations. No detector accuracy, speaker identity, or clinical
meaning is inferred from an ASR confidence score.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openmed.mlx.models.whisper import WhisperAssets, WhisperProcess
from openmed.multimodal.asr_hallucination import (
    AsrHypothesisSegment,
    AsrSegmentStatistics,
    HallucinationOutcome,
    VoiceActivityTimeline,
    check_asr_hallucinations,
)
from openmed.multimodal.runtime.thermal_gate import (
    MemoryPressure,
    PlatformObservation,
    ThermalState,
)
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    BufferState,
    LocalPhiScreener,
    ReleasedSpeechSegment,
    WithholdingBuffer,
    WithholdingPolicy,
)
from openmed.multimodal.streaming.speech.providers import (
    AudioChunk,
    FinalizedSegment,
    LanguageConfidence,
    PartialHypothesis,
    ProviderProvenance,
    SpeechProviderDeclaration,
    SpeechStreamLimits,
    StreamingAsrSession,
    TokenTime,
)
from openmed.multimodal.streaming.transcript.revision_ledger import TranscriptToken


class OfflineAsrError(ValueError):
    """A controlled failure which never includes audio or recognized text."""


def _require(condition: bool) -> None:
    if not condition:
        raise OfflineAsrError("invalid_offline_asr_input")


@dataclass(frozen=True, slots=True, kw_only=True)
class OfflineAsrPolicy:
    """Hard resource bounds for one serial stream; timings are milliseconds.

    The audio retention deadline starts with the first chunk of each utterance,
    never resets on a partial revision, and is enforced by a daemon timer.
    All calls except cancellation are serial. Cancellation may run concurrently.
    """

    max_window_ms: int = 15_000
    max_session_ms: int = 300_000
    audio_retention_ms: int = 60_000
    decode_timeout_ms: int = 30_000
    max_operations: int = 512
    min_language_confidence_ppm: int = 800_000
    min_growth_ms: int = 200
    min_available_bytes: int = 3 * 1024**3

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("max_window_ms", 1000, 15_000),
            ("max_session_ms", 1000, 3_600_000),
            ("audio_retention_ms", 1000, 60_000),
            ("decode_timeout_ms", 100, 60_000),
            ("max_operations", 1, 4096),
            ("min_language_confidence_ppm", 1, 1_000_000),
            ("min_growth_ms", 1, 1000),
            ("min_available_bytes", 1, 1 << 50),
        ):
            value = getattr(self, name)
            _require(type(value) is int and lower <= value <= upper)


@dataclass(frozen=True, slots=True)
class ScreenedAsrSegment:
    """Final screened output with source timings and no inferred speaker identity.

    Timing tokens contain no raw recognized words. Character mappings and
    lineage in ``screened`` refer to the original recognized text. Review is
    required before consuming the protected output.
    """

    screened: ReleasedSpeechSegment = field(repr=False)
    language: LanguageConfidence
    token_times: tuple[TokenTime, ...]
    speaker: str = "unknown"

    def reviewed_text(self, *, reviewer_confirmed: bool = False) -> str:
        """Return redacted text only after explicit caller review confirmation."""
        if reviewer_confirmed is not True:
            raise OfflineAsrError("review_required")
        return self.screened.text

    def to_dict(self) -> dict[str, Any]:
        """Return offsets, provenance and confidence without transcript content."""
        return {
            "screened": self.screened.to_dict(),
            "language": self.language.to_dict(),
            "token_times": [item.to_dict() for item in self.token_times],
            "speaker": self.speaker,
            "review_required": True,
        }


@dataclass(frozen=True, slots=True)
class AsrUpdate:
    """Content-free partial status plus any independently screened final prefix."""

    reason: str
    revision: int
    hypothesis_characters: int = 0
    language: LanguageConfidence | None = None
    released: tuple[ScreenedAsrSegment, ...] = field(default=(), repr=False)


class OfflineWhisperSession:
    """Transcribe bounded PCM windows using a dedicated local MLX process.

    Args:
        assets: Existing caller-managed local weights and revision/license snapshot.
        screener: Trusted local PHI screener qualified for all declared languages.
        languages: Explicit screened language coverage, sorted and unique.
        observe: Fresh local thermal, memory-pressure and headroom measurements.
            Missing, stale, serious-thermal or pressured readings stop the stream.
        policy: Hard audio, decode, operation and session ceilings.
        withholding_policy: Privacy/context policy; trailing context is retained
            across utterances until further input or explicit end of stream.
        clock: Monotonic seconds for host sensor integration and deterministic tests.

    Input is mono signed-16-bit little-endian PCM at 16 kHz. AudioChunk remains
    caller-owned. This object overwrites only its own mutable window and kills
    its worker on every terminal path. It writes no audio files. Use a context
    manager or close explicitly. This is Python/Apple Silicon only, not a Swift
    ASR implementation or a clinical transcription accuracy claim.
    """

    def __init__(
        self,
        assets: WhisperAssets,
        screener: LocalPhiScreener,
        *,
        languages: tuple[str, ...],
        observe: Callable[[], PlatformObservation],
        policy: OfflineAsrPolicy | None = None,
        withholding_policy: WithholdingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy or OfflineAsrPolicy()
        _require(
            type(self._policy) is OfflineAsrPolicy
            and callable(observe)
            and callable(clock)
        )
        self._policy.__post_init__()
        _require(type(assets) is WhisperAssets)
        self._lock = threading.RLock()
        self._clock, self._observe = clock, observe
        self._started = clock()
        self._last_clock = self._started
        self._window = bytearray(self._policy.max_window_ms * 32)
        self._used = self._operations = self._revision = self._offset_ms = 0
        self._previous: tuple[str, int] | None = None
        self._busy = self._closed = self._cleanup_failed = False
        self._reason = "ready"
        self._worker: WhisperProcess | None = None
        self._audio_timer: threading.Timer | None = None
        self._privacy_timer: threading.Timer | None = None
        self._session_timer: threading.Timer | None = None
        self._audio_generation = self._privacy_generation = 0
        self._pending_since: dict[str, float] = {}
        self._metadata: dict[str, tuple[LanguageConfidence, tuple[TokenTime, ...]]] = {}
        self._peak_rss = self._peak_device = 0
        self._versions: dict[str, str] = {}
        self._withholding_policy = withholding_policy or WithholdingPolicy()
        self._buffer = WithholdingBuffer(
            uuid.uuid4().hex, screener, policy=self._withholding_policy, clock=clock
        )
        limits = SpeechStreamLimits(
            max_buffered_chunks=4096,
            max_buffered_byte_count=len(self._window),
            max_buffered_duration_ms=self._policy.max_window_ms,
            max_stream_duration_ms=self._policy.max_session_ms,
            max_partial_hypotheses=self._policy.max_operations,
            max_segments=self._policy.max_operations,
        )
        self.declaration = SpeechProviderDeclaration(
            ProviderProvenance(
                "mlx-whisper", "openmed-stream-v1", "sha256:" + assets.fingerprint
            ),
            languages,
            (16_000,),
            limits,
        )
        self._contract = StreamingAsrSession(self.declaration)
        failed = False
        try:
            self._resources()
            self._worker = WhisperProcess(assets)
            self._versions = dict(self._worker.versions)
            self._resources()
        except Exception:
            failed = True
        if failed:
            self.cancel()
            raise OfflineAsrError("offline_asr_unavailable")
        self._session_timer = threading.Timer(
            max(0, self._policy.max_session_ms / 1000 - (clock() - self._started)),
            self._expire,
        )
        self._session_timer.daemon = True
        self._session_timer.start()

    def __enter__(self) -> OfflineWhisperSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.cancel()

    def _resources(self) -> None:
        observation = self._observe()
        now = self._clock()
        _require(
            type(now) in (int, float) and math.isfinite(now) and now >= self._last_clock
        )
        self._last_clock = now
        _require((now - self._started) * 1000 <= self._policy.max_session_ms)
        _require(type(observation) is PlatformObservation)
        observation.__post_init__()
        _require(0 <= now - observation.observed_at <= 2.0)
        _require(observation.thermal in (ThermalState.NOMINAL, ThermalState.FAIR))
        _require(observation.memory_pressure is MemoryPressure.NORMAL)
        _require(
            observation.available_bytes is not None
            and observation.available_bytes >= self._policy.min_available_bytes
        )

    def _admit(self) -> None:
        _require(not self._closed and not self._busy)
        self._operations += 1
        _require(self._operations <= self._policy.max_operations)
        self._resources()
        _require(self._buffer.tick() is BufferState.OPEN)

    def _expire(self) -> None:
        self.cancel(reason="retention_expired")

    def _expire_window(self, generation: int) -> None:
        with self._lock:
            if generation == self._audio_generation and self._used:
                self._expire()

    def _expire_privacy(self, generation: int) -> None:
        with self._lock:
            if generation == self._privacy_generation and self._pending_since:
                self._expire()

    def _schedule_privacy_expiry(self) -> None:
        self._privacy_generation += 1
        if self._privacy_timer is not None:
            self._privacy_timer.cancel()
            self._privacy_timer = None
        if self._pending_since:
            remaining = (
                min(self._pending_since.values())
                + self._withholding_policy.max_latency_ms / 1000
                - self._clock()
            )
            self._privacy_timer = threading.Timer(
                max(0, remaining),
                self._expire_privacy,
                args=(self._privacy_generation,),
            )
            self._privacy_timer.daemon = True
            self._privacy_timer.start()

    def push_chunk(self, chunk: AudioChunk) -> None:
        """Copy a sequential bounded PCM chunk; invalid input stops and cleans up."""
        failed = False
        with self._lock:
            try:
                self._admit()
                _require(type(chunk) is AudioChunk and chunk.channel_count == 1)
                _require(chunk.sample_rate_hz == 16_000 and chunk.duration_ms <= 1000)
                _require(len(chunk.payload) == chunk.duration_ms * 32)
                self._contract.push_chunk(chunk)
                if not self._used:
                    self._audio_timer = threading.Timer(
                        self._policy.audio_retention_ms / 1000,
                        self._expire_window,
                        args=(self._audio_generation,),
                    )
                    self._audio_timer.daemon = True
                    self._audio_timer.start()
                self._window[self._used : self._used + len(chunk.payload)] = (
                    chunk.payload
                )
                self._used += len(chunk.payload)
            except Exception:
                failed = True
        if failed:
            self.cancel(reason="input_or_resource_rejected")
            raise OfflineAsrError("offline_asr_input_rejected")

    def _candidate(
        self, raw: dict[str, Any]
    ) -> tuple[
        str, LanguageConfidence, tuple[TokenTime, ...], tuple[TranscriptToken, ...]
    ]:
        _require(raw.get("network_attempts") == 0)
        text = raw["text"]
        _require(type(text) is str and 0 < len(text) <= 8192 and text == text.strip())
        probability = raw["language_probability"]
        _require(
            type(probability) in (int, float)
            and math.isfinite(probability)
            and 0 <= probability <= 1
        )
        language = LanguageConfidence(raw["language"], round(probability * 1_000_000))
        _require(language.language in self.declaration.languages)
        _require(language.confidence_ppm >= self._policy.min_language_confidence_ppm)
        words = raw["words"]
        _require(type(words) is list and 0 < len(words) < 192)
        _require(type(raw["token_count"]) is int and 0 < raw["token_count"] < 192)
        _require("".join(word["word"] for word in words).strip() == text)
        cursor = 0
        timings: list[TokenTime] = []
        lineage: list[TranscriptToken] = []
        for index, word in enumerate(words):
            token = word["word"].strip()
            _require(bool(token))
            start = text.find(token, cursor)
            _require(start >= cursor and not text[cursor:start].strip())
            end = start + len(token)
            for key in ("start", "end", "probability"):
                _require(type(word[key]) in (int, float) and math.isfinite(word[key]))
            start_ms, end_ms = round(word["start"] * 1000), round(word["end"] * 1000)
            _require(0 <= start_ms <= end_ms <= self._used // 32)
            _require(0 <= word["probability"] <= 1)
            timing = TokenTime(
                index,
                self._offset_ms + start_ms,
                self._offset_ms + end_ms,
                round(word["probability"] * 1_000_000),
            )
            if timings:
                _require(
                    timing.start_ms >= timings[-1].start_ms
                    and timing.end_ms >= timings[-1].end_ms
                )
            timings.append(timing)
            lineage.append(
                TranscriptToken(
                    token_ref=uuid.uuid4().hex,
                    start=start,
                    end=end,
                    start_ms=timing.start_ms,
                    end_ms=timing.end_ms,
                )
            )
            cursor = end
        _require(not text[cursor:].strip())
        for key in ("peak_rss_bytes", "peak_device_bytes"):
            _require(type(raw[key]) is int and 0 <= raw[key] <= 1 << 50)
        self._peak_rss = max(self._peak_rss, raw["peak_rss_bytes"])
        self._peak_device = max(self._peak_device, raw["peak_device_bytes"])
        return text, language, tuple(timings), tuple(lineage)

    def partial(self) -> AsrUpdate:
        """Decode growing audio, retaining only a digest of its private hypothesis."""
        return self._decode(final=False, activity=None)

    def end_utterance(self, activity: VoiceActivityTimeline) -> AsrUpdate:
        """Finalize a stable window using matching, window-relative VAD evidence.

        Call after a local endpoint decision. At least one earlier partial on a
        smaller window must match. The method may release an earlier finalized
        utterance after screening the full trailing context. It never guesses
        voice activity, finalizes an unstable result, or retries automatically.
        """
        return self._decode(final=True, activity=activity)

    def _decode(
        self, *, final: bool, activity: VoiceActivityTimeline | None
    ) -> AsrUpdate:
        failed = False
        with self._lock:
            try:
                self._admit()
                _require(self._used >= 16_000 and self._worker is not None)
                if final:
                    _require(
                        type(activity) is VoiceActivityTimeline
                        and activity.duration_ms == self._used // 32
                    )
                self._busy = True
                view = memoryview(self._window)[: self._used].toreadonly()
            except Exception:
                failed = True
        if failed:
            self.cancel(reason="input_or_resource_rejected")
            return AsrUpdate(self._reason, self._revision)
        raw = None
        try:
            assert self._worker is not None
            raw = self._worker.decode(
                view, timeout=self._policy.decode_timeout_ms / 1000
            )
        except Exception:
            failed = True
        finally:
            view.release()
        with self._lock:
            self._busy = False
            if self._closed:
                return AsrUpdate(self._reason, self._revision)
            try:
                _require(not failed and type(raw) is dict)
                assert raw is not None
                self._resources()
                text, language, timings, lineage = self._candidate(raw)
                digest = hashlib.sha256(
                    json.dumps([text, language.language]).encode()
                ).hexdigest()
                stable = (
                    self._previous is not None
                    and self._previous[0] == digest
                    and self._used - self._previous[1]
                    >= self._policy.min_growth_ms * 32
                )
                self._previous = (digest, self._used)
                self._revision += 1
                self._contract.submit_partial(
                    PartialHypothesis(self._revision, text, False, timings, language)
                )
                if not final:
                    self._reason = "partial_withheld"
                    return AsrUpdate(self._reason, self._revision, len(text), language)
                _require(stable)
                stats = AsrSegmentStatistics(
                    raw["no_speech_probability"],
                    raw["avg_logprob"],
                    raw["compression_ratio"],
                )
                report = check_asr_hallucinations(
                    (AsrHypothesisSegment("window", 0, self._used // 32, stats, text),),
                    activity,
                )
                _require(report.decisions[0].outcome is HallucinationOutcome.SUPPORTED)
                segment_ref = uuid.uuid4().hex
                pending_since = self._clock()
                _require(self._buffer.append(segment_ref, text, lineage))
                _require(self._buffer.finalize(segment_ref, expected_revision=0))
                self._contract.submit_segment(
                    FinalizedSegment(
                        self._contract.finalized_segments,
                        text,
                        self._offset_ms,
                        self._offset_ms + self._used // 32,
                        timings,
                        language,
                    )
                )
                self._metadata[segment_ref] = (language, timings)
                self._pending_since[segment_ref] = pending_since
                self._clear_window()
                released = self._release(self._buffer.release_ready())
                self._schedule_privacy_expiry()
                self._resources()
                _require(self._buffer.state is BufferState.OPEN)
                self._reason = "released" if released else "context_or_phi_withheld"
                return AsrUpdate(
                    self._reason, self._revision, len(text), language, released
                )
            except Exception:
                failed = True
        self.cancel(reason="decode_or_safety_rejected")
        return AsrUpdate(self._reason, self._revision)

    def _release(
        self, segments: tuple[ReleasedSpeechSegment, ...]
    ) -> tuple[ScreenedAsrSegment, ...]:
        released = []
        for segment in segments:
            language, timings = self._metadata.pop(segment.segment_ref)
            self._pending_since.pop(segment.segment_ref)
            released.append(ScreenedAsrSegment(segment, language, timings))
        return tuple(released)

    def _clear_window(self) -> None:
        self._audio_generation += 1
        self._offset_ms += self._used // 32
        self._window[:] = b"\0" * len(self._window)
        self._used = 0
        self._previous = None
        self._contract.drain()
        if self._audio_timer is not None:
            self._audio_timer.cancel()
            self._audio_timer = None

    def finish(self) -> AsrUpdate:
        """Screen final trailing context and close; unfinished audio is discarded."""
        failed = False
        released: tuple[ScreenedAsrSegment, ...] = ()
        with self._lock:
            try:
                self._admit()
                _require(self._used == 0)
                released = self._release(self._buffer.finish())
                self._resources()
                _require(self._buffer.state is BufferState.CLOSED)
            except Exception:
                failed = True
            self.cancel(reason="finalized" if not failed else "final_withheld")
            if failed or self._cleanup_failed:
                released = ()
            return AsrUpdate(self._reason, self._revision, released=released)

    def cancel(self, *, reason: str = "cancelled") -> None:
        """Discard buffered text/audio and terminate native work, including in flight."""
        _require(
            reason
            in {
                "cancelled",
                "retention_expired",
                "input_or_resource_rejected",
                "decode_or_safety_rejected",
                "finalized",
                "final_withheld",
            }
        )
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._reason = reason
            self._clear_window()
            self._buffer.close()
            self._metadata.clear()
            self._pending_since.clear()
            self._schedule_privacy_expiry()
            self._contract.cancel()
            if self._session_timer is not None:
                self._session_timer.cancel()
            if self._worker is not None:
                try:
                    self._worker.close()
                except Exception:
                    self._cleanup_failed = True
                    self._reason = "cleanup_failed"

    def report(self) -> dict[str, Any]:
        """Return counts and measured peaks, never raw audio, tokens or transcript."""
        with self._lock:
            return {
                "schema": "openmed.offline_asr.v1",
                "closed": self._closed,
                "reason": self._reason,
                "buffered_bytes": self._used,
                "cleanup_failed": self._cleanup_failed,
                "operations": self._operations,
                "revisions": self._revision,
                "pending_segments": len(self._metadata),
                "peak_worker_rss_bytes": self._peak_rss,
                "peak_device_bytes": self._peak_device,
                "versions": dict(self._versions),
                "provider": self.declaration.to_dict(),
                "speaker": "unknown",
            }
