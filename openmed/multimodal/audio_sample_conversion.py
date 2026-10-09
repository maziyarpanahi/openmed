"""Optional, bounded local conversion of decoded normalized audio samples."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Iterable

from .asr_audio_profile import (
    AsrAudioProfile,
    AsrCompatibility,
    check_asr_compatibility,
)
from .audio_resample_plan import AudioResamplePlan, plan_audio_resampling
from .wav_metadata import WAVE_FORMAT_PCM, WavMetadata

AUDIO_CONVERSION_NOTICE = (
    "Converted audio is non-diagnostic. A reviewer must confirm provider handoff; "
    "transcripts require separate review before consequential use."
)
_FILTER_ID = "hann-sinc-32-zero-pad-v1"


class AudioConversionError(ValueError):
    """Controlled, payload-free conversion failure."""


class ChannelPolicy(str, Enum):
    """Explicit channel handling; never select an arbitrary speaker channel."""

    PRESERVE = "preserve"
    MEAN_MONO = "mean_mono"


@dataclass(frozen=True)
class ConversionBudget:
    """Bounds on owned sample storage, input chunks and filter operations."""

    max_buffer_bytes: int = 32 * 1024 * 1024
    max_chunk_frames: int = 8192
    max_filter_operations: int = 100_000_000

    def __post_init__(self) -> None:
        for value in (
            self.max_buffer_bytes,
            self.max_chunk_frames,
            self.max_filter_operations,
        ):
            if type(value) is not int or value <= 0:
                raise AudioConversionError("conversion_budget_invalid")


@dataclass(frozen=True)
class ConversionReport:
    """Audio-free transform identity and rational frame lineage."""

    transform_digest: str
    source_frames: int
    target_frames: int
    source_rate_hz: int
    target_rate_hz: int
    source_channels: int
    target_channels: int
    channel_policy: ChannelPolicy
    ratio_numerator: int
    ratio_denominator: int
    duration_error_seconds: float
    filter_id: str = _FILTER_ID
    notice: str = AUDIO_CONVERSION_NOTICE

    def source_position(self, target_frame: int) -> Fraction:
        """Map an output frame boundary to its exact source-frame position.

        The final rounded boundary is clamped to the source extent. The
        centered filter compensates delay but uses neighboring source samples.
        """
        if type(target_frame) is not int or not 0 <= target_frame <= self.target_frames:
            raise AudioConversionError("conversion_offset_invalid")
        return min(
            Fraction(self.source_frames),
            Fraction(target_frame * self.ratio_numerator, self.ratio_denominator),
        )

    def source_interval_seconds(
        self, start: int, end: int
    ) -> tuple[Fraction, Fraction]:
        """Map provider transcript frame offsets to exact source times."""
        a, b = self.source_position(start), self.source_position(end)
        if end < start:
            raise AudioConversionError("conversion_offset_invalid")
        return a / self.source_rate_hz, b / self.source_rate_hz


@dataclass(repr=False)
class ConvertedAudio:
    """Protected PCM16 buffer with explicit handoff review and disposal.

    Use ``close`` (or a context manager) to zero owned samples. Caller copies,
    source chunks and provider storage remain the caller's responsibility.
    """

    report: ConversionReport
    _samples: Any = field(repr=False)
    _closed: bool = field(default=False, repr=False)

    def __repr__(self) -> str:
        return "ConvertedAudio(protected_samples)"

    def samples_for_reviewed_handoff(self, *, reviewer_confirmed: bool) -> Any:
        """Borrow frame-major PCM16 samples only after explicit confirmation."""
        if reviewer_confirmed is not True:
            raise AudioConversionError("conversion_review_required")
        if self._closed:
            raise AudioConversionError("conversion_closed")
        return self._samples

    def close(self) -> None:
        """Zero and release the owned output; also invalidates borrowed views."""
        if not self._closed:
            self._samples.fill(0)
            self._samples = None
            self._closed = True

    def __enter__(self) -> ConvertedAudio:
        """Enter protected output lifetime."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Dispose protected samples when leaving the context."""
        self.close()


def _cancelled(cancelled: Callable[[], bool]) -> None:
    try:
        state = cancelled()
    except Exception:
        raise AudioConversionError("conversion_failed") from None
    if type(state) is not bool:
        raise AudioConversionError("conversion_failed")
    if state:
        raise AudioConversionError("conversion_cancelled")


def _checked_chunks(chunks: Iterable[Any]) -> Iterable[Any]:
    try:
        iterator = iter(chunks)
    except Exception:
        raise AudioConversionError("conversion_failed") from None
    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise AudioConversionError("conversion_failed") from None
        yield chunk


def _validate_contract(
    metadata: WavMetadata,
    plan: AudioResamplePlan,
    profile: AsrAudioProfile,
    policy: ChannelPolicy,
) -> tuple[int, int, float]:
    if (
        not isinstance(metadata, WavMetadata)
        or not isinstance(plan, AudioResamplePlan)
        or not isinstance(profile, AsrAudioProfile)
        or not isinstance(policy, ChannelPolicy)
    ):
        raise AudioConversionError("conversion_contract_invalid")
    for value in (
        metadata.sample_rate_hz,
        metadata.channels,
        metadata.bit_depth,
        metadata.frame_count,
        metadata.data_byte_count,
        metadata.format_code,
    ):
        if type(value) is not int or value <= 0:
            raise AudioConversionError("conversion_metadata_invalid")
    if (
        metadata.channels > 64
        or metadata.bit_depth not in (8, 16, 24, 32, 64)
        or metadata.format_code not in (1, 3)
        or (metadata.format_code == 3 and metadata.bit_depth not in (32, 64))
        or (metadata.format_code == 1 and metadata.bit_depth == 64)
        or metadata.data_byte_count
        != metadata.frame_count * metadata.channels * (metadata.bit_depth // 8)
        or type(metadata.duration_seconds) not in (int, float)
        or metadata.duration_seconds != metadata.frame_count / metadata.sample_rate_hz
    ):
        raise AudioConversionError("conversion_metadata_invalid")
    if plan != plan_audio_resampling(
        metadata.sample_rate_hz,
        plan.target_rate_hz,
        metadata.frame_count,
        rounding=plan.rounding,
    ):
        raise AudioConversionError("conversion_plan_mismatch")
    source_report = check_asr_compatibility(metadata, profile)
    if source_report.compatibility in (
        AsrCompatibility.REVIEW,
        AsrCompatibility.INCOMPATIBLE,
    ):
        raise AudioConversionError("conversion_source_incompatible")
    channels = metadata.channels if policy is ChannelPolicy.PRESERVE else 1
    if channels != metadata.channels and not profile.allow_downmix:
        raise AudioConversionError("conversion_downmix_forbidden")
    target = WavMetadata(
        WAVE_FORMAT_PCM,
        channels,
        plan.target_rate_hz,
        16,
        plan.target_frames * channels * 2,
        plan.target_frames,
        plan.target_frames / plan.target_rate_hz,
    )
    if not check_asr_compatibility(target, profile).is_compatible:
        raise AudioConversionError("conversion_target_incompatible")
    cutoff = 0.9 * min(1.0, plan.target_rate_hz / plan.source_rate_hz)
    radius = math.ceil(32 / cutoff)
    if radius > 512:
        raise AudioConversionError("conversion_ratio_unsupported")
    return channels, radius, cutoff


def convert_audio_samples(
    chunks: Iterable[Any],
    metadata: WavMetadata,
    plan: AudioResamplePlan,
    profile: AsrAudioProfile,
    *,
    channel_policy: ChannelPolicy,
    budget: ConversionBudget = ConversionBudget(),
    cancelled: Callable[[], bool] = lambda: False,
) -> ConvertedAudio:
    """Convert bounded decoded float chunks to local profile-compatible PCM16.

    Args:
        chunks: NumPy float32/float64 arrays shaped (frames, source channels),
            decoded and normalized by the caller into [-1, 1]. No file I/O.
        metadata: Original WAV metadata describing the decoded source.
        plan: Existing integer-exact resampling plan, revalidated before use.
        profile: Existing local ASR profile; review/incompatible inputs fail.
        channel_policy: Preserve channels or explicitly average into mono.
        budget: Storage and operation ceilings checked before allocation.
        cancelled: Cooperative callback checked during intake and filtering.

    Returns:
        Protected frame-major PCM16 output and audio-free lineage. All source
        chunks must pass validation before any output is made available.

    Raises:
        AudioConversionError: Controlled failure, including absent NumPy.

    Notes:
        NumPy (BSD licensed, existing multimodal extra) is loaded lazily.
        This bounded batch adapter is independent of input chunk boundaries;
        it does not implement microphone capture, streaming ASR or inference.
    """
    source = output = weights = None
    success = False
    try:
        channels, radius, cutoff = _validate_contract(
            metadata, plan, profile, channel_policy
        )
        if not isinstance(budget, ConversionBudget):
            raise AudioConversionError("conversion_budget_invalid")
        taps = 2 * radius + 1
        required = (
            8 * metadata.frame_count * channels
            + 2 * plan.target_frames * channels
            + 8 * taps
        )
        if (
            required > budget.max_buffer_bytes
            or plan.target_frames * channels * taps > budget.max_filter_operations
        ):
            raise AudioConversionError("conversion_budget_exceeded")
        _cancelled(cancelled)
        try:
            import numpy as np
        except ImportError:
            raise AudioConversionError("conversion_backend_missing") from None
        source = np.zeros((metadata.frame_count, channels), dtype=np.float64)
        output = np.zeros((plan.target_frames, channels), dtype=np.int16)
        weights = np.zeros(taps, dtype=np.float64)
        position = 0
        for chunk in _checked_chunks(chunks):
            _cancelled(cancelled)
            if (
                not isinstance(chunk, np.ndarray)
                or chunk.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or chunk.ndim != 2
                or chunk.shape[1] != metadata.channels
                or not 0 < chunk.shape[0] <= budget.max_chunk_frames
                or position + chunk.shape[0] > metadata.frame_count
            ):
                raise AudioConversionError("conversion_chunk_invalid")
            for row in chunk:
                _cancelled(cancelled)
                total = 0.0
                for channel, raw in enumerate(row):
                    value = float(raw)
                    if not math.isfinite(value):
                        raise AudioConversionError("conversion_sample_invalid")
                    if abs(value) > 1:
                        raise AudioConversionError("conversion_input_clipping")
                    if channel_policy is ChannelPolicy.PRESERVE:
                        source[position, channel] = value
                    total += value
                if channel_policy is ChannelPolicy.MEAN_MONO:
                    source[position, 0] = total / metadata.channels
                position += 1
        if position != metadata.frame_count:
            raise AudioConversionError("conversion_frame_mismatch")
        for frame in range(plan.target_frames):
            _cancelled(cancelled)
            # Compute from global rational coordinates, never a chunk-local phase.
            numerator = frame * plan.rate_ratio_numerator
            center, remainder = divmod(numerator, plan.rate_ratio_denominator)
            fraction = remainder / plan.rate_ratio_denominator
            left = center - radius
            for tap in range(taps):
                distance = tap - radius - fraction
                x = cutoff * distance
                sinc = 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)
                window = (
                    0.5 * (1 + math.cos(math.pi * distance / radius))
                    if abs(distance) <= radius
                    else 0
                )
                weights[tap] = cutoff * sinc * window
            normalization = sum(float(weight) for weight in weights)
            lo, hi = max(0, left), min(metadata.frame_count, left + taps)
            for channel in range(channels):
                if plan.source_rate_hz == plan.target_rate_hz:
                    value = float(source[frame, channel])
                else:
                    # Scalar accumulation avoids hidden contiguous scratch copies
                    # for interleaved channels inside array/BLAS dot operations.
                    value = (
                        sum(
                            float(source[index, channel]) * float(weights[index - left])
                            for index in range(lo, hi)
                        )
                        / normalization
                    )
                if not math.isfinite(value) or not -1 <= value <= 1:
                    raise AudioConversionError("conversion_output_clipping")
                # Asymmetric full-scale PCM16: nearest-even, without saturation.
                output[frame, channel] = round(value * (32768 if value < 0 else 32767))
        _cancelled(cancelled)
        identity = {
            "filter": _FILTER_ID,
            "plan": plan.to_dict(),
            "source_channels": metadata.channels,
            "target_channels": channels,
            "policy": channel_policy.value,
            "encoding": "pcm16-nearest-even-v1",
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("ascii")
        ).hexdigest()
        report = ConversionReport(
            digest,
            plan.source_frames,
            plan.target_frames,
            plan.source_rate_hz,
            plan.target_rate_hz,
            metadata.channels,
            channels,
            channel_policy,
            plan.rate_ratio_numerator,
            plan.rate_ratio_denominator,
            plan.duration_error_seconds,
        )
        result = ConvertedAudio(report, output)
        success = True
        return result
    except AudioConversionError:
        raise
    except Exception:
        raise AudioConversionError("conversion_failed") from None
    finally:
        for buffer in (source, weights, None if success else output):
            if buffer is not None:
                buffer.fill(0)
