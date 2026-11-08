"""Offline acquisition contracts for non-ECG waveforms, never diagnostic values."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

__all__ = [
    "NON_DIAGNOSTIC_NOTICE",
    "ChannelKind",
    "ChannelConstraints",
    "CHANNEL_CONSTRAINTS",
    "WaveformChannel",
    "WaveformProvenance",
    "WaveformContractError",
    "QualityState",
    "ChannelQuality",
    "RecordingQuality",
    "evaluate_recording",
]

NON_DIAGNOSTIC_NOTICE = (
    "Acquisition quality only; not vital-sign measurements, diagnosis or alarms. "
    "Consequential use requires explicit reviewer confirmation."
)
MAX_CHANNELS = 64
MAX_SAMPLES = 1_000_000


class WaveformContractError(ValueError):
    """Controlled, payload-free rejection at the acquisition boundary."""


class ChannelKind(str, Enum):
    """Supported channel semantics; ECG leads and unknown kinds are refused."""

    PPG = "ppg"
    RESPIRATION = "respiration"
    INVASIVE_PRESSURE = "invasive_pressure"
    NON_INVASIVE_PRESSURE = "non_invasive_pressure"
    CAPNOGRAPHY = "capnography"


class QualityState(str, Enum):
    """Acquisition usability codes, not assessments of patient condition."""

    PASS = "pass"
    LIMITED_USE = "limited_use"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ChannelConstraints:
    """Version-one encoding limits and per-kind motion-proxy step fraction.

    These engineering bounds are not reference intervals or clinical thresholds.
    """

    unit: str
    minimum: float
    maximum: float
    minimum_rate_hz: float
    maximum_rate_hz: float
    motion_step_fraction: float


CHANNEL_CONSTRAINTS = MappingProxyType(
    {
        ChannelKind.PPG: ChannelConstraints("normalized", 0, 1, 10, 2000, 0.4),
        ChannelKind.RESPIRATION: ChannelConstraints("normalized", -1, 1, 1, 200, 0.6),
        ChannelKind.INVASIVE_PRESSURE: ChannelConstraints(
            "mmHg", -50, 400, 10, 2000, 0.3
        ),
        ChannelKind.NON_INVASIVE_PRESSURE: ChannelConstraints(
            "mmHg", 0, 400, 1, 1000, 0.3
        ),
        ChannelKind.CAPNOGRAPHY: ChannelConstraints("mmHg", 0, 150, 1, 500, 0.5),
    }
)


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True, repr=False)
class WaveformProvenance:
    """Caller-supplied source SHA-256; no device names, paths or source headers.

    Args:
        source_sha256: Lowercase hexadecimal digest of caller-owned source bytes.
            The contract validates syntax, not the authenticity of acquisition.
    """

    source_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.source_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise WaveformContractError("invalid_provenance")

    def __repr__(self) -> str:
        return "WaveformProvenance()"


@dataclass(frozen=True, slots=True, repr=False)
class WaveformChannel:
    """Immutable protected samples on a regular relative acquisition clock.

    Args:
        kind: A supported non-ECG channel kind.
        unit: Exact encoding unit for this kind; no implicit conversion.
        sample_rate_hz: Nominal acquisition rate within version-one bounds.
        acquisition_minimum: Declared lower acquisition rail in the given unit.
        acquisition_maximum: Declared upper acquisition rail in the given unit.
        samples: At least two finite samples, or None for explicit dropout.
        offsets_seconds: Nonnegative relative offsets, one per sample. Missing
            slots must be represented by None, never silently removed.
        provenance: Caller-supplied source digest, excluded from quality reports.

    Samples remain sensitive and are not de-identified. No serializer is provided.
    """

    kind: ChannelKind
    unit: str
    sample_rate_hz: float
    acquisition_minimum: float
    acquisition_maximum: float
    samples: tuple[float | None, ...] = field(repr=False)
    offsets_seconds: tuple[float, ...] = field(repr=False)
    provenance: WaveformProvenance = field(repr=False)

    def __post_init__(self) -> None:
        try:
            kind = ChannelKind(self.kind)
        except (ValueError, TypeError):
            raise WaveformContractError("unknown_kind") from None
        limits = CHANNEL_CONSTRAINTS[kind]
        if type(self.unit) is not str or self.unit != limits.unit:
            raise WaveformContractError("invalid_unit")
        if not _number(self.sample_rate_hz) or not (
            limits.minimum_rate_hz <= self.sample_rate_hz <= limits.maximum_rate_hz
        ):
            raise WaveformContractError("invalid_sampling_rate")
        low, high = self.acquisition_minimum, self.acquisition_maximum
        if (
            not _number(low)
            or not _number(high)
            or not (limits.minimum <= low < high <= limits.maximum)
        ):
            raise WaveformContractError("invalid_acquisition_range")
        if type(self.provenance) is not WaveformProvenance:
            raise WaveformContractError("invalid_provenance")
        if type(self.samples) not in (tuple, list) or type(
            self.offsets_seconds
        ) not in (
            tuple,
            list,
        ):
            raise WaveformContractError("invalid_sample_shape")
        if not 2 <= len(self.samples) <= MAX_SAMPLES or len(self.samples) != len(
            self.offsets_seconds
        ):
            raise WaveformContractError("invalid_sample_shape")
        for value in self.samples:
            if value is not None and (not _number(value) or not low <= value <= high):
                raise WaveformContractError("invalid_sample")
        for index, offset in enumerate(self.offsets_seconds):
            if not _number(offset) or offset < 0:
                raise WaveformContractError("invalid_timing")
            if index:
                step = offset - self.offsets_seconds[index - 1]
                if step <= 0 or abs(step * self.sample_rate_hz - 1) > 0.01:
                    raise WaveformContractError("invalid_timing")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "offsets_seconds", tuple(self.offsets_seconds))

    @property
    def non_diagnostic_notice(self) -> str:
        """Return the mandatory acquisition-only boundary."""
        return NON_DIAGNOSTIC_NOTICE

    def __repr__(self) -> str:
        return (
            f"WaveformChannel(kind={self.kind.value}, sample_count={len(self.samples)})"
        )


@dataclass(frozen=True, slots=True)
class ChannelQuality:
    """Counts and controlled quality codes for one channel in input order."""

    kind: ChannelKind
    state: QualityState
    sample_count: int
    dropout_count: int
    saturation_count: int
    flatline_count: int
    motion_proxy_count: int
    codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not ChannelKind or type(self.state) is not QualityState:
            raise WaveformContractError("invalid_quality_report")
        counts = (
            self.sample_count,
            self.dropout_count,
            self.saturation_count,
            self.flatline_count,
            self.motion_proxy_count,
        )
        if any(
            type(count) is not int or not 0 <= count <= MAX_SAMPLES for count in counts
        ):
            raise WaveformContractError("invalid_quality_report")
        if type(self.codes) is not tuple or any(
            type(code) is not str
            or code not in {"dropout", "saturation", "flatline", "motion_proxy"}
            for code in self.codes
        ):
            raise WaveformContractError("invalid_quality_report")

    def to_dict(self) -> dict[str, object]:
        """Return only kind, counts and controlled codes."""
        return {
            "kind": self.kind.value,
            "state": self.state.value,
            "sample_count": self.sample_count,
            "dropout_count": self.dropout_count,
            "saturation_count": self.saturation_count,
            "flatline_count": self.flatline_count,
            "motion_proxy_count": self.motion_proxy_count,
            "codes": list(self.codes),
        }


@dataclass(frozen=True, slots=True)
class RecordingQuality:
    """Content-free mixed-kind report with a mandatory review boundary."""

    channels: tuple[ChannelQuality, ...]

    def __post_init__(self) -> None:
        if (
            type(self.channels) is not tuple
            or not 1 <= len(self.channels) <= MAX_CHANNELS
            or any(type(channel) is not ChannelQuality for channel in self.channels)
        ):
            raise WaveformContractError("invalid_quality_report")

    @property
    def non_diagnostic_notice(self) -> str:
        """Return the mandatory acquisition-only boundary."""
        return NON_DIAGNOSTIC_NOTICE

    def to_dict(self) -> dict[str, object]:
        """Return deterministic counts/codes only, excluding provenance and data."""
        return {
            "channel_count": len(self.channels),
            "codes": ["non_diagnostic", "reviewer_confirmation_required"],
            "channels": [channel.to_dict() for channel in self.channels],
        }

    def to_json(self) -> str:
        """Serialize the content-free report deterministically."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=True)

    def require_reviewer_confirmation(self, *, confirmed: bool = False) -> None:
        """Refuse consequential use until a caller explicitly confirms review.

        Confirmation never qualifies a signal or authorizes automated decisions.
        """
        if confirmed is not True:
            raise WaveformContractError("reviewer_confirmation_required")


def _quality(channel: WaveformChannel) -> ChannelQuality:
    limits = CHANNEL_CONSTRAINTS[channel.kind]
    values = channel.samples
    span = channel.acquisition_maximum - channel.acquisition_minimum
    dropout = sum(value is None for value in values)
    saturation = sum(
        value in (channel.acquisition_minimum, channel.acquisition_maximum)
        for value in values
        if value is not None
    )
    flatline = motion = run = 0
    # Count transitions in >=1 second contiguous near-constant runs. Dropouts
    # break runs and adjacency; no interpolated or derived physiological values.
    minimum_run = math.ceil(channel.sample_rate_hz)
    for previous, current in zip(values, values[1:]):
        if previous is not None and current is not None:
            step = abs(current - previous)
            motion += step > span * limits.motion_step_fraction
            if step <= span * 1e-6:
                run += 1
                continue
        if run >= minimum_run:
            flatline += run
        run = 0
    if run >= minimum_run:
        flatline += run
    codes = tuple(
        code
        for code, count in (
            ("dropout", dropout),
            ("saturation", saturation),
            ("flatline", flatline),
            ("motion_proxy", motion),
        )
        if count
    )
    state = QualityState.PASS
    if codes:
        state = QualityState.LIMITED_USE
    if flatline or dropout * 10 >= len(values) or saturation * 10 >= len(values):
        state = QualityState.REVIEW
    if motion * 10 >= len(values) - 1:
        state = QualityState.REVIEW
    return ChannelQuality(
        channel.kind, state, len(values), dropout, saturation, flatline, motion, codes
    )


def evaluate_recording(channels: tuple[WaveformChannel, ...]) -> RecordingQuality:
    """Validate and assess mixed kinds without resampling or clinical inference.

    Args:
        channels: One to 64 validated channels, each with its own regular relative
            clock. Rates and durations may differ; no clock alignment is claimed.

    Returns:
        Content-free channel quality results, in input order.

    Raises:
        WaveformContractError: For invalid channel collections or resource limits.
    """
    if type(channels) not in (tuple, list) or not 1 <= len(channels) <= MAX_CHANNELS:
        raise WaveformContractError("invalid_recording")
    if any(type(channel) is not WaveformChannel for channel in channels):
        raise WaveformContractError("invalid_channel")
    if sum(len(channel.samples) for channel in channels) > MAX_SAMPLES:
        raise WaveformContractError("resource_limit")
    return RecordingQuality(tuple(_quality(channel) for channel in channels))
