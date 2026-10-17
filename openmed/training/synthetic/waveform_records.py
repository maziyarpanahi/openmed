"""Declared, non-diagnostic synthetic waveform templates and event truth."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

_LEADS = frozenset(
    {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}
)
NOTICE = "Synthetic template waveform for software evaluation only; not diagnostic or clinical ground truth."


@dataclass(frozen=True)
class WaveformConfig:
    """Bounded local template parameters; rhythms describe generated timing only."""

    seed: int = 0
    duration_seconds: float = 5.0
    sample_rate_hz: int = 250
    rhythm: str = "regular"
    pulses_per_minute: float = 60.0
    noise_amplitude_mv: float = 0.0
    leads: tuple[str, ...] = ("I", "II")
    annotations: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**53
        ):
            raise ValueError("invalid waveform seed")
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or not 25 <= self.sample_rate_hz <= 2000
        ):
            raise ValueError("invalid synthetic sample rate")
        for value in (
            self.duration_seconds,
            self.pulses_per_minute,
            self.noise_amplitude_mv,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("waveform parameters must be finite numbers")
        if (
            not 0 < self.duration_seconds <= 60
            or not 20 <= self.pulses_per_minute <= 180
            or not 0 <= self.noise_amplitude_mv <= 1
        ):
            raise ValueError("synthetic waveform parameter outside bounds")
        if (
            not isinstance(self.rhythm, str)
            or self.rhythm not in {"regular", "irregular", "pause"}
            or not isinstance(self.annotations, bool)
        ):
            raise ValueError("invalid declared waveform timing mode")
        if (
            not isinstance(self.leads, Sequence)
            or isinstance(self.leads, str)
            or not self.leads
            or len(self.leads) > 12
            or any(not isinstance(lead, str) for lead in self.leads)
            or len(set(self.leads)) != len(self.leads)
            or not set(self.leads) <= _LEADS
        ):
            raise ValueError("provide distinct canonical lead codes")
        object.__setattr__(self, "leads", tuple(sorted(self.leads)))


@dataclass(frozen=True)
class SyntheticWaveformRecord:
    """Generated signals, template-event truth and explicit synthetic device metadata."""

    config: WaveformConfig
    samples: tuple[tuple[str, tuple[float, ...]], ...]
    events: tuple[tuple[str, int, float], ...]
    synthetic: bool = True
    non_diagnostic_notice: str = NOTICE

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config, WaveformConfig)
            or self.synthetic is not True
            or self.non_diagnostic_notice != NOTICE
        ):
            raise ValueError(
                "waveform record requires non-diagnostic synthetic declarations"
            )
        expected = math.ceil(self.config.duration_seconds * self.config.sample_rate_hz)
        if not isinstance(self.samples, Sequence) or len(self.samples) != len(
            self.config.leads
        ):
            raise ValueError("provide bounded waveform channel sequences")
        for row in self.samples:
            if (
                not isinstance(row, Sequence)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not isinstance(row[1], Sequence)
                or len(row[1]) != expected
            ):
                raise ValueError("provide bounded waveform sample sequences")
        rows = tuple((lead, tuple(values)) for lead, values in self.samples)
        if tuple(lead for lead, _values in rows) != self.config.leads:
            raise ValueError("waveform channels do not match declared leads")
        if any(
            len(values) != expected
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
            for _lead, values in rows
        ):
            raise ValueError("synthetic waveform samples are inconsistent")
        if (
            not isinstance(self.events, Sequence)
            or len(self.events) > 181
            or (not self.config.annotations and self.events)
        ):
            raise ValueError("provide bounded events consistent with annotation mode")
        if any(
            not isinstance(event, Sequence) or len(event) != 3 for event in self.events
        ):
            raise ValueError("invalid synthetic waveform event shape")
        events = tuple(tuple(event) for event in self.events)
        for kind, sample, seconds in events:
            if (
                not isinstance(kind, str)
                or kind not in {"template_pulse", "template_pause"}
                or isinstance(sample, bool)
                or not isinstance(sample, int)
                or not 0 <= sample < expected
                or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or not 0 <= seconds < self.config.duration_seconds
                or sample
                != min(expected - 1, round(seconds * self.config.sample_rate_hz))
            ):
                raise ValueError("invalid synthetic waveform event")
        object.__setattr__(self, "samples", rows)
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, object]:
        """Export generated values only, with non-diagnostic and source markers."""
        return {
            "schema_version": 1,
            "synthetic": True,
            "source": "openmed.synthetic.waveform_records",
            "non_diagnostic_notice": NOTICE,
            "rhythm": self.config.rhythm,
            "sample_rate_hz": self.config.sample_rate_hz,
            "units": "mV",
            "device": {
                "manufacturer": "openmed_synthetic",
                "device_digest": hashlib.sha256(
                    f"synthetic-device:{self.config.seed}".encode()
                ).hexdigest(),
            },
            "signals": {lead: list(values) for lead, values in self.samples},
            "events": [
                {"kind": kind, "sample_index": index, "relative_seconds": seconds}
                for kind, index, seconds in self.events
            ],
        }

    def report(self) -> dict[str, object]:
        """Return counts and digests without signal samples."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return {
            "synthetic": True,
            "non_diagnostic_notice": NOTICE,
            "lead_count": len(self.samples),
            "samples_per_lead": len(self.samples[0][1]),
            "event_count": len(self.events),
            "record_digest": hashlib.sha256(encoded).hexdigest(),
        }


def generate_waveform_record(
    config: WaveformConfig = WaveformConfig(),
) -> SyntheticWaveformRecord:
    """Generate fixed template pulses, bounded timing variation and seeded noise.

    Leads are scaled views of one declared pulse template, not physiological
    lead projections. Event truth describes the generator's timing decisions,
    never patient rhythms or diagnoses. Local PRNG state does not alter callers.
    """
    if not isinstance(config, WaveformConfig):
        raise ValueError("expected validated waveform configuration")
    rng = random.Random(config.seed)
    interval = 60 / config.pulses_per_minute
    centers = []
    events = []
    clock = 0.2
    beat = 0
    length = math.ceil(config.duration_seconds * config.sample_rate_hz)
    while clock < config.duration_seconds:
        sample = min(length - 1, int(round(clock * config.sample_rate_hz)))
        paused = config.rhythm == "pause" and beat % 4 == 3
        if not paused:
            centers.append(clock)
        if config.annotations:
            events.append(
                ("template_pause" if paused else "template_pulse", sample, clock)
            )
        jitter = rng.uniform(-0.2, 0.2) if config.rhythm == "irregular" else 0
        clock += interval * (1 + jitter)
        beat += 1
    base = []
    # Fixed Gaussian components are a software stimulus, not an ECG simulator
    # calibrated against clinical records or a diagnostic physiological model.
    components = (
        (-0.16, 0.12, 0.04),
        (-0.04, -0.15, 0.01),
        (0.0, 1.0, 0.01),
        (0.04, -0.25, 0.01),
        (0.22, 0.25, 0.06),
    )
    for index in range(length):
        time = index / config.sample_rate_hz
        value = math.fsum(
            amplitude * math.exp(-0.5 * ((time - center - offset) / width) ** 2)
            for center in centers
            if abs(time - center) < 0.6
            for offset, amplitude, width in components
        )
        value += rng.uniform(-config.noise_amplitude_mv, config.noise_amplitude_mv)
        base.append(value)
    channels = tuple(
        (lead, tuple(value * (1 + index / 20) for value in base))
        for index, lead in enumerate(config.leads)
    )
    return SyntheticWaveformRecord(config, channels, tuple(events))
