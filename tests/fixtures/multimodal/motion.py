"""Synthetic waist accelerometer windows with fall truth; never data from a person.

The generator draws deterministic tri-axial accelerometer windows for a sensor
worn at the waist. The event time is fixed first, so the truth is exact by
construction, and each window is then built from the phases that threshold fall
detectors look for: activity before the event, a descent in which the signal
magnitude drops while the trunk rotates, an impact pulse whose peak lies on the
truth onset sample, and an end posture that is held. Only the standard library
is used, nothing is written to disk and nothing touches the network.

Conventions:

* Samples are ``(x, y, z)`` tuples of proper acceleration (specific force) in
  g, at ``rate_hz``, with time known only relative to the first sample. Missing
  samples are ``None``. ``x`` points to the wearer's right, ``y`` to the head
  and ``z`` forward, so an upright wearer at rest reads ``(0, 1, 0)`` and a
  supine one ``(0, 0, 1)``. This is the Android sign; Core Motion reports the
  opposite sign, so its samples must be negated to compare.
* The onset is the sample of the impact or contact peak. Every window with an
  event reads at least ``peak_g`` in magnitude on that sample and less
  everywhere else before noise. The peak is placed on a sample, which is the
  favourable case: at low rates a real peak between samples reads lower.
* No-fall windows are confounders, each built to cross a published threshold
  while differing from a fall in one feature: fast sit-down, jump and
  stumble-recover end upright, lying down has no impact, and a dropped device
  is in true free fall (about 0 g) before a narrow pulse.

Magnitudes are design choices placed around published detector thresholds, not
population values: Bourke, O'Brien and Lyons, Gait Posture 26:194-199, 2007;
Kangas et al., Gait Posture 28:285-291, 2008; Bourke et al., J Biomech
43:3051-3057, 2010; all as compared on real-world falls by Bagala et al., PLoS
ONE 7(5):e37062, 2012, which found lower impacts than in simulated falls (hence
the soft fall). The activity list follows SisFall (Sucerquia, Lopez and
Vargas-Bonilla, Sensors 17(1):198, 2017); no data from any dataset is used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields, replace
from typing import Any

KINDS = (
    "standing",
    "walking",
    "fall",
    "fast_sit_down",
    "lying_down",
    "stumble_recover",
    "jump",
    "device_drop",
)
FALL_POSTURES = {
    "backward": "supine",
    "forward": "prone",
    "left": "left_side",
    "right": "right_side",
}
# World "up" in the sensor frame once the trunk is horizontal, per end posture.
_LYING_UP = {
    "supine": (0.0, 0.0, 1.0),
    "prone": (0.0, 0.0, -1.0),
    "left_side": (1.0, 0.0, 0.0),
    "right_side": (-1.0, 0.0, 0.0),
    "device_flat": (0.0, 0.0, 1.0),
}
_FORWARD_LEAN = (0.0, 0.0, -1.0)
_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
_STILL_AFTER_S = 1.0  # stillness truth starts this long after the onset
_PRE_EVENT_S = 0.6  # room needed before the descent or flight
_POST_EVENT_S = 3.0  # room needed after the onset for posture and stillness
_ROUND = 1e-5  # head-room for rounding samples to 6 decimals


class MotionFixtureError(ValueError):
    """Raised for parameters that would not give a window with exact truth."""


@dataclass(frozen=True)
class MotionSpec:
    """Parameters of one synthetic accelerometer window.

    ``peak_g`` is the smallest magnitude the onset sample can read once noise
    is added. ``trough_g`` is the lowest magnitude during a fall or sit-down
    descent, and ``descent_s`` the length of the descent, of the flight of a
    jump, or of the free fall of a dropped device. ``direction`` sets the end
    posture of a fall. ``gaps`` are half-open ``(start_s, stop_s)`` intervals;
    every sample whose timestamp falls inside one is reported as ``None``.
    Noise is Gaussian per axis, clipped at three standard deviations. A window
    whose readings could reach ``range_g`` is refused rather than clipped, and
    ``peak_g`` is only checked against ``range_g`` when the window has an event.

    Every number is checked before use: ``rate_hz`` lies in 1 to 1000 Hz,
    ``duration_s`` in (0, 600] s, ``impact_width_s`` in (0, 0.3] s and, for a
    jump or a dropped device, below ``descent_s`` so a free fall remains;
    ``seed`` is an integer. Booleans are not accepted as numbers.
    """

    kind: str
    rate_hz: float = 100.0
    duration_s: float = 10.0
    onset_s: float = 4.0
    peak_g: float = 4.0
    impact_width_s: float = 0.05
    trough_g: float = 0.35
    descent_s: float = 0.5
    direction: str = "backward"
    cadence_spm: float = 113.0
    noise_g: float = 0.01
    range_g: float = 16.0
    gaps: tuple[tuple[float, float], ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class MotionTruth:
    """Event truth of one window; indexes count samples from zero.

    Windows without an event (standing, walking) have no onset and no peak.
    ``still_from_index`` is the first sample of post-event stillness, or
    ``None`` when activity resumes. ``gaps`` are half-open sample ranges.
    """

    kind: str
    is_fall: bool
    onset_index: int | None
    onset_time_s: float | None
    peak_g: float | None
    posture_after: str
    still_from_index: int | None
    gaps: tuple[tuple[int, int], ...]
    onset_in_gap: bool


@dataclass(frozen=True)
class MotionWindow:
    """Samples, sensor description and truth of one synthetic window."""

    spec: MotionSpec
    rate_hz: float
    unit: str
    axes: tuple[str, str, str]
    samples: tuple[tuple[float, float, float] | None, ...]
    truth: MotionTruth


def synthetic_motion_window(spec: MotionSpec) -> MotionWindow:
    """Generate one window; equal specs give identical output.

    Noise comes from ``random.Random(spec.seed)`` and is drawn for every sample,
    including gap samples, so adding a gap never changes the other samples.
    """
    count, onset, gaps = _validate(spec)
    times = [index / spec.rate_hz for index in range(count)]
    clean = [_base(spec, t, onset) for t in times]
    if onset is not None:
        t_on = onset / spec.rate_hz
        peak = spec.peak_g + _noise_bound(spec) + _ROUND
        base = clean[onset]
        up = _up(spec, t_on, onset)
        along = sum(b * u for b, u in zip(base, up, strict=True))
        amplitude = -along + math.sqrt(along**2 - _norm2(base) + peak**2)
        sigma = spec.impact_width_s / _FWHM
        for index, t in enumerate(times):
            pulse = amplitude * _gauss(t - t_on, sigma)
            if spec.kind == "device_drop":  # one small bounce after landing
                pulse += 0.25 * amplitude * _gauss(t - t_on - 0.08, sigma)
            if pulse > 1e-12 and not _free_fall(spec, t, t_on):
                up = _up(spec, t, onset)
                clean[index] = tuple(
                    c + pulse * u for c, u in zip(clean[index], up, strict=True)
                )
        if max(_norm2(sample) for sample in clean) > _norm2(clean[onset]) * (1 + 1e-12):
            raise MotionFixtureError("motion_fixture_onset_not_peak")
    rng = random.Random(spec.seed)
    bound = 3.0 * spec.noise_g
    if max(abs(v) for sample in clean for v in sample) + bound + _ROUND > spec.range_g:
        raise MotionFixtureError("motion_fixture_signal_above_range")
    samples: list[tuple[float, float, float] | None] = []
    for sample in clean:
        noisy = []
        for value in sample:
            if spec.noise_g:
                value += max(-bound, min(bound, rng.gauss(0.0, spec.noise_g)))
            noisy.append(round(max(-spec.range_g, min(spec.range_g, value)), 6))
        samples.append((noisy[0], noisy[1], noisy[2]))
    for start, stop in gaps:
        samples[start:stop] = [None] * (stop - start)
    return MotionWindow(
        spec=spec,
        rate_hz=spec.rate_hz,
        unit="g",
        axes=("x", "y", "z"),
        samples=tuple(samples),
        truth=_truth(spec, onset, gaps),
    )


SCENARIOS = {
    "standing": MotionSpec(kind="standing"),
    "walking": MotionSpec(kind="walking"),
    "fall_backward": MotionSpec(kind="fall", direction="backward"),
    "fall_forward": MotionSpec(kind="fall", direction="forward"),
    "fall_lateral": MotionSpec(kind="fall", direction="right"),
    "soft_fall": MotionSpec(
        kind="fall",
        direction="left",
        peak_g=2.2,
        trough_g=0.65,
        descent_s=0.9,
        impact_width_s=0.12,
    ),
    "fall_with_gap": MotionSpec(kind="fall", gaps=((5.0, 7.0),)),
    "fast_sit_down": MotionSpec(
        kind="fast_sit_down",
        peak_g=2.4,
        trough_g=0.55,
        descent_s=0.6,
        impact_width_s=0.08,
    ),
    "lying_down": MotionSpec(
        kind="lying_down",
        peak_g=1.2,
        trough_g=0.88,
        descent_s=2.5,
        impact_width_s=0.3,
    ),
    "stumble_recover": MotionSpec(
        kind="stumble_recover", peak_g=2.6, trough_g=0.6, descent_s=0.3
    ),
    "jump": MotionSpec(kind="jump", peak_g=3.6, descent_s=0.28, impact_width_s=0.04),
    "device_drop": MotionSpec(
        kind="device_drop", peak_g=3.8, descent_s=0.45, impact_width_s=0.01
    ),
}


def scenario(name: str, **overrides: Any) -> MotionWindow:
    """Return one curated scenario from ``SCENARIOS``, with optional overrides."""
    if not isinstance(name, str) or name not in SCENARIOS:
        raise MotionFixtureError("motion_fixture_unknown_scenario")
    if not set(overrides) <= {field.name for field in fields(MotionSpec)}:
        raise MotionFixtureError("motion_fixture_unknown_override")
    return synthetic_motion_window(replace(SCENARIOS[name], **overrides))


def _validate(spec: MotionSpec) -> tuple[int, int | None, tuple[tuple[int, int], ...]]:
    """Return sample count, onset index and gap ranges, or fail closed."""
    if not isinstance(spec.kind, str) or spec.kind not in KINDS:
        raise MotionFixtureError("motion_fixture_unknown_kind")
    if not isinstance(spec.direction, str) or spec.direction not in FALL_POSTURES:
        raise MotionFixtureError("motion_fixture_unknown_direction")
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
        raise MotionFixtureError("motion_fixture_invalid_seed")
    if not isinstance(spec.gaps, tuple) or any(
        not isinstance(gap, tuple) or len(gap) != 2 for gap in spec.gaps
    ):
        raise MotionFixtureError("motion_fixture_invalid_gap")
    numbers = (
        spec.rate_hz,
        spec.duration_s,
        spec.onset_s,
        spec.peak_g,
        spec.impact_width_s,
        spec.trough_g,
        spec.descent_s,
        spec.cadence_spm,
        spec.noise_g,
        spec.range_g,
        *(edge for gap in spec.gaps for edge in gap),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in numbers
    ):
        raise MotionFixtureError("motion_fixture_invalid_parameter")
    if not all(math.isfinite(value) for value in numbers):
        raise MotionFixtureError("motion_fixture_non_finite")
    checks = (
        1 <= spec.rate_hz <= 1000,
        0 < spec.duration_s <= 600,
        0 < spec.impact_width_s <= 0.3,
        0 <= spec.trough_g < 1,
        spec.descent_s > 0,
        30 <= spec.cadence_spm <= 200,
        0 <= spec.noise_g <= 0.5,
        spec.range_g > 0,
    )
    if not all(checks):
        raise MotionFixtureError("motion_fixture_invalid_parameter")
    count = round(spec.duration_s * spec.rate_hz)
    if count < 2:
        raise MotionFixtureError("motion_fixture_too_few_samples")
    onset = None
    if spec.kind not in ("standing", "walking"):
        if spec.peak_g + 2 * _noise_bound(spec) + _ROUND > spec.range_g:
            raise MotionFixtureError("motion_fixture_peak_above_range")
        onset = round(spec.onset_s * spec.rate_hz)
        if not 0 <= onset < count:
            raise MotionFixtureError("motion_fixture_onset_outside_window")
        if onset / spec.rate_hz < spec.descent_s + _PRE_EVENT_S:
            raise MotionFixtureError("motion_fixture_no_room_before_event")
        if (count - 1 - onset) / spec.rate_hz < _POST_EVENT_S:
            raise MotionFixtureError("motion_fixture_no_room_after_event")
        if spec.peak_g <= 1.0:
            raise MotionFixtureError("motion_fixture_peak_not_above_gravity")
        if (
            spec.kind in ("jump", "device_drop")
            and spec.impact_width_s >= spec.descent_s
        ):
            raise MotionFixtureError("motion_fixture_no_free_fall")
    gaps = []
    previous = 0
    for start_s, stop_s in spec.gaps:
        start, stop = _first_index_at(start_s, spec), _first_index_at(stop_s, spec)
        if not previous <= start < stop <= count:
            raise MotionFixtureError("motion_fixture_invalid_gap")
        gaps.append((start, stop))
        previous = stop
    return count, onset, tuple(gaps)


def _truth(spec, onset, gaps):
    posture = "upright"
    if spec.kind == "fall":
        posture = FALL_POSTURES[spec.direction]
    elif spec.kind == "lying_down":
        posture = "supine"
    elif spec.kind == "device_drop":
        posture = "device_flat"
    still = None
    if onset is not None and spec.kind != "stumble_recover":
        still = onset + round(_STILL_AFTER_S * spec.rate_hz)
    return MotionTruth(
        kind=spec.kind,
        is_fall=spec.kind == "fall",
        onset_index=onset,
        onset_time_s=None if onset is None else round(onset / spec.rate_hz, 9),
        peak_g=None if onset is None else spec.peak_g,
        posture_after=posture,
        still_from_index=still,
        gaps=gaps,
        onset_in_gap=onset is not None and any(a <= onset < b for a, b in gaps),
    )


def _tilt(spec, t, t_on):
    """Return ``(angle, toward)``: trunk tilt from upright and its direction."""
    kind = spec.kind
    if kind == "standing":
        return math.radians(2.0) * math.sin(2 * math.pi * 0.3 * t), _FORWARD_LEAN
    if kind == "fall":
        start = t_on - spec.descent_s
        lying = _LYING_UP[FALL_POSTURES[spec.direction]]
        return math.pi / 2 * _ramp(t, start, t_on - 0.02), lying
    if kind == "fast_sit_down":
        start = t_on - spec.descent_s
        lean = 25.0 * _bump(t, start, t_on + 0.4) + 8.0 * _ramp(t, t_on, t_on + 0.6)
        return math.radians(lean), _FORWARD_LEAN
    if kind == "lying_down":
        start = t_on - spec.descent_s
        return math.pi / 2 * _ramp(t, start, t_on - 0.1), _LYING_UP["supine"]
    if kind == "stumble_recover":
        start = t_on - spec.descent_s
        return math.radians(30.0) * _bump(t, start, t_on + 0.6), _FORWARD_LEAN
    if kind == "device_drop":
        start = t_on - spec.descent_s
        return math.pi / 2 * _ramp(t, start, t_on), _LYING_UP["device_flat"]
    return 0.0, _FORWARD_LEAN


def _up(spec, t, onset):
    """World up in the sensor frame at time ``t``."""
    t_on = onset / spec.rate_hz
    angle, toward = _tilt(spec, t, t_on)
    cos, sin = math.cos(angle), math.sin(angle)
    return (sin * toward[0], cos, sin * toward[2])


def _base(spec, t, onset):
    """Return the reading at ``t`` without the impact pulse or noise."""
    kind = spec.kind
    t_on = 0.0 if onset is None else onset / spec.rate_hz
    start = t_on - spec.descent_s
    vertical = 0.0
    walk = (0.0, 0.0, 0.0)
    if kind == "walking" or kind == "stumble_recover":
        walk = _walk(spec, t)
    if kind == "fall":
        fade = 1.0 - _ramp(t, start - 0.2, start)
        walk = tuple(fade * value for value in _walk(spec, t))
    if kind in ("fall", "fast_sit_down", "stumble_recover", "lying_down"):
        vertical -= (1.0 - spec.trough_g) * _bump(t, start, t_on - 0.02)
    if _free_fall(spec, t, t_on):
        return (0.0, 0.0, 0.0)
    if kind == "jump":
        vertical += -0.3 * _bump(t, start - 0.6, start - 0.2)
        vertical += 0.8 * _bump(t, start - 0.25, start)
    up = (0.0, 1.0, 0.0) if onset is None else _up(spec, t, onset)
    return tuple((1.0 + vertical) * u + w for u, w in zip(up, walk, strict=True))


def _free_fall(spec, t, t_on):
    """True during the flight of a jump or the drop of a device."""
    if spec.kind not in ("jump", "device_drop"):
        return False
    return t_on - spec.descent_s <= t < t_on - spec.impact_width_s


def _walk(spec, t):
    """Waist acceleration of level walking: lateral, vertical, forward, in g."""
    step_hz = spec.cadence_spm / 60.0
    vertical = 0.22 * math.sin(2 * math.pi * step_hz * t)
    strike = round(t * step_hz) / step_hz + 0.02
    vertical += 0.35 * _gauss(t - strike, 0.015)  # heel-strike transient
    forward = 0.12 * math.sin(2 * math.pi * step_hz * t + 1.2)
    lateral = 0.08 * math.sin(math.pi * step_hz * t)
    return (lateral, vertical, forward)


def _first_index_at(time_s, spec):
    """Index of the first sample whose timestamp is at or after ``time_s``.

    Gaps cover the half-open interval ``[start_s, stop_s)``, so a sample
    belongs to a gap exactly when its timestamp ``index / rate_hz`` falls in
    that interval, whether or not the bounds sit on the sample grid.
    """
    return math.ceil(time_s * spec.rate_hz - 1e-9)


def _noise_bound(spec):
    """Largest change in magnitude that clipped per-axis noise can cause."""
    return 3.0 * spec.noise_g * math.sqrt(3.0)


def _norm2(vector):
    return sum(value * value for value in vector)


def _gauss(offset, sigma):
    return math.exp(-0.5 * (offset / sigma) ** 2)


def _bump(t, start, stop):
    """Raised cosine: zero outside ``(start, stop)``, one in the middle."""
    if t <= start or t >= stop:
        return 0.0
    return 0.5 - 0.5 * math.cos(2 * math.pi * (t - start) / (stop - start))


def _ramp(t, start, stop):
    """Minimum-jerk blend from 0 at ``start`` to 1 at ``stop``."""
    if t <= start:
        return 0.0
    if t >= stop:
        return 1.0
    u = (t - start) / (stop - start)
    return u * u * u * (10 - 15 * u + 6 * u * u)
