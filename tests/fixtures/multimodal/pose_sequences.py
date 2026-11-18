"""Synthetic side-view walkers with gait-cycle truth; never footage of a person.

The generator draws a deterministic sagittal-plane stick figure from gait timing
that is fixed first, so heel-strike and toe-off truth is exact by construction,
and then builds the joint trajectories from that timing. The stance foot stays
planted, the swing foot leaves and lands at pelvis speed (so pelvis-relative
event detectors are unbiased), and the knee comes from two-link inverse
kinematics with fixed thigh and shank lengths. Only the standard library is
used, nothing is written to disk and nothing touches the network.

Conventions:

* Coordinates are ``(x, y)`` tuples in normalized image space of a fixed 16:9
  camera: ``x`` grows to the right, ``y`` grows downwards, and the default
  walks stay inside ``[0, 1]``. Occluded joints are ``None``.
* Walkers move left to right with their left side towards the camera, so right
  limbs are the far side. Left and right hips and shoulders coincide in a pure
  sagittal view.
* Heel strike is initial contact (the heel first reaches the ground plane) and
  toe off is final contact (the toe leaves it). Event frames are
  ``round(time * fps)``; unquantized times are kept for tolerance choices.
* Joint names follow the BlazePose landmark naming. They are a fixture-local
  vocabulary, not the canonical pose contract.

Parameter provenance (scenario values are population means, not targets):

* Body proportions as fractions of height: Winter, *Biomechanics and Motor
  Control of Human Movement*, 4th ed., 2009, segment-length figure.
* Typical adult free walking (113 steps/min, 1.41 m stride, 60 % stance):
  Perry and Burnfield, *Gait Analysis: Normal and Pathological Function*,
  2nd ed., 2010.
* Older adults walking at self-selected speed (109 steps/min, 1.24 m stride,
  64 % stance): Hollman, McDade and Petersen, Gait Posture 34(1):111-118, 2011.
* Pelvis-relative event definitions used by the tests: Zeni, Richards and
  Higginson, Gait Posture 27(4):710-714, 2008.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

JOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
FEET = ("left", "right")
FAR_LEG_JOINTS = ("right_knee", "right_ankle", "right_heel", "right_foot_index")

# Fractions of body height (Winter 2009) and fixed foot geometry.
_HIP, _KNEE, _ANKLE, _SHOULDER, _NOSE, _FOOT = 0.530, 0.285, 0.039, 0.818, 0.91, 0.152
_ANKLE_FORWARD = 0.25  # ankle centre ahead of the heel, as a fraction of foot length
_STRIKE_PITCH = math.radians(15.0)  # toe up at initial contact
_OFF_PITCH = math.radians(-45.0)  # toe down at final contact
_FOOT_FLAT, _HEEL_OFF = 0.13, 0.52  # fractions of stance
_REACH = 0.995  # largest hip-ankle distance, as a fraction of thigh + shank
_MIN_HIP = 0.85  # lowest mean hip height, as a fraction of standing hip height
ASPECT = 16 / 9  # camera frame width over height
_MARGIN = 0.06


class PoseFixtureError(ValueError):
    """Raised for walker parameters the stick figure cannot realize."""


@dataclass(frozen=True)
class WalkerSpec:
    """Gait parameters of one synthetic walker.

    ``asymmetry`` lengthens the left step (right strike to left strike) by
    ``1 + asymmetry`` and shortens the right step by ``1 - asymmetry``; stride
    time and cadence stay unchanged. ``depth`` scales the walker as if further
    from the camera, and ``occlusions`` lists ``(joint, start, stop)`` frame
    ranges, half-open, in which the joint is reported as ``None``.
    """

    cadence_spm: float = 113.0
    stride_length_m: float = 1.41
    stance_fraction: float = 0.60
    asymmetry: float = 0.0
    height_m: float = 1.70
    swing_lift_m: float = 0.04
    start_phase: float = 0.0
    depth: float = 1.0
    occlusions: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class GaitTruth:
    """Event truth and the temporal-spatial values implied by one walker.

    ``double_support_fraction`` is the total of both double-support periods per
    stride; with step-time asymmetry the two periods differ in length.
    """

    heel_strikes: tuple[tuple[str, int], ...]
    toe_offs: tuple[tuple[str, int], ...]
    heel_strike_times_s: tuple[tuple[str, float], ...]
    toe_off_times_s: tuple[tuple[str, float], ...]
    cadence_spm: float
    left_step_time_s: float
    right_step_time_s: float
    stride_time_s: float
    stance_fraction: float
    double_support_fraction: float
    stride_length_m: float
    gait_speed_m_s: float


@dataclass(frozen=True)
class WalkerTrack:
    """Per-frame joints (indexed like ``JOINTS``) and truth for one walker."""

    spec: WalkerSpec
    frames: tuple[tuple[tuple[float, float] | None, ...], ...]
    truth: GaitTruth


@dataclass(frozen=True)
class PoseSequenceFixture:
    """A synthetic multi-walker pose sequence on one frame grid."""

    fps: float
    frame_count: int
    joints: tuple[str, ...]
    walkers: tuple[WalkerTrack, ...]


def gait_truth(spec: WalkerSpec, *, fps: float, frame_count: int) -> GaitTruth:
    """Return the event truth of ``spec`` for frames ``0 .. frame_count - 1``."""
    _validate(spec, fps, frame_count)
    step = 60.0 / spec.cadence_spm
    stride = 2.0 * step
    last = (frame_count - 1) / fps
    events: dict[str, list[tuple[str, float]]] = {"hs": [], "to": []}
    for foot, time, kind in _events(spec, last):
        events[kind].append((foot, time))
    return GaitTruth(
        heel_strikes=tuple((f, round(t * fps)) for f, t in events["hs"]),
        toe_offs=tuple((f, round(t * fps)) for f, t in events["to"]),
        heel_strike_times_s=tuple((f, round(t, 6)) for f, t in events["hs"]),
        toe_off_times_s=tuple((f, round(t, 6)) for f, t in events["to"]),
        cadence_spm=spec.cadence_spm,
        left_step_time_s=round(step * (1 + spec.asymmetry), 6),
        right_step_time_s=round(step * (1 - spec.asymmetry), 6),
        stride_time_s=round(stride, 6),
        stance_fraction=spec.stance_fraction,
        double_support_fraction=round(2 * spec.stance_fraction - 1, 6),
        stride_length_m=spec.stride_length_m,
        gait_speed_m_s=round(spec.stride_length_m / stride, 6),
    )


def synthetic_pose_sequence(
    *,
    walkers: tuple[WalkerSpec, ...] = (WalkerSpec(),),
    fps: float = 30.0,
    duration_s: float = 4.0,
    noise_sd: float = 0.0,
    seed: int = 0,
) -> PoseSequenceFixture:
    """Generate walkers on one frame grid; same arguments give identical output.

    ``noise_sd`` is Gaussian jitter in normalized image units, drawn from one
    ``random.Random(seed)`` in frame, walker, joint order. Occlusion is applied
    after noise, so changing occlusions never shifts the jitter of other joints.
    """
    if not walkers or not 1 <= fps <= 240 or not 1 <= duration_s <= 30:
        raise PoseFixtureError("pose_fixture_invalid_grid")
    if not 0 <= noise_sd <= 0.05:
        raise PoseFixtureError("pose_fixture_invalid_noise")
    frame_count = round(duration_s * fps)
    worlds = [_world_track(spec, fps, frame_count) for spec in walkers]
    # Fixed camera fitted to the walk and the body height, with a margin.
    xs = [p[0] for world in worlds for frame in world for p in frame]
    tallest = max(p[1] for world in worlds for frame in world for p in frame)
    span = max(max(xs) - min(xs), tallest * ASPECT) / (1 - 2 * _MARGIN)
    x0 = (max(xs) + min(xs) - span) / 2
    rng = random.Random(seed)
    rows: list[list[tuple[tuple[float, float] | None, ...]]] = [[] for _ in worlds]
    for index in range(frame_count):
        for spec, world, frames in zip(walkers, worlds, rows, strict=True):
            ground = 1 - _MARGIN - (1 - spec.depth) * 0.5
            row: list[tuple[float, float] | None] = []
            for joint, (wx, wy) in zip(JOINTS, world[index], strict=True):
                x = 0.5 + ((wx - x0) / span - 0.5) * spec.depth
                y = ground - wy * spec.depth * ASPECT / span
                if noise_sd:
                    x += rng.gauss(0, noise_sd)
                    y += rng.gauss(0, noise_sd)
                hidden = any(
                    name == joint and start <= index < stop
                    for name, start, stop in spec.occlusions
                )
                row.append(None if hidden else (round(x, 6), round(y, 6)))
            frames.append(tuple(row))
    tracks = tuple(
        WalkerTrack(
            spec=spec,
            frames=tuple(frames),
            truth=gait_truth(spec, fps=fps, frame_count=frame_count),
        )
        for spec, frames in zip(walkers, rows, strict=True)
    )
    return PoseSequenceFixture(
        fps=fps, frame_count=frame_count, joints=JOINTS, walkers=tracks
    )


def far_leg_crossings(
    spec: WalkerSpec, *, fps: float, frame_count: int, half_width_s: float = 0.1
) -> tuple[tuple[str, int, int], ...]:
    """Return occlusions hiding the far leg while it swings past the near leg."""
    _validate(spec, fps, frame_count)
    stride = 120.0 / spec.cadence_spm
    swing_mid = (spec.stance_fraction + (1 - spec.stance_fraction) / 2) * stride
    windows = []
    # Start one stride early: a swing whose heel strike precedes the clip can
    # still cross the near leg inside it.
    for foot, time, kind in _events(spec, (frame_count - 1) / fps, -stride):
        if foot == "right" and kind == "hs":
            centre = time + swing_mid
            start = max(0, round((centre - half_width_s) * fps))
            stop = min(frame_count, round((centre + half_width_s) * fps) + 1)
            if start < stop:
                windows.extend((joint, start, stop) for joint in FAR_LEG_JOINTS)
    return tuple(windows)


SCENARIOS = {
    "typical_adult": (WalkerSpec(),),
    "older_adult_self_selected": (
        WalkerSpec(cadence_spm=109.0, stride_length_m=1.24, stance_fraction=0.64),
    ),
    "short_stride_low_lift": (
        WalkerSpec(
            cadence_spm=96.0,
            stride_length_m=0.80,
            stance_fraction=0.68,
            swing_lift_m=0.025,
        ),
    ),
    "step_time_asymmetry": (WalkerSpec(asymmetry=0.10),),
    "far_leg_occlusion": (WalkerSpec(),),
    "joint_jitter": (WalkerSpec(),),
    "two_walkers": (
        WalkerSpec(),
        WalkerSpec(
            cadence_spm=109.0,
            stride_length_m=1.24,
            stance_fraction=0.64,
            height_m=1.62,
            start_phase=0.35,
            depth=0.7,
        ),
    ),
}


def scenario(name: str, *, fps: float = 30.0, seed: int = 0) -> PoseSequenceFixture:
    """Return one curated scenario from ``SCENARIOS`` on a 4 s clip."""
    if name not in SCENARIOS:
        raise PoseFixtureError("pose_fixture_unknown_scenario")
    walkers = SCENARIOS[name]
    frame_count = round(4.0 * fps)
    if name == "far_leg_occlusion":
        hidden = far_leg_crossings(walkers[0], fps=fps, frame_count=frame_count)
        walkers = (replace(walkers[0], occlusions=hidden),)
    noise = 0.002 if name == "joint_jitter" else 0.0
    return synthetic_pose_sequence(walkers=walkers, fps=fps, noise_sd=noise, seed=seed)


def _validate(spec: WalkerSpec, fps: float, frame_count: int) -> None:
    checks = (
        30 <= spec.cadence_spm <= 200,
        0.3 <= spec.stride_length_m <= 2.0,
        0.55 <= spec.stance_fraction <= 0.75,
        abs(spec.asymmetry) < 0.3,
        # Both double-support periods must stay positive, or the walker would
        # have a flight phase with neither foot on the ground.
        abs(spec.asymmetry) < 2 * spec.stance_fraction - 1,
        1.0 <= spec.height_m <= 2.2,
        0.0 < spec.swing_lift_m <= 0.15,
        0.0 <= spec.start_phase < 1.0,
        0.4 <= spec.depth <= 1.0,
    )
    if not all(checks):
        raise PoseFixtureError("pose_fixture_invalid_walker")
    if (frame_count - 1) / fps < 120.0 / spec.cadence_spm:
        raise PoseFixtureError("pose_fixture_clip_shorter_than_stride")
    for joint, start, stop in spec.occlusions:
        if joint not in JOINTS or not 0 <= start < stop <= frame_count:
            raise PoseFixtureError("pose_fixture_invalid_occlusion")


def _events(spec: WalkerSpec, last: float, first: float = 0.0):
    """Yield ``(foot, time, kind)`` in time order for events in ``[first, last]``."""
    step = 60.0 / spec.cadence_spm
    stride = 2.0 * step
    offsets = {"right": 0.0, "left": step * (1 + spec.asymmetry)}
    found = []
    for cycle in range(-2, int(last / stride) + 3):
        base = (cycle - spec.start_phase) * stride
        for foot in FEET:
            strike = base + offsets[foot]
            for kind, time in (
                ("hs", strike),
                ("to", strike + spec.stance_fraction * stride),
            ):
                if first <= time <= last:
                    found.append((time, foot, kind))
    for time, foot, kind in sorted(found):
        yield foot, time, kind


def _world_track(spec: WalkerSpec, fps: float, frame_count: int):
    """Return per-frame world joints in metres, y up, ground at 0."""
    _validate(spec, fps, frame_count)
    height = spec.height_m
    thigh, shank = (_HIP - _KNEE) * height, (_KNEE - _ANKLE) * height
    foot = _FOOT * height
    step = 60.0 / spec.cadence_spm
    stride_time = 2.0 * step
    speed = spec.stride_length_m / stride_time
    strike_ahead = 0.20 * spec.stride_length_m
    offsets = {"right": 0.0, "left": step * (1 + spec.asymmetry)}
    times = [index / fps for index in range(frame_count)]
    feet = {
        side: [
            _foot_pose(spec, t, offsets[side], foot, speed, strike_ahead) for t in times
        ]
        for side in FEET
    }
    ankles = {
        side: [
            _point(heel, pitch, foot * _ANKLE_FORWARD, _ANKLE * height)
            for heel, pitch in poses
        ]
        for side, poses in feet.items()
    }
    # Pelvis bob: lowest at mid double support, 2 cycles per stride.
    bob_amp = 0.0125 * height
    trough = (spec.stance_fraction - 0.5) * stride_time / 2
    bob = [
        -bob_amp
        * math.cos(2 * math.pi * (t + spec.start_phase * stride_time - trough) / step)
        for t in times
    ]
    reach = _REACH * (thigh + shank)
    hip_mean = _HIP * height
    for index, t in enumerate(times):
        for side in FEET:
            dx = ankles[side][index][0] - speed * t
            if abs(dx) >= reach:
                raise PoseFixtureError("pose_fixture_unreachable_foot")
            room = ankles[side][index][1] + math.sqrt(reach**2 - dx**2) - bob[index]
            hip_mean = min(hip_mean, room)
    # A stride the leg can only reach by crouching is not walking; refuse it
    # rather than let the knee sink towards the ground.
    if hip_mean < _MIN_HIP * _HIP * height:
        raise PoseFixtureError("pose_fixture_unreachable_foot")
    frames = []
    for index, t in enumerate(times):
        hip = (speed * t, hip_mean + bob[index])
        nose = (hip[0] + 0.03 * height, hip[1] + (_NOSE - _HIP) * height)
        joints = {"nose": nose}
        for side in FEET:
            heel, pitch = feet[side][index]
            ankle = ankles[side][index]
            joints[f"{side}_shoulder"] = (hip[0], hip[1] + (_SHOULDER - _HIP) * height)
            joints[f"{side}_hip"] = hip
            joints[f"{side}_knee"] = _knee(hip, ankle, thigh, shank)
            joints[f"{side}_ankle"] = ankle
            joints[f"{side}_heel"] = heel
            joints[f"{side}_foot_index"] = _point(heel, pitch, foot, 0.0)
        frames.append(tuple(joints[name] for name in JOINTS))
    return frames


def _foot_pose(spec, t, offset, foot, speed, strike_ahead):
    """Return ``(heel, pitch)`` of one foot at time ``t``."""
    stride_time = 120.0 / spec.cadence_spm
    stance = spec.stance_fraction
    shifted = t + spec.start_phase * stride_time - offset
    cycle = math.floor(shifted / stride_time)
    phase = shifted / stride_time - cycle
    strike_time = t - phase * stride_time
    land = speed * strike_time + strike_ahead
    if phase < stance:
        q = phase / stance
        if q < _FOOT_FLAT:
            return (land, 0.0), _STRIKE_PITCH * (1 - _smooth(q / _FOOT_FLAT))
        if q < _HEEL_OFF:
            return (land, 0.0), 0.0
        pitch = _OFF_PITCH * _smooth((q - _HEEL_OFF) / (1 - _HEEL_OFF))
        toe = land + foot
        return (toe - foot * math.cos(pitch), -foot * math.sin(pitch)), pitch
    q = (phase - stance) / (1 - stance)
    swing_time = (1 - stance) * stride_time
    start_x = land + foot - foot * math.cos(_OFF_PITCH)
    start_y = -foot * math.sin(_OFF_PITCH)
    end_x = land + spec.stride_length_m
    # Cubic Hermite: leaves and lands at pelvis speed, so pelvis-relative
    # extrema of heel and toe fall on the true events.
    tangent = speed * swing_time
    x = (
        (2 * q**3 - 3 * q**2 + 1) * start_x
        + (q**3 - 2 * q**2 + q) * tangent
        + (-2 * q**3 + 3 * q**2) * end_x
        + (q**3 - q**2) * tangent
    )
    # The heel drops early while the foot pitches up steadily, which gives the
    # usual two-peaked toe path with a minimum toe clearance in mid-swing.
    pitch = _OFF_PITCH + (_STRIKE_PITCH - _OFF_PITCH) * math.sin(math.pi * q / 2)
    y = start_y * (1 - _smooth(min(1.0, q / 0.55)))
    y += spec.swing_lift_m * math.sin(math.pi * q)
    # Keep the toe off the ground for the whole swing, whatever the parameters.
    y = max(
        y, -foot * math.sin(pitch) + 0.25 * spec.swing_lift_m * math.sin(math.pi * q)
    )
    return (x, y), pitch


def _point(heel, pitch, along, up):
    """Return a foot-frame point ``along`` the sole and ``up`` from it."""
    cos, sin = math.cos(pitch), math.sin(pitch)
    return (heel[0] + along * cos - up * sin, heel[1] + along * sin + up * cos)


def _knee(hip, ankle, thigh, shank):
    """Two-link inverse kinematics with the knee in front of the hip-ankle line."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    reach = math.hypot(dx, dy)
    bend = math.acos((thigh**2 + reach**2 - shank**2) / (2 * thigh * reach))
    angle = math.atan2(dy, dx) + bend
    return (hip[0] + thigh * math.cos(angle), hip[1] + thigh * math.sin(angle))


def _smooth(u):
    """Minimum-jerk blend from 0 to 1."""
    return u * u * u * (10 - 15 * u + 6 * u * u)
