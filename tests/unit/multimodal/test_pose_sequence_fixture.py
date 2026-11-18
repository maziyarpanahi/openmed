"""Truth, geometry and determinism checks for the synthetic walker fixture (#4760).

The truth is checked three ways: for internal consistency (alternation, cadence,
asymmetry, double support), against the generated geometry (contact frames),
and against an independent pelvis-relative event detector (Zeni et al. 2008),
so a fixture change that keeps the arithmetic but moves the visible events
fails loudly.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import json
import math
import os
import random
import socket
import sys
from pathlib import Path

import pytest

from tests.fixtures.multimodal import pose_sequences
from tests.fixtures.multimodal.pose_sequences import (
    ASPECT,
    FAR_LEG_JOINTS,
    FEET,
    JOINTS,
    SCENARIOS,
    PoseFixtureError,
    WalkerSpec,
    gait_truth,
    scenario,
    synthetic_pose_sequence,
)

J = {name: index for index, name in enumerate(JOINTS)}
NOISELESS = sorted(set(SCENARIOS) - {"joint_jitter"})
ON_GROUND = 1e-5

# Any change to the generated output must be a deliberate update of these pins.
SCENARIO_SHA256 = {
    "far_leg_occlusion": "5f4fe1a8bb217e1c4a28eae02f5ab273d35e4bd44925f448aba46021d0cb938e",
    "joint_jitter": "4dbd2799c308e6fe0c58b16baf34db6f201d062e0f392e3f7a97a6de92ae2f0e",
    "older_adult_self_selected": (
        "1c9e454f76e0f381864758eef9419054cb9130e2a6a130d7b7a406cf2c07eeb9"
    ),
    "short_stride_low_lift": "ae06659e9cfa3c303389bf96da2309d9f29d4a74747984ce0b88796a1e23f6a5",
    "step_time_asymmetry": "e8f229fc80647181ec8d839732af0fe88fec53eaa2e35e05fa7d44b4692ae504",
    "two_walkers": "32d2c04aac922cd62c18ce106e33958cdedb3d97ac7598192b555b0c68f309fc",
    "typical_adult": "f112a17749e448418e7f5dcb42ee449adae62347a32cec4a48bf89023cc13355",
}


def _digest(fixture):
    payload = json.dumps(
        dataclasses.asdict(fixture), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_distance(a, b):
    """Distance with both axes in image-width units (pixels are square)."""
    return math.hypot(a[0] - b[0], (a[1] - b[1]) / ASPECT)


def _series(track, joint, axis):
    return [None if f[J[joint]] is None else f[J[joint]][axis] for f in track.frames]


def _zeni(track, foot):
    """Heel strike at max heel-minus-pelvis x, toe off at min toe-minus-pelvis x."""
    pelvis = _series(track, "left_hip", 0)
    heel = [h - p for h, p in zip(_series(track, f"{foot}_heel", 0), pelvis)]
    toe = [t - p for t, p in zip(_series(track, f"{foot}_foot_index", 0), pelvis)]
    inner = range(1, len(pelvis) - 1)
    strikes = [i for i in inner if heel[i - 1] <= heel[i] > heel[i + 1]]
    offs = [i for i in inner if toe[i - 1] >= toe[i] < toe[i + 1]]
    return strikes, offs


def _interior(events, foot, frame_count):
    return [i for f, i in events if f == foot and 2 <= i <= frame_count - 3]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_same_arguments_give_identical_output(name):
    assert scenario(name) == scenario(name)
    assert _digest(scenario(name)) == _digest(scenario(name))


def test_seed_drives_noise_only():
    assert scenario("joint_jitter", seed=1) != scenario("joint_jitter", seed=2)
    assert scenario("typical_adult", seed=1) == scenario("typical_adult", seed=2)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_events_are_ordered_and_alternate(name):
    for track in scenario(name).walkers:
        truth = track.truth
        for events in (truth.heel_strike_times_s, truth.toe_off_times_s):
            times = [t for _, t in events]
            assert times == sorted(times) and len(set(times)) == len(times)
            feet = [f for f, _ in events]
            assert all(a != b for a, b in zip(feet, feet[1:]))
        for foot in FEET:
            merged = sorted(
                [(t, "hs") for f, t in truth.heel_strike_times_s if f == foot]
                + [(t, "to") for f, t in truth.toe_off_times_s if f == foot]
            )
            kinds = [k for _, k in merged]
            assert all(a != b for a, b in zip(kinds, kinds[1:]))


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0])
@pytest.mark.parametrize("cadence", [80.0, 113.0, 140.0])
def test_cadence_matches_events_within_one_frame(fps, cadence):
    spec = WalkerSpec(cadence_spm=cadence)
    track = synthetic_pose_sequence(walkers=(spec,), fps=fps).walkers[0]
    frames = [i for _, i in track.truth.heel_strikes]
    expected = 60.0 * fps / cadence
    assert len(frames) >= 4
    assert all(abs((b - a) - expected) <= 1 for a, b in zip(frames, frames[1:]))
    mean = (frames[-1] - frames[0]) / (len(frames) - 1)
    assert abs(60.0 * fps / mean - cadence) <= cadence / expected


@pytest.mark.parametrize("asymmetry", [-0.15, 0.0, 0.1])
def test_step_time_asymmetry_keeps_stride_time(asymmetry):
    truth = gait_truth(WalkerSpec(asymmetry=asymmetry), fps=60.0, frame_count=240)
    strikes = truth.heel_strike_times_s
    left = [b - a for (fa, a), (fb, b) in zip(strikes, strikes[1:]) if fb == "left"]
    right = [b - a for (fa, a), (fb, b) in zip(strikes, strikes[1:]) if fb == "right"]
    assert all(math.isclose(s, truth.left_step_time_s, abs_tol=1e-5) for s in left)
    assert all(math.isclose(s, truth.right_step_time_s, abs_tol=1e-5) for s in right)
    assert math.isclose(
        truth.left_step_time_s + truth.right_step_time_s,
        truth.stride_time_s,
        abs_tol=1e-5,
    )


@pytest.mark.parametrize("name", NOISELESS)
def test_double_support_follows_from_events(name):
    for track in scenario(name, fps=60.0).walkers:
        truth = track.truth
        periods = []
        for foot, strike in truth.heel_strike_times_s:
            off = [t for f, t in truth.toe_off_times_s if f != foot and t > strike]
            if off and off[0] - strike < truth.stride_time_s / 2:
                periods.append(off[0] - strike)
        assert len(periods) >= 4
        for first, second in zip(periods, periods[1:]):
            total = (first + second) / truth.stride_time_s
            assert math.isclose(total, truth.double_support_fraction, abs_tol=1e-4)


@pytest.mark.parametrize("name", NOISELESS)
def test_contacts_in_the_geometry_match_truth(name):
    fixture = scenario(name, fps=60.0)
    for track in fixture.walkers:
        for foot in FEET:
            heel = _series(track, f"{foot}_heel", 1)
            toe = _series(track, f"{foot}_foot_index", 1)
            ground = max(y for y in heel if y is not None)
            for frame in _interior(track.truth.heel_strikes, foot, len(heel)):
                if heel[frame - 2] is None or heel[frame + 1] is None:
                    continue
                assert ground - heel[frame - 2] > ON_GROUND
                assert ground - heel[frame + 1] < ON_GROUND
            for frame in _interior(track.truth.toe_offs, foot, len(toe)):
                if toe[frame - 1] is None or toe[frame + 2] is None:
                    continue
                assert ground - toe[frame - 1] < ON_GROUND
                assert ground - toe[frame + 2] > ON_GROUND


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0, 120.0])
@pytest.mark.parametrize("name", [n for n in NOISELESS if n != "far_leg_occlusion"])
def test_independent_detector_recovers_truth(name, fps):
    fixture = scenario(name, fps=fps)
    for track in fixture.walkers:
        for foot in FEET:
            strikes, offs = _zeni(track, foot)
            for detected, events in (
                (strikes, track.truth.heel_strikes),
                (offs, track.truth.toe_offs),
            ):
                for frame in _interior(events, foot, fixture.frame_count):
                    assert min(abs(d - frame) for d in detected) <= 1


@pytest.mark.parametrize("name", NOISELESS)
def test_body_is_rigid_plausible_and_in_frame(name):
    for track in scenario(name).walkers:
        points = [p for frame in track.frames for p in frame if p is not None]
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)
        for side in FEET:
            pairs = [("hip", "knee"), ("knee", "ankle"), ("heel", "foot_index")]
            for a, b in pairs:
                lengths = [
                    _metric_distance(f[J[f"{side}_{a}"]], f[J[f"{side}_{b}"]])
                    for f in track.frames
                    if f[J[f"{side}_{a}"]] and f[J[f"{side}_{b}"]]
                ]
                assert max(lengths) - min(lengths) < 1e-5
            heel = _series(track, f"{side}_heel", 1)
            toe = _series(track, f"{side}_foot_index", 1)
            ground = max(y for y in heel if y is not None)
            assert all(y <= ground + ON_GROUND for y in heel + toe if y is not None)
            for frame in track.frames:
                hip, knee, ankle = (
                    frame[J[f"{side}_{j}"]] for j in ("hip", "knee", "ankle")
                )
                if knee and ankle:
                    cross = (ankle[0] - hip[0]) * (knee[1] - hip[1]) - (
                        ankle[1] - hip[1]
                    ) * (knee[0] - hip[0])
                    assert cross <= 1e-9  # knee flexes forward, never backwards


def test_random_valid_walkers_stay_physical_or_fail_closed():
    rng = random.Random(4760)
    realized = 0
    for _ in range(60):
        spec = WalkerSpec(
            cadence_spm=rng.uniform(30, 200),
            stride_length_m=rng.uniform(0.3, 2.0),
            stance_fraction=rng.uniform(0.55, 0.75),
            asymmetry=rng.uniform(-0.29, 0.29),
            height_m=rng.uniform(1.0, 2.2),
            swing_lift_m=rng.uniform(0.01, 0.15),
            start_phase=rng.uniform(0, 0.99),
        )
        fps = rng.choice([15.0, 25.0, 30.0, 60.0])
        try:
            fixture = synthetic_pose_sequence(
                walkers=(spec,), fps=fps, duration_s=240 / spec.cadence_spm + 1
            )
        except PoseFixtureError as error:
            assert str(error) == "pose_fixture_unreachable_foot"
            continue
        realized += 1
        track = fixture.walkers[0]
        points = [p for frame in track.frames for p in frame]
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)
        for foot in FEET:
            heel = _series(track, f"{foot}_heel", 1)
            toe = _series(track, f"{foot}_foot_index", 1)
            assert max(toe) <= max(heel) + ON_GROUND
            strikes, _ = _zeni(track, foot)
            for frame in _interior(track.truth.heel_strikes, foot, len(heel)):
                assert min(abs(d - frame) for d in strikes) <= 1
    assert realized >= 40


def test_far_leg_occlusion_hides_only_the_far_leg_in_mid_swing():
    track = scenario("far_leg_occlusion", fps=60.0).walkers[0]
    hidden = {
        (joint, i)
        for joint, start, stop in track.spec.occlusions
        for i in range(start, stop)
    }
    assert hidden and {joint for joint, _ in hidden} == set(FAR_LEG_JOINTS)
    for i, frame in enumerate(track.frames):
        for joint in JOINTS:
            assert (frame[J[joint]] is None) == ((joint, i) in hidden)
    events = track.truth.heel_strikes + track.truth.toe_offs
    assert not any(("right_heel", i) in hidden for f, i in events if f == "right")


def test_occlusion_never_shifts_the_noise_of_other_joints():
    spec = WalkerSpec()
    hidden = WalkerSpec(occlusions=(("left_knee", 10, 40),))
    plain = synthetic_pose_sequence(walkers=(spec,), noise_sd=0.003, seed=7)
    masked = synthetic_pose_sequence(walkers=(hidden,), noise_sd=0.003, seed=7)
    for a, b in zip(plain.walkers[0].frames, masked.walkers[0].frames):
        assert [p for p in b if p is not None] == [
            p for p, q in zip(a, b) if q is not None
        ]


def test_doubling_the_frame_rate_doubles_event_frames():
    slow = scenario("typical_adult", fps=30.0).walkers[0].truth
    fast = scenario("typical_adult", fps=60.0).walkers[0].truth
    slow_frames = {
        timed: frame
        for (_, frame), timed in zip(slow.heel_strikes, slow.heel_strike_times_s)
    }
    shared = 0
    for (_, frame), timed in zip(fast.heel_strikes, fast.heel_strike_times_s):
        if timed in slow_frames:
            shared += 1
            assert abs(frame - 2 * slow_frames[timed]) <= 1
    assert shared >= 4


def test_each_walker_keeps_its_own_truth():
    fixture = scenario("two_walkers")
    assert len(fixture.walkers) == 2
    for track in fixture.walkers:
        assert track.truth == gait_truth(
            track.spec, fps=fixture.fps, frame_count=fixture.frame_count
        )
    assert fixture.walkers[0].truth != fixture.walkers[1].truth


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"walkers": (WalkerSpec(cadence_spm=20.0),)}, "pose_fixture_invalid_walker"),
        ({"walkers": (WalkerSpec(asymmetry=0.3),)}, "pose_fixture_invalid_walker"),
        (
            {"walkers": (WalkerSpec(stance_fraction=0.5),)},
            "pose_fixture_invalid_walker",
        ),
        (
            {"walkers": (WalkerSpec(occlusions=(("tail", 0, 3),)),)},
            "pose_fixture_invalid_occlusion",
        ),
        (
            {"walkers": (WalkerSpec(cadence_spm=40.0),), "duration_s": 2.0},
            "pose_fixture_clip_shorter_than_stride",
        ),
        (
            {"walkers": (WalkerSpec(height_m=1.0, stride_length_m=2.0),)},
            "pose_fixture_unreachable_foot",
        ),
        ({"walkers": ()}, "pose_fixture_invalid_grid"),
        ({"fps": 0.0}, "pose_fixture_invalid_grid"),
        ({"noise_sd": -0.1}, "pose_fixture_invalid_noise"),
    ],
)
def test_unrealizable_parameters_fail_with_stable_codes(kwargs, code):
    with pytest.raises(PoseFixtureError, match=f"^{code}$"):
        synthetic_pose_sequence(**kwargs)
    with pytest.raises(PoseFixtureError, match="^pose_fixture_unknown_scenario$"):
        scenario("unknown")


def test_standard_library_only_and_no_files_or_network(monkeypatch):
    source = Path(pose_sequences.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            assert root == "__future__" or root in sys.stdlib_module_names

    def refuse(*_args, **_kwargs):
        raise AssertionError("fixture attempted I/O")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    for name in SCENARIOS:
        scenario(name)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_match_pinned_digests(name):
    assert _digest(scenario(name)) == SCENARIO_SHA256[name]
