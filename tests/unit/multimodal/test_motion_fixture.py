"""Truth, separability and determinism checks for the accelerometer fixture (#4761).

Each window's truth is checked against its own samples (impact on the onset
sample, stillness and posture afterwards, gaps), and each scenario is run
through small re-implementations of published threshold detectors. They pin
which detector stage every confounder defeats, so a change that makes a
confounder trivially separable from a fall, or a fall invisible, fails loudly.
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

from tests.fixtures.multimodal import motion
from tests.fixtures.multimodal.motion import (
    KINDS,
    SCENARIOS,
    MotionFixtureError,
    MotionSpec,
    scenario,
    synthetic_motion_window,
)

RATES = (10.0, 25.0, 50.0, 100.0, 200.0)
EVENT_SCENARIOS = sorted(n for n, s in SCENARIOS.items() if s.kind in KINDS[2:])
FALLS = sorted(n for n, s in SCENARIOS.items() if s.kind == "fall")
NO_FALLS = sorted(n for n, s in SCENARIOS.items() if s.kind != "fall")
POSTURE_AXIS = {
    "upright": (1, 1.0),
    "supine": (2, 1.0),
    "prone": (2, -1.0),
    "left_side": (0, 1.0),
    "right_side": (0, -1.0),
    "device_flat": (2, 1.0),
}

# Any change to the generated output must be a deliberate update of these pins.
SCENARIO_SHA256 = {
    "device_drop": "673562811623c2b1175db449f83493401a783b731f1520cae5430fc6fc783030",
    "fall_backward": "75148a97cb9617c9989f145f6eff9e5dcebbc6d71904ab17ef70830ad1999021",
    "fall_forward": "7eb04b48a81e46ff3d2bcaa3ace4e253d58d14f233b31262282fd388e337a9c6",
    "fall_lateral": "0dc523d6e6bac9f7844a9088b655182e560effeb5daef5e4d3e8b014d9bf6363",
    "fall_with_gap": "ea2c47c5462252a5c64bc32369aa8aa0026833095f4162b92933f47a7424f738",
    "fast_sit_down": "b41f47eb586fc3bb18b42b47cd42105c5f2ddfa49d8295b05335fa3250736e3f",
    "jump": "d74246713a749b9935e567c8dc757d82f69e2d07766223a47625b577856503f4",
    "lying_down": "30a07d88b210f004a2c5a35bbb4d7cc226efb16b2dab6297dd16c7055080dd8c",
    "soft_fall": "fe1cbbe4782cd1940c0bc6363bfa0e4ab638f767cbe51fe34d07c94225be5e83",
    "standing": "447d59a73804e50a68b845f34bf4f03434efba6a3c3b7344e887270139a1f465",
    "stumble_recover": "25e7455f0a009a643b1e99f8137d20e33c56e07a26103aff64aff410dca0f03f",
    "walking": "8f7938b8c91b04800bb98453a9f2c58213aeecc7bc6d12004bda6670bdca2292",
}


def _digest(window):
    payload = json.dumps(
        dataclasses.asdict(window), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _svm(samples):
    return [None if s is None else math.sqrt(sum(v * v for v in s)) for s in samples]


def _noise_bound(spec):
    return 3.0 * spec.noise_g * math.sqrt(3.0)


# Witness detectors, re-implemented from the cited papers for these tests only.


def _impact(samples, rate, upper=1.79):
    """Peak magnitude above a single upper threshold."""
    return any(m is not None and m > upper for m in _svm(samples))


def _bourke(samples, rate, lower, upper, edge_s=0.6):
    """A trough below ``lower`` followed by a peak above ``upper`` within 0.6 s."""
    last_low = None
    for index, m in enumerate(_svm(samples)):
        if m is None:
            continue
        if m < lower:
            last_low = index
        if m > upper and last_low is not None and index - last_low <= edge_s * rate:
            return True
    return False


def _still(magnitudes, threshold=0.1):
    present = [m for m in magnitudes if m is not None]
    return bool(present) and max(present) - min(present) < threshold


def _impact_then_still(samples, rate, upper=2.0):
    """Impact above ``upper`` followed by 1 s to 3 s of inactivity (no posture)."""
    svm = _svm(samples)
    for index, m in enumerate(svm):
        if m is not None and m > upper:
            after = svm[index + int(rate) : index + int(3 * rate)]
            if len(after) >= rate and _still(after):
                return True
    return False


def _kangas(samples, rate, start=0.6, upper=2.0, lying=0.5):
    """Trough below 0.6 g, impact within 1 s, lying 2 s after the impact."""
    svm = _svm(samples)
    for index, m in enumerate(svm):
        if m is None or m >= start:
            continue
        window = range(index, min(len(svm), index + int(rate)))
        peaks = [(svm[j], j) for j in window if svm[j] is not None and svm[j] > upper]
        if not peaks:
            continue
        impact = max(peaks)[1]
        posture = samples[impact + int(2 * rate) : impact + int(2.4 * rate)]
        vertical = [s[1] for s in posture if s is not None]
        if vertical and sum(vertical) / len(vertical) <= lying:
            return True
    return False


def _long_lie(samples, rate, seconds=3.0):
    """Vertical axis within 0.5 g of zero and inactive for ``seconds`` (no impact)."""
    svm = _svm(samples)
    width = int(seconds * rate)
    for start in range(0, len(samples) - width, max(1, int(rate / 4))):
        rows = [i for i in range(start, start + width) if samples[i] is not None]
        if len(rows) > width // 2 and all(abs(samples[i][1]) <= 0.5 for i in rows):
            if _still([svm[i] for i in rows]):
                return True
    return False


DETECTORS = {
    "long_lie": _long_lie,
    "impact_1_79": _impact,
    "bourke_0_41_3_52": lambda s, r: _bourke(s, r, 0.41, 3.52),
    "bourke_0_73_1_79": lambda s, r: _bourke(s, r, 0.73, 1.79),
    "impact_then_still": _impact_then_still,
    "kangas": _kangas,
}
FALL_ROW = dict.fromkeys(DETECTORS, True)
QUIET_ROW = dict.fromkeys(DETECTORS, False)
EXPECTED = {
    "standing": QUIET_ROW,
    "walking": QUIET_ROW,
    "fall_backward": FALL_ROW,
    "fall_forward": FALL_ROW,
    "fall_lateral": FALL_ROW,
    "device_drop": FALL_ROW,  # the confounder no published stage rejects
    "soft_fall": {
        **FALL_ROW,
        "bourke_0_41_3_52": False,
        "kangas": False,
    },
    "fast_sit_down": {
        **QUIET_ROW,
        "impact_1_79": True,
        "bourke_0_73_1_79": True,
        "impact_then_still": True,
    },
    "lying_down": {**QUIET_ROW, "long_lie": True},
    "stumble_recover": {**QUIET_ROW, "impact_1_79": True},
    "jump": {**FALL_ROW, "long_lie": False, "kangas": False},
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_same_spec_gives_identical_output(name):
    assert scenario(name) == scenario(name)


def test_seed_drives_noise_only():
    assert scenario("fall_forward", seed=1) != scenario("fall_forward", seed=2)
    quiet = scenario("fall_forward", noise_g=0.0, seed=1)
    assert quiet == dataclasses.replace(
        scenario("fall_forward", noise_g=0.0, seed=2),
        spec=quiet.spec,
    )


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("name", EVENT_SCENARIOS)
def test_onset_sample_carries_the_requested_peak(name, rate):
    for seed in range(5):
        window = scenario(name, rate_hz=rate, seed=seed)
        truth = window.truth
        assert truth.onset_index == round(window.spec.onset_s * rate)
        if truth.onset_in_gap:
            continue
        observed = _svm(window.samples)[truth.onset_index]
        assert (
            truth.peak_g
            <= observed
            <= truth.peak_g + 2 * _noise_bound(window.spec) + 1e-4
        )
    clean = scenario(name, rate_hz=rate, noise_g=0.0)
    svm = _svm(clean.samples)
    present = [m for m in svm if m is not None]
    assert svm[clean.truth.onset_index] == max(present)


@pytest.mark.parametrize("name", NO_FALLS)
def test_no_fall_windows_stay_below_the_default_fall_impact(name):
    fall_peak = MotionSpec(kind="fall").peak_g
    for rate in RATES:
        window = scenario(name, rate_hz=rate)
        assert not window.truth.is_fall
        assert max(m for m in _svm(window.samples) if m is not None) < fall_peak


@pytest.mark.parametrize("name", EVENT_SCENARIOS)
def test_post_event_stillness_and_posture_match_truth(name):
    window = scenario(name)
    truth = window.truth
    axis, sign = POSTURE_AXIS[truth.posture_after]
    if truth.still_from_index is None:
        assert truth.kind == "stumble_recover"
        assert not _still(_svm(window.samples)[truth.onset_index + 100 :], 0.2)
        return
    rest = [s for s in window.samples[truth.still_from_index :] if s is not None]
    assert len(rest) >= 2 * window.rate_hz
    svm = _svm(rest)
    assert max(svm) - min(svm) <= 2 * _noise_bound(window.spec)
    assert all(abs(s[axis] - sign) <= 0.1 for s in rest)


def test_confounders_differ_from_falls_in_their_stated_feature():
    rate = 100.0
    drop = scenario("device_drop", rate_hz=rate)
    before = _svm(drop.samples)[: drop.truth.onset_index]
    assert sum(m < 0.2 for m in before) >= 0.4 * rate  # true free fall
    for name in FALLS:
        fall = scenario(name, rate_hz=rate)
        before = _svm(fall.samples)[: fall.truth.onset_index]
        assert min(before) >= 0.3  # a body never reaches free fall at the waist
    widths = {}
    for name in ("device_drop", "fall_backward"):
        window = scenario(name, rate_hz=1000.0, noise_g=0.0)
        svm = _svm(window.samples)
        half = 1 + (svm[window.truth.onset_index] - 1) / 2
        widths[name] = sum(m > half for m in svm) / 1000.0
    assert widths["device_drop"] < 0.02 <= 0.04 <= widths["fall_backward"]
    for name in ("fast_sit_down", "jump", "stumble_recover"):
        assert scenario(name).truth.posture_after == "upright"
    lying = scenario("lying_down")
    assert max(_svm(lying.samples)) < 1.79


@pytest.mark.parametrize("rate", (25.0, 100.0))
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_published_detectors_see_the_intended_traps(name, rate):
    for seed in range(3):
        samples = scenario(name, rate_hz=rate, seed=seed).samples
        seen = {key: bool(detect(samples, rate)) for key, detect in DETECTORS.items()}
        assert seen == EXPECTED[name]


def test_every_confounder_fools_one_stage_and_fails_another():
    for name in NO_FALLS:
        if name in ("standing", "walking", "device_drop"):
            continue
        row = EXPECTED[name]
        assert any(row.values()) and not all(row.values()), name


def test_gaps_hide_exactly_the_requested_samples():
    window = scenario("fall_with_gap")
    assert window.truth.gaps == ((500, 700),)
    assert not window.truth.onset_in_gap
    hidden = [i for i, s in enumerate(window.samples) if s is None]
    assert hidden == list(range(500, 700))
    plain = scenario("fall_with_gap", gaps=())
    assert all(
        a == b or a is None for a, b in zip(window.samples, plain.samples, strict=True)
    )
    covered = scenario("fall_backward", gaps=((3.9, 4.1),))
    assert covered.truth.onset_in_gap
    assert covered.samples[covered.truth.onset_index] is None


def test_doubling_the_rate_doubles_the_onset_index():
    for name in EVENT_SCENARIOS:
        slow = scenario(name, rate_hz=50.0).truth
        fast = scenario(name, rate_hz=100.0).truth
        assert fast.onset_index == 2 * slow.onset_index
        assert fast.onset_time_s == slow.onset_time_s


def test_readings_never_exceed_the_sensor_range():
    window = scenario("device_drop", range_g=4.0, peak_g=3.8, noise_g=0.005)
    assert all(abs(v) <= 4.0 for s in window.samples for v in s)


def test_random_specs_keep_their_truth_or_fail_closed():
    rng = random.Random(4761)
    realized = 0
    for _ in range(300):
        spec = MotionSpec(
            kind=rng.choice(KINDS),
            rate_hz=rng.choice((1.0, 5.0, 10.0, 12.5, 25, 50.0, 60.0, 100, 1000.0)),
            duration_s=rng.uniform(2.0, 15.0),
            onset_s=rng.uniform(0.0, 14.0),
            peak_g=rng.uniform(0.8, 6.0),
            impact_width_s=rng.uniform(0.005, 0.3),
            trough_g=rng.uniform(0.0, 0.99),
            descent_s=rng.uniform(0.1, 3.0),
            direction=rng.choice(("backward", "forward", "left", "right")),
            cadence_spm=rng.uniform(30.0, 200.0),
            noise_g=rng.choice((0.0, 0.005, 0.02, 0.1)),
            range_g=rng.choice((4.0, 8.0, 16.0, 16.0)),
            seed=rng.randrange(1000),
        )
        try:
            window = synthetic_motion_window(spec)
        except MotionFixtureError as error:
            assert str(error).startswith("motion_fixture_")
            continue
        realized += 1
        assert all(
            math.isfinite(v) and abs(v) <= spec.range_g
            for s in window.samples
            for v in s
        )
        truth = window.truth
        if truth.onset_index is None:
            continue
        svm = _svm(window.samples)
        assert svm[truth.onset_index] >= spec.peak_g
        bound = 2 * _noise_bound(spec) + 1e-4
        assert max(svm) <= svm[truth.onset_index] + bound
        assert truth.onset_index / spec.rate_hz >= spec.descent_s
        if truth.still_from_index is not None:
            rest = svm[truth.still_from_index :]
            assert len(rest) >= 2 * spec.rate_hz - 1
            assert max(rest) - min(rest) <= bound
    assert realized >= 60


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"kind": "running"}, "motion_fixture_unknown_kind"),
        ({"direction": "up"}, "motion_fixture_unknown_direction"),
        ({"peak_g": math.nan}, "motion_fixture_non_finite"),
        ({"rate_hz": 0.0}, "motion_fixture_invalid_parameter"),
        ({"trough_g": 1.0}, "motion_fixture_invalid_parameter"),
        ({"noise_g": -0.01}, "motion_fixture_invalid_parameter"),
        ({"duration_s": 0.01}, "motion_fixture_too_few_samples"),
        ({"peak_g": 15.99}, "motion_fixture_peak_above_range"),
        ({"onset_s": 12.0}, "motion_fixture_onset_outside_window"),
        ({"onset_s": 0.5}, "motion_fixture_no_room_before_event"),
        ({"onset_s": 8.0}, "motion_fixture_no_room_after_event"),
        ({"peak_g": 1.0}, "motion_fixture_peak_not_above_gravity"),
        ({"peak_g": 1.2}, "motion_fixture_onset_not_peak"),
        ({"gaps": ((6.0, 5.0),)}, "motion_fixture_invalid_gap"),
        ({"gaps": ((1.0, 3.0), (2.0, 4.0))}, "motion_fixture_invalid_gap"),
        ({"gaps": ((9.0, 11.0),)}, "motion_fixture_invalid_gap"),
        ({"gaps": ((1.0,),)}, "motion_fixture_invalid_gap"),
        ({"gaps": [(1.0, 2.0)]}, "motion_fixture_invalid_gap"),
        ({"gaps": ((1.0, math.inf),)}, "motion_fixture_non_finite"),
        ({"kind": ["fall"]}, "motion_fixture_unknown_kind"),
        ({"direction": {}}, "motion_fixture_unknown_direction"),
        ({"rate_hz": True}, "motion_fixture_invalid_parameter"),
        ({"rate_hz": "100"}, "motion_fixture_invalid_parameter"),
        ({"rate_hz": 0.5}, "motion_fixture_invalid_parameter"),
        ({"rate_hz": 1001.0}, "motion_fixture_invalid_parameter"),
        ({"rate_hz": -100.0}, "motion_fixture_invalid_parameter"),
        ({"rate_hz": math.inf}, "motion_fixture_non_finite"),
        ({"duration_s": -10.0}, "motion_fixture_invalid_parameter"),
        ({"duration_s": 601.0}, "motion_fixture_invalid_parameter"),
        ({"impact_width_s": 0.0}, "motion_fixture_invalid_parameter"),
        ({"impact_width_s": 0.31}, "motion_fixture_invalid_parameter"),
        ({"descent_s": 0.0}, "motion_fixture_invalid_parameter"),
        ({"cadence_spm": 201.0}, "motion_fixture_invalid_parameter"),
        ({"range_g": 0.0}, "motion_fixture_invalid_parameter"),
        ({"noise_g": False}, "motion_fixture_invalid_parameter"),
        ({"seed": 1.5}, "motion_fixture_invalid_seed"),
        ({"seed": True}, "motion_fixture_invalid_seed"),
        ({"seed": "0"}, "motion_fixture_invalid_seed"),
    ],
)
def test_parameters_without_exact_truth_fail_with_stable_codes(overrides, code):
    with pytest.raises(MotionFixtureError, match=f"^{code}$"):
        scenario("fall_backward", **overrides)


@pytest.mark.parametrize("name", ("jump", "device_drop"))
def test_impact_wider_than_the_flight_fails_closed(name):
    with pytest.raises(MotionFixtureError, match="^motion_fixture_no_free_fall$"):
        scenario(name, impact_width_s=0.3, descent_s=0.3)


def test_unknown_override_fails_closed():
    with pytest.raises(MotionFixtureError, match="^motion_fixture_unknown_override$"):
        scenario("fall_backward", sample_rate=100.0)


@pytest.mark.parametrize("name", ["unknown", "", None, ["fall_backward"], {}])
def test_unknown_scenario_fails_closed(name):
    with pytest.raises(MotionFixtureError, match="^motion_fixture_unknown_scenario$"):
        scenario(name)


def test_standard_library_only_and_no_files_or_network(monkeypatch):
    source = Path(motion.__file__).read_text(encoding="utf-8")
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
