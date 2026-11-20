"""Contract tests for privacy-safe pose sequences (#4669); synthetic values only."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from openmed.multimodal.frame_sampling_manifest import (
    FrameSamplingManifest,
    SamplingMode,
    canonical_coverage_digest,
)
from openmed.multimodal.pose.sequence import (
    CANONICAL_JOINTS,
    MAX_POSE_FRAMES,
    NON_DIAGNOSTIC_NOTICE,
    POSE_SCHEMA_VERSION,
    DepthKind,
    PoseContractError,
    PoseEstimatorProvenance,
    PoseFrame,
    PoseJointObservation,
    PoseSequence,
)
from tests.fixtures.multimodal.pose_sequences import JOINTS as GAIT_FIXTURE_JOINTS

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "parity"
    / "pose_sequence_v1.json"
)
SENTINEL = 0.7654321


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROVENANCE = PoseEstimatorProvenance(
    "synthetic-walker", _digest("revision"), _digest("terms")
)


def _manifest(ticks=(0, 10, 20), duration=30):
    return FrameSamplingManifest(
        duration_ticks=duration,
        time_base_num=1,
        time_base_den=30,
        sampling_mode=SamplingMode.KEYFRAME,
        timestamps=tuple(ticks),
        coverage_digest=canonical_coverage_digest(
            SamplingMode.KEYFRAME, 1, 30, tuple(ticks)
        ),
    )


def _joint(name="left_hip", x=0.5, y=0.5, z=None, visibility=0.9, presence=1.0):
    return PoseJointObservation(name, x, y, visibility, presence, z)


def _sequence(frames=None, depth_kind="none", manifest=None):
    if frames is None:
        frames = (PoseFrame(0, (_joint(),)), PoseFrame(10, (_joint("nose"),)))
    return PoseSequence(
        manifest=manifest or _manifest(),
        frames=frames,
        depth_kind=depth_kind,
        provenance=PROVENANCE,
    )


def _code(code):
    return pytest.raises(PoseContractError, match=f"^{code}$")


def test_vocabulary_is_closed_unique_and_covers_the_gait_fixture_joints():
    assert len(set(CANONICAL_JOINTS)) == len(CANONICAL_JOINTS)
    assert set(GAIT_FIXTURE_JOINTS) <= set(CANONICAL_JOINTS)
    for name in CANONICAL_JOINTS:
        if name.startswith("left_"):
            assert "right_" + name[5:] in CANONICAL_JOINTS


def test_valid_sequence_binds_the_manifest_digest_and_orders_joints():
    frame = PoseFrame(0, (_joint("right_ankle"), _joint("nose"), _joint("left_hip")))
    assert [j.joint for j in frame.joints] == ["nose", "left_hip", "right_ankle"]
    sequence = _sequence(frames=[frame])
    assert sequence.coverage_digest == _manifest().coverage_digest
    assert sequence.depth_kind is DepthKind.NONE
    assert sequence.notice == NON_DIAGNOSTIC_NOTICE


def test_report_lists_counts_notice_and_review_flag_only():
    frames = (
        PoseFrame(0, (_joint("nose"), _joint("left_hip"))),
        PoseFrame(10, (_joint("nose"),)),
        PoseFrame(20, ()),
    )
    report = _sequence(frames=frames).to_report()
    assert set(report) == {
        "schema_version",
        "notice",
        "frame_count",
        "joint_count",
        "missing_joint_counts",
        "review_required",
    }
    assert report["schema_version"] == POSE_SCHEMA_VERSION
    assert report["frame_count"] == 3
    assert report["joint_count"] == 2
    assert report["review_required"] is True
    missing = report["missing_joint_counts"]
    assert list(missing) == list(CANONICAL_JOINTS)
    assert missing["nose"] == 1 and missing["left_hip"] == 2 and missing["neck"] == 3
    assert json.loads(_sequence(frames=frames).to_json()) == report


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"name": "left_pinky"}, "unknown_joint"),
        ({"name": "Left_Hip"}, "unknown_joint"),
        ({"name": 11}, "unknown_joint"),
        ({"x": -0.0001}, "coordinate_invalid"),
        ({"x": 1.0001}, "coordinate_invalid"),
        ({"y": 2}, "coordinate_invalid"),
        ({"x": math.nan}, "coordinate_invalid"),
        ({"y": math.inf}, "coordinate_invalid"),
        ({"x": True}, "coordinate_invalid"),
        ({"x": "0.5"}, "coordinate_invalid"),
        ({"x": None}, "coordinate_invalid"),
        ({"x": 10**400}, "coordinate_invalid"),
        ({"visibility": 1.01}, "confidence_invalid"),
        ({"presence": -0.1}, "confidence_invalid"),
        ({"presence": math.nan}, "confidence_invalid"),
        ({"z": math.nan}, "depth_invalid"),
        ({"z": math.inf}, "depth_invalid"),
        ({"z": False}, "depth_invalid"),
    ],
)
def test_invalid_observations_fail_with_stable_value_free_codes(kwargs, code):
    with _code(code) as caught:
        _joint(**kwargs)
    assert caught.value.__cause__ is None


def test_frame_rules():
    with _code("duplicate_joint"):
        PoseFrame(0, (_joint("nose"), _joint("nose", x=0.1)))
    for tick in (-1, 1.0, True, "0", None):
        with _code("tick_invalid"):
            PoseFrame(tick, ())
    with _code("joints_invalid"):
        PoseFrame(0, ({"joint": "nose"},))
    with _code("joints_invalid"):
        PoseFrame(0, None)


def test_sequence_rules():
    with _code("tick_not_in_manifest"):
        _sequence(frames=(PoseFrame(5, ()),))
    with _code("ticks_not_increasing"):
        _sequence(frames=(PoseFrame(10, ()), PoseFrame(0, ())))
    with _code("ticks_not_increasing"):
        _sequence(frames=(PoseFrame(10, ()), PoseFrame(10, ())))
    with _code("frames_invalid"):
        _sequence(frames=())
    with _code("frames_invalid"):
        _sequence(frames=("frame",))
    with _code("manifest_invalid"):
        PoseSequence(_manifest().to_dict(), (PoseFrame(0, ()),), "none", PROVENANCE)
    with _code("depth_kind_invalid"):
        _sequence(depth_kind="stereo")
    with _code("provenance_invalid"):
        PoseSequence(_manifest(), (PoseFrame(0, ()),), "none", "synthetic-walker")


def test_frame_count_is_bounded():
    assert MAX_POSE_FRAMES == 100_000
    ticks = tuple(range(MAX_POSE_FRAMES + 1))
    manifest = _manifest(ticks=ticks, duration=len(ticks))
    with _code("frames_invalid"):
        _sequence(frames=[PoseFrame(t, ()) for t in ticks], manifest=manifest)


@pytest.mark.parametrize(
    ("depth_kind", "z", "code"),
    [
        ("none", 0.0, "depth_not_declared"),
        ("relative", 10.5, "depth_invalid"),
        ("relative", -10.5, "depth_invalid"),
        ("metric", 100.5, "depth_invalid"),
    ],
)
def test_depth_must_match_the_declared_kind(depth_kind, z, code):
    with _code(code):
        _sequence(frames=(PoseFrame(0, (_joint(z=z),)),), depth_kind=depth_kind)


@pytest.mark.parametrize("depth_kind", ["relative", "metric"])
def test_declared_depth_accepts_values_and_missing_depth(depth_kind):
    frames = (PoseFrame(0, (_joint(z=-0.25), _joint("nose"))),)
    assert _sequence(frames=frames, depth_kind=depth_kind).frames == frames


@pytest.mark.parametrize(
    ("estimator_id", "revision", "terms"),
    [
        ("BlazePose", _digest("r"), _digest("t")),
        ("/models/pose.task", _digest("r"), _digest("t")),
        ("", _digest("r"), _digest("t")),
        ("a" * 65, _digest("r"), _digest("t")),
        ("blazepose", _digest("r").upper(), _digest("t")),
        ("blazepose", _digest("r")[:63], _digest("t")),
        ("blazepose", _digest("r"), None),
    ],
)
def test_provenance_is_digest_only(estimator_id, revision, terms):
    with _code("provenance_invalid"):
        PoseEstimatorProvenance(estimator_id, revision, terms)


def test_sentinel_coordinates_never_appear_in_repr_reports_or_errors():
    joint = _joint(x=SENTINEL, y=SENTINEL, z=SENTINEL, visibility=SENTINEL)
    sequence = _sequence(frames=(PoseFrame(0, (joint,)),), depth_kind="relative")
    surfaces = [
        repr(joint),
        repr(PoseFrame(0, (joint,))),
        repr(sequence),
        str(sequence),
        sequence.to_json(),
        str(sequence.to_report()),
    ]
    attempts = (
        lambda: _joint(x=SENTINEL + 1),
        lambda: _joint(visibility=SENTINEL + 1),
        lambda: _sequence(
            frames=(PoseFrame(0, (_joint(z=SENTINEL * 100),)),), depth_kind="relative"
        ),
        lambda: _sequence(frames=(PoseFrame(0, (_joint(z=SENTINEL),)),)),
        lambda: sequence.require_reviewer_confirmation(),
    )
    for attempt in attempts:
        with pytest.raises(PoseContractError) as caught:
            attempt()
        surfaces += [str(caught.value), repr(caught.value)]
    assert all("7654" not in surface for surface in surfaces)


def test_reviewer_confirmation_fails_closed():
    sequence = _sequence()
    for value in (False, None, 1, "yes"):
        with _code("review_required"):
            sequence.require_reviewer_confirmation(confirmed=value)
    sequence.require_reviewer_confirmation(confirmed=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("frames"),
        lambda r: r.update(device="synthetic"),
        lambda r: r["provenance"].update(path="synthetic"),
        lambda r: r["frames"][0].update(timestamp=1),
        lambda r: r["frames"][0]["joints"][0].update(label="synthetic"),
        lambda r: r["frames"][0]["joints"][0].pop("presence"),
        lambda r: r["manifest"].update(sampling_mode="random"),
        lambda r: r["manifest"].update(frames=[]),
    ],
)
def test_records_with_unknown_or_missing_keys_fail_closed(mutate):
    record = _sequence().to_record()
    mutate(record)
    with pytest.raises(PoseContractError):
        PoseSequence.from_record(record)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda r: r.update(depth_kind="synthetic-secret"), "depth_kind_invalid"),
        (
            lambda r: r["manifest"].update(sampling_mode="synthetic-secret"),
            "manifest_invalid",
        ),
        (lambda r: r["manifest"].update(synthetic_secret=1), "manifest_invalid"),
    ],
)
def test_replaced_errors_keep_no_raw_exception_context(mutate, code):
    record = _sequence().to_record()
    mutate(record)
    with pytest.raises(PoseContractError, match=f"^{code}$") as caught:
        PoseSequence.from_record(record)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "synthetic" not in repr(caught.value)


@pytest.mark.parametrize(
    ("limits", "frame_count", "joints_per_frame", "code"),
    [
        ({"MAX_POSE_FRAMES": 2}, 3, 1, "frames_invalid"),
        ({}, 1, 50_000, "joints_invalid"),
        ({"MAX_OBSERVATIONS": 3}, 2, 2, "resource_limit"),
    ],
)
def test_oversized_records_fail_before_any_observation_is_built(
    monkeypatch, limits, frame_count, joints_per_frame, code
):
    from openmed.multimodal.pose import sequence as module

    record = _sequence().to_record()
    joint = record["frames"][0]["joints"][0]
    record["frames"] = [
        {"tick": tick, "joints": [dict(joint)] * joints_per_frame}
        for tick in range(frame_count)
    ]
    for name, value in limits.items():
        monkeypatch.setattr(module, name, value)
    built = []
    monkeypatch.setattr(
        module, "PoseJointObservation", lambda **fields: built.append(fields)
    )
    with pytest.raises(PoseContractError, match=f"^{code}$"):
        PoseSequence.from_record(record)
    assert built == []


def _encode(document):
    head = {key: value for key, value in document.items() if key != "cases"}
    lines = [
        json.dumps(case, sort_keys=True, separators=(",", ":"))
        for case in document["cases"]
    ]
    tail = json.dumps(head, sort_keys=True, separators=(",", ":"))[1:]
    return '{"cases":[\n' + ",\n".join(lines) + "\n]," + tail + "\n"


def test_parity_fixture_round_trips_byte_identically():
    text = FIXTURE.read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["synthetic"] is True
    assert document["schema_version"] == POSE_SCHEMA_VERSION
    rebuilt = []
    for case in document["cases"]:
        if "error" in case:
            with pytest.raises(PoseContractError) as caught:
                PoseSequence.from_record(case["record"])
            assert str(caught.value) == case["error"], case["name"]
            rebuilt.append(case)
            continue
        sequence = PoseSequence.from_record(case["record"])
        assert sequence.to_json() == case["report_json"], case["name"]
        rebuilt.append(
            {
                "name": case["name"],
                "record": sequence.to_record(),
                "report_json": sequence.to_json(),
            }
        )
    assert _encode({**document, "cases": rebuilt}) == text
    codes = {case["error"] for case in document["cases"] if "error" in case}
    assert {"tick_not_in_manifest", "coordinate_invalid", "duplicate_joint"} <= codes
    assert "unknown_joint" in codes


def test_random_observations_fail_with_a_code_or_keep_every_invariant():
    rng = random.Random(4669)
    numbers = (-0.5, 0.0, 0.5, 1.0, 1.5, 50.0, math.nan, math.inf, True, None)
    realized = 0
    for _ in range(2000):
        joints = []
        for _ in range(rng.randrange(0, 4)):
            joints.append(
                {
                    "joint": rng.choice((*CANONICAL_JOINTS[:4], "left_pinky")),
                    "x": rng.choice(numbers) if rng.random() < 0.1 else rng.random(),
                    "y": rng.random(),
                    "visibility": rng.random(),
                    "presence": rng.choice(numbers)
                    if rng.random() < 0.1
                    else rng.random(),
                }
            )
            if rng.random() < 0.3:
                joints[-1]["z"] = rng.choice(numbers)
        record = _sequence().to_record()
        record["depth_kind"] = rng.choice(("none", "relative", "metric"))
        record["frames"] = [{"tick": rng.choice((0, 5, 10, 20)), "joints": joints}]
        try:
            sequence = PoseSequence.from_record(record)
        except PoseContractError as error:
            assert str(error).isidentifier()
            assert error.__cause__ is None
            continue
        realized += 1
        assert PoseSequence.from_record(sequence.to_record()) == sequence
        for joint in sequence.frames[0].joints:
            assert 0 <= joint.x <= 1 and 0 <= joint.y <= 1
            assert 0 <= joint.visibility <= 1 and 0 <= joint.presence <= 1
            if joint.z is not None:
                assert sequence.depth_kind is not DepthKind.NONE
                assert math.isfinite(joint.z) and abs(joint.z) <= 100
        report = sequence.to_report()
        assert report["frame_count"] == 1
        assert sum(report["missing_joint_counts"].values()) == len(
            CANONICAL_JOINTS
        ) - len(sequence.frames[0].joints)
    assert realized >= 100
