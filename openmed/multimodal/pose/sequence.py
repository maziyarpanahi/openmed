"""Privacy-safe pose keypoint sequences bound to a frame-sampling manifest.

A ``PoseSequence`` holds per-frame body landmarks in a closed canonical joint
vocabulary, keyed by the integer ticks of a ``FrameSamplingManifest``. It reads
no files, runs no estimator and computes no metric. Coordinates are sensitive
biometric data: they stay out of ``repr``, reports and errors, and estimator
provenance is digest-only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn, TypeVar

from ..frame_sampling_manifest import FrameSamplingManifest, SamplingMode

__all__ = [
    "CANONICAL_JOINTS",
    "MAX_OBSERVATIONS",
    "MAX_POSE_FRAMES",
    "MAX_RELATIVE_DEPTH",
    "NON_DIAGNOSTIC_NOTICE",
    "POSE_SCHEMA_VERSION",
    "DepthKind",
    "PoseContractError",
    "PoseEstimatorProvenance",
    "PoseFrame",
    "PoseJointObservation",
    "PoseSequence",
]

POSE_SCHEMA_VERSION: Final = "openmed.multimodal.pose.sequence.v1"
NON_DIAGNOSTIC_NOTICE: Final = (
    "Pose landmarks only; not gait, balance, fall-risk or clinical assessment. "
    "Consequential use requires explicit reviewer confirmation."
)
CANONICAL_JOINTS: Final = (
    "nose",
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "spine",
    "pelvis",
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
MAX_POSE_FRAMES: Final = 100_000
MAX_OBSERVATIONS: Final = 1_000_000
MAX_RELATIVE_DEPTH: Final = 10.0
_METRIC_DEPTH_LIMIT_M: Final = 100.0
_E = TypeVar("_E", bound=Enum)
_JOINT_ORDER = {name: index for index, name in enumerate(CANONICAL_JOINTS)}
_ESTIMATOR_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PoseContractError(ValueError):
    """Stable, value-free rejection code; never echoes a coordinate."""


class DepthKind(str, Enum):
    """What ``z`` means: absent, estimator-scaled, or metres from an origin."""

    NONE = "none"
    RELATIVE = "relative"
    METRIC = "metric"


def _reject(code: str) -> NoReturn:
    raise PoseContractError(code)


def _member(enum: type[_E], value: object) -> _E | None:
    """Return the enum member for ``value``, or ``None`` if there is none.

    Callers raise outside the handler so no exception carrying the raw value
    survives, not even as ``__context__``.
    """
    try:
        return enum(value)
    except (TypeError, ValueError):
        return None


def _build_manifest(fields: dict[str, Any]) -> FrameSamplingManifest | None:
    """Build the manifest, or return ``None`` for unknown or missing fields.

    The ``TypeError`` names the offending key, so it is dropped here rather
    than chained to the value-free contract error.
    """
    try:
        return FrameSamplingManifest(**fields)
    except TypeError:
        return None


def _finite(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:
        return False


def _unit_interval(value: object, code: str) -> float:
    if not _finite(value) or not 0 <= value <= 1:  # type: ignore[operator]
        _reject(code)
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, repr=False)
class PoseJointObservation:
    """One canonical joint in one frame, in normalized image coordinates.

    Args:
        joint: A name from ``CANONICAL_JOINTS``; estimator-specific names must
            be mapped first.
        x: Horizontal position, 0 at the left image edge and 1 at the right.
        y: Vertical position, 0 at the top edge and 1 at the bottom.
        visibility: Estimator confidence that the joint is unoccluded, 0 to 1.
        presence: Estimator confidence that the joint is in the frame, 0 to 1.
        z: Optional depth, interpreted by the sequence's ``DepthKind``.

    A joint the estimator did not return is left out of the frame rather than
    given placeholder coordinates.
    """

    joint: str
    x: float = field(repr=False)
    y: float = field(repr=False)
    visibility: float = field(repr=False)
    presence: float = field(repr=False)
    z: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.joint) is not str or self.joint not in _JOINT_ORDER:
            _reject("unknown_joint")
        object.__setattr__(self, "x", _unit_interval(self.x, "coordinate_invalid"))
        object.__setattr__(self, "y", _unit_interval(self.y, "coordinate_invalid"))
        object.__setattr__(
            self, "visibility", _unit_interval(self.visibility, "confidence_invalid")
        )
        object.__setattr__(
            self, "presence", _unit_interval(self.presence, "confidence_invalid")
        )
        if self.z is not None:
            if not _finite(self.z):
                _reject("depth_invalid")
            object.__setattr__(self, "z", float(self.z))

    def __repr__(self) -> str:
        return f"PoseJointObservation(joint={self.joint})"


@dataclass(frozen=True, slots=True, repr=False)
class PoseFrame:
    """The observed joints of one sampled frame, keyed by a manifest tick.

    Joints are stored in ``CANONICAL_JOINTS`` order; each appears at most once.
    """

    tick: int
    joints: tuple[PoseJointObservation, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            _reject("tick_invalid")
        if type(self.joints) not in (tuple, list) or len(self.joints) > len(
            CANONICAL_JOINTS
        ):
            _reject("joints_invalid")
        if any(type(joint) is not PoseJointObservation for joint in self.joints):
            _reject("joints_invalid")
        names = [joint.joint for joint in self.joints]
        if len(set(names)) != len(names):
            _reject("duplicate_joint")
        ordered = sorted(self.joints, key=lambda joint: _JOINT_ORDER[joint.joint])
        object.__setattr__(self, "joints", tuple(ordered))

    def __repr__(self) -> str:
        return f"PoseFrame(joint_count={len(self.joints)})"


@dataclass(frozen=True, slots=True)
class PoseEstimatorProvenance:
    """Digest-only record of which estimator produced the landmarks.

    Args:
        estimator_id: Lowercase slug such as ``blazepose-ghum-3d``; no paths,
            device names or user text.
        revision_digest: Lowercase SHA-256 of the model or library revision.
        terms_digest: Lowercase SHA-256 of the licence or terms the caller
            accepted for that revision.
    """

    estimator_id: str
    revision_digest: str
    terms_digest: str

    def __post_init__(self) -> None:
        if type(self.estimator_id) is not str or not _ESTIMATOR_ID.fullmatch(
            self.estimator_id
        ):
            _reject("provenance_invalid")
        for digest in (self.revision_digest, self.terms_digest):
            if type(digest) is not str or not _HEX64.fullmatch(digest):
                _reject("provenance_invalid")


@dataclass(frozen=True, slots=True, repr=False)
class PoseSequence:
    """Pose frames for the ticks of one frame-sampling manifest.

    Args:
        manifest: The validated manifest whose selected ticks the frames use.
            The sequence keeps only its coverage digest and tick set.
        frames: Up to ``MAX_POSE_FRAMES`` frames with strictly increasing ticks,
            each one of the manifest's selected ticks. A selected frame with no
            pose may be omitted.
        depth_kind: ``none`` forbids ``z``; ``relative`` allows estimator-scaled
            depth up to ``MAX_RELATIVE_DEPTH`` in magnitude; ``metric`` allows
            metres from the estimator's origin, up to 100 in magnitude.
        provenance: Digest-only estimator provenance.

    Coordinates remain sensitive biometric data in memory; this contract is
    not de-identification and performs no measurement.
    """

    manifest: FrameSamplingManifest = field(repr=False)
    frames: tuple[PoseFrame, ...] = field(repr=False)
    depth_kind: DepthKind
    provenance: PoseEstimatorProvenance = field(repr=False)
    coverage_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.manifest) is not FrameSamplingManifest:
            _reject("manifest_invalid")
        depth = _member(DepthKind, self.depth_kind)
        if depth is None:
            _reject("depth_kind_invalid")
        if type(self.provenance) is not PoseEstimatorProvenance:
            _reject("provenance_invalid")
        if type(self.frames) not in (tuple, list) or not (
            1 <= len(self.frames) <= MAX_POSE_FRAMES
        ):
            _reject("frames_invalid")
        if any(type(frame) is not PoseFrame for frame in self.frames):
            _reject("frames_invalid")
        selected = frozenset(self.manifest.timestamps)
        previous = -1
        observations = 0
        for frame in self.frames:
            if frame.tick not in selected:
                _reject("tick_not_in_manifest")
            if frame.tick <= previous:
                _reject("ticks_not_increasing")
            previous = frame.tick
            observations += len(frame.joints)
            for joint in frame.joints:
                _check_depth(joint.z, depth)
        if observations > MAX_OBSERVATIONS:
            _reject("resource_limit")
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "depth_kind", depth)
        object.__setattr__(self, "coverage_digest", self.manifest.coverage_digest)

    @property
    def notice(self) -> str:
        """Return the mandatory landmarks-only boundary."""
        return NON_DIAGNOSTIC_NOTICE

    def __repr__(self) -> str:
        return (
            f"PoseSequence(frame_count={len(self.frames)}, "
            f"depth_kind={self.depth_kind.value})"
        )

    def require_reviewer_confirmation(self, *, confirmed: bool = False) -> None:
        """Fail closed unless the caller explicitly confirms consequential use.

        Confirmation records a caller decision; it does not qualify the
        estimator, the landmarks or any metric derived from them.
        """
        if confirmed is not True:
            _reject("review_required")

    def to_report(self) -> dict[str, object]:
        """Return counts, the notice and the review flag; never coordinates."""
        seen: dict[str, int] = {}
        for frame in self.frames:
            for joint in frame.joints:
                seen[joint.joint] = seen.get(joint.joint, 0) + 1
        frame_count = len(self.frames)
        return {
            "schema_version": POSE_SCHEMA_VERSION,
            "notice": NON_DIAGNOSTIC_NOTICE,
            "frame_count": frame_count,
            "joint_count": len(seen),
            "missing_joint_counts": {
                name: frame_count - seen.get(name, 0) for name in CANONICAL_JOINTS
            },
            "review_required": True,
        }

    def to_json(self) -> str:
        """Serialize the content-free report deterministically."""
        return json.dumps(self.to_report(), separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_record(cls, record: Any) -> PoseSequence:
        """Build a sequence from a ``to_record`` mapping, such as a parity case.

        Raises:
            PoseContractError: For unknown or missing keys, or invalid values.
            FrameSamplingManifestError: For an invalid embedded manifest.
        """
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            _reject("record_invalid")
        manifest = record["manifest"]
        provenance = record["provenance"]
        frames = record["frames"]
        if (
            type(manifest) is not dict
            or type(provenance) is not dict
            or set(provenance) != _PROVENANCE_KEYS
            or type(frames) is not list
            or any(
                type(frame) is not dict
                or set(frame) != {"tick", "joints"}
                or type(frame["joints"]) is not list
                for frame in frames
            )
        ):
            _reject("record_invalid")
        # Enforce the size limits before building any observation, so an
        # oversized record is rejected without materializing it.
        if not 1 <= len(frames) <= MAX_POSE_FRAMES:
            _reject("frames_invalid")
        if any(len(frame["joints"]) > len(CANONICAL_JOINTS) for frame in frames):
            _reject("joints_invalid")
        if sum(len(frame["joints"]) for frame in frames) > MAX_OBSERVATIONS:
            _reject("resource_limit")
        built = []
        for frame in frames:
            joints = []
            for joint in frame["joints"]:
                if type(joint) is not dict or not _OBSERVATION_REQUIRED <= set(
                    joint
                ) <= (_OBSERVATION_REQUIRED | {"z"}):
                    _reject("record_invalid")
                joints.append(PoseJointObservation(**joint))
            built.append(PoseFrame(frame["tick"], tuple(joints)))
        manifest_fields = dict(manifest)
        if "sampling_mode" in manifest_fields:
            mode = _member(SamplingMode, manifest_fields["sampling_mode"])
            if mode is None:
                _reject("manifest_invalid")
            manifest_fields["sampling_mode"] = mode
        if (
            "timestamps" in manifest_fields
            and type(manifest_fields["timestamps"]) is list
        ):
            manifest_fields["timestamps"] = tuple(manifest_fields["timestamps"])
        bound = _build_manifest(manifest_fields)
        if bound is None:
            _reject("manifest_invalid")
        return cls(
            manifest=bound,
            frames=tuple(built),
            depth_kind=record["depth_kind"],
            provenance=PoseEstimatorProvenance(**provenance),
        )

    def to_record(self) -> dict[str, object]:
        """Return the full sequence, coordinates included, for local storage.

        Use it for synthetic parity fixtures and caller-owned files only; it is
        not a report and must never be logged or sent anywhere.
        """
        return {
            "manifest": self.manifest.to_dict(),
            "depth_kind": self.depth_kind.value,
            "provenance": {
                "estimator_id": self.provenance.estimator_id,
                "revision_digest": self.provenance.revision_digest,
                "terms_digest": self.provenance.terms_digest,
            },
            "frames": [
                {
                    "tick": frame.tick,
                    "joints": [_observation_record(joint) for joint in frame.joints],
                }
                for frame in self.frames
            ],
        }


_RECORD_KEYS = frozenset({"manifest", "depth_kind", "provenance", "frames"})
_PROVENANCE_KEYS = frozenset({"estimator_id", "revision_digest", "terms_digest"})
_OBSERVATION_REQUIRED = frozenset({"joint", "x", "y", "visibility", "presence"})


def _check_depth(z: float | None, depth: DepthKind) -> None:
    if z is None:
        return
    if depth is DepthKind.NONE:
        _reject("depth_not_declared")
    limit = MAX_RELATIVE_DEPTH if depth is DepthKind.RELATIVE else _METRIC_DEPTH_LIMIT_M
    if abs(z) > limit:
        _reject("depth_invalid")


def _observation_record(joint: PoseJointObservation) -> dict[str, object]:
    record: dict[str, object] = {
        "joint": joint.joint,
        "x": joint.x,
        "y": joint.y,
        "visibility": joint.visibility,
        "presence": joint.presence,
    }
    if joint.z is not None:
        record["z"] = joint.z
    return record
