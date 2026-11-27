# Pose keypoint sequence contract

`openmed.multimodal.pose.sequence` validates body landmarks that a caller's
pose estimator produced for sampled video frames. It reads no files, runs no
estimator, makes no network calls and computes no metric. Joint mapping from
specific estimators, estimators themselves, gait and posture metrics and
biometric session handling are separate components that build on it. The
schema is `openmed.multimodal.pose.sequence.v1`.

Every report carries this boundary:

> Pose landmarks only; not gait, balance, fall-risk or clinical assessment.
> Consequential use requires explicit reviewer confirmation.

## Canonical representation

- `CANONICAL_JOINTS` is a closed vocabulary of 21 body joints: `nose`,
  `head`, `neck`, `spine`, `pelvis`, and left and right `shoulder`, `elbow`,
  `wrist`, `hip`, `knee`, `ankle`, `heel` and `foot_index`. Estimator-specific
  names (BlazePose landmark indices, Apple Vision joint names) must be mapped
  before admission; unknown names fail as `unknown_joint`.
- `PoseJointObservation` holds one joint: normalized image `x` (0 at the left
  edge, 1 at the right) and `y` (0 at the top, 1 at the bottom), `visibility`
  and `presence` from 0 to 1, and optional depth `z`. Out-of-range or
  non-finite values fail as `coordinate_invalid`, `confidence_invalid` or
  `depth_invalid`; nothing is clipped.
- `PoseFrame` is keyed by an integer tick of a `FrameSamplingManifest`. A joint
  the estimator did not return is left out, never filled with placeholder
  coordinates; a second observation of the same joint fails as
  `duplicate_joint`. Joints are stored in canonical order.
- `PoseSequence` is bound to one validated manifest and keeps its
  `coverage_digest`. Every frame tick must be one of the manifest's selected
  ticks (`tick_not_in_manifest`), in strictly increasing order. A selected
  frame without a detected person may be omitted or given an empty frame.
- `depth_kind` declares what `z` means: `none` forbids depth, `relative`
  allows estimator-scaled depth up to 10 in magnitude, and `metric` allows
  metres from the estimator's origin, up to 100 in magnitude.
- `PoseEstimatorProvenance` is digest-only: a lowercase estimator slug, the
  SHA-256 of the model or library revision and the SHA-256 of the licence or
  terms accepted for it. Paths, device names and free text are refused.
- A sequence holds at most 100,000 frames and 1,000,000 joint observations.

## Synthetic example

```python
import hashlib

from openmed.multimodal.frame_sampling_manifest import (
    FrameSamplingManifest,
    SamplingMode,
    canonical_coverage_digest,
)
from openmed.multimodal.pose.sequence import (
    PoseEstimatorProvenance,
    PoseFrame,
    PoseJointObservation,
    PoseSequence,
)

ticks = (0, 10)
manifest = FrameSamplingManifest(
    duration_ticks=30,
    time_base_num=1,
    time_base_den=30,
    sampling_mode=SamplingMode.UNIFORM,
    timestamps=ticks,
    step_ticks=10,
    coverage_digest=canonical_coverage_digest(SamplingMode.UNIFORM, 1, 30, ticks),
)
digest = hashlib.sha256(b"synthetic").hexdigest()
sequence = PoseSequence(
    manifest=manifest,
    frames=(
        PoseFrame(0, (PoseJointObservation("left_hip", 0.31, 0.52, 0.98, 1.0),)),
        PoseFrame(10, ()),
    ),
    depth_kind="none",
    provenance=PoseEstimatorProvenance("synthetic-walker", digest, digest),
)
assert sequence.to_report()["missing_joint_counts"]["left_hip"] == 1
```

## Privacy and review boundary

Landmark sequences are biometric data; validation is not de-identification.
`repr`, `to_report()` and `to_json()` carry only the schema version, notice,
frame count, the number of distinct joints observed, per-joint missing counts
and `review_required: true`. Errors are `PoseContractError` with a stable code
only and never echo a coordinate. An invalid embedded manifest raises the
manifest's own value-free `FrameSamplingManifestError`.

`to_record()` and `from_record()` carry the coordinates. They exist for
caller-owned storage and for the shared parity fixture
`tests/fixtures/parity/pose_sequence_v1.json`, which later Swift and Kotlin
ports load; never log or transmit a record.

`require_reviewer_confirmation()` fails closed unless the caller passes
`confirmed=True` after explicit review. Confirmation records a caller decision;
it does not qualify the estimator, the landmarks or any metric derived from
them.

All tests and fixtures use synthetic values. No dataset, model or dependency is
bundled.
