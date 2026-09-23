# Frame-sampling manifests

`FrameSamplingManifest` is a strict, content-free contract for reproducible
frame selection from clinical video. A sampler declares the clip duration in
integer ticks, a rational time base, the sampling mode, the selected tick
positions, per-mode strategy fields, and a coverage digest; validation
enforces monotonicity, uniqueness, bounds, and that the positions actually
satisfy the declared strategy before the manifest can exist.

```python
from openmed.multimodal.frame_sampling_manifest import (
    FrameSamplingManifest,
    SamplingMode,
    canonical_coverage_digest,
)

timestamps = (0, 500, 1000)
digest = canonical_coverage_digest(SamplingMode.UNIFORM, 1, 1000, timestamps)
manifest = FrameSamplingManifest(
    duration_ticks=10_000,
    time_base_num=1,
    time_base_den=1000,
    sampling_mode=SamplingMode.UNIFORM,
    timestamps=timestamps,
    coverage_digest=digest,
    step_ticks=500,
)
print(manifest.to_json())
```

## Supported boundary

Manifests carry no frames, audio, captions, pixel data, or identifiers:
only integers, the mode, and the digest. The time base is a reduced rational
(`gcd(num, den) == 1`) of positive bounded integers; tick positions are
non-negative integers strictly below `duration_ticks` and strictly
increasing, so reordering and duplicates fail closed. The schema version is
fixed at 1.

The three sampling modes are a closed set with per-mode strategy fields:

- `uniform` requires `step_ticks` (a positive integer). Every consecutive
  position difference must equal the declared step, so a misdeclared step
  fails instead of being guessed.
- `keyframe` forbids all strategy fields: keyframe positions are
  content-determined and only the monotonicity and bounds contract applies.
- `bounded_window` requires `window_start_ticks` and `window_end_ticks`
  (`0 <= start < end <= duration_ticks`, end exclusive) and every position
  must fall inside the declared window. `step_ticks` is forbidden.

## Coverage digest

`canonical_coverage_digest` returns the SHA-256 hex digest over the
deterministic JSON encoding of the schema version, the mode, the reduced
time base, and the selected positions. Two manifests covering the same
positions in the same time base produce the same digest regardless of how
they were built. A declared digest must be 64 lowercase hex characters and
must match the recomputed coverage, so an edited manifest cannot pass
silently.

## Failures

`FrameSamplingManifestError` is a `ValueError` with a stable `.category`;
its string is the same category, and no input value is echoed. Categories
are `frame_sampling_schema_unsupported`, `frame_sampling_time_base_invalid`,
`frame_sampling_time_base_not_reduced`, `frame_sampling_duration_invalid`,
`frame_sampling_mode_unsupported`, `frame_sampling_timestamps_invalid`,
`frame_sampling_timestamps_out_of_range`,
`frame_sampling_timestamps_not_increasing`, `frame_sampling_step_invalid`,
`frame_sampling_step_mismatch`, `frame_sampling_strategy_fields_invalid`,
`frame_sampling_window_invalid`, `frame_sampling_window_mismatch`,
`frame_sampling_coverage_digest_invalid`, and
`frame_sampling_coverage_digest_mismatch`.

Video decoding, scene detection, diagnosis, and sampling-policy
recommendation are out of scope; an accepted manifest is a reproducibility
record, not a clinical judgment.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_frame_sampling_manifest.py -q
```

Tests use synthetic tick positions only. They cover uniform, keyframe, and
bounded-window manifests, empty selections, digest reproducibility, digest
mismatches, reordered, duplicate, and out-of-range positions, invalid and
unreduced time bases, every misdeclared strategy field, and deterministic
serialization, offline.
