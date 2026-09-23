# Audio resampling plans

`plan_audio_resampling` turns declared sample rates and frame counts into a
deterministic, allocation-free conversion plan for ASR adapters: the reduced
rational rate, the target frame count under an explicit rounding policy, the
resulting duration error, and an exact or rounded status. It is a planner,
not a resampler.

```python
from openmed.multimodal.audio_resample_plan import RoundingPolicy, plan_audio_resampling

plan = plan_audio_resampling(44100, 16000, 44100, rounding=RoundingPolicy.NEAREST_EVEN)
print(plan.target_frames, plan.status.value, plan.duration_error_seconds)
```

## Supported boundary

`source_rate_hz` and `target_rate_hz` must be positive integers of at most
`MAX_AUDIO_RATE_HZ` (2^32 - 1). `source_frames` must be a non-negative
integer of at most `MAX_AUDIO_FRAMES`; zero frames are a valid empty plan.
The target frame count is `source_frames * target_rate_hz / source_rate_hz`
rounded by the chosen policy:

- `RoundingPolicy.FLOOR` always rounds down.
- `RoundingPolicy.CEILING` always rounds up.
- `RoundingPolicy.NEAREST_EVEN` rounds halves to the even quotient.

The status is `exact` when the conversion divides evenly and `rounded`
otherwise. The duration error is the signed difference between the resampled
duration and the declared duration, in seconds, computed from exact integer
arithmetic. The planner reads no audio, allocates no samples, performs no
filtering or channel mixing, and makes no claim about resampled quality.

## Limits and checked arithmetic

The `source_frames * target_rate_hz` product is checked by division against
the 63-bit manifest byte-size bound before it is materialized, so a caller
can never receive a frame count beyond that contract. An optional
`max_duration_error_seconds` tolerance (a finite, non-negative number)
fails the whole plan with `audio_resample_error_exceeded` when the chosen
policy's absolute duration error exceeds it; exact plans always pass.

## Failures

`AudioResamplePlanError` is a `ValueError` with a stable `.category`; its
string is the same category, and no rate, frame count, or caller value is
ever echoed. For each of `source_rate`, `target_rate`, and `frames` the
categories are `audio_resample_<field>_missing`, `audio_resample_<field>_boolean`,
`audio_resample_<field>_not_finite`, `audio_resample_<field>_not_integer`,
`audio_resample_<field>_not_positive` (zero included for rates, negative
included for frames), and `audio_resample_<field>_overflow`; plan arithmetic
adds `audio_resample_product_overflow` and `audio_resample_error_exceeded`.
Invalid `rounding` or `max_duration_error_seconds` API arguments are reported
separately as plain `ValueError` with constant messages.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_audio_resample_plan.py -q
```

Tests use synthetic rates and frame counts only. They cover identity,
integer, non-integer, tie, one-frame, zero-frame, and long-duration plans
against hand calculations, every rejection category, and deterministic
`to_dict`/`to_json` serialization, offline.
