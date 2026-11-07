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

## Executing bounded local sample conversion

`openmed.multimodal.audio_sample_conversion.convert_audio_samples` consumes a
validated `AudioResamplePlan`, original `WavMetadata` and `AsrAudioProfile`.
Install the existing `multimodal` extra to enable the optional BSD-licensed
NumPy backend. Importing the module requires no NumPy, model, network or audio
I/O. The caller decodes source chunks into normalized float32/float64 arrays
shaped `(frames, source_channels)`; compressed formats, integer arrays,
non-finite values and samples outside `[-1, 1]` are rejected. Declared source
format/depth must also satisfy the existing profile compatibility contract.
The output is frame-major signed PCM16 with nearest-even quantization, so the
profile must accept PCM16 at the plan's target rate and selected channels.

Declare `ChannelPolicy.PRESERVE` or `ChannelPolicy.MEAN_MONO`. Mean-to-mono
averages every source channel; it does not choose a speaker or perform source
separation. The profile must permit the transformation. Review/incompatible
source reports, forged plans, mismatched frame counts and duration violations
(including those introduced by rounding) fail closed before output is returned.

This is a **bounded batch adapter**, not an unbounded streaming resampler.
Input chunks are validated into one owned source buffer before conversion;
no partial output is exposed. `ConversionBudget` bounds sample-buffer bytes,
chunk frames and filter operations before allocation or source consumption.
Owned sample storage is `8 * source_frames * output_channels +
2 * target_frames * output_channels + 8 * filter_taps` bytes. These are array
payload ceilings, not a process RSS guarantee: interpreter/backend overhead,
caller-owned decoded chunks and provider copies are outside that ceiling.
Defaults are 32 MiB, 8192 chunk frames and 100 million filter operations.

The rate-changing filter is a centered, normalized Hann-windowed sinc with
cutoff `0.9 * min(1, target_rate/source_rate)` in source Nyquist units and
radius `ceil(32/cutoff)`. Ratios needing radius above 512 fail. Source boundaries
use zero padding; centered sampling compensates filter delay. Equal rates
bypass filtering. Global rational positions make results independent of input
chunk partitioning. Both input clipping and filter overshoot fail; the adapter
never silently saturates, normalizes, pads missing input or truncates excess
input. Engineering tone attenuation tests are not ASR qualification or clinical
validation.

```python
import numpy as np
from openmed.multimodal.asr_audio_profile import MONO_16K_PCM_PROFILE
from openmed.multimodal.audio_resample_plan import plan_audio_resampling
from openmed.multimodal.audio_sample_conversion import (
    ChannelPolicy, convert_audio_samples,
)
from openmed.multimodal.wav_metadata import WavMetadata

# Synthetic silence, not a recording or model result.
source = np.zeros((4800, 2), dtype=np.float64)
metadata = WavMetadata(1, 2, 48000, 16, 19200, 4800, 0.1)
plan = plan_audio_resampling(48000, 16000, 4800)
with convert_audio_samples(
    [source], metadata, plan, MONO_16K_PCM_PROFILE,
    channel_policy=ChannelPolicy.MEAN_MONO,
) as converted:
    # Set this only after an application reviewer explicitly confirms handoff.
    reviewed_pcm = converted.samples_for_reviewed_handoff(reviewer_confirmed=True)
    assert reviewed_pcm.shape == (1600, 1)
    assert converted.report.source_position(1600) == 4800
```

A cancellation callback is checked before allocation, for each source frame,
for each target frame and before return. Owned source/filter buffers are zeroed
on all exits; partial output is zeroed on failure or cancellation. Successful
output must be closed (prefer the context manager); `close()` zeroes borrowed
NumPy views. Caller source buffers and copies remain caller-owned. Conversion
errors use controlled codes and suppress source-bearing exception messages.
Do not serialize or log protected input/output arrays. The report contains only
counts, rates, a configuration digest, fixed filter identity, policy and notice;
it never retains source payloads or caller paths/profile names.

`source_position` maps target frame boundaries through the reduced rational
ratio, clamping rounded overshoot to the source extent.
`source_interval_seconds` maps downstream transcript frame offsets to exact
`Fraction` times. These describe time coordinates, not isolated sample influence:
the centered filter uses neighbors and zero padding. Conversion adds no ASR,
cloud fallback, model assets or autonomous clinical decisions. The non-diagnostic
notice remains bound to the output, and explicit reviewer confirmation is needed
for handoff; consequential transcript outputs require their own review.

### On-device Swift surface

OpenMedKit's `LocalAudioSampleConversion` provides the same filter, channel
policies, PCM16 quantization, budget defaults, cancellation and review gate.
`AudioSampleConversionPlan` performs checked integer frame rounding;
`AudioSampleConversionProfile` declares local output limits. Swift accepts
caller-decoded, interleaved normalized `Double` chunks from PCM16 sources, using
system Foundation math and no additional package or cloud service. Its report
binds the fixed filter identity, plan, channel policy and non-diagnostic notice;
`sourcePosition` returns exact source-frame numerator/denominator pairs.
Swift copies returned at handoff are caller-owned; `close()` disposes the owned
buffer and rejects later handoffs. The adapter is synchronous and callers must
serialize access to protected output lifetime on both platforms.

### Conversion verification

```bash
.venv/bin/python -m pytest tests/unit/multimodal/test_audio_sample_conversion.py tests/integration/test_local_audio_conversion.py -q
cd swift/OpenMedKit && swift test --filter AudioSampleConversionTests
```

Synthetic tones, impulses and silence exercise frame rounding, continuity,
channel handling, anti-aliasing, invalid samples, budgets, clipping,
cancellation, output disposal, exact timing and negative provider handoff.
