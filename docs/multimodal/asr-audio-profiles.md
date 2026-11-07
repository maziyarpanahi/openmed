# ASR Audio Profiles

`check_asr_compatibility` compares privacy-safe WAV metadata with an immutable
local ASR input profile. It is an explicit preflight helper: it never decodes,
resamples, downmixes, or transcribes, and it bundles no provider weights.

```python
from openmed.multimodal.asr_audio_profile import (
    AsrAudioProfile,
    check_asr_compatibility,
)
from openmed.multimodal.wav_metadata import WAVE_FORMAT_PCM, read_wav_metadata

profile = AsrAudioProfile(
    profile_id="local-asr",
    format_codes=(WAVE_FORMAT_PCM,),
    sample_rates_hz=(8_000, 16_000),
    channel_counts=(1,),
    bit_depths=(16,),
    min_duration_seconds=0.5,
    max_duration_seconds=600.0,
)

with open("synthetic.wav", "rb") as stream:
    metadata = read_wav_metadata(stream)
report = check_asr_compatibility(metadata, profile)
print(report.compatibility.value, report.reason_codes)
```

`MONO_16K_PCM_PROFILE` is a bundled generic mono 16 kHz PCM profile. It names
no provider; a deployment should declare its own profile.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `compatible` | The declared metadata already matches the profile. |
| `resample` | Only the sample rate has to change. |
| `downmix` | Channels must be mixed down, possibly alongside a resample. |
| `review` | A human decision is needed before the audio is sent. |
| `incompatible` | The profile cannot accept this audio at all. |

All mismatches are collected in one pass and the worst verdict wins, ranked
`compatible < resample < downmix < review < incompatible`. Reason codes are
returned in the fixed order of `ASR_REASON_CODES`: `empty_audio`,
`format_unsupported`, `bit_depth_unsupported`, `channel_count_unsupported`,
`sample_rate_unsupported`, `duration_below_minimum`, `duration_above_maximum`,
`downmix_required`, `resample_required`.

A channel count above everything the profile accepts becomes
`downmix_required` when `allow_downmix` is set, and
`channel_count_unsupported` otherwise. Too *few* channels is never a downmix.
A rate mismatch becomes `resample_required` when `allow_resample` is set, and
`sample_rate_unsupported` otherwise. Duration bounds are inclusive, and a
zero-frame file reports `empty_audio`, which is always incompatible.

## Profile validation

A profile is a frozen dataclass validated on construction.
`profile_id` matches `^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$`. Each accepted
value list must be a non-empty tuple of integers — booleans are rejected
because `type(value) is int` is checked — that is strictly ascending and
therefore unique. `bit_depths` and `format_codes` are closed against
`SUPPORTED_BIT_DEPTHS` and `SUPPORTED_FORMAT_CODES`, which mirror what
[WAV metadata preflight](wav-metadata.md) can report. Rates and channel counts
are bounded by `MAX_ASR_SAMPLE_RATE_HZ` and `MAX_ASR_CHANNEL_COUNT`. Durations
must be finite, within `MAX_ASR_DURATION_SECONDS`, normalized to `float`, and
ordered minimum-first. `allow_resample` and `allow_downmix` must be real
booleans.

`AsrAudioProfileError` is a `ValueError` with a stable `.category`; its string
is the same category. Categories are `asr_profile_id_invalid`, the
`asr_profile_<kind>_values_*` family for the four value lists,
`asr_profile_duration_invalid`, `asr_profile_duration_out_of_range`,
`asr_profile_duration_range_invalid`, `asr_profile_flag_invalid`,
`asr_metadata_type_invalid` and `asr_profile_type_invalid`. Submitted values
are never echoed.

## Privacy and scope

A report holds the profile identifier, a closed verdict, ordered reason codes,
and the numeric fields already present in the WAV header: format code,
channels, sample rate, bit depth, frame count and duration. There is no field
for samples, transcripts, speaker labels, filenames, or clinical text.
`to_dict()` preserves declared field order and `to_json()` sorts keys for
byte-identical payloads.

Decoding, resampling, downmixing, running ASR, and audio quality estimation
are out of scope. A `compatible` verdict is a format statement, not a clinical
or security approval.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_asr_audio_profile.py -q
```

Fixtures are synthetic. The comparison table pins exact, resample, stereo,
depth, format, empty and duration cases, and separate tests cover reason-code
order, worst-verdict selection, forbidden transforms, and every profile
validation category.
