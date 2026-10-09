# Multimodal asset limit profiles

A limit profile is one pre-decode admission policy for multimodal assets. It is
evaluated over the privacy-safe asset manifest before any decoder, tensor, or
model is loaded, so a page count, pixel dimension, frame count, or audio
duration that would exhaust memory is refused up front instead of during
inference.

## Overview

`evaluate_asset_limits(profile, manifest, modality)` accepts a `LimitProfile`,
a canonical `AssetManifest` instance or a metadata mapping, and a manifest
modality (`image`, `pdf`, `dicom`, `audio`, `waveform`, `dicom_non_pixel`). It returns a list of
`LimitFinding` values; an empty list means every rule the modality can express
was evaluated and passed.

The privacy and safety boundary is strict:

- Evaluation uses manifest metadata only. Files are never opened, decoded, or
  downsampled.
- A rule that cannot be evaluated is reported as `insufficient_metadata`, never
  assumed to pass. Missing evidence does not become acceptance.
- Raster geometry is never inferred. A PDF manifest carries no width or height
  under the PDF profile, so a PDF's two pixel rules are always unevaluable and
  are reported as such even when bytes and pages pass. Preflight must abstain
  on that basis rather than accept.
- Findings carry a field name, a reason code, the ceiling, and the observed
  number (or `null`). They never carry identifiers, paths, digests, or content.

Mapping callers remain responsible for running the structural asset-manifest
validator first. Values outside the manifest contract (booleans, zero,
negatives, non-finite numbers, values beyond the manifest bounds) raise
`AssetLimitError` with a content-free message.

## Profiles

Ceilings are inclusive: a value equal to the ceiling passes, a value above it
fails. Profiles are immutable; `profile.with_limits(...)` returns a re-validated
copy for callers that need their own policy.

| Limit | `MOBILE_V1` | `DESKTOP_V1` |
| --- | ---: | ---: |
| `max_byte_size` | 64 MiB (`64 * 1024**2`) | 256 MiB (`256 * 1024**2`) |
| `max_pages` | 10 | 100 |
| `max_pixels` (per image, page, or frame) | 10,000,000 | 40,000,000 |
| `max_total_pixels` | 25,000,000 | 100,000,000 |
| `max_frames` | 128 | 1,024 |
| `max_duration_seconds` | 300 | 1,800 |

These are explicit initial admission-policy choices, not measured device memory
budgets, and not a guarantee that decoding or inference will fit on any
hardware. The desktop page and pixel ceilings match the redacted-PDF renderer's
defaults (`render_pdf.py`) so the tree carries one policy; the other values are
unbenchmarked starting points. Existing renderer behaviour is unchanged by this
module.

## Waveform limit profiles (v1.1)

`MOBILE_V1_1` and `DESKTOP_V1_1` preserve the original ceilings above and add:

| Limit | `MOBILE_V1_1` | `DESKTOP_V1_1` |
| --- | ---: | ---: |
| `max_channel_count` | 32 | 256 |
| `max_sample_rate_hz` | 10,000 | 100,000 |
| `max_samples_per_channel` | 3,000,000 | 180,000,000 |
| `max_total_samples` | 16,000,000 | 64,000,000 |

These are unbenchmarked admission-policy choices, not measured device limits or
clinical qualification. The waveform modality checks bytes, duration, channels,
rate, per-channel samples and total samples. Total samples charge
`channel_count * max(samples_per_channel, ceil(duration_seconds * sample_rate_hz))`.
Missing any factor yields `insufficient_metadata`. Products saturate at
`2**63 - 1`; overflow produces `arithmetic_saturated` and rejects even when the
configured ceiling equals the saturation bound. Count consistency is a separate
metadata check and must also pass. Non-pixel DICOM checks bytes only.

A waveform evaluation requires a v1.1 limit profile; passing a v1.0 profile
raises the content-free `AssetLimitError`. Original v1.0 profiles and their
serialized output are unchanged. Use a v1.1 profile's `with_limits()` to change
waveform ceilings. All four waveform ceilings are required and revalidated.

## Rules by modality

| Rule | `image` | `pdf` | `dicom` | `audio` |
| --- | :-: | :-: | :-: | :-: |
| `byte_size` | ✓ | ✓ | ✓ | ✓ |
| `pages` | — | ✓ | — | — |
| `pixels` | width × height | unevaluable | width × height (per frame) | — |
| `total_pixels` | width × height | unevaluable | width × height × frames | — |
| `frames` | — | — | ✓ | — |
| `duration_seconds` | — | — | — | ✓ |

A rule marked — is not applicable to the modality and produces no finding. A
rule that is applicable but whose inputs are absent produces
`insufficient_metadata`. Pixel products use checked arithmetic: every factor
must be a bounded positive integer within the manifest count bound, and the
product is bounded by `MAX_PIXEL_PRODUCT` (the cube of that bound).

## Findings

Findings are returned in a fixed, documented order: `byte_size`, `pages`,
`pixels`, `total_pixels`, `frames`, `duration_seconds`, `channel_count`,
`sample_rate_hz`, `samples_per_channel`, `total_samples`. Each carries:

- `field_name`: one of the limit fields above.
- `reason_code`: `limit_exceeded` (the observed value is above the ceiling) or
  `insufficient_metadata` (the rule could not be evaluated; `observed` is
  `null`), or `arithmetic_saturated` (a waveform product overflowed; observed
  is the signed 64-bit saturation ceiling).
- `limit`: the inclusive ceiling that applied.
- `observed`: the number the manifest yielded, or `null`.

Findings cannot be constructed with arbitrary field names or reason codes, and
they are separate from the manifest-profile `ValidationFinding` and from the
coarse abstention reason codes. The preflight report carries the coarse
`RESOURCE_LIMIT` abstention alongside these detailed findings for exceeded or
unevaluable limits.
