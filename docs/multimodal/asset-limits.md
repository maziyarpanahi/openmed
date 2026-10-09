# Multimodal asset limit profiles

A limit profile is one pre-decode admission policy for multimodal assets. It is
evaluated over the privacy-safe asset manifest before any decoder, tensor, or
model is loaded, so a page count, pixel dimension, frame count, or audio
duration that would exhaust memory is refused up front instead of during
inference.

## Overview

`evaluate_asset_limits(profile, manifest, modality)` accepts a `LimitProfile`,
a canonical `AssetManifest` instance or a metadata mapping, and one of the four
manifest modalities (`image`, `pdf`, `dicom`, `audio`). It returns a list of
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
`pixels`, `total_pixels`, `frames`, `duration_seconds`. Each carries:

- `field_name`: one of the six limit fields above.
- `reason_code`: `limit_exceeded` (the observed value is above the ceiling) or
  `insufficient_metadata` (the rule could not be evaluated; `observed` is
  `null`).
- `limit`: the inclusive ceiling that applied.
- `observed`: the number the manifest yielded, or `null`.

Findings cannot be constructed with arbitrary field names or reason codes, and
they are separate from the manifest-profile `ValidationFinding` and from the
coarse abstention reason codes. The preflight report carries the coarse
`RESOURCE_LIMIT` abstention alongside these detailed findings for exceeded or
unevaluable limits.

## Admission in Python redaction handlers

`redact_image`, the image and PDF `redact_document` handlers, the PDF rasterizer,
and `redact_dicom_pixels` now enforce resource admission before pixel decoding.
They use `MOBILE_V1` by default. Set `asset_limit_profile` on the existing policy
mapping or object to select `DESKTOP_V1` or a validated `LimitProfile.with_limits`
override. There is no disable switch:

```python
from openmed.multimodal import MOBILE_V1, RedactionAdmissionError, redact_image

policy = {"asset_limit_profile": MOBILE_V1.with_limits(max_frames=32)}
try:
    result = redact_image(source, policy=policy, models=local_models)
except RedactionAdmissionError as error:
    # These fields contain controlled codes and numbers, never source details.
    finding = (error.reason_code, error.field_name, error.limit, error.observed)
```

Admission checks the byte ceiling and a bounded media signature first. PNG/JPEG
geometry uses the existing header readers. PNG animation declarations, JPEG
multi-picture indexes and classic TIFF directories are counted without decoding
pixels; secondary JPEG frame geometry is checked too. TIFF strip/tile payloads
are not read, and identifying tags are never interpreted or returned. Metadata
reads are capped at 64 KiB,
and PNG traversal at 4,096 chunk headers. Sparse TIFF directory offsets may
span the admitted file but remain subject to the cumulative read budget.
BigTIFF and unknown image signatures fail closed.

PDF admission uses the bounded page-tree reader and its existing object and
object-stream expansion bounds. Review/rejected geometry (including count
mismatches) is refused. Raster budgets use the full media boxes and the actual
rendering resolution, rounding each pixel dimension up; a small crop cannot
hide the renderer's initial full-page allocation. This is a
separate handler admission calculation; the manifest-only evaluator still
does not infer PDF raster geometry. Page count and raster budgets are checked
again before each rendering call, and page iteration is bounded during both
text and table extraction in the redaction handler.

DICOM admission uses pydicom in metadata-only mode over a transport that caps
cumulative reads at 64 KiB, bounds offsets by file size, and refuses unbounded
reads. Admission uses only numeric rows, columns and frame count. Deflated
datasets requiring an unbounded read, malformed headers, unknown signatures,
and missing pixel geometry fail closed. Metadata-only DICOM without pixels
retains the existing header-only behavior. The actual decoded array shape is
checked before copying or iterating, and excess frames are refused rather than
silently truncated. This does not sandbox the decoder or bound its internal
allocation when a malicious header under-declares the payload; decoder
isolation remains a separate workstream.

`RedactionAdmissionError` is a `ValueError` with `reason_code`, `field_name`,
`limit`, and `observed` attributes. Limit refusals reuse `limit_exceeded` and
`insufficient_metadata`; structural refusals retain bounded-reader categories
such as `pdf_page_limit`, `page_count_mismatch` or
`image_pixel_limit_exceeded`. Transport and profile failures use controlled
categories such as `header_byte_limit`, `header_offset_invalid`,
`preflight_source_read_error` and `limit_profile_invalid`. Exception messages
contain only these codes, numbers and nulls. Seekable caller streams are
restored after admission and are never closed. Refused admission creates no
output file. Accepted media retains its existing redaction and OCR behavior.

This change is scoped to the existing Python entry points named above; it does
not introduce a Swift session API, batch orchestration, cloud provider, model
qualification or clinical action. Resource admission is neither PHI-absence
verification nor a guarantee that decoding fits a device's memory budget.
