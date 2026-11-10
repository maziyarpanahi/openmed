# Multimodal manifest profiles

These profiles apply modality-specific rules to the canonical fields from the
privacy-safe asset manifest contract. They do not open or decode an asset. Legacy profiles
remain v1.0; waveform and non-pixel profiles use v1.1. Existing manifest v1
records and their serialized outputs are unchanged when the new optional fields
are absent.

## Overview

A manifest profile determines whether structurally validated metadata includes
the fields required for its declared modality.

`validate_manifest_metadata()` accepts either a canonical `AssetManifest`
instance or a metadata mapping. Mapping callers remain responsible for running
the structural asset-manifest validator first.

The privacy and safety boundary is strict:

- Validation uses manifest metadata only.
- Files are never opened or decoded by the validator.
- Metadata is not proof that the underlying file is trustworthy.
- This is a preflight check, not a clinical semantic validator.

## Profiles

### Image Profile (v1.0)
- **Modality:** `image`
- **Version:** `1.0`
- **Required Fields:** `width`, `height`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `pages`, `frames`, `duration_seconds`

### PDF Profile (v1.0)
- **Modality:** `pdf`
- **Version:** `1.0`
- **Required Fields:** `pages`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `width`, `height`, `frames`, `duration_seconds`

### DICOM Profile (v1.0)
- **Modality:** `dicom`
- **Version:** `1.0`
- **Required Fields:** `frames`, `width`, `height`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `pages`, `duration_seconds`

### Audio Profile (v1.0)
- **Modality:** `audio`
- **Version:** `1.0`
- **Required Fields:** `duration_seconds`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `width`, `height`, `pages`, `frames`

### Waveform Profile (`WAVEFORM_V1`, v1.1)

- **Modality:** `waveform`
- **Required Fields:** `channel_count`, `sample_rate_hz`, `samples_per_channel`,
  `duration_seconds`
- **Inapplicable Fields:** `pages`, `width`, `height`, `frames`

Declared waveform container media types are `application/edf` (EDF/EDF+) and
`application/wfdb` (a caller-described WFDB recording). These are explicit
OpenMed declarations, not claims of MIME registration or full file support.
Counts are positive integers bounded by `2**31 - 1`; rate and duration are
positive finite numbers bounded by the same value. A single shared rate/count
is required: heterogeneous EDF signal rates need a reader-side contract before
being represented here. No signal labels, sample amplitudes, patient headers,
filenames, paths, or companion-file references belong in a manifest.

The duration/rate-derived sample count must agree with `samples_per_channel`
within one sample for endpoint conventions; otherwise the profile returns
`sample_count_mismatch` for `samples_per_channel`. Fractional rates are allowed.
Validation uses exact rational arithmetic and never multiplies floating values
into an overflowing result.

### DICOM sub-profiles

The optional `dicom_subprofile` field is permitted only for
`application/dicom`, with controlled values `waveform` or `non_pixel`.
Omitting it retains the original pixel-oriented `DICOM_V1` behavior.

- `waveform` selects `WAVEFORM_V1`: ECG-shaped metadata requires waveform
  fields, not pixel dimensions or frames.
- `non_pixel` selects `DICOM_NON_PIXEL_V1` (v1.1, modality `dicom_non_pixel`):
  all geometry, duration and waveform fields are inapplicable. Structured
  reports and encapsulated objects receive the byte-size ceiling only.

Sub-profiles are declarations, not proof of DICOM SOP class, content safety or
absence of nested images. A reader must verify the underlying object and apply
limits to any nested content. Non-pixel decoded-memory plans remain unevaluable;
encoded byte size is not guessed to be decoded size. `application/dicom+json`
keeps its existing unsupported preflight modality.

### Synthetic example

```python
from openmed.multimodal import AssetManifest, WAVEFORM_V1, validate_manifest_metadata

manifest = AssetManifest(
    asset_id="synthetic-ecg-1",
    media_type="application/dicom",
    dicom_subprofile="waveform",
    sha256="a" * 64,
    byte_size=132,
    channel_count=12,
    sample_rate_hz=500,
    samples_per_channel=5000,
    duration_seconds=10,
)
assert validate_manifest_metadata(WAVEFORM_V1, manifest) == []
```

The digest above is a synthetic placeholder; a real preflight requires the
actual bounded source digest. Resource admission is non-diagnostic, adds no
clinical interpretation or clinical validation, and never triggers a clinical
decision. Downstream consequential outputs require an explicit reviewer
confirmation and a non-diagnostic notice; a preflight accept is not that approval.
This slice owns the existing Python manifest/preflight APIs. Waveform readers,
lead interpretation, SDK reader adapters and JSON Schema catalog updates
(#3051) remain separate work; no schemas or provider qualification are claimed.

## Validation Findings

The validator evaluates the metadata against these profiles and produces deterministic, categorical findings. Findings only contain the field name and a stable reason code.

**Categorical Reason Codes:**
- `missing_required`: A required metadata field is missing.
- `inapplicable_present`: A field marked as inapplicable for this modality is present in the manifest.
- `invalid_zero`: A numeric field was exactly zero (0).
- `invalid_boolean`: A python boolean (`True`/`False`) was provided for a numeric field.
- `non_finite_numeric`: A numeric field contained NaN, +Infinity, or -Infinity.
- `invalid_type`: The field value was of an unsupported type.
- `out_of_range`: A numeric value was negative or exceeded its fixed bound.
- `sample_count_mismatch`: Waveform count differs from duration × rate by more
  than one sample.

Findings cannot be constructed with arbitrary field names or reason codes. They
do not expose source paths, filenames, URLs, raw input values, embedded media
content, or arbitrary exception messages.
