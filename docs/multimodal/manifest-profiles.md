# Multimodal manifest profiles

These profiles apply modality-specific rules to the canonical fields from the
privacy-safe asset manifest contract. They do not open or decode an asset.

## Overview

A manifest profile determines whether structurally validated metadata includes
the fields required for its declared modality.

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

Findings cannot be constructed with arbitrary field names or reason codes. They
do not expose source paths, filenames, URLs, raw input values, embedded media
content, or arbitrary exception messages.
