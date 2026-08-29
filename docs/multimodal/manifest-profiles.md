# Multimodal Manifest Profiles

This document defines the versioned metadata profiles used for safe multimodal preflight validation. 

## Overview

A manifest profile allows the preflight system to determine if a generic asset manifest is structurally valid for its declared modality *without opening or decoding the underlying asset*.

**Strict Privacy and Safety Boundary:**
- Validation uses **manifest metadata only**.
- Files are **never opened or decoded** by the validator.
- Manifest metadata is **not proof that the underlying file is trustworthy**—it merely checks that the preflight metadata contract is fulfilled.
- This represents a preflight check, not a clinical semantic validator. Model tensor layouts and clinical semantics are out of scope.

## Profiles

### Image Profile (v1.0)
- **Modality:** `image`
- **Version:** `1.0`
- **Required Fields:** `width`, `height`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `page_count`, `frame_count`, `duration`

### PDF Profile (v1.0)
- **Modality:** `pdf`
- **Version:** `1.0`
- **Required Fields:** `page_count`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `width`, `height`, `frame_count`, `duration`

### DICOM Profile (v1.0)
- **Modality:** `dicom`
- **Version:** `1.0`
- **Required Fields:** `frame_count`, `width`, `height`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `page_count`, `duration`

### Audio Profile (v1.0)
- **Modality:** `audio`
- **Version:** `1.0`
- **Required Fields:** `duration`
- **Optional Fields:** (none)
- **Inapplicable Fields:** `width`, `height`, `page_count`, `frame_count`

## Validation Findings

The validator evaluates the metadata against these profiles and produces deterministic, categorical findings. Findings only contain the field name and a stable reason code. 

**Categorical Reason Codes:**
- `missing_required`: A required metadata field is missing.
- `inapplicable_present`: A field marked as inapplicable for this modality is present in the manifest.
- `invalid_zero`: A numeric field was exactly zero (0).
- `invalid_boolean`: A python boolean (`True`/`False`) was provided for a numeric field.
- `non_finite_numeric`: A numeric field contained NaN, +Infinity, or -Infinity.
- `invalid_type`: The field value was of an unsupported type.

Findings do NOT expose source paths, filenames, URLs, raw input values, embedded media content, or arbitrary exception messages.
