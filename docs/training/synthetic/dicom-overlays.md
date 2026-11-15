# Synthetic DICOM overlays

Generate real Part 10 DICOM files containing only internally generated pixels and privacy-test identifiers. Each fixture pairs native grayscale pixels with generated metadata, tight text masks, bounding boxes, a combined PHI mask and [source-bound mask lineage](../../multimodal/redaction-mask-lineage.md). These fixtures exercise image and metadata redaction without using patient-derived images or restricted datasets.

Install the existing image/DICOM dependencies with `pip install "openmed[multimodal]"`. The builder loads Pillow and pydicom when generation is requested. It makes no network calls and writes no files.

## Generate and verify

```python
from io import BytesIO

from pydicom import dcmread

from openmed.training.synthetic.dicom_overlays import (
    DicomCanvas,
    DicomOverlaySpec,
    generate_dicom_overlays,
    reproduce_dicom_overlays,
)

fixture = generate_dicom_overlays(
    spec=DicomOverlaySpec(seed=42, canvas=DicomCanvas.RINGS)
)
dataset = dcmread(BytesIO(fixture.dicom_bytes))
assert dataset.pixel_array.shape == (512, 640)
assert dataset.BurnedInAnnotation == "YES"
assert fixture.lineage.accepted
assert len(fixture.lineage.masks) == 5
assert reproduce_dicom_overlays(fixture).digest == fixture.digest
report = fixture.to_json()
```

`dicom_bytes`, `image_png`, `background_png`, `phi_mask_png`, `metadata` and `text` are explicit synthetic test payloads. Callers can save them deliberately in their fixture directories. Object representations and `to_json()` omit their raw values. Reports contain hashes, controlled field/role codes, coordinates, canonical labels and provenance.

## Actual DICOM and aligned truth

The output uses Secondary Capture Image Storage, Explicit VR Little Endian, native unsigned 8-bit MONOCHROME2 pixels and modality `OT`. Rows, columns, sample count, bit depth and PixelData agree with the PNG. Odd pixel counts receive the required DICOM padding byte without changing the image. Study, series and instance UIDs are deterministic and bound to the specification and actual rendered pixels; file-meta and dataset instance UIDs agree.

Five generated identifiers are present in both metadata and painted pixels:

| Metadata field | Canonical privacy label |
| --- | --- |
| `PatientName` | `PERSON` |
| `PatientID` | `ID_NUM` |
| `PatientBirthDate` | `DATE` |
| `StudyDate` | `DATE` |
| `AccessionNumber` | `ID_NUM` |

Study, series and instance UIDs also carry `ID_NUM` metadata truth, without being painted. The generated fixture deliberately retains identifiers and declares `BurnedInAnnotation=YES` and `PatientIdentityRemoved=NO`. These flags describe the redaction test input; the generator is not an anonymizer.

Each painted text segment has an exact span in the protected `text` buffer, a tight half-open pixel box, its actual font size and a tight grayscale glyph mask. Identifier segments link directly to their metadata field and label. Field labels, orientation marks, acquisition text and the synthetic disclosure are separately classified. The combined PHI mask is the union of the five identifier masks. No annotation or ruler is allowed to overwrite another text segment unnoticed.

The pixel ruler includes its tick positions, spacing, bounding box and actual mask. It measures image pixels only: there is no millimetre calibration or anatomical scale. Rings, gradient and bounded noise canvases are generated patterns, not scans or plausible clinical images. `L` and `R` are synthetic annotation stressors, not verified patient orientation.

The mask lineage binds privacy rectangles to the SHA-256 of the actual DICOM bytes and their pixel geometry. Source node IDs are the first 32 hexadecimal characters of the corresponding annotation ID. The initial lineage is an identity mapping; downstream consumers can use the existing transform API to test cropping, scaling or rotation separately.

## Reproducibility and limits

The receipt binds the seed, canvas, geometry, generated fields, font bytes, dependency/runtime fingerprint, actual DICOM/PNG bytes and masks. `reproduce_dicom_overlays` regenerates the fixture and compares the complete receipt. It rejects changed content, pixels, metadata, labels, geometry, fonts or runtime. Receipts are reproducibility evidence, not signatures or a certification that arbitrary external data is synthetic.

Defaults are 640 × 512 pixels and an 18-pixel maximum font size. Dimensions can range from 384 to 1024 per side; font size can range from 12 to 24. Long generated fields may shrink to 10 pixels to fit, with the actual size recorded. If a field cannot fit, generation fails rather than truncating it.

The default font is Pillow's explicit embedded Aileron face, recorded as CC0-1.0 and rendered from its digested bytes. A caller can provide an existing local `BurnedInFont` with its declared permissive license and face index. Missing fonts, unsupported glyphs and unavailable required shaping reject the fixture; the builder never searches platform font directories or downloads a replacement.

`DicomOverlayLimits` defaults to 1,048,576 pixels, 32 million mask-work pixels and 8 MiB per encoded payload. Overrides remain bounded. Failures return `DicomOverlayError("synthetic_dicom_overlay_rejected")` without backend exception context or raw values. These are cooperative bounds for trusted local assets, not process isolation for hostile font or codec inputs.

The fixture builder accepts no arbitrary image canvas, patient text or source DICOM. Its internally generated canvas path supplies patient-free fixture provenance. It does not claim anatomical fidelity, clinical correctness, redaction recall on real data or privacy compliance.
