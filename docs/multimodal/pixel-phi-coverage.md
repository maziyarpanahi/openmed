# Pixel PHI coverage gate

`audit_pixel_redaction` inspects actual image pixels before returning a redacted PNG bundle. It requires complete regional coverage of every decoded frame and carrier, resolves every detected or known critical region against the redaction masks, executes the requested transforms, verifies the encoded pixels, and runs an explicit residual detector. Any failure withholds the entire bundle, including outputs from earlier frames.

This gate applies only to the returned PNGs. It does not certify the input image or DICOM file, detector recall, compliance, or clinical correctness. Original containers, metadata, text fields, hidden alpha values and embedded metadata thumbnails are excluded by fresh encoding. The source remains protected data. Reports contain coordinates, counts, digests and controlled reason codes; these are still linkage metadata and need appropriate access and retention controls.

## Qualified local detection

Pass both `detector` and `residual_detector` explicitly as `LocalPixelDetector` registrations. The callback receives a fresh RGB Pillow crop and returns `PixelInspection(critical_regions=..., complete=True)`. Boxes use integer, half-open, top-left crop coordinates. All reported regions are critical and must be masked. Set `complete=False` for incomplete, uncertain or truncated inference. An absent detector never counts as clean evidence.

The registration digest should cover the model revision, preprocessing, OCR configuration and identifier policy. `local_only=True` declares a qualified implementation that uses no network, disk writes or content logging; it does not sandbox third-party code. Provide a locally installed OCR/identifier or visual detector that has been validated for the modality, languages, overlay styles, tile boundaries and selected display window. Translate any OCR identifiers into crop-space boxes inside the adapter and discard recognized text before constructing `PixelInspection`. There is no model download, cloud fallback or built-in claim that an empty detector result proves absence of PHI.

The following executable example uses a synthetic bright marker to exercise the gate. This detector is only a demonstration and must not be used for clinical PHI detection.

```python
from io import BytesIO
from PIL import Image
from openmed.multimodal.privacy import (
    LocalPixelDetector, PixelFramePlan, PixelInspection,
    audit_pixel_redaction, inventory_pixel_frames,
)

image = Image.new("RGB", (16, 12), (30, 30, 30))
image.paste("white", (5, 3, 9, 6))
stream = BytesIO()
image.save(stream, format="PNG")
image.close()
source = stream.getvalue()

def synthetic_marker(crop):
    with crop.convert("L") as gray:
        with gray.point(lambda value: 255 if value > 250 else 0) as mask:
            box = mask.getbbox()
    return PixelInspection((box,) if box else ())

detector = LocalPixelDetector(
    digest="a" * 64, inspect=synthetic_marker, local_only=True,
)
frames = inventory_pixel_frames(source)
plans = tuple(
    PixelFramePlan(reference=frame.reference, masks=((5, 3, 9, 6),))
    for frame in frames
)
result = audit_pixel_redaction(
    source, plans, detector=detector, residual_detector=detector,
)
assert result.accepted and len(result.outputs) == 1
safe_png = result.outputs[0]  # Protected bytes: do not log or publish automatically.
report = result.to_json()   # Geometry and hashes only, without OCR text or pixels.
```

## Complete regions and resolved masks

`inventory_pixel_frames` decodes the source in memory and returns source-bound frame references, carrier types and dimensions. The audit decodes again and requires exactly one `PixelFramePlan` per actual frame. Stale source hashes, invented frames, duplicates and omissions fail. Reordering plans does not change the inventory/output order.

`inspected_regions=None` scans the whole frame, including borders. Explicit regions must cover the full frame as a union: overlap cannot hide a gap or double-count coverage. Every crop is passed to the detector and its boxes are translated back to source coordinates. Any `ignored_regions` entry holds export, even when another crop overlaps it. A caller must qualify crop context; geometric coverage does not establish that an OCR engine detects words split across tiles.

`masks` are the source-space black rectangles the caller intends to apply. Every detector finding and every caller-supplied `critical_regions` rectangle must be covered by their union. Known identifiers remain required even if the detector misses them. The gate does not silently dismiss findings or invent masks for the caller.

Optional `crop`, clockwise `PageRotation` and `output_size` execute in that order. Rotations are lossless quarter turns; resizing uses nearest-neighbor sampling. The [mask lineage implementation](redaction-mask-lineage.md) recomputes the executed transforms and outward-rounded target masks. Cropping away any protected support fails. The [raster exporter](pdf-redaction-fidelity.md) then paints black masks and checks decoded output pixels, including unchanged pixels outside masks. The residual detector inspects the entire decoded PNG, and any residual region or incomplete call holds the bundle.

## Raster and DICOM carriers

| Input | What is inventoried and exported |
| --- | --- |
| PNG, JPEG, TIFF, GIF, BMP, WebP | Every Pillow frame as a flattened RGB PNG, using stored pixel coordinates. Original metadata, EXIF orientation instructions and embedded metadata thumbnails are not copied. Animation timing and source container structure are not preserved. |
| Native little-endian DICOM | Every root frame, root `IconImageSequence` image and explicit overlay-plane frame, each as an independent PNG. The root image is not composited with the overlays. |

DICOM needs `source_kind="dicom"` and an explicit `DisplayWindow` from `openmed.multimodal.vision.dicom.series_sampler`. Supported pixels are 8/16-bit MONOCHROME1/2 and interleaved unsigned 8-bit RGB. Stored-bit sign extension, rescale slope/intercept and the explicit grayscale display window are applied before inspection. This is a display-raster export, not a diagnostic DICOM renderer or a preservation of modality precision.

Overlay planes are independently inventoried, including portions outside the image bounds, so an image crop cannot hide them. DICOM defines overlays as separate bit planes and permits origins outside the main image; a single-frame overlay can apply to all image frames. This bundle retains one separate raster per encoded overlay frame and does not interpret placement or presentation-state semantics. See the [DICOM Overlay Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.9.2.html).

Compressed/encapsulated DICOM, floating-point pixels, retired embedded overlays, unsupported nested pixel carriers, per-frame rescale transformations and ambiguous geometry fail closed. Original DICOM tags, UIDs, private elements, presentation states and arbitrary nested metadata never enter the output bundle. Passing this pixel gate is not permission to distribute the original DICOM object; use the separate DICOM header/carrier de-identification workflow for that purpose.

## Sampling, evidence and limits

An optional `SeriesSamplingPlan` from [DICOM series sampling](vision/dicom-series-sampling.md) can describe this single input object's VLM frame selection. The gate checks its source/frame mapping against the actual decoded root frames and records its digest and selection flags. It does not requalify the model sampling algorithm. Every unselected VLM frame, icon and overlay still requires privacy inspection; sampling never grants privacy coverage.

Each report records source and detector digests, actual regional inspection requests, ignored regions, known/detected/unresolved critical regions, source and output masks, executed transforms, residual findings and successful output hashes. `complete=False` marks a frame whose audit did not finish. A failed audit may contain evidence from completed earlier frames but has `outputs=()` and must not be treated as permission to release anything.

`PixelCoverageLimits` bounds encoded input/output bytes, frame count, per-frame and total pixels, and region count. Parsing and detector calls are cooperative; native decoder or callback isolation and hard deadlines belong to the qualified runtime. `cancel_check` is checked between decode, inspect and export work. No temporary files, source writes, network calls or raw exception payloads are produced by this orchestration layer.

Run the synthetic regression suite with `python -m pytest tests/unit/multimodal/privacy/test_pixel_coverage.py -q`.
