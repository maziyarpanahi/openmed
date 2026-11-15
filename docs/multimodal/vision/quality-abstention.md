# Image quality and abstention

`openmed.multimodal.vision.safety` measures encoded PNG, JPEG and WebP images locally before visual extraction. `assess_image_quality()` compares deterministic aggregate measurements with an explicit task policy. `run_quality_gated_vlm()` performs that assessment before accessing the provider's metadata or inference method; an unmet threshold returns a report and no model result.

This is an opt-in whole-image gate. Existing `run_local_vlm()` callers are unchanged and must adopt the helper to enforce it. A passing image is suitable only for proceeding to the existing reviewed extraction flow. It is not a diagnosis, a completeness check, a PHI-safety certification or clinical validation. OpenMedKit parity is tracked separately in #3679.

## Supply thresholds for the task

There are no universal clinical quality defaults. Select and qualify thresholds against the intended capture device, modality, task, decoder version and representative evaluation data. The following executable example uses a synthetic checker pattern and deliberately illustrative thresholds; it is not a clinical policy.

```python
from io import BytesIO
from PIL import Image
from openmed.multimodal.vision.providers.base import ImageInput
from openmed.multimodal.vision.safety import QualityPolicy, assess_image_quality

image = Image.new("L", (64, 64))
image.putdata([
    48 if (x // 8 + y // 8) % 2 else 192
    for y in range(64) for x in range(64)
])
buffer = BytesIO()
image.save(buffer, format="PNG")
source = ImageInput(
    data=buffer.getvalue(), media_type="image/png", width=64, height=64,
)
policy = QualityPolicy(
    min_short_side=32, min_long_side=64,
    min_luminance=0.1, max_luminance=0.9,
    min_contrast=0.05, min_sharpness=0.00001,
    max_dark_fraction=0.5, max_bright_fraction=0.5,
)
report = assess_image_quality(source, policy)
assert report.accepted
audit_json = report.to_json()
```

For an existing local provider and whole-image request, call `report, result = run_quality_gated_vlm(provider, request, policy)`. If `result is None`, follow the report's reacquisition or human-review action. Successful results still enforce the provider contract's explicit human-review boundary. The helper refuses requests with selected regions because global statistics cannot establish the quality of an arbitrary crop. Prepare and assess a separately provenance-bound crop before using it as a whole-image request; the gate does not crop or rewrite source coordinates for you.

## Measurements and decisions

| Signal | Definition and decision |
| --- | --- |
| Resolution | Actual encoded width and height must match the declaration. The shorter and longer source sides must meet their respective policy minimums, independent of orientation. |
| Exposure | Mean grayscale luminance divided by 255, plus fractions of samples in 0–12 and 243–255. Policy bounds can identify underexposure, overexposure or excessive tonal clipping. |
| Contrast | Population grayscale standard deviation divided by 255. Low contrast can abstain independently of mean brightness. |
| Sharpness | Population variance of the four-neighbor discrete Laplacian, divided by 1020². Low values can indicate blur or insufficient texture; neither cause is diagnosed. |
| Expected blank boundary | Optional maximum foreground fraction on any of the four edges. A sample is foreground when it differs from the supplied background luminance by at least 32/255. Violations request reacquisition because content touches a margin expected to be blank. |

Measurements use Pillow's grayscale conversion and BOX downsampling, preserving aspect ratio within 512 × 512 pixels without upscaling. Resolution always uses the original source size. The sharpness statistic excludes the one-pixel outer ring. Unsupported analysis geometry, including a sampled side smaller than three pixels, abstains. Fixed processing and thresholds produce deterministic reports in a fixed decoder environment; requalify policies when changing preprocessing or decoder versions.

Boundary checking is disabled unless both `max_boundary_fraction` and `background_luminance` are supplied. It is intended for captures with a known blank margin, such as a document on a uniform background. It does not detect missing pages, clipped anatomy, an incomplete field of view or every cut-off text region. Global quality statistics can also miss a localized defect; callers remain responsible for task-specific validation and human review. No OCR, diagnostic content interpretation or model inference occurs during assessment.

Threshold violations return controlled reasons such as `insufficient_resolution`, `underexposure_or_dark_clipping`, `insufficient_sharpness` or `content_at_expected_blank_boundary`, with action `reacquire_image`. Corrupt headers, resource limits, geometry/format mismatches, unavailable decoders and unsupported images return `human_review_input`. Passing returns `continue_to_reviewed_extraction` and retains `clinical_review_required: true` in the report.

## Decode bounds and privacy

The input uses the existing `ImageInput` contract: at most 32 MiB of encoded bytes, without a filename. Bounded header readers enforce a 16-million-pixel decode ceiling before importing the decoder or allocating pixels. The decoded size and format are checked again. Only static 8-bit L, RGB and fully opaque RGBA images are accepted; animations, palettes, transparency and other modes require a separate reviewed conversion. Embedded EXIF orientation is not applied, and ICC profiles are not used for color management. These measurements are acquisition heuristics, not calibrated photometry.

Pillow is loaded only when evaluating an image and is already available through `openmed[multimodal]`; importing the safety package does not import it. Missing Pillow or an application configured to tolerate truncated images causes abstention. The gate does not enable permissive decoding or alter the application's decoder settings. Decoder warnings become abstentions. This helper serializes its decoder section while the shared Pillow warning context is active, ensuring concurrent assessments restore the original warning filters. Temporary filters prevent supported Pillow decoders from logging protected metadata on the current decoding thread; logger levels and other threads' records are preserved.

Reports contain the input digest, policy digest and thresholds, dimensions, aggregate metrics, controlled reasons and actions. They contain no pixels, text, paths, embedded metadata or exception details. The implementation writes no image files, logs no protected content and makes no network calls. Digests and aggregates remain linkable metadata and require appropriate application retention and access controls.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/vision/safety/test_quality_gate.py -q
```

Synthetic controls exercise real blur, exposure changes, low contrast, insufficient resolution, boundary contact, corrupt input, allocation limits, unsupported modes, metadata logging and suppression of provider calls after abstention. They establish software behavior, not clinical performance.
