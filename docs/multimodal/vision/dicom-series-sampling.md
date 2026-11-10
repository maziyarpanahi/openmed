# DICOM series sampling

`sample_dicom_series` selects frames from caller-supplied DICOM bytes before local VLM inference. Selection uses the decoded image's signal range and a small appearance preview, patient-space geometry, and acquisition metadata. The manifest lists every supplied frame as selected or unexamined. It never implies that the supplied objects form a complete study or that sampled frames exclude a finding elsewhere.

This Python orchestration module uses the existing optional `multimodal` dependencies: pydicom, NumPy, and Pillow. It does not download models, install codecs, open a network connection, or write patient material to disk. Native OpenMedKit model execution remains a separate integration.

## Supported input and coverage

Supply a tuple of complete in-memory objects belonging to one series. This implementation accepts native implicit/explicit VR little-endian, 8/16-bit, single-sample MONOCHROME1/2 images. It handles classic instances and enhanced multi-frame objects with explicit per-frame functional groups. Compressed, deflated, retired, color, malformed, or ambiguous objects are rejected before VLM dispatch. The [transfer-syntax preflight](../dicom-transfer-syntax-preflight.md) classifies other formats; this sampler does not silently decode or convert them.

Positions, orthonormal orientation vectors, pixel spacing, unique instance identifiers, and a frame of reference are required. Enhanced images use per-frame and shared functional groups; no slice position is invented from an instance number. Geometry follows the [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html). Identifiers are checked against [DICOM UID encoding rules](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_9.html), even when application parser warnings are disabled.

Frames with different coordinate systems, orientations, grids, or explicit acquisition/temporal/echo values cannot cover each other. The default requires at least one numeric acquisition discriminator. If a caller explicitly sets `require_acquisition=False`, each frame with no discriminator forms a separate group and must be selected; `unknown_acquisition` records those frames. Missing temporal metadata is never a guarantee that all acquisition phases are present.

Within each group, the sampler starts with the eligible frame having the largest display signal range, then adds frames until every supplied position is within `max_gap_mm` of a selected position. Distance is Euclidean in patient-space millimeters between corresponding pixel origins, not a measure of anatomical or diagnostic coverage. Stable source/frame digests break ties, so reordering the input tuple does not change the result. The deterministic greedy selection rejects a budget it cannot satisfy; it does not claim to find the mathematically smallest covering set.

Remaining slots may add frames whose 16-by-16 preview differs by at least `min_diversity_distance` in mean absolute 8-bit intensity. `min_signal_range` and preview diversity are selection heuristics, not image-quality certification or lesion detection. Low-signal frames remain explicitly unexamined and still count toward geometric coverage. If an acquisition has no eligible frame or the selected-frame budget cannot cover it, the operation raises `SeriesSamplingError`.

## Synthetic, offline example

The following creates only synthetic pixels and identifiers in memory. It does not read or write a clinical file.

```python
import io
import json

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from openmed.multimodal.vision.dicom.series_sampler import (
    DisplayWindow,
    SamplingPolicy,
    sample_dicom_series,
)

objects = []
for index in range(6):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = f"1.2.826.0.1.3680043.10.543.{index + 1}"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.1000"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.2000"
    ds.Modality = "CT"
    ds.AcquisitionNumber = 1
    ds.Rows = ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.ImagePositionPatient = [0, 0, index * 2]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.PixelData = np.arange(256, dtype="<u2").tobytes()
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    objects.append(stream.getvalue())

sources = tuple(objects)
window = DisplayWindow(center=128, width=256)
policy = SamplingPolicy(max_selected=4, max_gap_mm=4)
plan = sample_dicom_series(sources, window=window, policy=policy)
assert plan == sample_dicom_series(
    tuple(reversed(sources)), window=window, policy=policy
)
assert len(plan.selected) + len(plan.unexamined) == 6
assert plan.maximum_gap_mm <= 4
report = json.loads(plan.to_json())
print(report["schema_version"], len(plan.selected), len(plan.unexamined))
```

`DisplayWindow` is an explicit clipped linear transform after the declared rescale slope/intercept, with MONOCHROME1 inversion when required. The same transform produces previews and selected model images. It is not automatic clinical window selection or a complete DICOM VOI/presentation-state implementation. Choose and qualify the display window for the task; a saturated window can reject useful images.

Each entry in `plan.frames` maps a frame reference to its complete source SHA-256, zero-based frame index, and opaque coverage group. A reviewer can resolve selected or unexamined references against the original caller-owned bytes without placing filenames, UIDs, or patient coordinates in the manifest. `groups` records the frame count for each group; `frames_digest` binds the inspected geometry and preview signals for reproducible revalidation.

## Local inference and evidence scope

Pass the same sources, window, policy, and manifest to `run_sampled_dicom_vlm`, together with an already loaded local `EdgeVLMProvider` and a frame-scoped `Prompt`. The runner recalculates the complete selection and all bindings before making a provider call. Changing any source bytes, geometry, window, policy, or manifest invalidates the prior plan. The original prompt and generation limits are retained for each selected frame.

The runner decodes each selected frame to a fresh metadata-free PNG in memory and calls the existing [local VLM contract](edge-vlm-provider-contract.md). It never sends the original DICOM object to the provider. Pixel content can still contain burned-in identifiers: sampling is not de-identification. Apply and verify the appropriate image privacy processing before passing clinical material to a model.

`SampledSeriesInference.frames` holds protected generated drafts linked to exact frame references. `require_evidence(reference)` returns only a successful selected-frame result; an unselected frame or provider abstention raises a fixed-code error. `to_json()` includes examined and unexamined references, a manifest digest, and the mandatory selected-only, non-diagnostic human-review notice. Provider abstentions stay unexamined. A failure during inference withholds the entire returned result bundle, although already executed provider calls cannot be undone.

Display the scope notice with every downstream draft. Do not convert a frame-level statement into a whole-series conclusion. These APIs do not validate clinical accuracy or authorize autonomous decisions.

## Resource and privacy boundaries

`SamplingPolicy` bounds instance count, total frames, bytes per source and across all sources, metadata parsing, pixels per frame, and selected-frame count. Defaults allow 128 objects, 512 source frames, 64 MiB per object, 256 MiB total encoded input, a 1 MiB metadata boundary, 4 million pixels per frame, and at most 16 selected frames. Previews inspect all supplied native frames one at a time; only the selected set reaches the VLM. Inference revalidation repeats preview inspection intentionally.

These cooperative limits are not a process memory sandbox or a hostile-decoder isolation boundary. Full native pixel arrays and intermediate transforms occupy memory; use deployment isolation and the [VLM resource planner](vlm-resource-planning.md) for device/model admission. Parser calls are serialized internally while this thread's pydicom diagnostics are suppressed. Application code must not concurrently mutate pydicom or Python warning settings. Providers are trusted local code and must honor their own logging and network contracts.

Source bytes, pixels, raw UIDs, patient-space coordinates, acquisition values, prompts, and generated text are protected in-memory data. Public JSON reports contain only digests, opaque references, counts, policy-independent scope wording, and relative coverage. Digests remain linkable audit material, not anonymization. Keep source inputs and model drafts out of logs; serialize through `to_json()` rather than generic object dumping. Errors discard backend text and chained exception context. No patient tags, filenames, or pixel buffers are added to reports.
