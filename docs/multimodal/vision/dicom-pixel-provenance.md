# DICOM pixel provenance

`openmed.multimodal.vision.dicom.pixel_provenance` executes a bounded local DICOM display pipeline and records the actual intermediate pixels. It tracks transfer syntax, frame selection, decoding, rescaling, windowing, color conversion, orientation, resizing and PNG encoding. A record is accepted for downstream reuse only after the source is rendered again and the complete chain matches.

This is reproducibility evidence, not a diagnostic renderer, PHI redactor, cryptographic signature or proof that a model's interpretation is correct. Source bytes, rendered images, prompts and generated text remain protected in memory. Reports contain hashes and whitelisted numeric/enum parameters; they exclude patient tags, instance/series UIDs, filenames, OCR text and pixel values. Hashes and coordinates remain linkage metadata requiring appropriate access and retention controls.

## Actual operations and canonical records

`render_dicom_frame(source, spec=...)` returns a `ProvenancedFrame` containing an existing `ImageInput` and a `PixelProvenance` record. Each `PixelStage` binds its operation, parameter digest, actual output digest and predecessor digest. Operations that are identities are still recorded, so omitted transformations cannot masquerade as an equivalent workflow.

| Stage | Bound evidence |
| --- | --- |
| `transfer_syntax` | Complete original source digest, accepted syntax, explicit decoder and runtime fingerprint |
| `frame` | Actual native frame bytes or encapsulated frame payload, zero-based index, declared count and source-frame reference |
| `decode` | Decoded array after the selected decoder's unpacking, reshaping, subsampling and signedness handling |
| `rescale` | Executed slope/intercept pair and resulting array, or explicit color-image identity |
| `window` | Caller-supplied window and actual quantized output, or explicit color-image identity |
| `color` | Photometric conversion into RGB, including MONOCHROME1 inversion |
| `orientation` | Actual clockwise quarter turn in stored pixel coordinates |
| `resize` | Requested size, exact filter and actual RGB output |
| `encode` | Fresh PNG bytes, verified against the final decoded RGB pixels |

Array digests include dtype, shape and contiguous little-endian bytes. The runtime fingerprint binds the installed pydicom, NumPy, Pillow, JPEG codec and zlib versions without reporting arbitrary backend strings. `PixelProvenance.to_json()` includes the executed specification, whitelisted header parameters, stage chain and final pixel/image digests. It never serializes an arbitrary source dataset.

`reproduce_dicom_frame(source, provenance)` re-executes all operations and compares the entire record. Changes to source bytes, frame index, parameters, intermediate pixels, runtime fingerprint or encoded output reject reproduction. A decoder change is detected even if a later window clips both results to the same final image. Dependency upgrades require a new receipt; matching output pixels alone do not establish an identical execution history.

## Explicit local decoding and display choices

| `PixelDecoder` | Supported source path |
| --- | --- |
| `NATIVE` | Explicit or implicit VR little-endian; 8/16-bit monochrome, or unsigned 8-bit RGB/YBR_FULL/YBR_FULL_422 |
| `RLE` | DICOM RLE Lossless through pydicom's explicit local decoder; monochrome or supported 8-bit color |
| `JPEG` | Baseline 8-bit grayscale JPEG through the explicit local Pillow decoder |

No decoder is downloaded, auto-installed or selected by fallback. Source/header/frame/output limits are checked before work; RLE segment expansion and JPEG dimensions are checked before decompression. Other compressed syntaxes, color JPEG, unsupported sample layouts and ambiguous transforms fail closed. These are cooperative checks, not a native decoder sandbox; hard time/memory isolation belongs to the qualified runtime.

Monochrome input requires `DisplayWindow(center=..., width=...)`. The pipeline first resolves one complete rescale pair from the selected per-frame macro, shared macro or root dataset, in that order. Missing pairs default to identity; partial pairs are rejected. It then executes `floor(clip((value - (center - width/2)) / width, 0, 1) * 255)`. The caller's explicit window replaces any implicit selection of DICOM VOI presets. This is not the standard's default `LINEAR` center/width interpretation; DICOM distinguishes multiple [VOI functions](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html). Modality/presentation LUT sequences and inverse presentation LUT shapes are unsupported and rejected.

Color input requires `window=None` and identity rescale. Native planar RGB/YBR and native YBR_FULL_422 are normalized by the decoder, followed by explicit YBR-to-RGB conversion. Monochrome pixels become three equal RGB channels after any MONOCHROME1 inversion. The [pydicom pixel-array API](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.pixel_array.html) documents the processing performed even with `raw=True`; the recorded decode stage includes those effects.

`PageRotation` executes only lossless clockwise quarter turns. Patient orientation tags do not silently rotate the image. `output_size=(width, height)` applies the declared `ResizeFilter.NEAREST` or `BILINEAR` filter after rotation. No physical measurement or anatomical orientation is inferred from resized display pixels. Decoder correctness needs separate qualification; pydicom also documents [compressed-pixel limitations](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html).

## Executable synthetic example

This example constructs synthetic DICOM bytes, renders them, attaches a downstream region and verifies it by decoding the source again. All work stays in memory.

```python
from io import BytesIO
from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
from openmed.multimodal.vision.dicom.series_sampler import DisplayWindow
from openmed.multimodal.vision.dicom.pixel_provenance import (
    PixelRenderSpec, bind_region_provenance, render_dicom_frame,
    reproduce_dicom_frame, verify_region_provenance,
)
from openmed.multimodal.vision.providers.base import ImageRegion

ds = Dataset()
ds.file_meta = FileMetaDataset()
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
ds.SOPClassUID = CTImageStorage
ds.SOPInstanceUID = "1.2.826.0.1.3680043.10.543.2801"
ds.Rows = ds.Columns = 2
ds.SamplesPerPixel = 1
ds.BitsAllocated = ds.BitsStored = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
ds.PhotometricInterpretation = "MONOCHROME2"
ds.PixelData = bytes([0, 64, 128, 255])
buffer = BytesIO()
dcmwrite(buffer, ds, enforce_file_format=True)
source = buffer.getvalue()

frame = render_dicom_frame(
    source,
    spec=PixelRenderSpec(
        frame_index=0, window=DisplayWindow(center=127.5, width=255),
    ),
)
region = ImageRegion(
    image_digest=frame.image.digest, x0=0, y0=0, x1=1, y1=2,
)
bindings = bind_region_provenance(source, frame.provenance, (region,))
assert verify_region_provenance(source, frame.provenance, bindings) == (region,)
assert reproduce_dicom_frame(source, frame.provenance).image == frame.image
report = frame.provenance.to_json()  # Hashes and display parameters only.
```

## Downstream regions and local model evidence

`bind_region_provenance` accepts existing `ImageRegion` objects in the rendered image's coordinate space. It reproduces the source and records each actual crop's pixel digest plus the complete provenance digest. `verify_region_provenance` rerenders and compares chain, image, geometry and crop content before returning the verified regions. A structurally plausible or relinked hash chain is insufficient. Records are not signatures: they prove reproducibility, not who originally authored a region.

`run_provenanced_dicom_vlm(source, provider, prompt, spec=..., regions=...)` renders the image, calls the existing qualified local VLM contract and binds the provider's actual `VisualEvidence`. Optional regions restrict evidence to explicit rendered-space boxes. The result's `require_evidence(source)` revalidates pixels before exposing those evidence links and rejects model abstentions. `to_json()` includes only value-free model metadata, the chain and regional bindings; generated text remains in the protected `result` object and retains the existing human-review requirement.

The `frame_reference` uses the same source-digest/frame-index binding as the [DICOM series sampler](dicom-series-sampling.md). Join it to `SourceFrame.reference` when consuming a sampling manifest. This single-frame renderer does not claim that a series is complete or that an unselected frame was already examined. Series sampling policy and clinical adequacy remain separate from reproducing one explicitly selected frame.

Install the existing multimodal dependencies locally before use. The feature adds no dependencies or network calls. Run `python -m pytest tests/unit/multimodal/vision/dicom/test_pixel_provenance.py -q` for synthetic native/RLE/JPEG, transform, privacy and evidence-reproduction controls.
