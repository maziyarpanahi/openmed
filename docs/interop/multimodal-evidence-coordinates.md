# Multimodal Evidence Coordinates

OpenMed can normalize PDF, OCR, image, and DICOM-derived evidence into the
same immutable `EvidenceLocator` contract used by structured source adapters.
The adapters preserve exact source coordinates and transformation provenance;
they do not create facts from malformed or ambiguous geometry.

All processing is local after the caller has installed any explicitly chosen
parser or OCR assets. Source bytes remain caller-held and are identified by a
content digest.

## Visual coordinate model

`CoordinateTransformChain` represents how a source page or frame was prepared
before a PDF parser, OCR engine, or image tool emitted a box. Supported,
invertible steps are:

- identity;
- clockwise rotation by 90, 180, or 270 degrees;
- crop;
- generic scale;
- OCR-specific scale.

Every step records its input and output dimensions. Consecutive steps must
have matching dimensions, so a missing resize or an incorrect rotation cannot
silently shift evidence.

```python
from openmed.interop.ingest import (
    CoordinateTransformChain,
    CoordinateTransformStep,
    EvidenceAdapterContext,
    VisualEvidenceAdapter,
    VisualEvidenceFrame,
    VisualEvidenceInput,
    VisualEvidenceRegion,
)

chain = CoordinateTransformChain(
    source_width=400,
    source_height=200,
    steps=(
        CoordinateTransformStep(
            operation="crop",
            input_width=400,
            input_height=200,
            output_width=320,
            output_height=160,
            parameters={"box": [40, 20, 360, 180]},
        ),
        CoordinateTransformStep(
            operation="ocr_scale",
            input_width=320,
            input_height=160,
            output_width=640,
            output_height=320,
        ),
    ),
)

source = VisualEvidenceInput(
    source_bytes=b"synthetic-image-bytes",
    source_format="ocr",
    source_version="synthetic",
    media_type="image/png",
    frames=(
        VisualEvidenceFrame(
            frame_id="frame_0001",
            page=1,
            coordinate_space="pixels",
            transform=chain,
            tolerance=0.5,
        ),
    ),
    regions=(
        VisualEvidenceRegion(
            frame_id="frame_0001",
            box=(200, 100, 400, 200),
            region_kind="ocr_word",
        ),
    ),
)

result = VisualEvidenceAdapter().adapt(
    source,
    EvidenceAdapterContext(
        source_id="source_0123456789abcdef",
        recorded_at="2026-01-02T03:04:05Z",
    ),
)
```

On success, the persisted `page_box` is expressed in the original source
space. Its transform metadata retains the observed box, page or frame identity,
source and observed dimensions, ordered transform chain, coordinate convention,
and declared round-trip tolerance.

## Bridge existing PDF and OCR output

`VisualEvidenceInput.from_document(...)` converts an `ExtractedDocument` whose
spans carry page boxes. `VisualEvidenceInput.from_ocr(...)` converts the common
`OcrResult` word-box contract. Callers supply one `VisualEvidenceFrame` per
source page or image frame, including any preprocessing chain.

Every normalized character span is retained alongside the source box. A span
without a box, a missing frame declaration, an out-of-bounds box, or a
round-trip error larger than the declared tolerance produces a typed
quarantine instead of a partial trusted artifact.

## DICOM and DICOM-SR provenance

`DICOMEvidenceAdapter` emits `dicom_element` locators containing explicit
Study, Series, and Instance UIDs plus a normalized DICOM tag. The values of the
referenced elements are not copied into operational metadata.

`DICOMEvidenceInput.from_sr_document(...)` bridges the existing DICOM-SR
`ExtractedDocument` output. Each normalized span retains its stable SR
`node_path` and points to the DICOM Content Sequence tag `0040,A730`.

## Typed failures and quarantine

The adapters preserve `unknown`, `partial`, `conflict`, `unsupported`,
`denied`, and `failure` states. Non-success output is a value-free
`StructuredEvidenceQuarantine`; it contains digests, counts, versions, and a
controlled reason code, never source bytes or extracted values.

When used with `IngestionToFactPipeline`, quarantine stops at source
adaptation. Later stage failures also leave the fact store unchanged. Replaying
the same source, transform chain, policy, and pipeline produces the same
artifact, locator, and result identifiers.

## Privacy and compatibility

Coordinate records use schema version `1.0.0` and compatibility policy
`same_major`. The bundled `multimodal_coordinate.schema.json` validates visual
and DICOM coordinate records; the extended evidence-adapter schemas validate
their containing results and quarantine records.

Authorized evidence payloads can include DICOM UIDs and source coordinates.
Do not write full locators or `to_dict()` output to ordinary logs, traces, or
metrics. Default representations hide locator locations and transforms.
Committed golden fixtures contain only synthetic dimensions, coordinates, and
identifiers.
