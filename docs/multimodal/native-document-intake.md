# Native Document Intake Coordinates

OpenMedKit's Swift and Android document intake render PDF pages and decode images on the device, hand each page to a local OCR engine, and map every recognized region back to the original page. Both platforms emit the same page, crop and region records, in the same coordinates, with the same explicit outcomes. `openmed.multimodal.native_intake` is the Python reference mapping. The three implementations are checked against one synthetic fixture, `tests/fixtures/parity/document_intake_v1.json`.

The intake adds no OCR model and makes no network call. Pages are rendered into memory and released after recognition; nothing is written to storage. A mapped region is structural evidence for where text was recognized. It is not a claim that the OCR is correct, and it does not authorize disclosure or redaction export.

## Coordinate contract

Every record keeps the original page coordinates separate from the transform that produced the OCR image.

| Term | Meaning |
| --- | --- |
| Original page | Top-left origin, half-open boxes. Images use stored pixels before EXIF rotation. PDFKit pages use points on the media box. |
| Crop | Region of the original page handed to OCR. Absent means the whole page. |
| Rotation | Clockwise quarter turn (0, 90, 180 or 270) that made the crop upright. |
| OCR image | The upright crop at any render scale. Recognizer boxes use its pixels, top-left origin. |
| Span | Half-open Unicode scalar offsets into the document text, never UTF-16 indices or grapheme counts. |

An OCR box maps back in three steps, in this order: undo the render scale, undo the rotation, then undo the crop. The arithmetic order is fixed so that Python, Swift and Kotlin produce identical binary64 values; the fixture compares them bit for bit.

Two tolerances absorb normal engine behavior. Each OCR image dimension may differ from the upright crop's aspect by half a pixel, because rendering rounds to whole pixels. A box may overshoot the OCR image edge by up to 1% of that dimension and is clamped to the edge. A larger mismatch means the engine or renderer used a different coordinate space, so the page is held instead of being guessed.

Mapped pages are joined with a blank line (`"\n\n"`). Each mapped page produces one `page_geometry` visual record with the original extent and rotation, one `pixels` record for a declared crop, and one `ocr_span` record per region with its original-page box, display box and scalar span. These are the visual projections described in [OpenMedKit evidence contracts](openmedkit-contracts.md). Each record has an opaque reference: the first 32 hexadecimal digits of SHA-256 over `openmed.multimodal.document-intake.v1|<source digest>|<page>|<label>`, where the label is `page`, `crop` or `region:<index>`. The source digest is the SHA-256 of the document bytes.

## Outcomes and reasons

An intake is `complete` when every page was mapped, `partial` when at least one page was mapped and a reason applies, and `refused` when no page was mapped. Held pages contribute no text, so every region in a report indexes text that has a mapped region behind it. Reasons are value-free codes; no filename, path, page text or engine error appears in a report, description or exception.

| Reason | Applies when |
| --- | --- |
| `unsupported_document` | The media type or bytes cannot be read as a PDF or image. |
| `document_locked` | The PDF needs a password. |
| `permission_denied` | The document cannot be opened. |
| `size_limit` | The document exceeds the byte limit (64 MiB by default). |
| `empty_document` | The document has no pages. |
| `page_limit` | More pages exist than the page limit; later pages are not read. |
| `page_missing` | A declared page cannot be rendered. |
| `permission_lost` | Access ends after earlier pages were read; no later page is requested. |
| `recognition_failed` | The OCR engine failed on a page. |
| `resource_limit` | A page exceeds the region, text or render-pixel limit. |
| `ambiguous_transform` | The rotation is not a quarter turn, the crop is outside the page, the OCR image does not match the upright crop, or a box collapses when mapped. |
| `region_out_of_bounds` | An OCR box lies outside the OCR image beyond the tolerance. |
| `invalid_recognition` | A region's span does not index the page text, or a box is missing. |

Cancellation is not an outcome. It is thrown as `CancellationError` in Swift and `CancellationException` in Kotlin, after the open document and the current page image are released.

## Swift

`DocumentIntake` runs any `DocumentIntakeSource` with any `DocumentTextRecognizer`. `PDFKitDocumentIntakeSource` keeps each page's media box, crop box and `/Rotate`. `ImageIODocumentIntakeSource` reads every frame, takes EXIF quarter turns as the rotation, and holds mirrored orientations as `ambiguous_transform`. `VisionDocumentTextRecognizer` runs on-device Vision text recognition with language correction off, and reports word boxes. PDFKit and Vision are unavailable on watchOS; the mapper, protocols and ImageIO source build there.

```swift
import OpenMedKit

// syntheticURL points to a synthetic test document owned by the app.
let intake = DocumentIntake(recognizer: VisionDocumentTextRecognizer())
let report = try await intake.run(PDFKitDocumentIntakeSource(url: syntheticURL))
switch report.outcome {
case .complete, .partial:
    for region in report.regions(overlapping: 0..<12) {
        let box = region.sourceBox  // original-page points, top-left origin
        let packet = region.evidence  // validated `ocr_span` visual packet
    }
case .refused:
    let codes = report.reasons.map(\.rawValue)
}
```

Each page image is released as soon as its recognition ends, and the document is closed on every exit. Tests inject fakes for the source, document, image and recognizer. Visual packets are built with `MultimodalPacket.visual(...)`, which serializes canonically and admits the result through `MultimodalPacket.fromJSON`.

## Android

`DocumentIntake.intake(uri)` returns a `DocumentIntakeReport` with the outcome, reasons, `DocumentEvidence` records and, unless refused, a `DocumentIntakeResult` holding only mapped pages. `DocumentEvidence` mirrors the Swift records: `DocumentPageRecord`, `DocumentRegionRecord` and `DocumentVisualRecord` carry the box, rotation, crop and scalar span. `regionsForUtf16Span` converts native Kotlin indices for lookups.

```kotlin
val report = DocumentIntake(context = context, ocrAdapter = ocr).intake(syntheticDocumentUri)
when (report.outcome) {
    DocumentIntakeOutcome.COMPLETE, DocumentIntakeOutcome.PARTIAL -> {
        val evidence = report.evidence
        val regions = evidence.regionsForSpan(0, 12) // scalar offsets
    }
    DocumentIntakeOutcome.REFUSED -> report.reasons.map { it.code }
}
```

**Breaking:** `DocumentIntake.extract(uri)` now fails closed. It returns only complete intakes and throws `DocumentIntakeException`, whose message is the comma-separated reason codes, for partial and refused intakes. It previously returned text even when coordinates could not be mapped, and threw `IllegalArgumentException`, `IllegalStateException` or `SecurityException` on failures. Call `intake(uri)` to inspect partial results. OCR adapters must report token boxes inside the upright image they received; boxes that fall outside it by more than the tolerance now hold the page.

`OcrAdapter` token boxes use the upright image's pixels. `PdfPageRenderer` reports Android `PdfRenderer` pages in displayed points, because `PdfRenderer` does not expose the media box, crop box or `/Rotate`. A custom `PdfPageSource` that knows them declares them through `RenderedPdfPage.geometry`, and reports unreadable pages with `PdfPageEvent.Unavailable` from `renderPages`. A `DocumentImageSource` that hands OCR a crop of a larger original overrides `loadPage` to declare the original size and the crop.

## Python reference

```python
from openmed.multimodal.native_intake import (
    IntakePageGeometry,
    IntakePageInput,
    Recognition,
    RecognizedRegion,
    map_document,
)

page = IntakePageInput(
    geometry=IntakePageGeometry(width=400, height=300, rotation=270),
    recognition=Recognition(
        width=300,
        height=400,
        text="Synthetic text",
        regions=(RecognizedRegion(span=(0, 9), box=(30, 290, 50, 360)),),
    ),
)
report = map_document("a" * 64, [page])
assert report.regions[0].box == (40.0, 30.0, 110.0, 50.0)
assert report.to_dict()["outcome"] == "complete"
```

`IntakeReport.to_dict()` holds codes, offsets, references and digests only. The report text is excluded from it and from `repr`.

## Shared vectors and validation

The fixture authors each region as a truth box on a synthetic original page and derives the OCR box with an independent forward transform. Every platform must recover the truth exactly, or within one rendered pixel for the fractional render scale case. The cases cover multipage documents, all four rotations, crops, mixed Latin, Han, Devanagari, emoji and combining-mark text, edge overshoot, missing pages, permission loss, recognition failure, ambiguous transforms, out-of-bounds regions, invalid spans, page and region limits, and document-level refusals.

```bash
python -m pytest tests/unit/multimodal/test_native_intake.py -q
swift test --filter DocumentIntake
cd android && ./gradlew :openmedkit:testDebugUnitTest --tests '*DocumentEvidenceParityTest' --tests '*DocumentIntakeTest'
```

Regenerate the fixture with `python -m tests.fixtures.multimodal.document_intake` after an intentional contract change. The Swift tests also render synthetic PDFs through PDFKit and synthetic TIFFs through ImageIO, then recover the drawn region through the shared mapping. These checks establish coordinate parity; they do not measure OCR accuracy on real devices or documents.
