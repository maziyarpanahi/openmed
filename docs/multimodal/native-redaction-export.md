# Native Redaction Export

OpenMedKit's Swift and Android exporters turn a verified redaction result into a new, flattened PDF in private temporary storage, check that file, and hand it to the platform share or save interface only when the app asks. They build on [native document intake](native-document-intake.md): the same page sources render the pages, the same coordinates place the masks, and the same intake re-reads the output.

The exporter never changes the original document, never adds an annotation over it, and never copies anything from it except the rendered pixels of the pages that were examined. It makes no network call and shares nothing by itself.

## What goes in

An export request holds three things, and their SHA-256 `binding` identifies the artifact:

| Input | Meaning |
| --- | --- |
| Intake report | A `complete` report. A `partial` or `refused` report fails with `incomplete_report`, because some page was never examined. |
| Masks | `span` masks are Unicode scalar offsets into the report text. Every recognized region a span overlaps is painted, and every non-whitespace scalar in the span must belong to a region. `box` masks are boxes on a mapped page, in original-page coordinates, for content that OCR cannot read, such as a signature. |
| Policy | A caller policy identity (only its digest is bound), the padding in OCR-image pixels added on every side of a mask (2 by default), and the shortest masked token searched for during verification (2 scalars by default). |

The binding covers the source digest, a digest of the report text, the page count, the policy and the sorted, de-duplicated masks, so mask order and duplicates do not change it. Swift and Android compute the same value; both test suites assert one shared vector. Swift can build a request from a `DeidentificationResult`, masking every entity that was not kept, and refuses one computed on different text with `stale_request`. Android's `DocumentRedactionMask.fromActions` does the same for policy de-identification actions.

## The export profile

Every artifact uses one profile, `openmed.redaction-export.image-pdf.v1`. Each page is a single Flate-compressed 8-bit RGB image drawn over the whole media box by a fixed content stream. The file has no text layer or font, no annotation, attachment, form, action or outline, and no document information dictionary or XMP metadata. Both platforms write the same layout.

Pages are the upright crop that was handed to OCR, at the intake render scale. For a PDF this is the crop box, turned by `/Rotate`; the media box of the exported page is the crop size in points. For an image it is the declared crop, or the whole frame, turned upright by its EXIF quarter turn; one pixel becomes one point. Content outside the crop was never examined, so it is not exported.

Masks are mapped forward into page pixels by applying the crop, then the quarter turn, then the render scale, the inverse of the intake mapping. Each mask is rounded outward to whole pixels, padded and painted opaque black before the page is compressed.

## Verification before offering

The exporter writes the artifact to a pending file, reads it back from storage, and checks those stored bytes. The artifact is offered only if every check passes; otherwise the pending file is deleted and the export fails with a value-free reason.

| Check | Failure | What is verified |
| --- | --- | --- |
| `profile` | `profile_violation` | The file is exactly the profile layout, byte for byte apart from the image data. |
| `hidden_text` | `hidden_text` | No text layer or font. Swift also asks PDFKit for page text. |
| `attachments` | `attachments_present` | No annotations, attachments, forms, actions or outline. |
| `metadata` | `metadata_present` | No document information or XMP metadata. Swift also asks PDFKit for document attributes. |
| `page_count` | `page_count_mismatch` | One page per source page. |
| `visible_pixels` | `mask_not_opaque` | The artifact is re-read through `DocumentIntake`; every painted mask, inset by one pixel, is dark in the re-rendered page. |
| `residual_text` | `residual_text` | The caller's OCR re-extracts the artifact, the intake must be `complete` (otherwise `verification_incomplete`), and no masked token appears in the text. |

The residual search normalizes text with NFKC and lowercasing and splits it into runs of letters, marks and numbers. A masked token of at least the policy length fails the export when it appears as a whole extracted token, or, for scripts written without spaces such as Han, kana or Thai, inside one. An identifier that was masked in one place but also appears unmasked elsewhere therefore fails the export instead of leaking.

A rejected file reports the most specific category it carries: text or font structures as `hidden_text`, then annotations or attachments, then metadata, then `profile_violation`. The receipt of a successful export lists every check in this order.

## Lifecycle and invalidation

Artifacts live in a private directory: a folder in the app's temporary directory on Apple platforms, and in the app cache directory on Android. Each exporter owns a random session folder inside it.

- A failed or cancelled export deletes its pending output. Cancellation is rethrown as `CancellationError` or `CancellationException`.
- An artifact is deleted when the caller completes it after a share or save flow, when it is invalidated, and when the exporter is released (Swift) or closed (Android).
- A new exporter removes every session folder that no live exporter in the process owns, so leftovers from a previous run are swept at the next start.
- Every share or save action takes the caller's current request. When its binding differs from the artifact's, because the source, masks or policy changed, the artifact is deleted and the action fails with `stale_request`. `invalidate(keeping:)` deletes every artifact for an older binding ahead of time.
- The stored bytes are hashed again before every share or save. A changed file is deleted and the action fails with `artifact_changed`. No action ever returns the original document instead.

The export also re-opens the source and fails with `source_changed` when its digest, page count or page geometry no longer matches the report.

## Receipts and errors

Errors carry only a reason code, and their message is that code. A receipt holds the schema and profile, the binding, the source and output digests, the byte, page, mask and painted-region counts, and the passed checks. Neither includes text, file names, paths or the policy name.

| Reason | Applies when |
| --- | --- |
| `incomplete_report` | The report is not `complete`. |
| `mask_out_of_range` | A span is empty or outside the text, or a box is not finite or not inside a mapped page. |
| `mask_unplaced` | A span covers text that no recognized region places on a page. |
| `source_changed` | The source cannot be re-opened, or its digest, page count or geometry changed. |
| `render_failed` | A page cannot be rendered. |
| `resource_limit` | The artifact exceeds the document byte limit or a page exceeds the pixel limit. |
| `storage_unavailable` | Private storage cannot be created, written or read. |
| `stale_request`, `artifact_unavailable`, `artifact_changed` | A share or save action refers to an outdated, completed or modified artifact. |

The verification failures are listed in the table above.

## Swift

`DocumentRedactionExporter` is an actor. Pass the recognizer used for intake, export with the same source, and call `shareURL(for:matching:)` only when the user asks to share or save. The URL is for `UIActivityViewController`, `ShareLink` or a document exporter; call `complete(_:)` when that flow ends or is dismissed.

```swift
import OpenMedKit

// syntheticURL points to a synthetic test document owned by the app.
let recognizer = VisionDocumentTextRecognizer()
let source = PDFKitDocumentIntakeSource(url: syntheticURL)
let report = try await DocumentIntake(recognizer: recognizer).run(source)
let request = DocumentRedactionRequest(report: report, masks: [.span(0..<12)])

let exporter = try DocumentRedactionExporter(recognizer: recognizer)
let artifact = try await exporter.export(request, from: source)

// Later, from an explicit Share button:
let url = try await exporter.shareURL(for: artifact, matching: request)
let sheet = UIActivityViewController(activityItems: [url], applicationActivities: nil)
sheet.completionWithItemsHandler = { _, _, _, _ in
    Task { await exporter.complete(artifact) }
}
```

The exporter needs PDFKit, so it is unavailable on watchOS. The request, masks, binding, forward mapping and residual search are platform-neutral.

## Android

`DocumentRedactionExporter` takes the OCR adapter and page sources used for intake. `shareIntent` returns an `ACTION_SEND` chooser whose target intent carries only `FLAG_GRANT_READ_URI_PERMISSION` and the content URI as `EXTRA_STREAM` and clip data, so the platform grants read access to the single target the user picks. `saveTo` copies the verified bytes to a destination the user chose, for example with `ACTION_CREATE_DOCUMENT`. `complete` revokes the URI grants and deletes the file.

```kotlin
val report = DocumentIntake(context = context, ocrAdapter = ocr).intake(syntheticDocumentUri)
val request = DocumentRedactionRequest(report, listOf(DocumentRedactionMask.Span(0, 12)))
val exporter = DocumentRedactionExporter(context, ocr)
val artifact = exporter.export(request, syntheticDocumentUri)

// Later, from an explicit Share button:
val authority = "${context.packageName}.openmed.redaction-exports"
startActivity(exporter.shareIntent(artifact, request, authority))
// When the share flow returns, or the screen is dismissed:
exporter.complete(artifact)
```

Content URIs are served by `DocumentRedactionExportProvider`, a read-only, FileProvider-style provider that resolves only verified `<session>/<artifact>.pdf` files; pending, completed and invalidated files are not addressable. The core artifact does not declare it, so it is never exposed by default. An app that shares exports declares it, unexported and with URI grants; the provider refuses to start otherwise.

```xml
<provider
    android:name="com.openmed.openmedkit.export.DocumentRedactionExportProvider"
    android:authorities="${applicationId}.openmed.redaction-exports"
    android:exported="false"
    android:grantUriPermissions="true" />
```

Verification reads the artifact back with `DocumentRedactionPdfPageSource`, which accepts only the export profile. Pass `verificationPdfPageSource = PdfPageRenderer()` to have the platform renderer read it independently on a device.

## Validation

```bash
swift test --filter DocumentRedaction
cd android && ./gradlew :openmedkit:testDebugUnitTest --tests '*DocumentRedactionExportTest'
```

The tests use synthetic PDFs and images only. On the macOS host, Swift renders PDFs with an invisible text layer, a title and an annotation through PDFKit, and EXIF-rotated, cropped TIFFs through ImageIO, then checks that the artifact keeps none of them. The Android tests run on the JVM with Robolectric and inject synthetic page bitmaps. Both suites cover the shared intake vectors for the forward mapping, a residual-identifier negative control, tampered and foreign PDFs, stale and changed requests, cancellation, restart sweeps and the provider's read-only boundary. They do not exercise a real share sheet, a real OCR engine or a device.
