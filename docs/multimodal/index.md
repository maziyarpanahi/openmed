# Multimodal overview

OpenMed's local-first multimodal tools connect bounded media intake, document
extraction, redaction and content-free processing evidence. Use the navigation
groups in this section to follow a workflow from intake to review. Existing
guide URLs remain unchanged.

## Supported contracts and experimental surfaces

Here, **supported** means an implemented contract with the limits described in
its guide. It does not mean a provider, language, device or clinical use has been
qualified. Optional dependencies and caller-supplied local assets are required
where specified; no model weights or restricted datasets are bundled here.

| Surface | Available contract | Limits and experimental behavior |
| --- | --- | --- |
| Intake and preflight | Validated manifests, bounded type detection, resource limits and digest checks compose a deterministic accept/abstain report. | Acceptance is input-contract evidence, not a security approval or permission to run a clinical workflow. Missing evidence abstains. |
| Documents and PDF redaction | Format-specific extraction and redaction, source-offset projection, locally rendered redacted PDFs and fidelity checks. | Extraction is not redaction. A format's extraction guide does not promise an exporter. Scanned-page completeness needs separate OCR and residual-text review. |
| Images and OCR | Header and geometry checks, OCR word boxes, span projection and a local streaming Tesseract adapter. | OCR outputs remain **preview**. Confidence does not establish clinical correctness or identifier recall; selected-region pixel verification does not prove every identifier was selected. |
| DICOM | DICOM header and burned-in pixel redaction are listed in the [feature map](../feature-map.md); DICOM-SR provenance links findings to structural paths and offsets. | Provenance mapping does not interpret findings or copy DICOM values. Header redaction alone does not remove burned-in identifiers. |
| Audio | WAV metadata, ASR input-profile comparison, format summaries, resampling plans and bounded window plans. | These contracts do not transcribe, resample, downmix or qualify an ASR provider. Compatibility is a format statement only. |
| Video and provider evidence | Frame-sampling manifests and content-free provider-result envelopes. | A sampling manifest does not decode video; an envelope does not invoke or validate a provider. Model-driven interpretation and ambient clinical outputs remain experimental unless separately qualified with measured evidence. |

## Start with the preflight pipeline

1. Declare an opaque asset reference and digest using [asset manifests](asset-manifests.md)
   and the relevant [manifest profile](manifest-profiles.md).
2. Run the [preflight report](preflight.md): manifest validation, bounded
   [media-type detection](media-type-detection.md), metadata-profile validation,
   [resource limits](asset-limits.md), then [digest verification](asset-digests.md).
3. Handle [abstention reasons](abstention-reasons.md) explicitly before decoding
   or invoking a provider. Missing evidence must not become acceptance. In
   particular, the current PDF manifest lacks raster geometry: PDF preflight
   still abstains for unevaluated pixel limits. The separate
   [PDF geometry preflight](pdf-geometry-preflight.md) describes geometry checks;
   it does not silently change the manifest contract.
4. Select the appropriate local extractor or provider explicitly. Keep source
   content in the protected processing path and retain only content-free
   [provider results](provider-results.md) and
   [processing summaries](processing-summaries.md) as audit evidence.

Preflight does not decode media, run a model or choose clinical thresholds.

## Documents and image redaction

For digital PDFs, begin with [reading order](pdf-reading-order.md), retain source
geometry, project the selected text spans back to the document and follow
[redacted PDF rendering and fidelity](pdf-redaction-fidelity.md). That guide
distinguishes image-only export from the path-based selectable-text contract
and documents each verification boundary.

For scans and images, check [orientation](image-orientation-preflight.md) and
[image geometry](image-geometry.md) before [OCR](ocr-streaming.md). Use
[box normalization](box-normalization.md) and
[page-rotation transforms](page-rotation.md) to preserve coordinate meaning.
Review OCR completeness, reading order and residual identifiers separately.
Extracted text, OCR output and rendered media remain sensitive data.

Other document formats have their own contracts: [email](email-ingestion.md),
[EPUB](epub-extraction.md), [ODT/ODF](odt-extraction.md), [RTF](rtf-extraction.md),
[PPTX](pptx-redaction.md) and [XLSX](xlsx-cell-redaction.md). For shared source
mapping, see [document adapter provenance](../document-adapter-provenance.md).

## DICOM and audio

Use [DICOM-SR provenance](dicom-sr-provenance.md) to bind an opaque finding
reference to its content-tree path and source offsets. It is an evidence-linking
aid; review the underlying protected source before relying on a finding.

For audio intake, begin with [WAV preflight](wav-preflight.md) and
[WAV metadata](wav-metadata.md), then compare an explicitly selected local
[ASR audio profile](asr-audio-profiles.md). Consult
[format summaries](audio-format-summaries.md),
[resampling plans](audio-resampling-plans.md) and
[window planning](audio-windows.md) without treating a plan as an executed
transformation. [Frame-sampling manifests](frame-sampling-manifests.md) provide
the corresponding reproducibility boundary for video.

## Privacy and review boundary

Keep providers offline and use permissively licensed or caller-supplied model
assets. No PHI workflow gains a cloud fallback; Apple Foundation Models paths
remain on-device only. Diagnostics and audit evidence must contain only controlled
codes, counts, offsets, opaque references and digests, never raw source payloads,
transcripts, identifiers, credentials or private paths. Opaque references must
not encode patient information.

Multimodal outputs are **non-diagnostic**. Require **explicit reviewer
confirmation** of the protected evidence before any consequential use. A format
verdict, OCR score, redaction check or provider-success envelope must never
auto-trigger a clinical decision or imply clinical validation. This overview
contains no sample PHI and makes no provider qualification or release claim.
