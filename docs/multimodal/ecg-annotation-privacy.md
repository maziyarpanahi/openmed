# ECG annotation privacy

`openmed.multimodal.ecg.annotation_privacy` inventories ECG headers, annotations, filenames, calibration text, original lead labels and rendered pages, then produces a coordinated redaction plan. Export rewrites structured values and burns identifier masks into fresh PNGs in memory. It returns the bundle only after every page and field passes fresh residual checks. The normalized waveform is preserved unchanged.

This is a Python orchestration slice over a complete host extraction, not a native EDF, WFDB, aECG or DICOM parser or rewriter. Supply every extracted field and selected page/frame from the original record. A digest of an incomplete extraction cannot establish coverage. The output remains protected clinical data: retained annotations, traces and linkable provenance are not an anonymity or compliance certification.

## Plan, review and export

1. Construct `EcgAnnotationSource` from the actual normalized `EcgWaveform`, the original labels and positions, an independent `LeadLayout`, every text field and every rendered page. Include patient and device header fields, annotation values, calibration-panel text, footers and other metadata. Filenames and original lead labels are inventoried automatically. Names as well as values are inspected.
2. Supply an already loaded `LocalAnnotationDetector` that returns `OpenMedSpan` records in original Python character coordinates. Every returned span is redacted, independent of its score, action or proposed replacement. The host must qualify the detector for the language and ECG surfaces and prohibit its logging, persistence and network access. `local_only=True` is a contract, not a network sandbox.
3. For pages, supply `LocalAnnotationOcr` around `StreamingTesseractEngine`, with separately installed native runtime and language data. This adapter sends an in-memory PNG over stdin and consumes bounded TSV stdout; it does not use the temporary-image path of the ordinary Tesseract wrapper. No model download, remote fallback or automatic engine selection occurs.
4. Call `prepare_annotation_privacy`. Review the original source against every field and region in the exact plan. Confirm extraction completeness, handwriting and OCR omissions, trace/label boundaries and calibration preservation. Unknown coverage or lead identity prevents export; missing OCR text alone never proves a page clean. Correct the source declaration or qualified detector when review finds a missed identifier, then generate and review a new plan.
5. Construct `AnnotationReview` with the exact plan digest, opaque reviewer reference, retained review-evidence digest and every reviewed field/region reference. These are host attestations, not authenticated signatures or independent proof. Call `export_annotation_privacy` with the unchanged source, policy and providers.

Export recomputes the plan, checks the complete review, rewrites fields, renders and decodes the exact PNG output, and runs OCR and identifier detection again. A changed source, changed provider configuration, missing review reference, residual identifier or failure on any page withholds the entire bundle. No source or destination file is written by this API. The caller decides how to retain the successful bundle under its own access controls.

## Synthetic structured example

This executable example uses a deliberately narrow synthetic detector and dummy evidence digests. They demonstrate the API and are not production qualification or reviewer approval.

```python
import json

from openmed.core.schemas.span import OpenMedSpan
from openmed.multimodal.ecg.annotation_privacy import (
    AnnotationField, AnnotationPolicy, AnnotationReview, EcgAnnotationSource,
    LocalAnnotationDetector, SurfaceKind, export_annotation_privacy,
    prepare_annotation_privacy,
)
from openmed.multimodal.ecg.lead_layout import (
    AcquisitionKind, LeadLayout, LeadObservation,
)
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform


def synthetic_detector(text):
    value = "Casey Example"
    start = text.find(value)
    if start < 0:
        return ()
    return (OpenMedSpan(
        doc_id="synthetic", start=start, end=start + len(value),
        text_hash="hmac-sha256:" + "0" * 64,
        entity_type="name", canonical_label="PERSON",
    ),)


waveform = EcgWaveform(
    (EcgLead.from_samples("I", [0, 0.1, 0, -0.1] * 25, unit="mV"),),
    100, EcgProvenance("array", "a" * 64),
)
source = EcgAnnotationSource(
    source_digest="a" * 64, extraction_digest="b" * 64,
    filename="Casey Example.ecg",
    fields=(
        AnnotationField(SurfaceKind.HEADER, "patient", "Casey Example"),
        AnnotationField(SurfaceKind.CALIBRATION, "scale", "10 mm/mV"),
    ),
    pages=(), waveform=waveform, layout=LeadLayout.digital(("I",)),
    observations=(LeadObservation(
        label="I", row=0, start_seconds=0, duration_seconds=1,
        sample_rate_hz=100, sample_count=100,
    ),),
    acquisition=AcquisitionKind.DIGITAL,
)
policy = AnnotationPolicy(
    version="synthetic-v1", evidence_digest="c" * 64, language="en",
)
detector = LocalAnnotationDetector(
    detect=synthetic_detector, model_digest="d" * 64,
    qualification_digest="e" * 64, local_only=True,
)
plan = prepare_annotation_privacy(source, policy, detector)
assert not plan.findings
# Only construct this after reviewing every source field and region.
review = AnnotationReview(
    plan_digest=plan.digest, reviewer_ref="1" * 32, evidence_digest="f" * 64,
    reviewed_field_refs=tuple(item.field_ref for item in plan.fields),
    reviewed_region_refs=plan.region_refs,
)
result = export_annotation_privacy(source, policy, detector, plan, review)
assert result.waveform is waveform
assert "Casey Example" not in result.structured_json
assert json.loads(result.structured_json)["filename"] == "ecg-redacted"
assert "10 mm/mV" in result.structured_json
```

## Rendered-page coverage

Each `AnnotationPage` contains an orientation-normalized Pillow image and `PageRegion` rectangles in original pixel coordinates. Bounds are half-open `(x0, y0, x1, y1)`. Regions must cover every pixel exactly once without overlap, holes or out-of-image coordinates. Use `RegionKind.ANNOTATION` with a `SurfaceKind` for text surfaces, `RegionKind.TRACE` with its actual layout `(row, column)` for each waveform strip, and `RegionKind.BACKGROUND` for the remainder. Every declared layout slot must have exactly one trace region across the supplied pages. Region identities are opaque 32-hex references, not patient identifiers.

Every OCR word must lie entirely inside annotation regions. Text on a trace or background produces an unresolved coverage finding, even when the detector does not classify it as PHI. The detector's original text spans map to whole OCR word boxes with explicit pixel padding. Masks cannot cross outside annotation regions, overlap retained OCR words or remove a declared lead-label region. An identifier that collides with a canonical structured lead label also holds export. Resolve ambiguous overlap through source review rather than silently damaging a trace, scale or lead identity.

Identifiers detected in any field or page propagate to matching text throughout the supplied source. Matching preserves original offsets, ignores case, tolerates whitespace differences and respects word boundaries. It reuses the PII normalization map for digit-width, native-digit and invisible-control differences, and propagates numeric tokens of at least six digits from confirmed identifier spans even when the original hit includes an `MRN:` label. It does not infer aliases, transliterations, OCR misspellings or unseen identifiers. The deterministic structured-identifier safety sweep runs independently of the model, so a partial model span cannot suppress a longer sweep finding. Both fresh detection and the known-value matcher run again on actual rewritten fields and post-render OCR.

Fresh RGB PNG encoding discards original comments, EXIF, ICC, hidden alpha content and source text layers. The raster writer verifies the decoded pixels and that pixels outside masks remain unchanged. This verifies rendering, not OCR recall. Empty OCR, a high confidence score or a clean residual pass cannot replace the required source-bound review and host evaluation. In particular, handwriting, low contrast, mixed languages, identifiers drawn into traces and incomplete upstream extraction can require refusal.

## Results, limits and validation

`RedactedEcgAnnotations.structured_json`, `png_pages`, `waveform` and `provenance_json` are protected payloads. Keep them together: the provenance retains original field offsets, pixel masks, region references, source/policy/provider digests and the exact review. Field and mask references are deterministic for an unchanged source and change when its bound content changes. `to_report_json()` returns only controlled codes, counts and digests; object representations omit raw text, pixels, offsets and waveform values. Digests are still linkable and need retention controls. Do not log payloads or provider outputs.

Default limits include 1,024 supplied fields, eight pages, 512 regions per page, 64 lead observations, one million combined source/OCR characters, 12 million pixels per page, 72 million pixels total and 32 MB encoded export. Provider outputs are bounded to 25,000 spans/words. Cross-surface matching reuses the custom recognizer's rule-count, rule-size and scan-work caps and can refuse a large inventory before export. `AnnotationPolicy` can lower the documented limits and set the explicitly qualified OCR padding. Resource refusals must not be treated as clean results or silently bypassed by dropping pages or identifiers.

`AnnotationPrivacyError` exposes only a controlled `PrivacyReason`; raw backend exceptions and their chained contexts are suppressed. Timing/sampling findings from the [lead-layout assessment](ecg-lead-layout.md) remain in provenance. Privacy export does not clear unknown acquisition timing or qualify the waveform for measurement. This slice adds no Swift acquisition surface, native-format rewrite, automatic clinical action or clinical OCR qualification.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/ecg/test_annotation_privacy.py -q
```

Synthetic controls exercise actual PNG redaction, unchanged waveform/calibration, all structured surfaces, Unicode identifiers, independent safety sweeps, cross-surface matching, malformed coverage, stale reviews, backend privacy, residual failures and whole-bundle refusal. A native-adapter control tests stdin PNG encoding and TSV parsing with an injected process result. These tests do not measure a downloaded model or real Tesseract's recall on clinical ECGs; those evaluations belong to the deployment's qualification evidence.
