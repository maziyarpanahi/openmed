# Mandatory multimodal review notices

Multimodal measurements, visual descriptions and ambient drafts are for clinician
review and are not diagnoses. A notice is part of the result contract, including
JSON and text displays. Nothing in this API triggers a clinical decision.

## Published catalog

| Meaning | Notice identifier |
| --- | --- |
| Measurement candidate, ECG interval, WSI aggregate | `openmed.multimodal.measurement_for_review.v1` |
| Visual description or image-quality finding | `openmed.multimodal.visual_description.v1` |
| Ambient draft | `openmed.multimodal.draft_for_review.v1` |

Each notice has fixed, patient-value-free wording: the output is for clinician
review only, is not a diagnosis, requires independent source review and explicit
confirmation before consequential use, and must never automatically trigger a
clinical decision. The catalog rejects interpolated values, unknown identifiers
and changed wording. Published identifier/text pairs are immutable. To revise
wording, publish a new versioned identifier and retain the frozen v1 commitments
and fixtures; updating text under an existing identifier fails the wording gate.

## Python results and CLI

The three wrappers own a digest-bound output reference, not the producer's
clinical schema. They work independently of pending ECG, WSI or ambient producers:
the caller computes the digest over the final protected output and retains that
output separately. A notice wrapper does not verify the digest's source or prove
clinical validity.

```python
from openmed.multimodal import (
    NOTICE_CATALOG, MeasurementReviewResult, NoticeKind,
)

result = MeasurementReviewResult(
    output_digest="a" * 64,  # synthetic reference, not measured model evidence
    notice=NOTICE_CATALOG[NoticeKind.MEASUREMENT],
)
print(result)  # catalog notice followed by the opaque output digest
encoded = result.to_json()
assert MeasurementReviewResult.from_json(encoded) == result
result.require_reviewer_confirmation(reviewer_confirmed=True)
```

`VisualDescriptionResult` and `DraftReviewResult` use their respective catalog
entries. Missing, altered or mismatched notices fail construction and parsing.
JSON requires `schema_version`, `output_digest`, `notice.identifier`, `notice.text`,
`requires_reviewer_confirmation: true` and `is_diagnostic: false`. Unknown fields,
duplicate keys, invalid UTF-8 and JSON over 64 KiB fail with value-free errors.

Render an existing wrapper through the offline CLI:

```sh
openmed multimodal-notice --kind measurement_for_review --input result.json
openmed multimodal-notice --kind measurement_for_review --input result.json --json
```

Text includes the identifier, full notice and output digest. JSON follows the
standard CLI envelope and retains the same notice in `data.notice`. Invalid
input returns status 2 without reflecting payloads or private paths.

## Vision runtime and Swift parity

Python `VisionLanguageGeneration` carries the fixed visual-description notice and
supports `to_dict`, `to_json`, `from_dict` and `from_json`. Use `render_text()`
to display protected generated text with the notice. Ordinary string and debug
rendering omit generated text and token IDs. Treat result JSON and rendered text
as protected content.

OpenMedKit exposes `MultimodalNoticeKind`, `MultimodalNotice`, and the immutable
`Codable` `MultimodalReviewResult`. Its initializer requires `kind`, `outputDigest`
and the matching `notice`. OpenMedKit `OpenMedVisionLanguageGeneration` requires
the visual-description notice and uses the same generation wire keys as Python.
Both runtimes attach the catalog notice when producing a generation. Shared
synthetic fixtures check the complete JSON objects, notice identifiers and text.
Use `renderText()` to display protected text in Swift. Both runtimes expose
the content-free release envelope separately from generation JSON.

Python `generate()` and `generate_vision_text()` return screened text. For a
review display, call `generate_with_metadata().render_text()`. The existing
Python constructor attaches the fixed catalog entry. Decoding in both languages
requires an explicit, matching notice in the serialized result.

## Review, registry and privacy boundaries

Existing ECG measurements, slide feature aggregates, image-quality assessments,
vision results and ambient notes expose their catalog entry as `review_notice`.
Their audit payloads retain their existing format. Use the notice-bound review
wrappers to serialize a result reference with its catalog notice.

Every output advertises reviewer confirmation and non-diagnostic status. The
confirmation method rejects false confirmation (and Python rejects truthy
non-booleans). It performs no clinical action. Hosts must verify reviewer
identity, independently validate source evidence and enforce confirmation at
their consequential-use boundary. A notice does not replace consent, clinical
validation, provenance or an authorization receipt.

`validate_notice_registry` checks covered frozen dataclasses for a declared notice
kind and a required notice field. The static registry test inventories public
multimodal data/result classes and the MLX vision result. Newly added classes
fail until registered as notice-bound outputs or explicitly reviewed as
non-clinical operational/ingestion types. Existing transport, preflight, geometry,
OCR ingestion and redaction reports are explicit non-clinical exemptions;
`ProviderResultEnvelope` continues to carry counts and digests only. Producer
clinical outputs must use a matching wrapper or implement the notice-bound
contract when those producers land.

No providers are invoked by this layer. Tests use fixed synthetic results and
negative controls, and establish neither provider qualification nor intended-use
claims. There are no new dependencies, model assets, cloud fallbacks or telemetry.
