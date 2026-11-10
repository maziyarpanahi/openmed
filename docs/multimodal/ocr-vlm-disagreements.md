# OCR and VLM disagreement routing

`openmed.multimodal.arbitration.ocr_vlm_router` compares paired OCR and visual-language-model readings using explicit field identities, original page geometry, text normalization, declared semantic types and recognition confidence. It accepts only agreements permitted by a caller-supplied policy and routes every other result to a bounded review queue with both visual citations. It never chooses a more fluent conflicting answer or synthesizes a third value.

The router is a deterministic local Python component with no model loading, file writes or network calls. Recognition and field pairing happen upstream. Agreement does not prove either recognizer is correct, and matching errors remain possible. This module does not grant clinical approval, infer field semantics or supply a Swift review UI.

## Original observations and explicit pairing

`ReadingCandidate` holds an opaque candidate ID and field ID, an OCR/VLM source tag, protected original text, a declared `SemanticType`, original [evidence node](fused-evidence-graphs.md), the existing provider `Confidence` value and an explicit `handwritten` flag. Confidence is an uncalibrated score or `None`, not a probability of clinical correctness. The caller must classify handwriting honestly; the router does not recognize handwriting style.

OCR observations use original `OCR` nodes with a character span matching the recognized string's length. VLM observations use localized `PIXELS` nodes. Both keep their source digest, page geometry and original top-left box coordinates. Neither normalized strings nor model-generated explanations replace visual evidence.

`ReadingPair` supplies one OCR and one VLM candidate intended for the same field. Pairing is explicit rather than inferred from reading order or string similarity. Wrong field identities, source files, pages or geometry become reviewable conflicts. Each batch requires unique comparison, candidate and evidence-node IDs, and each field may appear only once per source; duplicated fields cannot silently produce multiple accepted values.

For form extraction, field identities should refer to the original labels identified through [visual form grounding](vision/visual-form-grounding.md). This router does not replace that module's association review. Handwritten observations always require review, even when their text and confidence agree; use the separate [handwriting review](vision/handwriting-uncertainty.md) gate for actual transcription decisions.

## Agreement policy

`AgreementPolicy` requires all settings explicitly. There are no clinical defaults.

| Setting | Behavior |
| --- | --- |
| `version` | Non-sensitive version reference bound into the policy digest. |
| `allowed_types` | Declared semantic types permitted through this agreement gate. An empty tuple routes everything to review; `UNKNOWN` cannot be allowed. |
| `normalization` | `EXACT`, or opt-in `NFC_WHITESPACE` for Unicode NFC normalization and whitespace collapsing during comparison only. |
| `minimum_geometry_iou` | Positive minimum intersection-over-union of the original boxes. Source, page and geometry must also match. |
| `minimum_confidence` | Required known score for each observation independently; scores are never averaged. |
| `preferred_source` | Which original source string to retain after every agreement condition passes. |
| `maximum_review_items` | Complete review-queue capacity, from zero to 128. Overflow blocks the whole operation. |

NFC/whitespace normalization never applies compatibility folding, case folding, fuzzy matching, numeric conversion, dose interpretation or date parsing. For example, `0.5 mg` versus `5 mg`, `mg` versus `Mg`, and differently ordered date strings remain disagreements. The normalized string is used only for comparison. Output always contains the exact original string from the policy's preferred source, including its original whitespace and spelling.

Geometry uses the original page units. The source-page declarations must be identical; coordinates from different page sizes, units, rotations or source revisions are not silently treated as equivalent. Intersection-over-union uses rational arithmetic internally so tiny valid boxes do not underflow to zero area. Non-overlap fails every valid positive threshold. Threshold equality is permitted.

## Disagreement classes and bounded review

`route_ocr_vlm` retains every applicable reason in a deterministic order: field mismatch, source mismatch, page mismatch, geometry mismatch, text mismatch, semantic-type mismatch, unknown semantics, policy-restricted type, unknown confidence, confidence below policy and handwriting review. A high score never cancels another reason. Unknown confidence remains unknown even when the policy's score floor is zero.

A comparison with no reasons is `policy_permitted_agreement` and identifies the original selected candidate. Other comparisons are `review_required` and have no selected candidate. `RoutingReport.review_queue` contains every unresolved comparison, both original evidence nodes, candidate/field IDs, semantic types, known/unknown scores and disagreement reasons. It does not truncate the queue or discard conflicts when capacity is exceeded; the router raises a fixed `ArbitrationError` and returns no partial report.

`to_review_packet()` creates findings and blocking review-required gates through the existing review renderer. The packet contains no candidate text. Applications can retrieve original candidates from protected storage using their IDs and show the original page boxes. The renderer normalizes citation identifiers; `review_citation_map` maps them back to unchanged source nodes, and `to_json()` includes the corresponding node-ID map. Rendering a review packet never resolves a conflict or grants approval.

Conflict resolution remains a separate review workflow. Corrected readings must be represented as new current observations with their own retained evidence and approval lineage; the router offers no constructor that can turn a disputed value into an approved one. No third transcription is generated.

## Protected agreement preparation

`prepare_agreed_values` accepts current original pairs and policy, freshly routes the complete batch and refuses to return any data when one comparison needs review. It does not trust a previously created report. Its `AgreedValues.data_json` contains the selected original strings, both visual citations and all comparison metadata. `to_report_json()` contains decisions and evidence without recognized strings.

The input digest binds the complete original candidates, their protected text, field/source identities, geometry, confidence, classification, ordering and policy. Changed text or policy invalidates that binding even when opaque candidate IDs are reused. The digest is reproducibility metadata, not an authentication or authorization token.

Prepared values are intermediate observations. Applications must retain the independent transcription, form-association, privacy, schema and clinical review requirements before downstream use. In particular, a model agreement cannot mark a form association as reviewed or trigger a clinical decision.

## Synthetic example

The strings, identifiers and thresholds below are synthetic controls, not a validated clinical operating point.

```python
import json

from openmed.multimodal.arbitration.ocr_vlm_router import (
    AgreementPolicy, Disagreement, ReadingCandidate, ReadingPair, ReadingSource,
    SemanticType, TextNormalization, prepare_agreed_values, route_ocr_vlm,
)
from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.providers.base import Confidence

geometry = PageGeometry(page=0, width=1000, height=1000)

def reading(identifier, source, text):
    return ReadingCandidate(
        candidate_id=identifier * 32, field_id="f" * 32, source=source,
        text=text, semantic_type=SemanticType.TEXT, confidence=Confidence(score=.9),
        evidence=EvidenceNode(
            node_id=identifier * 32, source_digest="a" * 64, geometry=geometry,
            kind=NodeKind.OCR if source is ReadingSource.OCR else NodeKind.PIXELS,
            region=(100, 100, 300, 130),
            span=(0, len(text)) if source is ReadingSource.OCR else None,
        ),
    )

policy = AgreementPolicy(
    version="synthetic-v1", allowed_types=(SemanticType.TEXT,),
    normalization=TextNormalization.NFC_WHITESPACE,
    minimum_geometry_iou=.8, minimum_confidence=.8,
    preferred_source=ReadingSource.OCR, maximum_review_items=16,
)
pair = ReadingPair(
    pair_id="3" * 32,
    ocr=reading("1", ReadingSource.OCR, "Synthetic  observation"),
    vlm=reading("2", ReadingSource.VLM, "Synthetic observation"),
)
exported = prepare_agreed_values((pair,), policy)
assert json.loads(exported.data_json)["items"][0]["value"] == "Synthetic  observation"
assert "Synthetic" not in exported.to_report_json()

disputed = ReadingPair(
    pair_id="4" * 32, ocr=pair.ocr,
    vlm=reading("5", ReadingSource.VLM, "Synthetic conflicting reading"),
)
report = route_ocr_vlm((disputed,), policy)
assert len(report.review_queue) == 1
assert Disagreement.TEXT in report.review_queue[0].reasons
assert report.review_queue[0].selected_candidate_id is None
assert report.review_queue[0].evidence == (disputed.ocr.evidence, disputed.vlm.evidence)
```

## Bounds, privacy and verification

Batches contain one to 128 pairs. Each original string contains one to 4096 valid UTF-8 code points and at least one non-whitespace character. Total protected text is limited to 262,144 code points per batch. Bounds are deterministic and fail closed. Queue capacity is an additional bound independent of batch size.

Original text is excluded from representations, reports, review packets and fixed exceptions. `data_json` and in-memory candidate strings are protected data, never public diagnostics. Generic dataclass serialization is not redaction. Opaque IDs, digests, geometry and semantic metadata can remain linkable and need application access and retention controls; hashes do not de-identify records. The module does not log or persist them.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/arbitration/test_ocr_vlm_router.py -q
```

Synthetic controls cover exact and normalized agreements, unchanged output strings, conflicting doses/case/dates, source/field/page mismatches, geometry thresholds and tiny boxes, unknown confidence, semantic policy restrictions, handwriting review, complete bounded queues, stale input bindings and the all-or-nothing agreement gate. They establish software behavior, not recognizer accuracy, clinical qualification or a compliance guarantee.
