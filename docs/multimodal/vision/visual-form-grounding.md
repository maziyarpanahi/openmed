# Visual form grounding and reviewed export

`openmed.multimodal.vision.forms.field_grounding` associates recognized form values with field labels using their original page geometry, declared sections and explicit layout rules. It retains eligible alternatives and abstains on ambiguous associations. Even a unique proposal requires explicit reviewer confirmation before structured export. The module runs locally without recognition models, file writes or network calls.

This is a Python association/export layer. It consumes existing OCR, visual-control and handwriting observations; it does not detect boxes, infer sections, recognize handwriting, supply a Swift UI or validate clinical meaning. Applications must show the original evidence and authenticate the reviewer. It must not trigger clinical decisions automatically.

## Original evidence and typed observations

All observations use the original [fused evidence nodes](../fused-evidence-graphs.md), with source digest, page geometry and top-left box coordinates. Labels and printed values may use `OCR` nodes with exact character spans, or localized `PIXELS` nodes. Sections, controls and handwritten candidates use `PIXELS` nodes. Nodes have distinct IDs for their section, label and value roles. A source/page pair must use consistent original geometry.

| Input | Purpose |
| --- | --- |
| `FormSection` | Explicit original page extent of one form section. |
| `FormLabel` | Protected label text, parent section, accepted value kinds and expected direction of its value. |
| `FormValue` | Printed text, a handwritten candidate, or an explicit checkbox/radio state. |
| `VisualForm` | Bounded immutable collection of all current observations. |
| `GroundingPolicy` | Versioned geometric thresholds and ambiguity margin supplied by the application. |

Labels must be fully contained in their declared section. Values outside every section or within multiple sections produce abstentions. Candidate links cannot cross source files, pages or section membership. Input reading order never determines which label is matched. Repeated labels remain separate visual observations even when their strings are identical.

Printed values are preserved as strings. Handwritten values retain the complete [handwriting candidate](handwriting-uncertainty.md) and its alternatives; no best-score transcription is selected. Checkbox and radio observations require `CHECKED`, `UNCHECKED` or `UNKNOWN`. Unknown is never converted to false. These are observations supplied by a recognizer or reviewer, not a control-state detection algorithm; mutually exclusive radio-group semantics require a separate downstream validator.

## Deterministic association and abstention

The caller explicitly declares whether each label expects a value to its right, below or left. Original boxes are normalized with the existing box-normalization utility for comparisons; original coordinates remain unchanged in reports and exports.

A label is eligible when its value kind matches, its section contains the value, and the boxes satisfy the declared direction, maximum gap and minimum overlap. The gap is measured as a fraction of page width for left/right relationships and page height for below relationships. Axis overlap is the intersection on the perpendicular axis divided by the shorter box's side. A candidate score is `overlap * (1 - gap / maximum_gap)`, rounded to twelve decimal places, and must meet `minimum_score`.

All eligible associations are retained, sorted by descending geometric score and then opaque association ID for deterministic presentation. The ID tie-break never selects a winner. Scores within `ambiguity_margin`, including exact ties, produce `association_ambiguous`. Missing or overlapping section membership, no eligible association, unknown control state and multiple values competing for the same top label also produce abstentions. A unique proposal is `review_required`, never automatically accepted.

`geometry_score` measures geometric compatibility, not recognition reliability or clinical correctness. `confidence` separately retains the lower of the label and value recognition scores only when both are known; otherwise it is `None`. Handwriting has no aggregate winner score at this stage. Neither score is a calibrated probability. No default geometric thresholds or clinical operating points are supplied: configure and qualify a policy for the actual form layouts.

`FormGroundingReport.to_json()` omits label and value text while retaining original evidence, eligible alternatives, scores, abstention reasons and exact-input bindings. `to_review_packet()` adapts the findings to the existing review renderer with blocking review-required gates. The renderer normalizes citation identifiers; `review_citation_map` maps those identifiers back to the original nodes and boxes, and `to_json()` includes the corresponding node-ID map. Rendering the packet does not authorize export.

## Explicit review and protected export

`FormAssociationReview` identifies one value and one association listed in the current report. It binds the complete grounding digest, reviewer reference, review-event ID and digest of externally retained approval evidence. `reviewer_confirmed` defaults to false. The grounding digest includes all observations, source coordinates, protected text, policy and computed associations. Changing any input invalidates existing association reviews.

`export_grounded_fields` freshly computes grounding from the original observations and requires exactly one current, explicitly confirmed review per value. A reviewer may resolve an ambiguous result by selecting a listed alternative. They cannot invent an association, assign one label to multiple values or export an unknown control. An unmatched value must be corrected or removed through a new, separately reviewed input revision; export does not silently drop it. Missing, stale, duplicate, conflicting or incomplete decisions block the whole export, with no partial JSON returned.

Handwriting requires both transcription review and association review. Supply exactly one accepted `HandwritingReview` per handwritten candidate, and bind its exact `digest` in the corresponding `FormAssociationReview.handwriting_review_digest`. The export path freshly verifies the selected hypothesis or correction through the handwriting gate. Rejected, abstained, unconfirmed or stale transcriptions cannot be exported. Changing the reviewed reading requires a new association approval as well.

The resulting `GroundedFormExport.data_json` is protected clinical data: it contains label/value pairs, value kinds, original section/label/value citations and review lineage. Checkbox/radio values are booleans only after explicit known-state review. `to_report_json()` contains the same review and evidence metadata without label or value text. No date, dose, unit, locale, schema or FHIR/OMOP interpretation is performed.

Applications must retain and authenticate real approval evidence. These immutable data objects and digests prevent stale or mismatched decisions; they do not authenticate a caller that fabricates confirmed reviews. Association review establishes a reviewed transcription-to-field link, not clinical validity or permission to act on it.

## Synthetic example

This example uses synthetic labels, values and identifiers. The thresholds are API controls, not validated defaults. A real application obtains the selected association from a reviewer inspecting the original page.

```python
import json

from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.forms.field_grounding import (
    FormAssociationReview, FormLabel, FormSection, FormValue,
    GroundingPolicy, GroundingState, LabelRelation, ValueKind, VisualForm,
    export_grounded_fields, ground_form_fields,
)

def region(identifier, box):
    return EvidenceNode(
        node_id=identifier * 32, kind=NodeKind.PIXELS,
        source_digest="a" * 64,
        geometry=PageGeometry(page=0, width=1000, height=1000),
        region=box,
    )

section = FormSection(region("1", (0, 0, 1000, 500)))
label = FormLabel(
    evidence=region("2", (100, 100, 200, 120)), text="Synthetic field",
    section_id=section.evidence.node_id, relation=LabelRelation.RIGHT,
    accepts=(ValueKind.TEXT,),
)
value = FormValue(
    evidence=region("3", (230, 100, 300, 120)), kind=ValueKind.TEXT,
    text="Synthetic observation",
)
form = VisualForm(sections=(section,), labels=(label,), values=(value,))
policy = GroundingPolicy(
    version="synthetic-v1", maximum_gap=.2, minimum_axis_overlap=.5,
    minimum_score=.1, ambiguity_margin=.05,
)
report = ground_form_fields(form, policy)
finding = report.values[0]
assert finding.state is GroundingState.REVIEW_REQUIRED
assert finding.associations[0].confidence is None

# Synthetic explicit reviewer selection; no score automatically confirms it.
review = FormAssociationReview(
    value_id=value.evidence.node_id,
    association_id=finding.associations[0].association_id,
    grounding_digest=report.grounding_digest,
    review_id="4" * 32, reviewer_id="5" * 32, decision_digest="b" * 64,
    reviewer_confirmed=True,
)
exported = export_grounded_fields(form, policy, (review,))
row = json.loads(exported.data_json)["fields"][0]
assert row["label"] == "Synthetic field"
assert row["value"] == "Synthetic observation"
assert row["evidence"][2]["region"] == [230, 100, 300, 120]
assert "Synthetic observation" not in exported.to_report_json()
assert "Synthetic field" not in report.to_json()
```

## Privacy, bounds and verification

Protected strings are excluded from representations and report methods. Fixed `FormGroundingError` messages do not include source text, paths or reviewer names. Generic dataclass serialization is not a redaction mechanism. Opaque IDs, digests and geometry remain linkable metadata requiring application access and retention controls; hashes do not de-identify a record.

Each form accepts one to 32 sections and one to 128 labels and values. Labels and printed values contain one to 4096 valid UTF-8 code points with at least one non-whitespace character. Handwriting retains its own candidate bounds. The complete form and export allow at most 262,144 protected code points across labels, printed values, all handwritten alternatives and review corrections. Malformed geometry and declarations fail closed. The module does not log, persist or transmit content.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/vision/forms/test_field_grounding.py -q
```

Synthetic controls cover directional geometry, column/section isolation, reading-order independence, competing values, retained alternatives, unknown control states and confidence, scale invariance, original citation preservation, current association/transcription bindings and all-or-nothing export. They establish software behavior only, not recognition accuracy or clinical qualification.
