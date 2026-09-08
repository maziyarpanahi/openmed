# Clinical-preserving privacy processing

`ClinicalPrivacyProcessor` is a preview integration over OpenMed's normalization,
detector, arbitration, policy, safety-sweep, and emission pipeline. It batches ONNX
inference while retaining each document's source offsets, controls, and outcome.
An executed request is not evidence that a language/model route is clinically
qualified. No language is qualified by default.

```python
from openmed.core.clinical_privacy import (
    ClinicalPrivacyDocument,
    ClinicalPrivacyOptions,
    ClinicalPrivacyProcessor,
)
from openmed.onnx import OnnxModel

model = OnnxModel.from_pretrained("/models/pinned-small-onnx", variant="fp32")
processor = ClinicalPrivacyProcessor(
    model,
    model_id="sidupadhyay/OpenMed-PII-SuperClinical-Small-44M-v1-ONNX",
    revision="deae3996155a44e002c7105adec3223853ef9949",
    batch_size=8,
    max_batch_tokens=4096,
)
results = processor.process_batch([
    ClinicalPrivacyDocument(
        id="synthetic-note-1",
        text="Patientin: Anna Beispiel. Keine Dyspnoe. Kurznarkose.",
        options=ClinicalPrivacyOptions(language="DE", method="mask"),
    )
])
assert results[0].status == "needs_review"
```

`complete` reports token processing coverage. `status` is `complete`,
`needs_review`, or `failed`. Review warnings include unqualified model/language
routes, uncertain or mixed language, narrowed policy, unresolved person roles,
and protected-term conflicts. Failed items have no transformed text. Model batch
failure fails all affected items; input/control failure remains per document.
IDs remain unique and outputs preserve input order. Deployment manifests must
establish qualification before setting `qualified_languages`.

## Language and policy

`DE`, `de-DE`, and `de_DE` select German; a region-specific value also selects its
locale. Conflicting language/locale controls fail. `auto` uses the existing SDK
language router with the optional `openmed[lid]` CLD2 adapter. Unreliable CLD2,
script-only fallback, and mixed paragraphs require review. Routing confidence
is a hint, not a clinical-accuracy measurement.

The versioned `clinical_preserve` profile masks names, identifiers, contact
details, addresses, and birth dates. Treatment dates, age, occupation, diagnoses,
medications, doses, measurements, and other clinical concepts remain unchanged.
German and English context rules supplement partial model names in explicit
patient/clinician headers and recognize anchored birth dates, phone numbers and
record identifiers. These bounded rules do not qualify free narrative text.

For German postal fields such as `Anschrift: Beispielweg 18, 10115 Berlin`,
the context detector covers the complete street, house number, postcode and
city span. It requires an explicit address header and a bounded postal grammar;
it does not treat an unanchored five-digit clinical number as a postcode.

`redact_categories` selects the explicit category expansion in
`openmed.core.clinical_policy.CATEGORY_LABELS`. Reducing the default set marks the
policy narrowed. `redact_roles` can select patient, clinician, or both. Unknown
name roles remain masked and require review when a role subset was requested.
`keep_labels` cannot contradict a selected redaction category. `keep_terms`
protects whole clinical spans from ambiguous name/location/organization labels;
it cannot silently exempt an identifier in a patient or doctor name field.
Custom terms outside the bundled vocabulary always require review,
including when the deployment otherwise qualifies the model/language route.
The same effective category/role policy is reapplied after the safety sweep.
Within this profile, partial-word predictions such as `Ke` inside `Keine` are
checked against the entire enclosing source word. A clinical word must match
the maintained vocabulary exactly; it cannot merely be a substring of a name.
Any overlap with an explicit personal-name context prevents this protection.

The maintained clinical vocabulary includes German regression terms and
negation cues. Context distinguishes a clinical eponym from an explicit name
field: `Morbus Parkinson` is preserved while `Patient: Parkinson` is masked.
This vocabulary is intentionally small and is not a substitute for an
independently reviewed clinical holdout.

Fragment protection also recognizes complete bundled phrases such as
`Morbus Parkinson` and `Morbus Crohn`. A prediction covering only `Mor` inside
the clinical phrase is suppressed; a matching fragment in
`Patient: Morbus Parkinson` still belongs to a person field and remains masked.
Spans that extend outside a protected phrase are not exempted by partial overlap.

## Output methods and privacy boundaries

- `mask` uses labeled placeholders; `remove` removes the selected spans.
- `replace` uses the SDK's locale-aware anonymizer with consistency scoped to
  one document. There is no cross-document raw-text mapping cache.
- `hash` requires a managed secret of at least 32 bytes on the processor and an
  explicit `pseudonym_scope` on the document. It uses domain-separated,
  scope-bound HMAC-SHA256 with a 128-bit output, not the legacy unkeyed hash
  operator. Hosting applications must bind the scope to the authenticated
  tenant. Keys, scopes and plaintext mappings are absent from response metadata.
- `shift_dates` requires an explicit non-zero `date_shift_days` within ten years.
  It selects treatment dates as well as birth dates, uses the SDK date shifter,
  and masks non-date identifiers. Unresolved date transformations require review.

Unchanged spans are emitted from the original source, preserving line endings,
paragraphs and punctuation. Detection can still miss identifiers; transformed
output remains sensitive until its qualification/review requirements are met.
Response spans contain offsets, labels, scores and actions, not original values.
The processor adds no client telemetry or raw-text result cache.

## ONNX execution

`OnnxModel.predict_batch` performs actual tensor batches with dynamic padding
and length buckets. `predict_batch_detailed` adds content-token counts, processed
counts and window counts. `max_length` is a window limit, not a truncation
instruction. Every window is built from the full tokenizer encoding, overlapping
tokens select the window with the most surrounding context, and decoding happens
once in original token order. Invalid offsets, uncovered token indices,
non-finite logits, output shape mismatch and exceeded budgets fail explicitly.

Both root and `onnx/` graph layouts work, including external weight files and
Hugging Face snapshot symlinks. Select a measured precision variant explicitly;
the low-level legacy `auto` order still prefers INT8 and is not a qualification
decision. Higher batch sizes do not guarantee higher CPU throughput.

`predict_batch_detailed` and `ClinicalPrivacyProcessor.process_batch` accept an
optional `execution_control` from `openmed.onnx.execution.OnnxExecutionControl`.
It applies one absolute deadline across windows, checks processing stages, and
signals `onnxruntime.RunOptions.terminate` during native inference. Cancellation
is scoped to those run options; it cannot cancel another caller's independent
session invocation. ORT termination is cooperative at runtime cancellation
points, not an OS process kill or guaranteed preemption of every provider
kernel. Coalescing services should cancel only when every caller in that model
batch has cancelled. Expired or cancelled executions never produce a successful
partial document.

The clinical processor validates the complete label head with the versioned
`clinical_label_map` projection. Source aliases such as `tax_id`, `fax_number`,
`device_identifier` and `health_plan_beneficiary_number` map to redacting
canonical categories. Explicit context attributes such as blood type remain
kept by this profile. Unrecognized labels fail validation instead of silently
becoming a kept `OTHER` entity.

The focused regressions live in `tests/unit/onnx/test_onnx_batch_inference.py` and
`tests/unit/core/test_clinical_*.py`. Their data is synthetic. Independent clinical
qualification, native/export parity, actual hosting performance, and safe
deployment/promotion remain application release gates.
