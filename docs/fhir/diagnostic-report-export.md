# Conservative FHIR DiagnosticReport Export

`openmed.clinical.exporters.fhir.diagnostic_report` provides a typed,
conservative `DiagnosticReport` projection that preserves evidence. It
shapes data you supply into a FHIR R4/R5 `DiagnosticReport`, does not
infer a diagnosis from an uncertain `conclusion`, and does not turn
`conclusion` into `conclusionCode`. All examples use synthetic fixtures
only, such as a synthetic glioma report with LOINC `60568-3`.

The exporter is local and assist-only. It does not make a clinical
decision, certify compliance, or contact a remote validator.

## Usage

Build a synthetic report mapping and project it, then optionally bundle
it with other resources:

```python
from openmed.clinical.exporters.fhir import to_diagnostic_report
from openmed.clinical.exporters.fhir import to_bundle

report = {
    "status": "final",
    "code": {
        "coding": [
            {
                "system": "http://loinc.org",
                "code": "60568-3",
                "display": "Pathology synoptic report",
            }
        ],
        "text": "synthetic glioma report",
    },
    "conclusion": "synthetic conclusion: no acute findings",
    "conclusionCode": [
        {"text": "synthetic code: no acute findings"}
    ],
    "result": [{"reference": "Observation/syn-1"}],
    "presentedForm": [
        {
            "contentType": "text/plain",
            "data": "c3ludGhldGljIGRhdGE=",
            "title": "synthetic",
        }
    ],
}

resource = to_diagnostic_report(
    report,
    subject_reference="Patient/synthetic",
    report_id="dr1",
)

bundle = to_bundle([resource], doc_id="synthetic-doc-1")
```

`report_id` is emitted as `DiagnosticReport.id` when non-empty; when
`report_id` is not supplied, `report["id"]` is used if non-empty.
`subject_reference` takes precedence over `report["subject"]` when both
are present (`subject` and `encounter`/`composition` accept string refs
normalized to `{"reference": ...}` or mapping). `doc_id` is accepted
for the deterministic `Bundle` contract and does not introduce a clock or
network call.

## Input mapping

`report` is a mapping. Only allowlisted keys are copied; every other
top-level key is rejected fail-closed (nested keys inside `code` or
`effectivePeriod` are not allowlist-checked). Scalar fields are
type-gated fail-closed: `conclusion`/`effectiveDateTime`/`issued` must be
string, `effectivePeriod`/`text` must be mapping,
`encounter`/`composition` must be `{"reference": ...}` or string ref
(normalized to `{"reference": ...}`). List fields preserve order with a
deep copy (nested dicts are isolated).

| Input key | FHIR target | Shape |
| --- | --- | --- |
| `status` | `DiagnosticReport.status` | string, normalized (see below) |
| `code` | `DiagnosticReport.code` | `CodeableConcept` mapping |
| `subject` | `DiagnosticReport.subject` | string ref or `{"reference": ...}` |
| `category` | `DiagnosticReport.category` | `list[CodeableConcept]` |
| `identifier` | `DiagnosticReport.identifier` | `list[Identifier]` |
| `basedOn` | `DiagnosticReport.basedOn` | `list[Reference]` |
| `encounter` | `DiagnosticReport.encounter` | `Reference` |
| `effectiveDateTime` | `DiagnosticReport.effectiveDateTime` | `dateTime` string |
| `effectivePeriod` | `DiagnosticReport.effectivePeriod` | `Period` mapping |
| `issued` | `DiagnosticReport.issued` | `instant` string, caller-supplied only |
| `performer` | `DiagnosticReport.performer` | `list[Reference]` |
| `resultsInterpreter` | `DiagnosticReport.resultsInterpreter` | `list[Reference]` |
| `specimen` | `DiagnosticReport.specimen` | `list[Reference]` |
| `result` | `DiagnosticReport.result` | `list[Reference]` |
| `imagingStudy` | `DiagnosticReport.imagingStudy` | `list[Reference]` (R4) |
| `study` | `DiagnosticReport.study` | `list[Reference]` (R5) |
| `media` | `DiagnosticReport.media` | `list[{link, comment}]` |
| `composition` | `DiagnosticReport.composition` | `Reference(Composition)` (R5) |
| `conclusion` | `DiagnosticReport.conclusion` | `string` |
| `conclusionCode` | `DiagnosticReport.conclusionCode` | `list[CodeableConcept]` |
| `presentedForm` | `DiagnosticReport.presentedForm` | `list[Attachment]` |
| `note` | `DiagnosticReport.note` | `list[Annotation]` (R5) |
| `supportingInfo` | `DiagnosticReport.supportingInfo` | `list[Reference]` (R5) |
| `text` | `DiagnosticReport.text` | `Narrative` mapping |
| `extension` | `DomainResource.extension` | `list[Extension]` |
| `modifierExtension` | `DomainResource.modifierExtension` | `list[Extension]` |

`code` defaults to `{"text": "synthetic diagnostic report"}` when missing
or empty. Pass a mapping; a string value is rejected. `result`,
`supportingInfo`, and `presentedForm` are preserved verbatim with stable
ordering.

No inference is performed from `conclusion` to `conclusionCode`. Both are
copied only when you supply them.

## Status handling

`status` is explicit. Missing, `None`, empty, or whitespace-only values
emit `"unknown"`:

```python
to_diagnostic_report({})["status"]  # -> "unknown"
to_diagnostic_report({"status": ""})["status"]  # -> "unknown"
to_diagnostic_report({"status": "   "})["status"]  # -> "unknown"
```

Values are casefolded and trimmed, so `"FINAL"` becomes `"final"` and
`"Entered-In-Error"` becomes `"entered-in-error"`.

Invalid values fail closed:

```python
to_diagnostic_report({"status": "bogus"})
# raises ValueError("invalid value for field 'status'")
```

The exception names only the field. The raw value is never echoed. Valid
codes are `registered`, `partial`, `preliminary`, `final`, `amended`,
`corrected`, `appended`, `cancelled`, `entered-in-error`, and `unknown`.

## R4/R5 shape validation

The emitted resource is validated against the R4/R5 **union** allowlist
(23 resource fields + 9 base fields = 32 keys) with minimal scalar
type gates. This is field-name allowlisting, not full FHIR type or
cardinality validation and not per-version strict validation:

- Base 9: `resourceType`, `id`, `meta`, `implicitRules`, `language`,
  `text`, `contained`, `extension`, `modifierExtension`.
- Resource 23: `identifier`, `basedOn`, `status`, `category`, `code`,
  `subject`, `encounter`, `effectiveDateTime`, `effectivePeriod`,
  `issued`, `performer`, `resultsInterpreter`, `specimen`, `result`,
  `imagingStudy`, `study`, `media`, `composition`, `conclusion`,
  `conclusionCode`, `presentedForm`, `note`, `supportingInfo`.

Both `imagingStudy` (R4) and `study` (R5) are accepted so a valid report
from either version passes the field-name gate; a resource containing
both passes the union gate even though it would be invalid in strict R4.
`composition`, `note`, and `supportingInfo` (R5 additions) are likewise
accepted. `effectiveDateTime` and `effectivePeriod` are mutually
exclusive; providing both is rejected. Any other top-level key is
rejected:

```python
to_diagnostic_report({"status": "final", "inferredField": "x"})
# raises ValueError("unsupported field 'inferredField'")
```

The error names only the unsupported field (e.g. `"unsupported field
'inferredField'"`) and never echoes the raw value. This is the same
field-name fail-closed boundary used by offline validators; FHIR
type/cardinality, terminology, and profile checks are a separate,
opt-in layer and do not replace this boundary.

## PresentedForm and evidence links

`presentedForm` is a `list[Attachment]` preserved verbatim with deep copy
(nested dicts are isolated from the caller). Attachments may carry
`contentType`, `data`, `url`, or `title`; the exporter does not
interpret or fetch them:

```python
report = {
    "status": "final",
    "presentedForm": [
        {"contentType": "text/plain", "data": "c3ludGhldGljIGRhdGE="},
        {"contentType": "application/pdf", "url": "http://example/synthetic.pdf"},
    ],
}
```

`result` and `supportingInfo` are `list[Reference]` values such as
`Observation/syn-1`. Order is preserved deterministically for downstream
Bundle reference rewriting. Logs are not emitted by this exporter; exceptions are field-name-only
(`"invalid value for field 'status'"`, `"unsupported field 'x'"`) and
never copy raw values or attachment bytes into the diagnostic message. For bundle-level reference integrity
counts, see [FHIR Reference Integrity Reports](./reference-integrity.md).

## Determinism and offline

The exporter is deterministic and offline:

- No network call, no telemetry, no mandatory remote validator.
- No wall-clock dependency; `issued` is emitted only when you supply it.
- No `random` or `uuid4` usage; the same input produces byte-identical
  JSON (`json.dumps(sort_keys=True)` stable) and preserves list order.
- `to_bundle()` assigns stable `urn:uuid` `fullUrl` values seeded by
  `doc_id` plus resource index and rewrites only in-Bundle references.

```python
import json

a = to_diagnostic_report(report, report_id="dr1")
b = to_diagnostic_report(report, report_id="dr1")
assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
```

## Privacy

Do not put raw PHI or direct identifiers in `report` values that may be
logged downstream. The exporter itself avoids PHI leakage:

- Exceptions reference only field names such as `'status'`, `'code'`, or
  `'unsupported field \'inferredField\''`; raw values are never echoed.
- `to_dict()`-style audit surfaces in related modules emit digests or
  counts, not payload bytes. Keep the same discipline when logging the
  returned `DiagnosticReport`.
- All fixtures in tests and docs are synthetic. The default `code` text
  is `"synthetic diagnostic report"` and examples use
  `Patient/synthetic` and LOINC `60568-3` with synthetic text.

## Scope

This is a local, mechanical projection for synthetic or already-governed
report data. It is an assist-only helper: it shapes what you provide
into a FHIR-typed resource, but it does not validate clinical meaning,
enforce a profile, or certify HIPAA, 21 CFR Part 11, or other compliance.
Keep de-identification, consent, and policy gates outside this call and
pass only synthetic or safely governed data in tests and examples.
