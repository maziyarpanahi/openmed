# FHIR Clinical Exchange Workbench

OpenMed's clinical exchange workbench has an explicit FHIR release boundary.
FHIR JSON does not identify its release, so import and export callers must
choose `R4` or `R5`; the default remains R4 for compatibility with the existing
FHIR exporters.

The supported-profile matrix is shipped as machine-readable JSON at
`openmed/interop/fhir/profile_matrix.json`. It is loaded and checked by the
focused profile tests, so a version change must update both the matrix and its
test evidence.

| Matrix id | Exact FHIR release | Package / guide | OpenMed support |
| --- | --- | --- | --- |
| `fhir-r4-core` | FHIR 4.0.1 (R4) | `hl7.fhir.r4.core#4.0.1` | First-class resource shaping and local structural checks |
| `fhir-r5-core` | FHIR 5.0.0 (R5) | `hl7.fhir.r5.core#5.0.0` | Loss-aware conversion subset from the R4 exchange surface |
| `ips` | FHIR 4.0.1 (R4) | `hl7.fhir.uv.ips#2.0.1` | Synthetic IPS patient-summary document shape |
| `ipa` | FHIR 4.0.1 (R4) | `hl7.fhir.uv.ipa#1.1.0` | Synthetic patient-access search Bundle shape |
| `clinical-document` | FHIR 4.0.1 (R4) | `hl7.fhir.uv.fhir-clinical-document#1.1.0` | Document Bundle, narrative, references, and provenance checks |

The exact FHIR core releases are [R4 4.0.1](https://hl7.org/fhir/R4/) and
[R5 5.0.0](https://hl7.org/fhir/R5/). The implementation-guide versions are
the published [IPS 2.0.1](https://www.hl7.org/fhir/uv/ips/en/),
[IPA 1.1.0](https://www.hl7.org/fhir/uv/ipa/STU1.1/), and
[FHIR Clinical Documents 1.1.0](https://build.fhir.org/ig/HL7/fhir-clinical-document/branches/1.1-fix-messages/en/).

## Version conversion

```python
from openmed.interop.fhir import FHIRVersion, convert_resource

r5 = convert_resource(resource, FHIRVersion.R4, FHIRVersion.R5)
r4 = convert_resource(r5, FHIRVersion.R5, FHIRVersion.R4)
```

The adapter covers the resource subset emitted by OpenMed, including
`Patient`, `Composition`, `Bundle`, `DocumentReference`, `Condition`,
`Observation`, `MedicationStatement`, `AllergyIntolerance`, `Procedure`, and
the related narrative/provenance resources listed in the matrix. R4
`MedicationStatement.medication[x]`, `reason[x]`, `context`, and release-specific
status values are mapped explicitly. Values with no lossless representation
are recorded in a dedicated preservation extension when the mapping is
defined; an unknown or unsupported field raises
`UnsupportedFHIRFieldError` with a resource path. Nothing is silently dropped.

## Privacy boundary

```python
from openmed.clinical.exporters.fhir import deidentify_fhir

safe_bundle = deidentify_fhir(
    bundle,
    document_id="synthetic-document",
    method="mask",
)
```

Narrative and free-text fields use OpenMed's local FHIR de-identification
operation. Identifier values, logical resource ids, Bundle `fullUrl` values,
and references are deterministically pseudonymized while coding systems,
supported codings, provenance structure, and internal links are retained.
Diagnostics and validation output contain paths and structural summaries only;
they do not echo resource values.

## Offline validation command

The local validator checks the supported subset, document Bundle structure,
profile release compatibility, internal `urn:uuid` references, IPS/IPA
requirements used by the examples, and clinical-document narrative. It does
not fetch terminology, invoke a remote validator, or establish legal,
regulatory, or certification compliance.

```bash
openmed fhir validate \
  --input tests/fixtures/fhir/synthetic_ips_r4.json \
  --version R4 \
  --profile ips \
  --json
```

The command emits a FHIR `OperationOutcome` in the result envelope and exits
with status 1 when a fatal or error issue is present. For complete
implementation-guide or terminology validation, run the receiving system's
approved local validator with the exact package versions shown above.
