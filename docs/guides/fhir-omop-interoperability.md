# FHIR and OMOP Interoperability

This guide connects OpenMed's clinical grounding output to its FHIR R4 and
OMOP interoperability helpers. The complete local flow is:

```text
grounded spans -> FHIR resources -> transaction Bundle -> profile checks
                                                   -> $de-identify
                                                   -> bulk NDJSON de-identification
             \-> Athena/Usagi routing -> OMOP CDM tables
```

All examples below use fabricated clinical data. OpenMed does not bundle
Athena, UMLS, SNOMED CT, CPT, or other restricted vocabulary content.

## Run the self-contained example

The runnable
[`examples/interop_fhir_export.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/interop_fhir_export.py)
script constructs three `GroundedSpan` objects, exports them through the public
`to_fhir()` facade, performs a dependency-free Bundle smoke check, and prints
JSON. It does not download models or contact a terminology service.

```bash
uv run python examples/interop_fhir_export.py > /tmp/openmed-bundle.json
```

The same API accepts grounded spans from a real extraction pipeline:

```python
from openmed.clinical.exporters import to_fhir

bundle = to_fhir(
    grounded_spans,
    document_id="stable-document-id",
    subject_reference="Patient/patient-123",
)
```

`to_fhir()` maps supported canonical labels to `Condition`,
`MedicationStatement`, `Observation`, or `Procedure`. For an iterable it calls
the Bundle assembler, which assigns deterministic `urn:uuid` full URLs,
rewrites internal references, and adds transaction request blocks. Treat
grounding as advisory: review coding and assertion context before clinical or
billing use.

## Validate base R4 and declared profiles

The example's smoke check confirms the output contract without adding a FHIR
runtime dependency. It is deliberately not a complete R4 validator. Before
sending a Bundle to another system, run the HL7 validator or the receiving
server's validator against FHIR R4 (`4.0.1`).

OpenMed also provides `check_bundle()` for predictable, offline checks against
a local npm-package-style implementation guide snapshot:

```python
from openmed.clinical.exporters.fhir import check_bundle

for entry in bundle["entry"]:
    resource = entry["resource"]
    resource.setdefault("meta", {}).setdefault("profile", []).append(
        f"http://hl7.org/fhir/StructureDefinition/{resource['resourceType']}"
    )

outcome = check_bundle(bundle, "/opt/fhir/packages/hl7.fhir.r4.core")
errors = [
    issue
    for issue in outcome["issue"]
    if issue["severity"] in {"fatal", "error"}
]
if errors:
    raise ValueError("FHIR profile check failed")
```

The checker evaluates only profiles explicitly listed in each resource's
`meta.profile`. Declare the correct base or implementation-guide canonical URL
when building the resource; otherwise there is no profile constraint to check.
The checker covers cardinality, fixed and pattern values, locally enumerable
bindings, and selected slices. It does not execute FHIRPath invariants, fetch
packages, or call remote terminology servers.

## Check US Core locally

Download the US Core package through your normal FHIR package workflow and
keep it outside the repository. Add the appropriate US Core canonical URL to
each exported resource's `meta.profile`, populate every required US Core field,
then run the same checker:

```python
from openmed.clinical.exporters.fhir import check_bundle

us_core_outcome = check_bundle(
    bundle,
    "/opt/fhir/packages/hl7.fhir.us.core",
)
```

Use the US Core version required by the receiving system. A base-R4-valid
resource is not automatically US Core conformant. See
[WHO SMART Guidelines Profile Checks](../fhir-smart-guidelines.md) for package
layout, supported constraints, post-de-identification comparison, and safe
`OperationOutcome` handling.

## Bind terminology with Athena and Usagi

Load a caller-supplied Athena export and, optionally, an approved Usagi mapping
to route source codes to standard OMOP concepts. The router preserves mapping
status and vocabulary provenance; it uses concept ID `0` instead of fabricating
a match when a term is unresolved.

```python
from openmed.interop.athena import load_athena_vocab, load_usagi_mapping
from openmed.interop.omop import VocabularyRouter

vocabulary = load_athena_vocab(
    "/secure/athena-export",
    vocabulary_ids={"ICD10CM", "RxNorm", "LOINC"},
)
usagi = load_usagi_mapping("/secure/mappings/usagi.csv")
router = VocabularyRouter(
    vocabulary,
    usagi,
    vocabulary_version="2026-02-01",
)

mapping = router.route(
    "E11.9",
    source_vocabulary_id="ICD10CM",
    domain_hint="Condition",
    source_code_description="Type 2 diabetes mellitus without complications",
)
print(mapping.to_dict())
```

The FHIR exporter reads the selected `Candidate` on each `GroundedSpan` and
emits its coding system, code, display, score provenance, and vocabulary
version. The OMOP path uses the Athena/Usagi router to resolve standard concept
IDs and CDM domains. These are two projections of the same reviewed grounding
result; do not infer an OMOP concept ID from a FHIR code without terminology
resolution.

## Run the FHIR `$de-identify` operation

Wrap the Bundle in an R4 `Parameters` resource and call the local operation:

```python
from openmed.interop.fhir_operations import de_identify

result = de_identify(
    {
        "resourceType": "Parameters",
        "parameter": [
            {"name": "bundle", "resource": bundle},
            {"name": "policy", "valueString": "hipaa_safe_harbor"},
            {"name": "method", "valueCode": "mask"},
        ],
    }
)
```

The returned `Parameters` contains the de-identified Bundle, the applied policy
and method, and a PHI-free `OperationOutcome` manifest of changed element paths.
Run profile checks again after de-identification. Passing the original Bundle
as `original_bundle` lets `check_bundle()` distinguish pre-existing failures
from failures introduced by removal or masking.

## De-identify FHIR Bulk NDJSON

FHIR Bulk Data uses one resource per NDJSON line. Process it as a stream so raw
resources are not materialized as a second complete file:

```python
from openmed.interop.fhir_bulk import deidentify_ndjson

summary = deidentify_ndjson(
    "/secure/incoming/Patient.ndjson",
    "/secure/outgoing/Patient.ndjson",
    policy="hipaa_safe_harbor",
    method="mask",
)
print(
    summary.resources_deidentified,
    summary.error_count,
    summary.output_sha256,
)
```

Malformed lines are skipped and recorded by line number without copying their
content into errors. Protect the input path as PHI, review nonzero error counts,
and validate the de-identified output before release. For streams arriving over
a trusted transport, use `deidentify_ndjson_stream()` or
`deidentify_ndjson_async()` to avoid persisting raw input.

## Operational checklist

1. Use stable document IDs and synthetic data in tests.
2. Review grounding and terminology mappings before export.
3. Validate base R4, then the exact receiving profile version.
4. De-identify locally and re-run profile checks on the transformed Bundle.
5. Keep Athena and restricted terminology artifacts caller-supplied.
6. Treat raw Bundles and input NDJSON as PHI until policy checks pass.

For the lower-level Bundle and `OperationOutcome` APIs, see
[FHIR Interop Helpers](../fhir-interop.md).
