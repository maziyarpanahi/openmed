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
    doc_id="stable-document-id",
    subject_reference="Patient/patient-123",
)

print(bundle.summary.to_dict())
# {
#     "exported_by_label": {"CONDITION": 1, "LAB_TEST": 1},
#     "unmapped_by_label": {"BODY_SITE": 1},
#     "resource_count": 2,
#     "unmapped_count": 1,
# }
```

`to_fhir()` maps supported canonical labels to `Condition`,
`MedicationStatement`, `Observation`, or `Procedure`. For an iterable it calls
the Bundle assembler, which assigns deterministic `urn:uuid` full URLs,
rewrites internal references, and adds transaction request blocks. Treat
grounding as advisory: review coding and assertion context before clinical or
billing use.

Iterable export never guesses a resource from the coding system when a span
has an unrecognized canonical label. Such spans are omitted and counted in the
PHI-free `bundle.summary.unmapped_by_label` sidecar. The summary is not part of
the FHIR mapping, so normal JSON serialization emits only a valid R4 Bundle.
Labels awaiting a dedicated exporter can therefore coexist with supported
labels without aborting the document export. OpenMed never creates a Patient
resource in this path; `subject_reference` remains an external reference unless
the caller separately supplies a Patient to `to_bundle()`.

`document_id` remains accepted as a compatibility alias for `doc_id`.
`bundle_type` can select another Bundle type such as `batch`. For an unlabeled
span only, callers may opt into a coding-system route such as
`systems={"LOINC": "Observation"}`; a non-empty unknown canonical label is
still reported as unmapped.

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

## Run the grounding/export conformance suite

The synthetic round-trip suite exports the committed fabricated grounding
fixture through `to_fhir()` and `to_omop()`. It validates the transaction
Bundle with the official HL7 FHIR R4 validator, runs a deliberately malformed
Observation as a negative control, applies the local ACHILLES-style OMOP smoke
check, and writes the result as a `BenchmarkReport` in JSON and Markdown.

The validator JAR is not bundled with OpenMed. Fetch it explicitly through an
approved artifact workflow, verify its checksum, and keep it outside the
repository. The empty `grounding-validate` extra documents this opt-in boundary
without installing Java code or changing the local-first runtime:

```bash
uv pip install -e ".[grounding-validate]"
curl --fail --location \
  https://github.com/hapifhir/org.hl7.fhir.core/releases/download/6.8.1/validator_cli.jar \
  --output /opt/fhir/validator_cli.jar
echo "7d05b31196557a8ed2748d3c8a1646deba9b6600f5b6946be598949bd11eefe2  /opt/fhir/validator_cli.jar" \
  | sha256sum --check --strict
uv run python -m openmed.eval.suites.grounding_export \
  --validator-jar /opt/fhir/validator_cli.jar \
  --output grounding-export-validation.json \
  --markdown-output grounding-export-validation.md
```

Artifact download is a one-time, user-directed setup step. Suite execution
uses `-tx n/a`, performs no terminology or telemetry calls, and sends no
Bundle data out of process beyond the local Java invocation. If no JAR is
supplied, the command runs only the dependency-free structural preflight and
marks `official_validator_executed` as `false`; official conformance evidence
requires the JAR-backed run. Validator prose is never copied into the report:
only severity counts, stable failure codes, and SHA-256 evidence fingerprints
are retained.

The documented ACHILLES-style subset is intentionally smaller than full OHDSI
ACHILLES. It checks the four emitted core table names, complete loader-owned
columns, primary-key type/uniqueness, nonnegative concept IDs, resolvable
concept/person/visit/note/NOTE_NLP references, NOTE_NLP offsets, and reciprocal
NOTE_NLP-to-domain-event reachability. Full ACHILLES still requires a deployed
CDM database and remains out of scope.

## Check US Core locally

For OpenMed's four supported export profiles, use the bundled US Core 9.0.0
subset directly. It runs base R4 validation first and requires no package
download:

```python
from openmed.clinical.exporters.fhir import check_us_core

results = [check_us_core(entry["resource"]) for entry in bundle["entry"]]
errors = [finding for result in results for finding in result.errors]
```

The compact checker covers US Core Condition problems/health concerns and
encounter diagnoses, laboratory Observation, MedicationRequest, and
AllergyIntolerance. Missing must-support fields produce warnings; required
cardinality and bundled administrative binding violations produce errors. It
does not bundle or claim validation of external clinical terminology value
sets.

For another US Core version or profile, download the implementation-guide
package through your normal FHIR package workflow, keep it outside the
repository, declare its canonical URL in `meta.profile`, and use
`check_bundle()`. A base-R4-valid resource is not automatically US Core
conformant. See
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
