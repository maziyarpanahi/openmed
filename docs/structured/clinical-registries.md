# Governed clinical registries

OpenMed registries turn an immutable saved-cohort execution and its Journey
facts into versioned, evidence-bound registry cases. The registry layer keeps
cohort logic, field extraction, validation, completion, assignment, review,
adjudication, correction, privacy, and export rules in one content-addressed
definition.

Registry output supports governed review and data operations only. It does not
diagnose, treat, enroll, contact, order for, or otherwise act on a patient.

## Define and version a registry

Each field selects one fact type and declares accepted, unknown, and conflicting
statuses plus cardinality. The workflow pins the owner scope and exact privacy
and export policies.

```python
from openmed.structured.registry import (
    RegistryDefinition,
    RegistryFieldRule,
    RegistryWorkflowPolicy,
    version_registry_definition,
)

definition = RegistryDefinition(
    registry_id="registry_syntheticregistry1",
    cohort_definition_version_id="cohortdefinition_syntheticcohort01",
    cohort_definition_digest="sha256:" + "1" * 64,
    fields=(
        RegistryFieldRule(
            field_id="condition",
            fact_type="condition",
            required=True,
            allowed_statuses=("active", "corrected"),
        ),
        RegistryFieldRule(
            field_id="medication",
            fact_type="medication",
            required=True,
            allowed_statuses=("active", "corrected"),
        ),
    ),
    workflow=RegistryWorkflowPolicy(
        policy_id="registry_governance",
        version="1.0.0",
        owner_scope_id="ownerscope_syntheticowner01",
        privacy_policy_digest="sha256:" + "2" * 64,
        export_policy_digest="sha256:" + "3" * 64,
    ),
    definition_version="1.0.0",
)
version = version_registry_definition(definition)
```

The version identifier and definition digest change when any cohort binding,
field rule, workflow gate, or policy digest changes.

## Materialize cases

`materialize_registry_cases` accepts the definition version, a
`CohortExecution`, and `ClinicalFact` or `RegistryFactBinding` inputs. Only
fully resolved eligible cohort members become cases. Non-eligible membership
states remain visible as aggregate exclusion counts.

Cases never serialize fact values. Every field stores only its state, reason
code, opaque fact and evidence identifiers, value digests, derivation digests,
and correction ancestry. Input ordering cannot change a case or materialization
digest.

| Field state | Meaning |
| --- | --- |
| `present` | Accepted evidence satisfies the configured rule |
| `missing_required` | A required fact is absent |
| `unknown` | The source fact explicitly carries uncertainty |
| `conflict` | Conflicting status or excess distinct values need resolution |
| `corrected` | A replacement fact preserves its correction ancestry |
| `not_applicable` | An optional field is absent |
| `unsupported` | The fact status is outside the versioned rule |

Existing `ClinicalReviewPacket` objects can be attached when their fact set is
fully contained in a case. The materialization emits the existing counts-only
review SLA summary; packet migrations remain handled by the existing clinical
review migration API.

## Guarded workflow

A case with a field state configured for review begins in `review_required`.
The default policy sends missing, unknown, conflicting, corrected, and
unsupported fields through review. A fully resolved case begins in
`export_ready`.

Owner-controlled assignment uses `RegistryAssignmentAuthorization`, which must
match the exact owner scope, definition version, workflow-policy digest, and
queue. Assignment records authorization custody and a queue, never reviewer
identity.

The valid governed path is:

```text
review_required -> assigned -> in_review
in_review -> adjudication_required -> export_ready
in_review -> export_ready
export_ready -> exported
```

Rejected review and adjudication have distinct terminal states. Configuration
can omit the assignment gate, but cannot bypass review or adjudication once a
case requires them. Persisted event history is revalidated, so a forged direct
transition to `exported` fails.

`correct_registry_field` requires the exact prior field digest. The replacement
must use the `corrected` state and carry prior opaque fact identifiers. A valid
correction appends an immutable event, changes the case digest, clears any
assignment, and restarts the configured governance gates.

## Privacy-safe export

`build_registry_export` accepts only cases in `export_ready` or `exported` and
requires `RegistryExportAuthorization` bound to the exact definition, privacy
policy, and export policy. The resulting envelope contains:

- registry and definition version identifiers;
- the definition digest;
- exact case identifiers and case digests;
- source Journey snapshot identifiers;
- authorization identifier and digest;
- privacy and export policy digests; and
- schema and compatibility versions.

It contains no fact values, source text, vault material, credentials, reviewer
identity, or direct identifiers. `mark_registry_case_exported` records the
exact export-manifest digest on the case event history.

## Project a protected NAACCR XML file

`write_naaccr_xml` projects cases in `export_ready` using the existing export
authorization and envelope. It does not advance case state or submit records.
Supply local UTF-8 dictionary bytes to `parse_naaccr_dictionary`, an explicit
field map and a trusted local Journey fact resolver. OpenMed never retrieves the
dictionary URI and bundles no NAACCR dictionary, value table or edits program.

```python
from openmed.structured.registry import (
    NAACCRFieldMapping,
    parse_naaccr_dictionary,
    write_naaccr_xml,
)

# Caller-owned, access-controlled inputs from the existing registry workflow.
dictionary = parse_naaccr_dictionary(dictionary_bytes)
report = write_naaccr_xml(
    cases,
    version,
    authorization=export_authorization,
    export_envelope=export_envelope,
    dictionary=dictionary,
    field_map=(NAACCRFieldMapping("condition", "callerCondition", ("code",)),),
    resolver=resolve_local_fact,  # (case, field_id, opaque_fact_id) -> ClinicalFact
    output_path=protected_output_path,  # New file; never an existing file or stdout.
    patient_key_item="callerPatientKey",
    patient_key_secret=caller_hmac_key,  # At least 32 bytes; never persisted here.
)
safe_receipt = report.to_dict()
```

The item identifiers in this example are illustrative caller metadata, not
standard NAACCR items. The dictionary must declare the reserved patient-key item
at `Patient` level and every mapped item at `NaaccrData`, `Patient` or `Tumor`
level. A static tuple path selects a scalar inside the complete bound Journey
value. No coding conversion, padding, trimming or clinical code inference occurs.

The projector rebuilds existing contracts, checks the exact approved policy and
case envelope, then verifies each resolved fact's identifier, subject, type,
status, complete evidence set, value digest and derivation digest. A stale or
unready case refuses the entire batch before resolving values or creating a file.
Unknown, conflicting, missing, unsupported and unmapped fields produce controlled
losses and are omitted. Missing or substituted facts are also omitted. Distinct
values for a shared patient-level or root-level item are omitted for all sources;
the projector never chooses one. A report with losses can describe a partial file.

Declared `digits`, `alpha` and `alphanumeric` types require exact length and ASCII
characters. `numeric` accepts unsigned decimal scalars without separators or
exponents. `date` accepts valid compact year/month/day precision within 1800–2099.
`dateTime` requires XML specification 1.8 and accepts FHIR-style partial dates or
full dates with seconds and a timezone; fractional seconds are refused. `text`
uses the declared maximum length, counting characters. Empty or all-whitespace
values, illegal XML characters and type/length violations raise a fixed
`NAACCRValueError` with a controlled code and input indices before file creation.
UTF-8 dictionaries must explicitly declare specification version 1.0–1.8;
the legacy unlimited-text flag is allowed only before 1.6.

Patient keys use a caller-owned, domain-separated HMAC secret and the declared
item alphabet/width. Collisions within a batch refuse output. Keys and content
digests are pseudonymous; this is not anonymization. Protect dictionary URIs,
item identifiers and value-path names as trusted schema metadata without patient
information. The resolver is trusted local code, not sandboxed code; its standard
output and error are suppressed, but its side effects are the caller's concern.

All values are validated in memory before writing a new file with mode `0600`.
Existing files and final-component symlinks are refused. A failed write removes
only the newly created inode. The receipt contains counts, digests and indexed
losses, never XML, patient/fact identifiers, output paths or clinical values.
Identical inputs and HMAC secret produce stable XML bytes without a wall-clock
timestamp. The XML file itself contains protected clinical values and requires
the caller's privacy controls.

Operational bounds are 128 cases, 256 fields per case, 512 mappings, 4,096 resolver
calls, 8,192 dictionary items and 4 MiB each for dictionary/case/XML payloads and
the aggregate UTF-8 size of selected candidate values.
Resolved fact JSON is limited to 1 MiB; dictionary nesting is limited to 32 levels.
Patient-key widths are capped at 128 for fixed-width types and 64 for text/numeric
types. Dictionary URIs accept bounded HTTP(S) or URN schema identifiers without
credentials, query strings or fragments.

This projection follows the hierarchy and item constraints described in the
[NAACCR XML Data Exchange Standard 1.8](https://www.naaccr.org/wp-content/uploads/2024/05/Data-Exchange-Standards_1.8_20240517.pdf).
It does not establish full registry conformance, evaluate clinical correctness,
run external registry edits or grant submission authority. Run the appropriate
caller-owned registry validation and edits after projection; receipts explicitly
retain `edits_validated=False` and `submitted=False`.

## Compatibility and verification

Registry artifacts declare schema version `1.0.0` with `same_major`
compatibility. The bundled `clinical_registry.schema.json` validates definition
versions, cases, materializations, and export envelopes. Loading definitions or
cases also verifies their content digests, field coverage, event chronology,
transition graph, review-packet custody, and compatibility policy.
