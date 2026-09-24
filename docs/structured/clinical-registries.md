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

## Compatibility and verification

Registry artifacts declare schema version `1.0.0` with `same_major`
compatibility. The bundled `clinical_registry.schema.json` validates definition
versions, cases, materializations, and export envelopes. Loading definitions or
cases also verifies their content digests, field coverage, event chronology,
transition graph, review-packet custody, and compatibility policy.
