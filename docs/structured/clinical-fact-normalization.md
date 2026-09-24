# Clinical fact normalization

OpenMed normalizes heterogeneous extraction, assertion, temporality, relation,
medication, laboratory, and observation outputs into one evidence-bound
`ClinicalFact` contract. The normalization layer is model-independent: source
components contribute explicit fragments, and no missing value is guessed.

The public API lives in `openmed.structured.facts`. The initial normalization
schema version is `1.0.0` with a `same_major` compatibility policy.

## Data flow

The normalization boundary has four records:

1. `ComponentOutputEnvelope` retains the exact component output with its
   component version, output-schema version, and canonical digest.
2. `FactFragment` maps only explicitly supplied component fields into the
   common fact vocabulary and carries evidence locator IDs.
3. `FactNormalizationRequest` groups fragments for one subject, encounter,
   profile, and optional parent facts.
4. `NormalizedClinicalFact` returns the Journey fact, source-output custody
   metadata, explicit field states, and the terminal normalization state.

`ClinicalFactNormalizer.normalize()` verifies evidence, reconciles fields,
validates the profile, computes deterministic identity and derivation hashes,
and returns a typed `StoreResult`.

## Common fields

Fragments may contribute only these fields:

- `value` and `unit`;
- `status`;
- `assertion`;
- `certainty`;
- `experiencer`;
- `effective_time`;
- `relation_participants`;
- `confidence`; and
- profile-specific `attributes`.

The normalizer never supplies a clinical value that was absent from component
output. Missing required fields become `unknown`; explicit `partial`,
`unsupported`, and `conflict` states are preserved. A conflict between two
component values stops normalization before a fact is created.

## Fact profiles

`FACT_PROFILE_SPECS` defines the current profiles:

| Profile | Required semantic fields | Example statuses |
| --- | --- | --- |
| `condition` | value, status, assertion, certainty, experiencer | active, inactive, resolved, history |
| `medication` | value, status, assertion, experiencer | active, completed, held, stopped, planned |
| `laboratory` | value, unit, status, effective time | preliminary, final, amended, corrected |
| `procedure` | value, status, assertion, experiencer, effective time | planned, completed, cancelled |
| `observation` | value, status, assertion, experiencer, effective time | preliminary, final, amended, corrected |
| `social_determinant` | value, status, assertion, certainty, experiencer, effective time | active, inactive, historical |

Unknown is valid as an explicit status, but a required unknown field produces
an `unknown` result rather than a successful one. An unrecognized profile or
status is `unsupported`; it is never coerced to the closest label.

## Assertion, certainty, and experiencer

The common axes are intentionally small:

- assertion: `affirmed`, `negated`, `uncertain`, `conditional`, `unknown`;
- certainty: `certain`, `probable`, `possible`, `uncertain`, `unknown`;
- experiencer: `patient`, `family`, `other`, `unknown`.

The selected values are copied into the fact's attributes. Negation and family
history therefore remain first-class semantics instead of changing or deleting
the extracted value.

## Effective time

Effective time uses an interval object with `start`, `end`, and `precision`.
Supported precision is `year`, `month`, `day`, `second`, or `unknown`.

Partial dates remain partial:

```json
{"precision": "month", "start": "2026-03"}
```

The normalizer does not invent the first day of a month or the end of a year.
It rejects impossible dates, timezone-free second precision, precision/date
mismatches, and reversed intervals.

## Relations and amendments

Relation participants contain a role, target ID, and target type (`fact` or
`evidence`). A target must be present in the current evidence set or the
request's parent facts. This makes cross-sentence and cross-component relations
auditable without accepting an unbound identifier.

Corrections and amendments use `parent_fact_ids`. Parent edges are included in
the deterministic fact identity and derivation hash. A correction creates a
new fact; it never overwrites the prior record.

## Evidence and derivation

Every normalized fact must reference at least one supplied `EvidenceLocator`.
The normalizer checks that each requested ID exists and that the mapping key
matches the locator's own ID. Missing or mismatched evidence is a `conflict` and
no fact is produced.

Derivation includes:

- the normalizer name and version;
- all evidence IDs;
- component-output digests;
- the common profile;
- explicit field states;
- a digest of normalized fields; and
- parent fact IDs.

The same fragments in a different order produce the same fact and derivation.
A changed component output, normalized value, policy state, or parent edge
creates a new identity.

## Heterogeneous mapping adapter

`MappingFactAdapter` converts mapping-shaped component results through an
explicit dotted-path map:

```python
from openmed.structured.facts import MappingFactAdapter

adapter = MappingFactAdapter(
    component="clinical.lab.extractor",
    component_version="1.0.0",
    output_schema="lab.component",
    output_schema_version="1.0.0",
    kind="value",
    field_paths={
        "value": "result.measurement",
        "unit": "result.unit",
        "status": "result.status",
    },
)

fragment_result = adapter.adapt(
    component_output,
    evidence_ids=("evidence_...",),
)
```

Only configured paths are read. A similarly named field elsewhere in the
component output is ignored rather than guessed. Components with different
shapes can use different adapters while producing the same fragment contract.

## Component-output custody

Exact component output is retained inside `ComponentOutputEnvelope` and is
available only through `protected_component_outputs()`. The default
`to_dict()` representation emits component name, versions, schema, and digest,
but not the original output.

Store the protected representation only in the identified projection under an
appropriate access policy. Logs, metrics, traces, ordinary audit events, and
exceptions should use the safe representation. The normalizer's own errors use
controlled codes and never echo component values.

## Typed terminal states

| State | Example code | Meaning |
| --- | --- | --- |
| `success` | none | All required fields are known and valid. |
| `partial` | `fact_partial` | A component explicitly reported incomplete information. |
| `unknown` | `fact_unknown` | A required field is absent or explicitly unknown. |
| `conflict` | `component_field_conflict` | Components disagree on one canonical field. |
| `conflict` | `evidence_not_found` | A required locator is unavailable. |
| `unsupported` | `fact_status_unsupported` | The profile does not define the supplied status. |
| `unsupported` | `effective_time_unsupported` | Temporal input cannot be represented exactly. |
| `failure` | `fact_contract_invalid` | Construction failed after validated reconciliation. |

Partial, unknown, and unsupported results may carry a fact whose uncertainty is
explicit. Conflict and validation failures do not. Callers must check
`StoreResult.state`; the presence of a value is not equivalent to success.

## Privacy and clinical safety

- Committed fixtures are synthetic.
- Component values are never included in controlled error messages.
- Safe normalization metadata contains digests and versions, not raw component
  output.
- Unsupported values remain unmapped or unknown rather than being guessed.
- A normalized fact is evidence-bound data, not a diagnosis, treatment
  recommendation, or authorization for patient-care action.
