# Machine-origin labels for proposed FHIR writes

Passive analytics exporters can emit `final` Observations and `confirmed`
Conditions or AllergyIntolerances. Those statuses do not establish clinician
verification. Before a write preview, apply the offline Python labeling policy:

```python
from openmed.interop.fhir import (
    FHIRWriteLabelPolicy,
    normalize_proposed_resource,
    validate_proposed_resource,
)

policy = FHIRWriteLabelPolicy(
    attesting_roles=frozenset({"role:org.openmed/clinical-attester"}),
)
normalized = normalize_proposed_resource(exported_resource, policy=policy)
assert not validate_proposed_resource(normalized.resource, policy=policy)
# Pass normalized.resource to the protected preview data path.
# Only finding.to_dict() is suitable for an audit artifact.
```

The declared subset is single FHIR R4 `Observation`, `Condition` and
`AllergyIntolerance` resources from the assertion-aware Python exporters.
Bundles, contained resources and other types fail closed. Validate each resource
in a write plan; do not treat validating one entry as validating an entire
transaction. This policy is structural, not complete FHIR/profile validation.
OpenMedKit has no corresponding FHIR write/export surface in this slice.

## Exact labels and provisional statuses

Every proposal requires both exact `(system, code)` pairs:

| Element | Default system | Default code |
| --- | --- | --- |
| `meta.security` | `http://terminology.hl7.org/CodeSystem/v3-ObservationValue` | `AIAST` |
| `meta.tag` | `https://openmed.ai/fhir/CodeSystem/write-origin` | `machine-generated` |

Configure `security_system`, `security_code`, `tag_system` and `tag_code` for
deployment policy. The OpenMed tag is a local origin convention. Matching just
the code, matching just the system, or splitting them across codings fails.
Validation always rejects absent labels. Normalization adds absent label
properties to unlabeled exporter output; an existing empty, malformed or
mismatched list is rejected rather than silently repaired. Additional labels
are retained when the required pair is present. Existing metadata is preserved.

| Resource | Final assertion normalized before preview | Replacement |
| --- | --- | --- |
| Observation | `final`, `amended`, `corrected` | `status = preliminary` |
| Condition | `confirmed` | `verificationStatus = provisional` |
| AllergyIntolerance | `confirmed` | `verificationStatus = unconfirmed` |

Missing statuses receive the same safe replacements; unknown or malformed
statuses fail closed. Verification concepts must contain exactly one coding
using the resource's standard HL7 verification system. Tentative states and
negative states (`refuted`, `cancelled`, `entered-in-error`) are retained;
normalization never turns a denied assertion into a positive one. Condition
`differential` and Observation `registered` remain tentative.

Only `meta` and the relevant status element change. Clinical status, concept
codes, values, subjects, references, extensions and all other clinical content
are deep-copied unchanged. Input is never mutated, including on rejection.

## Approval boundary

No role attests by default. `validate_proposed_resource` permits a supplied
final/confirmed status only when `approval_authorization` is a verified local
`ApprovalAuthorization` whose signed `reviewer_role` belongs to
`policy.attesting_roles`. Other roles cannot retain it.
Origin labels remain mandatory for every role. Normalization always applies
provisional statuses; it never promotes a machine assertion to a clinical fact.

The caller must obtain the protected authorization with the trusted single-use
token verifier's `consume_authorization(...)` for the exact approved action.
This labeling policy checks the verified role; the execution adapter must still
recheck the exclusive validity window, action binding and fresh authority at
dispatch. The v2 audit receipt alone has no role or validity window and cannot
attest. Receipt objects, dictionaries and role strings are rejected. Any
clinician finalization
requires a fresh preview and action-bound approval for that exact payload;
changing statuses after approval invalidates the prior action commitment.

Preview rendering (#2771 / PR #3442), field provenance (#2777 / PR #3449),
credentials and transport execution (#3662) remain separate. This API makes no
network calls and performs no autonomous clinical action. Integration must stop
on any validation finding or `WriteLabelError`, and recheck labels/status on the
exact payload before dispatch. Passing this gate alone never authorizes a write.

## Value-free findings

Findings expose only `resource_type`, `path` and a controlled `code`. Paths are
fixed schema elements; unknown resource types are reported as `Resource`.
Normalization reports `origin_label_added` and `provisional_status_applied`.
Rejections include `missing_origin_label`, `origin_label_mismatch`,
`invalid_origin_label`, `invalid_meta`, `invalid_status`, `missing_status`,
`attestation_required`, `invalid_authorization`, `unsupported_resource_type`,
`invalid_resource` and `nested_resources_unsupported`.

For example, a labeled confirmed Condition without verified attesting authority yields:

```json
{"resource_type":"Condition","path":"verificationStatus","code":"attestation_required"}
```

`NormalizedFHIRWrite.resource` remains sensitive clinical data; do not serialize
it or the result dataclass into logs or audit reports. The payload is hidden
from the result's representation. Exceptions contain a fixed message and
value-free findings. Synthetic unit and integration tests cover round trips,
configured labels, role controls, malformed inputs, denied assertions and
multilingual diagnostic leakage without model downloads or an EHR.
