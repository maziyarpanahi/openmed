# Clinical Output Use Policy

`openmed.clinical.output_use_policy` is a deterministic, local-first gate for
the declared use of a clinical output. It checks metadata before an output
leaves a guarded workflow; it does not inspect, classify, or approve the
clinical content itself.

!!! warning "Assistive workflow control"
    An allowed use is not a compliance certification, diagnosis, treatment
    decision, or guarantee of clinical safety. Keep clinical review and local
    governance in the workflow.

## Declare the use

Every release declaration must identify the output category, purpose, audience,
review state, whether the use can trigger a decision, and the fingerprint of
the policy being applied. The default policy is bundled and requires no network
call or external service.

```python
from openmed.clinical.output_use_policy import (
    DEFAULT_OUTPUT_USE_POLICY,
    DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
    OutputUseDeclaration,
    evaluate_output_use,
)

declaration = OutputUseDeclaration(
    category="summary",
    purpose="documentation",
    audience="clinician",
    review_state="reviewed",
    decision_triggering=False,
    policy_fingerprint=DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
)
decision = evaluate_output_use(
    declaration,
    policy=DEFAULT_OUTPUT_USE_POLICY,
)

if decision.allowed:
    # The caller may release its already-reviewed output here.
    pass
```

The declaration contains metadata only. Do not put source text, patient
identifiers, model prompts, or output payloads in it.

## Default compatibility rules

The default policy uses exact category/purpose/audience combinations and a
minimum review state. `approved` satisfies a `reviewed` requirement, while
`draft` and `pending_review` do not. Rejected declarations always fail.

| Category | Purpose | Audience | Minimum state |
| --- | --- | --- | --- |
| `summary` | `documentation` | `clinician` | `reviewed` |
| `summary` | `research` | `researcher` | `approved` |
| `summary` | `quality_assurance` | `quality_team` | `reviewed` |
| `summary` | `care_coordination` | `clinician` | `approved` |
| `summary` | `patient_communication` | `patient` | `approved` |
| `extraction` | `documentation` | `clinician` | `reviewed` |
| `extraction` | `research` | `researcher` | `approved` |
| `extraction` | `quality_assurance` | `quality_team` | `reviewed` |
| `extraction` | `care_coordination` | `clinician` | `approved` |
| `annotation` | `review` | `clinician` | `reviewed` |
| `annotation` | `quality_assurance` | `quality_team` | `reviewed` |
| `recommendation` | `review` | `clinician` | `approved` |
| `decision_support` | `review` | `clinician` | `approved` |

Decision-triggering uses are denied regardless of the matching rule. The
`decision_triggering` field must be explicitly `False`; action categories and
the `clinical_decision` purpose are also treated as decision-triggering.

## Stable, privacy-safe results

`OutputUseDecision.to_dict()` returns only the schema version, allow/deny
verdict, policy fingerprint, and stable `reason_codes`. It does not echo the
declaration or any output content. Common denial codes include:

- `category_undeclared`, `purpose_undeclared`, and `audience_undeclared`
- `review_state_insufficient` and `review_rejected`
- `incompatible_use`
- `decision_triggering_use`
- `policy_fingerprint_undeclared` and `policy_fingerprint_mismatch`

Use `enforce_output_use()` when the caller wants a fail-closed exception. Its
`OutputUsePolicyError` message and `reason_codes` contain only stable codes, so
they are safe to route to a log or audit report without adding raw clinical
values.
