# Trial-eligibility disagreement review

`openmed.agent.workflows` can compare deterministic eligibility-rule results
with evidence-backed model assessments at every trial criterion. The output is
a structured, value-free packet that classifies why human review is needed and
preserves local citations and uncertainty.

Comparison is deterministic, local, and in memory. It does not read source
content, access a filesystem or network, enroll a candidate, or initiate
clinical outreach.

## Privacy-safe inputs

First evaluate the exact eligibility definition with
`explain_cohort_membership`. Then provide one `ModelCriterionAssessment` per
criterion assessed by the model. Each assessment contains only:

- a developer-authored criterion identifier;
- a closed `met`, `not_met`, `unknown`, or `conflict` state;
- one or more digest-addressed source spans;
- a bounded uncertainty score from `0.0` to `1.0`; and
- a digest of the exact local model artifact.

Clinical text, normalized clinical values, patient identifiers, free-text
model explanations, and outreach details are not accepted. Digests and offsets
are still sensitive metadata and need the same access controls and retention
limits as other clinical audit records.

```python
from openmed.agent.workflows import (
    CriterionState,
    EligibilityCitation,
    ModelCriterionAssessment,
    build_trial_eligibility_review_packet,
)

assessment = ModelCriterionAssessment(
    criterion_id="trial.confirmed_condition",
    state=CriterionState.NOT_MET,
    citations=(
        EligibilityCitation(
            source_digest="sha256:" + "a" * 64,
            start_offset=120,
            end_offset=148,
            evidence_digest="sha256:" + "b" * 64,
        ),
    ),
    uncertainty=0.25,
    model_digest="sha256:" + "c" * 64,
)

packet = build_trial_eligibility_review_packet(
    rule_explanation,
    (assessment,),
)
```

`rule_explanation` is the value-free result returned by
`explain_cohort_membership`. Assessments may arrive in any order; packets and
citations are normalized before hashing and serialization.

## Disagreement classification

Every declared rule criterion appears in the packet. A criterion requests
review when one or more closed causes apply:

- `rule_evidence_missing` for an `unknown` rule result;
- `rule_evidence_conflict` for conflicting deterministic evidence;
- `model_assessment_missing` when no model result was supplied;
- `model_assessment_unknown` or `model_assessment_conflict` for an
  indeterminate model result; or
- `outcome_conflict` when decisive rule and model results contradict.

Matching decisive results have no disagreement cause. The packet still retains
their rule evidence digests, model citation spans, assessment digest, and
uncertainty so the comparison is auditable. Unknown model criteria, duplicate
assessments, malformed digests, missing citations, invalid spans, and invalid
uncertainty fail closed with value-free errors.

## Human-review and action boundary

`requires_human_review` is true when any criterion has a classified cause.
`authorizes_enrollment` and `authorizes_contact` are always false, even when
every rule and model result agrees. The packet is evidence for a reviewer; it
is not a reviewer decision, eligibility approval, enrollment instruction, or
outreach authorization.

Generic reviewer-handoff validation, presentation rendering, notifications,
and approval-token issuance remain separate governance layers. Until those
contracts are integrated, callers should retain this packet locally and must
not treat its digest or review status as authority for a high-impact action.
