# Cohort membership explanations

`openmed.agent.workflows` can evaluate a versioned declarative cohort definition
against digest-addressed record evidence. The result explains every inclusion
and exclusion criterion as `met`, `not_met`, `unknown`, or `conflict` and keeps
the exact evidence and time-window references used for that result.

Evaluation is deterministic, local, and in memory. It does not read clinical
content, access a filesystem or network, enroll a patient, or initiate contact.

## Privacy-safe contract

Keep clinical values and patient identifiers inside the caller's trusted data
boundary. This contract accepts only:

- developer-authored definition, criterion, and time-window reference IDs;
- exact positive schema versions;
- a SHA-256 digest for the local record;
- evidence digests, such as an
  `AbstractionEvidenceChain.chain_digest`; and
- a digest that binds each locally retained time window.

Time-window references identify and bind the temporal rule without serializing
patient-specific dates. Digests are still sensitive metadata and need the same
access and retention controls as other clinical audit records.

```python
from openmed.agent.workflows import (
    CohortCriterion,
    CohortDefinition,
    CohortRecordEvidence,
    CriterionEvidence,
    CriterionKind,
    EvidenceAssertion,
    TimeWindowReference,
    explain_criterion_membership,
)

lookback = TimeWindowReference(
    reference_id="window.index_prior_year",
    window_digest="sha256:" + "a" * 64,
)
definition = CohortDefinition(
    definition_id="cohort.synthetic_registry",
    version=3,
    criteria=(
        CohortCriterion(
            criterion_id="clinical.confirmed_condition",
            kind=CriterionKind.INCLUSION,
            time_window=lookback,
        ),
        CohortCriterion(
            criterion_id="clinical.excluded_medication",
            kind=CriterionKind.EXCLUSION,
        ),
    ),
)
record = CohortRecordEvidence(
    record_digest="sha256:" + "b" * 64,
    definition_id="cohort.synthetic_registry",
    definition_version=3,
    evidence=(
        CriterionEvidence(
            criterion_id="clinical.confirmed_condition",
            assertion=EvidenceAssertion.MET,
            evidence_digest="sha256:" + "c" * 64,
            time_window=lookback,
        ),
        CriterionEvidence(
            criterion_id="clinical.excluded_medication",
            assertion=EvidenceAssertion.NOT_MET,
            evidence_digest="sha256:" + "d" * 64,
        ),
    ),
)

explanation = explain_criterion_membership(record, definition)
```

The record must bind the exact definition identifier and version. Evidence for
a temporal criterion must carry the identical time-window reference. Unknown
criteria, definition mismatches, and time-window mismatches fail closed with
value-free errors.

## State rules

For each criterion:

- no evidence or only `unknown` assertions produces `unknown`;
- only `met` assertions produces `met`;
- only `not_met` assertions produces `not_met`; and
- both decisive assertions produces `conflict`.

An inconclusive assertion does not override decisive evidence. An inclusion is
satisfied by `met`; an exclusion is satisfied by `not_met`. A failed inclusion
or met exclusion makes the record `ineligible` when every criterion is
decisive. When any criterion is `unknown` or `conflict`, the whole explanation
is `review_required`, even when another criterion is already unfavorable.

The output contains ordered criterion IDs, kinds, states, evidence digests,
evidence counts, and time-window references. It also contains definition,
record, evidence-metadata, and explanation digests for audit correlation.

## Human review and action boundary

`requires_human_review` is true whenever a criterion is unknown or conflicting.
The explanation's `authorizes_enrollment` and `authorizes_contact` properties
are always false, including for an `eligible` result. Eligibility is an
explanation of the declared evidence, not an approval token, reviewer decision,
clinical recommendation, or authorization to enroll or contact anyone.

Integrations should route review through the separately governed reviewer
handoff contract and require the separately governed single-use approval
contract before any high-impact operational action. This module deliberately
does not issue either contract or perform that action.
