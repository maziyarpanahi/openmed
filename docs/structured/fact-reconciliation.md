# Fact reconciliation and correction history

OpenMed reconciles immutable `ClinicalFact` records through explicit,
versioned policy. The reconciler can deduplicate equivalent facts, detect
conflicts across clinical dimensions, append policy or human decisions, and
publish a value-free review queue. It never rewrites a source fact or evidence
locator.

The output is a plan rather than a claim of clinical truth. A plan is
`resolved` only when every logical fact group is unique, equivalent, or has a
deterministic policy decision. Otherwise its state is `review_required`, its
`StoreResult` is a typed conflict, and each unresolved conflict has a versioned
review packet.

## Inputs and equivalence

`FactReconciliationInput` combines one immutable `ClinicalFact` with:

- an opaque `reconciliation_id` identifying the logical record;
- a controlled source identifier used by explicit source-priority policy;
- an optional `amendment_of` fact identifier, which must also appear in the
  correction fact's `parent_fact_ids`.

Facts in the same logical record are classified as:

- `unique` when only one valid fact exists;
- `exact_duplicate` when their clinical content is byte-stable after canonical
  serialization;
- `policy_equivalent` when only configured status, unit, or time equivalence
  differs;
- conflicting when one or more protected dimensions disagree.

Exact and policy-equivalent groups retain the union of every evidence
identifier. They select a stable representative pointer, but do not discard or
modify the other facts.

## Conflict matrix

The reconciler emits one append-only `ConflictSet` for each detected dimension:

| Class | Detection boundary |
| --- | --- |
| `value` | coded/scalar value or semantic assertion, certainty, experiencer, relation, or mapping metadata differs |
| `unit` | units remain different after configured aliases |
| `time` | effective instants/intervals differ outside configured tolerance |
| `status` | statuses remain different after configured equivalence |
| `source` | distinct sources disagree on clinical content |
| `identity` | subject or encounter scope differs |
| `amendment` | a correction parent is missing, cyclic, or has competing incompatible children |

Identity conflicts can never be auto-resolved. Other classes remain unresolved
unless a versioned policy explicitly allows the class and one source has a
strictly higher priority. Equal priorities abstain. A policy may separately
prefer one valid, unambiguous amendment chain; competing corrections still
require review.

```python
from openmed.structured.facts import (
    FactReconciliationInput,
    FactReconciliationPolicy,
    FactReconciler,
)

policy = FactReconciliationPolicy(
    version="1.0.0",
    source_priority={"source.primary": 20, "source.secondary": 10},
    auto_resolve_conflicts=frozenset({"status", "source"}),
    status_equivalence={"completed": "final", "final": "final"},
    unit_equivalence={"mg / dl": "mg/dl", "mg/dl": "mg/dl"},
)
inputs = (
    FactReconciliationInput(
        fact=first_fact,
        reconciliation_id="canonical_aaaaaaaaaaaaaaaa",
        source="source.primary",
    ),
    FactReconciliationInput(
        fact=second_fact,
        reconciliation_id="canonical_aaaaaaaaaaaaaaaa",
        source="source.secondary",
    ),
)

outcome = FactReconciler(policy).reconcile(
    "subject_aaaaaaaaaaaaaaaa",
    inputs,
    occurred_at="2026-09-21T10:00:00Z",
)
plan = outcome.value
```

`outcome.ok` is false when review is required. The plan remains available on
the typed conflict result so an application can show its queue without
converting uncertainty into success.

## Append-only decisions and corrections

Each conflict receives a `ResolutionEvent`. Sufficient policy produces a
`select` event; insufficient policy produces `defer`. Resolution identities
include the policy digest, selected and rejected fact identifiers, injected
time, and prior event identity. Replaying the same inputs and policy is
deterministic. A policy change creates a new event whose
`supersedes_resolution_id` points to the prior decision.

A correction is a new fact. It references the corrected fact in
`parent_fact_ids`, and its reconciliation input declares the same identifier in
`amendment_of`. The prior fact, evidence, conflicts, and decisions remain
queryable. When a policy selects the correction, OpenMed appends a new
canonical pointer instead of editing the old pointer.

`persist_fact_reconciliation()` writes conflict sets, resolution events,
canonical pointers, and queued review packets in one Journey-store
transaction. Review packets use append-only job-metadata versions. Existing
`StorePoint` reads reproduce canonical and review state before later policies,
corrections, or reviewer actions.

## Guarded human review

`ClinicalReviewPacket` contains only opaque conflict/fact identifiers,
priority, timestamps, policy and provenance fingerprints, and transition
identifiers. It excludes reviewer identity and case content. Its state machine
permits:

- `queued` to `in_review` or `expired`;
- `in_review` to `approved`, `rejected`, or `expired`;
- terminal states to `reopened`;
- `reopened` to `in_review` or `expired`.

Every transition is immutable and chronological. Invalid skips return a typed
conflict. `persist_review_packet()` appends each packet version. Once a packet
is approved or rejected, `build_human_resolution()` creates an identity-free
human `ResolutionEvent` that supersedes the deferred policy event.

Packet schema 1.1 adds an explicit compatibility policy, extensions container,
and transition-id integrity list. `migrate_review_packet()` upgrades supported
1.0 packets without external access, rejects unsupported or lossy paths, and
returns a field-level report containing names and counts only.

## Counts-only queue operations

`summarize_review_queue()` accepts an injected timezone-aware clock and returns
only state, priority, age-bucket, expiry, and overdue counts. It never includes
packet keys, fact keys, reviewer identities, or clinical values. Critical,
high, normal, and low SLA durations are configurable and must all be present.

## Schemas and safety

Bundled JSON Schemas cover the reconciliation result, review packet,
transition, and queue summary. Reconciliation version `1.0.0` and review
version `1.1.0` both declare `same_major` compatibility.

All fixtures are synthetic. Serialized plans contain identifiers, evidence
pointers, policy metadata, counts, and hashes, not raw source text. Restricted
terminologies and datasets remain caller supplied. Reconciliation is assistive
and must not autonomously trigger diagnosis, treatment, ordering, enrollment,
outreach, or another patient-care action.
