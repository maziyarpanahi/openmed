# Guarded evidence packets

`openmed.clinical.evidence_packet` is a local, typed boundary for evidence
passed to downstream clinical-reasoning components. It is an assistive data
integrity layer, not a clinical decision or compliance guarantee.

## What enters a packet

An accepted `EvidenceReference` contains only:

- a caller-owned synthetic `reference_id` and optional `source_id`;
- a validated `queued → in_review → approved` transition history and verified status;
- a `sha256:<64 lowercase hex>` policy fingerprint; and
- a non-empty half-open source offset (`start`, `end`).

It never stores source text, excerpts, claims, or opaque payloads. References
must be explicitly synthetic (`synthetic: true`) and verified
(`verified: true`). Reference, source, and packet IDs require a `synthetic:` or
`fixture:` prefix (the hyphen forms also work); a caller flag alone cannot
make a patient value safe. IDs must be opaque and contain no patient values.
Their `review_state` must be `"approved"`, backed by an
ordered review history under the default transition policy. Every transition
must carry the provenance fingerprint for that exact reference, source, offset,
and evidence policy. A state string or approval event alone is insufficient.

```python
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_evidence_review,
    fingerprint_policy,
)
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)

policy_fingerprint = fingerprint_policy(
    {"policy": "synthetic-review", "version": 1}
)
provenance = fingerprint_evidence_review(
    reference_id="synthetic:ref-001",
    source_id="synthetic:document-001",
    start=8,
    end=17,
    policy_fingerprint=policy_fingerprint,
)
review = ReviewStateMachine()
for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
    review.transition(
        state,
        make_opaque_event_id(("synthetic:ref-001", state.value)),
        provenance,
    )
packet = build_evidence_packet(
    [
        {
            "reference_id": "synthetic:ref-001",
            "source_id": "synthetic:document-001",
            "start": 8,
            "end": 17,
            "review_state": "approved",
            "review_transitions": [item.to_dict() for item in review.transitions],
            "policy_fingerprint": policy_fingerprint,
            "synthetic": True,
            "verified": True,
        }
    ],
    policy_fingerprint=policy_fingerprint,
)
```

Accepted references are sorted by `(start, end, reference_id)`, so equivalent
inputs produce the same version-2 packet, JSON representation, and
`packet.digest`. Fingerprints are computed locally from canonical JSON; packet
construction makes no network call. Reopened, expired, rejected, skipped, or
source-mismatched review histories are rejected.

## Rejections and privacy

Invalid candidates are omitted. `packet.rejection_report` exposes only input,
accepted, rejected, and category counts. Stable categories include
`raw_text`, `unverified`, `not_synthetic`, `invalid_review_state`,
`invalid_policy_fingerprint`, `policy_mismatch`, `invalid_source_offset`,
`duplicate_reference`, and `invalid_reference`.

Validation exceptions expose only the category through
`EvidencePacketValidationError.category`; they do not include the rejected
record or any of its values. Keep fixtures synthetic and pass the original
source through a separately controlled review surface when a human needs to
inspect an offset.
