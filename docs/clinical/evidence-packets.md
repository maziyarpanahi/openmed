# Guarded evidence packets

`openmed.clinical.evidence_packet` is a local, typed boundary for evidence
passed to downstream clinical-reasoning components. It is an assistive data
integrity layer, not a clinical decision or compliance guarantee.

## What enters a packet

An accepted `EvidenceReference` contains only:

- a caller-owned synthetic `reference_id` and optional `source_id`;
- reviewed and verified status;
- a `sha256:<64 lowercase hex>` policy fingerprint; and
- a non-empty half-open source offset (`start`, `end`).

It never stores source text, excerpts, claims, or opaque payloads. References
must be explicitly synthetic (`synthetic: true`) and verified
(`verified: true`), and their `review_state` must be `"reviewed"`.

```python
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_policy,
)

policy_fingerprint = fingerprint_policy(
    {"policy": "synthetic-review", "version": 1}
)
packet = build_evidence_packet(
    [
        {
            "reference_id": "synthetic:ref-001",
            "source_id": "synthetic:document-001",
            "start": 8,
            "end": 17,
            "review_state": "reviewed",
            "policy_fingerprint": policy_fingerprint,
            "synthetic": True,
            "verified": True,
        }
    ],
    policy_fingerprint=policy_fingerprint,
)
```

Accepted references are sorted by `(start, end, reference_id)`, so equivalent
inputs produce the same packet and JSON representation. Fingerprints are
computed locally from canonical JSON; packet construction makes no network
call.

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
