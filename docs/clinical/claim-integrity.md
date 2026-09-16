# Clinical claim-packet integrity

`openmed.clinical.claim_integrity` provides a deterministic, local-only
integrity check for a guarded packet containing claims, citations, review
metadata, and policy records. It is a review aid and tamper detector; it is
not a clinical decision, compliance certification, or guarantee that a claim
is correct.

## Digest contract

`compute_claim_packet_digest()` binds a versioned canonical representation of
the four packet sections with a SHA-256 digest. Mapping keys and record order
are normalized, while the order of a claim's reference list is preserved. A
mapping containing the section names is accepted, as are positional
`claims, citations, reviews, policy` arguments.

```python
from openmed.clinical.claim_integrity import (
    check_claim_packet_integrity,
    compute_claim_packet_digest,
)

packet = {
    "claims": [
        {
            "claim_id": "claim-a",
            "citation_ids": ["citation-a"],
            "policy_id": "policy-a",
        }
    ],
    "citations": [{"citation_id": "citation-a", "source_kind": "synthetic"}],
    "reviews": [{"review_id": "review-a", "claim_ids": ["claim-a"]}],
    "policy": {"policy_id": "policy-a", "version": "v1"},
}

expected_digest = compute_claim_packet_digest(packet)
report = check_claim_packet_integrity(
    packet,
    expected_digest=expected_digest,
)
assert report.passed is True
```

Records may contain caller-controlled values, including source text, when
they are hashed locally. The digest function returns only the `sha256:` token.
`canonicalize_claim_packet()`, `ClaimPacketIntegrityReport.to_json()`, and
`to_markdown()` likewise expose only section counts, fixed reason codes, and
digests. They do not serialize claim text, citation values, reviewer
identities, policy contents, or opaque record identifiers.

## Integrity findings

Pass an `expected_packet` to classify changes rather than receiving only a
digest mismatch:

```python
candidate = dict(packet)
candidate["claims"] = [
    {
        "claim_id": "claim-a",
        "citation_ids": ["citation-a"],
        "policy_id": "policy-a",
    }
]

report = check_claim_packet_integrity(
    candidate,
    expected_packet=packet,
)
```

The aggregate report uses fixed reason codes for unresolved or malformed
references, duplicate records or references, missing or unexpected records,
reference reordering, record/reference mutation, and final digest mismatch.
Missing target records and duplicate identifiers fail closed. Reordering a
reference list changes the digest and is reported as `reordered_reference`
when the same reference multiset is present in the baseline.

The implementation performs no mandatory network call, reads no external
state, and emits no logs. Callers should keep committed fixtures synthetic
and should treat the report as provenance metadata rather than a clinical
decision or a substitute for qualified review.
