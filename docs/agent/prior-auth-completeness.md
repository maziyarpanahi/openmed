# Prior-authorization evidence completeness

`openmed.agent.workflows` can score the structural evidence completeness of a
prior-authorization packet against an exact local requirement-schema version.
The scorer is deterministic, performs no file or network access, and works for
synthetic packets or metadata derived locally from a user-supplied packet.

This contract does **not** decide coverage, certify compliance, infer whether a
clinical statement is true, or generate clinical claims. It only reports
missing evidence, missing citations, and bounded reviewer actions.

## Privacy-safe inputs

Keep packet content inside the caller's trusted clinical-data boundary. Pass
only:

- developer-authored schema and requirement identifiers;
- the exact positive integer schema version;
- SHA-256 digests that bind the packet, evidence items, and citations; and
- contradiction flags produced by a separate local check.

Do not put patient, member, provider, payer, diagnosis, treatment, or other
sensitive values in identifiers. Digests can still be sensitive metadata and
need the same access and retention controls as other clinical audit records.

```python
from openmed.agent.workflows import (
    PacketEvidence,
    PriorAuthRequirement,
    PriorAuthRequirementSchema,
    PriorAuthorizationPacket,
    score_prior_authorization_packet,
)

requirements = PriorAuthRequirementSchema(
    schema_id="payer.synthetic_imaging",
    version=3,
    requirements=(
        PriorAuthRequirement("clinical.prior_treatment"),
        PriorAuthRequirement("clinical.requested_service"),
    ),
)

packet = PriorAuthorizationPacket(
    packet_digest="sha256:" + "a" * 64,
    requirement_schema_id="payer.synthetic_imaging",
    requirement_schema_version=3,
    evidence=(
        PacketEvidence(
            requirement_id="clinical.requested_service",
            evidence_digest="sha256:" + "b" * 64,
            citation_digests=("sha256:" + "c" * 64,),
        ),
    ),
)

report = score_prior_authorization_packet(packet, requirements)
```

The packet must bind the exact schema identifier and version. A mismatch fails
closed with a value-free validation error instead of silently applying a newer
or unrelated rule set.

## Score and codes

`completeness_score` is the fraction of schema requirements that have at least
their configured minimum number of evidence items and, when required, at least
one citation on every submitted item. The report also includes the numerator
and denominator so callers do not need to reconstruct the score.

Missing evidence uses a closed code set:

- `required_evidence_missing`
- `required_citation_missing`

The associated reviewer actions are also closed:

- `provide_required_evidence`
- `add_required_citation`
- `review_contradiction`
- `review_unsupported_statement`

An evidence item whose requirement identifier is absent from the versioned
schema is treated as an unsupported statement and routed to review. A supplied
contradiction flag also routes the corresponding identifier to review. Neither
condition changes the structural completeness score, and neither action is an
approval, denial, or coverage recommendation.

Reports contain requirement identifiers, counts, the score, closed codes, and
digests. They exclude evidence and citation digests from serialized output,
except for aggregate metadata digests, and never include raw packet values.
