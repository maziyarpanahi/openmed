# Blinded clinician adjudication

Directly showing clinicians a model name, vendor, familiar writing style, or a
fixed left-to-right order can bias benchmark adjudication. OpenMed can render a
deterministic packet set in which candidate identities and sealed submission
manifest digests are kept outside the reviewer artifact. All rendering and
verification is local and performs no network requests.

This feature prepares human-review material. It does not score responses,
authorize a clinical action, certify a benchmark, or replace clinical judgment.

## Prepare the private inputs

Each case supplies shared source evidence and the candidate outputs keyed by
their private submission identities. Every case in one packet set must compare
the same submissions. Use opaque public references for cases, evidence, the
reviewer, and the packet set.

The renderer also requires:

- the sealed holdout commitment from the governed benchmark;
- one sealed workflow-manifest digest per candidate;
- one to eight bounded rubric questions;
- a digest of the reviewer's conflict-of-interest declaration; and
- an evaluator-held randomization key of at least 32 bytes.

Only a `cleared` conflict screen can render packets. A recused reviewer fails
closed before any packet is returned. Keep the underlying declaration in the
evaluator's controlled system; reviewer packets contain only its SHA-256 digest,
an opaque reviewer reference, and the closed status.

```python
from openmed.eval.governance import (
    ComparisonCase,
    ConflictOfInterestMetadata,
    RubricCriterion,
    SourceEvidence,
    render_blinded_adjudication_packets,
)

cases = (
    ComparisonCase(
        case_ref="case-001",
        source_evidence=(
            SourceEvidence("source-001", "Synthetic shared evidence."),
        ),
        candidate_outputs={
            "submission-red": "Synthetic candidate response one.",
            "submission-blue": "Synthetic candidate response two.",
        },
    ),
)
rubric = (
    RubricCriterion(
        criterion_ref="evidence-grounding",
        question="How fully is the response supported by the supplied evidence?",
        minimum_score=1,
        maximum_score=5,
    ),
)
conflicts = ConflictOfInterestMetadata(
    reviewer_ref="reviewer-017",
    declaration_digest="sha256:" + "1" * 64,
)

packets, sealed_mapping = render_blinded_adjudication_packets(
    packet_set_ref="adjudication-round-001",
    cases=cases,
    rubric=rubric,
    conflict_of_interest=conflicts,
    holdout_commitment_digest="sha256:" + "2" * 64,
    submission_manifest_digests={
        "submission-red": "sha256:" + "3" * 64,
        "submission-blue": "sha256:" + "4" * 64,
    },
    randomization_key=b"replace-with-32-or-more-secret-bytes",
)
```

`packets[0].to_json()` contains the shared evidence, rubric, conflict screen,
holdout commitment, and candidates named only `candidate-a`, `candidate-b`, and
so on. It does not contain submission identities or submission-manifest
digests. Supplying the same inputs and key produces byte-identical packets.

Source evidence and candidate responses are deliberately reviewer-visible and
may be sensitive in a real evaluation. Treat every public packet as a protected
clinical review artifact: apply the benchmark's access controls and retention
policy, and do not write its contents to application logs, exceptions, or
general-purpose audit reports.

## Check presentation balance

Cases receive a keyed deterministic permutation. Candidate positions use a
keyed base order followed by balanced rotations, so each candidate appears in
each presentation slot equally often, within one occurrence when the case count
is not divisible by the candidate count.

```python
from openmed.eval.governance import validate_packet_balance

balance = validate_packet_balance(sealed_mapping)
assert balance.balanced
```

The balance report contains only packet count, candidate count, maximum slot
imbalance, and pass/fail status. It never reports hidden identities, evidence,
or candidate text. This check addresses position balance; it does not establish
that response style, length, or clinical quality cannot reveal a submission.

## Store and audit the identity mapping

The returned `SealedIdentityMapping` is a separate, identity-bearing artifact.
Its HMAC commitment binds every alias to its private identity, sealed workflow
manifest, and exact output digest. That commitment is copied into every public
packet, but the assignments are not.

Store `sealed_mapping.to_private_json()` separately under access control and
retain the randomization key in an independent secret store. HMAC sealing
detects changes; it does not encrypt the private mapping. Never attach the
private JSON to reviewer packets or ordinary benchmark reports.

After adjudication, an authorized auditor can verify the mapping and packet
contents before opening the identities:

```python
from openmed.eval.governance import verify_sealed_identity_mapping

audit = verify_sealed_identity_mapping(
    packets,
    sealed_mapping,
    b"replace-with-32-or-more-secret-bytes",
)
assert audit.valid
```

Audit results contain only closed reason codes and counts. A changed key,
mapping, alias, case association, or candidate output fails verification without
echoing caller-provided values. Identity disclosure and adjudication-result
joining remain explicit, separately authorized evaluator operations.
