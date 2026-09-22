# Privacy-safe Agent Run Commitments

`openmed.agent.commit_run_summary()` binds an evidence bundle to an exact
[privacy-safe run summary](run-summaries.md) without copying the summary or
trusting a filename. It hashes the summary's deterministic canonical JSON
behind a fixed domain separator and returns a `sha256:<64 hex>` commitment.
`openmed.agent.verify_run_commitment()` reproduces that commitment and compares
it with constant-time equality.

## What is committed

The commitment is `SHA-256(domain || 0x00 || canonical_json)` where:

- `domain` is the fixed string `openmed.agent.run_commitment.v1`, so a digest
  can never be confused with a generic hash of the same bytes.
- `canonical_json` is `RunSummary.to_json()`: compact JSON with sorted keys and
  no insignificant whitespace.

Only validated `RunSummary` values are accepted. Raw events, prompts, tool
arguments, clinical text, paths, and credentials never reach the hash function.

## Example

```python
from openmed.agent import (
    OutcomeClass,
    RunEvent,
    RunSummary,
    WorkflowOutcome,
    commit_run_summary,
    verify_run_commitment,
    CommitmentStatus,
)

summary = RunSummary.from_events(
    [
        RunEvent(
            workflow_id="intake",
            outcome=WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
            tool_call_count=3,
            duration_seconds=1.5,
        )
    ]
)

commitment = commit_run_summary(summary)
assert commitment.startswith("sha256:")
assert verify_run_commitment(summary, commitment) is CommitmentStatus.MATCH
assert (
    verify_run_commitment(summary, "sha256:" + "0" * 64)
    is CommitmentStatus.MISMATCH
)
```

## Output contract

- Identical summaries always produce identical commitments; changing any
  validated metadata field changes the digest.
- `verify_run_commitment()` returns the closed `CommitmentStatus.MATCH` or
  `CommitmentStatus.MISMATCH` value and compares with `hmac.compare_digest`.
- A malformed commitment fails closed with `RunCommitmentError` and the stable
  `commitment: invalid_digest` message, rather than being reported as a
  mismatch.
- Every error message names only a field and a stable code; it never repeats a
  submitted value.

## Out of scope

Commitments are not digital signatures and carry no key management, ledger
storage, or non-repudiation. They hash validated summaries only, never raw
agent events or clinical content, and perform no I/O.
