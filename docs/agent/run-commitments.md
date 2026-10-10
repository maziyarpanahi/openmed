# Deterministic agent run commitments

`openmed.agent.compute_run_summary_commitment()` binds an evidence record to
one exact [privacy-safe run summary](run-summaries.md) without copying the
summary or relying on a filename. Only an already validated `RunSummary` can
cross this boundary.

## Commitment construction

The commitment version is `openmed.agent.run_commitment.v1`. Its SHA-256 input
is assembled byte-for-byte as:

1. the ASCII commitment version;
2. one NUL byte as an explicit domain separator; and
3. the canonical UTF-8 bytes returned by `RunSummary.to_json()`.

The public representation is lowercase `sha256:` followed by 64 hexadecimal
characters. Identical summaries therefore produce identical commitments, while
any change to workflow identifiers, outcome counts, tool calls, duration, or
artifact digests changes the commitment.

```python
from openmed.agent import (
    RunSummary,
    RunCommitmentVerificationResult,
    compute_run_summary_commitment,
    verify_run_summary_commitment,
)

summary = RunSummary.from_events([])
commitment = compute_run_summary_commitment(summary)

result = verify_run_summary_commitment(summary, commitment)
assert result is RunCommitmentVerificationResult.VERIFIED
```

## Verification contract

Verification uses `hmac.compare_digest()` for constant-time comparison and
returns one closed, content-free category:

| Result | Meaning |
| --- | --- |
| `verified` | The supplied commitment matches the canonical summary. |
| `mismatch` | The commitment is well formed but does not match. |
| `malformed_commitment` | The supplied value is not a canonical commitment. |

Results and errors never contain the submitted commitment. Invalid summary
types fail with the stable error `summary: invalid_type` instead of attempting
to parse, normalize, or hash them.

## Security and privacy boundaries

The helper hashes only the canonical metadata already admitted by
`RunSummary`. It performs no file or network I/O, does not read artifact
contents, and does not hash raw events, prompts, clinical text, tool arguments,
paths, credentials, or exception text.

A commitment proves equality with previously committed bytes; it is not a
digital signature and does not establish authorship, timestamping, key
ownership, or ledger inclusion. Signing, key management, and storage remain
outside this contract.
