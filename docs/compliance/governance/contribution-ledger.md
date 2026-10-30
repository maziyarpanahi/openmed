# Privacy-safe Contribution Ledger

`openmed.compliance.governance.contribution_ledger` keeps an append-only,
hash-chained history of federated participation receipts. It supports audit
and attribution without publishing institutions, patient counts, client-level
metrics, or local performance slices. It is evidence tooling, not a compliance
certification or a clinical decision mechanism.

## Receipt fields

Every receipt records exactly five fields, in this canonical order:

| Field | Value |
| --- | --- |
| `participant_handle` | `pseud:` plus an HMAC-SHA256 pseudonym under the per-federation key |
| `round_ref` | `sha256:` digest that references the federated round |
| `round_role` | `participant`, `coordinator`, `reviewer`, or `incident_responder` |
| `policy_version` | Positive integer governing policy version |
| `update_commitment` | Declared update `sha256:` digest for `participant`; `null` otherwise |

`append()` accepts no other field. Any extra keyword, such as `patient_count`,
`local_f1`, or `institution`, fails with `contribution_ledger_field_refused`
and nothing is appended. A validated `FederatedUpdateMetadata` may be passed as
`update_commitment`; its declared `update_digest` is recorded. The role
vocabulary matches the anonymous federation-charter roles.

## Pseudonyms

`participant_handle(key, label)` and `ContributionLedger.participant_handle()`
derive a handle from a local label with HMAC-SHA256 under a context-separated
derivation of the injected per-federation key (32 to 4096 bytes). The same key
and label always yield the same handle; another key yields an unlinkable one.
The label is used transiently and is never stored or echoed. `append()` accepts
only values in handle format, so a raw institution name cannot be recorded.
Keep the key in the coordinator's secret store; the ledger keeps only derived
keys and refuses to be pickled.

## Public ledger and selective disclosure

The public document holds, for each entry, a sequence number, the previous
entry digest, one salted SHA-256 commitment per field, and the entry digest.
No plaintext receipt value is published. Salts are derived deterministically
from the key, ledger identifier, sequence, and field name, so the same inputs
always produce the same bytes. Because every entry uses different salts, a
repeat participant is not linkable across public entries.

Entry digests reuse `openmed.compliance.audit_chain.AuditRecord` hashing,
starting from a genesis digest bound to the opaque `ledger_id`.

A holder keeps the private `ContributionReceipt` and calls `disclose()` with
the fields to reveal. The resulting `DisclosureProof` contains only those
fields' values and salts. `verify_disclosure()` checks the public ledger,
locates the entry, and recomputes each disclosed commitment.

```python
from openmed.compliance.governance.contribution_ledger import (
    ContributionLedger,
    verify_disclosure,
    verify_ledger,
)

key = b"synthetic-federation-key-0001-abcdefghijklmnop"
ledger = ContributionLedger("fed-synthetic-01", key)
receipt = ledger.append(
    participant_handle=ledger.participant_handle("synthetic-site-a"),
    round_ref="sha256:" + "a1" * 32,
    round_role="participant",
    policy_version=3,
    update_commitment="sha256:" + "c3" * 32,
)
checkpoint = ledger.checkpoint()
public_json = ledger.to_json()  # safe to publish

print(verify_ledger(public_json, checkpoint=checkpoint).reason.value)
# ok
proof = receipt.disclose(["round_role", "policy_version"])
result = verify_disclosure(public_json, proof.to_json())
print(result.reason.value, dict(result.disclosed))
# ok {'round_role': 'participant', 'policy_version': 3}
```

## Continuity verification

`verify_ledger()` needs only the public document. It returns a
`LedgerVerification` with one stable `LedgerReasonCode`:

| Code | Meaning |
| --- | --- |
| `ok` | Every check passed |
| `ledger_malformed` | Unsupported shape, field, schema, or digest format |
| `genesis_mismatch` | Genesis digest does not match the ledger identifier |
| `fork_detected` | Duplicate sequence, shared predecessor, or anchored prefix rewritten |
| `sequence_gap` | A sequence number is missing |
| `sequence_reorder` | Sequences are complete but out of order |
| `entry_tampered` | A recomputed entry digest differs from the stored digest |
| `link_mismatch` | An entry does not link to its predecessor |
| `truncated` | Fewer entries than a published checkpoint covers |

A dropped tail leaves an internally consistent prefix, so truncation is
detected only against a `LedgerCheckpoint` (length and head digest) that a
verifier recorded earlier. Publish checkpoints through a separate channel.

`verify_disclosure()` returns a `DisclosureReasonCode`: `ok`,
`proof_malformed`, `ledger_invalid`, `ledger_mismatch`, `entry_not_found`,
`entry_mismatch`, or `opening_invalid`. Disclosed values are returned only when
every check passes.

## Safety properties and limits

- Deterministic and local: no logging, clock, file access, or network call.
- Errors are `ContributionLedgerError` values whose message is a fixed
  `contribution_ledger_*` code; inputs are never echoed and errors are raised
  `from None`.
- Reports, reprs, and proofs carry fixed codes, digests, positions, and field
  names only. Receipt reprs show only the ledger position.
- The coordinator that holds the key can open every field. The ledger does
  not authenticate participants, sign entries, prove that an update digest
  matches real tensors, or hide the total number of receipts.
- The `ledger_id` must be an opaque federation identifier, not an institution
  name.

Run the synthetic controls with:

```sh
.venv/bin/python -m pytest tests/unit/compliance/governance/test_contribution_ledger.py -q
```
