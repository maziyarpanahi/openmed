# Bounded approved FHIR transactions

`openmed.interop.fhir.transactions.assemble_transaction()` builds one local
FHIR R4 transaction from caller-approved create/update entries. It returns
immutable canonical UTF-8 JSON bytes, a `sha256:<hex>` Bundle digest, and the
total entry count. The caller must bind final approval to that digest and
dispatch those exact bytes. Assembly never contacts a server, verifies tokens,
stores a payload, splits transactions, or executes a write.

## Minimal write protocol

Entries implement `ApprovedWriteEntry`: four read-only properties named
`resource`, `kind`, `conditional_predicate` and `expected_version`. A dataclass
with these attributes also satisfies the protocol. Resources contain JSON
objects, arrays and scalar values; unsupported values, non-finite numbers,
non-string keys and nesting beyond 64 levels fail closed.

| Input | Transaction request |
| --- | --- |
| `create` | `POST ResourceType` |
| `create` with predicate | `POST ResourceType`, `ifNoneExist=predicate` |
| `update` without predicate | `PUT ResourceType/id`; resource `id` required |
| `update` with predicate | `PUT ResourceType?predicate` |
| `update` with expected version | Add `ifMatch=W/"version"` |

Predicates are already encoded search queries without a leading `?`. The
assembler preserves their bytes and ordering. It checks bounded query syntax
(4,096 characters, at most 64 non-empty parameter/value pairs), valid percent
encoding, and absence of decoded control characters. It does not choose search
keys, authorize identifiers, determine match cardinality, or resolve version
conflicts. Bare R4 version IDs are accepted; caller-supplied ETag/header strings
are refused. Create entries cannot carry an expected version.

The resource types and shapes must pass OpenMed's local R4 exchange structural
validator. Nested Bundle writes and caller-supplied Provenance writes are
refused. This is the declared local subset, not a claim of complete FHIR/IG or
terminology conformance. Duplicate `ResourceType/id` values and duplicate
conditional or update request targets are rejected. Planning must also rule
out semantic overlap between different predicates.

## Synthetic example

```python
from dataclasses import dataclass
from datetime import datetime, timezone

from openmed.interop.fhir.transactions import (
    TransactionApproval,
    TransactionLimits,
    TransactionReviewerRole,
    assemble_transaction,
)

@dataclass
class Entry:
    resource: dict
    kind: str = "create"
    conditional_predicate: str | None = None
    expected_version: str | None = None

entry = Entry(
    {"resourceType": "Observation", "status": "preliminary",
     "code": {"text": "synthetic measurement"}},
    conditional_predicate="identifier=urn%3Asynthetic%7Cmeasurement",
)
approval = TransactionApproval(
    action_digest="sha256:" + "a" * 64,
    receipt_digest="sha256:" + "b" * 64,
    reviewer_role=TransactionReviewerRole.CLINICAL_REVIEWER,
    evidence_references=("urn:sha256:" + "c" * 64,),
)
recorded = datetime(2026, 1, 2, tzinfo=timezone.utc)
transaction = assemble_transaction(
    [entry], approval=approval, limits=TransactionLimits(2, 50_000),
    clock=lambda: recorded,
)
assert transaction.entry_count == 2  # includes one Provenance POST
assert transaction.bundle["type"] == "transaction"
# Authenticate final human approval for transaction.bundle_digest locally.
# A separately authorized executor submits transaction.serialized unchanged.
```

The example digests are synthetic and provide no proof of approval or evidence.
The caller authenticates the existing action review and supplies its receipt
digest, action digest and evidence commitments. That upstream receipt is
recorded in Provenance; **it does not authorize the final Bundle by itself**.
The final approval must bind the returned Bundle digest, including Provenance.
Keep the final receipt outside the Bundle to avoid a circular digest dependency.
The integration tests compose these two boundaries with the existing local
approval-token verifier, including a changed-version rejection.

## Determinism and Provenance

Input entry/evidence ordering is significant; object key ordering is not. All
keys are sorted, separators are compact, and Unicode is encoded as UTF-8.
Stable UUIDv5 `fullUrl` values commit to the ordered entry, request and action
digest. Literal references to resources in the transaction are rewritten to
their full URLs in a detached snapshot. Other clinical fields are preserved;
assembly performs no silent redaction, status normalization or clinical action.

One appended Provenance entry targets every write full URL, records the fixed
OpenMed software agent, and carries only controlled review metadata. The
reviewer category is a `reviewer-role` extension with `clinical-reviewer`,
`data-steward` or `operator` as its `valueCode`. The caller maps its authenticated
policy role to this category; the category is not an identity or authority.
`entity.what.identifier` stores the approval-receipt digest, action digest and
evidence references using `https://openmed.dev/fhir/sid/` systems.

Evidence references are canonical `urn:uuid:<uuid>` handles or
`urn:sha256:<64 lowercase hex>` commitments. They are identifiers, not Bundle
resource references, and must resolve through a trusted local evidence store.
Use keyed commitments for low-entropy source data. Source text, reviewer names,
credentials, paths and arbitrary URLs cannot be supplied as review metadata.
Extra planner fields are never copied into Provenance.

The required injected clock supplies an aware datetime; recording is normalized
to UTC with microsecond precision. Save that instant for reassembly. Changing
it, any resource, predicate, version, entry order, action/receipt digest, reviewer
category or evidence reference changes the final digest. There is no implicit
wall-clock default. `transaction.bundle` returns a detached dictionary;
mutating it cannot change `serialized` or its digest. Modified payloads require
reassembly and fresh approval.

## Limits and safety boundaries

`TransactionLimits` has inclusive positive-integer `max_entries` and `max_bytes`
limits. Count includes the single Provenance entry. Byte limits cover canonical
UTF-8 serialization, including requests and all provenance metadata; individual
prepared entries are also checked against the byte bound. Exceeded limits raise
`entry_limit_exceeded` or `size_limit_exceeded` before a result is available to
preview. Transactions are never automatically split or weakened to batches.

All assembly errors use fixed codes. The result representation hides bytes;
resources and predicates may contain sensitive data and belong only in protected
caller memory and authorized local previews, never diagnostic logs or audit
artifacts. Provenance carries no resource text. Assembly does not establish
lineage, patient scope, approval validity, server capabilities, conditional
match semantics or transaction support. Those gates remain required before
dispatch. HTTP execution and recovery belong to their separate adapters.

This slice is the Python FHIR export/interop boundary. OpenMedKit currently has
no corresponding transaction write surface; it gains no EHR or cloud fallback.

Validation:

```bash
.venv/bin/python -m pytest tests/unit/interop/fhir/test_transactions.py tests/integration/interop/test_fhir_transactions.py -q
make format
make lint
make format-check
.venv/bin/python -m pytest tests/ -q
make docs-build
```
