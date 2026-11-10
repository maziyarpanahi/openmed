# Tamper-evident audit log

OpenMed can link de-identification audit reports into an ordered, append-only
chain. Each entry commits to the preceding entry hash, its sequence number, the
report reproducibility hash, the input and de-identified output hashes, and a
minimal span summary. The retained entry count and head hash also make a
truncated tail detectable when the chain document is verified.

The chain is designed for local compliance evidence. Creation and verification
do not make network calls.

## Create and append

Generate an `AuditReport` with `deidentify(..., audit=True)`, then append it to
a chain file:

```python
from openmed import deidentify
from openmed.core.audit_chain import append_to_chain_file

report = deidentify(
    clinical_text,
    method="mask",
    policy="hipaa_safe_harbor",
    audit=True,
)
entry = append_to_chain_file("audit-chain.json", report)
print(entry.sequence, entry.entry_hash)
```

`append_to_chain_file` loads and verifies an existing chain before adding an
entry. It refuses to append a report whose reproducibility hash no longer
matches the report contents, then atomically replaces the chain document.

For in-memory workflows, use `AuditChain.append()` and `AuditChain.write()`:

```python
from openmed.core.audit_chain import AuditChain

chain = AuditChain()
chain.append(first_report)
chain.append(second_report)
assert chain.verify().valid
chain.write("audit-chain.json")
```

## Verify from the CLI

The existing audit verifier recognizes both individual reports and chain
documents:

```console
openmed audit verify audit-chain.json
```

Use `verify-chain` to verify a chain and confirm that a particular signed report
is one of its entries:

```console
openmed audit verify-chain audit-chain.json \
  --report audit-report.json \
  --key "$OPENMED_AUDIT_KEY"
```

For signed reports, omit `--key` to read `OPENMED_AUDIT_KEY`. Verification fails
when the report signature or reproducibility hash is invalid, the report is not
committed to the chain, or the chain shows insertion, deletion, reordering, or
mutation. Failure output identifies the detected condition and affected entry
when one can be identified.

## What is stored

The chain format is deliberately narrower than an `AuditReport`. An entry
contains only:

- the sequence number and previous-entry hash;
- the report, input, and de-identified output hashes;
- span start/end offsets, canonical labels, and text hashes; and
- the entry hash.

It does not serialize source or de-identified text, surrogates, context,
detector evidence, mappings, free-form metadata, or residual-risk notes.
Unknown fields are rejected when a chain is loaded so uncommitted data cannot
be hidden alongside the verified payload.

## Security boundary

A valid chain proves that its contents still match its retained count, head,
and links. It does not identify who created the chain and is not a replacement
for access controls, backups, or the HMAC signature on an individual audit
report. Keep the chain document read-only for normal consumers, retain its head
hash with release or compliance records, and verify both the signed report and
its chain membership when authenticity matters.

Distributed consensus and external timestamp authorities are outside this
feature's scope.
