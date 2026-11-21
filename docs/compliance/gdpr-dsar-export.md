# GDPR DSAR Subject-Access Export

Under GDPR Article 15 a controller must produce all personal data held about a
data subject. When OpenMed maintains a cross-document surrogate vault, the data
held about a subject is the set of pseudonymous vault entries whose source
surface hashes to one of that subject's known identifier values.

`openmed.compliance.dsar` assembles that Article 15 package locally, logs the
export to a tamper-evident audit chain, and offers a non-destructive Article 17
erasure companion.

OpenMed does not self-certify compliance. This helper does not verify the
requester's identity and does not discover data outside OpenMed. Validate
against your own data, jurisdiction, and counsel before releasing any export.

## How subject matching works

The surrogate vault is keyed by `(canonical_label, lang, text_hash)`, where
`text_hash` is an HMAC of a single source surface. The vault has no subject
grouping, so the helper takes the identifier surfaces the controller already
holds for the subject (name, MRN, date of birth, ...), recomputes each
`text_hash` through the existing vault configuration, and selects only the vault
entries whose hash matches. The package therefore contains only data tied to the
requested identifiers.

Because the vault stores HMAC hashes and surrogates rather than raw values, the
package and the audit log carry no raw PHI.

## Assembling an access package

```python
from openmed.compliance import (
    HashChainAuditLog,
    SubjectIdentifier,
    assemble_dsar_package,
    render_dsar_summary,
)
from openmed.core.surrogate_vault import SurrogateVault

vault = SurrogateVault.from_file("vault.json", hmac_secret=SECRET)
audit = HashChainAuditLog()

identifiers = [
    SubjectIdentifier("John Smith", "first_name"),
    SubjectIdentifier("MRN-12345", "id_num"),
]

package = assemble_dsar_package(
    identifiers,
    vault,
    audit_sink=audit,
    audit_references=["deid-run-2026-07-01"],
)

print(render_dsar_summary(package))
assert audit.verify()
```

`assemble_dsar_package` returns a `DsarPackage`:

| Field | Meaning |
|---|---|
| `subject_ref` | Stable, non-PHI reference derived from the identifier hashes. |
| `entries` | The matching `DsarEntry` holdings (canonical label, lang, `text_hash`, surrogate). |
| `categories` | The distinct canonical labels held for the subject. |
| `audit_references` | Caller-supplied references to related audit artifacts. |
| `audit_record` | The `AuditRecord` appended for this export, when an `audit_sink` is passed. |

## Audit logging

Export generation is recorded through an injected `AuditSink`. The default
`HashChainAuditLog` is an append-only ledger where each record commits to its
predecessor's hash, so tampering is detectable via `verify()`. The recorded
payload holds the subject reference, counts, categories, and content hashes
only, never raw PHI.

The `AuditSink` protocol is deliberately small so the shared OpenMed audit chain
can replace the default implementation without changing callers.

## Article 17 erasure companion

`plan_erasure` lists what an erasure would remove and is non-destructive by
default: it returns the erasable holdings and leaves the vault untouched.

```python
from openmed.compliance import plan_erasure

plan = plan_erasure(package, vault, audit_sink=audit)
assert plan.executed is False
for entry in plan.erasable:
    print(entry.canonical_label, entry.text_hash)
```

The preview is logged as a non-destructive `dsar.erasure_preview` event. Actual
deletion is a separate, deliberate operation the controller performs on the
vault store under its own key-custody and governance controls.

## Scope

- Identity verification of the requester is out of scope.
- Automated cross-system data discovery outside OpenMed is out of scope.

## Coordinating local withdrawal

An application can register its local store adapters with
`openmed.compliance.withdrawal.WithdrawalCoordinator`. Supply a keyed subject
reference digest. Calling `withdraw(subject_ref_digest)` previews the counts
and computes a confirmation token. After reviewing the preview, pass the same
digest and `confirmation=plan.confirmation_token` to apply and verify the
selected records. A change in membership or revision requires a fresh plan.

Adapters bind their selection digest to the affected records and check it again
under the store's mutation lock before erasure. Receipts identify the stores
by code and report counts, including unattributable holdings. A partial receipt
identifies which stores still need verification. Training and private-round
withdrawal use their own workflows.

Register `VaultWithdrawalParticipant(vault, identifiers)` from
`openmed.compliance.dsar` under `ParticipantCode.SURROGATE_VAULT`. Use the
DSAR package's `subject_ref` for preview and confirmation. Selection matches
the full label, language and keyed identifier used by the export. The adapter
persists a file-backed vault even when automatic saves are disabled.

For an associated AES-SIV reversible mapping, pass
`encrypted_mapping=(path, crypto)` with the controller's `SurrogateVaultCrypto`
instance. Only exact surrogate/source pairs belonging to selected vault
holdings are removed. Mirrored copies count once with their vault holdings.
Keep the file-backed vault under a single writer during withdrawal, as for
other mutations of a live vault object.

For live ambient capture, register an `AmbientWithdrawalParticipant` under
`ParticipantCode.AMBIENT_SESSIONS`. Register a separate instance under
`VOICE_EMBEDDINGS` for embedding owners. Call `participant.register(subject_digest, owner)`
when creating each session. A confirmed withdrawal closes every matching owner,
releases the registry's references and prevents new registrations for that
withdrawn digest. `participant.receipts(subject_digest)` exposes controlled
states and counts. Declared external copies keep the coordinator receipt partial.
