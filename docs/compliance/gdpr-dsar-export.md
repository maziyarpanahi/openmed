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
