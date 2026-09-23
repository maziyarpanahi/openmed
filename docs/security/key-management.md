# Key management and rotation

OpenMed keeps surrogate and audit-key custody local to the caller. It does not
send keys to a service, persist them in reports, or provide a cloud escrow. This
guide describes the operator-owned lifecycle for generating, storing, rotating,
retiring, and recovering keys without placing raw PHI or key material in logs.

## Separate keys by purpose

Use independent keys for each environment and purpose:

- audit-report HMAC signing;
- cross-document surrogate vaults;
- patient-consistent date shifting; and
- any application-specific stable content hashes.

Do not reuse a production key in development, tests, or multiple tenants. Key
identifiers are metadata and may appear in audit reports, so keep them short,
stable, and PHI-free.

## Generate and rotate audit keys

`KeyLifecycle` manages one active key and retained verification keys entirely in
memory. New keys are generated with Python's `secrets` module unless the caller
supplies key material from its own secret store.

```python
from openmed.core import KeyLifecycle

keys = KeyLifecycle.generate(prefix="audit")
first_id = keys.active_key_id
signed_report = keys.sign_audit(report)

rotation = keys.rotate()
assert rotation.key_id != first_id

# The old key is retired for signing but retained for verification.
assert keys.verify_audit(signed_report)
```

After a restart, load the caller-owned active and retained keys explicitly:

```python
import os

from openmed.core import KeyLifecycle

keys = KeyLifecycle.from_keys(
    {
        "audit-v0001": os.environ["OPENMED_AUDIT_KEY_V1"],
        "audit-v0002": os.environ["OPENMED_AUDIT_KEY_V2"],
    },
    active_key_id="audit-v0002",
    prefix="audit",
)
```

Keep a retired audit key for at least as long as its signed evidence must remain
verifiable. Removing it makes those reports cryptographically unverifiable.
`metadata()` is safe to inventory because it returns only IDs, versions, and
active/retired states; it never returns raw keys.

## Rotate a surrogate vault epoch

`SurrogateVault` derives versioned epoch IDs from its caller-supplied root
secret. Rotating an epoch re-HMACs and re-encrypts the vault entries. Because a
persisted vault contains no raw source surfaces, a non-empty vault requires the
operator to supply the source catalog during migration:

```python
from openmed.core import SurrogateSource, SurrogateVault

vault = SurrogateVault.from_file(
    "surrogate-vault.json",
    hmac_secret=os.environ["OPENMED_SURROGATE_KEY"],
)
sources = [SurrogateSource("synthetic-person", "NAME", "en")]
result = vault.rotate(sources, revoke_previous=True)
assert result.consistency is not None and result.consistency.passed
```

The source catalog is consumed in memory and is not serialized. Epoch rotation
changes the vault's derived HMAC linkage and encryption keys but does **not**
replace the caller-supplied root secret. Always retain a protected backup and
require a passing consistency report before committing the migration.

## Replace a surrogate-vault root secret

A lost root secret cannot be recovered from the vault file. If the root secret
must change, create a separate vault with the new secret and copy mappings using
the trusted in-memory source catalog. Do not overwrite the old vault in place:

```python
import os

from openmed.core import KeyLifecycle, SurrogateSource, SurrogateVault

keys = KeyLifecycle(
    os.environ["OPENMED_SURROGATE_KEY_V1"],
    prefix="surrogate",
)
old_vault = SurrogateVault.from_file("vault.json", hmac_secret=keys.active_key)
source = SurrogateSource("synthetic-person", "NAME", "en")
surrogate = old_vault.get(
    source.source_text,
    label=source.label,
    lang=source.lang,
)
if surrogate is None:
    raise RuntimeError("source catalog does not match the existing vault")

keys.rotate(os.environ["OPENMED_SURROGATE_KEY_V2"])
new_vault = SurrogateVault.from_file(
    "vault.next.json",
    hmac_secret=keys.active_key,
)
new_vault.get_or_create(
    source.source_text,
    label=source.label,
    lang=source.lang,
    create_surrogate=lambda _attempt: surrogate,
)
assert new_vault.get(
    source.source_text,
    label=source.label,
    lang=source.lang,
) == surrogate
```

Repeat the copy and equality check for every catalog entry, then atomically
switch consumers to the new vault and root key. Replacing a vault without this
mapping migration changes stable cross-document pseudonyms and can break
longitudinal joins.

## Rotation procedure

1. Inventory the active PHI-free key ID and every artifact or vault that uses
   it; never inventory raw key bytes.
2. Generate at least 32 random bytes in the deployment's approved local secret
   store and give the new version a non-PHI ID.
3. Back up encrypted vaults and retained verification keys before migration.
4. Rotate audit signing first, then verify reports made with both old and new
   key IDs.
5. For an epoch rotation, rotate each surrogate vault with its in-memory source
   catalog and require a passing consistency report. For root-secret rotation,
   build and verify a separate migrated vault as described above.
6. Deploy the new active key to every writer before retiring the old key.
7. Keep old audit keys read-only for the evidence-retention period. Revoke a
   compromised surrogate epoch only after its entries have migrated.
8. Record only dates, key IDs, owners, and verification results in the change
   record. Never record secrets or source identifiers.

Rotate on a documented cadence appropriate to the deployment and immediately
after suspected exposure, operator departure, backup compromise, or accidental
secret disclosure. Emergency rotation should use the same verification steps;
urgency is not a reason to skip consistency checks.

## Environment and file-permission checklist

- Prefer an OS keychain, mounted secret file, or local secret manager over
  command-line arguments or shell history.
- If environment variables are required, scope them to the service process,
  prevent debug dumps, and remember that child processes inherit them.
- Create secret and vault files under `umask 077`; require mode `0600` for files
  and `0700` for their parent directory.
- Keep keys, `.env` files, vault backups, and recovery bundles out of Git,
  container images, notebooks, fixtures, logs, crash reports, and support
  archives.
- Restrict backup access separately from application runtime access and test
  restoration on synthetic data.
- Never place raw PHI in a key ID, filename, exception, rotation record, or
  audit note.
- Verify both old and new audit signatures before deleting or disabling any
  retained key.

OpenMed cannot enforce storage permissions in a caller-owned secret store. The
deployment operator remains responsible for access control, backup protection,
retention, destruction, and incident response.
