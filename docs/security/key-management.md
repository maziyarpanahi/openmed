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

## Service credentials from mounted secret files

Every service credential setting also accepts a `_FILE` companion that names a
mounted secret file, such as a Compose secret under `/run/secrets` or a
Kubernetes Secret volume. Environment values are visible through container
inspection and process environments; secret files are not.

| Inline setting | File setting | Used by |
| --- | --- | --- |
| `OPENMED_SERVICE_AUTH_API_KEYS` | `OPENMED_SERVICE_AUTH_API_KEYS_FILE` | REST API-key hashes |
| `OPENMED_SERVICE_AUTH_JWKS` | `OPENMED_SERVICE_AUTH_JWKS_FILE` | REST JWT verification keys |
| `OPENMED_SERVICE_OTLP_HEADERS` | `OPENMED_SERVICE_OTLP_HEADERS_FILE` | OTLP exporter headers |
| `OPENMED_OPENHIM_PASSWORD` | `OPENMED_OPENHIM_PASSWORD_FILE` | OpenHIM management API |
| `OPENMED_SIDECAR_HASH_SECRET` | `OPENMED_SIDECAR_HASH_SECRET_FILE` | Desktop sidecar request-ID HMAC |
| `OPENMED_JOURNEY_POSTGRES_DSN` | `OPENMED_JOURNEY_POSTGRES_DSN_FILE` | `openmed-journey` migrate, worker, probe, golden |

The file rules are the same for every setting:

- Set either the inline variable or its `_FILE` companion. Setting both fails
  startup; an empty value counts as unset.
- The file must be a regular UTF-8 text file (symbolic links are followed, as
  Kubernetes Secret volumes require). Exactly one trailing line ending (`\n` or
  `\r\n`) is removed, so `echo` and editor newlines are harmless; an empty file
  is refused.
- Files are bounded: 1 MiB for API keys and JWKS, 64 KiB for the others.
- Under the production profile (`OPENMED_PROFILE` unset or `prod`) on POSIX
  hosts, a file that other users can access, or that its group can write, is
  refused. Group read is accepted only for root-owned files, which is how
  Kubernetes presents a Secret volume to a pod with an `fsGroup` (`0440`). Use
  mode `0400` or `0600` owned by the service user everywhere else. Other
  profiles skip the permission check.

Failures are reported with fixed codes and the setting name only, never the
path, the contents, or the operating-system error:

| Code | Meaning |
| --- | --- |
| `secret_source_conflict` | Both the inline variable and `_FILE` are set. |
| `secret_file_unreadable` | The file is missing or cannot be opened or read. |
| `secret_file_not_regular` | The path is a directory, FIFO, device, or socket. |
| `secret_file_too_large` | The file exceeds the size bound. |
| `secret_file_permissions_too_open` | Production profile and the mode is too permissive. |
| `secret_file_invalid_encoding` | The file is not UTF-8 text. |
| `secret_file_empty` | The file is empty after removing one line ending. |

The REST service fails startup with the setting and code in the error message.
`openmed-journey` prints `{"state": "not_ready", "code": "<code>"}` and exits
with status 2. The desktop sidecar reports `SIDECAR_SECRET_INVALID`. Programs
can reuse the same rules through `openmed.service.secret_files.read_secret_setting`.

`deploy/openmed-compose.yaml` and `examples/openhim-mediator/docker-compose.yml`
deliver credentials as Compose secrets, and the Helm chart mounts the Journey
database Secret and any `secretFiles` entries as read-only volumes. See
[Self-hosted Journey deployment](../deploy/self-hosted-compose.md) and
[Helm Deployment](../deploy/helm.md).

## Environment and file-permission checklist

- Prefer an OS keychain, mounted secret file, or local secret manager over
  command-line arguments or shell history. OpenMed service credentials accept
  `<NAME>_FILE` as described above.
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
