# Managed offline model-bundle import

Contract version 1 (OpenMed v4.0). OpenMedKit for Swift and OpenMedKit for
Android can receive a [signed portable bundle](../models/portable-bundle.md)
through a managed, local platform handoff, verify it fully offline against
trust roots the administrator provisions, and activate or roll it back
atomically in the [bundle generation cache](../models/bundle-cache.md). No
vendor-specific management server is needed: any mobile-device-management
(MDM) or enterprise-mobility-management (EMM) product that can set managed app
configuration and deliver a file to the device works.

| Kit | Importer | Managed configuration adapter | Local handoffs |
|---|---|---|---|
| Swift | `OpenMedManagedBundleImporter` | `OpenMedManagedAppConfiguration` (`UserDefaults` key `com.apple.configuration.managed`) | `file:` URL (archive or directory), bookmark data, `InputStream` |
| Android | `com.openmed.openmedkit.managed.ManagedBundleImporter` | `RestrictionsManagerConfigurationSource` (`RestrictionsManager` app restrictions) | `File` (archive or directory), `InputStream`, `content://` URI through `ContentResolverStreamOpener` |

The importer never opens a network connection, never follows a URL, never
executes bundled content, and never extracts an archive by its own entry
names. Bytes are copied only for the paths the signed manifest declares, into
isolated cache staging, and only a fully verified generation is committed and
activated.

## Platform handoffs

The managed configuration carries the trust roots and the import policy. The
bundle itself arrives as a local file or stream through one of these
handoffs, which the app approves:

- **Apple.** The MDM installs the managed app configuration dictionary, which
  the app reads from `UserDefaults.standard` under
  `com.apple.configuration.managed`. The bundle file reaches the app through a
  managed document flow: a managed "Open In" share, a document picker
  (security-scoped URL or bookmark data), an App Group container a managed
  companion app writes, or a file the app's own enterprise channel saved
  locally. The importer starts and stops security-scoped access around the
  import.
- **Android.** The EMM or device policy controller sets app restrictions, read
  through `RestrictionsManager.getApplicationRestrictions()`. The library ships
  `@xml/openmed_managed_bundle_restrictions`; reference it from the app
  manifest, or copy its entries into the app's own restrictions file:

  ```xml
  <meta-data
      android:name="android.content.APP_RESTRICTIONS"
      android:resource="@xml/openmed_managed_bundle_restrictions" />
  ```

  The bundle arrives as a `content://` URI from an approved authority (for
  example the EMM agent's `FileProvider`), a file in an approved app directory,
  or a stream the app opened itself.

A configured location is resolved only against what the app approves:

- a relative path resolves against the app's approved directories in order;
- a `file:` URL must lie inside an approved directory;
- a `content://` URI (Android) must use an approved authority;
- any other scheme, including `http:` and `https:`, is `unsupported` with
  `handoff_unsupported`.

A location that does not exist yet is `deferred` with `handoff_pending`, so the
app can retry when the file arrives after the configuration. Hosts can listen
for `UserDefaults.didChangeNotification` (Apple) or
`Intent.ACTION_APPLICATION_RESTRICTIONS_CHANGED` (Android) and import again.

## Managed configuration keys

Apple and Android use the same keys. Keys without the `openmed_bundle_`
prefix belong to the app and are ignored. An unknown key with that prefix
fails closed with `config_invalid`, so a misspelled key never silently
changes behaviour.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `openmed_bundle_config_version` | integer | yes | Configuration version; `1`. Another value is `unsupported` (`config_version_unsupported`). |
| `openmed_bundle_trust_policy` | string | yes | Trust-policy JSON (`openmed.portable_bundle.trust_policy` v1) with the Ed25519 public keys, validity windows, threshold, and revoked content ids. At most 64 KiB. |
| `openmed_bundle_key_pins` | string | no | `sha256:` fingerprints of raw 32-byte public keys, separated by spaces or commas. When set, every key in the trust policy must be pinned. |
| `openmed_bundle_location` | string | no | Where the delivered bundle is found (see above). At most 1,024 bytes. |
| `openmed_bundle_expected_id` | string | no | Bundle id the delivered bundle must carry. |
| `openmed_bundle_expected_content_id` | string | no | `sha256:` content identity of the exact manifest to accept. |
| `openmed_bundle_min_sequence` | integer | no | Anti-rollback floor for the bundle sequence. |
| `openmed_bundle_activation` | string | no | `activate` (default), `probation`, or `stage`. |

The configuration is checked in this order: version, unknown keys, trust
policy (`trust_policy_missing`, `trust_policy_invalid`), then each optional key
(`config_invalid`). Empty optional strings count as absent. Values hold public
keys, digests, and local locations only. Never put a private key, password,
or token in managed configuration.

## Trust-anchor provisioning

Trust roots always come from the administrator; no bundle carries its own.

1. Generate an Ed25519 signing key offline and keep the private key in the
   organization's signing system. Only the public key leaves it.
2. Write a trust policy with that public key as an RFC 8037 JWK, a validity
   window, and `status: active`. See
   [Signed portable model bundles](../models/portable-bundle.md#trust-policy-rotation-and-revocation).
3. Optionally compute the pin of each public key, the SHA-256 of the raw 32
   key bytes:

   ```bash
   openssl pkey -pubin -in signing-key.pub.pem -outform DER | tail -c 32 \
     | shasum -a 256 | awk '{print "sha256:" $1}'
   ```

   Apps can also compile their own pins in (`keyPins` on the request). Each
   non-empty pin set, from the configuration and from the app, must cover
   every trust-policy key, otherwise the import fails with
   `trust_anchor_unpinned`.
4. Set `openmed_bundle_trust_policy` (and the pins) in the MDM or EMM console
   for the app.

**Rotation.** Add the new key with a validity window that overlaps the old
one, add its pin, and sign bundles with both keys during the overlap. Once the
old key's `not_after` passes, bundles signed only by it fail with
`key_expired`; bundles also signed by the new key keep verifying. Remove the
old key and pin after the overlap.

**Revocation.** Set a key's `status` to `revoked` (`key_revoked`), or add a
bundle's content identity to `revoked_content_ids` (`bundle_revoked`), and push
the updated configuration. Revocation applies to every later import; the
generation already active stays active until a newer bundle or a rollback
replaces it.

## Packaging a bundle

Deliver a bundle as its directory or as a ZIP archive of it. Archive rules:

- the manifest and envelope sit at the archive root or under exactly one
  top-level directory;
- entries are stored or DEFLATE-compressed; encryption, ZIP64 (archives of 4
  GiB or more), multi-disk archives, and other methods are `unsupported`
  (`archive_format_unsupported`); deliver larger bundles as a directory;
- names follow the portable path rule (no `..`, absolute paths, backslashes,
  or hidden files); symbolic links and special files fail with
  `entry_unsafe`, and duplicate or case-folded duplicate names with
  `entry_duplicate`;
- the archive holds exactly the declared files plus the manifest and envelope.

For example:

```bash
cd my-bundle && zip -r -X -D ../my-bundle.zip .
```

The `-X` flag drops extra attributes and `-D` omits directory entries. Avoid
Finder's "Compress", which adds hidden `__MACOSX/._*` entries. CRC-32 values
are not trusted; integrity comes from the SHA-256 digests in the signed
manifest.

## Import pipeline

Each import runs these steps in order, under a per-state-directory lock, and
reports the first failure:

1. Parse the managed configuration (see above).
2. Check key pins against the delivered trust policy.
3. Resolve the handoff. A stream or `content://` URI is spooled into the
   importer's private `spool/` directory, bounded by the archive budget.
4. Check the archive: size (`budget_archive_bytes`), the end-of-central-
   directory record, the entry count (`budget_file_count`), every central
   directory entry (flags, method, ZIP64 markers, name, type, duplicates),
   every local header and overlap (`archive_malformed`), the uncompressed
   total (`budget_total_bytes`), and each DEFLATE entry's ratio
   (`budget_compression_ratio`).
5. Read the manifest and envelope within the context's size limits and
   verify signatures, schema, structure, validity, revocation, the sequence
   floor, and device compatibility with the portable-bundle verifier
   (`bundle_rejected` plus the verifier reason). The floor is the highest of
   the caller's floor, `openmed_bundle_min_sequence`, and the importer's own
   ledger.
6. Check `openmed_bundle_expected_id` and
   `openmed_bundle_expected_content_id` (`bundle_not_expected`).
7. Compare the inventory with the manifest: every declared file present with
   its declared size, then no undeclared file (`verification_failed` with
   `file_missing`, `file_size_mismatch`, or `file_undeclared`).
8. Acquire through the cache, which stages declared files chunk by chunk,
   resumes only verified chunks, checks storage and quota
   (`storage_pressure`, `quota_exceeded`), verifies every file's SHA-256
   (`verification_failed`), commits with one rename, and activates by one
   pointer replacement. A corrupt compressed stream is reported as
   `archive_malformed`.
9. Apply the activation policy and, if given, the activation check.

Nothing reaches the activation pointer before step 8 has verified every byte,
so every refusal above leaves the active and last-green generations exactly as
they were. Concurrent readers holding a lease keep reading their complete
generation; a new reader sees either the previous or the new complete
generation.

## Activation, rollback, and floors

- `activate`: the new generation becomes active and last green. With an
  activation check (for example loading the model through the lease), it is
  first activated on probation and confirmed only when the check passes.
- `probation`: the new generation becomes active, and the previous last-green
  generation stays protected until the host calls `confirm(bundleID:)`
  (Swift) or `confirm(bundleId)` (Android).
- `stage`: the generation is committed but the active generation does not
  change (`imported`).

When the activation check fails, the importer quarantines the new generation
(reason `rolled_back`) and falls the pointer back to the last-green
generation: the outcome is `rolled_back`, or `rejected` with
`activation_check_failed` when no earlier generation exists. A later delivery
of the same bundle restages it from scratch.

After an import or confirmation leaves a generation as last green, its
sequence becomes the importer's floor for that bundle, persisted in the state
file. Older bundles then fail with `rollback_rejected`, also after a restart.
Rolling back on purpose uses the cache's `rollback`, never an older import.

Re-delivering a bundle is idempotent: the same signed manifest maps to the same
generation, so a duplicate returns `already_present` without copying bytes,
and concurrent duplicate deliveries converge on one generation.

## Cancellation, process death, and recovery

Cancellation (`cancelled`), a failing stream (`transfer_interrupted`), and
low storage (`storage_pressure`) keep the verified staged chunks resumable.
The source identity of a managed import is derived from the manifest content
identity, so the next delivery of the same bundle resumes, whether it arrives
as a file, a stream, or a content URI.

Call `recover()` at startup. It runs the cache's crash recovery and deletes
spooled streams an interrupted import left behind, then reports how many it
purged. It is idempotent. A process death at any point leaves the previous
active and last-green generations usable.

For offline recovery from a damaged generation, open it with full
verification (`open(verification: .full)` or `BundleOpenVerification.FULL`):
a mismatch quarantines it and falls back to last green. Re-delivering the same
signed bundle restores it without any network access.

## Receipts

Every import returns a bounded, value-free receipt
(`openmed.managed_import.receipt`, version 1) and records it in the state file,
which keeps the last 16. A receipt holds opaque identities, a sequence number,
counts, a digest, and reason codes only; never a path, URL, configuration
value, key, or file content. The host can persist it, show it, or report it as
managed app feedback.

| Field | Meaning |
|---|---|
| `outcome` | `imported`, `already_present`, `activated`, `deferred`, `rolled_back`, `unsupported`, or `rejected` |
| `reason` | Managed-import reason code, or `null` |
| `bundle_reason` | Portable-bundle verifier reason for `bundle_rejected` and `verification_failed` |
| `handoff` | `file`, `directory`, `stream`, `bookmark` (Swift), or `content_uri` (Android) |
| `bundle_key`, `generation`, `sequence` | Opaque identities and the signed sequence, present once the manifest authenticated |
| `previous_active`, `active`, `last_green` | Pointer before and after the import |
| `files_verified`, `bytes_staged`, `resumed`, `evicted` | Cache counts |
| `trust_digest` | Binding digest of the trust policy used |

### Reason codes

| Reason | Outcome |
|---|---|
| `managed_config_absent` | `deferred` |
| `handoff_pending` | `deferred` |
| `config_version_unsupported` | `unsupported` |
| `handoff_unsupported` | `unsupported` |
| `archive_format_unsupported` | `unsupported` |
| `config_invalid` | `rejected` |
| `trust_policy_missing` | `rejected` |
| `trust_policy_invalid` | `rejected` |
| `trust_anchor_unpinned` | `rejected` |
| `location_unapproved` | `rejected` |
| `handoff_unreadable` | `rejected` |
| `budget_archive_bytes` | `rejected` |
| `budget_file_count` | `rejected` |
| `budget_total_bytes` | `rejected` |
| `budget_compression_ratio` | `rejected` |
| `archive_malformed` | `rejected` |
| `entry_unsafe` | `rejected` |
| `entry_duplicate` | `rejected` |
| `bundle_rejected` | `rejected` |
| `bundle_not_expected` | `rejected` |
| `verification_failed` | `rejected` |
| `storage_pressure` | `rejected` |
| `quota_exceeded` | `rejected` |
| `transfer_interrupted` | `rejected` |
| `cancelled` | `rejected` |
| `lock_timeout` | `rejected` |
| `storage_io` | `rejected` |
| `activation_check_failed` | `rolled_back`, or `rejected` without an earlier generation |

Default budgets: 4,294,967,295 archive bytes, 4,096 entries, 4,294,967,295
uncompressed bytes, and a DEFLATE ratio of 200.

## Using the importer

Swift:

```swift
let cache = try OpenMedBundleCache(rootDirectory: cacheDirectory)
let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: importStateDirectory)
_ = importer.recover()
let receipt = importer.importManaged(
    from: OpenMedManagedAppConfiguration(),
    approvedRoots: [appGroupInbox],
    context: try PortableBundleContext(now: nowTimestamp, target: deviceTarget),
    activationCheck: { lease in canLoadModel(at: lease.payloadDirectory) })
if receipt.outcome == .activated { reloadModel() }
```

Android:

```kotlin
val cache = ModelCache(context).bundleGenerations()
val importer = ManagedBundleImporter(
    cache, File(context.noBackupFilesDir, "managed-import"),
    ContentResolverStreamOpener(context.contentResolver),
)
importer.recover()
val receipt = importer.importManaged(
    source = RestrictionsManagerConfigurationSource(context),
    context = PortableBundleContext(now = nowTimestamp, target = deviceTarget),
    approvedRoots = listOf(File(context.filesDir, "managed-inbox")),
    approvedAuthorities = setOf("com.example.emm.files"),
    activationCheck = { lease -> canLoadModel(lease.payloadDirectory) },
)
```

Pass `handoff:` to import a bundle the app received itself, for example a
stream from a managed document provider, while still taking trust roots and
policy from the managed configuration. `importBundle` accepts a parsed
configuration directly for apps that provision trust roots another way.

## Shared vectors

`tests/fixtures/managed_import/vectors.json` holds tiny synthetic signed
bundles, deterministic ZIP archives (including hostile ones), managed
configurations, and scenarios covering idempotency, deflated and nested
archives, stream and directory handoffs, staging, probation, activation-check
rollback, signature, rotation, revocation, expiry, downgrade, pins,
expectations, configuration errors, archive and inventory failures, budgets,
storage pressure, cancellation, and process death at each checkpoint. Swift and
Android replay them with fake managed configurations, fake clocks, storage
faults, and crash checkpoints; each kit verifies against its own platform
target. Regenerate or check them with:

```bash
python tests/fixtures/managed_import/generate_vectors.py
python tests/fixtures/managed_import/generate_vectors.py --check
```

The archives are signed with TEST-ONLY keys. Never trust them in production.

## Limits of this contract

- The tests use fake managed configurations, fake content providers, and
  local files on the host and the JVM. Delivery through a real MDM or EMM
  product, and simulator or emulator test hosts, are not exercised.
- ZIP64 archives are refused; deliver bundles of 4 GiB or more as a directory.
- The importer does not discover or download bundles. Network acquisition stays
  with the app, which can plug a downloader into the cache as a range source.
- Activation never widens clinical or device support claims; the manifest's
  compatibility section and the app's own policy decide what the model may do.
