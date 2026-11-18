# Android Keystore Secrets

OpenMedKit Android keeps portable policy secrets, bundle-trust keys, and the
de-identification hash key inside the Android Keystore. The
`com.openmed.openmedkit.secrets` package defines a small, injectable
`OpenMedSecretProvider` contract and one production implementation,
`AndroidKeystoreSecretProvider`. The contract uses the same operations planned
for the Swift Keychain adapter (get, put, rotate, and delete by opaque
reference), so the two kits stay parallel.

The provider owns key custody on one device. It is not enterprise key
management, it does not sync keys, and it does not change any hashing or
surrogate algorithm.

## References and aliases

An `OpenMedSecretReference` is a purpose plus a caller-chosen label:

| Purpose | Algorithm | Operations |
| --- | --- | --- |
| `DEIDENTIFY_HASH` | HMAC-SHA256 | `mac` |
| `POLICY_SECRET` | HMAC-SHA256 | `mac` |
| `BUNDLE_TRUST` | HMAC-SHA256 | `mac` |
| `SECRET_STORAGE` | AES-256-GCM | `seal`, `open` |

Labels must match `[a-z0-9][a-z0-9-]{0,47}`, for example `default` or
`site-a`. Do not put names, record numbers, or other sensitive values in a
label. Each version is stored under the alias
`openmed.<purpose>.<label>.v<version>`, so purposes and labels never share a
key. The provider reads version state back from the Keystore alias list and
writes nothing to `SharedPreferences`, files, caches, or logs.

## Operations

```kotlin
val provider = AndroidKeystoreSecretProvider()
val reference = OpenMedSecretReference(OpenMedSecretPurpose.POLICY_SECRET, "site-a")

// Generate version 1 inside the Keystore. The material is never exportable.
val created = provider.rotate(reference).getOrThrow()

// Or import a portable secret shared with other kits as the next version.
provider.put(reference, portableSecretBytes)

when (val result = provider.get(reference)) {
    is OpenMedSecretResult.Success -> result.value.mac(payload)
    is OpenMedSecretResult.Failure -> showReason(result.failure.code)
}
```

| Call | Behavior |
| --- | --- |
| `get(reference, version, policy)` | Resolves the newest version (or `version`), checks it against `policy`, and uses the key once so a lock or invalidation is reported before real work. |
| `put(reference, material, policy)` | Imports caller-managed material as the next version. HMAC material must be at least 16 bytes; AES material must be 32 bytes. The provider keeps no copy. |
| `rotate(reference, policy)` | Generates the next version inside the Keystore. Older versions stay usable until deleted, so earlier digests and sealed blobs can still be verified. |
| `delete(reference, version)` | Deletes one version, or every version when `version` is `null`. |
| `describe(reference)` | Reports observed properties for every version without using a key. |

Every call returns `OpenMedSecretResult`. `getOrThrow()` raises
`OpenMedSecretException`, whose message is the reason code alone.

## Required protection and reported properties

`OpenMedSecretPolicy` states what a key must have. The provider creates keys
with an explicit `KeyGenParameterSpec` (or `KeyProtection` for imports) that
sets purposes, digests or GCM mode, user authentication, StrongBox, and
unlocked-device requirements. After creation it reads the platform `KeyInfo`.
If the observed protection is weaker than the policy, it deletes the new key
and returns `UNSUPPORTED`. It never falls back to a weaker key or to plaintext
storage.

| Policy field | Default | Notes |
| --- | --- | --- |
| `minimumSecurityLevel` | `TRUSTED_ENVIRONMENT` | Use `SOFTWARE` only for emulators and development builds. |
| `strongBox` | `OFF` | `PREFERRED` requests StrongBox and reports `strongBoxFallback = true` when it is unavailable. `REQUIRED` refuses instead and needs API 31+, where the tier is verifiable. |
| `userAuthentication` | `null` | Time-bound authentication with `validitySeconds >= 1`. Restricting to one authenticator type needs API 30+. |
| `invalidatedByBiometricEnrollment` | `true` | Applies to authentication-bound keys. |
| `unlockedDeviceRequired` | `false` | Needs API 28+. |

`OpenMedSecretProperties` reports what the platform observed, not what was
requested: `securityLevel`, `origin` (generated or imported), whether user
authentication is required and for how long, whether secure hardware enforces
it, and whether biometric enrollment invalidates the key. Security levels are:

| Level | Meaning |
| --- | --- |
| `STRONGBOX` | `KeyInfo` reports StrongBox (API 31+). |
| `TRUSTED_ENVIRONMENT` | `KeyInfo` reports the TEE (API 31+). |
| `SECURE_HARDWARE` | Hardware-backed, but API 30 and lower do not say which tier. It satisfies a TEE requirement and never a StrongBox requirement. |
| `SOFTWARE` | Software-backed Keystore, as on most emulators. |
| `UNKNOWN` | The platform reported no level. |

These values are platform reports. The library makes no blanket claim that a
device has secure hardware; qualify specific devices in a device lane.

## Failure codes

| Code | Raised when |
| --- | --- |
| `secret_missing` | No key exists for the reference or version. |
| `secret_locked` | The device is locked, or user authentication is absent or expired. |
| `secret_invalidated` | The key was permanently invalidated (`KeyPermanentlyInvalidatedException`), for example after the lock screen was removed or a biometric was enrolled. Recover with `rotate`, then `delete` the old version. |
| `secret_unsupported` | A required property cannot be met on this device or OS, the key is weaker than the policy, the operation does not fit the purpose, or the host has no `AndroidKeyStore`. |
| `secret_integrity_failed` | A sealed blob failed authentication, used other associated data, or belongs to another key version. |
| `secret_public_default` | A strict de-identification engine was given the public default salt. |
| `secret_provider_error` | Any other Keystore failure. |

Codes never include key material, aliases, payloads, or platform exception
messages, and `OpenMedSecretException` keeps no cause. Results, handles, and
sealed blobs print only purposes, versions, and byte counts.

## De-identification hashing

`com.openmed.openmedkit.deid.DeidentifyEngine` now keys its `HASH`
replacements and span digests through a provider:

```kotlin
val provider = AndroidKeystoreSecretProvider()
provider.rotate(OpenMedSecretReference.DEIDENTIFY_HASH)
val engine = DeidentifyEngine(provider)
```

To keep digests consistent with another kit or the Python surface, import the
same portable secret with `put(OpenMedSecretReference.DEIDENTIFY_HASH, bytes)`;
the digest format (`hmac-sha256:<hex>` over `LABEL\0surface`) is unchanged.

- `DeidentifyEngine()` no longer uses a built-in salt. It masks, removes, and
  replaces, leaves `textHash` empty, and refuses `HASH` with `secret_missing`.
- A provider-backed engine resolves the newest key version on each call, so a
  rotation applies to the next call. A locked, missing, or invalidated key
  stops the call before any output is produced.
- `DeidentifyEngine(hashSalt)` is deprecated. Under the default
  `HashKeyPolicy.STRICT` it refuses `LEGACY_PUBLIC_HASH_SALT`, and a
  provider-backed engine refuses a stored key equal to that public salt.
- To reproduce pre-4.0 digests during a migration, pass
  `HashKeyPolicy.ALLOW_LEGACY_PUBLIC_SALT` explicitly. The legacy salt is
  public, so those digests offer no protection against a dictionary attack.

## Backup exclusion for host apps

Keystore keys are bound to the device and are never part of Auto Backup or
device-to-device transfer. Data that depends on them is still backed up
unless the host app excludes it. After a restore on another device the keys
are absent, so sealed blobs cannot be opened and new digests no longer match.
Exclude any files, databases, or preferences that hold `OpenMedSealedSecret`
bytes or reference-dependent state.

For Android 11 and lower, point `android:fullBackupContent` at a rules file:

```xml
<!-- res/xml/backup_rules.xml -->
<full-backup-content>
    <exclude domain="file" path="openmed/" />
    <exclude domain="sharedpref" path="openmed_sealed.xml" />
</full-backup-content>
```

For Android 12 and higher, also set `android:dataExtractionRules`:

```xml
<!-- res/xml/data_extraction_rules.xml -->
<data-extraction-rules>
    <cloud-backup>
        <exclude domain="file" path="openmed/" />
        <exclude domain="sharedpref" path="openmed_sealed.xml" />
    </cloud-backup>
    <device-transfer>
        <exclude domain="file" path="openmed/" />
        <exclude domain="sharedpref" path="openmed_sealed.xml" />
    </device-transfer>
</data-extraction-rules>
```

```xml
<application
    android:fullBackupContent="@xml/backup_rules"
    android:dataExtractionRules="@xml/data_extraction_rules">
```

The library manifest sets neither attribute, so the host app keeps control of
its backup policy. The Maple demo app excludes every domain as a stricter
example.

## Testing

`AndroidKeystoreSecretProvider` takes an `OpenMedKeystore`, a narrow seam over
generate, import, delete, `KeyInfo`, and the HMAC and AES-GCM operations.
Host JVM tests inject an in-memory fake that throws the same platform
exceptions, so locked, invalidated, missing, rotated, and unsupported cases run
offline under Robolectric:

```bash
cd android
ANDROID_HOME=$HOME/Library/Android/sdk ./gradlew :openmedkit:testDebugUnitTest \
  --tests '*secrets*' --tests '*DeidentifyEngineTest'
```

The tests also check the exact `KeyGenParameterSpec` and `KeyProtection` the
production store builds, and that the real provider fails closed with
`secret_unsupported` on a host without `AndroidKeyStore`.

StrongBox, TEE, lock-screen, and biometric-invalidation behavior needs a real
device. The module has no `androidTest` source set yet; an optional device lane
should run the same scenarios against `AndroidKeystore()` on hardware and
record the reported `securityLevel` per device, rather than assume it.
