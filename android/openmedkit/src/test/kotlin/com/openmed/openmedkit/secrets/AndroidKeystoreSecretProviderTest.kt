package com.openmed.openmedkit.secrets

import android.security.keystore.KeyPermanentlyInvalidatedException
import android.security.keystore.StrongBoxUnavailableException
import android.security.keystore.UserNotAuthenticatedException
import java.security.InvalidAlgorithmParameterException
import java.security.ProviderException
import java.security.UnrecoverableKeyException
import javax.crypto.AEADBadTagException
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class AndroidKeystoreSecretProviderTest {
    private val reference = OpenMedSecretReference(OpenMedSecretPurpose.DEIDENTIFY_HASH, "site-a")
    private val payload = "synthetic-payload".toByteArray()

    @Test
    fun rotateGeneratesVersionedKeysInsideTheKeystore() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)

        val first = provider.rotate(reference).getOrThrow()
        val second = provider.rotate(reference).getOrThrow()

        assertEquals(1, first.properties.version)
        assertEquals(2, second.properties.version)
        assertEquals(OpenMedSecretOrigin.GENERATED, first.properties.origin)
        assertEquals(OpenMedSecurityLevel.TRUSTED_ENVIRONMENT, second.properties.securityLevel)
        assertFalse(second.strongBoxRequested)
        assertEquals(
            listOf("openmed.deid-hash.site-a.v1", "openmed.deid-hash.site-a.v2"),
            keystore.generatedAliases,
        )
        assertTrue(keystore.importedAliases.isEmpty())

        val latest = provider.get(reference).getOrThrow()
        val pinned = provider.get(reference, version = 1).getOrThrow()
        assertEquals(2, latest.properties.version)
        assertEquals(1, pinned.properties.version)
        assertFalse(
            latest.mac(payload).getOrThrow().contentEquals(pinned.mac(payload).getOrThrow()),
        )
        assertEquals(listOf(1, 2), provider.describe(reference).getOrThrow().map { it.version })
    }

    @Test
    fun purposesAndLabelsUseSeparateAliasNamespaces() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore())
        provider.rotate(reference).getOrThrow()

        val otherPurpose = OpenMedSecretReference(OpenMedSecretPurpose.POLICY_SECRET, "site-a")
        val prefixedLabel = OpenMedSecretReference(OpenMedSecretPurpose.DEIDENTIFY_HASH, "site-a1")
        assertEquals(OpenMedSecretFailure.MISSING, provider.get(otherPurpose).failureOrNull)
        assertEquals(OpenMedSecretFailure.MISSING, provider.get(prefixedLabel).failureOrNull)

        provider.rotate(prefixedLabel).getOrThrow()
        assertEquals(listOf(1), provider.describe(reference).getOrThrow().map { it.version })
    }

    @Test
    fun missingAliasesAndVersionsAreTypedFailures() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore())

        assertEquals(OpenMedSecretFailure.MISSING, provider.get(reference).failureOrNull)
        assertEquals(OpenMedSecretFailure.MISSING, provider.describe(reference).failureOrNull)
        assertEquals(OpenMedSecretFailure.MISSING, provider.delete(reference).failureOrNull)

        provider.rotate(reference).getOrThrow()
        assertEquals(
            OpenMedSecretFailure.MISSING,
            provider.get(reference, version = 7).failureOrNull,
        )
        assertEquals(OpenMedSecretFailure.MISSING, provider.delete(reference, 7).failureOrNull)
    }

    @Test
    fun authenticationBoundKeysReportLockedWithoutFallingBack() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        val policy = OpenMedSecretPolicy(userAuthentication = OpenMedUserAuthentication(30))

        val created = provider.rotate(reference, policy).getOrThrow().properties
        assertTrue(created.userAuthenticationRequired)
        assertEquals(30, created.userAuthenticationValiditySeconds)
        assertTrue(created.userAuthenticationEnforcedBySecureHardware)
        val requested = keystore.requestedSpecs.single()
        assertEquals(OpenMedUserAuthentication(30), requested.userAuthentication)

        keystore.userAuthenticated = false
        assertEquals(
            OpenMedSecretFailure.LOCKED,
            provider.get(reference, policy = policy).failureOrNull,
        )

        keystore.userAuthenticated = true
        val handle = provider.get(reference, policy = policy).getOrThrow()
        keystore.userAuthenticated = false
        assertEquals(OpenMedSecretFailure.LOCKED, handle.mac(payload).failureOrNull)
    }

    @Test
    fun unlockedDeviceRequiredKeysAreLockedWhileTheDeviceIsLocked() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        val policy = OpenMedSecretPolicy(unlockedDeviceRequired = true)

        provider.rotate(reference, policy).getOrThrow()
        assertTrue(keystore.requestedSpecs.single().unlockedDeviceRequired)

        keystore.deviceLocked = true
        assertEquals(
            OpenMedSecretFailure.LOCKED,
            provider.get(reference, policy = policy).failureOrNull,
        )

        val legacy = AndroidKeystoreSecretProvider(FakeKeystore(apiLevel = 27))
        assertEquals(
            OpenMedSecretFailure.UNSUPPORTED,
            legacy.rotate(reference, policy).failureOrNull,
        )
    }

    @Test
    fun invalidatedKeysAreReportedAndRotationRecovers() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        provider.rotate(reference).getOrThrow()

        keystore.invalidated += "openmed.deid-hash.site-a.v1"
        assertEquals(OpenMedSecretFailure.INVALIDATED, provider.get(reference).failureOrNull)

        provider.rotate(reference).getOrThrow()
        assertEquals(2, provider.get(reference).getOrThrow().properties.version)
        assertEquals(OpenMedSecretFailure.INVALIDATED, provider.get(reference, 1).failureOrNull)

        provider.delete(reference, version = 1).getOrThrow()
        assertEquals(listOf(2), provider.describe(reference).getOrThrow().map { it.version })
    }

    @Test
    fun preferredStrongBoxFallsBackAndReportsTheFallback() {
        val keystore = FakeKeystore(strongBoxAvailable = false)
        val provider = AndroidKeystoreSecretProvider(keystore)

        val created = provider.rotate(
            reference,
            OpenMedSecretPolicy(strongBox = OpenMedStrongBoxMode.PREFERRED),
        ).getOrThrow()

        assertTrue(created.strongBoxRequested)
        assertTrue(created.strongBoxFallback)
        assertEquals(OpenMedSecurityLevel.TRUSTED_ENVIRONMENT, created.properties.securityLevel)
        assertEquals(listOf(true, false), keystore.requestedSpecs.map { it.strongBoxBacked })
        assertEquals(setOf("openmed.deid-hash.site-a.v1"), keystore.aliases())
    }

    @Test
    fun preferredStrongBoxUsesStrongBoxWhenPresent() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore(strongBoxAvailable = true))

        val created = provider.rotate(
            reference,
            OpenMedSecretPolicy(strongBox = OpenMedStrongBoxMode.PREFERRED),
        ).getOrThrow()

        assertTrue(created.strongBoxRequested)
        assertFalse(created.strongBoxFallback)
        assertEquals(OpenMedSecurityLevel.STRONGBOX, created.properties.securityLevel)
    }

    @Test
    fun preferredStrongBoxBeforeApi28ReportsFallbackWithoutRequestingIt() {
        val keystore = FakeKeystore(apiLevel = 27, strongBoxAvailable = true)
        val provider = AndroidKeystoreSecretProvider(keystore)

        val created = provider.rotate(
            reference,
            OpenMedSecretPolicy(strongBox = OpenMedStrongBoxMode.PREFERRED),
        ).getOrThrow()

        assertFalse(created.strongBoxRequested)
        assertTrue(created.strongBoxFallback)
        assertFalse(keystore.requestedSpecs.single().strongBoxBacked)
    }

    @Test
    fun requiredStrongBoxRefusesInsteadOfFallingBack() {
        val required = OpenMedSecretPolicy(strongBox = OpenMedStrongBoxMode.REQUIRED)

        val unavailable = FakeKeystore(strongBoxAvailable = false)
        val refused = AndroidKeystoreSecretProvider(unavailable).rotate(reference, required)
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, refused.failureOrNull)
        assertTrue(unavailable.aliases().isEmpty())
        assertEquals(1, unavailable.requestedSpecs.size)

        val unverifiable = FakeKeystore(apiLevel = 30, strongBoxAvailable = true)
        val unverified = AndroidKeystoreSecretProvider(unverifiable).rotate(reference, required)
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, unverified.failureOrNull)
        assertTrue(unverifiable.requestedSpecs.isEmpty())

        val available = AndroidKeystoreSecretProvider(FakeKeystore(strongBoxAvailable = true))
        val created = available.rotate(reference, required).getOrThrow()
        assertEquals(OpenMedSecurityLevel.STRONGBOX, created.properties.securityLevel)
        assertTrue(available.get(reference, policy = required) is OpenMedSecretResult.Success)
    }

    @Test
    fun softwareOnlyKeysAreRefusedUnlessThePolicyAllowsThem() {
        val keystore = FakeKeystore(hardwareLevel = OpenMedSecurityLevel.SOFTWARE)
        val provider = AndroidKeystoreSecretProvider(keystore)

        assertEquals(OpenMedSecretFailure.UNSUPPORTED, provider.rotate(reference).failureOrNull)
        assertTrue(keystore.aliases().isEmpty())

        val software = OpenMedSecretPolicy(minimumSecurityLevel = OpenMedSecurityLevel.SOFTWARE)
        val created = provider.rotate(reference, software).getOrThrow()
        assertEquals(OpenMedSecurityLevel.SOFTWARE, created.properties.securityLevel)
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, provider.get(reference).failureOrNull)
        assertTrue(provider.get(reference, policy = software) is OpenMedSecretResult.Success)
    }

    @Test
    fun unreportedHardwareTierSatisfiesTheTeeButNeverStrongBox() {
        val tee = OpenMedSecurityLevel.TRUSTED_ENVIRONMENT
        val strongBox = OpenMedSecurityLevel.STRONGBOX

        assertTrue(OpenMedSecurityLevel.SECURE_HARDWARE.satisfies(tee))
        assertFalse(OpenMedSecurityLevel.SECURE_HARDWARE.satisfies(strongBox))
        assertFalse(OpenMedSecurityLevel.UNKNOWN.satisfies(tee))
        assertFalse(OpenMedSecurityLevel.SOFTWARE.satisfies(tee))
        assertTrue(strongBox.satisfies(tee))

        val keystore = FakeKeystore(hardwareLevel = OpenMedSecurityLevel.SECURE_HARDWARE)
        val provider = AndroidKeystoreSecretProvider(keystore)
        val created = provider.rotate(reference).getOrThrow()
        assertEquals(OpenMedSecurityLevel.SECURE_HARDWARE, created.properties.securityLevel)
    }

    @Test
    fun authenticationRequirementsThatCannotBeHonoredAreUnsupported() {
        val policy = OpenMedSecretPolicy(userAuthentication = OpenMedUserAuthentication(30))
        val noLockScreen = FakeKeystore(secureLockScreen = false)
        val refused = AndroidKeystoreSecretProvider(noLockScreen).rotate(reference, policy)
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, refused.failureOrNull)
        assertTrue(noLockScreen.aliases().isEmpty())

        val biometricOnly = OpenMedSecretPolicy(
            userAuthentication = OpenMedUserAuthentication(30, allowDeviceCredential = false),
        )
        val legacy = AndroidKeystoreSecretProvider(FakeKeystore(apiLevel = 29))
        assertEquals(
            OpenMedSecretFailure.UNSUPPORTED,
            legacy.rotate(reference, biometricOnly).failureOrNull,
        )
    }

    @Test
    fun storedKeysWeakerThanTheRequestedPolicyAreRefused() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore())
        val lenient = OpenMedSecretPolicy(userAuthentication = OpenMedUserAuthentication(300))
        provider.rotate(reference, lenient).getOrThrow()

        val stricter = OpenMedSecretPolicy(userAuthentication = OpenMedUserAuthentication(30))
        assertEquals(
            OpenMedSecretFailure.UNSUPPORTED,
            provider.get(reference, policy = stricter).failureOrNull,
        )

        val unauthenticated = OpenMedSecretReference(OpenMedSecretPurpose.BUNDLE_TRUST, "catalog")
        provider.rotate(unauthenticated).getOrThrow()
        assertEquals(
            OpenMedSecretFailure.UNSUPPORTED,
            provider.get(unauthenticated, policy = stricter).failureOrNull,
        )
    }

    @Test
    fun putImportsPortableMaterialAsNonExportableVersions() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        val material = ByteArray(32) { (it * 7 + 3).toByte() }

        val created = provider.put(reference, material).getOrThrow()
        val next = provider.put(reference, material.reversedArray()).getOrThrow()

        assertEquals(OpenMedSecretOrigin.IMPORTED, created.properties.origin)
        assertEquals(2, next.properties.version)
        assertEquals(
            listOf("openmed.deid-hash.site-a.v1", "openmed.deid-hash.site-a.v2"),
            keystore.importedAliases,
        )
        val firstHandle = provider.get(reference, 1).getOrThrow()
        assertArrayEquals(hmac(material, payload), firstHandle.mac(payload).getOrThrow())
        assertTrue(keystore.holdsMaterial(material))
        assertTrue(
            AndroidKeystoreSecretProvider::class.java.declaredFields.none {
                it.type == ByteArray::class.java
            },
        )
    }

    @Test
    fun putRejectsMaterialTooShortForThePurpose() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore())
        val storage = OpenMedSecretReference(OpenMedSecretPurpose.SECRET_STORAGE, "bundle-cache")

        val short = runCatching { provider.put(reference, ByteArray(8) { 0x41 }) }.exceptionOrNull()
        val aes128 = runCatching { provider.put(storage, ByteArray(16) { 0x41 }) }.exceptionOrNull()

        assertTrue(short is IllegalArgumentException)
        assertTrue(aes128 is IllegalArgumentException)
        assertFalse(short!!.message.orEmpty().contains("AAAA"))
    }

    @Test
    fun storageKeysSealAndOpenBoundToAssociatedData() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        val storage = OpenMedSecretReference(OpenMedSecretPurpose.SECRET_STORAGE, "bundle-cache")
        val plaintext = "synthetic-trust-token".toByteArray()
        val context = "bundle:synthetic".toByteArray()

        provider.rotate(storage).getOrThrow()
        assertEquals(OpenMedSecretAlgorithm.AES_256_GCM, keystore.requestedSpecs.single().algorithm)
        val handle = provider.get(storage).getOrThrow()
        val sealed = handle.seal(plaintext, context).getOrThrow()

        val sealedText = String(sealed.toByteArray(), Charsets.ISO_8859_1)
        assertFalse(sealedText.contains("synthetic-trust-token"))
        assertFalse(sealed.toString().contains("synthetic-trust-token"))
        assertArrayEquals(plaintext, handle.open(sealed, context).getOrThrow())
        assertEquals(OpenMedSecretFailure.INTEGRITY_FAILED, handle.open(sealed).failureOrNull)

        val tampered = sealed.toByteArray().also {
            it[it.size - 1] = (it.last().toInt() xor 1).toByte()
        }
        assertEquals(
            OpenMedSecretFailure.INTEGRITY_FAILED,
            handle.open(OpenMedSealedSecret(sealed.version, tampered), context).failureOrNull,
        )

        provider.rotate(storage).getOrThrow()
        val rotated = provider.get(storage).getOrThrow()
        assertEquals(
            OpenMedSecretFailure.INTEGRITY_FAILED,
            rotated.open(sealed, context).failureOrNull,
        )
        val original = provider.get(storage, sealed.version).getOrThrow()
        assertArrayEquals(plaintext, original.open(sealed, context).getOrThrow())
    }

    @Test
    fun operationsOutsideTheKeyPurposeAreUnsupported() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore())
        val storage = OpenMedSecretReference(OpenMedSecretPurpose.SECRET_STORAGE, "bundle-cache")
        provider.rotate(reference).getOrThrow()
        provider.rotate(storage).getOrThrow()

        val hmacHandle = provider.get(reference).getOrThrow()
        val storageHandle = provider.get(storage).getOrThrow()

        assertEquals(OpenMedSecretFailure.UNSUPPORTED, hmacHandle.seal(payload).failureOrNull)
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, storageHandle.mac(payload).failureOrNull)
        val sealed = storageHandle.seal(payload).getOrThrow()
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, hmacHandle.open(sealed).failureOrNull)
    }

    @Test
    fun deleteRemovesOneVersionOrEveryVersion() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        repeat(3) { provider.rotate(reference).getOrThrow() }

        provider.delete(reference, version = 2).getOrThrow()
        assertEquals(listOf(1, 3), provider.describe(reference).getOrThrow().map { it.version })
        assertEquals(4, provider.rotate(reference).getOrThrow().properties.version)

        provider.delete(reference).getOrThrow()
        assertTrue(keystore.aliases().isEmpty())
        assertEquals(OpenMedSecretFailure.MISSING, provider.get(reference).failureOrNull)
    }

    @Test
    fun platformFailuresBecomeValueFreeCodes() {
        val keystore = FakeKeystore()
        val provider = AndroidKeystoreSecretProvider(keystore)
        provider.rotate(reference).getOrThrow()
        keystore.failure = ProviderException(
            "SENTINEL-alias openmed.deid-hash.site-a.v1",
            RuntimeException("SENTINEL-cause"),
        )

        val result = provider.get(reference)
        val thrown = runCatching { result.getOrThrow() }.exceptionOrNull()

        assertEquals(OpenMedSecretFailure.PROVIDER_ERROR, result.failureOrNull)
        assertTrue(thrown is OpenMedSecretException)
        assertEquals("secret_provider_error", thrown!!.message)
        assertNull(thrown.cause)
        assertFalse(result.toString().contains("SENTINEL"))
        assertEquals(OpenMedSecretFailure.PROVIDER_ERROR, provider.rotate(reference).failureOrNull)
    }

    @Test
    fun wrappedPlatformExceptionsAreClassifiedThroughTheirCauses() {
        fun read(error: Throwable) = classifySecretFailure(error, creating = false)
        fun create(error: Throwable) = classifySecretFailure(error, creating = true)

        assertEquals(
            OpenMedSecretFailure.INVALIDATED,
            read(ProviderException(KeyPermanentlyInvalidatedException())),
        )
        assertEquals(OpenMedSecretFailure.INVALIDATED, read(UnrecoverableKeyException()))
        assertEquals(
            OpenMedSecretFailure.LOCKED,
            read(RuntimeException(UserNotAuthenticatedException())),
        )
        assertEquals(OpenMedSecretFailure.INTEGRITY_FAILED, read(AEADBadTagException()))
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, create(StrongBoxUnavailableException()))
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, create(InvalidAlgorithmParameterException()))
        assertEquals(
            OpenMedSecretFailure.PROVIDER_ERROR,
            read(InvalidAlgorithmParameterException()),
        )
        assertEquals(
            OpenMedSecretFailure.LOCKED,
            read(IllegalStateException(OpenMedSecretException(OpenMedSecretFailure.LOCKED))),
        )
        assertEquals(OpenMedSecretFailure.PROVIDER_ERROR, read(RuntimeException("SENTINEL")))
    }

    @Test
    fun resultsHandlesAndPropertiesNeverPrintSecretValues() {
        val provider = AndroidKeystoreSecretProvider(FakeKeystore())
        val material = ByteArray(32) { (0x30 + it % 10).toByte() }
        val created = provider.put(reference, material)
        val resolved = provider.get(reference)
        val handle = resolved.getOrThrow()
        val mac = handle.mac(payload)
        val macHex = mac.getOrThrow().joinToString("") { "%02x".format(it) }

        val rendered = listOf(created, resolved, handle, mac, handle.properties).joinToString("|")

        assertFalse(rendered.contains(String(material, Charsets.US_ASCII)))
        assertFalse(rendered.contains(macHex))
        assertFalse(rendered.contains("synthetic-payload"))
        assertTrue(OpenMedSecretFailure.values().all { it.code.matches(Regex("secret_[a-z_]+")) })
    }

    @Test
    fun referenceLabelsAreBoundedOpaqueIdentifiers() {
        val rejected = listOf(
            "", "Site-A", "jane.doe@example.org", "a.b", "-lead", "x".repeat(49), "a b",
        )
        for (label in rejected) {
            val error = runCatching {
                OpenMedSecretReference(OpenMedSecretPurpose.POLICY_SECRET, label)
            }.exceptionOrNull()
            assertTrue(label, error is IllegalArgumentException)
            if (label.isNotEmpty()) assertFalse(error!!.message.orEmpty().contains(label))
        }

        OpenMedSecretReference(OpenMedSecretPurpose.POLICY_SECRET, "x".repeat(48))
        assertNull(reference.versionOf("openmed.deid-hash.site-a.v01"))
        assertNull(reference.versionOf("openmed.deid-hash.site-a.v"))
        assertNull(reference.versionOf("openmed.deid-hash.site-a.v1x"))
        assertEquals(12, reference.versionOf("openmed.deid-hash.site-a.v12"))
    }

    @Test
    fun hostWithoutAndroidKeystoreFailsClosed() {
        val provider = AndroidKeystoreSecretProvider()

        assertEquals(OpenMedSecretFailure.UNSUPPORTED, provider.rotate(reference).failureOrNull)
        assertEquals(OpenMedSecretFailure.UNSUPPORTED, provider.get(reference).failureOrNull)
        assertEquals(
            OpenMedSecretFailure.UNSUPPORTED,
            provider.put(reference, ByteArray(32) { 1 }).failureOrNull,
        )
    }

    private fun hmac(key: ByteArray, data: ByteArray): ByteArray =
        Mac.getInstance("HmacSHA256").run {
            init(SecretKeySpec(key, "HmacSHA256"))
            doFinal(data)
        }
}
