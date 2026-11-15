package com.openmed.openmedkit.secrets

import android.security.keystore.KeyPermanentlyInvalidatedException
import android.security.keystore.StrongBoxUnavailableException
import android.security.keystore.UserNotAuthenticatedException
import java.security.InvalidAlgorithmParameterException
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.Mac
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * In-memory [OpenMedKeystore] for host tests.
 *
 * It throws the same platform exceptions as `AndroidKeyStore` (construct them
 * under Robolectric), so the provider's mapping is exercised end to end. Key
 * material stays inside this object; tests can only ask whether it holds a
 * given value.
 */
internal class FakeKeystore(
    override val apiLevel: Int = 33,
    var hardwareLevel: OpenMedSecurityLevel = OpenMedSecurityLevel.TRUSTED_ENVIRONMENT,
    var strongBoxAvailable: Boolean = false,
    var secureLockScreen: Boolean = true,
) : OpenMedKeystore {
    private class Entry(
        val material: ByteArray,
        val spec: OpenMedKeystoreKeySpec,
        val origin: OpenMedSecretOrigin,
        val level: OpenMedSecurityLevel,
    )

    private val entries = linkedMapOf<String, Entry>()
    private val random = SecureRandom()

    /** Every creation request, including ones the store rejected. */
    val requestedSpecs = mutableListOf<OpenMedKeystoreKeySpec>()
    val generatedAliases = mutableListOf<String>()
    val importedAliases = mutableListOf<String>()
    val invalidated = mutableSetOf<String>()
    var userAuthenticated = true
    var deviceLocked = false

    /** Thrown by every operation while set. */
    var failure: Exception? = null

    fun holdsMaterial(material: ByteArray): Boolean =
        entries.values.any { it.material.contentEquals(material) }

    override fun aliases(): Set<String> {
        injected()
        return entries.keys.toSet()
    }

    override fun generate(spec: OpenMedKeystoreKeySpec) {
        checkCreate(spec)
        entries[spec.alias] =
            Entry(randomBytes(32), spec, OpenMedSecretOrigin.GENERATED, levelFor(spec))
        generatedAliases += spec.alias
    }

    override fun import(spec: OpenMedKeystoreKeySpec, material: ByteArray) {
        checkCreate(spec)
        entries[spec.alias] =
            Entry(material.copyOf(), spec, OpenMedSecretOrigin.IMPORTED, levelFor(spec))
        importedAliases += spec.alias
    }

    override fun delete(alias: String) {
        injected()
        entries.remove(alias)?.material?.fill(0)
    }

    override fun keyInfo(alias: String): OpenMedKeystoreKeyInfo? {
        injected()
        val entry = entries[alias] ?: return null
        val auth = entry.spec.userAuthentication
        return OpenMedKeystoreKeyInfo(
            securityLevel = entry.level,
            origin = entry.origin,
            userAuthenticationRequired = auth != null,
            userAuthenticationValiditySeconds = auth?.validitySeconds ?: 0,
            userAuthenticationEnforcedBySecureHardware =
            auth != null && entry.level != OpenMedSecurityLevel.SOFTWARE,
            invalidatedByBiometricEnrollment =
            auth != null && entry.spec.invalidatedByBiometricEnrollment,
        )
    }

    override fun mac(alias: String, payload: ByteArray): ByteArray = use(alias) { material ->
        Mac.getInstance("HmacSHA256").run {
            init(SecretKeySpec(material, "HmacSHA256"))
            doFinal(payload)
        }
    }

    override fun encrypt(
        alias: String,
        plaintext: ByteArray,
        associatedData: ByteArray,
    ): ByteArray = use(alias) { material ->
        val iv = randomBytes(12)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(material, "AES"), GCMParameterSpec(128, iv))
        cipher.updateAAD(associatedData)
        iv + cipher.doFinal(plaintext)
    }

    override fun decrypt(
        alias: String,
        sealed: ByteArray,
        associatedData: ByteArray,
    ): ByteArray = use(alias) { material ->
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(
            Cipher.DECRYPT_MODE,
            SecretKeySpec(material, "AES"),
            GCMParameterSpec(128, sealed, 0, 12),
        )
        cipher.updateAAD(associatedData)
        cipher.doFinal(sealed, 12, sealed.size - 12)
    }

    private fun checkCreate(spec: OpenMedKeystoreKeySpec) {
        requestedSpecs += spec
        injected()
        if (spec.strongBoxBacked && !strongBoxAvailable) throw StrongBoxUnavailableException()
        if (spec.userAuthentication != null && !secureLockScreen) {
            throw InvalidAlgorithmParameterException("SENTINEL-lock-screen ${spec.alias}")
        }
    }

    private fun levelFor(spec: OpenMedKeystoreKeySpec): OpenMedSecurityLevel =
        if (spec.strongBoxBacked) OpenMedSecurityLevel.STRONGBOX else hardwareLevel

    private fun <T> use(alias: String, block: (ByteArray) -> T): T {
        injected()
        val entry = entries[alias] ?: throw OpenMedSecretException(OpenMedSecretFailure.MISSING)
        if (alias in invalidated) throw KeyPermanentlyInvalidatedException()
        if (entry.spec.userAuthentication != null && !userAuthenticated) {
            throw UserNotAuthenticatedException()
        }
        if (entry.spec.unlockedDeviceRequired && deviceLocked) {
            throw UserNotAuthenticatedException()
        }
        return block(entry.material)
    }

    private fun injected() {
        failure?.let { throw it }
    }

    private fun randomBytes(size: Int): ByteArray = ByteArray(size).also(random::nextBytes)
}
