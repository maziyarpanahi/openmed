package com.openmed.openmedkit.secrets

import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyInfo
import android.security.keystore.KeyProperties
import android.security.keystore.KeyProtection
import java.security.KeyStore
import java.security.KeyStoreException
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.Mac
import javax.crypto.SecretKey
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/** Key-creation request an [OpenMedKeystore] must honor exactly. */
data class OpenMedKeystoreKeySpec(
    val alias: String,
    val algorithm: OpenMedSecretAlgorithm,
    val strongBoxBacked: Boolean,
    val userAuthentication: OpenMedUserAuthentication?,
    val invalidatedByBiometricEnrollment: Boolean,
    val unlockedDeviceRequired: Boolean,
)

/** Platform-reported facts about one stored key; never the key itself. */
data class OpenMedKeystoreKeyInfo(
    val securityLevel: OpenMedSecurityLevel,
    val origin: OpenMedSecretOrigin,
    val userAuthenticationRequired: Boolean,
    val userAuthenticationValiditySeconds: Int,
    val userAuthenticationEnforcedBySecureHardware: Boolean,
    val invalidatedByBiometricEnrollment: Boolean,
)

/**
 * Narrow key-store seam behind [AndroidKeystoreSecretProvider].
 *
 * [AndroidKeystore] is the production implementation. Host JVM tests inject a
 * fake that throws the same platform exceptions. Implementations signal
 * failure by throwing; the provider maps every exception to a value-free
 * [OpenMedSecretFailure] and never surfaces the exception message.
 */
interface OpenMedKeystore {
    /** Android API level whose Keystore features this store exposes. */
    val apiLevel: Int

    /** Every alias currently in the store. */
    fun aliases(): Set<String>

    /** Generate a non-exportable key inside the store. */
    fun generate(spec: OpenMedKeystoreKeySpec)

    /** Import [material]; the stored copy becomes non-exportable. */
    fun import(spec: OpenMedKeystoreKeySpec, material: ByteArray)

    /** Delete [alias] if present. */
    fun delete(alias: String)

    /** Facts for [alias], or `null` when it does not exist. */
    fun keyInfo(alias: String): OpenMedKeystoreKeyInfo?

    /** HMAC-SHA256 of [payload] computed inside the store. */
    fun mac(alias: String, payload: ByteArray): ByteArray

    /** AES-GCM encrypt; returns the 12-byte IV followed by ciphertext and tag. */
    fun encrypt(alias: String, plaintext: ByteArray, associatedData: ByteArray): ByteArray

    /** AES-GCM decrypt output of [encrypt]. */
    fun decrypt(alias: String, sealed: ByteArray, associatedData: ByteArray): ByteArray
}

/**
 * [OpenMedKeystore] backed by the platform `AndroidKeyStore` provider.
 *
 * Keys are generated or imported with an explicit [KeyGenParameterSpec] or
 * [KeyProtection] and are never exportable. Keystore entries are bound to the
 * device and are not part of Auto Backup or device-to-device transfer. On a
 * host without `AndroidKeyStore` every operation fails with
 * [OpenMedSecretFailure.UNSUPPORTED]; there is no software fallback.
 */
class AndroidKeystore internal constructor(override val apiLevel: Int) : OpenMedKeystore {
    constructor() : this(Build.VERSION.SDK_INT)

    private val keyStore: KeyStore by lazy {
        val store = try {
            KeyStore.getInstance(PROVIDER)
        } catch (_: KeyStoreException) {
            throw OpenMedSecretException(OpenMedSecretFailure.UNSUPPORTED)
        }
        store.load(null)
        store
    }

    override fun aliases(): Set<String> = keyStore.aliases().toList().toSet()

    override fun generate(spec: OpenMedKeystoreKeySpec) {
        val generator = KeyGenerator.getInstance(spec.algorithm.jcaName, PROVIDER)
        generator.init(keyGenParameterSpec(spec))
        generator.generateKey()
    }

    override fun import(spec: OpenMedKeystoreKeySpec, material: ByteArray) {
        val entry = KeyStore.SecretKeyEntry(SecretKeySpec(material, spec.algorithm.jcaName))
        keyStore.setEntry(spec.alias, entry, keyProtection(spec))
    }

    override fun delete(alias: String) {
        if (keyStore.containsAlias(alias)) keyStore.deleteEntry(alias)
    }

    override fun keyInfo(alias: String): OpenMedKeystoreKeyInfo? {
        val key = keyStore.getKey(alias, null) as? SecretKey ?: return null
        val info = SecretKeyFactory.getInstance(key.algorithm, PROVIDER)
            .getKeySpec(key, KeyInfo::class.java) as KeyInfo
        @Suppress("DEPRECATION")
        val level = if (apiLevel >= 31) {
            securityLevelFromCode(info.securityLevel)
        } else {
            securityLevelFromHardwareFlag(info.isInsideSecureHardware)
        }
        return OpenMedKeystoreKeyInfo(
            securityLevel = level,
            origin = originFromCode(info.origin),
            userAuthenticationRequired = info.isUserAuthenticationRequired,
            userAuthenticationValiditySeconds = info.userAuthenticationValidityDurationSeconds,
            userAuthenticationEnforcedBySecureHardware =
            info.isUserAuthenticationRequirementEnforcedBySecureHardware,
            invalidatedByBiometricEnrollment = info.isInvalidatedByBiometricEnrollment,
        )
    }

    override fun mac(alias: String, payload: ByteArray): ByteArray {
        val mac = Mac.getInstance(OpenMedSecretAlgorithm.HMAC_SHA256.jcaName)
        mac.init(secretKey(alias))
        return mac.doFinal(payload)
    }

    override fun encrypt(
        alias: String,
        plaintext: ByteArray,
        associatedData: ByteArray,
    ): ByteArray {
        val cipher = Cipher.getInstance(AES_GCM)
        cipher.init(Cipher.ENCRYPT_MODE, secretKey(alias))
        cipher.updateAAD(associatedData)
        val body = cipher.doFinal(plaintext)
        val iv = cipher.iv
        if (iv.size != GCM_IV_BYTES) {
            throw OpenMedSecretException(OpenMedSecretFailure.PROVIDER_ERROR)
        }
        return iv + body
    }

    override fun decrypt(
        alias: String,
        sealed: ByteArray,
        associatedData: ByteArray,
    ): ByteArray {
        if (sealed.size <= GCM_IV_BYTES) {
            throw OpenMedSecretException(OpenMedSecretFailure.INTEGRITY_FAILED)
        }
        val cipher = Cipher.getInstance(AES_GCM)
        cipher.init(
            Cipher.DECRYPT_MODE,
            secretKey(alias),
            GCMParameterSpec(GCM_TAG_BITS, sealed, 0, GCM_IV_BYTES),
        )
        cipher.updateAAD(associatedData)
        return cipher.doFinal(sealed, GCM_IV_BYTES, sealed.size - GCM_IV_BYTES)
    }

    private fun secretKey(alias: String): SecretKey =
        keyStore.getKey(alias, null) as? SecretKey
            ?: throw OpenMedSecretException(OpenMedSecretFailure.MISSING)

    /** Build the exact generation spec for [spec]; exposed for host tests. */
    internal fun keyGenParameterSpec(spec: OpenMedKeystoreKeySpec): KeyGenParameterSpec {
        val builder = KeyGenParameterSpec.Builder(spec.alias, purposesFor(spec.algorithm))
        when (spec.algorithm) {
            OpenMedSecretAlgorithm.HMAC_SHA256 -> builder.setDigests(KeyProperties.DIGEST_SHA256)
            OpenMedSecretAlgorithm.AES_256_GCM -> builder
                .setKeySize(AES_KEY_BITS)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
        }
        spec.userAuthentication?.let { auth ->
            builder.setUserAuthenticationRequired(true)
            if (apiLevel >= 30) {
                builder.setUserAuthenticationParameters(auth.validitySeconds, authenticators(auth))
            } else {
                @Suppress("DEPRECATION")
                builder.setUserAuthenticationValidityDurationSeconds(auth.validitySeconds)
            }
            builder.setInvalidatedByBiometricEnrollment(spec.invalidatedByBiometricEnrollment)
        }
        if (spec.strongBoxBacked) builder.setIsStrongBoxBacked(true)
        if (spec.unlockedDeviceRequired) builder.setUnlockedDeviceRequired(true)
        return builder.build()
    }

    /** Build the exact import protection for [spec]; exposed for host tests. */
    internal fun keyProtection(spec: OpenMedKeystoreKeySpec): KeyProtection {
        val builder = KeyProtection.Builder(purposesFor(spec.algorithm))
        when (spec.algorithm) {
            OpenMedSecretAlgorithm.HMAC_SHA256 -> builder.setDigests(KeyProperties.DIGEST_SHA256)
            OpenMedSecretAlgorithm.AES_256_GCM -> builder
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
        }
        spec.userAuthentication?.let { auth ->
            builder.setUserAuthenticationRequired(true)
            if (apiLevel >= 30) {
                builder.setUserAuthenticationParameters(auth.validitySeconds, authenticators(auth))
            } else {
                @Suppress("DEPRECATION")
                builder.setUserAuthenticationValidityDurationSeconds(auth.validitySeconds)
            }
            builder.setInvalidatedByBiometricEnrollment(spec.invalidatedByBiometricEnrollment)
        }
        if (spec.strongBoxBacked) builder.setIsStrongBoxBacked(true)
        if (spec.unlockedDeviceRequired) builder.setUnlockedDeviceRequired(true)
        return builder.build()
    }

    private companion object {
        const val PROVIDER = "AndroidKeyStore"
        const val AES_GCM = "AES/GCM/NoPadding"
        const val AES_KEY_BITS = 256
        const val GCM_IV_BYTES = 12
        const val GCM_TAG_BITS = 128

        fun purposesFor(algorithm: OpenMedSecretAlgorithm): Int = when (algorithm) {
            OpenMedSecretAlgorithm.HMAC_SHA256 ->
                KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY
            OpenMedSecretAlgorithm.AES_256_GCM ->
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        }

        fun authenticators(auth: OpenMedUserAuthentication): Int {
            var types = 0
            if (auth.allowDeviceCredential) types = types or KeyProperties.AUTH_DEVICE_CREDENTIAL
            if (auth.allowStrongBiometric) types = types or KeyProperties.AUTH_BIOMETRIC_STRONG
            return types
        }
    }
}

/** Map a `KeyInfo.getSecurityLevel()` code (API 31+). */
internal fun securityLevelFromCode(code: Int): OpenMedSecurityLevel = when (code) {
    KeyProperties.SECURITY_LEVEL_STRONGBOX -> OpenMedSecurityLevel.STRONGBOX
    KeyProperties.SECURITY_LEVEL_TRUSTED_ENVIRONMENT -> OpenMedSecurityLevel.TRUSTED_ENVIRONMENT
    KeyProperties.SECURITY_LEVEL_UNKNOWN_SECURE -> OpenMedSecurityLevel.SECURE_HARDWARE
    KeyProperties.SECURITY_LEVEL_SOFTWARE -> OpenMedSecurityLevel.SOFTWARE
    else -> OpenMedSecurityLevel.UNKNOWN
}

/** Map the pre-API-31 `KeyInfo.isInsideSecureHardware()` flag. */
internal fun securityLevelFromHardwareFlag(insideSecureHardware: Boolean): OpenMedSecurityLevel =
    if (insideSecureHardware) {
        OpenMedSecurityLevel.SECURE_HARDWARE
    } else {
        OpenMedSecurityLevel.SOFTWARE
    }

/** Map a `KeyInfo.getOrigin()` code. */
internal fun originFromCode(code: Int): OpenMedSecretOrigin = when (code) {
    KeyProperties.ORIGIN_GENERATED -> OpenMedSecretOrigin.GENERATED
    KeyProperties.ORIGIN_IMPORTED, KeyProperties.ORIGIN_SECURELY_IMPORTED ->
        OpenMedSecretOrigin.IMPORTED
    else -> OpenMedSecretOrigin.UNKNOWN
}
