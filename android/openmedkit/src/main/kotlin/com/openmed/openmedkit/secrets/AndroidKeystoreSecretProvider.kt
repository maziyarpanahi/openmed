package com.openmed.openmedkit.secrets

import android.os.Build
import android.security.keystore.KeyPermanentlyInvalidatedException
import android.security.keystore.StrongBoxUnavailableException
import android.security.keystore.UserNotAuthenticatedException
import java.security.InvalidAlgorithmParameterException
import java.security.UnrecoverableKeyException
import javax.crypto.AEADBadTagException

/**
 * [OpenMedSecretProvider] that keeps every key inside the Android Keystore.
 *
 * Each [OpenMedSecretReference] maps to aliases
 * `openmed.<purpose>.<label>.v<version>`, and version state is read back from
 * the Keystore alias list, so the provider writes nothing to preferences,
 * files, or caches. New keys are generated (or imported) non-exportable with
 * the explicit purposes, authentication, and StrongBox settings of the
 * requested [OpenMedSecretPolicy]. After creation the provider reads the
 * platform `KeyInfo` and deletes the new key instead of returning it when the
 * observed protection is weaker than the policy requires.
 *
 * Platform exceptions become value-free codes: a permanently invalidated key is
 * [OpenMedSecretFailure.INVALIDATED], missing or expired user authentication is
 * [OpenMedSecretFailure.LOCKED], an unavailable required feature is
 * [OpenMedSecretFailure.UNSUPPORTED], and anything else is
 * [OpenMedSecretFailure.PROVIDER_ERROR].
 *
 * @param keystore Key-store seam; inject a fake for host JVM tests.
 */
class AndroidKeystoreSecretProvider(
    private val keystore: OpenMedKeystore = AndroidKeystore(),
) : OpenMedSecretProvider {
    override fun get(
        reference: OpenMedSecretReference,
        version: Int?,
        policy: OpenMedSecretPolicy,
    ): OpenMedSecretResult<OpenMedSecretHandle> = guarded {
        val resolved = version ?: versions(reference).lastOrNull() ?: return missing()
        val alias = reference.alias(resolved)
        val info = keystore.keyInfo(alias) ?: return missing()
        val properties = propertiesOf(reference, resolved, info)
        if (!policy.isSatisfiedBy(properties)) {
            return OpenMedSecretResult.Failure(OpenMedSecretFailure.UNSUPPORTED)
        }
        val handle = KeystoreSecretHandle(keystore, alias, properties)
        handle.probe()
        OpenMedSecretResult.Success(handle)
    }

    @Synchronized
    override fun put(
        reference: OpenMedSecretReference,
        material: ByteArray,
        policy: OpenMedSecretPolicy,
    ): OpenMedSecretResult<OpenMedSecretCreated> {
        val algorithm = reference.purpose.algorithm
        require(
            material.size >= algorithm.minimumImportBytes &&
                (algorithm != OpenMedSecretAlgorithm.AES_256_GCM || material.size == 32),
        ) { "secret material length does not fit the reference purpose" }
        return create(reference, policy, material)
    }

    @Synchronized
    override fun rotate(
        reference: OpenMedSecretReference,
        policy: OpenMedSecretPolicy,
    ): OpenMedSecretResult<OpenMedSecretCreated> =
        create(reference, policy, material = null)

    @Synchronized
    override fun delete(
        reference: OpenMedSecretReference,
        version: Int?,
    ): OpenMedSecretResult<Unit> = guarded {
        val present = versions(reference)
        val targets = if (version == null) present else present.filter { it == version }
        if (targets.isEmpty()) return missing()
        targets.forEach { keystore.delete(reference.alias(it)) }
        OpenMedSecretResult.Success(Unit)
    }

    override fun describe(
        reference: OpenMedSecretReference,
    ): OpenMedSecretResult<List<OpenMedSecretProperties>> = guarded {
        val described = versions(reference).mapNotNull { version ->
            keystore.keyInfo(reference.alias(version))?.let {
                propertiesOf(reference, version, it)
            }
        }
        if (described.isEmpty()) missing() else OpenMedSecretResult.Success(described)
    }

    /** Generate ([material] `null`) or import the next version. */
    private fun create(
        reference: OpenMedSecretReference,
        policy: OpenMedSecretPolicy,
        material: ByteArray?,
    ): OpenMedSecretResult<OpenMedSecretCreated> {
        if (!platformCanHonor(policy)) {
            return OpenMedSecretResult.Failure(OpenMedSecretFailure.UNSUPPORTED)
        }
        var alias: String? = null
        return try {
            val version = (versions(reference).lastOrNull() ?: 0) + 1
            val newAlias = reference.alias(version)
            alias = newAlias
            val requestStrongBox = policy.strongBox != OpenMedStrongBoxMode.OFF &&
                keystore.apiLevel >= STRONGBOX_REQUEST_API
            var fallback = policy.strongBox == OpenMedStrongBoxMode.PREFERRED &&
                !requestStrongBox
            try {
                store(specFor(newAlias, reference, policy, requestStrongBox), material)
            } catch (error: Exception) {
                val canFallBack = policy.strongBox == OpenMedStrongBoxMode.PREFERRED &&
                    isStrongBoxUnavailable(error)
                if (!canFallBack) throw error
                keystore.delete(newAlias)
                store(specFor(newAlias, reference, policy, strongBox = false), material)
                fallback = true
            }
            val info = keystore.keyInfo(newAlias)
                ?: throw OpenMedSecretException(OpenMedSecretFailure.PROVIDER_ERROR)
            val properties = propertiesOf(reference, version, info)
            if (!policy.isSatisfiedBy(properties)) {
                keystore.delete(newAlias)
                alias = null
                return OpenMedSecretResult.Failure(OpenMedSecretFailure.UNSUPPORTED)
            }
            alias = null
            OpenMedSecretResult.Success(
                OpenMedSecretCreated(properties, requestStrongBox, fallback),
            )
        } catch (error: Exception) {
            alias?.let { runCatching { keystore.delete(it) } }
            OpenMedSecretResult.Failure(classifySecretFailure(error, creating = true))
        }
    }

    private fun store(spec: OpenMedKeystoreKeySpec, material: ByteArray?) {
        if (material == null) keystore.generate(spec) else keystore.import(spec, material)
    }

    private fun platformCanHonor(policy: OpenMedSecretPolicy): Boolean {
        val api = keystore.apiLevel
        if (policy.strongBox == OpenMedStrongBoxMode.REQUIRED && api < STRONGBOX_VERIFY_API) {
            return false
        }
        if (policy.unlockedDeviceRequired && api < STRONGBOX_REQUEST_API) return false
        val auth = policy.userAuthentication ?: return true
        val bothAuthenticators = auth.allowDeviceCredential && auth.allowStrongBiometric
        return api >= AUTH_TYPES_API || bothAuthenticators
    }

    private fun specFor(
        alias: String,
        reference: OpenMedSecretReference,
        policy: OpenMedSecretPolicy,
        strongBox: Boolean,
    ) = OpenMedKeystoreKeySpec(
        alias = alias,
        algorithm = reference.purpose.algorithm,
        strongBoxBacked = strongBox,
        userAuthentication = policy.userAuthentication,
        invalidatedByBiometricEnrollment = policy.invalidatedByBiometricEnrollment,
        unlockedDeviceRequired = policy.unlockedDeviceRequired,
    )

    private fun versions(reference: OpenMedSecretReference): List<Int> =
        keystore.aliases().mapNotNull(reference::versionOf).sorted()

    private fun propertiesOf(
        reference: OpenMedSecretReference,
        version: Int,
        info: OpenMedKeystoreKeyInfo,
    ) = OpenMedSecretProperties(
        reference = reference,
        version = version,
        origin = info.origin,
        securityLevel = info.securityLevel,
        userAuthenticationRequired = info.userAuthenticationRequired,
        userAuthenticationValiditySeconds = info.userAuthenticationValiditySeconds,
        userAuthenticationEnforcedBySecureHardware =
        info.userAuthenticationEnforcedBySecureHardware,
        invalidatedByBiometricEnrollment = info.invalidatedByBiometricEnrollment,
    )

    private fun <T> missing(): OpenMedSecretResult<T> =
        OpenMedSecretResult.Failure(OpenMedSecretFailure.MISSING)

    private companion object {
        const val STRONGBOX_REQUEST_API = 28
        const val AUTH_TYPES_API = 30
        const val STRONGBOX_VERIFY_API = 31
    }
}

private class KeystoreSecretHandle(
    private val keystore: OpenMedKeystore,
    private val alias: String,
    override val properties: OpenMedSecretProperties,
) : OpenMedSecretHandle {
    private val algorithm = properties.algorithm

    override fun mac(payload: ByteArray): OpenMedSecretResult<ByteArray> = guarded {
        if (algorithm != OpenMedSecretAlgorithm.HMAC_SHA256) return unsupported()
        OpenMedSecretResult.Success(keystore.mac(alias, payload))
    }

    override fun seal(
        plaintext: ByteArray,
        associatedData: ByteArray,
    ): OpenMedSecretResult<OpenMedSealedSecret> = guarded {
        if (algorithm != OpenMedSecretAlgorithm.AES_256_GCM) return unsupported()
        val sealed = keystore.encrypt(alias, plaintext, associatedData)
        OpenMedSecretResult.Success(OpenMedSealedSecret(properties.version, sealed))
    }

    override fun open(
        sealed: OpenMedSealedSecret,
        associatedData: ByteArray,
    ): OpenMedSecretResult<ByteArray> = guarded {
        if (algorithm != OpenMedSecretAlgorithm.AES_256_GCM) return unsupported()
        if (sealed.version != properties.version) {
            return OpenMedSecretResult.Failure(OpenMedSecretFailure.INTEGRITY_FAILED)
        }
        OpenMedSecretResult.Success(keystore.decrypt(alias, sealed.toByteArray(), associatedData))
    }

    /** Use the key once so lock and invalidation surface before real work. */
    fun probe() {
        when (algorithm) {
            OpenMedSecretAlgorithm.HMAC_SHA256 -> keystore.mac(alias, PROBE)
            OpenMedSecretAlgorithm.AES_256_GCM -> keystore.encrypt(alias, ByteArray(0), PROBE)
        }
    }

    override fun toString(): String {
        val purpose = properties.reference.purpose
        return "OpenMedSecretHandle(purpose=$purpose, version=${properties.version})"
    }

    private fun <T> unsupported(): OpenMedSecretResult<T> =
        OpenMedSecretResult.Failure(OpenMedSecretFailure.UNSUPPORTED)

    private companion object {
        val PROBE = "openmed:secret-probe:v1".toByteArray(Charsets.UTF_8)
    }
}

private inline fun <T> guarded(block: () -> OpenMedSecretResult<T>): OpenMedSecretResult<T> =
    try {
        block()
    } catch (error: Exception) {
        OpenMedSecretResult.Failure(classifySecretFailure(error, creating = false))
    }

/**
 * Map a Keystore or JCA exception, including wrapped causes, to a reason code.
 *
 * The exception message is never inspected or kept.
 */
internal fun classifySecretFailure(error: Throwable, creating: Boolean): OpenMedSecretFailure {
    var current: Throwable? = error
    var depth = 0
    while (current != null && depth < MAX_CAUSE_DEPTH) {
        platformFailure(current)?.let { return it }
        current = current.cause
        depth += 1
    }
    val unsupportedRequest = error is InvalidAlgorithmParameterException ||
        error is IllegalStateException ||
        error.cause is InvalidAlgorithmParameterException
    return if (creating && unsupportedRequest) {
        OpenMedSecretFailure.UNSUPPORTED
    } else {
        OpenMedSecretFailure.PROVIDER_ERROR
    }
}

private fun platformFailure(error: Throwable): OpenMedSecretFailure? {
    val sdk = Build.VERSION.SDK_INT
    return when {
        error is OpenMedSecretException -> error.failure
        error is KeyPermanentlyInvalidatedException -> OpenMedSecretFailure.INVALIDATED
        error is UnrecoverableKeyException -> OpenMedSecretFailure.INVALIDATED
        error is UserNotAuthenticatedException -> OpenMedSecretFailure.LOCKED
        error is AEADBadTagException -> OpenMedSecretFailure.INTEGRITY_FAILED
        sdk >= 28 && error is StrongBoxUnavailableException -> OpenMedSecretFailure.UNSUPPORTED
        sdk >= 33 && error is android.security.KeyStoreException -> when {
            error.requiresUserAuthentication() -> OpenMedSecretFailure.LOCKED
            error.numericErrorCode ==
                android.security.KeyStoreException.ERROR_KEY_DOES_NOT_EXIST ->
                OpenMedSecretFailure.MISSING
            else -> null
        }
        else -> null
    }
}

private fun isStrongBoxUnavailable(error: Throwable): Boolean {
    if (Build.VERSION.SDK_INT < 28) return false
    var current: Throwable? = error
    var depth = 0
    while (current != null && depth < MAX_CAUSE_DEPTH) {
        if (current is StrongBoxUnavailableException) return true
        current = current.cause
        depth += 1
    }
    return false
}

private const val MAX_CAUSE_DEPTH = 6
