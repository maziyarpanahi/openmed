package com.openmed.openmedkit.secrets

/**
 * What a provider-held secret is for.
 *
 * Each purpose owns one alias namespace and one algorithm, so a key created
 * for de-identification hashing can never be used to seal storage blobs or to
 * vouch for a model bundle.
 */
enum class OpenMedSecretPurpose(
    internal val aliasSegment: String,
    val algorithm: OpenMedSecretAlgorithm,
) {
    /** HMAC key for de-identification hash replacements and span digests. */
    DEIDENTIFY_HASH("deid-hash", OpenMedSecretAlgorithm.HMAC_SHA256),

    /** HMAC key for portable policy secrets, such as pseudonym scopes. */
    POLICY_SECRET("policy", OpenMedSecretAlgorithm.HMAC_SHA256),

    /** HMAC key for local model-bundle trust records. */
    BUNDLE_TRUST("bundle-trust", OpenMedSecretAlgorithm.HMAC_SHA256),

    /** AES-GCM key that seals small secret blobs the host app stores itself. */
    SECRET_STORAGE("storage", OpenMedSecretAlgorithm.AES_256_GCM),
}

/** Key algorithms a provider may hold. */
enum class OpenMedSecretAlgorithm(
    internal val jcaName: String,
    internal val minimumImportBytes: Int,
) {
    /** HMAC-SHA256 with `sign` and `verify` purposes. */
    HMAC_SHA256("HmacSHA256", 16),

    /** AES-256 in GCM mode with randomized encryption. */
    AES_256_GCM("AES", 32),
}

/**
 * Opaque reference to one secret: a purpose plus a caller-chosen label.
 *
 * The label is an application identifier such as `default` or `site-a`. It
 * must match `[a-z0-9][a-z0-9-]{0,47}` and must never carry names, record
 * numbers, or other sensitive values. Keystore aliases are derived from the
 * reference and a version number; they never contain key material.
 */
data class OpenMedSecretReference(
    val purpose: OpenMedSecretPurpose,
    val label: String = DEFAULT_LABEL,
) {
    init {
        require(LABEL_PATTERN.matches(label)) {
            "secret reference label must match [a-z0-9][a-z0-9-]{0,47}"
        }
    }

    internal val aliasPrefix: String
        get() = "openmed.${purpose.aliasSegment}.$label.v"

    internal fun alias(version: Int): String = "$aliasPrefix$version"

    internal fun versionOf(alias: String): Int? {
        if (!alias.startsWith(aliasPrefix)) return null
        val suffix = alias.substring(aliasPrefix.length)
        if (suffix.isEmpty() || suffix[0] == '0' || !suffix.all { it in '0'..'9' }) {
            return null
        }
        return suffix.toIntOrNull()
    }

    companion object {
        /** Label used when a caller does not choose one. */
        const val DEFAULT_LABEL = "default"

        // Declared before DEIDENTIFY_HASH so the init check can use it.
        private val LABEL_PATTERN = Regex("[a-z0-9][a-z0-9-]{0,47}")

        /** Default reference for de-identification hashing. */
        val DEIDENTIFY_HASH = OpenMedSecretReference(OpenMedSecretPurpose.DEIDENTIFY_HASH)
    }
}

/**
 * Value-free reason codes for secret-provider failures.
 *
 * Codes never carry key material, aliases, payloads, or platform exception
 * messages, so they are safe to log, count, or show in diagnostics.
 */
enum class OpenMedSecretFailure(val code: String) {
    /** No key exists for the reference or requested version. */
    MISSING("secret_missing"),

    /** The device is locked, or required user authentication is absent or expired. */
    LOCKED("secret_locked"),

    /** The key was permanently invalidated, for example by a lock-screen change. */
    INVALIDATED("secret_invalidated"),

    /** A required protection property or operation is unavailable here. */
    UNSUPPORTED("secret_unsupported"),

    /** Sealed data failed authentication or belongs to another key version. */
    INTEGRITY_FAILED("secret_integrity_failed"),

    /** The key is the public OpenMedKit default and a strict policy refuses it. */
    PUBLIC_DEFAULT("secret_public_default"),

    /** Any other Keystore or provider failure. */
    PROVIDER_ERROR("secret_provider_error"),
}

/** Thrown by [OpenMedSecretResult.getOrThrow]; the message is the reason code only. */
class OpenMedSecretException(val failure: OpenMedSecretFailure) : RuntimeException(failure.code)

/** Typed, value-free outcome of a secret-provider operation. */
sealed class OpenMedSecretResult<out T> {
    /** Successful outcome. [toString] never prints [value]. */
    class Success<out T>(val value: T) : OpenMedSecretResult<T>() {
        override fun toString(): String = "OpenMedSecretResult.Success"
    }

    /** Failed outcome carrying only a reason code. */
    data class Failure(val failure: OpenMedSecretFailure) : OpenMedSecretResult<Nothing>()

    /** The failure reason, or `null` on success. */
    val failureOrNull: OpenMedSecretFailure?
        get() = (this as? Failure)?.failure

    /** Return the value or throw [OpenMedSecretException] with the reason code. */
    fun getOrThrow(): T = when (this) {
        is Success -> value
        is Failure -> throw OpenMedSecretException(failure)
    }
}

/**
 * Where a key lives, ordered by isolation strength.
 *
 * [SECURE_HARDWARE] means the key is hardware-backed but the OS (API 30 and
 * lower) does not say whether it is in the TEE or StrongBox. It satisfies a
 * trusted-environment requirement and never satisfies a StrongBox requirement.
 */
enum class OpenMedSecurityLevel(private val rank: Int, val code: String) {
    UNKNOWN(-1, "unknown"),
    SOFTWARE(0, "software"),
    SECURE_HARDWARE(1, "secure_hardware"),
    TRUSTED_ENVIRONMENT(1, "trusted_environment"),
    STRONGBOX(2, "strongbox"),
    ;

    /** Whether a key at this level meets [minimum]. */
    fun satisfies(minimum: OpenMedSecurityLevel): Boolean =
        if (minimum == STRONGBOX) this == STRONGBOX else rank >= minimum.rank
}

/** How a provider should use StrongBox when creating a key. */
enum class OpenMedStrongBoxMode {
    /** Never request StrongBox. */
    OFF,

    /** Request StrongBox and fall back to the TEE, reporting the fallback. */
    PREFERRED,

    /** Require a verifiable StrongBox key (API 31+); otherwise refuse. */
    REQUIRED,
}

/**
 * Time-bound user-authentication requirement for a key.
 *
 * @property validitySeconds Seconds a key stays usable after the user unlocks
 *   or authenticates. Must be at least 1; per-operation authentication through
 *   a `BiometricPrompt.CryptoObject` is outside this adapter.
 * @property allowDeviceCredential Accept PIN, pattern, or password.
 * @property allowStrongBiometric Accept a Class 3 biometric.
 */
data class OpenMedUserAuthentication(
    val validitySeconds: Int,
    val allowDeviceCredential: Boolean = true,
    val allowStrongBiometric: Boolean = true,
) {
    init {
        require(validitySeconds >= 1) { "validitySeconds must be at least 1" }
        require(allowDeviceCredential || allowStrongBiometric) {
            "at least one authenticator type must be allowed"
        }
    }
}

/**
 * Protection a caller requires of a key.
 *
 * Providers create keys with exactly these properties and refuse, with
 * [OpenMedSecretFailure.UNSUPPORTED], any key or device that cannot meet them.
 * They never fall back to a weaker key or to plaintext storage.
 *
 * @property minimumSecurityLevel [OpenMedSecurityLevel.TRUSTED_ENVIRONMENT]
 *   (default) or [OpenMedSecurityLevel.SOFTWARE] for emulators and development.
 *   Use [strongBox] for a StrongBox requirement.
 * @property strongBox StrongBox request mode for new keys.
 * @property userAuthentication Required user authentication, or `null`.
 * @property invalidatedByBiometricEnrollment Invalidate an authentication-bound
 *   key when a new biometric is enrolled.
 * @property unlockedDeviceRequired Refuse key use while the device is locked.
 */
data class OpenMedSecretPolicy(
    val minimumSecurityLevel: OpenMedSecurityLevel = OpenMedSecurityLevel.TRUSTED_ENVIRONMENT,
    val strongBox: OpenMedStrongBoxMode = OpenMedStrongBoxMode.OFF,
    val userAuthentication: OpenMedUserAuthentication? = null,
    val invalidatedByBiometricEnrollment: Boolean = true,
    val unlockedDeviceRequired: Boolean = false,
) {
    init {
        require(
            minimumSecurityLevel == OpenMedSecurityLevel.SOFTWARE ||
                minimumSecurityLevel == OpenMedSecurityLevel.TRUSTED_ENVIRONMENT,
        ) { "minimumSecurityLevel must be SOFTWARE or TRUSTED_ENVIRONMENT" }
    }

    /** The level a key must reach, including a StrongBox requirement. */
    val effectiveMinimumSecurityLevel: OpenMedSecurityLevel
        get() = if (strongBox == OpenMedStrongBoxMode.REQUIRED) {
            OpenMedSecurityLevel.STRONGBOX
        } else {
            minimumSecurityLevel
        }

    /** Whether a key with [properties] meets this policy. */
    fun isSatisfiedBy(properties: OpenMedSecretProperties): Boolean {
        if (!properties.securityLevel.satisfies(effectiveMinimumSecurityLevel)) return false
        val auth = userAuthentication ?: return true
        return properties.userAuthenticationRequired &&
            properties.userAuthenticationValiditySeconds in 1..auth.validitySeconds
    }

    companion object {
        /** Hardware-backed keys, no user authentication. */
        val DEFAULT = OpenMedSecretPolicy()
    }
}

/** How a key entered the provider. */
enum class OpenMedSecretOrigin {
    /** Generated inside the key store; the material never existed outside it. */
    GENERATED,

    /** Imported from caller-supplied material, for example a portable secret. */
    IMPORTED,

    /** The platform did not report an origin. */
    UNKNOWN,
}

/**
 * Security properties a provider actually observed for one key version.
 *
 * Values come from the platform (`KeyInfo` on Android), not from the request.
 * The object never holds key material.
 */
data class OpenMedSecretProperties(
    val reference: OpenMedSecretReference,
    val version: Int,
    val origin: OpenMedSecretOrigin,
    val securityLevel: OpenMedSecurityLevel,
    val userAuthenticationRequired: Boolean,
    val userAuthenticationValiditySeconds: Int,
    val userAuthenticationEnforcedBySecureHardware: Boolean,
    val invalidatedByBiometricEnrollment: Boolean,
) {
    /** Algorithm fixed by the reference purpose. */
    val algorithm: OpenMedSecretAlgorithm
        get() = reference.purpose.algorithm
}

/**
 * Result of creating a key version.
 *
 * @property strongBoxRequested Whether creation asked the platform for StrongBox.
 * @property strongBoxFallback Whether StrongBox was preferred but unavailable,
 *   so the key was created without it.
 */
data class OpenMedSecretCreated(
    val properties: OpenMedSecretProperties,
    val strongBoxRequested: Boolean,
    val strongBoxFallback: Boolean,
)

/**
 * AES-GCM output of [OpenMedSecretHandle.seal]: IV, ciphertext, and tag.
 *
 * The host app may store [toByteArray] itself; exclude that storage from
 * backup because the sealing key never leaves this device.
 */
class OpenMedSealedSecret(val version: Int, bytes: ByteArray) {
    private val payload = bytes.copyOf()

    init {
        require(version >= 1) { "version must be positive" }
        require(payload.isNotEmpty()) { "sealed payload must not be empty" }
    }

    /** Return a copy of the sealed bytes. */
    fun toByteArray(): ByteArray = payload.copyOf()

    override fun toString(): String =
        "OpenMedSealedSecret(version=$version, bytes=${payload.size})"
}

/**
 * Use of one resolved key version without exposing its material.
 *
 * HMAC purposes support [mac]; [OpenMedSecretPurpose.SECRET_STORAGE] supports
 * [seal] and [open]. Any other combination returns
 * [OpenMedSecretFailure.UNSUPPORTED].
 */
interface OpenMedSecretHandle {
    /** Observed properties of the resolved key version. */
    val properties: OpenMedSecretProperties

    /** Compute HMAC-SHA256 over [payload] inside the provider. */
    fun mac(payload: ByteArray): OpenMedSecretResult<ByteArray>

    /** Seal [plaintext] with AES-GCM bound to [associatedData]. */
    fun seal(
        plaintext: ByteArray,
        associatedData: ByteArray = ByteArray(0),
    ): OpenMedSecretResult<OpenMedSealedSecret>

    /** Open data sealed by this key version with the same [associatedData]. */
    fun open(
        sealed: OpenMedSealedSecret,
        associatedData: ByteArray = ByteArray(0),
    ): OpenMedSecretResult<ByteArray>
}

/**
 * Injectable custody boundary for portable policy and bundle-trust secrets.
 *
 * Operations address secrets by an opaque [OpenMedSecretReference] and return
 * typed, value-free results. Implementations keep material inside their store,
 * write nothing to preferences, caches, or logs, and refuse a required
 * protection property instead of weakening it.
 */
interface OpenMedSecretProvider {
    /**
     * Resolve the newest version, or [version], and confirm it is usable.
     *
     * Returns [OpenMedSecretFailure.MISSING], [OpenMedSecretFailure.LOCKED],
     * [OpenMedSecretFailure.INVALIDATED], or [OpenMedSecretFailure.UNSUPPORTED]
     * when the key is absent, locked, invalidated, or below [policy].
     */
    fun get(
        reference: OpenMedSecretReference,
        version: Int? = null,
        policy: OpenMedSecretPolicy = OpenMedSecretPolicy.DEFAULT,
    ): OpenMedSecretResult<OpenMedSecretHandle>

    /** Import caller-managed [material] as the next version of [reference]. */
    fun put(
        reference: OpenMedSecretReference,
        material: ByteArray,
        policy: OpenMedSecretPolicy = OpenMedSecretPolicy.DEFAULT,
    ): OpenMedSecretResult<OpenMedSecretCreated>

    /**
     * Generate the next version of [reference] inside the store.
     *
     * Earlier versions stay available for verification until [delete].
     */
    fun rotate(
        reference: OpenMedSecretReference,
        policy: OpenMedSecretPolicy = OpenMedSecretPolicy.DEFAULT,
    ): OpenMedSecretResult<OpenMedSecretCreated>

    /** Delete one version, or every version when [version] is `null`. */
    fun delete(
        reference: OpenMedSecretReference,
        version: Int? = null,
    ): OpenMedSecretResult<Unit>

    /** Report observed properties for every version, oldest first, without using a key. */
    fun describe(
        reference: OpenMedSecretReference,
    ): OpenMedSecretResult<List<OpenMedSecretProperties>>
}
