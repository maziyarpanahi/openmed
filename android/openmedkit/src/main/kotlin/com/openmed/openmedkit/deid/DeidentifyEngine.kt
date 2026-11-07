package com.openmed.openmedkit.deid

import com.openmed.openmedkit.secrets.OpenMedSecretException
import com.openmed.openmedkit.secrets.OpenMedSecretFailure
import com.openmed.openmedkit.secrets.OpenMedSecretPolicy
import com.openmed.openmedkit.secrets.OpenMedSecretProvider
import com.openmed.openmedkit.secrets.OpenMedSecretPurpose
import com.openmed.openmedkit.secrets.OpenMedSecretReference
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Locale
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.math.max
import kotlin.math.min

/** How a [DeidentifyEngine] treats the public, pre-4.0 default hash key. */
enum class HashKeyPolicy {
    /** Refuse the public default key however it is supplied (default). */
    STRICT,

    /** Accept the public default key only to reproduce pre-4.0 digests. */
    ALLOW_LEGACY_PUBLIC_SALT,
}

/**
 * Rewrites detected identifier spans into de-identified text.
 *
 * The engine accepts original-text offsets and applies replacements from
 * right to left, so each span is replaced at its original offsets even when
 * earlier spans expand or shrink the output.
 *
 * Hashing, meaning the [DeidentifyMethod.HASH] replacement and each applied
 * span's `textHash`, is keyed through an [OpenMedSecretProvider] such as
 * `AndroidKeystoreSecretProvider`, so the key never sits in app memory. An
 * engine built without a provider omits span digests and refuses
 * [DeidentifyMethod.HASH] with [OpenMedSecretException]
 * ([OpenMedSecretFailure.MISSING]); it never falls back to a public key.
 */
class DeidentifyEngine private constructor(
    private val hashKeys: HashKeySource?,
) {
    /** Keyless engine: masks, removes, and replaces; refuses hashing. */
    constructor() : this(null)

    /**
     * Engine whose hashing runs inside [secretProvider].
     *
     * Each `deidentify` call resolves the newest key version, so a rotation
     * takes effect on the next call. Provider failures surface as
     * [OpenMedSecretException] before any output is produced.
     *
     * @param secretProvider Custody boundary that holds the hash key.
     * @param hashKeyReference Reference with purpose
     *   [OpenMedSecretPurpose.DEIDENTIFY_HASH].
     * @param secretPolicy Protection the hash key must have.
     * @param hashKeyPolicy [HashKeyPolicy.STRICT] refuses a provider key that
     *   equals the public default salt.
     */
    constructor(
        secretProvider: OpenMedSecretProvider,
        hashKeyReference: OpenMedSecretReference = OpenMedSecretReference.DEIDENTIFY_HASH,
        secretPolicy: OpenMedSecretPolicy = OpenMedSecretPolicy.DEFAULT,
        hashKeyPolicy: HashKeyPolicy = HashKeyPolicy.STRICT,
    ) : this(ProviderHashKeys(secretProvider, hashKeyReference, secretPolicy, hashKeyPolicy))

    /**
     * Engine keyed by raw salt bytes held in app memory.
     *
     * Under [HashKeyPolicy.STRICT] the public [LEGACY_PUBLIC_HASH_SALT] is
     * refused with [OpenMedSecretFailure.PUBLIC_DEFAULT].
     */
    @Deprecated(
        "Raw salts live in app memory. Hold the hash key in an OpenMedSecretProvider.",
        ReplaceWith(
            "DeidentifyEngine(AndroidKeystoreSecretProvider())",
            "com.openmed.openmedkit.secrets.AndroidKeystoreSecretProvider",
        ),
    )
    constructor(
        hashSalt: ByteArray,
        hashKeyPolicy: HashKeyPolicy = HashKeyPolicy.STRICT,
    ) : this(RawSaltHashKeys(hashSalt, hashKeyPolicy))

    fun deidentify(
        text: String,
        spans: List<OpenMedSpan>,
        method: DeidentifyMethod,
    ): DeidentifyResult = deidentify(text, spans) { method }

    fun deidentify(
        text: String,
        spans: List<OpenMedSpan>,
        actionsByLabel: Map<String, DeidentifyMethod>,
        defaultMethod: DeidentifyMethod = DeidentifyMethod.MASK,
    ): DeidentifyResult {
        val normalizedActions = actionsByLabel.mapKeys { normalizeLabel(it.key) }
        return deidentify(text, spans) { span ->
            normalizedActions[normalizeLabel(span.canonicalLabel)] ?: defaultMethod
        }
    }

    fun deidentify(
        text: String,
        spans: List<OpenMedSpan>,
        actionForSpan: (OpenMedSpan) -> DeidentifyMethod,
    ): DeidentifyResult {
        if (spans.isEmpty()) {
            return DeidentifyResult(text, emptyList())
        }

        validateBounds(text, spans)
        val mac = hashKeys?.open()
        val replacementState = ReplacementState()
        val orderedSpans = resolveOverlaps(spans)
        val actions = mutableListOf<DeidentifyAction>()
        var outputDelta = 0

        for (span in orderedSpans) {
            val method = actionForSpan(span)
            val surface = text.substring(span.start, span.end)
            val label = normalizeLabel(span.canonicalLabel)
            val textHash = mac?.let { hmacDigest(it, label, surface) }
            val replacement = replacementFor(method, label, surface, replacementState, textHash)
            val outputStart = span.start + outputDelta
            val outputEnd = outputStart + replacement.length
            val appliedSpan = span.copy(
                canonicalLabel = label,
                textHash = textHash,
                action = method,
                replacement = replacement,
            )

            actions += DeidentifyAction(
                span = appliedSpan,
                method = method,
                replacement = replacement,
                outputStart = outputStart,
                outputEnd = outputEnd,
            )
            outputDelta += replacement.length - (span.end - span.start)
        }

        val redacted = StringBuilder(text)
        for (action in actions.asReversed()) {
            redacted.replace(action.span.start, action.span.end, action.replacement)
        }

        return DeidentifyResult(redacted.toString(), actions)
    }

    private fun validateBounds(text: String, spans: List<OpenMedSpan>) {
        for (span in spans) {
            require(span.end <= text.length) {
                "span end ${span.end} exceeds text length ${text.length}"
            }
            require(span.end > span.start) {
                "span must cover at least one character"
            }
        }
    }

    private fun resolveOverlaps(spans: List<OpenMedSpan>): List<OpenMedSpan> {
        val sorted = spans.sortedWith(
            compareBy<OpenMedSpan> { it.start }
                .thenByDescending { it.end }
                .thenByDescending { it.score ?: -1.0 }
                .thenBy { normalizeLabel(it.canonicalLabel) },
        )
        val resolved = mutableListOf<OpenMedSpan>()
        var current = sorted.first()
        var selected = current

        for (next in sorted.drop(1)) {
            if (next.start < current.end) {
                selected = chooseOverlapLabel(selected, next)
                current = selected.copy(
                    start = min(current.start, next.start),
                    end = max(current.end, next.end),
                    textHash = null,
                    action = null,
                    replacement = null,
                )
            } else {
                resolved += current
                current = next
                selected = next
            }
        }

        resolved += current
        return resolved
    }

    private fun chooseOverlapLabel(left: OpenMedSpan, right: OpenMedSpan): OpenMedSpan {
        val leftScore = left.score ?: -1.0
        val rightScore = right.score ?: -1.0
        if (rightScore != leftScore) {
            return if (rightScore > leftScore) right else left
        }

        val leftLength = left.end - left.start
        val rightLength = right.end - right.start
        if (rightLength != leftLength) {
            return if (rightLength > leftLength) right else left
        }

        return if (right.start < left.start) right else left
    }

    private fun replacementFor(
        method: DeidentifyMethod,
        label: String,
        surface: String,
        state: ReplacementState,
        textHash: String?,
    ): String {
        return when (method) {
            DeidentifyMethod.MASK -> state.tokenFor(
                method = method,
                label = label,
                surface = surface,
            ) { tokenLabel, index -> bracketedToken(tokenLabel, index) }

            DeidentifyMethod.REMOVE -> ""
            DeidentifyMethod.REPLACE -> state.tokenFor(
                method = method,
                label = label,
                surface = surface,
            ) { tokenLabel, index -> surrogateToken(tokenLabel, index) }

            DeidentifyMethod.HASH ->
                textHash ?: throw OpenMedSecretException(OpenMedSecretFailure.MISSING)
        }
    }

    private fun bracketedToken(label: String, index: Int): String {
        return if (index == 1) "[$label]" else "[${label}_$index]"
    }

    private fun surrogateToken(label: String, index: Int): String {
        return if (index == 1) {
            "${label}_SURROGATE"
        } else {
            "${label}_SURROGATE_$index"
        }
    }

    private fun hmacDigest(
        mac: (ByteArray) -> ByteArray,
        label: String,
        surface: String,
    ): String {
        val payload = "$label\u0000$surface".toByteArray(StandardCharsets.UTF_8)
        return "hmac-sha256:${mac(payload).toHex()}"
    }

    private data class ReplacementKey(
        val method: DeidentifyMethod,
        val label: String,
        val surface: String,
    )

    private class ReplacementState {
        private val counters = mutableMapOf<String, Int>()
        private val tokens = mutableMapOf<ReplacementKey, String>()

        fun tokenFor(
            method: DeidentifyMethod,
            label: String,
            surface: String,
            format: (String, Int) -> String,
        ): String {
            val key = ReplacementKey(method, label, surface)
            return tokens.getOrPut(key) {
                val nextIndex = (counters[label] ?: 0) + 1
                counters[label] = nextIndex
                format(label, nextIndex)
            }
        }
    }

    private interface HashKeySource {
        /** Resolve the MAC used for one `deidentify` call. */
        fun open(): (ByteArray) -> ByteArray
    }

    private class RawSaltHashKeys(salt: ByteArray, policy: HashKeyPolicy) : HashKeySource {
        private val key = salt.copyOf()

        init {
            require(key.isNotEmpty()) { "hashSalt must not be empty" }
            if (policy == HashKeyPolicy.STRICT && key.contentEquals(legacySalt())) {
                throw OpenMedSecretException(OpenMedSecretFailure.PUBLIC_DEFAULT)
            }
        }

        override fun open(): (ByteArray) -> ByteArray {
            val mac = Mac.getInstance(HMAC_SHA256)
            mac.init(SecretKeySpec(key, HMAC_SHA256))
            return { payload -> mac.doFinal(payload) }
        }
    }

    private class ProviderHashKeys(
        private val provider: OpenMedSecretProvider,
        private val reference: OpenMedSecretReference,
        private val secretPolicy: OpenMedSecretPolicy,
        private val hashKeyPolicy: HashKeyPolicy,
    ) : HashKeySource {
        init {
            require(reference.purpose == OpenMedSecretPurpose.DEIDENTIFY_HASH) {
                "hashKeyReference must have purpose DEIDENTIFY_HASH"
            }
        }

        override fun open(): (ByteArray) -> ByteArray {
            val handle = provider.get(reference, policy = secretPolicy).getOrThrow()
            if (hashKeyPolicy == HashKeyPolicy.STRICT) {
                val probe = handle.mac(PUBLIC_DEFAULT_PROBE).getOrThrow()
                val legacy = Mac.getInstance(HMAC_SHA256).run {
                    init(SecretKeySpec(legacySalt(), HMAC_SHA256))
                    doFinal(PUBLIC_DEFAULT_PROBE)
                }
                if (MessageDigest.isEqual(probe, legacy)) {
                    throw OpenMedSecretException(OpenMedSecretFailure.PUBLIC_DEFAULT)
                }
            }
            return { payload -> handle.mac(payload).getOrThrow() }
        }
    }

    private fun ByteArray.toHex(): String {
        val chars = CharArray(size * 2)
        forEachIndexed { index, byte ->
            val value = byte.toInt() and 0xff
            chars[index * 2] = HEX_CHARS[value ushr 4]
            chars[index * 2 + 1] = HEX_CHARS[value and 0x0f]
        }
        return String(chars)
    }

    companion object {
        /**
         * Public pre-4.0 default salt. It is not a secret; pass it only with
         * [HashKeyPolicy.ALLOW_LEGACY_PUBLIC_SALT] to reproduce old digests.
         */
        const val LEGACY_PUBLIC_HASH_SALT = "openmed-android-deidentify-v1"
        private const val HMAC_SHA256 = "HmacSHA256"
        private val PUBLIC_DEFAULT_PROBE =
            "openmed:deidentify:public-default-probe:v1".toByteArray(StandardCharsets.UTF_8)

        private fun legacySalt(): ByteArray =
            LEGACY_PUBLIC_HASH_SALT.toByteArray(StandardCharsets.UTF_8)
        private val HEX_CHARS = "0123456789abcdef".toCharArray()

        fun normalizeLabel(label: String): String {
            val normalized = label.trim()
                .uppercase(Locale.US)
                .replace(Regex("[^A-Z0-9]+"), "_")
                .trim('_')
            return normalized.ifEmpty { "PII" }
        }
    }
}
