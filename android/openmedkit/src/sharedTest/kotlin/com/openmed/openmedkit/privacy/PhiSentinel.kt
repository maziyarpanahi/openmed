package com.openmed.openmedkit.privacy

import java.security.SecureRandom
import java.util.Base64
import java.util.Random

/** Value-free refusal codes raised while configuring a sentinel sweep. */
enum class SweepRefusal(val code: String) {
    SENTINEL_IDENTIFIER_INVALID("sentinel_identifier_invalid"),
    SENTINEL_LENGTH_OUT_OF_RANGE("sentinel_length_out_of_range"),
    SENTINEL_IDENTIFIER_DUPLICATE("sentinel_identifier_duplicate"),
    SENTINELS_EMPTY("sentinels_empty"),
    SENTINELS_TOO_MANY("sentinels_too_many"),
    BOUNDS_INVALID("sweep_bounds_invalid"),
}

/** Configuration refusal whose message is only the value-free [SweepRefusal.code]. */
class PhiSentinelSweepException(val refusal: SweepRefusal) : IllegalArgumentException(refusal.code)

/**
 * A synthetic value planted in a workload so app storage and logs can be swept for it.
 *
 * Only [identifier] is reported. [plantedValue] is for feeding the workload under
 * test and never appears in [toString], exceptions or sweep reports.
 */
class PhiSentinel(val identifier: String, val plantedValue: String) {
    init {
        if (!isValidIdentifier(identifier)) {
            throw PhiSentinelSweepException(SweepRefusal.SENTINEL_IDENTIFIER_INVALID)
        }
        val length = plantedValue.toByteArray(Charsets.UTF_8).size
        if (length < MINIMUM_UTF8_LENGTH || length > MAXIMUM_UTF8_LENGTH) {
            throw PhiSentinelSweepException(SweepRefusal.SENTINEL_LENGTH_OUT_OF_RANGE)
        }
    }

    override fun toString(): String = "PhiSentinel($identifier)"

    override fun equals(other: Any?): Boolean =
        other is PhiSentinel && other.identifier == identifier && other.plantedValue == plantedValue

    override fun hashCode(): Int = 31 * identifier.hashCode() + plantedValue.hashCode()

    companion object {
        /** Shortest accepted sentinel in UTF-8 bytes, to keep accidental matches rare. */
        const val MINIMUM_UTF8_LENGTH = 8

        /** Longest accepted sentinel in UTF-8 bytes. */
        const val MAXIMUM_UTF8_LENGTH = 256

        /** Most sentinels one sweep accepts. */
        const val MAXIMUM_COUNT = 64

        private const val ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        private val IDENTIFIER_PATTERN = Regex("[a-z0-9][a-z0-9._-]{0,63}")

        /** Create a random synthetic sentinel that cannot collide with other runs. */
        fun synthetic(identifier: String, random: Random = SecureRandom()): PhiSentinel {
            val suffix = (0 until 13).map { ALPHABET[random.nextInt(ALPHABET.length)] }
            return PhiSentinel(identifier, "OMS" + suffix.joinToString(""))
        }

        internal fun isValidIdentifier(identifier: String): Boolean =
            IDENTIFIER_PATTERN.matches(identifier)
    }
}

/** The byte encoding under which a sentinel was found. */
enum class SentinelEncoding(val wireName: String) {
    /** The exact UTF-8 bytes of the planted value. */
    UTF8("utf8"),
    UTF16LE("utf16le"),
    UTF16BE("utf16be"),

    /** Standard base64 at each of the three byte alignments. */
    BASE64("base64"),

    /** URL-safe base64 (`-` and `_`) at each byte alignment. */
    BASE64URL("base64url"),

    /** RFC 3986 percent-encoding (upper- or lowercase hex, `%20` or `+` for space). */
    PERCENT("percent"),
}

/** One byte pattern derived from a sentinel under one encoding. */
internal class SentinelNeedle(
    val sentinelIndex: Int,
    val encoding: SentinelEncoding,
    val bytes: ByteArray,
)

internal object SentinelNeedles {
    /** Derive every distinct byte pattern for a planted value, in encoding order. */
    fun make(value: String, sentinelIndex: Int): List<SentinelNeedle> {
        val raw = value.toByteArray(Charsets.UTF_8)
        val base64 = alignedBase64(raw)
        val candidates = buildList {
            add(SentinelEncoding.UTF8 to raw)
            add(SentinelEncoding.UTF16LE to value.toByteArray(Charsets.UTF_16LE))
            add(SentinelEncoding.UTF16BE to value.toByteArray(Charsets.UTF_16BE))
            base64.forEach { add(SentinelEncoding.BASE64 to it) }
            base64.forEach { add(SentinelEncoding.BASE64URL to urlSafe(it)) }
            add(SentinelEncoding.PERCENT to percentEncoded(raw, uppercase = true, spaceAsPlus = false))
            add(SentinelEncoding.PERCENT to percentEncoded(raw, uppercase = false, spaceAsPlus = false))
            add(SentinelEncoding.PERCENT to percentEncoded(raw, uppercase = true, spaceAsPlus = true))
        }
        val seen = mutableSetOf<String>()
        return candidates
            .filter { (_, bytes) -> bytes.isNotEmpty() && seen.add(bytes.toHexKey()) }
            .map { (encoding, bytes) -> SentinelNeedle(sentinelIndex, encoding, bytes) }
    }

    /** Base64 characters fully determined by the value at byte offsets 0, 1 and 2. */
    fun alignedBase64(raw: ByteArray): List<ByteArray> = (0 until 3).map { offset ->
        val padded = ByteArray(offset) + raw + ByteArray(3)
        val encoded = Base64.getEncoder().encode(padded)
        val first = (8 * offset + 5) / 6
        val end = (8 * (offset + raw.size)) / 6
        encoded.copyOfRange(first, end)
    }

    private fun urlSafe(base64: ByteArray): ByteArray = ByteArray(base64.size) { index ->
        when (base64[index]) {
            '+'.code.toByte() -> '-'.code.toByte()
            '/'.code.toByte() -> '_'.code.toByte()
            else -> base64[index]
        }
    }

    private fun percentEncoded(raw: ByteArray, uppercase: Boolean, spaceAsPlus: Boolean): ByteArray {
        val digits = if (uppercase) "0123456789ABCDEF" else "0123456789abcdef"
        val output = StringBuilder()
        for (byte in raw) {
            val value = byte.toInt() and 0xff
            val char = value.toChar()
            when {
                char in 'A'..'Z' || char in 'a'..'z' || char in '0'..'9' || char in "-._~" ->
                    output.append(char)
                spaceAsPlus && value == 0x20 -> output.append('+')
                else -> output.append('%').append(digits[value shr 4]).append(digits[value and 0x0f])
            }
        }
        return output.toString().toByteArray(Charsets.US_ASCII)
    }

    private fun ByteArray.toHexKey(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }
}

/** Naive byte search; sweep inputs are bounded, so simplicity beats speed here. */
internal fun ByteArray.containsBytes(needle: ByteArray, length: Int = size): Boolean {
    if (needle.isEmpty() || needle.size > length) {
        return false
    }
    val first = needle[0]
    val last = length - needle.size
    var index = 0
    while (index <= last) {
        if (this[index] == first) {
            var matched = 1
            while (matched < needle.size && this[index + matched] == needle[matched]) {
                matched += 1
            }
            if (matched == needle.size) {
                return true
            }
        }
        index += 1
    }
    return false
}
