package com.openmed.openmedkit.receipt

import java.math.BigDecimal
import java.math.BigInteger
import java.math.MathContext
import java.math.RoundingMode
import java.security.MessageDigest
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Outcome of [ReceiptVerifier.verify]. [code] is the reason code shared with
 * the Swift and web verifiers: `valid`, `hash_mismatch`, `signature_mismatch`,
 * `unsupported_algorithm` or `malformed`. A result never carries report content
 * or key material.
 */
sealed class ReceiptVerification(val code: String) {
    /**
     * The `repro_hash` and HMAC-SHA256 signature verified. [spanOrder] is
     * `sorted`, or `stored` for a report signed before deterministic span order.
     */
    class Valid internal constructor(val spanOrder: String) : ReceiptVerification("valid")

    /** The report did not verify; [code] says why. */
    class Rejected internal constructor(code: String) : ReceiptVerification(code)

    override fun toString(): String = code
}

/**
 * Verifies a signed OpenMed `AuditReport` JSON exactly as Python
 * `AuditReport.verify()` does. Kits never sign, and the caller owns the key.
 *
 * The received report is canonicalized like Python `_canonical_json()`: members
 * sorted by Unicode code point, `,` and `:` separators, non-ASCII escaped as
 * lowercase `\uXXXX` with surrogate pairs, no NaN or infinities, and floats in
 * Python's shortest round-trip form (`1e-07`, `5e-324`, `1.0`). The verifier
 * compares the `repro_hash` (SHA-256 of the report without `repro_hash` and
 * `signature`) and the HMAC-SHA256 of the report without `signature`, trying
 * the sorted span order first and then the stored order, and compares the HMAC
 * in constant time. It is stricter than Python on one point: members Python
 * would drop or normalize while loading change the canonical bytes, so such a
 * report is rejected rather than accepted.
 */
object ReceiptVerifier {
    /** The only signature algorithm accepted. */
    const val ALGORITHM: String = "HMAC-SHA256"

    const val HASH_MISMATCH: String = "hash_mismatch"
    const val SIGNATURE_MISMATCH: String = "signature_mismatch"
    const val UNSUPPORTED_ALGORITHM: String = "unsupported_algorithm"
    const val MALFORMED: String = "malformed"

    private val integer = Regex("-?(0|[1-9][0-9]*)")
    private val number = Regex("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")
    private val codePointOrder = Comparator<String> { left, right ->
        var i = 0
        while (i < left.length && i < right.length) {
            val a = left.codePointAt(i)
            val b = right.codePointAt(i)
            if (a != b) return@Comparator a - b
            i += Character.charCount(a)
        }
        left.length - right.length
    }

    /**
     * Verifies [reportJson] against the HMAC [key]. Malformed input returns
     * `malformed` and is never echoed.
     *
     * @throws IllegalArgumentException when [key] is empty.
     */
    fun verify(reportJson: String, key: ByteArray): ReceiptVerification {
        require(key.isNotEmpty()) { "receipt verification requires a non-empty HMAC key" }
        val report: JsonObject
        val candidates: List<Triple<String, ByteArray, ByteArray>>
        try {
            report = Json.parseToJsonElement(reportJson) as JsonObject
            candidates = candidates(report)
        } catch (error: Exception) {
            return ReceiptVerification.Rejected(MALFORMED)
        } catch (error: StackOverflowError) {
            return ReceiptVerification.Rejected(MALFORMED)
        }
        val stored = report.text("repro_hash") ?: return ReceiptVerification.Rejected(MALFORMED)
        // Python's order, shared with the Swift and web verifiers: a missing or
        // null signature first, then the algorithm, then the hash, then the HMAC.
        val signature = report["signature"]
        if (signature == null || signature == JsonNull) return ReceiptVerification.Rejected(SIGNATURE_MISMATCH)
        if (signature !is JsonObject) return ReceiptVerification.Rejected(MALFORMED)
        val value = signature.text("value") ?: return ReceiptVerification.Rejected(MALFORMED)
        if (signature.text("algorithm") != ALGORITHM) return ReceiptVerification.Rejected(UNSUPPORTED_ALGORITHM)
        val expected = value.toByteArray(Charsets.UTF_8)
        var hashMatched = false
        for ((order, repro, signing) in candidates) {
            if (sha256(repro) != stored) continue
            hashMatched = true
            val mac = Mac.getInstance("HmacSHA256")
            mac.init(SecretKeySpec(key, "HmacSHA256"))
            if (MessageDigest.isEqual(hex(mac.doFinal(signing)).toByteArray(Charsets.US_ASCII), expected)) {
                return ReceiptVerification.Valid(order)
            }
        }
        return ReceiptVerification.Rejected(if (hashMatched) SIGNATURE_MISMATCH else HASH_MISMATCH)
    }

    /**
     * Span order, canonical repro payload and canonical signing payload, for
     * the sorted order and then the stored order when it differs.
     *
     * @throws IllegalArgumentException when the report is not a valid audit report.
     */
    internal fun candidates(report: JsonObject): List<Triple<String, ByteArray, ByteArray>> {
        val spans = report["spans"] as JsonArray
        val keyed = ArrayList<Pair<JsonObject, String>>(spans.size)
        for (span in spans) {
            span as JsonObject
            requireInteger(span, "start")
            requireInteger(span, "end")
            requireNotNull(span.text("canonical_label"))
            requireNotNull(span.text("action"))
            keyed += span to canonicalJson(span)
        }
        val sorted = keyed.sortedWith { a, b ->
            var result = compareIntegers(a.first, b.first, "start")
            if (result == 0) result = compareIntegers(a.first, b.first, "end")
            if (result == 0) result = codePointOrder.compare(a.first.text("canonical_label")!!, b.first.text("canonical_label")!!)
            if (result == 0) result = codePointOrder.compare(a.first.text("action")!!, b.first.text("action")!!)
            if (result == 0) result = codePointOrder.compare(a.second, b.second)
            result
        }.map { it.first }
        val result = ArrayList<Triple<String, ByteArray, ByteArray>>(2)
        for (order in if (sorted == spans) listOf("sorted") else listOf("sorted", "stored")) {
            val body = LinkedHashMap(report)
            body.remove("signature")
            body["spans"] = if (order == "sorted") JsonArray(sorted) else spans
            val signing = canonicalJson(JsonObject(body)).toByteArray(Charsets.US_ASCII)
            body.remove("repro_hash")
            result += Triple(order, canonicalJson(JsonObject(body)).toByteArray(Charsets.US_ASCII), signing)
        }
        return result
    }

    /** `sha256:` plus the hex SHA-256 of [bytes], as Python writes `repro_hash`. */
    internal fun sha256(bytes: ByteArray): String = "sha256:" + hex(MessageDigest.getInstance("SHA-256").digest(bytes))

    /**
     * Python `json.dumps(sort_keys=True, ensure_ascii=True, separators=(",", ":"),
     * allow_nan=False)` of [element].
     */
    internal fun canonicalJson(element: JsonElement): String = StringBuilder().also { append(it, element) }.toString()

    /** Python `repr()` of a finite double: the shortest digits that round-trip. */
    internal fun pythonFloatRepr(value: Double): String {
        require(!value.isNaN() && !value.isInfinite())
        if (value == 0.0) return if (1.0 / value < 0) "-0.0" else "0.0"
        val magnitude = Math.abs(value)
        val exact = BigDecimal(magnitude)
        var best: BigDecimal? = null
        var precision = 1
        while (best == null) {
            // The nearest digits usually round-trip; at a power-of-two boundary
            // only the other neighbour may, so try both and keep the closest.
            for (mode in arrayOf(RoundingMode.HALF_EVEN, RoundingMode.FLOOR, RoundingMode.CEILING)) {
                val candidate = exact.round(MathContext(precision, mode))
                if (java.lang.Double.parseDouble(candidate.toString()) == magnitude &&
                    (best == null || candidate.subtract(exact).abs() < best.subtract(exact).abs())
                ) {
                    best = candidate
                }
            }
            precision += 1
        }
        val shortest = best.stripTrailingZeros()
        val digits = shortest.unscaledValue().toString()
        // The value is 0.<digits> x 10^point, as in Python's repr.
        val point = digits.length - shortest.scale()
        val text = when {
            point <= -4 || point > 16 -> {
                val exponent = point - 1
                (if (digits.length > 1) digits[0] + "." + digits.substring(1) else digits) +
                    (if (exponent < 0) "e-" else "e+") + Math.abs(exponent).toString().padStart(2, '0')
            }
            point <= 0 -> "0." + "0".repeat(-point) + digits
            point >= digits.length -> digits + "0".repeat(point - digits.length) + ".0"
            else -> digits.substring(0, point) + "." + digits.substring(point)
        }
        return if (value < 0) "-$text" else text
    }

    private fun append(out: StringBuilder, element: JsonElement) {
        when (element) {
            is JsonObject -> {
                out.append('{')
                var first = true
                for (key in element.keys.sortedWith(codePointOrder)) {
                    if (!first) out.append(',')
                    first = false
                    quote(out, key)
                    out.append(':')
                    append(out, element.getValue(key))
                }
                out.append('}')
            }
            is JsonArray -> {
                out.append('[')
                for (index in element.indices) {
                    if (index > 0) out.append(',')
                    append(out, element[index])
                }
                out.append(']')
            }
            is JsonPrimitive -> {
                val token = element.content
                when {
                    element.isString -> quote(out, token)
                    token == "true" || token == "false" || token == "null" -> out.append(token)
                    integer.matches(token) -> out.append(BigInteger(token))
                    number.matches(token) -> out.append(pythonFloatRepr(token.toDouble()))
                    else -> throw IllegalArgumentException()
                }
            }
        }
    }

    private fun quote(out: StringBuilder, value: String) {
        out.append('"')
        for (character in value) {
            when (character) {
                '"' -> out.append("\\\"")
                '\\' -> out.append("\\\\")
                '\n' -> out.append("\\n")
                '\r' -> out.append("\\r")
                '\t' -> out.append("\\t")
                '\b' -> out.append("\\b")
                '\u000C' -> out.append("\\f")
                in ' '..'~' -> out.append(character)
                else -> out.append("\\u").append(character.code.toString(16).padStart(4, '0'))
            }
        }
        out.append('"')
    }

    private fun requireInteger(span: JsonObject, key: String) {
        val value = span[key] as JsonPrimitive
        require(!value.isString && integer.matches(value.content))
    }

    private fun compareIntegers(left: JsonObject, right: JsonObject, key: String): Int =
        BigInteger((left[key] as JsonPrimitive).content).compareTo(BigInteger((right[key] as JsonPrimitive).content))

    private fun JsonObject.text(key: String): String? {
        val value = this[key] as? JsonPrimitive ?: return null
        return if (value.isString) value.content else null
    }

    private fun hex(bytes: ByteArray): String {
        val out = StringBuilder(bytes.size * 2)
        for (byte in bytes) {
            out.append(Character.forDigit((byte.toInt() shr 4) and 0xf, 16))
            out.append(Character.forDigit(byte.toInt() and 0xf, 16))
        }
        return out.toString()
    }
}
