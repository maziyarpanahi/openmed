package com.openmed.openmedkit.multimodal

import java.math.BigDecimal
import java.math.BigInteger
import java.math.MathContext
import java.math.RoundingMode
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** A content-free error: rejected input and parser causes are never retained. */
class MultimodalContractException internal constructor() :
    IllegalArgumentException("multimodal contract is invalid")

internal fun requireContract(condition: Boolean) {
    if (!condition) throw MultimodalContractException()
}

/** Strict, bounded JSON shared by the experimental metadata contracts. */
internal object ContractJson {
    private const val MAX_BYTES = 64 * 1024
    private val integer = Regex("-?(0|[1-9][0-9]*)")
    private val number = Regex("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")

    fun parse(payload: String): JsonObject {
        requireContract(payload.length <= MAX_BYTES)
        requireContract(payload.toByteArray(Charsets.UTF_8).size <= MAX_BYTES)
        try {
            return Reader(payload).read() as? JsonObject ?: throw MultimodalContractException()
        } catch (_: Exception) {
            throw MultimodalContractException()
        }
    }

    fun fields(value: JsonObject, required: Set<String>, optional: Set<String> = emptySet()) {
        requireContract(value.keys.containsAll(required) && (value.keys - required - optional).isEmpty())
    }

    fun string(value: JsonElement?): String {
        requireContract(value is JsonPrimitive && value.isString)
        return (value as JsonPrimitive).content
    }

    fun int(value: JsonElement?, maximum: BigInteger, minimum: BigInteger = BigInteger.ZERO): BigInteger {
        requireContract(value is JsonPrimitive && !value.isString && integer.matches(value.content))
        val result = (value as JsonPrimitive).content.toBigInteger()
        requireContract(result >= minimum && result <= maximum)
        return result
    }

    fun numeric(value: JsonElement?, maximum: BigInteger, positive: Boolean = false): JsonPrimitive {
        requireContract(value is JsonPrimitive && !value.isString && number.matches(value.content))
        val raw = (value as JsonPrimitive).content
        if (integer.matches(raw)) {
            return JsonPrimitive(int(value, maximum, if (positive) BigInteger.ONE else BigInteger.ZERO))
        }
        val result = raw.toDoubleOrNull() ?: throw MultimodalContractException()
        requireContract(result.isFinite() && result >= 0 && (!positive || result > 0))
        requireContract(BigDecimal(result) <= BigDecimal(maximum))
        return numberPrimitive(pythonFloat(result))
    }

    fun float(value: JsonElement?, maximum: Double): JsonPrimitive {
        requireContract(value is JsonPrimitive && !value.isString && number.matches(value.content))
        val result = (value as JsonPrimitive).content.toDoubleOrNull() ?: throw MultimodalContractException()
        requireContract(result.isFinite() && result >= 0 && result <= maximum)
        return numberPrimitive(pythonFloat(result))
    }

    // These primitives must be unquoted JSON numbers, rather than strings.
    private fun numberPrimitive(numberText: String): JsonPrimitive = Json.parseToJsonElement(numberText) as JsonPrimitive

    fun encode(value: JsonObject, sorted: Boolean = false): String {
        fun render(item: JsonElement): String = when (item) {
            is JsonObject -> (if (sorted) item.toSortedMap() else item).entries.joinToString(",", "{", "}") {
                JsonPrimitive(it.key).toString() + ":" + render(it.value)
            }
            is JsonArray -> item.joinToString(",", "[", "]") { render(it) }
            else -> item.toString()
        }
        return render(value)
    }

    /** Python's shortest-roundtrip binary64 spelling and exponent conventions. */
    private fun pythonFloat(value: Double): String {
        if (value == 0.0) return if (value.toRawBits() < 0) "-0.0" else "0.0"
        val exact = BigDecimal(value)
        val shortest = (1..17).asSequence().map {
            exact.round(MathContext(it, RoundingMode.HALF_EVEN)).stripTrailingZeros()
        }.first { it.toDouble().toRawBits() == value.toRawBits() }
        val exponent = shortest.precision() - shortest.scale() - 1
        if (exponent in -4..15) {
            val plain = shortest.toPlainString()
            return if ('.' in plain) plain else "$plain.0"
        }
        val digits = shortest.unscaledValue().abs().toString()
        val mantissa = digits.take(1) + if (digits.length > 1) "." + digits.drop(1) else ""
        val sign = if (value < 0) "-" else ""
        val exponentSign = if (exponent >= 0) "+" else "-"
        return "$sign${mantissa}e$exponentSign${kotlin.math.abs(exponent).toString().padStart(2, '0')}"
    }

    private class Reader(private val text: String) {
        private var position = 0
        fun read(): JsonElement {
            val result = value(0)
            whitespace()
            requireContract(position == text.length)
            return result
        }
        private fun whitespace() {
            while (position < text.length && text[position] in " \t\r\n") position++
        }
        private fun consume(char: Char): Boolean {
            whitespace()
            if (position < text.length && text[position] == char) { position++; return true }
            return false
        }
        private fun value(depth: Int): JsonElement {
            requireContract(depth <= 16)
            whitespace()
            requireContract(position < text.length)
            if (text[position] == '"') return quoted()
            if (consume('{')) {
                val fields = linkedMapOf<String, JsonElement>()
                if (!consume('}')) {
                    do {
                        whitespace()
                        requireContract(position < text.length && text[position] == '"')
                        val key = quoted().content
                        requireContract(key !in fields && consume(':'))
                        fields[key] = value(depth + 1)
                    } while (consume(','))
                    requireContract(consume('}'))
                }
                return JsonObject(fields)
            }
            if (consume('[')) {
                val values = mutableListOf<JsonElement>()
                if (!consume(']')) {
                    do { values += value(depth + 1) } while (consume(','))
                    requireContract(consume(']'))
                }
                return JsonArray(values)
            }
            val start = position
            while (position < text.length && text[position] !in ",]} \t\r\n") position++
            val token = text.substring(start, position)
            requireContract(token in setOf("true", "false", "null") || number.matches(token))
            return Json.parseToJsonElement(token)
        }
        private fun quoted(): JsonPrimitive {
            val start = position++
            while (position < text.length) {
                val char = text[position++]
                requireContract(char.code >= 32)
                if (char == '\\') {
                    requireContract(position < text.length)
                    position++
                } else if (char == '"') {
                    return Json.parseToJsonElement(text.substring(start, position)) as JsonPrimitive
                }
            }
            throw MultimodalContractException()
        }
    }
}

internal val COUNT_MAX: BigInteger = BigInteger.valueOf(Int.MAX_VALUE.toLong())
internal val BYTE_MAX: BigInteger = BigInteger.valueOf(Long.MAX_VALUE)
internal val JSON_INTEGER_MAX: BigInteger = BigInteger.TEN.pow(65536)
internal val PIXEL_MAX: BigInteger = COUNT_MAX.pow(3)
internal val DIGEST = Regex("[0-9a-f]{64}")
internal fun absent(value: JsonElement?): Boolean = value == null || value == JsonNull
internal fun opaque(value: String, pattern: Regex) = requireContract(pattern.matches(value))
internal fun obj(vararg fields: Pair<String, JsonElement>): JsonObject = JsonObject(linkedMapOf(*fields))
internal fun text(value: String?): JsonElement = value?.let { JsonPrimitive(it) } ?: JsonNull
