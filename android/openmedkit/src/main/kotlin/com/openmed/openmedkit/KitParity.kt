package com.openmed.openmedkit

import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction

/**
 * A value-free kit-parity profile violation.
 *
 * Carries a stable reason [code] and a JSON [fieldPath] only. Unknown keys are
 * reported at the enclosing object path, so a key name is never echoed.
 */
class KitParityException(
    val code: String,
    val fieldPath: String = "$",
) : IllegalArgumentException("$code at $fieldPath")

/**
 * One applied span in a kit-parity result. Offsets are half-open Unicode
 * scalar offsets into the source text.
 */
data class KitParitySpan(
    val label: String,
    val canonicalLabel: String,
    val action: String,
    val start: Int,
    val end: Int,
    val confidencePpm: Int? = null,
    val replacement: String? = null,
)

/** A kit-parity projection of one de-identification result. */
data class KitParityResult(
    val deidentifiedText: String,
    val spans: List<KitParitySpan>,
)

/**
 * OpenMedKit kit-parity result profile (`openmed.kit_parity_result` v1).
 *
 * Every OpenMedKit surface projects its native de-identification result into
 * one value-free record and serializes it to the same canonical JSON bytes.
 * The shared golden vectors live in
 * `tests/fixtures/parity/kit_parity_vectors.json`.
 *
 * Offsets are Unicode scalar offsets; UTF-8 and UTF-16 offsets are derived
 * through the conversion helpers. `confidencePpm` is
 * `floor(confidence.toDouble() * 1_000_000 + 0.5)` and is omitted when a
 * surface has no score. `replacement` is omitted for `keep`. Canonical JSON
 * uses sorted keys, no whitespace, integers only, and RFC 8785 string
 * escaping.
 */
object KitParity {
    const val PROFILE: String = "openmed.kit_parity_result"
    const val PROFILE_VERSION: Int = 1
    const val OFFSET_UNIT: String = "unicode_scalar"
    const val CONFIDENCE_SCALE: Int = 1_000_000

    /** Offset-contract versions whose scalar and grapheme semantics are accepted. */
    val OFFSET_CONTRACT_VERSIONS: Set<Int> = setOf(1)

    /** Span decoder contract versions whose output is accepted. */
    val DECODER_CONTRACT_VERSIONS: Set<Int> = setOf(1)

    val ACTIONS: List<String> = listOf("hash", "keep", "mask", "redact", "remove", "replace")

    val CANONICAL_LABELS: List<String> = listOf(
        "ACCOUNT_NUMBER", "AGE", "AMOUNT", "ANTIBIOTIC", "API_KEY", "BIC",
        "BITCOIN_ADDRESS", "BODY_SITE", "BUILDING_NUMBER", "CONDITION", "CREDIT_CARD",
        "CREDIT_CARD_ISSUER", "CURRENCY", "CVV", "DATE", "DATE_OF_BIRTH", "DIET_TYPE",
        "EMAIL", "ETHEREUM_ADDRESS", "EYE_COLOR", "FEEDING_ROUTE", "FIRST_NAME", "GENDER",
        "GPS_COORDINATES", "HEIGHT", "IBAN", "ID_NUM", "IMEI", "IP_ADDRESS",
        "JOB_DEPARTMENT", "JOB_TITLE", "LAB_TEST", "LAST_NAME", "LITECOIN_ADDRESS",
        "LOCATION", "MAC_ADDRESS", "MASKED_NUMBER", "MEDICATION", "MICROORGANISM",
        "MIDDLE_NAME", "NUTRITIONAL_STATUS", "NUTRITION_TARGET", "OCCUPATION",
        "ORDINAL_DIRECTION", "ORGANIZATION", "OTHER", "PASSWORD", "PERSON", "PHONE", "PIN",
        "PREFIX", "PROCEDURE", "SSN", "STREET_ADDRESS", "SUSCEPTIBILITY", "TIME", "URL",
        "USERNAME", "USER_AGENT", "VEHICLE_REGISTRATION", "VIN", "ZIPCODE",
    )

    private val actionSet = ACTIONS.toSet()
    private val labelSet = CANONICAL_LABELS.toSet()
    private val labelPattern = Regex("^[A-Za-z][A-Za-z0-9_:-]{0,79}$")
    private val topLevelKeys = setOf(
        "deidentified_text",
        "offset_unit",
        "profile",
        "profile_version",
        "spans",
    )
    private val spanKeys = setOf(
        "action",
        "canonical_label",
        "confidence_ppm",
        "end",
        "label",
        "replacement",
        "start",
    )

    /** Convert a Unicode scalar offset into a JVM UTF-16 index. */
    @JvmStatic
    fun scalarToUtf16Offset(text: String, scalarOffset: Int): Int {
        checkScalarOffset(text, scalarOffset)
        return UnicodeOffsetContract.scalarToUtf16Index(text, scalarOffset)
    }

    /** Convert a Unicode scalar offset into a UTF-8 byte offset. */
    @JvmStatic
    fun scalarToUtf8Offset(text: String, scalarOffset: Int): Int {
        checkScalarOffset(text, scalarOffset)
        return UnicodeOffsetContract.scalarToUtf8Offset(text, scalarOffset)
    }

    /**
     * Convert a JVM UTF-16 index into a Unicode scalar offset. An index between
     * the halves of a surrogate pair throws `offset_not_scalar_boundary`.
     */
    @JvmStatic
    fun utf16ToScalarOffset(text: String, utf16Offset: Int): Int {
        if (utf16Offset !in 0..text.length) {
            throw KitParityException("offset_out_of_bounds", "$.offset")
        }
        if (!UnicodeOffsetContract.isCodePointBoundary(text, utf16Offset)) {
            throw KitParityException("offset_not_scalar_boundary", "$.offset")
        }
        return UnicodeOffsetContract.utf16ToScalarOffset(text, utf16Offset)
    }

    /**
     * Convert a UTF-8 byte offset into a Unicode scalar offset. An offset
     * inside a multi-byte sequence throws `offset_not_scalar_boundary`.
     */
    @JvmStatic
    fun utf8ToScalarOffset(text: String, utf8Offset: Int): Int {
        val byteLength = UnicodeOffsetContract.scalarToUtf8Offset(
            text,
            UnicodeOffsetContract.scalarLength(text),
        )
        if (utf8Offset !in 0..byteLength) {
            throw KitParityException("offset_out_of_bounds", "$.offset")
        }
        return try {
            UnicodeOffsetContract.utf8ToScalarOffset(text, utf8Offset)
        } catch (_: IllegalArgumentException) {
            throw KitParityException("offset_not_scalar_boundary", "$.offset")
        }
    }

    /** Encode a confidence in `[0, 1]` as integer parts per million. */
    @JvmStatic
    fun confidencePpm(confidence: Float): Int {
        val value = confidence.toDouble()
        if (!value.isFinite() || value < 0.0 || value > 1.0) {
            throw KitParityException("confidence_invalid", "$.confidence")
        }
        return Math.floor(value * CONFIDENCE_SCALE + 0.5).toInt()
    }

    /**
     * Reject bundle contracts this profile cannot interpret. Call this before
     * consuming any result produced under a model bundle.
     */
    @JvmStatic
    fun checkCompatibility(offsetContractVersion: Int, decoderContractVersion: Int) {
        if (offsetContractVersion !in OFFSET_CONTRACT_VERSIONS) {
            throw KitParityException(
                "offset_contract_version_unsupported",
                "$.offset_contract_version",
            )
        }
        if (decoderContractVersion !in DECODER_CONTRACT_VERSIONS) {
            throw KitParityException(
                "decoder_contract_version_unsupported",
                "$.decoder_contract_version",
            )
        }
    }

    /** Gate a bundle's contracts, then parse a result produced under it. */
    @JvmStatic
    fun accept(
        json: ByteArray,
        offsetContractVersion: Int,
        decoderContractVersion: Int,
    ): KitParityResult {
        checkCompatibility(offsetContractVersion, decoderContractVersion)
        return parse(json)
    }

    /** Validate a result against the profile vocabulary and ordering rules. */
    @JvmStatic
    fun validate(result: KitParityResult) {
        var previous: KitParitySpan? = null
        result.spans.forEachIndexed { index, span ->
            val path = "$.spans[$index]"
            checkSpan(span, path)
            checkOrder(span, previous, path)
            previous = span
        }
    }

    /** Return the canonical JSON text; encode it as UTF-8 for the bytes. */
    @JvmStatic
    fun serialize(result: KitParityResult): String {
        validate(result)
        val json = StringBuilder()
        json.append("{\"deidentified_text\":").appendQuoted(result.deidentifiedText)
        json.append(",\"offset_unit\":\"").append(OFFSET_UNIT).append('"')
        json.append(",\"profile\":\"").append(PROFILE).append('"')
        json.append(",\"profile_version\":").append(PROFILE_VERSION)
        json.append(",\"spans\":[")
        result.spans.forEachIndexed { index, span ->
            if (index > 0) {
                json.append(',')
            }
            json.append("{\"action\":").appendQuoted(span.action)
            json.append(",\"canonical_label\":").appendQuoted(span.canonicalLabel)
            span.confidencePpm?.let { json.append(",\"confidence_ppm\":").append(it) }
            json.append(",\"end\":").append(span.end)
            json.append(",\"label\":").appendQuoted(span.label)
            span.replacement?.let { json.append(",\"replacement\":").appendQuoted(it) }
            json.append(",\"start\":").append(span.start).append('}')
        }
        return json.append("]}").toString()
    }

    /**
     * Parse and strictly validate canonical kit-parity result JSON.
     *
     * Checks run in the order shared by every surface: JSON syntax, object
     * shape, profile, profile version, unknown top-level keys, offset unit,
     * de-identified text, each span, span ordering, then the canonical bytes.
     */
    @JvmStatic
    fun parse(json: ByteArray): KitParityResult {
        val text = try {
            Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(json))
                .toString()
        } catch (_: CharacterCodingException) {
            throw KitParityException("json_invalid")
        }
        val document = StrictJsonParser(text).parseDocument()
        if (document !is JsonValue.Obj) {
            throw KitParityException("result_not_object")
        }
        val fields = document.fields
        if (requiredString(fields, "profile", "$") != PROFILE) {
            throw KitParityException("profile_unknown", "$.profile")
        }
        if (requiredInteger(fields, "profile_version", "$") != PROFILE_VERSION) {
            throw KitParityException("profile_version_unsupported", "$.profile_version")
        }
        if (!topLevelKeys.containsAll(fields.keys)) {
            throw KitParityException("field_unknown", "$")
        }
        if (requiredString(fields, "offset_unit", "$") != OFFSET_UNIT) {
            throw KitParityException("offset_unit_unsupported", "$.offset_unit")
        }
        val deidentifiedText = requiredString(fields, "deidentified_text", "$")
        val rawSpans = requiredArray(fields, "spans", "$")

        val spans = mutableListOf<KitParitySpan>()
        rawSpans.forEachIndexed { index, raw ->
            val path = "$.spans[$index]"
            val span = parseSpan(raw, path)
            checkOrder(span, spans.lastOrNull(), path)
            spans += span
        }

        val result = KitParityResult(deidentifiedText, spans)
        if (!serialize(result).toByteArray(Charsets.UTF_8).contentEquals(json)) {
            throw KitParityException("result_not_canonical")
        }
        return result
    }

    /** Parse canonical kit-parity result JSON text. */
    @JvmStatic
    fun parse(json: String): KitParityResult = parse(json.toByteArray(Charsets.UTF_8))

    /** Apply span replacements right to left at Unicode scalar offsets. */
    @JvmStatic
    fun applySpans(source: String, spans: List<KitParitySpan>): String {
        // Ordered, non-empty, non-overlapping spans keep every range in bounds.
        validate(KitParityResult("", spans))
        val length = UnicodeOffsetContract.scalarLength(source)
        spans.forEachIndexed { index, span ->
            if (span.end > length) {
                throw KitParityException("offset_out_of_bounds", "$.spans[$index].end")
            }
        }
        var result = source
        for (span in spans.asReversed()) {
            val replacement = span.replacement ?: continue
            result = UnicodeOffsetContract.replaceScalarSpan(result, span.start, span.end, replacement)
        }
        return result
    }

    /** Check that a result's offsets and UTF-8 bytes agree with its source. */
    @JvmStatic
    fun verify(result: KitParityResult, source: String) {
        validate(result)
        val expected = applySpans(source, result.spans).toByteArray(Charsets.UTF_8)
        if (!expected.contentEquals(result.deidentifiedText.toByteArray(Charsets.UTF_8))) {
            throw KitParityException("deidentified_text_mismatch", "$.deidentified_text")
        }
    }

    private fun checkScalarOffset(text: String, scalarOffset: Int) {
        if (scalarOffset !in 0..UnicodeOffsetContract.scalarLength(text)) {
            throw KitParityException("offset_out_of_bounds", "$.offset")
        }
    }

    private fun parseSpan(raw: JsonValue, path: String): KitParitySpan {
        if (raw !is JsonValue.Obj) {
            throw KitParityException("field_type_invalid", path)
        }
        val fields = raw.fields
        if (!spanKeys.containsAll(fields.keys)) {
            throw KitParityException("field_unknown", path)
        }
        val label = requiredString(fields, "label", path)
        checkLabel(label, path)
        val canonicalLabel = requiredString(fields, "canonical_label", path)
        checkCanonicalLabel(canonicalLabel, path)
        val action = requiredString(fields, "action", path)
        checkAction(action, path)
        val start = requiredInteger(fields, "start", path)
        checkStart(start, path)
        val end = requiredInteger(fields, "end", path)
        checkEnd(start, end, path)
        val confidence = optionalInteger(fields, "confidence_ppm", path)
        checkConfidence(confidence, path)
        val replacement = optionalString(fields, "replacement", path)
        checkReplacement(action, replacement, path)
        return KitParitySpan(label, canonicalLabel, action, start, end, confidence, replacement)
    }

    private fun checkSpan(span: KitParitySpan, path: String) {
        checkLabel(span.label, path)
        checkCanonicalLabel(span.canonicalLabel, path)
        checkAction(span.action, path)
        checkStart(span.start, path)
        checkEnd(span.start, span.end, path)
        checkConfidence(span.confidencePpm, path)
        checkReplacement(span.action, span.replacement, path)
    }

    private fun checkLabel(label: String, path: String) {
        if (!labelPattern.matches(label)) {
            throw KitParityException("label_invalid", "$path.label")
        }
    }

    private fun checkCanonicalLabel(label: String, path: String) {
        if (label !in labelSet) {
            throw KitParityException("label_unknown", "$path.canonical_label")
        }
    }

    private fun checkAction(action: String, path: String) {
        if (action !in actionSet) {
            throw KitParityException("action_unknown", "$path.action")
        }
    }

    private fun checkStart(start: Int, path: String) {
        if (start < 0) {
            throw KitParityException("offset_invalid", "$path.start")
        }
    }

    private fun checkEnd(start: Int, end: Int, path: String) {
        if (end <= start) {
            throw KitParityException("offset_invalid", "$path.end")
        }
    }

    private fun checkConfidence(confidence: Int?, path: String) {
        if (confidence != null && confidence !in 0..CONFIDENCE_SCALE) {
            throw KitParityException("confidence_invalid", "$path.confidence_ppm")
        }
    }

    private fun checkReplacement(action: String, replacement: String?, path: String) {
        if (action == "keep") {
            if (replacement != null) {
                throw KitParityException("replacement_unexpected", "$path.replacement")
            }
        } else if (replacement == null) {
            throw KitParityException("field_missing", "$path.replacement")
        }
    }

    private fun checkOrder(span: KitParitySpan, previous: KitParitySpan?, path: String) {
        if (previous == null) {
            return
        }
        if (span.start < previous.start || (span.start == previous.start && span.end <= previous.end)) {
            throw KitParityException("span_order_invalid", path)
        }
        if (span.start < previous.end) {
            throw KitParityException("span_overlap", path)
        }
    }

    private fun field(fields: Map<String, JsonValue>, key: String, path: String): JsonValue? {
        val value = fields[key] ?: return null
        if (value is JsonValue.Null) {
            throw KitParityException("field_null", "$path.$key")
        }
        return value
    }

    private fun requiredString(fields: Map<String, JsonValue>, key: String, path: String): String =
        optionalString(fields, key, path) ?: throw KitParityException("field_missing", "$path.$key")

    private fun optionalString(fields: Map<String, JsonValue>, key: String, path: String): String? {
        val value = field(fields, key, path) ?: return null
        return (value as? JsonValue.Str)?.value
            ?: throw KitParityException("field_type_invalid", "$path.$key")
    }

    private fun requiredInteger(fields: Map<String, JsonValue>, key: String, path: String): Int =
        optionalInteger(fields, key, path) ?: throw KitParityException("field_missing", "$path.$key")

    private fun optionalInteger(fields: Map<String, JsonValue>, key: String, path: String): Int? {
        val value = field(fields, key, path) ?: return null
        return (value as? JsonValue.Integer)?.value
            ?: throw KitParityException("field_type_invalid", "$path.$key")
    }

    private fun requiredArray(
        fields: Map<String, JsonValue>,
        key: String,
        path: String,
    ): List<JsonValue> {
        val value = field(fields, key, path)
            ?: throw KitParityException("field_missing", "$path.$key")
        return (value as? JsonValue.Arr)?.items
            ?: throw KitParityException("field_type_invalid", "$path.$key")
    }

    private fun StringBuilder.appendQuoted(value: String): StringBuilder {
        append('"')
        for (character in value) {
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (character.code < 0x20) {
                    append("\\u").append(character.code.toString(16).padStart(4, '0'))
                } else {
                    append(character)
                }
            }
        }
        return append('"')
    }
}

/** Project this policy result into the kit-parity result profile. */
fun PolicyDeidentificationResult.toKitParityResult(): KitParityResult {
    val ordered = actions.sortedWith(compareBy<DeidentifiedSpanAction> { it.start }.thenBy { it.end })
    val spans = ordered.mapIndexed { index, record ->
        val confidence = try {
            KitParity.confidencePpm(record.confidence)
        } catch (_: KitParityException) {
            throw KitParityException("confidence_invalid", "$.spans[$index].confidence_ppm")
        }
        KitParitySpan(
            label = record.label,
            canonicalLabel = record.canonicalLabel,
            action = record.action.wireName,
            start = record.start,
            end = record.end,
            confidencePpm = confidence,
            replacement = record.replacement,
        )
    }
    return KitParityResult(redactedText, spans).also(KitParity::validate)
}

private sealed class JsonValue {
    object Null : JsonValue()

    data class Bool(val value: Boolean) : JsonValue()

    /** An integer literal (no fraction or exponent) that fits in an `Int`. */
    data class Integer(val value: Int) : JsonValue()

    /** Any other number: fractional, exponent, or out of `Int` range. */
    object Number : JsonValue()

    data class Str(val value: String) : JsonValue()

    data class Arr(val items: List<JsonValue>) : JsonValue()

    data class Obj(val fields: Map<String, JsonValue>) : JsonValue()
}

/** Minimal RFC 8259 parser that keeps integer literals distinct from other numbers. */
private class StrictJsonParser(private val text: String) {
    private var index = 0

    fun parseDocument(): JsonValue {
        skipWhitespace()
        val value = parseValue()
        skipWhitespace()
        if (index != text.length) {
            invalid()
        }
        return value
    }

    private fun parseValue(): JsonValue {
        if (index >= text.length) {
            invalid()
        }
        return when (text[index]) {
            '{' -> parseObject()
            '[' -> parseArray()
            '"' -> JsonValue.Str(parseString())
            't' -> literal("true", JsonValue.Bool(true))
            'f' -> literal("false", JsonValue.Bool(false))
            'n' -> literal("null", JsonValue.Null)
            else -> parseNumber()
        }
    }

    private fun parseObject(): JsonValue {
        index += 1
        val fields = LinkedHashMap<String, JsonValue>()
        skipWhitespace()
        if (peek() == '}') {
            index += 1
            return JsonValue.Obj(fields)
        }
        while (true) {
            skipWhitespace()
            if (peek() != '"') {
                invalid()
            }
            val key = parseString()
            skipWhitespace()
            expect(':')
            skipWhitespace()
            fields[key] = parseValue()
            skipWhitespace()
            when (peek()) {
                ',' -> index += 1
                '}' -> {
                    index += 1
                    return JsonValue.Obj(fields)
                }
                else -> invalid()
            }
        }
    }

    private fun parseArray(): JsonValue {
        index += 1
        val items = mutableListOf<JsonValue>()
        skipWhitespace()
        if (peek() == ']') {
            index += 1
            return JsonValue.Arr(items)
        }
        while (true) {
            skipWhitespace()
            items += parseValue()
            skipWhitespace()
            when (peek()) {
                ',' -> index += 1
                ']' -> {
                    index += 1
                    return JsonValue.Arr(items)
                }
                else -> invalid()
            }
        }
    }

    private fun parseString(): String {
        index += 1
        val builder = StringBuilder()
        while (true) {
            if (index >= text.length) {
                invalid()
            }
            val character = text[index]
            index += 1
            when {
                character == '"' -> return builder.toString()
                character == '\\' -> builder.append(parseEscape())
                character.code < 0x20 -> invalid()
                else -> builder.append(character)
            }
        }
    }

    private fun parseEscape(): Char {
        if (index >= text.length) {
            invalid()
        }
        val escape = text[index]
        index += 1
        return when (escape) {
            '"' -> '"'
            '\\' -> '\\'
            '/' -> '/'
            'b' -> '\b'
            'f' -> '\u000C'
            'n' -> '\n'
            'r' -> '\r'
            't' -> '\t'
            'u' -> {
                if (index + 4 > text.length) {
                    invalid()
                }
                val hex = text.substring(index, index + 4)
                if (!hex.all { it.isAsciiDigit() || it in 'a'..'f' || it in 'A'..'F' }) {
                    invalid()
                }
                index += 4
                hex.toInt(16).toChar()
            }
            else -> invalid()
        }
    }

    private fun parseNumber(): JsonValue {
        val start = index
        if (peek() == '-') {
            index += 1
        }
        when {
            peek() == '0' -> index += 1
            peek()?.let { it in '1'..'9' } == true -> skipDigits()
            else -> invalid()
        }
        var integral = true
        if (peek() == '.') {
            integral = false
            index += 1
            if (peek()?.isAsciiDigit() != true) {
                invalid()
            }
            skipDigits()
        }
        if (peek() == 'e' || peek() == 'E') {
            integral = false
            index += 1
            if (peek() == '+' || peek() == '-') {
                index += 1
            }
            if (peek()?.isAsciiDigit() != true) {
                invalid()
            }
            skipDigits()
        }
        val lexeme = text.substring(start, index)
        val integer = if (integral) lexeme.toIntOrNull() else null
        return if (integer != null) JsonValue.Integer(integer) else JsonValue.Number
    }

    private fun skipDigits() {
        while (peek()?.isAsciiDigit() == true) {
            index += 1
        }
    }

    private fun literal(word: String, value: JsonValue): JsonValue {
        if (!text.startsWith(word, index)) {
            invalid()
        }
        index += word.length
        return value
    }

    private fun expect(character: Char) {
        if (peek() != character) {
            invalid()
        }
        index += 1
    }

    private fun peek(): Char? = text.getOrNull(index)

    private fun skipWhitespace() {
        while (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r') {
            index += 1
        }
    }

    private fun Char.isAsciiDigit(): Boolean = this in '0'..'9'

    private fun invalid(): Nothing = throw KitParityException("json_invalid")
}
