package com.openmed.openmedkit.merge

import java.time.DateTimeException
import java.time.LocalDate

/**
 * Deterministic validators used by semantic PII patterns.
 *
 * The portable subset (SSN, Luhn, NPI, IBAN, Aadhaar, UK NHS number, Dutch
 * BSN, Spanish DNI, French NIR, Chinese resident ID and Canadian SIN) matches
 * the Python reference named in
 * `tests/fixtures/parity/identifier_validators_v1.json`.
 * Digits follow those references too: SSN, Luhn, NPI, US phone, UK NHS number,
 * Dutch BSN, Chinese resident ID and Canadian SIN read only the ASCII digits
 * 0-9, as Python's `[0-9]` does, so Arabic-Indic, fullwidth and other non-ASCII
 * digits never count toward them. The IBAN check digits, the Spanish DNI number and the
 * French NIR after its leading 1 or 2 accept any Unicode decimal digit, as
 * Python's `\d` and `int()` do, and Aadhaar accepts Arabic-Indic and Gurmukhi
 * digits.
 * Validators only return a Boolean: they never log, store or echo the input.
 */
object PiiValidators {
    /**
     * US SSN: nine ASCII digits outside the 000, 666 and 9xx areas, the 00
     * group and the 0000 serial. Every other character is ignored.
     */
    @JvmStatic
    fun validateSSN(text: String): Boolean {
        val digits = asciiDigits(text)
        if (digits.length != 9) {
            return false
        }

        val area = digits.substring(0, 3)
        val group = digits.substring(3, 5)
        val serial = digits.substring(5, 9)

        if (area == "000" || area == "666" || area.startsWith("9")) {
            return false
        }
        if (group == "00") {
            return false
        }
        if (serial == "0000") {
            return false
        }

        return true
    }

    /** Luhn checksum over at least 13 ASCII digits; every other character is ignored. */
    @JvmStatic
    fun validateLuhn(text: String): Boolean {
        val digits = asciiDigits(text)
        if (digits.length < 13) {
            return false
        }
        return luhnChecksum(digits) == 0
    }

    /** US NPI: ten ASCII digits that pass Luhn behind the 80840 prefix; every other character is ignored. */
    @JvmStatic
    fun validateNPI(text: String): Boolean {
        val digits = asciiDigits(text)
        if (digits.length != 10) {
            return false
        }
        return luhnChecksum("80840$digits") == 0
    }

    /**
     * US phone: ten ASCII digits, or eleven with a leading 1, whose area code
     * starts with 2-9 and exchange does not start with 0. Every other
     * character is ignored.
     */
    @JvmStatic
    fun validatePhoneUS(text: String): Boolean {
        val digits = asciiDigits(text)

        if (digits.length == 10) {
            val firstArea = digits[0]
            val firstExchange = digits[3]

            if (firstArea == '0' || firstArea == '1') {
                return false
            }
            if (firstExchange == '0') {
                return false
            }

            return true
        }

        if (digits.length == 11 && digits.first() == '1') {
            return validatePhoneUS(digits.drop(1))
        }

        return false
    }

    /**
     * ISO 13616 IBAN: country shape and length, then the mod-97 checksum.
     * Whitespace and hyphens are ignored and letters are case-insensitive.
     */
    @JvmStatic
    fun validateIBAN(text: String): Boolean {
        val cleaned = text.filterNot { isPythonSpace(it) || it == '-' }.uppercase()
        if (!IBAN_SHAPE.matches(cleaned)) return false
        // Python counts code points; a supplementary-plane digit is one character.
        val points = cleaned.codePoints().toArray()
        val length = tableValue(IBAN_LENGTHS, cleaned.substring(0, 2))
        if (length != null && points.size != length) return false
        if (length == null && points.size !in 15..34) return false
        var remainder = 0
        for (point in points.copyOfRange(4, points.size) + points.copyOfRange(0, 4)) {
            val digit = Character.digit(point, 10)
            for (value in if (digit >= 0) digit.toString() else (point - 55).toString()) {
                remainder = (remainder * 10 + (value - '0')) % 97
            }
        }
        return remainder == 1
    }

    /**
     * Indian Aadhaar: 12 digits not starting with 0 or 1, or the 4-4-4 spaced
     * form, with a Verhoeff check digit. Arabic-Indic and Gurmukhi digits count.
     */
    @JvmStatic
    fun validateAadhaar(text: String): Boolean {
        val candidate = buildString {
            for (character in text.trim(::isPythonSpace)) {
                append(
                    when (character) {
                        in '\u0660'..'\u0669' -> '0' + (character - '\u0660')
                        in '\u06F0'..'\u06F9' -> '0' + (character - '\u06F0')
                        in '\u0A66'..'\u0A6F' -> '0' + (character - '\u0A66')
                        else -> character
                    },
                )
            }
        }
        if (!AADHAAR_SHAPE.matches(candidate)) return false
        var check = 0
        candidate.replace(" ", "").reversed().forEachIndexed { index, character ->
            val permuted = VERHOEFF_P[(index % 8) * 10 + (character - '0')] - '0'
            check = VERHOEFF_D[check * 10 + permuted] - '0'
        }
        return check == 0
    }

    /** UK NHS Number: ten digits with the Modulus 11 check digit. */
    @JvmStatic
    fun validateUKNHSNumber(text: String): Boolean {
        val digits = asciiDigits(text)
        if (digits.length != 10) return false
        var total = 0
        for (index in 0 until 9) total += (10 - index) * (digits[index] - '0')
        val check = (11 - total % 11).let { if (it == 11) 0 else it }
        return check != 10 && digits[9] - '0' == check
    }

    /** Dutch BSN: eight or nine digits (eight are zero-padded) passing the elfproef. */
    @JvmStatic
    fun validateDutchBSN(text: String): Boolean {
        val digits = asciiDigits(text)
        if (digits.length != 8 && digits.length != 9) return false
        val padded = digits.padStart(9, '0')
        if (padded == "000000000") return false
        var checksum = -(padded[8] - '0')
        for (index in 0 until 8) checksum += (9 - index) * (padded[index] - '0')
        return checksum % 11 == 0
    }

    /** Spanish DNI: eight digits and the mod-23 check letter; whitespace is ignored. */
    @JvmStatic
    fun validateSpanishDNI(text: String): Boolean {
        val cleaned = text.filterNot(::isPythonSpace).uppercase()
        val match = DNI_SHAPE.matchEntire(cleaned) ?: return false
        val number = decimalValue(match.groupValues[1]) ?: return false
        return match.groupValues[2][0] == DNI_LETTERS[(number % 23).toInt()]
    }

    /**
     * French NIR: 15 characters starting with 1 or 2 whose two-digit key is
     * `97 - (first 13 digits mod 97)`; Corsican 2A and 2B count as 19 and 18.
     * Whitespace, dots and hyphens are ignored.
     */
    @JvmStatic
    fun validateFrenchNIR(text: String): Boolean {
        val cleaned = text.filterNot { isPythonSpace(it) || it == '.' || it == '-' }.uppercase()
        if (cleaned.codePointCount(0, cleaned.length) != 15 || (cleaned[0] != '1' && cleaned[0] != '2')) return false
        val split = cleaned.offsetByCodePoints(0, 13)
        // Python's int() reads the two-character key, which may carry a "+".
        val keyText = cleaned.substring(split)
        val key = decimalValue(if (keyText.startsWith('+')) keyText.substring(1) else keyText) ?: return false
        var body = cleaned.substring(0, split)
        if ('A' in body || 'B' in body) {
            if (!NIR_CORSICA.matches(body)) return false
            body = body.replace("2A", "19").replace("2B", "18")
        }
        val number = decimalValue(body) ?: return false
        return key == 97 - number % 97
    }

    /**
     * Mainland China 18-character resident ID: province prefix, prefecture and
     * county digits, a real birth date not after today, a non-zero sequence and
     * the ISO 7064 MOD 11-2 check character (`X` is case-insensitive).
     */
    @JvmStatic
    fun validateChineseResidentID(text: String): Boolean = validateChineseResidentID(text, LocalDate.now())

    internal fun validateChineseResidentID(text: String, today: LocalDate): Boolean {
        val cleaned = text.uppercase()
        if (!CHINESE_ID_SHAPE.matches(cleaned)) return false
        val prefecture = cleaned.substring(2, 4).toInt()
        if (tableValue(CHINESE_ID_PROVINCES, cleaned.substring(0, 2), valueWidth = 0) == null ||
            (prefecture > 70 && prefecture != 90) || prefecture == 0 || cleaned.substring(4, 6) == "00"
        ) {
            return false
        }
        val year = cleaned.substring(6, 10).toInt()
        val birth = try {
            LocalDate.of(year, cleaned.substring(10, 12).toInt(), cleaned.substring(12, 14).toInt())
        } catch (error: DateTimeException) {
            return false
        }
        if (year < 1 || birth.isAfter(today) || cleaned.substring(14, 17) == "000") return false
        var total = 0
        for (index in 0 until 17) total += CHINESE_ID_WEIGHTS[index] * (cleaned[index] - '0')
        return cleaned[17] == "10X98765432"[total % 11]
    }

    /** Canadian SIN: nine digits, plain or grouped 3-3-3 with one separator, not starting with 0, Luhn-valid. */
    @JvmStatic
    fun validateCanadianSIN(text: String): Boolean {
        val candidate = text.trim(::isPythonSpace)
        if (!SIN_SHAPE.matches(candidate)) return false
        val digits = asciiDigits(candidate)
        return digits[0] != '0' && luhnChecksum(digits) == 0
    }

    /** The characters Python's `str.isspace()` and regex `\s` treat as whitespace. */
    private fun isPythonSpace(character: Char): Boolean =
        character in '\t'..'\r' || character in '\u001C'..'\u001F' || character == '\u0085' ||
            Character.isSpaceChar(character)

    /** The ASCII digits 0-9 of [text], as Python's `re.sub(r"[^0-9]", "", text)` keeps. */
    private fun asciiDigits(text: String): String = text.filter { it in '0'..'9' }

    /**
     * Python `int()` of a run of Unicode decimal digits, read by code point so
     * supplementary-plane digits count; `null` when empty or not all digits.
     */
    private fun decimalValue(text: String): Long? {
        if (text.isEmpty()) return null
        var value = 0L
        for (point in text.codePoints()) {
            val digit = Character.digit(point, 10)
            if (digit < 0) return null
            value = value * 10 + digit
        }
        return value
    }

    /** Value for a 2-letter [code] in a packed table of codes each followed by [valueWidth] digits. */
    private fun tableValue(table: String, code: String, valueWidth: Int = 2): Int? {
        var index = 0
        while (index < table.length) {
            if (table.regionMatches(index, code, 0, 2)) {
                return if (valueWidth == 0) 0 else table.substring(index + 2, index + 2 + valueWidth).toInt()
            }
            index += 2 + valueWidth
        }
        return null
    }

    private val IBAN_SHAPE = Regex("[A-Z]{2}\\p{Nd}{2}[A-Z0-9]{11,30}")
    private val AADHAAR_SHAPE = Regex("[2-9][0-9]{11}|[2-9][0-9]{3} [0-9]{4} [0-9]{4}")
    private val DNI_SHAPE = Regex("(\\p{Nd}{8})([A-Z])")
    private val NIR_CORSICA = Regex("[12]\\p{Nd}{4}2[AB]\\p{Nd}{6}")
    private val CHINESE_ID_SHAPE = Regex("[0-9]{17}[0-9X]")
    private val SIN_SHAPE = Regex("[0-9]{9}|[0-9]{3}([ -])[0-9]{3}\\1[0-9]{3}")
    private const val DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
    private const val VERHOEFF_D =
        "0123456789123406789523401789563401289567401239567859876043216598710432765982104387659321049876543210"
    private const val VERHOEFF_P =
        "01234567891576283094580379614289160435279453126870428657390127938064157046913258"
    private val CHINESE_ID_WEIGHTS = intArrayOf(7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    private const val CHINESE_ID_PROVINCES = "11121314152122233132333435363741424344454650515253546162636465"

    /** ISO 13616 IBAN lengths by country, as in Python `clinical_ids._IBAN_LENGTHS`. */
    private const val IBAN_LENGTHS =
        "AD24AE23AL28AT20AZ28BA20BE16BG22BH22BR29BY28CH21CR22CY28CZ24DE22DK18DO28EE20EG29ES24FI18" +
            "FO18FR27GB22GE22GI23GL18GR27GT28HR21HU28IE22IL23IS26IT27JO30KW30KZ20LB28LC32LI21LT20LU20" +
            "LV21MC27MD24ME22MK19MR27MT31MU30NL18NO15PK24PL28PS29PT25QA29RO24RS22SA24SC31SE24SI19SK24" +
            "SM27TN24TR26UA29VA22VG24XK20"

    private fun luhnChecksum(digits: String): Int {
        var checksum = 0

        digits.reversed().forEachIndexed { index, char ->
            val value = char.digitToIntOrNull() ?: return@forEachIndexed
            checksum += if (index % 2 == 0) {
                value
            } else {
                val doubled = value * 2
                if (doubled > 9) doubled - 9 else doubled
            }
        }

        return checksum % 10
    }
}
