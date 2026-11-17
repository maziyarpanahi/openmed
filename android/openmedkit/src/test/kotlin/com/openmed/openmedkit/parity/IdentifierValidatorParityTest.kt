package com.openmed.openmedkit.parity

import com.openmed.openmedkit.merge.PiiValidators
import com.openmed.openmedkit.util.SafeLog
import com.openmed.openmedkit.util.SafeLogEvent
import java.time.LocalDate
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Replays `tests/fixtures/parity/identifier_validators_v1.json` through
 * [PiiValidators]. Every case names its Python reference; assertion messages
 * carry the validator and vector index only, never the value.
 */
class IdentifierValidatorParityTest {
    private val fixture: JsonObject = Json.parseToJsonElement(
        requireNotNull(javaClass.classLoader?.getResourceAsStream(FIXTURE)) {
            "shared $FIXTURE was not synced into Android test resources"
        }.use { String(it.readBytes(), Charsets.UTF_8) },
    ).jsonObject

    @Test
    fun fixtureIsTheSupportedSyntheticVersion() {
        assertEquals("identifier_validators_v1", fixture.getValue("fixture").jsonPrimitive.content)
        assertTrue(fixture.getValue("synthetic").jsonPrimitive.boolean)
    }

    @Test
    fun everyVectorMatchesThePythonReference() {
        val cases = fixture.getValue("cases").jsonArray.map { it.jsonObject }
        assertEquals(VALIDATORS.keys, cases.map { it.getValue("reference").jsonPrimitive.content }.toSet())
        var checked = 0
        for (case in cases) {
            val name = case.getValue("validator").jsonPrimitive.content
            val validate = VALIDATORS.getValue(case.getValue("reference").jsonPrimitive.content)
            for ((expected, key) in listOf(true to "valid", false to "invalid")) {
                case.getValue(key).jsonArray.forEachIndexed { index, value ->
                    assertEquals(expected, validate(value.jsonPrimitive.content), "$name $key[$index]")
                    checked += 1
                }
            }
        }
        assertTrue(checked >= 70)
    }

    @Test
    fun separatorsCaseAndUnicodeDigitsMatchThePythonReference() {
        // Expected results were computed with the Python references named in the fixture.
        for ((index, row) in EDGE_CASES.withIndex()) {
            val (name, value, expected) = row
            val validate = VALIDATORS.values.elementAt(NAMES.indexOf(name))
            assertEquals(expected, validate(value), "$name edge[$index]")
        }
    }

    @Test
    fun chineseResidentIdsBornAfterTodayAreRejected() {
        val value = "110101202001011234".let { it.dropLast(1) + checkCharacter(it.dropLast(1)) }
        assertTrue(PiiValidators.validateChineseResidentID(value, LocalDate.of(2020, 1, 1)))
        assertFalse(PiiValidators.validateChineseResidentID(value, LocalDate.of(2019, 12, 31)))
    }

    @Test
    fun validatorsNeverWriteToSafeLog() {
        val events = mutableListOf<SafeLogEvent>()
        val previous = SafeLog.installSinkForTesting(events::add)
        try {
            for (case in fixture.getValue("cases").jsonArray.map { it.jsonObject }) {
                val validate = VALIDATORS.getValue(case.getValue("reference").jsonPrimitive.content)
                (case.getValue("valid").jsonArray + case.getValue("invalid").jsonArray).forEach {
                    validate(it.jsonPrimitive.content)
                }
            }
            VALIDATORS.values.forEach { validate -> listOf("", " ", "SYNTH Jane Roe", "\u0000").forEach { validate(it) } }
        } finally {
            SafeLog.installSinkForTesting(previous)
        }
        assertEquals(emptyList(), events)
    }

    private fun checkCharacter(body: String): Char {
        val weights = intArrayOf(7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
        return "10X98765432"[body.indices.sumOf { weights[it] * (body[it] - '0') } % 11]
    }

    private companion object {
        const val FIXTURE = "identifier_validators_v1.json"
        const val CLINICAL = "openmed.core.anonymizer.providers.clinical_ids."
        const val I18N = "openmed.core.pii_i18n."

        val VALIDATORS: Map<String, (String) -> Boolean> = linkedMapOf(
            "${CLINICAL}validate_luhn" to PiiValidators::validateLuhn,
            "${CLINICAL}validate_ssn" to PiiValidators::validateSSN,
            "${CLINICAL}validate_npi" to PiiValidators::validateNPI,
            "${CLINICAL}validate_iban" to PiiValidators::validateIBAN,
            "${I18N}validate_aadhaar" to PiiValidators::validateAadhaar,
            "${CLINICAL}validate_uk_nhs_number" to PiiValidators::validateUKNHSNumber,
            "${I18N}validate_dutch_bsn" to PiiValidators::validateDutchBSN,
            "${I18N}validate_spanish_dni" to PiiValidators::validateSpanishDNI,
            "${I18N}validate_french_nir" to PiiValidators::validateFrenchNIR,
            "${I18N}validate_chinese_resident_id" to PiiValidators::validateChineseResidentID,
            "${CLINICAL}validate_canadian_sin" to PiiValidators::validateCanadianSIN,
        )
        val NAMES = listOf(
            "Luhn", "SSN", "NPI", "IBAN", "Aadhaar", "UK NHS number", "Dutch BSN",
            "Spanish DNI", "French NIR", "Chinese resident ID", "Canadian SIN",
        )

        val EDGE_CASES: List<Triple<String, String, Boolean>> = listOf(
            Triple("Aadhaar", "\u0665\u0668\u0668\u0662\u0663\u0663\u0668\u0664\u0666\u0668\u0667\u0661", true),
            Triple("Aadhaar", "\u0A6B\u0A6E\u0A6E\u0A68\u0A69\u0A69\u0A6E\u0A6A\u0A6C\u0A6E\u0A6D\u0A67", true),
            Triple("Aadhaar", "\u06F5\u06F8\u06F8\u06F2\u06F3\u06F3\u06F8\u06F4\u06F6\u06F8\u06F7\u06F1", true),
            Triple("Aadhaar", " 5882 3384 6871 ", true),
            Triple("Aadhaar", "5882-3384-6871", false),
            Triple("Aadhaar", "5882 33846871", false),
            Triple("Aadhaar", "\u00A0588233846871\u2003", true),
            Triple("IBAN", "de46 yvok ys9r yapb 2jju kq", true),
            Triple("IBAN", "DE46-YVOK-YS9R-YAPB-2JJU-KQ", true),
            Triple("IBAN", "DE46YVOKYS9RYAPB2JJUKQ ", true),
            Triple("IBAN", "XX46YVOKYS9RYAPB2JJUKQ", false),
            Triple("IBAN", "GB82WEST12345698765432", true),
            Triple("IBAN", "GB82 WEST 1234 5698 7654 32", true),
            Triple("IBAN", "DE\u0664\u0666YVOKYS9RYAPB2JJUKQ", true),
            Triple("IBAN", "DE46YVOKYS9RYAPB2JJUKQ\u00A0", true),
            Triple("IBAN", "DE46YVOKYS9RYAPB2JJUKQ\u001C", true),
            Triple("UK NHS number", "999 000 0018", true),
            Triple("UK NHS number", "999-000-0018", true),
            Triple("UK NHS number", "\u0669990000018", false),
            Triple("Dutch BSN", "1234.56.782", true),
            Triple("Dutch BSN", "12345678", false),
            Triple("Dutch BSN", "7123 4568 1", true),
            Triple("Dutch BSN", "111222333", true),
            Triple("Spanish DNI", "12345678z", true),
            Triple("Spanish DNI", " 12345678 Z ", true),
            Triple("Spanish DNI", "1234567-8Z", false),
            Triple("Spanish DNI", "\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668Z", true),
            Triple("French NIR", "1 85 05 75 001 234 75", true),
            Triple("French NIR", "1.85.05.75.001.234.75", true),
            Triple("French NIR", "185052A00123475", false),
            Triple("French NIR", "185052B00123475", false),
            Triple("French NIR", "18505750012347+", false),
            Triple("French NIR", "1850575001234+5", false),
            Triple("French NIR", "185052a001234 75", false),
            Triple("Chinese resident ID", "13244520091012315x", true),
            Triple("Chinese resident ID", "110101299901011234", false),
            Triple("Chinese resident ID", "110101000001011237", false),
            Triple("Chinese resident ID", "990101199001011234", false),
            Triple("Chinese resident ID", "119001199001011234", false),
            Triple("Chinese resident ID", "110100199001011234", false),
            Triple("Chinese resident ID", "110101199002301234", false),
            Triple("Chinese resident ID", "110101199001010003", false),
            Triple("Canadian SIN", "975 886 706", true),
            Triple("Canadian SIN", "975-886-706", true),
            Triple("Canadian SIN", "975 886-706", false),
            Triple("Canadian SIN", " 975886706 ", true),
            Triple("Canadian SIN", "975886706\u000A", true),
            Triple("Canadian SIN", "\u0669\u0667\u0665886706", false),
            Triple("French NIR", "185052A00123435", true),
            Triple("French NIR", "285052b00123412", true),
        )
    }
}
