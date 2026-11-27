package com.openmed.openmedkit.parity

import com.openmed.openmedkit.receipt.ReceiptVerification
import com.openmed.openmedkit.receipt.ReceiptVerifier
import com.openmed.openmedkit.util.SafeLog
import com.openmed.openmedkit.util.SafeLogEvent
import java.util.Base64
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Replays `tests/fixtures/parity/receipt_verification_v1.json`, generated from
 * Python `AuditReport.sign()` and `verify()`, through the Kotlin verifier. The
 * HMAC material in the fixture is published test-only material.
 */
class ReceiptVerificationParityTest {
    private val fixture: JsonObject = Json.parseToJsonElement(
        String(
            requireNotNull(javaClass.classLoader?.getResourceAsStream(FIXTURE)) {
                "shared $FIXTURE was not synced into Android test resources"
            }.use { it.readBytes() },
            Charsets.UTF_8,
        ),
    ).jsonObject
    private val cases = fixture.getValue("cases").jsonArray.map { it.jsonObject }
    private val keys = fixture.getValue("test_keys").jsonArray.map { it.jsonObject }.associate {
        it.text("key_id") to it.text("material").toByteArray(Charsets.UTF_8)
    }

    @Test
    fun fixtureIsTheSupportedSyntheticVersion() {
        assertEquals("openmed.receipt_verification_vectors", fixture.text("schema"))
        assertEquals(1, fixture.getValue("version").jsonPrimitive.int)
        assertTrue(fixture.getValue("synthetic_only").jsonPrimitive.boolean)
        assertEquals(ReceiptVerifier.ALGORITHM, fixture.text("signature_algorithm"))
        assertTrue(keys.isNotEmpty())
        assertTrue(cases.size >= 10)
    }

    @Test
    fun everyVectorReproducesTheCanonicalBytesHashesAndHmacs() {
        for (case in cases) {
            val id = case.text("id")
            val report = case.getValue("report").jsonObject
            val candidates = ReceiptVerifier.candidates(Json.parseToJsonElement(report.toString()).jsonObject)
            val expected = case.getValue("candidates").jsonArray.map { it.jsonObject }

            assertEquals(expected.map { it.text("span_order") }, candidates.map { it.first }, id)
            for ((want, got) in expected.zip(candidates)) {
                val (spanOrder, repro, signing) = got
                val order = "$id $spanOrder"
                assertContentEquals(decode(want.text("repro_payload_b64")), repro, "$order repro bytes")
                assertEquals(want.text("expected_repro_hash"), ReceiptVerifier.sha256(repro), "$order repro_hash")
                assertContentEquals(decode(want.text("signing_payload_b64")), signing, "$order signing bytes")
                assertEquals(want.text("expected_hmac"), hmacHex(keyFor(case), signing), "$order hmac")
            }
        }
    }

    @Test
    fun everyVectorProducesTheExpectedVerdict() {
        val codes = mutableSetOf<String>()
        for (case in cases) {
            val id = case.text("id")
            val report = case.getValue("report").jsonObject
            val expected = case.getValue("expected").jsonObject
            val result = ReceiptVerifier.verify(report.toString(), keyFor(case))
            codes += result.code

            assertEquals(expected.getValue("verifies").jsonPrimitive.boolean, result is ReceiptVerification.Valid, id)
            assertEquals(expectedCode(report, expected), result.code, id)
            val order = expected["matched_span_order"]?.takeIf { it is JsonPrimitive && it.isString }?.jsonPrimitive?.content
            assertEquals(order, (result as? ReceiptVerification.Valid)?.spanOrder, id)
            assertEquals(result is ReceiptVerification.Valid, result.code == "valid", id)
        }
        assertEquals(setOf("valid", "hash_mismatch", "signature_mismatch", "unsupported_algorithm"), codes)
        assertEquals("stored", (verify("legacy_span_order") as ReceiptVerification.Valid).spanOrder)
    }

    @Test
    fun floatsUsePythonRepr() {
        val expected = mapOf(
            0.1 to "0.1",
            1e-07 to "1e-07",
            5e-324 to "5e-324",
            1e16 to "1e+16",
            1.0 to "1.0",
            0.30000000000000004 to "0.30000000000000004",
            1e15 to "1000000000000000.0",
            1e-4 to "0.0001",
            1e-5 to "1e-05",
            123456789012345678.0 to "1.2345678901234568e+17",
            -0.0 to "-0.0",
            0.0 to "0.0",
            1.7976931348623157e308 to "1.7976931348623157e+308",
            2.2250738585072014e-308 to "2.2250738585072014e-308",
            9007199254740993.0 to "9007199254740992.0",
            2.82879384806159e17 to "2.82879384806159e+17",
            1e22 to "1e+22",
            1e23 to "1e+23",
            100.0 to "100.0",
            1.5e-07 to "1.5e-07",
            12345.678 to "12345.678",
            -2.5e-10 to "-2.5e-10",
            1234567890123456.0 to "1234567890123456.0",
            12345678901234567.0 to "1.2345678901234568e+16",
            3.337610787760802e-308 to "3.337610787760802e-308",
        )
        for ((value, repr) in expected) {
            assertEquals(repr, ReceiptVerifier.pythonFloatRepr(value), repr)
        }
    }

    @Test
    fun canonicalJsonSortsByCodePointAndEscapesLikePython() {
        val document = Json.parseToJsonElement(
            """{"z":1,"é":2,"�":3,"😀":4,"a\"b\\c\n\u007f\u0001/":5,""" +
                """"n":{"a":1E2,"b":-0,"c":12345678901234567890,"d":1e-400,"e":[true,false,null]}}""",
        )

        assertEquals(
            """{"a\"b\\c\n\u007f\u0001/":5,"n":{"a":100.0,"b":0,"c":12345678901234567890,"d":0.0,""" +
                "\"e\":[true,false,null]},\"z\":1,\"\\u00e9\":2,\"\\ufffd\":3,\"\\ud83d\\ude00\":4}",
            ReceiptVerifier.canonicalJson(document),
        )
    }

    @Test
    fun malformedInputReturnsMalformedWithoutEchoingIt() {
        val baseline = report("signed_baseline")
        val spans = baseline.getValue("spans").jsonArray
        val firstSpan = spans[0].jsonObject
        val broken = listOf(
            "not json $SENTINEL",
            "[\"$SENTINEL\"]",
            "{\"spans\": [",
            baseline.with("spans", JsonPrimitive(SENTINEL)).toString(),
            baseline.with("spans", Json.parseToJsonElement("[\"$SENTINEL\"]")).toString(),
            baseline.with("spans", JsonArrayOf(firstSpan.with("start", JsonPrimitive("8")))).toString(),
            baseline.with("spans", JsonArrayOf(firstSpan.without("canonical_label"))).toString(),
            baseline.with("signature", JsonPrimitive(SENTINEL)).toString(),
            baseline.with("signature", baseline.getValue("signature").jsonObject.without("value")).toString(),
            baseline.without("repro_hash").toString(),
            baseline.toString().replace("\"projected_leakage\":0.0", "\"projected_leakage\":NaN"),
            baseline.toString().replace("\"projected_leakage\":0.0", "\"projected_leakage\":1e400"),
            baseline.toString().replace("\"projected_leakage\":0.0", "\"projected_leakage\":$SENTINEL"),
            baseline.toString().replace(
                "\"projected_leakage\":0.0",
                "\"projected_leakage\":" + "[".repeat(100_000) + "]".repeat(100_000),
            ),
        )
        for (input in broken) {
            val result = ReceiptVerifier.verify(input, keys.getValue("test-only"))
            assertTrue(result is ReceiptVerification.Rejected, input.take(40))
            assertEquals(ReceiptVerifier.MALFORMED, result.code, input.take(40))
            assertEquals("malformed", result.toString())
        }
    }

    @Test
    fun keyNeverAppearsInResultsErrorsOrSafeLog() {
        val sentinelKey = "SYNTH_HMAC_KEY_SENTINEL".toByteArray()
        val events = mutableListOf<SafeLogEvent>()
        val previous = SafeLog.installSinkForTesting(events::add)
        try {
            for (case in cases) {
                val result = ReceiptVerifier.verify(case.getValue("report").toString(), sentinelKey)
                assertFalse(result is ReceiptVerification.Valid, case.text("id"))
                assertFalse("SYNTH_HMAC_KEY_SENTINEL" in result.toString())
            }
            val empty = assertFailsWith<IllegalArgumentException> {
                ReceiptVerifier.verify(report("signed_baseline").toString(), ByteArray(0))
            }
            assertEquals("receipt verification requires a non-empty HMAC key", empty.message)
        } finally {
            SafeLog.installSinkForTesting(previous)
        }
        assertEquals(emptyList(), events)
        assertEquals(ReceiptVerifier.SIGNATURE_MISMATCH, verify("signed_baseline", sentinelKey).code)
    }

    @Test
    fun reformattedValuesVerifyAndChangedValuesDoNot() {
        val baseline = report("signed_baseline")
        val text = baseline.toString()
        val key = keys.getValue("test-only")

        // Whitespace, member order and float spelling do not change canonical bytes.
        val reformatted = Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), baseline)
            .replace("0.97", "9.7e-1")
        assertTrue(ReceiptVerifier.verify(reformatted, key) is ReceiptVerification.Valid)
        val reordered = baseline.with("spans", JsonArrayOf(*baseline.getValue("spans").jsonArray.reversed().toTypedArray()))
        assertEquals("sorted", (ReceiptVerifier.verify(reordered.toString(), key) as ReceiptVerification.Valid).spanOrder)

        for (changed in listOf(
            text.replace("0.97", "0.98"),
            text.replace("\"start\":8", "\"start\":9"),
            baseline.with("extra", JsonPrimitive(1)).toString(),
        )) {
            assertEquals(ReceiptVerifier.HASH_MISMATCH, ReceiptVerifier.verify(changed, key).code)
        }
    }

    @Test
    fun signatureChecksRunBeforeTheHashInTheSharedOrder() {
        val baseline = report("signed_baseline")
        val signature = baseline.getValue("signature").jsonObject
        val tampered = Json.parseToJsonElement(baseline.toString().replace("\"start\":8", "\"start\":9")).jsonObject
        val key = keys.getValue("test-only")

        // A missing or null signature is signature_mismatch even when the content changed.
        for (unsigned in listOf(tampered.with("signature", JsonNull), tampered.without("signature"), baseline.without("signature"))) {
            assertEquals(ReceiptVerifier.SIGNATURE_MISMATCH, ReceiptVerifier.verify(unsigned.toString(), key).code)
        }
        // Any algorithm other than HMAC-SHA256, including none, is unsupported_algorithm.
        for (algorithm in listOf(null, JsonNull, JsonPrimitive(256), JsonPrimitive("hmac-sha256"))) {
            val changed = if (algorithm == null) signature.without("algorithm") else signature.with("algorithm", algorithm)
            assertEquals(
                ReceiptVerifier.UNSUPPORTED_ALGORITHM,
                ReceiptVerifier.verify(tampered.with("signature", changed).toString(), key).code,
            )
        }
        // A signature without a string value cannot be checked.
        for (changed in listOf(signature.with("value", JsonPrimitive(7)), JsonObject(emptyMap()))) {
            assertEquals(ReceiptVerifier.MALFORMED, ReceiptVerifier.verify(baseline.with("signature", changed).toString(), key).code)
        }
    }

    private fun expectedCode(report: JsonObject, expected: JsonObject): String {
        val signature = report["signature"] as? JsonObject
        return when {
            expected.getValue("verifies").jsonPrimitive.boolean -> "valid"
            signature != null && signature.text("algorithm") != ReceiptVerifier.ALGORITHM -> "unsupported_algorithm"
            !expected.getValue("repro_hash_matches").jsonPrimitive.boolean -> "hash_mismatch"
            else -> "signature_mismatch"
        }
    }

    private fun verify(id: String, key: ByteArray = keys.getValue("test-only")): ReceiptVerification =
        ReceiptVerifier.verify(report(id).toString(), key)

    private fun report(id: String): JsonObject = cases.single { it.text("id") == id }.getValue("report").jsonObject

    private fun keyFor(case: JsonObject): ByteArray = keys.getValue(case.text("key_id"))

    private fun decode(value: String): ByteArray = Base64.getDecoder().decode(value)

    private fun hmacHex(key: ByteArray, message: ByteArray): String =
        Mac.getInstance("HmacSHA256").run {
            init(SecretKeySpec(key, "HmacSHA256"))
            doFinal(message)
        }.joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun JsonObject.text(key: String): String = getValue(key).jsonPrimitive.content

    private fun JsonObject.with(key: String, value: kotlinx.serialization.json.JsonElement): JsonObject =
        JsonObject(this + (key to value))

    private fun JsonObject.without(key: String): JsonObject = JsonObject(this - key)

    @Suppress("FunctionName")
    private fun JsonArrayOf(vararg items: kotlinx.serialization.json.JsonElement) =
        kotlinx.serialization.json.JsonArray(items.toList())

    private companion object {
        const val FIXTURE = "receipt_verification_v1.json"
        const val SENTINEL = "SYNTH_PHI_SENTINEL Jane Roe"
    }
}
