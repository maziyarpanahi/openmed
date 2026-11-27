package com.openmed.openmedkit.parity

import com.openmed.openmedkit.deid.DateShift
import com.openmed.openmedkit.util.SafeLog
import com.openmed.openmedkit.util.SafeLogEvent
import java.math.BigInteger
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Replays `tests/fixtures/parity/date_shift_v1.json`, produced from Python
 * `openmed.core.date_shift`, through [DateShift]. Failures name vector ids only.
 */
class DateShiftParityTest {
    private val fixture: JsonObject = Json.parseToJsonElement(
        requireNotNull(javaClass.classLoader?.getResourceAsStream(FIXTURE)) {
            "shared $FIXTURE was not synced into Android test resources"
        }.use { String(it.readBytes(), Charsets.UTF_8) },
    ).jsonObject
    private val secrets: Map<String, ByteArray> = fixture.getValue("secrets").jsonArray.associate {
        val secret = it.jsonObject
        assertTrue(secret.getValue("test_only").jsonPrimitive.boolean)
        secret.text("id") to hex(secret.text("hex"))
    }

    @Test
    fun fixtureIsTheSupportedSyntheticVersion() {
        assertEquals(1, fixture.getValue("version").jsonPrimitive.int)
        assertTrue(fixture.getValue("synthetic_only").jsonPrimitive.boolean)
        val utf8 = fixture.getValue("secrets").jsonArray.map { it.jsonObject }.single { "utf8" in it }
        assertTrue(utf8.text("utf8").toByteArray(Charsets.UTF_8).contentEquals(secrets.getValue(utf8.text("id"))))
        assertEquals(
            "openmed-date-shift-seed-v1\u0000",
            String(hex(fixture.getValue("algorithm").jsonObject.text("seed_domain_hex")), Charsets.US_ASCII),
        )
    }

    @Test
    fun patientKeyOffsetsMatchPython() {
        val vectors = vectors("patient_key_offsets")
        assertTrue(vectors.size >= 40)
        for (vector in vectors) {
            val id = vector.text("id")
            val key = vector.text("patient_key")
            val secret = secrets.getValue(vector.text("secret_id"))
            val maxDays = vector.getValue("max_days").jsonPrimitive.int

            assertEquals(vector.text("digest_sha256_hex"), hexOf(DateShift.patientKeyDigest(key, secret)), id)
            assertEquals(vector.getValue("offset_days").jsonPrimitive.int, DateShift.stableOffset(key, secret, maxDays), id)
        }
    }

    @Test
    fun seedOffsetsMatchPythonInsideAndOutsideTheLongRange() {
        var outsideLong = 0
        for (vector in vectors("seed_offsets")) {
            val id = vector.text("id")
            val seed = BigInteger(vector.text("seed_decimal"))
            val maxDays = vector.getValue("max_days").jsonPrimitive.int
            val expected = vector.getValue("offset_days").jsonPrimitive.int

            assertEquals(vector.text("digest_sha256_hex"), hexOf(DateShift.seedDigest(seed)), id)
            assertEquals(expected, DateShift.stableOffsetFromSeed(seed, maxDays), id)
            if (seed.bitLength() < 64) {
                assertEquals(expected, DateShift.stableOffsetFromSeed(seed.toLong(), maxDays), id)
            } else {
                outsideLong += 1
            }
            vector["seed"]?.let { assertEquals(seed, BigInteger.valueOf(it.jsonPrimitive.long), id) }
        }
        assertTrue(outsideLong > 0)
    }

    @Test
    fun isoDatesShiftAcrossLeapDaysMonthEndsAndYears() {
        val covered = mutableSetOf<String>()
        for (vector in vectors("date_shifts")) {
            val id = vector.text("id")
            val maxDays = vector.getValue("max_days").jsonPrimitive.int
            val offset = if ("patient_key" in vector) {
                DateShift.stableOffset(vector.text("patient_key"), secrets.getValue(vector.text("secret_id")), maxDays)
            } else {
                DateShift.stableOffsetFromSeed(BigInteger(vector.text("seed_decimal")), maxDays)
            }

            assertEquals(vector.getValue("offset_days").jsonPrimitive.int, offset, id)
            assertEquals(vector.text("shifted_date"), DateShift.shift(vector.text("source_date"), offset), id)
            covered += vector.text("covers")
        }
        assertTrue(covered.containsAll(setOf("leap_day", "month_end", "year_boundary", "non_leap_century")))
    }

    @Test
    fun largeMaxDaysFollowTheFormulaOnEveryFixtureDigest() {
        // Buckets above Int.MAX_VALUE only occur when maxDays > 2^30.
        for (maxDays in listOf((1 shl 30) + 1, Int.MAX_VALUE)) {
            for (vector in vectors("patient_key_offsets")) {
                val id = vector.text("id")
                val secret = secrets.getValue(vector.text("secret_id"))
                assertEquals(
                    referenceOffset(vector.text("digest_sha256_hex"), maxDays),
                    DateShift.stableOffset(vector.text("patient_key"), secret, maxDays),
                    id,
                )
            }
            for (vector in vectors("seed_offsets")) {
                val id = vector.text("id")
                assertEquals(
                    referenceOffset(vector.text("digest_sha256_hex"), maxDays),
                    DateShift.stableOffsetFromSeed(BigInteger(vector.text("seed_decimal")), maxDays),
                    id,
                )
            }
        }
    }

    @Test
    fun offsetsStayInRangeAndAreNeverZero() {
        val secret = secrets.getValue("test-secret-bytes")
        for (maxDays in listOf(1, 2, 7, Int.MAX_VALUE)) {
            for (index in 0 until 50) {
                val offset = DateShift.stableOffset("SYNTH-PATIENT-$index", secret, maxDays)
                assertTrue(offset != 0 && offset >= -maxDays && offset <= maxDays, "$maxDays")
            }
        }
        assertEquals(
            DateShift.stableOffset("SYNTH-PATIENT-0001", secret),
            DateShift.stableOffset("SYNTH-PATIENT-0001", secret, DateShift.DEFAULT_MAX_DAYS),
        )
    }

    @Test
    fun invalidInputFailsWithoutEchoingKeysSecretsSeedsOrDates() {
        val secret = SECRET_SENTINEL.toByteArray()
        val events = mutableListOf<SafeLogEvent>()
        val previous = SafeLog.installSinkForTesting(events::add)
        val errors = try {
            listOf(
                assertFailsWith<IllegalArgumentException> { DateShift.stableOffset("", secret) },
                assertFailsWith<IllegalArgumentException> { DateShift.stableOffset(KEY_SENTINEL, ByteArray(0)) },
                assertFailsWith<IllegalArgumentException> { DateShift.stableOffset(KEY_SENTINEL, secret, 0) },
                assertFailsWith<IllegalArgumentException> { DateShift.stableOffsetFromSeed(7L, -1) },
            ) + listOf("2023-02-29", "2024-13-01", "24-01-01", "2024-01-01T00:00", "+12024-01-01", "0000-01-01", KEY_SENTINEL)
                .map { date -> assertFailsWith<IllegalArgumentException> { DateShift.shift(date, 1) } } +
                assertFailsWith<IllegalArgumentException> { DateShift.shift("9999-12-31", 1) } +
                assertFailsWith<IllegalArgumentException> { DateShift.shift("0001-01-01", -1) }
        } finally {
            SafeLog.installSinkForTesting(previous)
        }
        for (error in errors) {
            val text = "${error.message} ${error.cause}"
            for (value in listOf(KEY_SENTINEL, SECRET_SENTINEL, "2023-02-29", "9999-12-31", "0001-01-01", "+12024")) {
                assertFalse(value in text, text)
            }
        }
        assertEquals(emptyList(), events)
        // Negative control: valid input still works.
        assertEquals("2024-02-29", DateShift.shift("2024-03-01", -1))
    }

    private fun vectors(name: String): List<JsonObject> = fixture.getValue(name).jsonArray.map { it.jsonObject }

    /** The fixture's `algorithm` offset rule in arbitrary precision. */
    private fun referenceOffset(digestHex: String, maxDays: Int): Int {
        val limit = BigInteger.valueOf(maxDays.toLong())
        val bucket = BigInteger(digestHex, 16).mod(limit.shiftLeft(1))
        val offset = if (bucket < limit) bucket - limit else bucket - limit + BigInteger.ONE
        return offset.intValueExact()
    }

    private fun JsonObject.text(key: String): String = getValue(key).jsonPrimitive.content

    private fun hex(value: String): ByteArray = ByteArray(value.length / 2) {
        value.substring(it * 2, it * 2 + 2).toInt(16).toByte()
    }

    private fun hexOf(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private companion object {
        const val FIXTURE = "date_shift_v1.json"
        const val KEY_SENTINEL = "SYNTH Jane Roe 1970-01-01"
        const val SECRET_SENTINEL = "SYNTH_SECRET_SENTINEL"
    }
}
