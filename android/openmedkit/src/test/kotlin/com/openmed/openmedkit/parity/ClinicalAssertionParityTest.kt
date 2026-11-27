package com.openmed.openmedkit.parity

import com.openmed.openmedkit.ScalarSpan
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.clinical.ClinicalAssertion
import com.openmed.openmedkit.clinical.ClinicalContext
import com.openmed.openmedkit.packs.repositoryRoot
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.json.JSONObject

/**
 * Replays `tests/fixtures/parity/clinical_assertion_v1.json` through the
 * Kotlin ConText port. The vectors record today's Python output, including
 * the `reference_gap` cases, and every case must match it exactly.
 */
class ClinicalAssertionParityTest {
    private val vectors = JSONObject(resource("clinical_assertion_v1.json"))
    private val cases = vectors.getJSONArray("cases").let { array ->
        (0 until array.length()).map(array::getJSONObject)
    }

    @Test
    fun bundledCueTableIsTheSharedTableByteForByte() {
        val shared = repositoryRoot().resolve("tests/fixtures/parity/context_cues_en_v1.json")
        val bundled = requireNotNull(ClinicalContext.bundledCueTable())

        assertContentEquals(shared.readBytes(), bundled)
        assertEquals(vectors.getString("cue_table_sha256"), sha256(bundled))
        assertEquals("context_cues_en_v1.json", vectors.getString("cue_table"))
    }

    @Test
    fun reproducesEveryRecordedCaseWithItsSentences() {
        assertEquals("unicode_scalar", vectors.getString("offset_unit"))
        assertTrue(cases.size >= 40)
        for (case in cases) {
            val text = case.getString("text")
            val target = case.getJSONObject("target")
            val span = ScalarSpan(target.getInt("start"), target.getInt("end"))
            assertEquals(
                target.getString("text"),
                UnicodeOffsetContract.substring(text, span.start, span.end),
                case.getString("id"),
            )

            val actual = ClinicalContext.assertContext(text, listOf(span), sentences(case))

            assertAxes(case, requireNotNull(actual).single())
        }
    }

    @Test
    fun regexSentenceFallbackReproducesTheSameCases() {
        for (case in cases) {
            val target = case.getJSONObject("target")
            val actual = ClinicalContext.assertContext(
                case.getString("text"),
                listOf(ScalarSpan(target.getInt("start"), target.getInt("end"))),
            )

            assertAxes(case, requireNotNull(actual).single())
        }
    }

    @Test
    fun recordedNegationScopeGapsStayAffirmedLikePython() {
        val gaps = cases.filter {
            it.optString("reference_gap") in setOf(
                "negation_scope_ends_at_and",
                "negation_scope_ends_at_or",
            )
        }
        assertTrue(gaps.size >= 4)
        for (case in gaps) {
            val target = case.getJSONObject("target")
            val actual = ClinicalContext.assertContext(
                case.getString("text"),
                listOf(ScalarSpan(target.getInt("start"), target.getInt("end"))),
                sentences(case),
            )
            assertEquals("affirmed", requireNotNull(actual).single().negation, case.getString("id"))
        }
    }

    @Test
    fun nonAsciiCasesUseScalarOffsetsThatDifferFromUtf16() {
        val nonAscii = cases.filter { it.getBoolean("non_ascii") }
        assertTrue(nonAscii.isNotEmpty())
        val astral = nonAscii.first { case ->
            case.getString("text").codePointCount(0, case.getString("text").length) !=
                case.getString("text").length
        }
        val text = astral.getString("text")
        val target = astral.getJSONObject("target")
        assertNotEquals(text.indexOf(target.getString("text")), target.getInt("start"))
        val actual = ClinicalContext.assertContext(
            text,
            listOf(ScalarSpan(target.getInt("start"), target.getInt("end"))),
            sentences(astral),
        )
        assertAxes(astral, requireNotNull(actual).single())
    }

    @Test
    fun nonEnglishInputReturnsNoAssertion() {
        val text = "Patient denies chest pain."
        val spans = listOf(ScalarSpan(15, 25))

        assertNull(ClinicalContext.assertContext(text, spans, language = "de"))
        assertNull(ClinicalContext.assertContext(text, spans, language = "fr-CA"))
        assertEquals(
            "negated",
            ClinicalContext.assertContext(text, spans, language = "en-US")?.single()?.negation,
        )
        assertEquals("unsupported_language", ClinicalContext.UNSUPPORTED_LANGUAGE)
    }

    private fun assertAxes(case: JSONObject, actual: ClinicalAssertion) {
        val expected = case.getJSONObject("expected")
        val id = case.getString("id")
        assertEquals(expected.getString("negation"), actual.negation, "$id negation")
        assertEquals(expected.getString("uncertainty"), actual.uncertainty, "$id uncertainty")
        assertEquals(expected.getString("temporality"), actual.temporality, "$id temporality")
        assertEquals(expected.getString("experiencer"), actual.experiencer, "$id experiencer")
    }

    private fun sentences(case: JSONObject): List<ScalarSpan> {
        val array = case.getJSONArray("sentences")
        return (0 until array.length()).map { index ->
            val sentence = array.getJSONObject(index)
            ScalarSpan(sentence.getInt("start"), sentence.getInt("end"))
        }
    }

    private fun resource(name: String): String =
        requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) { "missing $name" }
            .use { it.readBytes().decodeToString() }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
