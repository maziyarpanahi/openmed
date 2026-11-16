package com.openmed.openmedkit.parity

import com.openmed.openmedkit.speech.SpeechAlignmentStatus
import com.openmed.openmedkit.speech.SpeechContractError
import com.openmed.openmedkit.speech.SpeechContractException
import com.openmed.openmedkit.speech.SpeechEngineProvenance
import com.openmed.openmedkit.speech.SpeechRecognition
import com.openmed.openmedkit.speech.SpeechTokenTiming
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
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
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Runs `tests/fixtures/parity/speech_contract_v1.json`, shared with Python
 * (`openmed.multimodal.streaming.speech.providers`) and Swift
 * (`SpeechContractParityTests`). Pure JVM: no audio, model or network.
 */
class SpeechContractParityTest {
    private val fixture: JsonObject = javaClass.classLoader!!.getResourceAsStream(
        "speech_contract_v1.json",
    )!!.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
    private val segment = fixture.getValue("segment").jsonObject
    private val tokenRows = segment.getValue("token_times").jsonArray.map { it.jsonObject }
    private val provenance = fixture.getValue("provenance").jsonObject
    private val protectedText = listOf(segment.getValue("text").jsonPrimitive.content) +
        tokenRows.map { it.getValue("token").jsonPrimitive.content }

    private fun token(row: Map<String, JsonElement>) = SpeechTokenTiming(
        tokenIndex = row.getValue("token_index").jsonPrimitive.int,
        startMilliseconds = row.getValue("start_ms").jsonPrimitive.long,
        endMilliseconds = row.getValue("end_ms").jsonPrimitive.long,
        confidencePpm = row.getValue("confidence_ppm").jsonPrimitive.int,
        text = row.getValue("token").jsonPrimitive.content,
    )

    private fun engine(row: Map<String, JsonElement>) = SpeechEngineProvenance(
        providerId = row.getValue("provider_id").jsonPrimitive.content,
        providerRevision = row.getValue("provider_revision").jsonPrimitive.content,
        modelFingerprint = row.getValue("model_fingerprint").jsonPrimitive.content,
        language = row.getValue("language").jsonPrimitive.content,
    )

    private fun refused(error: SpeechContractError, category: String, label: String, build: () -> Any) {
        val failure = assertFailsWith<SpeechContractException>(label) { build() }
        assertEquals(error, failure.error, label)
        assertEquals(category, failure.category, label)
        assertEquals("${error.code}: $category", failure.message, label)
        assertNull(failure.cause, label)
        for (text in protectedText) assertFalse(failure.toString().contains(text.trim()), label)
    }

    @Test
    fun sharedSegmentKeepsUnitsOrderAndProtectedText() {
        assertEquals(1, fixture.getValue("version").jsonPrimitive.int)
        assertTrue(fixture.getValue("synthetic_only").jsonPrimitive.boolean)
        val tokens = tokenRows.map(::token)
        val result = SpeechRecognition(text = segment.getValue("text").jsonPrimitive.content, tokenTimes = tokens)
        assertEquals(6, result.text.codePointCount(0, result.text.length))
        assertEquals(listOf(0, 1), result.tokenTimes.map { it.tokenIndex })
        assertEquals(listOf(500L, 650L), result.tokenTimes.map { it.startMilliseconds })
        assertEquals(listOf(650L, 900L), result.tokenTimes.map { it.endMilliseconds })
        assertEquals(listOf(750_000, 250_000), result.tokenTimes.map { it.confidencePpm })
        assertEquals(SpeechAlignmentStatus.ALIGNED, result.alignmentStatus)
        assertNull(result.decoderConfidencePpm)
        // Every token lies inside the declared segment, as Python FinalizedSegment requires.
        val start = segment.getValue("start_ms").jsonPrimitive.long
        val end = segment.getValue("end_ms").jsonPrimitive.long
        assertTrue(result.tokenTimes.all { it.startMilliseconds >= start && it.endMilliseconds <= end })
        assertEquals("SpeechRecognition(text=<private>)", result.toString())
        assertEquals("SpeechTokenTiming(text=<private>)", tokens.first().toString())
        assertEquals(SpeechAlignmentStatus.UNAVAILABLE, SpeechRecognition(text = "").alignmentStatus)
    }

    @Test
    fun sharedInvalidTokensFailWithPythonAndSwiftReasons() {
        val rows = fixture.getValue("invalid_tokens").jsonArray.map { it.jsonObject }
        // Python SpeechProviderError.category for each fixture row, in order.
        val categories = listOf(
            "token_start_ms_out_of_range",
            "token_end_ms_out_of_range",
            "token_confidence_ppm_out_of_range",
            "token_time_range_invalid",
        )
        assertEquals(categories.size, rows.size)
        for ((patch, category) in rows.zip(categories)) {
            refused(SpeechContractError.INVALID_HYPOTHESIS, category, patch.toString()) { token(tokenRows[0] + patch) }
        }
    }

    @Test
    fun sharedProvenanceGrammarMatchesPythonAndSwift() {
        val engine = engine(provenance)
        assertEquals("synthetic", engine.providerId)
        assertEquals("en-US", engine.language)
        val rows = fixture.getValue("invalid_provenance").jsonArray.map { it.jsonObject }
        val categories = mapOf(
            "provider_id" to "provider_id_invalid",
            "provider_revision" to "provider_revision_invalid",
            "model_fingerprint" to "provider_model_fingerprint_invalid",
            "language" to "provider_language_invalid",
        )
        assertEquals(categories.keys, rows.map { it.getValue("field").jsonPrimitive.content }.toSet())
        for (row in rows) {
            val field = row.getValue("field").jsonPrimitive.content
            refused(SpeechContractError.UNVERIFIED_ARTIFACT, categories.getValue(field), row.toString()) {
                engine(provenance + (field to row.getValue("value")))
            }
        }
    }

    @Test
    fun tokenOrderAndRecognitionBoundsFailClosed() {
        val (first, second) = tokenRows.map(::token)
        val text = segment.getValue("text").jsonPrimitive.content
        val invalid = SpeechContractError.INVALID_HYPOTHESIS
        fun copy(index: Int = 1, start: Long = 650, end: Long = 900, ppm: Int = 250_000, token: String = " x") =
            SpeechTokenTiming(index, start, end, ppm, token)
        refused(invalid, "token_times_unsorted", "reversed") { SpeechRecognition(text, listOf(second, first)) }
        refused(invalid, "token_times_unsorted", "repeated index") { SpeechRecognition(text, listOf(first, copy(index = 0))) }
        refused(invalid, "token_times_unsorted", "start regresses") { SpeechRecognition(text, listOf(first, copy(start = 499))) }
        refused(invalid, "token_times_unsorted", "end regresses") { SpeechRecognition(text, listOf(first, copy(start = 500, end = 600))) }
        refused(invalid, "token_index_out_of_range", "index") { copy(index = 4097) }
        refused(invalid, "token_index_out_of_range", "negative index") { copy(index = -1) }
        refused(invalid, "token_confidence_ppm_out_of_range", "negative ppm") { copy(ppm = -1) }
        refused(invalid, "token_text_invalid", "token text") { copy(token = "x".repeat(8193)) }
        refused(invalid, "hypothesis_text_invalid", "text") { SpeechRecognition("x".repeat(8193)) }
        refused(invalid, "token_times_out_of_range", "count") {
            SpeechRecognition(text, List(4097) { copy(index = it.coerceAtMost(4096), start = 0, end = 0) })
        }
        refused(invalid, "decoder_confidence_ppm_out_of_range", "decoder ppm") {
            SpeechRecognition(text, decoderConfidencePpm = 1_000_001)
        }
        refused(invalid, "alignment_status_invalid", "aligned without tokens") {
            SpeechRecognition(text, alignmentStatus = SpeechAlignmentStatus.ALIGNED)
        }
        refused(invalid, "alignment_status_invalid", "unaligned tokens") {
            SpeechRecognition(text, listOf(first), alignmentStatus = SpeechAlignmentStatus.OUTSIDE_WINDOW)
        }
        // Lengths count Unicode scalars, so 8192 supplementary characters still fit.
        val emoji = "😀".repeat(8192)
        assertEquals(emoji, SpeechRecognition(emoji).text)
        assertEquals(emoji, SpeechTokenTiming(0, 0, 86_400_000, 1_000_000, emoji).text)
        assertEquals(4096, SpeechTokenTiming(4096, 0, 0, 0, "").tokenIndex)
    }

    @Test
    fun publicTypesCarryNoAudioPathOrEndpointFields() {
        val forbidden = setOf("audio", "samples", "pcm", "path", "file", "url", "uri", "endpoint", "buffer")
        for (type in listOf(SpeechTokenTiming::class.java, SpeechRecognition::class.java, SpeechEngineProvenance::class.java)) {
            assertFalse(type.declaredFields.any { field -> forbidden.any { field.name.lowercase().contains(it) } }, type.simpleName)
        }
    }
}
