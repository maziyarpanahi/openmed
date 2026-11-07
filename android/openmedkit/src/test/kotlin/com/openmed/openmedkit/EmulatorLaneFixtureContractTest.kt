package com.openmed.openmedkit

import com.openmed.openmedkit.onnx.TokenPrediction
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Checks the emulator-lane fixture against the real Kotlin decoding path.
 *
 * The Python generator derives the expected `analyzeText` spans from ONNX
 * Runtime token predictions by mirroring OpenMedKit's aggregation rules. This
 * test feeds the same token predictions through `aggregateTokenPredictions`,
 * the decoder, span repair and the merger, so a drift between the Python
 * mirror and the Kotlin facade fails on the JVM before the emulator runs.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class EmulatorLaneFixtureContractTest {
    @Test
    fun facadeDecodingReproducesTheExpectedSpans() = runTest {
        val fixture = loadFixture()
        val threshold = fixture.getValue("contract").jsonObject.float("confidence_threshold")
        val labels = fixture.getValue("model").jsonObject.getValue("labels").jsonArray
            .map { it.jsonPrimitive.content }
        val cases = fixture.getValue("cases").jsonArray.map { it.jsonObject }
        assertTrue(cases.isNotEmpty(), "fixture has no cases")

        for (case in cases) {
            val caseId = case.getValue("id").jsonPrimitive.content
            val text = case.getValue("text").jsonPrimitive.content
            val predictions = case.getValue("tokens").jsonArray.map { it.jsonObject }
                .filterNot { it.getValue("special").jsonPrimitive.boolean }
                .map { token ->
                    val labelId = token.int("label_id")
                    TokenPrediction(
                        labelId = labelId,
                        label = labels[labelId],
                        score = token.float("score"),
                        startOffset = token.int("start"),
                        endOffset = token.int("end"),
                    )
                }
            val expected = case.getValue("spans").jsonArray.map { it.jsonObject }
            val kit = OpenMedKit(classifier = FixtureClassifier(predictions))

            for ((check, actual) in listOf(
                "analyze_text" to kit.analyzeText(text, threshold),
                "extract_pii" to kit.extractPii(text, threshold),
            )) {
                assertEquals(
                    expected.map { Triple(it.string("label"), it.int("start"), it.int("end")) },
                    actual.map { Triple(it.label, it.start, it.end) },
                    "$caseId $check spans",
                )
                actual.zip(expected).forEach { (prediction, reference) ->
                    assertTrue(
                        abs(prediction.confidence - reference.float("confidence")) <= 1e-5f,
                        "$caseId $check confidence",
                    )
                }
            }
        }
    }

    @Test
    fun laneTextLeavesUtf16AndScalarOffsetsApart() {
        // At least one case must contain a supplementary-plane character, so
        // the device lane distinguishes scalar offsets from UTF-16 indices.
        val texts = loadFixture().getValue("cases").jsonArray
            .map { it.jsonObject.getValue("text").jsonPrimitive.content }

        assertTrue(
            texts.any { UnicodeOffsetContract.scalarLength(it) != it.length },
            "no lane case contains a supplementary-plane character",
        )
    }

    private class FixtureClassifier(
        private val predictions: List<TokenPrediction>,
    ) : OnnxTokenClassifier {
        override suspend fun predict(text: String): List<TokenClassificationPrediction> =
            aggregateTokenPredictions(text, predictions)
    }

    private fun loadFixture(): JsonObject {
        val moduleDirectory = requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
            "openmedkit.moduleDirectory test property is missing"
        }
        val fixture = File(moduleDirectory, "src/androidTest/assets/emulator-lane/expected.json")
        return Json.parseToJsonElement(fixture.readText()).jsonObject
    }

    private fun JsonObject.string(key: String): String = getValue(key).jsonPrimitive.content

    private fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int

    private fun JsonObject.float(key: String): Float = getValue(key).jsonPrimitive.float
}
