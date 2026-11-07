package com.openmed.openmedkit.emulator

import androidx.test.platform.app.InstrumentationRegistry
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OpenMedKit
import kotlin.math.abs
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

/**
 * End-to-end check of the public facade on a real device runtime.
 *
 * `OpenMedKit.fromDirectory` loads the synthetic model with ONNX Runtime
 * Android, its default execution-provider selection and the DJL tokenizer.
 * `analyzeText` and `extractPii` must return the spans derived from the Python
 * ONNX Runtime reference: identical labels and Unicode scalar offsets, and
 * confidences within the fixture tolerance. No network socket may be opened.
 */
class OpenMedKitFromDirectoryTest {
    @get:Rule
    val noNetwork = NoNetworkSocketRule()

    @Test
    fun fromDirectoryMatchesThePythonReferenceSpans(): Unit = runBlocking {
        NativeLibraries.required.forEach(LaneArguments::assumeLoadable)
        val context = InstrumentationRegistry.getInstrumentation().context
        val fixture = LaneFixture.load(context)
        val modelDirectory = fixture.installModel(context)
        val threshold = fixture.contract.confidenceThreshold
        val tolerance = fixture.contract.spanConfidenceTolerance

        OpenMedKit.fromDirectory(modelDirectory, variant = "fp32").use { kit ->
            for (case in fixture.cases) {
                LaneRecorder.check(case.id, "facade_analyze_text") {
                    val spans = kit.analyzeText(case.text, threshold)
                    assertSpans(case, "analyze_text", spans, tolerance)
                }
                LaneRecorder.check(case.id, "facade_extract_pii") {
                    val spans = kit.extractPii(case.text, threshold)
                    assertSpans(case, "extract_pii", spans, tolerance)
                }
            }
        }
    }

    private fun assertSpans(
        case: LaneCase,
        check: String,
        actual: List<EntityPrediction>,
        tolerance: Float,
    ): List<LaneSpan> {
        // Compare value-free signatures so a failure message never contains
        // span surface text.
        val actualSpans = actual.map { LaneSpan(it.label, it.start, it.end) }
        assertEquals(
            "span_mismatch:${case.id}:$check",
            case.spans.map(ExpectedSpan::span),
            actualSpans,
        )
        actual.zip(case.spans).forEachIndexed { index, (prediction, expected) ->
            assertTrue(
                "span_confidence_out_of_tolerance:${case.id}:$check:$index",
                abs(prediction.confidence - expected.confidence) <= tolerance,
            )
        }
        return actualSpans
    }
}
