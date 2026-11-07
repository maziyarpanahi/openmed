package com.openmed.openmedkit.emulator

import androidx.test.platform.app.InstrumentationRegistry
import com.openmed.openmedkit.onnx.AcceleratorConfig
import com.openmed.openmedkit.onnx.AcceleratorProvider
import com.openmed.openmedkit.onnx.AcceleratorSession
import com.openmed.openmedkit.onnx.TokenOffset
import java.io.File
import kotlin.math.abs
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

/**
 * Runs the synthetic model through ONNX Runtime Android on the CPU provider.
 *
 * Token IDs come from the fixture, so this check needs no tokenizer and still
 * runs on lanes where the tokenizer library cannot load. Per-token labels must
 * match the Python ONNX Runtime reference exactly and scores within the
 * fixture tolerance.
 */
class OnnxRuntimeParityTest {
    @get:Rule
    val noNetwork = NoNetworkSocketRule()

    @Test
    fun cpuSessionMatchesThePythonReference(): Unit = runBlocking {
        LaneArguments.assumeLoadable(NativeLibraries.ONNX_RUNTIME)
        LaneArguments.assumeLoadable(NativeLibraries.ONNX_RUNTIME_JNI)
        val context = InstrumentationRegistry.getInstrumentation().context
        val fixture = LaneFixture.load(context)
        val modelDirectory = fixture.installModel(context)

        AcceleratorSession(
            File(modelDirectory, "model.onnx"),
            File(modelDirectory, "id2label.json"),
            AcceleratorConfig.cpuOnly(),
        ).use { session ->
            assertEquals(
                "onnxruntime_provider_mismatch",
                AcceleratorProvider.CPU,
                session.selection.selectedProvider,
            )
            for (case in fixture.cases) {
                LaneRecorder.check(case.id, "onnxruntime_tokens") {
                    val predictions = session.run(
                        inputIds = case.tokens.map { it.id.toInt() }.toIntArray(),
                        attentionMask = IntArray(case.tokens.size) { 1 },
                        offsets = case.tokens.map { TokenOffset(it.start, it.end) },
                    )
                    val expected = case.tokens.filterNot(LaneToken::special)
                    assertEquals(
                        "token_count_mismatch:${case.id}",
                        expected.size,
                        predictions.size,
                    )
                    predictions.zip(expected).forEachIndexed { index, (actual, reference) ->
                        assertEquals(
                            "token_label_mismatch:${case.id}:$index",
                            reference.labelId,
                            actual.labelId,
                        )
                        assertEquals(
                            "token_offset_mismatch:${case.id}:$index",
                            reference.start to reference.end,
                            actual.startOffset to actual.endOffset,
                        )
                        val score = requireNotNull(reference.score)
                        assertTrue(
                            "token_score_out_of_tolerance:${case.id}:$index",
                            abs(actual.score - score) <= fixture.contract.tokenScoreTolerance,
                        )
                    }
                    emptyList()
                }
            }
        }
    }
}
