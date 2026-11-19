package com.openmed.openmedkit.emulator

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import androidx.test.platform.app.InstrumentationRegistry
import com.openmed.openmedkit.DjlTelemetry
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

/**
 * Encodes the synthetic cases with the DJL tokenizer JNI library.
 *
 * Token IDs and Unicode scalar offsets must equal the Python `tokenizers`
 * reference exactly. Case `synth-lane-003` contains a character outside the
 * Basic Multilingual Plane, where scalar and UTF-16 offsets diverge.
 */
class TokenizerParityTest {
    @get:Rule
    val noNetwork = NoNetworkSocketRule()

    @Test
    fun jniTokenizerMatchesThePythonReference() {
        LaneArguments.assumeLoadable(NativeLibraries.TOKENIZER)
        val context = InstrumentationRegistry.getInstrumentation().context
        val fixture = LaneFixture.load(context)
        val modelDirectory = fixture.installModel(context)

        // OpenMedKit opts out of DJL telemetry before it creates a tokenizer;
        // this test calls DJL directly, so it does the same.
        DjlTelemetry.optOut()
        HuggingFaceTokenizer.newInstance(modelDirectory.toPath()).use { tokenizer ->
            for (case in fixture.cases) {
                LaneRecorder.check(case.id, "tokenizer_offsets") {
                    val encoding = tokenizer.encode(case.text)
                    assertEquals(
                        "token_ids_mismatch:${case.id}",
                        case.tokens.map(LaneToken::id),
                        encoding.ids.toList(),
                    )
                    val offsets = encoding.charTokenSpans.map { span ->
                        if (span == null) 0 to 0 else span.start to span.end
                    }
                    assertEquals(
                        "token_offsets_mismatch:${case.id}",
                        case.tokens.map { it.start to it.end },
                        offsets,
                    )
                    emptyList()
                }
            }
        }
    }
}
