package com.openmed.openmedkit.onnx

import java.nio.file.Files
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.withContext
import kotlinx.coroutines.test.runTest
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class OnnxTokenClassifierTest {
    @Test
    fun loadsId2LabelJsonWithStringKeys() {
        val id2Label = Files.createTempFile("id2label", ".json").toFile()
        try {
            id2Label.writeText("""{"0":"O","1":"B-NAME","2":"I-DATE"}""")

            assertEquals(
                mapOf(0 to "O", 1 to "B-NAME", 2 to "I-DATE"),
                OnnxTokenClassifier.loadId2Label(id2Label),
            )
        } finally {
            id2Label.delete()
        }
    }

    @Test
    fun runBuildsInt64TensorsAndDecodesPredictions() = runTest {
        val session = RecordingSession(
            mapOf(
                OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                    arrayOf(
                        floatArrayOf(7f, 0f, -1f),
                        floatArrayOf(0f, 4f, 1f),
                        floatArrayOf(0f, 1f, 3f),
                        floatArrayOf(8f, 0f, -2f),
                    )
                )
            )
        )
        val classifier = classifier(session, TensorElementType.INT64)

        val predictions = classifier.run(
            inputIds = intArrayOf(101, 11, 22, 102),
            attentionMask = intArrayOf(1, 1, 1, 1),
            offsets = listOf(
                TokenOffset(0, 0),
                TokenOffset(0, 4),
                TokenOffset(5, 10),
                TokenOffset(0, 0),
            ),
        )

        assertEquals(2, predictions.size)
        assertEquals(
            TokenPrediction(
                labelId = 1,
                label = "B-NAME",
                score = softmaxScore(floatArrayOf(0f, 4f, 1f), 1),
                startOffset = 0,
                endOffset = 4,
            ),
            predictions[0],
        )
        assertEquals(2, predictions[1].labelId)
        assertEquals("I-DATE", predictions[1].label)
        assertClose(softmaxScore(floatArrayOf(0f, 1f, 3f), 2), predictions[1].score)
        assertEquals(5, predictions[1].startOffset)
        assertEquals(10, predictions[1].endOffset)

        assertEquals(1, session.runCount)
        assertContentEquals(longArrayOf(1, 4), session.inputIdsShape)
        assertContentEquals(longArrayOf(1, 4), session.attentionMaskShape)
        assertEquals(TensorElementType.INT64, session.inputIdsType)
        assertEquals(TensorElementType.INT64, session.attentionMaskType)
        assertEquals(listOf(101L, 11L, 22L, 102L), session.inputIdsValues)
        assertEquals(listOf(1L, 1L, 1L, 1L), session.attentionMaskValues)
    }

    @Test
    fun runCanBuildInt32InputTensors() = runTest {
        val session = RecordingSession(
            mapOf(
                OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                    arrayOf(
                        floatArrayOf(0f, 5f),
                        floatArrayOf(3f, 0f),
                    )
                )
            )
        )
        val classifier = classifier(session, TensorElementType.INT32)

        val predictions = classifier.run(
            inputIds = intArrayOf(42, 43),
            attentionMask = intArrayOf(1, 0),
            offsets = listOf(TokenOffset(1, 3), TokenOffset(3, 5)),
        )

        assertEquals(2, predictions.size)
        assertEquals(TensorElementType.INT32, session.inputIdsType)
        assertEquals(TensorElementType.INT32, session.attentionMaskType)
        assertEquals(listOf(42, 43), session.inputIdsValues)
        assertEquals(listOf(1, 0), session.attentionMaskValues)
    }

    @Test
    fun runSuppliesZeroTokenTypeIdsWhenTheGraphRequiresThem() = runTest {
        val session = RecordingSession(
            outputs = mapOf(
                OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                    arrayOf(
                        floatArrayOf(0f, 1f),
                        floatArrayOf(1f, 0f),
                    )
                )
            ),
            inputNames = setOf(
                OnnxTokenClassifier.INPUT_IDS_NAME,
                OnnxTokenClassifier.ATTENTION_MASK_NAME,
                OnnxTokenClassifier.TOKEN_TYPE_IDS_NAME,
            ),
        )
        val classifier = classifier(session)

        classifier.run(
            inputIds = intArrayOf(101, 102),
            attentionMask = intArrayOf(1, 1),
            offsets = listOf(TokenOffset(0, 1), TokenOffset(1, 2)),
        )

        assertEquals(listOf(0L, 0L), session.tokenTypeIdsValues)
    }

    @Test
    fun missingLogitsOutputRaisesTypedError() = runTest {
        val session = RecordingSession(emptyMap())
        val classifier = classifier(session)

        val error = assertFailsWith<InferenceError.MissingOutput> {
            classifier.run(
                inputIds = intArrayOf(1),
                attentionMask = intArrayOf(1),
                offsets = listOf(TokenOffset(0, 1)),
            )
        }

        assertEquals("logits", error.outputName)
        assertTrue(error.message.orEmpty().contains("logits"))
    }

    @Test
    fun runChecksCancellationBeforeInvokingSession() = runTest {
        val session = RecordingSession(
            mapOf(
                OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                    arrayOf(floatArrayOf(0f, 1f))
                )
            )
        )
        val classifier = classifier(session)
        val canceledJob = Job().also { it.cancel() }

        assertFailsWith<CancellationException> {
            withContext(canceledJob) {
                classifier.run(
                    inputIds = intArrayOf(1),
                    attentionMask = intArrayOf(1),
                    offsets = listOf(TokenOffset(0, 1)),
                )
            }
        }

        assertEquals(0, session.runCount)
    }

    @Test
    fun closeDisposesSessionOnce() {
        val session = RecordingSession(
            mapOf(
                OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                    arrayOf(floatArrayOf(0f, 1f))
                )
            )
        )
        val classifier = classifier(session)

        classifier.close()
        classifier.close()

        assertTrue(session.closed)
        assertEquals(1, session.closeCount)
    }

    @Test
    fun invalidInputLengthsRaiseTypedError() = runTest {
        val classifier = classifier(
            RecordingSession(
                mapOf(
                    OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                        arrayOf(floatArrayOf(0f, 1f))
                    )
                )
            )
        )

        assertIs<InferenceError.InvalidInput>(
            assertFailsWith<InferenceError> {
                classifier.run(
                    inputIds = intArrayOf(1),
                    attentionMask = intArrayOf(1, 1),
                    offsets = listOf(TokenOffset(0, 1)),
                )
            }
        )
    }

    private fun classifier(
        session: RecordingSession,
        tensorElementType: TensorElementType = TensorElementType.INT64,
    ) = OnnxTokenClassifier(
        environment = null,
        session = session,
        id2Label = mapOf(0 to "O", 1 to "B-NAME", 2 to "I-DATE"),
        inputTensorType = tensorElementType,
        ownsEnvironment = false,
    )

    private fun softmaxScore(logits: FloatArray, labelId: Int): Float {
        val maxLogit = logits.max()
        val denominator = logits.sumOf { exp((it - maxLogit).toDouble()) }
        return (exp((logits[labelId] - maxLogit).toDouble()) / denominator).toFloat()
    }

    private fun assertClose(expected: Float, actual: Float) {
        assertTrue(abs(expected - actual) < 0.000001f, "expected=$expected actual=$actual")
    }

    private class RecordingSession(
        private val outputs: Map<String, Any?>,
        override val inputNames: Set<String> = setOf(
            OnnxTokenClassifier.INPUT_IDS_NAME,
            OnnxTokenClassifier.ATTENTION_MASK_NAME,
        ),
    ) : TokenClassificationSession {
        var runCount = 0
            private set
        var closeCount = 0
            private set
        var closed = false
            private set

        var inputIdsShape: LongArray = longArrayOf()
            private set
        var attentionMaskShape: LongArray = longArrayOf()
            private set
        var inputIdsType: TensorElementType? = null
            private set
        var attentionMaskType: TensorElementType? = null
            private set
        var inputIdsValues: List<Any> = emptyList()
            private set
        var attentionMaskValues: List<Any> = emptyList()
            private set
        var tokenTypeIdsValues: List<Any> = emptyList()
            private set

        override fun run(inputs: Map<String, TokenInputTensor>): Map<String, Any?> {
            assertFalse(closed)
            runCount += 1

            val inputIds = inputs.getValue(OnnxTokenClassifier.INPUT_IDS_NAME)
            val attentionMask = inputs.getValue(OnnxTokenClassifier.ATTENTION_MASK_NAME)
            inputIds.capture().also { capture ->
                inputIdsShape = capture.shape
                inputIdsType = capture.type
                inputIdsValues = capture.values
            }
            attentionMask.capture().also { capture ->
                attentionMaskShape = capture.shape
                attentionMaskType = capture.type
                attentionMaskValues = capture.values
            }
            inputs[OnnxTokenClassifier.TOKEN_TYPE_IDS_NAME]?.capture()?.also { capture ->
                tokenTypeIdsValues = capture.values
            }
            return outputs
        }

        override fun close() {
            closeCount += 1
            closed = true
        }
    }
}

private data class TensorCapture(
    val shape: LongArray,
    val type: TensorElementType,
    val values: List<Any>,
)

private fun TokenInputTensor.capture(): TensorCapture {
    return TensorCapture(
        shape = shape,
        type = elementType,
        values = when (elementType) {
            TensorElementType.INT64 -> values.map { it.toLong() }
            TensorElementType.INT32 -> values.toList()
        },
    )
}
