package com.openmed.openmedkit.onnx

import java.nio.file.Files
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class AcceleratorFallbackTest {
    @Test
    fun deviceFarmStubSelectsQnnAndMatchesCpuWithinTolerance() = runTest {
        val coverage = ModelFamilyOperatorCoverage(
            family = "bert",
            requiredOperators = setOf("MatMul", "Gelu", "LayerNormalization"),
            supportedOperators = mapOf(
                AcceleratorProvider.QNN to setOf("MatMul", "Gelu"),
            ),
        )
        val qnnFactory = RecordingSessionFactory()
        val cpuFactory = RecordingSessionFactory()
        val labels = mapOf(0 to "O", 1 to "B-NAME")
        val qnn = AcceleratorSession.createForTesting(
            id2Label = labels,
            config = AcceleratorConfig(modelCoverage = coverage),
            availableProviders = setOf(
                AcceleratorProvider.QNN,
                AcceleratorProvider.CPU,
            ),
            sessionFactory = qnnFactory,
        )
        val cpu = AcceleratorSession.createForTesting(
            id2Label = labels,
            config = AcceleratorConfig.cpuOnly(),
            availableProviders = setOf(AcceleratorProvider.CPU),
            sessionFactory = cpuFactory,
        )

        try {
            val qnnPredictions = qnn.run(
                TEST_INPUT_IDS,
                TEST_ATTENTION_MASK,
                TEST_OFFSETS,
            )
            val cpuPredictions = cpu.run(
                TEST_INPUT_IDS,
                TEST_ATTENTION_MASK,
                TEST_OFFSETS,
            )
            val evidence = AcceleratorValidationRecord(
                latency = DeviceTierLatencyRecord(
                    deviceTier = AndroidDeviceTier.DEVICE_FARM_STUB,
                    provider = qnn.selection.selectedProvider,
                    cpuP50Milliseconds = 24.0,
                    delegateP50Milliseconds = 8.0,
                    sampleCount = 50,
                ),
                cpuSpans = cpuPredictions.toSpanSignatures(),
                delegateSpans = qnnPredictions.toSpanSignatures(),
                cpuRecall = 1.0,
                delegateRecall = 1.0,
            ).requirePassing()

            assertEquals(AcceleratorProvider.QNN, qnn.selection.selectedProvider)
            assertTrue(qnn.selection.isAccelerated)
            assertTrue(qnn.selection.operatorCoverage.usesCpuPartition)
            assertEquals(
                setOf("LayerNormalization"),
                qnn.selection.operatorCoverage.cpuFallbackOperators,
            )
            assertEquals(listOf(AcceleratorProvider.QNN), qnnFactory.createdProviders)
            assertEquals(cpuPredictions, qnnPredictions)
            assertEquals(0.0, evidence.recallDelta)
            assertEquals(3.0, evidence.latency.speedup)
            assertTrue(evidence.passed)
        } finally {
            qnn.close()
            cpu.close()
        }
    }

    @Test
    fun absentDelegatesFallBackToCpuDeterministically() = runTest {
        val factory = RecordingSessionFactory()
        val session = AcceleratorSession.createForTesting(
            id2Label = mapOf(0 to "O", 1 to "B-NAME"),
            config = AcceleratorConfig(),
            availableProviders = setOf(AcceleratorProvider.CPU),
            sessionFactory = factory,
        )

        try {
            val predictions = session.run(
                TEST_INPUT_IDS,
                TEST_ATTENTION_MASK,
                TEST_OFFSETS,
            )

            assertEquals(AcceleratorProvider.CPU, session.selection.selectedProvider)
            assertFalse(session.selection.isAccelerated)
            assertEquals(listOf(AcceleratorProvider.CPU), factory.createdProviders)
            assertEquals(
                listOf(
                    AcceleratorProviderAttempt(
                        AcceleratorProvider.QNN,
                        AcceleratorAttemptOutcome.NOT_AVAILABLE,
                    ),
                    AcceleratorProviderAttempt(
                        AcceleratorProvider.NNAPI,
                        AcceleratorAttemptOutcome.NOT_AVAILABLE,
                    ),
                    AcceleratorProviderAttempt(
                        AcceleratorProvider.CPU,
                        AcceleratorAttemptOutcome.SELECTED,
                    ),
                ),
                session.selection.attempts,
            )
            assertEquals(listOf("B-NAME"), predictions.map(TokenPrediction::label))
        } finally {
            session.close()
        }
    }

    @Test
    fun absentQnnSelectsNnapiBeforeCpu() {
        val factory = RecordingSessionFactory()
        val session = AcceleratorSession.createForTesting(
            id2Label = mapOf(0 to "O", 1 to "B-NAME"),
            config = AcceleratorConfig(),
            availableProviders = setOf(
                AcceleratorProvider.NNAPI,
                AcceleratorProvider.CPU,
            ),
            sessionFactory = factory,
        )

        try {
            assertEquals(
                AcceleratorProvider.NNAPI,
                session.selection.selectedProvider,
            )
            assertTrue(session.selection.isAccelerated)
            assertEquals(listOf(AcceleratorProvider.NNAPI), factory.createdProviders)
            assertEquals(
                listOf(
                    AcceleratorProviderAttempt(
                        AcceleratorProvider.QNN,
                        AcceleratorAttemptOutcome.NOT_AVAILABLE,
                    ),
                    AcceleratorProviderAttempt(
                        AcceleratorProvider.NNAPI,
                        AcceleratorAttemptOutcome.SELECTED,
                    ),
                ),
                session.selection.attempts,
            )
        } finally {
            session.close()
        }
    }

    @Test
    fun delegateSessionCreationFailureRetriesCpu() {
        val factory = RecordingSessionFactory(
            failingProviders = setOf(AcceleratorProvider.QNN),
        )
        val session = AcceleratorSession.createForTesting(
            id2Label = mapOf(0 to "O", 1 to "B-NAME"),
            config = AcceleratorConfig(
                preferredProviders = listOf(
                    AcceleratorProvider.QNN,
                    AcceleratorProvider.CPU,
                ),
            ),
            availableProviders = setOf(
                AcceleratorProvider.QNN,
                AcceleratorProvider.CPU,
            ),
            sessionFactory = factory,
        )

        try {
            assertEquals(AcceleratorProvider.CPU, session.selection.selectedProvider)
            assertEquals(
                listOf(AcceleratorProvider.QNN, AcceleratorProvider.CPU),
                factory.createdProviders,
            )
            assertEquals(
                AcceleratorAttemptOutcome.SESSION_CREATION_FAILED,
                session.selection.attempts.first().outcome,
            )
        } finally {
            session.close()
        }
    }

    @Test
    fun measuredZeroCoverageSkipsDelegate() {
        val factory = RecordingSessionFactory()
        val session = AcceleratorSession.createForTesting(
            id2Label = mapOf(0 to "O", 1 to "B-NAME"),
            config = AcceleratorConfig(
                preferredProviders = listOf(
                    AcceleratorProvider.QNN,
                    AcceleratorProvider.CPU,
                ),
                modelCoverage = ModelFamilyOperatorCoverage(
                    family = "roberta",
                    requiredOperators = setOf("LayerNormalization", "Gather"),
                    supportedOperators = mapOf(
                        AcceleratorProvider.QNN to emptySet(),
                    ),
                ),
            ),
            availableProviders = setOf(
                AcceleratorProvider.QNN,
                AcceleratorProvider.CPU,
            ),
            sessionFactory = factory,
        )

        try {
            assertEquals(AcceleratorProvider.CPU, session.selection.selectedProvider)
            assertEquals(listOf(AcceleratorProvider.CPU), factory.createdProviders)
            assertEquals(
                AcceleratorAttemptOutcome.NO_SUPPORTED_OPERATORS,
                session.selection.attempts.first().outcome,
            )
        } finally {
            session.close()
        }
    }

    @Test
    fun manifestLoadsFamilyAndPerArtifactOperators() {
        val directory = Files.createTempDirectory("accelerator-manifest").toFile()
        try {
            val manifest = directory.resolve("openmed-onnx.json")
            manifest.writeText(
                """
                {
                  "family": "distilbert",
                  "artifacts": [
                    {
                      "path": "model.onnx",
                      "metadata": {
                        "operators": ["Gather", "MatMul", "LayerNormalization"]
                      }
                    }
                  ]
                }
                """.trimIndent()
            )

            val coverage = ModelFamilyOperatorCoverage.fromManifest(
                manifestFile = manifest,
                supportedOperators = mapOf(
                    AcceleratorProvider.NNAPI to setOf("Gather", "MatMul"),
                ),
            )

            assertEquals("distilbert", coverage.family)
            assertEquals(
                setOf("Gather", "MatMul", "LayerNormalization"),
                coverage.requiredOperators,
            )
            assertEquals(
                setOf("Gather", "MatMul"),
                coverage.supportedOperators[AcceleratorProvider.NNAPI],
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun recallDeltaOutsideToleranceIsRejected() {
        val evidence = AcceleratorValidationRecord(
            latency = DeviceTierLatencyRecord(
                deviceTier = AndroidDeviceTier.MID_RANGE,
                provider = AcceleratorProvider.NNAPI,
                cpuP50Milliseconds = 18.0,
                delegateP50Milliseconds = 12.0,
                sampleCount = 20,
            ),
            cpuSpans = listOf(AcceleratorSpanSignature("NAME", 0, 4)),
            delegateSpans = listOf(AcceleratorSpanSignature("NAME", 0, 4)),
            cpuRecall = 0.99,
            delegateRecall = 0.98,
            maxRecallDrop = 0.005,
        )

        assertFalse(evidence.recallWithinTolerance)
        assertFailsWith<IllegalStateException> { evidence.requirePassing() }
    }

    private class RecordingSessionFactory(
        private val failingProviders: Set<AcceleratorProvider> = emptySet(),
    ) : AcceleratorTokenSessionFactory {
        val createdProviders = mutableListOf<AcceleratorProvider>()

        override fun create(provider: AcceleratorProvider): TokenClassificationSession {
            createdProviders += provider
            if (provider in failingProviders) {
                throw IllegalStateException("stub provider failure")
            }
            return StaticTokenSession()
        }
    }

    private class StaticTokenSession : TokenClassificationSession {
        override val inputNames: Set<String> = setOf(
            OnnxTokenClassifier.INPUT_IDS_NAME,
            OnnxTokenClassifier.ATTENTION_MASK_NAME,
        )

        override fun run(inputs: Map<String, TokenInputTensor>): Map<String, Any?> {
            val sequenceLength = inputs.getValue(OnnxTokenClassifier.INPUT_IDS_NAME).values.size
            return mapOf(
                OnnxTokenClassifier.LOGITS_NAME to arrayOf(
                    Array(sequenceLength) { index ->
                        if (index == 1) {
                            floatArrayOf(0f, 5f)
                        } else {
                            floatArrayOf(5f, 0f)
                        }
                    }
                )
            )
        }

        override fun close() = Unit
    }

    private companion object {
        val TEST_INPUT_IDS = intArrayOf(101, 42, 102)
        val TEST_ATTENTION_MASK = intArrayOf(1, 1, 1)
        val TEST_OFFSETS = listOf(
            TokenOffset(0, 0),
            TokenOffset(4, 8),
            TokenOffset(0, 0),
        )
    }
}

private fun List<TokenPrediction>.toSpanSignatures(): List<AcceleratorSpanSignature> =
    map { prediction ->
        AcceleratorSpanSignature(
            label = prediction.label,
            startOffset = prediction.startOffset,
            endOffset = prediction.endOffset,
        )
    }
