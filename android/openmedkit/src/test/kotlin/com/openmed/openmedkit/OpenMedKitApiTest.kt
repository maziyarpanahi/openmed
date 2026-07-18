package com.openmed.openmedkit

import com.openmed.openmedkit.onnx.TokenPrediction
import com.openmed.openmedkit.policy.PolicyAction
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

class OpenMedKitApiTest {
    @Test
    fun aggregatesBioTokensAcrossWhitespace() {
        val entities = aggregateTokenPredictions(
            text = "Alice Nguyen",
            predictions = listOf(
                TokenPrediction(1, "B-PERSON", 0.99f, 0, 5),
                TokenPrediction(2, "E-PERSON", 0.97f, 6, 12),
            ),
        )

        assertEquals(
            listOf(
                TokenClassificationPrediction(
                    label = "PERSON",
                    text = "Alice Nguyen",
                    confidence = 0.98f,
                    start = 0,
                    end = 12,
                ),
            ),
            entities,
        )
    }

    @Test
    fun analyzeTextFiltersByConfidenceThreshold() = runTest {
        val kit = OpenMedKit(
            classifier = StaticClassifier(
                TokenClassificationPrediction("PERSON", "Jane", 0.94f, 8, 12),
                TokenClassificationPrediction("DATE", "2026-01-01", 0.42f, 16, 26),
            ),
        )

        val entities = kit.analyzeText("Patient Jane on 2026-01-01", confidenceThreshold = 0.5f)

        assertEquals(listOf("PERSON"), entities.map { it.label })
        assertEquals("Jane", entities.single().text)
    }

    @Test
    fun extractPiiRunsDecoderRepairAndMergerFlow() = runTest {
        val kit = OpenMedKit(
            classifier = StaticClassifier(
                TokenClassificationPrediction("PERSON", "Jane", 0.94f, 8, 12),
                TokenClassificationPrediction("PERSON", "Jane Doe", 0.90f, 8, 16),
            ),
        )

        val entities = kit.extractPii("Patient Jane Doe arrived", useSmartMerging = true)

        assertEquals(1, entities.size)
        assertEquals(EntityPrediction("PERSON", "Jane Doe", 0.90f, 8, 16), entities.single())
    }

    @Test
    fun extractPiiChunkedOffsetsReferenceOriginalTextAndDeduplicateOverlaps() = runTest {
        val text = "alpha beta Jane Doe gamma Jane Doe omega"
        val kit = OpenMedKit(classifier = RegexClassifier("Jane Doe", "PERSON"))

        val entities = kit.extractPiiChunked(
            text = text,
            chunkTokenLimit = 4,
            tokenOverlap = 2,
        )
        val positionalEntities = kit.extractPiiChunked(text, 4, 2)

        assertEquals(
            listOf(
                EntityPrediction("PERSON", "Jane Doe", 0.95f, 11, 19),
                EntityPrediction("PERSON", "Jane Doe", 0.95f, 26, 34),
            ),
            entities,
        )
        assertEquals(entities, positionalEntities)
        assertEquals("Jane Doe", text.substring(entities[0].start, entities[0].end))
        assertEquals("Jane Doe", text.substring(entities[1].start, entities[1].end))
    }

    @Test
    fun deidentifyUsesDefaultHipaaPolicyWhenNoneIsGiven() = runTest {
        val text = "Patient Jane has asthma"
        val kit = OpenMedKit(
            classifier = StaticClassifier(
                TokenClassificationPrediction("PERSON", "Jane", 0.99f, 8, 12),
                TokenClassificationPrediction("CONDITION", "asthma", 0.99f, 17, 23),
            ),
        )

        val result = kit.deidentify(text)

        assertEquals("hipaa_safe_harbor", result.policyName)
        assertEquals("Patient [PERSON] has [CONDITION]", result.redactedText)
        assertEquals(listOf(PolicyAction.MASK, PolicyAction.MASK), result.actions.map { it.action })
    }

    @Test
    fun deidentifySelectsActionsFromPolicyProfile() = runTest {
        val text = "Patient Jane has asthma"
        val kit = OpenMedKit(
            classifier = StaticClassifier(
                TokenClassificationPrediction("PERSON", "Jane", 0.99f, 8, 12),
                TokenClassificationPrediction("CONDITION", "asthma", 0.99f, 17, 23),
            ),
        )

        val result = kit.deidentify(text, policy = "research_limited_dataset")

        assertEquals("research_limited_dataset", result.policyName)
        assertEquals("Patient [PERSON] has asthma", result.redactedText)
        assertEquals(listOf(PolicyAction.MASK, PolicyAction.KEEP), result.actions.map { it.action })
    }

    @Test
    fun policyLoaderExposesTheSixBundledProfiles() {
        assertEquals(
            listOf(
                "hipaa_safe_harbor",
                "hipaa_expert_review_assist",
                "gdpr_pseudonymization",
                "research_limited_dataset",
                "strict_no_leak",
                "clinical_minimal_redaction",
            ),
            PolicyProfiles.bundledProfileNames,
        )
        assertEquals(
            PolicyAction.REPLACE,
            PolicyProfiles.load("gdpr").actionFor("B-PERSON"),
        )
        assertFailsWith<IllegalArgumentException> {
            PolicyProfiles.load("canada_pipeda")
        }
    }

    @Test
    fun backendConfigMirrorsOfflineModelAssets() {
        val backend = OpenMedBackend(
            modelDirectory = File("/models/openmed"),
            id2Label = mapOf(0 to "O", 1 to "PERSON"),
        )

        assertEquals(File("/models/openmed/model.onnx"), backend.modelFile)
        assertEquals(File("/models/openmed/tokenizer.json"), backend.tokenizerJson)
        assertEquals("PERSON", backend.id2Label[1])
    }

    private class StaticClassifier(
        vararg predictions: TokenClassificationPrediction,
    ) : OnnxTokenClassifier {
        private val predictions = predictions.toList()

        override suspend fun predict(text: String): List<TokenClassificationPrediction> = predictions
    }

    private class RegexClassifier(
        private val needle: String,
        private val label: String,
    ) : OnnxTokenClassifier {
        override suspend fun predict(text: String): List<TokenClassificationPrediction> {
            val predictions = mutableListOf<TokenClassificationPrediction>()
            var start = text.indexOf(needle)
            while (start >= 0) {
                val end = start + needle.length
                predictions += TokenClassificationPrediction(
                    label = label,
                    text = text.substring(start, end),
                    confidence = 0.95f,
                    start = start,
                    end = end,
                )
                start = text.indexOf(needle, startIndex = start + 1)
            }
            return predictions
        }
    }
}
