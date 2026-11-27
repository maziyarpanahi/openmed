package org.openmed.r8smoke

import com.openmed.openmedkit.DeidentifiedSpanAction
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.PolicyDeidentificationResult
import com.openmed.openmedkit.policy.PolicyAction
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class SmokeEvaluationTest {
    private val text = "Patient Avery Quill visited the clinic."
    private val case = SmokeCase(
        text = text,
        entities = listOf(SmokeSpan("NAME", 8, 19)),
        chunkTokenLimit = 4,
        tokenOverlap = 1,
        policy = "hipaa_safe_harbor",
        canonicalLabels = listOf("PERSON"),
    )

    @Test
    fun matchingEntitiesPassWithLabelsAndOffsetsOnly() {
        val check = evaluateEntities(
            "extract_pii",
            listOf(EntityPrediction("NAME", "Avery Quill", 0.96f, 8, 19)),
            case.entities,
        )

        assertEquals(SmokeStatus.PASS, check.status)
        assertEquals(
            "openmed_r8_smoke check=extract_pii status=PASS count=1 spans=NAME:8-19",
            check.format(),
        )
    }

    @Test
    fun shiftedOrMissingEntitiesFail() {
        val shifted = evaluateEntities(
            "extract_pii",
            listOf(EntityPrediction("NAME", "Avery", 0.96f, 8, 13)),
            case.entities,
        )
        val missing = evaluateEntities("extract_pii", emptyList(), case.entities)

        assertEquals(SmokeStatus.FAIL, shifted.status)
        assertEquals("spans_mismatch", shifted.reason)
        assertEquals(SmokeStatus.FAIL, missing.status)
        assertEquals(0, missing.count)
    }

    @Test
    fun unreportableModelLabelsAreReplaced() {
        val check = evaluateEntities(
            "extract_pii",
            listOf(EntityPrediction("Avery Quill", "Avery Quill", 0.96f, 8, 19)),
            case.entities,
        )

        assertFalse(check.format().contains("Avery"))
        assertEquals("unreportable_label", check.spans.single().label)
    }

    @Test
    fun redactionPassesOnlyWhenTheSurfaceIsGone() {
        val redacted = PolicyDeidentificationResult(
            redactedText = "Patient [PERSON] visited the clinic.",
            policyName = "hipaa_safe_harbor",
            actions = listOf(action(replacement = "[PERSON]")),
        )
        val leaked = redacted.copy(redactedText = text)
        val kept = redacted.copy(actions = listOf(action(replacement = null)))

        val passed = evaluateRedaction(redacted, case)
        assertEquals(SmokeStatus.PASS, passed.status)
        assertEquals(
            "openmed_r8_smoke check=deidentify status=PASS count=1 spans=PERSON:8-19",
            passed.format(),
        )
        assertEquals("surface_retained", evaluateRedaction(leaked, case).reason)
        assertFalse(evaluateRedaction(leaked, case).format().contains("Avery"))
        assertEquals("action_kept_surface", evaluateRedaction(kept, case).reason)
    }

    private fun action(replacement: String?) = DeidentifiedSpanAction(
        label = "NAME",
        canonicalLabel = "PERSON",
        action = PolicyAction.MASK,
        start = 8,
        end = 19,
        confidence = 0.96f,
        replacement = replacement,
    )
}
