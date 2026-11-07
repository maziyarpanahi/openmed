package com.openmed.openmedkit

import com.openmed.openmedkit.policy.PolicyAction
import com.openmed.openmedkit.policy.PolicyDetection
import com.openmed.openmedkit.policy.PolicyEngine
import com.openmed.openmedkit.policy.PolicyProfile
import com.openmed.openmedkit.segmentation.IcuTextSegmenter

/**
 * A single action applied to a detected span during policy de-identification.
 */
data class DeidentifiedSpanAction(
    val label: String,
    val canonicalLabel: String,
    val action: PolicyAction,
    val start: Int,
    val end: Int,
    val confidence: Float,
    val replacement: String?,
)

/**
 * Result of policy-driven de-identification.
 */
data class PolicyDeidentificationResult(
    val redactedText: String,
    val policyName: String,
    val actions: List<DeidentifiedSpanAction>,
)

/**
 * De-identification engine driven by bundled policy profile actions.
 *
 * Input and action offsets follow the shared Unicode scalar contract. Entities
 * are first snapped to grapheme boundaries (a platform pre-step); the shared
 * portable [PolicyEngine] then applies label normalization, thresholds,
 * overlap resolution, and rendering, so the output matches every other
 * OpenMedKit surface for the same spans.
 */
class DeidentifyEngine(
    private val segmenter: IcuTextSegmenter = IcuTextSegmenter(),
) {
    fun deidentify(
        text: String,
        entities: List<EntityPrediction>,
        policy: PolicyProfile,
        surrogateKey: ByteArray? = null,
    ): PolicyDeidentificationResult {
        val length = UnicodeOffsetContract.scalarLength(text)
        val detections = entities
            .map { it.snappedToGraphemeBoundaries(text, segmenter) }
            .filter { it.start >= 0 && it.end > it.start && it.end <= length }
            .map { entity ->
                PolicyDetection(
                    label = entity.label,
                    start = entity.start,
                    end = entity.end,
                    confidencePpm = KitParity.confidencePpm(entity.confidence),
                )
            }
        val outcome = PolicyEngine.decide(policy, text, detections, surrogateKey = surrogateKey)
        val actions = outcome.result.spans.map { span ->
            DeidentifiedSpanAction(
                label = span.label,
                canonicalLabel = span.canonicalLabel,
                action = PolicyAction.fromWireName(span.action),
                start = span.start,
                end = span.end,
                confidence = (span.confidencePpm ?: 0) / KitParity.CONFIDENCE_SCALE.toFloat(),
                replacement = span.replacement,
            )
        }
        return PolicyDeidentificationResult(
            redactedText = outcome.result.deidentifiedText,
            policyName = policy.name,
            actions = actions,
        )
    }
}
