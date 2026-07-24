package com.openmed.openmedkit

import com.openmed.openmedkit.policy.PolicyAction
import com.openmed.openmedkit.policy.PolicyProfile
import com.openmed.openmedkit.segmentation.IcuTextSegmenter
import java.security.MessageDigest

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
 * Input and action offsets follow the shared Unicode scalar contract. Native
 * UTF-16 ranges are derived only while slicing or replacing the source text.
 */
class DeidentifyEngine(
    private val segmenter: IcuTextSegmenter = IcuTextSegmenter(),
) {
    fun deidentify(
        text: String,
        entities: List<EntityPrediction>,
        policy: PolicyProfile,
    ): PolicyDeidentificationResult {
        val candidates = deidentificationCandidates(entities, text)
        val actions = candidates.map { entity ->
            val canonicalLabel = policy.canonicalLabel(entity.label)
            val action = policy.actionFor(entity.label)
            val original = UnicodeOffsetContract.substring(
                text,
                entity.start,
                entity.end,
            )
            DeidentifiedSpanAction(
                label = entity.label,
                canonicalLabel = canonicalLabel,
                action = action,
                start = entity.start,
                end = entity.end,
                confidence = entity.confidence,
                replacement = replacementText(action, canonicalLabel, original),
            )
        }

        var redacted = text
        val replacements = actions.mapNotNull { record ->
            val replacement = record.replacement ?: return@mapNotNull null
            UnicodeOffsetContract.utf16Span(
                text,
                record.start,
                record.end,
            ) to replacement
        }
        for ((range, replacement) in replacements.asReversed()) {
            redacted = redacted.replaceRange(range.start, range.end, replacement)
        }

        return PolicyDeidentificationResult(
            redactedText = redacted,
            policyName = policy.name,
            actions = actions,
        )
    }

    private fun deidentificationCandidates(
        entities: List<EntityPrediction>,
        text: String,
    ): List<EntityPrediction> {
        val sorted = entities
            .map { it.snappedToGraphemeBoundaries(text, segmenter) }
            .filter { it.end > it.start }
            .sortedWith(
                compareBy<EntityPrediction> { it.start }
                    .thenByDescending { it.end - it.start }
                    .thenByDescending { it.confidence },
            )

        val selected = mutableListOf<EntityPrediction>()
        for (entity in sorted) {
            if (selected.none { entity.start < it.end && entity.end > it.start }) {
                selected += entity
            }
        }
        return selected.sortedWith(compareBy<EntityPrediction> { it.start }.thenBy { it.end })
    }

    private fun replacementText(
        action: PolicyAction,
        canonicalLabel: String,
        original: String,
    ): String? = when (action.redactionEquivalent) {
        PolicyAction.KEEP -> null
        PolicyAction.MASK,
        PolicyAction.REDACT,
        -> "[$canonicalLabel]"
        PolicyAction.REPLACE -> "[${canonicalLabel}_REPLACED]"
        PolicyAction.REMOVE -> ""
        PolicyAction.HASH -> "${canonicalLabel}_${stableHash(original)}"
    }

    private fun stableHash(text: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(text.toByteArray())
        return digest.take(12).joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
