package com.openmed.openmedkit

import java.math.BigDecimal
import java.math.RoundingMode

/**
 * A single entity predicted by the OpenMedKit token-classification pipeline.
 *
 * Mirrors the Swift `OpenMedKit.EntityPrediction` value type and the Python
 * `OpenMedSpan` char-offset conventions: [start] and [end] are half-open
 * character offsets into the original text (`start` inclusive, `end`
 * exclusive), so `end >= start` and `end - start` is the span length.
 */
data class EntityPrediction(
    val label: String,
    val text: String,
    val confidence: Float,
    val start: Int,
    val end: Int,
) {
    /** Python-compatible entity type used by de-identification exports. */
    val entityType: String
        get() = label

    /**
     * Human-readable description matching the Swift `EntityPrediction`
     * format `[label] "text" (start:end) conf=0.00`.
     *
     * The confidence is rendered with two decimals using half-even
     * ("banker's") rounding on a period decimal separator, matching Swift's
     * `String(format: "%.2f", confidence)` (C `printf`) exactly, including
     * ties on exactly-representable halves such as `0.125 -> 0.12`.
     * Non-finite confidences are rendered without rounding so this never
     * throws.
     */
    override fun toString(): String {
        val conf =
            if (confidence.isFinite()) {
                BigDecimal(confidence.toDouble())
                    .setScale(2, RoundingMode.HALF_EVEN)
                    .toPlainString()
            } else {
                confidence.toString()
            }
        return "[$label] \"$text\" ($start:$end) conf=$conf"
    }
}
