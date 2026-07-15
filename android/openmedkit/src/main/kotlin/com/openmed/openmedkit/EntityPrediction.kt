package com.openmed.openmedkit

import java.util.Locale

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
     * format `[label] "text" (start:end) conf=0.00`, with a two-decimal
     * confidence rendered in [Locale.ROOT] so the decimal separator is a
     * period on every device locale.
     */
    override fun toString(): String =
        "[$label] \"$text\" ($start:$end) conf=${String.format(Locale.ROOT, "%.2f", confidence)}"
}
