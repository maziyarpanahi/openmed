package com.openmed.openmedkit

/**
 * A single entity predicted by the OpenMedKit token-classification pipeline.
 *
 * Offsets are character offsets into the source text and use an exclusive end.
 */
data class EntityPrediction(
    val label: String,
    val text: String,
    val confidence: Float,
    val start: Int,
    val end: Int,
) {
    val entityType: String
        get() = label
}
