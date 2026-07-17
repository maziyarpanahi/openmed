package com.openmed.openmedkit

/**
 * On-device view of the canonical OpenMed span record (OM-027 section 4.3).
 *
 * Carries the span fields relevant to Android detection, decoding, merging,
 * and de-identification while omitting the server-only fields (doc id, text
 * hash, regulatory tags, provenance, ...). [start] and [end] are half-open
 * character offsets into the original text (`start` inclusive, `end`
 * exclusive), identical to the Swift `EntityPrediction` and Python
 * `OpenMedSpan` conventions, so `end >= start`.
 *
 * @property start Inclusive character offset of the span start.
 * @property end Exclusive character offset of the span end.
 * @property text The exact substring covered by the span.
 * @property rawLabel The label emitted by the model (mirrors the Python
 *   `entity_type`).
 * @property canonicalLabel The canonical label; on device this defaults to
 *   [rawLabel], since canonical-label normalization is owned by the Python
 *   pipeline.
 * @property score Model confidence in `[0.0, 1.0]`.
 * @property schemaVersion On-device span schema version; must equal
 *   [SCHEMA_VERSION].
 */
data class OpenMedSpan(
    val start: Int,
    val end: Int,
    val text: String,
    val rawLabel: String,
    val canonicalLabel: String,
    val score: Float,
    val schemaVersion: Int = SCHEMA_VERSION,
) {
    init {
        require(start >= 0) { "start must be a non-negative offset: $start" }
        require(end >= start) { "end must be >= start: end=$end start=$start" }
        require(score in 0.0f..1.0f) { "score must be in [0.0, 1.0]: $score" }
        require(schemaVersion == SCHEMA_VERSION) {
            "schemaVersion must be $SCHEMA_VERSION: $schemaVersion"
        }
    }

    /**
     * Convert back to an [EntityPrediction], restoring [rawLabel] as the
     * prediction label and [score] as the confidence.
     */
    fun toEntityPrediction(): EntityPrediction =
        EntityPrediction(
            label = rawLabel,
            text = text,
            confidence = score,
            start = start,
            end = end,
        )

    companion object {
        /** On-device span schema version, mirroring Python `CURRENT_SCHEMA_VERSION`. */
        const val SCHEMA_VERSION: Int = 1

        /**
         * Build an [OpenMedSpan] from an [EntityPrediction]. The canonical
         * label defaults to the raw model [EntityPrediction.label]; pass
         * [canonicalLabel] explicitly to attach a normalized label.
         */
        fun fromEntityPrediction(
            prediction: EntityPrediction,
            canonicalLabel: String = prediction.label,
        ): OpenMedSpan =
            OpenMedSpan(
                start = prediction.start,
                end = prediction.end,
                text = prediction.text,
                rawLabel = prediction.label,
                canonicalLabel = canonicalLabel,
                score = prediction.confidence,
            )
    }
}
