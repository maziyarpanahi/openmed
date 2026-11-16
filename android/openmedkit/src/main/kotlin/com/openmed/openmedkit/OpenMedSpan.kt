package com.openmed.openmedkit

/**
 * On-device view of the canonical OpenMed span record (OM-027 section 4.3).
 *
 * Carries the span fields relevant to Android detection, decoding, merging,
 * and de-identification while omitting the server-only fields (doc id, text
 * hash, regulatory tags, provenance, ...). [start] and [end] are half-open
 * Unicode scalar (code point) offsets into the original text, identical to the
 * Swift `EntityPrediction` and Python `OpenMedSpan` conventions. They are not
 * Kotlin UTF-16 indices.
 *
 * @property start Inclusive Unicode scalar offset of the span start.
 * @property end Exclusive Unicode scalar offset of the span end.
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
    // Invalid fields throw OpenMedArgumentException (input_error), an
    // IllegalArgumentException whose message carries offsets but never text.
    init {
        if (start < 0) invalid("start must be a non-negative offset: $start")
        if (end < start) invalid("end must be >= start: end=$end start=$start")
        if (score !in 0.0f..1.0f) invalid("score must be in [0.0, 1.0]: $score")
        if (schemaVersion != SCHEMA_VERSION) {
            invalid("schemaVersion must be $SCHEMA_VERSION: $schemaVersion")
        }
    }

    private fun invalid(message: String): Nothing =
        throw OpenMedArgumentException(OpenMedErrorCode.INPUT_ERROR, message)

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
