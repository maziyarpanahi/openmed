package com.openmed.openmedkit

import com.openmed.openmedkit.segmentation.IcuTextSegmenter
import java.math.BigDecimal
import java.math.RoundingMode

/**
 * A single entity predicted by the OpenMedKit token-classification pipeline.
 *
 * Mirrors the Swift `OpenMedKit.EntityPrediction` value type and the Python
 * `OpenMedSpan` conventions: [start] and [end] are half-open Unicode scalar
 * (code point) offsets into the exact original text. They are never Kotlin
 * UTF-16 indices.
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
     * Return the native Kotlin UTF-16 range represented by this entity.
     */
    fun utf16SpanIn(source: String): Utf16Span? {
        val scalarLength = UnicodeOffsetContract.scalarLength(source)
        if (start !in 0..scalarLength || end !in start..scalarLength) {
            return null
        }
        return UnicodeOffsetContract.utf16Span(source, start, end)
    }

    /**
     * Return a copy whose scalar offsets enclose complete grapheme clusters.
     *
     * The copied [text] is sliced from [source] after converting the snapped
     * scalar coordinates to Kotlin's native UTF-16 indices.
     */
    fun snappedToGraphemeBoundaries(
        source: String,
        segmenter: IcuTextSegmenter = IcuTextSegmenter(),
    ): EntityPrediction {
        val snapped = segmenter.snapScalarSpan(source, start, end)
        return copy(
            text = UnicodeOffsetContract.substring(
                source,
                snapped.start,
                snapped.end,
            ),
            start = snapped.start,
            end = snapped.end,
        )
    }

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
