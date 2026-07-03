package com.openmed.openmedkit

/**
 * Raw token/span prediction emitted by an Android token classifier.
 */
data class TokenClassificationPrediction(
    val label: String,
    val text: String,
    val confidence: Float,
    val start: Int,
    val end: Int,
)

/**
 * Token-classification model seam used by OpenMedKit.
 *
 * Future ONNX-backed tasks can replace the default implementation without
 * changing the public facade.
 */
interface OnnxTokenClassifier {
    suspend fun predict(text: String): List<TokenClassificationPrediction>

    fun tokenOffsets(text: String): List<IntRange> = defaultTokenOffsets(text)
}

/**
 * Placeholder ONNX classifier for offline backend construction.
 *
 * The public API can be constructed from OpenMedBackend now; the actual ONNX
 * session implementation is owned by the lower-level Android runtime tasks.
 */
class BackendOnnxTokenClassifier(
    private val backend: OpenMedBackend,
) : OnnxTokenClassifier {
    override suspend fun predict(text: String): List<TokenClassificationPrediction> {
        throw UnsupportedOperationException(
            "ONNX token classification is not available yet for ${backend.modelFile.path}",
        )
    }
}

/**
 * Decodes classifier output into EntityPrediction records.
 */
class TokenClassificationDecoder {
    fun decode(
        predictions: List<TokenClassificationPrediction>,
        sourceText: String,
    ): List<EntityPrediction> {
        val textLength = sourceText.length
        return predictions.mapNotNull { prediction ->
            if (prediction.start < 0 || prediction.end <= prediction.start || prediction.end > textLength) {
                return@mapNotNull null
            }
            EntityPrediction(
                label = prediction.label,
                text = sourceText.substring(prediction.start, prediction.end),
                confidence = prediction.confidence,
                start = prediction.start,
                end = prediction.end,
            )
        }
    }
}

/**
 * Minimal span repair used by the facade before semantic merging.
 */
object SpanRepair {
    fun repair(
        entities: List<EntityPrediction>,
        sourceText: String,
    ): List<EntityPrediction> {
        val textLength = sourceText.length
        return entities.mapNotNull { entity ->
            val start = entity.start.coerceIn(0, textLength)
            val end = entity.end.coerceIn(start, textLength)
            if (end <= start) {
                return@mapNotNull null
            }
            entity.copy(text = sourceText.substring(start, end), start = start, end = end)
        }
    }
}

/**
 * Smart PII merger seam.
 *
 * This performs deterministic overlap de-duplication today and gives the
 * lower-level semantic merger task one stable integration point.
 */
class PiiEntityMerger {
    fun merge(
        entities: List<EntityPrediction>,
        sourceText: String,
    ): List<EntityPrediction> = OpenMedKit.deduplicateOverlappingEntities(
        SpanRepair.repair(entities, sourceText),
    )
}

internal fun defaultTokenOffsets(text: String): List<IntRange> {
    val offsets = mutableListOf<IntRange>()
    var tokenStart: Int? = null
    for (index in text.indices) {
        if (text[index].isWhitespace()) {
            val start = tokenStart
            if (start != null) {
                offsets += start until index
                tokenStart = null
            }
        } else if (tokenStart == null) {
            tokenStart = index
        }
    }
    tokenStart?.let { offsets += it until text.length }
    return offsets
}
