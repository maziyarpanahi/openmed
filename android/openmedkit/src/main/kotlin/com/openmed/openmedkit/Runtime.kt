package com.openmed.openmedkit

import ai.djl.huggingface.tokenizers.Encoding
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import com.openmed.openmedkit.onnx.OnnxTokenClassifier as RuntimeOnnxTokenClassifier
import com.openmed.openmedkit.onnx.TokenOffset
import com.openmed.openmedkit.onnx.TokenPrediction
import com.openmed.openmedkit.segmentation.IcuTextSegmenter
import java.io.Closeable

/**
 * Raw token/span prediction emitted by an Android token classifier.
 *
 * [start] and [end] use half-open Unicode scalar offsets.
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
interface OnnxTokenClassifier : Closeable {
    suspend fun predict(text: String): List<TokenClassificationPrediction>

    fun tokenOffsets(text: String): List<IntRange> = defaultTokenOffsets(text)

    override fun close() = Unit
}

/**
 * Hugging Face tokenizer and ONNX Runtime classifier for offline artifacts.
 */
class BackendOnnxTokenClassifier(
    private val backend: OpenMedBackend,
) : OnnxTokenClassifier {
    private val classifier = if (backend.id2Label.isEmpty()) {
        RuntimeOnnxTokenClassifier(backend.modelFile, backend.id2LabelFile)
    } else {
        RuntimeOnnxTokenClassifier(backend.modelFile, backend.id2Label)
    }
    private val tokenizer = try {
        loadTokenizer(backend)
    } catch (error: Throwable) {
        classifier.close()
        throw error
    }

    override suspend fun predict(text: String): List<TokenClassificationPrediction> {
        require(text.isNotEmpty()) { "text must not be empty" }
        val encoding = encode(text)
        val offsets = encoding.toTokenOffsets()
        val predictions = classifier.run(
            inputIds = encoding.ids.map(Long::toInt).toIntArray(),
            attentionMask = encoding.attentionMask.map(Long::toInt).toIntArray(),
            offsets = offsets,
        )
        return aggregateTokenPredictions(text, predictions)
    }

    override fun tokenOffsets(text: String): List<IntRange> =
        encode(text).toTokenOffsets()
            .filterNot { it.startOffset == 0 && it.endOffset == 0 }
            .mapNotNull { offset ->
                val scalarLength = UnicodeOffsetContract.scalarLength(text)
                if (
                    offset.startOffset !in 0..scalarLength ||
                    offset.endOffset !in offset.startOffset..scalarLength
                ) {
                    return@mapNotNull null
                }
                val utf16 = UnicodeOffsetContract.utf16Span(
                    text,
                    offset.startOffset,
                    offset.endOffset,
                )
                utf16.start until utf16.end
            }

    override fun close() {
        var failure: Throwable? = null
        try {
            classifier.close()
        } catch (error: Throwable) {
            failure = error
        }
        try {
            tokenizer.close()
        } catch (error: Throwable) {
            if (failure == null) {
                failure = error
            } else {
                failure.addSuppressed(error)
            }
        }
        failure?.let { throw it }
    }

    @Synchronized
    private fun encode(text: String): Encoding = tokenizer.encode(text)

    private fun Encoding.toTokenOffsets(): List<TokenOffset> =
        charTokenSpans.map { span ->
            if (span == null) {
                TokenOffset(0, 0)
            } else {
                TokenOffset(span.start, span.end)
            }
        }

    private fun loadTokenizer(backend: OpenMedBackend): HuggingFaceTokenizer {
        require(backend.tokenizerJson.isFile) {
            "tokenizer.json does not exist: ${backend.tokenizerJson.path}"
        }
        return HuggingFaceTokenizer.newInstance(backend.modelDirectory.toPath())
    }
}

internal fun aggregateTokenPredictions(
    text: String,
    predictions: List<TokenPrediction>,
): List<TokenClassificationPrediction> {
    val entities = mutableListOf<TokenClassificationPrediction>()
    var current: PendingTokenEntity? = null

    fun flush() {
        val entity = current ?: return
        val scalarLength = UnicodeOffsetContract.scalarLength(text)
        if (
            entity.start >= 0 &&
            entity.end <= scalarLength &&
            entity.end > entity.start
        ) {
            entities += TokenClassificationPrediction(
                label = entity.label,
                text = UnicodeOffsetContract.substring(
                    text,
                    entity.start,
                    entity.end,
                ),
                confidence = entity.scores.average().toFloat(),
                start = entity.start,
                end = entity.end,
            )
        }
        current = null
    }

    predictions.forEach { prediction ->
        val (prefix, label) = splitTokenLabel(prediction.label)
        if (prediction.startOffset == prediction.endOffset || label.equals("O", true)) {
            flush()
            return@forEach
        }
        val startsNew = current == null ||
            current?.label != label ||
            prefix in setOf("B", "S", "U") ||
            (prefix !in setOf("I", "E", "L") && prediction.startOffset > current!!.end)
        if (startsNew) {
            flush()
            current = PendingTokenEntity(
                label = label,
                start = prediction.startOffset,
                end = prediction.endOffset,
                scores = mutableListOf(prediction.score),
            )
        } else {
            current?.end = maxOf(current!!.end, prediction.endOffset)
            current?.scores?.add(prediction.score)
        }
        if (prefix in setOf("E", "L", "S", "U")) {
            flush()
        }
    }
    flush()
    return entities
}

private data class PendingTokenEntity(
    val label: String,
    val start: Int,
    var end: Int,
    val scores: MutableList<Float>,
)

private fun splitTokenLabel(rawLabel: String): Pair<String, String> {
    val label = rawLabel.trim()
    if (label.length > 2 && label[1] in setOf('-', '_')) {
        val prefix = label.substring(0, 1).uppercase()
        if (prefix in setOf("B", "I", "E", "L", "S", "U")) {
            return prefix to label.substring(2)
        }
    }
    return "" to label
}

/**
 * Decodes classifier output into EntityPrediction records.
 */
class TokenClassificationDecoder(
    private val segmenter: IcuTextSegmenter = IcuTextSegmenter(),
) {
    fun decode(
        predictions: List<TokenClassificationPrediction>,
        sourceText: String,
    ): List<EntityPrediction> {
        return predictions.mapNotNull { prediction ->
            if (prediction.end <= prediction.start) {
                return@mapNotNull null
            }
            EntityPrediction(
                label = prediction.label,
                text = prediction.text,
                confidence = prediction.confidence,
                start = prediction.start,
                end = prediction.end,
            ).snappedToGraphemeBoundaries(sourceText, segmenter)
                .takeIf { it.end > it.start }
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
        segmenter: IcuTextSegmenter = IcuTextSegmenter(),
    ): List<EntityPrediction> {
        return entities.mapNotNull { entity ->
            val snapped = entity.snappedToGraphemeBoundaries(sourceText, segmenter)
            if (snapped.end <= snapped.start) {
                return@mapNotNull null
            }
            snapped
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
    if (IcuTextSegmenter.requiresFallback(text)) {
        val segmenter = IcuTextSegmenter()
        val segmented = segmenter.fallbackWordSegments(text)
        if (segmented.isNotEmpty()) {
            return segmented.map { segment ->
                val utf16 = UnicodeOffsetContract.utf16Span(
                    text,
                    segment.start,
                    segment.end,
                )
                utf16.start until utf16.end
            }
        }
    }

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
