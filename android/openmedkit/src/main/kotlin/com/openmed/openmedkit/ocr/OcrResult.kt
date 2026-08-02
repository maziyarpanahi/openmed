package com.openmed.openmedkit.ocr

/**
 * Pixel-space rectangle for a recognized OCR token.
 */
data class OcrBoundingBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
)

/**
 * A recognized word-like OCR token with offsets into [OcrResult.text].
 */
data class OcrToken(
    val text: String,
    val start: Int,
    val end: Int,
    val confidence: Float,
    val bbox: OcrBoundingBox? = null,
    val page: Int = 0,
) {
    init {
        require(start >= 0) { "start must be non-negative" }
        require(end >= start) { "end must be greater than or equal to start" }
        require(confidence in 0.0f..1.0f) { "confidence must be in [0, 1]" }
        require(page >= 0) { "page must be non-negative" }
    }
}

/**
 * OCR output shared by Android adapters.
 *
 * [text] is the contiguous recognized text. Every token offset indexes into
 * this string, making downstream de-identification spans portable across
 * OpenMedKit surfaces.
 */
data class OcrResult(
    val text: String,
    val words: List<OcrToken>,
    val metadata: Map<String, String> = emptyMap(),
) {
    val tokens: List<OcrToken>
        get() = words

    init {
        words.forEach { token ->
            require(token.end <= text.length) { "token end offset exceeds text length" }
            require(text.substring(token.start, token.end) == token.text) {
                "token offsets must index back into the recognized text"
            }
        }
    }

    companion object {
        fun empty(metadata: Map<String, String> = emptyMap()): OcrResult =
            OcrResult(text = "", words = emptyList(), metadata = metadata)
    }
}

internal data class RecognizedOcrToken(
    val text: String,
    val confidence: Float,
    val bbox: OcrBoundingBox? = null,
    val page: Int = 0,
)

internal data class RecognizedOcrLine(
    val tokens: List<RecognizedOcrToken>,
    val bbox: OcrBoundingBox? = null,
)

internal data class RecognizedOcrBlock(
    val lines: List<RecognizedOcrLine>,
    val bbox: OcrBoundingBox? = null,
)

internal fun buildOcrResult(
    blocks: List<RecognizedOcrBlock>,
    metadata: Map<String, String> = emptyMap(),
): OcrResult {
    val text = StringBuilder()
    val tokens = mutableListOf<OcrToken>()

    var hasWrittenBlock = false
    orderedByPagePosition(blocks).forEach { (_, block) ->
        val orderedLines = orderedByPagePosition(block.lines)
            .map { it.value }
            .filter { line -> line.tokens.any { token -> token.text.isNotBlank() } }
        if (orderedLines.isEmpty()) {
            return@forEach
        }

        if (hasWrittenBlock) {
            text.append("\n\n")
        }
        hasWrittenBlock = true

        orderedLines.forEachIndexed { lineIndex, line ->
            if (lineIndex > 0) {
                text.append('\n')
            }

            orderedByPagePosition(line.tokens)
                .map { it.value }
                .filter { it.text.isNotBlank() }
                .forEachIndexed { tokenIndex, token ->
                    if (tokenIndex > 0) {
                        text.append(' ')
                    }

                    val start = text.length
                    text.append(token.text)
                    val end = text.length
                    tokens += OcrToken(
                        text = token.text,
                        start = start,
                        end = end,
                        confidence = token.confidence.coerceIn(0.0f, 1.0f),
                        bbox = token.bbox,
                        page = token.page,
                    )
                }
        }
    }

    return OcrResult(text = text.toString(), words = tokens, metadata = metadata)
}

private fun <T> orderedByPagePosition(
    items: List<T>,
): List<IndexedValue<T>> where T : Any =
    items.withIndex()
        .sortedWith(
            compareBy<IndexedValue<T>>(
                { it.value.topOrMax() },
                { it.value.leftOrMax() },
                { it.index },
            ),
        )

private fun Any.topOrMax(): Int =
    when (this) {
        is RecognizedOcrBlock -> bbox?.top
        is RecognizedOcrLine -> bbox?.top
        is RecognizedOcrToken -> bbox?.top
        else -> null
    } ?: Int.MAX_VALUE

private fun Any.leftOrMax(): Int =
    when (this) {
        is RecognizedOcrBlock -> bbox?.left
        is RecognizedOcrLine -> bbox?.left
        is RecognizedOcrToken -> bbox?.left
        else -> null
    } ?: Int.MAX_VALUE
