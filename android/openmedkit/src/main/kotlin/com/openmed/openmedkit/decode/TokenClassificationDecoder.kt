package com.openmed.openmedkit.decode

import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.segmentation.IcuTextSegmenter

/**
 * A single per-token prediction before entity grouping.
 *
 * Offsets use the shared half-open Unicode scalar contract.
 */
data class TokenPrediction(
    val labelId: Int,
    val label: String,
    val score: Float,
    val startOffset: Int,
    val endOffset: Int,
)

/**
 * A grouped entity prediction over the original text.
 *
 * [start] and [end] are Unicode scalar offsets, never Kotlin UTF-16 indices.
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

/**
 * Decodes BIO/BIOES token-classification predictions into grouped entity spans.
 */
object TokenClassificationDecoder {
    private val segmenter = IcuTextSegmenter()

    fun decodeEntities(
        tokens: List<TokenPrediction>,
        text: String,
        strategy: AggregationStrategy = AggregationStrategy.AVERAGE,
    ): List<EntityPrediction> {
        val entities = mutableListOf<EntityPrediction>()
        val textIndex = CharacterIndex(text)

        var currentLabel: String? = null
        var currentStart = 0
        var currentEnd = 0
        var currentScores = mutableListOf<Float>()

        fun flushCurrent() {
            val label = currentLabel ?: return
            entities.add(
                makeEntity(
                    label = label,
                    start = currentStart,
                    end = currentEnd,
                    scores = currentScores,
                    textIndex = textIndex,
                    strategy = strategy,
                )
            )
            currentLabel = null
            currentScores = mutableListOf()
        }

        for (token in tokens) {
            val parsedLabel = ParsedLabel.from(token.label)
            if (parsedLabel == null) {
                flushCurrent()
                continue
            }

            when (parsedLabel.boundary) {
                Boundary.SINGLE -> {
                    flushCurrent()
                    entities.add(
                        makeEntity(
                            label = parsedLabel.entityType,
                            start = token.startOffset,
                            end = token.endOffset,
                            scores = listOf(token.score),
                            textIndex = textIndex,
                            strategy = strategy,
                        )
                    )
                }

                Boundary.BEGIN, Boundary.UNPREFIXED -> {
                    flushCurrent()
                    currentLabel = parsedLabel.entityType
                    currentStart = token.startOffset
                    currentEnd = token.endOffset
                    currentScores = mutableListOf(token.score)
                }

                Boundary.INSIDE -> {
                    if (currentLabel == null || currentLabel != parsedLabel.entityType) {
                        flushCurrent()
                        currentLabel = parsedLabel.entityType
                        currentStart = token.startOffset
                        currentEnd = token.endOffset
                        currentScores = mutableListOf(token.score)
                    } else {
                        currentEnd = token.endOffset
                        currentScores.add(token.score)
                    }
                }

                Boundary.END -> {
                    if (currentLabel == null || currentLabel != parsedLabel.entityType) {
                        flushCurrent()
                        entities.add(
                            makeEntity(
                                label = parsedLabel.entityType,
                                start = token.startOffset,
                                end = token.endOffset,
                                scores = listOf(token.score),
                                textIndex = textIndex,
                                strategy = strategy,
                            )
                        )
                    } else {
                        currentEnd = token.endOffset
                        currentScores.add(token.score)
                        flushCurrent()
                    }
                }
            }
        }

        flushCurrent()
        return repairEntitySpans(entities, text)
    }

    fun repairEntitySpans(
        entities: List<EntityPrediction>,
        text: String,
    ): List<EntityPrediction> {
        if (text.isEmpty()) {
            return entities
        }

        val textIndex = CharacterIndex(text)

        return entities.map { entity ->
            val initial = segmenter.snapScalarSpan(text, entity.start, entity.end)
            var start = initial.start
            var end = initial.end

            if (
                !IcuTextSegmenter.requiresFallback(
                    textIndex.substring(start, end),
                )
            ) {
                var extended = 0
                while (end < textIndex.length && extended < 10) {
                    val character = textIndex.characterAt(end) ?: break
                    if (!isWordLike(character)) {
                        break
                    }
                    end += 1
                    extended += 1
                }
            }

            while (start < end && textIndex.characterAt(start)?.isWhitespaceCharacter() == true) {
                start += 1
            }

            while (end > start && textIndex.characterAt(end - 1)?.isWhitespaceCharacter() == true) {
                end -= 1
            }

            if (start >= end) {
                return@map entity
            }

            val snapped = segmenter.snapScalarSpan(text, start, end)
            start = snapped.start
            end = snapped.end
            val span = textIndex.substring(start, end)
            if (span.trim { it.isWhitespace() }.isEmpty()) {
                entity
            } else {
                entity.copy(text = span, start = start, end = end)
            }
        }
    }

    private fun makeEntity(
        label: String,
        start: Int,
        end: Int,
        scores: List<Float>,
        textIndex: CharacterIndex,
        strategy: AggregationStrategy,
    ): EntityPrediction {
        val confidence = when (strategy) {
            AggregationStrategy.FIRST -> scores.firstOrNull() ?: 0.0f
            AggregationStrategy.MAX -> scores.maxOrNull() ?: 0.0f
            AggregationStrategy.AVERAGE -> scores.averageOrZero()
        }

        val snapped = segmenter.snapScalarSpan(
            textIndex.source,
            start,
            end,
        )
        return EntityPrediction(
            label = label,
            text = textIndex.substring(snapped.start, snapped.end),
            confidence = confidence,
            start = snapped.start,
            end = snapped.end,
        )
    }

    private fun List<Float>.averageOrZero(): Float {
        if (isEmpty()) {
            return 0.0f
        }
        return sum() / size
    }

    private fun isWordLike(character: String): Boolean {
        return character.codePoints().anyMatch { codePoint ->
            when (Character.getType(codePoint)) {
                Character.UPPERCASE_LETTER.toInt(),
                Character.LOWERCASE_LETTER.toInt(),
                Character.TITLECASE_LETTER.toInt(),
                Character.MODIFIER_LETTER.toInt(),
                Character.OTHER_LETTER.toInt(),
                Character.NON_SPACING_MARK.toInt(),
                Character.COMBINING_SPACING_MARK.toInt(),
                Character.ENCLOSING_MARK.toInt(),
                Character.DECIMAL_DIGIT_NUMBER.toInt(),
                Character.LETTER_NUMBER.toInt(),
                Character.OTHER_NUMBER.toInt(),
                -> true

                else -> false
            }
        }
    }

    private fun String.isWhitespaceCharacter(): Boolean {
        return codePoints().allMatch { Character.isWhitespace(it) }
    }

    private enum class Boundary {
        BEGIN,
        INSIDE,
        END,
        SINGLE,
        UNPREFIXED,
    }

    private data class ParsedLabel(
        val entityType: String,
        val boundary: Boundary,
    ) {
        companion object {
            fun from(label: String): ParsedLabel? {
                if (label == "O") {
                    return null
                }

                if (label.length >= 2 && label[1] == '-') {
                    val entityType = label.substring(2)
                    return when (label[0]) {
                        'B' -> ParsedLabel(entityType, Boundary.BEGIN)
                        'I' -> ParsedLabel(entityType, Boundary.INSIDE)
                        'E' -> ParsedLabel(entityType, Boundary.END)
                        'S' -> ParsedLabel(entityType, Boundary.SINGLE)
                        else -> ParsedLabel(label, Boundary.UNPREFIXED)
                    }
                }

                return ParsedLabel(label, Boundary.UNPREFIXED)
            }
        }
    }

    private class CharacterIndex(text: String) {
        val source = text

        val length: Int = UnicodeOffsetContract.scalarLength(text)

        fun substring(start: Int, end: Int): String {
            if (start < 0 || end < start || end > length) {
                return ""
            }
            return UnicodeOffsetContract.substring(source, start, end)
        }

        fun characterAt(offset: Int): String? {
            if (offset < 0 || offset >= length) {
                return null
            }
            return UnicodeOffsetContract.substring(source, offset, offset + 1)
        }
    }
}
