package com.openmed.openmedkit.merge

import java.util.Locale
import java.util.regex.Pattern
import kotlin.math.max
import kotlin.math.min

/**
 * A PII entity span produced by a model or deterministic semantic pattern.
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

data class DominantLabel(
    val label: String,
    val averageConfidence: Float,
)

/**
 * Merges token-fragmented PII model spans with deterministic semantic units.
 */
object PiiEntityMerger {
    val defaultPiiPatterns: List<SemanticUnitPattern>
        get() = DefaultPiiPatterns.patterns

    @JvmStatic
    fun mergePIIEntities(
        entities: List<EntityPrediction>,
        text: String,
        useSemanticPatterns: Boolean = true,
        preferModelLabels: Boolean = true,
        allowSemanticOnlyMatches: Boolean = true,
        allowSemanticLabelExpansion: Boolean = true,
        patterns: List<SemanticUnitPattern> = defaultPiiPatterns,
    ): List<EntityPrediction> {
        val sortedEntities = entities.sortedWith(compareBy<EntityPrediction> { it.start }.thenBy { it.end })

        if (!useSemanticPatterns) {
            return sortedEntities
        }

        val semanticUnits = findSemanticUnits(text, patterns)
        if (semanticUnits.isEmpty()) {
            return sortedEntities
        }

        val merged = mutableListOf<EntityPrediction>()
        val usedEntityIndices = mutableSetOf<Int>()

        for (unit in semanticUnits) {
            val overlapping = sortedEntities.withIndex().filter { (_, entity) ->
                entity.start < unit.end && entity.end > unit.start
            }

            if (overlapping.isEmpty()) {
                if (allowSemanticOnlyMatches && unit.score >= 0.55f) {
                    merged += EntityPrediction(
                        label = unit.entityType,
                        text = substring(text, unit.start, unit.end),
                        confidence = unit.score,
                        start = unit.start,
                        end = unit.end,
                    )
                }
                continue
            }

            val overlappingEntities = overlapping.map { it.value }
            val dominant = calculateDominantLabel(overlappingEntities)

            val finalLabel = when {
                !allowSemanticLabelExpansion -> dominant.label
                preferModelLabels -> {
                    when {
                        isMoreSpecific(unit.entityType, dominant.label) -> unit.entityType
                        normalizeLabel(dominant.label) == normalizeLabel(unit.entityType) ||
                            isMoreSpecific(dominant.label, unit.entityType) -> dominant.label
                        else -> unit.entityType
                    }
                }
                overlappingEntities.any {
                    normalizeLabel(it.label) == normalizeLabel(unit.entityType)
                } -> dominant.label
                isMoreSpecific(dominant.label, unit.entityType) -> dominant.label
                else -> unit.entityType
            }

            val finalConfidence = if (unit.validated) {
                (0.6f * dominant.averageConfidence) + (0.4f * unit.score)
            } else {
                (0.9f * dominant.averageConfidence) + (0.1f * unit.score)
            }

            merged += EntityPrediction(
                label = finalLabel,
                text = substring(text, unit.start, unit.end),
                confidence = finalConfidence,
                start = unit.start,
                end = unit.end,
            )

            overlapping.forEach { usedEntityIndices += it.index }
        }

        sortedEntities.forEachIndexed { index, entity ->
            if (index !in usedEntityIndices) {
                merged += entity
            }
        }

        return merged.sortedWith(compareBy<EntityPrediction> { it.start }.thenBy { it.end })
    }

    @JvmStatic
    fun mergePiiEntities(
        entities: List<EntityPrediction>,
        text: String,
        useSemanticPatterns: Boolean = true,
        preferModelLabels: Boolean = true,
        allowSemanticOnlyMatches: Boolean = true,
        allowSemanticLabelExpansion: Boolean = true,
        patterns: List<SemanticUnitPattern> = defaultPiiPatterns,
    ): List<EntityPrediction> = mergePIIEntities(
        entities = entities,
        text = text,
        useSemanticPatterns = useSemanticPatterns,
        preferModelLabels = preferModelLabels,
        allowSemanticOnlyMatches = allowSemanticOnlyMatches,
        allowSemanticLabelExpansion = allowSemanticLabelExpansion,
        patterns = patterns,
    )

    @JvmStatic
    fun findSemanticUnits(
        text: String,
        patterns: List<SemanticUnitPattern> = defaultPiiPatterns,
    ): List<SemanticUnitMatch> {
        if (text.isEmpty()) {
            return emptyList()
        }

        val units = mutableListOf<SemanticUnitMatch>()

        patterns.sortedByDescending { it.priority }.forEach { semanticPattern ->
            val matcher = semanticPattern.regex.matcher(text)
            while (matcher.find()) {
                val captureGroup = semanticPattern.captureGroup
                val start = if (
                    captureGroup != null &&
                    captureGroup > 0 &&
                    captureGroup <= matcher.groupCount() &&
                    matcher.start(captureGroup) >= 0
                ) {
                    matcher.start(captureGroup)
                } else {
                    matcher.start()
                }
                val end = if (
                    captureGroup != null &&
                    captureGroup > 0 &&
                    captureGroup <= matcher.groupCount() &&
                    matcher.end(captureGroup) >= 0
                ) {
                    matcher.end(captureGroup)
                } else {
                    matcher.end()
                }

                if (start < 0 || end <= start) {
                    continue
                }

                val overlaps = units.any { existing ->
                    start < existing.end && end > existing.start
                }
                if (overlaps) {
                    continue
                }

                val matchedText = text.substring(start, end)
                var score = semanticPattern.baseScore

                if (
                    semanticPattern.contextWords.isNotEmpty() &&
                    findContextWords(text, start, end, semanticPattern.contextWords)
                ) {
                    score = min(1.0f, score + semanticPattern.contextBoost)
                }

                var validated = true
                val validator = semanticPattern.validator
                if (validator != null && !validator(matchedText)) {
                    score *= 0.3f
                    validated = false
                }

                units += SemanticUnitMatch(
                    start = start,
                    end = end,
                    entityType = semanticPattern.entityType,
                    score = score,
                    validated = validated,
                )
            }
        }

        return units.sortedBy { it.start }
    }

    @JvmStatic
    fun calculateDominantLabel(entities: List<EntityPrediction>): DominantLabel {
        require(entities.isNotEmpty()) {
            "Cannot calculate dominant label from empty entity list"
        }

        val labelCounts = linkedMapOf<String, Int>()
        val labelConfidences = linkedMapOf<String, MutableList<Float>>()

        entities.forEach { entity ->
            labelCounts[entity.label] = (labelCounts[entity.label] ?: 0) + 1
            labelConfidences.getOrPut(entity.label) { mutableListOf() } += entity.confidence
        }

        val maxCount = labelCounts.values.maxOrNull() ?: 0
        val candidates = labelCounts.keys.filter { labelCounts[it] == maxCount }
        val dominantLabel = if (candidates.size == 1) {
            candidates.first()
        } else {
            candidates.maxByOrNull { average(labelConfidences[it].orEmpty()) } ?: candidates.first()
        }

        return DominantLabel(
            label = dominantLabel,
            averageConfidence = average(entities.map { it.confidence }),
        )
    }

    @JvmStatic
    fun normalizeLabel(label: String): String = LabelNormalizer.normalizeLabel(label)

    @JvmStatic
    fun isMoreSpecific(label: String, than: String): Boolean =
        LabelNormalizer.isMoreSpecific(label, than)

    @JvmStatic
    fun deduplicateOverlappingEntities(
        entities: List<EntityPrediction>,
    ): List<EntityPrediction> {
        val selected = mutableListOf<EntityPrediction>()

        entities.sortedWith(entityComparator).forEach { entity ->
            val existingIndex = selected.indexOfFirst { areDuplicateCandidates(entity, it) }

            if (existingIndex == -1) {
                selected += entity
            } else if (isBetterDuplicate(entity, selected[existingIndex])) {
                selected[existingIndex] = entity
            }
        }

        return selected.sortedWith(compareBy<EntityPrediction> { it.start }.thenBy { it.end })
    }

    private fun findContextWords(
        text: String,
        start: Int,
        end: Int,
        contextWords: List<String>,
        contextWindow: Int = 100,
    ): Boolean {
        if (contextWords.isEmpty()) {
            return false
        }

        val windowStart = max(0, start - contextWindow)
        val windowEnd = min(text.length, end + contextWindow)
        val contextText = substring(text, windowStart, windowEnd).lowercase(Locale.US)

        return contextWords.any { contextWord ->
            val normalizedWord = contextWord.lowercase(Locale.US)
            contextText.contains(normalizedWord) ||
                Pattern.compile("\\b${Pattern.quote(normalizedWord)}\\b", Pattern.CASE_INSENSITIVE)
                    .matcher(contextText)
                    .find()
        }
    }

    private val entityComparator = Comparator<EntityPrediction> { lhs, rhs ->
        when {
            lhs.start != rhs.start -> lhs.start.compareTo(rhs.start)
            lhs.end != rhs.end -> entityLength(rhs).compareTo(entityLength(lhs))
            else -> rhs.confidence.compareTo(lhs.confidence)
        }
    }

    private fun areDuplicateCandidates(
        lhs: EntityPrediction,
        rhs: EntityPrediction,
    ): Boolean {
        if (!labelsAreCompatible(lhs.label, rhs.label)) {
            return false
        }

        val overlap = min(lhs.end, rhs.end) - max(lhs.start, rhs.start)
        if (overlap <= 0) {
            return false
        }

        val shorterLength = max(1, min(entityLength(lhs), entityLength(rhs)))
        return overlap.toDouble() / shorterLength.toDouble() >= 0.5
    }

    private fun labelsAreCompatible(lhs: String, rhs: String): Boolean {
        if (lhs == rhs) {
            return true
        }

        val normalizedLhs = normalizeLabel(lhs)
        val normalizedRhs = normalizeLabel(rhs)
        if (normalizedLhs == normalizedRhs) {
            return true
        }

        val lowerLhs = lhs.lowercase(Locale.US)
        val lowerRhs = rhs.lowercase(Locale.US)
        val nameTokens = listOf("name", "person")
        return nameTokens.any { lowerLhs.contains(it) && lowerRhs.contains(it) }
    }

    private fun isBetterDuplicate(
        candidate: EntityPrediction,
        existing: EntityPrediction,
    ): Boolean {
        val candidateLength = entityLength(candidate)
        val existingLength = entityLength(existing)

        if (candidate.start == existing.start && candidate.end == existing.end) {
            return candidate.confidence > existing.confidence
        }

        if (candidateLength > existingLength && candidate.confidence >= existing.confidence - 0.10f) {
            return true
        }

        return candidate.confidence > existing.confidence + 0.05f
    }

    private fun entityLength(entity: EntityPrediction): Int = max(0, entity.end - entity.start)

    private fun average(values: List<Float>): Float {
        if (values.isEmpty()) {
            return 0.0f
        }
        return values.sum() / values.size
    }

    private fun substring(text: String, start: Int, end: Int): String {
        if (start < 0 || end < start || end > text.length) {
            return ""
        }
        return text.substring(start, end)
    }
}
