package com.openmed.openmedkit

import com.openmed.openmedkit.policy.PolicyProfiles

/**
 * OpenMedKit public Kotlin facade for on-device clinical NLP.
 */
class OpenMedKit(
    private val classifier: OnnxTokenClassifier,
    private val decoder: TokenClassificationDecoder = TokenClassificationDecoder(),
    private val merger: PiiEntityMerger = PiiEntityMerger(),
    private val deidentifyEngine: DeidentifyEngine = DeidentifyEngine(),
) {
    constructor(
        backend: OpenMedBackend,
        decoder: TokenClassificationDecoder = TokenClassificationDecoder(),
        merger: PiiEntityMerger = PiiEntityMerger(),
        deidentifyEngine: DeidentifyEngine = DeidentifyEngine(),
    ) : this(
        classifier = BackendOnnxTokenClassifier(backend),
        decoder = decoder,
        merger = merger,
        deidentifyEngine = deidentifyEngine,
    )

    /**
     * Run token classification and return entities above the confidence threshold.
     */
    suspend fun analyzeText(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): List<EntityPrediction> {
        require(confidenceThreshold in 0.0f..1.0f) {
            "confidenceThreshold must be between 0.0 and 1.0"
        }
        val predictions = classifier.predict(text)
        return decoder.decode(predictions, text)
            .filter { it.confidence >= confidenceThreshold }
            .sortedByOffset()
    }

    /**
     * Run PII detection with span repair and optional smart merging.
     */
    suspend fun extractPii(
        text: String,
        confidenceThreshold: Float = 0.5f,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> {
        val repaired = SpanRepair.repair(
            analyzeText(text, confidenceThreshold),
            text,
        )
        return if (useSmartMerging) {
            merger.merge(repaired, text)
        } else {
            repaired.sortedByOffset()
        }
    }

    /**
     * Run PII detection over overlapping token windows.
     *
     * Returned offsets always reference the original full input text.
     */
    suspend fun extractPiiChunked(
        text: String,
        confidenceThreshold: Float = 0.5f,
        chunkTokenLimit: Int = 256,
        tokenOverlap: Int = 32,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> {
        val chunks = makeTokenChunks(text, chunkTokenLimit, tokenOverlap)
        if (chunks.size <= 1) {
            return extractPii(text, confidenceThreshold, useSmartMerging)
        }

        val chunkEntities = chunks.flatMap { chunk ->
            val chunkText = text.substring(chunk.start, chunk.end)
            extractPii(chunkText, confidenceThreshold, useSmartMerging)
                .mapNotNull { it.offsetBy(chunk.start, text) }
        }

        val repaired = SpanRepair.repair(
            deduplicateOverlappingEntities(chunkEntities),
            text,
        )
        return if (useSmartMerging) {
            deduplicateOverlappingEntities(merger.merge(repaired, text))
        } else {
            deduplicateOverlappingEntities(repaired)
        }
    }

    /**
     * Positional overload matching the public Kotlin API shape in the roadmap.
     */
    suspend fun extractPiiChunked(
        text: String,
        chunkTokenLimit: Int,
        tokenOverlap: Int,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> = extractPiiChunked(
        text = text,
        confidenceThreshold = 0.5f,
        chunkTokenLimit = chunkTokenLimit,
        tokenOverlap = tokenOverlap,
        useSmartMerging = useSmartMerging,
    )

    /**
     * De-identify text under a bundled policy profile.
     *
     * When no policy is supplied, OpenMedKit uses the documented
     * `hipaa_safe_harbor` default profile.
     */
    suspend fun deidentify(
        text: String,
        policy: String = PolicyProfiles.DEFAULT_PROFILE,
        confidenceThreshold: Float = 0.5f,
        useSmartMerging: Boolean = true,
    ): PolicyDeidentificationResult {
        val profile = PolicyProfiles.load(policy)
        val entities = extractPii(text, confidenceThreshold, useSmartMerging)
        return deidentifyEngine.deidentify(text, entities, profile)
    }

    internal fun makeTokenChunks(
        text: String,
        chunkTokenLimit: Int,
        tokenOverlap: Int,
    ): List<TextChunk> {
        if (text.isEmpty()) {
            return emptyList()
        }
        val offsets = classifier.tokenOffsets(text).filter { it.first < it.last + 1 }
        val tokenLimit = chunkTokenLimit.coerceAtLeast(1)
        if (offsets.size <= tokenLimit) {
            return listOf(TextChunk(0, text.length, 0, offsets.size))
        }

        val overlap = tokenOverlap.coerceIn(0, tokenLimit - 1)
        val chunks = mutableListOf<TextChunk>()
        var tokenStart = 0
        while (tokenStart < offsets.size) {
            val tokenEnd = minOf(tokenStart + tokenLimit, offsets.size)
            chunks += TextChunk(
                start = offsets[tokenStart].first,
                end = offsets[tokenEnd - 1].last + 1,
                tokenStart = tokenStart,
                tokenEnd = tokenEnd,
            )
            if (tokenEnd >= offsets.size) {
                break
            }
            tokenStart = maxOf(tokenStart + 1, tokenEnd - overlap)
        }
        return chunks
    }

    internal data class TextChunk(
        val start: Int,
        val end: Int,
        val tokenStart: Int,
        val tokenEnd: Int,
    )

    companion object {
        /**
         * Placeholder version until Android artifacts are released.
         */
        const val VERSION = "0.0.0-dev"

        internal fun deduplicateOverlappingEntities(
            entities: List<EntityPrediction>,
        ): List<EntityPrediction> {
            val selected = mutableListOf<EntityPrediction>()
            for (entity in entities.sortedWith(::entityComparator)) {
                val existingIndex = selected.indexOfFirst {
                    areDuplicateCandidates(entity, it)
                }
                if (existingIndex < 0) {
                    selected += entity
                } else if (isBetterDuplicate(entity, selected[existingIndex])) {
                    selected[existingIndex] = entity
                }
            }
            return selected.sortedByOffset()
        }

        private fun entityComparator(
            lhs: EntityPrediction,
            rhs: EntityPrediction,
        ): Int {
            if (lhs.start != rhs.start) {
                return lhs.start.compareTo(rhs.start)
            }
            val lhsLength = lhs.end - lhs.start
            val rhsLength = rhs.end - rhs.start
            if (lhsLength != rhsLength) {
                return rhsLength.compareTo(lhsLength)
            }
            return rhs.confidence.compareTo(lhs.confidence)
        }

        private fun areDuplicateCandidates(
            lhs: EntityPrediction,
            rhs: EntityPrediction,
        ): Boolean {
            if (!labelsAreCompatible(lhs.label, rhs.label)) {
                return false
            }
            val overlap = minOf(lhs.end, rhs.end) - maxOf(lhs.start, rhs.start)
            if (overlap <= 0) {
                return false
            }
            val shorter = maxOf(1, minOf(lhs.end - lhs.start, rhs.end - rhs.start))
            return overlap.toDouble() / shorter.toDouble() >= 0.5
        }

        private fun labelsAreCompatible(lhs: String, rhs: String): Boolean {
            if (lhs == rhs) {
                return true
            }
            val normalizedLhs = lhs.normalizedLabelToken()
            val normalizedRhs = rhs.normalizedLabelToken()
            if (normalizedLhs == normalizedRhs) {
                return true
            }
            return listOf("name", "person").any {
                normalizedLhs.contains(it) && normalizedRhs.contains(it)
            }
        }

        private fun isBetterDuplicate(
            candidate: EntityPrediction,
            existing: EntityPrediction,
        ): Boolean {
            val candidateLength = candidate.end - candidate.start
            val existingLength = existing.end - existing.start
            if (candidate.start == existing.start && candidate.end == existing.end) {
                return candidate.confidence > existing.confidence
            }
            if (candidateLength > existingLength && candidate.confidence >= existing.confidence - 0.10f) {
                return true
            }
            return candidate.confidence > existing.confidence + 0.05f
        }
    }
}

private fun EntityPrediction.offsetBy(
    baseOffset: Int,
    sourceText: String,
): EntityPrediction? {
    val start = start + baseOffset
    val end = end + baseOffset
    if (start < 0 || end <= start || end > sourceText.length) {
        return null
    }
    return copy(
        text = sourceText.substring(start, end),
        start = start,
        end = end,
    )
}

private fun List<EntityPrediction>.sortedByOffset(): List<EntityPrediction> =
    sortedWith(compareBy<EntityPrediction> { it.start }.thenBy { it.end })

private fun String.normalizedLabelToken(): String =
    removePrefix("B-")
        .removePrefix("I-")
        .removePrefix("E-")
        .removePrefix("S-")
        .lowercase()
        .filter { it.isLetterOrDigit() }
