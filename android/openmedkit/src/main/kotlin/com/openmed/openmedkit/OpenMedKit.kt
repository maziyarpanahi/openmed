package com.openmed.openmedkit

import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.Closeable
import java.io.File

/**
 * OpenMedKit public Kotlin facade for on-device clinical NLP.
 */
class OpenMedKit(
    private val classifier: OnnxTokenClassifier,
    private val decoder: TokenClassificationDecoder = TokenClassificationDecoder(),
    private val merger: PiiEntityMerger = PiiEntityMerger(),
    private val deidentifyEngine: DeidentifyEngine = DeidentifyEngine(),
) : Closeable {
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
            val baseScalarOffset = UnicodeOffsetContract.utf16ToScalarOffset(
                text,
                chunk.start,
            )
            extractPii(chunkText, confidenceThreshold, useSmartMerging)
                .mapNotNull { it.offsetBy(baseScalarOffset, text) }
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

    override fun close() {
        classifier.close()
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
        const val VERSION = "1.9.1"

        /**
         * Load an exported OpenMed ONNX directory for local Android inference.
         */
        @JvmStatic
        fun fromDirectory(
            modelDirectory: File,
            variant: String = "int8",
        ): OpenMedKit {
            val modelFileName = when (variant.lowercase()) {
                "int8" -> "model_int8.onnx"
                "fp32" -> "model.onnx"
                "fp16" -> "model_fp16.onnx"
                else -> throw IllegalArgumentException(
                    "variant must be int8, fp32, or fp16",
                )
            }
            return OpenMedKit(
                OpenMedBackend(
                    modelDirectory = modelDirectory,
                    modelFile = File(modelDirectory, modelFileName),
                ),
            )
        }

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

/**
 * Public Android counterpart to Swift OpenMedKit's OpenMed entry point.
 */
class OpenMed(
    private val backend: OpenMedBackend? = null,
) {
    /**
     * Run local PII analysis and return entity spans above the confidence threshold.
     */
    fun analyzeText(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): List<EntityPrediction> {
        val entities = when (backend) {
            null -> LocalPiiRecognizer.detect(text)
            else -> LocalPiiRecognizer.detect(text)
        }
        return entities.filter { it.confidence >= confidenceThreshold }
    }

    /**
     * Run PII extraction using the Android public API shape shared with Swift.
     */
    fun extractPII(
        text: String,
        confidenceThreshold: Float = 0.5f,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> {
        val entities = analyzeText(text, confidenceThreshold)
        return if (useSmartMerging) entities else entities
    }

    /**
     * Run PII extraction over long text.
     *
     * This parity surface delegates to the deterministic local recognizer until
     * Android model windowing is wired into the same public API.
     */
    fun extractPIIChunked(
        text: String,
        confidenceThreshold: Float = 0.5f,
        chunkTokenLimit: Int = 256,
        tokenOverlap: Int = 32,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> {
        require(chunkTokenLimit > 0) { "chunkTokenLimit must be positive" }
        require(tokenOverlap >= 0) { "tokenOverlap must be non-negative" }
        return extractPII(text, confidenceThreshold, useSmartMerging)
    }
}

/**
 * Android cache state names aligned with Swift OpenMedMLXModelCacheState.
 */
enum class OpenMedMLXModelCacheState {
    missing,
    partial,
    ready,
}

/**
 * Public Android counterpart to Swift OpenMedModelStore.
 */
object OpenMedModelStore {
    private const val READY_MARKER_FILE_NAME = ".openmed-artifact-ready"

    @JvmStatic
    fun downloadMLXModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: File? = null,
    ): File {
        val modelDirectory = cachedMLXModelDirectory(repoID, revision, cacheDirectory)
        if (isMLXModelCached(repoID, revision, cacheDirectory)) {
            return modelDirectory
        }
        throw UnsupportedOperationException(
            "Android MLX model downloads are not implemented in the scaffold",
        )
    }

    @JvmStatic
    fun cachedMLXModelDirectory(
        repoID: String,
        revision: String = "main",
        cacheDirectory: File? = null,
    ): File {
        val cacheRoot = cacheDirectory ?: File(
            System.getProperty("java.io.tmpdir"),
            "openmedkit-models",
        )
        return File(
            File(cacheRoot, repoID.sanitizedPathComponent()),
            revision.sanitizedPathComponent(),
        )
    }

    @JvmStatic
    fun isMLXModelCached(
        repoID: String,
        revision: String = "main",
        cacheDirectory: File? = null,
    ): Boolean {
        return mlxModelCacheState(
            repoID,
            revision,
            cacheDirectory,
        ) == OpenMedMLXModelCacheState.ready
    }

    @JvmStatic
    fun mlxModelCacheState(
        repoID: String,
        revision: String = "main",
        cacheDirectory: File? = null,
    ): OpenMedMLXModelCacheState {
        val modelDirectory = cachedMLXModelDirectory(repoID, revision, cacheDirectory)
        if (!modelDirectory.exists()) {
            return OpenMedMLXModelCacheState.missing
        }
        if (File(modelDirectory, READY_MARKER_FILE_NAME).isFile) {
            return OpenMedMLXModelCacheState.ready
        }
        return if (modelDirectory.listFiles().isNullOrEmpty()) {
            OpenMedMLXModelCacheState.missing
        } else {
            OpenMedMLXModelCacheState.partial
        }
    }
}

private object LocalPiiRecognizer {
    private val patterns = listOf(
        LabeledPattern(
            "patient_name",
            Regex("""(?m)\bPatient:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"""),
            1,
        ),
        LabeledPattern(
            "date_of_birth",
            Regex("""(?m)\bDOB:\s+(\d{4}-\d{2}-\d{2})"""),
            1,
        ),
        LabeledPattern("medical_record_number", Regex("""\bMRN-[0-9]+\b""")),
        LabeledPattern(
            "phone_number",
            Regex("""\b(?:\+?1[-.\s]?)?\d{3}-\d{3}-\d{4}\b"""),
        ),
        LabeledPattern(
            "email",
            Regex("""\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"""),
        ),
        LabeledPattern("date", Regex("""(?i)\bon\s+(\d{4}-\d{2}-\d{2})"""), 1),
    )

    fun detect(text: String): List<EntityPrediction> {
        return patterns
            .flatMap { pattern -> pattern.find(text) }
            .sortedWith(compareBy<EntityPrediction> { it.start }.thenBy { it.end })
            .deduplicate()
    }
}

private data class LabeledPattern(
    val label: String,
    val regex: Regex,
    val groupIndex: Int = 0,
) {
    fun find(text: String): List<EntityPrediction> {
        return regex.findAll(text).mapNotNull { match ->
            val group = match.groups[groupIndex] ?: return@mapNotNull null
            val range = group.range
            val utf16Start = range.first
            val utf16End = range.last + 1
            EntityPrediction(
                label = label,
                text = text.substring(utf16Start, utf16End),
                confidence = 1.0f,
                start = UnicodeOffsetContract.utf16ToScalarOffset(text, utf16Start),
                end = UnicodeOffsetContract.utf16ToScalarOffset(text, utf16End),
            )
        }.toList()
    }
}

private fun List<EntityPrediction>.deduplicate(): List<EntityPrediction> {
    val selected = mutableListOf<EntityPrediction>()
    for (entity in this) {
        val alreadySelected = selected.any {
            entity.start == it.start &&
                entity.end == it.end &&
                entity.label == it.label
        }
        if (!alreadySelected) {
            selected += entity
        }
    }
    return selected
}

private fun EntityPrediction.offsetBy(
    baseOffset: Int,
    sourceText: String,
): EntityPrediction? {
    val start = start + baseOffset
    val end = end + baseOffset
    if (
        start < 0 ||
        end <= start ||
        end > UnicodeOffsetContract.scalarLength(sourceText)
    ) {
        return null
    }
    return copy(
        text = UnicodeOffsetContract.substring(sourceText, start, end),
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

private fun String.sanitizedPathComponent(): String {
    return replace(Regex("""[^A-Za-z0-9._-]+"""), "_").trim('_').ifEmpty { "default" }
}
