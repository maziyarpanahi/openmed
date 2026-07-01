package com.openmed.openmedkit

import java.io.File

/**
 * Public entry point for the Android OpenMedKit library scaffold.
 */
object OpenMedKit {
    /**
     * Placeholder version until Android artifacts are released.
     */
    const val VERSION = "0.0.0-dev"
}

/**
 * Android backend selector for OpenMed public API construction.
 */
sealed class OpenMedBackend {
    /**
     * Deterministic, local-only recognizer used by the Android scaffold tests.
     */
    object LocalRules : OpenMedBackend()

    /**
     * Reserved for a future ONNX Runtime-backed Android model path.
     */
    data class OnnxModel(val modelDirectory: File) : OpenMedBackend()
}

/**
 * Public Android counterpart to Swift OpenMedKit's entity result shape.
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
 * Public Android counterpart to Swift OpenMedKit's OpenMed entry point.
 */
class OpenMed(
    private val backend: OpenMedBackend = OpenMedBackend.LocalRules,
) {
    /**
     * Run local PII analysis and return entity spans above the confidence threshold.
     */
    fun analyzeText(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): List<EntityPrediction> {
        val entities = when (backend) {
            OpenMedBackend.LocalRules -> LocalPiiRecognizer.detect(text)
            is OpenMedBackend.OnnxModel -> LocalPiiRecognizer.detect(text)
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
     * Run PII extraction over long text. The scaffold delegates to the same
     * local recognizer until Android model windowing lands.
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
        LabeledPattern("date_of_birth", Regex("""(?m)\bDOB:\s+(\d{4}-\d{2}-\d{2})"""), 1),
        LabeledPattern("medical_record_number", Regex("""\bMRN-[0-9]+\b""")),
        LabeledPattern("phone_number", Regex("""\b(?:\+?1[-.\s]?)?\d{3}-\d{3}-\d{4}\b""")),
        LabeledPattern("email", Regex("""\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b""")),
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
            EntityPrediction(
                label = label,
                text = text.substring(range.first, range.last + 1),
                confidence = 1.0f,
                start = range.first,
                end = range.last + 1,
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

private fun String.sanitizedPathComponent(): String {
    return replace(Regex("""[^A-Za-z0-9._-]+"""), "_").trim('_').ifEmpty { "default" }
}
