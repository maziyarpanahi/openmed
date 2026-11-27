package com.openmed.openmedkit.packs

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File

/**
 * One clinical-entity mention.
 *
 * @property category Pack canonical label the model label normalizes to
 *   (`CONDITION`, `DISEASE` or `PATHOLOGY` in pack version 1).
 * @property start Inclusive Unicode scalar offset into the input text.
 * @property end Exclusive Unicode scalar offset into the input text.
 * @property modelId Repository id of the model that produced the mention.
 */
class ClinicalEntity internal constructor(
    val category: String,
    val start: Int,
    val end: Int,
    val confidence: Float,
    val modelId: String,
)

/**
 * Clinical-entity task pack: runs a prepared token-classification model and
 * keeps only spans whose label normalizes, through the portable label
 * taxonomy, to one of the pack's canonical labels. Other labels are dropped,
 * never guessed into a category.
 *
 * The pack reports condition mentions and their offsets only. It does not
 * diagnose, rank or recommend care; show [TaskPackResult.notice] with its
 * output.
 *
 * @param catalog Maps candidate repositories to integrity-pinned entries,
 *   normally `ModelCatalog.load(context)`.
 * @param classifierFactory Opens a classifier over a verified cache directory;
 *   the default loads `model_int8.onnx` with its tokenizer and label map.
 * @param manifest Task pack manifest bytes; the bundled file by default.
 */
class ClinicalEntityPack(
    catalog: ModelCatalog,
    classifierFactory: (File) -> OnnxTokenClassifier = { defaultTaskPackClassifier(it) },
    manifest: ByteArray? = TaskPacks.bundledManifest(),
) : TaskPack(TaskPacks.CLINICAL_ENTITY, catalog, classifierFactory, manifest) {
    /**
     * Extract clinical entities at or above [confidenceThreshold].
     *
     * @throws TaskPackException with [TaskPackFailure.NOT_PREPARED] before a
     *   successful [prepare].
     */
    suspend fun extract(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): TaskPackResult<List<ClinicalEntity>> {
        val (kit, modelId) = preparedModel()
            ?: throw TaskPackException(TaskPackFailure.NOT_PREPARED)
        val entities = kit.analyzeText(text, confidenceThreshold).mapNotNull { entity ->
            PolicyProfiles.canonicalLabel(entity.label)
                .takeIf { it in resolution.canonicalLabels }
                ?.let { ClinicalEntity(it, entity.start, entity.end, entity.confidence, modelId) }
        }
        return result(entities, modelId)
    }
}
