package com.openmed.openmedkit.packs

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File

/**
 * One medication mention found in the input text.
 *
 * A mention is extraction output only. It records where a medication is
 * named, and has no dose, interaction, recommendation or action field.
 *
 * @property label Pack canonical label the model label normalizes to
 *   (`CHEMICAL`; PharmaDetect models emit `CHEM`).
 * @property start Inclusive Unicode scalar offset into the input text.
 * @property end Exclusive Unicode scalar offset into the input text.
 * @property modelId Repository id of the model that produced the mention.
 */
class MedicationMention internal constructor(
    val label: String,
    val start: Int,
    val end: Int,
    val confidence: Float,
    val modelId: String,
)

/**
 * Medication task pack: runs a prepared PharmaDetect model and returns the
 * medication mentions it finds, with their offsets.
 *
 * The pack is extraction-only. It never gives dosing or interaction advice,
 * never triggers a follow-up action and never contacts a service: results are
 * data for the app to show. Every [TaskPackResult] carries the pack's
 * extraction-only notice ([TaskPackResult.notice], also available as
 * `resolution.notice`), which apps must display with the mentions.
 *
 * @param catalog Maps candidate repositories to integrity-pinned entries,
 *   normally `ModelCatalog.load(context)`.
 * @param classifierFactory Opens a classifier over a verified cache directory;
 *   the default loads `model_int8.onnx` with its tokenizer and label map.
 * @param manifest Task pack manifest bytes; the bundled file by default.
 */
class MedicationPack(
    catalog: ModelCatalog,
    classifierFactory: (File) -> OnnxTokenClassifier = { defaultTaskPackClassifier(it) },
    manifest: ByteArray? = TaskPacks.bundledManifest(),
) : TaskPack(TaskPacks.MEDICATION, catalog, classifierFactory, manifest) {
    /**
     * Extract medication mentions at or above [confidenceThreshold].
     *
     * Spans whose label does not normalize to a pack canonical label are
     * dropped.
     *
     * @throws TaskPackException with [TaskPackFailure.NOT_PREPARED] before a
     *   successful [prepare].
     */
    suspend fun extract(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): TaskPackResult<List<MedicationMention>> {
        val (kit, modelId) = preparedModel()
            ?: throw TaskPackException(TaskPackFailure.NOT_PREPARED)
        val mentions = kit.analyzeText(text, confidenceThreshold).mapNotNull { entity ->
            PolicyProfiles.canonicalLabel(entity.label)
                .takeIf { it in resolution.canonicalLabels }
                ?.let { MedicationMention(it, entity.start, entity.end, entity.confidence, modelId) }
        }
        return result(mentions, modelId)
    }
}
