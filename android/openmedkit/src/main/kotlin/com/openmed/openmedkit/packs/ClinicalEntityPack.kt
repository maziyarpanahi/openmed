package com.openmed.openmedkit.packs

import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.ScalarSpan
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.clinical.ClinicalAssertion
import com.openmed.openmedkit.clinical.ClinicalContext
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
 * @property assertion ConText negation, uncertainty, temporality and
 *   experiencer of the mention, or `null` when [assertionReason] says why
 *   they were not computed. A negated, uncertain, hypothetical or
 *   non-patient mention is still returned, never dropped.
 * @property assertionReason [ClinicalContext.UNSUPPORTED_LANGUAGE] when
 *   [assertion] is `null`; `null` otherwise.
 */
class ClinicalEntity internal constructor(
    val category: String,
    val start: Int,
    val end: Int,
    val confidence: Float,
    val modelId: String,
    val assertion: ClinicalAssertion?,
    val assertionReason: String?,
)

/**
 * Clinical-entity task pack: runs a prepared token-classification model and
 * keeps only spans whose label normalizes, through the portable label
 * taxonomy, to one of the pack's canonical labels. Other labels are dropped,
 * never guessed into a category. Each kept span carries its ConText
 * assertion axes from [ClinicalContext].
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
     * Extract clinical entities at or above [confidenceThreshold], with
     * English assertions bounded by regex-split sentences.
     *
     * @throws TaskPackException with [TaskPackFailure.NOT_PREPARED] before a
     *   successful [prepare].
     */
    suspend fun extract(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): TaskPackResult<List<ClinicalEntity>> = extract(text, confidenceThreshold, null, null)

    /**
     * Extract clinical entities at or above [confidenceThreshold].
     *
     * @param language `null` or an `en` tag computes English assertions; any
     *   other language returns entities without an assertion.
     * @param sentences Sentence bounds in Unicode scalar offsets that limit
     *   assertion cues; a regex splitter is used when `null`.
     * @throws TaskPackException with [TaskPackFailure.NOT_PREPARED] before a
     *   successful [prepare].
     */
    suspend fun extract(
        text: String,
        confidenceThreshold: Float,
        language: String?,
        sentences: List<ScalarSpan>?,
    ): TaskPackResult<List<ClinicalEntity>> {
        val (kit, modelId) = preparedModel()
            ?: throw TaskPackException(TaskPackFailure.NOT_PREPARED)
        val kept = ArrayList<EntityPrediction>()
        val categories = ArrayList<String>()
        val spans = ArrayList<ScalarSpan>()
        for (entity in kit.analyzeText(text, confidenceThreshold)) {
            val category = PolicyProfiles.canonicalLabel(entity.label)
            if (category in resolution.canonicalLabels) {
                kept += entity
                categories += category
                spans += ScalarSpan(entity.start, entity.end)
            }
        }
        val assertions = ClinicalContext.assertContext(text, spans, sentences, language)
        val reason = if (assertions == null) ClinicalContext.UNSUPPORTED_LANGUAGE else null
        val entities = ArrayList<ClinicalEntity>(kept.size)
        for (index in kept.indices) {
            val entity = kept[index]
            entities += ClinicalEntity(
                categories[index],
                entity.start,
                entity.end,
                entity.confidence,
                modelId,
                assertions?.get(index),
                reason,
            )
        }
        return result(entities, modelId)
    }
}
