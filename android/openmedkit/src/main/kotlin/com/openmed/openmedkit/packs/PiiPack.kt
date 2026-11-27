package com.openmed.openmedkit.packs

import com.openmed.openmedkit.DeidentifyEngine
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMed
import com.openmed.openmedkit.PolicyDeidentificationResult
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File

/**
 * PII task pack: resolves a PII model from the bundled task pack manifest,
 * prepares it through [TaskPack.prepare], and runs `extractPii` and
 * `deidentify` on the [com.openmed.openmedkit.OpenMedKit] facade.
 *
 * Without a prepared model, [extract] and [deidentify] fail closed with
 * [TaskPackFailure.NOT_PREPARED]. Only when [allowLocalFallback] is `true` do
 * they run the deterministic local recognizer of [OpenMed] instead, and the
 * result then reports [TaskPackDetector.LOCAL_RECOGNIZER]. Results carry the
 * pack's review notice: automated detection can miss identifiers.
 *
 * @param catalog Maps candidate repositories to integrity-pinned entries,
 *   normally `ModelCatalog.load(context)`.
 * @param allowLocalFallback Run the local recognizer when no model is prepared.
 * @param classifierFactory Opens a classifier over a verified cache directory;
 *   the default loads `model_int8.onnx` with its tokenizer and label map.
 * @param manifest Task pack manifest bytes; the bundled file by default.
 */
class PiiPack(
    catalog: ModelCatalog,
    private val allowLocalFallback: Boolean = false,
    classifierFactory: (File) -> OnnxTokenClassifier = { defaultTaskPackClassifier(it) },
    manifest: ByteArray? = TaskPacks.bundledManifest(),
) : TaskPack(TaskPacks.PII, catalog, classifierFactory, manifest) {
    /** Detect PII spans above [confidenceThreshold], with span repair and merging. */
    suspend fun extract(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): TaskPackResult<List<EntityPrediction>> {
        val (kit, modelId) = preparedModel()
            ?: return result(local().extractPII(text, confidenceThreshold), null)
        return result(kit.extractPii(text, confidenceThreshold), modelId)
    }

    /** De-identify [text] under the bundled [policy] profile. */
    suspend fun deidentify(
        text: String,
        policy: String = PolicyProfiles.DEFAULT_PROFILE,
        confidenceThreshold: Float = 0.5f,
    ): TaskPackResult<PolicyDeidentificationResult> {
        val (kit, modelId) = preparedModel() ?: return result(
            DeidentifyEngine().deidentify(
                text,
                local().extractPII(text, confidenceThreshold),
                PolicyProfiles.load(policy),
            ),
            null,
        )
        return result(kit.deidentify(text, policy, confidenceThreshold), modelId)
    }

    private fun local(): OpenMed =
        if (allowLocalFallback) OpenMed() else throw TaskPackException(TaskPackFailure.NOT_PREPARED)
}
