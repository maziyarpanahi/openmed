package org.openmed.r8smoke

import android.content.Context
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.PolicyDeidentificationResult
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File
import kotlinx.coroutines.runBlocking
import org.json.JSONObject

/**
 * Exercises the OpenMedKit surface a host app uses, inside the minified APK.
 *
 * The run loads the bundled catalog and policy profiles, then builds the
 * public `OpenMedKit.fromDirectory` facade over a tiny synthetic ONNX model and
 * `tokenizer.json` bundled as assets, so DJL tokenizer construction, ONNX
 * Runtime session creation, decoding and redaction all execute through JNI
 * without any download. Every check catches `Throwable`, because the failures
 * R8 causes surface as `LinkageError`s rather than exceptions.
 */
object SmokeRunner {
    internal const val MODEL_ASSET_DIRECTORY = "r8-smoke-model"
    internal val MODEL_FILES = listOf("model.onnx", "tokenizer.json", "id2label.json")
    internal const val SMOKE_CASE_FILE = "smoke_case.json"

    // DJL's tokenizer factory sends a usage probe unless this property is set.
    // The manifest also removes INTERNET, so the kernel refuses inet sockets.
    private const val DJL_TRACKING_OPT_OUT = "OPT_OUT_TRACKING"

    fun run(context: Context): SmokeReport {
        System.setProperty(DJL_TRACKING_OPT_OUT, "true")
        val checks = mutableListOf<SmokeCheck>()
        checks += attempt("catalog") {
            val catalog = ModelCatalog.load(context)
            val status = if (catalog.entries.isNotEmpty()) SmokeStatus.PASS else SmokeStatus.FAIL
            SmokeCheck("catalog", status, count = catalog.filter(format = "onnx").size)
        }
        checks += attempt("policy_profiles") {
            val profile = PolicyProfiles.load()
            val status = if (profile.name == PolicyProfiles.DEFAULT_PROFILE) {
                SmokeStatus.PASS
            } else {
                SmokeStatus.FAIL
            }
            SmokeCheck("policy_profiles", status, count = PolicyProfiles.bundledProfileNames.size)
        }

        val case = try {
            context.assets.open("$MODEL_ASSET_DIRECTORY/$SMOKE_CASE_FILE").use { input ->
                SmokeCase.parse(input.bufferedReader(Charsets.UTF_8).readText())
            }
        } catch (error: Throwable) {
            checks += SmokeCheck.failure("smoke_case", error)
            return SmokeReport(FLAVOR, checks)
        }
        val modelDirectory = try {
            copyModel(context)
        } catch (error: Throwable) {
            checks += SmokeCheck.failure("model_assets", error)
            return SmokeReport(FLAVOR, checks)
        }
        checks += SmokeCheck("model_assets", SmokeStatus.PASS, count = MODEL_FILES.size)

        val kit = try {
            OpenMedKit.fromDirectory(modelDirectory, variant = "fp32")
        } catch (error: Throwable) {
            checks += SmokeCheck.failure("session", error)
            return SmokeReport(FLAVOR, checks)
        }
        checks += SmokeCheck("session", SmokeStatus.PASS)

        checks += attempt("extract_pii") {
            val entities = runBlocking { kit.extractPii(case.text) }
            evaluateEntities("extract_pii", entities, case.entities)
        }
        checks += attempt("extract_pii_chunked") {
            val entities = runBlocking {
                kit.extractPiiChunked(
                    case.text,
                    DEFAULT_CONFIDENCE_THRESHOLD,
                    case.chunkTokenLimit,
                    case.tokenOverlap,
                    true,
                )
            }
            evaluateEntities("extract_pii_chunked", entities, case.entities)
        }
        checks += attempt("deidentify") {
            evaluateRedaction(runBlocking { kit.deidentify(case.text, case.policy) }, case)
        }
        checks += attempt("close") {
            kit.close()
            SmokeCheck("close", SmokeStatus.PASS)
        }
        return SmokeReport(FLAVOR, checks)
    }

    private fun copyModel(context: Context): File {
        val directory = File(context.noBackupFilesDir, MODEL_ASSET_DIRECTORY)
        check(directory.isDirectory || directory.mkdirs()) { "model directory unavailable" }
        for (name in MODEL_FILES) {
            context.assets.open("$MODEL_ASSET_DIRECTORY/$name").use { input ->
                File(directory, name).outputStream().use { output -> input.copyTo(output) }
            }
        }
        return directory
    }

    private inline fun attempt(id: String, block: () -> SmokeCheck): SmokeCheck =
        try {
            block()
        } catch (error: Throwable) {
            SmokeCheck.failure(id, error)
        }

    private const val FLAVOR = "openmedkit"
    private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.5f
}

/** Expected outcome of the smoke run, generated with the synthetic model. */
internal class SmokeCase(
    val text: String,
    val entities: List<SmokeSpan>,
    val chunkTokenLimit: Int,
    val tokenOverlap: Int,
    val policy: String,
    val canonicalLabels: List<String>,
) {
    companion object {
        fun parse(json: String): SmokeCase {
            val root = JSONObject(json)
            require(root.getInt("schema_version") == 1) { "unsupported smoke case" }
            val entities = root.getJSONArray("entities")
            val chunked = root.getJSONObject("chunked")
            val deidentify = root.getJSONObject("deidentify")
            val labels = deidentify.getJSONArray("canonical_labels")
            return SmokeCase(
                text = root.getString("text"),
                entities = List(entities.length()) { index ->
                    val entity = entities.getJSONObject(index)
                    SmokeSpan(
                        entity.getString("label"),
                        entity.getInt("start"),
                        entity.getInt("end"),
                    )
                },
                chunkTokenLimit = chunked.getInt("chunk_token_limit"),
                tokenOverlap = chunked.getInt("token_overlap"),
                policy = deidentify.getString("policy"),
                canonicalLabels = List(labels.length()) { labels.getString(it) },
            )
        }
    }
}

/** Return [label] when it is safe to report, otherwise a fixed placeholder. */
internal fun reportableLabel(label: String): String =
    if (label.matches(SmokeCheck.LABEL_PATTERN)) label else "unreportable_label"

/** Compare detected entities with the expected labels and offsets. */
internal fun evaluateEntities(
    id: String,
    actual: List<EntityPrediction>,
    expected: List<SmokeSpan>,
): SmokeCheck {
    val spans = actual.map { SmokeSpan(reportableLabel(it.label), it.start, it.end) }
    val matches = spans == expected
    return SmokeCheck(
        id = id,
        status = if (matches) SmokeStatus.PASS else SmokeStatus.FAIL,
        count = spans.size,
        spans = spans,
        reason = if (matches) null else "spans_mismatch",
    )
}

/**
 * Check that every expected identifier was rewritten under the policy.
 *
 * The surface text is compared in memory only; the check reports the
 * canonical labels and offsets of the applied actions.
 */
internal fun evaluateRedaction(
    result: PolicyDeidentificationResult,
    case: SmokeCase,
): SmokeCheck {
    val retained = case.entities.count { span ->
        result.redactedText.contains(
            UnicodeOffsetContract.substring(case.text, span.start, span.end),
        )
    }
    val labels = result.actions.map { it.canonicalLabel }
    val reason = when {
        retained > 0 -> "surface_retained"
        labels != case.canonicalLabels -> "canonical_labels_mismatch"
        result.actions.any { it.replacement == null } -> "action_kept_surface"
        else -> null
    }
    return SmokeCheck(
        id = "deidentify",
        status = if (reason == null) SmokeStatus.PASS else SmokeStatus.FAIL,
        count = result.actions.size,
        spans = result.actions.map {
            SmokeSpan(reportableLabel(it.canonicalLabel), it.start, it.end)
        },
        reason = reason,
    )
}
