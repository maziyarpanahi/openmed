package com.openmed.openmedkit.packs

import com.openmed.openmedkit.BackendOnnxTokenClassifier
import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMedBackend
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.openCachedKit
import java.io.Closeable
import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Value-free reason a task pack cannot resolve, prepare or run. */
enum class TaskPackFailure {
    /** The bundled `task_packs.json` is missing or fails validation. */
    MANIFEST_INVALID,

    /** The manifest declares no pack with the requested id. */
    UNKNOWN_PACK,

    /** No Android candidate is declared or present in the model catalog. */
    NO_CANDIDATE,

    /** No candidate is cached and the caller did not allow a download. */
    NOT_CACHED,

    /** Every candidate download failed its transfer or integrity checks. */
    DOWNLOAD_FAILED,

    /** The cached model could not be opened. */
    LOAD_FAILED,

    /** No model is prepared and the pack has no opted-in fallback. */
    NOT_PREPARED,
}

/** Raised with a [TaskPackFailure]; the message is its lowercase name only. */
class TaskPackException(val failure: TaskPackFailure) :
    IllegalStateException(failure.name.lowercase())

/**
 * One task pack resolved for the Android runtime from the bundled manifest.
 *
 * @property repoIds Candidate ONNX repositories, best first; empty when the
 *   manifest records `no_candidate` for Android.
 * @property canonicalLabels Labels every candidate is guaranteed to emit.
 * @property noticeKey Key of the notice shown with pack output.
 * @property notice Default English text of [noticeKey].
 */
class TaskPackResolution internal constructor(
    val packId: String,
    val version: String,
    val task: String,
    val repoIds: List<String>,
    val canonicalLabels: List<String>,
    val noticeKey: String,
    val notice: String,
)

/**
 * Outcome of [TaskPack.prepare]: the prepared model id, or a typed [failure].
 *
 * @property fromCache `true` when no download was needed.
 */
class TaskPackPreparation internal constructor(
    val modelId: String?,
    val failure: TaskPackFailure?,
    val fromCache: Boolean,
)

/** Which detector produced a [TaskPackResult]. */
enum class TaskPackDetector {
    /** The prepared token-classification model named by `modelId`. */
    MODEL,

    /** The deterministic local recognizer the caller opted in to. */
    LOCAL_RECOGNIZER,
}

/**
 * Pack output labelled with its detector and the pack's notice.
 *
 * @property modelId Repository id of the model, or `null` for
 *   [TaskPackDetector.LOCAL_RECOGNIZER].
 */
class TaskPackResult<T> internal constructor(
    val value: T,
    val detector: TaskPackDetector,
    val modelId: String?,
    val noticeKey: String,
    val notice: String,
)

/**
 * Reader for the bundled `packs/task_packs.json`: a byte-identical copy of
 * `openmed/core/data/task_packs.json`, which
 * `scripts/manifest/build_task_packs.py` generates from `models.jsonl`.
 *
 * Resolution reads that file only. It never downloads, opens a socket or
 * loads a model.
 */
object TaskPacks {
    const val PII: String = "pii"
    const val CLINICAL_ENTITY: String = "clinical_entity"
    const val MEDICATION: String = "medication"

    internal fun bundledManifest(): ByteArray? =
        javaClass.classLoader?.getResourceAsStream("packs/task_packs.json")?.use { it.readBytes() }

    /**
     * Resolve [packId] for Android from [manifest], the bundled file by default.
     *
     * @throws TaskPackException with [TaskPackFailure.MANIFEST_INVALID] or
     *   [TaskPackFailure.UNKNOWN_PACK].
     */
    fun resolve(packId: String, manifest: ByteArray? = bundledManifest()): TaskPackResolution {
        var failure = TaskPackFailure.MANIFEST_INVALID
        try {
            val root = Json.parseToJsonElement(String(manifest!!, Charsets.UTF_8)).jsonObject
            check(root.text("schema") == "openmed.task_packs")
            check(root.text("schema_version").startsWith("1."))
            failure = TaskPackFailure.UNKNOWN_PACK
            for (element in root.getValue("packs").jsonArray) {
                val pack = element.jsonObject
                if (pack.text("pack_id") != packId) continue
                failure = TaskPackFailure.MANIFEST_INVALID
                val runtime = pack.getValue("runtimes").jsonObject.getValue("android").jsonObject
                val repoIds = ArrayList<String>()
                for (candidate in runtime.getValue("candidates").jsonArray) {
                    check(candidate.jsonObject.text("rank") == "${repoIds.size + 1}")
                    repoIds += candidate.jsonObject.text("repo_id")
                }
                check(runtime.text("format") == "onnx")
                check((runtime.text("status") == "resolved") == repoIds.isNotEmpty())
                val noticeKey = pack.text("notice_key")
                return TaskPackResolution(
                    packId = packId,
                    version = pack.text("version"),
                    task = pack.text("task"),
                    repoIds = repoIds,
                    canonicalLabels = pack.getValue("canonical_labels").jsonArray
                        .map { it.jsonPrimitive.content },
                    noticeKey = noticeKey,
                    notice = root.getValue("notices").jsonObject.text(noticeKey),
                )
            }
        } catch (_: RuntimeException) {
            failure = TaskPackFailure.MANIFEST_INVALID
        }
        throw TaskPackException(failure)
    }

    private fun JsonObject.text(key: String): String = getValue(key).jsonPrimitive.content
}

/**
 * Shared lifecycle of the Android task packs: resolve candidates from the
 * bundled manifest, prepare one from the cache or, with consent, a download,
 * and hold it open until [close].
 *
 * Preparation never touches the network unless `allowDownload` is `true`.
 * The prepared model holds a [com.openmed.openmedkit.cache.ModelCache] lease,
 * so eviction and budget trimming cannot remove it while the pack is open.
 * [close] releases the model; [prepare] may be called again afterwards.
 */
abstract class TaskPack internal constructor(
    packId: String,
    private val catalog: ModelCatalog,
    private val classifierFactory: (File) -> OnnxTokenClassifier,
    manifest: ByteArray?,
) : Closeable {
    /** Android candidates, canonical labels and notice of this pack. */
    val resolution: TaskPackResolution = TaskPacks.resolve(packId, manifest)

    private var prepared: Pair<OpenMedKit, String>? = null

    /** Repository id of the prepared model, or `null` before [prepare]. */
    val modelId: String?
        get() = preparedModel()?.second

    /**
     * Prepare the best candidate. Blocking: call it off the main thread.
     *
     * Cached candidates are tried first, in rank order. When none is cached
     * the result is [TaskPackFailure.NOT_CACHED] and the downloader's client
     * is never called, unless [allowDownload] is `true`; then candidates are
     * downloaded in rank order until one passes its integrity checks.
     */
    fun prepare(downloader: ModelDownloader, allowDownload: Boolean = false): TaskPackPreparation {
        modelId?.let { return TaskPackPreparation(it, null, true) }
        val entries = resolution.repoIds.mapNotNull(catalog::byRepoId)
        val cache = downloader.cache
        val cached = entries.firstOrNull(cache::isAvailable)
        val entry = cached ?: entries.firstOrNull { allowDownload && downloads(downloader, it) }
        if (entry == null) {
            val failure = when {
                entries.isEmpty() -> TaskPackFailure.NO_CANDIDATE
                allowDownload -> TaskPackFailure.DOWNLOAD_FAILED
                else -> TaskPackFailure.NOT_CACHED
            }
            return TaskPackPreparation(null, failure, false)
        }
        val kit = try {
            openCachedKit(cache, entry, classifierFactory)
        } catch (_: Exception) {
            return TaskPackPreparation(null, TaskPackFailure.LOAD_FAILED, false)
        }
        synchronized(this) {
            if (prepared == null) prepared = kit to entry.repoId else kit.close()
        }
        return TaskPackPreparation(modelId, null, cached != null)
    }

    /** Release the prepared model and its cache lease. Idempotent. */
    override fun close() {
        synchronized(this) { prepared.also { prepared = null } }?.first?.close()
    }

    internal fun preparedModel(): Pair<OpenMedKit, String>? = synchronized(this) { prepared }

    internal fun <T> result(value: T, modelId: String?) = TaskPackResult(
        value,
        if (modelId == null) TaskPackDetector.LOCAL_RECOGNIZER else TaskPackDetector.MODEL,
        modelId,
        resolution.noticeKey,
        resolution.notice,
    )

    private fun downloads(downloader: ModelDownloader, entry: ModelCatalogEntry): Boolean =
        try {
            downloader.download(entry)
            true
        } catch (_: Exception) {
            false
        }
}

/** Opens the int8 graph of a cached ONNX directory, like `OpenMedKit.fromDirectory`. */
internal fun defaultTaskPackClassifier(directory: File): OnnxTokenClassifier =
    BackendOnnxTokenClassifier(OpenMedBackend(directory, File(directory, "model_int8.onnx")))
