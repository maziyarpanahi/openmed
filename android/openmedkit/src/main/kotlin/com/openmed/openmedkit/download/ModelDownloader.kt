package com.openmed.openmedkit.download

import android.content.Context
import com.openmed.openmedkit.cache.CachedArtifactFile
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.cache.UnsafeModelPathException
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import java.io.EOFException
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.security.MessageDigest

/**
 * Downloads Hugging Face model artifacts and marks them ready only after
 * manifest and file integrity checks pass.
 */
class ModelDownloader(
    private val cache: ModelCache,
    private val client: HuggingFaceModelClient = HttpHuggingFaceModelClient(),
    private val logger: ModelDownloadLogger = ModelDownloadLogger.NONE,
) {
    constructor(
        context: Context,
        cacheBudgetBytes: Long = ModelCache.DEFAULT_CACHE_BUDGET_BYTES,
        client: HuggingFaceModelClient = HttpHuggingFaceModelClient(),
        logger: ModelDownloadLogger = ModelDownloadLogger.NONE,
    ) : this(
        ModelCache(context, cacheBudgetBytes),
        client,
        logger,
    )

    fun download(
        entry: ModelCatalogEntry,
        revision: String? = null,
    ): ModelDownloadResult {
        if (cache.isAvailable(entry)) {
            val sizeBytes = cache.sizeBytes(entry.repoId)
            logger.log(
                ModelDownloadLogEvent(
                    repoId = entry.repoId,
                    status = ModelDownloadStatus.CACHE_HIT,
                    sizeBytes = sizeBytes,
                ),
            )
            return ModelDownloadResult(
                repoId = entry.repoId,
                directory = cache.modelDirectory(entry.repoId),
                totalSizeBytes = sizeBytes,
                filesDownloaded = 0,
                fromCache = true,
                integrityHash = entry.reproducibilityHash,
            )
        }

        logger.log(
            ModelDownloadLogEvent(
                repoId = entry.repoId,
                status = ModelDownloadStatus.DOWNLOAD_STARTED,
                sizeBytes = 0L,
            ),
        )

        val modelInfo = client.fetchModelInfo(entry.repoId, revision)
        val actualIntegrityHash = verifiedManifestHash(entry, modelInfo)
        val filesToDownload = runnableFiles(entry, modelInfo)

        val stagingDirectory = cache.createStagingDirectory(entry.repoId)
        var completed = false
        try {
            val downloadRevision = revision ?: modelInfo.revisionSha ?: DEFAULT_REVISION
            filesToDownload.forEach { remoteFile ->
                val destination = resolveSafePath(stagingDirectory, remoteFile.path)
                destination.parentFile?.mkdirs()
                client.downloadFile(
                    repoId = entry.repoId,
                    path = remoteFile.path,
                    revision = downloadRevision,
                    destination = destination,
                )
                verifyDownloadedFile(entry.repoId, remoteFile, destination)
            }

            val readyDirectory = cache.storeReadyModel(
                repoId = entry.repoId,
                sourceDirectory = stagingDirectory,
                expectedIntegrityHash = entry.reproducibilityHash,
                actualIntegrityHash = actualIntegrityHash,
                revisionSha = modelInfo.revisionSha,
            )
            val sizeBytes = cache.sizeBytes(entry.repoId)
            completed = true
            logger.log(
                ModelDownloadLogEvent(
                    repoId = entry.repoId,
                    status = ModelDownloadStatus.DOWNLOAD_COMPLETE,
                    sizeBytes = sizeBytes,
                ),
            )
            return ModelDownloadResult(
                repoId = entry.repoId,
                directory = readyDirectory,
                totalSizeBytes = sizeBytes,
                filesDownloaded = filesToDownload.size,
                fromCache = false,
                integrityHash = actualIntegrityHash,
            )
        } finally {
            if (!completed) {
                stagingDirectory.deleteRecursively()
            }
        }
    }

    /**
     * Stage the runnable files for [entry] into [stagingDirectory] and verify them.
     *
     * The staging directory is stable across attempts: a file whose size and
     * SHA-256 are published is resumed with an HTTP Range request from the bytes
     * already staged, then re-hashed in full. A resumed file that fails the
     * re-check is discarded and fetched once from the start. Nothing is promoted
     * to the cache here; the caller decides when the staged set becomes ready.
     */
    internal fun stageArtifact(
        entry: ModelCatalogEntry,
        revision: String?,
        stagingDirectory: File,
        expectedTotalBytes: Long?,
        control: ModelTransferControl,
    ): StagedModelArtifact {
        control.throwIfCancelled()
        val modelInfo = client.fetchModelInfo(entry.repoId, revision)
        val integrityHash = verifiedManifestHash(entry, modelInfo)
        val filesToDownload = runnableFiles(entry, modelInfo)
        if (filesToDownload.none { isFormatPayload(it.path.lowercase(), entry.formats) }) {
            // Sidecars alone are not an artifact of the requested format.
            throw ModelDownloadException(
                "No Android-runnable files found for ${entry.repoId}",
                ModelFailure.NO_RUNNABLE_FILES,
            )
        }
        val publishedSizes = filesToDownload.mapNotNull { it.sizeBytes }
        if (expectedTotalBytes != null &&
            publishedSizes.size == filesToDownload.size &&
            publishedSizes.sum() != expectedTotalBytes
        ) {
            throw ModelIntegrityException("Artifact size mismatch", ModelFailure.SIZE)
        }

        val destinations = filesToDownload.map { resolveSafePath(stagingDirectory, it.path) }
        val expectedPaths = destinations.map { it.canonicalPath }.toSet()
        stagingDirectory.walkBottomUp()
            .filter { it != stagingDirectory && it.isFile && it.canonicalPath !in expectedPaths }
            .forEach { it.delete() }

        val downloadRevision = modelInfo.revisionSha ?: revision ?: DEFAULT_REVISION
        var resumed = false
        var bytesTransferred = 0L
        val stagedFiles = filesToDownload.zip(destinations) { remoteFile, destination ->
            destination.parentFile?.mkdirs()
            val offset = resumableOffset(remoteFile, destination)
            if (offset == STAGED_FILE_COMPLETE) {
                resumed = true
                return@zip CachedArtifactFile(
                    remoteFile.path,
                    destination.length(),
                    requireNotNull(remoteFile.sha256).lowercase(),
                )
            }
            var outcome =
                transferFile(entry, remoteFile, downloadRevision, destination, offset, control)
            bytesTransferred += outcome.bytesTransferred
            val sha256 = try {
                verifyDownloadedFile(entry.repoId, remoteFile, destination)
            } catch (error: ModelIntegrityException) {
                if (outcome.startOffset == 0L) {
                    throw error
                }
                destination.delete()
                outcome =
                    transferFile(entry, remoteFile, downloadRevision, destination, 0L, control)
                bytesTransferred += outcome.bytesTransferred
                verifyDownloadedFile(entry.repoId, remoteFile, destination)
            }
            if (outcome.startOffset > 0L) {
                resumed = true
            }
            CachedArtifactFile(remoteFile.path, destination.length(), sha256)
        }

        val stagedBytes = stagedFiles.sumOf { it.sizeBytes }
        if (expectedTotalBytes != null && stagedBytes != expectedTotalBytes) {
            throw ModelIntegrityException("Artifact size mismatch", ModelFailure.SIZE)
        }
        return StagedModelArtifact(
            files = stagedFiles,
            integrityHash = integrityHash,
            revisionSha = modelInfo.revisionSha,
            resumed = resumed,
            bytesTransferred = bytesTransferred,
        )
    }

    private fun transferFile(
        entry: ModelCatalogEntry,
        remoteFile: HuggingFaceModelFile,
        revision: String,
        destination: File,
        resumeFromBytes: Long,
        control: ModelTransferControl,
    ): ModelFileTransferOutcome {
        if (resumeFromBytes == 0L) {
            destination.delete()
        }
        val outcome = client.downloadFile(
            repoId = entry.repoId,
            path = remoteFile.path,
            revision = revision,
            destination = destination,
            transfer = ModelFileTransfer(resumeFromBytes, remoteFile.sizeBytes, control),
        )
        control.throwIfCancelled()
        val expectedSizeBytes = remoteFile.sizeBytes
        if (expectedSizeBytes != null && destination.length() < expectedSizeBytes) {
            // A short body is an interrupted transfer: keep the bytes for resume.
            throw EOFException("Model file transfer ended early")
        }
        return outcome
    }

    /**
     * Bytes already staged that are safe to resume from, or
     * [STAGED_FILE_COMPLETE] when the staged file is complete and verified.
     * Files without a published size and SHA-256 always restart from zero.
     */
    private fun resumableOffset(remoteFile: HuggingFaceModelFile, destination: File): Long {
        val expectedSizeBytes = remoteFile.sizeBytes
        val expectedSha256 = remoteFile.sha256
        if (!destination.isFile || expectedSizeBytes == null || expectedSha256 == null) {
            return 0L
        }
        val stagedBytes = destination.length()
        return when {
            stagedBytes in 1 until expectedSizeBytes -> stagedBytes
            stagedBytes == expectedSizeBytes &&
                sha256Hex(destination) == expectedSha256.lowercase() -> STAGED_FILE_COMPLETE
            else -> 0L
        }
    }

    private fun verifiedManifestHash(
        entry: ModelCatalogEntry,
        modelInfo: HuggingFaceModelInfo,
    ): String {
        val actualIntegrityHash = ModelIntegrity.reproducibilityHash(
            repoId = entry.repoId,
            revisionSha = modelInfo.revisionSha,
            released = modelInfo.released,
            siblings = modelInfo.files.map { it.path },
        )
        if (actualIntegrityHash != entry.reproducibilityHash) {
            throw ModelIntegrityException(
                "Manifest integrity mismatch for ${entry.repoId}",
                ModelFailure.MANIFEST,
            )
        }
        return actualIntegrityHash
    }

    private fun runnableFiles(
        entry: ModelCatalogEntry,
        modelInfo: HuggingFaceModelInfo,
    ): List<HuggingFaceModelFile> {
        val filesToDownload = requiredAndroidFiles(entry, modelInfo.files)
        if (filesToDownload.isEmpty()) {
            throw ModelDownloadException(
                "No Android-runnable files found for ${entry.repoId}",
                ModelFailure.NO_RUNNABLE_FILES,
            )
        }
        return filesToDownload
    }

    private fun verifyDownloadedFile(
        repoId: String,
        remoteFile: HuggingFaceModelFile,
        destination: File,
    ): String {
        if (!destination.isFile) {
            throw ModelIntegrityException(
                "Downloaded file missing for $repoId",
                ModelFailure.MISSING_FILE,
            )
        }
        remoteFile.sizeBytes?.let { expectedSizeBytes ->
            if (destination.length() != expectedSizeBytes) {
                throw ModelIntegrityException(
                    "Downloaded file size mismatch for $repoId",
                    ModelFailure.SIZE,
                )
            }
        }
        val actualSha256 = sha256Hex(destination)
        remoteFile.sha256?.let { expectedSha256 ->
            if (actualSha256 != expectedSha256.lowercase()) {
                throw ModelIntegrityException(
                    "Downloaded file checksum mismatch for $repoId",
                    ModelFailure.CHECKSUM,
                )
            }
        }
        return actualSha256
    }

    /** Model payload of a requested format, as opposed to a tokenizer or config sidecar. */
    private fun isFormatPayload(normalizedPath: String, formats: List<String>): Boolean {
        val wantsOnnx = formats.any { it.lowercase().startsWith("onnx") }
        val wantsTflite = formats.any { it.lowercase().startsWith("tflite") }
        return (
            wantsOnnx && (
                normalizedPath.endsWith(".onnx") ||
                    normalizedPath.endsWith(".ort") ||
                    normalizedPath.endsWith(".onnx_data") ||
                    normalizedPath.endsWith(".onnx.data")
                )
            ) ||
            (wantsTflite && normalizedPath.endsWith(".tflite"))
    }

    private fun requiredAndroidFiles(
        entry: ModelCatalogEntry,
        files: List<HuggingFaceModelFile>,
    ): List<HuggingFaceModelFile> =
        files
            .filter { file -> isRequiredAndroidFile(file.path, entry.formats) }
            .sortedBy { it.path }

    private fun isRequiredAndroidFile(path: String, formats: List<String>): Boolean {
        val normalizedPath = path.lowercase()
        val fileName = normalizedPath.substringAfterLast("/")
        if (fileName == ".gitattributes" ||
            fileName == "readme.md" ||
            fileName.startsWith("license")
        ) {
            return false
        }

        if (isFormatPayload(normalizedPath, formats)) {
            return true
        }

        return fileName in REQUIRED_SIDECAR_FILES ||
            (fileName.endsWith(".json") && REQUIRED_JSON_HINTS.any { fileName.contains(it) })
    }

    private fun resolveSafePath(root: File, relativePath: String): File {
        val segments = relativePath.split("/")
        var current = root
        segments.forEach { segment ->
            if (segment.isBlank() || segment == "." || segment == ".." || segment.contains('\\')) {
                throw UnsafeModelPathException()
            }
            current = File(current, segment)
        }
        val canonicalRoot = root.canonicalFile
        val canonicalFile = current.canonicalFile
        if (!canonicalFile.path.startsWith(canonicalRoot.path + File.separator)) {
            throw UnsafeModelPathException()
        }
        return current
    }

    private companion object {
        const val DEFAULT_REVISION = "main"
        const val STAGED_FILE_COMPLETE = -1L

        val REQUIRED_SIDECAR_FILES = setOf(
            "added_tokens.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "vocab.txt",
        )

        val REQUIRED_JSON_HINTS = listOf(
            "config",
            "preprocessor",
            "special_tokens",
            "tokenizer",
        )
    }
}

data class ModelDownloadResult(
    val repoId: String,
    val directory: File,
    val totalSizeBytes: Long,
    val filesDownloaded: Int,
    val fromCache: Boolean,
    val integrityHash: String,
)

fun interface ModelDownloadLogger {
    fun log(event: ModelDownloadLogEvent)

    companion object {
        val NONE = ModelDownloadLogger {}
    }
}

data class ModelDownloadLogEvent(
    val repoId: String,
    val status: ModelDownloadStatus,
    val sizeBytes: Long?,
)

enum class ModelDownloadStatus {
    CACHE_HIT,
    DOWNLOAD_STARTED,
    DOWNLOAD_COMPLETE,
}

interface HuggingFaceModelClient {
    fun fetchModelInfo(repoId: String, revision: String? = null): HuggingFaceModelInfo

    fun downloadFile(
        repoId: String,
        path: String,
        revision: String,
        destination: File,
    )

    /**
     * Download one file, resuming from [ModelFileTransfer.resumeFromBytes] when
     * the client supports it.
     *
     * The default implementation performs a full download through the
     * four-argument [downloadFile] and reports a start offset of zero, so custom
     * clients keep working and simply never resume. Implementations must append
     * only when the server confirms the requested range, must stop with
     * [ModelDownloadCancelledException] once [ModelTransferControl.isCancelled]
     * returns `true`, and must never put URLs, query values, credentials, or
     * file contents into exception messages.
     */
    fun downloadFile(
        repoId: String,
        path: String,
        revision: String,
        destination: File,
        transfer: ModelFileTransfer,
    ): ModelFileTransferOutcome {
        transfer.control.throwIfCancelled()
        downloadFile(repoId, path, revision, destination)
        return ModelFileTransferOutcome(startOffset = 0L, bytesTransferred = destination.length())
    }
}

/** Cooperative cancellation signal checked between transferred chunks. */
fun interface ModelTransferControl {
    fun isCancelled(): Boolean

    companion object {
        val NONE = ModelTransferControl { false }
    }
}

/** Options for one resumable file transfer. */
class ModelFileTransfer(
    val resumeFromBytes: Long = 0L,
    val expectedSizeBytes: Long? = null,
    val control: ModelTransferControl = ModelTransferControl.NONE,
)

/** Where a file transfer started and how many bytes it fetched. */
class ModelFileTransferOutcome(
    val startOffset: Long,
    val bytesTransferred: Long,
)

/** A transfer stopped because its [ModelTransferControl] was cancelled. */
class ModelDownloadCancelledException : RuntimeException("Model download cancelled")

internal fun ModelTransferControl.throwIfCancelled() {
    if (isCancelled()) {
        throw ModelDownloadCancelledException()
    }
}

/** Files staged and verified by [ModelDownloader.stageArtifact]. */
internal class StagedModelArtifact(
    val files: List<CachedArtifactFile>,
    val integrityHash: String,
    val revisionSha: String?,
    val resumed: Boolean,
    val bytesTransferred: Long,
)

data class HuggingFaceModelInfo(
    val repoId: String,
    val revisionSha: String?,
    val released: String?,
    val files: List<HuggingFaceModelFile>,
)

data class HuggingFaceModelFile(
    val path: String,
    val sizeBytes: Long? = null,
    val sha256: String? = null,
)

class ModelDownloadException internal constructor(
    message: String,
    internal val failure: ModelFailure,
    internal val httpStatus: Int? = null,
) : RuntimeException(message) {
    constructor(message: String) : this(message, ModelFailure.UNSPECIFIED)
}

class ModelIntegrityException internal constructor(
    message: String,
    internal val failure: ModelFailure,
) : RuntimeException(message) {
    constructor(message: String) : this(message, ModelFailure.UNSPECIFIED)
}

/** Typed cause carried by download and integrity exceptions. */
internal enum class ModelFailure {
    UNSPECIFIED,
    HTTP_STATUS,
    NO_RUNNABLE_FILES,
    MANIFEST,
    SIZE,
    CHECKSUM,
    MISSING_FILE,
}

/** Lowercase hex SHA-256 of [file], streamed. */
internal fun sha256Hex(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val read = input.read(buffer)
            if (read == -1) {
                break
            }
            digest.update(buffer, 0, read)
        }
    }
    return hex(digest.digest())
}

internal fun hex(bytes: ByteArray): String {
    val builder = StringBuilder(bytes.size * 2)
    bytes.forEach { byte ->
        val value = byte.toInt() and 0xff
        builder.append(HEX_DIGITS[value ushr 4]).append(HEX_DIGITS[value and 0x0f])
    }
    return builder.toString()
}

private const val HEX_DIGITS = "0123456789abcdef"

class HttpHuggingFaceModelClient(
    private val modelApiBaseUrl: String = "https://huggingface.co/api/models",
    private val resolveBaseUrl: String = "https://huggingface.co",
    private val connectTimeoutMillis: Int = 15_000,
    private val readTimeoutMillis: Int = 60_000,
) : HuggingFaceModelClient {
    override fun fetchModelInfo(
        repoId: String,
        revision: String?,
    ): HuggingFaceModelInfo {
        val suffix = revision?.let { "/revision/${encodePathSegment(it)}" }.orEmpty()
        val url = URL("$modelApiBaseUrl/${encodeRepoId(repoId)}$suffix?blobs=true")
        val payload = openConnection(url).useJsonResponse()
        val jsonObject = json.parseToJsonElement(payload).jsonObject
        val files = jsonObject["siblings"]
            ?.jsonArray
            ?.mapNotNull { sibling ->
                val siblingObject = sibling.jsonObject
                val path = siblingObject.stringValue("rfilename") ?: return@mapNotNull null
                val lfsObject = siblingObject["lfs"]?.jsonObject
                HuggingFaceModelFile(
                    path = path,
                    sizeBytes = lfsObject?.longValue("size") ?: siblingObject.longValue("size"),
                    sha256 = lfsObject?.stringValue("sha256")
                        ?: lfsObject?.stringValue("oid")
                        ?: siblingObject.stringValue("sha256"),
                )
            }
            ?: emptyList()
        return HuggingFaceModelInfo(
            repoId = repoId,
            revisionSha = jsonObject.stringValue("sha"),
            released = normalizeDate(
                jsonObject.stringValue("lastModified")
                    ?: jsonObject.stringValue("createdAt"),
            ),
            files = files,
        )
    }

    override fun downloadFile(
        repoId: String,
        path: String,
        revision: String,
        destination: File,
    ) {
        downloadFile(repoId, path, revision, destination, ModelFileTransfer())
    }

    /**
     * Download with HTTP Range resume.
     *
     * A non-zero resume offset sends `Range: bytes=<offset>-` and appends only
     * when the server answers `206` with a matching `Content-Range` start.
     * A `200`, a mismatched range, or `416` restarts the file from zero. A body
     * shorter than its `Content-Length` raises an [EOFException] and keeps the
     * received bytes for the next resume.
     */
    override fun downloadFile(
        repoId: String,
        path: String,
        revision: String,
        destination: File,
        transfer: ModelFileTransfer,
    ): ModelFileTransferOutcome {
        transfer.control.throwIfCancelled()
        val resumeFromBytes = transfer.resumeFromBytes
            .takeIf { it > 0L && destination.isFile && destination.length() == it }
            ?: 0L
        val url = URL(
            "$resolveBaseUrl/${encodeRepoId(repoId)}/resolve/" +
                "${encodePathSegment(revision)}/${encodeFilePath(path)}?download=true",
        )
        val connection = openConnection(url)
        if (resumeFromBytes > 0L) {
            connection.setRequestProperty("Range", "bytes=$resumeFromBytes-")
        }
        try {
            val statusCode = connection.responseCode
            val partial = statusCode == HttpURLConnection.HTTP_PARTIAL
            val append = resumeFromBytes > 0L &&
                partial &&
                contentRangeStart(connection) == resumeFromBytes
            if (resumeFromBytes > 0L &&
                !append &&
                (partial || statusCode == HTTP_RANGE_NOT_SATISFIABLE)
            ) {
                connection.disconnect()
                return downloadFile(
                    repoId,
                    path,
                    revision,
                    destination,
                    ModelFileTransfer(0L, transfer.expectedSizeBytes, transfer.control),
                )
            }
            if (statusCode !in 200..299 || (partial && !append)) {
                throw ModelDownloadException(
                    "Hugging Face request failed with status $statusCode",
                    ModelFailure.HTTP_STATUS,
                    statusCode,
                )
            }
            val startOffset = if (append) resumeFromBytes else 0L
            val declaredBodyBytes = connection.contentLengthLong
            val written = connection.inputStream.use { input ->
                FileOutputStream(destination, append).use { output ->
                    copyWithControl(input, output, startOffset, transfer)
                }
            }
            if (declaredBodyBytes >= 0L && written != declaredBodyBytes) {
                throw EOFException("Model file transfer ended early")
            }
            return ModelFileTransferOutcome(startOffset, written)
        } finally {
            connection.disconnect()
        }
    }

    private fun copyWithControl(
        input: InputStream,
        output: OutputStream,
        startOffset: Long,
        transfer: ModelFileTransfer,
    ): Long {
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        var written = 0L
        while (true) {
            transfer.control.throwIfCancelled()
            val read = input.read(buffer)
            if (read < 0) {
                return written
            }
            val expectedSizeBytes = transfer.expectedSizeBytes
            if (expectedSizeBytes != null && startOffset + written + read > expectedSizeBytes) {
                throw ModelIntegrityException(
                    "Downloaded file size mismatch",
                    ModelFailure.SIZE,
                )
            }
            output.write(buffer, 0, read)
            written += read
        }
    }

    private fun contentRangeStart(connection: HttpURLConnection): Long? =
        connection.getHeaderField("Content-Range")
            ?.trim()
            ?.removePrefix("bytes")
            ?.trim()
            ?.substringBefore('-')
            ?.toLongOrNull()

    private fun openConnection(url: URL): HttpURLConnection =
        (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = connectTimeoutMillis
            readTimeout = readTimeoutMillis
            requestMethod = "GET"
            instanceFollowRedirects = true
        }

    private fun HttpURLConnection.useJsonResponse(): String =
        useBinaryResponse { input ->
            input.reader(Charsets.UTF_8).readText()
        }

    private fun <T> HttpURLConnection.useBinaryResponse(block: (java.io.InputStream) -> T): T {
        try {
            val statusCode = responseCode
            if (statusCode !in 200..299) {
                throw ModelDownloadException(
                    "Hugging Face request failed with status $statusCode",
                    ModelFailure.HTTP_STATUS,
                    statusCode,
                )
            }
            return inputStream.use(block)
        } finally {
            disconnect()
        }
    }

    private fun JsonObject.stringValue(key: String): String? =
        this[key]?.jsonPrimitive?.contentOrNull

    private fun JsonObject.longValue(key: String): Long? =
        this[key]?.jsonPrimitive?.longOrNull

    private fun encodeRepoId(repoId: String): String =
        repoId.split("/").joinToString("/") { encodePathSegment(it) }

    private fun encodeFilePath(path: String): String =
        path.split("/").joinToString("/") { encodePathSegment(it) }

    private fun encodePathSegment(segment: String): String =
        URLEncoder.encode(segment, Charsets.UTF_8.name()).replace("+", "%20")

    private companion object {
        const val HTTP_RANGE_NOT_SATISFIABLE = 416

        val json = Json {
            ignoreUnknownKeys = true
        }
    }
}

object ModelIntegrity {
    fun reproducibilityHash(
        repoId: String,
        revisionSha: String?,
        released: String?,
        siblings: List<String>,
    ): String {
        val payload = buildString {
            append("{")
            append("\"released\":")
            appendJsonStringOrNull(normalizeDate(released))
            append(",\"repo_id\":")
            appendJsonString(repoId)
            append(",\"sha\":")
            appendJsonStringOrNull(revisionSha)
            append(",\"siblings\":[")
            siblings.sorted().forEachIndexed { index, sibling ->
                if (index > 0) {
                    append(",")
                }
                appendJsonString(sibling)
            }
            append("]}")
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(payload.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        return "sha256:$digest"
    }

    private fun StringBuilder.appendJsonStringOrNull(value: String?) {
        if (value == null) {
            append("null")
        } else {
            appendJsonString(value)
        }
    }

    private fun StringBuilder.appendJsonString(value: String) {
        append("\"")
        value.forEach { character ->
            when (character) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(character)
            }
        }
        append("\"")
    }
}

private fun normalizeDate(value: String?): String? =
    value?.take(10)?.ifBlank { null }
