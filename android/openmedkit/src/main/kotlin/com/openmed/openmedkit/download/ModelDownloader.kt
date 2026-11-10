package com.openmed.openmedkit.download

import android.content.Context
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import java.io.File
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
        val actualIntegrityHash = ModelIntegrity.reproducibilityHash(
            repoId = entry.repoId,
            revisionSha = modelInfo.revisionSha,
            released = modelInfo.released,
            siblings = modelInfo.files.map { it.path },
        )
        if (actualIntegrityHash != entry.reproducibilityHash) {
            throw ModelIntegrityException(
                "Manifest integrity mismatch for ${entry.repoId}",
            )
        }

        val filesToDownload = requiredAndroidFiles(entry, modelInfo.files)
        if (filesToDownload.isEmpty()) {
            throw ModelDownloadException(
                "No Android-runnable files found for ${entry.repoId}",
            )
        }

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

    private fun verifyDownloadedFile(
        repoId: String,
        remoteFile: HuggingFaceModelFile,
        destination: File,
    ) {
        if (!destination.isFile) {
            throw ModelIntegrityException("Downloaded file missing for $repoId")
        }
        remoteFile.sizeBytes?.let { expectedSizeBytes ->
            if (destination.length() != expectedSizeBytes) {
                throw ModelIntegrityException("Downloaded file size mismatch for $repoId")
            }
        }
        remoteFile.sha256?.let { expectedSha256 ->
            if (sha256(destination) != expectedSha256.lowercase()) {
                throw ModelIntegrityException("Downloaded file checksum mismatch for $repoId")
            }
        }
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

        val wantsOnnx = formats.any { it.lowercase().startsWith("onnx") }
        val wantsTflite = formats.any { it.lowercase().startsWith("tflite") }
        if (wantsOnnx && (
                normalizedPath.endsWith(".onnx") ||
                    normalizedPath.endsWith(".ort") ||
                    normalizedPath.endsWith(".onnx_data") ||
                    normalizedPath.endsWith(".onnx.data")
                )
        ) {
            return true
        }
        if (wantsTflite && normalizedPath.endsWith(".tflite")) {
            return true
        }

        return fileName in REQUIRED_SIDECAR_FILES ||
            (fileName.endsWith(".json") && REQUIRED_JSON_HINTS.any { fileName.contains(it) })
    }

    private fun resolveSafePath(root: File, relativePath: String): File {
        val segments = relativePath.split("/")
        require(segments.isNotEmpty()) { "empty model file path" }
        var current = root
        segments.forEach { segment ->
            require(segment.isNotBlank()) { "blank model file path segment" }
            require(segment != "." && segment != "..") {
                "unsafe model file path segment"
            }
            require(!segment.contains('\\')) { "unsafe model file path segment" }
            current = File(current, segment)
        }
        val canonicalRoot = root.canonicalFile
        val canonicalFile = current.canonicalFile
        require(canonicalFile.path.startsWith(canonicalRoot.path + File.separator)) {
            "model file path escapes cache directory"
        }
        return current
    }

    private fun sha256(file: File): String {
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
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val DEFAULT_REVISION = "main"

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
}

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

class ModelDownloadException(message: String) : RuntimeException(message)

class ModelIntegrityException(message: String) : RuntimeException(message)

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
        val url = URL(
            "$resolveBaseUrl/${encodeRepoId(repoId)}/resolve/" +
                "${encodePathSegment(revision)}/${encodeFilePath(path)}?download=true",
        )
        openConnection(url).useBinaryResponse { input ->
            destination.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

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
