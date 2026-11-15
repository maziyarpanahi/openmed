package com.openmed.openmedkit.cache

import android.content.Context
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.io.File
import java.security.MessageDigest
import java.util.UUID

/**
 * Offline-first app-private cache for downloaded OpenMed model artifacts.
 */
class ModelCache(
    internal val rootDirectory: File,
    private val cacheBudgetBytes: Long = DEFAULT_CACHE_BUDGET_BYTES,
    private val clock: ModelCacheClock = SystemModelCacheClock,
) {
    constructor(
        context: Context,
        cacheBudgetBytes: Long = DEFAULT_CACHE_BUDGET_BYTES,
        clock: ModelCacheClock = SystemModelCacheClock,
    ) : this(
        File(context.noBackupFilesDir, CACHE_DIRECTORY_NAME),
        cacheBudgetBytes,
        clock,
    )

    init {
        require(cacheBudgetBytes >= 0) { "cacheBudgetBytes must be non-negative" }
        rootDirectory.mkdirs()
    }

    @Synchronized
    fun isAvailable(repoId: String): Boolean =
        isAvailable(repoId, expectedIntegrityHash = null)

    @Synchronized
    fun isAvailable(entry: ModelCatalogEntry): Boolean =
        isAvailable(entry.repoId, entry.reproducibilityHash)

    @Synchronized
    fun totalSizeBytes(): Long =
        readyModelDirectories().sumOf { payloadSizeBytes(it.directory) }

    @Synchronized
    fun sizeBytes(repoId: String): Long {
        val directory = modelDirectory(repoId)
        return if (isReadyDirectory(directory, expectedIntegrityHash = null)) {
            payloadSizeBytes(directory)
        } else {
            0L
        }
    }

    @Synchronized
    fun evict(repoId: String): Boolean {
        val directory = modelDirectory(repoId)
        val existed = directory.exists()
        if (existed) {
            directory.deleteRecursively()
        }
        return existed
    }

    fun modelDirectory(repoId: String): File =
        File(rootDirectory, cacheKey(repoId))

    @Synchronized
    internal fun createStagingDirectory(repoId: String): File {
        rootDirectory.mkdirs()
        val stagingRoot = File(rootDirectory, STAGING_DIRECTORY_NAME)
        stagingRoot.mkdirs()
        return File(
            stagingRoot,
            "${cacheKey(repoId)}-${clock.nowEpochMillis()}-${UUID.randomUUID()}",
        ).also { it.mkdirs() }
    }

    @Synchronized
    internal fun storeReadyModel(
        repoId: String,
        sourceDirectory: File,
        expectedIntegrityHash: String,
        actualIntegrityHash: String,
        revisionSha: String?,
    ): File {
        val destination = modelDirectory(repoId)
        promote(sourceDirectory, destination)
        writeReadyMetadata(
            destination,
            repoId,
            expectedIntegrityHash,
            actualIntegrityHash,
            revisionSha,
            artifact = null,
        )
        trimToBudget(protectedDirectory = destination)
        return destination
    }

    /**
     * Cache directory for one format-aware artifact identity.
     *
     * Artifact directories are keyed by the full identity (repository, revision,
     * format, runtime, and digest), so ONNX and TFLite variants of one
     * repository never overwrite each other. The resolved path must stay inside
     * the cache root after symlink resolution.
     */
    @Synchronized
    internal fun artifactDirectory(identity: String): File =
        requireContained(File(rootDirectory, cacheKey(ARTIFACT_KEY_PREFIX + identity)))

    /**
     * Stable staging directory that keeps downloaded bytes for resume.
     *
     * Unlike [createStagingDirectory], the name derives from the artifact
     * identity, so an interrupted or cancelled acquisition can continue with an
     * HTTP Range request. Staging content is never reported as ready.
     */
    @Synchronized
    internal fun resumableStagingDirectory(identity: String): File =
        resumableStagingPath(identity).also { it.mkdirs() }

    @Synchronized
    internal fun hasResumableBytes(identity: String): Boolean {
        val directory = resumableStagingPath(identity)
        return directory.isDirectory &&
            directory.walkTopDown().any { it.isFile && it.length() > 0L }
    }

    @Synchronized
    internal fun storeReadyArtifact(
        repoId: String,
        identity: String,
        sourceDirectory: File,
        integrityHash: String,
        revisionSha: String?,
        artifact: CachedArtifactRecord,
    ): File {
        val destination = artifactDirectory(identity)
        promote(sourceDirectory, destination)
        destination.walkTopDown()
            .filter { it.isFile && it.name != METADATA_FILE_NAME }
            .forEach { it.setWritable(false, false) }
        writeReadyMetadata(
            destination,
            repoId,
            integrityHash,
            integrityHash,
            revisionSha,
            artifact,
        )
        trimToBudget(protectedDirectory = destination)
        return destination
    }

    /**
     * Ready artifact record for [identity], or `null` when it is missing,
     * partial, or no longer matches its recorded digest, file set, or sizes.
     */
    @Synchronized
    internal fun readyArtifact(
        identity: String,
        expectedIntegrityHash: String,
    ): CachedArtifactRecord? {
        val directory = artifactDirectory(identity)
        val metadata = readyMetadata(directory, expectedIntegrityHash) ?: return null
        val artifact = metadata.artifact ?: return null
        val filesIntact = artifact.files.all { file ->
            val payload = artifactFile(directory, file.path)
            payload != null && payload.isFile && payload.length() == file.sizeBytes
        }
        if (!filesIntact || artifact.files.sumOf { it.sizeBytes } != metadata.totalSizeBytes) {
            return null
        }
        writeMetadata(
            directory,
            metadata.copy(lastAccessedEpochMillis = clock.nowEpochMillis()),
        )
        return artifact
    }

    /** Resolve a recorded artifact file, or `null` when it escapes [directory]. */
    internal fun artifactFile(directory: File, relativePath: String): File? {
        val file = File(directory, relativePath)
        val canonicalDirectory = directory.canonicalFile.path + File.separator
        return file.takeIf { it.canonicalFile.path.startsWith(canonicalDirectory) }
    }

    @Synchronized
    internal fun evictArtifact(identity: String, includeStaging: Boolean): Boolean {
        val directory = artifactDirectory(identity)
        val existed = directory.exists()
        directory.deleteRecursively()
        if (includeStaging) {
            resumableStagingPath(identity).deleteRecursively()
        }
        return existed
    }

    private fun resumableStagingPath(identity: String): File {
        val stagingRoot = File(rootDirectory, STAGING_DIRECTORY_NAME)
        stagingRoot.mkdirs()
        return requireContained(
            File(stagingRoot, RESUME_PREFIX + cacheKey(ARTIFACT_KEY_PREFIX + identity)),
        )
    }

    private fun requireContained(file: File): File {
        val canonicalRoot = rootDirectory.canonicalFile.path + File.separator
        if (!file.canonicalFile.path.startsWith(canonicalRoot)) {
            throw UnsafeModelPathException()
        }
        return file
    }

    private fun promote(sourceDirectory: File, destination: File) {
        if (destination.exists()) {
            destination.deleteRecursively()
        }
        destination.parentFile?.mkdirs()

        if (!sourceDirectory.renameTo(destination)) {
            sourceDirectory.copyRecursively(destination, overwrite = true)
            sourceDirectory.deleteRecursively()
        }
    }

    private fun writeReadyMetadata(
        destination: File,
        repoId: String,
        expectedIntegrityHash: String,
        actualIntegrityHash: String,
        revisionSha: String?,
        artifact: CachedArtifactRecord?,
    ) {
        writeMetadata(
            destination,
            CacheMetadata(
                repoId = repoId,
                status = STATUS_READY,
                expectedIntegrityHash = expectedIntegrityHash,
                actualIntegrityHash = actualIntegrityHash,
                revisionSha = revisionSha,
                totalSizeBytes = payloadSizeBytes(destination),
                createdAtEpochMillis = clock.nowEpochMillis(),
                lastAccessedEpochMillis = clock.nowEpochMillis(),
                artifact = artifact,
            ),
        )
    }

    private fun isAvailable(
        repoId: String,
        expectedIntegrityHash: String?,
    ): Boolean {
        val directory = modelDirectory(repoId)
        val metadata = readyMetadata(directory, expectedIntegrityHash) ?: return false
        writeMetadata(
            directory,
            metadata.copy(
                totalSizeBytes = payloadSizeBytes(directory),
                lastAccessedEpochMillis = clock.nowEpochMillis(),
            ),
        )
        return true
    }

    private fun isReadyDirectory(
        directory: File,
        expectedIntegrityHash: String?,
    ): Boolean =
        readyMetadata(directory, expectedIntegrityHash) != null

    private fun readyMetadata(
        directory: File,
        expectedIntegrityHash: String?,
    ): CacheMetadata? {
        if (!directory.isDirectory) {
            return null
        }
        val metadata = readMetadata(directory) ?: return null
        if (metadata.status != STATUS_READY) {
            return null
        }
        if (metadata.repoId.isBlank()) {
            return null
        }
        if (expectedIntegrityHash != null &&
            metadata.expectedIntegrityHash != expectedIntegrityHash
        ) {
            return null
        }
        if (metadata.actualIntegrityHash != metadata.expectedIntegrityHash) {
            return null
        }
        val actualSizeBytes = payloadSizeBytes(directory)
        if (actualSizeBytes <= 0 || actualSizeBytes != metadata.totalSizeBytes) {
            return null
        }
        return metadata
    }

    private fun readyModelDirectories(): List<ReadyDirectory> =
        rootDirectory.listFiles()
            ?.asSequence()
            ?.filter { it.isDirectory && it.name != STAGING_DIRECTORY_NAME }
            ?.mapNotNull { directory ->
                val metadata = readyMetadata(directory, expectedIntegrityHash = null)
                metadata?.let { ReadyDirectory(directory, it) }
            }
            ?.toList()
            ?: emptyList()

    private fun trimToBudget(protectedDirectory: File) {
        var totalSizeBytes = totalSizeBytes()
        if (totalSizeBytes <= cacheBudgetBytes) {
            return
        }

        for (readyDirectory in readyModelDirectories().sortedBy {
            it.metadata.lastAccessedEpochMillis
        }) {
            if (totalSizeBytes <= cacheBudgetBytes) {
                return
            }
            if (readyDirectory.directory == protectedDirectory) {
                continue
            }
            val directorySizeBytes = payloadSizeBytes(readyDirectory.directory)
            readyDirectory.directory.deleteRecursively()
            totalSizeBytes -= directorySizeBytes
        }
    }

    private fun payloadSizeBytes(directory: File): Long {
        if (!directory.exists()) {
            return 0L
        }
        return directory.walkTopDown()
            .filter { it.isFile && it.name != METADATA_FILE_NAME }
            .sumOf { it.length() }
    }

    private fun readMetadata(directory: File): CacheMetadata? {
        val metadataFile = File(directory, METADATA_FILE_NAME)
        if (!metadataFile.isFile) {
            return null
        }
        return try {
            val jsonObject = json.parseToJsonElement(metadataFile.readText()).jsonObject
            CacheMetadata(
                repoId = jsonObject.stringValue("repo_id") ?: return null,
                status = jsonObject.stringValue("status") ?: return null,
                expectedIntegrityHash =
                    jsonObject.stringValue("expected_integrity_hash") ?: return null,
                actualIntegrityHash =
                    jsonObject.stringValue("actual_integrity_hash") ?: return null,
                revisionSha = jsonObject.stringValue("revision_sha"),
                totalSizeBytes = jsonObject.longValue("total_size_bytes") ?: return null,
                createdAtEpochMillis =
                    jsonObject.longValue("created_at_epoch_millis") ?: return null,
                lastAccessedEpochMillis =
                    jsonObject.longValue("last_accessed_epoch_millis") ?: return null,
                artifact = jsonObject["artifact"]?.let { readArtifact(it.jsonObject) },
            )
        } catch (_: RuntimeException) {
            null
        }
    }

    private fun writeMetadata(directory: File, metadata: CacheMetadata) {
        directory.mkdirs()
        val metadataFile = File(directory, METADATA_FILE_NAME)
        val payload = buildJsonObject {
            put("repo_id", metadata.repoId)
            put("status", metadata.status)
            put("expected_integrity_hash", metadata.expectedIntegrityHash)
            put("actual_integrity_hash", metadata.actualIntegrityHash)
            put("revision_sha", metadata.revisionSha)
            put("total_size_bytes", metadata.totalSizeBytes)
            put("created_at_epoch_millis", metadata.createdAtEpochMillis)
            put("last_accessed_epoch_millis", metadata.lastAccessedEpochMillis)
            metadata.artifact?.let { artifact ->
                val files = artifact.files.map { file ->
                    JsonObject(
                        mapOf(
                            "path" to JsonPrimitive(file.path),
                            "size_bytes" to JsonPrimitive(file.sizeBytes),
                            "sha256" to JsonPrimitive(file.sha256),
                        ),
                    )
                }
                put(
                    "artifact",
                    JsonObject(
                        mapOf(
                            "format" to JsonPrimitive(artifact.format),
                            "runtime" to JsonPrimitive(artifact.runtime),
                            "content_digest" to JsonPrimitive(artifact.contentDigest),
                            "files" to JsonArray(files),
                        ),
                    ),
                )
            }
        }
        metadataFile.writeText(payload.toString())
    }

    private fun readArtifact(jsonObject: JsonObject): CachedArtifactRecord? {
        val files = jsonObject["files"]?.jsonArray?.map { element ->
            val file = element.jsonObject
            CachedArtifactFile(
                path = file.stringValue("path") ?: return null,
                sizeBytes = file.longValue("size_bytes") ?: return null,
                sha256 = file.stringValue("sha256") ?: return null,
            )
        }
        return CachedArtifactRecord(
            format = jsonObject.stringValue("format") ?: return null,
            runtime = jsonObject.stringValue("runtime") ?: return null,
            contentDigest = jsonObject.stringValue("content_digest") ?: return null,
            files = files?.takeIf { it.isNotEmpty() } ?: return null,
        )
    }

    private fun JsonObject.stringValue(key: String): String? =
        this[key]?.jsonPrimitive?.contentOrNull

    private fun JsonObject.longValue(key: String): Long? =
        this[key]?.jsonPrimitive?.longOrNull

    private fun cacheKey(repoId: String): String {
        val safePrefix = repoId
            .map { character ->
                if (character.isLetterOrDigit() || character == '.' || character == '-') {
                    character
                } else {
                    '_'
                }
            }
            .joinToString("")
            .trim('_')
            .ifBlank { "model" }
            .take(80)
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(repoId.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
            .take(16)
        return "$safePrefix-$digest"
    }

    private data class ReadyDirectory(
        val directory: File,
        val metadata: CacheMetadata,
    )

    private data class CacheMetadata(
        val repoId: String,
        val status: String,
        val expectedIntegrityHash: String,
        val actualIntegrityHash: String,
        val revisionSha: String?,
        val totalSizeBytes: Long,
        val createdAtEpochMillis: Long,
        val lastAccessedEpochMillis: Long,
        val artifact: CachedArtifactRecord? = null,
    )

    companion object {
        const val DEFAULT_CACHE_BUDGET_BYTES = 2L * 1024L * 1024L * 1024L

        private const val CACHE_DIRECTORY_NAME = "openmed-model-cache"
        private const val METADATA_FILE_NAME = "metadata.json"
        private const val STAGING_DIRECTORY_NAME = ".tmp"
        private const val RESUME_PREFIX = "resume-"
        private const val ARTIFACT_KEY_PREFIX = "artifact:"
        private const val STATUS_READY = "ready"

        private val json = Json {
            ignoreUnknownKeys = true
        }
    }
}

/** Verified file list recorded with a format-aware artifact. */
internal class CachedArtifactRecord(
    val format: String,
    val runtime: String,
    val contentDigest: String,
    val files: List<CachedArtifactFile>,
)

/** One verified artifact file, relative to its artifact directory. */
internal class CachedArtifactFile(
    val path: String,
    val sizeBytes: Long,
    val sha256: String,
)

/** A model file or cache path that would resolve outside its cache root. */
internal class UnsafeModelPathException :
    IllegalArgumentException("model file path escapes cache directory")

fun interface ModelCacheClock {
    fun nowEpochMillis(): Long
}

object SystemModelCacheClock : ModelCacheClock {
    override fun nowEpochMillis(): Long = System.currentTimeMillis()
}
