package com.openmed.openmedkit.cache

import android.content.Context
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.io.Closeable
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Offline-first app-private cache for downloaded OpenMed model artifacts.
 */
class ModelCache(
    private val rootDirectory: File,
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

    private val leaseCounts = mutableMapOf<String, Int>()

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

    /**
     * Delete a cached model unless an open session holds a lease on it.
     *
     * @param repoId Repository identifier of the cached model.
     * @return True when the model existed and was deleted; false when it was
     * missing or is leased (leased models are never deleted underneath a run).
     */
    @Synchronized
    fun evict(repoId: String): Boolean {
        val directory = modelDirectory(repoId)
        if (isLeasedLocked(directory)) {
            return false
        }
        val existed = directory.exists()
        if (existed) {
            directory.deleteRecursively()
        }
        return existed
    }

    /**
     * Lease a verified cached model so it cannot be evicted while in use.
     *
     * The entry's integrity metadata is re-checked on every call. Budget
     * trimming and [evict] skip leased models until every lease is closed.
     *
     * @param entry Catalog entry whose integrity hash must match.
     * @return A lease, or null when no verified copy is cached.
     */
    @Synchronized
    fun acquireLease(entry: ModelCatalogEntry): ModelCacheLease? {
        if (!isAvailable(entry.repoId, entry.reproducibilityHash)) {
            return null
        }
        val directory = modelDirectory(entry.repoId)
        val key = directory.name
        leaseCounts[key] = (leaseCounts[key] ?: 0) + 1
        return ModelCacheLease(directory) { releaseLease(key) }
    }

    /** Whether an open lease currently protects [repoId] from eviction. */
    @Synchronized
    fun isLeased(repoId: String): Boolean = isLeasedLocked(modelDirectory(repoId))

    @Synchronized
    private fun releaseLease(key: String) {
        val remaining = (leaseCounts[key] ?: return) - 1
        if (remaining <= 0) {
            leaseCounts.remove(key)
        } else {
            leaseCounts[key] = remaining
        }
    }

    private fun isLeasedLocked(directory: File): Boolean =
        (leaseCounts[directory.name] ?: 0) > 0

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
        if (destination.exists()) {
            destination.deleteRecursively()
        }
        destination.parentFile?.mkdirs()

        if (!sourceDirectory.renameTo(destination)) {
            sourceDirectory.copyRecursively(destination, overwrite = true)
            sourceDirectory.deleteRecursively()
        }

        val totalSizeBytes = payloadSizeBytes(destination)
        writeMetadata(
            destination,
            CacheMetadata(
                repoId = repoId,
                status = STATUS_READY,
                expectedIntegrityHash = expectedIntegrityHash,
                actualIntegrityHash = actualIntegrityHash,
                revisionSha = revisionSha,
                totalSizeBytes = totalSizeBytes,
                createdAtEpochMillis = clock.nowEpochMillis(),
                lastAccessedEpochMillis = clock.nowEpochMillis(),
            ),
        )
        trimToBudget(protectedRepoId = repoId)
        return destination
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

    private fun trimToBudget(protectedRepoId: String?) {
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
            if (
                readyDirectory.metadata.repoId == protectedRepoId ||
                isLeasedLocked(readyDirectory.directory)
            ) {
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
        }
        metadataFile.writeText(payload.toString())
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
    )

    companion object {
        const val DEFAULT_CACHE_BUDGET_BYTES = 2L * 1024L * 1024L * 1024L

        private const val CACHE_DIRECTORY_NAME = "openmed-model-cache"
        private const val METADATA_FILE_NAME = "metadata.json"
        private const val STAGING_DIRECTORY_NAME = ".tmp"
        private const val STATUS_READY = "ready"

        private val json = Json {
            ignoreUnknownKeys = true
        }
    }
}

/**
 * Open lease on a verified cached model directory.
 *
 * While any lease is open, [ModelCache.evict] and budget trimming leave the
 * directory in place. Closing is idempotent.
 *
 * @property directory The leased model directory.
 */
class ModelCacheLease internal constructor(
    val directory: File,
    private val onClose: () -> Unit,
) : Closeable {
    private val open = AtomicBoolean(true)

    /** Whether this lease is still open. */
    val isOpen: Boolean
        get() = open.get()

    override fun close() {
        if (open.compareAndSet(true, false)) {
            onClose()
        }
    }
}

fun interface ModelCacheClock {
    fun nowEpochMillis(): Long
}

object SystemModelCacheClock : ModelCacheClock {
    override fun nowEpochMillis(): Long = System.currentTimeMillis()
}
