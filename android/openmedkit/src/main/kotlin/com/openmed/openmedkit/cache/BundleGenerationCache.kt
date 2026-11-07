package com.openmed.openmedkit.cache

import com.openmed.openmedkit.bundle.Base64Url
import com.openmed.openmedkit.bundle.Ed25519SignatureVerifier
import com.openmed.openmedkit.bundle.Json
import com.openmed.openmedkit.bundle.PortableBundle
import com.openmed.openmedkit.bundle.PortableBundleContext
import com.openmed.openmedkit.bundle.PortableBundleReason
import com.openmed.openmedkit.bundle.PortableBundleTrustPolicy
import com.openmed.openmedkit.bundle.Rfc8032Ed25519Verifier
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.channels.FileChannel
import java.nio.channels.FileLock
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock

/** Lifecycle states of one bundle generation (`docs/models/bundle-cache.md`). */
enum class BundleGenerationState(val code: String) {
    ABSENT("absent"),
    STAGING("staging"),
    RESUMABLE("resumable"),
    VERIFYING("verifying"),
    READY("ready"),
    ACTIVE("active"),
    SUPERSEDED("superseded"),
    EVICTABLE("evictable"),
    QUARANTINED("quarantined"),
    EVICTED("evicted"),
}

/** Why a generation is protected from eviction, in snapshot order. */
enum class BundleProtection(val code: String) {
    ACTIVE("active"),
    LAST_GREEN("last_green"),
    ROLLBACK("rollback"),
    PINNED("pinned"),
    LEASED("leased"),
}

/** Stable, value-free failure reasons of the generation cache. */
enum class BundleCacheReason(val code: String) {
    BUNDLE_REJECTED("bundle_rejected"),
    LOCK_TIMEOUT("lock_timeout"),
    QUOTA_EXCEEDED("quota_exceeded"),
    STORAGE_PRESSURE("storage_pressure"),
    TRANSFER_INTERRUPTED("transfer_interrupted"),
    CANCELLED("cancelled"),
    RANGE_MISMATCH("range_mismatch"),
    VERIFICATION_FAILED("verification_failed"),
    STORAGE_IO("storage_io"),
    GENERATION_NOT_READY("generation_not_ready"),
    GENERATION_CORRUPT("generation_corrupt"),
    NO_ACTIVE_GENERATION("no_active_generation"),
    ROLLBACK_UNAVAILABLE("rollback_unavailable"),
    LEASE_UNKNOWN("lease_unknown"),
}

/** Why staged bytes were discarded or partly dropped on resume. */
enum class BundleResumeReason(val code: String) {
    SOURCE_CHANGED("source_changed"),
    TRUST_POLICY_CHANGED("trust_policy_changed"),
    CHUNK_SIZE_CHANGED("chunk_size_changed"),
    PARTIAL_DIGEST_MISMATCH("partial_digest_mismatch"),
    RANGE_NOT_HONORED("range_not_honored"),
}

/** One crash-recovery action, in report order. */
enum class BundleRecoveryAction(val code: String) {
    STALE_LOCK_CLEARED("stale_lock_cleared"),
    TRASH_PURGED("trash_purged"),
    QUARANTINE_PURGED("quarantine_purged"),
    STAGING_ORPHANED("staging_orphaned"),
    STAGING_EXPIRED("staging_expired"),
    STAGING_RESUMABLE("staging_resumable"),
    GENERATION_QUARANTINED("generation_quarantined"),
    POINTER_REPAIRED("pointer_repaired"),
    LEASE_EXPIRED("lease_expired"),
}

/** How thoroughly [BundleGenerationCache.open] checks the active generation. */
enum class BundleOpenVerification {
    /** Manifest identity, declared file set, and sizes. */
    METADATA,

    /** Every file is also re-hashed against its declared SHA-256. */
    FULL,
}

/** Error carrying only controlled reason codes. */
class BundleCacheException(
    val reason: BundleCacheReason,
    val bundleReason: PortableBundleReason? = null,
) : Exception(reason.code)

/** A byte range returned by a [BundleRangeSource]. */
class BundleRangeResponse(
    /** Offset of the first byte of [stream]; `0` when the source ignored the range. */
    val offset: Long,
    val stream: InputStream,
) : Closeable {
    override fun close() = stream.close()
}

/**
 * Supplies bundle files by byte range. Implementations perform any network
 * access; the cache itself never does.
 */
interface BundleRangeSource {
    /** Opaque identity of the source (for example a URL plus an ETag). Only its digest is stored. */
    val identity: String

    /** Opens [path] (a declared bundle path) starting at [offset]. */
    fun open(path: String, offset: Long): BundleRangeResponse
}

/** Range source over a local prepared bundle directory. Performs no network access. */
class DirectoryBundleRangeSource(
    private val directory: File,
    override val identity: String = "local-directory",
) : BundleRangeSource {
    override fun open(path: String, offset: Long): BundleRangeResponse {
        val file = File(directory, path)
        val root = directory.canonicalFile.path + File.separator
        if (!file.canonicalFile.path.startsWith(root) || !file.isFile) throw IOException("file_missing")
        val input = FileInputStream(file)
        var skipped = 0L
        while (skipped < offset) {
            val step = input.skip(offset - skipped)
            if (step <= 0) break
            skipped += step
        }
        return BundleRangeResponse(skipped, input)
    }

    override fun toString(): String = "DirectoryBundleRangeSource"
}

/**
 * Tunables of a [BundleGenerationCache].
 *
 * @property quotaBytes byte budget over the ready generations of every bundle.
 * @property chunkBytes resume checkpoint size; each committed chunk records its SHA-256.
 * @property lockTimeoutMillis bounded wait for a bundle lock.
 * @property leaseTtlMillis how long a lease protects its generation once its owner is gone.
 * @property stagingMaxAgeMillis staging older than this since its last checkpoint is removed by recovery.
 * @property availableBytes free-storage probe; defaults to the cache root's usable space.
 * @property ownerId opaque id of this cache instance in lock and lease records.
 */
class BundleCacheOptions(
    val quotaBytes: Long = ModelCache.DEFAULT_CACHE_BUDGET_BYTES,
    val chunkBytes: Int = 4 shl 20,
    val lockTimeoutMillis: Long = 30_000L,
    val leaseTtlMillis: Long = 10L * 60_000L,
    val stagingMaxAgeMillis: Long = 7L * 24L * 3_600_000L,
    val clock: ModelCacheClock = SystemModelCacheClock,
    val availableBytes: ((File) -> Long)? = null,
    val ownerId: String = UUID.randomUUID().toString(),
) {
    init {
        require(quotaBytes >= 0) { "quotaBytes must be non-negative" }
        require(chunkBytes > 0) { "chunkBytes must be positive" }
        require(lockTimeoutMillis >= 0 && leaseTtlMillis > 0 && stagingMaxAgeMillis > 0) {
            "timeouts must be positive"
        }
    }
}

/**
 * One acquisition: a signed manifest and envelope plus the source of its files.
 *
 * @property isCancelled checked after every committed chunk; staged bytes stay resumable.
 */
class BundleAcquireRequest(
    val manifest: ByteArray,
    val envelope: ByteArray,
    val source: BundleRangeSource,
    val trustPolicy: PortableBundleTrustPolicy,
    val context: PortableBundleContext,
    val activate: Boolean = true,
    val confirm: Boolean = true,
    val isCancelled: () -> Boolean = { false },
    val verifier: Ed25519SignatureVerifier = Rfc8032Ed25519Verifier,
)

/** Value-free result of [BundleGenerationCache.acquire]. */
data class BundleAcquireResult(
    val ok: Boolean,
    val reason: BundleCacheReason?,
    val bundleReason: PortableBundleReason?,
    val bundleKey: String?,
    val generation: String?,
    val state: BundleGenerationState?,
    val fromCache: Boolean,
    val resumed: Boolean,
    val resumeReason: BundleResumeReason?,
    val bytesTransferred: Long,
    val verifiedBytes: Long,
    val filesVerified: Int,
    val evicted: List<String>,
)

/** One generation in a [BundleCacheSnapshot]. */
data class BundleGenerationSnapshot(
    val generation: String,
    val state: BundleGenerationState,
    val totalBytes: Long,
    val verifiedBytes: Long,
    val protections: List<BundleProtection>,
)

/** One quarantined generation and its reason code. */
data class BundleQuarantineEntry(val generation: String, val reason: String)

/** Versioned, value-free view of one bundle (`openmed.bundle_cache.snapshot` v1). */
data class BundleCacheSnapshot(
    val bundleKey: String,
    val active: String?,
    val lastGreen: String?,
    val rollback: String?,
    val generations: List<BundleGenerationSnapshot>,
    val quarantined: List<BundleQuarantineEntry>,
    val usedBytes: Long,
) {
    val schema: String get() = SCHEMA
    val schemaVersion: Int get() = 1

    companion object {
        const val SCHEMA = "openmed.bundle_cache.snapshot"
    }
}

/** Deterministic quota decision. */
data class BundleQuotaDecision(
    val quotaBytes: Long,
    val usedBytes: Long,
    val additionalBytes: Long,
    val evicted: List<String>,
    val satisfied: Boolean,
)

/** One recovery action. */
data class BundleRecoveryStep(
    val action: BundleRecoveryAction,
    val bundleKey: String,
    val generation: String?,
)

/** Ordered report of [BundleGenerationCache.recover]. */
data class BundleRecoveryReport(val actions: List<BundleRecoveryStep>)

/**
 * A lease on a verified generation. While held it is never evicted. [toString]
 * omits [payloadDirectory] so the private path does not reach logs.
 */
class BundleLease internal constructor(
    val id: String,
    val bundleKey: String,
    val generation: String,
    val payloadDirectory: File,
    val filesChecked: Int,
    private val cache: BundleGenerationCache,
) : Closeable {
    /** Releases the lease. Safe to call more than once. */
    override fun close() {
        cache.releaseQuietly(this)
    }

    override fun toString(): String =
        "BundleLease(bundleKey=$bundleKey, generation=$generation, filesChecked=$filesChecked)"
}

/** Thrown only by test checkpoints to simulate a process dying at a named point. */
internal class BundleCacheCrash(val checkpoint: String) : Error(checkpoint)

/**
 * Crash-safe, resumable, integrity-checked cache of signed portable bundles.
 *
 * Every bundle has isolated staging, verified generations, an atomic
 * activation pointer, a lock file, and lease records. Acquisition resumes only
 * from verified chunks, verifies every declared file against the signed
 * manifest, and commits with one directory rename. Quota eviction never removes
 * active, last-green, rollback, pinned, or leased generations. The cache never
 * opens a network connection and records only opaque ids, digests, sizes, and
 * reason codes. See `docs/models/bundle-cache.md`.
 */
class BundleGenerationCache(
    private val rootDirectory: File,
    private val options: BundleCacheOptions = BundleCacheOptions(),
) {
    /** Test seam: called at each named crash checkpoint. */
    internal var checkpointHook: ((String) -> Unit)? = null

    private val heldLeases = ConcurrentHashMap.newKeySet<String>()

    init {
        rootDirectory.mkdirs()
    }

    // region Public API

    /** Verifies, stages, commits, and optionally activates one signed bundle. */
    fun acquire(request: BundleAcquireRequest): BundleAcquireResult {
        val verification = PortableBundle.verify(
            request.manifest, request.envelope, request.trustPolicy, request.context, null, request.verifier,
        )
        val contentId = verification.contentId
        if (!verification.ok || contentId == null) {
            return failure(BundleCacheReason.BUNDLE_REJECTED, verification.reason, null, null)
        }
        val header = ManifestHeader.parse(request.manifest)
            ?: return failure(BundleCacheReason.BUNDLE_REJECTED, PortableBundleReason.MANIFEST_MALFORMED, null, null)
        val key = bundleKey(header.bundleId)
        val generation = generationId(contentId)
        val lock = lockBundle(key, options.lockTimeoutMillis)
            ?: return failure(BundleCacheReason.LOCK_TIMEOUT, null, key, generation)
        return lock.guarded {
            try {
                acquireLocked(request, header, key, generation, contentId)
            } catch (error: BundleCacheException) {
                failure(error.reason, error.bundleReason, key, generation)
            } catch (error: IOException) {
                failure(BundleCacheReason.STORAGE_IO, null, key, generation)
            }
        }
    }

    /** Activates a ready generation; with [confirm] it also becomes last green. */
    fun activate(bundleId: String, generation: String, confirm: Boolean = true) {
        locked(bundleKey(bundleId)) { key -> activateLocked(key, generation, confirm) }
    }

    /** Marks the active generation as last green after a probation activation. */
    fun confirm(bundleId: String) {
        locked(bundleKey(bundleId)) { key ->
            val pointer = readPointer(key)
            val active = pointer.active ?: throw BundleCacheException(BundleCacheReason.NO_ACTIVE_GENERATION)
            writePointer(key, pointer.copy(lastGreen = active))
        }
    }

    /** Reactivates the retained rollback generation without any transfer. */
    fun rollback(bundleId: String, quarantineAbandoned: Boolean = false) {
        locked(bundleKey(bundleId)) { key ->
            val pointer = readPointer(key)
            val target = pointer.rollback?.takeIf { metadataCheck(key, it) == null }
                ?: throw BundleCacheException(BundleCacheReason.ROLLBACK_UNAVAILABLE)
            val abandoned = pointer.active
            touchRecord(key, target, activated = true)
            val lastGreen = if (pointer.lastGreen == abandoned) target else pointer.lastGreen
            writePointer(key, Pointer(active = target, lastGreen = lastGreen, rollback = null))
            if (quarantineAbandoned && abandoned != null && abandoned != target) {
                quarantine(key, generationDirectory(key, abandoned), abandoned, QUARANTINE_ROLLED_BACK)
            }
        }
    }

    /** Pins a ready generation so eviction never removes it. */
    fun pin(bundleId: String, generation: String) = setPinned(bundleId, generation, true)

    /** Removes a pin. */
    fun unpin(bundleId: String, generation: String) = setPinned(bundleId, generation, false)

    /** Leases a ready generation (the active one by default) after a metadata check. */
    fun lease(bundleId: String, generation: String? = null): BundleLease =
        locked(bundleKey(bundleId)) { key ->
            val target = generation ?: readPointer(key).active
                ?: throw BundleCacheException(BundleCacheReason.NO_ACTIVE_GENERATION)
            if (readGenerationRecord(key, target) == null) {
                throw BundleCacheException(BundleCacheReason.GENERATION_NOT_READY)
            }
            createLease(key, target, BundleOpenVerification.METADATA)
        }

    /**
     * Opens the active generation for loading and returns a lease on it. A
     * mismatch quarantines the generation, falls the pointer back, and fails
     * with [BundleCacheReason.GENERATION_CORRUPT].
     */
    fun open(
        bundleId: String,
        verification: BundleOpenVerification = BundleOpenVerification.METADATA,
    ): BundleLease =
        locked(bundleKey(bundleId)) { key ->
            val active = readPointer(key).active
                ?: throw BundleCacheException(BundleCacheReason.NO_ACTIVE_GENERATION)
            createLease(key, active, verification)
        }

    /** Releases a lease. */
    fun release(lease: BundleLease) {
        locked(lease.bundleKey) { key ->
            val file = leaseFile(key, lease.generation, lease.id)
            val known = heldLeases.remove(lease.id)
            if (!file.delete() && !known) throw BundleCacheException(BundleCacheReason.LEASE_UNKNOWN)
        }
    }

    internal fun releaseQuietly(lease: BundleLease) {
        try {
            release(lease)
        } catch (_: BundleCacheException) {
        }
    }

    /** Evicts least recently used unprotected generations until [additionalBytes] fit. */
    fun evict(additionalBytes: Long = 0L): BundleQuotaDecision {
        require(additionalBytes >= 0) { "additionalBytes must be non-negative" }
        return applyQuota(additionalBytes, exclude = null, heldKey = null)
    }

    /** Idempotent, bounded crash recovery over every bundle; call at startup. */
    fun recover(): BundleRecoveryReport {
        val actions = mutableListOf<BundleRecoveryStep>()
        for (key in bundleKeys()) {
            val lock = lockBundle(key, options.lockTimeoutMillis) ?: continue
            lock.guarded { recoverLocked(key, lock.stale, actions) }
        }
        return BundleRecoveryReport(actions)
    }

    /** Value-free snapshot of one bundle. */
    fun snapshot(bundleId: String): BundleCacheSnapshot =
        locked(bundleKey(bundleId)) { key -> snapshotLocked(key) }

    // endregion

    // region Acquisition

    private fun acquireLocked(
        request: BundleAcquireRequest,
        header: ManifestHeader,
        key: String,
        generation: String,
        contentId: String,
    ): BundleAcquireResult {
        val files = header.files
        val totalBytes = files.sumOf { it.size }
        if (metadataCheck(key, generation) == null) {
            touchRecord(key, generation, activated = null)
            if (request.activate) activateLocked(key, generation, request.confirm)
            return success(key, generation, totalBytes, files.size, fromCache = true)
        }
        if (readGenerationRecord(key, generation) != null) {
            // A ready record whose files changed is unusable: quarantine it first.
            quarantineAndRepair(key, generation, QUARANTINE_PAYLOAD_MISMATCH)
        }

        stagingRoot(key).listFiles()?.sortedBy { it.name }?.forEach { other ->
            if (other.name != generation) discard(key, other)
        }
        quarantineDirectory(key, generation).takeIf { it.exists() }?.let { discard(key, it) }

        val stage = stagingDirectory(key, generation)
        val payload = File(stage, PAYLOAD)
        val sourceDigest = sha256Text(request.source.identity.toByteArray(StandardCharsets.UTF_8))
        val trust = trustDigest(request.trustPolicy)
        val chunk = options.chunkBytes
        var resumeReason: BundleResumeReason? = null
        var chunks: List<MutableList<String>> = files.map { mutableListOf() }
        val record = readStagingRecord(stage)
        var fresh = true
        if (record != null && record.contentId == contentId && record.files.size == files.size) {
            resumeReason = when {
                record.sourceDigest != sourceDigest -> BundleResumeReason.SOURCE_CHANGED
                record.trustDigest != trust -> BundleResumeReason.TRUST_POLICY_CHANGED
                record.chunkBytes != chunk -> BundleResumeReason.CHUNK_SIZE_CHANGED
                else -> null
            }
            if (resumeReason == null) {
                fresh = false
                chunks = files.indices.map { index ->
                    val kept = keptChunks(File(payload, files[index].path), files[index].size, record.files[index])
                    if (kept.mismatch) {
                        resumeReason = BundleResumeReason.PARTIAL_DIGEST_MISMATCH
                    }
                    kept.digests
                }
            }
        }
        if (fresh && stage.exists()) discard(key, stage)
        val keptBytes = files.indices.sumOf { verifiedBytes(chunks[it].size, files[it].size) }
        val resumed = keptBytes > 0

        val needed = totalBytes - keptBytes
        val available = options.availableBytes?.invoke(rootDirectory) ?: rootDirectory.usableSpace
        if (available < needed) {
            if (!fresh) writeStagingRecord(stage, stagingRecord(contentId, sourceDigest, trust, chunks, record))
            throw BundleCacheException(BundleCacheReason.STORAGE_PRESSURE)
        }
        val evicted = mutableListOf<String>()
        val plan = planQuota(totalBytes, exclude = null)
        if (!plan.satisfied) {
            if (!fresh) writeStagingRecord(stage, stagingRecord(contentId, sourceDigest, trust, chunks, record))
            throw BundleCacheException(BundleCacheReason.QUOTA_EXCEEDED)
        }
        evicted += applyPlan(plan.candidates, heldKey = key)

        if (fresh) {
            payload.mkdirs()
            writeFileAtomic(File(payload, PortableBundle.MANIFEST_FILENAME), request.manifest)
            writeFileAtomic(File(payload, PortableBundle.ENVELOPE_FILENAME), request.envelope)
        }
        val created = record?.takeIf { !fresh }?.createdAt ?: now()
        fun checkpointRecord(phase: String) = writeStagingRecord(
            stage,
            StagingRecord(contentId, sourceDigest, trust, chunk, phase, created, now(), chunks.map { it.toList() }),
        )
        checkpointRecord(PHASE_STAGING)
        if (fresh) checkpoint(CHECKPOINT_STAGING_RECORD_WRITTEN)

        var transferred = 0L
        for ((index, file) in files.withIndex()) {
            if (request.isCancelled()) throw BundleCacheException(BundleCacheReason.CANCELLED)
            val target = File(payload, file.path)
            var have = verifiedBytes(chunks[index].size, file.size)
            if (have == file.size) continue
            target.parentFile?.mkdirs()
            val response = try {
                request.source.open(file.path, have)
            } catch (error: IOException) {
                throw BundleCacheException(BundleCacheReason.TRANSFER_INTERRUPTED)
            } catch (error: RuntimeException) {
                throw BundleCacheException(BundleCacheReason.TRANSFER_INTERRUPTED)
            }
            response.use {
                if (response.offset != have) {
                    if (response.offset != 0L) throw BundleCacheException(BundleCacheReason.RANGE_MISMATCH)
                    chunks[index].clear()
                    have = 0L
                    if (resumeReason == null) resumeReason = BundleResumeReason.RANGE_NOT_HONORED
                }
                RandomAccessFile(target, "rw").use { output ->
                    output.setLength(have)
                    output.seek(have)
                    val buffer = ByteArray(chunk)
                    while (have < file.size) {
                        val want = minOf(chunk.toLong(), file.size - have).toInt()
                        val read = readFully(response.stream, buffer, want)
                        transferred += read
                        if (read < want) throw BundleCacheException(BundleCacheReason.TRANSFER_INTERRUPTED)
                        output.write(buffer, 0, want)
                        chunks[index].add(sha256Text(buffer, want))
                        have += want
                        checkpointRecord(PHASE_STAGING)
                        checkpoint(CHECKPOINT_CHUNK_COMMITTED)
                        if (request.isCancelled()) throw BundleCacheException(BundleCacheReason.CANCELLED)
                    }
                }
                val extra = try {
                    response.stream.read()
                } catch (error: IOException) {
                    -1
                }
                if (extra != -1) {
                    quarantine(key, stage, generation, PortableBundleReason.FILE_SIZE_MISMATCH.code)
                    throw BundleCacheException(
                        BundleCacheReason.VERIFICATION_FAILED, PortableBundleReason.FILE_SIZE_MISMATCH,
                    )
                }
            }
        }

        checkpointRecord(PHASE_VERIFYING)
        checkpoint(CHECKPOINT_VERIFYING_RECORD_WRITTEN)
        verifyPayload(payload, files, full = true)?.let { reason ->
            quarantine(key, stage, generation, reason.code)
            throw BundleCacheException(BundleCacheReason.VERIFICATION_FAILED, reason)
        }

        val stamp = now()
        writeGenerationRecord(
            File(stage, GENERATION_RECORD),
            GenerationRecord(contentId, header.sequence, totalBytes, files.size, false, false, stamp, stamp),
        )
        val destination = generationDirectory(key, generation)
        destination.parentFile?.mkdirs()
        if (destination.exists()) discard(key, destination)
        move(stage, destination)
        checkpoint(CHECKPOINT_GENERATION_COMMITTED)
        File(destination, STAGING_RECORD).delete()

        if (request.activate) activateLocked(key, generation, request.confirm)
        evicted += applyQuota(0L, exclude = generation, heldKey = key).evicted
        return success(
            key, generation, totalBytes, files.size, fromCache = false, resumed = resumed,
            resumeReason = resumeReason, transferred = transferred, evicted = evicted,
        )
    }

    private class KeptChunks(val digests: MutableList<String>, val mismatch: Boolean)

    private fun keptChunks(file: File, size: Long, recorded: List<String>): KeptChunks {
        val kept = mutableListOf<String>()
        if (!file.isFile) return KeptChunks(kept, false)
        var mismatch = false
        RandomAccessFile(file, "rw").use { input ->
            val length = input.length()
            val buffer = ByteArray(options.chunkBytes)
            for ((index, digest) in recorded.withIndex()) {
                val start = index.toLong() * options.chunkBytes
                val want = minOf(options.chunkBytes.toLong(), size - start).toInt()
                if (want <= 0 || start + want > length) break
                input.seek(start)
                input.readFully(buffer, 0, want)
                if (sha256Text(buffer, want) != digest) {
                    mismatch = true
                    break
                }
                kept += digest
            }
            input.setLength(verifiedBytes(kept.size, size))
        }
        return KeptChunks(kept, mismatch)
    }

    private fun verifiedBytes(chunkCount: Int, size: Long): Long =
        minOf(chunkCount.toLong() * options.chunkBytes, size)

    private fun readFully(input: InputStream, buffer: ByteArray, want: Int): Int {
        var total = 0
        try {
            while (total < want) {
                val read = input.read(buffer, total, want - total)
                if (read < 0) break
                total += read
            }
        } catch (error: IOException) {
            return total
        }
        return total
    }

    /** Returns the first verifier reason the payload fails, or `null`. */
    private fun verifyPayload(payload: File, files: List<DeclaredFile>, full: Boolean): PortableBundleReason? {
        for (file in files) {
            val target = File(payload, file.path)
            if (!target.isFile) return PortableBundleReason.FILE_MISSING
            if (target.length() != file.size) return PortableBundleReason.FILE_SIZE_MISMATCH
            if (full && sha256File(target) != file.sha256) return PortableBundleReason.FILE_DIGEST_MISMATCH
        }
        if (full) {
            val declared = files.map { it.path }.toSet() +
                setOf(PortableBundle.MANIFEST_FILENAME, PortableBundle.ENVELOPE_FILENAME)
            val undeclared = payload.walkTopDown().any { entry ->
                entry.isFile && entry.relativeTo(payload).invariantSeparatorsPath !in declared
            }
            if (undeclared) return PortableBundleReason.FILE_UNDECLARED
        }
        return null
    }

    // endregion

    // region Activation, leases, pins

    private fun activateLocked(key: String, generation: String, confirm: Boolean) {
        if (readGenerationRecord(key, generation) == null) {
            throw BundleCacheException(BundleCacheReason.GENERATION_NOT_READY)
        }
        metadataCheck(key, generation)?.let { reason ->
            quarantineAndRepair(key, generation, reason.code)
            throw BundleCacheException(BundleCacheReason.GENERATION_CORRUPT, reason)
        }
        val pointer = readPointer(key)
        val next = if (pointer.active == generation) {
            pointer.copy(lastGreen = if (confirm) generation else pointer.lastGreen)
        } else {
            val rollback = (pointer.active ?: pointer.rollback)?.takeIf { it != generation }
            Pointer(generation, if (confirm) generation else pointer.lastGreen, rollback)
        }
        touchRecord(key, generation, activated = true)
        writePointer(key, next)
        checkpoint(CHECKPOINT_POINTER_COMMITTED)
    }

    private fun setPinned(bundleId: String, generation: String, pinned: Boolean) {
        locked(bundleKey(bundleId)) { key ->
            val record = readGenerationRecord(key, generation)
                ?: throw BundleCacheException(BundleCacheReason.GENERATION_NOT_READY)
            writeGenerationRecord(generationRecordFile(key, generation), record.copy(pinned = pinned))
        }
    }

    private fun createLease(key: String, generation: String, verification: BundleOpenVerification): BundleLease {
        val record = readGenerationRecord(key, generation)
            ?: throw BundleCacheException(BundleCacheReason.GENERATION_NOT_READY)
        val payload = File(generationDirectory(key, generation), PAYLOAD)
        val reason = metadataCheck(key, generation)
            ?: if (verification == BundleOpenVerification.FULL) {
                verifyPayload(payload, declaredFiles(key, generation) ?: emptyList(), full = true)
            } else {
                null
            }
        if (reason != null) {
            quarantineAndRepair(key, generation, reason.code)
            throw BundleCacheException(BundleCacheReason.GENERATION_CORRUPT, reason)
        }
        val id = UUID.randomUUID().toString()
        val file = leaseFile(key, generation, id)
        file.parentFile?.mkdirs()
        writeFileAtomic(
            file,
            Json.encode(
                obj(
                    "schema" to Json.Str(LEASE_SCHEMA),
                    "schema_version" to Json.Num(1),
                    "lease" to Json.Str(id),
                    "generation" to Json.Str(generation),
                    "owner" to Json.Str(options.ownerId),
                    "expires_at_ms" to Json.Num(now() + options.leaseTtlMillis),
                ),
            ),
        )
        heldLeases += id
        touchRecord(key, generation, activated = null)
        return BundleLease(id, key, generation, payload, record.fileCount, this)
    }

    // endregion

    // region Quota and eviction

    private class Candidate(val key: String, val generation: String, val lastAccessed: Long, val size: Long)

    private class Plan(val candidates: List<Candidate>, val usedAfter: Long, val satisfied: Boolean)

    private fun planQuota(additionalBytes: Long, exclude: String?): Plan {
        var used = 0L
        val candidates = mutableListOf<Candidate>()
        for (key in bundleKeys()) {
            val pointer = readPointer(key)
            for (generation in readyGenerations(key)) {
                val record = readGenerationRecord(key, generation) ?: continue
                used += record.totalBytes
                if (generation != exclude && protections(key, generation, pointer, record).isEmpty()) {
                    candidates += Candidate(key, generation, record.lastAccessed, record.totalBytes)
                }
            }
        }
        candidates.sortWith(compareBy<Candidate> { it.lastAccessed }.thenBy { it.generation })
        val chosen = mutableListOf<Candidate>()
        for (candidate in candidates) {
            if (used + additionalBytes <= options.quotaBytes) break
            chosen += candidate
            used -= candidate.size
        }
        return Plan(chosen, used, used + additionalBytes <= options.quotaBytes)
    }

    private fun applyQuota(additionalBytes: Long, exclude: String?, heldKey: String?): BundleQuotaDecision {
        val plan = planQuota(additionalBytes, exclude)
        val evicted = applyPlan(plan.candidates, heldKey)
        val used = usedBytes()
        return BundleQuotaDecision(
            options.quotaBytes, used, additionalBytes, evicted, used + additionalBytes <= options.quotaBytes,
        )
    }

    private fun applyPlan(candidates: List<Candidate>, heldKey: String?): List<String> {
        val evicted = mutableListOf<String>()
        for (candidate in candidates) {
            val evict = {
                val record = readGenerationRecord(candidate.key, candidate.generation)
                val pointer = readPointer(candidate.key)
                if (record != null && protections(candidate.key, candidate.generation, pointer, record).isEmpty()) {
                    val trash = trashEntry(candidate.key, candidate.generation)
                    move(generationDirectory(candidate.key, candidate.generation), trash)
                    checkpoint(CHECKPOINT_EVICTION_COMMITTED)
                    trash.deleteRecursively()
                    File(leasesRoot(candidate.key), candidate.generation).deleteRecursively()
                    evicted += candidate.generation
                }
            }
            if (candidate.key == heldKey) {
                evict()
            } else {
                // Another bundle: wait only when no bundle lock is held, so two
                // acquisitions evicting each other's generations cannot deadlock.
                val wait = if (heldKey == null) options.lockTimeoutMillis else 0L
                lockBundle(candidate.key, wait)?.guarded(evict)
            }
        }
        return evicted
    }

    private fun usedBytes(): Long =
        bundleKeys().sumOf { key -> readyGenerations(key).sumOf { readGenerationRecord(key, it)?.totalBytes ?: 0L } }

    private fun protections(
        key: String,
        generation: String,
        pointer: Pointer,
        record: GenerationRecord,
    ): List<BundleProtection> {
        val result = mutableListOf<BundleProtection>()
        if (pointer.active == generation) result += BundleProtection.ACTIVE
        if (pointer.lastGreen == generation) result += BundleProtection.LAST_GREEN
        if (pointer.rollback == generation) result += BundleProtection.ROLLBACK
        if (record.pinned) result += BundleProtection.PINNED
        if (isLeased(key, generation)) result += BundleProtection.LEASED
        return result
    }

    private fun isLeased(key: String, generation: String): Boolean {
        val now = now()
        return File(leasesRoot(key), generation).listFiles()?.any { file ->
            val lease = readLease(file) ?: return@any false
            (lease.owner == options.ownerId && lease.id in heldLeases) || lease.expiresAt > now
        } ?: false
    }

    private fun state(key: String, generation: String, pointer: Pointer, record: GenerationRecord): BundleGenerationState {
        val protections = protections(key, generation, pointer, record)
        return when {
            BundleProtection.ACTIVE in protections -> BundleGenerationState.ACTIVE
            protections.isEmpty() -> BundleGenerationState.EVICTABLE
            record.activated -> BundleGenerationState.SUPERSEDED
            else -> BundleGenerationState.READY
        }
    }

    // endregion

    // region Recovery and snapshots

    private fun recoverLocked(key: String, stale: Boolean, actions: MutableList<BundleRecoveryStep>) {
        fun report(action: BundleRecoveryAction, generation: String?) {
            actions += BundleRecoveryStep(action, key, generation)
        }
        if (stale) report(BundleRecoveryAction.STALE_LOCK_CLEARED, null)
        File(bundleDirectory(key), POINTER + TMP_SUFFIX).delete()
        sortedChildren(trashRoot(key)).forEach { entry ->
            entry.deleteRecursively()
            report(BundleRecoveryAction.TRASH_PURGED, entry.name.substringBefore('.').takeIf { GENERATION.matches(it) })
        }
        sortedChildren(quarantineRoot(key)).forEach { entry ->
            discard(key, entry)
            report(BundleRecoveryAction.QUARANTINE_PURGED, entry.name.takeIf { GENERATION.matches(it) })
        }
        val ready = readyGenerations(key).toSet()
        sortedChildren(stagingRoot(key)).forEach { entry ->
            val generation = entry.name.takeIf { GENERATION.matches(it) }
            val record = readStagingRecord(entry)
            when {
                generation == null || record == null -> {
                    discard(key, entry)
                    report(BundleRecoveryAction.STAGING_ORPHANED, generation)
                }
                generation in ready -> discard(key, entry)
                now() - record.updatedAt > options.stagingMaxAgeMillis -> {
                    discard(key, entry)
                    report(BundleRecoveryAction.STAGING_EXPIRED, generation)
                }
                else -> {
                    if (record.phase != PHASE_STAGING) writeStagingRecord(entry, record.copy(phase = PHASE_STAGING))
                    report(BundleRecoveryAction.STAGING_RESUMABLE, generation)
                }
            }
        }
        sortedChildren(generationsRoot(key)).forEach { entry ->
            File(entry, STAGING_RECORD).delete()
            if (!GENERATION.matches(entry.name)) {
                discard(key, entry)
            } else if (metadataCheck(key, entry.name) != null) {
                quarantine(key, entry, entry.name, QUARANTINE_PAYLOAD_MISMATCH)
                report(BundleRecoveryAction.GENERATION_QUARANTINED, entry.name)
            }
        }
        if (repairPointer(key)) report(BundleRecoveryAction.POINTER_REPAIRED, null)
        val now = now()
        sortedChildren(leasesRoot(key)).forEach { directory ->
            sortedChildren(directory).forEach { file ->
                val lease = readLease(file)
                val expired = lease == null ||
                    (!(lease.owner == options.ownerId && lease.id in heldLeases) && lease.expiresAt <= now)
                if (expired) {
                    file.delete()
                    report(BundleRecoveryAction.LEASE_EXPIRED, directory.name.takeIf { GENERATION.matches(it) })
                }
            }
            if (directory.listFiles()?.isEmpty() == true) directory.delete()
        }
    }

    private fun snapshotLocked(key: String): BundleCacheSnapshot {
        val pointer = readPointer(key)
        val generations = mutableListOf<BundleGenerationSnapshot>()
        var used = 0L
        for (generation in readyGenerations(key)) {
            val record = readGenerationRecord(key, generation) ?: continue
            used += record.totalBytes
            generations += BundleGenerationSnapshot(
                generation,
                state(key, generation, pointer, record),
                record.totalBytes,
                record.totalBytes,
                protections(key, generation, pointer, record),
            )
        }
        for (entry in sortedChildren(stagingRoot(key))) {
            if (!GENERATION.matches(entry.name) || generations.any { it.generation == entry.name }) continue
            val record = readStagingRecord(entry) ?: continue
            val files = readManifestFiles(File(entry, PAYLOAD)) ?: continue
            val verified = files.indices.sumOf { index ->
                verifiedBytes(record.files.getOrNull(index)?.size ?: 0, files[index].size)
            }
            generations += BundleGenerationSnapshot(
                entry.name, BundleGenerationState.RESUMABLE, files.sumOf { it.size }, verified, emptyList(),
            )
        }
        val quarantined = sortedChildren(quarantineRoot(key)).filter { GENERATION.matches(it.name) }.map { entry ->
            val reason = (readJson(File(entry, QUARANTINE_RECORD)) as? Json.Obj)?.get("reason") as? Json.Str
            BundleQuarantineEntry(entry.name, reason?.value ?: "unknown")
        }
        return BundleCacheSnapshot(
            key, pointer.active, pointer.lastGreen, pointer.rollback,
            generations.sortedBy { it.generation }, quarantined, used,
        )
    }

    private fun quarantineAndRepair(key: String, generation: String, reason: String) {
        quarantine(key, generationDirectory(key, generation), generation, reason)
        repairPointer(key)
    }

    /** Falls an invalid active generation back to last green, then rollback. */
    private fun repairPointer(key: String): Boolean {
        val pointer = readPointer(key)
        val valid = { generation: String? -> generation != null && readGenerationRecord(key, generation) != null }
        val lastGreen = pointer.lastGreen.takeIf(valid)
        var rollback = pointer.rollback.takeIf(valid)
        val active = if (valid(pointer.active)) pointer.active else lastGreen ?: rollback
        if (rollback == active) rollback = null
        val next = Pointer(active, lastGreen, rollback)
        if (next == pointer) return false
        writePointer(key, next)
        return true
    }

    // endregion

    // region Records

    private data class Pointer(val active: String?, val lastGreen: String?, val rollback: String?)

    private data class GenerationRecord(
        val contentId: String,
        val sequence: Long,
        val totalBytes: Long,
        val fileCount: Int,
        val activated: Boolean,
        val pinned: Boolean,
        val verifiedAt: Long,
        val lastAccessed: Long,
    )

    private data class StagingRecord(
        val contentId: String,
        val sourceDigest: String,
        val trustDigest: String,
        val chunkBytes: Int,
        val phase: String,
        val createdAt: Long,
        val updatedAt: Long,
        val files: List<List<String>>,
    )

    private class LeaseRecord(val id: String, val owner: String, val expiresAt: Long)

    private class DeclaredFile(val path: String, val size: Long, val sha256: String)

    private class ManifestHeader(val bundleId: String, val sequence: Long, val files: List<DeclaredFile>) {
        companion object {
            fun parse(manifest: ByteArray): ManifestHeader? = try {
                val root = Json.parse(manifest) as Json.Obj
                val bundle = root["bundle"] as Json.Obj
                ManifestHeader(
                    (bundle["id"] as Json.Str).value,
                    (bundle["sequence"] as Json.Num).value,
                    (root["files"] as Json.Arr).items.map { item ->
                        val entry = item as Json.Obj
                        DeclaredFile(
                            (entry["path"] as Json.Str).value,
                            (entry["size_bytes"] as Json.Num).value,
                            (entry["sha256"] as Json.Str).value,
                        )
                    },
                )
            } catch (error: RuntimeException) {
                null
            } catch (error: Json.Invalid) {
                null
            }
        }
    }

    private fun readPointer(key: String): Pointer {
        val value = readJson(File(bundleDirectory(key), POINTER)) as? Json.Obj ?: return Pointer(null, null, null)
        fun field(name: String) = (value[name] as? Json.Str)?.value?.takeIf { GENERATION.matches(it) }
        return Pointer(field("active"), field("last_green"), field("rollback"))
    }

    private fun writePointer(key: String, pointer: Pointer) {
        fun ref(value: String?): Json = value?.let { Json.Str(it) } ?: Json.Null
        writeFileAtomic(
            File(bundleDirectory(key), POINTER),
            Json.encode(
                obj(
                    "schema" to Json.Str(POINTER_SCHEMA),
                    "schema_version" to Json.Num(1),
                    "active" to ref(pointer.active),
                    "last_green" to ref(pointer.lastGreen),
                    "rollback" to ref(pointer.rollback),
                    "updated_at_ms" to Json.Num(now()),
                ),
            ),
        )
    }

    private fun readGenerationRecord(key: String, generation: String): GenerationRecord? {
        if (!GENERATION.matches(generation)) return null
        val value = readJson(generationRecordFile(key, generation)) as? Json.Obj ?: return null
        return try {
            if (value["schema"] != Json.Str(GENERATION_SCHEMA) || value["state"] != Json.Str("ready")) return null
            GenerationRecord(
                (value["content_id"] as Json.Str).value,
                (value["sequence"] as Json.Num).value,
                (value["total_bytes"] as Json.Num).value,
                (value["file_count"] as Json.Num).value.toInt(),
                (value["activated"] as Json.Bool).value,
                (value["pinned"] as Json.Bool).value,
                (value["verified_at_ms"] as Json.Num).value,
                (value["last_accessed_ms"] as Json.Num).value,
            ).takeIf { generationId(it.contentId) == generation }
        } catch (error: RuntimeException) {
            null
        }
    }

    private fun writeGenerationRecord(file: File, record: GenerationRecord) {
        writeFileAtomic(
            file,
            Json.encode(
                obj(
                    "schema" to Json.Str(GENERATION_SCHEMA),
                    "schema_version" to Json.Num(1),
                    "state" to Json.Str("ready"),
                    "generation" to Json.Str(generationId(record.contentId)),
                    "content_id" to Json.Str(record.contentId),
                    "sequence" to Json.Num(record.sequence),
                    "total_bytes" to Json.Num(record.totalBytes),
                    "file_count" to Json.Num(record.fileCount.toLong()),
                    "activated" to Json.Bool(record.activated),
                    "pinned" to Json.Bool(record.pinned),
                    "verified_at_ms" to Json.Num(record.verifiedAt),
                    "last_accessed_ms" to Json.Num(record.lastAccessed),
                ),
            ),
        )
    }

    private fun touchRecord(key: String, generation: String, activated: Boolean?) {
        val record = readGenerationRecord(key, generation) ?: return
        writeGenerationRecord(
            generationRecordFile(key, generation),
            record.copy(activated = activated ?: record.activated, lastAccessed = now()),
        )
    }

    private fun stagingRecord(
        contentId: String,
        sourceDigest: String,
        trust: String,
        chunks: List<List<String>>,
        previous: StagingRecord?,
    ) = StagingRecord(
        contentId, sourceDigest, trust, options.chunkBytes, PHASE_STAGING,
        previous?.createdAt ?: now(), now(), chunks.map { it.toList() },
    )

    private fun readStagingRecord(stage: File): StagingRecord? {
        val value = readJson(File(stage, STAGING_RECORD)) as? Json.Obj ?: return null
        return try {
            if (value["schema"] != Json.Str(STAGING_SCHEMA)) return null
            StagingRecord(
                (value["content_id"] as Json.Str).value,
                (value["source_digest"] as Json.Str).value,
                (value["trust_digest"] as Json.Str).value,
                (value["chunk_bytes"] as Json.Num).value.toInt(),
                (value["phase"] as Json.Str).value,
                (value["created_at_ms"] as Json.Num).value,
                (value["updated_at_ms"] as Json.Num).value,
                (value["files"] as Json.Arr).items.map { file ->
                    (file as Json.Arr).items.map { (it as Json.Str).value }
                },
            )
        } catch (error: RuntimeException) {
            null
        }
    }

    private fun writeStagingRecord(stage: File, record: StagingRecord) {
        stage.mkdirs()
        writeFileAtomic(
            File(stage, STAGING_RECORD),
            Json.encode(
                obj(
                    "schema" to Json.Str(STAGING_SCHEMA),
                    "schema_version" to Json.Num(1),
                    "content_id" to Json.Str(record.contentId),
                    "source_digest" to Json.Str(record.sourceDigest),
                    "trust_digest" to Json.Str(record.trustDigest),
                    "chunk_bytes" to Json.Num(record.chunkBytes.toLong()),
                    "phase" to Json.Str(record.phase),
                    "created_at_ms" to Json.Num(record.createdAt),
                    "updated_at_ms" to Json.Num(record.updatedAt),
                    "files" to Json.Arr(record.files.map { file -> Json.Arr(file.map { Json.Str(it) }) }),
                ),
            ),
        )
    }

    private fun readLease(file: File): LeaseRecord? {
        val value = readJson(file) as? Json.Obj ?: return null
        return try {
            LeaseRecord(
                (value["lease"] as Json.Str).value,
                (value["owner"] as Json.Str).value,
                (value["expires_at_ms"] as Json.Num).value,
            )
        } catch (error: RuntimeException) {
            null
        }
    }

    private fun readManifestFiles(payload: File): List<DeclaredFile>? {
        val manifest = File(payload, PortableBundle.MANIFEST_FILENAME)
        if (!manifest.isFile) return null
        return ManifestHeader.parse(manifest.readBytes())?.files
    }

    private fun declaredFiles(key: String, generation: String): List<DeclaredFile>? =
        readManifestFiles(File(generationDirectory(key, generation), PAYLOAD))

    /** Cheap check: the record, the manifest identity, the declared files, and their sizes. */
    private fun metadataCheck(key: String, generation: String): PortableBundleReason? {
        val record = readGenerationRecord(key, generation) ?: return PortableBundleReason.FILE_MISSING
        val payload = File(generationDirectory(key, generation), PAYLOAD)
        val manifest = File(payload, PortableBundle.MANIFEST_FILENAME)
        if (!manifest.isFile) return PortableBundleReason.FILE_MISSING
        val bytes = manifest.readBytes()
        if (PortableBundle.contentId(bytes) != record.contentId) return PortableBundleReason.FILE_DIGEST_MISMATCH
        val files = ManifestHeader.parse(bytes)?.files ?: return PortableBundleReason.MANIFEST_MALFORMED
        if (files.size != record.fileCount || files.sumOf { it.size } != record.totalBytes) {
            return PortableBundleReason.FILE_SIZE_MISMATCH
        }
        return verifyPayload(payload, files, full = false)
    }

    private fun readyGenerations(key: String): List<String> =
        sortedChildren(generationsRoot(key)).map { it.name }.filter { readGenerationRecord(key, it) != null }

    // endregion

    // region Filesystem

    private fun bundleKeys(): List<String> =
        sortedChildren(rootDirectory).map { it.name }.filter { BUNDLE.matches(it) }

    private fun bundleDirectory(key: String) = File(rootDirectory, key)
    private fun stagingRoot(key: String) = File(bundleDirectory(key), STAGING)
    private fun generationsRoot(key: String) = File(bundleDirectory(key), GENERATIONS)
    private fun quarantineRoot(key: String) = File(bundleDirectory(key), QUARANTINE)
    private fun trashRoot(key: String) = File(bundleDirectory(key), TRASH)
    private fun leasesRoot(key: String) = File(bundleDirectory(key), LEASES)
    private fun stagingDirectory(key: String, generation: String) = File(stagingRoot(key), generation)
    private fun generationDirectory(key: String, generation: String) = File(generationsRoot(key), generation)
    private fun quarantineDirectory(key: String, generation: String) = File(quarantineRoot(key), generation)
    private fun generationRecordFile(key: String, generation: String) =
        File(generationDirectory(key, generation), GENERATION_RECORD)

    private fun leaseFile(key: String, generation: String, id: String) =
        File(File(leasesRoot(key), generation), "$id.json")

    private fun trashEntry(key: String, generation: String): File {
        trashRoot(key).mkdirs()
        return File(trashRoot(key), "$generation.${UUID.randomUUID()}")
    }

    private fun sortedChildren(directory: File): List<File> =
        directory.listFiles()?.filter { !it.name.endsWith(TMP_SUFFIX) }?.sortedBy { it.name } ?: emptyList()

    /** Moves [entry] out of reach with one rename, then deletes it. */
    private fun discard(key: String, entry: File) {
        if (!entry.exists()) return
        val trash = trashEntry(key, entry.name.substringBefore('.'))
        move(entry, trash)
        trash.deleteRecursively()
    }

    private fun quarantine(key: String, entry: File, generation: String, reason: String) {
        if (!entry.exists()) return
        writeFileAtomic(
            File(entry, QUARANTINE_RECORD),
            Json.encode(
                obj(
                    "schema" to Json.Str(QUARANTINE_SCHEMA),
                    "schema_version" to Json.Num(1),
                    "generation" to Json.Str(generation),
                    "reason" to Json.Str(reason),
                    "quarantined_at_ms" to Json.Num(now()),
                ),
            ),
        )
        val destination = quarantineDirectory(key, generation)
        if (destination.exists()) discard(key, destination)
        destination.parentFile?.mkdirs()
        move(entry, destination)
    }

    private fun move(source: File, destination: File) {
        destination.parentFile?.mkdirs()
        try {
            Files.move(source.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE)
        } catch (error: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), destination.toPath())
        }
    }

    private fun writeFileAtomic(file: File, data: ByteArray) {
        file.parentFile?.mkdirs()
        val temporary = File(file.parentFile, file.name + TMP_SUFFIX)
        temporary.writeBytes(data)
        try {
            Files.move(
                temporary.toPath(), file.toPath(),
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (error: AtomicMoveNotSupportedException) {
            Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun readJson(file: File): Json? = try {
        if (file.isFile) Json.parse(file.readBytes()) else null
    } catch (error: Json.Invalid) {
        null
    } catch (error: IOException) {
        null
    }

    private fun obj(vararg members: Pair<String, Json>) = Json.Obj(members.toList())

    private fun now(): Long = options.clock.nowEpochMillis()

    private fun checkpoint(name: String) {
        checkpointHook?.invoke(name)
    }

    // endregion

    // region Locks

    private inner class HeldLock(
        private val mutex: ReentrantLock,
        private val channel: FileChannel?,
        private val fileLock: FileLock?,
        private val record: File?,
        val stale: Boolean,
    ) : Closeable {
        private var released = false

        override fun close() = release(keepRecord = false)

        fun release(keepRecord: Boolean) {
            if (released) return
            released = true
            try {
                if (!keepRecord) record?.delete()
                fileLock?.release()
                channel?.close()
            } catch (_: IOException) {
            } finally {
                mutex.unlock()
            }
        }

        fun <T> guarded(block: () -> T): T {
            try {
                val result = block()
                release(keepRecord = false)
                return result
            } catch (crash: BundleCacheCrash) {
                release(keepRecord = true)
                throw crash
            } catch (error: Throwable) {
                release(keepRecord = false)
                throw error
            }
        }
    }

    private fun <T> locked(key: String, block: (String) -> T): T {
        val lock = lockBundle(key, options.lockTimeoutMillis)
            ?: throw BundleCacheException(BundleCacheReason.LOCK_TIMEOUT)
        return lock.guarded { block(key) }
    }

    private fun lockBundle(key: String, timeoutMillis: Long): HeldLock? {
        val directory = bundleDirectory(key)
        directory.mkdirs()
        val lockFile = File(directory, LOCK)
        val mutex = MUTEXES.computeIfAbsent(lockFile.canonicalPath) { ReentrantLock() }
        if (mutex.isHeldByCurrentThread) {
            mutex.lock()
            return HeldLock(mutex, null, null, null, stale = false)
        }
        if (!mutex.tryLock(timeoutMillis, TimeUnit.MILLISECONDS)) return null
        var channel: FileChannel? = null
        try {
            channel = RandomAccessFile(lockFile, "rw").channel
            val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis)
            var fileLock = channel.tryLock()
            while (fileLock == null && System.nanoTime() < deadline) {
                Thread.sleep(LOCK_POLL_MILLIS)
                fileLock = channel.tryLock()
            }
            if (fileLock == null) {
                channel.close()
                mutex.unlock()
                return null
            }
            val record = File(directory, LOCK_RECORD)
            val stale = record.exists()
            writeFileAtomic(
                record,
                Json.encode(
                    obj(
                        "schema" to Json.Str(LOCK_SCHEMA),
                        "schema_version" to Json.Num(1),
                        "owner" to Json.Str(options.ownerId),
                        "acquired_at_ms" to Json.Num(now()),
                    ),
                ),
            )
            return HeldLock(mutex, channel, fileLock, record, stale)
        } catch (error: Exception) {
            try {
                channel?.close()
            } catch (_: IOException) {
            }
            mutex.unlock()
            if (error is InterruptedException) Thread.currentThread().interrupt()
            throw BundleCacheException(BundleCacheReason.STORAGE_IO)
        }
    }

    // endregion

    private fun success(
        key: String,
        generation: String,
        totalBytes: Long,
        fileCount: Int,
        fromCache: Boolean,
        resumed: Boolean = false,
        resumeReason: BundleResumeReason? = null,
        transferred: Long = 0L,
        evicted: List<String> = emptyList(),
    ): BundleAcquireResult {
        val pointer = readPointer(key)
        val record = readGenerationRecord(key, generation)
        return BundleAcquireResult(
            ok = true, reason = null, bundleReason = null, bundleKey = key, generation = generation,
            state = record?.let { state(key, generation, pointer, it) }, fromCache = fromCache,
            resumed = resumed, resumeReason = resumeReason, bytesTransferred = transferred,
            verifiedBytes = totalBytes, filesVerified = fileCount, evicted = evicted,
        )
    }

    private fun failure(
        reason: BundleCacheReason,
        bundleReason: PortableBundleReason?,
        key: String?,
        generation: String?,
    ) = BundleAcquireResult(
        ok = false, reason = reason, bundleReason = bundleReason, bundleKey = key, generation = generation,
        state = null, fromCache = false, resumed = false, resumeReason = null, bytesTransferred = 0L,
        verifiedBytes = 0L, filesVerified = 0, evicted = emptyList(),
    )

    override fun toString(): String = "BundleGenerationCache(quotaBytes=${options.quotaBytes})"

    companion object {
        /** Contract version of the cache records and results. */
        const val CONTRACT_VERSION = 1

        /** Directory name used under [ModelCache]'s root. */
        const val DIRECTORY_NAME = "bundles"

        internal const val CHECKPOINT_STAGING_RECORD_WRITTEN = "staging_record_written"
        internal const val CHECKPOINT_CHUNK_COMMITTED = "chunk_committed"
        internal const val CHECKPOINT_VERIFYING_RECORD_WRITTEN = "verifying_record_written"
        internal const val CHECKPOINT_GENERATION_COMMITTED = "generation_committed"
        internal const val CHECKPOINT_POINTER_COMMITTED = "pointer_committed"
        internal const val CHECKPOINT_EVICTION_COMMITTED = "eviction_committed"

        private const val STAGING = "staging"
        private const val GENERATIONS = "generations"
        private const val QUARANTINE = "quarantine"
        private const val TRASH = "trash"
        private const val LEASES = "leases"
        private const val PAYLOAD = "payload"
        private const val POINTER = "pointer.json"
        private const val LOCK = "lock"
        private const val LOCK_RECORD = "lock.json"
        private const val STAGING_RECORD = "staging.json"
        private const val GENERATION_RECORD = "generation.json"
        private const val QUARANTINE_RECORD = "quarantine.json"
        private const val TMP_SUFFIX = ".tmp"
        private const val PHASE_STAGING = "staging"
        private const val PHASE_VERIFYING = "verifying"
        private const val QUARANTINE_PAYLOAD_MISMATCH = "payload_mismatch"
        private const val QUARANTINE_ROLLED_BACK = "rolled_back"
        private const val POINTER_SCHEMA = "openmed.bundle_cache.pointer"
        private const val GENERATION_SCHEMA = "openmed.bundle_cache.generation"
        private const val STAGING_SCHEMA = "openmed.bundle_cache.staging"
        private const val QUARANTINE_SCHEMA = "openmed.bundle_cache.quarantine"
        private const val LEASE_SCHEMA = "openmed.bundle_cache.lease"
        private const val LOCK_SCHEMA = "openmed.bundle_cache.lock"
        private const val LOCK_POLL_MILLIS = 10L

        private val GENERATION = Regex("g-[0-9a-f]{32}")
        private val BUNDLE = Regex("b-[0-9a-f]{32}")
        private val MUTEXES = ConcurrentHashMap<String, ReentrantLock>()

        /** Opaque directory key of a bundle id. */
        fun bundleKey(bundleId: String): String =
            "b-" + sha256Hex(bundleId.toByteArray(StandardCharsets.UTF_8)).take(32)

        /** Opaque generation id of a manifest content identity (`sha256:<hex>`). */
        fun generationId(contentId: String): String = "g-" + contentId.removePrefix("sha256:").take(32)

        /** Binding digest of a trust policy, recorded with resumable bytes. */
        fun trustDigest(policy: PortableBundleTrustPolicy): String {
            val keys = policy.keys.sortedBy { it.kid }.map { key ->
                Json.Obj(
                    listOf(
                        "kid" to Json.Str(key.kid),
                        "not_after" to Json.Str(key.notAfter),
                        "not_before" to Json.Str(key.notBefore),
                        "revoked" to Json.Bool(key.revoked),
                        "x" to Json.Str(Base64Url.encode(key.publicKey)),
                    ),
                )
            }
            val value = Json.Obj(
                listOf(
                    "keys" to Json.Arr(keys),
                    "revoked_content_ids" to Json.Arr(policy.revokedContentIds.sorted().map { Json.Str(it) }),
                    "threshold" to Json.Num(policy.threshold.toLong()),
                ),
            )
            return sha256Text(Json.encode(value))
        }

        private fun sha256Hex(data: ByteArray, length: Int = data.size): String {
            val digest = MessageDigest.getInstance("SHA-256")
            digest.update(data, 0, length)
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun sha256Text(data: ByteArray, length: Int = data.size): String = "sha256:" + sha256Hex(data, length)

        private fun sha256File(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1 shl 16)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return "sha256:" + digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
