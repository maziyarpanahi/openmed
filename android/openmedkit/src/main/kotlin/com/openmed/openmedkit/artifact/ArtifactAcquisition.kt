package com.openmed.openmedkit.artifact

import android.content.Context
import com.openmed.openmedkit.cache.CachedArtifactRecord
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.cache.UnsafeModelPathException
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.download.HttpHuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.ModelDownloadCancelledException
import com.openmed.openmedkit.download.ModelDownloadException
import com.openmed.openmedkit.download.ModelDownloadLogger
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.download.ModelFailure
import com.openmed.openmedkit.download.ModelIntegrityException
import com.openmed.openmedkit.download.ModelTransferControl
import com.openmed.openmedkit.download.StagedModelArtifact
import com.openmed.openmedkit.download.hex
import com.openmed.openmedkit.download.sha256Hex
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerializationException
import java.io.File
import java.io.IOException
import java.net.ConnectException
import java.net.UnknownHostException
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/** Whether an acquisition may use the network. Network use is never implied. */
enum class ArtifactNetworkConsent {
    /** Resolve from the offline cache only. */
    DENIED,

    /** The caller approved downloading the artifact when it is not cached. */
    GRANTED,
}

/** How thoroughly a cached artifact is checked before it is returned. */
enum class ArtifactCacheVerification {
    /** Recorded digest, file set, and file sizes. */
    METADATA,

    /** Every file is also re-hashed against its recorded SHA-256. */
    FULL,
}

/** Stable acquisition outcome states. */
enum class ArtifactAcquisitionState {
    /** Served from the offline cache after a metadata check. */
    CACHED,

    /** Fetched over the network, verified, and promoted atomically. */
    DOWNLOADED,

    /** Served from the offline cache after every file was re-hashed. */
    VERIFIED,

    /** The artifact cannot run on the selected backend. */
    UNSUPPORTED,

    /** The acquisition was cancelled; downloaded bytes are kept for resume. */
    CANCELLED,

    /** The acquisition failed; see the reason code. */
    FAILED,
}

/** Offline cache state of one artifact. */
enum class ArtifactCacheState {
    MISSING,
    PARTIAL,
    READY,
}

/**
 * Controlled, value-free reason codes. They never carry URLs, paths, tokens,
 * file contents, or exception messages.
 */
enum class ArtifactAcquisitionReason(val code: String) {
    FORMAT_NOT_ANDROID_RUNNABLE("format_not_android_runnable"),
    BUNDLE_VERIFIER_UNAVAILABLE("bundle_verifier_unavailable"),
    FORMAT_NOT_SUPPORTED_BY_BACKEND("format_not_supported_by_backend"),
    RUNTIME_MISMATCH("runtime_mismatch"),
    RUNTIME_VERSION_TOO_OLD("runtime_version_too_old"),
    OPSET_NOT_SUPPORTED("opset_not_supported"),
    ANDROID_SDK_TOO_OLD("android_sdk_too_old"),
    BACKEND_CAPABILITY_UNDECLARED("backend_capability_undeclared"),
    FORMAT_NOT_PUBLISHED("format_not_published"),
    FORMAT_SIGNATURE_MISMATCH("format_signature_mismatch"),
    CANCELLED("cancelled"),
    NETWORK_CONSENT_REQUIRED("network_consent_required"),
    MANIFEST_DIGEST_MISMATCH("manifest_digest_mismatch"),
    MANIFEST_INVALID("manifest_invalid"),
    FILE_DIGEST_MISMATCH("file_digest_mismatch"),
    SIZE_MISMATCH("size_mismatch"),
    CACHED_ARTIFACT_CORRUPT("cached_artifact_corrupt"),
    UNSAFE_PATH("unsafe_path"),
    HTTP_STATUS("http_status"),
    NETWORK_UNAVAILABLE("network_unavailable"),
    TRANSFER_INTERRUPTED("transfer_interrupted"),
    CACHE_IO("cache_io"),
    UNEXPECTED("unexpected"),
}

/**
 * A format-aware acquisition request.
 *
 * @property ref Artifact to resolve.
 * @property backend Capability of the backend that will open the artifact.
 * @property networkConsent Defaults to [ArtifactNetworkConsent.DENIED]: nothing
 *   is downloaded unless the caller opts in.
 * @property cacheVerification Depth of the check applied to a cache hit.
 */
class ArtifactAcquisitionRequest(
    val ref: ArtifactRef,
    val backend: ArtifactBackendCapability,
    val networkConsent: ArtifactNetworkConsent = ArtifactNetworkConsent.DENIED,
    val cacheVerification: ArtifactCacheVerification = ArtifactCacheVerification.METADATA,
)

/** One verified file of an [AcquiredArtifact], relative to its directory. */
data class AcquiredArtifactFile(
    val path: String,
    val sizeBytes: Long,
    val sha256: String,
)

/**
 * A verified, backend-compatible artifact that is safe to open.
 *
 * Payload files are read-only once promoted. [toString] omits [directory], so
 * the private on-device path does not leak into logs.
 *
 * @property contentDigest `sha256:` digest over the sorted file list (path,
 *   size, SHA-256): a stable identity for the bytes on disk.
 */
class AcquiredArtifact internal constructor(
    val directory: File,
    val format: ArtifactFormat,
    val runtime: ArtifactRuntime,
    val digest: String,
    val contentDigest: String,
    val files: List<AcquiredArtifactFile>,
) {
    val sizeBytes: Long
        get() = files.sumOf { it.sizeBytes }

    override fun toString(): String =
        "AcquiredArtifact(format=${format.wireName}, runtime=${runtime.wireName}, " +
            "digest=$digest, contentDigest=$contentDigest, sizeBytes=$sizeBytes, " +
            "files=${files.size})"
}

/**
 * Result of [ArtifactAcquisition.acquire]. Match the subtype or read [state];
 * every [Ready] artifact is verified, compatible, immutable, and safe to open.
 * [toString] carries only states, formats, digests, sizes, and reason codes.
 */
sealed class ArtifactAcquisitionResult {
    abstract val ref: ArtifactRef
    abstract val state: ArtifactAcquisitionState
    open val artifact: AcquiredArtifact? get() = null
    open val reason: ArtifactAcquisitionReason? get() = null

    sealed class Ready : ArtifactAcquisitionResult() {
        abstract override val artifact: AcquiredArtifact
    }

    class Cached internal constructor(
        override val ref: ArtifactRef,
        override val artifact: AcquiredArtifact,
    ) : Ready() {
        override val state get() = ArtifactAcquisitionState.CACHED
    }

    class Verified internal constructor(
        override val ref: ArtifactRef,
        override val artifact: AcquiredArtifact,
    ) : Ready() {
        override val state get() = ArtifactAcquisitionState.VERIFIED
    }

    /**
     * @property resumed `true` when staged bytes from an earlier attempt were reused.
     * @property bytesTransferred Bytes fetched by this attempt.
     */
    class Downloaded internal constructor(
        override val ref: ArtifactRef,
        override val artifact: AcquiredArtifact,
        val resumed: Boolean,
        val bytesTransferred: Long,
    ) : Ready() {
        override val state get() = ArtifactAcquisitionState.DOWNLOADED
    }

    class Unsupported internal constructor(
        override val ref: ArtifactRef,
        override val reason: ArtifactAcquisitionReason,
    ) : ArtifactAcquisitionResult() {
        override val state get() = ArtifactAcquisitionState.UNSUPPORTED
    }

    class Cancelled internal constructor(
        override val ref: ArtifactRef,
    ) : ArtifactAcquisitionResult() {
        override val state get() = ArtifactAcquisitionState.CANCELLED
        override val reason get() = ArtifactAcquisitionReason.CANCELLED
    }

    /** @property httpStatus Numeric status for [ArtifactAcquisitionReason.HTTP_STATUS]. */
    class Failed internal constructor(
        override val ref: ArtifactRef,
        override val reason: ArtifactAcquisitionReason,
        val httpStatus: Int? = null,
    ) : ArtifactAcquisitionResult() {
        override val state get() = ArtifactAcquisitionState.FAILED
    }

    override fun toString(): String =
        "ArtifactAcquisitionResult(state=$state, format=${ref.format.wireName}, " +
            "runtime=${ref.runtime.wireName}, digest=${ref.digest}, " +
            "reason=${reason?.code}, sizeBytes=${artifact?.sizeBytes})"
}

/** Receives every acquisition result; results are value-free evidence. */
fun interface ArtifactAcquisitionLogger {
    fun log(result: ArtifactAcquisitionResult)

    companion object {
        val NONE = ArtifactAcquisitionLogger {}
    }
}

/**
 * Format-aware model artifact acquisition for Android backends.
 *
 * Wraps the existing [ModelDownloader] (manifest digest, per-file SHA-256,
 * safe paths) and [ModelCache] (staging, atomic promotion, LRU budget). An
 * acquisition checks backend compatibility first, then the offline cache, and
 * downloads only when the request carries [ArtifactNetworkConsent.GRANTED].
 * Downloads resume with HTTP Range requests and are re-hashed in full before
 * promotion, so partial or mismatched files never become ready.
 *
 * Concurrent callers for the same artifact and cache root share one transfer
 * within the process. Cancelling one caller's coroutine leaves the transfer
 * running for the others; it stops, keeping its bytes for resume, once no
 * caller waits or [cancel] is called. Acquisition never receives clinical text
 * or patient context.
 */
class ArtifactAcquisition(
    private val cache: ModelCache,
    client: HuggingFaceModelClient = HttpHuggingFaceModelClient(),
    private val logger: ArtifactAcquisitionLogger = ArtifactAcquisitionLogger.NONE,
) {
    constructor(
        context: Context,
        client: HuggingFaceModelClient = HttpHuggingFaceModelClient(),
        logger: ArtifactAcquisitionLogger = ArtifactAcquisitionLogger.NONE,
    ) : this(ModelCache(context), client, logger)

    private val downloader = ModelDownloader(cache, client, ModelDownloadLogger.NONE)

    /** Resolve [request] to a ready artifact or a stable non-ready state. */
    suspend fun acquire(request: ArtifactAcquisitionRequest): ArtifactAcquisitionResult {
        val ref = request.ref
        val result = request.backend.incompatibility(ref)
            ?.let { ArtifactAcquisitionResult.Unsupported(ref, it) }
            ?: withContext(Dispatchers.IO) { resolveCached(request) }
            ?: if (request.networkConsent == ArtifactNetworkConsent.GRANTED) {
                awaitTransfer(request)
            } else {
                ArtifactAcquisitionResult.Failed(
                    ref,
                    ArtifactAcquisitionReason.NETWORK_CONSENT_REQUIRED,
                )
            }
        logger.log(result)
        return result
    }

    /** Ready artifact from the offline cache, or `null`. Never uses the network. */
    fun cachedArtifact(ref: ArtifactRef): AcquiredArtifact? {
        val record = try {
            cache.readyArtifact(ref.identity, ref.digest)
        } catch (error: UnsafeModelPathException) {
            null
        } ?: return null
        if (record.format != ref.format.wireName || record.runtime != ref.runtime.wireName) {
            return null
        }
        return AcquiredArtifact(
            directory = cache.artifactDirectory(ref.identity),
            format = ref.format,
            runtime = ref.runtime,
            digest = ref.digest,
            contentDigest = record.contentDigest,
            files = record.files.map { AcquiredArtifactFile(it.path, it.sizeBytes, it.sha256) },
        )
    }

    fun cacheState(ref: ArtifactRef): ArtifactCacheState = when {
        cachedArtifact(ref) != null -> ArtifactCacheState.READY
        cache.hasResumableBytes(ref.identity) -> ArtifactCacheState.PARTIAL
        else -> ArtifactCacheState.MISSING
    }

    /** Cancel the in-flight transfer for [ref]; waiting callers get `CANCELLED`. */
    fun cancel(ref: ArtifactRef): Boolean = synchronized(flights) {
        val flight = flights[flightKey(ref)] ?: return false
        flight.deferred.cancel()
        true
    }

    internal fun waiterCount(ref: ArtifactRef): Int =
        synchronized(flights) { flights[flightKey(ref)]?.waiters ?: 0 }

    /** Remove the ready artifact and, when no transfer is running, its staged bytes. */
    fun evict(ref: ArtifactRef): Boolean {
        val inFlight = synchronized(flights) { flights.containsKey(flightKey(ref)) }
        return cache.evictArtifact(ref.identity, includeStaging = !inFlight)
    }

    private fun resolveCached(request: ArtifactAcquisitionRequest): ArtifactAcquisitionResult? {
        val ref = request.ref
        val artifact = cachedArtifact(ref) ?: return null
        if (request.cacheVerification == ArtifactCacheVerification.METADATA) {
            return ArtifactAcquisitionResult.Cached(ref, artifact)
        }
        val intact = try {
            contentDigest(artifact.files) == artifact.contentDigest &&
                artifact.files.all { file ->
                    val payload = cache.artifactFile(artifact.directory, file.path)
                    payload != null && sha256Hex(payload) == file.sha256
                }
        } catch (error: IOException) {
            false
        }
        if (intact) {
            return ArtifactAcquisitionResult.Verified(ref, artifact)
        }
        cache.evictArtifact(ref.identity, includeStaging = false)
        return if (request.networkConsent == ArtifactNetworkConsent.GRANTED) {
            null
        } else {
            ArtifactAcquisitionResult.Failed(ref, ArtifactAcquisitionReason.CACHED_ARTIFACT_CORRUPT)
        }
    }

    private suspend fun awaitTransfer(
        request: ArtifactAcquisitionRequest,
    ): ArtifactAcquisitionResult {
        val flight = joinFlight(flightKey(request.ref), request)
        try {
            return flight.deferred.await()
        } catch (cancellation: CancellationException) {
            // Rethrows when this caller was cancelled; otherwise the shared
            // transfer was cancelled through cancel(ref).
            currentCoroutineContext().ensureActive()
            return ArtifactAcquisitionResult.Cancelled(request.ref)
        } finally {
            synchronized(flights) {
                flight.waiters -= 1
                if (flight.waiters == 0 && !flight.deferred.isCompleted) {
                    flight.deferred.cancel()
                }
            }
        }
    }

    private suspend fun joinFlight(key: String, request: ArtifactAcquisitionRequest): Flight {
        while (true) {
            val draining: Job = synchronized(flights) {
                val existing = flights[key]
                if (existing == null) {
                    val flight = Flight(
                        flightScope.async(start = CoroutineStart.LAZY) { transfer(request) },
                    )
                    flights[key] = flight
                    flight.deferred.invokeOnCompletion {
                        synchronized(flights) {
                            if (flights[key] === flight) {
                                flights.remove(key)
                            }
                        }
                    }
                    flight.deferred.start()
                    return flight
                }
                if (!existing.deferred.isCancelled) {
                    existing.waiters += 1
                    return existing
                }
                existing.deferred
            }
            // A cancelled transfer is still releasing its staging directory.
            draining.join()
        }
    }

    private suspend fun transfer(request: ArtifactAcquisitionRequest): ArtifactAcquisitionResult {
        val ref = request.ref
        val job = coroutineContext.job
        cachedArtifact(ref)?.let { return ArtifactAcquisitionResult.Cached(ref, it) }
        var staging: File? = null
        return try {
            val stagingDirectory = cache.resumableStagingDirectory(ref.identity)
            staging = stagingDirectory
            val staged = downloader.stageArtifact(
                entry = ModelCatalogEntry(
                    repoId = ref.repoId,
                    formats = listOf(ref.format.wireName),
                    tier = null,
                    paramCount = null,
                    languages = emptyList(),
                    license = null,
                    reproducibilityHash = ref.digest,
                ),
                revision = ref.revision,
                stagingDirectory = stagingDirectory,
                expectedTotalBytes = ref.sizeBytes,
                control = ModelTransferControl { !job.isActive },
            )
            if (!hasFormatSignature(ref.format, stagingDirectory, staged)) {
                stagingDirectory.deleteRecursively()
                ArtifactAcquisitionResult.Unsupported(
                    ref,
                    ArtifactAcquisitionReason.FORMAT_SIGNATURE_MISMATCH,
                )
            } else {
                promote(ref, stagingDirectory, staged)
            }
        } catch (error: ModelDownloadCancelledException) {
            ArtifactAcquisitionResult.Cancelled(ref)
        } catch (error: CancellationException) {
            throw error
        } catch (error: Exception) {
            val result = failure(ref, error)
            if (result.reason in DISCARD_STAGING_REASONS) {
                staging?.deleteRecursively()
            }
            result
        }
    }

    private fun promote(
        ref: ArtifactRef,
        stagingDirectory: File,
        staged: StagedModelArtifact,
    ): ArtifactAcquisitionResult {
        val files = staged.files.map { AcquiredArtifactFile(it.path, it.sizeBytes, it.sha256) }
        val contentDigest = contentDigest(files)
        val directory = try {
            cache.storeReadyArtifact(
                repoId = ref.repoId,
                identity = ref.identity,
                sourceDirectory = stagingDirectory,
                integrityHash = staged.integrityHash,
                revisionSha = staged.revisionSha,
                artifact = CachedArtifactRecord(
                    ref.format.wireName,
                    ref.runtime.wireName,
                    contentDigest,
                    staged.files,
                ),
            )
        } catch (error: IOException) {
            return ArtifactAcquisitionResult.Failed(ref, ArtifactAcquisitionReason.CACHE_IO)
        }
        return ArtifactAcquisitionResult.Downloaded(
            ref,
            AcquiredArtifact(directory, ref.format, ref.runtime, ref.digest, contentDigest, files),
            staged.resumed,
            staged.bytesTransferred,
        )
    }

    private fun failure(ref: ArtifactRef, error: Exception): ArtifactAcquisitionResult {
        val failure = (error as? ModelIntegrityException)?.failure
            ?: (error as? ModelDownloadException)?.failure
        if (failure == ModelFailure.NO_RUNNABLE_FILES) {
            return ArtifactAcquisitionResult.Unsupported(
                ref,
                ArtifactAcquisitionReason.FORMAT_NOT_PUBLISHED,
            )
        }
        if (failure == ModelFailure.HTTP_STATUS) {
            return ArtifactAcquisitionResult.Failed(
                ref,
                ArtifactAcquisitionReason.HTTP_STATUS,
                (error as ModelDownloadException).httpStatus,
            )
        }
        val reason = when {
            failure == ModelFailure.MANIFEST -> ArtifactAcquisitionReason.MANIFEST_DIGEST_MISMATCH
            failure == ModelFailure.SIZE -> ArtifactAcquisitionReason.SIZE_MISMATCH
            error is ModelIntegrityException -> ArtifactAcquisitionReason.FILE_DIGEST_MISMATCH
            error is UnsafeModelPathException -> ArtifactAcquisitionReason.UNSAFE_PATH
            error is SerializationException -> ArtifactAcquisitionReason.MANIFEST_INVALID
            error is UnknownHostException ||
                error is ConnectException ||
                error is SecurityException -> ArtifactAcquisitionReason.NETWORK_UNAVAILABLE
            error is IOException -> ArtifactAcquisitionReason.TRANSFER_INTERRUPTED
            else -> ArtifactAcquisitionReason.UNEXPECTED
        }
        return ArtifactAcquisitionResult.Failed(ref, reason)
    }

    private fun flightKey(ref: ArtifactRef): String =
        cache.rootDirectory.absoluteFile.normalize().path + "\u0000" + ref.identity

    private class Flight(val deferred: Deferred<ArtifactAcquisitionResult>) {
        var waiters = 1
    }

    private companion object {
        val flights = HashMap<String, Flight>()
        val flightScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

        val DISCARD_STAGING_REASONS = setOf(
            ArtifactAcquisitionReason.MANIFEST_DIGEST_MISMATCH,
            ArtifactAcquisitionReason.MANIFEST_INVALID,
            ArtifactAcquisitionReason.FILE_DIGEST_MISMATCH,
            ArtifactAcquisitionReason.SIZE_MISMATCH,
            ArtifactAcquisitionReason.FORMAT_NOT_PUBLISHED,
            ArtifactAcquisitionReason.UNSAFE_PATH,
        )

        fun contentDigest(files: List<AcquiredArtifactFile>): String {
            val digest = MessageDigest.getInstance("SHA-256")
            files.sortedBy { it.path }.forEach { file ->
                digest.update("${file.path}\t${file.sizeBytes}\t${file.sha256}\n".toByteArray())
            }
            return "sha256:" + hex(digest.digest())
        }

        /**
         * Staged primary files must carry their format's signature, so a
         * mislabeled file never becomes ready. An ONNX `ModelProto` serializes
         * `ir_version` (field 1, varint, tag `0x08`) first; ORT and TFLite
         * flatbuffers carry `ORTM` and `TFL3` identifiers at byte 4.
         */
        fun hasFormatSignature(
            format: ArtifactFormat,
            directory: File,
            staged: StagedModelArtifact,
        ): Boolean {
            val suffixes = if (format == ArtifactFormat.ONNX) ONNX_SUFFIXES else TFLITE_SUFFIXES
            val primary = staged.files.filter { file ->
                suffixes.any { file.path.lowercase().endsWith(it) }
            }
            return primary.isNotEmpty() && primary.all { file ->
                val header = ByteArray(8)
                val length = File(directory, file.path).inputStream().use { input ->
                    var read = 0
                    while (read < header.size) {
                        val count = input.read(header, read, header.size - read)
                        if (count < 0) {
                            break
                        }
                        read += count
                    }
                    read
                }
                val identifier = if (length == header.size) {
                    String(header, 4, 4, Charsets.US_ASCII)
                } else {
                    ""
                }
                when {
                    file.path.lowercase().endsWith(".onnx") -> length > 0 && header[0] == ONNX_TAG
                    file.path.lowercase().endsWith(".ort") -> identifier == "ORTM"
                    else -> identifier == "TFL3"
                }
            }
        }

        const val ONNX_TAG: Byte = 0x08
        val ONNX_SUFFIXES = listOf(".onnx", ".ort")
        val TFLITE_SUFFIXES = listOf(".tflite")
    }
}
