package com.openmed.openmedkit.artifact

import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer.Companion.ERROR_BODY_SENTINEL
import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer.Companion.REVISION_SHA
import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer.Companion.SIGNED_QUERY_SENTINEL
import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer.Companion.sha256
import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer.Companion.syntheticOnnx
import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer.Companion.syntheticTflite
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelInfo
import java.io.File
import java.security.Permission
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ArtifactAcquisitionTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val servers = mutableListOf<HuggingFaceFixtureServer>()

    @After
    fun closeServers() {
        servers.forEach { it.close() }
    }

    @Test
    fun downloadsVerifiesAndPromotesOnlyTheRequestedFormat() = runBlocking<Unit> {
        val server = server(
            "model.onnx" to ONNX_BYTES,
            "model.tflite" to syntheticTflite(4096),
            "config.json" to CONFIG,
            "README.md" to "not runnable".toByteArray(),
        )
        val cache = newCache()
        val acquisition = ArtifactAcquisition(cache, server.client())

        val onnx = acquisition.acquire(request(onnxRef(server)))
        val tflite = acquisition.acquire(
            request(
                onnxRef(server).copy(
                    format = ArtifactFormat.TFLITE,
                    runtime = ArtifactRuntime.TFLITE,
                ),
                backend = ArtifactBackendCapability.tfLite(),
            ),
        )

        assertIs<ArtifactAcquisitionResult.Downloaded>(onnx)
        assertEquals(ArtifactAcquisitionState.DOWNLOADED, onnx.state)
        assertFalse(onnx.resumed)
        assertEquals(listOf("config.json", "model.onnx"), onnx.artifact.files.map { it.path })
        assertEquals(sha256(ONNX_BYTES), onnx.artifact.files.last().sha256)
        assertEquals(ONNX_BYTES.size + CONFIG.size.toLong(), onnx.artifact.sizeBytes)
        assertIs<ArtifactAcquisitionResult.Downloaded>(tflite)
        assertEquals(listOf("config.json", "model.tflite"), tflite.artifact.files.map { it.path })
        assertTrue(onnx.artifact.directory != tflite.artifact.directory)
        assertModelBytes(onnx.artifact)
        assertTrue(server.cdnRequests("README.md").isEmpty())
        assertEquals(ArtifactCacheState.READY, acquisition.cacheState(onnxRef(server)))
    }

    @Test
    fun cacheHitIsServedOfflineAndFullVerificationReportsVerified() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)
        acquisition.acquire(request(ref))
        val requestsAfterDownload = server.requests.size

        val cached = acquisition.acquire(request(ref, ArtifactNetworkConsent.DENIED))
        val verified = acquisition.acquire(
            request(ref, ArtifactNetworkConsent.DENIED, ArtifactCacheVerification.FULL),
        )
        val grantedButCached = acquisition.acquire(request(ref))

        assertIs<ArtifactAcquisitionResult.Cached>(cached)
        assertIs<ArtifactAcquisitionResult.Verified>(verified)
        assertIs<ArtifactAcquisitionResult.Cached>(grantedButCached)
        assertEquals(cached.artifact.contentDigest, verified.artifact.contentDigest)
        assertEquals(requestsAfterDownload, server.requests.size)
    }

    @Test
    fun networkAcquisitionRequiresExplicitConsent() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)

        val result = acquisition.acquire(ArtifactAcquisitionRequest(ref, ONNX_BACKEND))

        assertIs<ArtifactAcquisitionResult.Failed>(result)
        assertEquals(ArtifactAcquisitionReason.NETWORK_CONSENT_REQUIRED, result.reason)
        assertTrue(server.requests.isEmpty())
        assertEquals(ArtifactCacheState.MISSING, acquisition.cacheState(ref))
    }

    @Test
    fun preparedDeviceResolvesOfflineWithNetworkAccessDenied() = runBlocking<Unit> {
        val root = temporaryFolder.newFolder("prepared-cache")
        val server = server()
        val ref = onnxRef(server)
        val prepared = ArtifactAcquisition(ModelCache(root), server.client()).acquire(request(ref))
        assertIs<ArtifactAcquisitionResult.Downloaded>(prepared)
        server.close()

        val attempts = CopyOnWriteArrayList<String>()
        val results = mutableListOf<ArtifactAcquisitionResult>()
        @Suppress("DEPRECATION")
        val previous = System.getSecurityManager()
        @Suppress("DEPRECATION")
        System.setSecurityManager(DenyNetworkSecurityManager(attempts))
        try {
            val offline = ArtifactAcquisition(ModelCache(root), NetworkForbiddenClient)
            results += offline.acquire(request(ref, ArtifactNetworkConsent.DENIED))
            results += offline.acquire(request(ref, ArtifactNetworkConsent.GRANTED))
            results += offline.acquire(
                request(ref, ArtifactNetworkConsent.DENIED, ArtifactCacheVerification.FULL),
            )
        } finally {
            @Suppress("DEPRECATION")
            System.setSecurityManager(previous)
        }

        assertEquals(
            listOf(
                ArtifactAcquisitionState.CACHED,
                ArtifactAcquisitionState.CACHED,
                ArtifactAcquisitionState.VERIFIED,
            ),
            results.map { it.state },
        )
        assertTrue(attempts.isEmpty(), "offline resolution attempted network I/O")
        assertEquals(prepared.artifact.contentDigest, results.last().artifact?.contentDigest)
    }

    @Test
    fun fullVerificationDetectsSameSizeTamperingAndEvicts() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)
        val downloaded = acquisition.acquire(request(ref))
        assertIs<ArtifactAcquisitionResult.Downloaded>(downloaded)
        val model = File(downloaded.artifact.directory, "model.onnx")
        model.setWritable(true)
        val tampered = model.readBytes().also { it[it.size - 1] = (it[it.size - 1] + 1).toByte() }
        model.writeBytes(tampered)

        val metadataOnly = acquisition.acquire(request(ref, ArtifactNetworkConsent.DENIED))
        val full = acquisition.acquire(
            request(ref, ArtifactNetworkConsent.DENIED, ArtifactCacheVerification.FULL),
        )

        assertIs<ArtifactAcquisitionResult.Cached>(metadataOnly)
        assertIs<ArtifactAcquisitionResult.Failed>(full)
        assertEquals(ArtifactAcquisitionReason.CACHED_ARTIFACT_CORRUPT, full.reason)
        assertEquals(ArtifactCacheState.MISSING, acquisition.cacheState(ref))
    }

    @Test
    fun digestMismatchesNeverBecomeReady() = runBlocking<Unit> {
        val manifestServer = server()
        val manifestMismatch = ArtifactAcquisition(newCache(), manifestServer.client())
            .acquire(request(onnxRef(manifestServer).copy(digest = "sha256:" + "f".repeat(64))))

        val fileServer = server()
        fileServer.advertisedSha256["model.onnx"] = "0".repeat(64)
        val fileAcquisition = ArtifactAcquisition(newCache(), fileServer.client())
        val fileMismatch = fileAcquisition.acquire(request(onnxRef(fileServer)))

        val sizeServer = server()
        val sizeMismatch = ArtifactAcquisition(newCache(), sizeServer.client()).acquire(
            request(onnxRef(sizeServer, sizeBytes = ONNX_BYTES.size + CONFIG.size + 1L)),
        )

        assertEquals(ArtifactAcquisitionReason.MANIFEST_DIGEST_MISMATCH, manifestMismatch.reason)
        assertTrue(manifestServer.cdnRequests("model.onnx").isEmpty())
        assertEquals(ArtifactAcquisitionReason.FILE_DIGEST_MISMATCH, fileMismatch.reason)
        assertNull(fileAcquisition.cachedArtifact(onnxRef(fileServer)))
        assertEquals(ArtifactCacheState.MISSING, fileAcquisition.cacheState(onnxRef(fileServer)))
        assertEquals(ArtifactAcquisitionReason.SIZE_MISMATCH, sizeMismatch.reason)
        assertTrue(sizeServer.cdnRequests("model.onnx").isEmpty())
        listOf(manifestMismatch, fileMismatch, sizeMismatch).forEach {
            assertEquals(ArtifactAcquisitionState.FAILED, it.state)
            assertNull(it.artifact)
        }
    }

    @Test
    fun formatMismatchesResolveAsUnsupported() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val base = onnxRef(server)
        val prechecked = listOf(
            base.copy(format = ArtifactFormat.MLX, runtime = ArtifactRuntime.MLX) to
                ArtifactAcquisitionReason.FORMAT_NOT_ANDROID_RUNNABLE,
            base.copy(format = ArtifactFormat.CORE_ML, runtime = ArtifactRuntime.CORE_ML) to
                ArtifactAcquisitionReason.FORMAT_NOT_ANDROID_RUNNABLE,
            base.copy(format = ArtifactFormat.SIGNED_PORTABLE_BUNDLE) to
                ArtifactAcquisitionReason.BUNDLE_VERIFIER_UNAVAILABLE,
            base.copy(format = ArtifactFormat.TFLITE, runtime = ArtifactRuntime.TFLITE) to
                ArtifactAcquisitionReason.FORMAT_NOT_SUPPORTED_BY_BACKEND,
        )
        prechecked.forEach { (ref, reason) ->
            val result = acquisition.acquire(request(ref))
            assertIs<ArtifactAcquisitionResult.Unsupported>(result)
            assertEquals(reason, result.reason)
        }
        assertTrue(server.requests.isEmpty(), "unsupported formats must not touch the network")

        val tfliteOnly = server("model.tflite" to syntheticTflite(2048), "config.json" to CONFIG)
        val notPublished = ArtifactAcquisition(newCache(), tfliteOnly.client())
            .acquire(request(onnxRef(tfliteOnly)))
        assertEquals(ArtifactAcquisitionReason.FORMAT_NOT_PUBLISHED, notPublished.reason)
        assertTrue(tfliteOnly.requests.none { it.path.startsWith("/cdn/") })

        val mislabeled = server("model.onnx" to syntheticTflite(2048), "config.json" to CONFIG)
        val mislabeledAcquisition = ArtifactAcquisition(newCache(), mislabeled.client())
        val signature = mislabeledAcquisition.acquire(request(onnxRef(mislabeled)))
        assertIs<ArtifactAcquisitionResult.Unsupported>(signature)
        assertEquals(ArtifactAcquisitionReason.FORMAT_SIGNATURE_MISMATCH, signature.reason)
        assertEquals(
            ArtifactCacheState.MISSING,
            mislabeledAcquisition.cacheState(onnxRef(mislabeled)),
        )
    }

    @Test
    fun backendIncompatibilityIsCheckedBeforeCacheOrNetwork() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val base = onnxRef(server)
        val backend = ArtifactBackendCapability.onnxRuntime(
            runtimeVersion = "1.25.1",
            maxOpsetVersion = 20,
            androidSdkLevel = 26,
        )
        val cases = listOf(
            base.copy(runtime = ArtifactRuntime.TFLITE) to
                ArtifactAcquisitionReason.RUNTIME_MISMATCH,
            base.copy(compatibility = ArtifactCompatibility(minRuntimeVersion = "1.30.0")) to
                ArtifactAcquisitionReason.RUNTIME_VERSION_TOO_OLD,
            base.copy(compatibility = ArtifactCompatibility(opsetVersion = 21)) to
                ArtifactAcquisitionReason.OPSET_NOT_SUPPORTED,
            base.copy(compatibility = ArtifactCompatibility(minAndroidSdk = 34)) to
                ArtifactAcquisitionReason.ANDROID_SDK_TOO_OLD,
        )
        cases.forEach { (ref, reason) ->
            val result = acquisition.acquire(request(ref, backend = backend))
            assertEquals(ArtifactAcquisitionState.UNSUPPORTED, result.state)
            assertEquals(reason, result.reason)
        }
        val undeclared = acquisition.acquire(
            request(base.copy(compatibility = ArtifactCompatibility(minRuntimeVersion = "1.0"))),
        )
        assertEquals(ArtifactAcquisitionReason.BACKEND_CAPABILITY_UNDECLARED, undeclared.reason)
        assertTrue(server.requests.isEmpty())

        val compatible = acquisition.acquire(
            request(
                base.copy(
                    compatibility = ArtifactCompatibility(
                        minRuntimeVersion = "1.25.0",
                        opsetVersion = 17,
                        minAndroidSdk = 26,
                    ),
                ),
                backend = backend,
            ),
        )
        assertIs<ArtifactAcquisitionResult.Downloaded>(compatible)
    }

    @Test
    fun resumesAnInterruptedTransferWithARangeRequest() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)
        server.dropAfterBytes.set(100_000)

        val interrupted = acquisition.acquire(request(ref))

        assertIs<ArtifactAcquisitionResult.Failed>(interrupted)
        assertEquals(ArtifactAcquisitionReason.TRANSFER_INTERRUPTED, interrupted.reason)
        assertEquals(ArtifactCacheState.PARTIAL, acquisition.cacheState(ref))
        assertNull(acquisition.cachedArtifact(ref))

        val resumed = acquisition.acquire(request(ref))

        assertIs<ArtifactAcquisitionResult.Downloaded>(resumed)
        assertTrue(resumed.resumed)
        val rangeStart = assertNotNull(server.cdnRequests("model.onnx").last().range)
            .removePrefix("bytes=").removeSuffix("-").toLong()
        assertTrue(rangeStart > 0L)
        assertEquals(CONFIG.size + ONNX_BYTES.size - rangeStart, resumed.bytesTransferred)
        assertModelBytes(resumed.artifact)
        assertEquals(ArtifactCacheState.READY, acquisition.cacheState(ref))
    }

    @Test
    fun corruptedResumedBytesFailTheIntegrityRecheckAndRestart() = runBlocking<Unit> {
        val cache = newCache()
        val server = server()
        val acquisition = ArtifactAcquisition(cache, server.client())
        val ref = onnxRef(server)
        server.dropAfterBytes.set(100_000)
        acquisition.acquire(request(ref))
        val staged = File(cache.resumableStagingDirectory(ref.identity), "model.onnx")
        staged.writeBytes(staged.readBytes().also { it[10] = (it[10] + 1).toByte() })

        val result = acquisition.acquire(request(ref))

        assertIs<ArtifactAcquisitionResult.Downloaded>(result)
        assertFalse(result.resumed)
        val modelRequests = server.cdnRequests("model.onnx")
        assertTrue(modelRequests[modelRequests.size - 2].range != null)
        assertNull(modelRequests.last().range)
        assertModelBytes(result.artifact)
    }

    @Test
    fun cancellationMidDownloadKeepsBytesForResumeAndNeverExposesThem() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)
        server.pauseAfterBytes.set(64 * 1024)

        val pending = async(Dispatchers.Default) { acquisition.acquire(request(ref)) }
        assertTrue(server.paused.await(30, TimeUnit.SECONDS))
        assertNull(acquisition.cachedArtifact(ref))
        assertTrue(acquisition.cancel(ref))
        server.release.countDown()
        val cancelled = pending.await()

        assertIs<ArtifactAcquisitionResult.Cancelled>(cancelled)
        assertEquals(ArtifactAcquisitionState.CANCELLED, cancelled.state)
        assertEquals(ArtifactAcquisitionReason.CANCELLED, cancelled.reason)
        assertEquals(ArtifactCacheState.PARTIAL, acquisition.cacheState(ref))

        val resumed = acquisition.acquire(request(ref))
        assertIs<ArtifactAcquisitionResult.Downloaded>(resumed)
        assertTrue(resumed.resumed)
        assertNotNull(server.cdnRequests("model.onnx").last().range)
        assertModelBytes(resumed.artifact)
    }

    @Test
    fun cancellingTheOnlyCallerStopsTheSharedTransfer() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)
        server.pauseAfterBytes.set(64 * 1024)

        val caller = launch(Dispatchers.Default, start = CoroutineStart.UNDISPATCHED) {
            acquisition.acquire(request(ref))
        }
        assertTrue(server.paused.await(30, TimeUnit.SECONDS))
        caller.cancel()
        caller.join()
        server.release.countDown()

        assertTrue(caller.isCancelled)
        val resumed = acquisition.acquire(request(ref))
        assertIs<ArtifactAcquisitionResult.Downloaded>(resumed)
        assertTrue(resumed.resumed)
        assertEquals(2, server.cdnRequests("model.onnx").size)
    }

    @Test
    fun concurrentCallersShareOneVerifiedArtifactWithoutSeeingPartialFiles() = runBlocking<Unit> {
        val cache = newCache()
        val server = server()
        val acquisition = ArtifactAcquisition(cache, server.client())
        val ref = onnxRef(server)
        server.pauseAfterBytes.set(64 * 1024)

        val callers = (1..CONCURRENT_CALLERS).map {
            async(Dispatchers.Default) { acquisition.acquire(request(ref)) }
        }
        assertTrue(server.paused.await(30, TimeUnit.SECONDS))
        withTimeout(30_000) {
            while (acquisition.waiterCount(ref) < CONCURRENT_CALLERS) {
                delay(5)
            }
        }
        val offlineWhileInFlight = acquisition.acquire(request(ref, ArtifactNetworkConsent.DENIED))
        assertEquals(
            ArtifactAcquisitionReason.NETWORK_CONSENT_REQUIRED,
            offlineWhileInFlight.reason,
        )
        assertNull(acquisition.cachedArtifact(ref))
        assertFalse(File(cache.artifactDirectory(ref.identity), "model.onnx").exists())
        server.release.countDown()
        val results = callers.awaitAll()

        assertTrue(results.all { it is ArtifactAcquisitionResult.Downloaded })
        assertEquals(1, results.map { it.artifact?.directory }.toSet().size)
        assertEquals(1, results.map { it.artifact?.contentDigest }.toSet().size)
        assertEquals(1, server.manifestRequests().size)
        assertEquals(1, server.cdnRequests("model.onnx").size)
        assertEquals(1, server.cdnRequests("config.json").size)
        assertEquals(0, acquisition.waiterCount(ref))
    }

    @Test
    fun downloadedArtifactIsReadOnlyAndEvictable() = runBlocking<Unit> {
        val server = server()
        val acquisition = ArtifactAcquisition(newCache(), server.client())
        val ref = onnxRef(server)

        val result = acquisition.acquire(request(ref))

        assertIs<ArtifactAcquisitionResult.Downloaded>(result)
        result.artifact.files.forEach {
            assertFalse(File(result.artifact.directory, it.path).canWrite())
        }
        assertTrue(acquisition.evict(ref))
        assertEquals(ArtifactCacheState.MISSING, acquisition.cacheState(ref))
        assertFalse(result.artifact.directory.exists())
    }

    @Test
    fun manifestPathsThatEscapeStagingAreRejected() = runBlocking<Unit> {
        val cache = newCache()
        val server = server("model.onnx" to ONNX_BYTES, "../config.json" to CONFIG)

        val result = ArtifactAcquisition(cache, server.client()).acquire(request(onnxRef(server)))

        assertEquals(ArtifactAcquisitionReason.UNSAFE_PATH, result.reason)
        assertTrue(server.requests.none { it.path.startsWith("/cdn/") })
        assertFalse(File(cache.rootDirectory, ".tmp/config.json").exists())
    }

    @Test
    fun httpFailuresCarryOnlyTheNumericStatus() = runBlocking<Unit> {
        val server = server()
        server.cdnStatusOverride = 503

        val result = ArtifactAcquisition(newCache(), server.client())
            .acquire(request(onnxRef(server)))

        assertIs<ArtifactAcquisitionResult.Failed>(result)
        assertEquals(ArtifactAcquisitionReason.HTTP_STATUS, result.reason)
        assertEquals(503, result.httpStatus)
    }

    @Test
    fun logsResultsAndCacheMetadataPassSentinelScans() = runBlocking<Unit> {
        val logged = CopyOnWriteArrayList<ArtifactAcquisitionResult>()
        val logger = ArtifactAcquisitionLogger { logged += it }
        val root = temporaryFolder.newFolder("sentinel-cache")
        val server = server()
        val ref = onnxRef(server)
        val acquisition = ArtifactAcquisition(ModelCache(root), server.client(), logger)
        val results = mutableListOf<ArtifactAcquisitionResult>()
        results += acquisition.acquire(request(ref))
        results += acquisition.acquire(
            request(ref, ArtifactNetworkConsent.DENIED, ArtifactCacheVerification.FULL),
        )
        results += acquisition.acquire(request(ref.copy(digest = "sha256:" + "e".repeat(64))))
        val failing = server().also { it.cdnStatusOverride = 500 }
        results += ArtifactAcquisition(ModelCache(root), failing.client(), logger)
            .acquire(request(onnxRef(failing).copy(revision = "main")))
        val corrupt = server().also { it.advertisedSha256["model.onnx"] = "1".repeat(64) }
        results += ArtifactAcquisition(ModelCache(root), corrupt.client(), logger)
            .acquire(request(onnxRef(corrupt).copy(revision = "v1")))

        val evidence = buildString {
            logged.forEach { appendLine(it.toString()) }
            results.forEach {
                appendLine(it.toString())
                appendLine(it.artifact.toString())
            }
            root.walkTopDown().forEach { file ->
                appendLine(file.relativeTo(root).path)
                if (file.isFile && file.name == "metadata.json") {
                    appendLine(file.readText())
                }
            }
        }
        val sentinels = listOf(
            PAYLOAD_SENTINEL,
            SIGNED_QUERY_SENTINEL,
            "X-Amz",
            ERROR_BODY_SENTINEL,
            "hf_live",
            "127.0.0.1",
            server.baseUrl,
            root.absolutePath,
            root.canonicalPath,
            temporaryFolder.root.name,
        )
        sentinels.forEachIndexed { index, sentinel ->
            assertFalse(evidence.contains(sentinel), "evidence leaked sentinel #$index")
        }
        assertEquals(results.map { it.state }, logged.map { it.state })
        assertEquals(
            listOf("downloaded", "verified", "failed", "failed", "failed"),
            results.map { it.state.name.lowercase() },
        )
        assertTrue(logged.mapNotNull { it.reason?.code }.all { it.matches(Regex("[a-z_]+")) })
    }

    @Test
    fun artifactReferencesRejectUnsafeIdentifiersWithoutEchoingThem() {
        val digest = "sha256:" + "a".repeat(64)
        val invalid = listOf(
            { ArtifactRef("../escape", ArtifactFormat.ONNX, ArtifactRuntime.ONNX_RUNTIME, digest) },
            {
                val runtime = ArtifactRuntime.ONNX_RUNTIME
                ArtifactRef("OpenMed/a..b", ArtifactFormat.ONNX, runtime, digest)
            },
            { ArtifactRef(REPO, ArtifactFormat.ONNX, ArtifactRuntime.ONNX_RUNTIME, "md5:abc") },
            {
                ArtifactRef(
                    REPO,
                    ArtifactFormat.ONNX,
                    ArtifactRuntime.ONNX_RUNTIME,
                    digest,
                    revision = "refs/../main",
                )
            },
            { ArtifactBackendCapability(ArtifactRuntime.MLX, setOf(ArtifactFormat.MLX)) },
        )
        invalid.forEach { build ->
            val error = runCatching { build() }.exceptionOrNull()
            assertIs<IllegalArgumentException>(error)
            assertFalse(error.message.orEmpty().contains("escape"))
            assertFalse(error.message.orEmpty().contains("refs/"))
        }
    }

    private fun server(vararg files: Pair<String, ByteArray>): HuggingFaceFixtureServer {
        val content = if (files.isEmpty()) {
            mapOf(
                "model.onnx" to ONNX_BYTES,
                "config.json" to CONFIG,
                "README.md" to "synthetic readme".toByteArray(),
            )
        } else {
            files.toMap()
        }
        return HuggingFaceFixtureServer(REPO, content).also { servers += it }
    }

    private fun newCache(): ModelCache = ModelCache(temporaryFolder.newFolder())

    private fun assertModelBytes(artifact: AcquiredArtifact) {
        assertTrue(File(artifact.directory, "model.onnx").readBytes().contentEquals(ONNX_BYTES))
    }

    private fun onnxRef(
        server: HuggingFaceFixtureServer,
        sizeBytes: Long? = null,
    ): ArtifactRef = ArtifactRef(
        repoId = REPO,
        format = ArtifactFormat.ONNX,
        runtime = ArtifactRuntime.ONNX_RUNTIME,
        digest = server.digest(),
        revision = REVISION_SHA,
        sizeBytes = sizeBytes,
    )

    private fun request(
        ref: ArtifactRef,
        consent: ArtifactNetworkConsent = ArtifactNetworkConsent.GRANTED,
        verification: ArtifactCacheVerification = ArtifactCacheVerification.METADATA,
        backend: ArtifactBackendCapability = ONNX_BACKEND,
    ) = ArtifactAcquisitionRequest(ref, backend, consent, verification)

    private object NetworkForbiddenClient : HuggingFaceModelClient {
        override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo =
            throw AssertionError("prepared artifacts must resolve without the network")

        override fun downloadFile(
            repoId: String,
            path: String,
            revision: String,
            destination: File,
        ) {
            throw AssertionError("prepared artifacts must resolve without the network")
        }
    }

    @Suppress("DEPRECATION")
    private class DenyNetworkSecurityManager(
        private val attempts: MutableList<String>,
    ) : SecurityManager() {
        override fun checkPermission(permission: Permission?) = Unit

        override fun checkPermission(permission: Permission?, context: Any?) = Unit

        override fun checkConnect(host: String?, port: Int) {
            attempts += "connect"
            throw SecurityException("network access is forbidden for prepared artifacts")
        }

        override fun checkConnect(host: String?, port: Int, context: Any?) {
            checkConnect(host, port)
        }
    }

    private companion object {
        const val REPO = "OpenMed/tiny-synthetic-onnx"
        const val CONCURRENT_CALLERS = 8
        const val PAYLOAD_SENTINEL = "SENTINEL-PAYLOAD-Jane-Roe-555-0100"
        val ONNX_BYTES = syntheticOnnx(256 * 1024, PAYLOAD_SENTINEL)
        val CONFIG = """{"hidden_size":8,"note":"synthetic"}""".toByteArray()
        val ONNX_BACKEND = ArtifactBackendCapability.onnxRuntime()
    }
}
