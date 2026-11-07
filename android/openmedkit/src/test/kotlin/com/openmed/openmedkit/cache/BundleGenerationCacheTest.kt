package com.openmed.openmedkit.cache

import com.openmed.openmedkit.bundle.Json
import com.openmed.openmedkit.bundle.PortableBundle
import com.openmed.openmedkit.bundle.PortableBundleContext
import com.openmed.openmedkit.bundle.PortableBundleTrustPolicy
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.security.Permission
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class BundleGenerationCacheTest {
    @get:Rule
    val temporary = TemporaryFolder()

    private val vectors: Json.Obj by lazy {
        val module = File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
        )
        val repoRoot = requireNotNull(module.absoluteFile.parentFile?.parentFile)
        Json.parse(File(repoRoot, "tests/fixtures/bundle_cache/vectors.json").readBytes()) as Json.Obj
    }

    private fun Json?.obj(): Json.Obj = this as Json.Obj
    private fun Json?.str(): String = (this as Json.Str).value
    private fun Json?.long(): Long = (this as Json.Num).value
    private fun Json?.bool(): Boolean = (this as Json.Bool).value
    private fun Json?.items(): List<Json> = (this as Json.Arr).items
    private fun Json?.optionalString(): String? = if (this == null || this == Json.Null) null else str()
    private fun Json?.optionalLong(): Long? = if (this == null || this == Json.Null) null else long()

    private val bundles: Json.Obj get() = vectors["bundles"].obj()
    private val bundleId: String get() = vectors["identities"].obj()["bundle_id"].str()
    private fun bundle(alias: String): Json.Obj = bundles[alias].obj()
    private fun generation(alias: String): String = bundle(alias)["generation"].str()

    private fun files(alias: String): Map<String, ByteArray> =
        bundle(alias)["files"].items().associate { entry ->
            entry.obj()["path"].str() to entry.obj()["text"].str().toByteArray(StandardCharsets.UTF_8)
        }

    private fun policy(name: String): PortableBundleTrustPolicy =
        PortableBundleTrustPolicy.fromJson(
            String(Json.encode(vectors["trust_policies"].obj()[name]!!), StandardCharsets.UTF_8),
        )

    private fun context() = PortableBundleContext(now = vectors["context"].obj()["now"].str())

    private class FakeClock(var now: Long) : ModelCacheClock {
        override fun nowEpochMillis(): Long = now
    }

    /** Serves synthetic bytes by range, with scripted faults; never touches the network. */
    private class FakeRangeSource(
        private val files: Map<String, ByteArray>,
        override val identity: String,
        private val failAfter: Long? = null,
        private val cancelAfter: Long? = null,
        private val corrupt: Pair<String, Int>? = null,
        private val extra: Pair<String, Int>? = null,
        private val ignoreRange: Boolean = false,
        private val onRead: () -> Unit = {},
    ) : BundleRangeSource {
        @Volatile
        var delivered = 0L
        val cancelled: Boolean get() = cancelAfter != null && delivered >= cancelAfter

        override fun open(path: String, offset: Long): BundleRangeResponse {
            var data = files[path] ?: throw IOException("missing")
            corrupt?.takeIf { it.first == path }?.let { (_, at) ->
                data = data.copyOf().also { it[at] = (it[at].toInt() xor 0xff).toByte() }
            }
            extra?.takeIf { it.first == path }?.let { (_, count) -> data += ByteArray(count) { 'X'.code.toByte() } }
            val start = if (ignoreRange) 0L else offset
            val bytes = data
            return BundleRangeResponse(
                start,
                object : InputStream() {
                    var position = start.toInt()

                    override fun read(): Int {
                        val one = ByteArray(1)
                        return if (read(one, 0, 1) <= 0) -1 else one[0].toInt() and 0xff
                    }

                    override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
                        onRead()
                        if (position >= bytes.size) return -1
                        var allowed = length.toLong()
                        if (failAfter != null) {
                            val left = failAfter - delivered
                            if (left <= 0) throw IOException("interrupted")
                            allowed = minOf(allowed, left)
                        }
                        val count = minOf(allowed, (bytes.size - position).toLong()).toInt()
                        System.arraycopy(bytes, position, buffer, offset, count)
                        position += count
                        delivered += count
                        return count
                    }
                },
            )
        }
    }

    private fun request(
        alias: String,
        source: BundleRangeSource,
        policyName: String = "default",
        activate: Boolean = true,
        confirm: Boolean = true,
        cancelled: () -> Boolean = { false },
    ) = BundleAcquireRequest(
        manifest = bundle(alias)["manifest"].str().toByteArray(StandardCharsets.UTF_8),
        envelope = bundle(alias)["envelope"].str().toByteArray(StandardCharsets.UTF_8),
        source = source,
        trustPolicy = policy(policyName),
        context = context(),
        activate = activate,
        confirm = confirm,
        isCancelled = cancelled,
    )

    // region Vector replay

    private inner class Replay(private val scenario: Json.Obj) {
        private val id = scenario["id"].str()
        private val options = scenario["options"].obj()
        private val root = temporary.newFolder()
        private val clock = FakeClock(vectors["clock"].obj()["start_ms"].long())
        private val step = vectors["clock"].obj()["step_ms"].long()
        private var owner = 1
        private var available = Long.MAX_VALUE / 4
        private val leases = mutableMapOf<String, BundleLease>()
        private var cache = newCache()

        private fun newCache() = BundleGenerationCache(
            root,
            BundleCacheOptions(
                quotaBytes = options["quota_bytes"].long(),
                chunkBytes = options["chunk_bytes"].long().toInt(),
                lockTimeoutMillis = 2_000L,
                leaseTtlMillis = options["lease_ttl_ms"].long(),
                stagingMaxAgeMillis = options["staging_max_age_ms"].long(),
                clock = clock,
                availableBytes = { available },
                ownerId = "owner-${owner++}",
            ),
        )

        fun run() {
            for ((index, raw) in scenario["steps"].items().withIndex()) {
                clock.now += step
                val item = raw.obj()
                val label = "$id step $index ${item["op"].str()}"
                val actual = execute(item)
                if (actual != null) {
                    assertEquals(
                        String(Json.encode(resolve(item["expect"]!!)), StandardCharsets.UTF_8),
                        String(Json.encode(actual), StandardCharsets.UTF_8),
                        label,
                    )
                }
            }
        }

        private fun installCrash(raw: Json?) {
            val crash = raw?.takeUnless { it == Json.Null }?.obj()
            if (crash == null) {
                cache.checkpointHook = null
                return
            }
            var seen = 0L
            val name = crash["checkpoint"].str()
            val occurrence = crash["occurrence"].long()
            cache.checkpointHook = { checkpoint: String ->
                if (checkpoint == name) {
                    seen += 1
                    if (seen == occurrence) throw BundleCacheCrash(checkpoint)
                }
            }
        }

        private fun crashed(block: () -> Unit): Json {
            assertFailsWith<BundleCacheCrash> { block() }
            return obj("crashed" to Json.Bool(true))
        }

        private fun failure(error: BundleCacheException) = obj(
            "ok" to Json.Bool(false),
            "reason" to Json.Str(error.reason.code),
            "bundle_reason" to (error.bundleReason?.let { Json.Str(it.code) } ?: Json.Null),
        )

        private fun attempt(block: () -> Json): Json = try {
            block()
        } catch (error: BundleCacheException) {
            failure(error)
        }

        private fun execute(item: Json.Obj): Json? = when (item["op"].str()) {
            "acquire" -> acquire(item)
            "restart" -> {
                clock.now += item["advance_ms"].long()
                leases.clear()
                cache = newCache()
                null
            }
            "recover" -> obj(
                "actions" to Json.Arr(
                    cache.recover().actions.map { action ->
                        obj(
                            "action" to Json.Str(action.action.code),
                            "generation" to (action.generation?.let { Json.Str(it) } ?: Json.Null),
                        )
                    },
                ),
            )
            "activate" -> attempt {
                cache.activate(bundleId, generation(item["generation"].str()), item["confirm"].bool())
                obj("ok" to Json.Bool(true))
            }
            "confirm" -> attempt {
                cache.confirm(bundleId)
                obj("ok" to Json.Bool(true))
            }
            "rollback" -> attempt {
                cache.rollback(bundleId, item["quarantine"].bool())
                obj("ok" to Json.Bool(true))
            }
            "pin", "unpin" -> attempt {
                val target = generation(item["generation"].str())
                if (item["op"].str() == "pin") cache.pin(bundleId, target) else cache.unpin(bundleId, target)
                obj("ok" to Json.Bool(true))
            }
            "lease" -> attempt {
                val lease = cache.lease(bundleId, item["generation"].optionalString()?.let { generation(it) })
                leases[item["lease"].str()] = lease
                obj("ok" to Json.Bool(true), "generation" to Json.Str(lease.generation))
            }
            "release" -> attempt {
                cache.release(leases.getValue(item["lease"].str()))
                obj("ok" to Json.Bool(true))
            }
            "open" -> attempt {
                val verification = when (item["verification"].str()) {
                    "full" -> BundleOpenVerification.FULL
                    else -> BundleOpenVerification.METADATA
                }
                val lease = cache.open(bundleId, verification)
                leases[item["lease"].str()] = lease
                obj(
                    "ok" to Json.Bool(true),
                    "generation" to Json.Str(lease.generation),
                    "files_checked" to Json.Num(lease.filesChecked.toLong()),
                )
            }
            "evict" -> {
                installCrash(item["crash_at"])
                if (item["expect"].obj()["crashed"] != null) {
                    crashed { cache.evict(item["additional_bytes"].long()) }
                } else {
                    val decision = cache.evict(item["additional_bytes"].long())
                    obj(
                        "evicted" to Json.Arr(decision.evicted.map { Json.Str(it) }),
                        "satisfied" to Json.Bool(decision.satisfied),
                        "used_bytes" to Json.Num(decision.usedBytes),
                    )
                }
            }
            "tamper" -> {
                tamper(item)
                null
            }
            "advance" -> {
                clock.now += item["ms"].long()
                null
            }
            "snapshot" -> snapshot()
            else -> error("unknown op in $id")
        }

        private fun acquire(item: Json.Obj): Json {
            val described = item["source"].obj()
            val corrupt = described["corrupt"].takeUnless { it == Json.Null }?.obj()
            val extra = described["extra_bytes"].takeUnless { it == Json.Null }?.obj()
            val source = FakeRangeSource(
                files(described["serve"].str()),
                described["identity"].str(),
                failAfter = described["fail_after_bytes"].optionalLong(),
                cancelAfter = described["cancel_after_bytes"].optionalLong(),
                corrupt = corrupt?.let { it["path"].str() to it["offset"].long().toInt() },
                extra = extra?.let { it["path"].str() to it["count"].long().toInt() },
                ignoreRange = described["ignore_range"].bool(),
            )
            val acquireRequest = request(
                item["bundle"].str(), source, item["policy"].str(),
                item["activate"].bool(), item["confirm"].bool(),
            ) { source.cancelled }
            installCrash(item["crash_at"])
            item["storage_available_bytes"].optionalLong()?.let { available = it }
            try {
                if (item["expect"].obj()["crashed"] != null) return crashed { cache.acquire(acquireRequest) }
                val result = cache.acquire(acquireRequest)
                if (!result.ok) {
                    return obj(
                        "ok" to Json.Bool(false),
                        "reason" to Json.Str(result.reason!!.code),
                        "bundle_reason" to (result.bundleReason?.let { Json.Str(it.code) } ?: Json.Null),
                    )
                }
                return obj(
                    "ok" to Json.Bool(true),
                    "generation" to Json.Str(result.generation!!),
                    "state" to Json.Str(result.state!!.code),
                    "from_cache" to Json.Bool(result.fromCache),
                    "resumed" to Json.Bool(result.resumed),
                    "resume_reason" to (result.resumeReason?.let { Json.Str(it.code) } ?: Json.Null),
                    "bytes_transferred" to Json.Num(result.bytesTransferred),
                    "verified_bytes" to Json.Num(result.verifiedBytes),
                    "files_verified" to Json.Num(result.filesVerified.toLong()),
                    "evicted" to Json.Arr(result.evicted.map { Json.Str(it) }),
                )
            } finally {
                available = Long.MAX_VALUE / 4
                cache.checkpointHook = null
            }
        }

        private fun tamper(item: Json.Obj) {
            val area = if (item["area"].str() == "ready") "generations" else "staging"
            val file = File(
                root,
                "${BundleGenerationCache.bundleKey(bundleId)}/$area/${generation(item["generation"].str())}/" +
                    "payload/${item["path"].str()}",
            )
            assertTrue(file.isFile, "$id: tamper target exists")
            val offset = item["offset"].long()
            RandomAccessFile(file, "rw").use { output ->
                when (item["mode"].str()) {
                    "truncate" -> output.setLength(offset)
                    else -> {
                        output.seek(offset)
                        val value = output.read()
                        output.seek(offset)
                        output.write(value xor 0xff)
                    }
                }
            }
        }

        private fun snapshot(): Json {
            val snapshot = cache.snapshot(bundleId)
            fun ref(value: String?): Json = value?.let { Json.Str(it) } ?: Json.Null
            return obj(
                "pointer" to obj(
                    "active" to ref(snapshot.active),
                    "last_green" to ref(snapshot.lastGreen),
                    "rollback" to ref(snapshot.rollback),
                ),
                "generations" to Json.Arr(
                    snapshot.generations.map { entry ->
                        obj(
                            "generation" to Json.Str(entry.generation),
                            "state" to Json.Str(entry.state.code),
                            "total_bytes" to Json.Num(entry.totalBytes),
                            "verified_bytes" to Json.Num(entry.verifiedBytes),
                            "protections" to Json.Arr(entry.protections.map { Json.Str(it.code) }),
                        )
                    },
                ),
                "quarantined" to Json.Arr(
                    snapshot.quarantined.map {
                        obj("generation" to Json.Str(it.generation), "reason" to Json.Str(it.reason))
                    },
                ),
                "used_bytes" to Json.Num(snapshot.usedBytes),
            )
        }
    }

    /** Replaces generation aliases in an expectation with generation ids. */
    private fun resolve(value: Json, key: String? = null): Json = when (value) {
        is Json.Obj -> Json.Obj(value.members.map { (name, member) -> name to resolve(member, name) })
        is Json.Arr -> Json.Arr(value.items.map { resolve(it, key) })
        is Json.Str -> if (key in ALIAS_KEYS) Json.Str(generation(value.value)) else value
        else -> value
    }

    private fun obj(vararg members: Pair<String, Json>) = Json.Obj(members.toList())

    @Test
    fun everySharedScenarioReplaysExactly() {
        val scenarios = vectors["scenarios"].items()
        assertTrue(scenarios.size >= 30)
        for (scenario in scenarios) Replay(scenario.obj()).run()
    }

    @Test
    fun identitiesMatchTheSharedVectors() {
        val identities = vectors["identities"].obj()
        assertEquals(identities["bundle_key"].str(), BundleGenerationCache.bundleKey(bundleId))
        for ((name, digest) in identities["trust_digests"].obj().members) {
            assertEquals(digest.str(), BundleGenerationCache.trustDigest(policy(name)), name)
        }
        for (alias in listOf("v1", "v2", "v3", "v4")) {
            val manifest = bundle(alias)["manifest"].str().toByteArray(StandardCharsets.UTF_8)
            assertEquals(bundle(alias)["content_id"].str(), PortableBundle.contentId(manifest))
            assertEquals(generation(alias), BundleGenerationCache.generationId(PortableBundle.contentId(manifest)))
        }
        assertEquals(Json.Bool(true), vectors["synthetic_only"])
        val reasons = vectors["reason_codes"].items().map { it.str() }
        assertEquals(reasons, BundleCacheReason.values().map { it.code })
        assertEquals(vectors["states"].items().map { it.str() }, BundleGenerationState.values().map { it.code })
        assertEquals(
            vectors["recovery_actions"].items().map { it.str() },
            BundleRecoveryAction.values().map { it.code },
        )
        assertEquals(
            vectors["resume_reasons"].items().map { it.str() },
            BundleResumeReason.values().map { it.code },
        )
    }

    // endregion

    private fun newCache(root: File, ownerId: String = "owner", lockTimeoutMillis: Long = 5_000L) =
        BundleGenerationCache(
            root,
            BundleCacheOptions(chunkBytes = 16, lockTimeoutMillis = lockTimeoutMillis, ownerId = ownerId),
        )

    @Test
    fun concurrentCallersConvergeOnOneVerifiedGenerationWithoutPartialReads() {
        val root = temporary.newFolder()
        val caches = listOf(newCache(root, "process-a"), newCache(root, "process-b"), newCache(root, "process-a"))
        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(4)
        val observed = CopyOnWriteArrayList<String>()
        val done = AtomicBoolean(false)
        val reader = pool.submit {
            start.await()
            while (!done.get()) {
                try {
                    caches[1].open(bundleId, BundleOpenVerification.FULL).use { lease ->
                        val sizes = files("v1").all { (path, bytes) ->
                            File(lease.payloadDirectory, path).length() == bytes.size.toLong()
                        }
                        observed += if (sizes) "complete" else "partial"
                    }
                } catch (error: BundleCacheException) {
                    observed += error.reason.code
                }
            }
        }
        val acquisitions = caches.map { cache ->
            pool.submit<BundleAcquireResult> {
                start.await()
                cache.acquire(request("v1", FakeRangeSource(files("v1"), "source-a") { Thread.yield() }))
            }
        }
        start.countDown()
        val results = acquisitions.map { it.get(30, TimeUnit.SECONDS) }
        done.set(true)
        reader.get(30, TimeUnit.SECONDS)
        pool.shutdownNow()

        assertTrue(results.all { it.ok }, results.toString())
        assertEquals(1, results.count { !it.fromCache })
        assertEquals(setOf(generation("v1")), results.map { it.generation }.toSet())
        assertFalse("partial" in observed)
        assertTrue(observed.all { it == "complete" || it == BundleCacheReason.NO_ACTIVE_GENERATION.code })
        val snapshot = caches[0].snapshot(bundleId)
        assertEquals(listOf(generation("v1")), snapshot.generations.map { it.generation })
        assertEquals(BundleGenerationState.ACTIVE, snapshot.generations.single().state)
    }

    @Test
    fun lockWaitIsBoundedAndTyped() {
        val root = temporary.newFolder()
        val holder = newCache(root, "holder")
        val waiter = newCache(root, "waiter", lockTimeoutMillis = 50L)
        val entered = CountDownLatch(1)
        val proceed = CountDownLatch(1)
        val pool = Executors.newSingleThreadExecutor()
        val held = pool.submit<BundleAcquireResult> {
            holder.acquire(
                request(
                    "v1",
                    FakeRangeSource(files("v1"), "source-a") {
                        entered.countDown()
                        proceed.await()
                    },
                ),
            )
        }
        assertTrue(entered.await(10, TimeUnit.SECONDS))
        val refused = waiter.acquire(request("v1", FakeRangeSource(files("v1"), "source-a")))
        assertEquals(BundleCacheReason.LOCK_TIMEOUT, refused.reason)
        assertFailsWith<BundleCacheException> { waiter.snapshot(bundleId) }.also {
            assertEquals(BundleCacheReason.LOCK_TIMEOUT, it.reason)
        }
        proceed.countDown()
        assertTrue(held.get(30, TimeUnit.SECONDS).ok)
        pool.shutdown()
        assertTrue(waiter.acquire(request("v1", FakeRangeSource(files("v1"), "source-a"))).fromCache)
    }

    @Test
    fun preparedBundleImportsAndOpensWithZeroNetworkAccess() {
        val prepared = temporary.newFolder()
        for ((path, bytes) in files("v1")) {
            File(prepared, path).apply { parentFile?.mkdirs() }.writeBytes(bytes)
        }
        val root = temporary.newFolder()
        val attempts = CopyOnWriteArrayList<String>()
        val previous = System.getSecurityManager()
        System.setSecurityManager(DenyNetwork(attempts))
        try {
            val imported = newCache(root, "importer").acquire(
                request("v1", DirectoryBundleRangeSource(prepared), activate = false),
            )
            assertTrue(imported.ok)
            assertEquals(BundleGenerationState.EVICTABLE, imported.state)
            val restarted = newCache(root, "restarted")
            assertTrue(restarted.recover().actions.isEmpty())
            restarted.activate(bundleId, generation("v1"))
            restarted.open(bundleId, BundleOpenVerification.FULL).use { lease ->
                assertEquals(files("v1").size, lease.filesChecked)
                assertEquals(generation("v1"), lease.generation)
            }
        } finally {
            System.setSecurityManager(previous)
        }
        assertTrue(attempts.isEmpty(), attempts.toString())
    }

    @Test
    fun metadataResultsAndErrorsCarryNoPathsOrSourceIdentities() {
        val root = temporary.newFolder("private-root-sentinel")
        val identity = "https://models.invalid/private?token=sentinel-credential"
        val cache = newCache(root)
        val interrupted = cache.acquire(request("v1", FakeRangeSource(files("v1"), identity, failAfter = 100)))
        assertEquals(BundleCacheReason.TRANSFER_INTERRUPTED, interrupted.reason)
        val result = cache.acquire(request("v1", FakeRangeSource(files("v1"), identity)))
        assertTrue(result.ok)
        val lease = cache.open(bundleId)
        val surfaces = listOf(
            result.toString(), interrupted.toString(), lease.toString(), cache.toString(),
            cache.snapshot(bundleId).toString(), cache.recover().toString(),
            assertFailsWith<BundleCacheException> { cache.rollback(bundleId) }.message.orEmpty(),
        )
        val sentinels = listOf(root.path, "private-root-sentinel", "sentinel-credential", "models.invalid", "token")
        for (surface in surfaces) {
            for (sentinel in sentinels) assertFalse(surface.contains(sentinel), "leak of $sentinel in $surface")
        }
        val records = root.walkTopDown().filter { it.isFile && it.name.endsWith(".json") && "payload" !in it.path }
            .toList()
        assertTrue(records.size >= 3)
        for (record in records) {
            val text = record.readText()
            for (sentinel in sentinels) assertFalse(text.contains(sentinel), "leak of $sentinel in ${record.name}")
        }
        lease.close()
    }

    @Test
    fun legacyModelCacheExposesTheGenerationCacheWithoutMixingEntries() {
        val root = temporary.newFolder()
        val legacy = ModelCache(root, cacheBudgetBytes = 1L shl 20)
        val generations = legacy.bundleGenerations(BundleCacheOptions(chunkBytes = 16))
        assertTrue(generations.acquire(request("v1", FakeRangeSource(files("v1"), "source-a"))).ok)
        assertEquals(0L, legacy.totalSizeBytes())
        assertFalse(legacy.isAvailable("openmed.synthetic.tiny-pii"))
        assertTrue(File(root, BundleGenerationCache.DIRECTORY_NAME).isDirectory)
    }

    private class DenyNetwork(private val attempts: MutableList<String>) : SecurityManager() {
        override fun checkPermission(permission: Permission?) = Unit

        override fun checkPermission(permission: Permission?, context: Any?) = Unit

        override fun checkConnect(host: String?, port: Int) {
            attempts += "$host:$port"
            throw SecurityException("network denied")
        }

        override fun checkConnect(host: String?, port: Int, context: Any?) = checkConnect(host, port)
    }

    private companion object {
        val ALIAS_KEYS = setOf("generation", "active", "last_green", "rollback", "evicted")
    }
}
