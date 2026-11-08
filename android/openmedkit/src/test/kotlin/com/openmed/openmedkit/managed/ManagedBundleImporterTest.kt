package com.openmed.openmedkit.managed

import com.openmed.openmedkit.bundle.Json
import com.openmed.openmedkit.bundle.PortableBundle
import com.openmed.openmedkit.bundle.PortableBundleContext
import com.openmed.openmedkit.bundle.PortableBundleTarget
import com.openmed.openmedkit.bundle.PortableBundleTrustPolicy
import com.openmed.openmedkit.cache.BundleCacheCrash
import com.openmed.openmedkit.cache.BundleCacheOptions
import com.openmed.openmedkit.cache.BundleGenerationCache
import com.openmed.openmedkit.cache.BundleQuarantineEntry
import com.openmed.openmedkit.cache.ModelCacheClock
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.charset.StandardCharsets
import java.security.Permission
import java.util.Base64
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import javax.xml.parsers.DocumentBuilderFactory
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** Replays `tests/fixtures/managed_import/vectors.json` against [ManagedBundleImporter]. */
class ManagedBundleImporterTest {
    @get:Rule
    val temporary = TemporaryFolder()

    private val module: File by lazy {
        File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
        )
    }
    private val repoRoot: File by lazy { requireNotNull(module.absoluteFile.parentFile?.parentFile) }

    private val vectors: Json.Obj by lazy {
        Json.parse(File(repoRoot, "tests/fixtures/managed_import/vectors.json").readBytes()) as Json.Obj
    }

    private fun Json?.obj(): Json.Obj = this as Json.Obj
    private fun Json?.str(): String = (this as Json.Str).value
    private fun Json?.long(): Long = (this as Json.Num).value
    private fun Json?.items(): List<Json> = (this as Json.Arr).items
    private fun Json?.optionalString(): String? = if (this == null || this == Json.Null) null else str()
    private fun Json?.optionalLong(): Long? = if (this == null || this == Json.Null) null else long()

    private val bundleId: String get() = vectors["identities"].obj()["bundle_id"].str()
    private fun bundle(alias: String): Json.Obj = vectors["bundles"].obj()[alias].obj()
    private fun generation(alias: String?): String? = alias?.let { bundle(it)["generation"].str() }
    private fun archive(alias: String): ByteArray =
        Base64.getDecoder().decode(vectors["archives"].obj()[alias].obj()["base64"].str())

    private fun plain(value: Json?): Any? = when (value) {
        null, Json.Null -> null
        is Json.Str -> value.value
        is Json.Num -> value.value
        is Json.Bool -> value.value
        else -> error("unsupported managed value")
    }

    private fun config(alias: String): MutableMap<String, Any?> =
        vectors["configs"].obj()[alias].obj().members.associate { (key, value) -> key to plain(value) }.toMutableMap()

    private fun policy(name: String): PortableBundleTrustPolicy =
        PortableBundleTrustPolicy.fromJson(String(Json.encode(vectors["trust_policies"].obj()[name]!!), StandardCharsets.UTF_8))

    private fun target(variant: String): PortableBundleTarget {
        val value = vectors["targets"].obj()["android"].obj()[variant].obj()
        return PortableBundleTarget(
            platform = value["platform"].str(),
            sdkName = value["sdk_name"].str(),
            sdkVersion = value["sdk_version"].str(),
            runtimes = value["runtimes"].items().map { it.str() },
            osVersion = value["os_version"].optionalString(),
            policyVersion = value["policy_version"].optionalLong(),
            memoryBytes = value["memory_bytes"].optionalLong(),
        )
    }

    private fun limits(value: Json.Obj) = ManagedImportLimits(
        maxArchiveBytes = value["max_archive_bytes"].long(),
        maxEntries = value["max_entries"].long().toInt(),
        maxTotalBytes = value["max_total_bytes"].long(),
        maxCompressionRatio = value["max_compression_ratio"].long(),
    )

    private fun writeBundleDirectory(alias: String, directory: File) {
        val value = bundle(alias)
        directory.mkdirs()
        File(directory, PortableBundle.MANIFEST_FILENAME).writeText(value["manifest"].str())
        File(directory, PortableBundle.ENVELOPE_FILENAME).writeText(value["envelope"].str())
        for (entry in value["files"].items()) {
            val file = File(directory, entry.obj()["path"].str())
            file.parentFile?.mkdirs()
            file.writeText(entry.obj()["text"].str())
        }
    }

    private fun fixed(values: Map<String, Any?>?) = ManagedConfigurationSource { values }

    private class Clock : ModelCacheClock {
        private val now = AtomicLong(1_000_000L)
        override fun nowEpochMillis(): Long = now.incrementAndGet()
    }

    private inner class Harness(val root: File, private val chunkBytes: Int) {
        val inbox = File(root, "inbox").also { it.mkdirs() }
        private val clock = Clock()
        @Volatile
        var available: Long? = null
        lateinit var cache: BundleGenerationCache
        lateinit var importer: ManagedBundleImporter
        private var restarts = 0

        init {
            start()
        }

        fun start() {
            restarts++
            cache = BundleGenerationCache(
                File(root, "cache"),
                BundleCacheOptions(
                    quotaBytes = 1L shl 40,
                    chunkBytes = chunkBytes,
                    clock = clock,
                    availableBytes = { available ?: Long.MAX_VALUE },
                    ownerId = "owner-$restarts",
                ),
            )
            importer = ManagedBundleImporter(cache, File(root, "state"))
        }
    }

    private fun defaultContext(variant: String = "default") =
        PortableBundleContext(now = "2026-11-07T00:00:00Z", target = target(variant))

    // region Shared vectors

    @Test
    fun sharedVectors() {
        val scenarios = vectors["scenarios"].items()
        assertTrue(scenarios.size >= 20)
        for (scenario in scenarios) replay(scenario.obj())
    }

    private fun replay(scenario: Json.Obj) {
        val id = scenario["id"].str()
        val harness = Harness(temporary.newFolder(), vectors["chunk_bytes"].long().toInt())
        val target = target(scenario["target"].str())
        val scenarioLimits = limits(scenario["limits"].obj())
        for ((index, item) in scenario["steps"].items().withIndex()) {
            val step = item.obj()
            val label = "$id#$index"
            val expect = step["expect"] as? Json.Obj ?: Json.Obj(emptyList())
            when (step["op"].str()) {
                "import" -> runImport(step, expect, harness, target, scenarioLimits, label)
                "restart" -> harness.start()
                "recover" -> assertEquals(
                    expect["spool_purged"].long().toInt(), harness.importer.recover().spoolPurged, label,
                )
                "confirm" -> harness.importer.confirm(bundleId)
                "state" -> {
                    val snapshot = harness.cache.snapshot(bundleId)
                    assertEquals(generation(expect["active"].optionalString()), snapshot.active, "$label active")
                    assertEquals(generation(expect["last_green"].optionalString()), snapshot.lastGreen, "$label green")
                    assertEquals(generation(expect["rollback"].optionalString()), snapshot.rollback, "$label rollback")
                    val quarantined = expect["quarantined"].items().map {
                        BundleQuarantineEntry(generation(it.obj()["generation"].str())!!, it.obj()["reason"].str())
                    }
                    assertEquals(quarantined, snapshot.quarantined, "$label quarantine")
                    assertEquals(
                        expect["sequence_floor"].optionalLong(), harness.importer.sequenceFloor(bundleId), "$label floor",
                    )
                    expect["receipts"].optionalLong()?.let {
                        assertEquals(it.toInt(), harness.importer.receipts().size, "$label receipts")
                    }
                }
                else -> error("unknown op in $label")
            }
        }
    }

    private fun runImport(
        step: Json.Obj,
        expect: Json.Obj,
        harness: Harness,
        target: PortableBundleTarget,
        scenarioLimits: ManagedImportLimits,
        label: String,
    ) {
        harness.available = step["storage_available_bytes"].optionalLong()
        val values = step["config"].optionalString()?.let { config(it) }
        var handoff: ManagedBundleHandoff? = null
        when (step["handoff"].str()) {
            "stream" -> {
                val bytes = archive(step["archive"].str())
                handoff = ManagedBundleHandoff.Stream { ByteArrayInputStream(bytes) }
            }
            "directory" -> {
                val name = "${step["bundle"].str()}-${UUID.randomUUID()}"
                writeBundleDirectory(step["bundle"].str(), File(harness.inbox, name))
                values?.put(ManagedBundleConfiguration.LOCATION_KEY, name)
            }
            else -> {
                val name = "${step["archive"].str()}-${UUID.randomUUID()}.zip"
                File(harness.inbox, name).writeBytes(archive(step["archive"].str()))
                values?.put(ManagedBundleConfiguration.LOCATION_KEY, name)
            }
        }
        val counts = HashMap<String, Int>()
        val cancelled = AtomicBoolean(false)
        val crashAt = step["crash_at"] as? Json.Obj
        val cancelAt = step["cancel_at"] as? Json.Obj
        val hook = { name: String ->
            val seen = (counts[name] ?: 0) + 1
            counts[name] = seen
            if (crashAt != null && crashAt["checkpoint"].str() == name && crashAt["occurrence"].long().toInt() == seen) {
                throw BundleCacheCrash(name)
            }
            if (cancelAt != null && cancelAt["checkpoint"].str() == name &&
                cancelAt["occurrence"].long().toInt() == seen
            ) {
                cancelled.set(true)
            }
        }
        harness.cache.checkpointHook = hook
        harness.importer.checkpointHook = hook
        val health = step["health"].optionalString()
        val run = {
            harness.importer.importManaged(
                source = fixed(values),
                context = PortableBundleContext(now = step["now"].str(), target = target),
                approvedRoots = listOf(harness.inbox),
                handoff = handoff,
                limits = scenarioLimits,
                activationCheck = health?.let { value -> { _ -> value == "pass" } },
                isCancelled = { cancelled.get() },
            )
        }
        try {
            if ((expect["crashed"] as? Json.Bool)?.value == true) {
                assertFailsWith<BundleCacheCrash>(label) { run() }
                return
            }
            val receipt = run()
            assertEquals(expect["outcome"].str(), receipt.outcome.code, "$label outcome")
            assertEquals(expect["reason"].optionalString(), receipt.reason?.code, "$label reason")
            assertEquals(expect["bundle_reason"].optionalString(), receipt.bundleReason?.code, "$label bundle reason")
            assertEquals(generation(expect["generation"].optionalString()), receipt.generation, "$label generation")
            assertEquals(
                generation(expect["previous_active"].optionalString()), receipt.previousActive, "$label previous",
            )
            assertEquals(generation(expect["active"].optionalString()), receipt.active, "$label active")
            assertEquals(generation(expect["last_green"].optionalString()), receipt.lastGreen, "$label last green")
            assertEquals((expect["resumed"] as Json.Bool).value, receipt.resumed, "$label resumed")
            assertEquals(expect["files_verified"].long().toInt(), receipt.filesVerified, "$label files")
            assertValueFree(receipt, harness.root, label)
        } finally {
            harness.cache.checkpointHook = null
            harness.importer.checkpointHook = null
        }
    }

    private fun assertValueFree(receipt: ManagedImportReceipt, root: File, label: String) {
        val text = receipt.toJson() + receipt.toString()
        assertTrue(text.contains("\"schema\":\"openmed.managed_import.receipt\""), label)
        for (needle in listOf(root.path, root.canonicalPath, "file:", "content:", ".zip", "synthetic")) {
            assertFalse(text.contains(needle), "$label leaks $needle")
        }
    }

    // endregion

    // region Handoffs and configuration

    private fun defaultConfiguration(location: String? = null): Map<String, Any?> =
        config("default").also { if (location != null) it[ManagedBundleConfiguration.LOCATION_KEY] = location }

    @Test
    fun configuredLocationsStayLocalAndApproved() {
        val root = temporary.newFolder()
        val inbox = File(root, "inbox").also { it.mkdirs() }
        val outside = File(root, "outside").also { it.mkdirs() }
        File(inbox, "bundle.zip").writeBytes(archive("v1"))
        File(outside, "bundle.zip").writeBytes(archive("v1"))
        val cache = BundleGenerationCache(File(root, "cache"))
        val importer = ManagedBundleImporter(cache, File(root, "state"))
        fun run(location: String?) = importer.importManaged(
            source = fixed(defaultConfiguration(location)),
            context = defaultContext(),
            approvedRoots = listOf(inbox),
            approvedAuthorities = setOf("org.example.emm.files"),
        )
        for (remote in listOf("https://mdm.example.invalid/b.zip", "http://10.0.0.1/b.zip", "ftp://host/b.zip")) {
            val receipt = run(remote)
            assertEquals(ManagedImportOutcome.UNSUPPORTED, receipt.outcome, remote)
            assertEquals(ManagedImportReason.HANDOFF_UNSUPPORTED, receipt.reason, remote)
        }
        assertEquals(ManagedImportReason.LOCATION_UNAPPROVED, run(File(outside, "bundle.zip").toURI().toString()).reason)
        assertEquals(ManagedImportReason.LOCATION_UNAPPROVED, run("../outside/bundle.zip").reason)
        java.nio.file.Files.createSymbolicLink(File(inbox, "link.zip").toPath(), File(outside, "bundle.zip").toPath())
        assertEquals(ManagedImportReason.LOCATION_UNAPPROVED, run("link.zip").reason)
        assertEquals(ManagedImportReason.LOCATION_UNAPPROVED, run("content://org.example.other/bundle.zip").reason)
        // An approved authority without a content opener is not a supported handoff.
        assertEquals(ManagedImportReason.HANDOFF_UNSUPPORTED, run("content://org.example.emm.files/bundle.zip").reason)
        val pending = run("later.zip")
        assertEquals(ManagedImportOutcome.DEFERRED, pending.outcome)
        assertEquals(ManagedImportReason.HANDOFF_PENDING, pending.reason)
        assertEquals(ManagedImportReason.HANDOFF_PENDING, run(null).reason)
        assertEquals(ManagedImportOutcome.ACTIVATED, run(File(inbox, "bundle.zip").toURI().toString()).outcome)
        assertEquals(generation("v1"), cache.snapshot(bundleId).active)
    }

    @Test
    fun contentUriHandoffsUseTheInjectedOpener() {
        val root = temporary.newFolder()
        val delivered = HashMap<String, ByteArray>()
        val opener = ManagedContentStreamOpener { uri ->
            ByteArrayInputStream(delivered[uri] ?: throw java.io.FileNotFoundException("not_delivered"))
        }
        val cache = BundleGenerationCache(File(root, "cache"))
        val importer = ManagedBundleImporter(cache, File(root, "state"), opener)
        val uri = "content://org.example.emm.files/bundles/model.zip"
        fun run() = importer.importManaged(
            source = fixed(defaultConfiguration(uri)),
            context = defaultContext(),
            approvedAuthorities = setOf("org.example.emm.files"),
        )
        val pending = run()
        assertEquals(ManagedImportOutcome.DEFERRED, pending.outcome)
        assertEquals(ManagedHandoffKind.CONTENT_URI, pending.handoff)
        delivered[uri] = archive("v2")
        val receipt = run()
        assertEquals(ManagedImportOutcome.ACTIVATED, receipt.outcome)
        assertEquals(ManagedHandoffKind.CONTENT_URI, receipt.handoff)
        assertEquals(ManagedImportOutcome.ALREADY_PRESENT, run().outcome)
        assertEquals(0, File(root, "state/spool").listFiles()?.size ?: 0)
    }

    @Test
    fun appKeyPinsConstrainTheDeliveredTrustPolicy() {
        val root = temporary.newFolder()
        val file = File(root, "bundle.zip").also { it.writeBytes(archive("v1")) }
        val importer = ManagedBundleImporter(BundleGenerationCache(File(root, "cache")), File(root, "state"))
        val pins = vectors["identities"].obj()["key_pins"].obj()
        val configuration = ManagedBundleConfiguration.fromManagedValues(defaultConfiguration())
        assertEquals(pins["openmed-test-root-a"].str(), ManagedBundleConfiguration.pin(configuration.trustPolicy.keys[0]))
        val refused = importer.importBundle(
            ManagedImportRequest(
                ManagedBundleHandoff.LocalFile(file), configuration, defaultContext(),
                keyPins = setOf(pins["openmed-test-root-b"].str()),
            ),
        )
        assertEquals(ManagedImportReason.TRUST_ANCHOR_UNPINNED, refused.reason)
        assertNull(refused.generation)
        val accepted = importer.importBundle(
            ManagedImportRequest(
                ManagedBundleHandoff.LocalFile(file), configuration, defaultContext(),
                keyPins = setOf(pins["openmed-test-root-a"].str()),
            ),
        )
        assertEquals(ManagedImportOutcome.ACTIVATED, accepted.outcome)
        assertEquals(BundleGenerationCache.trustDigest(policy("default")), accepted.trustDigest)
    }

    @Test
    fun platformZipWriterArchivesWithDataDescriptorsImport() {
        val root = temporary.newFolder()
        val value = bundle("v2")
        val bytes = ByteArrayOutputStream()
        ZipOutputStream(bytes).use { zip ->
            fun put(name: String, text: String) {
                zip.putNextEntry(ZipEntry(name))
                zip.write(text.toByteArray(StandardCharsets.UTF_8))
                zip.closeEntry()
            }
            put("bundle/${PortableBundle.MANIFEST_FILENAME}", value["manifest"].str())
            put("bundle/${PortableBundle.ENVELOPE_FILENAME}", value["envelope"].str())
            for (entry in value["files"].items()) put("bundle/" + entry.obj()["path"].str(), entry.obj()["text"].str())
        }
        val file = File(root, "platform.zip").also { it.writeBytes(bytes.toByteArray()) }
        val importer = ManagedBundleImporter(
            BundleGenerationCache(File(root, "cache"), BundleCacheOptions(chunkBytes = 16)), File(root, "state"),
        )
        val receipt = importer.importBundle(
            ManagedImportRequest(
                ManagedBundleHandoff.LocalFile(file),
                ManagedBundleConfiguration.fromManagedValues(defaultConfiguration()),
                defaultContext(),
            ),
        )
        assertEquals(ManagedImportOutcome.ACTIVATED, receipt.outcome, receipt.reason?.code)
        assertEquals(generation("v2"), receipt.generation)
    }

    @Test
    fun restrictionsXmlDeclaresEveryConfigurationKey() {
        val xml = File(module, "src/main/res/xml/openmed_managed_bundle_restrictions.xml")
        val document = DocumentBuilderFactory.newInstance().apply { isNamespaceAware = true }.newDocumentBuilder()
            .parse(xml)
        val nodes = document.getElementsByTagName("restriction")
        val keys = (0 until nodes.length).map {
            nodes.item(it).attributes.getNamedItemNS("http://schemas.android.com/apk/res/android", "key").nodeValue
        }
        assertEquals(ManagedBundleConfiguration.KEYS, keys)
        val shared = vectors["config_keys"].obj().members.map { it.second.str() }.toSet()
        assertEquals(shared, ManagedBundleConfiguration.KEYS.toSet())
    }

    // endregion

    // region Concurrency and isolation

    @Test
    fun concurrentReadersSeeOnlyCompleteGenerations() {
        val root = temporary.newFolder()
        val cache = BundleGenerationCache(File(root, "cache"), BundleCacheOptions(chunkBytes = 16))
        val importer = ManagedBundleImporter(cache, File(root, "state"))
        val configuration = ManagedBundleConfiguration.fromManagedValues(defaultConfiguration())
        for (alias in listOf("v1", "v2")) File(root, "$alias.zip").writeBytes(archive(alias))
        fun run(alias: String) = importer.importBundle(
            ManagedImportRequest(ManagedBundleHandoff.LocalFile(File(root, "$alias.zip")), configuration, defaultContext()),
        )
        assertEquals(ManagedImportOutcome.ACTIVATED, run("v1").outcome)
        val expected = listOf("v1", "v2").associate { alias ->
            generation(alias)!! to bundle(alias)["files"].items().associate { it.obj()["path"].str() to it.obj()["text"].str() }
        }
        val held = cache.open(bundleId)
        val stop = AtomicBoolean(false)
        val failures = CopyOnWriteArrayList<String>()
        val reader = Thread {
            while (!stop.get()) {
                val lease = try {
                    cache.open(bundleId)
                } catch (error: Exception) {
                    continue
                }
                lease.use {
                    for ((path, text) in expected[it.generation].orEmpty()) {
                        if (File(it.payloadDirectory, path).readText() != text) failures += it.generation
                    }
                }
            }
        }
        reader.start()
        val pool = Executors.newFixedThreadPool(3)
        val results = (0 until 3).map { pool.submit<ManagedImportReceipt> { run("v2") } }.map { it.get(60, TimeUnit.SECONDS) }
        pool.shutdown()
        stop.set(true)
        reader.join(10_000)
        assertEquals(emptyList(), failures.toList())
        assertEquals(1, results.count { it.outcome == ManagedImportOutcome.ACTIVATED })
        assertEquals(2, results.count { it.outcome == ManagedImportOutcome.ALREADY_PRESENT })
        held.use {
            for ((path, text) in expected[generation("v1")]!!) assertEquals(text, File(it.payloadDirectory, path).readText())
        }
        assertEquals(generation("v2"), cache.snapshot(bundleId).active)
    }

    @Suppress("DEPRECATION")
    @Test
    fun importsNeverTouchTheNetwork() {
        val root = temporary.newFolder()
        val inbox = File(root, "inbox").also { it.mkdirs() }
        File(inbox, "v2.zip").writeBytes(archive("v2"))
        writeBundleDirectory("v1", File(inbox, "v1"))
        val cache = BundleGenerationCache(File(root, "cache"))
        val importer = ManagedBundleImporter(cache, File(root, "state"))
        val attempts = CopyOnWriteArrayList<String>()
        val previous = System.getSecurityManager()
        System.setSecurityManager(DenyNetworkSecurityManager(attempts))
        try {
            fun run(location: String, handoff: ManagedBundleHandoff? = null) = importer.importManaged(
                source = fixed(defaultConfiguration(location)),
                context = defaultContext(),
                approvedRoots = listOf(inbox),
                handoff = handoff,
            )
            assertEquals(ManagedImportOutcome.ACTIVATED, run("v1").outcome)
            assertEquals(ManagedImportOutcome.ACTIVATED, run("v2.zip").outcome)
            val bytes = archive("v3")
            assertEquals(
                ManagedImportOutcome.ACTIVATED, run("ignored", ManagedBundleHandoff.Stream { ByteArrayInputStream(bytes) }).outcome,
            )
            assertEquals(ManagedImportReason.HANDOFF_UNSUPPORTED, run("https://example.invalid/v4.zip").reason)
            importer.recover()
        } finally {
            System.setSecurityManager(previous)
        }
        assertTrue(attempts.isEmpty(), "managed import attempted network I/O")

        // Source scan: the managed import, archive, cache, and verifier code
        // reference no networking API at all.
        val sources = File(module, "src/main/kotlin/com/openmed/openmedkit")
        for (path in listOf(
            "managed/ManagedBundleImporter.kt", "managed/BundleArchive.kt", "managed/ManagedHandoffAdapters.kt",
            "cache/BundleGenerationCache.kt", "bundle/PortableBundle.kt",
        )) {
            val text = File(sources, path).readText()
            for (api in listOf("java.net.Socket", "HttpURLConnection", "URLConnection", "okhttp", "java.net.URL", "openConnection")) {
                assertFalse(text.contains(api), "$path references $api")
            }
        }
    }

    private class DenyNetworkSecurityManager(private val attempts: MutableList<String>) : SecurityManager() {
        override fun checkPermission(permission: Permission?) = Unit

        override fun checkPermission(permission: Permission?, context: Any?) = Unit

        override fun checkConnect(host: String?, port: Int) {
            attempts += "connect"
            throw SecurityException("network access is forbidden during managed import")
        }

        override fun checkConnect(host: String?, port: Int, context: Any?) = checkConnect(host, port)
    }

    // endregion
}
