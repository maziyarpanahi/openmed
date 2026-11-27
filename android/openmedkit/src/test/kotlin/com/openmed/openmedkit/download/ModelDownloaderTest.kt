package com.openmed.openmedkit.download

import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import java.io.File
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ModelDownloaderTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun downloadsFilesAndMarksModelReadyAfterIntegrityChecks() {
        val repoId = "OpenMed/tiny-onnx"
        val files = mapOf(
            "model.onnx" to "weights",
            "config.json" to "{\"hidden_size\":1}",
            "README.md" to "not required",
        )
        val entry = catalogEntry(repoId, integrityHash(repoId, files.keys.toList()))
        val client = FakeHuggingFaceModelClient(repoId, files)
        val cache = ModelCache(temporaryFolder.newFolder("cache"))

        val result = ModelDownloader(cache, client).download(entry)

        assertFalse(result.fromCache)
        assertEquals(2, result.filesDownloaded)
        assertEquals(entry.reproducibilityHash, result.integrityHash)
        assertTrue(cache.isAvailable(entry))
        assertEquals(1, client.modelInfoCalls)
        assertEquals(listOf("config.json", "model.onnx"), client.downloadedPaths.sorted())
    }

    @Test
    fun rejectsChecksumFailuresWithoutMarkingTheModelReady() {
        val repoId = "OpenMed/tiny-onnx"
        val files = mapOf("model.onnx" to "weights")
        val entry = catalogEntry(repoId, integrityHash(repoId, files.keys.toList()))
        val client = FakeHuggingFaceModelClient(
            repoId = repoId,
            files = files,
            sha256Overrides = mapOf("model.onnx" to "0".repeat(64)),
        )
        val cache = ModelCache(temporaryFolder.newFolder("cache"))

        try {
            ModelDownloader(cache, client).download(entry)
        } catch (_: ModelIntegrityException) {
            assertFalse(cache.isAvailable(entry))
            return
        }

        throw AssertionError("Expected checksum failure")
    }

    @Test
    fun servesOfflineCacheHitWithoutNetworkCalls() {
        val repoId = "OpenMed/tiny-onnx"
        val files = mapOf("model.onnx" to "weights")
        val entry = catalogEntry(repoId, integrityHash(repoId, files.keys.toList()))
        val cache = ModelCache(temporaryFolder.newFolder("cache"))
        val firstClient = FakeHuggingFaceModelClient(repoId, files)

        ModelDownloader(cache, firstClient).download(entry)
        val offlineClient = object : HuggingFaceModelClient {
            override fun fetchModelInfo(
                repoId: String,
                revision: String?,
            ): HuggingFaceModelInfo =
                throw AssertionError("network should not be used for cache hits")

            override fun downloadFile(
                repoId: String,
                path: String,
                revision: String,
                destination: File,
            ) {
                throw AssertionError("network should not be used for cache hits")
            }
        }

        val result = ModelDownloader(cache, offlineClient).download(entry)

        assertTrue(result.fromCache)
        assertEquals(0, result.filesDownloaded)
        assertTrue(cache.isAvailable(entry))
    }

    @Test
    fun keepsLogsAndMetadataFreeOfModelContent() {
        val repoId = "OpenMed/tiny-onnx"
        val modelContent = "patient Jane Doe phone 555-0100"
        val files = mapOf("model.onnx" to modelContent)
        val entry = catalogEntry(repoId, integrityHash(repoId, files.keys.toList()))
        val events = mutableListOf<ModelDownloadLogEvent>()
        val cache = ModelCache(temporaryFolder.newFolder("cache"))

        ModelDownloader(
            cache = cache,
            client = FakeHuggingFaceModelClient(repoId, files),
            logger = ModelDownloadLogger { event -> events += event },
        ).download(entry)

        val metadata = File(cache.modelDirectory(repoId), "metadata.json").readText()
        val logs = events.joinToString("\n")
        assertFalse(metadata.contains("Jane Doe"))
        assertFalse(metadata.contains("555-0100"))
        assertFalse(logs.contains("Jane Doe"))
        assertFalse(logs.contains("555-0100"))
        assertTrue(metadata.contains(repoId))
        assertTrue(events.all { it.repoId == repoId })
    }

    // Lifecycle events and byte progress (tests/fixtures/parity/model_lifecycle_v1.json).

    @Test
    fun replaysTheSharedLifecycleSequences() {
        val fixture = Json.parseToJsonElement(resource("model_lifecycle_v1.json")).jsonObject
        assertEquals(ModelLifecycleEvent.SCHEMA_VERSION, fixture.getValue("event_schema_version").jsonPrimitive.content)
        val fields = fixture.getValue("event_fields").jsonArray.map { it.jsonPrimitive.content }
        val sequences = fixture.getValue("sequences").jsonArray.map { it.jsonObject }
        val replayed = mutableSetOf<String>()
        for (sequence in sequences) {
            val id = sequence.getValue("id").jsonPrimitive.content
            if (id in NOT_DOWNLOADER_SEQUENCES) {
                continue
            }
            val scenario = sequence.getValue("scenario").jsonObject
            val events = mutableListOf<ModelLifecycleEvent>()
            val outcome = runScenario(id, scenario, ModelLifecycleSink { events += it })
            val expected = sequence.getValue("events").jsonArray.map { it.jsonObject }

            assertEquals(expected.map { it.getValue("event").jsonPrimitive.content }, events.map { it.event }, id)
            assertEquals(expected, events.map { Json.parseToJsonElement(it.toJson()).jsonObject }, id)
            events.forEach {
                assertEquals(fields, Json.parseToJsonElement(it.toJson()).jsonObject.keys.toList(), id)
                assertSchemaValid(it)
            }
            assertEquals(sequence.getValue("outcome").jsonPrimitive.content, outcome, id)
            replayed += id
        }
        assertEquals(sequences.map { it.getValue("id").jsonPrimitive.content }.toSet() - NOT_DOWNLOADER_SEQUENCES, replayed)
        assertTrue(replayed.size >= 6)
    }

    @Test
    fun progressIsMonotonicPerFileAndEndsAtTheFileSize() {
        val files = mapOf("config.json" to "c".repeat(10), "model.onnx" to "w".repeat(25_000))
        val entry = catalogEntry(REPO_ID, integrityHash(REPO_ID, files.keys.toList(), COMMIT), listOf("onnx"))
        val progress = mutableMapOf<Int, MutableList<Long>>()
        val sizes = mutableMapOf<Int, Long?>()
        val client = ScenarioClient(REPO_ID, COMMIT, files, corrupt = mapOf("model.onnx" to 1), chunkBytes = 4096)

        ModelDownloader(
            ModelCache(temporaryFolder.newFolder("cache")),
            client,
            lifecycleSink = ModelLifecycleSink.NONE,
            progressListener = ModelDownloadProgressListener { index, count, bytes, size ->
                assertEquals(2, count)
                progress.getOrPut(index) { mutableListOf() } += bytes
                sizes[index] = size
            },
        ).download(entry)

        val ordered = files.keys.sorted()
        for ((index, path) in ordered.withIndex()) {
            val values = progress.getValue(index)
            val size = files.getValue(path).length.toLong()
            assertEquals(0L, values.first())
            assertEquals(size, values.last())
            assertEquals(size, sizes[index])
            assertTrue(values.zipWithNext().all { (a, b) -> b > a }, path)
        }
        // The large file reported chunks, including during its corrupt first attempt.
        assertTrue(progress.getValue(1).size > 3)
    }

    @Test
    fun progressNeverPassesThePublishedSizeAfterAnOversizedAttempt() {
        val files = mapOf("model.onnx" to "w".repeat(10_000))
        val entry = catalogEntry(REPO_ID, integrityHash(REPO_ID, files.keys.toList(), COMMIT), listOf("onnx"))
        val values = mutableListOf<Long>()
        val events = mutableListOf<ModelLifecycleEvent>()
        val client = ScenarioClient(REPO_ID, COMMIT, files, oversize = mapOf("model.onnx" to 1), chunkBytes = 4096)

        ModelDownloader(
            ModelCache(temporaryFolder.newFolder("cache")),
            client,
            lifecycleSink = ModelLifecycleSink { events += it },
            progressListener = ModelDownloadProgressListener { _, _, bytes, _ -> values += bytes },
        ).download(entry)

        assertEquals(
            listOf("requested", "fetching", "integrity_failed", "fetching", "verified", "ready"),
            events.map { it.event },
        )
        assertEquals("size_mismatch", events[2].reason)
        assertTrue(values.all { it <= 10_000L }, values.toString())
        assertEquals(10_000L, values.last())
    }

    @Test
    fun reproducibilityHashMismatchEmitsIntegrityFailedAndWritesNoReadyDirectory() {
        val files = mapOf("model.onnx" to "weights")
        val entry = catalogEntry(REPO_ID, "sha256:" + "0".repeat(64), listOf("onnx"))
        val cache = ModelCache(temporaryFolder.newFolder("cache"))
        val client = ScenarioClient(REPO_ID, COMMIT, files)
        val events = mutableListOf<ModelLifecycleEvent>()

        assertFailsWith<ModelIntegrityException> {
            ModelDownloader(cache, client, lifecycleSink = ModelLifecycleSink { events += it }).download(entry)
        }

        assertEquals(listOf("requested", "integrity_failed", "failed"), events.map { it.event })
        assertEquals(entry.reproducibilityHash, events[1].digest)
        assertEquals("digest_mismatch", events[1].reason)
        assertEquals("integrity_unverifiable", events[2].reason)
        assertEquals(emptyList(), client.downloaded)
        assertFalse(cache.isAvailable(entry))
        assertFalse(cache.modelDirectory(REPO_ID).exists())
        events.forEach(::assertSchemaValid)
    }

    @Test
    fun eventsCarryNoPathsUrlsOrFileNamesAndASinkCannotChangeTheOutcome() {
        val path = "onnx/Jane_Roe_private.onnx"
        val files = mapOf(path to "weights", "config.json" to "{}")
        val entry = catalogEntry(REPO_ID, integrityHash(REPO_ID, files.keys.toList(), COMMIT), listOf("onnx"))
        val bodies = mutableListOf<String>()
        val cache = ModelCache(temporaryFolder.newFolder("cache"))

        val result = ModelDownloader(
            cache,
            ScenarioClient(REPO_ID, COMMIT, files),
            lifecycleSink = ModelLifecycleSink { event ->
                bodies += event.toJson()
                throw IllegalStateException("sink failure")
            },
        ).download(entry)

        assertTrue(cache.isAvailable(entry))
        assertEquals(2, result.filesDownloaded)
        assertEquals(listOf("requested", "fetching", "verified", "fetching", "verified", "ready"), bodies.map {
            Json.parseToJsonElement(it).jsonObject.getValue("event").jsonPrimitive.content
        })
        for (body in bodies) {
            for (forbidden in listOf("Jane", ".onnx", "config.json", "/resolve", "http", cache.modelDirectory(REPO_ID).path)) {
                assertFalse(forbidden in body, body)
            }
        }
        // Identifiers outside the schema shapes are recorded as null.
        val unsafe = ModelLifecycleEvent("failed", "https://example.invalid/x", "main", null, null, "abc", "Bad Reason")
        assertEquals(listOf(null, null, null, null), listOf(unsafe.repoId, unsafe.revisionSha, unsafe.digest, unsafe.reason))
    }

    @Test
    fun legacyStatusesMapOntoTheLifecycleVocabulary() {
        assertEquals(
            mapOf("CACHE_HIT" to "ready", "DOWNLOAD_STARTED" to "requested", "DOWNLOAD_COMPLETE" to "ready"),
            ModelDownloadStatus.values().associate { it.name to it.lifecycleEvent },
        )
    }

    private fun runScenario(id: String, scenario: JsonObject, sink: ModelLifecycleSink): String {
        val repoId = scenario.getValue("repo_id").jsonPrimitive.content
        val commit = scenario.getValue("commit").jsonPrimitive.content
        val revision = scenario["revision"]?.takeIf { it is JsonPrimitive && it.isString }?.jsonPrimitive?.content
        // Android downloads runnable files only and events carry no file names,
        // so a non-ONNX weight file is served under an ONNX name.
        val specs = scenario.getValue("files").jsonArray.map { it.jsonObject }
        val files = specs.associate {
            val name = it.getValue("filename").jsonPrimitive.content
            (if (name.endsWith(".safetensors")) name.replace(".safetensors", ".onnx") else name) to
                it.getValue("content").jsonPrimitive.content
        }
        val corrupt = specs.associate {
            val name = it.getValue("filename").jsonPrimitive.content.replace(".safetensors", ".onnx")
            name to (it["corrupt_downloads"]?.jsonPrimitive?.content?.toInt() ?: 0)
        }
        val cache = ModelCache(temporaryFolder.newFolder(id))
        val mode = scenario.getValue("mode").jsonPrimitive.content
        if (mode == "offline_hit") {
            val cached = mapOf("config.json" to "synthetic-config-v1")
            val entry = catalogEntry(repoId, integrityHash(repoId, cached.keys.toList(), commit), listOf("onnx"))
            ModelDownloader(cache, ScenarioClient(repoId, commit, cached)).download(entry)
            ModelDownloader(cache, ScenarioClient(repoId, commit, emptyMap()), lifecycleSink = sink).download(entry)
            return "ok"
        }
        val entry = catalogEntry(repoId, integrityHash(repoId, files.keys.toList(), commit), listOf("onnx"))
        return try {
            ModelDownloader(cache, ScenarioClient(repoId, commit, files, corrupt), lifecycleSink = sink)
                .download(entry, revision)
            assertTrue(cache.isAvailable(entry), id)
            "ok"
        } catch (error: ModelIntegrityException) {
            assertFalse(cache.isAvailable(entry), id)
            "download_integrity_error"
        }
    }

    /** Checks one event body against `model_lifecycle_event.schema.json`. */
    private fun assertSchemaValid(event: ModelLifecycleEvent) {
        val schema = Json.parseToJsonElement(
            File(requireNotNull(System.getProperty("openmedkit.moduleDirectory")))
                .resolve("../../openmed/core/schemas/json/model_lifecycle_event.schema.json").readText(),
        ).jsonObject
        val properties = schema.getValue("properties").jsonObject
        val body = Json.parseToJsonElement(event.toJson()).jsonObject
        fun pattern(field: String): Regex = Regex(
            properties.getValue(field).jsonObject.getValue("oneOf").jsonArray[1].jsonObject
                .getValue("pattern").jsonPrimitive.content,
        )
        fun text(field: String): String? = body.getValue(field).takeUnless { it is JsonNull }?.jsonPrimitive?.content

        assertEquals(schema.getValue("required").jsonArray.map { it.jsonPrimitive.content }.toSet(), body.keys)
        assertEquals(properties.getValue("schema_version").jsonObject.getValue("const").jsonPrimitive.content, text("schema_version"))
        assertTrue(
            properties.getValue("event").jsonObject.getValue("enum").jsonArray.any { it.jsonPrimitive.content == event.event },
        )
        for (field in listOf("repo_id", "revision_sha", "digest", "reason")) {
            text(field)?.let { assertTrue(pattern(field).containsMatchIn(it), "$field=$it") }
        }
        for (field in listOf("file_count", "byte_count")) {
            text(field)?.let { assertTrue(it.toLong() >= 0, field) }
        }
        if (event.event in setOf("integrity_failed", "rolled_back", "failed")) {
            assertTrue(event.reason != null, event.event)
        }
        if (event.event in setOf("fetching", "verified", "integrity_failed")) {
            assertEquals(1, event.fileCount)
            assertTrue(event.byteCount != null && event.digest != null, event.event)
        }
    }

    private fun resource(name: String): String =
        requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) { "missing test resource $name" }
            .use { String(it.readBytes(), Charsets.UTF_8) }

    /** Fake Hub that serves fixed files, corrupting the first downloads of some. */
    private class ScenarioClient(
        private val repoId: String,
        private val commit: String,
        private val files: Map<String, String>,
        corrupt: Map<String, Int> = emptyMap(),
        private val chunkBytes: Int = 0,
        oversize: Map<String, Int> = emptyMap(),
    ) : HuggingFaceModelClient {
        private val corruptLeft = corrupt.toMutableMap()
        private val oversizeLeft = oversize.toMutableMap()
        val downloaded = mutableListOf<String>()

        override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo {
            check(repoId == this.repoId)
            return HuggingFaceModelInfo(
                repoId = repoId,
                revisionSha = commit,
                released = RELEASED,
                files = files.map { (path, content) ->
                    HuggingFaceModelFile(path, content.length.toLong(), sha256Hex(content))
                },
            )
        }

        override fun downloadFile(repoId: String, path: String, revision: String, destination: File) {
            throw AssertionError("ModelDownloader uses the transfer overload")
        }

        override fun downloadFile(
            repoId: String,
            path: String,
            revision: String,
            destination: File,
            transfer: ModelFileTransfer,
        ): ModelFileTransferOutcome {
            downloaded += path
            val content = files.getValue(path)
            val left = corruptLeft[path] ?: 0
            // A corrupt copy keeps the size, so only the digest check catches it.
            val bytes = (if (left > 0) "#" + content.drop(1) else content).toByteArray()
            corruptLeft[path] = left - 1
            val extra = oversizeLeft[path] ?: 0
            oversizeLeft[path] = extra - 1
            // An oversized copy appends bytes, so the size check catches it.
            val body = if (extra > 0) bytes + ByteArray(bytes.size) else bytes
            destination.outputStream().use { output ->
                val step = if (chunkBytes > 0) chunkBytes else body.size.coerceAtLeast(1)
                var written = 0
                while (written < body.size) {
                    val count = minOf(step, body.size - written)
                    output.write(body, written, count)
                    written += count
                    transfer.onBytes?.invoke(written.toLong())
                }
            }
            return ModelFileTransferOutcome(0L, body.size.toLong())
        }

        private fun sha256Hex(content: String): String =
            MessageDigest.getInstance("SHA-256").digest(content.toByteArray()).joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val REPO_ID = "OpenMed/synthetic-lifecycle-model"
        const val COMMIT = "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1"
        const val RELEASED = "2026-01-01"

        /**
         * Sequences outside `ModelDownloader`: Android verifies files without a
         * published digest by size instead of refusing them, has no offline mode
         * on this API, and eviction and activation are not download events.
         */
        val NOT_DOWNLOADER_SEQUENCES = setOf(
            "prefetch_metadata_missing_digest",
            "offline_cache_miss",
            "evict_least_recently_used",
            "evict_artifact_already_absent",
            "activation_then_rollback",
        )
    }

    private fun catalogEntry(
        repoId: String,
        reproducibilityHash: String,
        formats: List<String> = listOf("onnx"),
    ): ModelCatalogEntry =
        ModelCatalogEntry(
            repoId = repoId,
            formats = formats,
            tier = "tiny",
            paramCount = 1,
            languages = listOf("en"),
            license = "apache-2.0",
            reproducibilityHash = reproducibilityHash,
        )

    private fun integrityHash(
        repoId: String,
        siblings: List<String>,
        revisionSha: String = FakeHuggingFaceModelClient.REVISION_SHA,
    ): String =
        ModelIntegrity.reproducibilityHash(
            repoId = repoId,
            revisionSha = revisionSha,
            released = FakeHuggingFaceModelClient.RELEASED,
            siblings = siblings,
        )

    private class FakeHuggingFaceModelClient(
        private val repoId: String,
        private val files: Map<String, String>,
        private val sha256Overrides: Map<String, String> = emptyMap(),
    ) : HuggingFaceModelClient {
        var modelInfoCalls = 0
        val downloadedPaths = mutableListOf<String>()

        override fun fetchModelInfo(
            repoId: String,
            revision: String?,
        ): HuggingFaceModelInfo {
            check(repoId == this.repoId)
            modelInfoCalls += 1
            return HuggingFaceModelInfo(
                repoId = repoId,
                revisionSha = REVISION_SHA,
                released = RELEASED,
                files = files.map { (path, content) ->
                    HuggingFaceModelFile(
                        path = path,
                        sizeBytes = content.toByteArray().size.toLong(),
                        sha256 = sha256Overrides[path] ?: sha256(content),
                    )
                },
            )
        }

        override fun downloadFile(
            repoId: String,
            path: String,
            revision: String,
            destination: File,
        ) {
            check(repoId == this.repoId)
            check(revision == REVISION_SHA)
            downloadedPaths += path
            destination.writeText(files.getValue(path))
        }

        private fun sha256(content: String): String =
            MessageDigest.getInstance("SHA-256")
                .digest(content.toByteArray())
                .joinToString("") { "%02x".format(it) }

        companion object {
            const val REVISION_SHA = "abc123"
            const val RELEASED = "2026-01-01"
        }
    }
}
