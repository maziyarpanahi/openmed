package com.openmed.openmedkit.download

import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.security.MessageDigest

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

    @Test
    fun reproducibilityHashMatchesEverySharedPythonAndSwiftVector() {
        val cases = reproHashCases()
        assertTrue(cases.size >= 11)
        for (case in cases) {
            val id = case.text("id")!!
            val stamp = case.text("last_modified") ?: case.text("created_at")
            val expected = case.text("expected_hash")
            assertEquals(id, expected, vectorHash(case, stamp))
            assertEquals(id, expected, vectorHash(case, case.text("released")))
            assertEquals(id, expected, vectorHash(case, stamp, case.siblings().reversed()))
        }
    }

    @Test
    fun sharedVectorsCoverTheCanonicalizationEdges() {
        val cases = reproHashCases().associateBy { it.text("id")!! }
        val values = cases.values
        assertTrue(values.all { it.text("repo_id")!!.startsWith("OpenMed/Synthetic-Repro-") })
        assertTrue(values.any { it.text("sha") == null })
        assertTrue(values.any { it.text("released") == null })
        assertTrue(values.any { it.siblings() != codePointSorted(it.siblings()) })
        assertTrue(values.any { !printable(it.text("repo_id")!!) })
        assertTrue(values.any { case -> case.siblings().any { !printable(it) } })
        // Kotlin's String order compares UTF-16 code units, not Python's code points.
        assertTrue(values.any { it.siblings().sorted() != codePointSorted(it.siblings()) })
        val base = cases.getValue("unsorted-siblings-with-commit-and-date")
        val changed = cases.getValue("changed-sibling-list-changes-the-hash")
        val released = base.text("released")
        assertTrue(changed.siblings().containsAll(base.siblings()))
        assertEquals(changed.text("expected_hash"), vectorHash(base, released, base.siblings() + "extra.bin"))
        assertFalse(vectorHash(base, released, base.siblings().drop(1)) == base.text("expected_hash"))
    }

    @Test
    fun onlyNamesOutsidePrintableAsciiChangedFromTheEarlierAndroidEncoding() {
        var changed = 0
        for (case in reproHashCases()) {
            val names = case.siblings() + case.text("repo_id")!!
            val legacy = legacyHash(case)
            if (names.all(::printable)) {
                assertEquals(case.text("id"), case.text("expected_hash"), legacy)
            } else {
                changed += 1
                assertFalse(case.text("id"), legacy == case.text("expected_hash"))
            }
        }
        assertTrue(changed >= 4)
    }

    // Printable ASCII plus the short JSON escapes, which both encodings share.
    private fun printable(value: String): Boolean = value.all { it in ' '..'~' || it in "\b\u000C\n\r\t" }

    private fun codePointSorted(names: List<String>): List<String> = names.sortedWith { left, right ->
        val a = left.codePoints().toArray()
        val b = right.codePoints().toArray()
        val index = (0 until minOf(a.size, b.size)).firstOrNull { a[it] != b[it] }
        if (index == null) a.size.compareTo(b.size) else a[index].compareTo(b[index])
    }

    /** The encoding Android used before this fix: raw non-ASCII text and UTF-16 order. */
    private fun legacyHash(case: JsonObject): String {
        fun quote(value: String?): String {
            if (value == null) return "null"
            val escaped = value.map { character ->
                when (character) {
                    '\\' -> "\\\\"
                    '"' -> "\\\""
                    '\b' -> "\\b"
                    '\u000C' -> "\\f"
                    '\n' -> "\\n"
                    '\r' -> "\\r"
                    '\t' -> "\\t"
                    else -> character.toString()
                }
            }
            return "\"" + escaped.joinToString("") + "\""
        }
        val payload = "{\"released\":" + quote(case.text("released")) +
            ",\"repo_id\":" + quote(case.text("repo_id")) +
            ",\"sha\":" + quote(case.text("sha")) +
            ",\"siblings\":[" + case.siblings().sorted().joinToString(",") { quote(it) } + "]}"
        return "sha256:" + MessageDigest.getInstance("SHA-256")
            .digest(payload.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
    }

    private fun reproHashCases(): List<JsonObject> {
        val document = javaClass.classLoader!!.getResourceAsStream("model_repro_hash_v1.json")!!
            .bufferedReader(Charsets.UTF_8)
            .use { Json.parseToJsonElement(it.readText()).jsonObject }
        assertEquals("openmed.model_repro_hash_vectors", document.text("schema"))
        assertEquals("1", document.getValue("version").jsonPrimitive.content)
        return document.getValue("cases").jsonArray.map { it.jsonObject }
    }

    private fun JsonObject.text(key: String): String? = getValue(key).jsonPrimitive.contentOrNull

    private fun JsonObject.siblings(): List<String> =
        getValue("siblings").jsonArray.map { it.jsonPrimitive.content }

    private fun vectorHash(case: JsonObject, released: String?, siblings: List<String> = case.siblings()): String =
        ModelIntegrity.reproducibilityHash(
            repoId = case.text("repo_id")!!,
            revisionSha = case.text("sha"),
            released = released,
            siblings = siblings,
        )

    private fun catalogEntry(
        repoId: String,
        reproducibilityHash: String,
    ): ModelCatalogEntry =
        ModelCatalogEntry(
            repoId = repoId,
            formats = listOf("onnx"),
            tier = "tiny",
            paramCount = 1,
            languages = listOf("en"),
            license = "apache-2.0",
            reproducibilityHash = reproducibilityHash,
        )

    private fun integrityHash(repoId: String, siblings: List<String>): String =
        ModelIntegrity.reproducibilityHash(
            repoId = repoId,
            revisionSha = FakeHuggingFaceModelClient.REVISION_SHA,
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
