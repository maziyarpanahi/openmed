package com.openmed.openmedkit.download

import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
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
