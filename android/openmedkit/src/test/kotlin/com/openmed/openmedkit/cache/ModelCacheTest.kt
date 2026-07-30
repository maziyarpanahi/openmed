package com.openmed.openmedkit.cache

import com.openmed.openmedkit.catalog.ModelCatalogEntry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ModelCacheTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun tracksAvailabilitySizeAndEviction() {
        val cache = ModelCache(temporaryFolder.newFolder("cache"))
        val entry = catalogEntry("OpenMed/tiny-onnx")

        cache.storeReadyModel(
            repoId = entry.repoId,
            sourceDirectory = stagedModel("first", "model.onnx" to "weights"),
            expectedIntegrityHash = entry.reproducibilityHash,
            actualIntegrityHash = entry.reproducibilityHash,
            revisionSha = "abc123",
        )

        assertTrue(cache.isAvailable(entry))
        assertTrue(cache.isAvailable(entry.repoId))
        assertEquals(7L, cache.totalSizeBytes())
        assertEquals(7L, cache.sizeBytes(entry.repoId))

        assertTrue(cache.evict(entry.repoId))
        assertFalse(cache.isAvailable(entry))
        assertEquals(0L, cache.totalSizeBytes())
    }

    @Test
    fun evictsLeastRecentlyUsedModelWhenBudgetIsExceeded() {
        val clock = MutableClock()
        val cache = ModelCache(
            rootDirectory = temporaryFolder.newFolder("cache"),
            cacheBudgetBytes = 12L,
            clock = clock,
        )
        val first = catalogEntry("OpenMed/first")
        val second = catalogEntry("OpenMed/second")
        val third = catalogEntry("OpenMed/third")

        cache.storeReadyModel(
            repoId = first.repoId,
            sourceDirectory = stagedModel("first", "model.onnx" to "12345"),
            expectedIntegrityHash = first.reproducibilityHash,
            actualIntegrityHash = first.reproducibilityHash,
            revisionSha = "aaa",
        )
        clock.advance()
        cache.storeReadyModel(
            repoId = second.repoId,
            sourceDirectory = stagedModel("second", "model.onnx" to "12345"),
            expectedIntegrityHash = second.reproducibilityHash,
            actualIntegrityHash = second.reproducibilityHash,
            revisionSha = "bbb",
        )
        clock.advance()
        assertTrue(cache.isAvailable(first))
        clock.advance()
        cache.storeReadyModel(
            repoId = third.repoId,
            sourceDirectory = stagedModel("third", "model.onnx" to "12345"),
            expectedIntegrityHash = third.reproducibilityHash,
            actualIntegrityHash = third.reproducibilityHash,
            revisionSha = "ccc",
        )

        assertTrue(cache.isAvailable(first))
        assertFalse(cache.isAvailable(second))
        assertTrue(cache.isAvailable(third))
        assertEquals(10L, cache.totalSizeBytes())
    }

    private fun catalogEntry(repoId: String): ModelCatalogEntry =
        ModelCatalogEntry(
            repoId = repoId,
            formats = listOf("onnx"),
            tier = "tiny",
            paramCount = 1,
            languages = listOf("en"),
            license = "apache-2.0",
            reproducibilityHash =
                "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        )

    private fun stagedModel(
        name: String,
        vararg files: Pair<String, String>,
    ): File {
        val directory = temporaryFolder.newFolder("staged-$name")
        files.forEach { (path, content) ->
            val file = File(directory, path)
            file.parentFile?.mkdirs()
            file.writeText(content)
        }
        return directory
    }

    private class MutableClock : ModelCacheClock {
        private var current = 1_000L

        override fun nowEpochMillis(): Long = current

        fun advance() {
            current += 1_000L
        }
    }
}
