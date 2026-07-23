package com.openmed.openmedkit

import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelInfo
import com.openmed.openmedkit.download.ModelDownloader
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class ColdStartBudgetTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun libraryInitializationStaysWithinCommittedBudget() {
        val maximumMillis = requiredPositiveLongProperty("openmed.coldStart.maxMillis")
        val measurementFile = File(
            requireNotNull(System.getProperty("openmed.coldStart.measurementFile")) {
                "Missing openmed.coldStart.measurementFile system property"
            },
        )
        val context = RuntimeEnvironment.getApplication()
        val cacheDirectory = temporaryFolder.newFolder("cold-start-cache")

        val startedAtNanos = System.nanoTime()
        val catalog = ModelCatalog.load(context)
        val cache = ModelCache(cacheDirectory)
        val downloader = ModelDownloader(cache, NetworkRejectingClient)
        val elapsedNanos = System.nanoTime() - startedAtNanos
        val elapsedMillis = (elapsedNanos + NANOS_PER_MILLISECOND - 1) / NANOS_PER_MILLISECOND
        val passed = elapsedMillis <= maximumMillis

        measurementFile.parentFile?.mkdirs()
        measurementFile.writeText(
            """
            measuredMillis=$elapsedMillis
            maxMillis=$maximumMillis
            catalogEntries=${catalog.entries.size}
            status=${if (passed) "PASS" else "FAIL"}
            """.trimIndent() + "\n",
        )
        println(
            "OpenMedKit cold start: $elapsedMillis ms " +
                "(budget: $maximumMillis ms, catalog entries: ${catalog.entries.size})",
        )

        assertNotNull(downloader)
        assertTrue(
            "OpenMedKit cold start took $elapsedMillis ms, exceeding the " +
                "$maximumMillis ms budget",
            passed,
        )
    }

    private fun requiredPositiveLongProperty(name: String): Long {
        val value = requireNotNull(System.getProperty(name)) {
            "Missing $name system property"
        }
        return requireNotNull(value.toLongOrNull()?.takeIf { it > 0 }) {
            "$name must be a positive integer, got '$value'"
        }
    }

    private object NetworkRejectingClient : HuggingFaceModelClient {
        override fun fetchModelInfo(
            repoId: String,
            revision: String?,
        ): HuggingFaceModelInfo =
            throw AssertionError("Cold-start initialization must not access the network")

        override fun downloadFile(
            repoId: String,
            path: String,
            revision: String,
            destination: File,
        ) {
            throw AssertionError("Cold-start initialization must not access the network")
        }
    }

    private companion object {
        const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}
