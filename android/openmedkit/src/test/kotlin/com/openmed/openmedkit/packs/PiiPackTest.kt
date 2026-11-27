package com.openmed.openmedkit.packs

import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.util.SafeLog
import com.openmed.openmedkit.util.SafeLogEvent
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class PiiPackTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val first = TaskPacks.resolve(TaskPacks.PII).repoIds.first()

    @Test
    fun bundledManifestIsAByteIdenticalCopyOfTheSharedManifest() {
        val shared = repositoryRoot().resolve("openmed/core/data/task_packs.json")

        assertContentEquals(shared.readBytes(), TaskPacks.bundledManifest())
    }

    @Test
    fun resolvesOrderedAndroidCandidatesFromTheBundledManifest() {
        val resolution = PiiPack(fakeCatalog()).resolution

        assertEquals("pii", resolution.packId)
        assertEquals("pii_detection", resolution.task)
        assertEquals(3, resolution.repoIds.size)
        assertTrue(resolution.repoIds.all { it.endsWith("-onnx-android") })
        assertTrue("PERSON" in resolution.canonicalLabels)
        assertEquals("openmed.notice.pii.review_required", resolution.noticeKey)
        assertTrue(resolution.notice.isNotBlank())
    }

    @Test
    fun downloadsWithConsentThenExtractsSpansAndDeidentifies() = runTest {
        val client = FakeHubClient()
        val classifier = FakeLabelClassifier("Jane Roe" to "PERSON", PHONE to "PHONE")
        val pack = PiiPack(fakeCatalog(*allCandidates()), classifierFactory = { classifier })

        val preparation = pack.prepare(downloader(client), allowDownload = true)
        try {
            assertNull(preparation.failure)
            assertFalse(preparation.fromCache)
            assertEquals(first, preparation.modelId)
            assertTrue(client.calls.isNotEmpty())

            val extracted = pack.extract(NOTE)
            assertEquals(TaskPackDetector.MODEL, extracted.detector)
            assertEquals(first, extracted.modelId)
            assertEquals(listOf("PERSON", "PHONE"), extracted.value.map { it.label })
            assertEquals(listOf(9, 17), extracted.value.first().let { listOf(it.start, it.end) })
            assertEquals(pack.resolution.notice, extracted.notice)

            val deidentified = pack.deidentify(NOTE)
            assertEquals(TaskPackDetector.MODEL, deidentified.detector)
            assertFalse(deidentified.value.redactedText.contains("Jane Roe"))
            assertFalse(deidentified.value.redactedText.contains(PHONE))
            assertTrue(deidentified.value.redactedText.startsWith("Patient: "))
        } finally {
            pack.close()
        }
        assertTrue(classifier.closed)
    }

    @Test
    fun failsClosedWithoutConsentAndAnEmptyCache() = runTest {
        val client = FakeHubClient()
        val pack = PiiPack(fakeCatalog(*allCandidates()))

        val preparation = pack.prepare(downloader(client), allowDownload = false)

                assertEquals(TaskPackFailure.NOT_CACHED, preparation.failure)
        assertNull(preparation.modelId)
        assertEquals(emptyList(), client.calls)
        val error = assertFailsWith<TaskPackException> { pack.extract(NOTE) }
        assertEquals(TaskPackFailure.NOT_PREPARED, error.failure)
        assertEquals("not_prepared", error.message)
        assertFailsWith<TaskPackException> { pack.deidentify(NOTE) }
    }

    @Test
    fun usesACachedCandidateWithoutConsentOrNetwork() = runTest {
        val cacheDirectory = temporaryFolder.newFolder("cache")
        val catalog = fakeCatalog(*allCandidates())
        ModelDownloader(ModelCache(cacheDirectory), FakeHubClient())
            .download(requireNotNull(catalog.byRepoId(first)))
        val offline = FakeHubClient()
        val cache = ModelCache(cacheDirectory)
        val pack = PiiPack(catalog, classifierFactory = { FakeLabelClassifier() })

        val preparation = pack.prepare(ModelDownloader(cache, offline), allowDownload = false)

        assertNull(preparation.failure)
        assertTrue(preparation.fromCache)
        assertEquals(first, pack.modelId)
        assertEquals(emptyList(), offline.calls)
        assertTrue(cache.isLeased(first))
        pack.close()
        assertFalse(cache.isLeased(first))
        assertNull(pack.modelId)
    }

    @Test
    fun localRecognizerRunsOnlyWhenTheCallerOptsIn() = runTest {
        val pack = PiiPack(fakeCatalog(), allowLocalFallback = true)

        val extracted = pack.extract(NOTE)
        val deidentified = pack.deidentify(NOTE)

        assertEquals(TaskPackDetector.LOCAL_RECOGNIZER, extracted.detector)
        assertNull(extracted.modelId)
        assertEquals(listOf("patient_name", "phone_number"), extracted.value.map { it.label })
        assertEquals(TaskPackDetector.LOCAL_RECOGNIZER, deidentified.detector)
        assertFalse(deidentified.value.redactedText.contains("Jane Roe"))
        assertFalse(deidentified.value.redactedText.contains(PHONE))
    }

    @Test
    fun reportsTypedFailuresForMissingCandidatesDownloadsAndLoads() = runTest {
        val noCandidate = PiiPack(fakeCatalog("OpenMed/not-a-pack-candidate"))
            .prepare(downloader(FakeHubClient()), allowDownload = true)
        assertEquals(TaskPackFailure.NO_CANDIDATE, noCandidate.failure)

        val corruptCache = temporaryFolder.newFolder("corrupt")
        val corrupt = PiiPack(fakeCatalog(*allCandidates()))
            .prepare(ModelDownloader(ModelCache(corruptCache), FakeHubClient(corrupt = true)), true)
        assertEquals(TaskPackFailure.DOWNLOAD_FAILED, corrupt.failure)
        assertTrue(allCandidates().none { ModelCache(corruptCache).isAvailable(it) })

        val unloadable = PiiPack(
            fakeCatalog(*allCandidates()),
            classifierFactory = { directory -> error("cannot open ${directory.path}") },
        )
        val loadFailure = unloadable.prepare(downloader(FakeHubClient()), allowDownload = true)
        assertEquals(TaskPackFailure.LOAD_FAILED, loadFailure.failure)
        assertNull(unloadable.modelId)
    }

    @Test
    fun rejectsMissingTamperedOrUnknownManifestsWithValueFreeReasons() {
        val bundled = requireNotNull(TaskPacks.bundledManifest()).decodeToString()
        val tampered = bundled.replace("\"rank\": 1,", "\"rank\": 7,")
        val wrongSchema = bundled.replace("\"openmed.task_packs\"", "\"other\"")

        for (manifest in listOf(null, "not json".toByteArray(), tampered.toByteArray())) {
            val error = assertFailsWith<TaskPackException> {
                PiiPack(fakeCatalog(), manifest = manifest)
            }
            assertEquals(TaskPackFailure.MANIFEST_INVALID, error.failure)
        }
        assertEquals(
            TaskPackFailure.MANIFEST_INVALID,
            assertFailsWith<TaskPackException> {
                TaskPacks.resolve(TaskPacks.PII, wrongSchema.toByteArray())
            }.failure,
        )
        val unknown = assertFailsWith<TaskPackException> { TaskPacks.resolve("unknown_pack_id") }
        assertEquals(TaskPackFailure.UNKNOWN_PACK, unknown.failure)
        assertEquals("unknown_pack", unknown.message)
    }

    @Test
    fun safeLogRecordsContainNoInputText() = runTest {
        val captured = mutableListOf<SafeLogEvent>()
        val previous = SafeLog.installSinkForTesting(captured::add)
        val pack = PiiPack(
            fakeCatalog(*allCandidates()),
            allowLocalFallback = true,
            classifierFactory = { FakeLabelClassifier("Jane Roe" to "PERSON", PHONE to "PHONE") },
        )
        try {
            pack.extract(NOTE)
            pack.deidentify(NOTE)
            pack.prepare(downloader(FakeHubClient()), allowDownload = true)
            pack.extract(NOTE)
            pack.deidentify(NOTE)
        } finally {
            pack.close()
            SafeLog.installSinkForTesting(previous)
        }

        val diagnostics = captured.joinToString("\n")
        assertTrue(captured.isNotEmpty())
        for (sentinel in listOf(NOTE, "Jane Roe", "Jane", "Roe", PHONE)) {
            assertFalse(diagnostics.contains(sentinel))
        }
    }

    private fun downloader(client: FakeHubClient): ModelDownloader {
        val directory = File(temporaryFolder.root, "cache-${client.hashCode()}")
        return ModelDownloader(ModelCache(directory), client)
    }

    private fun allCandidates(): Array<String> =
        TaskPacks.resolve(TaskPacks.PII).repoIds.toTypedArray()

    private companion object {
        /** Synthetic note: every identifier in it is fabricated. */
        const val NOTE = "Patient: Jane Roe called 555-010-0199 about a refill."
        const val PHONE = "555-010-0199"
    }
}
