package com.openmed.openmedkit.packs

import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.download.ModelDownloader
import java.lang.reflect.Modifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
class MedicationPackTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val candidates = TaskPacks.resolve(TaskPacks.MEDICATION).repoIds

    @Test
    fun resolvesPharmaDetectCandidatesWithTheExtractionOnlyNotice() {
        val resolution = MedicationPack(fakeCatalog()).resolution

        assertEquals("medication_mention_extraction", resolution.task)
        assertEquals(listOf("CHEMICAL"), resolution.canonicalLabels)
        assertTrue(resolution.repoIds.all { "PharmaDetect" in it && it.endsWith("-onnx-android") })
        assertEquals("openmed.notice.medication.extraction_only", resolution.noticeKey)
        assertTrue("never gives dosing advice" in resolution.notice)
        assertTrue("never triggers an action" in resolution.notice)
    }

    @Test
    fun returnsMentionsWithScalarOffsetsAndTheNotice() = runTest {
        val text = "Synthetic note 🙂: started metformin, then ibuprofen; left knee."
        val pack = MedicationPack(
            fakeCatalog(*candidates.toTypedArray()),
            classifierFactory = {
                FakeLabelClassifier(
                    "metformin" to "CHEM",
                    "ibuprofen" to "Chemical",
                    "left knee" to "ANATOMY",
                    "Synthetic" to "PERSON",
                )
            },
        )
        try {
            assertNull(pack.prepare(downloader(FakeHubClient()), allowDownload = true).failure)

            val result = pack.extract(text)

            assertEquals(TaskPackDetector.MODEL, result.detector)
            assertEquals(candidates.first(), result.modelId)
            assertEquals("openmed.notice.medication.extraction_only", result.noticeKey)
            assertEquals(pack.resolution.notice, result.notice)
            assertEquals(listOf("CHEMICAL", "CHEMICAL"), result.value.map { it.label })
            val metformin = result.value.first()
            assertEquals(text.indexOf("metformin") - 1, metformin.start)
            assertEquals(metformin.start + "metformin".length, metformin.end)
            assertEquals(
                listOf("metformin", "ibuprofen"),
                result.value.map { UnicodeOffsetContract.substring(text, it.start, it.end) },
            )
            assertTrue(result.value.all { it.modelId == candidates.first() })
        } finally {
            pack.close()
        }
    }

    @Test
    fun noResultTypeHasADoseInteractionOrActionField() {
        val forbidden = listOf("dose", "dosage", "dosing", "interaction", "action", "recommend")
        for (type in listOf(MedicationMention::class.java, TaskPackResult::class.java)) {
            val members = type.declaredFields.map { it.name } + type.methods.map { it.name }
            for (member in members) {
                assertTrue(
                    forbidden.none { member.lowercase().contains(it) },
                    "${type.simpleName}.$member",
                )
            }
        }
    }

    @Test
    fun packExposesExtractionAndLifecycleOnly() {
        val allowed = setOf(
            "extract",
            "extract\$default",
            "prepare",
            "prepare\$default",
            "close",
            "getResolution",
            "getModelId",
        )
        val public = MedicationPack::class.java.methods
            .filter { it.declaringClass != Any::class.java && Modifier.isPublic(it.modifiers) }
            .map { it.name }
            .filterNot { "\$openmedkit" in it }
            .toSet()

        assertEquals(allowed, public)
    }

    @Test
    fun noDownloadHappensWithoutConsent() = runTest {
        val client = FakeHubClient()
        val pack = MedicationPack(fakeCatalog(*candidates.toTypedArray()))

        val preparation = pack.prepare(downloader(client), allowDownload = false)

        assertEquals(TaskPackFailure.NOT_CACHED, preparation.failure)
        assertEquals(emptyList(), client.calls)
        val error = assertFailsWith<TaskPackException> { pack.extract("Synthetic metformin.") }
        assertEquals(TaskPackFailure.NOT_PREPARED, error.failure)
    }

    private fun downloader(client: FakeHubClient): ModelDownloader =
        ModelDownloader(ModelCache(temporaryFolder.newFolder()), client)
}
