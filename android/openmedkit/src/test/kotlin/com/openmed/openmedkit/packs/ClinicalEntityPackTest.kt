package com.openmed.openmedkit.packs

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.TokenClassificationPrediction
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.download.ModelDownloader
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import org.json.JSONObject
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ClinicalEntityPackTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val candidates = TaskPacks.resolve(TaskPacks.CLINICAL_ENTITY).repoIds

    @Test
    fun resolvesDiseaseDetectCandidatesAndTheExtractionOnlyNotice() {
        val resolution = ClinicalEntityPack(fakeCatalog()).resolution

        assertEquals("clinical_entity_extraction", resolution.task)
        assertEquals(listOf("CONDITION", "DISEASE", "PATHOLOGY"), resolution.canonicalLabels)
        assertTrue(resolution.repoIds.all { "DiseaseDetect" in it && it.endsWith("-onnx-android") })
        assertEquals("openmed.notice.clinical_entity.extraction_only", resolution.noticeKey)
    }

    @Test
    fun returnsOnlyMappedLabelsWithScalarOffsetsFromTheSharedFixture() = runTest {
        val fixture = JSONObject(resource("clinical_entity_pack_v1.json"))
        assertEquals("clinical_entity", fixture.getString("pack_id"))
        assertEquals("unicode_scalar", fixture.getString("offset_unit"))
        val threshold = fixture.getDouble("confidence_threshold").toFloat()
        val cases = fixture.getJSONArray("cases")
        assertTrue(cases.length() >= 3)

        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val text = case.getString("text")
            val pack = preparedPack(FixtureClassifier(text, case.getJSONArray("model_output")))
            try {
                val result = pack.extract(text, threshold)
                val expected = case.getJSONArray("expected")
                val id = case.getString("id")

                assertEquals(TaskPackDetector.MODEL, result.detector, id)
                assertEquals(candidates.first(), result.modelId, id)
                assertEquals(pack.resolution.notice, result.notice, id)
                assertEquals(expected.length(), result.value.size, id)
                result.value.forEachIndexed { position, entity ->
                    val want = expected.getJSONObject(position)
                    assertEquals(want.getString("category"), entity.category, id)
                    assertEquals(want.getInt("start"), entity.start, id)
                    assertEquals(want.getInt("end"), entity.end, id)
                    assertEquals(
                        want.getString("text"),
                        UnicodeOffsetContract.substring(text, entity.start, entity.end),
                        id,
                    )
                    assertEquals(candidates.first(), entity.modelId, id)
                    assertTrue(entity.confidence >= threshold, id)
                }
            } finally {
                pack.close()
            }
        }
    }

    @Test
    fun nonAsciiOffsetsAreScalarsNotUtf16Indices() = runTest {
        val text = "🧪 note: 😷 influenza"
        val pack = preparedPack(FakeLabelClassifier("influenza" to "CONDITION"))
        try {
            val entity = pack.extract(text).value.single()

            assertEquals(10, entity.start)
            assertEquals(text.indexOf("influenza") - 2, entity.start)
            assertEquals(19, entity.end)
        } finally {
            pack.close()
        }
    }

    @Test
    fun noDownloadHappensWithoutConsent() = runTest {
        val client = FakeHubClient()
        val pack = ClinicalEntityPack(fakeCatalog(*candidates.toTypedArray()))

        val preparation = pack.prepare(downloader(client), allowDownload = false)

        assertEquals(TaskPackFailure.NOT_CACHED, preparation.failure)
        assertNull(preparation.modelId)
        assertEquals(emptyList(), client.calls)
        val error = assertFailsWith<TaskPackException> { pack.extract("Synthetic asthma.") }
        assertEquals(TaskPackFailure.NOT_PREPARED, error.failure)
    }

    private fun preparedPack(classifier: OnnxTokenClassifier): ClinicalEntityPack {
        val pack = ClinicalEntityPack(
            fakeCatalog(*candidates.toTypedArray()),
            classifierFactory = { classifier },
        )
        assertNull(pack.prepare(downloader(FakeHubClient()), allowDownload = true).failure)
        return pack
    }

    private fun downloader(client: FakeHubClient): ModelDownloader =
        ModelDownloader(ModelCache(temporaryFolder.newFolder()), client)

    private fun resource(name: String): String =
        requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) { "missing $name" }
            .use { it.readBytes().decodeToString() }

    /** Replays a fixture's `model_output` spans, already in scalar offsets. */
    private class FixtureClassifier(
        private val source: String,
        private val output: org.json.JSONArray,
    ) : OnnxTokenClassifier {
        override suspend fun predict(text: String): List<TokenClassificationPrediction> {
            check(text == source)
            return (0 until output.length()).map { index ->
                val span = output.getJSONObject(index)
                val start = span.getInt("start")
                val end = span.getInt("end")
                TokenClassificationPrediction(
                    label = span.getString("label"),
                    text = UnicodeOffsetContract.substring(text, start, end),
                    confidence = span.getDouble("confidence").toFloat(),
                    start = start,
                    end = end,
                )
            }
        }
    }
}
