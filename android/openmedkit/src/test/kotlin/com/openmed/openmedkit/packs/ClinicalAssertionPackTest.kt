package com.openmed.openmedkit.packs

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.ScalarSpan
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.clinical.ClinicalContext
import com.openmed.openmedkit.download.ModelDownloader
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ClinicalAssertionPackTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun deniedFindingIsReturnedNegatedNotDropped() = runTest {
        val entity = extract("Patient denies chest pain.", "chest pain" to "CONDITION").single()

        assertEquals(15, entity.start)
        assertEquals(25, entity.end)
        val assertion = assertNotNull(entity.assertion)
        assertEquals("negated", assertion.negation)
        assertEquals("certain", assertion.uncertainty)
        assertEquals("recent", assertion.temporality)
        assertEquals("patient", assertion.experiencer)
        assertNull(entity.assertionReason)
    }

    @Test
    fun donorFindingHasTheOtherExperiencer() = runTest {
        val entity = extract("The organ donor was CMV positive.", "CMV positive" to "DISEASE")
            .single()

        assertEquals("other", assertNotNull(entity.assertion).experiencer)
        assertEquals("affirmed", entity.assertion?.negation)
    }

    @Test
    fun nonAsciiNoteKeepsPythonScalarOffsetsAndAxes() = runTest {
        // Python assert_context gives dysuria (24, 31) negated/family and
        // hepatitis (43, 52) affirmed/other; UTF-16 indices are 26 and 45.
        val text = "Note 𝟙 🧪: mother denies dysuria; donor had hepatitis."
        val entities = extract(text, "dysuria" to "DISEASE", "hepatitis" to "DISEASE")

        assertEquals(listOf(24 to 31, 43 to 52), entities.map { it.start to it.end })
        assertEquals(listOf("negated", "affirmed"), entities.map { it.assertion?.negation })
        assertEquals(listOf("family", "other"), entities.map { it.assertion?.experiencer })
    }

    @Test
    fun callerSentencesBoundCueReach() = runTest {
        val text = "Patient denies fever. Cough is present."
        val entities = extract(
            text,
            "fever" to "CONDITION",
            "Cough" to "CONDITION",
            sentences = listOf(ScalarSpan(0, 21), ScalarSpan(22, 39)),
        )

        assertEquals(listOf("negated", "affirmed"), entities.map { it.assertion?.negation })
    }

    @Test
    fun nonEnglishInputReturnsSpansWithoutAnAssertion() = runTest {
        val entity = extract(
            "Paciente niega dolor torácico.",
            "dolor torácico" to "CONDITION",
            language = "es",
        ).single()

        assertNull(entity.assertion)
        assertEquals(ClinicalContext.UNSUPPORTED_LANGUAGE, entity.assertionReason)
        assertEquals("CONDITION", entity.category)
    }

    private suspend fun extract(
        text: String,
        vararg mentions: Pair<String, String>,
        language: String? = null,
        sentences: List<ScalarSpan>? = null,
    ): List<ClinicalEntity> {
        val classifier: OnnxTokenClassifier = FakeLabelClassifier(*mentions)
        val candidates = TaskPacks.resolve(TaskPacks.CLINICAL_ENTITY).repoIds
        val pack = ClinicalEntityPack(
            fakeCatalog(*candidates.toTypedArray()),
            classifierFactory = { classifier },
        )
        val downloader = ModelDownloader(ModelCache(temporaryFolder.newFolder()), FakeHubClient())
        assertNull(pack.prepare(downloader, allowDownload = true).failure)
        return try {
            pack.extract(text, 0.5f, language, sentences).value
        } finally {
            pack.close()
        }
    }
}
