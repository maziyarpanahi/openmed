package com.openmed.openmedkit.safety

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.TokenClassificationPrediction
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlinx.coroutines.runBlocking

class ClinicianModeTest {
    private val ack = OpenMedDisclaimer.forTier("restricted").acknowledge(Instant.EPOCH)!!

    @Test fun explicitModeIsScopedToOneInstanceAndCanBeRevoked() {
        val mode = ClinicianMode()
        assertNull(mode.requireCapability("deidentify"))
        assertFailsWith<ClinicianModeRequired> { mode.requireCapability("decisions") }
        assertFailsWith<NoticeAcknowledgementRequired> { mode.enable(ClinicianPurpose.REVIEW, null) }
        assertFailsWith<InvalidNoticeAcknowledgement> {
            mode.enable(ClinicianPurpose.REVIEW, ack.copy(versionDigest = "stale"))
        }
        val receipt = mode.enable(ClinicianPurpose.REVIEW, ack)
        assertEquals(ClinicianModeReceipt("review", ack.noticeId, ack.versionDigest), receipt)
        assertEquals(receipt, mode.requireCapability("decisions"))
        assertFailsWith<ClinicianModeRequired> { ClinicianMode().requireCapability("decisions") }
        mode.disable()
        assertFailsWith<ClinicianModeRequired> { mode.requireTier("future_tier") }
    }

    @Test fun catalogBoundFacadeChecksBeforeInference(): Unit = runBlocking {
        val mode = ClinicianMode()
        var calls = 0
        val classifier = object : OnnxTokenClassifier {
            override suspend fun predict(text: String): List<TokenClassificationPrediction> {
                calls++
                return emptyList()
            }
        }
        val entry = ModelCatalogEntry("synthetic", emptyList(), null, null, emptyList(), null, "test", "non_clinical", "decisions")
        OpenMedKit(classifier, entry, mode).use { kit ->
            assertFailsWith<ClinicianModeRequired> { kit.analyzeText("synthetic") }
            assertEquals(0, calls)
            mode.enable(ClinicianPurpose.QUALITY_ASSURANCE, ack)
            kit.analyzeText("synthetic")
            assertEquals(1, calls)
            mode.disable()
            assertFailsWith<ClinicianModeRequired> { kit.analyzeText("synthetic") }
        }
    }
}
