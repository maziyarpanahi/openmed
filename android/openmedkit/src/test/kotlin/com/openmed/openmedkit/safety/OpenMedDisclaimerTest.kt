package com.openmed.openmedkit.safety

import com.openmed.openmedkit.catalog.ModelCatalogEntry
import java.time.Instant
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test

class OpenMedDisclaimerTest {
    private val vectors = requireNotNull(javaClass.getResourceAsStream("/safety_primitives_v1.json"))
        .bufferedReader().use { Json.parseToJsonElement(it.readText()).jsonObject }

    @Test fun matchesEverySharedVector() {
        for (entry in vectors.getValue("tiers").jsonArray) {
            val row = entry.jsonObject
            val tier = row.getValue("tier").jsonPrimitive.content
            assertEquals(row.getValue("notice_id").jsonPrimitive.content, OpenMedDisclaimer.forTier(tier).id)
            val model = ModelCatalogEntry("test", emptyList(), null, null, emptyList(), null, "test", tier)
            assertEquals(row.getValue("obligations").jsonArray.map { it.jsonPrimitive.content }.toSet(), model.regulatoryObligations)
        }
        val instant = Instant.parse(vectors.getValue("acknowledged_at").jsonPrimitive.content)
        for (entry in vectors.getValue("notices").jsonArray) {
            val row = entry.jsonObject
            val notice = requireNotNull(OpenMedDisclaimer.byId(row.getValue("id").jsonPrimitive.content))
            assertEquals(row.getValue("version").jsonPrimitive.content, notice.version)
            assertEquals(row.getValue("version_digest").jsonPrimitive.content, notice.versionDigest)
            assertEquals(row.getValue("text").jsonPrimitive.content, notice.text)
            assertEquals(row.getValue("tier").jsonPrimitive.content, notice.tier)
            assertEquals(row.getValue("locale").jsonPrimitive.content, notice.locale)
            assertEquals(row.getValue("requires_acknowledgement").jsonPrimitive.boolean, notice.requiresAcknowledgement)
            assertEquals(!notice.requiresAcknowledgement, notice.permitsResults(null))
            val receipt = notice.acknowledge(instant)
            assertEquals(row.getValue("acknowledgement"), receipt?.toJson() ?: JsonNull)
            assertTrue(notice.permitsResults(receipt))
        }
        for (key in listOf("capability_notices", "entry_point_overrides")) {
            vectors.getValue(key).jsonObject.forEach { (code, id) ->
                assertEquals(id.jsonPrimitive.content, OpenMedDisclaimer.forCapability(code)?.id)
            }
        }
    }

    @Test fun requiresTheCurrentNoticeVersionAndAValidTime() {
        val notice = OpenMedDisclaimer.forTier("future")
        assertEquals("restricted", notice.tier)
        val receipt = requireNotNull(notice.acknowledge(Instant.parse("2026-01-01T00:00:00.123456Z")))
        assertEquals("2026-01-01T00:00:00.123456+00:00", receipt.acknowledgedAt)
        assertFalse(notice.permitsResults(receipt.copy(versionDigest = "stale")))
        assertFalse(notice.permitsResults(receipt.copy(acknowledgedAt = "invalid")))
        assertFalse(OpenMedDisclaimer.forTier("assistive_review").permitsResults(receipt))
    }
}
