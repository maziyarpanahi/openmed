package com.openmed.openmedkit.safety

import java.io.File
import kotlinx.serialization.json.*
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class CrisisRoutingTest {
    private fun text(name: String) = requireNotNull(javaClass.getResourceAsStream("/$name"))
        .bufferedReader().use { it.readText() }

    @Test fun sharedOfflineLocaleVectors() {
        val router = CrisisRouting()
        val vectors = Json.parseToJsonElement(text("crisis_routing_v1.json")).jsonObject
        for (item in vectors.getValue("cases").jsonArray) {
            val row = item.jsonObject
            val notice = assertNotNull(router.route(row.getValue("locale").jsonPrimitive.content, true))
            assertEquals(vectors.getValue("generic_notice").jsonPrimitive.content, notice.text)
            assertEquals(row.getValue("matched").jsonPrimitive.contentOrNull, notice.matchedLocale)
            assertEquals(row.getValue("ids").jsonArray.map { it.jsonPrimitive.content }, notice.resources.map { it.id })
        }
        assertNull(router.route("en-GB", false))
        val source = File("../../openmed/clinical/data/crisis_resources.json")
        assertEquals(source.readText(), text("crisis_resources.json"))
    }

    @Test fun overrideIsLocalAndInvalidMetadataIsRefused() {
        val row = """{"id":"local","label":"Site resource","url":"https://example.org/info","source_url":"https://example.org/source","verified_on":"2025-01-12"}"""
        val table = """{"schema_version":1,"locales":{"fr":[$row],"en-AU":[$row]}}"""
        val router = CrisisRouting(table)
        assertEquals("fr", router.route("fr-CA", true)?.matchedLocale)
        assertEquals("local", router.route("en-AU", true)?.resources?.single()?.id)
        assertEquals("nhs_emergency_departments", router.route("en-GB", true)?.resources?.single()?.id)
        for (bad in listOf(table.replace("https:", "tel:"), table.replace("2025-01-12", "2025-02-30"))) {
            assertEquals("invalid_crisis_table", assertFailsWith<IllegalArgumentException> { CrisisRouting(bad) }.message)
        }
    }
}
