package com.openmed.openmedkit.parity

import com.openmed.openmedkit.catalog.ModelCatalog
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Runs `tests/fixtures/parity/catalog_filters_v1.json` against
 * [ModelCatalog.filter] and [ModelCatalog.byRepoId]. Pure JVM: synthetic rows,
 * no asset, network or device.
 */
class CatalogFilterParityTest {
    private val fixture: JsonObject = javaClass.classLoader!!.getResourceAsStream(
        "catalog_filters_v1.json",
    )!!.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
    private val rows = fixture.getValue("rows").jsonArray.map { it.jsonObject }
    private val catalog = ModelCatalog.fromJsonLines(rows.joinToString("\n") { it.toString() })
    private val queryKeys = setOf("format", "tier", "max_param_count", "language", "license")

    private fun JsonObject.text(key: String): String? =
        this[key]?.takeIf { it != JsonNull }?.jsonPrimitive?.content

    @Test
    fun fixtureRowsAreSyntheticAndLoadInCatalogOrder() {
        assertEquals("openmed.model_catalog_filters", fixture.text("schema"))
        assertEquals(1, fixture.getValue("version").jsonPrimitive.int)
        assertTrue(fixture.getValue("synthetic_only").jsonPrimitive.boolean)
        val ids = rows.map { it.text("repo_id")!! }
        assertTrue(ids.all { it.startsWith("Synthetic/") })
        assertEquals(ids.size, ids.toSet().size)
        assertEquals(ids, catalog.entries.map { it.repoId })
        // The rows cover the documented edge cases.
        assertTrue(catalog.entries.any { it.tier == null })
        assertTrue(catalog.entries.any { it.paramCount == null })
        assertTrue(catalog.entries.any { it.license == null })
        assertTrue(catalog.entries.any { it.formats.isEmpty() })
        assertTrue(catalog.entries.any { entry -> entry.formats.any { it != it.lowercase() } })
    }

    @Test
    fun everyQueryReturnsTheExpectedRepoIdsInCatalogOrder() {
        val cases = fixture.getValue("cases").jsonArray.map { it.jsonObject }
        assertTrue(cases.size >= 15)
        for (case in cases) {
            val id = case.text("id")!!
            val query = case.getValue("query").jsonObject
            assertTrue(queryKeys.containsAll(query.keys), id)
            val actual = catalog.filter(
                format = query.text("format"),
                tier = query.text("tier"),
                maxParamCount = query["max_param_count"]?.jsonPrimitive?.long,
                language = query.text("language"),
                license = query.text("license"),
            ).map { it.repoId }
            val expected = case.getValue("expected_repo_ids").jsonArray.map { it.jsonPrimitive.content }
            assertEquals(expected, actual, id)
        }
    }

    @Test
    fun lookupsMatchRepoIdsExactly() {
        val lookups = fixture.getValue("lookups").jsonArray.map { it.jsonObject }
        assertTrue(lookups.any { !it.getValue("found").jsonPrimitive.boolean })
        for (lookup in lookups) {
            val repoId = lookup.text("repo_id")!!
            val entry = catalog.byRepoId(repoId)
            if (lookup.getValue("found").jsonPrimitive.boolean) {
                assertEquals(repoId, entry?.repoId, repoId)
            } else {
                assertNull(entry, repoId)
            }
        }
    }
}
