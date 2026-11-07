package com.openmed.openmedkit.intake

import com.openmed.openmedkit.UnicodeOffsetContract
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Consumes `tests/fixtures/parity/document_intake_v1.json`, shared with the
 * Python reference mapper and Swift OpenMedKit.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class DocumentEvidenceParityTest {
    private val corpus: JSONObject = loadFixture()

    @Test
    fun vocabulariesAndDefaultsMatchPython() {
        assertEquals(DocumentEvidenceMapper.SCHEMA_VERSION, corpus.getString("schema"))
        assertTrue(corpus.getBoolean("synthetic_only"))
        assertEquals(DocumentEvidenceMapper.PAGE_SEPARATOR, corpus.getString("page_separator"))
        assertEquals(DocumentEvidenceMapper.BOX_TOLERANCE, corpus.getDouble("box_tolerance"))
        assertEquals(DocumentIntakeOutcome.values().map { it.code }, corpus.getJSONArray("outcomes").strings())
        assertEquals(DocumentIntakeReason.values().map { it.code }, corpus.getJSONArray("reasons").strings())
        assertEquals(DocumentPageStatus.values().map { it.code }, corpus.getJSONArray("page_statuses").strings())
        val defaults = corpus.getJSONObject("default_limits")
        val limits = DocumentIntakeLimits()
        assertEquals(limits.maxPages, defaults.getInt("max_pages"))
        assertEquals(limits.maxRegionsPerPage, defaults.getInt("max_regions_per_page"))
        assertEquals(limits.maxPageScalars, defaults.getInt("max_page_scalars"))
    }

    @Test
    fun sharedVectorsProduceIdenticalPageRegionAndTransformRecords() {
        corpus.getJSONArray("cases").objects().forEach { case ->
            val id = case.getString("id")
            val expected = case.getJSONObject("expected")
            val evidence = run(case)
            assertEquals(expected.getString("outcome"), evidence.outcome.code, id)
            assertEquals(expected.getJSONArray("reasons").strings(), evidence.reasons.map { it.code }, id)
            assertEquals(expected.getString("text"), evidence.text, id)

            val pages = expected.getJSONArray("pages").objects()
            assertEquals(pages.size, evidence.pages.size, id)
            evidence.pages.zip(pages).forEach { (page, want) ->
                assertEquals(want.getInt("page"), page.page, id)
                assertEquals(want.getString("status"), page.status.code, id)
                assertEquals(want.stringOrNull("reason"), page.reason?.code, id)
                assertEquals(want.intsOrNull("text_span"), page.textSpan?.let { listOf(it.start, it.end) }, id)
                assertEquals(want.stringOrNull("node_ref"), page.evidence?.nodeRef, id)
                page.evidence?.let { visual ->
                    assertEquals(DocumentVisualKind.PAGE, visual.kind, id)
                    assertEquals(want.getString("display_width_f64"), bits(visual.displayWidth), id)
                    assertEquals(want.getString("display_height_f64"), bits(visual.displayHeight), id)
                }
                assertEquals(want.stringOrNull("crop_node_ref"), page.cropEvidence?.nodeRef, id)
                page.cropEvidence?.let { crop ->
                    assertEquals(DocumentVisualKind.PIXELS, crop.kind, id)
                    assertEquals(want.getJSONArray("crop_display_box_f64").strings(), bits(crop.displayBox), id)
                }
            }

            val regions = expected.getJSONArray("regions").objects()
            assertEquals(regions.size, evidence.regions.size, id)
            evidence.regions.zip(regions).forEach { (region, want) ->
                assertEquals(want.getInt("page"), region.page, id)
                assertEquals(want.ints("span"), listOf(region.span.start, region.span.end), id)
                assertEquals(want.getString("node_ref"), region.nodeRef, id)
                assertEquals(want.getJSONArray("box_f64").strings(), bits(region.sourceBox), id)
                assertEquals(want.getJSONArray("display_box_f64").strings(), bits(region.evidence.displayBox), id)
                assertEquals(DocumentVisualKind.OCR, region.evidence.kind, id)
                assertEquals(region.span, region.evidence.span, id)
                assertEquals(case.getString("source_digest"), region.evidence.sourceDigest, id)
            }

            expected.getJSONArray("probes").objects().forEach { probe ->
                val span = probe.ints("span")
                assertEquals(
                    probe.getJSONArray("node_refs").strings(),
                    evidence.regionsForSpan(span[0], span[1]).map { it.nodeRef },
                    id,
                )
            }
        }
    }

    @Test
    fun mappedRegionsRecoverTheAuthoredOriginalBoxes() {
        corpus.getJSONArray("cases").objects().filter { it.isNull("refusal") }.forEach { case ->
            val id = case.getString("id")
            val tolerance = case.getDouble("tolerance")
            val evidence = run(case)
            val sourcePages = case.getJSONArray("pages").objects()
            val truths = evidence.pages
                .filter { it.status == DocumentPageStatus.MAPPED }
                .flatMap { page ->
                    sourcePages[page.page].getJSONObject("ocr").getJSONArray("regions").objects()
                        .map { it.doubles("truth") }
                }
            assertEquals(truths.size, evidence.regions.size, id)
            evidence.regions.zip(truths).forEach { (region, truth) ->
                val mapped = region.sourceBox.let { listOf(it.left, it.top, it.right, it.bottom) }
                val error = mapped.zip(truth).maxOf { (a, b) -> Math.abs(a - b) }
                assertTrue(error <= tolerance, "$id region error $error")
            }
        }
    }

    @Test
    fun mixedScriptSpansAreUnicodeScalarsNotUtf16() {
        val case = corpus.getJSONArray("cases").objects().single { it.getString("id") == "mixed_script_image" }
        val evidence = run(case)
        assertEquals(22, UnicodeOffsetContract.scalarLength(evidence.text))
        assertNotEquals(22, evidence.text.length)
        val emoji = evidence.regionsForSpan(15, 16).single()
        val utf16 = UnicodeOffsetContract.utf16Span(evidence.text, emoji.span.start, emoji.span.end)
        assertEquals("😀", evidence.text.substring(utf16.start, utf16.end))
        assertEquals(listOf(emoji), evidence.regionsForUtf16Span(utf16.start, utf16.end))
        assertFalse(evidence.toString().contains("Patient"))
    }

    private fun run(case: JSONObject): DocumentEvidence {
        if (!case.isNull("refusal")) {
            return DocumentEvidenceMapper.refused(reason(case.getString("refusal")))
        }
        val defaults = DocumentIntakeLimits()
        val overrides = if (case.isNull("limits")) null else case.getJSONObject("limits")
        val limits = DocumentIntakeLimits(
            maxPages = overrides?.optInt("max_pages", defaults.maxPages) ?: defaults.maxPages,
            maxRegionsPerPage = overrides?.optInt("max_regions_per_page", defaults.maxRegionsPerPage)
                ?: defaults.maxRegionsPerPage,
            maxPageScalars = overrides?.optInt("max_page_scalars", defaults.maxPageScalars)
                ?: defaults.maxPageScalars,
        )
        val pages = case.getJSONArray("pages").objects().map(::input)
        return DocumentEvidenceMapper.map(case.getString("source_digest"), pages, limits)
    }

    private fun input(row: JSONObject): DocumentPageInput {
        if (!row.isNull("unavailable")) {
            return DocumentPageInput.Unavailable(reason(row.getString("unavailable")))
        }
        val geometry = row.getJSONObject("geometry")
        val ocr = row.getJSONObject("ocr")
        val text = ocr.getString("text")
        val regions = ocr.getJSONArray("regions").objects().map { region ->
            val span = region.ints("span")
            DocumentRecognizedRegion(
                start = utf16(text, span[0]),
                end = utf16(text, span[1]),
                box = region.doubles("box").let { DocumentBox(it[0], it[1], it[2], it[3]) },
            )
        }
        return DocumentPageInput.Rendered(
            geometry = DocumentPageGeometry(
                width = geometry.getDouble("width"),
                height = geometry.getDouble("height"),
                unit = DocumentSpatialUnit.values().single { it.code == geometry.getString("unit") },
                rotationDegrees = geometry.getInt("rotation"),
                crop = if (geometry.isNull("crop")) {
                    null
                } else {
                    geometry.doubles("crop").let { DocumentBox(it[0], it[1], it[2], it[3]) }
                },
            ),
            recognition = DocumentRecognition(
                width = ocr.getInt("width"),
                height = ocr.getInt("height"),
                text = text,
                regions = regions,
            ),
        )
    }

    /** Converts a shared scalar offset to the native index an OCR adapter reports. */
    private fun utf16(text: String, scalar: Int): Int {
        val length = UnicodeOffsetContract.scalarLength(text)
        return if (scalar <= length) {
            UnicodeOffsetContract.scalarToUtf16Index(text, scalar)
        } else {
            text.length + (scalar - length)
        }
    }

    private fun reason(code: String): DocumentIntakeReason = requireNotNull(DocumentIntakeReason.fromCode(code))

    private fun bits(value: Double): String = "%016x".format(java.lang.Double.doubleToRawLongBits(value))

    private fun bits(box: DocumentBox?): List<String> =
        requireNotNull(box).let { listOf(it.left, it.top, it.right, it.bottom).map(::bits) }

    private fun loadFixture(): JSONObject {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream("document_intake_v1.json")) {
            "shared document_intake_v1.json was not synced into Android test resources"
        }
        return stream.bufferedReader(Charsets.UTF_8).use { JSONObject(it.readText()) }
    }

    private fun JSONArray.objects(): List<JSONObject> = (0 until length()).map(::getJSONObject)

    private fun JSONArray.strings(): List<String> = (0 until length()).map(::getString)

    private fun JSONObject.ints(name: String): List<Int> =
        getJSONArray(name).let { array -> (0 until array.length()).map(array::getInt) }

    private fun JSONObject.intsOrNull(name: String): List<Int>? = if (isNull(name)) null else ints(name)

    private fun JSONObject.stringOrNull(name: String): String? = if (isNull(name)) null else getString(name)

    private fun JSONObject.doubles(name: String): List<Double> =
        getJSONArray(name).let { array -> (0 until array.length()).map(array::getDouble) }
}
