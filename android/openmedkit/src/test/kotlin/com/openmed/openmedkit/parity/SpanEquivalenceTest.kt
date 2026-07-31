package com.openmed.openmedkit.parity

import com.openmed.openmedkit.OpenMed
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class SpanEquivalenceTest {
    @Test
    fun androidDeidentificationSpansMatchSharedFixture() {
        val note = readFixtureText("synthetic_clinical_note.txt")
        val expected = readExpectedSpans()
        val actualEntities = OpenMed().extractPII(note)
        val actual = actualEntities.map {
            ExpectedSpan(label = it.label, start = it.start, end = it.end)
        }

        assertEquals(expected, actual)
        actualEntities.forEach { entity ->
            assertEquals(entity.label, note.substring(entity.start, entity.end), entity.text)
        }
    }

    private fun readExpectedSpans(): List<ExpectedSpan> {
        val payload = JSONObject(readFixtureText("expected_spans.json"))
        val spans = payload.getJSONArray("spans")
        return (0 until spans.length()).map { index ->
            val span = spans.getJSONObject(index)
            ExpectedSpan(
                label = span.getString("label"),
                start = span.getInt("start"),
                end = span.getInt("end"),
            )
        }
    }

    private fun readFixtureText(name: String): String {
        val stream = javaClass.classLoader?.getResourceAsStream(name)
        assertNotNull("Missing fixture resource: $name", stream)
        return stream!!.bufferedReader().use { it.readText() }
    }

    private data class ExpectedSpan(
        val label: String,
        val start: Int,
        val end: Int,
    )
}
