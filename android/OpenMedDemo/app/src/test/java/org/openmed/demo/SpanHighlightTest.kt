package org.openmed.demo

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SpanHighlightTest {
    @Test
    fun sampleClinicalNoteMapsHighlightedSpansToCharacterOffsets() {
        val note = sampleClinicalNote()
        val entities = DemoEntityDetector.detect(note)
        val highlighted = mapHighlightSegments(note, entities)
            .filter { it.entity != null }

        assertEquals(7, highlighted.size)

        val email = highlighted.single { it.entity?.label == "Email" }
        assertEquals("jordan.lee@example.test", email.text)
        assertEquals(note.indexOf("jordan.lee@example.test"), email.start)
        assertEquals(email.start + email.text.length, email.end)

        val address = highlighted.single { it.entity?.label == "Address" }
        assertEquals("123 Example Street, Springfield, CA 90000", address.text)
        assertEquals(note.indexOf(address.text), address.start)
        assertEquals(address.start + address.text.length, address.end)

        assertTrue(highlighted.all { it.text == note.substring(it.start, it.end) })
    }

    private fun sampleClinicalNote(): String {
        val candidates = listOf(
            File("src/main/res/raw/sample_clinical_note.txt"),
            File("OpenMedDemo/app/src/main/res/raw/sample_clinical_note.txt"),
        )
        return candidates.first { it.isFile }.readText().trimEnd()
    }
}
