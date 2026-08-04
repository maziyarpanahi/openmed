package com.openmed.openmedkit

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OpenMedSpanTest {
    @Test
    fun schemaVersionDefaultsToConstant() {
        val span = OpenMedSpan(0, 5, "Alice", "PERSON", "PERSON", 0.98f)

        assertEquals(1, OpenMedSpan.SCHEMA_VERSION)
        assertEquals(OpenMedSpan.SCHEMA_VERSION, span.schemaVersion)
    }

    @Test
    fun fromEntityPredictionDefaultsCanonicalToRawLabel() {
        val prediction = EntityPrediction("PERSON", "Alice", 0.98f, 0, 5)

        val span = OpenMedSpan.fromEntityPrediction(prediction)

        assertEquals("PERSON", span.rawLabel)
        assertEquals("PERSON", span.canonicalLabel)
        assertEquals("Alice", span.text)
        assertEquals(0.98f, span.score)
        assertEquals(0, span.start)
        assertEquals(5, span.end)
    }

    @Test
    fun fromEntityPredictionAcceptsExplicitCanonicalLabel() {
        val prediction = EntityPrediction("PER", "Alice", 0.98f, 0, 5)

        val span = OpenMedSpan.fromEntityPrediction(prediction, canonicalLabel = "PERSON")

        assertEquals("PER", span.rawLabel)
        assertEquals("PERSON", span.canonicalLabel)
    }

    @Test
    fun roundTripPreservesAllEntityPredictionFields() {
        val original = EntityPrediction("date_of_birth", "2024-01-02", 0.87f, 3, 13)

        val restored = OpenMedSpan.fromEntityPrediction(original).toEntityPrediction()

        assertEquals(original, restored)
    }

    @Test
    fun offsetsMustHaveEndAtLeastStart() {
        assertFailsWith<IllegalArgumentException> {
            OpenMedSpan(5, 3, "x", "PERSON", "PERSON", 0.5f)
        }
    }

    @Test
    fun startMustBeNonNegative() {
        assertFailsWith<IllegalArgumentException> {
            OpenMedSpan(-1, 3, "x", "PERSON", "PERSON", 0.5f)
        }
    }

    @Test
    fun scoreMustBeInUnitInterval() {
        assertFailsWith<IllegalArgumentException> {
            OpenMedSpan(0, 3, "x", "PERSON", "PERSON", 1.5f)
        }
    }

    @Test
    fun emptySpanWithEqualOffsetsIsAllowed() {
        val span = OpenMedSpan(4, 4, "", "PERSON", "PERSON", 0.5f)

        assertTrue(span.end >= span.start)
        assertEquals(0, span.end - span.start)
    }
}
