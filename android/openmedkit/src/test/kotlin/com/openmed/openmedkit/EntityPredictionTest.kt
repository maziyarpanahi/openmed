package com.openmed.openmedkit

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class EntityPredictionTest {
    @Test
    fun equalityMatchesOnAllFields() {
        val a = EntityPrediction("PERSON", "Alice", 0.98f, 0, 5)
        val b = EntityPrediction("PERSON", "Alice", 0.98f, 0, 5)
        val different = EntityPrediction("PERSON", "Alicia", 0.98f, 0, 6)

        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, different)
    }

    @Test
    fun descriptionMatchesSwiftFormat() {
        val prediction = EntityPrediction("PERSON", "Alice", 0.98f, 0, 5)

        assertEquals("[PERSON] \"Alice\" (0:5) conf=0.98", prediction.toString())
    }

    @Test
    fun descriptionRendersTwoDecimalConfidence() {
        val prediction = EntityPrediction("SSN", "123-45-6789", 0.5f, 10, 21)

        assertEquals("[SSN] \"123-45-6789\" (10:21) conf=0.50", prediction.toString())
    }

    @Test
    fun entityTypeMirrorsLabel() {
        val prediction = EntityPrediction("date_of_birth", "2024-01-02", 0.9f, 3, 13)

        assertEquals("date_of_birth", prediction.entityType)
    }

    @Test
    fun offsetsAreHalfOpenWithEndAtLeastStart() {
        val prediction = EntityPrediction("PERSON", "Alice", 0.98f, 0, 5)

        assertTrue(prediction.end >= prediction.start)
        assertEquals(prediction.text.length, prediction.end - prediction.start)
    }
}
