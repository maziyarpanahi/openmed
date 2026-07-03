package com.openmed.openmedkit.merge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PiiEntityMergerTest {
    @Test
    fun findsSemanticUnitsWithCaptureGroupsPriorityAndContextScoring() {
        val text = "DOB: 01/15/1970, SSN: 123-45-6789, Phone: (415) 555-1234, Email: jane@example.org"

        val units = PiiEntityMerger.findSemanticUnits(text)

        assertEquals(
            listOf("date_of_birth", "ssn", "phone_number", "email"),
            units.map { it.entityType },
        )
        assertEquals("01/15/1970", text.substring(units[0].start, units[0].end))
        assertEquals("123-45-6789", text.substring(units[1].start, units[1].end))
        assertEquals("(415) 555-1234", text.substring(units[2].start, units[2].end))
        assertEquals("jane@example.org", text.substring(units[3].start, units[3].end))

        assertEquals(0.99, units[0].score.toDouble(), 0.0001)
        assertEquals(0.98, units[1].score.toDouble(), 0.0001)
        assertEquals(0.98, units[2].score.toDouble(), 0.0001)
        assertEquals(0.99, units[3].score.toDouble(), 0.0001)
        assertTrue(units.all { it.validated })
    }

    @Test
    fun suppressesLowerPriorityOverlappingSemanticUnits() {
        val text = "SSN: 123-45-6789"
        val patterns = listOf(
            SemanticUnitPattern(
                "\\b\\d{3}-\\d{2}-\\d{4}\\b",
                entityType = "ssn",
                priority = 10,
                baseScore = 0.8f,
            ),
            SemanticUnitPattern(
                "\\d{3}-\\d{2}",
                entityType = "partial_id",
                priority = 1,
                baseScore = 0.8f,
            ),
        )

        val units = PiiEntityMerger.findSemanticUnits(text, patterns)

        assertEquals(1, units.size)
        assertEquals("ssn", units.single().entityType)
        assertEquals("123-45-6789", text.substring(units.single().start, units.single().end))
    }

    @Test
    fun mergesFragmentedModelSpansAndAddsHighConfidenceSemanticOnlyMatches() {
        val text = "DOB: 01/15/1970; email jane@example.org"
        val dateStart = text.indexOf("01/15/1970")

        val merged = PiiEntityMerger.mergePIIEntities(
            entities = listOf(
                EntityPrediction(
                    label = "date",
                    text = "01",
                    confidence = 0.711f,
                    start = dateStart,
                    end = dateStart + 2,
                ),
                EntityPrediction(
                    label = "date_of_birth",
                    text = "/15/1970",
                    confidence = 0.751f,
                    start = dateStart + 2,
                    end = dateStart + 10,
                ),
            ),
            text = text,
        )

        assertEquals(2, merged.size)
        assertEquals("date_of_birth", merged[0].label)
        assertEquals("01/15/1970", merged[0].text)
        assertEquals(dateStart, merged[0].start)
        assertEquals(dateStart + 10, merged[0].end)
        assertEquals(0.8346, merged[0].confidence.toDouble(), 0.0001)

        assertEquals("email", merged[1].label)
        assertEquals("jane@example.org", merged[1].text)
        assertEquals(1.0, merged[1].confidence.toDouble(), 0.0001)
    }

    @Test
    fun respectsSemanticOnlyAndLabelExpansionFlags() {
        val text = "DOB: 01/15/1970"
        val dateStart = text.indexOf("01/15/1970")
        val modelDate = EntityPrediction(
            label = "date",
            text = "01/15/1970",
            confidence = 0.9f,
            start = dateStart,
            end = dateStart + 10,
        )

        val expanded = PiiEntityMerger.mergePIIEntities(
            entities = listOf(modelDate),
            text = text,
        )
        val notExpanded = PiiEntityMerger.mergePIIEntities(
            entities = listOf(modelDate),
            text = text,
            allowSemanticLabelExpansion = false,
        )
        val semanticOnlyDisabled = PiiEntityMerger.mergePIIEntities(
            entities = emptyList(),
            text = text,
            allowSemanticOnlyMatches = false,
        )

        assertEquals("date_of_birth", expanded.single().label)
        assertEquals("date", notExpanded.single().label)
        assertTrue(semanticOnlyDisabled.isEmpty())
    }

    @Test
    fun calculatesDominantLabelAndNormalizesFamilies() {
        val entities = listOf(
            EntityPrediction("date", "01", 0.95f, 0, 2),
            EntityPrediction("date_of_birth", "/15", 0.75f, 2, 5),
        )

        val dominant = PiiEntityMerger.calculateDominantLabel(entities)

        assertEquals("date", dominant.label)
        assertEquals(0.85, dominant.averageConfidence.toDouble(), 0.0001)
        assertEquals("date", PiiEntityMerger.normalizeLabel("date_of_birth"))
        assertEquals("phone", PiiEntityMerger.normalizeLabel("fax_number"))
        assertEquals("provider_identifier", PiiEntityMerger.normalizeLabel("npi"))
        assertEquals("insurance_id", PiiEntityMerger.normalizeLabel("member_id"))
        assertEquals("payment_card", PiiEntityMerger.normalizeLabel("credit_debit_card"))
        assertTrue(PiiEntityMerger.isMoreSpecific("date_of_birth", "date"))
        assertTrue(PiiEntityMerger.isMoreSpecific("npi", "id"))
        assertFalse(PiiEntityMerger.isMoreSpecific("date", "date_of_birth"))
    }

    @Test
    fun deduplicatesOverlappingEntitiesUsingSwiftSelectionRules() {
        val entities = listOf(
            EntityPrediction("date", "01/15", 0.94f, 0, 5),
            EntityPrediction("date_of_birth", "01/15/1970", 0.90f, 0, 10),
            EntityPrediction("ssn", "123-45-6789", 0.80f, 20, 31),
            EntityPrediction("ssn", "123-45-6789", 0.95f, 20, 31),
            EntityPrediction("phone_number", "123-45-6789", 0.70f, 20, 31),
        )

        val deduplicated = PiiEntityMerger.deduplicateOverlappingEntities(entities)

        assertEquals(3, deduplicated.size)
        assertEquals("date_of_birth", deduplicated[0].label)
        assertEquals(0, deduplicated[0].start)
        assertEquals(10, deduplicated[0].end)
        assertEquals("ssn", deduplicated[1].label)
        assertEquals(0.95, deduplicated[1].confidence.toDouble(), 0.0001)
        assertEquals("phone_number", deduplicated[2].label)
    }
}
