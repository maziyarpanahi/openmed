package com.openmed.openmedkit.decode

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class TokenClassificationDecoderTest {
    @Test
    fun decodeEntitiesGroupsBioTokens() {
        val text = "Patient John Doe visited"
        val tokens = listOf(
            token("B-first_name", 0.95f, 8, 12),
            token("I-first_name", 0.90f, 13, 16),
        )

        val entities = TokenClassificationDecoder.decodeEntities(tokens, text)

        assertEquals(1, entities.size)
        assertEquals("first_name", entities[0].label)
        assertEquals("John Doe", entities[0].text)
        assertEquals(8, entities[0].start)
        assertEquals(16, entities[0].end)
        assertEquals(0.925f, entities[0].confidence, 0.001f)
    }

    @Test
    fun decodeEntitiesSplitsAdjacentBeginningTagsWithSameLabel() {
        val text = "John Jane"
        val tokens = listOf(
            token("B-NAME", 0.91f, 0, 4),
            token("B-NAME", 0.87f, 5, 9),
        )

        val entities = TokenClassificationDecoder.decodeEntities(tokens, text)

        assertEquals(
            listOf(
                EntityPrediction("NAME", "John", 0.91f, 0, 4),
                EntityPrediction("NAME", "Jane", 0.87f, 5, 9),
            ),
            entities,
        )
    }

    @Test
    fun decodeEntitiesSplitsOnLabelChangeWithoutOutsideToken() {
        val text = "John 1970"
        val tokens = listOf(
            token("B-NAME", 0.91f, 0, 4),
            token("I-DATE", 0.88f, 5, 9),
        )

        val entities = TokenClassificationDecoder.decodeEntities(tokens, text)

        assertEquals(
            listOf(
                EntityPrediction("NAME", "John", 0.91f, 0, 4),
                EntityPrediction("DATE", "1970", 0.88f, 5, 9),
            ),
            entities,
        )
    }

    @Test
    fun decodeEntitiesHandlesOutsideTokensAndMultipleEntities() {
        val text = "John Doe 555-1234"
        val tokens = listOf(
            token("B-first_name", 0.95f, 0, 4),
            token("I-first_name", 0.90f, 5, 8),
            token("O", 0.99f, 8, 9),
            token("B-phone", 0.85f, 9, 17),
        )

        val entities = TokenClassificationDecoder.decodeEntities(tokens, text)

        assertEquals(2, entities.size)
        assertEquals("first_name", entities[0].label)
        assertEquals("John Doe", entities[0].text)
        assertEquals("phone", entities[1].label)
        assertEquals("555-1234", entities[1].text)
    }

    @Test
    fun decodeEntitiesSupportsBioesSingletonAndEndTags() {
        val text = "Dr Ada Lovelace 555-0134"
        val tokens = listOf(
            token("B-NAME", 0.90f, 3, 6),
            token("E-NAME", 0.80f, 7, 15),
            token("S-PHONE", 0.95f, 16, 24),
        )

        val entities = TokenClassificationDecoder.decodeEntities(tokens, text)

        assertEquals(
            listOf(
                EntityPrediction("NAME", "Ada Lovelace", 0.85f, 3, 15),
                EntityPrediction("PHONE", "555-0134", 0.95f, 16, 24),
            ),
            entities,
        )
    }

    @Test
    fun decodeEntitiesTreatsMismatchedEndTagAsSingleton() {
        val text = "Ada 1970"
        val tokens = listOf(
            token("B-NAME", 0.91f, 0, 3),
            token("E-DATE", 0.89f, 4, 8),
        )

        val entities = TokenClassificationDecoder.decodeEntities(tokens, text)

        assertEquals(
            listOf(
                EntityPrediction("NAME", "Ada", 0.91f, 0, 3),
                EntityPrediction("DATE", "1970", 0.89f, 4, 8),
            ),
            entities,
        )
    }

    @Test
    fun decodeEntitiesUsesSelectedAggregationStrategy() {
        val text = "John Doe"
        val tokens = listOf(
            token("B-NAME", 0.90f, 0, 4),
            token("I-NAME", 0.80f, 5, 8),
        )

        val average = TokenClassificationDecoder.decodeEntities(
            tokens,
            text,
            AggregationStrategy.AVERAGE,
        )
        val maximum = TokenClassificationDecoder.decodeEntities(tokens, text, AggregationStrategy.MAX)
        val first = TokenClassificationDecoder.decodeEntities(tokens, text, AggregationStrategy.FIRST)

        assertEquals(0.85f, average[0].confidence, 0.001f)
        assertEquals(0.90f, maximum[0].confidence, 0.001f)
        assertEquals(0.90f, first[0].confidence, 0.001f)
    }

    @Test
    fun decodeEntitiesReturnsEmptyListForEmptyAndOutsideOnlyInputs() {
        assertTrue(TokenClassificationDecoder.decodeEntities(emptyList(), "").isEmpty())

        val text = "Hello world"
        val tokens = listOf(
            token("O", 0.99f, 0, 5),
            token("O", 0.99f, 6, 11),
        )

        assertTrue(TokenClassificationDecoder.decodeEntities(tokens, text).isEmpty())
    }

    @Test
    fun repairEntitySpansExtendsTruncatedEndToWordBoundary() {
        val text = "Patient Maria Garcia"
        val entities = listOf(
            EntityPrediction("NAME", "Mari", 0.90f, 8, 12),
        )

        val repaired = TokenClassificationDecoder.repairEntitySpans(entities, text)

        assertEquals(1, repaired.size)
        assertEquals(EntityPrediction("NAME", "Maria", 0.90f, 8, 13), repaired[0])
    }

    @Test
    fun repairEntitySpansExtendsUnicodeWordLikeCharacters() {
        val text = "Patient María Garcia"
        val entities = listOf(
            EntityPrediction("NAME", "Mar", 0.90f, 8, 11),
        )

        val repaired = TokenClassificationDecoder.repairEntitySpans(entities, text)

        assertEquals(EntityPrediction("NAME", "María", 0.90f, 8, 13), repaired[0])
    }

    @Test
    fun repairEntitySpansTrimsWhitespaceAndClampsOffsets() {
        val text = "  Patient Maria  "
        val entities = listOf(
            EntityPrediction("NAME", " Patient Maria  ", 0.90f, -4, 99),
        )

        val repaired = TokenClassificationDecoder.repairEntitySpans(entities, text)

        assertEquals(EntityPrediction("NAME", "Patient Maria", 0.90f, 2, 15), repaired[0])
    }

    @Test
    fun repairEntitySpansStopsWordExtensionAfterTenCharacters() {
        val text = "Token Supercalifragilistic"
        val entities = listOf(
            EntityPrediction("WORD", "Supe", 0.80f, 6, 10),
        )

        val repaired = TokenClassificationDecoder.repairEntitySpans(entities, text)

        assertEquals(EntityPrediction("WORD", "Supercalifragi", 0.80f, 6, 20), repaired[0])
    }

    private fun token(
        label: String,
        score: Float,
        startOffset: Int,
        endOffset: Int,
    ): TokenPrediction {
        return TokenPrediction(
            labelId = 0,
            label = label,
            score = score,
            startOffset = startOffset,
            endOffset = endOffset,
        )
    }
}
