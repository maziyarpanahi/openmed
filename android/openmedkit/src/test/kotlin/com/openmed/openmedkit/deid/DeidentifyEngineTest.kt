package com.openmed.openmedkit.deid

import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DeidentifyEngineTest {
    private val engine = DeidentifyEngine("unit-test-salt".toByteArray())

    @Test
    fun maskUsesConsistentPlaceholdersForRepeatedIdentifiers() {
        val text = "John Doe emailed john@example.com and John Doe replied."
        val spans = listOf(
            spanOf(text, "John Doe", "name"),
            spanOf(text, "john@example.com", "email"),
            spanOf(text, "John Doe", "name", fromIndex = text.lastIndexOf("John Doe")),
        )

        val result = engine.deidentify(text, spans, DeidentifyMethod.MASK)

        assertEquals(
            "[NAME] emailed [EMAIL] and [NAME] replied.",
            result.redactedText,
        )
        assertEquals(listOf("[NAME]", "[EMAIL]", "[NAME]"), replacements(result))
        assertFalse(result.toString().contains("John Doe"))
        assertFalse(result.toString().contains("john@example.com"))
    }

    @Test
    fun removeDeletesSpansWithoutChangingUnaffectedText() {
        val text = "Call John at 555-1212."
        val result = engine.deidentify(
            text,
            listOf(
                spanOf(text, "John", "name"),
                spanOf(text, "555-1212", "phone"),
            ),
            DeidentifyMethod.REMOVE,
        )

        assertEquals("Call  at .", result.redactedText)
        assertEquals(listOf("", ""), replacements(result))
    }

    @Test
    fun replaceUsesPerLabelSurrogateTokens() {
        val text = "John met Mary. John left."
        val spans = listOf(
            spanOf(text, "John", "name"),
            spanOf(text, "Mary", "name"),
            spanOf(text, "John", "name", fromIndex = text.lastIndexOf("John")),
        )

        val result = engine.deidentify(text, spans, DeidentifyMethod.REPLACE)

        assertEquals(
            "NAME_SURROGATE met NAME_SURROGATE_2. NAME_SURROGATE left.",
            result.redactedText,
        )
        assertEquals(
            listOf("NAME_SURROGATE", "NAME_SURROGATE_2", "NAME_SURROGATE"),
            replacements(result),
        )
    }

    @Test
    fun hashUsesStableSaltedDigestWithoutPlaintext() {
        val text = "MRN 12345 matched MRN 12345."
        val spans = listOf(
            spanOf(text, "12345", "medical_record_number"),
            spanOf(text, "12345", "medical_record_number", fromIndex = text.lastIndexOf("12345")),
        )

        val result = engine.deidentify(text, spans, DeidentifyMethod.HASH)
        val first = result.actions[0].replacement
        val second = result.actions[1].replacement

        assertEquals(first, second)
        assertTrue(first.matches(Regex("hmac-sha256:[0-9a-f]{64}")))
        assertFalse(result.redactedText.contains("12345"))
        assertFalse(result.toString().contains("12345"))
    }

    @Test
    fun labelActionMapCanApplyMixedMethods() {
        val text = "John called 555-1212 about MRN 12345."
        val result = engine.deidentify(
            text,
            listOf(
                spanOf(text, "John", "name"),
                spanOf(text, "555-1212", "phone"),
                spanOf(text, "12345", "medical_record_number"),
            ),
            mapOf(
                "NAME" to DeidentifyMethod.MASK,
                "PHONE" to DeidentifyMethod.REMOVE,
                "MEDICAL_RECORD_NUMBER" to DeidentifyMethod.HASH,
            ),
        )

        assertTrue(result.redactedText.startsWith("[NAME] called  about MRN hmac-sha256:"))
        assertFalse(result.redactedText.contains("John"))
        assertFalse(result.redactedText.contains("555-1212"))
        assertFalse(result.redactedText.contains("12345"))
    }

    @Test
    fun overlappingSpansAreCollapsedAndAdjacentSpansRemainSeparate() {
        val text = "JohnDoe555"
        val spans = listOf(
            OpenMedSpan(start = 0, end = 7, canonicalLabel = "name", score = 0.95),
            OpenMedSpan(start = 4, end = 7, canonicalLabel = "last_name", score = 0.80),
            OpenMedSpan(start = 7, end = 10, canonicalLabel = "phone", score = 0.90),
        )

        val result = engine.deidentify(text, spans, DeidentifyMethod.MASK)

        assertEquals("[NAME][PHONE]", result.redactedText)
        assertEquals(listOf(0 to 7, 7 to 10), result.actions.map { it.span.start to it.span.end })
    }

    @Test
    fun outputOffsetsReflectEarlierReplacementDeltas() {
        val text = "A John B 555-1212 C"
        val spans = listOf(
            spanOf(text, "John", "name"),
            spanOf(text, "555-1212", "phone"),
        )

        val result = engine.deidentify(text, spans, DeidentifyMethod.MASK)

        assertEquals("A [NAME] B [PHONE] C", result.redactedText)
        assertEquals(2, result.actions[0].span.start)
        assertEquals(6, result.actions[0].span.end)
        assertEquals(2, result.actions[0].outputStart)
        assertEquals(8, result.actions[0].outputEnd)
        assertEquals(result.redactedText.indexOf("[PHONE]"), result.actions[1].outputStart)
        assertEquals(result.actions[1].outputStart + "[PHONE]".length, result.actions[1].outputEnd)
    }

    @Test
    fun deidentificationDoesNotEmitRawIdentifiersToLogs() {
        val text = "Jane Patient called 555-1212."
        val logger = Logger.getLogger("")
        val handler = RecordingHandler()
        val oldOut = System.out
        val oldErr = System.err
        val stdout = ByteArrayOutputStream()
        val stderr = ByteArrayOutputStream()

        logger.addHandler(handler)
        logger.level = Level.ALL
        System.setOut(PrintStream(stdout))
        System.setErr(PrintStream(stderr))
        try {
            val result = engine.deidentify(
                text,
                listOf(
                    spanOf(text, "Jane Patient", "name"),
                    spanOf(text, "555-1212", "phone"),
                ),
                DeidentifyMethod.HASH,
            )

            assertFalse(result.toString().contains("Jane Patient"))
            assertFalse(result.toString().contains("555-1212"))
        } finally {
            System.setOut(oldOut)
            System.setErr(oldErr)
            logger.removeHandler(handler)
        }

        val logOutput = handler.messages.joinToString("\n")
        assertFalse(logOutput.contains("Jane Patient"))
        assertFalse(logOutput.contains("555-1212"))
        assertFalse(stdout.toString().contains("Jane Patient"))
        assertFalse(stderr.toString().contains("555-1212"))
    }

    private fun spanOf(
        text: String,
        surface: String,
        canonicalLabel: String,
        fromIndex: Int = 0,
    ): OpenMedSpan {
        val start = text.indexOf(surface, fromIndex)
        require(start >= 0) { "surface not found: $surface" }
        return OpenMedSpan(
            start = start,
            end = start + surface.length,
            canonicalLabel = canonicalLabel,
            score = 0.99,
        )
    }

    private fun replacements(result: DeidentifyResult): List<String> {
        return result.actions.map { it.replacement }
    }

    private class RecordingHandler : Handler() {
        val messages = mutableListOf<String>()

        override fun publish(record: LogRecord) {
            messages += record.message.orEmpty()
        }

        override fun flush() = Unit

        override fun close() = Unit
    }
}
