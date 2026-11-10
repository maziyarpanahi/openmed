package com.openmed.openmedkit.util

import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.TokenClassificationPrediction
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class NoPhiLoggingTest {
    @Test
    fun inferenceDiagnosticsContainHashesAndOffsetsButNoPlaintextIdentifiers() = runTest {
        val identifier = "Jane Patient"
        val phone = "555-1212"
        val source = "$identifier called $phone"
        val captured = mutableListOf<SafeLogEvent>()
        val previous = SafeLog.installSinkForTesting(captured::add)

        val kit = OpenMedKit(
            classifier = StaticClassifier(
                prediction(source, identifier, "PERSON"),
                prediction(source, phone, "PHONE"),
            ),
        )
        try {
            kit.analyzeText(source)
            kit.extractPii(source)
            val result = kit.deidentify(source)

            assertFalse(result.redactedText.contains(identifier))
            assertFalse(result.redactedText.contains(phone))
            assertFalse(result.actions.toString().contains(identifier))
            assertFalse(result.actions.toString().contains(phone))
        } finally {
            kit.close()
            SafeLog.installSinkForTesting(previous)
        }

        val diagnostics = captured.joinToString("\n")
        assertTrue(captured.isNotEmpty())
        assertFalse(diagnostics.contains(identifier))
        assertFalse(diagnostics.contains(phone))
        assertTrue(diagnostics.contains(sha256Hex(identifier)))
        assertTrue(diagnostics.contains(sha256Hex(phone)))
        assertTrue(captured.flatMap { it.spans }.all { it.start >= 0 && it.end > it.start })
    }

    @Test
    fun entityDescriptionCannotAccidentallyExposeItsSurfaceText() {
        val identifier = "123-45-6789"
        val prediction = EntityPrediction("SSN", identifier, 0.99f, 4, 15)

        assertFalse(prediction.toString().contains(identifier))
        assertTrue(prediction.toString().contains(sha256Hex(identifier)))
    }

    @Test
    fun productionKotlinSourcesDoNotBypassSafeLog() {
        val sourceRoot = locateMainKotlinSources()
        val forbidden = Regex(
            """android\.util\.Log|java\.util\.logging|Timber\.|""" +
                """\bLog\.(v|d|i|w|e|wtf)\s*\(|""" +
                """\bprintln\s*\(|\bprint\s*\(|System\.(out|err)""",
        )
        val violations = sourceRoot.walkTopDown()
            .filter { it.isFile && it.extension == "kt" && it.name != "SafeLog.kt" }
            .flatMap { file ->
                file.readLines().asSequence().mapIndexedNotNull { index, line ->
                    if (forbidden.containsMatchIn(line)) {
                        "${file.relativeTo(sourceRoot)}:${index + 1}"
                    } else {
                        null
                    }
                }
            }
            .toList()

        assertEquals(
            emptyList(),
            violations,
            "production logging must route through SafeLog",
        )
    }

    private fun locateMainKotlinSources(): File = File(
        requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
            "openmedkit.moduleDirectory test property is missing"
        },
        "src/main/kotlin",
    ).also {
        check(it.isDirectory) { "cannot locate OpenMedKit Android Kotlin sources" }
    }

    private fun prediction(
        source: String,
        surface: String,
        label: String,
    ): TokenClassificationPrediction {
        val start = source.indexOf(surface)
        return TokenClassificationPrediction(
            label = label,
            text = surface,
            confidence = 0.99f,
            start = start,
            end = start + surface.length,
        )
    }

    private class StaticClassifier(
        vararg predictions: TokenClassificationPrediction,
    ) : OnnxTokenClassifier {
        private val predictions = predictions.toList()

        override suspend fun predict(text: String): List<TokenClassificationPrediction> =
            predictions
    }
}
