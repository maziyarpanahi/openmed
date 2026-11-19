package com.openmed.openmedkit.deid

import com.openmed.openmedkit.secrets.AndroidKeystoreSecretProvider
import com.openmed.openmedkit.secrets.FakeKeystore
import com.openmed.openmedkit.secrets.OpenMedSecretException
import com.openmed.openmedkit.secrets.OpenMedSecretFailure
import com.openmed.openmedkit.secrets.OpenMedSecretPolicy
import com.openmed.openmedkit.secrets.OpenMedSecretPurpose
import com.openmed.openmedkit.secrets.OpenMedSecretReference
import com.openmed.openmedkit.secrets.OpenMedUserAuthentication
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class DeidentifyEngineTest {
    private val keystore = FakeKeystore()
    private val provider = AndroidKeystoreSecretProvider(keystore).also {
        it.put(OpenMedSecretReference.DEIDENTIFY_HASH, SYNTHETIC_SALT).getOrThrow()
    }
    private val engine = DeidentifyEngine(provider)

    @Test
    fun providerHashesMatchTheSamePortableSaltHeldAsRawBytes() {
        val text = "MRN 12345 matched."
        val spans = listOf(spanOf(text, "12345", "medical_record_number"))

        @Suppress("DEPRECATION")
        val raw = DeidentifyEngine(SYNTHETIC_SALT).deidentify(text, spans, DeidentifyMethod.HASH)
        val held = engine.deidentify(text, spans, DeidentifyMethod.HASH)

        assertEquals(raw.redactedText, held.redactedText)
        assertEquals(raw.actions[0].span.textHash, held.actions[0].span.textHash)
        assertEquals(
            expectedDigest(SYNTHETIC_SALT, "MEDICAL_RECORD_NUMBER", "12345"),
            held.actions[0].replacement,
        )
        assertFalse(keystore.holdsMaterial("unrelated-material".toByteArray()))
    }

    @Test
    fun keylessEngineMasksWithoutDigestsAndRefusesHashing() {
        val text = "Call Jane Patient at 555-1212."
        val spans = listOf(
            spanOf(text, "Jane Patient", "name"),
            spanOf(text, "555-1212", "phone"),
        )
        val keyless = DeidentifyEngine()

        val masked = keyless.deidentify(text, spans, DeidentifyMethod.MASK)
        val refused = runCatching { keyless.deidentify(text, spans, DeidentifyMethod.HASH) }
            .exceptionOrNull()

        assertEquals("Call [NAME] at [PHONE].", masked.redactedText)
        assertTrue(masked.actions.all { it.span.textHash == null })
        assertTrue(refused is OpenMedSecretException)
        assertEquals(OpenMedSecretFailure.MISSING, (refused as OpenMedSecretException).failure)
        assertEquals("secret_missing", refused.message)
    }

    @Test
    fun strictPolicyRefusesThePublicDefaultSaltHoweverItIsSupplied() {
        val text = "MRN 12345 matched."
        val spans = listOf(spanOf(text, "12345", "medical_record_number"))
        val publicSalt = DeidentifyEngine.LEGACY_PUBLIC_HASH_SALT.toByteArray()

        @Suppress("DEPRECATION")
        val raw = runCatching { DeidentifyEngine(publicSalt) }.exceptionOrNull()
        assertEquals(OpenMedSecretFailure.PUBLIC_DEFAULT, (raw as OpenMedSecretException).failure)

        val publicProvider = AndroidKeystoreSecretProvider(FakeKeystore()).also {
            it.put(OpenMedSecretReference.DEIDENTIFY_HASH, publicSalt).getOrThrow()
        }
        val held = runCatching {
            DeidentifyEngine(publicProvider).deidentify(text, spans, DeidentifyMethod.MASK)
        }.exceptionOrNull()
        assertEquals(OpenMedSecretFailure.PUBLIC_DEFAULT, (held as OpenMedSecretException).failure)
    }

    @Test
    fun legacyPublicSaltReproducesPre40DigestsOnlyWhenExplicitlyAllowed() {
        val text = "MRN 12345 matched."
        val spans = listOf(spanOf(text, "12345", "medical_record_number"))
        val publicSalt = DeidentifyEngine.LEGACY_PUBLIC_HASH_SALT.toByteArray()
        val expected = expectedDigest(publicSalt, "MEDICAL_RECORD_NUMBER", "12345")

        @Suppress("DEPRECATION")
        val raw = DeidentifyEngine(publicSalt, HashKeyPolicy.ALLOW_LEGACY_PUBLIC_SALT)
        val publicProvider = AndroidKeystoreSecretProvider(FakeKeystore()).also {
            it.put(OpenMedSecretReference.DEIDENTIFY_HASH, publicSalt).getOrThrow()
        }
        val held = DeidentifyEngine(
            publicProvider,
            hashKeyPolicy = HashKeyPolicy.ALLOW_LEGACY_PUBLIC_SALT,
        )

        val rawResult = raw.deidentify(text, spans, DeidentifyMethod.HASH)
        val heldResult = held.deidentify(text, spans, DeidentifyMethod.HASH)
        assertEquals(expected, rawResult.actions[0].replacement)
        assertEquals(expected, heldResult.actions[0].replacement)
    }

    @Test
    fun providerFailuresFailClosedBeforeAnyOutput() {
        val text = "Jane Patient called 555-1212."
        val spans = listOf(spanOf(text, "Jane Patient", "name"))
        val lockedStore = FakeKeystore()
        val policy = OpenMedSecretPolicy(userAuthentication = OpenMedUserAuthentication(30))
        val lockedProvider = AndroidKeystoreSecretProvider(lockedStore).also {
            it.rotate(OpenMedSecretReference.DEIDENTIFY_HASH, policy).getOrThrow()
        }
        lockedStore.userAuthenticated = false

        val locked = runCatching {
            DeidentifyEngine(lockedProvider, secretPolicy = policy)
                .deidentify(text, spans, DeidentifyMethod.MASK)
        }.exceptionOrNull()
        val missing = runCatching {
            DeidentifyEngine(AndroidKeystoreSecretProvider(FakeKeystore()))
                .deidentify(text, spans, DeidentifyMethod.HASH)
        }.exceptionOrNull()

        assertEquals(OpenMedSecretFailure.LOCKED, (locked as OpenMedSecretException).failure)
        assertEquals("secret_locked", locked.message)
        assertNull(locked.cause)
        assertEquals(OpenMedSecretFailure.MISSING, (missing as OpenMedSecretException).failure)
        assertFalse(locked.toString().contains("Jane Patient"))
    }

    @Test
    fun rotationChangesDigestsOnTheNextCall() {
        val text = "MRN 12345 matched."
        val spans = listOf(spanOf(text, "12345", "medical_record_number"))

        val before = engine.deidentify(text, spans, DeidentifyMethod.HASH).redactedText
        provider.rotate(OpenMedSecretReference.DEIDENTIFY_HASH).getOrThrow()
        val after = engine.deidentify(text, spans, DeidentifyMethod.HASH).redactedText

        assertFalse(before == after)
        assertFalse(after.contains("12345"))
    }

    @Test
    fun hashKeyReferenceMustUseTheDeidentifyPurpose() {
        val error = runCatching {
            DeidentifyEngine(
                provider,
                OpenMedSecretReference(OpenMedSecretPurpose.POLICY_SECRET),
            )
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
    }

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

    private fun expectedDigest(salt: ByteArray, label: String, surface: String): String {
        val digest = Mac.getInstance("HmacSHA256").run {
            init(SecretKeySpec(salt, "HmacSHA256"))
            doFinal("$label\u0000$surface".toByteArray())
        }
        return "hmac-sha256:" + digest.joinToString("") { "%02x".format(it) }
    }

    private companion object {
        val SYNTHETIC_SALT = "synthetic-unit-test-salt".toByteArray()
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
