package com.openmed.openmedkit.parity

import com.openmed.openmedkit.DeidentifiedSpanAction
import com.openmed.openmedkit.DeidentifyEngine
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OpenMed
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.policy.PolicyProfiles
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.security.MessageDigest
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Runs the repository-root Swift-Kotlin golden in
 * `fixtures/parity/expected_spans.json` through the public Android extraction
 * and policy de-identification path.
 *
 * The Swift counterpart is `SpanEquivalenceParityTests` in
 * `swift/OpenMedKit/Tests/OpenMedKitTests/`. Robolectric supplies Android ICU
 * so grapheme snapping follows the production segmenter. Failure messages
 * carry case identifiers, labels, and offsets only.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class SpanEquivalenceTest {
    @Test
    fun fixtureSchemaAndDigestAreCurrent() {
        val manifest = readJson(MANIFEST_FILE)
        val fixture = readJson(SPANS_FILE)

        validate(manifest, fixture, ::readFixtureBytes)
        assertEquals(
            setOf("python", "swift", "kotlin"),
            manifest.getJSONArray("consumers").objects()
                .map { it.getString("surface") }
                .toSet(),
        )
        assertTrue(
            manifest.getJSONArray("consumers").objects().any {
                it.getString("test").endsWith("SpanEquivalenceTest.kt") &&
                    it.getString("gate").contains("SpanEquivalenceTest")
            },
        )
        assertTrue(fixture.getJSONArray("cases").length() >= 4)
    }

    @Test
    fun androidExtractionMatchesSharedSpans() {
        val openMed = OpenMed()

        for (case in loadValidatedCases()) {
            val extracted = openMed.extractPII(case.text, useSmartMerging = false)
            assertEquals(
                emptyList<String>(),
                spanDifferences(case.id, case.spans.map { it.scalar }, extracted.map(::scalarSpan)),
            )
            extracted.forEachIndexed { index, entity ->
                assertTrue(
                    entity.text ==
                        UnicodeOffsetContract.substring(case.text, entity.start, entity.end),
                    "case=${case.id} span=$index text does not match offsets",
                )
            }

            val chunked = openMed.extractPIIChunked(case.text, useSmartMerging = false)
            assertEquals(
                emptyList<String>(),
                spanDifferences(case.id, case.spans.map { it.scalar }, chunked.map(::scalarSpan)),
            )
        }
    }

    @Test
    fun unicodeScalarUtf8AndUtf16ConversionsMatchFixture() {
        var sawSurrogatePairShift = false
        var sawMultibyteUtf8Shift = false

        for (case in loadValidatedCases()) {
            case.spans.forEachIndexed { index, span ->
                val tag = "case=${case.id} span=$index"
                listOf(
                    Triple(span.scalar.start, span.utf8Start, span.utf16Start),
                    Triple(span.scalar.end, span.utf8End, span.utf16End),
                ).forEach { (scalar, utf8, utf16) ->
                    assertEquals(
                        utf16,
                        UnicodeOffsetContract.scalarToUtf16Index(case.text, scalar),
                        tag,
                    )
                    assertEquals(
                        scalar,
                        UnicodeOffsetContract.utf16ToScalarOffset(case.text, utf16),
                        tag,
                    )
                    assertEquals(
                        utf8,
                        case.text.substring(0, utf16).toByteArray(Charsets.UTF_8).size,
                        tag,
                    )
                    assertTrue(UnicodeOffsetContract.isCodePointBoundary(case.text, utf16), tag)
                }
                sawSurrogatePairShift =
                    sawSurrogatePairShift || span.utf16Start != span.scalar.start
                sawMultibyteUtf8Shift = sawMultibyteUtf8Shift || span.utf8Start != span.utf16Start
            }
        }

        assertTrue(sawSurrogatePairShift, "fixture lacks a UTF-16 surrogate shift")
        assertTrue(sawMultibyteUtf8Shift, "fixture lacks a multibyte UTF-8 shift")
    }

    @Test
    fun policyDeidentificationMatchesSharedActionsAndBytes() {
        val openMed = OpenMed()
        val engine = DeidentifyEngine()

        for (case in loadValidatedCases()) {
            val entities = openMed.extractPII(case.text, useSmartMerging = false)
            val result = engine.deidentify(case.text, entities, PolicyProfiles.load(case.policy))

            assertEquals(case.policy, result.policyName, case.id)
            assertEquals(
                case.actions,
                result.actions.map(::actionRecord),
                "case=${case.id} policy actions differ",
            )
            assertNull(
                utf8Difference(case.expectedDeidentifiedText, result.redactedText),
                "case=${case.id} de-identified UTF-8 bytes differ",
            )
        }
    }

    @Test
    fun changedExpectedSpanFailsParityGate() {
        val case = loadValidatedCases().single { it.id == "emoji_and_combining_marks" }
        val actual = OpenMed().extractPII(case.text, useSmartMerging = false).map(::scalarSpan)

        val expected = case.spans.map { it.scalar }
        assertEquals(emptyList<String>(), spanDifferences(case.id, expected, actual))

        val shifted = expected.toMutableList()
        shifted[2] = shifted[2].copy(end = shifted[2].end + 1)
        assertEquals(
            listOf(
                "case=${case.id} span=2 end expected=${expected[2].end + 1} " +
                    "actual=${expected[2].end} delta=-1",
            ),
            spanDifferences(case.id, shifted, actual),
        )

        val relabeled = expected.toMutableList()
        relabeled[0] = relabeled[0].copy(label = "first_name")
        assertEquals(
            listOf(
                "case=${case.id} span=0 label expected=first_name " +
                    "actual=${expected[0].label}",
            ),
            spanDifferences(case.id, relabeled, actual),
        )

        assertEquals(
            listOf("case=${case.id} count expected=${expected.size - 1} actual=${actual.size}"),
            spanDifferences(case.id, expected.dropLast(1), actual),
        )

        val changedBytes = utf8Difference(
            case.expectedDeidentifiedText + " ",
            case.expectedDeidentifiedText,
        )
        assertTrue(changedBytes?.startsWith("first_differing_utf8_byte=") == true)
    }

    @Test
    fun changedDigestOrFixtureBytesFailValidation() {
        val manifest = readJson(MANIFEST_FILE)
        val fixture = readJson(SPANS_FILE)

        val staleDigest = JSONObject(manifest.toString())
        staleDigest.getJSONArray("files").getJSONObject(0).put("sha256", "0".repeat(64))
        val digestError = assertFailsWith<FixtureException> {
            validate(staleDigest, fixture, ::readFixtureBytes)
        }
        assertEquals("digest_mismatch", digestError.code)

        val missingFile = JSONObject(manifest.toString())
        val files = missingFile.getJSONArray("files").objects()
            .filter { it.getString("path") != NOTE_FILE }
        missingFile.put("files", JSONArray(files))
        val missingError = assertFailsWith<FixtureException> {
            validate(missingFile, fixture, ::readFixtureBytes)
        }
        assertEquals("undigested_file", missingError.code)

        // A changed expected span must fail the digest gate before parity runs.
        val end = fixture.getJSONArray("cases").getJSONObject(0)
            .getJSONArray("spans").getJSONObject(0).getInt("end")
        val original = String(readFixtureBytes(SPANS_FILE), Charsets.UTF_8)
        val changed = original.replace("\"end\": $end,", "\"end\": ${end + 1},")
        assertTrue(changed != original, "negative control did not change the fixture")
        val changedError = assertFailsWith<FixtureException> {
            validate(manifest, JSONObject(changed)) { name ->
                if (name == SPANS_FILE) {
                    changed.toByteArray(Charsets.UTF_8)
                } else {
                    readFixtureBytes(name)
                }
            }
        }
        assertEquals("digest_mismatch", changedError.code)

        val staleVersion = JSONObject(fixture.toString()).put("version", 1)
        val versionError = assertFailsWith<FixtureException> {
            validate(manifest, staleVersion, ::readFixtureBytes)
        }
        assertEquals("fixture_unsupported", versionError.code)
    }

    private fun loadValidatedCases(): List<FixtureCase> {
        val fixture = readJson(SPANS_FILE)
        validate(readJson(MANIFEST_FILE), fixture, ::readFixtureBytes)
        return fixture.getJSONArray("cases").objects().map { case ->
            FixtureCase(
                id = case.getString("id"),
                policy = case.getString("policy"),
                text = if (case.has("text")) {
                    case.getString("text")
                } else {
                    String(readFixtureBytes(case.getString("text_file")), Charsets.UTF_8)
                },
                spans = case.getJSONArray("spans").objects().map { span ->
                    FixtureSpan(
                        scalar = ScalarSpan(
                            label = span.getString("label"),
                            start = span.getInt("start"),
                            end = span.getInt("end"),
                        ),
                        utf8Start = span.getInt("utf8_start"),
                        utf8End = span.getInt("utf8_end"),
                        utf16Start = span.getInt("utf16_start"),
                        utf16End = span.getInt("utf16_end"),
                    )
                },
                actions = case.getJSONArray("actions").objects().map { action ->
                    ActionRecord(
                        label = action.getString("label"),
                        canonicalLabel = action.getString("canonical_label"),
                        action = action.getString("action"),
                        start = action.getInt("start"),
                        end = action.getInt("end"),
                        replacement = if (action.isNull("replacement")) {
                            null
                        } else {
                            action.getString("replacement")
                        },
                    )
                },
                expectedDeidentifiedText = case.getString("expected_deidentified_text"),
            )
        }
    }

    private fun validate(
        manifest: JSONObject,
        fixture: JSONObject,
        bytesFor: (String) -> ByteArray,
    ) {
        val supported = manifest.getString("schema") == MANIFEST_SCHEMA &&
            manifest.getInt("version") == 1 &&
            manifest.getString("algorithm") == "sha256" &&
            fixture.getString("schema") == FIXTURE_SCHEMA &&
            fixture.getInt("version") == FIXTURE_VERSION &&
            fixture.getString("offset_unit") == "unicode_scalar" &&
            manifest.getString("fixture_schema") == fixture.getString("schema") &&
            manifest.getInt("fixture_version") == fixture.getInt("version")
        if (!supported) {
            throw FixtureException("fixture_unsupported")
        }

        val required = sortedSetOf(SPANS_FILE)
        for (case in fixture.getJSONArray("cases").objects()) {
            if (case.has("text") == case.has("text_file")) {
                throw FixtureException("case_text_source_invalid", case.getString("id"))
            }
            if (case.has("text_file")) {
                required += case.getString("text_file")
            }
        }
        val files = manifest.getJSONArray("files").objects()
        val digested = files.map { it.getString("path") }.toSet()
        required.firstOrNull { it !in digested }?.let {
            throw FixtureException("undigested_file", it)
        }
        for (file in files) {
            val path = file.getString("path")
            if (sha256Hex(bytesFor(path)) != file.getString("sha256")) {
                throw FixtureException("digest_mismatch", path)
            }
        }
    }

    private fun spanDifferences(
        caseId: String,
        expected: List<ScalarSpan>,
        actual: List<ScalarSpan>,
    ): List<String> {
        if (expected.size != actual.size) {
            return listOf("case=$caseId count expected=${expected.size} actual=${actual.size}")
        }
        val differences = mutableListOf<String>()
        expected.zip(actual).forEachIndexed { index, (want, got) ->
            val prefix = "case=$caseId span=$index"
            if (want.label != got.label) {
                differences += "$prefix label expected=${want.label} actual=${got.label}"
            }
            listOf(
                Triple("start", want.start, got.start),
                Triple("end", want.end, got.end),
            ).filter { it.second != it.third }.forEach { (field, wantOffset, gotOffset) ->
                differences += "$prefix $field expected=$wantOffset actual=$gotOffset " +
                    "delta=${gotOffset - wantOffset}"
            }
        }
        return differences
    }

    private fun utf8Difference(expected: String, actual: String): String? {
        val lhs = expected.toByteArray(Charsets.UTF_8)
        val rhs = actual.toByteArray(Charsets.UTF_8)
        if (lhs.contentEquals(rhs)) {
            return null
        }
        val offset = lhs.indices.firstOrNull { it >= rhs.size || lhs[it] != rhs[it] }
            ?: minOf(lhs.size, rhs.size)
        return "first_differing_utf8_byte=$offset expected_bytes=${lhs.size} " +
            "actual_bytes=${rhs.size}"
    }

    private fun readJson(name: String): JSONObject =
        JSONObject(String(readFixtureBytes(name), Charsets.UTF_8))

    private fun readFixtureBytes(name: String): ByteArray {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) {
            "shared fixtures/parity/$name was not registered as an Android test resource"
        }
        return stream.use { it.readBytes() }
    }

    private fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun scalarSpan(entity: EntityPrediction) =
        ScalarSpan(label = entity.label, start = entity.start, end = entity.end)

    private fun actionRecord(action: DeidentifiedSpanAction) = ActionRecord(
        label = action.label,
        canonicalLabel = action.canonicalLabel,
        action = action.action.wireName,
        start = action.start,
        end = action.end,
        replacement = action.replacement,
    )

    private fun JSONArray.objects(): List<JSONObject> =
        (0 until length()).map(::getJSONObject)

    private class FixtureException(
        val code: String,
        detail: String? = null,
    ) : IllegalStateException(listOfNotNull(code, detail).joinToString(": "))

    private data class ScalarSpan(
        val label: String,
        val start: Int,
        val end: Int,
    )

    private data class FixtureSpan(
        val scalar: ScalarSpan,
        val utf8Start: Int,
        val utf8End: Int,
        val utf16Start: Int,
        val utf16End: Int,
    )

    /** Policy action projection compared across kits: no confidence, no text. */
    private data class ActionRecord(
        val label: String,
        val canonicalLabel: String,
        val action: String,
        val start: Int,
        val end: Int,
        val replacement: String?,
    )

    private data class FixtureCase(
        val id: String,
        val policy: String,
        val text: String,
        val spans: List<FixtureSpan>,
        val actions: List<ActionRecord>,
        val expectedDeidentifiedText: String,
    )

    private companion object {
        const val SPANS_FILE = "expected_spans.json"
        const val NOTE_FILE = "synthetic_clinical_note.txt"
        const val MANIFEST_FILE = "span_parity_manifest.json"
        const val FIXTURE_SCHEMA = "openmed.native_span_parity"
        const val FIXTURE_VERSION = 2
        const val MANIFEST_SCHEMA = "openmed.native_span_parity_manifest"
    }
}
