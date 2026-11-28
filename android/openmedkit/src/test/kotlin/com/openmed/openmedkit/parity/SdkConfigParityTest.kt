package com.openmed.openmedkit.parity

import com.openmed.openmedkit.OpenMed
import com.openmed.openmedkit.OpenMedSdkConfig
import com.openmed.openmedkit.OpenMedSdkConfigException
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelFile
import com.openmed.openmedkit.download.HuggingFaceModelInfo
import com.openmed.openmedkit.download.ModelDownloadException
import com.openmed.openmedkit.download.ModelDownloadLogger
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.download.ModelIntegrity
import java.io.File
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Replays `tests/fixtures/parity/sdk_config_v1.json`, generated from the Python
 * reference `openmed.core.sdk_config`, through [OpenMedSdkConfig], and checks
 * how `OpenMed` and `ModelDownloader` apply a parsed configuration offline.
 */
class SdkConfigParityTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val fixture: JsonObject = Json.parseToJsonElement(
        requireNotNull(javaClass.classLoader?.getResourceAsStream(FIXTURE)) {
            "shared $FIXTURE was not synced into Android test resources"
        }.use { String(it.readBytes(), Charsets.UTF_8) },
    ).jsonObject

    @Test
    fun fixtureDeclaresTheKotlinVersionAndDefaults() {
        assertEquals(OpenMedSdkConfig.VERSION, fixture.getValue("sdk_config_version").jsonPrimitive.int)
        assertTrue(fixture.getValue("synthetic").jsonPrimitive.boolean)
        assertEquals(fixture.getValue("defaults").jsonObject, project(OpenMedSdkConfig.DEFAULT))
        assertEquals(fixture.getValue("defaults").jsonObject, project(OpenMedSdkConfig.parse("{}")))
    }

    @Test
    fun validVectorsResolveToTheSharedForm() {
        val cases = fixture.getValue("valid").jsonArray.map { it.jsonObject }
        assertTrue(cases.size >= 10)
        for (case in cases) {
            val id = case.getValue("id").jsonPrimitive.content
            val text = case.getValue("config").toString()

            assertEquals(emptyList(), OpenMedSdkConfig.validate(text).map { it.toString() }, id)
            assertEquals(case.getValue("resolved").jsonObject, project(OpenMedSdkConfig.parse(text)), id)
        }
    }

    @Test
    fun invalidVectorsFailWithTheSharedOrderedIssues() {
        val codes = fixture.getValue("issue_codes").jsonArray.map { it.jsonPrimitive.content }.toSet()
        val seen = mutableSetOf<String>()
        for (case in fixture.getValue("invalid").jsonArray.map { it.jsonObject }) {
            val id = case.getValue("id").jsonPrimitive.content
            val text = case.getValue("config").toString()
            val expected = case.getValue("errors").jsonArray.map {
                val issue = it.jsonObject
                issue.getValue("code").jsonPrimitive.content to
                    issue.getValue("field").takeUnless { field -> field == JsonNull }?.jsonPrimitive?.content
            }

            assertEquals(expected, OpenMedSdkConfig.validate(text).map { it.code to it.field }, id)
            val error = assertFailsWith<OpenMedSdkConfigException>(id) { OpenMedSdkConfig.parse(text) }
            assertEquals(expected, error.issues.map { it.code to it.field }, id)
            seen += expected.map { it.first }
        }
        assertEquals(codes, seen)
    }

    @Test
    fun issuesAndErrorsNeverEchoInputValuesOrKeys() {
        for (text in listOf(
            "not json $SENTINEL",
            """{"$SENTINEL": 1}""",
            """{"model_id": "$SENTINEL", "policy_profile": "$SENTINEL", "local_only": "$SENTINEL"}""",
        )) {
            val error = assertFailsWith<OpenMedSdkConfigException> { OpenMedSdkConfig.parse(text) }
            assertFalse(SENTINEL in error.message.orEmpty(), text)
            assertFalse(SENTINEL in error.issues.joinToString(), text)
        }
        assertEquals(
            listOf("config_not_object"),
            OpenMedSdkConfig.validate("{\"local_only\": ").map { it.toString() },
        )
        // kotlinx reads unquoted tokens as literals; text that is not JSON is
        // config_not_object, as in Swift, and never a valid configuration.
        for (text in listOf(
            """{"confidence_threshold": 0.5f}""",
            """{"confidence_threshold": 0x1p-1}""",
            """{"confidence_threshold": NaN}""",
            """{"local_only": yes}""",
            """{"model_id": synthetic}""",
            """{"extra": [1, 2, nope]}""",
            "{\"x\": " + "[".repeat(100_000) + "]".repeat(100_000) + "}",
        )) {
            assertEquals(listOf("config_not_object"), OpenMedSdkConfig.validate(text).map { it.toString() }, text.take(40))
            assertFailsWith<OpenMedSdkConfigException> { OpenMedSdkConfig.parse(text) }
        }
        assertEquals(
            "invalid SDK configuration: unknown_field, local_only:wrong_type",
            assertFailsWith<OpenMedSdkConfigException> {
                OpenMedSdkConfig.parse("""{"extra": 1, "local_only": "yes"}""")
            }.message,
        )
    }

    @Test
    fun localOnlyFailsClosedBeforeTheModelClientIsCalled() {
        val files = mapOf("model.onnx" to "weights")
        val entry = entry(files)
        val cache = ModelCache(temporaryFolder.newFolder("cache"))
        val client = CountingClient(files)

        for (config in listOf(OpenMedSdkConfig.DEFAULT, OpenMedSdkConfig.parse("""{"local_only": true}"""))) {
            assertFailsWith<ModelDownloadException> { ModelDownloader(cache, config, client).download(entry) }
        }
        assertEquals(0, client.calls)
        assertFalse(cache.isAvailable(entry))

        // A network opt-in downloads; afterwards local_only serves the cached copy.
        ModelDownloader(cache, OpenMedSdkConfig.parse("""{"local_only": false}"""), client).download(entry)
        assertEquals(2, client.calls)
        val cached = ModelDownloader(cache, OpenMedSdkConfig.DEFAULT, client).download(entry)
        assertTrue(cached.fromCache)
        assertEquals(2, client.calls)
    }

    @Test
    fun configuredRevisionIsPinnedForTheConfiguredModelOnly() {
        val files = mapOf("model.onnx" to "weights")
        val entry = entry(files)
        val pinned = OpenMedSdkConfig.parse(
            """{"local_only": false, "model_id": "$REPO_ID", "model_revision": "$REVISION"}""",
        )
        val other = OpenMedSdkConfig.parse(
            """{"local_only": false, "model_id": "synthetic-org/other", "model_revision": "$REVISION"}""",
        )

        val pinnedClient = CountingClient(files)
        ModelDownloader(ModelCache(temporaryFolder.newFolder("a")), pinned, pinnedClient).download(entry)
        assertEquals(listOf<String?>(REVISION), pinnedClient.revisions)

        val otherClient = CountingClient(files)
        ModelDownloader(ModelCache(temporaryFolder.newFolder("b")), other, otherClient).download(entry)
        assertEquals(listOf<String?>(null), otherClient.revisions)

        val explicitClient = CountingClient(files)
        ModelDownloader(ModelCache(temporaryFolder.newFolder("c")), pinned, explicitClient).download(entry, "main")
        assertEquals(listOf<String?>("main"), explicitClient.revisions)
    }

    @Test
    fun existingConstructorsStillCompileAndConfigSetsTheDefaultThreshold() {
        val cache = ModelCache(temporaryFolder.newFolder("cache"))
        val client = CountingClient(emptyMap())
        listOf(
            ModelDownloader(cache),
            ModelDownloader(cache, client),
            ModelDownloader(cache, client, ModelDownloadLogger.NONE),
        ).forEach { assertTrue(it is ModelDownloader) }

        val text = "Patient: Jane Roe\nMRN-12345"
        assertEquals(OpenMed().extractPII(text), OpenMed(null).extractPII(text))
        val strict = OpenMed(null, OpenMedSdkConfig.parse("""{"confidence_threshold": 0.25}"""))
        assertEquals(0.25f, defaultThreshold(strict))
        assertEquals(0.5f, defaultThreshold(OpenMed()))
        assertEquals(OpenMed().extractPII(text), strict.extractPII(text))
    }

    private fun project(config: OpenMedSdkConfig): JsonObject = Json.parseToJsonElement(
        buildString {
            append("{\"confidence_threshold\":").append(config.confidenceThreshold)
            append(",\"local_only\":").append(config.localOnly)
            append(",\"model_id\":").append(quoted(config.modelId))
            append(",\"model_revision\":").append(quoted(config.modelRevision))
            append(",\"onnx_variant\":").append(quoted(config.onnxVariant))
            append(",\"policy_profile\":").append(quoted(config.policyProfile)).append('}')
        },
    ).jsonObject

    /** Compares the threshold by value, so the fixture's 0 and 0.0 resolve alike. */
    private fun JsonObject.normalized(): JsonObject = JsonObject(
        mapValues { (key, value) ->
            if (key == "confidence_threshold") {
                kotlinx.serialization.json.JsonPrimitive(value.jsonPrimitive.double)
            } else {
                value
            }
        },
    )

    private fun assertEquals(expected: JsonObject, actual: JsonObject, message: String? = null) =
        kotlin.test.assertEquals(expected.normalized(), actual.normalized(), message)

    private fun quoted(value: String?): String = if (value == null) "null" else "\"$value\""

    private fun defaultThreshold(openMed: OpenMed): Float =
        OpenMed::class.java.getDeclaredField("defaultThreshold").apply { isAccessible = true }.getFloat(openMed)

    private fun entry(files: Map<String, String>): ModelCatalogEntry = ModelCatalogEntry(
        repoId = REPO_ID,
        formats = listOf("onnx"),
        tier = "tiny",
        paramCount = 1,
        languages = listOf("en"),
        license = "apache-2.0",
        reproducibilityHash = ModelIntegrity.reproducibilityHash(
            repoId = REPO_ID,
            revisionSha = REVISION_SHA,
            released = RELEASED,
            siblings = files.keys.toList(),
        ),
    )

    private class CountingClient(private val files: Map<String, String>) : HuggingFaceModelClient {
        var calls = 0
        val revisions = mutableListOf<String?>()

        override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo {
            calls += 1
            revisions += revision
            return HuggingFaceModelInfo(
                repoId = repoId,
                revisionSha = REVISION_SHA,
                released = RELEASED,
                files = files.map { (path, content) ->
                    HuggingFaceModelFile(
                        path = path,
                        sizeBytes = content.toByteArray().size.toLong(),
                        sha256 = MessageDigest.getInstance("SHA-256").digest(content.toByteArray())
                            .joinToString("") { "%02x".format(it) },
                    )
                },
            )
        }

        override fun downloadFile(repoId: String, path: String, revision: String, destination: File) {
            calls += 1
            destination.writeText(files.getValue(path))
        }
    }

    private companion object {
        const val FIXTURE = "sdk_config_v1.json"
        const val SENTINEL = "SYNTH_PHI_SENTINEL Jane Roe"
        const val REPO_ID = "synthetic-org/synthetic-pii-model-onnx"
        const val REVISION = "0123456789abcdef0123456789abcdef01234567"
        const val REVISION_SHA = "abc123"
        const val RELEASED = "2026-01-01"
    }
}
