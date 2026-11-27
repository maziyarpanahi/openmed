package com.openmed.openmedkit.parity

import com.openmed.openmedkit.deid.DeidentifyEngine
import com.openmed.openmedkit.OpenMedCodedError
import com.openmed.openmedkit.OpenMedErrorCode
import com.openmed.openmedkit.OpenMedSpan
import com.openmed.openmedkit.artifact.ArtifactAcquisition
import com.openmed.openmedkit.artifact.ArtifactAcquisitionReason
import com.openmed.openmedkit.artifact.ArtifactAcquisitionRequest
import com.openmed.openmedkit.artifact.ArtifactAcquisitionResult
import com.openmed.openmedkit.artifact.ArtifactBackendCapability
import com.openmed.openmedkit.artifact.ArtifactFormat
import com.openmed.openmedkit.artifact.ArtifactNetworkConsent
import com.openmed.openmedkit.artifact.ArtifactRef
import com.openmed.openmedkit.artifact.ArtifactRuntime
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelInfo
import com.openmed.openmedkit.download.ModelDownloadException
import com.openmed.openmedkit.download.ModelIntegrityException
import com.openmed.openmedkit.onnx.InferenceError
import com.openmed.openmedkit.policy.PolicyDetection
import com.openmed.openmedkit.policy.PolicyEngine
import com.openmed.openmedkit.policy.PolicyProfileException
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Replays `tests/fixtures/parity/error_codes_v1.json` through the Android
 * OpenMedKit: every scenario whose Kotlin cell is `required` must surface the
 * scenario's portable registry code through [OpenMedErrorCode], offline and
 * without a model. Failure messages carry codes, field paths and offsets only.
 */
class ErrorCodeParityTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val fixture: JsonObject = Json.parseToJsonElement(
        requireNotNull(javaClass.classLoader?.getResourceAsStream(FIXTURE)) {
            "shared $FIXTURE was not synced into Android test resources"
        }.use { String(it.readBytes(), Charsets.UTF_8) },
    ).jsonObject
    private val scenarios = fixture.getValue("scenarios").jsonArray.map { it.jsonObject }

    @Test
    fun fixtureCodesAreTheKotlinEnum() {
        assertEquals("openmed.error_code_vectors", fixture.getValue("schema").jsonPrimitive.content)
        assertEquals(1, fixture.getValue("schema_version").jsonPrimitive.int)
        assertEquals(
            fixture.getValue("codes").jsonArray.map {
                val code = it.jsonObject
                code.getValue("code").jsonPrimitive.content to code.getValue("retryable").jsonPrimitive.boolean
            },
            OpenMedErrorCode.values().map { it.code to it.retryable },
        )
    }

    @Test
    fun everyRequiredKotlinScenarioRaisesItsCode() {
        val required = scenarios.filter { kotlinCell(it).getValue("status").jsonPrimitive.content == "required" }
        assertEquals(TRIGGERS.keys, required.map { it.getValue("id").jsonPrimitive.content }.toSet())
        for (scenario in required) {
            val id = scenario.getValue("id").jsonPrimitive.content
            val inputs = scenario.getValue("inputs").jsonObject
            val outcomes = TRIGGERS.getValue(id).invoke(this, inputs)
            assertTrue(outcomes.isNotEmpty(), id)
            for ((entry, code) in outcomes) {
                assertEquals(scenario.getValue("code").jsonPrimitive.content, code?.code, "$id $entry")
            }
        }
    }

    @Test
    fun otherKotlinCellsAreExplained() {
        for (scenario in scenarios) {
            val cell = kotlinCell(scenario)
            if (cell.getValue("status").jsonPrimitive.content != "required") {
                assertTrue(cell.getValue("reason").jsonPrimitive.content.isNotBlank())
            }
        }
        assertEquals("pending", kotlinCell(scenarios.single { it.id() == "invalid_sdk_config" })
            .getValue("status").jsonPrimitive.content)
    }

    @Test
    fun existingReasonCodesAndBaseTypesAreUnchanged() {
        val span = thrown { OpenMedSpan(5, 2, SENTINEL, "NAME", "NAME", 0.9f) }
        assertIs<IllegalArgumentException>(span)
        assertEquals("end must be >= start: end=2 start=5", span.message)
        val profile = thrown { PolicyProfiles.load("synthetic_unknown_profile") }
        assertIs<PolicyProfileException>(profile)
        assertEquals("profile_unknown", profile.code)
        assertEquals(
            OpenMedErrorCode.CONFIGURATION_ERROR,
            codeOf { PolicyEngine.decide(PolicyProfiles.load("hipaa_safe_harbor"), "x", emptyList(), surrogateKey = ByteArray(0)) },
        )
        assertEquals(OpenMedErrorCode.INPUT_ERROR, InferenceError.InvalidInput("synthetic").errorCode)
        assertEquals(OpenMedErrorCode.INFERENCE_ERROR, InferenceError.MissingOutput("logits").errorCode)
        assertEquals(OpenMedErrorCode.MODEL_LOAD_ERROR, InferenceError.SessionCreation().errorCode)
        assertEquals(OpenMedErrorCode.MODEL_LOAD_ERROR, ModelDownloadException("synthetic").errorCode)
        assertEquals(OpenMedErrorCode.MODEL_LOAD_ERROR, ModelIntegrityException("synthetic").errorCode)
        // A valid span is a negative control: nothing is thrown.
        OpenMedSpan(0, 4, "abcd", "NAME", "NAME", 0.9f)
    }

    // Scenario triggers: each returns (entry point, code) pairs.

    private fun emptyHashSecret(inputs: JsonObject): List<Pair<String, OpenMedErrorCode?>> {
        val secret = inputs.getValue("secret").jsonPrimitive.content.toByteArray()
        @Suppress("DEPRECATION")
        return listOf("DeidentifyEngine hashSalt" to codeOf { DeidentifyEngine(hashSalt = secret) })
    }

    private fun invalidOffsets(inputs: JsonObject): List<Pair<String, OpenMedErrorCode?>> {
        val start = inputs.getValue("start").jsonPrimitive.int
        val end = inputs.getValue("end").jsonPrimitive.int
        val text = SENTINEL.padEnd(inputs.getValue("text_length").jsonPrimitive.int, 'x')
            .take(inputs.getValue("text_length").jsonPrimitive.int)
        return listOf(
            "OpenMedSpan" to codeOf { OpenMedSpan(start, end, text, "NAME", "NAME", 0.9f) },
            "PolicyEngine detections" to codeOf {
                PolicyEngine.decide(
                    PolicyProfiles.load("hipaa_safe_harbor"),
                    text,
                    listOf(PolicyDetection("NAME", start, end, 900_000)),
                )
            },
        )
    }

    private fun unknownPolicyProfile(inputs: JsonObject): List<Pair<String, OpenMedErrorCode?>> {
        val profile = inputs.getValue("profile").jsonPrimitive.content
        return listOf("PolicyProfiles.load" to codeOf { PolicyProfiles.load(profile) })
    }

    private fun modelMissingOffline(inputs: JsonObject): List<Pair<String, OpenMedErrorCode?>> {
        assertTrue(inputs.getValue("offline").jsonPrimitive.boolean)
        val ref = ArtifactRef(
            repoId = inputs.getValue("model_id").jsonPrimitive.content,
            format = ArtifactFormat.ONNX,
            runtime = ArtifactRuntime.ONNX_RUNTIME,
            digest = "sha256:" + "0".repeat(64),
        )
        val acquisition = ArtifactAcquisition(ModelCache(temporaryFolder.newFolder()), NoNetworkClient)
        val result = runBlocking {
            acquisition.acquire(
                ArtifactAcquisitionRequest(ref, ArtifactBackendCapability.onnxRuntime(), ArtifactNetworkConsent.DENIED),
            )
        }
        assertIs<ArtifactAcquisitionResult.Failed>(result)
        assertEquals(ArtifactAcquisitionReason.NETWORK_CONSENT_REQUIRED, result.reason)
        assertFalse(inputs.getValue("model_id").jsonPrimitive.content in result.toString())
        assertNull(ArtifactAcquisitionResult.Cancelled(ref).errorCode)
        return listOf("ArtifactAcquisition.acquire" to result.errorCode)
    }

    private fun codeOf(block: () -> Unit): OpenMedErrorCode? {
        val error = thrown(block)
        val message = error.message.orEmpty()
        assertFalse(SENTINEL in message || "synthetic_unknown_profile" in message, message)
        return (error as? OpenMedCodedError)?.errorCode
    }

    private fun thrown(block: () -> Unit): Throwable {
        try {
            block()
        } catch (error: Throwable) {
            return error
        }
        fail("expected the scenario to fail")
    }

    private fun kotlinCell(scenario: JsonObject): JsonObject =
        scenario.getValue("runtimes").jsonObject.getValue("kotlin").jsonObject

    private fun JsonObject.id(): String = getValue("id").jsonPrimitive.content

    private object NoNetworkClient : HuggingFaceModelClient {
        override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo =
            throw AssertionError("offline scenarios must not reach the network")

        override fun downloadFile(repoId: String, path: String, revision: String, destination: File) {
            throw AssertionError("offline scenarios must not reach the network")
        }
    }

    private companion object {
        const val FIXTURE = "error_codes_v1.json"
        const val SENTINEL = "SYNTH Jane Roe"

        val TRIGGERS: Map<String, ErrorCodeParityTest.(JsonObject) -> List<Pair<String, OpenMedErrorCode?>>> = mapOf(
            "empty_hash_secret" to { inputs -> emptyHashSecret(inputs) },
            "invalid_offsets" to { inputs -> invalidOffsets(inputs) },
            "unknown_policy_profile" to { inputs -> unknownPolicyProfile(inputs) },
            "model_missing_offline" to { inputs -> modelMissingOffline(inputs) },
        )
    }
}
