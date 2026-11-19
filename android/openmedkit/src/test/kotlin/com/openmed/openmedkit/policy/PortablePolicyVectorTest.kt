package com.openmed.openmedkit.policy

import com.openmed.openmedkit.KitParity
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.security.MessageDigest
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Runs the shared portable policy vectors in
 * `tests/fixtures/parity/portable_policy_vectors.json` (derived from the
 * Python reference engine) through the Kotlin policy loader and engine.
 *
 * Failure messages carry case identifiers, field names, and reason codes only.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class PortablePolicyVectorTest {
    @Test
    fun bundledResourcesAreByteIdenticalAcrossKits() {
        val vectors = vectors()
        val manifestBytes = requireNotNull(PolicyProfiles.bundledResource(PolicyProfiles.MANIFEST_FILE))
        assertEquals(vectors.getString("manifest_sha256"), sha256Hex(manifestBytes))
        val manifest = PolicyProfiles.decodeManifest(manifestBytes)
        assertEquals(vectors.getJSONObject("profiles").toStringMap(), manifest.profiles)
        assertEquals(vectors.getJSONObject("excluded_profiles").toStringMap(), manifest.excludedProfiles)
        assertEquals(manifest.profiles.keys.sorted(), PolicyProfiles.bundledProfileNames)
        assertEquals(vectors.getJSONArray("supported_capabilities").strings(), PolicyProfiles.SUPPORTED_CAPABILITIES)

        val files = manifest.profiles.keys.map { "$it.json" } +
            listOf(PolicyProfiles.MANIFEST_FILE, PolicyProfiles.TAXONOMY_FILE)
        val swiftDirectory = repositoryRoot().resolve("swift/OpenMedKit/Sources/OpenMedKit/Resources/policies")
        for (file in files) {
            val bundled = requireNotNull(PolicyProfiles.bundledResource(file)) { file }
            assertContentEquals(swiftDirectory.resolve(file).readBytes(), bundled, file)
        }
        for ((name, digest) in manifest.profiles) {
            assertEquals(digest, PolicyProfiles.sha256Digest(requireNotNull(PolicyProfiles.bundledResource("$name.json"))))
        }
        for (name in manifest.excludedProfiles.keys) {
            assertNull(PolicyProfiles.bundledResource("$name.json"), name)
        }
        assertTrue(
            vectors.getJSONArray("consumers").objects().any {
                it.getString("surface") == "kotlin" && it.getString("test").endsWith("PortablePolicyVectorTest.kt")
            },
        )
    }

    @Test
    fun taxonomyIsTheKitParityVocabulary() {
        val taxonomy = PolicyProfiles.bundledTaxonomy()
        assertEquals(taxonomy.labels, KitParity.CANONICAL_LABELS)
        for (label in listOf("DIET_TYPE", "FEEDING_ROUTE", "NUTRITIONAL_STATUS", "NUTRITION_TARGET")) {
            assertEquals(label, PolicyProfiles.canonicalLabel(label))
        }
        assertEquals("PERSON", PolicyProfiles.canonicalLabel("patient_name"))
        assertEquals("PERSON", PolicyProfiles.canonicalLabel("B-PATIENT_NAME"))
        assertEquals("ID_NUM", PolicyProfiles.canonicalLabel("medicalrecordnumber"))
        assertEquals("ID_NUM", PolicyProfiles.canonicalLabel("medical_record_number"))
        assertEquals("OTHER", PolicyProfiles.canonicalLabel("season_hint"))
    }

    @Test
    fun decisionVectorsMatchThePythonReference() {
        val vectors = vectors()
        val cases = vectors.getJSONArray("decision_cases").objects()
        assertTrue(cases.size >= 25)
        for (item in cases) {
            val id = item.getString("id")
            val expected = item.getJSONObject("expected")
            val outcome = decide(item, vectors)
            val serialized = KitParity.serialize(outcome.result)

            assertEquals(expected.getString("result_json"), serialized, "case=$id result bytes")
            assertEquals(expected.getString("result_sha256"), sha256Hex(serialized.toByteArray(Charsets.UTF_8)), id)
            assertContentEquals(
                expected.getString("deidentified_text").toByteArray(Charsets.UTF_8),
                outcome.result.deidentifiedText.toByteArray(Charsets.UTF_8),
                "case=$id de-identified UTF-8 bytes",
            )
            assertEquals(expected.getString("policy_digest"), outcome.policyDigest, id)
            val decisions = expected.getJSONArray("decisions").objects()
            assertEquals(decisions.size, outcome.decisions.size, "case=$id decision count")
            outcome.decisions.zip(decisions).forEachIndexed { index, (actual, want) ->
                assertEquals(normalized(want), record(actual), "case=$id decision=$index")
            }
        }
    }

    @Test
    fun errorVectorsFailWithValueFreeReasons() {
        val vectors = vectors()
        for (item in vectors.getJSONArray("error_cases").objects()) {
            val id = item.getString("id")
            val expected = item.getJSONObject("expected")
            val error = assertFailsWith<PolicyProfileException>(id) { decide(item, vectors) }
            assertEquals(expected.getString("code"), error.code, id)
            assertEquals(expected.getString("field_path"), error.fieldPath, id)
            val text = item.getString("text")
            if (text.isNotEmpty()) {
                assertFalse(error.message.orEmpty().contains(text), id)
            }
        }
    }

    @Test
    fun loadVectorsPinDigestsAndFailClosed() {
        for (item in vectors().getJSONArray("load_cases").objects()) {
            val id = item.getString("id")
            val expected = item.getJSONObject("expected")
            if (expected.has("code")) {
                val error = assertFailsWith<PolicyProfileException>(id) { runLoadCase(item) }
                assertEquals(expected.getString("code"), error.code, id)
                assertEquals(expected.getString("field_path"), error.fieldPath, id)
                continue
            }
            val policy = runLoadCase(item)
            assertEquals(expected.getString("name"), policy.name, id)
            assertEquals(expected.getString("digest"), policy.digest, id)
            assertEquals(expected.getJSONArray("required_capabilities").strings(), policy.requiredCapabilities, id)
        }
    }

    @Test
    fun runtimeWithoutStrictCapabilityRefusesStrictProfiles() {
        val supported = PolicyProfiles.SUPPORTED_CAPABILITIES.filterNot { it.startsWith("strict") }
        val error = assertFailsWith<PolicyProfileException> {
            PolicyProfiles.load("strict_no_leak", supportedCapabilities = supported)
        }
        assertEquals("capability_missing", error.code)
        assertEquals("hipaa_safe_harbor", PolicyProfiles.load("hipaa_safe_harbor", supportedCapabilities = supported).name)
    }

    @Test
    fun surrogatesMatchAnIndependentHmac() {
        val keys = vectors().getJSONObject("surrogate_test_keys")
        val key = hexBytes(keys.getString("synthetic_test_key_a"))
        val mac = Mac.getInstance("HmacSHA256").apply { init(SecretKeySpec(key, "HmacSHA256")) }
        val message = "openmed.portable_surrogate.v1".toByteArray() + byteArrayOf(0) +
            "PERSON".toByteArray() + byteArrayOf(0) + "Ada Quill".toByteArray(Charsets.UTF_8)
        val expected = mac.doFinal(message).copyOf(8).joinToString("") { "%02x".format(it.toInt() and 0xff) }

        assertEquals(expected, PolicyEngine.surrogateToken(key, "PERSON", "Ada Quill"))
        assertNotEquals(
            expected,
            PolicyEngine.surrogateToken(hexBytes(keys.getString("synthetic_test_key_b")), "PERSON", "Ada Quill"),
        )
    }

    private fun decide(item: JSONObject, vectors: JSONObject): PolicyDecisionResult {
        val detections = item.getJSONArray("detections").objects().map {
            PolicyDetection(
                label = it.getString("label"),
                start = it.getInt("start"),
                end = it.getInt("end"),
                confidencePpm = it.getInt("confidence_ppm"),
                source = it.optString("source", "model"),
                validated = it.optBoolean("validated", true),
            )
        }
        val key = if (item.has("surrogate_key")) {
            hexBytes(vectors.getJSONObject("surrogate_test_keys").getString(item.getString("surrogate_key")))
        } else {
            null
        }
        val overrides = if (item.has("overrides")) item.getJSONObject("overrides").toStringMap() else emptyMap()
        return PolicyEngine.decide(
            PolicyProfiles.load(item.getString("profile")),
            item.getString("text"),
            detections,
            overrides,
            key,
        )
    }

    private fun runLoadCase(item: JSONObject): PolicyProfile {
        val profile = item.getString("profile")
        if (item.has("profile_patch")) {
            val policy = PolicyProfiles.load(profile)
            val document = JSONObject(String(requireNotNull(PolicyProfiles.bundledResource("${policy.name}.json"))))
            val patch = item.getJSONObject("profile_patch")
            for (field in patch.keys()) {
                document.put(field, patch.get(field))
            }
            return PolicyProfiles.decodeProfile(
                document.toString().toByteArray(Charsets.UTF_8),
                policy.name,
                policy.digest,
                policy.taxonomy,
            )
        }
        var resources: (String) -> ByteArray? = PolicyProfiles::bundledResource
        if (item.has("manifest_patch")) {
            val manifest = JSONObject(String(requireNotNull(PolicyProfiles.bundledResource(PolicyProfiles.MANIFEST_FILE))))
            val patch = item.getJSONObject("manifest_patch")
            for (field in patch.keys()) {
                val value = patch.get(field)
                val current = manifest.opt(field)
                if (value is JSONObject && current is JSONObject) {
                    for (key in value.keys()) {
                        current.put(key, value.get(key))
                    }
                } else {
                    manifest.put(field, value)
                }
            }
            val patched = manifest.toString().toByteArray(Charsets.UTF_8)
            resources = { file -> if (file == PolicyProfiles.MANIFEST_FILE) patched else PolicyProfiles.bundledResource(file) }
        }
        return PolicyProfiles.load(
            profile,
            expectedDigest = if (item.has("expected_digest")) item.getString("expected_digest") else null,
            resources = resources,
        )
    }

    private fun record(decision: PolicyDecision): Map<String, String> = buildMap {
        put("action", decision.action)
        put("canonical_label", decision.canonicalLabel)
        put("confidence_ppm", decision.confidencePpm.toString())
        put("end", decision.end.toString())
        put("escalate", decision.escalate.toString())
        put("inputs", decision.inputs.joinToString(","))
        put("keep_floor_ppm", decision.keepFloorPpm.toString())
        put("kind", decision.kind)
        put("label", decision.label)
        put("outcome", decision.outcome)
        put("recognized", decision.recognized.toString())
        put("start", decision.start.toString())
        put("strict_upgrade", decision.strictUpgrade.toString())
        decision.span?.let { put("span", it.toString()) }
    }

    private fun normalized(raw: JSONObject): Map<String, String> = buildMap {
        for (key in raw.keys()) {
            val value = raw.get(key)
            put(key, if (value is JSONArray) value.ints().joinToString(",") else value.toString())
        }
    }

    private fun vectors(): JSONObject {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream(VECTORS_FILE)) {
            "shared portable policy vectors were not registered as a test resource"
        }
        return JSONObject(String(stream.use { it.readBytes() }, Charsets.UTF_8))
    }

    private fun repositoryRoot(): File {
        var directory: File? = File(System.getProperty("user.dir")).absoluteFile
        while (directory != null && !directory.resolve("pyproject.toml").isFile) {
            directory = directory.parentFile
        }
        return requireNotNull(directory) { "repository root not found" }
    }

    private fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun hexBytes(hex: String): ByteArray = ByteArray(hex.length / 2) {
        hex.substring(it * 2, it * 2 + 2).toInt(16).toByte()
    }

    private fun JSONObject.toStringMap(): Map<String, String> = keys().asSequence().associateWith(::getString)

    private fun JSONArray.objects(): List<JSONObject> = (0 until length()).map(::getJSONObject)

    private fun JSONArray.strings(): List<String> = (0 until length()).map(::getString)

    private fun JSONArray.ints(): List<Int> = (0 until length()).map(::getInt)

    private companion object {
        const val VECTORS_FILE = "portable_policy_vectors.json"
    }
}
