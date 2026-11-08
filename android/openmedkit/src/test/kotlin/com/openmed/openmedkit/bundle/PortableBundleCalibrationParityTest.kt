package com.openmed.openmedkit.bundle

import java.io.File
import java.nio.charset.StandardCharsets
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.junit.Test

class PortableBundleCalibrationParityTest {
    private val vectors: Json.Obj by lazy {
        val module = File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
        )
        val repoRoot = requireNotNull(module.absoluteFile.parentFile?.parentFile) {
            "cannot locate the repository root"
        }
        val file = File(repoRoot, "tests/fixtures/portable_bundle_calibration/vectors.json")
        Json.parse(file.readBytes()) as Json.Obj
    }

    private fun Json?.obj(): Json.Obj = this as Json.Obj
    private fun Json?.str(): String = (this as Json.Str).value
    private fun Json?.long(): Long = (this as Json.Num).value
    private fun Json?.bool(): Boolean = (this as Json.Bool).value
    private fun Json?.items(): List<Json> = (this as Json.Arr).items
    private fun Json?.optionalString(): String? = if (this == null || this == Json.Null) null else str()
    private fun Json?.optionalLong(): Long? = if (this == Json.Null) null else long()

    private fun manifest(key: String): ByteArray {
        val entry = vectors["manifests"].obj()[key].obj()
        entry["text"]?.let { return it.str().toByteArray(StandardCharsets.UTF_8) }
        val base = manifest(entry["base"].str())
        val at = entry["at"].long().toInt()
        val delete = entry["delete"].long().toInt()
        val insert = entry["insert"].str().toByteArray(StandardCharsets.UTF_8)
        return base.copyOfRange(0, at) + insert + base.copyOfRange(at + delete, base.size)
    }

    private fun verification(raw: Json.Obj) = PortableBundleVerification(
        ok = raw["ok"].bool(),
        reason = raw["reason"].optionalString()?.let { code -> PortableBundleReason.values().first { it.code == code } },
        contentId = raw["content_id"].optionalString(),
        validSignatures = raw["valid_signatures"].long().toInt(),
        eligibleRuntimes = raw["eligible_runtimes"].items().map { it.long().toInt() },
        filesChecked = raw["files_checked"].long().toInt(),
        filesVerified = raw["files_verified"].bool(),
    )

    private fun context(understood: Set<String>): PortableBundleContext {
        val value = vectors["target"].obj()
        return PortableBundleContext(
            now = vectors["now"].str(),
            target = PortableBundleTarget(
                platform = value["platform"].str(),
                sdkName = value["sdk_name"].str(),
                sdkVersion = value["sdk_version"].str(),
                runtimes = value["runtimes"].items().map { it.str() },
                osVersion = value["os_version"].optionalString(),
                policyVersion = value["policy_version"].optionalLong(),
                memoryBytes = value["memory_bytes"].optionalLong(),
                storageBytes = value["storage_bytes"].optionalLong(),
            ),
            understoodExtensions = understood,
        )
    }

    private fun tree(item: Json.Obj): Map<String, ByteArray> {
        val files = linkedMapOf<String, ByteArray>()
        for (entry in vectors["tree"].items() + item["tree_set"].items()) {
            val value = entry.obj()
            files[value["path"].str()] = value["text"].str().toByteArray(StandardCharsets.UTF_8)
        }
        return files
    }

    private fun request(raw: Json.Obj) = DecisionCalibrationRequest(
        calibrationId = raw["calibration_id"].str(),
        runtimeId = raw["runtime_id"].str(),
        decisionSchemaVersion = raw["decision_schema_version"].str(),
        modelId = raw["model_id"].optionalString(),
        modelRevision = raw["model_revision"].optionalString(),
        runtimeDigest = raw["runtime_digest"].optionalString(),
        scopeKind = raw["scope_kind"].optionalString(),
        scopeDigest = raw["scope_digest"].optionalString(),
        profileDigest = raw["profile_digest"].optionalString(),
    )

    @Test
    fun vectorsShareTheReasonVocabulary() {
        assertEquals(Json.Bool(true), vectors["synthetic_only"])
        assertEquals(Json.Bool(true), vectors["test_only_keys"])
        assertEquals(PortableBundleCalibration.EXTENSION, vectors["extension"].str())
        assertEquals(PortableBundleCalibration.EXTENSION_VERSION, vectors["extension_version"].long())
        assertEquals(
            DecisionCalibrationReason.values().map { it.code },
            vectors["reason_codes"].items().map { it.str() },
        )
    }

    @Test
    fun verificationCasesMatchThePythonReference() {
        val policy = PortableBundleTrustPolicy.fromJson(
            String(Json.encode(vectors["trust_policy"]!!), StandardCharsets.UTF_8),
        )
        val cases = vectors["verification_cases"].items()
        assertTrue(cases.size > 10)
        for (entry in cases) {
            val item = entry.obj()
            val id = item["id"].str()
            val expected = item["expected"].obj()
            val result = PortableBundle.verify(
                manifest = manifest(item["manifest"].str()),
                envelope = item["envelope"].str().toByteArray(StandardCharsets.UTF_8),
                trustPolicy = policy,
                context = context(item["understood_extensions"].items().map { it.str() }.toSet()),
                files = tree(item),
            )
            assertEquals(expected["ok"].bool(), result.ok, id)
            assertEquals(expected["reason"].optionalString(), result.reason?.code, id)
            assertEquals(expected["content_id"].optionalString(), result.contentId, id)
            assertEquals(expected["valid_signatures"].long().toInt(), result.validSignatures, id)
            assertEquals(expected["eligible_runtimes"].items().map { it.long().toInt() }, result.eligibleRuntimes, id)
            assertEquals(expected["files_checked"].long().toInt(), result.filesChecked, id)
            assertEquals(expected["files_verified"].bool(), result.filesVerified, id)
        }
    }

    @Test
    fun readCasesMatchThePythonReference() {
        val cases = vectors["read_cases"].items()
        assertTrue(cases.size > 40)
        for (entry in cases) {
            val item = entry.obj()
            val id = item["id"].str()
            val expected = item["expected"].obj()
            val result = PortableBundleCalibration.read(
                manifest(item["manifest"].str()),
                verification(item["verification"].obj()),
            )
            assertEquals(expected["ok"].bool(), result.ok, id)
            assertEquals(expected["reason"].optionalString(), result.reason?.code, id)
            assertEquals(expected["content_id"].str(), result.contentId, id)
            assertEquals(
                expected["profiles"].items().map { it.str() },
                result.profiles.map { String(it.canonicalJson(), StandardCharsets.UTF_8) },
                id,
            )
        }
    }

    @Test
    fun resolveCasesMatchThePythonReference() {
        val cases = vectors["resolve_cases"].items()
        assertTrue(cases.size > 20)
        for (entry in cases) {
            val item = entry.obj()
            val id = item["id"].str()
            val expected = item["expected"].obj()
            val raw = item["request"].obj()
            val error = expected["error"]
            if (error != null) {
                val thrown = assertFailsWith<DecisionCalibrationException>(id) { request(raw) }
                assertEquals(error.str(), thrown.message, id)
                continue
            }
            val selection = PortableBundleCalibration.resolve(
                manifest(item["manifest"].str()),
                verification(item["verification"].obj()),
                request(raw),
            )
            assertEquals(expected["ok"].bool(), selection.ok, id)
            assertEquals(expected["reason"].optionalString(), selection.reason?.code, id)
            assertEquals(expected["content_id"].str(), selection.contentId, id)
            assertEquals(expected["profile_digest"].optionalString(), selection.profile?.profileDigest, id)
        }
    }

    @Test
    fun goldenDigestsAndTypedProfiles() {
        val golden = vectors["golden"].obj()
        val data = manifest("golden")
        assertEquals(golden["content_id"].str(), PortableBundle.contentId(data))
        for ((runtimeId, digest) in golden["runtime_digests"].obj().members) {
            assertEquals(digest.str(), PortableBundleCalibration.runtimeDigest(data, runtimeId))
        }
        assertNull(PortableBundleCalibration.runtimeDigest(data, "tflite-int8"))
        val verified = PortableBundleVerification(true, null, PortableBundle.contentId(data), 1, listOf(1, 2), 10, true)
        val read = PortableBundleCalibration.read(data, verified)
        assertEquals(golden["profile_digests"].items().map { it.str() }, read.profiles.map { it.profileDigest })
        val triage = read.profiles[0]
        assertEquals(DecisionCalibrationMethod.DeterministicMargin, triage.method)
        assertEquals(0.7, triage.minimumConfidence)
        assertEquals(0.15, triage.minimumMargin)
        assertEquals(listOf("coreml-fp16", "onnx-fp32"), triage.runtimeIds)
        assertEquals(DecisionCalibrationMethod.TemperatureScaling(1_250_000), read.profiles[1].method)
        assertEquals(DecisionCalibrationMethod.PlattScaling(-1_800_000, 250_000), read.profiles[2].method)
        val isotonic = read.profiles[3].method as DecisionCalibrationMethod.IsotonicRegression
        assertEquals(DecisionCalibrationPoint(250_000, 100_000), isotonic.points[1])
        assertFalse(read.profiles[3].evaluationPath != null)
    }

    @Test
    fun errorsCarryOnlyReasonCodes() {
        val error = assertFailsWith<DecisionCalibrationException> {
            DecisionCalibrationRequest(
                calibrationId = "openmed.synthetic.triage.v1",
                runtimeId = "onnx-fp32",
                decisionSchemaVersion = "1.1.0",
                modelRevision = "SENTINEL-REVISION",
            )
        }
        assertEquals("calibration_request_invalid", error.message)
        assertFalse(error.toString().contains("SENTINEL"))
    }
}
