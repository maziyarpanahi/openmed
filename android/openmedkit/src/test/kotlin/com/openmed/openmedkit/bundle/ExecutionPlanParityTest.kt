package com.openmed.openmedkit.bundle

import com.openmed.openmedkit.onnx.AcceleratorProvider
import com.openmed.openmedkit.onnx.AndroidCapabilityInputs
import com.openmed.openmedkit.onnx.AndroidCapabilityProbe
import com.openmed.openmedkit.onnx.AndroidDeviceTier
import java.io.File
import java.nio.charset.StandardCharsets
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class ExecutionPlanParityTest {
    private val vectors: Json.Obj by lazy {
        val module = File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
        )
        val repoRoot = requireNotNull(module.absoluteFile.parentFile?.parentFile) {
            "cannot locate the repository root"
        }
        Json.parse(File(repoRoot, "tests/fixtures/device_capabilities/vectors.json").readBytes()) as Json.Obj
    }

    private fun Json?.obj(): Json.Obj = this as Json.Obj
    private fun Json?.str(): String = (this as Json.Str).value
    private fun Json?.long(): Long = (this as Json.Num).value
    private fun Json?.items(): List<Json> = (this as Json.Arr).items
    private fun Json?.strings(): List<String> = items().map { it.str() }

    private fun text(value: ByteArray): String = String(value, StandardCharsets.UTF_8)

    private fun canonical(value: Json): String = text(Json.encode(value))

    private fun sentinels(): List<String> = vectors["identity_sentinels"].strings()

    private fun assertNoSentinels(data: ByteArray, label: String) {
        val output = text(data)
        for (sentinel in sentinels()) assertFalse(output.contains(sentinel), label)
    }

    private fun profile(name: String): DeviceCapabilityProfile =
        DeviceCapabilityProfile.fromDocument(vectors["profiles"].obj()[name].obj()["document"]!!)

    private fun constraints(name: String): ExecutionConstraints =
        ExecutionConstraints.fromJson(Json.encode(vectors["constraints"].obj()[name].obj()["document"]!!))

    private fun verification(raw: Json.Obj): PortableBundleVerification = PortableBundleVerification(
        ok = (raw["ok"] as Json.Bool).value,
        reason = (raw["reason"] as? Json.Str)?.value?.let { code -> PortableBundleReason.values().first { it.code == code } },
        contentId = (raw["content_id"] as? Json.Str)?.value,
        validSignatures = raw["valid_signatures"].long().toInt(),
        eligibleRuntimes = raw["eligible_runtimes"].items().map { it.long().toInt() },
        filesChecked = raw["files_checked"].long().toInt(),
        filesVerified = (raw["files_verified"] as Json.Bool).value,
    )

    private data class Inputs(
        val manifest: ByteArray,
        val verification: PortableBundleVerification,
        val profile: DeviceCapabilityProfile,
        val constraints: ExecutionConstraints,
    )

    private fun inputs(item: Json.Obj): Inputs = Inputs(
        manifest = vectors["manifests"].obj()[item["manifest"].str()].obj()["text"].str()
            .toByteArray(StandardCharsets.UTF_8),
        verification = verification(item["verification"].obj()),
        profile = profile(item["profile"].str()),
        constraints = constraints(item["constraints"].str()),
    )

    private fun resolve(inputs: Inputs): ExecutionPlan =
        ExecutionPlanResolver.resolve(inputs.manifest, inputs.verification, inputs.profile, inputs.constraints)

    private fun cases(): List<Json.Obj> = vectors["cases"].items().map { it.obj() }

    @Test
    fun androidAdapterNormalizesSharedProbes() {
        val items = vectors["normalization"].items().map { it.obj() }.filter { it["kit"].str() == "android" }
        assertTrue(items.size >= 5)
        for (item in items) {
            val probe = item["probe"].obj()
            val inputs = AndroidCapabilityInputs(
                sdkInt = probe["sdk_int"].long().toInt(),
                abi = probe["abi"].str(),
                availableProviders = probe["available_providers"].strings().map { AcceleratorProvider.valueOf(it) }.toSet(),
                memoryClassMb = (probe["memory_class_mb"] as? Json.Num)?.value?.toInt(),
                storageFreeBytes = (probe["storage_free_bytes"] as? Json.Num)?.value,
                deviceTier = (probe["device_tier"] as? Json.Str)?.value?.let { AndroidDeviceTier.valueOf(it) },
                maxSequenceLength = (probe["max_sequence_length"] as? Json.Num)?.value,
                operatorSets = (probe["operator_sets"] as? Json.Obj)?.members?.associate { it.first to it.second.long() }
                    ?: emptyMap(),
                network = (probe["network"] as? Json.Str)?.value ?: "unknown",
            )
            val profile = AndroidCapabilityProbe.profile(inputs, probe["sdk_version"].str())
            assertEquals(canonical(item["expected"]!!), text(profile.canonicalJson()), item["id"].str())
            assertNoSentinels(profile.canonicalJson(), item["id"].str())
        }
    }

    @Test
    fun profileDocumentsRoundTrip() {
        for ((name, entry) in vectors["profiles"].obj().members) {
            val profile = profile(name)
            assertEquals(canonical(entry.obj()["document"]!!), text(profile.canonicalJson()), name)
            assertEquals(entry.obj()["digest"].str(), profile.digest, name)
            assertEquals(profile, DeviceCapabilityProfile.fromJson(profile.canonicalJson()), name)
        }
    }

    @Test
    fun constraintDocumentsRoundTrip() {
        for ((name, entry) in vectors["constraints"].obj().members) {
            val constraints = constraints(name)
            assertEquals(canonical(entry.obj()["document"]!!), text(constraints.canonicalJson()), name)
            assertEquals(entry.obj()["digest"].str(), constraints.digest, name)
        }
        assertEquals(ExecutionConstraints(), constraints("default"))
    }

    @Test
    fun resolverReproducesEveryGoldenPlan() {
        for (item in cases()) {
            val plan = resolve(inputs(item))
            assertEquals(canonical(item["expected"]!!), text(plan.canonicalJson()), item["id"].str())
            assertNoSentinels(plan.canonicalJson(), item["id"].str())
        }
    }

    @Test
    fun invalidationVectors() {
        val byId = cases().associateBy { it["id"].str() }
        for (entry in vectors["invalidations"].items().map { it.obj() }) {
            val plan = resolve(inputs(byId.getValue(entry["plan"].str())))
            val current = inputs(byId.getValue(entry["inputs"].str()))
            val changed = ExecutionPlanResolver.invalidation(
                plan, current.manifest, current.verification, current.profile, current.constraints,
            )
            assertEquals(entry["expected"].strings(), changed, "${entry["plan"].str()} -> ${entry["inputs"].str()}")
        }
    }

    @Test
    fun vocabularyMatchesVectors() {
        val vocabulary = vectors["vocabulary"].obj()
        fun map(key: String): Map<String, List<String>> =
            vocabulary[key].obj().members.associate { it.first to it.second.strings() }
        assertEquals(vocabulary["reason_codes"].strings(), ExecutionPlanReason.values().map { it.code })
        assertEquals(
            vocabulary["provider_accelerators"].obj().members.associate { it.first to it.second.str() },
            DeviceCapabilityVocabulary.PROVIDER_ACCELERATORS,
        )
        assertEquals(map("platform_providers"), DeviceCapabilityVocabulary.PLATFORM_PROVIDERS.mapValues { it.value.sorted() })
        assertEquals(map("format_providers"), ExecutionPlanResolver.FORMAT_PROVIDERS.mapValues { it.value.sorted() })
        assertEquals(map("default_providers"), ExecutionPlanResolver.DEFAULT_PROVIDERS)
        assertEquals(vocabulary["features"].strings(), DeviceCapabilityVocabulary.FEATURES.sorted())
        assertEquals(vocabulary["cpu_architectures"].strings(), DeviceCapabilityVocabulary.CPU_ARCHITECTURES.sorted())
        assertEquals(vocabulary["model_tiers"].strings(), DeviceCapabilityVocabulary.MODEL_TIERS)
        assertEquals(vocabulary["network_states"].strings(), DeviceCapabilityVocabulary.NETWORK_STATES.sorted())
        assertEquals(vocabulary["step_roles"].strings(), ExecutionPlanResolver.STEP_ROLES)
        assertEquals(vocabulary["assumptions"].strings(), ExecutionPlanResolver.ASSUMPTIONS)
        assertEquals(vocabulary["invalidation_components"].strings(), ExecutionPlanResolver.INVALIDATION_COMPONENTS)
        assertEquals(vocabulary["execution_extension"].str(), ExecutionPlanResolver.EXTENSION_NAME)
        assertEquals(vocabulary["resolver_version"].long(), ExecutionPlanResolver.RESOLVER_VERSION)
        assertEquals(vocabulary["sdk_names"].obj()["android"].str(), DeviceCapabilityVocabulary.ANDROID_SDK_NAME)
        val ladder = vocabulary["budget_ladder"].obj()
        assertEquals(ladder["min_bytes"].long(), DeviceCapabilityVocabulary.MIN_BUDGET_BYTES)
        assertEquals(ladder["max_bytes"].long(), DeviceCapabilityVocabulary.MAX_BUDGET_BYTES)
    }

    @Test
    fun closedDocumentsRejectIdentityBearingMembers() {
        val base = vectors["profiles"].obj()["android_flagship"].obj()["document"].obj()
        val mutations = listOf(
            "device_id" to Json.Str(sentinels()[1]),
            "sdk_name" to Json.Str(sentinels()[0]),
            "providers" to Json.Arr(listOf(Json.Str("webgpu"))),
            "memory_budget_bytes" to Json.Num(3_000_000_000),
            "os_version" to Json.Str("34.0.1"),
            "accelerators" to Json.Arr(listOf(Json.Str("gpu"))),
            "network" to Json.Str(sentinels()[3]),
        )
        for ((key, value) in mutations) {
            val members = base.members.filter { it.first != key } + (key to value)
            val error = assertFailsWith<DeviceCapabilityException> {
                DeviceCapabilityProfile.fromDocument(Json.Obj(members))
            }
            assertEquals(DeviceCapabilityReason.PROFILE_INVALID, error.reason, key)
        }
        val defaults = vectors["constraints"].obj()["default"].obj()["document"].obj()
        for ((key, value) in listOf(
            "remote_endpoint" to Json.Str("x"),
            "max_steps" to Json.Num(0),
            "disallowed_providers" to Json.Arr(listOf(Json.Str("cloud"))),
        )) {
            val members = defaults.members.filter { it.first != key } + (key to value)
            assertFailsWith<DeviceCapabilityException> { ExecutionConstraints.fromJson(Json.encode(Json.Obj(members))) }
        }
    }

    @Test
    fun budgetLadderAndOsVersionsStayCoarse() {
        assertEquals(null, DeviceCapabilityVocabulary.budgetBucket(null))
        assertEquals(128L shl 20, DeviceCapabilityVocabulary.budgetBucket(150L shl 20))
        assertEquals(1L shl 40, DeviceCapabilityVocabulary.budgetBucket(1L shl 45))
        assertEquals(0L, DeviceCapabilityVocabulary.budgetBucket(1))
        assertFailsWith<DeviceCapabilityException> { DeviceCapabilityVocabulary.budgetBucket(-1) }
        assertEquals("17.4", DeviceCapabilityVocabulary.coarseOsVersion("17.4.1"))
        assertFailsWith<DeviceCapabilityException> { DeviceCapabilityVocabulary.coarseOsVersion("17.4-beta") }
    }
}
