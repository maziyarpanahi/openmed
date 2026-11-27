package com.openmed.openmedkit.bundle

import java.nio.charset.StandardCharsets

/** Value-free errors raised for malformed profiles, probes, or constraints. */
enum class DeviceCapabilityReason(val code: String) {
    PROFILE_INVALID("profile_invalid"),
    PROBE_INVALID("probe_invalid"),
    CONSTRAINTS_INVALID("constraints_invalid"),
}

/** Exception whose message is only a controlled reason code. */
class DeviceCapabilityException(val reason: DeviceCapabilityReason) : Exception(reason.code)

/**
 * Shared vocabulary for normalized device-capability profiles.
 *
 * Mirrors `openmed/core/device_capabilities.py`; the golden vectors in
 * `tests/fixtures/device_capabilities` pin every table here.
 */
object DeviceCapabilityVocabulary {
    const val SCHEMA = "openmed.device_capabilities"
    const val SCHEMA_VERSION = 1L
    const val ANDROID_SDK_NAME = "openmedkit-android"
    const val MIN_BUDGET_BYTES = 16L shl 20
    const val MAX_BUDGET_BYTES = 1L shl 40
    const val MAX_SEQUENCE_LENGTH = 1L shl 20
    const val MAX_OPERATOR_SET = 1000L

    /** Accelerator kind behind each normalized execution provider. */
    val PROVIDER_ACCELERATORS: Map<String, String> = sortedMapOf(
        "coreml" to "npu", "cpu" to "cpu", "metal" to "gpu", "nnapi" to "npu", "qnn" to "npu",
        "wasm" to "cpu", "webgpu" to "gpu",
    )

    /** Execution providers a profile may report on each platform. */
    val PLATFORM_PROVIDERS: Map<String, Set<String>> = sortedMapOf(
        "android" to setOf("cpu", "nnapi", "qnn"),
        "ios" to setOf("coreml", "cpu", "metal"),
        "linux" to setOf("cpu"),
        "macos" to setOf("coreml", "cpu", "metal"),
        "visionos" to setOf("coreml", "cpu"),
        "watchos" to setOf("coreml", "cpu"),
        "web" to setOf("wasm", "webgpu"),
        "windows" to setOf("cpu"),
    )
    val CPU_ARCHITECTURES: Set<String> = setOf("arm64", "armv7", "unknown", "wasm32", "x86_64")
    val FEATURES: Set<String> = setOf(
        "cpu_avx2", "cpu_fp16", "cpu_neon", "wasm_simd", "wasm_threads", "webgpu_shader_f16",
    )

    /** Model tiers in increasing capacity order. */
    val MODEL_TIERS: List<String> = listOf("nano", "tiny", "base")
    val NETWORK_STATES: Set<String> = setOf("offline", "online", "unknown")
    internal val ACCELERATORS: Set<String> = setOf("cpu", "gpu", "npu")
    internal val PRECISIONS: Set<String> = setOf("bf16", "fp16", "fp32", "int2", "int4", "int8")

    /** Round a measured budget down to the ladder: `0`, then powers of two from 16 MiB to 1 TiB. */
    fun budgetBucket(value: Long?): Long? {
        if (value == null) return null
        if (value < 0 || value > PortableBundle.MAX_SAFE_INTEGER) {
            throw DeviceCapabilityException(DeviceCapabilityReason.PROBE_INVALID)
        }
        if (value < MIN_BUDGET_BYTES) return 0
        var bucket = MIN_BUDGET_BYTES
        while (bucket < MAX_BUDGET_BYTES && bucket * 2 <= value) bucket *= 2
        return bucket
    }

    /** Keep only the major and minor components of a dotted OS version. */
    fun coarseOsVersion(value: String?): String? {
        if (value == null) return null
        if (!Patterns.VERSION.matches(value)) throw DeviceCapabilityException(DeviceCapabilityReason.PROBE_INVALID)
        return value.split(".").take(2).joinToString(".")
    }

    /** Return the capacity rank of a model tier (`nano` is 0), or -1. */
    fun tierRank(tier: String): Int = MODEL_TIERS.indexOf(tier)

    internal fun isBudget(value: Long?): Boolean = value == null || runCatching { budgetBucket(value) }.getOrNull() == value
}

/**
 * Coarse, closed description of what a device can execute.
 *
 * Collections are sets: they are sorted and de-duplicated on construction so
 * equivalent profiles encode to identical canonical bytes. A profile holds only
 * controlled vocabulary, dotted versions, and ladder budgets, so it can never
 * carry a Build model, fingerprint, serial, or exact memory size.
 */
class DeviceCapabilityProfile(
    val platform: String,
    val sdkName: String,
    val sdkVersion: String,
    /** `MAJOR` or `MAJOR.MINOR` (Android: the API level), or `null`. */
    val osVersion: String? = null,
    val cpuArchitecture: String = "unknown",
    providers: Collection<String> = emptyList(),
    runtimes: Collection<String> = emptyList(),
    precisions: Collection<String> = emptyList(),
    features: Collection<String> = emptyList(),
    operatorSets: Map<String, Long> = emptyMap(),
    val memoryBudgetBytes: Long? = null,
    val storageBudgetBytes: Long? = null,
    val maxSequenceLength: Long? = null,
    val maxModelTier: String? = null,
    /** Informational only; never changes a plan and is excluded from [digest]. */
    val network: String = "unknown",
) {
    val providers: List<String>
    val runtimes: List<String>
    val precisions: List<String>
    val features: List<String>
    val operatorSets: Map<String, Long>

    init {
        val v = DeviceCapabilityVocabulary
        val valid = platform in PortableBundle.PLATFORMS && Patterns.TOKEN.matches(sdkName) &&
            Patterns.VERSION.matches(sdkVersion) &&
            (osVersion == null || (Patterns.VERSION.matches(osVersion) && osVersion.split(".").size <= 2)) &&
            cpuArchitecture in v.CPU_ARCHITECTURES && v.isBudget(memoryBudgetBytes) && v.isBudget(storageBudgetBytes) &&
            (maxSequenceLength == null || maxSequenceLength in 1..v.MAX_SEQUENCE_LENGTH) &&
            (maxModelTier == null || v.tierRank(maxModelTier) >= 0) && network in v.NETWORK_STATES &&
            operatorSets.all { (key, value) -> key in PortableBundle.FORMAT_PLATFORMS && value in 1..v.MAX_OPERATOR_SET }
        if (!valid) throw DeviceCapabilityException(DeviceCapabilityReason.PROFILE_INVALID)
        val loadable = PortableBundle.FORMAT_PLATFORMS.filterValues { platform in it }.keys
        this.providers = set(providers, v.PLATFORM_PROVIDERS[platform].orEmpty())
        this.runtimes = set(runtimes, loadable)
        this.precisions = set(precisions, v.PRECISIONS)
        this.features = set(features, v.FEATURES)
        this.operatorSets = operatorSets.toSortedMap()
    }

    /** Sorted accelerator kinds behind [providers]. */
    val accelerators: List<String>
        get() = providers.mapNotNull { DeviceCapabilityVocabulary.PROVIDER_ACCELERATORS[it] }.distinct().sorted()

    /** The canonical profile document bytes (`device_capability_profile`). */
    fun canonicalJson(): ByteArray = Json.encode(document(includeNetwork = true))

    /** `sha256:` digest of every resolution-relevant member (not [network]). */
    val digest: String
        get() = "sha256:" + PortableBundle.sha256Hex(Json.encode(document(includeNetwork = false)))

    internal fun document(includeNetwork: Boolean): Json {
        val members = mutableListOf(
            "accelerators" to strings(accelerators),
            "cpu_architecture" to Json.Str(cpuArchitecture),
            "features" to strings(features),
            "max_model_tier" to optional(maxModelTier),
            "max_sequence_length" to optional(maxSequenceLength),
            "memory_budget_bytes" to optional(memoryBudgetBytes),
            "operator_sets" to Json.Obj(operatorSets.map { it.key to Json.Num(it.value) }),
            "os_version" to optional(osVersion),
            "platform" to Json.Str(platform),
            "precisions" to strings(precisions),
            "providers" to strings(providers),
            "runtimes" to strings(runtimes),
            "schema" to Json.Str(DeviceCapabilityVocabulary.SCHEMA),
            "schema_version" to Json.Num(DeviceCapabilityVocabulary.SCHEMA_VERSION),
            "sdk_name" to Json.Str(sdkName),
            "sdk_version" to Json.Str(sdkVersion),
            "storage_budget_bytes" to optional(storageBudgetBytes),
        )
        if (includeNetwork) members += "network" to Json.Str(network)
        return Json.Obj(members)
    }

    override fun equals(other: Any?): Boolean =
        other is DeviceCapabilityProfile && canonicalJson().contentEquals(other.canonicalJson())

    override fun hashCode(): Int = canonicalJson().contentHashCode()

    override fun toString(): String = String(canonicalJson(), StandardCharsets.UTF_8)

    companion object {
        private val DOCUMENT_KEYS = arrayOf(
            "accelerators", "cpu_architecture", "features", "max_model_tier", "max_sequence_length",
            "memory_budget_bytes", "network", "operator_sets", "os_version", "platform", "precisions", "providers",
            "runtimes", "schema", "schema_version", "sdk_name", "sdk_version", "storage_budget_bytes",
        )

        private fun set(items: Collection<String>, allowed: Set<String>): List<String> {
            if (!items.all { it in allowed }) throw DeviceCapabilityException(DeviceCapabilityReason.PROFILE_INVALID)
            return items.distinct().sorted()
        }

        /** Parse a profile document strictly; unknown or missing members fail. */
        fun fromJson(data: ByteArray): DeviceCapabilityProfile {
            val value = try {
                Json.parse(data)
            } catch (error: Json.Invalid) {
                throw DeviceCapabilityException(DeviceCapabilityReason.PROFILE_INVALID)
            }
            return fromDocument(value)
        }

        internal fun fromDocument(value: Json): DeviceCapabilityProfile {
            val fail = { DeviceCapabilityException(DeviceCapabilityReason.PROFILE_INVALID) }
            if (value !is Json.Obj || !value.hasExactKeys(*DOCUMENT_KEYS) ||
                value["schema"] != Json.Str(DeviceCapabilityVocabulary.SCHEMA) ||
                value["schema_version"] != Json.Num(DeviceCapabilityVocabulary.SCHEMA_VERSION)
            ) {
                throw fail()
            }
            val operatorSets = (value["operator_sets"] as? Json.Obj ?: throw fail()).members
            if (operatorSets.map { it.first }.toSet().size != operatorSets.size) throw fail()
            val profile = try {
                DeviceCapabilityProfile(
                    platform = value.string("platform") ?: throw fail(),
                    sdkName = value.string("sdk_name") ?: throw fail(),
                    sdkVersion = value.string("sdk_version") ?: throw fail(),
                    osVersion = value.optionalString("os_version"),
                    cpuArchitecture = value.string("cpu_architecture") ?: throw fail(),
                    providers = value.strings("providers") ?: throw fail(),
                    runtimes = value.strings("runtimes") ?: throw fail(),
                    precisions = value.strings("precisions") ?: throw fail(),
                    features = value.strings("features") ?: throw fail(),
                    operatorSets = operatorSets.associate { it.first to ((it.second as? Json.Num)?.value ?: throw fail()) },
                    memoryBudgetBytes = value.optionalLong("memory_budget_bytes"),
                    storageBudgetBytes = value.optionalLong("storage_budget_bytes"),
                    maxSequenceLength = value.optionalLong("max_sequence_length"),
                    maxModelTier = value.optionalString("max_model_tier"),
                    network = value.string("network") ?: throw fail(),
                )
            } catch (error: Json.Invalid) {
                throw fail()
            }
            val accelerators = value.strings("accelerators") ?: throw fail()
            if (accelerators.distinct().size != accelerators.size || accelerators.sorted() != profile.accelerators) {
                throw fail()
            }
            return profile
        }
    }
}

internal fun strings(items: List<String>): Json = Json.Arr(items.map { Json.Str(it) })

internal fun optional(value: String?): Json = value?.let { Json.Str(it) } ?: Json.Null

internal fun optional(value: Long?): Json = value?.let { Json.Num(it) } ?: Json.Null

internal fun Json.Obj.string(key: String): String? = (this[key] as? Json.Str)?.value

internal fun Json.Obj.long(key: String): Long? = (this[key] as? Json.Num)?.value

internal fun Json.Obj.bool(key: String): Boolean? = (this[key] as? Json.Bool)?.value

internal fun Json.Obj.strings(key: String): List<String>? {
    val items = (this[key] as? Json.Arr)?.items ?: return null
    val values = items.mapNotNull { (it as? Json.Str)?.value }
    return if (values.size == items.size) values else null
}

internal fun Json.Obj.optionalString(key: String): String? = when (val item = this[key]) {
    Json.Null -> null
    is Json.Str -> item.value
    else -> throw Json.Invalid()
}

internal fun Json.Obj.optionalLong(key: String): Long? = when (val item = this[key]) {
    Json.Null -> null
    is Json.Num -> item.value
    else -> throw Json.Invalid()
}
