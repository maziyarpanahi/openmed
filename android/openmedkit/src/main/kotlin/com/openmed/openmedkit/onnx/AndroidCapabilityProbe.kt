package com.openmed.openmedkit.onnx

import android.os.Build
import com.openmed.openmedkit.bundle.DeviceCapabilityException
import com.openmed.openmedkit.bundle.DeviceCapabilityProfile
import com.openmed.openmedkit.bundle.DeviceCapabilityReason
import com.openmed.openmedkit.bundle.DeviceCapabilityVocabulary

/**
 * Coarse Android facts that feed a normalized device-capability profile.
 *
 * Only the API level, the primary ABI, the [AcceleratorProvider] set that
 * [AcceleratorSession] discovers, and caller-supplied budgets are accepted.
 * There is deliberately no field for `Build.MODEL`, `Build.FINGERPRINT`,
 * serials, or any other identity-bearing value.
 */
data class AndroidCapabilityInputs(
    val sdkInt: Int,
    val abi: String,
    val availableProviders: Set<AcceleratorProvider>,
    /** `ActivityManager.getMemoryClass()` in MB, or `null` when unknown. */
    val memoryClassMb: Int? = null,
    val storageFreeBytes: Long? = null,
    /** Optional app classification used to cap the model tier. */
    val deviceTier: AndroidDeviceTier? = null,
    val maxSequenceLength: Long? = null,
    /** Highest ONNX opset the bundled ONNX Runtime supports, when known. */
    val operatorSets: Map<String, Long> = emptyMap(),
    val network: String = "unknown",
)

/**
 * Maps Android accelerator and platform probes into the shared profile.
 *
 * Mirrors `profile_from_android_probe` in `openmed/core/device_capabilities.py`.
 */
object AndroidCapabilityProbe {
    private const val NNAPI_MIN_SDK = 27
    private val ABIS = mapOf(
        "arm64-v8a" to "arm64",
        "armeabi-v7a" to "armv7",
        "x86" to "unknown",
        "x86_64" to "x86_64",
    )
    private val TIERS = mapOf(
        AndroidDeviceTier.ENTRY to "nano",
        AndroidDeviceTier.MID_RANGE to "tiny",
        AndroidDeviceTier.FLAGSHIP to "base",
        AndroidDeviceTier.DEVICE_FARM_STUB to null,
    )

    /** Normalize coarse inputs; NNAPI counts only from API 27, as in [AcceleratorSession]. */
    fun profile(inputs: AndroidCapabilityInputs, sdkVersion: String): DeviceCapabilityProfile {
        if (inputs.sdkInt < 0 || (inputs.memoryClassMb != null && inputs.memoryClassMb < 0)) {
            throw DeviceCapabilityException(DeviceCapabilityReason.PROBE_INVALID)
        }
        val providers = buildSet {
            add("cpu")
            if (AcceleratorProvider.QNN in inputs.availableProviders) add("qnn")
            if (AcceleratorProvider.NNAPI in inputs.availableProviders && inputs.sdkInt >= NNAPI_MIN_SDK) add("nnapi")
        }
        val architecture = ABIS[inputs.abi] ?: "unknown"
        return try {
            DeviceCapabilityProfile(
                platform = "android",
                sdkName = DeviceCapabilityVocabulary.ANDROID_SDK_NAME,
                sdkVersion = sdkVersion,
                osVersion = inputs.sdkInt.toString(),
                cpuArchitecture = architecture,
                providers = providers,
                runtimes = listOf("onnx", "ort"),
                precisions = listOf("fp16", "fp32", "int4", "int8"),
                features = if (architecture == "arm64") listOf("cpu_neon") else emptyList(),
                operatorSets = inputs.operatorSets,
                memoryBudgetBytes = DeviceCapabilityVocabulary.budgetBucket(
                    inputs.memoryClassMb?.let { it.toLong() shl 20 },
                ),
                storageBudgetBytes = DeviceCapabilityVocabulary.budgetBucket(inputs.storageFreeBytes),
                maxSequenceLength = inputs.maxSequenceLength,
                maxModelTier = inputs.deviceTier?.let { TIERS[it] },
                network = inputs.network,
            )
        } catch (error: DeviceCapabilityException) {
            throw DeviceCapabilityException(DeviceCapabilityReason.PROBE_INVALID)
        }
    }

    /**
     * Probe the running device: API level, primary ABI, and the ONNX Runtime
     * providers [AcceleratorSession] would try. Budgets come from the caller
     * (for example `ActivityManager.getMemoryClass()` and the cache volume).
     */
    fun current(
        sdkVersion: String,
        memoryClassMb: Int? = null,
        storageFreeBytes: Long? = null,
        deviceTier: AndroidDeviceTier? = null,
        operatorSets: Map<String, Long> = emptyMap(),
        network: String = "unknown",
    ): DeviceCapabilityProfile = profile(
        AndroidCapabilityInputs(
            sdkInt = Build.VERSION.SDK_INT,
            abi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty(),
            availableProviders = availableAcceleratorProviders(),
            memoryClassMb = memoryClassMb,
            storageFreeBytes = storageFreeBytes,
            deviceTier = deviceTier,
            operatorSets = operatorSets,
            network = network,
        ),
        sdkVersion,
    )
}
