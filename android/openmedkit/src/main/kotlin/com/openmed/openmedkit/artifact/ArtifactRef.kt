package com.openmed.openmedkit.artifact

import com.openmed.openmedkit.catalog.ModelCatalogEntry

/**
 * On-disk formats an OpenMed model artifact can be published in.
 *
 * Android backends open [ONNX] and [TFLITE] artifacts. [SIGNED_PORTABLE_BUNDLE]
 * is reserved for the signed cross-platform bundle and stays unsupported until
 * its verifier ships. [MLX] and [CORE_ML] are Apple-only formats: Android never
 * acquires them and no Android API runs them.
 */
enum class ArtifactFormat(val wireName: String) {
    ONNX("onnx"),
    TFLITE("tflite"),
    SIGNED_PORTABLE_BUNDLE("signed-portable-bundle"),
    MLX("mlx"),
    CORE_ML("coreml"),
}

/** Inference runtime an artifact is built for. */
enum class ArtifactRuntime(val wireName: String) {
    ONNX_RUNTIME("onnxruntime"),
    TFLITE("tflite"),
    MLX("mlx"),
    CORE_ML("coreml"),
}

/**
 * Minimum backend requirements an artifact declares.
 *
 * A requirement the backend declares no matching capability for is treated as
 * unsupported, so compatibility is proven rather than assumed.
 *
 * @property minRuntimeVersion Lowest runtime version, dotted numeric (`1.17.0`).
 * @property opsetVersion ONNX opset the graph needs.
 * @property minAndroidSdk Lowest Android API level the artifact supports.
 */
data class ArtifactCompatibility(
    val minRuntimeVersion: String? = null,
    val opsetVersion: Int? = null,
    val minAndroidSdk: Int? = null,
) {
    init {
        require(minRuntimeVersion == null || VERSION_PATTERN.matches(minRuntimeVersion)) {
            "minRuntimeVersion must be a dotted numeric version"
        }
        require((opsetVersion ?: 1) > 0 && (minAndroidSdk ?: 1) > 0) {
            "opsetVersion and minAndroidSdk must be positive"
        }
    }
}

/**
 * Backend-neutral reference to one immutable model artifact.
 *
 * @property repoId Provider-scoped artifact identifier; a repository ID for the
 *   Hugging Face adapter.
 * @property format On-disk format to acquire.
 * @property runtime Runtime the artifact is built for.
 * @property digest `sha256:` manifest digest (the catalog reproducibility hash)
 *   pinning the repository, its commit, and its file listing. Every file that
 *   publishes a SHA-256 is also verified individually.
 * @property revision Revision to resolve; prefer an immutable commit SHA. When
 *   `null`, the provider default is used and [digest] fails closed if it moved.
 * @property sizeBytes Expected total bytes of the acquired files, when known.
 * @property compatibility Requirements checked against the backend before any
 *   cache or network access.
 */
data class ArtifactRef(
    val repoId: String,
    val format: ArtifactFormat,
    val runtime: ArtifactRuntime,
    val digest: String,
    val revision: String? = null,
    val sizeBytes: Long? = null,
    val compatibility: ArtifactCompatibility = ArtifactCompatibility(),
) {
    init {
        require(REPO_ID_PATTERN.matches(repoId) && ".." !in repoId) {
            "repoId is not a valid artifact identifier"
        }
        require(DIGEST_PATTERN.matches(digest)) { "digest must be sha256: and 64 hex characters" }
        require(revision == null || (REVISION_PATTERN.matches(revision) && ".." !in revision)) {
            "revision is not a valid revision identifier"
        }
        require((sizeBytes ?: 1L) > 0L) { "sizeBytes must be positive" }
    }

    internal val identity: String
        get() = "$repoId\u0000${revision.orEmpty()}\u0000${format.wireName}\u0000" +
            "${runtime.wireName}\u0000$digest"

    companion object {
        /**
         * Reference one published format of a catalog entry.
         *
         * The catalog reproducibility hash becomes the artifact [digest]. A
         * format the entry does not publish resolves as unsupported during
         * acquisition instead of downloading anything else.
         */
        @JvmStatic
        fun fromCatalogEntry(
            entry: ModelCatalogEntry,
            format: ArtifactFormat,
            runtime: ArtifactRuntime,
            revision: String? = null,
            sizeBytes: Long? = null,
        ): ArtifactRef = ArtifactRef(
            repoId = entry.repoId,
            format = format,
            runtime = runtime,
            digest = entry.reproducibilityHash,
            revision = revision,
            sizeBytes = sizeBytes,
        )
    }
}

/**
 * What the configured Android inference backend can open.
 *
 * @property runtime The backend runtime.
 * @property formats Formats the backend opens; only ONNX and TFLite run on Android.
 * @property runtimeVersion Runtime version, dotted numeric, when known.
 * @property maxOpsetVersion Highest ONNX opset the runtime executes, when known.
 * @property androidSdkLevel Device API level, typically `Build.VERSION.SDK_INT`.
 */
class ArtifactBackendCapability(
    val runtime: ArtifactRuntime,
    val formats: Set<ArtifactFormat>,
    val runtimeVersion: String? = null,
    val maxOpsetVersion: Int? = null,
    val androidSdkLevel: Int? = null,
) {
    init {
        require(runtime in ANDROID_RUNTIMES) { "runtime is not an Android inference runtime" }
        require(formats.isNotEmpty() && formats.all { it in ANDROID_FORMATS }) {
            "Android backends open only ONNX or TFLite artifacts"
        }
        require(runtimeVersion == null || VERSION_PATTERN.matches(runtimeVersion)) {
            "runtimeVersion must be a dotted numeric version"
        }
    }

    /** First reason [ref] cannot run on this backend, or `null` when it can. */
    internal fun incompatibility(ref: ArtifactRef): ArtifactAcquisitionReason? {
        val minRuntimeVersion = ref.compatibility.minRuntimeVersion
        val opsetVersion = ref.compatibility.opsetVersion
        val minAndroidSdk = ref.compatibility.minAndroidSdk
        return when {
            ref.format == ArtifactFormat.SIGNED_PORTABLE_BUNDLE ->
                ArtifactAcquisitionReason.BUNDLE_VERIFIER_UNAVAILABLE
            ref.format !in ANDROID_FORMATS || ref.runtime !in ANDROID_RUNTIMES ->
                ArtifactAcquisitionReason.FORMAT_NOT_ANDROID_RUNNABLE
            ref.format !in formats -> ArtifactAcquisitionReason.FORMAT_NOT_SUPPORTED_BY_BACKEND
            ref.runtime != runtime -> ArtifactAcquisitionReason.RUNTIME_MISMATCH
            (minRuntimeVersion != null && runtimeVersion == null) ||
                (opsetVersion != null && maxOpsetVersion == null) ||
                (minAndroidSdk != null && androidSdkLevel == null) ->
                ArtifactAcquisitionReason.BACKEND_CAPABILITY_UNDECLARED
            minRuntimeVersion != null &&
                compareVersions(runtimeVersion.orEmpty(), minRuntimeVersion) < 0 ->
                ArtifactAcquisitionReason.RUNTIME_VERSION_TOO_OLD
            opsetVersion != null && opsetVersion > (maxOpsetVersion ?: 0) ->
                ArtifactAcquisitionReason.OPSET_NOT_SUPPORTED
            minAndroidSdk != null && minAndroidSdk > (androidSdkLevel ?: 0) ->
                ArtifactAcquisitionReason.ANDROID_SDK_TOO_OLD
            else -> null
        }
    }

    companion object {
        /** Capability of the bundled ONNX Runtime Android backend. */
        @JvmStatic
        fun onnxRuntime(
            runtimeVersion: String? = null,
            maxOpsetVersion: Int? = null,
            androidSdkLevel: Int? = null,
        ): ArtifactBackendCapability = ArtifactBackendCapability(
            ArtifactRuntime.ONNX_RUNTIME,
            setOf(ArtifactFormat.ONNX),
            runtimeVersion,
            maxOpsetVersion,
            androidSdkLevel,
        )

        /** Capability of a caller-supplied TensorFlow Lite backend. */
        @JvmStatic
        fun tfLite(
            runtimeVersion: String? = null,
            androidSdkLevel: Int? = null,
        ): ArtifactBackendCapability = ArtifactBackendCapability(
            ArtifactRuntime.TFLITE,
            setOf(ArtifactFormat.TFLITE),
            runtimeVersion,
            null,
            androidSdkLevel,
        )
    }
}

private fun compareVersions(lhs: String, rhs: String): Int {
    val left = lhs.split('.')
    val right = rhs.split('.')
    for (index in 0 until maxOf(left.size, right.size)) {
        val difference = (left.getOrNull(index)?.toInt() ?: 0)
            .compareTo(right.getOrNull(index)?.toInt() ?: 0)
        if (difference != 0) {
            return difference
        }
    }
    return 0
}

private val ANDROID_FORMATS = setOf(ArtifactFormat.ONNX, ArtifactFormat.TFLITE)
private val ANDROID_RUNTIMES = setOf(ArtifactRuntime.ONNX_RUNTIME, ArtifactRuntime.TFLITE)
private val VERSION_PATTERN = Regex("""\d{1,6}(\.\d{1,6}){0,3}""")
private val DIGEST_PATTERN = Regex("""sha256:[0-9a-f]{64}""")
private val REPO_ID_PATTERN =
    Regex("""[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?""")
private val REVISION_PATTERN = Regex("""[A-Za-z0-9][A-Za-z0-9._-]{0,127}""")
