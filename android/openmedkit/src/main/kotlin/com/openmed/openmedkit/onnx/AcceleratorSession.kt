package com.openmed.openmedkit.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.os.Build
import java.io.Closeable
import java.io.File
import java.util.EnumSet
import kotlin.math.abs
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Execution providers supported by the Android accelerator session. */
public enum class AcceleratorProvider {
    QNN,
    NNAPI,
    CPU,
}

/** Stable outcomes emitted while choosing an Android execution provider. */
public enum class AcceleratorAttemptOutcome {
    SELECTED,
    NOT_AVAILABLE,
    NO_SUPPORTED_OPERATORS,
    SESSION_CREATION_FAILED,
}

/** One privacy-safe provider selection attempt. */
public data class AcceleratorProviderAttempt(
    val provider: AcceleratorProvider,
    val outcome: AcceleratorAttemptOutcome,
)

/**
 * Required operators and device-observed support for one model family.
 *
 * A missing provider key means coverage is unknown and ONNX Runtime performs
 * its normal graph capability/partitioning pass. A present key, including an
 * empty set, is treated as measured device-farm evidence.
 */
public data class ModelFamilyOperatorCoverage(
    val family: String,
    val requiredOperators: Set<String>,
    val supportedOperators: Map<AcceleratorProvider, Set<String>> = emptyMap(),
) {
    init {
        require(family.isNotBlank()) { "family must not be blank" }
        require(requiredOperators.none(String::isBlank)) {
            "requiredOperators must not contain blank names"
        }
        require(supportedOperators.values.flatten().none(String::isBlank)) {
            "supportedOperators must not contain blank names"
        }
    }

    public companion object {
        /**
         * Load a model family and operator list from an OpenMed ONNX manifest.
         */
        public fun fromManifest(
            manifestFile: File,
            modelFileName: String = "model.onnx",
            supportedOperators: Map<AcceleratorProvider, Set<String>> = emptyMap(),
        ): ModelFamilyOperatorCoverage {
            if (!manifestFile.isFile) {
                throw InferenceError.InvalidInput("ONNX manifest does not exist")
            }
            if (modelFileName.isBlank()) {
                throw InferenceError.InvalidInput("modelFileName must not be blank")
            }

            val root = try {
                Json.parseToJsonElement(manifestFile.readText()).jsonObject
            } catch (error: Exception) {
                throw InferenceError.InvalidInput(
                    "ONNX manifest must contain a JSON object"
                )
            }
            val family = try {
                root["family"]?.jsonPrimitive?.contentOrNull
                    ?.takeIf(String::isNotBlank)
                    ?: "unknown"
            } catch (error: Exception) {
                throw InferenceError.InvalidInput(
                    "ONNX manifest family must be a string"
                )
            }
            val artifact = try {
                root["artifacts"]?.jsonArray
                    ?.map { it.jsonObject }
                    ?.firstOrNull { entry ->
                        entry["path"]?.jsonPrimitive?.contentOrNull == modelFileName
                    }
            } catch (error: Exception) {
                throw InferenceError.InvalidInput(
                    "ONNX manifest artifacts must be an array of objects"
                )
            } ?: throw InferenceError.InvalidInput(
                "ONNX manifest does not describe $modelFileName"
            )
            val operators = try {
                artifact["metadata"]?.jsonObject
                    ?.get("operators")?.jsonArray
                    ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                    ?.toSet()
                    .orEmpty()
            } catch (error: Exception) {
                throw InferenceError.InvalidInput(
                    "ONNX manifest operators must be an array of strings"
                )
            }
            if (operators.isEmpty()) {
                throw InferenceError.InvalidInput(
                    "ONNX manifest does not record operators for $modelFileName"
                )
            }
            return ModelFamilyOperatorCoverage(
                family = family,
                requiredOperators = operators,
                supportedOperators = supportedOperators,
            )
        }
    }
}

/** Configuration for deterministic QNN, NNAPI, and CPU selection. */
public data class AcceleratorConfig(
    val preferredProviders: List<AcceleratorProvider> = listOf(
        AcceleratorProvider.QNN,
        AcceleratorProvider.NNAPI,
        AcceleratorProvider.CPU,
    ),
    val qnnOptions: Map<String, String> = mapOf(
        "backend_path" to "libQnnHtp.so",
    ),
    val nnapiAllowFp16: Boolean = false,
    val nnapiUseNchw: Boolean = false,
    val intraOpThreadCount: Int = 1,
    val modelCoverage: ModelFamilyOperatorCoverage? = null,
) {
    init {
        require(preferredProviders.isNotEmpty()) {
            "preferredProviders must not be empty"
        }
        require(qnnOptions.keys.none(String::isBlank)) {
            "qnnOptions must not contain blank keys"
        }
        require(intraOpThreadCount > 0) {
            "intraOpThreadCount must be greater than zero"
        }
    }

    public companion object {
        /** A CPU-only configuration for deterministic baseline and parity runs. */
        public fun cpuOnly(intraOpThreadCount: Int = 1): AcceleratorConfig =
            AcceleratorConfig(
                preferredProviders = listOf(AcceleratorProvider.CPU),
                intraOpThreadCount = intraOpThreadCount,
            )
    }
}

/** Operator partition selected for a model family and provider. */
public data class AcceleratorOperatorCoverage(
    val family: String,
    val provider: AcceleratorProvider,
    val requiredOperators: Set<String>,
    val providerOperators: Set<String>,
    val cpuFallbackOperators: Set<String>,
    val known: Boolean,
) {
    /** Whether measured coverage assigns any operators to the CPU partition. */
    public val usesCpuPartition: Boolean
        get() = known && cpuFallbackOperators.isNotEmpty()

    /** Whether measured coverage assigns every required operator to the provider. */
    public val fullyCovered: Boolean
        get() = known && cpuFallbackOperators.isEmpty()
}

/** Result of the capability probe and provider/session creation sequence. */
public data class AcceleratorSelection(
    val selectedProvider: AcceleratorProvider,
    val availableProviders: Set<AcceleratorProvider>,
    val operatorCoverage: AcceleratorOperatorCoverage,
    val attempts: List<AcceleratorProviderAttempt>,
) {
    /** True when a QNN or NNAPI session was selected. */
    public val isAccelerated: Boolean
        get() = selectedProvider != AcceleratorProvider.CPU
}

/** Device tiers used by accelerator latency evidence. */
public enum class AndroidDeviceTier {
    ENTRY,
    MID_RANGE,
    FLAGSHIP,
    DEVICE_FARM_STUB,
}

/** Privacy-safe CPU/delegate latency evidence for one Android device tier. */
public data class DeviceTierLatencyRecord(
    val deviceTier: AndroidDeviceTier,
    val provider: AcceleratorProvider,
    val cpuP50Milliseconds: Double,
    val delegateP50Milliseconds: Double,
    val sampleCount: Int,
) {
    init {
        require(provider != AcceleratorProvider.CPU) {
            "latency record provider must be QNN or NNAPI"
        }
        require(cpuP50Milliseconds >= 0.0) {
            "cpuP50Milliseconds must be non-negative"
        }
        require(delegateP50Milliseconds >= 0.0) {
            "delegateP50Milliseconds must be non-negative"
        }
        require(sampleCount > 0) { "sampleCount must be greater than zero" }
    }

    /** CPU latency divided by delegate latency. */
    public val speedup: Double
        get() = if (delegateP50Milliseconds == 0.0) {
            Double.POSITIVE_INFINITY
        } else {
            cpuP50Milliseconds / delegateP50Milliseconds
        }
}

/** PHI-free span signature used for CPU/delegate parity evidence. */
public data class AcceleratorSpanSignature(
    val label: String,
    val startOffset: Int,
    val endOffset: Int,
) {
    init {
        require(label.isNotBlank()) { "label must not be blank" }
        require(startOffset >= 0) { "startOffset must be non-negative" }
        require(endOffset >= startOffset) {
            "endOffset must be greater than or equal to startOffset"
        }
    }
}

/**
 * Device-farm evidence gate comparing accelerator output with the CPU baseline.
 *
 * Only labels, offsets, aggregate recall, and latency are retained; raw input
 * text and span surface text are deliberately excluded.
 */
public data class AcceleratorValidationRecord(
    val latency: DeviceTierLatencyRecord,
    val cpuSpans: List<AcceleratorSpanSignature>,
    val delegateSpans: List<AcceleratorSpanSignature>,
    val cpuRecall: Double,
    val delegateRecall: Double,
    val boundaryToleranceCharacters: Int = 0,
    val maxRecallDrop: Double = 0.0,
) {
    init {
        require(cpuRecall in 0.0..1.0) { "cpuRecall must be between 0 and 1" }
        require(delegateRecall in 0.0..1.0) {
            "delegateRecall must be between 0 and 1"
        }
        require(boundaryToleranceCharacters >= 0) {
            "boundaryToleranceCharacters must be non-negative"
        }
        require(maxRecallDrop >= 0.0) { "maxRecallDrop must be non-negative" }
    }

    /** Delegate recall minus CPU recall. */
    public val recallDelta: Double
        get() = delegateRecall - cpuRecall

    /** Whether labels and boundaries match within the configured tolerance. */
    public val spansWithinTolerance: Boolean
        get() = cpuSpans.size == delegateSpans.size &&
            cpuSpans.zip(delegateSpans).all { (cpu, delegate) ->
                cpu.label == delegate.label &&
                    abs(cpu.startOffset - delegate.startOffset) <=
                    boundaryToleranceCharacters &&
                    abs(cpu.endOffset - delegate.endOffset) <=
                    boundaryToleranceCharacters
            }

    /** Whether delegate recall stays inside the configured CPU-relative budget. */
    public val recallWithinTolerance: Boolean
        get() = recallDelta + RECALL_COMPARISON_EPSILON >= -maxRecallDrop

    /** Whether both span parity and recall-delta gates pass. */
    public val passed: Boolean
        get() = spansWithinTolerance && recallWithinTolerance

    /** Reject evidence that does not meet span and recall-delta tolerances. */
    public fun requirePassing(): AcceleratorValidationRecord {
        check(spansWithinTolerance) {
            "delegate spans do not match the CPU baseline within tolerance"
        }
        check(recallWithinTolerance) {
            "delegate recall delta exceeds the CPU-relative tolerance"
        }
        return this
    }

    private companion object {
        const val RECALL_COMPARISON_EPSILON = 1e-12
    }
}

/**
 * ONNX token-classification session with QNN/NNAPI preference and CPU fallback.
 *
 * ONNX Runtime keeps the CPU execution provider enabled for unsupported graph
 * partitions. If provider registration or session creation fails, this class
 * retries the next configured provider and always ends with a CPU attempt.
 */
public class AcceleratorSession private constructor(
    private val classifier: OnnxTokenClassifier,
    public val selection: AcceleratorSelection,
) : Closeable {
    public constructor(
        modelFile: File,
        id2LabelFile: File,
        config: AcceleratorConfig = AcceleratorConfig(),
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(
        initialize(modelFile, id2LabelFile, config),
        inputTensorType,
    )

    public constructor(
        modelFile: File,
        id2Label: Map<Int, String>,
        config: AcceleratorConfig = AcceleratorConfig(),
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(
        initialize(modelFile, id2Label, config),
        inputTensorType,
    )

    public constructor(
        modelBytes: ByteArray,
        id2Label: Map<Int, String>,
        config: AcceleratorConfig = AcceleratorConfig(),
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(
        initialize(modelBytes, id2Label, config),
        inputTensorType,
    )

    private constructor(
        initialized: InitializedAcceleratorSession,
        inputTensorType: TensorElementType,
    ) : this(
        initialized.runtime,
        initialized.id2Label,
        inputTensorType,
    )

    private constructor(
        runtime: AcceleratorRuntimeComponents,
        id2Label: Map<Int, String>,
        inputTensorType: TensorElementType,
    ) : this(
        OnnxTokenClassifier(
            environment = runtime.environment,
            session = runtime.session,
            id2Label = id2Label,
            inputTensorType = inputTensorType,
            ownsEnvironment = runtime.environment != null,
        ),
        runtime.selection,
    )

    /** Run token classification off the main thread. */
    public suspend fun run(
        inputIds: IntArray,
        attentionMask: IntArray,
        offsets: List<TokenOffset>,
    ): List<TokenPrediction> = classifier.run(inputIds, attentionMask, offsets)

    public override fun close() {
        classifier.close()
    }

    internal companion object {
        internal fun createForTesting(
            id2Label: Map<Int, String>,
            config: AcceleratorConfig,
            availableProviders: Set<AcceleratorProvider>,
            sessionFactory: AcceleratorTokenSessionFactory,
            inputTensorType: TensorElementType = TensorElementType.INT64,
        ): AcceleratorSession {
            val labels = validateId2Label(id2Label)
            val selected = selectSession(
                config = config,
                availableProviders = availableProviders + AcceleratorProvider.CPU,
                sessionFactory = sessionFactory,
            )
            return AcceleratorSession(
                runtime = AcceleratorRuntimeComponents(
                    environment = null,
                    session = selected.session,
                    selection = selected.selection,
                ),
                id2Label = labels,
                inputTensorType = inputTensorType,
            )
        }

        private fun initialize(
            modelFile: File,
            id2LabelFile: File,
            config: AcceleratorConfig,
        ): InitializedAcceleratorSession {
            val labels = OnnxTokenClassifier.loadId2Label(id2LabelFile)
            return InitializedAcceleratorSession(
                runtime = createRuntime(modelFile, config),
                id2Label = labels,
            )
        }

        private fun initialize(
            modelFile: File,
            id2Label: Map<Int, String>,
            config: AcceleratorConfig,
        ): InitializedAcceleratorSession {
            val labels = validateId2Label(id2Label)
            return InitializedAcceleratorSession(
                runtime = createRuntime(modelFile, config),
                id2Label = labels,
            )
        }

        private fun initialize(
            modelBytes: ByteArray,
            id2Label: Map<Int, String>,
            config: AcceleratorConfig,
        ): InitializedAcceleratorSession {
            val labels = validateId2Label(id2Label)
            return InitializedAcceleratorSession(
                runtime = createRuntime(modelBytes, config),
                id2Label = labels,
            )
        }

        private fun createRuntime(
            modelFile: File,
            config: AcceleratorConfig,
        ): AcceleratorRuntimeComponents {
            if (!modelFile.isFile) {
                throw InferenceError.InvalidInput("model file does not exist")
            }
            val effectiveConfig = config.withDiscoveredCoverage(modelFile)
            return createRuntime(effectiveConfig) { environment, options ->
                environment.createSession(modelFile.absolutePath, options)
            }
        }

        private fun createRuntime(
            modelBytes: ByteArray,
            config: AcceleratorConfig,
        ): AcceleratorRuntimeComponents {
            if (modelBytes.isEmpty()) {
                throw InferenceError.InvalidInput("modelBytes must not be empty")
            }
            return createRuntime(config) { environment, options ->
                environment.createSession(modelBytes, options)
            }
        }

        private fun createRuntime(
            config: AcceleratorConfig,
            createOrtSession: (OrtEnvironment, OrtSession.SessionOptions) -> OrtSession,
        ): AcceleratorRuntimeComponents {
            val environment = OrtEnvironment.getEnvironment()
            return try {
                val availableProviders = availableAcceleratorProviders()
                val selected = selectSession(
                    config = config,
                    availableProviders = availableProviders,
                    sessionFactory = AcceleratorTokenSessionFactory { provider ->
                        createSessionOptions(config, provider).use { options ->
                            val session = createOrtSession(environment, options)
                            OnnxRuntimeTokenClassificationSession(environment, session)
                        }
                    },
                )
                AcceleratorRuntimeComponents(
                    environment = environment,
                    session = selected.session,
                    selection = selected.selection,
                )
            } catch (error: Throwable) {
                try {
                    environment.close()
                } catch (closeError: Throwable) {
                    error.addSuppressed(closeError)
                }
                throw error
            }
        }

        private fun validateId2Label(id2Label: Map<Int, String>): Map<Int, String> {
            if (id2Label.isEmpty()) {
                throw InferenceError.InvalidInput("id2Label must not be empty")
            }
            return id2Label.toMap()
        }
    }
}

internal fun interface AcceleratorTokenSessionFactory {
    fun create(provider: AcceleratorProvider): TokenClassificationSession
}

private data class SelectedTokenSession(
    val session: TokenClassificationSession,
    val selection: AcceleratorSelection,
)

private data class AcceleratorRuntimeComponents(
    val environment: OrtEnvironment?,
    val session: TokenClassificationSession,
    val selection: AcceleratorSelection,
)

private data class InitializedAcceleratorSession(
    val runtime: AcceleratorRuntimeComponents,
    val id2Label: Map<Int, String>,
)

private fun selectSession(
    config: AcceleratorConfig,
    availableProviders: Set<AcceleratorProvider>,
    sessionFactory: AcceleratorTokenSessionFactory,
): SelectedTokenSession {
    val attempts = mutableListOf<AcceleratorProviderAttempt>()
    val candidates = (config.preferredProviders + AcceleratorProvider.CPU).distinct()

    candidates.forEach { provider ->
        if (provider !in availableProviders) {
            attempts += AcceleratorProviderAttempt(
                provider,
                AcceleratorAttemptOutcome.NOT_AVAILABLE,
            )
            return@forEach
        }

        val coverage = operatorCoverage(config.modelCoverage, provider)
        if (
            provider != AcceleratorProvider.CPU &&
            coverage.known &&
            coverage.requiredOperators.isNotEmpty() &&
            coverage.providerOperators.isEmpty()
        ) {
            attempts += AcceleratorProviderAttempt(
                provider,
                AcceleratorAttemptOutcome.NO_SUPPORTED_OPERATORS,
            )
            return@forEach
        }

        try {
            val session = sessionFactory.create(provider)
            val completedAttempts = attempts + AcceleratorProviderAttempt(
                provider,
                AcceleratorAttemptOutcome.SELECTED,
            )
            return SelectedTokenSession(
                session = session,
                selection = AcceleratorSelection(
                    selectedProvider = provider,
                    availableProviders = availableProviders.toSet(),
                    operatorCoverage = coverage,
                    attempts = completedAttempts,
                ),
            )
        } catch (error: Exception) {
            if (provider == AcceleratorProvider.CPU) {
                throw InferenceError.SessionCreation(error)
            }
            attempts += AcceleratorProviderAttempt(
                provider,
                AcceleratorAttemptOutcome.SESSION_CREATION_FAILED,
            )
        } catch (error: UnsatisfiedLinkError) {
            if (provider == AcceleratorProvider.CPU) {
                throw InferenceError.SessionCreation(error)
            }
            attempts += AcceleratorProviderAttempt(
                provider,
                AcceleratorAttemptOutcome.SESSION_CREATION_FAILED,
            )
        }
    }

    throw InferenceError.SessionCreation()
}

private fun operatorCoverage(
    modelCoverage: ModelFamilyOperatorCoverage?,
    provider: AcceleratorProvider,
): AcceleratorOperatorCoverage {
    val family = modelCoverage?.family ?: "unknown"
    val required = modelCoverage?.requiredOperators.orEmpty()
    if (provider == AcceleratorProvider.CPU) {
        return AcceleratorOperatorCoverage(
            family = family,
            provider = provider,
            requiredOperators = required,
            providerOperators = required,
            cpuFallbackOperators = emptySet(),
            known = true,
        )
    }

    val support = modelCoverage?.supportedOperators?.get(provider)
        ?: return AcceleratorOperatorCoverage(
            family = family,
            provider = provider,
            requiredOperators = required,
            providerOperators = emptySet(),
            cpuFallbackOperators = emptySet(),
            known = false,
        )
    val delegated = required.intersect(support)
    return AcceleratorOperatorCoverage(
        family = family,
        provider = provider,
        requiredOperators = required,
        providerOperators = delegated,
        cpuFallbackOperators = required - delegated,
        known = true,
    )
}

private fun AcceleratorConfig.withDiscoveredCoverage(modelFile: File): AcceleratorConfig {
    if (modelCoverage != null) {
        return this
    }
    val manifestFile = File(modelFile.parentFile, "openmed-onnx.json")
    if (!manifestFile.isFile) {
        return this
    }
    val discovered = try {
        ModelFamilyOperatorCoverage.fromManifest(
            manifestFile = manifestFile,
            modelFileName = modelFile.name,
        )
    } catch (error: InferenceError.InvalidInput) {
        null
    }
    return if (discovered == null) this else copy(modelCoverage = discovered)
}

private fun availableAcceleratorProviders(): Set<AcceleratorProvider> {
    val ortProviders = OrtEnvironment.getAvailableProviders()
    return buildSet {
        add(AcceleratorProvider.CPU)
        if (OrtProvider.QNN in ortProviders) {
            add(AcceleratorProvider.QNN)
        }
        if (
            OrtProvider.NNAPI in ortProviders &&
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1
        ) {
            add(AcceleratorProvider.NNAPI)
        }
    }
}

private fun createSessionOptions(
    config: AcceleratorConfig,
    provider: AcceleratorProvider,
): OrtSession.SessionOptions {
    val options = OrtSession.SessionOptions()
    try {
        options.setIntraOpNumThreads(config.intraOpThreadCount)
        when (provider) {
            AcceleratorProvider.QNN -> options.addQnn(config.qnnOptions)
            AcceleratorProvider.NNAPI -> options.addNnapi(config.nnapiFlags())
            AcceleratorProvider.CPU -> Unit
        }
        return options
    } catch (error: Throwable) {
        options.close()
        throw error
    }
}

private fun AcceleratorConfig.nnapiFlags(): EnumSet<NNAPIFlags> =
    EnumSet.noneOf(NNAPIFlags::class.java).also { flags ->
        if (nnapiAllowFp16) {
            flags.add(NNAPIFlags.USE_FP16)
        }
        if (nnapiUseNchw) {
            flags.add(NNAPIFlags.USE_NCHW)
        }
    }
