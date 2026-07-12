package com.openmed.openmedkit.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.exp

public enum class TensorElementType {
    INT64,
    INT32,
}

public class OnnxTokenClassifier internal constructor(
    private val environment: OrtEnvironment?,
    private val session: TokenClassificationSession,
    id2Label: Map<Int, String>,
    private val inputTensorType: TensorElementType = TensorElementType.INT64,
    private val ownsEnvironment: Boolean = false,
) : Closeable {
    private val id2Label: Map<Int, String> = id2Label.toMap()

    @Volatile
    private var closed = false

    public constructor(
        modelPath: String,
        id2LabelPath: String,
        intraOpThreadCount: Int = 1,
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(
        File(modelPath),
        File(id2LabelPath),
        intraOpThreadCount,
        inputTensorType,
    )

    public constructor(
        modelFile: File,
        id2LabelFile: File,
        intraOpThreadCount: Int = 1,
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(
        createRuntime(modelFile, intraOpThreadCount),
        loadId2Label(id2LabelFile),
        inputTensorType,
    )

    public constructor(
        modelBytes: ByteArray,
        id2LabelFile: File,
        intraOpThreadCount: Int = 1,
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(
        createRuntime(modelBytes, intraOpThreadCount),
        loadId2Label(id2LabelFile),
        inputTensorType,
    )

    private constructor(
        runtime: RuntimeComponents,
        id2Label: Map<Int, String>,
        inputTensorType: TensorElementType,
    ) : this(
        runtime.environment,
        runtime.session,
        id2Label,
        inputTensorType,
        ownsEnvironment = true,
    )

    public suspend fun run(
        inputIds: IntArray,
        attentionMask: IntArray,
        offsets: List<TokenOffset>,
    ): List<TokenPrediction> = withContext(Dispatchers.Default) {
        currentCoroutineContext().ensureActive()
        val predictions = runBlocking(inputIds, attentionMask, offsets)
        currentCoroutineContext().ensureActive()
        predictions
    }

    public override fun close() {
        if (closed) {
            return
        }
        closed = true

        var closeFailure: Throwable? = null
        try {
            session.close()
        } catch (error: Throwable) {
            closeFailure = error
        }

        if (ownsEnvironment) {
            try {
                environment?.close()
            } catch (error: Throwable) {
                if (closeFailure == null) {
                    closeFailure = error
                } else {
                    closeFailure.addSuppressed(error)
                }
            }
        }

        if (closeFailure != null) {
            throw closeFailure
        }
    }

    private fun runBlocking(
        inputIds: IntArray,
        attentionMask: IntArray,
        offsets: List<TokenOffset>,
    ): List<TokenPrediction> {
        ensureOpen()
        validateInputs(inputIds, attentionMask, offsets)

        val inputs = mutableMapOf(
            INPUT_IDS_NAME to createInputTensor(inputIds),
            ATTENTION_MASK_NAME to createInputTensor(attentionMask),
        )
        if (TOKEN_TYPE_IDS_NAME in session.inputNames) {
            inputs[TOKEN_TYPE_IDS_NAME] = createInputTensor(IntArray(inputIds.size))
        }
        val outputs = session.run(inputs)
        val logitsOutput = outputs[LOGITS_NAME]
            ?: throw InferenceError.MissingOutput(LOGITS_NAME)
        return decodePredictions(normalizeLogits(logitsOutput), offsets)
    }

    private fun ensureOpen() {
        if (closed) {
            throw InferenceError.InvalidInput("OnnxTokenClassifier is closed")
        }
    }

    private fun validateInputs(
        inputIds: IntArray,
        attentionMask: IntArray,
        offsets: List<TokenOffset>,
    ) {
        if (inputIds.isEmpty()) {
            throw InferenceError.InvalidInput("inputIds must not be empty")
        }
        if (inputIds.size != attentionMask.size) {
            throw InferenceError.InvalidInput(
                "inputIds and attentionMask must have the same length"
            )
        }
        if (inputIds.size != offsets.size) {
            throw InferenceError.InvalidInput(
                "offsets must have the same length as inputIds"
            )
        }
        offsets.forEachIndexed { index, offset ->
            if (offset.startOffset < 0 || offset.endOffset < offset.startOffset) {
                throw InferenceError.InvalidInput(
                    "offsets[$index] must satisfy 0 <= startOffset <= endOffset"
                )
            }
        }
    }

    private fun createInputTensor(values: IntArray): TokenInputTensor {
        return TokenInputTensor(
            shape = longArrayOf(1L, values.size.toLong()),
            elementType = inputTensorType,
            values = values.copyOf(),
        )
    }

    private fun decodePredictions(
        logits: Array<Array<FloatArray>>,
        offsets: List<TokenOffset>,
    ): List<TokenPrediction> {
        if (logits.size != 1) {
            throw InferenceError.InvalidOutput(
                "logits must have batch size 1; got ${logits.size}"
            )
        }

        val sequenceLogits = logits[0]
        if (sequenceLogits.size != offsets.size) {
            throw InferenceError.InvalidOutput(
                "logits sequence length ${sequenceLogits.size} does not match " +
                    "offset length ${offsets.size}"
            )
        }

        return sequenceLogits.mapIndexedNotNull { tokenIndex, labelLogits ->
            val offset = offsets[tokenIndex]
            if (offset.isSpecialToken) {
                return@mapIndexedNotNull null
            }
            decodeToken(labelLogits, offset)
        }
    }

    private fun decodeToken(
        labelLogits: FloatArray,
        offset: TokenOffset,
    ): TokenPrediction {
        if (labelLogits.isEmpty()) {
            throw InferenceError.InvalidOutput("logits label dimension must not be empty")
        }

        var labelId = 0
        var maxLogit = Float.NEGATIVE_INFINITY
        labelLogits.forEachIndexed { index, logit ->
            if (logit > maxLogit) {
                maxLogit = logit
                labelId = index
            }
        }

        var denominator = 0.0
        labelLogits.forEach { logit ->
            denominator += exp((logit - maxLogit).toDouble())
        }

        val score = (1.0 / denominator).toFloat()
        return TokenPrediction(
            labelId = labelId,
            label = id2Label[labelId] ?: "O",
            score = score,
            startOffset = offset.startOffset,
            endOffset = offset.endOffset,
        )
    }

    private fun normalizeLogits(value: Any): Array<Array<FloatArray>> {
        val batch = value as? Array<*>
            ?: throw InferenceError.InvalidOutput("logits must be a rank-3 tensor")
        if (batch.isEmpty()) {
            throw InferenceError.InvalidOutput("logits batch dimension must not be empty")
        }

        return Array(batch.size) { batchIndex ->
            val sequence = batch[batchIndex] as? Array<*>
                ?: throw InferenceError.InvalidOutput(
                    "logits[$batchIndex] must be a rank-2 tensor"
                )
            Array(sequence.size) { sequenceIndex ->
                toFloatArray(
                    sequence[sequenceIndex],
                    "logits[$batchIndex][$sequenceIndex]",
                )
            }
        }
    }

    private fun toFloatArray(value: Any?, path: String): FloatArray {
        return when (value) {
            is FloatArray -> value
            is DoubleArray -> FloatArray(value.size) { index -> value[index].toFloat() }
            is Array<*> -> FloatArray(value.size) { index ->
                val element = value[index]
                if (element !is Number) {
                    throw InferenceError.InvalidOutput("$path[$index] must be numeric")
                }
                element.toFloat()
            }
            else -> throw InferenceError.InvalidOutput("$path must be a numeric vector")
        }
    }

    internal companion object {
        internal const val INPUT_IDS_NAME = "input_ids"
        internal const val ATTENTION_MASK_NAME = "attention_mask"
        internal const val TOKEN_TYPE_IDS_NAME = "token_type_ids"
        internal const val LOGITS_NAME = "logits"

        internal fun loadId2Label(id2LabelFile: File): Map<Int, String> {
            if (!id2LabelFile.isFile) {
                throw InferenceError.InvalidInput(
                    "id2label file does not exist: ${id2LabelFile.path}"
                )
            }

            val jsonObject = try {
                Json.parseToJsonElement(id2LabelFile.readText()).jsonObject
            } catch (error: Exception) {
                throw InferenceError.InvalidInput(
                    "id2label file must contain a JSON object"
                )
            }

            val labels = jsonObject.map { (key, value) ->
                val labelId = key.toIntOrNull()
                    ?: throw InferenceError.InvalidInput(
                        "id2label key '$key' must be an integer"
                    )
                labelId to value.jsonPrimitive.content
            }.toMap()

            if (labels.isEmpty()) {
                throw InferenceError.InvalidInput("id2label must not be empty")
            }
            return labels
        }

        private fun createRuntime(
            modelFile: File,
            intraOpThreadCount: Int,
        ): RuntimeComponents {
            if (!modelFile.isFile) {
                throw InferenceError.InvalidInput(
                    "model file does not exist: ${modelFile.path}"
                )
            }
            val environment = OrtEnvironment.getEnvironment()
            val session = createSessionOptions(intraOpThreadCount).use { options ->
                environment.createSession(modelFile.absolutePath, options)
            }
            return RuntimeComponents(
                environment,
                OnnxRuntimeTokenClassificationSession(environment, session),
            )
        }

        private fun createRuntime(
            modelBytes: ByteArray,
            intraOpThreadCount: Int,
        ): RuntimeComponents {
            if (modelBytes.isEmpty()) {
                throw InferenceError.InvalidInput("modelBytes must not be empty")
            }
            val environment = OrtEnvironment.getEnvironment()
            val session = createSessionOptions(intraOpThreadCount).use { options ->
                environment.createSession(modelBytes, options)
            }
            return RuntimeComponents(
                environment,
                OnnxRuntimeTokenClassificationSession(environment, session),
            )
        }

        private fun createSessionOptions(
            intraOpThreadCount: Int,
        ): OrtSession.SessionOptions {
            if (intraOpThreadCount < 1) {
                throw InferenceError.InvalidInput(
                    "intraOpThreadCount must be greater than zero"
                )
            }
            return OrtSession.SessionOptions().also { options ->
                options.setIntraOpNumThreads(intraOpThreadCount)
            }
        }
    }
}

internal interface TokenClassificationSession : Closeable {
    val inputNames: Set<String>

    fun run(inputs: Map<String, TokenInputTensor>): Map<String, Any?>
}

internal data class TokenInputTensor(
    val shape: LongArray,
    val elementType: TensorElementType,
    val values: IntArray,
)

private fun TokenInputTensor.toOnnxTensor(environment: OrtEnvironment): OnnxTensor {
    return when (elementType) {
        TensorElementType.INT64 -> OnnxTensor.createTensor(
            environment,
            arrayOf(LongArray(values.size) { index -> values[index].toLong() }),
        )
        TensorElementType.INT32 -> OnnxTensor.createTensor(
            environment,
            arrayOf(IntArray(values.size) { index -> values[index] }),
        )
    }
}

private data class RuntimeComponents(
    val environment: OrtEnvironment,
    val session: TokenClassificationSession,
)

private class OnnxRuntimeTokenClassificationSession(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
) : TokenClassificationSession {
    override val inputNames: Set<String>
        get() = session.inputNames

    override fun run(inputs: Map<String, TokenInputTensor>): Map<String, Any?> {
        val tensors = inputs.mapValues { (_, tensor) -> tensor.toOnnxTensor(environment) }
        try {
            session.run(tensors).use { result ->
                val logits = result.get(OnnxTokenClassifier.LOGITS_NAME)
                if (logits.isEmpty) {
                    return emptyMap()
                }
                return mapOf(OnnxTokenClassifier.LOGITS_NAME to logits.get().value)
            }
        } finally {
            tensors.values.forEach { tensor -> tensor.close() }
        }
    }

    override fun close() {
        session.close()
    }
}
