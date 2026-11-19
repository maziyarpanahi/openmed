package com.openmed.openmedkit.decision

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.openmed.openmedkit.onnx.OnnxRuntimeTokenClassificationSession
import com.openmed.openmedkit.onnx.TensorElementType
import com.openmed.openmedkit.onnx.TokenClassificationSession
import com.openmed.openmedkit.onnx.TokenInputTensor
import java.io.Closeable
import java.io.File

/**
 * Runs an ONNX export of a GLiClass-style uni-encoder with ONNX Runtime on CPU.
 *
 * The graph must take exactly `input_ids` and `attention_mask` (`[1, tokens]`)
 * and return a `[1, labels]` `logits` tensor, as the Python
 * `GLiClassOnnxRunner` requires. Tensors live in memory only; nothing is
 * written to disk or logged, and errors carry fixed codes without paths.
 */
public class OnnxDecisionEncoder internal constructor(
    private val environment: OrtEnvironment?,
    private val session: TokenClassificationSession,
    private val inputTensorType: TensorElementType = TensorElementType.INT64,
    private val ownsEnvironment: Boolean = false,
) : DecisionEncoder, Closeable {
    @Volatile
    private var closed = false

    init {
        if (session.inputNames != INPUT_NAMES) {
            session.close()
            throw DecisionRuntimeError.InvalidConfiguration("encoder_graph_invalid")
        }
    }

    /** Loads a local `.onnx` file; nothing is downloaded. */
    public constructor(
        modelFile: File,
        intraOpThreadCount: Int = 1,
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(createSession(intraOpThreadCount) { env, options ->
        if (!modelFile.isFile) throw DecisionRuntimeError.InvalidConfiguration("encoder_model_missing")
        env.createSession(modelFile.absolutePath, options)
    }, inputTensorType)

    /** Loads an in-memory ONNX graph. */
    public constructor(
        modelBytes: ByteArray,
        intraOpThreadCount: Int = 1,
        inputTensorType: TensorElementType = TensorElementType.INT64,
    ) : this(createSession(intraOpThreadCount) { env, options ->
        if (modelBytes.isEmpty()) throw DecisionRuntimeError.InvalidConfiguration("encoder_model_missing")
        env.createSession(modelBytes, options)
    }, inputTensorType)

    private constructor(runtime: Pair<OrtEnvironment, TokenClassificationSession>, type: TensorElementType) :
        this(runtime.first, runtime.second, type, ownsEnvironment = true)

    override fun logits(inputIds: IntArray, attentionMask: IntArray): DoubleArray {
        if (closed) throw DecisionEncoderException("encoder_input_invalid")
        if (inputIds.isEmpty() || inputIds.size != attentionMask.size) {
            throw DecisionEncoderException("encoder_input_invalid")
        }
        val shape = longArrayOf(1L, inputIds.size.toLong())
        val outputs = session.run(
            mapOf(
                INPUT_IDS to TokenInputTensor(shape, inputTensorType, inputIds.copyOf()),
                ATTENTION_MASK to TokenInputTensor(shape, inputTensorType, attentionMask.copyOf()),
            ),
        )
        val logits = outputs[LOGITS] ?: throw DecisionEncoderException("encoder_output_missing")
        val batch = logits as? Array<*>
        if (batch == null || batch.isEmpty()) throw DecisionEncoderException("encoder_output_invalid")
        return when (val row = batch[0]) {
            is FloatArray -> DoubleArray(row.size) { row[it].toDouble() }
            is DoubleArray -> row.copyOf()
            else -> throw DecisionEncoderException("encoder_output_invalid")
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            session.close()
        } finally {
            if (ownsEnvironment) environment?.close()
        }
    }

    internal companion object {
        const val INPUT_IDS = "input_ids"
        const val ATTENTION_MASK = "attention_mask"
        const val LOGITS = "logits"
        val INPUT_NAMES = setOf(INPUT_IDS, ATTENTION_MASK)

        private inline fun createSession(
            intraOpThreadCount: Int,
            open: (OrtEnvironment, OrtSession.SessionOptions) -> OrtSession,
        ): Pair<OrtEnvironment, TokenClassificationSession> {
            if (intraOpThreadCount < 1) {
                throw DecisionRuntimeError.InvalidConfiguration("encoder_threads_invalid")
            }
            val environment = OrtEnvironment.getEnvironment()
            val session = try {
                OrtSession.SessionOptions().use { options ->
                    options.setIntraOpNumThreads(intraOpThreadCount)
                    open(environment, options)
                }
            } catch (error: DecisionRuntimeError) {
                throw error
            } catch (error: Exception) {
                // ONNX Runtime messages can name local paths; keep the code only.
                throw DecisionRuntimeError.InvalidConfiguration("encoder_session_failed")
            }
            return environment to OnnxRuntimeTokenClassificationSession(environment, session)
        }
    }
}
